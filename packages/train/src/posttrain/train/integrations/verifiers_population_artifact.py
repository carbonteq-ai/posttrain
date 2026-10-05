"""Freeze existing native replay records before a resolved optimizer update."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

from posttrain.common import LocalArtifactRef, ProducedArtifact

from ..update_records import InvalidPolicyUpdate


@dataclass(frozen=True, slots=True)
class ConditioningTrace:
    """The message graph of one native trace: what policy scoring reads.

    Nodes and calls are the native Verifiers models; assessments, tool events and
    other trace content are not restored.
    """

    id: str
    nodes: tuple[Any, ...]
    calls: tuple[Any, ...]


def decode_native_population(
    evidence: bytes,
    *,
    format: Literal["verifiers-native-episodes", "verifiers-native-traces"],
    content: Literal["traces", "conditioning"] = "traces",
) -> Mapping[str, Any]:
    """Use native schema models to restore original graphs, never chat rows.

    Reject missing identities before native models can supply random defaults.
    Duplicate identities also reject, including unselected episode siblings.
    This decoder consumes the uncompressed JSONL produced by retain_population.

    ``content="conditioning"`` restores only each trace's message graph (nodes
    and model calls) through the native ``Branch`` schema. Policy scoring reads
    nothing else, and the assessment archives that dominate an episode's bytes
    are not validated again.
    """
    if format not in {"verifiers-native-episodes", "verifiers-native-traces"}:
        raise InvalidPolicyUpdate("unsupported native population artifact format")
    if content not in {"traces", "conditioning"}:
        raise InvalidPolicyUpdate("unsupported native population decode content")
    from verifiers.v1.episode import Episode  # pyright: ignore[reportMissingImports]
    from verifiers.v1.trace import Trace  # pyright: ignore[reportMissingImports]

    try:
        from verifiers.v1._validation_scope import (  # pyright: ignore[reportMissingImports]
            validation_scope,
        )
    except ImportError:  # older pinned runtimes: validate without proof reuse
        validation_scope = contextlib.nullcontext

    if not evidence or not evidence.endswith(b"\n"):
        raise InvalidPolicyUpdate("native population artifact requires complete JSONL records")
    traces: dict[str, Any] = {}
    for line in evidence.splitlines():
        record = json.loads(line)
        if not isinstance(record, dict):
            raise InvalidPolicyUpdate("native population records must be objects")
        raw_traces = record.get("traces") if format == "verifiers-native-episodes" else [record]
        if not isinstance(raw_traces, list) or not raw_traces:
            raise InvalidPolicyUpdate("native population record lacks original traces")
        for raw in raw_traces:
            if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw["id"]:
                raise InvalidPolicyUpdate("native population trace requires its original identity")
            if raw["id"] in traces:
                raise InvalidPolicyUpdate("native population has duplicate trace identities")
            # Reserve all IDs before validation, including duplicates in one envelope.
            traces[raw["id"]] = None
        if content == "conditioning":
            from verifiers.v1.trace import Branch  # pyright: ignore[reportMissingImports]

            for raw in raw_traces:
                branch = Branch.model_validate({"index": 0, "nodes": raw.get("nodes"), "calls": raw.get("calls", [])})
                traces[raw["id"]] = ConditioningTrace(raw["id"], tuple(branch.nodes), tuple(branch.calls))
            continue
        # Episodes repeat the same assessment sources across hundreds of batches.
        # A per-record validation scope validates each source once and reuses its
        # exact-match proof for the repeats (as the Verifiers env server and
        # client do); the proofs end with the record, so memory stays bounded.
        with validation_scope():
            restored = (
                Episode.model_validate(record).traces
                if format == "verifiers-native-episodes"
                else [Trace.model_validate(record)]
            )
        for trace in restored:
            traces[trace.id] = trace
    return traces


def _select(lines: Iterable[bytes], wanted: set[str], episodes: bool) -> list[bytes]:
    found: set[str] = set()
    selected: list[bytes] = []
    for line in lines:
        record = json.loads(line)
        traces = record.get("traces", []) if episodes else [record]
        ids = [trace.get("id") for trace in traces]
        matched = wanted.intersection(ids)
        if not matched:
            continue
        if found.intersection(matched) or any(ids.count(identity) != 1 for identity in matched):
            raise InvalidPolicyUpdate("native population has duplicate retained trace identities")
        found.update(matched)
        if not line.endswith(b"\n"):
            raise InvalidPolicyUpdate("native population source contains an incomplete record")
        selected.append(line)
    if found != wanted:
        raise InvalidPolicyUpdate("native population source lacks admitted trace identities")
    return selected


def _read_spans(stream: BinaryIO, spans: Sequence[tuple[int, int]]) -> Iterator[bytes]:
    for offset, length in spans:
        stream.seek(offset)
        line = stream.read(length)
        if len(line) != length:
            raise InvalidPolicyUpdate("native population source record span is truncated")
        yield line


def retain_native_population(
    source: Path,
    destination: Path,
    trace_ids: tuple[str, ...],
    *,
    episodes: bool,
    spans: Mapping[str, tuple[int, int]] | None = None,
) -> ProducedArtifact:
    """Copy complete native envelopes verbatim, with deterministic membership.

    The caller serializes this read with native writers. Episode siblings remain
    in their original envelope; they are evidence, not additional loss samples.
    Atomic content-addressed publication never modifies the append-only source.
    """
    if not trace_ids or any(not value for value in trace_ids) or len(set(trace_ids)) != len(trace_ids):
        raise InvalidPolicyUpdate("native population requires unique nonempty trace identities")
    wanted = set(trace_ids)
    selected: list[bytes] | None = None
    if spans is not None and wanted <= set(spans):
        # Read only the population's own records (in file order) rather than
        # parsing every record the run has appended so far. Any inconsistency
        # (a stale index after a restart, a foreign writer) falls back to the
        # authoritative full scan instead of failing the update.
        try:
            with source.open("rb") as stream:
                selected = _select(
                    _read_spans(stream, sorted({spans[identity] for identity in wanted})), wanted, episodes
                )
        except (InvalidPolicyUpdate, ValueError):
            selected = None
    if selected is None:
        with source.open("rb") as stream:
            selected = _select(stream, wanted, episodes)
    evidence = b"".join(selected)
    digest = hashlib.sha256(evidence).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{digest}.jsonl"
    descriptor, temporary = tempfile.mkstemp(prefix=".population-", dir=destination)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(evidence)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            if target.read_bytes() != evidence:
                raise InvalidPolicyUpdate("retained population content-addressed artifact was modified") from error
        directory = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)
    return ProducedArtifact(
        name=f"training/rollouts/populations/{digest}",
        kind="evaluation-traces",
        reference=LocalArtifactRef(target.resolve(), digest),
        metadata={
            "format": "verifiers-native-episodes" if episodes else "verifiers-native-traces",
            "replay_authority": True,
            "trace_ids": list(trace_ids),
            "native_record_count": len(selected),
        },
    )
