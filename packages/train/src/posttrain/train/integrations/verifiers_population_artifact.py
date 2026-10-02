"""Freeze existing native replay records before a resolved optimizer update."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from posttrain.common import LocalArtifactRef, ProducedArtifact

from ..update_records import InvalidPolicyUpdate


def decode_native_population(
    evidence: bytes, *, format: Literal["verifiers-native-episodes", "verifiers-native-traces"],
) -> Mapping[str, Any]:
    """Use native schema models to restore original graphs, never chat rows.

    Reject missing identities before native models can supply random defaults.
    Duplicate identities also reject, including unselected episode siblings.
    This decoder consumes the uncompressed JSONL produced by retain_population.
    """
    if format not in {"verifiers-native-episodes", "verifiers-native-traces"}:
        raise InvalidPolicyUpdate("unsupported native population artifact format")
    from verifiers.v1.episode import Episode  # pyright: ignore[reportMissingImports]
    from verifiers.v1.trace import Trace  # pyright: ignore[reportMissingImports]

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
        restored = (Episode.model_validate(record).traces if format == "verifiers-native-episodes"
                    else [Trace.model_validate(record)])
        for trace in restored:
            traces[trace.id] = trace
    return traces


def retain_native_population(
    source: Path, destination: Path, trace_ids: tuple[str, ...], *, episodes: bool,
) -> ProducedArtifact:
    """Copy complete native envelopes verbatim, with deterministic membership.

    The caller serializes this read with native writers. Episode siblings remain
    in their original envelope; they are evidence, not additional loss samples.
    Atomic content-addressed publication never modifies the append-only source.
    """
    if not trace_ids or any(not value for value in trace_ids) or len(set(trace_ids)) != len(trace_ids):
        raise InvalidPolicyUpdate("native population requires unique nonempty trace identities")
    wanted = set(trace_ids)
    found: set[str] = set()
    selected: list[bytes] = []
    with source.open("rb") as stream:
        for line in stream:
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
        name=f"training/rollouts/populations/{digest}", kind="evaluation-traces",
        reference=LocalArtifactRef(target.resolve(), digest),
        metadata={"format": "verifiers-native-episodes" if episodes else "verifiers-native-traces",
                  "replay_authority": True, "trace_ids": list(trace_ids),
                  "native_record_count": len(selected)},
    )
