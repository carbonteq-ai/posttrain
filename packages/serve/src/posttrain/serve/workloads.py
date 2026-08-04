"""Materialize and verify record populations owned by serving workloads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from posttrain.common import ContractError, Workload


@dataclass(frozen=True, slots=True)
class WorkloadMaterialization:
    """A reproducible serving-workload population and its materialized paths."""

    workload_id: str
    workload_revision: str
    corpus_id: str
    corpus_revision: str
    record_count: int
    content_sha256: str
    path: str
    manifest: str
    materialized: bool

    def to_payload(self) -> dict[str, object]:
        """Return a stable CLI and receipt representation."""

        return {
            "workload_id": self.workload_id,
            "workload_revision": self.workload_revision,
            "corpus_id": self.corpus_id,
            "corpus_revision": self.corpus_revision,
            "record_count": self.record_count,
            "content_sha256": self.content_sha256,
            "path": self.path,
            "manifest": self.manifest,
            "materialized": self.materialized,
        }


def materialize_workload(workload: Workload, *, output: Path) -> WorkloadMaterialization:
    """Build a workload's pinned population and write canonical bytes."""

    records_text, manifest_text, corpus_id, corpus_revision, digest = _build(workload)
    records_path, manifest_path = _output_paths(output.resolve(), corpus_id)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(records_text, encoding="utf-8")
    manifest_path.write_text(manifest_text, encoding="utf-8")
    return _result(
        workload,
        corpus_id=corpus_id,
        corpus_revision=corpus_revision,
        records_text=records_text,
        digest=digest,
        records_path=str(records_path),
        manifest_path=str(manifest_path),
        materialized=True,
    )


def verify_workload(workload: Workload) -> WorkloadMaterialization:
    """Rebuild a workload's population and compare it with packaged bytes."""

    records_text, manifest_text, corpus_id, corpus_revision, digest = _build(workload)
    resource_root = files("posttrain.serve.benchmarks.general_serving.resources")
    records_resource = resource_root.joinpath(f"{corpus_id}.jsonl")
    manifest_resource = resource_root.joinpath(f"{corpus_id}.manifest.json")
    if records_resource.read_text(encoding="utf-8") != records_text:
        raise ContractError(f"packaged workload corpus {corpus_id!r} differs from rebuilt content")
    if manifest_resource.read_text(encoding="utf-8") != manifest_text:
        raise ContractError(f"packaged workload corpus {corpus_id!r} manifest differs from rebuilt content")
    return _result(
        workload,
        corpus_id=corpus_id,
        corpus_revision=corpus_revision,
        records_text=records_text,
        digest=digest,
        records_path=str(records_resource),
        manifest_path=str(manifest_resource),
        materialized=False,
    )


def _build(workload: Workload) -> tuple[str, str, str, str, str]:
    corpus = workload.requests.get("corpus")
    if not isinstance(corpus, Mapping):
        raise ContractError(f"workload {workload.id!r} does not declare a materializable corpus")
    corpus_id = corpus.get("id")
    corpus_revision = corpus.get("revision")
    if not isinstance(corpus_id, str) or not isinstance(corpus_revision, str):
        raise ContractError(f"workload {workload.id!r} corpus requires string id and revision")
    if corpus_id != "general-serving-v1" or corpus_revision != "1":
        raise ContractError(
            f"no serving workload materializer is registered for corpus {corpus_id!r}@{corpus_revision!r}"
        )

    # Definition imports are inert; explicit materialization is the only path
    # that imports the network-backed builder.
    from posttrain.serve.benchmarks.general_serving.build import build
    from posttrain.serve.benchmarks.general_serving.definition import GENERAL_SERVING_V1

    records_text, manifest_text = build()
    digest = hashlib.sha256(records_text.encode("utf-8")).hexdigest()
    if digest != GENERAL_SERVING_V1.expected_content_sha256:
        raise ContractError(
            f"workload corpus {corpus_id!r} digest mismatch: expected "
            f"{GENERAL_SERVING_V1.expected_content_sha256}, got {digest}"
        )
    selected_digest = corpus.get("digest")
    if selected_digest is not None and selected_digest != digest:
        raise ContractError(
            f"workload corpus selection digest does not match rebuilt content: {selected_digest} != {digest}"
        )
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        raise ContractError("workload corpus builder returned an invalid manifest") from error
    if not isinstance(manifest, dict) or manifest.get("digest") != digest:
        raise ContractError("workload corpus builder returned an invalid manifest")
    return records_text, manifest_text, corpus_id, corpus_revision, digest


def _result(
    workload: Workload,
    *,
    corpus_id: str,
    corpus_revision: str,
    records_text: str,
    digest: str,
    records_path: str,
    manifest_path: str,
    materialized: bool,
) -> WorkloadMaterialization:
    return WorkloadMaterialization(
        workload_id=workload.id,
        workload_revision=workload.revision,
        corpus_id=corpus_id,
        corpus_revision=corpus_revision,
        record_count=len(records_text.splitlines()),
        content_sha256=digest,
        path=records_path,
        manifest=manifest_path,
        materialized=materialized,
    )


def _output_paths(output: Path, corpus_id: str) -> tuple[Path, Path]:
    if output.suffix == ".jsonl":
        return output, output.with_name(f"{corpus_id}.manifest.json")
    return output / f"{corpus_id}.jsonl", output / f"{corpus_id}.manifest.json"


__all__ = ["WorkloadMaterialization", "materialize_workload", "verify_workload"]
