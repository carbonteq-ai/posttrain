from __future__ import annotations

import json
from dataclasses import replace

import pytest
from posttrain.common import ContractError
from posttrain.execution import JobPackageManifest, RuntimeImageRef
from posttrain.execution_pack import PackageMaterializationRecord, PackageMaterializationStore


def _manifest() -> JobPackageManifest:
    digest = "a" * 64
    return JobPackageManifest(
        project_id="project",
        work_package_id="work",
        job_id="job",
        job_definition_id="definition",
        job_kind="train.sft",
        resolved_inputs_digest=digest,
        framework_source_digest=digest,
        project_source_digest=digest,
        runtime_dependencies_digest=digest,
        code_requirements_digest=digest,
        resolved_config_digest=digest,
        project_config_digest=digest,
        universal_image=RuntimeImageRef(f"registry.example/base@sha256:{'b' * 64}"),
        kind_image=RuntimeImageRef(f"registry.example/kind@sha256:{'c' * 64}"),
        runtime_variant="supervised",
    )


def test_materialization_record_round_trips_manifest_and_identity() -> None:
    manifest = _manifest()
    record = PackageMaterializationRecord(
        package_key=manifest.package_key,
        context_digest="d" * 64,
        publication_key="e" * 64,
        manifest=manifest,
    )

    restored = PackageMaterializationRecord.from_bytes(record.to_bytes())

    assert restored == record
    assert restored.manifest_digest


def test_materialization_record_rejects_tampered_manifest_digest() -> None:
    manifest = _manifest()
    record = PackageMaterializationRecord(
        package_key=manifest.package_key,
        context_digest="d" * 64,
        publication_key="e" * 64,
        manifest=manifest,
    )
    payload = json.loads(record.to_bytes())
    payload["manifest_digest"] = "f" * 64

    with pytest.raises(ContractError, match="manifest digest"):
        PackageMaterializationRecord.from_payload(payload)


def test_materialization_store_resolves_by_plan_and_publication(tmp_path) -> None:
    manifest = _manifest()
    record = PackageMaterializationRecord(
        package_key=manifest.package_key,
        context_digest="d" * 64,
        publication_key="e" * 64,
        manifest=manifest,
        plan_key="f" * 64,
    )
    store = PackageMaterializationStore(tmp_path.resolve())

    path = store.commit(record)

    assert path.is_file()
    assert path.stat().st_mode & 0o077 == 0
    assert path.name == f"{'e' * 64}.json"
    assert store.resolve("f" * 64, publication_key="e" * 64) == record
    store.commit(record)
    with pytest.raises(ContractError, match="conflicts"):
        store.commit(replace(record, context_digest="0" * 64))


def test_materialization_store_keeps_distinct_registry_publications(tmp_path) -> None:
    manifest = _manifest()
    first = PackageMaterializationRecord(
        package_key=manifest.package_key,
        context_digest="d" * 64,
        publication_key="e" * 64,
        manifest=manifest,
        plan_key="f" * 64,
    )
    second = replace(first, publication_key="0" * 64)
    store = PackageMaterializationStore(tmp_path.resolve())

    store.commit(first)
    store.commit(second)

    assert store.resolve("f" * 64, publication_key="e" * 64) == first
    assert store.resolve("f" * 64, publication_key="0" * 64) == second
    assert store.resolve_all("f" * 64) == (second, first)
    with pytest.raises(ContractError, match="ambiguous"):
        store.resolve("f" * 64)
