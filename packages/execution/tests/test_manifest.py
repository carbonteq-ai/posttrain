from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.execution import (
    ExecutionJobManifest,
    ManifestMount,
    build_execution_bundle,
    plan_execution_bundle,
    resolved_inputs_digest,
    verify_bundle,
)


def _manifest() -> ExecutionJobManifest:
    return ExecutionJobManifest(
        run_id="run-manifest-1",
        project_id="example",
        work_package_id="train/example",
        job_id="train",
        job_definition_id="train/trl-sft@1",
        provider="dstack",
        project_manifest=".posttrain/project.toml",
        work_package=".posttrain/work_packages/train.yaml",
        runtime_image=f"registry.lan/posttrain@sha256:{'a' * 64}",
        resolved_inputs_digest="b" * 64,
        expected_artifact_roles=("model", "summary"),
        environment_names=("POSTTRAIN_TRACKIO_SERVER_URL", "TRACKIO_WRITE_TOKEN"),
        mounts=(
            ManifestMount("run-workspace", "/opt/posttrain/run"),
            ManifestMount("model-cache", "/root/.cache/huggingface"),
        ),
        retention={"recovery_checkpoints": 1},
    )


def test_execution_manifest_round_trips_without_secret_values(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    path.write_bytes(_manifest().to_bytes())

    loaded = ExecutionJobManifest.load(path)

    assert loaded == _manifest()
    assert "secret" not in path.read_text(encoding="utf-8").lower()


def test_resolved_input_digest_is_order_independent() -> None:
    assert resolved_inputs_digest({"a": 1, "b": {"c": True}}) == resolved_inputs_digest({"b": {"c": True}, "a": 1})


def test_execution_bundle_covers_job_manifest_bytes(tmp_path: Path) -> None:
    project = tmp_path / "project.toml"
    project.write_text('project_id = "example"\n', encoding="utf-8")
    bundle = build_execution_bundle(
        {".posttrain/project.toml": project},
        (tmp_path / "bundle").resolve(),
        _manifest(),
    )
    plan = plan_execution_bundle(
        {".posttrain/project.toml": project},
        _manifest(),
    )
    verify_bundle(bundle)

    assert plan.digest == bundle.digest
    assert plan.file_count == 2
    assert plan.size_bytes > project.stat().st_size

    job_manifest = bundle.path / ".posttrain" / "job.json"
    payload = json.loads(job_manifest.read_text(encoding="utf-8"))
    payload["job_id"] = "different"
    job_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="file manifest"):
        verify_bundle(bundle)


def test_execution_manifest_rejects_values_and_path_escape() -> None:
    with pytest.raises(ContractError, match="names, not values"):
        replace(
            _manifest(),
            environment_names=("TRACKIO_WRITE_TOKEN=secret",),
        )
    with pytest.raises(ContractError, match="bundle-relative"):
        replace(_manifest(), work_package="../outside.yaml")
