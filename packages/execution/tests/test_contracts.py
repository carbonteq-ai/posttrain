from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from posttrain.common import ContractError, ExecutionTarget, JsonValue
from posttrain.execution import (
    DEFAULT_SHARED_MEMORY_GB,
    JOB_PACKAGE_WORKER_COMMAND,
    BundleRef,
    ExecutionHandle,
    ExecutionJournal,
    ExecutionPolicy,
    ExecutionRecord,
    ExecutionRequest,
    RuntimeImageRef,
    execution_shared_memory_gb,
)
from posttrain.tracking import RunSpec


def _run_spec() -> RunSpec:
    return RunSpec(
        project_id="tests",
        work_package_id="train/test",
        stage="train",
        job_kind="train.sft",
        job_definition_version="train/sft@1",
    )


def test_bundle_reference_can_be_planned_before_materialization(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "future-bundle").resolve()

    reference = BundleRef(path, hashlib.sha256(b"bundle").hexdigest())

    assert reference.path == path
    assert not path.exists()


def test_request_carries_only_environment_names() -> None:
    request = ExecutionRequest(
        run_spec=_run_spec(),
        job_definition_id="train/sft@1",
        image=RuntimeImageRef(f"registry.lan/posttrain@sha256:{'a' * 64}"),
        target=ExecutionTarget("targets/gpu", "1", "cuda", 24),
        command=JOB_PACKAGE_WORKER_COMMAND,
        idempotency_key="logical-run-attempt-1",
        policy=ExecutionPolicy(300),
        environment_names=("TRACKIO_URL", "TRACKIO_WRITE_TOKEN"),
        local_image="posttrain-local:request-contract",
    )
    assert request.environment_names == ("TRACKIO_URL", "TRACKIO_WRITE_TOKEN")
    launch = json.loads(request.launch_environment(provider="local-docker")["POSTTRAIN_EXECUTION"])
    assert launch["run"]["run_id"] == request.run_spec.run_id
    assert launch["provider"] == "local-docker"
    assert launch["attempt"] == 1
    assert launch["job_image"] == request.image.value
    assert "local_image" not in launch
    assert launch["target"]["id"] == request.target.id

    with pytest.raises(ContractError, match="names, not secret values"):
        replace(request, environment_names=("TRACKIO_WRITE_TOKEN=secret",))

    with pytest.raises(ContractError, match="attempt must be positive"):
        replace(request, attempt=0)

    with pytest.raises(ContractError, match="stable packaged worker entrypoint"):
        replace(request, command=("python", "payload.py"))


def test_execution_journal_is_append_only_and_mode_600(tmp_path: Path) -> None:
    path = (tmp_path / "state" / "execution.jsonl").resolve()
    journal = ExecutionJournal(path)
    record = ExecutionRecord(
        ExecutionHandle("local", "job-1", "key-1"),
        "queued",
        1,
        "targets/gpu",
        datetime.now(UTC),
        "pending",
    )
    journal.append(record)
    journal.append(record)
    assert len(path.read_text().splitlines()) == 2
    assert path.stat().st_mode & 0o777 == 0o600


def test_every_job_container_gets_an_explicit_shared_memory_size() -> None:
    def target(**placement: JsonValue) -> ExecutionTarget:
        return ExecutionTarget("targets/gpu", "1", "cuda", 24, placement=placement)

    # Docker and dstack default /dev/shm to 64 MiB; veRL's rollout server alone
    # needs 160 MiB at start.
    assert DEFAULT_SHARED_MEMORY_GB == 16
    assert execution_shared_memory_gb(target()) == 16
    assert execution_shared_memory_gb(target(shm_size_gb=48)) == 48
    # A declared host bounds the default to half of its memory.
    assert execution_shared_memory_gb(target(host_memory_gb=64)) == 16
    assert execution_shared_memory_gb(target(host_memory_gb=12)) == 6
    assert execution_shared_memory_gb(target(host_memory_gb=1.5)) == 1
    assert execution_shared_memory_gb(target(host_memory_gb=12, shm_size_gb=10)) == 10

    with pytest.raises(ContractError, match="exceeds its host_memory_gb 12"):
        execution_shared_memory_gb(target(host_memory_gb=12, shm_size_gb=13))
    for invalid in (0, -1, 1.5, "16", True):
        with pytest.raises(ContractError, match="shm_size_gb must be a positive integer"):
            execution_shared_memory_gb(target(shm_size_gb=invalid))
    for invalid in (0, -4, "64", False):
        with pytest.raises(ContractError, match="host_memory_gb must be a positive number"):
            execution_shared_memory_gb(target(host_memory_gb=invalid))

    request = ExecutionRequest(
        run_spec=_run_spec(),
        job_definition_id="train/sft@1",
        image=RuntimeImageRef(f"registry.lan/posttrain@sha256:{'a' * 64}"),
        target=target(shm_size_gb=24),
        command=JOB_PACKAGE_WORKER_COMMAND,
        idempotency_key="logical-run-attempt-1",
        policy=ExecutionPolicy(300),
    )
    assert request.shared_memory_gb == 24
    # A contradictory target cannot even become a request.
    with pytest.raises(ContractError, match="exceeds its host_memory_gb"):
        replace(request, target=target(host_memory_gb=8, shm_size_gb=16))
