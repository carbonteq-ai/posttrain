from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from posttrain.common import ContractError, ExecutionTarget
from posttrain.execution import (
    JOB_PACKAGE_WORKER_COMMAND,
    BundleRef,
    ExecutionHandle,
    ExecutionJournal,
    ExecutionPolicy,
    ExecutionRecord,
    ExecutionRequest,
    RuntimeImageRef,
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
    )
    assert request.environment_names == ("TRACKIO_URL", "TRACKIO_WRITE_TOKEN")
    launch = json.loads(
        request.launch_environment(provider="local-docker")["POSTTRAIN_EXECUTION"]
    )
    assert launch["run"]["run_id"] == request.run_spec.run_id
    assert launch["provider"] == "local-docker"
    assert launch["attempt"] == 1
    assert launch["job_image"] == request.image.value
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
