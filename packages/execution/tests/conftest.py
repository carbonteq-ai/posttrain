from __future__ import annotations

import pytest
from posttrain.common import ExecutionTarget
from posttrain.execution import (
    JOB_PACKAGE_WORKER_COMMAND,
    ExecutionPolicy,
    ExecutionRequest,
    RuntimeImageRef,
)
from posttrain.tracking import RunSpec


@pytest.fixture
def request_factory():
    def build(name: str) -> ExecutionRequest:
        return ExecutionRequest(
            run_spec=RunSpec(
                project_id="tests",
                work_package_id=f"train/{name}",
                stage="train",
                job_kind="train.sft",
                job_definition_version="train/sft@1",
            ),
            job_definition_id="train/sft@1",
            image=RuntimeImageRef(f"registry.lan/posttrain@sha256:{'a' * 64}"),
            target=ExecutionTarget("targets/gpu", "1", "cuda", 24),
            command=JOB_PACKAGE_WORKER_COMMAND,
            idempotency_key=name,
            policy=ExecutionPolicy(300),
        )

    return build
