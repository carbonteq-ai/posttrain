from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from posttrain.common import ExecutionTarget
from posttrain.execution import (
    ExecutionJournal,
    ExecutionPolicy,
    ExecutionRequest,
    RuntimeImageRef,
    build_bundle,
    latest_runtime_image,
    wait_for_terminal,
)
from posttrain.tracking import RunSpec
from posttrain_execution_local import LocalDockerExecutionProvider

ROOT = Path(__file__).resolve().parents[5]
INFRA = Path("/home/hammad/projects/ai-infra")


def _report_transition(record) -> None:
    print(
        f"[posttrain] {record.handle.provider_id}: {record.state} on {record.target_id} ({record.native_state})",
        flush=True,
    )


def _runtime_image() -> str:
    return latest_runtime_image(INFRA / ".state/artifacts/posttrain-runtime").value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    run_spec = RunSpec(
        project_id="infrastructure-executions",
        work_package_id="qualify/local-runtime",
        stage="qualify",
        job_kind="model.transform",
        job_definition_version="qualification/local-runtime@1",
    )
    provider = LocalDockerExecutionProvider(
        state_root=(ROOT / ".posttrain/state/local-docker").resolve(),
    )
    with tempfile.TemporaryDirectory(prefix="posttrain-local-runtime-smoke-") as temporary:
        bundle = build_bundle(
            {"job.py": ROOT / "packages/execution/tests/fixtures/runtime_smoke_job.py"},
            (Path(temporary) / "bundle").resolve(),
        )
        request = ExecutionRequest(
            run_spec=run_spec,
            job_definition_id=run_spec.job_definition_version,
            bundle=bundle,
            image=RuntimeImageRef(_runtime_image()),
            target=ExecutionTarget(
                id="targets/local-gpu",
                revision="1",
                device_class="cuda",
                memory_gb=20,
            ),
            command=("python", "job.py"),
            idempotency_key=f"{run_spec.run_id}-attempt-1",
            policy=ExecutionPolicy(args.timeout_seconds),
        )
        handle = provider.submit(provider.plan(request))
        journal = ExecutionJournal((ROOT / ".posttrain/state/execution.jsonl").resolve())
        wait_for_terminal(
            provider,
            handle,
            timeout_seconds=args.timeout_seconds,
            journal=journal,
            poll_interval_seconds=1,
            on_transition=_report_transition,
        )
        result = provider.collect(handle)
        logs = provider.logs(handle, limit=200)
        output = {
            "bundle_digest": bundle.digest,
            "image": request.image.value,
            "provider_id": handle.provider_id,
            "state": result.record.state,
            "target": result.record.target_id,
            "logs": logs.lines,
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        if result.record.state != "succeeded":
            raise RuntimeError(f"local Docker runtime smoke failed: {result.record.message}")
        provider.cleanup(
            handle,
            run_id=run_spec.run_id,
            run_workspace=None,
            runtime_image=request.image,
        )


if __name__ == "__main__":
    main()
