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
from posttrain_execution_dstack import DstackExecutionProvider

ROOT = Path(__file__).resolve().parents[2]
INFRA = Path("/home/hammad/projects/ai-infra")


def _report_transition(record) -> None:
    print(
        f"[posttrain] {record.handle.provider_id}: {record.state} on {record.target_id} ({record.native_state})",
        flush=True,
    )


def _runtime_image(profile: str) -> str:
    state_name = "posttrain-serving-runtime" if profile == "serve" else "posttrain-runtime"
    return latest_runtime_image(INFRA / f".state/artifacts/{state_name}").value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-gb", type=int, default=90)
    parser.add_argument("--profile", choices=("train", "serve"), default="train")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    run_spec = RunSpec(
        project_id="infrastructure-executions",
        work_package_id=f"qualify/dstack-{args.profile}-runtime",
        stage="qualify",
        job_kind="model.transform",
        job_definition_version=f"qualification/dstack-{args.profile}-runtime@1",
    )
    provider = DstackExecutionProvider.from_sdk_environment(
        project="main",
        python=INFRA / ".venv/bin/python",
        environment_file=INFRA / ".state/dstack/client.env",
    )
    with tempfile.TemporaryDirectory(prefix="posttrain-runtime-smoke-") as temporary:
        bundle = build_bundle(
            {"job.py": ROOT / "packages/execution/tests/fixtures/runtime_smoke_job.py"},
            (Path(temporary) / "bundle").resolve(),
        )
        request = ExecutionRequest(
            run_spec=run_spec,
            job_definition_id=run_spec.job_definition_version,
            bundle=bundle,
            image=RuntimeImageRef(_runtime_image(args.profile)),
            target=ExecutionTarget(
                id="targets/dstack-gpu",
                revision="1",
                device_class="cuda",
                memory_gb=args.memory_gb,
                placement={
                    "fleets": ["local-gpu-workers"],
                    "gpu_count": 1,
                    "disk_gb": 100,
                },
            ),
            command=("python", "job.py", "--profile", args.profile),
            idempotency_key=f"{run_spec.run_id}-attempt-1",
            policy=ExecutionPolicy(args.timeout_seconds),
        )
        plan = provider.plan(request)
        offers = plan.details.get("offers")
        if not isinstance(offers, int) or offers < 1:
            raise RuntimeError("dstack returned no runtime offers")
        handle = provider.submit(plan)
        journal = ExecutionJournal((ROOT / ".posttrain/state/execution.jsonl").resolve())
        wait_for_terminal(
            provider,
            handle,
            timeout_seconds=args.timeout_seconds,
            journal=journal,
            poll_interval_seconds=2,
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
            raise RuntimeError(f"dstack runtime smoke failed: {result.record.message}")


if __name__ == "__main__":
    main()
