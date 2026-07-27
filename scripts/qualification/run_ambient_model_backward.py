from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml
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
PROJECT = Path("/home/hammad/projects/ambient-agent")


def _report_transition(record) -> None:
    print(
        f"[posttrain] {record.handle.provider_id}: "
        f"{record.state} on {record.target_id} ({record.native_state})",
        flush=True,
    )


def _runtime_image() -> str:
    return latest_runtime_image(
        INFRA / ".state/artifacts/posttrain-runtime"
    ).value


def _tracking_environment() -> None:
    secrets = yaml.safe_load((INFRA / ".state/secrets/vars.yml").read_text())
    os.environ["TRACKIO_SERVER_URL"] = "http://192.168.110.53:7860"
    os.environ["TRACKIO_WRITE_TOKEN"] = secrets["trackio_write_token"]
    os.environ["HF_HUB_DISABLE_XET"] = "1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=("backward", "sft", "reinforce"),
        default="backward",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--memory-gb", type=int, default=90)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    _tracking_environment()
    job_kind = "train.grpo" if args.algorithm == "reinforce" else "train.sft"
    work_package_id = {
        "backward": "qualify/model-backward",
        "reinforce": "qualify/reinforce-15",
        "sft": "qualify/sft-15",
    }[args.algorithm]
    definition_id = {
        "backward": "qualification/model-backward@1",
        "reinforce": "qualification/reinforce-15@1",
        "sft": "qualification/sft-15@1",
    }[args.algorithm]
    run_spec = RunSpec(
        project_id="ambient-agent",
        work_package_id=work_package_id,
        stage="qualify",
        job_kind=job_kind,
        job_definition_version=definition_id,
        resolved_inputs={
            "algorithm": args.algorithm,
            "model": {"repo_id": args.model, "revision": args.revision},
            "backward_passes": args.steps,
        },
    )
    provider = DstackExecutionProvider.from_sdk_environment(
        project="main",
        python=INFRA / ".venv/bin/python",
        environment_file=INFRA / ".state/dstack/client.env",
    )
    with tempfile.TemporaryDirectory(prefix="ambient-model-backward-") as temporary:
        bundle = build_bundle(
            {"src/ambient_agent": PROJECT / "src/ambient_agent"},
            (Path(temporary) / "bundle").resolve(),
        )
        request = ExecutionRequest(
            run_spec=run_spec,
            job_definition_id=run_spec.job_definition_version,
            bundle=bundle,
            image=RuntimeImageRef(_runtime_image()),
            target=ExecutionTarget(
                id="targets/dstack-gpu",
                revision="1",
                device_class="cuda",
                memory_gb=args.memory_gb,
                placement={
                    "fleets": ["local-gpu-workers"],
                    "gpu_count": 1,
                    "disk_gb": 100,
                    "cache_mounts": [
                        {
                            "instance_path": "/var/lib/dstack-cache/huggingface",
                            "path": "/root/.cache/huggingface",
                        }
                    ],
                },
            ),
            command=(
                "python",
                {
                    "backward": "src/ambient_agent/limited_run.py",
                    "reinforce": "src/ambient_agent/limited_reinforce.py",
                    "sft": "src/ambient_agent/limited_sft.py",
                }[args.algorithm],
                "--run-id",
                run_spec.run_id,
                "--model",
                args.model,
                "--revision",
                args.revision,
                "--steps",
                str(args.steps),
            ),
            idempotency_key=f"{run_spec.run_id}-attempt-1",
            policy=ExecutionPolicy(args.timeout_seconds),
            environment_names=(
                "HF_HUB_DISABLE_XET",
                "TRACKIO_SERVER_URL",
                "TRACKIO_WRITE_TOKEN",
            ),
        )
        plan = provider.plan(request)
        offers = plan.details.get("offers")
        if not isinstance(offers, int) or offers < 1:
            raise RuntimeError("dstack returned no model-backward offers")
        handle = provider.submit(plan)
        journal = ExecutionJournal((ROOT / ".posttrain/state/execution.jsonl").resolve())
        wait_for_terminal(
            provider,
            handle,
            timeout_seconds=args.timeout_seconds,
            journal=journal,
            on_transition=_report_transition,
        )
        result = provider.collect(handle)
        logs = provider.logs(handle, limit=200)
        print(
            json.dumps(
                {
                    "bundle_digest": bundle.digest,
                    "algorithm": args.algorithm,
                    "model": args.model,
                    "provider_id": handle.provider_id,
                    "run_id": run_spec.run_id,
                    "state": result.record.state,
                    "target": result.record.target_id,
                    "logs": logs.lines,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if result.record.state != "succeeded":
            raise RuntimeError(f"bounded model-backward run failed: {result.record.message}")


if __name__ == "__main__":
    main()
