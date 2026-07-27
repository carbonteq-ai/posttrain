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


def _tracking_environment() -> None:
    secrets = yaml.safe_load((INFRA / ".state/secrets/vars.yml").read_text())
    os.environ["TRACKIO_SERVER_URL"] = "http://192.168.110.53:7860"
    os.environ["TRACKIO_WRITE_TOKEN"] = secrets["trackio_write_token"]
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    _tracking_environment()

    run_spec = RunSpec(
        project_id="ambient-agent",
        work_package_id="qualify/serve-benchmark-15",
        stage="qualify",
        job_kind="serve.benchmark",
        job_definition_version="qualification/serve-benchmark-15@1",
        resolved_inputs={
            "model": {
                "repo_id": "Qwen/Qwen3.5-0.8B",
                "revision": "2fc06364715b967f1860aea9cf38778875588b17",
            },
            "measured_requests": 15,
            "sampler_backend": "native",
        },
    )
    provider = DstackExecutionProvider.from_sdk_environment(
        project="main",
        python=INFRA / ".venv/bin/python",
        environment_file=INFRA / ".state/dstack/client.env",
    )
    image = latest_runtime_image(
        INFRA / ".state/artifacts/posttrain-serving-runtime"
    )
    with tempfile.TemporaryDirectory(prefix="ambient-serving-") as temporary:
        bundle = build_bundle(
            {"job.py": PROJECT / "src/ambient_agent/limited_serve.py"},
            (Path(temporary) / "bundle").resolve(),
        )
        request = ExecutionRequest(
            run_spec=run_spec,
            job_definition_id=run_spec.job_definition_version,
            bundle=bundle,
            image=RuntimeImageRef(image.value),
            target=ExecutionTarget(
                id="targets/dstack-rtx-pro-6000",
                revision="1",
                device_class="cuda",
                memory_gb=90,
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
            command=("python", "job.py", "--run-id", run_spec.run_id),
            idempotency_key=f"{run_spec.run_id}-attempt-1",
            policy=ExecutionPolicy(args.timeout_seconds),
            environment_names=(
                "HF_HUB_DISABLE_XET",
                "TRACKIO_SERVER_URL",
                "TRACKIO_WRITE_TOKEN",
                "VLLM_USE_FLASHINFER_SAMPLER",
            ),
        )
        plan = provider.plan(request)
        offers = plan.details.get("offers")
        if not isinstance(offers, int) or offers < 1:
            raise RuntimeError("dstack returned no serving-runtime offers")
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
        logs = provider.logs(handle, limit=300)
        print(
            json.dumps(
                {
                    "bundle_digest": bundle.digest,
                    "image": image.value,
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
            raise RuntimeError(f"bounded serving run failed: {result.record.message}")


if __name__ == "__main__":
    main()
