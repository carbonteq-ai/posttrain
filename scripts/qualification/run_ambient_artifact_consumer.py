from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml
from posttrain.common import ExecutionTarget, StoredArtifactRef
from posttrain.execution import (
    ExecutionJournal,
    ExecutionPolicy,
    ExecutionRequest,
    RuntimeImageRef,
    build_bundle,
    latest_runtime_image,
    wait_for_terminal,
)
from posttrain.tracking import ArtifactInput, RunSpec
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
    parser.add_argument("--producer-run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    _tracking_environment()

    artifact_input = ArtifactInput(
        StoredArtifactRef(
            provider="trackio",
            namespace="ambient-agent",
            name=args.artifact_name,
            version=args.artifact_version,
            digest=args.artifact_digest,
            provider_metadata={"producer_run_id": args.producer_run_id},
        ),
        "model",
    )
    run_spec = RunSpec(
        project_id="ambient-agent",
        work_package_id="qualify/cross-worker-artifact",
        stage="qualify",
        job_kind="eval.model",
        job_definition_version="qualification/cross-worker-artifact@1",
        resolved_inputs={
            "model": {"repo_id": args.model, "revision": args.revision},
            "producer_run_id": args.producer_run_id,
        },
        artifacts={"model_adapter": artifact_input},
    )
    provider = DstackExecutionProvider.from_sdk_environment(
        project="main",
        python=INFRA / ".venv/bin/python",
        environment_file=INFRA / ".state/dstack/client.env",
    )
    with tempfile.TemporaryDirectory(prefix="ambient-artifact-consumer-") as temporary:
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
                id="targets/dstack-rtx4090",
                revision="1",
                device_class="cuda",
                memory_gb=20,
                placement={
                    "fleets": ["local-gpu-workers"],
                    "gpu_count": 1,
                    "gpu_memory_max_gb": 30,
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
                "src/ambient_agent/consume_adapter.py",
                "--run-id",
                run_spec.run_id,
                "--producer-run-id",
                args.producer_run_id,
                "--model",
                args.model,
                "--revision",
                args.revision,
                "--artifact-name",
                args.artifact_name,
                "--artifact-version",
                args.artifact_version,
                "--artifact-digest",
                args.artifact_digest,
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
            raise RuntimeError("dstack returned no RTX 4090 artifact-consumer offers")
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
            raise RuntimeError(
                f"cross-worker artifact run failed: {result.record.message}"
            )


if __name__ == "__main__":
    main()
