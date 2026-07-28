"""Run a named substantial qualification scenario without editing Python."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from posttrain.common import ExecutionTarget  # noqa: E402
from posttrain.execution import (  # noqa: E402
    ExecutionJournal,
    ExecutionMount,
    ExecutionPolicy,
    ExecutionRequest,
    RuntimeImageRef,
    build_bundle,
    wait_for_terminal,
)
from posttrain.tracking import RunSpec  # noqa: E402
from posttrain_execution_dstack import DstackExecutionProvider  # noqa: E402

from scripts.qualification.algorithm_scenarios import (  # noqa: E402
    QualificationScenario,
    scenario_by_id,
)
from scripts.qualification.validate_algorithm_run import (  # noqa: E402
    RemoteAlgorithmEvidence,
    collect_remote_evidence,
)

DEFAULT_GPU_PYTHON = Path("/home/hammad/projects/verl/.venv313/bin/python")
REMOTE_WORKSPACE_ROOT = Path("/opt/posttrain/run")
_SCENARIO_WORK_PACKAGES: dict[str, tuple[str, str]] = {
    "automationbench-qwen35-08b-grpo-10": (
        ".posttrain/work_packages/automationbench_zapier_grpo.yaml",
        "grpo",
    ),
    "gsm8k-qwen35-08b-grpo-15": (
        ".posttrain/work_packages/gsm8k_qwen08b_grpo_qualification.yaml",
        "grpo",
    ),
}


@dataclass(frozen=True, slots=True)
class ScenarioLaunch:
    scenario_id: str
    run_id: str
    provider: Literal["local", "dstack"]
    target: str
    workspace: Path
    job_workspace: Path
    command: tuple[str, ...]


def render_launch(
    scenario: QualificationScenario,
    *,
    run_id: str,
    provider: Literal["local", "dstack"],
    target: str,
    workspace: Path,
    python_executable: Path,
    trackio_server_url: str,
) -> ScenarioLaunch:
    del python_executable, trackio_server_url
    if scenario.id not in _SCENARIO_WORK_PACKAGES:
        raise NotImplementedError(f"CLI launch is not mapped for scenario {scenario.id!r}")
    if not workspace.is_absolute():
        raise ValueError("qualification workspace must be absolute")
    if scenario.update_budget is None:
        raise ValueError("training scenario is missing its update budget")
    work_package, job_id = _SCENARIO_WORK_PACKAGES[scenario.id]
    job_workspace = workspace if provider == "local" else REMOTE_WORKSPACE_ROOT
    command = (
        "uv",
        "run",
        "posttrain",
        "job",
        "run",
        work_package,
        "--job",
        job_id,
        "--provider",
        provider,
        "--target",
        target,
        "--run-id",
        run_id,
    )
    return ScenarioLaunch(
        scenario_id=scenario.id,
        run_id=run_id,
        provider=provider,
        target=target,
        workspace=workspace,
        job_workspace=job_workspace,
        command=command,
    )


def _bundle_inputs() -> dict[str, Path]:
    roots = (
        REPOSITORY / "packages" / "common" / "src",
        REPOSITORY / "packages" / "catalog" / "src",
        REPOSITORY / "packages" / "data" / "src",
        REPOSITORY / "packages" / "eval" / "src",
        REPOSITORY / "packages" / "serve" / "src",
        REPOSITORY / "packages" / "train" / "src",
        REPOSITORY / "packages" / "tracking" / "src",
        REPOSITORY / "packages" / "tracking-trackio" / "src",
        REPOSITORY / "apps" / "lab" / "src",
        REPOSITORY / "apps" / "cli" / "src",
    )
    inputs: dict[str, Path] = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
                inputs[path.relative_to(REPOSITORY).as_posix()] = path
    return inputs


def _runtime_receipt(path: Path) -> tuple[RuntimeImageRef, dict[str, object]]:
    receipts = sorted(
        path.glob("*.json"),
        key=lambda value: (value.stat().st_mtime_ns, value.name),
    )
    if not receipts:
        raise FileNotFoundError(f"veRL runtime receipt is missing under {path}")
    payload = json.loads(receipts[-1].read_text(encoding="utf-8"))
    image = payload.get("image")
    if not isinstance(image, str):
        raise ValueError("veRL runtime receipt has no immutable image")
    return RuntimeImageRef(image), payload


def _number(value: object, *, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return value


def _remote_failures(
    scenario: QualificationScenario,
    evidence: RemoteAlgorithmEvidence,
) -> tuple[str, ...]:
    failures: list[str] = []
    acceptance = scenario.acceptance
    if evidence.status != "succeeded":
        failures.append(f"Trackio status is {evidence.status!r}")
    if evidence.optimizer_updates < acceptance.minimum_optimizer_updates:
        failures.append(
            f"optimizer updates {evidence.optimizer_updates} are below {acceptance.minimum_optimizer_updates}"
        )
    if evidence.trace_count < acceptance.minimum_complete_traces:
        failures.append(f"remote traces {evidence.trace_count} are below {acceptance.minimum_complete_traces}")
    if acceptance.require_nonzero_gradient and not evidence.nonzero_gradient_observed:
        failures.append("no non-zero gradient was observed remotely")
    if acceptance.require_reward_variance and not evidence.reward_variance_observed:
        failures.append("no non-zero reward standard deviation was observed remotely")
    if acceptance.require_model_artifact and not evidence.model_artifact_observed:
        failures.append("no model artifact was observed remotely")
    if acceptance.require_remote_observatory and (
        not evidence.observatory_complete or not evidence.observatory_research_ready
    ):
        failures.append("Observatory evidence is incomplete or not research-ready")
    return tuple(failures)


def _write_receipt(
    path: Path,
    *,
    scenario: QualificationScenario,
    launch: ScenarioLaunch,
    provider_id: str,
    native_state: str,
    target_id: str,
    bundle_digest: str,
    image: str,
    evidence: RemoteAlgorithmEvidence,
) -> None:
    payload = {
        "schema": "posttrain.qualification-receipt.v1",
        "scenario": scenario.to_manifest(),
        "run_id": launch.run_id,
        "provider": launch.provider,
        "provider_id": provider_id,
        "native_state": native_state,
        "target": target_id,
        "bundle_digest": bundle_digest,
        "runtime_image": image,
        "remote": asdict(evidence),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(
            descriptor,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_dstack(
    scenario: QualificationScenario,
    launch: ScenarioLaunch,
    *,
    trackio_url: str,
    runtime_receipts: Path,
    dstack_python: Path,
    dstack_environment_file: Path,
    dstack_project: str,
    dstack_fleet: str,
    attempt: int,
) -> None:
    # The dstack SDK resolves named environment variables from this process.
    # Keep the remote value identical to the already-resolved CLI setting.
    os.environ["TRACKIO_SERVER_URL"] = trackio_url
    launch.workspace.mkdir(parents=True, exist_ok=False)
    bundle = build_bundle(
        _bundle_inputs(),
        (launch.workspace / "bundle").resolve(),
    )
    image, runtime = _runtime_receipt(runtime_receipts)
    trackio_source_digest = runtime.get("trackio_source_digest")
    if not isinstance(trackio_source_digest, str):
        raise ValueError("veRL runtime receipt has no Trackio source digest")
    trackio_revision = runtime.get("trackio_revision")
    if not isinstance(trackio_revision, str):
        raise ValueError("veRL runtime receipt has no Trackio revision")
    command = launch.command + (
        "--framework-revision",
        subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "--framework-source-digest",
        bundle.digest,
        "--trackio-revision",
        trackio_revision,
        "--trackio-source-digest",
        trackio_source_digest,
        "--runtime-image-digest",
        image.value.rsplit("@sha256:", 1)[1],
    )
    launch = replace(launch, command=command)
    run_spec = RunSpec(
        project_id="foundation-models",
        work_package_id=(f"train/qwen3.5-0.8b/{scenario.environment_ref}-qualification"),
        stage="train",
        run_id=launch.run_id,
        job_kind=scenario.job_kind,
        job_definition_version=scenario.revision,
        resolved_inputs={
            "qualification_scenario": scenario.to_manifest(),
            "execution_provider": "dstack",
            "execution_target": launch.target,
        },
        source_metadata={
            "framework_source_digest": bundle.digest,
            "runtime_image_digest": image.value.rsplit("@sha256:", 1)[1],
            "submission_surface": "posttrain-job-cli",
        },
        required_artifact_roles=(
            "trained_model",
            "training_summary",
            "verifiers_traces",
        ),
    )
    target = ExecutionTarget(
        id=f"targets/{launch.target}",
        revision="1",
        device_class="cuda",
        memory_gb=float(
            _number(
                scenario.target_capabilities["minimum_vram_gib"],
                name="minimum_vram_gib",
            )
        ),
        placement={
            "gpu_count": int(
                _number(
                    scenario.target_capabilities["accelerator_count"],
                    name="accelerator_count",
                )
            ),
            "fleets": [dstack_fleet],
            "instances": [{"hostname": launch.target}],
            "disk_gb": 160,
        },
    )
    request = ExecutionRequest(
        run_spec=run_spec,
        job_definition_id=scenario.revision,
        bundle=bundle,
        image=image,
        target=target,
        command=launch.command,
        idempotency_key=(f"qualification:{scenario.id}:{launch.run_id}:attempt-{attempt}"),
        policy=ExecutionPolicy(
            timeout_seconds=scenario.maximum_duration_seconds,
            max_attempts=1,
        ),
        environment_names=("TRACKIO_SERVER_URL", "TRACKIO_WRITE_TOKEN"),
        mounts=(
            ExecutionMount(
                Path("/var/lib/posttrain/cache/huggingface"),
                Path("/root/.cache/huggingface"),
                "model-cache",
            ),
            ExecutionMount(
                Path("/var/lib/posttrain/cache/vllm"),
                Path("/root/.cache/vllm"),
                "compile-cache",
            ),
            ExecutionMount(
                Path("/var/lib/posttrain/cache/torch-inductor"),
                Path("/root/.cache/torch-inductor"),
                "compile-cache",
            ),
            ExecutionMount(
                Path("/var/lib/posttrain/cache/triton"),
                Path("/root/.cache/triton"),
                "compile-cache",
            ),
            ExecutionMount(
                Path("/var/lib/posttrain/runs") / launch.run_id,
                REMOTE_WORKSPACE_ROOT,
                "run-workspace",
            ),
        ),
    )
    provider = DstackExecutionProvider.from_sdk_environment(
        project=dstack_project,
        python=dstack_python,
        environment_file=dstack_environment_file,
    )
    plan = provider.plan(request)
    offers_value = plan.details.get("offers", 0)
    offers = (
        int(offers_value) if isinstance(offers_value, (str, int, float)) and not isinstance(offers_value, bool) else 0
    )
    if offers < 1:
        raise RuntimeError(f"dstack found no offer for target {launch.target!r}")
    print(f"bundle_digest={bundle.digest}")
    print(f"runtime_image={image.value}")
    print(f"dstack_offers={offers}")
    handle = provider.submit(plan)
    print(f"provider_id={handle.provider_id}")
    journal = ExecutionJournal((launch.workspace / "execution-journal.jsonl").resolve())
    terminal = wait_for_terminal(
        provider,
        handle,
        timeout_seconds=scenario.maximum_duration_seconds + 300,
        journal=journal,
        poll_interval_seconds=5,
        on_transition=lambda record: print(
            f"state={record.state} native={record.native_state} target={record.target_id}"
        ),
    )
    if terminal.state != "succeeded":
        page = provider.logs(handle, limit=40)
        for line in page.lines[-40:]:
            print(f"remote_log={line}", file=sys.stderr)
        raise RuntimeError(f"dstack execution ended in {terminal.state}: {terminal.message or terminal.native_state}")
    provider.collect(handle)
    evidence = asyncio.run(
        collect_remote_evidence(
            run_id=launch.run_id,
            project="foundation-models",
            server_url=trackio_url,
        )
    )
    failures = _remote_failures(scenario, evidence)
    if failures:
        raise RuntimeError("remote qualification failed: " + "; ".join(failures))
    receipt = launch.workspace / "qualification-receipt.json"
    _write_receipt(
        receipt,
        scenario=scenario,
        launch=launch,
        provider_id=handle.provider_id,
        native_state=terminal.native_state,
        target_id=terminal.target_id,
        bundle_digest=bundle.digest,
        image=image.value,
        evidence=evidence,
    )
    print(f"receipt={receipt}")
    print(f"trackio_run={launch.run_id}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("--provider", choices=("local", "dstack"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(
            os.environ.get(
                "POSTTRAIN_QUALIFICATION_PYTHON",
                str(DEFAULT_GPU_PYTHON),
            )
        ),
    )
    parser.add_argument(
        "--trackio-url",
        default=os.environ.get("TRACKIO_SERVER_URL"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Render the immutable launch without starting GPU work.",
    )
    parser.add_argument("--runtime-receipts", type=Path)
    parser.add_argument("--dstack-python", type=Path)
    parser.add_argument("--dstack-environment-file", type=Path)
    parser.add_argument("--dstack-project", default="main")
    parser.add_argument("--dstack-fleet", default="local-gpu-workers")
    parser.add_argument("--attempt", type=int, default=1)
    return parser


def main() -> None:
    args = _parser().parse_args()
    scenario = scenario_by_id(args.scenario)
    run_id = args.run_id or str(uuid.uuid4())
    workspace = (
        args.workspace.resolve()
        if args.workspace is not None
        else (REPOSITORY / ".posttrain" / "state" / "qualification" / f"{scenario.id}-{run_id}").resolve()
    )
    trackio_url = args.trackio_url
    if not trackio_url:
        raise SystemExit("TRACKIO_SERVER_URL or --trackio-url is required for qualification")
    if "TRACKIO_WRITE_TOKEN" not in os.environ and not args.print_plan:
        raise SystemExit("TRACKIO_WRITE_TOKEN must be injected by the execution environment")
    launch = render_launch(
        scenario,
        run_id=run_id,
        provider=args.provider,
        target=args.target,
        workspace=workspace,
        python_executable=args.python_executable.absolute(),
        trackio_server_url=trackio_url,
    )
    print(f"scenario={launch.scenario_id}")
    print(f"run_id={launch.run_id}")
    print(f"provider={launch.provider}")
    print(f"target={launch.target}")
    print(f"workspace={launch.workspace}")
    if args.print_plan:
        print("command=" + " ".join(launch.command))
        return
    if args.provider == "dstack":
        missing = [
            name
            for name, value in (
                ("--runtime-receipts", args.runtime_receipts),
                ("--dstack-python", args.dstack_python),
                ("--dstack-environment-file", args.dstack_environment_file),
            )
            if value is None
        ]
        if missing:
            raise SystemExit("dstack qualification requires " + ", ".join(missing))
        if args.attempt < 1:
            raise SystemExit("--attempt must be positive")
        _run_dstack(
            scenario,
            launch,
            trackio_url=trackio_url,
            runtime_receipts=args.runtime_receipts.resolve(),
            dstack_python=args.dstack_python.absolute(),
            dstack_environment_file=args.dstack_environment_file.resolve(),
            dstack_project=args.dstack_project,
            dstack_fleet=args.dstack_fleet,
            attempt=args.attempt,
        )
        return
    completed = subprocess.run(launch.command, cwd=REPOSITORY, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
