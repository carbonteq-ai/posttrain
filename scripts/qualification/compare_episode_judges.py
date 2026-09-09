"""Replay one frozen episode-judge corpus through one explicit judge binding.

The AutomationBench environment owns the scorer and rubric. This harness owns
only catalog resolution, managed/external service lifecycle, identical-input
replay, and secret-free evidence. Invoke it once per judge work package, then
compare outputs by their retained input digests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast

from posttrain.catalog import open_catalog
from posttrain.common import HostedInferenceBinding, InferenceBinding, NullObserver, RunContext
from posttrain.environment import EnvironmentBinding, VerifiersV1ConfigActivation
from posttrain.jobs import (
    ExternalInferenceServiceRequest,
    ManagedInferenceService,
    bind_inference_services,
    bind_native_judge_services,
)
from posttrain.serve import ServeLaunchRequest
from posttrain.work import load_work_package, resolve_work_package

WORKSPACE = Path(__file__).resolve().parents[2]
LAB_CATALOG = WORKSPACE / "apps" / "lab" / ".posttrain" / "catalog"


def service_request(
    inference: HostedInferenceBinding | InferenceBinding,
    *,
    port: int,
) -> ExternalInferenceServiceRequest | ManagedInferenceService:
    """Map a truthful catalog selection to its lifecycle request."""

    if isinstance(inference, HostedInferenceBinding):
        return ExternalInferenceServiceRequest(inference)
    return ManagedInferenceService(ServeLaunchRequest(inference, port=port))


def judge_config(environment: EnvironmentBinding, name: str) -> Any:
    """Read the task-owned judge config after endpoint injection."""

    from automationbench_v1.judge import TurnQualityConfig

    activation = environment.activation
    if not isinstance(activation, VerifiersV1ConfigActivation):
        raise ValueError("episode judge comparison requires declarative Verifiers activation")
    raw = cast(Any, activation.config)
    entries = raw.get("taskset", {}).get("task", {}).get("judges", [])
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"environment must declare exactly one judge named {name!r}")
    return TurnQualityConfig.model_validate(matches[0])


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from automationbench_v1.judge import AutomationBenchTurnJudge
    from calibrate_general_episode_prompt import run_with_judge
    from replay_episode_judge import run_materialized_with_judge

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite comparison evidence: {args.output}")
    package = load_work_package(args.work_package)
    catalog = open_catalog(scope=package.project_id, overlays=(LAB_CATALOG,))
    resolved = resolve_work_package(catalog, package)
    environment = resolved.seat("environment", EnvironmentBinding)
    selected = resolved.seats["judge_inference"].value
    if not isinstance(selected, HostedInferenceBinding | InferenceBinding):
        raise TypeError("judge_inference must resolve to hosted or managed inference")

    args.output.mkdir(parents=True)
    context = RunContext(
        project_id=package.project_id,
        work_package_id=package.work_package_id,
        run_id=args.run_id,
        job_kind="qualify.judge",
        job_definition_version="qualify/episode-judge-comparison@1",
        workspace=(args.output / "runtime").resolve(),
        observer=NullObserver(),
    )
    service_name = f"judge/{args.judge_name}"
    requests = {service_name: service_request(selected, port=args.port)}
    with bind_inference_services(context, requests) as services:
        service = services[service_name]
        receipt = service.trace_identity()
        with bind_native_judge_services(
            environment,
            services,
            {args.judge_name: service_name},
        ) as bound:
            judge = AutomationBenchTurnJudge(judge_config(bound, args.judge_name))
            if args.fixture is not None:
                result = await run_with_judge(args.fixture, args.output / "results", judge)
                input_kind = "reviewed-fixture"
                input_path = args.fixture
            else:
                assert args.replay_inputs is not None
                result = await run_materialized_with_judge(args.replay_inputs, args.output / "results", judge)
                input_kind = "materialized-replay"
                input_path = args.replay_inputs

    (args.output / "service-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    summary = {
        "schema": "posttrain.episode-judge-comparison.v1",
        "work_package": str(args.work_package.resolve()),
        "work_package_id": package.work_package_id,
        "judge_binding_id": selected.id,
        "judge_binding_revision": selected.revision,
        "input_kind": input_kind,
        "input": str(input_path.resolve()),
        "result": result,
        "service_receipt": "service-receipt.json",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fixture", type=Path)
    inputs.add_argument("--replay-inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="episode-judge-local-comparison")
    parser.add_argument("--judge-name", default="quality")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535")
    return cast(argparse.Namespace, args)


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(run(parse_args()))["result"].get("passed", True) else 1)
