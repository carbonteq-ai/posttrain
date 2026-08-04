"""Validate one substantial algorithm run from local and remote evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from posttrain.tracking import TraceQuery
from posttrain_observatory import ObservatoryService
from posttrain_tracking_trackio import TrackioDataSource

from posttrain_lab.qualification.scenarios import (
    QualificationScenario,
    scenario_by_id,
)


@dataclass(frozen=True, slots=True)
class LocalAlgorithmEvidence:
    optimizer_updates: int
    trace_count: int
    completed_trace_count: int
    reward_variant_group_observed: bool
    continued_after_tool_call_observed: bool
    nonzero_gradient_observed: bool
    adapter_digest: str | None
    checkpoint_digest: str | None
    runtime_seconds: float | None


@dataclass(frozen=True, slots=True)
class RemoteAlgorithmEvidence:
    status: str
    trace_count: int
    optimizer_updates: int
    gradient_points: int
    reward_std_points: int
    rollout_population_points: int
    nonzero_gradient_observed: bool
    reward_variance_observed: bool
    model_artifact_observed: bool
    observatory_mode: str
    observatory_complete: bool
    observatory_research_ready: bool
    tool_evidence_state: str | None
    provider_run_id: str | None


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise TypeError(f"{path} must contain JSON objects")
    return records


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _reward(record: Mapping[str, Any]) -> float | None:
    rewards = record.get("rewards")
    if not isinstance(rewards, Mapping):
        return None
    values = [
        float(value) for value in rewards.values() if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    return sum(values) if values else None


def _continued_after_tool_call(record: Mapping[str, Any]) -> bool:
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        return False
    sampled = [node for node in nodes if isinstance(node, Mapping) and node.get("sampled") is True]
    return len(sampled) > 1 and any(
        isinstance((message := node.get("message")), Mapping) and bool(message.get("tool_calls"))
        for node in sampled[:-1]
    )


def collect_local_evidence(
    scenario: QualificationScenario,
    workspace: Path,
) -> LocalAlgorithmEvidence:
    technique = scenario.job_kind.removeprefix("train.")
    root = workspace / "training" / technique
    metric_records = _json_lines(root / "trainer" / "verl-metrics.jsonl")
    trace_records = _json_lines(root / "verifiers-traces.jsonl")
    steps = {
        int(data["training/global_step"])
        for record in metric_records
        if isinstance((data := record.get("data")), Mapping)
        and isinstance(data.get("training/global_step"), int | float)
    }
    gradients = [
        float(value)
        for record in metric_records
        if isinstance((data := record.get("data")), Mapping)
        and isinstance((value := data.get("actor/grad_norm")), int | float)
        and not isinstance(value, bool)
    ]
    grouped_rewards: dict[tuple[int, str], list[float]] = defaultdict(list)
    for record in trace_records:
        run = record.get("run")
        info = record.get("info")
        reward = _reward(record)
        if (
            isinstance(run, Mapping)
            and isinstance(run.get("step"), int)
            and isinstance(info, Mapping)
            and reward is not None
        ):
            grouped_rewards[(run["step"], str(info.get("example_id") or ""))].append(reward)
    adapter = root / "trainer" / "model" / "lora_adapter"
    checkpoint = root / "trainer" / "checkpoints" / (f"global_step_{max(steps)}" if steps else "missing")
    runtimes = [
        float(record["data"]["perf/time_per_step"])
        for record in metric_records
        if isinstance(record.get("data"), Mapping) and isinstance(record["data"].get("perf/time_per_step"), int | float)
    ]
    return LocalAlgorithmEvidence(
        optimizer_updates=len(steps),
        trace_count=len(trace_records),
        completed_trace_count=sum(record.get("is_completed") is True for record in trace_records),
        reward_variant_group_observed=any(
            len(values) > 1 and statistics.pstdev(values) > 0 for values in grouped_rewards.values()
        ),
        continued_after_tool_call_observed=any(_continued_after_tool_call(record) for record in trace_records),
        nonzero_gradient_observed=any(value > 0 for value in gradients),
        adapter_digest=_tree_digest(adapter) if adapter.is_dir() else None,
        checkpoint_digest=_tree_digest(checkpoint) if checkpoint.is_dir() else None,
        runtime_seconds=sum(runtimes) if runtimes else None,
    )


async def collect_remote_evidence(
    *,
    run_id: str,
    project: str,
    server_url: str,
) -> RemoteAlgorithmEvidence:
    source = TrackioDataSource(project, server_url=server_url)
    detail = await source.get_run(run_id)
    traces = await source.traces(run_id, TraceQuery(limit=1000))
    artifacts = await source.artifacts(run_id)
    names = (
        "train/grad_norm",
        "train/rl/reward_std",
        "train/rl/rollouts_completed",
    )
    series = {item.name: item for item in await source.metric_series(run_id, names)}
    response = await ObservatoryService({"shared-trackio": source}).get_run_view_response(run_id)
    completeness = getattr(response.view, "completeness", None)
    tool_state = None
    if completeness is not None:
        tool = next(
            (requirement for requirement in completeness.requirements if requirement.key == "tool_behavior"),
            None,
        )
        tool_state = tool.state if tool is not None else None
    gradient_points = len(series["train/grad_norm"].points)
    gradient_values = [point.value for point in series["train/grad_norm"].points]
    reward_std_values = [point.value for point in series["train/rl/reward_std"].points]
    return RemoteAlgorithmEvidence(
        status=detail.summary.status,
        trace_count=len(traces.items),
        optimizer_updates=gradient_points,
        gradient_points=gradient_points,
        reward_std_points=len(series["train/rl/reward_std"].points),
        rollout_population_points=len(series["train/rl/rollouts_completed"].points),
        nonzero_gradient_observed=any(value > 0 for value in gradient_values),
        reward_variance_observed=any(value > 0 for value in reward_std_values),
        model_artifact_observed=any(
            item.direction == "output" and item.kind in {"model", "model-adapter"} for item in artifacts.items
        ),
        observatory_mode=response.resolved_mode,
        observatory_complete=(completeness is not None and completeness.state == "complete"),
        observatory_research_ready=(completeness is not None and completeness.research_ready),
        tool_evidence_state=tool_state,
        provider_run_id=detail.summary.provider_run_id,
    )


def acceptance_failures(
    scenario: QualificationScenario,
    local: LocalAlgorithmEvidence,
    remote: RemoteAlgorithmEvidence,
) -> tuple[str, ...]:
    required = scenario.acceptance
    failures: list[str] = []
    if local.optimizer_updates < required.minimum_optimizer_updates:
        failures.append("local optimizer update count is below the acceptance minimum")
    if remote.optimizer_updates < required.minimum_optimizer_updates:
        failures.append("remote optimizer update evidence is below the acceptance minimum")
    if local.completed_trace_count < required.minimum_complete_traces:
        failures.append("local complete trace count is below the acceptance minimum")
    if remote.trace_count < required.minimum_complete_traces:
        failures.append("remote live trace count is below the acceptance minimum")
    if required.require_reward_variance and not local.reward_variant_group_observed:
        failures.append("no within-group reward variance was observed")
    if required.require_nonzero_gradient and not local.nonzero_gradient_observed:
        failures.append("no non-zero gradient was observed")
    if required.require_model_artifact and (local.adapter_digest is None or not remote.model_artifact_observed):
        failures.append("the trained model artifact is incomplete")
    if remote.status != "succeeded":
        failures.append(f"remote run status is {remote.status!r}")
    if required.require_remote_observatory and not (
        remote.observatory_mode == "job" and remote.observatory_complete and remote.observatory_research_ready
    ):
        failures.append("remote Observatory is not complete and research-ready")
    return tuple(failures)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--trackio-project", default="posttrain-lab")
    parser.add_argument("--trackio-url", required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    scenario = scenario_by_id(args.scenario)
    workspace = args.workspace.resolve()
    local = collect_local_evidence(scenario, workspace)
    remote = asyncio.run(
        collect_remote_evidence(
            run_id=args.run_id,
            project=args.trackio_project,
            server_url=args.trackio_url,
        )
    )
    failures = acceptance_failures(scenario, local, remote)
    receipt = {
        "schema": "posttrain.algorithm-qualification.v1",
        "scenario": scenario.to_manifest(),
        "run_id": args.run_id,
        "workspace": str(workspace),
        "trackio_project": args.trackio_project,
        "local": asdict(local),
        "remote": asdict(remote),
        "failures": list(failures),
        "status": "succeeded" if not failures else "failed",
    }
    destination = (args.receipt or workspace / "qualification-receipt.json").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"receipt={destination}")
    if failures:
        raise SystemExit("qualification failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
