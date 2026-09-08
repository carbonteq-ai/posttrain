"""Evaluate a pinned base or retained LoRA adapter on a frozen AutomationBench slice."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
from pathlib import Path

from automationbench_structured_toy import FileObserver, plain
from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, ExecutionTarget, InferenceBinding, LocalArtifactRef, RunContext
from posttrain.environment import VerifiersV1ConfigActivation
from posttrain.environment.verifiers_evidence import (
    verifiers_trace_has_error,
    verifiers_trace_is_truncated,
)
from posttrain.eval import (
    EvaluateRequest,
    EvaluationBudget,
    EvaluationEndpoint,
    EvaluationNumericPredicate,
    EvaluationPlan,
    EvaluationSignalRef,
    EvaluationSuccessDefinition,
    evaluate,
)
from posttrain.serve import ServeLaunchRequest, launch


def _subject(training_output: Path | None):
    policy = open_catalog(scope="lfm26-comparison-eval").resolve(
        CatalogRef("model", "models/lfm2.5-2.6b@bf16")
    ).value
    if training_output is None:
        return policy
    payload = json.loads((training_output / "result.json").read_text())
    model = payload["model"]
    artifact = model["artifact"]
    return dataclasses.replace(
        policy,
        id=model["id"],
        artifact=LocalArtifactRef(Path(artifact["path"]), artifact["digest"]),
        form=model["form"],
        revision=model.get("revision"),
        digest=model["digest"],
        parent=model["parent"],
        provenance=model["provenance"],
    )


def _flatten_traces(path: Path):
    for line in path.read_text().splitlines():
        row = json.loads(line)
        yield from row.get("traces", (row,))


def _score_value(value) -> float:
    if isinstance(value, dict):
        return float(value["score"])
    return float(value)


def _summarize(path: Path) -> dict:
    traces = list(_flatten_traces(path))
    partial = [_score_value(trace["rewards"]["partial_credit"]) for trace in traces]
    success = [_score_value(trace["metrics"]["task_completed_correctly"]) for trace in traces]
    def mean(values):
        return math.fsum(values) / len(values)

    def wilson(values):
        count = len(values)
        proportion = mean(values)
        z = 1.959963984540054
        denominator = 1 + z**2 / count
        center = (proportion + z**2 / (2 * count)) / denominator
        margin = z * math.sqrt(proportion * (1 - proportion) / count + z**2 / (4 * count**2)) / denominator
        return [max(0.0, center - margin), min(1.0, center + margin)]

    per_task = {}
    per_domain = {}
    truncation_reasons: collections.Counter[str] = collections.Counter()
    for trace in traces:
        if verifiers_trace_is_truncated(trace):
            stop = str(trace.get("stop_condition") or "")
            calls = trace.get("calls", ())
            reason = stop if stop and stop != "agent_completed" else "provider_output_length"
            if calls and calls[-1].get("finish_reason") == "length":
                reason = "provider_output_length"
            truncation_reasons[reason] += 1
        task_data = trace["task"]["data"]
        task_name = str(task_data["task_name"])
        domain = str(task_data["domain"])
        row = per_task.setdefault(
            task_name,
            {
                "idx": int(task_data["idx"]),
                "domain": domain,
                "partial_credit": [],
                "exact_task_completion": [],
            },
        )
        row["partial_credit"].append(_score_value(trace["rewards"]["partial_credit"]))
        row["exact_task_completion"].append(_score_value(trace["metrics"]["task_completed_correctly"]))
        domain_row = per_domain.setdefault(domain, {"partial_credit": [], "exact_task_completion": []})
        domain_row["partial_credit"].append(_score_value(trace["rewards"]["partial_credit"]))
        domain_row["exact_task_completion"].append(
            _score_value(trace["metrics"]["task_completed_correctly"])
        )

    return {
        "trajectories": len(traces),
        "task_indices": sorted({int(trace["task"]["data"]["idx"]) for trace in traces}),
        "partial_credit_mean": mean(partial),
        "exact_task_completion_rate": mean(success),
        "exact_task_completion_wilson_95": wilson(success),
        "partial_credit_values": partial,
        "exact_task_completion_values": success,
        "per_task": {
            key: {
                "idx": value["idx"],
                "domain": value["domain"],
                "partial_credit_mean": mean(value["partial_credit"]),
                "exact_task_completion_rate": mean(value["exact_task_completion"]),
            }
            for key, value in sorted(per_task.items())
        },
        "per_domain": {
            key: {
                "trajectories": len(value["partial_credit"]),
                "partial_credit_mean": mean(value["partial_credit"]),
                "exact_task_completion_rate": mean(value["exact_task_completion"]),
            }
            for key, value in sorted(per_domain.items())
        },
        "complete": sum(bool(trace.get("ok", trace.get("is_completed", False))) for trace in traces),
        "failed": sum(verifiers_trace_has_error(trace) for trace in traces),
        "truncated": sum(verifiers_trace_is_truncated(trace) for trace in traces),
        "truncation_reasons": dict(sorted(truncation_reasons.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-output", type=Path)
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--task-mix", type=Path, required=True)
    parser.add_argument("--rollouts-per-task", type=int, default=2)
    parser.add_argument("--context-window", type=int, default=12_288)
    parser.add_argument("--max-concurrent", type=int, default=64)
    args = parser.parse_args()
    if args.context_window < 4_096:
        parser.error("context window must leave room for prompts and responses")
    if args.max_concurrent < 1 or args.rollouts_per_task < 1:
        parser.error("concurrency and rollouts per task must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    training_output = args.training_output.resolve() if args.training_output is not None else None
    task_mix = json.loads(args.task_mix.read_text())
    task_names = [row["name"] for row in task_mix["evaluation"]]
    model = _subject(training_output)
    catalog = open_catalog(scope="lfm26-comparison-eval")
    native = catalog.resolve(CatalogRef("environment", "automationbench-zapier-simple-grpo")).value
    environment = dataclasses.replace(
        native,
        activation=VerifiersV1ConfigActivation(
            {
                "taskset": {
                    "id": "automationbench-v1",
                    "domains": task_mix["domains"],
                    "task_names": task_names,
                    "task": {"toolset": "limited_zapier"},
                },
                "agent": {
                    "harness": {"id": "null"},
                    "runtime": {"type": "subprocess"},
                    "timeout": {"setup": 120, "rollout": 1800, "finalize": 60, "scoring": 300},
                    "max_turns": 12,
                    "max_total_tokens": 8192,
                },
            }
        ),
        sampling=dataclasses.replace(native.sampling, max_tokens=2048, temperature=0.8, top_p=0.95),
        parameters={
            **native.parameters,
            "domains": task_mix["domains"],
            "toolset": "limited_zapier",
            "max_turns": 12,
            "max_total_tokens": 8192,
        },
        num_tasks=len(task_names),
        num_rollouts=args.rollouts_per_task,
        max_concurrent=args.max_concurrent,
    )
    plan = EvaluationPlan(
        id="automationbench-lfm26-heldout-v1",
        kind="domain",
        environments=(environment,),
        success={
            environment.id: EvaluationSuccessDefinition(
                id="exact-task-completion",
                label="Exact task completion",
                source=EvaluationSignalRef("metric", "task_completed_correctly"),
                predicate=EvaluationNumericPredicate("eq", 1.0),
            )
        },
        comparison={
            "task_selection": "explicit-evaluation-list",
            "training_selection": "explicit-disjoint-training-list",
            "known_training_overlap": [],
        },
    )
    target = ExecutionTarget("targets/qualification-workstation", "1", "nvidia-cuda", 96, {"world_size": 1})
    inference = InferenceBinding(
        "inference/lfm26-comparison-eval",
        "1",
        model,
        "vllm@0.25.2.dev2",
        model.renderer_contract,
        {
            "max_model_len": args.context_window,
            "gpu_memory_utilization": 0.60,
            "dtype": "bfloat16",
            "enforce_eager": True,
            "max_num_seqs": args.max_concurrent,
            "text_only": True,
            "skip_mm_profiling": True,
            "enable_chunked_prefill": True,
        },
        {"max_tokens": 2048, "temperature": 0.8, "top_p": 0.95},
        target,
        ("eval",),
        capabilities=("tool-calling",),
        startup_timeout_seconds=600,
    )
    context = RunContext(
        "qualification",
        "lfm26-algorithm-comparison",
        output.name,
        "eval.domain",
        "1",
        output,
        FileObserver(output / "observations.jsonl"),
    )
    with launch(context, ServeLaunchRequest(inference, port=args.port)) as endpoint:
        request = EvaluateRequest(
            model=model,
            plan=plan,
            inference=inference,
            target=target,
            endpoint=EvaluationEndpoint(endpoint.base_url, endpoint.model),
            environment_id=environment.id,
            context_window=args.context_window,
            reasoning_mode="native",
            budget=EvaluationBudget(
                num_tasks=len(task_names),
                num_rollouts=args.rollouts_per_task,
                max_concurrent=args.max_concurrent,
                shuffle=False,
            ),
        )
        result = evaluate(context, request)
    (output / "selection.json").write_text(
        json.dumps(
            {
                "model": plain(model),
                "plan": plain(plan),
                "inference": plain(inference),
                "training_output": str(training_output) if training_output else None,
                "task_mix": task_mix,
            },
            default=str,
            indent=2,
        )
    )
    (output / "result.json").write_text(json.dumps(plain(result), default=str, indent=2))
    summary = _summarize(output / "evaluation" / environment.id / "traces.jsonl")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
