"""Materialize or execute the general episode judge over retained native traces.

The judge receives only observable messages plus exact task-selected tool
contracts. Benchmark assertions, native rewards and hidden end state remain
outside the model input so this is usable beyond AutomationBench and cannot
leak the benchmark oracle into the learned reward.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from automationbench_v1.episode_prompt import (
    EPISODE_RUBRICS,
    EpisodeVerdict,
    build_episode_judge_messages,
    validate_episode_verdict,
)
from automationbench_v1.judge import (
    AutomationBenchTurnJudge,
    TurnQualityConfig,
    project_tool_observation,
)
from automationbench_v1.limited_tools import selected_tool_definitions


def judge_input(record: dict[str, Any]) -> dict[str, Any]:
    """Project one retained native trace into the provider-neutral judge contract."""
    trajectory = []
    for index, node in enumerate(record["nodes"]):
        message = dict(node["message"])
        message["message_id"] = f"message-{index}"
        if message["role"] == "tool" and isinstance(message.get("content"), str):
            message["content"] = project_tool_observation(message["content"])
        trajectory.append(message)
    task = record["task"]["data"]
    messages, request, digest = build_episode_judge_messages(
        trace_id=record["id"],
        trajectory=trajectory,
        available_tools=selected_tool_definitions(tuple(task.get("zapier_tools", ()))),
    )
    return {
        "trace_id": record["id"],
        "task_name": task.get("task_name") or task.get("name"),
        "messages": [message.model_dump(mode="json", exclude_none=True) for message in messages],
        "assessment_request": request,
        "input_digest": digest,
    }


def read_records(paths: list[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            identity = record["id"]
            prior = by_id.get(identity)
            if prior is not None and prior != record:
                raise ValueError(f"conflicting retained records share trace ID {identity}")
            by_id[identity] = record
    records = list(by_id.values())
    if not records:
        raise ValueError("trace input is empty")
    return records


def _native_score(record: dict[str, Any]) -> float:
    total = 0.0
    for value in record.get("rewards", {}).values():
        if isinstance(value, dict):
            total += float(value.get("score", 0)) * float(value.get("weight", 1))
        else:
            total += float(value)
    return total


def _stratum(record: dict[str, Any]) -> tuple[str, str]:
    task = record["task"]["data"].get("task_name", "unknown")
    serialized = json.dumps(record.get("nodes", ()), sort_keys=True)
    if "posttrain.rejected_tool_call" in serialized:
        outcome = "rejected_action"
    elif '"success": false' in serialized or "error" in serialized.lower():
        outcome = "tool_error"
    else:
        score = _native_score(record)
        outcome = "native_zero" if score <= 0 else "native_positive"
    return task, outcome


def stratified(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Take a stable round-robin sample across task and observed outcome strata."""
    if limit is None or limit >= len(records):
        return sorted(records, key=lambda record: record["id"])
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(_stratum(record), []).append(record)
    for rows in buckets.values():
        rows.sort(key=lambda record: record["id"])
    selected = []
    keys = sorted(buckets)
    while len(selected) < limit and keys:
        remaining = []
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


async def execute(inputs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    api_key = os.environ.get(args.api_key_var)
    if not api_key:
        raise ValueError(f"{args.api_key_var} is required for --execute")
    judge = AutomationBenchTurnJudge(
        TurnQualityConfig(
            id="general-episode-replay",
            name="quality",
            assessment_scope="episode",
            model=args.model,
            model_revision=args.model_revision,
            code_revision=args.code_revision,
            base_url=args.base_url,
            api_key_var=args.api_key_var,
            input_budget_tokens=args.input_budget_tokens,
            attempts=1,
            timeout_seconds=args.timeout_seconds,
            sampling={
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": args.enable_thinking}},
            },
        )
    )
    return await execute_with_judge(inputs, judge)


async def execute_with_judge(
    inputs: list[dict[str, Any]], judge: AutomationBenchTurnJudge
) -> list[dict[str, Any]]:
    results = []
    for item in inputs:
        response = None
        try:
            wire = item["messages"]
            messages = [
                vf.SystemMessage(content=wire[0]["content"]),
                vf.UserMessage(content=wire[1]["content"]),
            ]
            response = await judge.complete(messages, schema=EpisodeVerdict)
            verdict = EpisodeVerdict.model_validate_json(response.text)
            known = {
                message["message_id"]
                for message in item["assessment_request"]["trajectory"]
            }
            validate_episode_verdict(verdict, known)
            result = {
                "trace_id": item["trace_id"],
                "task_name": item["task_name"],
                "input_digest": item["input_digest"],
                "verdict": verdict.model_dump(mode="json"),
                "usage": response.usage.model_dump(mode="json") if response.usage else None,
            }
        except Exception as error:  # noqa: BLE001 - retain invalid judge output.
            result = {
                "trace_id": item["trace_id"],
                "task_name": item["task_name"],
                "input_digest": item["input_digest"],
                "error_type": type(error).__name__,
                "error": str(error),
                "raw_response": response.text if response is not None else None,
                "usage": (
                    response.usage.model_dump(mode="json")
                    if response is not None and response.usage
                    else None
                ),
            }
        results.append(result)
        print(json.dumps({"trace_id": item["trace_id"], "valid": "verdict" in result}), flush=True)
    return results


async def run_materialized_with_judge(
    inputs_path: Path,
    output: Path,
    judge: AutomationBenchTurnJudge,
) -> dict[str, Any]:
    """Execute already materialized inputs without rebuilding or adding labels."""
    inputs = [json.loads(line) for line in inputs_path.read_text().splitlines() if line.strip()]
    if not inputs:
        raise ValueError("materialized judge input is empty")
    output.mkdir(parents=True, exist_ok=False)
    results = await execute_with_judge(inputs, judge)
    write_jsonl(output / "judge-results.jsonl", results)
    valid = [result for result in results if "verdict" in result]
    manifest = {
        "episodes": len(inputs),
        "valid_episodes": len(valid),
        "invalid_episodes": len(inputs) - len(valid),
        "input_digests": [item["input_digest"] for item in inputs],
        "summary": summarize(valid),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = {name: [] for name in EPISODE_RUBRICS}
    violations: Counter[str] = Counter()
    all_perfect = 0
    for result in results:
        verdict = result["verdict"]
        values = []
        for name, rating in verdict["assessments"].items():
            if rating["status"] == "valid":
                scores[name].append(rating["score"])
                values.append(rating["score"])
        all_perfect += bool(values) and all(score == 1 for score in values)
        for check in verdict["requirement_checks"]:
            if check["outcome"].startswith("violated_"):
                violations[check["outcome"].removeprefix("violated_")] += 1
    return {
        "episodes": len(results),
        "all_dimensions_perfect_episodes": all_perfect,
        "violations_by_severity": dict(sorted(violations.items())),
        "dimension_scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--trace-id", action="append", dest="trace_ids")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--model-revision", default="0" * 40)
    parser.add_argument("--code-revision", default="0" * 40)
    parser.add_argument("--api-key-var", default="POSTTRAIN_REPLAY_API_KEY")
    parser.add_argument("--input-budget-tokens", type=int, default=12_288)
    parser.add_argument("--max-tokens", type=int, default=12_288)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.execute and (not args.base_url or not args.model):
        parser.error("--execute requires --base-url and --model")
    if args.input_budget_tokens < 1 or args.max_tokens < 1:
        parser.error("token budgets must be positive")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    retained = read_records(args.traces)
    if args.trace_ids:
        requested = set(args.trace_ids)
        selected = [record for record in retained if record["id"] in requested]
        found = {record["id"] for record in selected}
        if found != requested:
            raise ValueError(f"unknown requested trace IDs: {sorted(requested - found)}")
        selected.sort(key=lambda record: args.trace_ids.index(record["id"]))
    else:
        selected = stratified(retained, args.max_episodes)
    inputs = [judge_input(record) for record in selected]
    write_jsonl(args.output / "judge-inputs.jsonl", inputs)
    write_jsonl(
        args.output / "corpus-index.jsonl",
        [
            {
                "trace_id": record["id"],
                "task_name": _stratum(record)[0],
                "outcome_stratum": _stratum(record)[1],
                "native_score": _native_score(record),
            }
            for record in selected
        ],
    )
    manifest = {
        "mode": "execute" if args.execute else "materialize-only",
        "episodes": len(inputs),
        "retained_unique_episodes": len(retained),
        "source_files": [str(path.resolve()) for path in args.traces],
        "strata": {
            "|".join(key): count
            for key, count in sorted(Counter(_stratum(record) for record in selected).items())
        },
        "contract": inputs[0]["assessment_request"]["contract"],
        "input_digests": [item["input_digest"] for item in inputs],
        "native_oracles_sent_to_judge": False,
    }
    if args.execute:
        results = asyncio.run(execute(inputs, args))
        write_jsonl(args.output / "judge-results.jsonl", results)
        manifest["summary"] = summarize(results)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
