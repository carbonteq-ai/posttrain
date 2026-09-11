"""Temporary, model-executable calibration harness for the general episode prompt."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from automationbench_v1.episode_prompt import (
    EpisodeVerdict,
    WireEpisodeVerdict,
    build_episode_judge_messages,
    normalize_wire_verdict,
    validate_episode_verdict,
)
from automationbench_v1.judge import AutomationBenchEpisodeJudge, EpisodeQualityConfig


def load_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fixture = json.loads(path.read_text())
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("calibration fixture requires cases")
    canonical = json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()
    return fixture, [
        {
            **case,
            "messages": [
                message.model_dump(mode="json", exclude_none=True)
                for message in build_episode_judge_messages(
                    trace_id=case["id"],
                    trajectory=case["trajectory"],
                    available_tools=case["available_tools"],
                )[0]
            ],
            "input_digest": build_episode_judge_messages(
                trace_id=case["id"],
                trajectory=case["trajectory"],
                available_tools=case["available_tools"],
            )[2],
            "fixture_sha256": hashlib.sha256(canonical).hexdigest(),
        }
        for case in cases
    ]


def assess_expectations(case: dict[str, Any], verdict: EpisodeVerdict) -> list[str]:
    failures = []
    expected = case.get("expect", {})
    violated_dimensions = {
        dimension
        for check in verdict.requirement_checks
        if check.outcome.startswith("violated_")
        for dimension in check.relevant_dimensions
    }
    missing = set(expected.get("violated_dimensions", ())) - violated_dimensions
    if missing:
        failures.append("missing violated dimensions: " + ", ".join(sorted(missing)))
    for name, maximum in expected.get("score_max", {}).items():
        score = verdict.assessments[name].score
        if score is None or score > maximum:
            failures.append(f"{name} score {score} exceeds {maximum}")
    for name, minimum in expected.get("score_min", {}).items():
        score = verdict.assessments[name].score
        if score is None or score < minimum:
            failures.append(f"{name} score {score} is below {minimum}")
    return failures


async def execute(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not os.environ.get(args.api_key_var):
        raise ValueError(f"{args.api_key_var} is required for --execute")
    judge = AutomationBenchEpisodeJudge(
        EpisodeQualityConfig(
            id="general-episode-calibration",
            name="quality",
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
    return await execute_with_judge(cases, judge)


async def execute_with_judge(cases: list[dict[str, Any]], judge: AutomationBenchEpisodeJudge) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        response = None
        try:
            wire = case["messages"]
            messages = [
                vf.SystemMessage(content=wire[0]["content"]),
                vf.UserMessage(content=wire[1]["content"]),
            ]
            response = await judge.complete(messages, schema=WireEpisodeVerdict)
            wire_verdict = WireEpisodeVerdict.model_validate_json(response.text)
            message_ids = [message["message_id"] for message in case["trajectory"]]
            verdict = normalize_wire_verdict(wire_verdict, message_ids)
            validate_episode_verdict(verdict, set(message_ids))
            failures = assess_expectations(case, verdict)
            row = {
                "case": case["id"],
                "passed": not failures,
                "failures": failures,
                "input_digest": case["input_digest"],
                "verdict": verdict.model_dump(mode="json"),
                "usage": response.usage.model_dump(mode="json") if response.usage else None,
            }
        except Exception as error:  # noqa: BLE001 - calibration retains invalid outputs.
            row = {
                "case": case["id"],
                "passed": False,
                "failures": [f"{type(error).__name__}: {error}"],
                "input_digest": case["input_digest"],
                "error_type": type(error).__name__,
                "raw_response": response.text if response is not None else None,
                "usage": (response.usage.model_dump(mode="json") if response is not None and response.usage else None),
            }
        rows.append(row)
        print(json.dumps({"case": row["case"], "passed": row["passed"]}), flush=True)
    return rows


async def run_with_judge(
    fixture_path: Path,
    output: Path,
    judge: AutomationBenchEpisodeJudge,
) -> dict[str, Any]:
    """Run the frozen controls against an already bound inference service."""
    fixture, cases = load_cases(fixture_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "fixture.json").write_text(json.dumps(fixture, indent=2))
    (output / "judge-inputs.jsonl").write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases))
    results = await execute_with_judge(cases, judge)
    (output / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
    summary = {
        "mode": "execute",
        "fixture_sha256": cases[0]["fixture_sha256"],
        "cases": len(cases),
        "passed": all(row["passed"] for row in results),
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--model-revision", default="0" * 40)
    parser.add_argument("--code-revision", default="0" * 40)
    parser.add_argument("--api-key-var", default="POSTTRAIN_CALIBRATION_API_KEY")
    parser.add_argument("--input-budget-tokens", type=int, default=12_288)
    parser.add_argument("--max-tokens", type=int, default=12_288)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.execute and (not args.base_url or not args.model):
        parser.error("--execute requires --base-url and --model")
    args.output.mkdir(parents=True, exist_ok=False)
    fixture, cases = load_cases(args.fixture)
    (args.output / "fixture.json").write_text(json.dumps(fixture, indent=2))
    (args.output / "judge-inputs.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
    )
    summary: dict[str, Any] = {
        "mode": "execute" if args.execute else "materialize-only",
        "fixture_sha256": cases[0]["fixture_sha256"],
        "cases": len(cases),
    }
    if args.execute:
        results = asyncio.run(execute(cases, args))
        (args.output / "results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results)
        )
        summary.update(passed=all(row["passed"] for row in results), results=results)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
