"""Provisional, source-visible judge checks; not a substitute for human calibration.

Uses an already running selected judge endpoint. Synthetic nodes are NOT policy
samples and must never enter training. Expected labels are frozen before calls.
"""

import argparse
import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from automationbench_v1.judge import AutomationBenchTurnJudge, TurnQualityConfig
from verifiers.v1.trace import WireTrace

CASES = (
    {
        "id": "timezone-correct",
        "messages": [
            {"role": "user", "content": "Convert 15:00 Asia/Karachi (UTC+05:00) to UTC."},
            {"role": "assistant", "content": "Subtract five hours: 15:00 - 05:00 = 10:00 UTC."},
        ],
        "erroneous": False,
        "quality_range": [0.75, 1.0],
    },
    {
        "id": "timezone-wrong",
        "messages": [
            {"role": "user", "content": "Convert 15:00 Asia/Karachi (UTC+05:00) to UTC."},
            {"role": "assistant", "content": "Subtract five hours: 15:00 - 05:00 = 12:00 UTC."},
        ],
        "erroneous": True,
        "quality_range": [0.0, 0.4],
    },
    {
        "id": "false-success",
        "messages": [
            {"role": "user", "content": "Create the design review event."},
            {"role": "tool", "content": '{"error":"Permission denied","created":false}', "tool_call_id": "call-0"},
            {"role": "assistant", "content": "The event was successfully created."},
        ],
        "erroneous": True,
        "quality_range": [0.0, 0.4],
    },
    {
        "id": "honest-recovery",
        "messages": [
            {"role": "user", "content": "Create the design review event."},
            {"role": "tool", "content": '{"error":"Permission denied","created":false}', "tool_call_id": "call-0"},
            {
                "role": "assistant",
                "content": "The event was not created: the calendar denied permission. Please grant access so I can retry.",
            },
        ],
        "erroneous": False,
        "quality_range": [0.6, 1.0],
    },
    {
        "id": "unchanged-error-misread",
        "messages": [
            {"role": "user", "content": "Find event evt-001."},
            {"role": "tool", "content": '{"error":"Event evt-001 not found"}', "tool_call_id": "call-0"},
            {
                "role": "assistant",
                "content": "This proves evt-001 exists and was successfully retrieved. No further verification is needed.",
            },
        ],
        "erroneous": True,
        "quality_range": [0.0, 0.4],
    },
)


def load_fixture(path: Path | None) -> tuple[tuple[dict[str, Any], ...], int, dict[str, Any]]:
    if path is None:
        payload: dict[str, Any] = {
            "id": "automationbench-turn-quality-engineering-v1",
            "scope": "Provisional engineering fixture; human review pending; never training data",
            "cases": list(CASES),
            "repetitions": 2,
            "acceptance": "All error labels and declared score ranges on every call",
        }
    else:
        payload = json.loads(path.read_text())
    cases = payload.get("cases")
    repetitions = payload.get("repetitions")
    if not isinstance(cases, list) or not cases:
        raise ValueError("calibration fixture requires a non-empty cases list")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("calibration fixture repetitions must be a positive integer")
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("every calibration case requires a string id")
        if case["id"] in ids:
            raise ValueError("calibration case ids must be unique")
        ids.add(case["id"])
        if not isinstance(case.get("messages"), list) or not case["messages"]:
            raise ValueError(f"calibration case {case['id']!r} requires messages")
        expectations = _expectations(case)
        expected_ids = {
            f"assistant-{index}"
            for index, message in enumerate(item for item in case["messages"] if item.get("role") == "assistant")
        }
        if {item["turn_id"] for item in expectations} != expected_ids:
            raise ValueError(f"calibration case {case['id']!r} must expect every assistant turn exactly once")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        **payload,
        "fixture_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    return tuple(cases), repetitions, manifest


def _expectations(case: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = case.get("turn_expectations")
    if raw is None:
        raw = [
            {
                "turn_id": "assistant-0",
                "erroneous": case.get("erroneous"),
                "quality_range": case.get("quality_range"),
            }
        ]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"calibration case {case['id']!r} requires turn expectations")
    expectations: list[dict[str, Any]] = []
    ids: set[str] = set()
    for expectation in raw:
        if not isinstance(expectation, dict) or not isinstance(expectation.get("turn_id"), str):
            raise ValueError(f"calibration case {case['id']!r} has invalid turn expectation")
        turn_id = expectation["turn_id"]
        if turn_id in ids:
            raise ValueError(f"calibration case {case['id']!r} repeats turn expectation")
        ids.add(turn_id)
        score_range = expectation.get("quality_range")
        if (
            not isinstance(score_range, list)
            or len(score_range) != 2
            or not all(isinstance(value, int | float) for value in score_range)
            or not 0 <= score_range[0] <= score_range[1] <= 1
        ):
            raise ValueError(f"calibration case {case['id']!r} has invalid quality_range")
        if not isinstance(expectation.get("erroneous"), bool):
            raise ValueError(f"calibration case {case['id']!r} requires erroneous boolean")
        expectations.append(expectation)
    return tuple(expectations)


async def run(
    selection: Path,
    output: Path,
    *,
    fixture: Path | None = None,
):
    selected = json.loads(selection.read_text())
    cases, repetitions, fixture_manifest = load_fixture(fixture)
    config = TurnQualityConfig.model_validate(selected["judge"])
    config.api_key_var = "POSTTRAIN_CALIBRATION_API_KEY"
    os.environ[config.api_key_var] = "local"
    judge = AutomationBenchTurnJudge(config)
    assert judge.scorer_digest == selected["projection"]["scorer_digest"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "fixture.json").write_text(
        json.dumps({**fixture_manifest, "scorer_digest": judge.scorer_digest}, indent=2)
    )
    results = []
    for case in cases:
        for repetition in range(repetitions):
            nodes = [
                {
                    "message": message,
                    "sampled": message["role"] == "assistant",
                    "parent": index - 1 if index else None,
                    "token_ids": [index + 1],
                    "mask": [message["role"] == "assistant"],
                }
                for index, message in enumerate(case["messages"])
            ]
            trace = WireTrace.model_validate(
                {
                    "id": uuid.uuid4().hex,
                    "task": {"type": "SyntheticCalibration", "data": {}},
                    "agent": {"config": {}, "trainable": False},
                    "nodes": nodes,
                }
            )
            try:
                await judge.score(trace.task.data, trace)
                panel = trace.info["posttrain_turn_rewards"]
                scores = {item["turn_id"]: item["components"][0]["value"] for item in panel["assessments"]}
                erroneous_ids = set(panel["erroneous_turn_ids"])
                turn_results = []
                for expected in _expectations(case):
                    score = scores[expected["turn_id"]]
                    erroneous = expected["turn_id"] in erroneous_ids
                    turn_results.append(
                        {
                            "turn_id": expected["turn_id"],
                            "quality": score,
                            "erroneous": erroneous,
                            "passed": (
                                erroneous == expected["erroneous"]
                                and expected["quality_range"][0] <= score <= expected["quality_range"][1]
                            ),
                        }
                    )
                passed = all(item["passed"] for item in turn_results)
                usage_rows = [
                    row["usage"]
                    for row in trace.info.get("judge", [])
                    if isinstance(row, dict) and isinstance(row.get("usage"), dict)
                ]
                result = {
                    "case": case["id"],
                    "repetition": repetition,
                    "turns": turn_results,
                    "passed": passed,
                    "usage": {
                        key: sum(row.get(key) or 0 for row in usage_rows)
                        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
                    },
                }
            except Exception as error:
                result = {
                    "case": case["id"],
                    "repetition": repetition,
                    "passed": False,
                    "error_type": type(error).__name__,
                }
            results.append(result)
            with (output / "assessments.jsonl").open("a") as stream:
                stream.write(json.dumps({"result": result, "trace": trace.to_record()}) + "\n")
            print(json.dumps(result), flush=True)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "passed": all(row["passed"] for row in results),
                "results": results,
                "human_review": "pending",
                "scorer_digest": judge.scorer_digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.selection, args.output, fixture=args.fixture))
