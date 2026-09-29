"""Episode-ending labels derived from native Verifiers records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from posttrain.common import EPISODE_ENDINGS, TRUNCATED_EPISODE_ENDINGS, episode_ending
from posttrain.environment import (
    project_verifiers_trace_facts,
    verifiers_episode_ending,
    verifiers_trace_attributes,
    verifiers_trace_has_error,
    verifiers_trace_is_truncated,
)

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "automationbench_episode_endings.json").read_text(encoding="utf-8")
)
_CASES: list[dict[str, Any]] = _FIXTURE["cases"]


@pytest.mark.parametrize("case", _CASES, ids=[case["record"]["id"][:8] for case in _CASES])
def test_real_automationbench_episodes_get_their_ending(case: dict[str, Any]) -> None:
    record = case["record"]
    expected = case["expected_ending"]
    assert verifiers_episode_ending(record) == expected
    # Knowing the served context length never changes these labels.
    assert verifiers_episode_ending(record, max_model_len=_FIXTURE["max_model_len"]) == expected
    attributes = verifiers_trace_attributes(record)
    assert attributes["episode_ending"] == expected
    # Truncation semantics are unchanged: every ending but completed is truncated,
    # none of these is an error, and every one keeps its environment reward.
    assert attributes["is_truncated"] is (expected != "completed")
    assert attributes["has_error"] is False
    facts = project_verifiers_trace_facts(record)
    assert facts.measures["task_reward"] is not None
    # The ending is also a fact dimension, so Trackio stores it in its own column.
    assert facts.dimensions["episode_ending"] == expected
    assert facts.calculator_version == "verifiers-trace-facts.v9"


def test_context_rejection_keeps_the_scored_reward() -> None:
    record = next(case["record"] for case in _CASES if case["expected_ending"] == "context_rejected")
    assert record["calls"][-1]["error"]["message"].startswith("Prompt length (34021) exceeds maximum context length")
    assert record["stop_condition"] == "agent_completed"
    assert project_verifiers_trace_facts(record).measures["task_reward"] == pytest.approx(-0.02)


def _call(
    *,
    finish: str | None = "stop",
    prompt: int | None = 100,
    completion: int | None = 50,
    max_tokens: int | None = 4096,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usage = (
        None if prompt is None and completion is None else {"prompt_tokens": prompt, "completion_tokens": completion}
    )
    return {
        "finish_reason": finish,
        "usage": usage,
        "sampling": {"max_tokens": max_tokens},
        "error": error,
    }


def _record(*calls: dict[str, Any], stop: str = "agent_completed", **fields: Any) -> dict[str, Any]:
    return {"id": "t", "version": 1, "ok": True, "errors": [], "stop_condition": stop, "calls": list(calls), **fields}


_OVERFLOW = {
    "type": "ProviderError",
    "status_code": 400,
    "message": "Prompt length (9) exceeds maximum context length (8).",
}


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (_record(_call()), "completed"),
        (_record(_call(finish="tool_calls"), stop="max_turns"), "turn_limit"),
        # A limit stop outranks a cut last reply: the agent asked for another turn.
        (_record(_call(finish="length", completion=4096), stop="max_turns"), "turn_limit"),
        (_record(_call(), stop="max_output_tokens"), "token_budget"),
        (_record(_call(), stop="max_input_tokens"), "token_budget"),
        (_record(_call(), stop="max_total_tokens"), "token_budget"),
        (_record(_call(), stop="harness_timeout"), "time_limit"),
        (_record(_call(), stop="context_length"), "context_rejected"),
        (_record(_call(), _call(finish=None, prompt=None, completion=None, error=_OVERFLOW)), "context_rejected"),
        # vLLM's other wording of the same refusal.
        (
            _record(
                _call(),
                _call(error={"type": "ProviderError", "status_code": 400, "message": "maximum context length is 8"}),
            ),
            "context_rejected",
        ),
        (_record(_call(finish="length", completion=4096)), "reply_token_limit"),
        (_record(_call(finish="length", completion=4000)), "context_limit_reply_cut"),
        # Without usage the cause cannot be told; a length finish is the per-call limit.
        (_record(_call(finish="length", prompt=None, completion=None)), "reply_token_limit"),
        (_record(_call(finish="length", completion=4000, max_tokens=None)), "reply_token_limit"),
        # A later error after a context refusal, or a refusal followed by more calls, is an error.
        (_record(_call(error=_OVERFLOW), _call()), "error"),
        (_record(_call(error={"type": "ProviderError", "status_code": 500, "message": "boom"})), "error"),
        (_record(_call(), ok=False), "error"),
        (_record(_call(), errors=[{"type": "ToolError"}]), "error"),
        ({"id": "legacy", "version": 1}, "completed"),
    ],
)
def test_ending_rules(record: dict[str, Any], expected: str) -> None:
    assert verifiers_episode_ending(record) == expected
    assert verifiers_trace_is_truncated(record) is (expected in TRUNCATED_EPISODE_ENDINGS)
    assert verifiers_trace_has_error(record) is (expected == "error")


def test_known_context_length_confirms_a_context_cut_even_at_max_tokens() -> None:
    # A client that clamps max_tokens to the room left records the clamped value.
    record = _record(_call(finish="length", prompt=4000, completion=96, max_tokens=96))
    assert verifiers_episode_ending(record) == "reply_token_limit"
    assert verifiers_episode_ending(record, max_model_len=4096) == "context_limit_reply_cut"


def test_labels_are_one_closed_vocabulary() -> None:
    assert set(TRUNCATED_EPISODE_ENDINGS) == set(EPISODE_ENDINGS) - {"completed", "error"}
    assert all(episode_ending(name) == name for name in EPISODE_ENDINGS)
    assert episode_ending("truncated") is None
    assert episode_ending(None) is None
