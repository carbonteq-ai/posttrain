"""Evaluation provider failures remain visible without exposing provider messages."""

from typing import Any

from posttrain.common import JsonValue
from posttrain.tracking import TraceRecord
from posttrain_observatory.models import TraceSummary
from posttrain_observatory.traces import _summary, _wire_error


def test_model_call_error_is_visible_when_episode_errors_are_empty() -> None:
    payload = {
        "ok": True,
        "errors": [],
        "calls": [
            {
                "error": {
                    "type": "ProviderError",
                    "status_code": 400,
                    "message": "context details must stay in the native artifact",
                }
            }
        ],
    }
    assert _wire_error(payload) == "ProviderError (HTTP 400)"
    summary = _summary(
        TraceRecord(
            trace_type="verifiers",
            external_id="provider-error",
            payload={**payload, "rewards": {"reward": 0.0}, "metrics": {"correct": 0.0}},
            attributes={},
        )
    )
    assert summary.error == "ProviderError (HTTP 400)"
    assert summary.reward is None
    assert summary.success is None


def test_successful_model_calls_have_no_error() -> None:
    assert _wire_error({"errors": [], "calls": [{"error": None}]}) is None


# Trimmed from trace 06d0f371052643dcbc7f250d8f8127e5 (run
# lfm26-sampo-cont120-g24x6-lr5e5-kl1e2-t05-20260928-r1, optimizer step 43): the
# final request was refused for exceeding the context, and the environment scored
# the state the episode reached.
_CONTEXT_REJECTED: dict[str, Any] = {
    "id": "06d0f371052643dcbc7f250d8f8127e5",
    "version": 1,
    "ok": True,
    "errors": [],
    "stop_condition": "agent_completed",
    "is_completed": True,
    "rewards": {
        "partial_credit": {"score": 0.0, "weight": 1.0},
        "tool_mistake_penalty": {"score": -0.02, "weight": 1.0},
    },
    "calls": [
        {
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 20312, "completion_tokens": 1409},
            "sampling": {"max_tokens": 4096},
            "error": None,
        },
        {
            "finish_reason": None,
            "usage": None,
            "sampling": {"max_tokens": 4096},
            "error": {
                "type": "ProviderError",
                "status_code": 400,
                "message": "Prompt length (34021) exceeds maximum context length (24576).",
            },
        },
    ],
}
# Trackio's list pages keep a summary payload without model calls.
_LIST_PAYLOAD = {key: value for key, value in _CONTEXT_REJECTED.items() if key != "calls"}


def _pair(attributes: dict[str, JsonValue]) -> tuple[TraceSummary, TraceSummary]:
    detail = _summary(
        TraceRecord(
            trace_type="verifiers", external_id="06d0f371", payload=dict(_CONTEXT_REJECTED), attributes=attributes
        )
    )
    listed = _summary(
        TraceRecord(trace_type="verifiers", external_id="06d0f371", payload=dict(_LIST_PAYLOAD), attributes=attributes)
    )
    return detail, listed


def test_context_rejection_is_the_same_truncated_scored_trace_in_list_and_detail() -> None:
    labelled: dict[str, JsonValue] = {"is_truncated": True, "has_error": False, "episode_ending": "context_rejected"}
    historical: dict[str, JsonValue] = {"is_truncated": True, "has_error": False}
    for attributes in (labelled, historical):
        detail, listed = _pair(attributes)
        for summary in (detail, listed):
            assert summary.error is None
            assert summary.outcome == "truncated"
            assert summary.truncated is True
            assert summary.reward == -0.02
        assert detail.ending == listed.ending == attributes.get("episode_ending")


def test_context_rejection_without_recorded_evidence_is_still_not_an_error() -> None:
    detail = _summary(
        TraceRecord(trace_type="verifiers", external_id="legacy", payload=dict(_CONTEXT_REJECTED), attributes={})
    )
    assert detail.error is None
    assert detail.truncated is True
    assert detail.reward == -0.02
    assert detail.ending is None


def test_recorded_error_is_an_error_even_on_a_list_page_without_calls() -> None:
    attributes: dict[str, JsonValue] = {"is_truncated": False, "has_error": True, "episode_ending": "error"}
    listed = _summary(
        TraceRecord(trace_type="verifiers", external_id="failed", payload=dict(_LIST_PAYLOAD), attributes=attributes)
    )
    assert listed.error == "trace reported an error"
    assert listed.outcome == "error"
    assert listed.reward is None
    assert listed.ending == "error"
