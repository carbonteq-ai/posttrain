"""Evaluation provider failures remain visible without exposing provider messages."""

from posttrain.tracking import TraceRecord
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
