"""Thinking and output token columns come from per-call usage for every model."""

from __future__ import annotations

from posttrain.tracking import TraceRecord
from posttrain_observatory.traces import _summary


def _record(model: str, usage: dict[str, int]) -> TraceRecord:
    return TraceRecord(
        trace_type="verifiers",
        external_id=f"{model}-trace",
        payload={
            "agent": {"model": model},
            "rewards": {"task": 1.0},
            "nodes": [
                {
                    "message": {"role": "assistant", "reasoning_content": "plan", "content": "done"},
                    "sampled": True,
                    "token_ids": [11, 12, 13, 248069, 14, 15],
                    "mask": [True, True, True, True, True, True],
                }
            ],
            "calls": [{"finish_reason": "stop", "error": None, "usage": {"prompt_tokens": 9, **usage}}],
        },
        attributes={"model": model},
    )


def test_renderer_reported_thinking_splits_completion_for_lfm() -> None:
    summary = _summary(_record("models/lfm2.5-2.6b@bf16", {"completion_tokens": 6, "reasoning_tokens": 4}))

    assert summary.thinking_tokens == 4
    assert summary.response_tokens == 2


def test_token_ids_are_not_reinterpreted_by_model_name() -> None:
    summary = _summary(_record("models/qwen3.5-2b@bf16", {"completion_tokens": 6}))

    assert summary.thinking_tokens is None
    assert summary.response_tokens is None
