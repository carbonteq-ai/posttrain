"""Contract tests for shared Verifiers trace-fact projection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from posttrain.common import SignalSource
from posttrain.environment import (
    ThinkingTokenContext,
    ThinkingTokenResult,
    project_verifiers_trace_facts,
)


def test_provider_usage_and_structured_calls_are_projected_without_model_rules() -> None:
    record = {
        "id": "trace-1",
        "version": 2,
        "run": {"type": "train", "id": "run-1", "step": 4},
        "agent": {"model": "models/future-2b"},
        "task": {"type": "ToolTask", "data": {}},
        "is_completed": True,
        "rewards": {"correct": 1.0},
        "calls": [
            {
                "node": 1,
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "reasoning_tokens": 2},
                "time": {"start": 1.0, "end": 1.2},
            },
            {
                "node": 3,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 18, "completion_tokens": 4, "reasoning_tokens": 1},
                "time": {"start": 2.0, "end": 2.3},
            },
        ],
        "nodes": [
            {"message": {"role": "user", "content": "q"}, "sampled": False},
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call-1", "name": "search", "arguments": "{}"}],
                },
                "sampled": True,
            },
            {"message": {"role": "tool", "content": "result"}, "sampled": False},
            {"message": {"role": "assistant", "content": "answer"}, "sampled": True},
        ],
    }

    facts = project_verifiers_trace_facts(record)

    assert facts.dimensions["rollout_step"] == 4
    assert facts.dimensions["is_truncated"] is False
    assert facts.measures["model_output_tokens"] == 10
    assert facts.measures["thinking_tokens"] == 3
    assert facts.measures["tool_calls"] == 1
    assert facts.measures["model_calls"] == 2
    assert facts.measures["trace_latency_ms"] == pytest.approx(500)
    assert facts.measures["task_reward"] == 1.0
    assert facts.reward_components[0].name == "correct"
    assert facts.reward_components[0].contribution == 1.0
    assert facts.provenance["thinking_tokens"] == "provider_reasoning_usage"


def test_multi_reward_components_remain_distinct_from_the_scalar_reward() -> None:
    facts = project_verifiers_trace_facts(
        {
            "id": "multi-reward",
            "version": 2,
            "rewards": {
                "correctness": 0.8,
                "format bonus": {"score": 1.0, "weight": 0.2},
                "broken": {"score": "unknown", "weight": 2.0},
            },
            "nodes": [],
            "calls": [],
        }
    )

    assert facts.measures["task_reward"] == pytest.approx(1.0)
    components = {item.name: item for item in facts.reward_components}
    assert components["correctness"].contribution == 0.8
    assert components["format bonus"].contribution == 0.2
    assert components["format bonus"].score == 1.0
    assert components["format bonus"].weight == 0.2
    assert "broken" not in components


def test_reward_component_sources_are_declared_not_inferred_from_native_judges() -> None:
    facts = project_verifiers_trace_facts(
        {
            "id": "sources",
            "rewards": {"rubric": 0.7, "format": 0.2},
            "info": {"judge": [{"model": "not-a-fact"}]},
            "nodes": [],
            "calls": [],
        },
        reward_component_sources={
            "rubric": SignalSource("llm_judge", "reference-rubric"),
            "format": SignalSource("deterministic", "format-check"),
        },
    )

    components = {item.name: item for item in facts.reward_components}
    assert components["rubric"].source == SignalSource("llm_judge", "reference-rubric")
    assert components["format"].source == SignalSource("deterministic", "format-check")
    assert all("judge" not in key for key in facts.measures)


def test_visible_and_reasoning_usage_normalize_to_total_model_output() -> None:
    facts = project_verifiers_trace_facts(
        {
            "id": "split-usage",
            "calls": [
                {
                    "usage": {"completion_tokens": 3, "reasoning_tokens": 5},
                    "finish_reason": "stop",
                }
            ],
            "nodes": [],
        }
    )

    assert facts.measures["model_output_tokens"] == 8
    assert facts.measures["thinking_tokens"] == 5
    assert facts.provenance["model_output_tokens"] == "provider_visible_plus_reasoning"


def test_qwen_rule_recovers_complete_and_proven_truncated_thinking() -> None:
    complete = {
        "id": "complete",
        "version": 2,
        "agent": {"model": "models/qwen3.5-2b@bf16"},
        "calls": [{"node": 0, "finish_reason": "stop", "usage": {"completion_tokens": 6}}],
        "nodes": [
            {
                "message": {"role": "assistant", "reasoning_content": "think", "content": "answer"},
                "sampled": True,
                "token_ids": [11, 12, 13, 248069, 14, 15],
                "mask": [True, True, True, True, True, True],
            }
        ],
    }
    truncated = {
        "id": "truncated",
        "version": 2,
        "agent": {"model": "models/qwen3.5-2b@bf16"},
        "calls": [{"node": 0, "finish_reason": "length", "usage": {"completion_tokens": 3}}],
        "nodes": [
            {
                "message": {"role": "assistant", "reasoning_content": "unfinished", "content": None},
                "sampled": True,
                "token_ids": [21, 22, 23],
                "mask": [True, True, True],
            }
        ],
    }

    complete_facts = project_verifiers_trace_facts(complete)
    truncated_facts = project_verifiers_trace_facts(truncated)

    assert complete_facts.measures["thinking_tokens"] == 3
    assert truncated_facts.dimensions["is_truncated"] is True
    assert truncated_facts.measures["thinking_tokens"] == 3
    assert truncated_facts.provenance["thinking_tokens"] == "qwen3.5-native-thinking.v1"


def test_ambiguous_unterminated_thinking_remains_missing() -> None:
    record = {
        "id": "ambiguous",
        "version": 2,
        "agent": {"model": "models/qwen3.5-2b@bf16"},
        "calls": [{"finish_reason": "length", "usage": {"completion_tokens": 3}}],
        "nodes": [
            {
                "message": {"role": "assistant", "reasoning_content": "unfinished", "content": None},
                "sampled": True,
                "token_ids": [21, 22, 23],
                "mask": [True, True, True],
            }
        ],
    }

    facts = project_verifiers_trace_facts(record)

    assert facts.dimensions["is_truncated"] is True
    assert facts.measures["model_output_tokens"] == 3
    assert facts.measures["thinking_tokens"] is None
    assert facts.provenance["thinking_tokens"] == "unsupported"


def test_evaluation_step_is_not_promoted_to_rollout_step() -> None:
    facts = project_verifiers_trace_facts(
        {
            "id": "eval",
            "version": 2,
            "run": {"type": "eval", "id": "run-1", "step": 9},
            "nodes": [],
            "calls": [],
        }
    )

    assert facts.dimensions["rollout_step"] is None


@dataclass(frozen=True)
class FutureModelRule:
    id: str = "future-model-thinking.v1"

    def matches(self, context: ThinkingTokenContext) -> bool:
        return context.model_family == "future-model" and context.tokenizer_revision == "tokenizer-sha-1"

    def calculate(self, context: ThinkingTokenContext) -> ThinkingTokenResult | None:
        del context
        return ThinkingTokenResult(tokens=7, method=self.id)


def test_new_model_family_is_added_as_a_versioned_rule_with_immutable_identity() -> None:
    facts = project_verifiers_trace_facts(
        {
            "id": "future",
            "version": 3,
            "agent": {"model": "models/future-2b"},
            "nodes": [],
            "calls": [{"finish_reason": "stop", "usage": {"completion_tokens": 11}}],
        },
        attributes={"model_family": "future-model", "tokenizer_revision": "tokenizer-sha-1"},
        thinking_rules=(FutureModelRule(),),
    )

    assert facts.measures["thinking_tokens"] == 7
    assert facts.provenance["thinking_tokens"] == "future-model-thinking.v1"
    assert facts.dimensions["tokenizer_revision"] == "tokenizer-sha-1"


def test_incompatible_thinking_fallback_is_not_persisted_as_more_than_total_output() -> None:
    facts = project_verifiers_trace_facts(
        {
            "id": "incompatible-thinking",
            "version": 3,
            "agent": {"model": "models/future-2b"},
            "nodes": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "sampled": True,
                    "token_ids": [1, 2, 3],
                    "mask": [True, True, True],
                }
            ],
            "calls": [{"finish_reason": "stop", "usage": {"completion_tokens": 3}}],
        },
        attributes={"model_family": "future-model", "tokenizer_revision": "tokenizer-sha-1"},
        thinking_rules=(FutureModelRule(),),
    )

    assert facts.measures["model_output_tokens"] == 3
    assert facts.measures["thinking_tokens"] is None
    assert facts.provenance["thinking_tokens"] == "incompatible_output_accounting"
    assert facts.state == "partial"
