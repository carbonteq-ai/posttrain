"""Tests for canonical dataset adapters."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from posttrain.data import (
    PreferenceDataset,
    PreferenceExample,
    TraceSelection,
    preferences_from_huggingface,
    preferences_from_nemo,
    supervised_from_huggingface,
    supervised_from_verifiers,
    to_huggingface_sft_rows,
    to_nemo_preference_rows,
)


@pytest.mark.parametrize(
    ("row", "source_format"),
    (
        (
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
            "messages",
        ),
        ({"prompt": "Q", "completion": "A"}, "prompt-completion"),
        ({"instruction": "Q", "input": "context", "output": "A"}, "alpaca"),
        (
            {"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]},
            "sharegpt",
        ),
    ),
)
def test_huggingface_sft_formats_normalize_to_one_contract(row, source_format) -> None:
    dataset = supervised_from_huggingface([row], dataset_id="examples/sft", revision="revision")

    example = dataset.examples[0]
    assert example.messages[-1]["role"] == "assistant"
    assert example.trainable_message_indices == (len(example.messages) - 1,)
    assert example.metadata["source_format"] == source_format


def test_huggingface_tool_definitions_survive_normalization() -> None:
    tool = {
        "type": "function",
        "function": {"name": "search", "parameters": {"type": "object"}},
    }
    dataset = supervised_from_huggingface(
        [{"messages": [{"role": "assistant", "content": "done"}], "tools": [tool]}],
        dataset_id="examples/tools",
        revision="revision",
    )

    assert dataset.examples[0].tool_records() == [tool]


def test_canonical_huggingface_export_round_trips_explicit_targets() -> None:
    source = supervised_from_huggingface(
        [
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "draft"},
                    {"role": "user", "content": "revise"},
                    {"role": "assistant", "content": "final"},
                ],
                "trainable_message_indices": [3],
            }
        ],
        dataset_id="examples/targets",
        revision="revision",
    )

    restored = supervised_from_huggingface(
        to_huggingface_sft_rows(source),
        dataset_id="examples/restored",
        revision="revision",
    )
    assert restored.examples[0].id == source.examples[0].id
    assert restored.examples[0].trainable_message_indices == (3,)


def test_trl_and_nemo_preferences_normalize_to_same_contract() -> None:
    trl = preferences_from_huggingface(
        [{"prompt": "Q", "chosen": "A", "rejected": "B"}],
        dataset_id="examples/trl",
        revision="revision",
    )
    nemo = preferences_from_nemo(
        [
            {
                "context": [{"role": "user", "content": "Q"}],
                "completions": [
                    {"rank": 1, "completion": [{"role": "assistant", "content": "B"}]},
                    {"rank": 0, "completion": [{"role": "assistant", "content": "A"}]},
                ],
            }
        ],
        dataset_id="examples/nemo",
        revision="revision",
    )

    assert trl.examples[0].prompt == nemo.examples[0].prompt
    assert trl.examples[0].chosen == nemo.examples[0].chosen
    assert trl.examples[0].rejected == nemo.examples[0].rejected
    assert nemo.examples[0].chosen_score == 0.0
    assert nemo.examples[0].rejected_score == -1.0


def test_nemo_preference_export_is_ranked_and_named() -> None:
    dataset = PreferenceDataset(
        "examples/preferences",
        "revision",
        (
            PreferenceExample(
                "rows/000000",
                ({"role": "user", "content": "Q"},),
                ({"role": "assistant", "content": "A"},),
                ({"role": "assistant", "content": "B"},),
            ),
        ),
    )

    row = to_nemo_preference_rows(dataset, task_name="math")[0]
    assert row["task_name"] == "math"
    assert [completion["rank"] for completion in row["completions"]] == [0, 1]


class _Value:
    def __init__(self, **values) -> None:
        self.values = values

    def model_dump(self, **_) -> dict:
        return dict(self.values)


def test_verifiers_trace_projects_sampled_branches_and_lineage() -> None:
    nodes = [
        SimpleNamespace(message=_Value(role="user", content="Q"), sampled=False),
        SimpleNamespace(message=_Value(role="assistant", content="A"), sampled=True),
    ]
    trace = SimpleNamespace(
        id="ABC123",
        stop_condition="done",
        has_error=False,
        is_truncated=False,
        reward=1.0,
        tools=[_Value(name="search", description="Search", parameters={"type": "object"})],
        branches=[SimpleNamespace(index=0, nodes=nodes)],
    )

    dataset = supervised_from_verifiers(
        [trace],
        dataset_id="examples/verifiers",
        revision="revision",
        selection=TraceSelection(min_reward=1.0),
    )

    example = dataset.examples[0]
    assert example.id == "traces/abc123/branches/0"
    assert example.trainable_message_indices == (1,)
    assert example.metadata["trace_id"] == "ABC123"
    assert example.tools[0]["type"] == "function"


def test_verifiers_trace_selection_drops_failed_generation() -> None:
    trace = SimpleNamespace(
        id="failed",
        stop_condition="error",
        has_error=True,
        is_truncated=False,
        reward=0.0,
        tools=[],
        branches=[],
    )

    with pytest.raises(ValueError, match="non-empty"):
        supervised_from_verifiers([trace], dataset_id="examples/empty", revision="revision")
