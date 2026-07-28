"""Tests for the pinned private Halcyon GraphQL supervised sources."""

from __future__ import annotations

import os

import pytest
from posttrain_lab.data import HalcyonGraphQLSupervisedSource, halcyon_graphql


def _row(split: str, index: int) -> dict[str, object]:
    return {
        "id": f"halcyon/{split}/{index:06d}",
        "messages": [
            {"role": "system", "content": "Emit exactly one read-only execute_graphql call."},
            {"role": "user", "content": "Find the matching entity."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {
                            "name": "execute_graphql",
                            "arguments": {
                                "query": "query Lookup($filter: Filter!) { entities(filter: $filter) { id id } }",
                                "variables": {"filter": {"active": False, "names": ["A", "B"]}},
                                "compact": True,
                            },
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_graphql",
                    "description": "Execute one read-only query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "variables": {"type": "object"},
                            "compact": {"type": "boolean"},
                        },
                        "required": ["query", "variables", "compact"],
                    },
                },
            }
        ],
        "trainable_message_indices": [2],
    }


@pytest.mark.parametrize(("split", "count"), (("train", 392), ("test", 31)))
def test_source_normalizes_and_validates_pinned_split(monkeypatch, split: str, count: int) -> None:
    rows = [_row(split, index) for index in range(count)]
    monkeypatch.setattr(halcyon_graphql, "_rows", lambda selected: rows)

    source = HalcyonGraphQLSupervisedSource(split)  # type: ignore[arg-type]
    dataset = source.load()

    assert dataset.descriptor == source.descriptor
    assert dataset.schema_version == 2
    assert len(dataset.examples) == count
    assert dataset.examples[0].trainable_message_indices == (2,)
    assert dataset.examples[0].messages[2]["tool_calls"]
    assert dataset.metadata["repository"] == "carbonteq/halcyon-graphql-sft"


def test_source_rejects_wrong_population_size(monkeypatch) -> None:
    monkeypatch.setattr(halcyon_graphql, "_rows", lambda split: [_row(split, 0)])

    with pytest.raises(ValueError, match="expected 392 examples"):
        HalcyonGraphQLSupervisedSource("train").load()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda row: row["messages"].pop(), "target index is outside"),  # type: ignore[union-attr]
        (lambda row: row.update(trainable_message_indices=[1]), "supervise only message index 2"),
        (lambda row: row["tools"].clear(), "exactly one execute_graphql tool"),  # type: ignore[union-attr]
        (
            lambda row: row["messages"][2]["tool_calls"][0]["function"].update(name="other"),  # type: ignore[index]
            "assistant must call execute_graphql",
        ),
    ),
)
def test_source_rejects_protocol_drift(monkeypatch, mutation, message: str) -> None:
    rows = [_row("test", index) for index in range(31)]
    mutation(rows[0])
    monkeypatch.setattr(halcyon_graphql, "_rows", lambda split: rows)

    with pytest.raises(ValueError, match=message):
        HalcyonGraphQLSupervisedSource("test").load()


@pytest.mark.network
def test_private_pinned_splits_materialize_and_are_disjoint() -> None:
    if not os.getenv("HF_TOKEN"):
        pytest.skip("private Halcyon GraphQL integration requires HF_TOKEN")

    train = HalcyonGraphQLSupervisedSource("train").load()
    validation = HalcyonGraphQLSupervisedSource("test").load()

    assert len(train.examples) == 392
    assert len(validation.examples) == 31
    assert {example.id for example in train.examples}.isdisjoint(example.id for example in validation.examples)
