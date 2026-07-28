"""Pinned private Halcyon GraphQL sources for the lab SFT canary."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from posttrain.data import DatasetDescriptor, SupervisedDataset, supervised_from_huggingface

HALCYON_GRAPHQL_REPOSITORY = "carbonteq/halcyon-graphql-sft"
HALCYON_GRAPHQL_REVISION = "a69e1c0c6ebb1f565be91cd0b6d95bd2b0e9110c"
_EXPECTED_EXAMPLES = {"train": 392, "test": 31}
_DATASET_IDS = {
    "train": "halcyon-graphql-stage1/train-v2",
    "test": "halcyon-graphql-stage1/validation-v2",
}


def _rows(split: Literal["train", "test"]) -> Any:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Halcyon GraphQL dataset access requires HF_TOKEN")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install posttrain-lab with the gpu-train extra") from error
    return load_dataset(
        HALCYON_GRAPHQL_REPOSITORY,
        split=split,
        revision=HALCYON_GRAPHQL_REVISION,
        token=token,
    )


def _function_name(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    function = value.get("function")
    return cast(str | None, function.get("name")) if isinstance(function, Mapping) else None


def _validate_example_shape(dataset: SupervisedDataset) -> None:
    for example in dataset.examples:
        if len(example.messages) != 3:
            raise ValueError(f"Halcyon GraphQL example {example.id!r} must have exactly three messages")
        roles = tuple(message.get("role") for message in example.messages)
        if roles != ("system", "user", "assistant"):
            raise ValueError(f"Halcyon GraphQL example {example.id!r} has invalid message roles: {roles!r}")
        if example.trainable_message_indices != (2,):
            raise ValueError(f"Halcyon GraphQL example {example.id!r} must supervise only message index 2")
        if len(example.tools) != 1 or _function_name(example.tools[0]) != "execute_graphql":
            raise ValueError(f"Halcyon GraphQL example {example.id!r} must define exactly one execute_graphql tool")
        raw_calls = example.messages[2].get("tool_calls")
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)) or len(raw_calls) != 1:
            raise ValueError(f"Halcyon GraphQL example {example.id!r} must contain exactly one assistant tool call")
        if _function_name(raw_calls[0]) != "execute_graphql":
            raise ValueError(f"Halcyon GraphQL example {example.id!r} assistant must call execute_graphql")


@dataclass(frozen=True, slots=True)
class HalcyonGraphQLSupervisedSource:
    """Lazy immutable view over one pinned private corpus split."""

    split: Literal["train", "test"]

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            id=_DATASET_IDS[self.split],
            revision=HALCYON_GRAPHQL_REVISION,
            kind="supervised",
            schema_version=2,
            metadata={
                "source": "huggingface",
                "repository": HALCYON_GRAPHQL_REPOSITORY,
                "split": self.split,
                "corpus_format": "posttrain_graphql_query_v1",
                "corpus_version": 2,
                "expected_examples": _EXPECTED_EXAMPLES[self.split],
            },
            num_examples=_EXPECTED_EXAMPLES[self.split],
        )

    @property
    def id(self) -> str:
        return self.descriptor.id

    @property
    def revision(self) -> str:
        return self.descriptor.revision

    def load(self) -> SupervisedDataset:
        descriptor = self.descriptor
        normalized = supervised_from_huggingface(
            _rows(self.split),
            dataset_id=descriptor.id,
            revision=descriptor.revision,
            format="messages",
            metadata=descriptor.metadata,
        )
        if len(normalized.examples) != descriptor.num_examples:
            raise ValueError(
                f"Halcyon GraphQL {self.split!r} split expected {descriptor.num_examples} examples, "
                f"received {len(normalized.examples)}"
            )
        dataset = SupervisedDataset(
            normalized.id,
            normalized.revision,
            normalized.examples,
            metadata=normalized.metadata,
            schema_version=descriptor.schema_version,
        )
        _validate_example_shape(dataset)
        return dataset


__all__ = [
    "HALCYON_GRAPHQL_REPOSITORY",
    "HALCYON_GRAPHQL_REVISION",
    "HalcyonGraphQLSupervisedSource",
]
