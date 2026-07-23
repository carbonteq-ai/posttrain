"""Explicit NeMo import/export adapters over the canonical records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from posttrain.common import JsonValue

from ..models import PreferenceDataset, SupervisedDataset
from .huggingface import preferences_from_huggingface, supervised_from_huggingface


def supervised_from_nemo(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    revision: str,
    metadata: Mapping[str, JsonValue] | None = None,
) -> SupervisedDataset:
    return supervised_from_huggingface(
        rows,
        dataset_id=dataset_id,
        revision=revision,
        format="messages",
        metadata=metadata,
    )


def preferences_from_nemo(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    revision: str,
    metadata: Mapping[str, JsonValue] | None = None,
) -> PreferenceDataset:
    return preferences_from_huggingface(
        rows,
        dataset_id=dataset_id,
        revision=revision,
        format="nemo-ranked",
        metadata=metadata,
    )


def to_nemo_sft_rows(dataset: SupervisedDataset) -> list[dict[str, Any]]:
    return [
        {
            "messages": example.message_records(),
            "tools": example.tool_records(),
        }
        for example in dataset.examples
    ]


def to_nemo_preference_rows(dataset: PreferenceDataset, *, task_name: str) -> list[dict[str, Any]]:
    if not task_name.strip():
        raise ValueError("NeMo preference export requires a task name")
    return [
        {
            "context": example.prompt_records(),
            "completions": [
                {"rank": 0, "completion": example.chosen_records()},
                {"rank": 1, "completion": example.rejected_records()},
            ],
            "task_name": task_name,
        }
        for example in dataset.examples
    ]


__all__ = [
    "preferences_from_nemo",
    "supervised_from_nemo",
    "to_nemo_preference_rows",
    "to_nemo_sft_rows",
]
