"""Pinned, bounded Smol-SmolTalk source for general conversational SFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from posttrain.data import DatasetDescriptor, SupervisedDataset, supervised_from_huggingface

SMOL_SMOLTALK_REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
MAX_PREFIX_CHARACTERS = 1_000
MAX_TOTAL_MESSAGE_CHARACTERS = 1_600


def _eligible(raw: dict[str, Any]) -> bool:
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return False
    prefix_characters = 0
    total_characters = sum(len(str(message.get("content") or "")) for message in messages if isinstance(message, dict))
    if total_characters > MAX_TOTAL_MESSAGE_CHARACTERS:
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        if message.get("role") == "assistant":
            return prefix_characters <= MAX_PREFIX_CHARACTERS
        prefix_characters += len(str(message.get("content") or ""))
    return False


def _rows(offset: int, count: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install posttrain-lab with the gpu-posttrain extra") from error
    stream = load_dataset(
        "HuggingFaceTB/smol-smoltalk",
        split="train",
        revision=SMOL_SMOLTALK_REVISION,
        streaming=True,
    )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(stream.skip(offset), start=offset):
        if not _eligible(raw):
            continue
        row = dict(raw)
        row["id"] = f"smol-smoltalk/train/{index:06d}"
        row["metadata"] = {
            "source_row": index,
            "source": str(row.get("source") or "unknown"),
        }
        rows.append(row)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f"requested {count} Smol-SmolTalk rows at offset {offset}, received {len(rows)}")
    return rows


@dataclass(frozen=True, slots=True)
class SmolSmolTalkSupervisedSource:
    """Lazy immutable window over the publisher's pinned training split."""

    count: int
    offset: int = 0

    def __post_init__(self) -> None:
        if self.count < 1 or self.offset < 0:
            raise ValueError("count must be positive and offset cannot be negative")

    @property
    def descriptor(self) -> DatasetDescriptor:
        end = self.offset + self.count
        return DatasetDescriptor(
            id=f"smol-smoltalk/train-prefix-filtered-{self.offset}-{end}-v1",
            revision=SMOL_SMOLTALK_REVISION,
            kind="supervised",
            metadata={
                "source": "huggingface",
                "repository": "HuggingFaceTB/smol-smoltalk",
                "split": "train",
                "offset": self.offset,
                "count": self.count,
                "license": "apache-2.0",
                "selection_filter": "bounded-conversation-characters-v1",
                "max_prefix_characters": MAX_PREFIX_CHARACTERS,
                "max_total_message_characters": MAX_TOTAL_MESSAGE_CHARACTERS,
            },
            num_examples=self.count,
        )

    @property
    def id(self) -> str:
        return self.descriptor.id

    @property
    def revision(self) -> str:
        return self.descriptor.revision

    def load(self) -> SupervisedDataset:
        descriptor = self.descriptor
        return supervised_from_huggingface(
            _rows(self.offset, self.count),
            dataset_id=descriptor.id,
            revision=descriptor.revision,
            format="messages",
            metadata=descriptor.metadata,
        )


__all__ = [
    "MAX_PREFIX_CHARACTERS",
    "MAX_TOTAL_MESSAGE_CHARACTERS",
    "SMOL_SMOLTALK_REVISION",
    "SmolSmolTalkSupervisedSource",
]
