"""Pinned GSM8K source producing canonical trainer-neutral conversations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from posttrain.data import DatasetDescriptor, SupervisedDataset, SupervisedExample

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def _rows(split: str) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install posttrain-lab with the gpu-posttrain extra") from error
    return load_dataset(
        "openai/gsm8k",
        "main",
        split=split,
        revision=GSM8K_REVISION,
    )


def _system_prompt() -> str:
    try:
        from gsm8k_v1.taskset import SYSTEM
    except ImportError as error:
        raise RuntimeError("the gsm8k-v1 Verifiers environment is required") from error
    return SYSTEM


@dataclass(frozen=True, slots=True)
class GSM8KSupervisedSource:
    """Lazy deterministic view over pinned GSM8K demonstrations."""

    count: int
    offset: int = 0

    def __post_init__(self) -> None:
        if self.count < 1 or self.offset < 0:
            raise ValueError("count must be positive and offset cannot be negative")

    @property
    def descriptor(self) -> DatasetDescriptor:
        end = self.offset + self.count
        return DatasetDescriptor(
            id=f"gsm8k/train-{self.offset}-{end}-v1",
            revision=GSM8K_REVISION,
            kind="supervised",
            metadata={
                "source": "huggingface",
                "repository": "openai/gsm8k",
                "split": "train",
                "offset": self.offset,
                "count": self.count,
                "prompt_contract": "gsm8k-v1",
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
        rows = _rows("train")
        end = self.offset + self.count
        if end > len(rows):
            raise ValueError(f"requested rows [{self.offset}:{end}] exceed the GSM8K train split")
        system_prompt = _system_prompt()
        examples = tuple(
            SupervisedExample(
                id=f"train/{index:06d}",
                messages=(
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(rows[index]["question"])},
                    {"role": "assistant", "content": str(rows[index]["answer"])},
                ),
                trainable_message_indices=(2,),
                metadata={"source_row": index},
            )
            for index in range(self.offset, end)
        )
        descriptor = self.descriptor
        return SupervisedDataset(
            descriptor.id,
            descriptor.revision,
            examples,
            metadata=descriptor.metadata,
            schema_version=descriptor.schema_version,
        )


__all__ = ["GSM8K_REVISION", "GSM8KSupervisedSource"]
