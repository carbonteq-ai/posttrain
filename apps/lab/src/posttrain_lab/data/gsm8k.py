"""Pinned GSM8K demonstrations and trace-derived preference pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from posttrain.train import (
    PreferenceDataset,
    PreferenceExample,
    SupervisedDataset,
    SupervisedExample,
)

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


@dataclass(frozen=True, slots=True)
class RejectedRollout:
    """A completed, scored rollout selected as the rejected DPO response."""

    example_id: str
    response: str
    score: float
    trace_id: str

    def __post_init__(self) -> None:
        if not self.response.strip() or not self.trace_id.strip():
            raise ValueError("rejected rollouts require response text and trace lineage")
        if self.score >= 1.0:
            raise ValueError("a rejected rollout must score below the GSM8K reference")


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


def load_gsm8k_supervised(*, count: int, offset: int = 0) -> SupervisedDataset:
    """Load deterministic train demonstrations using the eval environment's prompt contract."""

    if count < 1 or offset < 0:
        raise ValueError("count must be positive and offset cannot be negative")
    rows = _rows("train")
    end = offset + count
    if end > len(rows):
        raise ValueError(f"requested rows [{offset}:{end}] exceed the GSM8K train split")
    system_prompt = _system_prompt()
    examples = tuple(
        SupervisedExample(
            id=f"train/{index:06d}",
            prompt=str(rows[index]["question"]),
            response=str(rows[index]["answer"]),
            system_prompt=system_prompt,
        )
        for index in range(offset, end)
    )
    return SupervisedDataset(
        id=f"gsm8k/train-{offset}-{end}-v1",
        revision=GSM8K_REVISION,
        examples=examples,
    )


def preferences_from_rollouts(
    demonstrations: SupervisedDataset,
    rejected_rollouts: tuple[RejectedRollout, ...],
) -> PreferenceDataset:
    """Join gold demonstrations with failed, queryable rollout traces by stable example ID."""

    rejected = {rollout.example_id: rollout for rollout in rejected_rollouts}
    if len(rejected) != len(rejected_rollouts):
        raise ValueError("rejected rollout example ids must be unique")
    examples: list[PreferenceExample] = []
    for demonstration in demonstrations.examples:
        rollout = rejected.get(demonstration.id)
        if rollout is None:
            continue
        examples.append(
            PreferenceExample(
                id=demonstration.id,
                prompt=demonstration.prompt,
                chosen=demonstration.response,
                rejected=rollout.response,
                chosen_score=1.0,
                rejected_score=rollout.score,
                system_prompt=demonstration.system_prompt,
                rejected_trace_id=rollout.trace_id,
            )
        )
    if not examples:
        raise ValueError("no rejected rollouts matched the supervised demonstrations")
    return PreferenceDataset(
        id=f"{demonstrations.id}/trace-preferences",
        revision=demonstrations.revision,
        examples=tuple(examples),
    )
