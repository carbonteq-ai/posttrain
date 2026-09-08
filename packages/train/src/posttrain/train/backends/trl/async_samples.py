"""Strict projection of admitted environment groups into native TRL async samples.

This module is transport only. Reward construction, group admission, and
advantage estimation remain owned by the existing Posttrain algorithm path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...online_rl import BehaviorPolicySpan, EnvironmentRollout


class InvalidAsyncSampleGroup(ValueError):
    """An admitted group cannot be represented faithfully by native async TRL."""


@dataclass(frozen=True, slots=True)
class AsyncRolloutRecord:
    """One admitted rollout plus its message-level native projection."""

    occurrence_id: str
    rollout: EnvironmentRollout
    prompt_messages: tuple[Mapping[str, Any], ...] = ()
    completion_messages: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.occurrence_id:
            raise ValueError("async rollout occurrence id cannot be empty")


def project_async_group(
    records: Sequence[AsyncRolloutRecord],
    advantages: Sequence[float],
    *,
    group_id: int,
    expected_group_size: int,
) -> tuple[Any, ...]:
    """Atomically convert one complete admitted group to native ``RolloutSample`` values.

    The native type is imported at the backend boundary so importing Posttrain's
    neutral rollout contracts does not initialize TRL. No text is tokenized or
    reconstructed here.
    """

    if expected_group_size < 1:
        raise ValueError("expected async group size must be positive")
    if len(records) != expected_group_size:
        raise InvalidAsyncSampleGroup(
            f"async group is incomplete: expected {expected_group_size}, received {len(records)}"
        )
    if len(advantages) != len(records):
        raise InvalidAsyncSampleGroup("one finite advantage is required for every async rollout")
    if len({record.occurrence_id for record in records}) != len(records):
        raise InvalidAsyncSampleGroup("async group occurrence ids must be unique")
    if len({record.rollout.example_id for record in records}) != 1:
        raise InvalidAsyncSampleGroup("one async group cannot contain different tasks")
    if not all(math.isfinite(value) for value in advantages):
        raise InvalidAsyncSampleGroup("async group advantages must be finite")

    prepared: list[tuple[AsyncRolloutRecord, int, list[int], list[int], list[float]]] = []
    group_versions: set[int] = set()
    for record in records:
        rollout = record.rollout
        if rollout.is_truncated:
            raise InvalidAsyncSampleGroup("truncated rollouts are not admitted to async training")
        if len(rollout.sampling_logprobs) != len(rollout.completion_ids):
            raise InvalidAsyncSampleGroup("async training requires a behavior logprob for every completion token")
        if rollout.behavior_policy is None:
            raise InvalidAsyncSampleGroup("async training requires an episode behavior policy span")

        # The native learner uses this value only as a conservative freshness
        # bound. Exact behavior probabilities remain token-aligned below.
        policy_version = rollout.behavior_policy.start
        group_versions.add(policy_version)

        input_ids = [*rollout.prompt_ids, *rollout.completion_ids]
        completion_mask = [0] * len(rollout.prompt_ids) + [int(value) for value in rollout.env_mask]
        old_log_probs = [0.0] * len(rollout.prompt_ids) + [
            logprob if sampled else 0.0
            for logprob, sampled in zip(rollout.sampling_logprobs, rollout.env_mask, strict=True)
        ]
        prepared.append((record, policy_version, input_ids, completion_mask, old_log_probs))

    if len(group_versions) != 1:
        raise InvalidAsyncSampleGroup("all siblings in an async group must use the same behavior policy version")

    try:
        from trl.experimental.async_grpo.async_rollout_worker import (  # pyright: ignore[reportMissingImports]
            RolloutSample,
        )
    except ImportError as exc:  # pragma: no cover - detached planning catches this before execution
        raise RuntimeError("the selected TRL distribution does not provide native async rollout samples") from exc

    result = []
    for (record, policy_version, input_ids, completion_mask, old_log_probs), advantage in zip(
        prepared, advantages, strict=True
    ):
        result.append(
            RolloutSample(
                prompt=[dict(message) for message in record.prompt_messages],
                completion=[dict(message) for message in record.completion_messages],
                input_ids=input_ids,
                completion_mask=completion_mask,
                old_log_probs=old_log_probs,
                advantage=float(advantage),
                model_version=policy_version,
                group_id=group_id,
                metrics={"reward": float(record.rollout.reward)},
            )
        )
    return tuple(result)


__all__ = [
    "AsyncRolloutRecord",
    "BehaviorPolicySpan",
    "InvalidAsyncSampleGroup",
    "project_async_group",
]
