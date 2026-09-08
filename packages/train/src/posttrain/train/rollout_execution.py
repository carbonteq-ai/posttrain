"""Shared identity, capacity, and terminal-outcome contracts for online rollouts.

Backends own their process managers.  This module only makes the invariants
they must both preserve explicit and testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .online_rl import EnvironmentRollout


def _identifier(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} cannot be empty")


@dataclass(frozen=True, slots=True)
class CollectionKey:
    """One fixed-policy collection boundary within a training run."""

    run_id: str
    collection_id: str
    policy_version: str

    def __post_init__(self) -> None:
        _identifier(self.run_id, "run id")
        _identifier(self.collection_id, "collection id")
        _identifier(self.policy_version, "policy version")


@dataclass(frozen=True, slots=True)
class EpisodeKey:
    """A scheduled environment occurrence, stable across worker placement."""

    collection: CollectionKey
    example_id: str
    group_id: str
    occurrence_id: str
    seed: int

    def __post_init__(self) -> None:
        _identifier(self.example_id, "example id")
        _identifier(self.group_id, "group id")
        _identifier(self.occurrence_id, "occurrence id")
        if self.seed < 0:
            raise ValueError("episode seed must be non-negative")


@dataclass(frozen=True, slots=True)
class RolloutExecutionConfig:
    """Requested CPU worker topology for one globally bounded collection."""

    env_workers: int
    episodes_per_worker: int
    worker_native_threads: int = 1

    def __post_init__(self) -> None:
        if self.env_workers < 1:
            raise ValueError("environment worker count must be positive")
        if self.episodes_per_worker < 1:
            raise ValueError("episodes per worker must be positive")
        if self.worker_native_threads < 1:
            raise ValueError("worker native thread count must be positive")

    @property
    def episode_capacity(self) -> int:
        return self.env_workers * self.episodes_per_worker


def effective_cpu_count() -> int:
    """Return CPUs available to this process, respecting Linux affinity when set."""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, available)


def validate_execution_config(
    config: RolloutExecutionConfig,
    *,
    global_limit: int,
    effective_cpus: int | None = None,
) -> None:
    """Reject configurations that silently multiply global rollout capacity."""
    if global_limit < 1:
        raise ValueError("global rollout limit must be positive")
    if config.episode_capacity > global_limit:
        raise ValueError(
            "environment worker capacity exceeds the global rollout limit: "
            f"{config.env_workers} workers * {config.episodes_per_worker} episodes "
            f"> {global_limit}"
        )
    available = effective_cpu_count() if effective_cpus is None else effective_cpus
    if available < 1:
        raise ValueError("effective CPU count must be positive")
    native_threads = config.env_workers * config.worker_native_threads
    if native_threads > available:
        raise ValueError(
            "environment native-thread reservation exceeds available CPUs: "
            f"{config.env_workers} workers * {config.worker_native_threads} threads "
            f"> {available}"
        )


class EpisodeStatus(StrEnum):
    COMPLETED = "completed"
    INVALID = "invalid"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """One terminal result; only a completed outcome may enter reward admission."""

    key: EpisodeKey
    status: EpisodeStatus
    rollout: EnvironmentRollout | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status is EpisodeStatus.COMPLETED:
            if self.rollout is None or self.error is not None:
                raise ValueError("a completed episode outcome requires only a rollout")
        elif self.rollout is not None or not self.error or not self.error.strip():
            raise ValueError("a non-completed episode outcome requires only a non-empty error")


class CollectionExecutionError(RuntimeError):
    """A collection-wide infrastructure failure that must not become reward zero."""


def validate_outcome_identity(expected: EpisodeKey, outcome: EpisodeOutcome) -> None:
    """Fence late or misrouted completions before they reach reward admission."""
    if outcome.key != expected:
        raise CollectionExecutionError("rollout completion key does not match its scheduled occurrence")
    if outcome.status is not EpisodeStatus.COMPLETED:
        return
    assert outcome.rollout is not None  # checked by EpisodeOutcome
    if outcome.rollout.example_id != expected.example_id:
        raise CollectionExecutionError("completed rollout example id does not match its scheduled occurrence")
    evidence = outcome.rollout.reward_evidence
    if evidence is not None and (
        evidence.prompt_group_id != expected.group_id or evidence.rollout_id != expected.occurrence_id
    ):
        raise CollectionExecutionError("completed rollout reward evidence does not match its scheduled occurrence")
