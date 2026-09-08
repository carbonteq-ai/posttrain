"""Native Verifiers group production for TRL's asynchronous GRPO learner."""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from ..backends.trl.async_samples import AsyncRolloutRecord, InvalidAsyncSampleGroup, project_async_group
from ..online_rl import EnvironmentRollout
from ..profiles import GRPOSettings, shape_online_reward
from ..rollout_execution import (
    CollectionExecutionError,
    CollectionKey,
    EpisodeKey,
    EpisodeOutcome,
    EpisodeStatus,
    RolloutExecutionConfig,
    RolloutGroupRejected,
)


class _AsyncEnvironmentWorkers(Protocol):
    async def start(
        self,
        activation: Mapping[str, Any],
        client_config: Any,
        execution_config: RolloutExecutionConfig,
    ) -> None: ...

    async def open_run_admission(self, run_id: str) -> None: ...

    async def stop_run_admission(self, run_id: str) -> None: ...

    async def run_episode(self, key: EpisodeKey, task: Any, deadline: float) -> EpisodeOutcome: ...

    async def cancel(self, key: EpisodeKey) -> bool: ...

    async def aclose(self) -> None: ...


class _PolicyAdmission(Protocol):
    @property
    def base_url(self) -> str: ...

    @property
    def fatal_error(self) -> BaseException | None: ...

    async def start(self, initial_model_version: int) -> None: ...

    async def prepare_model_update(self, model_version: int) -> None: ...

    async def activate_model_version(self, model_version: int) -> None: ...

    async def aclose(self) -> None: ...


class _EnvironmentBridge(Protocol):
    @property
    def dataset(self) -> Any: ...

    def task_for_example_id(self, example_id: str) -> Any: ...


class _PromptSelector:
    """Deterministic in-memory prompt ordering; durable consumption is separate."""

    def __init__(self, example_ids: Sequence[str], *, seed: int, shuffle: bool) -> None:
        if not example_ids or any(not value.strip() for value in example_ids):
            raise ValueError("async prompt selection requires non-empty stable example ids")
        if len(set(example_ids)) != len(example_ids):
            raise ValueError("async prompt selection requires unique example ids")
        self._base = tuple(example_ids)
        self._seed = seed
        self._shuffle = shuffle
    @property
    def example_ids(self) -> tuple[str, ...]:
        return self._base

    def at(self, group_id: int) -> str:
        if group_id < 0:
            raise ValueError("async prompt group id cannot be negative")
        epoch, index = divmod(group_id, len(self._base))
        return self._order(epoch)[index]

    def _order(self, epoch: int) -> tuple[str, ...]:
        values = list(self._base)
        if self._shuffle:
            values.sort(
                key=lambda value: hashlib.sha256(
                    f"{self._seed}\0{epoch}\0{value}".encode()
                ).digest()
            )
        return tuple(values)


class VerifiersAsyncGroupProducer:
    """Collect, validate, score, and project complete native async GRPO groups.

    The producer owns ephemeral prompt scheduling only. It deliberately does
    not claim checkpoint recovery until TRL acknowledges learner consumption.
    """

    def __init__(
        self,
        *,
        workers: _AsyncEnvironmentWorkers,
        policy_admission: _PolicyAdmission,
        bridge: _EnvironmentBridge,
        activation: Mapping[str, Any],
        client_config_factory: Callable[[str], Any],
        execution_config: RolloutExecutionConfig,
        settings: GRPOSettings,
        run_id: str,
        initial_model_version: int,
        episode_timeout_s: float,
        max_group_tokens: int,
        seed: int,
    ) -> None:
        if settings.algorithm != "grpo":
            raise ValueError("native async Verifiers production currently supports GRPO only")
        if settings.advantage_scaling != "group":
            raise ValueError("native async Verifiers GRPO currently requires group advantage scaling")
        if settings.dynamic_sampling is not None or settings.active_sampling is not None:
            raise ValueError("native async Verifiers GRPO does not yet support dynamic or active sampling")
        if not run_id.strip():
            raise ValueError("async Verifiers run identity cannot be empty")
        if initial_model_version < 0:
            raise ValueError("initial async model version cannot be negative")
        if seed < 0:
            raise ValueError("async Verifiers seed cannot be negative")
        if episode_timeout_s <= 0 or max_group_tokens < 1:
            raise ValueError("async episode timeout and group-token bound must be positive")
        example_ids = tuple(str(example.id) for example in bridge.dataset.examples)
        self._workers = workers
        self._policy_admission = policy_admission
        self._bridge = bridge
        self._activation = dict(activation)
        self._client_config_factory = client_config_factory
        self._execution_config = execution_config
        self._settings = settings
        self._run_id = run_id
        self._active_model_version = initial_model_version
        self._episode_timeout_s = episode_timeout_s
        self._max_group_tokens = max_group_tokens
        self._seed = seed
        self._selector = _PromptSelector(example_ids, seed=seed, shuffle=settings.shuffle_prompts)
        self._next_group_id = 0
        self._replay_group_ids: list[int] = []
        self._consumed_counts: dict[int, int] = {}
        self._rejected_group_ids: set[int] = set()
        self._started = False
        self._closed = False
        self._policy_started = False
        self._workers_started = False
        self._run_admission_open = False
        self._state_lock = asyncio.Lock()
        self._active: dict[int, tuple[EpisodeKey, ...]] = {}

    async def astart(self) -> None:
        if self._closed:
            raise RuntimeError("async Verifiers group producer is closed")
        if self._started:
            raise RuntimeError("async Verifiers group producer is already started")
        try:
            self._policy_started = True
            await self._policy_admission.start(self._active_model_version)
            client_config = self._client_config_factory(self._policy_admission.base_url)
            self._workers_started = True
            await self._workers.start(self._activation, client_config, self._execution_config)
            await self._workers.open_run_admission(self._run_id)
            self._run_admission_open = True
        except BaseException:
            await self.aclose()
            raise
        self._started = True

    async def produce_group(self, target_policy_version: int) -> Sequence[Any]:
        self._require_running()
        if target_policy_version > self._active_model_version:
            raise CollectionExecutionError("async group requested an unpublished policy version")
        async with self._state_lock:
            if self._replay_group_ids:
                group_id = self._replay_group_ids.pop(0)
            else:
                group_id = self._next_group_id
                self._next_group_id += 1
            example_id = self._selector.at(group_id)
        collection = CollectionKey(
            run_id=self._run_id,
            collection_id=f"async/group/{group_id}",
            policy_version=str(target_policy_version),
            logical_step=target_policy_version,
        )
        keys = tuple(
            EpisodeKey(
                collection=collection,
                example_id=example_id,
                group_id=collection.collection_id,
                occurrence_id=f"{collection.collection_id}/response/{ordinal}",
                seed=_episode_seed(self._run_id, self._seed, group_id, ordinal),
                rollout_ordinal=ordinal,
            )
            for ordinal in range(self._settings.num_generations)
        )
        async with self._state_lock:
            self._active[group_id] = keys
        deadline = asyncio.get_running_loop().time() + self._episode_timeout_s
        task = self._bridge.task_for_example_id(example_id)
        episode_tasks = tuple(
            asyncio.create_task(
                self._workers.run_episode(key, task, deadline),
                name=f"posttrain-verifiers-{key.occurrence_id}",
            )
            for key in keys
        )
        try:
            outcomes = await asyncio.gather(*episode_tasks)
        except BaseException as error:
            try:
                await self._cancel(keys)
            except BaseException as cancellation_error:
                raise CollectionExecutionError(
                    "native Verifiers group failed and sibling cancellation also failed: "
                    f"{type(cancellation_error).__name__}: {cancellation_error}"
                ) from error
            finally:
                for episode_task in episode_tasks:
                    if not episode_task.done():
                        episode_task.cancel()
                await asyncio.gather(*episode_tasks, return_exceptions=True)
            raise
        finally:
            async with self._state_lock:
                self._active.pop(group_id, None)
        failures = [outcome for outcome in outcomes if outcome.status is not EpisodeStatus.COMPLETED]
        if failures:
            if self._policy_admission.fatal_error is not None:
                raise CollectionExecutionError(
                    f"async policy gateway failed: {self._policy_admission.fatal_error}"
                ) from self._policy_admission.fatal_error
            reasons = sorted({outcome.error or outcome.status.value for outcome in failures})
            await self._mark_rejected(group_id)
            raise RolloutGroupRejected(
                f"{len(failures)} of {len(outcomes)} native Verifiers episodes were rejected: {reasons}"
            )
        rollouts = tuple(self._completed_rollout(key, outcome) for key, outcome in zip(keys, outcomes, strict=True))
        token_count = sum(len(row.prompt_ids) + len(row.completion_ids) for row in rollouts)
        if token_count > self._max_group_tokens:
            await self._mark_rejected(group_id)
            raise RolloutGroupRejected(
                f"async Verifiers group uses {token_count} tokens, exceeding {self._max_group_tokens}"
            )
        shaped_rewards = tuple(
            shape_online_reward(self._settings, rollout.reward, len(rollout.completion_ids)) for rollout in rollouts
        )
        records = tuple(
            AsyncRolloutRecord(
                occurrence_id=key.occurrence_id,
                rollout=rollout,
                algorithm_reward=algorithm_reward,
            )
            for key, rollout, algorithm_reward in zip(keys, rollouts, shaped_rewards, strict=True)
        )
        advantages = _grpo_advantages(shaped_rewards)
        try:
            return project_async_group(
                records,
                advantages,
                group_id=group_id,
                expected_group_size=self._settings.num_generations,
            )
        except InvalidAsyncSampleGroup as error:
            await self._mark_rejected(group_id)
            raise RolloutGroupRejected(str(error)) from error

    async def prepare_model_update(self, model_version: int) -> None:
        self._require_running()
        if model_version <= self._active_model_version:
            raise ValueError("next async policy version must exceed the active version")
        await self._policy_admission.prepare_model_update(model_version)

    async def activate_model_version(self, model_version: int) -> None:
        self._require_running()
        if model_version <= self._active_model_version:
            raise ValueError("activated async policy version must exceed the active version")
        await self._policy_admission.activate_model_version(model_version)
        self._active_model_version = model_version

    async def acknowledge_consumed_samples(self, group_ids: Sequence[int]) -> None:
        """Advance recovery state from learner admission, never generation or enqueueing."""
        self._require_running()
        async with self._state_lock:
            for group_id in group_ids:
                if group_id < 0 or group_id >= self._next_group_id:
                    raise CollectionExecutionError(f"learner acknowledged unknown async group {group_id}")
                if group_id in self._rejected_group_ids:
                    raise CollectionExecutionError(f"learner acknowledged rejected async group {group_id}")
                count = self._consumed_counts.get(group_id, 0) + 1
                if count > self._settings.num_generations:
                    raise CollectionExecutionError(
                        f"learner acknowledged too many samples for async group {group_id}"
                    )
                self._consumed_counts[group_id] = count

    async def rollout_state_dict(self) -> Mapping[str, Any]:
        """Return replay-safe scheduling state, excluding all live runtime objects."""
        self._require_running()
        async with self._state_lock:
            partial = {
                group_id: count
                for group_id, count in self._consumed_counts.items()
                if count != self._settings.num_generations
            }
            if partial:
                raise CollectionExecutionError(
                    f"cannot checkpoint partially consumed async groups: {sorted(partial.items())}"
                )
            return {
                "format_version": 1,
                "run_id": self._run_id,
                "seed": self._seed,
                "example_ids": list(self._selector.example_ids),
                "num_generations": self._settings.num_generations,
                "next_group_id": self._next_group_id,
                "consumed_group_ids": sorted(self._consumed_counts),
                "rejected_group_ids": sorted(self._rejected_group_ids),
            }

    def load_rollout_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a validated cursor before any environment or policy runtime starts."""
        if self._started or self._closed:
            raise RuntimeError("async rollout state must be restored before producer startup")
        expected = {
            "format_version": 1,
            "run_id": self._run_id,
            "seed": self._seed,
            "example_ids": list(self._selector.example_ids),
            "num_generations": self._settings.num_generations,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"async rollout checkpoint {name} does not match the selected run")
        next_group_id = state.get("next_group_id")
        consumed = _group_id_set(state.get("consumed_group_ids"), name="consumed_group_ids")
        rejected = _group_id_set(state.get("rejected_group_ids"), name="rejected_group_ids")
        if not isinstance(next_group_id, int) or isinstance(next_group_id, bool) or next_group_id < 0:
            raise ValueError("async rollout checkpoint next_group_id must be a non-negative integer")
        if consumed & rejected or any(group_id >= next_group_id for group_id in consumed | rejected):
            raise ValueError("async rollout checkpoint group sets are inconsistent with its cursor")
        self._next_group_id = next_group_id
        self._consumed_counts = {
            group_id: self._settings.num_generations for group_id in consumed
        }
        self._rejected_group_ids = rejected
        settled = consumed | rejected
        self._replay_group_ids = [group_id for group_id in range(next_group_id) if group_id not in settled]

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        active = tuple(key for keys in self._active.values() for key in keys)
        await self._cancel(active)
        failures: list[BaseException] = []
        if self._run_admission_open:
            try:
                await self._workers.stop_run_admission(self._run_id)
            except BaseException as error:
                failures.append(error)
            self._run_admission_open = False
        if self._workers_started:
            try:
                await self._workers.aclose()
            except BaseException as error:
                failures.append(error)
            self._workers_started = False
        if self._policy_started:
            try:
                await self._policy_admission.aclose()
            except BaseException as error:
                failures.append(error)
            self._policy_started = False
        self._started = False
        if failures:
            raise CollectionExecutionError(
                "async Verifiers producer shutdown failed: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in failures)
            ) from failures[0]

    async def _cancel(self, keys: Sequence[EpisodeKey]) -> None:
        results = await asyncio.gather(*(self._workers.cancel(key) for key in keys), return_exceptions=True)
        errors = [value for value in results if isinstance(value, BaseException)]
        if errors:
            raise CollectionExecutionError(
                "async Verifiers cancellation failed: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            ) from errors[0]

    async def _mark_rejected(self, group_id: int) -> None:
        async with self._state_lock:
            self._rejected_group_ids.add(group_id)

    def _completed_rollout(self, key: EpisodeKey, outcome: EpisodeOutcome) -> EnvironmentRollout:
        if outcome.key != key or outcome.rollout is None:
            raise CollectionExecutionError("native Verifiers outcome identity changed before group projection")
        if outcome.rollout.example_id != key.example_id:
            raise CollectionExecutionError("native Verifiers rollout example identity changed")
        behavior_policy = outcome.rollout.behavior_policy
        if behavior_policy is None:
            raise CollectionExecutionError("native Verifiers rollout has no served policy-version evidence")
        try:
            target_version = int(key.collection.policy_version)
        except ValueError as error:
            raise CollectionExecutionError("async collection policy version is not numeric") from error
        if behavior_policy.start != target_version:
            raise CollectionExecutionError(
                "native Verifiers rollout started on a different policy than its collection"
            )
        return outcome.rollout

    def _require_running(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("async Verifiers group producer is not running")


def _grpo_advantages(rewards: Sequence[float], epsilon: float = 1e-8) -> tuple[float, ...]:
    if len(rewards) < 2 or not all(math.isfinite(value) for value in rewards):
        raise RolloutGroupRejected("async GRPO requires at least two finite group rewards")
    mean = math.fsum(rewards) / len(rewards)
    centered = tuple(value - mean for value in rewards)
    variance = math.fsum(value * value for value in centered) / len(centered)
    scale = math.sqrt(variance)
    return tuple(value / (scale + epsilon) for value in centered)


def _episode_seed(run_id: str, seed: int, group_id: int, ordinal: int) -> int:
    payload = f"{run_id}\0{seed}\0{group_id}\0{ordinal}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _group_id_set(value: Any, *, name: str) -> set[int]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value
    ):
        raise ValueError(f"async rollout checkpoint {name} must contain non-negative integers")
    result = set(value)
    if len(result) != len(value):
        raise ValueError(f"async rollout checkpoint {name} cannot contain duplicates")
    return result


__all__ = ["VerifiersAsyncGroupProducer"]
