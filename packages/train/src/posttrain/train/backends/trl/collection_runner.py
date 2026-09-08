"""Fixed-policy orchestration for native TRL environment collections."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ...integrations.verifiers import VerifiersRolloutFailure
from ...online_rl import EnvironmentRollout, RolloutBatch
from ...rollout_execution import (
    CollectionExecutionError,
    CollectionKey,
    EpisodeKey,
    EpisodeOutcome,
    RolloutExecutionConfig,
)


class _PolicySession(Protocol):
    async def synchronize_policy(self, version: str) -> None: ...

    async def suspend_for_update(self) -> None: ...

    async def aclose(self) -> None: ...


class _PolicyEndpoint(Protocol):
    @property
    def base_url(self) -> str: ...

    @property
    def fatal_error(self) -> BaseException | None: ...

    async def start(self, session: Any) -> None: ...

    async def open_admission(self, collection: CollectionKey) -> None: ...

    async def stop_admission(self, collection: CollectionKey) -> None: ...

    async def aclose(self) -> None: ...


class _EnvironmentWorkers(Protocol):
    async def start(
        self,
        activation: Mapping[str, Any],
        client_config: Any,
        execution_config: RolloutExecutionConfig,
    ) -> None: ...

    async def open_admission(self, collection: CollectionKey) -> None: ...

    async def stop_admission(self, collection: CollectionKey) -> None: ...

    async def run_episode(self, key: EpisodeKey, task: Any, deadline: float) -> EpisodeOutcome: ...

    async def cancel(self, key: EpisodeKey) -> bool: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ScheduledEpisode:
    key: EpisodeKey
    task: Any
    deadline: float

    def __post_init__(self) -> None:
        if self.deadline <= 0:
            raise ValueError("scheduled episode deadline must be a positive monotonic timestamp")


class TrlCollectionRunner:
    """Keep workers resident while each collection uses one fixed policy.

    Returning from :meth:`collect` guarantees that every scheduled episode is
    terminal and the colocated inference session is suspended for the optimizer.
    """

    def __init__(
        self,
        *,
        session: _PolicySession,
        endpoint: _PolicyEndpoint,
        workers: _EnvironmentWorkers,
        client_config_factory: Callable[[str], Any],
        execution_config: RolloutExecutionConfig,
    ) -> None:
        self._session = session
        self._endpoint = endpoint
        self._workers = workers
        self._client_config_factory = client_config_factory
        self._execution_config = execution_config
        self._started = False
        self._closed = False
        self._collection_lock = asyncio.Lock()

    async def start(self, activation: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("TRL collection runner is closed")
        if self._started:
            raise RuntimeError("TRL collection runner is already started")
        await self._endpoint.start(self._session)
        try:
            client_config = self._client_config_factory(self._endpoint.base_url)
            await self._workers.start(activation, client_config, self._execution_config)
        except BaseException:
            await self._endpoint.aclose()
            raise
        self._started = True

    async def collect(
        self,
        collection: CollectionKey,
        scheduled: Sequence[ScheduledEpisode],
    ) -> tuple[EpisodeOutcome, ...]:
        if not self._started or self._closed:
            raise RuntimeError("TRL collection runner has not started")
        if not scheduled:
            raise ValueError("TRL collection requires at least one scheduled episode")
        if any(item.key.collection != collection for item in scheduled):
            raise CollectionExecutionError("scheduled episode belongs to a different collection")
        if len({item.key for item in scheduled}) != len(scheduled):
            raise CollectionExecutionError("TRL collection contains duplicate episode identities")

        async with self._collection_lock:
            try:
                await self._session.synchronize_policy(collection.policy_version)
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise CollectionExecutionError(
                    f"TRL policy synchronization failed: {type(error).__name__}: {error}"
                ) from error
            endpoint_open = False
            workers_open = False
            tasks: list[asyncio.Task[EpisodeOutcome]] = []
            collection_error: BaseException | None = None
            outcomes: tuple[EpisodeOutcome, ...] = ()
            try:
                await self._endpoint.open_admission(collection)
                endpoint_open = True
                await self._workers.open_admission(collection)
                workers_open = True
                tasks = [
                    asyncio.create_task(
                        self._workers.run_episode(item.key, item.task, item.deadline),
                        name=f"verifiers-episode-{item.key.occurrence_id}",
                    )
                    for item in scheduled
                ]
                outcomes = tuple(await asyncio.gather(*tasks))
                if self._endpoint.fatal_error is not None:
                    raise CollectionExecutionError(
                        f"TRL policy endpoint failed: {self._endpoint.fatal_error}"
                    ) from self._endpoint.fatal_error
            except BaseException as error:
                collection_error = error
                for item, task in zip(scheduled, tasks, strict=True):
                    if not task.done():
                        try:
                            await self._workers.cancel(item.key)
                        except BaseException as cancel_error:
                            collection_error = CollectionExecutionError(
                                f"TRL collection cancellation failed: {cancel_error}"
                            )
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                cleanup_failures: list[BaseException] = []
                cleanup = (
                    (workers_open, lambda: self._workers.stop_admission(collection)),
                    (endpoint_open, lambda: self._endpoint.stop_admission(collection)),
                    (endpoint_open, self._session.suspend_for_update),
                )
                for required, operation in cleanup:
                    if not required:
                        continue
                    try:
                        await operation()
                    except BaseException as cleanup_error:
                        cleanup_failures.append(cleanup_error)
                if cleanup_failures:
                    collection_error = CollectionExecutionError(
                        "TRL collection did not reach a safe optimizer handoff: "
                        + "; ".join(
                            f"{type(error).__name__}: {error}" for error in cleanup_failures
                        )
                    )
            if collection_error is not None:
                if isinstance(collection_error, asyncio.CancelledError):
                    raise collection_error
                if isinstance(collection_error, CollectionExecutionError):
                    raise collection_error
                raise CollectionExecutionError(
                    f"TRL collection failed: {type(collection_error).__name__}: {collection_error}"
                ) from collection_error
            if len(outcomes) != len(scheduled):
                raise CollectionExecutionError("TRL collection did not terminalize every scheduled episode")
            return outcomes

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        for close in (self._workers.aclose, self._endpoint.aclose, self._session.aclose):
            try:
                await close()
            except BaseException as error:
                failures.append(error)
        self._started = False
        if failures:
            raise CollectionExecutionError(
                "TRL collection runner shutdown failed: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in failures)
            ) from failures[0]


class TrlNativeRolloutCollector:
    """Translate one framework rollout batch into native worker occurrences.

    Algorithm-specific group admission remains above this class. A terminal
    episode failure is returned through the existing partial-batch exception;
    collection infrastructure failures continue to abort the optimizer handoff.
    """

    def __init__(self, *, runner: TrlCollectionRunner, bridge: Any, episode_timeout: float) -> None:
        if episode_timeout <= 0:
            raise ValueError("native rollout episode timeout must be positive")
        self._runner = runner
        self._bridge = bridge
        self._episode_timeout = episode_timeout

    async def collect(
        self,
        batch: RolloutBatch,
        *,
        collection_id: str,
        policy_version: str,
    ) -> Sequence[EnvironmentRollout]:
        if not collection_id.strip() or not policy_version.strip():
            raise ValueError("native rollout collection and policy identities cannot be empty")
        if batch.prompt_group_ids and len(batch.prompt_group_ids) != len(batch.example_ids):
            raise ValueError("native rollout prompt-group identities are not aligned to the batch")
        if batch.rollout_ids and len(batch.rollout_ids) != len(batch.example_ids):
            raise ValueError("native rollout occurrence identities are not aligned to the batch")
        collection = CollectionKey(
            run_id=self._bridge.run_id,
            collection_id=collection_id,
            policy_version=policy_version,
            logical_step=batch.step,
        )
        deadline = asyncio.get_running_loop().time() + self._episode_timeout
        scheduled = tuple(
            ScheduledEpisode(
                key=EpisodeKey(
                    collection=collection,
                    example_id=example_id,
                    group_id=(
                        batch.prompt_group_ids[ordinal]
                        if batch.prompt_group_ids
                        else f"{collection_id}/group/{ordinal}"
                    ),
                    occurrence_id=(
                        batch.rollout_ids[ordinal]
                        if batch.rollout_ids
                        else f"{collection_id}/response/{ordinal}"
                    ),
                    seed=_episode_seed(collection, ordinal),
                    rollout_ordinal=ordinal,
                ),
                task=self._task(example_id),
                deadline=deadline,
            )
            for ordinal, example_id in enumerate(batch.example_ids)
        )
        outcomes = await self._runner.collect(collection, scheduled)
        completed: dict[int, EnvironmentRollout] = {}
        failures: dict[int, str] = {}
        for ordinal, outcome in enumerate(outcomes):
            if outcome.key != scheduled[ordinal].key:
                raise CollectionExecutionError("native rollout outcome order or identity changed")
            if outcome.rollout is None:
                failures[ordinal] = outcome.error or f"native episode ended as {outcome.status.value}"
            else:
                completed[ordinal] = outcome.rollout
        if failures:
            raise VerifiersRolloutFailure(
                f"{len(failures)} of {len(outcomes)} native Verifiers rollouts failed: "
                f"{sorted(set(failures.values()))}",
                completed=completed,
                failures=failures,
            )
        return [completed[ordinal] for ordinal in range(len(outcomes))]

    def _task(self, example_id: str) -> Any:
        return self._bridge.task_for_example_id(example_id)


def _episode_seed(collection: CollectionKey, ordinal: int) -> int:
    payload = "\0".join(
        (collection.run_id, collection.collection_id, collection.policy_version, str(ordinal))
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = ["ScheduledEpisode", "TrlCollectionRunner", "TrlNativeRolloutCollector"]
