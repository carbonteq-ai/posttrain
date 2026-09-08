"""Native Verifiers worker configuration for environment-driven training."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from posttrain.common import ModelVariant

from ..online_rl import BehaviorPolicySpan
from ..profiles import TrainingRenderer
from ..rendering import create_renderer_config
from ..rollout_execution import (
    CollectionExecutionError,
    CollectionKey,
    EpisodeKey,
    EpisodeOutcome,
    EpisodeStatus,
    InvalidNativeEpisode,
    RolloutExecutionConfig,
    validate_execution_config,
    validate_outcome_identity,
)

type EpisodeProjector = Callable[..., Any | Awaitable[Any]]
type EpisodePolicySpanProvider = Callable[
    [EpisodeKey, Any], BehaviorPolicySpan | Awaitable[BehaviorPolicySpan]
]


class _EpisodeDeadline(RuntimeError):
    pass


class _NativeEpisodeFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _EpisodeRegistration:
    task: asyncio.Task[Any]
    native_request_id: str


def create_verifiers_train_client_config(
    *,
    base_url: str,
    renderer_model_name: str,
    model: ModelVariant,
    renderer: TrainingRenderer,
    api_key_var: str = "POSTTRAIN_INPROCESS_POLICY",
    multiplex: int = 256,
) -> Any:
    """Build the serializable exact-token client used by native env workers.

    ``renderer_model_name`` is a resolved immutable artifact or local snapshot
    path supplied by composition. It must not be inferred from a mutable model
    alias here.
    """

    try:
        from verifiers.v1.configs.client import TrainClientConfig
    except ImportError as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    if "chat_template" not in TrainClientConfig.model_fields:
        raise RuntimeError(
            "native rollout workers require a Verifiers TrainClientConfig with exact chat-template support"
        )
    values: dict[str, Any] = {
        "base_url": base_url,
        "api_key_var": api_key_var,
        "renderer": create_renderer_config(model, renderer),
        "renderer_model_name": renderer_model_name,
        "chat_template": model.conversation.chat_template.text(),
        "multiplex": multiplex,
    }
    return TrainClientConfig(**values)


class VerifiersWorkerPool:
    """Own a fixed native Verifiers environment-process pool.

    The native pool owns environment processes and episode execution. This
    adapter owns logical occurrence identity, deadlines, admission scope, and
    translation into terminal Posttrain outcomes. Synchronous collectors use
    one exact collection; native async trainers admit multiple collections from
    one run. It never interprets an optimization algorithm.
    """

    def __init__(
        self,
        *,
        model: str,
        sampling: Any,
        project_episode: EpisodeProjector,
        behavior_policy_for_episode: EpisodePolicySpanProvider | None = None,
        global_limit: int,
        startup_timeout: float = 120.0,
        cancel_timeout: float = 10.0,
        pool_factory: Callable[..., Any] | None = None,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("native worker policy model cannot be empty")
        if global_limit < 1:
            raise ValueError("native worker global limit must be positive")
        if startup_timeout <= 0:
            raise ValueError("native worker startup timeout must be positive")
        if cancel_timeout <= 0:
            raise ValueError("native worker cancel timeout must be positive")
        self._model = model
        self._sampling = sampling
        self._project_episode = project_episode
        self._behavior_policy_for_episode = behavior_policy_for_episode
        self._global_limit = global_limit
        self._startup_timeout = startup_timeout
        self._cancel_timeout = cancel_timeout
        self._pool_factory = pool_factory
        self._client_factory = client_factory
        self._pool: Any | None = None
        self._client: Any | None = None
        self._pool_task: asyncio.Task[Any] | None = None
        self._client_config: Any | None = None
        self._collection: CollectionKey | None = None
        self._run_id: str | None = None
        self._requests: dict[EpisodeKey, _EpisodeRegistration] = {}
        self._cancelled: set[EpisodeKey] = set()
        self._fatal_error: BaseException | None = None
        self._closing = False
        self._lock = asyncio.Lock()

    @property
    def active_episode_keys(self) -> frozenset[EpisodeKey]:
        return frozenset(self._requests)

    @property
    def fatal_error(self) -> BaseException | None:
        return self._fatal_error

    async def start(
        self,
        activation: Mapping[str, Any],
        client_config: Any,
        execution_config: RolloutExecutionConfig,
    ) -> None:
        """Start all fixed workers and wait for the native broker to be healthy."""
        validate_execution_config(execution_config, global_limit=self._global_limit)
        async with self._lock:
            if self._pool is not None:
                raise RuntimeError("native Verifiers worker pool is already started")
            self._closing = False
            self._fatal_error = None
        pool_factory, client_factory = self._native_factories()
        pool = pool_factory(
            server_kwargs={
                "config_data": dict(activation),
                "max_concurrent": execution_config.episodes_per_worker,
            },
            max_workers=execution_config.env_workers,
            address="tcp://127.0.0.1:0",
            multiplex=execution_config.episodes_per_worker,
            elastic=False,
        )
        client = client_factory(pool.address)
        pool_task = asyncio.create_task(pool.run(), name="verifiers-env-server-pool")
        pool_task.add_done_callback(self._pool_finished)
        async with self._lock:
            self._pool = pool
            self._client = client
            self._pool_task = pool_task
            self._client_config = client_config
        startup = asyncio.create_task(client.wait_for_server_startup(timeout=self._startup_timeout))
        try:
            done, _ = await asyncio.wait({startup, pool_task}, return_when=asyncio.FIRST_COMPLETED)
            if pool_task in done:
                error = self._fatal_error or RuntimeError(
                    "native Verifiers worker broker stopped during startup"
                )
                raise CollectionExecutionError(str(error)) from error
            await startup
        except BaseException:
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)
            await self.aclose()
            raise

    async def open_admission(self, collection: CollectionKey) -> None:
        async with self._lock:
            self._require_healthy()
            if self._collection is not None or self._run_id is not None:
                raise RuntimeError("native Verifiers worker admission is already open")
            if self._requests:
                raise RuntimeError("cannot open worker admission while prior episodes remain active")
            self._collection = collection

    async def open_run_admission(self, run_id: str) -> None:
        """Admit independently versioned collections belonging to one async run."""

        if not run_id.strip():
            raise ValueError("native Verifiers worker run id cannot be empty")
        async with self._lock:
            self._require_healthy()
            if self._collection is not None or self._run_id is not None:
                raise RuntimeError("native Verifiers worker admission is already open")
            if self._requests:
                raise RuntimeError("cannot open worker admission while prior episodes remain active")
            self._run_id = run_id

    async def stop_admission(self, collection: CollectionKey) -> None:
        async with self._lock:
            self._require_started()
            if self._collection != collection:
                raise RuntimeError("native Verifiers worker collection identity does not match")
            self._collection = None

    async def stop_run_admission(self, run_id: str) -> None:
        async with self._lock:
            self._require_started()
            if self._run_id != run_id:
                raise RuntimeError("native Verifiers worker run identity does not match")
            self._run_id = None

    async def run_episode(self, key: EpisodeKey, task: Any, deadline: float) -> EpisodeOutcome:
        """Run one occurrence within its original monotonic deadline."""
        current = asyncio.current_task()
        if current is None:  # pragma: no cover - coroutine execution always has a task
            raise RuntimeError("native Verifiers episode is not running in an asyncio task")
        async with self._lock:
            client, client_config = self._require_healthy()
            pool_task = self._pool_task
            if pool_task is None:  # guarded by _require_healthy
                raise RuntimeError("native Verifiers worker broker has no lifecycle task")
            exact_collection = self._collection == key.collection
            async_run = self._run_id == key.collection.run_id
            if not exact_collection and not async_run:
                raise CollectionExecutionError(
                    "episode collection or run does not match native worker admission"
                )
            if key in self._requests:
                raise CollectionExecutionError("duplicate native worker episode identity")
            native_request_id = self._native_request_id(key)
            self._requests[key] = _EpisodeRegistration(current, native_request_id)
        request: asyncio.Task[Any] | None = None
        try:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise _EpisodeDeadline
            task_data = self._task_data(task)
            request = asyncio.create_task(
                client.run(
                    client=client_config,
                    model=self._model,
                    sampling=self._sampling,
                    task_data=task_data,
                    request_id=native_request_id,
                )
            )
            done, _ = await asyncio.wait({request, pool_task}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            if pool_task in done:
                if not request.done():
                    request.cancel()
                    await asyncio.gather(request, return_exceptions=True)
                self._raise_if_broker_failed()
                raise CollectionExecutionError("native Verifiers worker broker stopped during an episode")
            if not done:
                async with self._lock:
                    self._cancelled.add(key)
                await self._cancel_native(client, native_request_id)
                try:
                    async with asyncio.timeout(self._cancel_timeout):
                        await asyncio.shield(request)
                except (RuntimeError, asyncio.CancelledError):
                    pass
                except TimeoutError as error:
                    failure = CollectionExecutionError(
                        f"cancelled native Verifiers run did not terminalize for {native_request_id}"
                    )
                    if self._fatal_error is None:
                        self._fatal_error = failure
                    request.cancel()
                    await asyncio.gather(request, return_exceptions=True)
                    raise failure from error
                raise _EpisodeDeadline
            try:
                episode = await request
            except RuntimeError as error:
                self._raise_if_broker_failed()
                raise _NativeEpisodeFailure(str(error)) from error
            self._raise_if_broker_failed()
            if self._behavior_policy_for_episode is None:
                projected = self._project_episode(key, episode)
            else:
                pending_span = self._behavior_policy_for_episode(key, episode)
                behavior_policy = (
                    await pending_span if inspect.isawaitable(pending_span) else pending_span
                )
                projected = self._project_episode(
                    key,
                    episode,
                    behavior_policy=behavior_policy,
                )
            rollout = await projected if inspect.isawaitable(projected) else projected
            outcome = EpisodeOutcome(key=key, status=EpisodeStatus.COMPLETED, rollout=rollout)
            validate_outcome_identity(key, outcome)
            return outcome
        except _EpisodeDeadline:
            return EpisodeOutcome(
                key=key,
                status=EpisodeStatus.CANCELLED,
                error="native episode deadline exceeded",
            )
        except asyncio.CancelledError:
            if request is not None and not request.done():
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            async with self._lock:
                explicitly_cancelled = key in self._cancelled
            if not explicitly_cancelled:
                raise
            return EpisodeOutcome(
                key=key,
                status=EpisodeStatus.CANCELLED,
                error="native episode cancelled",
            )
        except InvalidNativeEpisode as error:
            return EpisodeOutcome(key=key, status=EpisodeStatus.INVALID, error=str(error))
        except CollectionExecutionError:
            raise
        except _NativeEpisodeFailure as error:
            return EpisodeOutcome(key=key, status=EpisodeStatus.FAILED, error=str(error))
        except Exception as error:
            raise CollectionExecutionError(
                f"unexpected native Verifiers episode failure: {type(error).__name__}: {error}"
            ) from error
        finally:
            async with self._lock:
                self._requests.pop(key, None)
                self._cancelled.discard(key)

    async def cancel(self, key: EpisodeKey) -> bool:
        """Cancel one active occurrence; the native client propagates its wire abort."""
        async with self._lock:
            registration = self._requests.get(key)
            if registration is None or registration.task.done():
                return False
            self._cancelled.add(key)
            client, _ = self._require_healthy()
        return await self._cancel_native(
            client,
            registration.native_request_id,
        )

    async def aclose(self) -> None:
        """Fence admission, cancel episodes, close the client, and reap workers."""
        async with self._lock:
            if self._pool is None:
                return
            self._closing = True
            self._collection = None
            self._run_id = None
            active = tuple(self._requests.items())
            for key, _registration in active:
                self._cancelled.add(key)
            client = self._client
            pool_task = self._pool_task
        close_error: BaseException | None = None
        try:
            if client is not None:
                for _key, registration in active:
                    try:
                        await self._cancel_native(client, registration.native_request_id)
                    except BaseException as error:
                        close_error = close_error or error
            if active:
                try:
                    async with asyncio.timeout(self._cancel_timeout):
                        await asyncio.gather(
                            *(registration.task for _, registration in active),
                            return_exceptions=True,
                        )
                except TimeoutError:
                    close_error = close_error or CollectionExecutionError(
                        "native Verifiers episodes did not drain during shutdown"
                    )
                    for _key, registration in active:
                        registration.task.cancel()
                    await asyncio.gather(
                        *(registration.task for _, registration in active),
                        return_exceptions=True,
                    )
            if client is not None:
                await client.close()
        finally:
            if pool_task is not None and not pool_task.done():
                pool_task.cancel()
            if pool_task is not None:
                await asyncio.gather(pool_task, return_exceptions=True)
            async with self._lock:
                self._pool = None
                self._client = None
                self._pool_task = None
                self._client_config = None
                self._requests.clear()
                self._cancelled.clear()
                self._closing = False
        if close_error is not None:
            raise CollectionExecutionError(
                f"native Verifiers worker shutdown could not prove drainage: {close_error}"
            ) from close_error

    def _native_factories(self) -> tuple[Callable[..., Any], Callable[[str], Any]]:
        if self._pool_factory is not None and self._client_factory is not None:
            return self._pool_factory, self._client_factory
        try:
            from verifiers.v1.serve.client import EnvClient
            from verifiers.v1.serve.pool import EnvServerPool
        except ImportError as error:
            raise RuntimeError("install the Verifiers integration dependencies") from error
        return self._pool_factory or EnvServerPool, self._client_factory or EnvClient

    def _pool_finished(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled() or self._closing:
            return
        error = task.exception()
        self._fatal_error = error or RuntimeError("native Verifiers worker broker stopped")

    def _require_started(self) -> tuple[Any, Any]:
        if self._pool is None or self._client is None or self._client_config is None:
            raise RuntimeError("native Verifiers worker pool has not started")
        return self._client, self._client_config

    def _require_healthy(self) -> tuple[Any, Any]:
        client = self._require_started()
        self._raise_if_broker_failed()
        return client

    def _raise_if_broker_failed(self) -> None:
        if self._fatal_error is not None:
            raise CollectionExecutionError(
                f"native Verifiers worker broker failed: {self._fatal_error}"
            ) from self._fatal_error

    async def _cancel_native(
        self,
        client: Any,
        request_id: str,
    ) -> bool:
        try:
            async with asyncio.timeout(self._cancel_timeout):
                return bool(await client.cancel(request_id))
        except TimeoutError as error:
            failure = CollectionExecutionError(
                f"native Verifiers cancellation was not acknowledged for {request_id}"
            )
            if self._fatal_error is None:
                self._fatal_error = failure
            raise failure from error
        except RuntimeError as error:
            failure = CollectionExecutionError(
                f"native Verifiers cancellation failed for {request_id}: {error}"
            )
            if self._fatal_error is None:
                self._fatal_error = failure
            raise failure from error

    @staticmethod
    def _native_request_id(key: EpisodeKey) -> str:
        identity = "\0".join(
            (
                key.collection.run_id,
                key.collection.collection_id,
                key.collection.policy_version,
                str(key.collection.logical_step),
                key.example_id,
                key.group_id,
                key.occurrence_id,
                str(key.seed),
                str(key.rollout_ordinal),
            )
        )
        return f"posttrain-{hashlib.sha256(identity.encode()).hexdigest()}"

    @staticmethod
    def _task_data(task: Any) -> dict[str, Any]:
        data = getattr(task, "data", None)
        if data is None or not hasattr(data, "model_dump"):
            raise CollectionExecutionError("native Verifiers task has no serializable data")
        payload = data.model_dump(mode="json")
        if not isinstance(payload, dict):
            raise CollectionExecutionError("native Verifiers task data did not serialize to an object")
        return payload


__all__ = [
    "InvalidNativeEpisode",
    "VerifiersWorkerPool",
    "create_verifiers_train_client_config",
]
