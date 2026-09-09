"""Persistent async inference and environment runtime for synchronous TRL rounds."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from posttrain.common import ModelVariant, TraceObservation

from ...integrations.verifiers_workers import VerifiersWorkerPool, create_verifiers_train_client_config
from ...online_rl import EnvironmentRollout, RolloutBatch
from ...profiles import TrainingRenderer
from ...rollout_execution import RolloutExecutionConfig
from .collection_runner import TrlCollectionRunner, TrlNativeRolloutCollector
from .policy_endpoint import TrlPolicyEndpoint


class TrlAsyncCollectionRuntime:
    """Expose continuous-batched rollouts to TRL's synchronous callback.

    One owned thread provides a stable event loop for vLLM ``AsyncLLM``, the
    loopback policy endpoint, and resident Verifiers workers. ``collect`` is a
    synchronous barrier: it returns only after the fixed-policy collection has
    drained and inference memory is suspended for the optimizer update.
    """

    def __init__(
        self,
        *,
        trainer: Any,
        bridge: Any,
        model: ModelVariant,
        renderer: TrainingRenderer,
        renderer_model_name: str,
        max_model_len: int,
        execution_config: RolloutExecutionConfig,
        episode_timeout: float,
        startup_timeout: float = 120.0,
        shutdown_timeout: float = 30.0,
    ) -> None:
        if episode_timeout <= 0 or startup_timeout <= 0 or shutdown_timeout <= 0:
            raise ValueError("TRL async runtime timeouts must be positive")
        if not renderer_model_name.strip():
            raise ValueError("TRL async runtime requires an immutable renderer model name")
        self._trainer = trainer
        self._bridge = bridge
        self._model = model
        self._renderer = renderer
        self._renderer_model_name = renderer_model_name
        self._max_model_len = max_model_len
        self._execution_config = execution_config
        self._episode_timeout = episode_timeout
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._runner: TrlCollectionRunner | None = None
        self._collector: TrlNativeRolloutCollector | None = None
        self._observer: Callable[[TraceObservation], None] | None = None
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._lock = threading.Lock()

    def collect(
        self,
        batch: RolloutBatch,
        *,
        collection_id: str,
        policy_version: str,
        observe_trace: Callable[[TraceObservation], None],
    ) -> Sequence[EnvironmentRollout]:
        self.start()
        with self._lock:
            loop = self._loop
            failure = self._failure
        if failure is not None:
            raise RuntimeError(f"TRL async collection runtime failed: {failure}") from failure
        if loop is None:
            raise RuntimeError("TRL async collection runtime has no serving event loop")
        observations: list[TraceObservation] = []
        future = asyncio.run_coroutine_threadsafe(
            self._collect(
                batch,
                collection_id=collection_id,
                policy_version=policy_version,
                observe_trace=observations.append,
            ),
            loop,
        )
        rollouts = future.result(timeout=self._episode_timeout + self._shutdown_timeout)
        # RunContext observers belong to the trainer thread. Native workers and
        # the async engine never call them from the serving-loop thread.
        for observation in observations:
            observe_trace(observation)
        return rollouts

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._failure = None
            self._ready.clear()
            thread = threading.Thread(target=self._thread_main, name="posttrain-trl-async-collection", daemon=True)
            self._thread = thread
            thread.start()
        if not self._ready.wait(timeout=self._startup_timeout):
            self.close()
            raise RuntimeError("TRL async collection runtime startup timed out")
        with self._lock:
            failure = self._failure
            loop = self._loop
        if failure is not None or loop is None:
            self.close()
            raise RuntimeError(f"TRL async collection runtime failed during startup: {failure}") from failure

    def close(self) -> None:
        with self._lock:
            thread = self._thread
            loop = self._loop
            stop_event = self._stop_event
        if thread is None:
            return
        if loop is not None and stop_event is not None:
            loop.call_soon_threadsafe(stop_event.set)
        thread.join(timeout=self._shutdown_timeout)
        if thread.is_alive():
            raise RuntimeError("TRL async collection runtime did not stop within its shutdown timeout")
        with self._lock:
            self._thread = None
            failure = self._failure
        if failure is not None:
            raise RuntimeError(f"TRL async collection runtime failed: {failure}") from failure

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            with self._lock:
                self._failure = error
        finally:
            self._ready.set()

    async def _serve(self) -> None:
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        with self._lock:
            self._loop = loop
            self._stop_event = stop_event
        try:
            await self._initialize()
            self._ready.set()
            await stop_event.wait()
        finally:
            runner = self._runner
            if runner is not None:
                await runner.aclose()
            with self._lock:
                self._loop = None
                self._stop_event = None

    async def _initialize(self) -> None:
        from verifiers.v1 import Sampling

        session = await self._trainer.vllm_generation.create_async_session()
        endpoint = TrlPolicyEndpoint(model_name=self._renderer_model_name, max_model_len=self._max_model_len)

        async def project_episode(key: Any, episode: Any, **kwargs: Any) -> EnvironmentRollout:
            return await self._bridge.project_native_episode(
                key,
                episode,
                on_completed=self._observe_trace,
                **kwargs,
            )

        sampling = self._bridge.sampling
        sampling_values: dict[str, Any] = {
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "repetition_penalty": sampling.repetition_penalty,
            "presence_penalty": sampling.presence_penalty,
        }
        workers = VerifiersWorkerPool(
            model=self._renderer_model_name,
            sampling=Sampling(**sampling_values),
            project_episode=project_episode,
            global_limit=self._bridge.max_concurrent,
            startup_timeout=self._startup_timeout,
        )
        runner = TrlCollectionRunner(
            session=session,
            endpoint=endpoint,
            workers=workers,
            client_config_factory=lambda base_url: create_verifiers_train_client_config(
                base_url=base_url,
                renderer_model_name=self._renderer_model_name,
                model=self._model,
                renderer=self._renderer,
                multiplex=self._execution_config.episode_capacity,
            ),
            execution_config=self._execution_config,
        )
        await runner.start(self._native_activation())
        self._runner = runner
        self._collector = TrlNativeRolloutCollector(
            runner=runner,
            bridge=self._bridge,
            episode_timeout=self._episode_timeout,
        )

    async def _collect(
        self,
        batch: RolloutBatch,
        *,
        collection_id: str,
        policy_version: str,
        observe_trace: Callable[[TraceObservation], None],
    ) -> Sequence[EnvironmentRollout]:
        if self._collector is None:
            raise RuntimeError("TRL async collection runtime is not initialized")
        if self._observer is not None:
            raise RuntimeError("TRL async collection runtime already has an active collection")
        self._observer = observe_trace
        try:
            return await self._collector.collect(
                batch,
                collection_id=collection_id,
                policy_version=policy_version,
            )
        finally:
            self._observer = None

    async def _observe_trace(self, trace: TraceObservation) -> None:
        observer = self._observer
        if observer is None:
            raise RuntimeError("native Verifiers trace arrived outside its admitted collection")
        observer(trace)

    def _native_activation(self) -> Mapping[str, Any]:
        return self._bridge.native_activation


__all__ = ["TrlAsyncCollectionRuntime"]
