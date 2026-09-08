"""Bounded producer lifecycle for TRL's native asynchronous rollout queue."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import Sequence
from typing import Any, Protocol


class AsyncGroupProducer(Protocol):
    """Produce one already-scored, complete group for a published policy hint."""

    async def produce_group(self, target_policy_version: int) -> Sequence[Any]: ...

    async def aclose(self) -> None: ...


class TrlAsyncRolloutWorker:
    """Adapt an async group producer to TRL's ``RolloutWorkerProtocol``.

    The producer will be backed by Verifiers, but it owns task selection,
    environment execution, admission, and advantage computation. This class
    only owns bounded concurrent production and native queue publication.
    """

    def __init__(
        self,
        producer: AsyncGroupProducer,
        *,
        initial_model_version: int = 0,
        max_inflight_groups: int,
        queue_maxsize: int,
        shutdown_timeout_s: float = 30.0,
    ) -> None:
        if initial_model_version < 0:
            raise ValueError("initial async model version cannot be negative")
        if max_inflight_groups < 1:
            raise ValueError("async rollout worker requires at least one in-flight group")
        if queue_maxsize < 1:
            raise ValueError("async rollout queue must have an explicit positive bound")
        if shutdown_timeout_s <= 0:
            raise ValueError("async rollout shutdown timeout must be positive")
        self.rollout_buffer: queue.Queue[Any] = queue.Queue(maxsize=queue_maxsize)
        self.metrics_queue: queue.Queue[dict[str, float]] = queue.Queue(maxsize=1)
        self._producer = producer
        self._model_version = initial_model_version
        self._max_inflight_groups = max_inflight_groups
        self._shutdown_timeout_s = shutdown_timeout_s
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._last_progress = time.monotonic()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("async rollout worker is already started")
            self._failure = None
            self._last_progress = time.monotonic()
            self._ready.clear()
            thread = threading.Thread(target=self._thread_main, name="posttrain-async-rollouts", daemon=True)
            self._thread = thread
            thread.start()
        if not self._ready.wait(timeout=self._shutdown_timeout_s):
            raise RuntimeError("async rollout worker did not initialize within its startup timeout")
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError(f"async rollout worker failed during startup: {failure}") from failure

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            loop = self._loop
            stop_event = self._stop_event
        if thread is None:
            return
        if loop is not None and stop_event is not None:
            loop.call_soon_threadsafe(stop_event.set)
        thread.join(timeout=self._shutdown_timeout_s)
        if thread.is_alive():
            raise RuntimeError("async rollout worker did not stop within its shutdown timeout")
        with self._lock:
            self._thread = None
            failure = self._failure
        if failure is not None:
            raise RuntimeError(f"async rollout worker failed: {failure}") from failure

    def update_model_version(self, model_version: int) -> None:
        with self._lock:
            if model_version < self._model_version:
                raise ValueError("async model version cannot move backwards")
            self._model_version = model_version

    def check_health(self, stale_after_s: float) -> None:
        if stale_after_s <= 0:
            raise ValueError("async worker health window must be positive")
        with self._lock:
            thread = self._thread
            failure = self._failure
            last_progress = self._last_progress
        if failure is not None:
            raise RuntimeError(f"async rollout worker failed: {failure}") from failure
        if thread is None or not thread.is_alive():
            raise RuntimeError("async rollout worker is not running")
        # TRL calls this only while its rollout queue is empty. A nonempty
        # bounded queue means the producer has useful work waiting.
        if self.rollout_buffer.empty() and time.monotonic() - last_progress > stale_after_s:
            raise RuntimeError("async rollout worker stopped making progress")

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except BaseException as error:
            with self._lock:
                self._failure = error
        finally:
            self._ready.set()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        with self._lock:
            self._loop = loop
            self._stop_event = stop_event
        self._ready.set()
        pending: set[asyncio.Task[tuple[int, Sequence[Any]]]] = set()
        try:
            while not stop_event.is_set():
                while len(pending) < self._max_inflight_groups and not stop_event.is_set():
                    with self._lock:
                        target_version = self._model_version
                    pending.add(asyncio.create_task(self._produce(target_version)))
                if not pending:
                    await asyncio.sleep(0)
                    continue
                done, _ = await asyncio.wait(pending, timeout=0.05, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    pending.remove(task)
                    target_version, samples = task.result()
                    self._validate_native_group(samples, target_version)
                    for sample in samples:
                        await self._publish(sample, stop_event)
                    with self._lock:
                        self._last_progress = time.monotonic()
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await self._producer.aclose()
            with self._lock:
                self._loop = None
                self._stop_event = None

    async def _produce(self, target_version: int) -> tuple[int, Sequence[Any]]:
        return target_version, await self._producer.produce_group(target_version)

    async def _publish(self, sample: Any, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                self.rollout_buffer.put_nowait(sample)
                return
            except queue.Full:
                await asyncio.sleep(0.01)

    @staticmethod
    def _validate_native_group(samples: Sequence[Any], target_version: int) -> None:
        if not samples:
            raise RuntimeError("async group producer returned an empty group")
        versions = {getattr(sample, "model_version", None) for sample in samples}
        group_ids = {getattr(sample, "group_id", None) for sample in samples}
        if None in versions or len(versions) != 1:
            raise RuntimeError("async group samples must share one behavior policy version")
        behavior_version = next(iter(versions))
        if not isinstance(behavior_version, int) or behavior_version > target_version:
            raise RuntimeError("async group was served by an unpublished policy version")
        if None in group_ids or len(group_ids) != 1:
            raise RuntimeError("async group samples must share one native group id")


__all__ = ["AsyncGroupProducer", "TrlAsyncRolloutWorker"]
