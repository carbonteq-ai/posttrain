import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.train.integrations.verifiers_workers import (
    InvalidNativeEpisode,
    VerifiersWorkerPool,
)
from posttrain.train.rollout_execution import (
    CollectionExecutionError,
    CollectionKey,
    EpisodeKey,
    EpisodeStatus,
    RolloutExecutionConfig,
)


class FakePool:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.address = "tcp://127.0.0.1:43123"
        self.stopped = asyncio.Event()
        self.instances.append(self)

    async def run(self):
        try:
            await asyncio.Event().wait()
        finally:
            self.stopped.set()


class FakeClient:
    def __init__(self, address):
        self.address = address
        self.started = False
        self.closed = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = 0
        self.tasks = {}

    async def wait_for_server_startup(self, timeout):
        assert timeout == 3.0
        self.started = True

    async def run(self, *, client, model, sampling, task_data, request_id):
        del client, sampling
        assert model == "policy-model"
        self.tasks[request_id] = asyncio.current_task()
        self.entered.set()
        if task_data.get("wait"):
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return SimpleNamespace(value=task_data["value"])

    async def cancel(self, request_id):
        task = self.tasks.get(request_id)
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def close(self):
        self.closed = True


class Data:
    def __init__(self, **values):
        self.values = values

    def model_dump(self, *, mode):
        assert mode == "json"
        return dict(self.values)


def episode_key(collection, occurrence="rollout-1"):
    return EpisodeKey(
        collection=collection,
        example_id="task-1",
        group_id="group-1",
        occurrence_id=occurrence,
        seed=7,
    )


def create_pool(client, projector=None, *, cancel_timeout=1.0):
    return VerifiersWorkerPool(
        model="policy-model",
        sampling=SimpleNamespace(max_tokens=8),
        project_episode=projector
        or (
            lambda key, episode: SimpleNamespace(
                example_id=key.example_id,
                reward_evidence=None,
                native_value=episode.value,
            )
        ),
        global_limit=4,
        startup_timeout=3.0,
        cancel_timeout=cancel_timeout,
        pool_factory=FakePool,
        client_factory=lambda _address: client,
    )


@pytest.mark.asyncio
async def test_fixed_native_pool_dispatches_and_fences_collection_identity():
    FakePool.instances.clear()
    client = FakeClient("unused")
    workers = create_pool(client)
    await workers.start(
        {"id": "test-env"},
        SimpleNamespace(type="train"),
        RolloutExecutionConfig(env_workers=2, episodes_per_worker=2),
    )
    collection = CollectionKey("run-1", "collection-1", "policy-1")
    await workers.open_admission(collection)
    try:
        outcome = await workers.run_episode(
            episode_key(collection),
            SimpleNamespace(data=Data(value=11)),
            asyncio.get_running_loop().time() + 5,
        )
        assert outcome.status is EpisodeStatus.COMPLETED
        assert cast(Any, outcome.rollout).native_value == 11
        native = FakePool.instances[-1]
        assert native.kwargs["max_workers"] == 2
        assert native.kwargs["multiplex"] == 2
        assert native.kwargs["elastic"] is False
        assert native.kwargs["server_kwargs"]["max_concurrent"] == 2

        wrong = episode_key(CollectionKey("run-1", "other", "policy-1"))
        with pytest.raises(CollectionExecutionError, match="does not match"):
            await workers.run_episode(
                wrong,
                SimpleNamespace(data=Data(value=12)),
                asyncio.get_running_loop().time() + 5,
            )
    finally:
        await workers.aclose()
    assert client.closed
    assert FakePool.instances[-1].stopped.is_set()


@pytest.mark.asyncio
async def test_deadline_and_explicit_cancel_are_episode_local_terminal_outcomes():
    client = FakeClient("unused")
    workers = create_pool(client)
    await workers.start(
        {"id": "test-env"},
        SimpleNamespace(type="train"),
        RolloutExecutionConfig(env_workers=1, episodes_per_worker=2),
    )
    collection = CollectionKey("run-1", "collection-1", "policy-1")
    await workers.open_admission(collection)
    try:
        timed_out = await workers.run_episode(
            episode_key(collection, "deadline"),
            SimpleNamespace(data=Data(value=1, wait=True)),
            asyncio.get_running_loop().time() + 0.01,
        )
        assert timed_out.status is EpisodeStatus.CANCELLED
        assert timed_out.error == "native episode deadline exceeded"

        client.entered.clear()
        running = asyncio.create_task(
            workers.run_episode(
                episode_key(collection, "cancelled"),
                SimpleNamespace(data=Data(value=2, wait=True)),
                asyncio.get_running_loop().time() + 5,
            )
        )
        await client.entered.wait()
        assert await workers.cancel(episode_key(collection, "cancelled"))
        cancelled = await running
        assert cancelled.status is EpisodeStatus.CANCELLED
        assert client.cancelled == 2
        assert workers.active_episode_keys == frozenset()
    finally:
        await workers.aclose()


@pytest.mark.asyncio
async def test_declared_invalid_episode_is_not_an_infrastructure_failure():
    def reject(_key, _episode):
        raise InvalidNativeEpisode("native trace is not trainable")

    client = FakeClient("unused")
    workers = create_pool(client, reject)
    await workers.start(
        {"id": "test-env"},
        SimpleNamespace(type="train"),
        RolloutExecutionConfig(env_workers=1, episodes_per_worker=1),
    )
    collection = CollectionKey("run-1", "collection-1", "policy-1")
    await workers.open_admission(collection)
    try:
        outcome = await workers.run_episode(
            episode_key(collection),
            SimpleNamespace(data=Data(value=1)),
            asyncio.get_running_loop().time() + 5,
        )
        assert outcome.status is EpisodeStatus.INVALID
        assert outcome.error == "native trace is not trainable"
        assert workers.fatal_error is None
    finally:
        await workers.aclose()


@pytest.mark.asyncio
async def test_unacknowledged_cancel_poisons_the_collection():
    class UnresponsiveCancelClient(FakeClient):
        async def cancel(self, request_id) -> bool:
            del request_id
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    client = UnresponsiveCancelClient("unused")
    workers = create_pool(client, cancel_timeout=0.01)
    await workers.start(
        {"id": "test-env"},
        SimpleNamespace(type="train"),
        RolloutExecutionConfig(env_workers=1, episodes_per_worker=1),
    )
    collection = CollectionKey("run-1", "collection-1", "policy-1")
    await workers.open_admission(collection)
    running = asyncio.create_task(
        workers.run_episode(
            episode_key(collection),
            SimpleNamespace(data=Data(value=1, wait=True)),
            asyncio.get_running_loop().time() + 5,
        )
    )
    await client.entered.wait()
    try:
        with pytest.raises(CollectionExecutionError, match="not acknowledged"):
            await workers.cancel(episode_key(collection))
        assert isinstance(workers.fatal_error, CollectionExecutionError)
    finally:
        with pytest.raises(CollectionExecutionError, match="could not prove drainage"):
            await workers.aclose()
        await running


@pytest.mark.asyncio
async def test_worker_broker_death_fails_an_active_episode_without_waiting_for_deadline():
    class DyingPool(FakePool):
        release = asyncio.Event()

        async def run(self):
            await self.release.wait()
            raise RuntimeError("worker process exited")

    DyingPool.release = asyncio.Event()
    client = FakeClient("unused")
    workers = VerifiersWorkerPool(
        model="policy-model",
        sampling=SimpleNamespace(max_tokens=8),
        project_episode=lambda key, episode: SimpleNamespace(
            example_id=key.example_id,
            reward_evidence=None,
            native_value=episode.value,
        ),
        global_limit=1,
        startup_timeout=3,
        cancel_timeout=1,
        pool_factory=DyingPool,
        client_factory=lambda _address: client,
    )
    await workers.start(
        {"id": "test-env"},
        SimpleNamespace(type="train"),
        RolloutExecutionConfig(env_workers=1, episodes_per_worker=1),
    )
    collection = CollectionKey("run-1", "collection-1", "policy-1")
    await workers.open_admission(collection)
    running = asyncio.create_task(
        workers.run_episode(
            episode_key(collection),
            SimpleNamespace(data=Data(value=1, wait=True)),
            asyncio.get_running_loop().time() + 60,
        )
    )
    await client.entered.wait()
    DyingPool.release.set()
    with pytest.raises(CollectionExecutionError, match="worker process exited"):
        await asyncio.wait_for(running, timeout=1)
    assert isinstance(workers.fatal_error, RuntimeError)
    await workers.aclose()
