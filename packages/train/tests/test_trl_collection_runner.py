import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.train.backends.trl.collection_runner import (
    ScheduledEpisode,
    TrlCollectionRunner,
    TrlNativeRolloutCollector,
)
from posttrain.train.integrations.verifiers import VerifiersRolloutFailure
from posttrain.train.online_rl import RolloutBatch
from posttrain.train.rollout_execution import (
    CollectionExecutionError,
    CollectionKey,
    EpisodeKey,
    EpisodeOutcome,
    EpisodeStatus,
    RolloutExecutionConfig,
)


def episode_key(collection, ordinal):
    return EpisodeKey(
        collection=collection,
        example_id=f"task-{ordinal}",
        group_id=f"group-{ordinal}",
        occurrence_id=f"rollout-{ordinal}",
        seed=ordinal,
    )


def completed(key):
    return EpisodeOutcome(
        key=key,
        status=EpisodeStatus.COMPLETED,
        rollout=cast(Any, SimpleNamespace(example_id=key.example_id, reward_evidence=None)),
    )


class Session:
    def __init__(self, events):
        self.events = events
        self.fail_suspend = False

    async def synchronize_policy(self, version):
        self.events.append(("synchronize", version))

    async def suspend_for_update(self):
        self.events.append(("suspend",))
        if self.fail_suspend:
            raise RuntimeError("engine did not drain")

    async def aclose(self):
        self.events.append(("session-close",))


class Endpoint:
    base_url = "http://127.0.0.1:43124/v1"

    def __init__(self, events):
        self.events = events
        self.fatal_error = None

    async def start(self, session):
        del session
        self.events.append(("endpoint-start",))

    async def open_admission(self, collection):
        self.events.append(("endpoint-open", collection.collection_id))

    async def stop_admission(self, collection):
        self.events.append(("endpoint-stop", collection.collection_id))

    async def aclose(self):
        self.events.append(("endpoint-close",))


class Workers:
    def __init__(self, events):
        self.events = events
        self.tasks = {}
        self.block = set()
        self.fail = set()

    async def start(self, activation, client_config, execution_config):
        self.events.append(
            (
                "workers-start",
                activation["id"],
                client_config.base_url,
                execution_config.episode_capacity,
            )
        )

    async def open_admission(self, collection):
        self.events.append(("workers-open", collection.collection_id))

    async def stop_admission(self, collection):
        self.events.append(("workers-stop", collection.collection_id))

    async def run_episode(self, key, task, deadline):
        del task, deadline
        self.tasks[key] = asyncio.current_task()
        self.events.append(("episode", key.occurrence_id))
        if key.occurrence_id in self.fail:
            raise CollectionExecutionError("broker failed")
        if key.occurrence_id in self.block:
            await asyncio.Event().wait()
        return completed(key)

    async def cancel(self, key):
        self.events.append(("cancel", key.occurrence_id))
        task = self.tasks.get(key)
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def aclose(self):
        self.events.append(("workers-close",))


def runner(events):
    session = Session(events)
    endpoint = Endpoint(events)
    workers = Workers(events)
    value = TrlCollectionRunner(
        session=session,
        endpoint=endpoint,
        workers=workers,
        client_config_factory=lambda base_url: SimpleNamespace(base_url=base_url),
        execution_config=RolloutExecutionConfig(env_workers=2, episodes_per_worker=2),
    )
    return value, session, endpoint, workers


@pytest.mark.asyncio
async def test_collection_returns_only_after_terminal_outcomes_and_safe_handoff():
    events = []
    value, _session, _endpoint, _workers = runner(events)
    await value.start({"id": "test-env"})
    collection = CollectionKey("run-1", "collection-1", "policy-7")
    scheduled = tuple(
        ScheduledEpisode(
            episode_key(collection, ordinal),
            SimpleNamespace(ordinal=ordinal),
            asyncio.get_running_loop().time() + 5,
        )
        for ordinal in (1, 2)
    )
    outcomes = await value.collect(collection, scheduled)
    assert [outcome.key.occurrence_id for outcome in outcomes] == ["rollout-1", "rollout-2"]
    assert events == [
        ("endpoint-start",),
        ("workers-start", "test-env", Endpoint.base_url, 4),
        ("synchronize", "policy-7"),
        ("endpoint-open", "collection-1"),
        ("workers-open", "collection-1"),
        ("episode", "rollout-1"),
        ("episode", "rollout-2"),
        ("workers-stop", "collection-1"),
        ("endpoint-stop", "collection-1"),
        ("suspend",),
    ]
    await value.aclose()
    assert events[-3:] == [("workers-close",), ("endpoint-close",), ("session-close",)]


@pytest.mark.asyncio
async def test_collection_failure_cancels_siblings_before_suspending_policy():
    events = []
    value, _session, _endpoint, workers = runner(events)
    await value.start({"id": "test-env"})
    collection = CollectionKey("run-1", "collection-1", "policy-7")
    workers.fail.add("rollout-1")
    workers.block.add("rollout-2")
    scheduled = tuple(
        ScheduledEpisode(
            episode_key(collection, ordinal),
            SimpleNamespace(ordinal=ordinal),
            asyncio.get_running_loop().time() + 5,
        )
        for ordinal in (1, 2)
    )
    try:
        with pytest.raises(CollectionExecutionError, match="broker failed"):
            await value.collect(collection, scheduled)
        assert ("cancel", "rollout-2") in events
        assert events[-3:] == [
            ("workers-stop", "collection-1"),
            ("endpoint-stop", "collection-1"),
            ("suspend",),
        ]
    finally:
        await value.aclose()


@pytest.mark.asyncio
async def test_failed_inference_suspend_blocks_optimizer_handoff():
    events = []
    value, session, _endpoint, _workers = runner(events)
    await value.start({"id": "test-env"})
    session.fail_suspend = True
    collection = CollectionKey("run-1", "collection-1", "policy-7")
    scheduled = (
        ScheduledEpisode(
            episode_key(collection, 1),
            SimpleNamespace(ordinal=1),
            asyncio.get_running_loop().time() + 5,
        ),
    )
    try:
        with pytest.raises(CollectionExecutionError, match="safe optimizer handoff"):
            await value.collect(collection, scheduled)
    finally:
        await value.aclose()


@pytest.mark.asyncio
async def test_native_collector_preserves_batch_identity_and_partial_failures():
    class Bridge:
        run_id = "run-1"

        def task_for_example_id(self, example_id):
            return SimpleNamespace(example_id=example_id)

    class NativeRunner:
        def __init__(self):
            self.scheduled = ()

        async def collect(self, collection, scheduled):
            self.scheduled = scheduled
            assert collection.logical_step == 4
            return (
                completed(scheduled[0].key),
                EpisodeOutcome(
                    key=scheduled[1].key,
                    status=EpisodeStatus.INVALID,
                    error="trace is not trainable",
                ),
            )

    native_runner = NativeRunner()
    collector = TrlNativeRolloutCollector(
        runner=cast(Any, native_runner),
        bridge=Bridge(),
        episode_timeout=30,
    )
    batch = RolloutBatch(
        example_ids=("task-1", "task-2"),
        step=4,
        model_id="policy",
        prompt_group_ids=("group-1", "group-2"),
        rollout_ids=("rollout-1", "rollout-2"),
    )
    with pytest.raises(VerifiersRolloutFailure) as captured:
        await collector.collect(batch, collection_id="collection-4", policy_version="policy-3")
    assert captured.value.completed[0].example_id == "task-1"
    assert captured.value.failures == {1: "trace is not trainable"}
    assert [item.key.rollout_ordinal for item in native_runner.scheduled] == [0, 1]
    assert [item.key.group_id for item in native_runner.scheduled] == ["group-1", "group-2"]
    assert [item.key.occurrence_id for item in native_runner.scheduled] == ["rollout-1", "rollout-2"]
    assert native_runner.scheduled[0].key.seed != native_runner.scheduled[1].key.seed


@pytest.mark.asyncio
async def test_native_collector_returns_completed_rollouts_in_source_order():
    class Bridge:
        run_id = "run-1"

        def task_for_example_id(self, example_id):
            return SimpleNamespace(example_id=example_id)

    class ReorderingRunner:
        async def collect(self, _collection, scheduled):
            await asyncio.sleep(0)
            return tuple(completed(item.key) for item in scheduled)

    collector = TrlNativeRolloutCollector(
        runner=cast(Any, ReorderingRunner()),
        bridge=Bridge(),
        episode_timeout=30,
    )
    batch = RolloutBatch(("task-1", "task-2"), 1, "policy")
    rollouts = await collector.collect(batch, collection_id="collection-1", policy_version="policy-0")
    assert [rollout.example_id for rollout in rollouts] == ["task-1", "task-2"]
