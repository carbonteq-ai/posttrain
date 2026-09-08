import asyncio
from types import SimpleNamespace

import pytest
from posttrain.common import TraceObservation
from posttrain.train.integrations.verifiers_async_groups import VerifiersAsyncGroupProducer
from posttrain.train.online_rl import EnvironmentRollout
from posttrain.train.profiles import GRPOSettings, TrainingLoop
from posttrain.train.rollout_execution import (
    CollectionExecutionError,
    EpisodeOutcome,
    EpisodeStatus,
    RolloutExecutionConfig,
    RolloutGroupRejected,
)


def settings(**overrides):
    values = {
        "id": "async-grpo-test",
        "loop": TrainingLoop(max_steps=2, per_device_batch_size=2),
        "num_prompts_per_step": 1,
        "num_generations": 2,
    }
    values.update(overrides)
    return GRPOSettings(**values)


def rollout(key, reward):
    return EnvironmentRollout(
        example_id=key.example_id,
        prompt_ids=(1, 2),
        completion_ids=(3, 4),
        sampling_logprobs=(-0.1, -0.2),
        env_mask=(True, True),
        reward=reward,
        is_truncated=False,
        trace=TraceObservation("verifiers", key.occurrence_id, {}),
    )


class PolicyAdmission:
    base_url = "http://127.0.0.1:8000/v1"

    def __init__(self):
        self.events = []

    async def start(self, initial_model_version):
        self.events.append(("start", initial_model_version))

    async def prepare_model_update(self, model_version):
        self.events.append(("prepare", model_version))

    async def activate_model_version(self, model_version):
        self.events.append(("activate", model_version))

    async def aclose(self):
        self.events.append(("close",))


class Workers:
    def __init__(self):
        self.events = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False
        self.invalid_ordinal: int | None = None
        self.failing_ordinal: int | None = None
        self.tasks = {}
        self.cancelled = []

    async def start(self, activation, client_config, execution_config):
        self.events.append(("start", activation["id"], client_config.base_url, execution_config.episode_capacity))

    async def open_run_admission(self, run_id):
        self.events.append(("open", run_id))

    async def stop_run_admission(self, run_id):
        self.events.append(("stop", run_id))

    async def run_episode(self, key, task, deadline):
        del task, deadline
        self.tasks[key] = asyncio.current_task()
        self.entered.set()
        if key.rollout_ordinal == self.failing_ordinal:
            await asyncio.sleep(0)
            raise CollectionExecutionError("worker broker failed")
        if self.block:
            await self.release.wait()
        if key.rollout_ordinal == self.invalid_ordinal:
            return EpisodeOutcome(key, EpisodeStatus.INVALID, error="invalid trace")
        return EpisodeOutcome(key, EpisodeStatus.COMPLETED, rollout=rollout(key, float(key.rollout_ordinal)))

    async def cancel(self, key):
        self.cancelled.append(key)
        task = self.tasks.get(key)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def aclose(self):
        self.events.append(("close",))


class Bridge:
    dataset = SimpleNamespace(examples=(SimpleNamespace(id="task-a"), SimpleNamespace(id="task-b")))

    def task_for_example_id(self, example_id):
        return SimpleNamespace(id=example_id)


def producer(workers=None, policy=None, *, selected_settings=None, max_group_tokens=32):
    workers = workers or Workers()
    policy = policy or PolicyAdmission()
    value = VerifiersAsyncGroupProducer(
        workers=workers,
        policy_admission=policy,
        bridge=Bridge(),
        activation={"id": "automationbench"},
        client_config_factory=lambda base_url: SimpleNamespace(base_url=base_url),
        execution_config=RolloutExecutionConfig(env_workers=1, episodes_per_worker=2),
        settings=selected_settings or settings(),
        run_id="run-1",
        initial_model_version=0,
        episode_timeout_s=5,
        max_group_tokens=max_group_tokens,
        seed=7,
    )
    return value, workers, policy


@pytest.mark.asyncio
async def test_produces_complete_native_group_and_tracks_policy_span_across_update():
    workers = Workers()
    workers.block = True
    value, _workers, policy = producer(workers)
    await value.astart()
    pending = asyncio.create_task(value.produce_group(0))
    await workers.entered.wait()
    await value.prepare_model_update(1)
    await value.activate_model_version(1)
    workers.release.set()

    samples = await pending

    assert len(samples) == 2
    assert [sample.group_id for sample in samples] == [0, 0]
    assert [sample.model_version for sample in samples] == [0, 0]
    assert samples[0].advantage == pytest.approx(-1.0)
    assert samples[1].advantage == pytest.approx(1.0)
    assert samples[0].metrics == {
        "reward": 0.0,
        "task_reward": 0.0,
        "behavior_policy_start": 0.0,
        "behavior_policy_end": 1.0,
    }
    assert policy.events[:3] == [("start", 0), ("prepare", 1), ("activate", 1)]
    await value.aclose()


@pytest.mark.asyncio
async def test_rejects_incomplete_native_group_without_publishing_partial_samples():
    workers = Workers()
    workers.invalid_ordinal = 1
    value, _, _ = producer(workers)
    await value.astart()

    with pytest.raises(RolloutGroupRejected, match="1 of 2"):
        await value.produce_group(0)

    await value.aclose()


@pytest.mark.asyncio
async def test_rejects_group_over_serialized_token_bound():
    value, _, _ = producer(max_group_tokens=7)
    await value.astart()

    with pytest.raises(RolloutGroupRejected, match="8 tokens"):
        await value.produce_group(0)

    await value.aclose()


@pytest.mark.asyncio
async def test_infrastructure_failure_cancels_siblings_and_remains_fatal():
    workers = Workers()
    workers.block = True
    workers.failing_ordinal = 0
    value, _, _ = producer(workers)
    await value.astart()

    with pytest.raises(CollectionExecutionError, match="worker broker failed"):
        await value.produce_group(0)

    assert {key.rollout_ordinal for key in workers.cancelled} == {0, 1}
    assert all(task.done() for task in workers.tasks.values())
    await value.aclose()


@pytest.mark.asyncio
async def test_failed_startup_closes_each_acquired_runtime_once():
    class FailingWorkers(Workers):
        async def start(self, activation, client_config, execution_config):
            await super().start(activation, client_config, execution_config)
            raise CollectionExecutionError("worker startup failed")

    workers = FailingWorkers()
    policy = PolicyAdmission()
    value, _, _ = producer(workers, policy)

    with pytest.raises(CollectionExecutionError, match="worker startup failed"):
        await value.astart()

    assert workers.events[-1] == ("close",)
    assert workers.events.count(("close",)) == 1
    assert policy.events == [("start", 0), ("close",)]
    await value.aclose()
    assert workers.events.count(("close",)) == 1


def test_rejects_algorithm_profiles_whose_credit_semantics_are_not_implemented():
    with pytest.raises(ValueError, match="GRPO only"):
        producer(selected_settings=settings(algorithm="dapo"))
    with pytest.raises(ValueError, match="group advantage scaling"):
        producer(selected_settings=settings(advantage_scaling="none"))
