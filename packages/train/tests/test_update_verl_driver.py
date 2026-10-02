"""Native transfer metadata and RPC lifecycle; not GPU driver qualification."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.train.backends.verl.policy_driver import (
    ResolvedVeRLDriverSteps,
    native_candidate_tasks,
    native_receipts,
    single_actor_result,
)
from posttrain.train.update_records import InvalidPolicyUpdate

tq = pytest.importorskip("transfer_queue")


class Driver(ResolvedVeRLDriverSteps):
    _resolved_active_candidates: int | None = None

    def __init__(self):
        self.global_steps = 1
        self.resolved_settings = SimpleNamespace(num_prompts_per_step=1, num_generations=2)
        self.applied = 0
        self.events = []
        self.actor_rollout_wg = SimpleNamespace(resolved_state=self.state, resolved_update=self.update)
        self.replay_buffer = SimpleNamespace(sample=self.sample)
        self.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: self.events.append("sleep"))
        self.on_sample_begin = lambda: self.events.append(("begin", self.global_steps))
        self.on_sample_end = lambda: self.events.append(("end", self.global_steps))
        self._add_batch_to_generate = lambda: self.events.append(("generate", self.global_steps))
        self._consume_rollout_metrics = lambda: {"rollout/metric": 7}

    def state(self):
        return [{"applied": self.applied, "needs_population": self.applied % 2 == 0}]

    def update(self, receipts, expected_applied):
        assert expected_applied == self.applied
        assert receipts == (("receipt-a", "receipt-b") if self.applied % 2 == 0 else None)
        self.events.append(("update", expected_applied))
        self.applied += 1
        return [{"train/rl/applied_optimizer_updates": self.applied, "train/rl/policy_loss": 0.2}]

    def sample(self, **kwargs):
        assert kwargs == {"global_steps": self.applied, "partition_id": "train", "batch_size": 1}
        return tq.KVBatchMeta(partition_id="train", keys=["a", "b"], tags=[{}, {}]), {}


def test_native_driver_collects_at_actor_version_and_reuses_without_flattening(monkeypatch):
    calls = []

    def read(**kwargs):
        calls.append(kwargs)
        return {
            "extra_fields": [
                SimpleNamespace(data={"posttrain_native_episode_receipt": value})
                for value in ("receipt-a", "receipt-b")
            ]
        }

    monkeypatch.setattr(tq, "kv_batch_get", read)
    driver = Driver()
    for step in range(1, 5):
        driver.global_steps = step
        metrics = {}
        batch = driver.step(metrics, {})
        assert batch.keys == (["a", "b"] if step % 2 else [])
        assert driver.global_steps == step == driver.applied
        assert metrics["train/rl/policy_loss"] == 0.2
        driver._compute_metrics(batch, metrics, {"update": 0.1}, step, 0)
        assert metrics["timing_s/update"] == 0.1
    assert len(calls) == 2
    assert all(call["select_fields"] == ["extra_fields"] for call in calls)
    assert [event for event in driver.events if isinstance(event, tuple) and event[0] == "generate"] == [
        ("generate", 0),
        ("generate", 2),
    ]
    assert driver.events.count("sleep") == 2


def test_native_collection_failure_restores_pending_commit(monkeypatch):
    driver = Driver()

    def fail(**kwargs):
        raise RuntimeError("collection interrupted")

    driver.replay_buffer.sample = fail
    with pytest.raises(RuntimeError, match="interrupted"):
        driver.step({}, {})
    assert driver.global_steps == 1 and driver.applied == 0
    assert not any(isinstance(event, tuple) and event[0] == "update" for event in driver.events)


@pytest.mark.parametrize("results", [[], {}, [None]])
def test_driver_rejects_unsupported_actor_dispatch(results):
    with pytest.raises(InvalidPolicyUpdate, match="requires native actor results"):
        single_actor_result(results)


def test_driver_accepts_identical_data_parallel_ranks_and_rejects_divergence():
    state = {"applied": 1, "attempts": 1, "needs_population": False}
    assert single_actor_result([state, dict(state)]) == state
    with pytest.raises(InvalidPolicyUpdate, match="different global state"):
        single_actor_result([state, {**state, "applied": 2}])


@pytest.mark.parametrize("fields", [[], [{}], [{"posttrain_native_episode_receipt": ""}]])
def test_driver_requires_complete_original_receipts(fields):
    with pytest.raises(InvalidPolicyUpdate):
        native_receipts({"extra_fields": fields}, 1)


def test_driver_rejects_stale_commit_before_collection():
    driver = Driver()
    driver.global_steps = 2
    with pytest.raises(InvalidPolicyUpdate, match="committed actor version"):
        driver.step({}, {})
    assert driver.events == []


def test_native_grpo_trainer_preserves_complete_groups_and_rejects_filters():
    pytest.importorskip("verl")
    from hydra import compose, initialize_config_module
    from posttrain.train.backends.verl.policy_driver import resolved_trainer_type
    from posttrain.train.profiles import GRPOSettings, TrainingLoop
    from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings
    from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

    settings = GRPOSettings(
        id="driver",
        loop=TrainingLoop(max_steps=4, per_device_batch_size=1),
        policy_updates=PolicyUpdateSettings(PolicyUpdateSchedule("episode", 2), PolicyExecutionBudget(1, 100, 10000)),
    )
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adv_estimator=grpo",
                "trainer.total_training_steps=4",
                f"data.train_batch_size={settings.num_prompts_per_step}",
                f"actor_rollout_ref.rollout.n={settings.num_generations}",
            ],
        )
    trainer_cls = resolved_trainer_type(type("Actor", (), {}), settings)
    trainer = trainer_cls(config)
    assert isinstance(trainer, PPOTrainerSync)
    assert type(trainer.replay_buffer) is ReplayBuffer
    assert not trainer.replay_buffer.refill_all_failed_groups
    assert trainer.replay_buffer.filter_groups_metric is None
    assert trainer.replay_buffer.refill_fn is None
    for key, value in (("filter_groups", True), ("active_sampling", True)):
        config.algorithm[key].enable = value
        with pytest.raises(InvalidPolicyUpdate, match="collection semantics"):
            trainer_cls(config)
        config.algorithm[key].enable = False


def _native_sampo_fixture():
    pytest.importorskip("verl")
    from hydra import compose, initialize_config_module
    from posttrain.train.backends.verl.policy_driver import resolved_trainer_type
    from posttrain.train.backends.verl.worker import _active_sampling_hydra_overrides
    from posttrain.train.profiles import SAMPOSettings, TrainingLoop
    from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings

    settings = SAMPOSettings(
        id="driver",
        loop=TrainingLoop(max_steps=4, per_device_batch_size=1),
        policy_updates=PolicyUpdateSettings(PolicyUpdateSchedule("episode", 2), PolicyExecutionBudget(1, 100, 10000)),
    )
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        # Compose the worker's own active-sampling overrides so the driver is
        # checked against the metric/epsilon the launch path actually emits.
        algorithm = SimpleNamespace(
            active_sampling_max_candidate_batches=3,
            active_sampling_oversample=0,
            active_sampling_oversample_refill=0,
            adaptive_curriculum=None,
        )
        active = _active_sampling_hydra_overrides(
            cast(Any, SimpleNamespace(payload=SimpleNamespace(algorithm=algorithm)))
        )
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adv_estimator=sampo",
                *active,
                "trainer.total_training_steps=4",
                f"data.train_batch_size={settings.num_prompts_per_step}",
                f"actor_rollout_ref.rollout.n={settings.num_generations}",
            ],
        )
    trainer_cls = resolved_trainer_type(type("Actor", (), {}), settings)
    return trainer_cls, config


def test_native_sampo_uses_existing_active_buffer_and_selected_reservation():
    from verl.trainer.ppo.v1.replay_buffer import ActiveSamplingReplayBuffer

    trainer_cls, config = _native_sampo_fixture()
    trainer = trainer_cls(config)
    assert type(trainer.replay_buffer) is ActiveSamplingReplayBuffer
    assert trainer.replay_buffer.dispatch_fn == trainer._dispatch_reserved
    assert trainer.replay_buffer.observe_fn == trainer._observe_prompt_groups
    assert trainer._resolved_active_candidates == 3
    config.algorithm.active_sampling.enable = False
    with pytest.raises(InvalidPolicyUpdate, match="collection semantics"):
        trainer_cls(config)
    config.algorithm.active_sampling.enable = True
    config.algorithm.active_sampling.max_candidate_batches = 4
    with pytest.raises(InvalidPolicyUpdate, match="selected candidate reservation"):
        trainer_cls(config)


def test_native_active_rounds_dispatch_reserved_tasks_and_evict_discarded_group(monkeypatch):
    """Native buffer/control flow over an in-memory TQ transport, not Ray qualification."""
    import numpy as np
    from verl.utils.tensordict_utils import get_tensordict

    trainer_cls, config = _native_sampo_fixture()
    trainer = trainer_cls(config)
    trainer.global_steps = 1
    trainer.prompt_selector = None
    trainer.on_sample_begin = trainer.on_sample_end = lambda: None
    trainer._consume_rollout_metrics = lambda: {}
    trainer._add_batch_to_generate = lambda: pytest.fail("active sampler received duplicate initial dispatch")
    pool = get_tensordict(
        {"example_id": np.asarray(["a", "b", "c"], dtype=object), "uid": np.asarray(["a", "b", "c"], dtype=object)}
    )
    reservations, dispatches, tags, fields = [], [], {}, {}

    def reserve(count):
        reservations.append(count)
        return pool

    trainer._next_train_batch = reserve

    def submit(batch):
        uids = [str(value) for value in batch["uid"]]
        dispatches.append(uids)
        for uid in uids:
            tags[uid] = {"is_prompt": True, "status": "finished", "global_steps": 0}
            for index in range(2):
                key = f"{uid}_{index}_0"
                tags[key] = {"is_prompt": False, "seq_len": 3, "global_steps": 0}
                fields[key] = {
                    "reward_extra_info": {"group_reward": index if uid == "b" else 0},
                    "posttrain_native_episode_receipt": f"{uid}{index}",
                }
        return len(batch)

    def clear(*, partition_id, keys):
        assert partition_id == "train"
        for key in keys:
            tags.pop(key, None)
            fields.pop(key, None)

    trainer._submit_batch_to_rollout = submit
    monkeypatch.setattr(tq, "kv_list", lambda: {"train": dict(tags)})
    monkeypatch.setattr(tq, "kv_clear", clear)
    monkeypatch.setattr(
        tq, "kv_batch_get", lambda *, keys, partition_id, select_fields: {"extra_fields": [fields[key] for key in keys]}
    )
    commits = []

    def update(receipts, *, expected_applied):
        assert expected_applied == 0 and receipts == ("b0", "b1")
        commits.append(receipts)
        return {"train/rl/applied_optimizer_updates": 1}

    trainer.actor_rollout_wg = SimpleNamespace(
        resolved_state=lambda: [{"applied": 0, "needs_population": True}],
        resolved_update=lambda *args, **kwargs: [update(*args, **kwargs)],
    )
    metrics = {}
    batch = trainer.step(metrics, {})
    assert reservations == [3] and dispatches == [["a"], ["b"]]
    assert trainer._resolved_candidate_tasks == ("a", "b", "c")
    assert batch.keys == ["b_0_0", "b_1_0"] and commits == [("b0", "b1")]
    assert set(tags) == set(batch.keys)  # rejected group removed; unused candidate never dispatched
    assert trainer.global_steps == 1


def test_active_driver_reserves_once_without_extra_initial_dispatch(monkeypatch):
    monkeypatch.setattr(
        tq,
        "kv_batch_get",
        lambda **kwargs: {
            "extra_fields": [{"posttrain_native_episode_receipt": value} for value in ("receipt-a", "receipt-b")]
        },
    )
    driver = Driver()
    driver._resolved_active_candidates = 3
    driver._candidate_pool = {"example_id": ["a", "b", "c"]}
    driver._reserve_candidates = lambda count: driver.events.append(("reserve", count, driver.global_steps))
    driver.step({}, {})
    assert driver._resolved_candidate_tasks == ("a", "b", "c")
    assert ("reserve", 3, 0) in driver.events
    assert not any(isinstance(event, tuple) and event[0] == "generate" for event in driver.events)


def test_active_driver_rejects_wrapped_inventory_before_any_rollout():
    driver = Driver()
    driver._resolved_active_candidates = 3
    driver._candidate_pool = {"example_id": ["a", "b", "a"]}
    driver._reserve_candidates = lambda count: None
    driver.replay_buffer.sample = lambda **kwargs: pytest.fail("duplicate task pool reached native sampling")
    with pytest.raises(InvalidPolicyUpdate, match="distinct complete task inventory"):
        driver.step({}, {})
    assert driver.global_steps == 1 and driver.applied == 0


def test_reserved_tasks_read_real_native_tensordict_wrappers():
    import numpy as np
    from verl.utils.tensordict_utils import get_tensordict

    batch = get_tensordict({"example_id": np.asarray(["a", "b", "c"], dtype=object)})
    assert native_candidate_tasks(batch, 3) == ("a", "b", "c")


def test_native_rpc_factory_retains_engine_identity_and_rejects_legacy_updates():
    pytest.importorskip("verl")
    from posttrain.train.backends.verl.resolved_workers import resolved_actor_worker_type

    engine = object()
    events = []

    class TrainingWorker:
        def create_engine(self):
            return engine

    class NativeActor:
        def init_model(self):
            events.append("native_init")
            self.actor = SimpleNamespace(engine=engine)

    session = SimpleNamespace(
        state=lambda: {"applied": 0},
        update=lambda receipts, expected_applied: {"applied": expected_applied + 1},
        save_checkpoint=lambda path, expected_applied: (path, expected_applied),
    )

    def create(actual_engine, worker):
        assert actual_engine is engine and worker.actor.engine is engine
        events.append("session_init")
        return session

    actor_cls = resolved_actor_worker_type(TrainingWorker, NativeActor, None, create)
    actor = actor_cls()
    actor.init_model()
    assert events == ["native_init", "session_init"]
    assert actor.resolved_state() == {"applied": 0}
    assert actor.resolved_update(None, 0) == {"applied": 1}
    with pytest.raises(InvalidPolicyUpdate, match="legacy flattened"):
        actor.update_actor(object())
    with pytest.raises(InvalidPolicyUpdate, match="sealed driver checkpoint owner"):
        actor.save_checkpoint("/unused", max_ckpt_to_keep=2)


def test_native_driver_saves_and_restores_dataloader_only_after_job_seal(tmp_path):
    pytest.importorskip("verl")
    from pathlib import Path

    from hydra import compose, initialize_config_module
    from posttrain.train.backends.verl.policy_checkpoint import inspect_driver_checkpoint
    from posttrain.train.backends.verl.policy_driver import resolved_trainer_type
    from posttrain.train.profiles import SAMPOSettings, TrainingLoop
    from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings

    from .test_update_verl_checkpoint import actor_checkpoint

    settings = SAMPOSettings(
        id="driver",
        loop=TrainingLoop(max_steps=4, per_device_batch_size=1),
        policy_updates=PolicyUpdateSettings(PolicyUpdateSchedule("episode", 2), PolicyExecutionBudget(1, 100, 10000)),
    )
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adv_estimator=sampo",
                "trainer.total_training_steps=4",
                f"data.train_batch_size={settings.num_prompts_per_step}",
                f"actor_rollout_ref.rollout.n={settings.num_generations}",
            ],
        )
    config.trainer.default_local_dir = str(tmp_path)
    config.algorithm.active_sampling.enable = True
    config.algorithm.active_sampling.max_candidate_batches = settings.active_sampling.max_candidate_batches
    config.algorithm.active_sampling.metric = "group_reward"
    config.data.prompt_selector.metric = "group_reward"
    config.trainer.resume_mode = "auto"
    trainer_cls = resolved_trainer_type(type("Actor", (), {}), settings)
    calls = []

    def save(local_path, remote_path, step, max_ckpt_to_keep):
        assert remote_path is None and step == 1 and max_ckpt_to_keep is None
        actor_checkpoint(Path(local_path).parent)
        calls.append("actor-save")

    trainer = trainer_cls(config)
    trainer.global_steps = 1
    trainer.prompt_selector = None
    trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=save, resolved_state=lambda: [{"applied": 1}])
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {"position": 3})
    trainer._save_checkpoint()
    checkpoint = tmp_path / "global_step_1"
    assert inspect_driver_checkpoint(checkpoint).native_applied_updates == 1
    trainer._save_checkpoint()
    assert calls == ["actor-save"]

    restored = trainer_cls(config)
    restored.prompt_selector = None
    restored.actor_rollout_wg = SimpleNamespace(
        load_checkpoint=lambda **kwargs: calls.append("actor-load"), resolved_state=lambda: [{"applied": 1}]
    )
    restored.train_dataloader = SimpleNamespace(load_state_dict=lambda state: calls.append(state))
    restored._load_checkpoint()
    assert restored.global_steps == 1
    assert calls[-2:] == ["actor-load", {"position": 3}]
    assert config.trainer.resume_mode == "auto"
    calls.clear()
    (checkpoint / "data.pt").write_bytes(b"corrupt")
    with pytest.raises(InvalidPolicyUpdate, match="bytes changed"):
        restored._load_checkpoint()
    assert calls == []
