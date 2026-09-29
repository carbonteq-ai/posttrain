"""The adaptive curriculum makes the same decisions on TRL's and veRL's active sampling.

TRL side: Posttrain's ``_prepare_adaptive_active_sampling_inputs`` (the TRL
backend's curriculum-driven OLMo 3 refill loop) on a stub trainer that uses
TRL post11's real row selection and concatenation. veRL side: the CarbonTeq
fork's ``ActiveSamplingReplayBuffer`` over a real TransferQueue, dispatching
through ``PosttrainCurriculumSelector``. Both see the same deterministic
rewards per task and occurrence over several updates; the curriculum's
decision records, observations and final controller state must be identical.
Needs the parity environment of ``docs/plan/verl-vortex-port.md``.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl.trainer.grpo_trainer", reason="requires CarbonTeq TRL 1.12.0.post11")
replay = pytest.importorskip("verl.trainer.ppo.v1.replay_buffer", reason="requires the CarbonTeq veRL fork")
if not hasattr(replay, "ActiveSamplingReplayBuffer"):
    pytest.skip("installed veRL has no active sampling", allow_module_level=True)
tq = pytest.importorskip("transfer_queue")

from posttrain.train.adaptive_curriculum_runtime import AdaptiveCurriculumRuntime  # noqa: E402
from posttrain.train.backends.trl.policy_curriculum import _prepare_adaptive_active_sampling_inputs  # noqa: E402
from posttrain.train.backends.verl.curriculum import PosttrainCurriculumSelector, SelectorConfig  # noqa: E402
from posttrain.train.profiles import AdaptiveCurriculum  # noqa: E402

GENERATIONS = 2
TARGET = 3
MAX_BATCHES = 4
STEPS = 4
TASKS = [{"example_id": f"task/{index:02d}", "domain": ("mail", "crm", "sheets")[index % 3]} for index in range(24)]


def _rewards(task_id: str, occurrence: int) -> list[float]:
    """Deterministic group rewards: some tasks never vary, others vary on some occurrences."""
    index = int(task_id.split("/")[1])
    if index % 4 == 0:
        return [1.0, 1.0]
    if index % 4 == 1:
        return [0.0, 1.0]
    if index % 4 == 2:
        return [0.0, 0.0] if occurrence % 2 == 0 else [0.25, 1.0]
    return [0.0, 0.75] if occurrence < 2 else [0.5, 0.5]


class _Context:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def event(self, name, attributes=None):
        self.events.append((name, json.loads(json.dumps(dict(attributes or {})))))

    def metrics(self, values, *, step, attributes=None):
        pass

    def metric(self, name, value, *, step, attributes=None):
        pass


class _Accelerator:
    device = torch.device("cpu")
    num_processes = 1
    process_index = 0

    def gather(self, value):
        return value.reshape(1) if value.ndim == 0 else value


def _settings() -> AdaptiveCurriculum:
    return AdaptiveCurriculum(class_field="domain", policy="yield_first", seed=11, history_groups=3)


def _trl_run(tmp_path: Path):
    from trl.trainer.grpo_trainer import GRPOTrainer

    context = _Context()
    runtime = AdaptiveCurriculumRuntime(
        cast(Any, context),
        TASKS,
        _settings(),
        num_generations=GENERATIONS,
        state_dir=tmp_path / "trl-state",
        resume_checkpoint=None,
    )
    occurrences: Counter[str] = Counter()
    trainer = SimpleNamespace(
        active_sampling_max_batches=MAX_BATCHES,
        active_sampling_oversample=1,
        active_sampling_oversample_refill=1,
        active_sampling_reward_std_epsilon=0.0,
        num_generations=GENERATIONS,
        accelerator=_Accelerator(),
        _metrics={"train": defaultdict(list)},
        _posttrain_native_group_reward_std=None,
        _select_dynamic_sampling_rows=GRPOTrainer._select_dynamic_sampling_rows,
    )
    trainer._concatenate_dynamic_sampling_batches = lambda batches: GRPOTrainer._concatenate_dynamic_sampling_batches(
        cast(Any, trainer), batches
    )
    step_holder = {"step": 0}

    def generate(candidate_batch):
        rewards: list[float] = []
        for offset in range(0, len(candidate_batch), GENERATIONS):
            task_id = str(candidate_batch[offset]["example_id"])
            rewards.extend(_rewards(task_id, occurrences[task_id]))
            occurrences[task_id] += 1
        # TRL's _calculate_rewards hook: the curriculum observes every scored group.
        runtime.observe_rewards(candidate_batch, [[value] for value in rewards], [1.0], step=step_holder["step"])
        grouped = torch.tensor(rewards).view(-1, GENERATIONS)
        std = grouped.std(dim=1).repeat_interleave(GENERATIONS)
        rows = len(candidate_batch)
        return {
            "example_id": [row["example_id"] for row in candidate_batch],
            "completion_ids": torch.ones(rows, 3, dtype=torch.long),
            "completion_mask": torch.ones(rows, 3),
            "group_reward_std": std,
        }

    trainer._generate_and_score_completions = generate
    kept: list[list[str]] = []
    for step in range(1, STEPS + 1):
        step_holder["step"] = step
        candidates = [{} for _ in range(TARGET * MAX_BATCHES * GENERATIONS)]
        batch = _prepare_adaptive_active_sampling_inputs(trainer, candidates, runtime, step=step)
        kept.append(list(batch["example_id"][::GENERATIONS]))
    state = runtime.controller.state()
    runtime.close()
    return context.events, kept, state


def _verl_run(tmp_path: Path):
    output = tmp_path / "verl"
    output.mkdir()
    SelectorConfig(
        settings={"class_field": "domain", "policy": "yield_first", "seed": 11, "history_groups": 3},
        num_generations=GENERATIONS,
        state_dir=output / "state",
        journal_path=output / "journal.jsonl",
    ).write(output / "selector.json")
    selector = PosttrainCurriculumSelector(TASKS, config_path=str(output / "selector.json"))
    partition = f"curriculum-{uuid.uuid4().hex}"
    occurrences: Counter[str] = Counter()
    uid_index: dict[str, int] = {}
    step_holder = {"step": 0}

    def dispatch(count: int, round_index: int | None = None) -> list[str]:
        indices = selector.select(
            count, global_steps=step_holder["step"], stage="active_sampling_refill", round_index=round_index
        )
        uids = []
        for index in indices:
            task_id = str(TASKS[index]["example_id"])
            rewards = _rewards(task_id, occurrences[task_id])
            occurrences[task_id] += 1
            uid = uuid.uuid4().hex
            uid_index[uid] = index
            for session, reward in enumerate(rewards):
                tq.kv_put(
                    key=f"{uid}_{session}_0",
                    partition_id=partition,
                    fields={
                        "input_ids": torch.tensor([1, 2, 3]),
                        "extra_fields": {"reward_extra_info": {"seq_reward": reward}},
                    },
                    tag={"is_prompt": False, "seq_len": 3, "global_steps": step_holder["step"]},
                )
            tq.kv_put(
                key=uid,
                partition_id=partition,
                tag={"is_prompt": True, "status": "finished", "global_steps": step_holder["step"]},
            )
            uids.append(uid)
        return uids

    def observe(groups):
        selector.observe([(uid_index[uid], values) for uid, values in groups], global_steps=step_holder["step"])

    buffer = replay.ActiveSamplingReplayBuffer(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=1,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0.01,
        refill_fn=lambda count: count,
        train_batch_size=TARGET,
        gen_batch_size=1,
        dispatch_fn=dispatch,
        observe_fn=observe,
        active_max_rounds=MAX_BATCHES,
        active_oversample=1,
        active_oversample_refill=1,
    )
    kept: list[list[str]] = []
    try:
        for step in range(1, STEPS + 1):
            step_holder["step"] = step
            batch, _ = buffer.sample(global_steps=step, partition_id=partition, batch_size=TARGET)
            uids = list(dict.fromkeys(key.split("_")[0] for key in batch.keys))
            order = {uid: position for position, uid in enumerate(uid_index)}
            kept.append([str(TASKS[uid_index[uid]]["example_id"]) for uid in sorted(uids, key=order.__getitem__)])
            tq.kv_clear(keys=list(batch.keys), partition_id=partition)
    finally:
        keys = list(tq.kv_list(partition_id=partition).get(partition, {}).keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition)
    assert selector.runtime is not None
    state = selector.runtime.controller.state()
    selector.close()
    events = [
        (record["name"], record["attributes"])
        for record in map(json.loads, (output / "journal.jsonl").read_text().splitlines())
        if record["kind"] == "event"
    ]
    return events, kept, state


@pytest.fixture(scope="module")
def transfer_queue():
    tq.init()
    yield
    tq.close()


def _comparable(events):
    return [(name, attributes) for name, attributes in events if name != "adaptive_curriculum_checkpointed"]


def test_curriculum_decisions_and_state_match_trl(transfer_queue, tmp_path: Path) -> None:
    trl_events, trl_kept, trl_state = _trl_run(tmp_path)
    verl_events, verl_kept, verl_state = _verl_run(tmp_path)

    decisions = [attributes for name, attributes in trl_events if name == "adaptive_curriculum_allocation_selected"]
    assert len(decisions) > STEPS  # some updates needed refill rounds
    assert any((decision.get("round_index") or 0) > 1 for decision in decisions)
    assert _comparable(verl_events) == _comparable(trl_events)
    assert verl_kept == trl_kept
    assert json.dumps(verl_state, sort_keys=True, default=str) == json.dumps(trl_state, sort_keys=True, default=str)
