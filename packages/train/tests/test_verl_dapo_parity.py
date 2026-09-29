"""TRL's DAPO dynamic sampling and the veRL fork's candidate-batch buffer make the same update.

TRL side: TRL post12's real ``GRPOTrainer._prepare_dynamic_sampling_inputs`` on a
stub trainer, fed by Posttrain's adaptive curriculum the way the TRL backend's
``AdaptiveCurriculumTrainer`` does (one ``initial_batch`` decision for the whole
candidate pool, rewards observed per scored candidate batch). Its generation
step scores each candidate batch with TRL's batch-scaled advantages (reward
minus group mean, divided by the candidate batch's ``nanstd`` + 1e-4). veRL
side: the fork's ``CandidateBatchReplayBuffer`` over a real TransferQueue with
Posttrain's curriculum selector, and the fork's GRPO estimator with the
recorded per-candidate-batch std. Over several updates both must make the same
curriculum decisions, keep the same tasks, report the same metrics and give
the same advantages. Needs the parity environment of
``docs/plan/verl-vortex-port.md``; skips elsewhere.
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
pytest.importorskip("trl.trainer.grpo_trainer", reason="requires CarbonTeq TRL 1.12.0.post12")
replay = pytest.importorskip("verl.trainer.ppo.v1.replay_buffer", reason="requires the CarbonTeq veRL fork")
if not hasattr(replay, "CandidateBatchReplayBuffer"):
    pytest.skip("installed veRL has no candidate-batch dynamic sampling", allow_module_level=True)
tq = pytest.importorskip("transfer_queue")

from posttrain.train.adaptive_curriculum_runtime import AdaptiveCurriculumRuntime  # noqa: E402
from posttrain.train.backends.verl.curriculum import PosttrainCurriculumSelector, SelectorConfig  # noqa: E402
from posttrain.train.profiles import AdaptiveCurriculum  # noqa: E402

GENERATIONS = 2
TARGET = 3
MAX_BATCHES = 4
STEPS = 3
TASKS = [{"example_id": f"task/{index:02d}", "domain": ("mail", "crm", "sheets")[index % 3]} for index in range(24)]
SETTINGS = {"class_field": "domain", "policy": "yield_first", "seed": 5, "history_groups": 3}


def _rewards(task_id: str, occurrence: int) -> list[float]:
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


def _trl_run(tmp_path: Path):
    from trl.trainer.grpo_trainer import GRPOTrainer
    from trl.trainer.utils import nanstd

    context = _Context()
    runtime = AdaptiveCurriculumRuntime(
        cast(Any, context),
        TASKS,
        AdaptiveCurriculum(**SETTINGS),
        num_generations=GENERATIONS,
        state_dir=tmp_path / "trl-state",
        resume_checkpoint=None,
    )
    occurrences: Counter[str] = Counter()
    step_holder = {"step": 0}

    def generate(candidate_batch):
        rewards: list[float] = []
        for offset in range(0, len(candidate_batch), GENERATIONS):
            task_id = str(candidate_batch[offset]["example_id"])
            rewards.extend(_rewards(task_id, occurrences[task_id]))
            occurrences[task_id] += 1
        # The TRL backend's _calculate_rewards hook observes every scored candidate batch.
        runtime.observe_rewards(candidate_batch, [[value] for value in rewards], [1.0], step=step_holder["step"])
        values = torch.tensor(rewards, dtype=torch.float64)
        grouped = values.view(-1, GENERATIONS)
        # TRL post12 _generate_and_score_completions, sum_then_normalize, scale_rewards="batch".
        mean = torch.nanmean(grouped, dim=1).repeat_interleave(GENERATIONS)
        batch_std = nanstd(values).expand_as(values)
        advantages = torch.nan_to_num((values - mean) / (batch_std + 1e-4), nan=0.0)
        group_std = nanstd(grouped, dim=1).repeat_interleave(GENERATIONS)
        rows = len(candidate_batch)
        return {
            "example_id": [row["example_id"] for row in candidate_batch],
            "advantages": advantages,
            "completion_ids": torch.ones(rows, 3, dtype=torch.long),
            "completion_mask": torch.ones(rows, 3),
            "group_reward_std": group_std,
        }

    trainer = SimpleNamespace(
        dynamic_sampling_max_batches=MAX_BATCHES,
        dynamic_sampling_reward_std_epsilon=0.0,
        num_generations=GENERATIONS,
        accelerator=_Accelerator(),
        _metrics={"train": defaultdict(list)},
        _generate_and_score_completions=generate,
        _select_dynamic_sampling_rows=GRPOTrainer._select_dynamic_sampling_rows,
    )
    trainer._concatenate_dynamic_sampling_batches = lambda batches: GRPOTrainer._concatenate_dynamic_sampling_batches(
        cast(Any, trainer), batches
    )
    kept, advantages, metrics = [], [], []
    for step in range(1, STEPS + 1):
        step_holder["step"] = step
        trainer._metrics = {"train": defaultdict(list)}
        pool = [{} for _ in range(TARGET * MAX_BATCHES * GENERATIONS)]
        # AdaptiveCurriculumTrainer._prepare_inputs: one decision for the whole candidate pool.
        pool = runtime.select_generation_batch(pool, step=step, selection_kind="initial_batch")
        batch = GRPOTrainer._prepare_dynamic_sampling_inputs(cast(Any, trainer), pool)
        kept.append(list(batch["example_id"][::GENERATIONS]))
        advantages.append(batch["advantages"].tolist())
        metrics.append({name: values[-1] for name, values in trainer._metrics["train"].items()})
    state = runtime.controller.state()
    runtime.close()
    return context.events, kept, advantages, metrics, state


def _verl_run(tmp_path: Path):
    import numpy as np
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    output = tmp_path / "verl"
    output.mkdir()
    SelectorConfig(
        settings=dict(SETTINGS),
        num_generations=GENERATIONS,
        state_dir=output / "state",
        journal_path=output / "journal.jsonl",
    ).write(output / "selector.json")
    selector = PosttrainCurriculumSelector(TASKS, config_path=str(output / "selector.json"))
    partition = f"dapo-{uuid.uuid4().hex}"
    occurrences: Counter[str] = Counter()
    uid_index: dict[str, int] = {}
    rewards_by_uid: dict[str, list[float]] = {}
    state = {"step": 0, "pool": [], "cursor": 0}

    def reserve(count: int) -> None:
        # The trainer's _reserve_candidates with a prompt selector: one initial_batch decision.
        state["pool"] = selector.select(count, global_steps=state["step"], stage="initial_batch", round_index=None)
        state["cursor"] = 0

    def dispatch(count: int, round_index: int | None = None) -> list[str]:
        indices = state["pool"][state["cursor"] : state["cursor"] + count]
        state["cursor"] += count
        uids = []
        for index in indices:
            task_id = str(TASKS[index]["example_id"])
            rewards = _rewards(task_id, occurrences[task_id])
            occurrences[task_id] += 1
            uid = uuid.uuid4().hex
            uid_index[uid] = index
            rewards_by_uid[uid] = rewards
            for session, reward in enumerate(rewards):
                tq.kv_put(
                    key=f"{uid}_{session}_0",
                    partition_id=partition,
                    fields={
                        "input_ids": torch.tensor([1, 2, 3]),
                        "extra_fields": {"reward_extra_info": {"group_reward": reward, "seq_reward": reward}},
                    },
                    tag={"is_prompt": False, "seq_len": 3, "global_steps": state["step"]},
                )
            tq.kv_put(
                key=uid,
                partition_id=partition,
                tag={"is_prompt": True, "status": "finished", "global_steps": state["step"]},
            )
            uids.append(uid)
        return uids

    def observe(groups):
        selector.observe([(uid_index[uid], values) for uid, values in groups], global_steps=state["step"])

    buffer = replay.CandidateBatchReplayBuffer(
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
        reserve_fn=reserve,
        observe_fn=observe,
        rows_per_group=GENERATIONS,
        active_max_rounds=MAX_BATCHES,
        active_metric="group_reward",
        active_observe_metric="seq_reward",
    )
    kept, advantages, metrics = [], [], []
    try:
        for step in range(1, STEPS + 1):
            state["step"] = step
            batch, step_metrics = buffer.sample(global_steps=step, partition_id=partition, batch_size=TARGET)
            order = {uid: position for position, uid in enumerate(uid_index)}
            uids = sorted({key.split("_")[0] for key in batch.keys}, key=order.__getitem__)
            kept.append([str(TASKS[uid_index[uid]]["example_id"]) for uid in uids])
            scores = torch.tensor([value for uid in uids for value in rewards_by_uid[uid]], dtype=torch.float64)
            token_rewards = scores.unsqueeze(-1)
            index = np.array([uid for uid in uids for _ in range(GENERATIONS)], dtype=object)
            row_std = np.array([buffer.candidate_std[uid] for uid in uids for _ in range(GENERATIONS)], dtype=object)
            step_advantages, _ = compute_grpo_outcome_advantage(
                token_rewards,
                torch.ones_like(token_rewards),
                index,
                epsilon=1e-4,
                std_scope="batch",
                trl_statistics=True,
                row_std=row_std,
            )
            advantages.append(step_advantages[:, 0].tolist())
            metrics.append(step_metrics)
            tq.kv_clear(keys=list(batch.keys), partition_id=partition)
    finally:
        keys = list(tq.kv_list(partition_id=partition).get(partition, {}).keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition)
    assert selector.runtime is not None
    final_state = selector.runtime.controller.state()
    selector.close()
    events = [
        (record["name"], record["attributes"])
        for record in map(json.loads, (output / "journal.jsonl").read_text().splitlines())
        if record["kind"] == "event"
    ]
    return events, kept, advantages, metrics, final_state


@pytest.fixture(scope="module")
def transfer_queue():
    tq.init()
    yield
    tq.close()


def test_dapo_candidate_batches_curriculum_and_batch_scaling_match_trl(transfer_queue, tmp_path: Path) -> None:
    trl_events, trl_kept, trl_advantages, trl_metrics, trl_state = _trl_run(tmp_path)
    verl_events, verl_kept, verl_advantages, verl_metrics, verl_state = _verl_run(tmp_path)

    assert any(metric["dynamic_sampling/candidate_batches"] > 1 for metric in trl_metrics)
    assert verl_events == trl_events
    assert verl_kept == trl_kept
    assert verl_metrics == pytest.approx(trl_metrics)
    assert verl_advantages == trl_advantages  # bitwise: same candidate-batch std, same arithmetic
    assert json.dumps(verl_state, sort_keys=True, default=str) == json.dumps(trl_state, sort_keys=True, default=str)
