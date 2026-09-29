"""TRL's and veRL's active sampling make the same round decisions for the same rewards.

TRL side: CarbonTeq TRL 1.12.0.post11 ``GRPOTrainer._prepare_active_sampling_inputs``
on a stub trainer whose generation step returns scripted group reward spreads.
veRL side: the CarbonTeq fork's ``ActiveSamplingReplayBuffer`` over a real
TransferQueue, with a dispatcher that completes each round with the same
scripted rewards. Both must dispatch the same round sizes, keep the same
candidate groups in the same order, fail in the same cases, and report the same
``active_sampling/...`` metrics. Needs the parity environment described in
``docs/plan/verl-vortex-port.md`` (TRL post11 and a veRL fork with active
sampling); skips elsewhere.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl.trainer.grpo_trainer", reason="requires CarbonTeq TRL 1.12.0.post11")
replay = pytest.importorskip("verl.trainer.ppo.v1.replay_buffer", reason="requires the CarbonTeq veRL fork")
if not hasattr(replay, "ActiveSamplingReplayBuffer"):
    pytest.skip("installed veRL has no active sampling", allow_module_level=True)
tq = pytest.importorskip("transfer_queue")

GENERATIONS = 2


class _Accelerator:
    device = torch.device("cpu")

    def gather(self, value):
        return value.reshape(1) if value.ndim == 0 else value


def _trl(target: int, max_batches: int, oversample: int, refill: int, keep: list[bool]):
    from trl.trainer.grpo_trainer import GRPOTrainer

    calls: list[int] = []

    def generate(candidate_batch):
        rows = len(candidate_batch)
        calls.append(rows // GENERATIONS)
        groups = [row["candidate"] for row in candidate_batch]
        std = torch.tensor([1.0 if keep[group] else 0.0 for group in groups])
        return {
            "candidate": torch.tensor(groups),
            "completion_ids": torch.ones(rows, 3, dtype=torch.long),
            "completion_mask": torch.ones(rows, 3),
            "group_reward_std": std,
        }

    stub = SimpleNamespace(
        active_sampling_max_batches=max_batches,
        active_sampling_oversample=oversample,
        active_sampling_oversample_refill=refill,
        active_sampling_reward_std_epsilon=0.0,
        num_generations=GENERATIONS,
        accelerator=_Accelerator(),
        _metrics={"train": defaultdict(list)},
        _generate_and_score_completions=generate,
        _select_dynamic_sampling_rows=GRPOTrainer._select_dynamic_sampling_rows,
    )
    stub._concatenate_dynamic_sampling_batches = lambda batches: GRPOTrainer._concatenate_dynamic_sampling_batches(
        cast(Any, stub), batches
    )
    candidates = [{"candidate": group} for group in range(target * max_batches) for _ in range(GENERATIONS)]
    try:
        batch = GRPOTrainer._prepare_active_sampling_inputs(cast(Any, stub), candidates)
    except RuntimeError as error:
        return calls, None, {}, str(error)
    selected = batch["candidate"].tolist()[::GENERATIONS]
    metrics = {name: values[-1] for name, values in stub._metrics["train"].items()}
    return calls, selected, metrics, None


def _verl(target: int, max_batches: int, oversample: int, refill: int, keep: list[bool]):
    partition = f"parity-{uuid.uuid4().hex}"
    calls: list[int] = []
    order: dict[str, int] = {}

    def dispatch(count: int, round_index: int | None = None) -> list[str]:
        calls.append(count)
        uids = []
        for _ in range(count):
            group = len(order)
            uid = uuid.uuid4().hex
            order[uid] = group
            rewards = (0.0, 1.0) if keep[group] else (1.0, 1.0)
            for session, reward in enumerate(rewards):
                tq.kv_put(
                    key=f"{uid}_{session}_0",
                    partition_id=partition,
                    fields={
                        "input_ids": torch.tensor([1, 2, 3]),
                        "extra_fields": {"reward_extra_info": {"seq_reward": reward}},
                    },
                    tag={"is_prompt": False, "seq_len": 3, "global_steps": 1},
                )
            tq.kv_put(key=uid, partition_id=partition, tag={"is_prompt": True, "status": "finished", "global_steps": 1})
            uids.append(uid)
        return uids

    buffer = replay.ActiveSamplingReplayBuffer(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=1,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0.01,
        refill_fn=lambda count: count,
        train_batch_size=target,
        gen_batch_size=1,
        dispatch_fn=dispatch,
        active_max_rounds=max_batches,
        active_oversample=oversample,
        active_oversample_refill=refill,
    )
    try:
        batch, metrics = buffer.sample(global_steps=1, partition_id=partition, batch_size=target)
    except RuntimeError as error:
        return calls, None, {}, str(error)
    finally:
        keys = list(tq.kv_list(partition_id=partition).get(partition, {}).keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition)
    selected = sorted({order[key.split("_")[0]] for key in batch.keys})
    return calls, selected, metrics, None


@pytest.fixture(scope="module")
def transfer_queue():
    tq.init()
    yield
    tq.close()


SCENARIOS = {
    # target, max_batches, oversample, oversample_refill, keep decisions per candidate group
    "first-round-full": (3, 3, 0, 0, [True, True, True] + [False] * 6),
    "refill-only-missing": (3, 4, 0, 0, [True, False, False, False, True, False, True] + [False] * 5),
    "oversample-surplus-discarded": (2, 3, 2, 0, [True, True, True, True] + [False] * 2),
    "refill-capped-at-first-round": (4, 3, 2, 5, [True, False, False, True, False, False] + [True] * 6),
    "extra-cut-to-pool": (2, 2, 1, 1, [True, False, False, True]),
    "pool-exhausted": (2, 2, 2, 0, [False] * 4),
    "rounds-exhausted": (2, 2, 0, 0, [False] * 4),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_active_sampling_rounds_match_trl(transfer_queue, scenario: str) -> None:
    target, max_batches, oversample, refill, keep = SCENARIOS[scenario]
    trl_calls, trl_selected, trl_metrics, trl_error = _trl(target, max_batches, oversample, refill, keep)
    verl_calls, verl_selected, verl_metrics, verl_error = _verl(target, max_batches, oversample, refill, keep)

    assert verl_calls == trl_calls
    assert (verl_error is None) == (trl_error is None), (trl_error, verl_error)
    if trl_error is not None:
        for cause in ("exhausted its bounded candidate pool", f"exhausted {max_batches} generation rounds"):
            assert (cause in trl_error) == (cause in cast(str, verl_error)), (trl_error, verl_error)
        return
    assert verl_selected == sorted(cast(list[int], trl_selected))
    assert cast(list[int], trl_selected) == sorted(cast(list[int], trl_selected))  # TRL keeps candidate order
    assert verl_metrics == pytest.approx(trl_metrics)


@pytest.mark.parametrize("seed", range(12))
def test_active_sampling_rounds_match_trl_on_random_reward_patterns(transfer_queue, seed: int) -> None:
    import random

    rng = random.Random(seed)
    target, max_batches = rng.randint(1, 4), rng.randint(1, 4)
    oversample, refill = rng.randint(0, 3), rng.randint(0, 3)
    keep = [rng.random() < 0.4 for _ in range(target * max_batches)]
    trl = _trl(target, max_batches, oversample, refill, keep)
    verl = _verl(target, max_batches, oversample, refill, keep)

    assert verl[0] == trl[0]
    assert (verl[3] is None) == (trl[3] is None), (trl[3], verl[3])
    if trl[3] is None:
        assert verl[1] == trl[1]
        assert verl[2] == pytest.approx(trl[2])
