"""The GRPO settings veRL used to reject now match TRL's real code on fixed inputs.

Each check configures both sides from one Posttrain selection: TRL through
``_online_rl_arguments`` (the TRL backend's translation) and veRL through the
launcher payload, the worker's Hydra overrides and veRL's composed
``ppo_trainer.yaml``. Covered: every sampler-correction mode and bound,
advantage scaling (group, batch, none) with and without masked truncated
completions, the resulting GRPO loss and gradient, group admission retries, and
the linear learning-rate schedule. Needs the parity environment described in
``docs/plan/verl-vortex-port.md`` (CarbonTeq TRL 1.12.0.post12 and the veRL fork
at a revision with every TRL-equivalence delta); skips elsewhere.
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl.trainer.grpo_trainer", reason="requires CarbonTeq TRL 1.12.0.post12")
rollout_corr = pytest.importorskip("verl.trainer.ppo.rollout_corr_helper", reason="requires the CarbonTeq veRL fork")
if "rollout_is_clip_min" not in rollout_corr.compute_rollout_correction_weights.__code__.co_varnames:
    pytest.skip("installed veRL lacks the TRL-equivalence deltas", allow_module_level=True)

from posttrain.common import ExecutionTarget, InferenceBinding  # noqa: E402
from posttrain.common.variants import QWEN_35_2B  # noqa: E402
from posttrain.data import RolloutDataset, RolloutExample  # noqa: E402
from posttrain.train import (  # noqa: E402
    QWEN35_RENDERER,
    GRPORequest,
    GRPOSettings,
    LoRAUpdate,
    TrainingBinding,
    TrainingLoop,
    TrainingRuntime,
)
from posttrain.train.backends.trl.policy_config import _online_rl_arguments  # noqa: E402
from posttrain.train.backends.verl.launcher import build_grpo_launch_plan  # noqa: E402
from posttrain.train.backends.verl.worker import build_hydra_overrides  # noqa: E402

REVISION = "ce8e0430018204b03c009b72bfba3b58968696c7"
GROUPS = 2
GENERATIONS = 4
ROWS = GROUPS * GENERATIONS
PROMPT = 4
RESPONSE = 10


@dataclass(frozen=True)
class _Environment:
    id: str = "envs/settings-parity@1"
    revision: str = "1"
    category: str = "tool-agentic"


@dataclass
class _Bridge:
    dataset: RolloutDataset = field(
        default_factory=lambda: RolloutDataset(
            "settings-parity-v1", "a" * 40, (RolloutExample("task-a", "a", {}), RolloutExample("task-b", "b", {}))
        )
    )
    max_concurrent: int = 16

    async def run(self, batch, generator):  # pragma: no cover - never called
        raise AssertionError


def _settings(**changes: Any) -> GRPOSettings:
    values: dict[str, Any] = {
        "id": "settings/trl-parity@1",
        "loop": TrainingLoop(
            max_steps=6,
            max_length=PROMPT + RESPONSE,
            per_device_batch_size=ROWS // 2,
            gradient_accumulation_steps=2,
            lr_scheduler_type="constant",
        ),
        "num_prompts_per_step": GROUPS,
        "num_generations": GENERATIONS,
        "max_prompt_length": PROMPT,
        "max_completion_length": RESPONSE,
    }
    values.update(changes)
    return GRPOSettings(**values)


def _request(settings: GRPOSettings, backend: str) -> GRPORequest:
    target = ExecutionTarget("targets/parity", "1", "nvidia-cuda", 8, {"world_size": 1})
    options: dict[str, Any] = {}
    if backend.startswith("verl@"):
        options = {
            "python_executable": "/opt/posttrain-verl/bin/python",
            "working_directory": "/opt/src/verl",
            "source_revision": REVISION,
        }
    training = TrainingBinding(
        "training/parity@1",
        "1",
        backend,
        QWEN35_RENDERER,
        LoRAUpdate(rank=8, alpha=16, target_modules=r".*\.(o_proj|down_proj)$"),
        target,
        runtime=TrainingRuntime(global_batch_size=ROWS, nodes=1, devices_per_node=1),
        backend_options=options,
    )
    inference = InferenceBinding(
        "inference/parity@1",
        "1",
        QWEN_35_2B,
        "vllm@0.25.1",
        QWEN_35_2B.renderer_contract,
        {"mode": "colocate", "sleep_during_optimization": True, "max_model_len": PROMPT + RESPONSE, "max_num_seqs": 16},
        {"temperature": 1.0, "max_tokens": RESPONSE},
        target,
        ("rollout",),
    )
    return GRPORequest(QWEN_35_2B, cast(Any, _Bridge()), settings, _Environment(), training, inference)


def _trl_config(settings: GRPOSettings, tmp_path: Path) -> Any:
    from trl.trainer.grpo_config import GRPOConfig

    arguments = _online_rl_arguments(_request(settings, "trl@1.12.0.post12"), tmp_path / "trl", {})
    arguments.update(bf16=False, fp16=False, use_cpu=True, report_to=[])
    return GRPOConfig(**arguments)


def _verl(settings: GRPOSettings, tmp_path: Path) -> tuple[Any, Any, list[str]]:
    import verl
    from hydra import compose, initialize_config_dir
    from posttrain.train.backends.verl import worker
    from verl.utils.config import omega_conf_to_dataclass

    manifest = build_grpo_launch_plan(_request(settings, f"verl@{REVISION[:7]}"), tmp_path / uuid.uuid4().hex)
    original = worker._model_path
    worker._model_path = lambda model: "/models/parity"
    try:
        overrides = build_hydra_overrides(manifest, tmp_path / "r.parquet", tmp_path / "a.json", tmp_path / "c")
    finally:
        worker._model_path = original
    config_dir = Path(str(verl.__file__)).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    return omega_conf_to_dataclass(config.actor_rollout_ref.actor), omega_conf_to_dataclass(config.algorithm), overrides


def _logprobs() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(11)
    lengths = [10, 7, 10, 4, 9, 10, 6, 10]
    mask = torch.zeros(ROWS, RESPONSE, dtype=torch.float64)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    rollout = -torch.rand(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 2.0 - 0.05
    old = rollout + torch.randn(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 0.8
    old[0, 1] = rollout[0, 1] + 24.0  # far beyond veRL's former |log ratio| <= 20 safety clamp
    old[2, 3] = rollout[2, 3] - 26.0
    current = old + torch.randn(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 0.3
    ref = old + torch.randn(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 0.3
    return {"mask": mask, "rollout": rollout, "old": old, "current": current, "ref": ref, "lengths": lengths}


class _Accelerator:
    num_processes = 1
    sync_gradients = True
    device = torch.device("cpu")
    process_index = 0

    def gather(self, value):
        return value

    def gather_for_metrics(self, value):
        return value

    def reduce(self, value, reduction="sum"):
        del reduction
        return value


def _trl_stub(config: Any, current: Any) -> Any:
    from trl.trainer.grpo_trainer import GRPOTrainer

    stub = SimpleNamespace(
        args=config,
        beta=config.beta,
        loss_type=config.loss_type,
        epsilon_low=config.epsilon,
        epsilon_high=config.epsilon_high,
        importance_sampling_level=config.importance_sampling_level,
        use_vllm=config.use_vllm,
        vllm_importance_sampling_correction=config.vllm_importance_sampling_correction,
        vllm_importance_sampling_mode=config.vllm_importance_sampling_mode,
        vllm_importance_sampling_clip_min=config.vllm_importance_sampling_clip_min,
        vllm_importance_sampling_clip_max=config.vllm_importance_sampling_clip_max,
        top_entropy_quantile=config.top_entropy_quantile,
        off_policy_mask_threshold=config.off_policy_mask_threshold,
        aux_loss_enabled=False,
        _entropy_bonus_enabled=False,
        model=SimpleNamespace(training=True),
        current_gradient_accumulation_steps=config.gradient_accumulation_steps,
        accelerator=_Accelerator(),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )
    stub.rows = slice(None)

    def per_token_logps(model, input_ids, attention_mask, logits_to_keep, **kwargs):
        del model, input_ids, attention_mask, logits_to_keep, kwargs
        logps = current[stub.rows]
        return logps, torch.zeros_like(logps.detach()), None

    stub._get_per_token_logps_and_entropies = per_token_logps
    stub._vllm_importance_sampling_ratio = lambda *args: GRPOTrainer._vllm_importance_sampling_ratio(
        cast(Any, stub), *args
    )
    return stub


def _verl_weights(algorithm: Any, batch: dict[str, Any], mask: Any) -> Any:
    from verl import DataProto
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

    proto = DataProto.from_dict(
        tensors={"old_log_probs": batch["old"], "rollout_log_probs": batch["rollout"], "response_mask": mask.clone()}
    )
    proto, _ = compute_rollout_correction_and_add_to_batch(proto, algorithm.rollout_correction)
    return proto.batch["rollout_is_weights"]


IS_CASES = [
    ("token_truncate", None, 2.0),
    ("token_truncate", 0.5, 2.0),
    ("token_truncate", 0.5, None),
    ("sequence_truncate", 0.1, 3.0),
    ("sequence_truncate", None, 3.0),
    ("token_mask", 0.5, 2.0),
    ("token_mask", None, 2.0),
    ("token_mask", 0.5, None),
    ("sequence_mask", 0.1, 3.0),
    ("sequence_mask", None, 3.0),
]


@pytest.mark.parametrize(("mode", "minimum", "maximum"), IS_CASES)
def test_sampler_correction_weights_match_trl(tmp_path: Path, mode: str, minimum: Any, maximum: Any) -> None:
    settings = _settings(
        importance_sampling_mode=mode, importance_sampling_clip_min=minimum, importance_sampling_clip_max=maximum
    )
    batch = _logprobs()
    mask = batch["mask"]
    trl_config = _trl_config(settings, tmp_path)
    stub = _trl_stub(trl_config, batch["current"])
    trl_ratio, _, _ = stub._vllm_importance_sampling_ratio(batch["old"], batch["rollout"], mask)
    _, algorithm, _ = _verl(settings, tmp_path)
    weights = _verl_weights(algorithm, batch, mask)
    expected = trl_ratio.expand_as(mask)
    torch.testing.assert_close(weights[mask.bool()], expected[mask.bool()], rtol=1e-12, atol=0)


def _trl_advantages(settings: GRPOSettings, rewards: Any, truncated: list[bool]) -> Any:
    # Transcribed from TRL 1.12.0.post12 GRPOTrainer._generate_and_score_completions
    # (multi_objective_aggregation="sum_then_normalize", one reward function of weight 1):
    # masked truncated completions get NaN rewards, the group mean and std ignore NaN,
    # advantages divide by std + 1e-4 unless scale_rewards="none", NaN advantages become 0.
    from trl.trainer.utils import nanstd

    rewards = rewards.clone()
    if settings.mask_truncated_completions:
        rewards[torch.tensor(truncated)] = torch.nan
    mean_grouped = torch.nanmean(rewards.view(-1, GENERATIONS), dim=1).repeat_interleave(GENERATIONS, dim=0)
    if settings.advantage_scaling == "batch":
        std = nanstd(rewards).expand_as(rewards)
    else:
        std = nanstd(rewards.view(-1, GENERATIONS), dim=1).repeat_interleave(GENERATIONS, dim=0)
    advantages = rewards - mean_grouped
    if settings.advantage_scaling != "none":
        advantages = advantages / (std + 1e-4)
    return torch.nan_to_num(advantages, nan=0.0)


REWARDS = [1.0, 0.0, 0.5, 0.25, 0.0, 0.0, 1.0, 0.75]
TRUNCATED = [False, True, False, False, False, True, True, False]


@pytest.mark.parametrize("scaling", ["group", "batch", "none"])
@pytest.mark.parametrize("mask_truncated", [False, True])
def test_grpo_advantages_loss_and_gradient_match_trl(tmp_path: Path, scaling: str, mask_truncated: bool) -> None:
    from tensordict import TensorDict
    from trl.trainer.grpo_trainer import GRPOTrainer
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.losses import ppo_loss

    settings = _settings(advantage_scaling=scaling, mask_truncated_completions=mask_truncated, beta=0.01)
    batch = _logprobs()
    mask = batch["mask"]
    rewards = torch.tensor(REWARDS, dtype=torch.float64)
    trl_advantages = _trl_advantages(settings, rewards, TRUNCATED)

    actor, algorithm, overrides = _verl(settings, tmp_path)
    assert ("algorithm.exclude_flagged_rows=true" in overrides) == mask_truncated
    token_rewards = torch.zeros(ROWS, RESPONSE, dtype=torch.float64)
    for row, length in enumerate(batch["lengths"]):
        token_rewards[row, length - 1] = rewards[row]
    data = DataProto.from_dict(
        tensors={"token_level_rewards": token_rewards, "response_mask": mask.clone()},
        non_tensors={"uid": [f"g{row // GENERATIONS}" for row in range(ROWS)]},
    )
    excluded = [flag and mask_truncated for flag in TRUNCATED]
    if algorithm.exclude_flagged_rows:
        import numpy as np

        data.non_tensor_batch["exclude_from_group_stats"] = np.array(excluded, dtype=bool)
    data = compute_advantage(
        data,
        adv_estimator="grpo",
        num_repeat=GENERATIONS,
        norm_adv_by_std_in_grpo=algorithm.norm_adv_by_std_in_grpo,
        config=algorithm,
    )
    verl_advantages = data.batch["advantages"]
    loss_mask = mask.clone()
    for row, flag in enumerate(excluded):
        if flag:  # the V1 trainer zeroes excluded rows after advantages
            loss_mask[row] = 0
            verl_advantages[row] = 0
    expanded = trl_advantages.unsqueeze(1).expand(ROWS, RESPONSE)
    assert torch.equal(verl_advantages[mask.bool()], (expanded * loss_mask)[mask.bool()])  # bitwise

    # TRL loss: masked truncated completions have completion_mask 0 (TRL's own masking).
    trl_config = _trl_config(settings, tmp_path)
    current = batch["current"].clone().requires_grad_(True)
    stub = _trl_stub(trl_config, current)
    ratio, _, _ = stub._vllm_importance_sampling_ratio(batch["old"], batch["rollout"], loss_mask)
    total = torch.zeros((), dtype=torch.float64)
    size = ROWS // 2
    for start in range(0, ROWS, size):
        rows = slice(start, start + size)
        stub.rows = rows
        inputs = {
            "prompt_ids": torch.ones(size, PROMPT, dtype=torch.long),
            "prompt_mask": torch.ones(size, PROMPT, dtype=torch.long),
            "completion_ids": torch.ones(size, RESPONSE, dtype=torch.long),
            "completion_mask": loss_mask[rows],
            "advantages": trl_advantages[rows],
            "old_per_token_logps": batch["old"][rows],
            "sampling_per_token_logps": batch["rollout"][rows],
            "ref_per_token_logps": batch["ref"][rows],
            "importance_sampling_ratio": ratio[rows],
            "num_items_in_batch": loss_mask.sum(),
        }
        total = total + GRPOTrainer._compute_loss(cast(Any, stub), None, inputs)
    (trl_gradient,) = torch.autograd.grad(total, current)

    weights = _verl_weights(algorithm, batch, mask) * loss_mask
    verl_current = batch["current"].clone().requires_grad_(True)
    verl_total = torch.zeros((), dtype=torch.float64)
    for start in range(0, ROWS, size):
        rows = slice(start, start + size)
        flat = torch.cat(
            [
                torch.cat(
                    [
                        torch.zeros(PROMPT - 1, dtype=torch.float64),
                        verl_current[row],
                        torch.zeros(1, dtype=torch.float64),
                    ]
                )
                for row in range(start, start + size)
            ]
        )
        batch_td = TensorDict(
            {
                "prompts": torch.ones(size, PROMPT, dtype=torch.long),
                "responses": torch.ones(size, RESPONSE, dtype=torch.long),
                "attention_mask": torch.ones(size, PROMPT + RESPONSE, dtype=torch.long),
                "response_mask": loss_mask[rows],
                "old_log_probs": batch["old"][rows],
                "advantages": verl_advantages[rows],
                "rollout_is_weights": weights[rows],
                "ref_log_prob": batch["ref"][rows],
            },
            batch_size=[size],
        )
        tu.assign_non_tensor(batch_td, dp_size=1, batch_num_tokens=int(loss_mask.sum().item()), global_batch_size=ROWS)
        loss, _ = ppo_loss(actor, {"log_probs": flat}, batch_td)
        verl_total = verl_total + loss
    (verl_gradient,) = torch.autograd.grad(verl_total, verl_current)
    # veRL's seq-mean-token-mean divides by (tokens + 1e-8) where TRL clamps at 1.
    torch.testing.assert_close(verl_total.detach(), total.detach(), rtol=1e-8, atol=1e-12)
    torch.testing.assert_close(verl_gradient, trl_gradient, rtol=1e-8, atol=1e-12)
    assert trl_gradient.abs().sum() > 0


# Per (group, attempt): True when the group's collection fails.
ADMISSION_SCRIPTS = {
    "retry-then-succeed": ({"g0": [True, False], "g1": [False]}, 3),
    "dropped-after-attempts": ({"g0": [True, True, True], "g1": [False]}, 3),
    "single-attempt-drops": ({"g0": [True], "g1": [True], "g2": [False]}, 1),
    "mixed": ({"g0": [True, True, False], "g1": [True, True], "g2": [False]}, 2),
}


@pytest.mark.parametrize("name", sorted(ADMISSION_SCRIPTS))
def test_group_admission_retries_and_drops_like_trl(name: str) -> None:
    import transfer_queue as tq
    from posttrain.train.online_rl import EnvironmentRollout, RolloutBatch
    from posttrain.train.reward_admission import admit_rollout_groups
    from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

    script, attempts = ADMISSION_SCRIPTS[name]
    groups = sorted(script)
    settings = _settings(max_admission_attempts=attempts)

    # TRL side: Posttrain's real admission loop (TRL backend) over scripted collections.
    calls: dict[str, int] = defaultdict(int)

    def collect(batch: RolloutBatch):
        from posttrain.common import TraceObservation
        from posttrain.train.online_rl import PartialRolloutBatchError

        completed, failures = {}, {}
        for ordinal, group in enumerate(batch.prompt_group_ids):
            index = calls[group] // GENERATIONS
            calls[group] += 1
            if script[group][index]:
                failures[ordinal] = "scripted_failure"
            else:
                completed[ordinal] = EnvironmentRollout(
                    example_id=batch.example_ids[ordinal],
                    prompt_ids=(1,),
                    completion_ids=(2,),
                    sampling_logprobs=(-0.1,),
                    env_mask=(True,),
                    reward=1.0,
                    is_truncated=False,
                    trace=TraceObservation("test", f"{group}-{ordinal}-{index}", {}),
                )
        if failures:
            raise PartialRolloutBatchError("scripted failures", completed=completed, failures=failures)
        return [completed[ordinal] for ordinal in range(len(batch.prompt_group_ids))]

    rows = [(group, generation) for group in groups for generation in range(GENERATIONS)]
    batch = RolloutBatch(
        example_ids=tuple(f"task-{group}" for group, _ in rows),
        step=1,
        model_id="m",
        prompt_group_ids=tuple(group for group, _ in rows),
        rollout_ids=tuple(f"{group}/{generation}" for group, generation in rows),
    )
    trl = admit_rollout_groups(
        batch, settings, collect, lambda failures: failures, max_attempts=None, retain_complete_on_exhaustion=True
    )
    trl_retained = sorted({batch.prompt_group_ids[position] for position in trl.retained_positions})
    trl_attempts = {group: calls[group] // GENERATIONS for group in groups}

    # veRL side: the fork's sync replay buffer with the same scripted outcomes per attempt.
    tq.init()
    partition = f"admission-{uuid.uuid4().hex}"
    origin: dict[str, str] = {}
    tries: dict[str, int] = defaultdict(int)

    def produce(group: str) -> str:
        uid = uuid.uuid4().hex
        origin[uid] = group
        failed = script[group][tries[group]]
        tries[group] += 1
        if not failed:
            for session in range(GENERATIONS):
                tq.kv_put(
                    key=f"{uid}_{session}_0",
                    partition_id=partition,
                    fields={"input_ids": torch.tensor([1, 2])},
                    tag={"is_prompt": False, "seq_len": 2, "global_steps": 1},
                )
        tq.kv_put(
            key=uid,
            partition_id=partition,
            tag={"is_prompt": True, "status": "failure" if failed else "finished", "global_steps": 1},
        )
        return uid

    try:
        for group in groups:
            produce(group)
        buffer = ReplayBuffer(
            trainer_mode="sync",
            trainer_config={},
            max_off_policy_threshold=8,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            poll_interval=0.01,
            refill_fn=lambda count: count,
            train_batch_size=len(groups),
            gen_batch_size=1,
            retry_fn=lambda failed_uid: produce(origin[failed_uid]),
            failed_group_attempts=attempts,
        )
        try:
            sampled, _ = buffer.sample(global_steps=1, partition_id=partition, batch_size=len(groups))
            verl_retained = sorted({origin[key.split("_")[0]] for key in sampled.keys})
        except RuntimeError as error:
            assert "retained no complete groups" in str(error)
            verl_retained = []
    finally:
        keys = list(tq.kv_list(partition_id=partition).get(partition, {}).keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition)
        tq.close()

    assert verl_retained == trl_retained
    assert dict(tries) == trl_attempts


def test_partial_batch_loss_is_scaled_to_retained_rows_like_trl() -> None:
    """TRL pads a partial admitted batch and multiplies its loss by scheduled/retained rows;
    the fork normalizes the seq-mean loss over real rows instead of padding rows."""

    from trl.trainer.rollout_admission import pad_admitted_rollout_batch
    from verl.trainer.ppo.core_algos import agg_loss

    generator = torch.Generator().manual_seed(3)
    retained = 6
    scheduled = ROWS
    per_token = torch.randn(retained, RESPONSE, generator=generator, dtype=torch.float64)
    mask = torch.ones(retained, RESPONSE, dtype=torch.float64)
    mask[1, 6:] = 0
    padded = pad_admitted_rollout_batch(
        {
            "completion_ids": torch.ones(retained, RESPONSE, dtype=torch.long),
            "completion_mask": mask.clone(),
            "advantages": torch.ones(retained, dtype=torch.float64),
        },
        scheduled,
    )
    trl_mask = padded["completion_mask"]
    trl_losses = per_token[torch.arange(scheduled) % retained]
    trl_loss = ((trl_losses * trl_mask).sum(-1) / trl_mask.sum(-1).clamp(min=1.0)).mean() * padded[
        "admission_loss_scale"
    ]
    verl_mask = torch.cat([mask, torch.zeros(scheduled - retained, RESPONSE, dtype=torch.float64)])
    verl_losses = torch.cat([per_token, torch.zeros(scheduled - retained, RESPONSE, dtype=torch.float64)])
    verl_loss = agg_loss(verl_losses, verl_mask, "seq-mean-token-mean", global_batch_size=retained)
    # TRL keeps admission_loss_scale (8/6) as a float32 tensor, so agreement is float32-limited.
    torch.testing.assert_close(verl_loss, trl_loss, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize(("warmup_ratio", "max_steps"), [(0.0, 6), (0.34, 6), (0.1, 25)])
def test_linear_learning_rate_matches_trl(tmp_path: Path, warmup_ratio: float, max_steps: int) -> None:
    from transformers import get_scheduler
    from verl.utils.torch_functional import get_linear_schedule_with_warmup

    settings = _settings(
        loop=replace(_settings().loop, max_steps=max_steps, warmup_ratio=warmup_ratio, lr_scheduler_type="linear")
    )
    trl_config = _trl_config(settings, tmp_path)
    _, _, overrides = _verl(settings, tmp_path)
    assert "actor_rollout_ref.actor.optim.lr_scheduler_type=linear" in overrides
    warmup = int(
        next(v for v in overrides if v.startswith("actor_rollout_ref.actor.optim.lr_warmup_steps=")).split("=")[1]
    )
    assert warmup == math.ceil(max_steps * warmup_ratio)

    def lrs(factory):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=settings.loop.learning_rate)
        scheduler = factory(optimizer)
        values = []
        for _ in range(max_steps):
            values.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        return values

    trl = lrs(
        lambda optimizer: get_scheduler(
            trl_config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=trl_config.get_warmup_steps(max_steps),
            num_training_steps=max_steps,
        )
    )
    verl = lrs(lambda optimizer: get_linear_schedule_with_warmup(optimizer, warmup, max_steps))
    assert verl == trl
