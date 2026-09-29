"""One OLMo 3 batch through TRL's and veRL's real loss code gives the same update.

Both sides are configured from one ``GRPOSettings`` by Posttrain's own adapters:
TRL through ``_online_rl_arguments``/``_trainer_arguments`` into
``Olmo3GRPOConfig``, veRL through ``grpo_algorithm_payload`` and the worker's
Hydra overrides composed onto veRL's ``ppo_trainer.yaml``. The test needs an
environment with CarbonTeq TRL 1.12.0.post12 and a CarbonTeq veRL revision that
registers ``token_clip``; it skips elsewhere. See Concrete Steps in
``docs/plan/verl-vortex-port.md`` for that environment.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl.trainer.olmo3_grpo_config", reason="requires CarbonTeq TRL with Olmo3GRPOConfig")
pytest.importorskip("verl.workers.utils.losses", reason="requires the CarbonTeq veRL fork")

from posttrain.common import ExecutionTarget, InferenceBinding, TraceObservation  # noqa: E402
from posttrain.common.variants import QWEN_35_2B  # noqa: E402
from posttrain.data import RolloutDataset, RolloutExample  # noqa: E402
from posttrain.train import (  # noqa: E402
    QWEN35_RENDERER,
    ActiveGroupSampling,
    GRPORequest,
    GRPOSettings,
    LoRAUpdate,
    TrainingBinding,
    TrainingLoop,
    TrainingRuntime,
)
from posttrain.train.backends.trl.policy_config import _online_rl_arguments  # noqa: E402
from posttrain.train.backends.trl.policy_optimization import _trainer_arguments  # noqa: E402
from posttrain.train.backends.verl.launcher import build_grpo_launch_plan, grpo_algorithm_payload  # noqa: E402
from posttrain.train.backends.verl.reward_fields import shaped_rollout_reward  # noqa: E402
from posttrain.train.backends.verl.worker import _FORK_NATIVE_NAME_REVISIONS, build_hydra_overrides  # noqa: E402
from posttrain.train.online_rl import EnvironmentRollout  # noqa: E402
from posttrain.train.profiles import shape_online_reward  # noqa: E402

GROUPS = 2
GENERATIONS = 4
ROWS = GROUPS * GENERATIONS
PROMPT = 4
RESPONSE = 16
MICRO_BATCHES = 2
BETA = 0.005
TRUNCATION_PENALTY = 0.2


@dataclass(frozen=True)
class _Environment:
    id: str = "envs/parity@1"
    revision: str = "1"
    category: str = "tool-agentic"


@dataclass
class _Bridge:
    dataset: RolloutDataset = field(
        default_factory=lambda: RolloutDataset(
            "olmo3-parity-v1", "a" * 40, (RolloutExample("task-a", "a", {}), RolloutExample("task-b", "b", {}))
        )
    )
    max_concurrent: int = 8

    async def run(self, batch, generator):  # pragma: no cover - never called
        raise AssertionError


def _settings() -> GRPOSettings:
    return GRPOSettings(
        "settings/olmo3-parity@1",
        TrainingLoop(
            max_steps=1,
            max_length=RESPONSE + PROMPT,
            lr_scheduler_type="constant",
            per_device_batch_size=ROWS // MICRO_BATCHES,
            gradient_accumulation_steps=MICRO_BATCHES,
        ),
        num_prompts_per_step=GROUPS,
        num_generations=GENERATIONS,
        max_prompt_length=PROMPT,
        max_completion_length=RESPONSE,
        beta=BETA,
        algorithm="olmo3",
        advantage_scaling="none",
        importance_sampling_mode="token_truncate",
        importance_sampling_clip_min=None,
        importance_sampling_clip_max=2.0,
        active_sampling=ActiveGroupSampling(max_candidate_batches=3),
        truncation_penalty=TRUNCATION_PENALTY,
    )


def _request(backend: str, source_revision: str | None = None) -> GRPORequest:
    target = ExecutionTarget("targets/parity", "1", "nvidia-cuda", 8, {"world_size": 1})
    options: dict[str, Any] = {}
    if source_revision is not None:
        options = {
            "python_executable": "/opt/posttrain-verl/bin/python",
            "working_directory": "/opt/src/verl",
            "source_revision": source_revision,
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
        {"mode": "colocate", "sleep_during_optimization": True, "max_model_len": PROMPT + RESPONSE},
        {"temperature": 1.0, "max_tokens": RESPONSE},
        target,
        ("rollout",),
    )
    return GRPORequest(QWEN_35_2B, cast(Any, _Bridge()), _settings(), _Environment(), training, inference)


def _batch() -> dict[str, Any]:
    """Fixed rollouts that exercise both clip bounds, the correction cap and a truncation."""

    generator = torch.Generator().manual_seed(20260928)
    lengths = [16, 11, 16, 7, 16, 13, 9, 16]
    mask = torch.zeros(ROWS, RESPONSE, dtype=torch.float64)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    mask[1, 3:6] = 0  # tool-observation tokens inside a multi-turn trajectory
    mask[6, 2:4] = 0
    rollout = -torch.rand(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 2.5 - 0.05
    old = rollout + torch.randn(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 0.6
    current = old + torch.randn(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 0.35
    ref = old + torch.randn(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 0.4
    # Rows 1 and 7 get negative advantages. Ratios above 3 there are where veRL's
    # dual-clip `vanilla` loss would differ; a reference far above the policy is
    # where the clamped `low_var_kl` estimate (capped at 10) would differ.
    current[1, 7:9] = old[1, 7:9] + 1.5
    ref[7, 10:12] = current[7, 10:12] + 3.2
    task_rewards = [1.0, 0.0, 0.5, 0.0, 0.0, 0.25, 1.0, 0.0]
    truncated = [False, False, False, True, True, False, False, False]
    return {
        "mask": mask,
        "rollout": rollout,
        "old": old,
        "current": current,
        "ref": ref,
        "task_rewards": task_rewards,
        "truncated": truncated,
        "lengths": lengths,
    }


def _environment_rollout(reward: float, length: int, truncated: bool) -> EnvironmentRollout:
    return EnvironmentRollout(
        example_id="task",
        prompt_ids=(1,) * PROMPT,
        completion_ids=(2,) * length,
        sampling_logprobs=(-0.1,) * length,
        env_mask=(True,) * length,
        reward=reward,
        is_truncated=truncated,
        trace=TraceObservation("test", "trace", {}),
    )


# ---------------------------------------------------------------------------
# TRL side


def _trl_config(tmp_path: Path) -> Any:
    from trl.trainer.olmo3_grpo_config import Olmo3GRPOConfig

    request = _request("trl@1.12.0.post12")
    arguments = _online_rl_arguments(request, tmp_path / "trl", {"enable_thinking": False})
    arguments.update(bf16=False, fp16=False, use_cpu=True, report_to=[])
    return _trainer_arguments(Olmo3GRPOConfig, arguments, request)


def _trl_advantages(rewards: Any) -> Any:
    # Transcribed from TRL 1.12.0.post12 GRPOTrainer._generate_and_score_completions,
    # multi_objective_aggregation="sum_then_normalize" with one reward function of
    # weight 1 and scale_rewards="none" (Olmo3GRPOConfig): the group mean is
    # subtracted and nothing is divided.
    mean_grouped = torch.nanmean(rewards.view(-1, GENERATIONS), dim=1).repeat_interleave(GENERATIONS, dim=0)
    return torch.nan_to_num(rewards - mean_grouped, nan=0.0)


class _Accelerator:
    num_processes = 1
    sync_gradients = True
    device = torch.device("cpu")

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


def _trl_update(config: Any, batch: dict[str, Any], advantages: Any):
    from trl.trainer.grpo_trainer import GRPOTrainer

    assert config.steps_per_generation == MICRO_BATCHES == config.gradient_accumulation_steps
    current = batch["current"].clone().requires_grad_(True)
    stub = _trl_stub(config, current)
    mask = batch["mask"]
    ratio, _, _ = stub._vllm_importance_sampling_ratio(batch["old"], batch["rollout"], mask)
    total = torch.zeros((), dtype=torch.float64)
    size = ROWS // MICRO_BATCHES
    for start in range(0, ROWS, size):
        rows = slice(start, start + size)
        stub.rows = rows
        inputs = {
            "prompt_ids": torch.ones(size, PROMPT, dtype=torch.long),
            "prompt_mask": torch.ones(size, PROMPT, dtype=torch.long),
            "completion_ids": torch.ones(size, RESPONSE, dtype=torch.long),
            "completion_mask": mask[rows],
            "advantages": advantages[rows],
            "old_per_token_logps": batch["old"][rows],
            "sampling_per_token_logps": batch["rollout"][rows],
            "ref_per_token_logps": batch["ref"][rows],
            "importance_sampling_ratio": ratio[rows],
            "num_items_in_batch": mask.sum(),
        }
        total = total + GRPOTrainer._compute_loss(cast(Any, stub), None, inputs)
    (gradient,) = torch.autograd.grad(total, current)
    return total.detach(), gradient, ratio


# ---------------------------------------------------------------------------
# veRL side


def _verl_configs(tmp_path: Path) -> tuple[Any, Any, list[str]]:
    import verl
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    revision = next(
        revision
        for revision, (_, names) in reversed(_FORK_NATIVE_NAME_REVISIONS.items())
        if {"active_sampling", "trl_sampler_correction"} <= names
    )
    request = _request(f"verl@{revision[:7]}", source_revision=revision)
    manifest = build_grpo_launch_plan(request, tmp_path / "verl")
    from posttrain.train.backends.verl import worker

    original = worker._model_path
    worker._model_path = lambda model: "/models/parity"
    try:
        overrides = build_hydra_overrides(manifest, tmp_path / "r.parquet", tmp_path / "a.json", tmp_path / "c")
    finally:
        worker._model_path = original
    config_dir = Path(str(verl.__file__)).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    algorithm = omega_conf_to_dataclass(config.algorithm)
    return actor, algorithm, overrides


def _verl_update(actor: Any, algorithm: Any, batch: dict[str, Any], rewards: Any):
    import numpy as np
    from tensordict import TensorDict
    from verl import DataProto
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.losses import ppo_loss

    mask = batch["mask"]
    token_level_rewards = torch.zeros(ROWS, RESPONSE, dtype=torch.float64)
    for row, length in enumerate(batch["lengths"]):
        token_level_rewards[row, length - 1] = rewards[row]
    index = np.array([f"group-{row // GENERATIONS}" for row in range(ROWS)], dtype=object)
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards, mask, index, norm_adv_by_std_in_grpo=algorithm.norm_adv_by_std_in_grpo
    )
    proto = DataProto.from_dict(
        tensors={"old_log_probs": batch["old"], "rollout_log_probs": batch["rollout"], "response_mask": mask.clone()}
    )
    proto, _ = compute_rollout_correction_and_add_to_batch(proto, algorithm.rollout_correction)
    weights = proto.batch["rollout_is_weights"]
    torch.testing.assert_close(proto.batch["response_mask"], mask, rtol=0, atol=0)

    current = batch["current"].clone().requires_grad_(True)
    total = torch.zeros((), dtype=torch.float64)
    size = ROWS // MICRO_BATCHES
    for start in range(0, ROWS, size):
        rows = slice(start, start + size)
        # Flat model output over prompt + response tokens; veRL left-shifts it by one token.
        flat = torch.cat(
            [
                torch.cat(
                    [torch.zeros(PROMPT - 1, dtype=torch.float64), current[row], torch.zeros(1, dtype=torch.float64)]
                )
                for row in range(start, start + size)
            ]
        )
        data = TensorDict(
            {
                "prompts": torch.ones(size, PROMPT, dtype=torch.long),
                "responses": torch.ones(size, RESPONSE, dtype=torch.long),
                "attention_mask": torch.ones(size, PROMPT + RESPONSE, dtype=torch.long),
                "response_mask": mask[rows],
                "old_log_probs": batch["old"][rows],
                "advantages": advantages[rows],
                "rollout_is_weights": weights[rows],
                "ref_log_prob": batch["ref"][rows],
            },
            batch_size=[size],
        )
        # The FSDP engine all-reduces the mini-batch's trainable-token count.
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(mask.sum().item()), global_batch_size=ROWS)
        loss, _ = ppo_loss(actor, {"log_probs": flat}, data)
        total = total + loss
    (gradient,) = torch.autograd.grad(total, current)
    return advantages, weights, total.detach(), gradient


def test_olmo3_batch_gives_identical_advantages_loss_and_gradient(tmp_path: Path) -> None:
    batch = _batch()
    settings = _settings()
    trl_rewards = torch.tensor(
        [
            shape_online_reward(settings, reward, length, is_truncated=truncated)
            for reward, length, truncated in zip(
                batch["task_rewards"], batch["lengths"], batch["truncated"], strict=True
            )
        ],
        dtype=torch.float64,
    )
    payload = grpo_algorithm_payload(settings)
    verl_rewards = torch.tensor(
        [
            shaped_rollout_reward(
                _environment_rollout(reward, length, truncated),
                max_completion_tokens=payload["max_completion_length"],
                overlong_buffer_tokens=payload["overlong_buffer_tokens"],
                overlong_penalty_factor=payload["overlong_penalty_factor"],
                truncation_penalty=payload["truncation_penalty"],
            )
            for reward, length, truncated in zip(
                batch["task_rewards"], batch["lengths"], batch["truncated"], strict=True
            )
        ],
        dtype=torch.float64,
    )
    assert torch.equal(trl_rewards, verl_rewards)
    assert trl_rewards[3] == pytest.approx(-TRUNCATION_PENALTY)

    trl_config = _trl_config(tmp_path)
    assert (trl_config.loss_type, trl_config.scale_rewards, trl_config.beta) == ("dapo", "none", BETA)
    trl_advantages = _trl_advantages(trl_rewards)
    trl_loss, trl_gradient, trl_ratio = _trl_update(trl_config, batch, trl_advantages)

    actor, algorithm, overrides = _verl_configs(tmp_path)
    assert actor.policy_loss.loss_mode == "token_clip" and actor.kl_loss_type == "k3_unclipped", "config"
    verl_advantages, verl_weights, verl_loss, verl_gradient = _verl_update(actor, algorithm, batch, verl_rewards)

    mask = batch["mask"].bool()
    expanded = trl_advantages.unsqueeze(1).expand(ROWS, RESPONSE)
    assert torch.equal(verl_advantages[mask], expanded[mask])
    torch.testing.assert_close(verl_weights[mask], trl_ratio[mask], rtol=1e-12, atol=1e-12)
    # The batch must exercise every branch of the objective.
    ratio = torch.exp(batch["current"] - batch["old"])
    assert ((ratio > 1.272) & mask & (expanded > 0)).any() and ((ratio < 0.8) & mask & (expanded < 0)).any()
    assert ((torch.exp(batch["old"] - batch["rollout"]) > 2.0) & mask).any()
    assert ((ratio > 3.0) & mask & (expanded < 0)).any()
    ref_minus_current = batch["ref"] - batch["current"]
    assert ((torch.expm1(ref_minus_current) - ref_minus_current > 10) & mask).any()
    torch.testing.assert_close(verl_loss, trl_loss, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(verl_gradient, trl_gradient, rtol=1e-10, atol=1e-12)
    assert overrides  # recorded for failure diagnostics
