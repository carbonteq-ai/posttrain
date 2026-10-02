"""One SAMPO batch through TRL's and veRL's real advantage and loss code gives the same update.

TRL side: Posttrain's ``compute_sampo_advantages`` (the TRL SAMPO path's
precomputed advantages, on shaped rewards) and TRL post14's
``GRPOTrainer._compute_loss`` configured by Posttrain's ``_online_rl_arguments``
(sequence-level ratio, clip 0.003/0.004, token-truncated vLLM correction capped
at 2, k3 KL). veRL side: the fork's ``compute_sampo_outcome_advantage``, rollout
correction and ``ppo_loss`` configured by the Hydra overrides Posttrain
generates for the same ``SAMPOSettings``. Needs the parity environment of
``docs/plan/verl-vortex-port.md``; skips elsewhere.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl.trainer.grpo_trainer", reason="requires CarbonTeq TRL 1.12.0.post14")
core_algos = pytest.importorskip("verl.trainer.ppo.core_algos", reason="requires the CarbonTeq veRL fork")
if "sampo_token_credit" not in getattr(core_algos, "POLICY_LOSS_REGISTRY", {}):
    pytest.skip("installed veRL has no qualified sampo_token_credit loss", allow_module_level=True)

from posttrain.common import ExecutionTarget, InferenceBinding, TraceObservation  # noqa: E402
from posttrain.common.variants import QWEN_35_2B  # noqa: E402
from posttrain.data import RolloutDataset, RolloutExample  # noqa: E402
from posttrain.train import (  # noqa: E402
    QWEN35_RENDERER,
    ActiveGroupSampling,
    AgenticTurn,
    LoRAUpdate,
    SAMPORequest,
    SAMPOSettings,
    TrainingBinding,
    TrainingLoop,
    TrainingRuntime,
)
from posttrain.train.backends.trl.policy_config import _online_rl_arguments  # noqa: E402
from posttrain.train.backends.verl.launcher import build_sampo_launch_plan, sampo_algorithm_payload  # noqa: E402
from posttrain.train.backends.verl.reward_fields import shaped_rollout_reward  # noqa: E402
from posttrain.train.backends.verl.worker import build_hydra_overrides  # noqa: E402
from posttrain.train.online_rl import EnvironmentRollout  # noqa: E402
from posttrain.train.profiles import shape_online_reward  # noqa: E402
from posttrain.train.sampo_advantages import compute_sampo_advantages  # noqa: E402

GROUPS = 2
GENERATIONS = 3
ROWS = GROUPS * GENERATIONS
PROMPT = 4
RESPONSE = 12
MICRO_BATCHES = 2
BETA = 0.005
REVISION = "d8e472db822f2916ed81a408b8d28192be95e678"


@dataclass(frozen=True)
class _Environment:
    id: str = "envs/sampo-parity@1"
    revision: str = "1"
    category: str = "tool-agentic"


@dataclass
class _Bridge:
    dataset: RolloutDataset = field(
        default_factory=lambda: RolloutDataset(
            "sampo-parity-v1", "a" * 40, (RolloutExample("task-a", "a", {}), RolloutExample("task-b", "b", {}))
        )
    )
    max_concurrent: int = 8

    async def run(self, batch, generator):  # pragma: no cover - never called
        raise AssertionError


def _settings(normalization: str) -> SAMPOSettings:
    return SAMPOSettings(
        "settings/sampo-parity@1",
        TrainingLoop(
            max_steps=1,
            max_length=RESPONSE + PROMPT,
            per_device_batch_size=ROWS // MICRO_BATCHES,
            gradient_accumulation_steps=MICRO_BATCHES,
            lr_scheduler_type="constant",
        ),
        num_prompts_per_step=GROUPS,
        num_generations=GENERATIONS,
        max_prompt_length=PROMPT,
        max_completion_length=RESPONSE,
        beta=BETA,
        discount_gamma=0.9,
        step_advantage_weight=0.7,
        advantage_normalization=normalization,  # type: ignore[arg-type]
        active_sampling=ActiveGroupSampling(3),
        truncation_penalty=0.2,
    )


def _request(settings: SAMPOSettings, backend: str) -> SAMPORequest:
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
        LoRAUpdate(rank=8, alpha=16, target_modules=r".*\\.(o_proj|down_proj)$"),
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
        {"mode": "colocate", "sleep_during_optimization": True, "max_model_len": PROMPT + RESPONSE, "max_num_seqs": 8},
        {"temperature": 1.0, "max_tokens": RESPONSE},
        target,
        ("rollout",),
    )
    return SAMPORequest(QWEN_35_2B, cast(Any, _Bridge()), settings, _Environment(), training, inference)


# Per rollout: (task reward, truncated, turns as (start, end, anchor, step reward)); gaps are tool tokens.
_TRAJECTORIES = [
    ("task-a", 1.0, False, [(0, 3, "s0", 0.2), (5, 8, "s1", 0.0), (9, 11, "s2", 0.8)]),
    ("task-a", 0.0, True, [(0, 3, "s0", 0.0), (5, 9, "s1x", 0.0), (10, 12, "s2", 0.0)]),
    ("task-a", 0.5, False, [(0, 2, "s0", 0.1), (4, 7, "s1", 0.4)]),
    ("task-b", 0.0, False, [(0, 4, "t0", None), (6, 9, "t1", None)]),
    ("task-b", 1.0, False, [(0, 4, "t0", None), (6, 8, "t1", None), (9, 12, "t2", None)]),
    ("task-b", 0.25, True, [(0, 4, "t0", None), (5, 10, "t9", None)]),
]


def _rollouts() -> list[EnvironmentRollout]:
    rollouts = []
    for index, (example_id, reward, truncated, turns) in enumerate(_TRAJECTORIES):
        length = max(end for _, end, _, _ in turns)
        env_mask = [False] * length
        for start, end, _, _ in turns:
            env_mask[start:end] = [True] * (end - start)
        rollouts.append(
            EnvironmentRollout(
                example_id=example_id,
                prompt_ids=(1,) * PROMPT,
                completion_ids=(2,) * length,
                sampling_logprobs=(-0.1,) * length,
                env_mask=tuple(env_mask),
                reward=reward,
                is_truncated=truncated,
                trace=TraceObservation("test", f"trace-{index}", {}),
                turns=tuple(AgenticTurn(start, end, anchor, step) for start, end, anchor, step in turns),
            )
        )
    return rollouts


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


def _logprobs(mask: Any) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(7)
    rollout = -torch.rand(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 2.0 - 0.05
    old = rollout + torch.randn(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 0.5
    # Small per-token drift: some sequence ratios inside, some outside the 0.003/0.004 clip range.
    current = old + torch.randn(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 0.004
    current[0] = old[0] + 0.0005
    current[3] = old[3] + 0.006  # sequence ratio above 1 + 0.004
    current[4] = old[4] - 0.006  # sequence ratio below 1 - 0.003
    ref = old + torch.randn(ROWS, RESPONSE, generator=generator, dtype=torch.float64) * 0.3
    return {"mask": mask, "rollout": rollout, "old": old, "current": current, "ref": ref}


@pytest.mark.parametrize("normalization", ["mean", "mean_std"])
def test_sampo_batch_gives_identical_advantages_evidence_loss_and_gradient(tmp_path: Path, normalization: str) -> None:
    import numpy as np
    import verl
    from hydra import compose, initialize_config_dir
    from tensordict import TensorDict
    from trl.trainer.grpo_config import GRPOConfig
    from verl import DataProto
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
    from verl.utils import tensordict_utils as tu
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.utils.losses import ppo_loss

    settings = _settings(normalization)
    rollouts = _rollouts()
    payload = sampo_algorithm_payload(settings)

    # --- advantages -------------------------------------------------------
    trl_shaped = [
        shape_online_reward(settings, r.reward, len(r.completion_ids), is_truncated=r.is_truncated) for r in rollouts
    ]
    verl_shaped = [
        shaped_rollout_reward(
            r,
            max_completion_tokens=payload["max_completion_length"],
            overlong_buffer_tokens=None,
            overlong_penalty_factor=payload["overlong_penalty_factor"],
            truncation_penalty=payload["truncation_penalty"],
        )
        for r in rollouts
    ]
    assert trl_shaped == verl_shaped
    trl = compute_sampo_advantages(
        settings,
        [r.example_id for r in rollouts],
        [replace(r, reward=shaped) for r, shaped in zip(rollouts, trl_shaped, strict=True)],
    )
    trl_advantages = torch.zeros(ROWS, RESPONSE, dtype=torch.float64)
    mask = torch.zeros(ROWS, RESPONSE, dtype=torch.float64)
    for row, (rollout, values) in enumerate(zip(rollouts, trl.token_advantages, strict=True)):
        trl_advantages[row, : len(values)] = torch.tensor(values, dtype=torch.float64)
        mask[row, : len(rollout.env_mask)] = torch.tensor(rollout.env_mask, dtype=torch.float64)

    manifest = build_sampo_launch_plan(_request(settings, f"verl@{REVISION[:7]}"), tmp_path / "verl")
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
    assert actor.policy_loss.loss_mode == "sampo_token_credit" and actor.kl_loss_type == "k3_unclipped"

    token_level_rewards = torch.zeros(ROWS, RESPONSE, dtype=torch.float64)
    for row, shaped in enumerate(verl_shaped):
        token_level_rewards[row, int(mask[row].nonzero().max())] = shaped

    def objects(values: list[Any]) -> Any:
        array = np.empty(len(values), dtype=object)
        array[:] = values
        return array

    metrics: dict[str, float] = {}
    verl_advantages, _ = core_algos.compute_sampo_outcome_advantage(
        token_level_rewards,
        mask,
        np.array([f"uid-{row // GENERATIONS}" for row in range(ROWS)], dtype=object),
        objects([[[start, end] for start, end, _, _ in turns] for _, _, _, turns in _TRAJECTORIES]),
        objects([[anchor for _, _, anchor, _ in turns] for _, _, _, turns in _TRAJECTORIES]),
        objects([[step for _, _, _, step in turns] for _, _, _, turns in _TRAJECTORIES]),
        num_repeat=GENERATIONS,
        config=algorithm,
        metrics=metrics,
    )
    torch.testing.assert_close(verl_advantages, trl_advantages, rtol=1e-12, atol=1e-12)
    assert (trl_advantages.abs() > 0).any()
    evidence = trl.hierarchy_evidence(settings.step_advantage_weight)
    for name, (value, _count) in evidence.items():
        assert metrics[name.replace("train/rl/", "sampo/")] == pytest.approx(value, rel=1e-12, abs=1e-12)
    sizes = [size for values in trl.anchor_group_sizes for size in values]
    assert metrics["sampo/anchor_group_size_mean"] == pytest.approx(sum(sizes) / len(sizes))

    # --- loss ------------------------------------------------------------
    batch = _logprobs(mask)
    trl_request = _request(settings, "trl@1.12.0.post14")
    arguments = _online_rl_arguments(trl_request, tmp_path / "trl", {"enable_thinking": False})
    arguments.update(bf16=False, fp16=False, use_cpu=True, report_to=[])
    trl_config = GRPOConfig(**arguments)
    assert trl_config.importance_sampling_level == "sequence" and trl_config.loss_type == "grpo"
    trl_loss, trl_gradient, trl_ratio = _trl_update(trl_config, batch, trl_advantages)

    proto = DataProto.from_dict(
        tensors={"old_log_probs": batch["old"], "rollout_log_probs": batch["rollout"], "response_mask": mask.clone()}
    )
    proto, _ = compute_rollout_correction_and_add_to_batch(proto, algorithm.rollout_correction)
    weights = proto.batch["rollout_is_weights"]
    torch.testing.assert_close(weights[mask.bool()], trl_ratio[mask.bool()], rtol=1e-12, atol=1e-12)
    current = batch["current"].clone().requires_grad_(True)
    total = torch.zeros((), dtype=torch.float64)
    size = ROWS // MICRO_BATCHES
    for start in range(0, ROWS, size):
        rows = slice(start, start + size)
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
                "advantages": verl_advantages[rows],
                "rollout_is_weights": weights[rows],
                "ref_log_prob": batch["ref"][rows],
            },
            batch_size=[size],
        )
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(mask.sum().item()), global_batch_size=ROWS)
        loss, _ = ppo_loss(actor, {"log_probs": flat}, data)
        total = total + loss
    (verl_gradient,) = torch.autograd.grad(total, current)
    # veRL's seq-mean-token-mean divides by (tokens + 1e-8) where TRL clamps at 1.
    torch.testing.assert_close(total.detach(), trl_loss, rtol=1e-8, atol=1e-12)
    torch.testing.assert_close(verl_gradient, trl_gradient, rtol=1e-8, atol=1e-12)
    assert trl_gradient.abs().sum() > 0
    ratios = torch.exp(((batch["current"] - batch["old"]) * mask).sum(-1) / mask.sum(-1))
    assert ((ratios > 1.004) | (ratios < 0.997)).any() and ((ratios < 1.004) & (ratios > 0.997)).any()
