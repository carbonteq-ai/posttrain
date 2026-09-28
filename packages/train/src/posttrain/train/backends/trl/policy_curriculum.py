"""TRL composition for pre-rollout adaptive task selection."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ...adaptive_curriculum import CURRICULUM_SNAPSHOT_NAME
from ...adaptive_curriculum_runtime import AdaptiveCurriculumRuntime

_SNAPSHOT_NAME = CURRICULUM_SNAPSHOT_NAME
_LOGGER = logging.getLogger(__name__)


def adaptive_curriculum_trainer_type(parent: type[Any], runtime: AdaptiveCurriculumRuntime) -> type[Any]:
    """Select tasks before generation and observe raw rewards afterwards."""

    if os.environ.get("POSTTRAIN_ACTIVE_SAMPLING_AUDIT") == "1":
        print(f"posttrain active-sampling audit wrapper_parent={parent.__module__}.{parent.__qualname__}", flush=True)

    class AdaptiveCurriculumTrainer(parent):
        def _prepare_inputs(self, generation_batch: Any) -> dict[str, Any]:
            if self.model.training:
                if getattr(self.accelerator, "num_processes", 1) != 1:
                    raise RuntimeError("adaptive curriculum currently requires one training process")
                generate_every = self.args.steps_per_generation * self.num_iterations
                if not getattr(self, "active_sampling", False) and (
                    self._step % generate_every == 0 or self._buffered_inputs is None
                ):
                    generation_batch = runtime.select_generation_batch(
                        cast(Sequence[Mapping[str, object]], generation_batch),
                        step=int(self.state.global_step) + 1,
                        selection_kind="initial_batch",
                    )
            return cast(dict[str, Any], super()._prepare_inputs(generation_batch))

        def _prepare_active_sampling_inputs(self, candidate_inputs: Any) -> dict[str, Any]:
            if os.environ.get("POSTTRAIN_ACTIVE_SAMPLING_AUDIT") == "1":
                print(
                    "posttrain active-sampling audit wrapper_entry "
                    f"enabled={getattr(self, 'active_sampling', False)} candidate_rows={len(candidate_inputs)}",
                    flush=True,
                )
            if not getattr(self, "active_sampling", False):
                return cast(dict[str, Any], super()._prepare_active_sampling_inputs(candidate_inputs))
            return _prepare_adaptive_active_sampling_inputs(
                self,
                cast(list[dict[str, Any]], candidate_inputs),
                runtime,
                step=int(self.state.global_step) + 1,
            )

        def _calculate_rewards(
            self,
            inputs: list[dict[str, Any]],
            prompts: list[Any],
            completions: list[Any],
            completion_ids_list: list[list[int]],
        ) -> Any:
            native_rewards = _native_algorithm_reward_rows(self, inputs)
            self._posttrain_native_group_reward_std = (
                _native_group_reward_std(self, inputs, reward_rows=native_rewards)
                if native_rewards is not None
                else None
            )
            rewards = (
                native_rewards
                if native_rewards is not None
                else super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
            )
            if self.model.training:
                reward_rows = _nested_floats(rewards)
                weights = _flat_floats(self.reward_weights)
                runtime.observe_rewards(
                    inputs,
                    reward_rows,
                    weights,
                    step=int(self.state.global_step) + 1,
                )
            return rewards

    return AdaptiveCurriculumTrainer


def _native_algorithm_reward_rows(trainer: Any, inputs: Sequence[Mapping[str, Any]]) -> Any | None:
    """Use rewards computed by a native rollout instead of scoring them again.

    The Posttrain rollout bridge owns environment execution and reward shaping. Its
    ``algorithm_reward`` field is therefore the authoritative learner reward for
    adaptive native rollouts. Routing that value back through a generic TRL reward
    callback creates a second, implicit reward contract at exactly the active-
    sampling boundary.
    """
    import torch

    present = ["algorithm_reward" in row for row in inputs]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError("native rollout algorithm rewards must be present for every completion")
    reward_funcs = getattr(trainer, "reward_funcs", ())
    if len(reward_funcs) != 1:
        raise RuntimeError("native scalar algorithm rewards require exactly one configured TRL reward function")
    raw_values = [row["algorithm_reward"] for row in inputs]
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in raw_values):
        raise RuntimeError(
            "native rollout algorithm rewards must be scalar numbers; structured environment rewards "
            "must be projected by the rollout bridge before trainer admission"
        )
    values = [float(value) for value in raw_values]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("native rollout algorithm rewards must be finite")
    local = torch.tensor(values, dtype=torch.float32, device=trainer.accelerator.device).unsqueeze(1)
    return trainer.accelerator.gather(local)


def _prepare_adaptive_active_sampling_inputs(
    trainer: Any,
    candidate_inputs: list[dict[str, Any]],
    runtime: AdaptiveCurriculumRuntime,
    *,
    step: int,
) -> dict[str, Any]:
    """Choose each OLMo refill after observing earlier rounds at the same weights."""
    import torch
    from trl.trainer.rollout_admission import NoAdmittedRollouts

    if any("image" in row or "images" in row for row in candidate_inputs):
        raise NotImplementedError("adaptive active sampling currently supports text-only GRPO datasets")

    max_batches = int(trainer.active_sampling_max_batches)
    target_size = len(candidate_inputs) // max_batches
    if target_size == 0 or len(candidate_inputs) % max_batches != 0:
        raise RuntimeError("adaptive active sampling received an incomplete candidate generation batch")

    # Oversampling mirrors TRL's GRPOTrainer: extra prompt groups in the first round
    # and in each refill, no round larger than the first, cut to the reservation.
    # Trainers from TRL releases before these settings never oversample.
    first_round_extra_rows = int(getattr(trainer, "active_sampling_oversample", 0)) * trainer.num_generations
    refill_extra_rows = int(getattr(trainer, "active_sampling_oversample_refill", 0)) * trainer.num_generations
    oversampling = bool(first_round_extra_rows or refill_extra_rows)
    retained_batches: list[dict[str, Any]] = []
    retained_count = 0
    candidate_count = 0
    candidate_cursor = 0
    generation_rounds = 0
    oversampled_count = 0
    for round_index in range(1, max_batches + 1):
        local_missing = max(target_size - retained_count, 0)
        missing_by_process = trainer.accelerator.gather(torch.tensor(local_missing, device=trainer.accelerator.device))
        synchronized_missing = int(missing_by_process.max().item())
        if synchronized_missing == 0:
            break
        if synchronized_missing % trainer.num_generations != 0:
            raise RuntimeError("adaptive active sampling refill size must contain complete prompt groups")
        round_size = synchronized_missing
        if oversampling:
            requested = min(
                synchronized_missing + (refill_extra_rows if round_index > 1 else first_round_extra_rows),
                target_size + first_round_extra_rows,
            )
            round_size = max(min(requested, len(candidate_inputs) - candidate_cursor), synchronized_missing)
        if candidate_cursor + round_size > len(candidate_inputs):
            raise RuntimeError("adaptive active sampling exhausted its bounded candidate capacity")

        candidate_batch = runtime.select_task_groups(
            round_size // trainer.num_generations,
            step=step,
            selection_kind="active_sampling_refill",
            round_index=round_index,
        )
        candidate_cursor += round_size
        oversampled_count += round_size - synchronized_missing

        try:
            scored_batch = trainer._generate_and_score_completions(candidate_batch)
        except NoAdmittedRollouts:
            generation_rounds += 1
            candidate_count += len(candidate_batch)
            continue
        trl_group_reward_std = scored_batch.pop("group_reward_std")
        native_group_reward_std = getattr(trainer, "_posttrain_native_group_reward_std", None)
        trainer._posttrain_native_group_reward_std = None
        group_reward_std = native_group_reward_std if native_group_reward_std is not None else trl_group_reward_std
        if native_group_reward_std is not None:
            if native_group_reward_std.shape != trl_group_reward_std.shape:
                raise RuntimeError("native and TRL group reward statistics are misaligned")
            comparable = native_group_reward_std.isfinite() & trl_group_reward_std.isfinite()
            delta = (native_group_reward_std[comparable] - trl_group_reward_std[comparable]).abs()
            trainer._metrics["train"]["active_sampling/native_reward_std_max_delta"].append(
                float(delta.max().item()) if delta.numel() else 0.0
            )
            trainer._metrics["train"]["active_sampling/native_reward_std_disagreement_fraction"].append(
                float(
                    (~native_group_reward_std.isclose(trl_group_reward_std, rtol=1e-5, atol=1e-7, equal_nan=True))
                    .float()
                    .mean()
                    .item()
                )
                if native_group_reward_std.numel()
                else 0.0
            )
        keep = group_reward_std > trainer.active_sampling_reward_std_epsilon
        native_group_values = (
            native_group_reward_std[:: trainer.num_generations].detach().cpu().tolist()
            if native_group_reward_std is not None
            else None
        )
        trl_group_values = trl_group_reward_std[:: trainer.num_generations].detach().cpu().tolist()
        kept_rows = int(keep.sum().item())
        _LOGGER.info(
            "adaptive active-sampling round=%d requested_rows=%d kept_rows=%d native_group_std=%s trl_group_std=%s",
            round_index,
            len(candidate_batch),
            kept_rows,
            native_group_values,
            trl_group_values,
        )
        if os.environ.get("POSTTRAIN_ACTIVE_SAMPLING_AUDIT") == "1":
            print(
                "posttrain active-sampling audit "
                f"round={round_index} requested_rows={len(candidate_batch)} kept_rows={kept_rows} "
                f"native_group_std={native_group_values} trl_group_std={trl_group_values}",
                flush=True,
            )
        round_metrics = {
            "train/rl/active_sampling_round_requested_rows": len(candidate_batch),
            "train/rl/active_sampling_round_kept_rows": kept_rows,
            "train/rl/active_sampling_round_kept_fraction": kept_rows / len(candidate_batch),
        }
        if oversampling:
            # Rows the round needed; the rest of the requested rows were oversampled.
            round_metrics["train/rl/active_sampling_round_missing_rows"] = synchronized_missing
        runtime.context.metrics(round_metrics, step=step, attributes={"round_index": round_index})
        generation_rounds += 1
        candidate_count += len(candidate_batch)
        if keep.any():
            retained_batches.append(trainer._select_dynamic_sampling_rows(scored_batch, keep))
            retained_count += int(keep.sum().item())

    local_ready = torch.tensor(retained_count >= target_size, device=trainer.accelerator.device)
    all_ready = trainer.accelerator.gather(local_ready)
    if not all_ready.all():
        retained_counts = trainer.accelerator.gather(
            torch.tensor(retained_count, device=trainer.accelerator.device)
        ).tolist()
        rejections = getattr(trainer, "_posttrain_admission_rejections", ())
        raise RuntimeError(
            "adaptive active sampling exhausted "
            f"{max_batches} generation rounds before every process filled its generation batch; "
            f"retained rows by process: {retained_counts}"
            + (f"; rollout admission rejected groups: {sorted(rejections)}" if rejections else "")
        )

    batch = trainer._concatenate_dynamic_sampling_batches(retained_batches)
    batch = trainer._select_dynamic_sampling_rows(
        batch,
        torch.arange(len(batch["completion_ids"]), device=trainer.accelerator.device) < target_size,
    )
    local_tokens = batch["completion_mask"].sum()
    batch["num_items_in_batch"] = trainer.accelerator.gather(local_tokens).sum()
    trainer._metrics["train"]["active_sampling/generation_rounds"].append(generation_rounds)
    trainer._metrics["train"]["active_sampling/retained_fraction"].append(retained_count / candidate_count)
    trainer._metrics["train"]["active_sampling/generated_rows"].append(candidate_count)
    trainer._metrics["train"]["active_sampling/candidate_groups_reserved"].append(len(candidate_inputs))
    trainer._metrics["train"]["active_sampling/candidate_groups_generated"].append(candidate_count)
    trainer._metrics["train"]["active_sampling/candidate_groups_retained"].append(retained_count)
    trainer._metrics["train"]["active_sampling/candidate_groups_unused"].append(
        len(candidate_inputs) - candidate_cursor
    )
    # The candidate_groups_* series above count rows (completions), and the
    # TRL frac_reward_zero_std / advantages/zero_fraction series average every
    # generation round before filtering. Report the batch actually optimized.
    trainer._metrics["train"]["active_sampling/retained_groups"].append(retained_count / trainer.num_generations)
    trainer._metrics["train"]["active_sampling/generated_groups"].append(candidate_count / trainer.num_generations)
    if oversampling:
        trainer._metrics["train"]["active_sampling/oversampled_groups"].append(
            oversampled_count // trainer.num_generations
        )
        trainer._metrics["train"]["active_sampling/discarded_groups"].append(
            max(retained_count - target_size, 0) // trainer.num_generations
        )
    advantages = batch.get("advantages")
    if isinstance(advantages, torch.Tensor) and advantages.numel() % trainer.num_generations == 0:
        zero = advantages.detach().abs().reshape(-1, trainer.num_generations) <= 1e-8
        trainer._metrics["train"]["trained/frac_reward_zero_std"].append(zero.all(dim=1).float().mean().item())
        trainer._metrics["train"]["trained/advantages_zero_fraction"].append(zero.float().mean().item())
    return cast(dict[str, Any], batch)


def _native_group_reward_std(
    trainer: Any,
    inputs: Sequence[Mapping[str, Any]],
    *,
    reward_rows: Any | None = None,
) -> Any | None:
    """Derive local admission variance from authoritative native rollout rewards."""
    import torch
    from trl.trainer.utils import nanstd

    reward_rows = _native_algorithm_reward_rows(trainer, inputs) if reward_rows is None else reward_rows
    if reward_rows is None:
        return None
    group_size = int(trainer.num_generations)
    if len(inputs) % group_size != 0:
        raise RuntimeError("native rollout rewards must contain complete prompt groups")
    scalar_rewards = reward_rows.squeeze(1)
    if getattr(trainer, "mask_truncated_completions", False):
        local_truncated = torch.tensor(
            [bool(row.get("is_truncated", False)) for row in inputs],
            dtype=torch.bool,
            device=trainer.accelerator.device,
        )
        truncated = trainer.accelerator.gather(local_truncated)
        if truncated.shape != scalar_rewards.shape:
            raise RuntimeError("native rollout rewards and truncation masks are misaligned")
        scalar_rewards = scalar_rewards.masked_fill(truncated, torch.nan)
    group_std = nanstd(scalar_rewards.view(-1, group_size), dim=1).repeat_interleave(group_size)
    process_index = int(getattr(trainer.accelerator, "process_index", 0))
    local_size = len(inputs)
    return group_std[process_index * local_size : (process_index + 1) * local_size]


def _nested_floats(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [[float(item) for item in row] for row in value]


def _flat_floats(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [float(item) for item in value]


__all__ = [
    "AdaptiveCurriculumRuntime",
    "adaptive_curriculum_trainer_type",
]
