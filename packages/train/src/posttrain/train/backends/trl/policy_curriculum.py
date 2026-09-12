"""TRL composition for pre-rollout adaptive task selection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from posttrain.common import JsonValue, RunContext

from ...adaptive_curriculum import (
    AdaptiveCurriculumController,
    CurriculumDecision,
    CurriculumObservation,
    QueuedJsonlCurriculumStateBackend,
)
from ...profiles import AdaptiveCurriculum

_SNAPSHOT_NAME = "adaptive-curriculum-state.json"


class AdaptiveCurriculumRuntime:
    """Bind task-neutral controller state to concrete TRL dataset rows."""

    def __init__(
        self,
        context: RunContext,
        rows: Sequence[Mapping[str, object]],
        settings: AdaptiveCurriculum,
        *,
        num_generations: int,
        state_dir: Path,
        resume_checkpoint: Path | None,
    ) -> None:
        task_rows: dict[str, dict[str, object]] = {}
        task_classes: dict[str, str] = {}
        for row in rows:
            task_id = row.get("example_id")
            class_id = row.get(settings.class_field)
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("adaptive curriculum requires a non-empty example_id on every rollout task")
            if task_id in task_rows:
                raise ValueError(f"adaptive curriculum task identity is duplicated: {task_id!r}")
            if not isinstance(class_id, str) or not class_id:
                raise ValueError(
                    f"adaptive curriculum class field {settings.class_field!r} must be a non-empty string "
                    f"on task {task_id!r}"
                )
            task_rows[task_id] = dict(row)
            task_classes[task_id] = class_id
        if num_generations < 2:
            raise ValueError("adaptive curriculum requires at least two generations per task group")
        state_dir.mkdir(parents=True, exist_ok=True)
        backend = QueuedJsonlCurriculumStateBackend(state_dir / "journal.jsonl")
        restored_state = None
        if resume_checkpoint is not None:
            snapshot_path = resume_checkpoint / _SNAPSHOT_NAME
            if not snapshot_path.is_file():
                backend.close()
                raise RuntimeError(
                    f"adaptive curriculum recovery snapshot is missing from checkpoint {resume_checkpoint}"
                )
            restored_state = backend.read_snapshot(snapshot_path)
        self.context = context
        self.settings = settings
        self.num_generations = num_generations
        self.state_dir = state_dir
        self.task_rows = task_rows
        try:
            self.controller = AdaptiveCurriculumController(
                task_classes,
                settings,
                backend,
                group_size=num_generations,
                restored_state=restored_state,
            )
        except BaseException:
            backend.close()
            raise
        context.event(
            "adaptive_curriculum_started",
            {
                "class_field": settings.class_field,
                "class_exploration": settings.class_exploration,
                "task_discovery": settings.task_discovery,
                "history_groups": settings.history_groups,
                "seed": settings.seed,
                "task_count": len(task_rows),
                "class_count": len(set(task_classes.values())),
                "restored": restored_state is not None,
            },
        )

    def select_generation_batch(
        self,
        generation_batch: Sequence[Mapping[str, object]],
        *,
        step: int,
        selection_kind: str = "initial_batch",
        round_index: int | None = None,
    ) -> list[dict[str, object]]:
        if len(generation_batch) % self.num_generations != 0:
            raise RuntimeError("adaptive curriculum received an incomplete prompt group batch")
        group_count = len(generation_batch) // self.num_generations
        return self.select_task_groups(
            group_count,
            step=step,
            selection_kind=selection_kind,
            round_index=round_index,
        )

    def select_task_groups(
        self,
        group_count: int,
        *,
        step: int,
        selection_kind: str,
        round_index: int | None = None,
    ) -> list[dict[str, object]]:
        decision = self.controller.select(
            group_count,
            step=step,
            selection_kind=selection_kind,
            round_index=round_index,
        )
        selected = [
            dict(self.task_rows[task_id])
            for task_id in decision.task_ids
            for _ in range(self.num_generations)
        ]
        self._emit_decision(decision)
        return selected

    def observe_rewards(
        self,
        inputs: Sequence[Mapping[str, object]],
        rewards_per_function: Sequence[Sequence[float]],
        reward_weights: Sequence[float],
        *,
        step: int,
    ) -> CurriculumObservation:
        if len(inputs) != len(rewards_per_function):
            raise RuntimeError("adaptive curriculum reward rows do not align with rollout inputs")
        if len(inputs) % self.num_generations != 0:
            raise RuntimeError("adaptive curriculum rewards do not contain complete prompt groups")
        total_rewards = [
            _weighted_reward(rewards, reward_weights) for rewards in rewards_per_function
        ]
        groups: list[tuple[str, Sequence[float]]] = []
        for offset in range(0, len(inputs), self.num_generations):
            rows = inputs[offset : offset + self.num_generations]
            task_ids = [row.get("example_id") for row in rows]
            task_id = task_ids[0]
            if not isinstance(task_id, str) or any(candidate != task_id for candidate in task_ids):
                raise RuntimeError("adaptive curriculum rollout group does not preserve one task identity")
            groups.append((task_id, total_rewards[offset : offset + self.num_generations]))
        observation = self.controller.observe(groups, step=step)
        self.context.event(
            "adaptive_curriculum_evidence_observed",
            cast(Mapping[str, JsonValue], observation.as_record()),
        )
        return observation

    def checkpoint(self, checkpoint: Path) -> None:
        self.controller.flush()
        self.controller.snapshot(checkpoint / _SNAPSHOT_NAME)
        self.context.event(
            "adaptive_curriculum_checkpointed",
            {
                "checkpoint": checkpoint.name,
                "decision_index": self.controller.decision_index,
            },
        )

    def save_final_state(self) -> Path:
        path = self.state_dir / _SNAPSHOT_NAME
        self.controller.snapshot(path)
        return path

    def close(self) -> None:
        self.controller.close()

    def _emit_decision(self, decision: CurriculumDecision) -> None:
        self.context.event(
            "adaptive_curriculum_allocation_selected",
            cast(Mapping[str, JsonValue], decision.as_record()),
        )


def adaptive_curriculum_trainer_type(parent: type[Any], runtime: AdaptiveCurriculumRuntime) -> type[Any]:
    """Select tasks before generation and observe raw rewards afterwards."""

    class AdaptiveCurriculumTrainer(parent):
        def _prepare_inputs(self, generation_batch: Any) -> dict[str, Any]:
            if self.model.training:
                if getattr(self.accelerator, "num_processes", 1) != 1:
                    raise RuntimeError("adaptive curriculum currently requires one training process")
                generate_every = self.args.steps_per_generation * self.num_iterations
                if (
                    not getattr(self, "active_sampling", False)
                    and (self._step % generate_every == 0 or self._buffered_inputs is None)
                ):
                    generation_batch = runtime.select_generation_batch(
                        cast(Sequence[Mapping[str, object]], generation_batch),
                        step=int(self.state.global_step) + 1,
                        selection_kind="initial_batch",
                    )
            return cast(dict[str, Any], super()._prepare_inputs(generation_batch))

        def _prepare_active_sampling_inputs(self, candidate_inputs: Any) -> dict[str, Any]:
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
            rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
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

    retained_batches: list[dict[str, Any]] = []
    retained_count = 0
    candidate_count = 0
    candidate_cursor = 0
    generation_rounds = 0
    for round_index in range(1, max_batches + 1):
        local_missing = max(target_size - retained_count, 0)
        missing_by_process = trainer.accelerator.gather(
            torch.tensor(local_missing, device=trainer.accelerator.device)
        )
        synchronized_missing = int(missing_by_process.max().item())
        if synchronized_missing == 0:
            break
        if synchronized_missing % trainer.num_generations != 0:
            raise RuntimeError("adaptive active sampling refill size must contain complete prompt groups")
        if candidate_cursor + synchronized_missing > len(candidate_inputs):
            raise RuntimeError("adaptive active sampling exhausted its bounded candidate capacity")

        candidate_batch = runtime.select_task_groups(
            synchronized_missing // trainer.num_generations,
            step=step,
            selection_kind="active_sampling_refill",
            round_index=round_index,
        )
        candidate_cursor += synchronized_missing

        try:
            scored_batch = trainer._generate_and_score_completions(candidate_batch)
        except NoAdmittedRollouts:
            generation_rounds += 1
            candidate_count += len(candidate_batch)
            continue
        group_reward_std = scored_batch.pop("group_reward_std")
        keep = group_reward_std > trainer.active_sampling_reward_std_epsilon
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
        raise RuntimeError(
            "adaptive active sampling exhausted "
            f"{max_batches} generation rounds before every process filled its generation batch; "
            f"retained rows by process: {retained_counts}"
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
    return cast(dict[str, Any], batch)


def _weighted_reward(rewards: Sequence[float], weights: Sequence[float]) -> float:
    if len(rewards) != len(weights):
        raise RuntimeError("adaptive curriculum reward functions do not align with their weights")
    values = [float(reward) * weight for reward, weight in zip(rewards, weights, strict=True) if math.isfinite(float(reward))]
    return sum(values) if values else float("nan")


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
