"""Backend-neutral binding of the adaptive curriculum controller to rollout task rows.

Both training backends use this runtime: the TRL trainer in-process, and the
veRL prompt selector (``backends/verl/curriculum.py``) inside veRL's trainer.
It turns controller decisions into task rows, observes group rewards, emits the
curriculum events and metrics, and writes controller snapshots into
checkpoints.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from posttrain.common import JsonValue, RunContext

from .adaptive_curriculum import (
    CURRICULUM_SNAPSHOT_NAME,
    AdaptiveCurriculumController,
    CurriculumDecision,
    CurriculumObservation,
    QueuedJsonlCurriculumStateBackend,
)
from .profiles import AdaptiveCurriculum

_SNAPSHOT_NAME = CURRICULUM_SNAPSHOT_NAME


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
        warm_start_state_dir: Path | None = None,
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
        warm_start_state = None
        if warm_start_state_dir is not None and resume_checkpoint is None:
            # A published adaptive-curriculum-state artifact is the source run's state
            # directory: its final snapshot plus the decision journal.
            snapshot_path = warm_start_state_dir / _SNAPSHOT_NAME
            if not snapshot_path.is_file():
                backend.close()
                raise RuntimeError(f"adaptive curriculum warm-start snapshot is missing from {warm_start_state_dir}")
            warm_start_state = backend.read_snapshot(snapshot_path)
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
                warm_start_state=warm_start_state,
            )
        except BaseException:
            backend.close()
            raise
        context.event(
            "adaptive_curriculum_started",
            {
                "policy": settings.policy,
                "class_field": settings.class_field,
                "class_exploration": settings.class_exploration if settings.policy == "quota" else None,
                "task_discovery": settings.task_discovery if settings.policy == "quota" else None,
                "exploration_share": settings.exploration_share if settings.policy == "yield_first" else None,
                "uncertainty_weight": settings.uncertainty_weight if settings.policy == "yield_first" else None,
                "history_groups": settings.history_groups,
                "seed": settings.seed,
                "task_count": len(task_rows),
                "class_count": len(set(task_classes.values())),
                "restored": restored_state is not None,
                "warm_started": warm_start_state is not None,
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
        selected = [dict(self.task_rows[task_id]) for task_id in decision.task_ids for _ in range(self.num_generations)]
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
        total_rewards = [_weighted_reward(rewards, reward_weights) for rewards in rewards_per_function]
        groups: list[tuple[str, Sequence[float]]] = []
        for offset in range(0, len(inputs), self.num_generations):
            rows = inputs[offset : offset + self.num_generations]
            task_ids = [row.get("example_id") for row in rows]
            task_id = task_ids[0]
            if not isinstance(task_id, str) or any(candidate != task_id for candidate in task_ids):
                raise RuntimeError("adaptive curriculum rollout group does not preserve one task identity")
            groups.append((task_id, total_rewards[offset : offset + self.num_generations]))
        return self.observe_groups(groups, step=step)

    def observe_groups(self, groups: Sequence[tuple[str, Sequence[float]]], *, step: int) -> CurriculumObservation:
        """Record each prompt group's rewards (one task per group) as curriculum evidence."""

        observation = self.controller.observe(groups, step=step)
        self.context.event(
            "adaptive_curriculum_evidence_observed",
            cast(Mapping[str, JsonValue], observation.as_record()),
        )
        return observation

    def capture_state(self) -> dict[str, object]:
        """Copy the controller state at an optimizer-step boundary.

        A cancellation checkpoint is written after the next rollout has already
        advanced the controller, so it restores this copy instead of the live state.
        """

        return copy.deepcopy(self.controller.state())

    def checkpoint(self, checkpoint: Path, state: Mapping[str, object] | None = None) -> None:
        self.controller.flush()
        if state is None:
            self.controller.snapshot(checkpoint / _SNAPSHOT_NAME)
        else:
            self.controller.backend.snapshot(checkpoint / _SNAPSHOT_NAME, state)
        decision_index = state.get("decision_index") if state is not None else self.controller.decision_index
        self.context.event(
            "adaptive_curriculum_checkpointed",
            {
                "checkpoint": checkpoint.name,
                "decision_index": decision_index if isinstance(decision_index, int) else None,
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
        attributes: dict[str, JsonValue] = {
            "selection_kind": decision.selection_kind,
            "round_index": decision.round_index,
        }
        self.context.metrics(
            {
                "train/rl/curriculum/candidate_groups": len(decision.task_ids),
                "train/rl/curriculum/unique_tasks": len(decision.task_ids) - decision.duplicate_fallbacks,
                "train/rl/curriculum/new_tasks": decision.new_tasks_selected,
                "train/rl/curriculum/discovery_reserved": decision.discovery_reserved,
                "train/rl/curriculum/discovery_fulfilled": decision.discovery_fulfilled,
                "train/rl/curriculum/duplicate_fallbacks": decision.duplicate_fallbacks,
                "train/rl/curriculum/refill_round": decision.round_index or 0,
            },
            step=decision.step,
            attributes=attributes,
        )
        for class_id, count in decision.selected_classes.items():
            self.context.metric(
                "train/rl/curriculum/class_candidate_groups",
                count,
                step=decision.step,
                attributes={**attributes, "class_id": class_id},
            )


def _weighted_reward(rewards: Sequence[float], weights: Sequence[float]) -> float:
    if len(rewards) != len(weights):
        raise RuntimeError("adaptive curriculum reward functions do not align with their weights")
    values = [
        float(reward) * weight for reward, weight in zip(rewards, weights, strict=True) if math.isfinite(float(reward))
    ]
    return sum(values) if values else float("nan")


__all__ = ["AdaptiveCurriculumRuntime"]
