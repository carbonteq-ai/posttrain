"""Posttrain's adaptive curriculum as a veRL prompt selector.

The CarbonTeq veRL fork (``data.prompt_selector``) asks a selector for the
dataset rows of every dispatch: each active-sampling round, numbered from 1,
or the step's initial batch. It reports every finished group's per-trajectory
``seq_reward`` (the shaped reward, the value TRL's path observes) in dispatch
order, and saves and loads selector state with each ``global_step_*``
checkpoint.

This selector wraps the same ``AdaptiveCurriculumRuntime`` the TRL backend
uses, so decisions, observations, controller state and snapshot format are
the TRL path's. The runtime runs inside veRL's trainer process, which has no
Posttrain ``RunContext``; its events and metrics are journaled to JSONL and
replayed by the trusted parent process after the worker exits
(``replay_curriculum_journal``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from posttrain.common import JsonValue, RunContext

from ...adaptive_curriculum import CURRICULUM_SNAPSHOT_NAME
from ...adaptive_curriculum_runtime import AdaptiveCurriculumRuntime
from ...profiles import AdaptiveCurriculum

CURRICULUM_JOURNAL_NAME = "verl-curriculum-events.jsonl"


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    """Everything the selector needs, written by the Posttrain worker as JSON."""

    settings: dict[str, Any]
    num_generations: int
    state_dir: Path
    journal_path: Path
    warm_start_state_dir: Path | None = None

    def write(self, path: Path) -> None:
        payload = {
            "settings": self.settings,
            "num_generations": self.num_generations,
            "state_dir": str(self.state_dir),
            "journal_path": str(self.journal_path),
            "warm_start_state_dir": str(self.warm_start_state_dir) if self.warm_start_state_dir else None,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> SelectorConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        warm = payload.get("warm_start_state_dir")
        return cls(
            settings=dict(payload["settings"]),
            num_generations=int(payload["num_generations"]),
            state_dir=Path(payload["state_dir"]),
            journal_path=Path(payload["journal_path"]),
            warm_start_state_dir=Path(warm) if warm else None,
        )


class _JournalContext:
    """The part of ``RunContext`` the curriculum runtime uses, written to JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _write(self, record: Mapping[str, object]) -> None:
        payload = json.dumps(record, separators=(",", ":"), allow_nan=False, sort_keys=True) + "\n"
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, payload.encode("utf-8"))
        finally:
            os.close(descriptor)

    def event(self, name: str, attributes: Mapping[str, JsonValue] | None = None) -> None:
        self._write({"kind": "event", "name": name, "attributes": dict(attributes or {})})

    def metrics(
        self, values: Mapping[str, float], *, step: int, attributes: Mapping[str, JsonValue] | None = None
    ) -> None:
        self._write({"kind": "metrics", "values": dict(values), "step": step, "attributes": dict(attributes or {})})

    def metric(self, name: str, value: float, *, step: int, attributes: Mapping[str, JsonValue] | None = None) -> None:
        self._write(
            {"kind": "metric", "name": name, "value": value, "step": step, "attributes": dict(attributes or {})}
        )


def _dataset_rows(dataset: Any, class_field: str) -> list[dict[str, object]]:
    """``example_id`` and class of every training-dataset row, in dataset-index order."""

    frame = getattr(dataset, "dataframe", None)
    if frame is not None:
        columns = frame.select_columns(["example_id", class_field]).to_list()
        return [dict(row) for row in columns]
    return [{"example_id": row["example_id"], class_field: row.get(class_field)} for row in dataset]


class PosttrainCurriculumSelector:
    """veRL ``PromptSelector`` backed by Posttrain's adaptive curriculum controller."""

    def __init__(self, dataset: Any, *, config_path: str) -> None:
        config = SelectorConfig.read(Path(config_path))
        self.settings = AdaptiveCurriculum(**config.settings)
        self.num_generations = config.num_generations
        self.state_dir = config.state_dir
        self.warm_start_state_dir = config.warm_start_state_dir
        self.context = _JournalContext(config.journal_path)
        self.rows = _dataset_rows(dataset, self.settings.class_field)
        self.index_by_task: dict[str, int] = {}
        for index, row in enumerate(self.rows):
            task_id = str(row["example_id"])
            if task_id in self.index_by_task:
                raise ValueError(f"adaptive curriculum task identity is duplicated in the veRL dataset: {task_id!r}")
            self.index_by_task[task_id] = index
        self.runtime: AdaptiveCurriculumRuntime | None = None

    def _runtime(self, resume_checkpoint: Path | None = None) -> AdaptiveCurriculumRuntime:
        if self.runtime is None or resume_checkpoint is not None:
            if self.runtime is not None:
                self.runtime.close()
            self.runtime = AdaptiveCurriculumRuntime(
                cast(RunContext, self.context),
                self.rows,
                self.settings,
                num_generations=self.num_generations,
                state_dir=self.state_dir,
                resume_checkpoint=resume_checkpoint,
                warm_start_state_dir=self.warm_start_state_dir if resume_checkpoint is None else None,
            )
        return self.runtime

    def select(self, num_prompts: int, *, global_steps: int, stage: str, round_index: int | None) -> list[int]:
        rows = self._runtime().select_task_groups(
            num_prompts, step=global_steps, selection_kind=stage, round_index=round_index
        )
        task_ids = [str(row["example_id"]) for row in rows[:: self.num_generations]]
        return [self.index_by_task[task_id] for task_id in task_ids]

    def observe(self, groups: Sequence[tuple[int, Sequence[float]]], *, global_steps: int) -> None:
        inputs: list[dict[str, object]] = []
        rewards: list[list[float]] = []
        for index, values in groups:
            if len(values) != self.num_generations:
                # A group that lost trajectories carries no complete evidence; TRL never scores it either.
                continue
            inputs.extend({"example_id": self.rows[index]["example_id"]} for _ in values)
            rewards.extend([float(value)] for value in values)
        if inputs:
            self._runtime().observe_rewards(inputs, rewards, [1.0], step=global_steps)

    def save_checkpoint(self, local_dir: str) -> None:
        self._runtime().checkpoint(Path(local_dir))

    def load_checkpoint(self, local_dir: str) -> None:
        self._runtime(resume_checkpoint=Path(local_dir))

    def save_final_state(self) -> Path:
        return self._runtime().save_final_state()

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()


def replay_curriculum_journal(context: RunContext, path: Path) -> int:
    """Emit journaled curriculum events and metrics through the parent's run context."""

    if not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record.get("kind")
        attributes = record.get("attributes") or {}
        if kind == "event":
            context.event(str(record["name"]), attributes)
        elif kind == "metrics":
            context.metrics(dict(record["values"]), step=int(record["step"]), attributes=attributes)
        elif kind == "metric":
            context.metric(str(record["name"]), float(record["value"]), step=int(record["step"]), attributes=attributes)
        else:
            raise ValueError(f"unknown veRL curriculum journal record kind {kind!r}")
        count += 1
    return count


def final_snapshot_from_checkpoint(checkpoint: Path, state_dir: Path) -> Path:
    """Copy the last checkpoint's controller snapshot as the run's final curriculum state."""

    source = checkpoint / CURRICULUM_SNAPSHOT_NAME
    if not source.is_file():
        raise RuntimeError(f"veRL checkpoint {checkpoint} has no adaptive curriculum snapshot")
    state_dir.mkdir(parents=True, exist_ok=True)
    target = state_dir / CURRICULUM_SNAPSHOT_NAME
    target.write_bytes(source.read_bytes())
    return target


__all__ = [
    "CURRICULUM_JOURNAL_NAME",
    "PosttrainCurriculumSelector",
    "SelectorConfig",
    "final_snapshot_from_checkpoint",
    "replay_curriculum_journal",
]
