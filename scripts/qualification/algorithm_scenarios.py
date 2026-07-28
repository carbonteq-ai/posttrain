"""Versioned, provider-neutral descriptions of substantial qualification runs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@+-]*$")
_ONLINE_JOBS = {"train.grpo", "train.sampo", "train.distill"}
_DATASET_JOBS = {"train.sft", "train.dpo"}
_TRAINING_JOBS = _ONLINE_JOBS | _DATASET_JOBS
_EVALUATION_JOBS = {"eval.general", "eval.domain"}
_SUPPORTED_JOBS = _TRAINING_JOBS | _EVALUATION_JOBS


def _identifier(value: str, field_name: str) -> str:
    resolved = value.strip()
    if not _ID.fullmatch(resolved):
        raise ValueError(f"{field_name} must be a stable selection identifier")
    return resolved


@dataclass(frozen=True, slots=True)
class QualificationAcceptance:
    minimum_optimizer_updates: int
    minimum_complete_traces: int
    require_reward_variance: bool
    require_nonzero_gradient: bool
    require_model_artifact: bool
    require_remote_observatory: bool = True

    def __post_init__(self) -> None:
        if self.minimum_optimizer_updates < 0:
            raise ValueError("minimum optimizer updates cannot be negative")
        if self.minimum_complete_traces < 0:
            raise ValueError("minimum complete traces cannot be negative")


@dataclass(frozen=True, slots=True)
class QualificationScenario:
    id: str
    revision: str
    job_kind: str
    model_ref: str
    environment_ref: str | None
    dataset_ref: str | None
    training_ref: str
    inference_ref: str | None
    update_budget: int | None
    task_budget: int | None
    rollouts_per_task: int | None
    maximum_duration_seconds: int
    target_capabilities: Mapping[str, object] = field(default_factory=dict)
    acceptance: QualificationAcceptance = field(
        default_factory=lambda: QualificationAcceptance(0, 0, False, False, False)
    )

    def __post_init__(self) -> None:
        for name in ("id", "revision", "model_ref", "training_ref"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.environment_ref is not None:
            object.__setattr__(
                self,
                "environment_ref",
                _identifier(self.environment_ref, "environment_ref"),
            )
        if self.dataset_ref is not None:
            object.__setattr__(
                self,
                "dataset_ref",
                _identifier(self.dataset_ref, "dataset_ref"),
            )
        if self.inference_ref is not None:
            object.__setattr__(
                self,
                "inference_ref",
                _identifier(self.inference_ref, "inference_ref"),
            )
        if self.job_kind not in _SUPPORTED_JOBS:
            raise ValueError(f"unsupported qualification job kind: {self.job_kind}")
        if self.maximum_duration_seconds < 1:
            raise ValueError("maximum duration must be positive")
        for value, name in (
            (self.update_budget, "update budget"),
            (self.task_budget, "task budget"),
            (self.rollouts_per_task, "rollouts per task"),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when selected")
        if self.job_kind in _ONLINE_JOBS:
            if self.environment_ref is None or self.dataset_ref is not None:
                raise ValueError(f"{self.job_kind} requires one environment and no dataset")
            if self.inference_ref is None:
                raise ValueError(f"{self.job_kind} requires rollout inference")
            if self.task_budget is None or self.rollouts_per_task is None:
                raise ValueError(f"{self.job_kind} requires task and rollout budgets")
        if self.job_kind in _DATASET_JOBS:
            if self.dataset_ref is None or self.environment_ref is not None:
                raise ValueError(f"{self.job_kind} requires one dataset and no environment")
        if self.job_kind in _TRAINING_JOBS:
            if self.update_budget is None:
                raise ValueError(f"{self.job_kind} requires an update budget")
            if self.update_budget < self.acceptance.minimum_optimizer_updates:
                raise ValueError("update budget cannot be below the acceptance minimum")
            if self.acceptance.minimum_optimizer_updates < 10:
                raise ValueError("algorithm qualification requires at least ten optimizer updates")
        elif self.update_budget is not None:
            raise ValueError("evaluation scenarios cannot select optimizer updates")
        if self.job_kind in _EVALUATION_JOBS and self.task_budget is None:
            raise ValueError("evaluation qualification requires a task budget")
        object.__setattr__(
            self,
            "target_capabilities",
            MappingProxyType(dict(self.target_capabilities)),
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": "posttrain.qualification-scenario.v1",
            "id": self.id,
            "revision": self.revision,
            "job_kind": self.job_kind,
            "model_ref": self.model_ref,
            "environment_ref": self.environment_ref,
            "dataset_ref": self.dataset_ref,
            "training_ref": self.training_ref,
            "inference_ref": self.inference_ref,
            "update_budget": self.update_budget,
            "task_budget": self.task_budget,
            "rollouts_per_task": self.rollouts_per_task,
            "maximum_duration_seconds": self.maximum_duration_seconds,
            "target_capabilities": dict(self.target_capabilities),
            "acceptance": asdict(self.acceptance),
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> QualificationScenario:
        values = dict(payload)
        if values.pop("schema", None) != "posttrain.qualification-scenario.v1":
            raise ValueError("qualification scenario schema is unsupported")
        acceptance = values.get("acceptance")
        if not isinstance(acceptance, Mapping):
            raise ValueError("qualification scenario acceptance is required")
        values["acceptance"] = QualificationAcceptance(**dict(acceptance))
        target = values.get("target_capabilities")
        if not isinstance(target, Mapping):
            raise ValueError("qualification target capabilities must be an object")
        return cls(**values)


_GRPO_ACCEPTANCE = QualificationAcceptance(
    minimum_optimizer_updates=10,
    minimum_complete_traces=40,
    require_reward_variance=True,
    require_nonzero_gradient=True,
    require_model_artifact=True,
)
_GRPO_15_ACCEPTANCE = QualificationAcceptance(
    minimum_optimizer_updates=15,
    minimum_complete_traces=120,
    require_reward_variance=True,
    require_nonzero_gradient=True,
    require_model_artifact=True,
)

SCENARIOS: Mapping[str, QualificationScenario] = MappingProxyType(
    {
        scenario.id: scenario
        for scenario in (
            QualificationScenario(
                id="automationbench-qwen35-08b-grpo-10",
                revision="1",
                job_kind="train.grpo",
                model_ref="models/qwen3.5-0.8b@bf16",
                environment_ref="automationbench-zapier-simple-grpo",
                dataset_ref=None,
                training_ref="training/qwen3.5-0.8b-verl-lora-qualification@1",
                inference_ref="inference/qwen3.5-0.8b-vllm-verl-rollout@1",
                update_budget=10,
                task_budget=2,
                rollouts_per_task=4,
                maximum_duration_seconds=7_200,
                target_capabilities={
                    "accelerator_vendor": "nvidia",
                    "accelerator_count": 1,
                    "minimum_vram_gib": 20,
                },
                acceptance=_GRPO_ACCEPTANCE,
            ),
            QualificationScenario(
                id="gsm8k-qwen35-08b-grpo-15",
                revision="2",
                job_kind="train.grpo",
                model_ref="models/qwen3.5-0.8b@bf16",
                environment_ref="gsm8k-train-candidates",
                dataset_ref=None,
                training_ref="training/qwen3.5-0.8b-verl-lora-qualification@1",
                inference_ref="inference/qwen3.5-0.8b-vllm-verl-rollout@1",
                update_budget=15,
                task_budget=2,
                rollouts_per_task=4,
                maximum_duration_seconds=7_200,
                target_capabilities={
                    "accelerator_vendor": "nvidia",
                    "accelerator_count": 1,
                    "minimum_vram_gib": 20,
                },
                acceptance=_GRPO_15_ACCEPTANCE,
            ),
        )
    }
)


def scenario_by_id(scenario_id: str) -> QualificationScenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as error:
        choices = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"unknown qualification scenario {scenario_id!r}; choose {choices}") from error


__all__ = [
    "QualificationAcceptance",
    "QualificationScenario",
    "SCENARIOS",
    "scenario_by_id",
]
