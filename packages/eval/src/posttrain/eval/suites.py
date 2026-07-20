"""Validated, reusable selections of independently packaged environments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


class EvalSuiteError(ValueError):
    """Raised when an evaluation selection is ambiguous or unsafe."""


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    id: str
    category: str
    taskset: dict[str, Any]
    harness: dict[str, Any]
    num_tasks: int
    num_rollouts: int
    max_concurrent: int
    sampling: dict[str, Any]
    timeout: dict[str, Any]
    source: dict[str, Any]

    def resolved_config(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "taskset": deepcopy(self.taskset),
            "harness": deepcopy(self.harness),
            "num_tasks": self.num_tasks,
            "num_rollouts": self.num_rollouts,
            "max_concurrent": self.max_concurrent,
            "sampling": deepcopy(self.sampling),
            "timeout": deepcopy(self.timeout),
            "source": deepcopy(self.source),
        }


@dataclass(frozen=True, slots=True)
class EvalSuite:
    id: str
    evaluation_kind: Literal["general", "domain"]
    environments: tuple[EnvironmentSpec, ...]


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvalSuiteError(f"{field} must be a mapping")
    return deepcopy(value)


def _positive(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise EvalSuiteError(f"{field} must be a positive integer")
    return value


def load_suite(path: Path) -> EvalSuite:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise EvalSuiteError(f"evaluation profile does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise EvalSuiteError(f"invalid evaluation profile {path}: {error}") from error
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise EvalSuiteError("evaluation profile must be a mapping with schema_version 1")
    suite_id = data.get("id")
    if not isinstance(suite_id, str) or not suite_id:
        raise EvalSuiteError("evaluation profile requires a non-empty id")
    evaluation_kind = data.get("evaluation_kind")
    if evaluation_kind not in {"general", "domain"}:
        raise EvalSuiteError("evaluation_kind must be general or domain")

    defaults = _mapping(data.get("defaults", {}), "defaults")
    raw_environments = data.get("environments")
    if not isinstance(raw_environments, list) or not raw_environments:
        raise EvalSuiteError("environments must be a non-empty list")

    environments: list[EnvironmentSpec] = []
    seen: set[str] = set()
    for raw in raw_environments:
        item = _mapping(raw, "environment")
        environment_id = item.get("id")
        category = item.get("category")
        if not isinstance(environment_id, str) or not environment_id or environment_id in seen:
            raise EvalSuiteError(f"environment id must be non-empty and unique: {environment_id!r}")
        if not isinstance(category, str) or not category:
            raise EvalSuiteError(f"environment {environment_id!r} requires category")
        seen.add(environment_id)

        taskset = _mapping(item.get("taskset"), f"{environment_id}.taskset")
        if not isinstance(taskset.get("id"), str) or not taskset["id"]:
            raise EvalSuiteError(f"environment {environment_id!r} requires taskset.id")
        harness = _mapping(item.get("harness", defaults.get("harness", {})), f"{environment_id}.harness")
        if not isinstance(harness.get("id"), str) or not harness["id"]:
            raise EvalSuiteError(f"environment {environment_id!r} requires harness.id")
        source = _mapping(item.get("source"), f"{environment_id}.source")
        if not isinstance(source.get("revision"), str) or not source["revision"]:
            raise EvalSuiteError(f"environment {environment_id!r} requires immutable source.revision")

        environments.append(
            EnvironmentSpec(
                id=environment_id,
                category=category,
                taskset=taskset,
                harness=harness,
                num_tasks=_positive(item.get("num_tasks", defaults.get("num_tasks")), f"{environment_id}.num_tasks"),
                num_rollouts=_positive(
                    item.get("num_rollouts", defaults.get("num_rollouts")), f"{environment_id}.num_rollouts"
                ),
                max_concurrent=_positive(
                    item.get("max_concurrent", defaults.get("max_concurrent")), f"{environment_id}.max_concurrent"
                ),
                sampling=_mapping(item.get("sampling", defaults.get("sampling", {})), f"{environment_id}.sampling"),
                timeout=_mapping(item.get("timeout", defaults.get("timeout", {})), f"{environment_id}.timeout"),
                source=source,
            )
        )
    return EvalSuite(id=suite_id, evaluation_kind=evaluation_kind, environments=tuple(environments))


__all__ = ["EnvironmentSpec", "EvalSuite", "EvalSuiteError", "load_suite"]
