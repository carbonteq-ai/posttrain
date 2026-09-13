"""Deterministic task filtering, allocation, and evaluation manifests."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from posttrain.environment import TaskDescriptor, task_inventory_digest

_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_MISSING = "__missing__"

type SelectionKind = Literal[
    "full",
    "uniform",
    "proportional",
    "balanced",
    "custom",
    "minimum_then_proportional",
]


@dataclass(frozen=True, slots=True)
class EvaluationFilterClause:
    """One predicate over a task facet or the reserved ``split`` field."""

    dimension: str
    operator: Literal["eq", "in"]
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.dimension != "split" and not _ID.fullmatch(self.dimension):
            raise ValueError("evaluation filter dimension must be split or a lowercase stable identifier")
        if not self.values or any(not value.strip() for value in self.values):
            raise ValueError("evaluation filter values must be non-empty")
        if self.operator == "eq" and len(self.values) != 1:
            raise ValueError("eq filter requires exactly one value")
        object.__setattr__(self, "values", tuple(sorted(set(self.values))))

    def matches(self, task: TaskDescriptor) -> bool:
        actual = (task.split,) if self.dimension == "split" and task.split is not None else task.values(self.dimension)
        expected = set(self.values)
        if self.operator == "eq":
            return self.values[0] in actual
        return bool(expected.intersection(actual))


@dataclass(frozen=True, slots=True)
class EvaluationTaskFilter:
    """Conjunction plus optional disjunction of task predicates."""

    all_of: tuple[EvaluationFilterClause, ...] = ()
    any_of: tuple[EvaluationFilterClause, ...] = ()

    def matches(self, task: TaskDescriptor) -> bool:
        return all(item.matches(task) for item in self.all_of) and (
            not self.any_of or any(item.matches(task) for item in self.any_of)
        )


@dataclass(frozen=True, slots=True)
class EvaluationSelectionPolicy:
    """A reusable population filter and distinct-task allocation policy."""

    kind: SelectionKind = "full"
    num_tasks: int | None = None
    dimensions: tuple[str, ...] = ()
    task_filter: EvaluationTaskFilter = EvaluationTaskFilter()
    weights: Mapping[str, float] = field(default_factory=dict)
    minimum_per_stratum: int = 0
    seed: int = 0
    missing: Literal["error", "bucket", "exclude"] = "error"
    exhaustion: Literal["error", "use_all"] = "error"

    def __post_init__(self) -> None:
        if self.num_tasks is not None and self.num_tasks < 1:
            raise ValueError("evaluation selection num_tasks must be positive")
        if self.kind == "full" and self.num_tasks is not None:
            raise ValueError("full evaluation selection does not accept num_tasks")
        if self.kind != "full" and self.num_tasks is None:
            raise ValueError(f"{self.kind} evaluation selection requires num_tasks")
        if self.kind in {"proportional", "balanced", "custom", "minimum_then_proportional"} and not self.dimensions:
            raise ValueError(f"{self.kind} evaluation selection requires allocation dimensions")
        if len(self.dimensions) != len(set(self.dimensions)) or any(not _ID.fullmatch(value) for value in self.dimensions):
            raise ValueError("allocation dimensions must be unique lowercase stable identifiers")
        if self.minimum_per_stratum < 0:
            raise ValueError("minimum_per_stratum must be non-negative")
        if self.kind != "minimum_then_proportional" and self.minimum_per_stratum:
            raise ValueError("minimum_per_stratum is only valid for minimum_then_proportional")
        weights = dict(self.weights)
        if self.kind == "custom":
            if not weights or any(not math.isfinite(value) or value < 0 for value in weights.values()):
                raise ValueError("custom allocation weights must be finite, non-negative, and non-empty")
            if sum(weights.values()) <= 0:
                raise ValueError("custom allocation weights must contain positive mass")
        elif weights:
            raise ValueError("allocation weights are only valid for custom selection")
        object.__setattr__(self, "weights", MappingProxyType(weights))


@dataclass(frozen=True, slots=True)
class ResolvedTaskSelection:
    task: TaskDescriptor
    stratum: str
    inclusion_probability: float
    target_weight: float


@dataclass(frozen=True, slots=True)
class ResolvedAllocationStratum:
    key: str
    eligible: int
    selected: int
    target_weight: float


@dataclass(frozen=True, slots=True)
class ResolvedEvaluationManifest:
    """Immutable output of resolving one policy against one task inventory."""

    schema_version: int
    inventory_digest: str
    eligible_count: int
    excluded_count: int
    requested_count: int
    shortfall: int
    policy: EvaluationSelectionPolicy
    tasks: tuple[ResolvedTaskSelection, ...]
    strata: tuple[ResolvedAllocationStratum, ...]
    digest: str

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "inventory_digest": self.inventory_digest,
            "eligible_count": self.eligible_count,
            "excluded_count": self.excluded_count,
            "requested_count": self.requested_count,
            "shortfall": self.shortfall,
            "policy": _policy_payload(self.policy),
            "tasks": [
                {
                    "task": item.task.to_payload(),
                    "stratum": item.stratum,
                    "inclusion_probability": item.inclusion_probability,
                    "target_weight": item.target_weight,
                }
                for item in self.tasks
            ],
            "strata": [
                {
                    "key": item.key,
                    "eligible": item.eligible,
                    "selected": item.selected,
                    "target_weight": item.target_weight,
                }
                for item in self.strata
            ],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def _policy_payload(policy: EvaluationSelectionPolicy) -> dict[str, object]:
    return {
        "kind": policy.kind,
        "num_tasks": policy.num_tasks,
        "dimensions": list(policy.dimensions),
        "filter": {
            "all_of": [vars_without_slots(item) for item in policy.task_filter.all_of],
            "any_of": [vars_without_slots(item) for item in policy.task_filter.any_of],
        },
        "weights": dict(sorted(policy.weights.items())),
        "minimum_per_stratum": policy.minimum_per_stratum,
        "seed": policy.seed,
        "missing": policy.missing,
        "exhaustion": policy.exhaustion,
    }


def vars_without_slots(clause: EvaluationFilterClause) -> dict[str, object]:
    return {"dimension": clause.dimension, "operator": clause.operator, "values": list(clause.values)}


def _stratum(task: TaskDescriptor, policy: EvaluationSelectionPolicy) -> str | None:
    if not policy.dimensions:
        return "all"
    parts: list[str] = []
    for dimension in policy.dimensions:
        values = task.values(dimension)
        if not values:
            if policy.missing == "error":
                raise ValueError(f"task {task.key!r} is missing allocation facet {dimension!r}")
            if policy.missing == "exclude":
                return None
            values = (_MISSING,)
        encoded = "+".join(sorted(values))
        parts.append(f"{dimension}={encoded}")
    return "|".join(parts)


def _largest_remainder(total: int, weights: Mapping[str, float], capacities: Mapping[str, int]) -> dict[str, int]:
    allocation = {key: 0 for key in weights}
    remaining = total
    available = {key for key, capacity in capacities.items() if capacity > 0 and weights.get(key, 0) > 0}
    while remaining and available:
        mass = sum(weights[key] for key in available)
        if mass <= 0:
            break
        ideals = {key: remaining * weights[key] / mass for key in available}
        floors = {key: min(capacities[key] - allocation[key], math.floor(ideals[key])) for key in available}
        assigned = sum(floors.values())
        for key, value in floors.items():
            allocation[key] += value
        remaining -= assigned
        available = {key for key in available if allocation[key] < capacities[key]}
        if not remaining or not available:
            break
        ranked = sorted(available, key=lambda key: (-(ideals[key] - math.floor(ideals[key])), key))
        made_progress = False
        for key in ranked:
            if remaining == 0:
                break
            if allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
                made_progress = True
        if not made_progress:
            break
    return allocation


def _quotas(
    groups: Mapping[str, Sequence[TaskDescriptor]],
    policy: EvaluationSelectionPolicy,
    count: int,
) -> tuple[dict[str, int], dict[str, float]]:
    capacities = {key: len(values) for key, values in groups.items()}
    if policy.kind in {"uniform", "proportional", "minimum_then_proportional"}:
        masses = {key: float(value) for key, value in capacities.items()}
    elif policy.kind == "balanced":
        masses = {key: 1.0 for key in groups}
    elif policy.kind == "custom":
        unknown = set(policy.weights) - set(groups)
        missing = set(groups) - set(policy.weights)
        if unknown or missing:
            details = []
            if unknown:
                details.append("unknown strata: " + ", ".join(sorted(unknown)))
            if missing:
                details.append("missing strata: " + ", ".join(sorted(missing)))
            raise ValueError("custom allocation weights do not match eligible strata (" + "; ".join(details) + ")")
        masses = dict(policy.weights)
    else:
        raise AssertionError(f"quota allocation is not defined for {policy.kind}")

    total_mass = sum(masses.values())
    target_weights = {key: value / total_mass for key, value in masses.items()}
    if policy.kind != "minimum_then_proportional":
        return _largest_remainder(count, masses, capacities), target_weights

    minimum = policy.minimum_per_stratum
    required = minimum * len(groups)
    if required > count:
        raise ValueError(f"minimum allocation requires {required} tasks but budget is {count}")
    too_small = sorted(key for key, capacity in capacities.items() if capacity < minimum)
    if too_small:
        raise ValueError("minimum allocation exceeds stratum capacity: " + ", ".join(too_small))
    base = {key: minimum for key in groups}
    residual_capacity = {key: capacities[key] - minimum for key in groups}
    remainder = _largest_remainder(count - required, masses, residual_capacity)
    return {key: base[key] + remainder[key] for key in groups}, target_weights


def resolve_evaluation_selection(
    inventory: Sequence[TaskDescriptor],
    policy: EvaluationSelectionPolicy,
) -> ResolvedEvaluationManifest:
    """Resolve a stable, without-replacement selection independent of input order."""

    complete_inventory = tuple(inventory)
    inventory_digest = task_inventory_digest(complete_inventory)
    filtered = tuple(sorted((task for task in complete_inventory if policy.task_filter.matches(task)), key=lambda item: item.key))
    grouped: defaultdict[str, list[TaskDescriptor]] = defaultdict(list)
    for task in filtered:
        key = _stratum(task, policy)
        if key is None:
            continue
        grouped[key].append(task)
    if not grouped:
        raise ValueError("evaluation selection has no eligible tasks")
    eligible_count = sum(len(values) for values in grouped.values())
    requested_count = eligible_count if policy.kind == "full" else policy.num_tasks
    assert requested_count is not None
    if requested_count > eligible_count:
        if policy.exhaustion == "error":
            raise ValueError(f"evaluation selection requests {requested_count} tasks from {eligible_count} eligible tasks")
        selected_count = eligible_count
    else:
        selected_count = requested_count

    if policy.kind == "full":
        quotas = {key: len(values) for key, values in grouped.items()}
        target_weights = {key: len(values) / eligible_count for key, values in grouped.items()}
    else:
        quotas, target_weights = _quotas(grouped, policy, selected_count)
        allocated = sum(quotas.values())
        if allocated != selected_count:
            raise ValueError(
                f"evaluation allocation can place only {allocated} of {selected_count} requested tasks under its weights"
            )

    rng = random.Random(policy.seed)
    resolved: list[ResolvedTaskSelection] = []
    strata: list[ResolvedAllocationStratum] = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda item: item.key)
        quota = quotas[key]
        chosen = candidates if quota == len(candidates) else rng.sample(candidates, quota)
        chosen = sorted(chosen, key=lambda item: item.key)
        inclusion = quota / len(candidates)
        per_task_weight = target_weights[key] / quota if quota else 0.0
        resolved.extend(ResolvedTaskSelection(task, key, inclusion, per_task_weight) for task in chosen)
        strata.append(ResolvedAllocationStratum(key, len(candidates), quota, target_weights[key]))
    resolved.sort(key=lambda item: item.task.key)
    excluded_count = len(complete_inventory) - eligible_count
    unsigned = {
        "schema_version": 1,
        "inventory_digest": inventory_digest,
        "eligible_count": eligible_count,
        "excluded_count": excluded_count,
        "requested_count": requested_count,
        "shortfall": requested_count - selected_count,
        "policy": _policy_payload(policy),
        "tasks": [
            {
                "task": item.task.to_payload(),
                "stratum": item.stratum,
                "inclusion_probability": item.inclusion_probability,
                "target_weight": item.target_weight,
            }
            for item in resolved
        ],
        "strata": [
            {
                "key": item.key,
                "eligible": item.eligible,
                "selected": item.selected,
                "target_weight": item.target_weight,
            }
            for item in strata
        ],
    }
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return ResolvedEvaluationManifest(
        schema_version=1,
        inventory_digest=inventory_digest,
        eligible_count=eligible_count,
        excluded_count=excluded_count,
        requested_count=requested_count,
        shortfall=requested_count - selected_count,
        policy=policy,
        tasks=tuple(resolved),
        strata=tuple(strata),
        digest=digest,
    )


__all__ = [
    "EvaluationFilterClause",
    "EvaluationSelectionPolicy",
    "EvaluationTaskFilter",
    "ResolvedAllocationStratum",
    "ResolvedEvaluationManifest",
    "ResolvedTaskSelection",
    "resolve_evaluation_selection",
]
