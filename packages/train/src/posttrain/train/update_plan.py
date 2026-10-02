"""Deterministic optimizer boundaries and capacity-checked execution packs.

Objective implementations supply already-resolved atomic contributions. This
module schedules them; it does not reinterpret credit or ratio mathematics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from .update_records import (
    ActionRef,
    ActionSelection,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationSnapshot,
    record_digest,
)


@dataclass(frozen=True, slots=True)
class PolicyUpdateSchedule:
    unit: Literal["episode", "turn", "selected-token"]
    budget: int
    epochs: int = 1
    order: Literal["native", "shuffle"] = "native"
    seed: int = 42
    final_policy: Literal["include", "drop", "error"] = "include"
    max_applied_updates: int | None = None

    def __post_init__(self) -> None:
        if self.unit not in {"episode", "turn", "selected-token"}:
            raise InvalidPolicyUpdate("unknown optimizer schedule unit")
        if any(type(value) is not int or value < 1 for value in (self.budget, self.epochs)):
            raise InvalidPolicyUpdate("schedule budget and epochs must be positive integers")
        if self.order not in {"native", "shuffle"} or self.final_policy not in {"include", "drop", "error"}:
            raise InvalidPolicyUpdate("unknown schedule order or final policy")
        if type(self.seed) is not int:
            raise InvalidPolicyUpdate("schedule seed must be an integer")
        if self.max_applied_updates is not None and (
            type(self.max_applied_updates) is not int or self.max_applied_updates < 1
        ):
            raise InvalidPolicyUpdate("applied update limit must be a positive integer")


@dataclass(frozen=True, slots=True)
class PolicyExecutionBudget:
    records: int
    context_tokens: int
    statistic_bytes: int
    oversized_policy: Literal["error"] = "error"

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (self.records, self.context_tokens, self.statistic_bytes)):
            raise InvalidPolicyUpdate("execution capacity must be positive integer hard limits")
        if self.oversized_policy != "error":
            raise InvalidPolicyUpdate("unqualified oversized execution strategy")


@dataclass(frozen=True, slots=True)
class PolicyUpdateSettings:
    """Opt-in catalog selection, independent of legacy accumulation fields."""

    schedule: PolicyUpdateSchedule
    execution: PolicyExecutionBudget
    objective_variant: Literal["algorithm", "semantic-spans", "turn-rows"] = "algorithm"
    policy_selection: ActionSelection = ActionSelection()
    kl_selection: ActionSelection = ActionSelection()
    denominator: Literal["selected", "original-eligible"] = "selected"
    empty_policy: Literal["reject", "omit", "zero"] = "reject"
    revision: Literal["1"] = "1"
    # Explicit external estimator identity (e.g. "group-centered-likelihood@1")
    # whose detached credit replaces the algorithm's own; the host composition
    # must inject a matching process-credit provider.
    credit_estimator: str | None = None

    def __post_init__(self) -> None:
        if self.credit_estimator is not None and (
                not isinstance(self.credit_estimator, str) or not self.credit_estimator.strip()):
            raise InvalidPolicyUpdate("credit_estimator must name an explicit estimator identity")
        if self.revision != "1" or self.objective_variant not in {"algorithm", "semantic-spans", "turn-rows"}:
            raise InvalidPolicyUpdate("unsupported policy update selection revision or variant")
        if self.denominator not in {"selected", "original-eligible"} or self.empty_policy not in {"reject", "omit", "zero"}:
            raise InvalidPolicyUpdate("unsupported policy update reduction")
        if self.objective_variant != "semantic-spans" and (
            self.policy_selection.mode != "all" or self.kl_selection.mode != "all"
            or self.denominator != "selected" or self.empty_policy != "reject"
        ):
            raise InvalidPolicyUpdate("algorithm variant retains original support; select semantic-spans explicitly")

    def validate_legacy_loop(self, *, max_steps: int, per_device_batch_size: int, gradient_accumulation_steps: int) -> None:
        if per_device_batch_size != 1 or gradient_accumulation_steps != 1:
            raise InvalidPolicyUpdate(
                "explicit policy_updates.execution replaces loop batch/accumulation fields; leave both at 1"
            )
        if self.schedule.max_applied_updates is not None and self.schedule.max_applied_updates != max_steps:
            raise InvalidPolicyUpdate("policy update max_applied_updates must agree with loop.max_steps")


@dataclass(frozen=True, slots=True)
class ContributionRef:
    id: str
    actions: tuple[ActionRef, ...]
    dependencies: tuple[ActionRef, ...]
    reduction_domain: tuple[ActionRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not (self.actions or self.reduction_domain):
            raise InvalidPolicyUpdate("atomic contribution requires identity and explicit reduction domain")
        if len(set(self.actions)) != len(self.actions) or len(set(self.dependencies)) != len(self.dependencies):
            raise InvalidPolicyUpdate("contribution action support cannot duplicate original actions")
        if not set(self.actions) <= set(self.dependencies):
            raise InvalidPolicyUpdate("dependency closure must include selected actions")
        if self.reduction_domain and not set(self.actions) <= set(self.reduction_domain):
            raise InvalidPolicyUpdate("selected actions exceed contribution reduction domain")
        if len(set(self.reduction_domain)) != len(self.reduction_domain):
            raise InvalidPolicyUpdate("reduction domain cannot duplicate original actions")


@dataclass(frozen=True, slots=True)
class ObjectivePopulation:
    """Internal output of a qualified objective resolver, not a public loss DSL."""

    definition_id: str
    credit_digest: str
    contributions: tuple[ContributionRef, ...]
    required_statistics: tuple[str, ...]
    # Retained per-action output statistics; adapters qualify and resolve the size.
    statistic_bytes_per_action: int
    contract_digest: str = "legacy-internal@1"

    def __post_init__(self) -> None:
        if not self.definition_id.strip() or not self.credit_digest.strip() or not self.contributions:
            raise InvalidPolicyUpdate("objective population requires definition, credit and contributions")
        if len({item.id for item in self.contributions}) != len(self.contributions):
            raise InvalidPolicyUpdate("objective contribution identities must be unique")
        if type(self.statistic_bytes_per_action) is not int or self.statistic_bytes_per_action < 1:
            raise InvalidPolicyUpdate("objective requires its retained statistic size")


@dataclass(frozen=True, slots=True)
class ResolvedUpdate:
    population: PopulationSnapshot
    objective: ObjectivePopulation
    schedule_digest: str
    epoch: int
    minibatch: int
    contributions: tuple[ContributionRef, ...]
    occurrence_ids: tuple[str, ...]
    dependencies: tuple[ActionRef, ...]
    discarded_contributions: tuple[str, ...]

    @property
    def digest(self) -> str:
        return record_digest(self)


@dataclass(frozen=True, slots=True)
class ExecutionCapabilities:
    definition_ids: tuple[str, ...]
    statistics: tuple[str, ...]
    max_context_tokens: int
    # A coupled backward graph must remain intact across packs unless qualified replay exists.
    cross_pack_dependencies: bool = False
    context_layout: Literal["ragged", "dense-pack", "dense-population"] = "ragged"

    def __post_init__(self) -> None:
        if self.context_layout not in {"ragged", "dense-pack", "dense-population"}:
            raise InvalidPolicyUpdate("unqualified physical context layout")


def population_context_width(population: PopulationSnapshot) -> int:
    """Maximum context actually addressed by sampled actions, including unselected turns."""
    used = {record.conditioning_id for record in population.actions}
    return max(view.context_tokens for view in population.conditioning if view.id in used)


def execution_context_tokens(
    population: PopulationSnapshot, context_ids: tuple[str, ...], capabilities: ExecutionCapabilities,
) -> int:
    """Count physical input slots; padding never becomes sampled or objective support."""
    if not context_ids:
        return 0
    contexts = {view.id: view for view in population.conditioning}
    if len(set(context_ids)) != len(context_ids) or any(key not in contexts for key in context_ids):
        raise InvalidPolicyUpdate("physical context cost requires unique retained conditioning identities")
    lengths = [contexts[key].context_tokens for key in context_ids]
    if max(lengths) > capabilities.max_context_tokens:
        raise InvalidPolicyUpdate("actual conditioning view exceeds qualified backend context capacity")
    if capabilities.context_layout == "ragged":
        return sum(lengths)
    width = max(lengths) if capabilities.context_layout == "dense-pack" else population_context_width(population)
    if width > capabilities.max_context_tokens:
        raise InvalidPolicyUpdate("physical padding width exceeds qualified backend context capacity")
    return len(context_ids) * width


@dataclass(frozen=True, slots=True)
class ExecutionPack:
    update_digest: str
    index: int
    actions: tuple[ActionRef, ...]
    context_ids: tuple[str, ...]
    context_tokens: int


@dataclass(frozen=True, slots=True)
class UpdateCursor:
    population_digest: str
    schedule_digest: str
    next_occurrence: int
    applied_updates: int
    attempts: int
    versions: PolicyVersions


def resolve_updates(
    snapshot: PopulationSnapshot, schedule: PolicyUpdateSchedule, objective: ObjectivePopulation,
) -> tuple[ResolvedUpdate, ...]:
    eligible = {record.action for record in snapshot.actions}
    if any(not (set(item.dependencies) | set(item.reduction_domain)) <= eligible for item in objective.contributions):
        raise InvalidPolicyUpdate("objective dependencies exceed admitted original actions")
    groups: dict[tuple[str, ...], list[ContributionRef]] = {}
    for contribution in objective.contributions:
        if schedule.unit == "selected-token":
            key = (contribution.id,)
        else:
            keys = {
                (action.episode_id,) if schedule.unit == "episode"
                else (action.episode_id, action.branch_id, action.turn_id)
                for action in (contribution.actions or contribution.reduction_domain)
            }
            if len(keys) != 1:
                raise InvalidPolicyUpdate(f"atomic contribution crosses {schedule.unit} scheduling units")
            key = next(iter(keys))
        groups.setdefault(key, []).append(contribution)
    atoms = tuple(tuple(items) for items in groups.values())
    result: list[ResolvedUpdate] = []
    schedule_digest = record_digest(schedule)
    for epoch in range(schedule.epochs):
        ordered = list(atoms)
        if schedule.order == "shuffle":
            random.Random(schedule.seed + epoch).shuffle(ordered)
        batches: list[tuple[ContributionRef, ...]] = []
        pending: list[ContributionRef] = []
        cost = 0
        for atom in ordered:
            atom_cost = len({action for item in atom for action in item.actions}) if schedule.unit == "selected-token" else 1
            if pending and cost + atom_cost > schedule.budget:
                batches.append(tuple(pending))
                pending, cost = [], 0
            pending.extend(atom)
            cost += atom_cost
            if cost >= schedule.budget:
                batches.append(tuple(pending))
                pending, cost = [], 0
        discarded: tuple[str, ...] = ()
        if pending:
            if schedule.final_policy == "error":
                raise InvalidPolicyUpdate("incomplete final optimizer minibatch")
            if schedule.final_policy == "drop":
                discarded = tuple(item.id for item in pending)
            else:
                batches.append(tuple(pending))
        if not batches:
            raise InvalidPolicyUpdate("schedule drops every contribution; no optimizer update remains")
        for minibatch, batch in enumerate(batches):
            result.append(ResolvedUpdate(
                snapshot, objective, schedule_digest, epoch, minibatch, batch,
                tuple(f"{snapshot.id}/{epoch}/{minibatch}/{item.id}" for item in batch),
                tuple(sorted({action for item in batch for action in item.dependencies})), discarded,
            ))
    return tuple(result)


def validate_update(update: ResolvedUpdate, capabilities: ExecutionCapabilities) -> None:
    if update.objective.definition_id not in capabilities.definition_ids:
        raise InvalidPolicyUpdate(f"backend has no qualified objective {update.objective.definition_id!r}")
    missing = set(update.objective.required_statistics) - set(capabilities.statistics)
    if missing:
        raise InvalidPolicyUpdate(f"backend lacks required statistics: {sorted(missing)}")
    records = {record.action: record for record in update.population.actions}
    contexts = {view.id: view for view in update.population.conditioning}
    if any(contexts[records[action].conditioning_id].context_tokens > capabilities.max_context_tokens
           for action in update.dependencies):
        raise InvalidPolicyUpdate("actual conditioning view exceeds qualified backend context capacity")


def plan_packs(
    update: ResolvedUpdate, execution_budget: PolicyExecutionBudget, capabilities: ExecutionCapabilities,
) -> tuple[ExecutionPack, ...]:
    validate_update(update, capabilities)
    if len(update.dependencies) * update.objective.statistic_bytes_per_action > execution_budget.statistic_bytes:
        raise InvalidPolicyUpdate("retained objective statistics exceed execution capacity")
    records = {record.action: record for record in update.population.actions}
    turns: dict[tuple[str, str, str], list[ActionRef]] = {}
    for action in update.dependencies:
        turns.setdefault((action.episode_id, action.branch_id, action.turn_id), []).append(action)
    packs: list[ExecutionPack] = []
    pending: list[ActionRef] = []
    count = 0

    def context_size(actions: list[ActionRef]) -> int:
        return execution_context_tokens(update.population,
            tuple(sorted({records[action].conditioning_id for action in actions})), capabilities)

    def emit() -> None:
        packs.append(ExecutionPack(update.digest, len(packs), tuple(pending),
                                   tuple(sorted({records[action].conditioning_id for action in pending})),
                                   context_size(pending)))

    for actions in turns.values():
        if context_size(actions) > execution_budget.context_tokens:
            raise InvalidPolicyUpdate("atomic turn conditioning exceeds hard pack capacity")
        if pending and (count == execution_budget.records or context_size(pending + actions) > execution_budget.context_tokens):
            emit()
            pending, count = [], 0
        pending.extend(actions)
        count += 1
    if pending:
        emit()
    if len(packs) > 1 and not capabilities.cross_pack_dependencies:
        pack_index = {action: pack.index for pack in packs for action in pack.actions}
        if any(len({pack_index[action] for action in item.dependencies}) > 1 for item in update.contributions):
            raise InvalidPolicyUpdate("objective dependency crosses packs without qualified graph retention or replay")
    return tuple(packs)
