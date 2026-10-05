"""Deterministic optimizer boundaries and capacity-checked execution packs.

Objective implementations supply already-resolved atomic contributions. This
module schedules them; it does not reinterpret credit or ratio mathematics.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .update_records import (
    ActionSelection,
    ConditioningView,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationSnapshot,
    array_digest,
    frozen_array,
    payload_digest,
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
        if any(
            type(value) is not int or value < 1 for value in (self.records, self.context_tokens, self.statistic_bytes)
        ):
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
            not isinstance(self.credit_estimator, str) or not self.credit_estimator.strip()
        ):
            raise InvalidPolicyUpdate("credit_estimator must name an explicit estimator identity")
        if self.revision != "1" or self.objective_variant not in {"algorithm", "semantic-spans", "turn-rows"}:
            raise InvalidPolicyUpdate("unsupported policy update selection revision or variant")
        if self.denominator not in {"selected", "original-eligible"} or self.empty_policy not in {
            "reject",
            "omit",
            "zero",
        }:
            raise InvalidPolicyUpdate("unsupported policy update reduction")
        if self.objective_variant != "semantic-spans" and (
            self.policy_selection.mode != "all"
            or self.kl_selection.mode != "all"
            or self.denominator != "selected"
            or self.empty_policy != "reject"
        ):
            raise InvalidPolicyUpdate("algorithm variant retains original support; select semantic-spans explicitly")

    def validate_legacy_loop(
        self, *, max_steps: int, per_device_batch_size: int, gradient_accumulation_steps: int
    ) -> None:
        if per_device_batch_size != 1 or gradient_accumulation_steps != 1:
            raise InvalidPolicyUpdate(
                "explicit policy_updates.execution replaces loop batch/accumulation fields; leave both at 1"
            )
        if self.schedule.max_applied_updates is not None and self.schedule.max_applied_updates != max_steps:
            raise InvalidPolicyUpdate("policy update max_applied_updates must agree with loop.max_steps")


@dataclass(frozen=True, slots=True)
class ContributionRef:
    """One atomic objective contribution: one sampled turn.

    Its selected actions are the objective's selected positions inside ``view``;
    its reduction domain is every position of ``view``. ``dependency_views``
    are the turns whose scores its ratio needs (empty when it selects nothing
    and needs no score).
    """

    id: str
    view: int
    dependency_views: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.id.strip() or type(self.view) is not int or self.view < 0:
            raise InvalidPolicyUpdate("atomic contribution requires identity and explicit reduction domain")
        if len(set(self.dependency_views)) != len(self.dependency_views) or any(
            type(view) is not int or view < 0 for view in self.dependency_views
        ):
            raise InvalidPolicyUpdate("contribution dependencies cannot duplicate original turns")
        if self.dependency_views and self.view not in self.dependency_views:
            raise InvalidPolicyUpdate("dependency closure must include selected actions")


@dataclass(frozen=True, slots=True, eq=False)
class ObjectivePopulation:
    """Internal output of a qualified objective resolver, not a public loss DSL.

    ``policy`` and ``kl`` are the frozen selections as masks over positions.
    """

    definition_id: str
    credit_digest: str
    contributions: tuple[ContributionRef, ...]
    policy: np.ndarray
    kl: np.ndarray
    required_statistics: tuple[str, ...]
    # Retained per-action output statistics; adapters qualify and resolve the size.
    statistic_bytes_per_action: int
    contract_digest: str = "legacy-internal@1"
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.definition_id.strip() or not self.credit_digest.strip() or not self.contributions:
            raise InvalidPolicyUpdate("objective population requires definition, credit and contributions")
        if len({item.id for item in self.contributions}) != len(self.contributions):
            raise InvalidPolicyUpdate("objective contribution identities must be unique")
        if type(self.statistic_bytes_per_action) is not int or self.statistic_bytes_per_action < 1:
            raise InvalidPolicyUpdate("objective requires its retained statistic size")
        policy, kl = frozen_array(self.policy, bool), frozen_array(self.kl, bool)
        if policy.ndim != 1 or policy.shape != kl.shape:
            raise InvalidPolicyUpdate("objective selections must be aligned position masks")
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "kl", kl)
        object.__setattr__(
            self,
            "digest",
            payload_digest(
                {
                    "schema": "posttrain.objective-population.v2",
                    "definition_id": self.definition_id,
                    "credit_digest": self.credit_digest,
                    "contributions": [[item.id, item.view, list(item.dependency_views)] for item in self.contributions],
                    "policy": array_digest(policy),
                    "kl": array_digest(kl),
                    "required_statistics": list(self.required_statistics),
                    "statistic_bytes_per_action": self.statistic_bytes_per_action,
                    "contract_digest": self.contract_digest,
                }
            ),
        )

    @property
    def selected(self) -> np.ndarray:
        return self.policy | self.kl


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedUpdate:
    """One optimizer occurrence: its contributions and the turns it must score.

    ``contributions`` index ``objective.contributions``. ``views`` are the
    dependency turns ordered by (episode, branch, turn id).
    """

    population: PopulationSnapshot
    objective: ObjectivePopulation
    schedule_digest: str
    epoch: int
    minibatch: int
    contributions: tuple[int, ...]
    occurrence_ids: tuple[str, ...]
    views: tuple[int, ...]
    discarded_contributions: tuple[str, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.objective.policy.shape != (self.population.size,):
            raise InvalidPolicyUpdate("resolved occurrence objective belongs to a different population")
        if len(self.occurrence_ids) != len(self.contributions) or any(
            not 0 <= index < len(self.objective.contributions) for index in self.contributions
        ):
            raise InvalidPolicyUpdate("resolved occurrence addresses absent contributions")
        object.__setattr__(
            self,
            "digest",
            payload_digest(
                {
                    "schema": "posttrain.resolved-update.v2",
                    "population": self.population.digest,
                    "objective": self.objective.digest,
                    "schedule_digest": self.schedule_digest,
                    "epoch": self.epoch,
                    "minibatch": self.minibatch,
                    "contributions": [self.objective.contributions[index].id for index in self.contributions],
                    "occurrence_ids": list(self.occurrence_ids),
                    "views": list(self.views),
                    "discarded_contributions": list(self.discarded_contributions),
                }
            ),
        )

    @property
    def dependency_count(self) -> int:
        """Positions whose scores this occurrence retains."""
        offsets = self.population.offsets
        return int(sum(int(offsets[view + 1] - offsets[view]) for view in self.views))


@dataclass(frozen=True, slots=True)
class ExecutionCapabilities:
    definition_ids: tuple[str, ...]
    statistics: tuple[str, ...]
    max_context_tokens: int
    # A coupled backward graph must remain intact across packs unless qualified replay exists.
    cross_pack_dependencies: bool = False
    context_layout: Literal["ragged", "dense-pack", "dense-population"] = "ragged"
    # The backend scores every turn whose context is a prefix of another turn's
    # context in the same pack from that longer context's single forward pass.
    prefix_sharing: bool = False

    def __post_init__(self) -> None:
        if self.context_layout not in {"ragged", "dense-pack", "dense-population"}:
            raise InvalidPolicyUpdate("unqualified physical context layout")


def population_context_width(population: PopulationSnapshot) -> int:
    """Maximum context actually addressed by sampled actions, including unselected turns."""
    return max(view.context_tokens for view in population.conditioning)


def execution_context_tokens(
    population: PopulationSnapshot,
    views: tuple[int, ...],
    capabilities: ExecutionCapabilities,
) -> int:
    """Count physical input slots; padding never becomes sampled or objective support."""
    if not views:
        return 0
    if len(set(views)) != len(views) or any(not 0 <= view < len(population.conditioning) for view in views):
        raise InvalidPolicyUpdate("physical context cost requires unique retained conditioning identities")
    lengths = [population.conditioning[view].context_tokens for view in views]
    if max(lengths) > capabilities.max_context_tokens:
        raise InvalidPolicyUpdate("actual conditioning view exceeds qualified backend context capacity")
    if capabilities.context_layout == "ragged":
        return sum(lengths)
    width = max(lengths) if capabilities.context_layout == "dense-pack" else population_context_width(population)
    if width > capabilities.max_context_tokens:
        raise InvalidPolicyUpdate("physical padding width exceeds qualified backend context capacity")
    return len(views) * width


@dataclass(frozen=True, slots=True)
class ExecutionPack:
    """Turns scored together, grouped by the contexts forwarded to score them.

    Each cover is (forwarded turn, turns it scores), see ``prefix_covers``.
    Without prefix sharing every turn covers only itself. ``context_tokens``
    is the physical input cost of the pack's forwards.
    """

    update_digest: str
    index: int
    covers: tuple[tuple[int, tuple[int, ...]], ...]
    context_tokens: int

    @property
    def views(self) -> tuple[int, ...]:
        return tuple(sorted(view for _, members in self.covers for view in members))

    @property
    def contexts(self) -> tuple[int, ...]:
        return tuple(cover for cover, _ in self.covers)


@dataclass(frozen=True, slots=True)
class UpdateCursor:
    population_digest: str
    schedule_digest: str
    next_occurrence: int
    applied_updates: int
    attempts: int
    versions: PolicyVersions


def turn_order(population: PopulationSnapshot, views: set[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Views ordered by (episode, branch, turn id): the order packs and scores follow."""
    conditioning = population.conditioning
    return tuple(
        sorted(
            views, key=lambda view: (conditioning[view].episode_id, conditioning[view].branch_id, conditioning[view].id)
        )
    )


def view_path(view: ConditioningView) -> tuple[str, tuple[int, ...]] | None:
    """(trace id, root-to-node path) from a view's native coordinates, if it has them."""
    try:
        coordinates = json.loads(view.token_ids_ref)
        trace, prefix, node = coordinates["trace_id"], coordinates["prefix_nodes"], coordinates["node_index"]
    except (TypeError, ValueError, KeyError):
        return None
    if not isinstance(trace, str) or type(node) is not int or not isinstance(prefix, list):
        return None
    if any(type(index) is not int for index in prefix):
        return None
    return trace, (*prefix, node)


def prefix_covers(population: PopulationSnapshot, views: tuple[int, ...]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Group turns under covering turns whose context contains theirs.

    A turn's causal context is the concatenation of its message-graph path, so
    when one turn's path is a prefix of another's in the same trace, its tokens
    are a prefix of the other's and causal attention gives it identical scores
    inside the longer forward. Each cover is (covering turn, covered turns in
    turn order, the cover included); turns without native coordinates cover only
    themselves. Covers keep the input turn order of their covering turns.
    """
    order = {view: rank for rank, view in enumerate(views)}
    paths = {view: view_path(population.conditioning[view]) for view in views}
    covers: dict[int, list[int]] = {}

    def longest_first(view: int) -> tuple[int, int]:
        path = paths[view]
        return (-len(path[1]) if path is not None else 0, order[view])

    # Longest paths first, so each turn joins the longest context that contains it.
    for view in sorted(views, key=longest_first):
        path = paths[view]
        host = None
        if path is not None:
            for cover in covers:
                candidate = paths[cover]
                if candidate is not None and candidate[0] == path[0] and candidate[1][: len(path[1])] == path[1]:
                    host = cover
                    break
        if host is None:
            covers[view] = [view]
        else:
            covers[host].append(view)
    return tuple(
        (cover, tuple(sorted(members, key=order.__getitem__)))
        for cover, members in sorted(covers.items(), key=lambda item: order[item[0]])
    )


def resolve_updates(
    snapshot: PopulationSnapshot,
    schedule: PolicyUpdateSchedule,
    objective: ObjectivePopulation,
) -> tuple[ResolvedUpdate, ...]:
    if objective.policy.shape != (snapshot.size,):
        raise InvalidPolicyUpdate("objective dependencies exceed admitted original actions")
    selected = objective.selected
    selected_per_view = np.add.reduceat(selected.astype(np.int64), snapshot.offsets[:-1]) if snapshot.size else []
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, contribution in enumerate(objective.contributions):
        view = snapshot.conditioning[contribution.view]
        if schedule.unit == "selected-token":
            key: tuple[str, ...] = (contribution.id,)
        elif schedule.unit == "episode":
            key = (view.episode_id,)
        else:
            key = (view.episode_id, view.branch_id, view.id)
        groups.setdefault(key, []).append(index)
    atoms = tuple(tuple(items) for items in groups.values())
    result: list[ResolvedUpdate] = []
    schedule_digest = record_digest(schedule)
    for epoch in range(schedule.epochs):
        ordered = list(atoms)
        if schedule.order == "shuffle":
            random.Random(schedule.seed + epoch).shuffle(ordered)
        batches: list[tuple[int, ...]] = []
        pending: list[int] = []
        cost = 0
        for atom in ordered:
            atom_cost = (
                sum(int(selected_per_view[objective.contributions[item].view]) for item in atom)
                if schedule.unit == "selected-token"
                else 1
            )
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
                discarded = tuple(objective.contributions[item].id for item in pending)
            else:
                batches.append(tuple(pending))
        if not batches:
            raise InvalidPolicyUpdate("schedule drops every contribution; no optimizer update remains")
        for minibatch, batch in enumerate(batches):
            contributions = [objective.contributions[item] for item in batch]
            result.append(
                ResolvedUpdate(
                    snapshot,
                    objective,
                    schedule_digest,
                    epoch,
                    minibatch,
                    batch,
                    tuple(f"{snapshot.id}/{epoch}/{minibatch}/{item.id}" for item in contributions),
                    turn_order(snapshot, {view for item in contributions for view in item.dependency_views}),
                    discarded,
                )
            )
    return tuple(result)


def validate_update(update: ResolvedUpdate, capabilities: ExecutionCapabilities) -> None:
    if update.objective.definition_id not in capabilities.definition_ids:
        raise InvalidPolicyUpdate(f"backend has no qualified objective {update.objective.definition_id!r}")
    missing = set(update.objective.required_statistics) - set(capabilities.statistics)
    if missing:
        raise InvalidPolicyUpdate(f"backend lacks required statistics: {sorted(missing)}")
    conditioning = update.population.conditioning
    if any(conditioning[view].context_tokens > capabilities.max_context_tokens for view in update.views):
        raise InvalidPolicyUpdate("actual conditioning view exceeds qualified backend context capacity")


def plan_packs(
    update: ResolvedUpdate,
    execution_budget: PolicyExecutionBudget,
    capabilities: ExecutionCapabilities,
) -> tuple[ExecutionPack, ...]:
    """Split an occurrence's turns into packs within the record and context budgets.

    The unit is a cover: a forwarded context and the turns it scores (with
    ``prefix_sharing``, every turn of the occurrence whose context it contains;
    otherwise just itself). Covers follow the occurrence's (episode, branch,
    turn id) order; a new pack starts when the forward count (``records``) or
    the physical context would exceed its budget. A cover is never split.
    """
    validate_update(update, capabilities)
    if update.dependency_count * update.objective.statistic_bytes_per_action > execution_budget.statistic_bytes:
        raise InvalidPolicyUpdate("retained objective statistics exceed execution capacity")
    population = update.population
    covers = (
        prefix_covers(population, update.views)
        if capabilities.prefix_sharing
        else tuple((view, (view,)) for view in update.views)
    )
    packs: list[ExecutionPack] = []
    pending: list[tuple[int, tuple[int, ...]]] = []

    def cost(contexts: list[int]) -> int:
        return execution_context_tokens(population, tuple(contexts), capabilities)

    def emit() -> None:
        packs.append(ExecutionPack(update.digest, len(packs), tuple(pending), cost([cover for cover, _ in pending])))

    for cover in covers:
        if cost([cover[0]]) > execution_budget.context_tokens:
            raise InvalidPolicyUpdate("atomic turn conditioning exceeds hard pack capacity")
        if pending and (
            len(pending) == execution_budget.records
            or cost([*(item for item, _ in pending), cover[0]]) > execution_budget.context_tokens
        ):
            emit()
            pending = []
        pending.append(cover)
    if pending:
        emit()
    if len(packs) > 1 and not capabilities.cross_pack_dependencies:
        pack_of = {view: pack.index for pack in packs for view in pack.views}
        for index in update.contributions:
            dependencies = update.objective.contributions[index].dependency_views
            if len({pack_of[view] for view in dependencies}) > 1:
                raise InvalidPolicyUpdate(
                    "objective dependency crosses packs without qualified graph retention or replay"
                )
    return tuple(packs)
