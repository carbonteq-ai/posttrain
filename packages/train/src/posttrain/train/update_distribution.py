"""Exclusive original-context ownership for distributed score/replay execution.

The objective is still evaluated over complete global scores. Ownership assigns
score and derivative work, never a new per-rank loss denominator. Native adapters
must separately qualify collectives, replay, empty ranks and update transactions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .update_objectives import ResolvedObjectiveTerm
from .update_plan import ResolvedUpdate
from .update_records import ActionRef, InvalidPolicyUpdate, record_digest


@dataclass(frozen=True, slots=True)
class RankScorePartition:
    rank: int
    context_ids: tuple[str, ...]
    actions: tuple[ActionRef, ...]


@dataclass(frozen=True, slots=True)
class RankScoreWork:
    context_id: str
    actions: tuple[ActionRef, ...]
    contributes: bool


@dataclass(frozen=True, slots=True)
class DistributedScorePlan:
    update_digest: str
    term_digest: str
    partitions: tuple[RankScorePartition, ...]

    def __post_init__(self) -> None:
        if not self.update_digest.strip() or not self.term_digest.strip() or not self.partitions:
            raise InvalidPolicyUpdate("distributed score plan requires resolved identities and ranks")
        contexts = tuple(context for part in self.partitions for context in part.context_ids)
        actions = tuple(action for part in self.partitions for action in part.actions)
        if (any(type(part.rank) is not int or part.rank != index for index, part in enumerate(self.partitions))
                or len(set(contexts)) != len(contexts) or len(set(actions)) != len(actions)
                or not actions or any(not value.strip() for value in contexts)):
            raise InvalidPolicyUpdate("distributed score ownership must be ordered and exclusive")
        if any(bool(part.actions) != bool(part.context_ids) for part in self.partitions):
            raise InvalidPolicyUpdate("empty distributed partitions cannot retain unowned contexts")

    @property
    def digest(self) -> str:
        return record_digest(self)

    @property
    def actions(self) -> tuple[ActionRef, ...]:
        return tuple(sorted(action for part in self.partitions for action in part.actions))


def resolve_score_ownership(
    update: ResolvedUpdate, term: ResolvedObjectiveTerm,
    context_owners: tuple[tuple[str, ...], ...],
) -> DistributedScorePlan:
    """Bind every required original context to exactly one averaging rank.

Cross-rank ratio dependencies remain in the global term. Empty partitions are
legal; their native executor must still join required collectives and backward.
"""
    if term.update_digest != update.digest or term.credit_digest != update.objective.credit_digest:
        raise InvalidPolicyUpdate("distributed ownership received a different resolved term")
    records = {record.action: record.conditioning_id for record in update.population.actions}
    required = {records[action] for action in update.dependencies}
    assigned = tuple(context for owners in context_owners for context in owners)
    if not context_owners or len(set(assigned)) != len(assigned) or set(assigned) != required:
        raise InvalidPolicyUpdate("distributed ownership must cover each required original context exactly once")
    selected = {value.action for value in (*term.policy_weights, *term.kl_weights)}
    ratio = {action for support in term.ratio_support for action in support}
    if not (selected | ratio) <= set(update.dependencies):
        raise InvalidPolicyUpdate("distributed ownership lost global objective dependencies")
    return DistributedScorePlan(update.digest, term.digest, tuple(
        RankScorePartition(rank, tuple(sorted(owners)),
                           tuple(action for action in update.dependencies if records[action] in owners))
        for rank, owners in enumerate(context_owners)
    ))


def resolve_score_rounds(
    update: ResolvedUpdate, term: ResolvedObjectiveTerm, plan: DistributedScorePlan,
) -> tuple[tuple[RankScoreWork, ...], ...]:
    """One full original context per rank per round, padding with zero-work graphs.

Padding uses an admitted context and action; it owns no objective contribution.
Every rank performs the same number of native forwards before one backward.
"""
    expected = resolve_score_ownership(update, term, tuple(part.context_ids for part in plan.partitions))
    if expected != plan:
        raise InvalidPolicyUpdate("distributed round ownership differs from the original update")
    contexts = {record.action: record.conditioning_id for record in update.population.actions}
    anchor = plan.actions[0]
    padding = RankScoreWork(contexts[anchor], (anchor,), False)
    work = tuple(tuple(RankScoreWork(context, tuple(action for action in part.actions if contexts[action] == context), True)
                       for context in part.context_ids) for part in plan.partitions)
    return tuple(tuple(rows[index] if index < len(rows) else padding for rows in work)
                 for index in range(max(len(rows) for rows in work)))
