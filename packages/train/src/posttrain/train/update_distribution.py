"""Exclusive original-context ownership for distributed score/replay execution.

The objective is still evaluated over complete global scores. Ownership assigns
score and derivative work, never a new per-rank loss denominator. Native adapters
must separately qualify collectives, replay, empty ranks and update transactions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .update_objectives import ResolvedObjectiveTerm
from .update_plan import ResolvedUpdate
from .update_records import InvalidPolicyUpdate, record_digest


@dataclass(frozen=True, slots=True)
class RankScorePartition:
    """The turns (conditioning views) one averaging rank scores."""

    rank: int
    views: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RankScoreWork:
    view: int
    contributes: bool


@dataclass(frozen=True, slots=True)
class DistributedScorePlan:
    update_digest: str
    term_digest: str
    partitions: tuple[RankScorePartition, ...]

    def __post_init__(self) -> None:
        if not self.update_digest.strip() or not self.term_digest.strip() or not self.partitions:
            raise InvalidPolicyUpdate("distributed score plan requires resolved identities and ranks")
        views = tuple(view for part in self.partitions for view in part.views)
        if (
            any(type(part.rank) is not int or part.rank != index for index, part in enumerate(self.partitions))
            or len(set(views)) != len(views)
            or not views
            or any(type(view) is not int or view < 0 for view in views)
        ):
            raise InvalidPolicyUpdate("distributed score ownership must be ordered and exclusive")

    @property
    def digest(self) -> str:
        return record_digest(self)

    @property
    def views(self) -> tuple[int, ...]:
        return tuple(sorted(view for part in self.partitions for view in part.views))


def resolve_score_ownership(
    update: ResolvedUpdate,
    term: ResolvedObjectiveTerm,
    view_owners: tuple[tuple[int, ...], ...],
) -> DistributedScorePlan:
    """Bind every required original turn to exactly one averaging rank.

    Cross-rank ratio dependencies remain in the global term. Empty partitions are
    legal; their native executor must still join required collectives and backward.
    """
    if term.update_digest != update.digest or term.credit_digest != update.objective.credit_digest:
        raise InvalidPolicyUpdate("distributed ownership received a different resolved term")
    assigned = tuple(view for owners in view_owners for view in owners)
    if not view_owners or len(set(assigned)) != len(assigned) or set(assigned) != set(update.views):
        raise InvalidPolicyUpdate("distributed ownership must cover each required original context exactly once")
    needed = (term.policy_weight > 0) | (term.kl_weight > 0) | (term.ratio_segment >= 0)
    if not update.population.views_mask(update.views)[needed].all():
        raise InvalidPolicyUpdate("distributed ownership lost global objective dependencies")
    return DistributedScorePlan(
        update.digest,
        term.digest,
        tuple(RankScorePartition(rank, tuple(sorted(owners))) for rank, owners in enumerate(view_owners)),
    )


def resolve_score_rounds(
    update: ResolvedUpdate,
    term: ResolvedObjectiveTerm,
    plan: DistributedScorePlan,
) -> tuple[tuple[RankScoreWork, ...], ...]:
    """One full original context per rank per round, padding with zero-work graphs.

    Padding uses an admitted turn; it owns no objective contribution. Every rank
    performs the same number of native forwards before one backward.
    """
    expected = resolve_score_ownership(update, term, tuple(part.views for part in plan.partitions))
    if expected != plan:
        raise InvalidPolicyUpdate("distributed round ownership differs from the original update")
    padding = RankScoreWork(plan.views[0], False)
    work = tuple(tuple(RankScoreWork(view, True) for view in part.views) for part in plan.partitions)
    return tuple(
        tuple(rows[index] if index < len(rows) else padding for rows in work)
        for index in range(max(len(rows) for rows in work))
    )


def owned_positions(update: ResolvedUpdate, plan: DistributedScorePlan, rank: int) -> np.ndarray:
    """Population positions whose scores ``rank`` owns."""
    return np.flatnonzero(update.population.views_mask(plan.partitions[rank].views))
