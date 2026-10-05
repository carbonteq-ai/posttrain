"""Closed policy objective definitions and pure global reduction contracts.

Mathematical qualification is separate from native backend qualification. Adding
an entry here does not advertise a public algorithm or authorize backend support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .update_credit import PreparedCredit
from .update_plan import ContributionRef, ObjectivePopulation, ResolvedUpdate
from .update_records import (
    ActionRef,
    ActionSelection,
    InvalidPolicyUpdate,
    PopulationSnapshot,
    array_digest,
    frozen_array,
    payload_digest,
    record_digest,
    require_identity,
)


@dataclass(frozen=True, slots=True)
class ObjectiveDefinition:
    id: str
    ratio: Literal["token", "episode-geometric", "turn-geometric"]
    derivative: Literal["coupled", "token-local", "detached-weight"]
    policy_loss: Literal["clipped-surrogate", "cispo-upper"]
    reduction: Literal["equal-episode", "equal-turn", "selected-token"]
    allows_spans: bool = False


# These are versioned formula contracts, not freely interchangeable switches.
_DEFINITIONS = (
    ObjectiveDefinition("grpo@1", "token", "coupled", "clipped-surrogate", "equal-episode"),
    ObjectiveDefinition("dapo@1", "token", "coupled", "clipped-surrogate", "selected-token"),
    ObjectiveDefinition("gdpo@1", "token", "coupled", "clipped-surrogate", "equal-episode"),
    ObjectiveDefinition("capo@1", "token", "coupled", "clipped-surrogate", "equal-episode"),
    ObjectiveDefinition("sampo@1", "episode-geometric", "token-local", "clipped-surrogate", "equal-episode"),
    ObjectiveDefinition("sampo-turns@1", "turn-geometric", "token-local", "clipped-surrogate", "equal-turn"),
    ObjectiveDefinition(
        "sampo-spans@1", "episode-geometric", "token-local", "clipped-surrogate", "equal-episode", True
    ),
    ObjectiveDefinition("gspo-token@1", "episode-geometric", "token-local", "clipped-surrogate", "equal-episode", True),
    ObjectiveDefinition("sequence-coupled@1", "episode-geometric", "coupled", "clipped-surrogate", "equal-episode"),
    ObjectiveDefinition("cispo-upper@1", "token", "detached-weight", "cispo-upper", "selected-token", True),
)


def objective_definition(identity: str) -> ObjectiveDefinition:
    for definition in _DEFINITIONS:
        if definition.id == identity:
            return definition
    raise InvalidPolicyUpdate(f"unsupported versioned objective {identity!r}")


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    definition_id: str
    clip_low: float = 0.2
    clip_high: float = 0.2
    beta: float = 0.0
    # CISPO uses an absolute upper weight bound, not PPO's epsilon_high.
    cispo_max_weight: float | None = None
    policy_selection: ActionSelection = ActionSelection()
    kl_selection: ActionSelection = ActionSelection()
    denominator: Literal["selected", "original-eligible"] = "selected"
    empty_policy: Literal["reject", "omit", "zero"] = "reject"

    def __post_init__(self) -> None:
        definition = objective_definition(self.definition_id)
        if any(
            isinstance(value, bool) or not math.isfinite(value) for value in (self.clip_low, self.clip_high, self.beta)
        ):
            raise InvalidPolicyUpdate("objective coefficients must be finite numeric values")
        if not 0 < self.clip_low < 1 or self.clip_high <= 0 or self.beta < 0:
            raise InvalidPolicyUpdate("invalid objective clipping or KL coefficients")
        if self.denominator not in {"selected", "original-eligible"} or self.empty_policy not in {
            "reject",
            "omit",
            "zero",
        }:
            raise InvalidPolicyUpdate("unsupported reduction denominator or empty policy")
        if not definition.allows_spans and (
            self.policy_selection.mode != "all"
            or self.kl_selection.mode != "all"
            or self.denominator != "selected"
            or self.empty_policy != "reject"
        ):
            raise InvalidPolicyUpdate(
                "legacy objective requires its original support and reductions; select a span variant"
            )
        if definition.policy_loss == "cispo-upper":
            if (
                self.cispo_max_weight is None
                or isinstance(self.cispo_max_weight, bool)
                or (not math.isfinite(self.cispo_max_weight) or self.cispo_max_weight <= 0)
            ):
                raise InvalidPolicyUpdate("CISPO upper definition requires an absolute positive weight cap")
        elif self.cispo_max_weight is not None:
            raise InvalidPolicyUpdate("CISPO weight cap is unsupported for this objective")

    @property
    def digest(self) -> str:
        return record_digest(self)


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedObjectiveTerm:
    """One occurrence's objective as arrays over population positions.

    ``policy_weight`` and ``kl_weight`` are each position's global reduction
    weight (zero when unselected). ``ratio_segment`` groups the positions that
    share one importance ratio (one token, one turn or one episode), numbered
    from zero, with -1 outside any ratio support.
    """

    spec: ObjectiveSpec
    update_digest: str
    credit_digest: str
    policy_weight: np.ndarray
    kl_weight: np.ndarray
    ratio_segment: np.ndarray
    segment_count: int
    # These explicit denominators are observation/recovery evidence.
    policy_denominators: tuple[tuple[str, int], ...]
    kl_denominators: tuple[tuple[str, int], ...]
    zero_policy_episodes: tuple[str, ...]
    zero_kl_episodes: tuple[str, ...]
    parameter_version: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        policy, kl = frozen_array(self.policy_weight, np.float64), frozen_array(self.kl_weight, np.float64)
        segment = frozen_array(self.ratio_segment, np.int64)
        if policy.ndim != 1 or policy.shape != kl.shape or policy.shape != segment.shape:
            raise InvalidPolicyUpdate("objective term arrays must align with population positions")
        if (policy < 0).any() or (kl < 0).any() or not (np.isfinite(policy).all() and np.isfinite(kl).all()):
            raise InvalidPolicyUpdate("objective reduction weights must be finite and non-negative")
        if segment.size and (segment.min() < -1 or segment.max() >= self.segment_count):
            raise InvalidPolicyUpdate("objective ratio segments must be numbered from zero")
        if ((policy > 0) & (segment < 0)).any():
            raise InvalidPolicyUpdate("selected policy action lacks its importance ratio support")
        object.__setattr__(self, "policy_weight", policy)
        object.__setattr__(self, "kl_weight", kl)
        object.__setattr__(self, "ratio_segment", segment)
        object.__setattr__(
            self,
            "digest",
            payload_digest(
                {
                    "schema": "posttrain.objective-term.v2",
                    "spec": self.spec.digest,
                    "update_digest": self.update_digest,
                    "credit_digest": self.credit_digest,
                    "policy_weight": array_digest(policy),
                    "kl_weight": array_digest(kl),
                    "ratio_segment": array_digest(segment),
                    "segment_count": self.segment_count,
                    "policy_denominators": [list(item) for item in self.policy_denominators],
                    "kl_denominators": [list(item) for item in self.kl_denominators],
                    "zero_policy_episodes": list(self.zero_policy_episodes),
                    "zero_kl_episodes": list(self.zero_kl_episodes),
                    "parameter_version": self.parameter_version,
                }
            ),
        )

    @property
    def policy_positions(self) -> np.ndarray:
        return np.flatnonzero(self.policy_weight > 0)

    @property
    def kl_positions(self) -> np.ndarray:
        return np.flatnonzero(self.kl_weight > 0)


def _per_view(snapshot: PopulationSnapshot, mask: np.ndarray) -> np.ndarray:
    """Count of true positions in each view."""
    return np.add.reduceat(mask.astype(np.int64), snapshot.offsets[:-1])


def objective_population(
    snapshot: PopulationSnapshot,
    spec: ObjectiveSpec,
    credit: PreparedCredit,
) -> ObjectivePopulation:
    """One atomic contribution per turn with the declared ratio dependencies."""
    credit.validate(snapshot)
    definition = objective_definition(spec.definition_id)
    policy = spec.policy_selection.resolve(snapshot)
    kl = spec.kl_selection.resolve(snapshot) if spec.beta else np.zeros(snapshot.size, dtype=bool)
    selected = policy | kl
    if not selected.any() and spec.empty_policy != "zero":
        raise InvalidPolicyUpdate("objective selects no contributions")
    policy_per_view, selected_per_view = _per_view(snapshot, policy), _per_view(snapshot, selected)
    episode_views: dict[tuple[str, str], list[int]] = {}
    for index, view in enumerate(snapshot.conditioning):
        episode_views.setdefault((view.episode_id, view.branch_id), []).append(index)
    contributions = []
    for index, view in enumerate(snapshot.conditioning):
        if policy_per_view[index] and definition.ratio == "episode-geometric":
            dependencies = tuple(episode_views[(view.episode_id, view.branch_id)])
        elif policy_per_view[index] and definition.ratio == "turn-geometric":
            dependencies = (index,)
        else:
            dependencies = (index,) if selected_per_view[index] else ()
        first = ActionRef(view.episode_id, view.branch_id, view.id, view.sampled[0])
        contributions.append(ContributionRef(record_digest(first), index, dependencies))
    # Validate empty semantics against the declared population before scheduling.
    everything = np.ones(snapshot.size, dtype=bool)
    _reduction(policy, everything, snapshot, spec)
    if spec.beta:
        _reduction(kl, everything, snapshot, spec)
    statistics = ("sampled-logp", "old-logp") + (("reference-logp",) if spec.beta else ())
    # FP32 score/adjoint plus frozen old scores, optional reference and correction.
    return ObjectivePopulation(
        spec.definition_id,
        credit.digest,
        tuple(contributions),
        policy,
        kl,
        statistics,
        20 if spec.beta else 16,
        spec.digest,
    )


def _view_units(snapshot: PopulationSnapshot, spec: ObjectiveSpec) -> list[str]:
    """Each view's reduction unit: its episode, or its turn under equal-turn reduction."""
    if objective_definition(spec.definition_id).reduction != "equal-turn":
        return [view.episode_id for view in snapshot.conditioning]
    return [
        f"turn/{record_digest(ActionRef(view.episode_id, view.branch_id, view.id, 0))}"
        for view in snapshot.conditioning
    ]


def _reduction(
    selected: np.ndarray,
    domain: np.ndarray,
    snapshot: PopulationSnapshot,
    spec: ObjectiveSpec,
) -> tuple[np.ndarray, tuple[tuple[str, int], ...], tuple[str, ...]]:
    """Global reduction weights over the selected positions, per the definition's unit.

    Units are the episodes (or turns) the domain touches. A unit without selected
    actions is rejected, omitted or kept with zero weight per ``empty_policy``.
    """
    definition = objective_definition(spec.definition_id)
    units = _view_units(snapshot, spec)
    selected_per_view, domain_per_view = _per_view(snapshot, selected), _per_view(snapshot, domain)
    sizes_per_view = np.diff(snapshot.offsets)
    unit_selected: dict[str, int] = {}
    unit_size: dict[str, int] = {}
    domain_units: set[str] = set()
    for view, unit in enumerate(units):
        unit_selected[unit] = unit_selected.get(unit, 0) + int(selected_per_view[view])
        unit_size[unit] = unit_size.get(unit, 0) + int(sizes_per_view[view])
        if domain_per_view[view]:
            domain_units.add(unit)
    episodes = tuple(sorted(domain_units))
    empty = tuple(unit for unit in episodes if not unit_selected[unit])
    if empty and spec.empty_policy == "reject":
        raise InvalidPolicyUpdate(f"empty objective selection for episodes {empty}")
    retained = tuple(unit for unit in episodes if unit_selected[unit] or spec.empty_policy == "zero")
    if not retained:
        raise InvalidPolicyUpdate("empty objective reduction after omission")
    denominators = tuple(
        (unit, unit_selected[unit] if spec.denominator == "selected" else unit_size[unit]) for unit in retained
    )
    weights = np.zeros(snapshot.size, dtype=np.float64)
    if definition.reduction == "selected-token":
        denominator = sum(value for _, value in denominators)
        if denominator == 0:
            if spec.empty_policy == "zero":
                return weights, (("global", 0),), empty
            raise InvalidPolicyUpdate("empty selected-token denominator")
        weights[selected] = 1 / denominator
        return weights, denominators, empty
    unit_weight = {unit: 1 / (len(retained) * value) for unit, value in denominators if value}
    view_weight = np.array([unit_weight.get(unit, 0.0) for unit in units], dtype=np.float64)
    weights[selected] = view_weight[snapshot.view_of[selected]]
    return weights, denominators, empty


def resolve_objective_term(
    update: ResolvedUpdate,
    spec: ObjectiveSpec,
    credit: PreparedCredit,
    *,
    parameter_version: str | None = None,
) -> ResolvedObjectiveTerm:
    snapshot = update.population
    parameter_version = snapshot.versions.current if parameter_version is None else parameter_version
    require_identity(parameter_version)
    credit.validate(snapshot)
    objective = update.objective
    if objective.definition_id != spec.definition_id or objective.contract_digest != spec.digest:
        raise InvalidPolicyUpdate("update objective differs from resolved specification")
    if objective.credit_digest != credit.digest:
        raise InvalidPolicyUpdate("update prepared credit identity changed")
    domain = snapshot.views_mask(tuple(objective.contributions[index].view for index in update.contributions))
    selected = domain & objective.selected
    policy = selected & objective.policy
    policy_weight, policy_denominators, zero_policy = _reduction(policy, domain, snapshot, spec)
    if spec.beta:
        kl_weight, kl_denominators, zero_kl = _reduction(selected & objective.kl, domain, snapshot, spec)
    else:
        kl_weight, kl_denominators, zero_kl = np.zeros(snapshot.size, dtype=np.float64), (), ()
    definition = objective_definition(spec.definition_id)
    segment = np.full(snapshot.size, -1, dtype=np.int64)
    if definition.ratio == "token":
        positions = np.flatnonzero(policy)
        segment[positions] = np.arange(positions.size)
        count = int(positions.size)
    else:
        owners = snapshot.episode_of if definition.ratio == "episode-geometric" else snapshot.view_of
        touched = np.unique(owners[policy])
        lookup = np.full(int(owners.max()) + 1 if owners.size else 0, -1, dtype=np.int64)
        lookup[touched] = np.arange(touched.size)
        segment = lookup[owners]
        count = int(touched.size)
    if not (segment < 0).all() and not snapshot.views_mask(update.views)[segment >= 0].all():
        raise InvalidPolicyUpdate("resolved update lost ratio dependencies")
    return ResolvedObjectiveTerm(
        spec,
        update.digest,
        credit.digest,
        policy_weight,
        kl_weight,
        segment,
        count,
        policy_denominators,
        kl_denominators,
        zero_policy,
        zero_kl,
        parameter_version,
    )


def distributed_weight(weight: float, *, averaging_ranks: int) -> float:
    """DDP averages gradients: each owned global contribution is multiplied by W.

    Empty ranks must still take part in native collectives. This does not validate
    rank ownership, nor authorize duplicate work or change a global denominator.
    """
    if type(averaging_ranks) is not int or averaging_ranks < 1 or not math.isfinite(weight) or weight < 0:
        raise InvalidPolicyUpdate("invalid distributed global reduction weight")
    return weight * averaging_ranks
