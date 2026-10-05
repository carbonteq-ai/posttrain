"""Closed policy objective definitions and pure global reduction contracts.

Mathematical qualification is separate from native backend qualification. Adding
an entry here does not advertise a public algorithm or authorize backend support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .update_credit import PreparedCredit
from .update_plan import ContributionRef, ObjectivePopulation, ResolvedUpdate
from .update_records import (
    ActionRef,
    ActionSelection,
    InvalidPolicyUpdate,
    PopulationSnapshot,
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


@dataclass(frozen=True, slots=True)
class ReductionWeight:
    action: ActionRef
    weight: float


@dataclass(frozen=True, slots=True)
class ResolvedObjectiveTerm:
    spec: ObjectiveSpec
    update_digest: str
    credit_digest: str
    policy_weights: tuple[ReductionWeight, ...]
    kl_weights: tuple[ReductionWeight, ...]
    ratio_support: tuple[tuple[ActionRef, ...], ...]
    # These explicit denominators are observation/recovery evidence.
    policy_denominators: tuple[tuple[str, int], ...]
    kl_denominators: tuple[tuple[str, int], ...]
    zero_policy_episodes: tuple[str, ...]
    zero_kl_episodes: tuple[str, ...]
    parameter_version: str

    @property
    def digest(self) -> str:
        return record_digest(self)


def objective_population(
    snapshot: PopulationSnapshot,
    spec: ObjectiveSpec,
    credit: PreparedCredit,
) -> ObjectivePopulation:
    """One atomic contribution per turn with the declared ratio dependencies."""
    credit.validate(snapshot)
    definition = objective_definition(spec.definition_id)
    policy = set(spec.policy_selection.resolve(snapshot))
    kl = set(spec.kl_selection.resolve(snapshot)) if spec.beta else set()
    selected = policy | kl
    if not selected and spec.empty_policy != "zero":
        raise InvalidPolicyUpdate("objective selects no contributions")
    by_episode: dict[str, list[ActionRef]] = {}
    for record in snapshot.actions:
        by_episode.setdefault(record.action.episode_id, []).append(record.action)
    episodes = {episode: tuple(actions) for episode, actions in by_episode.items()}
    turns: dict[tuple[str, str, str], list[ActionRef]] = {}
    for record in snapshot.actions:
        action = record.action
        turns.setdefault((action.episode_id, action.branch_id, action.turn_id), []).append(action)
    contributions = tuple(
        ContributionRef(
            record_digest(actions[0]),
            tuple(action for action in actions if action in selected),
            episodes[key[0]]
            if definition.ratio == "episode-geometric" and any(action in policy for action in actions)
            else tuple(actions)
            if definition.ratio == "turn-geometric" and any(action in policy for action in actions)
            else tuple(action for action in actions if action in selected),
            tuple(actions),
        )
        for key, actions in turns.items()
    )
    # Validate empty semantics against the declared population before scheduling.
    original = {record.action for record in snapshot.actions}
    _reduction(policy, original, original, spec)
    if spec.beta:
        _reduction(kl, original, original, spec)
    statistics = ("sampled-logp", "old-logp") + (("reference-logp",) if spec.beta else ())
    # FP32 score/adjoint plus frozen old scores, optional reference and correction.
    return ObjectivePopulation(
        spec.definition_id, credit.digest, contributions, statistics, 20 if spec.beta else 16, spec.digest
    )


def _reduction(
    selected: set[ActionRef],
    domain: set[ActionRef],
    original: set[ActionRef],
    spec: ObjectiveSpec,
) -> tuple[tuple[ReductionWeight, ...], tuple[tuple[str, int], ...], tuple[str, ...]]:
    definition = objective_definition(spec.definition_id)

    turn_units: dict[tuple[str, str, str], str] = {}

    def unit(action: ActionRef) -> str:
        if definition.reduction != "equal-turn":
            return action.episode_id
        key = (action.episode_id, action.branch_id, action.turn_id)
        value = turn_units.get(key)
        if value is None:
            value = turn_units[key] = f"turn/{record_digest(ActionRef(*key, 0))}"
        return value

    episodes = tuple(sorted({unit(action) for action in domain}))
    grouped: dict[str, list[ActionRef]] = {}
    for action in selected:
        grouped.setdefault(unit(action), []).append(action)
    support = {episode: tuple(sorted(grouped.get(episode, ()))) for episode in episodes}
    empty = tuple(episode for episode in episodes if not support[episode])
    if empty and spec.empty_policy == "reject":
        raise InvalidPolicyUpdate(f"empty objective selection for episodes {empty}")
    retained = tuple(episode for episode in episodes if support[episode] or spec.empty_policy == "zero")
    if not retained:
        raise InvalidPolicyUpdate("empty objective reduction after omission")
    sizes: dict[str, int] = {}
    if spec.denominator != "selected":
        for action in original:
            key = unit(action)
            sizes[key] = sizes.get(key, 0) + 1
    denominators = tuple(
        (episode, len(support[episode]) if spec.denominator == "selected" else sizes.get(episode, 0))
        for episode in retained
    )
    if definition.reduction == "selected-token":
        denominator = sum(value for _, value in denominators)
        if denominator == 0:
            if spec.empty_policy == "zero":
                return (), (("global", 0),), empty
            raise InvalidPolicyUpdate("empty selected-token denominator")
        weights = tuple(ReductionWeight(action, 1 / denominator) for action in sorted(selected))
    else:
        weights = tuple(
            ReductionWeight(action, 1 / (len(retained) * denominator))
            for episode, denominator in denominators
            for action in support[episode]
        )
    return weights, denominators, empty


def resolve_objective_term(
    update: ResolvedUpdate,
    spec: ObjectiveSpec,
    credit: PreparedCredit,
    *,
    parameter_version: str | None = None,
) -> ResolvedObjectiveTerm:
    parameter_version = update.population.versions.current if parameter_version is None else parameter_version
    require_identity(parameter_version)
    credit.validate(update.population)
    if update.objective.definition_id != spec.definition_id or update.objective.contract_digest != spec.digest:
        raise InvalidPolicyUpdate("update objective differs from resolved specification")
    if update.objective.credit_digest != credit.digest:
        raise InvalidPolicyUpdate("update prepared credit identity changed")
    selected = {action for item in update.contributions for action in item.actions}
    domain = {action for item in update.contributions for action in (item.reduction_domain or item.actions)}
    original = {record.action for record in update.population.actions}
    policy = selected & set(spec.policy_selection.resolve(update.population))
    kl = selected & set(spec.kl_selection.resolve(update.population)) if spec.beta else set()
    policy_weights, policy_denominators, zero_policy = _reduction(policy, domain, original, spec)
    if spec.beta:
        kl_weights, kl_denominators, zero_kl = _reduction(kl, domain, original, spec)
    else:
        kl_weights, kl_denominators, zero_kl = (), (), ()
    definition = objective_definition(spec.definition_id)
    if definition.ratio == "token":
        ratio_support = tuple((action,) for action in sorted(policy))
    elif definition.ratio == "episode-geometric":
        by_episode: dict[str, list[ActionRef]] = {}
        for action in original:
            by_episode.setdefault(action.episode_id, []).append(action)
        ratio_support = tuple(
            tuple(sorted(by_episode[episode])) for episode in sorted({action.episode_id for action in policy})
        )
    else:
        by_turn: dict[tuple[str, str, str], list[ActionRef]] = {}
        for action in original:
            by_turn.setdefault((action.episode_id, action.branch_id, action.turn_id), []).append(action)
        turns = sorted({(action.episode_id, action.branch_id, action.turn_id) for action in policy})
        ratio_support = tuple(tuple(sorted(by_turn[turn])) for turn in turns)
    if not {action for support in ratio_support for action in support} <= set(update.dependencies):
        raise InvalidPolicyUpdate("resolved update lost ratio dependencies")
    return ResolvedObjectiveTerm(
        spec,
        update.digest,
        credit.digest,
        policy_weights,
        kl_weights,
        ratio_support,
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
