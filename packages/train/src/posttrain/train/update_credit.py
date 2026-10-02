"""Detached advantage transport and validation on complete native populations."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from .online_rl import EnvironmentRollout
from .profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings, shape_online_reward
from .reward_advantages import compute_capo_advantages, compute_gdpo_advantages
from .reward_evidence import ObservationScope
from .sampo_advantages import compute_sampo_advantages
from .update_records import ActionRef, InvalidPolicyUpdate, PopulationSnapshot, record_digest, require_identity


@dataclass(frozen=True, slots=True)
class ActionCredit:
    action: ActionRef
    advantage: float

    def __post_init__(self) -> None:
        if isinstance(self.advantage, bool) or not math.isfinite(self.advantage):
            raise InvalidPolicyUpdate("prepared advantage must be a finite numeric value")


@dataclass(frozen=True, slots=True)
class PreparedCredit:
    population_digest: str
    estimator_id: str
    values: tuple[ActionCredit, ...]
    required_relations: tuple[str, ...]
    component_weights: tuple[tuple[str, float], ...]
    normalization: str
    observation_scope: ObservationScope
    evidence_digests: tuple[str, ...]
    meaning: Literal["detached-advantage"] = "detached-advantage"

    def __post_init__(self) -> None:
        require_identity(self.population_digest, self.estimator_id, self.normalization, *self.evidence_digests)
        if not self.evidence_digests or not self.values:
            raise InvalidPolicyUpdate("prepared credit requires retained evidence and action values")
        if self.meaning != "detached-advantage":
            raise InvalidPolicyUpdate("quality scores are not prepared detached advantages")
        if self.observation_scope not in {"prefix", "current-step", "full-trajectory"}:
            raise InvalidPolicyUpdate("prepared credit requires explicit estimator observation scope")
        if len({value.action for value in self.values}) != len(self.values):
            raise InvalidPolicyUpdate("prepared credit duplicates an original action")
        if len(set(self.required_relations)) != len(self.required_relations):
            raise InvalidPolicyUpdate("required credit populations must be unique")
        if len({name for name, _ in self.component_weights}) != len(self.component_weights):
            raise InvalidPolicyUpdate("credit component weights must have unique identities")
        for name, weight in self.component_weights:
            require_identity(name)
            if isinstance(weight, bool) or not math.isfinite(weight):
                raise InvalidPolicyUpdate("credit component weights must be finite")

    @property
    def digest(self) -> str:
        return record_digest(self)

    def validate(self, snapshot: PopulationSnapshot) -> None:
        if self.population_digest != snapshot.digest:
            raise InvalidPolicyUpdate("prepared credit belongs to different frozen evidence")
        if {value.action for value in self.values} != {record.action for record in snapshot.actions}:
            raise InvalidPolicyUpdate("prepared credit must cover eligible original actions exactly")
        relations = {relation.id: relation for relation in snapshot.relations}
        for relation_id in self.required_relations:
            relation = relations.get(relation_id)
            if relation is None or relation.completeness != "complete":
                raise InvalidPolicyUpdate(f"credit requires complete population {relation_id!r}")


class CreditEstimator(Protocol):
    """Estimator identity and declared populations are independently checked."""

    @property
    def id(self) -> str: ...

    @property
    def required_relations(self) -> tuple[str, ...]: ...

    def prepare(self, snapshot: PopulationSnapshot) -> PreparedCredit: ...


def prepare_credit(snapshot: PopulationSnapshot, estimator: CreditEstimator) -> PreparedCredit:
    relations = {relation.id: relation for relation in snapshot.relations}
    require_identity(estimator.id, *estimator.required_relations)
    for relation_id in estimator.required_relations:
        relation = relations.get(relation_id)
        if relation is None or relation.completeness != "complete":
            raise InvalidPolicyUpdate(f"estimator requires complete population {relation_id!r}")
    prepared = estimator.prepare(snapshot)
    if not isinstance(prepared, PreparedCredit):
        raise InvalidPolicyUpdate("estimator must return prepared advantages, not scorer evidence")
    if prepared.estimator_id != estimator.id or prepared.required_relations != estimator.required_relations:
        raise InvalidPolicyUpdate("estimator changed its declared identity or required populations")
    prepared.validate(snapshot)
    return prepared


@dataclass(frozen=True, slots=True)
class NativeCreditRows:
    """Adapter-proved original action addresses aligned with existing rollout rows.

    None denotes an ineligible observation/padding coordinate, never unavailable
    policy credit. This transports existing estimators without reimplementing them.
    Native bridge authenticity remains the adapter's separate evidence gate.
    """

    rollouts: tuple[EnvironmentRollout, ...]
    actions: tuple[tuple[ActionRef | None, ...], ...]
    evidence_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.rollouts or len(self.rollouts) != len(self.actions) or not self.evidence_digests:
            raise InvalidPolicyUpdate("native credit rows require aligned rollouts, coordinates and retained evidence")
        require_identity(*self.evidence_digests)
        for rollout, actions in zip(self.rollouts, self.actions, strict=True):
            if any(type(eligible) is not bool for eligible in rollout.env_mask):
                raise InvalidPolicyUpdate("native sampled eligibility must be boolean")
            if len(actions) != len(rollout.env_mask) or any(
                (action is not None) != eligible for action, eligible in zip(actions, rollout.env_mask, strict=True)
            ):
                raise InvalidPolicyUpdate("native credit coordinates must match original sampled eligibility")
            if len({(action.episode_id, action.branch_id) for action in actions if action is not None}) != 1:
                raise InvalidPolicyUpdate("one native credit row cannot combine different trajectories")
        addressed = [action for row in self.actions for action in row if action is not None]
        if len(set(addressed)) != len(addressed):
            raise InvalidPolicyUpdate("native credit rows duplicate an original action")

    def project(self, values: tuple[tuple[float, ...], ...]) -> tuple[ActionCredit, ...]:
        if len(values) != len(self.actions):
            raise InvalidPolicyUpdate("estimator output lost native row alignment")
        projected: list[ActionCredit] = []
        for actions, row in zip(self.actions, values, strict=True):
            if len(row) != len(actions):
                raise InvalidPolicyUpdate("estimator output lost original token alignment")
            for action, value in zip(actions, row, strict=True):
                if action is None:
                    if value != 0.0:
                        raise InvalidPolicyUpdate("estimator credited an ineligible native position")
                else:
                    projected.append(ActionCredit(action, value))
        return tuple(projected)


@dataclass(frozen=True, slots=True)
class ScalarGroupCreditEstimator:
    """GRPO/DAPO scalar credit with Posttrain's existing TRL-equivalent recipe.

    Group means use complete declared prompt groups. Standard deviations use
    Bessel's correction, with epsilon 1e-4, or are disabled by the selection.
    Batch scaling sees every admitted episode before optimizer minibatching.
    Raw native rewards remain unchanged; shaping affects estimator inputs only.
    """

    settings: GRPOSettings
    rows: NativeCreditRows
    required_relations: tuple[str, ...]

    @property
    def id(self) -> str:
        return f"{self.settings.algorithm}-scalar-credit@1:{record_digest(self.settings)}"

    def prepare(self, snapshot: PopulationSnapshot) -> PreparedCredit:
        if self.settings.algorithm not in {"grpo", "dapo"}:
            raise InvalidPolicyUpdate("scalar group credit requires the supported GRPO/DAPO recipe")
        relations = {relation.id: relation for relation in snapshot.relations if relation.kind == "prompt-group"}
        if set(self.required_relations) != set(relations):
            raise InvalidPolicyUpdate("scalar credit requires every declared complete prompt group")
        groups: dict[str, list[int]] = {}
        assigned: dict[ActionRef, str] = {}
        for identity, relation in relations.items():
            if relation.completeness != "complete":
                raise InvalidPolicyUpdate("scalar credit requires complete prompt groups")
            for action in relation.members:
                if action in assigned:
                    raise InvalidPolicyUpdate("scalar credit prompt groups overlap")
                assigned[action] = identity
        for index, actions in enumerate(self.rows.actions):
            memberships = {assigned.get(action) for action in actions if action is not None}
            if len(memberships) != 1 or None in memberships:
                raise InvalidPolicyUpdate("scalar credit row must belong to exactly one declared prompt group")
            identity = next(iter(memberships))
            assert identity is not None
            groups.setdefault(identity, []).append(index)
        if set(groups) != set(relations) or any(
            len(group) != self.settings.num_generations for group in groups.values()
        ):
            raise InvalidPolicyUpdate("scalar credit requires exactly num_generations episodes per prompt group")
        rewards = [
            shape_online_reward(
                self.settings, rollout.reward, len(rollout.completion_ids), is_truncated=rollout.is_truncated
            )
            for rollout in self.rows.rollouts
        ]
        if not all(math.isfinite(value) for value in rewards):
            raise InvalidPolicyUpdate("scalar credit requires finite admitted native rewards")
        batch_std = statistics.stdev(rewards) if self.settings.advantage_scaling == "batch" else None
        advantages = [0.0] * len(rewards)
        for group in groups.values():
            values = [rewards[index] for index in group]
            mean = math.fsum(values) / len(values)
            divisor = (
                1.0
                if self.settings.advantage_scaling == "none"
                else ((batch_std if batch_std is not None else statistics.stdev(values)) + 1e-4)
            )
            for index in group:
                advantages[index] = (rewards[index] - mean) / divisor
        tokens = tuple(
            tuple(advantage if action is not None else 0.0 for action in actions)
            for advantage, actions in zip(advantages, self.rows.actions, strict=True)
        )
        return PreparedCredit(
            snapshot.digest,
            self.id,
            self.rows.project(tokens),
            self.required_relations,
            (),
            f"grpo-trl-{self.settings.advantage_scaling}-sample-std-eps1e-4@1",
            "full-trajectory",
            self.rows.evidence_digests,
        )


@dataclass(frozen=True, slots=True)
class SampoCreditEstimator:
    """Prepare native raw rollout rewards using the existing SAMPO recipe.

    NativeCreditRows retain original rewards. Shaping is applied only to copies
    passed to the estimator, matching the production TRL collection boundary.
    Callers must not pre-shape these rows.
    """

    settings: SAMPOSettings
    rows: NativeCreditRows
    required_relations: tuple[str, ...]

    @property
    def id(self) -> str:
        return f"sampo-credit@2:{record_digest(self.settings)}"

    def prepare(self, snapshot: PopulationSnapshot) -> PreparedCredit:
        shaped_rollouts = tuple(
            replace(
                rollout,
                reward=shape_online_reward(
                    self.settings,
                    rollout.reward,
                    len(rollout.completion_ids),
                    is_truncated=rollout.is_truncated,
                ),
            )
            for rollout in self.rows.rollouts
        )
        result = compute_sampo_advantages(
            self.settings, tuple(rollout.example_id for rollout in shaped_rollouts), shaped_rollouts
        )
        return PreparedCredit(
            snapshot.digest,
            self.id,
            self.rows.project(result.token_advantages),
            self.required_relations,
            (("episode", 1.0), ("turn", self.settings.step_advantage_weight)),
            f"sampo-{self.settings.advantage_normalization}@1",
            "full-trajectory",
            self.rows.evidence_digests,
        )


@dataclass(frozen=True, slots=True)
class StructuredCreditEstimator:
    settings: GDPOSettings | CAPOSettings
    rows: NativeCreditRows
    required_relations: tuple[str, ...]

    @property
    def id(self) -> str:
        algorithm = "gdpo" if isinstance(self.settings, GDPOSettings) else "capo"
        return f"{algorithm}-credit@1:{record_digest(self.settings)}"

    def prepare(self, snapshot: PopulationSnapshot) -> PreparedCredit:
        evidence = []
        for rollout in self.rows.rollouts:
            if rollout.reward_evidence is None:
                raise InvalidPolicyUpdate("structured estimator requires every native reward evidence record")
            evidence.append(rollout.reward_evidence)
        masks = tuple(rollout.env_mask for rollout in self.rows.rollouts)
        if isinstance(self.settings, GDPOSettings):
            result = compute_gdpo_advantages(
                evidence,
                masks,
                component_names=self.settings.component_names,
                component_weights=self.settings.component_weights,
                group_size=self.settings.num_generations,
                epsilon=self.settings.epsilon,
            )
            weights = tuple(zip(self.settings.component_names, self.settings.component_weights, strict=True))
        else:
            result = compute_capo_advantages(
                evidence,
                masks,
                group_size=self.settings.num_generations,
                outcome_component=self.settings.outcome_component,
                outcome_weight=self.settings.outcome_weight,
                process_weight=self.settings.process_weight,
                epsilon=self.settings.epsilon,
            )
            weights = (("outcome", self.settings.outcome_weight), ("process", self.settings.process_weight))
        return PreparedCredit(
            snapshot.digest,
            self.id,
            self.rows.project(result.token_advantages),
            self.required_relations,
            weights,
            self.settings.numerical_profile,
            "full-trajectory",
            self.rows.evidence_digests,
        )
