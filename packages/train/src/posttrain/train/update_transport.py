"""Typed reconstruction of resolved checkpoint records, without recomputing credit.

This decodes the native checkpoint sidecar. It does not authenticate a
checkpoint: callers must verify its recovery seal and component hashes first.
Only framework record types are instantiated, field by field.

Schema v4 stores the population's conditioning views (with their sampled token
indices), the prepared advantages as one list aligned with population
positions, the objective selection and each occurrence's contributions and
turns. The objective is recomputed from the frozen inputs and must reproduce
its retained digest. Sidecars from the per-token engine (v1 to v3) are not
readable by this engine.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .update_credit import PreparedCredit
from .update_objectives import ObjectiveSpec, objective_population
from .update_plan import ExecutionCapabilities, PolicyExecutionBudget, ResolvedUpdate, plan_packs
from .update_records import (
    ActionInterval,
    ActionRef,
    ActionSelection,
    ConditioningView,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationRelation,
    PopulationSnapshot,
    SemanticSpan,
)
from .update_resolution import ResolvedPolicyPopulation

SCHEMA = "posttrain.resolved-population.v4"


@dataclass(frozen=True, slots=True)
class RetainedResolvedPopulation:
    resolved: ResolvedPolicyPopulation
    max_overflow_retries: int
    applied_update_offset: int
    attempt_offset: int


def _snapshot_payload(snapshot: PopulationSnapshot) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "native_evidence_ref": snapshot.native_evidence_ref,
        "native_evidence_digest": snapshot.native_evidence_digest,
        "conditioning": [asdict(view) for view in snapshot.conditioning],
        "spans": [asdict(span) for span in snapshot.spans],
        "relations": [asdict(relation) for relation in snapshot.relations],
        "versions": asdict(snapshot.versions),
        "selector_digest": snapshot.selector_digest,
        "digest": snapshot.digest,
    }


def _credit_payload(credit: PreparedCredit) -> dict[str, Any]:
    return {
        "population_digest": credit.population_digest,
        "estimator_id": credit.estimator_id,
        "advantages": credit.advantages.tolist(),
        "required_relations": list(credit.required_relations),
        "component_weights": [list(item) for item in credit.component_weights],
        "normalization": credit.normalization,
        "observation_scope": credit.observation_scope,
        "evidence_digests": list(credit.evidence_digests),
        "meaning": credit.meaning,
        "digest": credit.digest,
    }


def population_payload(population: Any) -> dict[str, Any]:
    """Encode a resolved population's frozen inputs and occurrences."""
    updates = population.updates
    objective = updates[0].objective
    return {
        "schema": SCHEMA,
        "population": _snapshot_payload(updates[0].population),
        "credit": _credit_payload(population.credit),
        "spec": asdict(population.spec),
        "execution": asdict(population.execution),
        "capabilities": asdict(population.capabilities),
        "max_overflow_retries": getattr(population, "max_overflow_retries", 0),
        "applied_update_offset": getattr(population, "applied_update_offset", 0),
        "attempt_offset": getattr(population, "attempt_offset", 0),
        "objective_digest": objective.digest,
        "updates": [
            {
                "digest": update.digest,
                "schedule_digest": update.schedule_digest,
                "epoch": update.epoch,
                "minibatch": update.minibatch,
                "contributions": [objective.contributions[index].id for index in update.contributions],
                "occurrence_ids": list(update.occurrence_ids),
                "views": list(update.views),
                "discarded_contributions": list(update.discarded_contributions),
            }
            for update in updates
        ],
    }


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise InvalidPolicyUpdate("retained population value does not match its record schema")
    return tuple(value)


def _integers(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise InvalidPolicyUpdate("retained population value does not match its record schema")
    return tuple(value)


def _action(value: dict[str, Any]) -> ActionRef:
    return ActionRef(value["episode_id"], value["branch_id"], value["turn_id"], value["token_index"])


def _snapshot(data: dict[str, Any]) -> PopulationSnapshot:
    views = tuple(ConditioningView(**{**view, "sampled": _integers(view["sampled"])}) for view in data["conditioning"])
    spans = tuple(
        SemanticSpan(
            span["id"],
            span["role"],
            span["projection_revision"],
            tuple(ActionInterval(_action(item["start"]), item["end"]) for item in span["action_intervals"]),
        )
        for span in data["spans"]
    )
    relations = tuple(
        PopulationRelation(
            relation["id"],
            relation["kind"],
            _strings(relation["members"]),
            relation["completeness"],
            _strings(relation["expected_members"]),
        )
        for relation in data["relations"]
    )
    snapshot = PopulationSnapshot(
        data["id"],
        data["native_evidence_ref"],
        data["native_evidence_digest"],
        views,
        spans,
        relations,
        PolicyVersions(**data["versions"]),
        data["selector_digest"],
    )
    if snapshot.digest != data["digest"]:
        raise InvalidPolicyUpdate("retained population differs from its frozen digest")
    return snapshot


def _credit(data: dict[str, Any]) -> PreparedCredit:
    advantages = data["advantages"]
    if not isinstance(advantages, list) or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in advantages
    ):
        raise InvalidPolicyUpdate("retained advantages must be finite numbers")
    credit = PreparedCredit(
        data["population_digest"],
        data["estimator_id"],
        np.asarray(advantages, dtype=np.float64),
        _strings(data["required_relations"]),
        tuple((name, weight) for name, weight in data["component_weights"]),
        data["normalization"],
        data["observation_scope"],
        _strings(data["evidence_digests"]),
        data["meaning"],
    )
    if credit.digest != data["digest"]:
        raise InvalidPolicyUpdate("retained credit differs from its frozen digest")
    return credit


def _spec(data: dict[str, Any]) -> ObjectiveSpec:
    return ObjectiveSpec(
        **{
            **data,
            "policy_selection": _selection(data["policy_selection"]),
            "kl_selection": _selection(data["kl_selection"]),
        }
    )


def _selection(data: dict[str, Any]) -> ActionSelection:
    return ActionSelection(data["mode"], _strings(data["span_ids"]), _strings(data["roles"]))


def decode_population_payload(payload: Any) -> RetainedResolvedPopulation:
    """Restore frozen credit, selections and occurrences from a v4 sidecar.

    No reward estimator runs here. The objective and execution packs are
    recomputed and must reproduce every retained digest.
    """
    try:
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise InvalidPolicyUpdate("unsupported retained population schema")
        expected_fields = {
            "schema",
            "population",
            "credit",
            "spec",
            "execution",
            "capabilities",
            "max_overflow_retries",
            "applied_update_offset",
            "attempt_offset",
            "objective_digest",
            "updates",
        }
        if set(payload) != expected_fields:
            raise InvalidPolicyUpdate("retained population fields differ from the supported schema")
        snapshot = _snapshot(payload["population"])
        credit = _credit(payload["credit"])
        spec = _spec(payload["spec"])
        execution = PolicyExecutionBudget(**payload["execution"])
        capabilities = ExecutionCapabilities(
            **{
                **payload["capabilities"],
                "definition_ids": _strings(payload["capabilities"]["definition_ids"]),
                "statistics": _strings(payload["capabilities"]["statistics"]),
            }
        )
        objective = objective_population(snapshot, spec, credit)
        if objective.digest != payload["objective_digest"]:
            raise InvalidPolicyUpdate("retained objective differs from its frozen resolved contract")
        index_of = {contribution.id: index for index, contribution in enumerate(objective.contributions)}
        updates = []
        for item in payload["updates"]:
            contributions = tuple(index_of[identity] for identity in _strings(item["contributions"]))
            update = ResolvedUpdate(
                snapshot,
                objective,
                item["schedule_digest"],
                item["epoch"],
                item["minibatch"],
                contributions,
                _strings(item["occurrence_ids"]),
                _integers(item["views"]),
                _strings(item["discarded_contributions"]),
            )
            if update.digest != item["digest"] or any(
                type(index) is not int or index < 0 for index in (update.epoch, update.minibatch)
            ):
                raise InvalidPolicyUpdate("retained occurrence digest or cursor differs")
            updates.append(update)
        if not updates or len({update.digest for update in updates}) != len(updates):
            raise InvalidPolicyUpdate("retained population has no or duplicate occurrences")
        offsets = tuple(payload[name] for name in ("max_overflow_retries", "applied_update_offset", "attempt_offset"))
        if any(type(value) is not int or value < 0 for value in offsets) or offsets[2] < offsets[1]:
            raise InvalidPolicyUpdate("retained population offsets or retry limit are incoherent")
        packs = tuple(plan_packs(update, execution, capabilities) for update in updates)
        resolved = ResolvedPolicyPopulation(snapshot, credit, spec, tuple(updates), packs, execution, capabilities)
        return RetainedResolvedPopulation(resolved, *offsets)
    except (TypeError, ValueError, KeyError) as error:
        if isinstance(error, InvalidPolicyUpdate):
            raise
        raise InvalidPolicyUpdate("retained population cannot reconstruct a valid resolved contract") from error
