"""Typed reconstruction of resolved checkpoint records, without recomputing credit.

This decodes the existing native checkpoint sidecar. It does not authenticate a
checkpoint: callers must verify its recovery seal and component hashes first.
Only framework record types supplied by this module are instantiated.
"""

from __future__ import annotations

import math
import types
from dataclasses import asdict, dataclass, fields, is_dataclass
from functools import lru_cache
from typing import Any, Literal, TypeAliasType, Union, get_args, get_origin, get_type_hints

from .update_credit import PreparedCredit
from .update_objectives import ObjectiveSpec, objective_population
from .update_plan import ExecutionCapabilities, ObjectivePopulation, PolicyExecutionBudget, ResolvedUpdate, plan_packs
from .update_records import InvalidPolicyUpdate, PopulationSnapshot, record_digest
from .update_resolution import ResolvedPolicyPopulation


@lru_cache
def _hints(record_type: type) -> dict[str, Any]:
    return get_type_hints(record_type)


def _decode(expected: Any, value: Any) -> Any:
    """Strict data-only decoding using trusted framework type annotations."""
    if isinstance(expected, TypeAliasType):
        return _decode(expected.__value__, value)
    origin, arguments = get_origin(expected), get_args(expected)
    if origin is Literal:
        if any(type(value) is type(option) and value == option for option in arguments):
            return value
    elif origin in (Union, types.UnionType):
        for option in arguments:
            try:
                return _decode(option, value)
            except InvalidPolicyUpdate:
                pass
    elif origin is tuple and isinstance(value, list):
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_decode(arguments[0], item) for item in value)
        if len(arguments) == len(value):
            return tuple(_decode(kind, item) for kind, item in zip(arguments, value, strict=True))
    elif isinstance(expected, type) and is_dataclass(expected):
        if isinstance(value, dict) and set(value) == {field.name for field in fields(expected)}:
            try:
                return expected(**{name: _decode(kind, value[name]) for name, kind in _hints(expected).items()})
            except (TypeError, ValueError) as error:
                raise InvalidPolicyUpdate(f"invalid retained {expected.__name__} record") from error
    elif expected is float:
        if type(value) in (int, float) and math.isfinite(value):
            # Preserve numeric representation: JSON integer coefficients are
            # valid inputs and their structural digest must not change.
            return value
    elif expected in (str, int, bool, type(None)) and type(value) is expected:
        return value
    raise InvalidPolicyUpdate("retained population value does not match its record schema")


@dataclass(frozen=True, slots=True)
class RetainedResolvedPopulation:
    resolved: ResolvedPolicyPopulation
    max_overflow_retries: int
    applied_update_offset: int
    attempt_offset: int


def population_payload(population: Any) -> dict[str, Any]:
    """Encode the existing sidecar schema without repeating population records."""
    updates, objectives = [], {}
    for update in population.updates:
        objectives[record_digest(update.objective)] = asdict(update.objective)
        updates.append(
            {
                "digest": update.digest,
                "objective_digest": record_digest(update.objective),
                "schedule_digest": update.schedule_digest,
                "epoch": update.epoch,
                "minibatch": update.minibatch,
                "contributions": [asdict(value) for value in update.contributions],
                "occurrence_ids": list(update.occurrence_ids),
                "dependencies": [asdict(value) for value in update.dependencies],
                "discarded_contributions": list(update.discarded_contributions),
            }
        )
    return {
        "schema": "posttrain.resolved-population.v3",
        "population": asdict(population.updates[0].population),
        "credit": asdict(population.credit),
        "spec": asdict(population.spec),
        "execution": asdict(population.execution),
        "capabilities": asdict(population.capabilities),
        "max_overflow_retries": getattr(population, "max_overflow_retries", 0),
        "applied_update_offset": getattr(population, "applied_update_offset", 0),
        "attempt_offset": getattr(population, "attempt_offset", 0),
        "objectives": objectives,
        "updates": updates,
    }


def decode_population_payload(payload: Any) -> RetainedResolvedPopulation:
    """Restore frozen credit, masks and occurrences from v1/v2/v3 sidecars.

    No reward estimator runs here. Recompute execution packs and validate the
    objective contracts, preserving all original occurrence and credit digests.
    Legacy v1 sidecars can describe only zero prior population offsets.
    Legacy v1/v2 capabilities retain their ragged cost contract for one release.
    """
    try:
        if not isinstance(payload, dict) or payload.get("schema") not in {
            "posttrain.resolved-population.v1",
            "posttrain.resolved-population.v2",
            "posttrain.resolved-population.v3",
        }:
            raise InvalidPolicyUpdate("unsupported retained population schema")
        data = dict(payload)
        if data["schema"] == "posttrain.resolved-population.v1":
            if data.get("applied_update_offset", 0) != 0 or data.get("attempt_offset", 0) != 0:
                raise InvalidPolicyUpdate("legacy retained population cannot declare nonzero offsets")
            data.update(applied_update_offset=0, attempt_offset=0)
        if set(data) != {
            "schema",
            "population",
            "credit",
            "spec",
            "execution",
            "capabilities",
            "max_overflow_retries",
            "applied_update_offset",
            "attempt_offset",
            "objectives",
            "updates",
        }:
            raise InvalidPolicyUpdate("retained population fields differ from the supported schema")
        snapshot = _decode(PopulationSnapshot, data["population"])
        credit = _decode(PreparedCredit, data["credit"])
        spec = _decode(ObjectiveSpec, data["spec"])
        execution = _decode(PolicyExecutionBudget, data["execution"])
        capability_data = data["capabilities"]
        if data["schema"] != "posttrain.resolved-population.v3":
            legacy_fields = {"definition_ids", "statistics", "max_context_tokens", "cross_pack_dependencies"}
            if not isinstance(capability_data, dict) or set(capability_data) != legacy_fields:
                raise InvalidPolicyUpdate("legacy population cannot declare a new physical context layout")
            capability_data = {**capability_data, "context_layout": "ragged"}
        capabilities = _decode(ExecutionCapabilities, capability_data)
        expected_objective = objective_population(snapshot, spec, credit)
        if not isinstance(data["objectives"], dict) or not isinstance(data["updates"], list):
            raise InvalidPolicyUpdate("retained objective and occurrence containers are invalid")
        objectives = {digest: _decode(ObjectivePopulation, value) for digest, value in data["objectives"].items()}
        if not objectives or any(
            digest != record_digest(value) or value != expected_objective for digest, value in objectives.items()
        ):
            raise InvalidPolicyUpdate("retained objective differs from its frozen resolved contract")
        updates = []
        used_objectives = set()
        for item in data["updates"]:
            values = dict(item)
            digest, objective_digest = values.pop("digest"), values.pop("objective_digest")
            used_objectives.add(objective_digest)
            objective = objectives[objective_digest]
            update = _decode(
                ResolvedUpdate,
                {**values, "population": data["population"], "objective": data["objectives"][objective_digest]},
            )
            if update.digest != digest or any(
                type(index) is not int or index < 0 for index in (update.epoch, update.minibatch)
            ):
                raise InvalidPolicyUpdate("retained occurrence digest or cursor differs")
            if any(contribution not in objective.contributions for contribution in update.contributions):
                raise InvalidPolicyUpdate("retained occurrence changes original contributions")
            updates.append(update)
        if used_objectives != set(objectives) or len({update.digest for update in updates}) != len(updates):
            raise InvalidPolicyUpdate("retained population has unused objectives or duplicate occurrences")
        offsets = tuple(
            _decode(int, data[name]) for name in ("max_overflow_retries", "applied_update_offset", "attempt_offset")
        )
        if any(value < 0 for value in offsets) or offsets[2] < offsets[1]:
            raise InvalidPolicyUpdate("retained population offsets or retry limit are incoherent")
        packs = tuple(plan_packs(update, execution, capabilities) for update in updates)
        resolved = ResolvedPolicyPopulation(snapshot, credit, spec, tuple(updates), packs, execution, capabilities)
        return RetainedResolvedPopulation(resolved, *offsets)
    except (TypeError, ValueError, KeyError) as error:
        raise InvalidPolicyUpdate("retained population cannot reconstruct a valid resolved contract") from error
