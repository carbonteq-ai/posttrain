"""Normalize immutable execution-target snapshots without reopening a catalog."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from posttrain.common import JsonValue

from .models import ExecutionTargetContext

_GIB = 1024**3


def _mapping(value: object) -> Mapping[str, object] | None:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else None


def _positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def _positive_float(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _context(value: Mapping[str, object], *, fallback_role: str | None = None) -> ExecutionTargetContext | None:
    selection_id = value.get("selection_id")
    if not isinstance(selection_id, str) or not selection_id:
        selection_id = value.get("target_id")
    if not isinstance(selection_id, str) or not selection_id:
        return None

    raw_roles = value.get("roles")
    roles = (
        tuple(sorted({role for role in raw_roles if isinstance(role, str) and role}))
        if isinstance(raw_roles, list | tuple)
        else ()
    )
    if fallback_role is not None and fallback_role not in roles:
        roles = tuple(sorted((*roles, fallback_role)))

    placement_value = _mapping(value.get("placement"))
    placement = dict(placement_value) if placement_value is not None else {}
    host_value = _mapping(value.get("host_constraints"))
    host_constraints = dict(host_value) if host_value is not None else {}
    memory_gb = _positive_float(value.get("memory_gb"))
    device_count = _positive_int(placement.get("world_size"))
    if device_count is None and memory_gb is not None:
        device_count = 1
    memory_bytes = memory_gb * _GIB if memory_gb is not None else None
    aggregate = memory_bytes * device_count if memory_bytes is not None and device_count else None
    revision = value.get("revision")
    device_class = value.get("device_class")
    return ExecutionTargetContext(
        selection_id=selection_id,
        revision=revision if isinstance(revision, str) and revision else None,
        roles=roles,
        device_class=device_class if isinstance(device_class, str) and device_class else None,
        device_count=device_count,
        memory_bytes_per_device=memory_bytes,
        aggregate_memory_bytes=aggregate,
        placement=cast(dict[str, JsonValue], placement),
        host_constraints=cast(dict[str, JsonValue], host_constraints),
        state=(
            "complete"
            if isinstance(revision, str)
            and revision
            and isinstance(device_class, str)
            and device_class
            and aggregate is not None
            else "partial"
        ),
    )


def execution_target_contexts(
    resolved_inputs: Mapping[str, JsonValue],
) -> tuple[ExecutionTargetContext, ...]:
    """Read the v1 snapshot and retain a conservative legacy-id fallback."""

    contexts: list[ExecutionTargetContext] = []
    envelope = _mapping(resolved_inputs.get("execution_targets"))
    raw_targets = envelope.get("targets") if envelope is not None else None
    if isinstance(raw_targets, list | tuple):
        for raw_target in raw_targets:
            target = _mapping(raw_target)
            if target is not None and (context := _context(target)) is not None:
                contexts.append(context)

    if not contexts:
        for role, raw_selection in resolved_inputs.items():
            selection = _mapping(raw_selection)
            if selection is None:
                continue
            resolved = _mapping(selection.get("resolved")) or selection
            target = _mapping(resolved.get("target"))
            if target is not None:
                context = _context(target, fallback_role=role)
            else:
                context = _context(
                    {
                        "target_id": resolved.get("target_id"),
                    },
                    fallback_role=role,
                )
            if context is not None:
                contexts.append(context)

    merged: dict[tuple[str, str | None], ExecutionTargetContext] = {}
    for context in contexts:
        identity = (context.selection_id, context.revision)
        previous = merged.get(identity)
        if previous is None:
            merged[identity] = context
            continue
        merged[identity] = context.model_copy(update={"roles": tuple(sorted(set(previous.roles) | set(context.roles)))})
    return tuple(merged.values())


def execution_target_capacity(
    contexts: tuple[ExecutionTargetContext, ...],
) -> tuple[Literal["available", "ambiguous", "unavailable"], float | None]:
    """Return a capacity only when every complete target agrees on the host aggregate."""

    capacities = {context.aggregate_memory_bytes for context in contexts if context.aggregate_memory_bytes is not None}
    if not capacities:
        return "unavailable", None
    if len(capacities) > 1:
        return "ambiguous", None
    return "available", next(iter(capacities))


__all__ = ["execution_target_capacity", "execution_target_contexts"]
