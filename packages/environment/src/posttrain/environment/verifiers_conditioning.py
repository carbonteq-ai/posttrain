"""Exact sampled-turn context references from native physical message graphs.

This projection reads original parent links, never a canonicalized transcript or
synthetic terminal training branch. It supports ordinary causal text attention;
custom positions, multimodal inputs and sparse attention need separate contracts.
No trajectory/token store is introduced: output retains coordinates and digests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class InvalidNativeConditioning(ValueError):
    """Native evidence cannot prove the requested action conditioning."""


@dataclass(frozen=True, slots=True)
class NativeConditioningRecord:
    trace_id: str
    node_index: int
    prefix_node_indices: tuple[int, ...]
    sampled_token_indices: tuple[int, ...]
    context_tokens: int
    input_digest: str
    context_contract: str


@dataclass(frozen=True, slots=True)
class NativeConditioningInput:
    """Ephemeral native input for one score pass, never an alternate trace store."""

    record: NativeConditioningRecord
    token_ids: tuple[int, ...]
    # Original sampled-node coordinates paired with physical input coordinates.
    action_positions: tuple[tuple[int, int], ...]


def materialize_native_conditioning(trace: Any, record: NativeConditioningRecord) -> NativeConditioningInput:
    """Revalidate retained graph bytes before exposing model input coordinates."""
    actual = native_conditioning_records(
        trace,
        sampled_node_indices=(record.node_index,),
        context_contract=record.context_contract,
    )[0]
    if actual != record:
        raise InvalidNativeConditioning("retained native conditioning differs from the frozen record")
    return conditioning_input(trace, record)


def conditioning_input(trace: Any, record: NativeConditioningRecord) -> NativeConditioningInput:
    """Model input for a record already derived from this same trace.

    Callers that derived ``record`` with ``native_conditioning_records`` from
    ``trace`` (and hold the trace privately) use this to skip a second
    derivation; any other caller uses ``materialize_native_conditioning``.
    """
    nodes = trace.nodes
    tokens: list[int] = []
    for index in (*record.prefix_node_indices, record.node_index):
        tokens.extend(nodes[index].token_ids)
    prefix_length = len(tokens) - len(nodes[record.node_index].token_ids)
    return NativeConditioningInput(
        record, tuple(tokens), tuple((local, prefix_length + local) for local in record.sampled_token_indices)
    )


def native_conditioning_records(
    trace: Any,
    *,
    sampled_node_indices: tuple[int, ...],
    context_contract: str,
) -> tuple[NativeConditioningRecord, ...]:
    """Derive each sampled node's exact causal context from the physical graph.

    Every node on a path is validated once per call, so passing all of a
    trace's sampled nodes together costs one pass over the graph, not one per
    turn of history.
    """
    if context_contract != "causal-text@1":
        raise InvalidNativeConditioning("unqualified native conditioning contract")
    trace_id = getattr(trace, "id", None)
    nodes = tuple(trace.nodes)
    if not isinstance(trace_id, str) or not trace_id.strip():
        raise InvalidNativeConditioning("native context requires a retained trace identity")
    if len(set(sampled_node_indices)) != len(sampled_node_indices):
        raise InvalidNativeConditioning("sampled node selection duplicates a native call")
    calls = [call.node for call in trace.calls if call.node is not None]
    checked: dict[int, tuple[int, ...]] = {}

    def node_tokens(index: int) -> tuple[int, ...]:
        ids = checked.get(index)
        if ids is None:
            source = nodes[index]
            ids, mask = tuple(source.token_ids), tuple(source.mask)
            if not ids or len(ids) != len(mask) or any(type(value) is not int or value < 0 for value in ids):
                raise InvalidNativeConditioning("native context requires aligned original text token IDs")
            if any(type(value) is not bool for value in mask):
                raise InvalidNativeConditioning("native sampled eligibility must be boolean")
            if getattr(source, "multi_modal_data", None) is not None:
                raise InvalidNativeConditioning("multimodal context is not qualified by causal-text@1")
            checked[index] = ids
        return ids

    result: list[NativeConditioningRecord] = []
    for node_index in sampled_node_indices:
        if type(node_index) is not int or not 0 <= node_index < len(nodes):
            raise InvalidNativeConditioning("sampled node index is outside the retained graph")
        node = nodes[node_index]
        message = node.message
        role = message.get("role") if isinstance(message, Mapping) else message.role
        if node.sampled is not True or role != "assistant" or calls.count(node_index) != 1:
            raise InvalidNativeConditioning("conditioning requires one original sampled assistant call")
        path = [node_index]
        on_path = {node_index}
        parent = node.parent
        while parent is not None:
            if type(parent) is not int or not 0 <= parent < len(nodes) or parent in on_path:
                raise InvalidNativeConditioning("native physical path has an invalid parent or cycle")
            path.append(parent)
            on_path.add(parent)
            parent = nodes[parent].parent
        path.reverse()
        tokens: list[int] = []
        for index in path:
            tokens.extend(node_tokens(index))
        selected = tuple(index for index, eligible in enumerate(node.mask) if eligible)
        if not selected:
            raise InvalidNativeConditioning("sampled assistant call has no eligible original actions")
        prefix_tokens = len(tokens) - len(node.token_ids)
        if prefix_tokens + selected[0] == 0:
            raise InvalidNativeConditioning("sampled action lacks a preceding conditioning token")
        payload = {
            "tokens": tokens,
            "attention": "causal",
            "positions": "contiguous-zero-based",
            "context_contract": context_contract,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result.append(
            NativeConditioningRecord(
                trace_id, node_index, tuple(path[:-1]), selected, len(tokens), digest, context_contract
            )
        )
    return tuple(result)


REASONING_PREFIX_REVISION = "verifiers.renderer-reasoning-prefix@1"


def native_reasoning_partition(
    trace: Any,
    record: NativeConditioningRecord,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split one sampled assistant call's original actions into reasoning and answer.

    The renderer that parsed the completion reports ``usage.reasoning_tokens``,
    counting generated thinking markers; thinking is the leading run of sampled
    tokens, and a reply cut off inside its thought is entirely reasoning. The
    call's ``completion_tokens`` must equal the record's sampled actions, so the
    split addresses exactly the original eligible coordinates.
    """
    actual = native_conditioning_records(
        trace,
        sampled_node_indices=(record.node_index,),
        context_contract=record.context_contract,
    )[0]
    if actual != record:
        raise InvalidNativeConditioning("reasoning extraction requires the retained conditioning record")
    calls = [call for call in trace.calls if call.node == record.node_index]
    usage = calls[0].usage if len(calls) == 1 else None
    reasoning = getattr(usage, "reasoning_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if type(reasoning) is not int or type(completion) is not int or reasoning < 0:
        raise InvalidNativeConditioning("sampled call lacks renderer reasoning-token accounting")
    sampled = record.sampled_token_indices
    if completion != len(sampled) or reasoning > completion:
        raise InvalidNativeConditioning("renderer reasoning accounting disagrees with sampled actions")
    return sampled[:reasoning], sampled[reasoning:]
