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
        trace, sampled_node_indices=(record.node_index,), context_contract=record.context_contract,
    )[0]
    if actual != record:
        raise InvalidNativeConditioning("retained native conditioning differs from the frozen record")
    tokens = tuple(value for index in (*record.prefix_node_indices, record.node_index)
                   for value in trace.nodes[index].token_ids)
    prefix_length = len(tokens) - len(trace.nodes[record.node_index].token_ids)
    return NativeConditioningInput(record, tokens,
                                  tuple((local, prefix_length + local) for local in record.sampled_token_indices))


def native_conditioning_records(
    trace: Any, *, sampled_node_indices: tuple[int, ...], context_contract: str,
) -> tuple[NativeConditioningRecord, ...]:
    if context_contract != "causal-text@1":
        raise InvalidNativeConditioning("unqualified native conditioning contract")
    trace_id = getattr(trace, "id", None)
    nodes = tuple(trace.nodes)
    if not isinstance(trace_id, str) or not trace_id.strip():
        raise InvalidNativeConditioning("native context requires a retained trace identity")
    if len(set(sampled_node_indices)) != len(sampled_node_indices):
        raise InvalidNativeConditioning("sampled node selection duplicates a native call")
    calls = [call.node for call in trace.calls if call.node is not None]
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
        parent = node.parent
        while parent is not None:
            if type(parent) is not int or not 0 <= parent < len(nodes) or parent in path:
                raise InvalidNativeConditioning("native physical path has an invalid parent or cycle")
            path.append(parent)
            parent = nodes[parent].parent
        path.reverse()
        tokens: list[int] = []
        for index in path:
            source = nodes[index]
            ids, mask = tuple(source.token_ids), tuple(source.mask)
            if not ids or len(ids) != len(mask) or any(type(value) is not int or value < 0 for value in ids):
                raise InvalidNativeConditioning("native context requires aligned original text token IDs")
            if any(type(value) is not bool for value in mask):
                raise InvalidNativeConditioning("native sampled eligibility must be boolean")
            if getattr(source, "multi_modal_data", None) is not None:
                raise InvalidNativeConditioning("multimodal context is not qualified by causal-text@1")
            tokens.extend(ids)
        selected = tuple(index for index, eligible in enumerate(node.mask) if eligible)
        if not selected:
            raise InvalidNativeConditioning("sampled assistant call has no eligible original actions")
        prefix_tokens = len(tokens) - len(node.token_ids)
        if prefix_tokens + selected[0] == 0:
            raise InvalidNativeConditioning("sampled action lacks a preceding conditioning token")
        payload = {"tokens": tokens, "attention": "causal", "positions": "contiguous-zero-based",
                   "context_contract": context_contract}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result.append(NativeConditioningRecord(trace_id, node_index, tuple(path[:-1]), selected, len(tokens), digest,
                                               context_contract))
    return tuple(result)
