"""Authenticate native evidence before exposing resolved model inputs.

Hosts own native trace storage and decoding. This reader is ephemeral: it never
writes a transcript or substitutes flattened tokens for the physical graph.
Both native backends consume the same callable input contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from posttrain.environment.verifiers_conditioning import (
    NativeConditioningInput,
    NativeConditioningRecord,
    materialize_native_conditioning,
    native_conditioning_records,
)

from ..update_records import ActionRef, ConditioningView, InvalidPolicyUpdate, PopulationSnapshot


@dataclass(frozen=True, slots=True)
class NativePopulationInputs:
    """Validated original contexts for one content-bound native population."""

    _views: Mapping[str, ConditioningView]
    _records: Mapping[str, NativeConditioningRecord]
    _traces: Mapping[str, Any]
    retained_evidence: bytes = field(repr=False)
    _sampled_scores: Mapping[str, tuple[Any, ...]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_evidence(
        cls,
        snapshot: PopulationSnapshot,
        evidence: bytes,
        decode: Callable[[bytes], Mapping[str, Any]],
    ) -> NativePopulationInputs:
        """Verify all views before any optimizer occurrence may be executed.

        The host decoder reads the native retained artifact format, including
        its exact graph representation. It must not reconstruct chat messages.
        Extra attempted/unadmitted traces may remain in the native artifact.
        """
        if not isinstance(evidence, bytes) or hashlib.sha256(evidence).hexdigest() != snapshot.native_evidence_digest:
            raise InvalidPolicyUpdate("native population evidence bytes differ from the frozen digest")
        if any(action.native_ref != snapshot.native_evidence_ref for action in snapshot.actions):
            raise InvalidPolicyUpdate("native action references different population evidence")
        traces = dict(decode(evidence))
        sampled_by_view: dict[str, list[int]] = {}
        for action in snapshot.actions:
            sampled_by_view.setdefault(action.conditioning_id, []).append(action.action.token_index)
        views, records, sampled_scores = {}, {}, {}
        for view in snapshot.conditioning:
            if view.native_ref != snapshot.native_evidence_ref:
                raise InvalidPolicyUpdate("native view references different population evidence")
            try:
                coordinates = json.loads(view.token_ids_ref)
            except (TypeError, ValueError) as error:
                raise InvalidPolicyUpdate("native view coordinates must be retained graph references") from error
            if not isinstance(coordinates, dict) or set(coordinates) != {"trace_id", "prefix_nodes", "node_index"}:
                raise InvalidPolicyUpdate("native view coordinates have an unsupported schema")
            trace_id, node_index, prefix = (coordinates[key] for key in ("trace_id", "node_index", "prefix_nodes"))
            if (
                not isinstance(trace_id, str)
                or type(node_index) is not int
                or not isinstance(prefix, list)
                or any(type(index) is not int for index in prefix)
            ):
                raise InvalidPolicyUpdate("native view coordinates require exact graph identities and indices")
            trace = traces.get(trace_id)
            if trace is None or getattr(trace, "id", None) != trace_id:
                raise InvalidPolicyUpdate("native evidence lacks the original trace identity")
            if view.attention_ref != "causal-text@1/attention" or view.positions_ref != "causal-text@1/positions":
                raise InvalidPolicyUpdate("native input reader requires the qualified causal text context contract")
            record = native_conditioning_records(
                trace,
                sampled_node_indices=(node_index,),
                context_contract="causal-text@1",
            )[0]
            sampled = tuple(sorted(sampled_by_view.get(view.id, ())))
            if (
                record.prefix_node_indices != tuple(prefix)
                or record.input_digest != view.digest
                or record.context_tokens != view.context_tokens
                or record.sampled_token_indices != sampled
            ):
                raise InvalidPolicyUpdate("native evidence differs from frozen context or sampled action coordinates")
            views[view.id], records[view.id] = view, record
            logprobs = getattr(trace.nodes[node_index], "logprobs", None)
            if isinstance(logprobs, (list, tuple)):
                sampled_scores[view.id] = tuple(logprobs)
        return cls(
            MappingProxyType(views),
            MappingProxyType(records),
            MappingProxyType(traces),
            evidence,
            MappingProxyType(sampled_scores),
        )

    def sampling_log_scores(self, snapshot: PopulationSnapshot) -> Mapping[ActionRef, float]:
        """Read compact sampled scores from authenticated original assistant nodes.

        Input-only artifacts may omit sampled scores; correction then rejects
        rather than substituting unit weights. Serialization/context tokens are
        excluded by original sampled coordinates, never by reconstructed text.
        """
        if hashlib.sha256(self.retained_evidence).hexdigest() != snapshot.native_evidence_digest:
            raise InvalidPolicyUpdate("sampler scores belong to different native evidence")
        scores_by_view = {}
        for view in snapshot.conditioning:
            inputs = self(view)
            record = inputs.record
            values = self._sampled_scores.get(view.id)
            node = self._traces[record.trace_id].nodes[record.node_index]
            live = getattr(node, "logprobs", None)
            if (
                values is None
                or not isinstance(live, (list, tuple))
                or tuple(live) != values
                or len(values) != len(record.sampled_token_indices)
                or any(type(value) not in (float, int) or not math.isfinite(value) for value in values)
            ):
                raise InvalidPolicyUpdate("native sampled log scores are missing, changed or misaligned")
            scores_by_view[view.id] = dict(zip(record.sampled_token_indices, values, strict=True))
        result = {}
        for action in snapshot.actions:
            if action.native_ref != snapshot.native_evidence_ref:
                raise InvalidPolicyUpdate("sampler action references different native evidence")
            value = scores_by_view.get(action.conditioning_id, {}).get(action.action.token_index)
            if value is None:
                raise InvalidPolicyUpdate("native sampler score lacks its original sampled coordinate")
            result[action.action] = float(value)
        return MappingProxyType(result)

    def __call__(self, view: ConditioningView) -> NativeConditioningInput:
        if self._views.get(view.id) != view:
            raise InvalidPolicyUpdate("input reader received a view outside its frozen population")
        record = self._records[view.id]
        # Revalidate graph content on each read: mutable native objects must not
        # silently change between scoring, backward replay and checkpoint resume.
        return materialize_native_conditioning(self._traces[record.trace_id], record)
