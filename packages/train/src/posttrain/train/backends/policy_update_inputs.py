"""Authenticate native evidence before exposing resolved model inputs.

Hosts own native trace storage and decoding. This reader never writes a
transcript or substitutes flattened tokens for the physical graph. Both native
backends consume the same callable input contract.

Admission verifies every view once: the retained bytes against the frozen
digest, and each view's conditioning record re-derived from the decoded graph
against the frozen view. The decoded graphs are private to the reader, so each
view's model input is built once from its verified record and reused for the
population's lifetime (old, reference and current scoring, replay, resume).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from posttrain.environment.verifiers_conditioning import (
    NativeConditioningInput,
    NativeConditioningRecord,
    conditioning_input,
    native_conditioning_records,
)

from ..update_records import ConditioningView, InvalidPolicyUpdate, PopulationSnapshot


@dataclass(frozen=True, slots=True)
class RetainedEvidence:
    """Retained native population bytes and their SHA-256, computed once.

    Bytes are immutable, so every later authenticity check compares this
    digest instead of hashing the (often gigabyte-sized) bytes again.
    """

    data: bytes = field(repr=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise InvalidPolicyUpdate("native artifact reader must return original retained bytes")
        object.__setattr__(self, "digest", hashlib.sha256(self.data).hexdigest())

    @classmethod
    def of(cls, evidence: bytes | RetainedEvidence) -> RetainedEvidence:
        return evidence if isinstance(evidence, RetainedEvidence) else cls(evidence)


@dataclass(frozen=True, slots=True)
class NativePopulationInputs:
    """Validated original contexts for one content-bound native population."""

    _views: Mapping[str, ConditioningView]
    _records: Mapping[str, NativeConditioningRecord]
    _traces: Mapping[str, Any]
    evidence: RetainedEvidence = field(repr=False)
    _sampled_scores: Mapping[str, tuple[Any, ...]] = field(default_factory=dict, repr=False)
    _inputs: Mapping[str, NativeConditioningInput] = field(default_factory=dict, repr=False)

    @classmethod
    def from_evidence(
        cls,
        snapshot: PopulationSnapshot,
        evidence: bytes | RetainedEvidence,
        decode: Callable[[bytes], Mapping[str, Any]],
    ) -> NativePopulationInputs:
        """Verify all views before any optimizer occurrence may be executed.

        The host decoder reads the native retained artifact format, including
        its exact graph representation. It must not reconstruct chat messages.
        Extra attempted/unadmitted traces may remain in the native artifact.
        """
        evidence = RetainedEvidence.of(evidence)
        if evidence.digest != snapshot.native_evidence_digest:
            raise InvalidPolicyUpdate("native population evidence bytes differ from the frozen digest")
        traces = dict(decode(evidence.data))
        coordinates_by_view: dict[str, tuple[str, int, tuple[int, ...]]] = {}
        nodes_by_trace: dict[str, list[int]] = {}
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
            coordinates_by_view[view.id] = (trace_id, node_index, tuple(prefix))
            nodes_by_trace.setdefault(trace_id, []).append(node_index)
        # One derivation per trace validates each graph node once for all its turns.
        derived: dict[tuple[str, int], NativeConditioningRecord] = {}
        for trace_id, node_indices in nodes_by_trace.items():
            for record in native_conditioning_records(
                traces[trace_id], sampled_node_indices=tuple(node_indices), context_contract="causal-text@1"
            ):
                derived[(trace_id, record.node_index)] = record
        views, records, inputs, sampled_scores = {}, {}, {}, {}
        for view in snapshot.conditioning:
            trace_id, node_index, prefix = coordinates_by_view[view.id]
            record = derived[(trace_id, node_index)]
            if (
                record.prefix_node_indices != prefix
                or record.input_digest != view.digest
                or record.context_tokens != view.context_tokens
                or record.sampled_token_indices != view.sampled
            ):
                raise InvalidPolicyUpdate("native evidence differs from frozen context or sampled action coordinates")
            trace = traces[trace_id]
            views[view.id], records[view.id] = view, record
            inputs[view.id] = conditioning_input(trace, record)
            logprobs = getattr(trace.nodes[node_index], "logprobs", None)
            if isinstance(logprobs, (list, tuple)):
                sampled_scores[view.id] = tuple(logprobs)
        return cls(
            MappingProxyType(views),
            MappingProxyType(records),
            MappingProxyType(traces),
            evidence,
            MappingProxyType(sampled_scores),
            MappingProxyType(inputs),
        )

    def sampling_log_scores(self, snapshot: PopulationSnapshot) -> np.ndarray:
        """Read compact sampled scores from authenticated original assistant nodes, per position.

        Input-only artifacts may omit sampled scores; correction then rejects
        rather than substituting unit weights. Serialization/context tokens are
        excluded by original sampled coordinates, never by reconstructed text.
        """
        if self.evidence.digest != snapshot.native_evidence_digest:
            raise InvalidPolicyUpdate("sampler scores belong to different native evidence")
        result = np.empty(snapshot.size, dtype=np.float64)
        for index, view in enumerate(snapshot.conditioning):
            if self._views.get(view.id) != view:
                raise InvalidPolicyUpdate("sampler action references different native evidence")
            record = self._records[view.id]
            values = self._sampled_scores.get(view.id)
            if (
                values is None
                or len(values) != len(record.sampled_token_indices)
                or any(type(value) not in (float, int) or not math.isfinite(value) for value in values)
            ):
                raise InvalidPolicyUpdate("native sampled log scores are missing, changed or misaligned")
            result[snapshot.view_positions(index)] = values
        result.setflags(write=False)
        return result

    @property
    def retained_evidence(self) -> bytes:
        return self.evidence.data

    def __call__(self, view: ConditioningView) -> NativeConditioningInput:
        if self._views.get(view.id) != view:
            raise InvalidPolicyUpdate("input reader received a view outside its frozen population")
        return self._inputs[view.id]
