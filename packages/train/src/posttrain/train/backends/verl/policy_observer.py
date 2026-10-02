"""Credential-free observation transport from the single native actor to its host."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from posttrain.common import (
    AppendOnlyJsonlTailer,
    EventObservation,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    RunContext,
    TraceFactUpdateObservation,
    TraceObservation,
)
from pydantic import TypeAdapter

from ...update_records import InvalidPolicyUpdate
from .contracts import VerlRunContext

JOURNAL_NAME = "posttrain-resolved-observations.jsonl"
_TYPES = {"event": EventObservation, "metric": MetricObservation, "metrics": MetricBatchObservation,
          "trace": TraceObservation, "trace_fact_update": TraceFactUpdateObservation, "artifact": ProducedArtifact}


class ResolvedWorkerObserver:
    """Serialized single-rank actor/driver writes, durable before returning.

    The shared local workspace is required by the current single-host contract.
    This journal transports observations; native rollout artifacts remain replay
    authority and the host observer retains ownership of tracking publication.
    """

    def __init__(self, path: Path, context: VerlRunContext):
        self.path, self.context = path, context

    def _write(self, kind: str, observation: Any) -> None:
        if kind == "artifact" and not isinstance(observation.reference, LocalArtifactRef):
            raise InvalidPolicyUpdate("resolved worker artifact transport requires a local retained artifact")
        encoded = TypeAdapter(_TYPES[kind]).dump_json(observation, fallback=dict, warnings=False)
        payload = {"id": str(uuid4()), "schema": "posttrain.resolved-worker-observation@1", "kind": kind,
                   "context": self.context.model_dump(mode="json"), "value": json.loads(encoded)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as stream:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def event(self, observation: EventObservation) -> None:
        self._write("event", observation)

    def metric(self, observation: MetricObservation) -> None:
        self._write("metric", observation)

    def metrics(self, observation: MetricBatchObservation) -> None:
        self._write("metrics", observation)

    def trace(self, observation: TraceObservation) -> None:
        self._write("trace", observation)

    def trace_fact_update(self, observation: TraceFactUpdateObservation) -> None:
        self._write("trace_fact_update", observation)

    def artifact(self, artifact: ProducedArtifact) -> None:
        self._write("artifact", artifact)


def observation_tailer(context: RunContext, path: Path) -> AppendOnlyJsonlTailer:
    def forward(record: Mapping[str, Any]) -> None:
        if (set(record) != {"id", "schema", "kind", "context", "value"}
                or record["schema"] != "posttrain.resolved-worker-observation@1"
                or record["kind"] not in _TYPES
                or record["context"] != VerlRunContext.model_validate({**context.identity_attributes,
                                                                       "workspace": context.workspace}).model_dump(mode="json")):
            raise InvalidPolicyUpdate("worker observation differs from the actual host run")
        kind = record["kind"]
        value = TypeAdapter(_TYPES[kind]).validate_json(json.dumps(record["value"]))
        getattr(context.observer, kind)(value)

    return AppendOnlyJsonlTailer(path, forward)
