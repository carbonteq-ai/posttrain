"""Translate framework-neutral observations into Trackio evidence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import trackio
from posttrain.common import (
    EventObservation,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    TraceFactSet,
    TraceFactUpdateObservation,
    TraceObservation,
)

if TYPE_CHECKING:
    from posttrain.tracking import ArtifactInput


class TrackioRun(Protocol):
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    def use_artifact(
        self,
        artifact_or_name: trackio.Artifact | str,
        type: str | None = None,
    ) -> trackio.Artifact: ...

    def log_artifact(
        self,
        artifact_or_path: trackio.Artifact | str | Path,
        name: str | None = None,
        type: str | None = None,
        aliases: list[str] | None = None,
    ) -> trackio.Artifact: ...


def _json_dict(values: Mapping[str, object]) -> dict[str, object]:
    return dict(values)


def _trace_fact_update(trace_type: str, external_id: str, facts: TraceFactSet) -> Any:
    update_type = getattr(trackio, "TraceFactUpdate", None)
    component_type = getattr(trackio, "TraceRewardComponent", None)
    if update_type is None or component_type is None:
        raise RuntimeError("configured Trackio does not support trace facts")
    return update_type(
        trace_type=trace_type,
        external_id=external_id,
        namespace=facts.namespace,
        calculator_version=facts.calculator_version,
        projection_id=facts.projection_id,
        dimensions=dict(facts.dimensions),
        measures=dict(facts.measures),
        reward_components=tuple(
            component_type(
                name=item.name,
                contribution=item.contribution,
                score=item.score,
                weight=item.weight,
                source_kind=item.source.kind,
                source_id=item.source.id,
            )
            for item in facts.reward_components
        ),
        provenance=dict(facts.provenance),
        state=facts.state,
        replace_reward_components=True,
    )


def trackio_artifact_name(logical_name: str) -> str:
    """Translate a logical artifact path into Trackio's portable name alphabet."""

    return re.sub(r"[^A-Za-z0-9._-]+", "-", logical_name).strip("-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class TrackioObserver:
    """Trackio adapter; operations only depend on the Observer protocol."""

    def __init__(self, run: TrackioRun) -> None:
        self._run = run

    def materialize_inputs(
        self,
        inputs: Mapping[str, ArtifactInput],
        root: Path,
        *,
        project: str,
    ) -> Mapping[str, LocalArtifactRef]:
        materialized: dict[str, LocalArtifactRef] = {}
        for logical_name, value in inputs.items():
            reference = value.reference
            if reference.provider != "trackio":
                raise ValueError(f"Trackio cannot materialize {reference.provider!r} artifacts")
            if reference.namespace != project:
                raise ValueError("cross-project Trackio artifact materialization is not supported")
            version = reference.version if reference.version.startswith("v") else f"v{reference.version}"
            artifact = self._run.use_artifact(f"{reference.name}:{version}", type=value.kind)
            destination = root / logical_name
            destination.mkdir(parents=True, exist_ok=False)
            path = Path(artifact.download(root=destination)).resolve()
            materialized[logical_name] = LocalArtifactRef(path, _tree_sha256(path))
        return materialized

    def event(self, observation: EventObservation) -> None:
        self._run.log(
            {
                "event/name": observation.name,
                "event/occurred_at": observation.occurred_at.isoformat(),
                "event/attributes": _json_dict(observation.attributes),
            }
        )

    def metric(self, observation: MetricObservation) -> None:
        values: dict[str, Any] = {observation.name: observation.value}
        if observation.attributes:
            values[f"{observation.name}/attributes"] = _json_dict(observation.attributes)
        self._run.log(values, step=observation.step)

    def metrics(self, observation: MetricBatchObservation) -> None:
        values: dict[str, Any] = dict(observation.values)
        if observation.attributes:
            values["metric/attributes"] = _json_dict(observation.attributes)
        self._run.log(values, step=observation.step)

    def trace(self, observation: TraceObservation) -> None:
        metadata = {
            "external_id": observation.external_id,
            "observation_type": observation.trace_type,
            **_json_dict(observation.attributes),
        }
        if observation.trace_type == "verifiers":
            if len(observation.facts) > 1:
                raise ValueError("a Verifiers trace may carry one complete source fact projection")
            trace_arguments: dict[str, Any] = {"record": dict(observation.payload), "metadata": metadata}
            if observation.facts:
                trace_arguments["trace_facts"] = _trace_fact_update(
                    observation.trace_type, observation.external_id, observation.facts[0]
                )
            trace = trackio.VerifiersTrace(**trace_arguments)
        else:
            messages = observation.payload.get("messages")
            if not isinstance(messages, list):
                raise ValueError("generic Trackio traces require a JSON `messages` list")
            display_messages: list[dict[str, Any]] = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("generic Trackio trace messages must be JSON objects")
                display_messages.append(dict(message))
            native = {key: value for key, value in observation.payload.items() if key != "messages"}
            trace = trackio.Trace(
                messages=display_messages,
                metadata={**native, **metadata},
            )
        self._run.log({f"traces/{observation.trace_type}": trace})

    def trace_fact_update(self, observation: TraceFactUpdateObservation) -> None:
        flush = getattr(self._run, "flush", None)
        upsert = getattr(self._run, "upsert_trace_facts", None)
        if not callable(flush) or not callable(upsert):
            raise RuntimeError("configured Trackio does not support trace-fact enrichment")
        flush()
        upsert(_trace_fact_update(observation.trace_type, observation.external_id, observation.facts))

    def artifact(self, artifact: ProducedArtifact) -> None:
        reference = artifact.reference
        if not isinstance(reference, LocalArtifactRef):
            raise TypeError("produced artifacts must be local outputs before Trackio promotion")
        path = reference.path
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            expected = reference.digest.removeprefix("sha256:")
            if _sha256(path) != expected:
                raise ValueError(f"artifact digest does not match file contents: {path}")

        logged = trackio.Artifact(
            name=trackio_artifact_name(artifact.name),
            type=artifact.kind,
            metadata={"logical_name": artifact.name, **dict(artifact.metadata)},
        )
        if path.is_dir():
            logged.add_dir(path)
        else:
            logged.add_file(path)
        self._run.log_artifact(logged)
