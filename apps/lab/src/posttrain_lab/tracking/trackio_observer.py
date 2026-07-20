"""Translate framework-neutral observations into Trackio evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import trackio
from posttrain.common import (
    EventObservation,
    LocalArtifactRef,
    MetricObservation,
    ProducedArtifact,
    TraceObservation,
)


class TrackioRun(Protocol):
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    def log_artifact(
        self,
        artifact_or_path: trackio.Artifact | str | Path,
        name: str | None = None,
        type: str | None = None,
        aliases: list[str] | None = None,
    ) -> trackio.Artifact: ...


def _json_dict(values: Mapping[str, object]) -> dict[str, object]:
    return dict(values)


def _artifact_name(logical_name: str) -> str:
    return logical_name.replace("/", "-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrackioObserver:
    """Trackio adapter; operations only depend on the Observer protocol."""

    def __init__(self, run: TrackioRun) -> None:
        self._run = run

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

    def trace(self, observation: TraceObservation) -> None:
        metadata = {
            "external_id": observation.external_id,
            **_json_dict(observation.attributes),
        }
        if observation.trace_type == "verifiers":
            trace: trackio.Trace = trackio.VerifiersTrace(
                record=dict(observation.payload),
                metadata=metadata,
            )
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
            name=_artifact_name(artifact.name),
            type=artifact.kind,
            metadata={"logical_name": artifact.name, **dict(artifact.metadata)},
        )
        if path.is_dir():
            logged.add_dir(path)
        else:
            logged.add_file(path)
        self._run.log_artifact(logged)
