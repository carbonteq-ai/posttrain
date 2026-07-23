"""Trackio writer and normalized reader implementations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import trackio
from posttrain.common import (
    ContractError,
    EventObservation,
    JsonValue,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    TraceObservation,
)
from posttrain.tracking import (
    ArtifactInput,
    ArtifactLink,
    ArtifactSet,
    EventRecord,
    MetricPoint,
    MetricSeries,
    RunDetail,
    RunOutcome,
    RunQuery,
    RunSpec,
    RunSummary,
    SafeRunError,
    StoredArtifact,
    TracePage,
    TraceQuery,
    TraceRecord,
    TrackingCapabilities,
)
from trackio.run import Run as TrackioSDKRun

_RESERVED_HISTORY_KEYS = {"step", "timestamp"}
_SYSTEM_METRICS: dict[str, tuple[str, float]] = {
    "system/gpu_utilization": ("gpu/mean_utilization", 1.0),
    "system/gpu_vram_used_bytes": ("gpu/total_allocated_memory", 1024**3),
    "system/cpu_percent": ("cpu/utilization", 1.0),
}


@dataclass(frozen=True, slots=True)
class TrackioSettings:
    """Physical Trackio destination selected by the composition root."""

    project: str | None = None
    server_url: str | None = None
    auto_log_gpu: bool = False
    auto_log_cpu: bool = False


def _artifact_name(logical_name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", logical_name).strip("-")
    if not name:
        raise ContractError("artifact logical name cannot normalize to an empty Trackio name")
    return name


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    if path.is_file():
        return _file_sha256(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ContractError(f"Trackio {field} must be an ISO timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ContractError(f"Trackio {field} must be timezone-aware")
    return parsed


def _json_mapping(value: object) -> dict[str, JsonValue]:
    return dict(value) if isinstance(value, dict) else {}


def _run_config(spec: RunSpec, started_at: datetime) -> dict[str, JsonValue]:
    return {
        "schema_version": 4,
        "provider": "trackio",
        "project_id": spec.project_id,
        "work_package_id": spec.work_package_id,
        "stage": spec.stage,
        "run_id": spec.run_id,
        "job_kind": spec.job_kind,
        "job_definition_version": spec.job_definition_version,
        "started_at": started_at.isoformat(),
        "resolved_selections": dict(spec.resolved_inputs),
        "source_metadata": dict(spec.source_metadata),
        "input_artifacts": {
            name: {
                "provider": value.reference.provider,
                "namespace": value.reference.namespace,
                "name": value.reference.name,
                "version": value.reference.version,
                "kind": value.kind,
            }
            for name, value in spec.artifacts.items()
        },
    }


class TrackioTrackedRun:
    """One open Trackio run with retry-safe canonical finalization."""

    def __init__(self, run: TrackioSDKRun, project: str, spec: RunSpec) -> None:
        self._run = run
        self._project = project
        self._spec = spec
        self._outcome: RunOutcome | None = None
        self._last_metric_step: int | None = None

    @property
    def run_id(self) -> str:
        return self._spec.run_id

    @property
    def provider_run_id(self) -> str:
        return self._run.id

    def _validate_step(self, step: int | None) -> None:
        if step is None:
            return
        if step < 0:
            raise ContractError("logical metric steps must be non-negative")
        if self._last_metric_step is not None and step < self._last_metric_step:
            raise ContractError("logical metric steps must be nondecreasing")
        self._last_metric_step = step

    def materialize_inputs(self, inputs: Mapping[str, ArtifactInput], root: Path) -> Mapping[str, LocalArtifactRef]:
        materialized: dict[str, LocalArtifactRef] = {}
        for logical_name, value in inputs.items():
            reference = value.reference
            if reference.provider != "trackio":
                raise ContractError(f"Trackio cannot materialize {reference.provider!r} artifacts")
            if reference.namespace != self._project:
                raise ContractError("cross-project Trackio artifact materialization is unsupported")
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
                "event/attributes": dict(observation.attributes),
            }
        )

    def metric(self, observation: MetricObservation) -> None:
        self._validate_step(observation.step)
        values: dict[str, Any] = {observation.name: observation.value}
        if observation.attributes:
            values[f"{observation.name}/attributes"] = dict(observation.attributes)
        self._run.log(values, step=observation.step)

    def metrics(self, observation: MetricBatchObservation) -> None:
        self._validate_step(observation.step)
        values: dict[str, Any] = dict(observation.values)
        if observation.attributes:
            values["metric/attributes"] = dict(observation.attributes)
        self._run.log(values, step=observation.step)

    def trace(self, observation: TraceObservation) -> None:
        metadata = {
            "external_id": observation.external_id,
            "observation_type": observation.trace_type,
            "posttrain_attributes": dict(observation.attributes),
            **dict(observation.attributes),
        }
        if observation.trace_type == "verifiers":
            trace: trackio.Trace = trackio.VerifiersTrace(dict(observation.payload), metadata=metadata)
        else:
            messages = observation.payload.get("messages")
            if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
                raise ContractError("generic Trackio traces require a JSON messages list")
            extra = {key: value for key, value in observation.payload.items() if key != "messages"}
            trace = trackio.Trace(
                [dict(item) for item in cast(list[dict[str, Any]], messages)],
                metadata={"posttrain_payload_extra": extra, **extra, **metadata},
            )
        self._run.log({f"traces/{observation.trace_type}": trace})

    def artifact(self, artifact: ProducedArtifact) -> None:
        if not isinstance(artifact.reference, LocalArtifactRef):
            raise ContractError("Trackio output artifacts must be local before promotion")
        path = artifact.reference.path
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file() and _file_sha256(path) != artifact.reference.digest.removeprefix("sha256:"):
            raise ContractError(f"artifact digest does not match file contents: {path}")
        logged = trackio.Artifact(
            _artifact_name(artifact.name),
            type=artifact.kind,
            metadata={"logical_name": artifact.name, **dict(artifact.metadata)},
        )
        logged.add_dir(path) if path.is_dir() else logged.add_file(path)
        self._run.log_artifact(logged)

    def finish(self, outcome: RunOutcome) -> None:
        if self._outcome is not None:
            if self._outcome == outcome:
                return
            raise ContractError("Trackio run was already finalized with a different outcome")
        values: dict[str, Any] = {
            "run/status": outcome.status,
            "run/started_at": outcome.started_at.isoformat(),
            "run/finished_at": outcome.finished_at.isoformat(),
        }
        if outcome.error is not None:
            values["run/error_type"] = outcome.error.type
            values["run/error_message"] = outcome.error.message
        self._run.log(values)
        self._run.finish()
        self._outcome = outcome


class TrackioBackend:
    def __init__(self, settings: TrackioSettings | None = None) -> None:
        self.settings = settings or TrackioSettings()

    def start_run(self, spec: RunSpec) -> TrackioTrackedRun:
        started_at = datetime.now(UTC)
        project = self.settings.project or spec.project_id
        run = trackio.init(
            project=project,
            name=f"{spec.job_kind}-{spec.run_id[:8]}",
            group=spec.work_package_id,
            server_url=self.settings.server_url,
            config=_run_config(spec, started_at),
            embed=False,
            auto_log_gpu=self.settings.auto_log_gpu,
            auto_log_cpu=self.settings.auto_log_cpu,
        )
        return TrackioTrackedRun(run, project, spec)


class TrackioDataSource:
    def __init__(self, project: str, *, server_url: str | None = None) -> None:
        self.project = project
        self._api = trackio.Api(server_url=server_url)

    @property
    def capabilities(self) -> TrackingCapabilities:
        capabilities = self._api.capabilities()
        return TrackingCapabilities(
            provider="trackio",
            live_metrics=capabilities["full_history"],
            live_traces=capabilities["live_traces"],
            artifacts=capabilities["artifact_lineage"],
            artifact_lineage=capabilities["artifact_lineage"],
        )

    def _summary(self, run: Any) -> RunSummary:
        raw = run.summary()
        config = raw.get("config")
        if not isinstance(config, dict) or config.get("schema_version") != 4:
            raise ContractError(f"Trackio run {run.id!r} is not a canonical posttrain run")
        status = "running"
        started_at = _datetime(config["started_at"], field="started_at")
        finished_at = None
        error = None
        for row in run.history(
            keys=[
                "run/status",
                "run/started_at",
                "run/finished_at",
                "run/error_type",
                "run/error_message",
            ]
        ):
            if "run/status" not in row:
                continue
            status = row["run/status"]
            started_at = _datetime(row["run/started_at"], field="started_at")
            finished_at = _datetime(row["run/finished_at"], field="finished_at")
            if status == "failed":
                error = SafeRunError(
                    type=str(row.get("run/error_type") or "RunFailed"),
                    message=str(row.get("run/error_message") or "run failed"),
                )
        return RunSummary(
            provider="trackio",
            provider_run_id=run.id,
            run_id=str(config["run_id"]),
            display_name=run.name,
            project_id=str(config["project_id"]),
            work_package_id=str(config["work_package_id"]),
            stage=config["stage"],
            job_kind=str(config["job_kind"]),
            job_definition_version=str(config["job_definition_version"]),
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
        )

    async def list_runs(self, query: RunQuery) -> tuple[RunSummary, ...]:
        summaries: list[RunSummary] = []
        for run in self._api.runs(self.project):
            try:
                summary = self._summary(run)
            except ContractError:
                continue
            if query.project_id is not None and summary.project_id != query.project_id:
                continue
            if query.work_package_id is not None and summary.work_package_id != query.work_package_id:
                continue
            if query.job_kinds and summary.job_kind not in query.job_kinds:
                continue
            if query.statuses and summary.status not in query.statuses:
                continue
            summaries.append(summary)
        summaries.sort(key=lambda item: item.started_at, reverse=True)
        return tuple(summaries[: query.limit])

    async def get_run(self, run_id: str) -> RunDetail:
        provider_run = next(
            (run for run in self._api.runs(self.project) if (run.config or {}).get("run_id") == run_id),
            None,
        )
        if provider_run is None:
            raise LookupError(f"posttrain run {run_id!r} was not found in Trackio project {self.project!r}")
        summary = self._summary(provider_run)
        config = provider_run.config or {}
        events: list[EventRecord] = []
        metric_names: set[str] = set()
        for row in provider_run.history():
            if "event/name" in row:
                events.append(
                    EventRecord(
                        name=str(row["event/name"]),
                        occurred_at=_datetime(row["event/occurred_at"], field="event occurred_at"),
                        attributes=_json_mapping(row.get("event/attributes")),
                    )
                )
            for name, value in row.items():
                if name in _RESERVED_HISTORY_KEYS or name.startswith(("event/", "run/")):
                    continue
                if name.endswith("/attributes") or name == "metric/attributes":
                    continue
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metric_names.add(name)
        system_names = set(provider_run.system_metric_names())
        metric_names.update(
            canonical for canonical, (provider_name, _) in _SYSTEM_METRICS.items() if provider_name in system_names
        )
        if system_names:
            metric_names.add("system/wall_time_s")
        traces = provider_run.traces()
        return RunDetail(
            summary=summary,
            resolved_inputs=_json_mapping(config.get("resolved_selections")),
            source_metadata=_json_mapping(config.get("source_metadata")),
            metric_names=tuple(sorted(metric_names)),
            events=tuple(events),
            trace_count=len(traces),
        )

    async def metric_series(self, run_id: str, names: tuple[str, ...]) -> tuple[MetricSeries, ...]:
        provider_run = self._provider_run(run_id)
        regular_names = tuple(name for name in names if not name.startswith("system/"))
        raw = provider_run.metric_series(regular_names)
        values_by_name: dict[str, MetricSeries] = {}
        for name in regular_names:
            points = []
            for point in raw.get(name, []):
                value = point.get("value")
                if not isinstance(value, int | float) or isinstance(value, bool):
                    continue
                observed_at = point.get("timestamp")
                points.append(
                    MetricPoint(
                        value=float(value),
                        step=point.get("step"),
                        observed_at=(
                            _datetime(observed_at, field="metric timestamp") if observed_at is not None else None
                        ),
                    )
                )
            values_by_name[name] = MetricSeries(name=name, points=tuple(points))
        requested_system_names = tuple(name for name in names if name.startswith("system/"))
        if requested_system_names:
            summary = self._summary(provider_run)
            started_at = summary.started_at
            finished_at = summary.finished_at
            rows = [
                (row, observed_at)
                for row in provider_run.system_history()
                if started_at <= (observed_at := _datetime(row["timestamp"], field="system metric timestamp"))
                and (finished_at is None or observed_at <= finished_at)
            ]
            for name in requested_system_names:
                points = []
                for index, (row, observed_at) in enumerate(rows):
                    if name == "system/wall_time_s":
                        value = max(0.0, (observed_at - started_at).total_seconds())
                    else:
                        source = _SYSTEM_METRICS.get(name)
                        if source is None or source[0] not in row:
                            continue
                        raw_value = row[source[0]]
                        if not isinstance(raw_value, int | float) or isinstance(raw_value, bool):
                            continue
                        value = float(raw_value) * source[1]
                    points.append(
                        MetricPoint(
                            value=value,
                            step=index,
                            observed_at=observed_at,
                        )
                    )
                values_by_name[name] = MetricSeries(name=name, points=tuple(points))
        return tuple(values_by_name.get(name, MetricSeries(name=name)) for name in names)

    async def traces(self, run_id: str, query: TraceQuery) -> TracePage:
        provider_run = self._provider_run(run_id)
        offset = int(query.cursor) if query.cursor is not None else 0
        raw = provider_run.traces(limit=1000, offset=0, sort="step_asc")
        normalized = []
        for record in raw:
            metadata = _json_mapping(record.get("metadata"))
            payload = record.get("payload")
            if not isinstance(payload, dict):
                payload = {"messages": record.get("messages") or []}
            payload_extra = metadata.get("posttrain_payload_extra")
            if isinstance(payload_extra, dict):
                payload = {**payload, **payload_extra}
            external_id = record.get("external_id") or metadata.get("external_id") or record["id"]
            trace_type = str(metadata.get("observation_type") or record.get("trace_type") or "trackio")
            if query.trace_type is not None and trace_type != query.trace_type:
                continue
            attributes = metadata.get("posttrain_attributes")
            if not isinstance(attributes, dict):
                attributes = {
                    key: value
                    for key, value in metadata.items()
                    if key
                    not in {
                        "external_id",
                        "observation_type",
                        "posttrain_payload_extra",
                        "posttrain_attributes",
                    }
                }
            normalized.append(
                TraceRecord(
                    trace_type=trace_type,
                    external_id=str(external_id),
                    payload=dict(payload),
                    attributes=dict(attributes),
                )
            )
        page = normalized[offset : offset + query.limit]
        has_more = offset + query.limit < len(normalized)
        return TracePage(
            items=tuple(page),
            next_cursor=str(offset + query.limit) if has_more else None,
            live=True,
        )

    async def artifacts(self, run_id: str) -> ArtifactSet:
        provider_run = self._provider_run(run_id)
        raw = provider_run.artifacts()
        links = []
        for direction in ("input", "output"):
            for record in raw[direction]:
                metadata = _json_mapping(record.get("metadata"))
                logical_name = str(metadata.pop("logical_name", record["name"]))
                links.append(
                    ArtifactLink(
                        direction=direction,
                        logical_name=logical_name,
                        kind=str(record["type"]),
                        artifact=StoredArtifact(
                            provider="trackio",
                            namespace=self.project,
                            name=str(record["name"]),
                            version=f"v{record['version']}",
                            digest=record.get("digest"),
                            provider_metadata={
                                "version_id": record["version_id"],
                                "size_bytes": record["size_bytes"],
                                **metadata,
                            },
                        ),
                    )
                )
        return ArtifactSet(items=tuple(links))

    def _provider_run(self, run_id: str) -> Any:
        for run in self._api.runs(self.project):
            if (run.config or {}).get("run_id") == run_id:
                return run
        raise LookupError(f"posttrain run {run_id!r} was not found in Trackio project {self.project!r}")
