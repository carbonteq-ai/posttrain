"""Trackio writer and normalized reader implementations."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from posttrain.common import (
    ContractError,
    EventObservation,
    JsonValue,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    PublishedArtifact,
    StoredArtifactRef,
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
from trackio.remote_client import RemoteClient
from trackio.run import Run as TrackioSDKRun
from trackio.utils import parse_trackio_server_url

import trackio

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
    gpu_log_interval: float = 1.0
    cpu_log_interval: float = 5.0

    def __post_init__(self) -> None:
        if self.gpu_log_interval <= 0:
            raise ValueError("Trackio GPU log interval must be positive")
        if self.cpu_log_interval <= 0:
            raise ValueError("Trackio CPU log interval must be positive")


def require_remote_trackio_ready(
    *,
    project: str,
    server_url: str | None,
    write_token: str | None = None,
) -> None:
    """Fail unless the selected remote Trackio accepts authenticated writes.

    The probe is intentionally non-mutating: checking an empty artifact-digest
    set exercises TLS, routing, server compatibility, write authorization, and
    the configured storage backend without creating a provider run.
    """

    if server_url is None or not server_url.strip():
        raise ContractError(
            "detached Trackio execution requires POSTTRAIN_TRACKIO_SERVER_URL"
        )
    base_url, url_token = parse_trackio_server_url(server_url)
    resolved_write_token = (
        write_token or url_token or os.getenv("TRACKIO_WRITE_TOKEN")
    )
    if not resolved_write_token:
        raise ContractError(
            "detached Trackio execution requires TRACKIO_WRITE_TOKEN"
        )
    try:
        client = RemoteClient(
            base_url,
            write_token=resolved_write_token,
            httpx_kwargs={"timeout": 10.0},
            verbose=False,
        )
        response = client.predict(
            api_name="/check_artifact_blobs",
            project=project,
            digests=[],
            hf_token=None,
        )
    except Exception:
        raise ContractError(
            "required remote Trackio evidence is unavailable; verify network, "
            "TLS trust, write authorization, and server storage health"
        ) from None
    if response != {"present": []}:
        raise ContractError(
            "required remote Trackio readiness probe returned an invalid response"
        )


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
        self._published_artifacts: list[PublishedArtifact] = []

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
            digest = _tree_sha256(path)
            declared_content_digest = reference.provider_metadata.get(
                "posttrain_content_digest"
            )
            declared_content_digest_kind = reference.provider_metadata.get(
                "posttrain_content_digest_kind"
            )
            if declared_content_digest_kind == "file":
                files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
                if len(files) != 1:
                    raise ContractError(
                        f"materialized Trackio file artifact contains {len(files)} files: "
                        f"{reference.name}:{version}"
                    )
                digest = _file_sha256(files[0])
            else:
                digest = _tree_sha256(path)
            expected_digest = (
                declared_content_digest.removeprefix("sha256:")
                if isinstance(declared_content_digest, str)
                else None
            )
            if expected_digest is not None and digest != expected_digest:
                raise ContractError(
                    f"materialized Trackio artifact content digest does not match "
                    f"{reference.name}:{version}"
                )
            materialized[logical_name] = LocalArtifactRef(path, digest)
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
            if messages is None:
                messages = []
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
            metadata={
                "logical_name": artifact.name,
                **dict(artifact.metadata),
                **({"posttrain_role": artifact.role} if artifact.role is not None else {}),
                "posttrain_content_digest": artifact.reference.digest.removeprefix(
                    "sha256:"
                ),
                "posttrain_content_digest_kind": (
                    "file" if path.is_file() else "tree"
                ),
            },
        )
        logged.add_dir(path) if path.is_dir() else logged.add_file(path)
        committed = self._run.log_artifact(logged)
        version = committed.version
        digest = committed.digest
        project = committed.project
        if version is None or digest is None or project is None:
            raise ContractError("Trackio did not return a committed artifact identity")
        if any(item.logical_name == artifact.name for item in self._published_artifacts):
            raise ContractError(
                f"Trackio run published duplicate logical artifact name: {artifact.name}"
            )
        size_bytes = committed.size
        self._published_artifacts.append(
            PublishedArtifact(
                logical_name=artifact.name,
                kind=artifact.kind,
                reference=StoredArtifactRef(
                    provider="trackio",
                    namespace=project,
                    name=committed.name,
                    version=version,
                    digest=str(digest),
                    provider_metadata={
                        "size_bytes": size_bytes,
                        "posttrain_content_digest": artifact.reference.digest.removeprefix(
                            "sha256:"
                        ),
                        "posttrain_content_digest_kind": (
                            "file" if path.is_file() else "tree"
                        ),
                    },
                ),
                required=artifact.required,
                size_bytes=size_bytes,
                metadata=artifact.metadata,
                role=artifact.role,
            )
        )

    def published_artifacts(self) -> tuple[PublishedArtifact, ...]:
        """Return only identities synchronously committed by Trackio."""

        return tuple(self._published_artifacts)

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
        return self._open_run(spec, resume="never")

    def resume_run(self, spec: RunSpec, *, started_at: datetime) -> TrackioTrackedRun:
        """Resume the provider run selected by the canonical Trackio run name."""

        return self._open_run(spec, resume="must", started_at=started_at)

    def _open_run(
        self,
        spec: RunSpec,
        *,
        resume: str,
        started_at: datetime | None = None,
    ) -> TrackioTrackedRun:
        started_at = started_at or datetime.now(UTC)
        project = self.settings.project or spec.project_id
        run = trackio.init(
            project=project,
            name=f"{spec.job_kind}-{spec.run_id[:8]}",
            group=spec.work_package_id,
            server_url=self.settings.server_url,
            config=_run_config(spec, started_at),
            resume=resume,
            embed=False,
            auto_log_gpu=self.settings.auto_log_gpu if resume == "never" else False,
            gpu_log_interval=self.settings.gpu_log_interval,
            auto_log_cpu=self.settings.auto_log_cpu if resume == "never" else False,
            cpu_log_interval=self.settings.cpu_log_interval,
        )
        return TrackioTrackedRun(run, project, spec)


class TrackioCancelledRunRecovery:
    """Exact-id Trackio writer for audited cancellation recovery."""

    def __init__(
        self,
        settings: TrackioSettings,
        *,
        write_token: str | None = None,
    ) -> None:
        self.settings = settings
        self._write_token = write_token

    def recover_cancelled(
        self,
        expected: RunSummary,
        *,
        finished_at: datetime,
    ) -> Literal["recovered", "already-cancelled"]:
        project = self.settings.project or expected.project_id
        if project != expected.project_id:
            raise ContractError(
                "Trackio recovery project does not match the expected run"
            )
        server_url = self.settings.server_url
        require_remote_trackio_ready(
            project=project,
            server_url=server_url,
            write_token=self._write_token,
        )
        assert server_url is not None
        provider_run = self._exact_run(expected)
        observed = TrackioDataSource(
            project,
            server_url=server_url,
        )._summary(provider_run)
        _validate_recovery_identity(expected, observed)
        if observed.status == "cancelled":
            return "already-cancelled"
        if observed.status != "running":
            raise ContractError(
                "Trackio recovery requires a running or already cancelled run"
            )
        if finished_at < observed.started_at:
            raise ContractError(
                "Trackio recovery finish time precedes the run start"
            )

        if not isinstance(provider_run.config, dict):
            raise ContractError("Trackio recovery run config is unavailable")
        raw_summary = provider_run.summary()
        base_url, url_token = parse_trackio_server_url(server_url)
        write_token = (
            self._write_token
            or url_token
            or os.getenv("TRACKIO_WRITE_TOKEN")
        )
        if not write_token:
            raise ContractError("Trackio recovery requires a write token")
        run = TrackioSDKRun(
            url=base_url,
            project=project,
            client=None,
            name=provider_run.name,
            run_id=expected.provider_run_id,
            group=expected.work_package_id,
            config={},
            server_base_url=base_url,
            write_token=write_token,
            existing_runs=[provider_run.name],
            initial_last_step=raw_summary.get("last_step"),
            auto_log_gpu=False,
            auto_log_cpu=False,
        )
        # This exact provider run already owns immutable canonical config.
        # Recovery appends one terminal row; it must not rewrite physical or
        # canonical configuration through the SDK's first-log initialization.
        run._config_logged = True
        tracked = TrackioTrackedRun(
            run,
            project,
            RunSpec(
                project_id=expected.project_id,
                work_package_id=expected.work_package_id,
                stage=expected.stage,
                run_id=expected.run_id,
                job_kind=expected.job_kind,
                job_definition_version=expected.job_definition_version,
            ),
        )
        tracked.finish(
            RunOutcome("cancelled", expected.started_at, finished_at)
        )

        verified = TrackioDataSource(
            project,
            server_url=server_url,
        )._summary(self._exact_run(expected))
        _validate_recovery_identity(expected, verified)
        if verified.status != "cancelled":
            raise ContractError(
                "Trackio cancellation recovery did not become durable"
            )
        return "recovered"

    def _exact_run(self, expected: RunSummary) -> Any:
        if expected.provider != "trackio":
            raise ContractError("Trackio recovery received a non-Trackio run")
        if expected.provider_run_id is None:
            raise ContractError(
                "Trackio recovery requires an exact provider run id"
            )
        source = TrackioDataSource(
            expected.project_id,
            server_url=self.settings.server_url,
        )
        canonical = []
        for run in source._api.runs(expected.project_id):
            config = run.config
            if (
                isinstance(config, dict)
                and config.get("run_id") == expected.run_id
            ):
                canonical.append(run)
        if len(canonical) != 1:
            raise ContractError(
                "Trackio recovery requires exactly one provider run for the "
                "canonical run id"
            )
        provider_run = canonical[0]
        if str(provider_run.id) != expected.provider_run_id:
            raise ContractError(
                "Trackio recovery provider run id does not match"
            )
        return provider_run


def _validate_recovery_identity(
    expected: RunSummary,
    observed: RunSummary,
) -> None:
    fields = (
        "provider",
        "provider_run_id",
        "run_id",
        "project_id",
        "work_package_id",
        "stage",
        "job_kind",
        "job_definition_version",
        "started_at",
    )
    for field in fields:
        if getattr(observed, field) != getattr(expected, field):
            raise ContractError(
                f"Trackio recovery {field.replace('_', ' ')} does not match"
            )


class TrackioDataSource:
    def __init__(self, project: str, *, server_url: str | None = None) -> None:
        self.project = project
        self._api = trackio.Api(server_url=server_url)
        self._provider_runs_by_id: dict[str, Any] = {}

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
        provider_runs = tuple(self._api.runs(self.project))
        self._provider_runs_by_id = {
            str(config["run_id"]): run
            for run in provider_runs
            if isinstance((config := run.config), dict) and isinstance(config.get("run_id"), str)
        }
        for run in provider_runs:
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
        provider_run = self._provider_run(run_id)
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
        raw: dict[str, list[dict[str, object]]] = {name: [] for name in names}
        attribute_names = tuple(f"{name}/attributes" for name in names)
        for row in provider_run.history(
            (*names, "metric/attributes", *attribute_names)
        ):
            batch_attributes = _json_mapping(row.get("metric/attributes"))
            for name in names:
                if name not in row:
                    continue
                attributes = _json_mapping(
                    row.get(f"{name}/attributes")
                ) or batch_attributes
                raw[name].append(
                    {
                        "value": row[name],
                        "step": row.get("step"),
                        "timestamp": row.get("timestamp"),
                        "attributes": attributes,
                    }
                )
        values_by_name: dict[str, MetricSeries] = {}
        for name in names:
            points = []
            for point in raw.get(name, []):
                value = point.get("value")
                if not isinstance(value, int | float) or isinstance(value, bool):
                    continue
                observed_at = point.get("timestamp")
                step = point.get("step")
                points.append(
                    MetricPoint(
                        value=float(value),
                        step=step if isinstance(step, int) and not isinstance(step, bool) else None,
                        observed_at=(
                            _datetime(observed_at, field="metric timestamp") if observed_at is not None else None
                        ),
                        attributes=_json_mapping(point.get("attributes")),
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
                direct = values_by_name.get(name)
                if direct is not None and direct.points:
                    continue
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
        raw = []
        provider_offset = 0
        while True:
            page = provider_run.traces(limit=1000, offset=provider_offset, sort="step_asc")
            raw.extend(page)
            if len(page) < 1000:
                break
            provider_offset += len(page)
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
        cached = self._provider_runs_by_id.get(run_id)
        if cached is not None:
            return cached
        for run in self._api.runs(self.project):
            config = run.config or {}
            posttrain_run_id = config.get("run_id")
            if isinstance(posttrain_run_id, str):
                self._provider_runs_by_id[posttrain_run_id] = run
            if posttrain_run_id == run_id:
                return run
        raise LookupError(f"posttrain run {run_id!r} was not found in Trackio project {self.project!r}")
