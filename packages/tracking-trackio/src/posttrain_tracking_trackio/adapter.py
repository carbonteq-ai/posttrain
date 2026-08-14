"""Trackio writer and normalized reader implementations."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import trackio
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
    TraceFactSet,
    TraceFactUpdateObservation,
    TraceObservation,
)
from posttrain.tracking import (
    ArtifactInput,
    ArtifactIntegrityResult,
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
    TraceAggregateBucket,
    TraceAggregateResult,
    TraceFactsQuery,
    TracePage,
    TraceQuery,
    TraceRecord,
    TrackingArtifactPurge,
    TrackingCapabilities,
    TrackingLifecycleAdmin,
    TrackingProjectDeletePlan,
    TrackingProjectDeleteReceipt,
    TrackingPurgePlan,
    TrackingPurgeReceipt,
)
from trackio.remote_client import RemoteClient
from trackio.run import Run as TrackioSDKRun
from trackio.utils import parse_trackio_server_url

_RESERVED_HISTORY_KEYS = {"step", "timestamp"}

# A run records its status and timings as ordinary metrics rather than as
# run-level fields, so these are the keys that describe its lifecycle.
_LIFECYCLE_KEYS = (
    "run/status",
    "run/started_at",
    "run/finished_at",
    "run/error_type",
    "run/error_message",
)
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
        raise ContractError("detached Trackio execution requires POSTTRAIN_TRACKIO_SERVER_URL")
    base_url, url_token = parse_trackio_server_url(server_url)
    resolved_write_token = write_token or url_token or os.getenv("TRACKIO_WRITE_TOKEN")
    if not resolved_write_token:
        raise ContractError("detached Trackio execution requires TRACKIO_WRITE_TOKEN")
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
        raise ContractError("required remote Trackio readiness probe returned an invalid response")


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


def _trackio_trace_facts(
    trace_type: str,
    external_id: str,
    facts: TraceFactSet,
    *,
    replace_reward_components: bool = True,
) -> Any:
    """Translate the neutral envelope only after the Trackio capability is present."""

    update_type = getattr(trackio, "TraceFactUpdate", None)
    component_type = getattr(trackio, "TraceRewardComponent", None)
    if update_type is None or component_type is None:
        raise ContractError(
            "the configured Trackio build does not support trace facts; "
            "install the declared Trackio trace-facts release before logging this run"
        )
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
                name=component.name,
                contribution=component.contribution,
                score=component.score,
                weight=component.weight,
                source_kind=component.source.kind,
                source_id=component.source.id,
            )
            for component in facts.reward_components
        ),
        provenance=dict(facts.provenance),
        state=facts.state,
        replace_reward_components=replace_reward_components,
    )


class TrackioTrackedRun:
    """One open Trackio run with retry-safe canonical finalization."""

    def __init__(self, run: TrackioSDKRun, project: str, spec: RunSpec) -> None:
        self._run = run
        self._project = project
        self._spec = spec
        self._outcome: RunOutcome | None = None
        self._last_metric_step: int | None = None
        self._published_artifacts: list[PublishedArtifact] = []
        self._pending_artifacts: list[tuple[ProducedArtifact, trackio.Artifact]] = []

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
            declared_content_digest = reference.provider_metadata.get("posttrain_content_digest")
            declared_content_digest_kind = reference.provider_metadata.get("posttrain_content_digest_kind")
            if declared_content_digest_kind == "file":
                files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
                if len(files) != 1:
                    raise ContractError(
                        f"materialized Trackio file artifact contains {len(files)} files: {reference.name}:{version}"
                    )
                digest = _file_sha256(files[0])
            else:
                digest = _tree_sha256(path)
            expected_digest = (
                declared_content_digest.removeprefix("sha256:") if isinstance(declared_content_digest, str) else None
            )
            if expected_digest is not None and digest != expected_digest:
                raise ContractError(
                    f"materialized Trackio artifact content digest does not match {reference.name}:{version}"
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
            if len(observation.facts) > 1:
                raise ContractError("a Verifiers trace may carry one complete source fact projection")
            trace_arguments: dict[str, Any] = {"metadata": metadata}
            if observation.facts:
                trace_arguments["trace_facts"] = _trackio_trace_facts(
                    observation.trace_type,
                    observation.external_id,
                    observation.facts[0],
                )
            trace = trackio.VerifiersTrace(dict(observation.payload), **trace_arguments)
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

    def trace_fact_update(self, observation: TraceFactUpdateObservation) -> None:
        flush = getattr(self._run, "flush", None)
        upsert = getattr(self._run, "upsert_trace_facts", None)
        if not callable(flush) or not callable(upsert):
            raise ContractError(
                "the configured Trackio build does not support trace-fact enrichment; "
                "install the declared Trackio trace-facts release"
            )
        # The source trace is queued by the normal logging path. Flush it before
        # sending the trace-keyed enrichment so a remote server never observes
        # the second phase before the physical parent row exists.
        flush()
        upsert(
            _trackio_trace_facts(
                observation.trace_type,
                observation.external_id,
                observation.facts,
                replace_reward_components=False,
            )
        )

    def artifact(self, artifact: ProducedArtifact) -> None:
        if not isinstance(artifact.reference, LocalArtifactRef):
            raise ContractError("Trackio output artifacts must be local before promotion")
        reference = cast(LocalArtifactRef, artifact.reference)
        path = reference.path
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file() and _file_sha256(path) != reference.digest.removeprefix("sha256:"):
            raise ContractError(f"artifact digest does not match file contents: {path}")
        logged = trackio.Artifact(
            _artifact_name(artifact.name),
            type=artifact.kind,
            metadata={
                "logical_name": artifact.name,
                **dict(artifact.metadata),
                **({"posttrain_role": artifact.role} if artifact.role is not None else {}),
                "posttrain_content_digest": reference.digest.removeprefix("sha256:"),
                "posttrain_content_digest_kind": ("file" if path.is_file() else "tree"),
            },
        )
        logged.add_dir(path) if path.is_dir() else logged.add_file(path)
        try:
            committed = cast(Any, self._run).log_artifact(logged, background=True)
        except RuntimeError as error:
            if "artifact publication queue is full" not in str(error):
                raise
            # Background publication is intentionally bounded. Drain completed
            # work before retrying so a short remote backlog does not turn an
            # otherwise valid training result into a queue-overflow failure.
            self.flush_artifacts(timeout=30)
            committed = cast(Any, self._run).log_artifact(logged, background=True)
        except TypeError as error:
            if "background" not in str(error):
                raise
            # Compatibility window for a previously released Trackio client;
            # the immutable dependency pin is updated only after the fork
            # release with background publication is qualified.
            committed = cast(Any, self._run).log_artifact(logged)
        self._pending_artifacts.append((artifact, committed))

    def _record_committed_artifact(self, artifact: ProducedArtifact, committed: trackio.Artifact) -> PublishedArtifact:
        reference = cast(LocalArtifactRef, artifact.reference)
        path = reference.path
        version = committed.version
        digest = committed.digest
        project = committed.project
        if version is None or digest is None or project is None:
            raise ContractError("Trackio did not return a committed artifact identity")
        if any(item.logical_name == artifact.name for item in self._published_artifacts):
            raise ContractError(f"Trackio run published duplicate logical artifact name: {artifact.name}")
        size_bytes = committed.size
        return PublishedArtifact(
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
                    "posttrain_content_digest": reference.digest.removeprefix("sha256:"),
                    "posttrain_content_digest_kind": ("file" if path.is_file() else "tree"),
                },
            ),
            required=artifact.required,
            size_bytes=size_bytes,
            metadata=artifact.metadata,
            role=artifact.role,
        )

    def _flush_pending_artifacts(self, timeout: float | None = None) -> tuple[PublishedArtifact, ...]:
        if not self._pending_artifacts:
            return ()
        committed: list[PublishedArtifact] = []
        remaining: list[tuple[ProducedArtifact, trackio.Artifact]] = []
        for artifact, handle in self._pending_artifacts:
            try:
                handle.wait(timeout=None if timeout is None else max(0, int(timeout)))
            except TimeoutError:
                remaining.append((artifact, handle))
                raise
            published = self._record_committed_artifact(artifact, handle)
            self._published_artifacts.append(published)
            committed.append(published)
        self._pending_artifacts = remaining
        return tuple(committed)

    def flush_artifacts(self, timeout: float | None = None) -> tuple[PublishedArtifact, ...]:
        """Drain queued Trackio publications before evidence reconciliation."""
        flusher = getattr(self._run, "flush_artifacts", None)
        if callable(flusher):
            flusher(timeout=timeout)
        return self._flush_pending_artifacts(timeout)

    def published_artifacts(self) -> tuple[PublishedArtifact, ...]:
        """Return only identities committed after an explicit bounded drain."""
        self.flush_artifacts(timeout=30)
        return tuple(self._published_artifacts)

    def finish(self, outcome: RunOutcome) -> None:
        if self._outcome is not None:
            if self._outcome == outcome:
                return
            raise ContractError("Trackio run was already finalized with a different outcome")
        self.flush_artifacts(timeout=30)
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
            raise ContractError("Trackio recovery project does not match the expected run")
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
            raise ContractError("Trackio recovery requires a running or already cancelled run")
        if finished_at < observed.started_at:
            raise ContractError("Trackio recovery finish time precedes the run start")

        if not isinstance(provider_run.config, dict):
            raise ContractError("Trackio recovery run config is unavailable")
        raw_summary = provider_run.summary()
        base_url, url_token = parse_trackio_server_url(server_url)
        write_token = self._write_token or url_token or os.getenv("TRACKIO_WRITE_TOKEN")
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
        tracked.finish(RunOutcome("cancelled", expected.started_at, finished_at))

        verified = TrackioDataSource(
            project,
            server_url=server_url,
        )._summary(self._exact_run(expected))
        _validate_recovery_identity(expected, verified)
        if verified.status != "cancelled":
            raise ContractError("Trackio cancellation recovery did not become durable")
        return "recovered"

    def _exact_run(self, expected: RunSummary) -> Any:
        if expected.provider != "trackio":
            raise ContractError("Trackio recovery received a non-Trackio run")
        if expected.provider_run_id is None:
            raise ContractError("Trackio recovery requires an exact provider run id")
        source = TrackioDataSource(
            expected.project_id,
            server_url=self.settings.server_url,
        )
        canonical = []
        for run in source._api.runs(expected.project_id):
            config = run.config
            if isinstance(config, dict) and config.get("run_id") == expected.run_id:
                canonical.append(run)
        if len(canonical) != 1:
            raise ContractError("Trackio recovery requires exactly one provider run for the canonical run id")
        provider_run = canonical[0]
        if str(provider_run.id) != expected.provider_run_id:
            raise ContractError("Trackio recovery provider run id does not match")
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
            raise ContractError(f"Trackio recovery {field.replace('_', ' ')} does not match")


class TrackioProjectCatalog:
    """Read the project catalog exposed by a pinned remote Trackio server."""

    def __init__(self, server_url: str) -> None:
        if not server_url.strip():
            raise ValueError("Trackio server URL cannot be empty")
        self.server_url = server_url

    def list_projects(self) -> tuple[str, ...]:
        raw = RemoteClient(self.server_url).predict(api_name="/get_all_projects")
        if not isinstance(raw, list) or not all(isinstance(project, str) for project in raw):
            raise ContractError("Trackio project catalog must be a list of project names")
        if any(not project.strip() for project in raw):
            raise ContractError("Trackio project names cannot be empty")
        return tuple(sorted(set(raw)))


class TrackioLifecycleAdmin:
    """Optional authenticated adapter for the fork's exact-run purge API.

    The adapter is deliberately capability-detected so the post6 client fails
    closed with a useful message until the fork release containing the purge
    endpoints is selected.
    """

    def __init__(self, server_url: str, *, write_token: str | None = None) -> None:
        if not server_url.strip():
            raise ValueError("Trackio server URL cannot be empty")
        base_url, url_token = parse_trackio_server_url(server_url)
        token = write_token or url_token or os.getenv("TRACKIO_WRITE_TOKEN")
        if not token:
            raise ContractError("Trackio purge requires TRACKIO_WRITE_TOKEN")
        self._client = RemoteClient(
            base_url,
            write_token=token,
            httpx_kwargs={"timeout": 30.0},
            verbose=False,
        )

    def plan_run_purge(
        self,
        *,
        project: str,
        provider_run_ids: tuple[str, ...],
    ) -> TrackingPurgePlan:
        method = getattr(self._client, "run_purge_plan", None)
        if method is None:
            raise ContractError("selected Trackio server does not support run purge")
        raw = method(project, provider_run_ids)
        if not isinstance(raw, dict):
            raise ContractError("Trackio run purge plan must be an object")
        artifacts = tuple(
            TrackingArtifactPurge(
                version_id=str(item["version_id"]),
                name=str(item["name"]),
                version=f"v{item['version']}",
                digest=None,
                logical_bytes=int(item.get("size_bytes", 0)),
                consumer_run_ids=tuple(str(value) for value in item.get("consumer_run_ids", [])),
            )
            for item in raw.get("artifacts", [])
            if isinstance(item, dict)
        )
        created_at = _datetime(raw.get("created_at"), field="purge created_at")
        return TrackingPurgePlan(
            provider=str(raw.get("provider", "trackio")),
            project=str(raw.get("project", project)),
            provider_run_ids=tuple(provider_run_ids),
            run_ids=tuple(str(value) for value in raw.get("run_ids", provider_run_ids)),
            artifacts=artifacts,
            blockers=tuple(str(value) for value in raw.get("blockers", [])),
            digest=str(raw["digest"]),
            created_at=created_at,
        )

    def apply_run_purge(self, plan: TrackingPurgePlan) -> TrackingPurgeReceipt:
        method = getattr(self._client, "purge_runs", None)
        if method is None:
            raise ContractError("selected Trackio server does not support run purge")
        raw = method(plan.project, plan.provider_run_ids, plan.digest)
        if not isinstance(raw, dict):
            raise ContractError("Trackio run purge receipt must be an object")
        return TrackingPurgeReceipt(
            provider=str(raw.get("provider", plan.provider)),
            project=str(raw.get("project", plan.project)),
            plan_digest=str(raw["plan_digest"]),
            deleted_provider_run_ids=tuple(str(value) for value in raw.get("deleted_provider_run_ids", [])),
            deleted_artifact_version_ids=tuple(str(value) for value in raw.get("deleted_artifact_version_ids", [])),
            already_absent_provider_run_ids=tuple(
                str(value) for value in raw.get("already_absent_provider_run_ids", [])
            ),
            completed_at=_datetime(raw.get("completed_at"), field="purge completed_at"),
        )

    def project_delete_plan(self, *, project: str) -> TrackingProjectDeletePlan:
        raw = self._client.project_delete_plan(project)
        if not isinstance(raw, dict) or not isinstance(raw.get("digest"), str):
            raise ContractError("selected Trackio server does not support digest-bound project purge")
        return TrackingProjectDeletePlan(
            provider=str(raw.get("provider", "trackio")),
            project=str(raw.get("project", project)),
            exists=bool(raw.get("exists", False)),
            runs=int(raw.get("runs", 0)),
            artifacts=int(raw.get("artifacts", 0)),
            artifact_versions=int(raw.get("artifact_versions", 0)),
            logical_bytes=int(raw.get("artifact_logical_bytes", 0)),
            storage_bytes=int(raw.get("artifact_storage_bytes", 0)) + int(raw.get("media_storage_bytes", 0)),
            blockers=tuple(str(value) for value in raw.get("blockers", [])),
            digest=str(raw["digest"]),
            created_at=_datetime(raw.get("created_at"), field="project purge created_at"),
        )

    def delete_project(self, plan: TrackingProjectDeletePlan) -> TrackingProjectDeleteReceipt:
        delete_method = getattr(self._client, "delete_project", None)
        if delete_method is None:
            raise ContractError("selected Trackio server does not support project purge")
        raw = delete_method(plan.project, plan.digest)
        if not isinstance(raw, dict):
            raise ContractError("Trackio project purge receipt must be an object")
        return TrackingProjectDeleteReceipt(
            provider=str(raw.get("provider", plan.provider)),
            project=str(raw.get("project", plan.project)),
            plan_digest=str(raw.get("plan_digest", plan.digest)),
            deleted=bool(raw.get("deleted", False)),
            completed_at=_datetime(raw.get("completed_at"), field="project purge completed_at"),
        )


class TrackioTraceFactWriter:
    """Authenticated, exact-run writer for already-retained trace facts.

    This intentionally exposes only the generic Trackio upsert endpoint.  The
    caller supplies a complete typed projection; model/template and Verifiers
    interpretation remain outside this backend adapter.
    """

    def __init__(self, server_url: str, *, write_token: str | None = None) -> None:
        if not server_url.strip():
            raise ValueError("Trackio server URL cannot be empty")
        base_url, url_token = parse_trackio_server_url(server_url)
        token = write_token or url_token or os.getenv("TRACKIO_WRITE_TOKEN")
        if not token:
            raise ContractError("Trackio trace-fact backfill requires TRACKIO_WRITE_TOKEN")
        self._client = RemoteClient(
            base_url,
            write_token=token,
            httpx_kwargs={"timeout": 60.0},
            verbose=False,
        )

    def upsert(
        self,
        *,
        project: str,
        run_name: str,
        provider_run_id: str,
        trace_type: str,
        external_id: str,
        facts: TraceFactSet,
    ) -> Any:
        """Persist one source projection without reopening or changing the run."""

        update = _trackio_trace_facts(trace_type, external_id, facts)
        response = self._client.predict(
            api_name="/upsert_trace_facts",
            project=project,
            run=run_name,
            run_id=provider_run_id,
            update=update.payload(),
        )
        receipt_type = getattr(trackio, "TraceFactWriteReceipt", None)
        if receipt_type is None:
            raise ContractError("the configured Trackio build does not expose trace-fact write receipts")
        if not isinstance(response, dict):
            raise ContractError("Trackio trace-fact backfill returned an invalid write receipt")
        return receipt_type(**response)


class TrackioPurgeActionExecutor:
    """Bridge one framework tracking action to digest-bound Trackio apply."""

    def __init__(self, admin: TrackingLifecycleAdmin) -> None:
        self._admin = admin
        self._plans: dict[str, TrackingPurgePlan | TrackingProjectDeletePlan] = {}

    def revalidate(self, action: Any) -> None:
        kind = getattr(action, "kind", None)
        if kind == "tracking.delete_project":
            target = getattr(action, "target", {})
            project = target.get("project") if isinstance(target, Mapping) else None
            action_id = getattr(action, "action_id", None)
            if not isinstance(project, str) or not project or not isinstance(action_id, str):
                raise ContractError("Trackio project purge action has an invalid target")
            plan = self._admin.project_delete_plan(project=project)
            if plan.blockers:
                raise ContractError("Trackio project purge is blocked: " + "; ".join(plan.blockers))
            self._plans[action_id] = plan
            return
        if kind != "tracking.delete_run":
            raise ContractError("unsupported Trackio purge action")
        target = getattr(action, "target", {})
        project = target.get("project")
        provider_run_id = target.get("provider_run_id")
        action_id = getattr(action, "action_id", None)
        if not (
            isinstance(project, str)
            and project
            and isinstance(provider_run_id, str)
            and provider_run_id
            and isinstance(action_id, str)
            and action_id
        ):
            raise ContractError("Trackio purge action has an invalid target")
        plan = self._admin.plan_run_purge(project=project, provider_run_ids=(provider_run_id,))
        if plan.blockers:
            raise ContractError("Trackio purge is blocked: " + "; ".join(plan.blockers))
        self._plans[action_id] = plan

    def apply(self, action: Any) -> None:
        action_id = getattr(action, "action_id", None)
        if not isinstance(action_id, str) or action_id not in self._plans:
            raise ContractError("Trackio purge action was not revalidated")
        plan = self._plans.pop(action_id)
        if isinstance(plan, TrackingPurgePlan):
            self._admin.apply_run_purge(plan)
        else:
            self._admin.delete_project(plan)


class TrackioDataSource:
    _DETAIL_EVENT_LIMIT = 256

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
            trace_facts=("available" if hasattr(trackio, "TraceFactsQuery") else "unavailable"),
        )

    def _summary(self, run: Any) -> RunSummary:
        raw = run.summary()
        config = raw.get("config")
        lifecycle: dict[str, Any] = {}
        # Lifecycle is written at run start and completion. Read only the tail
        # of the history; the old full-history call made opening one run scale
        # with every rollout metric ever recorded.
        log_count = raw.get("num_logs")
        offset = max(int(log_count) - 1000, 0) if isinstance(log_count, int) else 0
        for row in self._history_page(run, tuple(_LIFECYCLE_KEYS), limit=1000, offset=offset):
            if "run/status" in row:
                lifecycle = row
        return self._compose_summary(
            run_id=run.id,
            display_name=run.name,
            config=config,
            lifecycle=lifecycle,
        )

    @staticmethod
    def _history_page(
        run: Any,
        keys: tuple[str, ...],
        *,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        """Read one bounded page, with compatibility for pre-paged clients."""

        try:
            return run.history(keys=keys, limit=limit, offset=offset)
        except TypeError as error:
            if "unexpected keyword argument" not in str(error):
                raise
            rows = run.history(keys=keys)
            return rows[offset : offset + limit]

    def _compose_summary(
        self,
        *,
        run_id: str,
        display_name: str,
        config: Any,
        lifecycle: Mapping[str, Any],
    ) -> RunSummary:
        """Shape one run from its configuration and its lifecycle values.

        Both the per-run and the bulk listing paths land here, so a run cannot
        be described one way when read alone and another way when read with its
        neighbours.
        """
        if not isinstance(config, dict) or config.get("schema_version") != 4:
            raise ContractError(f"Trackio run {run_id!r} is not a canonical posttrain run")
        status = "running"
        started_at = _datetime(config["started_at"], field="started_at")
        finished_at = None
        error = None
        # A run that never recorded a status is still running as far as anyone
        # can tell, and keeps the start its configuration recorded.
        if "run/status" in lifecycle:
            status = lifecycle["run/status"]
            started_at = _datetime(lifecycle["run/started_at"], field="started_at")
            finished_at = _datetime(lifecycle["run/finished_at"], field="finished_at")
            if status == "failed":
                error = SafeRunError(
                    type=str(lifecycle.get("run/error_type") or "RunFailed"),
                    message=str(lifecycle.get("run/error_message") or "run failed"),
                )
        return RunSummary(
            provider="trackio",
            provider_run_id=run_id,
            run_id=str(config["run_id"]),
            display_name=display_name,
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
        # The Trackio client is synchronous. Awaiting it directly blocks the
        # event loop, so a caller listing several projects concurrently gets
        # them one after another instead.
        return await asyncio.to_thread(self._list_runs, query)

    def _list_runs(self, query: RunQuery) -> tuple[RunSummary, ...]:
        """Describe at most ``query.limit`` runs, newest first.

        Describing one run costs several round trips, so the listing is walked
        newest-first using the timestamp it already carries and stops as soon as
        enough runs match. Describing every run in the project before applying
        the limit made the cost of asking for one run proportional to the whole
        project's history: a four-project deployment spent fifty seconds
        answering a question about four.
        """
        provider_runs = sorted(
            self._api.runs(self.project),
            key=lambda run: str(getattr(run, "created_at", "") or ""),
            reverse=True,
        )
        # Two requests describe every run in the project. Reading each run
        # separately cost one request for its configuration and another for its
        # lifecycle, so listing was proportional to how many runs existed
        # rather than to how many were asked for.
        configs = self._api.run_configs(self.project)
        lifecycles = self._api.run_lifecycles(self.project)
        summaries: list[RunSummary] = []
        for run in provider_runs:
            if len(summaries) >= query.limit:
                break
            try:
                summary = self._compose_summary(
                    run_id=run.id,
                    display_name=run.name,
                    config=configs.get(run.id),
                    lifecycle=lifecycles.get(run.id) or {},
                )
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
            # Remember only what was described anyway; _provider_run resolves
            # anything else on demand.
            self._provider_runs_by_id.setdefault(summary.run_id, run)
            summaries.append(summary)
        summaries.sort(key=lambda item: item.started_at, reverse=True)
        return tuple(summaries)

    async def get_run(self, run_id: str) -> RunDetail:
        """Read run metadata without blocking Observatory's event loop."""

        return await asyncio.to_thread(self._get_run, run_id)

    def _get_run(self, run_id: str) -> RunDetail:
        provider_run = self._provider_run(run_id)
        summary = self._summary(provider_run)
        config = provider_run.config or {}
        # Run detail is a metadata request, not a request for the complete
        # time-series payload.  The summary already carries the metric catalog;
        # fetching every history row here made a large RL run block the
        # Observatory event loop before the page could render.
        raw_summary = provider_run.summary()
        metric_names = {
            str(name)
            for name in raw_summary.get("metrics", [])
            if isinstance(name, str)
            and name
            and name not in _RESERVED_HISTORY_KEYS
            and not name.startswith(("event/", "run/"))
            and not name.endswith("/attributes")
            and name != "metric/attributes"
        }
        events: list[EventRecord] = []
        for row in self._history_page(
            provider_run,
            ("event/name", "event/occurred_at", "event/attributes", "step", "timestamp"),
            limit=self._DETAIL_EVENT_LIMIT,
            offset=0,
        ):
            if "event/name" not in row:
                continue
            events.append(
                EventRecord(
                    name=str(row["event/name"]),
                    occurred_at=_datetime(row["event/occurred_at"], field="event occurred_at"),
                    attributes=_json_mapping(row.get("event/attributes")),
                )
            )
        system_names = set(provider_run.system_metric_names())
        metric_names.update(
            canonical for canonical, (provider_name, _) in _SYSTEM_METRICS.items() if provider_name in system_names
        )
        if system_names:
            metric_names.add("system/wall_time_s")
        trace_counter = getattr(provider_run, "trace_count", None)
        if callable(trace_counter):
            trace_count = int(cast(Any, trace_counter)())
        else:
            # Compatibility with a pre-count client. The old client has no
            # count endpoint, so retain its exact (but temporary) behavior;
            # the pinned client path above is the production bounded path.
            trace_count = len(provider_run.traces())
        return RunDetail(
            summary=summary,
            resolved_inputs=_json_mapping(config.get("resolved_selections")),
            source_metadata=_json_mapping(config.get("source_metadata")),
            metric_names=tuple(sorted(metric_names)),
            events=tuple(events),
            trace_count=trace_count,
        )

    async def metric_series(self, run_id: str, names: tuple[str, ...]) -> tuple[MetricSeries, ...]:
        """Read selected metric history on the provider worker pool."""

        return await asyncio.to_thread(self._metric_series, run_id, names)

    def _metric_series(self, run_id: str, names: tuple[str, ...]) -> tuple[MetricSeries, ...]:
        provider_run = self._provider_run(run_id)
        raw: dict[str, list[dict[str, object]]] = {name: [] for name in names}
        attribute_names = tuple(f"{name}/attributes" for name in names)
        for row in provider_run.history((*names, "metric/attributes", *attribute_names)):
            batch_attributes = _json_mapping(row.get("metric/attributes"))
            for name in names:
                if name not in row:
                    continue
                attributes = _json_mapping(row.get(f"{name}/attributes")) or batch_attributes
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
            system_source_names = tuple(
                source_name
                for name in requested_system_names
                if (source := _SYSTEM_METRICS.get(name)) is not None
                for source_name in (source[0],)
            )
            try:
                system_rows = provider_run.system_history(keys=system_source_names)
            except TypeError as error:
                if "unexpected keyword argument" not in str(error):
                    raise
                system_rows = provider_run.system_history()
            rows = [
                (row, observed_at)
                for row in system_rows
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

    @staticmethod
    def _normalize_trace_record(record: Mapping[str, Any]) -> TraceRecord:
        metadata = _json_mapping(record.get("metadata"))
        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = {"messages": record.get("messages") or []}
        payload_extra = metadata.get("posttrain_payload_extra")
        if isinstance(payload_extra, dict):
            payload = {**payload, **payload_extra}
        external_id = record.get("external_id") or metadata.get("external_id") or record["id"]
        trace_type = str(metadata.get("observation_type") or record.get("trace_type") or "trackio")
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
        return TraceRecord(
            trace_type=trace_type,
            external_id=str(external_id),
            payload=dict(payload),
            attributes=dict(attributes),
        )

    async def get_trace(self, run_id: str, external_id: str) -> TraceRecord | None:
        """Fetch one complete trace payload by its stable external ID."""

        return await asyncio.to_thread(self._get_trace, run_id, external_id)

    def _get_trace(self, run_id: str, external_id: str) -> TraceRecord | None:

        provider_run = self._provider_run(run_id)
        kwargs = {
            "search": external_id,
            "limit": 1,
            "offset": 0,
            "sort": "step_asc",
            "include_payload": True,
        }
        try:
            raw = provider_run.traces(**kwargs)
        except TypeError as error:
            if "unexpected keyword argument" not in str(error):
                raise
            kwargs.pop("include_payload", None)
            raw = provider_run.traces(**kwargs)
        for record in raw:
            normalized = self._normalize_trace_record(record)
            if normalized.external_id == external_id:
                return normalized
        return None

    async def traces(self, run_id: str, query: TraceQuery) -> TracePage:
        """Read one bounded trace page without blocking other HTTP requests."""

        return await asyncio.to_thread(self._traces, run_id, query)

    async def traces_by_provider_run_id(self, provider_run_id: str, query: TraceQuery) -> TracePage:
        """Read a bounded page for maintenance scoped by Trackio's physical run id."""

        return await asyncio.to_thread(self._traces_by_provider_run_id, provider_run_id, query)

    async def aggregate_trace_facts(self, run_id: str, query: TraceFactsQuery) -> TraceAggregateResult:
        return await asyncio.to_thread(self._aggregate_trace_facts, run_id, query)

    def _aggregate_trace_facts(self, run_id: str, query: TraceFactsQuery) -> TraceAggregateResult:
        query_type = getattr(trackio, "TraceFactsQuery", None)
        aggregate_type = getattr(trackio, "TraceAggregate", None)
        provider_run = self._provider_run(run_id)
        aggregate = getattr(provider_run, "aggregate_trace_facts", None)
        if query_type is None or aggregate_type is None or not callable(aggregate):
            return TraceAggregateResult(state="unavailable")
        response = cast(Any, aggregate(
            query_type(
                trace_type=query.trace_type,
                group_by=tuple(query.group_by),
                aggregates=tuple(
                    aggregate_type(
                        measure=item.measure,
                        operation=item.operation,
                        component_name=item.component_name,
                    )
                    for item in query.aggregates
                ),
                dimensions=dict(query.dimensions),
            )
        ))
        return TraceAggregateResult(
            state="available",
            buckets=tuple(
                TraceAggregateBucket(
                    dimensions=dict(bucket.dimensions),
                    trace_count=bucket.trace_count,
                    values=dict(bucket.values),
                    coverage=dict(bucket.coverage),
                )
                for bucket in response.buckets
            ),
        )

    def _traces(self, run_id: str, query: TraceQuery) -> TracePage:
        provider_run = self._provider_run(run_id)
        return self._traces_for_provider_run(provider_run, query)

    def _traces_by_provider_run_id(self, provider_run_id: str, query: TraceQuery) -> TracePage:
        return self._traces_for_provider_run(self._provider_run_by_id(provider_run_id), query)

    def _traces_for_provider_run(self, provider_run: Any, query: TraceQuery) -> TracePage:
        offset = int(query.cursor) if query.cursor is not None else 0
        provider_kwargs = {
            "limit": query.limit,
            "offset": offset,
            "sort": "step_asc",
            "include_payload": query.include_payload,
        }
        # Trackio's physical trace_type distinguishes Verifiers traces from
        # ordinary Trackio traces. Other post-training trace kinds live in
        # metadata, so those remain a bounded page followed by local shaping.
        if query.trace_type == "verifiers":
            provider_kwargs["trace_type"] = "verifiers"
        while True:
            try:
                raw = provider_run.traces(**provider_kwargs)
                break
            except TypeError as error:
                message = str(error)
                if "unexpected keyword argument" not in message:
                    raise
                if "include_payload" in message and "include_payload" in provider_kwargs:
                    provider_kwargs.pop("include_payload")
                    continue
                if "trace_type" in message and "trace_type" in provider_kwargs:
                    provider_kwargs.pop("trace_type")
                    continue
                raise
        normalized = []
        for record in raw:
            normalized_record = self._normalize_trace_record(record)
            trace_type = normalized_record.trace_type
            if query.trace_type is not None and trace_type != query.trace_type:
                continue
            normalized.append(normalized_record)
        page = normalized
        has_more = len(raw) == query.limit
        return TracePage(
            items=tuple(page),
            next_cursor=str(offset + len(raw)) if has_more else None,
            live=True,
        )

    async def artifacts(self, run_id: str) -> ArtifactSet:
        """Read artifact identities on the provider worker pool."""

        return await asyncio.to_thread(self._artifacts, run_id)

    async def verify_artifact(self, reference: StoredArtifactRef, *, deep: bool = False) -> ArtifactIntegrityResult:
        """Report provider verification support without downloading blobs implicitly."""

        if reference.provider != "trackio" or reference.namespace != self.project:
            return ArtifactIntegrityResult(
                "failed", failures=("artifact belongs to another provider/project",), deep=deep
            )
        return ArtifactIntegrityResult("unsupported", deep=deep)

    def _artifacts(self, run_id: str) -> ArtifactSet:
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

    def _provider_run_by_id(self, provider_run_id: str) -> Any:
        """Resolve the physical provider identity without assuming Posttrain config."""

        for run in self._api.runs(self.project):
            if str(run.id) == provider_run_id:
                return run
        raise LookupError(f"Trackio provider run {provider_run_id!r} was not found in project {self.project!r}")
