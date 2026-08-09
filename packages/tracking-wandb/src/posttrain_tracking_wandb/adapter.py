"""W&B writer and Public API normalized reader implementations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import wandb
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
    TracePage,
    TraceQuery,
    TraceRecord,
    TrackingCapabilities,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

_WANDB_SYSTEM_METRICS = {
    "system/cpu_percent": ("system.cpu", 1.0),
    "system/process_rss_bytes": ("system.proc.memory.rssMB", float(1024**2)),
    "system/wall_time_s": ("_runtime", 1.0),
}
_CANONICAL_WANDB_SYSTEM_METRICS = (
    "system/gpu_utilization",
    "system/gpu_vram_used_bytes",
    *_WANDB_SYSTEM_METRICS,
)
_T = TypeVar("_T")
_READ_PROCESS_POOL: ProcessPoolExecutor | None = None
_PROCESS_READERS: dict[str, WandbDataSource] = {}


def _read_process_pool() -> ProcessPoolExecutor:
    global _READ_PROCESS_POOL
    if _READ_PROCESS_POOL is None:
        _READ_PROCESS_POOL = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )
    return _READ_PROCESS_POOL


def _process_read(settings: WandbSettings, operation: str, arguments: tuple[object, ...]) -> object:
    key = settings.model_dump_json()
    reader = _PROCESS_READERS.get(key)
    if reader is None:
        overrides = {"base_url": settings.base_url} if settings.base_url else None
        reader = WandbDataSource(settings, api=wandb.Api(overrides=overrides))
        _PROCESS_READERS[key] = reader
    return getattr(reader, operation)(*arguments)


def _system_value(row: Mapping[str, object], name: str) -> tuple[float | None, tuple[str, ...]]:
    if name == "system/gpu_utilization":
        native_names = tuple(key for key in row if re.fullmatch(r"system\.gpu\.\d+\.gpu", key))
        values = [row[key] for key in native_names]
        numeric = [float(value) for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
        return ((sum(numeric) / len(numeric)) if numeric else None, native_names)
    if name == "system/gpu_vram_used_bytes":
        native_names = tuple(key for key in row if re.fullmatch(r"system\.gpu\.\d+\.memoryAllocatedBytes", key))
        values = [row[key] for key in native_names]
        numeric = [float(value) for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
        return (sum(numeric) if numeric else None, native_names)
    native_name, scale = _WANDB_SYSTEM_METRICS[name]
    value = row.get(native_name)
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None, (native_name,)
    return float(value) * scale, (native_name,)


class WandbSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    entity: str = Field(min_length=1)
    project: str = Field(min_length=1)
    base_url: str | None = None
    mode: Literal["online", "offline"] = "online"
    tags: tuple[str, ...] = ("posttrain",)

    @model_validator(mode="after")
    def validate_base_url(self) -> WandbSettings:
        if self.base_url is not None and not self.base_url.startswith(("http://", "https://")):
            raise ValueError("W&B base_url must be an HTTP(S) URL")
        return self


def _run_config(spec: RunSpec, started_at: datetime) -> dict[str, JsonValue]:
    return {
        "schema_version": 4,
        "provider": "wandb",
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


def _aware_datetime(value: object, field: str) -> datetime:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if not isinstance(value, str):
        raise ContractError(f"W&B {field} must be an ISO timestamp or epoch")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ContractError(f"W&B {field} must be timezone-aware")
    return parsed


def _mapping(value: object) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], dict(value)) if isinstance(value, dict) else {}


def _unflatten_attributes(row: Mapping[str, object], prefix: str) -> dict[str, JsonValue]:
    result: dict[str, Any] = {}
    dotted_prefix = f"{prefix}."
    for key, value in row.items():
        if not key.startswith(dotted_prefix):
            continue
        target = result
        parts = key.removeprefix(dotted_prefix).split(".")
        for part in parts[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                break
            target = child
        else:
            target[parts[-1]] = value
    return cast(dict[str, JsonValue], result)


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for child in files:
        if path.is_dir():
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(b"\0")
        with child.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def wandb_artifact_name(logical_name: str, run_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", logical_name).strip("-")
    if not normalized:
        raise ContractError("artifact logical name cannot normalize to an empty W&B name")
    return f"{normalized}-{run_id}"


class WandbTrackedRun:
    def __init__(self, run: Any, settings: WandbSettings, spec: RunSpec) -> None:
        self._run = run
        self._settings = settings
        self._spec = spec
        self._outcome: RunOutcome | None = None
        self._last_step: int | None = None
        self._sequence = 0
        self._trace_dir = tempfile.TemporaryDirectory(prefix="posttrain-wandb-")
        self._trace_path = Path(self._trace_dir.name) / "traces.jsonl"
        self._published_artifact_handles: list[tuple[ProducedArtifact, str, Any]] = []

    @property
    def run_id(self) -> str:
        return self._spec.run_id

    def _next_sequence(self) -> int:
        value = self._sequence
        self._sequence += 1
        return value

    def _validate_step(self, step: int | None) -> None:
        if step is None:
            return
        if step < 0:
            raise ContractError("logical metric steps must be non-negative")
        if self._last_step is not None and step < self._last_step:
            raise ContractError("logical metric steps must be nondecreasing")
        self._last_step = step

    def materialize_inputs(self, inputs: Mapping[str, ArtifactInput], root: Path) -> Mapping[str, LocalArtifactRef]:
        materialized: dict[str, LocalArtifactRef] = {}
        namespace = f"{self._settings.entity}/{self._settings.project}"
        for logical_name, value in inputs.items():
            reference = value.reference
            if reference.provider != "wandb" or reference.namespace != namespace:
                raise ContractError("W&B can only materialize artifacts from its configured namespace")
            artifact = self._run.use_artifact(
                f"{reference.namespace}/{reference.name}:{reference.version}",
                type=value.kind,
            )
            destination = root / logical_name
            destination.mkdir(parents=True, exist_ok=False)
            path = Path(artifact.download(root=str(destination))).resolve()
            materialized[logical_name] = LocalArtifactRef(path, _tree_sha256(path))
        return materialized

    def event(self, observation: EventObservation) -> None:
        self._run.log(
            {
                "posttrain/sequence": self._next_sequence(),
                "event/name": observation.name,
                "event/occurred_at": observation.occurred_at.isoformat(),
                "event/attributes": dict(observation.attributes),
            }
        )

    def metric(self, observation: MetricObservation) -> None:
        self._validate_step(observation.step)
        values: dict[str, Any] = {
            "posttrain/sequence": self._next_sequence(),
            observation.name: observation.value,
        }
        if observation.step is not None:
            values["posttrain/step"] = observation.step
        if observation.attributes:
            values[f"{observation.name}/attributes"] = dict(observation.attributes)
        self._run.log(values)

    def metrics(self, observation: MetricBatchObservation) -> None:
        self._validate_step(observation.step)
        values: dict[str, Any] = {
            "posttrain/sequence": self._next_sequence(),
            **dict(observation.values),
        }
        if observation.step is not None:
            values["posttrain/step"] = observation.step
        if observation.attributes:
            values["metric/attributes"] = dict(observation.attributes)
        self._run.log(values)

    def trace(self, observation: TraceObservation) -> None:
        record = {
            "trace_type": observation.trace_type,
            "external_id": observation.external_id,
            "payload": dict(observation.payload),
            "attributes": dict(observation.attributes),
        }
        with self._trace_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            destination.write("\n")

    def artifact(self, artifact: ProducedArtifact) -> None:
        if not isinstance(artifact.reference, LocalArtifactRef):
            raise ContractError("W&B output artifacts must be local before promotion")
        artifact_name = wandb_artifact_name(artifact.name, self._spec.run_id)
        logged = wandb.Artifact(
            artifact_name,
            type=artifact.kind,
            metadata={
                "logical_name": artifact.name,
                **({"posttrain_role": artifact.role} if artifact.role is not None else {}),
                **dict(artifact.metadata),
            },
        )
        path = artifact.reference.path
        if path.is_dir():
            logged.add_dir(str(path))
        else:
            logged.add_file(str(path))
        committed = self._run.log_artifact(logged)
        self._published_artifact_handles.append((artifact, artifact_name, committed or logged))

    def published_artifacts(self) -> tuple[PublishedArtifact, ...]:
        """Wait for W&B to assign exact immutable versions before cleanup."""

        published: list[PublishedArtifact] = []
        for produced, artifact_name, handle in self._published_artifact_handles:
            wait = getattr(handle, "wait", None)
            committed = wait() if callable(wait) else handle
            version = getattr(committed, "version", None)
            digest = getattr(committed, "digest", None)
            if not isinstance(version, str) or not version:
                raise ContractError(f"W&B did not return a committed version for {produced.name}")
            if not isinstance(digest, str) or not digest:
                raise ContractError(f"W&B did not return a committed digest for {produced.name}")
            size = getattr(committed, "size", None)
            size_bytes = size if isinstance(size, int) and size >= 0 else None
            published.append(
                PublishedArtifact(
                    logical_name=produced.name,
                    kind=produced.kind,
                    reference=StoredArtifactRef(
                        provider="wandb",
                        namespace=f"{self._settings.entity}/{self._settings.project}",
                        name=artifact_name,
                        version=version,
                        digest=digest,
                        provider_metadata={"size_bytes": size_bytes},
                    ),
                    required=produced.required,
                    size_bytes=size_bytes,
                    metadata=produced.metadata,
                    role=produced.role,
                )
            )
        return tuple(published)

    def flush_artifacts(self, timeout: float | None = None) -> tuple[PublishedArtifact, ...]:
        del timeout
        return self.published_artifacts()

    def finish(self, outcome: RunOutcome) -> None:
        if self._outcome is not None:
            if self._outcome == outcome:
                return
            raise ContractError("W&B run was already finalized with a different outcome")
        if self._trace_path.exists():
            traces = wandb.Artifact(
                f"posttrain-traces-{self._spec.run_id}",
                type="posttrain-traces",
                metadata={"posttrain_role": "traces", "run_id": self._spec.run_id},
            )
            traces.add_file(str(self._trace_path), name="traces.jsonl")
            self._run.log_artifact(traces)
        summary: dict[str, Any] = {
            "posttrain/status": outcome.status,
            "posttrain/started_at": outcome.started_at.isoformat(),
            "posttrain/finished_at": outcome.finished_at.isoformat(),
        }
        if outcome.error is not None:
            summary["posttrain/error_type"] = outcome.error.type
            summary["posttrain/error_message"] = outcome.error.message
        self._run.summary.update(summary)
        self._run.finish(exit_code=0 if outcome.status == "succeeded" else 1)
        self._outcome = outcome
        self._trace_dir.cleanup()


class WandbBackend:
    def __init__(self, settings: WandbSettings) -> None:
        self.settings = settings

    def start_run(self, spec: RunSpec) -> WandbTrackedRun:
        started_at = datetime.now(UTC)
        sdk_settings = wandb.Settings(base_url=self.settings.base_url) if self.settings.base_url else None
        run = wandb.init(
            entity=self.settings.entity,
            project=self.settings.project,
            id=spec.run_id,
            name=f"{spec.job_kind}-{spec.run_id[:8]}",
            group=spec.work_package_id,
            job_type=spec.job_kind,
            config=_run_config(spec, started_at),
            tags=list(self.settings.tags),
            mode=self.settings.mode,
            resume="never",
            settings=sdk_settings,
        )
        if run is None:
            raise RuntimeError("wandb.init returned no run")
        run.define_metric("*", step_metric="posttrain/step")
        return WandbTrackedRun(run, self.settings, spec)


class WandbDataSource:
    def __init__(self, settings: WandbSettings, *, api: Any | None = None) -> None:
        if settings.mode != "online":
            raise ContractError("W&B Public API reads require online mode and a synced run")
        self.settings = settings
        self._api_overrides = {"base_url": settings.base_url} if settings.base_url else None
        self._api = api
        self._read_lock = asyncio.Lock()

    async def _read(
        self,
        operation: str,
        arguments: tuple[object, ...],
        local_operation: Callable[[], _T],
    ) -> _T:
        """Keep blocking Public API I/O off the caller's event loop."""

        if self._api is None:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                _read_process_pool(),
                _process_read,
                self.settings,
                operation,
                arguments,
            )
            return cast(_T, result)
        async with self._read_lock:
            return await asyncio.to_thread(local_operation)

    def _client(self) -> Any:
        if self._api is None:
            self._api = wandb.Api(overrides=self._api_overrides)
        return self._api

    @property
    def capabilities(self) -> TrackingCapabilities:
        return TrackingCapabilities(
            provider="wandb",
            live_metrics=True,
            live_traces=False,
            artifacts=True,
            artifact_lineage=True,
        )

    @property
    def _path(self) -> str:
        return f"{self.settings.entity}/{self.settings.project}"

    def _summary(self, run: Any) -> RunSummary:
        config = dict(run.config)
        if config.get("schema_version") != 4 or config.get("provider") != "wandb":
            raise ContractError(f"W&B run {run.id!r} is not a canonical posttrain run")
        native_summary = dict(run.summary)
        status = native_summary.get("posttrain/status", "running")
        started_at = native_summary.get("posttrain/started_at", config["started_at"])
        finished = native_summary.get("posttrain/finished_at")
        error = None
        if status == "failed":
            error = SafeRunError(
                type=str(native_summary.get("posttrain/error_type") or "RunFailed"),
                message=str(native_summary.get("posttrain/error_message") or "run failed"),
            )
        return RunSummary(
            provider="wandb",
            provider_run_id=str(run.id),
            run_id=str(config["run_id"]),
            display_name=str(run.name or run.id),
            project_id=str(config["project_id"]),
            work_package_id=str(config["work_package_id"]),
            stage=config["stage"],
            job_kind=str(config["job_kind"]),
            job_definition_version=str(config["job_definition_version"]),
            status=status,
            started_at=_aware_datetime(started_at, "started_at"),
            finished_at=_aware_datetime(finished, "finished_at") if finished is not None else None,
            error=error,
        )

    async def list_runs(self, query: RunQuery) -> tuple[RunSummary, ...]:
        return await self._read("_list_runs", (query,), lambda: self._list_runs(query))

    def _list_runs(self, query: RunQuery) -> tuple[RunSummary, ...]:
        api = self._client()
        refresh = getattr(api, "flush", None)
        if callable(refresh):
            refresh()
        summaries = []
        for run in api.runs(self._path):
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

    def _run(self, run_id: str) -> Any:
        return self._client().run(f"{self._path}/{run_id}")

    async def get_run(self, run_id: str) -> RunDetail:
        return await self._read("_get_run", (run_id,), lambda: self._get_run(run_id))

    def _get_run(self, run_id: str) -> RunDetail:
        run = self._run(run_id)
        config = dict(run.config)
        events = []
        metric_names = set()
        for row in run.scan_history():
            if row.get("event/name") is not None:
                events.append(
                    EventRecord(
                        name=str(row["event/name"]),
                        occurred_at=_aware_datetime(row["event/occurred_at"], "event occurred_at"),
                        attributes=(
                            _mapping(row.get("event/attributes")) or _unflatten_attributes(row, "event/attributes")
                        ),
                    )
                )
        for key in run.summary.keys():
            if isinstance(run.summary[key], int | float) and not isinstance(run.summary[key], bool):
                metric_names.add(str(key))
        trace_count = len(self._traces(run_id, TraceQuery(limit=1000)).items)
        return RunDetail(
            summary=self._summary(run),
            resolved_inputs=_mapping(config.get("resolved_selections")),
            source_metadata=_mapping(config.get("source_metadata")),
            metric_names=tuple(sorted(metric_names)),
            events=tuple(events),
            trace_count=trace_count,
        )

    async def metric_series(self, run_id: str, names: tuple[str, ...]) -> tuple[MetricSeries, ...]:
        return await self._read(
            "_metric_series",
            (run_id, names),
            lambda: self._metric_series(run_id, names),
        )

    def _metric_series(self, run_id: str, names: tuple[str, ...]) -> tuple[MetricSeries, ...]:
        run = self._run(run_id)
        points = {name: [] for name in names}
        default_names = tuple(name for name in names if name not in _CANONICAL_WANDB_SYSTEM_METRICS)
        for row in run.scan_history():
            for name in default_names:
                value = row.get(name)
                if not isinstance(value, int | float) or isinstance(value, bool):
                    continue
                observed = row.get("_timestamp")
                points[name].append(
                    MetricPoint(
                        value=float(value),
                        step=row.get("posttrain/step"),
                        observed_at=_aware_datetime(observed, "metric timestamp") if observed else None,
                        attributes=(
                            _unflatten_attributes(row, f"{name}/attributes")
                            or _unflatten_attributes(row, "metric/attributes")
                        ),
                    )
                )
        system_names = tuple(name for name in names if name in _CANONICAL_WANDB_SYSTEM_METRICS)
        if system_names:
            system_rows = run.history(samples=10_000, pandas=False, stream="system")
            for index, row in enumerate(system_rows):
                observed = row.get("_timestamp")
                for name in system_names:
                    value, native_names = _system_value(row, name)
                    if value is None:
                        continue
                    points[name].append(
                        MetricPoint(
                            value=value,
                            step=index,
                            observed_at=(_aware_datetime(observed, "system metric timestamp") if observed else None),
                            attributes={"provider_metrics": list(native_names)},
                        )
                    )
        return tuple(MetricSeries(name=name, points=tuple(points[name])) for name in names)

    def _artifacts(self, run: Any, method: str) -> Iterable[Any]:
        value = getattr(run, method)
        return cast(Iterable[Any], value() if callable(value) else value)

    async def traces(self, run_id: str, query: TraceQuery) -> TracePage:
        return await self._read("_traces", (run_id, query), lambda: self._traces(run_id, query))

    def _traces(self, run_id: str, query: TraceQuery) -> TracePage:
        run = self._run(run_id)
        records: list[TraceRecord] = []
        for artifact in self._artifacts(run, "logged_artifacts"):
            if artifact.type != "posttrain-traces":
                continue
            directory = Path(artifact.download())
            with (directory / "traces.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    raw = json.loads(line)
                    if query.trace_type is not None and raw["trace_type"] != query.trace_type:
                        continue
                    records.append(TraceRecord.model_validate(raw, strict=True))
        offset = int(query.cursor or 0)
        page = records[offset : offset + query.limit]
        next_cursor = str(offset + query.limit) if offset + query.limit < len(records) else None
        return TracePage(items=tuple(page), next_cursor=next_cursor, live=False)

    async def artifacts(self, run_id: str) -> ArtifactSet:
        return await self._read(
            "_artifact_set",
            (run_id,),
            lambda: self._artifact_set(run_id),
        )

    async def verify_artifact(self, reference: StoredArtifactRef, *, deep: bool = False) -> ArtifactIntegrityResult:
        if reference.provider != "wandb" or reference.namespace != self._path:
            return ArtifactIntegrityResult(
                "failed", failures=("artifact belongs to another provider/project",), deep=deep
            )
        return ArtifactIntegrityResult("unsupported", deep=deep)

    def _artifact_set(self, run_id: str) -> ArtifactSet:
        run = self._run(run_id)
        links = []
        for direction, method in (("input", "used_artifacts"), ("output", "logged_artifacts")):
            for artifact in self._artifacts(run, method):
                if artifact.type == "posttrain-traces":
                    continue
                metadata = _mapping(getattr(artifact, "metadata", {}))
                logical_name = str(metadata.pop("logical_name", artifact.name))
                name = str(artifact.name).split(":", 1)[0]
                version = str(getattr(artifact, "version", "latest"))
                links.append(
                    ArtifactLink(
                        direction=cast(Literal["input", "output"], direction),
                        logical_name=logical_name,
                        kind=str(artifact.type),
                        artifact=StoredArtifact(
                            provider="wandb",
                            namespace=self._path,
                            name=name,
                            version=version,
                            digest=getattr(artifact, "digest", None),
                            provider_metadata={"id": str(artifact.id), **metadata},
                        ),
                    )
                )
        return ArtifactSet(items=tuple(links))
