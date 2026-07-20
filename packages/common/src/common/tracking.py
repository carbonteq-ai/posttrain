"""Narrow Trackio integration shared by train, eval, serve, and reports."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import trackio

from common import RUNS_DIR, TRACKIO_PROJECT
from common.profiles import ResolvedProfile
from common.runs import RunContext, RunKind

TRACKING_SCHEMA_VERSION = 1


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _model_fields(profile: ResolvedProfile | None) -> dict[str, Any]:
    if profile is None or profile.kind != "models":
        return {}
    model = profile.data["model"]
    return {
        "model_profile_id": profile.data["id"],
        "model_artifact": model["artifact"],
        "model_form": model["form"],
        "model_family": model.get("family"),
    }


@dataclass(slots=True)
class TrackedRun:
    """One typed run with Trackio telemetry and a local recovery bundle."""

    context: RunContext
    trackio_run: Any
    started_at: float = field(default_factory=time.perf_counter)
    _finished: bool = False

    @property
    def id(self) -> str:
        return str(self.trackio_run.id)

    @property
    def name(self) -> str:
        return str(self.trackio_run.name)

    @classmethod
    def start(
        cls,
        run_kind: RunKind,
        resolved_config: dict[str, Any],
        *,
        resolved_profile: ResolvedProfile | None = None,
        name: str | None = None,
        project: str = TRACKIO_PROJECT,
        runs_dir: Path = RUNS_DIR,
        job_id: str | None = None,
        branch_id: str | None = None,
        stage_id: str | None = None,
        auto_log_gpu: bool = True,
        auto_log_cpu: bool = True,
    ) -> TrackedRun:
        run_name = name or f"{run_kind}-{_timestamp()}-{uuid.uuid4().hex[:6]}"
        local_root = runs_dir / run_name
        context = RunContext.create(
            local_root,
            run_kind,
            resolved_config,
            resolved_profile,
        )
        config: dict[str, Any] = {
            "schema_version": TRACKING_SCHEMA_VERSION,
            "run_kind": run_kind,
            "job_id": job_id,
            "branch_id": branch_id,
            "stage_id": stage_id,
            **_model_fields(resolved_profile),
            "resolved_config": resolved_config,
        }
        config = {key: value for key, value in config.items() if value is not None}
        try:
            remote_run = trackio.init(
                project=project,
                name=run_name,
                group=run_kind,
                config=config,
                embed=False,
                auto_log_gpu=auto_log_gpu,
                auto_log_cpu=auto_log_cpu,
            )
        except BaseException as error:
            context.fail(error)
            raise
        context.update_metadata(
            trackio_project=project,
            trackio_run_id=str(remote_run.id),
            trackio_run_name=str(remote_run.name),
        )
        context.event("tracking_started", project=project, trackio_run_id=str(remote_run.id))
        return cls(context=context, trackio_run=remote_run)

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        values = dict(metrics)
        self.trackio_run.log(values, step=step)
        self.context.event("metrics_logged", step=step, keys=sorted(values))

    def log_artifact(
        self,
        path: Path,
        *,
        name: str,
        artifact_type: str,
        aliases: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        artifact = trackio.Artifact(
            name=name,
            type=artifact_type,
            metadata=dict(metadata or {}),
        )
        if resolved.is_dir():
            artifact.add_dir(resolved)
        else:
            artifact.add_file(resolved)
        logged = self.trackio_run.log_artifact(artifact, aliases=list(aliases))
        self.context.event(
            "artifact_logged",
            artifact_name=name,
            artifact_type=artifact_type,
            path=str(resolved),
            aliases=list(aliases),
        )
        return logged

    def use_artifact(self, reference: str, *, artifact_type: str | None = None) -> Any:
        artifact = self.trackio_run.use_artifact(reference, type=artifact_type)
        self.context.event(
            "artifact_consumed",
            reference=reference,
            artifact_type=artifact_type,
        )
        return artifact

    def finish(self, status: str = "complete") -> None:
        if self._finished:
            return
        duration = time.perf_counter() - self.started_at
        self.log(
            {
                "run/duration_seconds": duration,
                "run/status": status,
                "run/success": 1 if status == "complete" else 0,
            }
        )
        if status == "complete":
            self.context.complete()
        else:
            self.context.update_metadata(status=status, finished_at=datetime.now(UTC).isoformat())
            self.context.event("run_finished", status=status)
        bundle = trackio.Artifact(
            name=f"{self.name}-run-bundle",
            type="run-bundle",
            metadata={"run_kind": self.context.run_kind, "status": status},
        )
        for filename in (
            "resolved-config.yaml",
            "resolved-profile.yaml",
            "metadata.json",
            "events.jsonl",
        ):
            path = self.context.root / filename
            if path.is_file():
                bundle.add_file(path)
        self.trackio_run.log_artifact(bundle, aliases=["latest"])
        self.trackio_run.finish()
        self._finished = True

    def fail(self, error: BaseException) -> None:
        if self._finished:
            return
        duration = time.perf_counter() - self.started_at
        self.context.fail(error)
        self.trackio_run.log(
            {
                "run/duration_seconds": duration,
                "run/status": "failed",
                "run/success": 0,
                "run/error_type": type(error).__name__,
            }
        )
        failure = self.context.output_dir / "failure.json"
        failure.write_text(
            json.dumps(
                {"error_type": type(error).__name__, "error": str(error)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.trackio_run.log_artifact(
            failure,
            name=f"{self.name}-failure",
            type="run-failure",
            aliases=["latest"],
        )
        self.trackio_run.finish()
        self._finished = True

    def __enter__(self) -> TrackedRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc_value is None:
            self.finish()
        else:
            self.fail(exc_value)
        return False


__all__ = ["TRACKING_SCHEMA_VERSION", "TrackedRun"]
