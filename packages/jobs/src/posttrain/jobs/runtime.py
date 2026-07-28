"""Default project runtime construction."""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import partial
from typing import Literal, cast

from posttrain.data import DatasetLoadPlan, resolve_dataset_source
from posttrain.work import (
    JobDefinition,
    JobRuntime,
    ProjectExecutionRequest,
    ResolvedSeat,
    execute_run,
    execute_run_tracked_finalized,
)

from .definitions import standard_definitions


def build_job_runtime(
    request: ProjectExecutionRequest,
    *,
    tracking: str | None = None,
    extra_definitions: Mapping[str, JobDefinition] | None = None,
) -> JobRuntime:
    """Build the standard runtime for one discovered project."""

    definitions = standard_definitions()
    extras = dict(extra_definitions or {})
    shadowed = set(definitions).intersection(extras)
    if shadowed:
        raise ValueError(f"extra job definitions cannot shadow standard ids: {', '.join(sorted(shadowed))}")
    definitions.update(extras)
    scratch_root = request.state_dir / "scratch"
    selected_tracking = tracking or "trackio"
    if selected_tracking == "none":
        executor = partial(execute_run, scratch_root=scratch_root)
    elif selected_tracking == "trackio":
        from posttrain_tracking_trackio import TrackioBackend, TrackioSettings

        backend = TrackioBackend(
            TrackioSettings(
                project=os.getenv("POSTTRAIN_TRACKIO_PROJECT", request.project_id),
                server_url=os.getenv("POSTTRAIN_TRACKIO_SERVER_URL"),
                auto_log_gpu=True,
                auto_log_cpu=True,
            )
        )
        executor = partial(
            execute_run_tracked_finalized,
            backend=backend,
            scratch_root=scratch_root,
        )
    elif selected_tracking == "wandb":
        from posttrain_tracking_wandb import WandbBackend, WandbSettings

        entity = os.getenv("WANDB_ENTITY")
        if not entity:
            raise ValueError("W&B tracking requires WANDB_ENTITY")
        mode = os.getenv("WANDB_MODE", "online")
        if mode not in {"online", "offline"}:
            raise ValueError("WANDB_MODE must be online or offline")
        backend = WandbBackend(
            WandbSettings(
                entity=entity,
                project=os.getenv("WANDB_PROJECT", request.project_id),
                base_url=os.getenv("WANDB_BASE_URL"),
                mode=cast(Literal["online", "offline"], mode),
            )
        )
        executor = partial(
            execute_run_tracked_finalized,
            backend=backend,
            scratch_root=scratch_root,
        )
    else:
        raise ValueError(f"unknown tracking backend: {selected_tracking}")

    def resolve_seat(seat: ResolvedSeat):
        value = seat.value
        if isinstance(value, DatasetLoadPlan):
            return resolve_dataset_source(
                value,
                state_dir=request.state_dir,
                project_root=request.project_root,
            )
        return value

    return JobRuntime(
        catalog=request.catalog,
        definitions=definitions,
        project_brief=request.project_brief,
        source_metadata={
            "project_root": str(request.project_root),
            "tracking_backend": selected_tracking,
        },
        executor=executor,
        seat_resolver=resolve_seat,
    )


__all__ = ["build_job_runtime"]
