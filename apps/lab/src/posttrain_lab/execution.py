"""Compatibility exports for generic run execution now owned by `posttrain.work`."""

from posttrain.work.execution import (
    ArtifactInput,
    ArtifactMaterializer,
    RunOperation,
    RunSpec,
    execute_run,
    execute_run_tracked,
)

__all__ = [
    "ArtifactInput",
    "ArtifactMaterializer",
    "RunOperation",
    "RunSpec",
    "execute_run",
    "execute_run_tracked",
]
