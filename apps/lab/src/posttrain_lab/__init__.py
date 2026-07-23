"""Composition root for code-defined post-training jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .execution import RunSpec, execute_run, execute_run_tracked


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .execution import RunSpec, execute_run, execute_run_tracked

        return {
            "RunSpec": RunSpec,
            "execute_run": execute_run,
            "execute_run_tracked": execute_run_tracked,
        }[name]
    raise AttributeError(name)


__all__ = ["RunSpec", "execute_run", "execute_run_tracked"]
