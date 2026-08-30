"""Stable worker runtime for verified post-training jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execute import execute_manifest, qualify_manifest


def __getattr__(name: str) -> object:
    """Keep package import CUDA-neutral while preserving the public helpers."""

    if name == "execute_manifest":
        from .execute import execute_manifest

        return execute_manifest
    if name == "qualify_manifest":
        from .execute import qualify_manifest

        return qualify_manifest
    raise AttributeError(name)

__all__ = ["execute_manifest", "qualify_manifest"]
