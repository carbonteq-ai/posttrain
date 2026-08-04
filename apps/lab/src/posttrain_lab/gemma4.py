"""Compatibility access to the framework Gemma 4 conversation contract."""

from __future__ import annotations

from posttrain.common.variants import GEMMA4_RENDERER_CONTRACT, RENDERER_CONTRACTS


def register_gemma4_renderer() -> None:
    """Preserve the former Lab registration hook as an idempotent shim."""

    existing = RENDERER_CONTRACTS.get(GEMMA4_RENDERER_CONTRACT.id)
    if existing is not None and existing != GEMMA4_RENDERER_CONTRACT:
        raise RuntimeError(f"renderer contract {GEMMA4_RENDERER_CONTRACT.id!r} is already registered differently")
    RENDERER_CONTRACTS[GEMMA4_RENDERER_CONTRACT.id] = GEMMA4_RENDERER_CONTRACT


__all__ = ["GEMMA4_RENDERER_CONTRACT", "register_gemma4_renderer"]
