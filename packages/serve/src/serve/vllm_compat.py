"""Narrow, state-guarded fixes for released vLLM integration defects."""

from __future__ import annotations

from typing import Any


def apply_vllm_compatibility_patches() -> tuple[str, ...]:
    """Apply only defects detectable in the imported vLLM build.

    vLLM 0.25.1 creates ``TQFullAttentionSpec`` with the correct packed page
    size but leaves ``kv_quant_mode`` at ``NONE``. The GPU runner consequently
    replaces its TurboQuant dtype with ``auto`` during cache reshaping. Until
    upstream adds a dedicated TurboQuant mode, FP8_PER_TENSOR is used strictly
    as the existing non-per-token "quantized" marker; TQ's overridden page-size
    calculation and backend remain unchanged.
    """

    from vllm.v1.kv_cache_interface import (
        KVQuantMode,
        TQFullAttentionSpec,
        get_kv_quant_mode,
    )

    if get_kv_quant_mode("turboquant_k8v4") != KVQuantMode.NONE:
        return ()
    if getattr(TQFullAttentionSpec, "_lab_quant_marker_patch", False):
        return ("turboquant-quant-marker",)

    inherited_post_init: Any = TQFullAttentionSpec.__post_init__

    def tq_post_init(self: Any) -> None:
        inherited_post_init(self)
        if self.kv_quant_mode == KVQuantMode.NONE:
            object.__setattr__(self, "kv_quant_mode", KVQuantMode.FP8_PER_TENSOR)

    TQFullAttentionSpec.__post_init__ = tq_post_init
    TQFullAttentionSpec._lab_quant_marker_patch = True
    return ("turboquant-quant-marker",)


__all__ = ["apply_vllm_compatibility_patches"]
