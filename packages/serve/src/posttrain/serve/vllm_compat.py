"""Runtime activation hooks loaded in vLLM child processes."""

from __future__ import annotations

from typing import Any, cast


def apply_vllm_compatibility_patches() -> tuple[str, ...]:
    """Activate CUDA and repair the released TurboQuant marker defect.

    The general serve/eval image still pins the upstream vLLM 0.25.1 wheel.
    That build loses the requested TurboQuant dtype for hybrid full-attention
    layers because ``TQFullAttentionSpec`` reports no quantization mode.  The
    veRL image carries the source correction already, so this guarded patch is
    a no-op there.
    """

    applied: list[str] = []
    try:
        import torch
        from posttrain.common.cuda import TorchModule, activate_cuda_toolkit

        activate_cuda_toolkit(cast(TorchModule, torch))
        applied.append("cuda-toolkit")
    except Exception:
        # Plugin load must not crash vLLM when the optional CUDA toolkit is
        # absent; launch/offline paths still activate explicitly.
        pass

    try:
        from vllm.v1.kv_cache_interface import (  # pyright: ignore[reportMissingImports]
            KVQuantMode,
            TQFullAttentionSpec,
            get_kv_quant_mode,
        )
    except ImportError:
        return tuple(applied)

    if get_kv_quant_mode("turboquant_k8v4") == KVQuantMode.NONE:
        if not getattr(TQFullAttentionSpec, "_posttrain_quant_marker_patch", False):
            inherited_post_init: Any = TQFullAttentionSpec.__post_init__

            def tq_post_init(self: Any) -> None:
                inherited_post_init(self)
                if self.kv_quant_mode == KVQuantMode.NONE:
                    object.__setattr__(self, "kv_quant_mode", KVQuantMode.FP8_PER_TENSOR)

            spec_class: Any = TQFullAttentionSpec
            spec_class.__post_init__ = tq_post_init
            spec_class._posttrain_quant_marker_patch = True
        applied.append("turboquant-quant-marker")

    return tuple(applied)


__all__ = ["apply_vllm_compatibility_patches"]
