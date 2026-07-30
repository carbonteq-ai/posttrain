"""Runtime activation hooks loaded in vLLM child processes."""

from __future__ import annotations

from typing import cast


def apply_vllm_compatibility_patches() -> tuple[str, ...]:
    """Activate the pip CUDA toolkit view for vLLM child processes."""

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

    return tuple(applied)


__all__ = ["apply_vllm_compatibility_patches"]
