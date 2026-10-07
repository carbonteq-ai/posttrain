"""Spill saved activations to host memory once device memory passes a budget.

Resolved SAMPO updates score every pack at fixed parameters and keep each
pack's autograd graph until one backward, so device memory grows with the
update's total scored context. Under this context, a tensor that autograd saves
for backward is copied to pinned host memory instead of being kept on the
device when it is large, is not a parameter, and the device already holds more
than the budget. Backward copies it back. Values and gradients are unchanged;
only where saved activations live while the graph waits changes.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OffloadStats:
    """Bytes and tensors moved to host memory under one context."""

    bytes: int = 0
    tensors: int = 0
    peak_device_bytes: int = field(default=0)


@contextlib.contextmanager
def offload_overflow(
    model: Any,
    budget_bytes: int,
    *,
    min_bytes: int = 4 << 20,
) -> Iterator[OffloadStats]:
    """Keep saved activations on the device up to ``budget_bytes``; spill larger overflow to host."""

    import torch

    if type(budget_bytes) is not int or budget_bytes < 0 or type(min_bytes) is not int or min_bytes < 1:
        raise ValueError("activation offload requires a nonnegative byte budget and a positive minimum size")
    parameters = {parameter.data_ptr() for parameter in model.parameters()}
    stats = OffloadStats()

    def pack(tensor: Any) -> Any:
        if (
            tensor.is_cuda
            and tensor.numel() * tensor.element_size() >= min_bytes
            and tensor.untyped_storage().data_ptr() not in parameters
        ):
            allocated = torch.cuda.memory_allocated(tensor.device)
            stats.peak_device_bytes = max(stats.peak_device_bytes, allocated)
            if allocated > budget_bytes:
                host = torch.empty(tensor.shape, dtype=tensor.dtype, layout=tensor.layout, pin_memory=True)
                host.copy_(tensor)
                stats.bytes += host.numel() * host.element_size()
                stats.tensors += 1
                return (tensor.device, host)
        return tensor

    def unpack(packed: Any) -> Any:
        if isinstance(packed, tuple):
            device, host = packed
            return host.to(device, non_blocking=True)
        return packed

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield stats


__all__ = ["OffloadStats", "offload_overflow"]
