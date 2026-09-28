"""Ray worker setup hook: start veRL's fp16 ShardedGradScaler at the selected scale.

veRL 0.9.0.post5's and post7's FSDP engine creates ``ShardedGradScaler(growth_interval=400)``
inside its Ray actors, so the starting scale cannot be passed through its
configuration. The worker passes ``POSTTRAIN_FP16_INITIAL_LOSS_SCALE`` and this
hook in the Ray runtime environment; the hook runs in every Ray worker process
before veRL code and makes the scale the constructor's default. An explicit
``init_scale`` argument still wins, and growth keeps veRL's settings.
"""

from __future__ import annotations

import functools
import os
from typing import Any

from ...precision import FP16_INITIAL_LOSS_SCALE_ENV

_PATCHED = "_posttrain_initial_loss_scale"


def configure_initial_loss_scale() -> None:
    raw = os.environ.get(FP16_INITIAL_LOSS_SCALE_ENV)
    if raw is None:
        return
    scale = float(raw)
    if not scale > 0:
        raise ValueError(f"{FP16_INITIAL_LOSS_SCALE_ENV} must be positive, got {raw!r}")
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

    original = getattr(ShardedGradScaler.__init__, _PATCHED, None) or ShardedGradScaler.__init__

    @functools.wraps(original)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        # init_scale is the second positional parameter (after device).
        if len(args) < 2:
            kwargs.setdefault("init_scale", scale)
        original(self, *args, **kwargs)

    setattr(__init__, _PATCHED, original)
    ShardedGradScaler.__init__ = __init__  # type: ignore[method-assign]


__all__ = ["configure_initial_loss_scale"]
