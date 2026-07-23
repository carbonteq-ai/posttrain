"""Small reference job used to prove the composition boundary."""

from __future__ import annotations

from posttrain.common import RunContext


def run_noop(context: RunContext) -> str:
    context.metric("noop/completed", 1)
    return "ok"
