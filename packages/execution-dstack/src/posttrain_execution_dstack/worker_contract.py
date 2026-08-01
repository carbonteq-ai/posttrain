"""Stable filesystem contract supplied by dstack worker provisioning."""

from pathlib import Path

DSTACK_WORKER_STATE_ROOT = Path("/var/lib/posttrain")
DSTACK_WORKER_RUN_ROOT = DSTACK_WORKER_STATE_ROOT / "runs"
DSTACK_WORKER_MODEL_CACHE = DSTACK_WORKER_STATE_ROOT / "cache" / "huggingface"
DSTACK_WORKER_COMPILE_CACHE = DSTACK_WORKER_STATE_ROOT / "cache" / "compile"

__all__ = [
    "DSTACK_WORKER_COMPILE_CACHE",
    "DSTACK_WORKER_MODEL_CACHE",
    "DSTACK_WORKER_RUN_ROOT",
    "DSTACK_WORKER_STATE_ROOT",
]
