from pathlib import Path

from posttrain_execution_dstack import (
    DSTACK_WORKER_COMPILE_CACHE,
    DSTACK_WORKER_MODEL_CACHE,
    DSTACK_WORKER_RUN_ROOT,
    DSTACK_WORKER_STATE_ROOT,
)


def test_worker_storage_is_adapter_owned_and_stable() -> None:
    assert DSTACK_WORKER_STATE_ROOT == Path("/var/lib/posttrain")
    assert DSTACK_WORKER_RUN_ROOT == DSTACK_WORKER_STATE_ROOT / "runs"
    assert DSTACK_WORKER_MODEL_CACHE == DSTACK_WORKER_STATE_ROOT / "cache/huggingface"
    assert DSTACK_WORKER_COMPILE_CACHE == DSTACK_WORKER_STATE_ROOT / "cache/compile"
