"""Environment-worker memory is measured over process trees and bounded by a budget."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest
from posttrain.train.integrations.verifiers_workers import VerifiersWorkerPool, process_tree_rss_bytes
from posttrain.train.rollout_execution import RolloutExecutionConfig

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")


def _sleeper(*args):
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", *args])


def test_tree_rss_includes_children_and_skips_exited_processes():
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)",
        ]
    )
    try:
        for _ in range(50):
            alone = process_tree_rss_bytes(parent.pid)
            with open(f"/proc/{parent.pid}/task/{parent.pid}/children") as stream:
                if stream.read().split():
                    break
            subprocess.run(["sleep", "0.1"], check=True)
        assert process_tree_rss_bytes(parent.pid) > 0
        assert process_tree_rss_bytes(parent.pid) >= alone
    finally:
        subprocess.run(["pkill", "-P", str(parent.pid)], check=False)
        parent.kill()
        parent.wait()
    assert process_tree_rss_bytes(parent.pid) == 0


def test_budget_breach_fails_and_tears_down_the_pool():
    worker = _sleeper()

    async def scenario():
        pool = object.__new__(VerifiersWorkerPool)
        pool._pool = SimpleNamespace(workers=[{"process": worker}])
        pool._pool_task = asyncio.create_task(asyncio.sleep(60))
        pool._closing = False
        pool._fatal_error = None
        pool.peak_worker_rss_bytes = 0
        await asyncio.wait_for(pool._watch_memory(1e-6, interval=0.01), timeout=5)
        await asyncio.gather(pool._pool_task, return_exceptions=True)
        return pool

    try:
        pool = asyncio.run(scenario())
    finally:
        worker.kill()
        worker.wait()
    assert pool._pool_task.cancelled()
    assert "memory_budget_gb" in str(pool._fatal_error)
    assert pool.peak_worker_rss_bytes > 0


def test_no_budget_only_tracks_the_peak():
    worker = _sleeper()

    async def scenario():
        pool = object.__new__(VerifiersWorkerPool)
        pool._pool = SimpleNamespace(workers=[{"process": worker}])
        pool._pool_task = asyncio.create_task(asyncio.sleep(60))
        pool._closing = False
        pool._fatal_error = None
        pool.peak_worker_rss_bytes = 0
        watcher = asyncio.create_task(pool._watch_memory(None, interval=0.01))
        await asyncio.sleep(0.2)
        pool._closing = True
        await asyncio.wait_for(watcher, timeout=5)
        pool._pool_task.cancel()
        return pool

    try:
        pool = asyncio.run(scenario())
    finally:
        worker.kill()
        worker.wait()
    assert pool._fatal_error is None and pool.peak_worker_rss_bytes > 0


def test_memory_budget_must_be_positive():
    assert RolloutExecutionConfig(2, 3, memory_budget_gb=32).memory_budget_gb == 32
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            RolloutExecutionConfig(2, 3, memory_budget_gb=bad)
