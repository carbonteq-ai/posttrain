from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from posttrain.execution import (
    ExecutionHandle,
    ExecutionProviderPurgeExecutor,
    ExecutionRecord,
    ProviderCleanupDeferred,
    PurgeAction,
    PurgeActionDeferred,
)


def test_provider_purge_executor_revalidates_and_cleans_exact_run() -> None:
    calls: list[str] = []

    class Service:
        def submission(self, run_id: str):
            return SimpleNamespace(provider="local-docker")

        def status(self, run_id: str):
            calls.append(f"status:{run_id}")
            return ExecutionRecord(
                handle=ExecutionHandle("local-docker", "container-1", "idem"),
                state="succeeded",
                attempt=1,
                target_id="local",
                observed_at=datetime.now(UTC),
                native_state="exited",
            )

        def cleanup(self, run_id: str):
            calls.append(f"cleanup:{run_id}")

    action = PurgeAction(
        action_id="provider:run-1",
        plane="provider",
        kind="provider.cleanup",
        target={"provider": "local-docker", "provider_id": "container-1", "run_id": "run-1"},
    )
    executor = ExecutionProviderPurgeExecutor({"run-1": Service()})  # type: ignore[arg-type]
    executor.revalidate(action)
    executor.apply(action)
    assert calls == ["status:run-1", "cleanup:run-1"]


def test_provider_purge_executor_maps_busy_worker_to_deferred_action() -> None:
    class Service:
        def submission(self, run_id: str):
            return SimpleNamespace(provider="dstack")

        def status(self, run_id: str):
            return ExecutionRecord(
                handle=ExecutionHandle("dstack", "native-1", "idem"),
                state="succeeded",
                attempt=1,
                target_id="gpu-worker-a",
                observed_at=datetime.now(UTC),
                native_state="done",
            )

        def cleanup(self, run_id: str):
            raise ProviderCleanupDeferred("exact worker is occupied")

    action = PurgeAction(
        action_id="provider:run-1",
        plane="provider",
        kind="provider.cleanup",
        target={"provider": "dstack", "provider_id": "native-1", "run_id": "run-1"},
    )
    executor = ExecutionProviderPurgeExecutor({"run-1": Service()})  # type: ignore[arg-type]

    with pytest.raises(PurgeActionDeferred, match="occupied"):
        executor.apply(action)
