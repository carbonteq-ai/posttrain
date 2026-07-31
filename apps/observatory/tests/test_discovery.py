"""Trackio discovery lifecycle tests."""

from __future__ import annotations

import asyncio
import threading

import pytest
from posttrain_observatory import FixtureRunDataSource
from posttrain_observatory.discovery import TrackioSourceDiscovery
from posttrain_observatory.sources import RunSourceRegistry


class Catalog:
    def __init__(self, *responses: tuple[str, ...] | Exception) -> None:
        self.responses = list(responses)
        self.calls = 0

    def list_projects(self) -> tuple[str, ...]:
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_success_removes_missing_projects_but_failure_retains_last_snapshot() -> None:
    registry = RunSourceRegistry({"manual": FixtureRunDataSource()})
    catalog = Catalog(
        ("alpha", "beta"),
        ("beta",),
        RuntimeError("Trackio unavailable at https://token@trackio.example/api\nsecret detail"),
    )
    discovery = TrackioSourceDiscovery(
        registry,
        catalog,  # type: ignore[arg-type]
        lambda _: FixtureRunDataSource(),
        interval_seconds=300,
    )

    first = await discovery.refresh()
    assert first.state == "succeeded"
    assert registry.source_ids == ("alpha", "beta", "manual")

    second = await discovery.refresh()
    assert second.discovered_source_ids == ("beta",)
    assert registry.source_ids == ("beta", "manual")

    failed = await discovery.refresh()
    assert failed.state == "failed"
    assert failed.error == "Trackio unavailable at <Trackio server>"
    assert failed.last_success_at == second.last_success_at
    assert registry.source_ids == ("beta", "manual")


@pytest.mark.asyncio
async def test_start_refreshes_once_and_stop_cancels_periodic_wait() -> None:
    waiting = asyncio.Event()
    sleeps = 0

    async def sleeper(interval: float) -> None:
        nonlocal sleeps
        assert interval == 12
        sleeps += 1
        if sleeps == 1:
            return
        waiting.set()
        await asyncio.Event().wait()

    registry = RunSourceRegistry({})
    catalog = Catalog(("alpha",), ("beta",))
    discovery = TrackioSourceDiscovery(
        registry,
        catalog,  # type: ignore[arg-type]
        lambda _: FixtureRunDataSource(),
        interval_seconds=12,
        sleeper=sleeper,
    )

    await discovery.start()
    await asyncio.wait_for(waiting.wait(), timeout=1)
    await discovery.stop()

    assert catalog.calls == 2
    assert registry.source_ids == ("beta",)


@pytest.mark.asyncio
async def test_concurrent_manual_refreshes_are_serialized() -> None:
    class BlockingCatalog:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()
            self.lock = threading.Lock()
            self.active = 0
            self.maximum_active = 0
            self.calls = 0

        def list_projects(self) -> tuple[str, ...]:
            with self.lock:
                self.calls += 1
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            self.entered.set()
            assert self.release.wait(timeout=2)
            with self.lock:
                self.active -= 1
            return ("alpha",)

    catalog = BlockingCatalog()
    discovery = TrackioSourceDiscovery(
        RunSourceRegistry({}),
        catalog,  # type: ignore[arg-type]
        lambda _: FixtureRunDataSource(),
        interval_seconds=300,
    )

    first = asyncio.create_task(discovery.refresh())
    assert await asyncio.to_thread(catalog.entered.wait, 2)
    second = asyncio.create_task(discovery.refresh())
    await asyncio.sleep(0)
    catalog.release.set()
    await asyncio.gather(first, second)

    assert catalog.calls == 2
    assert catalog.maximum_active == 1
