"""Atomic dynamic source registry tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from posttrain.tracking import RunQuery
from posttrain_observatory import FixtureRunDataSource, RunLocator
from posttrain_observatory.sources import RunSourceRegistry


def test_discovered_reconciliation_preserves_configured_sources_and_collisions() -> None:
    configured = FixtureRunDataSource()
    registry = RunSourceRegistry({"manual": configured})

    installed = registry.reconcile_discovered(
        {
            "manual": FixtureRunDataSource(),
            "alpha": FixtureRunDataSource(),
        }
    )

    assert installed == ("alpha",)
    assert registry.source_ids == ("alpha", "manual")
    assert registry.resolve(RunLocator(source_id="manual", run_id="probe")) is configured


def test_concurrent_readers_observe_only_complete_registry_snapshots() -> None:
    registry = RunSourceRegistry({"manual": FixtureRunDataSource()})
    old = {"alpha": FixtureRunDataSource(), "beta": FixtureRunDataSource()}
    new = {"gamma": FixtureRunDataSource(), "delta": FixtureRunDataSource()}
    registry.reconcile_discovered(old)
    allowed = {
        ("alpha", "beta", "manual"),
        ("delta", "gamma", "manual"),
    }

    def swap() -> None:
        for index in range(2_000):
            registry.reconcile_discovered(old if index % 2 == 0 else new)

    def read() -> set[tuple[str, ...]]:
        return {registry.source_ids for _ in range(5_000)}

    with ThreadPoolExecutor(max_workers=4) as executor:
        writer = executor.submit(swap)
        readers = [executor.submit(read) for _ in range(3)]
        writer.result()
        observed = set().union(*(reader.result() for reader in readers))

    assert observed
    assert observed <= allowed


@pytest.mark.asyncio
async def test_source_scoped_run_list_and_direct_run_do_not_scan_other_sources() -> None:
    class CountingFixture(FixtureRunDataSource):
        def __init__(self) -> None:
            super().__init__()
            self.list_calls = 0
            self.get_calls = 0

        async def list_runs(self, query: RunQuery):
            self.list_calls += 1
            return await super().list_runs(query)

        async def get_run(self, run_id: str):
            self.get_calls += 1
            return await super().get_run(run_id)

    ambient = CountingFixture()
    other = CountingFixture()
    registry = RunSourceRegistry({"ambient-agent": ambient, "posttrain-lab": other})

    listed = await registry.list_runs(RunQuery(limit=10), source_id="ambient-agent")
    assert listed
    assert ambient.list_calls == 1
    assert other.list_calls == 0

    resolved = await registry.get_run(listed[0].locator)
    assert resolved.locator == listed[0].locator
    assert ambient.get_calls == 1
    assert other.get_calls == 0


@pytest.mark.asyncio
async def test_source_health_is_cached_until_discovery_changes() -> None:
    class CountingFixture(FixtureRunDataSource):
        def __init__(self) -> None:
            super().__init__()
            self.list_calls = 0

        async def list_runs(self, query: RunQuery):
            self.list_calls += 1
            return await super().list_runs(query)

    configured = CountingFixture()
    discovered = CountingFixture()
    registry = RunSourceRegistry({"manual": configured})

    assert await registry.sources() == await registry.sources()
    assert configured.list_calls == 1

    registry.reconcile_discovered({"discovered": discovered})
    assert len(await registry.sources()) == 2
    assert configured.list_calls == 2
    assert discovered.list_calls == 1
