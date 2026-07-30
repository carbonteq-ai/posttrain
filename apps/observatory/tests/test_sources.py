"""Atomic dynamic source registry tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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
