"""Dynamic Trackio source discovery and refresh lifecycle."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from posttrain.tracking import RunDataSource
from posttrain_tracking_trackio import TrackioProjectCatalog

from .models import SourceRefreshStatus
from .sources import RunSourceRegistry

type SourceFactory = Callable[[str], RunDataSource]
type Clock = Callable[[], datetime]
type Sleeper = Callable[[float], Awaitable[None]]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_error(error: Exception) -> str:
    message = str(error).splitlines()[0].strip() or type(error).__name__
    message = re.sub(r"https?://[^\s'\"()]+", "<Trackio server>", message)
    return message[:500]


class TrackioSourceDiscovery:
    """Reconcile a remote Trackio project catalog into one atomic registry layer."""

    def __init__(
        self,
        registry: RunSourceRegistry,
        catalog: TrackioProjectCatalog,
        source_factory: SourceFactory,
        *,
        interval_seconds: int,
        clock: Clock = _utc_now,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        if interval_seconds < 1:
            raise ValueError("Trackio discovery interval must be at least one second")
        self._registry = registry
        self._catalog = catalog
        self._source_factory = source_factory
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._sleeper = sleeper
        self._refresh_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._status = SourceRefreshStatus(enabled=True, state="pending")

    def status(self) -> SourceRefreshStatus:
        return self._status

    def _load_sources(self) -> dict[str, RunDataSource]:
        return {
            project: self._source_factory(project)
            for project in self._catalog.list_projects()
        }

    async def refresh(self) -> SourceRefreshStatus:
        async with self._refresh_lock:
            attempted_at = self._clock()
            self._status = self._status.model_copy(
                update={
                    "state": "refreshing",
                    "last_attempt_at": attempted_at,
                    "error": None,
                }
            )
            try:
                sources = await asyncio.to_thread(self._load_sources)
                source_ids = self._registry.reconcile_discovered(sources)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # backend failures are status at this boundary
                self._status = self._status.model_copy(
                    update={
                        "state": "failed",
                        "error": _safe_error(error),
                    }
                )
            else:
                self._status = SourceRefreshStatus(
                    enabled=True,
                    state="succeeded",
                    last_attempt_at=attempted_at,
                    last_success_at=self._clock(),
                    discovered_source_ids=source_ids,
                )
            return self._status

    async def run_periodically(self) -> None:
        while True:
            await self._sleeper(float(self._interval_seconds))
            await self.refresh()

    async def start(self) -> None:
        if self._task is not None:
            return
        await self.refresh()
        self._task = asyncio.create_task(self.run_periodically(), name="observatory-trackio-discovery")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


__all__ = ["TrackioSourceDiscovery"]
