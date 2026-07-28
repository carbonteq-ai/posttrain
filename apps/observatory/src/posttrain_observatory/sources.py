"""Multi-source registry and provider-neutral work-package projections."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping

from posttrain.tracking import RunDataSource, RunQuery

from .models import (
    JobDefinitionSummary,
    JobKindGroup,
    LocatedRunSummary,
    RunLocator,
    SourceSummary,
    WorkPackageRun,
    WorkPackageView,
)


def _contract_text(inputs: Mapping[str, object], contract: str, field: str) -> str | None:
    value = inputs.get(contract)
    if not isinstance(value, Mapping):
        return None
    text = value.get(field)
    if not isinstance(text, str):
        resolved = value.get("resolved")
        text = resolved.get(field) if isinstance(resolved, Mapping) else None
    return text.strip() if isinstance(text, str) and text.strip() else None


class RunSourceRegistry:
    def __init__(self, sources: Mapping[str, RunDataSource]) -> None:
        if not sources:
            raise ValueError("at least one Observatory source is required")
        if any(not source_id.strip() for source_id in sources):
            raise ValueError("source ids cannot be empty")
        self._sources = dict(sources)

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._sources))

    def resolve(self, locator: RunLocator) -> RunDataSource:
        try:
            return self._sources[locator.source_id]
        except KeyError as error:
            raise LookupError(f"Observatory source {locator.source_id!r} is not configured") from error

    async def sources(self) -> tuple[SourceSummary, ...]:
        async def probe(source_id: str, source: RunDataSource) -> SourceSummary:
            try:
                # One run is enough to prove the provider answers. Trackio's
                # bulk list path makes that cheap; do not raise the limit here
                # or /api/v1/sources pays for a full project inventory again.
                await source.list_runs(RunQuery(limit=1))
            except Exception as error:  # source isolation is deliberate at this boundary
                return SourceSummary(
                    source_id=source_id,
                    provider=source.capabilities.provider,
                    state="unavailable",
                    message=str(error),
                    capabilities=source.capabilities,
                )
            return SourceSummary(
                source_id=source_id,
                provider=source.capabilities.provider,
                state="healthy",
                capabilities=source.capabilities,
            )

        return tuple(
            await asyncio.gather(*(probe(source_id, source) for source_id, source in sorted(self._sources.items())))
        )

    async def list_runs(self, query: RunQuery) -> tuple[LocatedRunSummary, ...]:
        async def load(source_id: str, source: RunDataSource) -> tuple[LocatedRunSummary, ...]:
            try:
                values = await source.list_runs(query)
            except Exception:
                return ()
            return tuple(
                LocatedRunSummary(
                    locator=(locator := RunLocator(source_id=source_id, run_id=run.run_id)),
                    run_key=locator.key,
                    run=run,
                )
                for run in values
            )

        groups = await asyncio.gather(*(load(source_id, source) for source_id, source in sorted(self._sources.items())))
        merged = [item for group in groups for item in group]
        merged.sort(key=lambda item: item.run.started_at, reverse=True)
        return tuple(merged[: query.limit])

    async def locate_run(self, run_id: str) -> tuple[LocatedRunSummary, ...]:
        """Resolve one canonical run identity without scanning source histories."""

        async def load(
            source_id: str,
            source: RunDataSource,
        ) -> LocatedRunSummary | None:
            try:
                detail = await source.get_run(run_id)
            except Exception:
                return None
            locator = RunLocator(source_id=source_id, run_id=run_id)
            return LocatedRunSummary(
                locator=locator,
                run_key=locator.key,
                run=detail.summary,
            )

        values = await asyncio.gather(*(load(source_id, source) for source_id, source in sorted(self._sources.items())))
        return tuple(item for item in values if item is not None)

    async def work_package_view(
        self,
        work_package_id: str,
        *,
        project_id: str | None = None,
        source_id: str | None = None,
    ) -> WorkPackageView:
        located = await self.list_runs(RunQuery(project_id=project_id, work_package_id=work_package_id, limit=1000))
        if source_id is not None:
            self.resolve(RunLocator(source_id=source_id, run_id="source-probe"))
            located = tuple(item for item in located if item.locator.source_id == source_id)
        runs: list[WorkPackageRun] = []
        lineage = []
        package_description: str | None = None
        project_ids = {item.run.project_id for item in located}
        if project_id is None and len(project_ids) > 1:
            raise ValueError(f"work package {work_package_id!r} exists in multiple projects; provide project_id")
        for item in located:
            source = self.resolve(item.locator)
            detail, artifacts = await asyncio.gather(
                source.get_run(item.locator.run_id), source.artifacts(item.locator.run_id)
            )
            runs.append(
                WorkPackageRun(
                    locator=item.locator,
                    run_key=item.run_key,
                    run=item.run,
                    metric_names=detail.metric_names,
                    job_definition_description=_contract_text(detail.resolved_inputs, "job_definition", "description"),
                )
            )
            package_description = package_description or _contract_text(
                detail.resolved_inputs, "work_package", "description"
            )
            lineage.extend((item.locator, link) for link in artifacts.items)
        grouped: dict[str, list[WorkPackageRun]] = defaultdict(list)
        for run in runs:
            grouped[run.run.job_kind].append(run)
        return WorkPackageView(
            project_id=project_id or (next(iter(project_ids)) if project_ids else None),
            work_package_id=work_package_id,
            description=package_description,
            runs=tuple(runs),
            job_groups=tuple(
                JobKindGroup(
                    job_kind=kind,
                    run_keys=tuple(run.run_key for run in values),
                    statuses=tuple(run.run.status for run in values),
                    definitions=tuple(
                        JobDefinitionSummary(id=definition_id, description=description)
                        for definition_id, description in {
                            run.run.job_definition_version: run.job_definition_description for run in values
                        }.items()
                    ),
                )
                for kind, values in sorted(grouped.items())
            ),
            lineage=tuple(lineage),
        )


__all__ = ["RunSourceRegistry"]
