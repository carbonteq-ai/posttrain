"""The semantic layer's reads, served by the Observatory's sources and views."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from posttrain.tracking import MetricSeries, RunDetail, RunQuery, RunSummary, TraceAggregateResult, TraceFactsQuery
from posttrain.tracking.models import TraceFactAggregate

if TYPE_CHECKING:
    from ..service import ObservatoryService


class ServiceReader:
    """Implements `SemanticReader` over an `ObservatoryService`."""

    def __init__(self, service: ObservatoryService) -> None:
        self._service = service

    def _source(self, run_id: str) -> Any:
        return self._service.registry.resolve(self._service._locator(run_id))  # noqa: SLF001

    async def list_runs(self, query: RunQuery) -> tuple[RunSummary, ...]:
        located = await self._service.registry.list_runs(query)
        return tuple(item.run for item in located)

    async def run_detail(self, run_id: str) -> RunDetail:
        return await self._source(run_id).get_run(run_id)

    async def metric_series(
        self, run_id: str, names: tuple[str, ...], *, max_points: int
    ) -> tuple[tuple[MetricSeries, ...], bool]:
        series = await self._source(run_id).metric_series(run_id, names)
        downsampled = False
        result = []
        for item in series:
            points = tuple(point for point in item.points if point.step is not None)
            if len(points) > max_points:
                stride = -(-len(points) // max_points)
                points = (*points[::stride][: max_points - 1], points[-1])
                downsampled = True
            result.append(item.model_copy(update={"points": points}))
        return tuple(result), downsampled

    def trace_facts_available(self, run_id: str) -> bool:
        return self._source(run_id).capabilities.trace_facts == "available"

    async def trace_facts(self, run_id: str, query: TraceFactsQuery) -> TraceAggregateResult:
        return await self._source(run_id).aggregate_trace_facts(run_id, query)

    async def eval_tasks(self, run_id: str) -> tuple[Mapping[str, Any], ...]:
        view = await self._service.get_trace_evaluation_view(run_id, include_traces=False)
        if view.measurement is not None and view.measurement.tasks:
            return tuple(task.model_dump(mode="json") for task in view.measurement.tasks)
        return await self._eval_tasks_from_trace_facts(run_id)

    async def _eval_tasks_from_trace_facts(self, run_id: str) -> tuple[Mapping[str, Any], ...]:
        """Evaluations without a task manifest: per-task figures from rollout facts."""

        if not self.trace_facts_available(run_id):
            return ()
        result = await self.trace_facts(
            run_id,
            TraceFactsQuery(
                group_by=("task_id", "is_truncated", "has_error"),
                aggregates=(
                    TraceFactAggregate(measure="task_reward", operation="sum"),
                    TraceFactAggregate(measure="task_reward", operation="count"),
                ),
            ),
        )
        tasks: dict[str, dict[str, Any]] = {}
        for bucket in result.buckets:
            task = str(bucket.dimensions.get("task_id"))
            row = tasks.setdefault(
                task,
                {
                    "key": task,
                    "label": task,
                    "reward_sum": 0.0,
                    "valid_repetitions": 0,
                    "execution_failures": 0,
                    "truncations": 0,
                },
            )
            if bucket.dimensions.get("has_error"):
                row["execution_failures"] += bucket.trace_count
                continue
            if bucket.dimensions.get("is_truncated"):
                row["truncations"] += bucket.trace_count
            row["reward_sum"] += bucket.values.get("sum_task_reward") or 0.0
            row["valid_repetitions"] += int(bucket.values.get("count_task_reward") or 0)
        rows = []
        for row in tasks.values():
            valid = row["valid_repetitions"]
            rows.append({**row, "mean_reward": row["reward_sum"] / valid if valid else None, "success_frequency": None})
        return tuple(rows)

    async def load_levels(self, run_id: str) -> tuple[Mapping[str, Any], ...]:
        response = await self._service.get_run_view_response(run_id)
        points = getattr(response.view, "operating_points", None) or ()
        return tuple(point.model_dump(mode="json") for point in points)


__all__ = ["ServiceReader"]
