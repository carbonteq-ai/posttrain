"""Run-level rollout phase time reuses finished steps and re-reads only changed ones."""

from __future__ import annotations

import asyncio

from posttrain.tracking import TraceAggregateBucket, TraceAggregateResult, TraceFactsQuery, TracePayloadQuery
from posttrain_observatory.rollout_time import refresh_rollout_time_buckets, rollout_time_view


def _bucket(step: int | None, count: int, inference_s: float) -> TraceAggregateBucket:
    keys = ("inference_s", "tools_s", "setup_s", "scoring_s", "rollout_s")
    values = {
        "inference_s": inference_s,
        "tools_s": 1.0,
        "setup_s": 0.5,
        "scoring_s": 0.0,
        "rollout_s": inference_s + 1.5,
    }
    return TraceAggregateBucket(
        dimensions={"rollout_step": step},
        trace_count=count,
        values={key: values[key] for key in keys},
        coverage={key: count for key in keys},
    )


class Provider:
    def __init__(self, steps: dict[int | None, TraceAggregateBucket]) -> None:
        self.steps = steps
        self.payload_queries: list[TracePayloadQuery] = []

    async def payload(self, run_id: str, query: TracePayloadQuery) -> TraceAggregateResult:
        del run_id
        self.payload_queries.append(query)
        if query.group_by:
            return TraceAggregateResult(state="available", buckets=tuple(self.steps.values()))
        step = query.dimensions["rollout_step"]
        bucket = self.steps.get(step)  # type: ignore[arg-type]
        return TraceAggregateResult(state="available", buckets=(bucket,) if bucket else ())

    async def facts(self, run_id: str, query: TraceFactsQuery) -> TraceAggregateResult:
        del run_id, query
        return TraceAggregateResult(
            state="available",
            buckets=tuple(
                TraceAggregateBucket(dimensions={"rollout_step": step}, trace_count=bucket.trace_count)
                for step, bucket in self.steps.items()
                if step is not None
            ),
        )


def _refresh(provider: Provider, cached):
    return asyncio.run(
        refresh_rollout_time_buckets(
            "run", "verifiers", cached, aggregate_payload=provider.payload, aggregate_facts=provider.facts
        )
    )


def test_first_load_is_one_grouped_query_and_refresh_reads_only_changed_steps() -> None:
    provider = Provider({21: _bucket(21, 64, 100.0), 22: _bucket(22, 30, 40.0), None: _bucket(None, 5, 9.0)})
    first = _refresh(provider, None)
    assert [query.group_by for query in provider.payload_queries] == [("rollout_step",)]

    provider.payload_queries.clear()
    provider.steps[22] = _bucket(22, 64, 90.0)
    provider.steps[None] = _bucket(None, 2, 3.0)
    second = _refresh(provider, first)
    assert first is not None and second is not None

    assert [query.dimensions for query in provider.payload_queries] == [{"rollout_step": 22}, {"rollout_step": None}]
    assert second[21] is first[21]
    assert second[22].trace_count == 64
    assert second[None].trace_count == 2


def test_view_orders_steps_and_reports_milliseconds() -> None:
    view = rollout_time_view(
        {None: _bucket(None, 2, 3.0), 22: _bucket(22, 64, 90.0), 21: _bucket(21, 64, 100.0)}, live=True
    )

    assert [step.step for step in view.steps] == [21, 22, None]
    assert view.steps[0].inference_ms == 100_000.0
    assert view.steps[0].tools_ms == 1000.0
    assert view.live is True
    assert rollout_time_view(None, live=False).state == "unavailable"


def test_elapsed_time_is_the_union_of_step_spans_not_summed_rollout_time() -> None:
    def spanned(step: int | None, start: float, end: float) -> TraceAggregateBucket:
        bucket = _bucket(step, 64, 100.0)
        return bucket.model_copy(update={"values": {**bucket.values, "first_start": start, "last_end": end}})

    # Step 22 overlaps step 21 by 10s; unprojected traces sit inside step 22; step 23 follows a gap.
    view = rollout_time_view(
        {21: spanned(21, 0.0, 60.0), 22: spanned(22, 50.0, 120.0), None: spanned(None, 100.0, 110.0),
         23: spanned(23, 200.0, 230.0)},
        live=False,
    )

    assert view.elapsed_ms == 150_000.0
    assert view.steps[0].elapsed_ms == 60_000.0
    assert rollout_time_view({21: _bucket(21, 64, 1.0)}, live=False).elapsed_ms is None
