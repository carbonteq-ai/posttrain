"""Run-level rollout phase time, aggregated from stored Verifiers timing.

Every Verifiers trace records phase spans in ``timing``: environment setup, the
agent loop split into model-call time (``agent.model.duration``, served on the
inference GPU) and harness time (``agent.harness.duration``, tool execution on
CPU), then finalize and scoring. Trackio aggregates those values straight from
the stored payloads, grouped by optimizer step, in one storage query.

A whole-run aggregate parses every payload once, which takes seconds on a large
run. Finished optimizer steps never change, so a refresh first counts traces
per step from indexed facts (cheap) and re-aggregates only the steps whose
trace count changed, plus traces whose facts are not projected yet.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from posttrain.common import JsonValue
from posttrain.tracking import (
    TraceAggregateBucket,
    TraceAggregateResult,
    TraceFactAggregate,
    TraceFactsQuery,
    TracePayloadMeasure,
    TracePayloadQuery,
)
from posttrain.tracking.models import PayloadDimension

from .models import RolloutTimeStep, RolloutTimeView

ROLLOUT_TIME_MEASURES = (
    TracePayloadMeasure(key="inference_s", path="$.timing.agent.model.duration"),
    TracePayloadMeasure(key="tools_s", path="$.timing.agent.harness.duration"),
    TracePayloadMeasure(key="setup_s", path="$.timing.setup.end", minus="$.timing.setup.start"),
    TracePayloadMeasure(key="scoring_s", path="$.timing.scoring.end", minus="$.timing.finalize.start"),
    TracePayloadMeasure(key="rollout_s", path="$.timing.scoring.end", minus="$.timing.start"),
    # Wall-clock bounds: rollouts run concurrently, so summed phase time is not elapsed time.
    TracePayloadMeasure(key="first_start", path="$.timing.start", operation="min"),
    TracePayloadMeasure(key="last_end", path="$.timing.scoring.end", operation="max"),
)
# Above this many changed steps, one grouped query beats per-step queries.
_MAX_INCREMENTAL_STEPS = 6

type StepKey = int | None
type PayloadAggregate = Callable[[str, TracePayloadQuery], Awaitable[TraceAggregateResult]]
type FactAggregate = Callable[[str, TraceFactsQuery], Awaitable[TraceAggregateResult]]


def _step_key(bucket: TraceAggregateBucket) -> StepKey:
    value = bucket.dimensions.get("rollout_step")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


async def refresh_rollout_time_buckets(
    run_id: str,
    trace_type: str,
    cached: Mapping[StepKey, TraceAggregateBucket] | None,
    *,
    aggregate_payload: PayloadAggregate,
    aggregate_facts: FactAggregate | None,
) -> dict[StepKey, TraceAggregateBucket] | None:
    """Return per-step phase buckets, re-querying only steps that changed.

    Returns None when the tracking provider cannot aggregate payloads.
    """

    async def query(dimensions: dict[PayloadDimension, JsonValue] | None = None) -> TraceAggregateResult:
        return await aggregate_payload(
            run_id,
            TracePayloadQuery(
                measures=ROLLOUT_TIME_MEASURES,
                trace_type=trace_type,
                group_by=() if dimensions else ("rollout_step",),
                dimensions=dimensions or {},
            ),
        )

    async def full() -> dict[StepKey, TraceAggregateBucket] | None:
        result = await query()
        if result.state != "available":
            return None
        return {_step_key(bucket): bucket for bucket in result.buckets}

    if not cached or aggregate_facts is None:
        return await full()
    counts = await aggregate_facts(
        run_id,
        TraceFactsQuery(
            trace_type=trace_type,
            group_by=("rollout_step",),
            aggregates=(TraceFactAggregate(measure="model_calls", operation="count"),),
        ),
    )
    if counts.state != "available":
        return await full()
    fact_counts = {_step_key(bucket): bucket.trace_count for bucket in counts.buckets}
    changed = [
        step
        for step, count in fact_counts.items()
        if step is not None and (step not in cached or cached[step].trace_count != count)
    ]
    if len(changed) > _MAX_INCREMENTAL_STEPS:
        return await full()
    buckets: dict[StepKey, TraceAggregateBucket] = {step: bucket for step, bucket in cached.items() if step is not None}
    # Traces without projected facts have no step yet; always re-read them.
    for step in [*changed, None]:
        result = await query({"rollout_step": step})
        if result.state != "available":
            return await full()
        if result.buckets and result.buckets[0].trace_count:
            buckets[step] = result.buckets[0]
        else:
            buckets.pop(step, None)
    return buckets


def _milliseconds(bucket: TraceAggregateBucket, key: str) -> float | None:
    value = bucket.values.get(key)
    return float(value) * 1000 if isinstance(value, int | float) and bucket.coverage.get(key, 0) else None


def _elapsed_ms(spans: list[tuple[float, float]]) -> float | None:
    """Wall-clock covered by the union of step spans; overlapping steps count once."""

    if not spans:
        return None
    total = 0.0
    start, end = sorted(spans)[0]
    for next_start, next_end in sorted(spans)[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return (total + end - start) * 1000


def _span(bucket: TraceAggregateBucket) -> tuple[float, float] | None:
    start, end = bucket.values.get("first_start"), bucket.values.get("last_end")
    if isinstance(start, int | float) and isinstance(end, int | float) and end >= start:
        return float(start), float(end)
    return None


def rollout_time_view(
    buckets: Mapping[StepKey, TraceAggregateBucket] | None,
    *,
    live: bool,
) -> RolloutTimeView:
    """Summed phase time per optimizer step, oldest first; unprojected traces last."""

    if buckets is None:
        return RolloutTimeView(state="unavailable", live=live)
    ordered = sorted(buckets.items(), key=lambda item: (item[0] is None, item[0] or 0))
    steps: list[RolloutTimeStep] = []
    for step, bucket in ordered:
        span = _span(bucket)
        steps.append(
            RolloutTimeStep(
                step=step,
                rollouts=bucket.trace_count,
                timed_rollouts=bucket.coverage.get("rollout_s", 0),
                inference_ms=_milliseconds(bucket, "inference_s"),
                tools_ms=_milliseconds(bucket, "tools_s"),
                setup_ms=_milliseconds(bucket, "setup_s"),
                scoring_ms=_milliseconds(bucket, "scoring_s"),
                rollout_ms=_milliseconds(bucket, "rollout_s"),
                elapsed_ms=(span[1] - span[0]) * 1000 if span else None,
            )
        )
    spans = [span for bucket in buckets.values() if (span := _span(bucket)) is not None]
    return RolloutTimeView(
        state="available" if steps else "unavailable",
        steps=tuple(steps),
        elapsed_ms=_elapsed_ms(spans),
        live=live,
    )


__all__ = ["ROLLOUT_TIME_MEASURES", "refresh_rollout_time_buckets", "rollout_time_view"]
