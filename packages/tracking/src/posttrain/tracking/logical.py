"""Metric points at logical steps: the one normalization every tracking reader applies.

A training metric has one value per update, at the update's step, for each set
of tags (`06-observation-and-lineage.md`). Two things recorded before that
rule held are normalized here, once, so no consumer needs its own rules:

- Metrics recomputed from preserved Verifiers traces when a run finishes are
  stored at the next free provider step and carry the update they describe in
  `source_step`. They are moved to that step, and they replace live points of
  the same metric for that update ("replay authority").
- Runs recorded before rollout metrics were written once per update have one
  point per rollout batch. Those points are combined per update: counts,
  seconds and tokens are summed, rates and means are averaged. This legacy
  rule can be removed once no retained run needs it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal

from posttrain.common import JsonValue

from .models import MetricPoint, MetricSeries

type Combine = Literal["sum", "mean"]

LEGACY_ROLLOUT_BATCH_METRICS: Mapping[str, Combine] = {
    "train/rl/rollouts_requested": "sum",
    "train/rl/rollouts_attempted": "sum",
    "train/rl/admission_rounds": "sum",
    "train/rl/admission_rejected_groups": "sum",
    "train/rl/rollouts_completed": "sum",
    "train/rl/rollouts_failed": "sum",
    "train/rl/rollouts_replaced": "sum",
    "train/rl/rollouts_truncated": "sum",
    "train/rl/rollouts_unscorable": "sum",
    "train/rl/rollouts_missing": "sum",
    "train/rl/time/rollout_seconds": "sum",
    "train/rl/rollout_selected_tokens": "sum",
    "train/rl/rollout_tokens_per_second": "mean",
    "train/rl/rollout_selected_token_fraction": "mean",
    "train/rl/episode_advantage_mean": "mean",
    "train/rl/turn_advantage_mean": "mean",
    "train/rl/anchor_group_size_mean": "mean",
    "train/rl/sparse_reward_projection_fraction": "mean",
}
_BATCH = "rollout_batch_ordinal"


def logical_series(series: MetricSeries) -> MetricSeries:
    """The series with every point at its logical step, replay authority applied, legacy batches combined."""

    replay = [(point, step) for point in series.points if (step := _source_step(point)) is not None]
    if replay:
        # Replay is authoritative only for the updates it covers; live points
        # of other updates stay, so a partially finalized run keeps them.
        replay_steps = {step for _, step in replay}
        points = [
            point
            for point in series.points
            if point.attributes.get("observation_source") != "verifiers" and point.step not in replay_steps
        ]
        points += [point.model_copy(update={"step": step}) for point, step in replay]
    else:
        # Every stored point is one write (Trackio keys rows by log id), so equal points
        # are separate observations: rollout batches can repeat a value and an ordinal.
        points = list(series.points)
    combine = LEGACY_ROLLOUT_BATCH_METRICS.get(series.name)
    if combine is not None:
        points = _combine_batches(points, combine)
    points.sort(key=lambda point: point.step if point.step is not None else -1)
    return MetricSeries(name=series.name, points=tuple(points))


def _source_step(point: MetricPoint) -> int | None:
    step = point.attributes.get("source_step")
    if (
        point.attributes.get("observation_source") == "verifiers"
        and isinstance(step, int)
        and not isinstance(step, bool)
        and step >= 0
    ):
        return step
    return None


def _combine_batches(points: list[MetricPoint], combine: Combine) -> list[MetricPoint]:
    groups: dict[tuple[int | None, str], list[MetricPoint]] = {}
    for point in points:
        tags = {key: value for key, value in point.attributes.items() if key != _BATCH}
        groups.setdefault((point.step, json.dumps(tags, sort_keys=True, default=str)), []).append(point)
    combined: list[MetricPoint] = []
    for group in groups.values():
        if len(group) == 1 and _BATCH not in group[0].attributes:
            combined.append(group[0])
            continue
        values = [point.value for point in group]
        value = sum(values) if combine == "sum" else sum(values) / len(values)
        attributes: dict[str, JsonValue] = {key: item for key, item in group[-1].attributes.items() if key != _BATCH}
        attributes["rollout_batches"] = len(group)
        combined.append(group[-1].model_copy(update={"value": value, "attributes": attributes}))
    return combined


__all__ = ["LEGACY_ROLLOUT_BATCH_METRICS", "logical_series"]
