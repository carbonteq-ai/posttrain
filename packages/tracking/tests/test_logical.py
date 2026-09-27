from __future__ import annotations

from posttrain.tracking import MetricPoint, MetricSeries
from posttrain.tracking.logical import logical_series

_REPLAY = {"observation_source": "verifiers"}


def _series(name: str, *points: MetricPoint) -> MetricSeries:
    return MetricSeries(name=name, points=points)


def test_replayed_points_move_to_their_update_and_replace_live_points() -> None:
    series = logical_series(
        _series(
            "train/rl/reward_std",
            MetricPoint(value=0.2, step=1),
            MetricPoint(value=0.25, step=2),
            MetricPoint(value=0.1, step=56, attributes={**_REPLAY, "source_step": 1}),
            MetricPoint(value=0.4, step=57, attributes={**_REPLAY, "source_step": 3}),
        )
    )
    assert [(point.step, point.value) for point in series.points] == [(1, 0.1), (2, 0.25), (3, 0.4)]


def test_identical_duplicates_collapse_and_tagged_points_stay_distinct() -> None:
    series = logical_series(
        _series(
            "train/rl/curriculum/class_candidate_groups",
            MetricPoint(value=2.0, step=1, attributes={"class_id": "a"}),
            MetricPoint(value=2.0, step=1, attributes={"class_id": "a"}),
            MetricPoint(value=2.0, step=1, attributes={"class_id": "b"}),
        )
    )
    assert [point.attributes["class_id"] for point in series.points] == ["a", "b"]


def test_legacy_rollout_batches_combine_into_one_value_per_update() -> None:
    def batch(step: int, value: float, ordinal: int) -> MetricPoint:
        return MetricPoint(value=value, step=step, attributes={"rollout_batch_ordinal": ordinal})

    seconds = logical_series(
        _series(
            "train/rl/time/rollout_seconds", batch(1, 251.7, 1), batch(1, 60.9, 2), batch(1, 56.9, 3), batch(2, 98.3, 1)
        )
    )
    assert [(point.step, round(point.value, 1)) for point in seconds.points] == [(1, 369.5), (2, 98.3)]
    assert seconds.points[0].attributes == {"rollout_batches": 3}
    rate = logical_series(_series("train/rl/rollout_tokens_per_second", batch(1, 100.0, 1), batch(1, 300.0, 2)))
    assert [point.value for point in rate.points] == [200.0]
    # SAMPO advantages were written once per batch without a tag.
    advantage = logical_series(
        _series("train/rl/anchor_group_size_mean", MetricPoint(value=2.0, step=3), MetricPoint(value=3.0, step=3))
    )
    assert [point.value for point in advantage.points] == [2.5]
    # New runs write one value per update; it passes through unchanged.
    current = MetricPoint(value=369.5, step=1, attributes={"rollout_batches": 3})
    assert logical_series(_series("train/rl/time/rollout_seconds", current)).points == (current,)
