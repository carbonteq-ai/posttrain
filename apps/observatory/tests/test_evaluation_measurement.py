from __future__ import annotations

import pytest
from posttrain_observatory.evaluation_measurement import measure_evaluation
from posttrain_observatory.models import (
    EvaluationAttemptEvidence,
    EvaluationMeasurementFacet,
    EvaluationMeasurementPolicyView,
    EvaluationMeasurementTask,
)


def task(key: str, weight: float, *facets: tuple[str, str]) -> EvaluationMeasurementTask:
    return EvaluationMeasurementTask(
        key=key,
        label=key.upper(),
        target_weight=weight,
        facets=tuple(
            EvaluationMeasurementFacet(dimension=dimension, value=value, label=value.title())
            for dimension, value in facets
        ),
    )


def attempt(
    task_key: str,
    repetition: int,
    reward: float | None,
    *,
    success: bool | None = None,
    attempt_index: int = 0,
    error: str | None = None,
    truncated: bool = False,
) -> EvaluationAttemptEvidence:
    return EvaluationAttemptEvidence(
        task_key=task_key,
        repetition_index=repetition,
        attempt_index=attempt_index,
        reward=reward,
        success=success,
        execution_error=error,
        truncated=truncated,
        trace_id=f"{task_key}-{repetition}-{attempt_index}",
    )


def test_task_mean_does_not_weight_a_task_by_its_valid_repetition_count() -> None:
    view = measure_evaluation(
        manifest_digest="sha256:test",
        tasks=(task("a", 0.5), task("b", 0.5)),
        repetitions_per_task=3,
        attempts=(
            attempt("a", 0, 1, success=True),
            attempt("a", 1, 1, success=True),
            attempt("a", 2, 1, success=True),
            attempt("b", 0, 0, success=False),
        ),
    )

    assert view.estimate.available_case_value == pytest.approx(0.5)
    assert view.estimate.value is None
    assert view.estimate.state == "partial"
    assert view.coverage.valid_repetitions == 4
    assert view.coverage.missing_repetitions == 2
    assert view.tasks[0].mean_reward == 1
    assert view.tasks[1].mean_reward == 0
    assert view.tasks[1].any_of_k is None


def test_retry_is_an_attempt_for_one_repetition() -> None:
    view = measure_evaluation(
        manifest_digest="sha256:test",
        tasks=(task("a", 1),),
        repetitions_per_task=3,
        attempts=(
            attempt("a", 0, None, attempt_index=0, error="timeout"),
            attempt("a", 0, 1, success=True, attempt_index=1),
            attempt("a", 1, 0, success=False),
        ),
    )

    assert view.coverage.planned_repetitions == 3
    assert view.coverage.execution_attempts == 3
    assert view.coverage.retries == 1
    assert view.coverage.execution_failures == 1
    assert view.coverage.valid_repetitions == 2
    assert view.coverage.missing_repetitions == 1


def test_target_weighting_and_overlapping_facets_are_explicit() -> None:
    tasks = (
        task("a", 0.9, ("subject", "algebra"), ("skill", "reasoning")),
        task("b", 0.1, ("subject", "geometry"), ("skill", "reasoning")),
    )
    attempts = (attempt("a", 0, 1, success=True), attempt("b", 0, 0, success=False))
    weighted = measure_evaluation(
        manifest_digest="sha256:test",
        tasks=tasks,
        repetitions_per_task=1,
        attempts=attempts,
        policy=EvaluationMeasurementPolicyView(estimator="target_weighted"),
    )
    macro = measure_evaluation(
        manifest_digest="sha256:test",
        tasks=tasks,
        repetitions_per_task=1,
        attempts=attempts,
    )

    assert weighted.estimate.value == pytest.approx(0.9)
    assert macro.estimate.value == pytest.approx(0.5)
    reasoning = next(item for item in weighted.facets if item.value == "reasoning")
    assert reasoning.selected_tasks == 2
    assert weighted.coverage.selected_tasks == 2


def test_available_missing_policy_is_labeled_partial_but_returns_value() -> None:
    view = measure_evaluation(
        manifest_digest="sha256:test",
        tasks=(task("a", 0.5), task("b", 0.5)),
        repetitions_per_task=1,
        attempts=(attempt("a", 0, 1),),
        policy=EvaluationMeasurementPolicyView(missing="available"),
    )
    assert view.state == "partial"
    assert view.estimate.value == 1
    assert view.estimate.available_case_value == 1
    assert view.estimate.target_weight_observed == pytest.approx(0.5)


def test_measurement_rejects_invalid_population_and_attempt_identity() -> None:
    with pytest.raises(ValueError, match="weights must sum to one"):
        measure_evaluation(
            manifest_digest="sha256:test",
            tasks=(task("a", 0.8),),
            repetitions_per_task=1,
            attempts=(),
        )
    with pytest.raises(ValueError, match="unknown task"):
        measure_evaluation(
            manifest_digest="sha256:test",
            tasks=(task("a", 1),),
            repetitions_per_task=1,
            attempts=(attempt("b", 0, 1),),
        )
