"""Coverage-aware evaluation measurement over manifest tasks and attempts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean

from .models import (
    EvaluationAttemptEvidence,
    EvaluationCoverage,
    EvaluationEstimatorResult,
    EvaluationFacetResult,
    EvaluationMeasurementPolicyView,
    EvaluationMeasurementTask,
    EvaluationMeasurementView,
    EvaluationTaskResult,
)

EVALUATION_MEASUREMENT_CALCULATOR_VERSION = "evaluation-measurement@1"


def measure_evaluation(
    *,
    manifest_digest: str,
    tasks: Sequence[EvaluationMeasurementTask],
    repetitions_per_task: int,
    attempts: Sequence[EvaluationAttemptEvidence],
    policy: EvaluationMeasurementPolicyView | None = None,
) -> EvaluationMeasurementView:
    """Reduce retries into repetitions, then repetitions into equally identified tasks."""

    if repetitions_per_task < 1:
        raise ValueError("repetitions_per_task must be positive")
    policy = policy or EvaluationMeasurementPolicyView()
    task_by_key = {task.key: task for task in tasks}
    if len(task_by_key) != len(tasks):
        raise ValueError("evaluation measurement tasks must have unique keys")
    if not tasks:
        raise ValueError("evaluation measurement requires at least one task")
    total_weight = sum(task.target_weight for task in tasks)
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError("evaluation measurement task weights must sum to one")

    by_slot: defaultdict[tuple[str, int], list[EvaluationAttemptEvidence]] = defaultdict(list)
    for attempt in attempts:
        if attempt.task_key not in task_by_key:
            raise ValueError(f"attempt references unknown task {attempt.task_key!r}")
        if attempt.repetition_index >= repetitions_per_task:
            raise ValueError("attempt repetition index is outside the planned range")
        by_slot[(attempt.task_key, attempt.repetition_index)].append(attempt)
    for slot_attempts in by_slot.values():
        indices = [attempt.attempt_index for attempt in slot_attempts]
        if len(indices) != len(set(indices)):
            raise ValueError("execution attempt indices must be unique within a repetition")
        slot_attempts.sort(key=lambda attempt: attempt.attempt_index)

    task_results: list[EvaluationTaskResult] = []
    completed_repetitions = 0
    valid_repetitions = 0
    execution_failures = 0
    truncations = 0
    for task in sorted(tasks, key=lambda item: item.key):
        rewards: list[float] = []
        successes: list[bool] = []
        trace_ids: list[str] = []
        task_attempts = 0
        task_retries = 0
        task_failures = 0
        task_truncations = 0
        for repetition_index in range(repetitions_per_task):
            slot_attempts = by_slot.get((task.key, repetition_index), [])
            task_attempts += len(slot_attempts)
            task_retries += max(len(slot_attempts) - 1, 0)
            task_failures += sum(attempt.execution_error is not None for attempt in slot_attempts)
            task_truncations += sum(attempt.truncated for attempt in slot_attempts)
            if slot_attempts:
                completed_repetitions += 1
            usable = next(
                (
                    attempt
                    for attempt in slot_attempts
                    if attempt.execution_error is None and not attempt.truncated and attempt.reward is not None
                ),
                None,
            )
            if usable is None:
                continue
            valid_repetitions += 1
            rewards.append(usable.reward)  # pyright: ignore[reportArgumentType]
            if usable.success is not None:
                successes.append(usable.success)
            if usable.trace_id is not None:
                trace_ids.append(usable.trace_id)
        task_results.append(
            EvaluationTaskResult(
                key=task.key,
                label=task.label,
                target_weight=task.target_weight,
                planned_repetitions=repetitions_per_task,
                valid_repetitions=len(rewards),
                execution_attempts=task_attempts,
                retries=task_retries,
                execution_failures=task_failures,
                truncations=task_truncations,
                mean_reward=fmean(rewards) if rewards else None,
                success_frequency=fmean(successes) if successes else None,
                any_of_k=any(successes) if len(successes) == repetitions_per_task else None,
                all_of_k=all(successes) if len(successes) == repetitions_per_task else None,
                facets=task.facets,
                trace_ids=tuple(trace_ids),
            )
        )
        execution_failures += task_failures
        truncations += task_truncations

    observed = [task for task in task_results if task.mean_reward is not None]
    coverage = EvaluationCoverage(
        selected_tasks=len(tasks),
        observed_tasks=len(observed),
        planned_repetitions=len(tasks) * repetitions_per_task,
        completed_repetitions=completed_repetitions,
        valid_repetitions=valid_repetitions,
        execution_attempts=len(attempts),
        retries=sum(task.retries for task in task_results),
        execution_failures=execution_failures,
        truncations=truncations,
        missing_repetitions=len(tasks) * repetitions_per_task - valid_repetitions,
    )
    complete = coverage.missing_repetitions == 0
    if policy.estimator == "task_mean":
        available = fmean(task.mean_reward for task in observed if task.mean_reward is not None) if observed else None
        observed_weight = len(observed) / len(tasks)
    else:
        observed_weight = sum(task.target_weight for task in observed)
        available = (
            sum(task.target_weight * task.mean_reward for task in observed if task.mean_reward is not None)
            / observed_weight
            if observed_weight
            else None
        )
    strict_value = available if complete or policy.missing == "available" else None
    state = "complete" if complete else "partial" if observed else "unavailable"
    estimate = EvaluationEstimatorResult(
        estimator=policy.estimator,
        state=state,
        value=strict_value,
        available_case_value=available,
        task_denominator=len(observed),
        target_weight_observed=observed_weight,
    )

    facet_groups: defaultdict[tuple[str, str, str], list[EvaluationTaskResult]] = defaultdict(list)
    for task in task_results:
        for facet in task.facets:
            facet_groups[(facet.dimension, facet.value, facet.label)].append(task)
    facets = tuple(
        EvaluationFacetResult(
            dimension=dimension,
            value=value,
            label=label,
            selected_tasks=len(values),
            observed_tasks=sum(item.mean_reward is not None for item in values),
            valid_repetitions=sum(item.valid_repetitions for item in values),
            mean_reward=fmean(item.mean_reward for item in values if item.mean_reward is not None)
            if any(item.mean_reward is not None for item in values)
            else None,
            success_rate=fmean(item.success_frequency for item in values if item.success_frequency is not None)
            if any(item.success_frequency is not None for item in values)
            else None,
        )
        for (dimension, value, label), values in sorted(facet_groups.items())
    )
    return EvaluationMeasurementView(
        calculator_version=EVALUATION_MEASUREMENT_CALCULATOR_VERSION,
        manifest_digest=manifest_digest,
        state=state,
        policy=policy,
        coverage=coverage,
        estimate=estimate,
        tasks=tuple(task_results),
        facets=facets,
    )


__all__ = ["EVALUATION_MEASUREMENT_CALCULATOR_VERSION", "measure_evaluation"]
