from __future__ import annotations

import hashlib
from collections import Counter

import pytest
from posttrain.environment import TaskDescriptor, TaskFacetValue
from posttrain.eval import (
    EvaluationFilterClause,
    EvaluationSelectionPolicy,
    EvaluationTaskFilter,
    resolve_evaluation_selection,
)


def task(key: str, subject: str | tuple[str, ...], *, split: str = "test") -> TaskDescriptor:
    subjects = (subject,) if isinstance(subject, str) else subject
    return TaskDescriptor(
        key=key,
        fingerprint="sha256:" + hashlib.sha256(key.encode()).hexdigest(),
        split=split,
        facets=tuple(TaskFacetValue("subject", value) for value in subjects),
        source_reference={"row": key},
    )


def population() -> tuple[TaskDescriptor, ...]:
    return (
        task("task-a", "algebra"),
        task("task-b", "algebra"),
        task("task-c", "algebra"),
        task("task-d", "geometry"),
        task("task-e", "geometry"),
        task("task-f", "probability"),
    )


def test_uniform_selection_is_distinct_and_input_order_independent() -> None:
    policy = EvaluationSelectionPolicy(kind="uniform", num_tasks=3, seed=19)
    forward = resolve_evaluation_selection(population(), policy)
    reverse = resolve_evaluation_selection(tuple(reversed(population())), policy)
    assert forward.digest == reverse.digest
    assert [item.task.key for item in forward.tasks] == [item.task.key for item in reverse.tasks]
    assert len({item.task.key for item in forward.tasks}) == 3
    assert sum(item.target_weight for item in forward.tasks) == pytest.approx(1)
    assert {item.inclusion_probability for item in forward.tasks} == {0.5}


def test_balanced_selection_caps_small_strata_and_redistributes() -> None:
    manifest = resolve_evaluation_selection(
        population(),
        EvaluationSelectionPolicy(kind="balanced", num_tasks=5, dimensions=("subject",), seed=2),
    )
    counts = Counter(item.stratum for item in manifest.tasks)
    assert counts == {
        "subject=algebra": 2,
        "subject=geometry": 2,
        "subject=probability": 1,
    }
    assert sum(item.target_weight for item in manifest.tasks) == pytest.approx(1)
    assert all(item.target_weight == pytest.approx(1 / 3) for item in manifest.strata)


def test_proportional_and_minimum_allocation_have_expected_integer_quotas() -> None:
    proportional = resolve_evaluation_selection(
        population(),
        EvaluationSelectionPolicy(kind="proportional", num_tasks=4, dimensions=("subject",)),
    )
    assert {item.key: item.selected for item in proportional.strata} == {
        "subject=algebra": 2,
        "subject=geometry": 1,
        "subject=probability": 1,
    }
    minimum = resolve_evaluation_selection(
        population(),
        EvaluationSelectionPolicy(
            kind="minimum_then_proportional",
            num_tasks=5,
            dimensions=("subject",),
            minimum_per_stratum=1,
        ),
    )
    assert sum(item.selected for item in minimum.strata) == 5
    assert all(item.selected >= 1 for item in minimum.strata)


def test_filter_and_missing_facet_policies_are_explicit() -> None:
    inventory = population() + (TaskDescriptor(key="task-g", fingerprint="sha256:" + "9" * 64, split="train"),)
    filtered = resolve_evaluation_selection(
        inventory,
        EvaluationSelectionPolicy(
            kind="full",
            task_filter=EvaluationTaskFilter(
                all_of=(EvaluationFilterClause("split", "eq", ("test",)),),
                any_of=(EvaluationFilterClause("subject", "in", ("geometry", "probability")),),
            ),
        ),
    )
    assert {item.task.key for item in filtered.tasks} == {"task-d", "task-e", "task-f"}
    with pytest.raises(ValueError, match="missing allocation facet"):
        resolve_evaluation_selection(
            inventory,
            EvaluationSelectionPolicy(kind="balanced", num_tasks=4, dimensions=("subject",)),
        )
    bucketed = resolve_evaluation_selection(
        inventory,
        EvaluationSelectionPolicy(kind="balanced", num_tasks=4, dimensions=("subject",), missing="bucket"),
    )
    assert "subject=__missing__" in {item.key for item in bucketed.strata}


def test_multi_value_facets_form_one_canonical_membership_stratum() -> None:
    inventory = population() + (task("task-g", ("geometry", "algebra")),)
    manifest = resolve_evaluation_selection(
        inventory,
        EvaluationSelectionPolicy(kind="balanced", num_tasks=4, dimensions=("subject",)),
    )
    keys = {item.key for item in manifest.strata}
    assert "subject=algebra+geometry" in keys
    assert len({item.task.key for item in manifest.tasks}) == 4


def test_exhaustion_and_minimum_infeasibility_are_not_silent() -> None:
    with pytest.raises(ValueError, match="requests 8 tasks from 6"):
        resolve_evaluation_selection(population(), EvaluationSelectionPolicy(kind="uniform", num_tasks=8))
    manifest = resolve_evaluation_selection(
        population(),
        EvaluationSelectionPolicy(kind="uniform", num_tasks=8, exhaustion="use_all"),
    )
    assert len(manifest.tasks) == 6
    assert manifest.shortfall == 2
    with pytest.raises(ValueError, match="requires 6 tasks but budget is 2"):
        resolve_evaluation_selection(
            population(),
            EvaluationSelectionPolicy(
                kind="minimum_then_proportional",
                num_tasks=2,
                dimensions=("subject",),
                minimum_per_stratum=2,
            ),
        )


def test_custom_weights_must_cover_strata_and_have_enough_positive_capacity() -> None:
    with pytest.raises(ValueError, match="missing strata"):
        resolve_evaluation_selection(
            population(),
            EvaluationSelectionPolicy(
                kind="custom",
                num_tasks=3,
                dimensions=("subject",),
                weights={"subject=algebra": 1.0},
            ),
        )
    with pytest.raises(ValueError, match="can place only 1 of 4"):
        resolve_evaluation_selection(
            population(),
            EvaluationSelectionPolicy(
                kind="custom",
                num_tasks=4,
                dimensions=("subject",),
                weights={
                    "subject=algebra": 0.0,
                    "subject=geometry": 0.0,
                    "subject=probability": 1.0,
                },
            ),
        )
