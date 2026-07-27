"""Attributing a repack to the input that caused it."""

from __future__ import annotations

from posttrain.execution import compare_job_packages, unchanged_fields

_BASE: dict[str, object] = {
    "schema": "posttrain.job-package.v1",
    "project_id": "example",
    "work_package_id": "train/example",
    "job_id": "train",
    "job_kind": "train.sft",
    "resolved_config_digest": "a" * 64,
    "resolved_inputs_digest": "b" * 64,
    "project_source_digest": "c" * 64,
    "framework_source_digest": "d" * 64,
    "kind_image": "registry.lan/team/posttrain-kind-supervised@sha256:" + "e" * 64,
    "datasets": [{"dataset_id": "gsm8k", "digest": "f" * 64}],
    "backend_runtime": {"backend": "verl", "projection_digest": "1" * 64},
    "expected_artifact_roles": ["model"],
}


def _with(**changes: object) -> dict[str, object]:
    return {**_BASE, **changes}


def test_identical_packages_report_no_changes() -> None:
    assert compare_job_packages(_BASE, dict(_BASE)) == ()


def test_schema_is_not_reported_as_a_change() -> None:
    changed = compare_job_packages(_BASE, _with(schema="posttrain.job-package.v2"))
    assert changed == ()


def test_a_hyperparameter_change_is_attributed_and_isolated() -> None:
    """The question people actually ask: why did this repack?"""
    changes = compare_job_packages(_BASE, _with(resolved_config_digest="9" * 64))
    assert len(changes) == 1
    assert changes[0].field == "resolved_config_digest"
    assert "hyperparameters" in changes[0].explanation
    # Everything except `schema` and the one field that moved.
    assert len(unchanged_fields(_BASE, _with(resolved_config_digest="9" * 64))) == len(_BASE) - 2


def test_a_comment_only_edit_is_named_as_source_drift() -> None:
    """Surprising but correct, and the report has to say why."""
    changes = compare_job_packages(_BASE, _with(project_source_digest="9" * 64))
    assert len(changes) == 1
    assert "comments and formatting" in changes[0].explanation


def test_a_framework_release_move_is_distinguished_from_project_change() -> None:
    moved = "registry.lan/team/posttrain-kind-supervised@sha256:" + "9" * 64
    changes = compare_job_packages(_BASE, _with(kind_image=moved))
    assert len(changes) == 1
    assert "framework release moved" in changes[0].explanation


def test_digests_are_abbreviated_for_reading() -> None:
    change = compare_job_packages(_BASE, _with(resolved_config_digest="9" * 64))[0]
    assert change.previous == "a" * 12
    assert change.current == "9" * 12


def test_image_references_keep_repository_and_short_digest() -> None:
    moved = "registry.lan/team/posttrain-kind-supervised@sha256:" + "9" * 64
    change = compare_job_packages(_BASE, _with(kind_image=moved))[0]
    assert change.previous == "posttrain-kind-supervised@" + "e" * 12
    assert change.current == "posttrain-kind-supervised@" + "9" * 12


def test_nested_composites_name_the_field_that_moved() -> None:
    """Two long blobs would force the reader to diff by eye."""
    changes = compare_job_packages(
        _BASE,
        _with(backend_runtime={"backend": "verl", "projection_digest": "9" * 64}),
    )
    assert len(changes) == 1
    assert changes[0].field == "backend_runtime.projection_digest"
    assert changes[0].previous == "1" * 12
    assert changes[0].current == "9" * 12


def test_list_entries_are_reported_individually() -> None:
    changes = compare_job_packages(
        _BASE,
        _with(datasets=[{"dataset_id": "gsm8k", "digest": "9" * 64}]),
    )
    assert len(changes) == 1
    assert changes[0].field == "datasets[gsm8k]"
    assert "dataset changed" in changes[0].explanation


def test_added_and_removed_list_entries_are_named() -> None:
    added = compare_job_packages(
        _BASE,
        _with(
            datasets=[
                {"dataset_id": "gsm8k", "digest": "f" * 64},
                {"dataset_id": "math", "digest": "9" * 64},
            ]
        ),
    )
    assert [c.kind for c in added] == ["added"]
    assert added[0].current == "math"

    removed = compare_job_packages(_BASE, _with(datasets=[]))
    assert [c.kind for c in removed] == ["removed"]
    assert removed[0].previous == "gsm8k"


def test_a_field_absent_from_one_side_is_summarized_not_dumped() -> None:
    """An older package predating a field must stay readable."""
    older = {key: value for key, value in _BASE.items() if key != "datasets"}
    changes = compare_job_packages(older, _BASE)
    assert len(changes) == 1
    assert changes[0].kind == "added"
    assert changes[0].current == "1 entry: gsm8k"
    assert "{" not in changes[0].current


def test_every_change_carries_a_human_explanation() -> None:
    changes = compare_job_packages(
        _BASE,
        _with(
            resolved_config_digest="9" * 64,
            project_source_digest="8" * 64,
            framework_source_digest="7" * 64,
        ),
    )
    assert len(changes) == 3
    assert all(change.explanation for change in changes)
    assert all("->" in change.describe() for change in changes)
