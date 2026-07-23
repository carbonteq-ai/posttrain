"""Tests for reproducible supervised dataset partitions."""

from __future__ import annotations

import pytest
from posttrain.data import (
    SupervisedDataset,
    SupervisedExample,
    SupervisedPartitionPlan,
    partition_supervised_dataset,
)


def _dataset(*, reverse: bool = False, count: int = 240) -> SupervisedDataset:
    examples = [
        SupervisedExample(
            id=f"examples/{index:04d}",
            messages=(
                {"role": "user", "content": f"Question {index}"},
                {"role": "assistant", "content": f"Answer {index}"},
            ),
            trainable_message_indices=(1,),
            metadata={
                "conversation_id": f"conversation/{index // 2:04d}",
                "source": f"source-{index % 4}",
            },
        )
        for index in range(count)
    ]
    if reverse:
        examples.reverse()
    return SupervisedDataset("examples/sft", "source-revision", tuple(examples), metadata={"license": "test"})


def _plan(**overrides: object) -> SupervisedPartitionPlan:
    values = {
        "id": "examples/sft-split",
        "revision": "1",
        "validation_fraction": 0.15,
        "reserve_fraction": 0.1,
        "seed": 17,
    }
    values.update(overrides)
    return SupervisedPartitionPlan(**values)  # type: ignore[arg-type]


def test_partition_is_disjoint_complete_and_independent_of_provider_order() -> None:
    first = partition_supervised_dataset(_dataset(), _plan())
    reordered = partition_supervised_dataset(_dataset(reverse=True), _plan())

    assert first.manifest == reordered.manifest
    assert first.manifest.digest == reordered.manifest.digest
    assert first.manifest.as_dict()["source_revision"] == "source-revision"
    assert first.train.examples == reordered.train.examples
    assert first.validation is not None and reordered.validation is not None
    assert first.validation.examples == reordered.validation.examples
    assert first.reserve is not None and reordered.reserve is not None
    assert first.reserve.examples == reordered.reserve.examples

    populations = (
        set(first.manifest.train_ids),
        set(first.manifest.validation_ids),
        set(first.manifest.reserve_ids),
    )
    assert not populations[0] & populations[1]
    assert not populations[0] & populations[2]
    assert not populations[1] & populations[2]
    assert set.union(*populations) == {example.id for example in _dataset().examples}


def test_partition_metadata_cites_source_plan_and_manifest() -> None:
    partitioned = partition_supervised_dataset(_dataset(), _plan())

    for name, dataset in (
        ("train", partitioned.train),
        ("validation", partitioned.validation),
        ("reserve", partitioned.reserve),
    ):
        assert dataset is not None
        assert dataset.id == f"examples/sft/{name}"
        assert dataset.revision == partitioned.manifest.digest
        assert dataset.metadata["partition"] == name
        assert dataset.metadata["partition_plan_id"] == "examples/sft-split"
        assert dataset.metadata["source_dataset_revision"] == "source-revision"
        assert dataset.metadata["partition_manifest_digest"] == partitioned.manifest.digest


def test_grouped_partition_keeps_related_examples_together() -> None:
    partitioned = partition_supervised_dataset(_dataset(), _plan(group_by="conversation_id"))
    assignment = {
        example_id: name
        for name, identifiers in (
            ("train", partitioned.manifest.train_ids),
            ("validation", partitioned.manifest.validation_ids),
            ("reserve", partitioned.manifest.reserve_ids),
        )
        for example_id in identifiers
    }

    for index in range(0, 240, 2):
        assert assignment[f"examples/{index:04d}"] == assignment[f"examples/{index + 1:04d}"]


def test_stratified_partition_is_reproducible_and_records_the_key() -> None:
    partitioned = partition_supervised_dataset(_dataset(), _plan(stratify_by="source"))

    assert partitioned.train.metadata["partition_stratify_by"] == "source"
    assert partitioned.validation is not None
    assert {str(example.metadata["source"]) for example in partitioned.validation.examples} == {
        "source-0",
        "source-1",
        "source-2",
        "source-3",
    }


def test_group_cannot_span_multiple_strata() -> None:
    with pytest.raises(ValueError, match="spans strata"):
        partition_supervised_dataset(
            _dataset(),
            _plan(group_by="conversation_id", stratify_by="source"),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"validation_fraction": 0.0, "reserve_fraction": 0.0}, "must request"),
        ({"validation_fraction": 0.8, "reserve_fraction": 0.2}, "must leave"),
        ({"validation_fraction": -0.1}, "fractions"),
    ),
)
def test_partition_plan_rejects_invalid_allocations(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _plan(**overrides)


def test_partition_rejects_missing_group_metadata() -> None:
    with pytest.raises(ValueError, match="missing partition metadata"):
        partition_supervised_dataset(_dataset(), _plan(group_by="missing"))


def test_partition_does_not_silently_drop_a_requested_population() -> None:
    with pytest.raises(ValueError, match="requested validation records but produced none"):
        partition_supervised_dataset(
            _dataset(count=2),
            _plan(validation_fraction=0.01, reserve_fraction=0.0),
        )
