"""Project-owned entry used by the installed-wheel acceptance test."""

from __future__ import annotations

from posttrain.common import ExecutionTarget, RunContext
from posttrain.data import (
    SupervisedDataset,
    SupervisedExample,
    SupervisedPartitionPlan,
    partition_supervised_dataset,
)
from posttrain.jobs import build_job_runtime
from posttrain.work import (
    JobDefinition,
    JobRuntime,
    ProjectExecutionRequest,
    ResolvedSeats,
)


def _partition(context: RunContext, seats: ResolvedSeats) -> dict[str, object]:
    target = seats["target"]
    if not isinstance(target, ExecutionTarget) or target.device_class != "cpu":
        raise TypeError("consumer operation requires its project-owned CPU target")
    examples = tuple(
        SupervisedExample(
            id=f"example-{index:03d}",
            messages=(
                {"role": "user", "content": f"Count {index}"},
                {"role": "assistant", "content": str(index)},
            ),
            trainable_message_indices=(1,),
        )
        for index in range(64)
    )
    dataset = SupervisedDataset("consumer/examples", "fixture-v1", examples)
    partitioned = partition_supervised_dataset(
        dataset,
        SupervisedPartitionPlan(
            id="consumer/partition",
            revision="1",
            validation_fraction=0.2,
            reserve_fraction=0.1,
            seed=17,
        ),
    )
    validation_examples = len(partitioned.validation.examples) if partitioned.validation else 0
    reserve_examples = len(partitioned.reserve.examples) if partitioned.reserve else 0
    context.metric("data/train_examples", len(partitioned.train.examples))
    context.metric("data/validation_examples", validation_examples)
    context.metric("data/reserve_examples", reserve_examples)
    return {
        "partition_digest": partitioned.manifest.digest,
        "train_examples": len(partitioned.train.examples),
        "validation_examples": validation_examples,
        "reserve_examples": reserve_examples,
    }


def configure(request: ProjectExecutionRequest) -> JobRuntime:
    """Bind the fixture operation on top of standard jobs and local Trackio."""

    return build_job_runtime(
        request,
        extra_definitions={
            "data/cpu-check@1": JobDefinition(
                "data/cpu-check@1",
                "data.prepare",
                {"target": ExecutionTarget},
                _partition,
                "Partition deterministic fixture data and record the result.",
            )
        },
    )


__all__ = ["configure"]
