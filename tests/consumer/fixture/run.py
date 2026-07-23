"""Independent-consumer proof executed only from installed wheels."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from pathlib import Path

from posttrain.catalog import discover_project, open_catalog
from posttrain.common import CatalogRef, ExecutionTarget, RunContext
from posttrain.data import (
    SupervisedDataset,
    SupervisedExample,
    SupervisedPartitionPlan,
    partition_supervised_dataset,
)
from posttrain.work import (
    JobDefinition,
    ResolvedSeats,
    WorkPackageContext,
    execute_run_tracked,
    load_work_package,
    run_work_package,
)
from posttrain_observatory import ObservatoryService
from posttrain_tracking_trackio import TrackioBackend, TrackioDataSource, TrackioSettings


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


async def _main() -> None:
    layout = discover_project(Path.cwd())
    catalog = open_catalog(scope=layout.project_id, overlays=layout.catalog_overlays)
    base = catalog.resolve(CatalogRef("target", "targets/local-cuda-8gb"))
    overlay = catalog.resolve(CatalogRef("target", "targets/external-cpu"))
    if not isinstance(base.value, ExecutionTarget) or not isinstance(overlay.value, ExecutionTarget):
        raise TypeError("consumer target selections did not decode")

    package = load_work_package(layout.work_packages / "cpu_check.yaml")
    backend = TrackioBackend(
        TrackioSettings(
            project=layout.project_id,
            auto_log_gpu=False,
            auto_log_cpu=False,
        )
    )
    scratch_root = layout.state / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    result = run_work_package(
        WorkPackageContext(
            catalog=catalog,
            definitions={
                "data/cpu-check@1": JobDefinition(
                    "data/cpu-check@1",
                    "data.prepare",
                    {"target": ExecutionTarget},
                    _partition,
                    "Partition deterministic fixture data and record the result.",
                )
            },
            source_metadata={"revision": "consumer-fixture"},
            executor=partial(
                execute_run_tracked,
                backend=backend,
                scratch_root=scratch_root,
            ),
        ),
        package,
    )
    job = result.jobs[0]
    if job.run_id is None or not isinstance(job.value, dict):
        raise TypeError("consumer work package did not produce a tracked result")

    source = TrackioDataSource(layout.project_id)
    detail = await source.get_run(job.run_id)
    view = await ObservatoryService(source).get_run_view_response(job.run_id)
    print(
        json.dumps(
            {
                "project_id": layout.project_id,
                "project_root": str(layout.root),
                "base_catalog_release": catalog.base_id,
                "base_source": base.source_layer,
                "base_target": base.value.id,
                "overlay_source": overlay.source_layer,
                "overlay_id": overlay.overlay_id,
                "overlay_target": overlay.value.id,
                **job.value,
                "work_package_id": result.work_package_id,
                "job_status": job.status,
                "run_id": job.run_id,
                "tracking_status": detail.summary.status,
                "tracking_metrics": list(detail.metric_names),
                "observatory_mode": view.resolved_mode,
                "observatory_run_id": view.view.run.run_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(_main())
