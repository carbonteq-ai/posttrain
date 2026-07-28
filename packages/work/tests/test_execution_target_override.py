"""Execution-target override semantics at the work-package boundary."""

from __future__ import annotations

import pytest
from posttrain.common import Catalog, CatalogLayer, CatalogRef, ContractError, ExecutionTarget
from posttrain.train import FullParameterUpdate, TrainingBinding, TrainingRenderer
from posttrain.work import (
    JobDefinition,
    Recipe,
    RecipeJob,
    WorkPackage,
    WorkPackageContext,
    override_job_execution_target,
    prepare_work_package_job,
)


def _target(identifier: str, memory_gb: float) -> ExecutionTarget:
    return ExecutionTarget(
        identifier,
        "1",
        "nvidia-cuda",
        memory_gb,
        {"world_size": 1},
    )


def _context(
    selections: tuple[ExecutionTarget | TrainingBinding, ...],
    definition: JobDefinition,
) -> WorkPackageContext:
    entries = {
        CatalogRef(
            "training" if isinstance(value, TrainingBinding) else "target",
            value.id,
        ): value
        for value in selections
    }
    return WorkPackageContext(
        Catalog(CatalogLayer("framework-v1", entries), (), "example"),
        {definition.id: definition},
    )


def test_override_replaces_nested_sft_training_target_and_snapshot() -> None:
    local = _target("targets/local", 8)
    remote = _target("targets/remote", 24)
    training = TrainingBinding(
        "training/sft@1",
        "1",
        "trl@1.8.0",
        TrainingRenderer("renderers/test", "test", "default", "off"),
        FullParameterUpdate(),
        local,
    )
    definition = JobDefinition(
        "train/test-sft@1",
        "train.sft",
        {"training": TrainingBinding},
        lambda _context, _seats: None,
    )
    context = _context((local, remote, training), definition)
    package = WorkPackage(
        "example",
        "train/sft",
        "train",
        Recipe(
            "recipes/sft@1",
            "1",
            "train",
            {"training": "training"},
            (RecipeJob("train", "train.sft", definition.id),),
        ),
        {"training": CatalogRef("training", training.id)},
    )

    overridden = override_job_execution_target(
        context,
        package,
        "train",
        remote,
    )
    prepared = prepare_work_package_job(context, overridden, "train")

    selected = prepared.seats["training"]
    assert isinstance(selected, TrainingBinding)
    assert selected.target is remote
    assert prepared.resolved.seats["training"].source_layer == "inline"
    targets = prepared.spec.resolved_inputs["execution_targets"]["targets"]  # type: ignore[index]
    assert targets[0]["selection_id"] == remote.id  # type: ignore[index]


def test_override_replaces_direct_target() -> None:
    local = _target("targets/local", 8)
    remote = _target("targets/remote", 24)
    definition = JobDefinition(
        "data/check@1",
        "data.prepare",
        {"target": ExecutionTarget},
        lambda _context, _seats: None,
    )
    context = _context((local, remote), definition)
    package = WorkPackage(
        "example",
        "screen/check",
        "screen",
        Recipe(
            "recipes/check@1",
            "1",
            "screen",
            {"target": "target"},
            (RecipeJob("check", "data.prepare", definition.id),),
        ),
        {"target": CatalogRef("target", local.id)},
    )

    overridden = override_job_execution_target(context, package, "check", remote)

    assert prepare_work_package_job(context, overridden, "check").seats["target"] is remote
    with pytest.raises(ContractError, match="no-op"):
        override_job_execution_target(context, overridden, "check", remote)


def test_override_rejects_ambiguous_primary_targets() -> None:
    first = _target("targets/first", 8)
    second = _target("targets/second", 16)
    remote = _target("targets/remote", 24)
    definition = JobDefinition(
        "data/ambiguous@1",
        "data.prepare",
        {"first": ExecutionTarget, "second": ExecutionTarget},
        lambda _context, _seats: None,
    )
    context = _context((first, second, remote), definition)
    package = WorkPackage(
        "example",
        "screen/ambiguous",
        "screen",
        Recipe(
            "recipes/ambiguous@1",
            "1",
            "screen",
            {"first": "target", "second": "target"},
            (RecipeJob("check", "data.prepare", definition.id),),
        ),
        {
            "first": CatalogRef("target", first.id),
            "second": CatalogRef("target", second.id),
        },
    )

    with pytest.raises(ContractError, match="ambiguous across explicit target seats"):
        override_job_execution_target(context, package, "check", remote)
