"""Catalog-to-capability wiring that requires no project Python modules."""

from pathlib import Path

import pytest
from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, ExecutionTarget, ModelVariant
from posttrain.data import DatasetLoadPlan, SupervisedDataset, resolve_dataset_source
from posttrain.train import SFTRequest, SFTSettings, TrainingBinding


def test_global_dataset_builds_public_sft_request_without_project_code(tmp_path: Path) -> None:
    catalog = open_catalog(scope="empty-project")

    dataset_plan = catalog.resolve(CatalogRef("dataset", "datasets/posttrain-sft-smoke@1")).value
    model = catalog.resolve(CatalogRef("model", "models/qwen3.5-2b@bf16")).value
    settings = catalog.resolve(CatalogRef("training", "qwen3.5-2b/sft-smoke-v2")).value
    training = catalog.resolve(CatalogRef("training", "training/qwen3.5-trl-lora@1")).value

    assert isinstance(dataset_plan, DatasetLoadPlan)
    assert isinstance(model, ModelVariant)
    assert isinstance(settings, SFTSettings)
    assert isinstance(training, TrainingBinding)
    dataset = resolve_dataset_source(
        dataset_plan,
        state_dir=tmp_path / ".posttrain" / "state",
        project_root=tmp_path,
    )
    assert isinstance(dataset, SupervisedDataset)

    request = SFTRequest(model=model, data=dataset, settings=settings, training=training)
    assert request.data.descriptor.id == "datasets/posttrain-sft-smoke"
    assert request.data.descriptor.num_examples == 2


@pytest.mark.parametrize(
    ("target_id", "memory_gb", "accelerator_model", "architecture"),
    [
        ("targets/rtx-pro-4500-32gb", 32, "RTXPRO4500", "blackwell"),
        ("targets/rtx-pro-6000-96gb", 96, "RTXPRO6000", "blackwell"),
        ("targets/h100-80gb", 80, "H100", "hopper"),
        ("targets/h200-141gb", 141, "H200", "hopper"),
    ],
)
def test_framework_catalog_exposes_versioned_hardware_profile_facts(
    target_id: str,
    memory_gb: int,
    accelerator_model: str,
    architecture: str,
) -> None:
    catalog = open_catalog(scope="empty-project")

    target = catalog.resolve(CatalogRef("target", target_id)).value

    assert isinstance(target, ExecutionTarget)
    assert target.memory_gb == memory_gb
    assert target.hardware is not None
    assert target.hardware.accelerator_model == accelerator_model
    assert target.hardware.gpu_architecture == architecture
    assert target.hardware.supports_bf16 is True
    assert target.hardware.supports_mtp is True
    assert target.hardware.supports_turboquant is True
