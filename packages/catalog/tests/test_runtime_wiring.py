"""Catalog-to-capability wiring that requires no project Python modules."""

from pathlib import Path

from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, ModelVariant
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
