"""Catalog-to-capability wiring that requires no project Python modules."""

from pathlib import Path

from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, ModelVariant
from posttrain.data import DatasetLoadPlan, SupervisedDataset, resolve_dataset_source
from posttrain.train import SFTRequest, SFTSettings, TrainingBinding


def test_gemma4_opd_variants_preserve_immutable_template_fingerprints() -> None:
    catalog = open_catalog(scope="empty-project")

    student = catalog.resolve(CatalogRef("model", "models/gemma4-e2b-it@bf16")).value
    teacher = catalog.resolve(CatalogRef("model", "models/gemma4-12b-it@bf16")).value

    assert isinstance(student, ModelVariant)
    assert isinstance(teacher, ModelVariant)
    assert student.tokenizer_fingerprint == teacher.tokenizer_fingerprint
    assert student.chat_template_fingerprint == (
        "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
    )
    assert teacher.chat_template_fingerprint == (
        "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
    )
    assert student.chat_template_fingerprint != teacher.chat_template_fingerprint


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
