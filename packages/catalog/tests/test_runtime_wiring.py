"""Catalog-to-capability wiring that requires no project Python modules."""

from pathlib import Path

from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, InferenceBinding, ModelVariant
from posttrain.data import DatasetLoadPlan, SupervisedDataset, resolve_dataset_source
from posttrain.train import LoRAUpdate, SFTRequest, SFTSettings, TrainingBinding


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


def test_lfm_instruct_catalog_resolves_exact_eval_and_sft_contracts(tmp_path: Path) -> None:
    catalog = open_catalog(scope="empty-project")
    dataset_plan = catalog.resolve(CatalogRef("dataset", "datasets/posttrain-sft-smoke@1")).value
    assert isinstance(dataset_plan, DatasetLoadPlan)
    dataset = resolve_dataset_source(
        dataset_plan,
        state_dir=tmp_path / ".posttrain" / "state",
        project_root=tmp_path,
    )
    assert isinstance(dataset, SupervisedDataset)

    expected = (
        (
            "models/lfm2.5-350m@bf16",
            "inference/lfm2.5-350m-instruct-vllm-eval@1",
            "LiquidAI/LFM2.5-350M",
            "9e6c6ccf47cd318696e137d381a7ded8fe4df09f",
        ),
        (
            "models/lfm2.5-1.2b-instruct@bf16",
            "inference/lfm2.5-1.2b-instruct-vllm-eval@1",
            "LiquidAI/LFM2.5-1.2B-Instruct",
            "df58c174f05ff733f83f8cae10ea9298224c8006",
        ),
    )
    training = catalog.resolve(CatalogRef("training", "training/lfm2.5-instruct-trl-lora@1")).value
    assert isinstance(training, TrainingBinding)
    assert isinstance(training.update, LoRAUpdate)
    assert training.update.target_modules == "all-linear"
    assert training.renderer.id == "lfm2.5-instruct-v1"
    assert training.renderer.reasoning_mode == "off"

    settings_ids = (
        "lfm2.5-350m-instruct/sft-smoke-v1",
        "lfm2.5-1.2b-instruct/sft-smoke-v1",
    )
    for (model_id, inference_id, repo_id, revision), settings_id in zip(expected, settings_ids, strict=True):
        model = catalog.resolve(CatalogRef("model", model_id)).value
        inference = catalog.resolve(CatalogRef("inference", inference_id)).value
        settings = catalog.resolve(CatalogRef("training", settings_id)).value
        assert isinstance(model, ModelVariant)
        assert isinstance(inference, InferenceBinding)
        assert model.base.repo_id == repo_id
        assert model.base.revision == revision
        assert model.renderer.id == "lfm2.5-instruct-tools@1"
        assert inference.model is model
        assert inference.renderer == model.renderer.id
        assert inference.engine["dtype"] == "bfloat16"
        assert inference.engine["max_model_len"] == 32_768
        assert inference.engine["tool_call_parser"] == "lfm2"
        assert "reasoning_parser" not in inference.engine
        assert isinstance(settings, SFTSettings)
        request = SFTRequest(model=model, data=dataset, settings=settings, training=training)
        assert request.training.renderer.model_family == request.model.family
        assert request.data.descriptor.num_examples == 2
