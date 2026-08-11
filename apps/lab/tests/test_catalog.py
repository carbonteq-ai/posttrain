"""Tests for the lab filesystem catalog."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import pytest
from posttrain.catalog import load_catalog_layer, packaged_base_directory
from posttrain.common import CatalogRef, ContractError, ExecutionTarget, InferenceBinding, ModelVariant
from posttrain.environment import VerifiersV1ConfigActivation
from posttrain.eval import EnvironmentBinding, EnvironmentSource, EvaluationPlan
from posttrain.train import (
    DynamicGroupSampling,
    GRPOSettings,
    LoRAUpdate,
    OnPolicyDistillationSettings,
    QLoRAUpdate,
    QuantizationPlan,
    SAMPOSettings,
    SFTSettings,
    TrainingBinding,
)
from posttrain_lab.catalog import open_catalog

WORKSPACE = Path(__file__).resolve().parents[3]


def test_slice_one_and_two_selections_load_from_the_base_filesystem_catalog() -> None:
    catalog = open_catalog(scope="posttrain-lab")

    model = catalog.resolve(CatalogRef("model", "models/qwen3.5-2b@bf16"))
    plan = catalog.resolve(CatalogRef("evaluation", "general-smoke-v1"))
    environment = catalog.resolve(CatalogRef("environment", "math-gsm8k"))
    dapo = catalog.resolve(CatalogRef("training", "qwen3.5-2b/dapo-smoke-v1"))
    sampo = catalog.resolve(CatalogRef("training", "qwen3.5-2b/sampo-smoke-v1"))

    assert isinstance(model.value, ModelVariant)
    assert model.value.renderer.id == "qwen3.5-tools@1"
    assert model.source_layer == "base"
    assert isinstance(plan.value, EvaluationPlan)
    assert plan.value.environment("math-gsm8k") is environment.value
    assert plan.source_layer == "base"
    assert isinstance(dapo.value, GRPOSettings)
    assert dapo.value.algorithm == "dapo"
    assert dapo.value.dynamic_sampling == DynamicGroupSampling(max_candidate_batches=10)
    assert isinstance(sampo.value, SAMPOSettings)
    assert sampo.value.discount_gamma == 0.95
    assert sampo.value.dynamic_sampling == DynamicGroupSampling(max_candidate_batches=3)


def test_gemma4_unified_qualification_selections_resolve_as_one_support_plane() -> None:
    catalog = open_catalog(
        scope="posttrain-lab",
        overlays=(WORKSPACE / "apps" / "lab" / ".posttrain" / "catalog",),
    )
    model = catalog.resolve(CatalogRef("model", "models/gemma4-12b-it@bf16"))
    settings = catalog.resolve(CatalogRef("training", "gemma4-12b-it/sft-qualification-v1"))
    training = catalog.resolve(CatalogRef("training", "training/gemma4-12b-it-trl-lora-qualification@1"))
    mtp_settings = catalog.resolve(CatalogRef("training", "gemma4-12b-it/grpo-mtp-qualification-v1"))
    mtp_training = catalog.resolve(CatalogRef("training", "training/gemma4-12b-it-trl-grpo-mtp-qualification@1"))
    mtp_inference = catalog.resolve(CatalogRef("inference", "inference/gemma4-12b-it-vllm-grpo-mtp@1"))
    mtp_environment = catalog.resolve(CatalogRef("environment", "gemma4-gsm8k-mtp-qualification"))
    inference = catalog.resolve(CatalogRef("inference", "inference/gemma4-12b-it-vllm-screen@1"))
    evaluation_inference = catalog.resolve(CatalogRef("inference", "inference/gemma4-12b-it-vllm-eval@1"))

    assert isinstance(model.value, ModelVariant)
    assert model.value.family == "gemma4"
    assert model.value.provenance["upstream_model_type"] == "gemma4_unified"
    assert model.value.capabilities.mtp is True
    assert model.value.provenance["mtp_assistant_repo_id"] == "google/gemma-4-12B-it-assistant"
    assert model.value.renderer.id == "gemma4-tools@1"
    assert model.source_layer == "base"
    assert isinstance(settings.value, SFTSettings)
    assert settings.value.loop.max_steps == 2
    assert isinstance(training.value, TrainingBinding)
    assert isinstance(training.value.update, LoRAUpdate)
    assert training.value.renderer.implementation == "default"
    assert training.value.renderer.reasoning_mode == "off"
    assert training.value.target.id == "targets/carbonteq-rtx-pro-6000-96gb"
    assert "language_model" in training.value.update.target_modules
    assert isinstance(inference.value, InferenceBinding)
    assert inference.value.model == model.value
    assert inference.value.target == training.value.target
    assert inference.value.engine["text_only"] is True
    assert inference.value.engine["skip_mm_profiling"] is True
    assert inference.value.engine["tool_call_parser"] == "gemma4"
    assert inference.value.engine["reasoning_parser"] == "gemma4"
    assert isinstance(evaluation_inference.value, InferenceBinding)
    assert evaluation_inference.value.model == model.value
    assert evaluation_inference.value.target == training.value.target
    assert evaluation_inference.value.purpose == ("eval",)
    assert evaluation_inference.value.engine["max_model_len"] == 32768
    assert isinstance(mtp_settings.value, GRPOSettings)
    assert mtp_settings.value.loop.max_steps == 1
    assert mtp_settings.value.num_generations == 2
    assert isinstance(mtp_training.value, TrainingBinding)
    assert mtp_training.value.target == training.value.target
    assert isinstance(mtp_inference.value, InferenceBinding)
    speculative_config = mtp_inference.value.engine["speculative_config"]
    assert isinstance(speculative_config, Mapping)
    assert speculative_config["assistant_revision"] == ("364bd03c9952e5b7da73665ee30c9eccfc408345")
    assert isinstance(mtp_environment.value, EnvironmentBinding)
    assert mtp_environment.value.max_concurrent == 1
    assert all(value.source_layer == "overlay" for value in (settings, training, inference, evaluation_inference))
    assert all(
        value.overlay_id == "posttrain-lab-serving-capacity-v1"
        for value in (settings, training, inference, evaluation_inference)
    )


def test_gemma4_dense_matrix_selections_resolve_with_shared_runtime_contract() -> None:
    catalog = open_catalog(
        scope="posttrain-lab",
        overlays=(WORKSPACE / "apps" / "lab" / ".posttrain" / "catalog",),
    )
    expected = {
        "e2b": ("models/gemma4-e2b-it@bf16", "inference/gemma4-e2b-it-vllm-screen@1"),
        "e4b": ("models/gemma4-e4b-it@bf16", "inference/gemma4-e4b-it-vllm-screen@1"),
        "12b": ("models/gemma4-12b-it@bf16", "inference/gemma4-12b-it-vllm-screen@1"),
        "31b": ("models/gemma4-31b-it@bf16", "inference/gemma4-31b-it-vllm-screen@1"),
    }
    for model_id, inference_id in expected.values():
        model = catalog.resolve(CatalogRef("model", model_id)).value
        inference = catalog.resolve(CatalogRef("inference", inference_id)).value
        assert isinstance(model, ModelVariant)
        assert isinstance(inference, InferenceBinding)
        assert inference.model == model
        assert model.family == "gemma4"
        assert model.capabilities.mtp is True
        assert inference.engine["text_only"] is True
        assert inference.engine["tool_call_parser"] == "gemma4"
        assert inference.engine["reasoning_parser"] == "gemma4"


def test_every_evaluation_plan_declares_success_for_every_environment() -> None:
    catalog = open_catalog(scope="posttrain-lab")

    for ref in catalog.list("evaluation"):
        plan = catalog.resolve(ref).value
        assert isinstance(plan, EvaluationPlan)
        assert set(plan.success) == {environment.id for environment in plan.environments}


def test_automationbench_grpo_environment_is_category_and_budget_driven() -> None:
    catalog = open_catalog(scope="posttrain-lab")
    environment = catalog.resolve(CatalogRef("environment", "automationbench-zapier-simple-grpo")).value
    settings = catalog.resolve(CatalogRef("training", "automationbench/qwen3.5-0.8b/grpo-mtp-smoke-v1")).value
    training = catalog.resolve(
        CatalogRef("training", "training/qwen3.5-0.8b-trl-automationbench-lora-thinking@1")
    ).value

    assert isinstance(environment, EnvironmentBinding)
    assert isinstance(environment.source, EnvironmentSource)
    assert environment.source.package == "automationbench-v1"
    assert environment.source.repository == "https://github.com/carbonteq-ai/verifiers-environments"
    assert environment.source.revision == "b7bcb591facfcd2b073802f6d7496b24ab9c479e"
    assert environment.source.subdirectory == "environments/automationbench_v1"
    assert environment.parameters["domains"] == ["simple"]
    assert environment.parameters["sampling_seed"] == 17
    assert environment.parameters["toolset"] == "zapier"
    assert environment.required_inference_capabilities == ("tool-calling",)
    assert "task_indices" not in environment.parameters
    assert environment.num_tasks == 2
    assert environment.num_rollouts == 8
    assert isinstance(settings, GRPOSettings)
    assert settings.loop.max_steps == 1
    assert settings.loop.per_device_batch_size == 1
    assert settings.loop.gradient_accumulation_steps == 16
    assert settings.num_generations == environment.num_rollouts
    assert isinstance(training, TrainingBinding)
    assert isinstance(training.update, LoRAUpdate)
    assert training.renderer.reasoning_mode == "thinking"
    assert training.update.target_modules == r".*[.](o_proj|down_proj)$"
    assert training.runtime.global_batch_size == 16

    rollout = catalog.resolve(
        CatalogRef("inference", "inference/qwen3.5-0.8b-vllm-automationbench-rollout-mtp@1")
    ).value
    assert isinstance(rollout, InferenceBinding)
    assert rollout.capabilities == ("tool-calling",)
    assert rollout.model.conversation.tool_calls is not None
    assert rollout.model.conversation.tool_calls.id == "qwen3_xml"


def test_qwen4b_automationbench_eval_binding_declares_tool_protocol() -> None:
    catalog = open_catalog(scope="posttrain-lab", overlays=(WORKSPACE / "apps/lab/.posttrain/catalog",))
    environment = catalog.resolve(CatalogRef("environment", "automationbench-zapier-simple-qualification")).value
    inference = catalog.resolve(
        CatalogRef("inference", "inference/qwen3.5-4b-vllm-thinking-eval-mtp-concurrent16-32k@1")
    ).value

    assert isinstance(environment, EnvironmentBinding)
    assert environment.required_inference_capabilities == ("tool-calling",)
    assert environment.num_tasks == 200
    assert environment.max_concurrent == 16
    assert isinstance(environment.activation, VerifiersV1ConfigActivation)
    assert environment.activation.config.get("taskset") == {
        "id": "automationbench-v1",
        "domains": ["simple"],
        "task": {"toolset": "limited_zapier", "search_top_k": 20},
    }
    assert isinstance(inference, InferenceBinding)
    assert inference.capabilities == ("tool-calling",)
    assert inference.engine["max_num_seqs"] == 16
    assert inference.model.conversation.tool_calls is not None
    assert inference.model.conversation.tool_calls.id == "qwen3_xml"


def test_general_capability_catalog_and_library_qualification_are_pinned() -> None:
    catalog = open_catalog(scope="posttrain-lab")
    plan = catalog.resolve(CatalogRef("evaluation", "general-capability-balanced-v1")).value
    assert isinstance(plan, EvaluationPlan)
    assert [item.id for item in plan.environments] == [
        "knowledge-mmlu-pro-cot-5shot-balanced-v1",
        "instruction-ifeval-full-v1",
        "reasoning-gym-balanced-eval-v1",
        "math-python-balanced-eval-v1",
    ]
    assert {item.num_tasks for item in plan.environments} == {1400, 541, 1000, 500}
    for item in plan.environments:
        assert isinstance(item.source, EnvironmentSource)
        assert item.source.repository == "https://github.com/carbonteq-ai/verifiers-environments"
        assert item.source.revision == "b7bcb591facfcd2b073802f6d7496b24ab9c479e"


def test_project_overlay_directory_can_publish_a_new_selection(tmp_path: Path) -> None:
    overlay = _layer(
        tmp_path,
        "project-example-v1",
        """
target:
  targets/project-cuda-24gb:
    revision: "1"
    device_class: nvidia-cuda
    memory_gb: 24
    placement:
      world_size: 1
""",
    )
    catalog = open_catalog(
        scope="example",
        overlays=(overlay,),
    )
    resolved = catalog.resolve(CatalogRef("target", "targets/project-cuda-24gb"))

    assert isinstance(resolved.value, ExecutionTarget)
    assert resolved.value.memory_gb == 24
    assert resolved.source_layer == "overlay"
    assert resolved.overlay_id == "project-example-v1"


def test_peft_bindings_settings_and_quantization_load_from_filesystem_catalog() -> None:
    catalog = open_catalog(scope="posttrain-lab")

    lora = catalog.resolve(CatalogRef("training", "training/qwen3.5-trl-lora@1"))
    qlora = catalog.resolve(CatalogRef("training", "training/qwen3.5-trl-qlora@1"))
    settings = catalog.resolve(CatalogRef("training", "qwen3.5-2b/sft-smoke-v2"))
    quantization = catalog.resolve(CatalogRef("quantization", "qwen3.5-2b/awq-4bit-v1"))
    rtn = catalog.resolve(CatalogRef("quantization", "qwen3.5-2b/rtn-w4a16-v3"))

    assert isinstance(lora.value, TrainingBinding)
    assert isinstance(lora.value.update, LoRAUpdate)
    assert isinstance(qlora.value, TrainingBinding)
    assert isinstance(qlora.value.update, QLoRAUpdate)
    assert isinstance(settings.value, SFTSettings)
    assert isinstance(quantization.value, QuantizationPlan)
    assert quantization.value.calibration is not None
    assert quantization.value.calibration.dataset_id == "openai/gsm8k"
    assert isinstance(rtn.value, QuantizationPlan)
    assert rtn.value.method == "rtn"
    assert rtn.value.calibration is None
    assert rtn.value.excluded_modules == ("model.visual",)
    assert rtn.value.output_quantization["scope"] == "language_model"
    assert rtn.value.output_quantization["zero_point"] is False
    root = Path(__file__).resolve().parents[3]
    assert (
        lora.value.backend_options["dependency_lock_sha256"]
        == hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    )
    assert lora.value.backend_options["dependency_lock"] == "trl-fork@current"
    assert (
        quantization.value.dependency_lock_digest
        == hashlib.sha256((root / "tools" / "quantization" / "uv.lock").read_bytes()).hexdigest()
    )
    assert (
        quantization.value.recipe_digest == hashlib.sha256((root / quantization.value.recipe).read_bytes()).hexdigest()
    )
    assert rtn.value.recipe_digest == hashlib.sha256((root / rtn.value.recipe).read_bytes()).hexdigest()
    assert all(value.source_layer == "base" for value in (lora, qlora, settings, quantization, rtn))


def test_distillation_selections_share_exact_tokenizer_identity_and_separate_targets() -> None:
    catalog = open_catalog(scope="posttrain-lab")
    student = catalog.resolve(CatalogRef("model", "models/qwen3.5-0.8b@bf16")).value
    teacher = catalog.resolve(CatalogRef("model", "models/qwen3.5-2b@bf16")).value
    settings = catalog.resolve(CatalogRef("training", "qwen3.5-0.8b/on-policy-distill-smoke-v1")).value
    rollout = catalog.resolve(CatalogRef("inference", "inference/qwen3.5-0.8b-vllm-distill-rollout@1")).value
    scoring = catalog.resolve(CatalogRef("inference", "inference/qwen3.5-2b-vllm-teacher-score@1")).value

    assert isinstance(student, ModelVariant)
    assert isinstance(teacher, ModelVariant)
    assert student.tokenizer_fingerprint == teacher.tokenizer_fingerprint
    assert student.tokenizer_fingerprint == "544bc020ecb01661a305ed3ba1fffe49011d65eed195b059457edb69db4ded0c"
    assert isinstance(settings, OnPolicyDistillationSettings)
    assert "rollout" in rollout.purpose  # type: ignore[union-attr]
    assert "teacher-score" in scoring.purpose  # type: ignore[union-attr]
    assert rollout.target.id != scoring.target.id  # type: ignore[union-attr]


def test_overlay_can_shadow_an_existing_id_and_add_a_new_id(tmp_path: Path) -> None:
    overlay = _layer(
        tmp_path,
        "project-test-v1",
        """
target:
  targets/local-cuda-8gb:
    revision: '2'
    device_class: nvidia-cuda
    memory_gb: 16
    placement: {world_size: 1}
  targets/project-cpu:
    revision: '1'
    device_class: cpu
""",
    )
    catalog = open_catalog(scope="tests", overlays=(overlay,))

    shadowed = catalog.resolve(CatalogRef("target", "targets/local-cuda-8gb"))
    added = catalog.resolve(CatalogRef("target", "targets/project-cpu"))
    assert isinstance(shadowed.value, ExecutionTarget)
    assert shadowed.value.revision == "2"
    assert shadowed.source_layer == "overlay"
    assert shadowed.overlay_id == "project-test-v1"
    assert added.source_layer == "overlay"


def test_invalid_catalog_yaml_is_rejected_at_the_host_boundary(tmp_path: Path) -> None:
    directory = tmp_path / "invalid"
    directory.mkdir()
    (directory / "layer.yaml").write_text(
        "schema_version: 1\nlayer_id: invalid-v1\nfiles: [unknown.yaml]\n",
        encoding="utf-8",
    )
    (directory / "unknown.yaml").write_text(
        "unknown_family:\n  value: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="catalog_family_unavailable: unknown_family"):
        open_catalog(scope="invalid", overlays=(directory,))


def test_invalid_selection_fields_fail_before_catalog_is_returned(tmp_path: Path) -> None:
    overlay = _layer(
        tmp_path,
        "invalid-target-v1",
        """
target:
  targets/invalid:
    revision: '1'
    device_class: cpu
    embedded_engine_knob: true
""",
    )

    with pytest.raises(ContractError, match="extra_forbidden"):
        open_catalog(scope="tests", overlays=(overlay,))


def test_broken_catalog_links_fail_during_resolution_not_gpu_execution(tmp_path: Path) -> None:
    overlay = _layer(
        tmp_path,
        "broken-links-v1",
        """
inference:
  inference/broken@1:
    revision: '1'
    model: models/missing@bf16
    backend: vllm@0.25.1
    renderer: missing@1
    engine: {}
    sampling: {max_tokens: 1}
    target: targets/local-cuda-8gb
    purpose: [screen]
""",
    )

    with pytest.raises(ContractError, match="unresolved catalog link"):
        open_catalog(scope="tests", overlays=(overlay,))


def test_base_catalog_manifest_is_complete_and_manifest_controlled() -> None:
    with packaged_base_directory() as base:
        layer = load_catalog_layer(base)

    assert layer["layer_id"] == "framework-v1"
    assert set(layer) == {
        "layer_id",
        "model",
        "dataset",
        "target",
        "inference",
        "workload",
        "environment",
        "evaluation",
        "training",
        "quantization",
    }


def _layer(root: Path, layer_id: str, document: str) -> Path:
    directory = root / layer_id
    directory.mkdir()
    (directory / "layer.yaml").write_text(
        f"schema_version: 1\nlayer_id: {layer_id}\nfiles: [entries.yaml]\n",
        encoding="utf-8",
    )
    (directory / "entries.yaml").write_text(document.strip() + "\n", encoding="utf-8")
    return directory
