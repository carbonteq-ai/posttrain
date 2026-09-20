"""Tests for provider-free job validation reports."""

from dataclasses import replace
from typing import cast

from posttrain.catalog import open_catalog
from posttrain.common import (
    Catalog,
    CatalogLayer,
    CatalogRef,
    ExecutionTarget,
    InferenceBinding,
    JsonValue,
    ModelVariant,
    SettingOrigin,
)
from posttrain.common.variants import LFM_25_26B
from posttrain.train import LFM25_RENDERER, LoRAUpdate, TrainingBinding, TrainingParallelism, TrainingRuntime
from posttrain.work import JobValidationReport, ValidationCheck, WorkPackageContext
from posttrain.work.runner import ResolvedSeat, _configuration_findings, _execution_target_snapshot, _readiness_checks


def test_job_validation_report_is_stable_and_bound_to_resolved_inputs() -> None:
    origins = (SettingOrigin("policy.reasoning_mode", "off", "default", "qwen-renderer@1"),)
    checks = (ValidationCheck("static-configuration", "passed", "static checks passed"),)

    first = JobValidationReport.for_resolved_inputs({"b": 2, "a": {"value": 1}}, origins=origins, checks=checks)
    reordered = JobValidationReport.for_resolved_inputs({"a": {"value": 1}, "b": 2}, origins=origins, checks=checks)
    changed = JobValidationReport.for_resolved_inputs({"a": {"value": 1}, "b": 3}, origins=origins, checks=checks)

    assert first.resolved_input_digest == reordered.resolved_input_digest
    assert first.resolved_input_digest != changed.resolved_input_digest
    report_origins = cast(list[dict[str, JsonValue]], first.as_dict()["origins"])
    assert report_origins[0]["kind"] == "default"


def test_skip_preflight_skips_only_an_optional_host_probe() -> None:
    calls: list[object] = []

    def readiness_probe(seats):
        calls.append(seats)
        return (ValidationCheck("endpoint", "passed", "endpoint identity matched"),)

    context = WorkPackageContext(
        Catalog(CatalogLayer("base", {}), (), "test"),
        {},
        readiness_probe=readiness_probe,
    )

    skipped = _readiness_checks(context, {}, skip_preflight=True)
    checked = _readiness_checks(context, {}, skip_preflight=False)

    assert calls == [{}]
    assert skipped[0].outcome == "skipped"
    assert checked[0].outcome == "passed"


def test_hardware_advice_recommends_qualified_accelerations_without_mutating_binding() -> None:
    catalog = open_catalog(scope="empty-project")
    model = cast(ModelVariant, catalog.resolve(CatalogRef("model", "models/gemma4-12b-it@bf16")).value)
    target = cast(
        ExecutionTarget,
        catalog.resolve(CatalogRef("target", "targets/rtx-pro-6000-96gb")).value,
    )
    binding = InferenceBinding(
        "inference/gemma4-test@1",
        "1",
        model,
        "vllm@0.25.1",
        model.renderer.id,
        {"max_model_len": 4096},
        {"max_tokens": 512},
        target,
        ("eval",),
    )

    issues, checks = _configuration_findings({"judge": binding})

    assert {issue.code for issue in issues} == {"MTP_AVAILABLE", "TURBOQUANT_AVAILABLE"}
    assert all(issue.severity == "recommendation" for issue in issues)
    assert "speculative_config" not in binding.engine
    assert "kv_cache_dtype" not in binding.engine
    assert checks[0].outcome == "passed"

    teacher_issues, _ = _configuration_findings({"teacher": replace(binding, purpose=("teacher-score",))})
    assert "MTP_AVAILABLE" not in {issue.code for issue in teacher_issues}


def test_job_compiler_rejects_turboquant_with_flash_attention_four() -> None:
    catalog = open_catalog(scope="empty-project")
    model = cast(ModelVariant, catalog.resolve(CatalogRef("model", "models/gemma4-12b-it@bf16")).value)
    target = cast(
        ExecutionTarget,
        catalog.resolve(CatalogRef("target", "targets/rtx-pro-6000-96gb")).value,
    )
    binding = InferenceBinding(
        "inference/gemma4-invalid-tq-fa4@1",
        "1",
        model,
        "vllm@fbbba6698b2f8a912b94705cfc09eb4fd7243716",
        model.renderer.id,
        {
            "max_model_len": 16_384,
            "kv_cache_dtype": "turboquant_k8v4",
            "flash_attn_version": 4,
        },
        {"max_tokens": 1_024},
        target,
        ("rollout",),
    )

    issues, _ = _configuration_findings({"rollout_inference": binding})

    issue = next(issue for issue in issues if issue.code == "VLLM_TURBOQUANT_FLASH_ATTN_INCOMPATIBLE")
    assert issue.severity == "error"
    assert issue.path == "rollout_inference.engine.flash_attn_version"
    assert issue.related_paths == ("rollout_inference.engine.kv_cache_dtype",)


def test_job_compiler_rejects_native_flash_attention_four_on_sm120() -> None:
    catalog = open_catalog(scope="empty-project")
    model = cast(ModelVariant, catalog.resolve(CatalogRef("model", "models/lfm2.5-2.6b@bf16")).value)
    target = cast(
        ExecutionTarget,
        catalog.resolve(CatalogRef("target", "targets/rtx-pro-6000-96gb")).value,
    )
    binding = InferenceBinding(
        "inference/lfm-invalid-sm120-fa4@1",
        "1",
        model,
        "vllm@fbbba6698b2f8a912b94705cfc09eb4fd7243716",
        model.renderer.id,
        {"max_model_len": 24_576, "flash_attn_version": 4},
        {"max_tokens": 4_096},
        target,
        ("rollout",),
    )

    issues, _ = _configuration_findings({"rollout_inference": binding})

    issue = next(issue for issue in issues if issue.code == "VLLM_NATIVE_FA4_UNSUPPORTED_ON_SM120")
    assert issue.severity == "error"
    assert issue.path == "rollout_inference.engine.flash_attn_version"
    assert issue.related_paths == ("rollout_inference.target.hardware.accelerator_model",)
    assert "RTXPRO6000" in issue.message


def test_job_compiler_rejects_pinned_dspark_with_turboquant() -> None:
    catalog = open_catalog(scope="empty-project")
    model = cast(ModelVariant, catalog.resolve(CatalogRef("model", "models/nanbeige4.2-3b@bf16")).value)
    target = cast(
        ExecutionTarget,
        catalog.resolve(CatalogRef("target", "targets/rtx-pro-6000-96gb")).value,
    )
    binding = InferenceBinding(
        "inference/nanbeige-invalid-dspark-tq@1",
        "1",
        model,
        "vllm@62f6de733d7ae63b759329993bc209e67afdf431",
        model.renderer.id,
        {
            "max_model_len": 16_384,
            "kv_cache_dtype": "turboquant_k8v4",
            "flash_attn_version": 2,
            "speculative_config": {"method": "dspark", "num_speculative_tokens": 7},
        },
        {"max_tokens": 1_024},
        target,
        ("rollout",),
    )

    issues, _ = _configuration_findings({"rollout_inference": binding})

    issue = next(issue for issue in issues if issue.code == "VLLM_DSPARK_TURBOQUANT_INCOMPATIBLE")
    assert issue.severity == "error"
    assert "non-causal draft attention" in issue.message


def test_training_topology_cannot_request_more_devices_than_exact_target() -> None:
    catalog = open_catalog(scope="empty-project")
    training = cast(
        TrainingBinding,
        catalog.resolve(CatalogRef("training", "training/qwen3.5-trl-lora@1")).value,
    )
    target = cast(
        ExecutionTarget,
        catalog.resolve(CatalogRef("target", "targets/rtx-pro-4500-32gb")).value,
    )
    invalid = replace(
        training,
        target=target,
        runtime=TrainingRuntime(nodes=1, devices_per_node=2),
        parallelism=TrainingParallelism(tensor_parallel_size=4),
    )

    issues, checks = _configuration_findings({"training": invalid})

    assert {issue.code for issue in issues} == {
        "TRAINING_DEVICES_EXCEED_TARGET",
        "TRAINING_PARALLELISM_EXCEEDS_TOPOLOGY",
    }
    assert checks[0].outcome == "passed"


def test_execution_target_snapshot_retains_exact_accelerator_model() -> None:
    catalog = open_catalog(scope="empty-project")
    target = cast(
        ExecutionTarget,
        catalog.resolve(CatalogRef("target", "targets/rtx-pro-4500-32gb")).value,
    )

    snapshot = _execution_target_snapshot({"evaluation": ResolvedSeat("evaluation", target, None, "base")})

    entry = cast(dict[str, JsonValue], snapshot[0])
    hardware = cast(dict[str, JsonValue], entry["hardware"])
    assert hardware["accelerator_model"] == "RTXPRO4500"


def test_colocated_trl_rejects_policy_weight_floor_above_target_memory() -> None:
    target = ExecutionTarget("targets/local-8gb", "1", "nvidia-cuda", 8)
    training = TrainingBinding(
        "training/lfm-local@1",
        "1",
        "trl@1.12.0",
        LFM25_RENDERER,
        LoRAUpdate(rank=4, alpha=8),
        target,
    )
    rollout = InferenceBinding(
        "inference/lfm-local@1",
        "1",
        LFM_25_26B,
        "vllm@0.25.1",
        LFM_25_26B.renderer.id,
        {"mode": "colocate", "sleep_during_optimization": True},
        {"max_tokens": 512},
        target,
        ("rollout",),
    )

    issues, _ = _configuration_findings({"training": training, "rollout_inference": rollout})

    issue = next(issue for issue in issues if issue.code == "COLOCATED_TRL_WEIGHT_FLOOR_EXCEEDS_TARGET")
    assert issue.severity == "error"
    assert "10.02 GiB" in issue.message
    assert "during optimization only" in (issue.hint or "")


def test_colocated_trl_weight_floor_is_not_a_fit_claim() -> None:
    target = ExecutionTarget("targets/local-12gb", "1", "nvidia-cuda", 12)
    training = TrainingBinding(
        "training/lfm-local@1",
        "1",
        "trl@1.12.0",
        LFM25_RENDERER,
        LoRAUpdate(rank=4, alpha=8),
        target,
    )
    rollout = InferenceBinding(
        "inference/lfm-local@1",
        "1",
        LFM_25_26B,
        "vllm@0.25.1",
        LFM_25_26B.renderer.id,
        {"mode": "colocate", "sleep_during_optimization": True},
        {"max_tokens": 512},
        target,
        ("rollout",),
    )

    issues, _ = _configuration_findings({"training": training, "rollout_inference": rollout})

    assert "COLOCATED_TRL_WEIGHT_FLOOR_EXCEEDS_TARGET" not in {issue.code for issue in issues}
