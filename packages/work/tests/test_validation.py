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
from posttrain.train import TrainingBinding, TrainingParallelism, TrainingRuntime
from posttrain.work import JobValidationReport, ValidationCheck, WorkPackageContext
from posttrain.work.runner import ResolvedSeat, _configuration_findings, _execution_target_snapshot, _readiness_checks


def test_job_validation_report_is_stable_and_bound_to_resolved_inputs() -> None:
    origins = (SettingOrigin("policy.reasoning_mode", "off", "default", "qwen-renderer@1"),)
    checks = (ValidationCheck("static-configuration", "passed", "static checks passed"),)

    first = JobValidationReport.for_resolved_inputs(
        {"b": 2, "a": {"value": 1}}, origins=origins, checks=checks
    )
    reordered = JobValidationReport.for_resolved_inputs(
        {"a": {"value": 1}, "b": 2}, origins=origins, checks=checks
    )
    changed = JobValidationReport.for_resolved_inputs(
        {"a": {"value": 1}, "b": 3}, origins=origins, checks=checks
    )

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

    teacher_issues, _ = _configuration_findings(
        {"teacher": replace(binding, purpose=("teacher-score",))}
    )
    assert "MTP_AVAILABLE" not in {issue.code for issue in teacher_issues}


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

    snapshot = _execution_target_snapshot(
        {"evaluation": ResolvedSeat("evaluation", target, None, "base")}
    )

    entry = cast(dict[str, JsonValue], snapshot[0])
    hardware = cast(dict[str, JsonValue], entry["hardware"])
    assert hardware["accelerator_model"] == "RTXPRO4500"
