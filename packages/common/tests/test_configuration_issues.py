"""Tests for framework-neutral configuration explanation values."""

import pytest
from posttrain.common import ConfigurationIssue, ExecutionTarget, HardwareCapabilities, SettingOrigin


def test_setting_origin_serializes_explicit_false_without_treating_it_as_omitted() -> None:
    origin = SettingOrigin("rollout.engine.enforce_eager", False, "explicit", "inference/qwen-rollout@2", "2")

    assert origin.as_dict() == {
        "path": "rollout.engine.enforce_eager",
        "value": False,
        "kind": "explicit",
        "source": "inference/qwen-rollout@2",
        "revision": "2",
    }


def test_configuration_issue_has_stable_safe_shape() -> None:
    issue = ConfigurationIssue(
        "MODEL_CONTEXT_BUDGET_CONFLICT",
        "error",
        "static",
        "policy",
        "rollout.engine.max_model_len",
        "prompt and completion budgets exceed the engine context",
        "increase context or reduce an explicit budget",
        ("settings.max_prompt_length", "settings.max_completion_length"),
    )

    assert issue.as_dict()["code"] == "MODEL_CONTEXT_BUDGET_CONFLICT"
    assert issue.as_dict()["related_paths"] == ["settings.max_prompt_length", "settings.max_completion_length"]


def test_execution_target_keeps_declared_hardware_facts_typed() -> None:
    target = ExecutionTarget(
        "target/rtx-pro-6000",
        "1",
        "cuda",
        96,
        hardware=HardwareCapabilities(
            accelerator_count=1,
            accelerator_model="RTXPRO6000",
            gpu_architecture="blackwell",
            supports_bf16=True,
            supports_mtp=True,
            supports_turboquant=True,
        ),
    )

    assert target.hardware is not None
    assert target.hardware.accelerator_model == "RTXPRO6000"
    assert target.hardware.supports_turboquant is True


@pytest.mark.parametrize("code", ["lower_case", "HAS-DASH", ""])
def test_configuration_issue_rejects_unstable_codes(code: str) -> None:
    with pytest.raises(ValueError, match="uppercase snake case"):
        ConfigurationIssue(code, "warning", "static", "policy", "model", "message")
