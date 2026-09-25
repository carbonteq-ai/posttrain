"""Job validation runs the shared configuration rules on the job's recorded snapshot."""

import pytest
from posttrain.common import ContractError, ExecutionTarget, InferenceBinding
from posttrain.common.variants import LFM_25_26B
from posttrain.work.runner import _configuration_findings

TARGET = ExecutionTarget("targets/local-96gb", "1", "nvidia-cuda", 96)


def _rollout(acknowledgements: dict[str, str]) -> InferenceBinding:
    return InferenceBinding(
        "inference/lfm-rollout@1",
        "1",
        LFM_25_26B,
        "vllm@0.29.1.dev3",
        LFM_25_26B.renderer.id,
        {"enforce_eager": False},
        {"max_tokens": 4_096, "temperature": 0.8},
        TARGET,
        ("rollout",),
        performance_acknowledgements=acknowledgements,
    )


def test_rules_run_inside_job_configuration_findings() -> None:
    snapshot = {
        "rollout_inference": {
            "selection_id": "inference/lfm-rollout",
            "revision": "1",
            "resolved": {
                "backend": "vllm@0.29.1.dev3",
                "engine": {"enforce_eager": True, "enable_prefix_caching": True},
                "purpose": ["rollout"],
                "sampling": {"temperature": 0.8},
            },
        }
    }
    issues, _ = _configuration_findings({}, snapshot)
    assert [issue.code for issue in issues] == ["VLLM_EAGER_DISABLES_CUDA_GRAPHS"]


def test_acknowledgements_require_a_code_and_a_reason() -> None:
    with pytest.raises(ContractError, match="upper-case finding code"):
        _rollout({"eager": "because"})
    with pytest.raises(ContractError, match="needs a reason"):
        _rollout({"VLLM_EAGER_DISABLES_CUDA_GRAPHS": "  "})
