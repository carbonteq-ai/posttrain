"""Batch invariance is a recorded engine setting that becomes vLLM's environment switch."""

import pytest
from posttrain.serve.profiles.base import VllmEngineConfig


def test_batch_invariant_engines_export_the_vllm_switch_and_no_cli_flag() -> None:
    engine = VllmEngineConfig(max_model_len=4_096, gpu_memory_utilization=0.9, batch_invariant=True)
    assert engine.environment() == {"VLLM_BATCH_INVARIANT": "1"}
    assert "batch_invariant" not in engine.as_vllm_kwargs()
    assert VllmEngineConfig(max_model_len=4_096, gpu_memory_utilization=0.9).environment() == {}


def test_batch_invariant_must_be_a_boolean() -> None:
    with pytest.raises(ValueError, match="batch_invariant"):
        VllmEngineConfig(max_model_len=4_096, gpu_memory_utilization=0.9, batch_invariant="yes")  # type: ignore[arg-type]
