"""Focused tests for family-aware TRL model loading."""

import sys
from types import ModuleType

import pytest
from posttrain.common.variants import GEMMA_4_12B_IT, LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.train.backends.trl.common import trainable_model_factory, vllm_rollout_options


class CausalFactory:
    pass


class MultimodalFactory:
    pass


IMPORTS = {
    "AutoModelForCausalLM": CausalFactory,
    "AutoModelForMultimodalLM": MultimodalFactory,
}


def test_trainable_model_factory_uses_multimodal_loader_for_gemma4() -> None:
    assert trainable_model_factory(QWEN_35_2B, IMPORTS) is CausalFactory
    assert trainable_model_factory(LFM_25_12B_THINKING, IMPORTS) is CausalFactory
    assert trainable_model_factory(GEMMA_4_12B_IT, IMPORTS) is MultimodalFactory


def test_gemma_mtp_materializes_the_pinned_assistant_before_trl(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    hub = ModuleType("huggingface_hub")

    def snapshot_download(*, repo_id: str, revision: str) -> str:
        calls.append((repo_id, revision))
        return "/var/lib/posttrain/cache/huggingface/models--gemma-assistant/snapshots/pinned"

    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    speculative, kwargs = vllm_rollout_options(
        GEMMA_4_12B_IT,
        {
            "mode": "colocate",
            "speculative_config": {
                "method": "mtp",
                "num_speculative_tokens": 1,
                "assistant_model": "google/gemma-4-12B-it-assistant",
                "assistant_revision": "364bd03c9952e5b7da73665ee30c9eccfc408345",
            },
        },
    )
    assert speculative == {
        "method": "mtp",
        "num_speculative_tokens": 1,
        "model": "/var/lib/posttrain/cache/huggingface/models--gemma-assistant/snapshots/pinned",
    }
    assert kwargs == {"disable_log_stats": False}
    assert calls == [("google/gemma-4-12B-it-assistant", "364bd03c9952e5b7da73665ee30c9eccfc408345")]


def test_gemma_mtp_rejects_an_unpinned_or_incomplete_assistant() -> None:
    with pytest.raises(ValueError, match="assistant_model and assistant_revision"):
        vllm_rollout_options(
            GEMMA_4_12B_IT,
            {"mode": "colocate", "speculative_config": {"method": "mtp", "num_speculative_tokens": 1}},
        )
    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        vllm_rollout_options(
            GEMMA_4_12B_IT,
            {
                "mode": "colocate",
                "speculative_config": {
                    "method": "mtp",
                    "num_speculative_tokens": 1,
                    "assistant_model": "google/gemma-4-12B-it-assistant",
                    "assistant_revision": "main",
                },
            },
        )
