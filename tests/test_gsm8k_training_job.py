from __future__ import annotations

from posttrain.common import ModelVariant
from posttrain.common.profiles import QWEN_35_2B
from posttrain.train import QWEN35_DPO_SMOKE, QWEN35_SFT_SMOKE, DPORequest, SFTRequest
from posttrain_lab.data import RejectedRollout, load_gsm8k_supervised, preferences_from_rollouts
from posttrain_lab.data import gsm8k as gsm8k_data
from posttrain_lab.jobs import dpo_action, sft_action, training_inputs


def test_gsm8k_supervised_data_uses_environment_prompt_and_pinned_revision(monkeypatch) -> None:
    rows = [
        {"question": "What is 1 + 1?", "answer": "One plus one is two.\n#### 2"},
        {"question": "What is 2 + 3?", "answer": "Two plus three is five.\n#### 5"},
    ]
    monkeypatch.setattr(gsm8k_data, "_rows", lambda split: rows)
    monkeypatch.setattr(gsm8k_data, "_system_prompt", lambda: "environment-owned prompt")

    dataset = load_gsm8k_supervised(count=1, offset=1)

    assert dataset.id == "gsm8k/train-1-2-v1"
    assert dataset.revision == gsm8k_data.GSM8K_REVISION
    assert dataset.examples[0].id == "train/000001"
    assert dataset.examples[0].system_prompt == "environment-owned prompt"
    assert dataset.examples[0].response.endswith("#### 5")


def test_preferences_require_trace_derived_failed_rollouts(monkeypatch) -> None:
    rows = [{"question": "What is 1 + 1?", "answer": "Reasoning.\n#### 2"}]
    monkeypatch.setattr(gsm8k_data, "_rows", lambda split: rows)
    monkeypatch.setattr(gsm8k_data, "_system_prompt", lambda: "system")
    demonstrations = load_gsm8k_supervised(count=1)

    preferences = preferences_from_rollouts(
        demonstrations,
        (
            RejectedRollout(
                example_id="train/000000",
                response="Incorrect reasoning.\n#### 3",
                score=0.0,
                trace_id="trace-123",
            ),
        ),
    )

    assert preferences.examples[0].chosen_score == 1.0
    assert preferences.examples[0].rejected_score == 0.0
    assert preferences.examples[0].rejected_trace_id == "trace-123"


def test_training_actions_and_run_config_preserve_model_and_data_identity(monkeypatch) -> None:
    rows = [{"question": "What is 1 + 1?", "answer": "Reasoning.\n#### 2"}]
    monkeypatch.setattr(gsm8k_data, "_rows", lambda split: rows)
    monkeypatch.setattr(gsm8k_data, "_system_prompt", lambda: "system")
    demonstrations = load_gsm8k_supervised(count=1)
    model = ModelVariant.foundation(QWEN_35_2B)
    sft_request = SFTRequest(model, demonstrations, QWEN35_SFT_SMOKE)
    preferences = preferences_from_rollouts(
        demonstrations,
        (RejectedRollout("train/000000", "Wrong.\n#### 3", 0.0, "trace-123"),),
    )
    dpo_request = DPORequest(model, preferences, QWEN35_DPO_SMOKE)

    assert sft_action(sft_request).kind == "supervised-finetuning"
    assert dpo_action(dpo_request).kind == "preference-optimization"
    assert training_inputs(sft_request)["base_model_revision"] == QWEN_35_2B.artifact.revision
    assert training_inputs(dpo_request)["dpo_beta"] == QWEN35_DPO_SMOKE.beta
