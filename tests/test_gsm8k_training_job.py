from __future__ import annotations

from posttrain.common import ModelVariant
from posttrain.common.profiles import QWEN_35_2B
from posttrain.train import QWEN35_DPO_SMOKE, QWEN35_GRPO_SMOKE, QWEN35_SFT_SMOKE, DPORequest, SFTRequest
from posttrain_lab.data import RejectedRollout, load_gsm8k_supervised, preferences_from_rollouts
from posttrain_lab.data import gsm8k as gsm8k_data
from posttrain_lab.environments.gsm8k_grpo import _final_answer_conciseness
from posttrain_lab.jobs import (
    GSM8KGRPOJobRequest,
    dpo_action,
    grpo_action,
    grpo_job_inputs,
    sft_action,
    training_inputs,
)


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
                response="\n\nIncorrect reasoning.\n#### 3\n",
                score=0.0,
                trace_id="trace-123",
            ),
        ),
    )

    assert preferences.examples[0].chosen_score == 1.0
    assert preferences.examples[0].rejected_score == 0.0
    assert preferences.examples[0].rejected_trace_id == "trace-123"
    assert preferences.examples[0].rejected == "Incorrect reasoning.\n#### 3"


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


def test_grpo_job_config_records_environment_and_generation_policy() -> None:
    request = GSM8KGRPOJobRequest(
        ModelVariant.foundation(QWEN_35_2B),
        QWEN35_GRPO_SMOKE,
        (2,),
    )

    assert grpo_action(request).kind == "reinforcement-learning"
    inputs = grpo_job_inputs(request)
    assert inputs["environment_id"] == "gsm8k-v1"
    assert inputs["task_indices"] == "2"
    assert inputs["grpo_num_generations"] == 2
    assert inputs["rollout_engine"] == "vllm"
    assert inputs["rollout_sleep_during_optimization"] is True
    assert "rollout_speculative_method" not in inputs
    assert inputs["reward_shaping_id"] == "final-answer-conciseness-v1"


def test_grpo_shaping_requires_native_final_answer_format_and_rewards_concision() -> None:
    assert _final_answer_conciseness("Reasoning only", 64) == 0.0
    short = _final_answer_conciseness("Reasoning.\n#### 42", 64)
    long = _final_answer_conciseness("Long reasoning.\n#### 42", 128)
    assert short > long > 0.0
