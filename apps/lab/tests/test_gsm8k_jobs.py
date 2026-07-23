"""Tests for the lab GSM8K job compositions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from posttrain.common.variants import QWEN_35_2B
from posttrain.data import PreferencePairSource, RolloutDataset, RolloutExample, ScoredContinuation
from posttrain.train import (
    QWEN35_DPO_SMOKE,
    QWEN35_SFT_SMOKE,
    DPORequest,
    EnvironmentRolloutBridge,
    GRPORequest,
    SFTRequest,
)
from posttrain_lab.catalog import (
    AUTOMATIONBENCH_ZAPIER_GRPO,
    QWEN35_AUTOMATIONBENCH_GRPO_MTP,
    QWEN35_AUTOMATIONBENCH_TRL_LORA_THINKING,
    QWEN35_TRL_QLORA,
    QWEN_35_08B_BF16,
    QWEN_AUTOMATIONBENCH_GRPO_MTP_VLLM,
)
from posttrain_lab.data import GSM8KSupervisedSource
from posttrain_lab.data import gsm8k as gsm8k_data
from posttrain_lab.environments.gsm8k_grpo import _final_answer_conciseness
from posttrain_lab.jobs import (
    grpo_job_inputs,
    training_inputs,
)


def test_gsm8k_supervised_data_uses_environment_prompt_and_pinned_revision(monkeypatch) -> None:
    rows = [
        {"question": "What is 1 + 1?", "answer": "One plus one is two.\n#### 2"},
        {"question": "What is 2 + 3?", "answer": "Two plus three is five.\n#### 5"},
    ]
    monkeypatch.setattr(gsm8k_data, "_rows", lambda split: rows)
    monkeypatch.setattr(gsm8k_data, "_system_prompt", lambda: "environment-owned prompt")

    source = GSM8KSupervisedSource(count=1, offset=1)
    dataset = source.load()

    assert dataset.id == "gsm8k/train-1-2-v1"
    assert dataset.revision == gsm8k_data.GSM8K_REVISION
    assert dataset.examples[0].id == "train/000001"
    assert dataset.examples[0].messages[0]["content"] == "environment-owned prompt"
    assert str(dataset.examples[0].messages[2]["content"]).endswith("#### 5")
    assert dataset.examples[0].trainable_message_indices == (2,)
    assert dataset.descriptor == source.descriptor


def test_preferences_require_trace_derived_failed_rollouts(monkeypatch) -> None:
    rows = [{"question": "What is 1 + 1?", "answer": "Reasoning.\n#### 2"}]
    monkeypatch.setattr(gsm8k_data, "_rows", lambda split: rows)
    monkeypatch.setattr(gsm8k_data, "_system_prompt", lambda: "system")
    demonstrations = GSM8KSupervisedSource(count=1)

    source = PreferencePairSource(
        demonstrations=demonstrations,
        candidates=(
            ScoredContinuation(
                example_id="train/000000",
                messages=({"role": "assistant", "content": "Incorrect reasoning.\n#### 3"},),
                score=0.0,
                trace_id="trace-123",
                metadata={"source_project": "shared", "source_run_id": "run-123"},
            ),
        ),
        id_suffix="trace-preferences",
    )
    preferences = source.load()

    assert preferences.examples[0].chosen_score == 1.0
    assert preferences.examples[0].rejected_score == 0.0
    assert preferences.examples[0].rejected_trace_id == "trace-123"
    assert preferences.examples[0].rejected[0]["content"] == "Incorrect reasoning.\n#### 3"
    assert preferences.descriptor == source.descriptor
    assert preferences.metadata["candidate_trace_ids"] == ["trace-123"]
    assert preferences.metadata["candidate_sources"] == [{"source_project": "shared", "source_run_id": "run-123"}]


def test_training_run_config_preserves_model_and_data_identity(monkeypatch) -> None:
    rows = [{"question": "What is 1 + 1?", "answer": "Reasoning.\n#### 2"}]
    monkeypatch.setattr(gsm8k_data, "_rows", lambda split: rows)
    monkeypatch.setattr(gsm8k_data, "_system_prompt", lambda: "system")
    demonstrations = GSM8KSupervisedSource(count=1)
    model = QWEN_35_2B
    sft_request = SFTRequest(model, demonstrations, QWEN35_SFT_SMOKE, QWEN35_TRL_QLORA)
    preferences = PreferencePairSource(
        demonstrations,
        (
            ScoredContinuation(
                "train/000000",
                ({"role": "assistant", "content": "Wrong.\n#### 3"},),
                0.0,
                "trace-123",
            ),
        ),
    )
    dpo_request = DPORequest(model, preferences, QWEN35_DPO_SMOKE, QWEN35_TRL_QLORA)

    assert training_inputs(sft_request)["model_variant_id"] == QWEN_35_2B.id
    assert training_inputs(sft_request)["base_model_revision"] == QWEN_35_2B.base.revision
    assert training_inputs(dpo_request)["dpo_beta"] == QWEN35_DPO_SMOKE.beta


def test_grpo_job_config_records_environment_and_generation_policy() -> None:
    request = GRPORequest(
        policy=QWEN_35_08B_BF16,
        bridge=cast(
            EnvironmentRolloutBridge,
            SimpleNamespace(
                dataset=RolloutDataset(
                    "automationbench/simple",
                    AUTOMATIONBENCH_ZAPIER_GRPO.revision,
                    (RolloutExample("train/000000", "task", {}),),
                )
            ),
        ),
        environment=AUTOMATIONBENCH_ZAPIER_GRPO,
        settings=QWEN35_AUTOMATIONBENCH_GRPO_MTP,
        training=QWEN35_AUTOMATIONBENCH_TRL_LORA_THINKING,
        inference=QWEN_AUTOMATIONBENCH_GRPO_MTP_VLLM,
    )

    inputs = grpo_job_inputs(request)
    assert inputs["environment_id"] == "automationbench-zapier-simple-grpo"
    assert inputs["environment_package"] == "automationbench-v1"
    assert inputs["environment_domains"] == "simple"
    assert inputs["environment_sampling_seed"] == 17
    assert inputs["environment_toolset"] == "zapier"
    assert "task_indices" not in inputs
    assert "dataset_id" not in inputs
    assert inputs["grpo_num_generations"] == 8
    assert inputs["grpo_temperature"] == 1.0
    assert inputs["grpo_top_p"] == 0.95
    assert inputs["rollout_engine"] == "vllm"
    assert inputs["rollout_sleep_during_optimization"] is True
    assert inputs["rollout_skip_multimodal_profiling"] is True
    assert inputs["rollout_vllm_weight_sync_mode"] == "lora"
    assert inputs["rollout_speculative_method"] == "mtp"
    assert inputs["reasoning_mode"] == "thinking"


def test_grpo_shaping_requires_native_final_answer_format_and_rewards_concision() -> None:
    assert _final_answer_conciseness("Reasoning only", 64) == 0.0
    short = _final_answer_conciseness("Reasoning.\n#### 42", 64)
    long = _final_answer_conciseness("Long reasoning.\n#### 42", 128)
    assert short > long > 0.0
