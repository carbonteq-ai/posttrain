"""Tests for the lab-local Gemma 4 Halcyon GraphQL SFT runs."""

from __future__ import annotations

import math
import os

import pytest
from posttrain.common import Catalog
from posttrain.data import SupervisedDataset, supervised_from_huggingface
from posttrain.train import SFTRequest, render_supervised
from posttrain_lab.cli import _gemma4_halcyon_canary, _gemma4_halcyon_full_sft, _parser
from posttrain_lab.gemma4_halcyon import (
    GEMMA4_HALCYON_CANARY,
    GEMMA4_HALCYON_LORA,
    GEMMA4_HALCYON_LORA_FULL,
    GEMMA4_HALCYON_SFT,
    GEMMA4_MODEL_REVISION,
    GEMMA4_TARGET_MODULES,
    GEMMA_4_12B_IT,
    RUNPOD_RTX_PRO_6000_96GB,
)
from posttrain_lab.work_packages import WorkPackageContext, run_work_package, sft_definition


def _row() -> dict[str, object]:
    return {
        "id": "halcyon/synthetic/compositional",
        "messages": [
            {"role": "system", "content": "Call execute_graphql exactly once."},
            {"role": "user", "content": "Return matching active entities and repeated identifiers."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "execute_graphql",
                            "arguments": {
                                "query": "query Lookup($filter: Filter!) { entities(filter: $filter) { id id } }",
                                "variables": {"filter": {"active": False, "nested": {"names": ["A", "B"]}}},
                                "compact": True,
                            },
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_graphql",
                    "description": "Execute one read-only GraphQL query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "variables": {"type": "object"},
                            "compact": {"type": "boolean"},
                        },
                        "required": ["query", "variables", "compact"],
                    },
                },
            }
        ],
        "trainable_message_indices": [2],
    }


def test_lab_selections_are_pinned_and_language_model_scoped() -> None:
    protocol = GEMMA_4_12B_IT.conversation.tool_calls
    assert GEMMA_4_12B_IT.base.repo_id == "google/gemma-4-12B-it"
    assert GEMMA_4_12B_IT.base.revision == GEMMA4_MODEL_REVISION
    assert GEMMA_4_12B_IT.parameters == 11_959_730_176
    assert GEMMA_4_12B_IT.capabilities.modalities == ("text", "image", "audio")
    assert GEMMA_4_12B_IT.capabilities.native_context_window == 262_144
    assert GEMMA_4_12B_IT.default_reasoning_mode == "thinking"
    assert protocol is not None
    assert (protocol.start_token, protocol.end_token) == ("<|tool_call>", "<tool_call|>")
    assert RUNPOD_RTX_PRO_6000_96GB.memory_gb == 96
    assert GEMMA4_HALCYON_LORA.update.kind == "lora"
    assert GEMMA4_HALCYON_LORA.update.target_modules == GEMMA4_TARGET_MODULES
    assert GEMMA4_HALCYON_LORA.backend_options["use_liger_kernel"] is False
    assert GEMMA4_HALCYON_CANARY.loop.max_steps == 1
    assert GEMMA4_HALCYON_CANARY.loop.max_length == 2_048
    assert GEMMA4_HALCYON_CANARY.validation is not None
    assert GEMMA4_HALCYON_CANARY.validation.on_start is True


def test_full_profile_is_two_deterministic_passes_and_keeps_canary_distinct() -> None:
    loop = GEMMA4_HALCYON_SFT.loop
    validation = GEMMA4_HALCYON_SFT.validation

    assert loop.max_steps == 98
    assert loop.max_length == 2_048
    assert loop.per_device_batch_size == 1
    assert loop.gradient_accumulation_steps == 8
    assert loop.per_device_batch_size * loop.gradient_accumulation_steps == 8
    assert 392 // (loop.per_device_batch_size * loop.gradient_accumulation_steps) == 49
    assert loop.max_steps == 2 * 49
    assert loop.learning_rate == 0.0001
    assert loop.warmup_ratio == 0.05
    assert math.ceil(loop.max_steps * loop.warmup_ratio) == 5
    assert loop.max_grad_norm == 1.0
    assert loop.logging_steps == 1
    assert loop.checkpoint_steps == 49
    assert loop.checkpoint_limit == 2
    assert loop.seed == 42
    assert loop.gradient_checkpointing is True
    assert validation is not None
    assert validation.steps == 49
    assert validation.per_device_batch_size == 1
    assert validation.on_start is True
    assert validation.at_end is True
    assert GEMMA4_HALCYON_LORA_FULL.runtime.global_batch_size == 8
    assert GEMMA4_HALCYON_LORA_FULL.update is GEMMA4_HALCYON_LORA.update
    assert GEMMA4_HALCYON_LORA_FULL.renderer is GEMMA4_HALCYON_LORA.renderer
    assert GEMMA4_HALCYON_LORA_FULL.id != GEMMA4_HALCYON_LORA.id
    assert GEMMA4_HALCYON_SFT.id != GEMMA4_HALCYON_CANARY.id
    assert GEMMA4_HALCYON_CANARY.loop.max_steps == 1
    assert GEMMA4_HALCYON_LORA.runtime.global_batch_size == 1


def test_cli_accepts_full_and_canary_scenarios() -> None:
    parser = _parser()

    assert parser.parse_args(["gemma4-halcyon-graphql-sft"]).job == "gemma4-halcyon-graphql-sft"
    assert parser.parse_args(["gemma4-halcyon-graphql-sft-canary"]).job == "gemma4-halcyon-graphql-sft-canary"


def test_canary_composes_five_explicit_sft_seats() -> None:
    package, concrete = _gemma4_halcyon_canary("halcyon-graphql-sft")
    echo = sft_definition(
        lambda context, request: request,
        definition_id=concrete.id,
        with_validation=True,
    )

    result = run_work_package(
        WorkPackageContext(Catalog.open({}, scope="halcyon-graphql-sft"), {echo.id: echo}),
        package,
    )

    request = result.jobs[0].value
    assert isinstance(request, SFTRequest)
    assert set(concrete.seats) == {"model", "dataset", "validation_dataset", "settings", "training"}
    assert request.model is GEMMA_4_12B_IT
    assert request.data.descriptor.id == "halcyon-graphql-stage1/train-v2"
    assert request.validation_data is not None
    assert request.validation_data.descriptor.id == "halcyon-graphql-stage1/validation-v2"
    assert request.settings is GEMMA4_HALCYON_CANARY
    assert request.training is GEMMA4_HALCYON_LORA


def test_full_sft_composes_five_explicit_seats_from_foundation() -> None:
    package, concrete = _gemma4_halcyon_full_sft("halcyon-graphql-sft")
    echo = sft_definition(
        lambda context, request: request,
        definition_id=concrete.id,
        with_validation=True,
    )

    result = run_work_package(
        WorkPackageContext(Catalog.open({}, scope="halcyon-graphql-sft"), {echo.id: echo}),
        package,
    )

    request = result.jobs[0].value
    assert isinstance(request, SFTRequest)
    assert package.work_package_id == "train/gemma4-12b/halcyon-graphql-sft"
    assert set(concrete.seats) == {"model", "dataset", "validation_dataset", "settings", "training"}
    assert request.model is GEMMA_4_12B_IT
    assert request.model.form == "foundation"
    assert request.data.descriptor.id == "halcyon-graphql-stage1/train-v2"
    assert request.validation_data is not None
    assert request.validation_data.descriptor.id == "halcyon-graphql-stage1/validation-v2"
    assert request.settings is GEMMA4_HALCYON_SFT
    assert request.training is GEMMA4_HALCYON_LORA_FULL


def _tokenizer():
    if not os.getenv("HF_TOKEN"):
        pytest.skip("pinned Gemma tokenizer integration requires HF_TOKEN")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        GEMMA_4_12B_IT.base.repo_id,
        revision=GEMMA_4_12B_IT.base.revision,
        token=os.environ["HF_TOKEN"],
        trust_remote_code=False,
    )


@pytest.mark.network
def test_tokenizer_renders_native_tool_call_with_assistant_only_labels() -> None:
    pytest.importorskip("renderers")
    tokenizer = _tokenizer()
    normalized = supervised_from_huggingface(
        [_row()],
        dataset_id="halcyon/synthetic",
        revision="fixture-v1",
        format="messages",
    )
    dataset = SupervisedDataset(
        normalized.id,
        normalized.revision,
        normalized.examples,
        schema_version=2,
    )

    first = render_supervised(
        tokenizer,
        GEMMA_4_12B_IT,
        dataset,
        GEMMA4_HALCYON_LORA.renderer,
        max_length=2_048,
    )[0]
    second = render_supervised(
        tokenizer,
        GEMMA_4_12B_IT,
        dataset,
        GEMMA4_HALCYON_LORA.renderer,
        max_length=2_048,
    )[0]

    assert first == second
    assert first.source_length == len(first.input_ids)
    assert any(label == -100 for label in first.labels)
    assert any(label != -100 for label in first.labels)
    decoded = tokenizer.decode(first.input_ids, skip_special_tokens=False)
    assert "<|tool_call>call:execute_graphql" in decoded
    assert "<tool_call|>" in decoded
    assert "query Lookup" in decoded
    assert "active:false" in decoded
    assert "compact:true" in decoded
    assert "<|channel>thought" not in decoded
    for boundary in ("<|tool_call>", "<tool_call|>"):
        token_id = tokenizer.convert_tokens_to_ids(boundary)
        positions = [index for index, value in enumerate(first.input_ids) if value == token_id]
        assert positions
        assert all(first.labels[index] != -100 for index in positions)


@pytest.mark.network
def test_pinned_config_uses_unified_class_and_lora_stays_in_language_model() -> None:
    if not os.getenv("HF_TOKEN"):
        pytest.skip("pinned Gemma architecture integration requires HF_TOKEN")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    accelerate = pytest.importorskip("accelerate")
    config = transformers.AutoConfig.from_pretrained(
        GEMMA_4_12B_IT.base.repo_id,
        revision=GEMMA_4_12B_IT.base.revision,
        token=os.environ["HF_TOKEN"],
        trust_remote_code=False,
    )
    with accelerate.init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        adapted = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=GEMMA4_TARGET_MODULES,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

    trainable = [(name, parameter) for name, parameter in adapted.named_parameters() if parameter.requires_grad]
    assert type(model).__name__ == "Gemma4UnifiedForConditionalGeneration"
    assert sum(parameter.numel() for _, parameter in trainable) == 32_784_384
    assert trainable
    assert all(".language_model." in name for name, _ in trainable)
