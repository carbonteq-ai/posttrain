"""Tests for trainer-neutral sample rendering."""

from __future__ import annotations

import pytest
from posttrain.common.variants import (
    GEMMA_4_12B_IT,
    LFM_25_12B_INSTRUCT,
    LFM_25_12B_THINKING,
    LFM_25_350M,
    QWEN_35_2B,
)
from posttrain.data import PreferenceDataset, PreferenceExample, SupervisedDataset, SupervisedExample
from posttrain.train import (
    GEMMA4_RENDERER,
    LFM25_INSTRUCT_RENDERER,
    LFM25_RENDERER,
    QWEN35_RENDERER,
    render_preferences,
    render_supervised,
)

transformers = pytest.importorskip("transformers")
pytest.importorskip("renderers")


def _load_tokenizer(model):
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model.base.repo_id,
            revision=model.base.revision,
            local_files_only=True,
        )
    except OSError:
        pytest.skip(f"tokenizer for {model.base.repo_id}@{model.base.revision} is not cached")


@pytest.mark.parametrize(
    ("model", "profile"),
    (
        (QWEN_35_2B, QWEN35_RENDERER),
        (LFM_25_12B_THINKING, LFM25_RENDERER),
        (LFM_25_350M, LFM25_INSTRUCT_RENDERER),
        (LFM_25_12B_INSTRUCT, LFM25_INSTRUCT_RENDERER),
        (GEMMA_4_12B_IT, GEMMA4_RENDERER),
    ),
)
def test_renderer_builds_nonempty_assistant_only_sft_masks(model, profile) -> None:
    tokenizer = _load_tokenizer(model)
    dataset = SupervisedDataset(
        "gsm8k-sft-golden-v1",
        "a" * 40,
        (
            SupervisedExample(
                "gsm8k/train/0",
                (
                    {"role": "user", "content": "Solve 2 + 2 and end with #### N."},
                    {"role": "assistant", "content": "Two plus two is four.\n#### 4"},
                ),
                (1,),
            ),
        ),
    )
    sample = render_supervised(tokenizer, model, dataset, profile, max_length=512)[0]
    assert len(sample.input_ids) == len(sample.labels)
    assert any(label == -100 for label in sample.labels)
    assert any(label != -100 for label in sample.labels)
    decoded = tokenizer.decode(sample.input_ids, skip_special_tokens=False)
    assert "Solve 2 + 2" in decoded
    assert "#### 4" in decoded


@pytest.mark.parametrize("model", [LFM_25_350M, LFM_25_12B_INSTRUCT])
def test_lfm_instruct_sft_masks_only_assistant_and_preserves_stop_without_truncation(model) -> None:
    tokenizer = _load_tokenizer(model)
    target = '{"decision":"attach","edge_types":["CONDITIONED_BY"]}'
    dataset = SupervisedDataset(
        "policy-prism-lfm-instruct-sft-golden-v1",
        "c" * 40,
        (
            SupervisedExample(
                "policy-prism/train/0",
                (
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": "Classify the relationship."},
                    {"role": "assistant", "content": target},
                ),
                (2,),
            ),
        ),
    )

    sample = render_supervised(tokenizer, model, dataset, LFM25_INSTRUCT_RENDERER, max_length=512)[0]
    trained_ids = [
        token for token, label in zip(sample.input_ids, sample.labels, strict=True) if label != -100
    ]
    masked_ids = [
        token for token, label in zip(sample.input_ids, sample.labels, strict=True) if label == -100
    ]
    trained = tokenizer.decode(trained_ids, skip_special_tokens=False)
    masked = tokenizer.decode(masked_ids, skip_special_tokens=False)

    assert sample.source_length == len(sample.input_ids)
    assert sample.source_supervised_tokens == len(trained_ids)
    assert target in trained
    assert trained.endswith("<|im_end|>\n")
    assert "Classify the relationship." in masked
    assert "Classify the relationship." not in trained


def test_gemma_renderer_masks_tool_observations_from_sft_loss() -> None:
    tokenizer = _load_tokenizer(GEMMA_4_12B_IT)
    dataset = SupervisedDataset(
        "gemma4-tool-sft-golden-v1",
        "b" * 40,
        (
            SupervisedExample(
                "weather/train/0",
                (
                    {"role": "user", "content": "Weather in Lahore?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call-1",
                                "function": {"name": "weather", "arguments": {"city": "Lahore"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "name": "weather", "content": "Sunny"},
                    {"role": "assistant", "content": "It is sunny."},
                ),
                (1, 3),
                (
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "description": "Read weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    },
                ),
            ),
        ),
    )

    sample = render_supervised(tokenizer, GEMMA_4_12B_IT, dataset, GEMMA4_RENDERER, max_length=512)[0]
    trained = tokenizer.decode(
        [token for token, label in zip(sample.input_ids, sample.labels, strict=True) if label != -100],
        skip_special_tokens=False,
    )
    masked = tokenizer.decode(
        [token for token, label in zip(sample.input_ids, sample.labels, strict=True) if label == -100],
        skip_special_tokens=False,
    )

    assert "<|tool_call>call:weather" in trained
    assert "It is sunny." in trained
    assert "response:weather" in masked
    assert "Sunny" in masked


@pytest.mark.parametrize(
    ("model", "profile"),
    (
        (QWEN_35_2B, QWEN35_RENDERER),
        (LFM_25_12B_THINKING, LFM25_RENDERER),
    ),
)
def test_renderer_produces_equal_dpo_prompt_prefixes(model, profile) -> None:
    tokenizer = _load_tokenizer(model)
    dataset = PreferenceDataset(
        "gsm8k-dpo-golden-v1",
        "a" * 40,
        (
            PreferenceExample(
                "gsm8k/train/0",
                ({"role": "user", "content": "Solve 2 + 2 and end with #### N."},),
                ({"role": "assistant", "content": "Two plus two is four.\n#### 4"},),
                ({"role": "assistant", "content": "Two plus two is five.\n#### 5"},),
                1.0,
                0.0,
            ),
        ),
    )
    sample = render_preferences(tokenizer, model, dataset, profile, max_length=512)[0]
    assert sample.prompt_ids
    assert sample.chosen_ids
    assert sample.rejected_ids
    assert sample.chosen_ids != sample.rejected_ids
