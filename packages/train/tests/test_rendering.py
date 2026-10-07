"""Tests for trainer-neutral sample rendering."""

from __future__ import annotations

import pytest
from posttrain.common.variants import GEMMA_4_12B_IT, LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.data import PreferenceDataset, PreferenceExample, SupervisedDataset, SupervisedExample
from posttrain.train import (
    GEMMA4_RENDERER,
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


@pytest.mark.parametrize("trainable", [(2,), (4,), (2, 4)])
def test_lfm_sft_targets_exclude_injected_headers_and_tool_results(trainable: tuple[int, ...]) -> None:
    tokenizer = _load_tokenizer(LFM_25_12B_THINKING)
    messages = (
        {"role": "system", "content": "SYSTEM_SENTINEL"},
        {"role": "user", "content": "USER_SENTINEL"},
        {
            "role": "assistant",
            "content": "FIRST_ANSWER",
            "tool_calls": [{"type": "function", "id": "a", "function": {"name": "lookup", "arguments": '{"key":"x"}'}}],
        },
        {"role": "tool", "tool_call_id": "a", "content": "TOOL_SENTINEL"},
        {"role": "assistant", "content": "FINAL_ANSWER"},
    )
    dataset = SupervisedDataset("lfm-mask-regression", "c" * 40, (SupervisedExample("case", messages, trainable),))
    sample = render_supervised(tokenizer, LFM_25_12B_THINKING, dataset, LFM25_RENDERER, max_length=512)[0]
    targets = tokenizer.decode([label for label in sample.labels if label != -100], skip_special_tokens=False)
    assert all(
        marker not in targets
        for marker in ["SYSTEM_SENTINEL", "USER_SENTINEL", "TOOL_SENTINEL", "<|im_start|>assistant"]
    )
    assert ("FIRST_ANSWER" in targets) == (2 in trainable)
    assert ("FINAL_ANSWER" in targets) == (4 in trainable)
    assert [label for label in sample.labels if label != -100][-1] == tokenizer.convert_tokens_to_ids("<|im_end|>")


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


def test_lfm_full_rerender_accepts_sampled_turns_with_leading_newlines() -> None:
    from posttrain.train.rendering import create_renderer, full_render_messages

    tokenizer = _load_tokenizer(LFM_25_12B_THINKING)
    renderer = create_renderer(tokenizer, LFM_25_12B_THINKING, LFM25_RENDERER)
    call = {"id": "call_0", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    tool = {"type": "function", "function": {"name": "lookup", "description": "Look up.", "parameters": {}}}
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Find it."},
        # A parsed LFM tool-call turn keeps the newline after its reasoning block.
        {"role": "assistant", "content": "\n", "reasoning_content": "think", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_0", "name": "lookup", "content": "error"},
        {"role": "assistant", "content": "\n\nDone.", "reasoning_content": "think"},
    ]
    with pytest.raises(ValueError, match="generation-prompt token prefix"):
        renderer.render(messages[:3], tools=[tool], add_generation_prompt=True)
    for end in (3, 4, 5):
        rendered = renderer.render(full_render_messages(messages[:end]), tools=[tool], add_generation_prompt=True)
        assert rendered.token_ids
