"""Tests for common model renderer contracts."""

from __future__ import annotations

import hashlib

import pytest
from posttrain.common.variants import (
    GEMMA_4_12B_IT,
    GEMMA_4_31B_IT,
    LFM_25_12B_INSTRUCT,
    LFM_25_12B_THINKING,
    LFM_25_350M,
    QWEN_35_2B,
)

TOOLS = [
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
    }
]


def _local_tokenizer(profile):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            profile.base.repo_id,
            revision=profile.base.revision,
            local_files_only=True,
        )
    except OSError:
        pytest.skip(f"pinned tokenizer is not cached: {profile.base.repo_id}")


def test_pinned_qwen_template_renders_tools_and_thinking_control() -> None:
    tokenizer = _local_tokenizer(QWEN_35_2B)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Weather?"}],
        tools=TOOLS,
        enable_thinking=True,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert "<tools>" in rendered
    assert "<tool_call>" in rendered
    assert rendered.endswith("<think>\n")


def test_lfm_package_template_preserves_openai_tool_history() -> None:
    tokenizer = _local_tokenizer(LFM_25_12B_THINKING)
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "weather", "arguments": {"city": "Paris"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
        ],
        tools=TOOLS,
        chat_template=LFM_25_12B_THINKING.conversation.chat_template.text(),
        tokenize=False,
        add_generation_prompt=True,
    )

    assert '<|tool_call_start|>[weather(city="Paris")]<|tool_call_end|>' in rendered
    assert "<|im_start|>tool\nsunny<|im_end|>" in rendered
    assert "null" not in rendered


def test_lfm_instruct_template_is_exact_and_separate_from_thinking() -> None:
    instruct = LFM_25_12B_INSTRUCT.conversation.chat_template.text()
    thinking = LFM_25_12B_THINKING.conversation.chat_template.text()

    assert instruct is not None
    assert hashlib.sha256(instruct.encode()).hexdigest() == (
        "ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6"
    )
    assert instruct != thinking
    assert LFM_25_350M.conversation.chat_template.text() == instruct


@pytest.mark.parametrize("variant", [LFM_25_350M, LFM_25_12B_INSTRUCT])
def test_lfm_instruct_template_renders_deterministic_chatml_without_reasoning(variant) -> None:
    tokenizer = _local_tokenizer(variant)
    messages = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "Classify this clause."},
        {"role": "assistant", "content": '{"decision":"attach"}'},
    ]
    template = variant.conversation.chat_template.text()
    rendered = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False)
    repeated = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False)

    assert rendered == repeated
    assert rendered.startswith("<|startoftext|><|im_start|>system\n")
    assert rendered.endswith('{"decision":"attach"}<|im_end|>\n')
    assert "<think>" not in rendered


@pytest.mark.parametrize("variant", [GEMMA_4_12B_IT, GEMMA_4_31B_IT])
def test_pinned_gemma_template_renders_thinking_and_structured_tool_history(variant) -> None:
    tokenizer = _local_tokenizer(variant)
    prompt_off = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Think briefly."}],
        enable_thinking=False,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_on = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Think briefly."}],
        enable_thinking=True,
        tokenize=False,
        add_generation_prompt=True,
    )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {"name": "weather", "arguments": {"city": "Paris"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "weather", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ],
        tools=TOOLS,
        enable_thinking=False,
        tokenize=False,
    )

    assert prompt_off.endswith("<|channel>thought\n<channel|>")
    assert "<|think|>" not in prompt_off
    assert "<|think|>" in prompt_on
    assert prompt_on.endswith("<|turn>model\n")
    assert "<|tool>declaration:weather" in rendered
    assert "<|tool_call>call:weather" in rendered
    assert "<|tool_response>response:weather" in rendered
    assert rendered.endswith("It is sunny.<turn|>\n")
