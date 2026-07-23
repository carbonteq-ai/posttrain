"""Tests for common model renderer contracts."""

from __future__ import annotations

import pytest
from posttrain.common.variants import LFM_25_12B_THINKING, QWEN_35_2B

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
