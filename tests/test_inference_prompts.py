from __future__ import annotations

import unittest
from typing import Any

from posttrain.common.profiles import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.serve.prompts import (
    PromptError,
    PromptRecord,
    reasoning_template_kwargs,
    render_prompt,
    representative_prompt_records,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        self.messages = messages
        self.kwargs = kwargs
        return [1, 2, 3]


class InferencePromptTest(unittest.TestCase):
    def test_loads_canonical_messages_without_rendered_model_text(self) -> None:
        records = representative_prompt_records()

        self.assertEqual(len(records), 4)
        self.assertEqual(records[0].messages[0]["role"], "user")

    def test_qwen_maps_only_declared_reasoning_modes(self) -> None:
        self.assertEqual(
            reasoning_template_kwargs(QWEN_35_2B, "thinking"),
            {"enable_thinking": True},
        )
        self.assertEqual(
            reasoning_template_kwargs(QWEN_35_2B, "off"),
            {"enable_thinking": False},
        )
        with self.assertRaisesRegex(PromptError, "unsupported"):
            reasoning_template_kwargs(QWEN_35_2B, "medium")

    def test_lfm_thinking_checkpoint_exposes_only_native_mode(self) -> None:
        self.assertEqual(reasoning_template_kwargs(LFM_25_12B_THINKING, "native"), {})
        with self.assertRaisesRegex(PromptError, "unsupported"):
            reasoning_template_kwargs(LFM_25_12B_THINKING, "off")

    def test_render_passes_model_reasoning_and_tools_to_native_template(self) -> None:
        tokenizer = RecordingTokenizer()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Read weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        ids = render_prompt(
            tokenizer,
            PromptRecord("tool", ({"role": "user", "content": "Check weather"},), (), "thinking"),
            QWEN_35_2B,
            tools=tools,
        )

        self.assertEqual(ids, [1, 2, 3])
        self.assertEqual(tokenizer.kwargs["tools"], tools)
        self.assertTrue(tokenizer.kwargs["enable_thinking"])
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])


if __name__ == "__main__":
    unittest.main()
