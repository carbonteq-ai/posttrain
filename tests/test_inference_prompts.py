from __future__ import annotations

import unittest

from posttrain.common.profiles import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.serve.prompts import PromptError, reasoning_template_kwargs, representative_prompt_records


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


if __name__ == "__main__":
    unittest.main()
