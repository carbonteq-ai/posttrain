from __future__ import annotations

import unittest

from common import BENCHMARKS_DIR, PROFILES_DIR, ProfileResolver
from serve.prompts import PromptError, load_prompt_records, reasoning_template_kwargs


class InferencePromptTest(unittest.TestCase):
    def test_loads_canonical_messages_without_rendered_model_text(self) -> None:
        records = load_prompt_records(
            BENCHMARKS_DIR / "inference" / "corpora" / "representative-v1.jsonl"
        )

        self.assertEqual(len(records), 4)
        self.assertEqual(records[0].messages[0]["role"], "user")

    def test_qwen_maps_only_declared_reasoning_modes(self) -> None:
        profile = ProfileResolver(PROFILES_DIR).resolve("models", "qwen3.5-2b")

        self.assertEqual(
            reasoning_template_kwargs(profile.data, "thinking"),
            {"enable_thinking": True},
        )
        self.assertEqual(
            reasoning_template_kwargs(profile.data, "off"),
            {"enable_thinking": False},
        )
        with self.assertRaisesRegex(PromptError, "unsupported"):
            reasoning_template_kwargs(profile.data, "medium")

    def test_lfm_thinking_checkpoint_exposes_only_native_mode(self) -> None:
        profile = ProfileResolver(PROFILES_DIR).resolve(
            "models", "lfm2.5-1.2b-thinking"
        )

        self.assertEqual(reasoning_template_kwargs(profile.data, "native"), {})
        with self.assertRaisesRegex(PromptError, "unsupported"):
            reasoning_template_kwargs(profile.data, "off")


if __name__ == "__main__":
    unittest.main()
