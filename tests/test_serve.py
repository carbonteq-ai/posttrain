from __future__ import annotations

import unittest

from serve.benchmark import _controlled_prompt_ids, _hf_target, _percentile


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]


class ServeBenchmarkTest(unittest.TestCase):
    def test_parses_pinned_hugging_face_artifact(self) -> None:
        repository, revision = _hf_target("hf://org/model@abc123")
        self.assertEqual(repository, "org/model")
        self.assertEqual(revision, "abc123")

    def test_rejects_mutable_or_non_hugging_face_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable revision"):
            _hf_target("hf://org/model")
        with self.assertRaisesRegex(ValueError, "requires an hf"):
            _hf_target("trackio://models/model@v1")

    def test_controlled_prompts_have_exact_input_length(self) -> None:
        prompts = _controlled_prompt_ids(_Tokenizer(), 257, 8)

        self.assertEqual(len(prompts), 8)
        self.assertTrue(all(len(prompt["prompt_token_ids"]) == 257 for prompt in prompts))

    def test_percentile_handles_single_and_multiple_samples(self) -> None:
        self.assertEqual(_percentile([0.2], 0.95), 0.2)
        self.assertEqual(_percentile([0.1, 0.2, 0.3], 0.50), 0.2)


if __name__ == "__main__":
    unittest.main()
