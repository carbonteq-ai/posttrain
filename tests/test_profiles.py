from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common.profiles import ProfileError, ProfileResolver
from posttrain.common.profiles import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.serve.profiles import (
    LFM25_VLLM,
    LFM25_VLLM_TURBOQUANT_K8,
    QWEN35_VLLM_MTP,
    QWEN35_VLLM_TEXT,
    QWEN35_VLLM_TURBOQUANT_K8,
)


class ProfileResolverTest(unittest.TestCase):
    def test_lfm_foundation_and_serve_profiles_are_typed_and_separate(self) -> None:
        self.assertEqual(LFM_25_12B_THINKING.family, "lfm2.5")
        self.assertEqual(LFM_25_12B_THINKING.capabilities.native_context_window, 32_768)
        self.assertEqual(LFM25_VLLM.model_family, LFM_25_12B_THINKING.family)
        self.assertEqual(LFM25_VLLM.engine.max_model_len, 4_096)
        self.assertEqual(LFM25_VLLM_TURBOQUANT_K8.engine.max_model_len, 32_768)
        self.assertEqual(LFM25_VLLM_TURBOQUANT_K8.engine.kv_cache_dtype, "turboquant_k8v4")

    def test_qwen_exposes_standard_turboquant_and_mtp_serve_variants(self) -> None:
        self.assertEqual(QWEN_35_2B.capabilities.native_context_window, 262_144)
        self.assertEqual(QWEN35_VLLM_TEXT.model_family, QWEN_35_2B.family)
        self.assertEqual(QWEN35_VLLM_TURBOQUANT_K8.engine.max_model_len, 32_768)
        self.assertEqual(QWEN35_VLLM_TURBOQUANT_K8.engine.kv_cache_dtype, "turboquant_k8v4")
        assert QWEN35_VLLM_MTP.engine.speculative is not None
        self.assertEqual(QWEN35_VLLM_MTP.engine.speculative.method, "qwen3_next_mtp")

    def test_resolves_one_parent_and_deep_merges_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "models"
            model_dir.mkdir()
            (model_dir / "foundation.yaml").write_text(
                """\
id: foundation
model:
  artifact: hf://org/model@revision
  form: base
  weights: {format: safetensors}
  capabilities: {context_window: 4096}
defaults:
  train:
    sft: train/sft/default
  serve:
    vllm: serve/vllm/default
""",
                encoding="utf-8",
            )
            (model_dir / "derived.yaml").write_text(
                """\
id: derived
extends: foundation
model:
  artifact: trackio://models/adapter@1
  form: adapter
  required_base: hf://org/model@revision
defaults:
  train:
    dpo: train/dpo/default
""",
                encoding="utf-8",
            )

            resolved = ProfileResolver(root).resolve("models", "derived")

            self.assertEqual(resolved.data["model"]["form"], "adapter")
            self.assertEqual(resolved.data["defaults"]["train"]["sft"], "train/sft/default")
            self.assertEqual(resolved.data["defaults"]["train"]["dpo"], "train/dpo/default")
            self.assertEqual([path.name for path in resolved.sources], ["foundation.yaml", "derived.yaml"])

    def test_rejects_inheritance_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_dir = root / "train"
            train_dir.mkdir()
            (train_dir / "a.yaml").write_text("id: a\nextends: b\n", encoding="utf-8")
            (train_dir / "b.yaml").write_text("id: b\nextends: a\n", encoding="utf-8")

            with self.assertRaisesRegex(ProfileError, "inheritance cycle"):
                ProfileResolver(root).resolve("train", "a")

    def test_profile_ids_may_contain_model_version_dots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "models"
            model_dir.mkdir()
            (model_dir / "family2.5-1.2b.yaml").write_text(
                "id: family2.5-1.2b\nmodel:\n  artifact: hf://org/model@abc\n  form: base\n  weights: {format: safetensors}\n  capabilities: {context_window: 4096}\n",
                encoding="utf-8",
            )

            resolved = ProfileResolver(root).resolve("models", "family2.5-1.2b")

            self.assertEqual(resolved.data["id"], "family2.5-1.2b")

    def test_adapter_requires_an_immutable_base_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "models"
            model_dir.mkdir()
            (model_dir / "adapter.yaml").write_text(
                "id: adapter\nmodel:\n  artifact: trackio://models/a@1\n  form: adapter\n  weights: {format: peft}\n  capabilities: {context_window: 4096}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ProfileError, "required_base"):
                ProfileResolver(root).resolve("models", "adapter")


if __name__ == "__main__":
    unittest.main()
