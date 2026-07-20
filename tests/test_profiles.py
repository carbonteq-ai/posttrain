from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common.profiles import ProfileError, ProfileResolver


class ProfileResolverTest(unittest.TestCase):
    def test_resolves_checked_in_foundation_and_serve_profiles(self) -> None:
        root = Path(__file__).resolve().parents[1] / "profiles"
        resolver = ProfileResolver(root)

        model = resolver.resolve("models", "lfm2.5-1.2b-thinking")
        serve = resolver.resolve("serve", model.data["defaults"]["serve"]["vllm"])

        self.assertEqual(model.data["model"]["family"], "lfm2.5")
        self.assertEqual(serve.data["backend"], "vllm")
        self.assertEqual(serve.data["engine"]["max_model_len"], 4096)
        self.assertEqual(model.data["model"]["capabilities"]["context_window"], 32768)

        eval_serve = resolver.resolve("serve", model.data["defaults"]["eval"]["serve"])
        self.assertEqual(eval_serve.data["engine"]["max_model_len"], 32768)
        self.assertEqual(eval_serve.data["engine"]["kv_cache_dtype"], "turboquant_k8v4")

    def test_qwen_exposes_standard_turboquant_and_mtp_serve_variants(self) -> None:
        root = Path(__file__).resolve().parents[1] / "profiles"
        resolver = ProfileResolver(root)
        model = resolver.resolve("models", "qwen3.5-2b")

        self.assertEqual(model.data["model"]["capabilities"]["context_window"], 262144)
        self.assertEqual(
            model.data["defaults"]["eval"]["serve"],
            "serve/vllm/qwen3.5-2b-turboquant-k8v4",
        )
        mtp = resolver.resolve("serve", model.data["defaults"]["serve"]["variants"]["mtp"])
        self.assertEqual(mtp.data["engine"]["speculative_config"]["method"], "qwen3_next_mtp")

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
