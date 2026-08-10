"""Tests for the pinned TRL and vLLM compatibility contract."""

from __future__ import annotations

import importlib.metadata
import unittest
import warnings

TRL_RELEASE = "1.9.2.post1"


class TrlVllmCompatibilityTest(unittest.TestCase):
    def test_trl_is_installed_from_the_pinned_release(self) -> None:
        try:
            distribution = importlib.metadata.distribution("trl")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("TRL optional dependency is not installed")

        self.assertEqual(distribution.version, TRL_RELEASE)

    def test_olmo3_recipe_is_available_from_the_pinned_release(self) -> None:
        try:
            from trl.trainer.olmo3_grpo_config import Olmo3GRPOConfig
        except ImportError:
            self.skipTest("TRL optional dependency is not installed")

        config = Olmo3GRPOConfig(
            output_dir="/tmp/posttrain-trl-contract",
            report_to="none",
            use_cpu=True,
            bf16=False,
        )
        self.assertTrue(config.active_sampling)
        self.assertEqual(config.loss_type, "dapo")
        self.assertEqual(config.scale_rewards, "none")
        self.assertEqual(config.epsilon_high, 0.272)
        self.assertEqual(config.vllm_importance_sampling_mode, "token_truncate")
        self.assertEqual(config.vllm_importance_sampling_clip_max, 2.0)

    def test_trl_accepts_the_pinned_vllm_release(self) -> None:
        try:
            import vllm  # pyright: ignore[reportMissingImports]
            from trl.import_utils import is_vllm_available
        except ImportError:
            self.skipTest("vLLM optional dependency is not installed")

        self.assertEqual(vllm.__version__, "0.25.1")
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            self.assertTrue(is_vllm_available())

        compatibility_warnings = [
            warning for warning in caught_warnings if "TRL currently supports vLLM" in str(warning.message)
        ]
        self.assertEqual(compatibility_warnings, [])


if __name__ == "__main__":
    unittest.main()
