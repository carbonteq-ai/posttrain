"""Tests for the pinned TRL and vLLM compatibility contract."""

from __future__ import annotations

import importlib.metadata
import json
import unittest
import warnings

TRL_FORK_COMMIT = "b43a0a3d622ab1547f4d2abbd1b25eab3c52a0b9"


class TrlVllmCompatibilityTest(unittest.TestCase):
    def test_trl_is_installed_from_the_pinned_fork_commit(self) -> None:
        try:
            distribution = importlib.metadata.distribution("trl")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("TRL optional dependency is not installed")
        direct_url = distribution.read_text("direct_url.json")

        self.assertIsNotNone(direct_url)
        assert direct_url is not None
        source = json.loads(direct_url)
        self.assertEqual(source["url"], "https://github.com/carbonteq-ai/trl.git")
        self.assertEqual(source["vcs_info"]["commit_id"], TRL_FORK_COMMIT)
        self.assertEqual(distribution.version, "1.8.0")

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
