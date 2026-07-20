from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common import BENCHMARKS_DIR, PROFILES_DIR, ProfileResolver
from serve.suite_cli import _serve_profile_for_variant, select_cases
from serve.suites import SuiteError, load_suite


class InferenceSuiteTest(unittest.TestCase):
    def test_core_suite_expands_context_and_concurrency_matrix(self) -> None:
        suite = load_suite(BENCHMARKS_DIR / "inference" / "suites" / "core.yaml")

        self.assertEqual(suite.contexts, (1024, 2048, 4096, 8192, 16384, 32768))
        self.assertEqual(suite.concurrencies, (1, 2, 4, 8))
        self.assertEqual(len(suite.cases()), 96)
        long_context = [case for case in suite.cases() if case.context_window == 32768]
        self.assertEqual(len(long_context), 16)
        self.assertTrue(all(case.serve_variant == "turboquant_k8v4" for case in long_context))
        self.assertTrue(all(case.serve_variant is None for case in suite.cases() if case.context_window < 32768))

    def test_filters_matrix_without_changing_case_identity(self) -> None:
        suite = load_suite(BENCHMARKS_DIR / "inference" / "suites" / "core.yaml")

        selected = select_cases(
            suite.cases(),
            contexts=[4096],
            concurrencies=[8],
        )

        self.assertEqual(len(selected), 4)
        self.assertTrue(all(case.id.endswith("-ctx4096-c8") for case in selected))

    def test_model_profiles_resolve_turboquant_variant(self) -> None:
        resolver = ProfileResolver(PROFILES_DIR)
        for reference in ("lfm2.5-1.2b-thinking", "qwen3.5-2b"):
            model = resolver.resolve("models", reference)
            serve_reference = _serve_profile_for_variant(model.data, "turboquant_k8v4")
            self.assertIsNotNone(serve_reference)
            assert serve_reference is not None
            serve = resolver.resolve("serve", serve_reference)
            self.assertEqual(serve.data["engine"]["kv_cache_dtype"], "turboquant_k8v4")

    def test_rejects_shape_with_ambiguous_input_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                """schema_version: 1
id: bad
contexts: [1024]
concurrencies: [1]
shapes:
  - id: ambiguous
    input_tokens: 128
    input_fraction: 0.5
    output_tokens: 64
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SuiteError, "exactly one"):
                load_suite(path)


if __name__ == "__main__":
    unittest.main()
