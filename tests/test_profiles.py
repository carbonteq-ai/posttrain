from __future__ import annotations

import unittest

from posttrain.common.profiles import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.serve.profiles import (
    LFM25_VLLM,
    LFM25_VLLM_TURBOQUANT_K8,
    QWEN35_VLLM_MTP,
    QWEN35_VLLM_TEXT,
    QWEN35_VLLM_TURBOQUANT_K8,
)


class TypedProfileTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
