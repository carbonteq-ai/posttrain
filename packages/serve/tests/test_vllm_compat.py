"""Tests for pinned vLLM compatibility patches."""

from __future__ import annotations

import unittest


class VllmCompatibilityTest(unittest.TestCase):
    def test_fork_preserves_turboquant_dtype_without_runtime_patch(self) -> None:
        try:
            import torch
            from posttrain.serve.vllm_compat import apply_vllm_compatibility_patches
            from vllm.v1.kv_cache_interface import (  # pyright: ignore[reportMissingImports]
                KVQuantMode,
                TQFullAttentionSpec,
            )
            from vllm.v1.worker.gpu_model_runner import (  # pyright: ignore[reportMissingImports]
                _get_layer_cache_dtype_str,
            )
        except ImportError:
            self.skipTest("serve[vllm] optional dependencies are not installed")

        applied = apply_vllm_compatibility_patches()
        spec = TQFullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            head_size_v=128,
            dtype=torch.uint8,
            tq_slot_size=196,
        )

        self.assertNotIn("turboquant-quant-marker", applied)
        self.assertEqual(spec.kv_quant_mode, KVQuantMode.NONE)
        self.assertEqual(
            _get_layer_cache_dtype_str(spec, "turboquant_k8v4"),
            "turboquant_k8v4",
        )
        self.assertEqual(spec.real_page_size_bytes, 16 * 2 * 196)


if __name__ == "__main__":
    unittest.main()
