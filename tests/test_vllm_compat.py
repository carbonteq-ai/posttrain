from __future__ import annotations

import unittest


class VllmCompatibilityTest(unittest.TestCase):
    def test_turboquant_spec_keeps_packed_size_and_quantized_marker(self) -> None:
        try:
            import torch
            from serve.vllm_compat import apply_vllm_compatibility_patches
            from vllm.v1.kv_cache_interface import KVQuantMode, TQFullAttentionSpec
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

        self.assertIn("turboquant-quant-marker", applied)
        self.assertNotEqual(spec.kv_quant_mode, KVQuantMode.NONE)
        self.assertFalse(spec.kv_quant_mode.is_per_token_head)
        self.assertEqual(spec.real_page_size_bytes, 16 * 2 * 196)


if __name__ == "__main__":
    unittest.main()
