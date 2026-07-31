"""Tests for pinned vLLM compatibility patches."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from posttrain.serve.vllm_compat import apply_vllm_compatibility_patches


class VllmCompatibilityTest(unittest.TestCase):
    def test_upstream_turboquant_marker_patch_is_guarded_and_idempotent(self) -> None:
        class QuantMode:
            NONE = "none"
            FP8_PER_TENSOR = "fp8"

        class TQSpec:
            def __init__(self) -> None:
                self.__post_init__()

            def __post_init__(self) -> None:
                self.kv_quant_mode = QuantMode.NONE

        module = types.ModuleType("vllm.v1.kv_cache_interface")
        module.__dict__.update(
            KVQuantMode=QuantMode,
            TQFullAttentionSpec=TQSpec,
            get_kv_quant_mode=lambda _dtype: QuantMode.NONE,
        )
        modules = {
            "torch": None,
            "vllm": types.ModuleType("vllm"),
            "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.kv_cache_interface": module,
        }

        with patch.dict(sys.modules, modules):
            first = apply_vllm_compatibility_patches()
            second = apply_vllm_compatibility_patches()

        self.assertEqual(first, ("turboquant-quant-marker",))
        self.assertEqual(second, ("turboquant-quant-marker",))
        self.assertEqual(TQSpec().kv_quant_mode, QuantMode.FP8_PER_TENSOR)

    def test_fork_preserves_turboquant_dtype_without_runtime_patch(self) -> None:
        try:
            import torch
            from vllm.v1.kv_cache_interface import (  # pyright: ignore[reportMissingImports]
                KVQuantMode,
                TQFullAttentionSpec,
                get_kv_quant_mode,
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

        if get_kv_quant_mode("turboquant_k8v4") == KVQuantMode.NONE:
            self.assertIn("turboquant-quant-marker", applied)
        else:
            self.assertNotIn("turboquant-quant-marker", applied)
        self.assertEqual(spec.kv_quant_mode, KVQuantMode.NONE)
        self.assertEqual(
            _get_layer_cache_dtype_str(spec, "turboquant_k8v4"),
            "turboquant_k8v4",
        )
        self.assertEqual(spec.real_page_size_bytes, 16 * 2 * 196)


if __name__ == "__main__":
    unittest.main()
