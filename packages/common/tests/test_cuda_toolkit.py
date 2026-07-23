"""Tests for the common CUDA toolkit view."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from posttrain.common.cuda import build_toolkit_view


class CudaToolkitViewTests(unittest.TestCase):
    def test_builds_standard_layout_and_linker_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            toolkit = root / "toolkit"
            for directory in ("bin", "include", "nvvm", "lib"):
                (toolkit / directory).mkdir(parents=True)
            runtime = toolkit / "lib" / "libcudart.so.13"
            runtime.touch()

            view = build_toolkit_view(
                toolkit,
                cuda_version="13.0",
                cache_root=root / "cache",
            )

            self.assertEqual((view / "bin").resolve(), (toolkit / "bin").resolve())
            self.assertEqual(
                (view / "lib64" / "libcudart.so").resolve(),
                runtime.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
