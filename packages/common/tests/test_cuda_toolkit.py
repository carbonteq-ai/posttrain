"""Tests for the common CUDA toolkit view."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from posttrain.common.cuda import TorchModule, build_toolkit_view, cuda_environment


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

    def test_cuda_environment_prepends_toolkit_and_interpreter_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            toolkit = root / "nvidia" / "cu13"
            for directory in ("bin", "include", "nvvm", "lib"):
                (toolkit / directory).mkdir(parents=True)
            nvcc = toolkit / "bin" / "nvcc"
            nvcc.write_text(
                "#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.0'\n",
                encoding="utf-8",
            )
            nvcc.chmod(0o755)
            (toolkit / "include" / "curand.h").write_text("", encoding="utf-8")
            (toolkit / "lib" / "libcudart.so.13").touch()

            site = root
            (site / "torch").mkdir()
            (site / "torch" / "__init__.py").write_text("", encoding="utf-8")
            torch_module = cast(
                TorchModule,
                SimpleNamespace(
                    version=SimpleNamespace(cuda="13.0"),
                    __file__=str(site / "torch" / "__init__.py"),
                ),
            )

            env = cuda_environment(
                torch_module,
                environ={
                    "PATH": "/usr/bin",
                    "XDG_CACHE_HOME": str(root / "xdg-cache"),
                },
            )
            scripts = str(Path(sys.executable).resolve().parent)
            path_entries = env["PATH"].split(":")
            self.assertEqual(path_entries[0], env["CUDA_HOME"] + "/bin")
            self.assertEqual(path_entries[1], scripts)
            self.assertIn("/usr/bin", path_entries)
            self.assertTrue(Path(env["CUDA_HOME"]).joinpath("lib64").is_dir())


if __name__ == "__main__":
    unittest.main()
