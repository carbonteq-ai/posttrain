"""CUDA toolkit discovery for JIT-compiled inference kernels.

NVIDIA's pip CUDA packages install a complete toolkit under ``nvidia/cu*``,
but use ``lib`` and versioned shared-library names. Some CUDA build systems,
including FlashInfer's JIT path, expect the conventional toolkit layout with
``lib64`` and development linker names such as ``libcudart.so``. This module
creates a small cache-local view of the pinned toolkit instead of modifying the
installed wheels or disabling optimized kernels.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Protocol


class TorchVersion(Protocol):
    cuda: str | None


class TorchModule(Protocol):
    version: TorchVersion
    __file__: str


def _nvcc_release(nvcc: Path) -> str | None:
    output = subprocess.run(
        [str(nvcc), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"release (\d+\.\d+)", output)
    return match.group(1) if match else None


def _replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        if not link.is_symlink():
            raise RuntimeError(f"refusing to replace non-symlink CUDA view entry: {link}")
        link.unlink()
    link.symlink_to(target, target_is_directory=target.is_dir())


def _linker_name(library: Path) -> str | None:
    """Return the conventional unversioned linker name for a shared object."""

    match = re.match(r"(.+\.so)\.\d", library.name)
    return match.group(1) if match else None


def build_toolkit_view(toolkit: Path, *, cuda_version: str, cache_root: Path) -> Path:
    """Create the standard CUDA layout expected by third-party JIT builders."""

    view = cache_root / f"cuda-{cuda_version}"
    view.mkdir(parents=True, exist_ok=True)
    for directory in ("bin", "include", "nvvm"):
        source = toolkit / directory
        if not source.is_dir():
            raise RuntimeError(f"CUDA toolkit is missing {source}")
        _replace_symlink(view / directory, source)

    source_lib = toolkit / "lib"
    if not source_lib.is_dir():
        raise RuntimeError(f"CUDA toolkit is missing {source_lib}")
    lib64 = view / "lib64"
    lib64.mkdir(exist_ok=True)
    for source in source_lib.iterdir():
        _replace_symlink(lib64 / source.name, source)
        linker_name = _linker_name(source)
        if linker_name is not None:
            _replace_symlink(lib64 / linker_name, source)
    return view


def resolve_cuda_home(torch_module: TorchModule, *, cache_root: Path | None = None) -> Path:
    """Find a pip CUDA toolkit matching PyTorch and return its standard view."""

    expected_cuda = torch_module.version.cuda
    if not expected_cuda:
        raise RuntimeError("the installed PyTorch build does not include CUDA")
    torch_file = Path(torch_module.__file__).resolve()
    site_packages = torch_file.parents[1]
    candidates = sorted(site_packages.glob("nvidia/cu*/bin/nvcc"))
    matching_root: Path | None = None
    for nvcc in candidates:
        root = nvcc.parent.parent
        if (
            _nvcc_release(nvcc) == expected_cuda
            and (root / "include" / "curand.h").is_file()
            and any((root / "lib").glob("libcudart.so.*"))
        ):
            matching_root = root
            break
    if matching_root is None:
        raise RuntimeError(
            f"no complete CUDA {expected_cuda} toolkit matching PyTorch was found; "
            "sync the serve[vllm] environment from the workspace lock"
        )

    if cache_root is None:
        cache_base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_root = cache_base / "post-training-lab" / "cuda"
    return build_toolkit_view(
        matching_root,
        cuda_version=expected_cuda,
        cache_root=cache_root,
    )
