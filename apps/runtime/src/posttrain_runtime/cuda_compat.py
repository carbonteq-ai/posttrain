"""Select image-owned CUDA forward compatibility before backend imports."""

from __future__ import annotations

import ctypes
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

DECLARATION_PATH = Path("/opt/posttrain/runtime/cuda-compat.json")
MODE_VARIABLE = "POSTTRAIN_CUDA_COMPAT_MODE"
_REEXEC_GUARD = "_POSTTRAIN_CUDA_COMPAT_REEXEC"
_VALID_MODES = frozenset({"auto", "off", "force"})
_DECLARATION_FIELDS = frozenset({"schema_version", "runtime_api_version", "compat_path", "payload_digest"})
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class CudaCompatibilityError(RuntimeError):
    """A safe, typed failure raised before user code or a CUDA backend starts."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class CudaCompatibilityDeclaration:
    runtime_api_version: int
    compat_path: Path
    payload_digest: str

    @classmethod
    def read(cls, path: Path = DECLARATION_PATH) -> CudaCompatibilityDeclaration | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CudaCompatibilityError(
                "cuda_compat_declaration_invalid",
                "the image CUDA compatibility declaration cannot be read",
            ) from error
        if not isinstance(payload, dict) or set(payload) != _DECLARATION_FIELDS:
            raise CudaCompatibilityError(
                "cuda_compat_declaration_invalid",
                "the image CUDA compatibility declaration has unexpected fields",
            )
        if payload["schema_version"] != 1:
            raise CudaCompatibilityError(
                "cuda_compat_declaration_invalid",
                "the image CUDA compatibility declaration schema is unsupported",
            )
        runtime_api_version = payload["runtime_api_version"]
        compat_path = payload["compat_path"]
        payload_digest = payload["payload_digest"]
        if (
            isinstance(runtime_api_version, bool)
            or not isinstance(runtime_api_version, int)
            or runtime_api_version <= 0
        ):
            raise CudaCompatibilityError(
                "cuda_compat_declaration_invalid",
                "runtime_api_version must be a positive integer",
            )
        if not isinstance(compat_path, str) or not compat_path.startswith("/"):
            raise CudaCompatibilityError(
                "cuda_compat_declaration_invalid",
                "compat_path must be an absolute path",
            )
        if not isinstance(payload_digest, str) or _DIGEST.fullmatch(payload_digest) is None:
            raise CudaCompatibilityError(
                "cuda_compat_declaration_invalid",
                "payload_digest must be an immutable sha256 digest",
            )
        return cls(runtime_api_version, Path(compat_path), payload_digest)


@dataclass(frozen=True, slots=True)
class CudaCompatibilitySelection:
    mode: str
    selected: str
    driver_api_version: int | None
    runtime_api_version: int | None
    payload_digest: str | None


def _probe_cuda_driver() -> int:
    try:
        driver = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_LOCAL)
    except OSError as error:
        raise CudaCompatibilityError(
            "cuda_driver_unavailable",
            "libcuda.so.1 is unavailable on the execution host",
        ) from error
    initialize = driver.cuInit
    initialize.argtypes = [ctypes.c_uint]
    initialize.restype = ctypes.c_int
    result = int(initialize(0))
    if result != 0:
        raise CudaCompatibilityError(
            "cuda_driver_initialization_failed",
            f"the CUDA driver failed to initialize with code {result}",
        )
    get_version = driver.cuDriverGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    result = int(get_version(ctypes.byref(version)))
    if result != 0 or version.value <= 0:
        raise CudaCompatibilityError(
            "cuda_driver_version_failed",
            f"the CUDA driver version query failed with code {result}",
        )
    return int(version.value)


def _mode(environ: Mapping[str, str]) -> str:
    mode = environ.get(MODE_VARIABLE, "auto").strip().lower()
    if mode not in _VALID_MODES:
        raise CudaCompatibilityError(
            "cuda_compat_mode_invalid",
            f"{MODE_VARIABLE} must be auto, off, or force",
        )
    return mode


def _compat_path_is_active(path: Path, environ: Mapping[str, str]) -> bool:
    entries = [entry for entry in environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry]
    return bool(entries) and Path(entries[0]) == path


def _reexec_environment(
    declaration: CudaCompatibilityDeclaration,
    environ: Mapping[str, str],
) -> dict[str, str]:
    if not declaration.compat_path.is_dir():
        raise CudaCompatibilityError(
            "cuda_compat_payload_missing",
            "the declared CUDA compatibility payload is absent from the image",
        )
    entries = [
        entry
        for entry in environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry and Path(entry) != declaration.compat_path
    ]
    updated = dict(environ)
    updated["LD_LIBRARY_PATH"] = os.pathsep.join((str(declaration.compat_path), *entries))
    updated[_REEXEC_GUARD] = "1"
    return updated


def activate_cuda_compatibility(
    *,
    declaration_path: Path = DECLARATION_PATH,
    environ: MutableMapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
    executable: str | None = None,
    probe_driver: Callable[[], int] = _probe_cuda_driver,
    exec_process: Callable[[str, list[str], Mapping[str, str]], NoReturn] = os.execvpe,
) -> CudaCompatibilitySelection:
    """Select the native driver or re-exec once with the image compat payload."""

    environment = os.environ if environ is None else environ
    mode = _mode(environment)
    if mode == "off":
        return CudaCompatibilitySelection(mode, "native", None, None, None)

    declaration = CudaCompatibilityDeclaration.read(declaration_path)
    if declaration is None:
        if mode == "force":
            raise CudaCompatibilityError(
                "cuda_compat_declaration_missing",
                "force mode requires an image CUDA compatibility declaration",
            )
        return CudaCompatibilitySelection(mode, "native", None, None, None)

    guarded = environment.get(_REEXEC_GUARD) == "1"
    if guarded:
        if not _compat_path_is_active(declaration.compat_path, environment):
            raise CudaCompatibilityError(
                "cuda_compat_reexec_guard_invalid",
                "the CUDA compatibility re-exec guard is set without the declared path",
            )
        driver_version = probe_driver()
        if driver_version < declaration.runtime_api_version:
            raise CudaCompatibilityError(
                "cuda_compat_initialization_failed",
                "the compatibility payload did not satisfy the image CUDA runtime",
            )
        return CudaCompatibilitySelection(
            mode,
            "compat",
            driver_version,
            declaration.runtime_api_version,
            declaration.payload_digest,
        )

    driver_version = probe_driver()
    if mode == "auto" and driver_version >= declaration.runtime_api_version:
        return CudaCompatibilitySelection(
            mode,
            "native",
            driver_version,
            declaration.runtime_api_version,
            declaration.payload_digest,
        )

    updated = _reexec_environment(declaration, environment)
    program = executable or sys.executable
    command = [program, *(sys.argv if argv is None else argv)]
    exec_process(program, command, updated)
    raise CudaCompatibilityError(
        "cuda_compat_reexec_failed",
        "the CUDA compatibility re-exec returned unexpectedly",
    )


__all__ = [
    "DECLARATION_PATH",
    "MODE_VARIABLE",
    "CudaCompatibilityDeclaration",
    "CudaCompatibilityError",
    "CudaCompatibilitySelection",
    "activate_cuda_compatibility",
]
