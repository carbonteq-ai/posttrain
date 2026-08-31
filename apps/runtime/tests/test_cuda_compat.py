from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn

import pytest
from posttrain_runtime.cuda_compat import (
    CudaCompatibilityError,
    activate_cuda_compatibility,
)

_PAYLOAD_DIGEST = "sha256:" + "6" * 64


class _ReexecRequested(BaseException):
    def __init__(self, program: str, arguments: Sequence[str], environment: Mapping[str, str]) -> None:
        self.program = program
        self.arguments = tuple(arguments)
        self.environment = dict(environment)


def _declaration(root: Path, *, runtime_api_version: int = 13000) -> tuple[Path, Path]:
    compat = root / "compat"
    compat.mkdir()
    path = root / "cuda-compat.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_api_version": runtime_api_version,
                "compat_path": str(compat),
                "payload_digest": _PAYLOAD_DIGEST,
            }
        ),
        encoding="utf-8",
    )
    return path, compat


def _capture_exec(
    program: str,
    arguments: Sequence[str],
    environment: Mapping[str, str],
) -> NoReturn:
    raise _ReexecRequested(program, arguments, environment)


def test_auto_without_a_declaration_keeps_native_and_does_not_probe(tmp_path: Path) -> None:
    result = activate_cuda_compatibility(
        declaration_path=tmp_path / "missing.json",
        environ={},
        probe_driver=lambda: pytest.fail("driver probe must not run"),
    )

    assert result.selected == "native"
    assert result.driver_api_version is None


def test_cli_import_stays_cuda_neutral_until_command_dispatch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import sys; import posttrain_runtime.cli; assert 'posttrain_runtime.execute' not in sys.modules"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_auto_keeps_a_newer_native_driver(tmp_path: Path) -> None:
    declaration, _compat = _declaration(tmp_path)

    result = activate_cuda_compatibility(
        declaration_path=declaration,
        environ={},
        probe_driver=lambda: 13010,
    )

    assert result.selected == "native"
    assert result.driver_api_version == 13010


def test_auto_reexecs_once_for_an_older_native_driver(tmp_path: Path) -> None:
    declaration, compat = _declaration(tmp_path)

    with pytest.raises(_ReexecRequested) as captured:
        activate_cuda_compatibility(
            declaration_path=declaration,
            environ={"LD_LIBRARY_PATH": "/existing:/another"},
            argv=("posttrain-runtime", "execute", "--manifest", "/job/manifest.json"),
            executable="/usr/bin/python3",
            probe_driver=lambda: 12080,
            exec_process=_capture_exec,
        )

    assert captured.value.program == "/usr/bin/python3"
    assert captured.value.arguments[:2] == ("/usr/bin/python3", "posttrain-runtime")
    assert captured.value.environment["LD_LIBRARY_PATH"] == f"{compat}:/existing:/another"
    assert captured.value.environment["_POSTTRAIN_CUDA_COMPAT_REEXEC"] == "1"


def test_off_never_reads_a_malformed_declaration_or_probes(tmp_path: Path) -> None:
    declaration = tmp_path / "cuda-compat.json"
    declaration.write_text("not-json", encoding="utf-8")

    result = activate_cuda_compatibility(
        declaration_path=declaration,
        environ={"POSTTRAIN_CUDA_COMPAT_MODE": "off"},
        probe_driver=lambda: pytest.fail("driver probe must not run"),
    )

    assert result.selected == "native"


def test_force_requires_a_declaration(tmp_path: Path) -> None:
    with pytest.raises(CudaCompatibilityError) as captured:
        activate_cuda_compatibility(
            declaration_path=tmp_path / "missing.json",
            environ={"POSTTRAIN_CUDA_COMPAT_MODE": "force"},
        )

    assert captured.value.code == "cuda_compat_declaration_missing"


def test_force_reexecs_even_with_a_new_native_driver(tmp_path: Path) -> None:
    declaration, _compat = _declaration(tmp_path)

    with pytest.raises(_ReexecRequested):
        activate_cuda_compatibility(
            declaration_path=declaration,
            environ={"POSTTRAIN_CUDA_COMPAT_MODE": "force"},
            argv=("posttrain-runtime", "execute"),
            probe_driver=lambda: 14000,
            exec_process=_capture_exec,
        )


def test_reexec_guard_accepts_a_qualified_compat_driver(tmp_path: Path) -> None:
    declaration, compat = _declaration(tmp_path)

    result = activate_cuda_compatibility(
        declaration_path=declaration,
        environ={
            "_POSTTRAIN_CUDA_COMPAT_REEXEC": "1",
            "LD_LIBRARY_PATH": str(compat),
        },
        probe_driver=lambda: 13000,
    )

    assert result.selected == "compat"
    assert result.driver_api_version == 13000


def test_reexec_guard_rejects_a_loop_or_failed_compat_initialization(tmp_path: Path) -> None:
    declaration, compat = _declaration(tmp_path)

    with pytest.raises(CudaCompatibilityError) as captured:
        activate_cuda_compatibility(
            declaration_path=declaration,
            environ={
                "_POSTTRAIN_CUDA_COMPAT_REEXEC": "1",
                "LD_LIBRARY_PATH": str(compat),
            },
            probe_driver=lambda: 12080,
        )

    assert captured.value.code == "cuda_compat_initialization_failed"


def test_malformed_declaration_fails_with_a_safe_typed_error(tmp_path: Path) -> None:
    declaration = tmp_path / "cuda-compat.json"
    declaration.write_text('{"secret":"must-not-be-repeated"}', encoding="utf-8")

    with pytest.raises(CudaCompatibilityError) as captured:
        activate_cuda_compatibility(declaration_path=declaration, environ={})

    assert captured.value.code == "cuda_compat_declaration_invalid"
    assert "secret" not in str(captured.value)


def test_missing_compat_payload_fails_before_reexec(tmp_path: Path) -> None:
    declaration, compat = _declaration(tmp_path)
    compat.rmdir()

    with pytest.raises(CudaCompatibilityError) as captured:
        activate_cuda_compatibility(
            declaration_path=declaration,
            environ={},
            probe_driver=lambda: 12080,
            exec_process=_capture_exec,
        )

    assert captured.value.code == "cuda_compat_payload_missing"
