from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from posttrain.eval.backends.verifiers.runtime import prepare_preinstalled_uv_script


def test_standalone_eval_runtime_image_uses_private_prebuilt_parent() -> None:
    dockerfile = Path(__file__).parents[1].joinpath("runtime/Dockerfile").read_text()
    assert "ghcr.io/" not in dockerfile
    assert "POSTTRAIN_KIND_IMAGE" in dockerfile
    assert "pip install" not in dockerfile
    assert 'POSTTRAIN_VERIFIERS_PREINSTALLED="1"' in dockerfile


@dataclass
class _Result:
    exit_code: int
    stdout: str
    stderr: str = ""


@dataclass
class _Runtime:
    _uv_interpreters: dict[str, str] = field(default_factory=dict)
    _uv_script_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    writes: list[tuple[str, bytes]] = field(default_factory=list)
    commands: list[tuple[list[str], dict[str, str]]] = field(default_factory=list)

    async def write(self, path: str, data: bytes) -> None:
        self.writes.append((path, data))

    async def run(self, argv: list[str], env: dict[str, str]) -> _Result:
        self.commands.append((argv, env))
        return _Result(0, "/root/.cache/uv/environments-v2/abc/bin/python\n")


def test_preinstalled_bootstrap_uses_the_packed_job_interpreter_without_uv() -> None:
    runtime = _Runtime()

    argv = asyncio.run(prepare_preinstalled_uv_script(runtime, "print('ok')"))

    assert argv[0] == "/opt/posttrain/venv/bin/python"
    assert argv[1].startswith("/tmp/vf-scripts/")
    command = runtime.commands[0][0][2]
    assert "/opt/posttrain/venv/bin/python" in command
    assert "import httpx, mcp, openai, tenacity" in command
    assert "uv python" not in command
    assert "uv sync" not in command
    assert runtime.commands[0][1] == {}


def test_preinstalled_bootstrap_fails_closed_when_image_was_not_warmed() -> None:
    runtime = _Runtime()
    runtime.commands.clear()

    async def run_without_warmup(argv: list[str], env: dict[str, str]) -> _Result:
        runtime.commands.append((argv, env))
        return _Result(1, "", "missing mcp")

    runtime.run = run_without_warmup  # type: ignore[method-assign]

    try:
        asyncio.run(prepare_preinstalled_uv_script(runtime, "print('missing')"))
    except RuntimeError as error:
        assert "packed Verifiers harness dependencies are unavailable" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing prewarmed environment did not fail closed")
