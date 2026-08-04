"""Runtime bootstrap policy for prebuilt Verifiers evaluation images.

The upstream Verifiers runtime creates a separate PEP 723 environment for
each harness script. That is useful for arbitrary local scripts, but it splits
a packed evaluation into two independently resolved dependency graphs: the
job/tool-server graph and the harness/MCP-client graph. A digest-pinned job
image instead executes every Verifiers process from its one hash-locked image
environment. Runtime preparation only materializes the script bytes.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shlex
import uuid
from typing import Any

PREINSTALLED_ENV = "POSTTRAIN_VERIFIERS_PREINSTALLED"
_PATCHED = "_posttrain_preinstalled_runtime"
_JOB_PYTHON = "/opt/posttrain/venv/bin/python"


async def prepare_preinstalled_uv_script(
    runtime: Any,
    script: str | bytes,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Materialize a harness script and execute it from the packed job lock."""

    data = script.encode() if isinstance(script, str) else script
    digest = hashlib.sha256(data).hexdigest()
    path = f"/tmp/vf-scripts/{digest}.py"
    interpreters = runtime._uv_interpreters
    locks = runtime._uv_script_locks
    if digest not in interpreters:
        async with locks.setdefault(digest, asyncio.Lock()):
            if digest not in interpreters:
                tmp = f"{path}.{uuid.uuid4().hex}.tmp"
                await runtime.write(tmp, data)
                command = (
                    f"mv -f {shlex.quote(tmp)} {shlex.quote(path)} "
                    f"&& test -x {shlex.quote(_JOB_PYTHON)} "
                    f"&& {_JOB_PYTHON} -c "
                    f"{shlex.quote('import httpx, mcp, openai, tenacity')}"
                )
                result = await runtime.run(["sh", "-c", command], env or {})
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"packed Verifiers harness dependencies are unavailable: {result.stderr.strip()[-2000:]}"
                    )
                interpreters[digest] = _JOB_PYTHON
    return [interpreters[digest], path]


def configure_preinstalled_runtime() -> None:
    """Install the no-network policy into Verifiers when the image requests it."""

    if os.environ.get(PREINSTALLED_ENV) != "1":
        return
    from verifiers.v1.runtimes import base as runtime_base

    if getattr(runtime_base.Runtime, _PATCHED, False):
        return

    runtime_base.Runtime.prepare_uv_script = prepare_preinstalled_uv_script  # type: ignore[method-assign]
    setattr(runtime_base.Runtime, _PATCHED, True)


__all__ = [
    "PREINSTALLED_ENV",
    "configure_preinstalled_runtime",
    "prepare_preinstalled_uv_script",
]
