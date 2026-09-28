"""Run Verifiers harness scripts from a packed job's own locked environment.

Upstream Verifiers gives each harness script a separate PEP 723 environment:
at the first rollout it installs ``uv`` with pip and runs ``uv sync --script``
for the script's dependencies (``openai``, ``mcp``, ``httpx``, ``httpx2``,
``tenacity``), downloading them from PyPI. A digest-pinned job image already
contains those packages in the hash-locked interpreter that runs Verifiers, so
it must neither reach the network nor resolve a second dependency graph during
a rollout. With ``POSTTRAIN_VERIFIERS_PREINSTALLED=1`` (set by
``posttrain-runtime`` for every packed job) the script is only written to disk
and executed with that interpreter; a missing package fails immediately.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shlex
import sys
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

PREINSTALLED_ENV = "POSTTRAIN_VERIFIERS_PREINSTALLED"
HARNESS_IMPORT_CHECK = "import httpx, httpx2, mcp, openai, tenacity"
_PATCHED = "_posttrain_preinstalled_runtime"

type ScriptPreparer = Callable[..., Coroutine[Any, Any, list[str]]]


def _script_lock(locks: Any, digest: str) -> asyncio.Lock:
    """The Runtime's lock for one script: ``LoopLocks.get`` in Verifiers cdd2ec76, a dict of locks before."""

    if isinstance(locks, dict):
        return locks.setdefault(digest, asyncio.Lock())
    return locks.get(digest)


def preinstalled_uv_script_preparer(interpreter: str) -> ScriptPreparer:
    """A ``Runtime.prepare_uv_script`` replacement that runs scripts with ``interpreter``."""

    async def prepare_uv_script(
        runtime: Any,
        script: str | bytes,
        env: dict[str, str] | None = None,
        *,
        activate: bool = True,
    ) -> list[str]:
        # The packed interpreter already owns the job's environment, so Verifiers'
        # optional activation wrapper is unnecessary; both modes get the same argv.
        del activate
        data = script.encode() if isinstance(script, str) else script
        digest = hashlib.sha256(data).hexdigest()
        path = f"/tmp/vf-scripts/{digest}.py"
        interpreters = runtime._uv_interpreters
        if digest not in interpreters:
            async with _script_lock(runtime._uv_script_locks, digest):
                if digest not in interpreters:
                    temporary = f"{path}.{uuid.uuid4().hex}.tmp"
                    await runtime.write(temporary, data)
                    command = (
                        f"mv -f {shlex.quote(temporary)} {shlex.quote(path)} "
                        f"&& test -x {shlex.quote(interpreter)} "
                        f"&& {shlex.quote(interpreter)} -c {shlex.quote(HARNESS_IMPORT_CHECK)}"
                    )
                    result = await runtime.run(["sh", "-c", command], env or {})
                    if result.exit_code != 0:
                        raise RuntimeError(
                            f"packed Verifiers harness dependencies are unavailable: {result.stderr.strip()[-2000:]}"
                        )
                    interpreters[digest] = interpreter
        return [interpreters[digest], path]

    return prepare_uv_script


def configure_preinstalled_runtime(*, interpreter: str | None = None) -> bool:
    """Install the no-network harness policy when the job requests it; return whether it is active.

    ``interpreter`` defaults to the one running Verifiers in this process (the
    TRL control environment, or the veRL backend environment in a Ray worker).
    """

    if os.environ.get(PREINSTALLED_ENV) != "1":
        return False
    from verifiers.v1.runtimes import base as runtime_base  # pyright: ignore[reportMissingImports]

    if getattr(runtime_base.Runtime, _PATCHED, None) is None:
        runtime_base.Runtime.prepare_uv_script = preinstalled_uv_script_preparer(  # pyright: ignore[reportAttributeAccessIssue]
            interpreter or sys.executable
        )
        setattr(runtime_base.Runtime, _PATCHED, interpreter or sys.executable)
    return True


__all__ = [
    "HARNESS_IMPORT_CHECK",
    "PREINSTALLED_ENV",
    "configure_preinstalled_runtime",
    "preinstalled_uv_script_preparer",
]
