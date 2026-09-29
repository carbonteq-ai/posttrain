"""Runtime bootstrap policy for prebuilt Verifiers evaluation images.

The upstream Verifiers runtime creates a separate PEP 723 environment for
each harness script. That is useful for arbitrary local scripts, but it splits
a packed evaluation into two independently resolved dependency graphs: the
job/tool-server graph and the harness/MCP-client graph. A digest-pinned job
image instead executes every Verifiers process from its one hash-locked image
environment. Runtime preparation only materializes the script bytes. The
policy is shared with training (``posttrain.environment.verifiers_preinstalled``).
"""

from __future__ import annotations

from posttrain.environment.verifiers_preinstalled import (
    PREINSTALLED_ENV,
    preinstalled_uv_script_preparer,
)
from posttrain.environment.verifiers_preinstalled import (
    configure_preinstalled_runtime as _configure_preinstalled_runtime,
)

_JOB_PYTHON = "/opt/posttrain/venv/bin/python"

prepare_preinstalled_uv_script = preinstalled_uv_script_preparer(_JOB_PYTHON)
"""Materialize a harness script and execute it from the packed job lock."""


def configure_preinstalled_runtime() -> None:
    """Install the no-network policy into Verifiers when the image requests it."""

    from posttrain.environment.verifiers_runtime import enable_verifiers_fork_server

    enable_verifiers_fork_server()
    _configure_preinstalled_runtime(interpreter=_JOB_PYTHON)


__all__ = [
    "PREINSTALLED_ENV",
    "configure_preinstalled_runtime",
    "prepare_preinstalled_uv_script",
]
