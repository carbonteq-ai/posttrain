"""Private compatibility seam for the pinned Verifiers environment runtime."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, cast

# Client libraries every Verifiers harness program imports. A tool server's own
# module is learned by the fork server on first use, so no environment needs to
# be named here.
FORK_SERVER_PRELOAD = (
    "anyio",
    "httpx",
    "httpx2",
    "mcp",
    "mcp.client.streamable_http",
    "openai",
    "openai.resources",
    "tenacity",
)


def enable_verifiers_fork_server() -> None:
    """Start Verifiers programs from a warm fork server unless the job opts out.

    Each rollout starts a tool server and a harness program; without the fork
    server every one re-imports its dependencies (several seconds per episode
    under concurrency). An explicit ``VF_FORK_SERVER`` value, including ``0``,
    is kept; the variables reach every worker process started afterward.
    """
    os.environ.setdefault("VF_FORK_SERVER", "1")
    os.environ.setdefault("VF_FORK_SERVER_PRELOAD", ",".join(FORK_SERVER_PRELOAD))


def verifiers_environment_types() -> tuple[type[Any], type[Any]]:
    """Return the native config and environment types across the v0.2/v0.3 rename."""

    try:
        environment_module = import_module("verifiers.v1.env")
        environment_symbols = vars(environment_module)
        config_type = environment_symbols.get("EnvConfig")
        if config_type is None:
            config_type = vars(import_module("verifiers.v1.configs.env"))["EnvConfig"]
        environment_type = environment_symbols.get("Env")
        if environment_type is None:
            environment_type = environment_symbols["Environment"]
    except (ImportError, KeyError) as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    return cast(type[Any], config_type), cast(type[Any], environment_type)


def materialize_verifiers_environment(value: object) -> Any:
    """Return a native environment from either an environment or its config."""

    config_type, environment_type = verifiers_environment_types()
    if isinstance(value, environment_type):
        return value
    if isinstance(value, config_type):
        try:
            from verifiers.v1.utils.loaders import (  # pyright: ignore[reportMissingImports]
                load_environment,
            )
        except ImportError:
            # Verifiers v0.2 exposed a concrete Environment class and did not
            # provide the centralized environment loader used by v0.3.
            return environment_type(value)
        return load_environment(cast(Any, value))
    raise TypeError("environment activation did not produce a Verifiers EnvConfig")


__all__ = [
    "FORK_SERVER_PRELOAD",
    "enable_verifiers_fork_server",
    "materialize_verifiers_environment",
    "verifiers_environment_types",
]
