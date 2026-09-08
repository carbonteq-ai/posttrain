"""Private compatibility seam for the pinned Verifiers environment runtime."""

from __future__ import annotations

from typing import Any


def verifiers_environment_types() -> tuple[type[Any], type[Any]]:
    """Return the native config and environment types across the v0.2/v0.3 rename."""

    try:
        from verifiers.v1.env import EnvConfig  # pyright: ignore[reportMissingImports]

        try:
            from verifiers.v1.env import Environment  # pyright: ignore[reportMissingImports]
        except ImportError:
            from verifiers.v1.env import Env as Environment  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    return EnvConfig, Environment


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
        return load_environment(value)
    raise TypeError("environment activation did not produce a Verifiers EnvConfig")


__all__ = ["materialize_verifiers_environment", "verifiers_environment_types"]
