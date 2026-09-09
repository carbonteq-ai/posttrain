"""Private compatibility seam for the pinned Verifiers environment runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast


def verifiers_environment_types() -> tuple[type[Any], type[Any]]:
    """Return the native config and environment types across the v0.2/v0.3 rename."""

    try:
        environment_module = import_module("verifiers.v1.env")
        config_type = getattr(environment_module, "EnvConfig", None)
        if config_type is None:
            config_type = getattr(import_module("verifiers.v1.configs.env"), "EnvConfig")
        environment_type = getattr(environment_module, "Env", None)
        if environment_type is None:
            environment_type = getattr(environment_module, "Environment")
    except (AttributeError, ImportError) as error:
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


__all__ = ["materialize_verifiers_environment", "verifiers_environment_types"]
