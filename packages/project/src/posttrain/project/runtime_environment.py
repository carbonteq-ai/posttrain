"""Authoritative project runtime values loaded from an explicit env file."""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from posttrain.common import ContractError

_PROJECT_ENVIRONMENT_NAME = "posttrain.env"


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    """A redacted runtime value source selected independently of shell exports."""

    path: Path | None
    _values: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.path is not None and not self.path.is_absolute():
            raise ContractError("runtime environment path must be absolute")
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    def for_execution(self) -> dict[str, str]:
        """Return values only at an execution boundary that already knows their names."""

        return dict(self._values)

    def __repr__(self) -> str:
        source = str(self.path) if self.path is not None else "none"
        return f"RuntimeEnvironment(path={source!r}, names={tuple(sorted(self._values))!r})"


def resolve_runtime_environment(project_root: Path, *, env_file: Path | None = None) -> RuntimeEnvironment:
    """Select an explicit env file or ``<project>/posttrain.env``, never ``os.environ``."""

    root = project_root.resolve()
    selected = env_file.resolve() if env_file is not None else root / _PROJECT_ENVIRONMENT_NAME
    if not selected.exists():
        if env_file is not None:
            raise ContractError(f"explicit runtime environment file is missing: {selected}")
        return RuntimeEnvironment(None, {})
    _require_protected_file(selected)
    return RuntimeEnvironment(selected, _parse_environment(selected))


def _parse_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, raw_value = line.partition("=")
        if not separator or not name.strip():
            raise ContractError(f"invalid runtime environment file: {path}")
        parsed = shlex.split(raw_value, comments=False, posix=True)
        if len(parsed) != 1:
            raise ContractError(f"invalid runtime environment value for {name.strip()}")
        values[name.strip()] = parsed[0]
    return values


def _require_protected_file(path: Path) -> None:
    if not path.is_file():
        raise ContractError(f"runtime environment is not a file: {path}")
    if path.stat().st_mode & 0o077:
        raise ContractError(f"runtime environment must not be accessible by group or others: {path}")


__all__ = ["RuntimeEnvironment", "resolve_runtime_environment"]
