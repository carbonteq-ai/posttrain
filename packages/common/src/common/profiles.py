"""Resolution for model profiles and engine-owned base configurations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

ProfileKind = Literal["models", "train", "eval", "serve"]


class ProfileError(ValueError):
    """Raised when a profile reference or inheritance chain is invalid."""


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    """Resolved profile data plus the source files that contributed to it."""

    kind: ProfileKind
    reference: str
    data: dict[str, Any]
    sources: tuple[Path, ...]


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as error:
        raise ProfileError(f"profile does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise ProfileError(f"invalid YAML profile {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProfileError(f"profile must be a YAML mapping: {path}")
    return value


def _deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; child scalars and lists replace parent values."""

    merged = deepcopy(parent)
    for key, value in child.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class ProfileResolver:
    """Resolve one-parent profile inheritance from the checked-in profile tree."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def path_for(self, kind: ProfileKind, reference: str | Path) -> Path:
        value = Path(reference)
        if value.is_absolute():
            path = value
        else:
            parts = value.parts
            if parts and parts[0] == kind:
                value = Path(*parts[1:])
            path = self.root / kind / value
        if not str(path).endswith((".yaml", ".yml")):
            path = Path(f"{path}.yaml")
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ProfileError(f"profile reference escapes profile root: {reference}")
        return resolved

    def resolve(self, kind: ProfileKind, reference: str | Path) -> ResolvedProfile:
        path = self.path_for(kind, reference)
        data, sources = self._resolve_path(kind, path, ())
        self._validate(kind, data, path)
        return ResolvedProfile(
            kind=kind,
            reference=str(reference),
            data=data,
            sources=sources,
        )

    def _resolve_path(
        self,
        kind: ProfileKind,
        path: Path,
        stack: tuple[Path, ...],
    ) -> tuple[dict[str, Any], tuple[Path, ...]]:
        if path in stack:
            chain = " -> ".join(str(item) for item in (*stack, path))
            raise ProfileError(f"profile inheritance cycle: {chain}")

        child = _load_mapping(path)
        parent_reference = child.pop("extends", None)
        if parent_reference is None:
            return child, (path,)
        if not isinstance(parent_reference, str) or not parent_reference.strip():
            raise ProfileError(f"extends must be one non-empty profile reference: {path}")

        parent_path = self.path_for(kind, parent_reference)
        parent, sources = self._resolve_path(kind, parent_path, (*stack, path))
        return _deep_merge(parent, child), (*sources, path)

    @staticmethod
    def _validate(kind: ProfileKind, data: dict[str, Any], path: Path) -> None:
        profile_id = data.get("id")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ProfileError(f"profile requires a non-empty id: {path}")
        if kind != "models":
            return

        model = data.get("model")
        if not isinstance(model, dict):
            raise ProfileError(f"model profile requires a model mapping: {path}")
        artifact = model.get("artifact")
        if not isinstance(artifact, str) or not artifact.strip():
            raise ProfileError(f"model profile requires model.artifact: {path}")
        form = model.get("form")
        if form not in {"base", "adapter", "merged", "quantized", "checkpoint"}:
            raise ProfileError(f"unsupported model.form {form!r}: {path}")
        if form == "adapter" and not model.get("required_base"):
            raise ProfileError(f"adapter profile requires model.required_base: {path}")
        capabilities = model.get("capabilities")
        if not isinstance(capabilities, dict):
            raise ProfileError(f"model profile requires model.capabilities: {path}")
        context_window = capabilities.get("context_window")
        if not isinstance(context_window, int) or isinstance(context_window, bool) or context_window < 1:
            raise ProfileError(f"model.capabilities.context_window must be positive: {path}")
        weights = model.get("weights")
        if not isinstance(weights, dict) or not isinstance(weights.get("format"), str):
            raise ProfileError(f"model profile requires model.weights.format: {path}")


__all__ = ["ProfileError", "ProfileKind", "ProfileResolver", "ResolvedProfile"]
