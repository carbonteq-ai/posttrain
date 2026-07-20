"""Immutable foundation model facts used across jobs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .artifacts import HubModelRef
from .errors import ContractError

_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    modalities: tuple[str, ...]
    reasoning_modes: tuple[str, ...]
    native_context_window: int
    mtp: bool = False

    def __post_init__(self) -> None:
        if not self.modalities or any(not value for value in self.modalities):
            raise ContractError("model capabilities require at least one modality")
        if not self.reasoning_modes or any(not value for value in self.reasoning_modes):
            raise ContractError("model capabilities require at least one reasoning mode")
        if self.native_context_window < 1:
            raise ContractError("native context window must be positive")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    id: str
    artifact: HubModelRef
    family: str
    parameters: int
    instruction_tuned: bool
    renderer: str
    default_reasoning_mode: str
    capabilities: ModelCapabilities

    def __post_init__(self) -> None:
        if not _PROFILE_ID.fullmatch(self.id):
            raise ContractError(f"model profile id is invalid: {self.id!r}")
        if not self.family.strip():
            raise ContractError("model family cannot be empty")
        if self.parameters < 1:
            raise ContractError("parameter count must be positive")
        if not self.renderer.strip():
            raise ContractError("renderer selection cannot be empty")
        if self.default_reasoning_mode not in self.capabilities.reasoning_modes:
            raise ContractError("default reasoning mode must be declared by model capabilities")
