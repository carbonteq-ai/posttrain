"""Immutable foundation model facts used across jobs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

from .artifacts import ArtifactRef, HubModelRef
from .errors import ContractError

_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    modalities: tuple[str, ...]
    native_context_window: int
    mtp: bool = False

    def __post_init__(self) -> None:
        if not self.modalities or any(not value for value in self.modalities):
            raise ContractError("model capabilities require at least one modality")
        if self.native_context_window < 1:
            raise ContractError("native context window must be positive")


type TemplateValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class ChatTemplate:
    source: Literal["tokenizer", "package"]
    resource: str | None = None

    def __post_init__(self) -> None:
        if self.source == "package" and not self.resource:
            raise ContractError("package chat templates require a resource name")
        if self.source == "tokenizer" and self.resource is not None:
            raise ContractError("tokenizer chat templates cannot declare a package resource")

    def text(self) -> str | None:
        if self.source == "tokenizer":
            return None
        assert self.resource is not None
        return files("posttrain.common.templates").joinpath(self.resource).read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class ReasoningMode:
    id: str
    chat_template_kwargs: tuple[tuple[str, TemplateValue], ...] = ()

    def __post_init__(self) -> None:
        if not _PROFILE_ID.fullmatch(self.id):
            raise ContractError(f"reasoning mode id is invalid: {self.id!r}")
        keys = [key for key, _ in self.chat_template_kwargs]
        if len(keys) != len(set(keys)) or any(not key for key in keys):
            raise ContractError("chat-template keyword names must be non-empty and unique")

    def kwargs(self) -> dict[str, TemplateValue]:
        return dict(self.chat_template_kwargs)


@dataclass(frozen=True, slots=True)
class ToolCallProtocol:
    id: Literal["qwen3_xml", "lfm2_pythonic"]
    assistant_format: str
    start_token: str
    end_token: str
    tool_response_role: Literal["tool"] = "tool"
    parallel_calls: bool = True

    def __post_init__(self) -> None:
        if not self.assistant_format.strip() or not self.start_token or not self.end_token:
            raise ContractError("tool-call protocol requires a format and boundary tokens")


@dataclass(frozen=True, slots=True)
class ConversationProfile:
    chat_template: ChatTemplate
    roles: tuple[Literal["system", "user", "assistant", "tool"], ...]
    reasoning_modes: tuple[ReasoningMode, ...]
    default_reasoning_mode: str
    tool_calls: ToolCallProtocol | None = None
    strips_past_reasoning: bool = False

    def __post_init__(self) -> None:
        if not self.roles or len(self.roles) != len(set(self.roles)):
            raise ContractError("conversation roles must be non-empty and unique")
        mode_ids = tuple(mode.id for mode in self.reasoning_modes)
        if not mode_ids or len(mode_ids) != len(set(mode_ids)):
            raise ContractError("reasoning modes must be non-empty and unique")
        if self.default_reasoning_mode not in mode_ids:
            raise ContractError("default reasoning mode must be declared by the conversation profile")

    def reasoning_mode(self, mode_id: str) -> ReasoningMode:
        for mode in self.reasoning_modes:
            if mode.id == mode_id:
                return mode
        supported = ", ".join(mode.id for mode in self.reasoning_modes)
        raise ContractError(f"unsupported reasoning mode {mode_id!r}; supported modes: {supported}")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    id: str
    artifact: HubModelRef
    family: str
    parameters: int
    instruction_tuned: bool
    capabilities: ModelCapabilities
    conversation: ConversationProfile
    hf_text_generation_architecture: str | None = None

    def __post_init__(self) -> None:
        if not _PROFILE_ID.fullmatch(self.id):
            raise ContractError(f"model profile id is invalid: {self.id!r}")
        if not self.family.strip():
            raise ContractError("model family cannot be empty")
        if self.parameters < 1:
            raise ContractError("parameter count must be positive")
        if self.hf_text_generation_architecture is not None and not self.hf_text_generation_architecture.strip():
            raise ContractError("text-generation architecture cannot be blank")

    @property
    def default_reasoning_mode(self) -> str:
        return self.conversation.default_reasoning_mode


@dataclass(frozen=True, slots=True)
class ModelVariant:
    """One loadable weight artifact interpreted through a foundation profile."""

    profile: ModelProfile
    artifact: ArtifactRef
    format: Literal["foundation", "peft-adapter", "merged"]

    def __post_init__(self) -> None:
        if self.format == "foundation" and self.artifact != self.profile.artifact:
            raise ContractError("foundation variants must use the profile's pinned artifact")

    @classmethod
    def foundation(cls, profile: ModelProfile) -> ModelVariant:
        return cls(profile=profile, artifact=profile.artifact, format="foundation")

    @property
    def base_artifact(self) -> HubModelRef:
        return self.profile.artifact
