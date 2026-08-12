"""Immutable foundation model facts used across jobs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Literal

from .artifacts import ArtifactRef, HubModelRef, JsonValue, LocalArtifactRef, StoredArtifactRef, TrackioArtifactRef
from .errors import ContractError

_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VARIANT_ID = re.compile(r"^[a-z0-9][a-z0-9._/@:-]*$")


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
    id: Literal["qwen3_xml", "lfm2_pythonic", "gemma4_structured"]
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
class RendererContract:
    """Versioned model-interface contract used by train, eval, and serve adapters."""

    id: str
    model_family: str
    conversation: ConversationProfile

    def __post_init__(self) -> None:
        if not _VARIANT_ID.fullmatch(self.id):
            raise ContractError(f"renderer contract id is invalid: {self.id!r}")
        if not self.model_family.strip():
            raise ContractError("renderer contract model_family cannot be empty")


type ModelForm = Literal[
    "foundation",
    "adapter",
    "peft-adapter",
    "merged",
    "full-finetuned",
    "weight-quantized",
]


@dataclass(frozen=True, slots=True)
class ModelVariant:
    """One exact loadable weight state and its explicit model-interface facts."""

    id: str
    artifact: ArtifactRef
    form: ModelForm
    weight_precision: str
    family: str
    parameters: int
    instruction_tuned: bool
    renderer: RendererContract
    capabilities: ModelCapabilities
    base: HubModelRef
    revision: str | None = None
    digest: str | None = None
    tokenizer_fingerprint: str | None = None
    chat_template_fingerprint: str | None = None
    quantization: Mapping[str, JsonValue] = MappingProxyType({})
    parent: str | None = None
    provenance: Mapping[str, JsonValue] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not _VARIANT_ID.fullmatch(self.id):
            raise ContractError(f"model variant id is invalid: {self.id!r}")
        if not self.weight_precision.strip():
            raise ContractError("model variant weight_precision cannot be empty")
        if not self.family.strip() or self.parameters < 1:
            raise ContractError("model variant requires a family and positive parameter count")
        if self.renderer.model_family != self.family:
            raise ContractError("renderer contract is incompatible with the model family")
        revision = self.revision
        digest = self.digest
        if revision is None and isinstance(self.artifact, HubModelRef):
            revision = self.artifact.revision
        if digest is None and isinstance(self.artifact, LocalArtifactRef):
            digest = self.artifact.digest
        if revision is None and isinstance(self.artifact, (StoredArtifactRef, TrackioArtifactRef)):
            revision = self.artifact.version
        if revision is None and digest is None:
            raise ContractError("model variant requires an immutable revision or digest")
        if self.tokenizer_fingerprint is not None and re.fullmatch(r"[0-9a-f]{64}", self.tokenizer_fingerprint) is None:
            raise ContractError("tokenizer fingerprint must be a sha256 digest")
        if (
            self.chat_template_fingerprint is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.chat_template_fingerprint) is None
        ):
            raise ContractError("chat template fingerprint must be a sha256 digest")
        if self.form == "foundation" and self.artifact != self.base:
            raise ContractError("foundation variants must use their pinned base artifact")
        if self.form == "weight-quantized" and not self.quantization:
            raise ContractError("weight-quantized variants require quantization metadata")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "quantization", MappingProxyType(dict(self.quantization)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def renderer_contract(self) -> str:
        return self.renderer.id

    @property
    def conversation(self) -> ConversationProfile:
        return self.renderer.conversation

    @property
    def default_reasoning_mode(self) -> str:
        return self.conversation.default_reasoning_mode

    @property
    def artifact_uri(self) -> str:
        if isinstance(self.artifact, HubModelRef):
            return self.artifact.uri
        if isinstance(self.artifact, LocalArtifactRef):
            return f"file://{self.artifact.path}#{self.artifact.digest}"
        if isinstance(self.artifact, StoredArtifactRef):
            return f"{self.artifact.provider}://{self.artifact.namespace}/{self.artifact.name}@{self.artifact.version}"
        return f"trackio://{self.artifact.project}/{self.artifact.name}@{self.artifact.version}"


@dataclass(frozen=True, slots=True)
class ModelArtifactDescriptor:
    """Immutable model facts stored beside a checkpoint-derived model view."""

    form: ModelForm
    weight_precision: str
    family: str
    parameters: int
    instruction_tuned: bool
    renderer: RendererContract
    capabilities: ModelCapabilities
    base: HubModelRef
    tokenizer_fingerprint: str | None = None
    chat_template_fingerprint: str | None = None
    quantization: Mapping[str, JsonValue] = MappingProxyType({})
    parent: str | None = None
    source_run_id: str | None = None
    checkpoint_step: int | None = None
    checkpoint_snapshot_id: str | None = None
    projection_schema: str = "model-view@1"

    def __post_init__(self) -> None:
        if not self.weight_precision.strip() or not self.family.strip():
            raise ContractError("model artifact descriptor requires precision and family")
        if self.parameters < 1:
            raise ContractError("model artifact descriptor requires positive parameters")
        if self.renderer.model_family != self.family:
            raise ContractError("model artifact descriptor renderer is incompatible with family")
        if self.tokenizer_fingerprint is not None and re.fullmatch(r"[0-9a-f]{64}", self.tokenizer_fingerprint) is None:
            raise ContractError("model artifact descriptor tokenizer fingerprint must be a sha256 digest")
        if (
            self.chat_template_fingerprint is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.chat_template_fingerprint) is None
        ):
            raise ContractError("model artifact descriptor chat template fingerprint must be a sha256 digest")
        if self.checkpoint_step is not None and self.checkpoint_step < 0:
            raise ContractError("model artifact descriptor checkpoint step cannot be negative")
        if self.checkpoint_snapshot_id is not None and not self.checkpoint_snapshot_id.strip():
            raise ContractError("model artifact descriptor checkpoint snapshot id cannot be empty")
        if not self.projection_schema.strip():
            raise ContractError("model artifact descriptor projection schema cannot be empty")
        object.__setattr__(self, "quantization", MappingProxyType(dict(self.quantization)))

    @classmethod
    def from_model_variant(
        cls,
        model: ModelVariant,
        *,
        source_run_id: str | None = None,
        checkpoint_step: int | None = None,
        checkpoint_snapshot_id: str | None = None,
        projection_schema: str = "model-view@1",
    ) -> ModelArtifactDescriptor:
        return cls(
            form=model.form,
            weight_precision=model.weight_precision,
            family=model.family,
            parameters=model.parameters,
            instruction_tuned=model.instruction_tuned,
            renderer=model.renderer,
            capabilities=model.capabilities,
            base=model.base,
            tokenizer_fingerprint=model.tokenizer_fingerprint,
            chat_template_fingerprint=model.chat_template_fingerprint,
            quantization=model.quantization,
            parent=model.parent,
            source_run_id=source_run_id,
            checkpoint_step=checkpoint_step,
            checkpoint_snapshot_id=checkpoint_snapshot_id,
            projection_schema=projection_schema,
        )

    def to_model_variant(
        self,
        reference: StoredArtifactRef,
        *,
        variant_id: str,
        kind: Literal["model-adapter", "model-weights"],
    ) -> ModelVariant:
        """Rebuild a catalog-independent model variant from a committed view."""

        if kind == "model-adapter" and self.form not in {"adapter", "peft-adapter"}:
            raise ContractError("model-adapter views require an adapter model form")
        if kind == "model-weights" and self.form in {"adapter", "peft-adapter"}:
            raise ContractError("model-weights views cannot use an adapter model form")
        return ModelVariant(
            id=variant_id,
            artifact=reference,
            form=self.form,
            weight_precision=self.weight_precision,
            family=self.family,
            parameters=self.parameters,
            instruction_tuned=self.instruction_tuned,
            renderer=self.renderer,
            capabilities=self.capabilities,
            base=self.base,
            revision=reference.version,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            chat_template_fingerprint=self.chat_template_fingerprint,
            quantization=self.quantization,
            parent=self.parent,
            provenance={
                "projection_schema": self.projection_schema,
                "source_run_id": self.source_run_id,
                "checkpoint_step": self.checkpoint_step,
                "checkpoint_snapshot_id": self.checkpoint_snapshot_id,
                "artifact_kind": kind,
            },
        )
