"""Backend-neutral contracts for environment-driven online-RL rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from posttrain.common import JsonValue, ProducedArtifact, TraceObservation

from .data import RolloutDataset

type MessageRecord = Mapping[str, JsonValue]
type ToolRecord = Mapping[str, JsonValue]
type TokenSpan = tuple[int, int]


@dataclass(frozen=True, slots=True)
class PolicySampling:
    """Sampling controls shared by an environment and its policy generator."""

    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("policy sampling max_tokens must be positive")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("invalid policy sampling temperature or top_p")


@dataclass(frozen=True, slots=True)
class PolicyTurnRequest:
    """One model turn requested by an environment-owned rollout."""

    messages: tuple[MessageRecord, ...]
    sampling: PolicySampling
    tools: tuple[ToolRecord, ...] = ()
    session_id: str | None = None
    previous_prompt_ids: tuple[int, ...] = ()
    previous_completion_ids: tuple[int, ...] = ()
    tail_start: int = 0

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("policy turns require at least one message")
        if self.tail_start < 0 or self.tail_start > len(self.messages):
            raise ValueError("policy turn tail_start is outside the message sequence")
        if bool(self.previous_prompt_ids) != bool(self.previous_completion_ids):
            raise ValueError("incremental policy turns require both previous prompt and completion ids")

    @property
    def previous_token_ids(self) -> tuple[int, ...]:
        return (*self.previous_prompt_ids, *self.previous_completion_ids)


@dataclass(frozen=True, slots=True)
class PolicyTurnResult:
    """Exact policy output and token attribution for one environment turn."""

    message: MessageRecord
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    completion_logprobs: tuple[float, ...]
    finish_reason: Literal["stop", "length", "tool_calls"] | None
    prompt_message_spans: tuple[TokenSpan | None, ...] = ()
    prompt_is_content: tuple[bool, ...] = ()
    raw_response: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if not self.prompt_ids or not self.completion_ids:
            raise ValueError("policy turn results require prompt and completion token ids")
        if self.completion_logprobs and len(self.completion_logprobs) != len(self.completion_ids):
            raise ValueError("completion logprobs must align with completion ids")
        if self.prompt_is_content and len(self.prompt_is_content) != len(self.prompt_ids):
            raise ValueError("prompt content attribution must align with prompt ids")


class PolicyGenerator(Protocol):
    """Generate model turns without owning environment or trainer semantics."""

    async def generate(self, request: PolicyTurnRequest) -> PolicyTurnResult: ...


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """Stable task identities aligned with one trainer generation batch."""

    example_ids: tuple[str, ...]
    step: int
    model_id: str

    def __post_init__(self) -> None:
        if not self.example_ids:
            raise ValueError("online-RL batches require at least one example")
        if self.step < 0:
            raise ValueError("online-RL batch step cannot be negative")
        if not self.model_id.strip():
            raise ValueError("online-RL batch model id cannot be empty")


@dataclass(frozen=True, slots=True)
class TrainingRollout:
    """One scored trajectory translated into the sequence a trainer optimizes."""

    example_id: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    sampling_logprobs: tuple[float, ...]
    env_mask: tuple[bool, ...]
    reward: float
    is_truncated: bool
    trace: TraceObservation

    def __post_init__(self) -> None:
        if not self.prompt_ids or not self.completion_ids:
            raise ValueError("training rollouts require prompt and completion ids")
        if len(self.env_mask) != len(self.completion_ids):
            raise ValueError("environment mask must align with completion ids")
        if self.sampling_logprobs and len(self.sampling_logprobs) != len(self.completion_ids):
            raise ValueError("sampling logprobs must align with completion ids")
        if not any(self.env_mask):
            raise ValueError("training rollouts require at least one model-sampled token")


class OnlineRLBridge(Protocol):
    """Run native environment episodes against an injected policy generator."""

    @property
    def dataset(self) -> RolloutDataset: ...

    async def run(self, batch: RolloutBatch, generator: PolicyGenerator) -> Sequence[TrainingRollout]: ...

    def finalize(self) -> tuple[ProducedArtifact, ...]: ...


__all__ = [
    "MessageRecord",
    "OnlineRLBridge",
    "PolicyGenerator",
    "PolicySampling",
    "PolicyTurnRequest",
    "PolicyTurnResult",
    "RolloutBatch",
    "TokenSpan",
    "ToolRecord",
    "TrainingRollout",
]
