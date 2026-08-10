"""Backend-neutral contracts for environment-driven online-RL rollouts."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from posttrain.common import JsonValue, MetricBatchObservation, ProducedArtifact, TraceObservation
from posttrain.data import MessageRecord, RolloutDataset

type ToolRecord = Mapping[str, JsonValue]
type TokenSpan = tuple[int, int]


@dataclass(frozen=True, slots=True)
class AgenticTurn:
    """One sampled assistant turn aligned to a flattened trajectory."""

    completion_start: int
    completion_end: int
    anchor_state_key: str
    step_reward: float | None = None

    def __post_init__(self) -> None:
        if self.completion_start < 0 or self.completion_end <= self.completion_start:
            raise ValueError("agentic turn requires a positive half-open completion span")
        if not self.anchor_state_key.strip():
            raise ValueError("agentic turn anchor-state key cannot be empty")
        if self.step_reward is not None and not math.isfinite(self.step_reward):
            raise ValueError("agentic turn reward must be finite")


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
class EnvironmentRollout:
    """One scored trajectory translated into the sequence a trainer optimizes."""

    example_id: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    sampling_logprobs: tuple[float, ...]
    env_mask: tuple[bool, ...]
    reward: float
    is_truncated: bool
    trace: TraceObservation
    turns: tuple[AgenticTurn, ...] = ()

    def __post_init__(self) -> None:
        if not self.prompt_ids or not self.completion_ids:
            raise ValueError("training rollouts require prompt and completion ids")
        if len(self.env_mask) != len(self.completion_ids):
            raise ValueError("environment mask must align with completion ids")
        if self.sampling_logprobs and len(self.sampling_logprobs) != len(self.completion_ids):
            raise ValueError("sampling logprobs must align with completion ids")
        if not any(self.env_mask):
            raise ValueError("training rollouts require at least one model-sampled token")
        previous_end = 0
        for turn in self.turns:
            if turn.completion_start < previous_end or turn.completion_end > len(self.completion_ids):
                raise ValueError("agentic turn spans must be ordered, non-overlapping, and inside the completion")
            if not all(self.env_mask[turn.completion_start : turn.completion_end]):
                raise ValueError("agentic turn spans may contain only model-sampled tokens")
            previous_end = turn.completion_end


class EnvironmentRolloutBridge(Protocol):
    """Run native environment episodes against an injected policy generator."""

    @property
    def dataset(self) -> RolloutDataset: ...

    async def run(self, batch: RolloutBatch, generator: PolicyGenerator) -> Sequence[EnvironmentRollout]: ...

    def finalize(self) -> tuple[ProducedArtifact, ...]: ...


type RolloutCompletionObserver = Callable[[EnvironmentRollout], None]
type AsyncRolloutCompletionObserver = Callable[[EnvironmentRollout], Awaitable[None]]


class ObservedEnvironmentRolloutBridge(Protocol):
    """Optional bridge extension that exposes trajectories as they complete."""

    async def run_observed(
        self,
        batch: RolloutBatch,
        generator: PolicyGenerator,
        *,
        on_completed: AsyncRolloutCompletionObserver,
    ) -> Sequence[EnvironmentRollout]: ...


async def run_observed_rollouts(
    bridge: EnvironmentRolloutBridge,
    batch: RolloutBatch,
    generator: PolicyGenerator,
    observer: RolloutCompletionObserver,
) -> Sequence[EnvironmentRollout]:
    """Run a batch while submitting each completed rollout off the event loop.

    Observation is serialized because provider clients commonly maintain a
    run-local step counter. The provider is still responsible for queueing the
    remote write, so rollout workers wait only for bounded local submission,
    never for network persistence. Bridges without the optional streaming
    extension retain their batch-complete compatibility behavior.
    """

    observation_lock = asyncio.Lock()

    async def observe(rollout: EnvironmentRollout) -> None:
        async with observation_lock:
            await asyncio.to_thread(observer, rollout)

    if callable(getattr(bridge, "run_observed", None)):
        observed_bridge = cast(ObservedEnvironmentRolloutBridge, bridge)
        return await observed_bridge.run_observed(batch, generator, on_completed=observe)
    rollouts = await bridge.run(batch, generator)
    for rollout in rollouts:
        await observe(rollout)
    return rollouts


@dataclass(frozen=True, slots=True)
class EnvironmentRolloutEvidence:
    """Provider-neutral observations recovered from an isolated rollout runtime."""

    metrics: tuple[MetricBatchObservation, ...] = ()
    traces: tuple[TraceObservation, ...] = ()


__all__ = [
    "AgenticTurn",
    "AsyncRolloutCompletionObserver",
    "EnvironmentRolloutEvidence",
    "EnvironmentRolloutBridge",
    "EnvironmentRollout",
    "ObservedEnvironmentRolloutBridge",
    "PolicyGenerator",
    "PolicySampling",
    "PolicyTurnRequest",
    "PolicyTurnResult",
    "RolloutBatch",
    "RolloutCompletionObserver",
    "TokenSpan",
    "ToolRecord",
    "run_observed_rollouts",
]
