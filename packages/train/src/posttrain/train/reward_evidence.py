"""Validated reward evidence over original sampled trajectory coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

type RewardStatus = Literal["valid", "inapplicable", "abstained", "failed"]
type ObservationScope = Literal["prefix", "current-step", "full-trajectory"]


@dataclass(frozen=True, slots=True)
class SpanAssessment:
    """Scorer evidence on a retained span; this is not a prepared advantage."""

    evidence_ref: str
    span_id: str
    components: tuple[RewardValue, ...]
    scorer_revision: str
    observed_input_ref: str
    observation_scope: ObservationScope
    semantic_kind: str
    scorer_snapshot: str

    def __post_init__(self) -> None:
        identities = (
            self.evidence_ref,
            self.span_id,
            self.scorer_revision,
            self.observed_input_ref,
            self.semantic_kind,
            self.scorer_snapshot,
        )
        if any(not value.strip() for value in identities) or not self.components:
            raise InvalidRewardEvidence("span assessment requires scorer, input, span and retained evidence identities")
        if self.observation_scope not in {"prefix", "current-step", "full-trajectory"}:
            raise InvalidRewardEvidence("span assessment requires an explicit observation scope")
        if len({value.name for value in self.components}) != len(self.components):
            raise InvalidRewardEvidence("span assessment component names must be unique")


class InvalidRewardEvidence(ValueError):
    """Evidence cannot be admitted to an algorithm's normalization population."""


@dataclass(frozen=True, slots=True)
class CreditAttempt:
    """Coverage survives even when an assignment produced no contributions."""

    assignment_id: str
    rule_digest: str
    invocation_id: str
    attempt_id: str
    status: str
    requested_count: int
    missing_count: int
    overlap_policy: Literal["reject", "sum"] = "reject"

    def __post_init__(self) -> None:
        if not all((self.assignment_id, self.rule_digest, self.invocation_id, self.attempt_id)):
            raise InvalidRewardEvidence("credit attempt requires native identities")
        if self.status not in {"running", "complete", "partial", "failed", "interrupted"}:
            raise InvalidRewardEvidence("unknown credit attempt status")
        if not 0 <= self.missing_count <= self.requested_count:
            raise InvalidRewardEvidence("invalid credit attempt coverage")
        if self.overlap_policy not in {"reject", "sum"}:
            raise InvalidRewardEvidence("credit attempt requires an explicit overlap policy")

    def require_complete(self) -> None:
        if self.status != "complete" or self.missing_count:
            raise InvalidRewardEvidence("selected credit assignment has incomplete coverage")


@dataclass(frozen=True, slots=True)
class AssignedCreditEvidence:
    attempts: tuple[CreditAttempt, ...] = ()
    contributions: tuple[AssignedCredit, ...] = ()
    branch_digest: str | None = None

    def __post_init__(self) -> None:
        ids = {attempt.assignment_id for attempt in self.attempts}
        if len(ids) != len(self.attempts):
            raise InvalidRewardEvidence("duplicate credit assignment receipt")
        if any(item.assignment_id not in ids for item in self.contributions):
            raise InvalidRewardEvidence("credit contribution requires its assignment receipt")
        if self.attempts and not self.branch_digest:
            raise InvalidRewardEvidence("assigned evidence requires its selected path identity")
        if any(item.branch_digest != self.branch_digest for item in self.contributions):
            raise InvalidRewardEvidence("credit contribution belongs to another selected path")

    def select(
        self,
        *,
        rule_digest: str,
        invocation_id: str,
        attempt_id: str,
        channel: str,
    ) -> tuple[AssignedCredit, ...]:
        """Admit one complete attempt and channel; never combine retries implicitly."""
        attempts = tuple(
            item
            for item in self.attempts
            if (
                item.rule_digest == rule_digest
                and item.invocation_id == invocation_id
                and item.attempt_id == attempt_id
            )
        )
        if len(attempts) != 1:
            raise InvalidRewardEvidence("selected credit attempt is absent or ambiguous")
        attempts[0].require_complete()
        result = tuple(
            item
            for item in self.contributions
            if (item.assignment_id == attempts[0].assignment_id and item.channel == channel)
        )
        if not result or any(item.status != "valid" for item in result):
            raise InvalidRewardEvidence("selected credit channel is absent or unavailable")
        return result


@dataclass(frozen=True, slots=True)
class AssignedCredit:
    """One native contribution on a source-bound selected path, before advantages.

    Consumers must explicitly choose a rule, attempt and channel. Merely retaining
    these records never changes the episode reward or the optimizer objective.
    """

    assignment_id: str
    contribution_id: str
    rule_digest: str
    attempt_id: str
    channel: str
    semantics: str
    branch_digest: str
    status: Literal["valid", "unavailable", "gated"]
    alignment: Literal["exact", "unsupported", "failed", "absent"]
    allocation: Literal["turn_boundary", "broadcast", "fixed_mass"]
    value: float | None
    weight: float
    signal_digest: str
    units: str
    transformation: str
    parent_assessment_ids: tuple[str, ...]
    attribution: str | None
    recipient_kind: str
    recipient_id: str
    source_snapshot_id: str
    recipient_trace_id: str | None
    gate_ids: tuple[str, ...] = ()
    intervals: tuple[tuple[int, int], ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if not all(
            (
                self.assignment_id,
                self.contribution_id,
                self.rule_digest,
                self.attempt_id,
                self.channel,
                self.semantics,
                self.branch_digest,
                self.signal_digest,
                self.units,
                self.transformation,
                self.recipient_kind,
                self.recipient_id,
                self.source_snapshot_id,
            )
        ):
            raise InvalidRewardEvidence("assigned credit requires retained source and rule identities")
        if (
            self.status not in {"valid", "unavailable", "gated"}
            or self.alignment not in {"exact", "unsupported", "failed", "absent"}
            or self.allocation not in {"turn_boundary", "broadcast", "fixed_mass"}
        ):
            raise InvalidRewardEvidence("unknown assigned credit status or allocation")
        if isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight < 0:
            raise InvalidRewardEvidence("credit weight must be finite and nonnegative")
        if self.status == "valid":
            if self.value is None or isinstance(self.value, bool) or not math.isfinite(self.value):
                raise InvalidRewardEvidence("valid assigned credit requires a finite value")
            if not math.isfinite(self.value * self.weight):
                raise InvalidRewardEvidence("weighted assigned value must be finite")
        elif self.value is not None:
            raise InvalidRewardEvidence("unavailable assigned credit cannot carry a value")
        if self.alignment == "exact":
            if self.status != "valid" or not self.intervals:
                raise InvalidRewardEvidence("exact credit requires valid sampled support")
        elif self.intervals:
            raise InvalidRewardEvidence("unresolved alignment cannot expose policy support")
        for start, end in self.intervals:
            if type(start) is not int or type(end) is not int or start < 0 or end <= start:
                raise InvalidRewardEvidence("credit intervals must be positive half-open integers")

    def support(self, sampled_mask: tuple[bool, ...]) -> tuple[bool, ...]:
        if self.alignment != "exact" or self.status != "valid":
            raise InvalidRewardEvidence("consumer requires resolved valid assigned credit")
        result = [False] * len(sampled_mask)
        for start, end in self.intervals:
            if end > len(result) or not all(sampled_mask[start:end]):
                raise InvalidRewardEvidence("credit must address original sampled completion tokens")
            result[start:end] = [True] * (end - start)
        return tuple(result)

    def token_values(self, sampled_mask: tuple[bool, ...]) -> tuple[float, ...]:
        """Explicit allocation of raw reward, never returns or advantages."""
        if self.semantics not in {"outcome", "progress", "cost"}:
            raise InvalidRewardEvidence("signal requires an explicit reward conversion before allocation")
        support = self.support(sampled_mask)
        if self.allocation == "turn_boundary":
            raise InvalidRewardEvidence("boundary events require a turn-return consumer")
        assert self.value is not None
        value = self.value * self.weight
        if self.allocation == "fixed_mass":
            value /= sum(support)
        return tuple(value if selected else 0.0 for selected in support)


@dataclass(frozen=True, slots=True)
class RewardValue:
    name: str
    status: RewardStatus
    value: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or self.status not in {"valid", "inapplicable", "abstained", "failed"}:
            raise InvalidRewardEvidence("reward requires a name and recognized status")
        if self.status == "valid":
            if self.value is None or isinstance(self.value, bool) or not math.isfinite(self.value):
                raise InvalidRewardEvidence("valid reward requires a finite numeric value")
        elif self.value is not None:
            raise InvalidRewardEvidence("unavailable reward must not carry a numeric value")


@dataclass(frozen=True, slots=True)
class ProcessCredit:
    """Resolved critique spans; empty spans are valid only for a resolved critique."""

    status: RewardStatus
    projection_id: str
    evidence_ref: str
    error_spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"valid", "inapplicable", "abstained", "failed"}:
            raise InvalidRewardEvidence("critique requires a recognized status")
        if not self.projection_id.strip() or not self.evidence_ref.strip():
            raise InvalidRewardEvidence("critique requires projection and retained evidence identities")
        if self.status != "valid" and self.error_spans:
            raise InvalidRewardEvidence("unresolved critique cannot carry resolved error spans")
        for start, end in self.error_spans:
            if type(start) is not int or type(end) is not int or start < 0 or end <= start:
                raise InvalidRewardEvidence("error spans must be positive half-open integer token intervals")

    def error_mask(self, sampled_mask: tuple[bool, ...]) -> tuple[bool, ...]:
        if self.status != "valid":
            raise InvalidRewardEvidence("CAPO requires a resolved valid critique")
        result = [False] * len(sampled_mask)
        for start, end in self.error_spans:
            if end > len(sampled_mask) or not all(sampled_mask[start:end]):
                raise InvalidRewardEvidence("error span must address only original sampled policy tokens")
            result[start:end] = [True] * (end - start)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RewardEvidence:
    prompt_group_id: str
    rollout_id: str
    trace_id: str
    branch_id: str
    projection_id: str
    components: tuple[RewardValue, ...] = ()
    process: ProcessCredit | None = None

    def __post_init__(self) -> None:
        if not all(
            x.strip()
            for x in (self.prompt_group_id, self.rollout_id, self.trace_id, self.branch_id, self.projection_id)
        ):
            raise InvalidRewardEvidence("structured rewards require group, rollout, trace, branch and projection IDs")
        names = [component.name for component in self.components]
        if len(set(names)) != len(names):
            raise InvalidRewardEvidence("reward component names must be unique")

    def require_components(self, names: tuple[str, ...]) -> tuple[float, ...]:
        by_name = {component.name: component for component in self.components}
        result = []
        for name in names:
            component = by_name.get(name)
            if component is None or component.status != "valid" or component.value is None:
                raise InvalidRewardEvidence(f"required component {name!r} is unavailable")
            result.append(component.value)
        return tuple(result)
