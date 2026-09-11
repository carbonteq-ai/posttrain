"""Validated reward evidence over original sampled trajectory coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

type RewardStatus = Literal["valid", "inapplicable", "abstained", "failed"]


class InvalidRewardEvidence(ValueError):
    """Evidence cannot be admitted to an algorithm's normalization population."""


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
