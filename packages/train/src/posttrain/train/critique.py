"""Pure, versioned critique resolution over original sampled-token coordinates.

Scorers own rendering and judging. They return step IDs, not text spans to be
searched in a retokenized response. Resolution never calls a policy or replays
an environment. Each stochastic judgment has its own retained evidence identity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from .reward_evidence import InvalidRewardEvidence, ProcessCredit, RewardStatus


@dataclass(frozen=True, slots=True)
class CritiqueStep:
    id: str
    token_spans: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.token_spans:
            raise InvalidRewardEvidence("critique step requires an identity and original token spans")
        ProcessCredit("valid", "validation", "validation", self.token_spans)


@dataclass(frozen=True, slots=True)
class CritiqueVote:
    id: str
    status: RewardStatus
    evidence_ref: str
    erroneous_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ProcessCredit(self.status, "validation", self.evidence_ref)
        if not self.id.strip() or any(not step.strip() for step in self.erroneous_steps):
            raise InvalidRewardEvidence("critique vote requires named vote and steps")
        if self.status != "valid" and self.erroneous_steps:
            raise InvalidRewardEvidence("unresolved critique cannot carry resolved step errors")


def resolve_critique(
    steps: tuple[CritiqueStep, ...],
    votes: tuple[CritiqueVote, ...],
    sampled_mask: tuple[bool, ...],
    *,
    projection_id: str,
    evidence_ref: str,
    expected_votes: int,
    rule: Literal["intersection", "strict_majority", "at_least_half"] = "intersection",
) -> ProcessCredit:
    """Resolve all required votes; unknown IDs, failures and partial panels fail closed.

    The caller retains the complete panel/step mapping at evidence_ref before
    publishing this result. Even-count ties are explicit, never inferred from
    an ambiguous 'majority' label. Duplicate mentions in one vote count once.
    """
    if type(expected_votes) is not int or expected_votes < 1 or len(votes) != expected_votes:
        raise InvalidRewardEvidence("critique resolution requires the complete declared vote panel")
    if len({vote.id for vote in votes}) != len(votes):
        raise InvalidRewardEvidence("duplicate critique sample identity")
    if len({vote.evidence_ref for vote in votes}) != len(votes):
        raise InvalidRewardEvidence("one retained critique cannot count as multiple votes")
    if not sampled_mask or not any(sampled_mask) or any(type(value) is not bool for value in sampled_mask):
        raise InvalidRewardEvidence("critique requires an original boolean sampled-token mask")
    if any(vote.status != "valid" for vote in votes):
        raise InvalidRewardEvidence("all required critiques must resolve successfully")
    by_id = {step.id: step for step in steps}
    if not steps or len(by_id) != len(steps):
        raise InvalidRewardEvidence("critique projection requires unique step identities")
    for step in steps:
        ProcessCredit("valid", projection_id, evidence_ref, step.token_spans).error_mask(sampled_mask)
    counts: Counter[str] = Counter()
    for vote in votes:
        if not set(vote.erroneous_steps).issubset(by_id):
            raise InvalidRewardEvidence("critique refers to a step outside the retained projection")
        counts.update(set(vote.erroneous_steps))
    if rule == "intersection":
        threshold = expected_votes
    elif rule == "strict_majority":
        threshold = expected_votes // 2 + 1
    elif rule == "at_least_half":
        threshold = (expected_votes + 1) // 2
    else:
        raise InvalidRewardEvidence("unsupported critique voting rule")
    # Canonical union is independent of vote ordering, repeated mentions and
    # overlapping step projections. Excluded observation tokens remain holes.
    mask = [False] * len(sampled_mask)
    for step in steps:
        if counts[step.id] >= threshold:
            for start, end in step.token_spans:
                mask[start:end] = [True] * (end - start)
    spans: list[tuple[int, int]] = []
    start = None
    for index, selected in enumerate([*mask, False]):
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            spans.append((start, index))
            start = None
    return ProcessCredit("valid", projection_id, evidence_ref, tuple(spans))
