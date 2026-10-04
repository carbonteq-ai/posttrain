"""Explicit algorithm input projections over retained native assigned credit.

These helpers produce raw rewards, not advantages. They do not alter official
native outcomes, choose retries, combine channels or enable a training recipe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from .online_rl import EnvironmentRollout
from .reward_evidence import AssignedCredit, InvalidRewardEvidence


@dataclass(frozen=True, slots=True)
class CreditSelection:
    rule_digest: str
    invocation_id: str
    attempt_id: str
    channel: str

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.rule_digest,
                self.invocation_id,
                self.attempt_id,
                self.channel,
            )
        ):
            raise InvalidRewardEvidence("credit selection requires rule, invocation, attempt and channel")


def _selected(rollout: EnvironmentRollout, selection: CreditSelection) -> tuple[AssignedCredit, ...]:
    records = rollout.assigned_credit.select(
        rule_digest=selection.rule_digest,
        invocation_id=selection.invocation_id,
        attempt_id=selection.attempt_id,
        channel=selection.channel,
    )
    if len({(item.signal_digest, item.units, item.semantics) for item in records}) != 1:
        raise InvalidRewardEvidence("selected channel requires one declared signal and unit")
    if any(
        item.recipient_trace_id is not None and item.recipient_trace_id != rollout.trace.external_id for item in records
    ):
        raise InvalidRewardEvidence("selected credit belongs to another trace")
    return records


def episode_reward(rollout: EnvironmentRollout, selection: CreditSelection) -> float:
    """One explicitly assigned episode outcome; token alignment is unnecessary."""
    records = _selected(rollout, selection)
    if len(records) != 1:
        raise InvalidRewardEvidence("episode outcome requires exactly one contribution")
    record = records[0]
    if record.recipient_kind not in {"episode", "trace"} or record.semantics != "outcome":
        raise InvalidRewardEvidence("episode consumer requires an episode or trace outcome")
    if record.allocation != "turn_boundary" or record.reason != "explicit_turn_or_span_required":
        raise InvalidRewardEvidence("episode outcome has incompatible allocation or source alignment")
    if record.alignment != "unsupported" or (
        record.recipient_trace_id is not None and record.recipient_trace_id != rollout.trace.external_id
    ):
        raise InvalidRewardEvidence("episode outcome belongs to another selected trajectory")
    assert record.value is not None
    return record.value * record.weight


def with_turn_rewards(
    rollout: EnvironmentRollout,
    selection: CreditSelection,
    *,
    terminal_outcome: Literal["separate", "include_in_last_turn"],
) -> EnvironmentRollout:
    """Dense whole-turn boundary events for an existing turn-return estimator.

    Sparse missing turns remain unavailable. Environments can explicitly return
    valid zero events; this adapter never invents them. The caller explicitly
    chooses whether the episode outcome also enters the final turn return.
    SAMPO's separate episode component always continues to use rollout.reward.
    """
    records = _selected(rollout, selection)
    rewards: dict[int, float] = {}
    for record in records:
        if record.allocation != "turn_boundary" or record.semantics not in {"outcome", "progress", "cost"}:
            raise InvalidRewardEvidence("turn consumer requires converted boundary rewards")
        support = record.support(rollout.env_mask)
        matches = [
            index
            for index, turn in enumerate(rollout.turns)
            if support
            == tuple(turn.completion_start <= position < turn.completion_end for position in range(len(support)))
        ]
        if len(matches) != 1 or matches[0] in rewards:
            raise InvalidRewardEvidence("boundary credit must address exactly one unique whole turn")
        assert record.value is not None
        rewards[matches[0]] = record.value * record.weight
    if len(rewards) != len(rollout.turns) or not rewards:
        raise InvalidRewardEvidence("turn consumer requires complete explicit turn rewards")
    if terminal_outcome == "include_in_last_turn":
        rewards[len(rollout.turns) - 1] += rollout.reward
    elif terminal_outcome != "separate":
        raise InvalidRewardEvidence("turn consumer requires an explicit terminal outcome policy")
    if not all(math.isfinite(value) for value in rewards.values()):
        raise InvalidRewardEvidence("combined turn rewards must remain finite")
    return replace(
        rollout, turns=tuple(replace(turn, step_reward=rewards[index]) for index, turn in enumerate(rollout.turns))
    )


def local_token_rewards(rollout: EnvironmentRollout, selection: CreditSelection) -> tuple[float, ...]:
    """Explicit same-channel raw token allocation; normalization stays downstream."""
    records = _selected(rollout, selection)
    receipt = next(item for item in rollout.assigned_credit.attempts if item.assignment_id == records[0].assignment_id)
    occupied = set()
    for record in records:
        selected = {index for index, eligible in enumerate(record.support(rollout.env_mask)) if eligible}
        if occupied & selected and receipt.overlap_policy != "sum":
            raise InvalidRewardEvidence("overlapping local credit requires an explicit sum policy")
        occupied |= selected
    rows = tuple(record.token_values(rollout.env_mask) for record in records)
    values = tuple(math.fsum(column) for column in zip(*rows, strict=True))
    if not all(math.isfinite(value) for value in values):
        raise InvalidRewardEvidence("summed local rewards must remain finite")
    return values
