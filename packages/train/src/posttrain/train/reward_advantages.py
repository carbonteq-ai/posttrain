"""Complete-population numerical contracts for structured online rewards.

Inputs must contain the entire admitted logical batch, gathered before calling
this module. Output rows retain input order, including interleaved prompt groups.
Python float statistics use accurate summation independently of model precision.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .reward_evidence import InvalidRewardEvidence, RewardEvidence


@dataclass(frozen=True, slots=True)
class RewardAdvantages:
    token_advantages: tuple[tuple[float, ...], ...]
    raw_rewards: tuple[tuple[float, ...], ...]


def _normalize(values: Sequence[float], epsilon: float) -> tuple[float, ...]:
    if len(values) < 2:
        raise InvalidRewardEvidence("normalization requires at least two eligible values")
    # Center around a nearby origin before summation to retain small differences
    # on top of large common reward offsets.
    origin = values[0]
    offsets = [value - origin for value in values]
    mean = math.fsum(offsets) / len(offsets)
    centered = [value - mean for value in offsets]
    scale = math.hypot(*centered) / math.sqrt(len(centered) - 1)
    result = tuple(value / (scale + epsilon) for value in centered)
    if not all(math.isfinite(value) for value in result):
        raise InvalidRewardEvidence("reward normalization overflowed")
    return result


def _groups(
    evidence: Sequence[RewardEvidence],
    masks: Sequence[tuple[bool, ...]],
    group_size: int,
    epsilon: float,
) -> dict[str, list[int]]:
    if type(group_size) is not int or group_size < 2:
        raise InvalidRewardEvidence("group size must be an integer of at least two")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise InvalidRewardEvidence("normalization epsilon must be finite and positive")
    if not evidence or len(evidence) != len(masks):
        raise InvalidRewardEvidence("evidence must align with a nonempty mask batch")
    if len({item.rollout_id for item in evidence}) != len(evidence):
        raise InvalidRewardEvidence("duplicate rollout identity in logical batch")
    if len({item.projection_id for item in evidence}) != 1:
        raise InvalidRewardEvidence("one logical batch requires a consistent reward projection")
    if len({(item.trace_id, item.branch_id) for item in evidence}) != len(evidence):
        raise InvalidRewardEvidence("duplicate native trajectory in logical batch")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (item, mask) in enumerate(zip(evidence, masks, strict=True)):
        if not mask or not any(mask) or any(type(value) is not bool for value in mask):
            raise InvalidRewardEvidence("each response requires a boolean mask with sampled tokens")
        groups[item.prompt_group_id].append(index)
    if any(len(indices) != group_size for indices in groups.values()):
        raise InvalidRewardEvidence("logical batch requires complete prompt groups")
    return groups


def compute_gdpo_advantages(
    evidence: Sequence[RewardEvidence],
    masks: Sequence[tuple[bool, ...]],
    *,
    component_names: tuple[str, ...],
    component_weights: tuple[float, ...],
    group_size: int,
    epsilon: float = 1e-4,
) -> RewardAdvantages:
    """Sample-SD component normalization, weighted sum, rollout-wise whitening."""
    groups = _groups(evidence, masks, group_size, epsilon)
    if not component_names or len(set(component_names)) != len(component_names):
        raise InvalidRewardEvidence("GDPO requires unique ordered component names")
    if (
        len(component_weights) != len(component_names)
        or not all(math.isfinite(weight) and weight >= 0 for weight in component_weights)
        or not any(component_weights)
    ):
        raise InvalidRewardEvidence("GDPO requires aligned finite nonnegative weights with a positive weight")
    raw = tuple(item.require_components(component_names) for item in evidence)
    aggregate = [0.0] * len(evidence)
    for indices in groups.values():
        columns = [
            _normalize([raw[index][column] for index in indices], epsilon) for column in range(len(component_names))
        ]
        for local, index in enumerate(indices):
            aggregate[index] = math.fsum(
                weight * column[local] for weight, column in zip(component_weights, columns, strict=True)
            )
    normalized = _normalize(aggregate, epsilon)
    return RewardAdvantages(
        tuple(
            tuple(value if sampled else 0.0 for sampled in mask) for value, mask in zip(normalized, masks, strict=True)
        ),
        raw,
    )


def compute_capo_advantages(
    evidence: Sequence[RewardEvidence],
    masks: Sequence[tuple[bool, ...]],
    *,
    group_size: int,
    outcome_component: str = "outcome",
    outcome_weight: float = 2.0,
    process_weight: float = 1.0,
    epsilon: float = 1e-6,
) -> RewardAdvantages:
    """Paper-profile direct token rewards, sample-SD normalization per group."""
    groups = _groups(evidence, masks, group_size, epsilon)
    if not all(math.isfinite(value) and value >= 0 for value in (outcome_weight, process_weight)):
        raise InvalidRewardEvidence("CAPO weights must be finite and nonnegative")
    if outcome_weight <= process_weight:
        raise InvalidRewardEvidence("initial CAPO profile requires outcome-dominant weights")
    raw: list[tuple[float, ...]] = []
    for item, mask in zip(evidence, masks, strict=True):
        (outcome,) = item.require_components((outcome_component,))
        if outcome not in (0.0, 1.0):
            raise InvalidRewardEvidence("initial CAPO profile requires binary verified outcomes")
        if item.process is None:
            raise InvalidRewardEvidence("CAPO requires process evidence")
        errors = item.process.error_mask(mask)
        raw.append(
            tuple(
                outcome_weight * outcome - process_weight * error if sampled else 0.0
                for sampled, error in zip(mask, errors, strict=True)
            )
        )
    result = [[0.0] * len(mask) for mask in masks]
    for indices in groups.values():
        positions = [(index, token) for index in indices for token, sampled in enumerate(masks[index]) if sampled]
        normalized = _normalize([raw[index][token] for index, token in positions], epsilon)
        for (index, token), value in zip(positions, normalized, strict=True):
            result[index][token] = value
    return RewardAdvantages(tuple(tuple(row) for row in result), tuple(raw))
