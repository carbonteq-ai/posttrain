"""Turn-addressed assessments over a retained native token map, not advantages.

The bridge owns token provenance. Plugins return IDs and named scores only;
they cannot select arbitrary policy tokens by supplying their own mask.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .reward_evidence import InvalidRewardEvidence, ProcessCredit, RewardValue

TURN_PROJECTION = "assistant-turns@1"


@dataclass(frozen=True, slots=True)
class TurnTokenMap:
    id: str
    node_index: int
    token_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class TurnAssessment:
    turn_id: str
    components: tuple[RewardValue, ...]
    evidence_ref: str

    def __post_init__(self) -> None:
        if not self.turn_id.strip() or not self.evidence_ref.strip() or not self.components:
            raise InvalidRewardEvidence("turn assessment requires identity, evidence and components")
        if len({value.name for value in self.components}) != len(self.components):
            raise InvalidRewardEvidence("turn assessment component names must be unique")

    def require(self, name: str) -> float:
        for component in self.components:
            if component.name == name and component.status == "valid" and component.value is not None:
                return component.value
        raise InvalidRewardEvidence(f"turn {self.turn_id!r} has no valid {name!r} reward")


def native_turn_map(branch: Any) -> tuple[TurnTokenMap, ...]:
    """Map sampled assistant nodes without decoding or retokenizing the policy."""
    mask = tuple(branch.sampled_mask)
    if not mask or any(type(value) is not bool for value in mask) or not any(mask):
        raise InvalidRewardEvidence("turn projection requires original boolean sampled mask")
    first = mask.index(True)
    completion_mask = mask[first:]
    result: list[TurnTokenMap] = []
    offset = 0
    flattened: list[bool] = []
    token_ids: list[int] = []
    for index, node in enumerate(branch.nodes):
        node_mask = tuple(node.mask)
        if len(node_mask) != len(node.token_ids) or any(type(value) is not bool for value in node_mask):
            raise InvalidRewardEvidence("native node tokens and mask are misaligned")
        flattened.extend(node_mask)
        token_ids.extend(node.token_ids)
        if any(node_mask):
            message = node.message
            role = message.get("role") if isinstance(message, Mapping) else message.role
            if role != "assistant" or not node.sampled:
                raise InvalidRewardEvidence("turn credit may address only sampled assistant nodes")
            spans: list[tuple[int, int]] = []
            for local, included in enumerate(node_mask):
                if not included:
                    continue
                position = offset + local - first
                if spans and spans[-1][1] == position:
                    spans[-1] = (spans[-1][0], position + 1)
                else:
                    spans.append((position, position + 1))
            ProcessCredit("valid", TURN_PROJECTION, "native", tuple(spans)).error_mask(completion_mask)
            result.append(TurnTokenMap(f"assistant-{len(result)}", index, tuple(spans)))
        offset += len(node_mask)
    if tuple(flattened) != mask:
        raise InvalidRewardEvidence("native branch and node sampled masks disagree")
    if tuple(token_ids) != tuple(branch.token_ids):
        raise InvalidRewardEvidence("native branch and node token identities disagree")
    return tuple(result)


def validate_turn_assessments(
    assessments: tuple[TurnAssessment, ...],
    turn_ids: tuple[str, ...],
) -> tuple[TurnAssessment, ...]:
    """Require exact coverage, then return native order regardless of judge order."""
    by_id = {assessment.turn_id: assessment for assessment in assessments}
    if not turn_ids or len(set(turn_ids)) != len(turn_ids):
        raise InvalidRewardEvidence("native turn map must have unique non-empty coverage")
    if len(by_id) != len(assessments) or set(by_id) != set(turn_ids):
        raise InvalidRewardEvidence("turn assessments must cover every native turn exactly once")
    return tuple(by_id[turn_id] for turn_id in turn_ids)
