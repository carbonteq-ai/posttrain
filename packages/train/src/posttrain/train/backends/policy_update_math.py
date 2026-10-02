"""Private tensor evaluation of resolved policy terms, shared by native adapters.

No model/runtime is constructed here. Graph retention versus qualified replay is
an execution decision; the declared adjoints are independent of pack layout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from ..update_credit import PreparedCredit
from ..update_objectives import ResolvedObjectiveTerm, objective_definition
from ..update_records import ActionRef, InvalidPolicyUpdate


@dataclass(frozen=True)
class ScoreBundle:
    current: Mapping[ActionRef, torch.Tensor]
    old: Mapping[ActionRef, torch.Tensor]
    parameter_version: str
    reference: Mapping[ActionRef, torch.Tensor] = field(default_factory=dict)
    sampler_correction: Mapping[ActionRef, float] | None = None


@dataclass(frozen=True)
class ObjectiveEvaluation:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    ratios: Mapping[ActionRef, torch.Tensor]
    clipped_actions: tuple[ActionRef, ...]
    term_digest: str
    parameter_version: str


def _score(values: Mapping[ActionRef, torch.Tensor], action: ActionRef, role: str, *, detached: bool = False) -> torch.Tensor:
    value = values.get(action)
    if value is None or value.ndim != 0 or not value.is_floating_point() or not bool(torch.isfinite(value)):
        raise InvalidPolicyUpdate(f"missing or non-finite scalar {role} score for {action}")
    value = value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    return value.detach() if detached else value


def _sampled_k3(x: torch.Tensor) -> torch.Tensor:
    # Tenth-order series avoids expm1 backward cancellation near zero. Evaluate
    # only the selected branch so an unused exponential cannot poison backward.
    if bool(x.abs() <= 0.25):
        polynomial = torch.zeros_like(x)
        for power in range(10, 1, -1):
            polynomial = polynomial * x + 1 / math.factorial(power)
        return x.square() * polynomial
    return torch.expm1(x) - x


def evaluate(term: ResolvedObjectiveTerm, credit: PreparedCredit, scores: ScoreBundle) -> ObjectiveEvaluation:
    if credit.digest != term.credit_digest or not scores.parameter_version.strip():
        raise InvalidPolicyUpdate("score/credit identity missing or changed")
    if scores.parameter_version != term.parameter_version:
        raise InvalidPolicyUpdate("current scores belong to a different parameter version")
    definition = objective_definition(term.spec.definition_id)
    advantages = {value.action: value.advantage for value in credit.values}
    ratios: dict[ActionRef, torch.Tensor] = {}
    for support in term.ratio_support:
        deltas = tuple(_score(scores.current, action, "current") - _score(scores.old, action, "old", detached=True)
                       for action in support)
        log_ratio = deltas[0] if definition.ratio == "token" else torch.stack(deltas).mean()
        ratio = log_ratio.exp()
        if not bool(torch.isfinite(ratio)):
            raise InvalidPolicyUpdate("non-finite importance ratio; no implicit cap is qualified")
        for action in support:
            ratios[action] = ratio
    policy_terms: list[torch.Tensor] = []
    clipped_actions: list[ActionRef] = []
    for weighted in term.policy_weights:
        action = weighted.action
        current = _score(scores.current, action, "current")
        ratio = ratios[action]
        if definition.derivative == "token-local":
            ratio = (ratio.log().detach() + current - current.detach()).exp()
        if action not in advantages:
            raise InvalidPolicyUpdate("selected action lacks prepared credit")
        advantage = advantages[action]
        if definition.policy_loss == "cispo-upper":
            assert term.spec.cispo_max_weight is not None
            clipped = ratio.clamp(max=term.spec.cispo_max_weight)
            loss = -clipped.detach() * advantage * current
            active_clip = bool(ratio > term.spec.cispo_max_weight)
        else:
            plain = -ratio * advantage
            clipped = -ratio.clamp(1 - term.spec.clip_low, 1 + term.spec.clip_high) * advantage
            loss = torch.maximum(plain, clipped)
            active_clip = bool(clipped > plain)
        if active_clip:
            clipped_actions.append(action)
        correction = 1.0
        if scores.sampler_correction is not None:
            correction_value = scores.sampler_correction.get(action)
            if correction_value is None or isinstance(correction_value, bool) or (
                not math.isfinite(correction_value) or correction_value < 0
            ):
                raise InvalidPolicyUpdate("sampler correction requires detached finite nonnegative action weights")
            correction = correction_value
        policy_terms.append(loss * (weighted.weight * correction))
    kl_terms = [
        _sampled_k3(_score(scores.reference, item.action, "reference", detached=True)
                    - _score(scores.current, item.action, "current")) * (term.spec.beta * item.weight)
        for item in term.kl_weights
    ]
    # Empty terms retain a zero carrier when another term has a live device graph.
    anchor = next(iter(policy_terms or kl_terms), None)
    zero = anchor * 0 if anchor is not None else torch.tensor(0.0)
    policy_loss = torch.stack(policy_terms).sum() if policy_terms else zero
    kl_loss = torch.stack(kl_terms).sum() if kl_terms else zero
    loss = policy_loss + kl_loss
    if not bool(torch.isfinite(loss)):
        raise InvalidPolicyUpdate("non-finite resolved objective")
    return ObjectiveEvaluation(loss, policy_loss, kl_loss, ratios, tuple(clipped_actions), term.digest,
                               scores.parameter_version)
