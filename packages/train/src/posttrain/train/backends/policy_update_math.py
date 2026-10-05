"""Private tensor evaluation of resolved policy terms, shared by native adapters.

No model/runtime is constructed here. Graph retention versus qualified replay is
an execution decision; the declared adjoints are independent of pack layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ..update_credit import PreparedCredit
from ..update_objectives import ResolvedObjectiveTerm, objective_definition
from ..update_records import InvalidPolicyUpdate


@dataclass(frozen=True)
class ScoreBundle:
    """Per-position scores: ``current`` (live graph), frozen ``old`` and optional ``reference``.

    Every tensor has one float value per population position. ``scored`` marks
    the positions ``current`` actually holds scores for; any other position is
    zero and must not be needed by the objective. ``sampler_correction`` holds
    detached per-position weights.
    """

    current: torch.Tensor
    old: torch.Tensor
    parameter_version: str
    scored: np.ndarray
    reference: torch.Tensor | None = None
    sampler_correction: np.ndarray | None = None


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """The objective's value and its per-position evidence.

    ``ratios`` holds each ratio-supported position's importance ratio (detached,
    on the host) and ``clipped`` marks the selected policy positions whose
    clipped surrogate was active; both have one entry per population position.
    """

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    ratios: torch.Tensor
    clipped: np.ndarray
    term_digest: str
    parameter_version: str
    # Detached current-policy entropy per position (NaN where not measured).
    # Observation only; it never enters the loss.
    entropies: np.ndarray | None = field(default=None)


def _sampled_k3(x: torch.Tensor) -> torch.Tensor:
    """exp(x) - 1 - x elementwise, by a tenth-order series near zero.

    The series avoids expm1 backward cancellation near zero. Each branch only
    sees inputs from its own domain (zero elsewhere), so the unused branch can
    neither overflow nor poison the gradient through ``torch.where``.
    """
    small = x.abs() <= 0.25
    near = torch.where(small, x, torch.zeros_like(x))
    far = torch.where(small, torch.zeros_like(x), x)
    polynomial = torch.zeros_like(near)
    for power in range(10, 1, -1):
        polynomial = polynomial * near + 1 / math.factorial(power)
    return torch.where(small, near.square() * polynomial, torch.expm1(far) - far)


def evaluate(term: ResolvedObjectiveTerm, credit: PreparedCredit, scores: ScoreBundle) -> ObjectiveEvaluation:
    if credit.digest != term.credit_digest or not scores.parameter_version.strip():
        raise InvalidPolicyUpdate("score/credit identity missing or changed")
    if scores.parameter_version != term.parameter_version:
        raise InvalidPolicyUpdate("current scores belong to a different parameter version")
    definition = objective_definition(term.spec.definition_id)
    size = term.policy_weight.size
    current, device = scores.current, scores.current.device
    if current.shape != (size,) or scores.old.shape != (size,) or scores.scored.shape != (size,):
        raise InvalidPolicyUpdate("scores must align with the objective's population positions")
    current = current.float() if current.dtype in (torch.float16, torch.bfloat16) else current
    old = scores.old.to(device=device, dtype=current.dtype).detach()
    support = term.ratio_segment >= 0
    policy = term.policy_weight > 0
    kl = term.kl_weight > 0
    needed = support | policy | kl
    if not scores.scored[needed].all():
        raise InvalidPolicyUpdate("missing current score for a resolved objective dependency")
    index = torch.as_tensor(np.flatnonzero(support), dtype=torch.long, device=device)
    delta = current[index] - old[index]
    if not bool(torch.isfinite(delta).all()):
        raise InvalidPolicyUpdate("missing or non-finite scalar current or old score")
    if definition.ratio == "token":
        log_ratio = delta
    else:
        segment = torch.as_tensor(term.ratio_segment[support], dtype=torch.long, device=device)
        sums = torch.zeros(term.segment_count, dtype=delta.dtype, device=device).index_add(0, segment, delta)
        counts = torch.bincount(segment, minlength=term.segment_count).to(delta.dtype)
        log_ratio = (sums / counts)[segment]
    supported_ratio = log_ratio.exp()
    if not bool(torch.isfinite(supported_ratio).all()):
        raise InvalidPolicyUpdate("non-finite importance ratio; no implicit cap is qualified")
    support_positions = np.flatnonzero(support)
    policy_positions = np.flatnonzero(policy)
    policy_index = torch.as_tensor(policy_positions, dtype=torch.long, device=device)
    # Every selected policy position has ratio support (ResolvedObjectiveTerm checks it).
    ratio = supported_ratio[
        torch.as_tensor(np.searchsorted(support_positions, policy_positions), dtype=torch.long, device=device)
    ]
    # Per-position ratios are evidence only: kept detached on the host, so no
    # population-sized tensor outlives the update on the device.
    ratios = torch.zeros(size, dtype=torch.float32).index_put(
        (torch.as_tensor(support_positions, dtype=torch.long),), supported_ratio.detach().float().cpu()
    )
    selected_current = current[policy_index]
    if definition.derivative == "token-local":
        ratio = (ratio.log().detach() + selected_current - selected_current.detach()).exp()
    advantage = torch.as_tensor(credit.advantages[policy_positions], dtype=ratio.dtype, device=device)
    if definition.policy_loss == "cispo-upper":
        assert term.spec.cispo_max_weight is not None
        clipped_weight = ratio.clamp(max=term.spec.cispo_max_weight)
        losses = -clipped_weight.detach() * advantage * selected_current
        active = ratio > term.spec.cispo_max_weight
    else:
        plain = -ratio * advantage
        clipped_term = -ratio.clamp(1 - term.spec.clip_low, 1 + term.spec.clip_high) * advantage
        losses = torch.maximum(plain, clipped_term)
        active = clipped_term > plain
    weights = term.policy_weight[policy_positions]
    if scores.sampler_correction is not None:
        correction = np.asarray(scores.sampler_correction, dtype=np.float64)
        if correction.shape != (size,) or not np.isfinite(correction).all() or (correction < 0).any():
            raise InvalidPolicyUpdate("sampler correction requires detached finite nonnegative action weights")
        weights = weights * correction[policy_positions]
    policy_loss = (losses * torch.as_tensor(weights, dtype=losses.dtype, device=device)).sum()
    clipped = np.zeros(size, dtype=bool)
    clipped[policy_positions] = active.detach().cpu().numpy()
    kl_positions = np.flatnonzero(kl)
    if kl_positions.size:
        if scores.reference is None or scores.reference.shape != (size,):
            raise InvalidPolicyUpdate("missing or non-finite scalar reference score")
        kl_index = torch.as_tensor(kl_positions, dtype=torch.long, device=device)
        reference = scores.reference.to(device=device, dtype=current.dtype).detach()[kl_index]
        x = reference - current[kl_index]
        if not bool(torch.isfinite(x).all()):
            raise InvalidPolicyUpdate("missing or non-finite scalar reference score")
        kl_weight = torch.as_tensor(term.spec.beta * term.kl_weight[kl_positions], dtype=x.dtype, device=device)
        kl_loss = (_sampled_k3(x) * kl_weight).sum()
    else:
        kl_loss = policy_loss * 0 if policy_positions.size else torch.zeros((), device=device)
    if not policy_positions.size:
        policy_loss = kl_loss * 0
    loss = policy_loss + kl_loss
    if not bool(torch.isfinite(loss)):
        raise InvalidPolicyUpdate("non-finite resolved objective")
    return ObjectiveEvaluation(loss, policy_loss, kl_loss, ratios, clipped, term.digest, scores.parameter_version)
