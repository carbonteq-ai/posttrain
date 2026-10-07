"""Bounded score adjoints for native engines that backward each execution pack.

Only closed score-dependent objectives are supported. Native callers qualify
deterministic score replay, parameter continuity and microbatch coverage before
using this path. The gradient carrier is not the reported policy objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..update_credit import PreparedCredit
from ..update_objectives import ResolvedObjectiveTerm
from ..update_records import InvalidPolicyUpdate
from .policy_update_math import ObjectiveEvaluation, ScoreBundle, evaluate
from .policy_update_scoring import PositionScores


@dataclass(frozen=True)
class PreparedScoreAdjoints:
    """Detached current scores and the objective's derivative per population position."""

    term_digest: str
    parameter_version: str
    evaluation: ObjectiveEvaluation
    current: torch.Tensor
    adjoints: torch.Tensor
    scored: np.ndarray

    def carrier(
        self,
        scores: PositionScores,
        *,
        parameter_version: str,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 0.0,
    ) -> torch.Tensor:
        """Return a VJP carrier after verifying each replayed scalar probability."""
        if parameter_version != self.parameter_version:
            raise InvalidPolicyUpdate("score replay parameter version changed before backward")
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0
            for value in (absolute_tolerance, relative_tolerance)
        ):
            raise InvalidPolicyUpdate("score replay drift tolerances must be explicit finite nonnegative bounds")
        if not scores.positions.size or not self.scored[scores.positions].all():
            raise InvalidPolicyUpdate("score replay must contain a nonempty subset of resolved dependencies")
        values = scores.values
        if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
            raise InvalidPolicyUpdate("score replay requires finite scalar probabilities")
        index = torch.as_tensor(scores.positions, dtype=torch.long, device=self.current.device)
        expected = self.current[index].to(values.device)
        if not bool(
            torch.isclose(
                values.detach().float(), expected.float(), atol=absolute_tolerance, rtol=relative_tolerance
            ).all()
        ):
            raise InvalidPolicyUpdate("score replay drift exceeds its qualified tolerance")
        return (values * self.adjoints[index].to(values.device)).sum()


def prepare_score_adjoints(
    term: ResolvedObjectiveTerm,
    credit: PreparedCredit,
    scores: ScoreBundle,
) -> PreparedScoreAdjoints:
    """Evaluate the true complete objective with an independent current-score leaf."""
    if not scores.scored.any():
        raise InvalidPolicyUpdate("score adjoints require resolved current probabilities")
    detached = scores.current.detach()
    detached = detached.float() if detached.dtype in (torch.bfloat16, torch.float16) else detached
    index = torch.as_tensor(np.flatnonzero(scores.scored), dtype=torch.long, device=detached.device)
    if not bool(torch.isfinite(detached[index]).all()):
        raise InvalidPolicyUpdate("score adjoints require finite scalar current probabilities")
    current = detached.clone().requires_grad_()
    with torch.enable_grad():
        evaluation = evaluate(
            term,
            credit,
            ScoreBundle(
                current,
                scores.old,
                scores.parameter_version,
                scores.scored,
                scores.reference,
                scores.sampler_correction,
            ),
        )
        if not evaluation.loss.requires_grad:
            raise InvalidPolicyUpdate("empty objective has no score-only replay derivative")
        (gradient,) = torch.autograd.grad(evaluation.loss, (current,), allow_unused=True)
    adjoints = torch.zeros_like(current) if gradient is None else gradient.detach()
    if not bool(torch.isfinite(adjoints).all()):
        raise InvalidPolicyUpdate("resolved score adjoints must be finite before native backward")
    detached_evaluation = ObjectiveEvaluation(
        evaluation.loss.detach(),
        evaluation.policy_loss.detach(),
        evaluation.kl_loss.detach(),
        evaluation.ratios.detach(),
        evaluation.clipped,
        evaluation.term_digest,
        evaluation.parameter_version,
    )
    return PreparedScoreAdjoints(
        term.digest, scores.parameter_version, detached_evaluation, current.detach(), adjoints, scores.scored
    )
