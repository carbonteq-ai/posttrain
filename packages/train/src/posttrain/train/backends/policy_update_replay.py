"""Bounded score adjoints for native engines that backward each execution pack.

Only closed score-dependent objectives are supported. Native callers qualify
deterministic score replay, parameter continuity and microbatch coverage before
using this path. The gradient carrier is not the reported policy objective.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from ..update_credit import PreparedCredit
from ..update_objectives import ResolvedObjectiveTerm
from ..update_records import ActionRef, InvalidPolicyUpdate
from .policy_update_math import ObjectiveEvaluation, ScoreBundle, evaluate


@dataclass(frozen=True)
class PreparedScoreAdjoints:
    term_digest: str
    parameter_version: str
    evaluation: ObjectiveEvaluation
    current: Mapping[ActionRef, torch.Tensor]
    adjoints: Mapping[ActionRef, torch.Tensor]

    def carrier(self, scores: Mapping[ActionRef, torch.Tensor], *, parameter_version: str,
                absolute_tolerance: float = 0.0, relative_tolerance: float = 0.0) -> torch.Tensor:
        """Return a VJP carrier after verifying each replayed scalar probability."""
        if parameter_version != self.parameter_version:
            raise InvalidPolicyUpdate("score replay parameter version changed before backward")
        if any(isinstance(value, bool) or not math.isfinite(value) or value < 0
               for value in (absolute_tolerance, relative_tolerance)):
            raise InvalidPolicyUpdate("score replay drift tolerances must be explicit finite nonnegative bounds")
        if not scores or not set(scores) <= set(self.current):
            raise InvalidPolicyUpdate("score replay must contain a nonempty subset of resolved dependencies")
        terms = []
        for action, value in scores.items():
            expected = self.current[action]
            if value.ndim != 0 or not value.is_floating_point() or not bool(torch.isfinite(value)):
                raise InvalidPolicyUpdate("score replay requires finite scalar probabilities")
            if not bool(torch.isclose(value.detach().float(), expected.to(value.device).float(),
                                      atol=absolute_tolerance, rtol=relative_tolerance)):
                raise InvalidPolicyUpdate("score replay drift exceeds its qualified tolerance")
            terms.append(value * self.adjoints[action].to(value.device))
        return torch.stack(terms).sum()


def prepare_score_adjoints(
    term: ResolvedObjectiveTerm, credit: PreparedCredit, scores: ScoreBundle,
) -> PreparedScoreAdjoints:
    """Evaluate the true complete objective with independent current-score leaves."""
    if any(value.ndim != 0 or not value.is_floating_point() or not bool(torch.isfinite(value))
           for value in scores.current.values()):
        raise InvalidPolicyUpdate("score adjoints require finite scalar current probabilities")
    current = {action: (value.detach().float() if value.dtype in (torch.bfloat16, torch.float16) else value.detach())
               .clone().requires_grad_()
               for action, value in scores.current.items()}
    if not current:
        raise InvalidPolicyUpdate("score adjoints require resolved current probabilities")
    with torch.enable_grad():
        evaluation = evaluate(term, credit, ScoreBundle(current, scores.old, scores.parameter_version,
                                                       scores.reference, scores.sampler_correction))
        if not evaluation.loss.requires_grad:
            raise InvalidPolicyUpdate("empty objective has no score-only replay derivative")
        gradients = torch.autograd.grad(evaluation.loss, tuple(current.values()), allow_unused=True)
    values = MappingProxyType({action: value.detach() for action, value in current.items()})
    adjoints = MappingProxyType({action: gradient.detach() if gradient is not None else torch.zeros_like(current[action])
                                for action, gradient in zip(current, gradients, strict=True)})
    if any(not bool(torch.isfinite(value)) for value in adjoints.values()):
        raise InvalidPolicyUpdate("resolved score adjoints must be finite before native backward")
    detached_evaluation = ObjectiveEvaluation(
        evaluation.loss.detach(), evaluation.policy_loss.detach(), evaluation.kl_loss.detach(),
        MappingProxyType({action: value.detach() for action, value in evaluation.ratios.items()}),
        evaluation.clipped_actions, evaluation.term_digest, evaluation.parameter_version,
    )
    return PreparedScoreAdjoints(term.digest, scores.parameter_version, detached_evaluation, values, adjoints)
