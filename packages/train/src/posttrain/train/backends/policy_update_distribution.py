"""Private score collectives for globally resolved objectives and DDP averaging.

No native optimizer lifecycle is replaced here. This seam must be qualified by
each adapter before that adapter admits distributed execution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ..update_distribution import DistributedScorePlan
from ..update_records import InvalidPolicyUpdate, PopulationSnapshot
from .policy_update_replay import PreparedScoreAdjoints
from .policy_update_scoring import PositionScores


def gather_current_scores(
    plan: DistributedScorePlan,
    population: PopulationSnapshot,
    local: PositionScores,
    *,
    rank: int,
    parameter_version: str,
    device: torch.device,
    group: Any = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Gather detached FP32 scores after coordinated identity/coverage checks.

    Returns the population-length scores and the mask of scored positions. Each
    rank joins the control collective even if its local values are malformed; all
    ranks then reject before the tensor collective. This avoids one rank raising
    while its peers wait for scores. Communication failures remain native errors.
    """
    size = dist.get_world_size(group)
    valid = isinstance(parameter_version, str) and bool(parameter_version.strip()) and size == len(plan.partitions)
    owned = np.flatnonzero(population.views_mask(plan.partitions[rank].views)) if valid else None
    if valid:
        valid = owned is not None and np.array_equal(np.sort(local.positions), owned)
    if valid:
        values = local.values
        valid = values.dtype in (torch.float16, torch.bfloat16, torch.float32) and bool(torch.isfinite(values).all())
    reports = [None] * size
    dist.all_gather_object(reports, (plan.digest, parameter_version, valid), group=group)
    if any(report != (plan.digest, parameter_version, True) for report in reports):
        raise InvalidPolicyUpdate("distributed score identities, ownership or finite coverage differ across ranks")
    scores = torch.zeros(population.size, dtype=torch.float32, device=device)
    index = torch.as_tensor(local.positions, dtype=torch.long, device=device)
    scores[index] = local.values.detach().float().to(device)
    # Ownership is exclusive: this sum gathers scores, it does not average them.
    dist.all_reduce(scores, op=dist.ReduceOp.SUM, group=group)
    return scores, population.views_mask(plan.views)


def distributed_score_carrier(
    plan: DistributedScorePlan,
    population: PopulationSnapshot,
    prepared: PreparedScoreAdjoints,
    scores: PositionScores | None,
    *,
    rank: int,
    parameter_version: str,
    empty_anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route global score derivatives to owned scores under native DDP averaging.

    Nonempty calls may replay one owned pack. Empty ranks supply a real native model
    graph anchor, multiplied by zero, so backward joins the native collectives. The
    adapter still owns exact replay coverage and matching collective call schedules.
    """
    if (
        type(rank) is not int
        or not 0 <= rank < len(plan.partitions)
        or prepared.term_digest != plan.term_digest
        or prepared.parameter_version != parameter_version
        or not np.array_equal(prepared.scored, population.views_mask(plan.views))
    ):
        raise InvalidPolicyUpdate("distributed replay identity or rank changed")
    owned = population.views_mask(plan.partitions[rank].views)
    if scores is not None and scores.positions.size:
        if empty_anchor is not None or not owned[scores.positions].all():
            raise InvalidPolicyUpdate("distributed replay exceeds owned original scores")
        return prepared.carrier(scores, parameter_version=parameter_version) * len(plan.partitions)
    if (
        owned.any()
        or empty_anchor is None
        or empty_anchor.ndim != 0
        or not empty_anchor.requires_grad
        or not bool(torch.isfinite(empty_anchor))
    ):
        raise InvalidPolicyUpdate("empty rank requires a finite native graph anchor and no owned scores")
    return empty_anchor * 0


def all_ranks_finite(local_finite: bool, *, device: torch.device, group: Any = None) -> bool:
    """A shared finite decision; native adapters still own scaler/step transactions."""
    decision = torch.tensor(int(local_finite is True), dtype=torch.int32, device=device)
    dist.all_reduce(decision, op=dist.ReduceOp.MIN, group=group)
    return bool(decision.item())
