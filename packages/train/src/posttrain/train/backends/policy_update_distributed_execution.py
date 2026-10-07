"""Matched-round graph-retaining distributed scoring; native trainers own steps.

This is a private dense causal-text path. DDP, FSDP and trainer qualification are
separate gates; a completed CPU DDP check cannot authorize an FSDP model path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from posttrain.environment.verifiers_conditioning import NativeConditioningInput

from ..update_credit import PreparedCredit
from ..update_distribution import DistributedScorePlan, resolve_score_rounds
from ..update_objectives import ResolvedObjectiveTerm
from ..update_plan import ExecutionCapabilities, PolicyExecutionBudget, ResolvedUpdate, plan_packs
from ..update_records import ConditioningView, InvalidPolicyUpdate
from .policy_update_distribution import all_ranks_finite, distributed_score_carrier, gather_current_scores
from .policy_update_math import ObjectiveEvaluation, ScoreBundle
from .policy_update_replay import prepare_score_adjoints
from .policy_update_scoring import FrozenPopulationScores, PositionScores, score_views


@dataclass(frozen=True)
class DistributedGraphEvaluation:
    evaluation: ObjectiveEvaluation
    carrier: torch.Tensor
    ownership_digest: str
    forward_rounds: int
    zero_weight_forwards: int


def compute_distributed_graph_loss(
    model: Any,
    update: ResolvedUpdate,
    term: ResolvedObjectiveTerm,
    credit: PreparedCredit,
    plan: DistributedScorePlan,
    *,
    execution: PolicyExecutionBudget,
    capabilities: ExecutionCapabilities,
    old: FrozenPopulationScores,
    reference: FrozenPopulationScores | None,
    read_input: Callable[[ConditioningView], NativeConditioningInput],
    device: torch.device,
    score_temperature: float,
    score_contract: str,
    sampler_correction: np.ndarray | None,
    group: Any = None,
) -> DistributedGraphEvaluation:
    """Retain owned graphs with matched native forwards, then one true backward.

    Native inputs are authenticated/materialized before model collectives. Global
    scores and old/reference/correction identities agree across ranks. Reported loss
    is the complete objective; only carrier goes to native backward. No optimizer,
    scaler, checkpoint, model wrapping or population collection is constructed here.
    """
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    population = update.population
    valid, rounds, inputs, identity = True, (), {}, None
    preflight_error = None
    try:
        if size != len(plan.partitions) or term.credit_digest != credit.digest:
            raise InvalidPolicyUpdate("distributed graph rank count or credit changed")
        rounds = resolve_score_rounds(update, term, plan)
        plan_packs(update, execution, capabilities)
        old.validate(
            update.population,
            policy_version=update.population.versions.old_score,
            score_contract=score_contract,
            score_temperature=score_temperature,
        )
        if term.kl_weight.any():
            if reference is None or update.population.versions.reference is None:
                raise InvalidPolicyUpdate("distributed graph KL lacks frozen reference evidence")
            reference.validate(
                update.population,
                policy_version=update.population.versions.reference,
                score_contract=score_contract,
                score_temperature=score_temperature,
            )
        for index in {work.view for row in rounds for work in row}:
            view = population.conditioning[index]
            source = read_input(view)
            if (
                source.record.context_contract != "causal-text@1"
                or source.record.input_digest != view.digest
                or len(source.token_ids) != view.context_tokens
                or source.record.context_tokens != view.context_tokens
            ):
                raise InvalidPolicyUpdate("distributed graph native input differs from its original view")
            inputs[view.id] = source

        def frozen_digest(values: torch.Tensor) -> str:
            return hashlib.sha256(values.detach().float().cpu().numpy().tobytes()).hexdigest()

        correction = None if sampler_correction is None else np.asarray(sampler_correction, dtype=np.float64).tolist()
        identity = (
            plan.digest,
            term.digest,
            credit.digest,
            frozen_digest(old.values),
            frozen_digest(reference.values) if term.kl_weight.any() and reference is not None else None,
            json.dumps(correction, sort_keys=True, allow_nan=False),
            score_contract,
            score_temperature,
            execution,
            capabilities,
        )
    except Exception as error:
        # All ranks join this preflight even when their native artifact/input
        # validation fails. Native communication failures are not intercepted.
        valid = False
        preflight_error = error
    reports = [None] * size
    dist.all_gather_object(reports, (identity, valid), group=group)
    if not valid or any(report != (identity, True) for report in reports):
        raise InvalidPolicyUpdate(
            "distributed graph preflight differs or rejects native evidence across ranks"
        ) from preflight_error
    current: list[PositionScores] = []
    anchors = []
    for row in rounds:
        work = row[rank]
        try:
            scored = score_views(
                model,
                population,
                (work.view,),
                read_input=lambda view: inputs[view.id],
                device=device,
                score_temperature=score_temperature,
            )
            valid_scores = bool(torch.isfinite(scored.values).all())
        except InvalidPolicyUpdate:
            scored, valid_scores = None, False
        if not all_ranks_finite(valid_scores, device=device, group=group):
            raise InvalidPolicyUpdate("distributed graph scores are nonfinite on a rank")
        assert scored is not None
        if work.contributes:
            current.append(scored)
        else:
            anchors.append(scored.values.sum())
    local = (
        PositionScores(
            np.concatenate([part.positions for part in current]), torch.cat([part.values for part in current])
        )
        if current
        else PositionScores(np.zeros(0, dtype=np.int64), torch.zeros(0, device=device))
    )
    global_scores, scored_mask = gather_current_scores(
        plan, population, local, rank=rank, parameter_version=term.parameter_version, device=device, group=group
    )
    prepared = prepare_score_adjoints(
        term,
        credit,
        ScoreBundle(
            global_scores,
            old.values,
            term.parameter_version,
            scored_mask,
            reference.values if reference is not None else None,
            sampler_correction,
        ),
    )
    carrier = distributed_score_carrier(
        plan,
        population,
        prepared,
        local if current else None,
        rank=rank,
        parameter_version=term.parameter_version,
        empty_anchor=torch.stack(anchors).sum() if anchors and not current else None,
    )
    if current:
        for anchor in anchors:
            carrier = carrier + anchor * 0
    return DistributedGraphEvaluation(prepared.evaluation, carrier, plan.digest, len(rounds), len(anchors))
