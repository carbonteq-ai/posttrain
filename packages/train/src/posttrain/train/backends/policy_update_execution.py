"""Resolved loss bridge for native graph-retaining trainer execution.

Native trainers retain ownership of backward, accumulation, precision, optimizer
steps and checkpoint transactions. This bridge does not run an optimizer loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from posttrain.environment.verifiers_conditioning import NativeConditioningInput

from ..update_credit import PreparedCredit
from ..update_objectives import ResolvedObjectiveTerm
from ..update_plan import ExecutionPack, ResolvedUpdate
from ..update_records import ConditioningView, InvalidPolicyUpdate
from .policy_update_math import ObjectiveEvaluation, ScoreBundle, evaluate
from .policy_update_scoring import FrozenPopulationScores, dense_scores, score_covers


def compute_resolved_loss(
    model: Any,
    update: ResolvedUpdate,
    term: ResolvedObjectiveTerm,
    credit: PreparedCredit,
    packs: tuple[ExecutionPack, ...],
    *,
    old: FrozenPopulationScores,
    reference: FrozenPopulationScores | None,
    read_input: Callable[[ConditioningView], NativeConditioningInput],
    device: torch.device,
    score_temperature: float,
    score_contract: str,
    sampler_correction: np.ndarray | None,
    backward: Callable[[torch.Tensor], None] | None = None,
) -> ObjectiveEvaluation:
    """Score all dependency packs at fixed parameters, then form one true loss.

    Without ``backward`` this requires a native graph-retention capability; it is
    not bounded score/replay. The caller must not step between these score passes.
    A sequence ratio may depend on unselected actions across packs, so it is
    evaluated only after every dependency is available. There is no per-pack loss
    renormalization.

    With ``backward`` (the trainer's own backward, including its loss scaler) and
    every ratio segment inside one pack, each pack's graph is released as soon as
    the pack is scored. The objective is a sum of per-segment and per-position
    terms, so the gradient of the whole loss with respect to one pack's scores is
    the gradient of the loss evaluated with that pack live and every other
    position held at a detached value. Each pack's part goes to ``backward``; the
    returned evaluation is formed from all packs' detached scores and its loss is
    a gradient-free leaf, so a further backward on it adds nothing. Peak memory is
    then one pack's graph instead of the whole update's.
    """
    if term.update_digest != update.digest or term.credit_digest != credit.digest:
        raise InvalidPolicyUpdate("native loss received a different resolved update or prepared credit")
    population = update.population
    old.validate(
        population,
        policy_version=population.versions.old_score,
        score_contract=score_contract,
        score_temperature=score_temperature,
    )
    if term.kl_weight.any():
        if reference is None or population.versions.reference is None:
            raise InvalidPolicyUpdate("resolved KL requires frozen reference scores and identity")
        reference.validate(
            population,
            policy_version=population.versions.reference,
            score_contract=score_contract,
            score_temperature=score_temperature,
        )
    addressed = [view for pack in packs for view in pack.views]
    if (
        len(set(addressed)) != len(addressed)
        or set(addressed) != set(update.views)
        or any(pack.update_digest != update.digest or pack.index != index for index, pack in enumerate(packs))
    ):
        raise InvalidPolicyUpdate("native score packs must cover each resolved dependency exactly once in order")
    if any(
        pack.context_tokens != sum(population.conditioning[view].context_tokens for view in pack.contexts)
        for pack in packs
    ):
        raise InvalidPolicyUpdate("native pack lost its resolved original conditioning footprint")
    entropies = np.full(population.size, np.nan)
    if backward is not None and _segments_within_packs(update, term, packs):
        return _packwise_loss(
            model,
            update,
            term,
            credit,
            packs,
            old=old,
            reference=reference,
            read_input=read_input,
            device=device,
            score_temperature=score_temperature,
            sampler_correction=sampler_correction,
            entropies=entropies,
            backward=backward,
        )
    parts = [
        score_covers(
            model,
            population,
            pack.covers,
            read_input=read_input,
            device=device,
            score_temperature=score_temperature,
            entropies=entropies,
        )
        for pack in packs
    ]
    scored = np.zeros(population.size, dtype=bool)
    for part in parts:
        scored[part.positions] = True
    scores = ScoreBundle(
        dense_scores(population.size, parts, device=device),
        old.values,
        term.parameter_version,
        scored,
        reference.values if reference is not None else None,
        sampler_correction,
    )
    return replace(evaluate(term, credit, scores), entropies=entropies)


def _segments_within_packs(
    update: ResolvedUpdate, term: ResolvedObjectiveTerm, packs: tuple[ExecutionPack, ...]
) -> bool:
    """Whether every ratio segment's positions are scored by a single pack."""
    population = update.population
    view_pack = np.full(len(population.conditioning), -1, dtype=np.int64)
    for index, pack in enumerate(packs):
        view_pack[list(pack.views)] = index
    support = term.ratio_segment >= 0
    segment = term.ratio_segment[support]
    owner = view_pack[population.view_of][support]
    if (owner < 0).any():
        return False
    low = np.full(term.segment_count, len(packs), dtype=np.int64)
    high = np.full(term.segment_count, -1, dtype=np.int64)
    np.minimum.at(low, segment, owner)
    np.maximum.at(high, segment, owner)
    present = high >= 0
    return bool((low[present] == high[present]).all())


def _packwise_loss(
    model: Any,
    update: ResolvedUpdate,
    term: ResolvedObjectiveTerm,
    credit: PreparedCredit,
    packs: tuple[ExecutionPack, ...],
    *,
    old: FrozenPopulationScores,
    reference: FrozenPopulationScores | None,
    read_input: Callable[[ConditioningView], NativeConditioningInput],
    device: torch.device,
    score_temperature: float,
    sampler_correction: np.ndarray | None,
    entropies: np.ndarray,
    backward: Callable[[torch.Tensor], None],
) -> ObjectiveEvaluation:
    population = update.population
    reference_values = reference.values if reference is not None else None
    # Unscored positions hold their old score (ratio one) only so that the
    # objective is defined while one pack is live; they contribute no gradient.
    detached = old.values.to(device=device, dtype=torch.float32).clone()
    every = np.ones(population.size, dtype=bool)
    scored = np.zeros(population.size, dtype=bool)
    applied = False
    for pack in packs:
        part = score_covers(
            model,
            population,
            pack.covers,
            read_input=read_input,
            device=device,
            score_temperature=score_temperature,
            entropies=entropies,
        )
        if scored[part.positions].any():
            raise InvalidPolicyUpdate("score parts must not score one position twice")
        scored[part.positions] = True
        index = torch.as_tensor(part.positions, dtype=torch.long, device=device)
        values = part.values.to(device=device, dtype=torch.float32)
        live = detached.index_put((index,), values)
        bundle = ScoreBundle(live, old.values, term.parameter_version, every, reference_values, sampler_correction)
        loss = evaluate(term, credit, bundle).loss
        if loss.requires_grad:
            backward(loss)
            applied = True
        detached.index_put_((index,), values.detach())
        del part, values, live, bundle, loss
    bundle = ScoreBundle(detached, old.values, term.parameter_version, scored, reference_values, sampler_correction)
    evaluation = evaluate(term, credit, bundle)
    return replace(evaluation, loss=evaluation.loss.detach().requires_grad_(applied), entropies=entropies)
