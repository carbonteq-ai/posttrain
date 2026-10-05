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
) -> ObjectiveEvaluation:
    """Score all dependency packs at fixed parameters, then form one true loss.

    Requires a native graph-retention capability; it is not bounded score/replay.
    The caller must not backward/step between these score passes. A sequence ratio
    may depend on unselected actions across packs, so it is evaluated only after
    every dependency is available. There is no per-pack loss renormalization.
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
