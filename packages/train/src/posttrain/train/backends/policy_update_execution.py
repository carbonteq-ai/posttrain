"""Resolved loss bridge for native graph-retaining trainer execution.

Native trainers retain ownership of backward, accumulation, precision, optimizer
steps and checkpoint transactions. This bridge does not run an optimizer loop.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import torch
from posttrain.environment.verifiers_conditioning import NativeConditioningInput

from ..update_credit import PreparedCredit
from ..update_objectives import ResolvedObjectiveTerm
from ..update_plan import ExecutionPack, ResolvedUpdate
from ..update_records import ActionRef, ConditioningView, InvalidPolicyUpdate
from .policy_update_math import ObjectiveEvaluation, ScoreBundle, evaluate
from .policy_update_scoring import FrozenPopulationScores, score_actions


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
    sampler_correction: Mapping[ActionRef, float] | None,
) -> ObjectiveEvaluation:
    """Score all dependency packs at fixed parameters, then form one true loss.

    Requires a native graph-retention capability; it is not bounded score/replay.
    The caller must not backward/step between these score passes. A sequence ratio
    may depend on unselected actions across packs, so it is evaluated only after
    every dependency is available. There is no per-pack loss renormalization.
    """
    if term.update_digest != update.digest or term.credit_digest != credit.digest:
        raise InvalidPolicyUpdate("native loss received a different resolved update or prepared credit")
    old.validate(
        update.population,
        policy_version=update.population.versions.old_score,
        score_contract=score_contract,
        score_temperature=score_temperature,
    )
    if term.kl_weights:
        if reference is None or update.population.versions.reference is None:
            raise InvalidPolicyUpdate("resolved KL requires frozen reference scores and identity")
        reference.validate(
            update.population,
            policy_version=update.population.versions.reference,
            score_contract=score_contract,
            score_temperature=score_temperature,
        )
    addressed = tuple(action for pack in packs for action in pack.actions)
    if (
        len(set(addressed)) != len(addressed)
        or set(addressed) != set(update.dependencies)
        or any(pack.update_digest != update.digest or pack.index != index for index, pack in enumerate(packs))
    ):
        raise InvalidPolicyUpdate("native score packs must cover each resolved dependency exactly once in order")
    records = {record.action: record for record in update.population.actions}
    contexts = {view.id: view for view in update.population.conditioning}
    for pack in packs:
        expected_contexts = tuple(sorted({records[action].conditioning_id for action in pack.actions}))
        if pack.context_ids != expected_contexts or pack.context_tokens != sum(
            contexts[identity].context_tokens for identity in expected_contexts
        ):
            raise InvalidPolicyUpdate("native pack lost its resolved original conditioning footprint")
    current = {}
    entropies: dict[ActionRef, float] = {}
    for pack in packs:
        current.update(
            score_actions(
                model,
                update.population,
                pack.actions,
                read_input=read_input,
                device=device,
                score_temperature=score_temperature,
                entropies=entropies,
            )
        )
    scores = ScoreBundle(
        current,
        old.values,
        term.parameter_version,
        reference.values if reference is not None else {},
        sampler_correction,
    )
    return replace(evaluate(term, credit, scores), entropies=MappingProxyType(entropies))
