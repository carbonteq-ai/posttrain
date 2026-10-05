"""Private causal-logit projection shared by qualified native score adapters."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from posttrain.environment.verifiers_conditioning import NativeConditioningInput

from ..update_records import ActionRef, ConditioningView, InvalidPolicyUpdate, PopulationSnapshot, require_identity


def sampled_logprobs(
    logits: torch.Tensor,
    inputs: NativeConditioningInput,
    *,
    sampled_indices: tuple[int, ...],
    score_temperature: float,
    entropies: dict[int, float] | None = None,
) -> Mapping[int, torch.Tensor]:
    """Score selected original actions at their preceding causal positions.

    Inputs must come from materialize_native_conditioning over authenticated
    retained evidence. The native adapter owns model/attention qualification,
    parameter identity and score freezing. This projection does not establish
    those guarantees or score observations merely because they are in context.
    The score temperature is required explicitly: native trainers use tempered
    policy probabilities, which differ from raw model probabilities. This is not
    reconstruction of a top-p/top-k filtered sampler distribution.

    When `entropies` is supplied, the detached entropy of the same tempered
    distribution at each selected causal position is added to it. This is
    observation only: it never enters the returned scores or their graph.
    """
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(inputs.token_ids):
        raise InvalidPolicyUpdate("native logits must align with one complete original conditioning view")
    if not logits.is_floating_point() or logits.shape[2] < 1:
        raise InvalidPolicyUpdate("native score logits require a floating vocabulary axis")
    if isinstance(score_temperature, bool) or not math.isfinite(score_temperature) or score_temperature <= 0:
        raise InvalidPolicyUpdate("native score temperature must be finite and positive")
    if len(set(sampled_indices)) != len(sampled_indices):
        raise InvalidPolicyUpdate("native score selection duplicates an original action")
    positions = dict(inputs.action_positions)
    if any(type(index) is not int or index not in positions for index in sampled_indices):
        raise InvalidPolicyUpdate("native score selection includes an ineligible node token")
    if not sampled_indices:
        return {}
    action_positions = [positions[index] for index in sampled_indices]
    if any(position < 1 or position >= len(inputs.token_ids) for position in action_positions):
        raise InvalidPolicyUpdate("native action lacks a preceding causal score position")
    targets = [inputs.token_ids[position] for position in action_positions]
    if any(target < 0 or target >= logits.shape[2] for target in targets):
        raise InvalidPolicyUpdate("native sampled token is outside model vocabulary")
    selected = logits[0, [position - 1 for position in action_positions]]
    # Normalize only needed positions in FP32 for supported half precision.
    if selected.dtype in (torch.bfloat16, torch.float16):
        selected = selected.float()
    if not bool(torch.isfinite(selected).all()):
        raise InvalidPolicyUpdate("non-finite native logits at a required causal position")
    log_probabilities = (selected / score_temperature).log_softmax(dim=-1)
    values = log_probabilities.gather(
        1,
        torch.tensor(targets, dtype=torch.long, device=logits.device).unsqueeze(1),
    ).squeeze(1)
    if entropies is not None:
        entropies.update(zip(sampled_indices, _entropies(log_probabilities), strict=True))
    return dict(zip(sampled_indices, values.unbind(), strict=True))


_ENTROPY_ROWS = 256


def _entropies(log_probabilities: torch.Tensor) -> list[float]:
    """Detached per-row entropy, in row chunks to bound the vocabulary-sized temporaries."""
    result: list[float] = []
    with torch.no_grad():
        detached = log_probabilities.detach()
        for start in range(0, detached.shape[0], _ENTROPY_ROWS):
            chunk = detached[start : start + _ENTROPY_ROWS]
            result.extend(float(value) for value in (-(chunk.exp() * chunk).sum(dim=-1)).tolist())
    return result


def score_actions(
    model: Any,
    snapshot: PopulationSnapshot,
    actions: tuple[ActionRef, ...],
    *,
    read_input: Callable[[ConditioningView], NativeConditioningInput],
    device: torch.device,
    score_temperature: float,
    entropies: dict[ActionRef, float] | None = None,
) -> Mapping[ActionRef, torch.Tensor]:
    """Retain full-context model graphs for exactly the requested score support.

    The native caller authenticates retained bytes in read_input and qualifies
    the model's causal-text@1 attention/position behavior. No detached KV cache,
    retokenization, padding or generation call is used here. Calling this across
    planned packs retains graphs; native backward must occur before any update.
    """
    records = {record.action: record for record in snapshot.actions}
    if len(set(actions)) != len(actions) or any(action not in records for action in actions):
        raise InvalidPolicyUpdate("model score support must contain unique admitted original actions")
    contexts = {view.id: view for view in snapshot.conditioning}
    groups: dict[str, list[ActionRef]] = {}
    for action in actions:
        groups.setdefault(records[action].conditioning_id, []).append(action)
    scores: dict[ActionRef, torch.Tensor] = {}
    for context_id, members in groups.items():
        view = contexts[context_id]
        inputs = read_input(view)
        record = inputs.record
        if (
            record.context_contract != "causal-text@1"
            or record.input_digest != view.digest
            or (len(inputs.token_ids) != view.context_tokens or record.context_tokens != view.context_tokens)
        ):
            raise InvalidPolicyUpdate("materialized model input differs from the frozen conditioning view")
        if len({action.turn_id for action in members}) != 1 or len({action.token_index for action in members}) != len(
            members
        ):
            raise InvalidPolicyUpdate("one native conditioning view must identify one original sampled turn")
        token_ids = torch.tensor([inputs.token_ids], dtype=torch.long, device=device)
        output = model(
            input_ids=token_ids,
            attention_mask=torch.ones_like(token_ids),
            position_ids=torch.arange(token_ids.shape[1], device=device).unsqueeze(0),
            use_cache=False,
        )
        local_entropies: dict[int, float] | None = {} if entropies is not None else None
        values = sampled_logprobs(
            output.logits,
            inputs,
            sampled_indices=tuple(action.token_index for action in members),
            score_temperature=score_temperature,
            entropies=local_entropies,
        )
        scores.update((action, values[action.token_index]) for action in members)
        if entropies is not None and local_entropies is not None:
            entropies.update((action, local_entropies[action.token_index]) for action in members)
    return scores


@dataclass(frozen=True)
class FrozenPopulationScores:
    """Detached scores bound to one admitted population and score contract."""

    population_digest: str
    policy_version: str
    score_contract: str
    score_temperature: float
    values: Mapping[ActionRef, torch.Tensor]

    def validate(
        self, snapshot: PopulationSnapshot, *, policy_version: str, score_contract: str, score_temperature: float
    ) -> None:
        if (self.population_digest, self.policy_version, self.score_contract, self.score_temperature) != (
            snapshot.digest,
            policy_version,
            score_contract,
            score_temperature,
        ):
            raise InvalidPolicyUpdate("frozen policy scores belong to different evidence, policy or score contract")
        if set(self.values) != {record.action for record in snapshot.actions}:
            raise InvalidPolicyUpdate("frozen policy scores must cover the complete admitted population")
        if any(
            value.ndim != 0 or value.requires_grad or not bool(torch.isfinite(value)) for value in self.values.values()
        ):
            raise InvalidPolicyUpdate("frozen policy scores must remain detached finite scalars")


def freeze_population_scores(
    model: Any,
    snapshot: PopulationSnapshot,
    *,
    read_input: Callable[[ConditioningView], NativeConditioningInput],
    device: torch.device,
    policy_version: str,
    score_contract: str,
    score_temperature: float,
) -> FrozenPopulationScores:
    """Prepare old/reference probabilities before any update on this population."""
    require_identity(policy_version, score_contract)
    with torch.no_grad():
        values = score_actions(
            model,
            snapshot,
            tuple(record.action for record in snapshot.actions),
            read_input=read_input,
            device=device,
            score_temperature=score_temperature,
        )
    frozen = FrozenPopulationScores(
        snapshot.digest,
        policy_version,
        score_contract,
        score_temperature,
        MappingProxyType({action: value.detach().clone() for action, value in values.items()}),
    )
    frozen.validate(
        snapshot, policy_version=policy_version, score_contract=score_contract, score_temperature=score_temperature
    )
    return frozen
