"""Private causal-logit projection shared by qualified native score adapters.

Scores are carried as tensors aligned with population positions (see
``update_records``): one model forward per conditioning view yields that view's
sampled-token log-probabilities as one tensor, and a population's scores are
one tensor of length ``snapshot.size``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from posttrain.environment.verifiers_conditioning import NativeConditioningInput

from ..update_records import ConditioningView, InvalidPolicyUpdate, PopulationSnapshot, require_identity


def sampled_logprobs(
    logits: torch.Tensor,
    inputs: NativeConditioningInput,
    *,
    sampled_indices: tuple[int, ...],
    score_temperature: float,
    entropies: list[float] | None = None,
) -> torch.Tensor:
    """Score selected original actions at their preceding causal positions.

    Returns one value per entry of ``sampled_indices``, in that order. Inputs
    must come from materialize_native_conditioning over authenticated retained
    evidence. The native adapter owns model/attention qualification, parameter
    identity and score freezing. This projection does not establish those
    guarantees or score observations merely because they are in context. The
    score temperature is required explicitly: native trainers use tempered
    policy probabilities, which differ from raw model probabilities. This is
    not reconstruction of a top-p/top-k filtered sampler distribution.

    When `entropies` is supplied, the detached entropy of the same tempered
    distribution at each selected causal position is appended to it, in order.
    This is observation only: it never enters the returned scores or their graph.
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
        return logits.new_zeros(0, dtype=torch.float32)
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
        entropies.extend(_entropies(log_probabilities))
    return values


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


@dataclass(frozen=True)
class PositionScores:
    """Scores for some population positions: ``values[i]`` belongs to ``positions[i]``."""

    positions: np.ndarray
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.positions.ndim != 1 or self.values.shape != (self.positions.size,):
            raise InvalidPolicyUpdate("position scores must pair each position with one value")


def score_views(
    model: Any,
    snapshot: PopulationSnapshot,
    views: Sequence[int],
    *,
    read_input: Callable[[ConditioningView], NativeConditioningInput],
    device: torch.device,
    score_temperature: float,
    entropies: np.ndarray | None = None,
) -> PositionScores:
    """Score every sampled token of the given views, one full-context forward per view.

    The native caller authenticates retained bytes in read_input and qualifies
    the model's causal-text@1 attention/position behavior. No detached KV cache,
    retokenization, padding or generation call is used here. Calling this across
    planned packs retains graphs; native backward must occur before any update.
    When ``entropies`` (length ``snapshot.size``) is given, each scored
    position's detached entropy is written into it.
    """
    if len(set(views)) != len(views) or any(not 0 <= view < len(snapshot.conditioning) for view in views):
        raise InvalidPolicyUpdate("model score support must contain unique admitted original actions")
    positions, values = [], []
    for index in views:
        view = snapshot.conditioning[index]
        inputs = read_input(view)
        record = inputs.record
        if (
            record.context_contract != "causal-text@1"
            or record.input_digest != view.digest
            or (len(inputs.token_ids) != view.context_tokens or record.context_tokens != view.context_tokens)
        ):
            raise InvalidPolicyUpdate("materialized model input differs from the frozen conditioning view")
        token_ids = torch.tensor([inputs.token_ids], dtype=torch.long, device=device)
        output = model(
            input_ids=token_ids,
            attention_mask=torch.ones_like(token_ids),
            position_ids=torch.arange(token_ids.shape[1], device=device).unsqueeze(0),
            use_cache=False,
        )
        local_entropies: list[float] | None = [] if entropies is not None else None
        values.append(
            sampled_logprobs(
                output.logits,
                inputs,
                sampled_indices=view.sampled,
                score_temperature=score_temperature,
                entropies=local_entropies,
            )
        )
        span = snapshot.view_positions(index)
        positions.append(np.arange(span.start, span.stop, dtype=np.int64))
        if entropies is not None and local_entropies is not None:
            entropies[span] = local_entropies
    if not positions:
        return PositionScores(np.zeros(0, dtype=np.int64), torch.zeros(0, device=device))
    return PositionScores(np.concatenate(positions), torch.cat(values))


def dense_scores(size: int, parts: Sequence[PositionScores], *, device: torch.device) -> torch.Tensor:
    """One float32 tensor of length ``size`` holding each part's values at its positions (zero elsewhere)."""
    if not parts:
        return torch.zeros(size, dtype=torch.float32, device=device)
    positions = np.concatenate([part.positions for part in parts])
    if np.unique(positions).size != positions.size:
        raise InvalidPolicyUpdate("score parts must not score one position twice")
    values = torch.cat([part.values.to(device=device, dtype=torch.float32) for part in parts])
    index = torch.as_tensor(positions, dtype=torch.long, device=device)
    return torch.zeros(size, dtype=torch.float32, device=device).index_put((index,), values)


@dataclass(frozen=True)
class FrozenPopulationScores:
    """Detached scores bound to one admitted population and score contract.

    ``values`` is a float32 tensor with one score per population position.
    """

    population_digest: str
    policy_version: str
    score_contract: str
    score_temperature: float
    values: torch.Tensor

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
        if self.values.shape != (snapshot.size,):
            raise InvalidPolicyUpdate("frozen policy scores must cover the complete admitted population")
        if (
            self.values.requires_grad
            or not self.values.is_floating_point()
            or not bool(torch.isfinite(self.values).all())
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
        scored = score_views(
            model,
            snapshot,
            range(len(snapshot.conditioning)),
            read_input=read_input,
            device=device,
            score_temperature=score_temperature,
        )
        values = dense_scores(snapshot.size, (scored,), device=device).detach().clone()
    frozen = FrozenPopulationScores(snapshot.digest, policy_version, score_contract, score_temperature, values)
    frozen.validate(
        snapshot, policy_version=policy_version, score_contract=score_contract, score_temperature=score_temperature
    )
    return frozen
