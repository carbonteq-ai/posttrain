"""Sampled-token log-probabilities from final hidden states, in row chunks.

A policy score needs, per sampled token, ``log softmax(head(h) / T)[target]``
at the row preceding it. Materializing those rows' full-vocabulary logits (and
the FP32 log-softmax autograd keeps for backward) costs ``rows x vocabulary x 4``
bytes, about 260 KB per token for a 65,536-token vocabulary; one update of a
multi-turn population retains tens of gigabytes of it. Here the forward pass
computes logits chunk by chunk and keeps only each row's log-normalizer; the
backward pass recomputes a chunk's logits and applies the softmax derivative.
Retained memory per token becomes the hidden state (``hidden_size`` values) and
two floats.

Logits are formed exactly as the model's causal-LM head forms them: the output
projection in the model's dtype (plus bias, if any), then the model's declared
``final_logit_softcapping`` (``cap * tanh(z / cap)``) when its config has one,
then FP32 for the tempered log-softmax, as the trainer's FP32-logits contract
requires.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from ..update_records import InvalidPolicyUpdate

DEFAULT_CHUNK_ROWS = 256


def _chunk_logits(
    hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, softcap: float | None
) -> torch.Tensor:
    # The same op as the model's nn.Linear head, so logits round identically.
    logits = torch.nn.functional.linear(hidden, weight, bias).float()
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


class _SelectedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        targets: torch.Tensor,
        temperature: float,
        softcap: float | None,
        chunk_rows: int,
        with_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = hidden.shape[0]
        values = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        normalizers = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        entropies = torch.full((rows,), float("nan"), dtype=torch.float32, device=hidden.device)
        finite = torch.ones((), dtype=torch.bool, device=hidden.device)
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            tempered = _chunk_logits(hidden[start:stop], weight, bias, softcap) / temperature
            finite &= torch.isfinite(tempered).all()
            normalizer = tempered.logsumexp(dim=-1)
            values[start:stop] = tempered.gather(1, targets[start:stop].unsqueeze(1)).squeeze(1) - normalizer
            normalizers[start:stop] = normalizer
            if with_entropy:
                log_probabilities = tempered - normalizer.unsqueeze(1)
                entropies[start:stop] = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
        # One synchronization per call, not per chunk.
        if not bool(finite):
            raise InvalidPolicyUpdate("non-finite native logits at a required causal position")
        ctx.save_for_backward(hidden, weight, bias, targets, normalizers)
        ctx.temperature, ctx.softcap, ctx.chunk_rows = temperature, softcap, chunk_rows
        ctx.mark_non_differentiable(entropies)
        return values, entropies

    @staticmethod
    def backward(ctx: Any, grad_values: torch.Tensor, _grad_entropies: torch.Tensor) -> tuple[Any, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        hidden, weight, bias, targets, normalizers = ctx.saved_tensors
        temperature, softcap, chunk_rows = ctx.temperature, ctx.softcap, ctx.chunk_rows
        need_hidden, need_weight, need_bias = ctx.needs_input_grad[:3]
        grad_hidden = torch.zeros_like(hidden) if need_hidden else None
        grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if need_weight else None
        grad_bias = (
            torch.zeros(bias.shape, dtype=torch.float32, device=bias.device) if need_bias and bias is not None else None
        )
        for start in range(0, hidden.shape[0], chunk_rows):
            stop = min(start + chunk_rows, hidden.shape[0])
            chunk = hidden[start:stop]
            raw = torch.nn.functional.linear(chunk, weight, bias).float()
            capped = torch.tanh(raw / softcap) * softcap if softcap is not None else raw
            probabilities = (capped / temperature - normalizers[start:stop].unsqueeze(1)).exp()
            # d log p[target] / d tempered logits = onehot(target) - softmax.
            grad = -probabilities * grad_values[start:stop].unsqueeze(1)
            grad.scatter_add_(1, targets[start:stop].unsqueeze(1), grad_values[start:stop].unsqueeze(1))
            grad = grad / temperature
            if softcap is not None:
                grad = grad * (1 - torch.tanh(raw / softcap).square())
            if grad_hidden is not None:
                grad_hidden[start:stop] = (grad.to(weight.dtype) @ weight).to(hidden.dtype)
            if grad_weight is not None:
                grad_weight += grad.t() @ chunk.float()
            if grad_bias is not None:
                grad_bias += grad.sum(dim=0)
        return (
            grad_hidden,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None and bias is not None else None,
            None,
            None,
            None,
            None,
            None,
        )


def output_head(model: Any) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
    """The causal-LM output projection's weight and bias, and its declared final softcap."""
    head = model.get_output_embeddings()
    candidate = getattr(head, "weight", None)
    if not isinstance(candidate, torch.Tensor):
        raise ValueError("model output head must be a linear projection")
    weight = cast(torch.Tensor, candidate)
    if weight.ndim != 2:
        raise ValueError("model output head must be a linear projection")
    config = getattr(model, "config", None)
    text_config = config.get_text_config() if config is not None and hasattr(config, "get_text_config") else config
    softcap = getattr(text_config, "final_logit_softcapping", None)
    return weight, getattr(head, "bias", None), (float(softcap) if softcap is not None else None)


def selected_token_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    targets: torch.Tensor,
    *,
    temperature: float,
    softcap: float | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    with_entropy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``log softmax(head(hidden) / temperature)[targets]`` per row, and each row's entropy (detached).

    ``hidden`` holds the final hidden states of the rows preceding each target,
    shape (rows, hidden_size); ``targets`` the target token ids, shape (rows,).
    Entropies are NaN when ``with_entropy`` is false.
    """
    return _SelectedLogProbs.apply(
        hidden, weight, bias, targets, float(temperature), softcap, int(chunk_rows), bool(with_entropy)
    )
