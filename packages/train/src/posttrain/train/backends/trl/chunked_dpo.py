"""Chunked DPO log-probabilities: TRL's sigmoid (+ sft) DPO loss without full-vocabulary logits.

TRL's DPO path builds ``outputs.logits`` for the chosen and rejected rows together; at 32,768 tokens and a 248,320-word
vocabulary that is about 32 GB in bf16 before any gradient. Here the decoder runs to its last hidden state and the
output projection is applied to completion tokens only, a chunk at a time under gradient checkpointing, so peak logits
memory is ``chunk_size x vocab_size`` (the fork's SFT ``chunked_nll`` does the same for the supervised loss). The loss
and the logged rewards equal TRL's ``_compute_loss`` for ``loss_type=["sigmoid"]`` or ``["sigmoid", "sft"]`` with the
default reverse-KL divergence; the full-logits metrics (entropy, mean logits, token accuracy) are not logged.
"""

from __future__ import annotations

from typing import Any

CHUNK_SIZE = 256  # completion tokens per output-projection chunk (the fork's SFT chunked_nll value)


def _token_logps(hidden: Any, weight: Any, bias: Any, labels: Any) -> Any:
    import torch

    logits = hidden.float() @ weight.float().t()
    if bias is not None:
        logits = logits + bias.float()
    return logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def chunked_sequence_logps(
    hidden: Any, weight: Any, bias: Any, labels: Any, mask: Any, chunk_size: int = CHUNK_SIZE
) -> tuple[Any, Any]:
    """Per-row sums of log p(label) over masked positions, and the per-row masked token counts.

    ``hidden`` (B, S, H) is aligned with ``labels`` and ``mask`` (B, S): position i predicts ``labels[:, i]``.
    Only positions with ``mask == 1`` reach the output projection."""

    import torch
    from torch.utils.checkpoint import checkpoint

    rows, length, width = hidden.shape
    flat_hidden = hidden.reshape(-1, width)
    flat_labels = labels.reshape(-1)
    positions = mask.reshape(-1).bool().nonzero(as_tuple=True)[0]
    owner = torch.arange(rows, device=hidden.device).repeat_interleave(length)
    sums = hidden.new_zeros(rows, dtype=torch.float32)
    for start in range(0, positions.numel(), chunk_size):
        chosen = positions[start : start + chunk_size]
        logps = checkpoint(_token_logps, flat_hidden[chosen], weight, bias, flat_labels[chosen], use_reentrant=False)
        sums = sums.index_add(0, owner[chosen], logps)
    return sums, mask.sum(dim=1)


def hidden_and_head(model: Any, input_ids: Any, attention_mask: Any) -> tuple[Any, Any, Any]:
    """Last hidden state from the decoder (adapter layers included) and the output projection's weight and bias."""

    target = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = target.get_output_embeddings()
    if type(head).__module__.startswith("peft."):
        raise ValueError("chunked DPO reads the output projection directly; it must not carry an adapter")
    outputs = target.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return outputs.last_hidden_state, head.weight, head.bias


def completion_logps(model: Any, inputs: dict[str, Any], chunk_size: int = CHUNK_SIZE) -> tuple[Any, Any]:
    """Summed completion log-probabilities and token counts for every row (chosen rows, then rejected)."""

    hidden, weight, bias = hidden_and_head(model, inputs["input_ids"], inputs["attention_mask"])
    return chunked_sequence_logps(
        hidden[:, :-1], weight, bias, inputs["input_ids"][:, 1:], inputs["completion_mask"][:, 1:], chunk_size
    )


def reference_logps(trainer: Any, model: Any, inputs: dict[str, Any]) -> tuple[Any, Any]:
    """TRL's ``compute_ref_log_probs`` (no ld_alpha) on the chunked path."""

    import torch
    from trl.models.utils import disable_gradient_checkpointing  # pyright: ignore[reportMissingImports]

    if trainer.ld_alpha is not None:
        raise ValueError("chunked DPO does not implement ld_alpha")
    with torch.no_grad(), disable_gradient_checkpointing(trainer.model, trainer.args.gradient_checkpointing_kwargs):
        logps, _ = completion_logps(model, inputs)
    return logps.chunk(2, dim=0)


def compute_loss(trainer: Any, model: Any, inputs: dict[str, Any], return_outputs: bool) -> Any:
    """TRL's ``_compute_loss`` for sigmoid DPO (+ sft) with the reverse-KL divergence, on the chunked path."""

    import torch
    import torch.nn.functional as F
    from trl.models.utils import disable_gradient_checkpointing  # pyright: ignore[reportMissingImports]

    if return_outputs:
        raise ValueError("chunked DPO never materializes logits, so outputs cannot be returned")
    if trainer.ld_alpha is not None or trainer.use_weighting or trainer.aux_loss_enabled:
        raise ValueError("chunked DPO does not implement ld_alpha, WPO weighting or MoE aux losses")
    if trainer.f_divergence_type != "reverse_kl":
        raise ValueError("chunked DPO implements only the reverse-KL (standard DPO) divergence")
    if not set(trainer.loss_types) <= {"sigmoid", "sft"}:
        raise ValueError(f"chunked DPO implements only sigmoid and sft losses, not {trainer.loss_types}")
    mode = "train" if trainer.model.training else "eval"

    logps, counts = completion_logps(model, inputs)
    chosen_logps, rejected_logps = logps.chunk(2, dim=0)
    if trainer.precompute_ref_logps:
        ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
    else:
        with torch.no_grad(), disable_gradient_checkpointing(trainer.model, trainer.args.gradient_checkpointing_kwargs):
            if trainer.ref_model is None:
                raise ValueError("chunked DPO needs a separate reference model or precomputed reference scores")
            ref_logps, _ = completion_logps(trainer.ref_model, inputs)
        ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps
    delta_score = chosen_logratios - rejected_logratios
    loss = 0.0
    for loss_type, loss_weight in zip(trainer.loss_types, trainer.loss_weights, strict=True):
        if loss_type == "sigmoid":
            per_sequence_loss = -F.logsigmoid(trainer.beta * delta_score)
        else:  # sft: mean token NLL over the chosen completion tokens of the batch (TRL's batch_loss)
            chosen_counts, _ = counts.chunk(2, dim=0)
            batch_loss = -chosen_logps.sum() / chosen_counts.sum().clamp_min(1)
            per_sequence_loss = batch_loss.expand(chosen_logps.size(0))
        loss += per_sequence_loss.mean() * loss_weight

    accelerator = trainer.accelerator
    if mode == "train":
        trainer._total_train_tokens += accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    chosen_rewards = trainer.beta * chosen_logratios.detach()
    rejected_rewards = trainer.beta * rejected_logratios.detach()
    trainer._metrics[mode]["rewards/chosen"].append(accelerator.gather(chosen_rewards).mean().item())
    trainer._metrics[mode]["rewards/rejected"].append(accelerator.gather(rejected_rewards).mean().item())
    accuracies = (chosen_rewards > rejected_rewards).float()
    trainer._metrics[mode]["rewards/accuracies"].append(accelerator.gather(accuracies).mean().item())
    trainer._metrics[mode]["rewards/margins"].append(
        accelerator.gather(chosen_rewards - rejected_rewards).mean().item()
    )
    trainer._metrics[mode]["logps/chosen"].append(accelerator.gather(chosen_logps.detach()).mean().item())
    trainer._metrics[mode]["logps/rejected"].append(accelerator.gather(rejected_logps.detach()).mean().item())
    return loss


__all__ = ["CHUNK_SIZE", "chunked_sequence_logps", "completion_logps", "compute_loss", "reference_logps"]
