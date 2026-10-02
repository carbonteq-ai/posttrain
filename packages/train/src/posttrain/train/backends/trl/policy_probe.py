"""Bounded, opt-in movement measurements over the actual training tokens."""

from __future__ import annotations

import time
from typing import Any

from posttrain.common import RunContext


class PostUpdateProbe:
    """Capture pre-update actor scores and rescore them after optimizer execution.

    Rows are the first bounded rows encountered, not a random population sample.
    This observes movement; it does not apply clipping to an optimizer update.
    """

    def __init__(self, context: RunContext, rows: int) -> None:
        if isinstance(rows, bool) or not isinstance(rows, int) or not 0 <= rows <= 16:
            raise ValueError("post_update_probe_rows must be an integer between 0 and 16")
        self.context = context
        self.limit = rows
        self.trainer: Any = None
        self.snapshots: list[dict[str, Any]] = []
        self.seen_rows = 0
        self.captured_rows = 0

    def capture(self, inputs: dict[str, Any], old_logps: Any) -> None:
        if not self.limit:
            return
        self.seen_rows += int(old_logps.shape[0])
        count = min(self.limit - self.captured_rows, int(old_logps.shape[0]))
        if count <= 0:
            return
        if any(
            inputs.get(key) is not None
            for key in (
                "pixel_values",
                "image_grid_thw",
                "num_images",
                "pixel_attention_mask",
                "spatial_shapes",
                "num_tiles",
                "image_sizes",
                "token_type_ids",
                "mm_token_type_ids",
                "image_position_ids",
            )
        ):
            raise ValueError("post-update probing currently supports text-only inputs")
        snapshot = {
            key: inputs[key][:count].detach().cpu().clone()
            for key in (
                "prompt_ids",
                "prompt_mask",
                "completion_ids",
                "completion_mask",
                "advantages",
            )
        }
        if inputs.get("tool_mask") is not None:
            snapshot["tool_mask"] = inputs["tool_mask"][:count].detach().cpu().clone()
        snapshot["old_logps"] = old_logps[:count].detach().float().cpu().clone()
        self.snapshots.append(snapshot)
        self.captured_rows += count

    def finish(self, *, skipped: bool = False) -> None:
        if not self.limit:
            return
        import torch

        trainer = self.trainer
        step = int(trainer.state.global_step)
        metrics = {
            "rows": float(self.captured_rows),
            "seen_rows": float(self.seen_rows),
            "optimizer_skipped": float(skipped),
        }
        started = time.perf_counter()
        observations = []
        try:
            if not skipped:
                with torch.no_grad(), trainer.accelerator.autocast():
                    for snapshot in self.snapshots:
                        inputs = {key: value.to(trainer.accelerator.device) for key, value in snapshot.items()}
                        ids = torch.cat((inputs["prompt_ids"], inputs["completion_ids"]), dim=1)
                        attention = torch.cat((inputs["prompt_mask"], inputs["completion_mask"]), dim=1)
                        new_logps, _, _ = trainer._get_per_token_logps_and_entropies(
                            trainer.model_wrapped,
                            ids,
                            attention,
                            inputs["completion_ids"].shape[1],
                            compute_entropy=False,
                            compute_aux_loss=False,
                        )
                        mask = inputs["completion_mask"]
                        if "tool_mask" in inputs:
                            mask = mask * inputs["tool_mask"]
                        observations.append(
                            (new_logps.float().cpu() - snapshot["old_logps"], mask.cpu(), snapshot["advantages"])
                        )
                metrics.update(
                    movement_metrics(
                        observations, trainer.epsilon_low, trainer.epsilon_high, trainer.importance_sampling_level
                    )
                )
            metrics["seconds"] = time.perf_counter() - started
            self.context.metrics({f"train/rl/post_update/{key}": value for key, value in metrics.items()}, step=step)
        finally:
            self.snapshots.clear()
            self.seen_rows = self.captured_rows = 0


def movement_metrics(
    observations: list[tuple[Any, Any, Any]], low: float, high: float, level: str = "sequence"
) -> dict[str, float]:
    """Summarize sampled old-to-new movement with exactly the policy mask.

    The sampled k3 statistic is an old-policy KL proxy, not reference-model KL.
    Masks are applied before exponentiation so unscored tool tokens cannot
    overflow the statistics. Advantage-dependent clipping uses the loss signs.
    """
    import torch

    tokens, ratios, outside, clipped = [], [], [], []
    for delta, mask, advantages in observations:
        valid = mask.bool()
        rows = valid.any(dim=-1)
        if not bool(rows.any()):
            continue
        delta, mask, advantages = delta[rows].double(), mask[rows].double(), advantages[rows]
        valid = mask.bool()
        if not bool(torch.isfinite(delta[valid]).all()):
            raise ValueError("post-update probe encountered nonfinite scored log probabilities")
        safe_delta = torch.where(valid, delta, 0.0)
        tokens.append(delta[valid])
        if level == "sequence":
            ratio = (safe_delta.sum(-1) / mask.sum(-1)).exp().unsqueeze(-1)
            ratios.append(ratio.flatten())
        elif level == "token":
            ratio = safe_delta.exp()
            ratios.append(ratio[valid])
        else:
            raise ValueError(f"unsupported probe importance sampling level: {level}")
        if advantages.ndim == 1:
            advantages = advantages.unsqueeze(-1)
        out = (ratio < 1 - low) | (ratio > 1 + high)
        clip = ((ratio < 1 - low) & (advantages < 0)) | ((ratio > 1 + high) & (advantages > 0))
        outside.append((out * mask).sum(-1) / mask.sum(-1))
        clipped.append((clip * mask).sum(-1) / mask.sum(-1))
    if not tokens:
        return {"scored_tokens": 0.0, "scored_rows": 0.0}
    delta, ratio = torch.cat(tokens), torch.cat(ratios)
    return {
        "scored_tokens": float(delta.numel()),
        "scored_rows": float(sum(x.numel() for x in outside)),
        "ratio_min": float(ratio.min()),
        "ratio_p50": float(ratio.quantile(0.5)),
        "ratio_p95": float(ratio.quantile(0.95)),
        "ratio_max": float(ratio.max()),
        "outside_fraction": float(torch.cat(outside).mean()),
        "clip_fraction": float(torch.cat(clipped).mean()),
        "old_policy_sampled_k3": float((torch.expm1(-delta) + delta).mean()),
        "token_abs_log_ratio_max": float(delta.abs().max()),
    }
