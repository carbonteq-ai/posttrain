"""One value per update for rollout metrics that are measured once per rollout batch.

Active sampling generates several rollout batches for one optimizer update.
The observation baseline (`06-observation-and-lineage.md`) requires one value
per update, so batches add to these totals and the totals are written once,
when the update ends. Counts, seconds and tokens are sums; rates are
recomputed from the sums; means are weighted by how many items each batch had.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from posttrain.common import RunContext

ROLLOUT_BATCH_SECONDS = "train/rl/rollout_batch_seconds"


@dataclass
class _Update:
    batches: int = 0
    sums: dict[str, float] = field(default_factory=dict)
    # name -> (weighted sum, total weight)
    means: dict[str, tuple[float, float]] = field(default_factory=dict)


class RolloutUpdateTotals:
    """Accumulates rollout-batch measurements and writes each update's totals once."""

    def __init__(self, context: RunContext) -> None:
        self._context = context
        self._updates: dict[int, _Update] = {}

    def add_batch(
        self,
        step: int,
        *,
        sums: Mapping[str, float],
        means: Mapping[str, tuple[float, float]] = {},
        batch_seconds: float | None = None,
    ) -> None:
        """Add one rollout batch; `means` maps a name to (mean over the batch, number of items)."""

        update = self._updates.setdefault(step, _Update())
        update.batches += 1
        for name, value in sums.items():
            update.sums[name] = update.sums.get(name, 0.0) + float(value)
        self.add_means(step, means)
        if batch_seconds is not None:
            self._context.metrics(
                {ROLLOUT_BATCH_SECONDS: batch_seconds}, step=step, attributes={"rollout_batch_ordinal": update.batches}
            )

    def add_means(self, step: int, means: Mapping[str, tuple[float, float]]) -> None:
        update = self._updates.setdefault(step, _Update())
        for name, (mean, weight) in means.items():
            if weight <= 0:
                continue
            total, weights = update.means.get(name, (0.0, 0.0))
            update.means[name] = (total + float(mean) * weight, weights + weight)

    def flush(self, through_step: int | None = None) -> None:
        """Write the totals of every pending update up to `through_step` (all when None)."""

        for step in sorted(self._updates):
            if through_step is not None and step > through_step:
                continue
            update = self._updates.pop(step)
            values: dict[str, Any] = dict(update.sums)
            values.update({name: total / weight for name, (total, weight) in update.means.items()})
            seconds = update.sums.get("train/rl/time/rollout_seconds")
            completion = update.sums.get("train/rl/rollout_completion_tokens")
            selected = update.sums.get("train/rl/rollout_selected_tokens")
            if seconds is not None and completion is not None:
                values["train/rl/rollout_tokens_per_second"] = completion / seconds if seconds > 0 else 0.0
            if selected is not None and completion is not None:
                values["train/rl/rollout_selected_token_fraction"] = selected / completion if completion else 0.0
            values.pop("train/rl/rollout_completion_tokens", None)
            if values:
                self._context.metrics(
                    values,
                    step=step,
                    attributes={"rollout_population_scope": "candidate", "rollout_batches": update.batches},
                )


def update_totals_callback_type(imports: Mapping[str, Any], totals: RolloutUpdateTotals) -> type[Any]:
    """A TRL callback that writes an update's rollout totals when the update ends."""

    parent = imports["TrainerCallback"]

    class RolloutUpdateTotalsCallback(parent):
        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args
            totals.flush(int(state.global_step))
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args, state
            totals.flush()
            return control

    return RolloutUpdateTotalsCallback


__all__ = ["ROLLOUT_BATCH_SECONDS", "RolloutUpdateTotals", "update_totals_callback_type"]
