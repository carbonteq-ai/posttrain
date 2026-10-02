"""Round arithmetic of TRL's GRPO active sampling (CarbonTeq TRL 1.12.0.post11).

Counts are prompt groups. One update needs ``target`` groups whose reward
spread exceeds zero; the reserved candidate pool holds ``max_rounds * target``
groups. The first round dispatches ``target + oversample``; each later round
dispatches the missing groups plus ``oversample_refill``, never more than the
first round and never more than the pool left. A round is never cut below the
groups it is missing: that exhausts the pool instead. The update keeps the first
``target`` retained groups in dispatch order. veRL's native
``ActiveSamplingRounds`` implements the same arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .update_records import InvalidPolicyUpdate


@dataclass
class ActiveRoundPlan:
    target: int
    max_rounds: int
    oversample: int = 0
    oversample_refill: int = 0
    cursor: int = field(default=0, init=False)
    rounds: int = field(default=0, init=False)
    retained: int = field(default=0, init=False)
    oversampled: int = field(default=0, init=False)
    round_log: list[tuple[int, int, int]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.target < 1 or self.max_rounds < 1 or self.oversample < 0 or self.oversample_refill < 0:
            raise InvalidPolicyUpdate("active rounds require positive target/rounds and nonnegative oversampling")

    @property
    def pool(self) -> int:
        return self.max_rounds * self.target

    @property
    def missing(self) -> int:
        return max(self.target - self.retained, 0)

    def next_round(self) -> tuple[int, int] | None:
        """``(requested, round_size)`` of the next round, or ``None`` when full or out of rounds."""
        if self.missing == 0 or self.rounds >= self.max_rounds:
            return None
        missing = requested = round_size = self.missing
        if self.oversample or self.oversample_refill:
            extra = self.oversample_refill if self.rounds else self.oversample
            requested = min(missing + extra, self.target + self.oversample)
            round_size = max(min(requested, self.pool - self.cursor), missing)
        if self.cursor + round_size > self.pool:
            raise InvalidPolicyUpdate(
                f"active sampling exhausted its bounded candidate pool: {missing} groups are missing but only "
                f"{self.pool - self.cursor} of {self.pool} candidates remain")
        return requested, round_size

    def record(self, requested: int, round_size: int, retained: int) -> None:
        self.oversampled += round_size - self.missing
        self.cursor += round_size
        self.rounds += 1
        self.retained += retained
        self.round_log.append((requested, round_size, retained))

    def require_full(self) -> None:
        if self.retained < self.target:
            raise InvalidPolicyUpdate(
                f"active sampling exhausted {self.max_rounds} generation rounds before filling its update; "
                f"retained {self.retained} of {self.target} prompt groups")

    def metrics(self, group_size: int) -> dict[str, float]:
        """TRL post11's names and values; ``candidate_groups_*`` counters count rows."""
        generated = self.cursor
        values: dict[str, float] = {
            "generation_rounds": self.rounds,
            "retained_fraction": self.retained / generated if generated else 0.0,
            "generated_rows": generated * group_size,
            "candidate_groups_reserved": self.pool * group_size,
            "candidate_groups_generated": generated * group_size,
            "candidate_groups_retained": self.retained * group_size,
            "candidate_groups_unused": (self.pool - generated) * group_size,
        }
        if self.oversample or self.oversample_refill:
            for index, (requested, generated_groups, retained) in enumerate(self.round_log, start=1):
                values[f"round_{index}_requested_groups"] = requested
                values[f"round_{index}_generated_groups"] = generated_groups
                values[f"round_{index}_retained_groups"] = retained
            values["oversampled_groups"] = self.oversampled
            values["discarded_groups"] = max(self.retained - self.target, 0)
        return {f"train/rl/active_sampling_{name}": float(value) for name, value in values.items()}
