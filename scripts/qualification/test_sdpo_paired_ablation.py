"""The paired experiment shares budgets, never weights or optimizer state."""

from pathlib import Path

import pytest
from sdpo_paired_ablation import branch_commands


def config():
    return {"budget_slot": 9, "steps": 20, "context_limit": 32768, "reply_limit": 4096,
            "wall_time_cap_seconds": 7200, "lr": 1e-5, "lr_schedule": "constant", "lr_end_ratio": .1,
            "alpha": 1., "topk": 20, "teacher_rate": .01, "selection_mode": "failure-frontier"}


def test_branches_share_all_optimizer_sampling_and_token_settings_with_twenty_total_updates():
    branches = branch_commands(config(), Path("driver.py"), Path("/capsule"), Path("/model"), Path("/heldout"))
    assert [name for name, _ in branches] == ["control", "luna"]
    first, second = (command for _, command in branches)
    for key in ["--model", "--cohorts", "--context-limit", "--reply-limit", "--steps", "--lr",
                "--lr-schedule", "--alpha", "--topk", "--teacher-rate", "--selection-mode", "--online-rollouts"]:
        assert first[first.index(key) + 1] == second[second.index(key) + 1]
    assert int(first[first.index("--steps") + 1]) + int(second[second.index("--steps") + 1]) == 20
    assert "--luna-references" not in first and "--luna-references" in second
    assert "--qualify-capacity" in first
    assert not any("checkpoint" in argument for argument in first + second)


def test_bigger_budget_and_luna_cannot_spill_into_an_unregistered_slot():
    for change in [{"budget_slot": 10}, {"steps": 40}, {"reply_limit": 8192}, {"wall_time_cap_seconds": 14400}]:
        with pytest.raises(ValueError):
            branch_commands({**config(), **change}, Path("driver.py"), Path("/capsule"), Path("/model"), Path("/heldout"))
