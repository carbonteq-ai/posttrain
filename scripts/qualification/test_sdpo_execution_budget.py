"""Separate processes cannot reset time or silently change the selected profile."""

import pytest
from sdpo_execution_budget import remaining_runtime, reserve_profile, start_shared_budget


def test_fitting_and_evaluation_share_one_deadline_including_audit_and_preflight(tmp_path):
    budget = start_shared_budget(tmp_path / "budget.json", "owned-fit", 600, now=1000., boot_id="boot")
    assert remaining_runtime(budget, 4300, now=1000., boot_id="boot") == 4300
    #4300s fitting plus100s host audit leaves2170s, including30s kill headroom.
    assert remaining_runtime(budget, 7100, now=5400., boot_id="boot") == 2170
    with pytest.raises(FileExistsError):
        start_shared_budget(tmp_path / "budget.json", "reset", 0, now=5400., boot_id="boot")
    with pytest.raises(TimeoutError):
        remaining_runtime(budget, 7100, now=7570., boot_id="boot")


def test_reboot_clock_rewind_and_extended_deadline_are_rejected(tmp_path):
    budget = start_shared_budget(tmp_path / "budget.json", "owned-fit", 600, now=1000., boot_id="boot")
    for value, now, boot in [(budget, 1100., "newboot"), (budget, 999., "boot"),
                             ({**budget, "deadline_boot_seconds": 20000.}, 1100., "boot")]:
        with pytest.raises(ValueError):
            remaining_runtime(value, 4000, now=now, boot_id=boot)


def test_reserve_requires_explicit_matching_mode_and_token_budgets():
    selection = {"context_limit": 32768, "reply_limit": 4096, "thinking": True,
                 "tool_parser_mode": "schema-bound-v1"}
    assert reserve_profile(selection)["thinking"]
    for change in [{"reply_limit": 8192}, {"context_limit": 16384},
                   {"thinking": "false"}, {"tool_parser_mode": "native"}]:
        with pytest.raises(ValueError):
            reserve_profile({**selection, **change})


def test_real_subprocess_alarm_cannot_extend_the_shared_deadline(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    from pathlib import Path

    receipt = tmp_path / "budget.json"
    start_shared_budget(receipt, "owned-fit", 7166)
    source = str(Path(__file__).parent)
    code = ("import json,time;from pathlib import Path;"
            "from sdpo_execution_budget import install_deadline_timer;"
            f"install_deadline_timer(json.loads(Path({str(receipt)!r}).read_text()));time.sleep(6)")
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": source},
                            timeout=8, check=False)
    assert result.returncode == -signal.SIGALRM
