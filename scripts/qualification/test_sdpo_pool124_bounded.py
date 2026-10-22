"""Broad-corpus bounded variants cannot inherit narrow cohorts or changed budgets."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sdpo_long_campaign_config import execution_profile
from sdpo_paired_ablation import branch_commands, main
from sdpo_pool124_config import collection_budget, diagnostic_groups, step_tasks


def args(**changes):
    values = dict(run_profile="pool124-bounded", steps=6, batch_size=32, online_rollouts=2,
                  lr=1e-5, lr_end_ratio=.1, lr_schedule="cosine-decay", context_limit=32768,
                  reply_limit=4096, qualify_capacity=True, thinking=False,
                  teacher_feedback="solution-or-feedback", tool_parser_mode="schema-bound-v1",
                  selection_mode="failure-frontier", alpha=1., topk=20, teacher_rate=.01)
    return SimpleNamespace(**(values | changes))


def test_matched_variants_preserve_task_breadth_and_evaluate_at_ten():
    for changes in ({}, {"teacher_feedback": "luna-or-solution"}, {"thinking": True}):
        profile = execution_profile(args(**changes))
        assert profile["batch"] == 32 and profile["evaluate_every"] == 10
        assert profile["time_limit_s"] < 7200
    tasks = [f"task-{i}" for i in range(124)]
    selected = sum((step_tasks(tasks, step) for step in range(1, 7)), [])
    assert set(selected) == set(tasks) and len(selected) == 192


def test_no_extra_updates_or_silent_combined_treatment():
    for changes in ({"steps": 10}, {"steps": 20}, {"batch_size": 16}, {"reply_limit": 8192},
                    {"qualify_capacity": False}, {"selection_mode": "first-error-frontier"},
                    {"thinking": True, "teacher_feedback": "luna-or-solution"}):
        with pytest.raises(ValueError):
            execution_profile(args(**changes))


def test_bigger_training_budget_never_changes_full_heldout_budget():
    train = collection_budget("online_train", "pool124-bounded")
    assert train == dict(context=32768, reply=4096, turns=8, episode=12288, replicas=2)
    assert collection_budget("heldout62", "pool124-bounded") == dict(
        context=65536, reply=8192, turns=20, episode=32768, replicas=5)
    assert collection_budget("online_train")["context"] == 16384


def test_bounded_sentinels_removed_without_changing_main50_evaluation_cadence_or_population():
    tasks, heldout = [f"train-{i}" for i in range(124)], [f"test-{i}" for i in range(62)]
    refs = tasks[:5]
    assert diagnostic_groups(tasks, heldout, "pool124-bounded", refs) == [
        ("reference_train", refs), ("heldout62", heldout)]
    with pytest.raises(ValueError):
        diagnostic_groups(tasks, heldout, "pool124-bounded", ["test-0"]+refs[:4])
    main = diagnostic_groups(tasks, heldout, "pool124-50")
    assert main[0] == ("train", step_tasks(tasks, 1)) and main[1] == ("heldout62", heldout)
    assert diagnostic_groups(tasks, heldout, "pool124-qualify") == [main[0]]


def test_paired_commands_preserve_full_pool_with_two_budgeted_six_step_branches():
    config = dict(budget_slot=9, steps=12, training_tasks=124, batch_size=32,
                  context_limit=32768, reply_limit=4096, wall_time_cap_seconds=7200,
                  selection_mode="failure-frontier", lr=1e-5, lr_schedule="cosine-decay",
                  lr_end_ratio=.1, alpha=1., topk=20, teacher_rate=.01)
    branches = branch_commands(config, Path("driver.py"), Path("/capsule"), Path("/model"), Path("/heldout"))
    assert [name for name, _ in branches] == ["control", "luna"]
    for _, command in branches:
        assert command[command.index("--training-pool")+1] == "/capsule/train-pool.json"
        assert command[command.index("--run-profile")+1] == "pool124-bounded"
        assert command[command.index("--steps")+1] == "6"
        assert command[command.index("--batch-size")+1] == "32"
        assert "--qualify-capacity" in command and "--thinking" not in command
    for changes in ({"training_tasks": 8}, {"batch_size": 16}, {"steps": 40}):
        with pytest.raises(ValueError):
            branch_commands(config | changes, Path("d"), Path("/c"), Path("/m"), Path("/h"))


def test_paired_parent_reconciles_two_independent_six_step_receipts(tmp_path):
    config = dict(budget_slot=9, steps=12, training_tasks=124, batch_size=32,
                  context_limit=32768, reply_limit=4096, wall_time_cap_seconds=7200,
                  selection_mode="failure-frontier", lr=1e-5, lr_schedule="cosine-decay",
                  lr_end_ratio=.1, alpha=1., topk=20, teacher_rate=.01,
                  run_name="test-only-no-gpu", branch_wall_time_seconds=20, gpu_timeout_seconds=120)
    (tmp_path / "experiment.json").write_text(json.dumps(config))
    driver = tmp_path / "driver"
    driver.mkdir()
    (driver / "sdpo_agent_ablation.py").write_text('''
import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--steps',type=int);p.add_argument('--output',type=pathlib.Path)
a,_=p.parse_known_args();a.output.mkdir()
(a.output/'capacity-qualification.json').write_text(json.dumps(dict(qualified=True,optimizer_updates=0,context_limit=32768,reply_limit=4096)))
(a.output/'updates.json').write_text(json.dumps([dict(step=i) for i in range(1,a.steps+1)]))
(a.output/'receipt.json').write_text(json.dumps(dict(completed=True,optimizer_updates=a.steps)))
''')
    output = tmp_path / "results"
    main(SimpleNamespace(capsule=tmp_path, model=Path("/fake"), heldout=Path("/fake-heldout"), output=output))
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["optimizer_updates"] == 12 and len(receipt["branches"]) == 2
    assert [b["optimizer_updates"] for b in receipt["branches"]] == [6, 6]
    assert receipt["heldout62_evaluated"] is True
