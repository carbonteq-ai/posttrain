"""A failed qualification and an unrun arm must not appear to have reward gains."""

import json

import pytest
from summarize_sdpo_search import cohort_summary, summarize


def source(tmp_path):
    record = {"budget_slot": 1, "steps": 2, "status": "failed_reader_before_updates",
              "run_name": "synthetic-failure-fixture", "teacher_feedback": "native-feedback", "lr": 1e-6}
    (tmp_path / "experiment-ledger.json").write_text(json.dumps({"maximum_experiments": 10,
        "slots_launched": 1, "per_experiment_maximum_updates": 20, "experiments": [record]}))
    metric = {"full_population_task_mean": .5, "episode_coverage": 2}
    audit = {"completed": False, "updates": [], "unique_native_episodes": 2, "populations": [
        {"cohort": "train", "step": 0, "summary": {"episodes": 2, "errors": 0,
            "metrics": {"task_completed_correctly": metric, "partial_credit": metric}}}]}
    (tmp_path / "experiment01-audit.json").write_text(json.dumps(audit))
    return audit


def test_failed_and_unrun_attempts_do_not_get_invented_post_scores(tmp_path):
    source(tmp_path)
    result = summarize(tmp_path)
    failed, pending = result["rows"][:2]
    assert failed["actual_updates"] == 0 and not failed["runtime_completed"]
    assert failed["train"]["baseline"]["partial_credit"] == .5
    assert failed["train"]["final_post_update"] is None
    assert failed["train"]["paired_final_vs_baseline"] is None
    assert pending["actual_updates"] is None and pending["native_episodes"] is None
    assert result["attempts_launched"] == 1 and result["successful_runtime_receipts"] == 0


def test_full_population_scores_cannot_hide_missing_coverage(tmp_path):
    audit = source(tmp_path)
    audit["populations"][0]["summary"]["metrics"]["partial_credit"]["episode_coverage"] = 1
    (tmp_path / "experiment01-audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="incomplete episode coverage"):
        summarize(tmp_path)


def test_luna_and_control_keep_separate_rates_and_losses(tmp_path):
    branch = source(tmp_path)
    branch.update(completed=True, updates=[{"step": i, "loss": .1} for i in range(1, 11)])
    original = json.loads(json.dumps(branch))
    for value in (original, branch):
        value["populations"].append(json.loads(json.dumps(value["populations"][0])))
        value["populations"][-1]["step"] = 10
    branch["populations"][-1]["summary"]["metrics"]["partial_credit"]["full_population_task_mean"] = .75
    original["populations"][-1]["summary"]["metrics"]["partial_credit"]["full_population_task_mean"] = .25
    ledger = json.loads((tmp_path / "experiment-ledger.json").read_text())
    ledger["experiments"][0].update(budget_slot=9, steps=20, status="completed")
    (tmp_path / "experiment-ledger.json").write_text(json.dumps(ledger))
    (tmp_path / "experiment09-audit.json").write_text(json.dumps({"completed": True,
        "branches": {"control": original, "luna": branch}, "total_optimizer_updates": 20,
        "unique_native_episodes": 4, "luna_vs_matched_larger_budget_control": {}}))
    result = summarize(tmp_path)["rows"][8]
    assert result["actual_updates"] == 20 and result["train"] is None and result["loss_first"] is None
    assert result["branches"]["control"]["train"]["final_post_update"]["partial_credit"] == .25
    assert result["branches"]["luna"]["train"]["final_post_update"]["partial_credit"] == .75


def test_unstarted_luna_and_incomplete_post_are_not_invented(tmp_path):
    branch = source(tmp_path)
    branch["updates"] = [{"step": 1, "loss": .2}]
    branch["populations"][0]["cohort"] = "heldout62"
    ledger = json.loads((tmp_path / "experiment-ledger.json").read_text())
    ledger["experiments"][0].update(budget_slot=9, steps=12, status="terminal_timeout")
    (tmp_path / "experiment-ledger.json").write_text(json.dumps(ledger))
    (tmp_path / "experiment09-audit.json").write_text(json.dumps({"completed": False,
        "branches": {"control": branch}, "total_optimizer_updates": 1, "unique_native_episodes": 2}))
    row = summarize(tmp_path)["rows"][8]
    assert row["actual_updates"] == 1 and row["unstarted_branches"] == ["luna"]
    assert row["luna_vs_matched_control"] is None
    control = row["branches"]["control"]
    assert control["heldout62"]["baseline"]["partial_credit"] == .5
    assert control["heldout62"]["final_post_update"] is None
    assert control["development"]["baseline"] is None


def test_accepted_live_arm_without_native_audit_has_unknown_updates(tmp_path):
    source(tmp_path)
    ledger = json.loads((tmp_path / "experiment-ledger.json").read_text())
    ledger["experiments"][0].update(budget_slot=10, status="accepted_running_startup",
                                   launch_receipt="launch.json")
    (tmp_path / "experiment-ledger.json").write_text(json.dumps(ledger))
    (tmp_path / "launch.json").write_text(json.dumps({"run_name": "synthetic-failure-fixture"}))
    row = summarize(tmp_path)["rows"][9]
    assert row["actual_updates"] is None and row["native_episodes"] is None
    assert not row["runtime_completed"] and row["heldout62"] is None


def test_broad_cohort_keeps_its_paired_intervals(tmp_path):
    branch = source(tmp_path)
    baseline = branch['populations'][0]
    baseline['cohort'] = 'heldout62'
    post = json.loads(json.dumps(baseline))
    comparison = {'partial_credit': {'paired': {'ci95': [-.1, .2]},
                                    'full_population_gain_identification_bounds': [0., .1]}}
    post.update(step=6, paired_vs_base=comparison)
    branch['populations'].append(post)
    result = cohort_summary(branch, 'heldout62')
    assert result['paired_final_vs_baseline'] == comparison
    post['paired_vs_fresh_base'] = {'different': True}
    with pytest.raises(ValueError, match='conflicting paired comparison'):
        cohort_summary(branch, 'heldout62')
