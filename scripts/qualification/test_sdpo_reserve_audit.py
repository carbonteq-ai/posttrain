import json

import pytest
from analyze_sdpo_agent import audit
from sdpo_agent_config import MODEL_REVISION, fingerprint
from self_campaign_config import summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def reserve(tmp_path):
    cohort = {"reserve": [f"reserve{i}" for i in range(24)]}
    write(tmp_path / "primary-selection.json", {"frozen_before_reserve": True, "approved_for_reserve": True,
        "model_revision": MODEL_REVISION, "reserve_sha256": fingerprint(cohort["reserve"]),
        "tool_parser_mode": "native", "adapter_sha256": "a" * 64,
        "context_limit": 16384, "reply_limit": 2048, "thinking": False})
    write(tmp_path / "inputs.json", {"tool_parser_mode": "native",
        "context_limit": 16384, "reply_limit": 2048, "thinking": False})
    write(tmp_path / "shared-budget-snapshot.json", {"experiment_slot": 10, "total_wall_cap_seconds": 7200,
        "maximum_combined_optimizer_updates": 20, "clock": "CLOCK_BOOTTIME", "boot_id": "test-boot",
        "preflight_reservation_seconds": 600, "created_boot_seconds": 1000., "deadline_boot_seconds": 7600.})
    write(tmp_path / "receipt.json", {"completed": True, "optimizer_updates": 0, "trainable_parameters": 0,
        "elapsed_s": 30, "reserve_evaluated": True, "heldout62_evaluated": False,
        "matched_profile": {"context_limit": 16384, "reply_limit": 2048, "thinking": False},
        "finished_boot_seconds": 6000., "shared_deadline_boot_seconds": 7600.,
        "rollout_versions": [{"policy_version": i, "weights_sha256": str(i) * 64} for i in (0, 1)]})
    write(tmp_path / "candidate-load.json", {"all_adapter_tensors_exact": True, "optimizer_updates": 0,
                                           "source_weights_sha256": "a" * 64})
    write(tmp_path / "fresh-base-gate.json", {"max_abs_delta": 0})
    write(tmp_path / "batch-invariance-v0.json", {"exact_batch_invariance": True,
        "records": [{"peak_in_flight": n, "tokens": [1], "logprobs": [-.5]} for n in (1, 32)]})
    for step in (0, 1):
        write(tmp_path / f"parity-v{step}.json", {"policy_version": step, "actor": [-.5], "rollout": [-.5]})
        folder = tmp_path / f"step-{step:04d}/reserve"
        episodes = [{"id": f"{step}-{name}-{replica}", "task": {"data": {"name": name}},
            "traces": [{"nodes": [], "stop_condition": "agent_completed", "metrics": {
                "task_completed_correctly": float(step == 1 and name == "reserve0"),
                "tool_mistakes": 0, "tool_unknown_id": 0},
                "rewards": {"partial_credit": {"score": float(step == 1 and name == "reserve0")}}}]}
            for name in cohort["reserve"] for replica in range(2)]
        write(folder / "outcomes.json", summarize(episodes, cohort["reserve"], 2))
        (folder / "native").mkdir()
        (folder / "native/traces.jsonl").write_text("\n".join(map(json.dumps, episodes)))
        (folder / "generation-calls.jsonl").write_text(json.dumps({"policy_version": step,
            "adapter_sha256": str(step) * 64, "complete": True}))
    return tmp_path, cohort


def test_frozen_zero_update_reserve_has96_distinct_native_episodes(reserve):
    root, cohort = reserve
    result = audit(root, cohort)
    assert result["completed"] and result["unique_native_episodes"] == 96
    assert result["optimizer_updates"] == 0
    assert result["paired_vs_matched_fresh_base"]["task_completed_correctly"]["paired"]["delta"] == 1 / 24


def test_reserve_rejects_reused_ids_across_policies(reserve):
    root, cohort = reserve
    file = root / "step-0001/reserve/native/traces.jsonl"
    values = [json.loads(line) for line in file.read_text().splitlines()]
    values[0]["id"] = "0-reserve0-0"
    file.write_text("\n".join(map(json.dumps, values)))
    with pytest.raises(ValueError, match="reused"):
        audit(root, cohort)


def test_missing_score_is_not_zero_filled_or_dropped_silently(reserve):
    root, cohort = reserve
    folder = root / "step-0001/reserve"
    file = folder / "native/traces.jsonl"
    values = [json.loads(line) for line in file.read_text().splitlines()]
    values[0]["traces"][0]["metrics"].pop("task_completed_correctly")
    file.write_text("\n".join(map(json.dumps, values)))
    write(folder / "outcomes.json", summarize(values, cohort["reserve"], 2))
    result = audit(root, cohort)
    score = result["populations"][1]["summary"]["metrics"]["task_completed_correctly"]
    assert score["episode_coverage"] == 47 and score["full_population_task_mean"] is None
    assert result["paired_vs_matched_fresh_base"]["task_completed_correctly"]["excluded_tasks"] == ["reserve0"]


def test_candidate_load_must_match_frozen_adapter(reserve):
    root, cohort = reserve
    write(root / "candidate-load.json", {"all_adapter_tensors_exact": True, "optimizer_updates": 0,
        "source_weights_sha256": "b" * 64})
    with pytest.raises(ValueError, match="receipt"):
        audit(root, cohort)


def test_parser_cannot_change_after_selection(reserve):
    root, cohort = reserve
    inputs = json.loads((root / "inputs.json").read_text())
    write(root / "inputs.json", {**inputs, "tool_parser_mode": "schema-bound-v1"})
    with pytest.raises(ValueError, match="preselection"):
        audit(root, cohort)


def test_reserve_cannot_silently_increase_token_budget_or_enable_thinking(reserve):
    root, cohort = reserve
    inputs = json.loads((root / "inputs.json").read_text())
    for change in [{"reply_limit": 4096}, {"thinking": True}]:
        write(root / "inputs.json", {**inputs, **change})
        with pytest.raises(ValueError, match="profile"):
            audit(root, cohort)


def test_completed_reserve_receipt_cannot_exceed_original_shared_deadline(reserve):
    root, cohort = reserve
    receipt = json.loads((root / "receipt.json").read_text())
    write(root / "receipt.json", {**receipt, "finished_boot_seconds": 7700.})
    with pytest.raises(ValueError, match="combined experiment time"):
        audit(root, cohort)


def test_parity_checks_actual_values_rather_than_saved_claim(reserve):
    root, cohort = reserve
    write(root / "parity-v1.json", {"policy_version": 1, "actor": [-.5], "rollout": [-1.], "max_abs_delta": 0})
    with pytest.raises(ValueError, match="parity"):
        audit(root, cohort)
