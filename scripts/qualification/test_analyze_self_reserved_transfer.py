import hashlib
import json

import pytest
from analyze_self_reserved_transfer import audit, score_bounds
from self_campaign_config import summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def experiment(tmp_path):
    names = [f"task-{i}" for i in range(24)]
    candidates = [{"name": n, "adapter_sha256": n} for n in ["fresh_base", "checkpoint64", "primary"]]
    selection = {"candidates": candidates, "primary": "primary"}
    write(tmp_path / "selection-before-outcomes.json", selection)
    write(tmp_path / "selection-sha256.json", {"sha256": hashlib.sha256((tmp_path / "selection-before-outcomes.json").read_bytes()).hexdigest()})
    write(tmp_path / "population.json", {"tasks": names, "sorted_names_sha256": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()})
    write(tmp_path / "receipt.json", {"completed": True, "optimizer_updates": 0, "external_calls": 0, "candidates": 3, "elapsed_s": 300, "primary": "primary"})
    write(tmp_path / "inputs.json", {"context_limit": 16384, "reply_limit": 2048, "episode_output_limit": 12288,
          "turn_limit": 8, "server_sequences": 32, "episode_concurrency": 32, "kv_cache_gib": 16,
          "temperature": 0., "thinking": False, "optimizer_updates": 0})
    write(tmp_path / "zero-base-gate.json", {"all_B_zero": True, "only_finite_A_B": True, "configuration_checked": True,
          "actual_loaded_state_verified": True, "active_vs_disabled_max_abs_delta": 0})
    write(tmp_path / "candidate-provenance.json", [{"name": c["name"], "tensor_equality_verified": True,
          "candidate_sha256": c["name"], "exported_sha256": c["name"]} for c in candidates])
    for step, c in enumerate(candidates):
        write(tmp_path / f"parity-v{step}.json", {"policy_version": step, "max_abs_delta": .01})
        folder = tmp_path / f"step-{step:04d}/reserve_transfer"
        episodes = [{"id": f"{step}-{n}-{r}", "task": {"data": {"name": n, "initial_state": {"value": n}}},
                     "traces": [{"stop_condition": "agent_completed", "nodes": [],
                     "metrics": {"task_completed_correctly": float(step == 2)},
                     "rewards": {"partial_credit": {"score": float(step == 2)}}}]} for n in names for r in range(2)]
        write(folder / "outcomes.json", summarize(episodes, names, 2))
        (folder / "native").mkdir()
        (folder / "native/traces.jsonl").write_text("\n".join(map(json.dumps, episodes)))
        write(folder / "evaluation-admission.json", {"training_allowed": False, "optimizer_updates": 0,
              "context_rejected_calls": 0, "incomplete_generations": 0})
        write(folder / "generation-calls.jsonl", {"policy_version": step, "adapter_sha256": c["name"], "complete": True, "finish_reason": "stop"})
    return tmp_path


def test_complete_preselected_comparison_and_separate_uncertainty(experiment):
    result = audit(experiment)
    assert result["episodes"] == 144 and result["primary_comparison"] == "fresh_base_to_primary"
    paired = result["comparisons"]["fresh_base_to_primary"]["task_completed_correctly"]["paired"]
    assert paired["delta"] == 1 and paired["paired_delta_ci95"] == [1, 1]
    assert paired["after_absolute_ci95"] == [1, 1]
    assert result["primary_gain_identification_bounds"]["task_completed_correctly"]["gain_identification_bounds"] == [1, 1]


def test_changed_initial_state_cannot_be_compared(experiment):
    path = experiment / "step-0002/reserve_transfer/native/traces.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["task"]["data"]["initial_state"] = {"different": True}
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="initial states"):
        audit(experiment)


def test_wrong_adapter_and_unverified_base_are_rejected(experiment):
    path = experiment / "zero-base-gate.json"
    gate = json.loads(path.read_text())
    gate["actual_loaded_state_verified"] = False
    write(path, gate)
    with pytest.raises(ValueError, match="fresh-base"):
        audit(experiment)
    gate["actual_loaded_state_verified"] = True
    write(path, gate)
    write(experiment / "step-0002/reserve_transfer/generation-calls.jsonl",
          {"policy_version": 2, "adapter_sha256": "wrong", "complete": True})
    with pytest.raises(ValueError, match="served adapter"):
        audit(experiment)


def test_missing_scores_remain_bounds_not_zero_filled_means():
    rows = [{"traces": [{"metrics": {"task_completed_correctly": 1.}, "rewards": {}}]},
            {"traces": [{"metrics": {}, "rewards": {}}]}]
    result = score_bounds(rows)
    assert result["task_completed_correctly"]["missing"] == 1
    assert result["task_completed_correctly"]["identification_bounds"] == [.5, 1.]
    assert result["partial_credit"]["identification_bounds"] == [0., 1.]
