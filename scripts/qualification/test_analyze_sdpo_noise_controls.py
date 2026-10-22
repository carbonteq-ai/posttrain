import copy

import pytest
from analyze_sdpo_noise_controls import compare
from sdpo_agent_config import MODEL_REVISION


def population(label, step, value, names):
    episodes = [{"id": f"{label}-{name}-{rep}", "task": {"data": {"name": name}}, "traces": [{
        "stop_condition": "agent_completed", "metrics": {"task_completed_correctly": value},
        "rewards": {"partial_credit": {"score": value}}, "nodes": []}]} for name in names for rep in range(5)]
    return {"episodes": episodes, "model_revision": MODEL_REVISION, "runtime_key": "same-eager-MTP-runtime",
            "adapter_sha256": "base" if step == 0 else "student", "policy_version": step}


def fixture():
    names = [f"task{i}" for i in range(62)]
    return names, population("a", 0, 0., names), population("b", 0, 0., names), population("s", 10, 1., names)


def test_separate_zero_update_drift_and_two_learner_comparisons():
    names, a, b, s = fixture()
    result = compare(a, b, s, names, bootstrap_repeats=100)
    c = result["metrics"]["task_completed_correctly"]["comparisons"]
    assert c["zero_update_base_b_minus_a"]["joint_complete_task_sensitivity"]["delta"] == 0.
    assert c["student_minus_base_a"]["joint_complete_task_sensitivity"]["delta"] == 1.
    assert c["student_minus_base_b"]["joint_complete_task_sensitivity"]["delta"] == 1.
    assert result["strict_runtime_qualification_granted"] is False
    assert result["automatic_learning_verdict"] is None


def test_missing_outcomes_use_joint_tasks_and_nonzero_width_bounds():
    names, a, b, s = fixture()
    for e in s["episodes"][:5]:
        e["traces"][0]["metrics"] = {}
    result = compare(a, b, s, names, bootstrap_repeats=100)
    metric = result["metrics"]["task_completed_correctly"]
    assert metric["joint_complete_tasks"] == 61
    assert metric["zero_update_control_pair"]["tasks"] == 62
    assert metric["comparisons"]["student_minus_base_a"]["full_population_gain_identification_bounds"] == [61/62, 1.]
    assert result["summaries"][2]["metrics"]["task_completed_correctly"]["full_population_task_mean"] is None
    assert s["episodes"][0]["traces"][0]["metrics"] == {}


def test_reused_episode_ids_are_not_independent_controls():
    names, a, _, s = fixture()
    with pytest.raises(ValueError, match="independent complete"):
        compare(a, copy.deepcopy(a), s, names, bootstrap_repeats=100)


def test_different_runtime_or_base_weights_cannot_measure_zero_update_drift():
    names, a, b, s = fixture()
    b["runtime_key"] = "compiled"
    with pytest.raises(ValueError, match="unmatched"):
        compare(a, b, s, names, bootstrap_repeats=100)
    b["runtime_key"] = a["runtime_key"]
    b["adapter_sha256"] = "other-weights"
    with pytest.raises(ValueError, match="same untrained"):
        compare(a, b, s, names, bootstrap_repeats=100)
