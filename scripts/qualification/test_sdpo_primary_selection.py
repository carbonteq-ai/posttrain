"""Selection rewards native gains and preservation, never loss or reserve outcomes."""

import copy

import pytest
from sdpo_primary_selection import candidate, choose


def fixture(step=10):
    def metric(values):
        return {"complete_task_means": values, "full_population_task_mean": sum(values.values()) / len(values)}
    populations, phases = [], []
    for current in (0, step):
        for group in ("train", "development"):
            values = {"solved": 1., "hard": .2 if current == 0 or group == "development" else .6}
            summary = {"metrics": {m: metric(values) for m in ("task_completed_correctly", "partial_credit")}}
            summary["metrics"]["tool_mistakes"] = metric({"solved": 0., "hard": 1.})
            sha = f"{current}-{group}"
            populations.append({"step": current, "cohort": group, "summary": summary, "native_sha256": sha})
            phases.append({"phase": f"step-{current:04d}/{group}/native/traces.jsonl", "native_sha256": sha,
                "totals": {"unknown_initial_checks": 0, "preservation_checks": 1, "broken_preservation_checks": 0},
                "episodes": [{"task": "hard", "broken": []}]})
    metadata = {"budget_slot": 7, "context_limit": 16384, "reply_limit": 2048,
                "thinking": False, "tool_parser_mode": "schema-bound-v1"}
    review = {"improved_native_behavior_verified": True, "evidence_sha256": "a" * 64,
              "native_episode_ids": ["actual-native-episode"]}
    return {"completed": True, "populations": populations}, {"phases": phases}, metadata, review


def test_native_gain_with_same_development_and_no_added_guard_is_eligible():
    a, g, m, r = fixture()
    selected = candidate(a, g, 10, m, r)
    assert selected["eligible"]
    frozen = choose([selected], "reserve-digest")
    assert frozen["checkpoint_step"] == 10 and frozen["frozen_before_reserve"]


def test_lower_loss_alone_cannot_qualify_and_missing_scores_remain_missing():
    a, g, m, r = fixture()
    for p in a["populations"]:
        if p["step"] == 10:
            p["summary"] = copy.deepcopy(next(x["summary"] for x in a["populations"]
                                             if x["step"] == 0 and x["cohort"] == p["cohort"]))
    assert not candidate(a, g, 10, m, r)["eligible"]
    a["populations"][-1]["summary"]["metrics"]["partial_credit"]["full_population_task_mean"] = None
    assert any("missing_score" in reason for reason in candidate(a, g, 10, m, r)["reasons"])


def test_equal_harm_count_cannot_hide_a_new_violation_identity():
    a, g, m, r = fixture()
    base = next(p for p in g["phases"] if p["phase"].startswith("step-0000/development"))
    last = next(p for p in g["phases"] if p["phase"].startswith("step-0010/development"))
    for p, who in [(base, "person-A"), (last, "person-B")]:
        p["totals"]["broken_preservation_checks"] = 1
        p["episodes"][0]["broken"] = [{"spec": {"type": "do_not_contact", "person": who}}]
    assert any("added_preservation" in reason for reason in candidate(a, g, 10, m, r)["reasons"])


def test_missing_guard_count_is_not_treated_as_zero_harm():
    a, g, m, r = fixture()
    g["phases"][-1]["totals"].pop("unknown_initial_checks")
    assert not candidate(a, g, 10, m, r)["eligible"]


def test_predeclared_earlier_checkpoint_tiebreak_does_not_pick_training_peak():
    a, g, m, r = fixture(10)
    first = candidate(a, g, 10, m, r)
    a, g, m, r = fixture(15)
    later = candidate(a, g, 15, m, r)
    assert choose([later, first], "reserve-digest")["checkpoint_step"] == 10


def test_reserve_data_cannot_be_used_to_choose_a_primary():
    a, g, m, r = fixture()
    a["populations"].append({"cohort": "reserve"})
    with pytest.raises(ValueError, match="reserve outcomes"):
        candidate(a, g, 10, m, r)
