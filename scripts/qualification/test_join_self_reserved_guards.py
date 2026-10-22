import copy

import pytest
from join_self_reserved_guards import join


def example():
    names = [f"task{i}" for i in range(24)]
    audit = {"population": {"tasks": names}, "selection": {"primary": "primary"},
             "candidates": [{"candidate": name, "native_sha256": name} for name in ["fresh_base", "primary"]]}
    phases = []
    for i, name in enumerate(["fresh_base", "primary"]):
        rows = [{"task": task, "initially_unsatisfied_checks": 1, "newly_satisfied_checks": i,
                 "preservation_checks": int(task == "task0"), "broken": [], "unknown": [{}] if task == "task1" else []}
                for task in names for _ in range(2)]
        phases.append({"phase": f"step-{i:04d}/reserve_transfer/native/traces.jsonl", "native_sha256": name,
                       "episodes": rows, "totals": {"episodes": 48}, "preservation_violation_episode_rate_when_applicable": 0.})
    return audit, {"phases": phases}


def test_unknown_and_no_guard_tasks_are_not_treated_as_harm_free():
    a, g = example()
    result = join(a, g)
    assert result["candidates"][0]["unknown_applicability_tasks"] == ["task1"]
    assert result["primary_vs_base"]["preservation_violation"]["applicable_complete_tasks"] == ["task0"]
    assert len(result["primary_vs_base"]["preservation_violation"]["excluded_or_not_applicable"]) == 23
    assert result["primary_vs_base"]["target_fraction"]["paired"]["delta"] == 1.


def test_different_native_file_or_repetition_coverage_rejected():
    a, g = example()
    broken = copy.deepcopy(g)
    broken["phases"][0]["native_sha256"] = "wrong"
    with pytest.raises(ValueError, match="different native"):
        join(a, broken)
    g["phases"][0]["episodes"].pop()
    with pytest.raises(ValueError, match="coverage"):
        join(a, g)
