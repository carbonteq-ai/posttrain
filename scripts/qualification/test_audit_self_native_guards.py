import json

from audit_self_native_guards import audit


def test_held_broken_author_excluded_and_unknown_are_distinct(tmp_path):
    specs = [{"type": n} for n in ["held", "broken", "goal", "unknown"]] + [{"type": "info", "excluded": True}]
    results = [{"type": a["type"], "params": {k: v for k, v in a.items() if k != "type"},
                "passed": a["type"] in ["held", "goal", "info"], "excluded": a["type"] in ["held", "info"]}
               for a in specs]
    e = {"id": "e", "task": {"data": {"name": "task", "initial_state": {}, "assertions": specs}},
         "traces": [{"info": {"automationbench": {"assertions": results}}, "metrics": {}}]}
    path = tmp_path / "step-0000/failures/native/traces.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(e))

    def check(state, spec):
        if spec["type"] == "unknown":
            raise ValueError("unsupported assertion")
        return spec["type"] != "goal"

    result = audit(tmp_path, check)["phases"][0]
    assert result["totals"]["preservation_checks"] == 2
    assert result["totals"]["broken_preservation_checks"] == 1
    assert result["totals"]["unknown_initial_checks"] == 1
    assert result["totals"]["newly_satisfied_checks"] == 1
    assert result["preservation_violation_episode_rate_when_applicable"] == 1
    assert result["episodes"][0]["broken"][0]["spec"] == {"type": "broken"}
