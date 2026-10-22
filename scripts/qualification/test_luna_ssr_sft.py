import copy
import json

import pytest
from luna_ssr_sft import annotation_requests, balanced, inventory, recorded_perfect, task_names, validate_annotation
from prepare_luna_ssr_pilot import SELECTION, scope_rationale_schemas, service


def test_pilot_balances_domains_and_simple_tools_cover_distinct_services():
    from collections import Counter

    assert len(SELECTION) == 14
    assert set(Counter(name.split(".")[0] for name in SELECTION).values()) == {2}
    assert service("google_calendar_find_event") == "google_calendar"


def test_schema_scoping_preserves_observations_and_marks_missing_tools():
    tools = [{"function": {"name": name}} for name in ("search", "unused")]
    actions = [
        {"call": {"name": "search", "arguments": {}}, "result": "observed"},
        {"call": {"name": "missing", "arguments": {}}, "result": "later"},
    ]
    original = annotation_requests("task", {}, actions, {}, tools)
    scoped = scope_rationale_schemas(copy.deepcopy(original), tools)
    assert scoped[0]["prefix_evidence"][1]["value"]["relevant_native_definitions"] == [tools[0]]
    assert scoped[1]["prefix_evidence"][1]["value"]["missing_native_definitions"] == ["missing"]
    assert scoped[1]["prefix_evidence"][2:] == original[1]["prefix_evidence"][2:]
    assert scoped[1]["target_action"] == original[1]["target_action"]


def test_nested_reserved_evaluation_groups_are_excluded():
    assert task_names({"heldout": ["finance.a"], "difficulty_tiered": ["support.b"]}) == {"finance.a", "support.b"}


def episode(task="finance.a", identifier="e1", passed=True):
    trace = {
        "ok": True,
        "task": {"data": {"name": task}},
        "agent": {"config": {"model": "gpt-6-luna"}},
        "metrics": {
            "task_completed_correctly": int(passed),
            "assertions_passed": int(passed),
            "assertions_scored": 1,
            "assertions_excluded": 0,
        },
        "rewards": {"partial_credit": {"score": int(passed)}},
        "info": {},
    }
    return {"id": identifier, "ok": True, "traces": [trace]}


def save(root, directory, value):
    path = root / directory / "episode.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))


def test_success_requires_perfect_score_and_error_free_native_record():
    value = episode()
    trace = value["traces"][0]
    assert recorded_perfect(value, trace)
    trace["rewards"]["partial_credit"]["score"] = 0.9
    assert not recorded_perfect(value, trace)
    trace["rewards"]["partial_credit"]["score"] = 1
    trace["assessment_errors"] = [{"error": "missing goal"}]
    assert not recorded_perfect(value, trace)


def test_omitted_legacy_checks_are_visible_and_not_replay_qualification(tmp_path):
    value = episode()
    value["traces"][0]["metrics"]["assertions_excluded"] = 4
    save(tmp_path, "one", value)
    [row] = inventory([tmp_path], {}, set())["tasks"]
    assert row["one_perfect_source"]  # Native recorded score, not complete-check certification.
    assert row["preferred_perfect_source"]["assertions_excluded"] == 4
    assert not row["preferred_perfect_source"]["capture"]["state_payloads_present"]
    assert not row["sft_qualified"]


def test_one_perfect_demo_does_not_require_all_attempts_and_copies_are_not_attempts(tmp_path):
    save(tmp_path, "one", episode())
    save(tmp_path, "copy", episode())
    save(tmp_path, "failed", episode(identifier="e2", passed=False))
    [row] = inventory([tmp_path], {}, {"finance.a"})["tasks"]
    assert row["attempts"] == 2 and row["perfect_attempts"] == 1
    assert row["recorded_pass_rate"] == 0.5
    assert row["one_perfect_source"] and not row["all_attempts_perfect"]
    assert row["preferred_perfect_source"]["episode_id"] == "e1"


def test_eval_task_never_selected_even_when_perfect(tmp_path):
    save(tmp_path, "one", episode())
    rows = inventory([tmp_path], {"finance.a": ["heldout fixture"]}, set())["tasks"]
    result = balanced(rows, "one_perfect_source", 2)
    assert not result["candidate_tasks"]
    assert result["shortfall"]["finance"] == 2


def test_balance_never_duplicates_tasks_to_fill_categories():
    rows = [
        {
            "task": "finance.a",
            "domain": "finance",
            "attempts": 1,
            "one_perfect_source": True,
            "excluded_from_training": False,
        }
    ]
    result = balanced(rows, "one_perfect_source", 4)
    assert result["candidate_tasks"] == ["finance.a"]
    assert not result["balanced_complete"]
    assert result["shortfall"]["support"] == 4


def requests():
    actions = [
        {"call": {"name": "search", "arguments": {"q": "policy"}}, "result": {"id": "retrieved-id"}},
        {"call": {"name": "read", "arguments": {"id": "retrieved-id"}}, "result": "future-secret"},
    ]
    return annotation_requests("Read the policy", {"correct_answer": "teacher-secret"}, actions, [])


def test_action_prefix_never_contains_its_own_or_future_result():
    first, second = requests()
    assert "retrieved-id" not in json.dumps(first["prefix_evidence"])
    assert "retrieved-id" in json.dumps(second["prefix_evidence"])
    assert "future-secret" not in json.dumps(second["prefix_evidence"])
    assert "teacher-secret" not in json.dumps(second["prefix_evidence"])
    assert second["target_action"]["arguments"] == {"id": "retrieved-id"}


@pytest.mark.parametrize("reference", ["observation:0", "observation:99", "correct_answer"])
def test_initial_action_cannot_cite_future_or_privileged_evidence(reference):
    with pytest.raises(ValueError, match="future/privileged"):
        validate_annotation(
            requests()[0],
            {"supported": True, "action_index": 0, "rationale": "Retrieve the policy", "evidence_refs": [reference]},
        )


def test_valid_citations_do_not_establish_semantics_or_native_success():
    request = requests()[0]
    response = {
        "supported": True,
        "action_index": 0,
        "rationale": "Search for the requested policy",
        "evidence_refs": ["task"],
    }
    result = validate_annotation(request, response)
    assert result["format_and_reference_scope_valid"]
    assert not result["semantic_grounding_verified"] and not result["sft_qualified"]
    modified = copy.deepcopy(response)
    modified["replacement_action"] = {"name": "delete"}
    with pytest.raises(ValueError, match="replace"):
        validate_annotation(request, modified)
