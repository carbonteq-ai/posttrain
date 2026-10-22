"""Missing-score bounds and paired cohort completeness for broad monitoring."""

import json

import pytest
from analyze_sdpo_pool124 import bounded_scores, comparable_summary, generation_evidence, paired


def episode(identity, task, success, partial):
    return {"id": identity, "task": {"data": {"name": task}}, "traces": [{
        "metrics": {} if success is None else {"task_completed_correctly": success},
        "rewards": {} if partial is None else {"partial_credit": {"score": partial}},
        "stop_condition": "agent_completed"}]}


def test_missing_observations_remain_unknown_with_separate_bounds():
    episodes = [episode("a", "one", 1, .5), episode("b", "one", None, None)]
    summary, bounds = bounded_scores(episodes, ["one"], 2)
    assert summary["metrics"]["partial_credit"]["full_population_task_mean"] is None
    assert bounds["partial_credit"]["identification_bounds"] == [.25, .75]
    assert bounds["partial_credit"]["missing"] == 1


def test_tasks_missing_in_both_arms_stay_in_excluded_population():
    episodes = [episode("a", "one", 1, 1), episode("b", "two", None, None)]
    summary, bounds = bounded_scores(episodes, ["one", "two"], 1)
    entry = {"summary": summary, "bounded_metrics": bounds}
    result = paired(entry, entry, ["one", "two"], 1)
    assert result["partial_credit"]["excluded_tasks"] == ["two"]
    assert result["partial_credit"]["full_population_gain_identification_bounds"] == [-.5, .5]
    assert "all1repetitions" in result["partial_credit"]["paired"]["unit"]


def test_native_score_outside_declared_bounds_fails_instead_of_clipping():
    with pytest.raises(ValueError):
        bounded_scores([episode("a", "one", 1, 1.2)], ["one"], 1)


def test_no_missing_scores_have_exactly_collapsed_ordered_bounds():
    episodes = [episode(str(i), "one", 1, 4 / 7) for i in range(310)]
    _, result = bounded_scores(episodes, ["one"], 310)
    lower, upper = result["partial_credit"]["identification_bounds"]
    assert lower == upper
    assert lower == pytest.approx(4 / 7)


def test_unset_error_traceback_normalization_never_drops_errors_or_scores():
    native = {"errors": [{"type": "ProviderError", "message": "failed", "status_code": 500}],
              "metrics": {"score": None}}
    saved = native | {"errors": [native["errors"][0] | {"traceback": None}]}
    assert comparable_summary(native) == comparable_summary(saved)
    assert comparable_summary(native) != comparable_summary(native | {"errors": []})
    assert comparable_summary(native) != comparable_summary(native | {"metrics": {"score": 0}})
    assert comparable_summary(native) != comparable_summary(native | {
        "errors": [native["errors"][0] | {"traceback": "actual evidence"}]})


def test_streamed_call_audit_keeps_all_limit_and_finish_events(tmp_path):
    calls = [{"policy_version": 6, "complete": True, "finish_reason": "stop"},
             {"policy_version": 6, "complete": False, "finish_reason": "length"},
             {"policy_version": 6, "complete": False, "limit_reason": "context_limit"}]
    path = tmp_path / 'calls.jsonl'
    path.write_text('\n'.join(json.dumps(c) for c in calls) + '\n')
    assert generation_evidence(path, 6) == {
        "incomplete_generations": 2, "finish_reasons": {"stop": 1, "length": 1, "not_recorded": 1},
        "context_refusals": 1}
    # Fresh early records cannot hide a stale record at the end of a population.
    calls[-1]['policy_version'] = 5
    path.write_text('\n'.join(json.dumps(c) for c in calls) + '\n')
    with pytest.raises(ValueError, match='stale native policy'):
        generation_evidence(path, 6)
    with path.open('w') as stream:
        stream.write(json.dumps(calls[0]) + '\n{"incomplete_record":')
    with pytest.raises(json.JSONDecodeError):
        generation_evidence(path, 6)
