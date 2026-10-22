import pytest
from select_self_reserve_primary import choose


def row(name, **changes):
    value = {"name": name, "experiment": 9, "step": 10, "complete_scores": True, "native_errors": 0,
             "unknown_guard_checks": 0, "preservation_checks": 14, "broken_preservation_checks": 0,
             "retention_success": 1, "retention_partial": 1, "retention_agent_completed": 8,
             "retention_repeats": 0, "qualified": False, "success": .5, "partial": .6, "failure_repeats": 0}
    value.update(changes)
    return value


def test_unsafe_high_partial_or_missing_coverage_cannot_be_primary():
    result = choose([row("safe"), row("unsafe", success=1, partial=1, broken_preservation_checks=1),
                     row("missing", success=1, partial=1, complete_scores=False)])
    assert result["primary"] == "safe" and result["primary_qualified"] is False


def test_predeclared_qualification_and_earlier_step_tie():
    result = choose([row("unqualified", success=1), row("qualified10", qualified=True),
                     row("qualified5", qualified=True, step=5)])
    assert result["primary"] == "qualified5"


def test_no_safe_complete_primary_is_not_silently_relaxed():
    with pytest.raises(ValueError, match="no eligible"):
        choose([row("unknown", unknown_guard_checks=1), row("loops", retention_repeats=1)])
