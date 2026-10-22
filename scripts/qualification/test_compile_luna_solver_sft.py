from copy import deepcopy

import pytest
from compile_luna_solver_sft import REVIEW_CHECKS, selected_records


def example():
    episode = {"id": "e1", "task": {"data": {"name": "simple.test"}}, "ok": True,
               "traces": [{"ok": True, "metrics": {"task_completed_correctly": 1},
                           "rewards": {"partial_credit": {"score": 1}}, "stop_condition": "agent_completed"}]}
    review = {"episode_id": "e1", "task": "simple.test", "evidence": ["native nodes reviewed"],
              **dict.fromkeys(REVIEW_CHECKS, True)}
    return episode, review


def test_full_score_alone_is_not_admission_and_protected_tasks_stay_out():
    episode, review = example()
    assert not selected_records([episode], [], set())[0]
    assert not selected_records([episode], [review], {"simple.test"})[0]
    assert selected_records([episode], [review], set())[0] == [episode]


def test_reasoning_and_errors_cannot_be_overridden_by_scalar_success():
    episode, review = example()
    review["no_privileged_context_dependency"] = False
    assert not selected_records([episode], [review], set())[0]
    episode, review = example()
    episode["traces"][0]["errors"] = ["environment finalization failed"]
    assert not selected_records([episode], [review], set())[0]
    episode, review = example()
    episode["traces"][0]["stop_condition"] = "max_turns"
    assert not selected_records([episode], [review], set())[0]


def test_reviews_join_by_episode_and_task_identity():
    episode, review = example()
    with pytest.raises(ValueError):
        selected_records([episode], [review, deepcopy(review)], set())
    review["task"] = "simple.other"
    assert not selected_records([episode], [review], set())[0]
    review["episode_id"] = "unknown"
    with pytest.raises(ValueError):
        selected_records([episode], [review], set())
