"""Offline replay keeps benchmark oracles out of the general judge request."""

import json

from replay_episode_judge import judge_input, summarize


def test_judge_input_is_role_separated_grounded_and_oracle_free():
    record = {
        "id": "trace-1",
        "task": {
            "data": {
                "task_name": "calendar",
                "zapier_tools": ["google_calendar_create_detailed_event"],
                "assertions": [{"secret_reference": "must-not-leak"}],
                "initial_state": {"secret": "must-not-leak"},
            }
        },
        "rewards": {"partial": 1.0},
        "nodes": [
            {"message": {"role": "user", "content": "Create an all-day event."}},
            {
                "message": {
                    "role": "assistant",
                    "content": "Done",
                    "tool_calls": [],
                }
            },
        ],
    }
    item = judge_input(record)
    assert [message["role"] for message in item["messages"]] == ["system", "user"]
    request = item["assessment_request"]
    assert request["trajectory"][0]["message_id"] == "message-0"
    [tool] = request["available_tools"]
    assert tool["function"]["name"] == "google_calendar_create_detailed_event"
    assert "all_day" in tool["function"]["parameters"]["properties"]
    serialized = json.dumps(item)
    assert "must-not-leak" not in serialized
    assert '"rewards"' not in serialized


def test_summary_exposes_uniform_perfect_outputs_and_violations():
    assessments = {
        name: {"status": "valid", "score": 1.0, "reason": "x", "evidence": ["message-0"]}
        for name in (
            "problem_understanding_planning",
            "logical_correctness",
            "evidence_state_grounding",
            "verification_self_correction",
            "progress_efficiency",
            "action_quality",
            "answer_quality",
        )
    }
    result = summarize(
        [
            {
                "verdict": {
                    "assessments": assessments,
                    "requirement_checks": [
                        {"outcome": "violated_major"},
                    ],
                }
            }
        ]
    )
    assert result["all_dimensions_perfect_episodes"] == 1
    assert result["violations_by_severity"] == {"major": 1}
