import json

from audit_luna_solver_pilot import audit, observation_flags


def test_only_complete_assessments_are_deduplicated_and_missing_stays_missing(tmp_path):
    root = tmp_path / "4b/control/native"
    root.mkdir(parents=True)
    assessment = {
        "assessment_id": "a1",
        "signal": {"signal_id": "goal", "direction": "higher"},
        "status": "valid",
        "value": 1,
    }
    trace = {
        "assessment_batches": [
            {"run": {"status": "partial"}, "assessments": [{**assessment, "assessment_id": "partial"}]},
            {"run": {"status": "complete"}, "assessments": [assessment]},
            {"run": {"status": "complete"}, "assessments": [assessment]},
        ]
    }
    episode = {"id": "e1", "task": {"data": {"name": "task1"}}, "traces": [trace]}
    (root / "traces.jsonl").write_text(json.dumps(episode) + "\n")
    value = audit(tmp_path)["models"]
    assert value["4b"]["observed_episodes"] == 1
    assert len(value["4b"]["episodes"][0]["native_assessments"]) == 1
    assert value["4b"]["episodes"][0]["traces"][0]["metrics"] is None
    assert value["9b"]["observed_episodes"] == 0
    assert value["9b"]["completed_receipt"] is False


def test_creation_claim_in_same_batch_is_flagged_even_if_tools_score_success():
    rows = [
        {
            "clean_student_request": {"messages": []},
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "hubspot_create_deal", "arguments": "{}"}},
                                {
                                    "function": {
                                        "name": "slack_send_channel_message",
                                        "arguments": json.dumps({"text": "A deal has been created."}),
                                    }
                                },
                            ]
                        }
                    }
                ]
            },
        }
    ]
    assert observation_flags(rows) == [
        {"kind": "creation_claim_in_creation_batch", "tool": "slack_send_channel_message"}
    ]
