"""Source-state and current-tool checks must not silently translate demonstrations."""

import json

from inventory_sdpo_luna_pool124 import digest, inspect


def fixture():
    data = {"initial_state": {"value": 1}, "assertions": [{"type": "test"}],
            "prompt": [{"role": "user", "content": "original"}]}
    episode = {"ok": True, "traces": [{"task": {"data": data},
        "metrics": {"task_completed_correctly": 1}, "info": {"automationbench_capture": {
            "events": [{"tool_name": "execute_tool", "arguments_json": '{"tool_name":"write_row"}'}],
            "snapshots": {"hash": "{}"}}}}]}
    binding = {"initial_state_sha256": digest(data["initial_state"]),
               "assertions_sha256": digest(data["assertions"]),
               "user_messages_sha256": digest(["original"]), "available_tools": ["write_row"]}
    return episode, binding


def test_bound_reference_is_still_unqualified_pending_native_review():
    episode, binding = fixture()
    result = inspect(episode, binding)
    assert result["candidate"] and result["current_native_checks_qualified"] is False


def test_changed_state_or_unavailable_tool_is_not_silently_adapted():
    episode, binding = fixture()
    assert not inspect(episode, binding | {"initial_state_sha256": "different"})["candidate"]
    result = inspect(episode, binding | {"available_tools": ["different_tool"]})
    assert not result["candidate"] and result["unavailable_tools"] == ["write_row"]


def test_missing_snapshot_placeholder_is_not_retained_state_evidence():
    episode, binding = fixture()
    capture = episode["traces"][0]["info"]["automationbench_capture"]
    capture["snapshots"] = "payload omitted by archive"
    assert inspect(episode, binding)["reason"] == "unavailable_snapshot_payload"
    capture["snapshots"] = json.dumps({"hash": "{}"})
    assert inspect(episode, binding)["candidate"]
