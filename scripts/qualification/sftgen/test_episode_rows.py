"""episode_rows.py on a real teacher-in-runner episode (mock teacher, pinned environment 86994309, world 2026-10-28):
what the teacher saw becomes the row, with reasoning per turn and parallel calls kept."""
import json
from pathlib import Path

import episode_rows

HERE = Path(__file__).resolve().parent / "fixtures" / "teacher_e2e"


def test_real_episode_becomes_an_exact_row():
    rows, skipped = episode_rows.build(HERE / "run", HERE / "proxy", model="qwen3.5-397b-a17b", prefix="t")
    assert skipped == [] and len(rows) == 1
    row = rows[0]
    meta = row["metadata"]
    assert meta["task"] == "operations.notion_gmail_checklist_update" and meta["exact"] is True
    assert meta["task_completed_correctly"] == 1.0 and meta["stop_condition"] == "agent_completed"
    assert meta["environment"].startswith("86994309") and meta["eval_code"].startswith("77cabd08")
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "tool", "assistant"]
    assistants = [m for m in row["messages"] if m["role"] == "assistant"]
    script = json.loads((HERE / "mock_script.json").read_text())["turns"]
    assert [a["reasoning_content"] for a in assistants] == [t["reasoning"] for t in script]
    assert [len(a.get("tool_calls") or []) for a in assistants] == [1, 1, 2, 0]
    assert isinstance(assistants[0]["tool_calls"][0]["function"]["arguments"], dict)
    assert list(assistants[0]["tool_calls"][0]["function"]["arguments"]) == ["query", "max_results"]
    assert row["trainable_message_indices"] == [2, 4, 6, 9]
    assert all(t["type"] == "function" for t in row["tools"]) and len(row["tools"]) == 6


def test_thinking_gates_on_the_real_row(monkeypatch):
    import validate_row

    monkeypatch.setattr(validate_row, "THINKING", True)
    row = episode_rows.build(HERE / "run", HERE / "proxy", model="m", prefix="t")[0][0]
    report = validate_row.validate(row, replay=False)
    assert report["gates"]["reasoning_every_turn"]["pass"] and report["gates"]["grounding"]["pass"]
    broken = json.loads(json.dumps(row))
    broken["messages"][4]["reasoning_content"] = ""
    gate = validate_row.validate(broken, replay=False)["gates"]["reasoning_every_turn"]
    assert not gate["pass"] and gate["turns_without_reasoning"] == [4]


def test_thinking_grounding_uses_reasoning_and_tool_definitions_allow_their_words(monkeypatch):
    import validate_row

    monkeypatch.setattr(validate_row, "THINKING", True)
    row = episode_rows.build(HERE / "run", HERE / "proxy", model="m", prefix="t")[0][0]
    row = json.loads(json.dumps(row))
    last = row["messages"][-1]
    last["content"] += " Usage was 66.67% of the effective budget."
    assert not validate_row.validate(row, replay=False)["gates"]["grounding"]["pass"]
    row["messages"][4]["reasoning_content"] += " Usage: $10,000 / $15,000 = 66.67%."
    assert validate_row.validate(row, replay=False)["gates"]["grounding"]["pass"]
    row["messages"][4]["reasoning_content"] += " The add-row tool follows the rubric."
    assert not validate_row.validate(row, replay=False)["gates"]["no_harness_words"]["pass"]
    row["tools"][0]["function"]["description"] += " Rows follow the team rubric."
    assert validate_row.validate(row, replay=False)["gates"]["no_harness_words"]["pass"]


def test_quarantine_refuses_evaluation_tasks(tmp_path):
    import shutil

    import pytest

    run = tmp_path / "run"
    shutil.copytree(HERE / "run", run)
    summary = json.loads((run / "summary.json").read_text())
    summary["tasks"] = summary["tasks"] + [sorted(episode_rows.evaluation_tasks())[0]]
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="evaluation tasks"):
        episode_rows.build(run, HERE / "proxy", model="m", prefix="t")


def test_row_ids_must_be_canonical():
    import pytest

    with pytest.raises(ValueError, match="not canonical"):
        episode_rows.build(HERE / "run", HERE / "proxy", model="m", prefix="Not-Canonical")


def test_validate_refuses_a_row_the_cross_check_did_not_confirm():
    import validate_row

    row = episode_rows.build(HERE / "run", HERE / "proxy", model="m", prefix="t")[0][0]
    assert validate_row.validate(row, replay=False)["gates"]["teacher_request_exact"]["pass"]
    row["metadata"]["exact"] = False
    report = validate_row.validate(row, replay=False)
    assert not report["gates"]["teacher_request_exact"]["pass"] and not report["pass"]


def test_malformed_call_arguments_skip_the_episode(tmp_path):
    import gzip
    import shutil

    run = tmp_path / "run"
    shutil.copytree(HERE / "run", run)
    records = [json.loads(line) for line in gzip.open(run / "episodes.jsonl.gz", "rt")]
    for node in records[0]["traces"][0]["nodes"]:
        for call in (node.get("message") or {}).get("tool_calls") or []:
            call["arguments"] = '{"type": 2, "id": zoom_noise_011}'
    with gzip.open(run / "episodes.jsonl.gz", "wt") as stream:
        stream.write("".join(json.dumps(r) + "\n" for r in records))
    rows, skipped = episode_rows.build(run, HERE / "proxy", model="m", prefix="t")
    assert rows == [] and skipped and skipped[0]["reason"].startswith("malformed call arguments")


def _teacher_row():
    import validate_row

    return validate_row, json.loads(json.dumps(episode_rows.build(HERE / "run", HERE / "proxy", model="m", prefix="t")[0][0]))


def test_teacher_rows_allow_everyday_words_but_not_evaluation_words(monkeypatch):
    validate_row, row = _teacher_row()
    monkeypatch.setattr(validate_row, "THINKING", True)
    row["messages"][4]["reasoning_content"] += " I will send a brief follow-up after the checks."
    assert validate_row.validate(row, replay=False)["gates"]["no_harness_words"]["pass"]
    row["messages"][4]["reasoning_content"] += " The grader wants this."
    assert not validate_row.validate(row, replay=False)["gates"]["no_harness_words"]["pass"]


def test_identifiers_copied_from_context_are_grounded(monkeypatch):
    validate_row, row = _teacher_row()
    monkeypatch.setattr(validate_row, "THINKING", True)
    tool = next(m for m in row["messages"] if m["role"] == "tool")
    tool["content"] += " sensor HVAC-NW-303 offline"
    row["messages"][-1]["content"] += " Checked HVAC-NW-303."
    assert validate_row.validate(row, replay=False)["gates"]["grounding"]["pass"]
    row["messages"][-1]["content"] += " Also HVAC-NW-404."
    assert not validate_row.validate(row, replay=False)["gates"]["grounding"]["pass"]


def test_failed_calls_are_allowed_only_in_masked_turns():
    validate_row, row = _teacher_row()
    first_tool = next(k for k, m in enumerate(row["messages"]) if m["role"] == "tool")
    row["messages"][first_tool]["content"] = '{"success": false, "error": "Worksheet \'Policy\' not found"}'
    issuer = first_tool - 1
    gates = validate_row.validate(row, replay=False)["gates"]
    assert not gates["zero_tool_errors"]["pass"]
    row["metadata"]["masked_turns"] = [issuer]
    row["trainable_message_indices"] = [k for k in row["trainable_message_indices"] if k != issuer]
    gates = validate_row.validate(row, replay=False)["gates"]
    assert gates["zero_tool_errors"]["pass"] and gates["structure"]["pass"], gates["zero_tool_errors"]
    clean = next(k for k in row["trainable_message_indices"] if row["messages"][k].get("tool_calls"))
    row["metadata"]["masked_turns"] = [issuer, clean]  # masking a clean turn without a reason is refused
    row["trainable_message_indices"] = [k for k in row["trainable_message_indices"] if k != clean]
    assert not validate_row.validate(row, replay=False)["gates"]["zero_tool_errors"]["pass"]
    row["metadata"]["masked_reasons"] = {str(clean): "names left-out items in a message"}  # ... allowed with a reason
    assert validate_row.validate(row, replay=False)["gates"]["zero_tool_errors"]["pass"]
    row["metadata"]["masked_reasons"] = {str(clean): "wrong action, corrected later"}
    assert validate_row.validate(row, replay=False)["gates"]["zero_tool_errors"]["pass"]
    row["metadata"]["masked_reasons"] = {str(clean): "looked odd"}  # ... but not with any other reason
    assert not validate_row.validate(row, replay=False)["gates"]["zero_tool_errors"]["pass"]
    del row["metadata"]["masked_reasons"]
    row["metadata"]["masked_turns"] = [issuer]  # trainable indices that do not match the mask are refused
    assert not validate_row.validate(row, replay=False)["gates"]["structure"]["pass"]


def test_a3_prefix_reason_only_for_a3_rows_and_text_gates_skip_only_the_prefix(monkeypatch):
    """SDPO plan v4 A3 (13 Nov): the student's own turns before the continuation point are masked with the A3 reason;
    the reason is refused on any other row; their text gates are skipped, every trained turn keeps every gate."""
    validate_row, row = _teacher_row()
    monkeypatch.setattr(validate_row, "THINKING", True)
    prefix = next(k for k in row["trainable_message_indices"] if row["messages"][k].get("tool_calls"))
    row["messages"][prefix]["reasoning_content"] = ""  # a student slip: no reasoning, an invented number
    row["messages"][prefix]["content"] = "Paying 98765 now."
    row["metadata"]["masked_turns"] = [prefix]
    row["metadata"]["masked_reasons"] = {str(prefix): validate_row.A3_PREFIX_REASON}
    row["trainable_message_indices"] = [k for k in row["trainable_message_indices"] if k != prefix]
    gates = validate_row.validate(row, replay=False)["gates"]  # not an A3 row: the reason is refused, gates apply
    assert not gates["zero_tool_errors"]["pass"]
    assert not gates["reasoning_every_turn"]["pass"] and not gates["grounding"]["pass"]
    row["metadata"]["source"] = validate_row.A3_SOURCE
    gates = validate_row.validate(row, replay=False)["gates"]
    assert gates["zero_tool_errors"]["pass"] and gates["structure"]["pass"]
    assert gates["reasoning_every_turn"]["pass"] and gates["grounding"]["pass"], gates["grounding"]
    row["messages"][-1]["content"] += " Total 55555."  # a trained turn still fails its own gates
    assert not validate_row.validate(row, replay=False)["gates"]["grounding"]["pass"]
