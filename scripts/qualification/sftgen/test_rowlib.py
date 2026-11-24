"""Unit tests for rowlib and validate_row's pure gates (no environment, no tokenizer needed).

  CAPPED_MEM=4G .../bin/capped <any python with pytest> -m pytest scripts/qualification/sftgen/test_rowlib.py -q
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rowlib  # noqa: E402
import validate_row  # noqa: E402

PROMPT = [{"role": "system", "content": "You are a workflow automation agent. Budget of ~50 tool-using turns."},
          {"role": "user", "content": "Email the variance analysis from sheet ss_1. Check Slack before calculating."}]
TOOLS = [{"name": "sheets_get", "description": "Get rows.", "parameters": {"type": "object", "properties": {
    "spreadsheet": {"type": "string"}, "includeGridData": {"type": "boolean"}}}},
         {"name": "gmail_send_email", "description": "Send.", "parameters": {"type": "object", "properties": {
             "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}}}]
RESULT = '{"success": true, "rows": [{"Account": "SaaS", "Jan": "$185,000", "Dec": "$160,000"}]}'


def make_session(tmp_path: Path, *, reply: str | None = "I emailed the SaaS variance of $25,000 (15.6%).",
                 note2: str = "185,000 - 160,000 = 25,000, and 25,000 / 160,000 = 15.6%. I'll email it.") -> Path:
    s = tmp_path / "session"
    s.mkdir()
    turns = [
        {"turn": 1, "note": "I'll read the sheet first.", "calls": [
            {"index": 1, "tool": "sheets_get", "arguments": {"spreadsheet": "ss_1", "includeGridData": True},
             "arguments_text": "", "status": "ok", "result": RESULT}]},
        {"turn": 2, "note": note2, "calls": [
            {"index": 1, "tool": "gmail_send_email", "arguments": {"to": "c@x.com", "subject": "Variance",
                                                                   "body": "SaaS +$25,000 (+15.6%)"},
             "arguments_text": "", "status": "ok", "result": '{"success": true}'}]}]
    meta = {"task": "finance.unit_case", "tool": "abtool2", "tool_sha256": "x", "turns": 2, "calls": 2,
            "world_date": "2026-10-28", "fresh_setup": False, "world_clock": "2026-02-03T10:00:00+00:00",
            "env_commit": "bd655be7ae63395a6ae35b9f2505ff09ce8980a7"}
    (s / "meta.json").write_text(json.dumps(meta))
    (s / "prompt.json").write_text(json.dumps({"prompt": PROMPT, "tools": TOOLS}))
    (s / "turns.jsonl").write_text("".join(json.dumps(t) + "\n" for t in turns))
    if reply is not None:
        (s / "reply.json").write_text(json.dumps({"text": reply, "turn": 3}))
    return s


def row_of(tmp_path: Path, **kw) -> dict:
    return rowlib.session_row(rowlib.read_session(make_session(tmp_path, **kw)), model="sonnet", attempt=1)


# ----------------------------------------------------------------------------- mapping
def test_session_row_shape_and_key_order(tmp_path):
    row = row_of(tmp_path)
    assert list(row) == ["id", "messages", "tools", "trainable_message_indices", "metadata"]
    assert row["id"] == "sft-route-b/sonnet/finance.unit_case"
    m = row["messages"]
    assert m[:2] == PROMPT
    assert [x["role"] for x in m] == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert list(m[2]) == ["role", "content", "tool_calls"]
    assert m[2]["content"] == "I'll read the sheet first."
    call = m[2]["tool_calls"][0]
    assert call["id"] == "call_1_1" and list(call["function"]["arguments"]) == ["spreadsheet", "includeGridData"]
    assert m[3] == {"role": "tool", "tool_call_id": "call_1_1", "name": "sheets_get", "content": RESULT}
    assert m[-1] == {"role": "assistant", "content": "I emailed the SaaS variance of $25,000 (15.6%)."}
    assert row["trainable_message_indices"] == [2, 4, 6]
    assert row["tools"][1] == {"type": "function", "function": TOOLS[1]}
    md = row["metadata"]
    assert md["model"] == "sonnet" and md["attempt"] == 1 and md["env_commit"].startswith("bd655be")
    assert md["replay_result"] == "not run" and md["source"] == "teacher-route-b"
    assert rowlib.row_calls(row)[1]["arguments"]["to"] == "c@x.com"
    assert rowlib.serialize_row(row) == json.dumps(row) + "\n"


def test_session_without_reply_gives_no_row(tmp_path):
    with pytest.raises(ValueError, match="no final reply"):
        row_of(tmp_path, reply=None)


# ----------------------------------------------------------------------------- arithmetic and grounding
def test_expressions_are_evaluated():
    found = rowlib.expressions("SaaS: 185,000 - 160,000 = 25,000, and 25,000 / 160,000 = 15.6%. "
                               "Wrong: 48,000 - 35,000 = 12,000. Rate (24,000 - 2,000) / 60 = 366.67; 3 x 4 = 12")
    assert [(e["rhs"], e["ok"]) for e in found] == [("25,000", True), ("15.6%", True), ("12,000", False),
                                                     ("366.67", True), ("12", True)]


def test_grounding_rules():
    context = rowlib.number_values(RESULT)
    ok = rowlib.grounding("185,000 - 160,000 = 25,000, so 25,000 / 160,000 = 15.6%.", context)
    assert ok["ok"], ok
    bad = rowlib.grounding("The variance is 26,000.", context)
    assert not bad["ok"] and bad["ungrounded"] == ["26,000"]
    wrong = rowlib.grounding("185,000 - 160,000 = 26,000.", context)
    assert not wrong["ok"] and wrong["wrong_expressions"]
    rounded = rowlib.grounding("(24,000 - 2,000) / 60 = 366.67, rounded to 367; 8,800 + 367 = 9,167.",
                               rowlib.number_values("24000 2000 60 8800"))
    assert rounded["ok"], rounded
    dates = rowlib.grounding("Today is 2026-01-28 and 2026-01-28 minus 30 days = 2025-12-29.",
                             rowlib.number_values("2026-01-28T10:00 30 days"))
    assert dates["ok"], dates
    wrong_date = rowlib.grounding("2026-01-28 minus 30 days = 2025-12-28.", rowlib.number_values("2026-01-28 30"))
    assert not wrong_date["ok"]
    assert rowlib.number_values("$1,234.50 and 0012") == {Decimal("1234.5"), Decimal("12")}


def test_grounding_chains_steps_percentages_markers_and_dates():
    context = rowlib.number_values("Deal Value 50,000; rate 9%; bonus $500. Rate table: 0.06")
    # a chain is checked pair by pair; the intermediate 4,500 (= 50,000 x 0.09) is an operation of the written sum
    chain = rowlib.grounding("Commission: ($50,000 × 0.09) + $500 = $4,500 + $500 = **$5,000**", context)
    assert chain["ok"], chain
    # a percentage grounds its fraction and the other way round
    assert rowlib.grounding("The rate is 0.09 and the table rate is 6%.", context)["ok"]
    # ordered-list markers are layout; other small numbers are not exempt
    assert rowlib.grounding("1. Read the deals\n2. Email the reps", set())["ok"]
    assert rowlib.grounding("Email the 2 reps", set())["ungrounded"] == ["2"]
    # an ISO date is never read as a subtraction, so a day count after it is simply ungrounded
    dated = rowlib.grounding("Last day 2026-04-25 = 10 days away.", rowlib.number_values("2026-04-25"))
    assert dated["wrong_expressions"] == [] and dated["ungrounded"] == ["10"]
    wrong = rowlib.grounding("($80,000 × 0.12) + $500 = $9,600 + $500 = $10,600", rowlib.number_values("80,000 0.12 500"))
    assert not wrong["ok"] and wrong["wrong_expressions"] and "10,600" not in wrong["ungrounded"]


def test_harness_words_allow_words_the_task_uses():
    allowed = rowlib.words_in(PROMPT[1]["content"])
    assert rowlib.harness_hits("I'll check Slack first.", set()) == []  # the verb is ordinary prose
    assert rowlib.harness_hits("All checks should pass now.", set()) == ["checks"]
    assert rowlib.harness_hits("All checks should pass now.", rowlib.words_in("Deposit the two checks.")) == []
    assert set(rowlib.harness_hits("Running the bash command with JSON for the session.", allowed)) == \
        {"bash", "command", "json", "session"}
    assert rowlib.harness_hits("Done, see turns.jsonl", allowed) == ["turns.jsonl"]


# ----------------------------------------------------------------------------- gates without replay or tokenizer
def gates(row):
    return validate_row.validate(row, replay=False)["gates"]


def test_clean_row_passes_pure_gates(tmp_path):
    g = gates(row_of(tmp_path))
    for name in ("structure", "note_before_calls", "grounding", "no_harness_words", "final_reply", "reply_names_no_skipped_items",
                 "zero_tool_errors", "no_identical_call_5", "call_numbers_shown"):
        assert g[name]["pass"], (name, g[name])


def test_gate_failures_are_reported_separately(tmp_path):
    row = row_of(tmp_path, note2="I'll email it now.", reply="Sent; I skipped the Dec rows in this session. Total 99.")
    g = gates(row)
    assert g["grounding"]["pass"] is False and g["grounding"]["ungrounded"][0]["numbers"] == ["99"]
    assert g["no_harness_words"]["pass"] is False and g["no_harness_words"]["hits"][0]["words"] == ["session"]
    assert g["final_reply"]["pass"] and g["final_reply"]["review_flags"]
    assert not g["reply_names_no_skipped_items"]["pass"]
    assert g["call_numbers_shown"]["pass"] is False  # 25,000 / 15.6 computed but not shown before the email
    row["messages"][2]["content"] = ""
    row["messages"][3]["content"] = '{"success": false, "error": "Worksheet not found"}'
    g = gates(row)
    assert not g["note_before_calls"]["pass"] and g["note_before_calls"]["calling_turns_without_text"] == [2]
    assert not g["zero_tool_errors"]["pass"] and g["zero_tool_errors"]["count"] == 1
    row["trainable_message_indices"] = [2, 4]
    assert not gates(row)["structure"]["pass"]


def test_identical_calls_and_call_cap(tmp_path):
    row = row_of(tmp_path)
    call = row["messages"][2]["tool_calls"][0]
    tool = row["messages"][3]
    turn = [{"role": "assistant", "content": "Again.", "tool_calls": [call]}, tool]
    row["messages"] = row["messages"][:-1] + turn * 4 + row["messages"][-1:]
    row["trainable_message_indices"] = [i for i, m in enumerate(row["messages"]) if m["role"] == "assistant"]
    g = gates(row)
    assert not g["no_identical_call_5"]["pass"] and g["no_identical_call_5"]["max_identical"] == 5
    assert "calls_le_45" not in g and g["no_identical_call_5"]["calls"] == 6
    (tmp_path / "many").mkdir()
    many = row_of(tmp_path / "many")
    first = many["messages"][2]
    first["tool_calls"] = [dict(first["tool_calls"][0], id=f"c{k}") for k in range(50)]
    many["messages"] = many["messages"][:3] + [dict(many["messages"][3], tool_call_id=f"c{k}") for k in range(50)] \
        + many["messages"][4:]
    many["trainable_message_indices"] = [i for i, m in enumerate(many["messages"]) if m["role"] == "assistant"]
    g = gates(many)  # 50 calls in one turn: no call cap (identical-call gate still applies)
    assert "calls_le_45" not in g and g["no_identical_call_5"]["calls"] >= 50


def test_running_total_notation_is_not_wrong_arithmetic():
    from decimal import Decimal

    from rowlib import expressions, grounding

    text = "Bob: ($50,000 x 9% = $4,500 + $500 referral bonus = $5,000)"
    assert all(e["ok"] for e in expressions(text)), expressions(text)
    assert grounding(text, {Decimal(50000), Decimal(9), Decimal(500)})["ungrounded"] == []
    bad = "Bob: ($50,000 x 9% = $4,500 + $500 referral bonus = $6,000)"
    g = grounding(bad, {Decimal(50000), Decimal(9), Decimal(500)})
    assert g["ungrounded"] or g["wrong_expressions"], g
    wrong_start = "Bob: ($50,000 x 9% = $4,700 + $500 = $5,200)"
    assert not all(e["ok"] for e in expressions(wrong_start))


def test_skip_phrases_catch_exclusion_wording():
    from rowlib import SKIP_PHRASES

    for text in ("**Exclusions applied:** TechRadar report (Tier 3)", "No Action Required (beyond 60-day window): Brian",
                 "- Exclude Archived/Draft content", "Two vendors were omitted", "Excluded: Dave Kim's deal",
                 "Skipped 2 buildings with incomplete data"):
        assert SKIP_PHRASES.search(text), text
    for text in ("Sam Nguyen -> Disqualified (known competitor)", "Denied (3): Horizon Labs: onboarding not complete",
                 "Invoice sent for the inclusive package"):
        assert not SKIP_PHRASES.search(text), text
