"""Unit tests for shape_report (pure Python).

  CAPPED_MEM=4G .../bin/capped <any python with pytest> -m pytest scripts/qualification/sftgen/test_shape_report.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import shape_report  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "sheets_get", "description": "Get rows.", "parameters": {
    "type": "object", "properties": {"spreadsheet": {"type": "string"}, "includeGridData": {"type": "boolean"}}}}}]


def call(i: int, name: str, args: dict) -> dict:
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args}}


def row(task: str, turns: list[tuple[str, int]], reply: str, source: str = "own-clean-success") -> dict:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    k = 0
    for text, n in turns:
        calls = [call(k + j, "sheets_get", {"spreadsheet": "ss_1", "includeGridData": True}) for j in range(n)]
        k += n
        messages.append({"role": "assistant", "content": text, "tool_calls": calls})
        messages += [{"role": "tool", "tool_call_id": c["id"], "name": "sheets_get", "content": "{}"} for c in calls]
    messages.append({"role": "assistant", "content": reply})
    return {"id": f"x/{task}", "messages": messages, "tools": TOOLS,
            "trainable_message_indices": [i for i, m in enumerate(messages) if m["role"] == "assistant"],
            "metadata": {"task": task, "source": source}}


NOTE = "I need the sheet and the inbox; reading both now so I can compare the two lists of contacts."
REPLY = "I updated the three contacts and emailed the summary to the controller with the new totals."
OWN = [row("t.a", [(NOTE, 1), (NOTE, 2), ("", 1)], REPLY), row("t.b", [(NOTE, 1), (NOTE, 3)], REPLY)]


def bars(result: dict) -> dict:
    return {b["metric"]: b["pass"] for b in result["bars"]}


def test_reference_against_itself_passes_and_measures():
    result = shape_report.report(OWN, OWN, None, shape_report.BARS)
    assert result["all_pass"], result["bars"]
    ref = result["reference"]
    assert ref["calling_turns"] == 5 and ref["text_share"] == 0.8 and ref["multi_call_share"] == 0.4
    assert ref["calls_per_turn_histogram"] == {"1": 3, "2": 1, "3": 1, "4": 0, "5": 0, "6+": 0}
    assert ref["sorted_args_share"] == 0.0 and ref["note_chars_median"] == len(NOTE)


def test_one_call_per_turn_without_text_fails_like_the_12_nov_rows():
    flat = [row("t.a", [("", 1)] * 5, REPLY, "claude"), row("t.b", [("", 1)] * 4, "Done: sheets_get.", "claude")]
    result = shape_report.report(flat, OWN, None, shape_report.BARS)
    b = bars(result)
    assert not result["all_pass"]
    assert not b["text_share"] and not b["multi_call_share"] and not b["note_chars_median"]


def test_key_sorted_rows_fail_the_key_order_bars():
    sorted_rows = [json.loads(json.dumps(r, sort_keys=True)) for r in OWN]
    result = shape_report.report(sorted_rows, OWN, None, shape_report.BARS)
    b = bars(result)
    assert result["candidate"]["sorted_args_share"] == 1.0 and not b["sorted_args_share"]
    assert not b["tools_key_order"]


def test_runner_tools_comparison_and_cli(tmp_path):
    runner = {"t.a": {"tools": TOOLS}, "t.b": {"tools": TOOLS}}
    assert bars(shape_report.report(OWN, OWN, runner, shape_report.BARS))["tools_key_order"]
    other = [{**OWN[0], "tools": [{"type": "function", "function": {**TOOLS[0]["function"], "description": "x"}}]},
             OWN[1]]
    assert not bars(shape_report.report(other, OWN, runner, shape_report.BARS))["tools_key_order"]
    own_file, rows_file, out = tmp_path / "own.jsonl", tmp_path / "rows.jsonl", tmp_path / "r.json"
    own_file.write_text("".join(json.dumps(r) + "\n" for r in OWN + [row("t.c", [("", 1)], "x", "claude")]))
    rows_file.write_text("".join(json.dumps(r) + "\n" for r in OWN))
    assert shape_report.main([str(rows_file), "--own", str(own_file), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["reference"]["rows"] == 2  # only source own* rows are the reference
    assert shape_report.main([str(rows_file), "--own", str(own_file), "--bar", "text_share_offset=-0.5"]) == 1
