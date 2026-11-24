"""render_check on the materialized file (wt-sft .venv: posttrain packages, transformers; the Qwen3.5-4B tokenizer).

  CAPPED_MEM=4G .../bin/capped .venv/bin/python -m pytest scripts/qualification/sftgen/test_render_check.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
pytest.importorskip("posttrain.train")
pytest.importorskip("transformers")

import render_check  # noqa: E402

if not render_check.TOKENIZER.exists():
    pytest.skip("Qwen3.5-4B tokenizer files not present", allow_module_level=True)

TOOLS = [{"type": "function", "function": {"name": "sheets_get", "description": "Get rows.", "parameters": {
    "type": "object", "properties": {"spreadsheet": {"type": "string"}, "includeGridData": {"type": "boolean"}},
    "required": ["spreadsheet"]}}},
         {"type": "function", "function": {"name": "gmail_send_email", "description": "Send.", "parameters": {
             "type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}}}}]


def rows() -> list[dict]:
    messages = [
        {"role": "system", "content": "You are a workflow automation agent."},
        {"role": "user", "content": "Email the variance — café “quoted”."},
        {"role": "assistant", "content": "Reading the sheet and the inbox together.", "tool_calls": [
            {"id": "call_1_1", "type": "function",
             "function": {"name": "sheets_get", "arguments": {"spreadsheet": "ss_1", "includeGridData": True}}},
            {"id": "call_1_2", "type": "function",
             "function": {"name": "gmail_send_email", "arguments": {"to": "c@x.example.com", "body": "Ünïcode"}}}]},
        {"role": "tool", "tool_call_id": "call_1_1", "name": "sheets_get", "content": '{"success": true}'},
        {"role": "tool", "tool_call_id": "call_1_2", "name": "gmail_send_email", "content": '{"success": true}'},
        {"role": "assistant", "content": "Sent the variance to the controller."},
    ]
    one = {"id": "unit/render-a", "messages": messages, "tools": TOOLS, "trainable_message_indices": [2, 5],
           "metadata": {"task": "unit.a", "source": "unit"}}
    two = json.loads(json.dumps(one))
    two["id"] = "unit/render-b"
    two["messages"][2]["content"] = ""  # a calling turn without text renders too
    return [one, two]


def test_materialized_file_renders_like_the_runner_and_masks_only_assistant_turns(tmp_path):
    src = tmp_path / "rows.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows()))
    report = render_check.run(src, tmp_path / "out", render_check.TOKENIZER, None, set())
    summary = report["summary"]
    assert summary["all_pass"], report
    assert summary["chat_template_is_file"] and summary["rows_with_multi_call_turns"] == 2
    assert all(r["mask_exact"] and r["ids_equal_runner_render"] for r in report["rows"])


def test_a_key_sorted_materialized_file_fails(tmp_path):
    built = rows()
    src = tmp_path / "rows.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in built))
    sorted_file = tmp_path / "sorted.jsonl"
    sorted_file.write_text("".join(json.dumps({**r, "metadata": {**r["metadata"], "source_row": i,
                                                                   "source_format": "messages"}}, sort_keys=True)
                                   + "\n" for i, r in enumerate(built)))
    report = render_check.run(src, tmp_path / "out", render_check.TOKENIZER, sorted_file, set())
    assert not report["summary"]["all_pass"]
    assert report["summary"]["round_trip_line_equal"] == 0 and report["summary"]["ids_equal_runner_render"] == 0
