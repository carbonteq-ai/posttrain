"""attempt_rows: an exact record converts unchanged; every cross-check fails when its condition is broken."""

import copy

from attempt_rows import attempt_row

CALL = '<tool_call>\n<function=lookup>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>'


def record_and_chain():
    nodes = [
        {"parent": None, "message": {"role": "system", "content": "sys"}},
        {"parent": 0, "message": {"role": "user", "content": "do it"}},
        {"parent": 1, "message": {"role": "assistant", "content": "Looking it up.",
                                  "tool_calls": [{"id": "c1", "name": "lookup", "arguments": '{"q": "x"}'}]}},
        {"parent": 2, "message": {"role": "tool", "tool_call_id": "c1", "content": "found"}},
        {"parent": 3, "message": {"role": "assistant", "content": "Done."}},
    ]
    record = {"id": "e1", "traces": [{"nodes": nodes, "task": {"data": {"task_name": "t.one"}},
                                      "info": {"automationbench": {"assertions": [
                                          {"type": "a", "params": {}, "passed": True}]}}}]}
    chain = {"tools": [{"type": "function", "function": {"name": "lookup"}}], "segments": [
        {"kind": "context", "ids": [1]},
        {"kind": "action", "raw": "plan the lookup</think>\n\nLooking it up.\n\n" + CALL, "tool_call_ids": ["c1"]},
        {"kind": "context", "ids": [2]},
        {"kind": "action", "raw": "all good</think>\n\nDone.", "tool_call_ids": []}]}
    return record, chain


def test_exact_record_converts_unchanged():
    record, chain = record_and_chain()
    row = attempt_row(record, chain, {"success": 1.0})
    assert row["metadata"]["exact"] is True, row["metadata"]["chain_problems"]
    m = row["messages"]
    assert [x["role"] for x in m] == ["system", "user", "assistant", "tool", "assistant"]
    assert m[2]["reasoning_content"] == "plan the lookup" and m[2]["content"] == "Looking it up."
    assert m[2]["tool_calls"][0]["function"]["arguments"] == {"q": "x"}
    assert m[3]["content"] == "found" and m[4]["content"] == "Done."
    assert row["trainable_message_indices"] == [2, 4]


def test_unclosed_tool_call_is_flagged():
    record, chain = record_and_chain()
    chain["segments"][1]["raw"] = "plan</think>\n\nLooking it up.\n\n" + CALL.split("</parameter>")[0]
    row = attempt_row(record, chain, {})
    assert any("unclosed tool call" in p for p in row["metadata"]["chain_problems"])
    assert row["metadata"]["exact"] is False


def test_visible_text_mismatch_is_flagged():
    record, chain = record_and_chain()
    chain["segments"][3]["raw"] = "ok</think>\n\nDone, differently."
    row = attempt_row(record, chain, {})
    assert any("visible text differs" in p for p in row["metadata"]["chain_problems"])


def test_call_id_and_count_mismatches_are_flagged():
    record, chain = record_and_chain()
    bad = copy.deepcopy(chain)
    bad["segments"][1]["tool_call_ids"] = ["other"]
    assert any("call ids differ" in p for p in attempt_row(record, bad, {})["metadata"]["chain_problems"])
    short = copy.deepcopy(chain)
    short["segments"] = short["segments"][:2]
    assert any("chain actions" in p for p in attempt_row(record, short, {})["metadata"]["chain_problems"])


def test_missing_result_and_missing_final_reply_are_flagged():
    record, chain = record_and_chain()
    rec = copy.deepcopy(record)
    rec["traces"][0]["nodes"] = rec["traces"][0]["nodes"][:3]
    cut = copy.deepcopy(chain)
    cut["segments"] = cut["segments"][:2]
    problems = attempt_row(rec, cut, {})["metadata"]["chain_problems"]
    assert any("has no result" in p for p in problems) and "no final reply" in problems
