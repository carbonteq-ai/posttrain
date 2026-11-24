"""Tests for sdpo_a3 (SDPO plan v4 phase A3). The integration test reads one recorded guard10 attempt of P9 as code
test material only (never training material) and needs the environment interpreter (SDPO_A3_EPISODES set)."""
import asyncio
import gzip
import json
import os
import sys
from pathlib import Path

import pytest

import sdpo_a3 as a3


def test_point_turn_and_digest():
    context = {"first_message": "m", "notes": {3: "x", 5: "y"}, "final_note": {"turn": 7, "items": ["z"]}}
    assert a3.point_turn(context) == 3
    assert a3.point_turn({"notes": {}, "final_note": {"turn": 7}}) == 7
    assert a3.point_turn({"notes": {}, "final_note": None}) is None
    same = {"first_message": "m", "notes": {"5": "y", "3": "x"}, "final_note": {"turn": 7, "items": ["z"]}}
    assert a3.context_digest(context) == a3.context_digest(same)
    assert a3.context_digest(context) != a3.context_digest({**context, "first_message": "m2"})


def test_constants_match_validate_row():
    sftgen = Path(__file__).resolve().parents[3] / "wt-sft/scripts/qualification/sftgen/validate_row.py"
    text = sftgen.read_text()
    assert f'A3_SOURCE = "{a3.A3_SOURCE}"' in text and f'A3_PREFIX_REASON = "{a3.A3_PREFIX_REASON}"' in text


class _Tok:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def test_continue_episode_runs_every_call_and_stops():
    replies = [([5, 6], "stop", [{"id": "a", "function": {"name": "t_send", "arguments": "{}"}},
                                 {"id": "b", "function": {"name": "t_find", "arguments": "{}"}}]),
               ([7], "stop", [])]
    executed = []

    async def generate(tokens, budget, index):
        return replies[index][0], replies[index][1]

    def parse(text):
        return next(r[2] for r in replies if " ".join(map(str, r[0])) == text)

    async def run_one(name, arguments):
        executed.append(name)
        return f"result-{name}", "ok"

    out = asyncio.run(a3.continue_episode([1, 2], tokenizer=_Tok(), generate=generate, parse=parse, run_one=run_one,
                                          render=lambda messages: [9] * len(messages),
                                          limits={"max_turns": 50, "reply": 100, "context": 1000},
                                          turns_before=2, output_before=0))
    assert executed == ["t_send", "t_find"] and out["stop"] == "final_reply" and len(out["turns"]) == 2
    assert [c["result"] for c in out["turns"][0]["calls"]] == ["result-t_send", "result-t_find"]
    cut = asyncio.run(a3.continue_episode([1], tokenizer=_Tok(), generate=generate, parse=parse, run_one=run_one,
                                          render=lambda m: [], limits={"max_turns": 3, "reply": 100, "context": 1000},
                                          turns_before=3, output_before=0))
    assert cut["stop"] == "max_turns" and not cut["turns"]


def _row(reasoning, text, args=None):
    return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "send the Q1 total to cfo"},
                         {"role": "assistant", "content": text, "reasoning_content": reasoning,
                          **({"tool_calls": [{"id": "1", "function": {"name": "gmail_send_email",
                                                                      "arguments": args}}]} if args else {})}],
            "trainable_message_indices": [2]}


def test_leak_findings_both_ways():
    context = {"first_message": "Still not done at this point: a gmail message with body containing \"59,000\"",
               "notes": {2: "Note on your next action: SPON-558-Q1 is required"}, "final_note": None}
    leaky = a3.leak_findings(_row("The feedback says the total is 59,000.", "Sending."), context)
    assert {f["kind"] for f in leaky} == {"feedback_phrase", "privileged_value"}
    assert a3.leak_findings(_row("", "ok", {"body": "ref SPON-558-Q1"}), context)[0]["text"] == "SPON-558-Q1"
    clean = a3.leak_findings(_row("Let me check the sheet and add the notes column total.", "Sending the report."),
                             context)
    assert clean == []
    seen = _row("", "Total 59,000.")
    seen["messages"][1]["content"] += " The Q1 total is 59,000."  # the value is in the request: not privileged
    assert a3.leak_findings(seen, context) == []
    for phrase in ("Per the hint, ", "The grader wants ", "This task forbids ", "the reference solution shows "):
        assert a3.leak_findings(_row(phrase + "x.", "y"), context), phrase
    for ordinary in ("Add a note to the ticket.", "the notes field is empty", "check the inventory first",
                     "solution architect"):
        assert a3.leak_findings(_row(ordinary, "y"), context) == [], ordinary


@pytest.mark.skipif(not os.environ.get("SDPO_A3_EPISODES"), reason="needs SDPO_A3_EPISODES and the env interpreter")
def test_row_of_the_students_own_rest_replays_like_the_recorded_attempt():
    """Integration: cutting a recorded successful attempt at a turn and appending the student's own remaining turns as
    the 'continuation' must give a row whose structure passes and whose replay equals the recorded score (1.0)."""
    run = Path(os.environ["SDPO_A3_EPISODES"])
    records = [json.loads(l) for l in gzip.open(run / "episodes.jsonl.gz", "rt")]
    chains = {c["episode_id"]: c for c in map(json.loads, gzip.open(run / "chains.jsonl.gz", "rt"))}
    record = next(r for r in records if r["traces"][0]["metrics"]["task_completed_correctly"] == 1.0
                  and len([m for m in a3.assistant_path(r) if m.get("role") == "assistant"]) >= 3)
    chain = chains[record["id"]]
    path = a3.assistant_path(record)
    results = {m["tool_call_id"]: m["content"] for m in path if m.get("role") == "tool"}
    assistants = [m for m in path if m.get("role") == "assistant"]
    actions = [s for s in chain["segments"] if s["kind"] == "action"]
    turn = 2
    rest = {"stop": "final_reply", "turns": [
        {"text": actions[k].get("raw") or "", "calls": [
            {"id": c["id"], "name": c["name"], "arguments": c["arguments"], "status": "ok", "result": results[c["id"]]}
            for c in assistants[k].get("tool_calls") or []]} for k in range(turn - 1, len(assistants))]}
    row = a3.to_row(record, chain, turn, rest, arm="T-A2", item_id="itest")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "wt-sft/scripts/qualification/sftgen"))
    import validate_row

    validate_row.THINKING = True
    report = validate_row.validate(row, replay=True)
    assert report["gates"]["structure"]["pass"], report["gates"]["structure"]
    assert report["gates"]["replay_score"]["pass"] and report["gates"]["results_equal_replay"]["pass"], report
    assert a3.calls_before(record, 1) == 0
    assert a3.calls_before(record, turn) == len(assistants[0].get("tool_calls") or [])


def test_token_classes():
    # turn 1: reasoning r r </think>(9) \n(5) <tool_call>(1) a b </tool_call>(2); turn 2: r </think> \n text text
    targets = [7, 7, 9, 5, 1, 3, 4, 2, 7, 9, 5, 6, 6]
    turns = [1] * 8 + [2] * 5
    got = a3.token_classes(targets, turns, think_close=9, opener=1, closer=2, newline_ids={5})
    assert got == ["reasoning"] * 3 + ["text", "decision", "call", "call", "call", "reasoning", "reasoning", "text",
                                       "decision", "text"]
    assert a3.token_classes([7, 7], [1, 1], 9, 1, 2) == ["reasoning", "reasoning"]  # cut inside the thinking


def test_worked_out_values_are_not_leaks_and_copied_ones_are():
    context = {"first_message": "Not done: body containing \"15.3\" and \"SPON-558\"", "notes": {}, "final_note": None}
    worked = _row("DPO = 3000 / 18000 * 92 = 15.33 so about 15.3 days.", "Metro Supply DPO 15.3 days.")
    assert [f for f in a3.leak_findings(worked, context) if f["kind"] == "privileged_value"] == []
    copied = _row("The value is 15.3.", "Metro Supply DPO 15.3 days.")
    assert any(f["text"] == "15.3" for f in a3.leak_findings(copied, context))
    wrong = _row("DPO = 3000 / 18000 * 92 = 17.0 so 15.3.", "15.3")  # a wrong expression grounds nothing
    assert any(f["text"] == "15.3" for f in a3.leak_findings(wrong, context))


def test_attempt_references_are_leaks():
    context = {"first_message": "", "notes": {}, "final_note": None}
    for phrase in ("as per the feedback from the earlier unsuccessful attempt", "my previous attempt failed",
                   "unlike the first try"):
        assert a3.leak_findings(_row(phrase, "x"), context), phrase
    for ordinary in ("the delivery attempt is logged", "retry the API call", "first, list the rows"):
        assert a3.leak_findings(_row(ordinary, "x"), context) == [], ordinary


def test_quiet_arm_context():
    context = {"first_message": "Correctly solve the original question.", "notes": {2: "n"}, "final_note": None}
    quiet = a3.teacher_context("T-A2q", context)
    assert quiet["first_message"].endswith(a3.QUIET) and quiet["notes"] == context["notes"]
    assert a3.teacher_context("T-A2", context) is context
    assert a3.context_digest(context) != a3.context_digest(quiet)  # items carry the engine context's digest


def test_loading_the_sft_reader_leaves_imports_untouched():
    a3._EXPRESSIONS = None  # load it here, not from an earlier test's cache
    path, mods = list(sys.path), set(sys.modules)
    assert 4 in a3.computed_values("2 + 2 = 4")
    assert sys.path == path and set(sys.modules) == mods


def test_derived_on_a_line():
    line = "DPO = ($3,000 / $18,000) * 92 = 0.1667... * 92 = 15.333... ≈ 15.3 days"
    assert a3.derived_on_a_line("15.3", line)
    assert not a3.derived_on_a_line("15.3", "The value is 15.3 days.")
    assert not a3.derived_on_a_line("15.3", "Target = 15.3")  # an '=' alone, no arithmetic before it
    assert not a3.derived_on_a_line("15.3", "3 * 4 = 12\nthe target is 15.3")  # arithmetic on another line
    context = {"first_message": "Not done: body containing \"15.3\"", "notes": {}, "final_note": None}
    assert a3.leak_findings(_row(line, "DPO 15.3 days."), context) == []


def test_derived_requires_a_correct_step():
    assert a3.derived_on_a_line("15.33", "DPO = (3,000 / 18,000) * 92 = 15.33")
    assert a3.derived_on_a_line("15.3", "DPO: ($3,000 / $18,000) * 92 = 0.1667... * 92 = 15.333... ≈ 15.3 days")
    assert not a3.derived_on_a_line("15.3", "2 * 3 = 15.3")
    assert not a3.derived_on_a_line("15.3", "16.9 ≈ 15.3")


def test_no_feedback_control_uses_the_students_own_tokens():
    """15 Nov: T-0 continues from exactly the student's recorded tokens (no feedback anywhere); other arms unchanged."""
    chain = {"segments": [{"kind": "context", "ids": [1, 2, 3]}, {"kind": "action", "ids": [4, 5]},
                          {"kind": "result", "ids": [6]}, {"kind": "action", "ids": [7]}]}
    assert a3.student_prefix(chain, 3) == [1, 2, 3, 4, 5, 6]
    assert a3.student_prefix(chain, 1) == [1, 2, 3]
    assert a3.ARMS["T-0"]["no_feedback"] and a3.ARMS["T-0"].get("base") == "T-A2"
    assert not any(a3.ARMS[arm].get("no_feedback") for arm in ("T-A2", "T-A1", "T-A2q"))
    assert {k: v for k, v in a3.ARMS["T-0"].items() if k not in ("no_feedback",)} == {**a3.ARMS["T-A2"], "base": "T-A2"}


# ------------------------------------------------------------------ signal check (plan of 16 Nov, Step 2)

THINK = 9


class _FakeRunner:
    """token_logprobs returns, per target token, +1 under the teacher prefix (it starts with the teacher marker 77)
    and 0 under the student prefix, for the correct turn's tokens (marked 5); every other token scores -1 for both."""

    @staticmethod
    def engine_teacher_prefix(tok, chain, position, context):
        return [77] + a3.student_prefix(chain, position)

    @staticmethod
    def token_logprobs(model, ids, positions, targets):
        teacher = ids[0] == 77
        return [(1.0 if teacher else 0.0) if t == 5 else -1.0 for t in targets]


def _signal_chain():
    return {"segments": [{"kind": "context", "ids": [1, 2]}, {"kind": "action", "ids": [3, THINK, 4, 4]},
                         {"kind": "context", "ids": [6]}, {"kind": "action", "ids": [3, THINK, 4]}]}


def test_action_tokens_and_signal_value():
    assert a3.action_tokens([3, 3, THINK, 4, 5], THINK) == [3, 4]
    assert a3.action_tokens([3, 3, 4], THINK) is None
    assert a3.action_tokens([3, THINK], THINK) is None
    assert a3.signal_value(-1.0, -3.0, -2.0, -1.0) == 3.0
    assert a3.signal_value(-2.0, -2.0, -1.0, -1.0) == 0.0


def test_signal_item_scores_action_tokens_under_both_prefixes(monkeypatch):
    context = {"first_message": "m", "notes": {}, "final_note": None}
    item = {"digest": a3.context_digest(context), "turn": 1, "arm": "T-A2", "correct_ids": [3, THINK, 5, 5]}
    out = a3.signal_item(item, _signal_chain(), context, _FakeRunner, None, THINK, type("RT", (), {"model": None}))
    # correct action tokens [5, 5]: teacher +2, student 0; wrong action tokens [4, 4]: both -2
    assert out["teacher_correct"] == 2.0 and out["student_correct"] == 0.0
    assert out["teacher_wrong"] == -2.0 and out["student_wrong"] == -2.0
    assert out["tokens_correct"] == 2 and out["tokens_wrong"] == 2
    assert out["signal"] == 2.0


def test_signal_item_skips():
    context = {"first_message": "m", "notes": {}, "final_note": None}
    rt = type("RT", (), {"model": None})
    good = a3.context_digest(context)
    chain = _signal_chain()
    assert a3.signal_item({"digest": "x", "turn": 1, "arm": "T-A2", "correct_ids": [3, THINK, 5]}, chain, context,
                          _FakeRunner, None, THINK, rt) == "digest"
    assert a3.signal_item({"digest": good, "turn": 1, "arm": "T-A2", "correct_ids": [3, THINK, 4, 4]}, chain, context,
                          _FakeRunner, None, THINK, rt) == "same_turn"
    assert a3.signal_item({"digest": good, "turn": 1, "arm": "T-A2", "correct_ids": [3, 5, 5]}, chain, context,
                          _FakeRunner, None, THINK, rt) == "unclosed_thinking"
    # turn 2: correct action [5] (teacher +1, student 0), wrong action [4] (both -1)
    assert a3.signal_item({"digest": good, "turn": 2, "arm": "T-A2", "correct_ids": [3, THINK, 5]}, chain, context,
                          _FakeRunner, None, THINK, rt)["signal"] == 1.0
