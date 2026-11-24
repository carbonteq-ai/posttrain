"""SDPO v6 (19 Nov): the pure pieces of sdpo_online (reflection prompt and parsing, hint cleaning, teacher contexts,
truncation, the P1 numbers and the pre-registered pick, the round monitors) and train_pass's new hooks."""
import json

import pytest

import sdpo_online as so


class FakeTok:
    """ids are indices into a vocabulary of strings; decode joins them."""

    def __init__(self, vocab):
        self.vocab = list(vocab)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[i] for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.vocab.index(token)


def _chain(vocab):
    tok = FakeTok(vocab)
    # turn 1: think + a call; tool result; turn 2: think + final text
    segments = [{"kind": "context", "ids": [0]},
                {"kind": "action", "ids": [1, 2, 3]},
                {"kind": "result", "ids": [4]},
                {"kind": "action", "ids": [1, 2, 5]}]
    chain = {"messages0": [{"role": "system", "content": "s"}, {"role": "user", "content": "Email the team the total."}],
             "segments": segments}
    return tok, chain


VOCAB = ["<prompt>", "<think>pondering", "</think>",
         "<tool_call><function=sheet_read><parameter=id> ss_1 </parameter></function></tool_call>",
         "<tool_response>rows: a=10, b=32</tool_response>", "Done: the total is 42."]


def test_decode_steps_and_visible_text():
    tok, chain = _chain(VOCAB)
    steps = so.decode_steps(tok, chain)
    assert steps[0] == ("", ["sheet_read(id=ss_1)"], "rows: a=10, b=32")
    assert steps[1][0] == "Done: the total is 42." and steps[1][1] == []
    assert "rows: a=10, b=32" in so.visible_text(tok, chain) and so.task_text(chain).startswith("Email")


def test_reflection_messages_findings_only_for_failures():
    steps = [("", ["sheet_read(id=ss_1)"], "rows"), ("Done.", [], "")]
    fail = so.reflection_messages("TASKTEXT", steps, False, "Still needed: a gmail message")
    ok = so.reflection_messages("TASKTEXT", steps, True, "Still needed: a gmail message")
    assert "GRADER FINDINGS" in fail[1]["content"] and "GRADER FINDINGS" not in ok[1]["content"]
    assert "Step 1: sheet_read(id=ss_1)" in fail[1]["content"] and "Step 2: final reply: Done." in fail[1]["content"]
    assert "Do not quote values the agent never saw" in fail[1]["content"]


def test_parse_reflection_variants():
    assert so.parse_reflection('{"step1": "Good read.", "step2": ""}', 2) == {1: "Good read."}
    assert so.parse_reflection('<think>x</think>{"step1": {"hint": "A"}, "step9": "B"}', 2) == {1: "A"}
    assert so.parse_reflection('noise {"step1": "A", "step2": "B" broken', 2) == {}
    assert so.parse_reflection('{"step1": "A", "step2": "B",}', 2) == {1: "A", 2: "B"}  # regex fallback
    assert so.parse_reflection("no json", 3) == {}


def test_clean_hint_drops_unseen_values_and_later_steps():
    seen = so._norm("rows: a=10, b=32 Email the team the total. alice@x.com")
    assert so.clean_hint("Sum a and b. Send 42 to the team.", 2, seen) == "Sum a and b."
    assert so.clean_hint("Use the value 32 from the sheet.", 2, seen) == "Use the value 32 from the sheet."
    assert so.clean_hint("Write to bob@y.com.", 1, seen) == ""
    assert so.clean_hint("Write to alice@x.com.", 1, seen) == "Write to alice@x.com."
    assert so.clean_hint('Mention "Quarterly Plan" here.', 1, seen) == ""
    assert so.clean_hint("Check row 3. At step 5 you post.", 2, seen) == "Check row 3."
    assert so.clean_hint("As in step 1, read first.", 2, seen) == "As in step 1, read first."
    assert len(so.clean_hint("Read first. " * 100, 1, seen)) <= so.HINT_MAX_CHARS


def test_contexts():
    win = so.outcome_context("T", True)
    assert win["first_message"] == f"T\n\n{so.HEAD} {so.SUCCESS_LINE}\n\n{so.TAIL}" and win["notes"] == {}
    lose = so.outcome_context("T", False, {"required_passed": 5.0, "required_total": 8})
    assert "it failed; 3 of the task's 8 required checks were not met." in lose["first_message"]
    assert "some of the task's checks" in so.outcome_context("T", False, {})["first_message"]
    hint = so.hint_context("T", False, {1: "Read first.", 3: "Do not send yet.", 4: "x"}, last_turn=3)
    assert hint["notes"] == {1: so.HINT_PREFIX + "Read first.", 3: so.HINT_PREFIX + "Do not send yet."}
    assert "Hints for some steps appear" in hint["first_message"]
    assert so.hint_context("T", True, {2: "Good."})["notes"] == {2: so.HINT_PREFIX + "Good."}


def test_truncate_keeps_aligned_turns():
    seq = {"student": {"ids": [9], "positions": [1, 2, 3, 4], "targets": [5, 6, 7, 8], "old": [-1, -1, -1, -1],
                       "turns": [1, 1, 2, 3]},
           "teacher": {"ids": [9], "positions": [11, 12, 13, 14], "targets": [5, 6, 7, 8]}}
    out = so.truncate(seq, 2)
    assert out["student"]["targets"] == [5, 6, 7] and out["teacher"]["positions"] == [11, 12, 13]
    assert so.truncate(seq, None) is seq


def test_post_think_indices():
    tok = FakeTok(["a", "</think>", "call", "x"])
    student = {"targets": [0, 1, 2, 2, 0, 1, 3], "turns": [1, 1, 1, 1, 2, 2, 2]}
    assert so.post_think_indices(tok, student, 1) == [2, 3]
    assert so.post_think_indices(tok, student, 2) == [6]
    assert so.post_think_indices(tok, student, 3) == []


def test_audit_row_summary_and_pick():
    assert so.audit_row([], [], None, False) is None
    s = so.audit_row([-1.0, -1.0], [-1.5, -2.0], None, True)
    assert s == {"success": True, "kl": pytest.approx(0.75)}
    f = so.audit_row([-1.0, -1.0, -1.0], [-1.0, -3.0, -1.5], [1], False)
    assert f["m1"] == pytest.approx(-2.0) and f["drift"] == pytest.approx(0.25) and f["decision_tokens"] == 1

    def rows(design, m1, drift, kl, forbid_m1):
        out = [{"design": design, "success": False, "m1": m1, "drift": drift, "forbidden": False},
               {"design": design, "success": False, "m1": forbid_m1, "drift": drift, "forbidden": True},
               {"design": design, "success": True, "kl": kl}]
        return out

    summary = so.summarise(rows("V5", -0.5, 0.3, 0.05, 0.2) + rows("OUT", -0.1, 0.1, 0.02, -0.1)
                           + rows("HINT", -0.9, 0.25, 0.03, -0.4))
    assert summary["HINT"]["M4"] == pytest.approx(-0.4) and summary["V5"]["M4"] == pytest.approx(0.2)
    assert so.pick_design(summary) == {"design": "HINT", "rule": "strict"}
    # HINT drifts more than V5 -> not strict; the concentration rule then picks among M4 < 0
    summary2 = so.summarise(rows("V5", -0.5, 0.3, 0.05, 0.2) + rows("OUT", -0.1, 0.1, 0.02, -0.1)
                            + rows("HINT", -0.9, 0.9, 0.03, -0.4))
    assert so.pick_design(summary2)["design"] in ("OUT", "HINT") and so.pick_design(summary2)["rule"] in ("strict", "concentration")
    summary3 = so.summarise(rows("V5", -0.5, 0.3, 0.05, 0.2) + rows("OUT", -0.1, 0.1, 0.02, 0.1))
    assert so.pick_design(summary3)["design"] is None


def test_monitor():
    first = {"success": 0.40, "reasoning": 1000.0, "runaway": 0.01}
    assert so.monitor(first, {"success": 0.36, "reasoning": 1200.0, "runaway": 0.02}) == []
    assert so.monitor(first, {"success": 0.33, "reasoning": 1000.0, "runaway": 0.01}) == []  # 7 points: noise
    assert so.monitor(first, {"success": 0.31, "reasoning": 1000.0, "runaway": 0.01}) != []  # 9 points: stop
    # 20 Nov: the reasoning cap is 2.0x (was 1.3x): 1.9x does not stop, 2.1x does
    assert so.monitor(first, {"success": 0.40, "reasoning": 1900.0, "runaway": 0.01}) == []
    assert so.monitor(first, {"success": 0.40, "reasoning": 2100.0, "runaway": 0.01}) == ["reasoning chars above 2.0x round 1"]
    reasons = so.monitor(first, {"success": 0.30, "reasoning": 2100.0, "runaway": 0.05})
    assert len(reasons) == 3


def test_infra_check():
    def outs(n, bad):
        return {str(i): {"infrastructure_failure": i < bad} for i in range(n)}
    assert so.infra_check(outs(100, 0), 1) == 0
    assert so.infra_check(outs(100, 5), 2) == 5  # 5% allowed
    with pytest.raises(RuntimeError, match="infrastructure failures"):
        so.infra_check(outs(100, 6), 2)
    with pytest.raises(RuntimeError):
        so.infra_check(outs(384, 384), 2)  # the pilot r1 round 2: a dead engine, not a success reading
    with pytest.raises(RuntimeError):
        so.infra_check({}, 1)


def test_release_training_memory_without_cuda_logs_nothing():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU-only check")
    logged = []
    assert so.release_training_memory(lambda **k: logged.append(k)) == 0.0
    assert logged == []


def test_train_pass_hooks_called_per_step():
    torch = pytest.importorskip("torch")
    from sdpo_classic import train_pass

    model = torch.nn.Linear(2, 1)
    calls = []

    def loss_fn(_model, student, _teacher):
        out = model(torch.ones(1, 2)).sum() * student
        return out, [1.0]

    prepared = [({"episode_id": str(i)}, {"student": 1.0, "teacher": None}) for i in range(5)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps, losses, oom, _ = train_pass(model, prepared, loss_fn, 1e-3, 2, "/dev/null", lambda **k: None,
                                       peak_mib=lambda: 0, after_step=lambda: calls.append(1), optimizer=optimizer)
    assert len(steps) == 3 and len(calls) == 3 and oom == 0 and len(losses) == 5


def test_prompt_ids_reads_input_ids_from_any_encoding():
    class Enc(dict):
        pass

    class BatchLike:  # not a dict: like transformers 5's BatchEncoding (a UserDict)
        def __init__(self, ids):
            self._d = {"input_ids": ids, "attention_mask": [1] * len(ids)}

        def __getitem__(self, k):
            return self._d[k]

    class Tok:
        def __init__(self, out):
            self.out = out

        def apply_chat_template(self, *a, **k):
            assert k["return_dict"] and not k["enable_thinking"] and k["add_generation_prompt"]
            return self.out

    assert so.prompt_ids(Tok(BatchLike([5, 6, 7])), []) == [5, 6, 7]
    assert so.prompt_ids(Tok(Enc(input_ids=[[1, 2]])), []) == [1, 2]


def test_truth_filter_drops_hints_that_contradict_the_grader():
    hints = {1: "Good: you read the sheet first.", 2: "This step was correct.", 3: "You should have checked the policy."}
    kept, dropped = so.truth_filter(hints, False, 2)  # turn 2 is the located wrong decision
    assert kept == {1: hints[1], 3: hints[3]} and dropped["endorses_located_mistake"] == 1
    kept, _ = so.truth_filter({2: "Correct target, but you should not include the draft rows."}, False, 2)
    assert 2 in kept  # endorses one part but names the mistake: kept
    kept, dropped = so.truth_filter({1: "This was a mistake.", 2: "Good read.", 3: "You could search once instead."},
                                    True, None)
    assert kept == {2: "Good read.", 3: "You could search once instead."} and dropped["error_claim_on_success"] == 1
    ctx = so.hint_context("T", False, so.truth_filter({2: "Right call.", 1: "Read first."}, False, 2)[0], 2)
    assert ctx["notes"] == {1: so.HINT_PREFIX + "Read first."}


class Tok2:
    """id -> token string; encode for newlines; the real tags are single ids."""
    VOCAB = ["r1", "r2", "</think>", "\n", "<tool_call>", "c1", "</tool_call>", "Done", ".", "<|im_end|>"]

    def convert_tokens_to_ids(self, t):
        return self.VOCAB.index(t)

    def encode(self, text, add_special_tokens=False):
        return [self.VOCAB.index("\n")] if text in ("\n", "\n\n") else []


def test_with_decision_tokens_keeps_reasoning_calls_and_the_decision():
    V = Tok2.VOCAB
    ids = lambda *t: [V.index(x) for x in t]  # noqa: E731
    # turn 1: reasoning, </think>, newline, a call; turn 2: reasoning, </think>, newline, text reply
    targets = ids("r1", "</think>", "\n", "<tool_call>", "c1", "</tool_call>", "\n", "r2", "</think>", "\n", "Done", ".", "<|im_end|>")
    turns = [1] * 7 + [2] * 6
    seq = {"student": {"ids": [0], "positions": list(range(13)), "targets": targets, "old": [0.0] * 13, "turns": turns},
           "teacher": {"ids": [0], "positions": list(range(100, 113)), "targets": targets}}
    out = so.with_decision_tokens(Tok2(), seq)
    kept = [V[t] for t in out["student"]["targets"]]
    assert kept == ["r1", "</think>", "<tool_call>", "c1", "</tool_call>", "r2", "</think>", "Done"]
    assert out["teacher"]["positions"] == [100, 101, 103, 104, 105, 107, 108, 110]


def test_note_kind():
    assert so.note_kind({"notes": {3: "Note: x. This task forbids it."}}, 3) == "forbidden"
    assert so.note_kind({"notes": {}, "final_note": {"line": "Still not done at this point: (1) ..."}}, 9) == "missed or wrong content"
    assert so.note_kind({"notes": {2: "gmail_find_email returned an error: not found"}}, 2) == "tool error"
    assert so.note_kind(None, 2) == "none"


class Tok3:
    """r3 fix A: tokens shaped like the real Qwen3.5 call tokens (tags split into pieces, as the real tokenizer does)."""
    VOCAB = ["r1", "</think>", "\n", "<tool_call>", "<", "function", "=gmail_send_email", ">", "parameter", "=", "to",
             "bob@x.com", "body", "Hello", " team", ",", " totals", "</", "subject", "Q4", " report", "</tool_call>",
             "cells", '{"A": 1}', "Done", "<|im_end|>"]

    def convert_tokens_to_ids(self, t):
        return self.VOCAB.index(t)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.VOCAB[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [self.VOCAB.index("\n")] if text in ("\n", "\n\n") else []


def _call_seq():
    V = Tok3.VOCAB
    toks = ["r1", "</think>", "\n", "<tool_call>", "\n", "<", "function", "=gmail_send_email", ">", "\n",
            "<", "parameter", "=", "to", ">", "\n", "bob@x.com", "\n", "</", "parameter", ">", "\n",
            "<", "parameter", "=", "body", ">", "\n", "Hello", " team", ",", " totals", "\n", "</", "parameter", ">", "\n",
            "<", "parameter", "=", "subject", ">", "\n", "Q4", " report", "\n", "</", "parameter", ">", "\n",
            "<", "parameter", "=", "cells", ">", "\n", '{"A": 1}', "\n", "</", "parameter", ">", "\n",
            "</", "function", ">", "\n", "</tool_call>"]
    targets = [V.index(t) for t in toks]
    turns = [1] * len(targets)
    n = len(targets)
    seq = {"student": {"ids": [0], "positions": list(range(n)), "targets": targets, "old": [0.0] * n, "turns": turns},
           "teacher": {"ids": [0], "positions": list(range(100, 100 + n)), "targets": targets}}
    return toks, targets, turns, seq


def test_call_text_indices_marks_only_free_text_values():
    toks, targets, turns, _ = _call_seq()
    marked = sorted(so.call_text_indices(Tok3(), targets, turns))
    assert [toks[i] for i in marked] == ["Hello", " team", ",", " totals", "Q4", " report"]
    # recipient, cells value, tool name, parameter names and every tag stay trained
    for keep in ("bob@x.com", '{"A": 1}', "=gmail_send_email", "body", "subject", "to", "cells", "<", ">", "</"):
        assert all(toks[i] != keep for i in marked)


def test_without_call_text_and_classes():
    toks, targets, turns, seq = _call_seq()
    out = so.without_call_text(Tok3(), so.with_decision_tokens(Tok3(), seq))
    kept = [Tok3.VOCAB[t] for t in out["student"]["targets"]]
    assert "Hello" not in kept and " report" not in kept and "bob@x.com" in kept and '{"A": 1}' in kept
    assert len(out["teacher"]["positions"]) == len(out["student"]["targets"])
    classes = so.token_classes(Tok3(), targets, turns)
    assert classes[0] == "reasoning" and classes[1] == "reasoning" and classes[2] == "text"
    assert classes[3] == "decision"
    assert {toks[i] for i, c in enumerate(classes) if c == "call_text"} == {"Hello", " team", ",", " totals", "Q4", " report"}
    assert classes[toks.index("bob@x.com")] == "call"
    # nothing to drop -> the same object back
    plain = {"student": {"ids": [0], "positions": [0], "targets": [Tok3.VOCAB.index("Done")], "old": [0.0], "turns": [1]},
             "teacher": {"ids": [0], "positions": [9], "targets": [Tok3.VOCAB.index("Done")]}}
    assert so.without_call_text(Tok3(), plain) is plain


def test_r3_audit_row_classes_and_trained_view():
    classes = ["reasoning", "reasoning", "decision", "call", "call_text", "call", "text"]
    student = [-1.0] * 7
    teacher = [-1.2, -0.9, -1.5, -1.4, -3.0, -1.0, -2.0]
    row = so.audit_row_r3(student, teacher, [2, 3, 4, 5, 6], False, classes)
    # m1A uses only the decision and call tokens of the wrong action (index 2, 3, 5), not call_text or text
    assert row["m1A"] == pytest.approx(((-0.5) + (-0.4) + 0.0) / 3)
    assert row["driftA"] == pytest.approx(-((-0.2) + 0.1) / 2)  # reasoning tokens, outside the decision
    assert row["class_abs"]["call_text"] == pytest.approx(2.0) and row["class_n"]["text"] == 1
    ok = so.audit_row_r3(student, teacher, None, True, classes)
    assert ok["klA"] == pytest.approx(-((-0.2) + 0.1 + (-0.5) + (-0.4) + 0.0) / 5)


def test_r3_task_bootstrap_and_pick_rule():
    m, r = so.task_bootstrap({"a": [-0.1, -0.2], "b": [-0.15], "c": [-0.05, -0.1]}, draws=2000)
    assert m == pytest.approx(-0.12) and r[1] < 0
    assert so.task_bootstrap({}) == (None, None)

    def row(design, eid, task, m1a, kind="missed or wrong content", forbidden=False):
        return {"design": design, "episode_id": eid, "task": task, "success": False, "m1": m1a, "m1A": m1a,
                "driftA": 0.1, "kind": kind, "forbidden": forbidden,
                "class_abs": {c: 1.0 for c in so.CLASSES}, "class_n": {c: 1 for c in so.CLASSES}}
    rows = []
    for i in range(30):
        task = f"t{i % 10}"
        rows.append(row("V5", f"e{i}", task, -0.01, forbidden=(i % 3 == 0)))
        rows.append(row("W1", f"e{i}", task, -0.05, forbidden=(i % 3 == 0)))  # clearly better everywhere
        rows.append(row("W2", f"e{i}", task, -0.01 + (0.001 if i % 2 else -0.001), forbidden=(i % 3 == 0)))  # no change
    s = so.summarise_r3(rows)
    pick = so.pick_design_r3(s)
    assert pick["design"] == "W1" and pick["qualified"] == ["W1"]
    assert any(x.startswith("a:") for x in s["W2"]["pick_reasons"])
    # nothing qualifies -> V5
    assert so.pick_design_r3(so.summarise_r3([r for r in rows if r["design"] != "W1"]))["design"] == "V5"
