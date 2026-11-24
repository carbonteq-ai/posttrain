"""Round 6 (21 Nov, SDPO closure): a note may be given as its own user message (context note_role "user") instead of
text appended to the last tool result. CPU tests with a chat-template stand-in, both ways: without the key every
teacher sequence is the one the old code built; with it the note follows the tool result as a user message and the
student's own tokens and their positions' meaning are unchanged."""
import copy
import json

import sdpo_a3
import sdpo_ab_runner as runner


class _Tokenizer:
    """'<role>content<E>' per message, '<assistant>' as the generation prompt; one id per character, eos one id."""
    eos_token = "<E>"
    eos_token_id = 1

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False, return_dict=False):
        text = "".join(f"<{m['role']}>{m.get('content') or ''}{self.eos_token}" for m in messages)
        text += "<assistant>" if add_generation_prompt else ""
        if not tokenize:
            return text
        ids = self.encode(text)
        return {"input_ids": ids} if return_dict else ids

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            if text.startswith(self.eos_token, i):
                out.append(self.eos_token_id)
                i += len(self.eos_token)
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.eos_token if i == self.eos_token_id else chr(i) for i in ids)


TOK = _Tokenizer()


def _chain():
    messages0 = [{"role": "system", "content": "S"}, {"role": "user", "content": "Do it"}]
    tool1 = {"role": "tool", "tool_call_id": "c1", "name": "t", "content": "r1"}
    tool2 = {"role": "tool", "tool_call_id": "c2", "name": "t", "content": "r2"}
    action = lambda text, calls: {"kind": "action", "ids": TOK.encode(text) + [TOK.eos_token_id],
                                  "logprobs": [0.0] * (len(text) + 1), "complete": True, "tool_call_ids": calls}
    return {"messages0": messages0, "tools": [], "segments": [
        {"kind": "context", "ids": runner.first_prompt(TOK, messages0, []), "messages": None},
        action("call one", ["c1"]),
        {"kind": "context", "ids": runner.render_delta(TOK, [tool1]), "messages": [tool1]},
        action("call two", ["c2"]),
        {"kind": "context", "ids": runner.render_delta(TOK, [tool2]), "messages": [tool2]},
        action("All done.", [])]}


def _old_teacher_sequence(tokenizer, chain, context):
    """The code before round 6 (wt-sdpo-close 40e43d70 / wt-close-b3 660d3708), kept here as the reference."""
    first = copy.deepcopy(chain["messages0"])
    user_index = next(i for i, m in enumerate(first) if m["role"] == "user")
    content = context["first_message"]
    if 1 in context["notes"]:
        content += "\n\n" + context["notes"][1]
    first[user_index] = {"role": "user", "content": content}
    ids = runner.first_prompt(tokenizer, first[:user_index + 1], chain["tools"])
    positions, targets, turn = [], [], 0
    for index, segment in enumerate(chain["segments"]):
        if index == 0:
            continue
        if segment["kind"] == "action":
            turn += 1
            for k, token in enumerate(segment["ids"]):
                positions.append(len(ids) + k - 1)
                targets.append(token)
            ids += segment["ids"]
        elif segment["messages"] is not None and (turn + 1) in context["notes"]:
            messages = copy.deepcopy(segment["messages"])
            target = max(i for i, m in enumerate(messages) if m["role"] in ("tool", "user"))
            body = messages[target].get("content")
            body = body if isinstance(body, str) else json.dumps(body)
            messages[target]["content"] = body + "\n\n" + context["notes"][turn + 1]
            ids += runner.render_delta(tokenizer, messages)
        else:
            ids += segment["ids"]
    return {"ids": ids, "positions": positions, "targets": targets}


def _context(notes, role=None):
    c = {"first_message": "Do it\n\nFEEDBACK", "notes": dict(notes), "final_note": None}
    if role:
        c["note_role"] = role
    return c


def test_without_the_key_every_sequence_is_the_old_one():
    chain = _chain()
    for notes in ({}, {1: "N1"}, {2: "N2"}, {3: "N3"}, {1: "N1", 2: "N2", 3: "N3"}):
        assert runner.teacher_sequence(TOK, chain, _context(notes)) == _old_teacher_sequence(TOK, chain, _context(notes))
    # and the old placement is what it was: inside the tool result
    text = TOK.decode(runner.teacher_sequence(TOK, chain, _context({2: "N2"}))["ids"])
    assert "<tool>r1\n\nN2<E><assistant>" in text and "<user>N2" not in text


def test_a_user_role_note_is_its_own_user_message_after_the_tool_result():
    chain = _chain()
    plain = runner.teacher_sequence(TOK, chain, _context({}))
    user = runner.teacher_sequence(TOK, chain, _context({2: "N2", 3: "N3"}, "user"))
    text = TOK.decode(user["ids"])
    assert "<tool>r1<E><user>N2<E><assistant>call two" in text
    assert "<tool>r2<E><user>N3<E><assistant>All done." in text
    assert "r1\n\nN2" not in text and "r2\n\nN3" not in text
    # the student's own tokens are the targets, unchanged, and each target's position still predicts that token
    assert user["targets"] == plain["targets"] == runner.student_sequence(chain)["targets"]
    for pos, target in zip(user["positions"], user["targets"]):
        assert user["ids"][pos + 1] == target


def test_a_turn_one_note_joins_the_first_message_either_way():
    chain = _chain()
    a = runner.teacher_sequence(TOK, chain, _context({1: "N1"}))
    b = runner.teacher_sequence(TOK, chain, _context({1: "N1"}, "user"))
    assert a == b
    assert "<user>Do it\n\nFEEDBACK\n\nN1<E>" in TOK.decode(a["ids"])


def test_the_cut_prefix_ends_with_the_user_note_and_the_generation_prompt():
    chain = _chain()
    position = 3  # the action of turn 2
    prefix = runner.engine_teacher_prefix(TOK, chain, position, _context({2: "N2"}, "user"))
    assert TOK.decode(prefix).endswith("<tool>r1<E><user>N2<E><assistant>")
    old = runner.engine_teacher_prefix(TOK, chain, position, _context({2: "N2"}))
    assert TOK.decode(old).endswith("<tool>r1\n\nN2<E><assistant>")


def test_teacher_context_carries_note_role_only_when_the_rewrite_has_it(monkeypatch):
    context = {"first_message": "engine", "notes": {}, "final_note": None}
    digest = sdpo_a3.context_digest(context)
    monkeypatch.setattr(sdpo_a3, "_REWRITES", {digest: {
        "UU": {"first_message": "m", "notes": {"2": "N2"}, "final_note": None, "note_role": "user"},
        "PP": {"first_message": "m", "notes": {"2": "N2"}, "final_note": None}}})
    monkeypatch.setitem(sdpo_a3.ARMS, "T-UU", {"rewrite": "UU"})
    monkeypatch.setitem(sdpo_a3.ARMS, "T-PP", {"rewrite": "PP"})
    u = sdpo_a3.teacher_context("T-UU", context)
    p = sdpo_a3.teacher_context("T-PP", context)
    assert u["note_role"] == "user" and u["notes"] == {2: "N2"}
    assert "note_role" not in p and p["notes"] == {2: "N2"}
    # the item digest is taken on the engine context, never on the rewrite, so U arms share the base arm's items
    assert sdpo_a3.context_digest(context) == digest
