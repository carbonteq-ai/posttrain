"""r3 (20 Nov, weekend/sdpo-v6/PLAN.md "r3"): the four rewrites of V5 (feedback_versions.rewrite_w1..w4, w4_guard,
r3_check). The other feedback versions are tested by the weekend/engine-req tools."""
import feedback_versions as fv
# ------------------------------------------------------------------ r3 rewrites (20 Nov)
def _v5_ctx(lines, notes=None, final=None):
    first = f"Do the task.\n\n{fv.OLD_HEAD}\n\n" + "\n".join(lines) + f"\n\n{fv.OLD_TAIL}"
    return {"first_message": first, "notes": notes or {}, "final_note": final}


FORBID_LINE = 'At turn 3, gmail_send_email produced a gmail message to "a@x.com". This task forbids it.'
MUST_NOT_NOTE = (fv.NOTE_OLD + 'gmail_send_email produced a gmail message to "c@x.com" with body containing "FA-3"; the task '
                 'requires one with body not containing "FA-3".')


def test_r3_w1_adds_the_guide_only_with_a_prohibition():
    v5 = _v5_ctx([FORBID_LINE, "Still needed: a slack message."])
    w1 = fv.rewrite_w1(v5)
    assert fv.FORBID_GUIDE in w1["first_message"] and fv.r3_check(v5, w1, "W1") == []
    plain = _v5_ctx(["Still needed: a slack message."])
    assert fv.rewrite_w1(plain) is plain


def test_r3_w2_generic_note_names_no_value():
    note = fv.NOTE_OLD + 'gmail_send_email produced a gmail message to "a@x.com". This task forbids it.'
    v5 = _v5_ctx([FORBID_LINE], notes={3: note, 5: MUST_NOT_NOTE, 7: fv.NOTE_OLD + "jira_create_issue returned an error: x"})
    w2 = fv.rewrite_w2(v5)
    assert w2["notes"][3] == fv.NOTE_OLD + fv.W2_SENTENCE
    assert '"FA-3"' not in w2["notes"][5] and w2["notes"][5].startswith(fv.NOTE_OLD + fv.W2_SENTENCE)
    assert w2["notes"][7] == v5["notes"][7] and w2["first_message"] == v5["first_message"]
    assert fv.r3_check(v5, w2, "W2") == []


def test_r3_w3_drops_still_notes_and_the_final_note():
    v5 = _v5_ctx(["Still needed: a slack message."],
                 notes={4: fv.NOTE_OLD + "Still not done at this point:\n1. a slack message", 2: fv.NOTE_OLD + "x returned an error: y"},
                 final={"line": "Still not done at this point: (1) a slack message.", "turn": 4})
    w3 = fv.rewrite_w3(v5)
    assert list(w3["notes"]) == [2] and w3["final_note"] is None and fv.r3_check(v5, w3, "W3") == []


def test_r3_w4_guard_and_wording():
    hidden = {"facts": [{"id": "f1", "text": f"Not done: a gmail message with body containing {fv.PLACEHOLDER}."}]}
    shown = {"facts": [{"id": "f1", "text": 'Not done: a gmail message with body containing "60,400".'}],
             "fix_hints": {"lines": []}}
    v5 = _v5_ctx(["Still needed: a gmail message, whose body needs a value you have not read (still missing)."])
    w4 = fv.rewrite_w4(v5, hidden, shown)
    assert fv.W4_ONE in w4["first_message"] and fv.r3_check(v5, w4, "W4") == []
    located = {**shown, "fix_hints": {"lines": [{"kind": "value_location", "value": "60,400", "line": "In the initial world, ..."}]}}
    assert fv.rewrite_w4(v5, hidden, located) is v5  # the engine located the value somewhere: no "not written" claim
    unaligned = {"facts": [{"id": "f1", "text": "Not done: something else."}], "fix_hints": {"lines": []}}
    assert fv.rewrite_w4(v5, hidden, unaligned) is v5
