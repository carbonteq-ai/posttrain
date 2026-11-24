"""Arms 2/3 loss tokens (plan of 15 Nov, Ali's default 3): reasoning and tool calls are trained, visible text is not."""
import json
import random

import pytest

import sdpo_a3
import feedback_versions
import sdpo_ab_runner as runner

THINK, OPEN, CLOSE, NL, NL2, END = 80, 90, 91, 70, 71, 92


class _Tok:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"</think>": THINK, "<tool_call>": OPEN, "</tool_call>": CLOSE}[token]

    def encode(self, text, add_special_tokens=False):
        return {"\n": [NL], "\n\n": [NL2]}[text]


def _student(targets, turns):
    n = len(targets)
    return {"ids": list(range(n + 10)), "positions": [100 + i for i in range(n)], "targets": targets,
            "old": [-0.5] * n, "turns": turns}


def test_reasoning_and_calls_are_kept_and_visible_text_is_not():
    # turn 1: reasoning (1 2), </think>, newline, call (OPEN 5 CLOSE), text (3), end
    # turn 2: reasoning (4), </think>, text answer (6 7), end      turn 3: cut inside the thinking (8 9)
    targets = [1, 2, THINK, NL, OPEN, 5, CLOSE, 3, END, 4, THINK, 6, 7, END, 8, 9]
    turns = [1] * 9 + [2] * 5 + [3] * 2
    keep = runner.reasoning_and_call_indices(_Tok(), _student(targets, turns))
    assert [targets[i] for i in keep] == [1, 2, THINK, OPEN, 5, CLOSE, 4, THINK, 8, 9]
    assert not {3, 6, 7, NL, END} & {targets[i] for i in keep}


def test_same_classes_as_the_a4_token_classifier():
    rng = random.Random(0)
    vocab = [1, 2, 3, THINK, OPEN, CLOSE, NL, NL2, END]
    for _ in range(500):
        n = rng.randrange(1, 40)
        targets = [rng.choice(vocab) for _ in range(n)]
        turns = sorted(rng.randrange(1, 4) for _ in range(n))
        classes = sdpo_a3.token_classes(targets, turns, THINK, OPEN, CLOSE, {NL, NL2})
        expected = [i for i, (c, t) in enumerate(zip(classes, targets))
                    if c in ("reasoning", "call") or (c == "decision" and t == OPEN)]
        assert runner.reasoning_and_call_indices(_Tok(), _student(targets, turns)) == expected


def test_prepare_sequences_reasoning_and_calls(monkeypatch):
    targets = [1, THINK, NL, OPEN, 5, CLOSE, 3, END]
    student = _student(targets, [1] * 8)
    teacher = {"ids": list(range(30)), "positions": [300 + i for i in range(8)], "targets": list(targets)}
    monkeypatch.setattr(runner, "student_sequence", lambda chain: {k: list(v) for k, v in student.items()})
    monkeypatch.setattr(runner, "teacher_sequence", lambda tokenizer, chain, context: {k: list(v) for k, v in teacher.items()})
    q = runner.prepare_sequences(_Tok(), {}, {"facts": []}, "reasoning+calls")
    assert q["student"]["targets"] == [1, THINK, OPEN, 5, CLOSE] == q["teacher"]["targets"]
    assert q["student"]["positions"] == [100, 101, 103, 104, 105] and q["teacher"]["positions"] == [300, 301, 303, 304, 305]
    assert q["anchor"] is None and q["decision"] is None and q["assistant_tokens"] == 8


# ------------------------------------------------------------------ leak check v2 (15 Nov)

def _row(reasoning, visible_tool="", content=""):
    return {"messages": [{"role": "system", "content": "You are an agent."},
                         {"role": "user", "content": "Post the PTO summary. The Notes column says who is excluded."},
                         {"role": "tool", "content": visible_tool},
                         {"role": "assistant", "reasoning_content": reasoning, "content": content}],
            "trainable_message_indices": [3]}


_CTX = {"first_message": "Task...\n\nThe following is feedback from your unsuccessful earlier attempt:\n\nAt turn 3, "
                         "slack_send_channel_message produced a slack message in channel \"benefits\" with text containing "
                         "\"Felix Braun\"; this task forbids text containing \"Felix Braun\" in that slack message.",
        "notes": {}, "final_note": None}


def test_leak_v2_ignores_ordinary_task_words():
    # v1 flags these (the note / feedback / hints); none of them refers to the teacher's feedback
    row = _row("The note says Sam handles pricing. The customer feedback is positive. The column names give hints.",
               visible_tool="Notes: Sam handles pricing")
    assert any(f["kind"] == "feedback_phrase" for f in sdpo_a3.leak_findings(row, _CTX))
    assert sdpo_a3.leak_findings_v2(row, _CTX) == []


def test_leak_v2_catches_references_and_copied_feedback_wording():
    said = _row("The feedback mentions that previous attempts failed, so I will leave Felix out.")
    kinds = {f["kind"] for f in sdpo_a3.leak_findings_v2(said, _CTX)}
    assert kinds == {"feedback_only_phrase"}
    copied = _row("I must not send a slack message in channel benefits with text containing Felix Braun.")
    assert {f["kind"] for f in sdpo_a3.leak_findings_v2(copied, _CTX)} == {"feedback_words"}
    # the same words are no leak when the student saw them in a tool result
    seen = _row("I must not send a slack message in channel benefits with text containing Felix Braun.",
                visible_tool="policy: never send a slack message in channel benefits with text containing Felix Braun")
    assert sdpo_a3.leak_findings_v2(seen, _CTX) == []


# ------------------------------------------------------------------ feedback versions N and P (15 Nov)

def test_rewritten_arms_read_their_version_by_digest_and_drop_the_notes(monkeypatch):
    ctx = {"first_message": "Task\n\nThe following is feedback from your unsuccessful earlier attempt:\n\nNot done: x."
                            "\n\nCorrectly solve the original question.", "notes": {3: "Note on your next action: y"},
           "final_note": {"kind": "final_not_done"}, "solution_episode": None}
    digest = sdpo_a3.context_digest(ctx)
    monkeypatch.setattr(sdpo_a3, "_REWRITES", {digest: {"N": "Task N", "P": "Task P"}})
    n, p = sdpo_a3.teacher_context("T-N", ctx), sdpo_a3.teacher_context("T-P", ctx)
    assert n["first_message"] == "Task N" and p["first_message"] == "Task P"
    assert n["notes"] == {} == p["notes"] and n["final_note"] is None
    assert sdpo_a3.teacher_context("T-A2", ctx) is ctx
    import pytest
    with pytest.raises(KeyError):
        sdpo_a3.teacher_context("T-N", {**ctx, "first_message": "other"})


def test_committed_rewrites_keep_the_task_and_drop_the_old_framing():
    import json as _json
    rewrites = _json.loads((sdpo_a3.REWRITES_FILE.parent / "a3-rewrites-v1.json").read_text())  # the 15-16 Nov file
    assert len(rewrites) == 268
    for r in rewrites.values():
        for version in ("N", "P"):
            text = r[version]
            assert "The following is feedback" not in text and "Correctly solve the original question" not in text
            assert "never observed" not in text or version == "N"
            assert text.endswith("Do the task correctly now.")
        assert r["Q"] == feedback_versions.q_text(r["P"])
        assert (feedback_versions.FORBID_GUIDE in r["Q"]) == (feedback_versions.FORBIDS in r["P"].split(feedback_versions.NEW_HEAD)[1])


def test_version_q_adds_one_guide_line_only_when_a_line_forbids():
    head = "Task\n\n" + feedback_versions.NEW_HEAD + "\n\n"
    plain = head + "Not done: x.\n\nDo the task correctly now."
    assert feedback_versions.q_text(plain) == plain
    forbid = head + 'At turn 3, gmail_send_email produced a gmail message to "a@b"; this task forbids it.\n\nDo the task correctly now.'
    q = feedback_versions.q_text(forbid)
    assert q.count(feedback_versions.FORBID_GUIDE) == 1 and q.endswith("\n\nDo the task correctly now.")
    assert q.replace(feedback_versions.FORBID_GUIDE + "\n\n", "") == forbid
    task_says = "Task: this task forbids nothing.\n\n" + feedback_versions.NEW_HEAD + "\n\nNot done: x.\n\nDo the task correctly now."
    assert feedback_versions.q_text(task_says) == task_says  # only the feedback lines count, not the task text
    import pytest
    with pytest.raises(ValueError):
        feedback_versions.q_text("not a P message")
    ctx = {"first_message": "Task\n\nThe following is feedback from your unsuccessful earlier attempt:\n\nNot done: x."
                            "\n\nCorrectly solve the original question.", "notes": {}, "final_note": None}
    digest = sdpo_a3.context_digest(ctx)
    import sdpo_a3 as a3
    a3._REWRITES, saved = {digest: {"Q": "Task Q"}}, a3._REWRITES
    try:
        assert sdpo_a3.teacher_context("T-Q", ctx)["first_message"] == "Task Q"
    finally:
        a3._REWRITES = saved


# ------------------------------------------------------------------ arms 2/3 classic-train (15 Nov)

def test_classic_train_trains_only_failed_scored_attempts_of_the_fit_tasks(monkeypatch):
    import sdpo_classic
    import sdpo_ab_world

    monkeypatch.setattr(sdpo_ab_world, "task_name", lambda record: record["task"])
    records = {i: {"id": i, "task": t} for i, t in (("a", "x.one"), ("b", "x.one"), ("c", "x.two"), ("d", "x.one"))}
    outcomes = {"a": {"success": 0.0}, "b": {"success": 1.0}, "c": {"success": 0.0},
                "d": {"success": 0.0, "infrastructure_failure": True}}
    kept = sdpo_classic.admitted_attempts(records, outcomes, frozenset({"x.one"}))
    assert [r["id"] for r in kept] == ["a"]


def test_classic_train_flags_parse_and_default_off():
    args = runner.build_parser().parse_args(["--phase", "classic-train", "--model", "m", "--output", "o",
                                             "--classic-version", "P", "--loss-tokens", "reasoning+calls"])
    assert args.classic_version == "P" and args.classic_batch == 16 and args.classic_limit == 0
    assert not args.classic_longest_first and args.loss_tokens == "reasoning+calls"


def test_leak_v2_reads_a_final_note_without_a_line():
    ctx = {**_CTX, "notes": {4: "Still not done at this point: x"}, "final_note": {"kind": "final_not_done", "line": None}}
    assert sdpo_a3.leak_findings_v2(_row("I will post the summary now."), ctx) == []


# ------------------------------------------------------------------ version V1 (16 Nov, Step 3 change 1)

_OLD = ("Task\n\n" + feedback_versions.OLD_HEAD + "\n\n"
        'Not done: a gmail message to a value this attempt never observed with body containing "X".\n\n'
        + feedback_versions.OLD_TAIL)
_FACT = 'Not done: a gmail message to a value this attempt never observed with body containing "X".'
_NOTE = ('Note on your next action: Still not done at this point: (1) a gmail message to a value this attempt never '
         'observed with body containing "X".')
_V1_CTX = {"first_message": _OLD, "notes": {5: _NOTE},
           "final_note": {"kind": "final_not_done", "turn": 5, "line": _NOTE.split(": ", 1)[1],
                          "texts": ['a gmail message to a value this attempt never observed with body containing "X"']},
           "facts": [{"id": "f1", "text": _FACT}], "solution_episode": None}
_V1_SHOWN = {"facts": [{"id": "f1", "text": _FACT.replace("a value this attempt never observed", '"bob@x.com"')}],
             "fix_hints": {"lines": [
                 {"kind": "value_location", "value": "bob@x.com",
                  "line": 'In the initial world, "bob@x.com" is the email of salesforce contact "Bob" (salesforce_query returns it).'},
                 {"kind": "missed_write", "line": "No call wrote a gmail message."}]}}
_POINTER = 'the email of salesforce contact "Bob" (salesforce_query returns it)'


def test_version_v1_keeps_the_old_layout_with_p_pointers_in_lines_notes_and_final_note():
    v1, counts = feedback_versions.version_v1(_V1_CTX, _V1_SHOWN)
    p, _ = feedback_versions.version_p(_V1_CTX, _V1_SHOWN)
    assert v1["first_message"].startswith("Task\n\n" + feedback_versions.OLD_HEAD)
    assert v1["first_message"].endswith(feedback_versions.OLD_TAIL)
    assert v1["first_message"].split(feedback_versions.OLD_HEAD)[1].strip()[: -len(feedback_versions.OLD_TAIL)].strip() \
        == p["first_message"].split(feedback_versions.NEW_HEAD)[1].strip()[: -len(feedback_versions.NEW_TAIL)].strip()
    assert _POINTER in v1["first_message"] and "No call wrote a gmail message." in v1["first_message"]
    assert list(v1["notes"]) == [5] and _POINTER in v1["notes"][5] and v1["notes"][5].startswith("Note on your next action")
    assert _POINTER in v1["final_note"]["line"] and _POINTER in v1["final_note"]["texts"][0]
    seen = json.dumps([v1["first_message"], v1["notes"], v1["final_note"]])  # what the teacher reads (facts are not)
    assert "never observed" not in seen and "bob@x.com" not in seen
    assert counts["note_by_fact"] == 1 and counts["missed_write"] == 1


def test_version_v1_equals_the_old_context_when_there_is_nothing_to_replace():
    old = {"first_message": "Task\n\n" + feedback_versions.OLD_HEAD + "\n\nAt turn 2, x; this task forbids it.\n\n"
                            + feedback_versions.OLD_TAIL, "notes": {2: "Note on your next action: x; this task forbids it."},
           "final_note": None, "facts": [{"id": "f", "text": "At turn 2, x; this task forbids it."}]}
    v1, _ = feedback_versions.version_v1(old, {"facts": [], "fix_hints": {"lines": []}})
    assert v1 == old


def test_version_v1_falls_back_to_p_wording_without_a_location_and_never_leaves_a_placeholder():
    shown = {"facts": _V1_SHOWN["facts"], "fix_hints": {"lines": []}}
    v1, _ = feedback_versions.version_v1(_V1_CTX, shown)
    assert feedback_versions.NO_PLACE in v1["first_message"] and feedback_versions.NO_PLACE in v1["notes"][5]
    assert feedback_versions.PLACEHOLDER not in json.dumps([v1["first_message"], v1["notes"], v1["final_note"]])


def test_v1_rewrite_is_read_as_a_whole_context_with_integer_note_turns(monkeypatch):
    digest = sdpo_a3.context_digest(_V1_CTX)
    v1, _ = feedback_versions.version_v1(_V1_CTX, _V1_SHOWN)
    stored = json.loads(json.dumps({"V1": {k: v1[k] for k in ("first_message", "notes", "final_note")}}))
    monkeypatch.setattr(sdpo_a3, "_REWRITES", {digest: stored})
    got = sdpo_a3.teacher_context("T-V1", _V1_CTX)
    assert got["first_message"] == v1["first_message"] and got["notes"] == {5: v1["notes"][5]}
    assert got["final_note"] == v1["final_note"]


def test_committed_rewrites_v2_keep_v1_texts_and_cover_every_failed_attempt():
    import json as _json
    v2 = _json.loads((sdpo_a3.REWRITES_FILE.parent / "a3-rewrites-v2.json").read_text())
    v1 = _json.loads((sdpo_a3.REWRITES_FILE.parent / "a3-rewrites-v1.json").read_text())
    assert all(v2[d][k] == v1[d][k] for d in v1 for k in v1[d])
    assert len(v2) == 1087 and all("V1" in r for r in v2.values())
    for r in v2.values():
        text = _json.dumps(r["V1"])
        assert feedback_versions.PLACEHOLDER not in text
        assert r["V1"]["first_message"].endswith(feedback_versions.OLD_TAIL)


# ------------------------------------------------------------------ version V2 (16 Nov, Step 3 change 2)

_F_LINE = 'At turn 7, gmail_send_email produced a gmail message to "alice@x.com"; this task forbids it.'
_V2_CTX = {"first_message": "Task\n\n" + feedback_versions.OLD_HEAD + "\n\n" + _F_LINE + "\n\n" + feedback_versions.OLD_TAIL,
           "notes": {7: 'Note on your next action: gmail_send_email produced a gmail message to "alice@x.com"; this task forbids it.'},
           "final_note": None, "facts": [{"id": "native:3", "turn": 7, "text": _F_LINE}]}
_LOC = {"native:3": ['In the initial world, "alice@x.com" is the "Email" cell of google sheets row "Alice Park" in worksheet '
                     '"Week 1" (google_sheets_get_many_rows returns it).']}


def test_forbid_sentence_says_where_the_record_is_and_nothing_about_a_reason():
    s = feedback_versions.forbid_sentence(_LOC["native:3"][0])
    assert s == ('"alice@x.com" is the "Email" cell of google sheets row "Alice Park" in worksheet "Week 1" '
                 '(google_sheets_get_many_rows returns it); check that record before acting on it.')
    assert feedback_versions.forbid_sentence("something else") is None


def test_version_v2_appends_the_record_sentence_to_the_line_and_the_note_of_that_turn():
    v2, counts = feedback_versions.version_v2(_V2_CTX, {"facts": [], "fix_hints": {"lines": []}}, _LOC)
    s = feedback_versions.forbid_sentence(_LOC["native:3"][0])
    assert v2["first_message"].split("\n\n")[2] == _F_LINE + " " + s
    assert v2["notes"][7].endswith("this task forbids it. " + s)
    assert counts["forbid_line"] == 1 and counts["forbid_note"] == 1


def test_version_v2_equals_v1_without_locations_and_drops_a_sentence_with_a_hidden_value():
    shown = {"facts": [], "fix_hints": {"lines": [{"kind": "value_location", "value": "Alice Park",
                                                   "line": 'In the initial world, "Alice Park" is the name of x "y".'}]}}
    v1, _ = feedback_versions.version_v1(_V2_CTX, shown)
    v2, _ = feedback_versions.version_v2(_V2_CTX, shown, {})
    assert {k: v2[k] for k in ("first_message", "notes")} == {k: v1[k] for k in ("first_message", "notes")}
    v2h, counts = feedback_versions.version_v2(_V2_CTX, shown, _LOC)  # "Alice Park" is a hidden value here
    assert "Alice Park" not in v2h["first_message"] + json.dumps(v2h["notes"])
    assert counts["forbid_dropped_hidden_value"] == 1


def test_committed_rewrites_v3_keep_v1_texts_and_add_v1_v2_for_every_failed_attempt():
    import json as _json
    v3 = _json.loads(sdpo_a3.REWRITES_FILE.read_text())
    v1 = _json.loads((sdpo_a3.REWRITES_FILE.parent / "a3-rewrites-v1.json").read_text())
    assert all(v3[d][k] == v1[d][k] for d in v1 for k in v1[d])
    assert len(v3) == 1087
    for r in v3.values():
        a, b = r["V1"], r["V2"]
        assert feedback_versions.PLACEHOLDER not in _json.dumps(a) + _json.dumps(b)
        la = a["first_message"].split("\n")
        lb = b["first_message"].split("\n")
        assert len(la) == len(lb) and all(y.startswith(x) for x, y in zip(la, lb))
        assert set(a["notes"]) == set(b["notes"]) and all(b["notes"][k].startswith(a["notes"][k]) for k in a["notes"])


def test_version_v1_never_gives_one_fact_the_pointer_of_another_with_the_same_wording():
    hidden = "Not done: a gmail message a value this attempt never observed is required to be marked as read."
    ctx = {"first_message": "Task\n\n" + feedback_versions.OLD_HEAD + "\n\n" + hidden + "\n\n" + feedback_versions.OLD_TAIL,
           "notes": {4: "Note on your next action: Still not done at this point: (1) a gmail message a value this attempt "
                         "never observed is required to be marked as read; (2) a gmail message a value this attempt never "
                         "observed is required to be marked as read."},
           "final_note": None,
           "facts": [{"id": "n1", "text": hidden}, {"id": "n2", "text": hidden}]}
    shown = {"facts": [{"id": "n1", "text": hidden.replace("a value this attempt never observed", '"m1"')},
                       {"id": "n2", "text": hidden.replace("a value this attempt never observed", '"m2"')}],
             "fix_hints": {"lines": [
                 {"kind": "value_location", "value": "m1", "line": 'In the initial world, "m1" is the id of gmail message "A" (gmail_list_emails returns it).'},
                 {"kind": "value_location", "value": "m2", "line": 'In the initial world, "m2" is the id of gmail message "B" (gmail_list_emails returns it).'}]}}
    v1, counts = feedback_versions.version_v1(ctx, shown)
    seen = v1["first_message"] + json.dumps(v1["notes"])
    assert 'gmail message "A"' not in seen and 'gmail message "B"' not in seen
    assert feedback_versions.NO_PLACE in v1["first_message"] and feedback_versions.NO_PLACE in v1["notes"][4]
    assert counts["line_ambiguous"] == 1 and counts["note_ambiguous_core"] == 1


# ------------------------------------------------------------------ changes 3 and 4 (16 Nov, Step 3)

_TE_LINE = "At turn 6, google_sheets_find_worksheet failed: Worksheet 'Schedule' not found in spreadsheet 'ss_calendar'."
_TE_CTX = {"first_message": "Task\n\n" + feedback_versions.OLD_HEAD + "\n\n" + _TE_LINE + "\n\n" + feedback_versions.OLD_TAIL,
           "notes": {6: "Note on your next action: google_sheets_find_worksheet failed: Worksheet 'Schedule' not found in "
                         "spreadsheet 'ss_calendar'."}, "final_note": None, "facts": []}
_CF = ('The value "Schedule" given at turn 6 is the title of google sheets worksheet (id "ws_schedule") of google sheets '
       'spreadsheet "Social Calendar" (id "ss_calendar") in the initial world.')


def test_call_fix_lines_go_to_the_error_line_and_the_note_of_their_turn():
    shown = {"facts": [], "fix_hints": {"lines": [{"kind": "call_fix", "turn": 6, "tool": "google_sheets_find_worksheet",
                                                   "line": _CF}]}}
    out, counts = feedback_versions.add_call_fixes(_TE_CTX, _TE_CTX, shown)
    assert out["first_message"].split("\n\n")[2] == _TE_LINE + " " + _CF
    assert out["notes"][6].endswith(_CF) and counts == {"call_fix_line": 1, "call_fix_note": 1}
    hidden = {"facts": [], "fix_hints": {"lines": shown["fix_hints"]["lines"] + [
        {"kind": "value_location", "value": "ws_schedule", "line": 'In the initial world, "ws_schedule" is the id of x "y".'}]}}
    out2, counts2 = feedback_versions.add_call_fixes(_TE_CTX, _TE_CTX, hidden)
    assert out2["first_message"] == _TE_CTX["first_message"] and counts2 == {"call_fix_dropped_hidden_value": 1}


def test_error_check_names_a_lookup_tool_only_when_the_task_has_it():
    tools = ["google_sheets_find_worksheet", "google_sheets_get_spreadsheet_by_id"]
    out, counts = feedback_versions.add_error_checks(_TE_CTX, tools)
    want = ("Use an exact worksheet title or ID. google_sheets_get_spreadsheet_by_id returns a spreadsheet with its "
            "worksheets.")
    assert out["first_message"].split("\n\n")[2] == _TE_LINE + " " + want
    assert out["notes"][6].endswith(want) and counts == {"error_check_line": 1, "error_check_note": 1}
    out2, _ = feedback_versions.add_error_checks(_TE_CTX, ["google_sheets_find_worksheet"])
    assert out2["first_message"].split("\n\n")[2] == _TE_LINE + " Use an exact worksheet title or ID."
    shown = {"facts": [], "fix_hints": {"lines": [{"kind": "call_fix", "turn": 6, "tool": "google_sheets_find_worksheet",
                                                   "line": _CF}]}}
    out3, counts3 = feedback_versions.add_error_checks(_TE_CTX, tools, shown)
    assert out3 == {**_TE_CTX, "notes": dict(_TE_CTX["notes"])} and counts3 == {"error_check_skipped_call_fix": 1}


def test_error_check_rules_and_placement():
    ec = feedback_versions.error_check
    assert ec("slack_send_channel_message", "Channel 'ops' not found", ["slack_list_channels"]) == \
        "Use an exact channel name or ID. slack_list_channels lists all channels."
    assert ec("slack_list_channels", "Channel 'ops' not found", ["slack_list_channels"]) == "Use an exact channel name or ID."
    assert ec("google_ads_set_campaign_status", "Campaign not found in this account", ["google_ads_find_campaign_by_name"]) \
        == "The campaign is in another account. Check the mainAccountId value."
    assert ec("google_ads_set_campaign_status", "Campaign not found", ["google_ads_find_campaign_by_name"]).endswith(
        "google_ads_find_campaign_by_name finds a campaign by name.")
    assert ec("salesforce_query", "Invalid SOQL WHERE clause: Unsupported operator 'BY'.", []) is None
    assert ec("google_sheets_update_row", "Row row_id:2 not found in worksheet 'ws_a' (and 2 other "
              "google_sheets_update_row errors at turn 4)", []).startswith("The row argument takes")
    ctx = {"first_message": "Task\n\n" + feedback_versions.OLD_HEAD + "\n\n"
                            "At turns 2, 9, slack_send_channel_message failed: Channel 'ops' not found\n\n"
                            + feedback_versions.OLD_TAIL,
           "notes": {2: "Note on your next action: slack_send_channel_message failed: Channel 'ops' not found "
                        "gmail_send_email failed: bad", 9: "Note on your next action: slack_send_channel_message failed: "
                        "Channel 'ops' not found"}}
    out, counts = feedback_versions.add_error_checks(ctx, [])
    assert out["notes"][2] == ("Note on your next action: slack_send_channel_message failed: Channel 'ops' not found "
                               "Use an exact channel name or ID. gmail_send_email failed: bad")
    assert out["notes"][9].endswith("not found Use an exact channel name or ID.")
    assert counts == {"error_check_line": 1, "error_check_note": 2}


def test_calm_wording_keeps_every_fact_and_removes_double_negatives():
    old = {"first_message": "Task\n\n" + feedback_versions.OLD_HEAD + "\n\n"
                            'Not done: a slack message in channel "ops" exists, but none with text containing "Q3".\n'
                            'Not done: no gmail message label message "m1" label "vip" exists; one is required.\n'
                            'At turn 4, gmail_send_email produced a gmail message to "a@x.com"; this task forbids it.\n'
                            + _TE_LINE + "\n\n" + feedback_versions.OLD_TAIL,
           "notes": {4: 'Note on your next action: gmail_send_email produced a gmail message to "a@x.com"; this task forbids it.',
                     6: 'Note on your next action: Still not done at this point: (1) a slack message in channel "ops" '
                        'exists, but none with text containing "Q3"; (2) a slack message in channel "ops" exists, but '
                        'none with text containing "Q3".'}}
    new = feedback_versions.version_calm(old)
    assert feedback_versions.calm_check(old, new) == []
    lines = new["first_message"].split("\n\n")[2].split("\n")
    assert lines[0] == 'Still needed: a slack message in channel "ops" with text containing "Q3" (still missing).'
    assert lines[1] == 'Still needed: gmail message label message "m1" label "vip".'
    assert lines[2] == 'At turn 4, gmail_send_email produced a gmail message to "a@x.com". This task forbids it.'
    assert "returned an error:" in lines[3] and "failed:" not in lines[3]
    assert new["first_message"].split("\n\n")[1] == feedback_versions.CALM_HEAD
    assert new["first_message"].endswith(feedback_versions.CALM_TAIL)
    assert new["notes"][4].startswith("Note: ") and "none with" not in new["notes"][6]
    assert "exists, but" not in new["first_message"] + new["notes"][6]


def test_calm_short_statements_keep_every_fact():
    head = "Task\n\n" + feedback_versions.OLD_HEAD + "\n\n"
    rec = '"a@x.com" is the email of salesforce contact "Al" (salesforce_contact_find returns it); ' + feedback_versions.FORBID_CHECK
    rec2 = '"b@x.com" is the email of salesforce contact "Bo" (salesforce_contact_find returns it); ' + feedback_versions.FORBID_CHECK
    old = {"first_message": head
           + 'At turn 3, gmail_send_email produced 2 gmail messages; this task forbids them. ' + rec + " " + rec2 + "\n"
           + "At turn 6, salesforce_contact_update failed: Contact with id '00Q1' not found The value \"00Q1\" given at turn 6 "
             'is the id of salesforce lead "Jen" in the initial world, not a contact.\n'
           + 'At turn 2, google_sheets_update_row updated a google sheets row in spreadsheet "s" worksheet "w" with cell "A" = '
             'not "x", with cell "A" = not "y".\n'
           + 'At turn 4, notion_create_page produced a page (with values containing "0.12" or with values containing "12%").'
           + "\n\n" + feedback_versions.OLD_TAIL, "notes": {}}
    new = feedback_versions.version_calm(old)
    assert feedback_versions.calm_check(old, new) == []
    lines = new["first_message"].split("\n\n")[2].split("\n")
    assert lines[0].count(feedback_versions.FORBID_CHECK) == 1 and lines[0].endswith(feedback_versions.FORBID_ONCE)  # R1
    assert '"00Q1" given at turn 6 is not a contact. It is the id of salesforce lead "Jen" in the initial world.' in lines[1]  # R2
    assert "not found. The value" in lines[1]                                                       # R5
    assert 'worksheet "w". Its cell "A" = not any of "x", "y".' in lines[2]                      # R6 + R3
    assert '(with values containing "0.12" or "12%")' in lines[3]                                  # R4
    one = {"first_message": head + 'At turn 3, x failed: e. ' + rec + "\n\n" + feedback_versions.OLD_TAIL, "notes": {}}
    assert feedback_versions.FORBID_ONCE not in feedback_versions.version_calm(one)["first_message"]  # R1 only for 2+


def test_calm_cell_split_never_adds_a_repeat():
    note = ('Note on your next action: google_sheets_update_row updated a google sheets row in spreadsheet "s" worksheet "w" '
            'with cell "A" = "1". google_sheets_update_row updated a google sheets row in spreadsheet "s" worksheet "w" '
            'with cell "A" = "2".')
    old = {"first_message": "Task\n\n" + feedback_versions.OLD_HEAD + "\n\nx\n\n" + feedback_versions.OLD_TAIL, "notes": {5: note}}
    new = feedback_versions.version_calm(old)
    assert "Its cell" not in new["notes"][5] and feedback_versions.calm_check(old, new) == []


def test_calm_check_catches_a_lost_or_added_fact():
    old = {"first_message": "Task\n\n" + feedback_versions.OLD_HEAD + '\n\nNot done: a row with "A".\n\n'
                            + feedback_versions.OLD_TAIL, "notes": {}}
    new = feedback_versions.version_calm(old)
    assert feedback_versions.calm_check(old, new) == []
    bad = {**new, "first_message": new["first_message"].replace('"A"', '"B"')}
    assert feedback_versions.calm_check(old, bad)


def test_say_unread_removes_every_stand_in_and_keeps_every_fact():
    u = feedback_versions.NO_PLACE_NEW
    calm = {"first_message": "Task\n\n" + feedback_versions.CALM_HEAD + "\n\n"
            f'Still needed: a gmail message to "a@x.com" with body containing {u} (still missing).\n'
            f'Still needed: a gmail message to {u} with body containing "Q3".\n'
            f'Still needed: a slack message in channel "ops" with text containing "Q3" and {u}, {u} (still missing).\n'
            f'At turn 4, helpscout_add_note added a helpscout conversation note on conversation "h1" with body containing {u}.'
            ' This task forbids the body shown in that helpscout conversation note.\n'
            f'Still needed: a google sheets row in spreadsheet "s" worksheet "w" with cells Subject = "Hi"; Assignee = {u} (still missing).\n'
            f'Still needed: a docusign envelope signer for {u}.\n'
            f'Still needed: a gmail message to "c@x.com" is required with body not containing {u}.\n'
            f'Still needed: the attempt sent no slack message in channel {u}; one is required.\n\n' + feedback_versions.CALM_TAIL,
            "notes": {6: f'Note: Still needed: (1) a slack message in channel "ops" with text containing {u} (still missing). '
                         '(2) a gmail message to "b@x.com" with body containing "Hi" (still missing).'}}
    new = feedback_versions.version_unread(calm)
    assert feedback_versions.unread_check(calm, new) == []
    lines = new["first_message"].split("\n\n")[2].split("\n")
    assert lines[0] == 'Still needed: a gmail message to "a@x.com", whose body needs a value you have not read (still missing).'
    assert lines[1] == 'Still needed: a gmail message with body containing "Q3", whose recipient is one you have not read.'
    assert lines[2] == ('Still needed: a slack message in channel "ops" with text containing "Q3", whose text also needs values '
                        'you have not read (still missing).')
    assert lines[3].startswith('At turn 4, helpscout_add_note added a helpscout conversation note on conversation "h1", '
                               'whose body held a value you have not read. This task forbids')
    assert lines[4].endswith('with cells Subject = "Hi", whose Assignee is one you have not read (still missing).')
    assert lines[5] == "Still needed: a docusign envelope signer for a value you have not read."  # no field: words in place
    assert lines[6] == ('Still needed: a gmail message to "c@x.com" is required, whose body has to leave out a value you '
                        'have not read.')                                                  # "not containing" kept as a must-not
    assert lines[7] == "Still needed: the attempt sent no slack message in a channel unknown to you; one is required."
    assert new["notes"][6].startswith('Note: Still needed: (1) a slack message in channel "ops", whose text needs a value')
    assert u not in new["first_message"] + new["notes"][6]


def test_unread_check_catches_a_stand_in_a_changed_value_or_a_lost_word():
    u = feedback_versions.NO_PLACE_NEW
    calm = {"first_message": f'Still needed: a gmail message to "a@x.com" with body containing {u} (still missing).',
            "notes": {}}
    new = feedback_versions.version_unread(calm)
    assert feedback_versions.unread_check(calm, new) == []
    assert feedback_versions.unread_check(calm, calm)                                          # stand-in left
    assert feedback_versions.unread_check(calm, {**new, "first_message": new["first_message"].replace("a@x", "b@x")})
    assert feedback_versions.unread_check(calm, {**new, "first_message": new["first_message"].replace("gmail ", "")})
    neg = {"first_message": 'Still needed: a gmail message to "c@x.com" with subject not "Hi" and body not containing '
                            + u + ".", "notes": {}}
    out = feedback_versions.version_unread(neg)
    assert "to leave out" in out["first_message"] and feedback_versions.unread_check(neg, out) == []
    assert feedback_versions.unread_check(neg, {**out, "first_message": out["first_message"].replace('not "Hi"', '"Hi"')})


def test_tidy_fixes_punctuation_field_words_and_qualifier_repeats():
    head = "Task\n\n" + feedback_versions.CALM_HEAD + "\n\n"
    before = {"first_message": head
              + 'At turn 7,. x_tool returned an error: Worksheet "a,. b" not found.\n'
              + 'Still needed: a gmail message to the from of gmail message "Policy" with body containing "Ben".\n'
              + 'At turn 4, g_send produced a gmail message with body containing "JM". This task forbids body containing '
                '"JM" in that gmail message for "Clara".\n'
              + 'At turn 4, g_send produced a gmail message with body containing "JM". This task forbids body containing '
                '"JM" in that gmail message for "Elena".\n'
              + 'At turn 5, g_send produced a gmail message to "s@x.com". This task forbids it for "Support".\n\n'
              + feedback_versions.CALM_TAIL, "notes": {}}
    after = feedback_versions.version_tidy(before)
    assert feedback_versions.tidy_check(before, after) == []
    lines = after["first_message"].split("\n\n")[2].split("\n")
    assert lines[0] == 'At turn 7, x_tool returned an error: Worksheet "a,. b" not found.'      # quoted value untouched
    assert lines[1] == 'Still needed: a gmail message to the sender of gmail message "Policy" with body containing "Ben".'
    assert lines[2] == ('At turn 4, g_send produced a gmail message with body containing "JM". This task forbids body '
                        'containing "JM" in that gmail message.')                                  # said once
    assert lines[3] == 'At turn 5, g_send produced a gmail message to "s@x.com". This task forbids it for "Support".'
    assert len(lines) == 4
    lost = {**after, "first_message": after["first_message"].replace(lines[3] + "\n", "")}
    assert feedback_versions.tidy_check(before, lost)                                              # a fact dropped
    changed = {**after, "first_message": after["first_message"].replace('"Ben"', '"Bo"')}
    assert feedback_versions.tidy_check(before, changed)                                           # a value changed
    note = ('Note: g_send produced a gmail message with body containing "JM". This task forbids body containing "JM" in '
            'that gmail message for "Greta". g_send produced a gmail message to "a@x.com". This task forbids body '
            'containing "JM" in that gmail message. g_send produced a gmail message to "b@x.com".')
    ctx = {"first_message": before["first_message"], "notes": {4: note}}
    out = feedback_versions.version_tidy(ctx)
    assert feedback_versions.tidy_check(ctx, out) == []
    assert out["notes"][4] == note                    # two different facts (different messages): nothing merged
    note2 = ('Note: g_send produced a gmail message with body containing "JM". This task forbids it for "Greta". '
             'g_send produced a gmail message to "b@x.com". This task forbids it. '
             'g_send produced a gmail message with body containing "JM". This task forbids it for "Clara".')
    ctx2 = {"first_message": before["first_message"], "notes": {4: note2}}
    out2 = feedback_versions.version_tidy(ctx2)
    assert feedback_versions.tidy_check(ctx2, out2) == []
    assert out2["notes"][4] == ('Note: g_send produced a gmail message with body containing "JM". This task forbids it. '
                                'g_send produced a gmail message to "b@x.com". This task forbids it.')  # whole fact merged
    twice = {**out2, "notes": {4: out2["notes"][4] + ' g_send produced a gmail message to "b@x.com". This task forbids it.'}}
    assert feedback_versions.tidy_check(ctx2, twice)                                               # a repeat is caught
    orphan = {**out2, "notes": {4: out2["notes"][4].replace(' g_send produced a gmail message to "b@x.com".', '')}}
    assert feedback_versions.tidy_check(ctx2, orphan)                          # "This task forbids it" moved to another fact


def test_feedback_text_finds_every_heading():
    import sdpo_a3
    for head in (feedback_versions.OLD_HEAD, feedback_versions.CALM_HEAD, "Results of an earlier attempt at this same task:"):
        ctx = {"first_message": "Task: email Bob.\n\n" + head + "\n\nStill needed: x.", "notes": {}, "final_note": None}
        text = sdpo_a3.feedback_text(ctx)
        assert text.startswith(head) and "email Bob" not in text


def test_rewritten_version_reads_the_shipped_text(monkeypatch):
    """Step 5 (17 Nov): a Step 3 version trains with exactly the rewrite the solve tests measured, looked up by the
    digest of the T-A2 context; a context without a rewrite is refused (KeyError), never trained on other text."""
    import pytest

    import sdpo_a3
    import sdpo_classic

    base = {"first_message": "Task\n\nThe following is feedback from your unsuccessful earlier attempt:\n\nx\n\nCorrectly "
                             "solve the original question.", "notes": {3: "Note on your next action: x"}, "final_note": None}
    digest = sdpo_a3.context_digest(base)
    calm = {"first_message": "Task\n\nNotes on an earlier attempt at this task:\n\nx\n\nNow do the task.",
            "notes": {"3": "Note: x"}, "final_note": None}
    monkeypatch.setattr(sdpo_a3, "_REWRITES", {digest: {"V4-32": calm}})
    out, made = sdpo_classic.teacher_context({}, [], "V4-32", lambda *a: dict(base), None)
    assert out["first_message"] == calm["first_message"] and out["notes"] == {3: "Note: x"} and made == {}
    assert sdpo_a3.context_digest(out) != digest  # the audited digest is the text the teacher reads
    other = dict(base, first_message=base["first_message"] + " y")
    with pytest.raises(KeyError):
        sdpo_classic.teacher_context({}, [], "V4-32", lambda *a: dict(other), None)
    assert set(sdpo_classic.REWRITE_VERSIONS) <= {k[2:] for k in sdpo_a3.ARMS}
    assert "all" in sdpo_classic.LOSS_TOKENS and "reasoning+calls" in sdpo_classic.LOSS_TOKENS


def test_version_frame_restores_v1_framing_and_frame_check_catches_other_changes():
    import feedback_versions as F
    v4 = {"first_message": "Do the job.\n\nNotes on an earlier attempt at this task:\n\nAt turn 2, x_send produced a y. "
                           "This task forbids it.\n\nNow do the task.",
          "notes": {"2": "Note: x_send produced a y. This task forbids it."}, "final_note": {"turn": 2}}
    v5 = F.version_frame(v4)
    assert v5["first_message"] == ("Do the job.\n\nThe following is feedback from your unsuccessful earlier attempt:\n\n"
                                   "At turn 2, x_send produced a y. This task forbids it.\n\nCorrectly solve the original question.")
    assert v5["notes"] == {"2": "Note on your next action: x_send produced a y. This task forbids it."}
    assert v5["final_note"] == {"turn": 2}
    assert F.frame_check(v4, v5) == []
    bad = {**v5, "first_message": v5["first_message"].replace("forbids it", "forbids them")}
    assert F.frame_check(v4, bad)
    bad = {**v5, "notes": {"2": v5["notes"]["2"] + " Extra."}}
    assert F.frame_check(v4, bad)
    bad = {**v5, "final_note": {"turn": 3}}
    assert F.frame_check(v4, bad)
    with pytest.raises(ValueError):
        F.version_frame(v5)
