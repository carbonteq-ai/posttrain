"""v12 (21 Nov) readable hidden-value wording and enumeration-safe masking: each rule both ways (it fires where written
and leaves other text alone), plus the readability gate."""
import feedback_c as C
import feedback_z as Z


def red(line, hidden):
    return C._post(C._redact_line(line, lambda v: v not in hidden, __import__("collections").Counter()))


def test_identity_of_entity():
    out = red('Still needed: a gmail message to the recipient of gmail message "msg_7" (gmail_find_email returns it) '
              'with body containing "GlobalTech".', {"msg_7"})
    assert out == ('Still needed: a gmail message to the recipient of a gmail message you have not read '
                   '(gmail_find_email returns it) with body containing "GlobalTech".'), out
    assert not C.unreadable(out)


def test_identity_slot_with_preposition_and_article():
    assert "on a ticket you have not read" in red('Still needed: a zendesk ticket comment on ticket "t1" with body '
                                                  'containing "x".', {"t1"})
    assert "to a user you have not read" in red('Still needed: a slack dm to user "u9", with text containing "y".', {"u9"})
    assert "for an account you have not read" in red('Still needed: a salesforce case for account "a1", with subject '
                                                      '"s".', {"a1"})


def test_seen_value_untouched():
    line = 'Still needed: a zendesk ticket comment on ticket "t1" with body containing "x".'
    assert red(line, set()) == line


def test_worksheet_comma_only_before_column():
    a = red('Still needed: a google sheets row in spreadsheet "ss" worksheet "ws" column "ID", with value "v".', {"ws"})
    assert 'in spreadsheet "ss", in a worksheet you have not read, column "ID"' in a, a
    b = red('Still needed: a google sheets row in spreadsheet "ss" worksheet "ws" with cells Tier = "Gold" (still '
            'missing).', {"ws"})
    assert 'in a worksheet you have not read with cells Tier = "Gold"' in b, b


def test_list_counts_hidden_items():
    out = red('At turn 5, gmail_send_email produced a gmail message with body containing "Intro", without "a", "b", '
              '"c" or a phrase the task states on that line.', {"b"})
    assert 'without "a", "c" or a value you have not read or a phrase' in out, out
    both = red('Still needed: a slack message with text containing "p" and "q" (still missing).', {"q"})
    assert 'containing "p" and a value you have not read' in both, both


def test_forbid_cells_and_cell():
    a = red('At turn 3, google_sheets_update_row updated a google sheets row in spreadsheet "ss" row 3 with cells '
            'containing Status = "Done". This task forbids cells containing Status = "Done" in that google sheets row.',
            {"Done"})
    assert "This task forbids the Status cell shown in that google sheets row." in a, a
    b = red('At turn 9, google_sheets_add_row produced a google sheets row in spreadsheet "ss" worksheet "ws". Its cell '
            '"src" = "Guide". This task forbids cell "src" = "Guide" in that google sheets row.', {"Guide"})
    assert 'Its cell "src" held a value you have not read.' in b and 'This task forbids the cell "src" shown' in b, b


def test_same_value_required_left_out():
    out = red('At turn 14, gmail_send_email produced a gmail message to "ops@x.example.com" with body containing '
              '"Stratos"; the task requires one with body not containing "Stratos".', {"Stratos"})
    assert out == ('At turn 14, gmail_send_email produced a gmail message to "ops@x.example.com", whose body held a '
                   'value you have not read; the task requires one whose body leaves that value out.'), out


def test_keep_check_and_row_label():
    assert "; its group had to stay unchanged" in red('At turn 6, freshdesk_update_ticket changed a freshdesk ticket on '
                                                      'ticket "fd1"; its group had to stay "g1".', {"g1"})
    assert ' row 3, whose' in red('Still needed: a google sheets row in spreadsheet "ss" row 3 ("Bob") with cells '
                                  'containing "30" (still missing).', {"Bob", "30"})


def test_schema_names_never_hidden():
    out = red('Still needed: a salesforce field in "accounts" on record "001" field "description" with value "x".',
              {"accounts", "001"})
    assert 'in "accounts"' in out and 'field "description"' in out, out


def test_unreadable_gate_both_ways():
    assert C.unreadable("Still needed: a gmail message to the recipient of gmail message a value you have not read.")
    assert C.unreadable("At turn 2, x_y produced a row, whose cells Current Load was one you have not read.")
    assert C.unreadable("produced a message, whose the task requires one with body had to leave out a value you have "
                        "not read.")
    assert C.unreadable('(with text "a" or or with text "b")')
    assert not C.unreadable("Still needed: a gmail message, whose body needs a value you have not read.")
    assert not C.unreadable('Still needed: a row, whose Status cell is one you have not read (still missing).')
    assert not C.unreadable('without "a", "c" or 2 values you have not read or a phrase the task states.')


def test_mask_error_keeps_allowed_values():
    err = ("1 validation error for buffer_add_to_queueArguments method Input should be 'queue', 'schedule', "
           "'share_next' or 'draft' (...)")
    assert Z.mask_error(err, {"schedule"}) == (err, 0)
    masked, n = Z.mask_error("Worksheet 'Policy' not found in spreadsheet 'ss_a'.", {"Policy", "ss_a"})
    assert masked == "Worksheet not found in spreadsheet." and n == 2


def test_tool_doc_strings():
    docs = [{"name": "t", "parameters": {"properties": {"status": {"enum": ["sent", "draft"]}}}}]
    s = C.tool_doc_strings(docs)
    assert "sent" in s and "status" in s and "draft" in s


# ------------------------------------------------------------------ v13 (21 Nov, D1-10): R-c2, each rule both ways
HEAD = "The following is feedback from your unsuccessful earlier attempt:"
TAIL = "Correctly solve the original question."


def msg(*lines):
    return "Do the task.\n\n" + HEAD + "\n\n" + "\n".join(lines) + "\n\n" + TAIL


def body(text):
    return Z.split(text)[1]


FORBID = ('At turn 7, slack_send_channel_message produced a slack message in channel "support-ops" with text containing '
          '"billing-error: 0". This task forbids text containing "billing-error: 0" in that slack message.')
STILL = 'Still needed: a slack message in channel "support-ops", whose text needs a value you have not read (still missing).'


def test_required_values_positive_still_needed_only():
    req = C.required_values("\n".join([FORBID, STILL, 'Still needed: a gmail message to "a@x.com" with body not '
                                       'containing "Secret".', 'Still needed: a google sheets row column "URL" with value '
                                       '"u1".']))
    assert req == {"support-ops", "a@x.com", "u1"}, req  # not the forbidden text, the negated value or the column name


def test_rc2_names_forbidden_value_hides_required_target():
    unseen = {"support-ops", "billing-error: 0"}
    text, counts, problems = C.redact(msg(FORBID, STILL), lambda v: v not in unseen, required_only=True)
    b = body(text)
    assert not problems, problems
    assert '"billing-error: 0"' in b[0] and "a channel you have not read, with text containing" in b[0], b[0]
    assert '"support-ops"' not in text and "a channel you have not read" in b[1], b[1]
    # R-c (the default) hides both, as v12 did
    text_c, _c, _p = C.redact(msg(FORBID, STILL), lambda v: v not in unseen)
    assert '"billing-error: 0"' not in text_c and "This task forbids the text shown" in text_c, text_c


def test_rc2_same_as_rc_when_everything_required_or_seen():
    unseen = {"support-ops"}
    a, _c1, _p1 = C.redact(msg(STILL), lambda v: v not in unseen, required_only=True)
    b, _c2, _p2 = C.redact(msg(STILL), lambda v: v not in unseen)
    assert a == b, (a, b)
    seen_all, _c3, _p3 = C.redact(msg(FORBID, STILL), lambda v: True, required_only=True)
    assert seen_all == msg(FORBID, STILL)


def test_rc2_forbidden_value_also_required_stays_hidden():
    both = ('At turn 4, google_sheets_update_row updated a google sheets row in spreadsheet "ss" row 3 ("Ravi") with cells '
            'containing Status = "Processed". This task forbids cells containing Status = "Processed" in that google '
            'sheets row.')
    need = 'Still needed: a google sheets row in spreadsheet "ss" row 2 ("Priya") with cells containing Status = "Processed".'
    text, _c, problems = C.redact(msg(both, need), lambda v: v != "Processed", required_only=True)
    assert not problems and '"Processed"' not in text, text  # the task needs it written for Priya: hidden (M18, F14.4)


def test_unread_with_comma_both_ways():
    assert C.UNREAD_WITH.sub(r"\1, \2", "a slack message in a channel you have not read with text") == \
        "a slack message in a channel you have not read, with text"
    keep = 'a slack message in channel "x" with text'
    assert C.UNREAD_WITH.sub(r"\1, \2", keep) == keep


def test_pointer_notes_true_of_merged_still_lines():
    first = msg('At turn 3, gmail_send_email produced a gmail message to "r@x.com". This task forbids it.',
                'Still needed, 2 of these: a slack message in a channel you have not read (still missing).')
    notes = C.y_pointer_notes(first, {"3": "Note on your next action: gmail_send_email produced a gmail message to "
                                           '"r@x.com". This task forbids it.',
                                      "5": "Note on your next action: Still needed: (1) a slack message ..."})
    assert notes == {3: Z.P_TURN.format(t=3), 5: Z.P_STILL}, notes
    try:
        C.y_pointer_notes(first, {"9": "Note on your next action: something at turn 9 not listed"})
    except ValueError:
        pass
    else:
        raise AssertionError("a note with no true pointer must raise")


# ------------------------------------------------------------------ v14 (21 Nov, round 5): R-d and R-n, both ways
def _engine_on_path():
    import sys
    from pathlib import Path

    src = str(Path(C.__file__).resolve().parent / "abce_vendor" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _turns():
    """turn 1: a read (slack_list_channels shows "leads-priority") and a write echoing "Hot"; turn 2: a reply. (The engine's
    observed test matches a value anywhere in the normalised text, so the channel name must not contain "hot".)"""
    return [{"role": "user", "content": "Route the leads."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "slack_list_channels", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "salesforce_lead_create",
                                                              "arguments": '{"status": "Hot"}'}}]},
            {"role": "tool", "tool_call_id": "a", "name": "slack_list_channels",
             "content": '{"channels": [{"name": "leads-priority"}, {"name": "general"}]}'},
            {"role": "tool", "tool_call_id": "b", "name": "salesforce_lead_create",
             "content": '{"success": true, "status": "Hot"}'},
            {"role": "assistant", "content": "Done."}]


def test_where_shown_reading_tools_only():
    _engine_on_path()
    where = C.where_shown(_turns(), ("slack_list_channels", "salesforce_lead_create"))
    assert where("leads-priority", 1) == "slack_list_channels"
    assert where("Hot", 1) is None  # only a write tool's result shows it: the attempt produced it itself
    assert where("leads-priority", 2) is None  # nothing read from turn 2 on shows it
    assert where("nowhere", 1) is None
    # a value under 4 characters gets no tool ("lea" is inside "leads-priority", "VP" is a word of a title): where it
    # first appears says nothing about where it comes from; 4 characters or more do
    assert where("lea", 1) is None
    words = [{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "tool_calls": [
                 {"id": "a", "type": "function", "function": {"name": "slack_list_channels", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "a", "name": "slack_list_channels", "content": '{"tier": "VP of Sales"}'}]
    assert C.where_shown(words, ())("VP", 1) is None
    assert C.where_shown(words, ())("VP of Sales", 1) == "slack_list_channels"


NEED_CH = 'Still needed: a slack message in channel "hot-leads" with text containing "Acme" (still missing).'
FORBID_CH = ('At turn 3, slack_send_channel_message produced a slack message in channel "hot-leads" with text containing '
             '"x". This task forbids text containing "x" in that slack message.')


def test_name_where_points_once_in_the_still_needed_line():
    tools = {"hot-leads": "slack_list_channels", "Acme": "slack_list_channels"}
    text, counts, problems = C.name_where(msg(FORBID_CH, NEED_CH), lambda v: v == "x", lambda v, t: tools.get(v), 1)
    b = body(text)
    assert not problems, problems
    assert b[0] == FORBID_CH, b[0]  # no pointer outside a still-needed line
    assert b[1] == ('Still needed: a slack message in channel "hot-leads" (read it with slack_list_channels first) with '
                    'text containing "Acme" (still missing).'), b[1]  # one pointer per tool in a line
    assert counts["pointers"] == 1 and counts["values_under_a_pointer_of_the_line"] == 1, counts
    # both ways: everything seen at the cut -> the text is unchanged
    same, c2, _p = C.name_where(msg(FORBID_CH, NEED_CH), lambda v: True, lambda v, t: "slack_list_channels", 1)
    assert same == msg(FORBID_CH, NEED_CH) and not c2.get("pointers"), same


def test_name_where_keeps_engine_pointer_and_negated_values():
    line = ('Still needed: a gmail message to the sender of gmail message "Policy" (gmail_find_email returns it) with body '
            'not containing "Secret".')
    text, counts, _p = C.name_where(msg(line), lambda v: False, lambda v, t: "gmail_find_email", 1)
    assert body(text)[0] == line, body(text)[0]
    assert counts["engine_pointer_kept"] == 1 and not counts.get("pointers"), counts


def test_name_where_hides_a_value_no_reading_tool_shows():
    line = 'Still needed: a google sheets row in spreadsheet "ss" row 2 ("Priya") with cells containing Status = "Processed".'
    text, counts, problems = C.name_where(msg(line), lambda v: v != "Processed", lambda v, t: None, 1)
    assert not problems and '"Processed"' not in text and "you have not read" in text, text
    assert counts["values_written_only_hidden"] == 1, counts
    # both ways: a reading tool shows it -> named with the pointer, nothing hidden
    text2, _c, _p = C.name_where(msg(line), lambda v: v != "Processed", lambda v, t: "google_sheets_get_rows", 1)
    assert '"Processed" (read it with google_sheets_get_rows first)' in text2 and "you have not read" not in text2, text2


def test_at_turn_note_moves_the_cut_turn_once():
    a = 'At turn 2, gmail_send_email produced a gmail message to "r@x.com". This task forbids it.'
    c = 'At turn 5, zendesk_update_ticket changed a zendesk ticket on ticket "t1"; its group had to stay "g1".'
    first, notes = C.at_turn_note(msg(a, c, NEED_CH), 2)
    assert notes == {2: 'Note on your next action: gmail_send_email produced a gmail message to "r@x.com". This task '
                        'forbids it.'}, notes
    assert body(first) == [c, NEED_CH], body(first)
    assert not C.note_repeats(notes[2], first)
    # both ways: no line at the cut -> nothing moves; the only line at the cut -> the summary says where it is
    assert C.at_turn_note(msg(c, NEED_CH), 2) == (msg(c, NEED_CH), {})
    only, notes2 = C.at_turn_note(msg(a), 2)
    assert body(only) == [C.ONLY_NOTE] and 2 in notes2, (only, notes2)


def test_at_turn_note_never_repeats():
    a = 'At turn 2, zendesk_update_ticket changed a zendesk ticket on ticket "t1"; its group had to stay "g1".'
    c = 'At turn 5, zendesk_update_ticket changed a zendesk ticket on ticket "t1"; its priority had to stay 2.'
    assert C.at_turn_note(msg(a, c), 2) == (msg(a, c), {})  # the note would share a clause with turn 5's line
    s1 = ('At turn 2, google_sheets_add_row produced a google sheets row with cell "a" = "1". This task forbids the cell '
          '"a" shown in each of these google sheets rows.')
    s2 = ('At turn 2, google_sheets_add_row produced a google sheets row with cell "a" = "2". This task forbids the cell '
          '"a" shown in each of these google sheets rows.')
    assert C.at_turn_note(msg(s1, s2), 2) == (msg(s1, s2), {})  # a sentence twice inside the note


def test_acting_quiet_instructions_drop_reason_only():
    for q in (C.QUIET_R_P, C.QUIET_R_V5):
        assert q.startswith("Act on ") and "reason only" not in q and "Do not mention" in q
