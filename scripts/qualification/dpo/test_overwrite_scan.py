"""overwrite_scan.overwrites: a trained second write of the same record with other values flags; the same values,
another record, a read, or a correction of an already masked wrong turn do not."""
from overwrite_scan import overwrites


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


def row(turns, trained):
    msgs = [{"role": "system"}, {"role": "user"}]
    for calls in turns:
        msgs += [{"role": "assistant", "tool_calls": calls}, {"role": "tool"}]
    msgs.append({"role": "assistant"})
    return {"messages": msgs, "trainable_message_indices": trained}


UPD = "google_sheets_update_row"


def test_second_write_with_other_values_flags():
    r = row([[call(UPD, spreadsheet="s", worksheet="w", row=2, cells="APPROVED")],
             [call(UPD, spreadsheet="s", worksheet="w", row=2, cells="Approved for Payment")]], [2, 4, 6])
    assert len(overwrites(r)) == 1


def test_same_values_other_record_and_reads_do_not_flag():
    r = row([[call(UPD, spreadsheet="s", worksheet="w", row=2, cells="A")],
             [call(UPD, spreadsheet="s", worksheet="w", row=2, cells="A"),
              call(UPD, spreadsheet="s", worksheet="w", row=3, cells="B"),
              call("google_sheets_get_many_rows", spreadsheet="s", worksheet="w")]], [2, 4, 6])
    assert overwrites(r) == []


def test_correction_of_a_masked_wrong_turn_is_flagged_only_if_trained():
    turns = [[call(UPD, spreadsheet="s", worksheet="w", row=2, cells="APPROVED")],
             [call(UPD, spreadsheet="s", worksheet="w", row=2, cells="Approved for Payment")]]
    assert overwrites(row(turns, [4, 6])) != []   # wrong turn 2 masked: its correction (turn 4) is still a change
    assert overwrites(row(turns, [2, 6])) == []   # the later turn is untrained: nothing trained overwrites


def test_other_fields_other_cards_and_same_value_formats_do_not_flag():
    r = row([[call(UPD, spreadsheet="s", worksheet="w", row=4, cells='{"Result": "Conditional"}'),
              call("trello_card_update", board="b", card="c1", due="2026-02-22")],
             [call(UPD, spreadsheet="s", worksheet="w", row=4, cells='{"Status": "Legal Notified"}'),
              call("trello_card_update", board="b", card="c2", due="2026-02-25"),
              call("hubspot_update_contact", contact_id="x", payment_retry_count="2")],
             [call("hubspot_update_contact", contact_id="x", additional_properties_json={"payment_retry_count": 2})]],
            [2, 4, 6, 8])
    assert overwrites(r) == []
