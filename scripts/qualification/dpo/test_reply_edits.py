"""reply_edits.edit_reply: whole lines and exact fragments are deleted and logged; anything else is refused."""
import pytest

from reply_edits import edit_reply

REPLY = "Done.\n\n**Sent:**\n- Alice (a@x.com)\n- Bob (b@x.com)\n\n**Excluded:**\n- Carol (on hold)\n\nAll logged; skipped Carol."


def test_lines_and_fragments_are_deleted_and_logged():
    new, deleted = edit_reply(REPLY, {"lines": [6, 7], "fragments": [[9, "; skipped Carol"]]})
    assert new == "Done.\n\n**Sent:**\n- Alice (a@x.com)\n- Bob (b@x.com)\n\nAll logged."
    assert sorted(deleted) == sorted(["**Excluded:**", "- Carol (on hold)", "; skipped Carol"])


def test_fragment_must_occur_exactly_once():
    with pytest.raises(ValueError, match="occurs 0 times"):
        edit_reply(REPLY, {"fragments": [[3, "Dave"]]})
    with pytest.raises(ValueError, match="occurs 2 times"):
        edit_reply("a, b, a, b", {"fragments": [[0, "a"]]})


def test_result_must_not_be_empty():
    with pytest.raises(ValueError, match="empties"):
        edit_reply("Skipped everything.", {"lines": [0]})


def test_only_deletions_can_pass():
    from reply_edits import delete_only
    assert delete_only("Sent to Alice; skipped Bob.", "Sent to Alice.")
    assert not delete_only("Sent to Alice.", "Sent to Alicia.")
    assert not delete_only("Sent to Alice.", "Sent to Alice and Bob.")
