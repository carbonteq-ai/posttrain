"""mask_turns.mask: the wrong turn leaves the trained set with its reason; existing masks stay; the final reply and
non-assistant messages can never be masked."""
import pytest

from mask_turns import REASON, mask


def row():
    msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "tool"}, {"role": "assistant"},
            {"role": "tool"}, {"role": "assistant"}, {"role": "tool"}, {"role": "assistant"}]
    return {"messages": msgs, "trainable_message_indices": [2, 6, 8],
            "metadata": {"masked_turns": [4], "masked_reasons": {"4": "failed call"}}}


def test_wrong_turn_is_masked_with_reason_and_old_masks_kept():
    r = mask(row(), [2])
    assert r["trainable_message_indices"] == [6, 8]
    assert r["metadata"]["masked_turns"] == [2, 4]
    assert r["metadata"]["masked_reasons"] == {"4": "failed call", "2": REASON}


def test_final_reply_and_tool_messages_cannot_be_masked():
    with pytest.raises(ValueError):
        mask(row(), [8])
    with pytest.raises(ValueError):
        mask(row(), [3])


def test_input_row_is_not_modified():
    r = row()
    mask(r, [6])
    assert r["trainable_message_indices"] == [2, 6, 8]
