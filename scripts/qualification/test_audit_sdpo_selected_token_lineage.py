"""Normalization must preserve proven reasoning strings without guessing content."""

import pytest
from audit_sdpo_selected_token_lineage import restore_reasoning_alias


def test_alias_restoration_is_nonmutating_and_rejects_conflicting_history():
    original = [{'role': 'assistant', 'reasoning': 'original thought', 'content': 'visible'}]
    restored, count = restore_reasoning_alias(original)
    assert count == 1 and restored[0]['reasoning_content'] == 'original thought'
    assert original[0]['reasoning'] == 'original thought' and 'reasoning_content' not in original[0]
    with pytest.raises(ValueError, match='conflicting'):
        restore_reasoning_alias([original[0] | {'reasoning_content': 'different thought'}])
    with pytest.raises(ValueError, match='non-string'):
        restore_reasoning_alias([original[0] | {'reasoning': [{'text': 'unqualified projection'}]}])
