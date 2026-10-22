import json

import pytest
from probe_focused_localized_spans import exact_positions, localized_terms


def test_only_explicit_localized_feedback_supplies_diagnostic_terms():
    text = json.dumps({'precise_attempt_feedback': [
        {'kind': 'failed_recipient_prohibition', 'proposal_call': 2, 'recipient_already_in_proposal': 'seen@x'},
        {'unknown_expected_recipient': 'hidden@x'}]})
    result = localized_terms(text)
    assert len(result) == 1 and result[0]['term'] == 'seen@x' and result[0]['call'] == 2
    assert result[0]['term'][result[0]['focus_start']:result[0]['focus_end']] == 'x'


class OriginalPieces:
    pieces = {1: '<tool_call>', 2: '</tool_call>', 10: 'reasoning bad@x',
              11: 'to="bad', 12: '@x"', 13: 'bad@x'}

    def decode(self, ids, **kwargs):
        return ''.join(self.pieces[i] for i in ids)

    def convert_tokens_to_ids(self, token):
        return 1 if token == '<tool_call>' else 2


def test_original_noncanonical_segments_and_exact_call_scope_are_retained():
    ids = [10, 1, 11, 12, 2, 1, 13, 2]
    result = exact_positions(ids, OriginalPieces(), [{'call': 1, 'term': 'bad@x', 'kind': 'test'}])
    assert result[0]['positions'] == [2, 3]  # Neither reasoning nor the second tool call.
    with pytest.raises(ValueError, match='Incomplete'):
        exact_positions(ids[:-1], OriginalPieces(), [])


def test_distinctive_field_part_excludes_shared_prefix_tokens():
    result = exact_positions([1, 11, 12, 2], OriginalPieces(),
                             [{'call': 1, 'term': 'bad@x', 'focus_start': 4, 'focus_end': 5}])
    assert result[0]['positions'] == [1, 2]
    assert result[1]['positions'] == [2]
    assert result[1]['focused_text'] == 'x'


def test_noncompositional_decoding_rejects_approximate_alignment():
    class Broken(OriginalPieces):
        def decode(self, ids, **kwargs):
            return 'different' if len(ids) > 1 else super().decode(ids, **kwargs)

    with pytest.raises(ValueError, match='noncompositional'):
        exact_positions([1, 13, 2], Broken(), [])
