"""pair_parity: the intended mask drops untrained turns and the forced thinking opener; the span check catches a
tool-result or untrained token in the mask; deletion_matches accepts only the logged deletions."""
from pair_parity import deletion_matches, intended_mask, mask_ok, spans

OPEN = [7, 8]  # "<think>", "\n"
#            turn 0 (no opener)   tool    turn 1 (opener)      tool    turn 2 (opener)
IDS = [1, 2, 3, 9, 9, 7, 8, 4, 5, 9, 9, 7, 8, 6]
MASK = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1]


def test_spans():
    assert spans(MASK) == [(0, 3), (5, 9), (11, 14)]


def test_opener_dropped_after_first_turn_and_untrained_turn_zeroed():
    m = intended_mask(MASK, {1}, IDS, OPEN)
    assert m == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    m2 = intended_mask(MASK, set(), IDS, OPEN)
    assert m2 == [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]


def test_mask_check_both_ways():
    good = intended_mask(MASK, set(), IDS, OPEN)
    allowed = [(0, 3), (7, 9), (13, 14)]
    assert mask_ok(good, allowed)
    with_tool = list(good)
    with_tool[3] = 1  # a tool-result token trained
    assert not mask_ok(with_tool, allowed)
    with_untrained = intended_mask(MASK, set(), IDS, OPEN)  # turn 1 should have been untrained
    assert not mask_ok(with_untrained, [(0, 3), (13, 14)])


def test_deletion_matches_only_logged_deletions():
    before = "Done.\n- Sent to Alice\n- Excluded: Bob (hold)\nAll logged."
    after = "Done.\n- Sent to Alice\nAll logged."
    assert deletion_matches(before, after, ["- Excluded: Bob (hold)"])
    assert not deletion_matches(before, after, ["- Excluded: Bob"])
    assert not deletion_matches(before, "Done.\n- Sent to Alicia\nAll logged.", ["- Excluded: Bob (hold)"])


def test_deletion_order_does_not_matter():
    before = "A\n- x: excluded\n- y: excluded\nB"
    after = "A\nB"
    assert deletion_matches(before, after, ["- y: excluded", "- x: excluded"])
    assert not deletion_matches(before, after, ["- y: excluded"])


def test_json_literals_python_spelling():
    from pair_parity import python_literals
    text = "<parameter=file_types>\nnull\n</parameter>\n<parameter=flag>\ntrue\n</parameter>\n<parameter=q>\nnull value\n</parameter>"
    out, n = python_literals(text)
    assert n == 2 and "None\n</parameter>" in out and "True\n</parameter>" in out and "null value" in out


def test_deletion_applies_to_the_reply_not_the_reasoning():
    before = "<think>\n- x: excluded is the rule\n</think>\n\nDone.\n- x: excluded\nB"
    after = "<think>\n- x: excluded is the rule\n</think>\n\nDone.\nB"
    assert deletion_matches(before, after, ["- x: excluded"])
