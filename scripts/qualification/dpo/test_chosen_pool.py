"""chosen_pool.admit: passing rows are admitted; a harness-word failure passes only with exactly the cleared hits."""

from chosen_pool import admit


def report(rid, failed=(), hits=None):
    gates = {g: {"pass": g not in failed} for g in ("structure", "replay_score", "no_harness_words", "grounding",
                                                     "call_numbers_shown")}
    if hits is not None:
        gates["no_harness_words"]["hits"] = hits
    return {"id": rid, "gates": gates}


HITS = [{"message": 6, "words": ["brief"]}]


def test_all_gates_pass():
    assert admit(report("a"), {}) == (True, "every gate passes")


def test_advisory_gate_does_not_block():
    assert admit(report("a", failed=("call_numbers_shown",)), {})[0] is True


def test_cleared_harness_hits_pass():
    assert admit(report("a", failed=("no_harness_words",), hits=HITS), {"a": HITS})[0] is True


def test_uncleared_or_different_harness_hits_fail():
    assert admit(report("a", failed=("no_harness_words",), hits=HITS), {})[0] is False
    more = HITS + [{"message": 9, "words": ["checks"]}]
    assert admit(report("a", failed=("no_harness_words",), hits=more), {"a": HITS})[0] is False
    assert admit(report("b", failed=("no_harness_words",), hits=HITS), {"a": HITS})[0] is False


def test_clearance_never_excuses_another_gate():
    r = report("a", failed=("no_harness_words", "grounding"), hits=HITS)
    assert admit(r, {"a": HITS}) == (False, "gates failed: grounding, no_harness_words")


def test_later_layer_replaces_the_form_and_review_drops_apply():
    from chosen_pool import build

    def r(rid):
        return {"id": rid, "metadata": {"task": "t." + rid}}
    bad = report("a", failed=("reply_names_no_skipped_items",))
    layers = [("as recorded", {"a": r("a"), "b": r("b")}, {"a": bad, "b": report("b")}),
              ("final reply delete-only edits", {"a": r("a")}, {"a": report("a")})]
    chosen, manifest = build(layers, [{"drop": {"b": "invented value"}}], [{"cleared": {}}])
    assert [x["id"] for x in chosen] == ["a"]
    assert manifest["a"]["form"] == "final reply delete-only edits" and manifest["a"]["admitted"]
    assert manifest["b"]["reason"] == "dropped on reading: invented value"


def test_undrop_lifts_an_earlier_drop_only():
    import pytest
    from chosen_pool import merge_reviews
    m = merge_reviews([{"drop": {"a": "x", "b": "y"}}, {"undrop": {"a": "masked now"}, "drop": {"c": "z"}}])
    assert m["drop"] == {"b": "y", "c": "z"}
    with pytest.raises(ValueError):
        merge_reviews([{"drop": {}}, {"undrop": {"a": "never dropped"}}])
