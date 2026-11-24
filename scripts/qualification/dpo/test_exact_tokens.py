"""exact_tokens.py helpers: chain sides, trained-span placement (both ways), text identity."""
from exact_tokens import chain_parts, side_from_chain, text_equal, trained_text_inside

decode = lambda ids: "".join(chr(t) for t in ids)
enc = lambda text: [ord(c) for c in text]

CHAIN = {"segments": [
    {"kind": "context", "ids": enc("<sys>task<a><t>")},
    {"kind": "action", "ids": enc("plan;call1$")},
    {"kind": "context", "ids": enc("\n[tool1]<a><t>")},
    {"kind": "action", "ids": enc("oops;call2$")},
    {"kind": "context", "ids": enc("\n[tool2]<a><t>")},
    {"kind": "action", "ids": enc("reply$")},
]}


def test_chain_side_trains_actions_only_and_skips_untrained_turns():
    prompt, segments = chain_parts(CHAIN)
    assert decode(prompt) == "<sys>task<a><t>"
    ids, mask = side_from_chain(segments, untrained={1})
    assert decode(ids) == "plan;call1$\n[tool1]<a><t>oops;call2$\n[tool2]<a><t>reply$"
    trained = "".join(c for c, m in zip(decode(ids), mask) if m)
    assert trained == "plan;call1$reply$"  # turn 1 (oops) untrained, tool results and the opener <t> never


def test_trained_spans_must_sit_inside_rendered_assistant_text():
    _, segments = chain_parts(CHAIN)
    ids, mask = side_from_chain(segments, untrained=set())
    # the rendering's assistant spans include the opener <t> of later turns and a final newline outside
    rendered_text = "plan;call1$\n[tool1]<a><t>oops;call2$\n[tool2]<a><t>reply$\n"
    rendered_mask = [0] * len(rendered_text)
    for part in ("plan;call1$", "<t>oops;call2$", "<t>reply$"):
        start = rendered_text.index(part)
        rendered_mask[start:start + len(part)] = [1] * len(part)
    assert trained_text_inside(decode, ids, mask, enc(rendered_text), rendered_mask)
    wrong = [1] * len(mask)  # tool results trained too
    assert not trained_text_inside(decode, ids, wrong, enc(rendered_text), rendered_mask)


def test_text_equal_allows_only_the_renderers_final_newline():
    assert text_equal(decode, enc("abc$"), enc("abc$\n"))
    assert text_equal(decode, enc("abc$"), enc("abc$"))
    assert not text_equal(decode, enc("abc$\n"), enc("abc$"))
    assert not text_equal(decode, enc("abd$"), enc("abc$\n"))
