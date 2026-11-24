"""H (round 7 candidate, 21 Nov): the hidden-value phrase said as what "no tool has shown". CPU tests, both ways."""
import pytest

import feedback_c as C


def test_the_swap_is_exact_and_reversible():
    text = ('Still needed: a gmail message to "a@b.example.com", whose body needs a value you have not read (still '
            'missing).\nStill needed: a gmail message, whose recipient is one you have not read.\nSome details are '
            'values you have not read.')
    out = C.say_unshown(text)
    assert out.count(C.UNSHOWN_NEW) == 3 and C.UNSHOWN_OLD not in out
    assert out.replace(C.UNSHOWN_NEW, C.UNSHOWN_OLD) == text
    # same word count and the same one negation per phrase
    assert len(out.split()) == len(text.split())
    assert len(C.NEG.findall(out)) == len(C.NEG.findall(text))


def test_quoted_values_and_text_without_the_phrase_are_untouched():
    text = 'At turn 4, gmail_send_email produced a gmail message with body containing "Docker".'
    assert C.say_unshown(text) == text
    assert C.say_unshown(None) is None


def test_the_swap_refuses_when_it_could_not_be_reversed_or_would_change_a_value():
    with pytest.raises(ValueError):
        C.say_unshown("a value no tool has shown and a value you have not read")
    with pytest.raises(ValueError):
        C.say_unshown('a message with subject "what you have not read" and a value you have not read')


def test_the_h_design_is_registered_on_its_base():
    assert C.DESIGNS_H == ("S1PUH",)
    assert all(d[:-1] in C.DESIGNS_U for d in C.DESIGNS_H)
