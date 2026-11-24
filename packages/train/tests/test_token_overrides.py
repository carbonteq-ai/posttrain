"""Token overrides replace rendered preference ids only when they are the same text and train only assistant text."""

from __future__ import annotations

import pytest
from posttrain.data import PreferenceDataset, PreferenceExample
from posttrain.train import RenderedPreferenceExample, apply_token_overrides


class CharTokenizer:
    """One token per character (ord), enough to test text identity and span placement."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(token) for token in ids)


def ids(text: str) -> tuple[int, ...]:
    return tuple(ord(char) for char in text)


PROMPT = "<sys>task<a>"
# rendered sides: assistant text (mask 1) then a tool result (mask 0) then assistant text, plus a final "\n" (mask 0)
CHOSEN_PARTS = (("good call;", 1), ("[tool ok]", 0), ("done.$", 1), ("\n", 0))
REJECTED_PARTS = (("bad call;", 1), ("[tool err]", 0), ("oops.$", 1), ("\n", 0))


def rendered_side(parts):
    return ids("".join(text for text, _ in parts)), tuple(m for text, m in parts for _ in text)


def rendered_sample() -> RenderedPreferenceExample:
    chosen, chosen_mask = rendered_side(CHOSEN_PARTS)
    rejected, rejected_mask = rendered_side(REJECTED_PARTS)
    return RenderedPreferenceExample("pairs/one", ids(PROMPT), chosen, rejected, chosen_mask, rejected_mask)


def override(**changes):
    """The sampled form: same text, no trailing newline, the tool result untrained."""

    chosen, chosen_mask = rendered_side(CHOSEN_PARTS[:3])
    rejected, rejected_mask = rendered_side(REJECTED_PARTS[:3])
    value = {
        "prompt": list(ids(PROMPT)),
        "chosen": list(chosen),
        "chosen_mask": list(chosen_mask),
        "rejected": list(rejected),
        "rejected_mask": list(rejected_mask),
    }
    value.update(changes)
    return value


def dataset(*token_ids) -> PreferenceDataset:
    examples = []
    for index, value in enumerate(token_ids):
        examples.append(
            PreferenceExample(
                id="pairs/one" if index == 0 else f"pairs/n{index}",
                prompt=({"role": "user", "content": "task"},),
                chosen=({"role": "assistant", "content": "good"},),
                rejected=({"role": "assistant", "content": "bad"},),
                metadata={} if value is None else {"token_ids": value},
            )
        )
    return PreferenceDataset("tests/pairs", "1", tuple(examples))


def test_override_replaces_ids_and_masks_when_text_matches():
    (got,) = apply_token_overrides(CharTokenizer(), (rendered_sample(),), dataset(override()), max_length=1000)
    want = override()
    assert list(got.prompt_ids) == want["prompt"]
    assert list(got.chosen_ids) == want["chosen"] and list(got.chosen_mask) == want["chosen_mask"]
    assert list(got.rejected_ids) == want["rejected"] and list(got.rejected_mask) == want["rejected_mask"]


def test_examples_without_overrides_keep_the_rendering():
    sample = rendered_sample()
    assert apply_token_overrides(CharTokenizer(), (sample,), dataset(None), max_length=1000) == (sample,)


def test_override_with_different_text_is_refused():
    changed = list(override()["chosen"])
    changed[0] = ord("G")
    with pytest.raises(ValueError, match="differs from the rendered text"):
        apply_token_overrides(CharTokenizer(), (rendered_sample(),), dataset(override(chosen=changed)), max_length=1000)


def test_override_that_trains_a_tool_result_is_refused():
    mask = [1] * len(override()["chosen_mask"])  # loss on "[tool ok]" too
    with pytest.raises(ValueError, match="trains non-assistant text"):
        apply_token_overrides(
            CharTokenizer(), (rendered_sample(),), dataset(override(chosen_mask=mask)), max_length=1000
        )


def test_override_with_bad_mask_or_prompt_is_refused():
    with pytest.raises(ValueError, match="invalid rejected override mask"):
        apply_token_overrides(
            CharTokenizer(), (rendered_sample(),), dataset(override(rejected_mask=[0, 1])), max_length=1000
        )
    with pytest.raises(ValueError, match="override prompt differs"):
        apply_token_overrides(
            CharTokenizer(), (rendered_sample(),), dataset(override(prompt=list(ids("<sys>other<a>")))), max_length=1000
        )


def test_overrides_must_cover_every_example_and_fit():
    sample = rendered_sample()
    other = RenderedPreferenceExample(
        "pairs/n1", sample.prompt_ids, sample.chosen_ids, sample.rejected_ids, sample.chosen_mask, sample.rejected_mask
    )
    with pytest.raises(ValueError, match="every preference example or none"):
        apply_token_overrides(CharTokenizer(), (sample, other), dataset(override(), None), max_length=1000)
    with pytest.raises(ValueError, match="exceeds max_length"):
        apply_token_overrides(CharTokenizer(), (sample,), dataset(override()), max_length=20)
