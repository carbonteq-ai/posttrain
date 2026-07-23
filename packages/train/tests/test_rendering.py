"""Tests for trainer-neutral sample rendering."""

from __future__ import annotations

import pytest
from posttrain.common.variants import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.data import PreferenceDataset, PreferenceExample, SupervisedDataset, SupervisedExample
from posttrain.train import (
    LFM25_RENDERER,
    QWEN35_RENDERER,
    render_preferences,
    render_supervised,
)

transformers = pytest.importorskip("transformers")
pytest.importorskip("renderers")


def _load_tokenizer(model):
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model.base.repo_id,
            revision=model.base.revision,
            local_files_only=True,
        )
    except OSError:
        pytest.skip(f"tokenizer for {model.base.repo_id}@{model.base.revision} is not cached")


@pytest.mark.parametrize(
    ("model", "profile"),
    (
        (QWEN_35_2B, QWEN35_RENDERER),
        (LFM_25_12B_THINKING, LFM25_RENDERER),
    ),
)
def test_renderer_builds_nonempty_assistant_only_sft_masks(model, profile) -> None:
    tokenizer = _load_tokenizer(model)
    dataset = SupervisedDataset(
        "gsm8k-sft-golden-v1",
        "a" * 40,
        (
            SupervisedExample(
                "gsm8k/train/0",
                (
                    {"role": "user", "content": "Solve 2 + 2 and end with #### N."},
                    {"role": "assistant", "content": "Two plus two is four.\n#### 4"},
                ),
                (1,),
            ),
        ),
    )
    sample = render_supervised(tokenizer, model, dataset, profile, max_length=512)[0]
    assert len(sample.input_ids) == len(sample.labels)
    assert any(label == -100 for label in sample.labels)
    assert any(label != -100 for label in sample.labels)
    decoded = tokenizer.decode(sample.input_ids, skip_special_tokens=False)
    assert "Solve 2 + 2" in decoded
    assert "#### 4" in decoded


@pytest.mark.parametrize(
    ("model", "profile"),
    (
        (QWEN_35_2B, QWEN35_RENDERER),
        (LFM_25_12B_THINKING, LFM25_RENDERER),
    ),
)
def test_renderer_produces_equal_dpo_prompt_prefixes(model, profile) -> None:
    tokenizer = _load_tokenizer(model)
    dataset = PreferenceDataset(
        "gsm8k-dpo-golden-v1",
        "a" * 40,
        (
            PreferenceExample(
                "gsm8k/train/0",
                ({"role": "user", "content": "Solve 2 + 2 and end with #### N."},),
                ({"role": "assistant", "content": "Two plus two is four.\n#### 4"},),
                ({"role": "assistant", "content": "Two plus two is five.\n#### 5"},),
                1.0,
                0.0,
            ),
        ),
    )
    sample = render_preferences(tokenizer, model, dataset, profile, max_length=512)[0]
    assert sample.prompt_ids
    assert sample.chosen_ids
    assert sample.rejected_ids
    assert sample.chosen_ids != sample.rejected_ids
