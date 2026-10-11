from __future__ import annotations

import pytest
from posttrain.common import CatalogRef
from posttrain.train.catalog_schema import decode_training_selection
from posttrain.train.profiles import DPOSettings, TrainingLoop


def _decode(**values: object) -> DPOSettings:
    settings = decode_training_selection(
        CatalogRef("training", "tests/dpo"),
        {"selection_type": "dpo-settings", "id": "tests/dpo", "loop": {"max_steps": 1}, **values},
        {},
    )
    assert isinstance(settings, DPOSettings)
    return settings


def test_dpo_settings_default_to_plain_preference_loss() -> None:
    assert _decode().sft_weight == 0.0


def test_dpo_settings_carry_a_supervised_weight_on_the_chosen_side() -> None:
    assert _decode(sft_weight=0.2).sft_weight == 0.2


def test_dpo_settings_reject_a_negative_supervised_weight() -> None:
    with pytest.raises(ValueError, match="sft_weight"):
        DPOSettings("tests/dpo", TrainingLoop(max_steps=1), sft_weight=-0.1)
