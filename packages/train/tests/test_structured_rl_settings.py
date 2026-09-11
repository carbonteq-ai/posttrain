"""Catalog and typed settings share one validated numerical contract."""

import pytest
from posttrain.common import CatalogRef
from posttrain.train.catalog_schema import decode_training_selection
from posttrain.train.profiles import CAPOSettings, GDPOSettings, TrainingLoop
from posttrain.train.reward_projection import RewardProjection


def test_reward_projection_is_a_catalog_selection():
    projection = decode_training_selection(
        CatalogRef("training", "reward"),
        {
            "selection_type": "reward-projection",
            "id": "reward",
            "revision": "1",
            "components": [{"name": "outcome", "source": "scalar"}],
            "process_info_key": "resolved_credit",
        },
        {},
    )
    assert isinstance(projection, RewardProjection)
    assert projection.components[0].name == "outcome"
    assert projection.process_info_key == "resolved_credit"


def test_structured_settings_allow_rank_sharded_logical_batch():
    settings = CAPOSettings(id="test", loop=TrainingLoop(max_steps=1, per_device_batch_size=1))
    assert settings.num_prompts_per_step * settings.num_generations == 2


@pytest.mark.parametrize(
    "kind,extra,expected",
    [
        ("gdpo", {"component_names": ["outcome", "quality"], "component_weights": [1.0, 2.0]}, GDPOSettings),
        ("capo", {}, CAPOSettings),
    ],
)
def test_catalog_resolves_structured_algorithm(kind, extra, expected):
    settings = decode_training_selection(
        CatalogRef("training", "test"),
        {
            "selection_type": f"{kind}-settings",
            "id": "test",
            "revision": "1",
            "loop": {"max_steps": 2, "gradient_accumulation_steps": 2},
            **extra,
        },
        {},
    )
    assert isinstance(settings, expected)
    assert isinstance(settings, GDPOSettings | CAPOSettings)
    assert settings.num_generations == 2
    assert settings.max_admission_attempts == 3


@pytest.mark.parametrize(
    "extra",
    [
        {"beta": float("nan")},
        {"num_generations": True},
        {"clip_epsilon_low": 1.1},
        {"epsilon": 0},
        {"outcome_weight": 1, "process_weight": 2},
    ],
)
def test_invalid_capo_settings_fail_before_launch(extra):
    with pytest.raises(ValueError):
        CAPOSettings(id="test", loop=TrainingLoop(max_steps=2, gradient_accumulation_steps=2), **extra)


@pytest.mark.parametrize("names,weights", [(("a", "a"), (1, 2)), (("a",), (0,)), (("a",), (float("inf"),))])
def test_invalid_gdpo_components_fail_before_launch(names, weights):
    with pytest.raises(ValueError):
        GDPOSettings(
            id="test",
            loop=TrainingLoop(max_steps=2, gradient_accumulation_steps=2),
            component_names=names,
            component_weights=weights,
        )
