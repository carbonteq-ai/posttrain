"""Focused tests for family-aware TRL model loading."""

from types import SimpleNamespace

import pytest
from posttrain.common.variants import GEMMA_4_12B_IT, GEMMA_4_E4B_IT, LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.train import LoRAUpdate
from posttrain.train.backends.trl.common import (
    _disable_model_cache,
    _validate_gemma4_lora_targets,
    trainable_model_factory,
)


class CausalFactory:
    pass


class MultimodalFactory:
    pass


IMPORTS = {
    "AutoModelForCausalLM": CausalFactory,
    "AutoModelForMultimodalLM": MultimodalFactory,
}


def test_trainable_model_factory_uses_multimodal_loader_for_gemma4() -> None:
    assert trainable_model_factory(QWEN_35_2B, IMPORTS) is CausalFactory
    assert trainable_model_factory(LFM_25_12B_THINKING, IMPORTS) is CausalFactory
    assert trainable_model_factory(GEMMA_4_12B_IT, IMPORTS) is MultimodalFactory
    assert trainable_model_factory(GEMMA_4_E4B_IT, IMPORTS) is MultimodalFactory


def test_disable_model_cache_updates_composite_and_nested_text_configs() -> None:
    text_config = SimpleNamespace(use_cache=True)
    config = SimpleNamespace(use_cache=True, get_text_config=lambda: text_config)

    _disable_model_cache(SimpleNamespace(config=config))

    assert config.use_cache is False
    assert text_config.use_cache is False


class FakeGemmaModel:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def named_modules(self):
        return ((name, object()) for name in self.names)


GEMMA4_TEXT_TARGETS = (
    r"^model[.]language_model[.]layers[.]\d+[.]"
    r"(self_attn[.](q_proj|k_proj|v_proj|o_proj)|mlp[.](gate_proj|up_proj|down_proj))$"
)
GEMMA4_TEXT_MODULES = [
    *[
        f"model.language_model.layers.0.self_attn.{projection}"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    ],
    *[f"model.language_model.layers.0.mlp.{projection}" for projection in ("gate_proj", "up_proj", "down_proj")],
]


def test_gemma_lora_targets_cover_language_projections_without_matching_multimodal_towers() -> None:
    model = FakeGemmaModel(
        [
            *GEMMA4_TEXT_MODULES,
            "model.vision_tower.layers.0.self_attn.q_proj",
            "model.audio_tower.layers.0.self_attn.o_proj",
        ]
    )

    _validate_gemma4_lora_targets(model, GEMMA_4_E4B_IT, LoRAUpdate(target_modules=GEMMA4_TEXT_TARGETS))


@pytest.mark.parametrize(
    ("target_modules", "names", "message"),
    (
        ("all-linear", GEMMA4_TEXT_MODULES, "explicitly target"),
        (GEMMA4_TEXT_TARGETS, [], "matched no modules"),
        (
            r"^model[.].*[.](q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            [*GEMMA4_TEXT_MODULES, "model.vision_tower.layers.0.q_proj"],
            "non-text or unsupported modules",
        ),
        (
            r"^model[.]language_model[.]layers[.]\d+[.].*$",
            [*GEMMA4_TEXT_MODULES, "model.language_model.layers.0.input_layernorm"],
            "non-text or unsupported modules",
        ),
        (
            r"^model[.]language_model[.]layers[.]\d+[.]self_attn[.](q_proj|k_proj|v_proj|o_proj)$",
            GEMMA4_TEXT_MODULES,
            "missed required text projections",
        ),
    ),
)
def test_gemma_lora_target_validation_rejects_unsafe_or_incomplete_selections(
    target_modules: str,
    names: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_gemma4_lora_targets(
            FakeGemmaModel(names),
            GEMMA_4_E4B_IT,
            LoRAUpdate(target_modules=target_modules),
        )
