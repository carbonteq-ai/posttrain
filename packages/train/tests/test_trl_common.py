"""Focused tests for family-aware TRL model loading."""

from posttrain.common.variants import GEMMA_4_12B_IT, LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.train.backends.trl.common import trainable_model_factory


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
