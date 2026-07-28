"""Focused tests for pinned Transformers architecture and Gemma LoRA loading."""

from types import SimpleNamespace

import pytest
from posttrain.common.variants import GEMMA4_RENDERER_CONTRACT, GEMMA_4_31B_IT, GEMMA_4_E4B_IT
from posttrain.train import GEMMA4_RENDERER, LoRAUpdate, QLoRAUpdate
from posttrain.train.backends.trl.common import (
    _validate_gemma4_lora_targets,
    load_frozen_model,
    resolve_model_factory,
)
from posttrain.train.backends.trl.distillation import _validate_gemma4_distillation_topology


class FakeAutoConfig:
    architectures = ["Gemma4ForConditionalGeneration"]
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        cls.calls.append((repo_id, kwargs))
        return cls()


class FakeLoadedModel:
    def __init__(self) -> None:
        self.requires_grad_value = None
        self.eval_called = False

    def requires_grad_(self, value):
        self.requires_grad_value = value
        return self

    def eval(self):
        self.eval_called = True
        return self


class FakeConditionalFactory:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        cls.calls.append((repo_id, kwargs))
        return FakeLoadedModel()


class FakeCausalFactory(FakeConditionalFactory):
    calls = []


def _imports(*, conditional=True):
    transformers = (
        SimpleNamespace(Gemma4ForConditionalGeneration=FakeConditionalFactory)
        if conditional
        else SimpleNamespace()
    )
    return {
        "AutoConfig": FakeAutoConfig,
        "AutoModelForCausalLM": FakeCausalFactory,
        "transformers": transformers,
        "torch": SimpleNamespace(bfloat16="bf16"),
    }


def test_declared_conditional_generation_factory_wins_without_remote_code() -> None:
    FakeAutoConfig.calls.clear()

    factory, config = resolve_model_factory(GEMMA_4_E4B_IT, _imports())

    assert factory is FakeConditionalFactory
    assert isinstance(config, FakeAutoConfig)
    assert FakeAutoConfig.calls == [
        (
            "google/gemma-4-E4B-it",
            {
                "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
                "trust_remote_code": False,
            },
        )
    ]


def test_gemma_training_renderer_matches_the_common_renderer_contract() -> None:
    assert GEMMA4_RENDERER.id == GEMMA4_RENDERER_CONTRACT.id
    assert GEMMA4_RENDERER.model_family == GEMMA4_RENDERER_CONTRACT.model_family
    assert GEMMA4_RENDERER.implementation == "default"
    assert GEMMA4_RENDERER.reasoning_mode == "native"


def test_missing_local_conditional_generation_factory_falls_back_to_causal_auto() -> None:
    factory, _ = resolve_model_factory(GEMMA_4_E4B_IT, _imports(conditional=False))

    assert factory is FakeCausalFactory


def test_frozen_teacher_uses_resolved_factory_and_unquantized_bf16() -> None:
    FakeConditionalFactory.calls.clear()

    loaded = load_frozen_model(GEMMA_4_E4B_IT, _imports())

    assert loaded.requires_grad_value is False
    assert loaded.eval_called is True
    _, options = FakeConditionalFactory.calls[-1]
    assert options["dtype"] == "bf16"
    assert options["device_map"] == {"": 0}
    assert options["trust_remote_code"] is False
    assert "quantization_config" not in options


class FakeGemmaModel:
    def __init__(self, names):
        self._names = names

    def named_modules(self):
        return [(name, object()) for name in self._names]


GEMMA_TARGETS = (
    r"model[.]language_model[.]layers[.][0-9]+[.].*"
    r"[.](q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def test_gemma_lora_targets_cover_text_projections_and_exclude_multimodal_towers() -> None:
    text_names = [
        f"model.language_model.layers.0.block.{suffix}"
        for suffix in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    ]
    multimodal_names = ["model.vision_tower.layers.0.q_proj", "model.audio_tower.layers.0.o_proj"]

    _validate_gemma4_lora_targets(
        FakeGemmaModel([*text_names, *multimodal_names]),
        GEMMA_4_E4B_IT,
        LoRAUpdate(target_modules=GEMMA_TARGETS),
    )


def test_gemma_lora_rejects_any_multimodal_match() -> None:
    broad = r"model[.].*[.](q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    names = [
        *[
            f"model.language_model.layers.0.block.{suffix}"
            for suffix in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        ],
        "model.vision_tower.layers.0.q_proj",
    ]

    with pytest.raises(ValueError, match="non-text modules"):
        _validate_gemma4_lora_targets(
            FakeGemmaModel(names),
            GEMMA_4_E4B_IT,
            LoRAUpdate(target_modules=broad),
        )


def test_gemma_distillation_topology_requires_unquantized_bf16_lora() -> None:
    request = SimpleNamespace(
        student=GEMMA_4_E4B_IT,
        teacher=GEMMA_4_31B_IT,
        training=SimpleNamespace(update=LoRAUpdate()),
        quantization=None,
    )

    _validate_gemma4_distillation_topology(request)  # type: ignore[arg-type]

    request.training.update = QLoRAUpdate()
    with pytest.raises(ValueError, match="BF16 LoRA"):
        _validate_gemma4_distillation_topology(request)  # type: ignore[arg-type]
