import json
from pathlib import Path

import pytest
import torch
from self_reserve_config import (
    MODEL_REVISION,
    REFERENCE_ADAPTER_SHA,
    candidates,
    check_adapter_config,
    population,
    verify_loaded_state,
    zero_effect_state,
)


def selection():
    return {"model_revision": MODEL_REVISION, "primary": "candidate", "primary_qualified": False,
            "decision_basis": "local repair evidence only", "candidates": [
                {"name": "fresh_base", "kind": "zero_effect_base"},
                {"name": "checkpoint64", "kind": "adapter", "path": "/source/checkpoint-64", "adapter_sha256": REFERENCE_ADAPTER_SHA},
                {"name": "candidate", "kind": "adapter", "path": "/candidate/checkpoint-10", "adapter_sha256": "a" * 64,
                 "receipt_path": "/candidate/receipt.json", "step": 10}]}


def test_candidate_controls_cannot_be_renamed_or_unpinned():
    assert len(candidates(selection())) == 3
    value = selection()
    value["candidates"][0]["kind"] = "adapter"
    with pytest.raises(ValueError, match="fresh base"):
        candidates(value)
    value = selection()
    value["candidates"][1]["adapter_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="retained reference"):
        candidates(value)


def test_zero_effect_adapter_preserves_source_and_matches_bare_linear_output():
    state = {"module.lora_A.weight": torch.tensor([[1., 2., 3.]]), "module.lora_B.weight": torch.tensor([[2.], [4.]])}
    zero = zero_effect_state(state)
    x, weight = torch.ones(2, 3), torch.ones(2, 3)
    bare = x @ weight.T
    actual = bare + (x @ zero["module.lora_A.weight"].T) @ zero["module.lora_B.weight"].T
    torch.testing.assert_close(actual, bare, atol=0, rtol=0)
    assert torch.count_nonzero(state["module.lora_B.weight"]) == 2


def test_extra_saved_weights_or_nonfinite_adapter_rejected():
    state = {"module.lora_A.weight": torch.ones(1), "module.lora_B.weight": torch.ones(1)}
    with pytest.raises(ValueError, match="non-LoRA"):
        zero_effect_state({**state, "lm_head.weight": torch.ones(1)})
    state["module.lora_A.weight"][0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        zero_effect_state(state)
    with pytest.raises(ValueError, match="configuration"):
        check_adapter_config({"bias": "all", "r": 16, "lora_alpha": 32, "lora_dropout": 0})


def test_predeclared_population_is_exact():
    root = Path(__file__).resolve().parents[2]
    reserve, protocol, pool, heldout = [json.loads((root / p).read_text()) for p in [
        "docs/research/evidence/self-vllm-20261017/adaptive-transfer-reserve24.json",
        "docs/research/evidence/self-vllm-20261017/task-protocol.json",
        "scripts/qualification/fixtures/automationbench_train_pool_luna07_v1.json",
        "scripts/qualification/fixtures/automationbench_heldout62_v1.json"]]
    assert len(population(reserve, protocol, pool, heldout)) == 24
    reserve["task_names"].reverse()
    with pytest.raises(ValueError, match="identity"):
        population(reserve, protocol, pool, heldout)


def test_loaded_state_must_match_every_selected_tensor():
    state = {"a": torch.ones(2)}
    verify_loaded_state(state, {"a": torch.ones(2)})
    for loaded in [{}, {"a": torch.zeros(2)}, {"a": torch.ones(2), "extra": torch.ones(1)}]:
        with pytest.raises(ValueError, match="loaded adapter"):
            verify_loaded_state(state, loaded)
