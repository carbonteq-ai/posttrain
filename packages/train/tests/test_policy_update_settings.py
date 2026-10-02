"""Explicit catalogs decouple collection and updates without activating trainers."""

from dataclasses import asdict, replace

import pytest
from posttrain.common import CatalogRef
from posttrain.train import ActionSelection, PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings
from posttrain.train.catalog_schema import decode_training_selection
from posttrain.train.profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings, TrainingLoop

from .test_api import FakeEnvironment, FakeRLBridge, _inference, _training


def update_settings() -> PolicyUpdateSettings:
    return PolicyUpdateSettings(PolicyUpdateSchedule("episode", 2, epochs=2, max_applied_updates=3),
                                PolicyExecutionBudget(2, 1024, 16384))


@pytest.mark.parametrize("algorithm", ["grpo", "sampo", "gdpo", "capo"])
def test_catalog_roundtrip_preserves_explicit_settings(algorithm) -> None:
    updates = update_settings()
    payload = {
        "selection_type": f"{algorithm}-settings", "id": "explicit-updates", "revision": "1",
        "loop": {"max_steps": 3}, "num_prompts_per_step": 4, "num_generations": 2,
        "policy_updates": asdict(updates),
    }
    if algorithm == "gdpo":
        payload.update(component_names=["outcome"], component_weights=[1.0])
    decoded = decode_training_selection(CatalogRef("training", "explicit-updates"), payload, {})
    assert isinstance(decoded, GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings)
    assert decoded.policy_updates == updates
    assert decoded.loop.per_device_batch_size * decoded.loop.gradient_accumulation_steps == 1
    assert decoded.num_prompts_per_step * decoded.num_generations == 8
    dumped = asdict(decoded)
    assert decode_training_selection(CatalogRef("training", "explicit-updates"),
                                     {"selection_type": f"{algorithm}-settings", **dumped}, {}) == decoded


def test_legacy_catalog_retains_batch_equality_without_explicit_settings() -> None:
    loop = TrainingLoop(max_steps=3, per_device_batch_size=2)
    assert GRPOSettings("legacy-grpo", loop).policy_updates is None
    assert SAMPOSettings("legacy-sampo", loop).policy_updates is None
    with pytest.raises(ValueError, match="effective batch"):
        GRPOSettings("legacy", replace(loop, per_device_batch_size=1))
    with pytest.raises(ValueError, match="effective batch"):
        SAMPOSettings("legacy", replace(loop, per_device_batch_size=1))


def test_explicit_settings_reject_conflicting_legacy_counts_and_limits() -> None:
    for loop in (TrainingLoop(max_steps=3, per_device_batch_size=2),
                 TrainingLoop(max_steps=3, gradient_accumulation_steps=2)):
        with pytest.raises(ValueError, match="replaces loop batch"):
            GRPOSettings("conflict", loop, policy_updates=update_settings())
    with pytest.raises(ValueError, match="agree with loop.max_steps"):
        SAMPOSettings("conflict", TrainingLoop(max_steps=4), policy_updates=update_settings())


def test_span_variant_is_explicit_and_separate_from_algorithm_identity() -> None:
    with pytest.raises(ValueError, match="select semantic-spans"):
        replace(update_settings(), policy_selection=ActionSelection("spans", ("thinking",)))
    selected = replace(update_settings(), objective_variant="semantic-spans",
                       policy_selection=ActionSelection("spans", ("thinking",)))
    assert selected.kl_selection.mode == "all"


@pytest.mark.parametrize("mutation", [
    {"schedule": {"unit": "episode", "budget": True}},
    {"schedule": {"unit": "episode", "budget": 1, "unrecognized": 1}},
    {"execution": {"records": 1, "context_tokens": 100, "statistic_bytes": 100, "oversized_policy": "truncate"}},
])
def test_catalog_rejects_unknown_or_coerced_update_contracts(mutation) -> None:
    updates = {**asdict(update_settings()), **mutation}
    with pytest.raises(ValueError):
        decode_training_selection(CatalogRef("training", "invalid"), {
            "selection_type": "grpo-settings", "id": "invalid", "loop": {"max_steps": 3},
            "policy_updates": updates,
        }, {})


def test_native_request_rejects_unqualified_explicit_executor_before_launch() -> None:
    from posttrain.common.variants import QWEN_35_2B
    from posttrain.train import GRPORequest

    settings = GRPOSettings("explicit", TrainingLoop(max_steps=3), policy_updates=update_settings())
    with pytest.raises(ValueError, match="no qualified resolved policy update executor"):
        GRPORequest(QWEN_35_2B, FakeRLBridge(), settings, FakeEnvironment(), _training(),
                    _inference(QWEN_35_2B, max_tokens=settings.max_completion_length))
