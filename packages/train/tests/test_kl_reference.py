"""The KL reference setting: resolution, TRL translation, veRL limits, and the reward contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common import CatalogRef
from posttrain.environment import EnvironmentBinding, EnvironmentSource, SamplingPolicy, VerifiersV1ConfigActivation
from posttrain.train import (
    SAMPOSettings,
    TrainingLoop,
    describe_kl_reference,
    kl_reference_problem,
    resolved_kl_reference,
)
from posttrain.train.backends.trl.policy_config import _online_rl_arguments
from posttrain.train.backends.trl.policy_optimization import _trainer_arguments
from posttrain.train.catalog_schema import decode_training_selection
from posttrain.train.reward_projection import RewardComponentProjection, RewardProjection
from posttrain.train.reward_recovery import reward_contract_digest


def _settings(**changes: Any) -> SAMPOSettings:
    values: dict[str, Any] = {
        "id": "sampo-kl",
        "loop": TrainingLoop(max_steps=1, max_length=8, per_device_batch_size=2),
        "max_prompt_length": 2,
        "max_completion_length": 6,
        "beta": 0.01,
    }
    values.update(changes)
    return SAMPOSettings(**values)


def _request(settings: SAMPOSettings, form: str) -> Any:
    return SimpleNamespace(
        settings=settings,
        policy=SimpleNamespace(provenance={}, form=form, weight_precision="bf16"),
        training=SimpleNamespace(backend="trl@1.12.0.post11", backend_options={}),
        inference=SimpleNamespace(backend="transformers@1", sampling={}, engine={}),
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_native_determinism_selection_reaches_trainer_arguments(tmp_path, enabled) -> None:
    request = _request(_settings(), "adapter")
    request.training.backend_options = {"full_determinism": enabled}
    arguments = _online_rl_arguments(request, tmp_path, {})
    assert arguments["full_determinism"] is enabled


def test_kl_reference_defaults_to_the_base_model_and_is_validated() -> None:
    assert _settings().kl_reference == "base"
    assert _settings(kl_reference="start").kl_reference == "start"
    with pytest.raises(ValueError, match="KL reference must be 'base' or 'start'"):
        _settings(kl_reference="adapter")
    data = {
        "selection_type": "sampo-settings",
        "id": "sampo-kl",
        "loop": {"max_steps": 1, "max_length": 8, "per_device_batch_size": 2},
        "max_prompt_length": 2,
        "max_completion_length": 6,
    }
    decoded = decode_training_selection(CatalogRef("training", "sampo-kl"), data, {})
    assert isinstance(decoded, SAMPOSettings) and decoded.kl_reference == "base"
    decoded = decode_training_selection(CatalogRef("training", "sampo-kl"), {**data, "kl_reference": "start"}, {})
    assert isinstance(decoded, SAMPOSettings) and decoded.kl_reference == "start"


@pytest.mark.parametrize(
    ("beta", "setting", "form", "expected"),
    [
        (0.0, "base", "adapter", "off"),
        # A run from the foundation model, including a fresh zero-initialized adapter, has one reference.
        (0.01, "start", "foundation", "base"),
        (0.01, "base", "foundation", "base"),
        (0.01, "base", "adapter", "base"),
        (0.01, "start", "adapter", "start"),
        # GDPO and CAPO have no setting and keep TRL's frozen copy of the starting adapter.
        (0.01, None, "adapter", "start"),
    ],
)
def test_resolved_kl_reference(beta: float, setting: str | None, form: str, expected: str) -> None:
    assert resolved_kl_reference(beta, setting, form) == expected


def test_trl_scores_a_continued_adapter_against_the_base_model_only_when_selected(tmp_path) -> None:
    continued = _online_rl_arguments(_request(_settings(), "adapter"), tmp_path, {})
    assert continued["peft_reference"] == "base"
    for request in (
        _request(_settings(kl_reference="start"), "adapter"),
        _request(_settings(), "foundation"),
        _request(_settings(beta=0.0), "adapter"),
    ):
        # TRL's default already gives the intended reference, so earlier releases keep working.
        assert "peft_reference" not in _online_rl_arguments(request, tmp_path, {})

    from posttrain.train.backends.trl.policy_config import kl_reference

    assert kl_reference(_request(_settings(), "adapter")) == "base"
    assert kl_reference(_request(_settings(kl_reference="start"), "adapter")) == "start"


def test_earlier_trl_releases_reject_a_base_reference_for_a_continued_adapter() -> None:
    from dataclasses import dataclass

    @dataclass
    class ReleasedConfig:
        beta: float = 0.0

    request = cast(Any, SimpleNamespace(training=SimpleNamespace(backend="trl@1.12.0.post10")))
    with pytest.raises(RuntimeError, match="does not provide peft_reference; .*post11"):
        _trainer_arguments(ReleasedConfig, {"peft_reference": "base"}, request)


def test_verl_rejects_only_references_it_cannot_provide() -> None:
    assert kl_reference_problem("trl@1.12.0.post11", 0.01, "base", "adapter") is None
    assert kl_reference_problem("trl@1.12.0.post11", 0.01, "start", "full-finetuned") is None
    assert kl_reference_problem("verl@candidate", 0.01, "base", "foundation") is None
    # veRL continues an adapter with the adapter disabled as its reference: the base model.
    assert kl_reference_problem("verl@candidate", 0.01, "base", "adapter") is None
    assert kl_reference_problem("verl@candidate", 0.01, "start", "full-finetuned") is None
    assert kl_reference_problem("verl@candidate", 0.0, "start", "adapter") is None
    adapter = kl_reference_problem("verl@candidate", 0.01, "start", "adapter")
    assert adapter is not None and "cannot hold a frozen copy of the starting adapter" in adapter
    full = kl_reference_problem("verl@candidate", 0.01, "base", "full-finetuned")
    assert full is not None and "set kl_reference: start" in full


def test_plan_line_names_the_reference() -> None:
    assert describe_kl_reference(0.0, "base") == "KL penalty: off (beta 0)"
    assert describe_kl_reference(0.01, "base") == "KL reference: base model (kl_reference: base, beta 0.01)"
    assert describe_kl_reference(0.01, "start").startswith("KL reference: starting checkpoint")


def test_reward_contract_digest_changes_with_the_kl_reference() -> None:
    environment = EnvironmentBinding(
        "environments/turns",
        "tool-use",
        EnvironmentSource("test", "https://example.test/env", "a" * 40),
        VerifiersV1ConfigActivation({"taskset": {}}),
        SamplingPolicy(max_tokens=128),
        num_tasks=20,
    )
    projection = RewardProjection("test", "1", (RewardComponentProjection("outcome", "scalar"),))

    def digest(settings: SAMPOSettings) -> str:
        request = SimpleNamespace(
            settings=settings, environment=environment, bridge=SimpleNamespace(reward_projection=projection)
        )
        return reward_contract_digest(cast(Any, request))

    base = digest(_settings())
    start = digest(_settings(kl_reference="start"))
    assert base != start
    # Choosing surplus groups by learning signal changes what is trained on; candidate order does not.
    from posttrain.train.profiles import ActiveGroupSampling
    from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings

    resolved = {
        "loop": TrainingLoop(max_steps=1, max_length=8, per_device_batch_size=1),
        "policy_updates": PolicyUpdateSettings(PolicyUpdateSchedule("episode", 1), PolicyExecutionBudget(1, 100, 1000)),
    }
    ordered = digest(_settings(**resolved, active_sampling=ActiveGroupSampling(3)))
    assert ordered == digest(_settings(**resolved, active_sampling=ActiveGroupSampling(3, retain="first")))
    assert ordered != digest(_settings(**resolved, active_sampling=ActiveGroupSampling(3, retain="learning_signal")))
    # "start" hashes like settings written before the field existed.
    import hashlib
    import json
    from dataclasses import asdict

    from posttrain.train.reward_recovery import _identity_value

    legacy = asdict(_settings(kl_reference="start"))
    legacy["loop"].pop("max_steps")
    legacy.pop("kl_reference")
    # Pre-engine checkpoints did not contain the additive selection field.
    legacy.pop("policy_updates")
    legacy["active_sampling"] = {"max_candidate_batches": legacy["active_sampling"]["max_candidate_batches"]}
    # Nor goal-relative turn credit, nor the projection's turn outcome selection.
    legacy.pop("goal_credit")
    legacy_projection = asdict(projection)
    legacy_projection.pop("turn_goal_prefix")
    legacy_projection.pop("turn_harm_key")
    payload = {
        "schema": "posttrain.reward-contract.v1",
        "settings": legacy,
        "projection": legacy_projection,
        "environment": environment,
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False, default=_identity_value).encode()
    ).hexdigest()
    assert start == expected
