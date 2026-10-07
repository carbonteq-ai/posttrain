from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest
from posttrain.environment import EnvironmentBinding, EnvironmentSource, SamplingPolicy, VerifiersV1ConfigActivation
from posttrain.train.reward_projection import RewardComponentProjection, RewardProjection
from posttrain.train.reward_recovery import retain_reward_contract, validate_reward_recovery


def test_missing_changed_or_corrupt_contract_cannot_resume(tmp_path):
    with pytest.raises(ValueError, match="lacks"):
        validate_reward_recovery(tmp_path, "a" * 64)
    retain_reward_contract(tmp_path, "a" * 64)
    validate_reward_recovery(tmp_path, "a" * 64)
    retain_reward_contract(tmp_path, "a" * 64)
    with pytest.raises(ValueError, match="differs"):
        validate_reward_recovery(tmp_path, "b" * 64)
    with pytest.raises(ValueError, match="differs"):
        retain_reward_contract(tmp_path, "b" * 64)


def test_absent_update_settings_preserve_pre_engine_recovery_digest() -> None:
    from posttrain.train.profiles import TrainingLoop
    from posttrain.train.reward_recovery import reward_contract_digest

    @dataclass
    class LegacySettings:
        loop: TrainingLoop

    @dataclass
    class AdditiveSettings:
        loop: TrainingLoop
        policy_updates: object | None = None

    loop = TrainingLoop(max_steps=3, per_device_batch_size=2)
    projection = RewardProjection("test", "1", (RewardComponentProjection("outcome", "scalar"),))
    request: Any = SimpleNamespace(
        settings=LegacySettings(loop), environment={"id": "env@1"}, bridge=SimpleNamespace(reward_projection=projection)
    )
    before = reward_contract_digest(request)
    request.settings = AdditiveSettings(loop)
    assert reward_contract_digest(request) == before


def test_changed_native_judge_config_invalidates_resume_even_with_same_package_revision(tmp_path):
    from posttrain.train.reward_recovery import reward_contract_digest

    @dataclass
    class Loop:
        max_steps: int = 5

    @dataclass
    class Settings:
        loop: Loop

    environment = EnvironmentBinding(
        "environments/turns",
        "tool-use",
        EnvironmentSource("test", "https://example.test/env", "a" * 40),
        VerifiersV1ConfigActivation({"taskset": {"task": {"judges": [{"rubric": "first"}]}}}),
        SamplingPolicy(max_tokens=128),
        num_tasks=1,
    )
    projection = RewardProjection("test", "1", (RewardComponentProjection("outcome", "scalar"),))
    request: Any = SimpleNamespace(
        settings=Settings(Loop()), environment=environment, bridge=SimpleNamespace(reward_projection=projection)
    )
    digest = reward_contract_digest(request)
    retain_reward_contract(tmp_path, digest)
    request.settings.loop.max_steps = 10
    assert reward_contract_digest(request) == digest
    request.environment = replace(
        environment, activation=VerifiersV1ConfigActivation({"taskset": {"task": {"judges": [{"rubric": "second"}]}}})
    )
    assert request.environment.revision == environment.revision
    with pytest.raises(ValueError, match="differs"):
        validate_reward_recovery(tmp_path, reward_contract_digest(request))


def test_goal_credit_off_and_unselected_turn_outcomes_preserve_the_recovery_digest() -> None:
    from posttrain.train.profiles import SAMPOSettings, TrainingLoop
    from posttrain.train.reward_recovery import reward_contract_digest

    loop = TrainingLoop(max_steps=3, per_device_batch_size=2)
    settings = SAMPOSettings("sampo", loop, max_prompt_length=2, max_completion_length=6)
    projection = RewardProjection("test", "1", (RewardComponentProjection("outcome", "scalar"),))
    request: Any = SimpleNamespace(
        settings=settings, environment={"id": "env@1"}, bridge=SimpleNamespace(reward_projection=projection)
    )
    import dataclasses
    import hashlib
    import json

    legacy = dataclasses.asdict(settings)
    legacy.pop("goal_credit")
    legacy.pop("goal_credit_scale")
    legacy.pop("anchor_fallback")
    legacy.pop("policy_updates")
    legacy["loop"].pop("max_steps")
    legacy.pop("kl_reference") if legacy.get("kl_reference") == "start" else None
    legacy["active_sampling"].pop("oversample", None)
    legacy["active_sampling"].pop("oversample_refill", None)
    legacy["active_sampling"].pop("retain", None)
    old_projection = dataclasses.asdict(projection)
    old_projection.pop("turn_goal_prefix")
    old_projection.pop("turn_harm_key")
    old_projection.pop("turn_state_key")
    expected = hashlib.sha256(
        json.dumps(
            {
                "schema": "posttrain.reward-contract.v1",
                "settings": legacy,
                "projection": old_projection,
                "environment": {"id": "env@1"},
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert reward_contract_digest(request) == expected
    request.settings = dataclasses.replace(settings, goal_credit="group-relative")
    assert reward_contract_digest(request) != expected
