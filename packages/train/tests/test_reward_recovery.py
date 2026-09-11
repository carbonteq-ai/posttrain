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
