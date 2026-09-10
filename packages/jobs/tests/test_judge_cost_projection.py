"""Standard judged jobs derive conservative paid-service usage before launch."""

from posttrain.environment import (
    EnvironmentBinding,
    EnvironmentSource,
    SamplingPolicy,
    VerifiersV1ConfigActivation,
)
from posttrain.jobs.definitions import _maximum_trajectories
from posttrain.jobs.native_judges import project_native_judge_usage
from posttrain.train import ActiveGroupSampling, GDPOSettings, GRPOSettings, TrainingLoop


def test_episode_judge_projection_covers_training_admission_and_judge_attempts() -> None:
    environment = EnvironmentBinding(
        "environments/test-paid-judge",
        "tool-use",
        EnvironmentSource("test", "https://example.test/environment", "a" * 40),
        VerifiersV1ConfigActivation(
            {
                "taskset": {
                    "task": {
                        "judges": [
                            {
                                "name": "quality",
                                "attempts": 2,
                                "input_budget_tokens": 8_192,
                                "sampling": {"max_tokens": 16_384},
                            }
                        ]
                    }
                },
                "agent": {"max_turns": 12},
            }
        ),
        SamplingPolicy(max_tokens=128),
        num_tasks=1,
    )
    settings = GDPOSettings(
        id="training/test-gdpo",
        revision="1",
        loop=TrainingLoop(max_steps=5, max_length=512, per_device_batch_size=1, gradient_accumulation_steps=32),
        num_prompts_per_step=8,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=256,
        max_admission_attempts=3,
        component_names=("task", "quality"),
        component_weights=(2.0, 1.0),
    )

    trajectories = _maximum_trajectories(settings)
    usage = project_native_judge_usage(environment, trajectories, {"quality": "judge/shared"})["judge/shared"]

    assert usage.requests == 5 * 8 * 4 * 3 * 2
    assert usage.input_tokens == usage.requests * 8_192
    assert usage.output_tokens == usage.requests * 16_384


def test_grpo_family_exposes_only_a_generic_collection_ceiling() -> None:
    settings = GRPOSettings(
        id="training/test-olmo",
        loop=TrainingLoop(max_steps=5, max_length=512, per_device_batch_size=1, gradient_accumulation_steps=32),
        num_prompts_per_step=8,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=256,
        algorithm="olmo3",
        beta=0.0,
        advantage_scaling="none",
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.272,
        importance_sampling_mode="token_truncate",
        importance_sampling_clip_min=None,
        importance_sampling_clip_max=2.0,
        active_sampling=ActiveGroupSampling(max_candidate_batches=10),
    )

    assert settings.max_collection_attempts == 10
    assert _maximum_trajectories(settings) == 5 * 8 * 4 * 10
