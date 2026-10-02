"""Native resolved recipe entrypoint; public launch stays qualification-gated."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Any

from posttrain.common import RunContext

from ...online_rl import policy_sampling_from_mapping
from ...profiles import SAMPOSettings
from ...update_records import InvalidPolicyUpdate
from .contracts import VerlLaunchManifest
from .policy_capabilities import native_context_layout, native_execution_capabilities, validate_native_execution_profile
from .policy_driver import resolved_trainer_type
from .policy_observer import JOURNAL_NAME, ResolvedWorkerObserver
from .policy_runner import resolved_task_runner_type
from .policy_runtime import SCORE_CONTRACT, base_reference_provider, native_job_identity
from .policy_session import actor_session_from_manifest
from .resolved_workers import resolved_actor_worker_type


def validate_native_selection(manifest: VerlLaunchManifest) -> None:
    selected = manifest.payload.resolved_settings
    if selected is None or manifest.run_context is None:
        raise InvalidPolicyUpdate("resolved native entrypoint requires typed settings and actual host context")
    settings = selected.settings
    updates = settings.policy_updates
    runtime = manifest.payload.training.runtime
    if (updates is None
            or manifest.payload.training.target.world_size != 1 or runtime.nodes != 1
            or runtime.devices_per_node not in (None, 1)
            or settings.mask_truncated_completions or updates.objective_variant == "semantic-spans"
            or any(getattr(settings, name, None) is not None for name in (
                "adaptive_curriculum", "dynamic_sampling"))
            or (getattr(settings, "active_sampling", None) is not None and not isinstance(settings, SAMPOSettings))):
        raise InvalidPolicyUpdate("resolved native entrypoint has not qualified this execution or collection selection")
    native_context_layout(manifest)
    examples = manifest.payload.environment.examples
    if len({value.id for value in examples}) != len(examples) or len(examples) < settings.num_prompts_per_step:
        raise InvalidPolicyUpdate("resolved native collection requires enough distinct tasks for complete groups")
    sampling = policy_sampling_from_mapping(manifest.payload.rollout.sampling, settings.max_completion_length)
    if (sampling.top_p != 1 or sampling.top_k != 0 or sampling.min_p not in (None, 0)
            or sampling.repetition_penalty != 1 or sampling.presence_penalty != 0):
        raise InvalidPolicyUpdate("resolved native entrypoint requires unwarped synchronous sampler evidence")


def run_native(manifest: VerlLaunchManifest, config: Any) -> None:
    """Install the native actor/trainer/runner recipe before model initialization."""
    validate_native_selection(manifest)
    import ray  # pyright: ignore[reportMissingImports]
    from verl.trainer.main_ppo import run_ppo  # pyright: ignore[reportMissingImports]

    # Native RunContext contains process-local synchronization state. Construct
    # the trainer and its context inside the runner, rather than capturing that
    # live object in a dynamically serialized Ray actor class.
    build_native_trainer(manifest, config)
    runner = resolved_task_runner_type(partial(build_native_trainer, manifest))
    run_ppo(config, ray.remote(runner))


def build_native_trainer(manifest: VerlLaunchManifest, config: Any) -> Any:
    """Construct process-local context and native recipe in the owning process."""
    from verl.workers.engine import EngineRegistry  # pyright: ignore[reportMissingImports]
    from verl.workers.engine_workers import (  # pyright: ignore[reportMissingImports]
        ActorRolloutRefWorker,
        TrainingWorker,
    )

    selected = manifest.payload.resolved_settings
    assert selected is not None
    settings = selected.settings
    validate_native_execution_profile(manifest, config.actor_rollout_ref.model,
        config.actor_rollout_ref.actor.fsdp_config, actor_config=config.actor_rollout_ref.actor)
    behavior = policy_sampling_from_mapping(manifest.payload.rollout.sampling, settings.max_completion_length)
    if config.actor_rollout_ref.rollout.temperature != behavior.temperature:
        raise InvalidPolicyUpdate("native rollout temperature differs from the selected scoring distribution")
    # Validate the real native driver config before starting Ray/TransferQueue.
    # The constructor only creates the replay buffer, not model workers.
    context = manifest.run_context
    assert context is not None
    observer = ResolvedWorkerObserver(manifest.output_directory / JOURNAL_NAME, context)
    checkpoint_context = RunContext(**context.model_dump(), observer=observer)
    resolved_trainer_type(ActorRolloutRefWorker, settings, checkpoint_context)(config)

    def session_factory(engine, worker):
        runtime_identity, template = native_job_identity(manifest, engine)
        context = manifest.run_context
        assert context is not None
        observer = ResolvedWorkerObserver(manifest.output_directory / JOURNAL_NAME, context)
        capabilities = native_execution_capabilities(manifest, engine)
        session = actor_session_from_manifest(manifest, engine, observer, capabilities=capabilities,
            runtime_identity=runtime_identity, template_revision=template, score_temperature=behavior.temperature,
            score_contract=SCORE_CONTRACT, reference_scores=base_reference_provider(manifest, engine))
        session.context.event("resolved_actor_initialized", {"runtime_identity": runtime_identity,
            "template_revision": template, "score_contract": SCORE_CONTRACT, "score_temperature": behavior.temperature})
        return session

    actor = resolved_actor_worker_type(TrainingWorker, ActorRolloutRefWorker, EngineRegistry, session_factory,
        validate_engine=partial(validate_native_execution_profile, manifest))
    trainer = resolved_trainer_type(actor, settings, checkpoint_context)
    return trainer(config=config)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m posttrain.train.backends.verl.policy_native MANIFEST.json [HYDRA_OVERRIDE...]")
    manifest = VerlLaunchManifest.read(Path(sys.argv[1]).resolve())
    validate_native_selection(manifest)
    from hydra import compose, initialize_config_module
    from verl.trainer.ppo.utils import need_critic, need_reference_policy  # pyright: ignore[reportMissingImports]
    from verl.utils.config import validate_config  # pyright: ignore[reportMissingImports]
    from verl.utils.device import auto_set_device  # pyright: ignore[reportMissingImports]

    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        config = compose(config_name="ppo_trainer", overrides=sys.argv[2:])
    auto_set_device(config)
    validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run_native(manifest, config)


if __name__ == "__main__":
    main()
