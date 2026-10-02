"""Recipe worker types that retain native veRL initialization and stepping."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...update_records import InvalidPolicyUpdate
from .policy_updates import resolved_verl_engine_type


def resolved_worker_types(
    training_worker: type, actor_rollout_worker: type, registry: Any, *,
    validate_engine: Callable[[Any, Any], None] | None = None,
) -> tuple[type, type]:
    """Specialize native construction through the explicit engine factory seam.

    Import and provide worker classes from the selected isolated native runtime.
    No initialized engine or global registry is mutated. Public launch remains
    guarded until driver/actor composition and runtime qualification complete.
    """
    if not callable(getattr(training_worker, "create_engine", None)):
        raise InvalidPolicyUpdate("selected veRL worker lacks the engine factory extension")

    class ResolvedTrainingWorker(training_worker):
        def create_engine(self) -> Any:
            if (self.engine_config.strategy != "fsdp" or self.model_config.get("use_remove_padding", False)
                    or self.engine_config.use_fused_kernels or self.engine_config.use_dynamic_bsz
                    or self.engine_config.micro_batch_size_per_gpu != 1):
                raise InvalidPolicyUpdate("resolved veRL worker requires qualified dense one-context FSDP execution")
            if validate_engine is not None:
                validate_engine(self.model_config, self.engine_config)
            native_engine = registry.get_engine_cls(model_type=self.config.model_type,
                                                    backend=self.engine_config.strategy)
            return resolved_verl_engine_type(native_engine)(model_config=self.model_config,
                engine_config=self.engine_config, optimizer_config=self.optimizer_config,
                checkpoint_config=self.checkpoint_config)

    class ResolvedActorRolloutWorker(actor_rollout_worker):
        actor_worker_cls = ResolvedTrainingWorker

    return ResolvedTrainingWorker, ResolvedActorRolloutWorker


def resolved_actor_worker_type(
    training_worker: type, actor_rollout_worker: type, registry: Any,
    session_factory: Callable[[Any, Any], Any],
    *, validate_engine: Callable[[Any, Any], None] | None = None,
) -> type:
    """Attach one resolved session after native model/optimizer initialization.

    RPC payloads contain durable native receipt JSON, not flattened PPO rows.
    Native ONE_TO_ALL dispatch is qualified here only for a single actor rank;
    the session validates that condition before consuming any optimizer work.
    """
    from verl.single_controller.base.decorator import Dispatch, register  # pyright: ignore[reportMissingImports]

    _, parent = resolved_worker_types(training_worker, actor_rollout_worker, registry, validate_engine=validate_engine)

    class ResolvedActorWorker(parent):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            super().init_model()
            self.resolved_session = session_factory(self.actor.engine, self)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def resolved_state(self):
            return self.resolved_session.state()

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def resolved_update(self, receipts, expected_applied):
            return self.resolved_session.update(receipts, expected_applied=expected_applied)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
            if hdfs_path is not None:
                raise InvalidPolicyUpdate("resolved veRL remote checkpoint promotion is not qualified")
            if max_ckpt_to_keep is not None:
                raise InvalidPolicyUpdate("resolved checkpoint retention requires the sealed driver checkpoint owner")
            return self.resolved_session.save_checkpoint(Path(local_path), expected_applied=global_step)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
            if hdfs_path is not None or del_local_after_load:
                raise InvalidPolicyUpdate("resolved veRL recovery requires retained local sealed evidence")
            return self.resolved_session.load_checkpoint(Path(local_path))

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def update_actor(self, data):
            raise InvalidPolicyUpdate("resolved actor cannot execute legacy flattened PPO batches")

    return ResolvedActorWorker
