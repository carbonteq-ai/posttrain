"""Resolved update lifecycle inside the native synchronous veRL V1 driver."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from posttrain.common import RunContext

from ...profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings
from ...update_records import InvalidPolicyUpdate
from .policy_checkpoint import (
    inspect_driver_checkpoint,
    latest_driver_checkpoint,
    prune_staged_checkpoint_sources,
    publish_driver_checkpoint,
    seal_driver_checkpoint,
)


def single_actor_result(results: Any) -> Mapping[str, Any]:
    """ONE_TO_ALL results must describe one resolved actor state.

    Data-parallel ranks hold identical global populations, objectives and
    counters, so every rank must report exactly the same state and metrics;
    any divergence is a failed distributed update, not a value to average.
    """
    if (
        not isinstance(results, (list, tuple))
        or not results
        or not all(isinstance(value, Mapping) for value in results)
    ):
        raise InvalidPolicyUpdate("resolved driver requires native actor results")
    if any(dict(value) != dict(results[0]) for value in results[1:]):
        raise InvalidPolicyUpdate("resolved data-parallel actor ranks report different global state")
    return results[0]


def native_receipts(data: Any, expected_count: int) -> tuple[str, ...]:
    """Read only original episode receipts from TransferQueue extra fields."""
    values = list(data["extra_fields"])
    if len(values) != expected_count:
        raise InvalidPolicyUpdate("native rollout batch is not a complete resolved population")
    receipts = []
    for value in values:
        extra = getattr(value, "data", value)
        receipt = extra.get("posttrain_native_episode_receipt") if isinstance(extra, Mapping) else None
        if not isinstance(receipt, str) or not receipt:
            raise InvalidPolicyUpdate("native rollout lacks its original episode receipt")
        receipts.append(receipt)
    return tuple(receipts)


def native_candidate_tasks(batch: Any, expected_count: int) -> tuple[str, ...]:
    """Validate the entire reserved native prompt pool before any dispatch.

    Native dataloaders may wrap at an epoch boundary. Unique generated UIDs do
    not establish unique tasks, including candidates later discarded by refill.
    """
    try:
        values = tuple(getattr(value, "data", value) for value in batch["example_id"])
    except (KeyError, TypeError, AttributeError) as error:
        raise InvalidPolicyUpdate("native candidate reservation lacks task identities") from error
    if (
        len(values) != expected_count
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise InvalidPolicyUpdate("native candidate reservation requires distinct complete task inventory")
    return values


class ResolvedVeRLDriverSteps:
    """Mixin for PPOTrainerSync; native fit retains sync, logging and cleanup.

    Composition supplies ``resolved_settings`` and installs the resolved actor
    worker before initialization. This is not a public launch entrypoint. The
    Public composition still gates native active collection qualification.
    Asynchronous rollout, critics and skipping remain unsupported.
    """

    actor_rollout_wg: Any
    global_steps: int
    resolved_settings: Any
    replay_buffer: Any
    checkpoint_manager: Any
    on_sample_begin: Any
    on_sample_end: Any
    _add_batch_to_generate: Any
    _consume_rollout_metrics: Any
    _reserve_candidates: Any
    _candidate_pool: Any
    _resolved_candidate_tasks: tuple[str, ...]

    def _prepare_resolved_collection(self) -> None:
        reserved = getattr(self, "_resolved_active_candidates", 0)
        if reserved:
            self._reserve_candidates(reserved)
            self._resolved_candidate_tasks = native_candidate_tasks(self._candidate_pool, reserved)
            from .policy_collection import NativeActiveCollectionBuffer

            if isinstance(self.replay_buffer, NativeActiveCollectionBuffer):
                uids = tuple(str(getattr(value, "data", value)) for value in self._candidate_pool["uid"])
                self.replay_buffer.begin(self._resolved_candidate_tasks, uids, self.global_steps)
        else:
            self._add_batch_to_generate()

    def step(self, metrics: dict, timing_raw: dict) -> Any:
        import transfer_queue as tq  # pyright: ignore[reportMissingImports]
        from transfer_queue import KVBatchMeta  # pyright: ignore[reportMissingImports]

        state = single_actor_result(self.actor_rollout_wg.resolved_state())
        applied = state.get("applied")
        fresh = state.get("needs_population")
        if type(applied) is not int or applied < 0 or type(fresh) is not bool:
            raise InvalidPolicyUpdate("native actor returned an invalid resolved boundary")
        if self.global_steps != applied + 1:
            raise InvalidPolicyUpdate("driver step does not follow the committed actor version")
        receipts = None
        batch = KVBatchMeta(partition_id="train", keys=[], tags=[])
        if fresh:
            # Native fit counts the pending commit, but generation version tags
            # must identify the already-applied actor policy, including step 0.
            pending_commit = self.global_steps
            try:
                self.global_steps = applied
                self._selected_prompt_indices = {}
                self._dispatched_prompts = {}
                self.on_sample_begin()
                self._prepare_resolved_collection()
                batch, rollout_metrics = self.replay_buffer.sample(
                    global_steps=applied,
                    partition_id="train",
                    batch_size=self.resolved_settings.num_prompts_per_step,
                )
                metrics.update(rollout_metrics)
                self.on_sample_end()
                metrics.update(self._consume_rollout_metrics())
            finally:
                self.global_steps = pending_commit
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
            receipts = native_receipts(
                data, self.resolved_settings.num_prompts_per_step * self.resolved_settings.num_generations
            )
        else:
            # Previous native on_step_end woke rollout replicas for weight sync;
            # reclaim colocated rollout memory even when no collection is needed.
            self.checkpoint_manager.sleep_replicas()
        result = single_actor_result(self.actor_rollout_wg.resolved_update(receipts, expected_applied=applied))
        if result.get("train/rl/applied_optimizer_updates") != self.global_steps:
            raise InvalidPolicyUpdate("driver cannot advance without one committed actor update")
        metrics.update(result)
        return batch

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        # Stock metrics read discarded GRPO advantages from flattened PPO rows.
        # The actor supplies evaluated objective and selected prepared credit.
        metrics.update({"training/global_step": global_steps})
        metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})


def resolved_trainer_type(
    actor_worker: type,
    settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings,
    checkpoint_context: RunContext | None = None,
) -> type:
    """Compose the native synchronous trainer without changing global registries."""
    import ray  # pyright: ignore[reportMissingImports]
    from verl.trainer.ppo.utils import Role  # pyright: ignore[reportMissingImports]
    from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer  # pyright: ignore[reportMissingImports]
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync  # pyright: ignore[reportMissingImports]

    if settings.policy_updates is None:
        raise InvalidPolicyUpdate("resolved native trainer requires selected policy updates")

    class ResolvedTrainer(ResolvedVeRLDriverSteps, PPOTrainerSync):
        def __init__(self, config):
            sampler = config.trainer.v1.sampler
            custom = sampler.get("custom_sampler", {}) or {}
            active = settings.active_sampling if isinstance(settings, SAMPOSettings) else None
            native_active = config.algorithm.active_sampling
            if bool(native_active.enable) != (active is not None):
                raise InvalidPolicyUpdate("native driver configuration changes resolved collection semantics")
            if active is not None and (
                native_active.max_candidate_batches != active.max_candidate_batches
                or native_active.oversample != active.oversample
                or native_active.oversample_refill != active.oversample_refill
                or native_active.reward_std_epsilon != 0.0
                or native_active.metric != "group_reward"
            ):
                raise InvalidPolicyUpdate("native active sampling differs from selected candidate reservation")
            if (
                config.trainer.v1.trainer_mode != "sync"
                or config.trainer.v1.get("sync", {}).get("parameter_sync_step", 1) != 1
                or config.data.train_batch_size != settings.num_prompts_per_step
                or config.actor_rollout_ref.rollout.n != settings.num_generations
                or config.trainer.total_training_steps != settings.loop.max_steps
                or config.algorithm.use_kl_in_reward
                or config.algorithm.filter_groups.enable
                or sampler.get("sync_refill_failed_groups", False)
                or sampler.get("refill_all_failed_groups", False)
                or sampler.get("failed_group_attempts", 0)
                or custom.get("path")
                or custom.get("name")
                or config.trainer.get("rollout_data_dir")
                or config.trainer.get("default_hdfs_dir")
                or config.trainer.get("del_local_ckpt_after_load", False)
                or (
                    config.trainer.get("max_actor_ckpt_to_keep") is not None
                    and (
                        checkpoint_context is None
                        or config.trainer.max_actor_ckpt_to_keep != settings.loop.checkpoint_limit
                    )
                )
                or config.trainer.get("remove_previous_ckpt_in_save", False)
                or config.actor_rollout_ref.actor.get("checkpoint", {}).get("async_save", False)
                or any(
                    value.get("enable", False)
                    for name, value in config.get("skip", {}).items()
                    if not name.startswith("_")
                )
            ):
                raise InvalidPolicyUpdate("native driver configuration changes resolved collection semantics")
            self.resolved_settings = settings
            self._resolved_active_candidates = (
                settings.num_prompts_per_step * active.max_candidate_batches if active is not None else 0
            )
            PPOTrainerSync.__init__(self, config)
            if self.use_critic or self.use_teacher_policy:
                raise InvalidPolicyUpdate("resolved native driver does not support critic or teacher pipelines")

        def _build_replay_buffer(self):
            if self._resolved_active_candidates:
                # Reuse native round sizing, classification, observation,
                # eviction and surplus cleanup. Dispatch from one checked pool
                # rather than wrapping the dataloader independently each round.
                replay = super()._build_replay_buffer()
                replay.dispatch_fn = self._dispatch_reserved
                if checkpoint_context is not None:
                    from .policy_collection import NativeActiveCollectionBuffer

                    return NativeActiveCollectionBuffer(
                        replay,
                        checkpoint_context,
                        Path(self.config.trainer.default_local_dir).parent / "collection-evidence",
                        settings.num_generations,
                    )
                return replay
            # Unfiltered selections admit original complete groups directly.
            sampler = self.config.trainer.v1.sampler
            return ReplayBuffer(
                trainer_mode="sync",
                trainer_config=self.config.trainer.v1.get("sync", {}),
                max_off_policy_threshold=sampler.max_off_policy_threshold,
                max_off_policy_strategy=sampler.max_off_policy_strategy,
                sampler_kwargs=sampler.sampler_kwargs,
            )

        def _init_dataloader(self):
            super()._init_dataloader()
            # Native fit's epoch guard counts driver updates, whereas collection
            # can reuse a population for several updates. One optimizer window
            # spans the declared budget; native collection keeps its own cursor.
            self.steps_per_epoch = settings.loop.max_steps
            self.total_training_steps = settings.loop.max_steps
            self.config.trainer.total_epochs = 1

        def _init_resource_pool_mgr(self):
            super()._init_resource_pool_mgr()
            roles = [role for role in self.role_worker_mapping if role in (Role.ActorRollout, Role.ActorRolloutRef)]
            if len(roles) != 1:
                raise InvalidPolicyUpdate("native trainer does not expose one actor role")
            self.role_worker_mapping[roles[0]] = ray.remote(actor_worker)

        def _save_checkpoint(self):
            from ...update_recovery import FILENAME

            checkpoint = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
            if (checkpoint / FILENAME).exists():
                state = inspect_driver_checkpoint(checkpoint)
                actor = single_actor_result(self.actor_rollout_wg.resolved_state())
                if actor.get("applied") != state.native_applied_updates:
                    raise InvalidPolicyUpdate("existing driver checkpoint differs from the active actor")
            else:
                # The native model-only writer cannot own rotation. Defer it
                # until the complete driver save is staged for host publication.
                keep = self.config.trainer.max_actor_ckpt_to_keep
                try:
                    self.config.trainer.max_actor_ckpt_to_keep = None
                    super()._save_checkpoint()
                    seal_driver_checkpoint(checkpoint)
                finally:
                    self.config.trainer.max_actor_ckpt_to_keep = keep
            if checkpoint_context is not None and settings.loop.checkpoint_steps > 0:
                publication_root = Path(self.config.trainer.default_local_dir).parent / "checkpoint-publications"
                publish_driver_checkpoint(checkpoint_context, checkpoint, publication_root)
                prune_staged_checkpoint_sources(
                    Path(self.config.trainer.default_local_dir), publication_root, settings.loop.checkpoint_limit
                )

        def _load_checkpoint(self):
            config = self.config.trainer
            mode, original_path = config.resume_mode, config.resume_from_path
            if mode == "disable":
                return super()._load_checkpoint()
            if mode == "auto":
                checkpoint = latest_driver_checkpoint(Path(config.default_local_dir))
            elif mode == "resume_path":
                checkpoint = Path(original_path)
                inspect_driver_checkpoint(checkpoint)
            else:
                raise InvalidPolicyUpdate("unknown resolved native recovery mode")
            if checkpoint is not None:
                # Authenticate all actor and driver bytes before either native
                # model loading or torch.load of the dataloader state.
                inspect_driver_checkpoint(checkpoint)
            try:
                config.resume_mode = "disable" if checkpoint is None else "resume_path"
                config.resume_from_path = None if checkpoint is None else str(checkpoint.resolve())
                super()._load_checkpoint()
                actor = single_actor_result(self.actor_rollout_wg.resolved_state())
                if actor.get("applied") != self.global_steps:
                    raise InvalidPolicyUpdate("restored driver and actor applied counters disagree")
            finally:
                config.resume_mode, config.resume_from_path = mode, original_path

    return ResolvedTrainer
