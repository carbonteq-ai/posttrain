"""Rollout collection and credit projection for TRL policy optimization."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

from posttrain.common import (
    EPISODE_ENDING_ATTRIBUTE,
    RunContext,
    TraceFactSet,
    TraceFactUpdateObservation,
    TraceObservation,
)

from ...grpo_observations import episode_ending_counts
from ...online_rl import BehaviorPolicySpan, EnvironmentRollout, RolloutBatch, run_observed_rollouts
from ...profiles import shape_online_reward
from ...requests import CAPORequest, GDPORequest, GRPORequest, SAMPORequest
from ...reward_advantages import compute_capo_advantages, compute_gdpo_advantages
from ...sampo_advantages import compute_sampo_advantages
from ...update_plan import ExecutionCapabilities
from ...update_records import InvalidPolicyUpdate, PolicyVersions, SemanticSpan
from ...update_telemetry import collection_metrics
from ..policy_update_admission import AdmittedNativePopulation
from .update_totals import RolloutUpdateTotals

type PolicyTechnique = Literal["grpo", "dapo", "olmo3", "sampo", "gdpo", "capo"]


def collect_resolved_population(
    context: RunContext,
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest,
    tokenizer: Any,
    trainer: Any,
    rows: list[dict[str, Any]],
    capabilities: ExecutionCapabilities,
    *,
    population_id: str,
    template_revision: str,
    versions: PolicyVersions,
    selector_digest: str,
    attempt_offset: int,
    spans: tuple[SemanticSpan, ...] = (),
    max_overflow_retries: int = 0,
    totals: RolloutUpdateTotals | None = None,
    process_credit: Any = None,
) -> AdmittedNativePopulation:
    """Reuse ordinary collection and complete-group admission before resolution.

    The caller supplies the complete logical collection rows, independently of
    the native dataloader's single resolved-update slot. Publish the validated
    replay artifact before returning to the native optimizer lifecycle.
    """
    bridge = cast(Any, request.bridge)
    if getattr(bridge, "policy_update_context_contract", None) != "causal-text@1" or not callable(
        getattr(bridge, "retain_population", None)
    ):
        raise InvalidPolicyUpdate("resolved collection requires native conditioning and retained artifact support")
    if request.settings.policy_updates is None:
        raise InvalidPolicyUpdate("resolved collection requires an explicit policy update selection")
    if trainer.accelerator.num_processes != 1 or bool(getattr(trainer, "active_sampling", False)):
        raise InvalidPolicyUpdate(
            "resolved production collection has not qualified distributed or native active refills"
        )
    applied = trainer.state.global_step
    if type(applied) is not int:
        raise InvalidPolicyUpdate("resolved collection requires an exact native applied counter")
    AdmittedNativePopulation._validate_counters(applied, attempt_offset, max_overflow_retries)  # noqa: SLF001
    collector = rollout_function(context, request, tokenizer, totals, retain_native=True)
    rollouts = collector([row["prompt"] for row in rows], trainer, inputs=rows)
    if not isinstance(rollouts, tuple):
        raise InvalidPolicyUpdate("resolved collector returned flattened trainer rows")
    artifact = bridge.retain_population(rollouts)
    admitted = AdmittedNativePopulation.from_retained_artifact(
        artifact,
        rollouts,
        request.settings,
        capabilities,
        population_id=population_id,
        template_revision=template_revision,
        versions=versions,
        sampler_step=applied,
        selector_digest=selector_digest,
        spans=spans,
        applied_update_offset=applied,
        attempt_offset=attempt_offset,
        max_overflow_retries=max_overflow_retries,
        process_credit=process_credit,
    )
    context.artifact(artifact)
    _observe_collection(context, request.settings, rollouts, admitted, step=applied + 1)
    return admitted


def collect_active_resolved_population(
    context: RunContext,
    request: SAMPORequest,
    tokenizer: Any,
    trainer: Any,
    reserved: list[dict[str, Any]],
    capabilities: ExecutionCapabilities,
    *,
    evidence_directory: Path,
    population_id: str,
    template_revision: str,
    versions: PolicyVersions,
    selector_digest: str,
    attempt_offset: int,
    max_overflow_retries: int = 0,
    totals: RolloutUpdateTotals | None = None,
    process_credit: Any = None,
) -> AdmittedNativePopulation:
    """Run TRL post11 active rounds over one reserved task pool, then admit the selection.

    The resolved path collects explicit rows rather than TRL's dataloader, so the
    native refill cannot run here; ``ActiveRoundPlan`` applies TRL's exact round
    arithmetic and zero-epsilon spread rule on the shaped reward that credit uses.
    Every reserved candidate is accounted for in content-addressed evidence before
    admission: selected, finished-but-surplus, failed or unused. Only selected
    groups enter the retained population; every collected episode is still
    observed as a trace by the ordinary collector.
    """
    from ...update_active_rounds import ActiveRoundPlan
    from ..policy_collection_evidence import publish_collection_snapshot

    bridge = cast(Any, request.bridge)
    settings = request.settings
    active = settings.active_sampling
    if getattr(bridge, "policy_update_context_contract", None) != "causal-text@1" or not callable(
        getattr(bridge, "retain_population", None)
    ):
        raise InvalidPolicyUpdate("resolved collection requires native conditioning and retained artifact support")
    if settings.policy_updates is None or active is None or settings.adaptive_curriculum is not None:
        raise InvalidPolicyUpdate("resolved active collection requires explicit updates and no curriculum")
    if trainer.accelerator.num_processes != 1 or bool(getattr(trainer, "active_sampling", False)):
        raise InvalidPolicyUpdate("resolved active collection replaces, and cannot nest, native TRL refills")
    applied = trainer.state.global_step
    if type(applied) is not int:
        raise InvalidPolicyUpdate("resolved collection requires an exact native applied counter")
    AdmittedNativePopulation._validate_counters(applied, attempt_offset, max_overflow_retries)  # noqa: SLF001
    plan = ActiveRoundPlan(
        settings.num_prompts_per_step, active.max_candidate_batches, active.oversample, active.oversample_refill
    )
    tasks = [str(row["example_id"]) for row in reserved]
    if len(tasks) != plan.pool or len(set(tasks)) != len(tasks):
        raise InvalidPolicyUpdate("resolved active collection requires one distinct task per reserved candidate")
    generations = settings.num_generations
    state: dict[str, Any] = {
        "schema": "posttrain.trl-active-collection@1",
        "sampler_step": applied,
        "metric": "shaped_reward",
        "reward_std_epsilon": 0.0,
        "reserved": [{"uid": f"candidate-{index}", "task": task} for index, task in enumerate(tasks)],
        "rounds": [],
        "groups": [],
        "selected": None,
        "status": "reserved",
    }
    destination = evidence_directory.resolve()
    publish_collection_snapshot(context, destination, state)
    collector = rollout_function(context, request, tokenizer, totals, retain_native=True)
    retained: list[tuple[str, tuple[EnvironmentRollout, ...]]] = []
    try:
        while (next_round := plan.next_round()) is not None:
            requested, size = next_round
            batch = list(range(plan.cursor, plan.cursor + size))
            state["rounds"].append({"index": plan.rounds + 1, "uids": [f"candidate-{index}" for index in batch]})
            state["status"] = "dispatching"
            publish_collection_snapshot(context, destination, state)
            rows = [reserved[index] for index in batch for _ in range(generations)]
            rollouts = collector([row["prompt"] for row in rows], trainer, inputs=rows)
            if not isinstance(rollouts, tuple):
                raise InvalidPolicyUpdate("resolved collector returned flattened trainer rows")
            by_task: dict[str, list[EnvironmentRollout]] = {}
            for rollout in rollouts:
                if rollout.example_id not in {tasks[index] for index in batch}:
                    raise InvalidPolicyUpdate("resolved active round returned an unreserved task")
                by_task.setdefault(rollout.example_id, []).append(rollout)
            kept = 0
            for index in batch:
                group = tuple(by_task.get(tasks[index], ()))
                complete = len(group) == generations
                rewards = [
                    shape_online_reward(
                        settings, rollout.reward, len(rollout.completion_ids), is_truncated=rollout.is_truncated
                    )
                    for rollout in group
                ]
                if complete and not all(math.isfinite(value) for value in rewards):
                    raise InvalidPolicyUpdate("resolved active classification requires finite shaped rewards")
                eligible = complete and max(rewards) > min(rewards)
                state["groups"].append(
                    {
                        "uid": f"candidate-{index}",
                        "terminal": "finished" if complete else "failed",
                        "traces": [rollout.trace.external_id for rollout in group],
                        "metric_values": rewards,
                        "spread_eligible": eligible,
                    }
                )
                if eligible:
                    kept += 1
                    retained.append((f"candidate-{index}", group))
            plan.record(requested, size, kept)
            state["status"] = "round-observed"
            publish_collection_snapshot(context, destination, state)
        plan.require_full()
        selected = retained[: settings.num_prompts_per_step]
        state["selected"] = [uid for uid, _ in selected]
        state["status"] = "selected"
        publish_collection_snapshot(context, destination, state)
    except Exception:
        state["status"] = "failed"
        publish_collection_snapshot(context, destination, state)
        raise
    population = tuple(rollout for _, group in selected for rollout in group)
    artifact = bridge.retain_population(population)
    admitted = AdmittedNativePopulation.from_retained_artifact(
        artifact,
        population,
        settings,
        capabilities,
        population_id=population_id,
        template_revision=template_revision,
        versions=versions,
        sampler_step=applied,
        selector_digest=selector_digest,
        applied_update_offset=applied,
        attempt_offset=attempt_offset,
        max_overflow_retries=max_overflow_retries,
        process_credit=process_credit,
    )
    context.artifact(artifact)
    context.metrics(
        plan.metrics(generations), step=applied + 1, attributes={"measurement_scope": "resolved-active-collection"}
    )
    _observe_collection(context, settings, population, admitted, step=applied + 1)
    return admitted


def _report_device_memory(context: RunContext, step: int) -> None:
    """Device memory the trainer holds as colocated vLLM wakes, and its peak since the last wake.

    The sampler reclaims a fixed share of the device on wake; these values show
    what the trainer still holds when it does, so a wake-up out-of-memory can be
    attributed (live tensors versus allocator reservation versus other users).
    """
    import torch

    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    gib = 1024**3
    values = {
        "train/rl/device_memory_allocated_gib": torch.cuda.memory_allocated() / gib,
        "train/rl/device_memory_reserved_gib": torch.cuda.memory_reserved() / gib,
        "train/rl/device_memory_free_gib": free / gib,
        "train/rl/device_memory_total_gib": total / gib,
        "train/rl/device_memory_peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
    }
    torch.cuda.reset_peak_memory_stats()
    print(  # noqa: T201 - also in job logs, next to the sampler's own wake-up errors
        "device memory before sampler wake: "
        + ", ".join(f"{name.rsplit('/', 1)[-1]}={value:.2f}" for name, value in values.items()),
        flush=True,
    )
    context.metrics(values, step=step + 1, attributes={"measurement_scope": "sampler-wake"})


def _observe_collection(
    context: RunContext,
    settings: Any,
    population: tuple[EnvironmentRollout, ...],
    admitted: AdmittedNativePopulation,
    *,
    step: int,
) -> None:
    """Report the admitted population once, at the first update that trains on it.

    Rollout counts and timings stay with the candidate-scope rollout totals; these
    values describe only the admitted groups that credit was prepared on.
    """
    values = collection_metrics(settings, population, credit_estimator_id=admitted.resolved.credit.estimator_id)
    if values:
        context.metrics(values, step=step, attributes={"measurement_scope": "resolved-collection"})


def technique(request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest) -> PolicyTechnique:
    if isinstance(request, GRPORequest):
        return request.settings.algorithm
    if isinstance(request, GDPORequest):
        return "gdpo"
    return "capo" if isinstance(request, CAPORequest) else "sampo"


def rollout_function(
    context: RunContext,
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest,
    tokenizer: Any,
    totals: RolloutUpdateTotals | None = None,
    *,
    retain_native: bool = False,
) -> Any:
    """Translate TRL generation batches into the public environment-rollout bridge contract.

    Rollout metrics go to `totals`, which writes one value per update when the
    trainer's step-end callback flushes it. Without a callback, an update's
    totals are written when the next update's first batch arrives.
    """

    from .policy_config import _rollout_execution_config

    flush_on_next_step = totals is None
    update_totals = totals if totals is not None else RolloutUpdateTotals(context)

    rollout_execution = _rollout_execution_config(request)
    rollout_batch_step: int | None = None
    rollout_batch_ordinal = 0
    collection_ordinal = 0

    def run_rollouts(
        prompts: list[Any],
        trainer: Any,
        *,
        inputs: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | tuple[EnvironmentRollout, ...]:
        nonlocal collection_ordinal, rollout_batch_ordinal, rollout_batch_step
        if inputs is None or len(inputs) != len(prompts):
            raise ValueError("TRL must provide dataset rows aligned with rollout prompts")
        try:
            example_ids = tuple(str(row["example_id"]) for row in inputs)
        except KeyError as error:
            raise ValueError("every online-RL dataset row requires an example_id") from error
        optimizer_step = int(trainer.state.global_step) + 1
        if rollout_batch_step != optimizer_step:
            if flush_on_next_step and rollout_batch_step is not None:
                update_totals.flush(rollout_batch_step)
            rollout_batch_step = optimizer_step
            rollout_batch_ordinal = 0
        rollout_batch_ordinal += 1
        attributes = {
            "technique": technique(request),
            "model_variant_id": request.policy.id,
            "training_settings_id": request.settings.id,
            "optimizer_step": optimizer_step,
            "rollout_batch_ordinal": rollout_batch_ordinal,
        }

        endings: list[object] = []

        def observe_trace(trace: TraceObservation) -> None:
            # Every attempted rollout is observed once, including failed and
            # replaced ones, so endings count the attempted population.
            endings.append(trace.attributes.get(EPISODE_ENDING_ATTRIBUTE))
            context.trace(
                TraceObservation(
                    trace_type=trace.trace_type,
                    external_id=trace.external_id,
                    payload=trace.payload,
                    attributes={**trace.attributes, **attributes},
                    facts=trace.facts,
                )
            )

        started_at = time.perf_counter()
        admission = None
        with context.phase("rollout", {"backend": "trl", "logical_step": optimizer_step}):
            batch = _rollout_batch(request, trainer, example_ids, optimizer_step, rollout_batch_ordinal)
            if retain_native:
                applied = trainer.state.global_step
                batch = replace(batch, behavior_policy=BehaviorPolicySpan(applied, applied))

            def collect(selected: RolloutBatch) -> Sequence[EnvironmentRollout]:
                nonlocal collection_ordinal
                if rollout_execution is None:
                    from .online_rl import TrlPolicyGenerator

                    generator = TrlPolicyGenerator(
                        trainer,
                        tokenizer,
                        request.policy,
                        request.settings,
                        request.training,
                        **({"retain_generation_logprobs": True} if retain_native else {}),
                    )
                    return asyncio.run(run_observed_rollouts(request.bridge, selected, generator, observe_trace))

                from .async_collection_runtime import TrlAsyncCollectionRuntime

                runtime = getattr(trainer, "_posttrain_async_collection_runtime", None)
                if runtime is None:
                    bridge = cast(Any, request.bridge)
                    rollout_timeout = bridge.rollout_timeout_seconds
                    renderer_model_name = str(trainer.vllm_generation.model.name_or_path)
                    max_model_len = request.inference.engine.get(
                        "max_model_len",
                        request.settings.max_prompt_length + request.settings.max_completion_length,
                    )
                    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
                        raise ValueError("TRL async rollout max_model_len must be an integer")
                    runtime = TrlAsyncCollectionRuntime(
                        trainer=trainer,
                        bridge=bridge,
                        model=request.policy,
                        renderer=request.training.renderer,
                        renderer_model_name=renderer_model_name,
                        max_model_len=max_model_len,
                        execution_config=rollout_execution,
                        episode_timeout=rollout_timeout,
                        startup_timeout=request.inference.startup_timeout_seconds,
                    )
                    trainer._posttrain_async_collection_runtime = runtime  # noqa: SLF001 - backend lifecycle state
                collection_ordinal += 1
                if trainer.state.global_step != trainer._last_loaded_step:  # noqa: SLF001 - TRL's sync marker
                    # TRL syncs before its own generation; resolved collection runs
                    # outside it, so the sampler must receive the applied policy here.
                    trainer.vllm_generation.sync_weights()
                    trainer._last_loaded_step = trainer.state.global_step  # noqa: SLF001
                # Waking colocated vLLM needs the memory the trainer's allocator still
                # caches; TRL's batch generation path releases it the same way. Collect
                # unreachable objects first so no dead tensor keeps a block reserved.
                import gc

                from trl.generation.vllm_generation import empty_cache

                gc.collect()
                empty_cache()
                _report_device_memory(context, optimizer_step)
                outcomes = runtime.collect(
                    selected,
                    collection_id=f"step-{optimizer_step:08d}/collection-{collection_ordinal:06d}",
                    # The exact optimizer version is numeric because it is
                    # also persisted as BehaviorPolicySpan evidence on every
                    # native rollout. Collection identity retains the readable
                    # step label independently.
                    policy_version=str(int(trainer.state.global_step)),
                    observe_trace=observe_trace,
                )
                from trl.generation.vllm_generation import _accumulate_spec_decode_metrics

                _accumulate_spec_decode_metrics(
                    trainer._metrics["train"],  # noqa: SLF001 - TRL's native metric buffer
                    trainer.vllm_generation.last_generation_metrics,
                )
                return outcomes

            if isinstance(request, GDPORequest | CAPORequest) or (
                isinstance(request, GRPORequest | SAMPORequest) and batch.prompt_group_ids
            ):
                from accelerate.utils import gather_object

                from ...reward_admission import admit_rollout_groups

                # Admission is group-atomic: an invalid occurrence excludes its
                # complete prompt group, while healthy complete groups remain
                # trainable. OLMo3 may refill the missing group from its reserved
                # candidate pool; structured RL instead trains the smaller valid
                # population. Retrying is controlled independently by the
                # settings profile and is never required merely to avoid a
                # batch-wide failure.
                active_sampling = bool(getattr(trainer, "active_sampling", False)) and (
                    isinstance(request, SAMPORequest)
                    or (isinstance(request, GRPORequest) and request.settings.algorithm == "olmo3")
                )
                admission = admit_rollout_groups(
                    batch,
                    request.settings,
                    collect,
                    gather_object,
                    max_attempts=1 if active_sampling else None,
                    retain_complete_on_exhaustion=True,
                )
                rollouts = admission.rollouts
                if admission.rejection_reasons:
                    trainer._posttrain_admission_rejections = admission.rejection_reasons  # noqa: SLF001
            else:
                rollouts = collect(batch)
        elapsed = time.perf_counter() - started_at
        if len(rollouts) != len(inputs) and not (
            admission is not None and len(admission.retained_positions) == len(rollouts)
        ):
            raise ValueError("online-RL bridge returned a rollout count that does not match the trainer batch")
        completion_tokens = sum(len(rollout.completion_ids) for rollout in rollouts)
        selected_tokens = sum(sum(rollout.env_mask) for rollout in rollouts)
        truncated_rollouts = sum(rollout.is_truncated for rollout in rollouts)
        update_totals.add_batch(
            optimizer_step,
            sums={
                "train/rl/rollouts_requested": len(inputs),
                "train/rl/rollouts_attempted": len(inputs) if admission is None else admission.attempted_rollouts,
                "train/rl/admission_rounds": 1 if admission is None else admission.rounds,
                "train/rl/admission_rejected_groups": 0 if admission is None else admission.rejected_groups,
                "train/rl/rollouts_completed": len(rollouts),
                "train/rl/rollouts_failed": 0 if admission is None else admission.failed_rollouts,
                "train/rl/rollouts_replaced": 0 if admission is None else admission.attempted_rollouts - len(inputs),
                "train/rl/rollouts_truncated": truncated_rollouts,
                "train/rl/rollouts_unscorable": sum(not math.isfinite(rollout.reward) for rollout in rollouts),
                "train/rl/rollouts_missing": len(inputs) - len(rollouts),
                "train/rl/time/rollout_seconds": elapsed,
                "train/rl/rollout_completion_tokens": completion_tokens,
                "train/rl/rollout_selected_tokens": selected_tokens,
                **episode_ending_counts(endings),
            },
            batch_seconds=elapsed,
        )
        if rollouts and request.settings.mask_truncated_completions and truncated_rollouts == len(rollouts):
            raise RuntimeError(
                f"all {len(rollouts)} rollouts reached a truncation boundary, so this batch has no trainable "
                "tokens while mask_truncated_completions is enabled; increase max_completion_length or correct "
                "the policy's termination behavior before retrying"
            )
        if isinstance(request, GRPORequest | SAMPORequest):
            shaped_rewards = [
                shape_online_reward(
                    request.settings,
                    rollout.reward,
                    len(rollout.completion_ids),
                    is_truncated=rollout.is_truncated,
                )
                for rollout in rollouts
            ]
        else:
            shaped_rewards = [rollout.reward for rollout in rollouts]
        for rollout, algorithm_reward in zip(rollouts, shaped_rewards, strict=True):
            if isinstance(request, GDPORequest | CAPORequest):
                # A native scalar is not GDPO's component vector or CAPO's token
                # reward. Their evidence is projected below after normalization.
                continue
            observed_algorithm_reward = algorithm_reward if math.isfinite(algorithm_reward) else None
            context.trace_fact_update(
                TraceFactUpdateObservation(
                    trace_type="verifiers",
                    external_id=rollout.trace.external_id,
                    facts=TraceFactSet(
                        namespace="posttrain.train.reward",
                        calculator_version=(
                            f"{request.settings.algorithm}-algorithm-reward.v1"
                            if isinstance(request, GRPORequest)
                            else f"{technique(request)}-algorithm-reward.v1"
                        ),
                        measures={"algorithm_reward": observed_algorithm_reward},
                        provenance={
                            "algorithm_reward": (
                                "trainer_reward_shaping" if observed_algorithm_reward is not None else "unsupported"
                            )
                        },
                    ),
                    attributes={"optimizer_step": optimizer_step},
                )
            )
        if retain_native:
            # Complete admitted groups retain native conditioning and reward
            # evidence. The resolved engine owns credit; do not compute legacy
            # trainer advantages or flatten away the original graphs here.
            if trainer.state.global_step != optimizer_step - 1 or any(
                rollout.behavior_policy != BehaviorPolicySpan(optimizer_step - 1, optimizer_step - 1)
                for rollout in rollouts
            ):
                raise InvalidPolicyUpdate("native collection lost its fixed applied behavior policy")
            return tuple(rollouts)
        result = {
            "prompt_ids": [list(rollout.prompt_ids) for rollout in rollouts],
            "completion_ids": [list(rollout.completion_ids) for rollout in rollouts],
            "logprobs": [list(rollout.sampling_logprobs) for rollout in rollouts],
            "env_mask": [list(rollout.env_mask) for rollout in rollouts],
            "rollout_reward": shaped_rewards,
            "task_reward": [rollout.reward for rollout in rollouts],
            "algorithm_reward": shaped_rewards,
            "is_truncated": [rollout.is_truncated for rollout in rollouts],
            "rollout_trace_id": [rollout.trace.external_id for rollout in rollouts],
        }
        if admission is not None and admission.retained_positions != tuple(range(len(inputs))):
            result["retained_input_indices"] = list(admission.retained_positions)
        if isinstance(request, GDPORequest | CAPORequest):
            from accelerate.utils import gather_object

            local = [(rollout.reward_evidence, rollout.env_mask, rollout.is_truncated) for rollout in rollouts]
            gathered = gather_object(local)
            if any(item[0] is None or item[2] for item in gathered):
                raise ValueError("structured RL requires complete evidence and nontruncated responses")
            evidence = [item[0] for item in gathered]
            masks = [item[1] for item in gathered]
            if isinstance(request, GDPORequest):
                computed = compute_gdpo_advantages(
                    evidence,
                    masks,
                    component_names=request.settings.component_names,
                    component_weights=request.settings.component_weights,
                    group_size=request.settings.num_generations,
                    epsilon=request.settings.epsilon,
                )
                result["reward_components"] = [
                    list(rollout.reward_evidence.require_components(request.settings.component_names))
                    for rollout in rollouts
                    if rollout.reward_evidence is not None
                ]
            else:
                computed = compute_capo_advantages(
                    evidence,
                    masks,
                    group_size=request.settings.num_generations,
                    outcome_component=request.settings.outcome_component,
                    outcome_weight=request.settings.outcome_weight,
                    process_weight=request.settings.process_weight,
                    epsilon=request.settings.epsilon,
                )
            by_id = {item.rollout_id: row for item, row in zip(evidence, computed.token_advantages, strict=True)}
            local_advantages = []
            for (item, _, _), rollout in zip(local, rollouts, strict=True):
                if item is None:
                    raise ValueError("local structured evidence was lost after gathering")
                local_advantages.append(list(by_id[item.rollout_id]))
                sampled = [
                    value for value, include in zip(by_id[item.rollout_id], rollout.env_mask, strict=True) if include
                ]
                algorithm_reward = math.fsum(sampled) / len(sampled)
                context.trace_fact_update(
                    TraceFactUpdateObservation(
                        trace_type="verifiers",
                        external_id=rollout.trace.external_id,
                        facts=TraceFactSet(
                            namespace="posttrain.train.reward",
                            calculator_version=request.settings.numerical_profile,
                            measures={"algorithm_reward": algorithm_reward},
                            provenance={"algorithm_reward": f"{technique(request)}_sampled_token_advantage_mean"},
                        ),
                        attributes={"optimizer_step": optimizer_step, "reward_projection": item.projection_id},
                    )
                )
            result["precomputed_advantages"] = local_advantages
        if isinstance(request, SAMPORequest) and not rollouts:
            # Admission kept no complete group; TRL's refill treats the empty batch
            # as NoAdmittedRollouts and draws the next candidate round.
            result["precomputed_advantages"] = []
        elif isinstance(request, SAMPORequest):
            # Hierarchical advantages see the shaped episode reward, so a truncated
            # rollout ranks below an equally scored finished one, as in VORTEX.
            advantages = compute_sampo_advantages(
                request.settings,
                _admitted_example_ids(example_ids, None if admission is None else admission.retained_positions),
                [replace(rollout, reward=reward) for rollout, reward in zip(rollouts, shaped_rewards, strict=True)],
            )
            result["precomputed_advantages"] = [list(values) for values in advantages.token_advantages]
            update_totals.add_means(optimizer_step, _sampo_update_means(advantages, request.settings))
        return result

    return run_rollouts


def _sampo_update_means(advantages: Any, settings: Any) -> dict[str, tuple[float, int]]:
    """(mean, count) per SAMPO credit metric for one admitted population.

    The centred advantage means stay for continuity; the hierarchy evidence is
    what a reader can use (magnitudes, the turn share of credit and coverage).
    The resolved engine reports the same credit evidence per collection.
    """

    return {
        **advantages.credit_evidence(settings.step_advantage_weight),
        **advantages.policy_credit_evidence(),
    }


def _admitted_example_ids(example_ids: tuple[str, ...], retained_positions: Sequence[int] | None) -> tuple[str, ...]:
    """Example identities of the rollouts group admission kept, in rollout order."""

    if retained_positions is None:
        return example_ids
    return tuple(example_ids[position] for position in retained_positions)


def _bridge_reward(rollout_reward: list[float], **_: Any) -> list[float]:
    """Return rewards already computed by the native online-RL environment."""

    return [float(value) for value in rollout_reward]


def _rollout_batch(request: Any, trainer: Any, example_ids: tuple[str, ...], step: int, ordinal: int) -> RolloutBatch:
    if not isinstance(request, GRPORequest | SAMPORequest | GDPORequest | CAPORequest):
        return RolloutBatch(example_ids=example_ids, step=step, model_id=request.policy.id)
    if isinstance(request, GRPORequest | SAMPORequest) and not hasattr(trainer, "accelerator"):
        # Lightweight bridge consumers predating explicit group identities keep
        # their compatibility path. Real TRL trainers always expose Accelerator
        # and therefore use complete-group admission below.
        return RolloutBatch(example_ids=example_ids, step=step, model_id=request.policy.id)
    from accelerate.utils import gather_object

    ranks = gather_object([list(example_ids)])
    rank = int(trainer.accelerator.process_index)
    offset = sum(len(rows) for rows in ranks[:rank])
    all_examples = [example for rows in ranks for example in rows]
    size = request.settings.num_generations
    _validate_group_relative_examples(request.settings, all_examples)
    prefix = f"step/{step}/batch/{ordinal}"
    return RolloutBatch(
        example_ids=example_ids,
        step=step,
        model_id=request.policy.id,
        prompt_group_ids=tuple(f"{prefix}/group/{(offset + i) // size}" for i in range(len(example_ids))),
        rollout_ids=tuple(f"{prefix}/response/{offset + i}" for i in range(len(example_ids))),
    )


def _validate_group_relative_examples(settings: Any, example_ids: Sequence[str]) -> None:
    """Validate one complete batch or one OLMo3 active-sampling refill."""

    size = settings.num_generations
    expected = settings.num_prompts_per_step * size
    # Active sampling is a GRPO/OLMo and SAMPO capability, not part of the
    # structured GDPO/CAPO settings contract. Keep the shared validator capability-safe.
    active_sampling = getattr(settings, "active_sampling", None)
    if active_sampling is not None:
        # A round holds whole prompt groups, at most the oversampled first round.
        largest = active_sampling.largest_round_groups(settings.num_prompts_per_step) * size
        if not example_ids or len(example_ids) > largest or len(example_ids) % size != 0:
            raise ValueError(
                "an active-sampling round must contain one or more complete prompt groups and at most "
                f"{largest} rows ((prompts + oversample) x generations); got {len(example_ids)}"
            )
    elif len(example_ids) != expected:
        raise ValueError("group-relative RL generation must contain the complete logical batch")
    for start in range(0, len(example_ids), size):
        if len(set(example_ids[start : start + size])) != 1:
            raise ValueError("trainer generation order split or mixed prompt occurrences")


def reward_functions(request: Any) -> Any:
    if not isinstance(request, GDPORequest):
        return _bridge_reward

    def component_function(index: int, name: str) -> Any:
        def reward(reward_components: list[list[float]], **_: Any) -> list[float]:
            return [row[index] for row in reward_components]

        reward.__name__ = f"component_{name}"
        return reward

    return [component_function(index, name) for index, name in enumerate(request.settings.component_names)]


__all__ = ["reward_functions", "rollout_function", "technique"]
