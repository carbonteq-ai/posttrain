"""Rollout collection and credit projection for TRL policy optimization."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from typing import Any, Literal, cast

from posttrain.common import RunContext, TraceFactSet, TraceFactUpdateObservation, TraceObservation

from ...online_rl import EnvironmentRollout, RolloutBatch, run_observed_rollouts
from ...profiles import shape_online_reward
from ...requests import CAPORequest, GDPORequest, GRPORequest, SAMPORequest
from ...reward_advantages import compute_capo_advantages, compute_gdpo_advantages
from ...sampo_advantages import compute_sampo_advantages

type PolicyTechnique = Literal["grpo", "dapo", "olmo3", "sampo", "gdpo", "capo"]


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
) -> Any:
    """Translate TRL generation batches into the public environment-rollout bridge contract."""

    from .policy_config import _rollout_execution_config

    rollout_execution = _rollout_execution_config(request)
    rollout_batch_step: int | None = None
    rollout_batch_ordinal = 0
    collection_ordinal = 0

    def run_rollouts(
        prompts: list[Any],
        trainer: Any,
        *,
        inputs: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        nonlocal collection_ordinal, rollout_batch_ordinal, rollout_batch_step
        if inputs is None or len(inputs) != len(prompts):
            raise ValueError("TRL must provide dataset rows aligned with rollout prompts")
        try:
            example_ids = tuple(str(row["example_id"]) for row in inputs)
        except KeyError as error:
            raise ValueError("every online-RL dataset row requires an example_id") from error
        optimizer_step = int(trainer.state.global_step) + 1
        if rollout_batch_step != optimizer_step:
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

        def observe_trace(trace: TraceObservation) -> None:
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
                return runtime.collect(
                    selected,
                    collection_id=f"step-{optimizer_step:08d}/collection-{collection_ordinal:06d}",
                    policy_version=f"optimizer-step-{int(trainer.state.global_step):08d}",
                    observe_trace=observe_trace,
                )

            if isinstance(request, GDPORequest | CAPORequest) or (
                isinstance(request, GRPORequest) and batch.prompt_group_ids
            ):
                from accelerate.utils import gather_object

                from ...reward_admission import admit_rollout_groups

                # OLMo3 owns bounded active sampling above this bridge. A
                # Verifiers failure therefore excludes its complete prompt
                # group from this candidate round; OLMo3 draws another group
                # from its reserved pool. Retrying the same failed occurrence
                # here both wastes the candidate budget and can terminate the
                # entire optimizer step despite healthy surplus groups.
                active_sampling = (
                    isinstance(request, GRPORequest)
                    and request.settings.algorithm == "olmo3"
                    and bool(getattr(trainer, "active_sampling", False))
                )
                admission = admit_rollout_groups(
                    batch,
                    request.settings,
                    collect,
                    gather_object,
                    max_attempts=1 if active_sampling else None,
                    retain_complete_on_exhaustion=isinstance(request, GRPORequest),
                )
                rollouts = admission.rollouts
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
        context.metrics(
            {
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
                "train/rl/rollout_tokens_per_second": completion_tokens / elapsed if elapsed > 0 else 0.0,
                "train/rl/rollout_selected_tokens": selected_tokens,
                "train/rl/rollout_selected_token_fraction": (
                    selected_tokens / completion_tokens if completion_tokens else 0.0
                ),
            },
            step=optimizer_step,
            attributes={
                "rollout_population_scope": "candidate",
                "rollout_batch_ordinal": rollout_batch_ordinal,
            },
        )
        if rollouts and request.settings.mask_truncated_completions and truncated_rollouts == len(rollouts):
            raise RuntimeError(
                f"all {len(rollouts)} rollouts reached a truncation boundary, so this batch has no trainable "
                "tokens while mask_truncated_completions is enabled; increase max_completion_length or correct "
                "the policy's termination behavior before retrying"
            )
        if isinstance(request, GRPORequest):
            shaped_rewards = [
                shape_online_reward(request.settings, rollout.reward, len(rollout.completion_ids))
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
                measures = {
                    f"raw_component/{component.name}": component.value
                    for component in item.components
                    if component.value is not None
                }
                sampled = [
                    value for value, include in zip(by_id[item.rollout_id], rollout.env_mask, strict=True) if include
                ]
                measures["sampled_advantage_mean"] = math.fsum(sampled) / len(sampled)
                context.trace_fact_update(
                    TraceFactUpdateObservation(
                        trace_type="verifiers",
                        external_id=rollout.trace.external_id,
                        facts=TraceFactSet(
                            namespace="posttrain.train.structured_reward",
                            calculator_version=request.settings.numerical_profile,
                            measures=measures,
                        ),
                        attributes={"optimizer_step": optimizer_step, "reward_projection": item.projection_id},
                    )
                )
            result["precomputed_advantages"] = local_advantages
        if isinstance(request, SAMPORequest):
            advantages = compute_sampo_advantages(request.settings, example_ids, rollouts)
            result["precomputed_advantages"] = [list(values) for values in advantages.token_advantages]
            flat_turn_advantages = [value for values in advantages.turn_advantages for value in values]
            flat_group_sizes = [value for values in advantages.anchor_group_sizes for value in values]
            context.metrics(
                {
                    "train/rl/episode_advantage_mean": (
                        sum(advantages.episode_advantages) / len(advantages.episode_advantages)
                    ),
                    "train/rl/turn_advantage_mean": (sum(flat_turn_advantages) / len(flat_turn_advantages)),
                    "train/rl/anchor_group_size_mean": sum(flat_group_sizes) / len(flat_group_sizes),
                    "train/rl/sparse_reward_projection_fraction": (
                        sum(advantages.used_sparse_rewards) / len(advantages.used_sparse_rewards)
                    ),
                },
                step=optimizer_step,
            )
        return result

    return run_rollouts


def _bridge_reward(rollout_reward: list[float], **_: Any) -> list[float]:
    """Return rewards already computed by the native online-RL environment."""

    return [float(value) for value in rollout_reward]


def _rollout_batch(request: Any, trainer: Any, example_ids: tuple[str, ...], step: int, ordinal: int) -> RolloutBatch:
    if not isinstance(request, GRPORequest | GDPORequest | CAPORequest):
        return RolloutBatch(example_ids=example_ids, step=step, model_id=request.policy.id)
    if isinstance(request, GRPORequest) and not hasattr(trainer, "accelerator"):
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
    # Active sampling is a GRPO/OLMo capability, not part of the structured
    # GDPO/CAPO settings contract. Keep the shared validator capability-safe.
    active_refill = getattr(settings, "active_sampling", None) is not None
    if active_refill:
        if not example_ids or len(example_ids) > expected or len(example_ids) % size != 0:
            raise ValueError("OLMo3 active-sampling refill must contain one or more complete prompt groups")
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
