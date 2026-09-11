"""TRL online policy optimization over task-neutral rollout prompts and rewards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from posttrain.common import RunContext
from posttrain.common.cuda import TorchModule, activate_cuda_toolkit

from ...grpo_observations import GRPOObservationFeatures
from ...requests import CAPORequest, GDPORequest, GRPORequest, SAMPORequest
from .common import (
    BackendTrainingResult,
    callback_type,
    checkpoint_callback_type,
    emit_parameter_counts,
    emit_runtime_versions,
    finish_training,
    framework_imports,
    load_tokenizer,
    load_trainable_model,
    preserve_recovery_checkpoint_after_error,
    trainer_lifecycle,
)
from .policy_config import (
    _configure_liger_loss,
    _configure_torch_compile,
    _online_rl_arguments,
    _online_rl_runtime_attributes,
)
from .policy_curriculum import (
    AdaptiveCurriculumRuntime as _AdaptiveCurriculumRuntime,
)
from .policy_curriculum import (
    adaptive_curriculum_trainer_type as _adaptive_curriculum_trainer_type,
)
from .policy_rollouts import (
    reward_functions as _reward_functions,
)
from .policy_rollouts import (
    rollout_function as _rollout_function,
)
from .policy_rollouts import (
    technique as _technique,
)
from .policy_telemetry import (
    ActorUpdateTelemetry as _ActorUpdateTelemetry,
)
from .policy_telemetry import (
    actor_update_callback_type as _actor_update_callback_type,
)
from .policy_telemetry import (
    actor_update_trainer_type as _actor_update_trainer_type,
)
from .policy_telemetry import (
    normalize_live_metrics as _normalize_live_grpo_metrics,
)


def run_grpo(
    context: RunContext,
    request: GRPORequest,
    output_dir: Path,
) -> BackendTrainingResult:
    return _run_online_rl(context, request, output_dir)


def run_sampo(
    context: RunContext,
    request: SAMPORequest,
    output_dir: Path,
) -> BackendTrainingResult:
    return _run_online_rl(context, request, output_dir)


def run_structured_rl(
    context: RunContext,
    request: GDPORequest | CAPORequest,
    output_dir: Path,
) -> BackendTrainingResult:
    return _run_online_rl(context, request, output_dir)


def _run_online_rl(
    context: RunContext,
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest,
    output_dir: Path,
) -> BackendTrainingResult:
    reward_contract = None
    if isinstance(request, GDPORequest | CAPORequest) or (
        isinstance(request, SAMPORequest) and getattr(request.bridge, "reward_projection", None) is not None
    ):
        from ...reward_recovery import reward_contract_digest, validate_reward_recovery

        reward_contract = reward_contract_digest(request)
        if request.resume_from is not None:
            validate_reward_recovery(request.resume_from.path, reward_contract)
    _configure_torch_compile(request.inference.engine)
    if request.inference.backend.split("@", 1)[0] == "vllm":
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is not installed; install posttrain-train[trl-vllm]") from error
        activate_cuda_toolkit(cast(TorchModule, torch))
    try:
        from trl.trainer.grpo_config import GRPOConfig  # pyright: ignore[reportMissingImports]
        from trl.trainer.grpo_trainer import GRPOTrainer  # pyright: ignore[reportMissingImports]
        from trl.trainer.olmo3_grpo_config import Olmo3GRPOConfig  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-train with the trl extra") from error

    imports = framework_imports()
    emit_runtime_versions(context, imports)
    with context.phase("model_loading", {"backend": "trl"}):
        tokenizer = load_tokenizer(request.policy, imports)
        model = load_trainable_model(request.policy, request.training.update, request.settings.loop, imports)
    rows = []
    template_kwargs = request.policy.conversation.reasoning_mode(request.training.renderer.reasoning_mode).kwargs()
    for example in request.bridge.dataset.examples:
        prompt = [{"role": "user", "content": example.prompt}]
        rendered = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **template_kwargs,
        )
        if not isinstance(rendered, list) or any(not isinstance(token_id, int) for token_id in rendered):
            raise TypeError("chat template must return one flat token-id list")
        if len(rendered) > request.settings.max_prompt_length:
            raise ValueError(
                f"rollout example {example.id!r} has {len(rendered)} prompt tokens; "
                f"settings permit {request.settings.max_prompt_length}"
            )
        rows.append({"prompt": prompt, "example_id": example.id, **dict(example.metadata)})
    dataset = imports["Dataset"].from_list(rows)
    emit_parameter_counts(context, model, request.training.update)
    arguments = _online_rl_arguments(request, output_dir, template_kwargs)
    observation_features = (
        GRPOObservationFeatures.from_request(request)
        if isinstance(request, GRPORequest)
        else GRPOObservationFeatures(
            reference_kl_enabled=request.settings.beta > 0,
            decoupled_rollout=request.inference.backend.split("@", 1)[0] == "vllm",
            tool_environment=True,
        )
    )
    technique = _technique(request)
    config_type = (
        Olmo3GRPOConfig if isinstance(request, GRPORequest) and request.settings.algorithm == "olmo3" else GRPOConfig
    )
    context.event("grpo_runtime_resolved", _online_rl_runtime_attributes(request))
    actor_update = _ActorUpdateTelemetry(context)
    trainer_type = _actor_update_trainer_type(GRPOTrainer, actor_update)
    curriculum = None
    if isinstance(request, GRPORequest) and request.settings.adaptive_curriculum is not None:
        curriculum = _AdaptiveCurriculumRuntime(
            context,
            rows,
            request.settings.adaptive_curriculum,
            num_generations=request.settings.num_generations,
            state_dir=output_dir.parent / "adaptive-curriculum",
            resume_checkpoint=request.resume_from.path if request.resume_from is not None else None,
        )
        trainer_type = _adaptive_curriculum_trainer_type(trainer_type, curriculum)
    checkpoint_callback = checkpoint_callback_type(
        context,
        imports,
        model=request.policy,
        technique=technique,
        settings=request.settings,
        update=request.training.update,
        workspace=output_dir.parent,
        reward_contract=reward_contract,
        checkpoint_state_writer=curriculum.checkpoint if curriculum is not None else None,
    )()
    callbacks = [
        callback_type(
            context,
            imports,
            metric_normalizer=lambda step, native: _normalize_live_grpo_metrics(
                step,
                native,
                observation_features,
            ),
        )(),
        _actor_update_callback_type(imports, actor_update)(),
        checkpoint_callback,
    ]

    failure: BaseException | None = None
    try:
        with context.phase("runtime_initialization", {"backend": "trl"}):
            trainer = trainer_type(
                model=model,
                reward_funcs=_reward_functions(request),
                rollout_func=cast(Any, _rollout_function(context, request, tokenizer)),
                args=config_type(**arguments),
                train_dataset=dataset,
                processing_class=tokenizer,
                callbacks=callbacks,
            )
            _configure_liger_loss(trainer, request)
        resume = str(request.resume_from.path) if request.resume_from is not None else None
        with trainer_lifecycle(trainer):
            try:
                train_output = trainer.train(resume_from_checkpoint=resume)
                if actor_update.active:
                    raise RuntimeError("TRL training completed before the active actor update reached an optimizer step")
                if curriculum is not None:
                    curriculum.save_final_state()
                with context.phase("artifact_export", {"backend": "trl"}):
                    return finish_training(
                        context,
                        trainer,
                        train_output,
                        tokenizer,
                        output_dir.parent,
                        technique,
                        request.training.update,
                        imports,
                    )
            except BaseException as error:
                actor_update.fail(error)
                preserve_recovery_checkpoint_after_error(
                    context,
                    trainer,
                    error,
                    technique=technique,
                    model=request.policy,
                    settings=request.settings,
                    update=request.training.update,
                    imports=imports,
                    checkpoint_state_writer=curriculum.checkpoint if curriculum is not None else None,
                )
                raise
    except BaseException as error:
        failure = error
        raise
    finally:
        if curriculum is not None:
            try:
                curriculum.close()
            except BaseException as close_error:
                if failure is None:
                    raise
                failure.add_note(f"failed to close adaptive curriculum state: {close_error!r}")


__all__ = ["run_grpo", "run_sampo"]
