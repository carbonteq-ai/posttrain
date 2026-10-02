"""Worker entrypoint executed by the isolated veRL Python environment."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import site
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ...bindings import _peft_target_modules
from ...online_rl import policy_sampling_from_mapping
from ...precision import training_precision, verl_mixed_precision_overrides, verl_rollout_dtype
from ...rollout_execution import RolloutExecutionConfig, validate_execution_config
from ..retention import finalize_training_outputs
from .contracts import (
    VerlLaunchManifest,
    VerlModelArtifact,
    VerlPayload,
    VerlTrainingSummary,
    VerlWorkerResult,
)
from .metrics import read_verl_metric_records

_METRIC = re.compile(r"'([^']+)':\s*(?:np\.float\d+\()?([-+0-9.eE]+)")
_INLINE_METRIC = re.compile(r"(?<![\w/])([A-Za-z_][\w]*(?:/[A-Za-z0-9_]+)*):(?:np\.(?:float|int)\d+\()?([-+0-9.eE]+)")
_ROLLOUT_EXECUTION_FORK_REVISIONS_BASE = frozenset({"5dbf667c99b29db613d1dfcded1ed90440ef6311"})
# Native veRL names Posttrain selects that upstream veRL v0.9.0 does not register.
# Each maps to the CarbonTeq fork commits that do register it, with the fork
# version at that commit. A clean checkout at any other revision is rejected
# before veRL starts instead of failing at its first actor update.
# ``active_sampling`` names the fork's ``algorithm.active_sampling`` config block
# (round-based refill with TRL's semantics); ``prompt_selector`` its
# ``data.prompt_selector`` extension point (used by the adaptive curriculum).
_OLMO3_OBJECTIVE_NAMES = frozenset({"token_clip", "k3_unclipped"})
_TRL_SETTING_NAMES = frozenset(
    {"trl_sampler_correction", "grpo_scaling", "row_exclusion", "admission_retries", "linear_lr"}
)
_TRL_DAPO_NAMES = frozenset({"candidate_batches"})
_FORK_ONLY_NATIVE_NAMES = (
    _OLMO3_OBJECTIVE_NAMES
    | {"active_sampling", "prompt_selector", "sequence_clip", "sampo_hierarchy"}
    | _TRL_SETTING_NAMES
    | _TRL_DAPO_NAMES
)
# Keep historical capability sets unchanged: older revisions do not implement
# the uncapped, token-local sequence-ratio gradient required by SAMPO.
_SAMPO_TOKEN_CREDIT_NAMES = frozenset({"sampo_token_credit"})
_REQUESTABLE_NATIVE_NAMES = _FORK_ONLY_NATIVE_NAMES | _SAMPO_TOKEN_CREDIT_NAMES
# The version is recorded for release commits only; a development commit shares
# its parent release's version string without its content.
_FORK_NATIVE_NAME_REVISIONS: dict[str, tuple[str | None, frozenset[str]]] = {
    # codex/vortex development commit for the OLMo 3 objective.
    "a4d84ad30b94c11c4de41b3d915eca6399ad2b6a": (None, _OLMO3_OBJECTIVE_NAMES),
    # carbonteq-v0.9.0.post5 release commit and its asset receipt.
    "9fd6e7a31396ba33a29233cc869ab05b0a9e5a80": ("0.9.0.post5", _OLMO3_OBJECTIVE_NAMES),
    "9c10bd1a5931e7f73dfa4b570eb2c8e767d225ca": ("0.9.0.post5", _OLMO3_OBJECTIVE_NAMES),
    # codex/vortex-active-sampling development commit (post5 + active sampling).
    "6c7295cd411c4d3973ddc206e43816560c842336": (None, _OLMO3_OBJECTIVE_NAMES | {"active_sampling"}),
    # codex/vortex-active-sampling: plus the data.prompt_selector extension point.
    "24920b395f8571f8f5be6b9d8469737f2355dcc9": (
        None,
        _OLMO3_OBJECTIVE_NAMES | {"active_sampling", "prompt_selector"},
    ),
    # codex/vortex-active-sampling: plus the sequence_clip loss and SAMPO hierarchy metrics.
    "d344b545eeb60b8fa1de8174fb65924df26757aa": (None, _FORK_ONLY_NATIVE_NAMES - _TRL_SETTING_NAMES),
    "4d37a18bc492f0f4f9c224285603740ef4a2ba54": (None, _FORK_ONLY_NATIVE_NAMES - _TRL_SETTING_NAMES),
    # codex/vortex-active-sampling: plus TRL's correction bounds, GRPO scaling, row exclusion,
    # group admission retries and the linear LR schedule.
    "c55867dcfa6ca0716bccf57b492e2f4b6f7b0717": (None, _FORK_ONLY_NATIVE_NAMES - _TRL_DAPO_NAMES),
    # codex/vortex-active-sampling: TRL-mode GRPO statistics bitwise equal to TRL's nanstd.
    "2607b91d3cccc9d73aae924734b5104bf8cfb590": (None, _FORK_ONLY_NATIVE_NAMES - _TRL_DAPO_NAMES),
    # codex/vortex-active-sampling: plus TRL's candidate-batch DAPO dynamic sampling.
    "ce8e0430018204b03c009b72bfba3b58968696c7": (None, _FORK_ONLY_NATIVE_NAMES),
    "94606019cadacb656f6e9245f4c793023df6ba12": (None, _FORK_ONLY_NATIVE_NAMES),
    "7d850ef5ad39920446690345af052d781325fc79": (None, _FORK_ONLY_NATIVE_NAMES),
    # carbonteq-v0.9.0.post6 release commit and its asset receipt.
    "1badbebd22aee7af5b85185760275f697af3a073": ("0.9.0.post6", _FORK_ONLY_NATIVE_NAMES),
    "1cd7702f6b2f6aadfa149ec261e277552178eb5a": ("0.9.0.post6", _FORK_ONLY_NATIVE_NAMES),
    "100a0a889c31f6c617988dc68e3caa185e9dae38": ("0.9.0.post6", _FORK_ONLY_NATIVE_NAMES),
    # codex/vortex-lora-sync: synced LoRA tensors keep constituent module names.
    "f74e84e49be243b963153a6c88a6d49ccd75e0b5": (None, _FORK_ONLY_NATIVE_NAMES),
    # carbonteq-v0.9.0.post7 release commit and its asset receipt.
    "6069abe14e2b3d27c89815a6502b849f15124e12": ("0.9.0.post7", _FORK_ONLY_NATIVE_NAMES),
    "07ecac23596d7fd6babdfb88e9dc0442dfc65a72": ("0.9.0.post7", _FORK_ONLY_NATIVE_NAMES),
    # carbonteq-v0.9.0.post8 release commit (post7 plus the agent-loop config
    # defaults) and its asset receipt.
    "ef1c37715fa75de5973ae5b3c398383cd7e0093d": ("0.9.0.post8", _FORK_ONLY_NATIVE_NAMES),
    "be582879e2efd45a7206be49010ab6e6fcd868e9": ("0.9.0.post8", _FORK_ONLY_NATIVE_NAMES),
    # Published math/precision candidate; runtime image adoption is separate.
    "d8e472db822f2916ed81a408b8d28192be95e678": (None, _REQUESTABLE_NATIVE_NAMES),
    # Published engine/TaskRunner recipe extension candidate; no release adoption.
    "70baba82c0b2b0a8981089ca62c5ab474efe1816": (None, _REQUESTABLE_NATIVE_NAMES),
    "ef5aac6ff92d5a69f72cfe222f0a409af4220314": (None, _REQUESTABLE_NATIVE_NAMES),
    "076072b92baf336c2e18e9f38bf7434e4a5f3cb7": (None, _REQUESTABLE_NATIVE_NAMES),
    "77fe49a9de909f036aa72d8568cc957d226b1e7c": (None, _REQUESTABLE_NATIVE_NAMES),
    "8f0de2365f1041954b67f74df5a14c7ba0532755": (None, _REQUESTABLE_NATIVE_NAMES),
    "7cf687284ba054e836b8ce34475a3e695b3dd22b": (None, _REQUESTABLE_NATIVE_NAMES),
    # Published scoped-arithmetic candidate; inherits registered objective names.
    # This is source compatibility, not ordinary multi-record qualification.
    "c45392d22df0c1ab2258e9c0675d3a03093094af": (None, _REQUESTABLE_NATIVE_NAMES),
    # carbonteq-v0.9.0.post9 release commit (post8 plus the scoped-arithmetic
    # candidate above) and its asset receipt.
    "8e513f3bf3bfccb4c413846b5eb184e0b42ea9d8": ("0.9.0.post9", _REQUESTABLE_NATIVE_NAMES),
    "6b3ceef7fe00045d3a8909dc74104d372def7162": ("0.9.0.post9", _REQUESTABLE_NATIVE_NAMES),
}
# Every recorded fork commit descends from post2, which added bounded rollout execution.
_ROLLOUT_EXECUTION_FORK_REVISIONS = _ROLLOUT_EXECUTION_FORK_REVISIONS_BASE | frozenset(_FORK_NATIVE_NAME_REVISIONS)
_TOKEN_CLIP_FORK_REVISIONS = frozenset(
    revision for revision, (_, names) in _FORK_NATIVE_NAME_REVISIONS.items() if "token_clip" in names
)


def validate_policy_update_entrypoint(payload: VerlPayload) -> None:
    """Do not silently run a resolved selection through the legacy PPO host."""
    if payload.algorithm.policy_updates is not None:
        from ...update_records import InvalidPolicyUpdate

        raise InvalidPolicyUpdate(
            "resolved veRL job requires the resolved native population host; "
            "legacy main_ppo cannot consume policy_updates"
        )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m posttrain.train.backends.verl.worker MANIFEST.json")
    manifest_path = Path(sys.argv[1]).resolve()
    manifest = VerlLaunchManifest.read(manifest_path)
    resolved = manifest.payload.algorithm.policy_updates is not None
    if resolved:
        from .policy_native import validate_native_selection

        validate_native_selection(manifest)
    else:
        validate_policy_update_entrypoint(manifest.payload)
    _validate_runtime(manifest)
    output_dir = manifest.output_directory
    payload = manifest.payload
    dataset_path = output_dir / "rollouts.parquet"
    agent_config_path = output_dir / "agent-loop.json"
    checkpoint_dir = output_dir / "checkpoints"
    native_log = output_dir / "verl-native.log"
    metrics_file = output_dir / "verl-metrics.jsonl"
    _write_dataset(payload, dataset_path)
    _write_agent_config(payload, agent_config_path)
    _write_curriculum_selector_config(manifest)
    overrides = build_hydra_overrides(manifest, dataset_path, agent_config_path, checkpoint_dir)
    os.environ["VERL_FILE_LOGGER_PATH"] = str(metrics_file.resolve())
    if _uses_turboquant(payload):
        os.environ["VERL_ENABLE_TURBOQUANT_COMPAT"] = "1"
    if payload.rollout.engine.get("batch_invariant") is True:
        # vLLM reads batch invariance only from the environment of the process
        # that starts the engine; the veRL subprocess inherits it.
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    trainer_module = "verl.trainer.main_ppo"
    trainer_arguments = overrides
    if resolved:
        trainer_module = "posttrain.train.backends.verl.policy_native"
        trainer_arguments = [str(manifest_path), *overrides]
    started = time.perf_counter()
    completed = _run_tee(
        [sys.executable, "-m", trainer_module, *trainer_arguments],
        native_log,
    )
    runtime = time.perf_counter() - started
    if completed != 0:
        raise SystemExit(completed)
    if resolved:
        from .policy_checkpoint import latest_driver_checkpoint

        latest = latest_driver_checkpoint(checkpoint_dir)
        if latest is None:
            raise RuntimeError("resolved native job did not retain a complete terminal checkpoint")
    else:
        latest = _latest_checkpoint(checkpoint_dir)
    if payload.algorithm.adaptive_curriculum is not None:
        from .curriculum import final_snapshot_from_checkpoint

        final_snapshot_from_checkpoint(latest, output_dir / CURRICULUM_STATE_DIR)
    model_dir = output_dir / "model"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(latest / "actor"),
            "--target_dir",
            str(model_dir),
        ],
        check=True,
    )
    recovery_checkpoint = latest
    if payload.training.loop.checkpoint_steps == 0:
        shutil.rmtree(checkpoint_dir)
        recovery_checkpoint = None
    update_kind = payload.training.update.kind
    records = read_verl_metric_records(
        metrics_file, loss_scaling=training_precision(payload.training.backend_options) == "fp16"
    )
    metrics = records[-1].data
    observed_step = metrics.get("training/global_step")
    steps = int(observed_step) if isinstance(observed_step, int | float) else records[-1].step
    loss_names = (
        "train/rl/loss",
        "distillation/loss",
        "actor/pg_loss",
        "actor/policy_loss",
    )
    try:
        train_loss = next(metrics[name] for name in loss_names if name in metrics)
    except StopIteration as error:
        raise RuntimeError(f"veRL completed without a recognized training loss metric; see {native_log}") from error
    if isinstance(train_loss, bool) or not isinstance(train_loss, int | float):
        raise TypeError(f"veRL training loss metric must be numeric; see {native_log}")
    retention = finalize_training_outputs(
        workspace=output_dir,
        model_dir=model_dir,
        checkpoint_root=checkpoint_dir,
        recovery_checkpoint=recovery_checkpoint,
        update_kind=update_kind,
        checkpoint_limit=payload.training.loop.checkpoint_limit,
        manifest_path=output_dir / "retention-manifest.json",
    )
    batch_size = payload.training.runtime.global_batch_size or 1
    result = VerlWorkerResult(
        summary=VerlTrainingSummary(
            global_step=steps,
            train_loss=train_loss,
            runtime_seconds=runtime,
            samples_per_second=steps * batch_size / runtime if runtime > 0 else 0.0,
            steps_per_second=steps / runtime if runtime > 0 else 0.0,
        ),
        model_dir=retention.model_dir,
        recovery_checkpoint=retention.recovery_checkpoint,
        metrics_file=metrics_file.resolve(),
        retention_manifest=retention.manifest_path,
    )
    result.write(manifest.result_file)


def verl_response_budget(max_completion_length: int, max_model_len: object) -> int:
    """Tokens veRL reserves for everything after an episode's first prompt.

    ``max_completion_length`` caps each assistant reply, as on TRL (it reaches
    vLLM as the per-turn ``max_tokens``). A multi-turn Verifiers episode's
    response is every turn after the first prompt (replies, tool results and
    template tokens), and TRL trains all of it, bounded only by the rollout
    context. veRL pads responses to a fixed length and rejects longer ones, so
    its budget is the rollout context; a single reply never exceeds it.
    """

    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 1:
        raise ValueError("veRL rollout max_model_len must be a positive integer")
    return max(max_completion_length, max_model_len)


def build_hydra_overrides(
    manifest: VerlLaunchManifest,
    dataset_path: Path,
    agent_config_path: Path,
    checkpoint_dir: Path,
) -> list[str]:
    payload = manifest.payload
    algorithm = payload.algorithm
    training = payload.training
    loop = training.loop
    engine = payload.rollout.engine
    runtime_options = training.runtime
    backend_options = training.backend_options
    model = payload.policy if manifest.operation in {"grpo", "sampo", "gdpo", "capo"} else payload.student
    assert model is not None
    # A PEFT-adapter model trains the adapter on its foundation weights: the actor
    # and rollout load the foundation, the actor attaches the starting adapter, and
    # the first weight sync gives the rollout that adapter. With LoRA the KL
    # reference is the actor with the adapter disabled, i.e. the base model.
    model_path = _model_path(model.base if model.base is not None else model.artifact)
    world_size = training.target.world_size
    nnodes = runtime_options.nodes
    n_gpus_per_node = runtime_options.devices_per_node or world_size // nnodes
    if nnodes < 1 or n_gpus_per_node < 1 or nnodes * n_gpus_per_node != world_size:
        raise ValueError("veRL training nnodes multiplied by n_gpus_per_node must equal target world_size")
    rollout_tp = _positive_int_option(engine.get("tensor_parallel_size"), "tensor_parallel_size", 1)
    kv_cache_dtype = engine.get("kv_cache_dtype")
    rollout_dtype, _source = verl_rollout_dtype(engine)
    # One optimizer step per update over every row (prompt groups x
    # generations), in micro-batches of per_device_batch_size rows per device;
    # the launcher checked that the split is exact.
    actor_mini_batch = algorithm.num_prompts_per_step
    micro_batch = loop.per_device_batch_size
    update = training.update
    resume_from = payload.resume_from
    rollout_load_format = engine.get("load_format", "safetensors" if update.kind == "lora" else "dummy")
    attention_implementation = backend_options.get("attention_implementation")
    parameter_offload = runtime_options.parameter_offload
    optimizer_offload = runtime_options.optimizer_offload
    max_model_len = engine.get("max_model_len", algorithm.max_prompt_length + algorithm.max_completion_length)
    # The behavior policy TRL resolves from the same binding. veRL passes its rollout
    # temperature/top_p/top_k to every agent-loop episode as sampling overrides and
    # scales the actor's and reference's logits by the same temperature, so they must
    # be the binding's, not veRL's defaults (1.0, 1.0, -1).
    behavior = policy_sampling_from_mapping(payload.rollout.sampling, algorithm.max_completion_length)
    response_budget = verl_response_budget(algorithm.max_completion_length, max_model_len)
    overrides = [
        f"algorithm.adv_estimator={algorithm.advantage_estimator}",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={dataset_path}",
        f"data.val_files={dataset_path}",
        f"data.train_batch_size={algorithm.num_prompts_per_step}",
        f"data.max_prompt_length={algorithm.max_prompt_length}",
        f"data.max_response_length={response_budget}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"data.shuffle={str(bool(algorithm.shuffle_prompts)).lower()}",
        # loop.seed drives prompt order, the rollout sampler, the actor's
        # mini-batch order and the FSDP engines, as seed and data_seed do on TRL.
        f"data.seed={loop.seed}",
        f"actor_rollout_ref.rollout.seed={loop.seed}",
        f"actor_rollout_ref.actor.data_loader_seed={loop.seed}",
        f"actor_rollout_ref.actor.fsdp_config.seed={loop.seed}",
        f"actor_rollout_ref.ref.fsdp_config.seed={loop.seed}",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=False",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={str(loop.gradient_checkpointing).lower()}",
        f"actor_rollout_ref.actor.optim.lr={loop.learning_rate}",
        # Transformers "constant" and "constant_with_warmup" are veRL constant with 0 or the
        # selected warmup steps; "linear" is the fork's Transformers-identical linear decay.
        f"actor_rollout_ref.actor.optim.lr_scheduler_type={'linear' if loop.lr_scheduler_type == 'linear' else 'constant'}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={loop.warmup_steps}",
        # TrainingLoop has no weight decay; the TRL backend trains with
        # Transformers' default 0.0, and veRL's own default is 0.01.
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        f"actor_rollout_ref.actor.optim.clip_grad={loop.max_grad_norm}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={actor_mini_batch}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_batch}",
        "actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.use_kl_loss={str((algorithm.beta or 0.0) > 0).lower()}",
        f"actor_rollout_ref.actor.kl_loss_coef={algorithm.beta or 0.0}",
        f"actor_rollout_ref.actor.kl_loss_type={_kl_loss_type(manifest)}",
        "actor_rollout_ref.actor.use_torch_compile=False",
        f"actor_rollout_ref.actor.fsdp_config.offload_policy={str(parameter_offload or optimizer_offload).lower()}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={str(parameter_offload).lower()}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(optimizer_offload).lower()}",
        "actor_rollout_ref.ref.strategy=fsdp2",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro_batch}",
        "actor_rollout_ref.ref.use_torch_compile=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        f"actor_rollout_ref.rollout.dtype={rollout_dtype}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={micro_batch}",
        f"actor_rollout_ref.rollout.prompt_length={algorithm.max_prompt_length}",
        f"actor_rollout_ref.rollout.response_length={response_budget}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={engine.get('gpu_memory_utilization', 0.4)}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={engine.get('max_num_batched_tokens', algorithm.max_prompt_length + algorithm.max_completion_length)}",
        f"actor_rollout_ref.rollout.max_num_seqs={engine.get('max_num_seqs', algorithm.num_generations)}",
        f"actor_rollout_ref.rollout.free_cache_engine={str(bool(engine.get('free_cache_engine', True))).lower()}",
        "+actor_rollout_ref.rollout.enable_sleep_mode="
        f"{str(bool(engine.get('sleep_during_optimization', True))).lower()}",
        f"actor_rollout_ref.rollout.enforce_eager={str(bool(engine.get('enforce_eager', True))).lower()}",
        f"actor_rollout_ref.rollout.load_format={rollout_load_format}",
        f"actor_rollout_ref.rollout.n={algorithm.num_generations}",
        f"actor_rollout_ref.rollout.temperature={behavior.temperature}",
        f"actor_rollout_ref.rollout.top_p={behavior.top_p}",
        # TRL and vLLM treat top_k 0 as disabled; veRL's disabled value is -1.
        f"actor_rollout_ref.rollout.top_k={behavior.top_k if behavior.top_k > 0 else -1}",
        # Prefix caching stays off unless the binding enables it: veRL rollouts
        # with prefix caching across LoRA weight syncs are not yet qualified.
        "actor_rollout_ref.rollout.enable_prefix_caching="
        f"{str(bool(engine.get('enable_prefix_caching', False))).lower()}",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_config_path}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=posttrain_verifiers",
        "reward.custom_reward_function.path=null",
        "trainer.logger=['console','file']",
        "trainer.project_name=posttrain",
        f"trainer.experiment_name={manifest.operation}-{training.binding_id.replace('/', '-')}",
        f"trainer.n_gpus_per_node={n_gpus_per_node}",
        f"trainer.nnodes={nnodes}",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.total_training_steps={loop.max_steps}",
        # veRL saves the terminal model only when save_freq is positive. A
        # framework checkpoint_steps value of zero disables retained recovery
        # state, but still needs one terminal save so the adapter can be merged.
        f"trainer.save_freq={loop.checkpoint_steps or loop.max_steps + 1}",
        f"trainer.max_actor_ckpt_to_keep={loop.checkpoint_limit}",
        f"trainer.max_critic_ckpt_to_keep={loop.checkpoint_limit}",
        f"trainer.resume_mode={'resume_path' if resume_from is not None else 'disable'}",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
    ]
    if manifest.operation in {"grpo", "sampo", "gdpo", "capo"}:
        loss_agg_mode = "token-mean" if algorithm.online_rl_algorithm in {"dapo", "olmo3"} else "seq-mean-token-mean"
        overrides.extend(
            [
                f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
                f"actor_rollout_ref.actor.clip_ratio_low={algorithm.clip_epsilon_low}",
                f"actor_rollout_ref.actor.clip_ratio_high={algorithm.clip_epsilon_high}",
            ]
        )
        if algorithm.online_rl_algorithm == "olmo3":
            overrides.extend(_olmo3_hydra_overrides(manifest))
        elif manifest.operation == "grpo":
            overrides.extend(_trl_grpo_hydra_overrides(manifest))
        if manifest.operation in {"grpo", "sampo"}:
            overrides.extend(_trl_shared_hydra_overrides(manifest))
        if algorithm.active_sampling:
            overrides.extend(_active_sampling_hydra_overrides(manifest))
        if algorithm.adaptive_curriculum is not None:
            overrides.extend(_curriculum_hydra_overrides(manifest))
        if manifest.operation in {"gdpo", "capo"}:
            structured = {
                "reward_contract_digest": algorithm.reward_contract_digest,
                "group_size": algorithm.num_generations,
                "epsilon": algorithm.normalization_epsilon,
                "component_names": algorithm.component_names,
                "component_weights": algorithm.component_weights,
                "outcome_component": algorithm.outcome_component,
                "outcome_weight": algorithm.outcome_weight,
                "process_weight": algorithm.process_weight,
                "max_admission_attempts": algorithm.max_admission_attempts,
            }
            overrides.extend(
                f"+algorithm.structured_rewards.{key}={json.dumps(value)}"
                for key, value in structured.items()
                if value is not None
            )
            overrides.extend(
                [
                    "actor_rollout_ref.actor.policy_loss.loss_mode=token_clip",
                    "algorithm.filter_groups.enable=false",
                    "trainer.v1.sampler.sync_refill_failed_groups=True",
                ]
            )
        if manifest.operation == "sampo":
            if algorithm.reward_contract_digest is not None:
                overrides.append(
                    f"+algorithm.structured_rewards.reward_contract_digest={algorithm.reward_contract_digest}"
                )
            overrides.extend(
                [
                    # A shared sequence-ratio value with token-local credit, matching corrected TRL SAMPO.
                    "actor_rollout_ref.actor.policy_loss.loss_mode=sampo_token_credit",
                    f"algorithm.gamma={algorithm.discount_gamma}",
                    f"algorithm.sampo.discount_gamma={algorithm.discount_gamma}",
                    f"algorithm.sampo.step_advantage_weight={algorithm.step_advantage_weight}",
                    f"algorithm.sampo.advantage_normalization={algorithm.advantage_normalization}",
                ]
            )
        if algorithm.dynamic_sampling:
            overrides.extend(
                [
                    f"data.gen_batch_size={algorithm.num_prompts_per_step}",
                    "algorithm.filter_groups.enable=true",
                    # NaN marks a trajectory TRL excludes from group statistics.
                    "algorithm.filter_groups.metric=group_reward",
                    # TRL's DAPO: whole candidate batches, one decision point per round.
                    "algorithm.filter_groups.candidate_batches=true",
                    f"algorithm.filter_groups.max_num_gen_batches={algorithm.dynamic_sampling_max_candidate_batches}",
                ]
            )
        overrides.extend(_rollout_execution_hydra_overrides(manifest))
    overrides.extend(verl_mixed_precision_overrides(backend_options))
    if resume_from is not None:
        overrides.append(f"trainer.resume_from_path={json.dumps(str(resume_from))}")
    if kv_cache_dtype is not None:
        overrides.append(f"+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype={kv_cache_dtype}")
    if engine.get("kv_cache_memory_bytes") is not None:
        overrides.append(
            f"+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes={engine['kv_cache_memory_bytes']}"
        )
    speculative_config = engine.get("speculative_config")
    if speculative_config is not None:
        if not isinstance(speculative_config, dict):
            raise ValueError("veRL rollout speculative_config must be a mapping")
        method = speculative_config.get("method")
        num_speculative_tokens = speculative_config.get("num_speculative_tokens")
        if method != "mtp":
            raise ValueError("veRL currently supports only native MTP speculative rollout")
        if (
            isinstance(num_speculative_tokens, bool)
            or not isinstance(num_speculative_tokens, int)
            or num_speculative_tokens < 1
        ):
            raise ValueError("veRL MTP num_speculative_tokens must be a positive integer")
        overrides.extend(
            [
                "actor_rollout_ref.model.mtp.enable=true",
                "actor_rollout_ref.model.mtp.enable_train=false",
                "actor_rollout_ref.model.mtp.enable_rollout=true",
                f"actor_rollout_ref.model.mtp.method={method}",
                f"actor_rollout_ref.model.mtp.num_speculative_tokens={num_speculative_tokens}",
                "actor_rollout_ref.rollout.disable_log_stats=false",
            ]
        )
    if "enable_chunked_prefill" in engine:
        overrides.append(
            f"actor_rollout_ref.rollout.enable_chunked_prefill={str(bool(engine['enable_chunked_prefill'])).lower()}"
        )
    if attention_implementation is not None:
        overrides.append(f"+actor_rollout_ref.model.override_config.attn_implementation={attention_implementation}")
    if bool(engine.get("text_only", False)):
        overrides.extend(
            [
                "+actor_rollout_ref.rollout.limit_images=0",
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.skip_mm_profiling=true",
            ]
        )
    if update.kind == "lora":
        overrides.extend(
            [
                f"actor_rollout_ref.model.lora_rank={update.rank}",
                f"actor_rollout_ref.model.lora_alpha={update.alpha}",
                f"actor_rollout_ref.model.target_modules={json.dumps(_peft_target_modules(update.target_modules))}",
            ]
        )
        if model.base is not None:
            overrides.append(
                f"actor_rollout_ref.model.lora_adapter_path={json.dumps(str(_model_path(model.artifact)))}"
            )
    if manifest.operation == "distill":
        teacher = payload.teacher
        teacher_scoring = payload.teacher_scoring
        assert teacher is not None and teacher_scoring is not None
        teacher_engine = teacher_scoring.engine
        teacher_kv_cache_dtype = teacher_engine.get("kv_cache_dtype")
        teacher_dtype = teacher_engine.get(
            "dtype",
            "float16" if str(teacher_kv_cache_dtype).startswith("turboquant_") else "bfloat16",
        )
        teacher_world_size = teacher_scoring.target.world_size
        teacher_nnodes = 1
        teacher_gpus_per_node = teacher_world_size
        teacher_tp = _positive_int_option(teacher_engine.get("tensor_parallel_size"), "tensor_parallel_size", 1)
        teacher_ep = _positive_int_option(teacher_engine.get("expert_parallel_size"), "expert_parallel_size", 1)
        teacher_dp = _positive_int_option(teacher_engine.get("data_parallel_size"), "data_parallel_size", 1)
        per_replica_world_size = teacher_tp * teacher_ep * teacher_dp
        if (
            teacher_nnodes < 1
            or teacher_gpus_per_node < 1
            or teacher_nnodes * teacher_gpus_per_node != teacher_world_size
            or teacher_world_size % per_replica_world_size
        ):
            raise ValueError("veRL teacher topology must exactly partition the teacher target world_size")
        teacher_replicas = teacher_world_size // per_replica_world_size
        teacher_required_context_len = algorithm.max_prompt_length + algorithm.max_completion_length + 1
        overrides.extend(
            [
                "distillation.enabled=True",
                "distillation.enable_resource_pool=False",
                f"distillation.n_gpus_per_node={teacher_gpus_per_node}",
                f"distillation.nnodes={teacher_nnodes}",
                f"distillation.teacher_models.teacher_model.model_path={_model_path(teacher.artifact)}",
                f"distillation.teacher_models.teacher_model.num_replicas={teacher_replicas}",
                "distillation.teacher_models.teacher_model.inference.name=vllm",
                f"distillation.teacher_models.teacher_model.inference.dtype={teacher_dtype}",
                f"distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size={teacher_tp}",
                f"distillation.teacher_models.teacher_model.inference.expert_parallel_size={teacher_ep}",
                f"distillation.teacher_models.teacher_model.inference.data_parallel_size={teacher_dp}",
                "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="
                f"{teacher_engine.get('gpu_memory_utilization', 0.4)}",
                "distillation.teacher_models.teacher_model.inference.max_model_len="
                f"{teacher_engine.get('max_model_len', teacher_required_context_len)}",
                "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="
                f"{teacher_engine.get('max_num_batched_tokens', teacher_required_context_len)}",
                "distillation.teacher_models.teacher_model.inference.max_num_seqs="
                f"{teacher_engine.get('max_num_seqs', algorithm.num_generations)}",
                f"distillation.distillation_loss.loss_mode={algorithm.loss_mode}",
                f"distillation.distillation_loss.use_task_rewards={str(algorithm.use_task_rewards).lower()}",
                f"distillation.distillation_loss.use_policy_gradient={str(algorithm.use_policy_gradient).lower()}",
            ]
        )
        if teacher_kv_cache_dtype is not None:
            overrides.append(
                "+distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.kv_cache_dtype="
                f"{teacher_kv_cache_dtype}"
            )
        if "enable_chunked_prefill" in teacher_engine:
            overrides.append(
                "distillation.teacher_models.teacher_model.inference.enable_chunked_prefill="
                f"{str(bool(teacher_engine['enable_chunked_prefill'])).lower()}"
            )
    overrides.extend(_backend_hydra_overrides(backend_options))
    if payload.algorithm.policy_updates is not None:
        # The normalizer admits complete native groups. Stock PPO admission
        # retries/drop rules must not mutate that population before it sees it.
        overrides = [value for value in overrides if not value.startswith("trainer.v1.sampler.failed_group_attempts=")]
        overrides.extend(
            [
                "trainer.use_v1=True",
                "trainer.v1.trainer_mode=sync",
                "actor_rollout_ref.actor.strategy=fsdp",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.actor.use_dynamic_bsz=False",
                "actor_rollout_ref.actor.use_fused_kernels=False",
                "+actor_rollout_ref.actor.ppo_infer_micro_batch_size_per_gpu=1",
                "trainer.v1.sampler.failed_group_attempts=0",
            ]
        )
        if (
            backend_options.get("resolved_context_layout") == "dense-population"
            and training_precision(backend_options) == "bf16"
        ):
            # Expose the selected native default explicitly for the checked dense
            # profile. Half reductions remain FP32; callers cannot override this
            # selected precision through raw Hydra fields.
            policy = "{param_dtype:bf16,reduce_dtype:fp32,buffer_dtype:fp32}"
            overrides.extend(
                [
                    f"+actor_rollout_ref.actor.fsdp_config.mixed_precision={policy}",
                    f"+actor_rollout_ref.ref.fsdp_config.mixed_precision={policy}",
                ]
            )
    _validate_fork_native_names(manifest, overrides)
    return overrides


def requested_fork_native_names(overrides: list[str]) -> frozenset[str]:
    """Fork-only policy-loss and KL names a Hydra override list asks veRL to use."""

    keys = ("actor_rollout_ref.actor.policy_loss.loss_mode=", "actor_rollout_ref.actor.kl_loss_type=")
    plain = [value.lstrip("+") for value in overrides]
    selected = {value.split("=", 1)[1] for value in plain if value.startswith(keys)}
    if "algorithm.active_sampling.enable=true" in plain:
        selected.add("active_sampling")
    if "algorithm.adv_estimator=sampo" in plain:
        selected.add("sampo_hierarchy")
    if any(value.startswith("data.prompt_selector.class_path=") and not value.endswith("=null") for value in plain):
        selected.add("prompt_selector")
    if any(value.startswith("algorithm.rollout_correction.rollout_is_log_ratio_bound=") for value in plain):
        selected.add("trl_sampler_correction")
    if any(value.startswith("algorithm.grpo_std_") for value in plain):
        selected.add("grpo_scaling")
    if "algorithm.exclude_flagged_rows=true" in plain:
        selected.add("row_exclusion")
    if any(value.startswith("trainer.v1.sampler.failed_group_attempts=") for value in plain):
        selected.add("admission_retries")
    if "actor_rollout_ref.actor.optim.lr_scheduler_type=linear" in plain:
        selected.add("linear_lr")
    if "algorithm.filter_groups.candidate_batches=true" in plain:
        selected.add("candidate_batches")
    return frozenset(selected & _REQUESTABLE_NATIVE_NAMES)


def fork_native_names(revision: str) -> frozenset[str]:
    """Fork-only native names registered at a CarbonTeq veRL commit (empty when unknown)."""

    entry = _FORK_NATIVE_NAME_REVISIONS.get(revision)
    return entry[1] if entry is not None else frozenset()


def _validate_fork_native_names(manifest: VerlLaunchManifest, overrides: list[str]) -> None:
    missing = requested_fork_native_names(overrides) - fork_native_names(manifest.backend_source_revision)
    if not missing:
        return
    # A dirty candidate checkout is identified by its content digest, not by a
    # release; its maintainer owns what it registers.
    if manifest.payload.training.backend_options.get("source_dirty") is True:
        return
    raise ValueError(
        f"selected veRL source revision {manifest.backend_source_revision} does not register "
        f"{', '.join(sorted(missing))}, which the {manifest.operation} objective requires; select CarbonTeq "
        "veRL source that registers these names; legacy post8 lacks sampo_token_credit"
    )


def _kl_loss_type(manifest: VerlLaunchManifest) -> str:
    """veRL's KL estimator: TRL's unclipped k3 wherever the objective must match it."""

    if manifest.operation in {"grpo", "gdpo", "capo", "sampo"}:
        return "k3_unclipped"
    return "low_var_kl"


def _olmo3_hydra_overrides(manifest: VerlLaunchManifest) -> list[str]:
    """The OLMo 3 objective in veRL's native terms, matching TRL's Olmo3GRPOConfig.

    - policy loss ``token_clip``: asymmetric PPO token clipping with no dual clip;
    - ``norm_adv_by_std_in_grpo=false``: advantage = reward - group mean;
    - decoupled rollout correction: per-token weight min(exp(old - rollout), cap),
      old log-probabilities recomputed by the actor (bypass mode off);
    - token-mean aggregation and the unclipped k3 KL are set by the caller.
    """

    algorithm = manifest.payload.algorithm
    if (
        algorithm.normalize_advantage_by_std is not False
        or algorithm.rollout_importance_sampling != "token"
        or algorithm.rollout_importance_sampling_cap is None
    ):
        raise ValueError("the OLMo 3 veRL manifest is missing its advantage or sampler-correction settings")
    return [
        "actor_rollout_ref.actor.policy_loss.loss_mode=token_clip",
        "algorithm.norm_adv_by_std_in_grpo=false",
    ]


def _trl_grpo_hydra_overrides(manifest: VerlLaunchManifest) -> list[str]:
    """TRL's GRPO/DAPO objective: plain token clipping and TRL's advantage scaling (epsilon 1e-4)."""

    algorithm = manifest.payload.algorithm
    overrides = ["actor_rollout_ref.actor.policy_loss.loss_mode=token_clip"]
    if algorithm.normalize_advantage_by_std is False:
        overrides.append("algorithm.norm_adv_by_std_in_grpo=false")
    else:
        overrides.extend(
            [
                "algorithm.norm_adv_by_std_in_grpo=true",
                "algorithm.grpo_std_epsilon=0.0001",
                f"algorithm.grpo_std_scope={algorithm.advantage_std_scope or 'group'}",
            ]
        )
    return overrides


def _trl_shared_hydra_overrides(manifest: VerlLaunchManifest) -> list[str]:
    """Sampler correction, truncated-completion exclusion and group admission as TRL applies them."""

    algorithm = manifest.payload.algorithm
    overrides = _rollout_correction_hydra_overrides(manifest)
    if algorithm.mask_truncated_completions:
        overrides.append("algorithm.exclude_flagged_rows=true")
    if not algorithm.active_sampling:
        # TRL retries a failed group with its prompt, then drops it. Under active sampling it
        # makes one attempt and refills the missing group, which the active buffer already does.
        overrides.append(f"trainer.v1.sampler.failed_group_attempts={algorithm.max_admission_attempts or 1}")
    return overrides


def _rollout_correction_hydra_overrides(manifest: VerlLaunchManifest) -> list[str]:
    """TRL's vLLM sampler correction: truncate to [min, cap] or mask outside it, from exact ratios."""

    algorithm = manifest.payload.algorithm
    if algorithm.rollout_importance_sampling is None or algorithm.rollout_importance_sampling_mode is None:
        raise ValueError("the veRL manifest selects no sampler correction")
    cap = (
        "inf" if algorithm.rollout_importance_sampling_cap is None else repr(algorithm.rollout_importance_sampling_cap)
    )
    if algorithm.rollout_importance_sampling_mode == "truncate":
        threshold = cap
        minimum = (
            "null"
            if algorithm.rollout_importance_sampling_min is None
            else repr(algorithm.rollout_importance_sampling_min)
        )
    else:
        # IcePop keeps lower <= w <= upper. Weights are positive, so a vanishing lower bound is
        # TRL's missing minimum (a ratio that underflows to 0 is 0 either way).
        lower = (
            "1e-300"
            if algorithm.rollout_importance_sampling_min is None
            else repr(algorithm.rollout_importance_sampling_min)
        )
        threshold = f"'{lower}_{cap}'"
        minimum = "null"
    return [
        f"algorithm.rollout_correction.rollout_is={algorithm.rollout_importance_sampling}",
        f"algorithm.rollout_correction.rollout_is_threshold={threshold}",
        f"algorithm.rollout_correction.rollout_is_clip_min={minimum}",
        "algorithm.rollout_correction.rollout_is_log_ratio_bound=null",
        "algorithm.rollout_correction.rollout_is_batch_normalize=false",
        "algorithm.rollout_correction.rollout_rs=null",
        "algorithm.rollout_correction.bypass_mode=false",
    ]


CURRICULUM_SELECTOR_CONFIG = "curriculum-selector.json"
CURRICULUM_STATE_DIR = "adaptive-curriculum"
CURRICULUM_CHECKPOINT_VIEWS_DIR = "curriculum-checkpoints"


def _curriculum_hydra_overrides(manifest: VerlLaunchManifest) -> list[str]:
    """Run Posttrain's adaptive curriculum as the fork's prompt selector."""

    config_path = manifest.output_directory / CURRICULUM_SELECTOR_CONFIG
    return [
        "data.prompt_selector.class_path=posttrain.train.backends.verl.curriculum.PosttrainCurriculumSelector",
        f"+data.prompt_selector.kwargs.config_path={json.dumps(str(config_path))}",
        "data.prompt_selector.metric=seq_reward",
    ]


def _write_curriculum_selector_config(manifest: VerlLaunchManifest) -> None:
    from .curriculum import CURRICULUM_JOURNAL_NAME, SelectorConfig

    payload = manifest.payload
    settings = payload.algorithm.adaptive_curriculum
    if settings is None:
        return
    output = manifest.output_directory
    SelectorConfig(
        settings=dict(settings),
        num_generations=payload.algorithm.num_generations,
        state_dir=output / CURRICULUM_STATE_DIR,
        journal_path=output / CURRICULUM_JOURNAL_NAME,
        warm_start_state_dir=payload.curriculum_from,
        checkpoint_views_dir=output / CURRICULUM_CHECKPOINT_VIEWS_DIR,
    ).write(output / CURRICULUM_SELECTOR_CONFIG)


def _active_sampling_hydra_overrides(manifest: VerlLaunchManifest) -> list[str]:
    """Round-based active sampling with TRL post11's semantics (fork ``algorithm.active_sampling``).

    The groups' spread is measured on ``group_reward``, the shaped reward the agent
    loop reports (truncation penalty included, NaN only for trajectories TRL
    excludes), with TRL's zero epsilon.
    """

    algorithm = manifest.payload.algorithm
    assert algorithm.active_sampling_max_candidate_batches is not None
    overrides = [
        "algorithm.active_sampling.enable=true",
        f"algorithm.active_sampling.max_candidate_batches={algorithm.active_sampling_max_candidate_batches}",
        f"algorithm.active_sampling.oversample={algorithm.active_sampling_oversample or 0}",
        f"algorithm.active_sampling.oversample_refill={algorithm.active_sampling_oversample_refill or 0}",
        "algorithm.active_sampling.reward_std_epsilon=0.0",
        "algorithm.active_sampling.metric=group_reward",
    ]
    if getattr(algorithm, "adaptive_curriculum", None) is None:
        # Without a selector, native group observation only feeds collection
        # evidence; observe the same value the spread check uses.
        overrides.append("data.prompt_selector.metric=group_reward")
    return overrides


def _uses_turboquant(payload: VerlPayload) -> bool:
    rollout_dtype = payload.rollout.engine.get("kv_cache_dtype", "")
    teacher_dtype = payload.teacher_scoring.engine.get("kv_cache_dtype", "") if payload.teacher_scoring else ""
    return str(rollout_dtype).startswith("turboquant_") or str(teacher_dtype).startswith("turboquant_")


def _positive_int_option(value: object, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"veRL engine option {name} must be a positive integer")
    return value


def _rollout_execution_hydra_overrides(manifest: VerlLaunchManifest) -> list[str]:
    """Translate the framework worker topology into enforceable native veRL settings."""
    payload = manifest.payload
    raw = payload.training.backend_options.get("rollout_execution")
    if raw is None:
        return []
    if manifest.backend_source_revision not in _ROLLOUT_EXECUTION_FORK_REVISIONS:
        raise ValueError(
            "selected veRL source revision does not support bounded rollout_execution; "
            "select a qualified CarbonTeq rollout-execution fork revision"
        )
    if not isinstance(raw, dict):
        raise ValueError("veRL backend_options.rollout_execution must be a mapping")
    expected = {"env_workers", "episodes_per_worker", "worker_native_threads"}
    unknown = set(raw).difference(expected)
    missing = expected.difference(raw)
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {', '.join(sorted(unknown))}")
        raise ValueError(f"invalid veRL rollout_execution mapping: {'; '.join(details)}")
    values = {}
    for key in expected:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"veRL rollout_execution.{key} must be an integer")
        values[key] = value
    execution = RolloutExecutionConfig(**values)
    global_limit = payload.environment.max_concurrent
    if global_limit is None:
        raise ValueError("veRL rollout_execution requires the environment bridge to declare max_concurrent")
    validate_execution_config(execution, global_limit=global_limit)
    return [
        f"actor_rollout_ref.rollout.agent.num_workers={execution.env_workers}",
        f"actor_rollout_ref.rollout.agent.num_cpus_per_worker={execution.worker_native_threads}",
        f"actor_rollout_ref.rollout.agent.max_concurrent_episodes={global_limit}",
        f"actor_rollout_ref.rollout.agent.max_concurrent_episodes_per_worker={execution.episodes_per_worker}",
        # GRPO, DAPO, OLMo 3 and SAMPO follow TRL's group admission instead of refilling failures.
        *(["trainer.v1.sampler.refill_all_failed_groups=True"] if manifest.operation in {"gdpo", "capo"} else []),
    ]


def _backend_hydra_overrides(options: dict[str, Any]) -> list[str]:
    raw = options.get("hydra_overrides", [])
    if not isinstance(raw, list) or any(not isinstance(value, str) or not value.strip() for value in raw):
        raise ValueError("veRL backend_options.hydra_overrides must be a list of non-empty strings")
    protected = (
        "data.train_files=",
        "data.val_files=",
        "actor_rollout_ref.model.path=",
        "actor_rollout_ref.rollout.dtype=",
        "actor_rollout_ref.actor.fsdp_config.mixed_precision",
        "actor_rollout_ref.ref.fsdp_config.mixed_precision",
        "actor_rollout_ref.actor.loss_agg_mode=",
        "actor_rollout_ref.actor.clip_ratio_low=",
        "actor_rollout_ref.actor.clip_ratio_high=",
        "actor_rollout_ref.actor.policy_loss.loss_mode=",
        "actor_rollout_ref.actor.use_kl_loss=",
        "actor_rollout_ref.actor.kl_loss_type=",
        "actor_rollout_ref.actor.kl_loss_coef=",
        "algorithm.use_kl_in_reward=",
        "algorithm.norm_adv_by_std_in_grpo=",
        "algorithm.rollout_correction",
        "algorithm=",
        "algorithm.structured_rewards=",
        "algorithm.adv_estimator=",
        "algorithm.sampo.",
        "algorithm.structured_rewards.",
        "data.gen_batch_size=",
        "algorithm.filter_groups.",
        "algorithm.active_sampling",
        "algorithm.grpo_std_",
        "algorithm.exclude_flagged_rows",
        "trainer.v1.sampler.failed_group_attempts",
        "actor_rollout_ref.actor.optim.lr_scheduler_type",
        "data.prompt_selector",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path=",
        "actor_rollout_ref.rollout.agent.num_workers=",
        "actor_rollout_ref.rollout.agent.num_cpus_per_worker=",
        "actor_rollout_ref.rollout.agent.max_concurrent_episodes=",
        "actor_rollout_ref.rollout.agent.max_concurrent_episodes_per_worker=",
        "trainer.v1.sampler.refill_all_failed_groups=",
        "trainer.default_local_dir=",
        "trainer.save_freq=",
        "trainer.max_actor_ckpt_to_keep=",
        "trainer.max_critic_ckpt_to_keep=",
        "trainer.resume_mode=",
        "trainer.resume_from_path=",
    )
    # Hydra's +/++/~ forms must not bypass ownership of selected contracts.
    if any(value.lstrip("+~").startswith(protected) for value in raw):
        raise ValueError(
            "veRL backend overrides cannot replace selected data, model, algorithm, checkpoint, or artifact policy"
        )
    return list(raw)


def _write_dataset(payload: VerlPayload, path: Path) -> None:
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError("the isolated veRL environment must include datasets") from error
    model = payload.policy or payload.student
    assert model is not None
    rows = [
        {
            "prompt": [{"role": "user", "content": example.prompt}],
            "agent_name": "posttrain_verifiers",
            "example_id": example.id,
            "model_id": model.id,
            **example.metadata,
        }
        for example in payload.environment.examples
    ]
    if payload.algorithm.policy_updates is not None:
        identities = [row["example_id"] for row in rows]
        if any(not isinstance(identity, str) or not identity for identity in identities):
            raise ValueError("resolved collection inventory requires nonempty task identities")
        if len(set(identities)) != len(identities):
            raise ValueError("resolved collection inventory contains duplicate tasks")
        if len(rows) < payload.algorithm.num_prompts_per_step:
            raise ValueError("resolved collection inventory cannot fill distinct complete groups")
    # veRL's v1 trainer drops incomplete prompt batches and derives
    # steps_per_epoch from dataset_size // train_batch_size even when an exact
    # total_training_steps is supplied. Small qualification datasets may
    # intentionally contain one reusable task, so cycle them deterministically
    # to make at least one complete prompt batch.
    if len(rows) < payload.algorithm.num_prompts_per_step and payload.algorithm.adaptive_curriculum is not None:
        raise ValueError("adaptive curriculum needs at least one distinct task per prompt group of an update")
    if len(rows) < payload.algorithm.num_prompts_per_step:
        rows = [dict(rows[index % len(rows)]) for index in range(payload.algorithm.num_prompts_per_step)]
    Dataset.from_list(rows).to_parquet(str(path))


def _write_agent_config(payload: VerlPayload, path: Path) -> None:
    renderer = payload.training.renderer
    algorithm = payload.algorithm
    config = [
        {
            "name": "posttrain_verifiers",
            "_target_": "posttrain.train.backends.verl.agent_loop.PosttrainVerifiersAgentLoop",
            "bridge_snapshot": str(payload.environment.bridge_snapshot),
            "renderer": renderer.model_dump(mode="json"),
            "mask_truncated_completions": algorithm.mask_truncated_completions,
            "max_completion_tokens": algorithm.max_completion_length,
            "overlong_buffer_tokens": algorithm.overlong_buffer_tokens,
            "overlong_penalty_factor": algorithm.overlong_penalty_factor,
            "truncation_penalty": algorithm.truncation_penalty,
            "emit_sampo_metadata": algorithm.advantage_estimator == "sampo",
            "retain_policy_update_evidence": algorithm.policy_updates is not None,
            "structured_algorithm": (
                algorithm.advantage_estimator if algorithm.advantage_estimator in {"gdpo", "capo"} else None
            ),
            "reward_component_names": (
                list(algorithm.component_names or ())
                if algorithm.advantage_estimator == "gdpo"
                else [algorithm.outcome_component]
                if algorithm.advantage_estimator == "capo"
                else None
            ),
        }
    ]
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _model_path(artifact: VerlModelArtifact) -> str:
    if artifact.kind == "hub":
        try:
            from huggingface_hub import constants, snapshot_download
            from huggingface_hub.file_download import repo_folder_name
        except ImportError as error:
            raise RuntimeError("the isolated veRL environment must include huggingface-hub") from error
        revision = artifact.revision
        if constants.HF_HUB_OFFLINE and re.fullmatch(r"[0-9a-f]{40}", revision):
            # Offline snapshot_download requires every repository file, including
            # unused ones such as README.md. A pinned commit's local snapshot is
            # immutable, so use it directly when the model files are present.
            local = (
                Path(constants.HF_HUB_CACHE)
                / repo_folder_name(repo_id=artifact.repo_id, repo_type="model")
                / "snapshots"
                / revision
            )
            if (local / "config.json").is_file():
                return str(local)
        return snapshot_download(repo_id=artifact.repo_id, revision=artifact.revision)
    return str(artifact.path)


def _validate_runtime(manifest: VerlLaunchManifest) -> None:
    projection_value = os.environ.get("POSTTRAIN_VERL_PYTHONPATH")
    if projection_value is not None:
        projection = Path(projection_value)
        if (
            projection != Path("/opt/posttrain-verl/projection")
            or os.environ.get("PYTHONPATH") != projection_value
            or site.ENABLE_USER_SITE
        ):
            raise RuntimeError("veRL capsule worker must use only its packaged projection")
        modules = tuple(
            importlib.import_module(name)
            for name in (
                "posttrain.common",
                "posttrain.data",
                "posttrain.train",
                "posttrain.train.backends.verl.worker",
            )
        )
        root = projection.resolve()
        for module in modules:
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError(f"veRL worker module escaped packaged projection: {module.__name__}")
    try:
        from importlib.metadata import version

        installed = version("verl")
    except Exception as error:
        raise RuntimeError("the selected isolated interpreter does not contain veRL") from error
    if not installed:
        raise RuntimeError("could not resolve the installed veRL version")
    worktree = manifest.working_directory
    head, dirty, dirty_digest = _worktree_source_state(worktree)
    if head != manifest.backend_source_revision:
        raise RuntimeError(
            f"veRL worktree is at {head}, expected immutable revision {manifest.backend_source_revision}"
        )
    backend_options = manifest.payload.training.backend_options
    expected_dirty = backend_options.get("source_dirty")
    expected_digest = backend_options.get("source_dirty_digest")
    if expected_dirty is not None:
        if dirty is not expected_dirty:
            raise RuntimeError(f"veRL worktree dirty state is {dirty}, expected {expected_dirty}")
        if expected_digest is not None and dirty_digest != expected_digest:
            raise RuntimeError("veRL worktree content changed after the training selection was resolved")


_SOURCE_REVISION_MARKER = ".posttrain-source-revision"


def _worktree_source_state(worktree: Path) -> tuple[str, bool, str | None]:
    """The veRL source revision, whether it differs from that revision, and a digest of the difference.

    The veRL job kind ships an immutable source snapshot: it removes the Git
    metadata and records the revision in ``.posttrain-source-revision``, exactly
    as ``posttrain-runtime`` verifies it before starting the worker; that
    snapshot is clean by construction. A development checkout is read with Git.
    """

    marker = worktree / _SOURCE_REVISION_MARKER
    if marker.is_file():
        if (worktree / ".git").exists():
            raise RuntimeError("veRL immutable source snapshot unexpectedly retains Git metadata")
        return marker.read_text(encoding="utf-8").strip(), False, None
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty, digest = _git_source_state(worktree)
    return head, dirty, digest


def _git_source_state(worktree: Path) -> tuple[bool, str | None]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not status.strip():
        return False, None
    digest = hashlib.sha256()
    digest.update(
        subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
            cwd=worktree,
            check=True,
            capture_output=True,
        ).stdout
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=worktree,
        check=True,
        capture_output=True,
    ).stdout
    for encoded in sorted(item for item in untracked.split(b"\0") if item):
        digest.update(b"\0untracked\0")
        digest.update(encoded)
        file_path = worktree / encoded.decode("utf-8", errors="surrogateescape")
        if file_path.is_file():
            digest.update(file_path.read_bytes())
    return True, digest.hexdigest()


def _run_tee(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        return process.wait()


def _latest_checkpoint(root: Path) -> Path:
    checkpoints = sorted(
        root.glob("global_step_*"),
        key=lambda path: int(path.name.removeprefix("global_step_")),
    )
    if not checkpoints:
        raise RuntimeError(f"veRL completed without a recovery checkpoint under {root}")
    return checkpoints[-1]


def _last_metrics(log_path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        for name, value in (*_METRIC.findall(line), *_INLINE_METRIC.findall(line)):
            try:
                values[name] = float(value)
            except ValueError:
                continue
    return values


if __name__ == "__main__":
    main()
