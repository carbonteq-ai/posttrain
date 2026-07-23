"""Worker entrypoint executed by the isolated veRL Python environment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .metrics import read_verl_metric_records

_METRIC = re.compile(r"'([^']+)':\s*(?:np\.float\d+\()?([-+0-9.eE]+)")
_INLINE_METRIC = re.compile(r"(?<![\w/])([A-Za-z_][\w]*(?:/[A-Za-z0-9_]+)*):(?:np\.(?:float|int)\d+\()?([-+0-9.eE]+)")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m posttrain.train.backends.verl.worker MANIFEST.json")
    manifest_path = Path(sys.argv[1]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_runtime(manifest)
    output_dir = Path(manifest["output_directory"])
    payload = manifest["payload"]
    dataset_path = output_dir / "rollouts.parquet"
    agent_config_path = output_dir / "agent-loop.json"
    checkpoint_dir = output_dir / "checkpoints"
    native_log = output_dir / "verl-native.log"
    metrics_file = output_dir / "verl-metrics.jsonl"
    _write_dataset(payload, dataset_path)
    _write_agent_config(payload, agent_config_path)
    overrides = build_hydra_overrides(manifest, dataset_path, agent_config_path, checkpoint_dir)
    os.environ["VERL_FILE_LOGGER_PATH"] = str(metrics_file.resolve())
    if _uses_turboquant(payload):
        os.environ["VERL_ENABLE_TURBOQUANT_COMPAT"] = "1"
    started = time.perf_counter()
    completed = _run_tee(
        [sys.executable, "-m", "verl.trainer.main_ppo", *overrides],
        native_log,
    )
    runtime = time.perf_counter() - started
    if completed != 0:
        raise SystemExit(completed)
    latest = _latest_checkpoint(checkpoint_dir)
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
    update_kind = payload["training"]["update"]["kind"]
    materialized_model = model_dir / "lora_adapter" if update_kind == "lora" else model_dir
    records = read_verl_metric_records(metrics_file)
    metrics = records[-1].data
    observed_step = metrics.get("training/global_step")
    steps = int(observed_step) if isinstance(observed_step, int | float) else records[-1].step
    loss_names = (
        "distillation/loss",
        "actor/pg_loss",
        "actor/policy_loss",
    )
    try:
        train_loss = next(metrics[name] for name in loss_names if name in metrics)
    except StopIteration as error:
        raise RuntimeError(f"veRL completed without a recognized training loss metric; see {native_log}") from error
    batch_size = int(payload["training"]["runtime"].get("global_batch_size", 1))
    result = {
        "schema_version": 1,
        "summary": {
            "global_step": steps,
            "train_loss": train_loss,
            "runtime_seconds": runtime,
            "samples_per_second": steps * batch_size / runtime if runtime > 0 else 0.0,
            "steps_per_second": steps / runtime if runtime > 0 else 0.0,
        },
        "model_dir": str(materialized_model.resolve()),
        "recovery_checkpoint": str(latest.resolve()),
        "metrics_file": str(metrics_file.resolve()),
    }
    Path(manifest["result_file"]).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_hydra_overrides(
    manifest: dict[str, Any],
    dataset_path: Path,
    agent_config_path: Path,
    checkpoint_dir: Path,
) -> list[str]:
    payload = manifest["payload"]
    algorithm = payload["algorithm"]
    training = payload["training"]
    loop = training["loop"]
    rollout = payload["rollout"]
    engine = rollout["engine"]
    runtime_options = training["runtime"]
    backend_options = training.get("backend_options", {})
    model = payload["policy"] if manifest["operation"] == "grpo" else payload["student"]
    model_path = _model_path(model)
    world_size = int(training["target"].get("world_size", 1))
    nnodes = int(runtime_options.get("nodes", runtime_options.get("nnodes", 1)))
    n_gpus_per_node = int(
        runtime_options.get("devices_per_node", runtime_options.get("n_gpus_per_node", world_size // nnodes))
    )
    if nnodes < 1 or n_gpus_per_node < 1 or nnodes * n_gpus_per_node != world_size:
        raise ValueError("veRL training nnodes multiplied by n_gpus_per_node must equal target world_size")
    rollout_tp = int(engine.get("tensor_parallel_size", 1))
    kv_cache_dtype = engine.get("kv_cache_dtype")
    rollout_dtype = engine.get(
        "dtype",
        "float16" if str(kv_cache_dtype).startswith("turboquant_") else "bfloat16",
    )
    actor_mini_batch = int(algorithm["num_prompts_per_step"])
    micro_batch = 1
    update = training["update"]
    rollout_load_format = engine.get("load_format", "safetensors" if update["kind"] == "lora" else "dummy")
    attention_implementation = backend_options.get("attention_implementation")
    parameter_offload = bool(
        runtime_options.get("parameter_offload", runtime_options.get("actor_param_offload", False))
    )
    optimizer_offload = bool(
        runtime_options.get("optimizer_offload", runtime_options.get("actor_optimizer_offload", False))
    )
    overrides = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={dataset_path}",
        f"data.val_files={dataset_path}",
        f"data.train_batch_size={algorithm['num_prompts_per_step']}",
        f"data.max_prompt_length={algorithm['max_prompt_length']}",
        f"data.max_response_length={algorithm['max_completion_length']}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.shuffle=False",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=False",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={str(loop['gradient_checkpointing']).lower()}",
        f"actor_rollout_ref.actor.optim.lr={loop['learning_rate']}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={loop['warmup_steps']}",
        f"actor_rollout_ref.actor.optim.clip_grad={loop['max_grad_norm']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={actor_mini_batch}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_batch}",
        "actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.use_kl_loss={str(float(algorithm.get('beta', 0.0)) > 0).lower()}",
        f"actor_rollout_ref.actor.kl_loss_coef={float(algorithm.get('beta', 0.0))}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
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
        f"actor_rollout_ref.rollout.prompt_length={algorithm['max_prompt_length']}",
        f"actor_rollout_ref.rollout.response_length={algorithm['max_completion_length']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={engine.get('gpu_memory_utilization', 0.4)}",
        f"actor_rollout_ref.rollout.max_model_len={engine.get('max_model_len', algorithm['max_prompt_length'] + algorithm['max_completion_length'])}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={engine.get('max_num_batched_tokens', algorithm['max_prompt_length'] + algorithm['max_completion_length'])}",
        f"actor_rollout_ref.rollout.max_num_seqs={engine.get('max_num_seqs', algorithm['num_generations'])}",
        f"actor_rollout_ref.rollout.free_cache_engine={str(bool(engine.get('free_cache_engine', True))).lower()}",
        "+actor_rollout_ref.rollout.enable_sleep_mode="
        f"{str(bool(engine.get('sleep_during_optimization', True))).lower()}",
        f"actor_rollout_ref.rollout.enforce_eager={str(bool(engine.get('enforce_eager', True))).lower()}",
        f"actor_rollout_ref.rollout.load_format={rollout_load_format}",
        f"actor_rollout_ref.rollout.n={algorithm['num_generations']}",
        "actor_rollout_ref.rollout.enable_prefix_caching=False",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_config_path}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=posttrain_verifiers",
        "reward.custom_reward_function.path=null",
        "trainer.logger=['console','file']",
        "trainer.project_name=posttrain",
        f"trainer.experiment_name={manifest['operation']}-{training['binding_id'].replace('/', '-')}",
        f"trainer.n_gpus_per_node={n_gpus_per_node}",
        f"trainer.nnodes={nnodes}",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.total_training_steps={loop['max_steps']}",
        f"trainer.save_freq={loop['checkpoint_steps']}",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
    ]
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
    if update["kind"] == "lora":
        overrides.extend(
            [
                f"actor_rollout_ref.model.lora_rank={update['rank']}",
                f"actor_rollout_ref.model.lora_alpha={update['alpha']}",
                f"actor_rollout_ref.model.target_modules={json.dumps(update['target_modules'])}",
            ]
        )
    if manifest["operation"] == "distill":
        teacher = payload["teacher"]
        teacher_engine = payload["teacher_scoring"]["engine"]
        teacher_kv_cache_dtype = teacher_engine.get("kv_cache_dtype")
        teacher_dtype = teacher_engine.get(
            "dtype",
            "float16" if str(teacher_kv_cache_dtype).startswith("turboquant_") else "bfloat16",
        )
        teacher_world_size = int(payload["teacher_scoring"]["target"].get("world_size", 1))
        teacher_nnodes = int(runtime_options.get("teacher_nodes", runtime_options.get("teacher_nnodes", 1)))
        teacher_gpus_per_node = int(
            runtime_options.get(
                "teacher_devices_per_node",
                runtime_options.get("teacher_n_gpus_per_node", teacher_world_size // teacher_nnodes),
            )
        )
        teacher_tp = int(teacher_engine.get("tensor_parallel_size", 1))
        teacher_ep = int(teacher_engine.get("expert_parallel_size", 1))
        teacher_dp = int(teacher_engine.get("data_parallel_size", 1))
        per_replica_world_size = teacher_tp * teacher_ep * teacher_dp
        if (
            teacher_nnodes < 1
            or teacher_gpus_per_node < 1
            or teacher_nnodes * teacher_gpus_per_node != teacher_world_size
            or teacher_world_size % per_replica_world_size
        ):
            raise ValueError("veRL teacher topology must exactly partition the teacher target world_size")
        teacher_replicas = teacher_world_size // per_replica_world_size
        overrides.extend(
            [
                "distillation.enabled=True",
                f"distillation.n_gpus_per_node={teacher_gpus_per_node}",
                f"distillation.nnodes={teacher_nnodes}",
                f"distillation.teacher_models.teacher_model.model_path={_model_path(teacher)}",
                f"distillation.teacher_models.teacher_model.num_replicas={teacher_replicas}",
                "distillation.teacher_models.teacher_model.inference.name=vllm",
                f"distillation.teacher_models.teacher_model.inference.dtype={teacher_dtype}",
                f"distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size={teacher_tp}",
                f"distillation.teacher_models.teacher_model.inference.expert_parallel_size={teacher_ep}",
                f"distillation.teacher_models.teacher_model.inference.data_parallel_size={teacher_dp}",
                "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="
                f"{teacher_engine.get('gpu_memory_utilization', 0.4)}",
                "distillation.teacher_models.teacher_model.inference.max_model_len="
                f"{teacher_engine.get('max_model_len', algorithm['max_prompt_length'] + algorithm['max_completion_length'])}",
                "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="
                f"{teacher_engine.get('max_num_batched_tokens', algorithm['max_prompt_length'] + algorithm['max_completion_length'])}",
                "distillation.teacher_models.teacher_model.inference.max_num_seqs="
                f"{teacher_engine.get('max_num_seqs', algorithm['num_generations'])}",
                f"distillation.distillation_loss.loss_mode={algorithm['loss_mode']}",
                f"distillation.distillation_loss.use_task_rewards={str(algorithm['use_task_rewards']).lower()}",
                f"distillation.distillation_loss.use_policy_gradient={str(algorithm['use_policy_gradient']).lower()}",
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
    return overrides


def _uses_turboquant(payload: dict[str, Any]) -> bool:
    rollout_dtype = payload.get("rollout", {}).get("engine", {}).get("kv_cache_dtype", "")
    teacher_dtype = payload.get("teacher_scoring", {}).get("engine", {}).get("kv_cache_dtype", "")
    return str(rollout_dtype).startswith("turboquant_") or str(teacher_dtype).startswith("turboquant_")


def _backend_hydra_overrides(options: dict[str, Any]) -> list[str]:
    raw = options.get("hydra_overrides", [])
    if not isinstance(raw, list) or any(not isinstance(value, str) or not value.strip() for value in raw):
        raise ValueError("veRL backend_options.hydra_overrides must be a list of non-empty strings")
    protected = (
        "data.train_files=",
        "data.val_files=",
        "actor_rollout_ref.model.path=",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path=",
        "trainer.default_local_dir=",
    )
    if any(value.startswith(protected) for value in raw):
        raise ValueError("veRL backend overrides cannot replace selected data, model, agent, or artifact paths")
    return list(raw)


def _write_dataset(payload: dict[str, Any], path: Path) -> None:
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError("the isolated veRL environment must include datasets") from error
    model = payload.get("policy") or payload["student"]
    rows = [
        {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "agent_name": "posttrain_verifiers",
            "example_id": example["id"],
            "model_id": model["id"],
            **example["metadata"],
        }
        for example in payload["environment"]["examples"]
    ]
    Dataset.from_list(rows).to_parquet(str(path))


def _write_agent_config(payload: dict[str, Any], path: Path) -> None:
    renderer = payload["training"]["renderer"]
    config = [
        {
            "name": "posttrain_verifiers",
            "_target_": "posttrain.train.backends.verl.agent_loop.PosttrainVerifiersAgentLoop",
            "bridge_snapshot": payload["environment"]["bridge_snapshot"],
            "enable_thinking": renderer["reasoning_mode"] == "thinking",
        }
    ]
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _model_path(model: dict[str, Any]) -> str:
    artifact = model["artifact"]
    if artifact["kind"] == "hub":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("the isolated veRL environment must include huggingface-hub") from error
        return snapshot_download(repo_id=artifact["repo_id"], revision=artifact["revision"])
    return artifact["path"]


def _validate_runtime(manifest: dict[str, Any]) -> None:
    try:
        from importlib.metadata import version

        installed = version("verl")
    except Exception as error:
        raise RuntimeError("the selected isolated interpreter does not contain veRL") from error
    if not installed:
        raise RuntimeError("could not resolve the installed veRL version")
    worktree = Path(manifest["working_directory"])
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != manifest["backend_source_revision"]:
        raise RuntimeError(
            f"veRL worktree is at {head}, expected immutable revision {manifest['backend_source_revision']}"
        )
    backend_options = manifest["payload"]["training"].get("backend_options", {})
    expected_dirty = backend_options.get("source_dirty")
    expected_digest = backend_options.get("source_dirty_digest")
    if expected_dirty is not None:
        dirty, dirty_digest = _git_source_state(worktree)
        if dirty is not expected_dirty:
            raise RuntimeError(f"veRL worktree dirty state is {dirty}, expected {expected_dirty}")
        if expected_digest is not None and dirty_digest != expected_digest:
            raise RuntimeError("veRL worktree content changed after the training selection was resolved")


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
