"""Five-update installed-package qualification with native hosted judge composition.

Development candidate only until wheel metadata, source deltas and OCI gates pass.
The task plugin owns the rubric; this host selects and closes its model endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import torch
from automationbench_structured_toy import FileObserver, plain
from automationbench_v1.episode_prompt import (
    EPISODE_RUBRICS,
    GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT,
    build_episode_judge_messages,
)
from automationbench_v1.judge import RUBRIC, AutomationBenchTurnJudge, TurnQualityConfig
from automationbench_v1.limited_tools import selected_tool_definitions
from automationbench_v1.taskset import AutomationBenchTaskset
from episode_reward_profile import episode_component_weights
from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, ExecutionTarget, InferenceBinding, RunContext
from posttrain.environment import VerifiersV1ConfigActivation
from posttrain.jobs import bind_native_judges
from posttrain.serve import ServeLaunchRequest, launch
from posttrain.serve.backends.vllm import VllmServer
from posttrain.train import (
    ActiveGroupSampling,
    CAPOSettings,
    GDPOSettings,
    GRPOSettings,
    LoRAUpdate,
    RewardComponentProjection,
    RewardProjection,
    TrainingBinding,
    TrainingLoop,
    TrainingRenderer,
    TrainingRuntime,
    build_verifiers_grpo_request,
    build_verifiers_structured_request,
    capo,
    gdpo,
    grpo,
)
from transformers import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("grpo", "olmo3", "gdpo", "capo"), required=True)
    parser.add_argument("--backend", choices=("trl", "verl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--judge-code-revision", required=True, help="Immutable judge source Git blob SHA for this candidate."
    )
    parser.add_argument("--environment-wheel-sha256", required=True, help="Exact candidate environment wheel digest.")
    parser.add_argument("--judge-profile", choices=("qwen2b", "gemma12b", "nanbeige3b"), default="gemma12b")
    parser.add_argument("--episode-rewards", action="store_true")
    parser.add_argument("--judge-vllm-executable", help="Explicit inference runtime, independent of the trainer venv.")
    parser.add_argument("--judge-vllm-version", help="Required version of the explicitly selected inference runtime.")
    parser.add_argument("--verl-source", type=Path)
    parser.add_argument("--judge-port", type=int, default=8123)
    parser.add_argument(
        "--calibrate-only", action="store_true", help="Test the selected judge before any optimizer update."
    )
    parser.add_argument("--calibration-fixture", type=Path)
    parser.add_argument("--replay-inputs", type=Path)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--policy-profile", choices=("qwen08b", "lfm26b"), default="qwen08b")
    parser.add_argument("--task-mix", type=Path, help="Frozen task-mix manifest; training entries are selected.")
    parser.add_argument("--policy-context-window", type=int, default=12_288)
    parser.add_argument("--rollout-concurrency", type=int, default=64)
    parser.add_argument("--judge-concurrency", type=int, default=32)
    parser.add_argument("--judge-speculative-tokens", type=int, default=0)
    parser.add_argument("--judge-timeout-seconds", type=int, default=900)
    parser.add_argument("--prompts-per-step", type=int, default=1)
    parser.add_argument("--generations-per-prompt", type=int, default=2)
    parser.add_argument("--train-micro-batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-steps", type=int, default=10)
    args = parser.parse_args()
    if bool(args.judge_vllm_executable) != bool(args.judge_vllm_version):
        parser.error("explicit judge runtime requires both executable and version")
    judge_launcher = launch
    runtime_identity = None
    if args.judge_vllm_executable:
        executable = shutil.which(args.judge_vllm_executable)
        if executable is None:
            raise ValueError("selected judge inference executable is unavailable")
        version = subprocess.check_output([executable, "--version"], text=True, timeout=60).strip()
        if args.judge_vllm_version not in version.splitlines():
            raise ValueError(f"judge runtime version mismatch: {version}")
        runtime_identity = {"executable": executable, "version": args.judge_vllm_version}

        def server_factory(request, log_path, template_path):
            server = VllmServer(request, log_path, template_path)
            server.command = (executable, *server.command[1:])
            return server

        judge_launcher = partial(launch, server_factory=server_factory)
    if args.episode_rewards and args.algorithm != "gdpo":
        parser.error("episode reward comparison uses GDPO")
    scalar_grpo = args.algorithm in {"grpo", "olmo3"}
    if scalar_grpo and args.calibrate_only:
        parser.error("scalar GRPO-family controls have no judge to calibrate")
    if args.max_steps < 1:
        parser.error("max steps must be positive")
    if args.policy_context_window < 4_096:
        parser.error("policy context window must leave room for prompts and responses")
    counts = (
        args.rollout_concurrency,
        args.judge_concurrency,
        args.judge_timeout_seconds,
        args.prompts_per_step,
        args.generations_per_prompt,
        args.train_micro_batch_size,
        args.checkpoint_steps,
    )
    if any(value < 1 for value in counts) or args.generations_per_prompt < 2:
        parser.error("concurrency, group, batch, and checkpoint counts must be positive")
    if args.judge_speculative_tokens < 0:
        parser.error("judge speculative token count must be non-negative")
    if args.judge_speculative_tokens and args.judge_profile != "gemma12b":
        parser.error("this qualification supports judge MTP only for the paired-assistant Gemma 12B profile")
    update_batch_size = args.prompts_per_step * args.generations_per_prompt
    if update_batch_size % args.train_micro_batch_size:
        parser.error("update batch must be divisible by train micro-batch size")
    if args.calibrate_only and args.episode_rewards:
        if (args.calibration_fixture is None) == (args.replay_inputs is None):
            parser.error("episode calibration requires exactly one fixture or replay input")
    if args.backend == "verl" and args.verl_source is None:
        parser.error("veRL requires its exact candidate checkout")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task_mix = json.loads(args.task_mix.read_text()) if args.task_mix is not None else None
    domains = task_mix["domains"] if task_mix is not None else ["simple"]
    task_names = [row["name"] for row in task_mix["training"]] if task_mix is not None else []
    task_mix_digest = hashlib.sha256(args.task_mix.read_bytes()).hexdigest() if args.task_mix is not None else None
    judge_source = inspect.getsourcefile(AutomationBenchTurnJudge)
    assert judge_source is not None
    source_bytes = Path(judge_source).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(source_bytes)).encode() + b"\0" + source_bytes).hexdigest()
    if blob != args.judge_code_revision:
        raise ValueError("installed judge source differs from declared immutable source blob")
    (output / "judge-source.py").write_bytes(source_bytes)
    judge_source_manifest = {}
    for name, owner in {
        "judge.py": AutomationBenchTurnJudge,
        "episode_prompt.py": build_episode_judge_messages,
        "limited_tools.py": selected_tool_definitions,
        "taskset.py": AutomationBenchTaskset,
    }.items():
        path = inspect.getsourcefile(owner)
        assert path is not None
        payload = Path(path).read_bytes()
        (output / f"judge-source-{name}").write_bytes(payload)
        judge_source_manifest[name] = hashlib.sha256(payload).hexdigest()
    set_seed(42)
    torch.set_num_threads(8)
    catalog = open_catalog(scope="native-judge-qualification")
    policy_model_id = {
        "qwen08b": "models/qwen3.5-0.8b@bf16",
        "lfm26b": "models/lfm2.5-2.6b@bf16",
    }[args.policy_profile]
    policy = catalog.resolve(CatalogRef("model", policy_model_id)).value
    judge_model_id = {
        "gemma12b": "models/gemma4-12b-it@bf16",
        "nanbeige3b": "models/nanbeige4.2-3b@bf16",
        "qwen2b": "models/qwen3.5-2b@bf16",
    }[args.judge_profile]
    judge_model = catalog.resolve(CatalogRef("model", judge_model_id)).value
    native = catalog.resolve(CatalogRef("environment", "automationbench-zapier-simple-grpo")).value
    target = ExecutionTarget("targets/qualification-workstation", "1", "nvidia-cuda", 96, {"world_size": 1})
    judge_sampling = {
        "temperature": 0.0,
        "max_tokens": 16_384,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": args.judge_profile != "qwen2b"}},
    }
    judge_engine = {
        "max_model_len": 24_576,
        "gpu_memory_utilization": 0.35 if args.judge_profile == "gemma12b" else 0.20,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "max_num_seqs": args.judge_concurrency,
        "text_only": True,
        "skip_mm_profiling": True,
    }
    if args.judge_profile == "gemma12b":
        judge_engine["reasoning_parser"] = "gemma4"
    elif args.judge_profile == "nanbeige3b":
        judge_engine["reasoning_parser"] = "nanbeige"
    if args.judge_speculative_tokens:
        judge_engine["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.judge_speculative_tokens,
            "draft_model": {
                "repo_id": "google/gemma-4-12B-it-assistant",
                "revision": "364bd03c9952e5b7da73665ee30c9eccfc408345",
                "path": "/root/.cache/huggingface/hub/models--google--gemma-4-12B-it-assistant/snapshots/364bd03c9952e5b7da73665ee30c9eccfc408345",
            },
        }
    judge_inference = InferenceBinding(
        "inference/automationbench-judge",
        "1",
        judge_model,
        f"vllm@{args.judge_vllm_version or '0.25.2.dev2'}",
        judge_model.renderer_contract,
        judge_engine,
        judge_sampling,
        target,
        ("eval",),
        startup_timeout_seconds=600,
    )
    judge_request = ServeLaunchRequest(judge_inference, port=args.judge_port)
    judge_config = TurnQualityConfig(
        assessment_scope="episode" if args.episode_rewards else "turn",
        rubric=GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT if args.episode_rewards else RUBRIC,
        id="automationbench-v1",
        name="quality",
        model=judge_request.endpoint.model,
        model_revision=judge_model.revision,
        code_revision=args.judge_code_revision,
        base_url=judge_request.endpoint.base_url,
        sampling=judge_sampling,
        input_budget_tokens=8_192,
        timeout_seconds=args.judge_timeout_seconds,
    )
    # Freeze native defaults too: the selection and plugin must use one exact
    # sampling contract, rather than relying on different runtime defaults.
    judge_inference = dataclasses.replace(judge_inference, sampling=judge_config.sampling.model_dump(mode="json"))
    judge_request = dataclasses.replace(judge_request, inference=judge_inference)
    scorer_digest = AutomationBenchTurnJudge(judge_config).scorer_digest
    sampling = {"max_tokens": 2048, "temperature": 0.8, "top_p": 0.95}
    task_config = {"toolset": "limited_zapier"}
    if not scalar_grpo:
        task_config["judges"] = [judge_config.model_dump(mode="json")]
    environment = dataclasses.replace(
        native,
        activation=VerifiersV1ConfigActivation(
            {
                "taskset": {
                    "id": "automationbench-v1",
                    "domains": domains,
                    "task_names": task_names,
                    "task": task_config,
                },
                "agent": {
                    "harness": {"id": "null"},
                    "runtime": {"type": "subprocess"},
                    "timeout": {
                        "setup": 120,
                        "rollout": 1800,
                        "finalize": 60,
                        "scoring": args.judge_timeout_seconds,
                    },
                    "max_turns": 12,
                    "max_total_tokens": 8192,
                },
            }
        ),
        sampling=dataclasses.replace(native.sampling, **sampling),
        parameters={
            **native.parameters,
            "domains": domains,
            "toolset": "limited_zapier",
            "max_turns": 12,
            "max_total_tokens": 8192,
        },
        num_tasks=len(task_names) if task_names else 2,
        max_concurrent=args.rollout_concurrency,
    )
    rollout_output_budget = environment.parameters["max_total_tokens"]
    judge_output_budget = judge_sampling["max_tokens"]
    judge_context_window = judge_engine["max_model_len"]
    if not scalar_grpo:
        if judge_config.input_budget_tokens < rollout_output_budget:
            raise ValueError("judge input budget cannot be smaller than the rollout output budget")
        if judge_config.input_budget_tokens + judge_output_budget > judge_context_window:
            raise ValueError("judge input and output budgets exceed the selected model context")
    backend_options = {"use_liger_kernel": False, "logits_chunk_size": 128}
    backend = "trl@95a787b6c04f91a5d485fd827d31b1e1fb67ae8e"
    if args.backend == "verl":
        source = args.verl_source.resolve()
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        backend = f"verl@{revision}"
        backend_options = {
            "python_executable": sys.executable,
            "working_directory": str(source),
            "source_revision": revision,
            "attention_implementation": "sdpa",
        }
    training = TrainingBinding(
        f"training/native-{args.backend}",
        "1",
        backend,
        TrainingRenderer(
            "renderers/native-qualification",
            policy.family,
            "qwen3.5" if policy.family == "qwen3.5" else "default",
            "thinking" if policy.family == "qwen3.5" else "native",
        ),
        LoRAUpdate(rank=4, alpha=8),
        target,
        runtime=TrainingRuntime(global_batch_size=update_batch_size, devices_per_node=1, timeout_seconds=7200),
        backend_options=backend_options,
    )
    engine = {
        "mode": "colocate",
        "sleep_during_optimization": True,
        "max_model_len": args.policy_context_window,
        "gpu_memory_utilization": 0.20 if args.backend == "trl" else 0.25,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "text_only": True,
        "skip_mm_profiling": True,
        "max_num_seqs": args.rollout_concurrency,
        "max_num_batched_tokens": args.policy_context_window,
        "kv_cache_memory_bytes": 512 * 1024 * 1024,
        "weight_sync_mode": "lora",
        "weight_name_prefix": "language_model.",
        "enable_chunked_prefill": True,
        "free_cache_engine": True,
    }
    inference = InferenceBinding(
        "inference/native-policy",
        "1",
        policy,
        "vllm@0.25.2.dev2",
        policy.renderer_contract,
        engine,
        sampling,
        target,
        ("rollout",),
        capabilities=("tool-calling",),
    )
    loop = TrainingLoop(
        max_steps=args.max_steps,
        max_length=args.policy_context_window,
        per_device_batch_size=args.train_micro_batch_size,
        gradient_accumulation_steps=update_batch_size // args.train_micro_batch_size,
        learning_rate=1e-5,
        checkpoint_steps=args.checkpoint_steps,
        checkpoint_limit=3,
    )
    extra = dict(
        id=f"native/{args.algorithm}",
        loop=loop,
        num_prompts_per_step=args.prompts_per_step,
        num_generations=args.generations_per_prompt,
        max_prompt_length=args.policy_context_window - 2_048,
        max_completion_length=2_048,
    )
    if args.algorithm == "grpo":
        settings = GRPOSettings(**extra)
    elif args.algorithm == "olmo3":
        settings = GRPOSettings(
            **extra,
            algorithm="olmo3",
            advantage_scaling="none",
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.272,
            importance_sampling_mode="token_truncate",
            importance_sampling_clip_min=None,
            importance_sampling_clip_max=2.0,
            active_sampling=ActiveGroupSampling(max_candidate_batches=10),
        )
    elif args.algorithm == "gdpo":
        settings = GDPOSettings(
            **extra,
            component_names=("partial_credit", *EPISODE_RUBRICS)
            if args.episode_rewards
            else ("partial_credit", "quality"),
            component_weights=episode_component_weights(tuple(EPISODE_RUBRICS)) if args.episode_rewards else (1.0, 1.0),
        )
    else:
        settings = CAPOSettings(**extra)
    projection = RewardProjection(
        "native/automationbench",
        "1",
        (
            RewardComponentProjection("partial_credit", "scalar"),
            RewardComponentProjection("outcome", "native_metric", "task_completed_correctly"),
            RewardComponentProjection("quality", "turn_mean", "quality"),
        ),
        scorer_digest=scorer_digest,
        turns_info_key="posttrain_turn_rewards",
        turn_error_key="erroneous_turn_ids" if args.algorithm == "capo" else None,
    )
    if args.episode_rewards:
        projection = RewardProjection(
            "native/automationbench-episode",
            "1",
            (
                RewardComponentProjection("partial_credit", "scalar"),
                *(RewardComponentProjection(name, "annotation", f"episode_reward/{name}") for name in EPISODE_RUBRICS),
            ),
            scorer_digest=scorer_digest,
        )
    context = RunContext(
        "qualification",
        "automationbench-native-judges",
        output.name,
        f"train.{args.algorithm}",
        "1",
        output,
        FileObserver(output / "observations.jsonl"),
    )

    def build_request(active_environment):
        if scalar_grpo:
            return build_verifiers_grpo_request(
                policy=policy,
                environment=active_environment,
                settings=settings,
                training=training,
                inference=inference,
                trace_path=output / "traces.jsonl",
                run_id=output.name,
            )
        return build_verifiers_structured_request(
            policy=policy,
            environment=active_environment,
            settings=settings,
            reward_projection=projection,
            training=training,
            inference=inference,
            trace_path=output / "traces.jsonl",
            run_id=output.name,
        )

    # Validate the complete training selection before allocating judge inference.
    build_request(environment)

    def execute(active_environment):
        request = build_request(active_environment)
        selection = {
            "policy": plain(policy),
            "settings": plain(settings),
            "episode_rubrics": EPISODE_RUBRICS if args.episode_rewards else None,
            "training": plain(training),
            "environment": plain(active_environment),
            "candidate_unpublished": True,
            "environment_wheel_sha256": args.environment_wheel_sha256,
            "task_mix": task_mix,
            "task_mix_sha256": task_mix_digest,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        if not scalar_grpo:
            selection.update(
                {
                    "judge": judge_config.model_dump(mode="json"),
                    "judge_context_window": judge_context_window,
                    "judge_runtime": runtime_identity,
                    "projection": plain(projection),
                    "judge_source_sha256": judge_source_manifest,
                }
            )
        (output / "selection.json").write_text(json.dumps(selection, default=str, indent=2))
        if args.calibrate_only:
            if args.episode_rewards:
                judges = active_environment.activation.config["taskset"]["task"]["judges"]
                connected = AutomationBenchTurnJudge(TurnQualityConfig.model_validate(judges[0]))
                if args.replay_inputs is not None:
                    from replay_episode_judge import run_materialized_with_judge

                    asyncio.run(run_materialized_with_judge(args.replay_inputs, output / "replay", connected))
                else:
                    from calibrate_general_episode_prompt import run_with_judge

                    asyncio.run(run_with_judge(args.calibration_fixture, output / "calibration", connected))
            else:
                from calibrate_automationbench_judge import run

                asyncio.run(run(output / "selection.json", output / "calibration"))
            return
        if scalar_grpo:
            result = grpo(context, request)
        elif args.algorithm == "gdpo":
            result = gdpo(context, request)
        else:
            result = capo(context, request)
        (output / "result.json").write_text(json.dumps(plain(result), default=str, indent=2))

    if scalar_grpo:
        execute(environment)
    else:
        with bind_native_judges(context, environment, {"quality": judge_request}, launcher=judge_launcher) as bound:
            execute(bound)
    print("NATIVE_JUDGE_TRAINING_COMPLETED", args.backend, args.algorithm, flush=True)


if __name__ == "__main__":
    main()
