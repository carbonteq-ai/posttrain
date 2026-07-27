"""Qualify Qwen 3.5 0.8B LoRA GRPO on complete AutomationBench episodes."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

REPOSITORY = Path(__file__).resolve().parents[1]
TRACKIO_FORK = REPOSITORY.parent / "trackio"
SOURCE_ROOTS = (
    TRACKIO_FORK,
    REPOSITORY / "packages" / "common" / "src",
    REPOSITORY / "packages" / "catalog" / "src",
    REPOSITORY / "packages" / "data" / "src",
    REPOSITORY / "packages" / "eval" / "src",
    REPOSITORY / "packages" / "serve" / "src",
    REPOSITORY / "packages" / "train" / "src",
    REPOSITORY / "packages" / "tracking" / "src",
    REPOSITORY / "packages" / "tracking-trackio" / "src",
    REPOSITORY / "apps" / "lab" / "src",
    REPOSITORY / "environments" / "automationbench_v1" / "src",
)
sys.path[:0] = [str(path) for path in SOURCE_ROOTS]

from posttrain.catalog import open_catalog  # noqa: E402
from posttrain.common import CatalogRef, ExecutionTarget, InferenceBinding, RunContext  # noqa: E402
from posttrain.common.variants import QWEN_35_08B  # noqa: E402
from posttrain.eval import EnvironmentBinding  # noqa: E402
from posttrain.tracking import RunError, RunOutcome, RunSpec  # noqa: E402
from posttrain.train import (  # noqa: E402
    QWEN35_RENDERER,
    GRPORequest,
    GRPOSettings,
    LoRAUpdate,
    TrainingBinding,
    TrainingLoop,
    TrainingRuntime,
    grpo,
)
from posttrain.train.integrations import (  # noqa: E402
    NativeVerifiersEnvironmentFactory,
    create_verifiers_training_bridge,
    preflight_verifiers_environment,
)
from posttrain_lab.jobs import GSM8K_TRAINING_ROLLOUTS, grpo_job_inputs  # noqa: E402
from posttrain_tracking_trackio import TrackioBackend, TrackioSettings  # noqa: E402
from run_workspace import prepare_run_workspace  # noqa: E402

VERL_REVISION = "a35908ca3c9632859c58d6a2855d858918ae21dc"
_INLINE_METRIC = re.compile(
    r"(?<![\w/])([A-Za-z_][\w]*(?:/[A-Za-z0-9_]+)*):"
    r"(?:np\.(?:float|int)\d+\()?([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _environment_bridge(
    *,
    environment_kind: str,
    task_indices: Sequence[int],
    trace_path: Path,
    run_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    parameters: Mapping[str, Any],
) -> tuple[EnvironmentBinding, Any]:
    if environment_kind == "automationbench":
        catalog = open_catalog(scope="foundation-models")
        base = catalog.resolve(
            CatalogRef("environment", "automationbench-zapier-simple-grpo")
        ).value
        if not isinstance(base, EnvironmentBinding):
            raise TypeError(
                "expected AutomationBench environment binding from the base catalog"
            )
        environment = replace(
            base,
            num_tasks=len(task_indices),
            parameters={**dict(base.parameters), **dict(parameters)},
        )
    elif environment_kind == "gsm8k":
        environment = replace(
            GSM8K_TRAINING_ROLLOUTS.environment("gsm8k-train-candidates"),
            num_tasks=len(task_indices),
            num_rollouts=1,
            parameters={"sampling_seed": 17},
        )
    else:
        raise ValueError(f"unsupported qualification environment {environment_kind!r}")
    config = preflight_verifiers_environment(environment)
    available = NativeVerifiersEnvironmentFactory(config)().taskset.load()
    tasks = {index: available[index] for index in task_indices}
    bridge = create_verifiers_training_bridge(
        environment,
        trace_path,
        run_id,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        purpose="grpo",
        tasks=tasks,
    )
    return environment, bridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--environment",
        choices=("automationbench", "gsm8k"),
        default="automationbench",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path("/home/hammad/projects/verl/.venv313/bin/python"),
    )
    parser.add_argument(
        "--verl-worktree",
        type=Path,
        default=Path("/home/hammad/projects/verl-upstream"),
    )
    parser.add_argument("--task-indices", type=int, nargs="+")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Canonical framework run ID; a UUID is generated when omitted.",
    )
    parser.add_argument(
        "--trackio-server-url",
        default=os.environ.get("TRACKIO_SERVER_URL"),
        help="Remote Trackio base URL. TRACKIO_WRITE_TOKEN supplies write access.",
    )
    parser.add_argument("--framework-revision")
    parser.add_argument("--framework-source-digest")
    parser.add_argument("--trackio-revision")
    parser.add_argument("--trackio-source-digest")
    parser.add_argument("--runtime-image-digest")
    parser.add_argument(
        "--mtp",
        action="store_true",
        help="Enable native Qwen 3.5 MTP-1 for rollout acceleration only.",
    )
    args = parser.parse_args()
    run_id = args.run_id or str(uuid.uuid4())
    task_indices = args.task_indices or (
        [194, 198] if args.environment == "automationbench" else [0, 1]
    )

    workspace = args.output.resolve()
    prepare_run_workspace(workspace)
    python_path = os.pathsep.join(str(path) for path in SOURCE_ROOTS)
    existing_python_path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        python_path if not existing_python_path else os.pathsep.join((python_path, existing_python_path))
    )

    target = ExecutionTarget("targets/local-cuda-8gb", "1", "nvidia-cuda", 8, {"world_size": 1})
    training = TrainingBinding(
        id="training/qwen3.5-0.8b-verl-lora-qualification@1",
        revision="1",
        backend="verl@a35908c",
        renderer=QWEN35_RENDERER,
        update=LoRAUpdate(
            rank=8,
            alpha=16,
            target_modules=r".*language_model.*[.](o_proj|down_proj)$",
        ),
        target=target,
        runtime=TrainingRuntime(
            global_batch_size=len(task_indices) * args.num_generations,
            devices_per_node=1,
            nodes=1,
            parameter_offload=True,
            optimizer_offload=True,
        ),
        backend_options={
            "python_executable": str(args.python_executable.absolute()),
            "working_directory": str(args.verl_worktree.resolve()),
            "source_revision": VERL_REVISION,
            "attention_implementation": "sdpa",
            "hydra_overrides": [
                "actor_rollout_ref.rollout.agent.num_workers=2",
                "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=16",
                "actor_rollout_ref.actor.entropy_from_logits_with_chunking=true",
                "actor_rollout_ref.actor.entropy_from_logits_chunk_size=256",
                "actor_rollout_ref.actor.fsdp_config.use_torch_compile=false",
                "actor_rollout_ref.model.use_fused_kernels=true",
                "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch",
                "reward.num_workers=2",
                "data.dataloader_num_workers=1",
            ],
        },
    )
    engine = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.55 if args.mtp else 0.70,
        "max_model_len": 8192,
        "max_num_batched_tokens": 4096 if args.mtp else 8192,
        "max_num_seqs": 4,
        "free_cache_engine": True,
        "sleep_during_optimization": True,
        "enforce_eager": True,
        "text_only": True,
    }
    if args.mtp:
        engine["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": 1,
        }
    inference = InferenceBinding(
        id=(
            "inference/qwen3.5-0.8b-vllm-verl-rollout-mtp@1"
            if args.mtp
            else "inference/qwen3.5-0.8b-vllm-verl-rollout@1"
        ),
        revision="1",
        model=QWEN_35_08B,
        backend="vllm@0.25.1" if args.mtp else "vllm@0.18.0",
        renderer=QWEN_35_08B.renderer_contract,
        engine=engine,
        sampling={"max_tokens": 512, "temperature": 1.0, "top_p": 0.95},
        target=target,
        purpose=("rollout",),
    )
    settings = GRPOSettings(
        id=f"qwen3.5-0.8b/{args.environment}-grpo-qualification@1",
        loop=TrainingLoop(
            max_steps=args.max_steps,
            max_length=8192,
            per_device_batch_size=len(task_indices) * args.num_generations,
            learning_rate=1e-5,
        ),
        num_prompts_per_step=len(task_indices),
        num_generations=args.num_generations,
        max_prompt_length=2048,
        max_completion_length=6144,
    )
    environment, bridge = _environment_bridge(
        environment_kind=args.environment,
        task_indices=tuple(task_indices),
        trace_path=workspace / "training" / "grpo" / "verifiers-traces.jsonl",
        run_id=run_id,
        max_tokens=512,
        temperature=1.0,
        top_p=0.95,
        parameters={
            "search_top_k": 1,
            "max_turns": 50,
            "max_total_tokens": 8192,
            "toolset": "limited_zapier",
        },
    )
    request = GRPORequest(
        policy=QWEN_35_08B,
        bridge=bridge,
        settings=settings,
        environment=environment,
        training=training,
        inference=inference,
    )
    source_metadata = {
        "framework_revision": args.framework_revision
        or _git_revision(REPOSITORY),
        "verl_revision": VERL_REVISION,
        "trackio_revision": args.trackio_revision
        or _git_revision(TRACKIO_FORK),
    }
    for name, value in (
        ("framework_source_digest", args.framework_source_digest),
        ("trackio_source_digest", args.trackio_source_digest),
        ("runtime_image_digest", args.runtime_image_digest),
    ):
        if value:
            source_metadata[name] = value
    run_spec = RunSpec(
        project_id="foundation-models",
        work_package_id=f"train/qwen3.5-0.8b/{args.environment}-grpo-qualification",
        stage="train",
        run_id=run_id,
        job_kind="train.grpo",
        job_definition_version="1",
        resolved_inputs=grpo_job_inputs(request),
        source_metadata=source_metadata,
        required_artifact_roles=("trained_model", "training_summary", "verifiers_traces"),
    )
    tracked_run = TrackioBackend(
        TrackioSettings(
            project="foundation-models",
            server_url=args.trackio_server_url,
            auto_log_gpu=True,
            auto_log_cpu=True,
        )
    ).start_run(run_spec)
    context = RunContext(
        project_id=run_spec.project_id,
        work_package_id=run_spec.work_package_id,
        run_id=run_spec.run_id,
        job_kind=run_spec.job_kind,
        job_definition_version=run_spec.job_definition_version,
        workspace=workspace,
        observer=tracked_run,
        source_metadata=run_spec.source_metadata,
    )
    started_at = datetime.now(UTC)
    try:
        result = grpo(
            context,
            request,
        )
        _assert_qualification(
            workspace,
            expected_steps=args.max_steps,
            expected_trajectories=len(task_indices)
            * args.num_generations
            * args.max_steps,
            expected_mtp=args.mtp,
            require_tool_continuation=args.environment == "automationbench",
        )
    except BaseException as error:
        tracked_run.finish(
            RunOutcome(
                status="failed",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                error=RunError(type=type(error).__name__, message=_safe_error(error)),
            )
        )
        _write_terminal_marker(workspace, run_id=run_id, status="failed")
        raise
    tracked_run.finish(
        RunOutcome(
            status="succeeded",
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )
    )
    _write_terminal_marker(workspace, run_id=run_id, status="succeeded")
    print(result)


def _assert_qualification(
    workspace: Path,
    *,
    expected_steps: int,
    expected_trajectories: int,
    expected_mtp: bool = False,
    require_tool_continuation: bool = False,
) -> None:
    trainer = workspace / "training" / "grpo" / "trainer"
    traces_path = workspace / "training" / "grpo" / "verifiers-traces.jsonl"
    traces = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines()]
    if len(traces) != expected_trajectories:
        raise RuntimeError(f"expected {expected_trajectories} complete trajectories, observed {len(traces)}")
    grouped_rewards: dict[tuple[int, str], set[float]] = {}
    for trace in traces:
        step = int(trace["run"]["step"])
        example_id = str(trace["info"]["example_id"])
        grouped_rewards.setdefault((step, example_id), set()).add(float(sum(trace["rewards"].values())))
    if not any(len(rewards) > 1 for rewards in grouped_rewards.values()):
        raise RuntimeError("GRPO qualification requires reward variance within at least one rollout group")
    if require_tool_continuation and not any(
        _continued_after_tool_call(trace) for trace in traces
    ):
        raise RuntimeError("GRPO qualification requires a tool call followed by another sampled model turn")

    metric_series: dict[str, list[float]] = {}
    native_log = (trainer / "verl-native.log").read_text(encoding="utf-8")
    for line in native_log.splitlines():
        for name, value in _INLINE_METRIC.findall(line):
            metric_series.setdefault(name, []).append(float(value))
    if int(metric_series.get("training/global_step", [0])[-1]) != expected_steps:
        raise RuntimeError("GRPO qualification did not complete every selected optimizer step")
    if not any(value > 0 for value in metric_series.get("actor/grad_norm", [])):
        raise RuntimeError("GRPO qualification requires a non-zero backward gradient")
    if expected_mtp:
        for metric in (
            "rollout/spec_num_draft_tokens",
            "rollout/spec_num_accepted_tokens",
            "rollout/spec_accept_rate",
        ):
            values = metric_series.get(metric, [])
            if len(values) < expected_steps or any(value <= 0 for value in values[-expected_steps:]):
                raise RuntimeError(f"MTP qualification requires positive {metric} on every optimizer step")

    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("qualification requires safetensors to verify the exported adapter") from error
    adapter = load_file(str(trainer / "model" / "lora_adapter" / "adapter_model.safetensors"))
    if not any("lora_B" in name and bool(tensor.count_nonzero()) for name, tensor in adapter.items()):
        raise RuntimeError("GRPO qualification requires an exported LoRA update with changed B matrices")


def _continued_after_tool_call(trace: dict[str, object]) -> bool:
    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        return False
    sampled = [node for node in nodes if isinstance(node, dict) and node.get("sampled") is True]
    if len(sampled) < 2:
        return False
    return any(
        isinstance(node.get("message"), dict) and bool(node["message"].get("tool_calls")) for node in sampled[:-1]
    )


def _git_revision(repository: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _safe_error(error: BaseException) -> str:
    message = str(error).strip().replace("\n", " ")
    return message[:500] or type(error).__name__


def _write_terminal_marker(
    workspace: Path,
    *,
    run_id: str,
    status: Literal["succeeded", "failed"],
) -> None:
    marker = workspace / ".posttrain-terminal.json"
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema": "carbonteq.posttrain-run-terminal.v1",
                "run_id": run_id,
                "status": status,
                "finished_at": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)


if __name__ == "__main__":
    main()
