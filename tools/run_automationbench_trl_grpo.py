"""Qualify TRL native-MTP GRPO on complete Qwen 3.5 0.8B AutomationBench episodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
TRL_FORK = REPOSITORY.parent / "trl"
SOURCE_ROOTS = (
    TRL_FORK,
    REPOSITORY / "packages" / "common" / "src",
    REPOSITORY / "packages" / "data" / "src",
    REPOSITORY / "packages" / "train" / "src",
    REPOSITORY / "packages" / "tracking" / "src",
    REPOSITORY / "apps" / "lab" / "src",
    REPOSITORY / "environments" / "automationbench_v1" / "src",
)
sys.path[:0] = [str(path) for path in SOURCE_ROOTS]

from posttrain.common import ExecutionTarget, InferenceBinding, RunContext  # noqa: E402
from posttrain.common.variants import QWEN_35_08B  # noqa: E402
from posttrain.train import (  # noqa: E402
    QWEN35_RENDERER,
    GRPORequest,
    GRPOSettings,
    LoRAUpdate,
    TrainingBinding,
    TrainingLoop,
    grpo,
)
from posttrain_lab.environments import (  # noqa: E402
    AUTOMATIONBENCH_REVISION,
    create_automationbench_training_bridge,
)

TRL_REVISION = "1e284c15717894e9ab2612e14251975512ec6c7c"


@dataclass(frozen=True, slots=True)
class AutomationBenchEnvironmentSelection:
    id: str = "environments/automationbench-v1@a321764"
    revision: str = AUTOMATIONBENCH_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--task-index", type=int, default=194)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--automationbench-source",
        type=Path,
        default=Path("/tmp/codex-automationbench-a321764"),
    )
    args = parser.parse_args()

    if not (args.automationbench_source / "automationbench").is_dir():
        raise ValueError("--automationbench-source must be the pinned AutomationBench checkout root")
    # The pinned source runs on 3.12, but its current package metadata requires 3.13. Append only the checkout root
    # after this environment's site-packages so compiled Torch and vLLM dependencies cannot be shadowed.
    automationbench_source = str(args.automationbench_source.resolve())
    sys.path.append(automationbench_source)
    subprocess_paths = os.pathsep.join([*(str(path) for path in SOURCE_ROOTS[1:]), automationbench_source])
    existing_python_path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        subprocess_paths if not existing_python_path else os.pathsep.join((subprocess_paths, existing_python_path))
    )

    workspace = args.output.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    target = ExecutionTarget("targets/local-cuda-8gb", "1", "nvidia-cuda", 8, {"world_size": 1})
    training = TrainingBinding(
        id="training/qwen3.5-0.8b-trl-lora-mtp-qualification@1",
        revision="1",
        backend="trl@1.8.0",
        renderer=QWEN35_RENDERER,
        update=LoRAUpdate(rank=8, alpha=16, target_modules="all-linear"),
        target=target,
        runtime={
            "global_batch_size": args.num_generations,
            "backend_source_revision": TRL_REVISION,
        },
    )
    inference = InferenceBinding(
        id="inference/qwen3.5-0.8b-vllm-trl-rollout-mtp@1",
        revision="1",
        model=QWEN_35_08B,
        backend="vllm@0.25.1",
        renderer=QWEN_35_08B.renderer_contract,
        engine={
            "mode": "colocate",
            "sleep_during_optimization": True,
            "gpu_memory_utilization": 0.40,
            "tensor_parallel_size": 1,
            "max_model_len": 32768,
            "text_only": True,
            "skip_mm_profiling": True,
            "enforce_eager": True,
            "kv_cache_memory_bytes": 640 * 1024 * 1024,
            "kv_cache_dtype": "auto",
            "weight_sync_mode": "lora",
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
        },
        sampling={"max_tokens": 512, "temperature": 1.0, "top_p": 0.95},
        target=target,
        purpose=("rollout",),
    )
    settings = GRPOSettings(
        id="qwen3.5-0.8b/automationbench-trl-mtp-qualification@1",
        loop=TrainingLoop(
            max_steps=args.max_steps,
            max_length=32768,
            per_device_batch_size=1,
            gradient_accumulation_steps=args.num_generations,
            learning_rate=1e-5,
        ),
        num_prompts_per_step=1,
        num_generations=args.num_generations,
        max_prompt_length=32256,
        max_completion_length=512,
    )
    trace_path = workspace / "training" / "grpo" / "verifiers-traces.jsonl"
    bridge = create_automationbench_training_bridge(
        args.task_index,
        trace_path,
        "automationbench-qwen35-08b-trl-mtp-qualification",
        max_tokens=512,
        temperature=1.0,
        top_p=0.95,
        search_top_k=1,
        max_turns=50,
        max_total_tokens=32768,
        toolset="limited_zapier",
    )
    context = RunContext(
        project_id="foundation-models",
        work_package_id="train/qwen3.5-0.8b/automationbench-trl-mtp-qualification",
        run_id="automationbench-qwen35-08b-trl-mtp-qualification",
        job_kind="train.grpo",
        job_definition_version="1",
        workspace=workspace,
    )
    result = grpo(
        context,
        GRPORequest(
            policy=QWEN_35_08B,
            bridge=bridge,
            settings=settings,
            environment=AutomationBenchEnvironmentSelection(),
            training=training,
            inference=inference,
        ),
    )
    _assert_qualification(workspace, args.max_steps, args.num_generations)
    print(result)


def _assert_qualification(workspace: Path, expected_steps: int, num_generations: int) -> None:
    trainer = workspace / "training" / "grpo" / "trainer"
    traces_path = workspace / "training" / "grpo" / "verifiers-traces.jsonl"
    traces = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines()]
    expected_trajectories = expected_steps * num_generations
    if len(traces) != expected_trajectories:
        raise RuntimeError(f"expected {expected_trajectories} complete trajectories, observed {len(traces)}")
    if not any(_continued_after_tool_call(trace) for trace in traces):
        raise RuntimeError("qualification requires a tool call followed by another sampled model turn")

    summary = json.loads((trainer.parent / "training-summary.json").read_text(encoding="utf-8"))
    if int(summary["summary"]["global_step"]) != expected_steps:
        raise RuntimeError("qualification did not complete every selected optimizer step")
    history = summary["log_history"]
    if not any(float(row.get("grad_norm", 0.0)) > 0 for row in history):
        raise RuntimeError("qualification requires a non-zero backward gradient")
    for metric in (
        "rollout/spec_num_draft_tokens",
        "rollout/spec_num_accepted_tokens",
        "rollout/spec_accept_rate",
    ):
        if not any(float(row.get(metric, 0.0)) > 0 for row in history):
            raise RuntimeError(f"qualification requires positive {metric}")

    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("qualification requires safetensors to verify the exported adapter") from error
    adapter = load_file(str(trainer.parent / "adapter" / "adapter_model.safetensors"))
    if not any("lora_B" in name and bool(tensor.count_nonzero()) for name, tensor in adapter.items()):
        raise RuntimeError("qualification requires an exported LoRA update with changed B matrices")


def _continued_after_tool_call(trace: dict[str, object]) -> bool:
    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        return False
    sampled = [node for node in nodes if isinstance(node, dict) and node.get("sampled") is True]
    return len(sampled) > 1 and any(
        isinstance(node.get("message"), dict) and bool(node["message"].get("tool_calls")) for node in sampled[:-1]
    )


if __name__ == "__main__":
    main()
