"""Real AutomationBench updates with synthetic turn inputs, not an LLM judge.

Install this directory's deterministic_turn_fixture module in the selected
interpreter before using an isolated veRL worker. No PYTHONPATH bypass is used.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from automationbench_structured_toy import FileObserver, plain
from deterministic_turn_fixture import ANNOTATION_KEY, FIXTURE, SCORER_DIGEST, DeterministicTurnFixture
from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, ExecutionTarget, InferenceBinding, RunContext
from posttrain.environment import VerifiersV1ConfigActivation
from posttrain.train import (
    CAPOSettings,
    GDPOSettings,
    LoRAUpdate,
    RewardComponentProjection,
    RewardProjection,
    TrainingBinding,
    TrainingLoop,
    TrainingRenderer,
    TrainingRuntime,
    build_verifiers_structured_request,
    capo,
    gdpo,
)
from transformers import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("gdpo", "capo"), required=True)
    parser.add_argument("--backend", choices=("trl", "verl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verl-source", type=Path)
    args = parser.parse_args()
    if args.backend == "verl" and args.verl_source is None:
        parser.error("veRL requires its exact candidate checkout")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    torch.set_num_threads(8)
    catalog = open_catalog(scope="deterministic-qualification")
    policy = catalog.resolve(CatalogRef("model", "models/qwen3.5-0.8b@bf16")).value
    native = catalog.resolve(CatalogRef("environment", "automationbench-zapier-simple-grpo")).value
    target = ExecutionTarget("targets/qualification-workstation", "1", "nvidia-cuda", 96, {"world_size": 1})
    sampling = {"max_tokens": 7168, "temperature": 0.8, "top_p": 0.95}
    environment = dataclasses.replace(
        native,
        activation=VerifiersV1ConfigActivation(
            {
                "taskset": {"id": "automationbench-v1"},
                "agent": {
                    "harness": {"id": "null"},
                    "runtime": {"type": "subprocess"},
                    "timeout": {"setup": 120, "rollout": 1800, "finalize": 60, "scoring": 300},
                    "max_turns": 12,
                    "max_total_tokens": 8192,
                },
            }
        ),
        sampling=dataclasses.replace(native.sampling, **sampling),
        parameters={
            **native.parameters,
            "max_turns": 12,
            "max_total_tokens": 8192,
            "qualification_fixture_digest": SCORER_DIGEST,
        },
        num_tasks=2,
        max_concurrent=1,
    )
    backend = "trl@95a787b6c04f91a5d485fd827d31b1e1fb67ae8e"
    options = {"use_liger_kernel": False, "logits_chunk_size": 128}
    if args.backend == "verl":
        source = args.verl_source.resolve()
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        backend = f"verl@{revision}"
        options = {
            "python_executable": sys.executable,
            "working_directory": str(source),
            "source_revision": revision,
            "attention_implementation": "sdpa",
        }
        (output / "verl-source.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=source))
    training = TrainingBinding(
        f"training/deterministic-{args.backend}",
        "1",
        backend,
        TrainingRenderer("renderers/qualification", "qwen3.5", "qwen3.5", "off"),
        LoRAUpdate(rank=4, alpha=8),
        target,
        runtime=TrainingRuntime(global_batch_size=2, devices_per_node=1, timeout_seconds=7200),
        backend_options=options,
    )
    engine = {"max_model_len": 8192, "enable_thinking": False, "disable_torch_compile": True}
    if args.backend == "verl":
        engine = {
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.4,
            "enforce_eager": True,
            "text_only": True,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 8192,
        }
    inference = InferenceBinding(
        "inference/deterministic-policy",
        "1",
        policy,
        "vllm@0.25.2.dev2" if args.backend == "verl" else "transformers@5.14.1",
        policy.renderer_contract,
        engine,
        sampling,
        target,
        ("rollout",),
        capabilities=("tool-calling",),
    )
    loop = TrainingLoop(
        max_steps=5,
        max_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        checkpoint_steps=2,
        checkpoint_limit=3,
    )
    # A rollout response includes later turns and observations. The current
    # request contract requires the call cap to match this response cap; the
    # native agent still enforces the separate 8192-token total trajectory cap.
    extra = dict(id=f"deterministic/{args.algorithm}", loop=loop, max_prompt_length=1024, max_completion_length=7168)
    settings = (
        GDPOSettings(**extra, component_names=("fixture_a", "fixture_b"), component_weights=(1.0, 0.5))
        if args.algorithm == "gdpo"
        else CAPOSettings(**extra)
    )
    projection = RewardProjection(
        "qualification/deterministic-turns",
        "1",
        (
            RewardComponentProjection("fixture_a", "turn_mean", "fixture_a"),
            RewardComponentProjection("fixture_b", "turn_mean", "fixture_b"),
            RewardComponentProjection("outcome", "native_metric", "task_completed_correctly"),
        ),
        scorer_digest=SCORER_DIGEST,
        turns_info_key=ANNOTATION_KEY,
        turn_error_key="erroneous_turn_ids" if args.algorithm == "capo" else None,
    )
    context = RunContext(
        "qualification",
        "automationbench-deterministic",
        output.name,
        f"train.{args.algorithm}",
        "1",
        output,
        FileObserver(output / "observations.jsonl"),
    )
    request = build_verifiers_structured_request(
        policy=policy,
        environment=environment,
        settings=settings,
        reward_projection=projection,
        training=training,
        inference=inference,
        trace_path=output / "traces.jsonl",
        run_id=output.name,
    )
    request.bridge.enrichers = (DeterministicTurnFixture(),)
    selection = {
        "policy": plain(policy),
        "settings": plain(settings),
        "training": plain(training),
        "inference": plain(inference),
        "environment": plain(environment),
        "projection": plain(projection),
        "fixture": FIXTURE,
        "candidate_unpublished": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "selection.json").write_text(json.dumps(selection, default=str, indent=2))
    (output / "fixture-source.py").write_bytes(
        Path(sys.modules[DeterministicTurnFixture.__module__].__file__).read_bytes()
    )
    result = gdpo(context, request) if args.algorithm == "gdpo" else capo(context, request)
    (output / "result.json").write_text(json.dumps(plain(result), default=str, indent=2))
    print("DETERMINISTIC_TRAINING_COMPLETED", args.backend, args.algorithm, flush=True)


if __name__ == "__main__":
    main()
