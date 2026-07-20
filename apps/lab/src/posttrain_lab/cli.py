"""Thin CLI for invoking code-defined jobs."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from posttrain.common import ModelVariant
from posttrain.common.profiles import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.eval import EvaluationBudget, EvaluationResult
from posttrain.eval.programs import GENERAL_SMOKE
from posttrain.serve import (
    LFM25_VLLM,
    LFM25_VLLM_TURBOQUANT_K8,
    QWEN35_VLLM_TEXT,
    QWEN35_VLLM_TURBOQUANT_K8,
    BenchmarkCell,
    BenchmarkRequest,
    BenchmarkResult,
    GenerationResult,
    LaunchRequest,
)
from posttrain.train import LFM25_SFT_SMOKE, QWEN35_SFT_SMOKE, SFTRequest, TrainingResult

from .data import load_gsm8k_supervised
from .execution import AttemptSpec, execute, execute_tracked
from .jobs import (
    GSM8K_TRAINING_ROLLOUTS,
    ManagedEvaluationRequest,
    evaluation_action,
    foundation_screening_job,
    gsm8k_posttraining_job,
    noop_action,
    noop_job,
    online_smoke_action,
    rollout_collection_action,
    run_managed_evaluation,
    run_noop,
    run_online_smoke,
    run_serving_cell,
    run_sft,
    serving_benchmark_action,
    sft_action,
    training_inputs,
)
from .source import resolve_git_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posttrain-lab")
    parser.add_argument(
        "job",
        choices=(
            "noop",
            "foundation-qwen-smoke",
            "foundation-lfm-smoke",
            "foundation-lfm-online-smoke",
            "foundation-qwen-gsm8k",
            "foundation-lfm-gsm8k",
            "gsm8k-qwen-sft-smoke",
            "gsm8k-lfm-sft-smoke",
            "gsm8k-qwen-preference-rollouts",
            "gsm8k-lfm-preference-rollouts",
        ),
    )
    parser.add_argument("--tracked", action="store_true")
    parser.add_argument("--project", default="posttrain-platform")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = _parser().parse_args()
    source = resolve_git_source(args.repository.resolve())
    if args.job == "noop":
        spec = AttemptSpec(
            job=noop_job(source.revision),
            action=noop_action(),
            source_metadata=source.metadata(),
        )
        operation = run_noop
    elif args.job in {"gsm8k-qwen-preference-rollouts", "gsm8k-lfm-preference-rollouts"}:
        model, profile = (
            (QWEN_35_2B, QWEN35_VLLM_TURBOQUANT_K8)
            if args.job == "gsm8k-qwen-preference-rollouts"
            else (LFM_25_12B_THINKING, LFM25_VLLM_TURBOQUANT_K8)
        )
        request = ManagedEvaluationRequest(
            launch=LaunchRequest(model, profile),
            program=GSM8K_TRAINING_ROLLOUTS,
            environment_id="gsm8k-train-candidates",
            context_window=8_192,
        )
        spec = AttemptSpec(
            job=gsm8k_posttraining_job(source.revision),
            action=rollout_collection_action(model.id),
            inputs={
                "model_profile_id": model.id,
                "serve_profile_id": profile.id,
                "program_id": request.program.id,
                "program_kind": request.program.kind,
                "environment_id": request.environment_id,
                "context_window": request.context_window,
                "num_tasks": request.budget.resolve(request.program.environment(request.environment_id))[0],
                "num_rollouts": request.budget.resolve(request.program.environment(request.environment_id))[1],
            },
            source_metadata=source.metadata(),
        )
        operation = partial(run_managed_evaluation, request=request)
    elif args.job in {"gsm8k-qwen-sft-smoke", "gsm8k-lfm-sft-smoke"}:
        model, profile = (
            (QWEN_35_2B, QWEN35_SFT_SMOKE)
            if args.job == "gsm8k-qwen-sft-smoke"
            else (LFM_25_12B_THINKING, LFM25_SFT_SMOKE)
        )
        request = SFTRequest(
            model=ModelVariant.foundation(model),
            dataset=load_gsm8k_supervised(count=2),
            profile=profile,
        )
        spec = AttemptSpec(
            job=gsm8k_posttraining_job(source.revision),
            action=sft_action(request),
            inputs=training_inputs(request),
            source_metadata=source.metadata(),
        )
        operation = partial(run_sft, request=request)
    elif args.job in {"foundation-qwen-gsm8k", "foundation-lfm-gsm8k"}:
        model, profile = (
            (QWEN_35_2B, QWEN35_VLLM_TURBOQUANT_K8)
            if args.job == "foundation-qwen-gsm8k"
            else (LFM_25_12B_THINKING, LFM25_VLLM_TURBOQUANT_K8)
        )
        request = ManagedEvaluationRequest(
            launch=LaunchRequest(model, profile),
            program=GENERAL_SMOKE,
            environment_id="math-gsm8k",
            context_window=8_192,
            budget=EvaluationBudget(num_tasks=1, max_concurrent=1),
        )
        spec = AttemptSpec(
            job=foundation_screening_job(source.revision),
            action=evaluation_action(request),
            inputs={
                "model_profile_id": model.id,
                "serve_profile_id": profile.id,
                "program_id": request.program.id,
                "program_kind": request.program.kind,
                "environment_id": request.environment_id,
                "context_window": request.context_window,
                "num_tasks": request.budget.num_tasks,
                "max_concurrent": request.budget.max_concurrent,
            },
            source_metadata=source.metadata(),
        )
        operation = partial(run_managed_evaluation, request=request)
    elif args.job == "foundation-lfm-online-smoke":
        request = LaunchRequest(LFM_25_12B_THINKING, LFM25_VLLM)
        spec = AttemptSpec(
            job=foundation_screening_job(source.revision),
            action=online_smoke_action(request.model),
            inputs={
                "model_profile_id": request.model.id,
                "serve_profile_id": request.profile.id,
                "endpoint_kind": "openai-compatible",
            },
            source_metadata=source.metadata(),
        )
        operation = partial(run_online_smoke, request=request)
    else:
        model, profile = (
            (QWEN_35_2B, QWEN35_VLLM_TEXT) if args.job == "foundation-qwen-smoke" else (LFM_25_12B_THINKING, LFM25_VLLM)
        )
        request = BenchmarkRequest(
            model=model,
            profile=profile,
            cell=BenchmarkCell(
                "foundation-smoke-v1",
                "short-interactive",
                1_024,
                1,
                128,
                32,
                1,
                1,
            ),
        )
        spec = AttemptSpec(
            job=foundation_screening_job(source.revision),
            action=serving_benchmark_action(request),
            inputs={
                "model_profile_id": request.model.id,
                "serve_profile_id": request.profile.id,
                "suite_id": request.cell.suite_id,
                "cell_id": request.cell.id,
            },
            source_metadata=source.metadata(),
        )
        operation = partial(run_serving_cell, request=request)
    if args.tracked:
        result = execute_tracked(spec, operation, project=args.project)
    else:
        result = execute(spec, operation)
    if isinstance(result, BenchmarkResult):
        print(json.dumps(result.as_json(), indent=2, sort_keys=True))
    elif isinstance(result, GenerationResult):
        print(json.dumps(result.summary(), indent=2, sort_keys=True))
    elif isinstance(result, EvaluationResult):
        print(
            json.dumps(
                {
                    "program_id": result.program_id,
                    "environment_id": result.environment_id,
                    "model_profile_id": result.model_profile_id,
                    "trace_ids": result.trace_ids,
                    "trace_sync_complete": result.synchronization.complete,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif isinstance(result, TrainingResult):
        print(
            json.dumps(
                {
                    "technique": result.technique,
                    "model_profile_id": result.model.profile.id,
                    "model_format": result.model.format,
                    "global_step": result.summary.global_step,
                    "train_loss": result.summary.train_loss,
                    "runtime_seconds": result.summary.runtime_seconds,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(result)


if __name__ == "__main__":
    main()
