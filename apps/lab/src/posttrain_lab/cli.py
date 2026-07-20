"""Thin CLI for invoking code-defined jobs."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from posttrain.common.profiles import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.serve import (
    LFM25_VLLM,
    QWEN35_VLLM_TEXT,
    BenchmarkCell,
    BenchmarkRequest,
    BenchmarkResult,
    GenerationResult,
    LaunchRequest,
)

from .execution import AttemptSpec, execute, execute_tracked
from .jobs import (
    foundation_screening_job,
    noop_action,
    noop_job,
    online_smoke_action,
    run_noop,
    run_online_smoke,
    run_serving_cell,
    serving_benchmark_action,
)
from .source import resolve_git_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posttrain-lab")
    parser.add_argument(
        "job",
        choices=("noop", "foundation-qwen-smoke", "foundation-lfm-smoke", "foundation-lfm-online-smoke"),
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
    else:
        print(result)


if __name__ == "__main__":
    main()
