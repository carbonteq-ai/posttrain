"""Thin CLI for invoking code-defined jobs."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from posttrain.common.profiles import QWEN_35_2B
from posttrain.serve import QWEN35_VLLM_TEXT, BenchmarkCell, BenchmarkRequest

from .execution import AttemptSpec, execute, execute_tracked
from .jobs import (
    foundation_screening_job,
    noop_action,
    noop_job,
    run_noop,
    run_serving_cell,
    serving_benchmark_action,
)
from .source import resolve_git_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posttrain-lab")
    parser.add_argument("job", choices=("noop", "foundation-qwen-smoke"))
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
    else:
        request = BenchmarkRequest(
            model=QWEN_35_2B,
            profile=QWEN35_VLLM_TEXT,
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
    print(result)


if __name__ == "__main__":
    main()
