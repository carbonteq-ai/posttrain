"""Thin CLI for invoking code-defined jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .execution import AttemptSpec, execute, execute_tracked
from .jobs import noop_action, noop_job, run_noop
from .source import resolve_git_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posttrain-lab")
    parser.add_argument("job", choices=("noop",))
    parser.add_argument("--tracked", action="store_true")
    parser.add_argument("--project", default="posttrain-platform")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = _parser().parse_args()
    source = resolve_git_source(args.repository.resolve())
    spec = AttemptSpec(
        job=noop_job(source.revision),
        action=noop_action(),
        source_metadata=source.metadata(),
    )
    if args.tracked:
        result = execute_tracked(spec, run_noop, project=args.project)
    else:
        result = execute(spec, run_noop)
    print(result)


if __name__ == "__main__":
    main()
