"""Plan and execute inference benchmark suites as isolated Trackio runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

from common import BENCHMARKS_DIR, PROFILES_DIR, ProfileResolver

from .suites import BenchmarkCase, load_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a context/concurrency benchmark suite.")
    parser.add_argument("model_profile", help="Reference under profiles/models")
    parser.add_argument(
        "--suite",
        type=Path,
        default=BENCHMARKS_DIR / "inference" / "suites" / "core.yaml",
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--context-window", action="append", type=int, dest="contexts")
    parser.add_argument("--concurrency", action="append", type=int, dest="concurrencies")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def select_cases(
    cases: tuple[BenchmarkCase, ...],
    *,
    case_ids: list[str] | None = None,
    contexts: list[int] | None = None,
    concurrencies: list[int] | None = None,
    limit: int | None = None,
) -> tuple[BenchmarkCase, ...]:
    selected = tuple(
        case
        for case in cases
        if (not case_ids or case.id in case_ids)
        and (not contexts or case.context_window in contexts)
        and (not concurrencies or case.concurrency in concurrencies)
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    return selected


def _command(
    model_profile: str,
    case: BenchmarkCase,
    execution_id: str,
    serve_profile: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "serve.benchmark",
        model_profile,
        "--context-window",
        str(case.context_window),
        "--concurrency",
        str(case.concurrency),
        "--input-tokens",
        str(case.input_tokens),
        "--output-tokens",
        str(case.output_tokens),
        "--warmup-iterations",
        str(case.warmup_iterations),
        "--iterations",
        str(case.iterations),
        "--suite-id",
        case.suite_id,
        "--suite-execution-id",
        execution_id,
        "--case-id",
        case.id,
        "--shape-id",
        case.shape_id,
        "--reasoning-mode",
        case.reasoning_mode,
        "--name",
        f"{model_profile}-{case.id}-{execution_id}",
    ]
    if serve_profile is not None:
        command.extend(["--serve-profile", serve_profile])
    return command


def _serve_profile_for_variant(model_profile: dict, variant: str | None) -> str | None:
    if variant is None:
        return None
    serve_defaults = model_profile.get("defaults", {}).get("serve", {})
    variants = serve_defaults.get("variants", {}) if isinstance(serve_defaults, dict) else {}
    reference = variants.get(variant) if isinstance(variants, dict) else None
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"model profile does not declare serve variant {variant!r}")
    return reference


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suite = load_suite(args.suite)
    model = ProfileResolver(PROFILES_DIR).resolve("models", args.model_profile)
    cases = select_cases(
        suite.cases(),
        case_ids=args.case_ids,
        contexts=args.contexts,
        concurrencies=args.concurrencies,
        limit=args.limit,
    )
    if not cases:
        raise ValueError("suite filters selected no benchmark cases")
    execution_id = uuid.uuid4().hex[:10]
    plan = {
        "suite_id": suite.id,
        "suite_execution_id": execution_id,
        "model_profile": args.model_profile,
        "cases": [
            {
                **case.as_config(),
                "serve_profile": _serve_profile_for_variant(model.data, case.serve_variant),
            }
            for case in cases
        ],
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    failures: list[str] = []
    for case in cases:
        serve_profile = _serve_profile_for_variant(model.data, case.serve_variant)
        completed = subprocess.run(
            _command(args.model_profile, case, execution_id, serve_profile)
        )
        if completed.returncode != 0:
            failures.append(case.id)
            if not args.continue_on_error:
                break
    if failures:
        print(json.dumps({"failed_cases": failures}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
