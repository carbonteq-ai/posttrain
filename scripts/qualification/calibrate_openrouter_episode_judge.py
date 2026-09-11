"""Qualify the selected hosted judge with a model-authored episode assessment frame.

The bridge owns only the stable verdict schema and paid-service lifecycle.  The
judge first produces a structured assessment frame in its own vocabulary for
each episode, then uses that frame to fill the stable verdict.  This keeps the
wire contract reproducible without teaching a model a human-authored phrasing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, HostedInferenceBinding, NullObserver, RunContext
from posttrain.jobs import ExternalInferenceServiceRequest, ExternalInferenceUsageProjection
from posttrain.jobs.providers.openrouter import OpenRouterResolver

WORKSPACE = Path(__file__).resolve().parents[2]
BENCHMARK = Path(__file__).with_name("episode_judge_vllm_benchmark.py")
VOCABULARY_CALIBRATOR = Path(__file__).with_name("optimize_episode_judge_prompt.py")
DEFAULT_BINDING_ID = "hosted-inference/deepseek-v4-flash-openrouter-judge@1"
LOCAL_AUTOMATIONBENCH_SOURCE = (
    WORKSPACE.parent / "verifiers-environments" / "environments" / "automationbench_v1" / "src"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--binding",
        default=DEFAULT_BINDING_ID,
        help="Explicit hosted-inference binding; the provider is never inferred or allowed to fall back.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).with_name("fixtures") / "episode_judge_observable_labels_v2.json",
        help="Reviewed-control manifest that selects the admissible replay digests.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        help="Frozen model-authored prompt.json to replay; it replaces both rubric prose surfaces.",
    )
    parser.add_argument("--run-id", default="openrouter-episode-frame-calibration")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--case-limit",
        type=int,
        help="Bound the replay to the first N cases for transport preflight only.",
    )
    parser.add_argument(
        "--response-format-mode",
        choices=("strict_schema", "json_object"),
        default="strict_schema",
    )
    parser.add_argument(
        "--frame-response-format-mode",
        choices=("strict_schema", "json_object"),
        help="Optional transport mode for the intermediate assessment frame.",
    )
    parser.add_argument(
        "--review-after-verdict",
        action="store_true",
        help=(
            "Ask the same judge to reconcile its model-native frame and provisional verdict "
            "before emitting the admitted verdict."
        ),
    )
    parser.add_argument("--input-tokens-per-call", type=int, default=16_384)
    parser.add_argument(
        "--vocabulary-max-tokens",
        type=int,
        default=1_536,
        help="Bound the one-time model-native vocabulary profile separately from episode verdicts.",
    )
    parser.add_argument("--verdict-max-tokens", type=int, default=4_096)
    parser.add_argument("--frame-max-tokens", type=int, default=2_048)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Per-request ceiling; timeouts become retained failed calibration evidence.",
    )
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("at least two repetitions are required for the calibration gate")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("case-limit must be positive when supplied")
    if (
        min(
            args.concurrency,
            args.input_tokens_per_call,
            args.vocabulary_max_tokens,
            args.verdict_max_tokens,
            args.frame_max_tokens,
        )
        < 1
    ):
        parser.error("concurrency and token limits must be positive")
    if args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing evidence directory: {args.output}")
    return args


def _admissible_case_count(path: Path, reviewed_digests: list[str]) -> int:
    """Fail before paid service resolution unless every reviewed trace is present once."""

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("replay corpus is empty")
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("replay corpus rows must be JSON objects")
        digest = row.get("source_input_digest", row.get("input_digest"))
        if isinstance(digest, str):
            counts[digest] = counts.get(digest, 0) + 1
    missing = [digest for digest in reviewed_digests if counts.get(digest, 0) == 0]
    duplicates = [digest for digest in reviewed_digests if counts.get(digest, 0) > 1]
    if missing or duplicates:
        raise ValueError(
            "replay corpus must contain each reviewed source-input digest exactly once; "
            f"missing={missing}, duplicates={duplicates}"
        )
    return len(reviewed_digests)


def _reviewed_digests(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, dict) or not cases:
        raise ValueError("reviewed-control manifest must contain a non-empty cases object")
    digests = [value.get("source_input_digest") for value in cases.values() if isinstance(value, dict)]
    if any(not isinstance(value, str) or not value for value in digests) or len(set(digests)) != len(digests):
        raise ValueError("reviewed-control manifest must contain unique non-empty source_input_digest values")
    return cast(list[str], digests)


def _binding(binding_id: str) -> HostedInferenceBinding:
    return cast(
        HostedInferenceBinding,
        open_catalog(scope="openrouter-episode-calibration").resolve(CatalogRef("hosted-inference", binding_id)).value,
    )


def _source_environment(environment: dict[str, str]) -> dict[str, str]:
    """Prefer the sibling adapter source only for this local calibration host.

    Production images install the pinned adapter through their immutable lock.
    This narrow local source seam avoids building an image merely to replay
    retained messages, while still exercising the exact adapter contract.
    """

    configured = os.environ.get("POSTTRAIN_AUTOMATIONBENCH_ENV_SOURCE")
    source = Path(configured) if configured else LOCAL_AUTOMATIONBENCH_SOURCE
    if not source.is_dir():
        return environment
    updated = dict(environment)
    existing = updated.get("PYTHONPATH")
    updated["PYTHONPATH"] = str(source) if not existing else f"{source}{os.pathsep}{existing}"
    return updated


def _run(args: argparse.Namespace) -> dict[str, Any]:
    reviewed_digests = _reviewed_digests(args.labels)
    available_cases = _admissible_case_count(args.inputs, reviewed_digests)
    cases = min(available_cases, args.case_limit) if args.case_limit is not None else available_cases
    # Each replay has a frame request and verdict request, with an optional
    # reconciliation request. The verdict includes the whole bounded frame, so
    # count both original inputs plus the frame rather than treating it as free.
    # Include the one model-native vocabulary calibration call in the paid
    # projection; it is part of this qualification, never a free prelude.
    requests_per_replay = 3 if args.review_after_verdict else 2
    requests = cases * args.repetitions * requests_per_replay + 1
    replay_input_tokens = 2 * args.input_tokens_per_call + args.frame_max_tokens
    replay_output_tokens = args.frame_max_tokens + args.verdict_max_tokens
    if args.review_after_verdict:
        # The review sees the original bounded episode, the retained frame, and
        # the provisional verdict.  Account for all three instead of treating
        # a self-review as free when enforcing the run's service reservation.
        replay_input_tokens += args.input_tokens_per_call + args.frame_max_tokens + args.verdict_max_tokens
        replay_output_tokens += args.verdict_max_tokens
    input_tokens = cases * args.repetitions * replay_input_tokens + 2_048
    output_tokens = cases * args.repetitions * replay_output_tokens + 4_096
    binding = _binding(args.binding)
    args.output.mkdir(parents=True)
    context = RunContext(
        project_id="openrouter-episode-calibration",
        work_package_id="qualify/openrouter-episode-frame",
        run_id=args.run_id,
        job_kind="qualify.judge",
        job_definition_version="qualify/openrouter-episode-frame@1",
        workspace=(args.output / "runtime").resolve(),
        observer=NullObserver(),
    )
    request = ExternalInferenceServiceRequest(
        binding,
        ExternalInferenceUsageProjection(
            requests=requests,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )
    with OpenRouterResolver(timeout_seconds=90)(context, "judge/quality", request) as service:
        endpoint_key_var = "POSTTRAIN_CALIBRATION_ENDPOINT_KEY"
        endpoint_environment = _source_environment(os.environ.copy())
        endpoint_environment[endpoint_key_var] = service.endpoint.api_key
        extra_body = dict(cast(dict[str, Any], binding.sampling.get("extra_body", {})))
        extra_body["provider"] = service.provider["route"]
        omit_temperature = "temperature" not in binding.sampling
        omit_top_p = "top_p" not in binding.sampling
        vocabulary_profile = args.output / "vocabulary-profile.json"
        vocabulary_command = [
            sys.executable,
            str(VOCABULARY_CALIBRATOR),
            "calibrate-vocabulary",
            str(vocabulary_profile),
            "--elicitation-prompt",
            str(Path(__file__).with_name("fixtures") / "episode_judge_vocabulary_elicitation_v1.txt"),
            "--base-url",
            service.endpoint.base_url,
            "--api-key-env",
            endpoint_key_var,
            "--model",
            service.endpoint.model,
            "--model-revision",
            binding.model.revision,
            "--runtime-identity",
            f"openrouter/{binding.provider}",
            "--max-tokens",
            str(args.vocabulary_max_tokens),
            "--omit-chat-template-kwargs",
        ]
        if omit_temperature:
            vocabulary_command.append("--omit-temperature")
        subprocess.run(
            vocabulary_command,
            check=True,
            env=endpoint_environment,
        )
        command = [
            sys.executable,
            str(BENCHMARK),
            str(args.inputs.resolve()),
            str((args.output / "report.json").resolve()),
            "--base-url",
            service.endpoint.base_url,
            "--api-key-env",
            endpoint_key_var,
            "--model",
            service.endpoint.model,
            "--concurrency",
            str(args.concurrency),
            "--source-input-digests-json",
            json.dumps(reviewed_digests),
            "--response-format-mode",
            args.response_format_mode,
            "--repetitions",
            str(args.repetitions),
            "--max-tokens",
            str(args.verdict_max_tokens),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--restatement-first",
            "--current-production-protocol",
            "--vocabulary-profile",
            str(vocabulary_profile),
            "--restatement-max-tokens",
            str(args.frame_max_tokens),
            "--wire-evidence-indexes",
            "--omit-top-k",
            "--omit-chat-template-kwargs",
            "--extra-body-json",
            json.dumps(extra_body, sort_keys=True),
            "--label",
            f"{args.binding.replace('/', '-').replace('@', '-')}-episode-frame",
            "--model-revision",
            binding.model.revision,
            "--runtime-identity",
            f"openrouter/{binding.provider}",
            "--retain-content",
        ]
        if args.frame_response_format_mode is not None:
            command.extend(("--frame-response-format-mode", args.frame_response_format_mode))
        if omit_temperature:
            command.append("--omit-temperature")
        if omit_top_p:
            command.append("--omit-top-p")
        if args.review_after_verdict:
            command.append("--review-after-verdict")
        if args.prompt_artifact is not None:
            command.extend(("--prompt-artifact", str(args.prompt_artifact.resolve())))
        if args.case_limit is not None:
            command.extend(("--case-limit", str(args.case_limit)))
        subprocess.run(command, check=True, env=endpoint_environment)
        receipt = service.trace_identity()
    report = json.loads((args.output / "report.json").read_text())
    summary = {
        "schema_version": 1,
        "judge_binding": {"id": binding.id, "revision": binding.revision},
        "method": (
            "model-authored-episode-assessment-frame-review@1"
            if args.review_after_verdict
            else "model-authored-episode-assessment-frame@1"
        ),
        "response_format_mode": args.response_format_mode,
        "frame_response_format_mode": args.frame_response_format_mode or args.response_format_mode,
        "available_cases": available_cases,
        "reviewed_controls": str(args.labels.resolve()),
        "calls": requests,
        "projected_input_tokens": request.usage.input_tokens,
        "projected_output_tokens": request.usage.output_tokens,
        "prompt_artifact": str(args.prompt_artifact.resolve()) if args.prompt_artifact else None,
        "report": "report.json",
        "service_receipt": receipt,
        "structured_output_valid": report["structured_output_valid"],
        "requests": report["requests"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(_run(_arguments()), indent=2, sort_keys=True))
