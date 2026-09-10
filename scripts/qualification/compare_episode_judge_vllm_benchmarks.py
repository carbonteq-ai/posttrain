"""Compare saved episode-judge benchmarks without exposing replay content."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or not isinstance(value.get("cases"), list):
        raise ValueError(f"{path} is not a structured judge benchmark")
    return value


def _case_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["input_digest"]: case for case in run["cases"]}


def _summary(run: dict[str, Any]) -> dict[str, Any]:
    valid = int(run["structured_output_valid"])
    wall = float(run["wall_time_seconds"])
    return {
        "label": run["label"],
        "model": run["model"],
        "model_revision": run["model_revision"],
        "runtime_identity": run["runtime_identity"],
        "enable_thinking": run["enable_thinking"],
        "chat_template_kwargs": run.get("chat_template_kwargs"),
        "sampling": run.get("sampling"),
        "requests": run["requests"],
        "valid_verdicts": valid,
        "valid_rate": valid / int(run["requests"]),
        "valid_verdicts_per_second": valid / wall,
        "length_truncations": run["finish_reasons"].get("length", 0),
        "completion_tokens_per_second": run["completion_tokens_per_second"],
        "mean_latency_seconds": run["request_latency_seconds"]["mean"],
        "wall_time_seconds": wall,
    }


def _agreement(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_cases = _case_map(left)
    right_cases = _case_map(right)
    shared = sorted(left_cases.keys() & right_cases.keys())
    absolute_errors: list[float] = []
    exact = 0
    compared = 0
    valid_cases = 0
    for digest in shared:
        left_scores = left_cases[digest].get("scores")
        right_scores = right_cases[digest].get("scores")
        if not isinstance(left_scores, dict) or not isinstance(right_scores, dict):
            continue
        if left_scores.keys() != right_scores.keys():
            raise ValueError("judge score dimensions differ")
        valid_cases += 1
        for dimension in left_scores:
            error = abs(float(left_scores[dimension]) - float(right_scores[dimension]))
            absolute_errors.append(error)
            exact += error == 0
            compared += 1
    return {
        "left": left["label"],
        "right": right["label"],
        "shared_inputs": len(shared),
        "jointly_valid_inputs": valid_cases,
        "compared_scores": compared,
        "exact_score_agreement": exact / compared if compared else None,
        "mean_absolute_score_error": (
            sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
        ),
    }


def main() -> None:
    args = _arguments()
    runs = [_load(path) for path in args.inputs]
    digest_sets = [set(_case_map(run)) for run in runs]
    if any(digests != digest_sets[0] for digests in digest_sets[1:]):
        raise ValueError("benchmark runs do not contain the same input digests")
    report = {
        "schema_version": 1,
        "input_digests": sorted(digest_sets[0]),
        "runs": [_summary(run) for run in runs],
        "pairwise_score_agreement": [
            _agreement(left, right)
            for left, right in itertools.combinations(runs, 2)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
