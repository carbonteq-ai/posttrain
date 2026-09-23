#!/usr/bin/env python3
"""Compare replay results against a baseline, with bootstrap 95% intervals."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

COLUMNS = (
    ("collection_mean_s", "collection s", "{:.1f}"),
    ("episodes_per_hour", "episodes/h", "{:.1f}"),
    ("episode_wall_mean_s", "episode s", "{:.1f}"),
    ("ttft_mean_s", "TTFT ms", "{:.0f}"),
    ("ttft_p95_s", "TTFT p95 ms", "{:.0f}"),
    ("turn_latency_mean_s", "turn s", "{:.2f}"),
    ("output_tokens_per_s", "out tok/s", "{:.0f}"),
    ("prefilled_tokens", "prefilled tok", "{:,.0f}"),
    ("prefix_hit_rate", "prefix hit", "{:.1%}"),
    ("preemptions", "preempt", "{:.0f}"),
)


SKIP_FIRST = False


def _values(result: dict, key: str) -> list[float]:
    # The first repeat can pay one-time compilation for the sampling path the
    # warm-up did not exercise; --skip-first-repeat drops it for every config.
    repeats = result["repeats"][1:] if SKIP_FIRST and len(result["repeats"]) > 1 else result["repeats"]
    return [r["summary"].get(key, float("nan")) for r in repeats]


def _speedup_interval(base: list[float], cand: list[float], draws: int = 2000) -> tuple[float, float, float]:
    point = statistics.fmean(cand) / statistics.fmean(base)
    rng = random.Random(0)
    samples = sorted(
        statistics.fmean(rng.choices(cand, k=len(cand))) / statistics.fmean(rng.choices(base, k=len(base)))
        for _ in range(draws)
    )
    return point, samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--skip-first-repeat", action="store_true")
    parser.add_argument("--metric", default="collection_mean_s", help="speedup metric; lower is better for *_s")
    args = parser.parse_args()
    global SKIP_FIRST
    SKIP_FIRST = args.skip_first_repeat
    baseline = json.loads(args.baseline.read_text())
    rows = [baseline] + [json.loads(p.read_text()) for p in args.results if p != args.baseline]
    header = ["config"] + [label for _, label, _ in COLUMNS] + ["speedup (95% CI)", "spread"]
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    base_metric = _values(baseline, args.metric)
    lower_is_better = args.metric.endswith("_s")
    for result in rows:
        cells = [result["label"]]
        for key, _, fmt in COLUMNS:
            value = statistics.fmean(_values(result, key))
            if key.startswith("ttft"):
                value *= 1000.0
            cells.append(fmt.format(value))
        eph = _values(result, args.metric)
        # Speedup is always "how many times faster": invert time metrics.
        if lower_is_better:
            point, lo, hi = _speedup_interval(eph, base_metric)
        else:
            point, lo, hi = _speedup_interval(base_metric, eph)
        cells.append(f"{point:.3f}x ({lo:.3f}-{hi:.3f})")
        spread = (max(eph) - min(eph)) / statistics.fmean(eph) if len(eph) > 1 else 0.0
        cells.append(f"{spread:.1%}")
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
