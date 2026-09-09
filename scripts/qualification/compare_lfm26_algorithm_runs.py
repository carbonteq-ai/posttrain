"""Compare held-out AutomationBench results from the matched LFM2.5 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path, name: str) -> dict:
    return json.loads((path / name).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--grpo", type=Path, required=True)
    parser.add_argument("--olmo3", type=Path, required=True)
    parser.add_argument("--gdpo", type=Path, required=True)
    parser.add_argument("--grpo-training", type=Path, required=True)
    parser.add_argument("--olmo3-training", type=Path, required=True)
    parser.add_argument("--gdpo-training", type=Path, required=True)
    parser.add_argument("--task-mix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mix = json.loads(args.task_mix.read_text())
    training_names = {row["name"] for row in mix["training"]}
    evaluation_names = {row["name"] for row in mix["evaluation"]}
    assert len(training_names) == len(evaluation_names) == 20
    assert training_names.isdisjoint(evaluation_names)

    summaries = {
        "base": _read(args.base, "summary.json"),
        "grpo": _read(args.grpo, "summary.json"),
        "olmo3": _read(args.olmo3, "summary.json"),
        "gdpo": _read(args.gdpo, "summary.json"),
    }
    for name, summary in summaries.items():
        assert summary["trajectories"] == 40, (name, summary["trajectories"])
        assert summary["failed"] == 0, name
        assert summary["truncation_reasons"].get("context_length", 0) == 0, name
        assert len(summary["per_task"]) == 20, name
    task_indices = {tuple(summary["task_indices"]) for summary in summaries.values()}
    assert len(task_indices) == 1, "evaluation arms did not use the same held-out tasks"

    base = summaries["base"]
    deltas = {
        name: {
            "partial_credit_mean": summary["partial_credit_mean"] - base["partial_credit_mean"],
            "exact_task_completion_rate": summary["exact_task_completion_rate"] - base["exact_task_completion_rate"],
        }
        for name, summary in summaries.items()
        if name != "base"
    }
    audits = {
        "grpo": _read(args.grpo_training, "scalar-audit.json"),
        "olmo3": _read(args.olmo3_training, "scalar-audit.json"),
        "gdpo": _read(args.gdpo_training, "audit.json"),
    }
    assert all(audit["global_steps"] == 50 for audit in (audits["grpo"], audits["olmo3"]))
    assert audits["gdpo"]["optimizer_steps"] == 50
    assert all(audit["learning_signal_exercised"] for audit in (audits["gdpo"],))
    assert audits["grpo"]["nonzero_update"] and audits["olmo3"]["nonzero_update"]

    report = {
        "comparison": "matched 50-update engineering comparison; not a benchmark claim",
        "policy": "LiquidAI/LFM2.5-2.6B@654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
        "task_mix": {
            "id": mix["id"],
            "domains": mix["domains"],
            "training_tasks": len(training_names),
            "evaluation_tasks": len(evaluation_names),
            "training_evaluation_disjoint": True,
            "evaluation_rollouts_per_task": 2,
        },
        "held_out": summaries,
        "delta_from_base": deltas,
        "training_audits": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
