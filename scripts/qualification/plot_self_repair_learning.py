"""Plot retained local repair losses and native outcomes without new model calls."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(evidence, output):
    updates = json.loads((evidence / "adaptive09-updates.json").read_text())
    audit = json.loads((evidence / "adaptive09-coverage-audit.json").read_text())
    phases = [p for p in audit["phases"] if p["cohort"] == "failures"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
    fig.suptitle("Conditional loss falls while selected task success peaks at step 5", fontsize=14, x=.12, ha="left")
    axes[0].plot([u["step"] for u in updates], [u["repair_loss"] for u in updates], marker="o", color="#3366a1")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Repair forward KL\nunweighted, log scale")
    for metric, label, color in [("task_completed_correctly", "Task success", "#3366a1"),
                                  ("partial_credit", "Partial credit", "#b45a22")]:
        values = [100 * p["summary"]["metrics"][metric]["full_population_task_mean"] for p in phases]
        axes[1].plot([p["step"] for p in phases], values, label=label, marker="o", color=color)
        for p, value in zip(phases[1:], values[1:], strict=True):
            axes[1].annotate(f"{value:.1f}%", (p["step"], value), xytext=(0, 8 if metric == "partial_credit" else -16),
                             textcoords="offset points", ha="center", color=color, fontsize=10)
    axes[1].set_ylabel("Native score (%)")
    axes[1].set_xlabel("Optimizer updates")
    axes[1].set_ylim(-5, 100)
    axes[1].set_xlim(0, 10.5)
    axes[1].set_xticks(range(0, 11))
    axes[1].legend(loc="upper left", frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.12, .025, "Experiment 09: four adapted failures, two episodes each; not heldout generalization.\n"
             "Loss is measured before each update; outcomes use the saved post-update policy. Teacher changes via EMA.",
             fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.evidence, args.output)
