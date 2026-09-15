"""Synthetic feedback experiment; no model training or claim of GRPO fidelity.

Run: python docs/research/proposals/simulations/curriculum_feedback.py
Each policy receives the same generated-attempt budget. The latent success
probabilities are used only by the simulator and final evaluator.
"""
import json
import random
import statistics
from collections import deque
from pathlib import Path

CLASSES = ["arithmetic", "algebra", "geometry", "probability"]
SCENARIOS = ["learnable", "already_mastered", "forgetting", "impossible",
             "irreducible_noise", "verifier_outage"]


def run(scenario, adaptive, seed, reserve=0.2, window=5):
    rng = random.Random(seed)
    p = [0.25, 0.35, 0.15, 0.45]
    rates = [0.025, 0.018, 0.012, 0.020]
    if scenario == "already_mastered":
        p[0] = 0.999
    if scenario == "impossible":
        p[2], rates[2] = 0.0, 0.0
    if scenario == "irreducible_noise":
        p[2], rates[2] = 0.5, 0.0
    history = [deque(maxlen=window) for _ in CLASSES]
    shares = []
    for t in range(200):
        if scenario == "forgetting" and t == 100:
            p[0] = 0.35
        scores = []
        for h in history:
            n = sum(n for _, n in h)
            # Unknown uses neutral base weight; never a fabricated zero.
            phi = sum(x for x, _ in h) / n if n else 1.0
            scores.append(reserve + (1 - reserve) * phi if adaptive else 1.0)
        q = [s / sum(scores) for s in scores]
        shares.append(q)
        counts, mixed, valid = [0]*4, [0]*4, [0]*4
        # Freeze the student for this batch; outcomes precede learning.
        for _ in range(40):
            c = rng.choices(range(4), weights=q)[0]
            successes = sum(rng.random() < p[c] for _ in range(4))
            if scenario == "verifier_outage" and c == 2 and 70 <= t < 100:
                continue
            counts[c] += 1
            valid[c] += 1
            mixed[c] += 0 < successes < 4
        for c in range(4):
            history[c].append((mixed[c], valid[c]))
            # Assumed learning law: useful groups improve learnable classes.
            # This assumption favors signal sampling; the noise case violates it.
            p[c] += rates[c] * mixed[c] * (1 - p[c])
    return {"success": sum(p)/4, "class_success": p,
            "late_share": [statistics.mean(q[c] for q in shares[-40:])
                           for c in range(4)]}


def main():
    results = {}
    for scenario in SCENARIOS:
        results[scenario] = {}
        for name, adaptive in [("fixed", False), ("adaptive", True)]:
            runs = [run(scenario, adaptive, seed) for seed in range(40)]
            results[scenario][name] = {
                "mean_success": statistics.mean(r["success"] for r in runs),
                "sd_across_seeds": statistics.stdev(r["success"] for r in runs),
                "late_class_shares": [statistics.mean(r["late_share"][c] for r in runs)
                                      for c in range(4)]}
        diffs = [run(scenario, True, s)["success"] - run(scenario, False, s)["success"]
                 for s in range(40)]
        results[scenario]["paired_difference"] = {
            "mean": statistics.mean(diffs), "sd": statistics.stdev(diffs)}
    sensitivity = {}
    for reserve in [0.1, 0.2, 0.4]:
        for window in [1, 5, 20]:
            sensitivity[f"reserve={reserve},window={window}"] = {
                scenario: statistics.mean(run(scenario, True, s, reserve, window)["success"]
                                          for s in range(20))
                for scenario in SCENARIOS}
    output = {"assumptions": {"seeds": 40, "intervals": 200, "groups": 40,
                              "attempts_per_group": 4, "reserve": 0.2, "window": 5},
              "results": results, "sensitivity": sensitivity}
    path = Path(__file__).with_name("curriculum_feedback_results.json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    for scenario, result in results.items():
        print(scenario, *(f"{result[p]['mean_success']:.4f}" for p in ["fixed", "adaptive"]),
              "delta", f"{result['paired_difference']['mean']:+.4f}")
    print(path)


if __name__ == "__main__":
    main()
