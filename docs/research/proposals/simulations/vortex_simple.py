"""Research replay: retain current useful-group estimator, soften task routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from controller_policy_experiment import NullBackend
from posttrain.train import AdaptiveCurriculum
from posttrain.train.adaptive_curriculum import AdaptiveCurriculumController
from vortex_next import Sampler, Settings, World, draw

BASE_SETTINGS = Settings(stability_penalty=0, uncertainty_weight=0)
POLICIES = ("current_controller", "binary_soft20", "soft_yield", "soft_yield_explore", "production_yield_first")


def score_tasks(controller, sampler, step):
    """Use the current controller's predictive yield, plus epistemic uncertainty."""
    uncertainty = sampler.scores(step)
    return {task: (controller.task_priority(task), uncertainty[task][1]) for task in sampler.inventory}


def run_simple(groups, policy, seed, scenario="stationary", settings=BASE_SETTINGS):
    if policy not in POLICIES:
        raise ValueError(f"unsupported simple policy: {policy}")
    world = World(groups, seed, scenario)
    rng = random.Random(seed)
    sampler = Sampler(world.inventory, settings, binary=True)
    controller_settings = (
        AdaptiveCurriculum(
            "class",
            seed=seed,
            policy="yield_first",
            exploration_share=settings.exploration_share,
            uncertainty_weight=settings.uncertainty_weight,
            evidence_half_life_steps=settings.half_life,
            yield_prior_strength=settings.prior_strength,
        )
        if policy == "production_yield_first"
        else AdaptiveCurriculum("class", seed=seed)
    )
    controller = AdaptiveCurriculumController(world.inventory, controller_settings, NullBackend(), group_size=4)
    counts = Counter()
    seen = set()
    per_step = []
    for step in range(1, 101):
        selected = set()
        retained = 0
        rounds = 0
        for round_index in range(1, 5):
            missing = 8 - retained
            if not missing:
                break
            eligible = sorted(set(world.inventory) - selected)
            if len(eligible) < missing:
                break
            rounds += 1
            if policy in {"current_controller", "production_yield_first"}:
                chosen = list(controller.select(missing, step=step, round_index=round_index).task_ids)
                lanes = [policy] * len(chosen)
            else:
                scores = sampler.scores(step) if policy == "binary_soft20" else score_tasks(controller, sampler, step)
                chosen = []
                lanes = []
                for _ in range(missing):
                    eligible = sorted(set(world.inventory) - selected - set(chosen))
                    explore = rng.random() < settings.exploration_share
                    # The no-bonus ablation retains lane accounting, but both lanes
                    # are deliberately identical. No new/familiar pool is imposed.
                    task = draw(sampler.probabilities(eligible, scores, explore), rng)
                    chosen.append(task)
                    lanes.append("explore" if explore else "variance")
            selected.update(chosen)
            observations = []
            for task, lane in zip(chosen, lanes, strict=True):
                item = world.group(task, step)
                rewards = item["rewards"]
                variance = statistics.pvariance(rewards)
                std = variance**0.5
                useful = int(variance > 1e-12)
                counts["candidates"] += 1
                counts["retained"] += useful
                counts["zero_variance"] += 1 - useful
                counts["mastered_constant"] += int(min(rewards) == 1)
                counts["low_constant"] += int(max(rewards) == 0)
                counts["std_sum"] += std
                counts["variance_sum"] += variance
                counts["output_tokens"] += item["output_tokens"]
                counts["input_tokens"] += item["input_tokens"]
                counts[f"{lane}_selected"] += 1
                counts[f"{lane}_new"] += int(task not in seen)
                counts["new"] += int(task not in seen)
                counts["late_new"] += int(step > 50 and task not in seen)
                counts["late_candidates"] += int(step > 50)
                retained += useful
                seen.add(task)
                observations.append((task, rewards))
            # Batch outcomes become visible only after its entire selection.
            for task, rewards in observations:
                sampler.observe(task, rewards, step)
            controller.observe(observations, step=step)
        counts["completed_steps"] += int(retained == 8)
        counts["rounds"] += rounds
        per_step.append({"step": step, "retained": retained, "rounds": rounds})
    candidates = counts["candidates"]
    return {
        "policy": policy,
        "seed": seed,
        "scenario": scenario,
        **dict(counts),
        "covered_tasks": len(seen),
        "candidates_per_retained": candidates / counts["retained"] if counts["retained"] else None,
        "zero_variance_fraction": counts["zero_variance"] / candidates,
        "new_fraction": counts["new"] / candidates,
        "late_new_fraction": counts["late_new"] / counts["late_candidates"],
        "mean_rounds": counts["rounds"] / 100,
        "mean_std": counts["std_sum"] / candidates,
        "mean_variance": counts["variance_sum"] / candidates,
        "variance_per_million_output_tokens": counts["variance_sum"] * 1_000_000 / counts["output_tokens"],
        "per_step": per_step,
    }


def experiment(snapshot, seeds=10):
    rows = []
    support = []
    for source in snapshot["runs"]:
        groups = [group for group in source["groups"] if group["size"] == 4 and not group["error_rollouts"]]
        tasks = len({group["task_id"] for group in groups})
        support.append({"name": source["name"], "groups": len(groups), "tasks": tasks})
        if tasks < 32:
            continue
        for scenario in ("stationary", "saturation_recovery"):
            for policy in POLICIES:
                settings = replace(
                    BASE_SETTINGS,
                    uncertainty_weight=(
                        1.0
                        if policy == "binary_soft20"
                        else 0.2
                        if policy == "soft_yield_explore"
                        else 4.0
                        if policy == "production_yield_first"
                        else 0
                    ),
                )
                for seed in range(seeds):
                    row = run_simple(groups, policy, seed, scenario, settings)
                    row["source"] = source["name"]
                    rows.append(row)
            print(source["name"], scenario, flush=True)
    summaries = []
    keys = sorted({(row["source"], row["scenario"], row["policy"]) for row in rows})
    for key in keys:
        subset = [row for row in rows if (row["source"], row["scenario"], row["policy"]) == key]
        metrics = {}
        for metric in (
            "completed_steps",
            "candidates_per_retained",
            "zero_variance_fraction",
            "mean_rounds",
            "new_fraction",
            "late_new_fraction",
            "covered_tasks",
            "mastered_constant",
            "low_constant",
            "mean_std",
            "mean_variance",
            "variance_per_million_output_tokens",
        ):
            values = [row[metric] for row in subset]
            metrics[metric] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
        summaries.append({"source": key[0], "scenario": key[1], "policy": key[2], "metrics": metrics})
    return {"settings": asdict(BASE_SETTINGS), "seeds": seeds, "support": support, "summaries": summaries, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    snapshot = json.loads(args.input.read_text())
    result = experiment(snapshot, args.seeds)
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result["prototype_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["previous_simulator_sha256"] = hashlib.sha256(
        Path(__file__).with_name("vortex_next.py").read_bytes()
    ).hexdigest()
    result["controller_sha256"] = hashlib.sha256(
        Path("packages/train/src/posttrain/train/adaptive_curriculum.py").read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
