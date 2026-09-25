"""Research-only VORTEX next: two soft lanes, bounded variance, aging evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from controller_policy_experiment import NullBackend
from posttrain.train import AdaptiveCurriculum
from posttrain.train.adaptive_curriculum import AdaptiveCurriculumController


@dataclass(frozen=True)
class Settings:
    exploration_share: float = 0.2
    uncertainty_weight: float = 1.0
    magnitude_scale: float = 0.1
    prior_strength: float = 4.0
    half_life: float = 20.0
    stability_penalty: float = 4.0


DEFAULT_SETTINGS = Settings()


def signal(rewards, scale=0.1, binary=False):
    if len(rewards) != 4 or any(not math.isfinite(r) or not 0 <= r <= 1 for r in rewards):
        raise ValueError("four finite normalized rewards in [0,1] required")
    variance = statistics.pvariance(rewards)
    return float(variance > 1e-12) if binary else variance / (variance + scale**2)


def normalize(weights):
    total = sum(weights.values())
    return {key: value / total if total else 1 / len(weights) for key, value in weights.items()}


def draw(weights, rng):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[key] for key in keys], k=1)[0]


class Sampler:
    def __init__(self, inventory, settings=DEFAULT_SETTINGS, binary=False):
        self.inventory = inventory
        self.settings = settings
        self.binary = binary
        # Sufficient statistics decay in place; no unbounded rollout history.
        self.history = defaultdict(lambda: [0.0, 0.0, 0.0, -1])

    def _state(self, task, step):
        state = self.history[task]
        if state[3] >= 0 and step > state[3]:
            decay = 2 ** (-(step - state[3]) / self.settings.half_life)
            for index in range(3):
                state[index] *= decay
        state[3] = step
        return state

    def observe(self, task, rewards, step):
        y = signal(rewards, self.settings.magnitude_scale, self.binary)
        state = self._state(task, step)
        state[0] += 1
        state[1] += y
        state[2] += y * y

    def scores(self, step):
        cfg = self.settings
        moments = {}
        classes = defaultdict(lambda: [0.0, 0.0, 0.0])
        own_class_contribution = {}
        for task, cls in self.inventory.items():
            n, first, second, _ = self._state(task, step)
            moments[task] = (n, first, second)
            if n:
                # Each observed task contributes at most one unit to its class.
                confidence = min(1.0, n)
                contribution = (confidence, confidence * first / n, confidence * second / n)
                own_class_contribution[task] = contribution
                totals = classes[cls]
                for index, value in enumerate(contribution):
                    totals[index] += value
        result = {}
        for task, cls in self.inventory.items():
            own = own_class_contribution.get(task, (0.0, 0.0, 0.0))
            peer_n, peer_first, peer_second = (max(0.0, total - own[index]) for index, total in enumerate(classes[cls]))
            # Uniform Beta(1,1) moments for the bounded signal: E[y]=1/2, E[y²]=1/3.
            prior_mean = (1 + peer_first) / (2 + peer_n)
            prior_second = (2 / 3 + peer_second) / (2 + peer_n)
            n, first, second = moments[task]
            mass = cfg.prior_strength + n
            mean = (cfg.prior_strength * prior_mean + first) / mass
            mean_second = (cfg.prior_strength * prior_second + second) / mass
            spread = max(0.0, mean_second - mean * mean)
            base = mean / (1 + cfg.stability_penalty * spread)
            uncertainty = math.sqrt(spread / (n + 1))
            result[task] = (base, uncertainty)
        return result

    def probabilities(self, eligible, scores, explore):
        by_class = defaultdict(list)
        for task in eligible:
            by_class[self.inventory[task]].append(task)
        task_weights = {
            task: scores[task][0] + (self.settings.uncertainty_weight * scores[task][1] if explore else 0)
            for task in eligible
        }
        class_weights = normalize(
            {cls: statistics.mean(task_weights[t] for t in tasks) for cls, tasks in by_class.items()}
        )
        result = {}
        for cls, tasks in by_class.items():
            within = normalize({task: task_weights[task] for task in tasks})
            result.update({task: class_weights[cls] * within[task] for task in tasks})
        return result


class World:
    def __init__(self, groups, seed, scenario, shrinkage=4.0):
        self.seed, self.scenario, self.shrinkage = seed, scenario, shrinkage
        self.by_task = defaultdict(list)
        self.by_class = defaultdict(list)
        self.visits = Counter()
        for group in groups:
            self.by_task[group["task_id"]].append(group)
            self.by_class[group["class"]].append(group)
        self.inventory = {task: rows[0]["class"] for task, rows in self.by_task.items()}
        self.peers = {
            task: [g for g in self.by_class[cls] if g["task_id"] != task] or self.by_task[task]
            for task, cls in self.inventory.items()
        }

    def group(self, task, step):
        digest = hashlib.sha256(f"{self.seed}:{task}:{self.visits[task]}".encode()).digest()
        self.visits[task] += 1
        own = self.by_task[task]
        pool = (
            own
            if int.from_bytes(digest[:8], "big") / 2**64 < len(own) / (len(own) + self.shrinkage)
            else self.peers[task]
        )
        item = dict(pool[int.from_bytes(digest[8:16], "big") % len(pool)])
        rewards = item["rewards"][:]
        cohort = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) % 4
        # Explicit stress assumptions, independent of sampler actions.
        if self.scenario == "saturation_recovery":
            if cohort == 0 and step >= 40:
                rewards = [1.0] * 4
            if cohort == 1:
                rewards = [0.0] * 4 if step < 60 else [0.0, 0.2, 0.4, 0.6]
        item["rewards"] = rewards
        return item


def run(groups, policy, seed, scenario="stationary", settings=DEFAULT_SETTINGS, shrinkage=4.0):
    world = World(groups, seed, scenario, shrinkage)
    rng = random.Random(seed)
    binary = policy == "binary_soft20"
    sampler = Sampler(world.inventory, settings, binary)
    controller = None
    if policy == "current_controller":
        controller = AdaptiveCurriculumController(
            world.inventory, AdaptiveCurriculum("class", seed=seed), NullBackend(), group_size=4
        )
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
            rounds += 1
            chosen = []
            lanes = []
            eligible = sorted(set(world.inventory) - selected)
            if len(eligible) < missing:
                break
            if controller:
                chosen = list(controller.select(missing, step=step, round_index=round_index).task_ids)
                lanes = ["current"] * len(chosen)
            else:
                scores = sampler.scores(step)
                for _ in range(missing):
                    eligible = sorted(set(world.inventory) - selected - set(chosen))
                    explore = rng.random() < settings.exploration_share
                    if policy == "uniform":
                        scores = {task: (1.0, 0.0) for task in world.inventory}
                    task = draw(sampler.probabilities(eligible, scores, explore), rng)
                    chosen.append(task)
                    lanes.append("explore" if explore else "variance")
            selected.update(chosen)
            observations = []
            for task, lane in zip(chosen, lanes, strict=True):
                item = world.group(task, step)
                rewards = item["rewards"]
                useful = int(statistics.pvariance(rewards) > 1e-12)
                counts["candidates"] += 1
                counts["retained"] += useful
                counts["zero_variance"] += 1 - useful
                counts["mastered_constant"] += int(min(rewards) == 1)
                counts["low_constant"] += int(max(rewards) == 0)
                counts["signal_sum"] += signal(rewards)
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
            # Preserve simultaneous batch semantics: selection sees no current batch rewards.
            for task, rewards in observations:
                sampler.observe(task, rewards, step)
            if controller:
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
        "mean_signal": counts["signal_sum"] / candidates,
        "per_step": per_step,
    }


def experiment(snapshot, seeds=10):
    rows = []
    support = []
    policies = ["uniform", "current_controller", "binary_soft20", "magnitude_soft20", "vortex_next"]
    for source in snapshot["runs"]:
        groups = [g for g in source["groups"] if g["size"] == 4 and not g["error_rollouts"]]
        # Validation is strict: missing numeric rewards cannot be reconstructed from useful flags.
        for group in groups:
            signal(group["rewards"])
        tasks = len({g["task_id"] for g in groups})
        support.append(
            {
                "name": source["name"],
                "groups": len(groups),
                "tasks": tasks,
                "excluded": len(source["groups"]) - len(groups),
            }
        )
        if tasks < 32:
            continue
        for scenario in ["stationary", "saturation_recovery"]:
            for policy in policies:
                settings = Settings(stability_penalty=0 if policy in {"binary_soft20", "magnitude_soft20"} else 4)
                for seed in range(seeds):
                    row = run(groups, policy, seed, scenario, settings)
                    row["source"] = source["name"]
                    rows.append(row)
            print(source["name"], scenario, flush=True)
    summaries = []
    for key in sorted({(r["source"], r["scenario"], r["policy"]) for r in rows}):
        subset = [r for r in rows if (r["source"], r["scenario"], r["policy"]) == key]
        metrics = {}
        for metric in [
            "completed_steps",
            "candidates_per_retained",
            "zero_variance_fraction",
            "mean_rounds",
            "new_fraction",
            "late_new_fraction",
            "covered_tasks",
            "mean_signal",
            "mastered_constant",
            "low_constant",
        ]:
            values = [r.get(metric, 0) for r in subset]
            metrics[metric] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
        summaries.append({"source": key[0], "scenario": key[1], "policy": key[2], "metrics": metrics})
    return {"settings": asdict(Settings()), "seeds": seeds, "support": support, "summaries": summaries, "rows": rows}


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
    result["controller_sha256"] = hashlib.sha256(
        Path("packages/train/src/posttrain/train/adaptive_curriculum.py").read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
