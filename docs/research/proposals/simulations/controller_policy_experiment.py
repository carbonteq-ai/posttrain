"""Offline qualification of the real adaptive controller against random sampling.

The synthetic learner is deliberately simple. Results test the controller's
selection behavior and accounting; they are not evidence of an LLM training
gain. Run from the repository root with:

    uv run python docs/research/proposals/simulations/controller_policy_experiment.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from posttrain.train import AdaptiveCurriculum
from posttrain.train.adaptive_curriculum import AdaptiveCurriculumController

CLASSES = ("arithmetic", "algebra", "geometry", "probability")
TASKS_PER_CLASS = 100
TARGET_GROUPS = 10
GROUP_SIZE = 4
MAX_ROUNDS = 4
STEPS = 20
SEEDS = tuple(range(40))


class NullBackend:
    def append(self, record: Mapping[str, object]) -> None:
        del record

    def flush(self) -> None:
        pass

    def snapshot(self, path: Path, state: Mapping[str, object]) -> None:
        del path, state

    def close(self) -> None:
        pass


@dataclass
class Learner:
    probabilities: dict[str, float]
    rates: dict[str, float]
    attempts: Counter[str]

    @classmethod
    def create(cls, scenario: str) -> Learner:
        probabilities: dict[str, float] = {}
        rates: dict[str, float] = {}
        for class_index, class_id in enumerate(CLASSES):
            for task_index in range(TASKS_PER_CLASS):
                task_id = f"{class_id}/{task_index:03d}"
                offset = ((task_index * 7) % 11 - 5) * 0.025
                probability = min(0.95, max(0.05, (0.2 + class_index * 0.15) + offset))
                rate = 0.12
                if scenario == "fast_saturation":
                    probability, rate = 0.45 + offset / 2, 0.45
                elif scenario == "rare_geometry" and class_id == "geometry":
                    probability = 0.5 if task_index < 2 else 0.995
                    rate = 0.08 if task_index < 2 else 0.0
                elif scenario == "persistent_noise" and class_id == "probability":
                    probability, rate = 0.5, 0.0
                probabilities[task_id] = probability
                rates[task_id] = rate
        return cls(probabilities, rates, Counter())

    def group(self, task_id: str, seed: int) -> list[float]:
        rewards: list[float] = []
        for _ in range(GROUP_SIZE):
            attempt = self.attempts[task_id]
            self.attempts[task_id] += 1
            digest = hashlib.sha256(f"{seed}:{task_id}:{attempt}".encode()).digest()
            draw = int.from_bytes(digest[:8], "big") / 2**64
            rewards.append(float(draw < self.probabilities[task_id]))
        return rewards

    def update(self, retained: Sequence[str], scenario: str, step: int) -> None:
        for task_id in retained:
            probability = self.probabilities[task_id]
            self.probabilities[task_id] = min(
                0.995,
                probability + self.rates[task_id] * (0.995 - probability),
            )
        retained_classes = Counter(task_id.split("/", 1)[0] for task_id in retained)
        for task_id, probability in self.probabilities.items():
            class_id = task_id.split("/", 1)[0]
            transfer = min(0.04, retained_classes[class_id] * 0.002)
            self.probabilities[task_id] = min(0.995, probability + transfer * (0.995 - probability))
        if scenario == "regression" and step == 10:
            for task_id in self.probabilities:
                if task_id.startswith("arithmetic/"):
                    self.probabilities[task_id] = 0.45


def task_classes() -> dict[str, str]:
    return {
        f"{class_id}/{task_index:03d}": class_id
        for class_id in CLASSES
        for task_index in range(TASKS_PER_CLASS)
    }


def has_variance(rewards: Sequence[float]) -> bool:
    mean = sum(rewards) / len(rewards)
    return sum((reward - mean) ** 2 for reward in rewards) / len(rewards) > 0


def run(seed: int, scenario: str, policy: str) -> dict[str, object]:
    inventory = task_classes()
    learner = Learner.create(scenario)
    controller = None
    if policy == "adaptive":
        controller = AdaptiveCurriculumController(
            inventory,
            AdaptiveCurriculum(
                "class",
                class_exploration=0.2,
                task_discovery=0.2,
                history_groups=4,
                seed=seed,
            ),
            NullBackend(),
            group_size=GROUP_SIZE,
        )
    rng = random.Random(seed)
    shuffled = list(inventory)
    rng.shuffle(shuffled)
    shuffle_position = 0
    seen: set[str] = set()
    class_candidates: Counter[str] = Counter()
    candidates = useful = completed_steps = duplicates = 0

    for step in range(1, STEPS + 1):
        selected_this_step: set[str] = set()
        retained: list[str] = []
        for round_index in range(1, MAX_ROUNDS + 1):
            missing = TARGET_GROUPS - len(retained)
            if missing == 0:
                break
            if controller is not None:
                decision = controller.select(
                    missing,
                    step=step,
                    selection_kind="initial_batch" if round_index == 1 else "active_sampling_refill",
                    round_index=round_index,
                )
                selected = list(decision.task_ids)
                duplicates += decision.duplicate_fallbacks
            else:
                selected = []
                while len(selected) < missing:
                    if shuffle_position == len(shuffled):
                        rng.shuffle(shuffled)
                        shuffle_position = 0
                    task_id = shuffled[shuffle_position]
                    shuffle_position += 1
                    if task_id in selected_this_step:
                        continue
                    selected.append(task_id)
            selected_this_step.update(selected)
            observations: list[tuple[str, Sequence[float]]] = []
            for task_id in selected:
                rewards = learner.group(task_id, seed)
                observations.append((task_id, rewards))
                candidates += 1
                seen.add(task_id)
                class_candidates[inventory[task_id]] += 1
                if has_variance(rewards):
                    retained.append(task_id)
                    useful += 1
            if controller is not None:
                controller.observe(observations, step=step)
        if len(retained) == TARGET_GROUPS:
            learner.update(retained, scenario, step)
            completed_steps += 1

    return {
        "policy": policy,
        "scenario": scenario,
        "seed": seed,
        "candidate_groups": candidates,
        "useful_groups": useful,
        "retained_fraction": useful / candidates if candidates else 0,
        "completed_steps": completed_steps,
        "candidate_groups_per_completed_step": candidates / completed_steps if completed_steps else None,
        "unique_tasks": len(seen),
        "duplicate_fallbacks": duplicates,
        "class_candidate_share": {
            class_id: class_candidates[class_id] / candidates if candidates else 0
            for class_id in CLASSES
        },
    }


def confidence_interval(values: Sequence[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    # Two-sided 95% Student-t critical value for the fixed 40-seed experiment (df=39).
    margin = 2.022691 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "lower_95": mean - margin, "upper_95": mean + margin}


def validate(rows: Sequence[Mapping[str, object]], scenarios: Sequence[str]) -> None:
    expected = len(scenarios) * 2 * len(SEEDS)
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} runs, found {len(rows)}")
    keys = {(row["scenario"], row["policy"], row["seed"]) for row in rows}
    if len(keys) != expected:
        raise AssertionError("scenario, policy, and seed do not uniquely identify every run")
    for row in rows:
        candidates = int(row["candidate_groups"])
        useful = int(row["useful_groups"])
        completed = int(row["completed_steps"])
        unique = int(row["unique_tasks"])
        shares = row["class_candidate_share"]
        if not isinstance(shares, dict):
            raise AssertionError("class candidate shares must be a mapping")
        if not (0 <= useful <= candidates <= STEPS * TARGET_GROUPS * MAX_ROUNDS):
            raise AssertionError(f"invalid candidate accounting in {row}")
        if not (0 <= completed <= STEPS and 0 <= unique <= candidates):
            raise AssertionError(f"invalid completion or diversity accounting in {row}")
        if abs(sum(float(value) for value in shares.values()) - 1.0) > 1e-9:
            raise AssertionError(f"class candidate shares do not sum to one in {row}")
        if int(row["duplicate_fallbacks"]) != 0:
            raise AssertionError(f"the 400-task inventory should not require duplicate fallbacks: {row}")


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    paired_deltas: list[dict[str, object]] = []
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        for policy in ("random", "adaptive"):
            subset = [row for row in rows if row["scenario"] == scenario and row["policy"] == policy]
            summaries.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    **{
                        key: statistics.mean(float(row[key]) for row in subset)
                        for key in (
                            "candidate_groups",
                            "useful_groups",
                            "retained_fraction",
                            "completed_steps",
                            "unique_tasks",
                            "duplicate_fallbacks",
                            "candidate_groups_per_completed_step",
                        )
                    },
                    "class_candidate_share": {
                        class_id: statistics.mean(
                            float(row["class_candidate_share"][class_id])  # type: ignore[index]
                            for row in subset
                        )
                        for class_id in CLASSES
                    },
                }
            )
        by_policy_seed = {
            (str(row["policy"]), int(row["seed"])): row
            for row in rows
            if row["scenario"] == scenario
        }
        paired_deltas.append(
            {
                "scenario": scenario,
                "direction": "adaptive_minus_random",
                **{
                    key: confidence_interval(
                        [
                            float(by_policy_seed[("adaptive", seed)][key])
                            - float(by_policy_seed[("random", seed)][key])
                            for seed in SEEDS
                        ]
                    )
                    for key in (
                        "candidate_groups",
                        "retained_fraction",
                        "completed_steps",
                        "unique_tasks",
                        "candidate_groups_per_completed_step",
                    )
                },
            }
        )
    return {
        "description": "Synthetic controller-policy experiment; not an LLM training result.",
        "settings": {
            "seeds": len(SEEDS),
            "steps": STEPS,
            "target_groups": TARGET_GROUPS,
            "group_size": GROUP_SIZE,
            "max_rounds": MAX_ROUNDS,
            "task_discovery": 0.2,
            "class_exploration": 0.2,
        },
        "summaries": summaries,
        "paired_deltas": paired_deltas,
    }


def main() -> None:
    scenarios = ("normal", "fast_saturation", "rare_geometry", "regression", "persistent_noise")
    rows = [run(seed, scenario, policy) for scenario in scenarios for policy in ("random", "adaptive") for seed in SEEDS]
    validate(rows, scenarios)
    result = summarize(rows)
    directory = Path(__file__).parent
    output = directory / "controller_policy_results.json"
    raw_jsonl = directory / "controller_policy_runs.jsonl"
    raw_csv = directory / "controller_policy_runs.csv"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with raw_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "scenario", "policy", "seed", "candidate_groups", "useful_groups",
            "retained_fraction", "completed_steps", "candidate_groups_per_completed_step",
            "unique_tasks", "duplicate_fallbacks",
            *(f"{class_id}_candidate_share" for class_id in CLASSES),
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            shares = row["class_candidate_share"]
            assert isinstance(shares, dict)
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key in row},
                    **{f"{class_id}_candidate_share": shares[class_id] for class_id in CLASSES},
                }
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
