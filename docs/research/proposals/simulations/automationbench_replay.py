"""Compact AutomationBench trace export and noncausal curriculum replay.

Run ``export`` once with access to the retained Trackio runs. ``simulate`` is
offline and deterministic for a fixed snapshot. No prompts or tool transcripts
are written. See docs/plan/automationbench-curriculum-replay.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT = "posttrain-lab"
SERVER = "https://trackio.carbonteq.com"
DEFAULT_RUNS = (
    "train.grpo-lfm26-olmo3-random20-20260912-r3",
    "train.grpo-lfm26-olmo3-adaptive20-20260912-r4",
    "train.grpo-lfm26-olmo3-adaptive20-discovery-v4-20260912-r1",
    "train.grpo-lfm26-olmo3-adaptive-oversample10x4-12k-20260915-r1",
    "train.grpo-lfm26-adaptive-grpo-50-c32-20260920f",
    "train.grpo-lfm26-vortex-v2-agentic-20260923-r1",
    "train.grpo-lfm26-vortex-v3-lr2e4-20260923-r1",
)


def compact_trace(trace: dict[str, Any]) -> dict[str, Any] | None:
    metadata = trace.get("metadata") or {}
    payload = trace.get("payload") or {}
    info = payload.get("info") or {}
    group_id = info.get("posttrain_prompt_group_id")
    task_id = metadata.get("example_id") or info.get("example_id")
    domain = metadata.get("domain") or (info.get("task_facets") or {}).get("domain")
    reward = metadata.get("reward")
    step = metadata.get("optimizer_step")
    if not group_id or not task_id or not domain or reward is None or step is None:
        return None
    return {
        "group_id": str(group_id),
        "task_id": str(task_id),
        "class": str(domain),
        "step": int(step),
        "reward": float(reward),
        "input_tokens": int(payload.get("input_tokens") or 0),
        "output_tokens": int(payload.get("completion_tokens") or 0),
        "truncated": bool(metadata.get("is_truncated")),
        "error": bool(metadata.get("has_error")),
    }


def group_traces(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["group_id"]].append(row)
    groups: list[dict[str, Any]] = []
    issues: Counter[str] = Counter()
    for group_id, members in buckets.items():
        identity = {(m["task_id"], m["class"], m["step"]) for m in members}
        if len(identity) != 1:
            issues["mixed_identity_groups"] += 1
            continue
        rewards = [m["reward"] for m in members]
        if not all(math.isfinite(x) for x in rewards):
            issues["nonfinite_reward_groups"] += 1
            continue
        task_id, domain, step = next(iter(identity))
        groups.append(
            {
                "group_id": group_id,
                "task_id": task_id,
                "class": domain,
                "step": step,
                "size": len(members),
                "rewards": rewards,
                "reward_mean": statistics.mean(rewards),
                "reward_variance": statistics.pvariance(rewards),
                "useful": int(max(rewards) - min(rewards) > 1e-9),
                "input_tokens": sum(m["input_tokens"] for m in members),
                "output_tokens": sum(m["output_tokens"] for m in members),
                "truncated_rollouts": sum(m["truncated"] for m in members),
                "error_rollouts": sum(m["error"] for m in members),
            }
        )
    groups.sort(key=lambda x: (x["step"], x["group_id"]))
    issues["singleton_groups"] = sum(g["size"] == 1 for g in groups)
    issues["non_four_groups"] = sum(g["size"] != 4 for g in groups)
    return groups, dict(issues)


def export(path: Path, run_names: tuple[str, ...], page_size: int = 100) -> dict[str, Any]:
    import trackio

    api = trackio.Api(server_url=SERVER)
    available = {run.name: run for run in api.runs(PROJECT)}
    missing = set(run_names) - available.keys()
    if missing:
        raise ValueError(f"missing runs: {sorted(missing)}")
    result: dict[str, Any] = {
        "schema": "automationbench-group-replay-v2",
        "capture_started_at_utc": datetime.now(UTC).isoformat(),
        "source": {"project": PROJECT, "server": SERVER},
        "runs": [],
    }
    for name in run_names:
        run = available[name]
        count = run.trace_count()
        projected: list[dict[str, Any]] = []
        missing_required = 0
        identities: set[str] = set()
        provenance = {}
        for offset in range(0, count, page_size):
            page = run.traces(
                limit=min(page_size, count - offset),
                offset=offset,
                sort="step_asc",
                include_payload=False,
            )
            if not page:
                raise RuntimeError(f"trace pagination stopped at {offset}/{count} for {name}")
            for trace in page:
                identity = str(trace.get("id") or trace.get("external_id"))
                if identity in identities:
                    raise RuntimeError(f"duplicate trace in {name}: {identity}")
                identities.add(identity)
                if not provenance:
                    metadata = trace.get("metadata") or {}
                    provenance = {
                        key: metadata.get(key)
                        for key in ("model", "model_revision", "environment_id", "training_settings_id")
                    }
                row = compact_trace(trace)
                if row is None:
                    missing_required += 1
                else:
                    projected.append(row)
        groups, issues = group_traces(projected)
        run_result = {
            "name": name,
            "provider_run_id": run.id,
            "provenance": provenance,
            "trace_count_at_start": count,
            "trace_count_at_end": run.trace_count(),
            "projected_trace_count": len(projected),
            "missing_required_trace_count": missing_required,
            "group_count": len(groups),
            "issues": issues,
            "groups": groups,
        }
        if run_result["trace_count_at_end"] != count:
            run_result["issues"]["live_count_changed"] = 1
        result["runs"].append(run_result)
        print(f"{name}: {len(projected)}/{count} traces, {len(groups)} groups, {issues}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


@dataclass
class Posterior:
    successes: int = 0
    trials: int = 0

    def add(self, useful: int) -> None:
        self.successes += useful
        self.trials += 1

    def mean(self, alpha: float, beta: float) -> float:
        return (alpha + self.successes) / (alpha + beta + self.trials)


@dataclass
class CostHistory:
    total_tokens: int = 0
    groups: int = 0

    def add(self, group: dict[str, Any]) -> None:
        self.total_tokens += group["input_tokens"] + group["output_tokens"]
        self.groups += 1


def predicted_token_cost(
    task_id: str,
    task_class: str,
    global_cost: CostHistory,
    class_cost: dict[str, CostHistory],
    task_cost: dict[str, CostHistory],
    prior_strength: float = 4.0,
) -> float:
    global_mean = global_cost.total_tokens / global_cost.groups if global_cost.groups else 1.0
    class_state = class_cost[task_class]
    class_mean = (class_state.total_tokens + prior_strength * global_mean) / (class_state.groups + prior_strength)
    task_state = task_cost[task_id]
    return max(1.0, (task_state.total_tokens + prior_strength * class_mean) / (task_state.groups + prior_strength))


def binomial_success_floor(successes: int, attempts: int, reliability: float = 0.99) -> float:
    """Smallest independent-group success rate meeting a completion target."""
    if successes > attempts:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        tail = sum(
            math.comb(attempts, k) * mid**k * (1 - mid) ** (attempts - k) for k in range(successes, attempts + 1)
        )
        if tail >= reliability:
            hi = mid
        else:
            lo = mid
    return hi


def predicted_yield(
    task_id: str,
    task_class: str,
    class_history: dict[str, Posterior],
    task_history: dict[str, Posterior],
    prior_strength: float = 4.0,
) -> float:
    """Same-unit next-group probability for unseen and previously seen tasks."""
    class_posterior = class_history[task_class]
    class_mean = class_posterior.mean(1.0, 1.0)
    task_posterior = task_history[task_id]
    return task_posterior.mean(prior_strength * class_mean, prior_strength * (1 - class_mean))


def sampled_yield(
    task_id: str,
    task_class: str,
    class_history: dict[str, Posterior],
    task_history: dict[str, Posterior],
    rng: random.Random,
    prior_strength: float = 4.0,
) -> float:
    """Posterior sampling permits uncertain previously seen tasks to be rechecked."""
    class_mean = class_history[task_class].mean(1.0, 1.0)
    task = task_history[task_id]
    return rng.betavariate(
        prior_strength * class_mean + task.successes,
        prior_strength * (1 - class_mean) + task.trials - task.successes,
    )


def calibration(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Prequential prediction with an optimizer-step barrier against leakage."""
    class_history: dict[str, Posterior] = defaultdict(Posterior)
    task_history: dict[str, Posterior] = defaultdict(Posterior)
    sq_errors: list[float] = []
    baseline_errors: list[float] = []
    known_errors: list[float] = []
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_step[group["step"]].append(group)
    for step in sorted(by_step):
        # All groups at one optimizer step are scored from previous-step state.
        # Active-sampling refills can observe within-step outcomes, so this is
        # conservative but does not manufacture a stronger information set.
        for group in by_step[step]:
            task_id, task_class, actual = group["task_id"], group["class"], group["useful"]
            known = task_history[task_id].trials > 0
            prediction = predicted_yield(task_id, task_class, class_history, task_history)
            baseline = (1 + sum(p.successes for p in class_history.values())) / (
                2 + sum(p.trials for p in class_history.values())
            )
            sq_errors.append((prediction - actual) ** 2)
            baseline_errors.append((baseline - actual) ** 2)
            if known:
                known_errors.append((prediction - actual) ** 2)
        for group in by_step[step]:
            class_history[group["class"]].add(group["useful"])
            task_history[group["task_id"]].add(group["useful"])
    return {
        "groups": len(groups),
        "known_task_groups": len(known_errors),
        "brier_hierarchical": statistics.mean(sq_errors) if sq_errors else None,
        "brier_global": statistics.mean(baseline_errors) if baseline_errors else None,
        "brier_known_task": statistics.mean(known_errors) if known_errors else None,
    }


def cross_run_calibration(
    source_groups: list[dict[str, Any]],
    target_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score a separate run without looking at any of its outcomes while fitting."""
    class_history: dict[str, Posterior] = defaultdict(Posterior)
    task_history: dict[str, Posterior] = defaultdict(Posterior)
    for group in source_groups:
        class_history[group["class"]].add(group["useful"])
        task_history[group["task_id"]].add(group["useful"])
    errors: list[float] = []
    global_errors: list[float] = []
    shared = 0
    source_global = (1 + sum(p.successes for p in class_history.values())) / (
        2 + sum(p.trials for p in class_history.values())
    )
    for group in target_groups:
        prediction = predicted_yield(group["task_id"], group["class"], class_history, task_history)
        errors.append((prediction - group["useful"]) ** 2)
        global_errors.append((source_global - group["useful"]) ** 2)
        shared += task_history[group["task_id"]].trials > 0
    return {
        "target_groups": len(target_groups),
        "shared_task_groups": shared,
        "brier_hierarchical": statistics.mean(errors) if errors else None,
        "brier_source_global": statistics.mean(global_errors) if global_errors else None,
    }


def observed_profile(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive counts; novelty uses only tasks seen in earlier steps."""
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_step[group["step"]].append(group)
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for step in sorted(by_step):
        for group in by_step[step]:
            novelty = "known" if group["task_id"] in seen else "new"
            counts[f"{novelty}_groups"] += 1
            counts[f"{novelty}_useful"] += group["useful"]
        seen.update(group["task_id"] for group in by_step[step])
    total = counts["new_groups"] + counts["known_groups"]
    return {
        **dict(counts),
        "new_group_fraction": counts["new_groups"] / total if total else None,
        "new_useful_fraction": (counts["new_useful"] / counts["new_groups"] if counts["new_groups"] else None),
        "known_useful_fraction": (counts["known_useful"] / counts["known_groups"] if counts["known_groups"] else None),
    }


def replay(
    groups: list[dict[str, Any]],
    policy: str,
    seed: int,
    *,
    steps: int = 100,
    target_groups: int = 8,
    max_rounds: int = 4,
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_class: dict[str, str] = {}
    for group in groups:
        by_task[group["task_id"]].append(group)
        task_class[group["task_id"]] = group["class"]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_class[group["class"]].append(group)
    fallback = {
        task: [g for g in by_class[task_class[task]] if g["task_id"] != task] or by_task[task] for task in by_task
    }
    tasks = sorted(by_task)
    if not tasks:
        raise ValueError("no replayable groups")
    rng = random.Random(seed)
    class_history: dict[str, Posterior] = defaultdict(Posterior)
    task_history: dict[str, Posterior] = defaultdict(Posterior)
    global_cost = CostHistory()
    class_cost: dict[str, CostHistory] = defaultdict(CostHistory)
    task_cost: dict[str, CostHistory] = defaultdict(CostHistory)
    visits: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    completion_floor = binomial_success_floor(target_groups, target_groups * max_rounds)
    for _step in range(steps):
        selected_this_step: set[str] = set()
        retained = 0
        for _round in range(max_rounds):
            needed = target_groups - retained
            if needed <= 0:
                break
            eligible = [task for task in tasks if task not in selected_this_step]
            if len(eligible) < needed:
                counts["insufficient_distinct_tasks"] += 1
                break
            if policy == "uniform":
                chosen = rng.sample(eligible, needed)
            elif policy in {
                "posterior_mean",
                "posterior_sample",
                "posterior_sample_cost",
                "posterior_sample_cost_guarded",
            }:
                scored = []
                for task in eligible:
                    mean = predicted_yield(task, task_class[task], class_history, task_history)
                    if policy == "posterior_mean":
                        value = mean
                        value += rng.uniform(-1e-8, 1e-8)
                    else:
                        value = sampled_yield(task, task_class[task], class_history, task_history, rng)
                    if policy in {"posterior_sample_cost", "posterior_sample_cost_guarded"}:
                        value /= predicted_token_cost(task, task_class[task], global_cost, class_cost, task_cost)
                    scored.append((value, task, mean))
                if policy == "posterior_sample_cost_guarded":
                    feasible = [item for item in scored if item[2] >= completion_floor]
                    if len(feasible) >= needed:
                        scored = feasible
                    else:
                        # The chance target is infeasible from current beliefs;
                        # prioritize usefulness rather than a cheap failed step.
                        scored.sort(key=lambda item: item[2], reverse=True)
                        scored = scored[:needed]
                scored.sort(reverse=True)
                chosen = [task for _, task, _ in scored[:needed]]
            else:
                raise ValueError(f"unknown policy: {policy}")
            selected_this_step.update(chosen)
            for task in chosen:
                # Common random numbers: the same task visit in both policies
                # reads the same empirical draw, independent of selection order.
                digest = hashlib.sha256(f"{seed}:{task}:{visits[task]}".encode()).digest()
                # One observed group must not become an endlessly certain task
                # oracle. Borrow from peer tasks in the same class when source
                # support is thin; this is an explicit modeling assumption.
                support = len(by_task[task])
                own_share = support / (support + 4)
                pool = by_task[task] if int.from_bytes(digest[:8], "big") / 2**64 < own_share else fallback[task]
                draw = int.from_bytes(digest[8:16], "big") % len(pool)
                outcome = pool[draw]
                novelty = "new" if task_history[task].trials == 0 else "reuse"
                counts[f"{novelty}_candidates"] += 1
                counts["candidate_groups"] += 1
                counts["input_tokens"] += outcome["input_tokens"]
                counts["output_tokens"] += outcome["output_tokens"]
                counts["useful_groups"] += outcome["useful"]
                retained += outcome["useful"]
                class_history[task_class[task]].add(outcome["useful"])
                task_history[task].add(outcome["useful"])
                global_cost.add(outcome)
                class_cost[task_class[task]].add(outcome)
                task_cost[task].add(outcome)
                visits[task] += 1
        if retained >= target_groups:
            counts["completed_steps"] += 1
    return {
        "policy": policy,
        "seed": seed,
        "supported_tasks": len(tasks),
        **dict(counts),
        "candidate_groups": counts["candidate_groups"],
        "useful_groups": counts["useful_groups"],
        "completed_steps": counts["completed_steps"],
        "new_candidates": counts["new_candidates"],
        "reuse_candidates": counts["reuse_candidates"],
        "input_tokens": counts["input_tokens"],
        "output_tokens": counts["output_tokens"],
        "useful_fraction": (
            counts["useful_groups"] / counts["candidate_groups"] if counts["candidate_groups"] else 0.0
        ),
        "new_fraction": (counts["new_candidates"] / counts["candidate_groups"] if counts["candidate_groups"] else 0.0),
        "candidates_per_completed_step": (
            counts["candidate_groups"] / counts["completed_steps"] if counts["completed_steps"] else None
        ),
    }


def simulate(
    snapshot: dict[str, Any],
    seeds: int = 20,
    *,
    exclude_truncated: bool = False,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema": "automationbench-replay-results-v1",
        "runs": [],
        "cross_run_calibration": [],
        "exclude_truncated": exclude_truncated,
    }
    complete_by_run = {
        source["name"]: [
            g for g in source["groups"] if g["size"] == 4 and (not exclude_truncated or g["truncated_rollouts"] == 0)
        ]
        for source in snapshot["runs"]
    }
    for source in snapshot["runs"]:
        groups = complete_by_run[source["name"]]
        supported_tasks = len({g["task_id"] for g in groups})
        replayable = supported_tasks >= 8 * 4
        summaries = {}
        paired_comparison = None
        if replayable:
            results = [
                replay(groups, policy, seed)
                for seed in range(seeds)
                for policy in (
                    "uniform",
                    "posterior_mean",
                    "posterior_sample",
                    "posterior_sample_cost",
                    "posterior_sample_cost_guarded",
                )
            ]
            for policy in (
                "uniform",
                "posterior_mean",
                "posterior_sample",
                "posterior_sample_cost",
                "posterior_sample_cost_guarded",
            ):
                subset = [r for r in results if r["policy"] == policy]
                summaries[policy] = {
                    key: statistics.mean(r[key] for r in subset)
                    for key in (
                        "candidate_groups",
                        "useful_groups",
                        "completed_steps",
                        "useful_fraction",
                        "new_fraction",
                        "input_tokens",
                        "output_tokens",
                        "candidates_per_completed_step",
                    )
                    if all(r[key] is not None for r in subset)
                }
            by_policy_seed = {(r["policy"], r["seed"]): r for r in results}
            reductions = []
            for seed in range(seeds):
                baseline = by_policy_seed[("uniform", seed)]["candidates_per_completed_step"]
                candidate = by_policy_seed[("posterior_sample", seed)]["candidates_per_completed_step"]
                if baseline and candidate is not None:
                    reductions.append(1 - candidate / baseline)
            if reductions:
                ordered = sorted(reductions)
                paired_comparison = {
                    "seed_count": len(reductions),
                    "mean_candidate_per_completed_step_reduction": statistics.mean(reductions),
                    "min_seed_reduction": ordered[0],
                    "max_seed_reduction": ordered[-1],
                }
        output["runs"].append(
            {
                "name": source["name"],
                "observed_groups": len(groups),
                "excluded_incomplete_groups": sum(g["size"] != 4 for g in source["groups"]),
                "excluded_truncated_groups": sum(
                    g["size"] == 4 and g["truncated_rollouts"] > 0 for g in source["groups"]
                )
                if exclude_truncated
                else 0,
                "observed_tasks": len({g["task_id"] for g in groups}),
                "observed_classes": len({g["class"] for g in groups}),
                "replayable_full_rounds": replayable,
                "observed_useful_fraction": statistics.mean(g["useful"] for g in groups),
                "group_sizes": dict(Counter(str(g["size"]) for g in groups)),
                "observed_profile": observed_profile(groups),
                "calibration": calibration(groups),
                "policies": summaries,
                "paired_comparison": paired_comparison,
            }
        )
    for source in snapshot["runs"]:
        for target in snapshot["runs"]:
            if source["name"] == target["name"]:
                continue
            output["cross_run_calibration"].append(
                {
                    "source": source["name"],
                    "target": target["name"],
                    **cross_run_calibration(complete_by_run[source["name"]], complete_by_run[target["name"]]),
                }
            )
    output["method_limits"] = [
        "Only tasks selected in the source run are replayable; no unseen-task counterfactuals exist.",
        "Empirical outcomes are sampled with replacement; tasks with little support borrow outcomes from peers in their class at weight 4/(task observations + 4).",
        "Outcome distributions stay stationary across 100 simulated steps; model learning, task difficulty drift, and policy feedback are not modeled.",
        "Input plus output tokens are a work proxy, not measured GPU time.",
        "Runs remain separate because model policy and rollout configuration changed.",
        "Groups with fewer than four observed rollouts are excluded from fitted probabilities and replay, not labeled as zero-variance failures.",
        "This does not replay the production VORTEX controller or a fixed 20-percent exploration quota; it compares candidate selection rules on empirical outcome pools.",
        "The cost guard uses a binomial independent-group approximation and does not guarantee the intended completion probability under correlated task outcomes.",
    ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--run", action="append", dest="runs")
    export_parser.add_argument("--page-size", type=int, default=100)
    simulate_parser = commands.add_parser("simulate")
    simulate_parser.add_argument("--input", type=Path, required=True)
    simulate_parser.add_argument("--output", type=Path, required=True)
    simulate_parser.add_argument("--seeds", type=int, default=20)
    simulate_parser.add_argument("--exclude-truncated", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        export(args.output, tuple(args.runs or DEFAULT_RUNS), args.page_size)
    else:
        snapshot = json.loads(args.input.read_text())
        result = simulate(snapshot, args.seeds, exclude_truncated=args.exclude_truncated)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        for item in result["runs"]:
            print(item["name"])
            print(
                json.dumps(
                    {
                        "support": item["observed_groups"],
                        "calibration": item["calibration"],
                        "policies": item["policies"],
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
