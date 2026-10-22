"""Pure schedule and cohort checks for the local multi-round SELF experiment."""

import math
from collections import Counter, defaultdict

from analyze_self_pilot import paired_bootstrap


def schedule(total, per_round, batch):
    if min(total, per_round, batch) < 1 or total % per_round:
        raise ValueError("positive counts and complete collection rounds required")
    return [(start, start + per_round, per_round * batch) for start in range(0, total, per_round)]


def summarize(episodes, expected, rollouts):
    """Keep missing native scores explicit; never turn them into failures or zeros."""
    values = [e.model_dump(mode="json") if hasattr(e, "model_dump") else e for e in episodes]
    counts = Counter(e["task"]["data"]["name"] for e in values)
    if set(counts) != set(expected) or set(counts.values()) != {rollouts}:
        raise ValueError("collection differs from the sealed task/repetition cohort")
    if len({e["id"] for e in values}) != len(values):
        raise ValueError("duplicate native episode IDs")
    metrics, errors, stops = defaultdict(lambda: defaultdict(list)), [], Counter()
    for episode in values:
        name = episode["task"]["data"]["name"]
        for key in ("errors", "credit_errors", "assessment_errors"):
            errors.extend(episode.get(key, []))
        if len(episode["traces"]) != 1:
            raise ValueError("expected one native trace per episode")
        for trace in episode["traces"]:
            errors.extend(trace.get("errors", []))
            stops[trace["stop_condition"]] += 1
            raw = dict(trace.get("metrics", {}))
            raw["partial_credit"] = trace.get("rewards", {}).get("partial_credit", {}).get("score")
            for key, value in raw.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[key][name].append(value)
    result = {}
    for key in ("task_completed_correctly", "partial_credit", "tool_mistakes", "tool_unknown_id"):
        groups = metrics[key]
        complete = {name: sum(v) / rollouts for name, v in groups.items() if len(v) == rollouts}
        result[key] = {"episode_coverage": sum(map(len, groups.values())),
                       "total_episodes": len(values), "complete_task_means": complete,
                       "full_population_task_mean": sum(complete.values()) / len(expected)
                       if len(complete) == len(expected) else None}
    return {"episodes": len(values), "tasks": len(counts), "errors": errors,
            "stop_conditions": dict(stops), "metrics": result,
            "goals_and_harms": "not assessed in this unchanged minimal-feedback setup"}


def comparisons(baseline, current):
    result = {}
    for key, metric in current["metrics"].items():
        before = baseline["metrics"][key]["complete_task_means"]
        after = metric["complete_task_means"]
        common = sorted(set(before) & set(after))
        result[key] = {"scope": "complete-task sensitivity; inspect coverage separately",
                       "excluded_tasks": sorted((set(before) | set(after)) - set(common)),
                       "paired": paired_bootstrap({k: before[k] for k in common},
                                                  {k: after[k] for k in common}) if common else None}
    return result
