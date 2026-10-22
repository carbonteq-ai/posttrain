"""Freeze a task-diverse ten-step schedule without using model outcome scores."""

import hashlib
from collections import Counter


def intended_split(pool, heldout, historical_exclusions):
    """Keep the declared split; historical cohorts are evidence, not exclusions."""
    names = pool["task_names"]
    evaluation = heldout["task_names"]
    if len(names) != 124 or len(set(names)) != 124:
        raise ValueError("expected 124 unique training tasks")
    if len(evaluation) != 62 or len(set(evaluation)) != 62:
        raise ValueError("expected 62 unique heldout tasks")
    if set(names) & set(evaluation):
        raise ValueError("training and heldout62 overlap")
    review = []
    for name in names:
        reasons = historical_exclusions.get("reasons", {}).get(name, [])
        if reasons:
            review.append({
                "task": name, "historical_sources": reasons,
                "decision": "retain_in_training",
                "justification": "The user reaffirmed the 124/62 split. Historical training, evaluation, development or reserve membership alone does not establish a current independent exclusion. Retire overlapping diagnostic cohorts from unseen-generalization claims instead.",
            })
    return {
        "policy": "intended-124-train-62-heldout-v2",
        "reasons": {name: ["authoritative heldout62"] for name in evaluation},
        "source_hashes": historical_exclusions.get("source_hashes", {}),
        "training_task_names": names[:], "heldout_task_names": evaluation[:],
        "additional_exclusions": [], "historical_exclusion_review": review,
        "prior_training_disqualifies": False,
        "future_additional_exclusion_rule": "Require an individual task, concrete current justification and explicit protocol decision; no automatic unions of old fixtures or cohorts.",
        "evaluation_claim_rule": "Overlapping old development/reserve readouts are training diagnostics, not unseen validation. Heldout62 remains separate; do not tune on its outcomes.",
    }


def schedule(source_audit, batch=32, steps=10, seed="progress-corpus-20261020"):
    names = source_audit["candidate_tasks"]
    admitted = source_audit["source_qualified_tasks"]
    if len(names) != len(set(names)) or set(names) != set(admitted):
        raise ValueError("repair every candidate; do not silently drop failing tasks")
    if set(names) & set(source_audit["protected_tasks"]):
        raise ValueError("protected task in training")
    if not 1 <= batch <= len(names) or steps < 1:
        raise ValueError("invalid distinct-task schedule")
    grouped = {}
    for name in names:
        grouped.setdefault(name.split(".")[0], []).append(name)
    for group in grouped.values():
        group.sort(key=lambda n: hashlib.sha256((seed + ":" + n).encode()).hexdigest())
    counts = Counter({domain: len(group) for domain, group in grouped.items()})
    quota = {domain: batch * count // len(names) for domain, count in counts.items()}
    order = sorted(grouped, key=lambda d: (-(batch * counts[d] % len(names)), d))
    for domain in order[:batch - sum(quota.values())]:
        quota[domain] += 1
    if any(not q for q in quota.values()):
        raise ValueError("batch cannot represent all declared categories")
    rounds = []
    for step in range(steps):
        selections = {d: [grouped[d][(step * q + i) % len(grouped[d])] for i in range(q)] for d, q in quota.items()}
        tasks = [selections[d][i] for i in range(max(quota.values())) for d in sorted(selections) if i < len(selections[d])]
        if len(tasks) != batch or len(set(tasks)) != batch:
            raise ValueError("duplicate or incomplete optimizer task population")
        rounds.append({"step": step + 1, "tasks": tasks, "category_counts": dict(Counter(n.split(".")[0] for n in tasks))})
    coverage = Counter(n for r in rounds for n in r["tasks"])
    if set(coverage) != set(names):
        raise ValueError("schedule does not cover the full candidate corpus")
    return {"corpus_tasks": names, "task_count": len(names), "steps": rounds,
            "batch_distinct_tasks": batch, "trajectories_per_task": 2,
            "expected_online_episodes": steps * batch * 2, "task_exposures": dict(coverage),
            "category_quota_per_update": quota, "seed": seed,
            "guidance_and_native_batch_qualified": False,
            "scope": "Source-qualified candidate schedule only; teacher coverage and runtime admission still required. Refill the same task if necessary; no truncation, silent task replacement or protected-set reuse."}
