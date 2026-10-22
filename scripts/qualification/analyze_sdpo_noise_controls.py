"""Candidate descriptive repeated-control analysis; does not qualify a runtime.

This is analysis only. Switching the campaign to this exploratory protocol and
renewing expired windows requires the pending user decisions. No GPU launcher
uses this module.
"""

from analyze_sdpo_pool124 import bounded_scores
from analyze_self_pilot import paired_bootstrap
from sdpo_agent_config import MODEL_REVISION


def compare(base_a, base_b, student, names, *, bootstrap_repeats=10000):
    """Keep zero-update drift, learner deltas and missing-data bounds separate.

    Callers must derive runtime/adapter identities from actual source, serving
    inputs and native generation receipts. This function validates equality and
    native populations; it is not a checkpoint or provenance verifier.
    """
    populations = [base_a, base_b, student]
    if len(names) != len(set(names)) or len(names) != 62:
        raise ValueError("frozen heldout62 identities required")
    for key in ("runtime_key", "model_revision"):
        if not populations[0].get(key) or any(p.get(key) != populations[0][key] for p in populations):
            raise ValueError("unmatched model/runtime controls")
    if base_a["model_revision"] != MODEL_REVISION:
        raise ValueError("wrong pinned4B model revision")
    if (base_a["adapter_sha256"] != base_b["adapter_sha256"]
            or [p["policy_version"] for p in populations] != [0, 0, 10]):
        raise ValueError("controls must be the same untrained weights and student must be step10")
    summaries, bounds, seen = [], [], set()
    for population in populations:
        episodes = population["episodes"]
        ids = {e["id"] for e in episodes}
        if len(episodes) != 310 or len(ids) != 310 or seen & ids:
            raise ValueError("three independent complete native episode populations required")
        seen |= ids
        summary, bounded = bounded_scores(episodes, names, 5)
        summaries.append(summary)
        bounds.append(bounded)
    pairs = [("zero_update_base_b_minus_a", 0, 1),
             ("student_minus_base_a", 0, 2), ("student_minus_base_b", 1, 2)]
    metrics = {}
    keys = set().union(*(s["metrics"] for s in summaries))
    for key in sorted(keys):
        available = [s["metrics"].get(key) for s in summaries]
        if not all(x is not None for x in available):
            metrics[key] = {"status": "metric_missing_in_a_population", "comparisons": None}
            continue
        means = [x["complete_task_means"] for x in available]
        joint = set(means[0]) & set(means[1]) & set(means[2])
        controls = set(means[0]) & set(means[1])
        control_interval = paired_bootstrap({n: means[0][n] for n in controls},
            {n: means[1][n] for n in controls}, repeats=bootstrap_repeats) if controls else None
        if control_interval:
            control_interval["unit"] = "task; all five replicas; control-pair completeness independent of student missingness"
        comparisons = {}
        for label, before, after in pairs:
            interval = paired_bootstrap({n: means[before][n] for n in joint},
                {n: means[after][n] for n in joint}, repeats=bootstrap_repeats) if joint else None
            if interval:
                interval["unit"] = "task; preserve all five replicas, using the same joint-complete tasks in all three comparisons"
            gain_bounds = None
            if all(key in b for b in bounds):
                left = bounds[before][key]["identification_bounds"]
                right = bounds[after][key]["identification_bounds"]
                gain_bounds = [right[0] - left[1], right[1] - left[0]]
            comparisons[label] = {"joint_complete_task_sensitivity": interval,
                                  "full_population_gain_identification_bounds": gain_bounds}
        metrics[key] = {"joint_complete_tasks": len(joint), "expected_tasks": 62,
                        "zero_update_control_pair": control_interval,
                        "control_pair_excluded_tasks": sorted(set(names) - controls),
                        "excluded_tasks": sorted(set(names) - joint), "comparisons": comparisons}
    return {"protocol": "candidate-matched-noise-controls-v1", "unique_native_episodes": len(seen),
            "summaries": summaries, "absolute_missing_score_identification_bounds": bounds,
            "metrics": metrics, "student_adapter_changed": student["adapter_sha256"] != base_a["adapter_sha256"],
            "automatic_learning_verdict": None, "strict_runtime_qualification_granted": False,
            "scope": "Descriptive candidate analysis, not execution authorization or promotion of failed runtime gates. "
                     "Zero-update drift includes environment and inference variation; it is not a causal correction "
                     "to subtract from learning. Absolute intervals, paired deltas and identification bounds differ. "
                     "Native guard coverage, checkpoint lineage and qualitative tool correctness require separate audits. "
                     "Reused controls across studies are references, not new independent episodes."}
