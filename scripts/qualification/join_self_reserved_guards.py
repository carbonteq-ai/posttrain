"""Join strict native initial/final assertion evidence to the reserved audit."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analyze_self_pilot import paired_bootstrap


def join(audit, guards):
    expected = audit["candidates"]
    phases = {p["phase"]: p for p in guards["phases"]}
    if len(phases) != len(expected):
        raise ValueError("guard populations do not match completed candidates")
    output, rates = [], {}
    for step, candidate in enumerate(expected):
        phase = phases[f"step-{step:04d}/reserve_transfer/native/traces.jsonl"]
        if phase["native_sha256"] != candidate["native_sha256"] or phase["totals"]["episodes"] != 48:
            raise ValueError("guard audit uses a different native population")
        groups = defaultdict(list)
        for episode in phase["episodes"]:
            groups[episode["task"]].append(episode)
        if set(groups) != set(audit["population"]["tasks"]) or any(len(v) != 2 for v in groups.values()):
            raise ValueError("guard task/repetition coverage differs")
        native_rates, missing = {"target_fraction": {}, "preservation_violation": {}}, []
        details = {}
        for name, rows in groups.items():
            unknown = sum(len(r["unknown"]) for r in rows)
            details[name] = {"initially_unsatisfied_checks": sum(r["initially_unsatisfied_checks"] for r in rows),
                             "newly_satisfied_checks": sum(r["newly_satisfied_checks"] for r in rows),
                             "preservation_checks": sum(r["preservation_checks"] for r in rows),
                             "broken_preservation_checks": sum(len(r["broken"]) for r in rows),
                             "unknown_initial_checks": unknown}
            if unknown:
                missing.append(name)
                continue
            if all(r["initially_unsatisfied_checks"] > 0 for r in rows):
                native_rates["target_fraction"][name] = sum(
                    r["newly_satisfied_checks"] / r["initially_unsatisfied_checks"] for r in rows) / 2
            if all(r["preservation_checks"] > 0 for r in rows):
                native_rates["preservation_violation"][name] = sum(bool(r["broken"]) for r in rows) / 2
        rates[candidate["candidate"]] = native_rates
        output.append({"candidate": candidate["candidate"], "totals": phase["totals"], "native_sha256": phase["native_sha256"],
                       "violation_episode_rate_when_applicable": phase["preservation_violation_episode_rate_when_applicable"],
                       "task_mean_target_fraction": sum(native_rates["target_fraction"].values()) / len(native_rates["target_fraction"])
                       if native_rates["target_fraction"] else None,
                       "unknown_applicability_tasks": missing, "task_details": details})
    primary = audit["selection"]["primary"]
    comparisons = {}
    for metric in ["target_fraction", "preservation_violation"]:
        before, after = rates["fresh_base"][metric], rates[primary][metric]
        common = sorted(before.keys() & after.keys())
        comparisons[metric] = {"applicable_complete_tasks": common,
                               "excluded_or_not_applicable": sorted(set(audit["population"]["tasks"]) - set(common)),
                               "paired": paired_bootstrap({k: before[k] for k in common}, {k: after[k] for k in common})
                               if common else None}
    return {"candidates": output, "primary_vs_base": comparisons,
            "scope": "Read-only native assertion diagnostics, not replacement scores. Target fractions exclude initially passing checks. "
                     "Violation rates apply only to episodes with known initially passing non-excluded checks. "
                     "No applicable checks is undefined, not harm-free. Unknown applicability remains explicit. "
                     "Preservation constraints are not a comprehensive independently labelled harm taxonomy."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ["audit", "guards", "output"]:
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    result = join(json.loads(args.audit.read_text()), json.loads(args.guards.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
