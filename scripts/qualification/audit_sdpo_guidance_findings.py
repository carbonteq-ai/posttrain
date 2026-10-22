"""Audit declared training goal/harm evidence without inferring absent safety."""

import argparse
import json
from collections import Counter
from pathlib import Path

from sdpo_guidance_context import digest, native_findings


def audit(episodes, materials, normalization=None):
    sources = {row["task"]: row["material"] for row in materials["rows"]}
    result = []
    identities = set()
    for episode in episodes:
        if episode["id"] in identities:
            raise ValueError("duplicate native episode identity")
        identities.add(episode["id"])
        task = episode["task"]["data"]
        source = sources[task["name"]]
        expected = source["native_data"]["initial_state"]
        if normalization is not None:
            if (not normalization["qualified"] or normalization["mismatch"]
                    or normalization["raw_source_hashes"][task["name"]] != digest(expected)):
                raise ValueError("normalization source binding differs")
            expected = normalization["normalized_states"][task["name"]]
            if normalization["canonical_hashes"][task["name"]] != digest(expected):
                raise ValueError("normalization canonical binding differs")
        if digest(task["initial_state"]) != digest(expected):
            raise ValueError("manifest audit initial-state binding differs")
        user = "\n\n".join(m["content"] for m in task["prompt"] if m["role"] == "user")
        if digest(user) != digest(source["user_request"]):
            raise ValueError("manifest audit public-request binding differs")
        checks = [c for c in source["goal_manifest"]["checks"] if c["role"] in ("goal", "harm")]
        evidence = native_findings(episode, {
            "check_roles": {c["check_id"]: c["role"] for c in checks},
            "check_signals": {c["check_id"]: c["signal_id"] for c in checks}})
        invalid = Counter()
        for item in evidence["summary"]:
            invalid.update(item["other_status_counts"])
        result.append({"episode_id": episode["id"], "task": task["name"],
            "declared_goals": sum(c["role"] == "goal" for c in checks),
            "declared_harms": sum(c["role"] == "harm" for c in checks),
            "observed_goal_zero_instances": sum(r["role"] == "goal" and r["status"] == "valid"
                                                and r["value"] == 0 for r in evidence["findings"]),
            "observed_harm_one_instances": sum(r["role"] == "harm" and r["status"] == "valid"
                                               and r["value"] == 1 for r in evidence["findings"]),
            "missing_signal_ids": evidence["missing_signal_ids"],
            "other_status_counts": dict(invalid), "signals": evidence["summary"],
            "assessment_finalization_states": [t.get("assessment_finalization_state")
                                                for t in episode["traces"]],
            "harm_free": None,
            "harm_free_note": "This audit does not prove complete closed guard scope; no findings is not safety."})
    return {"episodes": result, "unique_episodes": len(identities),
        "missing_signal_episode_count": sum(bool(r["missing_signal_ids"]) for r in result),
        "episodes_with_observed_goal_failures": sum(r["observed_goal_zero_instances"] > 0 for r in result),
        "episodes_with_observed_harm_violations": sum(r["observed_harm_one_instances"] > 0 for r in result),
        "scope": "Source-bound deduplicated native training findings, not causal per-action credit or complete safety certification."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization", type=Path)
    args = parser.parse_args()
    episodes = [json.loads(line) for line in args.traces.read_text().splitlines() if line]
    normalization = json.loads(args.normalization.read_text()) if args.normalization else None
    args.output.write_text(json.dumps(audit(episodes, json.loads(args.materials.read_text()), normalization), indent=2) + "\n")
