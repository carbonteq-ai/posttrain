"""Freeze a tool-diverse historical-procedure cohort without evaluation peeking."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def tools(row):
    return {action["call"]["name"] for action in row["reference_actions"]}


def select(rows, protected, simple_count=11):
    names = [row["task"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("duplicate candidate task")
    eligible = [row for row in rows if row["task"] not in protected]
    chosen = sorted((r for r in eligible if not r["task"].startswith("simple.")), key=lambda r: r["task"])
    if len(chosen) != 9:
        raise ValueError("expected nine audited nonSimple tasks; changed inventory needs review")
    remaining = [r for r in eligible if r["task"].startswith("simple.")]
    seen = set().union(*(tools(r) for r in chosen))
    simple = []
    for _ in range(simple_count):
        if not remaining:
            raise ValueError("insufficient diverse Simple candidates")
        # Native tool diversity, then multi-tool dependencies, then stable name.
        row = min(remaining, key=lambda r: (-len(tools(r) - seen), -len(tools(r)), r["task"]))
        simple.append(row)
        remaining.remove(row)
        seen |= tools(row)
    return chosen + simple


def main(args):
    from luna_ssr_sft import exclusions

    audit = json.loads(args.audit.read_text())
    protected, sources = exclusions(args.project)
    selected = select(audit["eligible"], protected)
    payload = {"train": [r["task"] for r in selected], "development": [], "reserve": [],
        "domain_counts": dict(Counter(r["task"].split(".")[0] for r in selected)),
        "selection": "all nine audited nonSimple plus eleven greedily tool-diverse Simple; no outcome selection",
        "protected_sources": sources, "protected_overlap": [],
        "source_audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "historical_state_equal_count": sum(r["binding"]["historical_state_equal"] for r in selected),
        "warning": "Historical Luna procedures are not current-state answers. Cohort not seven-domain balanced.",
        "rows": selected}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "training-materials.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "cohorts.json").write_text(json.dumps({k: payload[k] for k in ("train", "development", "reserve")}, indent=2) + "\n")
    print(json.dumps({"domains": payload["domain_counts"], "tasks": payload["train"],
                      "historical_state_equal": payload["historical_state_equal_count"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
