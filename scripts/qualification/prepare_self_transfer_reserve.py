"""Freeze a domain-balanced evaluation cohort outside pilot/development tasks."""

import argparse
import hashlib
import json
from pathlib import Path

QUOTAS = {"simple": 12, "operations": 3, "finance": 2, "hr": 2,
          "marketing": 2, "support": 2, "sales": 1}
SALT = "self-adaptive-reserve24-20261017-v1"


def select(protocol, pool, heldout):
    pilot = set(protocol["cohorts"]["pilot"]["task_names"])
    development = set(protocol["cohorts"]["development"]["task_names"])
    eligible = set(protocol["cohorts"]["new_update_eligible"]["task_names"])
    source, final = set(pool["task_names"]), set(heldout["task_names"])
    if (len(source) != 124 or len(pool["task_names"]) != 124 or len(pilot) != 32
            or len(development) != 24 or len(final) != 62 or pilot & development
            or source & final or eligible != source - development or not pilot <= eligible):
        raise ValueError("invalid source pool/cohort separation")
    remaining = eligible - pilot
    if len(remaining) != 68:
        raise ValueError("expected68 tasks outside original pilot/development")
    names = []
    for domain, count in QUOTAS.items():
        candidates = sorted((n for n in remaining if n.split(".")[0] == domain),
                            key=lambda n: hashlib.sha256(f"{SALT}:{n}".encode()).hexdigest())
        if len(candidates) < count:
            raise ValueError("insufficient domain population; no silent fallback")
        names.extend(candidates[:count])
    return sorted(names)


def main(args):
    protocol, pool, heldout = [json.loads(p.read_text()) for p in [args.protocol, args.pool, args.heldout]]
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
              [("protocol", args.protocol), ("pool", args.pool), ("heldout", args.heldout)]}
    if (hashes["pool"] != protocol["source_files"]["new_training"]["sha256"]
            or hashes["heldout"] != protocol["source_files"]["final_test"]["sha256"]):
        raise ValueError("source fixtures changed")
    names = select(protocol, pool, heldout)
    result = {"status": "reserved; no evaluations launched", "task_names": names,
              "sorted_names_sha256": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest(),
              "domain_quotas": QUOTAS, "selection_salt": SALT, "source_sha256": hashes,
              "environment_revision": protocol["environment_revision"],
              "historical_novelty_audit_cutoff": protocol["novelty_audit_cutoff"],
              "scope": "Excluded from original pilot32/development24 and adaptive repair updates. "
                       "Historical use outside this campaign after the audit cutoff is not reverified. "
                       "Research transfer diagnostic, not heldout62. Never use these tasks for updates.",
              "selection_timing": "Before inspecting adaptive06 native outcomes or any candidate outcomes on this reserve.",
              "primary_selection": "Record the primary method/checkpoint from repair/retention results before reading reserve outcomes; do not select best by reserve results."}
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["protocol", "pool", "heldout", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    main(parser.parse_args())
