"""Read-only audit of native initially satisfied assertion preservation."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def audit(root, check_initial):
    initial_cache, phases = {}, []
    for path in sorted(root.glob("step-*/*/native/traces.jsonl")):
        episodes = [json.loads(line) for line in path.read_text().splitlines() if line]
        totals, results = Counter(), []
        for episode in episodes:
            data = episode["task"]["data"]
            name, state, specs = data["name"], data["initial_state"], data["assertions"]
            state_hash = hashlib.sha256(canonical(state).encode()).hexdigest()
            key = canonical([state_hash, specs])
            if key not in initial_cache:
                values = []
                for spec in specs:
                    try:
                        value, error = bool(check_initial(state, spec)), None
                    except Exception as exc:
                        value, error = None, f"{type(exc).__name__}: {exc}"
                    values.append({"initially_passed": value, "error": error})
                initial_cache[key] = values
            native = episode["traces"][0].get("info", {}).get("automationbench", {}).get("assertions")
            if native is None or len(native) != len(specs):
                raise ValueError("native assertion coverage missing")
            guards, broken, unknown, gained, targets = [], [], [], [], []
            for index, (spec, initial, final) in enumerate(zip(specs, initial_cache[key], native, strict=True)):
                if canonical(spec) != canonical({"type": final["type"], **final["params"]}):
                    raise ValueError("assertion identities/order differ from source task")
                if spec.get("scored") is False or spec.get("excluded") is True:
                    totals["author_excluded_checks"] += 1
                    continue
                detail = {"index": index, "spec": spec, "final_passed": final["passed"]}
                if initial["initially_passed"] is None:
                    unknown.append({**detail, "initial_error": initial["error"]})
                elif initial["initially_passed"]:
                    guards.append(detail)
                    if not final["passed"]:
                        broken.append(detail)
                else:
                    targets.append(detail)
                    if final["passed"]:
                        gained.append(detail)
            totals.update(episodes=1, preservation_checks=len(guards), broken_preservation_checks=len(broken),
                          episodes_with_preservation_checks=bool(guards), episodes_breaking_preservation=bool(broken),
                          unknown_initial_checks=len(unknown), initially_unsatisfied_checks=len(targets),
                          newly_satisfied_checks=len(gained))
            results.append({"episode_id": episode["id"], "task": name, "initial_state_sha256": state_hash,
                            "preservation_checks": len(guards), "broken": broken, "unknown": unknown,
                            "initially_unsatisfied_checks": len(targets), "newly_satisfied_checks": len(gained),
                            "native_assertions_scored": episode["traces"][0]["metrics"].get("assertions_scored"),
                            "native_assertions_excluded": episode["traces"][0]["metrics"].get("assertions_excluded")})
        applicable = totals["episodes_with_preservation_checks"]
        phases.append({"phase": str(path.relative_to(root)),
                       "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "totals": dict(totals),
                       "preservation_violation_episode_rate_when_applicable":
                       totals["episodes_breaking_preservation"] / applicable if applicable else None,
                       "episodes": results})
    if not phases:
        raise ValueError("no native populations found")
    return {"phases": phases, "initial_state_variants": len(initial_cache),
            "scope": "Initially satisfied, non-author-excluded native checks are preservation constraints. "
                     "They are not a complete independently labelled harm taxonomy. Unknown checks stay unknown. "
                     "Native partial credit excludes held pre-satisfied checks but penalizes broken ones; its denominator can change. "
                     "No rescoring, source changes, external model calls or replacement rewards."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["AUTOMATIONBENCH_STRICT_ASSERTIONS"] = "1"
    from automationbench.rubric import AssertionRegistry, WorldState

    def check(initial, spec):
        return AssertionRegistry.check(WorldState.model_validate(initial), dict(spec))

    args.output.write_text(json.dumps(audit(args.root, check), indent=2) + "\n")
