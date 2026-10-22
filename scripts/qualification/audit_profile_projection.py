"""Compare old/new deterministic worlds without rewriting task fixtures."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def audit(names):
    from automationbench.schema.world import WorldState
    from automationbench_v1.simulation import simulated_setup, world_clock_base
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskset
    from automationbench_v1.world_codec import dump_world

    rows = []
    for name in names:
        [task] = AutomationBenchTaskset(AutomationBenchConfig(domains=[name.split(".")[0]], task_names=[name])).load()
        initial = copy.deepcopy(task.data.initial_state)
        before = digest(initial)
        with simulated_setup(initial, world_clock_base(initial)):
            world = WorldState.model_validate(initial)
        rows.append({"task": name, "source_before_sha256": before, "source_after_sha256": digest(initial),
                     "canonical_world_sha256": digest(dump_world(world)), "assertions_sha256": digest(task.data.assertions)})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(json.loads(args.source_audit.read_text())["candidate_tasks"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tasks": len(result)}))
