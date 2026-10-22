"""Source readiness of the protected-set-clear training pool, without model calls."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(pool, protected):
    from automationbench_v1.contracts.loader import load_task_contract
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskConfig, AutomationBenchTaskset
    from automationbench_v1.world_codec import load_world
    from self_feedback_signals import authority_errors

    names = pool["task_names"]
    if len(names) != 124 or len(set(names)) != 124:
        raise ValueError("expected the frozen124pool")
    candidates = [n for n in names if n not in protected["reasons"]]
    rows = []
    for name in candidates:
        try:
            [task] = AutomationBenchTaskset(AutomationBenchConfig(domains=[name.split(".")[0]], task_names=[name],
                task=AutomationBenchTaskConfig(toolset="limited_zapier", capture_actions=True,
                    deterministic_world=True, world_time_context_if_declared=True, strict_tool_arguments=True))).load()
            data = task.data.model_dump(mode="json")
            state = copy.deepcopy(data["initial_state"])
            load_world(state)
            contract = load_task_contract(name).model_dump(mode="json", exclude_none=True)
            errors = authority_errors(contract, state, data["prompt"])
            rows.append({"task": name, "source_qualified": not errors, "authority_errors": errors,
                         "tools": data["zapier_tools"], "declared_time": data["initial_state"].get("meta", {}).get("current_time")})
        except Exception as error:
            rows.append({"task": name, "source_qualified": False, "error": type(error).__name__, "message": str(error)})
    admitted = [r["task"] for r in rows if r["source_qualified"]]
    return {"pool_tasks": 124, "protected_tasks": {n: protected["reasons"][n] for n in names if n in protected["reasons"]},
            "candidate_tasks": candidates, "source_qualified_tasks": admitted, "rows": rows,
            "category_counts": dict(Counter(n.split(".")[0] for n in admitted)),
            "protected_source_hashes": protected["source_hashes"], "optimizer_updates": 0, "model_calls": 0,
            "scope": "Constructor/manifest/input readiness only. Native execution, task access and guidance coverage remain gates. No selection from student rewards."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--protected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(json.loads(args.pool.read_text()), json.loads(args.protected.read_text()))
    result["pool_sha256"] = hashlib.sha256(args.pool.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidates": len(result["candidate_tasks"]), "source_qualified": len(result["source_qualified_tasks"]),
                      "category_counts": result["category_counts"]}))
