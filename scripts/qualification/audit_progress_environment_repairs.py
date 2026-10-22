"""CPU proof of ten-task clock visibility and explicit strict tool admission."""

import argparse
import copy
import json
from pathlib import Path


def audit(names):
    from automationbench_v1.contracts.loader import load_task_contract
    from automationbench_v1.taskset import (
        AutomationBenchConfig,
        AutomationBenchTask,
        AutomationBenchTaskConfig,
        AutomationBenchTaskset,
    )
    from automationbench_v1.world_codec import load_world
    from progress_teacher_context import digest
    from self_feedback_signals import authority_errors

    rows = []
    for name in names:
        common = dict(toolset="limited_zapier", turn_budget=8, capture_actions=True, deterministic_world=True)
        original = AutomationBenchTaskConfig(**common)
        repaired = AutomationBenchTaskConfig(**common, world_time_context_if_declared=True, strict_tool_arguments=True)
        records = []
        for config in [original, repaired]:
            [task] = AutomationBenchTaskset(AutomationBenchConfig(
                domains=[name.split(".")[0]], task_names=[name], task=config)).load()
            records.append(task.data.model_dump(mode="json"))
        old, new = records
        for key in ["initial_state", "assertions", "zapier_tools", "name"]:
            if old[key] != new[key]:
                raise ValueError(f"repair changed {key}: {name}")
        old_users = [p for p in old["prompt"] if p["role"] == "user"]
        if old_users != [p for p in new["prompt"] if p["role"] == "user"]:
            raise ValueError("repair changed public user request")
        declared = new["initial_state"].get("meta", {}).get("current_time")
        if declared and declared not in new["prompt"][0]["content"]:
            raise ValueError("declared clock still hidden")
        if not declared and "Simulation context:" in new["prompt"][0]["content"]:
            raise ValueError("invented a task clock")
        state = copy.deepcopy(new["initial_state"])
        load_world(state)
        contract = load_task_contract(name).model_dump(mode="json", exclude_none=True)
        errors = authority_errors(contract, state, new["prompt"])
        if errors:
            raise ValueError(f"repaired native task authority failed: {name}: {errors}")
        [tools] = AutomationBenchTask.toolsets(repaired.model_copy(update={"allowed_tools": tuple(new["zapier_tools"])}))
        if tools.config.strict_tool_arguments is not True:
            raise ValueError("strict config lost before toolset")
        rows.append({"task": name, "declared_time": declared,
            "clock_exposed": bool(declared), "state_unchanged": True, "assertions_unchanged": True,
            "user_request_unchanged": True, "tool_inventory_unchanged": True,
            "strict_tool_arguments": True, "manifest_authority_errors": [],
            "old_prompt_sha256": digest(old["prompt"]), "new_prompt_sha256": digest(new["prompt"]),
            "initial_state_sha256": digest(new["initial_state"]),
            "assertions_sha256": digest(new["assertions"])})
    return {"qualified": True, "model_calls": 0, "optimizer_updates": 0, "tasks": rows,
            "scope": "CPU source/binding/clock proof; strict admission has separate real MCP transport tests. "
                     "New public context/runtime profile requires fresh matched model controls; no historical score pooling."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohorts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(json.loads(args.cohorts.read_text())["train"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"qualified": True, "tasks": len(result["tasks"]),
                      "clocks_exposed": sum(r["clock_exposed"] for r in result["tasks"])}))
