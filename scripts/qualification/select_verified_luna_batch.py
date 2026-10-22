"""CPU-only native binding screen before bounded Luna-guided SFT generation."""

import argparse
import json
from collections import Counter
from pathlib import Path


def select(rows, quota):
    from automationbench_v1.contracts.loader import load_task_contract
    from automationbench_v1.limited_tools import selected_tool_definitions
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskConfig, AutomationBenchTaskset
    from self_feedback_signals import authority_errors

    selected, audits = [], []
    counts = Counter()
    # Input is already sorted by category, source quality and deterministic task
    # identity. Never select using generated outcomes or protected evaluation.
    for row in rows:
        name = row["task"]
        domain = name.split(".")[0]
        if counts[domain] >= quota:
            continue
        try:
            task = AutomationBenchTaskset(AutomationBenchConfig(
                domains=[domain], task_names=[name],
                task=AutomationBenchTaskConfig(toolset="limited_zapier", turn_budget=12,
                                              capture_actions=True, deterministic_world=True),
            )).load()[0]
            native = task.data.model_dump(mode="json")
            contract = load_task_contract(name).model_dump(mode="json", exclude_none=True)
            errors = authority_errors(contract, native["initial_state"], native["prompt"])
            if errors:
                raise ValueError(f"manifest authority mismatch: {errors}")
            from luna_ssr_sft import sha

            source = row["source_native"]
            if sha(source["initial_state"]) != sha(native["initial_state"]):
                raise ValueError("reference initial state mismatch")
            if source["assertions"] != native["assertions"]:
                raise ValueError("reference assertions mismatch")
            users = [m["content"] for m in native["prompt"] if m["role"] == "user"]
            if users != row["user_messages"]:
                raise ValueError("reference request mismatch")
            available = {t["function"]["name"] for t in selected_tool_definitions(native["zapier_tools"])}
            if any(a["call"]["name"] not in available for a in row["reference_actions"]):
                raise ValueError("reference uses unavailable tools")
            selected.append({k: v for k, v in row.items() if k != "source_native"})
            counts[domain] += 1
            audits.append({"task": name, "admitted_for_generation": True})
        except Exception as error:
            audits.append({"task": name, "admitted_for_generation": False,
                           "reason": f"{type(error).__name__}: {error}"})
    return {"tasks": selected, "counts": dict(counts), "audit": audits,
            "sft_admitted": False, "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quota", type=int, default=2)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(select(json.loads(args.candidates.read_text())["tasks"], args.quota), stream, indent=2)
