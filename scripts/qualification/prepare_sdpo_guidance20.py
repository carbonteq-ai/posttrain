"""CPU audit of current tasks and historical procedural references, no model calls."""

import argparse
import copy
import hashlib
import inspect
import json
from collections import Counter


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def canonical_data(data, normalizer, source_sha256):
    original, canonical = copy.deepcopy(data), copy.deepcopy(data)
    normalizer(canonical["initial_state"])
    return canonical, {"raw_initial_state_sha256": digest(original["initial_state"]),
        "canonical_initial_state_sha256": digest(canonical["initial_state"]),
        "normalizer_sha256": source_sha256,
        "scope": "Actual pinned WorldState constructor on a deep copy before unchanged manifest digest checks. "
                 "No source/predicate/digest override; native generated-state equality still required."}


def prepare(candidates, *, normalize_sources=False):
    from automationbench_v1.contracts.loader import load_task_contract
    from automationbench_v1.limited_tools import selected_tool_definitions
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskConfig, AutomationBenchTaskset
    from self_feedback_signals import authority_errors

    normalizer, normalizer_sha = None, None
    if normalize_sources:
        from pathlib import Path

        from automationbench_v1.world_codec import load_world

        normalizer = load_world
        normalizer_sha = hashlib.sha256(Path(inspect.getsourcefile(load_world)).read_bytes()).hexdigest()

    admitted, audit = [], []
    for row in candidates["tasks"]:
        name = row["task"]
        try:
            task = AutomationBenchTaskset(AutomationBenchConfig(
                domains=[name.split(".")[0]], task_names=[name],
                task=AutomationBenchTaskConfig(toolset="limited_zapier", turn_budget=8,
                                              capture_actions=True, deterministic_world=True))).load()[0]
            data = task.data.model_dump(mode="json")
            raw_data, projection = copy.deepcopy(data), None
            if normalize_sources:
                data, projection = canonical_data(data, normalizer, normalizer_sha)
            contract = load_task_contract(name).model_dump(mode="json", exclude_none=True)
            errors = authority_errors(contract, data["initial_state"], data["prompt"])
            if errors:
                raise ValueError(f"current manifest binding: {errors}")
            users = [m["content"] for m in data["prompt"] if m["role"] == "user"]
            if users != row["user_messages"]:
                raise ValueError("historical/current user request differs")
            tools = selected_tool_definitions(data["zapier_tools"])
            available = {t["function"]["name"] for t in tools}
            missing = {a["call"]["name"] for a in row["reference_actions"]} - available
            if missing:
                raise ValueError(f"historical procedure uses unavailable tools: {sorted(missing)}")
            historical = copy.deepcopy(row["source_native"]["initial_state"])
            if normalize_sources:
                normalizer(historical)
            binding = {"current_native_state_sha256": digest(data["initial_state"]),
                       "current_manifest_sha256": digest(contract),
                       "historical_state_equal": data["initial_state"] == historical,
                       "historical_assertions_equal": data["assertions"] == row["source_native"]["assertions"],
                       "scope": "Historical procedure only; no current-state or perfect-reference claim"}
            admitted.append({**{k:v for k,v in row.items() if k != "source_native"},
                             "binding": binding,
                             "material": {"task":name,"user_request":"\n\n".join(users),
                                          "tool_definitions":tools,"native_data":data,"goal_manifest":contract,
                                          "reference_actions":row["reference_actions"],"source":row["luna_source"],
                                          **({"raw_native_data": raw_data, "state_projection": projection}
                                             if normalize_sources else {})}})
            audit.append({"task":name,"eligible":True,**binding})
        except Exception as error:
            audit.append({"task":name,"eligible":False,"error":f"{type(error).__name__}: {error}"})
    return {"eligible":admitted,"audit":audit,
            "counts":dict(Counter(r["task"].split(".")[0]for r in admitted)),
            "model_calls":0,"optimizer_updates":0,
            "normalized_sources": normalize_sources}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates")
    parser.add_argument("output")
    parser.add_argument("--normalize-sources", action="store_true")
    args=parser.parse_args()
    from pathlib import Path

    result=prepare(json.loads(Path(args.candidates).read_text()), normalize_sources=args.normalize_sources)
    Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"counts":result["counts"],"eligible":len(result["eligible"]),"screened":len(result["audit"])}))
