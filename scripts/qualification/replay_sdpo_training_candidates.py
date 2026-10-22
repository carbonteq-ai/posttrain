"""Replay candidate procedures through pinned native capture on isolated states.

This is a fixture/API compatibility check, not generated student performance or
proof of complete goal/harm coverage. Import the pinned environment only in run.
"""

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def response(value, tool=None):
    value = json.loads(value) if isinstance(value, str) else value
    if isinstance(value, dict) and tool in ("salesforce_query", "salesforce_find_records"):
        if (set(value) == {"error"} and isinstance(value["error"], str)
                or isinstance(value.get("results"), list)
                and type(value.get("count")) is int
                and value["count"] == len(value["results"])
                and set(value) <= {"results", "count", "total_count"}):
            return value
    if not isinstance(value, dict) or not isinstance(value.get("success"), bool):
        raise ValueError("native response lacks explicit success status")
    return value


def succeeded(value):
    """Pinned Salesforce reads return count/results or error, without success."""
    return value["success"] if "success" in value else "error" not in value


def verify_capture(result, envelopes):
    if len(envelopes) != len(result["calls"]):
        raise ValueError("native capture does not cover every replayed call")
    for call, envelope in zip(result["calls"], envelopes, strict=True):
        action = envelope["action"]
        if (action["tool_name"] != call["tool"] or action["occurrence_index"] != call["index"]
                or action["status"] != "returned"
                or response(json.loads(action["result_json"]), call["tool"]) != call["observed"]):
            raise ValueError("native capture action/result differs from replay")
    return {"calls": len(envelopes), "evidence_sha256": digest(envelopes)}


def replay(row, native_data, invoke, state_factory=None, scorer=None):
    """Fail closed on source drift, unexpected errors or an unfinished path."""
    source = row["material"]["native_data"]
    for key in ("name", "initial_state", "prompt", "zapier_tools", "assertions"):
        if source[key] != native_data[key]:
            raise ValueError(f"candidate current native {key} differs from audited source")
    allowed = native_data["zapier_tools"]
    initial_state = copy.deepcopy(native_data["initial_state"])
    state = (state_factory(initial_state) if state_factory else
             SimpleNamespace(world=initial_state, capture_actions=True))
    owner = SimpleNamespace(state=state, config=SimpleNamespace(allowed_tools=allowed))
    initial = digest(owner.state.world)
    records = []
    for index, action in enumerate(row["reference_actions"]):
        call = action["call"]
        if call["name"] not in allowed:
            raise ValueError("historical reference tool is not enabled")
        expected = response(action["result"]["observed_result"], call["name"])
        if action["result"].get("observed_error") is not None:
            raise ValueError("historical thrown-error reference needs explicit review")
        observed = response(invoke(owner, call["name"], **copy.deepcopy(call["arguments"])), call["name"])
        if succeeded(observed) != succeeded(expected):
            raise ValueError(f"native response status changed at action{index}")
        if not succeeded(observed) and observed.get("error") != expected.get("error"):
            raise ValueError(f"native expected error changed at action{index}")
        records.append({"index": index, "tool": call["name"], "arguments": call["arguments"],
                        "expected_success": succeeded(expected), "observed": observed})
    if not records or not succeeded(records[-1]["observed"]):
        raise ValueError("candidate reference does not finish with successful native execution")
    if digest(owner.state.world) == initial:
        raise ValueError("candidate write procedure did not persist a native world change")
    scoring = (scorer(world=copy.deepcopy(owner.state.world),
                      initial_state=copy.deepcopy(native_data["initial_state"]),
                      assertions=tuple(copy.deepcopy(native_data["assertions"]))) if scorer else None)
    return {"task": row["task"], "qualified": True, "calls": records,
            "native_scoring": scoring,
            "initial_state_sha256": digest(native_data["initial_state"]),
            "final_state_sha256": digest(owner.state.world),
            "expected_failed_calls": sum(not r["expected_success"] for r in records),
            "successful_calls": sum(succeeded(r["observed"]) for r in records),
            "scope": "Historical procedure replay through native invoke/capture on a fresh copied state. "
                     "Not model-generated behavior, task scoring, or complete goal/harm validation."}


def run(args):
    import automationbench_v1.limited_tools as limited
    from automationbench_v1.capture import collect_local_evidence
    from automationbench_v1.simulation import simulated_setup, world_clock_base
    from automationbench_v1.taskset import (
        AutomationBenchConfig,
        AutomationBenchTaskConfig,
        AutomationBenchTaskset,
        score_world,
    )
    from automationbench_v1.tools import AutomationBenchState
    from automationbench_v1.world_codec import dump_world, load_world

    def scorer(**kwargs):
        value = score_world(**kwargs)
        return {name: getattr(value, name) for name in (
            "partial_credit", "task_completed_correctly", "assertions_passed", "assertions_scored",
            "assertions_excluded", "assertion_results")}

    def state_factory(initial):
        clock = world_clock_base(initial)
        with simulated_setup(initial, clock):
            world = dump_world(load_world(initial))
        return AutomationBenchState.model_construct(
            world=world, initial_state=initial, capture_actions=True,
            deterministic_world=True, world_clock=clock.isoformat())

    toolset = next(value for value in vars(limited).values()
                   if isinstance(value, type) and "invoke" in value.__dict__)
    if args.output.exists():
        raise FileExistsError("retained replay receipt exists; do not overwrite an attempt")
    evidence_dir = args.output.with_suffix("")
    evidence_dir.mkdir(exist_ok=False)
    materials = json.loads(args.materials.read_text())
    rows = materials.get("eligible", materials.get("rows", []))
    selected = {r["task"]: r for r in rows if r["task"] in args.tasks}
    if len(args.tasks) != len(set(args.tasks)) or set(selected) != set(args.tasks):
        raise ValueError("candidate task identities missing or duplicated")
    results = []
    for name in args.tasks:
        task = AutomationBenchTaskset(AutomationBenchConfig(
            domains=[name.split(".")[0]], task_names=[name],
            task=AutomationBenchTaskConfig(toolset="limited_zapier", turn_budget=8,
                                          capture_actions=True, deterministic_world=True))).load()[0]
        data = task.data.model_dump(mode="json")
        material = selected[name]["material"]
        if "raw_native_data" in material:
            from prepare_sdpo_guidance20 import canonical_data

            for key in ("name", "initial_state", "prompt", "zapier_tools", "assertions"):
                if material["raw_native_data"][key] != data[key]:
                    raise ValueError("current raw native source differs before constructor projection")
            normalizer_sha = hashlib.sha256(Path(inspect.getsourcefile(load_world)).read_bytes()).hexdigest()
            data, projection = canonical_data(data, load_world, normalizer_sha)
            if projection != material["state_projection"]:
                raise ValueError("pinned constructor source/projection provenance differs")
        with collect_local_evidence() as envelopes:
            result = replay(selected[name], data, toolset.invoke, state_factory, scorer)
        result["native_capture"] = verify_capture(result, envelopes)
        capture_file = evidence_dir / f"{name}.json"
        capture_file.write_text(json.dumps(envelopes, indent=2) + "\n")
        result["native_capture"]["path"] = str(capture_file)
        results.append(result)
    receipt = {"qualified": True, "tasks": results, "model_calls": 0, "optimizer_updates": 0,
               "materials_sha256": hashlib.sha256(args.materials.read_bytes()).hexdigest(),
               "invoke_source_sha256": hashlib.sha256(inspect.getsource(toolset.invoke).encode()).hexdigest(),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "CPU-only isolated fixture compatibility; no live state or source modification. "
                        "New-task native generation, score and original-token qualification remain required."}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
