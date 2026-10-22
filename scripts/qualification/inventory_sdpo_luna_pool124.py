"""Inventory existing offline demonstrations; never declare them gold or call a model."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inspect(episode, binding):
    trace = episode["traces"][0]
    data = trace["task"]["data"]
    if not episode.get("ok") or trace.get("metrics", {}).get("task_completed_correctly") != 1:
        return {"candidate": False, "reason": "not_recorded_full_success"}
    actual = {"initial_state_sha256": digest(data["initial_state"]),
              "assertions_sha256": digest(data["assertions"]),
              "user_messages_sha256": digest([m["content"] for m in data["prompt"] if m["role"] == "user"])}
    mismatch = [key for key, value in actual.items() if binding[key] != value]
    if mismatch:
        return {"candidate": False, "reason": "current_task_binding_mismatch", "mismatched_fields": mismatch}
    capture = trace.get("info", {}).get("automationbench_capture", {})
    events = capture.get("events", [])
    if not events or not capture.get("snapshots"):
        return {"candidate": False, "reason": "no_retained_execution_state_evidence"}
    snapshots = capture["snapshots"]
    if isinstance(snapshots, str):
        try:
            snapshots = json.loads(snapshots)
        except ValueError:
            return {"candidate": False, "reason": "unavailable_snapshot_payload"}
    if not isinstance(snapshots, dict) or not snapshots:
        return {"candidate": False, "reason": "unavailable_snapshot_payload"}
    used = set()
    for event in events:
        name = event["tool_name"]
        if name in ("search_tools", "get_tool_schema", "get_tool_schemas"):
            continue
        if name == "execute_tool":
            arguments = json.loads(event["arguments_json"])
            if not isinstance(arguments, dict) or not isinstance(arguments.get("tool_name"), str):
                return {"candidate": False, "reason": "unresolved_underlying_tool"}
            name = arguments["tool_name"]
        used.add(name)
    unavailable = sorted(used - set(binding["available_tools"]))
    return {"candidate": not unavailable, "reason": "requires_current_rubric_review" if not unavailable
            else "unavailable_reference_tools", "unavailable_tools": unavailable,
            "recorded_domain_tools": sorted(used), "captured_events": len(events),
            "capture_coverage": capture.get("coverage"), "recorded_success": True,
            "current_native_checks_qualified": False, "hidden_thinking_available": False}


def inventory(roots, bindings):
    entries, reasons, seen_files = [], Counter(), set()
    for root in roots:
        for path in sorted(root.rglob("episode.json")):
            if path.resolve() in seen_files:
                continue
            seen_files.add(path.resolve())
            raw = path.read_bytes()
            episode = json.loads(raw)
            traces = episode.get("traces", [])
            if not traces:
                continue
            name = traces[0].get("task", {}).get("data", {}).get("name")
            if name not in bindings:
                continue  # Never admit demonstrations of the62 heldout tasks.
            result = inspect(episode, bindings[name])
            reasons[result["reason"]] += 1
            entries.append({"task": name, "episode_id": episode["id"], "source_episode_path": str(path),
                            "source_episode_sha256": hashlib.sha256(raw).hexdigest(), **result})
    tasks = {e["task"] for e in entries if e["candidate"]}
    return {"training_tasks": len(bindings), "source_files_examined": len(seen_files),
            "tasks_with_bound_tool_compatible_candidates": len(tasks),
            "tasks_without_bound_tool_compatible_candidates": sorted(set(bindings) - tasks),
            "entry_reasons": dict(reasons), "entries": entries, "external_model_calls": 0,
            "qualified": False,
            "scope": "Offline inventory only; requires current final-state/preservation rubric review, "
                     "complete reference rendering and token-length admission. No per-action gold or hidden thinking claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.root, json.loads(args.bindings.read_text())["bindings"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "entries"}))
