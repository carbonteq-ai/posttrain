"""Read-only current-rubric review and explicit rendering of offline Luna observations."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def render_events(events):
    """Project complete domain executions; omit discovery schemas and routing duplication."""
    def decode_result(value):
        if value is None:
            return None
        decoded = json.loads(value)
        if isinstance(decoded, str):
            try:
                nested = json.loads(decoded)
                if isinstance(nested, (dict, list)):
                    decoded = nested
            except (ValueError, TypeError):
                pass
        return decoded

    rows = []
    for event in events:
        if event["tool_name"] == "search_tools":
            continue  # Tool discovery schemas are already supplied by the live task tools.
        arguments = json.loads(event["arguments_json"])
        row = {"source_tool": event["tool_name"],
               "observed_result": decode_result(event.get("result_json")),
               "observed_error": decode_result(event.get("error_json")), "status": event["status"],
               "source_occurrence_index": event["occurrence_index"]}
        if event["tool_name"] == "execute_tool":
            row["underlying_tool"] = arguments["tool_name"]
            value = arguments["arguments"]
            row["underlying_arguments"] = json.loads(value) if isinstance(value, str) else value
            extras = {k: v for k, v in arguments.items() if k not in ("tool_name", "arguments")}
            if extras:
                row["extra_source_arguments"] = extras
        else:
            row["source_arguments"] = arguments
        rows.append(row)
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def review(root, check):
    started = time.monotonic()
    manifest = json.loads((root / "candidate-manifest.json").read_text())
    references, audits = {}, []
    for source in manifest["candidates"]:
        raw = (root / source["episode_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["source_episode_sha256"]:
            raise ValueError("candidate source digest changed")
        episode = json.loads(raw)
        trace = episode["traces"][0]
        data = trace["task"]["data"]
        final = trace["info"]["automationbench"]["end_state"]
        checks, errors = [], []
        for spec in data["assertions"]:
            if spec.get("scored") is False or spec.get("excluded") is True:
                continue
            try:
                initial_passed, final_passed = check(data["initial_state"], spec), check(final, spec)
                checks.append({"spec": spec, "initial_passed": initial_passed, "final_passed": final_passed})
            except Exception as exc:
                errors.append({"spec": spec, "error": f"{type(exc).__name__}: {exc}"})
        qualified = bool(checks) and not errors and all(c["final_passed"] for c in checks)
        row = {"task": source["task"], "source_episode_sha256": source["source_episode_sha256"],
               "qualified": qualified, "checks": checks, "errors": errors,
               "preservation_checks": sum(c["initial_passed"] for c in checks),
               "broken_preservation_checks": sum(c["initial_passed"] and not c["final_passed"] for c in checks),
               "newly_satisfied_checks": sum(not c["initial_passed"] and c["final_passed"] for c in checks),
               "full_prompt_matches": source["full_prompt_matches"],
               "prompt_policy": "Reference renders executed observations only; historical system/clock hints omitted"}
        audits.append(row)
        if qualified:
            observations = render_events(trace["info"]["automationbench_capture"]["events"])
            references[source["task"]] = {"episode_id": source["episode_id"],
                "source_episode_sha256": source["source_episode_sha256"],
                "initial_state_sha256": source["initial_state_sha256"],
                "assertions_sha256": source["assertions_sha256"], "observations": observations,
                "user_messages_sha256": hashlib.sha256(json.dumps(
                    [m["content"] for m in data["prompt"] if m["role"] == "user"],
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "current_native_checks_pass": True,
                "warning": "Retained execution observations, not full token-aligned thinking or per-action gold"}
            references[source["task"]]["rendering_policy"] = (
                "All retained domain executions and complete results; tool-discovery schemas and "
                "duplicate routing removed, nested JSON decoded without dropping payload fields")
    return {"references": references, "audits": audits, "missing": manifest["missing"],
            "source_model": "offline Luna reference campaign", "external_model_calls": 0,
            "elapsed_s": time.monotonic() - started,
            "current_rubric_state_checks_complete": True,
            "teacher_context_length_and_live_binding_qualified": False,
            "scope": "Read-only source qualification; not native re-execution, training updates or reward gain"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["AUTOMATIONBENCH_STRICT_ASSERTIONS"] = "1"
    from automationbench.rubric import AssertionRegistry, WorldState

    def check(state, spec):
        return bool(AssertionRegistry.check(WorldState.model_validate(state), dict(spec)))

    result = review(args.root, check)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"references": len(result["references"]), "candidates": len(result["audits"]),
                      "missing": result["missing"], "elapsed_s": result["elapsed_s"]}))
