"""Read-only reconstruction using retained native historical execution arguments."""

import argparse
import copy
import json
from pathlib import Path

from qualify_combined_teacher_contexts import replay


def audit(root, output):
    cases = {c["row"]["task"]: c for c in json.loads((root / "source-cases.json").read_text())["cases"]}
    original = json.loads((root / "same-decision-proposals.json").read_text())
    results = []
    for p in original:
        item = {"task": p["task"], "mode": p["mode"], "complete": p["complete"],
                "original_replay_error": p.get("native_error"), "message": copy.deepcopy(p.get("message"))}
        if p["complete"] and p.get("message"):
            try:
                item["replayed"] = replay(cases[p["task"]], p["message"])
            except Exception as error:
                item["replay_error"] = {"type": type(error).__name__, "message": str(error)}
        results.append(item)
        output.write_text(json.dumps({"completed": False, "proposals": results}, indent=2) + "\n")
    output.write_text(json.dumps({"completed": True, "optimizer_updates": 0, "model_calls": 0,
        "proposals": results,
        "scope": "Historical inputs bound to captured native arguments with complete expected-response matching. "
                 "New proposed calls use strict native invoke; this is not proof of every MCP coercion/default behavior. "
                 "Incomplete generations remain missing; raw model outputs and original replay failures preserved."}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
