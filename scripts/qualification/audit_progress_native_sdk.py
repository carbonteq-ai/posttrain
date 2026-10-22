"""Re-execute fixed model proposals using actual MCP argument admission.

This is a separate read-only qualification measurement, not a rescore of frozen
native benchmark episodes. Original proposals and replay artifacts stay intact.
"""

import argparse
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from qualify_combined_teacher_contexts import replay, write


def audit(root):
    receipt = json.loads((root / "receipt.json").read_text())
    cases = json.loads((root / "source-cases.json").read_text())["cases"]
    proposals = json.loads((root / "same-decision-proposals.json").read_text())
    if (not receipt["runtime_completed"] or receipt["native_episodes"] != 496
            or receipt["optimizer_updates"] != 0 or len(cases) != 124 or len(proposals) != 372):
        raise ValueError("complete fixed qualification population required")
    by_name = {case["row"]["task"]: case for case in cases}
    if len(by_name) != 124 or Counter((p["task"], p["mode"]) for p in proposals) != Counter(
            (name, mode) for name in by_name for mode in ("student", "original_teacher", "revised_teacher")):
        raise ValueError("all task/mode identities required")
    results = []
    for proposal in proposals:
        result = {"task": proposal["task"], "mode": proposal["mode"], "complete": proposal["complete"],
                  "prefix_sha256": hashlib.sha256(json.dumps(proposal["prefix_ids"]).encode()).hexdigest(),
                  "action_sha256": hashlib.sha256(json.dumps(proposal["action_ids"]).encode()).hexdigest()}
        if proposal["complete"] and proposal.get("message"):
            try:
                actual = replay(by_name[proposal["task"]], proposal["message"])
                old = proposal.get("native_replay")
                if old and actual["decision_state_sha256"] != old["decision_state_sha256"]:
                    raise ValueError("pre-decision state changed under admission audit")
                result["native_replay"] = actual
            except Exception as error:
                result["native_error"] = {"type": type(error).__name__, "message": str(error)}
        else:
            result["missing_reason"] = "incomplete_or_unparsed_proposal"
        results.append(result)
    return {"task_count": 124, "proposals": results, "model_calls": 0, "optimizer_updates": 0,
            "mcp_version": version("mcp"), "source_proposals_sha256": hashlib.sha256(
                (root / "same-decision-proposals.json").read_bytes()).hexdigest(),
            "processor_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                  for name in ("audit_progress_native_sdk.py", "qualify_combined_teacher_contexts.py")},
            "scope": "Same actual model prefix/action/message; isolated native tools with actual SDK coercion/defaults. Original replay and full native episode scores remain untouched. Semantic review still required."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    write(args.output, result)
    print(json.dumps({"tasks": result["task_count"], "proposals": len(result["proposals"]),
                      "errors": sum(bool(p.get("native_error")) for p in result["proposals"]),
                      "model_calls": 0, "optimizer_updates": 0}))
