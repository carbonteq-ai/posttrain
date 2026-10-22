"""Independent native/grounding flags; never automatically approve teaching."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from analyze_progress_teacher_ten import analyze
from progress_teacher_context import forbidden_terms


def review_flags(case, proposal):
    messages = case["call"]["messages"]
    authority = '\n'.join(json.dumps(m.get("content"), ensure_ascii=False) for m in messages
                          if m.get("role") in ("system", "user", "tool"))
    source = authority.casefold()
    flags = []
    if not proposal.get("complete"):
        return [{"kind": "incomplete_proposal", "missing_not_zero": True}]
    if proposal.get("native_error"):
        flags.append({"kind": "native_replay_error", "error": proposal["native_error"]})
    replay = proposal.get("native_replay", {})
    for decision in replay.get("decisions", []):
        if decision.get("raised") or decision.get("strict_admission_rejected"):
            flags.append({"kind": "execution_error", "tool": decision["tool"],
                          "error": decision.get("error", decision.get("result"))})
    assertions = case["episode"]["traces"][0].get("info", {}).get("automationbench", {}).get("assertions", [])
    forbidden = []
    for assertion in assertions:
        if assertion.get("excluded"):
            continue
        forbidden.extend(forbidden_terms(assertion))
    identifier = re.compile(r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b(?:ss|ws|msg|thread|brd|lst|list|hs)_[A-Za-z0-9_-]+\b|\b00[13][A-Za-z0-9]{10,18}\b)")
    for call in proposal.get("message", {}).get("tool_calls") or []:
        function = call["function"]
        arguments = function["arguments"]
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        def values(value):
            if isinstance(value, dict):
                return [s for v in value.values() for s in values(v)]
            if isinstance(value, list):
                return [s for v in value for s in values(v)]
            return [value] if isinstance(value, str) else []

        text = '\n'.join(values(arguments))
        unknown = sorted({v for v in identifier.findall(text) if v.casefold() not in source})
        if unknown:
            flags.append({"kind": "unobserved_identifier_or_address_candidate", "tool": function["name"], "values": unknown,
                          "scope": "Review necessary; this screen does not prove invalidity or infer aliases."})
        outgoing = '\n'.join(str(v) for k, v in arguments.items() if k in (
            "body", "body_text", "text", "message", "subject", "description", "content"))
        if any(word in function["name"] for word in ("send", "reply", "post", "create", "update")):
            hits = sorted({v for v in forbidden if isinstance(v, str) and v.casefold() in outgoing.casefold()})
            if hits:
                flags.append({"kind": "native_forbidden_text_candidate", "tool": function["name"], "values": hits,
                              "scope": "Native negative-check literals, not teacher targets. Review destination and check scope."})
    return flags


def audit(root, names, reference=None):
    native = analyze(root, names, reference)
    cases = {c["row"]["task"]: c for c in json.loads((root / "source-cases.json").read_text())["cases"]}
    proposals = json.loads((root / "same-decision-proposals.json").read_text())
    reviews = []
    for name in names:
        modes = {p["mode"]: p for p in proposals if p["task"] == name}
        flags = review_flags(cases[name], modes["revised_teacher"])
        reviews.append({"task": name, "decision": "pending_independent_review", "flags": flags,
                        "primary_generation_index": cases[name]["row"]["generation_index"],
                        "one_decision_scores": {mode: p.get("native_replay", {}).get("scoring_after_one_decision")
                                                for mode, p in modes.items()},
                        "score_limit": "A grounded prerequisite may earn zero next-decision credit. This is not a full rollout improvement estimate."})
    return {"task_count": len(names), "native": native, "task_reviews": reviews,
            "flag_counts": dict(Counter(flag["kind"] for row in reviews for flag in row["flags"])),
            "semantic_qualified": False, "qualified_for_training": False,
            "scope": "Complete population reconciliation plus deterministic flags. No flags is not a semantic pass; no missing scores zero-filled."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, json.loads(args.cohorts.read_text())["train"], args.reference)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({"tasks": result["task_count"], "flags": result["flag_counts"], "qualified": False}))
