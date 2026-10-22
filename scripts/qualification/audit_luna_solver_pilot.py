"""Read-only small-pilot audit. Keep legacy scores separate from native checks."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def observation_flags(calls):
    """Review cues only; scores and these heuristics are separate evidence."""
    flags = []
    for row in calls:
        batch = row["response"]["choices"][0]["message"].get("tool_calls") or []
        prior = json.dumps(row["clean_student_request"]["messages"], ensure_ascii=False)
        has_create = any("create" in c["function"]["name"] for c in batch)
        for call in batch:
            function = call["function"]
            try:
                args = (
                    json.loads(function["arguments"])
                    if isinstance(function["arguments"], str)
                    else function["arguments"]
                )
            except (ValueError, TypeError):
                flags.append({"kind": "unparsed_arguments", "tool": function["name"]})
                continue
            for field, value in args.items():
                if (
                    field in ("id", "mobile_phone", "email", "record_id", "contact_id")
                    and isinstance(value, str)
                    and value not in prior
                ):
                    flags.append({"kind": "literal_not_in_prior_messages", "tool": function["name"], "field": field})
            if (
                has_create
                and "message" in function["name"]
                and re.search(r"\b(?:has been created|deal created|deal has been created)\b", str(args), re.I)
            ):
                flags.append({"kind": "creation_claim_in_creation_batch", "tool": function["name"]})
    return flags


def audit(root):
    output = {}
    for size in ("4b", "9b", "27b"):
        model = root / size
        receipt = json.loads((model / "receipt.json").read_text()) if (model / "receipt.json").exists() else {}
        calls = jsonl(model / "solver-calls.jsonl")
        episodes = []
        for arm in ("control", "guided"):
            for episode in jsonl(model / arm / "native/traces.jsonl"):
                task = episode["task"]["data"]["name"]
                assessments = {}
                for trace in episode["traces"]:
                    for batch in trace.get("assessment_batches", []):
                        if batch.get("run", {}).get("status") == "complete":
                            for assessment in batch.get("assessments", []):
                                assessments[assessment["assessment_id"]] = assessment
                selected = [c for c in calls if c["task"] == task and c["arm"] == arm]
                empty_responses, error_responses, unparsed_responses = [], [], []
                for trace in episode["traces"]:
                    for node in trace.get("nodes", []):
                        msg = node.get("message") or {}
                        if msg.get("role") != "tool":
                            continue
                        try:
                            payload = json.loads(msg["content"])
                        except (TypeError, ValueError, KeyError):
                            unparsed_responses.append(msg.get("tool_call_id"))
                            continue
                        if isinstance(payload, dict):
                            if payload.get("result_count") == 0 or payload.get("count") == 0:
                                empty_responses.append(
                                    {"tool": msg.get("name"), "tool_call_id": msg.get("tool_call_id")}
                                )
                            if payload.get("success") is False or payload.get("error"):
                                error_responses.append(
                                    {"tool": msg.get("name"), "tool_call_id": msg.get("tool_call_id")}
                                )
                tool_calls, finishes, reasoning = [], [], []
                for call in selected:
                    choice = call["response"]["choices"][0]
                    message = choice["message"]
                    finishes.append(choice["finish_reason"])
                    reasoning.append(message.get("reasoning_content") or message.get("reasoning") or "")
                    tool_calls.extend(message.get("tool_calls") or [])
                repeats = Counter(json.dumps(c["function"], sort_keys=True) for c in tool_calls)
                episodes.append(
                    {
                        "task": task,
                        "arm": arm,
                        "episode_id": episode["id"],
                        "errors": episode.get("errors"),
                        "assessment_errors": episode.get("assessment_errors"),
                        "traces": [
                            {k: t.get(k) for k in ("metrics", "rewards", "errors", "stop_condition")}
                            for t in episode["traces"]
                        ],
                        "call_finish_reasons": dict(Counter(finishes)),
                        "tool_calls": tool_calls,
                        "parallel_batches": [
                            len(c["response"]["choices"][0]["message"].get("tool_calls") or []) for c in selected
                        ],
                        "repeated_exact_tool_calls": {k: n for k, n in repeats.items() if n > 1},
                        "reasoning_characters": [len(r) for r in reasoning],
                        "wait_word_count": sum(r.lower().split().count("wait") for r in reasoning),
                        "observation_review_flags": observation_flags(selected),
                        "observed_empty_tool_responses": empty_responses,
                        "observed_error_tool_responses": error_responses,
                        "unparsed_tool_responses": unparsed_responses,
                        "native_assessments": [
                            {
                                "assessment_id": a["assessment_id"],
                                "signal": a["signal"]["signal_id"],
                                "direction": a["signal"].get("direction"),
                                "status": a["status"],
                                "value": a.get("value"),
                                "reason": a.get("reason"),
                            }
                            for a in assessments.values()
                        ],
                    }
                )
        output[size] = {
            "completed_receipt": (model / "receipt.json").exists(),
            "expected_episodes": receipt.get("maximum_native_episodes", (
                2 * len(json.loads((model / "materials.json").read_text()))
                if (model / "materials.json").exists() else 4
            )),
            "observed_episodes": len(episodes),
            "stages": [
                {"task": s["task"], "stage": s["stage"], "output": s["output"]}
                for s in jsonl(model / "instruction-stages.jsonl")
            ],
            "episodes": episodes,
        }
    return {
        "scope": "Bounded candidates; legacy scores and native assessments remain separate. No SFT qualification or generalization claim.",
        "models": output,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.evidence), indent=2) + "\n")
