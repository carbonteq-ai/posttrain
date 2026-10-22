"""Reconcile a bounded SELF vLLM cycle against its retained native episodes."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit_phase(folder, version):
    calls = [json.loads(line) for line in (folder / "generation-calls.jsonl").read_text().splitlines()]
    episodes = [json.loads(line) for line in (folder / "native/traces.jsonl").read_text().splitlines()]
    rows = json.loads((folder / "transitions.json").read_text())
    observed, terminal, stops, metrics, errors = {}, set(), [], [], []
    for episode in episodes:
        errors.extend(episode["errors"] + episode["credit_errors"] + episode["assessment_errors"])
        for trace in episode["traces"]:
            errors.extend(trace["errors"])
            stops.append(trace["stop_condition"])
            metrics.append({"task": episode["task"]["key"], "rewards": trace["rewards"], "metrics": trace["metrics"]})
            for node in trace["nodes"]:
                message = node.get("message") or {}
                if message.get("role") == "tool":
                    observed[message["tool_call_id"]] = episode["id"]
            if trace["stop_condition"] == "max_turns":
                last = trace["nodes"][-1]["message"]
                if last.get("role") == "assistant":
                    terminal.update(t["id"] for t in last.get("tool_calls", []))
    assert not errors, errors
    paired, omitted, signatures = 0, [], Counter()
    for call in calls:
        assert call["policy_version"] == version
        assert call["complete"] and len(call["sampled_logprobs"]) == len(call["action_ids"])
        tools = call["message"].get("tool_calls", [])
        for tool in tools:
            signatures[json.dumps(tool["function"], sort_keys=True)] += 1
        ids = [tool["id"] for tool in tools]
        if not ids:
            continue
        if all(key in terminal and key not in observed for key in ids):
            omitted.extend(ids)
            continue
        assert all(key in observed for key in ids), "executed action missing feedback"
        matches = [
            r
            for r in rows
            if r["action_ids"] == call["action_ids"]
            and r["student_prefix_ids"] == call["prefix_ids"]
            and r["trajectory_id"] == observed[ids[0]]
        ]
        assert len(matches) == 1, "original action/prefix/episode identity mismatch"
        assert matches[0]["policy_revision"] == f"round{version}"
        paired += 1
    assert paired == len(rows)
    return {
        "episodes": len(episodes),
        "generations": len(calls),
        "paired_transitions": paired,
        "terminal_unexecuted_actions": omitted,
        "native_errors": errors,
        "stop_conditions": stops,
        "maximum_repetition_of_identical_tool_request": max(signatures.values(), default=0),
        "native_observations_not_a_learning_benchmark": metrics,
        "native_traces_sha256": hashlib.sha256((folder / "native/traces.jsonl").read_bytes()).hexdigest(),
        "all_original_action_and_prefix_ids_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    results = {"round0": audit_phase(args.input, 0), "round1": audit_phase(args.input / "round1", 1)}
    for version in (0, 1):
        batch = json.loads((args.receipts / f"batch-invariance-v{version}.json").read_text())
        assert batch["exact_batch_invariance"]
        parity = json.loads((args.receipts / f"parity-v{version}.json").read_text())
        assert parity["max_abs_delta"] < 0.15
    receipt = json.loads((args.receipts / "native-receipt.json").read_text())
    assert receipt["ema_verified"] and receipt["applied_steps"] == 3
    assert receipt["external_model_calls"] == 0 and not receipt["heldout62_overlap"]
    assert receipt["refreshed_policy_validation"]["selected_token_logprob_movement"] > 0.01
    results["engineering_cycle_qualified"] = True
    results["learning_quality_qualified"] = False
    results["scope"] = "one non-heldout task, three updates; second trajectory hit turn limit; no reward-gain claim"
    (args.receipts / "native-audit.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
