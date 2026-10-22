"""Audit native repair-transfer populations and all paired candidate comparisons."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from self_campaign_config import comparisons, summarize


def read(path):
    return json.loads(path.read_text())


def audit(root):
    population = read(root / "population.json")
    names = population["tasks"]
    digest = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
    if len(names) != 24 or len(set(names)) != 24 or digest != population["sorted_names_sha256"]:
        raise ValueError("population identity mismatch")
    receipt = read(root / "receipt.json")
    if not receipt["completed"] or receipt["optimizer_updates"] != 0 or receipt["candidates"] != 3:
        raise ValueError("incomplete or unexpected experiment")
    provenance = read(root / "candidate-provenance.json")
    expected = ["checkpoint64", "argument_sft", "argument_self_hybrid"]
    if [p["name"] for p in provenance] != expected or not all(p["tensor_equality_verified"] for p in provenance):
        raise ValueError("candidate adapter parity not verified")
    summaries, phases, all_ids = {}, [], set()
    for step, name in enumerate(expected):
        folder = root / f"step-{step:04d}" / "repair_transfer"
        path = folder / "native/traces.jsonl"
        episodes = [json.loads(line) for line in path.read_text().splitlines() if line]
        ids = {e["id"] for e in episodes}
        if all_ids & ids:
            raise ValueError("native episode IDs reused across candidates")
        all_ids |= ids
        summary = summarize(episodes, names, 2)
        if summary != read(folder / "outcomes.json"):
            raise ValueError("retained summary disagrees with native traces")
        admission = read(folder / "training-admission.json")
        if admission["training_allowed"] or admission["truncated_training_sequences"]:
            raise ValueError("transfer evaluation admitted training or truncation")
        calls = [json.loads(line) for line in (folder / "generation-calls.jsonl").read_text().splitlines() if line]
        context_rejections = 0
        for call in calls:
            if call["policy_version"] != step:
                raise ValueError("stale policy in native collection")
            if (call.get("limit_reason") == "context_limit" and not call.get("action_ids")
                    and not call.get("raw") and not call.get("complete")):
                context_rejections += 1  # Rejected before GPU generation; no serving adapter receipt exists.
            elif call.get("adapter_sha256") != provenance[step]["exported_sha256"]:
                raise ValueError("stale policy in native collection")
        finishes = Counter(str(c.get("finish_reason")) for c in calls)
        repeated = Counter()
        for episode in episodes:
            signatures = []
            for trace in episode["traces"]:
                for node in trace.get("nodes", []):
                    for call in (node.get("message") or {}).get("tool_calls", []):
                        function = call.get("function", call)
                        arguments = function.get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                pass
                        signatures.append(json.dumps([function.get("name"), arguments], sort_keys=True))
            repeated[episode["task"]["data"]["name"]] += len(signatures) - len(set(signatures))
        summaries[name] = summary
        phases.append({"candidate": name, "episodes": len(episodes), "summary": summary,
                       "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "generation_calls": len(calls), "finish_reasons": dict(finishes),
                       "context_rejected_calls_without_generation": context_rejections,
                       "repeated_exact_tool_calls_by_task": dict(repeated),
                       "repeated_exact_tool_calls": sum(repeated.values()),
                       "retry_count": None,
                       "admission": admission})
    if len(all_ids) != 144:
        raise ValueError("expected144 unique native episodes")
    pairs = [(expected[0], expected[1]), (expected[0], expected[2]), (expected[1], expected[2])]
    changed_tasks = {}
    for a, b in pairs:
        changed_tasks[f"{a}_to_{b}"] = {}
        for metric in ("task_completed_correctly", "partial_credit", "tool_mistakes"):
            before = summaries[a]["metrics"][metric]["complete_task_means"]
            after = summaries[b]["metrics"][metric]["complete_task_means"]
            changed_tasks[f"{a}_to_{b}"][metric] = {
                name: {"before": before[name], "after": after[name], "delta": after[name] - before[name]}
                for name in sorted(set(before) & set(after)) if after[name] != before[name]}
    return {"receipt": receipt, "episodes": len(all_ids), "population": population,
            "provenance": provenance, "candidates": phases,
            "changed_tasks": changed_tasks,
            "comparisons": {f"{a}_to_{b}": comparisons(summaries[a], summaries[b]) for a, b in pairs},
            "scope": "Repair-excluded tasks within original training pool; not heldout generalization. "
                     "Paired intervals resample tasks; greedy repetitions are not independent draws. "
                     "Unknown harms are not harm-free; retry counts unavailable."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root), indent=2) + "\n")
