"""CPU replay of retained native messages through exact new SDPO admission/rendering."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from sdpo_agent_ablation import associate
from sdpo_agent_config import balanced_rows
from transformers import AutoTokenizer
from verifiers.v1.episode import Episode


def main(args):
    start = time.monotonic()
    folder = args.prior if args.direct_folder else args.prior / "step-0001" / "online_train"
    calls = [json.loads(s) for s in (folder / "generation-calls.jsonl").read_text().splitlines()]
    # Native persistence omits unset optional fields. Rehydrate through the
    # actual runner's Episode type so model_dump matches run_eval's in-memory
    # values, including tool_calls=None on text-only assistant messages.
    episodes = [Episode.model_validate(json.loads(s)).model_dump(mode="json")
                for s in (folder / "native/traces.jsonl").read_text().splitlines()]
    nulls = sum(n.get("message", {}).get("tool_calls", "absent") is None
                for e in episodes for t in e["traces"] for n in t["nodes"] if n.get("message"))
    assert nulls > 0, "regression fixture must contain the actual null tool_calls case"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    references = json.loads(args.luna_references.read_text())["references"] if args.luna_references else None
    native_tasks = None
    if references is not None:
        from sdpo_luna_context import native_task_bindings

        retained = [json.loads(line) for line in (folder / "native/traces.jsonl").read_text().splitlines() if line]
        native_tasks = native_task_bindings(episodes, retained)
    rows, excluded = associate(calls, episodes, tokenizer, args.feedback_mode, references, native_tasks,
                               args.context_limit)
    tasks = json.loads(args.cohorts.read_text())["train"] if args.cohorts else None
    if tasks is not None:
        rows = [r for r in rows if r["task"] in tasks]
    args.output.with_suffix(".admission.json").write_text(json.dumps({
        "stage": "before_task_coverage_gate", "admitted": len(rows), "exclusions": excluded,
        "admitted_task_counts": dict(Counter(r["task"] for r in rows))}, indent=2) + "\n")
    if args.batch_size == 32:
        from sdpo_long_campaign_config import campaign_batch

        batch = campaign_batch(rows, tasks)
    else:
        batch = balanced_rows(rows, args.batch_size, tasks=tasks, mode=args.selection_mode)
    for row in rows:
        original = calls[row["generation_index"]]
        assert row["student_prefix_ids"] == original["prefix_ids"]
        assert row["action_ids"] == original["action_ids"]
        assert max(len(row[k]) + len(row["action_ids"])
                   for k in ("student_prefix_ids", "teacher_prefix_ids")) <= args.context_limit
    output = {"passed": True, "native_episodes": len(episodes), "actual_null_tool_call_messages": nulls,
              "admitted": len(rows), "excluded": excluded, "selected": len(batch),
              "task_coverage": dict(Counter(r["task"] for r in batch)),
              "selected_terminal_actions": sum(r["terminal_text"] for r in batch),
              "selection_mode": args.selection_mode,
              "unique_selected_actions": len({(r["episode_id"], r["generation_index"]) for r in batch}),
              "selected_observed_tool_errors": sum(r["observed_tool_error"] for r in batch),
              "selected_actions": [{k: r[k] for k in ("task", "generation_index", "terminal_text",
                                                       "observed_tool_error")} for r in batch],
              "student_prefix_action_parity": True, "no_truncation": True,
              "feedback_mode": args.feedback_mode,
              "successful_reference_actions": sum(r["solution_used"] for r in rows),
              "offline_luna_reference_actions": sum(r["offline_luna_reference_used"] for r in rows),
              "offline_luna_fallback_actions": sum(r["offline_luna_missing_fallback"] for r in rows),
              "max_teacher_context_plus_target": max(len(r["teacher_prefix_ids"]) + len(r["action_ids"]) for r in rows),
              "own_success_references": sum(r["solution_used"] and
                  r["reference_episode_id"] == r["episode_id"] for r in rows),
              "elapsed_s": time.monotonic() - start, "optimizer_updates": 0,
              "scope": "CPU protocol regression on prior failed run, not new learning evidence"}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--context-limit", type=int, choices=(16384, 32768), default=16384)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--direct-folder", action="store_true")
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feedback-mode", choices=("native-feedback", "solution-or-feedback", "luna-or-solution"),
                        default="native-feedback")
    parser.add_argument("--luna-references", type=Path)
    parser.add_argument("--selection-mode", choices=("legacy-balanced", "distinct-balanced", "native-error-priority", "failure-frontier", "first-error-frontier"),
                        default="legacy-balanced")
    main(parser.parse_args())
