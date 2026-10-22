"""CPU-only same-state repair contrasts against recorded full-batch Adam effects."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from self_failure_diagnosis import corrected_action, replay
from self_feedback_signals import repair_candidates
from transformers import AutoTokenizer


def audit(root, tokenizer, maximum=4):
    updates = json.loads((root / "updates.json").read_text())
    found, errors, counts = [], [], Counter()
    for update in updates:
        step = update["step"]
        folder = root / f"step-{step:04d}/online_train"
        rows = json.loads((folder / "selected-transitions.json").read_text())
        calls = [json.loads(s) for s in (folder / "generation-calls.jsonl").read_text().splitlines()]
        episodes = {e["id"]: e for e in (json.loads(s) for s in
                    (folder / "native/traces.jsonl").read_text().splitlines())}
        directions = {d["generation_index"]: d for d in json.loads((folder / "teacher-adam-directions.json").read_text())}
        for row in rows:
            counts["actually_optimized_actions"] += 1
            call = copy.deepcopy(calls[row["generation_index"]])
            tools = call["message"].get("tool_calls") or []
            if len(tools) != 1:
                continue
            episode = episodes[row["episode_id"]]
            response = next((n["message"] for n in episode["traces"][0]["nodes"]
                             if (n.get("message") or {}).get("tool_call_id") == tools[0]["id"]), None)
            if response is None:
                continue
            feedback = json.dumps([{"tool": tools[0]["function"]["name"], "response": response["content"]}])
            proposals = repair_candidates(call, feedback)
            if len(proposals) != 1:
                continue
            counts["one_argument_repair_candidates"] += 1
            for message in call["messages"]:
                if message.get("tool_calls") is None:
                    message["tool_calls"] = []
            case = {"episode": episode, "call": call, "row": row, "repair": proposals[0], "feedback": feedback}
            try:
                same_state = replay(case)
                repaired, position = corrected_action(case, tokenizer)
                direction = directions[row["generation_index"]]
                diagnostic = direction["teacher_support_diagnostic"]
                good, bad = repaired[position], row["action_ids"][position]
                top = diagnostic["teacher_top1_ids"][position]
                same_mode = diagnostic.get("teacher_top1_delta_modes") == "eval_before/eval_after"
                margin = (diagnostic["actual_adam_teacher_top1_logprob_deltas"][position] -
                          direction["actual_adam_logprob_deltas"][position]) if same_mode and good == top else None
                found.append({"step": step, "task": row["task"], "episode_id": row["episode_id"],
                    "generation_index": row["generation_index"], "repair": proposals[0],
                    "native_success": row["native_success"], "same_state_replay": same_state,
                    "first_changed_token_position": position, "bad_token": tokenizer.decode([bad]),
                    "repaired_token": tokenizer.decode([good]), "teacher_top1_token": tokenizer.decode([top]),
                    "teacher_top1_is_repaired_token": good == top,
                    "teacher_sampled_advantage_at_bad_token": direction["teacher_sampled_advantages"][position],
                    "same_mode_before_after": same_mode,
                    "actual_adam_correct_minus_bad_logprob_delta": margin,
                    "teacher_feedback_sha256": hashlib.sha256(row["teacher_feedback"].encode()).hexdigest(),
                    "scope": "Repair passes immediate same-state execution/retrieval, not full task/guard correctness. "
                             "Correct-token Adam delta is unknown unless recorded teacher-top1 equals repaired token; "
                             "missing values are never zero-filled."})
            except Exception as exc:
                errors.append({"step": step, "task": row["task"], "generation_index": row["generation_index"],
                               "error": f"{type(exc).__name__}: {exc}"})
    # Prefer distinct tasks, then retain additional distinct episode instances.
    ordered, used_tasks = [], set()
    for case in found:
        if case["task"] not in used_tasks:
            ordered.append(case)
            used_tasks.add(case["task"])
    used = {(c["episode_id"], c["generation_index"]) for c in ordered}
    ordered += [c for c in found if (c["episode_id"], c["generation_index"]) not in used]
    return {"cases": ordered[:maximum], "replay_qualified_contrasts": len(found), "counts": dict(counts),
            "repair_or_replay_errors": errors, "optimizer_updates_added": 0, "gpu_used": False,
            "external_model_calls": 0, "scope": "Recorded applied SDPO updates only; no new model scoring, "
            "training, corrected-call CE, or invented full-gradient correctness evidence. "
            "A negative local contrast is a batch interaction observation, not an optimizer bug proof."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    result = audit(args.root, tokenizer)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(result["cases"]), "qualified_contrasts": result["replay_qualified_contrasts"],
                      "counts": result["counts"], "errors": len(result["repair_or_replay_errors"])}))
