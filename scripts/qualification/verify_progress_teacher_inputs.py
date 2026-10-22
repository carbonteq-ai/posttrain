"""Pinned CPU token/source gate after the ten-task native teacher readout."""

import argparse
import json
import math
from pathlib import Path

from qualify_combined_teacher_contexts import native_messages


def verify(root, model, reference=None, native_reference=None):
    from sdpo_prompting import prompt_ids
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokenizer._sdpo_model_family = "qwen3.5"
    rows = json.loads((root / "selected-transitions.json").read_text())
    if reference is not None and (root / "selected-transitions.json").read_bytes() != (
            reference / "selected-transitions.json").read_bytes():
        raise ValueError("referenced selected batch differs")
    if reference is not None and native_reference is not None:
        raise ValueError("choose one reference mode")
    calls_root = native_reference or reference or root
    calls = [json.loads(s) for s in (calls_root / "step-0000/online_train/generation-calls.jsonl").read_text().splitlines() if s]
    alternatives = json.loads((root / "teacher-alternatives.json").read_text())
    checks = []
    for row in rows:
        call = calls[row["generation_index"]]
        messages = native_messages(call["messages"])
        if prompt_ids(tokenizer, messages, call["tools"], thinking=True) != row["student_prefix_ids"]:
            raise ValueError("original student serialization differs")
        if prompt_ids(tokenizer, messages + [{"role": "user", "content": row["teacher_feedback"]}],
                      call["tools"], thinking=True) != row["teacher_prefix_ids"]:
            raise ValueError("original feedback teacher serialization differs")
        if call["action_ids"] != row["action_ids"] or call["sampled_logprobs"] != row["sampled_logprobs"]:
            raise ValueError("original action/logprob changed")
        if len(row["sampled_logprobs"]) != len(row["action_ids"]) or not all(math.isfinite(x) for x in row["sampled_logprobs"]):
            raise ValueError("invalid exact sampled support")
        key = f"{row['episode_id']}-{row['generation_index']}"
        supplement = (root / "contexts" / key / "teacher-supplement.txt").read_text()
        revised = prompt_ids(tokenizer, messages + [{"role": "user", "content": supplement}], call["tools"], thinking=True)
        if revised != alternatives[key] or tokenizer.decode(revised, skip_special_tokens=False) != (
                root / "contexts" / key / "full-teacher-context.txt").read_text():
            raise ValueError("reviewed revised teacher serialization differs")
        if max(len(row["student_prefix_ids"]), len(row["teacher_prefix_ids"]), len(revised)) + len(row["action_ids"]) > 32768:
            raise ValueError("overlong whole transition; no truncation")
        checks.append({"task": row["task"], "episode_id": row["episode_id"], "generation_index": row["generation_index"],
                       "student_tokens": len(row["student_prefix_ids"]), "teacher_tokens": len(revised),
                       "action_tokens": len(row["action_ids"]), "policy_version": row["policy_version"]})
    expected = max(32, len(json.loads((root / "receipt.json").read_text())["tasks"]))
    if len(checks) != expected or len({(c["episode_id"], c["generation_index"]) for c in checks}) != expected:
        raise ValueError("incomplete distinct batch")
    return {"qualified": True, "optimizer_updates": 0, "exact_student_action_logprob_teacher_bindings": checks,
            "no_truncation": True, "distinct_tasks": len({c["task"] for c in checks}),
            "scope": "Serialization/source gate only, not semantic or gradient qualification"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--native-reference-root", type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.model, args.reference_root, args.native_reference_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"qualified": result["qualified"], "distinct_tasks": result["distinct_tasks"]}))
