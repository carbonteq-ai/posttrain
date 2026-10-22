"""Read actual SDPO Adam evidence by reasoning/action region, without model calls."""

import argparse
import json
import math
import re
from pathlib import Path


def argument_schema_directions(row, call, direction, tokenizer):
    """Original-token evidence at schema cues, never a generated repair claim."""
    from sdpo_action_risks import action_risks

    cues = [c for c in action_risks(call, []) if c["kind"] == "undeclared_arguments"]
    if not cues:
        return {"available": True, "cues": []}
    ids = row["action_ids"]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, NotImplementedError) as error:
        return {"available": False, "reason": type(error).__name__, "cues": cues}
    if list(encoded["input_ids"]) != ids:
        return {"available": False, "reason": "decoded action does not roundtrip to exact original IDs", "cues": cues}
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(ids):
        raise ValueError("original-token offset count differs")
    native = set(regions(ids, tokenizer.encode("</think>", add_special_tokens=False), False)["native_tool_action"])
    support = direction["teacher_support_diagnostic"]
    result = []
    for cue in cues:
        for field in cue["fields"]:
            owners = []
            for tool in call["message"].get("tool_calls") or []:
                function = tool.get("function", tool)
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        continue
                if isinstance(arguments, dict) and field in arguments:
                    owners.append(function.get("name"))
            positions = set()
            if owners == [cue["tool"]]:
                patterns = [r'(?<!\\)"(' + re.escape(field) + r')"\s*:',
                            r'<parameter\s*=\s*(' + re.escape(field) + r')\s*>']
                for pattern in patterns:
                    for match in re.finditer(pattern, text):
                        positions.update(i for i, (start, end) in enumerate(offsets)
                                         if i in native and start < match.end(1) and end > match.start(1))
            result.append({"tool": cue["tool"], "field": field,
                "forbidden_by_schema": cue["forbidden_by_schema"],
                "token_attribution_available": bool(positions),
                "field_owners_in_native_calls": owners,
                "original_argument_tokens": [{"position": i,
                    "sampled_token": tokenizer.decode([ids[i]], skip_special_tokens=False),
                    "teacher_top1_token": tokenizer.decode([support["teacher_top1_ids"][i]], skip_special_tokens=False),
                    "teacher_minus_student_sampled_logp": direction["teacher_sampled_advantages"][i],
                    "actual_adam_sampled_logp_delta": direction["actual_adam_logprob_deltas"][i]}
                    for i in sorted(positions)]})
    return {"available": True, "cues": result,
            "scope": "Undeclared field cues on original native-action tokens. A cue is not proof of API failure; "
                     "pointwise top1 alternatives are not a coherent corrected tool call or successful rollout."}


def regions(ids, closing, terminal):
    end = next((i + len(closing) for i in range(len(ids) - len(closing) + 1)
                if ids[i:i + len(closing)] == closing), None)
    if not closing or end is None:
        raise ValueError("complete thinking action lacks its original closing marker")
    return {"reasoning": range(end - len(closing)),
            "final_answer" if terminal else "native_tool_action": range(end, len(ids))}


def audit(rows, directions, calls, tokenizer):
    by_key = {(d["task"], d["generation_index"]): d for d in directions}
    if len(by_key) != len(directions) or len(rows) != len(directions):
        raise ValueError("selected/actual Adam evidence identity differs")
    result = []
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    for row in rows:
        d = by_key.pop((row["task"], row["generation_index"]))
        call = calls[row["generation_index"]]
        ids = row["action_ids"]
        if call["action_ids"] != ids or call["prefix_ids"] != row["student_prefix_ids"]:
            raise ValueError("original native action/prefix differs")
        if row["teacher_feedback"] not in tokenizer.decode(row["teacher_prefix_ids"], skip_special_tokens=False):
            raise ValueError("saved teacher context is absent from its actual serialized prefix")
        support = d["teacher_support_diagnostic"]
        vectors = {"teacher_minus_student_sampled_logp": d["teacher_sampled_advantages"],
            "actual_adam_sampled_logp_delta": d["actual_adam_logprob_deltas"],
            "actual_adam_teacher_top1_logp_delta": support["actual_adam_teacher_top1_logprob_deltas"],
            "teacher_top1_student_rank": support["teacher_top1_student_ranks"],
            "teacher_mass_on_student_topk": support["teacher_mass_on_student_support"]}
        for name, key in (("actor_minus_rollout_sampled_logp", "actor_rollout_logp_differences"),
                          ("importance_weight", "importance_weights")):
            if key in support:
                vectors[name] = support[key]
        if (any(len(v) != len(ids) or not all(math.isfinite(x) for x in v) for v in vectors.values())
                or len(support["teacher_top1_ids"]) != len(ids)):
            raise ValueError("token-direction vector length/finite check failed")
        parts = {}
        for name, indices in regions(ids, closing, row["terminal_text"]).items():
            indices = list(indices)
            entry = {"tokens": len(indices), "means": {key: sum(v[i] for i in indices) / len(indices)
                     if indices else None for key, v in vectors.items()}}
            weights = vectors.get("importance_weight")
            entry["importance_weight_at_cap2_fraction"] = (
                sum(weights[i] >= 2. for i in indices) / len(indices) if weights is not None and indices else None)
            discrepancy = vectors.get("actor_minus_rollout_sampled_logp")
            entry["actor_rollout_max_abs_logp_difference"] = (
                max(abs(discrepancy[i]) for i in indices) if discrepancy is not None and indices else None)
            entry["teacher_top1_outside_student_top20_fraction"] = (
                sum(vectors["teacher_top1_student_rank"][i] > 20 for i in indices) / len(indices)
                if indices else None)
            entry["teacher_top1_outside_student_top100_fraction"] = (
                sum(vectors["teacher_top1_student_rank"][i] > 100 for i in indices) / len(indices)
                if indices else None)
            lowest = sorted(indices, key=lambda i: vectors["teacher_minus_student_sampled_logp"][i])[:6]
            entry["lowest_sampled_teacher_advantages"] = [{"position": i,
                "sampled_token": tokenizer.decode([ids[i]], skip_special_tokens=False),
                "teacher_top1_token": tokenizer.decode([support["teacher_top1_ids"][i]], skip_special_tokens=False),
                "original_token_window": tokenizer.decode(ids[max(0, i - 8):i + 9], skip_special_tokens=False),
                "teacher_minus_student_sampled_logp": vectors["teacher_minus_student_sampled_logp"][i],
                "actual_adam_sampled_logp_delta": vectors["actual_adam_sampled_logp_delta"][i]}
                for i in lowest]
            parts[name] = entry
        result.append({"task": row["task"], "episode_id": row["episode_id"],
            "generation_index": row["generation_index"], "native_success": row.get("native_success"),
            "sampler_observed_tool_error": row.get("observed_tool_error"),
            "native_message": call["message"], "original_generated_text": call["raw"],
            "teacher_feedback": row["teacher_feedback"], "regions": parts,
            "argument_schema_directions": argument_schema_directions(row, call, d, tokenizer)})
    return {"selected_actions": len(result), "rows": result,
        "scope": "Pointwise teacher logits under original student action prefixes and actual full-batch Adam changes. "
                 "Top1 tokens are not a generated teacher trajectory or correctness proof. "
                 "Terminal task success is not per-action credit; sampled error flags retain the running sampler's definition."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    updates = json.loads((args.root / "updates.json").read_text())
    record = next(u for u in updates if u["step"] == args.step)
    folder = args.root / f"step-{args.step:04d}/online_train"
    rows = json.loads((folder / "selected-transitions.json").read_text())
    if record["batch"] != len(rows) or record["optimizer_moment_steps"] != [args.step]:
        raise ValueError("applied update/moment/batch evidence differs")
    result = audit(rows, json.loads((folder / "teacher-adam-directions.json").read_text()),
        [json.loads(line) for line in (folder / "generation-calls.jsonl").read_text().splitlines() if line],
        AutoTokenizer.from_pretrained(args.model, local_files_only=True))
    result["update"] = record
    args.output.write_text(json.dumps(result, indent=2) + "\n")
