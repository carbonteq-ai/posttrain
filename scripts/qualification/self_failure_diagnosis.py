"""Read-only replay and separate SELF gradient diagnosis on four retained failures."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from automationbench.tools import ALL_TOOLS
from automationbench_v1.world_codec import load_world
from peft import PeftModel
from self_feedback_signals import repair_candidates, teacher_message
from self_teacher_signal_probe import compare
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl.experimental.self_feedback.self_trainer import chunked_reverse_kl, feedback_cross_entropy

TASKS = {"simple.gmail_reply_meeting_request", "simple.slack_ceo_announcement",
         "simple.email_sf_contact_mobile_update", "hr.offboarding_automation"}


def key(row):
    return (row["trajectory_id"], tuple(row["student_prefix_ids"]), tuple(row["action_ids"]))


def cases_from(campaign, step):
    folder = campaign / f"step-{step:04d}/train"
    episodes = [json.loads(s) for s in (folder / "native/traces.jsonl").read_text().splitlines()]
    by_tool = {n["message"]["tool_call_id"]: (e, n["message"])
               for e in episodes for t in e["traces"] for n in t["nodes"]
               if (n.get("message") or {}).get("role") == "tool"}
    rows = {key(r): r for r in json.loads((folder / "transitions.json").read_text())}
    planned_samples = json.loads((campaign / f"step-{step:04d}/sampled-training-transitions.json").read_text())
    completed_updates = len((campaign / "updates.jsonl").read_text().splitlines())
    used_exposures = max(0, min(32, completed_updates - step)) * 16
    samples = Counter(key(r) for r in planned_samples[:used_exposures])
    result = {}
    for line in (folder / "generation-calls.jsonl").read_text().splitlines():
        call = json.loads(line)
        tools = call["message"].get("tool_calls", [])
        if not call["complete"] or len(tools) != 1 or tools[0]["id"] not in by_tool:
            continue
        episode, response = by_tool[tools[0]["id"]]
        name = episode["task"]["data"]["name"]
        if name not in TASKS or name in result:
            continue
        feedback = json.dumps([{"tool": response.get("name"), "response": response["content"]}], sort_keys=True)
        repairs = repair_candidates(call, feedback, call.get("teacher_goals"))
        if len(repairs) != 1:
            continue
        row_key = (episode["id"], tuple(call["prefix_ids"]), tuple(call["action_ids"]))
        if row_key not in rows:
            raise ValueError("selected failure was not an admitted training transition")
        if not samples[row_key]:
            continue  # Diagnose an actually optimized row, not merely a planned example.
        result[name] = {"task": name, "episode": episode, "call": call, "feedback": feedback,
                        "repair": repairs[0], "row": rows[row_key], "sample_exposures": samples[row_key],
                        "completed_updates": completed_updates, "round_exposures_consumed": used_exposures,
                        "task_exposures": sum(n for k, n in samples.items()
                                              if any(e["id"] == k[0] and e["task"]["data"]["name"] == name
                                                     for e in episodes))}
    if set(result) != TASKS:
        raise ValueError(f"missing replayable failures: {TASKS - set(result)}")
    return list(result.values())


def replay(case):
    functions = {f.__name__: f for f in ALL_TOOLS}
    world = load_world(case["episode"]["task"]["data"]["initial_state"])
    for m in case["call"]["messages"]:
        for c in m.get("tool_calls", []):
            f = c["function"]
            functions[f["name"]](world, **json.loads(f["arguments"]))
    bad = case["call"]["message"]["tool_calls"][0]["function"]
    original = json.loads(functions[bad["name"]](copy.deepcopy(world), **json.loads(bad["arguments"])))
    fixed = case["repair"]
    corrected = json.loads(functions[fixed["function"]](copy.deepcopy(world), **fixed["arguments"]))
    expected = json.loads(json.loads(case["feedback"])[0]["response"])
    for field in ("success", "error", "result_count"):
        if original.get(field) != expected.get(field):
            raise ValueError("original replay differs from native response")
    if not corrected.get("success") or corrected.get("messages") == []:
        raise ValueError("repaired call did not execute/retrieve evidence")
    return {"original": original, "repaired": corrected,
            "scope": "same-state execution/retrieval, not full task correctness"}


def logits(model, prefix, target):
    ids = torch.tensor([prefix + target[:-1]], device="cuda")
    positions = torch.arange(len(prefix) - 1, ids.shape[1], device="cuda")
    return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                 logits_to_keep=positions).logits[0]


def corrected_action(case, tokenizer):
    row = case["row"]
    raw = tokenizer.decode(row["action_ids"], skip_special_tokens=False)
    if tokenizer.encode(raw, add_special_tokens=False) != row["action_ids"]:
        raise ValueError("original native IDs do not roundtrip")
    bad = json.loads(case["call"]["message"]["tool_calls"][0]["function"]["arguments"])
    changed = [k for k in bad if case["repair"]["arguments"].get(k) != bad[k]]
    if len(changed) != 1:
        raise ValueError("one-argument contrast required")
    field = changed[0]
    start = raw.index(f"<parameter={field}>")
    prefix, suffix = raw[:start], raw[start:]
    fixed = prefix + suffix.replace(str(bad[field]), str(case["repair"]["arguments"][field]), 1)
    repaired = tokenizer.encode(fixed, add_special_tokens=False)
    divergence = next(i for i, (a, b) in enumerate(zip(row["action_ids"], repaired, strict=False)) if a != b)
    return repaired, divergence


def projection(gradients, reference):
    return -sum((g.detach().float() * r).sum().item()
                for g, r in zip(gradients, reference, strict=True) if g is not None)


def score_case(model, case, tokenizer, gradients):
    row = case["row"]
    repaired, pos = corrected_action(case, tokenizer)
    original = row["action_ids"]
    bad, good = original[pos], repaired[pos]
    if max(len(row[p]) + len(row[t]) for p, t in
           [("student_prefix_ids", "action_ids"), ("teacher_prefix_ids", "action_ids"),
            ("feedback_prefix_ids", "feedback_ids")]) > 16384:
        raise ValueError("overlong diagnostic; no truncation")
    model.eval()
    model.set_adapter("self_teacher")
    with torch.no_grad():
        teacher = logits(model, row["teacher_prefix_ids"], original).detach()
    model.set_adapter("default")
    model.train()
    params = [p for n, p in model.named_parameters() if ".default." in n and p.requires_grad]
    student = logits(model, row["student_prefix_ids"], original)
    p, q = student[pos].float().log_softmax(-1), teacher[pos].float().log_softmax(-1)
    result = {"position": pos, "action_tokens": len(original),
              "bad_token": tokenizer.decode([bad]), "correct_token": tokenizer.decode([good]),
              "token_signal": compare(p.detach(), q, bad, good)}
    if not gradients:
        return result
    margin = p[good] - p[bad]
    margin_grads = torch.autograd.grad(margin, params, retain_graph=True, allow_unused=True)
    reference = [g.detach().float() if g is not None else torch.zeros_like(v, dtype=torch.float32)
                 for g, v in zip(margin_grads, params, strict=True)]
    action = chunked_reverse_kl(student, teacher, recompute=False)
    action_grads = torch.autograd.grad(action, params, allow_unused=True)
    action_projection = projection(action_grads, reference)
    result["action_kl"] = float(action.detach())
    del student, teacher, p, q, margin, margin_grads, action_grads, action
    feedback_logits = logits(model, row["feedback_prefix_ids"], row["feedback_ids"])
    target = torch.tensor(row["feedback_ids"], device="cuda")
    feedback = feedback_cross_entropy(feedback_logits, target, recompute=False)
    feedback_grads = torch.autograd.grad(feedback, params, allow_unused=True)
    feedback_projection = projection(feedback_grads, reference)
    result.update(feedback_ce=float(feedback.detach()),
                  gradient_margin_direction={"action_kl": action_projection,
                                             "feedback_ce_unweighted": feedback_projection,
                                             "feedback_ce_weighted": .2 * feedback_projection,
                                             "combined": action_projection + .2 * feedback_projection},
                  gradient_scope="negative dot product with correct-minus-bad log-probability gradient; "
                                 "positive improves argument under infinitesimal SGD, not an actual Adam/batch update")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cases = cases_from(args.campaign, args.step)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    reports = []
    for case in cases:
        row = case["row"]
        teacher_text = tokenizer.decode(row["teacher_prefix_ids"], skip_special_tokens=False)
        feedback_text = tokenizer.decode(row["feedback_ids"], skip_special_tokens=False)
        message = teacher_message(case["call"], case["feedback"], case["call"].get("teacher_goals"))
        if message not in teacher_text or feedback_text != case["feedback"]:
            raise ValueError("saved teacher/feedback tokens differ from actual expected messages")
        reports.append({"task": case["task"], "episode_id": case["episode"]["id"],
                        "bad_call": case["call"]["message"]["tool_calls"][0]["function"],
                        "repair": case["repair"], "same_state_replay": replay(case),
                        "teacher_message": message, "feedback_target": feedback_text,
                        "sample_exposures": case["sample_exposures"], "task_exposures": case["task_exposures"],
                        "completed_updates": case["completed_updates"],
                        "round_exposures_consumed": case["round_exposures_consumed"],
                        "action_tokens": len(row["action_ids"]), "teacher_tokens_verified": True,
                        "scores": []})
    (args.output / "prompt-and-replay-audit.json").write_text(json.dumps(reports, indent=2))
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
    checkpoint = args.campaign / "checkpoints" / f"checkpoint-{args.step}"
    manifest = json.loads((checkpoint / "campaign-checkpoint.json").read_text())
    for f in ["adapter_model.safetensors", "self_teacher/adapter_model.safetensors"]:
        if hashlib.sha256((checkpoint / f).read_bytes()).hexdigest() != manifest["file_sha256"][f]:
            raise ValueError("checkpoint student/teacher digest mismatch")
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    model.load_adapter(checkpoint / "self_teacher", adapter_name="self_teacher", is_trainable=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for case, report in zip(cases, reports, strict=True):
        repaired, pos = corrected_action(case, tokenizer)
        row = case["row"]
        with torch.no_grad(), model.disable_adapter():
            model.eval()
            initial = logits(model, row["student_prefix_ids"], row["action_ids"][:pos + 1])[pos].float()
            report["base_correct_minus_bad_logit"] = float(initial[repaired[pos]] - initial[row["action_ids"][pos]])
        result = score_case(model, case, tokenizer, gradients=True)
        report["scores"].append({"checkpoint": args.step, **result})
        (args.output / "gradient-audit.json").write_text(json.dumps(reports, indent=2))
        print(json.dumps({"task": case["task"], **result}), flush=True)
    (args.output / "receipt.json").write_text(json.dumps({"completed": True, "cases": len(cases),
        "checkpoint": args.step, "optimizer_updates": 0, "external_calls": 0,
        "scope": "four actual retained failures; replay, exact teacher inputs, separate parameter gradients"}, indent=2))


if __name__ == "__main__":
    main()
