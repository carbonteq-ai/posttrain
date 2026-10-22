"""Zero-update component/batch diagnosis; projections are not rollout qualification."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from self_argument_repair_control import corrected_ce, first_cases
from self_corrected_call_ablation import RETENTION, ce, retention_rows
from self_failure_diagnosis import cases_from, corrected_action, logits, replay
from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
from trl.experimental.self_feedback.self_trainer import chunked_reverse_kl


def forward_kl(student, teacher):
    """KL(teacher || student), teacher detached; a deliberate SELF variant."""
    p = student.float().log_softmax(-1)
    q = teacher.detach().float().log_softmax(-1)
    return (q.exp() * (q - p)).sum(-1).mean()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def component(model, case, tokenizer, name):
    row = case["row"]
    if max(len(row[p]) + len(row[t]) for p, t in [
        ("student_prefix_ids", "action_ids"), ("teacher_prefix_ids", "action_ids"),
        ("feedback_prefix_ids", "feedback_ids")]) > 16384:
        raise ValueError("overlong diagnostic; no truncation")
    if name == "argument_ce":
        return corrected_ce(model, case, tokenizer)
    if name == "feedback_ce":
        return ce(model, row["feedback_prefix_ids"], row["feedback_ids"])
    model.set_adapter("self_teacher")
    model.eval()
    with torch.no_grad():
        teacher = logits(model, row["teacher_prefix_ids"], row["action_ids"])
    model.set_adapter("default")
    model.train()
    student = logits(model, row["student_prefix_ids"], row["action_ids"])
    if name == "reverse_kl":
        return chunked_reverse_kl(student, teacher, recompute=False)
    if name == "forward_kl":
        return forward_kl(student, teacher)
    if name == "repair_position_forward_kl":
        _, pos = corrected_action(case, tokenizer)
        return forward_kl(student[pos:pos + 1], teacher[pos:pos + 1])
    raise ValueError(name)


def main(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(17)
    torch.set_num_threads(2)
    retries = cases_from(args.campaign, 64)
    cases = first_cases(args.campaign, retries) + retries
    controls = retention_rows(args.campaign)
    heldout = set(json.loads(args.heldout.read_text())["task_names"])
    if ({c["task"] for c in cases} | set(RETENTION)) & heldout:
        raise ValueError("heldout contamination")
    for case in cases:
        replay(case)
    checkpoint = args.campaign / "checkpoints/checkpoint-64"
    manifest = json.loads((checkpoint / "campaign-checkpoint.json").read_text())
    for filename in ["adapter_model.safetensors", "self_teacher/adapter_model.safetensors"]:
        if hashlib.sha256((checkpoint / filename).read_bytes()).hexdigest() != manifest["file_sha256"][filename]:
            raise ValueError("checkpoint digest mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    model.load_adapter(checkpoint / "self_teacher", adapter_name="self_teacher", is_trainable=False)
    model.set_adapter("default")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for n, p in model.named_parameters() if ".default." in n and p.requires_grad]
    before = [p.detach().cpu().clone() for p in params]
    gradients, losses = {}, {}
    for name in ["argument_ce", "retention_ce", "reverse_kl", "feedback_ce", "forward_kl",
                 "repair_position_forward_kl"]:
        if time.monotonic() - started > 6800:
            raise TimeoutError("bounded total runtime")
        model.set_adapter("default")
        model.train()
        model.zero_grad(set_to_none=True)
        values = []
        examples = controls * 2 if name == "retention_ce" else cases
        for case in examples:
            loss = (ce(model, case["student_prefix_ids"], case["action_ids"])
                    if name == "retention_ce" else component(model, case, tokenizer, name))
            (loss / 16).backward()
            values.append(float(loss.detach()))
        gradients[name] = [p.grad.detach().float().cpu().clone() if p.grad is not None
                           else torch.zeros_like(p, device="cpu", dtype=torch.float32) for p in params]
        if not all(torch.isfinite(g).all() for g in gradients[name]):
            raise ValueError("nonfinite batch gradient")
        losses[name] = sum(values) / len(values)
        save(args.output / "component-losses.json", losses)
        print(json.dumps({"stage": "component_gradient", "component": name, "loss": losses[name]}), flush=True)
    reports = []
    model.zero_grad(set_to_none=True)
    for case in cases:
        row = case["row"]
        fixed, pos = corrected_action(case, tokenizer)
        model.set_adapter("self_teacher")
        model.eval()
        with torch.no_grad():
            teacher = logits(model, row["teacher_prefix_ids"], row["action_ids"][:pos + 1])[pos].float()
        model.set_adapter("default")
        scores = logits(model, row["student_prefix_ids"], row["action_ids"][:pos + 1])[pos].float()
        good, bad = fixed[pos], row["action_ids"][pos]
        competitors = scores.detach().clone()
        competitors[good] = -torch.inf
        top_other = int(competitors.argmax())
        report = {"task": case["task"], "tool_history": sum(m.get("role") == "tool" for m in case["call"]["messages"]),
                  "correct_token": tokenizer.decode([good]), "bad_token": tokenizer.decode([bad]),
                  "top_other_token": tokenizer.decode([top_other]),
                  "student_correct_rank": int((scores > scores[good]).sum()) + 1,
                  "teacher_correct_rank": int((teacher > teacher[good]).sum()) + 1,
                  "student_correct_probability": float(scores.detach().softmax(-1)[good]),
                  "teacher_correct_probability": float(teacher.softmax(-1)[good]),
                  "student_top5": [tokenizer.decode([int(i)]) for i in scores.topk(5).indices],
                  "teacher_top5": [tokenizer.decode([int(i)]) for i in teacher.topk(5).indices],
                  "projections": {}}
        for index, (label, other) in enumerate([("correct_vs_recorded_bad", bad), ("correct_vs_top_other", top_other)]):
            refs = torch.autograd.grad(scores[good] - scores[other], params, retain_graph=index == 0, allow_unused=True)
            direction = {name: -sum(float((r.detach().float().cpu() * g).sum())
                         for r, g in zip(refs, grads, strict=True) if r is not None)
                         for name, grads in gradients.items()}
            direction["self_components_plus_supervised_retention"] = direction["reverse_kl"] + .2 * direction["feedback_ce"] + direction["retention_ce"]
            direction["sft_control"] = direction["argument_ce"] + direction["retention_ce"]
            direction["tested_hybrid"] = direction["sft_control"] + direction["reverse_kl"] + .2 * direction["feedback_ce"]
            direction["forward_variant_plus_retention"] = direction["forward_kl"] + direction["retention_ce"]
            direction["repair_position_variant_plus_retention"] = direction["repair_position_forward_kl"] + direction["retention_ce"]
            report["projections"][label] = direction
        reports.append(report)
        save(args.output / "batch-directions.json", reports)
    if not all(torch.equal(p.detach().cpu(), original) for p, original in zip(params, before, strict=True)):
        raise ValueError("zero-update diagnostic changed student weights")
    save(args.output / "receipt.json", {"completed": True, "optimizer_updates": 0, "external_calls": 0,
         "elapsed_s": time.monotonic() - started, "batch_exposures": 16,
         "failure_exposures": len(cases), "retention_exposures": len(controls) * 2,
         "distinct_failure_contexts": len({tuple(c["row"]["student_prefix_ids"]) for c in cases}),
         "checkpoint_file_sha256": manifest["file_sha256"],
         "scope": "Batch component gradients on recorded contexts, infinitesimal SGD only; not Adam or native reward proof. "
                  "Repair-position masking uses verified repair locations and is an augmented variant, not paper SELF."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    for name in ["campaign", "heldout", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    main(parser.parse_args())
