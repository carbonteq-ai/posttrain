"""Measure per-case parameter-gradient directions without an optimizer update.

Generated teacher tool calls are diagnostic references, never supervised targets.
Their correctness needs an independent native/source review before interpretation.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


def run(args):
    import torch
    from peft import PeftModel
    from probe_combined_teacher_tool_gradients import tool_positions
    from sdpo_agent_ablation import loss_for, teacher_logits
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    receipt = json.loads((args.root / "receipt.json").read_text())
    if not receipt["runtime_completed"] or receipt["optimizer_updates"] != 0:
        raise ValueError("complete zero-update native/context qualification required")
    if (Path(args.model).name != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
            or hashlib.sha256((args.root / "rollout-policy-v0/adapter_model.safetensors").read_bytes()).hexdigest()
            != "e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c"):
        raise ValueError("pinned fresh base/adapter required")
    cases = {c["row"]["task"]: c for c in json.loads((args.root / "source-cases.json").read_text())["cases"]}
    proposals = {p["task"]: p for p in json.loads((args.root / "same-decision-proposals.json").read_text())
                 if p["mode"] == "revised_teacher"}
    if len(args.task) != len(set(args.task)) or not set(args.task) <= set(cases):
        raise ValueError("unique, present, predeclared diagnostic tasks required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(args.model,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True),
        args.root / "rollout-policy-v0", is_trainable=True)
    model.load_adapter(args.root / "rollout-policy-v0", adapter_name="sdpo_teacher", is_trainable=False)
    model.set_adapter("default")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    parameters = [p for name, p in model.named_parameters() if ".default." in name and p.requires_grad]
    if not parameters or any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()):
        raise ValueError("trainable zero-dropout student required")
    opening, closing = [tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call>", "</tool_call>")]
    rows = []
    for name in args.task:
        case, proposal = cases[name], proposals[name]
        row = copy.deepcopy(case["row"])
        row["teacher_prefix_ids"] = case["revised_teacher_prefix_ids"]
        result = {"task": name, "episode_id": row["episode_id"], "generation_index": row["generation_index"],
                  "teacher_proposal_complete": proposal["complete"], "teacher_native_error": proposal.get("native_error")}
        if not proposal["complete"] or proposal.get("native_error") or not proposal.get("message"):
            result.update(measurable=False, reason="incomplete, unparsed or unreplayed teacher proposal")
        else:
            target = proposal["action_ids"]
            positions = tool_positions(target, opening, closing)
            if not positions:
                result.update(measurable=False, reason="teacher produced no complete tool span")
            elif max(len(row["teacher_prefix_ids"]) + len(row["action_ids"]),
                     len(row["student_prefix_ids"]) + len(target)) > 32768:
                result.update(measurable=False, reason="whole transition overlong; no truncation")
            else:
                teacher = teacher_logits(model, row)
                loss, _ = loss_for(model, row, SimpleNamespace(topk=20, alpha=1.), teacher=teacher)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite SDPO diagnostic loss")
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
                loss_value = float(loss.detach())
                del loss, teacher
                model.train()
                ids = torch.tensor([row["student_prefix_ids"] + target[:-1]], device="cuda")
                support = torch.tensor([len(row["student_prefix_ids"]) - 1 + i for i in positions], device="cuda")
                selected = torch.tensor([target[i] for i in positions], device="cuda")
                values = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                               logits_to_keep=support).logits[0].float()
                logprob = values.log_softmax(-1).gather(-1, selected[:, None]).mean()
                reference = torch.autograd.grad(logprob, parameters, allow_unused=True)
                direction = sum(-float((g.float() * r.float()).sum())
                                for g, r in zip(gradients, reference, strict=True) if g is not None and r is not None)
                norm2 = sum(float(g.float().square().sum()) for g in gradients if g is not None)
                if not math.isfinite(direction) or not math.isfinite(norm2) or norm2 <= 0:
                    raise ValueError("nonfinite/zero parameter-gradient diagnostic")
                result.update(measurable=True, loss=loss_value, parameter_gradient_norm=norm2 ** .5,
                    tool_tokens=len(positions), teacher_tool_logprob_under_student=float(logprob.detach()),
                    negative_loss_gradient_dot_teacher_tool_logprob_gradient=direction,
                    desired_direction_if_reference_is_correct="positive",
                    student_prefix_sha256=hashlib.sha256(json.dumps(row["student_prefix_ids"]).encode()).hexdigest(),
                    teacher_prefix_sha256=hashlib.sha256(json.dumps(row["teacher_prefix_ids"]).encode()).hexdigest(),
                    reference_action_sha256=hashlib.sha256(json.dumps(target).encode()).hexdigest())
                del gradients, reference, values, logprob, ids, support, selected
                torch.cuda.empty_cache()
        rows.append(result)
        (args.output / "directions.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    summary = {"completed": True, "optimizer_updates": 0, "tasks": args.task, "results": rows,
        "source_proposals_sha256": hashlib.sha256((args.root / "same-decision-proposals.json").read_bytes()).hexdigest(),
        "scope": "Isolated per-case infinitesimal SGD parameter direction toward generated tool spans, conditioned on the proposal's reasoning. Not full-batch Adam, a correctness label, actual student rollout improvement or generalization. No reference enters the loss; no optimizer exists."}
    (args.output / "receipt.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
