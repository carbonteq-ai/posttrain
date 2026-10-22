"""Adaptive experiment01: argument-only SFT learnability control, not SELF."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from aiohttp import web
from peft import PeftModel
from self_campaign import Campaign, write
from self_corrected_call_ablation import RETENTION, retention_rows
from self_failure_diagnosis import cases_from, corrected_action, key, logits, replay
from self_feedback_signals import repair_candidates
from self_feedback_vllm import LocalVLLMRollout
from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
from trl.experimental.self_feedback.self_trainer import chunked_reverse_kl, feedback_cross_entropy
from verifiers_preinstalled import configure_preinstalled_runtime


def first_cases(campaign, retries):
    folder = campaign / "step-0064/train"
    wanted = {c["task"] for c in retries}
    episodes = [json.loads(line) for line in (folder / "native/traces.jsonl").read_text().splitlines()]
    tools = {n["message"]["tool_call_id"]: (e, n["message"])
             for e in episodes for t in e["traces"] for n in t["nodes"]
             if (n.get("message") or {}).get("role") == "tool"}
    rows = {key(r): r for r in json.loads((folder / "transitions.json").read_text())}
    found = {}
    for line in (folder / "generation-calls.jsonl").read_text().splitlines():
        call = json.loads(line)
        calls = call["message"].get("tool_calls", [])
        if not call["complete"] or len(calls) != 1 or calls[0]["id"] not in tools:
            continue
        episode, response = tools[calls[0]["id"]]
        name = episode["task"]["data"]["name"]
        if name not in wanted or name in found:
            continue
        # Actual first decision, not a retry after extra observation context.
        if any(m.get("role") == "tool" for m in call["messages"]):
            continue
        feedback = json.dumps([{"tool": response.get("name"), "response": response["content"]}], sort_keys=True)
        repairs = repair_candidates(call, feedback, call.get("teacher_goals"))
        row = rows.get((episode["id"], tuple(call["prefix_ids"]), tuple(call["action_ids"])))
        if len(repairs) != 1 or row is None:
            raise ValueError("first-decision repair not replayable/admitted")
        found[name] = {"task": name, "episode": episode, "call": call,
                       "feedback": feedback, "repair": repairs[0], "row": row}
    if set(found) != wanted:
        raise ValueError("missing first-decision evidence")
    return [found[c["task"]] for c in retries]


def changed_span(original, repaired):
    start = next(i for i, (a, b) in enumerate(zip(original, repaired, strict=False)) if a != b)
    suffix = 0
    while (suffix < min(len(original), len(repaired)) - start
           and original[-suffix - 1] == repaired[-suffix - 1]):
        suffix += 1
    end = len(repaired) - suffix
    if end <= start:
        raise ValueError("empty correction span")
    return start, end


def corrected_ce(model, case, tokenizer):
    repaired, _ = corrected_action(case, tokenizer)
    prefix, original = case["row"]["student_prefix_ids"], case["row"]["action_ids"]
    if len(prefix) + len(repaired) > 16384:
        raise ValueError("no overlong supervision or truncation")
    start, end = changed_span(original, repaired)
    scores = logits(model, prefix, repaired)[start:end]
    target = torch.tensor(repaired[start:end], device="cuda")
    return feedback_cross_entropy(scores, target, recompute=False)


def repair_teacher_kl(model, case, tokenizer, full_span=False, prefix_anchor=False, return_parts=False):
    """Soft privileged teacher at verified repair positions; no hard-token CE."""
    if prefix_anchor and not full_span:
        raise ValueError("prefix anchor requires full corrected span")
    repaired, pos = corrected_action(case, tokenizer)
    row = case["row"]
    if full_span:
        start, end = changed_span(row["action_ids"], repaired)
        target = repaired
    else:
        start, end = pos, pos + 1
        target = row["action_ids"][:pos + 1]
    if max(len(row[k]) + len(target) for k in ["student_prefix_ids", "teacher_prefix_ids"]) > 16384:
        raise ValueError("no truncation of teacher-position loss")
    model.set_adapter("self_teacher")
    model.eval()
    with torch.no_grad():
        q = logits(model, row["teacher_prefix_ids"], target)[start:end].float().log_softmax(-1)
    model.set_adapter("default")
    model.train()
    p = logits(model, row["student_prefix_ids"], target)[start:end].float().log_softmax(-1)
    per_token = (q.exp() * (q - p)).sum(-1)
    span, first = per_token.mean(), per_token[0]
    loss = span + first if prefix_anchor else span
    if return_parts:
        return loss, {"span_forward_kl": float(span.detach()), "first_position_forward_kl": float(first.detach()),
                      "prefix_anchor_weight": float(prefix_anchor)}
    return loss


def equalize_task_gradients(groups):
    """Equalize each repair task's total gradient norm to the median, then sum."""
    norms = [sum(float((g.double() ** 2).sum()) for g in group) ** .5 for group in groups]
    if not norms or any(not (0 < n < float("inf")) for n in norms):
        raise ValueError("finite nonzero task gradients required")
    ordered = sorted(norms)
    middle = len(ordered) // 2
    median = (ordered[middle] + ordered[(len(ordered) - 1) // 2]) / 2
    scales = [median / n for n in norms]
    combined = [sum((group[i] * scale for group, scale in zip(groups, scales, strict=True)),
                    torch.zeros_like(groups[0][i])) for i in range(len(groups[0]))]
    return combined, norms, scales


@torch.no_grad()
def teacher_span_audit(model, cases, tokenizer):
    """Teacher-forced token audit, not a claim of freely generated correct calls."""
    reports = []
    model.eval()
    model.set_adapter("self_teacher")
    try:
        for case in cases:
            fixed, _ = corrected_action(case, tokenizer)
            row = case["row"]
            start, end = changed_span(row["action_ids"], fixed)
            if len(row["teacher_prefix_ids"]) + len(fixed) > 16384:
                raise ValueError("no overlong teacher audit or truncation")
            scores = logits(model, row["teacher_prefix_ids"], fixed)[start:end].float()
            tokens = []
            for i, token in enumerate(fixed[start:end]):
                score = scores[i]
                tokens.append({"position": start + i, "expected": tokenizer.decode([token]),
                               "teacher_top1": tokenizer.decode([int(score.argmax())]),
                               "correct_rank": int((score > score[token]).sum()) + 1,
                               "correct_probability": float(score.softmax(-1)[token])})
            reports.append({"task": case["task"], "history_tool_messages": sum(m.get("role") == "tool"
                            for m in case["call"]["messages"]),
                            "corrected_span": tokenizer.decode(fixed[start:end]), "tokens": tokens,
                            "scope": "teacher-forced corrected branch; full native/free teacher generation remains unproven"})
    finally:
        model.set_adapter("default")
    return reports


def self_terms(model, row, kl_weight, feedback_weight):
    result = torch.zeros((), device="cuda")
    metrics = {"action_kl": 0., "feedback_ce": 0.}
    if kl_weight:
        model.eval()
        model.set_adapter("self_teacher")
        with torch.no_grad():
            teacher = logits(model, row["teacher_prefix_ids"], row["action_ids"])
        model.set_adapter("default")
        model.train()
        action = chunked_reverse_kl(logits(model, row["student_prefix_ids"], row["action_ids"]),
                                    teacher, recompute=False)
        result = result + kl_weight * action
        metrics["action_kl"] = float(action.detach())
    if feedback_weight:
        feedback = feedback_cross_entropy(
            logits(model, row["feedback_prefix_ids"], row["feedback_ids"]),
            torch.tensor(row["feedback_ids"], device="cuda"), recompute=False)
        result = result + feedback_weight * feedback
        metrics["feedback_ce"] = float(feedback.detach())
    return result, metrics


@torch.no_grad()
def margins(model, cases, tokenizer):
    model.eval()
    model.set_adapter("default")
    result = []
    for c in cases:
        fixed, pos = corrected_action(c, tokenizer)
        row = c["row"]
        scores = logits(model, row["student_prefix_ids"], row["action_ids"][:pos + 1])[pos].float()
        top = scores.topk(5)
        result.append({"task": c["task"], "history_tool_messages": sum(m.get("role") == "tool"
                        for m in c["call"]["messages"]),
                       "correct_minus_bad_logit": float(scores[fixed[pos]] - scores[row["action_ids"][pos]]),
                       "correct_token_rank": int((scores > scores[fixed[pos]]).sum()) + 1,
                       "top_tokens": [{"token_id": int(i), "text": tokenizer.decode([int(i)]),
                                       "logit": float(v)} for i, v in zip(top.indices, top.values, strict=True)],
                       "scope": "one next-token context on recorded prefix; full native rollout remains necessary"})
    return result


async def main(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "inputs.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    retries = cases_from(args.campaign, 64)
    cases = first_cases(args.campaign, retries) + retries
    controls = retention_rows(args.campaign)
    heldout = set(json.loads(args.heldout.read_text())["task_names"])
    if (set(RETENTION) | {c["task"] for c in cases}) & heldout:
        raise ValueError("heldout used")
    for c in cases:
        replay(c)
    write(args.output / "repair-provenance.json", [{"task": c["task"], "repair": c["repair"],
          "history_tool_messages": sum(m.get("role") == "tool" for m in c["call"]["messages"])} for c in cases])
    os.environ["POSTTRAIN_VERIFIERS_PREINSTALLED"] = "1"
    os.environ["SELF_LOCAL_KEY"] = "EMPTY"
    configure_preinstalled_runtime(interpreter=sys.executable)
    set_seed(17)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
    checkpoint = args.campaign / "checkpoints/checkpoint-64"
    manifest = json.loads((checkpoint / "campaign-checkpoint.json").read_text())
    if hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest() != manifest["file_sha256"]["adapter_model.safetensors"]:
        raise ValueError("student checkpoint digest mismatch")
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    use_teacher = bool(args.kl_weight or args.repair_objective.startswith("teacher_forward_"))
    hybrid = bool(use_teacher or args.feedback_weight)
    if use_teacher or args.audit_teacher_span:
        teacher_file = "self_teacher/adapter_model.safetensors"
        if hashlib.sha256((checkpoint / teacher_file).read_bytes()).hexdigest() != manifest["file_sha256"][teacher_file]:
            raise ValueError("teacher checkpoint digest mismatch")
        model.load_adapter(checkpoint / "self_teacher", adapter_name="self_teacher", is_trainable=False)
        model.set_adapter("default")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    args.steps, args.updates_per_round, args.batch_size = 10, 10, 16
    args.context_limit, args.reply_limit, args.feedback_mode = 16384, 2048, "minimal"
    campaign = Campaign(args)
    campaign.model, campaign.tokenizer = model, tokenizer
    rollout = await LocalVLLMRollout.create(args.model, model, tokenizer, args.output, 16384, 32, 16)
    campaign.rollout = rollout
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", campaign.completions)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18774).start()

    async def collect(step):
        for cohort, names in [("failures", [c["task"] for c in retries]), ("retention", RETENTION)]:
            await campaign.collect(step, cohort, names)
        write(args.output / f"argument-margins-{step}.json", margins(model, cases, tokenizer))
        await rollout.suspend_for_update()

    logs = []
    try:
        await collect(0)
        if args.audit_teacher_span:
            write(args.output / "teacher-span-audit.json", teacher_span_audit(model, cases, tokenizer))
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)
        for step in range(1, 11):
            if time.monotonic() - started >= 7000:
                raise TimeoutError("stop before total2h cap")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            correction_losses, retention_losses = [], []
            self_metrics, repair_parts = [], []
            tasks = sorted({c["task"] for c in cases})
            groups = [[c for c in cases if c["task"] == task] for task in tasks] if args.normalize_repair_tasks else [cases]
            task_gradients = []
            for group in groups:
                if args.normalize_repair_tasks:
                    optimizer.zero_grad(set_to_none=True)
                for c in group:
                    if args.repair_objective.startswith("teacher_forward_"):
                        correction, parts = repair_teacher_kl(model, c, tokenizer,
                            full_span=args.repair_objective in ["teacher_forward_repair_span", "teacher_forward_prefix_anchor"],
                            prefix_anchor=args.repair_objective == "teacher_forward_prefix_anchor", return_parts=True)
                        repair_parts.append(parts)
                    else:
                        correction = corrected_ce(model, c, tokenizer)
                    extra, metrics = self_terms(model, c["row"], args.kl_weight, args.feedback_weight)
                    loss = correction + extra
                    (loss / 16).backward()
                    correction_losses.append(float(correction.detach()))
                    self_metrics.append(metrics)
                if args.normalize_repair_tasks:
                    task_gradients.append([p.grad.detach().float().cpu().clone() if p.grad is not None
                        else torch.zeros_like(p, device="cpu", dtype=torch.float32) for p in parameters])
            norm_log = None
            if args.normalize_repair_tasks:
                combined, norms, scales = equalize_task_gradients(task_gradients)
                for p, g in zip(parameters, combined, strict=True):
                    p.grad = g.to(device=p.device, dtype=p.dtype)
                norm_log = {task: {"norm": norm, "scale": scale} for task, norm, scale in zip(tasks, norms, scales, strict=True)}
            for _repeat in range(2):
                for row in controls:
                    scores = logits(model, row["student_prefix_ids"], row["action_ids"])
                    target = torch.tensor(row["action_ids"], device="cuda")
                    loss = feedback_cross_entropy(scores, target, recompute=False)
                    (loss / 16).backward()
                    retention_losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite gradients")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if use_teacher:
                with torch.no_grad():
                    named = dict(model.named_parameters())
                    for name, parameter in named.items():
                        if ".self_teacher." in name:
                            parameter.lerp_(named[name.replace(".self_teacher.", ".default.")], .01)
            log = {"step": step, "batch_size": 16, "repair_objective": args.repair_objective,
                   "repair_loss": sum(correction_losses) / 8,
                   "repair_component_means": {k: sum(x[k] for x in repair_parts) / 8 for k in repair_parts[0]} if repair_parts else None,
                   "argument_ce": sum(correction_losses) / 8 if args.repair_objective == "argument_ce" else None,
                   "task_gradient_normalization": norm_log,
                   "retention_ce": sum(retention_losses) / 8, "grad_norm": float(norm),
                   "self_component_means": {k: sum(x[k] for x in self_metrics) / 8 for k in self_metrics[0]},
                   "ema_updates": step if use_teacher else 0}
            logs.append(log)
            write(args.output / "updates.json", logs)
            print(json.dumps({"stage": "update", **log}), flush=True)
            if args.audit_every_update:
                write(args.output / f"actual-update-margins-{step}.json", margins(model, cases, tokenizer))
            if step in [5, 10]:
                model.save_pretrained(args.output / f"checkpoint-{step}")
                await rollout.refresh()
                await collect(step)
        write(args.output / "receipt.json", {"completed": True, "experiment": args.run_name,
              "optimizer_updates": 10, "external_calls": 0, "elapsed_s": time.monotonic() - started,
              "repair_objective": args.repair_objective, "task_gradient_normalization": args.normalize_repair_tasks,
              "prefix_anchor_weight": float(args.repair_objective == "teacher_forward_prefix_anchor"),
              "loss_log_scope": "Unweighted component means; detached task normalization scales alter the update gradient when enabled.",
              "scope": ("prefix-anchored corrected-argument soft distillation with supervised retention; augmented offline algorithm, not paper SELF"
                        if args.repair_objective == "teacher_forward_prefix_anchor" else
                        "full corrected-argument-branch soft-teacher forward KL with supervised retention; augmented offline algorithm, not paper SELF"
                        if args.repair_objective == "teacher_forward_repair_span" else
                        "first repair-position soft-teacher forward KL with supervised retention; augmented algorithm, not paper SELF"
                        if args.repair_objective == "teacher_forward_position" else
                        "argument-supervised SELF hybrid; four trained failures/four retention tasks" if hybrid else
                        "pure supervised learnability control, not SELF; four trained failures/four retention tasks")})
    finally:
        await runner.cleanup()
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", default="self-adaptive01-argument-sft-control")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=float, default=0.)
    parser.add_argument("--feedback-weight", type=float, default=0.)
    parser.add_argument("--repair-objective", choices=("argument_ce", "teacher_forward_position", "teacher_forward_repair_span", "teacher_forward_prefix_anchor"), default="argument_ce")
    parser.add_argument("--normalize-repair-tasks", action="store_true")
    parser.add_argument("--audit-every-update", action="store_true")
    parser.add_argument("--audit-teacher-span", action="store_true")
    args = parser.parse_args()
    if not (0 < args.lr <= 3e-4) or min(args.kl_weight, args.feedback_weight) < 0:
        parser.error("invalid bounded ablation settings")
    asyncio.run(main(args))
