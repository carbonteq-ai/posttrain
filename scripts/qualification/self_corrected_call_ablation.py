"""Six isolated four-update corrected-call/feedback-weight qualification arms."""

import argparse
import asyncio
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from aiohttp import web
from peft import PeftModel
from self_campaign import Campaign, write
from self_failure_diagnosis import cases_from, corrected_action, logits, projection, replay
from self_feedback_vllm import LocalVLLMRollout
from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
from trl.experimental.self_feedback.self_trainer import chunked_reverse_kl, feedback_cross_entropy
from verifiers_preinstalled import configure_preinstalled_runtime

RETENTION = ["simple.mailchimp_add_subscriber", "simple.sf_lead_revenue_update",
             "simple.trello_catering_contract", "finance.po_email_logging"]


def ce(model, prefix, target):
    if not target or len(prefix) + len(target) > 16384:
        raise ValueError("no incomplete or truncated supervision")
    return feedback_cross_entropy(logits(model, prefix, target), torch.tensor(target, device="cuda"), recompute=False)


def loss_parts(model, row, repaired):
    model.eval()
    model.set_adapter("self_teacher")
    with torch.no_grad():
        teacher = logits(model, row["teacher_prefix_ids"], row["action_ids"])
    model.set_adapter("default")
    model.train()
    action = chunked_reverse_kl(logits(model, row["student_prefix_ids"], row["action_ids"]),
                                teacher, recompute=False)
    feedback = ce(model, row["feedback_prefix_ids"], row["feedback_ids"])
    corrected = ce(model, row["student_prefix_ids"], repaired)
    return action, feedback, corrected


def directions(model, case, tokenizer):
    row = case["row"]
    repaired, pos = corrected_action(case, tokenizer)
    model.set_adapter("default")
    model.train()
    parameters = [p for n, p in model.named_parameters() if ".default." in n and p.requires_grad]
    margin_logits = logits(model, row["student_prefix_ids"], row["action_ids"][:pos + 1])[pos]
    margin = margin_logits[repaired[pos]].float() - margin_logits[row["action_ids"][pos]].float()
    gradient = torch.autograd.grad(margin, parameters, allow_unused=True)
    reference = [g.detach().float() if g is not None else torch.zeros_like(p, dtype=torch.float32)
                 for g, p in zip(gradient, parameters, strict=True)]
    del margin, margin_logits, gradient
    action, feedback, corrected = loss_parts(model, row, repaired)
    result = {}
    for index, (name, value) in enumerate([("action_kl", action), ("feedback_ce", feedback), ("corrected_ce", corrected)]):
        grads = torch.autograd.grad(value, parameters, allow_unused=True, retain_graph=index < 2)
        result[name] = projection(grads, reference)
    return result


def retention_rows(campaign):
    folder = campaign / "step-0000/train"
    episodes = [json.loads(line) for line in (folder / "native/traces.jsonl").read_text().splitlines()]
    identities = {}
    for e in episodes:
        name = e["task"]["data"]["name"]
        if name in RETENTION:
            if e["traces"][0]["metrics"]["task_completed_correctly"] != 1:
                raise ValueError("retention example not successful")
            identities[e["id"]] = name
    rows = {}
    for row in json.loads((folder / "transitions.json").read_text()):
        name = identities.get(row["trajectory_id"])
        if name and name not in rows:
            rows[name] = row
    if set(rows) != set(RETENTION):
        raise ValueError("missing successful retention rows")
    return [rows[n] for n in RETENTION]


def qualify(gradient_directions, baseline, after):
    gradient_pass = all(x > 0 for x in gradient_directions.values())
    checks = {}
    for cohort in ["failures", "retention"]:
        b, a = baseline[cohort]["summary"], after[cohort]["summary"]
        for name in ["task_completed_correctly", "partial_credit"]:
            bv = b["metrics"][name]["complete_task_means"]
            av = a["metrics"][name]["complete_task_means"]
            if set(bv) != set(av) or len(bv) != 4:
                raise ValueError("incomplete outcome comparison")
            checks[f"{cohort}_{name}_no_regression"] = all(av[n] >= bv[n] for n in bv)
        if cohort == "failures":
            checks["failure_partial_improved"] = (a["metrics"]["partial_credit"]["full_population_task_mean"]
                                                  > b["metrics"]["partial_credit"]["full_population_task_mean"])
            checks["failure_success_improved"] = (a["metrics"]["task_completed_correctly"]["full_population_task_mean"]
                                                  > b["metrics"]["task_completed_correctly"]["full_population_task_mean"])
    return {"gradient_pass": gradient_pass, "rollout_checks": checks,
            "qualified": gradient_pass and all(checks.values())}


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "inputs.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    cases = cases_from(args.campaign, 64)
    controls = retention_rows(args.campaign)
    heldout = set(json.loads(args.heldout.read_text())["task_names"])
    if (set(RETENTION) | {c["task"] for c in cases}) & heldout:
        raise ValueError("heldout contamination")
    for case in cases:
        replay(case)
    write(args.output / "replay-qualified.json", {c["task"]: c["repair"] for c in cases})
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
    for name in ["adapter_model.safetensors", "self_teacher/adapter_model.safetensors"]:
        if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != manifest["file_sha256"][name]:
            raise ValueError("checkpoint digest mismatch")
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    model.load_adapter(checkpoint / "self_teacher", adapter_name="self_teacher", is_trainable=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    snapshots = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}
    measured = {}
    for case in cases:
        measured[case["task"]] = directions(model, case, tokenizer)
        write(args.output / "component-gradient-directions.json", measured)
        print(json.dumps({"stage": "gradient", "task": case["task"], **measured[case["task"]]}), flush=True)
    args.steps, args.updates_per_round, args.batch_size = 4, 4, 16
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
        for cohort, names in [("failures", [c["task"] for c in cases]), ("retention", RETENTION)]:
            await campaign.collect(step, cohort, names)
        return {r["cohort"]: r for r in campaign.curve if r["step"] == step}

    outcomes = []
    try:
        baseline = await collect(0)
        await rollout.suspend_for_update()
        for arm, (corrected_weight, feedback_weight) in enumerate(
                [(c, f) for c in [0., 1.] for f in [0., .05, .2]], 1):
            name = f"corrected{corrected_weight:g}-feedback{feedback_weight:g}"
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in snapshots:
                        p.copy_(snapshots[n].to(p.device))
            model.set_adapter("default")
            parameters = [p for n, p in model.named_parameters() if ".default." in n and p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.)
            logs = []
            for step in range(1, 5):
                optimizer.zero_grad(set_to_none=True)
                metrics = []
                for _repeat in range(2):
                    for case in cases:
                        row = case["row"]
                        repaired, _ = corrected_action(case, tokenizer)
                        action, feedback, corrected = loss_parts(model, row, repaired)
                        loss = action + feedback_weight * feedback + corrected_weight * corrected
                        (loss / 16).backward()
                        metrics.append({"action_kl": float(action.detach()), "feedback_ce": float(feedback.detach()),
                                        "corrected_ce": float(corrected.detach()), "total": float(loss.detach())})
                        del action, feedback, corrected, loss
                    for row in controls:
                        model.set_adapter("default")
                        model.train()
                        loss = ce(model, row["student_prefix_ids"], row["action_ids"])
                        (loss / 16).backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite ablation gradient")
                optimizer.step()
                with torch.no_grad():
                    named = dict(model.named_parameters())
                    for n, p in named.items():
                        if ".self_teacher." in n:
                            p.lerp_(named[n.replace(".self_teacher.", ".default.")], .01)
                entry = {"step": step, "batch_size": 16, "grad_norm": float(norm),
                         "failure_means": {k: sum(m[k] for m in metrics) / 8 for k in metrics[0]}}
                logs.append(entry)
                write(args.output / f"{name}-updates.json", logs)
                print(json.dumps({"stage": "update", "arm": name, **entry}), flush=True)
            optimizer.zero_grad(set_to_none=True)
            arm_folder = args.output / name
            model.save_pretrained(arm_folder, safe_serialization=True)
            await rollout.refresh()
            model.eval()
            after = await collect(arm)
            await rollout.suspend_for_update()
            signal = {n: d["action_kl"] + feedback_weight * d["feedback_ce"] + corrected_weight * d["corrected_ce"]
                      for n, d in measured.items()}
            result = {"arm": name, "corrected_weight": corrected_weight, "feedback_weight": feedback_weight,
                      "initial_gradient_direction": signal, "outcomes": after,
                      **qualify(signal, baseline, after), "updates": logs}
            outcomes.append(result)
            write(args.output / "ablation-results.json", outcomes)
            print(json.dumps({"stage": "arm_complete", "arm": name, "qualified": result["qualified"]}), flush=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
        write(args.output / "receipt.json", {"completed": True, "arms": 6, "updates_per_arm": 4,
              "total_optimizer_updates": 24, "batch_size": 16, "external_calls": 0,
              "qualified_arms": [r["arm"] for r in outcomes if r["qualified"]],
              "scope": "failure-focused adaptation from checkpoint64; no heldout generalization proof; "
                       "full256 campaign not restarted"})
    finally:
        await runner.cleanup()
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", default="self-corrected-call-ablation")
    asyncio.run(main(parser.parse_args()))
