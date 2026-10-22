"""Zero-update capacity gates for the actual SDPO actor and optimized rollout."""

import json
import math
import time

import torch


async def qualify_rollout(args, model, tokenizer, rollout):
    started = time.monotonic()
    seed = tokenizer.encode("A local capacity probe uses neutral repeated text. ", add_special_tokens=False)
    length = args.context_limit - args.reply_limit
    prefix = (seed * (length // len(seed) + 1))[:length]
    target = (seed * (args.reply_limit // len(seed) + 1))[:args.reply_limit]
    parity = await rollout.score_probe(model, prefix, target[:16])
    action, complete, evidence = await rollout.generate(prefix, max_tokens=args.reply_limit)
    if not action:
        raise ValueError("near-limit rollout returned no action tokens")
    concurrency = getattr(args, "concurrency", 32)
    batch = await rollout.batch_probe(prefix, maximum=concurrency)
    receipt = {"context_limit": args.context_limit, "reply_limit": args.reply_limit,
               "generation_prompt_tokens": len(prefix), "generated_tokens": len(action),
               "complete": complete, "generation": evidence, "parity": parity,
               "batch_invariance": batch, "elapsed_s": time.monotonic() - started,
               "scope": f"Near-limit prefix, one up-to-reply-cap generation and{concurrency}in-flight16-token probes; "
                        "not simultaneous forced4K replies or a throughput claim"}
    (args.output / "capacity-rollout.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return prefix, target, receipt


def qualify_actor(args, model, tokenizer, capacity, teacher_logits, loss_for, score_sampled):
    """Run while inference is suspended, immediately before the first native update."""
    started = time.monotonic()
    prefix, target, rollout_receipt = capacity
    hint = tokenizer.encode("Use the observed result to choose a correct next action. ", add_special_tokens=False)
    teacher_prefix = prefix[:-len(hint)] + hint
    row = {"student_prefix_ids": prefix, "teacher_prefix_ids": teacher_prefix, "action_ids": target}
    snapshot = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    row["sampled_logprobs"] = score_sampled(model, row).tolist()
    teacher = teacher_logits(model, row)
    model.train()
    loss, _ = loss_for(model, row, args, teacher=teacher)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite full-context SDPO capacity loss")
    loss.backward()
    gradients = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    if (not gradients or any(name not in snapshot or not torch.isfinite(g).all() for name, g in gradients)
            or not any(g.abs().max() > 0 for _, g in gradients)):
        raise ValueError("invalid full-context student gradient or frozen teacher isolation")
    parameters = dict(model.named_parameters())
    if any(not torch.equal(parameters[name], old) for name, old in snapshot.items()):
        raise ValueError("zero-update capacity gate changed student weights")
    receipt = {"qualified": True, "context_limit": args.context_limit, "reply_limit": args.reply_limit,
               "student_tokens": len(prefix) + len(target), "teacher_tokens": len(teacher_prefix) + len(target),
               "scored_target_tokens": len(target), "loss": float(loss.detach()),
               "actor_peak_cuda_bytes": torch.cuda.max_memory_allocated(),
               "actor_peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
               "finite_nonzero_student_gradients": True, "weights_unchanged": True,
               "optimizer_updates": 0, "external_model_calls": 0,
               "rollout_probe_elapsed_s": rollout_receipt["elapsed_s"],
               "actor_elapsed_s": time.monotonic() - started,
               "scope": "Actual selected SDPO loss/backward at full context while native engine is suspended; "
                        "controlled capacity evidence, not native learning or reward"}
    if not math.isfinite(receipt["loss"]):
        raise ValueError("nonfinite capacity receipt")
    model.zero_grad(set_to_none=True)
    (args.output / "capacity-qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
