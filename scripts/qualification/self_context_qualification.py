"""Controlled 16K generation/scoring capacity gate; never training evidence."""

import json
import math
import time

import torch
from datasets import Dataset
from transformers import set_seed
from trl import SFTConfig
from trl.experimental.self_feedback import SELFTrainer


async def qualify_context(args, model, tokenizer, rollout):
    set_seed(17)
    seed = tokenizer.encode("A local context-capacity test uses repeated neutral text. ", add_special_tokens=False)
    reply_limit = getattr(args, "reply_limit", 512)
    prefix_length = args.context_limit - reply_limit
    prefix = (seed * (prefix_length // len(seed) + 1))[:prefix_length]
    target = (seed * (reply_limit // len(seed) + 1))[:reply_limit]
    parity = await rollout.score_probe(model, prefix, target[:16])
    action, complete, evidence = await rollout.generate(prefix, max_tokens=reply_limit)
    if not action:
        raise ValueError("near-limit vLLM generation returned no tokens")
    concurrency_probe = None
    if args.episode_concurrency > 1:
        concurrency_probe = await rollout.batch_probe(prefix, maximum=args.episode_concurrency)
    await rollout.suspend_for_update()
    row = {"trajectory_id": "controlled-capacity-test", "student_prefix_ids": prefix,
           "teacher_prefix_ids": prefix, "feedback_prefix_ids": prefix,
           "action_ids": target, "feedback_ids": target}
    trainer = SELFTrainer(
        model=model, processing_class=tokenizer, sequence_limit=args.context_limit,
        train_dataset=Dataset.from_list([row]), data_collator=lambda x: {"transitions": x},
        args=SFTConfig(output_dir=str(args.output / "controlled"), max_steps=1,
                       per_device_train_batch_size=1, bf16=True, report_to="none",
                       remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True},
                       max_length=None, save_strategy="no"),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss = trainer.compute_loss(model, {"transitions": [row]})
    loss.backward()
    torch.cuda.synchronize()
    gradients = [(name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None]
    if not math.isfinite(loss.item()) or not gradients or any(
        ".default." not in name or not torch.isfinite(gradient).all() for name, gradient in gradients
    ):
        raise ValueError("invalid long-context loss, gradient or frozen-parameter isolation")
    receipt = {"qualified": True, "context_limit": args.context_limit, "reply_limit": reply_limit,
               "episode_concurrency": args.episode_concurrency, "server_sequences": args.server_sequences,
               "kv_cache_gib": args.kv_cache_gib, "concurrency_probe": concurrency_probe,
               "scoring_sequence_tokens": len(prefix) + len(target),
               "scored_action_tokens": len(target), "scored_feedback_tokens": len(target),
               "generation_prompt_tokens": len(prefix), "generated_tokens": len(action),
               "generation_complete": complete, "generation": evidence,
               "actor_rollout_max_delta": parity["max_abs_delta"],
               "loss": loss.item(), "backward_s": time.perf_counter() - started,
               "actor_peak_cuda_bytes": torch.cuda.max_memory_allocated(),
               "finite_student_gradients": True, "optimizer_updates": 0,
               "external_model_calls": 0, "truncated_training_sequences": 0,
               "scope": "controlled capacity/gradient gate, not native task success or learning"}
    model.zero_grad(set_to_none=True)
    await rollout.aclose()
    (args.output / "context-qualification.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
