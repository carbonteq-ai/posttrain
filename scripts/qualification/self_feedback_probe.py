"""Bounded SELF math/memory and local-model update qualification, not a reward experiment."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, TrainerCallback
from trl import SFTConfig
from trl.experimental.self_feedback import SELFTrainer, chunked_reverse_kl


def benchmark(output: Path) -> dict:
    torch.manual_seed(17)
    student = torch.randn(512, 248320, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn_like(student)
    results = {}
    gradients = {}
    for name, chunk, recompute in (("dense", 512, False), ("chunked", 64, True)):
        times, peaks, loss_values = [], [], []
        for repeat in range(4):
            student.grad = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss = chunked_reverse_kl(student, teacher, chunk, recompute)
            loss.backward()
            loss_values.append(loss.item())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if repeat:
                times.append(elapsed)
                peaks.append(torch.cuda.max_memory_allocated() - baseline)
        results[name] = {
            "mean_forward_backward_s": sum(times) / len(times),
            "peak_extra_bytes": max(peaks),
            "loss": loss_values[-1],
        }
        assert student.grad is not None
        gradients[name] = student.grad.detach().cpu()
    torch.testing.assert_close(gradients["dense"], gradients["chunked"], rtol=0.01, atol=1e-7)
    results["gradient_parity"] = True
    results["shape"] = [512, 248320]
    (output / "loss-benchmark.json").write_text(json.dumps(results, indent=2))
    return results


def distillation_profile(trainer, row, output):
    """Matched full-model shape test, including teacher, both student losses and backward."""
    controlled = copy.deepcopy(row)
    for name in ("action_ids", "feedback_ids"):
        original = controlled[name]
        controlled[name] = (original * (512 // len(original) + 1))[:512]
    # Repeated targets exercise a controlled tensor shape, not on-policy
    # learning data. No optimizer updates or EMA changes occur in this test.
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer.model.enable_input_require_grads()
    trainer.model.train()
    results, gradients = {}, {}
    for name, chunk, recompute in (("dense", 512, False), ("chunked_retained", 64, False), ("chunked", 64, True)):
        trainer.loss_chunk_size, trainer.recompute_loss = chunk, recompute
        timings, peaks, losses = [], [], []
        previous = {}
        for repeat in range(4):
            trainer.model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss = trainer.compute_loss(trainer.model, {"transitions": [controlled]})
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if repeat:
                timings.append(elapsed)
                peaks.append(torch.cuda.max_memory_allocated() - baseline)
                losses.append(loss.item())
            if repeat == 2:
                previous = {
                    n: p.grad.detach().cpu().clone() for n, p in trainer.model.named_parameters() if p.grad is not None
                }
        gradients[name] = {
            n: p.grad.detach().cpu().clone() for n, p in trainer.model.named_parameters() if p.grad is not None
        }
        assert previous
        results[name] = {
            "forward_backward_s": timings,
            "mean_s": sum(timings) / len(timings),
            "peak_extra_bytes": max(peaks),
            "losses": losses,
            "scoring": trainer.last_self_metrics,
            "repeat_gradient_difference": gradient_difference(previous, gradients[name]),
        }
        print(json.dumps({"distillation_profile": name, **results[name]}), flush=True)
    assert gradients["dense"].keys() == gradients["chunked"].keys()
    results["gradient_comparison"] = {
        name: gradient_difference(gradients["dense"], gradients[name]) for name in ("chunked_retained", "chunked")
    }
    # Near-zero elementwise relative errors are unstable after BF16 model
    # backward. Use an explicit global error budget and retain the measured
    # differences; FP32 loss-kernel gradients are separately checked tightly.
    results["bf16_gradient_budget"] = {"max_relative_l2": 0.001, "min_cosine": 0.999999}
    results["gradient_budget_passed"] = all(
        c["relative_l2"] <= 0.001 and c["cosine"] >= 0.999999 for c in results["gradient_comparison"].values()
    )
    results["purpose"] = "controlled 512 action/512 feedback token shape; not on-policy learning"
    (output / "distillation-profile.json").write_text(json.dumps(results, indent=2))
    if not results["gradient_budget_passed"]:
        raise ValueError("BF16 full-model gradient difference exceeds qualification budget")
    trainer.model.zero_grad(set_to_none=True)


def gradient_difference(reference, candidate):
    r2 = sum(g.double().square().sum().item() for g in reference.values())
    c2 = sum(g.double().square().sum().item() for g in candidate.values())
    delta2 = sum((reference[n].double() - candidate[n].double()).square().sum().item() for n in reference)
    dot = sum((reference[n].double() * candidate[n].double()).sum().item() for n in reference)
    return {
        "relative_l2": (delta2 / r2) ** 0.5,
        "cosine": dot / (r2 * c2) ** 0.5,
        "max_absolute": max((reference[n] - candidate[n]).abs().max().item() for n in reference),
    }


def encode(tokenizer, messages):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    return encoded["input_ids"][0].tolist()


def observed_response(action: str) -> str:
    """A real, deterministic toy tool; deliberately separate from AutomationBench."""
    try:
        request = json.loads(action)
    except json.JSONDecodeError:
        return json.dumps({"error": "Expected a JSON object with tool and key fields."})
    if request == {"tool": "lookup_code", "key": "readiness"}:
        return json.dumps({"key": "readiness", "value": "READY-2047"})
    return json.dumps({"error": "Unknown tool or key. Supported: lookup_code, readiness."})


class UpdateAudit(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.pairs = trainer.callback_handler.callbacks[-1].pairs
        self.expected = [t.detach().float().cpu().clone() for _, t in self.pairs]
        self.updates = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        bad = [
            n
            for n, p in self.trainer.model.named_parameters()
            if p.grad is not None and ("lora_" not in n or ".default." not in n)
        ]
        if bad:
            raise AssertionError(f"nonstudent gradients: {bad[:3]}")
        norm = (
            sum(
                p.grad.detach().float().norm().item() ** 2
                for p in self.trainer.model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        if not 0 < norm < float("inf"):
            raise AssertionError(f"invalid student gradient norm {norm}")

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer.accelerator.optimizer_step_was_skipped:
            return
        for i, (student, teacher) in enumerate(self.pairs):
            self.expected[i].lerp_(student.detach().float().cpu(), 0.01)
            torch.testing.assert_close(teacher.detach().float().cpu(), self.expected[i], atol=1e-6, rtol=1e-5)
        row = {"step": state.global_step, **self.trainer.last_self_metrics}
        self.updates.append(row)
        print(json.dumps(row), flush=True)


def model_probe(args, output):
    torch.manual_seed(17)
    if args.deterministic_reference:
        torch.use_deterministic_algorithms(True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    cls = AutoModelForImageTextToText if config.model_type == "qwen3_5" else AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = cls.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="eager" if args.deterministic_reference else "sdpa",
        local_files_only=True,
    )
    model = get_peft_model(
        model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    )
    messages = [
        {
            "role": "user",
            "content": "Look up the readiness code. Reply only with a JSON "
            'object having "tool":"lookup_code" and "key":"readiness".',
        }
    ]
    prefix = encode(tokenizer, messages)
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([prefix], device="cuda")
        generated = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    action = generated[0, len(prefix) :].tolist()
    (output / "raw-generation.json").write_text(
        json.dumps(
            {
                "prefix_ids": prefix,
                "action_ids": action,
                "raw_text": tokenizer.decode(action, skip_special_tokens=False),
                "native_eos_token_id": tokenizer.eos_token_id,
            },
            indent=2,
        )
    )
    if action[-1] != tokenizer.eos_token_id:
        raise ValueError("qualification action hit token limit; do not train truncated action")
    action_text = tokenizer.decode(action, skip_special_tokens=True).strip()
    feedback = observed_response(action_text)
    teacher_messages = messages + [
        {
            "role": "user",
            "content": "Observed environment response to an attempted "
            f"action: {feedback}\nChoose the action for the original request.",
        }
    ]
    feedback_messages = messages + [
        {"role": "assistant", "content": action_text},
        {"role": "user", "content": "Predict the next tool response as JSON."},
    ]
    row = {
        "student_prefix_ids": prefix,
        "teacher_prefix_ids": encode(tokenizer, teacher_messages),
        "feedback_prefix_ids": encode(tokenizer, feedback_messages),
        "action_ids": action,
        "feedback_ids": tokenizer.encode(feedback, add_special_tokens=False),
        "trajectory_id": "local-readiness-probe",
        "policy_revision": "round0-before-update",
    }
    (output / "transition.json").write_text(
        json.dumps({"row": row, "action": action_text, "feedback": feedback}, indent=2)
    )
    training_args = SFTConfig(
        output_dir=str(output / "checkpoint"),
        max_steps=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        lr_scheduler_type="constant",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        seed=17,
        remove_unused_columns=False,
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_pin_memory=False,
    )
    trainer = SELFTrainer(
        profile_scoring=True,
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list([row]),
        data_collator=lambda rows: {"transitions": rows},
    )
    audit = UpdateAudit(trainer)
    trainer.add_callback(audit)
    if args.profile_distillation:
        distillation_profile(trainer, row, output)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started
    trainer.save_model(str(output / "student-and-teacher"))
    assert len(audit.updates) == 3
    receipt = {
        "purpose": "synthetic transition GPU qualification; not AutomationBench reward training",
        "model_path": args.model,
        "action": action_text,
        "feedback": feedback,
        "applied_updates": len(audit.updates),
        "updates": audit.updates,
        "wall_s": elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "train_metrics": result.metrics,
        "ema_verified": True,
        "nonstudent_gradients": False,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "trl", "transformers", "peft", "accelerate")},
        "selected_trl_source": "290c03d8e1553db509b0d033907f60fc1cb47b82 plus local SELF candidate",
        "qualification_dropout": 0.0,
        "external_model_calls": 0,
    }
    (output / "gpu-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profile-distillation", action="store_true")
    parser.add_argument("--deterministic-reference", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.benchmark:
        print(json.dumps(benchmark(args.output)), flush=True)
    if args.model:
        gc.collect()
        torch.cuda.empty_cache()
        model_probe(args, args.output)


if __name__ == "__main__":
    main()
