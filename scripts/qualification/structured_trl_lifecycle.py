"""Installed-fork CUDA optimizer/resume/export gate using deterministic reward fixtures.

This is not a live Verifiers, judge, vLLM, LoRA, or research-quality qualification.
Run with the upgraded TRL candidate environment and this workspace's train source.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from posttrain.train.reward_advantages import compute_capo_advantages, compute_gdpo_advantages
from posttrain.train.reward_evidence import ProcessCredit, RewardEvidence, RewardValue
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
REVISION = "4b10ebee6e13a2669155516652960f50984399fd"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("gdpo", "capo"), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if version("trl") != "1.12.0.post2" or not torch.cuda.is_available():
        raise RuntimeError("Gate requires installed TRL 1.12.0.post2 and CUDA")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list([{"prompt": "What is 1+1?"}] * 8)

    def rollout(prompts, trainer, inputs=None):
        assert len(prompts) == 2
        masks = [(True, False, True)] * 2
        evidence = [
            RewardEvidence(
                "g",
                f"r{i}",
                f"trace{i}",
                "0",
                "fixture@1",
                (RewardValue("outcome", "valid", float(i)),),
                ProcessCredit("valid", "steps@1", f"critique{i}", ((0, 1),)),
            )
            for i in range(2)
        ]
        if args.algorithm == "gdpo":
            advantages = compute_gdpo_advantages(
                evidence, masks, component_names=("outcome",), component_weights=(1.0,), group_size=2
            )
        else:
            advantages = compute_capo_advantages(evidence, masks, group_size=2)
        return {
            "prompt_ids": [tokenizer.encode(prompt) for prompt in prompts],
            "completion_ids": [[10 + i, 20, 30 + i] for i in range(2)],
            "logprobs": [[-1.0, 0.0, -1.0]] * 2,
            "env_mask": [list(mask) for mask in masks],
            "precomputed_advantages": [list(row) for row in advantages.token_advantages],
        }

    def reward(completions, **kwargs):
        return [float(i) for i in range(len(completions))]

    def make_trainer():
        return GRPOTrainer(
            model=AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32),
            processing_class=tokenizer,
            train_dataset=dataset,
            rollout_func=rollout,
            reward_funcs=reward,
            args=GRPOConfig(
                output_dir=str(output / "train"),
                max_steps=2,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                num_generations=2,
                max_completion_length=8,
                learning_rate=1e-4,
                save_strategy="steps",
                save_steps=1,
                logging_steps=1,
                report_to="none",
                disable_tqdm=True,
                bf16=False,
                use_vllm=False,
                use_liger_kernel=False,
                use_precomputed_advantages=True,
                loss_type="grpo",
                beta=0.1,
                use_bias_correction_kl=False,
                seed=42,
            ),
        )

    trainer = make_trainer()
    result = trainer.train()
    norms = [float(row["grad_norm"]) for row in trainer.state.log_history if "grad_norm" in row]
    assert math.isfinite(result.training_loss) and trainer.state.global_step == 2
    assert len(norms) == 2 and all(math.isfinite(value) and value > 0 for value in norms)
    expected = {name: value.detach().cpu().clone() for name, value in trainer.model.state_dict().items()}
    checkpoint = output / "train" / "checkpoint-1"
    assert all(
        (checkpoint / name).is_file()
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    )
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    resumed = make_trainer()
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 2
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), expected[name], rtol=1e-5, atol=1e-6, msg=name)
    export = output / "export"
    resumed.save_model(str(export))
    del resumed
    gc.collect()
    torch.cuda.empty_cache()
    restored = AutoModelForCausalLM.from_pretrained(export).to("cuda").eval()
    inputs = tokenizer("What is 1+1?", return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated = restored.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert generated.shape[1] > inputs.input_ids.shape[1]
    receipt = {
        "schema": "posttrain.structured-trl-lifecycle.v1",
        "algorithm": args.algorithm,
        "trl": version("trl"),
        "torch": torch.__version__,
        "model": MODEL,
        "revision": REVISION,
        "optimizer_updates": 2,
        "beta": 0.1,
        "gradient_norms": norms,
        "checkpoint_resume_matches_uninterrupted": True,
        "export_generated_tokens": generated.shape[1] - inputs.input_ids.shape[1],
        "scope": "tiny CUDA full-parameter deterministic reward fixture; excludes live bridge/judge/vLLM/LoRA",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
