"""Bounded installed-TRL CUDA checkpoint gate; not a vLLM or model-quality gate.

Run with the candidate runtime Python and an empty --output directory. Uses an
immutable tiny model fixture, a local teacher, and native on-policy generation.
No tracking service or training dataset download is required.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl.experimental.iw_opd import IWOPDConfig, IWOPDTrainer

MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
REVISION = "4b10ebee6e13a2669155516652960f50984399fd"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("This lifecycle gate requires a CUDA GPU")
    if version("trl") != "1.12.0.post2":
        raise RuntimeError("Expected retained TRL 1.12.0.post2")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}
            for prompt, answer in [("What is 1+1?", "2"), ("What is 2+2?", "4")] * 4
        ]
    )

    def make_trainer() -> IWOPDTrainer:
        student = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
        teacher = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
        # A distinct frozen teacher prevents a vacuous identical-policy zero gradient.
        with torch.no_grad():
            teacher.get_output_embeddings().weight.mul_(1.1)
        return IWOPDTrainer(
            model=student,
            teacher_model=teacher,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=IWOPDConfig(
                output_dir=str(output / "train"),
                per_device_train_batch_size=2,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_steps=2,
                learning_rate=1e-4,
                save_strategy="steps",
                save_steps=1,
                logging_steps=1,
                report_to="none",
                disable_tqdm=True,
                use_cpu=False,
                bf16=False,
                lmbda=1.0,
                distillation_objective="iw_opd",
                max_length=64,
                max_completion_length=8,
                use_vllm=False,
                seed=42,
            ),
        )

    trainer = make_trainer()
    result = trainer.train()
    if not math.isfinite(result.training_loss) or trainer.state.global_step != 2:
        raise RuntimeError("Native CUDA training did not complete two finite updates")
    gradients = [float(record["grad_norm"]) for record in trainer.state.log_history if "grad_norm" in record]
    if len(gradients) != 2 or not all(math.isfinite(value) and value > 0 for value in gradients):
        raise RuntimeError(f"Expected two finite nonzero gradients, got {gradients}")
    expected = {name: value.detach().cpu().clone() for name, value in trainer.model.state_dict().items()}
    checkpoint = output / "train" / "checkpoint-1"
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "model.safetensors"):
        if not (checkpoint / name).is_file():
            raise RuntimeError(f"Incomplete recovery checkpoint: {name}")
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    resumed = make_trainer()
    resumed.train(resume_from_checkpoint=str(checkpoint))
    if resumed.state.global_step != 2:
        raise RuntimeError("Checkpoint resume did not reach the second update")
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
    if generated.shape[1] <= inputs.input_ids.shape[1]:
        raise RuntimeError("Exported model produced no new tokens")
    receipt = {
        "schema": "posttrain.trl-retained-fork-lifecycle.v1",
        "trl": version("trl"),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "device": torch.cuda.get_device_name(),
        "model": MODEL,
        "model_revision": REVISION,
        "optimizer_updates": 2,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_norms": gradients,
        "checkpoint_resume_matches_uninterrupted": True,
        "export_generated_tokens": generated.shape[1] - inputs.input_ids.shape[1],
        "scope": "tiny-fixture native CUDA IW-OPD; excludes vLLM, LoRA, veRL and quality qualification",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
