"""Summarize the offline sampler x trainer log-probability mismatch matrix.

Reads samples_<sampler>.json (gen_vllm.py / score_hf.py --generate) and
scores_<trainer>*.json (score_hf.py) from the output directory given as the
first argument, and prints, for every sampler row and trainer column, the mean /
99th percentile / maximum absolute per-token difference between the sampler's
and the trainer's log-probability of each sampled token, and the mean absolute
per-sequence sum of those differences (the sequence-level log-ratio).
Trainer log-probabilities are reported three ways:

* native_logps: log-softmax in the logits dtype (TRL today for bf16/fp16 logits)
* fp32_logps: model-dtype logits upcast to float32 before the log-softmax
  (backend_options.logits_float32)
* head32_logps: an fp32 LM head (model-dtype hidden state and weight, float32
  products and logits), for comparison with vLLM's hf_overrides head_dtype
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

KEYS = ("native_logps", "fp32_logps", "head32_logps")


def main(out: Path) -> None:
    samples = {
        Path(path).stem.removeprefix("samples_"): json.loads(Path(path).read_text())["records"]
        for path in sorted(glob.glob(str(out / "samples_*.json")))
    }
    scores: dict[str, dict[str, list[dict]]] = {}
    for path in sorted(glob.glob(str(out / "scores_*.json"))):
        payload = json.loads(Path(path).read_text())
        for sampler, rows in payload["scores"].items():
            scores.setdefault(payload["dtype"], {})[sampler] = rows
    for key in KEYS:
        print(f"\ntrainer log-probs: {key} (mean / p99 / max per token; mean |sequence sum|)")
        for sampler, records in samples.items():
            cells = []
            for trainer, by_sampler in scores.items():
                rows = {row["id"]: row for row in by_sampler.get(sampler, [])}
                if not rows or key not in next(iter(rows.values())):
                    continue
                tokens, sequences = [], []
                for record in records:
                    delta = np.array(record["sampler_logps"]) - np.array(rows[record["id"]][key])
                    tokens.append(np.abs(delta))
                    sequences.append(delta.sum())
                flat = np.concatenate(tokens)
                cells.append(
                    f"{trainer}: {flat.mean():.4f}/{np.quantile(flat, 0.99):.3f}/{flat.max():.2f} "
                    f"seq {np.mean(np.abs(sequences)):.2f} nonfinite {int((~np.isfinite(flat)).sum())}"
                )
            if cells:
                print(f"  sampler {sampler:14s} " + " | ".join(cells))
    for trainer_path in sorted(glob.glob(str(out / "scores_*.json"))):
        payload = json.loads(Path(trainer_path).read_text())
        hooks = payload["hooks"]
        bad = {name: value for name, value in hooks.items() if value["inf"] or value["nan"]}
        top = sorted(hooks.items(), key=lambda item: -item[1]["max"])[:3]
        print(
            f"\nHF {Path(trainer_path).stem}: {len(hooks)} hooked outputs, {len(bad)} with inf/NaN, "
            f"largest |activation| " + ", ".join(f"{name} {value['max']:.1f}" for name, value in top)
        )
    for hooks_path in sorted(glob.glob(str(out / "vllm_hooks_*.json"))):
        payload = json.loads(Path(hooks_path).read_text())
        for phase in ("decode", "prefill"):
            hooks = payload[phase]
            bad = {name: value for name, value in hooks.items() if value["inf"] or value["nan"]}
            top = sorted(hooks.items(), key=lambda item: -item[1]["max"])[:3]
            print(
                f"vLLM {Path(hooks_path).stem} {phase}: {len(hooks)} hooked outputs, {len(bad)} with inf/NaN, "
                "largest |activation| " + ", ".join(f"{name} {value['max']:.1f}" for name, value in top)
            )


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
