"""Summarize the offline sampler x trainer log-probability mismatch matrix.

usage: analyze.py <out-dir>   (runs on the host; needs numpy only)

Reads samples_<tag>.json (gen_vllm.py, or score_hf.py --generate), scores_<dtype>*.json
(score_hf.py), vllm_hooks_<tag>.json and failed_<tag>.json from <out-dir>, prints the
report and writes <out-dir>/report.json. Each matrix cell is the mean / 99th percentile /
maximum absolute per-token difference between the sampler's and the trainer's
log-probability of each sampled token, and the mean absolute per-sequence sum of those
differences (the sequence-level log-ratio). Trainer log-probabilities are reported as

* native_logps: log-softmax in the logits dtype (TRL today for bf16/fp16 logits)
* fp32_logps: model-dtype logits upcast to float32 first (backend_options.logits_float32)
* head32_logps: an fp32 LM head (float32 products and logits)
* nucleus_logps: fp32_logps renormalized over the sampler's top-k/top-p nucleus, the
  distribution vLLM's processed log-probabilities describe; the gap between fp32_logps
  and nucleus_logps rows is the top-p offset, not precision.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

KEYS = ("native_logps", "fp32_logps", "head32_logps", "nucleus_logps")
FP16_MAX = 65504.0


def _cell(records: list[dict[str, Any]], rows: dict[str, dict[str, Any]], key: str) -> dict[str, float]:
    tokens, sequences = [], []
    for record in records:
        delta = np.array(record["sampler_logps"], dtype=np.float64) - np.array(rows[record["id"]][key])
        tokens.append(np.abs(delta))
        sequences.append(delta.sum())
    flat = np.concatenate(tokens) if tokens else np.zeros(0)
    finite = flat[np.isfinite(flat)]
    return {
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "p99": float(np.quantile(finite, 0.99)) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "sequence_abs_mean": float(np.mean(np.abs(sequences))) if sequences else float("nan"),
        "sequence_mean": float(np.mean(sequences)) if sequences else float("nan"),
        "nonfinite": int(flat.size - finite.size),
        "tokens": int(flat.size),
    }


def _hooks(hooks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bad = {name: value for name, value in hooks.items() if value["inf"] or value["nan"]}
    top = sorted(hooks.items(), key=lambda item: -item[1]["max"])[:5]
    return {
        "outputs": len(hooks),
        "nonfinite_outputs": {
            name: {k: value[k] for k in ("inf", "nan") if k in value}
            | ({"first_bad_pos": value["first_bad_pos"]} if value.get("first_bad_pos") else {})
            for name, value in bad.items()
        },
        "largest": [[name, value["max"]] for name, value in top],
        "fp16_headroom": FP16_MAX / top[0][1]["max"] if top and top[0][1]["max"] > 0 else None,
    }


def main(out: Path) -> None:
    samples = {
        Path(path).stem.removeprefix("samples_"): json.loads(Path(path).read_text())
        for path in sorted(glob.glob(str(out / "samples_*.json")))
    }
    scores: dict[str, dict[str, list[dict[str, Any]]]] = {}
    hooks: dict[str, Any] = {}
    for path in sorted(glob.glob(str(out / "scores_*.json"))):
        payload = json.loads(Path(path).read_text())
        for sampler, rows in payload["scores"].items():
            scores.setdefault(payload["dtype"], {})[sampler] = rows
        hooks[f"trainer {Path(path).stem}"] = _hooks(payload["hooks"])
    for path in sorted(glob.glob(str(out / "vllm_hooks_*.json"))):
        payload = json.loads(Path(path).read_text())
        for phase in ("decode", "prefill"):
            if payload.get(phase):
                hooks[f"vllm {Path(path).stem} {phase}"] = _hooks(payload[phase])
    report: dict[str, Any] = {
        "samplers": {tag: payload["summary"] for tag, payload in samples.items()},
        "failed": {
            Path(path).stem.removeprefix("failed_"): json.loads(Path(path).read_text())["error"]
            for path in sorted(glob.glob(str(out / "failed_*.json")))
        },
        "matrix": {},
        "hooks": hooks,
    }
    for key in KEYS:
        print(f"\ntrainer log-probs: {key} (mean / p99 / max per token; mean |sequence sum|)")
        for sampler, payload in samples.items():
            cells = []
            for trainer, by_sampler in scores.items():
                rows = {row["id"]: row for row in by_sampler.get(sampler, [])}
                if not rows or key not in next(iter(rows.values())):
                    continue
                cell = _cell(payload["records"], rows, key)
                report["matrix"].setdefault(key, {}).setdefault(sampler, {})[trainer] = cell
                cells.append(
                    f"{trainer}: {cell['mean']:.4f}/{cell['p99']:.3f}/{cell['max']:.2f} "
                    f"seq {cell['sequence_abs_mean']:.2f}"
                    + (f" nonfinite {cell['nonfinite']}" if cell["nonfinite"] else "")
                )
            if cells:
                print(f"  sampler {sampler:16s} " + " | ".join(cells))
    print("\nsampler throughput (all prompts in one generate call)")
    for tag, summary in report["samplers"].items():
        rate = summary.get("tokens_per_second")
        print(
            f"  {tag:16s} {summary.get('sampler', 'vllm'):12s} {summary.get('generated_tokens', 0):>8} tokens "
            f"{summary.get('seconds', 0):8.1f} s" + (f" {rate:9.1f} tok/s" if rate else "")
        )
    for tag, error in report["failed"].items():
        print(f"  {tag:16s} FAILED: {error[:160]}")
    print("\nactivations (largest finite |module output|; fp16 max 65504)")
    for name, summary in hooks.items():
        largest = ", ".join(f"{module} {value:.1f}" for module, value in summary["largest"][:3])
        print(f"  {name}: {summary['outputs']} outputs, {len(summary['nonfinite_outputs'])} with inf/NaN; {largest}")
        for module, detail in list(summary["nonfinite_outputs"].items())[:5]:
            print(f"    NON-FINITE {module}: {detail}")
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
