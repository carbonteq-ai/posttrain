# GRPO heuristics (local 8 GB)

- Default: **no vLLM** (`use_vllm=False`); Transformers generate is safer on one 8 GB card.
- Keep `num_generations` at 2–4 for smoke; 8 is often too heavy here.
- Cap `max_completion_length` (e.g. 256); long CoT blows VRAM and wall time.
- Combine **format** + **exact-match** rewards so the model learns `\boxed{}` before accuracy.
- SFT warm-start often stabilizes GRPO vs cold instruct checkpoint.
- Reward collapse / all-zeros: check extraction, prompt contract, and that `solution` column is wired.
- The pinned TRL fork validates vLLM 0.25.1. Any compatibility warning means the
  lock or runtime drifted beyond the tested pair and should fail setup rather
  than be ignored.
