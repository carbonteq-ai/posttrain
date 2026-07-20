# GRPO heuristics (local 8 GB)

- Keep both rollout modes explicit. Transformers generation is the single-weight
  fallback for an 8 GB card; optimized profiles use colocated vLLM with sleep.
- Colocated vLLM still owns a separate inference representation. Sleep level 2
  discards its weights and KV cache before backprop; Verifiers must never load a
  third policy copy.
- For composite checkpoints, keep the vLLM implementation that can load the
  native checkpoint and disable unused towers with zero multimodal limits.
  Declare any trainer-to-vLLM weight namespace in the rollout profile; do not
  rewrite names in the job or reward bridge.
- Enable native MTP only through a compatible typed rollout profile. Record the
  method, speculative-token count, acceptance, and importance-sampling metrics.
- On the RTX 3070 Ti, Qwen3.5-2B colocated MTP reaches the native vLLM drafter
  but cannot allocate its additional ~970 MiB embedding beside the training and
  inference representations. Use the non-MTP colocated profile here and retain
  the MTP profile for the larger acceptance machine.
- Keep `num_generations` at 2–4 for smoke; 8 is often too heavy here.
- Size `max_completion_length` from observed termination. A clipped rollout is
  not a useful memory optimization; reduce the group or use more memory instead.
- Combine **format** + **exact-match** rewards so the model learns `\boxed{}` before accuracy.
- SFT warm-start often stabilizes GRPO vs cold instruct checkpoint.
- Reward collapse / all-zeros: check extraction, prompt contract, and that `solution` column is wired.
- The pinned TRL fork validates vLLM 0.25.1. Any compatibility warning means the
  lock or runtime drifted beyond the tested pair and should fail setup rather
  than be ignored.
