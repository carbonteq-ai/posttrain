# GRPO heuristics (local 8 GB)

- Keep both rollout modes explicit. Transformers generation is the single-weight
  fallback for an 8 GB card; optimized profiles use colocated vLLM with sleep.
- Colocated vLLM still owns a separate inference representation. Native-LoRA
  QLoRA uses level-1 sleep: the immutable bitsandbytes base is CPU-backed while
  its GPU allocation and KV cache are released before backprop. Level 2 is only
  valid when full-weight synchronization can reconstruct discarded weights.
  Verifiers must never load a third policy copy.
- For composite checkpoints, keep the vLLM implementation that can load the
  native checkpoint and disable unused towers through its native text-only
  mode. Zero multimodal limits alone can leave composite dummy-input profiling
  in an invalid state.
  Declare any trainer-to-vLLM weight namespace in the rollout profile; do not
  rewrite names in the job or reward bridge.
- Enable native MTP only through a compatible typed rollout profile. Record the
  method, speculative-token count, acceptance, and importance-sampling metrics.
- Do not inherit TRL's importance-sampling defaults implicitly. Long Qwen smoke
  completions produced raw sequence ratios around `1e-4` despite a mean
  per-token log-probability difference below `0.1`; an unbounded lower tail
  nearly erased the policy gradient. The local profile explicitly uses
  sequence-level truncated importance sampling with `[0.1, 3.0]`. Keep the
  theoretically appropriate sequence ratio, but bound measured train/inference
  numerical drift on both sides and record the selected bounds in run inputs.
- On the RTX 3070 Ti, Qwen3.5-2B colocated MTP reaches the native vLLM drafter
  but cannot allocate its additional ~970 MiB embedding beside the training and
  inference representations. Use the non-MTP colocated profile here and retain
  the MTP profile for the larger acceptance machine.
- Keep `num_generations` at 2–4 for smoke; 8 is often too heavy here.
- Size `max_completion_length` from observed termination and version the profile
  when the bound changes. The Qwen3.5 local smoke moved from 256 to 384 after
  observing a 164-token termination and one 256-token clip; its 640-token engine
  window covers the declared 256-token prompt and 384-token completion bounds.
- Combine task correctness with contract-specific format rewards. GSM8K uses a
  final `#### number`; other environments may use `\boxed{}` or structured data.
- SFT warm-start often stabilizes GRPO vs cold instruct checkpoint.
- Reward collapse / all-zeros: check extraction, prompt contract, and that `solution` column is wired.
- The pinned TRL fork validates vLLM 0.25.1. Any compatibility warning means the
  lock or runtime drifted beyond the tested pair and should fail setup rather
  than be ignored.
