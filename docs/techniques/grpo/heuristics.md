# GRPO heuristics (local 8 GB)

- Use the detailed [single-GPU optimization guide](./single-gpu-optimization.md)
  for phase budgeting, benchmark procedure, failure diagnosis, and current
  qualification status.
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
- Keep `num_generations` at 2–4 for a minimal construction smoke. A larger
  algorithm batch, including the current eight-generation benchmark, must use
  physical microbatching and gradient accumulation without changing its
  declared GRPO group.
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

## LoRA for RL

Evidence and sources for these rules are in
[LoRA for RL: settings and evidence](./lora-rl.md).

- Compare LoRA learning rates in alpha × learning rate, not alpha / rank: with
  PEFT's initialisation and Adam, a fresh adapter's first steps scale with
  alpha × LR at any rank. 1e-5 at alpha 32 equals 4e-5 at alpha 8.
- Do not size RL learning rates from SFT evidence. LoRA Without Regret's 15×
  short-run multiplier and large-batch penalty, and Tinker's `get_lr` formula,
  are SFT results; Tinker's own RL recipes use 1e-5 at rank 32.
- Anchor the KL penalty to the base model, including on continuations. With one
  optimizer step per batch the policy ratio is 1 and clipping never fires, so KL
  and the learning rate are the only drift brakes. Under mean-only advantage
  normalisation a published β acts about 1/σ stronger.
- Sample training rollouts with top-p 1.0. A nucleus below 1 adds a constant
  sampler/trainer log-probability gap that biases importance sampling.
- In float16 training, keep log-probabilities, the KL term and the loss in
  float32; a half-precision `exp` in the k3 KL overflows on masked tool tokens.
