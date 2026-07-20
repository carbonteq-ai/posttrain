# Inference implementation heuristics

These are evidence-backed working rules for the inference engine. They are
expected to change when a newer runtime proves a better rule.

## CUDA and optimized kernels

- Treat PyTorch CUDA, the vLLM wheel, NVCC, NVVM, CRT, CCCL, and CUDA headers as
  one versioned toolchain.
- Validate the compiler minor version against `torch.version.cuda` before
  importing vLLM or starting a worker.
- Adapt pip CUDA's filesystem layout through a generated toolkit view; do not
  patch installed wheels or silently fall back from FlashInfer.
- A failed setup or artifact-upload run is not benchmark evidence, even if it
  emitted partial metrics. Preserve it for diagnosis and rerun cleanly.

## Measurements

- Report cold start separately from steady-state inference. Cold start includes
  weight loading, JIT compilation, and CUDA graph capture.
- Warm every request shape used by the measured workload. A JIT warning during
  measurement means the warmup is incomplete.
- Always record prompt count, concurrency, input/output tokens, sampling
  settings, model revision, engine profile, GPU, and package versions.
- Compare model-weight memory, CUDA graph memory, and KV-cache capacity as
  separate quantities.
- Keep configured context length separate from occupied input length. Include at
  least one genuinely input-heavy cell before claiming long-context performance.
- Use exact token-ID prompts and forced output lengths for controlled systems
  comparisons. Use canonical messages plus the tokenizer-native chat template
  for representative workloads; never merge these cohorts.
- Treat concurrency as an execution filter over a reusable suite. This machine
  stops at 4; the suite retains concurrency 8 for larger hardware.

## Configuration ownership

- Put immutable model identity and the default compatible serve profile in
  `profiles/models`.
- Put backend knobs and model-specific runtime compatibility in
  typed profiles shipped with `packages/serve`.
- Put reusable execution and measurement behavior in `packages/serve`.
- Put TurboQuant, MTP, and custom-kernel options in serve profiles only after
  runtime support is validated on the target model; a declared option is not
  evidence that the combination works.
- Resolve the TurboQuant K8V4 serve variant for every 32K suite cell on this
  hardware. Record the resolved variant on the run rather than inferring it from
  the context length later.

## Compatibility fixes

- Prefer a backend's supported plugin lifecycle for process-wide compatibility
  fixes; vLLM workers use `spawn`, so a parent-only monkey patch is incomplete.
- Keep each compatibility fix narrow and state-guarded, test the invariant it
  repairs, and make it a no-op once upstream behavior is correct.
- Never edit an installed wheel or silently disable the requested optimized
  path to make a benchmark pass.
