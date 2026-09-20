# vLLM inference optimization on Blackwell

> **Technical delivery architecture, not a product-contract amendment.**
> The frozen product baseline remains
> [docs/post-training/](../post-training/README.md). It already assigns runtime
> kernel, scheduler, speculative-decoding, and cache choices to an inference
> binding. This document records the implementation and qualification evidence
> for those choices.

Status: experimental candidate, not released  
Last revised: 2026-09-19

## Decision in one sentence

Retain the native K2/Uno topology as **full target decode CUDA graphs + eager
Uno proposer + extracted SM120 FA4 + the complete eight-token Uno candidate
block**. The current strict, shape-tuned checkpoint is **445.36 aggregate output
tokens/s** at c4 and remains exact at c8/c16/c32. The earlier 526.41 tok/s fast
result is still a performance lead only because its outputs were not batch
invariant.

## What the current fast path actually does

```text
four HTTP requests
  -> vLLM scheduler and continuous batching
  -> Uno prepares [clean root, draft_1, ..., draft_7]
       -> deterministic prompt-stable noise
       -> position-gated Uno system LoRA
       -> eager proposer forward
  -> target verifies all eight candidates in one pass
       -> SM120_FA4 paged attention
       -> full decode CUDA graph where eligible
  -> rejection sampler commits the accepted prefix plus target correction
  -> metrics expose eight candidate positions
```

`--compilation-config '{"mode":0}'` means that vLLM's Inductor compilation is
disabled. It does **not** mean all CUDA graphs are disabled. The retained server
captured 62 full decode graphs for the target. It did not capture the 88
PIECEWISE segments present in Stage 1. The server still reports ten proposer
capture descriptors during memory planning, but the Stage 6b log contains no
live Uno graph qualification or replay; without the breakable wrapper the Uno
proposer remained eager.

This distinction matters because “compiled,” “CUDA graphed,” and “fast” are not
synonyms. Stage 1 used `CompilationMode.NONE` too, but retained
`FULL_AND_PIECEWISE` breakable execution. Stage 6b normalized that combination
to `FULL_DECODE_ONLY` and removed the segmented execution overhead.

## Correct Uno token accounting

For configured block width `B = 8`, Uno produces:

```text
candidate block = [one clean target-policy root] + [seven future Uno drafts]
```

All eight tokens must remain candidates to a one-pass autoregressive target.
The target must consume the root before its next-position logits can verify the
first future draft. If the root is committed separately and removed from the
verification block, every later candidate is checked against the wrong
conditional position.

The maximum scheduler output from one successful cycle is nine tokens: eight
accepted candidates plus one target correction/lookahead token. Metrics count
eight candidate positions; “seven future drafts” describes the algorithm, not
the scheduler reservation.

The experiment that removed the clean root from the candidate block confirmed
the failure mode: acceptance collapsed to **3.29%** and throughput to **196.21
tokens/s**. That design is rejected. The old name “authoritative prefix” is
therefore misleading for the retained one-pass path; the root is
target-distributed but remains candidate position zero.

## Optimization inventory

### Measurement and correctness

- Added warm-only benchmark boundaries so model loading, kernel JIT, graph
  capture, and cache construction are excluded from timed intervals.
- Fixed prompt population, output budget, sampling, concurrency, prefix-cache
  policy, model revisions, and response hashes in retained artifacts.
- Added draft totals, accepted totals, per-position acceptance, KV occupancy,
  preemptions, and resolved server configuration to the evidence path.
- Added eager-versus-replay qualification over hidden states, logits, selected
  tokens, and selected-token logprobs, including changed-input replay and batch
  churn.
- Added token-level repeat and batch-position diagnostics. They exonerated Uno
  sampling and localized the first divergence to target attention after
  batch-shaped prefill reductions had changed historical KV state.
- Fixed SM120 prefill to the qualified M64N64 reduction tile in deterministic
  mode. A fixed-Q/K/V kernel oracle is bit-exact at c1/c2/c4/c8; automatic
  tiling changed 191,314 BF16 elements at c2-c8.
- Established a strict full-model checkpoint using the fixed tile plus vLLM
  batch-invariant linears: c8 8/8, c16 16/16, and c32 32/32 outputs matched.
  Its initial c4 mean was 270.45 tok/s, which localized the remaining cost to
  broad invariant-linear execution rather than attention correctness.
- Added a dedicated SM120 invariant-matmul config family and production-layout
  sweeps for K2's qkv, output, gate/up, down, proposer, and FP32 vocabulary
  shapes. Decode/proposal buckets use measured small-M tiles while prefill keeps
  the proven 128x128 tile. This raised strict c4 throughput from 270.45 to
  445.36 tok/s (+64.67%) while retaining exact c8/c16/c32 repeats.

### Blackwell attention

- Audited current vLLM support before carrying old patches forward. Native
  `b12x` improved K2 by 1.47% at 1K outputs and 2.65% at 16K outputs versus
  FA2, but it is not the best K2/Uno operating point.
- Extracted the SGLang SM120 paged-KV and SplitKV closure into the independent
  `sm120-paged-attention` package rather than importing the SGLang runtime.
- Added an opt-in vLLM `SM120_FA4` adapter while retaining vLLM ownership of
  page tables, KV writes, scheduling, output buffers, eligibility, and fallback.
- Preserved Uno's uniform `(request, block, head, dim)` shape through the
  adapter instead of flattening every speculative batch through varlen APIs.
- Kept Gemma 4 on Triton. Both the head-dimension-512 SM120 extension and the
  mixed SM120/Triton path corrupted full-model output despite a passing narrow
  kernel probe.

### Proposer execution and LoRA ownership

- Introduced a generic proposer execution context rather than a K2-only second
  scheduler.
- Separated target and proposer ownership, reserved the Uno system-adapter slot,
  and added owner-scoped Punica routing banks.
- Added proposer-owned persistent token, position, sequence, block-table,
  slot-mapping, noise-identity, and LoRA-ID buffers.
- Keyed graph descriptors by batch shape, draft width, dtype, and LoRA
  structure, with descriptor-local eager fallback.
- Fixed the qualification oracle to ignore dispatcher-padding rows, use real KV
  state, preserve touched KV blocks, and compare initial and changed-input
  replay.
- Preserved policy-LoRA composition as a required feature, but it remains a
  release gate: a real optimizer-step refresh, fresh logprobs, and policy-version
  fencing have not yet been qualified on the 526-token/s topology.

### Scheduling and hot-path work

- Used prompt-stable deterministic Uno noise to remove batch-slot identity from
  proposals.
- Removed the invalid clean-root separation and restored the complete
  eight-candidate verification block.
- Removed proposal-loop transient metadata work and avoidable host/device
  synchronization in Stage 1. The two-run gain was only 0.63%, below observed
  run variance, so this change is retained for structure but is not credited as
  a measured speedup.
- Tested asynchronous scheduling separately. It did not explain the regression;
  synchronous Stage 5 averaged about 357.52 tokens/s, essentially the same
  region as the asynchronous/compiled candidates.
- Restored `CompilationMode.NONE` explicitly and allowed vLLM to normalize
  target execution to `FULL_DECODE_ONLY`. This removed the 88-piece breakable
  execution plan while retaining target decode graphs.

### Framework and release safety

- Unified serving and colocated-TRL translation of attention, cache,
  prefix-cache, scheduler, and speculative settings.
- Moved deterministic flag-combination policy into Posttrain job compilation so
  known-invalid jobs fail before packing or provider submission.
- Kept dynamic hardware and runtime checks in vLLM as a backstop.
- Defined Gemma paired-assistant MTP as the first non-Uno regression consumer of
  the generic proposer/composer seam. Its retained MTP-2 result is 253.93
  tokens/s versus 181.43 without MTP, with 98.30% acceptance.

## Matched concurrency-four stage history

All rows below used four unique roughly 5K-token prompts, 1,024 output tokens
per request, 4,096 total outputs, greedy target sampling, prompt-stable Uno
noise, SM120 FA4, FP32 output head, block width eight, and no prefix caching.

| Stage | Execution change | Mean aggregate output tok/s | Mean acceptance | Verdict |
| --- | --- | ---: | ---: | --- |
| 0 | Pre-hot-path baseline | 419.43 | 48.89% | Reference |
| 1 | Fewer host synchronizations and transient allocations | 422.08 | 48.45% | +0.63%; within variance |
| 2 | Incorrectly remove clean root from candidate block | 196.21 | 3.29% | Reject: wrong autoregressive alignment |
| 3b | Restore full eight-candidate block under compiled/async topology | 345.80 | 50.06% | Correcter math, slow topology |
| 4 | Add experimental SM120 attention graph path | 357.53 | 47.68% | No decisive gain |
| 5 | Disable async scheduling | 357.52 | 50.46% | Async is not the main cause |
| 6b | Full target decode graphs, eager proposer, no PIECEWISE plan | **526.41** | **54.64%** | Unqualified performance lead |

Stage 6b generated the same 4,096 outputs with 9.31% fewer draft candidates
than Stage 1. Its two throughputs were 528.12 and 524.70 tokens/s, only 0.65%
apart, and all four response hashes matched between repeats. Two mechanisms are
therefore observed together:

1. lower execution overhead after removing PIECEWISE graph fragmentation; and
2. fewer speculative cycles because acceptance increased.

The current artifacts do not independently attribute the 24.72% gain between
those mechanisms. A factorial A/B is required before assigning a percentage to
either cause.

## Where to go next

### 1. Optimize from the shape-tuned strict checkpoint

The numerical divergence is localized. Automatic SM120 prefill tiling changed
historical KV state, and ordinary model linears remained execution-shape
dependent after that kernel fix. The strict configuration fixes prefill at
M64N64 and uses batch-invariant linears. A dedicated SM120 config family now
matches the production transposed-weight layout and separates small decode
buckets from large prefill. It averages 445.36 tok/s at c4 and matches c8 8/8,
c16 16/16, and c32 32/32 responses.

Use 445.36 tok/s as the oracle-backed baseline and 526.41 tok/s as the remaining
performance lead. The FP32 250,624-way vocabulary projection is now the largest
measured invariant GEMM; optimizing or fusing that path is higher leverage than
weakening invariant routing.

### 2. Attribute the gain instead of guessing

Run a same-revision factorial matrix with fixed proposal noise and token outputs:

| Cell | Target execution | Proposer execution | Purpose |
| --- | --- | --- | --- |
| A | full decode graphs | eager | Stage 6b control |
| B | full + PIECEWISE | independent graph | quantify fragmentation cost |
| C | full decode graphs | dedicated whole-proposer graph | test graph benefit without segmentation |
| D | eager target | eager proposer | bound total graph value |

Record scheduler steps, target forwards, proposer forwards, accepted tokens per
cycle, CUDA kernel time, CPU launch gaps, and bytes copied host-to-device. This
separates cycle efficiency from time per cycle.

### 3. Build the proposer graph at the right boundary

Do not revive 88 breakable model segments. Capture one dedicated proposer graph
per structural descriptor, with SM120 attention included only after its launch
plan, workspace, and auxiliary metadata have stable graph-owned addresses. An
uncaptured shape falls back to eager. Promotion requires eager/replay equality
under request-count churn and changed inputs.

### 4. Remove remaining CPU/GPU bubbles

Profile Stage 6b before changing it. The leading candidates are request-to-slot
mapping construction, LoRA metadata staging, speculative acceptance readback,
Python scheduler boundaries, and unfused KV append/attention. Move only an
observed bottleneck to GPU, one change per matched A/B. Do not optimize the
already-saturated c16/c32 case at the expense of the RL-relevant c4 case.

For stochastic RL, remove dense probability traffic through a logits-native
contract rather than a local tensor micro-optimization. At c32, width seven,
and vocabulary near 250K, one FP32 `[32, 7, V]` tensor is about 214 MiB; the
legacy path materializes probabilities, exponential noise, and a probability
clone. The fork already contains the newer upstream logits-native rejection
kernels, but the active legacy runner does not call them.

Port that contract across the proposer and active runner as one correctness
slice: retain draft logits, sample with deterministic per-request RNG, verify
and draw the residual directly from logits, and preserve greedy/random mixed
batches, target processors, penalties, bad words, grammar masks, bonus tokens,
raw/processed logprobs, synthetic verification, and heterogeneous vocabulary
handling. First compare emitted-token distributions and exact greedy outputs
against the legacy sampler; only then profile allocations and throughput.
The first co-resident RTX PRO run passed 45 kernel tests. Eight large
statistical block-verification fixtures were not exercised because each needed
3.1-7.5 GiB while the strict K2 server held 87.34 GiB of the 94.97 GiB GPU.
They remain an explicit no-server qualification gate.

### 5. Qualify RL LoRA behavior

Run one 20-step AutomationBench/Vortex experiment with the retained topology.
At an optimizer step, update policy LoRA tensors in place, advance a policy
generation fence, invalidate stale request state, and prove that rollout
logprobs come from the new policy. Test abort, drain, sleep/wake, and batch churn.
Policy LoRA plus Uno must remain functional even if its first qualified mode is
eager.

### 6. Prove the composer is generic

Route Gemma MTP through the shared lifecycle and observability seam without
rewriting its specialized Q-only assistant, target-KV sharing, or fused decode.
The acceptance gate is no regression from 253.93 tokens/s and exact target
output equivalence. Only then add dynamic draft-width policy and unified
proposer metrics.

### 7. Release only after evidence and fork ownership are complete

Remove dead authoritative-prefix helpers and misleading names, commit the vLLM
fork with focused tests, update `CARBONTEQ_FORK.md` and the Posttrain consumer
ledger, publish the fork, then advance immutable dependency pins and lockfiles.
The released binding should select a qualified topology; it must not infer one
from a model name or silently enable an unqualified graph mode.

## Evidence and reproduction

- Narrative and benchmark index:
  `/home/hammad/projects/k2-horizon-inference/results/README.md`
- Stage artifacts and server logs:
  `/home/hammad/projects/k2-horizon-inference/results/authoritative-prefix/`
- Reproducible candidate launcher:
  `/home/hammad/projects/k2-horizon-inference/scripts/serve-vllm-uno-authoritative.sh`
- Experimental vLLM worktree: `/home/hammad/projects/vllm-sm120`
- Extracted kernel package: `/home/hammad/projects/sm120-paged-attention`
- Living execution plan:
  [vllm-inference-optimization-platform.md](../plan/vllm-inference-optimization-platform.md)

The candidate launcher explicitly selects SM120 FA4, deterministic noise salt
41, block width eight, an FP32 output head, disabled prefix caching, and
`--compilation-config '{"mode":0}'`. Reproduction must preserve those choices
and must record the resolved server configuration rather than relying on the
script name.

## Revision history

- 2026-09-19: Created from the retained benchmark artifacts and server logs;
  corrected the Uno block contract, superseded the separate authoritative-root
  design, and recorded Stage 6b plus the next qualification sequence.
