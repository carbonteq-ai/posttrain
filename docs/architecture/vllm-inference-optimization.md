# vLLM inference optimization on Blackwell

> **Technical delivery architecture, not a product-contract amendment.**
> The frozen product baseline remains
> [docs/post-training/](../post-training/README.md). It already assigns runtime
> kernel, scheduler, speculative-decoding, and cache choices to an inference
> binding. This document records the implementation and qualification evidence
> for those choices.

Status: experimental candidate, not released  
Last revised: 2026-09-23

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

The retained counters do separate these two mechanisms to first order; see
[Gain attribution](#gain-attribution) below. A factorial A/B remains worthwhile
to isolate graph fragmentation from proposer execution, but it is no longer a
prerequisite for assigning a percentage.

## Gain attribution

Throughput factors exactly as `tokens per round x rounds per second`, where
rounds are `speculative_draft_tokens / 7` and tokens per round are
`total_output_tokens / rounds`. Both counters are already present in the
retained artifacts, so the stage history separates without new runs.

| Checkpoint | tok/s | tok/round | rounds/s | ms/round |
| --- | ---: | ---: | ---: | ---: |
| Stage 1 | 422.08 | 4.252 | 99.25 | 10.09 |
| Stage 6b fast | 526.41 | 4.688 | 112.29 | 8.91 |
| Strict (fixed tile + batch-invariant linears) | 270.45 | 4.870 | 55.54 | 18.01 |
| Strict + SM120 invariant-GEMM family | 445.36 | 4.880 | 91.28 | 10.96 |

Three conclusions follow.

- **The 64.67% invariant-GEMM gain is execution-only.** Tokens per round moved
  0.21% while rounds per second moved 64.35%. Stronger than the counters: all
  four response hashes are identical between
  `stage6b-strict-fixed-prefill-final-c4-run1` and
  `strict-k2-decode-tuned-v2-matched-c4-run1`. Same token streams, same
  acceptance, same attention kernel, same graph mode; only per-step execution
  time changed.
- **The 24.72% Stage 1 to Stage 6b gain factors as 10.25% speculative
  efficiency and 13.14% execution**, or roughly 44/56 in log-share terms. Two
  caveats remain: Stage 6b's outputs differ from the strict family (0/4 hash
  match), so the numerics change the proposals themselves and the acceptance
  delta is not independent; and Stage 1's own repeats span 8.6%, which is wider
  than several inter-stage deltas.
- **The stage table cannot speak to attention.** Every row ran
  `--attention-backend SM120_FA4`; this is confirmed in
  `server-stage0-c4-ab-20260919.argv.json`,
  `server-stage1-c4-ab-20260919.argv.json`, and the four retained server logs.
  SM120 FA4 is held constant, not varied. The only attention change inside the
  native series is the M64N64 prefill pin, which moved 526.41 to 488.30, a 7.24%
  cost.

### What the attention kernel is measurably worth

One confound-free FA2/FA4 pair exists, in the standalone Uno runtime at c4 with
256 fixed outputs, identical prompts and adapter revision, block width 16 on
both sides:

| Artifact | backend | tok/s | model GPU ms/forward |
| --- | --- | ---: | ---: |
| `uno-warm-fa2-linear-b16-fixed-256.json` | fa2 | 233.01 | 19.446 |
| `uno-warm-fa4-linear-b16-fixed-256.json` | fa4 | 276.12 | 15.665 |

That is +18.50% throughput from a 19.44% reduction in model GPU time, with
accepts identical at 1020/1020. The gain is kernel time, not scheduling.

Two corrections to how this result has been carried forward. The benchmark index
at `k2-horizon-inference/results/README.md` cites 276.96 tok/s for the FA4 side,
which is the **block-width-8** artifact; comparing it against FA2 at block width
16 varies two factors. The matched pair is the one tabulated above. Second, this
result has **never been reproduced inside vLLM**. In-vLLM FA2/FA4 evidence is at
most +2.44% and is confounded by a simultaneous CUDA 13.2 to 13.4 move,
FlashInfer sampling, and a different server instance;
`direct-vllm-platform-20260918/README.md` records that SM120 FA4 was not
selectable natively when that sweep ran.

For scale, holding attention constant, Uno speculative decoding itself is worth
+58.12% (183.26 to 289.77 at 16K outputs). Any attention claim below roughly 3%
is inside run variance and should not be quoted.

### Determinism accounting

The invariant-GEMM work is best described as recovering a cost, not adding
speed:

| Configuration | tok/s | versus non-invariant |
| --- | ---: | ---: |
| Batch invariance off | 526.41 | reference |
| Batch invariance on, untuned | 270.45 | -48.62% |
| Batch invariance on, SM120-tuned | 445.36 | -15.40% |

So the retained result is **batch-invariant determinism for a 15.4% tax rather
than a 48.6% tax**. That is the honest headline, and it is conditional on
determinism having value; if bit-exactness is not required, clearing
`VLLM_BATCH_INVARIANT` yields 526.41 with no kernel work. For RL rollouts the
requirement is real, because rollout logprobs must match what training
recomputes or the importance ratios are wrong.

**cuBLAS cannot supply this property.** It selects kernels and split-K
decomposition heuristically from problem size, so the reduction order changes
with M and therefore with batch composition. Setting
`CUBLAS_WORKSPACE_CONFIG=:4096:8` with deterministic algorithms yields
run-to-run determinism at a fixed shape; it does not yield invariance across
batch composition. Replacing cuBLAS with a fixed-reduction-order Triton kernel
is upstream vLLM's design decision, not a local shortcut.

The mechanism of the 64.67% is then narrow and unglamorous: upstream
`_get_tuned_matmul_arch_family` returned `None` for compute capability 12, so
every batch-invariant linear on SM120 used the generic 128x128x64 default. At c4
decode the real row count is about 36 against a 128-row tile. Production-layout
sweeps, recorded in
`invariant-matmul-transposed-config-sweep-c4-c32-20260919.json`:

| Shape | M | tuned | 128x128 | speedup |
| --- | ---: | ---: | ---: | ---: |
| qkv | 36 | 29.65 us | 100.29 us | 3.38x |
| o_proj | 36 | 20.85 us | 72.36 us | 3.47x |
| gate_up | 36 | 131.85 us | 279.87 us | 2.12x |
| down | 36 | 51.88 us | 185.25 us | 3.57x |
| proposer | 32 | 16.35 us | 74.37 us | 4.55x |
| vocab (FP32 head) | 4 | 2501.55 us | 2755.24 us | 1.10x |

## Generalization boundary

The retained work is three separable artifacts with three different reuse
profiles. Nothing here transfers by copying a config.

- **SM120 invariant-GEMM family.** Mechanism general, values K2-only. The table
  is keyed on exact `(N, K)` with four entries — `(6144,4096)`, `(4096,4096)`,
  `(24576,4096)`, `(4096,12288)` — implying hidden 4096 and intermediate 12288.
  A model that misses the lookup falls through to the upstream 128x128 default:
  no error and no gain. This is now generalized by a shape-agnostic rule; see
  [Cross-model generalization](#cross-model-generalization).
- **FP32 output head — corrected 2026-09-23.** An earlier revision of this
  section said the fp32 `M <= 64` branch in `matmul_persistent` covers the FP32
  head. It does not. With `head_dtype=float32`, `logits_processor.py` calls
  `torch.mm(..., out_dtype=torch.float32)`, which is the `aten::mm.dtype`
  overload; batch-invariant mode never overrides it, so the head ran on cuBLAS.
  Measured on SM120, cuBLAS switches kernels at 16 rows: from M=16 onward
  96–99% of a row's logits differ from the same row computed alone. K2 token
  streams stayed exact only because the differences (~2e-5) rarely flip a greedy
  argmax; logprobs were not batch-invariant at 16 or more concurrent sequences.
  Fixed on the fork branch by routing the invariant FP32 head through
  `matmul_persistent(..., out_dtype=torch.float32)`, which keeps the bf16 tile
  and reduction order and only widens the store.
- **Extracted SM120 FA4 attention.** Head sizes 64, 96, 128, 256 only. Head
  dimension 512 is not a tuning gap: the smallest candidate tile needs 198,656
  bytes of shared memory against SM120's 101,376-byte capacity, and head
  dimension 256 at (64,64) already lands at exactly 101,376 with zero headroom.
  This is silicon, not configuration. Gemma 4's mixed sliding/full topology
  remains unqualified independently of that.
- **Hybrid models.** Batch-invariant mode refused every conv/linear-attention
  backend. Measured 2026-09-23: LFM2.5's short-conv is invariant and is now
  declared so on the fork branch; Qwen3.5's GDN is genuinely not invariant and
  stays gated. `SM120_FA4` on a hybrid KV-cache layout remains untested.

## Cross-model generalization

Added 2026-09-23. Fork branch `codex/sm120-generic-invariant-gemm` (uncommitted
at time of writing); tools and artifacts under `k2-horizon-inference/benchmarks`
and `results/remote-dev/`.

### Method

1. **Capture real shapes.** `capture_invariant_gemm_shapes.py` runs each model
   in an in-process vLLM engine (dummy weights, batch-invariant, eager) and
   records every `(M, N, K)` sent through `_get_matmul_config`. Ten models:
   K2-Horizon-7B, Gemma-4 12B/31B/E4B/E2B, Qwen2.5-0.5B, LFM2.5 2.6B/1.2B,
   Qwen3.5 2B/27B — 71 unique bf16 shapes, hidden sizes 896 to 5376.
2. **Sweep.** `sweep_invariant_gemm.py` times up to 128 tile configurations per
   shape at M = 1, 2, 4, 8, 16, 32, 36, 64, 72, 128, 144, 288. Unlike the
   original K2 tool it rotates enough weight copies to exceed L2 (a production
   layer reads weights from HBM) and times CUDA-graph replays (no Python launch
   overhead). Best decode configurations reach ~86% of HBM bandwidth.
3. **Invariance check in the sweep.** Every configuration sharing `BLOCK_K = 64`
   produced bit-identical rows for the same inputs: **949 configurations per
   shape, 71 shapes, zero mismatches.** Any M-bucketing of such configurations
   is therefore batch-invariant by construction.
4. **Fit and cross-validate.** `analyze_invariant_gemm_sweep.py` fits one
   configuration per (N range x M range) cell — N edges 512/2048/8192/32768/∞,
   M edges 1/4/16/32/36/64/72/144/288 — and validates leave-one-model-out: the
   rule is fitted without a model's shapes and scored on them.

### Kernel-level result

The 45-cell rule is 1.03x the per-shape optimum overall and 2.7x faster than the
upstream default. Held out, every model lands at 1.02–1.08x its own optimum
(mean 1.039), so the rule generalizes rather than memorizes. Buckets at M=36 and
M=72 were added because Uno blocks land at 9 x concurrency; they improved the
held-out error, which rules out overfitting as the reason.

The fork checks the exact K2 table first and falls back to the rule on a miss,
so K2 keeps its validated configurations. M > 288 keeps the upstream tile until
the prefill sweep is folded in; early prefill gains at M=2048 are only
1.01–1.14x, as expected for compute-bound GEMMs.

### End-to-end result

Real weights, in-process engine, full decode CUDA graphs, `mode: 0`, 512-token
prompts, 256 greedy output tokens, no speculative decoding, two repeats.
Aggregate decode tok/s:

| Model | c | invariance off | invariant, upstream default | invariant, generic rule | determinism cost default → generic |
| --- | ---: | ---: | ---: | ---: | --- |
| K2-Horizon-7B (fp32 head) | 4 | 333 | 134 | 316 | -60% → -5.1% |
| K2-Horizon-7B (fp32 head) | 32 | 1,887 | 1,026 | 1,858 | -46% → -1.6% |
| LFM2.5-2.6B | 4 | 908 | 249 | 849 | -73% → -6.6% |
| LFM2.5-2.6B | 32 | 5,645 | 2,069 | 5,390 | -63% → -4.5% |
| Gemma-4-12B | 4 | 225 | 87 | 204 | -61% → -9.2% |
| Gemma-4-12B | 32 | 1,187 | 645 | 1,142 | -46% → -3.8% |
| Qwen2.5-0.5B | 4 | 2,532 | 772 | 2,267 | -69% → -10.5% |
| Qwen2.5-0.5B | 32 | 16,126 | 6,383 | 15,676 | -60% → -2.8% |
| Gemma-4-E4B | 4 | 476 | 155 | 401 | -67% → -15.8% |
| Gemma-4-E4B | 32 | 2,792 | 1,246 | 2,606 | -55% → -6.7% |

On K2 the generic rule matched the hand-tuned table at every concurrency (c4
316 vs 311, c32 1,858 vs 1,860), so the table is no longer necessary for plain
decode; Uno's M=36 operating point should get one end-to-end check before the
table is removed. Smaller models keep more residual cost because GEMMs are a
smaller share of their step and invariance's other substitutions (softmax,
log-softmax, mean) weigh more.

### Invariance evidence

- **Batch size.** In every invariant run, request 0's tokens and per-token
  sampled logprobs were bit-identical at c1/c4/c8/c16/c32 and across repeats.
  With invariance off, every model drifted: max |Δlogprob| 0.067 (LFM) to 0.19
  (Gemma-4-E4B), and Gemma's greedy first token flipped from c4 upward.
- **Continuous batching.** `e2e_staggered_invariance.py` adds long-prompt
  arrivals while request 0 decodes, forcing 24 steps where prefill chunks share
  the step with its decode. LFM2.5-2.6B, K2 and Gemma-4-12B stayed bit-exact;
  the invariance-off control drifted at 136 steps.
- **Chunk boundaries.** With arrivals queued ahead of it, request 0's own
  3,000-token prompt is split at different boundaries than when run solo.
  LFM2.5 stayed bit-exact; the control drifted at all 160 steps (max 0.075).
- **Hybrid layers.** Forced through the gate, short-conv (LFM2.5) was invariant
  in all three tests above; GDN (Qwen3.5-2B) drifted by up to 0.033 in both the
  batch-size and staggered tests, identically with or without invariance. The
  generic rule still recovers Qwen3.5's GEMM throughput (c4 918 vs 345 tok/s)
  but cannot make it deterministic.

### Hybrid layers after the GEMM rule

Three GDN defects were found by hooking every module of Qwen3.5-2B and finding
the first tensor that diverged between a solo and a batched run. Each is fixed
on the fork branch:

- FlashInfer's GDN prefill kernel writes a request's output and final state
  differently depending on its batch. Under invariance the fork uses the
  in-tree Triton prefill and rejects an explicit FlashInfer or CuTe DSL request.
- Pure-decode steps ran the packed recurrent kernel while steps shared with a
  prefill ran the sigmoid-fused kernel, and the two disagree in the last bits.
  Mixed steps now run decode tokens through the packed kernel.
- The gated RMSNorm tiled 1, 2 or 4 rows per program by batch row count,
  changing each row's reduction order; invariance now forces one row.

Qwen3.5-2B and 27B are then bit-exact at c1–c32 and across mixed steps (c4:
345 → 921 tok/s on the 2B, 47 → 101 on the 27B). One gap remains: with chunked
prefill a long prompt can be split at different points depending on its
neighbours, and GDN's result depends on the split point (a 3,000-token prompt
split at 1,536 vs 2,048 flips its second token). With chunked prefill off both
staggered tests are bit-exact, and the fork warns when both are enabled. The
cause is not yet located: the state-carry kernel keeps fp32 state across calls
symmetrically. Separately, several FLA kernels (`chunk_o`,
`chunk_scaled_dot_kkt`, `l2norm`) autotune configurations that change the
reduction split, so two processes can pick different configurations and
produce different bits; within one process the choice is fixed.

### Invariant attention: fixed-segment split-KV

After the GEMM rule, Gemma's remaining cost was attention. The Triton unified
attention kernel splits a decode request's KV cache into 16 segments only at
small batch, so upstream invariant mode never splits and Gemma-4-12B ran 32
attention programs on 188 SMs at c4. The fork cuts KV at fixed 128-token
boundaries (absolute positions; the length grows only with `max_model_len` to
cap segments at 256) and folds segments in order through one shared merge
function in both kernels. The split and single-pass kernels then return
identical bits, so the backend keeps choosing by batch size. Head-512 layers
use one program per segment to fit shared memory. Moving the sliding-window V
mask into the V load also fixed a head-512 sliding-window launch failure that
occurs with invariance off.

| Model | c | invariance off | GEMM rule only | + split-KV | determinism cost |
| --- | ---: | ---: | ---: | ---: | --- |
| Gemma-4-12B | 1 | 56.9 | 51.3 | 54.8 | -9.8% → -3.7% |
| Gemma-4-12B | 4 | 224.9 | 204.0 | 214.7 | -9.3% → -4.5% |
| Gemma-4-12B | 32 | 1,195.5 | 1,147.4 | 1,155.0 | -4.0% → -3.4% |
| Gemma-4-E4B | 1 | 115.1 | 97.7 | 106.1 | -15.1% → -7.8% |
| Gemma-4-E4B | 4 | 476.4 | 400.9 | 434.5 | -15.8% → -8.8% |
| Gemma-4-E4B | 32 | 2,791.8 | 2,605.8 | 2,620.8 | -6.7% → -6.1% |

All runs are bit-exact at c1–c32 and across staggered arrivals. At c4,
Gemma-4-12B attention fell from 1,619 to 747 µs per step (712 with invariance
off). The remaining +784 µs is norm (+280), GEMM (+252) and small kernels
(+217, of which 98 µs is the segment reduce). FlashAttention models (K2, Qwen,
LFM) still run unsplit attention under invariance; K2's attention costs +26%
at c4.

CUDA graph memory in these runs was 0.01–0.06 GiB per model when capturing up
to batch 32; the 0.2–0.8 GiB seen in the K2 server logs comes from capturing 54
sizes up to 512.

### Fork changes

- `batch_invariant_configs.py`: `_SM120_GENERIC_RULE` and
  `_sm120_generic_matmul_config`, used on SM120 when the exact table misses.
- `batch_invariant.py`: `matmul_persistent(..., out_dtype=...)`.
- `logits_processor.py`: invariant FP32 head through `matmul_persistent`.
- `short_conv_attn.py`: `supports_batch_invariance() -> True`.
- `test_matmul_batch_invariant.py`: rule contract, table precedence, bucket-edge
  bit-equality on five unseen shapes, FP32-head invariance, hybrid
  declarations, GDN prefill routing. 45 tests pass on the RTX PRO 6000.
- `qwen_gdn_linear_attn.py`, `gdn/base.py`, `layernorm_guard.py`,
  `bailing_moe_v3.py`, `gdn_attn.py`: the three GDN fixes and a per-class
  `batch_invariance_validated` gate that keeps unvalidated GDN families
  (OLMo-hybrid, Kimi, Bailing) out of invariant mode.
- `triton_unified_attention.py`, `triton_attention_helpers.py`,
  `triton_attn.py`: fixed-segment split-KV and the V-load window mask.
  `test_attention_batch_invariant_segments.py` checks bit-equality across
  kernels, batch mixes, decode vs prefill and chunked prefill for head sizes
  128, 256 and 512 (34 tests); the 1,588 existing Triton attention tests pass.

## Known defects

These are open and were found by inspection on 2026-09-23, not by a failing run.

1. **The kernel package's test suite asserts nothing about numerics.** One file
   is collected by pytest: three tests, no GPU, no arithmetic. The three GPU
   scripts are not named `test_*`, are never collected, and contain no
   assertions — they print and exit 0 on corrupt output. A narrow probe passing
   while a full model is corrupt is permitted by construction, which is the
   Gemma failure mode exactly.
2. **`capabilities.py` is dead code.** The head-size allowlist is imported by
   nothing on the launch path, so an unsupported head dimension surfaces as a
   tile-selector `ValueError` rather than a clean capability rejection.
3. **`tile_mn` is not part of the decode launch-plan cache key** and is not
   forwarded to `try_cached_paged_decode`. Once a plan is cached for a tensor
   signature, a later call with a different `tile_mn` silently reuses the cached
   tile. This weakens the determinism pin precisely under continuous batching,
   where shape churn is highest.
4. **`set_batch_invariant(True)` is a no-op on SM120.** The `batch_invariant`
   argument to `BlockInfo` is wired only into the SM100 kernel; the SM120 path
   constructs it without that argument and defaults to `False`.
5. **Four hardcoded shapes stand in for a rule.** Addressed on the fork branch
   by the generic rule; see Cross-model generalization.
6. **The aten-level invariant overrides are never installed on SM120.**
   `enable_batch_invariant_mode` registers the Triton `mm`/`addmm`/`matmul`/
   `linear` overrides only for SM80; every other capability, SM120 included,
   relies on cuBLASLt with split-K disabled, on the stated assumption that
   split-K is the only source of batch variance. On SM120 that assumption is
   false: cuBLAS changes kernels at 16 rows. Model projections are unaffected
   because `UnquantizedLinearMethod` calls `linear_batch_invariant` directly,
   and the FP32 head is fixed on the branch, but any other `torch.mm`,
   `F.linear` or `torch.matmul` on the hot path still runs on cuBLAS.
   Bit-exact end-to-end logprobs for five models suggest none of them currently
   matter; a new model or feature that adds one would silently lose invariance.
7. **GDN (Qwen3.5 linear attention) depends on chunked-prefill split points.**
   Batch composition and mixed steps are fixed on the branch; split-point
   dependence remains (see Hybrid layers after the GEMM rule). Disable chunked
   prefill for full reproducibility until it is fixed.
8. **FLA kernels autotune configurations that change reduction order.** Two
   engine processes can choose different configurations and disagree in the
   last bits. Invariant mode should pin one configuration per kernel.

Items 1 and 3 should be closed before the kernel is carried to any further
model, because together they mean a new model's correctness cannot currently be
established.

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

Partly answered; see [Gain attribution](#gain-attribution). The retained
counters already split Stage 1 to Stage 6b into roughly 44% speculative
efficiency and 56% execution. The matrix below is still worth running, but its
purpose is now narrower: separating graph fragmentation from proposer execution,
and adding the FA2-versus-FA4 cell that has never been measured natively.

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
- 2026-09-23: Added gain attribution, the determinism accounting, the
  generalization boundary, and known defects. Factored the 24.72% Stage 1 to
  Stage 6b gain from the retained counters rather than deferring it to a
  factorial A/B; established that the 64.67% invariant-GEMM gain is
  execution-only on identical response hashes; recorded that SM120 FA4 is held
  constant across the entire stage history and that its one clean measurement
  (+18.50%) exists only in the standalone runtime; corrected the block-width
  mismatch in the benchmark index's FA2/FA4 comparison; and recorded the shared
  memory limit that makes head dimension 512 unreachable on SM120.
- 2026-09-23 (later): Generalized the invariant GEMM tuning to ten models with a
  cross-validated shape-agnostic rule and measured it end to end on five with
  real weights. Corrected the earlier claim that the FP32 `M <= 64` heuristic
  covers the FP32 head: the head used `aten::mm.dtype` on cuBLAS and was not
  batch-invariant at 16+ sequences. Recorded that the aten overrides are
  SM80-only, that short-conv is invariant while GDN is not, and the
  staggered-arrival and chunk-boundary evidence.
- 2026-09-23 (evening): Recorded the three GDN fixes and the remaining
  split-point dependence, the FLA autotuning risk, and fixed-segment split-KV
  for invariant Triton attention (Gemma-4-12B c4 cost 9.3% → 4.5%, E4B 15.8% →
  8.8%, bit-exact).
