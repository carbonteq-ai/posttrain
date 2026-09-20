# Make vLLM inference optimizations reusable across model families

This ExecPlan is a living document and must be maintained in accordance with
`docs/templates/PLAN.md`. The repository does not currently contain the
`.agents/PLAN.md` named by the local workflow skill, so the checked-in template
is the operative plan authority.

## Purpose / Big Picture

After this work, Posttrain can select and reproduce hardware-appropriate vLLM
attention kernels, speculative-proposer execution modes, and KV-cache formats
without embedding K2, Uno, Nanbeige, or one GPU name in the reusable adapter.
An inference binding will express these runtime choices, detached validation
will reject combinations known to be invalid, TRL and serving will translate
the same logical choices, and retained benchmarks will prove which combinations
are actually faster and correct on each hardware target.

The work preserves a critical distinction. A capability being present in vLLM
does not make it a default. A catalog binding selects a qualified operating
point, while a workload defines warmup, prompt population, output budget, and
concurrency. This allows a Blackwell FA4 binding, an FA2 plus TurboQuant binding,
and an eager fallback to coexist without changing model identity.

## Progress

- [x] (2026-09-20 02:20Z) Published the release-clean SM120 kernel as
  `sm120-paged-attention` v0.1.0 at
  `99a6fe0acbb4756735aa8e47236f8b74e3f7c4be`, then published CarbonTeq vLLM
  `carbonteq-v0.29.1.dev2` at
  `fbbba6698b2f8a912b94705cfc09eb4fd7243716`. The retained vLLM archive
  SHA-256 is `cd44782606be23fa4a0195fd90e7377a5adf8f3d9bcc7984a21743a046b7d44e`.
  Dev2 supersedes dev1 because the first source release exposed a missing fork
  tag-provenance rule and resolved as vLLM 0.26.1 instead of 0.29.1.
- [x] (2026-09-20 02:25Z) Removed rejected authoritative-prefix,
  proposer-graph, deterministic-noise, async-debug, and SM120 environment
  switches from the release surface. The released topology is eager Uno
  proposal, ordinary target decode graphs, native policy-LoRA composition,
  released SM120 FA4, and shape-tuned invariant linears. The released-kernel
  matched c4 control produced 435.78 output tok/s with zero preemptions.
- [x] (2026-09-20 02:30Z) Advanced Posttrain's direct and runtime vLLM pins to
  the immutable dev2 commit, paired it with the upstream-base CUDA 13 wheel
  `2b8f85d32b952cb05743c2edb02c4ae5be59921445b97069a609f186fdf6c5f0`,
  and made the compile-time policy admit Uno inference and native LoRA only.
  Full-weight and QLoRA Uno refresh now fail before submission.
- [x] (2026-09-19 10:35Z) Recovered the matched Stage 0/1 artifacts, corrected
  the Uno block contract, and completed the Stage 6b c4 repeat. The retained
  experimental topology is full target decode CUDA graphs, eager Uno proposer,
  SM120 FA4, deterministic noise, FP32 output head, and all eight Uno block
  positions presented to target verification. It produced 528.12 and 524.70
  output tokens/s (526.41 mean), 24.72% above the Stage 1 mean, with identical
  response hashes across repeats and zero preemptions.
- [x] (2026-09-19 10:40Z) Identified the two observed contributors without
  over-attributing them. Stage 6b removed Stage 1's 88 PIECEWISE graph segments
  while retaining 62 full target decode graphs, and mean acceptance increased
  from 48.45% to 54.64%, reducing candidate drafts by 9.31% for the same 4,096
  outputs. The proposer was eager in Stage 6b; no live proposer-graph
  qualification appears in its server log.
- [x] (2026-09-19 10:45Z) Rejected the separate authoritative-root execution
  model. Uno width eight is one clean target-policy root plus seven future
  drafts, but all eight remain candidates in one-pass autoregressive target
  verification. Removing the root from that block collapsed acceptance to
  3.29% and throughput to 196.21 output tokens/s.
- [x] (2026-09-19 10:55Z) Added the durable
  [Blackwell vLLM optimization architecture](../architecture/vllm-inference-optimization.md)
  and updated the benchmark report with the complete retained/rejected stage
  history and prioritized qualification roadmap.
- [ ] Run the Stage 6b attribution matrix with fixed proposals and outputs:
  full target graphs plus eager proposer; full plus PIECEWISE with independent
  proposer graphs; a dedicated whole-proposer graph without target
  fragmentation; and all-eager. Record cycles, forward counts, CUDA time, CPU
  gaps, and host/device traffic before assigning causal percentages.
- [ ] Qualify the winning topology in fresh processes at c1/c4/c8/c16/c32 and
  on the fixed-16K/KV-growth workload, then run the real policy-LoRA optimizer
  refresh, fresh-logprob, and policy-generation-fence gate.
- [x] (2026-09-19 11:20Z) Extended Stage 6b on the same warm server to c8,
  c16, and c32 with two repeats per level. Mean aggregate throughput was
  704.75, 1,201.27, and 1,490.23 output tokens/s, with zero preemptions and
  peak KV occupancy of 12.16%, 24.07%, and 49.36%. This is 6.82% above the
  earlier native path at c16/c32 and within 2.52%/2.32% of standalone.
- [x] (2026-09-19 12:10Z) Localized the remaining high-concurrency divergence
  with batch-position, target-only, layer, and prompt-boundary probes. The
  extracted SM120 kernel chose batch-dependent prefill tiles; automatic tiling
  changed 191,314 BF16 elements for a fixed row, while M64N64 was bit-exact at
  c1/c2/c4/c8. Full-model exactness additionally required vLLM's invariant
  linear path. The strict final diagnostic matched c8 8/8, c16 16/16, and c32
  32/32 responses; two c4 runs averaged 270.45 output tok/s.
- [ ] (2026-09-19) Optimize the qualified native Uno hot path in independently measured stages. Stage 1 removes proposal-loop CPU/GPU synchronizations and transient metadata allocations while preserving proposer-owned CUDA-graph buffers. Later stages will separately qualify GPU-resident speculative scheduling, fused probabilistic verification, policy-LoRA routing, and fused SM120 KV append/attention. Each stage requires focused CPU tests, eager-versus-replay correctness, and a matched warm GPU benchmark before the next stage is attributed any gain.
- [x] (2026-09-19) Reproduced greedy c4 output divergence with actual token IDs and top-two logprobs; c1 repeated exactly. FP32 output projection reduced c4 mismatches from four requests to one on the 512-token diagnostic, so it is not by itself a qualified fix.
- [x] (2026-09-19) Implemented request-stable deterministic Uno noise in the experimental `vllm-sm120` fork. CPU and CUDA results match standalone reference noise for four salts, four widths, and reordered/subset batches. Random-noise defaults are unchanged.
- [ ] Qualify the integrated noise fix, isolate target/clean-row numerical disagreement, and retain a fast configuration only after token-level repeats and independent target verification. No framework baseline amendment or dependency pin change is required for this experimental diagnosis.
- [x] (2026-09-19) Matched standalone and native concurrency sweeps on unique 5K prompts with 1K fixed outputs. Standalone reached 1,232.36 mean aggregate tok/s at c16 and 1,525.63 at c32 with exact repeat hashes. Native reached 1,124.56 and 1,395.06 respectively, but retained four changing responses at each high-concurrency level.
- [x] (2026-09-19) Isolated one native variance source in the extracted SM120 kernel. Automatic splitting changed identical attention outputs at c16/c32; one split was bit-exact through c32. Added an SM120-only deterministic policy, which repaired the c1/c4 token gate without the 41–53% throughput loss of global batch-invariant mode.
- [x] (2026-09-19 17:20Z) Recover the strict path's model-linear throughput
  without weakening the
  exactness gate. The broad persistent Triton invariant-linear dispatch costs
  approximately 48.6% versus the unqualified Stage 6b fast mean. Evaluate a
  shape-stable faster linear strategy against the retained c8/c16/c32 token
  oracle before selecting a release profile.
- [x] (2026-09-19 17:20Z) Added a dedicated SM120 invariant-matmul config family
  after a production-layout sweep of K2 qkv, output, gate/up, down, proposer,
  and FP32 vocabulary shapes at c4/c8/c16/c32. The retained table keeps a fixed
  K tile, measured decode buckets, and the original large-prefill tile. Matched
  c4 throughput rose from 270.45 to 445.36 tok/s (+64.67%); two-repeat token
  diagnostics remained exact at c8 8/8, c16 16/16, and c32 32/32. A single
  throughput sweep measured 601.18, 1,018.87, and 1,106.51 tok/s respectively.
- [x] (2026-09-19 17:05Z) Rejected selective invariant-linear routing after it
  matched only 5/8 c8 responses, and rejected a generic small-M tile after it
  displaced the large-prefill configuration. Also corrected the GEMM benchmark
  to use vLLM's real transposed `[N,K]` weight layout before retaining results.
- [ ] Port the newer logits-native rejection-sampling contract into the active
  legacy GPU runner. Preserve mixed greedy/stochastic sampling, per-request
  generators, penalties and masks, target correction, bonus tokens, raw and
  processed logprobs, heterogeneous vocabulary safety, and deterministic
  reference equivalence before measuring memory traffic or speed. Do not copy
  the narrower standalone fused kernel without these contracts.
- [x] (2026-09-19 12:17Z) Ran the newer logits-native kernel qualification
  suite alongside the retained strict K2 server. Forty-five tests passed. Eight
  statistical block-verification cases failed before kernel execution because
  their 3.1-7.5 GiB fixtures could not coexist with the server's 87.34 GiB GPU
  allocation on a 94.97 GiB card. Rerun those eight with the server stopped as
  a dedicated gate; do not treat the allocation failures as correctness
  failures or as passing evidence.

- [x] (2026-09-19) Documented the [CUDA graph debugging guideline](../tooling/vllm/cuda-graph-debugging.md): upstream isolation controls, import-time breakable activation, saved-batch replay, first-divergence localization, and correctness-before-performance gates. Documentation only; no new GPU qualification is claimed.
- [x] (2026-09-19 07:15Z) Verified import-time breakable activation and 36 eager attention breaks across 37 captured segments, removed duplicate target/proposer graph ownership, and repaired the live qualifier. The apparent descriptor corruption was confined to one dispatcher-padding row per request: validating only the real Uno rows produced bit-exact eager, initial-replay, and changed-input-replay outputs, logits, selected tokens, and logprobs for request counts 1-4 under batch churn. The first warm medium c4 fixed-128 diagnostic reached 192.99 output tokens/s at 46.14% draft acceptance with zero preemptions; this short run is not comparable to the retained 16K baseline.
- [x] (2026-09-18 10:10Z) Audited the released vLLM tag, Posttrain 0.4.3 bindings, retained K2/Uno benchmarks, and older Nanbeige/TurboQuant branch.
- [x] (2026-09-18 10:20Z) Confirmed that the frozen product baseline already assigns attention kernels, speculative execution, and KV-cache format to `InferenceBinding.engine`; no baseline amendment is required.
- [x] (2026-09-18 10:45Z) Unified attention, cache, prefix-cache, and scheduler flag translation across serving and colocated TRL while keeping the low-level translators policy-free.
- [x] (2026-09-18 11:35Z) Moved deterministic vLLM composition policy into resolved-job compilation: explicit TurboQuant plus FA3/FA4 and the pinned DSpark plus TurboQuant composition now fail before packing or provider submission, including when optional readiness preflight is skipped.
- [x] (2026-09-18 12:30Z) Ran source-overlay experiments directly on the RTX PRO, without Docker or Posttrain submission, and rejected two unsafe fast paths: Uno proposer CUDA replay collapsed acceptance to 0.23-0.35%, and native vLLM FA4 on SM120 reached the bundled kernel but rejected paged KV.
- [x] (2026-09-18 12:40Z) Compared eager Uno draft widths K=3, K=5, and K=7 on one warm c4 workload. K=7 retained the best point throughput at 178.87 output tokens/s despite lower per-draft acceptance.
- [x] (2026-09-18 12:50Z) Refreshed the vLLM community audit through upstream `44dd18fe0`: the maintained path now includes a paged causal `b12x` backend for SM120/SM121 and a newer native FA4 head-dimension-256 path, while general SM120 FA4 paged-KV coverage remains shape- and feature-constrained.
- [x] (2026-09-18 17:55Z) Rebased the CarbonTeq Uno delta onto upstream `44dd18fe0` in the separate active `/home/hammad/projects/vllm-sm120` worktree, preserving the existing `vllm-uno` worktree and its uncommitted proposer-safety clarification.
- [x] (2026-09-18 18:15Z) Added ordered attention-backend preferences with platform fallbacks, resolved backend-preferred KV block size before model construction, and made an unsupported requested FA version warn and fall back to FA2 rather than fail after GPU allocation.
- [x] (2026-09-18 18:29Z) Completed the first matched SM120 `b12x` versus FA2 experiment on K2 at warm concurrency four: `b12x` improved aggregate output throughput by 1.47% at fixed 1,024-token outputs and 2.65% at fixed 16,384-token outputs.
- [x] (2026-09-18 18:55Z) Audited upstream vLLM issues, pull requests, and public branches for SM120 paged FA4. The closest work is closed PR #40110's eager adapter, open PR #47218's capability gate, and open PR #48156's warmup work; none implements the complete SGLang-style SM120 paged-KV and SplitKV path.
- [x] (2026-09-18 19:00Z) Extracted SGLang v0.5.19's qualified SM120 specialization and required CuTe support closure into the independent `/home/hammad/projects/sm120-paged-attention` distribution. Its 308 KB compressed wheel (about 1.6 MB installed source) has no SGLang or vLLM runtime import and retains Apache-2.0/BSD-3-Clause provenance.
- [x] (2026-09-18 19:05Z) Added an opt-in `SM120_FA4` backend adapter to the current-upstream vLLM candidate. It retains vLLM ownership of block tables, cache writes, output buffers, eligibility, and fallback, and remains outside automatic platform defaults pending GPU qualification.
- [x] (2026-09-18 19:45Z) Qualified the extracted package against vLLM's CUTLASS 4.7.1 environment on the RTX PRO: direct paged-GQA output was finite with 0.004304 maximum absolute error, first compilation took 1.746 seconds, and warm replay took about 90 microseconds.
- [x] (2026-09-18 20:20Z) Served K2 through the opt-in vLLM adapter and ran a matched warm, concurrency-four, 8K-30K-prompt control. The extracted cached path delivered 66.57 output tokens/s versus FA2's 66.04 (+0.80%), establishing correctness/parity but not a material optimization.
- [x] (2026-09-18 20:55Z) Reproduced the retained 16K-output matrix on clean current-vLLM servers. Native vLLM plus eager K=7 Uno and the extracted SM120 FA4 target produced 288.84 output tokens/s with zero prefix-cache hits; current-vLLM Gemma 4 12B plus Triton reproduced its retained control at 192.03 output tokens/s.
- [x] (2026-09-18 21:20Z) Tested and rejected two attempted Gemma extensions before a long run: an experimental head-dimension-512 specialization produced 8.15 output tokens/s and degraded text, while a per-KV-kind hybrid using SM120 FA4 for head-dimension-256 sliding-window layers and Triton for head-dimension-512 full-attention layers produced 8.49 output tokens/s with the same corruption. Reverted the HD512 production eligibility and retained only the direct GPU smoke harness.
- [x] (2026-09-18 21:59Z) Implemented the owner-isolation groundwork for Uno proposal graphs: proposer-specific graph namespaces, persistent token/position/sequence/block-table/slot/LoRA-ID buffers, full padded-row refresh, capture keys covering batch shape, draft width, dtype, and LoRA structure, and eager fallback for uncaptured shapes. Kept proposal graphs disabled after the real GPU gate exposed an unresolved interaction between vLLM's dummy-LoRA capture lifecycle and the position-gated Uno adapter; policy-plus-Uno composition is also explicitly eager.
- [x] (2026-09-18 22:35Z) Reframed the failed Uno graph attempt as a generic proposer-execution problem. Source inspection separated three contracts that must be repaired independently: proposer capture must not run inside target dummy-LoRA capture, system adapters need stable reserved slots outside request/dummy LRU churn, and target/proposer LoRA routing metadata needs owner-scoped storage. The implementation will extend vLLM's existing proposer hierarchy rather than introduce a parallel speculative-decoding framework.
- [x] (2026-09-18 22:50Z) Implemented the first generic proposer-execution slice in `/home/hammad/projects/vllm-sm120`: typed capture ownership, target/proposer capture separation, reserved system-adapter slots, compiler-stable owner-banked Punica metadata, startup eager-versus-capture/replay qualification, and automatic eager fallback. RTX PRO qualification proved ordinary no-LoRA proposer graphs exact for batch sizes 1-4 but rejected position-gated Uno/Punica capture before replay; the server then started safely in eager proposer mode and completed 1,792 warm smoke tokens at 360.21 output tokens/s with 42.70% draft acceptance.
- [x] (2026-09-18 23:10Z) Established the pre-composer Gemma 4 12B control on the RTX PRO using the pinned `google/gemma-4-12B-it-assistant@364bd03c9952e5b7da73665ee30c9eccfc408345`. On identical warm, cache-neutral, concurrency-four prompts with 4,096 fixed output tokens per request, ordinary decoding produced 181.43 output tokens/s and paired-assistant MTP-2 produced 253.93 output tokens/s (+39.96%) while reducing wall time from 90.30 to 64.52 seconds (-28.55%). MTP accepted 10,943 of 11,132 proposed tokens (98.30%).
- [x] (2026-09-18 22:18Z) Added the first target-authoritative-prefix slice in `/home/hammad/projects/vllm-sm120`. Uno can now sample its adapter-disabled seed row through vLLM's target sampler, retain the exact policy logprob, emit that root immediately, and send only later Uno candidates to rejection sampling. The path is explicit and automatically falls back for stateful penalties, allowed-token/grammar masks, bad words, custom processors, full-vocabulary logprobs, and thinking budgets. Posttrain serving and TRL preserve the opt-in setting; 22 focused fork tests, 113 focused Posttrain tests, Ruff, and diff checks pass.
- [x] (2026-09-18 22:32Z) Attempted a bounded RTX source-overlay identity smoke through the healthy dstack control plane. The first task exited before Python because the image executes task blocks with `/bin/sh`, which rejects `pipefail`; the corrected retry received no offer while that worker remained in teardown. Both attempts were aborted, no model process ran, and no GPU result is claimed.
- [x] (2026-09-19 01:17Z) Replaced the unsafe startup-only Uno graph oracle with a live, real-KV qualification gate on the RTX PRO. The gate snapshots and restores only touched KV blocks, discards unstable weak capture outputs, compares eager against the first replay, mutates token IDs and the system-LoRA row mask for a second eager/replay comparison, and permanently falls back per descriptor on mismatch. Batch descriptors 1-4 all failed the initial-replay hidden-state gate, so none were promoted. The fail-closed warm c4 control remained healthy at 174.75 output tokens/s, 40.45% draft acceptance, all eight positions contributing, and zero preemptions; raw artifact: `/home/hammad/projects/k2-horizon-inference/results/authoritative-prefix/replay-qualified-graph-medium-c4-fixed128.json`.
- [x] (2026-09-19 07:15Z) Corrected that live gate after proving it compared padded dispatcher rows as model outputs. The production candidate now has one graph owner, proposer-descriptor allocator pools, retained eager-boundary and final-output buffers, a same-stream capture warmup, actual-token-aware qualification, and descriptor-local eager fallback. Real rows are bit-exact for c1/c2/c3/c4 and for mutated-input replay. Raw short diagnostic: `/home/carbonteq-ai-workstation/src/k2-horizon-inference/results/authoritative-prefix/native-graphs-correct-medium-c4-fixed128.json`.
- [x] (2026-09-19 07:17Z) Ran the larger warm 8.8K/17.6K/26.4K/30.8K-prompt, c4, fixed-1,024-output diagnostic after all descriptors qualified. It produced 153.21 output tokens/s, 38.19% acceptance, zero preemptions, and 21.68% peak KV occupancy, versus 142.67 output tokens/s and 29.13% acceptance in the earlier lazy-live-graph artifact. Raw artifact: `/home/carbonteq-ai-workstation/src/k2-horizon-inference/results/authoritative-prefix/native-graphs-correct-large-c4-fixed1024.json`. This is still not the retained fixed-16K comparison.
- [x] (2026-09-19 07:31Z) Completed the matched warm c4 fixed-16,384 FA2 control with the corrected proposer graphs: 65,536 output tokens in 428.43 seconds, or 152.97 output tokens/s, at 36.26% acceptance, zero preemptions, and 32.94% peak KV occupancy. This result is not comparable to the optimized 335.25 standalone or 288.84 native controls because those selected the extracted SM120 FA4 target backend; it establishes that proposer replay alone does not overcome the FA2 target cost. Raw artifact: `/home/carbonteq-ai-workstation/src/k2-horizon-inference/results/authoritative-prefix/native-graphs-correct-large-c4-fixed16384.json`.
- [x] (2026-09-19 07:53Z) Added and directly qualified an opt-in SM120 uniform-block path that preserves `(request, block, head, dim)` rather than flattening every speculative batch through the varlen interface, then enabled uniform-batch attention capture behind a separate experimental gate. Initial and changed-input replay were bit-exact on semantic hidden states, logits, selected tokens, and logprobs for live c1-c4 batch churn. The matched stochastic fixed-16K run produced 345.75 output tokens/s at 55.77% acceptance, zero preemptions, and 34.65% peak KV occupancy. It exceeded the older stochastic standalone 335.25 run, but it did **not** beat the repeatable deterministic standalone 4K operating point of 474.67 output tokens/s or the one-off deterministic standalone 16K diagnostic of 557.72 output tokens/s. Because random-noise acceptance has historically varied materially, this is candidate evidence rather than an execution-speed improvement claim. Raw artifact: `/home/carbonteq-ai-workstation/src/k2-horizon-inference/results/authoritative-prefix/native-fullgraphs-sm120-fa4-uniform4d-large-c4-fixed16384.json`.
- [ ] Qualify native FA4, `b12x`, and any SGLang-derived SM120 adapter independently before promoting one as a default.
- [ ] Add generic proposer-execution policy to the maintained vLLM fork while preserving eager fallback and existing speculative methods.
- [ ] Prove a dedicated shape-safe compiled proposer path on the RTX PRO without sharing target graph input buffers.
- [x] (2026-09-19 10:45Z) Ran the total-width-eight GPU oracle and rejected
  separate root commitment after it collapsed autoregressive acceptance. The
  retained complete candidate block passed c4 throughput and repeatability;
  target logprobs, KV advancement, cancellation, and batch churn remain part of
  the release gate below rather than evidence for the rejected execution model.
- [ ] Prove emitted tokens, target logprobs, KV advancement, cancellation,
  batch churn, and policy-LoRA version fencing on the retained Stage 6b
  complete-block topology before release.
- [ ] Add versioned diagnostic bindings and one matched warm workload covering FA2/FA4, eager/compiled proposer, native/TurboQuant KV cache, and ordinary/speculative decoding.
- [ ] Run CPU tests, fork regression tests, then bounded GPU correctness and performance gates; retain full resolved configuration and raw counters.
- [ ] Publish a new immutable vLLM fork release and advance Posttrain only after the fork commit and artifacts exist.

## Surprises & Discoveries

- Observation: a correct source commit can still publish the wrong Python
  version when the fork does not expose the relevant release tags to
  `setuptools-scm`.
  Evidence: the first dev1 Git dependency resolved as
  `0.26.1rc1.dev2404+g2cbbd413d` even though its source base was v0.29.1rc0.
  Mirroring the upstream ancestry tag was insufficient because vLLM's custom
  describe command excluded CarbonTeq tags. Dev2 explicitly admits both
  upstream and `carbonteq-v*` tags and resolves as `0.29.1.dev2`.

- Observation: the fastest current native c4 path uses fewer graph layers, not
  more compilation.
  Evidence: Stage 1 and Stage 6b both report `CompilationMode.NONE`, but Stage 1
  retained `FULL_AND_PIECEWISE` and profiled 88 PIECEWISE, 62 FULL, and 10
  proposer descriptors. Stage 6b normalized to `FULL_DECODE_ONLY`, captured 62
  FULL target graphs, and contains no live Uno graph qualification. Its proposer
  therefore ran eager while aggregate throughput rose from 422.08 to 526.41
  output tokens/s. The next A/B must isolate fragmentation from the simultaneous
  acceptance change.

- Observation: Uno's clean root is target-distributed but cannot be removed
  from the target candidate block in a one-pass verifier.
  Evidence: the target must consume candidate position zero before its logits
  represent the distribution for candidate position one. Separately committing
  the root shifted all later conditionals, reduced acceptance to 3.29%, and
  delivered only 196.21 output tokens/s. Restoring `[root + seven drafts]`
  restored all eight acceptance positions and roughly 50% acceptance even
  before the fast execution topology was restored.

- Observation: Stage 6b improved both time per speculative cycle and the number
  of cycles required, so throughput alone cannot attribute the cause.
  Evidence: its mean acceptance was 54.64% versus Stage 1's 48.45%, and it
  proposed 12,232 candidates across two repeats versus Stage 1's 13,488, a
  9.31% reduction for equal output. Its repeat throughput differed by only
  0.65%, but a factorial graph-topology experiment remains necessary.

- Observation: Stage 6b closes most of the native-versus-standalone gap at high
  concurrency without approaching KV exhaustion, but output determinism still
  depends on execution shape.
  Evidence: same-process means were 1,201.27 output tokens/s at c16 and
  1,490.23 at c32, 6.82% above the earlier native path and 2.52%/2.32% below
  standalone. Peak KV occupancy was only 24.07%/49.36%, and no request was
  preempted. Yet four of sixteen c16 responses and two of thirty-two c32
  responses changed hashes between repeats; c8 changed three of eight.

- Correction (2026-09-19 07:15Z): the live replay rejection recorded at 01:17Z was a qualification-harness error, not model corruption. Speculative dispatch pads total width eight to nine executed rows per request; the extra row is refreshed for graph safety but is not a semantic model output. Comparing all 9/18/27/36 rows reported exactly 1/2/3/4 padding-token mismatches and large padding-only hidden/logit deltas. Restricting the oracle to the actual 8/16/24/32 rows made eager-repeat, initial replay, and changed-input replay bit-exact for every live batch shape. The earlier fallback measurements remain valid performance controls, but their causal conclusion is superseded.
- Observation: correct native proposer graphs currently recover only a modest fraction of the standalone performance gap.
  Evidence: the first warm medium c4 fixed-128 run delivered 192.99 output tokens/s at 46.14% acceptance versus the prior fail-closed eager diagnostic near 174.75 output tokens/s. The larger fixed-1,024 run improved from 142.67 to 153.21 output tokens/s (+7.39%) while acceptance increased from 29.13% to 38.19%. The matched fixed-16K FA2 control reached only 152.97 output tokens/s. Breakable replay still executes 37 graph segments and 36 Python-driven eager attention boundaries per proposal. The next controlled composition must select SM120 FA4 before attributing the remaining gap to graph orchestration; after that, graph-safe/fused attention execution or a lower-overhead captured execution plan is the likely high-leverage target.

- Observation: preserving the uniform speculative block through the SM120 adapter unlocks graph-safe attention, but the current stochastic headline does not prove a speedup over optimized standalone Uno.
  Evidence: the 4-D/full-graph path passed exact semantic replay for c1-c4 and its first fixed-16K stochastic run reached 345.75 output tokens/s. However, acceptance was 55.77%, versus 28.66% in the older 335.25 standalone run, and draft-token processing remained approximately 506 per second versus approximately 511 per second in the 283.82 native control. The correct optimized standalone target is the repeatable deterministic 4K mean of 474.67 output tokens/s, not 335.25. Fixed-noise matched-budget repeats are required to isolate execution improvement.

- Observation: the first native deterministic-noise diagnostic was not sampling-equivalent to the standalone 474.67 control.
  Evidence: native produced 257.21 output tokens/s at 30.44% acceptance, but its request harness still sent `temperature=1.0` and `top_p=0.95`, while the standalone deterministic control used greedy target sampling and a prompt-derived noise hash with salt 20260918, reaching 64.95% acceptance. The native counter-hash diagnostic is useful only for A/B tests within native vLLM; it is not a standalone parity result.
- Historical correction (2026-09-19): earlier entries had overstated the elimination of the qualification harness and blamed position-gated Punica before import-time breakable capture was active. That caution led to the saved-batch audit above; the later actual-token oracle supersedes the earlier causal claims while retaining their measurements as historical controls.
- Observation: serving accepts `flash_attn_version`, but colocated TRL rollout translation drops it.
  Evidence: `VllmEngineConfig.as_vllm_kwargs()` emits `attention_config`, while `packages/train/src/posttrain/train/backends/trl/common.py::vllm_rollout_options` forwards no attention selection.

- Observation: current upstream vLLM already enables prefix caching by default, while Posttrain's standalone serving dataclass defaults it to false.
  Evidence: released vLLM `CacheConfig.enable_prefix_caching` is true; `VllmEngineConfig.enable_prefix_caching` is false. Every retained run must therefore record the resolved value rather than infer it from absence.

- Observation: vLLM already has a proposer-specific CUDA-graph dispatcher and persistent proposer buffers, but Uno forces `SpeculativeConfig.enforce_eager = True` after an earlier shared-runner replay produced stale inputs.
  Evidence: `SpecDecodeBaseProposer.initialize_cudagraph_keys` owns PIECEWISE graph dispatch, while the Uno branch overwrites `enforce_eager` in `SpeculativeConfig.__post_init__`.

- Observation: the old v0.25.1 TurboQuant reshape patch is not a patch that should be copied mechanically onto the v0.26.1 line.
  Evidence: the new base has a reorganized TurboQuant backend and no longer contains `TQFullAttentionSpec` at the old seam. Compatibility must be asserted behaviorally through cache allocation, generation, and recall gates.

- Observation: the retained native-vLLM source and environment contain the FA4 CuTe interface, but native FA4 serving is not implemented for SM120 paged KV.
  Evidence: after temporarily admitting compute capability 12.0, launch first exposed SM120's single-split requirement; after resolving that to one split, the bundled kernel failed explicitly with `Paged KV not supported on SM 12.0 in this PR`. The earlier standalone FA4 speedup therefore cannot be transferred to vLLM by selecting a flag.

- Observation: selecting a LoRA-aware proposer graph and matching capture-time Uno position mappings does not repair native proposer replay.
  Evidence: the first corrected graph attempt accepted 15 of 4,333 draft tokens (0.35%) at 74.10 output tokens/s; the position-matched capture accepted 10 of 4,354 (0.23%) at 73.60 output tokens/s. The eager proposer accepted 507 of 1,071 (47.34%) and delivered 178.87 output tokens/s on the matched workload. The graph-state defect is deeper than LoRA cardinality or capture mapping.

- Observation: maximizing aggregate draft acceptance does not maximize Uno throughput.
  Evidence: on identical warm prompts of 1,782, 7,062, 14,102, and 3,564 tokens at concurrency four with fixed 128-token outputs, K=3 accepted 79.19% but delivered 162.70 output tokens/s; K=5 accepted 61.63% at 173.84; K=7 accepted 47.34% at 178.87.

- Observation: current upstream has already generalized more of the SM120 problem than the released CarbonTeq pin, but not as one universal FA4 path.
  Evidence: upstream `44dd18fe0` contains `vllm/v1/attention/backends/b12x.py` for paged causal SM120/SM121 attention and newer FA4 selection logic for supported Blackwell shapes. The released CarbonTeq source overlay is still based on `75c71390d`; the feature branch is four commits ahead and seventy commits behind refreshed upstream main.

- Observation: SGLang v0.5.19 already contains a real SM120 paged-KV and SplitKV path, but its kernel cannot safely be reduced to seven files layered over the public `flash-attn-4` wheel.
  Evidence: `Sm120PagedKVManager` consumes a page table and paged K/V tensors, while the transitive CuTe support layer contains SGLang changes for SplitKV, batch invariance, launch-plan caching, and SM120 scheduling. A source comparison against `flash-attn-4==4.0.0b31` found material differences across the shared modules. The reusable boundary is therefore a small standalone kernel distribution containing that qualified closure, followed by a vLLM adapter; it is not a full SGLang dependency and not an import substitution.

- Observation: the vLLM community has several adjacent SM120 FA4 efforts, but no complete equivalent of the extracted kernel path.
  Evidence: closed PR #40110 provides a useful eager backend skeleton but no CUDA graphs and a different kernel bundle; open PR #47218 only widens the FA4 capability gate; open PR #48156 adds warmup for supported dense/paged paths; issue #51405 still records the native SM120 paged-KV gap. No public vLLM branch or pull request found in the audit carries SGLang's SM120 paged loader and SplitKV scheduler.

- Observation: the extracted kernel works with vLLM's CUTLASS 4.7.1 stack and is numerically sound, but the first vLLM integration is only at performance parity with FA2.
  Evidence: a direct BF16 paged-GQA smoke on SM120 compiled and replayed successfully with 0.004304 maximum and 0.000552 mean absolute error against a PyTorch reference. With identical K2 chat rendering, prompts of 8,822, 17,622, 26,422, and 30,822 tokens, concurrency four, and fixed 128-token outputs, `SM120_FA4` produced 66.57 aggregate output tokens/s and FA2 produced 66.04. The 0.80% delta is too small to promote; cached-plan hit/miss telemetry and stage-level profiling are the next gates.

- Observation: current upstream chooses `b12x` correctly on SM120, but its preferred 128-token KV block size was resolved only after model construction.
  Evidence: the first K2 launch selected `B12X` and then failed in `B12xPagedAttentionImpl` because `cache_config.block_size` was still 16. Resolving the selected backend's preferred block size in the attention selector allowed the same model to load, capture CUDA graphs, and serve with 128-token pages.

- Observation: forcing FA4 on SM120 failed late instead of falling back.
  Evidence: upstream logged that FA4 supports compute capabilities 9.x, 10.x, or 11.x, returned version `None`, loaded the model, and asserted during GPU profiling. The candidate now warns once and falls back to FA2 before model construction; the same launch reaches a healthy server.

- Observation: `b12x` is measurably faster than FA2 for K2 on the RTX PRO, but the current package cannot be promoted unchanged.
  Evidence: on fresh servers with identical 8K-30K prompts, Torch sampling, concurrency four, and fixed outputs, `b12x` measured 168.33 versus 165.89 output tokens/s at 1,024 tokens and 183.64 versus 178.89 output tokens/s at 16,384 tokens. However, `b12x==1.3.0` pins `nvidia-cutlass-dsl==4.6.2`, while current vLLM requires `==4.7.1` and quack-kernels requires `>=4.7`; `uv pip check` therefore reports two incompatibilities.

- Observation: composing Uno with the extracted target kernel closes most, but not all, of the gap to the retained standalone runtime.
  Evidence: on a fresh server with identical large prompts, concurrency four, fixed 16,384-token outputs per request, and zero prefix-cache hits, current vLLM produced 65,536 output tokens in 226.89 seconds (288.84 output tokens/s). That is 61.5% above current FA2 K2 at 178.89 output tokens/s and 13.8% below the retained standalone Uno result of 335.25 output tokens/s. A preceding 286.73 output-token/s run had 171,120 prefix-cache hits and is explicitly excluded from the clean comparison.

- Observation: the current Gemma Triton control reproduces, but the extracted SM120 adapter is not yet semantically valid for Gemma.
  Evidence: current vLLM plus Triton generated 65,536 tokens at concurrency four in 341.28 seconds (192.03 output tokens/s), within 0.18% of the retained 192.37 result and with zero prefix-cache hits. A direct BF16 paged-decode smoke for the experimental head-dimension-512 kernel was finite (0.00440 maximum absolute error), yet full Gemma serving fell to 8.15 output tokens/s and emitted repetitive/degraded text. Restricting SM120 FA4 to Gemma's head-dimension-256 sliding-window layers while routing its head-dimension-512 full-attention layers to Triton still produced degraded text at 8.49 output tokens/s. The defect therefore includes unqualified Gemma sliding-window/GQA semantics and cannot be inferred away from a one-token numerical smoke.

- Observation: Gemma 4 paired-assistant MTP is a strong existing proposer path and a useful generic-composer regression target.
  Evidence: with current vLLM, Triton attention, compiled target execution, native Gemma Q-only KV sharing, CUDA-graph-enabled speculation, four independent 8K-30K prompts, 4,096 fixed outputs each, and zero prefix-cache hits, MTP-2 delivered 253.93 output tokens/s versus 181.43 without MTP. It accepted 98.30% of proposed tokens overall. Composer integration must preserve this specialized execution and beat or match the retained result; merely routing it through a common interface is not a performance improvement.

- Observation: graph ownership and complete padded-row refresh are necessary but not sufficient for Uno replay; LoRA slot lifetime is a separate capture contract.
  Evidence: the dedicated-key implementation eliminated the earlier live lazy-capture failure and a Triton-backed graph server survived batch churn at request counts 1, 4, 2, 3, and 4. However, capturing with a zero LoRA mapping yielded only 504 accepted tokens from 2,779 drafts (18.1%), while attempts to capture with the real position-gated adapter encountered an illegal address as vLLM's target dummy-LoRA cache activated and recycled adapter slots. Pre-activating the adapter did not remove that capture-time fault. The retained eager control remains the correctness authority; no proposer-graph result is promotable.

- Observation: SM120 attention graph qualification remains independent from proposer graph qualification.
  Evidence: the extracted backend still declares `AttentionCGSupport.NEVER`, so vLLM correctly limits it to eager attention inside PIECEWISE model graphs. A synchronous diagnostic launch could finish startup capture, but the first graph replay raised a device-side assertion. Triton isolated the same Uno/LoRA lifecycle issue, confirming that attention support must not be inferred from proposer progress and that the two graph gates require separate output-oracle tests.

- Observation: the failed real-adapter capture is not one stale-input bug; it crosses capture scheduling, slot identity, and routing-metadata ownership.
  Evidence: `GPUModelRunner._dummy_run()` invokes the drafter while the target is still inside `maybe_dummy_run_with_lora()`. `LoRAModelManager` exposes one mutable `lora_index_to_id` slot table, while every LoRA layer references one shared Punica wrapper whose token, sampler, embedding, and Triton kernel metadata tensors are overwritten by `set_adapter_mapping()`. Upstream issue #28334 documents the same unsafe class of draft-graph replay during target LoRA graph specialization and retains only a mitigation. A proposer-owned token buffer cannot make this lifecycle safe on its own.

- Observation: after separating capture ownership and reserving the Uno adapter slot, the remaining corruption is specifically the captured position-gated Punica contribution, not explicit proposer inputs, KV metadata, or replay-only state.
  Evidence: two identical eager dummy forwards were bit-identical. A graph-shaped forward with LoRA routing disabled passed exact hidden-state, logit, proposal-token, and chosen-token-logprob comparison for batch sizes 1-4. Re-enabling the real seed/base plus noise/Uno mapping made the capture forward itself differ before replay: 24 of 28 proposal tokens changed at batch four, with 7.65625 maximum logit error and 0.4375 maximum chosen-token-logprob error. Pointer diagnostics found stable explicit model arguments. The unsafe boundary is therefore position-gated Punica inside PIECEWISE capture; graph opt-in now downgrades to eager during startup instead of serving corrupted proposals or aborting the server.

- Observation: startup dummy attention metadata was a false qualification environment, while capture-time weak graph-pool outputs were a false oracle; correcting both still rejects live Uno replay.
  Evidence: deferring capture to real KV-backed batches restored the eager system-overlay path to 40-44% draft acceptance across long prompts and batch-shape churn, proving that the earlier sub-1% result came from attention-free dummy capture. A live qualifier then restored all touched KV blocks around each comparison and ignored the wrapper's intentionally weak capture output. First replay still diverged from eager for request counts 1-4, with mean hidden-state error 0.115-0.133 and maximum error 6.56-7.09. The descriptor-scoped fallback completed serving at 174.75 output tokens/s without an engine error; the remaining defect is therefore inside graph replay state, not the qualification harness, dummy KV metadata, or fallback lifecycle.

- Observation: mask-noise configuration can fail asynchronously when the documented mask ID is an exclusive vocabulary bound rather than an embedded token.
  Evidence: K2's mask value 250624 was passed to an embedding with valid IDs 0-250623 and triggered a device-side assertion on the first request. Config validation now rejects that combination before GPU execution and directs this checkpoint to `random_uniform` noise.

- Observation: the retained standalone and native Uno width labels did not describe the same speculative work.
  Evidence: standalone block width eight contains one adapter-disabled clean root plus seven future Uno candidates. Native `num_speculative_tokens=7` previously produced one clean root plus only six future candidates, then sent all seven through ordinary rejection sampling. The retained 16K counters therefore compare 7.32 outputs per standalone cycle with 4.86 per native cycle; they are not a pure engine-cost comparison.

- Observation: Uno's clean root can be target-authoritative without weakening policy ownership, but only under an explicit sampling-state contract.
  Evidence: position-gated routing disables Uno on the seed row and retains the request policy LoRA in composite mode. The new path samples that row with vLLM's target `Sampler` and retains its target logprob. It refuses the optimization when request-history-dependent processors have not been advanced through the tokens emitted earlier in the same engine step.

- Observation: RTX capacity was idle, but direct host SSH and immediate source-overlay retry were not usable in this session.
  Evidence: all retained direct SSH identities were rejected. The dstack task reached the RTX worker, but its shell preflight failed before Python; after abort, the worker remained in its bounded teardown interval and the corrected retry received `no offers`. This is an access/orchestration boundary, not evidence about Uno correctness or speed.

## Decision Log

- Decision: Release the qualified Uno surface as inference plus native LoRA,
  and reject full-weight/QLoRA Uno jobs before submission.
  Rationale: the real LoRA optimizer gate has changed-weight, finite-logprob,
  and policy-version evidence. Full-weight and QLoRA do not; retaining their
  configuration paths would turn missing evidence into accidental support.
  Date/Author: 2026-09-20 / Codex

- Decision: Do not amend the frozen product baseline.
  Rationale: `InferenceBinding.engine` already owns kernel, scheduler, speculative, and KV-cache choices; `Workload` already owns warmup and benchmark population.
  Date/Author: 2026-09-18 / Codex

- Decision: Represent attention selection as backend-native typed engine configuration, not a model capability or new catalog family.
  Rationale: the valid attention backend depends on model architecture, cache format, vLLM build, and hardware target, while the model weights remain unchanged.
  Date/Author: 2026-09-18 / Codex

- Decision: Posttrain's resolved-job compiler rejects deterministic incompatibilities before submission; low-level serve and train translators remain policy-free.
  Rationale: the selected backend revision already states that TurboQuant requires FlashAttention 2, and the pinned Nanbeige runtime already establishes that DSpark's non-causal draft attention cannot use TurboQuant. These are compilation rules over immutable selections, not host readiness probes and not reasons to spend GPU allocation on runtime discovery. vLLM retains runtime validation as a backstop for facts unavailable before submission.
  Date/Author: 2026-09-18 / Codex

- Decision: Keep proposer compilation fail-closed with an explicit eager fallback.
  Rationale: a faster graph that reads stale token, position, LoRA, or KV metadata is a correctness failure. Promotion requires matching target tokens/logprobs and graph-specific buffer isolation under batch-shape churn.
  Date/Author: 2026-09-18 / Codex

- Decision: Treat TurboQuant as one generic KV-cache choice and qualify compositions, rather than merging an old maintenance branch wholesale.
  Rationale: the current vLLM base has substantially different native TurboQuant infrastructure; only still-missing behavior should become maintained fork delta.
  Date/Author: 2026-09-18 / Codex

- Decision: Prefer current upstream's native FA4 and `b12x` backend contracts; treat SGLang SM120 reuse as a bounded experimental backend.
  Rationale: this minimizes permanent fork delta and preserves paged KV, continuous batching, prefix caching, CUDA graphs, and automatic fallback. The SGLang kernel is promoted only if a matched benchmark shows a material gap and its adapter passes the same correctness matrix.
  Date/Author: 2026-09-18 / Codex

- Decision: Package the SGLang-derived SM120 kernel as an independent distribution rather than vendoring SGLang into vLLM or importing vLLM internals from the kernel.
  Rationale: the kernel has a coherent, license-compatible source closure, while SGLang's serving runtime is irrelevant to vLLM. A framework-neutral package gives vLLM a small stable call surface, keeps backend selection and KV ownership in vLLM, and permits the same kernel to be tested independently. The adapter stays opt-in until CUTLASS 4.7.1 compatibility, CUDA-graph behavior, and matched correctness/performance are proven.
  Date/Author: 2026-09-18 / Codex

- Decision: Do not rebase the dirty `/home/hammad/projects/vllm-uno` worktree in place.
  Rationale: it contains an uncommitted clarification to proposer safety. The current-upstream composition will be built in a separate named active worktree, and consolidation will occur only after its tests pass.
  Date/Author: 2026-09-18 / Codex

- Decision: Keep Gemma on Triton and do not advertise SM120 FA4 eligibility for it.
  Rationale: both the direct HD512 extension and the safer per-KV-kind hybrid failed an inexpensive 128-token correctness/performance probe. Passing an isolated paged-decode oracle is necessary but insufficient for full-model attention semantics. Promotion now requires layer-aware reference tests covering Gemma's sliding-window masks, GQA layout, soft-cap behavior, mixed attention kinds, prefill, and decode before another long benchmark.

- Decision: Retain eager Uno proposal with full target decode CUDA graphs as the
  current experimental c4 operating point; keep independent proposer graphs as
  a qualification candidate, not an assumed optimization.
  Rationale: the 526.41-token/s Stage 6b path did not replay live proposer
  graphs, while the breakable FULL_AND_PIECEWISE topology was slower. A future
  proposer graph must capture the proposer as one owned execution unit without
  reintroducing target-model fragmentation. Request/policy-LoRA composition
  remains functional-first and eager until a two-adapter capture passes the
  same changed-input and policy-generation gates.
  Date/Author: 2026-09-19 / Codex

- Superseded decision: Keep Uno proposer CUDA graphs disabled and retain only fail-closed ownership scaffolding until LoRA-aware capture is redesigned.
  Superseded by the 2026-09-19 decision above. The safety rationale remains
  valid, but correct draft-only live graph qualification was later achieved;
  performance evidence now shows that graph boundary and fragmentation, not
  merely graph availability, determine whether replay is worthwhile.
  Date/Author: 2026-09-18 / Codex

- Decision: Extend the existing speculative proposer hierarchy with a generic execution context; do not create a second block-proposer framework.
  Rationale: Uno, MTP, EAGLE, DFlash, and draft-model proposers already share scheduling and verification machinery but have different algorithm state. A generic execution context should own graph policy, capture descriptors, persistent-state ownership, LoRA metadata-bank identity, fallback reason, and policy version. Each proposer continues to own only candidate preparation and model-specific state.
  Date/Author: 2026-09-18 / Codex

- Decision: Treat Gemma MTP as a preservation-and-tuning consumer of the proposer composition layer, not as code to rewrite into the Uno implementation.
  Rationale: Gemma's existing Q-only assistant, target-KV sharing, fused multi-step decode, and speculator-owned CUDA graphs already provide a measured 39.96% throughput uplift. The generic layer may add lifecycle ownership, qualification, observability, version fencing, and dynamic draft-width policy, but promotion requires no regression against the retained MTP-2 baseline and exact target-output equivalence.
  Date/Author: 2026-09-18 / Codex

- Decision: Separate proposer graph capture from target graph capture and reserve system-adapter slots before adding metadata banks.
  Rationale: this removes the known nested-capture hazard and makes adapter-slot meaning stable before attempting replay. Owner-scoped Punica banks then remove cross-owner routing mutation. The changes are independently testable and preserve eager fallback throughout.
  Date/Author: 2026-09-18 / Codex

- Decision: Keep Uno's clean root at candidate position zero in the complete
  eight-token target-verification block.
  Rationale: the root is target-distributed because Uno is disabled on that row,
  but a one-pass autoregressive verifier must consume it before producing logits
  for the later drafts. Scheduler candidate count is eight, future-draft count
  is seven, and maximum committed output is nine including the target
  correction/lookahead token. Metrics follow scheduler candidates.
  Date/Author: 2026-09-19 / Codex

- Superseded decision: Represent a target-authoritative proposer prefix separately from speculative candidates.
  Superseded by the 2026-09-19 complete-block decision. The attempted execution
  violated autoregressive alignment and collapsed acceptance; the old helpers
  and misleading option name should be removed before release.
  Date/Author: 2026-09-18 / Codex

## Outcomes & Retrospective

The release-clean implementation is published as CarbonTeq vLLM
`carbonteq-v0.29.1.dev2` and `sm120-paged-attention` v0.1.0. Posttrain selects
the exact fork commit, pairs it with the binary wheel from the same upstream
base, and rejects unqualified Uno update modes at compilation. Stable
promotion work remains in progress. The strongest diagnostic native c4
checkpoint is
Stage 6b: 526.41 mean aggregate output tokens/s across two matched warm repeats,
with 54.64% mean acceptance, identical response hashes, and zero preemptions.
It uses SM120 FA4, a full eight-candidate Uno block, full target decode CUDA
graphs, and an eager proposer. It is 24.72% above Stage 1, but the gain combines
removal of PIECEWISE graph fragmentation with a 6.19-point acceptance increase;
causal attribution and long-output/fresh-process qualification remain open.

Serving and TRL now preserve the same attention,
cache, and scheduler settings. Known-invalid vLLM flag compositions fail in the
resolved-job compiler before submission, with 127 focused tests passing. Direct
RTX PRO experiments retained K=7 plus eager Uno proposal as the current best
safe operating point. The current-upstream experiment also proves generic
ordered backend selection, early preferred-block-size resolution, and safe FA4
fallback. `b12x` is 1.47-2.65% faster than FA2 in the first matched K2 tests,
but remains experimental until its CUTLASS dependency converges and broader
correctness/concurrency qualification passes. The SGLang-derived SM120 kernel
now has an independent package and an opt-in vLLM adapter. Direct paged-KV
correctness and CUTLASS 4.7.1 compatibility are proven, and end-to-end K2
serving is healthy. Its first matched long-prompt result is only 0.80% above
FA2, so it remains opt-in while cached-plan telemetry, CUDA-graph behavior,
broader shapes, and fallback receive direct qualification.
The first separate authoritative-prefix execution has been rejected. The
correct width-eight contract keeps the clean root followed by seven future
drafts in one target candidate block. The next release boundary is a real RTX
PRO policy-LoRA optimizer-step refresh on the Stage 6b topology, followed by
fresh target logprobs, policy-version fencing, stable KV progression, abort,
drain, and sleep/wake qualification. No fork release or Posttrain dependency
pin moves until those gates pass.
When combined with eager K=7 Uno on a clean 16K-output workload, the same
target backend reaches 288.84 output tokens/s: materially faster than ordinary
K2 and Gemma, though still 13.8% below the retained standalone Uno runtime.
Current-vLLM Gemma reproduces its Triton baseline at 192.03 output tokens/s.
The attempted Gemma HD512 and mixed-backend SM120 paths both corrupt output and
run below 8.5 output tokens/s, so their long runs were intentionally skipped
and HD512 production eligibility was reverted.
Independent proposer CUDA replay remains opt-in evidence rather than the
retained performance path. Unqualified SM120 FA4 model/shape combinations
remain rejected rather than being exposed as defaults.

## Context and Orientation

`/home/hammad/projects/rl` is the Posttrain framework. Reusable vLLM serving
translation lives in `packages/serve/src/posttrain/serve/backends/vllm` and
typed vLLM settings live in `packages/serve/src/posttrain/serve/profiles`.
Colocated TRL rollout translation lives in
`packages/train/src/posttrain/train/backends/trl/common.py`. Project-specific
qualification bindings live in `apps/lab/.posttrain/catalog`.

`/home/hammad/projects/vllm-uno` is the maintained CarbonTeq vLLM worktree.
`vllm/config/speculative.py` owns speculative configuration,
`vllm/v1/spec_decode/llm_base_proposer.py` owns generic proposer buffers and
graph dispatch, `vllm/v1/spec_decode/uno.py` owns only Uno's algorithm-specific
input and adapter behavior, and `vllm/v1/worker/gpu_model_runner.py` coordinates
target and proposer execution.

`/home/hammad/projects/sm120-paged-attention` is the framework-neutral kernel
distribution extracted from SGLang v0.5.19. `/home/hammad/projects/vllm-sm120`
contains the current-upstream candidate and its opt-in `SM120_FA4` adapter. The
kernel package owns CuTe compilation and launch policy; vLLM owns cache storage,
metadata, scheduling, backend eligibility, and fallback.

FlashAttention 4 (FA4) is an attention-kernel implementation optimized for
supported recent NVIDIA GPUs. CUDA graphs replay a previously captured GPU
execution sequence to reduce launch overhead; replay is correct only if every
input buffer and metadata pointer has stable ownership. TurboQuant compresses
the runtime KV cache and changes attention-backend compatibility without
changing model weights. A warm benchmark excludes model loading, compilation,
graph capture, and cache construction from the measured interval.

## Plan of Work

First, strengthen Posttrain's typed vLLM configuration. Validate the syntax of
attention versions 2, 3, and 4, translate them identically into serving and
colocated TRL, and record explicit prefix-cache policy. Preserve requested
settings in the backend translators, then apply deterministic cross-selection
policy in the resolved Posttrain job compiler. Explicit TurboQuant plus
FlashAttention 3 or 4 and the pinned DSpark plus TurboQuant composition fail
before packing or provider submission. The worker runtime still owns dynamic
checks that cannot be known from immutable job inputs.

Second, refactor the vLLM fork so speculative proposer execution policy is
generic. The policy must distinguish eager, automatic, and independently
captured PIECEWISE execution. Uno must not mutate a shared configuration field
behind the caller's back. Instead, the generic proposer reports whether its
inputs and LoRA mapping are graph-safe, and the runner either dispatches its
dedicated graph or records the eager fallback reason.

Third, introduce a generic `ProposerExecutionContext` under the existing
speculative proposer hierarchy. It identifies the execution owner, eager or
independent-capture policy, structural graph key, LoRA metadata bank, and
policy-weight generation. `GPUModelRunner` captures independent proposers only
after all target graph captures and outside `maybe_dummy_run_with_lora()`.
Existing proposers retain nested capture until explicitly migrated, so the
change is compatible and reviewable.

Fourth, formalize system LoRA ownership. Uno and future runtime-owned adapters
receive pinned reserved slots that are not counted as request/dummy capacity,
not removed by ordinary capture cleanup, and never reassigned for the engine
lifetime. Target dummy LoRAs may occupy only request slots. A graph key contains
the structural slot topology and rank/dtype layout, but not the policy version:
same-shaped RL updates copy weights into the same stable tensors and advance a
separate generation fence.

Fifth, give each execution owner an independent Punica metadata bank. A bank
contains token, sampler, padded-sampler, embedding, prefill, and Triton kernel
metadata tensors. Target and proposer graphs may share immutable model and
stacked LoRA weight storage, but never routing buffers. The active bank is
selected by the execution context before staging and forward execution; an
uncaptured owner/shape falls back to eager rather than capturing in a request.

Sixth, make Uno the first consumer. Its graph reads only proposer-owned input
IDs, positions, noise, slot mappings, sequence/block metadata, and the Uno
Punica bank. Deterministic mask-noise tests compare eager and replay proposal
tokens, logits, and logprobs exactly or within dtype tolerance across batch
churn. Only after draft-only Uno passes does policy-plus-Uno composition gain
two stable slots and in-place policy/composite refresh with drain, synchronize,
generation increment, cache invalidation where required, and resume.

Seventh, migrate one existing proposer, preferably MTP or EAGLE, through the
same execution context to prove the seam is generic. Dynamic draft width and
unified metrics follow only after two methods pass; position gating remains an
Uno implementation detail rather than part of the generic contract.

Fourth, create diagnostic inference bindings rather than changing the released
RL binding in place. The matrix will compare FA2 and FA4 on supported
Blackwell targets, eager and compiled proposers, native and TurboQuant KV cache
where the model/backend pair supports it, and ordinary versus speculative
decoding. One shared workload will retain representative warmup, independent
large prompts, concurrency four, and a 16,384-token output ceiling.

Before that matrix, rebase the maintained CarbonTeq delta onto a named current
upstream candidate in a separate worktree. Retain upstream's FA4 and `b12x`
implementations unchanged first. Add a small capability resolver and structured
selection/fallback report around the existing attention backend interface. Keep
the extracted SM120 kernel package optional. Its adapter must consume vLLM's
existing page table and cache tensors, first in eager mode; CUDA-graph promotion
requires stable compiled launch plans and workspaces under vLLM capture.

Finally, qualify from cheapest to most expensive. Static and unit tests precede
GPU smoke. GPU correctness precedes throughput. Only a configuration that
passes finite target-logprob, tool-call, abort/drain, sleep/wake, cache
invalidation, and mixed-batch churn gates may be compared for speed. Publish
the fork first, then advance immutable Posttrain pins, locks, images, ledgers,
and release evidence.

## Concrete Steps

From `/home/hammad/projects/rl`, run focused contract tests during the first
milestone:

    uv run pytest packages/serve/tests/test_vllm_bindings.py packages/train/tests/test_trl_common.py packages/train/tests/test_api.py
    uv run ruff check packages/serve packages/train apps/lab
    uv run pyright
    uv run lint-imports
    git diff --check

From `/home/hammad/projects/vllm-uno`, run fork tests after each proposer change:

    pytest -q tests/config/test_uno_speculative_config.py tests/v1/spec_decode/test_uno.py
    ruff check vllm/config/speculative.py vllm/v1/spec_decode/uno.py vllm/v1/spec_decode/llm_base_proposer.py vllm/v1/worker/gpu_model_runner.py tests/config/test_uno_speculative_config.py tests/v1/spec_decode/test_uno.py
    git diff --check

The final locked Posttrain ladder is:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

## Validation and Acceptance

Configuration acceptance requires the same binding to produce equivalent
vLLM attention, prefix-cache, scheduler, cache-format, and speculative settings
through serving and colocated TRL. Structurally malformed values and
deterministic cross-flag incompatibilities fail during detached job
compilation. These checks are not bypassed by `--skip-preflight`. Runtime-only
incompatibilities remain typed worker failures, but no known-invalid flag
combination may reach provider submission.

Graph acceptance is layered. Unit tests first prove that target and proposer
metadata banks do not alias, reserved system slots survive dummy/request LRU
churn, capture is ordered after target capture, structural keys are stable, and
unknown shapes select eager. A GPU micro-harness then alternates target and
proposer mappings without the server and compares eager/replay LoRA layer
outputs. End-to-end deterministic mask-noise runs compare proposal tokens,
proposal logits/logprobs, target logits/logprobs, and accepted prefixes across
request-count churn `1 -> 4 -> 2 -> 3 -> 4`. Random-noise runs additionally
require acceptance-rate parity. Abort, drain, sleep/wake, and same-shaped
policy updates must preserve stable slot addresses while stale policy
generations are rejected. No throughput result is promotable before these
oracles pass.

Performance acceptance requires matched warm inputs, outputs, sampling,
concurrency, target weights, and hardware. The retained artifact reports
resolved attention version, proposer mode and fallback reason, KV-cache dtype,
prefix query/hit tokens, draft/accept counts by position, actual token counts,
TTFT, throughput, peak VRAM, truncations, errors, and preemptions. A faster
configuration that fails correctness or recall is rejected.

## Idempotence and Recovery

All new options remain opt-in until qualified. Existing bindings retain their
meaning. Work occurs on the existing clean feature branches; unrelated
worktrees are not reset or deleted. Failed GPU jobs may remove only their own
temporary processes and job state. Shared model caches and retained tracking
evidence are preserved. Posttrain pins do not move until the fork commit is
published and fetchable.

## Artifacts and Notes

The retained standalone K2 benchmark under
`/home/hammad/projects/k2-horizon-inference/results/README.md` is a hypothesis
and workload source. Standalone compiled-Uno results are not release evidence
for native vLLM. The current vLLM release base is
`44dd18fe0bb0f13157f97a5aa029b6604468fca6`; the selected CarbonTeq release
commit is `fbbba6698b2f8a912b94705cfc09eb4fd7243716`.

The direct native-vLLM experiment artifacts are retained under
`/home/hammad/projects/k2-horizon-inference/results/direct-vllm-platform-20260918/`.
They include the five benchmark JSON files and matching Prometheus snapshots
for eager K=3/K=5/K=7 and the two rejected proposer-graph attempts. All runs
used model revision `586b03f0fd1fbbf2f13eeafc33749e95ae34dd10`, Uno adapter
revision `ec92bbd768f4a404319625204544782e3377bcd7`, FA2, 32,768 context,
concurrency four, independent warm prompts, and fixed 128-token measured
outputs. The worker was returned to 303 MiB used, 96,944 MiB free, and no
compute process after the matrix.

## Interfaces and Dependencies

At the end of the first milestone, both `VllmEngineConfig.as_vllm_kwargs()` and
`vllm_rollout_options()` expose one validated attention configuration and
explicit cache policy. At the end of the fork milestone, speculative proposer
execution has a backend-generic mode and a structured fallback reason; Uno
uses the same interface rather than forcing eager execution internally. At the
end of qualification, versioned catalog bindings select only combinations
proven on their declared execution target.

Revision note (2026-09-18): Created after the Posttrain 0.4.3 release audit
showed that native Uno correctness was released while FA4 selection, compiled
proposal execution, and the retained large-output operating point were not.

Revision note (2026-09-18): Corrected ownership after review: backend adapters
preserve requested FA/cache settings, while the Posttrain resolved-job
compiler rejects known-invalid combinations before packing or submission. The
backend runtime remains only the backstop for dynamic facts unavailable to
compilation.

Revision note (2026-09-18): Added direct RTX PRO source-overlay evidence. It
replaces the earlier incorrect package-closure hypothesis with the observed
SM120 paged-KV kernel boundary, rejects native Uno proposer CUDA replay on
correctness and performance evidence, and retains K=7 as the current warm c4
operating point.

Revision note (2026-09-18): Expanded the plan from the earlier K2-specific FA4
experiment into a general SM120 backend program after refreshing upstream.
Current vLLM already supplies paged `b12x` and newer native FA4 paths, so the
fork will reuse those contracts and evaluate an optional SGLang-derived adapter
only against an observed remaining performance gap.

Revision note (2026-09-18): Replaced the proposed SGLang runtime dependency with
an independent kernel distribution after source-closure and community audits.
Recorded the opt-in vLLM adapter, corrected the earlier dense-cache description,
and kept automatic selection blocked until direct GPU qualification.

Revision note (2026-09-18): Added the target-authoritative-prefix design and
first implementation after cycle-level analysis showed that most of the
standalone/native gap came from different token advancement semantics rather
than raw kernel time. Recorded the block-width mismatch, fail-closed sampling
boundary, CPU validation, and required RTX PRO policy-LoRA gate.

Revision note (2026-09-19): Superseded separate authoritative-root execution
after the GPU experiment proved it shifts autoregressive verification and
collapses acceptance. Recorded the complete eight-candidate Uno contract,
Stage 0-6b history, the 526.41-token/s full-target-graph/eager-proposer
checkpoint, and the factorial attribution plus RL qualification roadmap.
