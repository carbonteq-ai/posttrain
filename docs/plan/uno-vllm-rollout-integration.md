# Integrate Uno Psi-Spec into the maintained vLLM rollout runtime

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current as implementation proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

After this change, a Posttrain inference binding can select Uno Psi-Spec for K2-Horizon rollouts while continuing to use the maintained CarbonTeq vLLM runtime. Uno is a lossless speculative accelerator: the current K2 policy remains the target distribution that accepts or rejects draft tokens, and a separately pinned Uno adapter proposes several future tokens. The rollout result must include target-policy token log probabilities and the exact policy version so GRPO and other on-policy algorithms can train from it safely.

The observable outcome is a warm K2 rollout on the RTX PRO 6000 that uses the same prompt, sampling policy, output budget, concurrency, and current policy weights in both ordinary autoregressive vLLM and Uno-accelerated vLLM. Both paths must produce valid tool-call-capable output and complete log probabilities. The Uno path must report proposal acceptance and improve measured warm throughput without changing the target sampling distribution.

## Progress

- [x] (2026-09-17 15:30Z) Located the prior qualified vLLM revision and created `/home/hammad/projects/vllm-uno` on `codex/uno-spec-decoding` at upstream commit `75c71390d5b399f5397a9166920fc45902f99f14`.
- [x] (2026-09-17 15:30Z) Confirmed that the old CarbonTeq `codex/carbonteq-v0.25.1-fixes` checkout remains untouched with its two unpublished K2/Gemma commits.
- [x] (2026-09-17 15:30Z) Confirmed that the frozen Posttrain baseline already assigns speculative settings to `InferenceBinding.engine`; no product-baseline amendment is required.
- [x] (2026-09-17 15:30Z) Identified the central implementation constraint: the Uno adapter is a position-gated draft overlay, not a normal serving LoRA, and must coexist with a changing target policy.
- [x] (2026-09-17 17:05Z) Added native `method: uno`, typed adapter identity, target-model reuse, scheduler slot accounting, proposer construction, and focused configuration/proposer tests to the CarbonTeq vLLM fork.
- [x] (2026-09-17 17:05Z) Ported the minimal linear Psi-Spec draft path: bounded random noise, compact seed-plus-noise inputs, draft probabilities, vLLM-owned rejection sampling, and native acceptance telemetry.
- [x] (2026-09-18 00:10Z) Added separate full-policy and native LoRA-policy paths. LoRA target rows retain the policy adapter; Uno noise rows use an atomically refreshed rank-concatenated `policy + Uno` adapter.
- [x] (2026-09-17 19:05Z) Preserved target sampling, per-token target log probabilities, abort/drain, sleep/wake, prefix-cache invalidation, numeric policy-version fencing, and exact `BehaviorPolicySpan` projection in the code path. Real changed-weight optimizer gates remain unqualified.
- [x] (2026-09-18 05:50Z) Extended Posttrain's rollout option resolver, K2 model variant, catalogs, cache/speculation metrics, and focused tests for the pinned Uno adapter revision.
- [ ] Run distributional correctness, tool-call, long-prompt, 16K-output, concurrency, warm-throughput, and RL weight-update qualification on the RTX PRO 6000.
- [x] (2026-09-17 15:44Z) Ran the retained short Uno reference smoke on the RTX PRO and audited its public lifecycle interfaces; generation passed but RL admission failed on missing target logprobs, policy-version fencing, weight refresh, abort, and sleep/wake.
- [x] (2026-09-17 17:05Z) Ran the native vLLM Uno path on the RTX PRO: 128/128 and 256/256 finite target logprobs, a parsed K2 tool call, and four concurrent 128-token requests with no preemptions.
- [x] (2026-09-17 17:18Z) Checkpointed the fork implementation and ledger as local commit `8946dee55`; publication and Posttrain pin advancement remain gated on lifecycle qualification.
- [x] (2026-09-17 17:28Z) Enabled loopback-only vLLM lifecycle controls and passed level-1 sleep/wake, post-wake 32/32 target logprobs, single-request abort, and pause/cache-clear/resume on the RTX PRO.
- [x] (2026-09-17 17:31Z) Set and read back the exact base policy revision through vLLM's weight-version register; Posttrain still must attach and fence that value on each rollout because OpenAI completion responses omit it.
- [x] (2026-09-18 00:10Z) Passed the primary live K2 LoRA optimizer gate on RTX PRO: policy version `0` to `1`, finite target logprobs, trainable delta norm `0.1214`, stable tokens, and maximum sampled-logprob movement `0.0655` at learning rate `1e-4`.
- [x] (2026-09-18 01:15Z) Closed the colocated rollout cache-observability gap: synchronous and continuous-async vLLM now retain per-collection prefix query tokens, hit tokens, hit rate, KV peak usage, and speculative counters; Posttrain projects them into provider-neutral `serve/backend/*` metrics.
- [x] (2026-09-18 01:40Z) Cancelled the first VORTEX 20-step qualification before any optimizer update after active sampling repeatedly requested full refills. Removed the native-reward callback indirection at the adaptive trainer boundary and added mixed-reward and incomplete-population regression coverage.
- [x] (2026-09-18 03:35Z) Proved that AutomationBench's structured reward objects were not the refill cause. The environment bridge emits a finite scalar `algorithm_reward`; environments without that field retain TRL's generic single- or multi-reward-function path.
- [x] (2026-09-18 03:40Z) Fixed modern Verifiers multi-turn projection at the trace boundary. The bridge now resolves canonical unsampled assistant/tool continuations back to their exact sampled siblings, preserves sampled token IDs/logprobs, and fails closed on ambiguous graphs.
- [x] (2026-09-18 03:45Z) Proved genuine bounded active sampling live: a 16-row probe reduced its second request to 12 rows, and the final four-row probe admitted all four rows on round two with matching native and TRL group standard deviation `0.1507936567`.
- [x] (2026-09-18 04:04Z) Completed the one-update K2/Uno/VORTEX gate. Trackio run `744d74f0-d963-44c6-8594-18eb2f53810a` admitted four rows after two rounds and finished one LoRA optimizer step with finite loss `0.005787`, gradient norm `0.002367`, and zero native/TRL variance disagreement.
- [x] (2026-09-18 04:08Z) Pushed the generic TRL corrections at `carbonteq-ai/trl@096145f37e64d92f36e664643a1d0e9917fe3cf3`: real MoE detection across five trainers plus chunked GRPO log-probabilities with router auxiliary loss. The focused fork tests pass 2/2 on the RTX host.
- [x] (2026-09-18 05:50Z) Published vLLM `carbonteq-v0.26.1.dev1`, TRL `carbonteq-v1.12.0.post9`, and veRL `carbonteq-v0.9.0.post3` from immutable commits. TRL and veRL retained-asset workflows `35292803386` and `35292934219` passed exact-byte development readback and clean installation.
- [x] (2026-09-18 05:50Z) Advanced the Posttrain root lock to Torch 2.13, TRL post9, and vLLM commit `37706e7d`; regenerated the veRL Python 3.13.12 image lock for post3 and the commit-matched vLLM binary base. The fork ledger and 15 release-definition tests pass.
- [x] (2026-09-18 06:45Z) Rebuilt the hash-locked Torch 2.13/CUDA 13 universal base as isolated candidate digest `sha256:9a20c278e3d87e1bff03352bc74f06c49485eef6bf12afcf68754ce54edbaecf` and passed the real veRL Docker/Bake import smoke against release commit `18338a0e`.
- [x] (2026-09-18 08:10Z) Diagnosed the first protected Posttrain 0.4.3 candidate failure at its two owning layers: generic vLLM kinds attempted a CUDA source build without `CUDA_HOME`, and the isolated transform closure still selected Torch 2.11 against the Torch 2.13 base. Added the verified retained binary overlay to the shared vLLM stage, aligned transform to the exact Torch 2.13 CUDA wheel, and passed direct `serve-smoke` and `transform-smoke` BuildKit builds against the retained candidate base.
- [x] (2026-09-18 08:45Z) Candidate `0.4.3rc1` passed protected image publication, clean index installation, and packed RTX PRO qualification. Promoted TRL post9, veRL post3, and Trackio dev24 unchanged bytes to stable. The final stable preflight then caught one stale Trackio sdist digest in Posttrain's fork receipt; corrected it to the retained release and stable-index digest before allocating a replacement RC.
- [x] (2026-09-18 09:10Z) Replacement candidate `0.4.3rc2` passed with the corrected fork receipt. Its final release replay exposed an ordering error in `build-python-distributions`: strict source validation ran before the accepted materialization was applied. Restricted pending-lock admission to invocations that supply a retained materialization; the staged tree remains strictly verified after projection.
- [ ] Run the bounded GPU qualification, then promote TRL/veRL unchanged bytes and the vLLM source overlay from candidate to stable.

## Surprises & Discoveries

- Observation: The benchmark's disposable latest-vLLM checkout had already been deleted, but its exact revision remains the current `upstream/main` ref in `/home/hammad/projects/vllm`.
  Evidence: `upstream/main` and the benchmark pin both resolve to `75c71390d5b399f5397a9166920fc45902f99f14`; `/home/hammad/projects/k2-horizon-inference/vendor/vllm` no longer exists.

- Observation: The published CarbonTeq fork is still based on vLLM 0.25.1 and is not a safe Uno base.
  Evidence: `/home/hammad/projects/vllm/CARBONTEQ_FORK.md` records upstream base `752a3a504`, while the qualified K2 benchmark records vLLM `0.26.1rc1.dev2329+g75c71390d`.

- Observation: Current vLLM already has an experimental custom proposer seam, native draft probability transport, rejection sampling, async speculative decoding, and draft-weight update hooks. The custom proposer constructor receives only `VllmConfig`, however, so it cannot execute a same-model Uno draft through the target runner without extending the seam or adding a native proposer.
  Evidence: `vllm/v1/spec_decode/custom_class_proposer.py` constructs the proposer with only `vllm_config`; `vllm/v1/worker/gpu_model_runner.py` passes the runner to native model proposers.

- Observation: The released Uno adapter cannot be loaded as a normal vLLM LoRA. Uno applies it only to future draft-noise rows; the causal seed and all target verification rows use the target policy without the Uno overlay.
  Evidence: `nano_vllm_uno/engine/two_pass_decoding.py` builds a LoRA mask of ones for draft positions and explicitly zeros the seed position.

- Observation: The cleanup removed the benchmark's `.venv-vllm`, source checkout, and project-local model cache, so compiled binaries cannot be reattached directly. The benchmark harness, exact revisions, reports, and runtime settings remain.
  Evidence: `/home/hammad/projects/k2-horizon-inference` retains `scripts/`, `benchmarks/`, `results/`, and `vendor/uno`, but has no `.venv-vllm`, `cache/`, or `vendor/vllm`.

- Observation: Posttrain's existing environment can lint and compile the new source, but it cannot execute vLLM's test configuration because the cleaned native vLLM extension and the vLLM test-only `tblib` dependency are absent.
  Evidence: Ruff and `git diff --check` pass; pytest first reports missing `tblib`, and direct import then reports missing `vllm._C_stable_libtorch`.

- Observation: The complete qualified runtime, model snapshots, and CUDA environment survived on the RTX PRO even though the local disposable copies were cleaned.
  Evidence: `/home/dstack/k2-horizon-inference` retains a 6.8 GB vLLM environment, 41 GB project cache, both pinned K2/Uno snapshots, and vLLM source at `75c71390d`; the four focused Uno configuration tests pass there.

- Observation: The released Uno reference runtime passes inference but fails Posttrain's RL lifecycle contract.
  Evidence: A live 256-token smoke completed at 88.33 output tokens/s and 21.53% aggregate draft-token acceptance, but emitted no target logprobs. `LLM`, `LLMEngine`, and `AsyncLLMEngine` expose no abort, sleep/wake, or weight-update operation.

- Observation: Replaying the already-cached target query inside the Uno draft pass is incorrect; the proposer input is only the target-sampled seed token followed by future noise positions.
  Evidence: The full-query prototype produced about 1% acceptance. Compact seed-plus-noise input raised native acceptance to 31.4% in the 128- and 256-token live smokes.

- Observation: The shared target runner's CUDA graph capture aliases target-runner input buffers and cannot currently be reused safely by the differently shaped Uno proposer forward.
  Evidence: With proposer graph replay, the first draft token disagreed with the target next token; keeping the target compiled while running only the proposer eagerly made the first proposal equal the target token and restored acceptance. The configuration now applies this guard automatically.

- Observation: The pinned Uno adapter plus vLLM's profiling adapter requires two LoRA slots even though a request uses only one serving adapter.
  Evidence: `max_loras=1` failed with all slots pinned during profiling; `max_loras=2` loaded successfully and completed the live suite.

- Observation: Current vLLM's LoRA mapping selects one adapter identifier per token row; it cannot directly evaluate the sum of a changing policy LoRA and the fixed Uno LoRA on one draft-noise row.
  Evidence: the model-runner `LoRAMapping` contains one integer mapping per token. The native Uno proposer therefore needs either a synthesized composite adapter or a materialized policy delta.

- Observation: vLLM exposes prefix-cache reuse as scheduler-local token deltas, while the existing TRL bridge retained only capacity and peak usage and its asynchronous path did not snapshot scheduler metrics at all.
  Evidence: `SchedulerStats.prefix_cache_stats` reports `queries` and `hits` in tokens. The new runtime tracker resets at collection admission, snapshots after drain and before sleep, and derives rates only after raw-count aggregation.

- Observation: Verifiers reward components are objects, but the learner never receives those objects.
  Evidence: `Trace.reward` sums each component's `score * weight`; `VerifiersEnvironmentRolloutBridge._project` then converts that value to `float`, and the rollout adapter emits finite scalar `algorithm_reward` rows. A controlled `[0, 1, 1, 0]` bridge probe also preserved the values exactly through TRL's generic callback.

- Observation: the failed VORTEX attempt nevertheless routed its authoritative native scalar back through TRL's generic reward-function interface before group-variance admission.
  Evidence: `policy_rollouts.py` emitted `algorithm_reward`, then `_bridge_reward` reconstructed the same scalar from a mutable input row. The adaptive trainer now consumes the explicit native field directly, rejects partial/non-finite populations, and exposes it to the unchanged TRL group-statistics implementation as a `[rows, 1]` tensor.

- Observation: modern Verifiers represents a multi-turn tool episode as several sampled/raw physical branches plus a canonical unsampled assistant sibling that owns the following tool message.
  Evidence: the live projection failures reported 2, 5, 6, 7, and 12 trainable branches. Resolving the unique deepest terminal path through exact sampled siblings produced complete token/logprob trajectories and allowed active-sampling variance to reach admission.

- Observation: `logits_chunk_size` and real MoE router auxiliary loss were independently implemented but artificially mutually exclusive in the TRL fork; separately, TRL falsely classified dense K2-Horizon as an MoE.
  Evidence: the four-row live probe filled its batch and then failed before backward with `logits_chunk_size is not supported with router auxiliary loss`. The pinned K2 config has `num_experts=0` and `num_experts_per_tok=0` but still exposes `output_router_logits=False`; TRL's field-presence test therefore enabled a nonexistent objective. MoE detection now requires a positive expert count, while actual MoEs carry router logits through chunked scoring.

- Observation: the prior shared runtime base could not safely parent the new veRL closure because it carried Torch 2.11 and older cuBLAS, cuDNN, cuSPARSELt, and NCCL distributions.
  Evidence: fail-closed shared-fallback validation rejected inherited `nvidia-cublas==13.1.0.3` against the backend lock's `13.1.1.3`; regenerating and rebuilding the universal base installed the exact Torch 2.13 shared set and made the real veRL image gate pass without duplicate CUDA packages.

- Observation: an optional BuildKit trust-bundle secret could reuse a cached layer created without the secret because secret contents do not participate in Docker RUN cache identity.
  Evidence: the first base retry still failed with `UnknownIssuer` after mounting the LAN CA. The runtime builder now supplies the bundle SHA-256 as a non-secret build argument, which invalidated that layer and allowed the hash-locked private-index installation to complete.

- Observation: building a vLLM Git dependency from source still runs its Python packaging step even when the intended CUDA extension bytes already exist, and that packaging step fails in generic runtime stages unless vLLM's supported precompiled-wheel overlay is selected explicitly.
  Evidence: the first protected 0.4.3 candidate failed with `CUDA_HOME is not set`; the direct rebuilt `serve-smoke` used `VLLM_USE_PRECOMPILED=1`, installed fork version `0.26.1.dev1+g37706e7d9`, and completed without CUDA compilation.

- Observation: the transform runtime has an independent quantization lock, so advancing the universal base to Torch 2.13 does not update transform automatically.
  Evidence: the first candidate could not satisfy `torch==2.11.0+cu130` from the transform lock; regenerating that isolated lock from an exact Torch 2.13 CUDA URL made the direct `transform-smoke` pass with llmcompressor and Trackio dev24.

- Observation: Trackio's retained release and both package indexes carried sdist digest `f7e3ba065f1085bff171b384899d06d83d11067e934078dc67d49b8b6f542704`, while Posttrain's consumer receipt still recorded a stale locally built digest `7ba6ac88cb6f50b1682c4a6e196c5722dd9973626a896aa3ce13cc894135eb62`.
  Evidence: protected final run `35304179400` rejected stable fork verification before publishing any Posttrain bytes; retained promotion run `35302677562` proved the `f7e3...` bytes identical in development and stable.

- Observation: final distribution building accepted a materialization directory but called strict release consistency validation before projecting that materialization, making the retained-candidate path fail on the pending state it is designed to resolve.
  Evidence: final run `35305485843` verified stable forks and rc2 intact, then stopped before building final artifacts because `build-python-distributions` rejected the authored runtime locks prior to `materialization-apply`.

## Decision Log

- Decision: Base the new fork line on exact upstream commit `75c71390d5b399f5397a9166920fc45902f99f14`, the revision already qualified in the K2 comparison, rather than rebasing the v0.25.1 maintenance branch.
  Rationale: This preserves benchmark comparability and avoids porting K2 onto obsolete fused-MoE APIs. Existing worktrees and unpublished commits remain recoverable and unchanged.
  Date/Author: 2026-09-17 / Codex

- Decision: Implement Uno as a native vLLM speculative method rather than embedding or invoking the standalone Nano-vLLM runtime.
  Rationale: Native integration retains vLLM scheduling, continuous batching, prefix caching, target log probabilities, cancellation, sleep mode, policy updates, and veRL/TRL compatibility.
  Date/Author: 2026-09-17 / Codex

- Decision: Treat K2/current RL weights as the target policy and the Uno adapter as a draft-only overlay.
  Rationale: This preserves lossless target-policy sampling. Training directly against the Uno adapter would optimize the drafter rather than the policy being verified and would invalidate rollout log-probability semantics.
  Date/Author: 2026-09-17 / Codex

- Decision: Support both full-weight and policy-LoRA target updates. Do not declare RL support if the implementation cannot compose a policy LoRA with the draft-only Uno overlay.
  Rationale: Posttrain's update plan permits full, LoRA, and QLoRA training. A rollout accelerator that silently falls back to foundation weights for verification would produce stale-policy samples.
  Date/Author: 2026-09-17 / Codex

- Decision: Reject merged-bfloat16 LoRA materialization and retain native policy-LoRA precision.
  Rationale: live probes showed realistic `1e-4` and `1e-3` merged updates disappeared after bfloat16 materialization, while an exaggerated `0.02` step collapsed output. The accepted path keeps policy LoRA native and synthesizes a rank-concatenated policy-plus-Uno adapter for noise rows only. Full-policy training continues to use independent full-weight refresh.
  Date/Author: 2026-09-17 / Codex

- Decision: Do not amend the frozen product baseline.
  Rationale: `InferenceBinding.engine` already owns backend-specific speculative settings, while `packages/train` owns rollout correctness and the binding revision captures the new behavior.
  Date/Author: 2026-09-17 / Codex

- Decision: Keep target execution compiled and force only native Uno proposal forwards to eager mode until Uno has a shape-safe dedicated CUDA graph.
  Rationale: This preserves target-path performance while preventing stale/aliased target buffers from corrupting proposal tokens.
  Date/Author: 2026-09-17 / Codex

- Decision: Build proposal rows from the target-sampled seed plus noise positions, never from the full target query, and reserve two LoRA slots for the pinned adapter and vLLM profiling.
  Rationale: The target query is already represented in KV state; replaying it changes positions and destroys acceptance. Two slots avoid eviction of the pinned adapter during memory profiling.
  Date/Author: 2026-09-17 / Codex

- Decision: Persist prefix-cache query and hit token counts as the source evidence and derive the hit rate from their aggregate at each fixed-policy collection boundary.
  Rationale: averaging scheduler-iteration rates would weight small and large prefixes equally, while cumulative process counters would leak prior policy versions into later optimizer steps. Raw per-collection counts preserve an exact denominator and respect policy-cache invalidation.
  Date/Author: 2026-09-18 / Codex

- Decision: Treat Posttrain's native `algorithm_reward` as authoritative at the adaptive trainer boundary instead of round-tripping it through a generic TRL callback.
  Rationale: Posttrain owns environment execution and algorithm-specific shaping for this rollout path. Direct finite scalar rows eliminate a second interpretation seam while preserving structured components in trace evidence and ordinary TRL single- or multi-reward functions for environments that do not provide native rewards.
  Date/Author: 2026-09-18 / Codex

- Decision: Preserve both chunked LM-head scoring and the model's router auxiliary loss.
  Rationale: disabling either would hide a fork integration gap and would make the probe unlike the intended memory-efficient MoE training profile. The backbone output contains both hidden states and router logits, so the two objectives are compatible.
  Date/Author: 2026-09-18 / Codex

- Decision: Build generic vLLM job kinds from the immutable CarbonTeq source commit with the SHA-verified binary wheel from its recorded upstream base as the extension overlay.
  Rationale: this retains the fork's Python-level Uno changes and reproducible source identity while reusing ABI-compatible CUDA extension bytes; generic job-kind stages no longer need the CUDA toolkit merely to package the fork.
  Date/Author: 2026-09-18 / Codex

## Outcomes & Retrospective

The native proposer and position-gated adapter now run inside vLLM on the RTX PRO. The primary K2 LoRA optimizer-step gate passes with exact policy versions, finite target logprobs, stable completion tokens, and measurable post-update logprob movement. The implementation retains native policy-LoRA precision and applies a refreshed policy-plus-Uno composite only to draft-noise rows. The bounded K2/Uno/VORTEX probe also completes a real LoRA optimizer step: modern Verifiers multi-turn projection, environment-neutral scalar reward handoff, two-round active sampling, chunked policy scoring, and finite backward all passed together. The failed attempts exposed and corrected integration bugs at their ownership boundaries: canonical Verifiers branch projection, router-loss capability detection, and stale runtime dependency pins. The three maintained forks now have immutable candidate releases and exact-byte development publication. Stable promotion remains gated on the rebuilt veRL image, bounded GPU qualification, full-policy changed-weight qualification, mixed-batch abort churn, the 20-step stability run, and the retained warm long-prompt comparison.

## Context and Orientation

Three repositories participate. `/home/hammad/projects/vllm-uno` is the new CarbonTeq vLLM worktree and owns generic Uno speculative execution. `/home/hammad/projects/rl` owns Posttrain inference-binding validation, rollout integration, immutable dependency pins, catalogs, observation, and release qualification. `/home/hammad/projects/k2-horizon-inference/vendor/uno` is a dirty experimental checkout of `ifm-ai/uno` used only as reference input; its code and local telemetry optimizations must not be mistaken for a published dependency or modified implicitly.

Psi-Spec is Uno's two-pass speculative algorithm. One draft pass evaluates a row shaped like `[causal seed, noise, ...]` with the Uno adapter active only on noise positions. It produces one clean target token plus several candidate future tokens and their draft probabilities. A target-policy pass verifies those candidates. Rejection sampling accepts a prefix while preserving the target policy's probability distribution, then emits a correction or lookahead token. “Lossless” here means distributionally equivalent to sampling the target policy under the same request parameters; it does not mean byte-identical output for unrelated random-number streams.

The primary vLLM seams are `vllm/config/speculative.py`, `vllm/v1/spec_decode/`, and `vllm/v1/worker/gpu_model_runner.py`. The target model and its KV cache already live in the GPU model runner. The integration must not allocate a second full K2 model. The proposer needs a bounded same-model draft-forward API, must roll back draft KV state before target verification, and must return both token IDs and draft probabilities in the form expected by vLLM's rejection sampler.

The Posttrain seam is `packages/train/src/posttrain/train/backends/trl/common.py::vllm_rollout_options`. It currently admits only native MTP. `packages/train/src/posttrain/train/online_rl.py` and the TRL async sample path require one finite sampling log probability per completion token. The catalogs and qualification code live under `apps/lab` and `.posttrain` overlays. The consumer documentation is `docs/tooling/vllm/README.md`; the fork ledger is `/home/hammad/projects/vllm-uno/CARBONTEQ_FORK.md`.

## Plan of Work

First, add `uno` to vLLM's speculative configuration with explicit fields for the immutable adapter path, maximum draft width, noise mode, and any verification controls that materially affect behavior. Reject unsupported combinations at configuration time, including structured output if correctness has not been demonstrated, pipeline parallelism beyond the implemented topology, and missing K2 architecture markers. Do not overload ordinary `enable_lora` or a request LoRA identifier to mean the Uno overlay.

Second, add a native proposer under `vllm/v1/spec_decode/uno.py`. Reuse vLLM's scheduler, batch metadata, sampler, rejection sampler, and metrics. Port only the algorithmic pieces absent from vLLM: deterministic bounded-noise construction, the same-model draft input shape, draft distribution extraction, KV rollback/commit bookkeeping, and adapter gating. Attribute the source and retain the compatible upstream license notices. Start with the linear block path; tree verification is a later optimization and must not block correctness qualification.

Third, introduce a conditional adapter overlay owned by the model runner. Full-policy targets use Uno alone on draft-noise rows. LoRA-policy targets keep the native policy adapter on target and seed rows; TRL atomically synthesizes a rank-concatenated policy-plus-Uno adapter for noise rows and reloads both at every optimizer boundary. Full-weight updates remain independent and commit through the IPC lifecycle.

Fourth, connect the proposer to `GPUModelRunner` using a narrow native interface. The runner creates the proposer with access to itself, executes one same-model draft forward without a second model allocation, returns draft IDs and probabilities, and restores the committed KV frontier before target verification. Existing target sampling and output log-probability code remains authoritative.

Fifth, extend Posttrain. `vllm_rollout_options` accepts `method: uno` only for a K2 model variant that declares a new qualified capability and whose binding carries the exact Uno adapter repository and 40-character revision. Resolve the adapter snapshot before engine construction, just as paired Gemma assistants are resolved. Add a versioned experimental rollout binding rather than changing existing defaults. Record adapter identity, draft width, proposed and accepted counts, acceptance length, and effective warm throughput.

Finally, qualify in increasing cost order: pure configuration tests, deterministic algorithm tests, tiny GPU smoke, fixed-weight distributional comparison, warm long-context benchmark, then a real rollout session across at least one optimizer update. Only after the fork commit is pushed should Posttrain advance its immutable pin and lockfile. The prior standalone benchmark remains a performance reference, not release evidence for the integrated runtime.

## Concrete Steps

Work in `/home/hammad/projects/vllm-uno` for the fork:

    git status --short --branch
    uv run --python 3.13 pytest -q tests/config/test_speculative.py
    uv run --python 3.13 pytest -q tests/v1/spec_decode
    uv run --python 3.13 ruff check vllm/config/speculative.py vllm/v1/spec_decode tests/v1/spec_decode
    git diff --check

Use the existing K2 benchmark cache and pinned model inputs for GPU work, but execute the integrated vLLM branch rather than the standalone Uno engine. Capture the exact engine config and warmup marker separately from measured requests.

Work in `/home/hammad/projects/rl` for the consumer:

    uv run pytest packages/train/tests/test_trl_common.py
    uv run ruff check packages/train apps/lab
    uv run pyright
    uv run lint-imports
    git diff --check

Before release, run the complete locked validation ladder from `/home/hammad/projects/rl`:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

## Validation and Acceptance

Configuration acceptance requires `method: uno` with a pinned adapter to resolve reproducibly and malformed or unsupported combinations to fail before model allocation. Existing MTP, draft-model, n-gram, and no-speculation tests must remain unchanged and pass.

Algorithm acceptance requires deterministic unit vectors for draft construction and rejection sampling, including immediate rejection, full-block acceptance, EOS, output-budget truncation, mixed request lengths, and batch churn. A seeded statistical test must show that integrated Uno and ordinary target sampling agree within a declared tolerance on a small vocabulary distribution.

Serving acceptance requires warm-only runs with the retained large-prompt suite: approximately 8.8K, 17.6K, 26.4K, and 30.8K input tokens at concurrency four and a 16,384-token output ceiling. Both normal K2 and integrated Uno must use identical target weights, sampling parameters, prompt population, and warmup exclusion. The report must include concurrency, actual input/output counts, target throughput, TTFT, acceptance by position, errors, truncations, and peak VRAM.

RL acceptance requires a rollout before and after a known optimizer update. The primary native-LoRA gate passed on 2026-09-18 with policy versions `0` and `1`, 32/32 finite sampled logprobs, stable tokens, and maximum common-token logprob movement `0.0655067`. A separate full-policy update remains required for full-update support. QLoRA remains unsupported until independently designed and qualified.

## Idempotence and Recovery

The new worktree is additive and does not mutate `/home/hammad/projects/vllm` or `/home/hammad/projects/vllm-nanbeige`. If the selected upstream revision proves unsuitable, create a new branch from a named immutable revision; do not reset the existing maintenance branches. Keep the Uno feature disabled by default until qualification. Failed GPU runs may be retried after removing only their own temporary process and cache state; never delete shared Hugging Face model blobs or retained Trackio evidence as part of test cleanup.

If a fork change must be abandoned, preserve its branch and record the failed gate here. Do not advance the Posttrain dependency pin until the fork commit exists on `origin` and can be fetched into a clean environment.

## Artifacts and Notes

Current immutable inputs:

    vLLM base: 75c71390d5b399f5397a9166920fc45902f99f14
    K2 base: IFM/K2-Horizon-7B@586b03f0fd1fbbf2f13eeafc33749e95ae34dd10
    Uno adapter: IFM/K2-Horizon-7B-Uno@ec92bbd768f4a404319625204544782e3377bcd7
    Uno reference runtime: ifm-ai/uno@56dca56b274ba1a11b0c2dbb32d9beb76797e8e4

The earlier standalone warm 16K comparison is retained under `/home/hammad/projects/k2-horizon-inference/results/README.md`. It demonstrates potential performance but does not prove integrated-vLLM correctness or RL lifecycle support.

## Interfaces and Dependencies

At the end of the fork milestone, `SpeculativeConfig` accepts a native `method: uno` and immutable adapter identity. `GPUModelRunner` owns one `UnoProposer` on the final pipeline rank. The proposer returns draft token IDs and draft probabilities through existing vLLM speculative fields; target verification and output log probabilities remain vLLM-owned.

At the end of the Posttrain milestone, `vllm_rollout_options(model, engine)` translates a validated Uno engine mapping into vLLM configuration without importing vLLM types into public framework models. A versioned K2 rollout binding selects the accelerator explicitly. Existing bindings remain unchanged.

Revision note (2026-09-17): Initial plan created after locating the deleted disposable checkout's exact vLLM revision and determining that the published CarbonTeq fork had not yet been upgraded. Updated after adding the initial Uno configuration contract, recovering the retained RTX runtime, passing its focused tests, and recording the reference runtime's failed RL lifecycle admission.
