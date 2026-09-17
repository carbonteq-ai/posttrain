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
- [ ] Extend Posttrain's rollout option resolver, catalogs, metrics, and tests for a pinned Uno adapter revision. (Resolver and focused tests are complete; catalog selection and released dependency pins remain open.)
- [ ] Run distributional correctness, tool-call, long-prompt, 16K-output, concurrency, warm-throughput, and RL weight-update qualification on the RTX PRO 6000.
- [x] (2026-09-17 15:44Z) Ran the retained short Uno reference smoke on the RTX PRO and audited its public lifecycle interfaces; generation passed but RL admission failed on missing target logprobs, policy-version fencing, weight refresh, abort, and sleep/wake.
- [x] (2026-09-17 17:05Z) Ran the native vLLM Uno path on the RTX PRO: 128/128 and 256/256 finite target logprobs, a parsed K2 tool call, and four concurrent 128-token requests with no preemptions.
- [x] (2026-09-17 17:18Z) Checkpointed the fork implementation and ledger as local commit `8946dee55`; publication and Posttrain pin advancement remain gated on lifecycle qualification.
- [x] (2026-09-17 17:28Z) Enabled loopback-only vLLM lifecycle controls and passed level-1 sleep/wake, post-wake 32/32 target logprobs, single-request abort, and pause/cache-clear/resume on the RTX PRO.
- [x] (2026-09-17 17:31Z) Set and read back the exact base policy revision through vLLM's weight-version register; Posttrain still must attach and fence that value on each rollout because OpenAI completion responses omit it.
- [x] (2026-09-18 00:10Z) Passed the primary live K2 LoRA optimizer gate on RTX PRO: policy version `0` to `1`, finite target logprobs, trainable delta norm `0.1214`, stable tokens, and maximum sampled-logprob movement `0.0655` at learning rate `1e-4`.
- [ ] Commit and push the vLLM fork before updating immutable Posttrain dependency pins and lockfiles.

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

## Outcomes & Retrospective

The native proposer and position-gated adapter now run inside vLLM on the RTX PRO. The primary K2 LoRA optimizer-step gate passes with exact policy versions, finite target logprobs, stable completion tokens, and measurable post-update logprob movement. The implementation retains native policy-LoRA precision and applies a refreshed policy-plus-Uno composite only to draft-noise rows. Full-policy changed-weight qualification, mixed-batch abort churn, and the retained warm long-prompt comparison remain open.

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
