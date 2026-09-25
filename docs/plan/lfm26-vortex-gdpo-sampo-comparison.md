# Compare GDPO and SAMPO under a shared VORTEX collection policy

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current as work proceeds. Follow `docs/templates/PLAN.md` and the frozen product baseline in `docs/post-training/README.md`.

## Purpose / Big Picture

The primary question is whether GDPO or SAMPO trains the LFM2.5-2.6B AutomationBench tool agent better when both receive the same VORTEX task-allocation and candidate-collection policy. GDPO uses normalized named episode-reward components; SAMPO uses episode-relative and anchor-state-relative turn advantages with sequence-level clipping. The experiment must isolate that algorithm difference. Each new arm should have 50 real optimizer updates, the same starting checkpoint, task population, prompt-group size, rollout model and serving policy, rollout budget, and held-out evaluation. Record both generated and retained trajectories so equal update counts cannot hide unequal rollout compute.

VORTEX currently names a concrete OLMo3/GRPO composition, not a generic switch. This plan first extracts and qualifies its adaptive curriculum and bounded group-refill policy as a reusable collection layer while leaving GDPO and SAMPO mathematics unchanged. The old no-oversampling instruction conflicts with full VORTEX active refill; treat the user's latest request for GDPO with VORTEX as superseding no-oversampling for the new comparison, but cap and report candidate work identically across arms. Do not submit another 50-update run until that semantics is validated.

## Progress

- [x] (2026-09-20) Corrected the primary comparison from adaptive-GRPO versus GDPO to VORTEX+GDPO versus VORTEX+SAMPO. Preserve completed GRPO and failed GDPO attempts as diagnostic lineage only.
- [x] (2026-09-20) Confirmed from local code that VORTEX active sampling and adaptive curriculum live on `GRPOSettings` with `algorithm=olmo3`; `GDPOSettings` and `SAMPOSettings` have separate objective and admission contracts.
- [ ] Define a shared collection-policy contract and make the narrow canonical baseline amendment before implementation. Keep OLMo3-specific loss invariants inside GRPO; do not rename either GDPO or SAMPO as OLMo3.
- [ ] Implement adaptive task allocation and bounded candidate-group refill for GDPO and SAMPO with independent evidence, checkpoint state, and fail-closed admission.
- [ ] Qualify one common multi-turn episode scorer/evidence stream for both objectives. GDPO must receive complete named components, and SAMPO must receive original sampled-turn spans and stable preceding observation keys. Specify exactly how the same underlying episode evidence is reduced for each objective.
- [ ] Fix and qualify the Gemma judge's rendered-token budget and scorer calibration from failed GDPO attempt `20260920c`; qualify actual compiled/MTP or DSpark behavior before changing the judge binding.
- [x] (2026-09-20) Added a candidate vLLM `/tokenize` preflight and context-aware output cap in the dirty `../verifiers-environments` checkout. It is opt-in, has not been committed/pinned, and has not been GPU-qualified; therefore the judge gate above remains open.
- [x] (2026-09-20) Added reproducible offline measurement in `scripts/qualification/episode_judge_token_budget.py` and measured all 50 retained one-pass judge inputs with Gemma's pinned tokenizer revision. This establishes 48K/40K/8K as the lower-cost candidate, subject to serving-template parity and GPU qualification.
- [ ] Create two new immutable LFM work packages and run bounded 1–3-update pilot jobs, then matched 50-update jobs only after both pilots pass. Evaluate both final adapters on the same held-out task set.
- [ ] Publish a comparison that separates task quality, algorithm effect, generated/retained rollout compute, truncation, reward coverage, and GPU cost. Report all failed and superseded attempts.

## Surprises & Discoveries

- Observation: The previous experimental design did not answer the user's question. Its `lfm26-adaptive-grpo-50-c32-20260920f` run completed 50 updates, but its objective was GRPO, not SAMPO, and it had no VORTEX active refill. Evidence: the old plan and the `GRPOSettings` catalog selection `lfm2.5-2.6b/automationbench-adaptive-grpo-no-oversampling-50-local-v1`.
- Observation: Current VORTEX is coupled to OLMo3 in code. Evidence: `packages/train/src/posttrain/train/profiles.py` rejects active sampling unless `GRPOSettings.algorithm == "olmo3"`; `packages/train/src/posttrain/train/backends/trl/policy_optimization.py` installs the curriculum trainer only for `GRPORequest`.
- Observation: GDPO and SAMPO cannot be made comparable by merely selecting the same training settings. Evidence: `docs/post-training/02-primitives.md` gives GDPO component-wise group and batch normalization while SAMPO needs token-aligned turns and anchor states. `SAMPOSettings.dynamic_sampling` and GDPO's `max_admission_attempts` currently describe different selection behaviors.
- Observation: The earlier GDPO retry is not ready to become a comparison arm. Attempt `20260920c` finished 11 updates, then failed when a Gemma judge call reserved 16,384 output tokens after at least 16,385 rendered prompt tokens against a 32,768-token context. The step-10 checkpoint and traces remain diagnostic evidence, not a 50-step result.
- Observation: The new guard is locally tested but not a substitute for capacity qualification. Evidence: 33 AutomationBench tests pass and targeted Pyright on the new budget path reports zero errors. Whole-package Pyright still reports two pre-existing `taskset.py` override errors. The retained 50-case input corpus shows that compact JSON syntax saves about 1,030 characters per episode on average, which is far smaller than the longest trajectories; it does not establish that 12K input is sufficient.
- Observation: A retained two-case direct-protocol diagnostic already includes a 32,928-token prompt with a 958-token completion under an 8,192-token output allowance. Evidence: `outputs/qualification/lfm26-oversample-training-sample-50-20260920/gemma4-dspark-v16-invalid2-direct-report.json`. This is a different dirty v16 scorer/runtime diagnostic, not qualification of the pinned v13 production scorer, but it proves a blanket 12K input ceiling would exclude at least some realistic long episodes and that 16K output reservation is likely excessive for those cases.
- Observation: Offline Gemma tokenization over the complete retained 50-case corpus now quantifies the gap. Evidence: running `scripts/qualification/episode_judge_token_budget.py` with model revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` gives original input median 18,052.5, p95 32,619, max 33,364 and 39/50 above 12,288; lossless compact JSON gives median 17,183.5, p95 31,366, max 32,019 and 37/50 above 12,288. Eight compact inputs exceed the 24,320 tokens left by a 32K context with 8K output and a 256-token margin. This is tokenizer-local evidence, not a proof of exact vLLM `/tokenize` parity.

## Decision Log

- Decision: Compare VORTEX+GDPO with VORTEX+SAMPO, not adaptive-GRPO with GDPO. Rationale: this is the user's corrected primary question. Date/Author: 2026-09-20 / User and Codex.
- Decision: Interpret VORTEX's reusable part as adaptive task allocation plus bounded active candidate collection, while retaining each arm's own objective. Rationale: importing the OLMo3 loss into GDPO or SAMPO would erase the intended algorithm comparison. This interpretation requires an explicit narrow amendment to the frozen baseline before code changes. Date/Author: 2026-09-20 / Codex.
- Decision: Keep the prior 50-update GRPO run as a secondary diagnostic baseline only; do not count it as either primary arm or silently resume failed GDPO attempt `20260920c`. Rationale: objective and collection semantics differ, and a scorer/runtime fix requires a new immutable run identity. Date/Author: 2026-09-20 / Codex.
- Decision: Use the same candidate cap and measure both attempted and retained groups, rather than claiming that 50 updates alone equals matched compute. Rationale: the two objectives may admit different populations and consume different judge work. Date/Author: 2026-09-20 / Codex.

## Outcomes & Retrospective

The comparison has been reframed but not yet implemented or run. No SAMPO LFM arm or VORTEX+GDPO arm exists. The earlier GRPO result is complete and the GDPO judge-budget failure is diagnosed; neither answers the corrected primary question. The immediate release gate is a common, versioned collection policy with mathematically unchanged objectives and a bounded, calibrated judge.

Judge-budget work has begun in `../verifiers-environments` without publishing or changing the active environment revision: `src/automationbench_v1/judge_budget.py` measures the exact server-rendered chat through the local vLLM `/tokenize` endpoint, and `judge.py` caps the output reservation against measured input and the smaller of configured and server context. The existing input ceiling now fails closed for the opt-in path, with an `unjudgeable` attempt rather than an overflowing completion request. `episode_prompt.py` removes JSON formatting whitespace without dropping any field. This does not yet solve cases genuinely longer than the 12K input ceiling, nor the all-invalid-group admission case; measure representative one-pass prompt tokens and qualify 48K/64K context and a smaller output reservation before selecting a new production profile.

The next candidate should not retain the old 12,288-input/16,384-output/32,768-context triple. The preferred 48K/40K/8K point is arithmetically valid (40,000 + 8,192 + 256 = 48,448 < 49,152) and covers all 50 offline-tokenized direct-mode episodes with at least 7,981 tokens of input headroom after compaction. A 64K/56K/8K point (56,000 + 8,192 + 256 = 64,448 < 65,536) remains a fallback if later task populations require it and the GPU can support it. These are test points, not production selections: the 50-case frame report sums multiple calls per case, and local Hugging Face chat-template counts must be checked against the actual vLLM `/tokenize` endpoint before promotion. Choose the smallest context that covers the agreed population while preserving actual c32 scheduler throughput, KV admission, and rubric validity. If some episodes exceed the feasible context, mark them explicitly unjudgeable and redesign scoring rather than silently truncating tool evidence.

KV-cache terminology matters here. Planning only validates a declared token envelope; it does not allocate GPU blocks. The selected Gemma profile has `gpu_memory_utilization: 0.47` and no `kv_cache_memory_bytes`, so this vLLM fork profiles memory and creates one fixed-capacity GPU block pool at engine startup. Its scheduler allocates and frees blocks to individual requests as tokens arrive; it does not expand the physical pool at runtime. vLLM's startup check requires enough pool capacity for one request at `max_model_len`, which is why a larger declared context can fail before a short request runs. The 48K/64K qualification should therefore measure both minimum one-request admission and real pooled occupancy/queueing at c32, without confusing dynamic per-request block assignment with elastic growth of the entire arena.

## Context and Orientation

The canonical product boundary is in `docs/post-training/README.md`, `02-primitives.md`, `04-framework.md`, `05-apis.md`, and `06-observation-and-lineage.md`. Training setting types are in `packages/train/src/posttrain/train/profiles.py`, catalog decoding in `catalog_schema.py`, the TRL online-RL path in `backends/trl/policy_optimization.py`, adaptive task allocation and refill in `backends/trl/policy_curriculum.py`, and rollout admission in `backends/trl/policy_rollouts.py` and `reward_admission.py`. The experiment catalog is `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml`; work packages are in `apps/lab/.posttrain/work_packages/`.

A prompt group is one task occurrence with four policy trajectories. A candidate group is a generated group that may or may not be retained for an optimizer update. An anchor state is the user or tool observation preceding a sampled assistant turn. The native Verifiers episode and trace remain replay authority. Invalid, absent, or abstained reward evidence never becomes a numeric zero.

`../trl` is the maintained trainer fork; its observed checkout on 2026-09-20 was `codex/bounded-vllm-waves` at `c9af78c1c2ea04ad271e95b26b93dfadf8b9fca1`. Resolve the executable pin from `packages/train/pyproject.toml` and `uv.lock` before editing it. `../verifiers-environments` was at `a6d779fc1fdfde23f86e297125b3381b140cec2f` with existing dirty files `episode_prompt.py`, `limited_tools.py`, and `tests/test_judge.py`; preserve them and do not treat the dirty checkout as a published scorer. Any fork change needs its own commit, push, fork ledger, consumer-page update, and immutable pin.

## Plan of Work

First amend the baseline narrowly to make curriculum allocation and active candidate collection composable with `train.gdpo` and `train.sampo` without changing their advantage or loss equations. Define a versioned collection policy separate from algorithm settings. It must state retained prompt groups, generations per group, maximum candidate rounds, reward-based admission signal, missing-evidence handling, and whether SAMPO's existing dynamic filter is replaced or composed. For this primary comparison, use one shared admission decision over the same valid episode evidence; never let GDPO filter on one score while SAMPO filters on another. Record the decision and generated/retained group identities in native traces and tracking.

Implement the shared policy at the training collection boundary rather than duplicating VORTEX through two trainer-specific methods. Preserve group atomicity, distinct task identity, policy-version fences, native token/turn provenance, and checkpoint/resume state. Keep GDPO's complete component-vector normalization and SAMPO's hierarchical turn advantages in their existing objective owners. Add tests using the same synthetic fixed-policy episode population in both operations: both must request identical task classes and candidate groups; each must retain the same group IDs; only the calculated advantages and losses should differ. Add missing-score, uniform-reward, incomplete-group, max-round, batch-churn, and resume tests.

Next make the episode judge safe at the actual tokenizer-rendered request boundary. Bound prompt tokens plus reserved output plus margin within the selected Gemma context, avoid dropping material tool observations silently, and replay long episodes that broke `20260920c`. Use a reviewed frozen case set to verify structured-output validity and rubric labels. If the same judge emits named episode components and turn rewards, pin both projections explicitly. If not, qualify the common scorer/evidence stream before claiming the comparison isolates the optimizer. Verify target/proposer compilation and actual MTP or DSpark counters rather than inferring execution from `enforce_eager: false`.

Create new versioned GDPO and SAMPO selections and work packages for LFM2.5-2.6B with the same starting policy, environment task mix, 8 retained groups × 4 generations, 32 rollout concurrency, length limits, LoRA update plan, candidate cap, and inference binding. Use one shared held-out evaluation suite. Run short pilot jobs first and require at least three optimizer steps in each arm with valid optimizer, policy-version, reward-coverage, and saturation evidence. Only then submit two new 50-update runs; if total compute becomes materially higher than the earlier 100-update budget, surface that cost before launch. Do not reuse the old GRPO or GDPO run identities.

## Concrete Steps

Run from `/home/hammad/projects/rl` unless noted otherwise:

    git status --short --branch
    git -C ../trl status --short --branch
    git -C ../verifiers-environments status --short --branch
    rg -n 'trl|verifiers-environments' packages/train/pyproject.toml packages/eval/pyproject.toml uv.lock
    uv run pytest packages/train/tests/test_adaptive_curriculum.py packages/train/tests/test_api.py packages/train/tests/test_reward_admission.py apps/lab/tests/test_catalog.py apps/lab/tests/test_work_packages.py -q
    uv run ruff check packages/train apps/lab
    uv run pyright
    uv run lint-imports
    git diff --check

After versioned work packages exist, run `uv run posttrain work-package validate <new-package>.yaml` from `/home/hammad/projects/rl/apps/lab` for each arm and the held-out evaluation. Compile both full job plans against the same released runtime before submission. Exact future selection names and run IDs must be added here when created; do not invent evidence ahead of publication.

## Validation and Acceptance

Unit and integration tests must prove both algorithms receive identical task-class and candidate-group sequences for a deterministic seed and frozen synthetic rewards. GDPO must emit finite, correctly normalized component advantages; SAMPO must emit finite episode/turn advantages aligned to sampled policy tokens. Neither may accept a missing reward as zero or continue after an all-invalid population without an explicit bounded outcome. The 1–3-step pilot for each arm must complete real optimizer updates and show correct policy refresh, no unsupported fallbacks, and generated/retained group telemetry. Judge replay must include a formerly overflowing episode whose rendered prompt plus output reservation fits the actual server context.

Final comparison acceptance requires two new 50-update adapters, the same held-out tasks evaluated separately against each, and a report with quality by task type, tool correctness, truncation, total rollout/judge tokens, candidate rounds, retained fraction, GPU time, and uncertainty. Matched step count alone is insufficient if one arm spends much more compute or has different scorer coverage.

## Idempotence and Recovery

Read-only inspection, static validation, and job planning may be repeated. Use new immutable run IDs after any policy, scorer, or runtime change. Never rewrite or delete the completed GRPO run, the 11-step GDPO attempt, or their traces. Preserve dirty sibling-fork work; commit and push maintained fork changes before updating exact pins. On pilot failure, retain the traces and repair the owning layer before another submission.

## Artifacts and Notes

Observed contract boundary:

    GRPOSettings: active_sampling is valid only with algorithm="olmo3"
    GDPOSettings: ordered named reward components, no VORTEX collection field
    SAMPOSettings: hierarchical turn objective and its own dynamic_sampling field

Thus setting an `olmo3` flag on either requested arm would not implement the experiment.

## Interfaces and Dependencies

The new collection contract belongs in `packages/train`, not the Verifiers environment, judge plugin, or a work-package-only flag. The environment owns task identities and native trace evidence; the judge owns rubric meaning; the training package owns candidate admission and algorithm-specific advantages. The lab host composes model, environment, scorer, algorithm settings, collection policy, training binding, and rollout/judge inference bindings. External runtime dependencies are the pinned CarbonTeq TRL and vLLM forks and the immutable Verifiers environment commit. The GPU pilot needs the RTX PRO 6000 and accessible Trackio/Doris lineage; lack of access is a qualification gap, not a zero metric.

Revision note (2026-09-20): Replaced the mistaken adaptive-GRPO-versus-GDPO experimental framing after the user specified GDPO with VORTEX and GDPO-versus-SAMPO as the primary comparison. The former plan is retained only as history and points here.

Revision note (2026-09-20): Began the judge-context repair first at the user's request. The opt-in tokenizer guard and lossless JSON compaction are local candidate changes in the already-dirty environment fork; no catalog pin or run was changed. Remaining qualification is actual long-episode token distribution, scorer calibration, context/KV capacity, and immutable fork publication.

Revision note (2026-09-20): Added explicit 48K/64K candidate judge envelopes after the user pointed out that fail-closed 12K alone does not let the rubric read long episodes. They remain unselected until exact one-pass token counts and GPU saturation evidence distinguish them.

Revision note (2026-09-20): Clarified that the framework's job plan does not reserve GPU KV blocks; vLLM initializes a fixed pool and allocates blocks to requests dynamically. This avoids treating an unimplemented elastic-pool mechanism as an existing setting.

Revision note (2026-09-20): Measured all 50 retained direct-mode inputs with the exact pinned Gemma tokenizer. This selects 48K/40K/8K as the first capacity candidate, replacing the earlier speculative 64K-first posture; live `/tokenize` parity and GPU qualification remain open.
