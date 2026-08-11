# Qualify and run Policy Prism Gemma 4 E2B from 12B OPD

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated whenever work advances. Maintain it according to `docs/templates/PLAN.md` and the frozen product baseline under `docs/post-training/`.

## Purpose / Big Picture

This plan delivers a reproducible and throughput-qualified Policy Prism on-policy distillation (OPD) experiment on the in-house RTX PRO 6000. Gemma 4 E2B is the trainable student and Gemma 4 12B is the frozen teacher. The student generates legal-interpretation responses, the teacher scores the exact generated token IDs, and the maintained TRL 1.9.2.post1 Importance-Weighted OPD (IW-OPD) objective updates a rank-16 LoRA adapter. IW-OPD uses the sampled-token student and teacher log-probabilities and prefix-drift weighting; it is a deliberate replacement for the feature branch's older sparse reverse-KL implementation, not a rename of the same loss.

The previous attempts produced a valid intermediate checkpoint at optimizer step 96 but did not complete training. This plan preserves that checkpoint as historical evidence, merges current `origin/main` into only the experiment feature branch, ports the Policy-specific correctness and memory-safety deltas onto the maintained IW-OPD backend, and records a completed correctness qualification plus bounded capacity probes. Offline and isolated-image gates exhaustively validate the historical structural failures; live evidence covers three repeated logical-four updates, logical-12 heterogeneous batching, physical batches through three, memory, artifacts, and throughput. Completion means that all 384 logical targets are optimized once under the frozen logical-12 configuration, the final adapter and checkpoints are retained in Trackio, the adapter is privately published and freshly verified on Hugging Face, sealed scope and recovery evaluations succeed, and both evaluations are finalized in Policy Prism's normal five-file `evaluation-runs` format.

This plan does not change either `main` branch. All PostTrain commits use `feat/gemma-policy-prism-opd-e2b-12b`; all Policy Prism commits use `feat/scope-opd-e2b-12b-environment-v1`. Failed attempts and their Trackio evidence remain retained.

## Progress

- [x] (2026-08-07) Build and package the deterministic Policy Prism production plan containing 384 primary targets and 96 shared reserves, with sealed-family exclusion, exact target distributions, source review, and E2B token-budget validation.
- [x] (2026-08-07) Implement exact-token E2B-from-12B OPD, memory-safe sparse loss, structured XGrammar projection, stable target projection, Trackio checkpoints, and explicit checkpoint resume on the PostTrain feature branch.
- [x] (2026-08-08) Complete 66 finite updates in `opda06-e2b12b-r16-scope384-v1`; retain checkpoints through step 64 before an external dstack workstation-reachability failure.
- [x] (2026-08-10) Prove exact step-64 restore in `opda08rs-e2b12b-r16-resume64-scope384-v1`; diagnose its target-78 failure as overly narrow prompt-profile reserve matching.
- [x] (2026-08-10) Push Policy Prism plan-v2 reserve semantics in commits `3976d993a2e96dd4c91c3ad91481767299755364` and `47ee4f43366e3a1a68fa59db8ed3ab46613c677c`, allowing shared stage/quartile/decision reserves to inherit the logical target's profile and shape.
- [x] (2026-08-10) Resume from step 64 as `opda09rs-e2b12b-r16-resume64-scope384-v2`, pass target 78, retain checkpoints at steps 80 and 96, and diagnose target 98 before optimizer step 99.
- [x] (2026-08-10) Audit r9 attempt traces: ten of eleven reserve-triggering structural failures were exact `source_provision_id` copying mistakes; one was malformed long JSON.
- [x] (2026-08-10) Constrain each rules-stage wire schema to the exact task-owned `source_provision_id`, preserve canonical downstream validation, pass 20 focused Policy Prism tests, Ruff, and strict mypy, then push Policy Prism commit `147ac75997579f08154145ea9bdc6215b4aa7ec4`.
- [x] (2026-08-10) Include and push the user's local Policy Prism commit `85e0e12d102ea8e32e1b31fe9926179b21cbe2fb`; remote feature-branch tip `147ac759...` is its direct descendant.
- [x] (2026-08-10) Verify checkpoint 96 contains adapter, optimizer, scheduler, trainer, RNG, tokenizer, and SQLite allocation-ledger state; all inspected losses and gradients are finite and teacher failures are zero.
- [x] (2026-08-10) Publish checkpoint 96 privately to `carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-checkpoint-96`, fresh-download it, verify every uploaded file byte-for-byte, and resolve immutable Hugging Face revision `4f1fe9c75031396a11bcc44e2193f96df9003054`.
- [x] (2026-08-11) Select the minimal checkpoint-96 qualification path: serve the immutable private Hugging Face PEFT adapter directly, materialize it at the pinned commit before vLLM launch, and correct local Verifiers chat-template options under OpenAI `extra_body`; the complete serve/eval package suites pass.
- [ ] Run checkpoint 96 through the sealed 18-case scope and 17-case recovery jobs sequentially, finalize both native artifacts into Policy Prism's standard five-file evidence directories, validate the catalog, and push the results.
- [x] (2026-08-10) Cancel the failed r9 provider workspace after checkpoint preservation. It is terminal with provider state `terminated`; reconciliation records failed/inconsistent evidence and retains three training artifacts plus native rollout traces.
- [x] (2026-08-10) Stop before new GPU submissions and update this plan for a separate batch-qualification and full-run execution goal.
- [x] (2026-08-10) Fetch and compare current `origin/main` at `6ffe634432a3f92e8c6dd561d3cd85b2b2ba45cd`, the experiment branch at `dbb5c7b5538124913b99620801339f702c58089e`, and the immutable TRL 1.9.2.post1 fork source at `a82ecebc0fa081efd58302a34a553445fc73271d` without changing either branch.
- [x] (2026-08-10) Audit the target-78 and target-98 Policy Prism fixes, discover the stale production source pin, identify the shared-reserve prompt-budget audit gap, and select the representative 12-target seed-2907 live cohort.
- [x] (2026-08-10) Audit logical rollout batching, physical actor batching, vLLM resident waves, chunked prefill, prefix caching, MTP, sleep mode, KV dtype, and CUDA execution mode; select one research-grounded production candidate rather than a live parameter sweep.
- [x] (2026-08-10) Merge `origin/main` into only `feat/gemma-policy-prism-opd-e2b-12b`, adopt the release/runtime/checkpoint structure, and port the enumerated Policy-specific deltas onto TRL IW-OPD; merge commit `7926e87` and qualification-boundary commit `5b1b87d` are pushed.
- [x] (2026-08-10) Complete Policy Prism's actual-materialization token audit in source commit `411fa6e`, pin the qualification environment to that exact source, and add the resident-two emergency work package in catalog commit `77506c6`.
- [x] (2026-08-10) Pass offline IW-OPD loss/gradient and accumulation equivalence, target-78 forced cross-profile reserve recovery, exact target-98 source enums, all actual-schema XGrammar compilation, concurrent/restart-safe real-ledger claims, selected-stage masking, heterogeneous batching, and checkpoint-ledger gates. The 765 distinct candidate/profile/shape materializations expand to 1,293 stage schemas, all compiled under XGrammar 0.2.3.
- [x] (2026-08-10) Package corrected resident-four and resident-two actual-job images. After the first provider canary exposed a fail-closed private-hook name mismatch, align the exact pinned IW-OPD hook to `aligned_prompt_length`, pass its real-class regression, push PostTrain commit `e8fbd51`, and repack immutable images `sha256:b646c50e...` and `sha256:4b83bea8...` respectively.
- [x] (2026-08-11) Complete and consistently reconcile the required 12-target/three-update resident-four qualification as `opdq-fast01c-iwopd-e2b12b-c12-lb4-rseq4-nomtp`. It finished in 41.61 minutes with finite losses and gradients, zero teacher failures, all required artifacts, and no OOM. Its 18 grouped submissions reached resident wave four, but 2/12 targets required reserves because of graph/dependency structural failures.
- [x] (2026-08-11) Run the urgent logical-batch-12 serving ceiling probe at physical batch one. `opdq-ceil01-iwopd-e2b12b-c12-lb12-pb1-rseq12` completed all 12 targets in 14.44 minutes submission-to-reconciliation, formed a real resident wave of 11, used 34.74 GiB peak trainer memory, and reduced measured training time from 2,112.83 to 634.11 seconds (3.33x).
- [x] (2026-08-11) Remove the branch-local physical-batch-one admission guard, prove exact memory-safe IW-OPD loss and gradient equivalence for physical batches 1/2/4, and push PostTrain commit `1607dd5`. Physical batch two completed without OOM at 38.10 GiB peak trainer memory; its stochastic primary rules output exposed a Policy-side dependency-control gap rather than a trainer defect.
- [x] (2026-08-11) Stop structurally invalid evidence/rules prefixes before downstream generation while retaining their raw output and final harness admission. Push Policy Prism source commit `420a554855248a05e29f07e12c6221244167f99f` and catalog pin commit `b6bab9b`; the focused executor tests, Ruff, strict mypy, and isolated image qualification pass.
- [x] (2026-08-11) Complete and consistently reconcile the physical-batch-three probe on corrected Policy source `420a554...`. It admitted all 12 targets, retained complete checkpoint/model/trace evidence, and produced finite loss `-1.85628` with zero teacher failures. The one update took 1,130.05 seconds, peak trainer memory was 49.83 GiB, and system VRAM peaked at 67.48 GiB; physical batch three was memory-safe but slower than physical batch one.
- [x] (2026-08-11) Add, validate, and push Policy Prism commit `1ded27e` with a logical/resident/concurrent-24 probe at physical batch one, so rollout scaling is measured without confounding it with activation batching. The separate `feat/normative-completeness-pipeline-v1` worktree remains clean and untouched.
- [x] (2026-08-11) Stop live qualification at the user's meeting deadline. The direct physical-12 probe remained healthy through model loading and six rollout waves (maximum wave nine) but was cancelled before backward and reconciled consistently as cancelled with no required artifact role missing, so it is inconclusive and does not qualify physical batch 12. Do not run the packaged logical-24 probe. Freeze logical 12 / physical 1 / accumulation 12 / concurrent-resident 12 as the fastest qualified configuration; physical 3 is the largest proven memory-safe backward batch but is not recommended.
- [ ] Build and push one fresh 384-target work package using the selected batch and target-normalized schedule.
- [ ] Run the complete OPD experiment from the base E2B model, retain all milestones, and reconcile it consistently.
- [ ] Publish and freshly verify the final private LoRA adapter on Hugging Face.
- [ ] Run sealed 18-case scope and 17-case recovery evaluation jobs sequentially and apply their scientific gates.
- [ ] Finalize both native evaluation artifacts into Policy Prism `evaluation-runs`, validate them, and push the final evidence and living-plan updates.

## Surprises & Discoveries

- Observation: the step-78 and target-98 failures had different causes.
  Evidence: r8 found only one reserve because prompt profile was incorrectly part of reserve compatibility. Plan v2 removed that key and r9 passed target 78. At target 98, r9 had the primary plus two compatible reserves, but the model repeatedly misspelled task-owned source identifiers that the schema had left unconstrained.
- Observation: the dominant target-98 failure was preventable during generation without weakening scientific admission.
  Evidence: ten of eleven inspected structural rejections used a wrong `source_provision_id`, such as omitting a segment from an eCFR identifier. Policy Prism commit `147ac759...` adds a task-specific enum to the deep-copied rules schema. The canonical schema and post-generation admission checks remain unchanged.
- Observation: checkpoint 96 is healthy even though its producing run is not successful.
  Evidence: Trackio artifact `training-models-gemma4-e2b-it-bf16-distill-checkpoint-step-0096:v0` has provider digest `ddde25b171407c76fba02e1b44b23a6531bcede4bd99c7859cc9809496e7f7e7`, PostTrain tree digest `cc53e8330d2af3160d098341dece1c82cf06a559366950f689c72caf24f16dba`, global step 96, and complete resumable state. R9 itself failed before update 99 and was later cancelled to release the GPU.
- Observation: r9's reconciliation is correctly inconsistent rather than a checkpoint-integrity failure.
  Evidence: the provider was explicitly cancelled after the harness raised, while Trackio retained a failed outcome. PostTrain reports `provider cancelled but retained tracking outcome is failed`, no missing artifact role, and retained step-80, step-96, and rollout-trace outputs.
- Observation: batch size one was a branch-local safety guard, not a universal PostTrain constraint.
  Evidence: `_validate_memory_safe_sparse_request` in `packages/train/src/posttrain/train/backends/trl/distillation.py` was deliberately limited to one prompt, one generation, batch one, and accumulation one because no live multi-prompt sparse long-context configuration had been qualified and earlier instructions prohibited smoke tests.
- Observation: the RTX PRO 6000 was underutilized mainly during serial autoregressive student generation and dependency stages.
  Evidence: observed utilization was commonly about 20–22% while approximately 65 GiB of 96 GiB VRAM was allocated. Teacher scoring was usually sub-second; long student generations dominated wall time. More prompt concurrency may fill decode bubbles, but larger training batches also increase peak activation memory and must be measured live.
- Observation: loss variation alone does not show collapse or prove legal improvement.
  Evidence: all inspected losses and gradient norms through step 97 were finite, teacher failures were zero, and isolated spikes recovered immediately under gradient clipping. The median loss over steps 81–96 was lower than earlier windows, which is encouraging, but only sealed evaluations can establish legal quality.
- Observation: the measured batch-one rate is materially slower than the original estimate.
  Evidence: r9 advanced roughly 34 optimizer updates in about 2.5 hours including long heterogeneous generations, implying approximately 27–30 hours for a fresh 384-update batch-one run. Batch-two and batch-four duration must be derived from identical live smokes rather than extrapolated from H200 or E4B/31B runs.
- Observation: changing the optimizer batch invalidates exact continuation from checkpoint 96 for the production comparison.
  Evidence: a batch-two or batch-four run changes targets per optimizer update, scheduler steps, checkpoint step numbers, and sampler advancement. Resuming checkpoint 96 under that schedule would mix two optimization regimes. The checkpoint remains valuable evidence, but the final selected configuration starts from the frozen base E2B model.
- Observation: the Policy Prism target-98 correction initially existed in source but was not selected by the committed production catalog; this launch blocker is closed.
  Evidence: the earlier catalog pinned `2b7a5dc...`. The qualification environment now resolves source/audit commit `411fa6e`, a descendant of target-98 commit `147ac759...`, and the packaged run view records that exact revision.
- Observation: the existing four-task seed is too small, while the earlier 16-target matrix is unnecessary for a 60-minute qualification.
  Evidence: seed 539 resolves `[98, 151, 284, 367]`; it omits target 78 and gives logical batch four only one update. Seed 2907 with 12 tasks resolves `[31, 49, 70, 75, 78, 98, 126, 163, 182, 233, 278, 329]`. It covers both historical boundaries, every stage, every length quartile, all four prompt profiles, four full trajectories, and produces three optimizer updates/twelve physical backward passes at logical batch four. Deterministic reserve failure remains an offline fault-injection test because natural sampling cannot guarantee it.
- Observation: shared reserves exposed a small build-time token-audit mismatch, not a capacity failure.
  Evidence: runtime renders a reserve with its logical primary's profile and shape, while the current audit renders it with the reserve-owned profile. Auditing the 765 actual candidate/profile/shape combinations gives maxima of 8,799 evidence, 10,940 rules, and 9,057 graph tokens; all are safely below 32,768, but the committed graph maximum of 9,034 is not exact.
- Observation: current `origin/main` is a material runtime upgrade, not a drop-in merge.
  Evidence: main pins TRL 1.9.2.post1 at fork commit `a82ecebc...`, selects `IWOPDTrainer`, fixes the fully-on-policy zero-denominator path, supports bounded resident rollout waves, and adds checkpoint-scoped recovery/model views. Main does not contain the feature branch's Gemma tokenizer fingerprint, selected-stage loss projection, XGrammar wire-schema projection, per-call token envelope, memory-safe vocabulary projection, or SQLite allocation-ledger sidecar.
- Observation: logical rollout batch and physical actor batch are independent.
  Evidence: IW-OPD requires `generation_batch_size * num_generations == per_device_train_batch_size * gradient_accumulation_steps`. A safe logical batch of four is therefore physical micro-batch one plus accumulation four. TRL generates four prompts together, then trains four one-sequence micro-slices. Increasing physical batch to four would multiply 49K-token activation memory without being necessary for rollout concurrency.
- Observation: setting `max_concurrent` above one does not by itself prove student batching for Policy Prism.
  Evidence: the current `TrlPolicyGenerator` groups pending requests by maximum tokens and serialized JSON schema, then executes those groups serially. The target-specific source-ID enum makes many rules schemas distinct. Qualification must record actual request-group and resident-wave sizes and must prove that heterogeneous schemas share a bounded vLLM call before a logical batch is credited with concurrency.
- Observation: student prefix caching is not currently a safe acceleration.
  Evidence: the pinned LoRA synchronization path reloads the adapter without an explicit prefix-cache invalidation invariant. A stale prefix after an optimizer update would be scientifically invalid. Student caching remains off unless a regression proves reset after every LoRA sync. Teacher prefix caching remains safe because the teacher never changes.
- Observation: E2B MTP is a real but unqualified candidate.
  Evidence: the model profile pins `google/gemma-4-E2B-it-assistant` at `2d874ef7d29f9a30599a1e4b3c1cbc9595f005df`; its approximately 190 MB assistant declares `Gemma4AssistantForCausalLM`, four layers, and the matching 262,144-token vocabulary. Existing live evidence covers another model/workload, so E2B Policy IW-OPD would require a matched MTP-off/MTP-1 smoke with acceptance and post-sync counters. That comparison is deliberately excluded from this 60-minute qualification and from the resulting production run.
- Observation: the feature branch's linear learning-rate scaling confounds throughput with a new optimizer regime.
  Evidence: the IW-OPD reference configuration uses physical batch one, gradient accumulation, and learning rate `1e-5`. Capacity arms must keep `1e-5` fixed. The previous proposed `2e-5`/`4e-5` values are removed unless a separate learning-rate qualification later justifies them.
- Observation: the corrected qualification crossed the complete model/runtime boundary and produced one valid IW-OPD update before an external fleet outage.
  Evidence: `opdq-fast01b-iwopd-e2b12b-c12-lb4-rseq4-nomtp` admitted targets 31, 70, 75, and 329, including a complete graph trajectory; all selected stages ended with `stop` and passed structural admission. Step one scored 4,725 tokens with zero teacher failures, finite loss `-1.7350585`, finite gradient norm `10.37675`, and one clipped gradient. Trackio retained paired step-one model and recovery artifacts with digests `03d0c961...` and `5e9e6d4d...` plus the allocation ledger.
- Observation: resident four is memory-safe on the target RTX PRO 6000, but heterogeneous Policy schemas limit average decode concurrency.
  Evidence: 1,347 Trackio GPU samples show median utilization 20%, p95 22%, and peak 100%. Median device memory was 72,441,135,104 bytes and peak was 72,722,153,472 bytes (67.73 GiB), leaving about 27.9 GiB against the 96-GiB device. The twelve recorded generation submissions had request/resident sizes `[1,1,3,1,1,4,1,1,1,1,1,1]`; only one group reached four and one reached three because task-specific schemas partition the rest. Step one took 592.70 seconds. Teacher scoring stayed healthy at 181--1,170 ms across four sequences, so student structured autoregressive generation remains the dominant bottleneck.
- Observation: the corrected attempt ended because the shared GPU fleet became unreachable, not because resident four exhausted memory or the runtime failed.
  Evidence: dstack recorded `termination_reason=instance_unreachable`, no container exit status, and no Python/CUDA exception after the finite update. Immediately afterward both `carbonteq-ai-workstation.lan` and the independent RTX 4090 worker were `unreachable=true`, healthy, idle, and holding zero blocks. The abrupt termination also left the Trackio lifecycle at `running`; PostTrain correctly reconciles the provider outcome as failed with incomplete final roles.
- Observation: the maintained heterogeneous-wave path materially changes serving throughput.
  Evidence: the completed logical-four qualification spent 2,112.83 trainer seconds over 12 targets (176.07 seconds/target). With the same 12 target identities, logical batch 12/resident 12 at physical batch one spent 634.11 seconds (52.84 seconds/target), a 3.33x trainer-time speedup. Submission through reconciliation fell from 41.61 to 14.44 minutes, a 2.88x end-to-end speedup. Actual request/resident waves were `[1, 11, 1, 3, 1, 2]`; this is observed batching, not configured concurrency alone.
- Observation: increasing physical backward batch is a memory-capacity question, not the main throughput lever.
  Evidence: physical batch one used 34.74 GiB peak trainer memory and physical batch two used 38.10 GiB, while both runs used the same logical batch 12 and resident capacity 12. System VRAM peaked at 72,454,111,232 bytes (67.48 GiB) in both because the student and frozen-teacher serving allocations dominate. The physical-batch-two run was slower because stochastic outputs required more generated tokens and a reserve, so its wall time cannot be interpreted as a batching regression or gain.
- Observation: physical batch three is safe but provides no throughput benefit on this long-context workload.
  Evidence: the corrected physical-three run admitted the same 12 logical targets and completed one finite update, but spent 1,130.05 seconds versus 634.11 seconds at physical one. It scored 11,803 tokens versus 9,661 and reached 49.83 GiB trainer peak memory. Even after accounting for the 22.2% larger scored-token volume, elapsed training was 78.2% longer while median/p95 GPU utilization remained 21%/22%. Activation batching increases memory and backward cost; heterogeneous student generation remains the dominant latency.
- Observation: physical batch 12 did not establish a backward-memory boundary.
  Evidence: `opdq-ceil04-iwopd-e2b12b-c12-lb12-pb12-rseq12` loaded both models, reached six heterogeneous generation calls with a maximum resident wave of nine, and held about 67.48 GiB system VRAM without a provider or CUDA error. It was cancelled at the user's meeting deadline before teacher scoring or backward, so no physical-12 loss, gradient, trainer peak, checkpoint, or safe/OOM conclusion exists. The largest completed physical backward batch remains three.
- Observation: final harness admission occurred too late to protect a downstream diagnostic call.
  Evidence: physical-batch-two target 126 emitted two non-identical rules with duplicate `rule_id` values. Canonical JSON Schema did not express identity uniqueness, so the isolated executor still requested graph; the graph then referenced an unusable identifier. Policy source `420a554...` applies the same structural dependency criteria before the next stage, preserves the raw rejected rules, skips graph, and lets deterministic reserve replacement proceed.
- Observation: checkpoint 96 shows modest noisy improvement, not demonstrated convergence.
  Evidence: across 96 old sparse-reverse-KL updates, loss mean/median were `0.09788/0.08572`, the fitted slope was only `-2.38e-5` per step, and no value was non-finite. The last 16-step median (`0.06471`) was 23.8% below the first (`0.08488`), token-weighted loss was 11.6% lower, and median gradient norm was 31.7% lower; however, window means oscillated and checkpoint 96 had exposed only 96 of 384 targets. It supports completing one full target pass, not extrapolating a mathematical “fully converged” step.
- Observation: old and new OPD loss values are not numerically comparable.
  Evidence: checkpoint 96 used the earlier `trl.experimental.distillation.DistillationTrainer` fork and a positive sampled sparse reverse-KL plus tail bucket. Current qualification uses pinned `trl==1.9.2.post1`, `IWOPDTrainer`, and prefix-drift importance weighting; valid IW-OPD losses can be negative. The maintained backend also fixes the fully-on-policy denominator, bounds resident vLLM waves, and publishes paired checkpoint/model views. PostTrain retained the Policy selected-stage mask, XGrammar projection, memory-safe LM-head projection, tokenizer identity, and SQLite ledger state around that backend.

## Decision Log

- Decision: preserve all work on the two feature branches and never modify or push either `main` branch.
  Rationale: this isolates the experiment and retains reproducible review boundaries.
  Date/Author: 2026-08-10 / Codex.
- Decision: publish checkpoint 96 as an explicitly intermediate private repository rather than call it the final model.
  Rationale: the weights are valid and useful, but only 96 of 384 targets were optimized and the producing run failed later.
  Date/Author: 2026-08-10 / Codex.
- Decision: qualify checkpoint 96 from its immutable Hugging Face revision instead of adding a generic checkpoint-projection subsystem.
  Rationale: the repository already contains the exact adapter weights at a verified commit. Materializing that pinned Hub adapter directly before vLLM launch preserves identity while avoiding a synthetic model-projection run and unrelated framework surface.
  Date/Author: 2026-08-11 / Codex.
- Decision: keep r0-r9, checkpoints, Trackio traces, and provider evidence; do not delete or relabel failed attempts.
  Rationale: they explain each corrected boundary and are required for honest incident history.
  Date/Author: 2026-08-10 / Codex.
- Decision: fix target-owned identifier copying in Policy Prism's generated rules schema, not by sanitizing output or loosening admission.
  Rationale: the grammar can require the only scientifically valid identifier while canonical validation still evaluates the exact generated response.
  Date/Author: 2026-08-10 / Codex.
- Decision: merge current `origin/main` into only the experiment feature branch; do not rebase, force-push, or modify `main`.
  Rationale: the maintained TRL IW-OPD, runtime locks, bounded rollout waves, and checkpoint artifacts should be adopted together. A merge preserves the prior incident lineage and makes conflict resolution reviewable.
  Date/Author: 2026-08-10 / Codex.
- Decision: use TRL 1.9.2.post1 `IWOPDTrainer` with explicit `distillation_objective=iw_opd`, gamma `0.5`, epsilon `1e-8`, one generation, temperature `1`, top-p `1`, and learning rate `1e-5`.
  Rationale: this is the maintained, fork-tested on-policy backend and the published IW-OPD reference configuration. The former feature loss remains test/reference input while its memory-safe projection is ported to the new sampled-token objective.
  Date/Author: 2026-08-10 / Codex.
- Decision: retain logical four / physical one / accumulation four as the required correctness baseline, then use one direct logical-12 ceiling cohort and one doubled-logical boundary rather than a stepwise parameter sweep.
  Rationale: the required baseline completed all three updates. The user then explicitly extended qualification to find the hardware boundary quickly. Logical 12 reduced measured trainer time 3.33x and proved resident wave 11; testing physical 12 directly brackets the physical ceiling in one attempt, while logical/resident/concurrent 24 tests whether more simultaneous work improves the generation bottleneck. Intermediate physical values are used only to bracket a conclusive OOM, not as a broad optimization sweep.
  Date/Author: 2026-08-11 / Codex.
- Decision: use one deterministic 12-target seed-2907 cohort for the live smoke.
  Rationale: `[31, 49, 70, 75, 78, 98, 126, 163, 182, 233, 278, 329]` covers both prior failures, every stage/profile/quartile, four full trajectories, and three optimizer updates/twelve backward passes. That is the smallest representative cohort satisfying the TRL fork's retained ten-backward live qualification gate.
  Date/Author: 2026-08-10 / Codex.
- Decision: hold prefill, MTP, prefix caching, GPU reservations, LoRA, learning rate, seed, and source plan fixed while probing only logical/physical/accumulation/concurrency/resident capacity.
  Rationale: changing multiple controls would make the timing evidence uninterpretable. MTP still requires a matched post-sync acceptance test, and student prefix caching still lacks a proven invalidation invariant. The ceiling probes therefore isolate only the two relevant capacity dimensions and preserve the scientifically qualified scheduler/model behavior.
  Date/Author: 2026-08-11 / Codex.
- Decision: keep student prefix caching off, teacher prefix caching on, eager execution on, student sleep enabled, TP1, and teacher BF16.
  Rationale: the student changes every optimizer update and lacks a proven LoRA-sync cache reset; the teacher is immutable. CUDA graphs, disabled sleep, teacher quantization/MTP, TurboQuant, and arbitrary attention overrides add memory or scientific risk without addressing the measured generation bottleneck.
  Date/Author: 2026-08-10 / Codex.
- Decision: reduce the live qualification cap from 90 minutes to 60 minutes without adding or shrinking the scientific cohort.
  Rationale: the first valid update took 592.70 seconds and the corrected run reached it about thirteen minutes after submission, projecting three updates within the new cap while retaining targets 78/98 and two post-sync cycles. The 12-target cohort remains the minimum complete live gate; accepting only four targets would hide the exact historical boundaries.
  Date/Author: 2026-08-10 / Codex.
- Decision: do not invoke resident two for an `instance_unreachable` termination.
  Rationale: resident two is permitted only for a conclusive OOM/KV-capacity failure. The retained peak memory leaves substantial headroom, and both independent workers disappeared together. Retrying a slower configuration would not correct network or workstation reachability.
  Date/Author: 2026-08-10 / Codex.
- Decision: select production by wall seconds per admitted target subject to correctness and memory gates; report the largest safe physical batch separately from the fastest production batch.
  Rationale: maximizing allocation or instantaneous utilization can slow the experiment. Physical batch three already uses more trainer memory and takes longer than physical one, while logical batching yields the material speedup. The production choice optimizes end-to-end time without presenting a merely survivable batch as the best configuration.
  Date/Author: 2026-08-11 / Codex.
- Decision: run the final 384-target experiment from the frozen base E2B model after the fixed configuration qualifies.
  Rationale: one consistent batch, learning-rate schedule, sampler, and ledger produces cleaner scientific lineage than mixing checkpoint-96 batch-one history with a new batch.
  Date/Author: 2026-08-10 / Codex.
- Decision: schedule one complete 384-target pass, which is 32 optimizer updates at logical batch 12; do not claim or pre-schedule a “fully converged” step.
  Rationale: the old checkpoint-96 curve is noisy and belongs to a different sparse reverse-KL objective. It proves finite optimization and modest improvement but not an asymptote. Compare checkpoints 8/16/24/32 and the final sealed evaluations; only that evidence can justify a later second pass.
  Date/Author: 2026-08-11 / Codex.
- Decision: stop this goal after the boundary evidence, production configuration, convergence interpretation, and backend comparison are committed and pushed; do not start the full 384-target run.
  Rationale: the user explicitly separates qualification from the next end-to-end execution goal.
  Date/Author: 2026-08-11 / Codex.

## Outcomes & Retrospective

Checkpoint 96 is now a complete, independently downloadable intermediate result rather than an artifact trapped inside a failed run. Its private Hugging Face repository and immutable revision are verified. The previous run did not finish because reserve matching first over-fragmented candidates at target 78, and then the model was allowed to invent an exact source identifier at target 98. Both causes now have narrow, tested corrections: broader scientifically compatible reserve sharing and a task-specific schema enum.

No claim is made that checkpoint 96 is a completed or converged model. Its old sparse reverse-KL metrics improve modestly but noisily: the last 16-step median loss is 23.8% below the first, token-weighted loss is 11.6% lower, and median gradient norm is 31.7% lower, while the fitted loss slope is nearly flat and window means oscillate. It covers only 96 of 384 targets. The defensible next schedule is one complete fresh 384-target pass; neither these metrics nor the new IW-OPD loss scale can justify a numeric “fully converged” step beyond sealed evaluation.

The current IW-OPD stack is operationally qualified. Logical 12 / physical 1 / accumulation 12 completed the representative 12 targets with finite loss and gradient, zero teacher/source/graph/truncation/provider failures, positive scored tokens, complete artifacts, and consistent reconciliation. It is the production choice because it cut measured trainer time from 2,112.83 to 634.11 seconds, a 3.33x speedup, while physical batches two and three were slower. Physical three is the largest completed safe backward batch at 49.83 GiB trainer peak; physical 12 was cancelled before backward and is not a qualified ceiling. The resulting fresh full run is projected at approximately 5.6 trainer hours and 6--7 hours including one initialization, four milestone checkpoints, final upload, and reconciliation; evaluation time remains additional.

The maintained backend was not stable as a blind merge, but its one integration defect was found before production. The old run used Git-pinned `DistillationTrainer` and a positive sampled top-1 sparse reverse-KL/tail loss. The merged runtime uses released `trl==1.9.2.post1`, `IWOPDTrainer`, a prefix-drift importance-weighted objective, a corrected fully-on-policy denominator, bounded vLLM waves, concurrency control, and paired recovery/model checkpoints. PostTrain had to correct its private-hook guard from `_compute_prompt_length` to the pinned backend's `aligned_prompt_length`, then prove loss/gradient equivalence across physical micro-batches. Subsequent finite GPU updates had zero teacher failures. Policy-specific schema projection, selected-stage masking, memory-safe vocabulary projection, tokenizer identity, and SQLite ledger recovery remain explicit PostTrain integrations rather than assumed TRL behavior.

## Context and Orientation

PostTrain is `/home/ali-awais-safdar/Post-Train/posttrain`. It owns model profiles, OPD request contracts, the TRL/IW-OPD runtime adapter, memory-safe loss projection, packaging, dstack execution, Trackio evidence, and Observatory. Policy Prism is `/home/ali-awais-safdar/Policy Prism`. It owns the source-only legal task plan, prompts, schemas, admission, replacement ledger, project catalog/work packages, and final five-file evaluation format.

OPD means that the current student generates a response and the frozen teacher scores the same token IDs. The teacher does not generate a replacement answer. A logical target is one selected evidence, rules, or graph output that receives loss. A reserve is a reviewed alternate source candidate used only when the primary attempt is structurally unusable. Schema-valid legal mistakes remain trainable; malformed JSON, truncation, unknown identifiers, and unusable dependencies are rejected.

The production plan has 384 targets: 77 evidence, 230 rules, and 77 graph. It has 96 shared reserve candidates. Plan v2 matches reserves by target stage, source-length quartile, and reviewed decision class; the logical target supplies prompt profile and task shape. The target-98 rules schema now enumerates the exact source identifier, so XGrammar cannot generate the historically dominant mismatch.

Immutable model inputs are:

    Student: google/gemma-4-E2B-it
    Student revision: 3e22461f65e89153144f8adb70e3b8c2cc9845a7
    Teacher: google/gemma-4-12B-it
    Teacher revision: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
    Canonical token-ID fingerprint: 059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416
    GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GiB

The selected experiment hyperparameters are rank 16, alpha 32, dropout 0, learning rate `1e-5`, IW-OPD gamma `0.5`, IW-OPD epsilon `1e-8`, maximum prompt 32,768 tokens, per-call sequence cap 40,960, trainer maximum length 49,152, gradient clipping 1.0, no warmup, a linear scheduler, gradient checkpointing enabled, and one student generation per prompt. Serving uses logical batch 12, physical actor batch one, accumulation 12, twelve concurrent environment tasks, at most twelve resident student sequences, 4,096-token chunked prefill, eager execution, FP8 KV, LoRA sleep/sync, student prefix caching off, teacher prefix caching on, and MTP off. Physical batch three is the measured safe ceiling, not the selected production batch; physical 12 and logical 24 remain unqualified.

## Plan of Work

### Milestone 1: reconcile the experiment branch with the maintained release

Fetch current `origin/main`, record its SHA, verify both worktrees are clean, and merge it with `--no-ff` into only `feat/gemma-policy-prism-opd-e2b-12b`. Never switch, commit to, push, rebase, or force-update `main`. Use main's release dependency closure, TRL 1.9.2.post1/IW-OPD adapter, Trackio update, bounded resident waves, environment semaphore, and checkpoint-scoped recovery/model artifacts. Build a fresh actual-job image from the merged tree; do not reuse the feature branch's old online-RL image.

Resolve changed-both files deliberately. Preserve or port: the verified E2B/12B tokenizer fingerprint; generation-only XGrammar projection; Gemma whitespace and EOS behavior; per-turn prompt/sequence limits; exact selected-target-stage and loss mask; stable task ordering; native rollout traces; LoRA-only policy synchronization; and Policy Prism SQLite ledger snapshot/restore. Main's actual runtime is Python 3.13.12 even though the historical profile identifier contains `py312`; do not introduce a Python migration. Treat OCI digest and lock digest as authority when the committed framework-version label lags the source version.

### Milestone 2: port the Policy path onto exact IW-OPD

Replace the old `DistillationTrainer` subclass with an `IWOPDTrainer` integration that explicitly records `distillation_objective=iw_opd`, `iw_opd_gamma=0.5`, `iw_opd_epsilon=1e-8`, `lmbda=1`, sampled top-1, and weight-sync frequency one. Port the memory-safe path so it computes the exact IW-OPD sampled-token student log-probabilities from hidden states in bounded LM-head chunks instead of materializing sequence-by-vocabulary logits. Teacher scoring remains the external vLLM exact-token path.

The accumulated loss must have one global valid-token numerator and denominator across all micro-slices. Prove full-reference versus chunked loss and gradients for variable lengths, padding, selected-stage masks, and logical batches 1/2/4/8 expressed as physical batch one plus matching accumulation. Rejected candidates and non-selected dependency/diagnostic stages must contribute exactly zero loss. Fail closed if the pinned TRL private seams or signatures differ from the qualified source commit.

Preserve deterministic Policy target order under buffered IW-OPD generation. Add a bounded heterogeneous-response-format path so distinct task-specific schemas can enter one vLLM submission while retaining one sampling object per prompt and restoring output order. This path must be version-guarded and covered against vLLM 0.25.1; if it cannot be proven, keep serial generation and do not describe `max_concurrent > 1` as batching. A temporary PostTrain adapter override is permitted on this feature branch, but it must be documented as fork-upstream debt rather than silently changing a public contract.

### Milestone 3: finish and pin the Policy Prism source contract

On `feat/scope-opd-e2b-12b-environment-v1`, validate every actual reserve/primary profile/shape materialization rather than a reserve's original profile only. Rebuild the deterministic summary with the measured maxima (currently 8,799 evidence, 10,940 rules, 9,057 graph), rerun the focused plan tests, and commit that source change. Call this immutable commit `POLICY_SOURCE_SHA`.

In a later Policy Prism catalog commit, update the production environment and smoke selection to `POLICY_SOURCE_SHA`; the present `2b7a5dc...` pin is a launch blocker. Add one 12-target seed-2907 selection and its binding/work package. Resolve and assert exactly:

    31, 49, 70, 75, 78, 98,
    126, 163, 182, 233, 278, 329

Every smoke starts from base E2B and uses its own SQLite ledger, output directory, artifact names, and run ID. The production selection remains the same immutable 384 primaries plus 96 reserves; no smoke adapter or ledger is an input to production.

### Milestone 4: pass offline and isolated-image gates

Before reserving the GPU, pass all of these gates:

1. Reject the exact target-78 primary after pre-claiming its former same-profile reserve; prove one cross-profile reserve is uniquely and atomically claimed, rendered with the logical primary's profile/shape, and recovered after restart without duplication.
2. Materialize and compile all 765 actual candidate/profile/shape schemas under the image's XGrammar 0.2.3, not only target 98. Prove every rules schema retains its exact task-owned source-ID enum in canonical and wire copies and rejects the historical misspelling.
3. Bind graph wire-schema references from the admitted rules dependency's exact rule and qualification IDs, while retaining the canonical graph schema for final validation. Prove invented/unknown graph references are rejected before generation can consume a reserve. This is the corresponding preventive fix for the next identifier-copy boundary.
4. Exercise the real reserve pools with temporary ledgers at concurrency four, maximum three replacements, and restart; prove no duplicate claim, SQLite locking error, reserve-profile drift, or known-stratum exhaustion. Keep `uniqueItems`-dependent duplicate/repetition admission checks because XGrammar 0.2.3 cannot enforce them.
5. Prove selected prompt/completion digests match the actual teacher request, rejected tokens receive zero loss, malformed/truncated/duplicate/unknown-ID outputs are not optimized, and every candidate/profile/shape prompt stays within its cap.
6. Prove four heterogeneous schemas retain task order and enter one bounded vLLM request/wave. If the adapter still serializes by schema, the smoke is not started; `max_concurrent=4` alone is not evidence.
7. Build the actual-job image from the merged source and current main published kind, run its isolated import/entry smoke, and verify source, TRL fork, lock, environment, student, and teacher revisions from inside the image.

Historical failure closure is explicit:

| Failure | Required correction | Proof before/live smoke |
| --- | --- | --- |
| XGrammar rejected `uniqueItems` | Remove unsupported keys only from the generation copy; retain canonical validation | Compile all materialized schemas offline; canonical admission remains strict |
| Gemma emitted whitespace/EOS pathologies | Preserve the proven bounded whitespace pattern and grammar-owned stop behavior | Exact renderer/grammar tests plus 12 clean live completions |
| Target 78 exhausted a profile-fragmented reserve pool | Match reserves by stage/quartile/decision and render with primary profile/shape | Forced cross-profile reserve claim, concurrency/restart test, live target 78 |
| Target 98 misspelled `source_provision_id` | Exact task-owned enum in every rules schema and pin the corrected Policy source | All-schema compile/misspelling rejection plus live target 98 |
| A future graph stage invents dependency IDs | Bind graph wire references to admitted rule/qualification IDs | Dynamic-schema unit/fault tests and three live graph targets |
| Fully on-policy loss divided by zero | Adopt the fork's post-generation valid-token denominator | Reference/accumulation tests and three finite live updates |
| Full 49K-by-vocabulary logits risk OOM | Port exact sampled-token IW-OPD through chunked hidden-state projection | Full/chunked loss-gradient equivalence and live peak-memory gate |
| Concurrent replacements duplicate or drift after resume | Atomic SQLite claims plus checkpoint ledger sidecar | Real-pool concurrency/restart tests and matching live ledger |
| `max_concurrent=4` still serialized unique schemas | Heterogeneous per-request schema batching with stable output order | One four-request offline batch and recorded live request/wave size |

### Milestone 5: qualify correctness and the useful hardware ceiling

The required logical-four run completed all 12 seed-2907 targets, three finite updates, target 78 and 98, complete artifacts, and consistent reconciliation in 41.61 minutes. The subsequent fixed-cohort capacity evidence is:

| Logical / physical / accumulation | Result | Trainer seconds | Trainer peak | Useful conclusion |
| --- | --- | ---: | ---: | --- |
| `4 / 1 / 4` | passed, three updates | 2,112.83 | 35.85 GiB | operational baseline; rejected attempts exposed graph-reference debt later fixed in Policy |
| `12 / 1 / 12` | passed, one update | 634.11 | 34.74 GiB | fastest qualified |
| `12 / 2 / 6` | passed, one update | 805.64 | 38.10 GiB | safe, slower |
| `12 / 3 / 4` | passed, one update | 1,130.05 | 49.83 GiB | largest proven safe, slower |
| `12 / 12 / 1` | cancelled before backward | n/a | n/a | inconclusive; not qualified |

Logical 12 formed an observed resident wave of 11 and reduced training time 3.33x versus logical four. Physical batching did not improve the autoregressive generation bottleneck and increased activation memory. Raw device telemetry remains low during decode (typically median 20--21%, p95 22%) even though the logical-12 path materially improves elapsed time.

### Milestone 6: freeze the production configuration

Use the fastest completed configuration for one fresh pass over 384 distinct optimized targets:

| Setting | Production value |
| --- | --- |
| Logical / physical / accumulation | `12 / 1 / 12` |
| Optimizer updates | 32 |
| Learning rate / warmup / scheduler | `1e-5` / 0 / linear |
| Quarter checkpoints | 8, 16, 24, 32 |
| Environment concurrency / resident sequences | `12 / 12` |
| Prefill cap | 4,096 |
| Student / teacher GPU reservation | `0.20 / 0.35` |
| Prefix cache | student off / teacher on |
| MTP | off |

The production work package must use Policy source `420a554...` or a reviewed descendant and start from frozen base E2B, not checkpoint 96 or a qualification adapter. Use a unique run ID that exposes the selected configuration, for example `opdprod01-iwopd-e2b12b-r16-lb12-pb1-rseq12-scope384`.

### Milestone 7: run and reconcile full OPD

Submit one full job from frozen base E2B with a 432,000-second timeout. Remain attached through model loading, corrected XGrammar compilation, at least one admitted logical batch, one finite update, positive scored tokens, zero teacher failures, and the first complete paired checkpoint plus ledger. Then monitor provider state, admitted unique targets, loss, gradient norm, actual rollout waves, generation throughput, replacement pressure by stratum, trace synchronization, peak VRAM, and every quarter checkpoint.

At completion require provider success, PostTrain reconciliation `consistent`, exactly 384 unique optimized targets, finite metrics, zero teacher failures, no duplicate reserve or logical target, complete native traces, the final adapter/model view, recovery view, ledger, and all quarter checkpoints. A valid but surprising loss is retained for sealed qualification; an operational or structural failure is not silently accepted. Resume an infrastructure interruption only from an exact complete checkpoint with the identical frozen configuration and a new run ID.

### Milestone 8: publish the final adapter

Materialize the exact final Trackio model artifact into ignored local state, verify provider/PostTrain tree digests, PEFT rank/alpha/dropout, base repository/revision, safetensors loadability, objective/configuration, and trainer/checkpoint lineage. Add a model card containing data/environment lineage, exact IW-OPD and runtime settings, optimization evidence, intended use, limitations, failed-attempt history, and evaluation status.

Create or update only the private repository `carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-lora-v1`. Fresh-download the returned immutable 40-character revision into a different ignored directory and require byte-identical model card, adapter config, and weights. Record both Trackio and Hugging Face identities in the Policy Prism model selection.

### Milestone 9: qualify sequentially and finalize evidence

Add an exact adapter model selection, vLLM LoRA evaluation binding with maximum rank 16 and 131,072 context, and two domain-evaluation work packages in project `policy-prism-scope-opd-e2b-12b`. Run scope first under `opdprod01-eval-scope-v11`; require 18 expected/included traces, zero failures/truncations/errors, complete Claude judging, one native evaluation artifact, and consistent reconciliation. Only after it releases the GPU run recovery under `opdprod01-eval-recovery-v1`; require the same gates for 17 traces.

Materialize the native artifacts into ignored `.posttrain/state/native-evals/<run-id>/` directories. Generate serving metadata from PostTrain evidence, finalize directly into `Policy Prism/evaluation-runs`, require the standard `manifest.json`, `traces.jsonl`, `business-kpis.json`, `engineering-metrics.json`, and `semantic-diagnostics.json`, then run `validate-runs`. Compare scope only with compatibility-matching scope runs and recovery only with recovery runs. Update `evaluation-runs/catalog.json`, commit/push final Policy Prism evidence, and update/push this living plan on the PostTrain feature branch.

## Concrete Steps

The commands below are executed by the later goal. They are grouped by working directory and do not print secrets.

First, in the PostTrain terminal, freeze refs and merge main into the feature branch only:

    export POSTTRAIN_ROOT=/home/ali-awais-safdar/Post-Train/posttrain
    export POLICY_ROOT="/home/ali-awais-safdar/Policy Prism"
    export KIT=/home/ali-awais-safdar/Post-Train/posttrain-setup-v0.2.2-20260728/posttrain-setup
    export POSTTRAIN_ENV_FILE="$POLICY_ROOT/.env.posttrain"

    cd "$POSTTRAIN_ROOT"
    git switch feat/gemma-policy-prism-opd-e2b-12b
    test -z "$(git status --porcelain)"
    git fetch origin main feat/gemma-policy-prism-opd-e2b-12b
    test "$(git rev-parse HEAD)" = "$(git rev-parse origin/feat/gemma-policy-prism-opd-e2b-12b)"
    git rev-parse origin/main
    git merge --no-ff origin/main -m "merge: align Policy Prism OPD with current main"

The pre-merge `origin/main` expectation is `6ffe634432a3f92e8c6dd561d3cd85b2b2ba45cd`; if it changes, re-audit the delta before merging. Resolve conflicts using the ownership rules in Milestones 1 and 2, run focused tests, commit only on the feature branch, and push only that branch. Never use `git reset --hard`, rebase, or force push.

From Policy Prism, complete the source audit before editing catalog pins:

    cd "$POLICY_ROOT"
    git switch feat/scope-opd-e2b-12b-environment-v1
    test -z "$(git status --porcelain)"
    test "$(git rev-parse HEAD)" = 147ac75997579f08154145ea9bdc6215b4aa7ec4

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync pytest \
      packages/normative-verifiers/tests/test_scope_opd.py
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync ruff check \
      packages/normative-verifiers/src/policy_prism_normative_verifiers \
      packages/normative-verifiers/tests/test_scope_opd.py
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync mypy --strict \
      packages/normative-verifiers/src/policy_prism_normative_verifiers
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers build-scope-opd-plan \
      --registry packages/normative-verifiers/src/policy_prism_normative_verifiers/resources/scope_opd_plan/review-registry.json \
      --output-dir packages/normative-verifiers/src/policy_prism_normative_verifiers/resources/scope_opd_plan \
      --tokenizer google/gemma-4-E2B-it \
      --check
    git diff --check

After the actual-materialization audit change passes, commit and push it, capture `POLICY_SOURCE_SHA=$(git rev-parse HEAD)`, then make a separate catalog/work-package commit that pins every OPD environment to that exact 40-character SHA. Validate locally that no `2b7a5dc` reference remains in resolved OPD seats.

From PostTrain, run the merged IW-OPD focused ladder before the broad suite:

    cd "$POSTTRAIN_ROOT"
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync pytest \
      packages/train/tests/test_api.py \
      packages/train/tests/test_trl_sparse_distillation.py \
      packages/train/tests/test_trl_online_rl.py \
      packages/train/tests/test_verifiers_grpo_bridge.py \
      packages/train/tests/test_trl_checkpoint_artifacts.py \
      packages/train/tests/test_checkpoints.py
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync ruff check \
      packages/train/src packages/train/tests
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync pyright \
      packages/train/src
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync lint-imports
    git diff --check

The exact test names may move during the merge, but the accepted behaviors may not be dropped: IW-OPD zero-denominator correction; full/chunked loss and gradient equivalence at accumulation four; selected-stage masks; heterogeneous schema ordering and bounded waves; XGrammar canonical/wire separation; E2B/12B fingerprint; paired checkpoints and SQLite sidecar restore; and failure retention. Then run the repository's locked sync and relevant broad suite before packaging:

    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv sync --all-packages --locked --python 3.13
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run ruff check .
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run pyright
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run lint-imports
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run pytest
    git diff --check

Configure the control shell only after code and catalog commits are immutable:

    cd "$POSTTRAIN_ROOT"
    set -a
    . "$KIT/bundle/posttrain.env"
    . /home/ali-awais-safdar/.config/posttrain/credentials/huggingface.env
    . "$POSTTRAIN_ENV_FILE"
    set +a

    pt() {
      XDG_CONFIG_HOME=/tmp/posttrain-opd-empty-config \
      UV_CACHE_DIR=/tmp/posttrain-uv-cache \
      uv run --no-sync --package posttrain posttrain \
        --project-root "$POLICY_ROOT" \
        --env-file "$POSTTRAIN_ENV_FILE" \
        "$@"
    }

    test -n "$HF_TOKEN"
    test -n "$DSTACK_TOKEN"
    pt doctor
    pt catalog validate
    pt workers

Expected preflight is a valid catalog, healthy internal services, and `carbonteq-ai-workstation.lan` healthy with one idle RTX PRO 6000 block. Do not expose credential values. Ensure at least 80 GiB local free space before rebuilding; current free space was approximately 89 GiB after trash cleanup.

Qualification is complete; do not rerun or resume any qualification adapter. The authoritative completed runs are `opdq-fast01c-iwopd-e2b12b-c12-lb4-rseq4-nomtp`, `opdq-ceil01-iwopd-e2b12b-c12-lb12-pb1-rseq12`, and `opdq-ceil03-iwopd-e2b12b-c12-lb12-pb3-rseq12`. The cancelled physical-12 run and packaged logical-24 package are evidence only.

Under the later end-to-end goal, first add a production settings selection and work package that exactly encode logical 12 / physical 1 / accumulation 12, `max_steps=32`, checkpoints `8/16/24/32`, Policy source `420a554...` or a reviewed descendant, environment concurrency 12, and rollout resident capacity 12. Then validate, package, and start it:

    export OPD_RUN=opdprod01-iwopd-e2b12b-r16-lb12-pb1-rseq12-scope384
    export OPD_PACKAGE=gemma4_e2b_scope_opd_iwopd_scope384_final.yaml

    pt catalog validate
    pt work-package validate "$OPD_PACKAGE"
    pt job plan "$OPD_PACKAGE" --job distill
    pt job pack "$OPD_PACKAGE" --job distill --build-missing
    pt job run "$OPD_PACKAGE" \
      --job distill \
      --provider dstack \
      --env HF_TOKEN \
      --timeout-seconds 43200 \
      --run-id "$OPD_RUN"
    pt run status "$OPD_RUN"
    pt run logs "$OPD_RUN" --follow

After terminal success:

    pt run wait "$OPD_RUN" --timeout-seconds 43200
    pt run reconcile "$OPD_RUN"
    pt --json run show "$OPD_RUN"

Publish only the reconciled final model artifact, not a smoke adapter. Materialize it to an ignored export directory, verify the tree, write the model card, and then:

    export HF_MODEL_REPO=carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-lora-v1
    export ADAPTER_DIR="$POLICY_ROOT/.posttrain/state/exports/$OPD_RUN/adapter"

    test -f "$ADAPTER_DIR/adapter_config.json"
    find "$ADAPTER_DIR" -type f -name '*.safetensors' -print -quit | grep -q .
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync --package posttrain-train \
      hf repos create "$HF_MODEL_REPO" --repo-type model --private --exist-ok
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync --package posttrain-train \
      hf upload "$HF_MODEL_REPO" "$ADAPTER_DIR" . --repo-type model --private \
      --commit-message "Publish Policy Prism Gemma 4 E2B from 12B IW-OPD adapter"
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync --package posttrain-train \
      python -c 'import os; from huggingface_hub import HfApi; print(HfApi().model_info(os.environ["HF_MODEL_REPO"]).sha)'

Fresh-download that returned revision into a different ignored directory and compare file hashes before registering the evaluation model.

Run sealed evaluations sequentially after registration:

    export SCOPE_RUN=opdprod01-eval-scope-v11
    export RECOVERY_RUN=opdprod01-eval-recovery-v1

    pt job pack gemma4_e2b_scope_opd_from_12b_scope_eval.yaml \
      --job evaluate --build-missing
    pt job run gemma4_e2b_scope_opd_from_12b_scope_eval.yaml \
      --job evaluate --provider dstack --env HF_TOKEN --env OPENROUTER_API_KEY \
      --timeout-seconds 21600 --run-id "$SCOPE_RUN"
    pt run logs "$SCOPE_RUN" --follow
    pt run wait "$SCOPE_RUN" --timeout-seconds 21600
    pt run reconcile "$SCOPE_RUN"

Require the 18-case scientific gate and GPU release before packing/submitting recovery. Then run the analogous recovery package and require 17 completed cases with zero failures.

Finally, from Policy Prism, materialize each native artifact, generate serving metadata from its run view, and finalize directly:

    cd "$POLICY_ROOT"
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers finalize-run \
      --input "$SCOPE_NATIVE" \
      --run-id gemma-4-e2b-policy-prism-iwopd-r16-from-12b-v1-v11-sealed-scope \
      --serving-metadata "$SCOPE_SERVING_METADATA" \
      --output-root "$POLICY_ROOT/evaluation-runs"

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers finalize-run \
      --input "$RECOVERY_NATIVE" \
      --run-id gemma-4-e2b-policy-prism-iwopd-r16-from-12b-v1-v11-sealed-recovery \
      --serving-metadata "$RECOVERY_SERVING_METADATA" \
      --output-root "$POLICY_ROOT/evaluation-runs"

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers validate-runs --root "$POLICY_ROOT/evaluation-runs"

Only after fresh HF verification and `validate-runs` pass should final evidence and plan outcomes be committed and pushed to their respective feature branches.

## Validation and Acceptance

Checkpoint preservation is accepted because the private HF repository is private, resolves to immutable revision `4f1fe9c75031396a11bcc44e2193f96df9003054`, and a fresh download matched all twelve uploaded checkpoint files byte-for-byte. The original Trackio artifact remains the lineage authority.

The reserve corrections are accepted offline only when plan v2 still has exactly 384 primaries and 96 reserves, all target distributions and immutable selection hashes remain valid, target 78 can claim the broader shared pool, every rules task schema enumerates its exact `source_provision_id`, the canonical source schema is not mutated, malformed/unknown outputs are still rejected, and all actual reserve/profile/shape tokenizations are audited. The resolved job must name `POLICY_SOURCE_SHA`, never the stale `2b7a5dc...` revision.

The IW-OPD integration is accepted offline only when the pinned TRL source and private seams match, full-reference and chunked losses/gradients agree through accumulation 1/2/4/8, the global token denominator is nonzero and exact, rejected/dependency/diagnostic tokens have zero loss, heterogeneous schemas retain order in bounded waves, and checkpoint/model/ledger views resume atomically. Passing the former sparse reverse-KL tests without these IW-OPD tests is insufficient.

A runtime configuration is supported by two complementary live gates on the exact model, teacher, IW-OPD objective, and engine: the logical-four run proves three finite updates and repeated post-sync rollout cycles, while logical 12 / physical 1 proves the selected one-update capacity and measured resident wave 11. The record includes seconds per admitted target, actual request/wave sizes, utilization, peak memory, loss/gradient evidence, teacher failures, replacements, artifacts, and consistent reconciliation. Physical 12 and logical 24 are explicitly not qualified.

The full run is accepted only with all 384 target identities optimized once, finite loss and gradient norms, zero teacher failures, complete native trace synchronization, no duplicate reserve/target, matching SQLite ledger, final model/recovery views and all quarter artifacts, and consistent PostTrain reconciliation. A provider `succeeded` state without these gates is insufficient.

Scope evaluation requires 18 expected and included traces; recovery requires 17. Both require zero failures, truncations, and trace errors, complete Claude semantic judging, one native `verifiers-evaluation` output, and consistent reconciliation. Finalized directories must contain exactly the five standard evidence files, update `evaluation-runs/catalog.json`, and pass `validate-runs`.

## Idempotence and Recovery

Validation and content-addressed packing may be rerun. Never reuse a run ID after provider admission. If a smoke fails operationally before model execution, fix the proven infrastructure cause and use an incremented suffix; do not reinterpret it as a batch failure. If it fails by OOM or numerical instability, reject that batch and continue with the next smaller candidate.

The final run starts from base weights. If infrastructure fails after a valid quarter checkpoint, resume only from the exact immutable checkpoint under a new run ID and only with identical batch, schedule, model, teacher, environment, plan, image, and ledger digests. Scientific failures such as exhausted reserves, invalid traces, NaN/Inf, or systematic teacher failure require diagnosis and a new fresh run; they are not automatically resumed.

Do not delete r0-r9, checkpoint 96, Trackio artifacts, native traces, finalized evaluation directories, or Hugging Face repositories. Cleanup is limited to new provider workspaces after all remote artifacts and five-file outputs are verified.

## Artifacts and Notes

Historical r9 identity:

    PostTrain run: opda09rs-e2b12b-r16-resume64-scope384-v2
    Trackio display: train.distill-opda09rs
    Trackio internal run ID: d9568a8d94e4433283b507635e341239
    Provider run: pt-5dfbcf7a92d8c14a24439628
    Terminal status: failed; provider terminated after explicit cancellation
    Retained: step-80 checkpoint, step-96 checkpoint, native rollout traces

Checkpoint 96 identity:

    Trackio project: policy-prism-scope-opd-e2b-12b
    Artifact: training-models-gemma4-e2b-it-bf16-distill-checkpoint-step-0096:v0
    Trackio manifest digest: ddde25b171407c76fba02e1b44b23a6531bcede4bd99c7859cc9809496e7f7e7
    PostTrain tree digest: cc53e8330d2af3160d098341dece1c82cf06a559366950f689c72caf24f16dba
    Size: 322,632,497 bytes
    HF repository: carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-checkpoint-96
    HF revision: 4f1fe9c75031396a11bcc44e2193f96df9003054
    Visibility: private

The fixed 12-target qualification cohort selected by seed 2907 is:

    31, 49, 70, 75, 78, 98,
    126, 163, 182, 233, 278, 329

It contains seven rules, two evidence, three graph, all four prompt profiles, all four source-length quartiles, four full trajectories, two constructed-incomplete targets, and both historical failure boundaries. At logical batch four it produces three updates/twelve backward passes. Offline deterministic faults, not natural sampling, prove reserve recovery.

The required logical-four qualification completed in 41.61 minutes. The selected logical-12 run completed in 14.44 minutes submission-to-reconciliation and used 634.11 trainer seconds for 12 targets. Its measured rate projects `634.11 / 12 * 384 = 20,291.40` trainer seconds, or 5.64 hours, for one full pass. Budget 6--7 hours for training, initialization, four milestone checkpoints, artifact publication, and reconciliation. Allow another 2--6 hours afterward for Hugging Face publication, two sequential sealed evaluations, Claude judging, and five-file finalization. This is a measured workload estimate, not a guarantee; stochastic completion lengths and reserve consumption remain the largest variance.

## Interfaces and Dependencies

`posttrain.train.backends.trl.distillation` owns the qualified IW-OPD adapter, exact dependency guard, memory-safe sampled-token projection, teacher scoring, and global accumulated loss. `TrlPolicyGenerator` in `packages/train/src/posttrain/train/backends/trl/online_rl.py` owns per-turn rendering, task-specific response formats, and heterogeneous bounded submissions. `posttrain.train.integrations.verifiers.VerifiersEnvironmentRolloutBridge` owns task selection, selected-stage projection, concurrent exact-token rollouts, and stable ordering. The pinned TRL fork owns the generic IW-OPD buffer, post-generation denominator correction, LoRA synchronization/sleep lifecycle, bounded resident waves, and speculative counters.

Policy Prism `scope_opd_tasks.py` owns task-specific generated schemas; `scope_opd_admission.py`, `harness.py`, and `scope_opd_ledger.py` own structural admission, reserve allocation, and resumable target identity. PostTrain must not import Policy Prism into reusable packages. Policy Prism must not weaken canonical validation to accommodate vLLM.

External services are Hugging Face for gated base models and private adapter publication, the Live Kit OCI registry for content-addressed images, dstack for GPU placement, Trackio for metrics/traces/artifacts, OpenRouter for Claude judging, and Observatory for read-only inspection. Credentials remain in permission-protected environment files and never enter commits or terminal output.

Revision note (2026-08-11): the logical-four correctness run passed, then the user requested a direct capacity boundary rather than a broad sweep. Logical 12 / physical 1 / accumulation 12 is frozen as the fastest qualified production configuration after a 3.33x measured trainer-time improvement. Physical batches two and three passed but slowed the workload; three is the largest proven safe backward batch. Physical 12 was cancelled before backward at the meeting deadline and logical 24 was packaged but deliberately not submitted. MTP, student prefix caching, prefill changes, and GPU-reservation changes remain excluded.
