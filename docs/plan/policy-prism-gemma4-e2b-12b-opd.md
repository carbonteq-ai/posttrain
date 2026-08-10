# Qualify and run Policy Prism Gemma 4 E2B from 12B OPD

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated whenever work advances. Maintain it according to `docs/templates/PLAN.md` and the frozen product baseline under `docs/post-training/`.

## Purpose / Big Picture

This plan delivers a reproducible and throughput-qualified Policy Prism on-policy distillation (OPD) experiment on the in-house RTX PRO 6000. Gemma 4 E2B is the trainable student and Gemma 4 12B is the frozen teacher. The student generates legal-interpretation responses, the teacher scores the exact generated token IDs, and the maintained TRL 1.9.2.post1 Importance-Weighted OPD (IW-OPD) objective updates a rank-16 LoRA adapter. IW-OPD uses the sampled-token student and teacher log-probabilities and prefix-drift weighting; it is a deliberate replacement for the feature branch's older sparse reverse-KL implementation, not a rename of the same loss.

The previous attempts produced a valid intermediate checkpoint at optimizer step 96 but did not complete training. This plan preserves that checkpoint as historical evidence, merges current `origin/main` into only the experiment feature branch, ports the Policy-specific correctness and memory-safety deltas onto the maintained IW-OPD backend, and runs one research-selected integrated GPU smoke under a hard 60-minute limit. Offline and isolated-image gates exhaustively validate the historical structural failures; the live smoke verifies the combined models, three policy updates, two post-sync rollout cycles, heterogeneous batching, memory, artifacts, and throughput. Completion means that all 384 logical targets were optimized once under one frozen configuration, the final adapter and checkpoints are retained in Trackio, the adapter is privately published and freshly verified on Hugging Face, sealed scope and recovery evaluations succeed, and both evaluations are finalized in Policy Prism's normal five-file `evaluation-runs` format.

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
- [x] (2026-08-10) Cancel the failed r9 provider workspace after checkpoint preservation. It is terminal with provider state `terminated`; reconciliation records failed/inconsistent evidence and retains three training artifacts plus native rollout traces.
- [x] (2026-08-10) Stop before new GPU submissions and update this plan for a separate batch-qualification and full-run execution goal.
- [x] (2026-08-10) Fetch and compare current `origin/main` at `6ffe634432a3f92e8c6dd561d3cd85b2b2ba45cd`, the experiment branch at `dbb5c7b5538124913b99620801339f702c58089e`, and the immutable TRL 1.9.2.post1 fork source at `a82ecebc0fa081efd58302a34a553445fc73271d` without changing either branch.
- [x] (2026-08-10) Audit the target-78 and target-98 Policy Prism fixes, discover the stale production source pin, identify the shared-reserve prompt-budget audit gap, and select the representative 12-target seed-2907 live cohort.
- [x] (2026-08-10) Audit logical rollout batching, physical actor batching, vLLM resident waves, chunked prefill, prefix caching, MTP, sleep mode, KV dtype, and CUDA execution mode; select one research-grounded production candidate rather than a live parameter sweep.
- [x] (2026-08-10) Merge `origin/main` into only `feat/gemma-policy-prism-opd-e2b-12b`, adopt the release/runtime/checkpoint structure, and port the enumerated Policy-specific deltas onto TRL IW-OPD; merge commit `7926e87` and qualification-boundary commit `5b1b87d` are pushed.
- [x] (2026-08-10) Complete Policy Prism's actual-materialization token audit in source commit `411fa6e`, pin the qualification environment to that exact source, and add the resident-two emergency work package in catalog commit `77506c6`.
- [x] (2026-08-10) Pass offline IW-OPD loss/gradient and accumulation equivalence, target-78 forced cross-profile reserve recovery, exact target-98 source enums, all actual-schema XGrammar compilation, concurrent/restart-safe real-ledger claims, selected-stage masking, heterogeneous batching, and checkpoint-ledger gates. The 765 distinct candidate/profile/shape materializations expand to 1,293 stage schemas, all compiled under XGrammar 0.2.3.
- [x] (2026-08-10) Package corrected resident-four and resident-two actual-job images. After the first provider canary exposed a fail-closed private-hook name mismatch, align the exact pinned IW-OPD hook to `aligned_prompt_length`, pass its real-class regression, push PostTrain commit `e8fbd51`, and repack immutable images `sha256:b646c50e...` and `sha256:4b83bea8...` respectively.
- [ ] Run one fixed 12-target, three-update integrated GPU smoke and complete its scientific/runtime reconciliation within a hard 60-minute window. Corrected attempt `opdq-fast01b-iwopd-e2b12b-c12-lb4-rseq4-nomtp` reached one finite update before dstack terminated it as `instance_unreachable`; both fleet workers then became unreachable. This is retained infrastructure evidence, not a qualified run.
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

## Decision Log

- Decision: preserve all work on the two feature branches and never modify or push either `main` branch.
  Rationale: this isolates the experiment and retains reproducible review boundaries.
  Date/Author: 2026-08-10 / Codex.
- Decision: publish checkpoint 96 as an explicitly intermediate private repository rather than call it the final model.
  Rationale: the weights are valid and useful, but only 96 of 384 targets were optimized and the producing run failed later.
  Date/Author: 2026-08-10 / Codex.
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
- Decision: qualify one production candidate: logical batch four, physical actor batch one, accumulation four, resident vLLM capacity four, and no MTP.
  Rationale: logical four attacks the measured serial-generation bottleneck while one-sequence backward slices bound 49K-token activation memory. The maintained fork's 4,096-token prefill chunks and bounded waves are the researched scheduler choice. Live b1/b2/b8 and physical-batch sweeps do not fit the user's 60-minute budget and are not required to prove the selected path. This is the best defensible configuration under the fixed qualification budget, not a claim that an untested point is the global throughput optimum.
  Date/Author: 2026-08-10 / Codex.
- Decision: use one deterministic 12-target seed-2907 cohort for the live smoke.
  Rationale: `[31, 49, 70, 75, 78, 98, 126, 163, 182, 233, 278, 329]` covers both prior failures, every stage/profile/quartile, four full trajectories, and three optimizer updates/twelve backward passes. That is the smallest representative cohort satisfying the TRL fork's retained ten-backward live qualification gate.
  Date/Author: 2026-08-10 / Codex.
- Decision: do not spend this qualification window on prefill, MTP, prefix-cache, physical-batch, or GPU-reservation A/B runs.
  Rationale: MTP requires a matched off/on test and post-LoRA-sync evidence; student prefix caching lacks a proven invalidation invariant; physical batches multiply long-context activations; and a prefill sweep consumes most of the 60-minute budget. Use the maintained fork default `max_num_batched_tokens=4096`, MTP off, student cache off, and the already-safe teacher cache. MTP can be a later independent acceleration experiment after this OPD result.
  Date/Author: 2026-08-10 / Codex.
- Decision: keep student prefix caching off, teacher prefix caching on, eager execution on, student sleep enabled, TP1, and teacher BF16.
  Rationale: the student changes every optimizer update and lacks a proven LoRA-sync cache reset; the teacher is immutable. CUDA graphs, disabled sleep, teacher quantization/MTP, TurboQuant, and arbitrary attention overrides add memory or scientific risk without addressing the measured generation bottleneck.
  Date/Author: 2026-08-10 / Codex.
- Decision: reduce the live qualification cap from 90 minutes to 60 minutes without adding or shrinking the scientific cohort.
  Rationale: the first valid update took 592.70 seconds and the corrected run reached it about thirteen minutes after submission, projecting three updates within the new cap while retaining targets 78/98 and two post-sync cycles. The 12-target cohort remains the minimum complete live gate; accepting only four targets would hide the exact historical boundaries.
  Date/Author: 2026-08-10 / Codex.
- Decision: do not invoke resident two for an `instance_unreachable` termination.
  Rationale: resident two is permitted only for a conclusive OOM/KV-capacity failure. The retained peak memory leaves substantial headroom, and both independent workers disappeared together. Retrying a slower configuration would not correct network or workstation reachability.
  Date/Author: 2026-08-10 / Codex.
- Decision: qualify the fixed resident-four configuration by wall seconds per admitted target; resident two is an emergency capacity fallback, not a comparison arm.
  Rationale: throughput is the user goal, but the 60-minute budget permits one integrated qualification rather than a sweep. GPU utilization is supporting evidence, not the primary objective.
  Date/Author: 2026-08-10 / Codex.
- Decision: run the final 384-target experiment from the frozen base E2B model after the fixed configuration qualifies.
  Rationale: one consistent batch, learning-rate schedule, sampler, and ledger produces cleaner scientific lineage than mixing checkpoint-96 batch-one history with a new batch.
  Date/Author: 2026-08-10 / Codex.
- Decision: stop the current goal after checkpoint preservation and this launch-ready plan; submit no new GPU job until a new execution goal begins.
  Rationale: the revised user objective explicitly separates planning from the next experiment.
  Date/Author: 2026-08-10 / Codex.

## Outcomes & Retrospective

Checkpoint 96 is now a complete, independently downloadable intermediate result rather than an artifact trapped inside a failed run. Its private Hugging Face repository and immutable revision are verified. The previous run did not finish because reserve matching first over-fragmented candidates at target 78, and then the model was allowed to invent an exact source identifier at target 98. Both causes now have narrow, tested corrections: broader scientifically compatible reserve sharing and a task-specific schema enum.

No claim is made that checkpoint 96 is a completed or qualified model. The corrected IW-OPD qualification proved the selected resident-four configuration can load both models, execute heterogeneous structured rollouts, score exact tokens, backpropagate, and publish a paired checkpoint without OOM. It remains unqualified because an external fleet reachability failure stopped the job after 4/12 targets and 1/3 updates, before targets 78 and 98. Once the workstation is reachable, the only valid next GPU action is a fresh resident-four run with a new ID and the same immutable image/configuration; resident two is not indicated. A passing run within 60 minutes freezes the fresh full-run configuration.

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

The selected experiment hyperparameters are rank 16, alpha 32, dropout 0, learning rate `1e-5`, IW-OPD gamma `0.5`, IW-OPD epsilon `1e-8`, maximum prompt 32,768 tokens, per-call sequence cap 40,960, trainer maximum length 49,152, gradient clipping 1.0, no warmup, a linear scheduler, gradient checkpointing enabled, and one student generation per prompt. Serving uses logical batch four, physical actor batch one, accumulation four, four concurrent environment tasks, at most four resident student sequences, 4,096-token chunked prefill, eager execution, FP8 KV, LoRA sleep/sync, student prefix caching off, teacher prefix caching on, and MTP off.

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

### Milestone 5: run one integrated smoke within 60 minutes

Run exactly one required GPU job from fresh base weights. It uses seed 2907's 12 targets, logical batch four, physical batch one, accumulation four, and therefore three optimizer updates with twelve physical backward passes. It exercises two generations after LoRA synchronization, meeting the maintained TRL fork's ten-backward live gate without a separate confirmation run.

The 60-minute limit is the complete provider/GPU qualification window, from submission through reconciliation. Source integration, CPU/unit fault tests, image construction, model-access checks, and job packing happen beforehand without holding the GPU. The single live run then verifies that all of those corrections work together on the real hardware within the same 60-minute window.

The frozen smoke configuration is:

| Control | Value |
| --- | --- |
| Next run ID | `opdq-fast01c-iwopd-e2b12b-c12-lb4-rseq4-nomtp` |
| Targets / optimizer updates | 12 / 3 |
| Logical / physical / accumulation | `4 / 1 / 4` |
| Environment concurrency | 4 |
| Student resident sequences | 4 |
| Student prefill cap | 4,096 tokens |
| Student / teacher GPU reservation | `0.20 / 0.35` |
| Student / teacher KV | FP8 / FP8 |
| Chunked prefill / eager / sleep | on / on / on |
| Student / teacher prefix cache | off / on |
| MTP | off |
| LoRA | rank 16 / alpha 32 / dropout 0 |
| IW-OPD | gamma 0.5 / epsilon `1e-8` / LR `1e-5` |
| Checkpoints | every update / retain latest 2 |

Pack the primary configuration and a resident-two fallback before reserving the GPU. The 60-minute clock starts at provider submission. Time gates are: engine ready by minute 10, first admitted batch by minute 15, first finite update by minute 20, terminal training by minute 52, and reconciliation/scientific gate by minute 60. Set the provider job timeout to 3,120 seconds so a hung job leaves eight minutes for reconciliation. Submit the resident-two fallback only when the primary fails conclusively during the first 15 minutes from a KV-capacity/OOM condition; an infrastructure outage is not a fallback trigger.

Acceptance requires all 12 target identities exactly once; three finite optimizer updates and at least twelve backward passes; successful post-sync rollouts; target 78 and 98 completion; zero source-ID or graph-reference mismatch; zero truncation/provider/teacher error; positive scored-token count; no duplicate reserve; complete native trace sync; one complete paired checkpoint/model view with its SQLite ledger; and consistent reconciliation. Record end-to-end seconds per admitted target, actual heterogeneous request and resident-wave sizes, rollout tokens/s, actor/teacher/sync/sleep timing, median/p95 utilization, peak VRAM, KV preemptions, replacement reasons, and artifact digests. Raw GPU utilization alone cannot pass or fail the run.

### Milestone 6: freeze the production configuration

If the integrated smoke passes, use the identical runtime and optimization settings for production. One epoch is 384 distinct optimized targets:

| Setting | Production value |
| --- | --- |
| Logical / physical / accumulation | `4 / 1 / 4` |
| Optimizer updates | 96 |
| Learning rate / warmup / scheduler | `1e-5` / 0 / linear |
| Quarter checkpoints | 24, 48, 72, 96 |
| Resident sequences | 4, or 2 only if the accepted fallback established it |
| Prefill cap | 4,096 |
| Student / teacher GPU reservation | `0.20 / 0.35` |
| Prefix cache | student off / teacher on |
| MTP | off |

Record the smoke evidence, exact PostTrain/Policy commits, dependency/OCI/selection digests, and final resident-wave choice in this document before submission. The production run ID is `opdprod01-iwopd-e2b12b-r16-lb4-nomtp-scope384`. Start from frozen base E2B, not checkpoint 96 or the smoke adapter.

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

Package the primary smoke and its resident-two fallback before starting the 60-minute clock:

    export SMOKE_PACKAGE=gemma4_e2b_scope_opd_from_12b_qualify_c12_lb4.yaml
    export FALLBACK_PACKAGE=gemma4_e2b_scope_opd_from_12b_qualify_c12_lb4_rseq2.yaml
    export SMOKE_RUN=opdq-fast01c-iwopd-e2b12b-c12-lb4-rseq4-nomtp

    for package in "$SMOKE_PACKAGE" "$FALLBACK_PACKAGE"; do
      pt work-package validate "$package"
      pt job plan "$package" --job distill
      pt job pack "$package" --job distill --build-missing
    done

Each pack must report one immutable actual-job image and isolated-runtime success. Start the clock at this submission:

    pt job run "$SMOKE_PACKAGE" \
      --job distill \
      --provider dstack \
      --env HF_TOKEN \
      --timeout-seconds 3120 \
      --run-id "$SMOKE_RUN"
    pt run logs "$SMOKE_RUN" --follow
    pt run wait "$SMOKE_RUN" --timeout-seconds 3120
    pt run reconcile "$SMOKE_RUN"
    pt --json run show "$SMOKE_RUN"

Use the fallback only for a conclusive resident-capacity failure reported in the first 15 minutes. Never retry under the same run ID:

    export SMOKE_RUN=opdq-fast02-iwopd-e2b12b-c12-lb4-rseq2-nomtp
    pt job run "$FALLBACK_PACKAGE" \
      --job distill \
      --provider dstack \
      --env HF_TOKEN \
      --timeout-seconds 3900 \
      --run-id "$SMOKE_RUN"
    pt run logs "$SMOKE_RUN" --follow
    pt run wait "$SMOKE_RUN" --timeout-seconds 3900
    pt run reconcile "$SMOKE_RUN"
    pt --json run show "$SMOKE_RUN"

Do not start any other capacity, prefill, MTP, cache, or physical-batch smoke. Stop work at minute 60 if the integrated gate is not complete. Save the normalized run view under ignored `.posttrain/state/qualification/` and update this plan with measured throughput and the accepted resident-wave size.

After that single smoke passes, validate, package, and start the frozen production package under the later end-to-end goal:

    export OPD_RUN=opdprod01-iwopd-e2b12b-r16-lb4-nomtp-scope384
    export OPD_PACKAGE=gemma4_e2b_scope_opd_iwopd_scope384_final.yaml

    pt catalog validate
    pt work-package validate "$OPD_PACKAGE"
    pt job plan "$OPD_PACKAGE" --job distill
    pt job pack "$OPD_PACKAGE" --job distill --build-missing
    pt job run "$OPD_PACKAGE" \
      --job distill \
      --provider dstack \
      --env HF_TOKEN \
      --timeout-seconds 432000 \
      --run-id "$OPD_RUN"
    pt run status "$OPD_RUN"
    pt run logs "$OPD_RUN" --follow

After terminal success:

    pt run wait "$OPD_RUN" --timeout-seconds 432000
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

A runtime configuration is qualified only by the successful 12-target seed-2907 GPU run on the exact model, teacher, image, IW-OPD objective, and frozen engine. Local tensor tests do not qualify memory or throughput. The record must include seconds per admitted target, actual request/wave sizes, rollout throughput, phase timing, utilization distribution, peak VRAM, loss/gradient evidence, teacher failures, replacements, and artifact digest. It must complete three optimizer updates and at least twelve physical backwards within the 60-minute wall limit; no additional live confirmation is required.

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

The integrated smoke has a hard 60-minute wall budget. The corrected resident-four attempt measured 592.70 seconds for its first four-target optimizer update, including checkpoint publication. Repeating that observed step time projects about 29.6 minutes for three updates; the first attempt reached update one roughly thirteen minutes after submission, so the complete run is expected to remain inside 45 minutes plus reconciliation. This remains a projection until a full 12-target run succeeds. A valid logical-four production estimate remains approximately 10–18 hours, but immediately before production replace that range with:

    selected_smoke_seconds_per_target * 384 + measured initialization/checkpoint overhead

Exclude initialization/download time from the per-target rate, then add the smoke's measured initialization and checkpoint overhead. Allow another 2–6 hours after training for final HF publication, two sequential sealed evaluations, and finalization.

## Interfaces and Dependencies

`posttrain.train.backends.trl.distillation` owns the qualified IW-OPD adapter, exact dependency guard, memory-safe sampled-token projection, teacher scoring, and global accumulated loss. `TrlPolicyGenerator` in `packages/train/src/posttrain/train/backends/trl/online_rl.py` owns per-turn rendering, task-specific response formats, and heterogeneous bounded submissions. `posttrain.train.integrations.verifiers.VerifiersEnvironmentRolloutBridge` owns task selection, selected-stage projection, concurrent exact-token rollouts, and stable ordering. The pinned TRL fork owns the generic IW-OPD buffer, post-generation denominator correction, LoRA synchronization/sleep lifecycle, bounded resident waves, and speculative counters.

Policy Prism `scope_opd_tasks.py` owns task-specific generated schemas; `scope_opd_admission.py`, `harness.py`, and `scope_opd_ledger.py` own structural admission, reserve allocation, and resumable target identity. PostTrain must not import Policy Prism into reusable packages. Policy Prism must not weaken canonical validation to accommodate vLLM.

External services are Hugging Face for gated base models and private adapter publication, the Live Kit OCI registry for content-addressed images, dstack for GPU placement, Trackio for metrics/traces/artifacts, OpenRouter for Claude judging, and Observatory for read-only inspection. Credentials remain in permission-protected environment files and never enter commits or terminal output.

Revision note (2026-08-10): replaced the multi-hour device/prefill/MTP matrix with exhaustive offline correction gates and one seed-2907 integrated smoke. The selected logical-four/physical-one configuration performs three updates and twelve backward passes within a hard 60-minute window, then freezes the fresh 384-target production configuration. MTP, student prefix caching, physical batching, and further serving sweeps are explicitly deferred. A corrected qualification reached one finite update and paired checkpoint before a shared fleet reachability outage; it is retained as partial operational evidence and is not accepted as qualification.
