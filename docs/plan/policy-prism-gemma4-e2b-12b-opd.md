# Correct and complete Policy Prism Gemma 4 E2B from 12B OPD

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated as implementation and execution proceed. Maintain this file in accordance with `docs/templates/PLAN.md`.

## Purpose / Big Picture

This plan produces one scientifically defensible Policy Prism on-policy distillation experiment in which base Gemma 4 E2B is the trainable student and base Gemma 4 12B is the frozen teacher. The final run keeps the fastest configuration already qualified on the in-house RTX PRO 6000, but corrects two deeper contracts that the earlier checkpoint-96 experiment did not satisfy: the teacher must score each completion after its own model-native prompt, and every student/teacher probability used by the loss must be normalized over the same XGrammar-allowed token set that generated the completion.

The plan deliberately avoids another parameter sweep. It preserves the measured logical-12, physical-1, accumulation-12 geometry and runs only two short live GPU canaries after exhaustive offline and exact-image verification. The first canary executes one real 12-target optimizer update with all corrected training contracts. The second serves that one-step adapter through the exact managed evaluation path on two non-sealed structured tasks. Only after both pass does one fresh 384-target production run start from base E2B. Completion includes retained checkpoints, reconciliation, private Hugging Face publication, sequential sealed scope and recovery evaluations, and the normal five-file Policy Prism `evaluation-runs` evidence.

This work changes one frozen product meaning narrowly. Existing documentation says the teacher scores the exact student token sequence. That is valid when student and teacher chat templates are identical, but E2B and 12B share token meanings while rendering different prompt-control tokens. The canonical baseline must therefore define cross-template OPD as preserving the exact student completion token IDs while rendering the same semantic messages with each model's immutable native template. The teacher scores only those completion positions. Freshness, one-time consumption, identical ordered vocabulary, selected-stage masking, and Verifiers trace authority remain unchanged.

## Progress

- [x] (2026-08-07 through 2026-08-10) Implement the original E2B-from-12B exact-token OPD path, memory-safe sparse projection, selected-stage masking, XGrammar wire-schema projection, stable task allocation, native traces, and checkpoint/SQLite recovery.
- [x] (2026-08-10) Retain a complete historical step-96 checkpoint from the earlier sampled sparse reverse-KL experiment, publish it privately at immutable Hugging Face revision `4f1fe9c75031396a11bcc44e2193f96df9003054`, and preserve failed-run evidence.
- [x] (2026-08-10 through 2026-08-11) Fix and qualify the target-78 shared-reserve boundary, target-98 exact source-ID constraint, unusable dependency propagation, heterogeneous rollout waves, and logical/physical accumulation behavior.
- [x] (2026-08-11) Measure logical `12`, physical `1`, accumulation `12` as the fastest completed configuration: 634.11 trainer seconds for 12 targets and 34.74 GiB trainer peak. Physical batches two and three were safe but slower; physical twelve was not qualified.
- [x] (2026-08-12) Evaluate historical checkpoint 96 on sealed scope and recovery, finalize both runs in Policy Prism, and verify complete native and five-file evidence.
- [x] (2026-08-12) Diagnose checkpoint 96. Verify the E2B/12B chat-template mismatch, the raw-versus-XGrammar-constrained probability mismatch, the weak historical sampled-token/tail objective, six evaluation whitespace terminations, model-facing raw YAML, constructed-incomplete label leakage, stage-inappropriate invariants, unrepresentative first-96 ordering, and a reproducible 12-worker SQLite cold-start race.
- [x] (2026-08-12) Reduce future live qualification to one 12-target corrected-training canary and one two-case managed-evaluation canary. Reject further batch, MTP, prefill, cache, LoRA, learning-rate, memory, HF, Trackio, or Claude smoke sweeps.
- [x] (2026-08-12) Amend the canonical OPD baseline for model-native prompt prefixes with exact completion-token alignment and constrained probability-space evidence.
- [x] (2026-08-12) Implement and push generic constrained teacher scoring in the CarbonTeq TRL fork at `b2dcbd0050f17383f97093d226b227d4b25acd75`; build collision-free `trl==1.9.2.post5` distributions with wheel SHA-256 `57f1b0c605e80bb30a499da0d2016264f109f823856e591c0b0dee79c6df2143`.
- [x] (2026-08-12) Pin collision-free TRL post5 in PostTrain; implement model-template identity, selected semantic-message projection, constrained memory-safe student loss, and managed XGrammar evaluation settings.
- [x] (2026-08-12) Correct Policy Prism prompt rendering, hidden-label leakage, stage invariants, checkpoint-prefix ordering, and SQLite initialization; regenerate task-plan hash `b8f3d62d9640c90950e514bf850f4bf28281b81663a581ef4e281080fbbadcee` and add corrected canary/production/evaluation work packages.
- [x] (2026-08-12) Preserve and reconcile three exact-image canary failures that closed catalog template identity, vLLM processor FQCN, and Verifiers sampled-mask digest boundaries before optimizer step one.
- [x] (2026-08-12) Prove the post5 constrained teacher replay path accepts all twelve corrected rollouts but takes more than one hour for a single 12,216-token teacher call; cancel it as operationally unfit rather than launching production.
- [x] (2026-08-12) Publish TRL `1.9.2.post6` at `11a526afd98bdbec3db9d6dd9473141c8c3b4d45`, precomputing exact XGrammar masks outside the per-token callback without changing probability semantics; pin runtime overlay `sha256:6f5bc755ef659fc9a5217a599fcdb1797c50d1735321ed7c4ae317de0d28a424` and pack actual-job image `sha256:8e04e3181c6adb428007c294d46ee14883bd157f10d80633ecc12a0103bf6550`.
- [x] (2026-08-13) Replace operationally unfit serial vLLM teacher replay with one frozen local-teacher forward plus chunked constrained vocabulary projection; preserve the fixed assistant grammar scaffold explicitly, publish TRL `1.9.2.post7` at `78b61a4d37a7bf8ad7e61bd604ba9e3c3c316897`, and publish runtime overlay `sha256:05dd0b4e3b80faffeeb7b3f7df043eb200c141ecb0c7ce25b5c3e462ead3952f`.
- [x] (2026-08-13) Preserve and reconcile canary `opd2can05-prefixq-e2b12b-c12-r16-v1`; reproduce its scorer-only XGrammar property-order mismatch against retained evidence, correct scorer schema ordering, and prove all three retained structured stages accept and terminate under the corrected replay FSM.
- [ ] Pass all offline, exact-image, credential, catalog, and package gates.
- [ ] Run and reconcile the one-update corrected-training canary.
- [ ] Run and reconcile the two-case managed-evaluation serving canary against the canary adapter.
- [ ] Freeze the exact production commits/image/configuration and launch one fresh 384-target run from base E2B.
- [ ] Reconcile training, verify 32 updates and checkpoints 8/16/24/32, and pass the Policy Prism 384-target completion gate.
- [ ] Publish and fresh-verify the step-32 LoRA adapter privately on Hugging Face.
- [ ] Run sealed scope then sealed recovery sequentially, finalize their standard Policy Prism evidence, validate all runs, and push both feature branches.

## Surprises & Discoveries

- Observation: identical tokenizer mappings did not make the two Gemma variants prompt-compatible.
  Evidence: both pinned `tokenizer.json` files hash to `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`, but E2B's chat template hashes to `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5` and 12B's hashes to `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`. The 12B non-thinking template adds four channel-control tokens. Current validation in `packages/train/src/posttrain/train/requests.py` checks only the token mapping.

- Observation: the earlier and current loss paths used probabilities from a different distribution than the one that sampled the response.
  Evidence: vLLM 0.25.1 defaults to `raw_logprobs` and calculates returned log probabilities before XGrammar masks logits. The response is sampled from the constrained distribution, while rollout, teacher, and current-student terms were raw full-vocabulary probabilities.

- Observation: checkpoint 96 used an especially weak teacher signal even though the teacher transported successfully.
  Evidence: the historical beta-one, sampled-top-one, tail-bucket branch included the student's sampled token and an aggregate all-other bucket; teacher top one was fetched but did not enter that support. The teacher could reduce confidence in one sampled mistake but could not name its preferred actor, effect, qualification, boundary, or abstention token.

- Observation: the checkpoint-96 sealed scope result is serving-confounded.
  Evidence: six rules responses reached exactly 16,384 tokens; 62.27% to 98.63% of each failed response was trailing whitespace. The training rollout path bounded whitespace, while the managed evaluation path did not. The raw sealed prompt/schema hash matched the base comparison for the inspected case, so the failure is not explained by a different benchmark prompt.

- Observation: raw YAML metadata is sent to both training and sealed-evaluation models.
  Evidence: `scope_opd_prompts.py` returns the full YAML text, `scope_opd_tasks.py` stores it as the system prompt, and `program.py` sends it verbatim. This is unnecessary noise, but it does not by itself explain checkpoint 96's relative regression because both compared models received the same sealed prompt.

- Observation: the future full plan contains real hidden-label leakage that checkpoint 96 never encountered.
  Evidence: constructed-incomplete units contain `::incomplete::<hash>` in their model-facing `context_id`. The first 96 historical targets contained zero constructed-incomplete tasks, but the complete plan contains 34.

- Observation: the first 96 targets were not a representative checkpoint prefix.
  Evidence: the prefix contained evidence/rules/graph `24/55/17`, full/standalone `29/67`, determinate/multiple/incomplete `87/9/0`, and length quartiles `24/15/23/34`. It could not demonstrate incomplete-context learning or balanced graph transfer.

- Observation: the Policy allocation ledger is not cold-start safe at production concurrency.
  Evidence: 12 simultaneous constructors against fresh database files failed 3 of 20 forced rounds with `sqlite3.OperationalError: database is locked`. The committed test pre-created the database, masking first-open WAL/DDL contention.

- Observation: throughput tuning is already sufficient and should not be repeated.
  Evidence: logical/physical/accumulation `12/1/12` took 634.11 seconds for 12 targets; `4/1/4` took 2,112.83 seconds, `12/2/6` took 805.64 seconds, and `12/3/4` took 1,130.05 seconds. Generation is the bottleneck, so larger physical batches spent more memory without improving it.

- Observation: the existing production work package is stale.
  Evidence: `.posttrain/work_packages/gemma4_e2b_scope_opd_from_12b.yaml` still resolves 384 optimizer steps, logical batch one, accumulation one, environment concurrency one, resident sequence one, and the historical sparse loss.

- Observation: the first packed corrected image still contained `trl==1.9.2.post1`.
  Evidence: actual jobs inherit backend dependencies from the published online-RL job-kind and install framework source with `--no-deps`. Immutable inspection of image `sha256:8e6fc6aaf98ddf9f86b874623e7fe932e8c5cd46f572b7c9ff155646abbb685f` proved the stale version before any GPU submission.

- Observation: `trl==1.9.2.post2` was already assigned to a different CarbonTeq source line.
  Evidence: remote tag `carbonteq-v1.9.2.post2` resolves to commit `216023d99324fae89dd58629130ba3bb043582ed`, not this experiment's fork. The first correction became post3; a later remote post4 collision was also preserved, so the final constrained-replay loader correction is the immutable post5 release.

- Observation: two exact-image canaries correctly failed before optimizer step one and exposed integration seams that offline component tests had not represented.
  Evidence: `opd2can01-nativeq-e2b12b-c12-r16-v1` failed because Python model variants carried chat-template fingerprints but catalog serialization omitted them. Retry `...-r1` failed because vLLM 0.25.1 loads custom logits processors using `module:Class`, while the fork supplied `module.Class`. Both runs were preserved and reconciled; catalog serialization now has a regression, and TRL post5 uses and tests the exact vLLM loader contract.

- Observation: Verifiers stores the trailing generation-prompt scaffold on the sampled assistant node with a false mask.
  Evidence: corrected canary `opd2can01-nativeq-e2b12b-c12-r16-v1-r2` reached real heterogeneous rollout generation, then failed closed because Policy hashed the whole assistant node as completion while the generator ledger hashed only sampled completion IDs. Policy and PostTrain now split at the single contiguous sampled-mask boundary: preceding nodes plus the false-mask scaffold are the exact prompt; the true-mask suffix is the exact completion. Focused Policy and PostTrain regressions exercise the real scaffold shape before another image is packed.

- Observation: exact constrained teacher replay is correct but the first implementation is not operationally viable.
  Evidence: `opd2can01-nativeq-e2b12b-c12-r16-v1-r3` accepted all twelve rollouts and recorded 12,216 scored tokens, crossing every earlier deterministic failure boundary, but did not return teacher latency, loss, or a backward pass after more than one hour. The teacher endpoint autoregressively forced every known completion token while repeating XGrammar state transitions and vocabulary masking inside vLLM's per-token callback. That projects to more than 32 hours for 32 updates, so production was not submitted.

- Observation: PostTrain's generated context was correct while the dedicated BuildKit builder retained a stale named-context mount.
  Evidence: the new package manifest hashed to `f44ff96009454e816e1b55339c25fc408b89802934713c46ac383163d49147cc`, but the builder repeatedly observed the old `04f4e6a8...` manifest at that path and failed closed. Quarantining the one generated cache directory was insufficient; restarting only `posttrain-opd-clean` made the isolated qualification consume the correct manifest. No source or run evidence was removed.

- Observation: the selected semantic completion begins after a fixed assistant scaffold that must also advance XGrammar replay state.
  Evidence: retained canary trace `7c220e2cb452476dadde144ff97a6b49` has 169 selected-node tokens with mask `[false, false, false, true, ...]`; the three-token prefix is `[105, 4368, 107]`. The previous scorer started XGrammar at the first true-mask token and correctly failed it as disallowed. PostTrain now appends that exact scaffold to the teacher-native prompt and seeds both teacher and current-student matchers before normalizing selected tokens; TRL post7 preserves the field through buffered micro-slices.

- Observation: alphabetically serializing a JSON schema creates a different XGrammar FSM from the colocated rollout path.
  Evidence: canary `opd2can05-prefixq-e2b12b-c12-r16-v1` generated twelve admitted rollouts and reached local teacher scoring, then failed at selected token position one because `_xgrammar_matcher()` used `sort_keys=True`. The retained rules/graph schemas declare required-property order that differs from alphabetical `properties` order. Direct replay failed under alphabetical serialization but accepted and terminated for all retained evidence, rules, and graph outputs when object properties were ordered recursively by their declared `required` arrays. The correction affects only scorer grammar construction; canonical Policy validation and schema digests remain unchanged.

## Decision Log

- Decision: amend the canonical baseline before changing cross-template teacher scoring.
  Rationale: model-native teacher prefixes alter the previous meaning of “exact token scoring.” The exact completion token sequence remains preserved, but the conditioning prefix becomes model-native. This must be explicit rather than hidden in an adapter.
  Date/Author: 2026-08-12 / Codex.

- Decision: use corrected constrained IW-OPD as the production objective; do not add the proposed sparse teacher-top-k anchor to this run.
  Rationale: constrained IW-OPD is the maintained objective and can be made mathematically consistent. A hybrid anchor introduces a second, unqualified objective and a new top-k hyperparameter. A one-step smoke cannot establish its quality advantage, and an objective sweep conflicts with the request to start quickly. The historical top-one failure is adopted as a prohibition against reusing that weak loss, not as evidence for an untested replacement.
  Date/Author: 2026-08-12 / Codex.

- Decision: retain the previously qualified hyperparameters and runtime geometry.
  Rationale: rank, learning rate, context limits, prefill, memory reservations, cache choices, MTP choice, and logical/physical batching have direct live evidence. The new fixes change correctness contracts, not the measured hardware bottleneck.
  Date/Author: 2026-08-12 / Codex.

- Decision: run exactly two live pre-production canaries.
  Rationale: offline tests can close deterministic math, rendering, schema, allocation, and recovery boundaries. One live training update is still required to prove student generation, native teacher scoring, constrained loss, accumulation, memory, and artifacts together. One managed-evaluation canary is required because the historical whitespace defect exists only on that distinct serving path. Nothing else has an unclosed live boundary.
  Date/Author: 2026-08-12 / Codex.

- Decision: reuse the verified seed-2907 twelve-target cohort for the training canary.
  Rationale: its exact indices are `31, 49, 70, 75, 78, 98, 126, 163, 182, 233, 278, 329`. It covers all three stages, all four prompt profiles, all four quartiles, four full trajectories, target 78, target 98, and two constructed-incomplete tasks. It produces one optimizer update under the production `12/1/12` geometry.
  Date/Author: 2026-08-12 / Codex.

- Decision: do not repeat live MTP, prefix-cache, prefill, physical-batch, logical-batch, LoRA, learning-rate, Trackio, HF, or Claude probes.
  Rationale: MTP and student prefix caching remain deliberately off; prefill 4,096 and `12/1/12` are measured winners; HF publication and Trackio/Claude evaluation were exercised successfully by checkpoint 96. Repeating them cannot validate the new teacher/loss contracts.
  Date/Author: 2026-08-12 / Codex.

- Decision: predeclare checkpoint 32 as the experiment model.
  Rationale: Policy Prism currently has no independent non-sealed Gold domain suite suitable for selecting among quarter checkpoints. Using sealed scope to choose a checkpoint would leak qualification evidence, while one-step or source-only structural scores cannot identify legal quality. Checkpoints 8/16/24 remain recovery and later exploratory evidence. A separate future experiment may add a reviewed non-sealed Gold selection suite.
  Date/Author: 2026-08-12 / Codex.

- Decision: keep existing business KPI definitions unchanged.
  Rationale: KPI redesign and historical re-derivation are outside this experiment. Raw traces and compatibility metadata remain available for honest interpretation.
  Date/Author: 2026-08-12 / Codex.

- Decision: work only on the existing PostTrain and Policy Prism OPD feature branches and one new TRL fork branch.
  Rationale: the PostTrain feature branch already contains the qualified release merge and Policy-specific deltas. Merging a newer `main` now would invalidate qualification and introduce conflicts unrelated to this experiment.
  Date/Author: 2026-08-12 / Codex.

- Decision: use a private, hash-labelled OCI overlay for the corrected TRL runtime closure.
  Rationale: the internal devpi index is readable but this workstation has no upload credential; an attempted upload correctly returned HTTP 401. The overlay derives from the immutable published online-RL parent, replaces only TRL with the locally built post5 wheel using `--no-deps`, asserts its imported version during build, and records source and wheel hashes in OCI labels. Digest `sha256:6f00649ac8c5bc75db479389f4957fa62a603cbf842c3e51c2f5182fc7c84cc4` remains private under `registry.lan/carbonteq`. The exact actual-job image must still pass immutable import and behavioral gates before use.
  Date/Author: 2026-08-12 / Codex.

- Decision: reject the post5 teacher replay runtime and test exactly one semantics-preserving post6 optimization.
  Rationale: production cannot start from a measured greater-than-one-hour update. Post6 pre-walks the known completion's grammar states once on CPU and reuses exact allowed-token sets during scoring; it preserves the selected token, schema digest, constrained denominator, and per-position alignment evidence. Unit tests prove bool and signed compressed XGrammar masks decode identically and that runtime callbacks perform no matcher traversal. A single identical GPU canary is necessary and sufficient to measure whether this removes the bottleneck; no parameter sweep is introduced.
  Date/Author: 2026-08-12 / Codex.

- Decision: use a frozen colocated Transformers teacher for constrained scoring instead of token-serial vLLM replay.
  Rationale: both vLLM replay implementations remained proportional to completion length and missed the overnight budget. The local path performs one teacher hidden-state forward and bounded vocabulary chunks, retains teacher-native prompt rendering and exact student completion IDs, and applies the identical XGrammar allowed set and digest evidence at every selected position. Dense-vs-chunked tests prove exact constrained probabilities; the single live canary remains the release gate for memory, latency, finite loss, and backward behavior.
  Date/Author: 2026-08-13 / Codex.

- Decision: canonicalize scorer-only XGrammar object properties by declared required order.
  Rationale: this matches the property ordering proven by actual colocated vLLM outputs and prevents the teacher/current-student scorers from constructing an alphabetically different constrained probability space. The transformation is recursive, deterministic, non-mutating, and does not weaken or rewrite Policy's canonical schema.
  Date/Author: 2026-08-13 / Codex.

## Outcomes & Retrospective

The historical experiment produced a recoverable checkpoint and useful failure evidence, not a successful final model. Its optimizer was numerically stable and teacher transport did not fail, but the teacher was conditioned on the wrong model template, the objective mixed constrained sampling with raw probabilities, the first 96 tasks were unrepresentative, and six scope evaluation pipelines were lost to a serving whitespace loop. More steps under that same contract would not have repaired those issues.

The final outcome has not yet been produced. At completion, update this section with the exact TRL, PostTrain, and Policy Prism commits; image digest; canary and production run IDs; measured update time; checkpoint artifacts; Hugging Face revision; sealed results; finalized run directories; and any residual scientific limitation. A low domain score remains a valid negative experiment. Operational success must never be rewritten as scientific improvement.

## Context and Orientation

Three repositories participate and must remain independently reproducible.

`/home/ali-awais-safdar/Post-Train/posttrain` is the framework workspace. Work stays on `feat/gemma-policy-prism-opd-e2b-12b`, currently rooted at `8c0d76637cc0eba87a6c3d360a2d92f6a99d29db`. `packages/train` owns typed distillation requests and private TRL integration; `packages/serve` owns managed-vLLM configuration; `packages/eval` owns the Verifiers evaluation adapter; `packages/common` owns model identity; `.posttrain` project files do not live here for this experiment.

`/home/ali-awais-safdar/Policy Prism` owns the environment and project composition. Work stays on `feat/scope-opd-e2b-12b-environment-v1`, currently rooted at `79627530d907b4e3565ddd912db2327f64f72174`. `packages/normative-verifiers` owns prompts, staged tasks, admission, allocation, deterministic plans, completion validation, and finalization. `.posttrain/catalog` and `.posttrain/work_packages` bind exact framework selections. `evaluation-runs` owns permanent finalized benchmark evidence.

`/home/ali-awais-safdar/Post-Train/trl` is the expected sibling checkout of the CarbonTeq TRL fork. It is not presently available locally. Create it from `carbonteq-ai/trl`, verify `origin` and `upstream`, and branch from the exact currently consumed fork source `a82ecebc0fa081efd58302a34a553445fc73271d`. Generic teacher-server constrained scoring belongs there. The fork must be committed, pushed, documented in `CARBONTEQ_FORK.md`, released immutably, and only then pinned by PostTrain.

The immutable model inputs remain:

| Role | Repository | Revision |
| --- | --- | --- |
| Student | `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| Teacher | `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Ordered token mapping | both | `059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416` |

The GPU target is one NVIDIA RTX PRO 6000 Blackwell Workstation Edition with 96 GiB. “Logical batch” means how many fresh targets are generated and scored for one optimizer update. “Physical batch” means how many sequences enter one differentiable forward/backward slice. Gradient accumulation combines twelve physical-one slices into the logical-twelve update.

The probability names used below are important. `pS` and `pT` mean raw full-vocabulary student and teacher distributions. `qS` and `qT` mean those distributions after applying the exact XGrammar allowed-token mask and renormalizing. Structured rollouts are sampled from `qS`; therefore constrained IW-OPD must use `qS`, `qT`, and the current trainable student's `qCurrent`.

## Frozen experiment configuration

The original experiment's useful findings are retained exactly unless the corrected one-update canary disproves compatibility.

| Setting | Production value | Evidence/rationale |
| --- | --- | --- |
| Population | 384 reviewed targets, each optimized once | One complete task-plan pass; no replay of checkpoint 96 |
| Stage totals | 77 evidence / 230 rules / 77 graph | Original Policy design |
| Logical / physical / accumulation | `12 / 1 / 12` | Fastest measured, 634.11 seconds per 12 targets |
| Optimizer updates | 32 | `384 / 12` |
| LoRA | rank 16 / alpha 32 / dropout 0 | Stable historical and qualified configuration |
| Learning rate | `1e-5` | Qualified IW-OPD value; no LR sweep |
| Scheduler / warmup | linear / 0 | Original configuration |
| Gradient clipping | 1.0 | Existing safety setting |
| IW-OPD gamma / epsilon | `0.5 / 1e-8` | Maintained backend defaults already qualified numerically |
| Probability space | XGrammar-constrained | Corrected contract; raw full-vocabulary is rejected |
| Teacher prompt alignment | model-native prefix, exact completion IDs | Corrected cross-template contract |
| Maximum prompt / call / trainer sequence | `32,768 / 40,960 / 49,152` | Existing task/token audit and training envelope |
| Maximum outputs | evidence 2,048 / rules 16,384 / graph 8,192 | Original stage contract |
| Environment concurrency / resident sequences | `12 / 12` | Qualified heterogeneous rollout wave |
| Chunked prefill | enabled, 4,096 tokens | Fastest qualified prefill setting |
| Student / teacher memory reservation | `0.20 / 0.35` | Qualified on 96 GiB GPU |
| Prefix caching | student off / teacher on | Student invalidation after LoRA sync is unproven; teacher is frozen |
| MTP | off | No proven gain for this Policy IW-OPD path |
| KV / execution / parallelism | FP8 KV / eager / TP1 | Qualified path |
| Checkpoints | 8, 16, 24, 32; retain four | Recovery at every 96 targets |
| Seed | `20260807` | Predeclared production seed |
| Provider timeout | 86,400 seconds | Safety ceiling, not duration estimate |

The old selection-lock and task-plan hashes are historical parents:

    selection-lock: 0e6cf112560e6b6a2c55ba3d622adda4c728ffef7a7442494be948d27921ec37
    task-plan:      678465cc2f81e7d965be85587e02b69326d72f3c9358316811c3ef0535e3b0cc

Prompt rendering, opaque incomplete IDs, and deterministic reordering will change the executable release hashes. Regenerate them once, record both new 64-character values in this plan, and make the final resolved job assert them. Do not continue to claim the old hashes after changing model-facing tasks.

## Plan of Work

### Milestone 1: amend the exact-token OPD contract

Update `docs/post-training/02-primitives.md` so cross-template distillation preserves the exact generated completion IDs while each model renders the same semantic messages with its own immutable template. Update `docs/post-training/05-apis.md` to add typed prompt-alignment and probability-space settings. Update `docs/post-training/06-observation-and-lineage.md` to require student/teacher template fingerprints, both prompt-prefix digests, completion-token digest, grammar/schema digest, allowed-set evidence, and q-logprob alignment.

Define two typed settings rather than backend-option strings:

    TeacherPromptAlignment = "exact_full_sequence" | "model_native_prefix_exact_completion"
    DistillationProbabilitySpace = "raw_full_vocab" | "generation_constrained"

The Policy work package must select the second value of each. `exact_full_sequence` remains valid for model pairs with identical templates. `generation_constrained` requires one identical structured-output grammar over the exact completion token positions and fails closed when any position cannot be replayed.

Acceptance is a documentation diff that retains on-policy freshness and same-token-mapping requirements while making E2B-from-12B behavior unambiguous. No implementation begins under an undocumented reinterpretation.

### Milestone 2: correct Policy Prism's model-facing contract

In `scope_opd_prompts.py` parse prompt YAML into structured semantic fields. Keep `prompt_id`, `stage`, `tangents`, response configuration, source stratification, and hidden decision data in trace/config metadata, but render only role, objective, instructions, stage-specific output rules, skeleton, and examples into the system message. Apply the same semantic renderer to sealed evaluation so future train/eval prompt behavior is intentional and snapshot-tested.

In `normative_stages.py`, remove the model-facing `context_id` or replace it with an opaque hash that contains no decision-class word. Emit stage-specific invariants: evidence covers every supplied segment exactly once and in order; rules may emit zero, one, or many rules per segment while supplied IDs are fixed and prior counts are advisory; graph fixes admitted rule/qualification IDs and forbids duplicate nodes, edges, or attachments. Do not tell graph generation about segment-count rules.

Keep all four training profiles, including the benchmark profile. Add semantic coverage assertions—not word-count assertions—for actors, agency duties, legal effects, parent inheritance, zero/one/many decomposition, conditions/exceptions, ambiguity/insufficient context, exact quotes, fixed IDs, duplicate prevention, and one closed JSON object. Add missing independently authored guidance/examples only where a coverage assertion proves a gap.

Preserve the intended OPD admission boundary. Malformed JSON, schema failure, unknown references, duplicate IDs, repetition, provider error, and `finish_reason=length` remain non-trainable. A structurally valid but legally wrong actor, effect, qualification, quote, or ambiguity decision remains trainable; do not add `require_abstention=True` as a structural gate for constructed-incomplete tasks.

Reorder the same 384 reviewed targets through a deterministic stratified interleaver. Every consecutive 96-target checkpoint block must contain all stages, all profiles, both shapes, all decision classes, all quartiles, and proportional source domains. Require exactly 24 targets from each length quartile and exactly 32 full plus 64 standalone tasks in every block. Stage, profile, and decision counts may differ by at most one from their proportional allocation while preserving final totals. Preserve family isolation, sealed-family exclusion, instrument caps, and reserve compatibility. Classify or reject the one current `unknown` source domain.

Make `ScopeOpdLedger` cold-start safe. Acquire a persistent sibling `fcntl.flock` before first schema initialization; under the lock open SQLite with a 60-second busy timeout, set WAL once, create tables transactionally, set `PRAGMA user_version=1`, and close. Migrate version-zero databases idempotently and reject unknown versions. Normal task constructors must not rerun mutating WAL/DDL. Operational connections retain foreign keys, the busy timeout, and the selected durability.

Regenerate candidates, task plan, selection lock, token-budget summary, and hashes. Audit every actual reserve rendered with its logical primary's profile/shape. Pin the resulting source commit in later catalog selections.

### Milestone 3: implement generic teacher-native constrained scoring in the TRL fork

Clone or update `/home/ali-awais-safdar/Post-Train/trl`, verify `origin=carbonteq-ai/trl` and `upstream=huggingface/trl`, and create a feature branch from `a82ecebc0fa081efd58302a34a553445fc73271d`. Do not place this generic scoring behavior only in PostTrain.

Extend IW-OPD teacher requests so each row may provide a teacher-native prompt prefix, exact student completion IDs, a structured-output specification, and the expected completion/schema digests. The teacher server must replay the same grammar from the completion boundary, compute the selected token's logit and `logsumexp` over the exact allowed tokens at each position, and return `log qT` plus selected-token IDs, allowed-token counts, and a deterministic allowed-set digest. It must not generate a teacher completion or retokenize student completion text.

Expose processed student rollout log probabilities `log qS` from the colocated vLLM path. Fail if the selected token is disallowed, if output token IDs differ, if the grammar cannot be replayed, or if a row lacks the required schema under `generation_constrained` mode. Keep the raw path available for unconstrained users.

The implementation must avoid retaining sequence-by-vocabulary tensors or a full completion-by-vocabulary bitmask. Advance one grammar matcher per row and process bounded token-position chunks. Document the maintained delta, compatibility constraints, and tests in the fork's `CARBONTEQ_FORK.md`; run the fork suite; commit and push; publish the next immutable internal release; and record its commit, tag, wheel hash, and sdist hash.

### Milestone 4: integrate corrected OPD in PostTrain

Add immutable concrete chat-template fingerprints to Gemma model variants. E2B and 12B may retain the verified common ordered-token fingerprint, but they must not pretend to share one concrete template identity. Validate both identities in `OnPolicyDistillationRequest` according to the selected prompt-alignment mode.

Extend the backend-neutral Verifiers distillation projection so the selected training node retains the exact semantic messages, response format, selected completion IDs, prompt/completion digests, and schema digest needed for teacher-native rendering. Continue to verify the selected branch/node and mask only selected target-stage tokens. A rejected candidate's messages or tokens must never enter the selected row.

In the TRL adapter, render the student prefix with E2B and teacher prefix with 12B from the same semantic messages, append identical completion IDs, and align teacher results by completion position. Record both prefix hashes and template hashes. Replace the current full-vocabulary denominator in `_memory_safe_server_iw_opd_loss` with a chunked XGrammar-allowed `logsumexp`; use the returned `qS` and `qT`. Retain the global valid-token numerator/denominator across physical-one accumulation slices and the memory-safe Gemma hidden-state/LM-head path.

Add typed managed-vLLM structured-output backend support in `packages/serve`. Policy Gemma evaluation bindings must start vLLM with XGrammar and send the generation-only schema plus request-level `whitespace_pattern=r" ?"`. Preserve the canonical schema for local admission. The already-correct Verifiers `chat_template_kwargs` nesting under `extra_body` remains unchanged and regression-tested.

Add one intentional artifact-export CLI flow backed by the existing tracked materialization contract so publication/finalization never depends on ad hoc `/tmp` scripts. It must use a uniquely named export/provenance run, select exact logical artifact names, verify provider and PostTrain content digests, download atomically under `.posttrain/state`, write receipts, reject ambiguity/overwrite, and resume idempotently. This run is expected to appear in Trackio because it records a real artifact-consumption edge; its name and purpose must make it distinct from training/evaluation.

Update `packages/train/pyproject.toml`, `uv.lock`, runtime locks, `docs/tooling/trl/README.md`, and CI wheel references only after the TRL fork is committed, pushed, and published. Build a fresh job-kind/actual-job closure; do not reuse the old image.

### Milestone 5: close every deterministic boundary offline and in the exact image

Run focused tests before broad suites. The required behavior—not a particular test-file layout—is:

1. E2B and 12B ordered token mappings match; template fingerprints differ; each model renders its exact known native prefix; the teacher request contains the 12B prefix and the exact E2B completion IDs.
2. A tiny vocabulary/schema reference computes dense `qS`, `qT`, `qCurrent` and exact constrained IW-OPD loss/gradients. The memory-safe chunked implementation matches it. Physical-one/accumulation-twelve matches one logical-twelve batch with variable masks and lengths.
3. vLLM's sampled-token processed log probability equals an independently replayed XGrammar probability. Teacher and current-student allowed-set digests match at every selected position. Raw-vs-constrained mismatches fail closed.
4. Controlled non-sealed legal fixtures prove teacher signal direction before optimization: for eight source-family-disjoint cases, the 12B teacher ranks the correct structured completion above a matched wrong actor/effect/qualification/abstention perturbation in at least seven cases and at least one case of each perturbation class. Use length-normalized completion q-loglikelihood and record paired margins.
5. All twelve training profile/stage prompts render semantic content only, contain every required legal concept, and contain no model-facing metadata/hidden-label values. Every stage receives only its own invariants.
6. Every actual candidate/reserve/profile/shape prompt is under cap after rendering. All 1,293 static stage schemas and empty/ordinary/maximum realistic dynamic graph schemas compile under XGrammar 0.2.3. Canonical validation still enforces unsupported wire constraints such as `uniqueItems`.
7. Target 78's forced cross-profile reserve claim, target 98's exact source ID, dynamic graph dependency IDs, maximum-three replacement behavior, restart, and selected-branch purity all pass.
8. Fifty fresh 12-thread and twenty fresh 12-process ledger rounds complete with zero lock error, duplicate claim, or duplicate slot. Checkpoint backup/restore preserves exact schema version, attempts, claims, accepted targets, and integrity.
9. The reordered plan preserves the exact 384-target population and final distribution while every 96-target block satisfies the stratification contract.
10. Managed evaluation emits XGrammar server configuration and request-level bounded whitespace, preserves `extra_body.chat_template_kwargs`, and rejects top-level `chat_template_kwargs`.
11. Checkpoint model/recovery pairs retain adapter, optimizer, scheduler, RNG, trainer state, tokenizer identities, and SQLite ledger together.

Then pack both canaries and production. Execute a release verifier inside each relevant image. It must import the pinned TRL and Policy packages, assert exact source commits and dependency versions, compile the real schemas, run the cold-ledger barrier, render both model prefixes, and execute the analytic constrained-loss check. A host-only green suite is insufficient.

### Milestone 6: run only the two necessary live canaries

The total GPU canary budget is 90 minutes, excluding OCI build time. Do not add an exploratory arm when a canary fails; diagnose the failed contract and rerun only the same canary after a reviewed fix.

The first canary is one real optimizer update from base E2B using the exact production objective, LoRA, model revisions, `12/1/12` geometry, concurrency/resident 12, prefill 4,096, memory reservations, cache choices, XGrammar settings, and seed-2907 cohort. Its twelve indices are:

    31, 49, 70, 75, 78, 98, 126, 163, 182, 233, 278, 329

This is not an optimization sweep. It is the minimal end-to-end proof of student generation, teacher-native constrained scoring, one finite accumulated backward/update, target 78/98 behavior, constructed-incomplete handling, ledger concurrency, checkpoint sidecar, and Trackio evidence.

The training canary passes only when all twelve logical targets are accepted exactly once; every selected completion has matching student/teacher completion IDs and allowed-set digests; both model-native prefix hashes are present and different; teacher controlled-fixture ranking passes; scored tokens are positive; teacher/provider/truncation/schema/source/graph/SQLite failures are zero; loss and gradient are finite; resident wave exceeds one; peak system memory stays below 85 GiB; and step-one model plus recovery artifacts reconcile consistently. The prior measured update was 634.11 seconds. Investigate if the corrected update exceeds 1,200 trainer seconds or the provider run exceeds 45 minutes.

The second canary uses the one-step adapter as an explicit model input to `eval/verifiers-managed@1`. It runs exactly two non-sealed diagnostic cases through the final managed Gemma evaluation binding: one long rules-only q4 candidate and one long full graph q4 candidate. Use stable candidate/source identities rather than sealed cases. Claude judging is disabled because this canary tests serving, not legal quality.

The serving canary passes only when both pipelines end with `finish_reason=stop`, both canonical schemas validate, no stage reaches its token ceiling, no response contains a trailing-whitespace run beyond the bounded pattern, dynamic graph IDs match admitted rules, the LoRA adapter is actually loaded, native traces upload, and reconciliation is consistent. The job must finish within the remaining 90-minute combined canary budget.

No other live smoke is required. Trackio writing, HF publication, OpenRouter/Claude, dstack placement, and the base serving stack already succeeded during checkpoint-96 qualification; credentials and service health are checked read-only before launch.

### Milestone 7: freeze and package production

After both canaries pass, capture immutable TRL, PostTrain, and Policy Prism commits and push each owning branch in dependency order: TRL first, PostTrain pin/integration second, Policy source third, Policy catalog/work packages fourth. Both primary feature worktrees must be clean and match their remote branches.

Add new selections instead of mutating historical ones: a production environment with 384 tasks and concurrency 12; settings with 32 steps, physical batch one, accumulation 12, and checkpoints every eight; a constrained-IW-OPD rank-16 binding; the qualified rollout/teacher bindings; and `gemma4_e2b_scope_opd_iwopd_scope384_final.yaml`. Resolve exact new selection-lock/task-plan hashes and exact model/template/tokenizer/source identities. The production job must have no checkpoint-96, canary-adapter, canary-ledger, or previous-run input.

`pt job diff` between the corrected-training canary and production packages must show only the deliberate task population, step/checkpoint cadence, artifact retention, and run-description changes. Objective, models, templates, runtime code, dependency lock, LoRA, sampling, generation, memory, concurrency, and target must be identical.

### Milestone 8: launch, monitor, recover, and validate the 384-target run

Use a unique run ID beginning with `opdprod2`. Submit with a 24-hour safety timeout. Remain attached through model loading, the controlled teacher-signal gate, the first twelve accepted targets, the first finite update, and the first successfully uploaded trace batch. Because the first configured checkpoint is step eight, do not claim recoverability after step one.

At every update require `accepted_count == global_step * 12`, twelve unique targets, positive scored tokens, zero teacher failures, finite loss/gradient, no duplicate candidate/reserve, and reserve use within each stage/quartile/decision stratum. Negative IW-OPD loss is allowed. Warn when a stratum consumes more than half its reserves before half its primaries; stop on exhaustion or repeated systematic structural failure. Investigate two consecutive updates above 1,800 seconds or unexplained sustained system memory above 85 GiB.

At checkpoints 8/16/24/32 require paired model/recovery artifacts and ledger accepted counts 96/192/288/384. Infrastructure-only interruption resumes from the latest complete pair under a new run ID with identical image, models, plan, objective, batches, scheduler, seeds, and restored SQLite ledger. Before step eight restart from base. Never resume deterministic schema/ledger/reserve failure, NaN/Inf, systematic teacher failure, or any code/configuration change.

After provider success, reconcile and materialize the exact terminal traces, summary, step-32 model view, and step-32 recovery view through the export flow. Run a Policy Prism completion validator. It must prove 32 finite updates, 384 accepted unique logical targets, exact slots `0..383`, twelve targets per update, final stage totals `77/230/77`, exact plan IDs/hashes, zero teacher errors/truncations, trace-ledger-candidate/digest agreement, SQLite integrity, checkpoint pairs at all four milestones, and no missing artifact roles. Provider success alone is insufficient.

### Milestone 9: publish checkpoint 32 and qualify it sequentially

Predeclared checkpoint 32 is the experiment result. Materialize only its `model-adapter` view for publication; never upload the `training-checkpoint`. Verify PEFT rank 16, alpha 32, dropout zero, exact base repository/revision, safetensors loadability, absence of optimizer/scheduler/RNG/base weights, provider digest, and PostTrain content digest.

Publish privately to `carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-lora-v1`. The model card records model/template/tokenizer identities, all three source commits, training run/step, task hashes, constrained-IW-OPD definition, runtime configuration, canary evidence, known limitations, and evaluation status. Resolve the immutable 40-character HF revision, fresh-download it into a different ignored directory, and require a byte-identical complete file manifest.

Register one final OPD adapter variant and inference binding using the exact Trackio and HF identities. Run sealed scope first with a unique `opd2sc32...` ID. Require 18 expected/included traces, zero failures/truncations/errors, bounded whitespace, complete Claude judging, one native evaluation artifact, and consistent reconciliation. Only after scope releases the GPU run recovery under `opd2rc32...`; require the corresponding 17/17 gate.

Materialize both native evaluation artifacts through the export flow and finalize directly into `Policy Prism/evaluation-runs`. Require `manifest.json`, `traces.jsonl`, `business-kpis.json`, `engineering-metrics.json`, and `semantic-diagnostics.json` for each, plus the updated catalog. Run `validate-runs` and the existing KPI check without changing KPI definitions. Prompt and serving corrections will produce new compatibility hashes; record them and compare with historical runs as cross-prompt/cross-serving evidence, as explicitly accepted for this experiment.

Commit and push the final Policy Prism model catalog, evaluation directories, and `evaluation-runs/catalog.json`. Update this living plan with final evidence, commit and push the PostTrain branch, then clean only the new provider workspaces after HF fresh verification and Policy finalization. Retain Trackio, HF, checkpoints, failed-run, and evaluation evidence.

## Concrete Steps

The later execution goal runs these commands and updates them with newly produced immutable values. Never print secrets.

First verify the three repository boundaries. In PostTrain:

    export POSTTRAIN_ROOT=/home/ali-awais-safdar/Post-Train/posttrain
    export POLICY_ROOT="/home/ali-awais-safdar/Policy Prism"
    export TRL_ROOT=/home/ali-awais-safdar/Post-Train/trl
    export KIT=/home/ali-awais-safdar/Post-Train/posttrain-setup-v0.2.2-20260728/posttrain-setup
    export POSTTRAIN_ENV_FILE="$POLICY_ROOT/.env.posttrain"

    cd "$POSTTRAIN_ROOT"
    git switch feat/gemma-policy-prism-opd-e2b-12b
    git add docs/plan/policy-prism-gemma4-e2b-12b-opd.md
    git commit -m "docs(opd): finalize corrected E2B from 12B experiment plan"
    git push origin feat/gemma-policy-prism-opd-e2b-12b
    test -z "$(git status --porcelain)"
    test "$(git rev-parse HEAD)" = "$(git rev-parse origin/feat/gemma-policy-prism-opd-e2b-12b)"

In Policy Prism:

    cd "$POLICY_ROOT"
    git switch feat/scope-opd-e2b-12b-environment-v1
    test -z "$(git status --porcelain)"
    test "$(git rev-parse HEAD)" = "$(git rev-parse origin/feat/scope-opd-e2b-12b-environment-v1)"

Create the missing fork checkout without changing either main branch:

    if [ ! -d "$TRL_ROOT/.git" ]; then
      git clone git@github.com:carbonteq-ai/trl.git "$TRL_ROOT"
    fi
    cd "$TRL_ROOT"
    git remote get-url origin
    if ! git remote get-url upstream >/dev/null 2>&1; then
      git remote add upstream https://github.com/huggingface/trl.git
    fi
    git remote get-url upstream
    git fetch origin
    if git show-ref --verify --quiet \
      refs/heads/feat/iwopd-native-template-constrained-logprobs; then
      git switch feat/iwopd-native-template-constrained-logprobs
    else
      git switch -c feat/iwopd-native-template-constrained-logprobs \
        a82ecebc0fa081efd58302a34a553445fc73271d
    fi

If the directory already exists, fetch and create/switch the same feature branch from the pinned commit instead of cloning again. Do not base the work on an unreviewed newer fork head.

After implementing each repository's milestone, run its focused tests, update its maintained documentation, commit only that repository, and push only its feature branch. Publish the TRL release before editing PostTrain's immutable pin.

Policy Prism focused validation runs from `$POLICY_ROOT`:

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync pytest \
      packages/normative-verifiers/tests/test_scope_opd.py \
      packages/normative-verifiers/tests/test_program.py \
      packages/normative-verifiers/tests/test_data_and_plugins.py \
      packages/normative-verifiers/tests/test_yaml_prompts_and_quotes.py

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync ruff check \
      packages/normative-verifiers/src/policy_prism_normative_verifiers \
      packages/normative-verifiers/tests

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

The new release verifier must also be invokable explicitly and produce a machine-readable pass receipt:

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers verify-scope-opd-runtime \
      --plan-revision policy-prism-scope-opd-plan-v3 \
      --xgrammar-version 0.2.3 \
      --ledger-concurrency 12 \
      --output .posttrain/state/opd-verification/host-receipt.json

PostTrain focused validation runs from `$POSTTRAIN_ROOT`:

    UV_CACHE_DIR=/tmp/posttrain-uv-cache \
    uv run --no-sync pytest \
      packages/common/tests \
      packages/train/tests/test_api.py \
      packages/train/tests/test_trl_sparse_distillation.py \
      packages/train/tests/test_trl_online_rl.py \
      packages/train/tests/test_verifiers_grpo_bridge.py \
      packages/train/tests/test_trl_checkpoint_artifacts.py \
      packages/train/tests/test_checkpoints.py \
      packages/serve/tests/test_vllm_bindings.py \
      packages/eval/tests/test_api.py

    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync ruff check .
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync pyright
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run --no-sync lint-imports
    git diff --check

After the immutable dependency pin is final, run the locked and broad release ladder:

    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv sync --all-packages --locked --python 3.13
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run ruff check .
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run pyright
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run lint-imports
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run pytest
    git diff --check

Configure the PostTrain control terminal only after commits and locks are immutable:

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
    test -n "$OPENROUTER_API_KEY"
    pt doctor
    pt catalog validate
    pt workers

Expected preflight is a valid catalog, reachable Trackio/registry/dstack/HF services, and `carbonteq-ai-workstation.lan` healthy, reachable, idle, and exposing one RTX PRO 6000 block. Verify HF read access to both gated Gemma revisions and private CarbonTeq write access without displaying the token.

Validate, plan, and pack the three new packages:

    pt work-package validate gemma4_e2b_scope_opd_correctness_c12_lb12.yaml
    pt work-package validate gemma4_e2b_scope_opd_eval_serving_canary.yaml
    pt work-package validate gemma4_e2b_scope_opd_iwopd_scope384_final.yaml

    pt job plan gemma4_e2b_scope_opd_correctness_c12_lb12.yaml --job train
    pt job plan gemma4_e2b_scope_opd_eval_serving_canary.yaml --job evaluate
    pt job plan gemma4_e2b_scope_opd_iwopd_scope384_final.yaml --job train

    pt job pack gemma4_e2b_scope_opd_correctness_c12_lb12.yaml \
      --job train --build-missing
    pt job pack gemma4_e2b_scope_opd_eval_serving_canary.yaml \
      --job evaluate --build-missing
    pt job pack gemma4_e2b_scope_opd_iwopd_scope384_final.yaml \
      --job train --build-missing

Record all OCI digests. Run the Policy runtime verifier inside the packed training image and the two-case request/whitespace verifier inside the packed eval image. `pt job diff` must show no unintended objective/runtime/model difference between canary and production training packages.

Run the corrected training canary first:

    export OPD_CANARY_RUN=opdcorr1-nativeq-e2b12b-r16-lb12-step1

    pt job run gemma4_e2b_scope_opd_correctness_c12_lb12.yaml \
      --job train \
      --provider dstack \
      --env HF_TOKEN \
      --timeout-seconds 5400 \
      --run-id "$OPD_CANARY_RUN"

    pt run logs "$OPD_CANARY_RUN" --follow
    pt run wait "$OPD_CANARY_RUN" --timeout-seconds 5400
    pt run reconcile "$OPD_CANARY_RUN"
    pt run checkpoint show "$OPD_CANARY_RUN" --step 1 --files
    pt run checkpoint verify "$OPD_CANARY_RUN" --step 1

Do not start the serving canary unless the complete training-canary gate in Milestone 6 passes.

Run the managed serving canary against the exact step-one model view:

    export OPD_EVAL_CANARY_RUN=opdev01-nativeq-e2b12b-step1-managed-eval

    pt job run gemma4_e2b_scope_opd_eval_serving_canary.yaml \
      --job evaluate \
      --provider dstack \
      --env HF_TOKEN \
      --timeout-seconds 3600 \
      --run-id "$OPD_EVAL_CANARY_RUN" \
      --model-from-run "$OPD_CANARY_RUN" \
      --model-checkpoint-step 1 \
      --model-seat model

    pt run logs "$OPD_EVAL_CANARY_RUN" --follow
    pt run wait "$OPD_EVAL_CANARY_RUN" --timeout-seconds 3600
    pt run reconcile "$OPD_EVAL_CANARY_RUN"

After both receipts pass and the GPU is idle, submit production:

    export OPD_RUN=opdprod2-nativeq-iwopd-e2b12b-r16-lb12-scope384

    pt job run gemma4_e2b_scope_opd_iwopd_scope384_final.yaml \
      --job train \
      --provider dstack \
      --env HF_TOKEN \
      --timeout-seconds 86400 \
      --run-id "$OPD_RUN"

    pt run status "$OPD_RUN"
    pt run logs "$OPD_RUN" --follow

At completion:

    pt run wait "$OPD_RUN" --timeout-seconds 86400
    pt run reconcile "$OPD_RUN"
    pt run checkpoint list "$OPD_RUN"
    for step in 8 16 24 32; do
      pt run checkpoint show "$OPD_RUN" --step "$step" --files
      pt run checkpoint verify "$OPD_RUN" --step "$step"
    done

Start one explicit artifact-export run and add the exact step-32 model, step-32 recovery, native trace, and summary logical names obtained from `pt --json run show "$OPD_RUN"` and `pt --json run checkpoint show "$OPD_RUN" --step 32 --files`. The maintained interface is:

    export MODEL_EXPORT_RUN=opdexp02-nativeq-e2b12b-step32-model
    export MODEL_EXPORT_ROOT="$POLICY_ROOT/.posttrain/state/exports/$MODEL_EXPORT_RUN"

    pt artifact export begin \
      --run-id "$MODEL_EXPORT_RUN" \
      --output "$MODEL_EXPORT_ROOT"

    pt artifact export add "$MODEL_EXPORT_RUN" \
      --source-run "$OPD_RUN" \
      --logical-name "<exact-step-32-model-logical-name>" \
      --destination model
    pt artifact export add "$MODEL_EXPORT_RUN" \
      --source-run "$OPD_RUN" \
      --logical-name "<exact-step-32-recovery-logical-name>" \
      --destination recovery
    pt artifact export add "$MODEL_EXPORT_RUN" \
      --source-run "$OPD_RUN" \
      --logical-name "<exact-native-traces-logical-name>" \
      --destination traces
    pt artifact export add "$MODEL_EXPORT_RUN" \
      --source-run "$OPD_RUN" \
      --logical-name "<exact-summary-logical-name>" \
      --destination summary
    pt artifact export finish "$MODEL_EXPORT_RUN"

Replace only the four angle-bracket values after reading the immutable output links; do not guess them. `finish` must report matching provider/content digests for every receipt.

Capture the provider-neutral run view, then validate the exact exported evidence rather than mutable provider state:

    pt --json run show "$OPD_RUN" > "$MODEL_EXPORT_ROOT/run-view.json"

    cd "$POLICY_ROOT"
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers validate-scope-opd-run \
      --run-id "$OPD_RUN" \
      --run-view "$MODEL_EXPORT_ROOT/run-view.json" \
      --traces "$MODEL_EXPORT_ROOT/traces/artifact/traces.jsonl" \
      --summary-root "$MODEL_EXPORT_ROOT/summary/artifact" \
      --checkpoint-root "$MODEL_EXPORT_ROOT/recovery/artifact" \
      --materialization-root "$MODEL_EXPORT_ROOT" \
      --output "$POLICY_ROOT/.posttrain/state/opd-completion/$OPD_RUN/completion.json"

The command must report `pass: true`, `global_step: 32`, `accepted_targets: 384`, and `teacher_failures: 0` before publication or evaluation.

Publish the verified adapter directory from the model receipt:

    export ADAPTER_DIR="$MODEL_EXPORT_ROOT/model/artifact"
    export HF_MODEL_REPO=carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-lora-v1

    test -f "$ADAPTER_DIR/adapter_config.json"
    find "$ADAPTER_DIR" -name '*.safetensors' -type f

    cd "$POSTTRAIN_ROOT"
    uv run --no-sync --package posttrain-train hf repos create \
      "$HF_MODEL_REPO" \
      --repo-type model \
      --private \
      --exist-ok

    uv run --no-sync --package posttrain-train hf upload \
      "$HF_MODEL_REPO" \
      "$ADAPTER_DIR" . \
      --repo-type model \
      --private \
      --commit-message "Publish corrected Policy Prism E2B from 12B constrained IW-OPD adapter"

    export HF_MODEL_REVISION="$(
      uv run --no-sync --package posttrain-train python -c \
      'import os; from huggingface_hub import HfApi; print(HfApi().model_info(os.environ["HF_MODEL_REPO"]).sha)'
    )"
    test "${#HF_MODEL_REVISION}" -eq 40

Fresh-download that exact revision into a different ignored directory and compare the complete SHA-256 manifest before adding the final Policy model/catalog entries.

After the final immutable Policy model and evaluation work packages validate and pack, run scope then recovery sequentially:

    export SCOPE_RUN=opd2sc32-nativeq-e2b12b-r16-scope-v11
    export RECOVERY_RUN=opd2rc32-nativeq-e2b12b-r16-recovery-v1

    pt job run gemma4_e2b_scope_opd_final_scope_eval.yaml \
      --job evaluate \
      --provider dstack \
      --env HF_TOKEN \
      --env OPENROUTER_API_KEY \
      --timeout-seconds 21600 \
      --run-id "$SCOPE_RUN"

    pt run wait "$SCOPE_RUN" --timeout-seconds 21600
    pt run reconcile "$SCOPE_RUN"

Run the 18-case scientific gate and verify the GPU placement is released before submitting recovery:

    pt --json run show "$SCOPE_RUN" | jq -e '
      .view as $v |
      ($v.run.status == "succeeded") and
      ($v.completeness.state == "complete") and
      ($v.evaluation.expected == 18) and
      ($v.evaluation.included == 18) and
      ($v.evaluation.failures == 0) and
      ($v.evaluation.truncated == 0) and
      ([ $v.evaluation.traces[] | select(.error != null) ] | length == 0) and
      ([ $v.summary[] | select(.key == "trace_sync_complete") | .value ][0] == 1) and
      ([ $v.artifacts.items[] |
         select(.direction == "output" and .kind == "verifiers-evaluation") ] |
       length == 1)
    '

This must print `true`. Also require `pt workers` to show no placement held by the scope run.

    pt job run gemma4_e2b_scope_opd_final_recovery_eval.yaml \
      --job evaluate \
      --provider dstack \
      --env HF_TOKEN \
      --env OPENROUTER_API_KEY \
      --timeout-seconds 21600 \
      --run-id "$RECOVERY_RUN"

    pt run wait "$RECOVERY_RUN" --timeout-seconds 21600
    pt run reconcile "$RECOVERY_RUN"

Run the 17-case scientific gate:

    pt --json run show "$RECOVERY_RUN" | jq -e '
      .view as $v |
      ($v.run.status == "succeeded") and
      ($v.completeness.state == "complete") and
      ($v.evaluation.expected == 17) and
      ($v.evaluation.included == 17) and
      ($v.evaluation.failures == 0) and
      ($v.evaluation.truncated == 0) and
      ([ $v.evaluation.traces[] | select(.error != null) ] | length == 0) and
      ([ $v.summary[] | select(.key == "trace_sync_complete") | .value ][0] == 1) and
      ([ $v.artifacts.items[] |
         select(.direction == "output" and .kind == "verifiers-evaluation") ] |
       length == 1)
    '

This must print `true`. Materialize each exact `verifiers-evaluation` output through a separate, clearly named evaluation-evidence export run, then finalize from Policy Prism:

    export EVAL_EXPORT_RUN=opdexp03-nativeq-e2b12b-sealed-evidence
    export EVAL_EXPORT_ROOT="$POLICY_ROOT/.posttrain/state/native-evals"

    pt artifact export begin \
      --run-id "$EVAL_EXPORT_RUN" \
      --output "$EVAL_EXPORT_ROOT"
    pt artifact export add "$EVAL_EXPORT_RUN" \
      --source-run "$SCOPE_RUN" \
      --logical-name "<exact-scope-verifiers-evaluation-logical-name>" \
      --destination "$SCOPE_RUN"
    pt artifact export add "$EVAL_EXPORT_RUN" \
      --source-run "$RECOVERY_RUN" \
      --logical-name "<exact-recovery-verifiers-evaluation-logical-name>" \
      --destination "$RECOVERY_RUN"
    pt artifact export finish "$EVAL_EXPORT_RUN"

The exporter writes each provider payload below its destination's `artifact/` directory and a verified receipt beside it. It refuses either destination when existing bytes do not match. Then finalize from Policy Prism:

    cd "$POLICY_ROOT"
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers finalize-run \
      --input ".posttrain/state/native-evals/$SCOPE_RUN/artifact" \
      --run-id "gemma-4-e2b-policy-prism-nativeq-iwopd-r16-from-12b-opdprod2-v11-sealed-scope" \
      --serving-metadata ".posttrain/state/finalization/$SCOPE_RUN/serving-metadata.json" \
      --output-root "$POLICY_ROOT/evaluation-runs"

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers finalize-run \
      --input ".posttrain/state/native-evals/$RECOVERY_RUN/artifact" \
      --run-id "gemma-4-e2b-policy-prism-nativeq-iwopd-r16-from-12b-opdprod2-v1-sealed-recovery" \
      --serving-metadata ".posttrain/state/finalization/$RECOVERY_RUN/serving-metadata.json" \
      --output-root "$POLICY_ROOT/evaluation-runs"

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers validate-runs --root evaluation-runs

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
    uv run --no-sync --package policy-prism-normative-verifiers \
    policy-prism-verifiers derive-business-kpis \
      --all \
      --root evaluation-runs \
      --check

The serving-metadata files must be generated from verified PostTrain/HF receipts, not written from guessed digests. Before committing, require 18/17 trace counts, semantic status `complete`, the standard five files in each directory, and catalog entries for both.

## Validation and Acceptance

The implementation is ready for the production GPU only when every deterministic gate passes in the packed images and both live canaries pass within the 90-minute GPU budget. “Ready” means all known deterministic boundaries are closed; it is not a promise that no stochastic output or infrastructure failure can occur.

The experiment is complete only when all of the following are observable:

- the canonical contract describes model-native prefixes and exact completion-token alignment;
- the pinned TRL release returns constrained teacher log probabilities and reproducible allowed-set evidence;
- the PostTrain run proves E2B/12B native prefix digests, exact completion IDs, and q-probability alignment;
- Policy prompts contain no model-facing metadata/hidden labels and use stage-correct invariants;
- the ledger passes cold-start and checkpoint recovery at concurrency 12;
- the canary update and managed serving cases are fully valid;
- production has 32 finite updates over exactly 384 accepted unique targets and four complete checkpoint pairs;
- step 32 is privately published and fresh-download verified;
- sealed scope is 18/18 and recovery is 17/17 operationally complete with Claude evidence;
- both evaluations exist in standard five-file Policy directories and validate;
- TRL, PostTrain, and Policy Prism branches are clean and pushed.

A final domain score below base E2B is a scientifically valid negative outcome if every operational gate above passes. Do not silently extend training, select a sealed-best checkpoint, or change the objective after seeing sealed results.

## Idempotence and Recovery

All generated plans, verification receipts, packs, and artifact exports must be content-addressed and safe to rerun. Commands should reuse a verified cache only when source/config/content digests still match. A changed prompt, schema, model identity, dependency pin, source commit, or task order creates a new package and run ID.

Never reuse a failed run ID. Preserve failed evidence and diagnose it. Retry the same canary after a code fix with an incremented suffix; do not introduce a new parameter arm. Submit recovery only for infrastructure interruption from a complete paired checkpoint, and restore model/trainer/optimizer/scheduler/RNG/ledger together. No cleanup occurs until HF fresh verification and both Policy finalizations succeed.

Do not delete historical checkpoint-96, qualification, Trackio, HF, or evaluation evidence. Do not reset, rebase, force-push, or mutate either main branch. Do not merge current main into the qualified PostTrain experiment branch during this plan.

## Artifacts and Notes

Authoritative historical evidence retained by this plan includes:

- PostTrain historical checkpoint-96 plan/evaluation commit: `8c0d76637cc0eba87a6c3d360a2d92f6a99d29db`.
- Policy historical finalized-evaluation commit: `79627530d907b4e3565ddd912db2327f64f72174`.
- Base E2B non-thinking scope run: `gemma-4-e2b-it-bf16-runpod-a100-sxm-prompt-v2-v11-sealed-scope-20260803`.
- Checkpoint-96 scope run: `gemma-4-e2b-policy-prism-opd-sparse-rkl-r16-from-12b-step96-v11-sealed-scope-20260812`.
- Historical checkpoint-96 private HF revision: `4f1fe9c75031396a11bcc44e2193f96df9003054`.
- Fastest capacity run: logical/physical/accumulation `12/1/12`, 634.11 trainer seconds, 34.74 GiB trainer peak.
- Largest completed physical batch: three, 1,130.05 trainer seconds, 49.83 GiB trainer peak, 67.48 GiB system peak.

The previous 5h38m point estimate equals `634.11 * 32` and excludes new constrained-scoring overhead. After the one-update canary, replace it with:

    measured_training_hours = corrected_canary_trainer_seconds * 32 / 3600

If the corrected step remains 634 to 900 seconds, expect roughly 5.6 to 8.0 trainer hours and about 7 to 10 hours through reconciliation. If it is 900 to 1,200 seconds, expect 8 to 10.7 trainer hours. Above 1,200 seconds, diagnose the implementation before production rather than accepting an unmeasured overnight duration. Scope/recovery evaluation and HF/finalization typically add 2 to 5 hours. The 24-hour provider timeout remains a ceiling.

## Interfaces and Dependencies

At completion, PostTrain public distillation settings expose typed teacher prompt alignment and probability space. `OnPolicyDistillationRequest` validates ordered token mapping, concrete template fingerprints, inference renderer compatibility, and structured grammar requirements. The backend-neutral rollout row carries semantic messages and response-format identity separately from student token IDs. The private TRL adapter constructs model-native prefixes and exact completion alignment.

The TRL fork accepts per-row teacher-native prompt IDs, exact completion IDs, and a structured-output specification; its teacher server returns selected-token IDs, constrained log probabilities, allowed-token counts, and allowed-set digests without generating a completion. Its colocated generation returns processed sampled-token log probabilities when constrained mode is selected.

Policy Prism owns semantic prompt rendering, task schemas, source/decision metadata, staged admission, dynamic graph references, deterministic plan ordering, allocation/replacement, ledger state, and the 384-target completion validator. PostTrain and TRL must not encode Policy-specific legal meanings.

Managed evaluation uses PostTrain's typed vLLM XGrammar backend selection and Policy's per-request generation-only schema plus bounded whitespace. Canonical JSON Schema validation remains Policy-owned after generation.

Credentials are read only from existing protected environment files. Required live services are Hugging Face, the CarbonTeq OCI registry, dstack, Trackio, and OpenRouter for final Claude judging. No secret, signed URL, or raw token is written into commits, manifests, receipts, traces, or this plan.

Revision note (2026-08-12): this revision supersedes the earlier launch plan after a full checkpoint-96 root-cause audit. It retains the proven hardware/configuration findings, adds the required cross-template and constrained-probability corrections, removes KPI redesign and unvalidated semantic-abstention rejection, narrows live preflight to two evidence-driven canaries, and preserves the complete training/publication/evaluation/finalization path.
