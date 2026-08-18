# Complete the resilient Policy Prism Gemma 4 E2B from 12B IW-OPD experiment

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current while executing it. Maintain it according to `docs/templates/PLAN.md`. This revision supersedes the pre-production plan because retained production evidence changed the diagnosis of the long-rules failure.

## Purpose / Big Picture

This plan produces one complete and reproducible Policy Prism on-policy distillation experiment. Base Gemma 4 E2B is the trainable student and base Gemma 4 12B is the frozen teacher. The resulting rank-16 LoRA adapter is published privately to Hugging Face, evaluated on the sealed scope and recovery benchmarks, and finalized into Policy Prism's standard five-file `evaluation-runs` format.

The plan also makes the long run resilient. A malformed or length-limited student attempt remains diagnostic evidence, but it cannot become a training target, a downstream dependency, or by itself terminate hours of otherwise healthy training. Policy Prism retries the same logical target in place and then selects a safe alternate through a globally capacity-proven compatibility graph until it obtains exactly one complete trainable response. PostTrain continues to receive exactly twelve accepted rows per optimizer update.

No system can honestly guarantee that a GPU, network, registry, Trackio, Hugging Face, or OpenRouter service will never fail. The acceptance claim is narrower and testable: every deterministic failure seen so far is reproduced and closed; one rejected student attempt is non-fatal; infrastructure interruptions preserve a paired model/recovery checkpoint; and the experiment is not declared complete unless all 384 targets, 32 optimizer updates, publication checks, and both domain evaluations pass.

## Launch Verdict

The two required live canaries passed and the fresh production run completed 29 finite updates without a numerical, teacher, memory, or selected-output failure. It then stopped at a deterministic rolling-instrument allocation boundary. The corrected Policy plan has an exhaustively proved bounded resilience fallback, and its step-28 checkpoint is scientifically compatible because the existing 336 accepted assignments are unchanged. A first recovery attempt exposed a separate TRL repeated-dataloader skip defect and performed zero updates; that evidence is retained and must not be used as a model result. Resume again from the original step-28 pair only after the pinned TRL post12 loader fix passes exact-image qualification.

## Progress

- [x] (2026-08-07 through 2026-08-13) Implement and qualify E2B-from-12B model-native prompting, exact selected-stage projection, XGrammar-constrained IW-OPD, memory-safe local-teacher scoring, LoRA synchronization, and SQLite checkpoint state.
- [x] (2026-08-10 through 2026-08-13) Fix target-78 reserve matching, target-98 task-owned source IDs, unusable dependency propagation, XGrammar wire-schema projection, Gemma whitespace handling, teacher schema insertion order, and managed evaluation whitespace.
- [x] (2026-08-11) Qualify logical/physical/accumulation `12/1/12` as the fastest tested geometry on the RTX PRO 6000.
- [x] (2026-08-13) Preserve production attempt `opd2prod01-correctedq-e2b12b-c384-r16-v1`; reproduce and fix teacher-side schema-order divergence; replay all 24 retained selected completions successfully.
- [x] (2026-08-14) Preserve failed production attempt `opd2prod02-schemaexact-e2b12b-c384-r16-v1`: 185 outputs admitted, 180 consumed by 15 complete finite optimizer updates, zero teacher failures, no OOM, and a verified paired step-8 checkpoint.
- [x] (2026-08-17) Classify all 197 candidate attempts retained by production attempt two. Prove the seven 16,384-token failures were meaningful unfinished JSON with zero trailing whitespace, not whitespace or repetition loops.
- [x] (2026-08-17) Identify target 171's exact capacity failure and the globally greedy reserve-allocation starvation described below.
- [x] (2026-08-17) Reconcile throughput evidence and reject an unqualified logical-24 production change.
- [x] (2026-08-17) Repair idempotent checkpoint publication, add guarded terminal Trackio recovery, and reconcile the stranded failed run without duplicating its artifacts.
- [x] (2026-08-17) Add dynamic per-request rules capacity, deterministic attempt ordinals, capacity-aware reserve chains, output-feasibility replacement, and selected-stage independence in Policy Prism.
- [x] (2026-08-17) Certify all 1,303 model-facing candidate/stage materializations, strengthen stage-specific semantic coverage, remove trace metadata/label leakage, and retain the existing evidence that the teacher is stronger without another model comparison.
- [x] (2026-08-17) Regenerate the immutable 384-target/96-reserve plan and add resilient canary, production, serving, scope, and recovery work packages.
- [x] (2026-08-17) Add maintained exact release, explicit-target completion, read-only materialization, Trackio write-preflight, and serving-metadata commands. Focused Policy and PostTrain tests pass.
- [x] (2026-08-17) Reconstruct trusted base `sha256:8ea6238d52895e67716b2ca6d5de185e14bd234495a67a5846e99d55e0ce2040`, then reject the first online-RL rebuild after exact inspection showed that it still contained TRL post1.
- [x] (2026-08-17) Isolate the OPD online-RL dependency closure from the shared supervised lock; preserve the shared post1 closure and publish the corrected TRL post7 online-RL runtime `sha256:ac3496de75a61dddc840650d7956b51dc90511cd2e588d6bcff16516d0caa8d7` with lock digest `11e1f0b6f32d656186143a25ef21b527113437b06738249c25a90f7e28f231fd`.
- [x] (2026-08-18) Pack canary image `sha256:1d6baef8cafaee39ae9f311caea53867bca971dc63030a26c0dd98d41c95a433`; verify TRL post7, all 1,303 schemas, 384 targets/480 candidates, and 50 twelve-worker cold-ledger rounds inside that exact image.
- [x] (2026-08-18) Preserve and reconcile admission-only canary `opd3can01-longrules-e2b12b-c12-r16-v1`. It failed before model load or target generation because the resilient 32,768+32,768 trainer envelope exceeded rollout revision 7's 49,152 declared context. Add a distillation static validator and Policy rollout revision 9 with a 65,536 declared context; Policy commit `b6222c7` is pushed.
- [x] (2026-08-18) Pack corrected training-canary image `sha256:ce4c0b89b61af0a612e2a4cf701eeb16a8f8760dea1ae29cd794aba7307b0ae6`; verify TRL post7, all 1,303 schemas, 384 targets/480 candidates, and 50 twelve-worker cold-ledger rounds inside that exact image.
- [x] (2026-08-18) Run and reconcile training canary `opd3can03-resilient-longrules-c12-r16-v2`: all twelve historical/highest-risk targets completed, target 171 closed below its effective cap, one finite update scored 18,178 tokens with zero teacher failures or selected truncations, and the paired step-1 checkpoint plus exact completion receipt validate.
- [x] (2026-08-18) Correct the completion validator to compare explicit canary targets against their immutable non-contiguous production slots. Policy Prism commit `33fd15b` is pushed; the retained canary receipt is valid without rerunning training.
- [x] (2026-08-18) Preserve serving canary `opd3can04-managedserve-step1-r16-v2`: all six model stages completed, but the evaluation adapter incorrectly required training-only sampled-token masks and reported two finalization failures. Separate serving from training finalization in Policy Prism commit `67bd383`, version the immutable serving selection in `5a4817b`, and retain the failed evidence.
- [x] (2026-08-18) Run corrected managed-serving canary `opd3can05-managedserve-step1-r16-v3` against the exact canary checkpoint-1 adapter: 2/2 traces, all six stage admissions accepted with `finish_reason=stop`, zero failures/truncations, complete trace sync, and consistent reconciliation. Pack production image `sha256:d5f6217a9821410baf09055b9f5b2b9d10084426b4bb13433e0c189ae24b9783` with package key `eb23a05e62548578d5ffba55cf99302fc16d52977cd23544f00df8e84e235555`.
- [x] (2026-08-18) Preserve production run `opd3prod01-resilient-e2b12b-c384-r16-v1` after 29 finite updates. The step-28 model/recovery pair contains 336 accepted targets and a valid SQLite ledger; update 29 was not checkpointed and will be replayed. The run stopped only because target 149 had no strict-cap-three fallback under the rolling instrument window.
- [x] (2026-08-18) Add three reviewed reserves without changing any prior candidate identity or primary assignment; prefer the strict cap and permit bounded cap four only when strict allocation is impossible. Prove the exact checkpoint-28 tail and every single remaining-target forced failure complete all 384 targets. Policy commits `f6fdf35` and `775d655` are pushed.
- [x] (2026-08-18) Retain and reconcile zero-update recovery attempt `opd3prod01r1-resume28-e2b12b-c384-r16-v1`. It loaded the exact step-28 recovery pair and resolved the correct max-step-32 configuration, but Accelerate interpreted 336 repeated batches as 336 underlying logical batches, exhausted the 32-batch loader, and exited at global step 28.
- [x] (2026-08-18) Pin and exact-image qualify TRL post12 from `3b3e1a6d1fc53f7e52807e676cc0cd9a020250a9`. Runtime `sha256:2fa925615d103d580790c37b3cfcc0226cc40cbdf05140b4b0ef663354394d04` has lock digest `c46660338b06e25996df1864810bbe23172aec7063fab3e5474406e049ba8468`; exact execution leaves 48 repeated batches beginning at rows 336-347 and ending at 372-383. Resume the original step-28 checkpoint under a new run ID and require updates 29-32 plus the combined 384-target completion receipt.
- [ ] Materialize, verify, document, and privately publish the exact step-32 model adapter.
- [ ] Run sealed scope and recovery sequentially against that adapter; finalize, validate, compare, commit, and push the results.

## What Failed and Why Earlier Validation Missed It

The previous launch did not fail because the GPU ran out of memory, the teacher was unavailable, or the loss became unstable. It failed because execution advanced while the living plan's exhaustive production-schema, allocator-pressure, failure-after-checkpoint, and final live-canary gates were still unrecorded or incomplete. The corrected plan must not repeat that mistake.

The known failure chain is:

| Boundary | Observed failure | Verified correction and proof required now |
| --- | --- | --- |
| XGrammar support | Canonical schemas containing `uniqueItems` were sent directly to XGrammar. | Strip unsupported keys only from a generation copy; retain canonical validation. Compile every packaged wire schema in the exact image. |
| Runtime closure | An early corrected job image still contained the published parent image's older TRL because framework sources install with `--no-deps`. | Inspect imported versions and immutable wheel/source labels inside every actual-job image before GPU admission. Retain the qualified post7 closure. |
| Catalog serialization | Python variants carried chat-template identity, but initial catalog serialization dropped it. | Round-trip fingerprints through the catalog and assert them in the resolved manifest and image. |
| vLLM loader | The custom logits-processor path used dotted class syntax while vLLM 0.25.1 requires `module:Class`. | Keep an exact CLI/loader regression using the installed vLLM version. |
| Selected completion boundary | Verifiers stored a three-token assistant scaffold with a false mask before the sampled suffix; Policy initially hashed the entire node as completion. | Split prompt/completion at the single contiguous selected-mask boundary and seed XGrammar with the scaffold. Preserve exact digest/mask tests. |
| Teacher replay performance | Correct token-serial constrained replay took more than one hour for one teacher call. | Keep the qualified frozen local-teacher forward plus chunked constrained vocabulary projection in TRL post7; gate latency and finite gradients live. |
| Gemma termination | Unbounded whitespace could consume an evaluation output budget. | Preserve bounded Gemma whitespace in training and managed evaluation; run a managed-serving canary that closes JSON and stops. |
| Cross-model prompt | E2B and 12B share token IDs but not chat-template control tokens. | Render the same semantic messages through each immutable native template while preserving exact student completion IDs. Keep prompt/template digest regressions. |
| Probability space | Structured rollouts were sampled from a constrained distribution but scored with incompatible raw probabilities. | Student, teacher, and current-student likelihoods must use the identical XGrammar allowed set and normalized constrained probability. Keep analytic loss/gradient and live alignment tests. |
| Target 78 | Reserve matching was fragmented by prompt profile. | Match by stage/quartile/decision and materialize the selected primary's profile/shape. Keep exact forced-rejection coverage. |
| Target 98 | The small model copied an invalid source identifier. | Put the exact task-owned identifier in canonical and wire-schema enums. Compile and replay primary and reserves. |
| Dependencies | A structurally unusable prefix could trigger a non-required downstream stage. | Only usable dependencies may feed later stages. Failure of a diagnostic stage after the selected stage is accepted cannot invalidate that selected target. |
| SQLite cold start | Twelve concurrent task constructors raced on WAL/DDL initialization. | Initialize under a cross-process lock with bounded busy retry and schema versioning. Repeat fresh 12-worker construction/claims at least 50 times in the exact image. |
| Schema order | Teacher scoring reordered JSON-schema properties and built a different XGrammar state machine. | Preserve the exact rollout-schema insertion order. Replay all retained completions and all generated schemas. PostTrain commit `90291b8` is the current fix. |
| Build context | A dedicated BuildKit worker retained a stale named-context mount after the generated manifest changed. | Inspect the manifest hash inside the packed image. If it differs, restart only the dedicated OPD builder and repack; never trust host context alone. |
| Target 171 | Two meaningful rules responses hit 16,384 tokens; globally safer reserves were already claimed. | Raise rules capacity to the safe request envelope and replace greedy shared allocation with a deterministic capacity/risk-aware compatibility graph and reassignment. |
| Failure finalization | Step 8 was already published, then the error path tried to publish the same logical checkpoint name again. | Same logical name plus same digest is an idempotent reuse; same name plus different digest fails closed. Test checkpoint publication followed by later failure. |
| Evaluation model binding | Existing corrected evaluation packages bind base E2B. | New final packages must resolve the exact step-32 `model-adapter`, never the recovery checkpoint or base model. Assert this before rollout one. |
| Tracking credential | A later Trackio token rotation temporarily prevented new tracked jobs even though earlier qualification had succeeded. | Load the current protected credential file, perform a non-secret authenticated preflight immediately before submission, and fail before GPU allocation if write access is rejected. |
| Prompt qualification | Current code removes trace metadata and label leakage, but its inventory test checks semantic words across an entire profile rather than the exact stage message. | Audit every rendered production message and enforce per-stage legal rubrics. Existing evaluation evidence already establishes that 12B is stronger than E2B, so no extra teacher-versus-student GPU comparison is required. |
| Distillation planning | The resilient settings allowed 32,768 prompt plus 32,768 completion tokens, but rollout revision 7 declared 49,152 total tokens. Distillation had no static seat validator, so validation and packaging passed before the worker rejected the request. | Reuse runtime request selection validation during static work-package preparation and regression-test the mismatch. Rollout revision 9 declares 65,536 for admission while Policy's actual per-call sequence cap remains 40,960. |

## Surprises & Discoveries

- Observation: target 171 did not enter a whitespace or repetition loop.
  Evidence: its primary prompt was 5,206 tokens and therefore had 35,754 completion tokens available within the 40,960 call cap. It generated 16,384 meaningful tokens, 67,054 characters, and 39 rules before ending mid-sentence with zero trailing whitespace. Its attempted reserve had a 9,044-token prompt, 31,916 tokens available, and likewise stopped at the artificial 16,384 cap after 45 rules.

- Observation: global reserve selection made target 171's fallback harder rather than safer.
  Evidence: the primary contained 43 source segments. Its chosen remaining reserve contained 74. Two compatible 20/21-segment candidates had already been claimed by targets 322 and 243. Compatibility alone was correct; greedy global allocation was not capacity-safe.

- Observation: some unexecuted candidates are inherently too large for one legal JSON response even after raising the cap.
  Evidence: the plan has 480 candidates. Above 64 segments are four primary rules, three reserve rules, one full-graph primary, and one evidence primary. The 106-segment target-132 packet spans definitions, prohibitions, duties, records, notices, training, review, and planning. Target 37 has 93 segments and target 283 is a 76-segment full graph. Their prompts can fit while their complete structured outputs cannot be assumed to close inside the 40,960-token request envelope. Prompt-length validation alone is insufficient.

- Observation: the failed run made more progress than the optimizer count alone shows.
  Evidence: retained traces contain 185 accepted unique outputs, but only the first 180 were consumed and scored by 15 complete twelve-row optimizer updates. Five rows of update 16 were admitted before target 171 failed; no backward/update consumed those five. Always report admitted and optimizer-consumed counts separately.

- Observation: logical 12 is supported by sustained production evidence.
  Evidence: logical 4/physical 1 achieved 5.56 scored tokens/s; logical 12/physical 1 achieved 15.23; physical 2 and 3 fell to 9.51 and 10.44 while consuming more memory. Production attempt two sustained 14.49 scored tokens/s across 15 updates.

- Observation: low utilization is caused by heterogeneous autoregressive decode, not a shortage of the configured resident limit.
  Evidence: rollout occupied 95-97% of update time. GPU utilization was about 18% and peak VRAM 58.52 GiB, but the 131 production vLLM calls had a median request batch of two and maximum ten. No call reached the existing resident limit of twelve. Target 171's primary and reserve each consumed about 923 seconds, and its fatal trace spent 1,847.6 seconds in model generation. One long straggler dominates a logical wave; increasing `max_num_seqs` cannot make that straggler finish sooner.

- Observation: logical 24 changes the experiment, not just throughput.
  Evidence: with physical one and accumulation 24, 384 targets yield only 16 policy-refresh/optimizer updates rather than 32. Generating 24 and updating twice would make the second half stale/off-policy after the first update. No evidence shows this produces an equally good student.

- Observation: incomplete prefixes can be teacher-scored but should not be added as auxiliary loss in this production experiment.
  Evidence: the current bridge has one accepted positive-mask output per logical row. Adding a rejected prefix changes target weighting and the qualified objective, cannot teach tokens that were never sampled, and may reinforce a teacher-likely runaway prefix. Retain it as diagnostic evidence and obtain one complete replacement instead.

- Observation: true end-of-run ragged backfill is not a Policy-only change.
  Evidence: PostTrain's Verifiers bridge and IW-OPD trainer require exact input/output cardinality and one nonempty trainable branch per row. This plan uses robust in-slot retry/fallback. A future dynamic sampler/replacement-ID framework contract may add deferred backfill, but it is unnecessary overhead for this run.

- Observation: exact per-stage prompt qualification closed the earlier raw-prompt metadata and profile-wide inventory gaps.
  Evidence: `render_semantic_prompt()` emits only role, objective, instructions, output rules, skeleton, and examples; model-visible context IDs are opaque hashes; request invariants are stage-specific; and the maintained release verifier checks every applicable profile/shape/stage materialization. It rejects trace identifiers, decision labels, Gold/expected answers or counts, stage-inappropriate invariants, altered source text/IDs, token-budget violations, and student/teacher semantic-message divergence before native template rendering.

- Observation: prior qualification did not rebuild the universal/runtime images because their immutable manifests were already present in the internal registry; the current launch discovered that those pinned manifests had since been removed.
  Evidence: registry inspection rejected the configured base and online-RL digests before any GPU submission. Reconstructing the current source/lock closure then exposed two supply-chain gaps: the runtime builder assumed the base already existed, and the internal package index required the machine CA inside the universal base. PostTrain now builds the base before the kind, accepts a typed machine trust bundle without disabling TLS, records a local-rebuild digest instead of impersonating the historical release digest, and uses a 300-second package-download timeout for multi-hundred-megabyte CUDA wheels. These changes affect packaging only; they do not alter the OPD objective or configuration.

- Observation: the framework source pin alone did not determine the TRL installed in an actual job.
  Evidence: exact inspection found `trl==1.9.2.post1` in the first rebuilt parent because job source wheels install with `--no-deps`; the runtime profile and constraint lock remain authoritative for third-party packages. Replacing the shared workspace lock would have silently changed every supervised/evaluation runtime, so the corrected design gives only `online-rl-trl-py312` a dedicated OPD lock. Runtime tests now require the online profile and lock to match `packages/train/pyproject.toml`, while the shared supervised profile remains post1.

- Observation: the provisional `post8` version collided with different bytes already retained by the internal stable index, and Docker could not reliably fetch either Git history or a release asset from GitHub.
  Evidence: the stable index advertised `trl==1.9.2.post8` with wheel digest `993375...`, while the checkpoint-safe candidate wheel was `dbdffd...`. The corrected release is uniquely versioned and tagged as `trl==1.9.2.post12` at `3b3e1a6d1fc53f7e52807e676cc0cd9a020250a9`. Its dedicated runtime context vendors the byte-identical release wheel with digest `18fb203...`, so BuildKit installs a local, hash-verified file without external GitHub access.

- Observation: work-package validation previously did not validate distillation's cross-seat context envelope.
  Evidence: the first resilient canary reached the provider and Trackio but raised `rollout model length must cover distillation prompt and completion limits` before loading either model. GRPO already had a static validator; distillation did not. The shared selection validator is now called by both runtime request construction and static job preparation, and the exact 640-versus-639 regression fails before packaging.

- Observation: a gradient-accumulated IW-OPD checkpoint could restore weights and global step while silently skipping all remaining data.
  Evidence: the step-28 recovery resolved 384 repeated batches over 32 underlying logical batches. Transformers requested that Accelerate skip 336 repeated batches, but `_RepeatBatchDataLoader` exposed the underlying batch sampler, so Accelerate attempted to skip 336 of 32 logical batches and returned an empty loader. TRL post12 exposes a sampler describing the repeated sequence; the exact regression now returns 48 batches and begins at dataset rows 336-347.

## Decision Log

- Decision: keep production geometry at logical 12, physical 1, accumulation 12.
  Rationale: it is the fastest validated geometry and preserves 32 optimizer updates. More physical batching was slower; more resident capacity alone has no effect; logical 24 creates an unqualified convergence trade-off.
  Date/Author: 2026-08-17 / Codex.

- Decision: increase the rules-stage ceiling to 32,768 while retaining the 40,960 per-call sequence cap.
  Rationale: PostTrain already computes `effective_max = min(stage_cap, trainer_cap, sequence_cap - rendered_prompt_tokens)`. Target 171 therefore receives 32,768 tokens and its long reserve 31,916, never exceeding the existing call or model envelope. Evidence and graph caps remain 2,048 and 8,192.
  Date/Author: 2026-08-17 / Codex.

- Decision: declare a 65,536-token rollout context for the resilient 32,768+32,768 trainer envelope without raising Policy's actual 40,960 per-call cap.
  Rationale: the model natively supports the declared envelope, while every production request remains dynamically clipped by the stricter Policy contract. This satisfies framework admission without weakening the source-specific safety cap or reducing the newly required long-rules capacity.
  Date/Author: 2026-08-18 / Codex.

- Decision: do not split legal source units by default.
  Rationale: splitting can destroy parent-child, inherited qualification, and multiple-valid context. First replace an infeasible candidate with another complete same-stratum legal unit. If replacement is impossible, split only at reviewed top-level legal boundaries while copying required parent context. Never use arbitrary token windows. Multiple-valid and constructed-incomplete registry units should normally be replaced because splitting may change their decision class.
  Date/Author: 2026-08-17 / Codex.

- Decision: use deterministic in-slot retry followed by capacity-aware alternate selection.
  Rationale: it maintains the exact fixed twelve-row trainer contract. Every attempt gets a persisted ordinal, seed, reason, and digest. Only one accepted response enters the loss. Rejected attempts remain trace evidence and cannot feed dependencies, while augmenting-path reassignment prevents an early target from permanently consuming a rare target's only safe reserve.
  Date/Author: 2026-08-17 / Codex.

- Decision: compute fallback assignment globally before launch rather than claim compatible reserves greedily at runtime.
  Rationale: rare-degree and long q4/rules targets must receive safe capacity first. A deterministic capacity-one compatibility graph with augmenting-path/min-cost-flow reassignment can protect scarce candidates while preferring lower prompt/segment/output risk. Its exact observed and doubled per-stratum failure vectors must satisfy a Hall-style capacity proof.
  Date/Author: 2026-08-17 / Codex.

- Decision: do not use failure-prefix auxiliary loss in this experiment.
  Rationale: it changes the qualified objective and fixed target weighting, requires ragged trainer rows, and does not teach an unseen valid closing structure. It may be studied separately with its own baseline amendment and live qualification.
  Date/Author: 2026-08-17 / Codex.

- Decision: start final production from base E2B.
  Rationale: changed caps, retry identity, fallback assignment, and plan hashes make the retained step-8 recovery scientifically incompatible. The failed checkpoint remains reproducibility evidence, not a resume source.
  Date/Author: 2026-08-17 / Codex.

- Decision: predeclare step 32 as the final model.
  Rationale: there is no independent non-sealed Gold set for checkpoint selection. Selecting among 8/16/24/32 with the sealed benchmark would leak qualification evidence. Earlier checkpoints are recovery and exploratory evidence only.
  Date/Author: 2026-08-17 / Codex.

- Decision: run only two live canaries and keep their combined GPU time at or below 90 minutes.
  Rationale: existing evaluation evidence already establishes that 12B is stronger than E2B. Offline/exact-image tests close deterministic failures; one production-shaped update proves exact student/teacher prompt handling, target-171 completion, cardinality, constrained probabilities, loss, and artifacts; one managed-serving canary proves the distinct evaluation path. No extra teacher comparison or MTP, cache, prefill, LoRA, LR, memory, or batch sweep is justified.
  Date/Author: 2026-08-17 / Codex.

- Decision: do not bypass the missing runtime manifests by submitting an unverified mutable tag or reusing a historical job image.
  Rationale: the current experiment depends on corrected TRL, PostTrain, and Policy source that earlier images do not contain. Rebuilding and pinning exact digests is the shortest reproducible path; it is a one-time registry repair, not an additional model smoke.
  Date/Author: 2026-08-17 / Codex.

- Decision: isolate TRL post7 to the OPD online-RL runtime rather than updating the shared workspace closure.
  Rationale: only the OPD backend consumes the new constrained IW-OPD fork. A shared-lock replacement would invalidate unrelated SFT, evaluation, and runtime manifests. A dedicated immutable constraint lock makes the actual installed closure explicit and testable without broadening this experiment's change surface.
  Date/Author: 2026-08-17 / Codex.

- Decision: resume the original step-28 checkpoint with TRL post12 rather than restart 336 completed targets.
  Rationale: the Policy tail correction preserves every already accepted candidate and all objective-defining settings. The only trainer change corrects data-position restoration; weights, optimizer, scheduler, RNG, source plan, ledger, batch geometry, and constrained IW-OPD objective remain unchanged. The zero-update recovery produced no optimizer change and is evidence only, not a resume source.
  Date/Author: 2026-08-18 / Codex.

- Decision: keep existing business KPI definitions.
  Rationale: the user explicitly excluded KPI redesign and historical re-derivation from this experiment.
  Date/Author: 2026-08-17 / Codex.

## Outcomes & Retrospective

The experiment is not complete. The current qualified backend has demonstrated stable memory-safe constrained IW-OPD, 15 finite optimizer updates, zero teacher failures, and a valid step-8 checkpoint pair. It has not produced a final model because target-level output capacity and fallback allocation were incomplete, and failure finalization was not idempotent.

At completion, replace this paragraph with exact PostTrain, Policy Prism, and TRL commits; task-plan and selection hashes; OCI image digest; canary and production run IDs; measured update/runtime figures; step-32 artifact and Hugging Face revisions; scope/recovery results; finalized directories; and the explicit `accept`, `revise`, or `reject` qualification decision. Operational completion does not imply scientific improvement: the final scope comparison must use base E2B non-thinking, and recovery must not regress beyond the predeclared tolerance.

## Context and Orientation

Three repositories participate:

* `/home/ali-awais-safdar/Post-Train/posttrain` is the framework repository. Work only on `feat/gemma-policy-prism-opd-e2b-12b`. `packages/train` owns the TRL bridge, constrained loss, checkpoint publication, and artifact callbacks. `apps/cli` owns run recovery and materialization commands. Do not change `main`.
* `/home/ali-awais-safdar/Policy Prism` owns the project environment, task plans, admission, ledger, prompts, catalog overlays, work packages, and permanent evaluation evidence. Work only on `feat/scope-opd-e2b-12b-environment-v1`, currently at pushed commit `775d655`. The environment source commit embedded in a job must be this reviewed commit or an explicitly validated descendant, not merely another repository HEAD.
* `/home/ali-awais-safdar/Post-Train/trl` owns generic constrained IW-OPD trainer behavior. The checkpoint-safe release is `3b3e1a6d1fc53f7e52807e676cc0cd9a020250a9` (`trl==1.9.2.post12`), with the functional loader change at `5c95ef32444bd71c8408d4f94cf19f6bb5b25278`.
* The CarbonTeq Trackio fork owns generic read-only artifact hydration. No checkout is currently present. If the installed `carbonteq-trackio==0.31.5.post12` cannot fetch an immutable artifact without calling `use_artifact`, acquire the fork at its exact consumed source, add a small read-only `Api.artifact(...)` surface with a no-consumer-edge regression, publish it immutably, and update the PostTrain pin. Do not implement host export by opening a cosmetic Trackio run.

The immutable base inputs are:

| Role | Model | Revision |
| --- | --- | --- |
| Student | `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| Teacher | `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Ordered token mapping | both | `059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416` |

The training project is `policy-prism-scope-opd-e2b-12b`. The GPU is one NVIDIA RTX PRO 6000 Blackwell Workstation Edition with 96 GiB. A logical batch is the twelve fresh targets generated for one optimizer update. Physical batch one means the differentiable forward/backward sees one long sequence at a time. Accumulation twelve combines those gradients into one update.

An attempt is one model response for a candidate. An admitted output is structurally complete and eligible for a logical target. A consumed output is one of the twelve admitted rows actually included in a completed optimizer update. A reserve is an alternate complete source candidate. The compatibility graph records which capacity-one reserves can serve which targets and the deterministic cost used when runtime reassignment is needed.

## Frozen Production Configuration

| Setting | Value |
| --- | --- |
| Population | 384 unique logical targets, each consumed once |
| Stage totals | 77 evidence / 230 rules / 77 graph |
| Logical / physical / accumulation | `12 / 1 / 12` |
| Optimizer updates | 32 |
| LoRA | rank 16 / alpha 32 / dropout 0 |
| Learning rate | `1e-5` |
| Scheduler / warmup / clip | linear / 0 / 1.0 |
| IW-OPD gamma / epsilon | `0.5 / 1e-8` |
| Prompt / call / trainer max | 32,768 / 40,960 / 49,152 |
| Stage output caps | evidence 2,048 / rules 32,768 / graph 8,192 |
| Concurrency / resident sequences | 12 / 12 |
| Prefill | chunked, 4,096 tokens |
| Student vLLM memory utilization | 0.20 |
| Teacher backend | colocated Transformers 4.57.6 BF16 |
| Student prefix cache / MTP | off / off |
| Teacher prefix cache | not applicable to the qualified local scoring backend |
| KV / execution / TP | FP8 / eager / 1 |
| Checkpoints | recovery pairs every 4 steps, retain eight; scientific model views at 8, 16, 24, 32 |
| Production seed | `20260807` |
| Provider timeout | 86,400 seconds |

Do not change these during production. Any scientific/configuration change requires a new run ID and a fresh start from base.

## Plan of Work

### Milestone 1: preserve and close the failed-run evidence

First make PostTrain checkpoint publication idempotent. In `packages/train/src/posttrain/train/backends/trl/common.py`, centralize the publication registry used by the periodic callback and the later failure-preservation path. Reuse an existing logical artifact only when its provider identity and content digest are identical. A collision with different bytes must remain a hard error.

Add a regression in `packages/train/tests/test_trl_common.py` that publishes step 8, injects a later trainer failure, calls failure preservation, observes no second upload, and records the run as failed. The error path must reuse the latest already-published complete checkpoint; it must not serialize newer incomplete state under the old step-8 name. Same name plus same digest is a no-op, same name plus different digest fails closed, and genuinely newer complete state gets its own actual-step logical name. Keep Trackio's global duplicate-name rejection strict; idempotence belongs to PostTrain's checkpoint publisher.

Extend `apps/cli/src/posttrain_cli/commands/run_cmd.py` with `pt run recover-terminal-tracking RUN_ID`, a guarded recovery path for a provider-terminal run whose tracking record remained non-terminal because finalization failed. The command must inspect the provider state and immutable run record, refuse a running provider, append the missing terminal evidence once, and reconcile. Do not rewrite metrics or artifacts. Use it to finalize `opd2prod02-schemaexact-e2b12b-c384-r16-v1`, preserving its step-8 artifacts and trace evidence.

Acceptance is a reconciled failed run with Trackio terminal, no duplicate artifact, and unchanged step-8 digests.

### Milestone 2: make long structured targets complete without changing the objective

In Policy Prism's `scope_opd_tokens.py`, raise only the rules cap from 16,384 to 32,768. In PostTrain's production settings, set trainer `max_completion_length` to 32,768. Preserve the 40,960 per-call sequence cap. The existing generator in `packages/train/src/posttrain/train/backends/trl/online_rl.py` must continue to compute the effective limit from the actual rendered prompt, so no request exceeds the sequence envelope.

Add source-derived generation-only `maxItems` and string limits where they can be proven from supplied source structure, but never use Gold answers or expected rule counts. Canonical validation schemas remain unchanged. The bounds must admit every historical valid output. They are guards against impossible unbounded structures, not substitutes for the larger legitimate capacity.

Do not impose a fixed global `maxItems=64`. Historical maximum 54 does not prove that 64 is sufficient for an unseen 93/106-segment legal unit. Compute a reviewed source-specific conservative upper bound from supplied source structure that exceeds every plausible independently operative rule. Replay all historical accepted outputs against it. If that bound cannot fit the request's token envelope, replace or explicitly repartition the candidate before packaging rather than forcing valid JSON to close while omitting rules.

Add an output-feasibility gate for all 480 primaries and reserves. It must estimate the conservative structured expansion from source sections, object skeleton, quotes, and nested qualifications—not merely prompt tokens. Reject or replace any candidate whose complete reviewed legal unit cannot fit `min(32,768, 40,960 - rendered_prompt_tokens)`. Offline reviewed/silver inventories may support this estimate but must never enter model messages, schemas, or traces. Replacement with a complete same-stratum unit is mandatory first. If replacement is impossible, a reviewed top-level legal partition becomes an explicit new candidate/target: copy required parent context, recalculate decision class and every distribution/instrument cap, rebalance the 384-target plan, and regenerate all hashes. A partition cannot transparently masquerade as the old logical target. This gate must explicitly cover target 132's 106-segment rules primary, target 37's 93-segment constructed-incomplete rules primary, target 283's 76-segment full-graph primary, and every candidate above 50 segments. Keep target 171's 43-segment primary for the live canary; demote or remove its 74-segment reserve in favor of the compatible 20/21-segment class.

In `scope_opd_ledger.py`, persist an attempt ordinal, deterministic seed, rejection reason, selected candidate, and request/response digest. In `harness.py`, retry the same candidate once for a structural failure using the next deterministic seed. A length failure under the new larger effective cap immediately advances to the assigned fallback. The rejected text is retained in the trace but receives no loss and cannot feed a later stage.

If the selected stage is already admitted, failure of an optional later diagnostic stage must not invalidate it. If a required prefix is unusable, the whole candidate is rejected and the same logical target selects the next safe alternate through the compatibility graph.

### Milestone 2A: certify the Policy Prism prompt environment

Do not treat YAML parsing or a profile-wide keyword search as prompt qualification. Render the exact model-facing system and user messages for every production primary, reserve, execution profile, shape, and stage after plan regeneration. Write an ignored-state audit receipt containing their hashes and checks. The receipt must prove:

* trace-only `prompt_id`, profile ID, hidden decision class, registry identity, Gold labels, expected answers, and expected counts are absent from model messages;
* model-visible context identifiers are opaque and cannot reveal `constructed_incomplete` or another hidden class;
* source text and task-owned IDs survive the YAML round trip byte-for-byte;
* evidence, rules, and graph requests contain only their stage-specific invariants and exact required dependency;
* a rejected candidate branch never appears in the selected prompt, dependency, mask, or loss row;
* student and teacher receive the same semantic messages rendered through their own pinned native chat templates, with template and prompt-token digests; and
* every rendered request passes prompt, completion, and full-sequence budgets under the final tokenizer and schema.

Replace the current combined-profile keyword gate with a stage-specific semantic rubric for all twelve training prompts. Evidence prompts must fully specify inventory coverage, inherited parent context, exact source preservation, and stopping. Rules prompts must fully specify actors and agency duties; obligations, prohibitions, permissions, rights, powers, definitions, and constitutive effects; zero/one/many decomposition; parent inheritance; conditions and exceptions; exact quotes; multiple-valid and insufficient-context behavior; fixed IDs; duplicate prevention; and terminal JSON. Graph prompts must fully specify admitted-ID ownership, qualifier attachment, source relationships, duplicate prevention, and terminal JSON. Examples across each profile must exercise legally difficult cases rather than merely contain the relevant words.

Review and strengthen any semantically thin prompt before regenerating the plan. The current variation files are materially shorter and contain fewer examples than the benchmark contracts; brevity is acceptable only when the stage rubric and examples prove equivalent legal instructions. The benchmark profile remains in training as explicitly chosen by the user, while sealed source families and text remain excluded. Record one human-readable audit table and one automated snapshot for every exact rendered stage message.

An incorrect but schema-valid legal answer, including a wrong ambiguity or abstention decision, remains trainable. Prompt qualification must not convert semantic mistakes into structural rejection. Only provider errors, non-stop truncation, malformed or schema-invalid JSON, repetition, unknown IDs, or unusable dependencies trigger retry/fallback.

### Milestone 3: replace greedy shared reserves with a capacity-proven plan

Change the Policy plan builder in `scope_opd_data.py` so fallback capacity is planned globally and deterministically. Build a compatibility graph whose reserve nodes have capacity one. Process rare-degree targets first and use runtime augmenting-path/min-cost-flow reassignment when a claim would otherwise strand another target. Cost must prefer lower rendered-prompt tokens, fewer source segments, lower historical output-risk, and unused instrument exposure while retaining exact stage/quartile/decision compatibility and the primary's execution profile/shape. Do not pretend 96 reserves provide 384 independent dedicated fallbacks.

The full plan must prove a complete assignment under:

* the exact twelve rejected attempts observed in production attempt two;
* target 171 and every retained 16K case failing its first attempt;
* a per-stratum failure vector computed from the exact twelve observed rejections, plus its component-wise doubled stress vector;
* restart after any completed checkpoint; and
* concurrent atomic claims without duplicate candidate consumption.

Record both exact failure vectors in the generated summary and this plan. Prove Hall-style capacity for every stressed compatibility subset and runtime reassignment under those vectors. If the existing 96 reserves cannot satisfy the proof, add reviewed complete candidates before launch. Do not weaken legal-family exclusion, sealed isolation, source-domain limits, instrument caps, or exact stage/decision/quartile distributions merely to increase capacity.

Regenerate the plan, selection lock, summary, candidate schemas, and hashes. Record the new values in this document and in immutable catalog entries. Never mutate historical selections or packages.

### Milestone 4: create maintained evidence/materialization surfaces

Add one narrow read-only command:

    pt run artifact materialize RUN_ID --logical-name LOGICAL_NAME [--output DIRECTORY]

It must resolve one exact output edge, fetch the immutable provider version without creating a Trackio consumer run, verify provider manifest and PostTrain content digests, download atomically under ignored `.posttrain/state/materialized-artifacts`, and write a deterministic receipt. Ambiguous names, missing digests, or mismatches fail closed. A cached rerun must revalidate bytes and return the same receipt.

Implement the generic read-only fetch in the Trackio fork if needed, following `docs/tooling/forks.md`: commit/push the fork first, update its `CARBONTEQ_FORK.md`, then update `docs/tooling/trackio/README.md`, the exact package pin, and `uv.lock` in PostTrain. This work may proceed while the long training run uses the already qualified tracking writer, but it must finish before publication/finalization.

Add the Policy Prism completion validator `policy-prism-verifiers validate-scope-opd-completion`, rather than relying on an untracked `/tmp` script. It must validate run view, reconciliation, training summary, all trace rows, checkpoint-32 recovery ledger, plan identities, and materialization receipts. It writes an atomic deterministic ignored-state receipt; it does not create another Trackio run.

Add `policy-prism-verifiers verify-scope-opd-release` as the maintained exhaustive release gate. Given the regenerated plan and retained-attempt fixture, it checks all schema compilation, prompt/output envelopes, exact historical replay, fallback-capacity vectors, cold-ledger stress receipts, and immutable source hashes. The packed job must invoke the same gate inside its isolated image before publication.

Add `pt tracking check --write` as a credential-safe authenticated preflight. It reports endpoint/project/write capability without printing the token and without creating a user-visible training/evaluation run. A rejected or missing token stops before GPU submission.

Add `policy-prism-verifiers build-serving-metadata` to construct and validate finalization metadata from training/evaluation run views, exact materialization receipts, and the immutable Hugging Face revision. This replaces fragile hand-written `jq` objects.

These surfaces may be implemented while images build, but both must pass before training is declared complete or any adapter is published.

### Milestone 5: prove all deterministic boundaries before GPU use

Policy Prism tests must cover:

* all 197 retained candidate attempts, with 185 admissions and twelve classified rejections;
* target 171's exact prompt budgets and successful completion/fallback under the new cap;
* all 480 primaries/reserves and every actual fallback/profile/shape materialization under prompt and call caps;
* output-feasibility of every primary/reserve, explicitly including the 93/106-segment packets and every 74-105-segment candidate;
* canonical and exact XGrammar wire schema compilation, including dynamic graph schemas;
* target 78 forced cross-profile fallback and target 98 exact identifiers;
* deterministic attempt ordinals/seeds across retry and restart;
* global capacity assignment under twice observed correlated failure;
* twelve-thread and twelve-process cold-ledger initialization/claims, repeated at least 50 fresh paths;
* an accepted selected stage surviving an optional downstream diagnostic failure; and
* no rejected branch appearing in selected messages, masks, or loss inputs;
* exact rendered-message snapshots and hashes for every production materialization;
* stage-specific semantic rubrics for all twelve training prompts;
* absence of trace-only and hidden labels from every system and user message; and
* exact student/teacher semantic-message identity before native template rendering.

PostTrain tests must cover:

* constrained student/teacher/current-student log-probability and gradient equality;
* model-native template and schema-order identity;
* exact one-output-per-input bridge cardinality for retry-resolved rows;
* 12/1/12 sliced loss/gradient equivalence;
* periodic checkpoint followed by later failure, with idempotent reuse;
* terminal tracking recovery idempotence;
* read-only artifact materialization digest checks; and
* exact adapter override for evaluation, proving `model-adapter` step 32 rather than base or `training-checkpoint`.

Pack new actual-job images and inspect their immutable contents. Fail before GPU submission unless imported TRL, PostTrain, Policy Prism source, task-plan hash, selection hash, settings, and image labels exactly match the final commits. Compile schemas and rerun target-171 budget/allocation checks inside the image.

### Milestone 6: run only the two necessary canaries

Create a new production-shaped canary selection with exactly these twelve risk boundaries: target 171; target 132 or its reviewed same-stratum replacement/split; target 283 or its reviewed replacement/split; target 37; retained length-failure targets 322, 211, 358, and 372; duplicate-rule targets 243 and 212; and graph-reference targets 354 and 003. If plan regeneration changes logical IDs, freeze the immutable candidate/source identity rather than relying on the old index.

Run one update at 12/1/12 from base E2B and configure the canary to publish a paired checkpoint at step 1. It passes only when:

* 12/12 unique targets are admitted and consumed;
* target 171 completes below its effective limit;
* no request ends `finish_reason=length`;
* the stressed compatibility graph retains a valid alternate assignment;
* rejected attempts remain diagnostic and absent from dependencies/loss;
* bridge cardinality is exactly twelve;
* loss and gradient norm are finite, scored tokens positive, and teacher failures zero;
* ledger, trace, attempt ordinal, seed, and digests agree;
* one model/recovery checkpoint pair publishes exactly once; and
* reconciliation is consistent.

Then run the existing two-case non-sealed managed evaluation canary against that adapter. It passes only when the LoRA is loaded, both JSON responses close and stop, bounded whitespace is active, there are zero truncations/errors, and a Verifiers artifact reconciles consistently.

The combined GPU wall time for the training and serving canaries is capped at 90 minutes. Use only the fixed cases and settings above. A failure stops production and is diagnosed; do not turn this into a parameter sweep.

### Milestone 7: freeze and launch fresh production

Create additive immutable project entries and these new work-package names:

* `gemma4_e2b_scope_opd_resilient_canary12.yaml`
* `gemma4_e2b_scope_opd_resilient_production384.yaml`
* `gemma4_e2b_scope_opd_resilient_serving_canary.yaml`
* `gemma4_e2b_scope_opd_resilient_scope_eval.yaml`
* `gemma4_e2b_scope_opd_resilient_recovery_eval.yaml`

Do not modify historical corrected packages. The production resolved plan must assert 384 targets, 32 steps, physical one, accumulation twelve, concurrency/resident twelve, rules cap 32,768, all immutable source/model hashes, and no checkpoint input.

Use this unique production ID:

    opd3prod01-resilient-e2b12b-c384-r16-v1

Submit from base E2B only. Stay attached through the first finite optimizer update and the first complete model/recovery checkpoint. During the run require, after every completed step, exactly `global_step * 12` consumed unique targets, finite metrics, positive scored tokens, zero teacher failures, no duplicate selected candidate, and a feasible remaining compatibility assignment. Record rejected attempts and fallback pressure by stage/quartile/decision without treating individual rejections as run failure.

Publish paired recovery checkpoints every four updates and retain all eight pairs, limiting an infrastructure interruption to at most three completed updates of replay. At steps 8, 16, 24, and 32, additionally mark the model views used for scientific comparison and verify ledger accepted counts 96, 192, 288, and 384. Provider success alone is insufficient. Reconcile and run the Policy completion validator; require 384 unique consumed targets, 32 finite updates, exact stage/plan distributions, zero selected errors/truncations, all trace sync complete, and consistent artifact digests.

### Milestone 8: publish the exact step-32 adapter

Materialize the step-32 `model-adapter` view, not the `training-checkpoint`. Verify `adapter_config.json`, rank 16, alpha 32, dropout zero, base revision, safetensors readability, file/tree digest, and absence of optimizer, scheduler, RNG, or full base weights.

Create an accurate model card recording student/teacher revisions, all three source commits, run ID, step 32, task-plan hashes, constrained IW-OPD contract, training configuration, artifact identities/digests, limitations, and evaluation status. Publish privately to:

    carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-lora-v1

Resolve the immutable Hugging Face commit, download it fresh at that commit, and compare the complete file SHA-256 manifest. Record both the evaluated weight commit and any later model-card-only commit.

### Milestone 9: evaluate scope then recovery against the adapter

Register one immutable OPD model variant/inference binding with the exact Trackio artifact/content digest, base/teacher revisions, step 32, LoRA config, and Hugging Face weight revision. New evaluation packages must consume this variant or use the explicit checkpoint override:

    --model-from-run "$OPD_RUN" --model-checkpoint-step 32 --model-seat model

Before submission, inspect and materialize the ready step-32 checkpoint model view and
assert kind `model-adapter`, optimizer step 32, exact content digest, base revision,
LoRA rank 16, and the OPD project. Use the managed-serving-canary-qualified `job run`
override path, then assert the retained evaluation input lineage is that exact model
view and contains no `training-checkpoint`. Refuse base E2B or a recovery view.

Run sealed scope first with a unique ID such as `opd3prod01-ckpt32-scope-v11`. Require 18 expected/included, zero failures/truncations/errors, complete trace sync and Claude judging, one Verifiers artifact, provider success, and consistent reconciliation. Only after the GPU is free run recovery as `opd3prod01-ckpt32-recovery-v1`; require the analogous 17-case gate. The two jobs must not overlap on the single dstack target.

### Milestone 10: finalize, compare, decide, and push

Materialize each exact native `verifiers-evaluation` artifact into ignored Policy Prism state using the maintained read-only command. Generate serving metadata from the verified training, adapter, HF, evaluation-image, work-package, and evaluation-artifact receipts; never substitute an evaluation digest for the adapter digest.

Run `policy-prism-verifiers finalize-run` directly into `/home/ali-awais-safdar/Policy Prism/evaluation-runs`. Each directory must contain:

* `manifest.json`
* `traces.jsonl`
* `business-kpis.json`
* `engineering-metrics.json`
* `semantic-diagnostics.json`

Validate all runs and catalog updates. Compare scope only with the base E2B non-thinking scope baseline. Compare recovery only with the corresponding base E2B recovery baseline. Keep the current KPI definitions. Record one explicit decision: `accept`, `revise`, or `reject` under the predeclared domain and recovery criteria.

The model may remain on Hugging Face with an accurate negative-result card if scientifically rejected. Commit and push finalized Policy evidence on the Policy OPD feature branch and the completed living plan on the PostTrain OPD feature branch. Do not modify either repository's main/develop branch.

## Concrete Steps

All secrets are loaded from the maintained PostTrain environment file and must never be printed. Exact generated selection IDs, source commits, image digests, artifact logical names, and HF revisions must be written into this living plan as they become known.

Configure a control terminal from the PostTrain repository:

    export POSTTRAIN_ROOT=/home/ali-awais-safdar/Post-Train/posttrain
    export POLICY_ROOT="/home/ali-awais-safdar/Policy Prism"
    export POSTTRAIN_ENV_FILE="$POLICY_ROOT/.env.posttrain"

    cd "$POSTTRAIN_ROOT"

    pt() {
      UV_CACHE_DIR=/tmp/posttrain-uv-cache \
      uv run --no-sync --package posttrain posttrain \
        --project-root "$POLICY_ROOT" \
        --env-file "$POSTTRAIN_ENV_FILE" \
        "$@"
    }

Before edits, assert repository identity and cleanliness:

    git branch --show-current
    git status --short
    git -C "$POLICY_ROOT" branch --show-current
    git -C "$POLICY_ROOT" status --short

Expected branches are `feat/gemma-policy-prism-opd-e2b-12b` and `feat/scope-opd-e2b-12b-environment-v1`. Preserve unrelated user changes if either tree is dirty.

Commit only the implementation files on the two guarded feature branches before packing:

    cd "$POSTTRAIN_ROOT"
    test "$(git branch --show-current)" = feat/gemma-policy-prism-opd-e2b-12b
    git add docs/plan/policy-prism-gemma4-e2b-12b-opd.md \
      packages/train apps/cli packages/tracking-trackio \
      docs/tooling/trackio uv.lock
    git diff --cached --check
    git commit -m "fix(opd): make resilient production completion reproducible"
    git push origin feat/gemma-policy-prism-opd-e2b-12b

    cd "$POLICY_ROOT"
    test "$(git branch --show-current)" = feat/scope-opd-e2b-12b-environment-v1
    git add packages/normative-verifiers .posttrain/catalog .posttrain/work_packages
    git diff --cached --check
    git commit -m "fix(opd): close long-target and fallback capacity boundaries"
    git push origin feat/scope-opd-e2b-12b-environment-v1

If the Trackio fork is changed, commit and push its dedicated feature branch before the PostTrain pin commit, as required by `docs/tooling/forks.md`. Inspect `git status --short` before every `git add`; never include unrelated user files.

After implementing the guarded recovery, terminalize and reconcile the stranded run exactly once:

    pt run recover-terminal-tracking \
      opd2prod02-schemaexact-e2b12b-c384-r16-v1
    pt run reconcile opd2prod02-schemaexact-e2b12b-c384-r16-v1

Expected recovery reports a provider-terminal failed run, one reused step-8 checkpoint pair, and a saved audit receipt. Repeating it must report an idempotent already-recovered result.

After implementation, run focused tests first and then the relevant release ladders. Exact test paths/names added by implementation must replace the descriptive filters below in this document:

    cd "$POSTTRAIN_ROOT"
    uv sync --all-packages --locked --python 3.13
    uv run pytest packages/train/tests/test_trl_common.py \
      packages/train/tests/test_trl_checkpoint_artifacts.py \
      packages/train/tests/test_trl_sparse_distillation.py \
      apps/cli/tests/test_cli.py -q
    uv run pytest packages/train/tests apps/cli/tests \
      packages/tracking-trackio/tests -q
    uv run ruff check packages/train apps/cli
    uv run pyright packages/train apps/cli
    uv run lint-imports
    git diff --check

    cd "$POLICY_ROOT"
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
      uv run pytest packages/normative-verifiers/tests/test_scope_opd.py -q
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache uv run ruff check \
      packages/normative-verifiers
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache uv run mypy --strict \
      packages/normative-verifiers/src
    git diff --check

Materialize the retained attempt authority and run the maintained host release gate:

    export FAILED_RUN=opd2prod02-schemaexact-e2b12b-c384-r16-v1
    export FAILED_STATE="$POLICY_ROOT/.posttrain/state/opd-release/$FAILED_RUN"
    install -d "$FAILED_STATE"
    pt --json run show "$FAILED_RUN" > "$FAILED_STATE/run-view.json"
    export FAILED_TRACE_LOGICAL="$(
      jq -er '.view.artifacts.items[]
        | select(.direction == "output" and .kind == "evaluation-traces")
        | .logical_name' "$FAILED_STATE/run-view.json"
    )"
    pt run artifact materialize "$FAILED_RUN" \
      --logical-name "$FAILED_TRACE_LOGICAL" \
      --output "$FAILED_STATE/traces"

    cd "$POLICY_ROOT"
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
      uv run --no-sync --package policy-prism-normative-verifiers \
      policy-prism-verifiers verify-scope-opd-release \
        --selection production \
        --retained-traces "$FAILED_STATE/traces/artifact/traces.jsonl" \
        --output "$FAILED_STATE/release-verification.json"

Expected receipt contains `"pass": true`, all 480 candidates, exact observed/doubled per-stratum failure vectors, and the target 171/132/283/37 results. This same command must be an isolated pack-time preflight in both training packages.

Immediately before any live submission, run the credential-safe check:

    cd "$POSTTRAIN_ROOT"
    pt tracking check --write

Expected output identifies `policy-prism-scope-opd-e2b-12b` and `write: available` without displaying a token or creating a run.

Validate and pack the newly created immutable packages from the PostTrain control terminal:

    pt catalog validate
    pt work-package validate gemma4_e2b_scope_opd_resilient_canary12.yaml
    pt work-package validate gemma4_e2b_scope_opd_resilient_production384.yaml
    pt work-package validate gemma4_e2b_scope_opd_resilient_serving_canary.yaml
    pt work-package validate gemma4_e2b_scope_opd_resilient_scope_eval.yaml
    pt work-package validate gemma4_e2b_scope_opd_resilient_recovery_eval.yaml

    pt job plan gemma4_e2b_scope_opd_resilient_production384.yaml --job distill
    pt job pack gemma4_e2b_scope_opd_resilient_canary12.yaml \
      --job distill --build-missing
    pt job pack gemma4_e2b_scope_opd_resilient_production384.yaml \
      --job distill --build-missing
    pt job pack gemma4_e2b_scope_opd_resilient_serving_canary.yaml \
      --job evaluate --build-missing
    pt job pack gemma4_e2b_scope_opd_resilient_scope_eval.yaml \
      --job evaluate --build-missing
    pt job pack gemma4_e2b_scope_opd_resilient_recovery_eval.yaml \
      --job evaluate --build-missing

The exact-image gate must run after packing and before submission. It must print the final source commits, plan hashes, TRL version, image digest, schema compilation result, target-171 budget result, capacity-assignment result, and checkpoint-finalization result. Record that transcript under `Artifacts and Notes`.

Use these canary IDs:

    export TRAIN_CANARY_RUN=opd3can03-resilient-longrules-c12-r16-v2
    export SERVE_CANARY_RUN=opd3can05-managedserve-step1-r16-v3

Submit the training canary, wait, reconcile, and run its scientific gate:

    pt job run gemma4_e2b_scope_opd_resilient_canary12.yaml \
      --job distill \
      --provider dstack \
      --env HF_TOKEN \
      --timeout-seconds 3900 \
      --run-id "$TRAIN_CANARY_RUN"

    pt run logs "$TRAIN_CANARY_RUN" --follow
    pt run wait "$TRAIN_CANARY_RUN" --timeout-seconds 3900
    pt run reconcile "$TRAIN_CANARY_RUN"
    export TRAIN_CANARY_STATE="$POLICY_ROOT/.posttrain/state/completion/$TRAIN_CANARY_RUN"
    install -d "$TRAIN_CANARY_STATE"
    pt --json run show "$TRAIN_CANARY_RUN" > "$TRAIN_CANARY_STATE/run-view.json"
    pt --json run checkpoint show "$TRAIN_CANARY_RUN" --step 1 --files \
      > "$TRAIN_CANARY_STATE/checkpoint-1.json"

    export TRAIN_CANARY_TRACES="$(jq -er '.view.artifacts.items[]
      | select(.direction == "output" and .kind == "evaluation-traces")
      | .logical_name' "$TRAIN_CANARY_STATE/run-view.json")"
    export TRAIN_CANARY_SUMMARY="$(jq -er '.view.artifacts.items[]
      | select(.direction == "output" and .kind == "training-summary")
      | .logical_name' "$TRAIN_CANARY_STATE/run-view.json")"
    export TRAIN_CANARY_RECOVERY="$(jq -er '.recovery.logical_name'
      "$TRAIN_CANARY_STATE/checkpoint-1.json")"

    pt run artifact materialize "$TRAIN_CANARY_RUN" \
      --logical-name "$TRAIN_CANARY_TRACES" \
      --output "$TRAIN_CANARY_STATE/traces"
    pt run artifact materialize "$TRAIN_CANARY_RUN" \
      --logical-name "$TRAIN_CANARY_SUMMARY" \
      --output "$TRAIN_CANARY_STATE/summary"
    pt run artifact materialize "$TRAIN_CANARY_RUN" \
      --logical-name "$TRAIN_CANARY_RECOVERY" \
      --output "$TRAIN_CANARY_STATE/recovery"

    cd "$POLICY_ROOT"
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
      uv run --no-sync --package policy-prism-normative-verifiers \
      policy-prism-verifiers validate-scope-opd-completion \
        --run-view "$TRAIN_CANARY_STATE/run-view.json" \
        --reconciliation "$POLICY_ROOT/.posttrain/state/executions/$TRAIN_CANARY_RUN/reconciliation.json" \
        --traces "$TRAIN_CANARY_STATE/traces/artifact/verifiers-traces.jsonl" \
        --checkpoint-root "$TRAIN_CANARY_STATE/recovery/artifact" \
        --plan-root "$POLICY_ROOT/packages/normative-verifiers/src/policy_prism_normative_verifiers/resources/scope_opd_plan" \
        --expected-targets 12 \
        --expected-updates 1 \
        --expected-target-id scope-opd-0003 \
        --expected-target-id scope-opd-0037 \
        --expected-target-id scope-opd-0132 \
        --expected-target-id scope-opd-0171 \
        --expected-target-id scope-opd-0211 \
        --expected-target-id scope-opd-0212 \
        --expected-target-id scope-opd-0243 \
        --expected-target-id scope-opd-0283 \
        --expected-target-id scope-opd-0322 \
        --expected-target-id scope-opd-0354 \
        --expected-target-id scope-opd-0358 \
        --expected-target-id scope-opd-0372 \
        --output "$TRAIN_CANARY_STATE/completion.json"

Expected output is a deterministic receipt with `"valid": true`, twelve consumed rows, one finite update, zero teacher failures, completed target 171, and a unique paired step-1 checkpoint. `pt --json run show` alone is not the scientific gate.

Run the managed serving canary against the canary's model view using the exact override supported by the resolved job:

    pt job run gemma4_e2b_scope_opd_resilient_serving_canary.yaml \
      --job evaluate \
      --provider dstack \
      --env HF_TOKEN \
      --model-from-run "$TRAIN_CANARY_RUN" \
      --model-checkpoint-step 1 \
      --model-seat model \
      --timeout-seconds 3600 \
      --run-id "$SERVE_CANARY_RUN"

    pt run logs "$SERVE_CANARY_RUN" --follow
    pt run wait "$SERVE_CANARY_RUN" --timeout-seconds 900
    pt run reconcile "$SERVE_CANARY_RUN"

    pt --json run show "$SERVE_CANARY_RUN" | jq -e '
      .view as $v |
      ($v.run.status == "succeeded") and
      ($v.completeness.state == "complete") and
      ($v.evaluation.expected == 2) and
      ($v.evaluation.included == 2) and
      ($v.evaluation.failures == 0) and
      ($v.evaluation.truncated == 0) and
      ([ $v.evaluation.traces[] | select(.error != null) ] | length == 0) and
      ([ $v.summary[] | select(.key == "trace_sync_complete") | .value ][0] == 1)
    '

Do not launch production unless both canary gates pass. Their individual ceilings total 80 minutes, leaving ten minutes for the transition inside one shared 90-minute wall-clock budget measured from the first GPU assignment. Cancel the sequence at the shared deadline even if an individual command still has time.

Submit fresh production:

    export OPD_RUN=opd3prod01-resilient-e2b12b-c384-r16-v1

    pt job run gemma4_e2b_scope_opd_resilient_production384.yaml \
      --job distill \
      --provider dstack \
      --env HF_TOKEN \
      --timeout-seconds 86400 \
      --run-id "$OPD_RUN"

    pt run logs "$OPD_RUN" --follow
    pt run wait "$OPD_RUN" --timeout-seconds 86400
    pt run reconcile "$OPD_RUN"
    pt run checkpoint list "$OPD_RUN"
    for STEP in 4 8 12 16 20 24 28 32; do
      pt run checkpoint verify "$OPD_RUN" --step "$STEP" --deep
    done

    export OPD_STATE="$POLICY_ROOT/.posttrain/state/completion/$OPD_RUN"
    install -d "$OPD_STATE"
    pt --json run show "$OPD_RUN" > "$OPD_STATE/run-view.json"
    pt --json run checkpoint show "$OPD_RUN" --step 32 --files \
      > "$OPD_STATE/checkpoint-32.json"

Resolve and materialize the terminal traces, summary, and step-32 recovery logical names:

    export OPD_TRACES="$(jq -er '.view.artifacts.items[]
      | select(.direction == "output" and .kind == "evaluation-traces")
      | .logical_name' "$OPD_STATE/run-view.json")"
    export OPD_SUMMARY="$(jq -er '.view.artifacts.items[]
      | select(.direction == "output" and .kind == "training-summary")
      | .logical_name' "$OPD_STATE/run-view.json")"
    export OPD_RECOVERY="$(jq -er '.recovery.logical_name'
      "$OPD_STATE/checkpoint-32.json")"

    pt run artifact materialize "$OPD_RUN" \
      --logical-name "$OPD_TRACES" --output "$OPD_STATE/traces"
    pt run artifact materialize "$OPD_RUN" \
      --logical-name "$OPD_SUMMARY" --output "$OPD_STATE/summary"
    pt run artifact materialize "$OPD_RUN" \
      --logical-name "$OPD_RECOVERY" --output "$OPD_STATE/recovery"

Then run:

    cd "$POLICY_ROOT"
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
      uv run --no-sync --package policy-prism-normative-verifiers \
      policy-prism-verifiers validate-scope-opd-completion \
        --run-view "$OPD_STATE/run-view.json" \
        --reconciliation "$POLICY_ROOT/.posttrain/state/executions/$OPD_RUN/reconciliation.json" \
        --traces "$OPD_STATE/traces/artifact/verifiers-traces.jsonl" \
        --checkpoint-root "$OPD_STATE/recovery/artifact" \
        --plan-root "$POLICY_ROOT/packages/normative-verifiers/src/policy_prism_normative_verifiers/resources/scope_opd_plan" \
        --expected-targets 384 \
        --expected-updates 32 \
        --output "$OPD_STATE/completion.json"

Expected output is `"pass": true`. Do not publish on provider success or reconciliation alone.

Resolve and materialize the exact step-32 model view. Do not materialize the recovery view as a model:

    export MODEL_LOGICAL_NAME="$(
      jq -er '.model.logical_name' "$OPD_STATE/checkpoint-32.json"
    )"
    export MODEL_STAGE="$POLICY_ROOT/.posttrain/state/publication/$OPD_RUN/model-step32"
    pt run artifact materialize "$OPD_RUN" \
      --logical-name "$MODEL_LOGICAL_NAME" \
      --output "$MODEL_STAGE"
    export ADAPTER_DIR="$MODEL_STAGE/artifact"

    test -f "$ADAPTER_DIR/adapter_config.json"
    find "$ADAPTER_DIR" -name '*.safetensors' -type f -print -quit | grep -q .
    test ! -e "$ADAPTER_DIR/optimizer.pt"
    test ! -e "$ADAPTER_DIR/scheduler.pt"

Create `$ADAPTER_DIR/README.md` with the verified model card described in Milestone 8. Load the protected Hugging Face credential without printing it and publish privately:

    set -a
    . /home/ali-awais-safdar/.config/posttrain/credentials/huggingface.env
    set +a

    export HF_MODEL_REPO=carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-lora-v1
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
      --commit-message "Publish Policy Prism Gemma 4 E2B from 12B IW-OPD step-32 adapter"

    export HF_WEIGHT_REVISION="$(
      uv run --no-sync --package posttrain-train python -c \
        'import os; from huggingface_hub import HfApi; print(HfApi().model_info(os.environ["HF_MODEL_REPO"]).sha)'
    )"
    test -n "$HF_WEIGHT_REVISION"

    export HF_VERIFY_DIR="$POLICY_ROOT/.posttrain/state/publication/$OPD_RUN/hf-verify"
    test ! -e "$HF_VERIFY_DIR"
    uv run --no-sync --package posttrain-train python -c \
      'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["HF_MODEL_REPO"], revision=os.environ["HF_WEIGHT_REVISION"], local_dir=os.environ["HF_VERIFY_DIR"])'

    (cd "$ADAPTER_DIR" && find . -path './.cache' -prune -o -type f -print0 \
      | sort -z | xargs -0 sha256sum) \
      > "$MODEL_STAGE/source-sha256.txt"
    (cd "$HF_VERIFY_DIR" && find . -path './.cache' -prune -o -type f -print0 \
      | sort -z | xargs -0 sha256sum) \
      > "$MODEL_STAGE/download-sha256.txt"
    diff -u "$MODEL_STAGE/source-sha256.txt" "$MODEL_STAGE/download-sha256.txt"

Record `HF_WEIGHT_REVISION` and both manifests in this plan. If the model card is updated after evaluation, record a second card revision while retaining the evaluated weight revision.

Submit final evaluations sequentially only after the final adapter binding validates:

    export SCOPE_EVAL_RUN=opd3prod01-ckpt32-scope-v11
    export RECOVERY_EVAL_RUN=opd3prod01-ckpt32-recovery-v1

    export FINAL_LAUNCH_STATE="$POLICY_ROOT/.posttrain/state/finalization/$OPD_RUN/launch"
    install -d "$FINAL_LAUNCH_STATE"
    pt --json run checkpoint show "$OPD_RUN" --step 32 --files \
      > "$FINAL_LAUNCH_STATE/checkpoint-32.json"
    jq -e '
      .ready == true and
      .step == 32 and
      .model.kind == "model-adapter" and
      .model.metadata.checkpoint_step == 32 and
      .model.metadata.checkpoint_view == "model" and
      .recovery.kind == "training-checkpoint" and
      .recovery.metadata.checkpoint_step == 32
    ' "$FINAL_LAUNCH_STATE/checkpoint-32.json"

`job plan` and `job pack` intentionally do not accept model overrides. The supported
override is applied atomically by `job run`; the successful managed-serving canary
already proves that path. Retain and inspect the resulting input lineage after each
evaluation instead of inventing an unsupported dry-run command.

    pt job run gemma4_e2b_scope_opd_resilient_scope_eval.yaml \
      --job evaluate \
      --provider dstack \
      --env HF_TOKEN \
      --env OPENROUTER_API_KEY \
      --model-from-run "$OPD_RUN" \
      --model-checkpoint-step 32 \
      --model-seat model \
      --timeout-seconds 21600 \
      --run-id "$SCOPE_EVAL_RUN"

    pt run logs "$SCOPE_EVAL_RUN" --follow
    pt run wait "$SCOPE_EVAL_RUN" --timeout-seconds 21600
    pt run reconcile "$SCOPE_EVAL_RUN"

    pt --json run show "$SCOPE_EVAL_RUN" > "$FINAL_LAUNCH_STATE/scope-run-view.json"
    jq -e --arg training_run "$OPD_RUN" '
      ([.view.artifacts.items[]
        | select(.direction == "input"
                 and .kind == "model-adapter"
                 and .artifact.provider_metadata.checkpoint_step == 32
                 and .artifact.provider_metadata.checkpoint_view == "model"
                 and .artifact.provider_metadata.run_id == $training_run)]
        | length == 1) and
      ([.view.artifacts.items[]
        | select(.direction == "input" and .kind == "training-checkpoint")]
        | length == 0)
    ' "$FINAL_LAUNCH_STATE/scope-run-view.json"

    pt --json run show "$SCOPE_EVAL_RUN" | jq -e '
      .view as $v |
      ($v.run.status == "succeeded") and
      ($v.completeness.state == "complete") and
      ($v.evaluation.expected == 18) and
      ($v.evaluation.included == 18) and
      ($v.evaluation.failures == 0) and
      ($v.evaluation.truncated == 0) and
      ([ $v.evaluation.traces[] | select(.error != null) ] | length == 0) and
      ([ $v.summary[] | select(.key == "trace_sync_complete") | .value ][0] == 1) and
      ([ $v.artifacts.items[]
        | select(.direction == "output" and .kind == "verifiers-evaluation") ]
        | length == 1)
    '

Only after the 18-case gate is true:

    pt job run gemma4_e2b_scope_opd_resilient_recovery_eval.yaml \
      --job evaluate \
      --provider dstack \
      --env HF_TOKEN \
      --env OPENROUTER_API_KEY \
      --model-from-run "$OPD_RUN" \
      --model-checkpoint-step 32 \
      --model-seat model \
      --timeout-seconds 21600 \
      --run-id "$RECOVERY_EVAL_RUN"

    pt run logs "$RECOVERY_EVAL_RUN" --follow
    pt run wait "$RECOVERY_EVAL_RUN" --timeout-seconds 21600
    pt run reconcile "$RECOVERY_EVAL_RUN"

    pt --json run show "$RECOVERY_EVAL_RUN" > "$FINAL_LAUNCH_STATE/recovery-run-view.json"
    jq -e --arg training_run "$OPD_RUN" '
      ([.view.artifacts.items[]
        | select(.direction == "input"
                 and .kind == "model-adapter"
                 and .artifact.provider_metadata.checkpoint_step == 32
                 and .artifact.provider_metadata.checkpoint_view == "model"
                 and .artifact.provider_metadata.run_id == $training_run)]
        | length == 1) and
      ([.view.artifacts.items[]
        | select(.direction == "input" and .kind == "training-checkpoint")]
        | length == 0)
    ' "$FINAL_LAUNCH_STATE/recovery-run-view.json"

    pt --json run show "$RECOVERY_EVAL_RUN" | jq -e '
      .view as $v |
      ($v.run.status == "succeeded") and
      ($v.completeness.state == "complete") and
      ($v.evaluation.expected == 17) and
      ($v.evaluation.included == 17) and
      ($v.evaluation.failures == 0) and
      ($v.evaluation.truncated == 0) and
      ([ $v.evaluation.traces[] | select(.error != null) ] | length == 0) and
      ([ $v.summary[] | select(.key == "trace_sync_complete") | .value ][0] == 1) and
      ([ $v.artifacts.items[]
        | select(.direction == "output" and .kind == "verifiers-evaluation") ]
        | length == 1)
    '

Materialize each exact evaluation artifact and derive every finalization path from receipts:

    export SCOPE_STATE="$POLICY_ROOT/.posttrain/state/finalization/$SCOPE_EVAL_RUN"
    export RECOVERY_STATE="$POLICY_ROOT/.posttrain/state/finalization/$RECOVERY_EVAL_RUN"
    install -d "$SCOPE_STATE" "$RECOVERY_STATE"

    pt --json run show "$SCOPE_EVAL_RUN" > "$SCOPE_STATE/run-view.json"
    pt --json run show "$RECOVERY_EVAL_RUN" > "$RECOVERY_STATE/run-view.json"

    export SCOPE_EVAL_LOGICAL="$(jq -er '.view.artifacts.items[]
      | select(.direction == "output" and .kind == "verifiers-evaluation")
      | .logical_name' "$SCOPE_STATE/run-view.json")"
    export RECOVERY_EVAL_LOGICAL="$(jq -er '.view.artifacts.items[]
      | select(.direction == "output" and .kind == "verifiers-evaluation")
      | .logical_name' "$RECOVERY_STATE/run-view.json")"

    pt run artifact materialize "$SCOPE_EVAL_RUN" \
      --logical-name "$SCOPE_EVAL_LOGICAL" \
      --output "$SCOPE_STATE/native"
    pt run artifact materialize "$RECOVERY_EVAL_RUN" \
      --logical-name "$RECOVERY_EVAL_LOGICAL" \
      --output "$RECOVERY_STATE/native"

    export SCOPE_NATIVE="$SCOPE_STATE/native/artifact"
    export RECOVERY_NATIVE="$RECOVERY_STATE/native/artifact"
    export SCOPE_SERVING_METADATA="$SCOPE_STATE/serving-metadata.json"
    export RECOVERY_SERVING_METADATA="$RECOVERY_STATE/serving-metadata.json"
    export SCOPE_FINAL_ID=gemma-4-e2b-policy-prism-iwopd-r16-from-12b-opd3prod01-v11-sealed-scope
    export RECOVERY_FINAL_ID=gemma-4-e2b-policy-prism-iwopd-r16-from-12b-opd3prod01-v11-sealed-recovery

    test -f "$SCOPE_NATIVE/config.toml"
    test -f "$SCOPE_NATIVE/traces.jsonl"
    test "$(wc -l < "$SCOPE_NATIVE/traces.jsonl")" -eq 18
    test -f "$RECOVERY_NATIVE/config.toml"
    test -f "$RECOVERY_NATIVE/traces.jsonl"
    test "$(wc -l < "$RECOVERY_NATIVE/traces.jsonl")" -eq 17

Create and validate serving metadata from exact receipts, not manually copied values:

    cd "$POLICY_ROOT"
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
      uv run --no-sync --package policy-prism-normative-verifiers \
      policy-prism-verifiers build-scope-opd-serving-metadata \
        --training-run-view "$OPD_STATE/run-view.json" \
        --evaluation-run-view "$SCOPE_STATE/run-view.json" \
        --model-materialization "$MODEL_STAGE/materialization.json" \
        --evaluation-materialization "$SCOPE_STATE/native/materialization.json" \
        --hf-repository "$HF_MODEL_REPO" \
        --hf-revision "$HF_WEIGHT_REVISION" \
        --output "$SCOPE_SERVING_METADATA"

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
      uv run --no-sync --package policy-prism-normative-verifiers \
      policy-prism-verifiers build-scope-opd-serving-metadata \
        --training-run-view "$OPD_STATE/run-view.json" \
        --evaluation-run-view "$RECOVERY_STATE/run-view.json" \
        --model-materialization "$MODEL_STAGE/materialization.json" \
        --evaluation-materialization "$RECOVERY_STATE/native/materialization.json" \
        --hf-repository "$HF_MODEL_REPO" \
        --hf-revision "$HF_WEIGHT_REVISION" \
        --output "$RECOVERY_SERVING_METADATA"

    export MODEL_CONTENT_DIGEST="$(jq -er '.content_digest' "$MODEL_STAGE/materialization.json")"
    jq -e \
      --arg revision "$HF_WEIGHT_REVISION" \
      --arg training_run "$OPD_RUN" \
      --arg model_revision "$MODEL_CONTENT_DIGEST" \
      '.hf_revision == $revision
       and .training_run_id == $training_run
       and .model_revision == $model_revision' \
      "$SCOPE_SERVING_METADATA" "$RECOVERY_SERVING_METADATA"

Finalize from the Policy Prism repository:

    cd "$POLICY_ROOT"
    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
      uv run --no-sync --package policy-prism-normative-verifiers \
      policy-prism-verifiers finalize-run \
        --input "$SCOPE_NATIVE" \
        --run-id "$SCOPE_FINAL_ID" \
        --serving-metadata "$SCOPE_SERVING_METADATA" \
        --output-root "$POLICY_ROOT/evaluation-runs"

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
      uv run --no-sync --package policy-prism-normative-verifiers \
      policy-prism-verifiers finalize-run \
        --input "$RECOVERY_NATIVE" \
        --run-id "$RECOVERY_FINAL_ID" \
        --serving-metadata "$RECOVERY_SERVING_METADATA" \
        --output-root "$POLICY_ROOT/evaluation-runs"

    UV_CACHE_DIR=/tmp/policy-prism-uv-cache \
      uv run --no-sync --package policy-prism-normative-verifiers \
      policy-prism-verifiers validate-runs --root evaluation-runs

    test -f "$POLICY_ROOT/evaluation-runs/$SCOPE_FINAL_ID/manifest.json"
    test -f "$POLICY_ROOT/evaluation-runs/$RECOVERY_FINAL_ID/manifest.json"

The scientific decision is predeclared against these exact base E2B non-thinking directories:

* scope: `gemma-4-e2b-it-bf16-runpod-a100-sxm-prompt-v2-v11-sealed-scope-20260803`;
* recovery: `gemma-4-e2b-it-bf16-runpod-a100-sxm-prompt-v2-v11-sealed-recovery-20260803`.

Mark `accept` only when scope is operationally 18/18, expected rules matched is strictly above the base 40/68, ambiguity handled is at least 12/18, contract/source checks are at least 4/18, required legal text is at least 32/33, and rule subjects captured are at least 16/68. These keep the scope-tangent improvement primary while preventing an apparent gain from discarding basic coverage.

Recovery is a no-regression gate: require operational 17/17, expected rules found at least 223/460, contract/source checks at least 7/17, required legal text at least 238/376, and recovered rule meaning at least 87/460. Mark `revise` when all operational/evidence gates pass but the scope improvement or recovery thresholds do not; mark `reject` when operational reliability fails or the adapter materially regresses both scope coverage and recovery. Record exact metrics and the decision in this plan and the model card.

Update only the verified model card with the final scope/recovery metrics and decision, then record a separate card revision:

    uv run --no-sync --package posttrain-train hf upload \
      "$HF_MODEL_REPO" \
      "$ADAPTER_DIR/README.md" README.md \
      --repo-type model \
      --private \
      --commit-message "Record sealed Policy Prism qualification results"

    export HF_CARD_REVISION="$(
      uv run --no-sync --package posttrain-train python -c \
        'import os; from huggingface_hub import HfApi; print(HfApi().model_info(os.environ["HF_MODEL_REPO"]).sha)'
    )"
    test -n "$HF_CARD_REVISION"

The evaluated weight identity remains `HF_WEIGHT_REVISION`; `HF_CARD_REVISION` records metadata only. Re-download the safetensors at the card revision and verify their digest still equals the evaluated weight manifest.

Commit and push permanent results only after all final validation passes:

    cd "$POLICY_ROOT"
    test "$(git branch --show-current)" = feat/scope-opd-e2b-12b-environment-v1
    git add "evaluation-runs/$SCOPE_FINAL_ID" \
      "evaluation-runs/$RECOVERY_FINAL_ID" \
      evaluation-runs/catalog.json
    git diff --cached --check
    git commit -m "eval(opd): retain E2B from 12B sealed qualification"
    git push origin feat/scope-opd-e2b-12b-environment-v1

    cd "$POSTTRAIN_ROOT"
    test "$(git branch --show-current)" = feat/gemma-policy-prism-opd-e2b-12b
    git add docs/plan/policy-prism-gemma4-e2b-12b-opd.md
    git diff --cached --check
    git commit -m "docs(opd): record final E2B from 12B outcome"
    git push origin feat/gemma-policy-prism-opd-e2b-12b

    git status --short
    git -C "$POLICY_ROOT" status --short

## Validation and Acceptance

The implementation is launch-ready only when all deterministic host and exact-image tests pass, both Git trees are clean and pushed on the two specified feature branches, credentials are accepted without printing them, the registry and Trackio are writable, and the GPU is healthy and idle.

The prompt environment is accepted only when the exact-render receipt and all twelve stage-specific rubrics pass. The training canary is accepted only by all gates in Milestone 6. Production is accepted only when provider state is succeeded, reconciliation is consistent, no required artifact role is missing, 384 unique targets are consumed exactly once, 32 update rows have finite loss and gradient norms, teacher-failure sum is zero, selected outputs have no errors/truncations, alternate assignment remains feasible, trace sync is complete, every four-step recovery pair verifies, and checkpoints 8/16/24/32 have the expected ledger counts.

Scope is accepted operationally only at 18/18 with zero failures/truncations/errors and complete Claude evidence. Recovery is accepted operationally only at 17/17 under the same rules. Scientific acceptance additionally requires the predeclared scope improvement and recovery no-regression decision. A null generic Verifiers mean reward is not itself a failure when Policy Prism's domain metrics are authoritative.

Hugging Face publication is accepted only when a fresh download at the immutable commit produces the same complete file manifest and a loadable adapter. Finalization is accepted only when each directory contains the standard five files, `evaluation-runs/catalog.json` is updated, all run validation passes, and serving metadata separates adapter, evaluation artifact, training, and HF identities.

## Monitoring and Grounded Time Budget

The failed corrected run took 27,258 trainer seconds for 15 complete updates, or about 30.3 minutes per update. A straight 32-update projection is 16.2 trainer hours. Larger per-request capacity may lengthen a few calls, while avoiding discarded 16K retries may recover time. Therefore use the target-171 canary to recalculate, but reserve 14-19 hours for training and another 2-4 hours for publication, scope/recovery, finalization, and commits. The provider timeout remains 24 hours as a safety ceiling.

During production, investigate if the first finite update is absent after 60 minutes, if two consecutive updates exceed 60 minutes without trace/GPU progress, if CUDA OOM occurs, if system memory grows unexpectedly above 85 GiB, if teacher failures become nonzero, or if a fallback stratum approaches proven capacity. Quiet logs alone are not a failure; correlate provider state, trace heartbeat, and GPU work.

Do not try to maximize the GPU utilization percentage. Autoregressive heterogeneous generation is latency-bound, and measured throughput—not utilization—is the optimization objective. Logical 12 remains the best evidence-backed balance of speed and 32 policy updates.

## Idempotence and Recovery

Catalog entries and work packages are additive and versioned. Never reuse a failed or completed run ID. Re-running validation, packing, artifact materialization, completion validation, and final run validation must be idempotent.

An infrastructure-only interruption may resume from the latest complete paired recovery checkpoint when image, code, plan, model identities, batch geometry, optimizer/scheduler, seeds, and ledger are unchanged. Roll back model/optimizer/RNG and SQLite ledger together. Before step 8, restart from base. A deterministic schema, allocation, cap, NaN/Inf, systematic teacher, or code/config failure requires a fresh run and a new ID.

The production attempt in this plan starts fresh because its task contract changes. Do not delete prior runs. Preserve Trackio artifacts, provider workspaces, failed traces, checkpoint pairs, materialization receipts, and finalized evidence until the model is fresh-verified on Hugging Face and both Policy directories validate. Cleanup only the new provider workspaces afterward; never delete Trackio, HF, or `evaluation-runs` evidence as part of cleanup.

## Artifacts and Notes

Historical evidence that must remain traceable:

* failed production: `opd2prod02-schemaexact-e2b12b-c384-r16-v1`;
* admitted outputs: 185; optimizer-consumed outputs: 180; complete updates: 15;
* scored tokens: 394,989; teacher failures: zero;
* step-8 model digest prefix: `ae192ce4`; recovery digest prefix: `df5d999b`;
* fatal target: `scope-opd-0171`;
* retained candidate attempts: 197; accepted: 185; rejected: twelve;
* retained 16,384-token incomplete prefixes: seven, all meaningful and with zero trailing whitespace;
* checkpoint-safe TRL source: `3b3e1a6d1fc53f7e52807e676cc0cd9a020250a9`;
* current PostTrain fix head: `8af9467`; current Policy Prism fix head: `61b0b83`.

As execution proceeds, append concise evidence here: final commits, task/selection hashes, exact package/image digests, exact canary cohort, test counts, canary results, production checkpoints, final adapter receipt, Hugging Face weight/card revisions, evaluation artifact receipts, finalized directory names, domain comparison, and qualification decision.

## Interfaces and Dependencies

Policy Prism must end with:

* stage-specific dynamic output budgeting with rules cap 32,768 and effective sequence clipping;
* generation-only schema bounds derived from supplied source structure, never Gold counts;
* a complete output-feasibility audit that replaces unsafe units or splits only at reviewed legal boundaries;
* ledger attempts identified by logical target, candidate, and ordinal with deterministic seed/reason/digests;
* a deterministic capacity-one compatibility graph with runtime augmenting-path reassignment and an offline Hall-style capacity proof;
* selected-stage independence from optional diagnostic failure;
* a committed `validate-scope-opd-completion` CLI and negative fixtures; and
* new immutable resilient environment/catalog/work-package entries.

PostTrain must end with:

* one idempotent checkpoint publication registry shared by periodic and failure paths;
* guarded terminal tracking recovery for provider-terminal/tracking-nonterminal runs;
* read-only digest-verifying artifact materialization that creates no cosmetic Trackio run;
* deep checkpoint verification backed by that read-only provider fetch rather than Trackio's current `unsupported` result;
* exact evaluation checkpoint model override assertions; and
* regressions for every new behavior.

TRL post12 changes only repeated-dataloader checkpoint positioning relative to the qualified post7 runtime; the constrained objective and model behavior remain unchanged. Trackio's duplicate-artifact rule remains strict. Hugging Face receives only the adapter/model card, never optimizer or recovery state. OpenRouter receives sealed context/model outputs only during the two explicitly authorized Claude-judged final evaluations.

## Revision Note

2026-08-17: rewrote the plan after exact production-attempt-two trace analysis. The prior plan incorrectly described target 171 as a repeated malformed/whitespace loop and proposed insufficient schema bounds. The retained responses were meaningful JSON prefixes stopped by an artificial 16,384-token cap, while greedy reserve allocation had consumed safer candidates. This revision adds the 32,768 dynamic rules capacity, capacity-aware fallback assignment, deterministic in-slot retry, idempotent checkpoint finalization, exact target-171 canary, grounded 12/1/12 throughput decision, fresh-run requirement, corrected evaluation adapter binding, and complete publication/finalization path.
