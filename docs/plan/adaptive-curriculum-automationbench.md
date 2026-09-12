# Add a persistent adaptive curriculum to GRPO training

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`; `.agents/PLAN.md` is not present in this checkout.

## Purpose / Big Picture

After this change, a GRPO training profile can optionally select rollout tasks from observed learning signal instead of relying on a fixed shuffled mixture. The first qualification compares two 20-update runs of the repository's OLMo 3 profile on the same AutomationBench task population. The control arm uses OLMo 3 with its normal shuffled mixture. The treatment arm changes only task exposure by enabling the adaptive curriculum. The comparison will show which classes and tasks were proposed, what reward variation the current student produced, and how the next allocation changed.

The controller changes exposure before generation. OLMo 3 active sampling remains enabled and identical in both arms; it retains useful generated groups for the optimizer after rewards are computed. Holding the model, update rule, active sampling, task population, and training budget constant isolates the effect of adaptive curriculum selection as far as one paired run can.

## Progress

- [x] (2026-09-12) Created the implementation branch without reverting the existing proposal edits, then replayed the controller commits onto `codex/adaptive-curriculum-automationbench-v04` from stable `main` (`2bdae383`), which contains the v0.4 release and its dependency-closure hardening.
- [x] (2026-09-12) Confirmed AutomationBench exposes `domain` as a task-class facet and that the Verifiers bridge carries it into every TRL dataset row.
- [x] (2026-09-12) Confirmed TRL forms repeated prompt groups before rollout generation and OLMo 3 performs active group retention after rewards are computed.
- [x] (2026-09-12) Chose a queued file backend with checkpoint-aligned snapshots for controller persistence.
- [x] (2026-09-12) Added the narrow canonical contract amendment and typed profile configuration.
- [x] (2026-09-12) Implemented the controller, queued file backend, TRL integration, telemetry, and recovery snapshots.
- [x] (2026-09-12) Added unit and integration tests for selection, persistence, schema decoding, trainer composition, and checkpoint ordering.
- [x] (2026-09-12) Added the adaptive AutomationBench/OLMo 3 catalog selection and 20-step work package; both work packages pass static composition validation.
- [x] (2026-09-12) Registered the queued journal and final snapshot as one durable controller-state artifact.
- [x] (2026-09-12) Preserved the v0.4 dependency closure: TRL post8, Verifiers `1f6793f7d46e8a650a54b2a585193b4010578fa6`, and AutomationBench environment revision `1181585ea66c6f89432864a476b5110794afc9fe`.
- [x] (2026-09-12) Passed 463 train and lab tests with 10 expected skips, Ruff, focused Pyright, import boundaries, and diff checks on the v0.4 branch.
- [x] (2026-09-12) Collected one rollout batch from each R1 arm on the older branch; both failed closed before optimizer step one at the LFM actor/vLLM parity gate.
- [x] (2026-09-12) Stopped the R2 vanilla-GRPO/adaptive-OLMo comparison after the vanilla arm completed two healthy optimizer updates because it changed both the update rule and task selection. The replacement comparison uses OLMo 3 in both arms.
- [x] (2026-09-12) Stopped queued adaptive run `lfm26-olmo3-adaptive20-20260912-r3` before admission after discovering that its complete candidate pool was selected before OLMo active sampling began.
- [x] (2026-09-12) Moved OLMo curriculum decisions into each active-sampling refill while retaining one initial selection for algorithms without refill sampling; added decision-stage evidence and a fixed-policy refill test.
- [x] (2026-09-12) Passed the full repository validation after the refill change: 1,712 tests passed with 25 expected skips, plus Ruff, Pyright, import boundaries, and diff checks.
- [x] (2026-09-12) Committed refill-time selection as `fd154ed7` and submitted replacement adaptive run `lfm26-olmo3-adaptive20-20260912-r4` (`pt-66c1c609dfa568471a62b4b9`); it is queued for the RTX PRO worker behind the running control.
- [x] (2026-09-12) Replaced the interim variance-mixture controller with separate class coverage, cumulative task discovery, best-effort step-wide task diversity, and reward-aware predictive evidence.
- [x] (2026-09-12) Validated the revised implementation: the full suite reached 1,674 passing tests with only the new registry-count assertion remaining; after correcting that expected inventory, 109 focused controller, catalog, API, and registry tests passed with Ruff, Pyright, import boundaries, package composition, and diff checks.
- [x] (2026-09-12) Submitted versioned work package `lfm26_automationbench_olmo3_adaptive_20_local_v2.yaml` as run `lfm26-olmo3-adaptive20-discovery-v2-20260912-r1` (`pt-315a5edf6ca3a900440b5af2`); it is queued for `carbonteq-ai-workstation.lan`.
- [ ] Run both 20-update training jobs.
- [ ] Compare run evidence and record the result here.
- [x] (2026-09-12) Separated standard GRPO, OLMo active-sampling, and adaptive-controller evidence in Observatory; added controller-owned projection metrics and a per-step class allocation view beside traces.
- [x] (2026-09-12) Diagnosed the first six adaptive steps: reused tasks with prior positive variance yielded usable groups 61.1% of the time, while reused tasks with prior zero variance yielded 18.4%; the controller nevertheless divided reuse almost evenly because its reward-mean proxy and unbounded class prior overwhelmed task variance history.
- [x] (2026-09-12) Replaced the reward-mean proxy with a posterior over the actual positive-variance admission event, bounded class shrinkage to two pseudo-observations, exposed current task variance and pre-selection task priority in controller audit records, and advanced the adaptive selection revision to `3`.
- [x] (2026-09-12) Added distinct-task uncertainty to class discovery, cumulative class-coverage accounting, adaptive discovery above the 20% floor, and a task score that predicts the next mixed group from both recent reward level and observed variance; advanced the adaptive selection revision to `4`.
- [x] (2026-09-12) Qualified the selection machinery offline over 400 paired synthetic runs and preserved row-level CSV/JSONL, aggregate paired intervals, and an analytical report. Adaptive sampling used fewer candidates in four profiles and more in the deliberately adverse fast-saturation profile.

## Surprises & Discoveries

- Observation: OLMo 3 active sampling reserves up to ten candidate batches and then generates only enough candidates to fill the retained batch.
  Evidence: the installed TRL `GRPOTrainer._prepare_active_sampling_inputs` slices a bounded candidate pool, scores each generated group, and retains groups whose reward standard deviation is nonzero.

- Observation: selecting the reserved candidate pool before entering OLMo active sampling prevents evidence from one refill round from changing the next round's task identities.
  Evidence: TRL holds model weights fixed throughout `_prepare_active_sampling_inputs`, while the earlier framework wrapper replaced all reserved rows before calling that method. The revised adapter requests task groups immediately before each `_generate_and_score_completions` call.

- Observation: a step count is not a sound evidence horizon because a class may receive no examples during a step.
  Evidence: the rollout dataset is sampled by prompt group; evidence freshness must therefore be counted per task from completed groups rather than inferred from optimizer steps.

- Observation: the pinned Trackio fork already supports bounded, named-field Doris reads with per-point attributes and preserves multiple attributed points at one logical step.
  Evidence: framework pin `03ff6e0d7c7458b26a23a69242f519bc700ff920` projects requested metric and attribute keys, applies step bounds and empty-row filtering in Doris, and passed a regression test with equal-valued class points at the same step. A new Trackio table or controller trace type would duplicate existing capability.

- Observation: the live revision-2 controller history predicts OLMo admission, but the implemented priority did not use that prediction directly.
  Evidence: across 118 reconstructed candidate groups from one live snapshot, 36 reused tasks with prior positive variance produced another positive-variance group 61.1% of the time; 38 reused tasks with prior zero variance did so 18.4% of the time. The reuse population was therefore divided almost evenly despite a large observed yield difference. Constant fractional rewards reveal the mismatch: OLMo rejects `[0.2, 0.2, 0.2, 0.2]`, while a Beta-Bernoulli model fitted to reward mean predicts mixed binary outcomes.

- Observation: a fixed 20% discovery ceiling is inefficient when familiar tasks saturate, while a fixed 20% floor alone does not guarantee that adaptive reuse beats broad without-replacement sampling.
  Evidence: the final 40-seed paired simulation lets additional discovery compete with familiar practice by predicted yield. It reduced candidate groups in normal, regression, persistent-noise, and rare-geometry profiles. In the fast-saturation profile it still used 14.25 more candidate groups on average because every fresh task was useful and reused tasks lost contrast almost immediately.

- Observation: uncertainty must depend on the number of distinct tasks supporting a class estimate.
  Evidence: two useful tasks among five produce a much wider Beta posterior and a higher optimistic discovery index than two useful tasks among one hundred. Repeated groups from one task update that task but cannot multiply the class sample size.

- Observation: the machine config contains a retired `[providers.dstack.runtime_secrets]` table while this checkout reads runtime credentials from `[services.runtime_credentials]`.
  Evidence: `posttrain machine show` rejected `providers.dstack.runtime_secrets` as an unknown field. Qualification commands use an ephemeral config copy with only the retired duplicate table removed; the user's config remains unchanged.

- Observation: the R1 jobs were packed from commit `756f943d`, based on the pre-release development branch rather than v0.4. Both completed one 32-group rollout batch and then failed the pre-update parity gate: mean selected-token log-probability delta was `0.458367` for vanilla GRPO and `0.378127` for adaptive OLMo 3, above the `0.05` bound.
  Evidence: that branch selected TRL post5 and the older LFM `language_model.` adapter prefix. The v0.4 branch selects TRL post8, which contains the published row-wise LFM parity repair and native LFM module naming. No R1 optimizer update occurred, so the controller did not cause the divergence.

- Observation: the R1 vanilla rollout took `1396.71` seconds at `97.20` aggregate rollout tokens/s. The adaptive OLMo 3 rollout took `1341.67` seconds at `121.35` aggregate rollout tokens/s.
  Evidence: retained Trackio metrics from the first batches. These measure the whole rollout system, not optimizer throughput; 22 of 32 vanilla and 24 of 32 adaptive episodes were truncated.

## Decision Log

- Decision: Put adaptive curriculum configuration on `GRPOSettings` as an optional profile capability, independent of `GRPOSettings.algorithm`.
  Rationale: task selection is a reusable training capability, while `grpo`, `dapo`, and `olmo3` name different update rules. The requested qualification composes the capability with OLMo 3 without making it an OLMo-only feature.
- Decision: Replace the single `exploration` setting with `class_exploration` and `task_discovery`, both set to 0.2 for this qualification.
  Rationale: class coverage is a probability mixture over eligible classes, while task discovery is a cumulative counted obligation over candidate groups. Treating them as one mixture neither guarantees unseen-task coverage nor prevents repeated task identities.
- Decision: Keep every committed task identity in a current-step exclusion set and sample outside it while distinct candidates remain. If the eligible inventory is exhausted, allow and record a repeat rather than failing the training run.
  Rationale: four generations of one prompt are the intended reward group, so avoiding repeats uses candidate compute better. Diversity is a controller policy and observability concern, not a validity condition for an optimizer update.
  Date/Author: 2026-09-12 / Codex

- Decision: Use the environment-provided `domain` field as the AutomationBench class and `example_id` as task identity.
  Rationale: the framework already preserves both fields. No AutomationBench-specific controller or new difficulty label is needed.
  Date/Author: 2026-09-12 / Codex

- Decision: Begin with equal class and within-class task probabilities, then mix a fixed exploration reserve with normalized recent reward-variance signal.
  Rationale: equal initialization covers unknown classes; recent variance directs practice toward tasks where the current student produces mixed outcomes; the exploration reserve revisits low-signal and previously saturated tasks so model changes can correct stale beliefs. If every score is zero or unavailable, the allocation returns to the equal base mixture.
  Date/Author: 2026-09-12 / Codex

- Decision: Measure recency by each task's last completed rollout groups, not optimizer steps.
  Rationale: optimizer steps do not guarantee class coverage. A bounded per-task evidence window changes only when that task is actually observed.
  Date/Author: 2026-09-12 / Codex

- Decision: Persist an ordered JSONL journal through one bounded writer queue and write an atomic controller snapshot when the model checkpoint callback runs.
  Rationale: ordinary evidence writes stay off the rollout critical path, writer failures remain visible, and resume pairs controller state with the exact model checkpoint instead of replaying newer evidence against older weights.
  Date/Author: 2026-09-12 / Codex

- Decision: Project sampling semantics in Observatory from resolved algorithm settings, active-sampling metrics, and compact controller-owned metrics, with controller events as a compatibility fallback for older runs.
  Rationale: under standard GRPO, zero-variance groups describe the optimizer population; under OLMo 3, the currently emitted zero-variance fraction describes generated candidates that active sampling may discard. Named metric reads avoid scanning high-cardinality event payloads, while the event and state journal remain audit and recovery evidence.
  Date/Author: 2026-09-12 / Codex

- Decision: Show adaptive class allocation as one stacked bar per optimizer step in the trace view, paired with compact new-task, revisit, discovery, and fallback counts.
  Rationale: controller decisions are candidate-selection evidence and belong beside rollout traces for debugging. Aggregating controller metrics on the server keeps high-cardinality task identities out of chart payloads and prevents the frontend from inventing event semantics.
  Date/Author: 2026-09-12 / Codex

- Decision: Reuse Trackio's existing sparse metric history path instead of adding controller-specific Trackio storage.
  Rationale: the pinned fork already projects only requested JSON fields and attributes and drops unrelated rows. The controller emits eight low-cardinality series; Observatory requests those series and reconstructs the class distribution. Trackio remains generic and the current run remains readable through event fallback because its image predates these metrics.
  Date/Author: 2026-09-12 / Codex

- Decision: Estimate familiar-task utility as the posterior probability that the next reward group is mixed, multiplied by a recency-weighted empirical useful-group rate. Use separate fixed-strength class priors for success level and useful-group yield.
  Rationale: reward level makes the score fall as a task approaches all-success or all-failure, while the empirical factor prevents constant continuous rewards at an intermediate value from being mistaken for likely contrast. A fixed two-observation class prior transfers evidence without allowing a large class to erase contradictory recent task history.
  Date/Author: 2026-09-12 / Codex

- Decision: Treat 20% task discovery as a cumulative floor. On nonreserved slots, let unseen-class yield compete with familiar-task yield so the controller may discover more tasks when known tasks have weak predicted contrast.
  Rationale: the controller should need no fixed run length or fixed batch size, and a saturated familiar pool should release budget automatically. The floor still guarantees breadth when reuse appears attractive.
  Date/Author: 2026-09-12 / Codex

- Decision: Qualify the first implementation on one training process.
  Rationale: the requested local RTX PRO 6000 run uses one process. Cross-rank evidence aggregation and synchronized allocation require an explicit distributed design and must not be silently approximated.
  Date/Author: 2026-09-12 / Codex

- Decision: Retain the controller directory as a produced run artifact in addition to checkpoint snapshots and telemetry events.
  Rationale: queued file persistence is useful only if the journal and final snapshot survive remote workspace cleanup and can be inspected with the other run outputs.
  Date/Author: 2026-09-12 / Codex

- Decision: Run R2 from the stable v0.4 code line and its existing TRL post8 runtime instead of transplanting only the parity fix into the older feature branch.
  Rationale: v0.4 already contains the published LFM parity repair, native module naming, runtime locks, and subsequent integration work. Keeping the `0.05` gate unchanged preserves the behavior-policy safety check.
  Date/Author: 2026-09-12 / Codex

- Decision: Compare OLMo 3 with its normal shuffled task mixture against OLMo 3 with adaptive curriculum selection.
  Rationale: changing only the curriculum capability makes any observed difference interpretable. The earlier vanilla-GRPO/OLMo comparison changed the algorithm at the same time and was stopped once the mismatch was recognized.
  Date/Author: 2026-09-12 / Codex

- Decision: Treat initial batch selection and active-sampling refill selection as two compositions of the same controller capability.
  Rationale: algorithms without a refill loop can select only at the generation boundary. OLMo 3 can use newly observed variance between refill rounds because no optimizer update occurs inside that collection phase. The algorithm still owns the retain-or-refill rule; the controller owns only the task identities proposed for each request.
  Date/Author: 2026-09-12 / Codex

## Outcomes & Retrospective

Implementation is validated on the v0.4 branch. R1 runs `lfm26-grpo20-random-20260912-r1` and `lfm26-olmo3-adaptive20-20260912-r1` are diagnostic failures from the older branch, not training results. The mismatched v0.4 R2 comparison was canceled and retained only as diagnostic evidence. The corrected qualification uses OLMo 3 for both the normal-mixture and adaptive-mixture arms.

## Context and Orientation

`packages/train/src/posttrain/train/profiles.py` owns typed algorithm/profile settings. `packages/train/src/posttrain/train/catalog_schema.py` decodes project YAML into those settings. `packages/train/src/posttrain/train/backends/trl/policy_optimization.py` materializes Verifiers rollout examples as a Hugging Face dataset and creates the TRL trainer. Each row includes `example_id` and environment observation facets such as AutomationBench's `domain`.

A task class is the category used by the sampler. For this experiment it is an AutomationBench domain such as `sales` or `support`; internally they are all classes. A task is one concrete rollout example. A prompt group is one task repeated for four fresh student attempts, which GRPO compares to compute relative advantages.

The adaptive controller owns an inventory of tasks, a bounded recent evidence window for each task, current-step exclusions, cumulative discovery accounting, a deterministic decision counter, and a persistence backend. It records reward mean and within-group variance, then predicts whether another group is likely to contain mixed outcomes. An all-failure group and an all-success group both have zero variance but imply different student performance; neither permanently removes a task. Class coverage supplies reassessment, while task discovery spends a counted share on unseen identities.

The existing `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml` owns the model, environment, inference, and training selections for this comparison. The control work package is `apps/lab/.posttrain/work_packages/lfm26_automationbench_olmo3_20_local.yaml`. The revised treatment is `apps/lab/.posttrain/work_packages/lfm26_automationbench_olmo3_adaptive_20_local_v2.yaml`. Their model, environment, inference, OLMo 3 algorithm, and optimization settings match; the treatment adds the versioned `adaptive_curriculum` block.

## Plan of Work

First amend the canonical training-settings contract in `docs/post-training/02-primitives.md` and `docs/post-training/05-apis.md` so an optional curriculum capability may select rollout examples before an algorithm consumes them. The amendment must keep environment task ownership and algorithm identity unchanged.

Add `AdaptiveCurriculum` to `profiles.py` and a matching strict Pydantic schema to `catalog_schema.py`. Its small public surface names the class field, class exploration probability, cumulative task-discovery fraction, recent groups retained per task, and seed. The controller derives equal base weights from the resolved inventory; users do not configure per-class weights for this first profile.

Create a train-owned adaptive curriculum module containing the pure selection/state logic and a persistence protocol. The file implementation will append versioned records to JSONL through one bounded queue. It will expose `flush`, `snapshot`, and `close`; errors raised by the writer thread must surface on the next public operation. Snapshots use write-then-rename so a model checkpoint either has a complete matching controller state or no controller state.

Compose a small TRL subclass around the existing telemetry subclass. For ordinary GRPO-family collection, it replaces the scheduled prompt groups once at the initial generation boundary. For OLMo 3 active sampling, the scheduled candidate pool supplies only bounded capacity: the adapter asks the controller for exactly the missing task groups immediately before every refill generation. When TRL calculates raw rewards, it records one observation per complete task group before making the next refill decision. Allocation evidence names the initial or refill stage and refill round. A trainer callback flushes and snapshots controller state on checkpoint saves. The backend closes the writer in `finally` on success or failure.

Add tests that use small synthetic task inventories and rewards. They must prove equal initialization, high-variance prioritization, exploration-based revisits, fallback when all scores are zero, per-task evidence windows, deterministic replay, ordered queued writes, snapshot restore, invalid class metadata rejection, and unchanged behavior when the profile field is absent. Trainer-wrapper tests must prove that ordinary algorithms select once before generation and OLMo 3 selects again after observing each refill round without requiring a GPU.

Add a new OLMo 3 training selection with the adaptive curriculum configured for `domain`, then bind it in a new 20-update work package. Keep the existing OLMo 3 work package as the normal shuffled-mixture control. Validate both packages before running them.

Run the OLMo 3 control to completion, then run the adaptive OLMo 3 arm on the same machine and resolved task population. Record run identities before waiting. Compare completed updates, rollout cost, reward signal, class/task allocations, active-sampling retention, and terminal artifacts.

## Concrete Steps

All commands run from `/home/hammad/projects/rl` unless stated otherwise.

Implement and test incrementally:

    uv run pytest packages/train/tests/test_adaptive_curriculum.py packages/train/tests/test_catalog.py packages/train/tests/test_api.py
    uv run ruff check packages/train apps/lab
    uv run pyright
    uv run lint-imports
    git diff --check

Validate and launch each work package from `apps/lab`:

    uv run --package posttrain posttrain work-package validate .posttrain/work_packages/lfm26_automationbench_olmo3_20_local.yaml
    uv run --package posttrain posttrain work-package validate .posttrain/work_packages/lfm26_automationbench_olmo3_adaptive_20_local_v2.yaml
    uv run --package posttrain posttrain job run .posttrain/work_packages/lfm26_automationbench_olmo3_20_local.yaml --job train --provider dstack
    uv run --package posttrain posttrain job run .posttrain/work_packages/lfm26_automationbench_olmo3_adaptive_20_local_v2.yaml --job train --provider dstack

The exact run and log inspection commands will be added here after launch because the CLI returns the durable run identifiers.

The revised treatment run is `lfm26-olmo3-adaptive20-discovery-v2-20260912-r1`; its dstack provider id is `pt-315a5edf6ca3a900440b5af2`.

The live Trackio event stream confirms that the treatment process constructed the controller with 160 tasks, seven classes, `class_exploration=0.2`, and `task_discovery=0.2`. Its first step selected eight distinct unseen task identities and recorded `duplicate_fallbacks=0`. The first rollout population was still running when this evidence was recorded, so reward and optimizer comparisons remain pending.

The offline controller experiment is reproducible with `uv run python docs/research/proposals/simulations/controller_policy_experiment.py`. Its row-level outputs and analysis are under `docs/research/proposals/simulations/`. They qualify controller selection and accounting only; the next revision-4 AutomationBench run remains the model-learning qualification.

## Validation and Acceptance

The capability is accepted when profile decoding rejects malformed settings, the pure controller tests pass, the queued backend produces an ordered replayable journal, and a checkpoint contains an atomic controller snapshot that restores the same evidence windows and next allocation.

Both work packages must validate against the same model, OLMo 3 update rule, OLMo 3 active-sampling settings, AutomationBench environment revision, task-mix digest, rollout sampling, optimizer budget, and local inference binding. The control job must complete 20 optimizer updates with no adaptive-controller events. The treatment job must complete 20 optimizer updates and produce controller journal records showing refill round 1 selection, its observed evidence, and a later same-step refill decision made before the optimizer update. OLMo active-sampling retention metrics remain separately visible.

The final report must distinguish observed facts from inference and treat one run per arm as qualification evidence rather than a statistically conclusive quality result.

## Idempotence and Recovery

Unit tests and package validation are safe to repeat. A repeated training command may resume only from a complete retained model checkpoint. At each model checkpoint, the controller callback flushes its journal and writes a matching snapshot inside that checkpoint directory. Resume loads that snapshot; it does not infer controller state from journal entries written after the checkpoint. If the writer thread fails, the job fails rather than continuing with unaudited state.

The existing proposal edits in `docs/research/README.md` and `docs/research/proposals/` predate this branch and must remain intact. Implementation commits will stage only their intended files.

## Artifacts and Notes

The v0.4 AutomationBench environment is pinned to `carbonteq-ai/verifiers-environments` commit `1181585ea66c6f89432864a476b5110794afc9fe`; the framework selects Verifiers commit `1f6793f7d46e8a650a54b2a585193b4010578fa6`. Its resolved training population contains 160 selected tasks, each with four fresh attempts, and declares `domain` as an observation facet. The controller file names and successful R2 run identifiers will be recorded after launch.

## Interfaces and Dependencies

In `packages/train/src/posttrain/train/profiles.py`, define an immutable `AdaptiveCurriculum` value and add `adaptive_curriculum: AdaptiveCurriculum | None` to `GRPOSettings`.

In a new train-owned module, define a controller with operations equivalent to:

    select(group_count: int, step: int) -> tuple[task rows, decision evidence]
    observe(groups: sequence of task identity, class identity, rewards, step) -> observation evidence
    snapshot(path: Path) -> None
    close() -> None

Define a persistence protocol whose initial implementation loads records, enqueues JSON-compatible records, flushes queued writes, writes an atomic snapshot, and closes. Do not introduce an AutomationBench import into `packages/train`; the controller consumes only row metadata named by configuration.

Change note, 2026-09-12: created the implementation plan after source inspection; recorded the separation between curriculum selection and OLMo 3 active sampling, per-task evidence windows, and queued checkpoint-aligned file persistence.

Change note, 2026-09-12: updated progress after implementing the controller and catalog profile; recorded the machine-config compatibility issue and current repository-wide Pyright baseline before remote qualification.

Change note, 2026-09-12: recorded the published dependency-closure alignment found during job packing and made the queued controller directory a durable run artifact.

Change note, 2026-09-12: moved qualification to the stable v0.4/post8 code line after the two post5 R1 parity failures; recorded retained rollout speed and truncation evidence.

Change note, 2026-09-12: corrected the qualification design to compare normal-mixture OLMo 3 with adaptive-curriculum OLMo 3, holding the update algorithm and active sampling constant.

Change note, 2026-09-12: moved adaptive OLMo task choice from an eagerly selected candidate pool to each fixed-policy refill boundary; retained initial-only selection for algorithms without refill sampling.

Change note, 2026-09-12: made within-step uniqueness controller-owned, including across OLMo refill rounds. Exhausting the distinct eligible inventory now records a nonfatal duplicate fallback instead of failing generation. Confirmed the live treatment run entered this path and selected eight unique tasks in its first decision.
