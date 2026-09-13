# Reproducible evaluation selection and explicit measurement

This ExecPlan follows `docs/templates/PLAN.md`. Revision 2, 2026-09-13. Maintain Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective throughout implementation. `.agents/PLAN.md`, referenced by the planning skill, is absent; the repository's complete plan template governs this document.

## Purpose / Big Picture

Developers should be able to choose an evaluation population, allocate a task budget, request repeated executions, and declare how results are weighted without writing environment-specific reporting scripts. Observatory should explain exactly which tasks and attempts support each score. Two checkpoints should be comparable using the same resolved selection, while still acknowledging that backends differ in reproducibility support.

This is a production framework design, not an AutomationBench-only experiment. The present request authorizes planning only. This plan does not launch jobs or change existing evaluation behavior. Implementation includes public Python and catalog interfaces, standard jobs, native Verifiers integration, durable evidence, Observatory HTTP/MCP/export/UI, migration, and real integration qualification. Precision-driven adaptive evaluation and automatic statistical stopping are deferred: they require an additional estimator design. All five basic allocation policies and minimum-coverage allocation are in scope.

## Progress

- [x] 2026-09-13: Inspected evaluation request, environment facet, native adapter, Observatory aggregation, and frontend contracts; wrote revision 1.
- [x] 2026-09-13: Extended the plan with the Observatory evaluation journey, projection contracts, comparison behavior, query boundaries, migration, and acceptance evidence.
- [x] 2026-09-13: Amended the canonical evaluation contracts for manifest-backed selection, task-weighted measurement, and rebuildable Observatory projections.
- [x] 2026-09-13: Added and published generic exact task-key dispatch in CarbonTeq Verifiers commit `89e90cf2c9e942f74cd73874e1a18cd4a4e45f73`; the complete v1 suite passes with credential-gated Prime tests skipped.
- [x] 2026-09-13: Implemented provider-neutral task descriptors, native finite inventory projection, typed filters and allocation policies, deterministic immutable manifests, catalog decoding, and manifest-to-Verifiers dispatch.
- [ ] Implement repetitions, optional reproducibility controls, generation resolution, and retry accounting.
- [ ] Implement explicit aggregation, denominators, and paired comparison.
- [ ] Integrate standard jobs, catalog authoring, preview, and portable artifacts.
- [ ] Implement Observatory reporting, migrations, and documentation.
- [ ] Complete deterministic tests, real backend qualification, and browser acceptance.

## Surprises & Discoveries

The name `TaskSliceMetadata` currently describes one task identity and its attributes. `EvaluationSlice` groups attempts by task identity; `EvaluationFacet` groups observations by semantic attributes. These names do not define a general task-selection contract. Evidence: `apps/observatory/src/posttrain_observatory/models.py` and `traces.py:trace_evaluation_view`.

The existing facet projection accepts multiple values per dimension and compound reports support reject or cross-product membership. This is useful reporting behavior but does not establish allocation probabilities for overlapping groups.

Current overall aggregation averages valid rollout values directly. Missing results and unequal repetition counts can therefore alter task weights. `EvaluationPlan.aggregation` and `.comparison` are dictionaries, not evidence that all proposed estimators already execute.

`trace_evaluation_view` currently reads trace pages into memory up to a safety limit of 5,000 and then computes scores, task groups, facets, and compound breakdowns. A large or highly repeated evaluation can therefore be partial because of a transport limit, and recomputing the same view can repeatedly scan the same evidence.

The current frontend presents facets in preference to task groups when facet evidence exists. That makes task reliability disappear precisely when semantic breakdowns are available. Tasks, facets, and compound breakdowns answer different questions and must remain separately accessible.

Native evaluation currently receives `num_tasks`, `num_rollouts`, and `shuffle`. The local Verifiers runner selects a head or shuffled taskset. `SamplingPolicy` exposes generation controls but no typed per-attempt seed schedule. The adapter reads environment sampling explicitly, so changing only an inference binding may not change actual request settings.

The primary checkout is dirty with unrelated release and curriculum work. Planning inspected HEAD `756f943d41db8c7c469e24a4034d196be2071d8f`; preserve all existing changes. Implementation started from Verifiers pin `36eac9d5e04ef29b584b6fa4f027af00cd76ea19` and now selects published exact-dispatch commit `89e90cf2c9e942f74cd73874e1a18cd4a4e45f73`. Sibling checkout HEADs remain non-authoritative unless selected by an immutable consumer pin.

Implementation uses isolated worktrees `/tmp/rl-evaluation-selection` on `codex/evaluation-selection-measurement` and `/tmp/verifiers-evaluation-task-selection` on `codex/evaluation-task-selection`; the original dirty checkout remains untouched. The pinned Verifiers runner had only fixed shuffle/head selection, but native tasks already exposed stable `key` and content `hash` values. The maintained fork now accepts exact ordered `task_keys` and validates them before episode dispatch.

Target weight and inclusion probability are separate quantities. If a stratum represents one third of the target population and two of its tasks are selected, each selected representative receives one sixth of the estimator weight; its inclusion probability still records two divided by the number of eligible tasks in that stratum.

## Decision Log

Decision: extend existing environment task semantics, evaluation plans, and artifact lineage; do not add a competing dataset store or reporting service. Rationale: environments own task meaning and Observatory owns computed evidence. Date: 2026-09-13.

Decision: keep task selection, repetitions, generation, and score weighting independent. Rationale: balanced selection can estimate either an equal-class target or a population target with explicit weighting. Date: 2026-09-13.

Decision: a repetition requires a fresh execution, not a mandatory seed. Optional seed controls declare scope and actual backend support; matching seeds do not promise matching model outputs. Date: 2026-09-13.

Decision: support overlapping facets through identity deduplication and a declared disjoint allocation partition. Rationale: independently allocating overlapping labels silently distorts inclusion probabilities. Date: 2026-09-13.

Decision: version new behavior and preserve legacy scores as legacy rollout-weighted scores. Rationale: historical reports must not change meaning when the UI is upgraded. Date: 2026-09-13.

Decision: Observatory will show Tasks, Facets, Combined breakdowns, Reliability, and Performance as separate views over one calculation result. Rationale: an overlapping facet is a reporting lens, not a replacement for task-level evidence. Date: 2026-09-13.

Decision: raw traces and the resolved evaluation manifest remain replay authority; a finalized evaluation projection is a versioned, rebuildable acceleration artifact. Rationale: Observatory needs fast repeatable views without creating a second source of truth. Date: 2026-09-13.

Decision: evaluation comparison begins with a compatibility assessment and then reports both full-run and common-task results where meaningful. Rationale: a single delta can hide different populations, generation policies, or missing coverage. Date: 2026-09-13.

## Outcomes & Retrospective

Implementation has begun in revision 2. The selection and exact-dispatch foundation is implemented and focused tests pass; repetitions, measurement, complete Observatory product work, standard-job/CLI preparation, real model qualification, and checkpoint evaluations remain pending. Future updates must distinguish tested behavior from proposed interfaces and record exact backend pins and evidence links.

## Context and Orientation

The repository root is `/home/hammad/projects/rl`, a uv workspace. `packages/environment` owns reusable environment bindings and native evidence projection. `packages/eval` owns evaluation policy and backend adapters. `packages/jobs` composes standard jobs; `packages/work` owns portable execution and artifact consumption. `packages/tracking` owns provider-neutral evidence. `apps/observatory` owns computations, queries, exports, and UI. `apps/cli` is the public developer entry point; `apps/lab` contains examples and qualification configurations, not required runtime imports.

A task is one stable environment-owned problem identity. A facet is one semantic dimension/value attached to that task, such as language=Spanish. A slice is a named predicate over attributes; slices can overlap. A repetition is one planned fresh execution of a task. A retry is another execution attempt for the same planned repetition after an eligible execution error. An allocation stratum is a disjoint task group used to apportion a sampling budget. The target population is the set and weights whose expected performance a score represents.

Canonical authority is `docs/post-training/README.md` and documents 01–06. This work changes public evaluation meaning and therefore requires a narrow amendment: 02 for reusable selections, 03 for existing evaluation-bundle artifact use, 04 for ownership, 05 for public contracts, and 06 for weighting, coverage, and evidence. Preserve the current semantics separating execution failure, truncation, missing evidence, and semantic failure. Update the README amendment index. No new workflow stage or model-lineage primitive is required.

## Observatory Product Experience

Observatory should let a reader answer five questions in order: what population was intended, what actually ran, how the score was computed, where performance differs, and which native traces prove it. The primary journey is:

    Evaluation overview
      -> population and coverage
      -> Tasks | Facets | Combined breakdowns | Reliability | Performance
      -> task and repetition detail
      -> native trace

The overview identifies the subject checkpoint, resolved manifest, allocation policy, repetition count, generation policy, success definition, estimator, and calculator revision. Its headline score is always paired with coverage. A reader sees, for example, “20 of 20 tasks, 58 of 60 valid repetitions; task mean, partial” rather than an unexplained reward number. Counts use distinct labels for eligible tasks, selected tasks, planned repetitions, execution attempts, retries, and valid results.

Population and coverage explains selection before results. It shows eligible and selected tasks by the disjoint allocation stratum, target weight, inclusion probability when defined, planned repetitions, actual attempts, and unresolved coverage. Reporting facets may overlap and appear later; this view must make clear that they did not cause duplicate task dispatch. A prepared but unstarted run can render this view from the manifest alone.

Tasks is the primary evidence table. One row represents one task identity and shows its task-level mean, success frequency, any-of-k and all-of-k when complete, planned and valid repetitions, retry/error/truncation counts, and facet labels. Opening a row shows planned repetition slots. Each slot expands into its execution attempts in order, including retry reason, applied seed or unsupported control, initialization identity, and the native trace link. This hierarchy prevents a retry from looking like an additional independent repetition.

Facets reports every declared dimension/value independently and permits overlap. Combined breakdowns renders only explicitly requested two-dimensional matrices and exposes missing or rejected memberships. Both views display eligible tasks, selected tasks, observed tasks, valid repetitions, target weight when applicable, score, and coverage. Reliability shows the distribution across repetitions, disagreement, and within-task variation; it does not treat repetitions as independent population samples. Performance shows latency, tokens, tool use, and execution failures without mixing operational efficiency into semantic quality.

The comparison journey starts with a compatibility panel for manifest population and revision, task identities, success definition, generation and initialization policy, repetition plan, estimator, and calculator revision. A strict matched comparison rejects material mismatches. A descriptive comparison may proceed with a visible warning, showing each run's full-population result and a separately labeled common-task result. The main evidence is an overall paired delta with its supported uncertainty statement, followed by per-task deltas, per-facet deltas, regressions, improvements, missing pairs, and links to the paired repetition evidence. Observatory must never turn a common-task subset into an unlabeled substitute for the requested population.

Live runs show planned coverage beside observed coverage and label all estimates provisional. Finalization produces a content-addressed projection receipt after evidence ingestion reaches a terminal revision. If evidence later changes through an authorized repair, the projection receives a new evidence revision and digest; stale projections are detectable and rebuildable.

## Plan of Work

### Milestone 1: establish inventory and dispatch contracts

First amend the canonical documents with the decisions above. Inspect the exact pinned Verifiers `TaskData`, `Taskset`, evaluation runner, repetition/resume logic, and client request construction using `git show` against the pin. Prove that tasks can be enumerated without generating model answers, identified stably, and reloaded for each repetition. Do not infer facets by performing evaluation first.

Add provider-neutral `TaskDescriptor` and `TaskFacetValue` to `packages/environment/src/posttrain/environment/requests.py` or a focused new `task_inventory.py`. A descriptor contains source-scoped identity, task revision or fingerprint, source reference, split, declared facets, and environment-supported initialization controls. Reuse the existing facet-field projection behavior through `verifiers_evidence.py`; Observatory maps this shared representation to its presentation models without becoming a dependency of environment or eval. Enumeration must not initialize every expensive task container. Large inventories may stream descriptors; v1 requires a finite, explicitly bounded population. Unbounded generated populations must first declare a finite generation selection.

Add an eval backend-private inventory/dispatch adapter under `packages/eval/src/posttrain/eval/backends/verifiers/`. Demonstrate exact identity dispatch without relying on the positional head of a reordered taskset. If the pinned native API cannot accept selected tasks or expose repetition controls, implement a generic native seam in an isolated checkout of `../verifiers` based on the consumer pin. Do not copy its runner into RL. Fork tests must cover exact selection, fresh task loading, repetitions, retry identity, and optional seeds. Publish that fork change before updating consumer pins.

Acceptance: two environments, one with no facets and one with multiple labels, expose inventories and execute the exact requested identities even after inventory enumeration order changes. Task identity collisions with different content fail before model inference.

### Milestone 2: selection and manifest resolution

Introduce `EvaluationSelectionPolicy`, `ResolvedEvaluationManifest`, and `resolve_evaluation_selection(inventory, policy)` in new `packages/eval/src/posttrain/eval/selection.py`. Extend `EvaluationPlan` and `catalog_schema.py` with typed policy references. The manifest uses the existing evaluation-bundle artifact mechanism: record inventory fingerprint, eligible population, selected task references, target weights, allocation strata, per-task inclusion probabilities where defined, quota counts, selection RNG version/seed, and manifest digest. It is reusable across models and contains no model-specific outputs or credentials.

Policies are full population; uniform distinct-task subset; proportional stratification; balanced strata; custom stratum weights; and minimum-per-stratum then proportional remainder. Filters support exact values, membership, conjunction, and disjunction over declared facets and split. Single-valued dimensions and their tuples form disjoint strata. For multi-valued dimensions, v1 supports explicitly selected canonical membership-set strata (for example the sorted set {algebra, geometry}) or requires an explicit single allocation label. Do not silently interpret balanced overlapping memberships as balanced labels. Reporting slices retain all memberships regardless of allocation partition.

Uniform sampling is without replacement. Stratified sampling selects uniformly within each disjoint stratum after deterministic capped largest-remainder allocation. Tie breaking uses stable stratum keys. Apply feasible minimum quotas first, then weighted remainder, redistributing saturated-stratum surplus to eligible strata. Reject infeasible explicit minimums with a useful diagnostic. Default task-budget exhaustion behavior is an error; an explicit use-all policy returns fewer tasks and records the shortfall. Missing allocation facets default to error, with explicit missing bucket or exclusion alternatives and recorded exclusion counts. Missing reporting facets do not remove tasks from the overall population.

If a target stratum has positive target weight and receives no evaluation tasks, mark its target score unavailable or partial; never silently renormalize the represented strata to 100%. Population-weighted stratified estimates use declared target stratum masses and within-stratum task means. Unsupported weighting estimators must fail validation instead of guessing weights for custom nonuniform task sampling.

Acceptance: identical population/policy/seed produces the same selected identities and digest irrespective of input order. A task belonging to two reporting slices appears only once in the selection. Fixed manifests consumed by two subjects have identical planned task identities.

### Milestone 3: execution and generation policy

Extend `EvaluationBudget` and `EvaluateRequest` in `requests.py`, and `_build_native`/`_native_sampling` in `backends/verifiers/adapter.py`. Preserve `num_rollouts` as the ordinary repetitions setting; no second competing repetition count. Add typed initialization and optional reproducibility controls. Default repetitions reload the same task with a fresh isolated environment. Explicit varied-initialization repetitions require native support and record resulting initialization identity. Separate task-selection, environment, and model-generation RNG domains using a versioned stable derivation from an optional root seed, task identity, and repetition index. Do not use process-randomized Python hash values. When no root seed is supplied, record the generated seed where controllable. Record supported, applied, and unsupported controls; optional controls may degrade visibly, required controls reject before dispatch.

Every planned repetition has a stable identity independent of retries. Execution attempts have separate attempt numbers and statuses. A semantic failure is a valid result and never triggers a retry. Retry only declared execution failures; preserve all attempt evidence and select the first valid terminal result under the explicit retry policy. Exhausted attempts remain unresolved or failed coverage. Re-running a completed repetition is an explicit new evaluation, not a way to replace an unfavorable answer. Native snapshots or fresh task construction must prevent tool-state leakage between repetitions and subjects.

Generation resolution must have one effective source: a versioned evaluation-purpose inference binding, including an optional curated model-recommended preset. Environment limits remain compatibility constraints. Explicit job overrides create a recorded resolved revision/digest. During migration legacy environment sampling is preserved under the legacy schema; conflicting new sources fail rather than silently overriding. Presets snapshot values and source revision; no online model-card scraping at runtime. Unset top-p is distinct from explicitly setting it, and actual backend defaults must be captured or made explicit for paired comparisons. LFM2.5's pinned recommendation (temperature 0.1, top-k 50, repetition penalty 1.1) is a qualification example, not a universal default. Seed support and penalties are validated against the actual backend.

Acceptance: three repetitions yield three logical results per selected task, even when one requires a retry. All retries remain inspectable. Changing concurrency or request order does not change derived attempt seeds; exact model output reproducibility is not promised.

### Milestone 4: measurement and comparisons

Replace unvalidated aggregation/comparison dictionaries for the new schema with typed `EvaluationMeasurementPolicy`. Keep normalized intent in eval and computed statistics in new `apps/observatory/src/posttrain_observatory/evaluation_measurement.py`, used by existing service, HTTP, MCP, and exports. Raw attempt results and the manifest remain authoritative; any aggregate cache records calculator version, population digest, policy, and evidence revision.

Default semantic measurement first computes each task's mean over valid planned repetitions, then averages tasks equally. Always expose planned/observed unique tasks, planned/completed/valid repetitions, terminal execution failures, truncations, missing scores, and retry counts. Incomplete task results are explicitly conditional on available evidence; strict qualification scores require the requested coverage. Do not silently remove missing tasks and report a complete population estimate. A separately named operational-success score may count execution failures or truncation as unsuccessful delivery; this does not alter semantic success.

Support task mean, equal-slice macro for a declared dimension or named slice collection, and declared target-distribution weighting. Name the macro axis: there is no single macro score over unrelated dimensions. Overlapping slice macro intentionally repeats task influence through memberships and must disclose that interpretation. Overall task means always deduplicate identities. Report repetition success frequency, any-of-k and all-of-k, with k explicit. Do not label observed any-of-k as a general unbiased pass@k estimator. Do not claim all-of-k or any-of-k complete when repetitions are missing.

Add paired comparisons keyed by environment revision, manifest task identity, and execution-policy identity. Show full-population scores and a separately labeled common-task comparison when coverage differs. Different generation settings or initialization policies produce a compatibility warning or a rejected strict comparison. Estimate uncertainty with a deterministic, paired task-cluster bootstrap, preserving strata/weights and keeping all repetitions of a task together; do not treat repeated attempts as independent tasks. For a tiny number of tasks or a census with no task-sampling estimand, label the uncertainty scope and suppress unsupported intervals rather than implying false precision.

Acceptance includes a numerical oracle: task A has scores 1,1,1 and task B has score 0 with two missing attempts. The observed task-mean estimate is 0.5, legacy rollout mean is 0.75, completion is 4/6, and strict complete score is unavailable. For disjoint classes with target masses 0.9 and 0.1 and class means 1 and 0, balanced selection yields population score 0.9 and macro score 0.5. Duplicating evidence pages or retry rows changes neither result.

### Milestone 5: developer experience and standard jobs

Expose the typed policies through existing Python catalog decoders and YAML evaluation seats, updating `packages/jobs/src/posttrain/jobs/definitions.py`, `packages/work` artifact binding/packing, and `apps/cli` planning commands. Both authoring forms must resolve to identical policies. Preserve detached planning: inspect declared contracts without importing ML backends; perform native inventory materialization explicitly in an appropriate runtime. A prepared manifest can be reused in subsequent jobs without re-enumeration.

Add `posttrain eval prepare` and `posttrain eval inspect` as proposed public commands, implemented through existing project/catalog/artifact services. Prepare resolves the selected evaluation plan and writes/publishes a manifest through the configured artifact service; inspect is read-only and prints population revision, per-stratum selected/eligible counts, exclusions, planned attempt count, weights, generation controls, and reproducibility support. Finalize exact arguments in CLI help and this plan before implementation acceptance. Ordinary developers must not write a host or import Verifiers classes. Retain existing work-package execution for launching evaluation subjects; do not introduce a parallel scheduler.

Use short configuration defaults: selecting a plan and `num_rollouts: 3` is sufficient for repeated evaluation. Advanced settings are typed opt-ins. Provide examples for a plain task population, balanced math subjects, overlapping skills, and production-weighted tool tasks. Keep preparation separate from inference so users can review the selection before expensive jobs.

### Milestone 6: Observatory and migration

Build one provider-neutral evaluation projection in `apps/observatory/src/posttrain_observatory/evaluation_measurement.py`. It consumes the resolved manifest and normalized attempt facts and returns typed population, coverage, estimator, task, facet, compound, reliability, performance, and comparison results. Extend `models.py` with explicit models such as `EvaluationPopulationSummary`, `EvaluationCoverage`, `EvaluationEstimatorResult`, `EvaluationTaskResult`, `EvaluationFacetResult`, `EvaluationAllocationAudit`, `EvaluationAttemptHistory`, and `EvaluationComparisonView`; finalize names during the baseline/API amendment. Keep `traces.py` responsible for native trace normalization and paging rather than owning the statistical meaning. `service.py`, HTTP, MCP, HTML exports, and the frontend must all call or serialize the same projection.

Evolve the existing run view, trace-evaluation endpoint, and `/api/v1/runs/compare` contract additively. Do not place thousands of task rows inside the main run response. The overview response contains headline summaries, coverage, breakdown summaries, and stable links/cursors. A paginated task-results operation returns sortable task rows; a task-detail operation returns repetition slots and execution attempts; comparison returns compatibility plus summary and paginated delta results. Mirror these operations in MCP and generated OpenAPI/TypeScript types. Keep run locators and provider-neutral source capabilities in every operation.

Update the current evaluation overview and capability area in `frontend/src/App.tsx`, `EvaluationCharts.tsx`, and focused feature components. Present Population and coverage first, followed by separate Tasks, Facets, Combined breakdowns, Reliability, and Performance views. Do not preserve the current fallback in which facets replace task rows. Use stable terms in UI copy: Tasks for task identities, Facets for overlapping semantic labels, Combined breakdowns for declared matrices, Repetitions for planned independent executions, and Attempts for retries. Every headline score names its estimator and target weighting and shows planned, completed, valid, and missing coverage. Facet and matrix cells show their denominators and membership behavior. A prepared run with no traces still shows its intended population.

Implement task detail as a hierarchy rather than a flat trace list: task identity and facets, then planned repetition slots, then execution attempts, then the native trace. Extend `TraceTable.tsx` only for the final evidence drilldown; do not overload every trace row with manifest-level fields. The detail view shows semantic failure separately from execution error, truncation, missing score, and retry. It records the applied generation/initialization controls and makes unsupported reproducibility controls visible.

Extend Compare so a reader sees compatibility before deltas. Match the manifest population/revision, success definition, resolved generation and initialization policy, repetition plan, estimator, and calculator revision. For compatible runs show the paired overall delta, supported interval, task delta distribution, per-task table, facet and combined deltas, regressions/improvements, missing pairs, and evidence links. For partially compatible runs show both full-run results and an explicitly labeled common-task analysis. Strict comparison remains unavailable when the declared comparison contract is violated.

Do not force clients or Observatory to fetch all traces on every request. At completion, persist a content-addressed, versioned evaluation projection or summary artifact whose inputs are manifest digest, evidence revision, measurement policy, and calculator revision. It is an acceleration layer and must be rebuildable from the manifest plus native trace facts. Live views update incrementally from the last projected cursor and remain provisional. A safety-limit truncation, unsupported provider operation, or stale projection sets explicit partial/stale state; it never silently becomes a complete score.

Keep provider access asymptotically independent of task and facet count. A run overview may perform one manifest/artifact lookup, one summary/projection lookup, and bounded provider metadata calls; it must not issue one trace query per task, repetition, or facet. Task and comparison tables page server-side. If the existing Trackio reader cannot query or resume trace facts efficiently, make the smallest generic change in `../trackio`: cursor-stable projection input, grouped trace fields, or immutable summary-artifact access. Keep post-training estimators, task/facet meanings, compatibility checks, and UI in Observatory. Commit and publish Trackio first, update its fork ledger and consumer page, then update the immutable pin and `uv.lock` here. Do not store task IDs as scalar metric labels or introduce a second authoritative aggregate table.

Introduce versioned response fields additively for one release. Legacy evaluations retain their recorded generation settings and rollout-weighted interpretation, visibly labeled legacy. Never infer missing seeds, task populations, repetitions, or target weights for historical runs. Legacy runs without a manifest can use the existing trace-derived task grouping, but Population and coverage must say “unknown” rather than constructing a fictitious eligible population. Deprecate confusing Python names using aliases for one release; remove internal alias usage after all callers migrate, then remove public aliases in the following documented release. Do not reinterpret old `EvaluationSlice` keys as newly defined semantic slice identities.

### Milestone 7: qualification

Use a real pinned Verifiers evaluation with a small no-tool taskset and a tool-using taskset exposing multiple facets, each with three repetitions. Exercise one managed local inference binding and one supported remote OpenAI-compatible test endpoint; the remote endpoint can be a controlled local service for protocol tests, but live model inference remains a separate required gate. Capture actual task IDs, state reset evidence, seed support, resolved request parameters, retry identity, artifacts, and Observatory results. Credentials come from existing environment/service configuration and are never stored in the plan or fixtures. GPU/backend unavailability means qualification remains incomplete, not that fake tests establish support.

After generic qualification, configure matched final-checkpoint evaluations as a separate application of the framework. Confirm held-out task identity exclusion against both training populations before calling the result held-out. Reuse published model-adapter/model-weights artifacts, not training recovery artifacts. This plan does not itself authorize submission now; the user explicitly paused launches to discuss framework design.

## Concrete Steps

Run from `/home/hammad/projects/rl`. Before implementation inspect `git status --short` and `git worktree list`; create an isolated `codex/` branch preserving dirty work. Read the canonical docs and re-resolve pins before editing. Add proposed tests alongside their owning packages: `packages/eval/tests/test_evaluation_selection.py`, `test_evaluation_repetitions.py`, `test_evaluation_manifest.py`; `packages/environment/tests/test_task_inventory.py`; `apps/observatory/tests/test_evaluation_measurement.py`, `test_evaluation_projection.py`, `test_evaluation_comparison.py`, and `test_evaluation_http.py`; CLI/catalog/job tests beside existing tests. Add focused frontend tests for overview/coverage, task repetition detail, facet and combined views, reliability, and matched/mismatched comparison rather than growing only `App.test.tsx`.

Focused commands after the respective files exist:

    uv run pytest packages/environment/tests packages/eval/tests -q
    uv run pytest packages/jobs/tests apps/cli/tests -q
    uv run pytest apps/observatory/tests -q
    uv run lint-imports

Frontend commands from `/home/hammad/projects/rl/apps/observatory/frontend`:

    npm ci
    npm test
    npm run build
    npm run test:e2e

Run the repository validation ladder before completion:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Add a credential/GPU-marked real integration test in `packages/eval/tests/test_evaluation_selection_integration.py` and document its exact invocation, required environment variables, model identity, and backend pin as soon as the integration fixture is implemented. Use `uv run pytest packages/eval/tests/test_evaluation_selection_integration.py -m integration -q` for the gate. Missing prerequisites must skip with an explicit reason and leave the gate incomplete. Record browser screenshots and API/export equality for the completed fixture in the plan.

## Validation and Acceptance

Allocation tests cover no facets, multiple dimensions, overlapping labels, missing fields, filtered-empty populations, zero/negative weights, strata smaller than quotas, infeasible minimums, budgets smaller than class count, and reproducible randomized draws. Repeated-seed tests verify expected inclusion frequencies against an independent finite-population oracle. Verify quotas are integer, sum to the feasible budget, and do not duplicate task identities. Tests must assert behavior, not simply recreate the implementation formula.

Execution tests cover concurrency reorder, state reset, three successful logical repetitions, one transient error followed by success, exhausted retries, no retry on semantic failure, unsupported seed controls, manifest mismatch, duplicate evidence ingestion, and resume. Generation tests verify that effective outgoing parameters match the recorded policy and detect environment/inference conflicts.

Measurement tests cover the exact numerical examples above, all-missing scores, unequal repetition counts, overlapping macro groups, missing positive-weight strata, nonbinary rewards, partial transport pages, and paired comparisons with missing tasks. UI tests verify that 20 tasks with three repetitions display 20 selected tasks and 60 planned repetitions. If one repetition retries once, the same view shows 61 execution attempts without changing either earlier count. An interrupted run cannot show a complete qualification score. Clicking a task exposes its repetition slots, retry chain, and original native trace.

An overlapping-facet fixture proves that one task contributes once to the overall task mean, appears in each declared facet, and was dispatched once per planned repetition. A prepared but unstarted fixture proves that population and allocation audit render with zero observed attempts. A legacy fixture proves that rollout-weighted scoring stays labeled legacy and that unknown population data remains unknown. A comparison fixture proves that identical manifests permit strict paired comparison, while changed generation policy or task population produces the declared incompatibility state and an explicitly labeled common-task view where supported.

Instrument provider calls in Observatory tests. Overview and finalized comparison call counts remain constant as task count increases; task and delta tables use cursors. A 5,000-trace cold projection and warm finalized view receive recorded local benchmark results before release. Establish the release target from that baseline in this plan rather than inventing a hardware-independent latency promise. Any response stopped by the existing 5,000-trace safety limit is partial and cannot populate a finalized score. API, MCP, HTML views, CSV/JSON exports, and the browser must agree on calculator revision, estimator, weights, denominators, and compatibility state.

Acceptance requires a portable project using standard jobs to prepare one selection and evaluate two subjects without custom host code. Reusing the manifest produces exactly matching task plans; all discrepancies in completed evidence remain visible. Full evaluation and uniform-subset examples must work without class metadata.

## Idempotence and Recovery

Manifest preparation is content-addressed and idempotent. Changing inventory, filters, allocation, or relevant policy creates a new digest; never overwrite a resolved selection. Logical repetition keys and execution-attempt keys prevent ingestion duplication. Resume may reuse complete results only under the identical model artifact, manifest, generation, initialization, and measurement identities. It schedules only unfinished repetitions and preserves failed attempts. Native environments that cannot resume safely restart the run under a new identity with the old run marked partial.

Roll out the new schema as opt-in, migrate first-party examples, and retain legacy report interpretation during the compatibility window. Rollback selects legacy jobs; it must not rewrite old evidence or reinterpret new evidence through a legacy estimator. Leave unrelated dirty files and live services intact.

## Artifacts and Notes

The primary artifact is the immutable evaluation manifest and consumed/produced edges in existing tracking. Native Verifiers episodes remain replay authority. Raw trace metadata adds manifest ID, source task key, repetition index, execution attempt index, supported/applied seeds, initialization identity, and policy digest. The optional finalized Observatory projection records manifest digest, evidence revision, measurement policy, calculator revision, projection cursor, and its own content digest. It contains derived values only and can be discarded and rebuilt. High-cardinality values stay in artifacts and traces.

Qualification evidence must retain exact command, dependency pins, model refs, manifest digest, expected/completed counts, calculation version, and UI/API examples. Historical OLMo training comparisons are motivation, not proof of evaluation correctness.

## Interfaces and Dependencies

Proposed public interfaces are `TaskDescriptor`/`TaskFacetValue` in environment, and `EvaluationSelectionPolicy`, `ResolvedEvaluationManifest`, `EvaluationMeasurementPolicy`, and optional `EvaluationReproducibility` in eval. Existing `EvaluationPlan`, `EvaluationBudget`, and `EvaluateRequest` compose them. Proposed core functions are `resolve_evaluation_selection(inventory, policy)` and Observatory `measure_evaluation(manifest, attempts, measurement_policy)`. Observatory also needs a projection builder keyed by manifest/evidence/calculator revisions and paginated task/detail/comparison query methods; these remain app-owned interfaces rather than reusable framework primitives. Types serialize through existing catalog and artifact machinery; exact identifiers are finalized consistently in the baseline amendment, schemas, CLI, tests, and this plan.

Do not import train or serve into eval; managed serving remains standard-job composition. No reusable package imports Observatory or lab. If Verifiers changes are required, the order is fork tests, fork ledger update, commit/push, consumer `docs/tooling/verifiers/README.md` update, immutable pin and `uv.lock` update, affected runtime-image locks, then real consumer qualification. Use `docs/tooling/forks.md` and the fork's `CARBONTEQ_FORK.md`; record old/new SHAs and exact fork regression commands here. Do not silently adopt sibling HEAD or mix uncommitted changes across repositories.

Revision note: 2026-09-13, revision 1 created to implement reusable evaluation selection, repetitions, generation policy, explicit aggregation, and coverage-aware Observatory reporting. Current behavior and proposed changes are distinguished; implementation and live evaluation remain pending.

Revision note: 2026-09-13, revision 2 expanded Observatory from a reporting checklist into a product, projection, comparison, drilldown, query-efficiency, and migration design. It preserves native traces and the manifest as authority and makes finalized projections rebuildable.
