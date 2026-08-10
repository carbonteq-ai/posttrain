# Qualify versioned evaluation evidence and finish Observatory Overview/Compare

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current
as implementation proceeds. It follows `docs/templates/PLAN.md`; the repository
does not contain the `.agents/PLAN.md` file expected by the workflow skill.

## Purpose / Big Picture

After this work, an evaluation run will carry its own versioned description of
what was evaluated, how reward and pass rate are interpreted, which task fields
form readable categories, and which immutable population it may be compared
against. Observatory will read that persisted contract instead of guessing from
reward values or consulting whatever catalog happens to be installed later.
The Overview will lead with evaluation quality, coverage, pass rate when one is
declared, task facets, latency, token usage, and the actual runtime conditions.
Compare will offer only runs with the same evaluation population and metric
meaning, while allowing model and inference settings to differ visibly.

The initial release proof consisted of two new 4B evaluations: IFEval and
Reasoning Gym. On 2026-08-05 the qualification scope expanded by one selected
environment, Math Python, so the same evidence model is exercised on a
tool-using symbolic-math population rather than only instruction-following and
procedural-reasoning populations. All three use the same Qwen3.5-4B BF16
thinking runtime with native MTP enabled and evaluator concurrency eight. There
is no concurrency or MTP experiment matrix. The Math Python run declares a
32,768-token evaluation budget and a 32,768-token per-turn generation ceiling,
while vLLM reserves a 65,536-token engine context so a tool continuation can
carry its accumulated prompt without reducing the declared output budget.
IFEval and Reasoning Gym retain their already-qualified 32K engine profiles.

This plan now makes one narrow amendment to the frozen product baseline: every
`eval.general` and `eval.domain` run must snapshot an explicit typed success
definition for the selected environment. Resolved inputs remain an immutable
run snapshot, native Verifiers traces remain the evidence authority, completed
rows are streamed while work is active, and Observatory remains a read-only
computed view. Training rollouts keep their own reward semantics and are not
subject to this evaluation-only requirement.

## Progress

- [x] (2026-08-03) Ran real Qwen3.5-4B thinking evaluations for IFEval and
  Reasoning Gym and used their native Verifiers traces to redesign the
  evaluation-first Overview and the initial Compare flow.
- [x] (2026-08-03) Added native metric projection, readable task facets,
  pass/review outcomes, latency, input/completion tokens, response/thinking
  lengths, tool/model call counts, and overlap-safe scatter controls.
- [x] (2026-08-04) Moved evaluation meaning into the resolved environment
  declaration and added the versioned
  `posttrain.eval.verifiers-observation` run envelope plus an Observatory reader
  registry with versioned, legacy, and unsupported states.
- [x] (2026-08-04) Fixed the `ResolvedSeat.value` contract-materialization bug,
  rebuilt the IFEval and Reasoning Gym job images, and passed focused contract,
  work, catalog, Observatory, and purge tests.
- [x] (2026-08-04) Purged five superseded Qwen4B runs across provider, OCI,
  Trackio, and local workspaces with digest-bound plans; moved their project
  execution receipts into a recoverable holding area and fixed the local purge
  root to include configured provider storage.
- [x] (2026-08-04) Completed diagnostic run
  `observatory-qwen4b-ifeval-contract-v2-20260804`. It persisted contract
  version 1, produced 16 of 16 traces, had zero failures and zero truncations,
  and Observatory reported strict prompt pass rate/mean reward 0.9375. It used
  concurrency one without MTP, so it is diagnostic evidence rather than the
  final release baseline.
- [x] (2026-08-04) Made completed Verifiers traces visible progressively during
  a running job by changing the synchronizer default batch size to one and
  adding regression coverage.
- [x] (2026-08-04) Added one immutable Qwen3.5-4B evaluation runtime selection with MTP-1,
  `max_num_seqs: 8`, evaluator `max_concurrent: 8`, `max_model_len: 32768`, and
  native `max_total_tokens: 32768`; use it for both final environments.
- [x] (2026-08-04) Replaced the unqualified capacity-probe and separate standard/MTP package
  variants with two canonical release work packages, one for IFEval and one for
  Reasoning Gym, without repointing IDs used by historical runs.
- [x] (2026-08-04) Made the versioned persisted contract authoritative in Observatory and
  restrict environment-binding heuristics to explicitly labeled legacy runs.
- [x] (2026-08-04) Finalized comparison eligibility, including contract version,
  dataset/task revision, split/subset, seed, metric schema, pass
  predicate, and aggregation; show model, MTP, and concurrency as differences,
  not compatibility requirements.
- [x] (2026-08-04) Published the private-OCI eval kind image with the Verifiers
  null-harness environment persisted in an image layer and repacked both jobs
  against its immutable digest. The runtime adapter now resolves that baked
  interpreter and fails closed instead of installing packages.
- [x] (2026-08-04) Completed the full IFEval release job through `posttrain`:
  16 tasks, 16 traces, 0 failures, 0 truncations, and Observatory strict
  prompt accuracy/pass rate 0.875. Container process inspection showed workers
  executing from `/root/.cache/uv/environments-v2/.../bin/python3` with no
  runtime `pip install` or `uv sync`. Run ID:
  `observatory-qwen4b-ifeval-mtp-c8-32k-preinstalled-20260804`; provider run
  `pt-ec08de36487bf2c1ec1be6c5`; job image
  `registry.lan/carbonteq/posttrain-job@sha256:dea8bdf9c0b8e9a72afe47145a8cb96c60d0c50e6bad7ecfcb59fd6a171708da`.
- [x] (2026-08-04) Completed the Reasoning Gym release job through `posttrain`
  with the private OCI registry: 10 tasks, 9 scored, 1 task error, and 1
  truncated trace. Observatory reported mean native reward 0.556213; it did
  not invent a pass rate because this contract exposes continuous native
  reward only. Container inspection showed all workers using the baked
  content-addressed Verifiers interpreter, with no runtime package install.
  Run ID: `observatory-qwen4b-reasoning-gym-mtp-c8-32k-preinstalled-20260804`;
  provider run `pt-bc55a010eadcf927c10b5744`; job image
  `registry.lan/carbonteq/posttrain-job@sha256:e34bfc69b763e546866c1da2fec0cbee3474ed0d9f4968aec87f11aafdfec892`.
- [x] (2026-08-04) Verified the live Overview payloads for both final runs:
  rollout counts, failures, truncations, and trace synchronization are
  populated from Trackio. The comparison endpoint correctly rejects IFEval
  versus Reasoning Gym as different populations while their job kind matches.
- [x] (2026-08-04) Purged eight superseded posttrain-lab evaluation runs
  (six environment-library jobs, the failed MTP bootstrap run, and the
  diagnostic contract run) with digest-bound cross-plane plans. The default
  `posttrain run list` now shows only the two final release runs; `--include-
  purged` exposes the labeled retained admission history. Foreign
  foundation-models and ambient-agent ledger entries were not touched.
- [x] (2026-08-04) Passed focused eval, runtime-image, work, Observatory,
  catalog, Ruff, Pyright, import-boundary, and `git diff --check` validation.
- [x] (2026-08-04) Full repository validation passed: Ruff, Pyright, import
  boundaries, `git diff --check`, and 1,015 tests (18 expected skips).
- [x] (2026-08-04) Recorded the final run lineages, immutable image and
  evaluation/serving-artifact digests, metric semantics, and eight purge plan
  IDs in `release-evidence/cross-plane-purge/evaluation-release-20260804.json`
  and this plan.
- [x] (2026-08-04) Made the shared trace surface job-aware across evaluation,
  GRPO/SAMPO/distillation rollouts, and generic request traces. Binary evals
  lead with configured pass rate; reward-only populations lead with mean
  reward; generic request traces omit reward UI. The table now leads with task
  or request identity, exposes declared reward signals, and keeps trace IDs as
  secondary evidence. Reward components precede a chat-style transcript in the
  inspector. Serving trace reads now use the trace type declared by the job
  telemetry definition rather than assuming Verifiers.
- [x] (2026-08-04) Replaced the interim environment-level `pass_rate_metric`
  selection with a typed, run-snapshotted evaluation success predicate and
  retained the version-1 field only as a historical compatibility reader.
  The new Reasoning Gym run resolved `metric.native_score >= 0.99`, projected
  5 pass, 2 fail, and 3 truncated traces, and correctly reported 71.4% over
  the seven completed semantic outcomes rather than counting truncations as
  failures. Run ID: `observatory-qwen4b-reasoning-gym-success-v2-20260804`.
- [x] (2026-08-04) Retained `rewards` and `metrics` as separate namespaces on
  every trace summary. The shared table now renders declared components and
  verifier metrics without fetching each trace detail record; the aggregate
  reward remains a separate sortable column.
- [x] (2026-08-04) Replaced the split Verifiers PEP-723/MCP runtime with one
  locked job interpreter at `/opt/posttrain/venv/bin/python`; focused runtime
  and image tests verify that execution performs no package resolution or
  installation.
- [x] (2026-08-04) Added a portable tool-using environment compatibility
  contract. AutomationBench requires `tool-calling`; Qwen3.5 inference bindings
  advertise it; detached planning rejects a mismatch; and the vLLM adapter
  derives and emits the Qwen3.5 `qwen3_xml` tool parser from the model renderer
  contract.
- [x] (2026-08-04) Repacked and ran the first parser-qualified AutomationBench
  Qwen3.5-4B image (`sha256:2745b1fec6fb...`). Run
  `observatory-qwen4b-automationbench-success-v2d-20260804` succeeded as a job
  and persisted two traces with native component/success metrics, but both
  task rollouts were truncated after calling concrete tool names against the
  selected meta-tool surface. This is diagnostic evidence, not qualification.
- [x] (2026-08-04) Repacked AutomationBench with the task-scoped
  `limited_zapier` surface as package `45c547e62973...` and private-OCI image
  `sha256:5f9527356e4b...`. Run
  `observatory-qwen4b-automationbench-success-v2e-20260804` completed 2/2
  traces with reward 1.0 and pass rate 100%, zero truncations/errors, 3--5
  successful concrete tool calls per trace, native component/success metrics,
  and complete token, latency, response, and thinking evidence in Observatory.
- [x] (2026-08-04) Ran the release-quality AutomationBench population over all
  200 Simple tasks with evaluation and vLLM concurrency 16. Run
  `observatory-qwen4b-automationbench-simple-full-c16-20260804` used package
  `c9dd29b27597...` and private-OCI image `sha256:b21680ffb782...`; it finished
  with 200/200 included and scored, 184 pass, 4 fail, 12 truncated, mean reward
  0.9275, and configured pass rate 97.87% over the 188 completed semantic
  outcomes. The earlier count of 201 included the `get_simple_dataset()`
  aggregator itself; the pinned aggregator explicitly returns 200 tasks and
  the run covered indices 0--199. The two-task run remains architecture
  qualification, and the superseded c8/n16 submission was cancelled before
  task execution.
- [x] (2026-08-04) Completed the release validation ladder after the full run:
  Ruff, Pyright, all 1,026 Python tests (18 skipped), all eight import-boundary
  contracts, `git diff --check`, 26 Observatory frontend tests, and the
  production frontend build passed. The first full-suite pass identified eight
  legacy evaluation plans/fixtures without explicit success definitions; those
  declarations were added, along with a catalog regression requiring success
  coverage for every environment in every evaluation plan.
- [x] (2026-08-04) Added schema-backed evaluation performance distributions
  computed from the same retained trace population as the verdict. Overview
  now shows p50/p95 end-to-end latency, generated tokens, thinking tokens when
  present, and tool calls when applicable. The live 200-task AutomationBench
  run reports p50/p95 latency 14.5s/232.5s, generated tokens 671/6,199, and
  tool calls 2/14. A single full-population facet is now presented as the
  selected benchmark split rather than as a misleading capability breakdown.
- [ ] (2026-08-05) Run the user-approved Math Python release evaluation over
  all 500 balanced test tasks with the pinned environment and dataset
  revisions, subprocess Python tool execution, MTP-1, evaluator concurrency
  eight, an eight-turn tool budget, and the 32K evaluation budget. Diagnostic
  attempts exposed environment-owned MCP entrypoint, state-schema,
  answer-format, and unbounded-reasoning defects, plus an upstream Verifiers
  fixed-per-turn output reservation that can overrun a growing multi-turn
  prompt. The environment fixes are published at revision
  `ee096746ec3cf28eceffd49f29226e8a8dc7bc31`. Package
  `d79782ef7555...` and private OCI image `sha256:242ca3b9d704...`
  are running the final full population as
  `observatory-qwen4b-math-python-mtp-c8-budget32k-context64k-final-20260805`.
  The release profile uses a 32K per-turn ceiling and 64K engine context
  headroom so tool continuations fit. Model repetition that consumes all eight
  turns or reaches the output ceiling remains a real truncation outcome rather
  than an infrastructure error.
- [ ] Complete the explicit publication review: inspect the intended commit
  split, confirm the private-OCI references and release wording, then obtain
  approval before publishing or tagging.
- [x] (2026-08-05) Added evaluation-contract schema v3 for run-owned compound
  reporting breakdowns. Environment bindings still declare independently
  filterable native facets; evaluation plans now select structured dimension
  combinations. Math Python revision 2 declares `problem_type × difficulty`.
  Observatory returns coverage-aware matrix groups with reward, configured
  pass rate, scored/observed denominator, errors, and truncations, while v1/v2
  runs retain their original one-dimensional interpretation.
- [x] (2026-08-05) Re-ran the release validation ladder after the schema-v3,
  trace-presentation, and compound-breakdown changes: Ruff lint and format,
  Pyright, all eight import contracts, 1,030 Python tests (18 skipped), 32
  frontend tests, the production frontend build, and `git diff --check` passed.
  Moved the focused evaluation visual-QA report and its three comparison
  captures into their owned frontend and `docs/design/observatory/audit`
  locations instead of leaving release evidence at repository root.
- [x] (2026-08-05) Prepared the coordinated 0.3.1 source metadata, staged and
  built all 24 distributions with 107 exact internal pins, and published the
  six 0.3.1 kind images directly to `registry.lan/carbonteq`. The publisher
  reused the immutable base image, read every resulting digest back from the
  registry, and regenerated `published.toml`; GHCR was not used.
- [x] (2026-08-05) Packed schema-v3 package
  `2f3e017b8145e4a87af79c5a68c957b145c9a165b37ba92cbc5dfcf3098f6ae6`
  as private OCI image
  `registry.lan/carbonteq/posttrain-job@sha256:b49d051ccf2f9928d20054301562ec4639f4caf91d06abcad11778f7bc672dce`
  and submitted run
  `observatory-qwen4b-math-python-mtp-c8-schema-v3-20260805` (provider
  `pt-04267749121271639bd93957`). Live qualification has confirmed the
  versioned contract, configured pass predicate, compound matrix, frozen
  500-task denominator, subprocess Python tools, MTP-1, and eight concurrent
  requests. The user accepted this live evidence as the 0.3.1 qualification on
  2026-08-05 and explicitly required the active run to remain untouched. Its
  eventual terminal reconciliation is an operational follow-up, not a release
  blocker; no 500/500 completion claim is made here.

## Surprises & Discoveries

- Observation: flattening Math Python's `difficulty` and `problem_type` facets
  into one list hid their interaction and repeated internal keys in the UI.
  Concatenating values would improve the label but destroy independent
  filtering and create delimiter/cardinality ambiguity. Schema v3 therefore
  snapshots a declared pair of dimension ids and keeps every group value
  structured; the UI alone renders labels such as `Algebra · Level 4`.

- Observation: using Trackio's live `trace_count` as the evaluation denominator
  produced an impossible transient such as `10/9 traces observed` because run
  metadata and paged traces are not one atomic read. Versioned evaluations now
  take their expected count from the frozen `num_tasks × num_rollouts`
  population and use provider trace count only as a legacy fallback. The live
  schema-v3 run therefore reports `N/500` and remains partial until the declared
  population is present.

- Observation: reward coverage and pass-rate coverage can diverge. In the live
  Math Python run, output-boundary traces retained a primary reward but did not
  carry a valid configured success signal, so `reward scored` exceeded the
  pass-rate denominator. `TraceEvaluationView` now reports `passed` and
  `pass_scored` independently, and both Overview and Traces show the configured
  pass rate with its own numerator and denominator.

- Observation: the existing unit tests exercised Math Python's child evaluator
  but never launched the toolset through Verifiers' real `python -m` MCP
  lifecycle. The first live run therefore exposed a missing module entrypoint.
  After that was fixed, the next live run exposed a base-task-state versus
  `PythonState` mismatch and a missing system instruction for the strict boxed
  answer contract. The environment package now tests server port reporting,
  task/tool state compatibility, and the boxed-answer instruction; framework
  consumers are pinned to published commit
  `ee096746ec3cf28eceffd49f29226e8a8dc7bc31`.
- Observation: Verifiers v1 applies one fixed `sampling.max_tokens` value to
  every turn and treats token limits as soft between turns. With a 24K output
  reservation, a valid 10K first response plus tool result caused vLLM to
  reject the continuation because the growing prompt and unchanged 24K
  reservation exceeded the 32K model context. The release profile therefore
  keeps the user-approved 32K evaluation and per-turn output budget but raises
  vLLM engine capacity to 64K. A future shared Verifiers change should make the
  per-turn reservation context-aware and enforce a hard cumulative token
  budget; Observatory must continue to distinguish context exhaustion from
  provider errors.
- Observation: Math Python uses taskset keys `repository` and `revision`, while
  other environments use `dataset_repo` and `dataset_revision`. The versioned
  evaluation population now snapshots a normalized dataset identity, and the
  Observatory reader retains aliases for already-persisted schema-v2 runs.

- Observation: AutomationBench's current run contract declares `domain` as its
  only facet and selects only the `simple` domain, so every one of the 200
  traces necessarily aggregates into `Domain: Simple`. This is a benchmark
  split, not a useful task-slice or capability taxonomy. Observatory now labels
  that state honestly and does not synthesize application categories from task
  names. A future immutable environment-package revision must emit explicit
  application and workflow facets, after which a new run can populate the
  capability breakdown without framework-specific parsing.

- Observation: the shared trace endpoint always queried `verifiers`, even when
  a job telemetry definition declared `inference` traces (for example,
  `serve.benchmark`). A generic frontend alone could not make the view reusable
  because the service returned the wrong population. The service now resolves
  the trace type from the selected job definition, with focused regression
  coverage.
  Evidence: `JobTelemetryDefinition.trace_sections`, the serving fixture, and
  `test_trace_population_uses_the_job_telemetry_trace_type`.

- Observation: IFEval does not model its numeric task key as a semantic
  category. Each task carries one or more instruction IDs such as
  `detectable_format:json_format`; the prefix is the useful multi-label facet.
  Reasoning Gym instead exposes `generator`. Therefore category fields belong
  in the environment binding and cannot be hard-coded in Observatory.
  Evidence: the pinned environment sources and the live projected traces.
- Observation: reward is not universally pass/fail. IFEval's
  `strict_prompt_accuracy` is a configured binary pass signal, while Reasoning
  Gym's `native_reward` is continuous and must remain a score unless its run
  contract explicitly declares a success predicate. Reward components must be
  retained separately from the aggregate reward.
  Evidence: live Verifiers traces and
  `docs/architecture/proposed-evaluation-signal-interpretation.md`.
- Observation: the trace detail projection retains native `rewards`, but
  `TraceSummary` retains only a merged metric map. The population table
  therefore cannot reliably distinguish or render reward components. The
  summary contract must carry reward and metric namespaces separately while
  preserving the merged field for compatibility.
- Observation: the first schema-v2 live projection let truncated traces fall
  through to the version-1 binary-metric adapter. Two truncated Reasoning Gym
  traces therefore entered the pass-rate denominator even though their table
  outcome was `Truncated`.
  Correction: a typed schema-v2 definition is now authoritative even when the
  trace is operationally incomplete. Error and truncation rows retain a null
  semantic outcome and never fall through to legacy inference; focused tests
  cover this boundary.
- Observation: the first AutomationBench component/success qualification
  failed before any model call. The prebuilt PEP 723 harness interpreter was
  placed first on `PATH`, so its child `python -m automationbench_v1.tools`
  process could not import the environment package already installed in the
  job virtualenv.
  A PATH-only correction let the tool server start but exposed the deeper
  defect: the unpinned PEP 723 environment had resolved `mcp==2.0.0`, while
  the hash-locked job and tool server used `mcp==1.28.1`. Non-tool evaluations
  could not reveal this split because they never initialized an MCP session.
  Correction: packed evaluations now have one dependency graph and one
  interpreter. Harness clients, MCP tool servers, environment packages, and
  the framework all execute from `/opt/posttrain/venv`; runtime preparation
  only writes the selected script bytes and never resolves or installs a
  second environment.
- Observation: after the unified runtime allowed AutomationBench to initialize
  MCP and reach its first model call, vLLM rejected `tool_choice="auto"` because
  the selected Qwen3.5 inference binding did not enable automatic tool choice
  or a tool-call parser. This proved that environment transport compatibility
  and inference tool capability are separate contracts.
  Correction: the environment now declares portable `tool-calling` as a
  requirement, the complete inference binding declares the capability, and the
  vLLM adapter derives Qwen3.5's `qwen3_xml` parser from the selected model's
  renderer contract. MCP remains private to the packed environment runtime.
- Observation: the first parser-qualified run proved the renderer and parser
  path but not task semantics. Qwen3.5-4B used `search_tools`, then called the
  concrete Gmail and Salesforce names returned by discovery. The selected
  `zapier` server exposed only `search_tools` and `execute_tool`, so all direct
  calls returned unknown-tool errors and both traces truncated after 17 calls.
  Correction: the qualification environment selects AutomationBench's existing
  `limited_zapier` mode, which exposes only each task's declared concrete tools.
  Model syntax remains catalog-owned; tool-surface semantics remain
  environment-owned.
- Observation: counting every `get_simple_*` function in the pinned source
  produced 201 because that pattern also matches the `get_simple_dataset()`
  aggregator. The aggregator's own contract and the loaded taskset contain 200
  tasks. The full run covered task indices 0--199; the release catalog now
  declares 200 so selection intent matches the actual immutable population.
- Observation: the full AutomationBench run separated reward from configured
  success in real evidence. Mean reward was 0.9275; 184 traces passed, 4 failed,
  and 12 truncated traces retained null semantic success. Observatory reported
  97.87% over the 188 completed semantic outcomes instead of treating partial
  credit or truncation as a binary result.
- Observation: the full run persisted `partial_credit` plus the four native
  assertion/success metrics for all 200 traces, but all projected
  `thinking_tokens` values are null even though Qwen reasoning parsing was
  enabled. Aggregate completion/input tokens, latency, tool calls, and model
  calls are present. Separate thinking evidence therefore remains a projection
  gap and must not be shown as zero in the UI.
- Observation: BuildKit v0.31.2 reused a stale local named-context source across
  content-addressed package directories. A package-keyed COPY destination alone
  did not prevent it. The actual-job publisher now reads the smoke graph
  uncached, verifies the canonical manifest key before any other context input,
  then reuses those verified layers for the publication build. Package
  `c9dd29b27597...` qualified and published only after this guard passed.
- Observation: the stale named-context source recurred while changing only the
  Math Python inference budget: BuildKit supplied package `20e15c8...` while
  the build requested `d79782e...`, and the manifest guard stopped publication.
  Removing only the 12.9 MB `source.local` cache allowed the verified package
  to build as image `sha256:242ca3b9d704...`. The existing uncached smoke pass
  is therefore a safety barrier, not yet a complete cache invalidation fix.
- Observation: the first versioned run lacked its envelope because work
  resolution inspected a `ResolvedSeat` wrapper rather than its `.value`.
  Direct resolved-plan inspection caught the bug before the run was accepted.
  Evidence: the regression in `packages/work/tests` and the corrected live run.
- Observation: the completed diagnostic IFEval run is clean at a 16,384-token
  generation ceiling: 16/16 included, 0 failures, 0 truncations, and 0.9375
  strict prompt accuracy. Earlier 4K and 8K runs had genuine
  `finish_reason=length` stops, so UI changes could not repair them.
  Evidence: `/api/v1/runs/{run_key}/view` for
  `observatory-qwen4b-ifeval-contract-v2-20260804`.
- Observation: while the diagnostic run was active, Observatory showed zero
  scanned traces; all 16 appeared at finalization. The adapter already polls
  `traces.jsonl` every 100 ms, but `VerifiersTraceSynchronizer` defaults to a
  batch size of 16. A 16-task run therefore emits nothing until the batch fills
  or `finalize()` flushes it.
  Evidence: `packages/eval/src/posttrain/eval/backends/verifiers/adapter.py`
  and `synchronization.py`.
- Observation: vLLM previously reported roughly 316,757 cached tokens on the
  local 24-GiB target, or about 9.67 theoretical 32K sequences. This is
  directional capacity evidence, not a new experiment. The user selected
  concurrency eight with MTP as the release profile; the two final runs are the
  qualification gate.
- Observation: the first attempted MTP/concurrency-eight IFEval run exposed a
  real bootstrap defect: Verifiers' upstream null runtime ran `pip install -U
  uv` for every worker and all eight initial rollouts timed out during harness
  setup. The run was cancelled and is not release evidence.
  Correction: the Posttrain adapter patches the pinned Verifiers runtime only
  when `POSTTRAIN_VERIFIERS_PREINSTALLED=1`. It materializes the harness script,
  verifies the locked harness dependencies are importable, and executes the
  script with `/opt/posttrain/venv/bin/python`; it never invokes `pip`, `uv
  sync`, `uv python find`, or a package index during a run.
- Observation: an audit of the base, job-kind, actual-job, serving, training,
  and runtime launch paths found no equivalent runtime package installation in
  non-eval jobs. Their `uv pip install`, `uv sync`, and `apt-get` commands are
  Docker build steps; the stable worker entrypoint only executes the packaged
  manifest. The stale standalone eval Dockerfile was aligned to the private
  prebuilt eval parent and covered by a regression test.
- Observation: the final Reasoning Gym run completed with one native task
  error and one generation truncation at the selected 16,384-token output
  ceiling. This is correctly represented as incomplete scoring (9 of 10), not
  converted into a pass rate or hidden as a harness failure. The run still
  proves the build/runtime boundary because all eight workers resolved the
  image-baked environment and the provider exited successfully.
- Observation: cross-plane purge removed provider, OCI, Trackio, and global
  workspaces, but terminal machine admission records intentionally remained.
  As a result, `posttrain run list` could show purged runs as if they were
  live, and the machine ledger also contained foreign-project records. The
  default operational list now scopes entries to the current project and
  suppresses completed runs with purge receipts; `--include-purged` exposes
  labeled retained history without deleting the ledger.

## Decision Log

- Decision: compound evaluation reporting belongs to the evaluation plan, not
  the environment binding or Observatory defaults. The environment declares
  available native dimensions; the plan selects meaningful combinations; the
  resolved run freezes both. Compound values remain structured and default to
  rejecting ambiguous multi-valued combinations.
  Rationale: environments own data semantics, evaluations own the analytical
  question, and historical runs must never change when catalogs or UI code do.
  Date/Author: 2026-08-05, Codex and user.

- Decision: add Math Python as a third release evidence population with a new
  environment ID and dedicated evaluation plan; do not mutate the historical
  two-task `math-python-qualification` binding or the completed IFEval and
  Reasoning Gym plan.
  Rationale: historical runs keep their original meaning while the new run
  explicitly declares 500 balanced tasks, symbolic correctness, problem type,
  difficulty, subprocess tooling, concurrency eight, MTP-1, and 32K context.
  Date/Author: 2026-08-05, Codex and user.

- Decision: treat output-bound repetition as model evidence, while preventing
  avoidable multi-turn context-overreservation with a 32K per-turn ceiling, an
  explicit eight-turn/evaluation budget, and 64K engine context headroom.
  Rationale: a repeated reasoning loop is a legitimate truncated eval outcome;
  a continuation rejected only because the client reuses an oversized output
  reservation is a harness configuration defect. Separating the declared eval
  budget from engine capacity preserves that distinction instead of retrying
  until the score looks clean.
  Date/Author: 2026-08-05, Codex and user.

- Decision: persist evaluation meaning in the run's resolved `evaluation`
  envelope and select its reader by contract ID and schema version.
  Rationale: an old run must not change meaning when a catalog or environment
  package changes. Unknown future versions must be reported as unsupported,
  not guessed.
  Date/Author: 2026-08-04, Codex and user.
- Decision: split responsibility between the environment binding and the eval
  run. The binding declares environment-native fields and labels; the resolved
  run freezes those declarations together with selection, sampling, and
  aggregation.
  Rationale: environment authors know their native trace schema, while only the
  run knows the exact population and policy that were executed.
  Date/Author: 2026-08-04, Codex and user.
- Decision: permit a small safe expression language for configured success and
  derived metrics, but never arbitrary Python and never Observatory-only
  inference such as `reward > 0`.
  Rationale: predicates are useful for evaluation filtering and future GRPO
  analysis, but they must be portable, validated, versioned, and persisted.
  Date/Author: 2026-08-04, Codex and user.
- Decision: make pass rate conditional. If a binary metric or configured
  predicate exists, show configured pass rate; otherwise show score coverage
  and distribution without inventing pass/fail.
  Rationale: continuous reward and composite reward are not categorical
  success signals.
  Date/Author: 2026-08-04, Codex and user.
- Decision: use one shared trace shell with three explicit presentation modes:
  evaluation, optimization rollout, and generic request. Job kind selects the
  vocabulary and defaults, while the persisted trace/evaluation contract
  decides whether pass rate, reward, components, and task facets exist.
  Rationale: visual consistency should not force evaluation semantics onto
  GRPO/SAMPO/distillation or inference traces. Missing reward or success
  evidence is omitted or labeled unavailable rather than rendered as zero.
  Date/Author: 2026-08-04, Codex and user.
- Decision: compare only runs with the same job kind and immutable evaluation
  population/meaning fingerprint. Model and inference settings may differ and
  are shown as comparison dimensions.
  Rationale: comparison is meaningful only when dataset, task selection,
  metric, pass rule, and aggregation match.
  Date/Author: 2026-08-04, Codex and user.
- Decision: use one Qwen3.5-4B release runtime for AutomationBench:
  BF16, thinking enabled, MTP with one speculative token, concurrency sixteen,
  and a 32K total context window. Do not run standard-versus-MTP or concurrency
  sweeps.
  Rationale: the full Simple population can keep sixteen sequences occupied;
  release work should qualify the selected profile directly instead of
  spending time on extra experiments. This supersedes the earlier concurrency
  eight decision for this AutomationBench run.
  Date/Author: 2026-08-04, user.
- Decision: keep the generation ceiling at 16,384 while setting the total
  context ceiling to 32,768.
  Rationale: `max_tokens` counts new output, while the context window counts
  prompt plus output. Setting both to 32,768 would advertise an impossible
  budget for any non-empty prompt.
  Date/Author: 2026-08-04, Codex.
- Decision: do not add a separate live smoke run. Use unit/contract tests,
  catalog validation, resolved-plan inspection, package smoke qualification,
  and the already completed diagnostic run before launching exactly two final
  full jobs.
  Rationale: this preserves a fast development gate without creating extra GPU
  experiments.
  Date/Author: 2026-08-04, Codex.
- Decision: treat package installation as an image-build invariant for every
  job kind, not an eval-only convention. Runtime may materialize immutable
  datasets, but it may not install or upgrade Python packages.
  Rationale: build-time dependency closure gives reproducible startup and makes
  package failures distinct from task/model failures. The audit found the
  existing non-eval job paths already obey this boundary; the Verifiers null
  harness was the exception and now consumes the packed eval lock directly.
  Date/Author: 2026-08-04, Codex and user.
- Decision: make run listing project-scoped and separate operational history
  from purge audit history. The default list includes current-project
  submissions and active admission states; `--include-purged` adds completed
  runs with successful purge receipts and labels them `purged`. Foreign
  project entries in the machine admission ledger remain out of scope.
  Rationale: the admission ledger is machine-wide, while the user operates on
  one project; deleting ledger rows would destroy audit lineage and could hide
  ownership mistakes.
  Date/Author: 2026-08-04, Codex and user.
- Decision: model tool use as an environment-to-inference capability match;
  keep MCP out of the public inference contract.
  Rationale: environments care whether the model can make structured tool
  calls, while MCP is one harness-side transport. The full inference binding,
  not the model alone, must include renderer and backend parser support.
  Date/Author: 2026-08-04, Codex and user.
- Decision: keep model rendering and environment tool-surface selection as
  separate contracts.
  Rationale: the model catalog must resolve Qwen chat-template, thinking, and
  tool-call syntax automatically. The environment must choose whether its task
  exposes concrete tools or discovery/execution meta-tools. Verifiers owns the
  interaction loop across both without becoming a second catalog authority.
  Date/Author: 2026-08-04, Codex and user.

## Outcomes & Retrospective

The evaluation-first UI, native trace projection, versioned contract envelope,
and safe purge path exist in the working tree. The two selected 4B jobs now
run from private-OCI images with dependencies installed at build time: IFEval
is clean at 16/16, while Reasoning Gym correctly exposes 9/10 scored with one
task error and one truncation. The non-eval audit found no matching runtime
installer defect. Eight superseded posttrain-lab evaluation runs are now
purged with retained audit receipts. The subsequent AutomationBench
qualification exposed and corrected the generic packed-MCP and inference-tool
capability boundaries. Its first parser-qualified run then isolated the
environment tool-surface mismatch; the task-scoped rerun proved the full path
with two passing multi-turn traces and native score/success evidence. The full
Simple baseline then completed all 200 tasks at concurrency 16: 184 pass, 4
fail, 12 truncated, no provider failures, 0.9275 mean reward, and 97.87%
configured pass rate over completed semantic outcomes. It also exposed the
remaining thinking-token projection gap and the need to present the long-tail
truncations explicitly. The remaining gate is publication review. Math Python stayed paused until this
cross-layer tool path was proven. Foreign-project admission records remain untouched. The
redacted release record remains at
`release-evidence/cross-plane-purge/evaluation-release-20260804.json`.

The main lesson is that a full 16-task, concurrency-one run is appropriate
release evidence but a poor schema feedback loop: it took about 28 minutes and
showed no partial traces because the synchronization batch equaled the entire
population. Progressive evidence and concurrency eight address that without a
separate experimental matrix.

## Context and Orientation

The repository is a Python `uv` workspace rooted at
`/home/hammad/projects/rl`. `packages/eval` owns the provider-neutral evaluation
API and the private Verifiers adapter. `packages/common` owns small observation
values and must not import Verifiers or Trackio. `apps/lab` is the reference
composition host and owns concrete observer wiring, project catalog overlays,
and work packages. `apps/observatory` is a read-only evidence product. It reads
provider-neutral tracking data and may compute views, but it may not redefine a
run's meaning.

A resolved evaluation contract is the JSON-compatible snapshot stored at
`resolved_inputs["evaluation"]`. Its contract ID is
`posttrain.eval.verifiers-observation` and its current schema version is 1. The
snapshot contains the evaluation plan, environment/task population, signal
manifest, and native-evidence description. `apps/observatory/src/posttrain_observatory/evaluation_contracts.py`
selects the reader for that ID/version. A legacy run has no versioned envelope;
it may use only declarations embedded in that run's historical resolved input,
never today's catalog. An unsupported run has an explicit contract whose ID or
version this Observatory build cannot read; the UI must explain that state and
must not infer metrics.

The authoritative task evidence is the Verifiers JSONL bundle. During a run,
`packages/eval/src/posttrain/eval/backends/verifiers/adapter.py` starts native
`run_eval`, and `VerifiersTraceSynchronizer` in `synchronization.py` tails
completed JSONL lines and emits `TraceObservation` values. The Trackio observer
in `apps/lab/src/posttrain_lab/tracking/trackio_observer.py` converts each value
to a native `trackio.VerifiersTrace`. At present the synchronizer's default
batch size of 16 hides partial progress for a 16-task run.

The model runtime catalog lives in
`packages/catalog/src/posttrain/catalog/base/inference.yaml`. The project
environment/evaluation overlay is
`apps/lab/.posttrain/catalog/algorithm-qualification.yaml`; executable YAML
work packages live under `apps/lab/.posttrain/work_packages/`. The final
runtime must be a new immutable selection rather than a mutation of a selection
already recorded by old runs. Its effective values are:

    model: models/qwen3.5-4b@bf16
    renderer: qwen3.5-tools-thinking@1
    engine.max_model_len: 32768
    engine.max_num_seqs: 8
    engine.max_num_batched_tokens: 8192
    engine.kv_cache_dtype: auto
    engine.speculative_config.method: mtp
    engine.speculative_config.num_speculative_tokens: 1
    sampling.max_tokens: 16384
    environment.max_total_tokens: 32768
    environment.max_concurrent: 8

`max_num_batched_tokens` is a vLLM scheduler work budget for one iteration; it
does not reduce the 32K per-request context window. Chunked prefill remains
enabled so long prompts can be scheduled without monopolizing an iteration.

## Plan of Work

### Milestone 1: make the contract path live and authoritative

Change `packages/eval/src/posttrain/eval/backends/verifiers/adapter.py` to
construct `VerifiersTraceSynchronizer` with a one-record emission batch, or
change the synchronizer's public default to one if all callers and tests agree.
Retain incomplete-line handling, trace-ID deduplication, retry on finalization,
and best-effort observation semantics. Extend
`packages/eval/tests/test_trace_sync.py` and the adapter tests with a case that
writes one complete record while the run is active and proves the observer sees
it before finalization. Do not emit run-level coverage or pass-rate metrics from
partial data as if they were final; Observatory may display a clearly labeled
live partial aggregate over the traces observed so far.

In `apps/observatory/src/posttrain_observatory/evaluation_contracts.py` and
`service.py`, route versioned runs entirely through the selected contract reader.
The reader must yield the primary metric, optional pass signal/predicate,
  component metrics, facet specifications, aggregation, and
comparison inputs. The current environment-observation resolver remains only
for runs classified `legacy`. An explicit but unsupported ID/version returns
the unsupported state with the raw ID/version and no guessed score.

Add regression tests under `apps/observatory/tests` for versioned authority,
legacy embedded fallback, unsupported versions, composite reward components,
continuous reward without pass rate, a configured binary pass metric, and a
safe expression predicate. The expression evaluator must accept only a small
documented grammar over known scalar fields, reject missing/type-incompatible
values as unknown rather than false, and reject calls, attribute access,
imports, assignments, and arbitrary code.

This milestone is complete when a test can add one trace to an active run and
observe `scanned=1`, and when changing the current catalog cannot change a
persisted versioned run's interpretation.

### Milestone 2: define the one final 4B runtime and two release packages

Add a new immutable inference selection in
`packages/catalog/src/posttrain/catalog/base/inference.yaml`, named clearly for
MTP, concurrency eight, and 32K context. Use the exact effective settings shown
in Context and Orientation. Do not repoint the prior `@1` standard, long,
extra-long, MTP, concurrency-two, or concurrency-four selections because old
run snapshots refer to them.

In `apps/lab/.posttrain/catalog/algorithm-qualification.yaml`, add canonical
IFEval and Reasoning Gym release environments. Both use `max_concurrent: 8`,
sampling `max_tokens: 16384`, native total-token limit 32,768, sampling seed 0,
the pinned external environment revision, subprocess harnesses, and bounded
timeouts. IFEval keeps 16 deterministic tasks from the pinned `google/IFEval`
revision and declares `strict_prompt_accuracy` as primary and binary pass
metric plus `instruction_id_list` as a prefix-transformed multi-label facet.
Reasoning Gym keeps one example from each of the 10 named generators and
declares continuous `native_reward` plus `generator` as its facet; it does not
declare pass rate unless the environment contract supplies a true binary
signal.

Add one evaluation-plan selection containing these two environments and create
two work packages under `apps/lab/.posttrain/work_packages/`, one per final
run, both referencing the same new inference binding. Remove the uncommitted
capacity-probe work packages and unused experimental catalog entries only after
`git grep` and resolved-run inspection prove that no accepted run depends on
them; historical immutable run records and receipts are not rewritten.

Validate the catalog, resolve both plans, and inspect their JSON. Acceptance is
not the YAML text alone: both resolved plans must show MTP-1, concurrency eight,
32,768 total context, 16,384 output ceiling, the correct task population, and
contract schema version 1. Pack both jobs to the existing private OCI registry.
Do not configure or publish to GHCR.

### Milestone 3: run exactly two full release evaluations

Launch IFEval first, then Reasoning Gym, using durable run IDs that include the
environment, MTP, concurrency, context, and date. Sequential job submission is
acceptable even though each job has eight concurrent evaluator requests; this
avoids two model servers competing for the same GPU. Do not launch standard,
concurrency-one/two/four, or MTP-off comparison cells.

While IFEval runs, query Observatory after the first completed trace and prove
that the run is `running`, the persisted contract is versioned, and `scanned`
is between 1 and 15 before finalization. At completion require 16/16 traces,
zero missing coverage, zero provider/request failures, zero synchronization
loss, and zero `finish_reason=length` truncations. Do not require a particular
pass rate; record the observed strict/loose metrics without treating quality as
an infrastructure gate.

For Reasoning Gym require 10/10 traces, one for every declared generator, zero
missing coverage, zero provider/request failures, zero synchronization loss,
and zero length truncations. Record continuous reward and its native components
without labeling it pass rate. For both runs capture the provider receipt,
resolved-input digest, OCI image digest, Trackio provider run ID, model and
environment revisions, MTP configuration, concurrency, total context budget,
latency, token usage, and peak GPU/KV evidence when available. Missing
provider-specific reasoning-token counts remain null; character lengths are a
display fallback and are not fabricated token counts.

If either final run fails, diagnose the owning layer and retry that same
selected profile only after fixing the cause. Do not respond by creating a
capacity matrix. A real failure of the selected profile blocks release and is
recorded in Surprises & Discoveries.

### Milestone 4: finish Overview, Compare, and lifecycle presentation

Use the two final runs to verify the frontend in
`apps/observatory/frontend/src/App.tsx` and its supporting components/styles.
The first viewport must show evaluation identity, primary metric, configured
pass rate only when valid, coverage, failures/truncation, and evidence state.
The next hierarchy is component metrics and task facets, followed by latency,
token/response/thinking distributions. Rollout and raw trace navigation remain
drill-down evidence, not headline cards. The lineage sidebar must show model
revision, environment/dataset revision, task population, seed, sampling,
reasoning mode, MTP method/count, concurrency, context/output limits, KV-cache
dtype, engine sequence/batched-token limits, source image digest, and contract
ID/version.

The `Compare this run` CTA starts from a selected run. Candidate filtering must
require the same job kind, contract ID/version, dataset and task
revision, split/subset, deterministic task-selection identity, seed, metric
schema, pass predicate, and aggregation. It may allow different models and
runtime settings; those differences appear as columns and annotations. Since
IFEval and Reasoning Gym are different populations, they must not be offered as
comparison candidates for each other. A final 4B IFEval run may compare with a
future model's IFEval run only when every population/meaning field matches.

Finally, update the default `posttrain run list` behavior in
`apps/cli/src/posttrain_cli/commands/run_cmd.py` so completed audit entries whose
cross-plane evidence was purged do not appear as current operational runs.
Preserve the admission ledger as audit evidence and expose purged history via
an explicit, labeled option such as `--include-purged`; do not delete the ledger
as a side effect of listing. Extend CLI and purge integration tests.

This milestone is complete when the local Observatory shows accurate Overview
and lineage for both final runs, Compare excludes the mismatched environment,
and default run listing no longer presents purged records as live while the
explicit audit view retains their labels.

### Milestone 5: release validation and publication review

Run focused tests after each owning-package edit, then run the full validation
ladder. Update Progress with exact test counts and Artifacts and Notes with the
two final run IDs and immutable digests. Review the complete diff by repository
and separate commits by concern: evaluation contract/streaming, catalog and
work packages, Observatory service/UI, and lifecycle listing. Do not commit,
push, publish images beyond the approved private OCI pack step, update a tag, or
deploy until the user explicitly reviews the evidence and authorizes that
publication action.

## Concrete Steps

All commands run from `/home/hammad/projects/rl` unless a command says
otherwise. Begin by recording the dirty tree and never revert unrelated work:

    git status --short
    git diff --stat

Run focused contract and streaming tests during Milestone 1:

    uv run --package posttrain-eval pytest packages/eval/tests/test_trace_sync.py packages/eval/tests
    uv run --package posttrain-observatory pytest apps/observatory/tests -k 'evaluation or comparison or trace'
    uv run ruff check packages/eval/src packages/eval/tests apps/observatory/src apps/observatory/tests

After Milestone 2, validate and resolve both new work packages. Substitute the
final filenames chosen in the implementation for `<ifeval-release.yaml>` and
`<reasoning-gym-release.yaml>`:

    uv run --package posttrain posttrain --project-root apps/lab catalog validate
    uv run --package posttrain posttrain --project-root apps/lab --json job plan apps/lab/.posttrain/work_packages/<ifeval-release.yaml> --job evaluate > /tmp/posttrain-ifeval-release-plan.json
    uv run --package posttrain posttrain --project-root apps/lab --json job plan apps/lab/.posttrain/work_packages/<reasoning-gym-release.yaml> --job evaluate > /tmp/posttrain-reasoning-gym-release-plan.json

Inspect the resolved values, not only source YAML:

    jq '.resolved_inputs.evaluation, .resolved_inputs.inference, .resolved_inputs.environment' /tmp/posttrain-ifeval-release-plan.json
    jq '.resolved_inputs.evaluation, .resolved_inputs.inference, .resolved_inputs.environment' /tmp/posttrain-reasoning-gym-release-plan.json

Pack each job through the configured private OCI registry. The environments are
currently deferred, so use the explicit waiver and record the returned digest:

    uv run --package posttrain posttrain --project-root apps/lab job pack apps/lab/.posttrain/work_packages/<ifeval-release.yaml> --job evaluate --allow-deferred-qualification --build-missing
    uv run --package posttrain posttrain --project-root apps/lab job pack apps/lab/.posttrain/work_packages/<reasoning-gym-release.yaml> --job evaluate --allow-deferred-qualification --build-missing

Launch the two final runs sequentially. Use fresh date-stamped IDs and wait for
the first to reconcile before starting the second:

    uv run --package posttrain posttrain --project-root apps/lab job run apps/lab/.posttrain/work_packages/<ifeval-release.yaml> --job evaluate --provider local --run-id observatory-qwen4b-ifeval-mtp-c8-32k-20260804 --allow-deferred-qualification
    uv run --package posttrain posttrain --project-root apps/lab run wait observatory-qwen4b-ifeval-mtp-c8-32k-20260804 --timeout-seconds 7200
    uv run --package posttrain posttrain --project-root apps/lab job run apps/lab/.posttrain/work_packages/<reasoning-gym-release.yaml> --job evaluate --provider local --run-id observatory-qwen4b-reasoning-gym-mtp-c8-32k-20260804 --allow-deferred-qualification
    uv run --package posttrain posttrain --project-root apps/lab run wait observatory-qwen4b-reasoning-gym-mtp-c8-32k-20260804 --timeout-seconds 7200

Start Observatory against the real project and inspect the API/browser:

    uv run --package posttrain posttrain --project-root apps/lab observatory up --host 127.0.0.1 --port 7861
    curl -fsS http://127.0.0.1:7861/health/ready
    curl -fsS http://127.0.0.1:7861/api/v1/runs | jq .

Resolve each `run_key` from `/api/v1/runs`, then call
`/api/v1/runs/{run_key}/view`, `/comparison-key`, and
`/traces-evaluation`. Expected final values are `state: complete`, scanned and
included counts equal to the environment population, zero truncations, and a
versioned contract. During IFEval execution, at least one query must show a
nonzero partial scanned count before the run becomes terminal.

Validate the frontend and full workspace:

    npm --prefix apps/observatory/frontend test
    npm --prefix apps/observatory/frontend run check
    npm --prefix apps/observatory/frontend run build
    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

## Validation and Acceptance

The contract is accepted when versioned runs are interpreted only from their
persisted envelope, legacy runs are visibly labeled and use only embedded
historical declarations, unsupported versions display an unsupported state,
and continuous reward never produces an inferred pass rate. Expression tests
must prove safe valid predicates and rejection of arbitrary code.

Progressive evidence is accepted when a real final run displays one or more
scanned traces before terminal finalization. The native JSONL artifact remains
complete and authoritative, trace IDs are emitted exactly once, and an
observation upload failure cannot erase the native bundle.

The runtime is accepted when both resolved plans and both final run lineages
show MTP-1, concurrency eight, a 32,768-token total context limit, a 16,384 new
token ceiling, and the pinned Qwen3.5-4B/environment revisions. IFEval must have
16/16 synchronized, nontruncated traces. Reasoning Gym must have 10/10 across
all named generators. Infrastructure completeness is the gate; metric value is
reported, not used as a release pass threshold.

Overview is accepted when the browser leads with evaluation meaning and
quality, exposes components/facets and latency/token evidence, and keeps full
runtime lineage visible. Compare is accepted when cross-environment runs are
excluded and same-population runs with different models/runtime settings are
allowed and clearly differentiated. No card or chart may fabricate missing
tokens, pass/fail, or reward components.

Lifecycle presentation is accepted when cross-plane purged runs do not appear
in the default operational list, while an explicit audit view can still show
their labeled retained admission history. All focused and full validation
commands must pass before publication review.

## Idempotence and Recovery

Catalog validation, job planning, tests, API reads, and frontend builds are
safe to repeat. Job packing is content-addressed and may be repeated; record and
reuse immutable digests rather than publishing redundant tags. A run ID is a
durable identity. Before retrying a failed run ID, inspect its submission,
provider status, reconciliation, and retained evidence; reuse it only through
the framework's supported retry path. Otherwise create a new ID and retain the
failed attempt as evidence.

The diagnostic and superseded environment-library runs were purged only after
both final runs were accepted and each exact preview named its provider, OCI,
Trackio, and local targets. Applies used the recorded plan digest. If a future
purge partially fails, resume the same plan/receipt; do not generate a broader
replacement. Shared images, environment caches, final 4B evidence, and
unrelated project state remain out of scope.

If concurrency-eight MTP fails, preserve the run and diagnose model load, MTP
initialization, vLLM scheduling, KV exhaustion, request timeout, scoring, trace
sync, and Trackio finalization in that order. The selected release profile is a
gate, so do not silently reduce concurrency or disable MTP and call the result
qualified. Record the blocker in this plan and return to the user with evidence.

## Artifacts and Notes

The two retained release runs are the publication-review evidence. Both use the
resolved Qwen3.5-4B BF16 thinking profile with native MTP-1, vLLM
`max_num_seqs: 8`, evaluator `max_concurrent: 8`, a 32,768-token model/context
budget, and a 16,384-token output ceiling. The provider and retained evidence
reconciliations are `consistent`; each run's evaluation artifact is immutable in
Trackio.

### IFEval release run

    run_id: observatory-qwen4b-ifeval-mtp-c8-32k-preinstalled-20260804
    provider_run_id: pt-ec08de36487bf2c1ec1be6c5
    tracking_provider_run_id: ed6438ddd9c045a8a6347c8c49c12c9c
    status: succeeded
    traces: 16 scanned / 16 included / 16 scored
    failures: 0
    truncations: 0
    strict_prompt_accuracy: 0.875
    evaluation_artifact_digest: e15735fcb042d55a1b2080075fc44ab3d83cd5565bf7f94b0bc65d97d9b84a10
    serving_log_digest: 3a99a1a0838fc932dbaa50f121a26d5c9ec60435148f11dbd2019471835cb668
    job_image: registry.lan/carbonteq/posttrain-job@sha256:dea8bdf9c0b8e9a72afe47145a8cb96c60d0c50e6bad7ecfcb59fd6a171708da

### Reasoning Gym release run

    run_id: observatory-qwen4b-reasoning-gym-mtp-c8-32k-preinstalled-20260804
    provider_run_id: pt-bc55a010eadcf927c10b5744
    tracking_provider_run_id: 48ac08838fe5425c9967842dd4abbe63
    status: succeeded
    traces: 10 scanned / 10 included / 9 scored
    failures: 1
    truncations: 1
    native_reward_mean: 0.5562130177514794
    pass_rate: null (continuous native reward; no configured binary predicate)
    evaluation_artifact_digest: e3cf098c2f0b7cccfdaf93a74c84f907bca9bba0b90322438ea1df7b30c096a6
    serving_log_digest: ff2518bc01f4120dfe4ff966573b1a996168c9fa46a8c6300fccdf20ca380268
    job_image: registry.lan/carbonteq/posttrain-job@sha256:e34bfc69b763e546866c1da2fec0cbee3474ed0d9f4968aec87f11aafdfec892

The final IFEval payload carries dataset revision
`966cd89545d6b6acfd7638bc708b98261ca58e84`, source/environment revision
`017ac72f543f79f48400cbb4cb641d6df4c3adfa`, and model revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Its configured primary signal is
`strict_prompt_accuracy`, so the displayed 0.875 is a declared pass rate. The
Reasoning Gym payload declares `native_reward` as its primary continuous signal
and exposes the ten generator facets; the one failed/truncated task is retained
in the native trace population and is not converted into an inferred pass rate.

The redacted release record is checked in at
`release-evidence/cross-plane-purge/evaluation-release-20260804.json`; the
cleanup evidence is retained under the project purge store and
`release-evidence/cross-plane-purge/`. This turn applied eight exact,
previewed run-purge plans: six superseded environment-library runs plus the
failed MTP bootstrap and contract diagnostic run. The recorded plan IDs are
`purge-5e25203fdb1c9593`, `purge-2e8ac18b2520e7be`,
`purge-45983e584fc6a14c`, `purge-3bc2a5433cbd1d98`,
`purge-a99dfdba23e650c6`, `purge-6e09475fb1c794d3`,
`purge-a9ce2c86fd94cfe9`, and `purge-90d25542c5898bd2`. Their digest-bound
applies are recorded in the corresponding release receipts; shared images,
machine caches, final runs, and foreign `foundation-models`/`ambient-agent`
records were not targeted. Default `posttrain run list` is therefore an
operational view of the two retained runs, while `run list --include-purged`
is the explicit audit view.

### Publication review handoff

The working tree should be reviewed as separate logical commits rather than a
single release blob:

1. **Evaluation runtime and preinstalled Verifiers** — runtime Dockerfiles,
   image publication metadata, the Verifiers runtime adapter, and their tests.
2. **Contract, Observatory, and comparison UI** — the versioned evaluation
   contract reader, service/API models, frontend projections, and Observatory
   tests.
3. **Catalog, work packages, and Qwen4B qualification** — environment/model/
   target catalog entries, the two release work packages, qualification gates,
   and their tests.
4. **Lifecycle and release evidence** — project-scoped `run list`, purge
   presentation tests, this plan, and the redacted evidence record.

Before staging, inspect each group against the canonical baseline and preserve
any unrelated dirty files. Run the package-specific tests for each staged
group, then the full validation ladder already recorded above. Publication is
not implied by this evidence record: no commit, push, image retag, or release
tag has been made.

The clean diagnostic run below remains useful as contract and long-output
evidence, but is not the release baseline:

    run_id: observatory-qwen4b-ifeval-contract-v2-20260804
    status: succeeded
    provider_run_id: a73e7968d8ea49b5994f8dc88a64353f
    traces: 16 scanned / 16 included / 16 scored
    failures: 0
    truncations: 0
    strict_prompt_accuracy: 0.9375
    evaluation_artifact_digest: 54478716447718d16d96b3a5c34caa66fb74122958985314a7be53a931ec0ef0

Its immutable IFEval dataset revision is
`966cd89545d6b6acfd7638bc708b98261ca58e84`; the environment source revision is
`017ac72f543f79f48400cbb4cb641d6df4c3adfa`. The model revision recorded in the
catalog is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. These values are evidence for
the current working tree, not substitutes for recording the final runs' own
resolved snapshots.

Initial design-audit screenshots are outside the repository at
`/home/hammad/.codex/visualizations/2026/08/03/observatory-eval-audit/`.
Final browser evidence should be stored in a new date/run-specific directory so
the old and final states are not conflated.

## Interfaces and Dependencies

`posttrain.eval.backends.verifiers.adapter.run_verifiers` remains the only
Verifiers execution entry used by the reusable eval package. It returns
`VerifiersRunResult`; native records are represented as framework-neutral
`TraceObservation` values before the composition host translates them to
Trackio. `posttrain.common` must not import Verifiers, Trackio, vLLM, train, or
`apps/lab`, and `packages/eval` must not import train or serve. Run
`uv run lint-imports` after boundary changes.

The contract reader registry in
`apps/observatory/src/posttrain_observatory/evaluation_contracts.py` must expose
a deterministic result for `(contract_id, schema_version)`. Its normalized
result carries contract state, primary metric, optional pass definition,
component metrics, facet declarations, aggregation, and
comparison fields. The service and UI consume this normalized result; they do
not parse environment-specific fields directly.

Verifiers and its environment packages remain external immutable Git
dependencies. Environment-native task and reward extraction belongs in those
environment packages or the resolved environment binding, not in Observatory.
Trackio owns generic trace storage/query behavior; post-training-specific
Overview and Compare meaning stays in this repository. The only OCI registry in
scope is the one configured by this project. GHCR configuration is explicitly
out of scope.

Plan revision note (2026-08-04): rewrote the remaining execution path around a
versioned run-owned evaluation contract, progressive trace evidence, and two
full release runs. Removed the proposed MTP/concurrency experiment matrix after
the user selected one operating profile: Qwen3.5-4B BF16 thinking with MTP-1,
concurrency eight, and 32K total context for both IFEval and Reasoning Gym. The
revision also records the completed clean diagnostic IFEval run and explains
why it is not the final release baseline.

Plan revision note (2026-08-04): the user required every evaluation to expose
pass/fail. This revision makes success mandatory at evaluation execution,
keeps the rule in `EvaluationPlan` rather than Observatory or training, adds a
versioned reader-before-writer rollout, and expands live qualification beyond
the two original environments to cover binary and component-reward shapes.
