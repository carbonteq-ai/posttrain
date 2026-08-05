# Deliver constraint-driven serving capacity screening and Observatory views

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not contain `.agents/PLAN.md`. This document must therefore be maintained in accordance with `docs/templates/PLAN.md`, the checked-in ExecPlan authority named when this plan was created.

## Purpose / Big Picture

After this change, a project developer can state the serving behavior the product requires—required context window, minimum sustained aggregate output-token throughput, latency limits, and failure-rate limit—separately from the model, inference backend, workload methodology, and hardware profile. A `serve.benchmark` run will sweep concurrency to measure how completely one model and inference binding use a fixed execution target. It will retain the measured operating curve without deciding whether the model should be selected.

The decision-grade capacity workload will use a versioned, provenance-aware representative prompt corpus rather than repeated synthetic seed text. The first framework corpus combines deterministically selected GSM8K reasoning questions, HumanEval code prompts, and reviewed first-party chat, extraction, structured-output, and tool-use prompts. The benchmark renders those messages through each model's declared renderer, fixes the output-token budget, and measures systems behavior without scoring task correctness. Exact-token synthetic workloads remain available as a separate diagnostic cohort and are never merged into the product-facing capacity result.

Observatory will turn that evidence into two connected product views. The run view will explain one model/backend/hardware capacity experiment, show the backend-specific runtime settings, and report the highest-throughput operating point that satisfies the project requirements. The screen work-package view will compare all contenders on the same hardware and requirements, expose capacity and constraint failures, and show a Pareto set without selecting a universal winner.

The first demonstrable scenario uses a project requirement of at least 50 sustained aggregate output tokens per second. A model producing 58 tokens per second while meeting context, latency, and reliability limits is eligible. A model producing 71 raw tokens per second while violating the latency limit is not. A model unable to load the required context is a context failure, and a model saturating at 46 tokens per second is below capacity. All four results remain visible.

## Progress

- [x] (2026-07-24 14:45Z) Read `docs/templates/PLAN.md`, the repository guide, all canonical post-training documents, and the current serving, work-package, project-layout, tracking, Observatory service, HTTP, MCP, frontend, and test surfaces relevant to the change.
- [x] (2026-07-24 14:45Z) Confirmed the product boundary: requirements are project policy; concurrency and backend settings are search variables; `serve.benchmark` records evidence; Observatory computes eligibility and Pareto membership.
- [x] (2026-07-24 14:45Z) Created this implementation plan without modifying the existing dirty `apps/observatory/obseratory.pen` design file.
- [x] (2026-07-24 15:02Z) Refined the benchmark-population design after confirming that the shipped four-record representative corpus is not used by the offline benchmark; selected a 128-record mixed representative corpus for capacity and retained exact-token prompts as a separate diagnostic workload.
- [x] (2026-07-24 15:15Z) Created branch `codex/serving-capacity-observatory` from `main` while preserving the unrelated dirty Observatory design file.
- [x] (2026-07-24 15:15Z) Amended the frozen canonical baseline narrowly to formalize project serving requirements, representative and controlled saturation workloads, point-level capacity evidence, and Observatory-owned interpretation.
- [x] (2026-07-24 15:20Z) Added a strict portable project brief, stable digest and run snapshot, schema-version-1 compatibility, schema-version-2 discovery, CLI/scaffold integration, root project requirements, and serving context preflight.
- [x] (2026-07-25) Replaced the naive prompt population and refactored `serve.benchmark` into one bounded representative capacity sweep, including strict corpus contracts, deterministic 128-record GSM8K/HumanEval/first-party materialization, native rendering, ordered one-engine execution, complete timing traces, canonical direct counters, typed terminal point failures, and the schema-version-2 `serving-result` artifact.
- [x] (2026-07-25 00:49Z) Ran the real Qwen3.5-0.8B vLLM comparison and MTP capacity search on the local RTX 3070 Ti 8 GiB target using all 128 `general-serving-v1` records per point; retained the successful and failed attempts in Trackio and opened the winning C64 run in Observatory.
- [x] (2026-07-25) Reduced the serving experiment set from 35 exploratory runs to 10 decision-quality runs: four C4 backend comparisons, the selected-binding C16–C64 curve, and one latency-boundary result. Training/evaluation evidence in the shared Trackio project was not modified.
- [x] (2026-07-25) Added serving runtime phases for model loading, initialization, warmup, measured inference, cleanup, and artifact export; changed Trackio host sampling to 1 second for GPU and 5 seconds for CPU; extended Observatory's SVG timeline to align GPU utilization, VRAM, and stage durations.
- [x] (2026-07-25) Re-ran the representative MTP C64 point with fine-grained telemetry and verified 44 host samples, including five samples during measured inference.
- [x] (2026-07-25) Grouped the shared runtime timeline into startup, inference/training execution, and finalization; added 128 timing-only request traces with queue/prefill/decode/engine timing plus group-aware vLLM KV-cache capacity, peak, and nine timestamped scheduler-pressure samples; verified the result on real C64 hardware and in the live Observatory UI.
- [x] (2026-07-25) Added the `serve.benchmark` telemetry definition, `serving-capacity-v1` projector, strict serving run models, constraint-relative single-point/legacy interpretation, curated plus additional backend-runtime settings, benchmark-population projection, generated transport schemas, and a dedicated frontend Overview. The live C64 run now reports `job.serving`, five passing constraints, and `unsaturated` instead of the generic fallback.
- [x] (2026-07-25) Added a transitional server-owned work-package capacity projection and Overview table for the ten retained single-point runs. It groups rows by recorded inference binding and exposes concurrency, aggregate TPS, mean response length, p95 TTFT/TPOT, request coverage, duration, failures, peak VRAM, and evidence state without pretending those historical runs are one canonical sweep.
- [x] (2026-07-25) Extended `serving-capacity-v1` from compatibility single-point evidence to canonical multi-point sweeps, typed resource/unsupported/failure boundaries, terminal saturation, complete decision-state fixtures, and accessible throughput/latency operating curves.
- [x] (2026-07-25) Added the strict screen work-package capacity/Pareto query through Python, HTTP, MCP, exports, and the frontend; all contenders remain visible and only equivalent eligible runs enter the frontier.
- [x] (2026-07-25) Updated the base workload catalog, foundation screen, Qwen 0.8B runner, developer documentation, generated OpenAPI/MCP/TypeScript schemas, deterministic fixtures, and an opt-in three-point real-vLLM integration gate.
- [x] (2026-07-25) Ran focused serve/Trackio/Observatory tests, frontend tests and production build, the representative vLLM GPU integration, `uv sync --all-packages --locked --python 3.12`, Ruff, Pyright, all eight import contracts, the full 337-pass/15-skip pytest suite, and `git diff --check`.
- [x] (2026-07-25) Re-ran the complete validation ladder after the serving Overview slice: Ruff and Pyright passed, all eight import contracts remained intact, pytest reported 339 passed and 15 skipped, Vitest reported 14 passed, the production frontend build passed, generated schemas were refreshed, and the rebuilt live Observatory returned `job.serving` for `serve.benchmark-00483a03`.
- [x] (2026-07-25) Verified the complete researcher-facing Overview in the in-app Observatory: response shape, sample size, duration, failures, memory, constraints, runtime settings, benchmark population, and all ten compatibility rows render from server-owned values. Reusing Trackio provider run handles reduced the live capacity-table request from about 16 seconds to about 3 seconds without changing normalized evidence.
- [x] (2026-07-25) Replaced the selected binding's cross-run reconstruction with one canonical `serve.benchmark` job that loads vLLM once and measures C4, C8, C12, C16, C24, C32, C48, and C64 in order. The live `serve.benchmark-0a15b893` Overview renders all eight points side by side directly from that run and labels the still-rising curve `unsaturated`.
- [x] (2026-07-25) Re-ran the complete validation ladder after the multi-point sweep slice: Ruff and Pyright passed, all eight import contracts remained intact, pytest reported 342 passed and 15 skipped, Vitest reported 14 passed, the production frontend build passed, generated schemas were refreshed, and `git diff --check` passed.
- [x] (2026-07-25) Re-audited this plan against all six canonical post-training documents and the current branch. The ownership model remains correct; the remaining gaps are typed point failures and canonical versioned serving-result evidence, complete decision-state transport fixtures, the strict contender/Pareto work-package view, operating-curve and Pareto visualizations, and final developer/release documentation.
- [x] (2026-07-25) Closed the re-audit gaps: canonical writers now persist only direct point evidence; resource boundaries remain typed and visible; the run and work-package views share one calculator; strict Pareto comparison is available through every transport; and frontend component tests cover the concurrency curves, all decision states, and incomparable evidence.
- [x] (2026-07-25) Completed the release validation: locked workspace sync, Ruff lint, Pyright, all eight import contracts, 348 passing Python tests with 16 expected skips, 15 passing Vitest tests, the production frontend build, three passing Playwright flows at desktop and narrow widths, catalog/work-package validation, corpus reproducibility, generated schemas, and `git diff --check`.
- [x] (2026-07-25) Rebuilt the local Observatory stack and verified the live Trackio-backed run `serve.benchmark-0a15b893`: the benchmark-specific Overview renders eight concurrency points and both operating curves, while the parent work package renders the strict comparison basis, one comparable unsaturated run, ten visible incomparable historical runs, and an empty frontier rather than falsely promoting an unsaturated contender.

## Surprises & Discoveries

- Observation: The public `Workload` already permits several concurrency values, but `ServeBenchmarkRequest` rejects every workload containing more than one, so the current API cannot represent a hardware-saturation run.
  Evidence: `packages/common/src/posttrain/common/selections.py` defines `Workload.concurrency: tuple[int, ...]`, while `packages/serve/src/posttrain/serve/requests.py` raises unless its length is exactly one.

- Observation: The public workload contract was already shaped for an ordered concurrency sweep; the vLLM adapter collapsed it into a single cell and the request layer rejected the remaining values.
  Evidence: The corrected adapter constructs one cell per ordered workload concurrency, initializes vLLM once, and the live run records eight metric/trace point indices under one run id.

- Observation: Reaching the serving binding's `max_num_seqs` is not proof that the hardware or product envelope saturated. The selected C64 point still passed every constraint while aggregate throughput increased from C48.
  Evidence: `serve.benchmark-0a15b893` measured 3,413.4 aggregate output TPS at C48 and 3,745.8 at C64, with 400.6 ms p95 TTFT, 14.3 ms p95 TPOT, no failures, and 90.9% peak KV-cache scheduler usage at C64.

- Observation: The packaged representative corpus contains only four prompts, and the offline vLLM capacity path does not use it. Instead, `_controlled_prompt_ids` repeats and rotates a few synthetic sentences until it reaches the target token length.
  Evidence: `packages/serve/src/posttrain/serve/benchmarks/resources/corpora/representative-v1.jsonl` has four records; `packages/serve/src/posttrain/serve/backends/vllm/offline.py` builds every request through `_controlled_prompt_ids` and never calls `representative_prompt_records`.

- Observation: GSM8K is useful as a realistic reasoning-prompt source but is too narrow to represent the framework's chat, code, structured-output, and tool-agent use cases by itself. Its questions can be used without answers because capacity measurement does not score correctness.
  Evidence: The pinned Hugging Face data revision `e53f048856ff4f594e959d75785d2c2d37b678ee` contains the GSM8K `main` train/test parquet data under an MIT license. The dataset consists of English grade-school math word problems, so it does not exercise code or tool-message rendering.

- Observation: OpenAI HumanEval provides a small MIT-licensed code-prompt population that complements GSM8K without introducing generated-code execution into the serving benchmark.
  Evidence: Hugging Face revision `e9b53e1677523f1e61e4d0960fd7502694a24bd4` contains the 164-record HumanEval dataset and license. The serving corpus uses only the prompt field and never executes or scores generated code.

- Observation: The current serving result already computes aggregate output-token throughput correctly as total generated output tokens divided by the measurement wall-clock interval, but it emits legacy metric names and only aggregate timing summaries.
  Evidence: `packages/serve/src/posttrain/serve/backends/vllm/offline.py` calculates `total_output_tokens / elapsed`; `packages/serve/src/posttrain/serve/results.py` emits `serve/output_token_throughput`, `serve/p95_ttft`, and similar names instead of the canonical `serve/request/*`, `serve/run/*`, and `serve/backend/*` evidence model.

- Observation: The canonical observation contract says request latency belongs on `InferenceTrace` and derived rates and percentiles should be computed from the lowest trustworthy evidence. The current benchmark emits assistant text samples as inference traces but does not retain per-request timing, token counts, or error state.
  Evidence: `docs/post-training/06-observation-and-lineage.md` defines per-request TTFT, ITL, end-to-end latency, token counts, truncation, error, and warmup fields; `packages/serve/src/posttrain/serve/api.py` currently emits only generated text and target token counts.

- Observation: Observatory has no `serve.benchmark` telemetry definition and its run Overview is training-specific. The Run config page does preserve unknown resolved inputs, which provides a safe fallback during migration.
  Evidence: `DEFAULT_TELEMETRY_DEFINITIONS` in `apps/observatory/src/posttrain_observatory/telemetry.py` contains training and evaluation definitions only; `apps/observatory/frontend/src/App.tsx` branches explicitly for SFT, DPO, and GRPO and has no serving overview.

- Observation: Observatory's original redaction expression treated every key containing `token` as a credential. It therefore hid product requirements and ordinary runtime fields such as `required_context_tokens`, `max_tokens`, and `max_num_batched_tokens`.
  Evidence: the live serving run returned `[REDACTED]` for those numeric fields even though they are necessary to explain the benchmark. The redaction policy now matches credential-shaped token keys such as access/auth/bearer/refresh/session tokens while preserving token-count fields; `api_key`, secrets, and passwords remain redacted.

- Observation: Backend-specific inference settings are already retained in the resolved `InferenceBinding.engine` snapshot. Observatory can curate and label them without importing `posttrain.serve` or duplicating backend validation.
  Evidence: `packages/work/src/posttrain/work/runner.py` snapshots resolved seats, and `apps/observatory` is intentionally forbidden from importing execution backends.

- Observation: The project manifest currently contains paths and tracking configuration but no project brief. Work-package metadata is intentionally forbidden from carrying outcomes and is not a sufficient typed home for durable serving requirements.
  Evidence: `packages/catalog/src/posttrain/catalog/project.py` defines the strict `.posttrain/project.toml` schema; `packages/work/src/posttrain/work/contracts.py` permits descriptive metadata but rejects outcome and decision fields.

- Observation: The global `workloads/foundation-smoke-v1@1` selection allocates
  only 1,024 context tokens and one concurrency point, so it cannot satisfy the
  repository project's new 32,768-token serving requirement.
  Evidence: The new work-package preflight accepts the workload at a
  1,024-token project requirement and rejects it at 32,768; the focused test
  suite reports 45 passing tests after covering both paths.

- Observation: `uv run --package posttrain-data` does not expose the package's
  `datasets` dependency in the active root environment, despite `packages/data`
  declaring it.
  Evidence: Both `--package posttrain-data` and `--project packages/data`
  raised `ModuleNotFoundError`; the reproducible maintainer command is
  `uv run --with 'datasets>=4.6.1,<4.7' posttrain workload verify
  workloads/general-serving-32k-sweep@1`.

- Observation: The materialized corpus contains exactly 128 model-visible
  prompt records with digest
  `9a9467fd8a5e744968d09a4d8fd6f4d92a089c50a84e1e6e7e5c5520a9f4e50e`.
  Evidence: The rebuild check succeeds from the immutable revisions; field
  inspection finds no GSM8K `answer` or HumanEval `canonical_solution`,
  `test`, or `entry_point` fields.

- Observation: Qwen3.5's tokenizer returns a `BatchEncoding` containing
  `input_ids` from `apply_chat_template`, while some processors return one
  nested token sequence. Treating either value as a flat sequence passes
  dictionary keys or a nested list to vLLM.
  Evidence: The first representative GPU run failed before warmup with string
  token ids; direct inspection of the pinned tokenizer returned
  `BatchEncoding({"input_ids": ..., "attention_mask": ...})`. The shared
  renderer now normalizes and validates both structured forms, and the focused
  serving suite covers the regression.

- Observation: `gpu_memory_utilization` is an engine allocation budget, not
  model-weight memory. At 0.85, vLLM requested 6.49 GiB of the GPU's 7.63 GiB
  usable capacity while only 6.22 GiB was free; the model weights themselves
  occupied about 1.53 GiB.
  Evidence: vLLM rejected the 0.85 binding before model load. The 0.80 binding
  loaded the same pinned model and reported the distinct weight and KV-cache
  allocations.

- Observation: MTP high-concurrency warmup needs workspace outside the KV-cache
  budget. A 0.80 binding with `max_num_seqs` 32 or 64 exhausts memory in the
  rejection sampler, while 0.75 with `max_num_seqs` 64 loads and performs
  substantially better than a 0.70/128-sequence binding at the same C64 point.
  Evidence: the 0.75/64 binding measured 3,365.67 output TPS, 449.77 ms p95
  TTFT, and 15.78 ms p95 TPOT at C64. The 0.70/128 binding measured 2,378.40
  output TPS and 1,996.02 ms p95 TTFT at C64, violating the project latency
  requirement.

- Observation: Observatory receives the Trackio runs and exposes all raw serving
  metrics, but still has no registered `serve.benchmark` summary or capacity
  calculator.
  Evidence: before experiment cleanup, the local UI listed 35 runs under
  `screen/qwen3.5-0.8b-serving-capacity`, shows the representative artifact and
  metrics for `serve.benchmark-7c3592ac`, and explicitly renders “No job view is
  registered for serve.benchmark.”

- Observation: Trackio's upstream 10-second automatic GPU sampling default is
  too coarse for short serving stages. It can record a correct 4.8-second
  measured-inference phase with no GPU sample inside that interval.
  Evidence: the first phase-instrumented C64 run recorded only seven host
  samples and none during measured inference. After the framework Trackio
  adapter selected a 1-second GPU interval, the replacement run recorded 44
  samples overall and five during measured inference.

- Observation: The telemetry-complete representative C64 run spent 29.45
  seconds loading the engine, 2.84 seconds warming up, 4.80 seconds measuring
  inference, and 1.67 seconds cleaning up. Measured inference averaged 82.2%
  GPU utilization and reached 100%; warmup averaged 81.7% and reached 100%.
  Evidence: Observatory's provider-neutral runtime-phase projection over
  `serve.benchmark-622e3b2b` reports the phase durations and per-phase
  aggregates from 44 timestamped Trackio host samples.

- Observation: Queue, prefill, and decode are request-grain intervals and
  overlap across concurrent requests, so rendering them as child run phases
  would imply a false additive wall-clock decomposition.
  Evidence: vLLM 0.25.1 exposes `queued_ts`, `scheduled_ts`,
  `first_token_ts`, and `last_token_ts` on each `RequestOutput.metrics`, while
  scheduler KV-cache usage is emitted independently for each engine step.

- Observation: The Trackio adapter previously required every non-Verifiers
  trace to contain conversational messages, even though the framework trace
  contract also permits timing-only inference evidence.
  Evidence: the first live request-trace run completed inference and then
  failed at persistence with `generic Trackio traces require a JSON messages
  list`; allowing an absent message list to normalize to an empty display
  transcript preserved the complete payload and passed a provider round-trip
  regression test.

- Observation: The first serving projector accidentally included the minimum
  throughput threshold in point validity. That made a below-capacity point
  invalid before eligibility could distinguish product capacity from operating
  safety.
  Evidence: point validity now uses context, latency, reliability, and complete
  evidence only; `output_tps` remains a recorded requirement violation and is
  applied by the eligibility calculator after the best valid point is chosen.

- Observation: A canonical resource-exhausted point has no request traces, so
  inferring a terminal boundary only from request denominators loses its
  meaning.
  Evidence: new writers emit typed
  `serve/run/point_resource_exhausted`, `point_unsupported`, or `point_failed`
  evidence and schema-version-2 artifacts retain the safe failure class and
  message. Observatory renders the boundary as a partial terminal point and can
  select the valid lower-concurrency point.

- Observation: A strict cross-run frontier needs a comparison basis, not just a
  shared work-package id.
  Evidence: the work-package service compares the requirements digest,
  execution target, workload, corpus digest, representative cohort, and
  calculator version against the newest decision-grade multi-point run.
  Mismatches are retained as incomparable instead of being filtered out.

## Decision Log

- Decision: Treat 50 tokens per second as a project-configured minimum sustained aggregate output-token capacity, not a framework-wide constant and not a per-request decode-rate threshold.
  Rationale: The canonical workflow assigns numeric thresholds to the project. Aggregate capacity is the relevant measure when concurrency is deliberately swept to saturate a fixed hardware profile. The framework must support other projects without silently inheriting this project's threshold.
  Date/Author: 2026-07-24 / Codex and user

- Decision: Product constraints are required context window, minimum sustained aggregate output tokens per second, p95 latency limits, and maximum request failure rate. Concurrency, batching, scheduler behavior, cache policy, and backend-native knobs are search variables.
  Rationale: The product describes the serving envelope it needs. The benchmark searches for an operating point on fixed hardware that satisfies that envelope.
  Date/Author: 2026-07-24 / Codex and user

- Decision: Add a typed project brief rather than a ninth catalog primitive or a `ServingRequirements` seat on `serve.benchmark`.
  Rationale: Requirements are project policy and interpretation context, not reusable framework assets and not backend execution settings. Keeping them out of the catalog preserves the existing eight primitive families and prevents the serving package from owning accept/reject policy. Every run snapshots the exact brief so historical evidence remains interpretable.
  Date/Author: 2026-07-24 / Codex

- Decision: Keep `Workload` as the reproducible benchmark methodology. It owns the representative request token shape, warmup and measurement policy, and ordered bounded concurrency sweep. It does not own the project's minimum TPS or latency thresholds.
  Rationale: The same workload must be reusable across candidates, while the project requirements state what counts as sufficient. Request shape remains necessary to reproduce engine pressure, but it is methodology rather than the headline product decision.
  Date/Author: 2026-07-24 / Codex

- Decision: Make a versioned prompt-corpus manifest part of the serving `Workload`; do not add a training `DatasetSelection` seat to `serve.benchmark`.
  Rationale: The corpus defines the request population used by an operating benchmark. It is not training data and does not carry evaluation scoring semantics. Keeping it inside the workload preserves the existing primitive and job-seat model while making prompt provenance reproducible.
  Date/Author: 2026-07-24 / Codex and user

- Decision: Publish `general-serving-v1` with 128 records: 64 GSM8K `main/train` questions selected deterministically from revision `e53f048856ff4f594e959d75785d2c2d37b678ee`, 32 HumanEval prompts selected deterministically from revision `e9b53e1677523f1e61e4d0960fd7502694a24bd4`, and 32 reviewed first-party message records covering chat, extraction, structured output, and tool use.
  Rationale: GSM8K gives substantially more realistic reasoning text than the synthetic seed, HumanEval adds code-shaped prefill, and the first-party records cover product-relevant message roles that neither public dataset contains. A 128-record population avoids repeating the same four prompts across ordinary sweeps while remaining small enough to audit and ship.
  Date/Author: 2026-07-24 / Codex

- Decision: Select public-source records by the lowest SHA-256 hashes of stable source identity and normalized prompt text, store the selected model-visible prompts plus source metadata in a checked-in JSONL resource, and verify the resulting corpus digest in tests.
  Rationale: The checked-in materialization removes runtime network dependence. Hash-based selection, immutable upstream revisions, source keys, licenses, and a rebuild script make the subset reproducible rather than an unexplained hand-picked sample.
  Date/Author: 2026-07-24 / Codex

- Decision: Use only GSM8K questions and HumanEval prompts; exclude answers, canonical solutions, tests, and correctness rewards from serving evidence.
  Rationale: The job measures inference capacity, not model quality. Gold answers would add unused evaluation content and invite accidental conflation with `eval.general`.
  Date/Author: 2026-07-24 / Codex

- Decision: Keep representative and controlled cohorts separate. The project-facing capacity gate uses model-rendered representative messages with a fixed output-token budget. Exact-token synthetic prompts remain a diagnostic systems workload with their own workload id and results.
  Rationale: Representative prompts exercise the tokenizer, native chat template, role support, and realistic prefill distribution. Exact-token prompts isolate backend performance. Combining their measurements would create a number with no coherent population.
  Date/Author: 2026-07-24 / Codex

- Decision: Label the current offline capacity method as fixed-length generation and report output-target attainment beside mean and p95 response length.
  Rationale: The runner sets minimum and maximum generation length to the selected target and ignores EOS. Researchers must not read the resulting response-length distribution as natural model stopping behavior; it describes controlled decode work for a systems comparison.
  Date/Author: 2026-07-25 / Codex

- Decision: Publish the tested MTP capacity binding with
  `gpu_memory_utilization: 0.75` and `max_num_seqs: 64` for the local 8 GiB
  target.
  Rationale: The larger 0.80 reservation prevents high-concurrency MTP warmup,
  while the 0.70/128-sequence alternative makes the same C64 workload both
  slower and latency-ineligible. The selected binding is the highest measured
  valid operating boundary on this hardware and representative population.
  Date/Author: 2026-07-25 / Codex

- Decision: One `serve.benchmark` run measures one model variant, inference binding, execution target, context allocation, and workload across the complete configured concurrency sweep.
  Rationale: The user needs one capacity curve and one operating-point explanation per contender. Separate runs per concurrency make run-level evidence fragmentary and force the UI to reconstruct one logical benchmark from unrelated attempts.
  Date/Author: 2026-07-24 / Codex

- Decision: Until the canonical one-engine sweep replaces retained single-cell runs, Observatory may present a `cross_run_compatibility` table for those rows, but must group them by inference binding and explicitly state that it is not one decision-grade sweep.
  Rationale: Researchers need access to existing concurrency evidence now, while target, binding, corpus, and requirement differences must remain visible instead of being silently merged into one curve.
  Date/Author: 2026-07-25 / Codex

- Decision: A canonical multi-point run renders its concurrency table directly from that run. The work-package `cross_run_compatibility` projection is used only for historical single-point evidence and is not fetched for a multi-point Overview.
  Rationale: Side-by-side points must share one run identity, model load, resolved binding, corpus, target, and requirement snapshot. Reconstructing the table from separate runs would weaken comparability and repeat run-level decisions on point rows.
  Date/Author: 2026-07-25 / Codex and user

- Decision: The serving operation emits measurements and point outcomes but never applies project thresholds. Observatory computes the best valid operating point and eligibility from the snapshotted requirements.
  Rationale: `packages/serve` must not own project thresholds, and Observatory is the established read-only interpretation layer. This also allows requirements to change without rewriting raw evidence, while preserving the exact requirement snapshot used for each historical view.
  Date/Author: 2026-07-24 / Codex

- Decision: A valid operating point satisfies context execution, latency, failure-rate, and measurement-completeness requirements. The selected operating point is the valid point with the greatest sustained aggregate output TPS. Eligibility additionally requires that point to meet the minimum TPS.
  Rationale: A raw high-throughput point that violates latency or reliability is not usable product capacity. Selecting the greatest valid point gives the screen a deterministic capacity value without selecting a model winner.
  Date/Author: 2026-07-24 / Codex

- Decision: A bounded sweep is considered complete when it observes either a throughput plateau, a higher-concurrency resource/unsupported boundary, or a higher-concurrency point that violates a hard product constraint. If the final configured point remains valid and throughput is still improving materially, the result is `unsaturated` and cannot produce a final eligibility decision.
  Rationale: Reporting the last tested point as capacity when the hardware is still scaling would understate the operating envelope and produce an arbitrary concurrency recommendation. A product constraint can legitimately cap usable capacity before raw hardware throughput plateaus.
  Date/Author: 2026-07-24 / Codex

- Decision: Observatory reports `eligible`, `below_capacity`, `latency_constrained`, `reliability_constrained`, `context_failed`, `unsaturated`, and `insufficient_evidence` as computed evidence states. These are not run statuses and do not mutate or promote artifacts.
  Rationale: Run status describes execution. Eligibility describes how completed evidence relates to project requirements. Keeping the vocabularies separate prevents expected saturation boundaries from appearing as infrastructure failures.
  Date/Author: 2026-07-24 / Codex

- Decision: Backend-runtime summaries are data-driven Observatory presentation definitions over the redacted resolved-input snapshot. Backend adapters continue to validate native settings; React contains no vLLM field list.
  Rationale: Observatory owns presentation, `InferenceBinding.engine` remains backend-native, and unknown settings remain available in Run config. This avoids a cross-package import and prevents frontend/backend configuration lists from drifting.
  Date/Author: 2026-07-24 / Codex

- Decision: New writers emit only canonical serving evidence. Observatory readers contain a bounded compatibility normalization for historical legacy metric names; the operation does not dual-write canonical and legacy values.
  Rationale: Dual writes violate the rule to record evidence once. Read-time normalization keeps existing runs useful during the migration window without making legacy names permanent.
  Date/Author: 2026-07-24 / Codex

- Decision: Serving adapters emit provider-neutral runtime phase events while
  Trackio remains the owner of timestamped host/GPU sampling. Observatory
  aligns those two evidence streams at read time and renders both utilization
  and memory in the shared SVG runtime timeline.
  Rationale: Capability code should describe semantic stages without importing
  a tracking backend. Reusing `RunContext.phase` and `system/*` avoids a second
  serving-only telemetry schema and gives all job kinds one interpretation
  path.
  Date/Author: 2026-07-25 / Codex

- Decision: The default Trackio adapter samples GPU telemetry every 1 second
  and CPU telemetry every 5 seconds, with both intervals validated and
  configurable through `TrackioSettings`.
  Rationale: A 10-second GPU interval misses ordinary 3–5-second serving
  measurement windows. One-second NVML sampling is sufficiently fine for stage
  attribution without coupling vLLM to Trackio; CPU sampling can remain coarser
  to limit long-run storage.
  Date/Author: 2026-07-25 / Codex

- Decision: Group semantic run phases as startup, inference or training
  execution, evaluation, and finalization, but keep request queue, prefill, and
  decode timings in a separate inference-detail projection.
  Rationale: Model loading and cleanup are one-time run costs. Request stages
  overlap under concurrency and must be compared as distributions rather than
  summed as a second run timeline.
  Date/Author: 2026-07-25 / Codex and user

- Decision: Record KV-cache usage as backend scheduler evidence and distinguish
  its group-aware token capacity and scheduler-reported usage ratio from
  process-wide allocated VRAM.
  Rationale: KV cache is one consumer inside the vLLM memory budget. Overall
  GPU memory cannot be relabeled as cache memory, and
  `num_gpu_blocks * block_size` is not correct for every hybrid-cache model.
  Date/Author: 2026-07-25 / Codex and user

- Decision: Capacity-screening inference traces may be timing-only. Tracking
  adapters must preserve their complete payload and may project an empty
  display transcript rather than inventing user or assistant messages.
  Rationale: ordinary serving traces intentionally omit prompt and response
  text under the benchmark retention policy. Requiring chat content would
  either reject valid evidence or force fabricated transcript data.
  Date/Author: 2026-07-25 / Codex

- Decision: Keep only decision-quality serving runs in the active Trackio
  project. Preserve the four representative backend comparisons, the chosen
  binding's C16–C64 capacity curve, and the higher-capacity latency boundary;
  remove failed/cancelled attempts, controlled synthetic diagnostics,
  superseded capacity searches, and replaced instrumentation runs.
  Rationale: Observatory should present the evidence used for product
  decisions, while transient debugging attempts make comparisons harder to
  scan. Cleanup is restricted to `serve.benchmark`; unrelated training and
  evaluation evidence remains intact.
  Date/Author: 2026-07-25 / Codex and user

- Decision: Persist only irreducible serving counters and complete request
  traces from new runs. Direct Python results may expose convenience rates and
  percentiles, but Trackio and the serving artifact do not make them a second
  source of truth.
  Rationale: aggregate TPS, failure rate, and latency percentiles are
  deterministic projections of token counts, measurement duration,
  denominators, and the complete measured request population.
  Date/Author: 2026-07-25 / Codex

- Decision: Define strict Pareto dominance as higher aggregate output TPS,
  lower p95 TTFT, and lower peak VRAM, with at least one strict improvement.
  Only comparable eligible contenders participate; all other states remain in
  the response and UI.
  Rationale: the three dimensions expose useful serving trade-offs without
  manufacturing a single winner or hiding constrained and incomparable
  evidence.
  Date/Author: 2026-07-25 / Codex

## Outcomes & Retrospective

Milestones 1 and 2 are complete on `codex/serving-capacity-observatory`. The six
canonical documents now agree that serving requirements are project policy,
capacity methodology belongs to `Workload`, `serve.benchmark` emits
uninterpreted point evidence, and Observatory alone computes eligibility and
comparable Pareto views. Projects now load a strict YAML brief through manifest
schema version 2, while schema-version-1 projects remain compatible. The
standard runtime snapshots the brief and digest and rejects serving workloads
or models below required context before opening a run.

Milestones 3 through 6 are complete. The benchmark supports two explicit and
non-interchangeable cohorts. Representative runs verify the checked-in
128-record corpus digest, deterministically schedule every record before
cycling, render through the model-native chat template, force an equal
128-output-token budget, avoid retaining generated text, and record corpus and
natural input-length evidence. Controlled exact-token runs remain available as
diagnostics. One operation now executes every ordered concurrency point under
one engine, retains point-indexed metrics and complete request timing traces,
and emits one schema-version-2 `serving-result` artifact. Higher-concurrency
resource, unsupported, and failure boundaries are typed, stop unsafe continued
execution, and preserve valid lower-concurrency evidence.

The first real hardware qualification used Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`, vLLM 0.25.1, and the local RTX
3070 Ti 8 GiB GPU. At C4, standard, MTP, TurboQuant, and MTP plus TurboQuant
measured 274.60, 365.97, 280.27, and 333.06 aggregate output TPS respectively
over all 128 representative records. The selected MTP 0.75/64 capacity binding
rose from 1,326.40 TPS at C16 to 3,603.40 TPS at the telemetry-complete C64
rerun, with 403.26 ms p95 TTFT and 14.80 ms p95 TPOT, while remaining below
the 1,000 ms p95 TTFT and 30 ms p95 TPOT requirements. A 0.70/128-sequence binding
was slower at the same C64 point and violated p95 TTFT at 1,996.02 ms, providing
the next-configuration product boundary. Trackio now retains only the ten
decision-quality serving runs. Observatory's dedicated serving Overview
computes the current C64 run's five constraint checks from the snapshotted
project brief and complete request traces. All five pass, but the run remains
`unsaturated` because it retained one operating point rather than the terminal
multi-point sweep required for a final eligibility claim.

Serving runtime stages now use the shared provider-neutral phase contract.
Trackio samples GPU metrics every second and CPU metrics every five seconds,
and Observatory's System Metrics timeline aligns GPU utilization, VRAM,
KV-cache pressure, and grouped phase spans. The replacement C64 proof
`serve.benchmark-00483a03` recorded 44 host samples and 128 request traces:
model loading took 29.89 seconds, warmup 3.12 seconds, measured inference 4.55
seconds, and cleanup 1.73 seconds. Measured inference averaged 94.6% GPU
utilization, with 7.66 GiB peak observed VRAM against the declared 8 GiB
target. The backend reported 241,113 tokens of group-aware KV-cache capacity,
91.1% peak scheduler usage, and nine timestamped usage samples. Request p50
queue, prefill, decode, and engine end-to-end timings were 0.1, 291, 1,707,
and 2,032 milliseconds respectively; Observatory labels these as overlapping
request distributions rather than additive run phases.

The canonical real-hardware proof is now `serve.benchmark-0a15b893`. It loaded
Qwen3.5-0.8B and the selected MTP binding once, then measured C4, C8, C12, C16,
C24, C32, C48, and C64 over the representative corpus. Every point passed the
recorded product constraints; C64 was the best observed point at 3,745.8
aggregate output TPS, 400.6 ms p95 TTFT, 14.3 ms p95 TPOT, zero failures, and
7.60 GiB peak observed VRAM. Observatory correctly reports the run as
`unsaturated`, because throughput was still rising at the binding's configured
C64 ceiling. Historical single-point runs remain available through the
explicitly labeled compatibility projection, while canonical multi-point runs
render their own side-by-side table without a cross-run fetch.

The researcher-facing product is complete for the first vLLM backend.
`serving-capacity-v1` derives all constraints and rates from canonical counters
and request traces. The run Overview explains requirements, population,
response length, every concurrency point, operating curves, runtime settings,
phase timing, GPU/VRAM/KV pressure, and terminal boundaries. The work-package
view exposes the strict comparison basis, all contender states, and the
throughput/TTFT/VRAM Pareto frontier through the same Python service used by
HTTP, MCP, exports, and React. Legacy single-point runs remain inspectable via
bounded read aliases and are never promoted into a strict frontier.

The checked-in integration test is an opt-in, three-point Qwen3.5-0.8B real
vLLM gate and supports standard, MTP, TurboQuant, and MTP plus TurboQuant
variants. The broader real-hardware release evidence above was already run on
the local RTX 3070 Ti and remains visible in Observatory. The current
limitation is deliberate: only vLLM has a qualified execution adapter; future
serving backends must emit the same provider-neutral evidence before joining
the calculator.

## Context and Orientation

The canonical product model is frozen in `docs/post-training/README.md` and `docs/post-training/01-workflow.md` through `06-observation-and-lineage.md`. A project owns its operating constraints and thresholds. A `screen` work package asks which foundation model and serving approach the project should start from. `serve.benchmark` is one job that supplies serving evidence to that decision. A model contender is the combination of an immutable `ModelVariant`, an `InferenceBinding` containing backend and engine settings, an `ExecutionTarget` describing hardware, and a `Workload` describing how measurement requests are generated.

The new term “serving requirements” means project-owned product constraints used to interpret benchmark evidence. It is not a catalog selection and is not accepted by the serving operation. The initial fields are required context allocation in tokens, minimum sustained aggregate output tokens per second, maximum p95 time to first token, optional maximum p95 time per output token, and maximum request failure rate. The exact serialized snapshot and its digest are retained on every run.

“Aggregate output tokens per second” means the number of successfully generated output tokens in one measured concurrency point divided by that point's measurement wall-clock duration. It is not the sum or average of each request's decode rate. “Operating point” means one measured concurrency value and its throughput, latency, failure, and resource evidence. “Best valid operating point” means the valid point with the highest aggregate output TPS. “Saturation” means that the configured sweep found the usable boundary: a throughput plateau, a resource/unsupported boundary, or a product-constraint boundary at higher concurrency.

“Prompt corpus” means the immutable model-visible message population referenced by a workload. Each record has a stable id, messages, category tags, reasoning mode, source identity, source revision, source record key, and license identifier. A corpus manifest has its own schema version and content digest. The prompt corpus is not a ninth primitive: it is a versioned resource owned by the serving workload. It is also not an evaluation plan, because `serve.benchmark` does not inspect answers or assign rewards.

`packages/common/src/posttrain/common/selections.py` and `catalog_schema.py`
contain the backend-neutral `Workload`, `ExecutionTarget`, and
`InferenceBinding` selections. `packages/serve/src/posttrain/serve/requests.py`,
`results.py`, `api.py`, and `backends/vllm/offline.py` implement the one-engine,
ordered concurrency sweep.
`packages/serve/src/posttrain/serve/benchmarks/workloads.py` contains
code-defined workload helpers. `packages/jobs/src/posttrain/jobs/definitions.py`
maps the standard `serve.benchmark` seats to the operation.

`packages/serve/src/posttrain/serve/prompts.py` parses and renders the packaged
`general-serving-v1` messages through model-native chat templates. The offline
benchmark uses that corpus for representative work and retains synthetic
token-id prompts only for the explicitly separate controlled diagnostic cohort.

Portable project discovery lives in `packages/catalog/src/posttrain/catalog/project.py`. Work-package contracts and resolved run snapshots live in `packages/work/src/posttrain/work/contracts.py` and `runner.py`. The current `.posttrain/project.toml` has no project-brief path. The current `.posttrain/work_packages/foundation_screen.yaml` and `packages/catalog/src/posttrain/catalog/base/workloads.yaml` demonstrate a single-concurrency serving smoke benchmark.

`packages/common/src/posttrain/common/execution.py` defines observer metrics, events, traces, and artifacts. `packages/tracking/src/posttrain/tracking/models.py` is the normalized provider-neutral read model. A `MetricPoint` can retain attributes, so a serving point can record `sweep_index`, `concurrency`, and measurement phase without pretending concurrency is a training step.

Observatory lives in `apps/observatory`. `telemetry.py` owns the versioned job definitions used by every read surface. `models.py` owns strict product-facing response types. `service.py` constructs views from `posttrain.tracking` readers. `sources.py` constructs work-package views. `http.py` and `mcp.py` expose the same service. `frontend/src/App.tsx` renders Overview and Run config. The committed `openapi.json` and generated `frontend/src/lib/api-schema.ts` must be regenerated whenever response contracts change.

The existing dirty file `apps/observatory/obseratory.pen` belongs to the user. This plan does not require modifying, reverting, staging, or regenerating it. Any implementation that uses that design source must first inspect the diff and coordinate the intended edit without overwriting unrelated work.

## Plan of Work

### Milestone 1: Amend the frozen product baseline

First make the product decision durable before changing code. Add a dated amendment to `docs/post-training/README.md` that states: serving requirements are typed project policy; benchmark workloads perform bounded concurrency sweeps on one target; `serve.benchmark` records capacity evidence; Observatory computes a constraint-relative operating point and eligibility; no operation or view selects a winning model.

Revise `docs/post-training/01-workflow.md` so the project brief explicitly owns required context, minimum sustained aggregate output TPS, latency, and reliability constraints. In the Screen section, describe concurrency and backend settings as search variables used to characterize one fixed target.

Revise the `Workload` section of `docs/post-training/02-primitives.md` to distinguish the product constraint from measurement methodology. Keep `Workload` as one of the existing eight primitive families. Explain that a workload may contain an ordered concurrency sweep and saturation method. Add the invariant that a workload does not own project acceptance thresholds.

In the same section, formalize the workload's prompt-corpus reference and the separation between representative model-rendered cohorts and exact-token diagnostic cohorts. State that corpus source, immutable revision, selection algorithm, digest, category mix, renderer, and output-token policy are comparability evidence. Serving corpora do not become `DatasetSelection` seats and do not carry evaluation answers or scores.

Revise `docs/post-training/03-work-and-evidence.md` so the Serving Pareto view compares each contender at its best valid operating point under the work package's snapshotted requirements and target. State that failed, incomplete, and constrained candidates remain visible.

Revise `docs/post-training/04-framework.md` to assign project-brief parsing and snapshot composition to `posttrain.work`, measurement to `posttrain.serve`, and eligibility/Pareto interpretation to Observatory. Preserve the rule that `posttrain.serve` does not own project thresholds.

Revise `docs/post-training/05-apis.md` with the `ProjectBrief`, `ServingRequirements`, capacity-sweep request/result, and Observatory query names described below. Revise `docs/post-training/06-observation-and-lineage.md` with the canonical point metrics, inference-trace fields, specialized serving view, eligibility states, and calculator provenance.

This milestone is accepted when the six documents agree on the same ownership split and `git diff --check` reports no whitespace errors.

### Milestone 2: Add the typed project brief and reproducible requirement snapshots

Add an optional project-brief path to `_ProjectManifest` and `ProjectLayout` in `packages/catalog/src/posttrain/catalog/project.py`. Introduce project manifest schema version 2 while continuing to read schema version 1. Version 1 projects have no project brief and remain runnable. New scaffolds write version 2 and point at `.posttrain/project.yaml`.

Create `packages/work/src/posttrain/work/project_brief.py`. Define strict immutable `ProjectBrief` and `ServingRequirements` types and a `load_project_brief(path)` function. The initial interface is:

    class ServingRequirements:
        required_context_tokens: int
        min_sustained_output_tokens_per_second: float
        max_p95_ttft_ms: float
        max_p95_tpot_ms: float | None
        max_failure_rate: float

    class ProjectBrief:
        schema_version: int
        objective: str
        serving: ServingRequirements | None

All counts and thresholds must be positive except `max_failure_rate`, which is within zero and one inclusive. Serialize the strict model in canonical key order and calculate a SHA-256 digest. Include the redacted serialized brief, schema version, and digest under `resolved_inputs.project_brief` for every run. Do not place the brief in `source_metadata`, because it is decision context rather than host metadata.

Extend `WorkPackageContext` and the standard runtime construction in `packages/jobs/src/posttrain/jobs/runtime.py` and the CLI runtime path under `apps/cli/src/posttrain_cli` to carry the loaded brief. Preserve direct-library callers by making the brief optional. Update `posttrain project show` to report whether a serving requirement is configured and its digest, without interpreting benchmark results.

Add serving preflight validation in `packages/work`, not `packages/serve`. For a `serve.benchmark` job with configured requirements, require the bound workload context allocation to be at least `required_context_tokens`, require the selected model's declared context support to be at least that value, and require the inference binding and explicit target to agree. The actual run still proves whether the declared configuration loads and operates.

Update the project scaffold, `.posttrain/project.toml`, and a new `.posttrain/project.yaml`. The repository's foundation-model project declares 50 minimum sustained aggregate output tokens per second and explicit context, latency, and reliability thresholds. Existing schema-version-1 fixtures remain in tests to prove compatibility.

Test project discovery, invalid briefs, digest stability, snapshot propagation, schema-version compatibility, and serving preflight in `packages/catalog/tests`, `packages/work/tests`, `packages/jobs/tests`, and `apps/cli/tests`.

This milestone is accepted when a new project prints its configured serving envelope, every serving run records the exact requirement digest, and an existing project without a brief still runs but later appears as `requirements_missing` in Observatory.

### Milestone 3: Refactor `serve.benchmark` into a capacity sweep

Keep the public seats of `ServeBenchmarkRequest` unchanged: inference binding, workload, and target. Remove the exactly-one-concurrency restriction. Tighten `Workload` validation so concurrency values are unique, positive, and strictly increasing. Add workload-owned saturation methodology fields with conservative defaults: material throughput improvement threshold, number of consecutive plateau intervals, and consecutive point-failure stop count. These fields describe when enough load has been attempted; they do not contain project requirements.

Before changing the sweep runner, replace the naive prompt population. Extend `PromptRecord` in `packages/serve/src/posttrain/serve/prompts.py` with immutable provenance fields and add a strict `PromptCorpusManifest` containing corpus id, revision, schema version, digest, record count, category counts, sources, selection algorithm, and license notices. A workload references a packaged corpus by id, revision, and digest and declares `cohort: representative` or `cohort: controlled`.

Add generic `posttrain workload materialize|verify` commands backed by a public
serve-owned operation beside the serving package resources. The operation uses
the workspace's existing `datasets` tooling, fetches only immutable revisions,
normalizes source text, applies the recorded SHA-256 selection rule, and writes
deterministic JSONL and manifest output. The generated resources are checked in
under `packages/serve/src/posttrain/serve/benchmarks/general_serving/resources/`;
ordinary benchmark execution has no network or `datasets` dependency.

Materialize `general-serving-v1` as:

    64 records  openai/gsm8k
                revision e53f048856ff4f594e959d75785d2c2d37b678ee
                config main, split train, field question
                category reasoning

    32 records  openai/openai_humaneval
                revision e9b53e1677523f1e61e4d0960fd7502694a24bd4
                split test, fields task_id and prompt
                category code

    32 records  first-party reviewed messages
                categories chat, extraction, structured-output, tool-use

The checked-in public-source records retain only stable source identity and model-visible prompt content. Exclude GSM8K answers and HumanEval canonical solutions/tests. Preserve MIT license notices. Review the 32 first-party records for secrets, personal data, accidental answer keys, and model-specific wording.

At runtime, deterministically shuffle the corpus from the workload seed and distribute distinct records across concurrency lanes and measured repetitions before cycling. Render every representative record through `render_prompt` and the selected model's declared native chat template. Record the corpus id/revision/digest, record id, category tags, reasoning mode, renderer id, and rendered input-token count on each trace. Do not store raw prompt text in ordinary traces.

Use the representative cohort for the product-facing capacity gate. It retains natural rendered input lengths and fixes the generated output-token count with `min_tokens`, `max_tokens`, and ignored EOS so different models perform the same amount of measured decode work. Record actual input and output counts. If a model cannot render a required role or reasoning mode, retain an explicit unsupported-record count; do not silently replace or drop those records.

Retain `_controlled_prompt_ids` behind a separate `controlled-capacity-v1` workload used for backend diagnostics and regression tests. Controlled results use exact input and output token counts and their own workload/corpus identity. Observatory never merges controlled and representative points into one capacity or Pareto calculation. Long-context occupied-input testing also remains a separate input-heavy workload rather than padding or repeating GSM8K text until it fills the context.

Replace the current single `BenchmarkResult` public meaning with:

    class BenchmarkPointResult:
        sweep_index: int
        concurrency: int
        status: measured | resource_exhausted | unsupported | failed
        requests_attempted: int
        requests_measured: int
        requests_failed: int
        output_tokens_measured: int
        measurement_duration_s: float | None
        peak_vram_bytes: int | None
        failure: safe typed failure | None

    class ServeBenchmarkResult:
        schema_version: int
        backend: str
        model_variant_id: str
        inference_binding_id: str
        workload_id: str
        execution_target_id: str
        context_tokens: int
        engine_start_duration_s: float
        points: tuple[BenchmarkPointResult, ...]
        saturation_observed: bool
        saturation_reason: plateau | resource_boundary | sweep_exhausted

The exact Python names may be adjusted during implementation only if the canonical API amendment and all consumers are updated in the same revision. Do not include project eligibility or a selected model in this result.

Refactor `packages/serve/src/posttrain/serve/backends/vllm/offline.py` so the vLLM engine loads once with the required context allocation and the adapter measures every configured concurrency point. Warm up each point according to the workload. A resource-exhausted or unsupported higher-concurrency point is a measured boundary, not automatically an operation failure; stop safely if the engine cannot be trusted after the error. An infrastructure error before any usable point fails the run.

For every measured request, emit an `InferenceTrace` containing concurrency, sweep index, warmup/measured phase, input tokens, output tokens, queue milliseconds, prefill milliseconds, decode milliseconds, TTFT milliseconds, time-per-output-token milliseconds when defined, engine end-to-end milliseconds, truncation, and error class. Queue is `scheduled_ts - queued_ts`, prefill is `first_token_ts - scheduled_ts`, decode is `last_token_ts - first_token_ts`, and engine end-to-end is `last_token_ts - queued_ts` when the pinned backend exposes valid monotonic timestamps. These request intervals overlap across concurrency lanes and are distributions, not additive run phases. Do not retain generated prompt or response text by default for capacity screening. If diagnostic samples are retained, honor the project trace retention/redaction policy.

Emit canonical irreducible point evidence as one metric batch per sweep point with attributes `sweep_index`, `concurrency`, `workload_id`, `inference_binding_id`, and `execution_target_id`. At minimum record:

    serve/run/requests_attempted
    serve/run/requests_measured
    serve/run/requests_failed
    serve/run/requests_unsupported
    serve/run/output_tokens_measured
    serve/run/measurement_duration_s
    serve/backend/peak_vram_bytes

The adapter may emit backend scheduler, cache, speculative-decoding, and utilization counters when available. For vLLM, retain the group-aware `kv_cache_size_tokens`, the measured peak of `SchedulerStats.kv_cache_usage`, and a bounded timestamped usage-ratio series during measured inference. Do not derive KV-cache bytes from process VRAM. Do not persist output TPS, failure rate, or p95 latency as a second source of truth; Observatory calculates them from counts, duration, and the full measured trace population. The serving telemetry definition must declare full request-trace retention as required evidence for a decision-grade benchmark. If a provider cannot return the complete population, eligibility is `insufficient_evidence`.

Write one versioned `serving-result` artifact containing the result, safe point failures, backend/version, calculator input counts, and the redacted resolved runtime configuration. Retain the existing artifact only through an explicit compatibility reader; new artifacts use one canonical kind and schema.

Add a legacy normalization module in Observatory that recognizes pre-change `serve/output_token_throughput`, `serve/p95_ttft`, `serve/p95_tpot`, and GiB memory values. It can construct a single-point historical view marked `legacy_single_point`; it cannot claim saturation or full eligibility unless enough evidence exists. New serving code must not emit legacy aliases.

Unit-test corpus schema and digest validation, deterministic source selection, license/source retention, absence of answer fields, deterministic seeded request scheduling, native rendering, unsupported-role accounting, and separation of controlled and representative cohorts. Also test request validation, sweep ordering, one-engine execution, plateau and resource-boundary termination, per-point metric attributes, inference traces, safe partial evidence, artifact schema, and legacy read normalization. Add a real vLLM integration test marked `gpu` and `network`; it must skip with a clear message unless an immutable model revision and compatible NVIDIA target are configured.

This milestone is accepted when the corpus rebuild is byte-for-byte deterministic, the representative workload uses all 128 audited records before cycling under the ordinary measurement budget, one operation produces a reproducible concurrency curve, direct evidence contains no project decision, and a fake sweep with concurrencies 1, 2, 4, and 8 yields four point records and complete request traces.

### Milestone 4: Add the serving run view and backend-runtime presentation

Add `SERVE_BENCHMARK_TELEMETRY` to `apps/observatory/src/posttrain_observatory/telemetry.py`. It declares the raw metric and trace requirements, calculator version, summary labels, artifact role, comparison keys, and configuration presentation. Add it to `DEFAULT_TELEMETRY_DEFINITIONS`.

Do not create a generic expression language for derived metrics. Add one registered first-party serving-capacity projector, version it as `serving-capacity-v1`, and reference that projector from the telemetry definition. The projector loads the canonical point metric batches, the complete inference trace population, execution target, inference binding, workload and corpus manifests, and project-brief snapshot. It rejects a capacity/Pareto calculation when the recorded corpus digest does not match the workload or when representative and controlled cohorts are mixed.

Add strict models in `apps/observatory/src/posttrain_observatory/models.py` for `ServingRequirementView`, `ServingOperatingPoint`, `ServingEligibility`, `RuntimeSettingValue`, `RuntimeSettingGroup`, and `ServingBenchmarkRunView`. Add `job.serving` to the `RunViewVariant` discriminator.

For each point, calculate aggregate output TPS as measured output tokens divided by measurement duration, failure rate as failed divided by attempted, and latency percentiles from measured non-warmup request traces. A point is valid only when evidence is complete and every configured context, latency, and reliability requirement passes. Calculate the best valid point by maximum aggregate output TPS. Calculate the eligibility state using the decision log above. Return explicit margins for TPS and every configured latency constraint.

Add a data-only runtime configuration presentation definition to the telemetry contract. Each field names a label, group, redacted resolved-input path, unit or formatter, importance, and optional backend selector. Register a vLLM serving presentation for model/context, parallelism, memory/cache, scheduler/batching, and acceleration. Curate at least backend version, dtype, tensor parallel size, max model length, GPU memory utilization, max sequences, max batched tokens, KV-cache dtype, eager/CUDA-graph mode, chunked prefill, prefix caching, and speculative decoding when present.

Add a benchmark-population projection showing corpus id/revision/digest, source datasets and immutable revisions, category counts, selected/requested/measured record counts, renderer, natural rendered input-token distribution, forced output-token budget, and unsupported-record count. It must make clear that task correctness was not evaluated.

The service, not React, resolves those fields from the already-redacted `InferenceBinding.engine` snapshot. Missing fields remain missing. Unknown backend fields remain available under “Additional backend settings” and in Run config; they are never dropped. Model weight precision stays under the model variant and is not mislabeled as a runtime setting.

Extend `ObservatoryService.get_run_view_response` to return `ServingBenchmarkRunView` for `serve.benchmark`. Expose the same model through the existing run HTTP route, exports, Python surface, and MCP `get_run_view`. Regenerate `apps/observatory/openapi.json`, `apps/observatory/mcp-schema.json` when required by the repository workflow, and `apps/observatory/frontend/src/lib/api-schema.ts`.

Add deterministic serving fixtures representing eligible, raw-fast-but-latency-constrained, context-failed, below-capacity, unsaturated, legacy-single-point, and missing-trace runs. Test the same calculation through the Python service, HTTP response, MCP tool, and export path.

This milestone is accepted when the fixture with 58 valid TPS reports `eligible`, the fixture with 71 raw TPS and excessive p95 TTFT reports `latency_constrained`, and neither the frontend nor any transport calculates those states independently.

### Milestone 5: Add the work-package serving-capacity and Pareto view

Add `ServingContenderView`, `ServingParetoPoint`, and `ServingCapacityWorkPackageView` to Observatory's strict models. Add:

    ObservatoryService.get_serving_capacity_view(
        work_package_id,
        project_id=None,
        source_id=None,
    ) -> ServingCapacityWorkPackageView

The service loads all `serve.benchmark` runs in the work package and applies the same `serving-capacity-v1` projector used by the run view. It verifies that each contender refers to the intended project requirement digest and execution target. Requirement or target mismatch remains visible and is marked `incomparable`; it is not silently omitted.

Eligible contenders participate in the Pareto calculation using maximum valid aggregate TPS, p95 latency, and peak VRAM. Ineligible, failed, unsupported, stale, and incomplete contenders remain in the response with their exact reason. The view reports the Pareto set but never ranks it into one winner.

Require the same representative workload id/revision, corpus digest, requirement digest, target, and capacity-calculator version for a strict Pareto set. A controlled diagnostic run or a different corpus remains visible as methodology evidence but is not silently mixed into the decision frontier.

Expose the view through Python, an HTTP route under `/api/v1/serving-capacity/work-packages/{work_package_id:path}`, an MCP tool named `get_serving_capacity_view`, and report exports. Keep the existing generic work-package route unchanged.

Test a work package containing the four headline scenarios. The output must show one eligible 58-TPS contender, one latency-constrained 71-raw-TPS contender, one context failure, and one 46-TPS below-capacity contender. Add mismatch tests for target and requirement digest, and source-qualified identity tests for Trackio and W&B readers.

This milestone is accepted when all read surfaces return the same contender states, values, margins, and Pareto membership from one service calculation.

### Milestone 6: Build the Observatory product surfaces

Split the serving Overview out of the training-specific `Overview` function in `apps/observatory/frontend/src/App.tsx`. Use the `view_kind` discriminator so `job.serving` renders a dedicated component and training/evaluation behavior remains unchanged. If the file has become too large to safely extend, extract serving components into `apps/observatory/frontend/src/features/serving/` without performing an unrelated frontend rewrite.

The run page begins with the configured requirements and fixed hardware profile. Its primary cards show selected valid aggregate output TPS, TPS margin, operating concurrency, p95 TTFT, optional p95 TPOT, peak VRAM, and eligibility. Its primary chart plots aggregate output TPS by concurrency with a 50-TPS requirement line from response data, not a hardcoded frontend constant. A linked latency-versus-throughput view identifies which higher-concurrency points violate latency or reliability. Point tooltips show measured counts, failure rate, context allocation, and evidence state.

Add a compact “Benchmark population” section. It shows the corpus and source revisions, category composition, rendered input-token distribution, output-token budget, and record coverage. It labels the run “capacity only; correctness not scored.” Do not show raw GSM8K, HumanEval, project, or generated text on Overview.

Render backend runtime settings from `RuntimeSettingGroup` values returned by the service. Provide a clear link to Run config for the complete resolved binding. Never call runtime settings training hyperparameters on a serving page.

Add a work-package serving-capacity page using the dedicated service response. Show the project requirements and hardware once, then a contender table with context, best valid TPS, operating concurrency, latency, memory, margins, and decision state. Add a throughput-versus-latency Pareto plot. Keep ineligible contenders visible below or beside the eligible frontier instead of filtering them out.

Provide empty, partial, loading, unsupported, legacy, unsaturated, and provider-unavailable states. Missing evidence is rendered as missing, never zero. Keyboard navigation, chart text alternatives, color-independent statuses, and responsive layouts are required.

Add Vitest component tests and Playwright coverage for the fixture scenarios. Build the production frontend and inspect the resulting screen at desktop and narrow viewport widths. If the existing `apps/observatory/obseratory.pen` design is used as a reference or updated, preserve the user's existing uncommitted changes and record the coordination in this plan before editing it.

This milestone is accepted when a user can open a serving run and understand why it passes or fails without opening raw metrics, then open its screen work package and compare all contenders without losing failure evidence.

### Milestone 7: Documentation, real integration, and release gates

Update the root `README.md`, `docs/developer-experience.md`, and relevant catalog/project examples so a developer can configure the project brief, select a fixed execution target, publish an inference binding with backend settings, select a representative prompt corpus and concurrency sweep, execute `serve.benchmark`, and open the Observatory capacity view. Explain that 50 TPS is the current project requirement, not a hidden framework default.

Document backend-specific settings as `InferenceBinding.engine` fields and explain that the Overview is curated while Run config is complete. Document legacy run behavior and the project-manifest schema-version-1 compatibility window.

Document the initial corpus composition, immutable public-source revisions, licenses, deterministic selection rule, checked-in digest, and rebuild command. Explain why GSM8K is one cohort rather than the whole serving workload, why gold answers are excluded, why output length is forced for capacity, and why controlled exact-token results are kept separate.

Add one real vLLM integration path in `packages/serve/tests/test_vllm_capacity_integration.py`. The test requires a CUDA host, an immutable public or authenticated model revision supplied through documented environment variables, and enough memory for the selected target. It executes at least three concurrency points, verifies canonical metrics and complete inference traces, and writes a `serving-result` artifact. Credentials, if needed for model access, come from the standard Hugging Face environment and are never stored in config, snapshots, test output, or artifacts.

Run the real benchmark on the target hardware profile used by the current project. Record the target id/revision, GPU model/count, driver/runtime versions, model variant, inference binding revision, workload revision, requirement digest, selected operating point, and eligibility in `Surprises & Discoveries` and `Outcomes & Retrospective`. Do not claim the integration complete if the credentialed/network/GPU release gate was skipped.

This milestone is accepted after the full validation ladder passes, the frontend production build succeeds, and the real target-hardware run is visible in Observatory with the same values as its artifact.

## Concrete Steps

All commands run from `/home/hammad/projects/rl` unless a command explicitly changes directory.

Before editing, confirm the active state and preserve unrelated work:

    git status --short
    git diff -- apps/observatory/obseratory.pen

During baseline and contract work, run:

    uv run pytest packages/catalog/tests packages/work/tests packages/jobs/tests packages/serve/tests -q
    uv run ruff check packages/catalog packages/work packages/jobs packages/serve
    uv run pyright packages/catalog packages/work packages/jobs packages/serve
    uv run lint-imports
    git diff --check

During Observatory service work, run:

    uv run pytest apps/observatory/tests packages/tracking/tests packages/tracking-trackio/tests packages/tracking-wandb/tests -q
    uv run ruff check apps/observatory packages/tracking packages/tracking-trackio packages/tracking-wandb
    uv run pyright apps/observatory packages/tracking packages/tracking-trackio packages/tracking-wandb

Regenerate transport contracts only after the Python models and routes are stable:

    uv run --package posttrain-observatory posttrain-observatory schema --openapi apps/observatory/openapi.json
    npm --prefix apps/observatory/frontend run generate:api

If the MCP schema has a repository command, use it and record the exact command here. If no generator exists, update `apps/observatory/mcp-schema.json` through the established application schema export rather than hand-maintaining a second contract.

Validate the frontend:

    npm --prefix apps/observatory/frontend test
    npm --prefix apps/observatory/frontend run build
    npm --prefix apps/observatory/frontend run test:e2e

Run the real backend integration after documenting the immutable model revision and target:

    uv run --with 'datasets>=4.6.1,<4.7' posttrain workload verify workloads/general-serving-32k-sweep@1

    POSTTRAIN_RUN_SERVE_GPU_INTEGRATION=1 \
    POSTTRAIN_SERVE_GPU_VARIANT=mtp \
    uv run --no-sync pytest packages/serve/tests/test_vllm_capacity_integration.py -m "gpu and network" -q -s

Before completion, run the repository validation ladder:

    uv sync --all-packages --locked --python 3.12
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    npm --prefix apps/observatory/frontend test
    npm --prefix apps/observatory/frontend run build
    git diff --check

Expected focused fixture evidence is structurally:

    Requirements: context=32768, min_output_tps=50, p95_ttft<=1000ms
    Model A: best_valid_tps=58, concurrency=6, state=eligible
    Model B: raw_peak_tps=71, state=latency_constrained
    Model C: state=context_failed
    Model D: best_valid_tps=46, state=below_capacity

Do not paste provider credentials, signed URLs, prompts, or generated user content into this plan when recording real-run evidence.

## Validation and Acceptance

The product contract is accepted when requirements are explicitly project-owned, concurrency is explicitly a workload search dimension, serving adapters emit no accept/reject decision, and Observatory is the only component calculating eligibility and Pareto membership.

Project compatibility is accepted when schema-version-1 projects still discover and execute, schema-version-2 projects load a strict brief, invalid thresholds fail before a run opens, and every new run records a stable requirement digest. A serving run without requirements remains inspectable but cannot be called eligible.

Serving correctness is accepted when aggregate output TPS uses output-token count divided by the exact point measurement interval; p95 latency uses the complete measured non-warmup request population; failures have explicit denominators; warmups never contribute to capacity; and context, target, workload, model, binding, backend version, and runtime settings are retained.

Benchmark-population correctness is accepted when the 128-record corpus rebuild matches the committed digest; source revisions and licenses are retained; no GSM8K answer, HumanEval solution, or HumanEval test field enters the corpus; representative prompts pass through each model's native renderer; every request identifies its corpus record without retaining raw text; and controlled synthetic measurements cannot enter a representative capacity decision.

Saturation correctness is accepted when a test proves each terminal condition: plateau, resource boundary, product-constraint boundary, and sweep exhausted while still rising. The last case must report `unsaturated`, not eligible or below capacity.

Observatory correctness is accepted when the same fixture returns byte-for-byte equivalent serving values and states through Python, HTTP, MCP, export, and frontend data. The frontend may format values but cannot recalculate or reclassify them.

Configuration presentation is accepted when a vLLM run displays curated backend settings on Overview, the full redacted engine mapping remains on Run config, an unknown engine field appears under additional settings, and a secret-shaped field is redacted before any transport sees it.

Work-package comparison is accepted when all contenders remain visible, only eligible contenders enter the Pareto set, target/requirement mismatches are explicit, and no API field names a winning model.

Backend integration is accepted only when the real vLLM test runs on a documented target rather than being skipped. If the model needs credentials, the test must skip clearly without them during ordinary development, but the release gate must be executed before marking the plan complete.

No implementation milestone is complete if `uv run lint-imports` fails or if `apps/observatory` imports `posttrain.serve`, vLLM, Trackio, or W&B directly.

## Idempotence and Recovery

Project schema migration is additive. Existing version-1 manifests remain readable and no command rewrites them automatically. New scaffolds use version 2. A later cleanup plan may remove version-1 support only after consumer fixtures and published migration guidance prove the compatibility window can end.

Benchmark runs never overwrite prior evidence. Retrying creates another run. A point-level OOM or unsupported configuration remains in the result artifact and does not delete lower-concurrency evidence. If the engine is unsafe after a resource error, stop the sweep and finalize the already measured points rather than attempting recovery in the same process.

Generated OpenAPI and TypeScript schemas are reproducible. Regenerate them from the Python application after model changes; do not repair generated types by hand. If generation fails halfway, discard only the generated-file changes from that failed invocation after confirming they do not overlap user work, fix the source model, and regenerate.

The implementation must not touch the existing `apps/observatory/obseratory.pen` changes unless that file is deliberately brought into scope. Never reset or overwrite the repository to clear that dirty file.

If canonical metric migration breaks old fixtures, add read-time legacy normalization and retain a dedicated legacy fixture. Do not reintroduce legacy emission or rewrite historical provider data.

If full inference trace reads are unavailable from one tracking backend, report `insufficient_evidence` for decision-grade eligibility and retain the artifact. Do not estimate p95 latency from a retained trace sample or coerce missing requests to successful zeros.

## Artifacts and Notes

The project brief should be easy to recognize in a run snapshot:

    project_brief:
      schema_version: 1
      digest: <sha256>
      objective: Select a foundation model that meets the product serving envelope.
      serving:
        required_context_tokens: 32768
        min_sustained_output_tokens_per_second: 50
        max_p95_ttft_ms: 1000
        max_p95_tpot_ms: 30
        max_failure_rate: 0.01

The workload remains methodology:

    workload:
      requests:
        context_window: 32768
        cohort: representative
        corpus:
          id: general-serving-v1
          revision: "1"
          digest: <sha256>
        selection_seed: 17
        output_tokens: 128
      concurrency: [1, 2, 4, 6, 8, 12, 16]
      warmup_repetitions: 1
      measured_repetitions: 5
      plateau_improvement_ratio: 0.05
      plateau_intervals: 2
      max_consecutive_point_failures: 1

The serving operation's artifact contains measurements, not policy:

    result_schema_version: 2
    model_variant_id: models/example@bf16
    inference_binding_id: inference/example-vllm-screen@2
    execution_target_id: targets/a6000-48gb@1
    points:
      - concurrency: 4
        status: measured
        output_tokens_measured: 8192
        measurement_duration_s: 141.2
      - concurrency: 8
        status: measured
        output_tokens_measured: 16384
        measurement_duration_s: 231.0

Observatory derives TPS, latency percentiles, validity, operating point, and eligibility and cites `serving-capacity-v1` plus the project-brief digest in its response.

## Interfaces and Dependencies

In `packages/work/src/posttrain/work/project_brief.py`, define and export:

    ServingRequirements
    ProjectBrief
    load_project_brief(path: Path) -> ProjectBrief
    project_brief_digest(brief: ProjectBrief) -> str

In `packages/catalog/src/posttrain/catalog/project.py`, extend `ProjectLayout` with an optional absolute `project_brief` path and accept manifest schema versions 1 and 2. `posttrain.catalog` discovers the path but does not interpret serving requirements.

In `packages/common/src/posttrain/common/selections.py`, retain `Workload` as the public selection and add only backend-neutral saturation methodology fields. Do not add project thresholds or a new selection family.

In `packages/serve/src/posttrain/serve/prompts.py`, define the strict
prompt-record and corpus-manifest contracts and keep native renderer resolution.
In `posttrain.serve.workloads`, implement deterministic corpus materialization
and verification. In
`packages/serve/src/posttrain/serve/results.py`, define `BenchmarkResult` for
one measured point, `BenchmarkPointFailure` for a safe terminal boundary, and
`BenchmarkSweepResult` for the ordered schema-version-2 result. In
`requests.py`, accept ordered multi-concurrency workloads and validate corpus
identity. In `api.py`, emit canonical point evidence, corpus-aware request
traces, lifecycle events, and the versioned artifact. In
`backends/vllm/offline.py`, load one engine and execute the representative or
controlled sweep without mixing them.

In `apps/observatory/src/posttrain_observatory/telemetry.py`, define `SERVE_BENCHMARK_TELEMETRY` and the data-only runtime configuration presentation fields. In a focused module such as `apps/observatory/src/posttrain_observatory/serving_capacity.py`, implement `serving-capacity-v1`. Keep all interpretation pure and deterministic so fixtures can invoke it without a provider.

In `apps/observatory/src/posttrain_observatory/models.py`, add the specialized serving run and work-package response models. In `service.py`, add the serving run and work-package query methods. In `http.py` and `mcp.py`, expose those service methods without reimplementing calculations.

No new external Python dependency is required for the core calculation. Continue using the pinned vLLM dependency in `packages/serve/pyproject.toml`, Pydantic in `packages/work` and Observatory, and ECharts in the frontend. If implementation reveals that an additional dependency is necessary, record the reason and lockfile change in the Decision Log before adding it.

The implementation touches only this repository. It does not require changes to `../trl`, `../trackio`, `../verl-upstream`, or any environment fork unless real provider conformance reveals a missing generic read capability. If a sibling repository becomes necessary, stop, update this plan with exact repository/commit ordering and validation commands, and do not mix uncommitted multi-repository changes.

Revision note (2026-07-24): Created the initial plan to formalize the user's decision that TPS, latency, and context are product constraints while concurrency and backend settings are search variables used to saturate a fixed hardware profile.

Revision note (2026-07-24): Replaced the naive four-prompt/synthetic-seed benchmark population with a provenance-aware 128-record representative corpus using GSM8K, HumanEval, and first-party message prompts; kept exact-token prompts as a separate diagnostic cohort so systems controls do not contaminate product capacity evidence.

Revision note (2026-07-24): Started implementation on `codex/serving-capacity-observatory` and completed the required narrow frozen-baseline amendment before changing runtime contracts.

Revision note (2026-07-24): Completed the typed project-brief slice, including portable discovery, compatibility, scaffold and CLI presentation, run snapshot propagation, and serving context preflight.

Revision note (2026-07-24): Completed the representative corpus and workload-saturation contract portion of Milestone 3, recorded the actual reproducible `uv --with` materialization command, and left the single-cell benchmark replacement explicitly in progress.

Revision note (2026-07-25): Implemented and validated representative single-cell execution, fixed structured Qwen chat-template output normalization, ran the real 128-record standard/MTP/TurboQuant comparison and MTP capacity search on the local 8 GiB GPU, and recorded the tested 0.75/64 binding plus the higher-capacity latency boundary without marking the one-engine sweep or Observatory calculator complete.

Revision note (2026-07-25): Cleaned the Trackio serving experiments to ten decision-quality runs, added provider-neutral serving phase events and one-second GPU sampling, extended the shared Observatory SVG timeline with utilization, and verified the result on a new representative MTP C64 run.

Revision note (2026-07-25): Refined runtime reporting into grouped one-time phases, overlapping per-request queue/prefill/decode distributions, and backend KV-cache pressure so the timeline does not imply false sequential work at concurrency.

Revision note (2026-07-25): Implemented and live-validated the grouped timeline, timing-only inference trace round trip, and vLLM KV-cache series on `serve.benchmark-00483a03`; replaced the older C64 instrumentation run while keeping the curated serving set at ten runs.

Revision note (2026-07-25): Closed the generic serving-Overview gap with a strict `job.serving` response and dedicated UI. The current complete-trace C64 point passes context, TPS, TTFT, TPOT, and failure-rate constraints, while the calculator deliberately reports `unsaturated` because the retained run contains one operating point rather than a terminal bounded sweep.

Revision note (2026-07-25): Added the cross-run compatibility table requested during live review and expanded workload-shape evidence with mean/p95 input and response lengths, output budget, request counts, and measurement duration. The table is a migration aid; the planned one-run, multi-point artifact remains the decision-grade destination.

Revision note (2026-07-25): Clarified response-length semantics in the Overview. The benchmark population now reports fixed-length policy and output-target hit rate, and the UI explains that these lengths measure controlled decode work rather than natural stopping behavior.

Revision note (2026-07-25): Re-audited the living plan after converting the benchmark to a one-engine, multi-concurrency job. The product boundary did not change; the remaining implementation work was narrowed to typed partial-point evidence, strict Pareto comparison, missing product visualizations and states, transport parity, documentation, and release gates.

Revision note (2026-07-25): Completed the re-audited scope. Added typed terminal
point evidence and schema-version-2 artifacts, corrected operating validity
versus throughput eligibility, implemented the strict work-package comparison
basis and Pareto frontier across all transports, added accessible run and
work-package visualizations plus desktop/narrow browser coverage, updated
catalog and developer guidance, and added the opt-in three-point Qwen 0.8B
vLLM gate without touching the user's existing `obseratory.pen` changes.

Revision note (2026-08-02): The checked-in representative corpus is now owned
by `posttrain.serve.benchmarks.general_serving` and verified through the
generic `posttrain workload verify workloads/general-serving-32k-sweep@1`
command. Older package-specific materializer examples in this historical plan
describe the pre-migration workflow.
