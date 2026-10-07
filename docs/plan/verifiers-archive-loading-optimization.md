# Speed up native assessment archive loading

## Purpose / Big Picture

Recorded AutomationBench episodes should load without repeatedly verifying the same large source and observation input for every assessment batch. This change preserves native evidence and rejection rules while making archive replay practical. It does not change the frozen product baseline, task rewards, training algorithms, wire format or package boundaries.

## Progress

- [x] (2026-10-05) Reproduce DocuSign scored-archive cache eviction churn; implement lossless compact large-string keys within existing limits. Reload improves 84.013 → 3.891 seconds with identical bytes. Full native v1 gate and scoped Ruff pass; external/opt-in skips remain.

- [x] (2026-10-05) Profile two actual saved scored episodes and demonstrate exact output equivalence with bounded validation-scope reuse.
- [x] (2026-10-05) Implement native Episode/Trace wrap-validators and extend the existing archive integration test across Python, JSON and TypeAdapter inputs.
- [x] (2026-10-05) Validate native tests (348 passed, 83 skipped), ordinary default-loader timings (2.076 s / 0.260 s), malicious inputs and 118 environment regressions.
- [x] (2026-10-05) Update fork ledger, consumer notes and measured report with final results. Scoped Ruff and diff checks pass; full pre-commit encounters existing Markdown and package-index trust failures.

## Context and Orientation

The framework lives at `/home/hammad/projects/rl`. Native evidence implementation belongs to `/home/hammad/projects/verifiers-credit-candidate-20261003`, branch `codex/native-assessment-credit`, starting HEAD `959da6381394942596cb15e0cbe78f4df8594c22`. The native checkout was clean before these edits. Preserve unrelated dirty framework files.

An intrinsic proof records that an exact typed source or observation view passed its content checks. The existing private `verifiers/v1/_validation_scope.py` keeps successful proofs under an execution owner, bounded to 64 entries and 64 MiB of accounted storage. It clears them on exit and limits async inheritance. Context-dependent validity, such as which source a batch may use, is never established by this proof.

The framework pins the starting native commit in dependency manifests and `uv.lock`. This local optimization is not automatically adopted by installed consumers. Publication is outside this task: commit and push a qualified fork before any future immutable pin update. Do not mutate pins in this execution.

## Plan of Work

First change `restore_assessment_archive` in native `verifiers/v1/episode.py` and `verifiers/v1/trace.py` from a before-validator to a wrap-validator. Open `validation_scope(borrow=True)`, restore dictionary archive input, then call the entire Pydantic handler inside that scope. Standalone loads own their scope; nested Trace loads borrow an authorized Episode scope.

Next extend the existing archive integration test in `tests/v1/test_scoring.py`, preserving its real native execution and retained-credit fixture. Check Python, JSON and TypeAdapter loads, successful output equality, one source verification per exact source, shared nested ownership, correct credit retention, fresh-owner cleanup and cleanup after invalid archive rejection. Use existing tests for eviction, coordinate forgery, provenance and async isolation; do not introduce a separate cache.

Finally remeasure both saved scored episodes through their ordinary default loader, without an external prototype scope, using the existing profiler. Keep baseline evidence intact. Run available native and environment checks, then update this plan, the research report, native `CARBONTEQ_FORK.md` and framework `docs/tooling/verifiers/README.md` with results and limitations.

## Concrete Steps

From the native checkout:

    uv run --no-sync pytest tests/v1/test_scoring.py tests/v1/test_trace.py -q
    uv run --no-sync pytest tests/ -q
    uv run --no-sync ruff check verifiers/v1/episode.py verifiers/v1/trace.py tests/v1/test_scoring.py
    git diff --check

From the environment package `/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`, use its existing interpreter through `uv run --no-project`, with `PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src:tests`. Run the profiler retained at `/home/hammad/projects/rl/docs/research/verifiers-assessment-qualification/reward-candidate/assessment-reload-performance-data/profile_real_load.py`, passing an original episode path, `--mode baseline` and a new output filename. Here baseline means no external scope; after the implementation it exercises the optimized default loader. Do not overwrite the pre-change baseline JSON.

Original archives are `/tmp/automationbench-luna-sales-20261004/scored-roundtrips/sales.apply_project_label.json` and `sales.zoom_calendar_conflict.json` in the same directory. They must still exist. No new model run, credentials or external service is needed.

## Validation and Acceptance

Both real episodes must produce the same reserialized hashes as before. Each exact pooled source/view must have one intrinsic miss rather than one per batch. Default loading should approach the measured scoped timings of 2.09 seconds and 0.253 seconds; timing is evidence, not a brittle test assertion. Tampered data must still fail. Existing credit, lifecycle, linked-parent and isolation checks must pass. Native tests requiring unavailable external resources must be reported explicitly rather than called qualified.

## Idempotence and Recovery

Read-only replay and tests can be repeated. Preserve original archives and measured baselines. If a check fails, repair only the changed validators/tests; do not reset the whole worktree. The patch can be reversed narrowly if necessary. No migrations or wire changes are involved.

## Interfaces and Dependencies

Keep all existing public Episode, Trace and assessment APIs unchanged. The only internal signature change is the validator's added Pydantic handler argument. No dependencies are added. Native source owns the implementation; the framework owns the plan and recorded qualification evidence.

## Surprises & Discoveries

Pooling alone did not prevent nested after-validators from running repeatedly. The large recorded episode took 258.08 seconds unscoped and 2.09 seconds scoped with identical output. Memory barely changed. Both measured real episodes lack credit assignments; the native integration fixture has retained credit and supplies an additional regression gate, not real-workload credit performance evidence.

## Decision Log

Decision (2026-10-05): use the existing exact-content proof owner around full model validation, rather than a new cache or archive revision. It addresses the measured bottleneck with minimal new machinery and retains every contextual check. Defer private execution-ledger indexing until representative prefix-view workloads show it matters.

## Outcomes & Retrospective

The local implementation is complete. Ordinary loading reproduces the earlier
prototype speedup with unchanged output hashes: 258.08 s to 2.076 s and 7.26 s
to 0.260 s. Native tests finish with 348 passed and 83 skipped; the environment
subset passes all 118 tests. Integration tests cover retained native credit,
Python/JSON/TypeAdapter loading and nested scope reuse. Default-loader controls
retain tested rejection behavior. Real recorded credit-assignment performance
is not measured because both available benchmark archives contain none.

Repository-wide pre-commit fails on the pre-existing fork ledger heading-level
issue and private package-index certificate trust. Scoped Ruff and diff checks
pass. No training throughput improvement, publication or dependency adoption is
claimed. No dependency pins, model rollouts or commits were made. Publishing
and pinning this local change require their own release validation.

Revision note (2026-10-05): created this execution plan from the completed recorded-episode investigation and the user's authorization to implement archive-loading optimization.

Revision note (2026-10-05): marked the local implementation complete after
ordinary-loader benchmarks, native suite and environment regressions; recorded
pre-commit failures and skipped external gates explicitly.
