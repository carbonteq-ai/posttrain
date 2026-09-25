# Make evaluation transport failures visible and non-scorable

This ExecPlan is a living document maintained under `docs/templates/PLAN.md`.

## Purpose / Big Picture

An evaluation must not report a model-call failure as a zero-scoring but completed task. After this change, a provider error within a Verifiers episode is an invalid evaluation attempt, the run records bounded failure counters, and Observatory warns about it even if individual traces fail to synchronize. A trace-schema mismatch also fails a release compatibility test rather than silently dropping every trace.

## Progress

- [x] (2026-09-23 19:40Z) Reproduced the v3 failure from the exact `sha256:fca53de...` job image and its retained `traces.jsonl`.
- [x] Classify call-level provider errors in shared Verifiers evidence and evaluation population; exclude their rewards from Observatory aggregates.
- [x] Emit run-level failure and trace-sync diagnostics and render independent Observatory warnings.
- [x] Pin the compatible Trackio release, test the real trace-fact conversion contract, and fail before opening an incompatible evaluation run.
- [x] Validate the affected capability, Observatory, catalog and release surfaces (325 passed, 4 skipped), frontend tests/build, locks, import boundaries, and release inputs; replay all 60 retained records without external writes.
- [ ] Merge and publish an immutable runtime, then qualify a new evaluation and separately reconcile the old v3 run.

## Surprises & Discoveries

The v3 evaluation image contained `carbonteq-trackio==0.31.5.post14.dev24`, whose `TraceFactUpdate` rejects `task_id`. Its Posttrain trace projector emitted `task_id`, so all 60 trace uploads failed and each was retried once: 120 failed batches. The local published dev25 package accepts `task_id`; all 60 retained v3 traces construct successfully with it. The native artifact also contains 39/60 episodes with model-call HTTP 400 context errors, including all 36 non-simple episodes, while episode `ok` remained true and the run reported 60 complete, 0 failed, 0 truncated. The existing Observatory view warned only that trace sync was incomplete.

## Decision Log

- Decision: classify a serialized `calls[].error` as execution failure, independent of the episode-level `ok` field. Rationale: a provider failure is not a semantic task failure or valid reward zero. Date/Author: 2026-09-23, Codex.
- Decision: publish bounded run counters and typed warning categories, not raw provider messages. Rationale: alerts must work without synchronized traces and must not leak prompts or credentials. Date/Author: 2026-09-23, Codex.
- Decision: use the already-published Trackio dev25 dependency rather than widening the old client allowlist in the framework. Rationale: the maintained fork owns the generic trace-fact schema. Date/Author: 2026-09-23, Codex.
- Decision: probe the installed Trackio trace-fact constructor before opening an evaluation run. Rationale: release tests alone did not prevent the old image from wasting 60 episodes; a producer-side startup gate rejects incompatible images before GPU work. Date/Author: 2026-09-24, Codex.
- Decision: evaluation status is partial when execution attempts fail or truncate, even if all planned requests produced records. Rationale: transport completeness and valid measurement are different. Date/Author: 2026-09-24, Codex.

## Outcomes & Retrospective

Implementation is complete in an isolated worktree. The affected capability, Observatory, catalog and release tests passed (325 passed, 4 skipped); frontend tests and production build passed; import contracts, Ruff, both lock checks, release authored-input check, fork-ledger render, and diff check passed. The unqualified full suite initially stopped at collection because two training tests import optional `trl`, which this isolated environment lacks. Excluding those two yielded 1,694 passes, 39 skips and 25 failures: seven release/catalog pin assertions were corrected here, and the remaining training failures require the optional TRL runtime. Pyright has one pre-existing unrelated optional-member error in `packages/train/tests/test_adaptive_curriculum.py:70`; no changed file has a reported error. The old v3 run remains immutable evidence; this plan does not rewrite its historical metrics or claim its evaluation scores are qualified. No new runtime image has been published yet.

PR #118 is the review artifact. Its first quality run found two files needing repository-wide `ruff format --check`; those files were formatted, and the local format check now passes. The still-independent prompt-budget defect must be addressed in the Verifiers request path plus an appropriately qualified serving context before another held-out evaluation; this PR prevents it from masquerading as model failure but does not claim to remove the overflow.

## Context and Orientation

`packages/environment/src/posttrain/environment/verifiers_evidence.py` derives shared facts from native Verifiers traces. `packages/eval/src/posttrain/eval/backends/verifiers/adapter.py` counts evaluation attempts and streams native traces. `packages/eval/src/posttrain/eval/api.py` emits run metrics. `packages/tracking-trackio` translates trace facts into the maintained Trackio fork. `apps/observatory/src/posttrain_observatory/telemetry.py` declares eval summary and health rules; `apps/observatory/src/posttrain_observatory/traces.py` derives per-trace errors for the UI. The native `traces.jsonl` artifact remains the replay authority if transport synchronization fails. This change does not alter the frozen product meaning: the canonical `docs/post-training/06-observation-and-lineage.md` already requires execution errors and missing evidence to remain distinct from semantic failures.

## Plan of Work

First, add focused tests with an episode whose `ok` is true but one model call has `error={type: ProviderError, status_code: 400}`. Make shared `has_error` and reward projection recognize it; count the episode as failed, not complete, and expose safe error counters. Second, make the evaluation telemetry show provider-error counts and a threshold warning independent of trace ingestion; retain the existing trace-sync warning and expose its mismatch category. Third, pin Trackio dev25 and add a contract test that constructs a `TraceFactUpdate` from the framework's task dimensions using the installed Trackio package. Check that schema before opening an evaluation run. Synchronize the release ledger, CI wheel mirror, runtime profile, quantization pin, and catalog lock digest. Confirm all 60 archived v3 trace records convert in-process. Do not submit a new evaluation or alter the running v4 training run in this plan.

## Concrete Steps

From `/home/hammad/.codex/worktrees/eval-evidence-repair/rl`, run focused tests with the repository Python environment. Pin the exact published Trackio dev25 dependency and use `uv lock --check` to guard resolution drift; an upgrade command produced unrelated lock churn and was discarded before the minimal lock amendment. Run `uv run pytest packages/environment/tests/test_verifiers_evidence.py packages/eval/tests/test_api.py packages/tracking-trackio/tests/test_adapter.py apps/observatory/tests/test_product_service.py`, followed by `uv run ruff check .`, `uv run lint-imports`, `uv run pyright`, and `git diff --check`. A failing compatibility test under dev24 and a passing one under dev25 prove the dependency boundary.

## Validation and Acceptance

A synthetic provider-400 episode must produce `has_error=true`, no semantic reward scalar, one failed evaluation attempt, and one provider-error counter. A healthy episode must retain its previous success behavior. The Observatory eval view must include a provider-error warning from metrics even with zero synchronized traces, while also showing the separate sync warning. The exact v3 native artifact must yield 39 provider-error episodes and 60 successful local Trackio conversions under dev25. No new eval is claimed until an immutable runtime with these changes is published and its serving budget is qualified.

## Idempotence and Recovery

Tests and in-process replay are read-only. `uv lock` may be repeated and must preserve all unrelated dependency resolutions. Keep the primary checkout's dirty edits untouched. If registry access or package installation fails, retain this branch and report the gate rather than publishing a partial release or resubmitting an evaluation.

## Artifacts and Notes

Exact prior job image: `registry.lan/carbonteq/posttrain-lab/posttrain-job@sha256:fca53de46e61562166b0b21e8a9394ca08e7f27f363bd7ebe21687677d7f0bc6`. Historical v3 run: `eval-lfm26-heldout20x3-vortex-v3-agentic-20260923-r1` in `posttrain-lab`. The local audit copy at `/tmp/lfm-heldout-audit-RCYkOb/v3/traces.jsonl` is diagnostic only; the published artifact is authoritative.

## Interfaces and Dependencies

Do not add a new evaluator contract. Extend `EvaluationPopulation` with optional bounded provider-error counts, leave native trace payloads intact, and use the existing `TraceObservation`/`TraceFactSet` contracts. Trackio remains an external, exact-version dependency. Observatory reads run-level metrics for alerts so it can diagnose transport failures without native trace availability.

Revision note (2026-09-23): Initial plan records the reproduced exact-image failure and separates transport compatibility from provider-call classification.
