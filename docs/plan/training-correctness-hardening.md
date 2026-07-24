# Harden training result and evidence correctness

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

## Purpose / Big Picture

Training operations must never publish malformed backend output, mislabel
distillation evidence as GRPO, or turn missing teacher evidence into a numeric
zero. After this change, SFT, DPO, GRPO, and distillation validate their common
summary and filesystem outputs before materialization. Environment-backed
training preserves the original execution failure even when trace finalization
also fails. A developer can observe the behavior through focused contract tests
and the unchanged public training operations.

This is a correctness and internal-abstraction change. It does not amend the
frozen post-training product baseline: the baseline already requires native
Verifiers traces, explicit missing evidence, workspace-scoped outputs, and
teacher-scoring identity.

## Progress

- [x] (2026-07-24) Read the plan template, relevant frozen baseline sections,
  current training result types, operation orchestration, Verifiers bridge, and
  focused tests.
- [x] (2026-07-24) Added common validation for completed training summaries
  and backend output paths.
- [x] (2026-07-24) Corrected distillation trace and teacher-scoring semantics.
- [x] (2026-07-24) Made bridge finalization preserve an existing training
  failure.
- [x] (2026-07-24) Added focused regression tests and ran the package
  validation ladder.
- [x] (2026-07-24) Recorded the completed outcome and remaining abstraction
  work.

## Surprises & Discoveries

- Observation: the reusable Verifiers bridge already receives `purpose="grpo"`
  or `purpose="distill"` when built, but discards that value after using it in
  the dataset id.
  Evidence: `packages/train/src/posttrain/train/verifiers_requests.py` supplies
  the purpose, while
  `packages/train/src/posttrain/train/integrations/verifiers.py` currently
  writes `technique="grpo"` unconditionally during finalization.

- Observation: the public API emits `train/distill/teacher_failures=0` on every
  successful distillation result even though only the TRL teacher-client wrapper
  emits request-level failure observations.
  Evidence: `packages/train/src/posttrain/train/api.py` constructs the final
  zero without a backend result field that could support it.

- Observation: the full workspace test suite currently has one pre-existing
  catalog-integrity failure unrelated to this slice.
  Evidence: 292 tests passed and 15 skipped; only
  `apps/lab/tests/test_catalog.py::test_peft_bindings_settings_and_quantization_load_from_filesystem_catalog`
  failed because the base catalog records lock digest `cc45cf...` while the
  current `uv.lock` digest is `93bad6...`. Neither file is changed by this plan.

## Decision Log

- Decision: keep technique-specific requests and backend adapters; add shared
  validation only at their common completed-result boundary.
  Rationale: a universal trainer abstraction would hide meaningful SFT, DPO,
  GRPO, and distillation differences. Summary and workspace integrity are truly
  shared.
  Date/Author: 2026-07-24 / Codex.

- Decision: describe teacher scoring as `exact-token` and separately retain the
  selected inference binding and backend.
  Rationale: exact-token scoring is the stable algorithmic meaning. Whether the
  scorer is an external TRL server or an isolated veRL worker is execution
  topology, not the scoring mode.
  Date/Author: 2026-07-24 / Codex.

- Decision: remove the unsupported final teacher-failure zero instead of
  inventing a backend-neutral aggregate.
  Rationale: the frozen API explicitly says missing evidence is never coerced to
  zero. A future aggregate must be backed by an explicit backend result
  contract.
  Date/Author: 2026-07-24 / Codex.

## Outcomes & Retrospective

The hardening slice is complete. Completed summaries now reject impossible or
non-finite values. Backend outputs are validated as existing, absolute, and
workspace-scoped before online trace finalization or public artifact
materialization. Distillation trace artifacts retain their correct technique,
and portable snapshots preserve that identity. Teacher scoring now reports the
stable `exact-token` meaning together with the selected inference binding and
backend; the unsupported final failure zero is gone. When training and trace
finalization both fail, the original training exception is retained with a
diagnostic note.

The focused suite passed with 38 tests and one optional-dependency skip. The
complete training package passed with 95 tests and four optional-dependency
skips during the final workspace run. Ruff, Pyright, all eight import contracts,
and `git diff --check` passed. The full workspace suite remains red only because
of the unrelated stale catalog lock digest recorded above.

The next abstraction slice should type normalized `TrainingBinding.runtime`
values and the veRL cross-process manifest. That work is deliberately not mixed
into this correctness change.

## Context and Orientation

The repository is a Python 3.12 `uv` workspace. `packages/train` owns public
training operations and private backend adapters. A backend returns
`BackendTrainingResult`, which contains a `TrainingSummary` plus paths to the
materialized model, recovery checkpoint, and native summary. The public
operation then hashes and publishes those paths as artifacts.

`packages/train/src/posttrain/train/api.py` orchestrates SFT, DPO, GRPO, and
distillation. `packages/train/src/posttrain/train/results.py` owns public result
values. `packages/train/src/posttrain/train/backends/common.py` owns the private
backend result. `packages/train/src/posttrain/train/integrations/verifiers.py`
executes and preserves native environment traces for both GRPO and
distillation.

The working tree already contains an unrelated accepted checkpoint-policy
change in `packages/train/src/posttrain/train/backends/verl/worker.py`,
`packages/train/tests/test_verl_backend.py`, and
`docs/plan/verl-qwen35-grpo-distillation.md`. Preserve those edits.

## Plan of Work

First, add invariant checks to `TrainingSummary`: completed steps must be
positive, loss and performance values must be finite, and durations and rates
must not be negative. Add a validation method to `BackendTrainingResult` that
requires a model directory, summary file, optional existing checkpoint, and
ensures every output remains below the supplied run workspace.

Second, call backend-result validation once in the shared materialization path
in `api.py`. Remove the unsupported final teacher-failure zero. Change
`TeacherScoringSummary` so `mode` records `exact-token`, while binding identity
and backend record how scoring executed.

Third, retain the GRPO/distillation technique on
`VerifiersEnvironmentRolloutBridge` and its portable snapshot. Use that value
when constructing the trace artifact. Keep GRPO as the default for direct
bridge construction compatibility, while the public request builders continue
to pass the explicit purpose.

Fourth, replace duplicated online-training `try/finally` blocks with one small
internal helper that finalizes trace artifacts. If training has already raised
and finalization also raises, attach the cleanup failure as a note to the
original exception and re-raise the original. If training succeeds,
finalization failures remain ordinary failures.

## Concrete Steps

Run commands from `/home/hammad/projects/rl`.

After implementing each focused part, run:

    uv run pytest packages/train/tests/test_api.py packages/train/tests/test_verifiers_grpo_bridge.py -q

Then run:

    uv run pytest packages/train/tests -q
    uv run ruff check packages/train
    uv run pyright packages/train
    uv run lint-imports
    git diff --check

The focused suite should exercise invalid summaries, escaped or missing backend
paths, correct distillation trace metadata, exact-token scoring identity, and a
training failure accompanied by a finalization failure.

## Validation and Acceptance

Acceptance requires all existing package tests plus new regression tests to
pass. Constructing a `TrainingSummary` with NaN loss, zero completed steps,
negative runtime, or a negative rate must fail immediately. Returning a model
or summary outside the run workspace must fail before any artifact is emitted.
A distillation bridge must publish trace metadata with
`technique="distill"`. A completed distillation operation must not emit an
invented `teacher_failures=0`. When a backend raises `training failed` and
bridge finalization raises `finalize failed`, the caller must receive
`training failed` with a diagnostic note naming the finalization error.

## Idempotence and Recovery

All changes are ordinary Python source and tests with no migration or
destructive operation. Commands may be repeated. If a focused refactor fails,
revert only the files named in this plan; do not modify the pre-existing veRL
checkpoint-policy edits.

## Artifacts and Notes

No generated artifacts are required. Concise passing-test transcripts and any
unexpected behavior discovered during implementation belong in
`Surprises & Discoveries` and `Outcomes & Retrospective`.

## Interfaces and Dependencies

`TrainingSummary` remains a frozen dataclass and gains invariant validation.
`BackendTrainingResult` remains a frozen private dataclass and gains:

    def validate(self, workspace: Path) -> None

`TeacherScoringSummary` records the exact-token semantic mode plus the selected
inference binding id, revision, and backend. The change adds no dependency.

`VerifiersEnvironmentRolloutBridge` and `VerifiersBridgeSnapshot` retain a
`technique` value restricted to `grpo` or `distill`. The isolated snapshot
continues to reconstruct the same bridge behavior.

Revision note (2026-07-24): Created the plan for the first training
correctness-hardening slice after the user prioritized semantic correctness and
abstraction quality over resource planning and raw coverage.

Revision note (2026-07-24): Completed the first slice, recorded passing focused
and package validation, and isolated the one unrelated full-workspace failure
to a stale catalog lock digest.
