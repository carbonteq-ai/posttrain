# Fix LinkedIn world reload and rerun heldout62

This living plan follows docs/templates/PLAN.md. The frozen product baseline
is unchanged: this repairs environment serialization, not scoring semantics.

## Purpose / Big Picture

Make LinkedIn posts survive the environment's JSON save/reload path so all
heldout62 attempts can reach final scoring. Publish a new immutable environment
revision, then run a separate comparison without overwriting old evidence.

## Progress

- [x] (2026-10-17) Confirmed no scoring-fixed reruns exist in retained execution state.
- [x] Isolated exact prior source a0c7378b179e2b267382b8125192876502d82f89
  at /home/hammad/projects/worktrees/abv1-heldout62-scoring-fix; unrelated dirty
  environment and framework checkouts are preserved.
- [x] Demonstrated three regression failures on old source; repair passes
  58 relevant tests, including real milestone assertions and world-codec tests.
- [x] Built all six wheels, installed together in a disposable Python 3.12
  environment, and activated all six tasksets through the native loader.
- [x] User chose "Keep the fix local" after commit/publication authorization
  request. No commit, push, registry image publication or new GPU submission.
- [x] Prepared five local-snapshot work packages and a separate catalog overlay;
  these are preparation only, not accepted run receipts.
- [ ] Publish approved immutable source (not authorized; do not perform).
- [ ] Add versioned framework environment/eval selections and fresh run IDs.
- [ ] Sequential fresh evaluations and native 62-task/five-attempt reconciliation.
- [ ] Update all-model report and Page with comparable repaired-version results.

## Surprises & Discoveries

PostContent has aliased article fields and inherits extra=forbid without
population by field name. dump_world emits field names, while model_validate
expects aliases. Thus model-produced JSON can fail reload even with null fields.
The old 24B retry changed admission only and did not fix this serialization bug.
The owning environment repository forbids commit/tag/publication without
explicit authorization; prepare a tested patch before requesting that final step.

## Decision Log

- Decision: keep the patch local following the user's explicit response on
  2026-10-17. Do not commit or push. Do not infer authorization to distribute
  the local patch from the earlier requested Git publication option.
- Decision: the supported project-path environment source can retain a tree
  digest and wheel hash without a Git commit. Prepared this alternative under
  ignored `.posttrain/state/local-environments/heldout62-linkedin-fix/automationbench_v1`.
  This is an experiment snapshot of the externally owned adapter, not a new
  framework-owned adapter. No repaired-version remote run has been started.
- Decision: start from the exact old environment commit in an isolated checkout.
  Rationale: current sibling checkouts contain broad unrelated schema/tool work;
  the experiment needs only the measured fix. Date: 2026-10-17.
- Decision: retain the original source pin and all prior results as history.
  Rationale: a repaired environment must be a separately versioned comparison,
  never hidden rescoring or pooling of successful repetitions. Date: 2026-10-17.

## Outcomes & Retrospective

Local repair and validation complete. No fixed-version GPU evaluation has been
submitted. The old 24B r2 eliminated timeouts but still had five TaskErrors.
The workstation was observed idle (0% GPU, 41 MiB, no compute apps) on this turn.

Evidence: `/tmp/heldout62-scoring-regressions.log` records 58 passes;
`/tmp/heldout62-scoring-combined.log` records all six native taskset activations.
Focused Ruff and Pyright pass; package-wide Ruff has 37 unrelated findings,
format check would change 182 existing files, and correctly configured Pyright
has 220 findings elsewhere. The new regression file has zero Pyright errors.
A full 4880-test invocation was deliberately interrupted after 63 passes in
126 seconds in favor of the bounded 58-test relevant suite; do not claim a full
suite pass. Six-package independence and diff checks pass.

The repaired AutomationBench wheel is
`/tmp/heldout62-scoring-wheels/automationbench_v1-0.5.1-py3-none-any.whl`, SHA256
`2b2d9816f78becfbbaba98f98ee198b48ba99fdf8041b82e66e4b91e5ba9c6ea`.
Schema patch SHA256 is
`11db8e2a82c64e7ae63eec8a7bde5b090f267e084f1f90e6f94c54108fa80a63`.
The snapshot contains base provenance and the exact schema patch. Do not
confuse the unchanged package version/base SHA with a published repaired source.

Prepared overlay: `apps/lab/.posttrain/catalog/heldout62-scoring-fix-local.yaml`.
Prepared work packages:
`apps/lab/.posttrain/work_packages/{lfm26,lfm8b,lfm24b,qwen4b,qwen9b}_heldout62_scoringfix_local.yaml`.
All preserve tasks and budgets, use a separate source cohort, and admit eight
episodes. Both MoE strict read-only plans pass. LFM2.6B and both Qwen strict
plans reject existing inference warnings (batch invariance disabled; Qwen also
advertises optional speculative decoding). They need deliberate new inference
settings or explicit performance acknowledgements before launch. Separate logs
are at `/tmp/heldout62-scoring-plan-<arm>.log`. No image has been packed.

## Context and Orientation

Framework work is in /home/hammad/projects/worktrees/rl-rft on
codex/rft-warm-start, with existing uncommitted MoE/report work preserved.
Environment work is in /home/hammad/projects/worktrees/abv1-heldout62-scoring-fix,
a detached checkout of carbonteq-ai/verifiers-environments at a0c7378b179e2b267382b8125192876502d82f89.
Its standalone package is environments/automationbench_v1. The schema file
src/automationbench/schema/linkedin/post.py owns PostContent. The adapter codec
src/automationbench_v1/world_codec.py stores a WorldState as JSON and reloads it
before tools and scoring. Existing tests/test_world_codec.py covers Sheets
private scoring state and provides the integration regression location.
Artifacts under /tmp/lfm-moe-heldout62-results retain the failed native episodes.

## Plan of Work

### Milestone 1 Reproduce and repair the owning schema

Use model construction, dump_world and load_world to reproduce extra-forbidden
article fields. Enable field-name input on PostContent while preserving alias
input and strict unknown-field rejection. Add tests with nonempty article values,
JSON round trips and scoring of the milestone task. Run the package's focused
tests and broader tests, then build/install all six environment wheels as its
AGENTS.md requires. Document the delta and exact base in its root fork ledger.

### Milestone 2 Publish source and prepare reproducible selections

After explicit commit/publication authorization required by that repository,
commit and push only the tested isolated delta to a codex branch before changing
consumer pins. Update framework docs/tooling/verifiers/README.md and add new
environment/evaluation IDs using the full published SHA. Keep old bindings
unchanged. Build normal packaged jobs using registry.lan/carbonteq/posttrain-lab.
Do not mix repository changes in one implicit commit. The environment source
commit must be reachable remotely before a consumer recipe selects it.

### Milestone 3 Rerun and report

Rerun models available from the ten-model table under the repaired source;
confirm adapter availability before attempting trained variants. Use eight
concurrent episodes for MoEs to match their eight serving slots. Preserve
62 tasks x five attempts, 20 turns, 8192 reply tokens, 32768 episode output
tokens, 65536 context and model-specific recorded sampling. Retain failed runs
with distinct IDs; never overwrite or stitch episodes across attempts. Inspect
GPU occupancy and dstack before every submission, and run sequentially.

## Concrete Steps

From the environment package, run the focused codec/schema regression with the
matching dependency environment; record exact interpreter and commands once
resolved. Then run package validation from AGENTS.md and build six wheels.
From rl-rft, validate each new work package, inspect job plan, pack and run via
.venv/bin/posttrain --project-root apps/lab with --provider dstack for job run
and --registry registry.lan/carbonteq/posttrain-lab for both pack/run.

## Validation and Acceptance

Regression must fail on the old source and pass after repair, preserving both
camelCase aliases and snake_case fields and rejecting genuinely unknown fields.
An affected milestone post must reload and score successfully without deleting
state. Fresh runs must retain 310 unique episodes, the original ordered task
digest ed81cf1fe88f68bb8420bb46c5fdf3afde24660c7043812da680bb1c7bc9196f,
and no instance of the repaired finalization error. Report any other error,
timeout, truncation or scoring gap explicitly. Compare models only within the
same repaired source cohort; pair task bootstrap uncertainty separately from
absolute uncertainty.

## Idempotence and Recovery

Check active packaging and retained/dstack run IDs before submissions. Never
interrupt unrelated jobs. Preserve exact BF16/native prompts/parsers and all
original budgets. A failed arm keeps its ID/evidence and receives a new suffix
only after diagnosis. A missing publication permission blocks dependent remote
packaging, not local repair/testing.

## Artifacts and Notes

Previous report: docs/research/lfm-moe-heldout62-evaluation-2026-10-16.md.
Previous all-model audit: docs/research/lfm-heldout62-all-models-table-2026-10-16.json.
Current status question was answered from live retained states: the old retry
only removed timeouts; no scoring-fixed run exists.

## Interfaces and Dependencies

Keep public WorldState and dump_world/load_world signatures unchanged. Repair
the model input contract locally rather than stripping legitimate article data
or ignoring extra fields globally. No framework capability boundary changes.

Revision 2026-10-17: initial plan for isolated repair and versioned reruns.
