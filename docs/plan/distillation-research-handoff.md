# Commit and hand over the distillation investigation

This living execution plan follows `docs/templates/PLAN.md`. Maintain Progress,
Surprises & Discoveries, Decision Log and Outcomes & Retrospective. The user
explicitly authorized committing the investigation on2026-10-22. This overrides
historical no-commit instructions for this handoff; it does not authorize new
experiments, publication, dependency repinning or restarting training.

## Purpose / Big Picture

A colleague should be able to understand the five most useful previous studies,
inspect the latest real teacher contexts, restore the exact retained evidence,
run the CPU checks and identify the next unresolved decision without reading
the chat. This is a research handoff, not a product release or a claim that
distillation has qualified. The frozen product baseline is unchanged.

## Progress

- [x] Inventory current framework, experimental scripts, documentation and two
  related environment repositories; preserve all dirty work.
- [x] Identify approximately1.45GB of retained evidence and the latest explicit
  rejection of training in `focused-pool-qualification-20261020-r1`.
- [x] Create checksummed, lossless archives and an idempotent restoration tool.
- [x] Write the colleague guide and source-backed five-study comparison.
- [x] Validate CPU tests, changed framework/environment tests, archive integrity,
  secret screening and documented local links.
- [x] Commit related environment source repairs, then framework/runtime changes,
  then research source/evidence/handoff. Record exact commit IDs.
- [x] Produce portable Git bundles outside the repositories and verify them.

## Surprises & Discoveries

Most experimental scripts and evidence are untracked despite extensive written
plans. Committing only a prose summary would omit the actual implementation and
make the historical measurements difficult to inspect. Raw text dumps alone
would add approximately1.45GB to Git; lossless per-study archives can preserve
every selected byte while keeping the checkout practical.

The guidance ledger contains stale intermediate statuses. Final native audits,
saved checkpoint evidence and later plan decisions take precedence. The largest
student success percentage is not automatically the best controlled gain:
thinking modes, task sets, environment versions and repeated heldout inspection
must remain explicit.

## Decision Log

- Publication amendment (2026-10-22, subsequent direct user instruction): push
  all three handoff branches and inform Ali Safdar in Mattermost through Chrome.
  This supersedes the original local-only publication decision, while leaving
  dependency pins, release qualification and the failed training gate unchanged.
  Benchmark branch b3d26ab and environment branch c0f0baa were pushed first;
  the main research branch follows with updated GitHub links/provenance. Initial
  offline bundles retain their original checksums and are not silently replaced.

- Decision: retain all currently untracked research evidence in verified,
  lossless archives, leaving local originals intact. Ignore only those exact
  archived raw directory paths, not future evidence by default. Rationale:
  preserve the investigation without committing giant JSON/JSONL blobs directly.
  Date/author:2026-10-22/Codex.
- Decision: select five reference studies for evidence quality and relevance,
  including negative controls, rather than inventing five successful runs.
  Within-study student outcomes and their own matched controls remain primary.
  Date/author:2026-10-22/Codex.
- Decision: commit source locally, with no push or immutable consumer-pin change.
  Portable Git bundles provide a colleague-transfer option. Runtime/model/image
  availability remains a separate access requirement, not a claim that a local
  source commit reproduces a GPU result on an arbitrary machine.
  Date/author:2026-10-22/Codex.

## Context and Orientation

The main repository is `/home/hammad/projects/worktrees/rl-rft`, branch
`codex/rft-warm-start`, initially at`16edeaca`. Private experimental runners
belong to `scripts/qualification/`; frozen task identities are in its fixtures.
Plans in `docs/plan/` are historical execution records. Native captured tool
responses, original model tokens and final assertions remain evidence; prose
summaries are interpretations.

Related edits are owned by `/home/hammad/projects/automationbench`, branch
`codex/fix-six-task-data`, base`f96cb666066577dd8c203e20579fa282753f7ae4`,
and `/home/hammad/projects/worktrees/abv1-heldout62-scoring-fix`, initially
detached at`a0c7378b179e2b267382b8125192876502d82f89`. The first owns task
fixture corrections; the second owns the vendored environment, strict argument
admission, clock/tool access and LinkedIn compatibility repairs. Commit the
benchmark first, environment second, consumer last. No dependency pin is changed
before publication and release qualification. The TRL checkout is clean at
`c9af78c1c2ea04ad271e95b26b93dfadf8b9fca1`; historical run-selected TRL and
runtime snapshots must not be silently replaced with that checkout.

## Plan of Work

Create `docs/research/distillation-handoff/README.md` as the entry point. Include
the question, five-study results, current context workflow, source map, exact
known failures, data split, reproduction steps and experiment restart gates.
Keep compact result metadata readable in Git and archive the raw evidence under
`docs/research/experiment-archives/`. A manifest binds archive and original-file
SHA256 values. The restoration tool must reject traversal, unexpected archive
members and changed existing files instead of overwriting them.

Run existing targeted tests before committing the previously uncommitted source.
Record failures and limits honestly; no GPU allocation is needed for this task.
Do not modify unrelated running containers or any automation.

## Concrete Steps

From the main repository, inspect `git status --short` and the evidence manifest.
Run the documented archive checker and CPU suites. Review the exact staged diff,
check it for protected credentials and oversized files, then create scoped
commits. From each repository run `git bundle create` into a separate local
handoff directory and verify the bundle with `git bundle verify`.

## Validation and Acceptance

The guide must name every result population and source, distinguish process
completion from learning qualification, and describe how to restore all archived
paths. Archive extraction must be idempotent and refuse a conflicting local file.
The relevant CPU suites and framework/environment repair tests must have actual
logged results. Every related commit and bundle must be listed in the final
transfer receipt. Missing GPU images, cached model weights, private package
access and historical workstation capsules must be explicit remaining inputs.

## Idempotence and Recovery

Never delete raw evidence, reset dirty checkouts, amend historical commits or
overwrite a conflicting evidence file. Rebuilding an archive uses a fresh
temporary file and verifies its source manifest before replacing the generated
archive. A failed test does not authorize weakening an experiment gate. No old
time window is renewed, no old job is restarted and the training gate stays false.

## Artifacts and Notes

The final guide links original plans and restored evidence. Portable bundles and
a transfer receipt live outside the repositories, preventing a Git bundle from
recursively including itself. They contain source commits, not credentials or
model checkpoints. The user can share them through their own channel; this task
does not authorize sending messages to the colleague.

## Interfaces and Dependencies

The restoration utility uses Python's standard library, SHA256, JSON and tar/gzip.
No dependency changes or external services are required. Existing runtime
profiles and dependency pins remain executable historical authority; local fork
commits are explicitly unpublished candidates.

## Outcomes & Retrospective

The colleague guide and five-study comparison are written. All4863 evidence files
were verified, restored in a temporary tree, and verified idempotently on a second
pass. Three archive safety tests pass; framework49, benchmark65 and environment24
focused tests pass. Research sweep:521 passed/1 skipped/8 optional-dependency
collection errors, so no full-suite clean claim. Import contracts:9 kept/0 broken.
Source/evidence credential-format screens found no matches. Related fork commits
are b3d26ab and c0f0baa; runtime consumer commit c4c7ee9e. Research commit and
portable bundle receipt are resolved by the final transfer manifest outside Git.
Historical GPU experiments remain unchanged and the current training gate is false.

Final transfer: runtime commit`c4c7ee9e`, research/evidence commit`a6e1fed9`.
All three full-history bundles verified with no prerequisites. The transfer manifest
at `/home/hammad/projects/handoffs/distillation-20261022/transfer-manifest.json`
records exact delivered branch tips and SHA256 values. All three worktrees were
clean after source commits; no push, new GPU job or dependency repin occurred.
