# Separate local release readiness from remote release promotion

This ExecPlan is a living document. Maintain it with
`docs/templates/PLAN.md`; keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` current after every milestone.
The canonical product baseline is `docs/post-training/README.md`. This work
does not change the meaning of projects, jobs, runs, artifacts, or evidence; it
only changes how an already-qualified framework version is published.

## Purpose / Big Picture

The release command should finish in roughly twenty minutes once code is ready.
Developers should receive fast feedback on a workstation or pull request, while
the final release job should perform only checks that require the release
runner, private package indexes, private image registry, dstack, or GitHub tag
state. A release must never publish a different source tree, wheel, or runtime
image merely to save time.

After this work, a maintainer can run a local readiness command that produces a
machine-readable readiness receipt. A candidate workflow consumes that receipt
and builds immutable wheels and runtime images once. The final workflow then
verifies the exact merged tree, consumes the candidate receipt and artifacts,
performs the bounded remote checks, promotes unchanged bytes, and creates the
tag and GitHub release. Re-running a failed final job resumes from its retained
receipt instead of rebuilding or re-running completed checks.

## Progress

- [x] (2026-08-09T14:20Z) Observed that PR, candidate, and final workflows each
  repeat broad source checks; the final runner spent several minutes in a full
  `uv sync` and test suite before reaching candidate restoration.
- [x] (2026-08-09T14:23Z) Identified the first final-release blocker: a valid
  candidate commit was rejected after squash merge because it was not an
  ancestor of the merged commit.
- [x] (2026-08-09T14:27Z) Added an ancestry-or-identical-tree check in PR #41;
  the first implementation still assumed the candidate commit object existed
  in the release checkout.
- [x] (2026-08-09T14:32Z) Identified the second blocker: the squashed candidate
  object was not available locally on the release runner.
- [x] (2026-08-09T14:34Z) Added PR #42 to resolve the candidate tree through the
  GitHub commit API and compare it with the merged tree. Local release tests
  pass; remote checks are in progress.
- [x] (2026-08-09T14:34Z) Merged and qualified PR #42; candidate run
  `31317350952` remains accepted after the GitHub tree lookup fix.
- [x] (2026-08-09T14:45Z) Merged PR #44, which permits only release plumbing
  and plan-document changes after candidate creation; build-input changes still
  block promotion.
- [x] (2026-08-09T14:51Z) Merged PR #45, removing the duplicate full test suite
  from final promotion while retaining locked-environment validation.
- [x] (2026-08-09T15:00Z) Final run `31319572261` succeeded: candidate
  provenance, final wheel build, dev/stable index checks, registry verification,
  real dstack canary, receipt retention, and `v0.3.5` tagging all passed.
- [ ] Add a local readiness receipt and make the candidate workflow consume it.
- [x] Remove duplicate full-suite execution from the final workflow while
  retaining exact-SHA, index, registry, dstack, and promotion checks.
- [ ] Add explicit resume checkpoints between final remote stages; the existing
  retained receipt remains the safe retry boundary for now.
- [x] Validate the successful 0.3.5 promotion and retain its receipt, release
  URL, stable-index evidence, image digests, and dstack evidence.

## Surprises & Discoveries

- Observation: the first candidate passed real dstack qualification, but final
  publication failed before any stable upload. Evidence: candidate run
  `31317350952` succeeded; final run `31318113663` failed in candidate
  restoration.
- Observation: squash merging preserves the candidate tree but not candidate
  commit ancestry. Evidence: the tree for candidate `84c82cb7` and merged
  commit `413ade31` was identical, while `git merge-base --is-ancestor`
  returned false.
- Observation: a full final validation run is redundant with green PR quality
  evidence and candidate validation, but remote artifact/index/image checks are
  not redundant. Evidence: final run `31318496543` spent about eight minutes
  in source validation before reaching the same candidate-restoration gate.
- Observation: a candidate commit may not be present in a shallow or
  squash-merged release checkout. Candidate provenance must therefore use a
  commit/tree API or a retained source manifest, not an assumed local Git ref.

## Decision Log

- Decision: Keep broad code checks in local readiness and required PR quality;
  do not repeat the full suite in final promotion.
  Rationale: source checks are deterministic and already run before merge;
  repeating them on a specialized release runner consumed the time budget
  without adding remote evidence.
  Date/Author: 2026-08-09 / user and Codex.
- Decision: Keep exact-SHA/tree provenance, immutable receipt verification,
  private registry verification, dstack canary, dev-to-stable promotion, and
  tag checks in final promotion.
  Rationale: these depend on external state and cannot be proven from a local
  checkout or PR artifact alone.
  Date/Author: 2026-08-09 / Codex.
- Decision: Accept a candidate when its commit is an ancestor of the merged
  source or its immutable Git tree is identical to the merged source.
  Rationale: this supports both merge strategies without accepting a
  same-version or unrelated candidate.
  Date/Author: 2026-08-09 / Codex.
- Decision: Treat the candidate receipt and wheelhouse as the build boundary;
  final promotion must reuse them and must never rebuild unchanged bytes.
  Rationale: rebuilding expands the failure surface and can silently change
  dependency or image provenance.
  Date/Author: 2026-08-09 / Codex.
- Decision: Retain a final receipt after every irreversible remote stage and
  make each stage idempotent.
  Rationale: a failed canary or index request must resume safely without
  republishing, retagging, or deleting another run's artifacts.
  Date/Author: 2026-08-09 / Codex.

## Outcomes & Retrospective

The 0.3.5 release is complete. Final run `31319572261` finished in 8m27s,
reused the accepted candidate runtime inputs, built the final-version wheelhouse
once, passed the real packed dstack canary, promoted unchanged bytes to stable,
and created [GitHub release v0.3.5](https://github.com/carbonteq-ai/posttrain/releases/tag/v0.3.5).
The remaining follow-up is a first-class local readiness receipt and more
granular resume checkpoints; neither is required to consume this release.

## Context and Orientation

`.github/workflows/quality.yml` runs deterministic code readiness checks on
every change. `.github/workflows/release-candidate.yml` builds a versioned
candidate, publishes development-index bytes and runtime images, and runs a
real consumer qualification. `.github/workflows/release.yml` is the final
promotion workflow; it currently repeats source validation, restores candidate
runtime metadata, builds final distributions, proves an index-only install,
checks the private registry, runs a dstack canary, promotes bytes to stable,
and tags the release.

`release/manifest.toml` is the authored framework version. The candidate
`python-release-receipt.json` records the exact distribution hashes and source
revision. `published.toml` and `workspace.lock.txt` record runtime image
digests and dependency locks. The final release must preserve those values.

“Local readiness” means checks that need only the repository, lockfile, local
toolchain, and hermetic fixtures: `posttrain-release check`, Ruff, formatting,
Pyright, import-boundary checks, and tests. “Remote promotion” means checks
that need private or irreversible systems: package indexes, registry, dstack,
GitHub tags/releases, and the release environment.

## Plan of Work

First, finish PR #42 and record the ancestry-or-tree provenance behavior in the
release tests. The final candidate-restoration step must query the candidate
tree through the GitHub API when the candidate object is not in the checkout,
then compare it with the checked-out merged tree before downloading evidence.

Next, add `scripts/release/readiness` (or an equivalent `posttrain-release
readiness` command) that runs the local checks once and writes a signed-by-hash
JSON receipt containing the source commit, source tree, lockfile digest,
framework version, tool versions, test result, and timestamp. The receipt is
evidence, not an authorization: the PR workflow must still publish the check
status, and the final workflow must verify the receipt source and tree.

Then update the candidate workflow to consume the readiness receipt, retain it
with the candidate evidence, and run only candidate-specific work after the
receipt is valid. Candidate-specific work remains runtime-image publication,
wheelhouse construction, development-index publication, clean consumer install,
private-registry verification, and the real dstack canary.

Finally, update the final workflow into short idempotent stages. It should wait
for the green push quality status for the exact merged SHA, verify candidate
tree/receipt/version/hash equality, restore the candidate wheelhouse and image
manifest, check the development index, perform only the final consumer and
remote-runtime checks that were not already part of the accepted candidate,
promote unchanged bytes to stable, and create the tag/release last. Each stage
must write or validate a retained receipt so a `resume_from_run_id` retry skips
completed stages safely.

## Concrete Steps

From `/home/hammad/projects/rl`, run the focused release tests while developing:

    uv run --no-sync pytest apps/release/tests/test_release.py -q
    uv run --no-sync ruff check apps/release/tests/test_release.py
    git diff --check

For local readiness, run the documented command from a clean worktree and
expect a JSON receipt whose `source_sha`, `source_tree`, `uv_lock_sha256`, and
`framework_version` match the checkout. A mismatch must stop candidate
publication rather than silently refreshing the receipt.

For final qualification, inspect the retained candidate receipt before dispatch
and expect the exact distribution hashes, Trackio post12 pin, runtime image
digests, and accepted dstack target. After promotion, query the stable index and
GitHub release and compare their hashes with the retained receipt.

## Validation and Acceptance

PR acceptance requires the existing quality and package-import checks plus the
new provenance regression: an ancestor candidate passes, a squash-merged
identical tree passes, and a different tree fails.

Candidate acceptance requires a valid local readiness receipt, clean consumer
installation from the development index, all committed runtime image digests
present in the private registry, and a successful real dstack canary.

Final acceptance requires no full-suite rerun on the release runner, exact
merged-tree/candidate-tree equality, unchanged distribution and image hashes,
successful stable-index promotion, a matching Git tag and GitHub release, and a
retained receipt that allows a retry without rebuilding or republishing.

The target timing budget is: local readiness under ten minutes on a warm
workstation, candidate-specific remote work under ten minutes when images are
unchanged, and final promotion under ten minutes excluding unavoidable dstack
queue time. If dstack capacity is unavailable, the release remains safely
deferred rather than claiming success.

## Idempotence and Recovery

No step may overwrite an immutable package version, runtime image digest, or
GitHub tag. Development and stable index checks must reuse exact bytes when
already present and reject partial or mismatched versions. A failed final run
must be resumed with its retained receipt; it must not rebuild distributions or
images unless the receipt is absent or fails hash verification. Candidate and
final cleanup must remain scoped to the current run.

## Artifacts and Notes

Relevant evidence from the current release investigation:

    candidate run 31317350952: succeeded; source 84c82cb7; real dstack qualification passed
    final run 31318113663: failed before publication; candidate commit was not an ancestor after squash merge
    final run 31318496543: source validation passed; candidate object was unavailable in checkout
    final run 31319572261: success in 8m27s after fast final-validation path
    PR #42: tree comparison now uses the GitHub commit API
    PR #44: release plumbing/build-input comparison
    PR #45: duplicate final suite removed

## Interfaces and Dependencies

The readiness receipt must expose `source_sha`, `source_tree`,
`framework_version`, `uv_lock_sha256`, `checks`, and `created_at`. The candidate
and final workflows must validate these fields before using any wheelhouse or
runtime manifest. The implementation may reuse `posttrain-release check`,
`receipt-check`, and `index-check`; it must not add Trackio, dstack, or registry
imports to framework-neutral packages.

Revision note (2026-08-09): created after the final 0.3.5 promotion exposed
squash-merge provenance failures and redundant release-run validation. The
scope is intentionally limited to release readiness evidence and promotion
latency; product and training semantics remain unchanged.
