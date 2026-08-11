# Separate local release readiness from remote release promotion

This ExecPlan is a living document. Maintain it with
`docs/templates/PLAN.md`; keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` current after every milestone.
The canonical product baseline is `docs/post-training/README.md`. This work
does not change the meaning of projects, jobs, runs, artifacts, or evidence; it
only changes how an already-qualified framework version is published.

## Purpose / Big Picture

The release command should finish in less than fifteen minutes once code is
ready and a GPU is available. Fork publication is intentionally outside that
clock: Trackio, TRL, and other maintained dependencies are published and
qualified manually before their immutable package versions are pinned here.
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
- [x] (2026-08-09T16:05Z) Measured the local readiness baseline on the merged
  `main` checkout: `posttrain-release check` passed in 0.25s, Ruff and format
  checks passed, Pyright passed in 11.7s, import-boundary checks passed, the
  full suite passed (`1115 passed, 21 skipped`) in 18.3s, and the focused
  release/checkpoint/CLI gate passed (`89 passed`) in 7.6s. These checks do not
  require a package index, registry, dstack, or GitHub mutation.
- [x] (2026-08-12T02:46Z) Add `posttrain-release readiness` and
  `readiness-check`, run the deterministic source gate locally, and retain the
  exact-source receipt from Quality for candidate consumption. Local evidence:
  1,210 tests passed in 56.978s; Pyright passed in 21.094s; total 78.255s.
- [x] (2026-08-12T02:46Z) Change the candidate to build the authored final
  version in development rather than allocating RC-only wheel versions.
- [x] (2026-08-12T02:46Z) Make final promotion reuse the candidate wheelhouse
  and its one real packed dstack canary; it no longer rebuilds, re-installs, or
  runs a second GPU canary after source/tree and receipt verification.
- [x] (2026-08-12T02:46Z) Add focused receipt and workflow regression tests for
  final-version candidate artifacts and promotion provenance.
- [x] (2026-08-12T03:15Z) Make runtime image reuse depend on its runtime
  source digest, relevant dependency lock, parent digest, and installed CA
  bundle digest rather than the framework distribution version. Existing
  `0.3.7` image receipts intentionally lack that provenance and will rebuild
  once for `0.3.8`; later patch releases reuse registry-verified digests.
- [ ] Run a real candidate/final release-path canary from the merged workflow
  before using it for the next production release.
- [x] (2026-08-12T03:30Z) Run candidate `31541881780` against the authored
  0.3.8 bytes: exact-source readiness, fresh OCI receipts, wheelhouse,
  development-index publication, clean consumer installation, and one packed
  RTX PRO 6000 dstack qualification all passed in 10m09s.
- [ ] Resume final publication after the final GitHub-release asset fix. Run
  `31542795537` promoted the verified candidate bytes to stable and pushed
  `v0.3.8`, but stopped before creating the GitHub release because the final
  workspace did not restore `release-SHA256SUMS` from candidate evidence.
- [ ] Permit a release-plumbing-only commit after a squash merge by comparing
  it with the merged commit whose tree is identical to the candidate, not with
  GitHub's merge-base comparison. Final run `31543313634` exposed this second
  resume-path defect before any new promotion operation began.
- [ ] Bind a tag to the immutable artifact-tree commit, rather than a later
  workflow-only approval commit. Final run `31543674936` accepted the retained
  candidate and stable bytes but correctly refused to move the existing tag.
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
- Observation: v0.3.7's candidate took 8m13s and final took 6m38s. Candidate
  source validation consumed 2m23s; its dstack canary consumed 3m28s; final's
  second dstack canary consumed 3m17s. The canaries cover the same boundary but
  different bytes because the candidate used an RC version and final rebuilt
  the authored final version.
  Evidence: GitHub Actions runs `31536929524` and `31537708801`.
- Observation: the actual local shared readiness gate takes 78.255 seconds on
  the development machine, not under one minute: tests account for 56.978s and
  Pyright for 21.094s.
  Evidence: `posttrain-release readiness` receipt created 2026-08-12T02:46Z.
- Observation: the candidate called the runtime-image publisher for every
  framework version change, while its build request embedded the framework
  version and full source revision in the image identity. This made an
  unrelated patch release rebuild all runtime variants even when the actual
  runtime inputs were unchanged.
  Evidence: `publish_release()` selected every variant when no explicit
  variant was passed, and candidate publication passed the authored framework
  version to each BuildKit request.
- Observation: the initial 0.3.8 final run correctly promoted the retained
  candidate bytes and pushed the tag, but it could not create the GitHub
  release because `release-SHA256SUMS` was not copied from candidate evidence
  into the final workspace. The original tag guard then made a safe retry
  impossible even though that tag pointed to the correct merged commit.
  Evidence: final run `31542795537` completed stable-index promotion and
  `git push` for `v0.3.8`, then failed with `stat .release/release-SHA256SUMS`.
- Observation: GitHub's `candidate...release` comparison uses the merge base.
  After the candidate was squash-merged and a release-only repair was merged,
  that comparison treated the original candidate implementation as a fresh
  diff and rejected safe finalization.
  Evidence: final run `31543313634` listed the candidate's runtime and release
  source files even though merged commit `7cc40913` has the candidate's exact
  tree; `git diff 7cc40913..71b47e91` contains only the permitted workflow,
  test, and plan changes.
- Observation: once a release tag is published it must remain on the commit
  whose tree produced the wheelhouse. A later workflow-only repair is an
  approval/finalization commit, not a new artifact source and must not become
  the tag target.
  Evidence: `v0.3.8` resolves to `7cc40913`, whose tree matches candidate
  `22565c48`; final run `31543674936` used later approval commit `7d7a1262`
  and was correctly rejected by the existing-tag guard.

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
- Decision: Maintained forks are a separately published supply-chain
  prerequisite, not a Posttrain release-workflow stage.
  Rationale: candidate consumes already immutable, pinned Trackio/TRL packages.
  Building forks during framework promotion couples independent repositories
  and makes the release duration depend on work that should be qualified first.
  Date/Author: 2026-08-12 / user and Codex.
- Decision: Publish the authored final version to development during candidate
  qualification, then promote its verified bytes unchanged.
  Rationale: RC and final versions are different artifacts. Qualifying an RC
  cannot prove the final wheelhouse, so it forces a rebuild and second GPU test.
  Development is the safe staging channel; stable publication and tagging stay
  final-only.
  Date/Author: 2026-08-12 / user and Codex.
- Decision: One real packed dstack canary is required per immutable final
  artifact set, in the candidate workflow.
  Rationale: that canary proves the consumer package, image manifest, registry,
  and dstack path. Repeating it after tree/receipt verification adds latency but
  no evidence.
  Date/Author: 2026-08-12 / user and Codex.
- Decision: Treat runtime images as independently-versioned immutable
  dependencies, not as a framework-versioned output. Reuse requires exact
  runtime source, relevant lock, parent, and CA-bundle evidence; the framework
  wheelhouse remains the release-specific component injected into the packed
  job image.
  Rationale: framework source-only changes must not trigger a multi-image
  rebuild, but reusing an image after any actual runtime input changes would
  be unsafe. The first migrated release rebuilds because older receipts cannot
  prove all four inputs.
  Date/Author: 2026-08-12 / user and Codex.
- Decision: Final tag/release publication is idempotent: restore all release
  assets from candidate evidence, reuse an existing tag only when it resolves
  to the accepted merged commit, and upload assets to an existing release with
  replacement semantics.
  Rationale: a failure after tag push must be repairable by the same immutable
  candidate receipt; it must never require a new wheel build, stable-index
  overwrite, or second GPU qualification.
  Date/Author: 2026-08-12 / Codex.
- Decision: For a non-ancestor candidate whose original tree is no longer the
  release tree, find the equivalent tree commit in the merged history and
  evaluate only its direct diff to the release target.
  Rationale: this preserves the same build-input boundary after a squash merge
  and a release-only repair, without accepting a changed candidate source.
  Date/Author: 2026-08-12 / Codex.
- Decision: Track two immutable commit identities during finalization:
  `RELEASE_SOURCE_SHA` is the reviewed main commit that authorizes promotion,
  while `RELEASE_TAG_SHA` is the main-history commit whose tree is the actual
  candidate artifact source.
  Rationale: later workflow-only repair commits may authorize a resumable
  finalization but cannot truthfully become the version tag's source.
  Date/Author: 2026-08-12 / Codex.

## Outcomes & Retrospective

The 0.3.5 release is complete. Final run `31319572261` finished in 8m27s,
reused the accepted candidate runtime inputs, built the final-version wheelhouse
once, passed the real packed dstack canary, promoted unchanged bytes to stable,
and created [GitHub release v0.3.5](https://github.com/carbonteq-ai/posttrain/releases/tag/v0.3.5).
The v0.3.7 release is also complete: candidate `31536929524` succeeded in
8m13s and final `31537708801` succeeded in 6m38s. It retained valid release
evidence, but two versioned wheelhouses and two GPU canaries made the combined
path exceed the target. This plan now converts that evidence into a
single-canary final-version promotion path.

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

First, add a `posttrain-release readiness` command and JSON receipt. The
command runs deterministic source checks locally or in the normal Quality
workflow, then records the exact Git commit and tree, framework version,
`uv.lock` digest, manual-fork package identities, command results, and elapsed
time. `readiness-check` rejects a receipt from another tree, version, lock, or
incomplete command set. Quality uploads the receipt for its exact source SHA;
candidate downloads it only after the required Quality run succeeds.

Next, change `release-candidate.yml` to use the authored version from
`release/manifest.toml`. It builds that final version once, publishes it only
to development, verifies its hashes, installs it into a clean consumer,
verifies the runtime image manifest, and runs one real packed dstack canary. It
retains the final-version wheelhouse, readiness receipt, runtime manifest, and
dstack evidence. If a different artifact already occupies that final version in
development, the workflow fails rather than overwriting it; changed source needs
a new release version.

Finally, make `release.yml` promotion-only. It verifies candidate/merged
ancestry or identical-tree provenance, rechecks the downloaded candidate
wheelhouse and development-index bytes, and promotes them unchanged to stable.
It writes a promotion receipt binding candidate source/tree to merged tag
target, retains it before tagging, and creates the Git tag and GitHub release
last. It does not build wheels, create a clean consumer, verify the registry
again, or submit another dstack canary.

## Concrete Steps

From `/home/hammad/projects/rl`, run the focused release tests while developing:

    uv run --no-sync pytest apps/release/tests/test_release.py -q
    uv run --no-sync ruff check apps/release/tests/test_release.py
    git diff --check

For local readiness, run the documented command from a clean worktree and
expect a JSON receipt whose `source_sha`, `source_tree`, `uv_lock_sha256`,
`framework_version`, fork package identities, and successful command list match
the checkout. A mismatch must stop candidate publication rather than silently
refreshing the receipt.

For final qualification, inspect the retained candidate receipt before dispatch
and expect the exact distribution hashes, Trackio post12 pin, runtime image
digests, and accepted dstack target. After promotion, query the stable index and
GitHub release and compare their hashes with the retained receipt.

## Validation and Acceptance

PR acceptance requires the existing quality and package-import checks plus the
new provenance regression: an ancestor candidate passes, a squash-merged
identical tree passes, and a different tree fails.

Candidate acceptance requires a valid Quality-generated readiness receipt,
clean consumer installation from development, all committed runtime image
digests in the private registry, and a successful real dstack canary using the
exact final-version wheelhouse.

Final acceptance requires no source-suite rerun, wheel rebuild, consumer
install, registry recheck, or dstack canary. It requires exact
merged-tree/candidate-tree equality, unchanged development distribution and
image hashes, stable promotion, a matching Git tag and release, and a retained
promotion receipt that supports a retry without rebuilding or republishing.

The measured local source gate is currently under two minutes on this warm
workstation. The acceptance budget is: local/PR readiness under two minutes once
the environment is installed; candidate remote work under ten minutes with
cached images and available GPU; promotion under three minutes. The
candidate-to-tag path must remain under fifteen minutes excluding explicit human
approval and unavoidable dstack queue time. If dstack capacity is unavailable,
the release remains safely deferred rather than claiming success.

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
`framework_version`, `uv_lock_sha256`, `fork_packages`, `checks`, and
`created_at`. Candidate and final validate those fields before using any
wheelhouse or runtime manifest. The candidate receipt reuses the distribution
receipt plus retained wheelhouse and runtime evidence. The promotion receipt
adds `candidate_run_id`, candidate source/tree, merged source/tree, and the
candidate receipt digest. The implementation may reuse `posttrain-release
check`, `receipt-check`, and `index-check`; it must not add Trackio, dstack, or
registry imports to framework-neutral packages.

Revision note (2026-08-12): v0.3.7 measured candidate/final timings showed that
RC-to-final rebuilding causes two canaries. The target changed from roughly
twenty minutes to a sub-fifteen-minute path by qualifying the final version in
development once and making final a receipt-verified promotion only. Product
and training semantics remain unchanged.
