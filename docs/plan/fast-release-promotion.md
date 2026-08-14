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
- [x] Run a real candidate/final release-path canary from the merged workflow
  before using it for the next production release.
- [x] (2026-08-12T03:30Z) Run candidate `31541881780` against the authored
  0.3.8 bytes: exact-source readiness, fresh OCI receipts, wheelhouse,
  development-index publication, clean consumer installation, and one packed
  RTX PRO 6000 dstack qualification all passed in 10m09s.
- [x] (2026-08-12T03:45Z) Resume final publication after the GitHub-release
  asset fix. Final `31544143377` verified the retained candidate, stable-index
  bytes, GitHub release assets, and the immutable `v0.3.8` tag in 1m05s.
- [x] (2026-08-12T03:45Z) Permit a release-plumbing-only commit after a squash
  merge by comparing it with the merged commit whose tree is identical to the
  candidate, not with GitHub's merge-base comparison.
- [x] (2026-08-12T03:45Z) Bind a tag to the immutable artifact-tree commit,
  rather than a later workflow-only approval commit.
- [x] Remove duplicate full-suite execution from the final workflow while
  retaining exact-SHA, index, registry, dstack, and promotion checks.
- [ ] Add explicit resume checkpoints between final remote stages; the existing
  retained receipt remains the safe retry boundary for now.
- [x] Validate the successful 0.3.5 promotion and retain its receipt, release
  URL, stable-index evidence, image digests, and dstack evidence.
- [x] (2026-08-12T03:55Z) Repair the published 0.3.8 checksum asset in place
  and make future release checksum files portable. A fresh GitHub-release
  download verifies `posttrain-wheelhouse-0.3.8.tar.gz` successfully.
- [ ] (2026-08-14) Expand the maintained-fork readiness boundary beyond the
  current Trackio/TRL-only receipt. Current audit: TRL `1.9.2.post11`, veRL
  `0.9.0.dev2`, and AutomationBench `1.0.5.post1` have stable-index bytes;
  Trackio `0.31.5.post14.dev16` has an immutable GitHub prerelease and a
  byte-identical stable-index readback; the selected CarbonTeq vLLM source is
  now the manually published source-overlay prerelease
  `carbonteq-v0.25.2.dev2` at `7817d845727af570352622dc8d58f2d43c76d89d`
  with retained archive SHA-256
  `8d4736461fbc3bf72075b4d84417208b3c5fc9ffc6f48bf26cbe9ef955cf307b`;
  AutomationBench is now the manual prerelease
  `carbonteq-v1.0.5.post1` at `908db2abd4a868acc37ab0850474bff653bea25c`
  with retained wheel/source checksums; and the dstack
  server/runner/shim candidate has no component release or production
  deployment. Posttrain candidate creation remains blocked on those
  independent receipts.
- [x] (2026-08-14) Remove forced recompression from actual-job publication.
  Live registry evidence showed a 4.21 GB compressed eval job whose job delta
  is only tens of megabytes. The focused job-image/runtime-image suite passes
  (`46 passed, 1 skipped`). Live manifest-overlap qualification remains open.
- [x] (2026-08-14) Correct runtime-lock closure projection to retain
  dependencies selected through locked package extras (including Torch's
  `cuda-toolkit[cudart]` edge). Regenerated every affected narrow lock and
  added a base-image regression test before retrying the candidate build.
- [x] (2026-08-14) Scope the base-image writable uv cache to the immutable
  lock digest. A failed candidate must not reuse an interrupted artifact from
  another reviewed closure; Docker layer reuse and immutable registry parents
  remain available for identical inputs.
- [x] (2026-08-14) Preserve explicitly declared, hash-addressed artifact roots
  when projecting the base runtime lock. The base now installs the reviewed
  Triton wheel from the internal non-volatile mirror, instead of resolving the
  same version through an upstream path that returned different bytes.
- [x] (2026-08-14) Bound concurrent job-kind publication to two workers and
  retry one known transient delivery failure. Candidate `31825254382` built the
  base and reached all kind publications, but a six-way fan-out overloaded the
  internal package index and registry. The release path now limits pressure
  while preserving content-addressed layer reuse and immutable request identity.
- [x] (2026-08-14) Qualify the bounded publisher against the stable dependency
  channel. Manual candidate `31828403358` completed in 14m14s, retained
  artifact `manual-runtime-0.3.16rc25-31828403358` (ID `9230111474`), and
  read back the base plus all seven kind-image digests. Its generated manifest
  and source-compatible per-kind locks pass both `posttrain-release check` and
  `posttrain-release lock-runtime-dependencies --check` locally.
- [x] (2026-08-14) Merge the current release line into the trace-facts/image
  branch, preserve the current Trackio and veRL receipts, and correct the
  runtime-image verification tests to model per-kind locks. The exact merged
  tree `14a8b8950a8fb5b0c4c43d2894eed2702095150b` passed deterministic
  readiness locally: 1,251 passed, 23 skipped, Ruff/check-format, Pyright,
  and import contracts all passed.
- [x] (2026-08-14) The first formal candidate against merged main correctly
  rejected the already-published `0.3.16` development bytes. Stable inspection
  confirmed that `v0.3.16` had already been promoted, so the release target is
  advanced to `0.3.17` rather than overwriting immutable artifacts. Candidate
  preparation now checks the stable `posttrain` index before expensive image
  or wheel work and fails with an actionable version-bump message.
- [x] (2026-08-15T20:20Z) Confirmed `0.3.17` is absent from stable while failed
  candidate `31833287598` occupies development. Kept the authored version at
  `0.3.17` and defined audited whole-version retirement for this failed,
  never-promoted candidate instead of advancing the stable version.
- [x] (2026-08-15T20:25Z) Proved the canary failure belonged to the transform
  kind: its image profile omitted the shared Trackio runtime even though all
  other kinds installed `profiles/common.txt`. Added the common transform
  closure, a transform import smoke, and an actual-job pre-GPU import gate.
- [x] (2026-08-15T20:27Z) Corrected candidate target handling so the default
  96-GB work package does not receive a rejected no-op override. Removed the
  stale 4090 release profile after the physical host no longer matched dstack's
  cached inventory. Focused validation: 42 tests passed.
- [x] (2026-08-15T20:45Z) Implemented the protected failed-candidate retirement
  gate. It binds the failed GitHub run to its retained receipt, compares the
  complete 26-package set with replacement source, verifies every development
  hash and an empty stable version, removes the coordinated version, proves
  absence, and retains preflight plus completion receipts. Repository
  validation: 1,255 tests passed, 23 skipped; Ruff, Pyright, import contracts,
  workflow parsing, release checks, lock checks, and diff checks passed.
- [ ] Retire the failed coordinated `0.3.17` development version after exact
  receipt/hash verification and retain the deletion receipt.
- [ ] Publish and qualify the corrected `0.3.17` candidate, then promote its
  exact bytes to stable and create `v0.3.17`.

## Surprises & Discoveries

- Observation: a development index can inherit the stable release of the same
  version. A candidate's hash mismatch is therefore evidence of an immutable
  version collision, not safe grounds to delete the development entry. The
  authored release manifest must advance before candidate publication.

- Observation: candidate `31833287598` packed a transform job. The transform
  Dockerfile installed only `profiles/transform.txt`; therefore its lock and
  image omitted `carbonteq-trackio`, and the runtime failed before tracking
  started with `ModuleNotFoundError: No module named 'trackio'`.
  Evidence: the Actions log identifies `model.transform`, kind digest
  `sha256:46c3a3...`, package key `d9730d...`, and actual-job digest
  `sha256:8bd5b7...` before the provider failed.

- Observation: the release workflow and dstack fleet cache still advertised a
  24-GiB RTX 4090 profile after the physical `pop-os` host changed and reported
  an RTX 3070 Ti. Release qualification is now restricted to the healthy
  96-GiB RTX PRO 6000 profile; stale scheduler identity is not accepted as
  hardware proof.

- Observation: `posttrain-release readiness` currently records only Trackio
  and TRL even though the executable runtime also selects CarbonTeq veRL and
  vLLM revisions, AutomationBench arrives through an external environment
  package, and dstack is a deployed component fork. A green readiness receipt
  can therefore omit four maintained supply-chain boundaries.
- Observation: publishing a fork package is not the same as eliminating
  source builds from runtime images. The veRL kind still resolves both `verl`
  and `vllm` from Git in its backend lock even though veRL dev2 is already on
  the stable index. Completing the vLLM release and switching those lock
  entries to retained release artifacts removes Git build work from the
  Posttrain image transaction.

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
- Observation: the first `0.3.16rc18` image candidate failed before publishing
  because the base lock omitted `nvidia-cuda-runtime`, which a selected
  `cuda-toolkit[cudart]` extra required. The lock projector traversed only
  ordinary package dependencies, even though `uv.lock` represents requested
  extras on the edge and their requirements under `optional-dependencies`.
  Evidence: GitHub Actions run `31824322845`; its image build stopped in
  `uv pip install --require-hashes` before any OCI output was pushed.
- Observation: the corrected candidate `0.3.16rc19` reached dependency
  installation but rejected a cached Triton artifact whose bytes did not match
  the reviewed PyPI receipt. The cache mount had one static name across lock
  closures, so a failed/partial download could survive a dependency update.
  Evidence: GitHub Actions run `31824686383`; the expected and observed
  SHA-256 values were different and no OCI output was pushed.
- Observation: the lock projector had collapsed an explicit direct artifact
  requirement into a normal versioned entry from the generic workspace export.
  That discarded the selected mirror URL even though the source profile had a
  full SHA-256 receipt. The projection now retains only direct HTTPS artifacts
  that carry a SHA-256 fragment; it does not broaden Git or ordinary package
  resolution behavior.
- Observation: the candidate called the runtime-image publisher for every
  framework version change, while its build request embedded the framework
  version and full source revision in the image identity. This made an
  unrelated patch release rebuild all runtime variants even when the actual
  runtime inputs were unchanged.
  Evidence: `publish_release()` selected every variant when no explicit
  variant was passed, and candidate publication passed the authored framework
  version to each BuildKit request.
- Observation: candidate `31825254382` built the corrected base successfully,
  but publishing all six kind images at once caused two independent transient
  failures: the veRL registry push returned `blob upload unknown` after layer
  work completed, and the serve build exhausted three attempts against the
  internal dev-index endpoint for `anyio` with `operation timed out`. No
  candidate manifest was retained and no promotion occurred.
  Evidence: GitHub Actions run `31825254382`; its publisher used
  `ThreadPoolExecutor(max_workers=len(RUNTIME_VARIANTS))` on the eight-vCPU
  release runner.
- Observation: a stable-channel runtime candidate can safely reuse immutable
  parents while building the two online-RL kinds concurrently. The veRL layer
  was the long pole, but the runner remained within the expanded 242 GiB root
  volume (worst observed use: 199 GiB) and completed without registry errors.
  Evidence: candidate `31828403358`, from 18:23:26Z to 18:37:40Z.
- Observation: tests that simulated registry facts defaulted to the supervised
  lock and therefore masked per-kind lock selection. The live manifest exposes
  different lock digests by design, so veRL test facts must use the veRL lock;
  image-level validation remains independent of lock equality. The same merge
  surfaced pre-existing formatting drift in 26 Python files, which the
  repository formatter corrected without behavioral changes.
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
- Observation: `sha256sum` received the absolute path to the wheelhouse
  archive, so the original `v0.3.8` checksum asset was cryptographically
  correct but unusable after download. The repaired asset contains only the
  archive basename and verifies with a fresh GitHub-release download.

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
- Decision: no Posttrain candidate starts until a generated fork ledger proves
  every selected CarbonTeq fork in direct packages, runtime locks, environment
  packages, and deployed execution services.
  Rationale: checking only Trackio and TRL permits a candidate to bake an
  unpublished vLLM source, a Git-built veRL dependency, or an unreleased dstack
  component while still calling readiness complete. The ledger distinguishes
  Python distributions, source-backed runtimes, and matched component
  releases, because those require different evidence but the same immutable
  boundary.
  Date/Author: 2026-08-14 / user and Codex.
- Decision: Publish the authored final version to development during candidate
  qualification, then promote its verified bytes unchanged.
  Rationale: RC and final versions are different artifacts. Qualifying an RC
  cannot prove the final wheelhouse, so it forces a rebuild and second GPU test.
  Development is the safe staging channel; stable publication and tagging stay
  final-only.
  Date/Author: 2026-08-12 / user and Codex.
- Decision: Keep `0.3.17` as the intended stable version and treat a GitHub
  candidate run as the RC identity. Permit retirement of a failed development
  version only when stable is empty, the run failed, every indexed file matches
  its retained receipt, and the entire coordinated version is removed with a
  deletion receipt before replacement.
  Rationale: a PEP 440 `0.3.17rcN` artifact cannot be renamed or promoted as
  byte-identical `0.3.17`. The audited exception preserves the stable version
  without weakening stable or accepted-candidate immutability.
  Date/Author: 2026-08-15 / user and Codex.
- Decision: Every job-kind control environment must install the shared
  framework runtime profile, and every actual-job build must import the default
  Trackio adapter before submission.
  Rationale: job-kind locks are runtime capability contracts, not merely lists
  of possible constraints; failure must occur during image build rather than
  after allocating a GPU.
  Date/Author: 2026-08-15 / Codex.
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
- Decision: Release checksum assets reference archive basenames, never
  runner-local paths.
  Rationale: a release asset must be directly usable after downloading from
  GitHub, independently of the build runner's filesystem layout.
  Date/Author: 2026-08-12 / Codex.
- Decision: Interpret parallel job-kind publication as bounded concurrency of
  two workers, and retry exactly once for explicitly classified registry/index
  delivery failures (`invalid content range`, `blob upload unknown`, and
  `operation timed out`).
  Rationale: these failures leave the immutable build request unchanged, so a
  fresh BuildKit delivery attempt is safe and can reuse completed layers. A
  small worker bound prevents six heavyweight builds from competing for the
  same runner CPU, registry upload sessions, and dev-index capacity; retries
  remain bounded so persistent dependency or build failures surface promptly.
  Date/Author: 2026-08-14 / user and Codex.
- Decision: Treat a successful manual stable-channel image candidate as the
  immutable runtime-input receipt for integration, but require the formal
  final-version candidate after the merged Posttrain source is green.
  Rationale: the manual candidate validates the image builder and its exact
  lock closure; only the formal candidate qualifies the final wheelhouse and
  the packed GPU job for the exact merged release tree.
  Date/Author: 2026-08-14 / user and Codex.
- Decision: Test runtime image verification against the selected variant's
  constraint lock, never an assumed shared workspace lock.
  Rationale: base and kind layers intentionally have different closures; the
  lock label proves closure identity while the image-level label proves that a
  runnable kind image occupies the slot.
  Date/Author: 2026-08-14 / Codex.

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

The 0.3.8 release is complete: candidate `31541881780` qualified the authored
final bytes, including one packed RTX PRO 6000 canary, in 10m09s; final
`31544143377` promoted the exact retained bytes, created the immutable tag,
and published the GitHub release in 1m05s. The initial runtime-image migration
accounts for most candidate time; subsequent source-only releases can reuse
digest-qualified images. The published checksum asset was verified again after
a portable-filename repair, and its generator now prevents runner-local paths
from recurring.

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
development, the workflow fails rather than overwriting it. A failed,
never-stable version may be retired only through the exact receipt gate; all
other changed source needs a new release version.

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

No step may overwrite an accepted package version, runtime image digest, or
GitHub tag. Development and stable index checks must reuse exact bytes when
already present and reject partial or mismatched versions. The only deletion
path is whole-version retirement of a failed, never-stable candidate after
matching the retained receipt; a mismatch fails closed. A failed final run
must be resumed with its retained receipt; it must not rebuild distributions or
images unless the receipt is absent or fails hash verification. Candidate and
final cleanup must remain scoped to the current run.

## Artifacts and Notes

Relevant evidence from the current release investigation:

    candidate run 31317350952: succeeded; source 84c82cb7; real dstack qualification passed
    final run 31318113663: failed before publication; candidate commit was not an ancestor after squash merge
    final run 31318496543: source validation passed; candidate object was unavailable in checkout
    final run 31319572261: success in 8m27s after fast final-validation path
    candidate run 31833287598: failed; transform kind omitted Trackio; stable 0.3.17 remained empty
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

Revision note (2026-08-15): Failed candidate `31833287598` exposed both a
transform-kind runtime-closure gap and ambiguity in the phrase “new RC.” The
plan now treats the workflow run as the RC identity, permits an audited
whole-version development retirement only before stable publication, keeps
`0.3.17` as the target, and requires Trackio importability before GPU
submission.
