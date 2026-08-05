# Generate releases from one manifest and verified build receipts

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

Source finding: `docs/dx-improvements/v0.2.5/README.md` finding 21. This plan
is self-contained.

## Purpose / Big Picture

Posttrain releases all first-party packages in lockstep (one coordinated
version). Today a release bump is authored by hand: the v0.2.5 bump commit
(`9e65b2ba`) touched 27 files — 26 `pyproject.toml` files each restating the
version in their own `version` field and in exact `==` pins on every
first-party dependency (~80 literals), plus hand-edited
`dependency_lock_sha256` values inside `packages/catalog/src/posttrain/
catalog/base/training.yaml`, a regenerated `uv.lock`, and a lab test that
hardcodes the version string. Pinned image digests live in
`packages/runtime-images/src/posttrain/runtime_images/published.toml` and
fork constraints in `release/github-constraints.txt`. Any missed edit ships
silently and surfaces as consumer-side drift.

After this plan, one generated release PR changes one authoritative version,
expands all repeated metadata, records immutable build receipts, and fails CI
when any generated file drifts. Changed images are built and pushed once, the
merged commit is re-verified, and only then is the matching tag created. A
consumer installs the framework and maintained forks from the internal index
without an out-of-band constraints file.

## Progress

- [x] (2026-08-01) Plan authored from the v0.2.5 release-scoped critique.
- [x] (2026-08-01) Cross-plan architecture review completed; release manifest,
      lock, receipt, fork-distribution, and changelog decisions revised.
- [x] (2026-08-01) Follow-up review made the manifest the milestone-1 checker
      authority rather than comparing repeated metadata to itself.
- [x] (2026-08-01) Milestone 1: CI consistency and derived-data drift checks.
- [x] (2026-08-01) Milestone 2: one release manifest and isolated deterministic
      metadata expansion. All 24 sdists and wheels built from the staged tree;
      wheel inspection found 106 exact first-party pins at version 0.3.0.
- [x] Milestone 3: generated dependency locks and indexed maintained forks.
- [x] Milestone 4 implementation: captured image receipts and a protected
      candidate/final publication flow. The protected environment was approved
      and the v0.3.1 final release completed through the LAN runner.
- [x] (2026-08-01) Added the milestone-1 authority and primary drift gate:
      `release/manifest.toml` is the authored version; `posttrain-release
      check` verifies 25 package versions, 109 internal pins, the catalog's
      `uv.lock` digest, and the published runtime-image manifest. CI runs it.
- [x] (2026-08-01) Replaced in-place TOML rewriting with release-neutral source
      templates and `posttrain-release stage DESTINATION`. `prepare` changes
      only the manifest; staging renders static metadata in an isolated copy.
- [x] (2026-08-01) Replaced six copied training lock digests and source
      revisions with `trl-fork@current` plus one generated `locks.toml` record.
- [x] (2026-08-05) Documented the accepted release control plane, isolated LAN
      runner boundary, one-build receipt, dev qualification, stable promotion,
      tag-last finalization, and retry semantics.
- [x] (2026-08-05) Added the immutable `rcN` candidate loop, automated
      changed-image qualification, final-version canary, and rule that a failed
      candidate never consumes or mutates the target stable version.
- [x] (2026-08-05) Provisioned and live-qualified the dedicated LAN runner in
      `../ai-infra` with repository-scoped `lan-release` labels, protected
      index/registry credentials, hash-locked dstack/devpi clients, NVMe-backed
      rootless BuildKit, bounded cache GC, and private-CA readback.
- [x] (2026-08-05) Replaced `.github/workflows/release.yml` with the protected
      manual final workflow and added `.github/workflows/release-candidate.yml`.
      Both build receipts on the LAN runner, keep GPU qualification in dstack,
      promote only after readback, and tag only after stable publication. GitHub
      environment rules and the first merged-branch dispatch remain the
      activation gate.
- [x] (2026-08-05) Built the authored `0.3.1` staged tree locally as a release
      rehearsal: all 24 workspace packages produced 48 wheel/sdist artifacts,
      and the receipt verified every artifact hash. The final workflow now also
      proves an index-only clean consumer install, retains final receipt/cache
      evidence, and refuses a pre-existing tag that names another commit.
- [x] (2026-08-05) Ran `posttrain --project-root apps/lab runtime images
      verify` against `registry.lan/carbonteq`; all six committed runtime
      variants returned matching lock digests and the accepted framework
      revision. The protected workflows retain this readback as JSON evidence.
- [x] (2026-08-05) The repository-wide validation ladder is green: Ruff,
      Pyright, import-boundary checks, and `1036 passed, 18 skipped` tests.
- [x] (2026-08-05) Audited both release execution planes. `ai-release` is
      healthy on NVMe with rootless BuildKit and preinstalled release clients.
      dstack's generic 24-GiB selector was tightened to the explicitly
      declared idle 96-GiB RTX PRO worker after host telemetry showed the
      local RTX 4090 was already occupied by an unrelated vLLM process.
- [x] (2026-08-05) Repaired the dstack fleet's stale Pop!_OS endpoint without
      changing worker identity: the fleet now uses explicit `internal_ip`
      values, the control plane has a reversible `/etc/hosts` routing
      override, and the local admin scripts use the same endpoint map. The
      new fleet is active with two healthy idle unsliced workers, and the
      reconnect and worker-component qualification receipts pass.
- [x] (2026-08-05) Changed real-checkout release staging to use `git archive
      HEAD` instead of copying the persistent runner worktree. The synthetic
      fixture fallback remains available for unit tests; a regression test
      proves ignored runner state and virtualenv files cannot enter a staged
      release tree.
- [x] (2026-08-05) Qualified the repaired release workflow through the
      packaged consumer environment: quality run `30973419701` passed after
      validating the exact framework and lab wheels, and candidate runs now
      reconcile and clean up their canonical run IDs. Earlier candidates
      exposed and fixed credential-scope, target-selection, wheelhouse, and
      lab-host dependency defects.
- [x] (2026-08-05) Diagnosed the first real job-image build failure on `ai-release`:
      rootless BuildKit reached the Dockerfile but `runc` could not mount
      `/proc` for the default process sandbox. The dedicated runner now sets
      `noProcessSandbox = true` (ai-infra commit `14c66f8`), retains bounded
      NVMe-backed cache GC, and has been reconfigured and requalified.
- [x] (2026-08-05) Candidate `30974205956` completed the packed dstack
      submission and BuildKit job-image build successfully as `0.3.1rc10`.
      The run exposed two final workflow issues: qualification returned after
      submission instead of waiting for terminal evidence, and GitHub's
      artifact action ignored the hidden `.release` directory. Both workflows
      now wait for the exact run to finish and retain hidden evidence with a
      fail-closed artifact check.
- [x] (2026-08-05) Final run `30975505448` passed source validation, built and
      published the final `0.3.1` distributions to `carbonteq/dev`, and passed
      the packed dstack canary, but stopped before stable promotion because
      the provisioned devpi client lacked the runner's private CA bundle.
- [x] (2026-08-05) Made stable promotion retry-safe with the installed CA and
      verified that the first final run's development-index artifacts matched
      its retained receipt byte-for-byte. A later retry correctly refused to
      rebuild `0.3.1` from a newer workflow commit when the packaged release
      test sdist differed.
- [x] (2026-08-05) Added the explicit `resume_from_run_id` final-workflow
      path. It restores the retained receipt/wheelhouse, derives and validates
      the original source revision, reruns consumer/OCI/dstack qualification,
      and tags the receipt source without overwriting immutable dev bytes.
- [x] (2026-08-05) Resume run `30976912055` completed successfully: stable
      readback verified all 24 coordinated packages and 48 artifacts, final
      evidence was retained, and `v0.3.1` was created last at the receipt
      source commit `271bd685c5c598616e127cf6d39c0228a176d9a0`.
- [x] (2026-08-05) Runner/release hardening now polls candidate quality for the
      exact source SHA, validates that resume evidence comes from a successful
      final-release run, initializes the evidence directory before any failure
      point, and reuses the LAN runner's persistent UV cache.

## Surprises & Discoveries

- Observation: a tag-derived version conflicts with the existing safe release
  order in which artifacts are built and verified before the release tag is
  created.
  Evidence: the v0.2.5 process qualified immutable image digests before tagging;
  a VCS-only version would require the tag in order to build final metadata.
- Observation: uv can validate publishable metadata without workspace sources,
  so a custom in-repo PEP 517 metadata plugin is unnecessary for the target
  invariant.
  Evidence: the installed `uv build --help` exposes both `--all-packages` and
  `--no-sources`; the release plan validates sdists and wheels through that
  path.
- Observation: regex expansion over every TOML string changed the coverage
  source `posttrain` into `posttrain==0.3.0` and made release bumps rewrite 109
  dependency strings.
  Evidence: the initial 0.3.0 candidate diff and the malformed coverage source
  exposed both semantic mutation and review noise before publication.
- Observation: the v0.3.0 tag workflow failed before building because a
  GitHub-hosted runner could not resolve the private `pypi.lan` dependency
  source.
  Evidence: GitHub Actions run `30686603520` failed during `uv sync` while
  fetching `carbonteq-trackio`; no release assets were produced.
- Observation: Posttrain currently has no registered self-hosted GitHub runner.
  Evidence: the repository runner API returned `total_count: 0` on 2026-08-05.
- Observation: provisioning is a security-boundary change rather than ordinary
  VM setup because the persistent worker executes repository code while holding
  credentials that can publish irreversible package versions.
  Evidence: implementation review required explicit approval of that exact
  capability before ai-infra provisioning could proceed; partial scaffolding
  was removed and no runner or VM was created.
- Observation: the existing documentation prescribed both direct stable upload
  and development-index promotion, which are materially different safety
  models.
  Evidence: `docs/publishing.md` uploaded the merged build directly to
  `carbonteq/stable`, while `../ai-infra/docs/operations/python-index.md`
  requires development upload followed by `devpi push` promotion.
- Observation: a single final-version candidate does not provide a repair loop.
  Evidence: previous final tags made a failed publication identity immutable and
  forced a new framework version; PEP 440 RC versions preserve the target final
  version while keeping every attempted artifact set traceable.
- Observation: the repository already defines Bake smoke targets for all six
  runtime variants, actual-job runtime qualification, dstack smoke launchers,
  and remote GPU evidence requirements, but `images publish` does not compose
  them into one release gate.
  Evidence: `docker-bake.hcl` owns the `smoke` group while
  `apps/release/src/posttrain_release/publish.py` invokes published targets and
  registry verification without running the real-job matrix.
- Observation: dstack scheduler idleness is not host idleness. The local
  worker was reported `idle`, but read-only `nvidia-smi` showed a vLLM engine
  using about 20.7 GiB of its 24 GiB GPU at 100% utilization and an active
  Posttrain container. The remote RTX PRO worker was at 0% utilization and
  remained reachable and healthy.
  Consequence: the release canary must select and verify a known worker rather
  than relying on a memory-only target that may land on a manually occupied
  host.
- Observation: `pop-os.lan` had resolved to a stale address while the worker's
  reachable wired address was `192.168.30.116`. The control plane and local
  qualification tools now use explicit routed endpoints while retaining the
  stable worker hostname in dstack identity and evidence. The fleet was
  recreated only after verifying that no run was active; its new identity and
  both instance identities are recorded in the refreshed enrollment receipt.
- Observation: the candidate workflow's build failure was caused by the
  staging boundary, not by Python, uv, package metadata, or runner capacity.
  The worktree copy included persistent generated state after validation; the
  staged tree reached roughly 7.7 GB and `uv build` failed with only
  `maximum recursion depth exceeded`. Building the same RC tree from committed
  source under the real `github-runner` user succeeded. Production staging now
  archives `HEAD`, making the source boundary deterministic and excluding
  state, virtualenvs, frontend dependencies, and other ignored files.
- Observation: the first candidate that reached actual OCI job-image creation
  failed below the Dockerfile, not in registry resolution or GPU capacity.
  The runner journal recorded `runc run failed` while mounting `proc` with
  `operation not permitted`; the BuildKit worker was rootless and using its
  default process sandbox. Setting `noProcessSandbox = true` in the isolated
  worker is the smallest compatible fix. The runner's post-fix configure run
  reports 234 GiB free on `/dev/vda1`, BuildKit active, and private index and
  registry TLS checks passing.
- Observation: a green candidate can still be an incomplete release gate if
  `job run` only submits to dstack. Candidate `30974205956` printed
  `submitted`/`provisioning` during immediate reconcile and cleanup, proving
  that submission is not terminal qualification. The workflows now invoke
  `run wait` with a bounded one-hour deadline before reconciliation and
  evidence-gated cleanup.
- Observation: GitHub artifact upload excludes hidden paths unless explicitly
  enabled. The candidate generated its receipt and cache evidence, but the
  upload step reported no files because all paths were under `.release`.
  Evidence retention is now explicit with `include-hidden-files: true` and
  `if-no-files-found: error` in both protected workflows.
- Observation: the merge-triggered quality run can still be in progress when
  a manually dispatched final workflow starts. An immediate `conclusion ==
  success` check created a false release failure before any build or publish.
  The final workflow now polls the exact merged-SHA push run with a bounded
  timeout and fails fast on terminal failure.
- Observation: final distribution bytes are sensitive to later source changes
  even when those changes only improve release workflow tests. After `0.3.1`
  was already present in `carbonteq/dev`, rebuilding from a newer workflow
  commit changed `posttrain_release-0.3.1.tar.gz`; the immutable-index check
  correctly rejected it. Final retries must restore the retained receipt and
  wheelhouse rather than rebuild.
- Observation: the system CA store was sufficient for `uv` and `curl`, but the
  hash-locked devpi client did not inherit it automatically. Stable promotion
  now passes `REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt` while
  retaining certificate verification.
- Observation: the workstation currently has no active Posttrain job to stop;
  only the controller, Observatory servers, BuildKit, and unrelated services
  are running. The admission snapshot still contains historical submitted and
  terminal-pending-evidence entries, so those must not be mistaken for live
  local processes or deleted as part of a release-runner change.
- Observation: the candidate workflow checked only the latest quality run and
  could race a still-running exact-SHA push. The final workflow's bounded
  polling pattern is now shared by candidate qualification, so a transient
  dispatch timing issue cannot consume a candidate attempt.
- Observation: resume accepted an arbitrary workflow run ID as long as it
  contained a plausible receipt. It now requires a completed successful
  `Publish release` workflow-dispatch run before downloading retained bytes.
- Observation: the stable-promotion step always attempted every devpi push,
  even when a retry followed a fully completed stable promotion. The final
  workflow now performs an exact receipt readback first and skips that side
  effect when stable already contains the accepted bytes.

## Decision Log

- Decision: milestone 1 lands before any versioning change.
  Rationale: drift checks are pure additions that immediately catch the
  missed-edit failure mode while the invasive migration is validated; they
  also become the acceptance harness for milestones 2–3.
  Date/Author: 2026-08-01 / plan author.
- Decision: milestone 1 introduces `release/manifest.toml` immediately and the
  checker compares every generated occurrence to it.
  Rationale: agreement among 27 repeated values can still agree on the wrong
  value and does not establish which value is authored. The invariant is
  exactly one authored version in the manifest; all other occurrences are
  generated projections.
  Date/Author: 2026-08-01 / architecture review follow-up.
- Decision: use `release/manifest.toml` as the single release version and
  generate static package metadata only in an isolated release tree; the tag
  verifies the qualified commit. Source projects use version `0.0.0` and bare
  workspace dependencies, while staged sdists and wheels contain the manifest
  version and exact sibling pins.
  Rationale: this keeps build-before-tag and standards-readable artifacts while
  removing per-release source churn and preventing broad regex mutation.
  Date/Author: 2026-08-01 / implementation correction (supersedes committed
  static expansion into every source pyproject).
- Decision: deterministic dependency locks are regenerated, while image
  digests are captured from immutable build receipts.
  Rationale: a dependency hash is reproducible from source inputs; a registry
  digest is an output and must never be reconstructed from a mutable tag.
  Date/Author: 2026-08-01 / architecture review.
- Decision: use curated changelog fragments or PR labels rather than requiring
  conventional syntax on every development commit.
  Rationale: release automation needs structured release intent, not a new
  constraint on every intermediate commit or merge strategy.
  Date/Author: 2026-08-01 / architecture review.
- Decision: GitHub is the release control plane and an isolated LAN-connected
  self-hosted runner is the execution plane.
  Rationale: source approval and evidence belong in GitHub, while build and
  publication require private DNS, CA trust, and registry access. The runner
  polls GitHub over outbound HTTPS and therefore needs no public IP.
  Date/Author: 2026-08-05 / user and implementation review.
- Decision: build one immutable distribution set for each release candidate,
  qualify that candidate through `carbonteq/dev`, then build the final-version
  distribution set once after merge and promote those exact final bytes from
  `carbonteq/dev` to non-volatile `carbonteq/stable`; create the tag last.
  Rationale: a PEP 440 release candidate cannot be renamed into a final release,
  while rebuilding or uploading the final version independently after its
  qualification would break byte-level provenance. Tagging before final
  qualification represents an unaccepted build as a release.
  Date/Author: 2026-08-05 / user and implementation review.
- Decision: do not run the release worker on `ai-control` and do not use GHCR
  or public PyPI for the current Posttrain release path.
  Rationale: repository code should not execute beside authoritative services,
  and current consumers already resolve from the private Python and OCI
  registries.
  Date/Author: 2026-08-05 / implementation review.
- Decision: keep the authored target version in `release/manifest.toml` and
  derive immutable `X.Y.ZrcN` versions only in isolated candidate stages.
  Rationale: candidate repair must not rewrite the target version or overwrite
  prior files. RC metadata cannot be renamed into final metadata, so each RC
  and the final version independently satisfy build-once.
  Date/Author: 2026-08-05 / user and implementation review.
- Decision: automate OCI publishing on the protected runner with rootless
  BuildKit and qualify changed digests through build smoke, registry readback,
  cold pull, real packed jobs, Trackio evidence, and Observatory readback.
  Rationale: publication is recurring and must not depend on one operator's
  workstation; a Docker socket would unnecessarily grant root-equivalent host
  control, while a successful push alone does not prove job execution.
  Date/Author: 2026-08-05 / user and implementation review.
- Decision: release qualification uses the explicit
  `targets/carbonteq-rtx-pro-6000-96gb` target and a sanitized dstack capacity
  receipt, instead of `targets/carbonteq-cuda-24gb-plus`.
  Rationale: the transformation work package already declares the RTX PRO
  target, and dstack's `idle` state does not account for manually launched
  GPU processes on the 24-GiB workstation. This avoids making the release
  canary compete with unrelated local research work while preserving the
  generic target for ordinary jobs.
  Date/Author: 2026-08-05 / runner audit.
- Decision: keep worker hostnames as logical dstack identities and maintain a
  single explicit SSH endpoint map for control-plane and operator tooling.
  Rationale: `.lan` DNS can be stale on one network segment; silently changing
  the hostname would make receipts and placement policy ambiguous, while
  requiring every operator to repair workstation DNS makes a healthy fleet
  appear broken. The endpoint map is bounded to the declared two workers and
  fails closed for unknown names.
  Date/Author: 2026-08-05 / infrastructure qualification.
- Decision: final retries after development-index publication resume from a
  retained workflow run rather than rebuilding the target version.
  Rationale: package source archives can include release tests and therefore
  change when workflow-only commits land; one version must never name two
  different byte sets. The `resume_from_run_id` input derives the source SHA
  from the immutable receipt, verifies ancestry, and keeps the tag aligned to
  those bytes.
  Date/Author: 2026-08-05 / release qualification.
- Decision: use a persistent runner-owned UV cache when `UV_CACHE_DIR` is
  configured, while retaining a release-local fallback for developer runs.
  Rationale: BuildKit layer retention and Python build metadata are separate
  caches; discarding the latter on every protected run adds avoidable index and
  build latency without changing the receipt's byte-level authority.
  Date/Author: 2026-08-05 / runner audit.
- Decision: candidate quality and final resume provenance are fail-closed
  workflow guards, not operator instructions.
  Rationale: a release workflow should wait for the exact source evidence and
  reject artifacts from the wrong workflow before it reaches private indexes,
  rather than relying on a maintainer to inspect run IDs manually.
  Date/Author: 2026-08-05 / runner audit.
- Decision: stable promotion begins with a byte-level receipt check and only
  invokes devpi promotion when stable is incomplete or unreadable.
  Rationale: retries after a successful promotion must not republish immutable
  files; the readback is the safe idempotency barrier and still leaves the
  existing final verification in place after a partial promotion.
  Date/Author: 2026-08-05 / release hardening.

## Outcomes & Retrospective

- Planning review outcome: release inputs that can be regenerated are separated
  from external build outputs that must be captured and verified. The tag is a
  final assertion, not a build prerequisite. The release execution boundary is
  now documented in `docs/architecture/lan-release-runner.md` and implemented
  on the release branch.
- Candidate-lifecycle outcome: release documentation now reserves final tags
  and stable publication for accepted artifacts. RCs remain immutable dev
  evidence, and OCI qualification is defined across build, registry, and real
  job layers. The first workflow intentionally gates one packed transformation
  canary; a complete changed-kind matrix remains open work.
- Runner-audit outcome: the release runner is suitable for publication; the
  dstack canary path is explicit and fail-closed for the selected worker. The
  full fleet is now also qualified: the active fleet has two healthy idle
  unsliced GPUs, reconnect preserves fleet and instance identities, and the
  installed worker component versions match the deployed receipt. The
  final protected workflow completed on the selected healthy worker; runner
  capacity and release-environment approval are no longer open gates for
  v0.3.1.
- Final-release outcome: GitHub Release [v0.3.1](https://github.com/carbonteq-ai/posttrain/releases/tag/v0.3.1)
  is published with the wheelhouse, receipt, and checksum assets. Stable
  index readback matched every receipt artifact, the annotated tag points to
  `271bd685c5c598616e127cf6d39c0228a176d9a0`, and no GHCR publication was
  introduced. The remaining follow-up is the broader changed-kind matrix,
  not a v0.3.1 publication defect.
- Hardening outcome: the v0.3.1 release remains immutable; these changes affect
  future candidate/final dispatches and do not rebuild or mutate its stable
  artifacts. The workstation audit found no job process to kill, and the
  runner's persistent cache is now explicitly treated as an optimization,
  never as qualification evidence.

## Context and Orientation

The repository is a uv workspace: a root `pyproject.toml` plus ~26 member
packages under `packages/*` and `apps/*`, each with its own `pyproject.toml`
naming a `posttrain-*` distribution. First-party cross-dependencies are
declared with exact pins (for example `packages/work/pyproject.toml` depends
on `posttrain-common==0.2.5`) while development resolution goes through
workspace sources. Derived values live in:

- `packages/catalog/src/posttrain/catalog/base/training.yaml` — training
  entries embed `backend_options.source_revision` (a fork commit) and
  `dependency_lock_sha256` (the hash of that fork's dependency lock). Three
  entries carry the same hash today, hand-updated in step.
- `packages/runtime-images/src/posttrain/runtime_images/published.toml` —
  the per-release manifest of published base/kind image digests, with
  validation in `manifest.py` alongside it.
- `release/github-constraints.txt` — pins for maintained forks, required at
  install time.
- `apps/lab/tests/test_work_packages.py` — asserts a hardcoded version
  string (changed in every bump commit).
- `apps/release/` — the framework-owner release tooling package; the natural
  home for the commands this plan adds.
- `CHANGELOG.md` — hand-written per release, good quality, keep the format.
- `.github/workflows/release.yml` — currently triggers on `v*` and runs on
  `ubuntu-latest`. That order and network placement are incompatible with the
  accepted tag-last, private-index release path and must be replaced.
- `docs/architecture/lan-release-runner.md` — owns the release trust boundary,
  state machine, network requirements, and failure semantics.
- `../ai-infra` — owns the dedicated runner VM, private CA, local machine
  configuration, registration lifecycle, and health checks. It must never
  commit registration or package-index credentials.

## Plan of Work

Milestone 1 adds `release/manifest.toml`, initialized to the currently released
version, and guardrails without otherwise changing how releases are made. Add
`posttrain-release check` in `apps/release` that reads the sole authored version
from that manifest and asserts every member `pyproject.toml` version and every
first-party `==` pin equals it; recomputes
each `dependency_lock_sha256` from its named fork lock and diffs against the
committed YAML; validates `published.toml` shape (reachability checks stay in
`posttrain runtime images verify`). Wire it into CI. Change the lab test to
read `importlib.metadata.version("posttrain")` instead of a literal, and add
a CI scan forbidding unmanaged version literals outside the allowed generated
files (`CHANGELOG.md`, `uv.lock`, and the member pyprojects). The checker has an
explicit allowlist of generated targets and fails if another authored version
source appears.

Milestone 2 builds deterministic expansion around that manifest.
`posttrain-release prepare X.Y.Z` updates only the manifest, while
`posttrain-release stage DESTINATION` copies the source tree and expands the
version and exact first-party `==` pins only there. Source pyprojects remain
release-neutral and `uv.lock` changes only for dependency changes. The staged
values remain standards-compliant static metadata in both sdists and wheels.
`posttrain-release check` renders and validates the same projection in memory.
A tag check asserts `vX.Y.Z`
matches the manifest when building a release, without requiring a tag for PR
qualification. Build both sdists and wheels with workspace sources disabled
and verify wheel metadata:

    uv run posttrain-release stage /tmp/posttrain-release
    uv build --directory /tmp/posttrain-release --all-packages --no-sources
    unzip -p dist/posttrain_work-*.whl '*/METADATA' | grep posttrain-
    # expected: Requires-Dist: posttrain-common==<manifest version>, etc.

Milestone 3 makes deterministic dependency locks generated. Catalog entries
replace raw hashes with a named lock such as `trl-fork@6e7739b8`; a generated
`packages/catalog/src/posttrain/catalog/base/locks.toml` owns the resolved
commit and hash. `posttrain-release lock-dependencies` regenerates that table
and the release-build constraints from pinned source inputs, and CI diffs the
result. In parallel, publish maintained TRL, Verifiers, and Trackio fork wheels
with immutable PEP 440 versions to the internal index and generate exact normal
dependencies on them. The constraints file remains a build input during the
migration but ceases to be a consumer prerequisite.

Milestone 4 adds a curated release-PR and two protected publication workflows.
Contributors add a small
changelog fragment or label a PR; `posttrain-release prepare` renders the
existing CHANGELOG format and updates the release PR. Changed runtime images
are built and pushed once from the PR commit. The build pipeline emits signed
receipts containing the immutable digest, source commit, image definition, and
builder identity; `posttrain-release record-images RECEIPTS...` verifies and
writes `published.toml`.

Before merge, a maintainer dispatches **Prepare candidate** through a protected
GitHub environment for an internal release branch. The runner derives the next
unused RC from the target manifest, stages and builds that candidate once,
publishes it only to `carbonteq/dev`, and runs index-only consumer, packing, and
dstack canary gates. When image inputs changed, it publishes with rootless
BuildKit, verifies registry readback, and records the generated image receipt.
The first protected workflow executes one bounded packed transformation canary
through dstack; a changed-kind matrix is intentionally a follow-up gate rather
than an unverified promise. Accepted image receipts regenerate `published.toml`;
failed digests remain unreferenced. A fix produces the next RC.

After a candidate passes and the generated image records merge, a maintainer
dispatches **Publish release** for the exact merged default-branch commit. It
stages and builds final distributions once, uploads them to `carbonteq/dev`,
runs an index-only install and final dstack canary, verifies accepted OCI
receipts without rebuilding, promotes the exact final files server-side to
`carbonteq/stable`, and verifies stable readback. The final tag and GitHub
Release are created only afterward. The runner is an isolated VM managed by
`../ai-infra`, has no public IP or inbound GitHub route, uses no host Docker
socket, and never runs automatic PR workflows.

## Concrete Steps

Work from the repository root.

    uv run posttrain-release check
    # milestone 1 expected output:
    #   authored version: release/manifest.toml = 0.2.5
    #   generated versions: OK (27 files equal manifest)
    #   authored version sources: OK (exactly 1)
    #   dependency locks: OK (3 entries match trl-fork@6e7739b8)
    #   published images: OK (manifest shape valid)

    uv build --all-packages --no-sources

After milestone 2, deliberately create the failure the checker exists for:
edit one pin back to the previous version in a scratch branch and confirm CI
fails with a message naming the file. After milestone 3:

    uv run posttrain-release lock-dependencies
    git diff --exit-code   # clean tree = committed tables match regeneration

Before implementing milestone 4, verify the runner inventory:

    gh api repos/carbonteq-ai/posttrain/actions/runners
    # expected after provisioning: one online runner with the lan-release label

The release canary must also verify its actual dstack placement before
submitting a job:

    scripts/release/verify-dstack-capacity .release/dstack-capacity.json
    # expected: carbonteq-ai-workstation.lan, healthy/reachable/idle,
    #          one NVIDIA GPU with at least 90 GiB, one unsliced block

This is intentionally narrower than the infrastructure repository's full
two-worker reconnect gate. The broader gate is now independently green; the
receipt is kept separate so release publication still proves the exact worker
selected for the canary rather than relying on aggregate fleet idleness.

Dispatch Prepare candidate from an internal release branch. Its summary must
name the target version, allocated RC, source commit, distribution receipt,
image receipts, changed/reused image variants, dstack run IDs, Trackio run IDs,
and Observatory readback. After that candidate and its generated records merge,
dispatch Publish release from the default branch. Its summary must name one
source commit, one final receipt, successful dev canary, successful stable
readback, and a final tag pointing to that commit.

## Validation and Acceptance

- Milestone 1: CI reports exactly one authored version in
  `release/manifest.toml`; changing either the manifest or one generated
  package value without regenerating fails and names both expected and observed
  values. Adding a second non-generated version source also fails. The release
  process itself is otherwise unchanged.
- Milestone 2: `posttrain-release check` reports that every source template is
  release-neutral and its staged package version and internal pins match
  `release/manifest.toml`. Sdists build wheels with `--no-sources`, and their
  metadata carries the manifest version and exact internal pins.
- Milestone 3: `posttrain-release lock-dependencies && git diff --exit-code` is
  clean on main; hand-editing a hash fails CI; a clean consumer installs all
  maintained forks from the internal index without a constraints file.
- Milestone 4: the release PR records immutable image receipts; the protected
  LAN candidate workflow allocates immutable RCs and proves dev-channel plus
  one packed OCI qualification canary; the final workflow builds one final
  distribution set from the merged commit; dev canary and stable readback match
  one final receipt; and the created final tag equals the manifest version and
  points at the verified commit. Neither workflow can run on `pull_request`
  events or a runner without `lan-release`.
- Consumer-visible invariant throughout: `uv pip install posttrain==X.Y.Z`
  from the internal index resolves the identical dependency set before and
  after each milestone (compare `uv pip freeze` output).

## Idempotence and Recovery

Every step is additive and reversible: milestone 1 adds checks only;
milestone 2 leaves source templates stable and can stage any previous manifest
version without modifying them; milestone 3 keeps generated tables in-tree so a regeneration
bug is caught by `git diff`, and the previous committed table remains the
working fallback. Do not remove `release/github-constraints.txt` from release
builds until indexed fork wheels pass clean external-consumer installation.
`record-images` is append/replace by exact release key and refuses a receipt
whose source commit or definition digest differs, so retrying cannot silently
record another image. Distribution publication follows the same rule. A retry
reuses files whose hashes match the retained receipt and blocks on any
mismatch; a corrected candidate allocates the next RC rather than overwriting.
Qualification failure creates no final tag. If promotion
succeeds but GitHub finalization fails, retry only tag and Release creation
against the retained receipt; never rebuild.

## Artifacts and Notes

Keep here: the first passing checker transcript, the sdist-to-wheel metadata
check, an external index-only install transcript, the captured image-receipt
summary, and the first fully generated release-PR diff summary.

## Interfaces and Dependencies

Extend the existing `apps/release` distribution with:

    posttrain-release prepare VERSION
    posttrain-release stage DESTINATION
    posttrain-release check
    posttrain-release lock-dependencies
    posttrain-release record-images RECEIPT...
    posttrain-release candidate next
    posttrain-release distributions build --version VERSION --receipt PATH
    posttrain-release distributions verify --receipt PATH
    posttrain-release distributions publish-dev --receipt PATH
    posttrain-release distributions promote --receipt PATH
    posttrain-release distributions verify-stable --receipt PATH
    posttrain-release images qualify --receipt-root PATH

Milestone 4 additionally introduces commands or equivalent internal interfaces
for building a distribution receipt, publishing it to the development index,
qualifying it, promoting it, and verifying stable readback. These interfaces
resolve the named Python-index service through Posttrain machine configuration;
workflow inputs never carry index URLs, usernames, passwords, artifact hashes,
or a hand-authored candidate version. `candidate next` derives `rcN` from the
target manifest plus immutable dev-index/readback evidence and refuses reuse.
`images qualify` composes static validators, Bake smoke targets, registry
readback, cold pull, changed-kind dstack jobs, Trackio evidence, and Observatory
readback; it never treats build success alone as acceptance.

The initial manifest is deliberately small:

    schema_version = 1
    version = "0.2.5"

`release/manifest.toml` is the only human-facing version source. `stage`
provides the temporary output root used to qualify generated member metadata
without mutating the checkout. `uv.lock` and dependency lock tables change only
when dependency inputs change; constraints, CHANGELOG sections, and
`published.toml` remain committed and reviewable. No custom PEP 517 plugin or
import-time metadata rewrite is required.

## Revision Notes

- 2026-08-01: Architecture review replaced tag-derived dynamic metadata and a
  self-hosted build hook with one release manifest plus deterministic static
  expansion, separated reproducible dependency locks from captured image
  receipts, moved maintained fork wheels onto the internal index, and selected
  curated release fragments over mandatory conventional commits.
- 2026-08-01: Follow-up review moved creation of `release/manifest.toml` into
  milestone 1 and made the checker enforce exactly one authored version plus
  equality of every generated package version and internal pin to that
  manifest.
- 2026-08-01: Implementation review replaced committed static expansion with
  isolated release staging after the original regex mutated unrelated TOML and
  recreated the many-file release diff the plan was meant to remove.
- 2026-08-05: Release review replaced the tag-triggered GitHub-hosted build and
  direct stable upload with a protected, manually dispatched workflow on an
  isolated LAN runner. It records one build receipt, qualifies through
  `carbonteq/dev`, promotes unchanged artifacts to `carbonteq/stable`, verifies
  readback, and creates the tag last. The runner requires outbound access only;
  GHCR and public PyPI are outside the current release path.
- 2026-08-05: Candidate-lifecycle review split release preparation from final
  publication. Prepare candidate derives immutable RCs, qualifies dev packages
  and changed OCI digests, and permits repairs without consuming the final
  version. Publish release builds final metadata from the merged commit,
  qualifies it on dev, promotes unchanged bytes, and tags last. OCI automation
  uses rootless BuildKit and requires real packed-job evidence.
- 2026-08-05: Fleet qualification repaired stale worker routing by preserving
  logical hostnames and adding explicit routed endpoints to the control-plane
  and operator paths. Reconnect and component receipts now pass for the new
  active fleet; the release plan no longer treats fleet readiness as an open
  DNS-owner dependency.
- 2026-08-05: Completed v0.3.1 publication through the protected LAN workflow.
  Added a bounded merged-CI wait, private-CA-aware devpi promotion, and an
  evidence-bound resume path after the first promotion interruption. The
  final receipt, stable readback, dstack qualification, annotated tag, and
  GitHub Release are now recorded above.
