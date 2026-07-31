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
- [ ] Milestone 1: CI consistency and derived-data drift checks (no process
      change).
- [ ] Milestone 2: one release manifest and deterministic metadata expansion.
- [ ] Milestone 3: generated dependency locks and indexed maintained forks.
- [ ] Milestone 4: captured image receipts and a curated release-PR flow.

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
  generate static package metadata; the tag verifies the qualified commit.
  Rationale: a tag-derived build requires tagging before final qualification
  and complicates source-distribution and archive builds. Generated static
  metadata supports build-before-tag, remains standards-readable, and makes
  drift detectable by regeneration.
  Date/Author: 2026-08-01 / architecture review (supersedes the original
  dynamic-version decision).
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

## Outcomes & Retrospective

- Planning review outcome: release inputs that can be regenerated are separated
  from external build outputs that must be captured and verified. The tag is a
  final assertion, not a build prerequisite. Implementation outcomes remain
  pending.

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

Milestone 2 builds deterministic expansion around that manifest with a
`posttrain-release prepare X.Y.Z` command. The command updates the manifest,
expands the version and exact first-party `==` pins into every member
`pyproject.toml`, refreshes `uv.lock`, and changes tests to read installed
metadata. The repeated values remain in standards-compliant static metadata,
but they are generated and never edited directly. `posttrain-release check`
regenerates into a temporary tree and diffs it. A tag check asserts `vX.Y.Z`
matches the manifest when building a release, without requiring a tag for PR
qualification. Build both sdists and wheels with workspace sources disabled
and verify wheel metadata:

    uv build --all-packages --no-sources
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

Milestone 4 adds a curated release-PR workflow. Contributors add a small
changelog fragment or label a PR; `posttrain-release prepare` renders the
existing CHANGELOG format and updates the release PR. Changed runtime images
are built and pushed once from the PR commit. The build pipeline emits signed
receipts containing the immutable digest, source commit, image definition, and
builder identity; `posttrain-release record-images RECEIPTS...` verifies and
writes `published.toml`. After merge, CI verifies the same source metadata and
registry digests without rebuilding or repushing, builds/publishes Python
artifacts, and creates `vX.Y.Z` only if it matches `manifest.toml`. A failed
tag or GitHub Release step is retried against the same verified commit and
artifacts.

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

## Validation and Acceptance

- Milestone 1: CI reports exactly one authored version in
  `release/manifest.toml`; changing either the manifest or one generated
  package value without regenerating fails and names both expected and observed
  values. Adding a second non-generated version source also fails. The release
  process itself is otherwise unchanged.
- Milestone 2: `posttrain-release check` reports that every generated package
  version and internal pin matches `release/manifest.toml`; changing one pin
  names that file. Sdists build wheels with `--no-sources`, and their metadata
  carries the manifest version and exact internal pins.
- Milestone 3: `posttrain-release lock-dependencies && git diff --exit-code` is
  clean on main; hand-editing a hash fails CI; a clean consumer installs all
  maintained forks from the internal index without a constraints file.
- Milestone 4: the release PR records immutable image receipts, merge-time CI
  verifies those exact digests without rebuilding, and the created tag equals
  the manifest version and points at the verified commit.
- Consumer-visible invariant throughout: `uv pip install posttrain==X.Y.Z`
  from the internal index resolves the identical dependency set before and
  after each milestone (compare `uv pip freeze` output).

## Idempotence and Recovery

Every step is additive and reversible: milestone 1 adds checks only;
milestone 2 keeps static package metadata and can regenerate the previous
manifest version; milestone 3 keeps generated tables in-tree so a regeneration
bug is caught by `git diff`, and the previous committed table remains the
working fallback. Do not remove `release/github-constraints.txt` from release
builds until indexed fork wheels pass clean external-consumer installation.
`record-images` is append/replace by exact release key and refuses a receipt
whose source commit or definition digest differs, so retrying cannot silently
record another image.

## Artifacts and Notes

Keep here: the first passing checker transcript, the sdist-to-wheel metadata
check, an external index-only install transcript, the captured image-receipt
summary, and the first fully generated release-PR diff summary.

## Interfaces and Dependencies

Extend the existing `apps/release` distribution with:

    posttrain-release prepare VERSION
    posttrain-release check
    posttrain-release lock-dependencies
    posttrain-release record-images RECEIPT...

The initial manifest is deliberately small:

    schema_version = 1
    version = "0.2.5"

`prepare` and `lock-dependencies` must support `--check` or a temporary output
root so CI proves regeneration without first mutating the checkout.
`release/manifest.toml` is the only human-facing version source. Generated
member `pyproject.toml` values, `uv.lock`, dependency lock tables, constraints,
CHANGELOG sections, and `published.toml` remain committed and reviewable. No
custom PEP 517 plugin or import-time metadata rewrite is required.

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
