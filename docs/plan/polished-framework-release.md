# Publish a polished post-training framework release

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

Maintain this document in accordance with `docs/templates/PLAN.md`. The
canonical product authority is `docs/post-training/README.md` and the six
documents it indexes. This plan builds on the portable project boundary recorded
in `docs/decisions/0013-portable-project-layout.md` and implemented through
`docs/plan/portable-project-layout-and-consumer.md`. It does not change the
frozen meanings of selections, projects, work packages, jobs, runs, artifacts,
or observation.

## Purpose / Big Picture

After this work, a CarbonTeq developer can create or open an ordinary repository,
install versioned framework packages from a Python package index, initialize and
inspect its `.posttrain` project, validate its catalog and work packages, run a
work package through a stable `posttrain` command, record evidence through a
supported tracking backend, and inspect that evidence through a versioned
Observatory deployment. The same documented commands work on a developer
machine, in CI, and on a remote GPU server without this monorepo being present.

The intended audience is CarbonTeq projects. Public PyPI and GHCR are
distribution mechanisms, not a commitment to a third-party plugin marketplace
or unrestricted backend support. Release quality is nevertheless product-grade:
the CLI is coherent, documentation is executable, package metadata is complete,
dependencies resolve as published, upgrades are described, and release
automation publishes immutable artifacts.

The visible proof is a release-candidate acceptance job. It installs built
wheels and the Observatory image exactly as a remote consumer would, initializes
or loads a fixture project, validates and executes a CPU-safe work package,
records it through a real local tracking backend, and reads the same run through
Observatory. A separate documented GPU release gate exercises one supported
training or evaluation path on a remote server.

## Progress

- [x] (2026-07-23T12:00Z) Confirmed the release intent: polished developer
  experience and documentation for CarbonTeq-managed projects, with public
  package registries allowed as transport.
- [x] (2026-07-23T12:00Z) Audited canonical package, API, work/evidence, and
  observation boundaries and confirmed that a CLI host can be added without
  changing primitive or run semantics.
- [x] (2026-07-23T12:06Z) Added the `posttrain` distribution and first stable
  CLI slice: `init`, `version`, `doctor`, `project show`, and catalog
  list/show/validate; proved initialization and diagnostics from an installed
  wheel outside the workspace.
- [ ] Extract generic work-package contracts, loading, validation, and execution
  from `apps/lab` into a reusable first-party boundary; add
  `posttrain work-package validate` and `posttrain work-package run`.
  (Completed: `posttrain-work` extraction, lab compatibility exports, installed
  composition-level validation; remaining: concrete job-definition preflight
  and execution through the primary CLI.)
- [ ] Make every published dependency resolve equivalently with workspace source
  overrides disabled, including the CarbonTeq TRL and Trackio forks and
  supported accelerator indexes. (Completed: immutable TRL, Verifiers, and
  AutomationBench Git revisions are present in built wheel metadata and a
  no-sources lock resolves through those forks; remaining: publish the
  AutomationBench and Trackio fork distributions and settle accelerator-index
  policy.)
- [x] (2026-07-23T18:00Z) Prepared maintained-fork registry artifacts:
  `carbonteq-trackio==0.31.5.post1` and
  `carbonteq-automation-bench==1.0.5.post1` build wheel and source
  distributions, install cleanly on Python 3.12, preserve their existing
  import/CLI names, and have tag-gated Trusted Publishing workflows. Framework
  dependency replacement remains gated on fork commit, push, and registry
  publication.
- [x] (2026-07-23T18:20Z) Committed and pushed the fork release candidates:
  Trackio at `c47bcc0e0a15030ec6f20cdc7d294a820ab617b2` and
  AutomationBench at `6e3c50209731c0b06c3bc6d3dbb30bc7fdf10a38`.
  Opened draft reviews at `carbonteq-ai/trackio#3` and
  `carbonteq-ai/AutomationBench#1`. Review/merge, PyPI project ownership,
  trusted-publisher configuration, and explicit release tags remain external
  publication gates.
- [ ] Publish the maintained fork distributions. AutomationBench PR 1 merged
  at `908db2abd4a868acc37ab0850474bff653bea25c`; Trackio PR 3 merged at
  `c5072198b3b1556d31ed96ffc246a03f65418ab8`. Both repositories have a `pypi`
  GitHub Actions environment and their tag-only release workflows are present
  on `main`. Remaining: configure PyPI pending publishers and create the two
  explicit `carbonteq-v*` release tags.
- [x] (2026-07-23T18:30Z) Created the public
  `carbonteq-ai/posttrain` source repository and adopted GitHub Releases as the
  initial team distribution channel. The framework now pins the merged Trackio
  and AutomationBench commits under their release distribution names, and a
  tag workflow builds a hashed wheelhouse after the full validation ladder.
- [x] (2026-07-23T17:05Z) Expanded the independent-consumer fixture to execute
  a deterministic CPU work package through `posttrain-work`, write
  evidence through a real local Trackio backend, and read it through
  Observatory from installed wheels outside the workspace.
- [ ] Write and test the installation, quickstart, project layout, catalog,
  work-package, remote server, tracking, Observatory, troubleshooting, support
  matrix, and upgrade documentation. (Completed: developer-facing root
  quickstart and framework overview, release/consumption guide, package graph,
  remote workflow, and explicit release gates; remaining: task-specific
  guides, troubleshooting, compatibility, upgrades, and CI-executed
  documentation examples.)
- [ ] Complete distribution metadata, licensing, changelog, compatibility
  policy, and release notes for all maintained artifacts.
- [ ] Add tag-driven PyPI Trusted Publishing and GHCR workflows with staged
  release-candidate validation, provenance, and immutable version/digest output.
- [ ] Install a release candidate into a separate CarbonTeq project and run the
  documented GPU release gate on a remote server.

## Surprises & Discoveries

- Observation: The framework already has a portable project and catalog
  boundary, but the general work-package types and runner remain under
  `apps/lab`, while the lab CLI exposes a fixed list of reference jobs.
  Evidence: `packages/catalog/src/posttrain/catalog/project.py` owns portable
  discovery; `apps/lab/src/posttrain_lab/work_packages/` owns composition and
  execution; `apps/lab/src/posttrain_lab/cli.py` defines fixed `job` choices.

- Observation: The current independent-consumer proof installs five local wheels
  and executes catalog resolution plus a CPU data partition, but it does not run
  a work package or prove tracking/Observatory readback.
  Evidence: `tests/consumer/test_wheel_project.py` and
  `tests/consumer/fixture/run.py`.

- Observation: Workspace source declarations can hide published dependency
  differences. `posttrain-train[trl]` declares `trl==1.8.0` in wheel metadata
  while local uv resolution replaces it with an immutable CarbonTeq Git fork;
  Trackio is a direct Git dependency in tracking and lab packages.
  Evidence: `packages/train/pyproject.toml`,
  `packages/tracking-trackio/pyproject.toml`, and `apps/lab/pyproject.toml`.

- Observation: Immutable direct references in wheel metadata preserve the
  selected fork, but uv requires a transitive URL dependency such as Trackio to
  be repeated as a top-level direct requirement. A `posttrain-tracking-trackio`
  wheel therefore does not yet provide a one-name installation.
  Evidence: the clean consumer install failed until it explicitly requested
  `trackio @ git+https://github.com/carbonteq-ai/trackio.git@9cf451c...`.

- Observation: With the direct Trackio requirement made explicit, a clean
  environment can install the built framework wheels, execute the fixture work
  package, persist three data metrics and a terminal succeeded outcome in local
  Trackio, and retrieve the same run through Observatory's generic fallback.
  Evidence: `uv run pytest -q tests/consumer/test_wheel_project.py -vv`
  reports one passed test.

- Observation: Renaming only the maintained distribution, rather than its
  import package, gives projects ordinary registry dependencies without
  changing `import trackio`, `import automationbench`, the `trackio` command,
  or the `auto-bench` command.
  Evidence: clean wheel installations report
  `carbonteq-trackio==0.31.5.post1` with `trackio.__version__` at the same
  version, and `carbonteq-automation-bench==1.0.5.post1` imports
  `automationbench`.

- Observation: Published-resolution fidelity is now explicit for the training
  and evaluation forks. Without workspace sources, TRL, Verifiers, and
  AutomationBench resolve to their immutable CarbonTeq revisions. Resolution
  stops at the expected unpublished `automationbench-v1` distribution rather
  than selecting an upstream TRL or Verifiers implementation.
  Evidence: built wheel `METADATA` and `uv lock --dry-run --no-sources`.

- Observation: A newly initialized project needs a valid but empty catalog
  overlay. The current catalog-layer schema rejects `files: []`, forcing an
  initializer either to invent a project selection or omit the overlay it just
  created.
  Evidence: `CatalogLayerManifestSchema.safe_unique_files` in
  `packages/catalog/src/posttrain/catalog/files.py`.

- Observation: Generic work-package behavior is mixed with lab-specific job
  definition factories. Contracts, YAML loading, catalog resolution, evidence
  snapshots, and execution are host-neutral; `definitions.py` imports
  lab-specific GRPO and distillation request wrappers.
  Evidence: `apps/lab/src/posttrain_lab/work_packages/contracts.py`,
  `runner.py`, `apps/lab/src/posttrain_lab/execution.py`, and
  `work_packages/definitions.py`.

- Observation: Adding a workspace distribution changes `uv.lock`, whose digest
  is deliberately embedded in packaged training bindings. The complete suite
  caught the stale digest immediately.
  Evidence: the first full run failed one catalog assertion; updating six
  `dependency_lock_sha256` values to the new `uv.lock` hash produced 267 passed
  and 5 credential-gated skips.

- Observation: A clean base installation intentionally omits large or
  credentialed backend dependencies such as Torch, TRL, Verifiers, and
  Transformers. Pyright otherwise reports those guarded optional imports as
  missing even though the base package graph and tests are valid.
  Evidence: the clean GitHub quality job passed tests and lint but reported
  only `reportMissingImports` diagnostics; optional Verifiers behavior passed
  in its dedicated extra-enabled job.

- Observation: The portable project manifest used `posttrain-platform`, while
  every tracked work package used `foundation-models`. The former is also the
  default Trackio/W&B storage namespace, but provider storage namespace and
  product project identity are separate concepts.
  Evidence: `.posttrain/project.toml`, all files under
  `.posttrain/work_packages/`, and the `--project` default in
  `apps/lab/src/posttrain_lab/cli.py`.

## Decision Log

- Decision: Optimize the release for CarbonTeq-managed projects while keeping
  the packages publicly installable.
  Rationale: Public registries simplify reproducible installation on remote
  servers; the supported use cases, compatibility matrix, and operational
  guarantees can remain deliberately first-party.
  Date/Author: 2026-07-23 / user and Codex.

- Decision: Publish a small `posttrain` distribution as the primary command-line
  and convenience entry point; retain component wheels and keep
  `posttrain-lab` as a reference composition host.
  Rationale: Developers need one stable command without making the lab's
  hard-coded example jobs the public framework contract. Component packages
  remain useful for narrow notebook and service consumers.
  Date/Author: 2026-07-23 / Codex.

- Decision: Implement project and catalog commands before moving work-package
  execution.
  Rationale: These commands depend only on the already portable catalog package,
  provide immediate installable behavior, and let the runner extraction happen
  as a separate boundary-preserving migration with compatibility re-exports.
  Date/Author: 2026-07-23 / Codex.

- Decision: Treat public documentation, CLI behavior, dependency fidelity, and
  upgrade behavior as release gates even though the first users are internal.
  Rationale: Internal scope narrows what is supported; it does not justify
  checkout-relative installs, undocumented commands, or packages whose metadata
  resolves different code on a remote server.
  Date/Author: 2026-07-23 / user and Codex.

- Decision: Distribute generic composition as `posttrain-work`, imported as
  `posttrain.work`, while keeping concrete job-definition factories and
  provider wiring in hosts.
  Rationale: The canonical API defines an optional thin runner, and other
  projects need the same recipe/work-package validation without importing the
  lab. The composition package can depend on capability contracts without
  creating cross-imports among train, eval, and serve.
  Date/Author: 2026-07-23 / Codex.

- Decision: Publish maintained fork distributions before claiming a polished
  one-command install. In particular, the Trackio fork needs a
  CarbonTeq-owned distribution identity while retaining the `trackio` import
  package; the framework adapter should then depend on that distribution
  instead of a transitive Git URL.
  Rationale: A direct Git pin is reproducible and suitable for an alpha gate,
  but uv intentionally requires consumers to repeat transitive URL
  dependencies. Publishing the fork removes that consumer-visible workaround
  and gives remote servers an ordinary immutable package artifact.
  Date/Author: 2026-07-23 / Codex.

- Decision: Use upstream-derived PEP 440 post-release versions for maintained
  fork distributions.
  Rationale: `0.31.5.post1` and `1.0.5.post1` retain visible upstream lineage
  while allowing CarbonTeq to publish additional compatibility releases before
  moving the upstream base. Distribution names carry ownership; import and CLI
  names remain compatible.
  Date/Author: 2026-07-23 / Codex.

- Decision: Use the public `carbonteq-ai/posttrain` repository and hashed GitHub
  Release wheelhouses as the initial team distribution channel; keep PyPI as a
  later transport improvement.
  Rationale: Every current consumer already has GitHub access, public Git
  commits work on remote servers without account-specific package-index setup,
  and a tagged wheelhouse preserves built-artifact installation and hashes
  without weakening the eventual registry release contract.
  Date/Author: 2026-07-23 / user and Codex.

- Decision: Keep explicit legacy root-layout support and the historical root
  `artifacts/` tree unchanged during the release work.
  Rationale: ADR 0013 reserves the legacy constructor for an additive
  compatibility window and requires a separate retention/classification
  decision for historical evidence. New projects and installed consumers
  already use packaged base assets plus `.posttrain/` source and state paths.
  Date/Author: 2026-07-23 / Codex.

## Outcomes & Retrospective

The first CLI and independent-consumer milestones are complete. Built wheels
initialize and diagnose an external project, compose its overlay with the
packaged base, resolve and execute a CPU-safe work package, record a canonical
terminal run through real local Trackio storage, and retrieve that same run
through Observatory. The live repository reports 41 base selections. The
latest full suite reports 270 passed and 5 credential-gated W&B skips; Pyright,
Ruff, all eight import contracts, both lock checks, the external consumer, and
`git diff --check` pass. The configured formatter was applied across the
publishable Python tree, and all 203 checked files now pass the formatting
gate. A clean installation from the generated GitHub wheelhouse succeeds when
using its bundled immutable fork constraints.

## Context and Orientation

This repository is a Python 3.12 uv workspace. `packages/common` owns shared
identities and protocols. `packages/catalog` owns the versioned framework base
catalog, `.posttrain/project.toml` loading, project discovery, and overlay
composition. `packages/data`, `packages/serve`, `packages/eval`, and
`packages/train` own reusable operations. `packages/tracking` plus its Trackio
and W&B adapter distributions own evidence writes and reads.
`apps/observatory` is the read-only evidence product. `apps/lab` is the existing
reference host and contains concrete jobs and work-package execution.

A distribution is the object installed from PyPI, such as
`posttrain-catalog`; an import package is the Python module it provides, such as
`posttrain.catalog`. The new distribution is named `posttrain` and provides the
`posttrain` console command from an implementation module named
`posttrain_cli`. Keeping the implementation outside the shared `posttrain`
namespace avoids implying that a command host belongs to a capability package.

The CLI is a host. It may import first-party capability and composition
packages, create local project configuration, choose tracking adapters, and
coordinate operations. Capability packages must not import the CLI or
`apps/lab`. The initial CLI milestone is intentionally read-mostly: only
`posttrain init` writes files, and it refuses to replace an existing project
manifest.

The project control directory is `.posttrain/`. Its `project.toml`, catalog
overlays, and work packages are tracked inputs. Its `state/` directory is
ignored local state. Framework base selections come from `posttrain-catalog`;
durable run artifacts remain tracking-backend values.

## Plan of Work

### Maintained-fork publication order

The Trackio packaging change spans two repositories and must be completed in
this order:

1. In `/home/hammad/projects/trackio`, change the Python distribution identity
   from upstream-owned `trackio` to CarbonTeq-owned `carbonteq-trackio` while
   preserving the `trackio` import package and CLI. Give the fork an
   upstream-derived PEP 440 version, update self-deployment requirements and
   `CARBONTEQ_FORK.md`, build wheel and sdist, inspect metadata, install the
   wheel into a clean environment, and run focused deployment/storage tests.
2. Commit and push the Trackio fork change, configure PyPI Trusted Publishing,
   reserve the distribution with a release candidate, and record the immutable
   fork commit plus artifact hashes. Do not describe the fork distribution as
   reproducible before these steps succeed.
3. Only after the fork artifact exists, update
   `/home/hammad/projects/rl/packages/tracking-trackio/pyproject.toml` to depend
   on the published `carbonteq-trackio` version, remove the direct Git
   requirement from `apps/lab/pyproject.toml`, update `uv.lock`, and update both
   `docs/tooling/trackio/README.md` and the fork ledger.
4. Build the framework wheels and run `tests/consumer/` without a direct
   Trackio URL or sibling checkout. Then run the framework validation ladder.

The AutomationBench path follows the same rule. First publish the maintained
`automation-bench` fork under a CarbonTeq-owned distribution identity, then
replace the adapter's transitive Git URL, publish `automationbench-v1`, and only
then publish `posttrain-lab[gpu-posttrain]`. Verifiers requires the same
published-fork decision before Git-free installation of eval/train extras can
be claimed.

First, add `apps/cli/pyproject.toml` as distribution `posttrain`, depending on
`posttrain-catalog`. Add `apps/cli/src/posttrain_cli/` with a thin argparse
entrypoint and separately testable command functions. The command accepts a
global `--project-root` and `--json`. It exposes `version`, `init`, `doctor`,
`project show`, and `catalog list`, `catalog show`, and `catalog validate`.
Human output is concise; JSON output has deterministic keys for scripts.
Expected configuration or validation failures return exit code 1 with a safe
message on stderr. Argparse usage errors retain exit code 2.

`posttrain init [PATH] --project-id ID` creates
`.posttrain/project.toml`, `.posttrain/catalog/layer.yaml`,
`.posttrain/work_packages/README.md`, `.posttrain/.gitignore`, and the local
state directory. It never overwrites an existing manifest. The generated
catalog layer is valid with no entries. Adjust
`packages/catalog/src/posttrain/catalog/files.py` to accept an empty `files`
tuple while continuing to reject duplicate or unsafe filenames. Add focused
catalog tests proving empty layers compose with the packaged base.

Add CLI tests under `apps/cli/tests`. Tests call `main()` with explicit argument
lists, assert exit codes and JSON contracts, verify idempotent-safe failure on
reinitialization, exercise discovery from a nested directory, and validate a
generated empty overlay against the packaged base. Update root pytest,
coverage, import-linter roots, CI package imports, and the independent-consumer
wheel list so the installed command is tested outside the workspace.

Second, inventory `apps/lab/src/posttrain_lab/work_packages/` and move generic
contracts, YAML loading, seat validation, and runner behavior into a new
first-party package selected after the inventory. The package name must reflect
composition rather than create a second workflow ontology. Preserve lab imports
through compatibility re-exports until all current callers move. Add
`work-package validate` before `work-package run`; validation must resolve all
catalog refs and job definitions before any tracking or execution side effect.

Third, audit every built wheel with workspace sources disabled. Build all
first-party distributions, inspect `METADATA`, and create a clean lock/install
from only release-candidate artifacts plus declared indexes or immutable public
Git dependencies. Resolve the CarbonTeq TRL and Trackio fork strategy
explicitly. The alpha may use immutable public Git commits if remote hosts have
Git access; the polished release must document that requirement or publish
separate fork distributions. Do not allow the released train extra to silently
install behaviorally different upstream code.

Fourth, expand `tests/consumer/` so its fixture contains an executable no-op or
CPU-safe work package using the distributed runner. Use a real isolated local
tracking backend, retain a terminal run record, and query that same logical run
through Observatory. The test environment must exclude this checkout from
`sys.path`, disable workspace source fallbacks, and print enough evidence to
diagnose package, base-catalog, project, run, backend, and Observatory versions.

Fifth, restructure the root README into an install-first landing page and add a
developer documentation section for quickstart, concepts, project configuration,
catalog overlays, work packages, remote GPU operation, tracking, Observatory,
troubleshooting, compatibility, and upgrades. Documentation commands must run
in CI against built release artifacts. Keep backend-specific operational detail
in `docs/tooling/` and link to it rather than duplicating mutable fork notes.

Finally, complete release metadata and automation. Use one coordinated
pre-1.0 framework version for first-party wheels. A tag such as `v0.2.0a1`
builds wheels and source distributions once, validates them, installs them in
dependency order from a staging index, and publishes through trusted identity
rather than a stored upload token. Build Observatory once, test the exact image,
publish semantic-version and commit tags to GHCR, and record the immutable
digest. A release is not called stable until upgrade and remote GPU gates pass.

## Concrete Steps

Run commands from `/home/hammad/projects/rl` unless stated otherwise. Preserve
the existing dirty worktree and inspect it before each milestone:

    git status --short

For the first CLI slice, run focused tests:

    uv lock
    uv run pytest -q apps/cli/tests packages/catalog/tests
    uv run ruff check apps/cli packages/catalog
    uv run pyright apps/cli packages/catalog

Build and inspect the new distribution:

    uv build --package posttrain --wheel
    unzip -p dist/posttrain-*.whl '*/METADATA'
    unzip -l dist/posttrain-*.whl

Exercise the installed command from a temporary directory outside the
repository:

    uv run pytest -q tests/consumer

Expected user-visible behavior after the first milestone includes:

    $ posttrain init /tmp/example --project-id example
    Initialized post-training project example at /tmp/example

    $ cd /tmp/example
    $ posttrain catalog validate
    Catalog valid: framework-v1, 0 project entries

For the full release, run the repository validation ladder:

    uv sync --all-packages --locked --python 3.12
    uv run ruff check .
    uv run ruff format --check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

The staged release workflow must additionally build all artifacts, install with
workspace sources disabled, run documentation examples, run the
external-consumer acceptance, and print wheel hashes and the Observatory image
digest.

## Validation and Acceptance

The first CLI milestone is accepted when a clean installed wheel can initialize
a project, discovery works from a nested directory, `project show` reports
resolved absolute paths, an empty project overlay composes with the packaged
base, catalog list/show output identifies source layers, invalid projects return
exit code 1 without a traceback, and all commands support deterministic JSON
where applicable.

The work-package milestone is accepted when a fixture YAML file validates all
seats and catalog refs without side effects, then executes through the same
distributed runner used by the lab compatibility path. The CLI must report the
logical project, work package, job, run, terminal status, and produced artifact
identities.

The dependency milestone is accepted when a new environment outside this
workspace can resolve and install every documented extra from release artifacts
and documented public sources without `[tool.uv.sources]` from this repository.
The installed TRL and Trackio implementations must match the maintained
revisions declared by the release.

The consumer milestone is accepted when one command path starts from installed
wheels, executes a work package, records a succeeded run through Trackio, and
retrieves it through Observatory. No source-tree path, editable install, or
test-only tracking fake may participate.

The documentation milestone is accepted when a new developer can follow the
quickstart in a clean directory and CI executes the same commands. The release
milestone is accepted when PyPI artifacts and the GHCR image are retrievable by
immutable version, their hashes or digest are recorded, and a remote CarbonTeq
GPU server completes the documented gate from a frozen project lock.

## Idempotence and Recovery

All inspection and validation commands are read-only. `posttrain init` creates
new paths but refuses to overwrite `.posttrain/project.toml`; a failed partial
initialization may be inspected and completed by rerunning only after the user
removes or relocates the incomplete new control directory. It must never remove
an existing project, catalog entry, work package, artifact, or state directory.

Package builds write ordinary `dist/` outputs and may be repeated after removing
only the specific generated release directory. PyPI versions and OCI digests are
immutable; never reuse a version after any registry accepts it. Increment the
prerelease number and rebuild from a clean tag instead.

Runner extraction is additive. Keep lab compatibility re-exports until the lab,
tests, external fixture, and documentation all use the distributed boundary.
If extraction fails, restore imports to the compatibility module without
changing existing work-package YAML or evidence identities.

## Artifacts and Notes

The completed portability and consumer proof currently reports:

    external consumer: 1 passed
    full suite: 270 passed, 5 credential-gated W&B tests skipped
    import contracts: 8 kept, 0 broken

The consumer fixture installs built distributions into a separate Python 3.12
environment, executes `screen/cpu-check`, records
`data/train_examples`, `data/validation_examples`, and
`data/reserve_examples`, and confirms Observatory returns the same run ID with
a generic view. Its temporary top-level direct Trackio requirement is an
explicit remaining publication gate, not the final install contract.

## Interfaces and Dependencies

`apps/cli/src/posttrain_cli/cli.py` must expose:

    def main(argv: Sequence[str] | None = None) -> int

The `posttrain` console script calls that function. Command implementation may
use `posttrain.catalog.discover_project`, `load_project_layout`,
`open_catalog`, and `posttrain.common.CatalogRef`. It must catch
`posttrain.common.ContractError`, expected filesystem errors, and missing
catalog refs at the command boundary and convert them into concise user errors.
Unexpected programming errors remain visible during testing.

The initialized manifest uses schema version 1 and the existing
`ProjectLayout` contract. Catalog output uses the existing `Catalog.list` and
`Catalog.resolve` methods. The CLI must not import TRL, vLLM, Verifiers,
Trackio, W&B, `posttrain_lab`, or Observatory for project/catalog commands.
Later command groups may load optional tracking or execution dependencies only
when invoked.

The future distributed work-package boundary must preserve the canonical
`Recipe`, `JobDefinition`, `WorkPackage`, and `run_work_package` meanings from
`docs/post-training/05-apis.md`. It must not import concrete tracking providers;
the CLI host selects and injects a provider-neutral tracking backend.

Revision note (2026-07-23): Created after the user clarified that the framework
is optimized for CarbonTeq-managed projects but should still receive a polished
CLI, documentation, and public-registry release. The plan separates that release
effort from the already implemented portable-layout migration.
