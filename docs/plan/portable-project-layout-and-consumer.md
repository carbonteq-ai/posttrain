# Make the framework portable to an independent consumer

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

Maintain this document in accordance with `docs/templates/PLAN.md`. The
canonical product authority is `docs/post-training/README.md` and the six
documents it indexes. This work does not change the frozen meaning of projects,
work packages, catalog selections, runs, artifacts, or observation. It changes
how a host discovers project configuration and runtime state, and how the
reference framework assets are distributed.

## Purpose / Big Picture

After this work, a developer can create a repository outside this monorepo,
install the framework's built wheels, initialize a `.posttrain` project
directory, resolve a framework base selection plus a project overlay, execute a
CPU-safe work package, record its evidence through a real tracking backend, and
read the resulting run through Observatory. The developer does not need this
source checkout, its absolute path, workspace dependency declarations, or
top-level `catalog/`, `work_packages/`, `artifacts/`, or provider-state
directories.

The visible proof is an automated external-consumer acceptance test. It builds
the relevant wheels, creates a temporary repository that is not inside this
workspace, installs only those wheels into a clean environment, and runs the
same commands documented in the getting-started guide. A failure caused by an
implicit source-tree path, missing packaged catalog resource, or undeclared
dependency fails that test.

## Progress

- [x] (2026-07-23T11:29Z) Audited the current catalog, work-package, artifact,
  scratch, provider-state, packaging, README, and CI path assumptions.
- [x] (2026-07-23T11:29Z) Confirmed that the existing root `artifacts/`
  directory is a 59 GiB mixed historical evidence tree and must not be moved or
  deleted as part of the additive layout slice.
- [x] (2026-07-23T11:29Z) Recorded the durable `.posttrain` ownership and
  discovery decision in ADR 0013.
- [x] (2026-07-23T11:42Z) Implemented `ProjectLayout`, project discovery,
  strict source-path containment, absolute state overrides, and an explicit
  legacy constructor; 41 focused lab tests pass.
- [x] (2026-07-23T11:47Z) Added `.posttrain/project.toml`, moved the three
  repository work-package YAML files into `.posttrain/work_packages/`, moved
  the example overlay into `.posttrain/catalog/`, and ignored
  `.posttrain/state/`.
- [x] (2026-07-23T12:00Z) Added the independently buildable
  `posttrain-catalog` distribution, moved the framework base YAML into its
  package resources, and retained lab named-selection compatibility exports.
- [x] (2026-07-23T12:08Z) Added an independent-consumer fixture that builds and
  installs five wheels into a clean environment, discovers a copied external
  project, resolves a base selection plus overlay, and executes a deterministic
  CPU data partition; `1 passed`.
- [ ] Add executable no-op/work-package and tracking/Observatory evidence to the
  independent fixture (completed: clean install, project discovery, catalog
  composition, CPU data operation; remaining: distributed runner, real local
  backend, Observatory read).
- [ ] Remove absolute-path and workspace-only onboarding assumptions; repair the
  stale package-import CI matrix; add build, install, and consumer acceptance
  jobs. (Completed: README paths/onboarding, maintained package matrix,
  external-consumer CI job; remaining: public release workflow and registry
  installation.)
- [ ] Define release metadata, compatibility policy, licensing prerequisites,
  and upgrade/migration documentation without claiming a public release that
  has not occurred.
- [ ] Classify the existing `artifacts/` tree into retained evidence,
  reproducible output, and disposable cache before proposing any physical
  migration.
- [x] (2026-07-23T12:19Z) Ran the complete local validation ladder: 256 passed,
  5 credential-gated W&B tests skipped; external consumer 1 passed; Ruff,
  Pyright, all eight import contracts, and `git diff --check` passed.

## Surprises & Discoveries

- Observation: The canonical runtime model is already portable at the operation
  boundary. `apps/lab/src/posttrain_lab/execution.py` creates a temporary
  workspace, materializes tracked inputs into it, and deletes it after the run.
  The root `artifacts/` directory is therefore not required by `RunContext`.
  Evidence: `execute_run` uses `tempfile.TemporaryDirectory`; capability
  packages write only beneath `context.workspace`.

- Observation: The reference host is not portable even though the capability
  packages are. `apps/lab/src/posttrain_lab/catalog.py` derives `CATALOG_ROOT`
  from its source-file parents and resolves the entire catalog at module import
  time. The CLI independently assumes `<repository>/catalog` and
  `<repository>/work_packages`.
  Evidence: `CATALOG_ROOT = Path(__file__).resolve().parents[4] / "catalog"`,
  `_REFERENCE_CATALOG = open_catalog(...)`, and CLI path joins at startup.

- Observation: The framework base catalog is not presently a self-contained
  data bundle. It references an AutomationBench factory owned by the lab,
  repository-relative quantization recipes, and dependency-lock digests from
  this checkout.
  Evidence: `catalog/base/environments.yaml`,
  `catalog/base/quantization.yaml`, `catalog/base/training.yaml`, and
  `apps/lab/tests/test_catalog.py`.

- Observation: The configured ADR workflow names
  `docs/technical/adr/ADR.md`, but that template is absent. This repository
  already maintains accepted ADRs under `docs/decisions/`.
  Evidence: the template path does not exist; `docs/decisions/0011` and `0012`
  contain the current ADR structure.

- Observation: The package-import CI matrix is stale. It still installs the
  deleted `posttrain-reports` distribution and omits `posttrain-data`,
  `posttrain-tracking`, both tracking adapters, and Observatory.
  Evidence: `.github/workflows/quality.yml`.

- Observation: A package initializer cannot safely import
  `importlib.resources.files` under the name `files` when the same package has a
  `files.py` submodule; importing the submodule replaces that package attribute.
  Evidence: the first packaged-catalog test failed with `TypeError: 'module'
  object is not callable`; aliasing the resource function fixed it.

- Observation: Hatchling already includes non-Python files beneath an included
  package directory. A `force-include` for the same base YAML duplicates wheel
  archive paths.
  Evidence: the first wheel build rejected a second
  `posttrain/catalog/base/environments.yaml`; removing `force-include` produced
  a wheel containing all base resources.

- Observation: The current base catalog can be loaded from an installed wheel,
  but its quantization plans still name repository-relative recipe paths.
  Evidence: the clean consumer resolves common targets successfully; executing
  `model.transform` outside the checkout remains a later acceptance gate.

## Decision Log

- Decision: Use `.posttrain`, without a hyphen, as the project control
  directory.
  Rationale: it matches the `posttrain.*` Python namespace and `POSTTRAIN_*`
  environment-variable prefix while avoiding another spelling of the product.
  Date/Author: 2026-07-23 / user and Codex.

- Decision: Permit tracked configuration and ignored runtime state beneath the
  same control directory, with an explicit boundary:
  `.posttrain/project.toml`, `.posttrain/catalog/`, and
  `.posttrain/work_packages/` are source; `.posttrain/state/` is local state.
  Rationale: catalog overlays and work packages are reproducibility inputs,
  whereas scratch files, recovery checkpoints, downloads, and provider caches
  are replaceable machine state.
  Date/Author: 2026-07-23 / Codex.

- Decision: Durable artifacts remain in the selected tracking/artifact backend.
  `.posttrain/state/artifacts/` may be used only by an explicitly selected local
  backend or development mode; it is not a second lineage registry.
  Rationale: the canonical observation contract requires immutable artifact
  identities and consumed/produced edges, not directory ancestry.
  Date/Author: 2026-07-23 / Codex.

- Decision: Project discovery follows explicit configuration before implicit
  discovery. The precedence is an API/CLI project root, then
  `POSTTRAIN_PROJECT_ROOT`, then upward discovery of
  `.posttrain/project.toml`; absence is a typed error rather than silently
  interpreting an arbitrary current directory.
  Rationale: explicit precedence is reproducible in notebooks, CI, services,
  and nested working directories.
  Date/Author: 2026-07-23 / Codex.

- Decision: Keep legacy `<repository>/catalog` and
  `<repository>/work_packages` support only through an explicit compatibility
  layout, not as an automatic fallback in the new discovery API.
  Rationale: current commands and user work must remain runnable during the
  migration, but silent fallback would make external acceptance pass for the
  wrong reason.
  Date/Author: 2026-07-23 / Codex.

- Decision: Prove portability from a temporary repository and clean
  environment rather than by adding more in-workspace imports.
  Rationale: an in-workspace test can accidentally resolve source packages,
  package data, and undeclared dependencies from the checkout.
  Date/Author: 2026-07-23 / user and Codex.

## Outcomes & Retrospective

The first portability milestone is complete. Project identity, overlays, work
packages, and ignored local state now have a `.posttrain` contract. The
framework base is a wheel-packaged `posttrain-catalog` release rather than a
source-tree-relative directory. The repository itself uses
`.posttrain/project.toml` and `.posttrain/work_packages/`, and the reference CLI
successfully runs `posttrain-lab noop` through discovered project state.

The independent proof builds `posttrain-common`, `posttrain-data`,
`posttrain-eval`, `posttrain-train`, and `posttrain-catalog` wheels, installs
them into a clean environment, copies a fixture outside this workspace, resolves
base and overlay targets, and partitions a canonical dataset. This closes the
path-discovery and package-resource risk.

The full objective is not complete. The independent fixture does not yet
execute the distributed work-package runner, persist a run through Trackio, or
query it through Observatory. Quantization recipe resources also remain
repository-relative. Public registry publication, license/contributor policy,
upgrade compatibility, and classification of the 59 GiB historical artifact
tree remain explicit release gates.

## Context and Orientation

This repository is a Python 3.12 `uv` workspace. Reusable capability
distributions live under `packages/`: `posttrain-common`, `posttrain-data`,
`posttrain-serve`, `posttrain-eval`, `posttrain-train`, `posttrain-tracking`,
and provider adapters. `apps/lab` is the reference execution host and
`apps/observatory` is the read-only evidence product.

A catalog is a versioned set of concrete selections such as model variants,
training bindings, evaluation plans, and execution targets. A base catalog is a
framework release. A project overlay adds or replaces selections for one
consumer. A work package binds catalog selections to jobs for the `screen`,
`train`, or `qualify` stage.

The framework base catalog now lives inside the `posttrain-catalog`
distribution at `packages/catalog/src/posttrain/catalog/base/`. This
repository's project manifest, overlay example, and work-package YAML live
under `.posttrain/`. `apps/lab/src/posttrain_lab/catalog.py` retains
lab-facing named selection exports while delegating catalog loading to
`posttrain.catalog`. `apps/lab/src/posttrain_lab/execution.py` provides the
ephemeral workspace boundary.

The new project control directory is `.posttrain`. The term project source
means files that must be committed to reproduce behavior. The term runtime
state means machine-local files that can be regenerated or recovered from a
durable backend.

## Plan of Work

First, add `apps/lab/src/posttrain_lab/project.py`. Define an immutable
`ProjectLayout` with absolute `root`, `control_dir`, `manifest`,
`catalog_overlays`, `work_packages`, and `state` paths. Define
`discover_project(start, explicit_root=None, environ=None)` and
`load_project_layout(root)` so path precedence and validation are testable
without changing the process working directory. Parse
`.posttrain/project.toml` with Python 3.12 `tomllib`. The manifest initially
contains a schema version, project ID, zero or more catalog overlay directories,
one work-package directory, and optional state-directory override. Reject
relative paths that escape the project root.

Second, add CLI project-root support without removing `--repository` in the
same patch. New portable commands use `--project-root` and discovery.
Existing fixed reference commands construct an explicit legacy layout from
`--repository`, which makes the compatibility boundary visible in code and
tests. Route scratch state through `ProjectLayout.state / "scratch"` unless the
caller supplied `--scratch-root`.

Third, separate base catalog resolution from the lab source tree. Move
manifest-controlled YAML loading into a reusable distribution boundary and
package the framework base assets with that distribution. Because the current
base includes host-owned AutomationBench factories and repository-relative
quantization recipes, split those entries into a lab/project overlay or package
the referenced resources and factories deliberately. Never weaken immutable
revision or digest checks merely to make packaging easier. The selected base
catalog release identity must be available to the composed catalog and recorded
with resolved run inputs.

Fourth, create `tests/consumer/` with a small fixture project containing
`.posttrain/project.toml`, a catalog overlay, a work package, and a Python
entrypoint. Its test harness builds wheels into a temporary wheelhouse, creates
a temporary virtual environment outside this repository, installs from that
wheelhouse with index access disabled for local framework packages, and runs
the fixture from its own directory with source-path environment variables
removed.

The first fixture operation is a no-op that still resolves a target and records
the canonical run snapshot. The second is a CPU-safe real operation, preferably
dataset validation/materialization from `posttrain.data`, so it exercises a
capability package without downloading a model or requiring Docker/GPU. The
tracking proof must use a real local backend. If Trackio's configured local
mode cannot be isolated deterministically, introduce a small durable filesystem
tracking adapter only if it is a supported backend rather than a test fake.
Observatory then reads the recorded run through the corresponding
`RunDataSource`.

Finally, rewrite the root README around an installable-consumer path and repair
CI. Add wheel builds and independent imports for every maintained
distribution. Add the external-consumer test as a separate job so it cannot see
the editable workspace. Document version compatibility, immutable base-catalog
selection, upgrade behavior, and the absence of a public release until a real
registry/repository release occurs. Licensing and contributor-policy gaps are
release blockers, not files to invent without owner approval.

## Concrete Steps

Run all commands from `/home/hammad/projects/rl` unless a step says otherwise.

Inspect the current dirty worktree before every slice:

    git status --short

Implement and test project layout first:

    uv run pytest -q apps/lab/tests/test_project.py apps/lab/tests/test_catalog.py
    uv run ruff check apps/lab/src/posttrain_lab/project.py apps/lab/tests/test_project.py
    uv run pyright apps/lab/src/posttrain_lab/project.py apps/lab/tests/test_project.py

Build and inspect each maintained wheel after distribution changes:

    uv build --package posttrain-common --wheel
    uv build --package posttrain-data --wheel
    uv build --package posttrain-lab --wheel

The exact catalog distribution build command will be added here when its
package name is finalized. Inspect wheel contents and verify that packaged
catalog resources and manifests are present.

Run the external acceptance harness:

    uv run pytest -q tests/consumer

The test must print or retain evidence showing an external temporary project
path, successful catalog base and overlay resolutions, a succeeded run ID, and
an Observatory read of that same run.

Run the repository validation ladder after all slices:

    uv sync --all-packages --locked --python 3.12
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

## Validation and Acceptance

The project-layout slice is accepted when a test starts below a temporary
project root, discovers `.posttrain/project.toml`, obtains absolute nonescaping
paths, and rejects a manifest path such as `../../other`. An explicit root and
`POSTTRAIN_PROJECT_ROOT` must override upward discovery in that order.

The distribution slice is accepted when the base catalog resolves after
installation from wheels in an environment that has neither this repository on
`sys.path` nor a `catalog/` directory beside the installed package.

The independent-consumer slice is accepted when a temporary external repository
can resolve one base target and one overlay target, load a work package, execute
the no-op and CPU-safe operation, persist a terminal succeeded outcome, and
retrieve the run through Observatory. The test must fail if any implementation
uses `/home/hammad/projects/rl`, `Path.cwd() / "catalog"`, or a workspace source
fallback.

The migration is accepted when existing reference-host tests still pass through
the explicit legacy layout, new documentation uses `.posttrain`, and no files
inside the existing 59 GiB `artifacts/` tree were moved or deleted.

## Idempotence and Recovery

All layout and packaging changes are additive until the external fixture passes.
Repeated discovery and initialization must not rewrite an existing manifest.
Tests use temporary directories and environments and remove only paths they
created.

Do not move or delete root `artifacts/` during the additive milestones. The
tracked catalog and work-package source files have moved to their accepted
package and `.posttrain` locations; the explicit legacy constructor remains for
other checkouts. If a new layout implementation breaks reference commands, fix
callers rather than restoring implicit current-directory discovery. If wheel
installation exposes a missing resource, add it to package metadata and
rebuild; do not copy it into the test environment.

## Artifacts and Notes

Initial audit evidence:

    root artifacts: 59G, 78 top-level directories
    former lab catalog root: Path(__file__).resolve().parents[4] / "catalog"
    packaged base: posttrain/catalog/base/*.yaml
    project source: .posttrain/project.toml, catalog/, work_packages/

The direct public package endpoints for `posttrain-common` and
`posttrain-train` returned 404 on 2026-07-23. This plan therefore treats public
release as future work and does not claim existing external adoption.

## Interfaces and Dependencies

In `apps/lab/src/posttrain_lab/project.py`, the first slice must provide:

    @dataclass(frozen=True, slots=True)
    class ProjectLayout:
        project_id: str
        root: Path
        control_dir: Path
        manifest: Path
        catalog_overlays: tuple[Path, ...]
        work_packages: Path
        state: Path

        @classmethod
        def legacy(cls, repository: Path, project_id: str) -> ProjectLayout: ...

    def discover_project(
        start: Path,
        *,
        explicit_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> ProjectLayout: ...

    def load_project_layout(root: Path) -> ProjectLayout: ...

The manifest uses standard-library `tomllib`; no new configuration dependency
is required. Every returned path is absolute. Source paths must stay within the
project root. An absolute state override may live outside the project root
because large scratch and recovery files may require a separate disk.

Catalog APIs continue to expose `Catalog.open`, `Catalog.resolve`, and
`Catalog.list`. The packaging milestone may introduce a catalog distribution,
but capability packages must remain independent and must not import the
reference lab.

Revision note (2026-07-23T11:29Z): Created the plan after the portability and
external-adoption audit. Recorded the additive migration constraint because the
existing root artifact tree contains large historical evidence referenced by
documentation.

Revision note (2026-07-23T12:08Z): Recorded the completed portable layout,
packaged base catalog, clean-wheel consumer proof, package-resource discoveries,
and the remaining execution/tracking/Observatory acceptance gap.

Revision note (2026-07-23T12:19Z): Recorded the passing full validation ladder,
portable reference-host no-op, CI/onboarding updates, first-milestone outcome,
and remaining release and end-to-end evidence gates.
