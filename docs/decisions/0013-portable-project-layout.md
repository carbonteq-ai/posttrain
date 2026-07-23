# ADR 0013 — Portable project layout and runtime-state boundary

## Status

Accepted.

## Context

The reusable post-training capability packages are intended to run from another
CLI, notebook, service, or repository, but the reference host currently assumes
this monorepo's source layout. It derives the base catalog from parents of
`posttrain_lab/catalog.py`, joins `catalog/` and `work_packages/` onto a
repository argument, and defaults that argument to the current working
directory. The root also accumulates `artifacts/`, `wandb/`, and optional
scratch output beside tracked project configuration.

These paths mix three different ownership classes. Framework-shared catalog
entries are versioned framework assets. Project catalog overlays and work
packages are reproducibility inputs that should be reviewed and committed.
Scratch files, recovery checkpoints, downloads, and provider caches are
machine-local runtime state. Durable model, dataset, evaluation, and report
artifacts are neither project configuration nor anonymous local files: they are
immutable values published through a selected tracking or artifact backend with
consumed and produced lineage edges.

An external consumer needs one deterministic way to locate project
configuration without inheriting this checkout's absolute paths. The migration
must preserve current work and must not move the existing large root
`artifacts/` tree implicitly.

## Decision

- Use `.posttrain/` as the project control directory. The spelling matches the
  `posttrain.*` Python namespace and `POSTTRAIN_*` environment-variable prefix.
- Store tracked project configuration at:
  `.posttrain/project.toml`, `.posttrain/catalog/`, and
  `.posttrain/work_packages/`.
- Store ignored machine-local runtime state under `.posttrain/state/`.
  Subdirectories may include scratch workspaces, recovery checkpoints, caches,
  provider-local files, and local-development artifact material.
- Keep durable artifacts in the configured tracking/artifact backend.
  `.posttrain/state/` is not a parallel run registry and directory ancestry is
  not lineage.
- Discover a project in this precedence order: an explicit API or CLI project
  root, `POSTTRAIN_PROJECT_ROOT`, then upward search for
  `.posttrain/project.toml`. If none is found, return a typed error.
- Require project-source paths in the manifest to remain within the project
  root. Permit an explicit absolute state-directory override so large scratch
  and recovery files can use another disk.
- Package the framework base catalog as a versioned framework asset rather than
  requiring a consumer to copy it into `.posttrain/catalog/`. The project
  catalog contains overlays only.
- Record the selected base catalog release and the resolved source layer for
  run inputs.
- Support consuming the former top-level `catalog/` and `work_packages/`
  layout during an additive compatibility period. Reference-host callers may
  select it through an explicit legacy layout constructor; the new discovery
  API does not silently fall back to it.
- Do not move or delete the existing top-level `artifacts/` tree as part of the
  layout implementation. Classify its contents and references in a separate
  migration step.
- Prove the layout and distribution boundary with an automated fixture
  repository outside the workspace that installs built wheels into a clean
  environment.

## Consequences

- A repository has one predictable post-training control directory without
  presenting generated state as ordinary project source.
- Catalog overlays and work packages remain visible to Git review even though
  they live under a hidden control directory.
- Hosts can start from nested working directories, notebooks, CI jobs, or
  services and resolve the same project through an explicit precedence rule.
- Large scratch and recovery state can move to another disk without changing
  project identities or catalog selections.
- The framework base catalog needs a real distribution boundary. Entries that
  depend on lab-owned factories, local recipes, or a monorepo lockfile must be
  packaged deliberately or moved into a project/lab overlay.
- Reference-host code must distinguish portable project discovery from legacy
  repository compatibility. This creates temporary duplication but avoids a
  flag day and prevents tests from passing through accidental current-directory
  behavior.
- Provider adapters may require configuration such as `WANDB_DIR` to keep
  provider-local state under the project state root. Such directories remain
  backend implementation details.

## Alternatives Considered

### Keep `catalog/` and `work_packages/` at repository root

Rejected as the portable default because it exposes host-specific configuration
as unrelated root directories and provides no home for project identity or
runtime-state policy. It remains available during the compatibility period.

### Put all catalog and artifact material under `.posttrain/`

Rejected because it collapses tracked configuration, ephemeral state, and
durable artifact lineage into one filesystem convention. Only project overlays
belong to project source; the framework base is distributed, and durable
artifacts belong to an artifact backend.

### Put all runtime state in an operating-system global cache

Rejected as the sole policy because project-local inspection and cleanup are
useful, while training scratch and recovery may require a project-specific
disk. Hosts may still select an XDG or other external state directory through
the explicit override.

### Infer a project from any directory containing `catalog/`

Rejected because arbitrary current-directory inference is ambiguous and lets
tests or notebooks select the wrong project silently.

### Move the existing root directories immediately

Rejected because the root artifact tree contains large historical qualification
and Observatory evidence referenced by documentation. Migration requires an
inventory and retention decision.

## Implementation Notes

- Implement the first layout contract in
  `apps/lab/src/posttrain_lab/project.py`.
- Parse `.posttrain/project.toml` with Python 3.12 `tomllib`.
- Add focused tests in `apps/lab/tests/test_project.py`.
- Route new portable CLI commands through project discovery. Keep
  `--repository` only for explicitly constructed legacy layouts until all
  reference commands and documentation migrate.
- Move manifest-controlled catalog loading and packaged base assets out of
  source-tree-relative lookup. Preserve the public `Catalog.resolve` semantics.
- Add the external-consumer acceptance harness under `tests/consumer/`.
- Maintain the living execution plan at
  `docs/plan/portable-project-layout-and-consumer.md`.

## Revision History

- 2026-07-23: Accepted `.posttrain/` as the portable project control directory,
  separated tracked configuration from ignored runtime state, defined discovery
  precedence and legacy compatibility, and required an independent-consumer
  acceptance test.
- 2026-07-23: Clarified that legacy top-level layouts remain readable for other
  checkouts while this repository's tracked files move to the packaged base and
  `.posttrain/` locations.
