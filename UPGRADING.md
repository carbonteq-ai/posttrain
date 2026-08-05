# Upgrading Posttrain

Posttrain release artifacts are immutable. Upgrade by building a new
environment from one exact GitHub Release; never overwrite or mix wheels from
two framework releases.

## Before upgrading

1. Read the target release notes and [CHANGELOG.md](./CHANGELOG.md).
2. Confirm the target release supports the project's Python, operating system,
   tracking provider, training backend, inference backend, and GPU target.
3. Commit the project manifest, catalog overlays, work packages, dependency
   declarations, and lock information.
4. Confirm every active run has a terminal outcome and every required artifact
   has been published through its tracking or artifact backend.

`.posttrain/state/` is cache, scratch, recovery, and provider-local state. It is
not the durable lineage authority and should not be copied as the migration
mechanism.

## Install the target release

Download the target wheelhouse and verify its recorded archive hash:

```bash
gh release download <target-tag> \
  --repo carbonteq-ai/posttrain \
  --pattern 'posttrain-wheelhouse-*.tar.gz' \
  --pattern release-SHA256SUMS

sha256sum --check release-SHA256SUMS
mkdir posttrain-wheelhouse
tar -xzf posttrain-wheelhouse-*.tar.gz -C posttrain-wheelhouse
(cd posttrain-wheelhouse && sha256sum --check SHA256SUMS)
```

Create a fresh environment so stale transitive packages cannot survive:

```bash
uv venv .venv-next --python 3.13
uv pip install \
  --python .venv-next/bin/python \
  --constraint posttrain-wheelhouse/github-constraints.txt \
  --find-links posttrain-wheelhouse \
  posttrain
```

Add the same capability packages and extras used by the project. Keep the old
environment until validation is complete.

## Validate the project

```bash
.venv-next/bin/posttrain doctor
.venv-next/bin/posttrain catalog validate
.venv-next/bin/posttrain work-package validate \
  .posttrain/work_packages/<package>.yaml
```

No `--host` is required. Standard jobs come from `posttrain.jobs`. If the
project declares `entry = "MODULE:configure"` in `.posttrain/project.toml`, the
CLI loads that entry; otherwise it builds the default job runtime. Keep
`--host` / `--entry` only as temporary overrides during migration.

Run one CPU-safe package, then the project's supported GPU smoke package. For
tracked execution, confirm the terminal run and required evidence through
Observatory before promoting the new environment.

If the release changes a project or catalog schema, apply the release-specific
migration before validation and commit the migrated files together with the new
release selection.

## Rollback

Stop starting new runs with the candidate environment and return to the
previous untouched environment and release selection. Do not rewrite completed
run metadata, provider storage, or artifact versions. If a new run produced an
invalid artifact, leave its evidence intact and mark the artifact unsuitable in
project decision records rather than deleting lineage.
