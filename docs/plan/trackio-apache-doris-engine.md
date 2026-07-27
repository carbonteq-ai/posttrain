# Add Apache Doris as a first-class Trackio database engine

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

This document must be maintained in accordance with `docs/templates/PLAN.md`.
It is a multi-repository plan: generic storage implementation belongs in the
sibling `../trackio` fork, deployment belongs in the sibling `../ai-infra`
repository, and framework integration evidence belongs in this repository.

## Purpose / Big Picture

After this change, a self-hosted Trackio server can set
`TRACKIO_DATABASE_ENGINE=doris` and use Apache Doris as its authoritative
database for experiments. The existing `sqlite` and `turso` choices continue
to work. Trackio clients, its HTTP API, and Observatory do not change their
logical contracts: a run written through the normal Trackio SDK is readable
through Trackio and appears in Observatory regardless of which database engine
backs the server.

Doris stores structured metadata and evidence: projects, runs, configs,
metrics, system metrics, traces, alerts, artifact manifests and aliases, and
run-to-artifact lineage. Model files, media files, and other artifact bytes are
not relational rows and remain behind Trackio's existing server-managed
artifact byte-storage boundary. This is still native Doris support: Doris is
the authoritative Trackio metadata database, not an asynchronous copy or
sidecar projection.

## Progress

- [x] (2026-07-26 11:11Z) Corrected the requirement from an analytical
  projection to a first-class Trackio engine.
- [x] (2026-07-26 11:11Z) Audited the current storage implementation and
  established that the SQLite DB-API shim is not a safe Doris extension point.
- [x] (2026-07-26 11:11Z) Recorded the multi-repository ownership, migration,
  rollback, and end-to-end acceptance boundaries in this plan.
- [x] (2026-07-26 11:43Z) Defined and tested the Trackio storage-provider
  contract without changing
  SQLite or Turso behavior.
- [x] (2026-07-26 11:43Z) Implemented the native Doris schema, connection
  management, writes, reads,
  mutations, and project isolation in `../trackio`.
- [x] (2026-07-26 11:43Z) Added and exercised an idempotent
  SQLite-to-Doris migration command with
  reconciliation and rollback receipts.
- [x] (2026-07-26 11:43Z) Deployed a parallel Doris-backed Trackio candidate
  from `../ai-infra`
  and run provider conformance, restart, concurrency, and failure tests.
- [x] (2026-07-26 11:52Z) Migrated four retained Trackio projects, reconciled
  their counts and 27 run identities, switched the shared service to Doris,
  and retained the hashed SQLite cutover snapshot and explicit rollback
  composition.
- [x] (2026-07-26 11:43Z) Added the deployed Doris candidate as an Observatory
  source and passed authenticated readiness with the complete healthy source
  set. Sequence 34 records the cutover evidence.
- [x] (2026-07-27 05:34Z) Re-audited the unpublished candidate against the
  SQLite provider semantics and migration safety rather than treating the
  successful deployment as release proof. Focused Doris/provider tests passed
  12 cases, but the audit found release-blocking migration, run-resolution,
  and schema-version defects.
- [x] (2026-07-27 05:44Z) Made migration consume a SQLite backup snapshot that
  includes committed WAL rows,
  make verify-only genuinely non-initializing, and reconcile canonical
  key/content digests for every authoritative table including artifact
  metadata and aliases.
- [x] (2026-07-27 05:44Z) Matched SQLite same-name run resolution by ordering
  grouped run identities
  by their first event time, then add late-metric parity tests.
- [x] (2026-07-27 05:44Z) Added fail-closed Doris schema-version negotiation.
  Only an empty schema bootstraps, the version record is written last, and
  partial, older, or newer managed schemas are rejected before any write.
- [x] (2026-07-27 05:44Z) Revalidated the correction with 31 focused Doris
  safety tests, 370 non-GPU unit tests, repository-wide Ruff, locked
  dependency resolution, wheel/sdist build, and diff checks. The four real
  Doris integration cases remain intentionally skipped without a configured
  target and are the next release gate.
- [x] (2026-07-27 05:46Z) Ran all four real-Doris integration cases using the
  corrected local source against the dedicated `trackio_candidate` database.
  Core evidence, artifact lineage and run mutations, SQLite migration with
  content reconciliation plus non-initializing verify-only, and four-writer
  project isolation passed. Five exact synthetic project IDs were empty before
  the suite; cleanup removed all 120 generated rows afterward.
- [x] (2026-07-27 05:51Z) Repeated all four real-Doris cases against a uniquely
  named empty database and least-scope disposable user. Current source
  bootstrapped the complete managed schema, wrote the version record only
  after table creation, and passed the suite. The database and user were
  removed in the guaranteed cleanup path. An initial attempt stopped before
  SQL because the administrative account could not read the Compose
  environment without sudo; the successful retry used its configured
  non-interactive sudo path.
- [x] (2026-07-27 06:07Z) Closed the remaining release-relevant provider
  parity gaps for metric time windows, canonical metric/artifact run records,
  nullable legacy identities, scalar/media/report/trace tab classification,
  artifact relog identity preservation, legacy-link folding, and deterministic
  ordering. The focused parity suite passed 93 cases, the full non-hardware
  unit suite passed 376 with 3 skips, and the four real-Doris cases passed
  again with all 120 synthetic rows removed.
- [x] (2026-07-27 06:45Z) Close the bundled-dashboard security gate discovered
  during final packaging. Upgrade to Vega 6.3.1, Vega-Lite 6.4.3, Vega Embed
  7.1.0, Svelte 5.56.8, Vite 6.4.3, and Vitest 3.2.7; emit v6 chart schemas,
  retain explicit canvas rendering, and escape generated axis-label
  expressions. Production-only and complete npm audits now report zero
  vulnerabilities. Full frontend lint, 48 tests including two hostile-label
  compile/evaluate regressions, a warning-free production build, clean
  `npm ci` wheel build, and 17 Python frontend/distribution tests pass.
  Candidate formatting was corrected in seven post3 files and its 93 focused
  tests still pass. Nine pre-existing files outside post3 remain outside the
  repository-wide Ruff-format baseline; no functional or release file is
  hidden by that known baseline debt.
- [ ] Update the Trackio fork ledger, framework consumer page, immutable fork
  pin, lockfile, and deployment image only after the fork change is committed
  and published.

## Surprises & Discoveries

- Observation: Turso is currently treated as a SQLite-compatible driver, not a
  separate storage provider.
  Evidence: `../trackio/trackio/database.py` permits only `turso` and `sqlite`
  and exports SQLite-shaped `Connection`, `Cursor`, error, and `connect`
  symbols.

- Observation: `SQLiteStorage` mixes logical Trackio operations with
  engine-specific filesystem and SQL behavior.
  Evidence: `../trackio/trackio/sqlite_storage.py` is more than 5,000 lines and
  includes SQLite pragmas, per-project `.db` paths, file locks, autoincrement
  IDs, SQLite authorizers, Parquet sidecars, arbitrary SQLite query support,
  and artifact metadata.

- Observation: Doris provides a MySQL wire protocol but does not promise
  SQLite semantics.
  Evidence: Doris clients connect to the FE query port through the MySQL
  protocol. Doris supports atomic small batch `INSERT INTO VALUES`, explicit
  write transactions with read-committed isolation, Duplicate Key tables for
  append-only events, and Unique Key merge-on-write tables for idempotent
  upserts. These primitives support Trackio, but their SQL and mutation
  semantics require a native provider.

- Observation: the public deployed path survives a Trackio container restart
  without losing 15 metric updates or artifact lineage.
  Evidence: the candidate on `ai-control:7862` retained all 15
  `qualification/update` points and the `qualification-model:v0` output after
  its container was restarted.

- Observation: a single deployed Trackio candidate accepted concurrent HTTP
  writers without loss or duplication.
  Evidence: four client processes wrote 25 explicitly stepped updates each;
  the remote API reconciled four runs and exactly 100 update rows.

- Observation: native Verifiers traces preserve their component rewards, but
  Observatory's aggregate summary only derives `mean_reward` from a top-level
  `reward` or `score`.
  Evidence: a real Doris-backed trace with
  `rewards={"correct": 1.0}` appeared complete with its component intact, while
  its aggregate `mean_reward` remained unset. This is an Observatory semantic
  gap, not Doris data loss; multiple reward components must not be summed
  without an explicit aggregation contract.

- Observation: artifact version allocation is process-serialized in the
  candidate implementation.
  Evidence: `DorisStorage` uses an in-process lock around version allocation.
  The deployment therefore supports one Trackio server replica; horizontal
  Trackio replication requires a Doris-safe allocator before it can be
  qualified.

- Observation: the original migration gate compared counts rather than
  authoritative contents and opened the SQLite source with `immutable=1`.
  Evidence: a live WAL-backed source may have committed rows outside the main
  database file, while equal table counts do not prove matching keys,
  artifact manifests, aliases, or lineage. The current verify-only path also
  initializes the target schema, so it is not read-only.

- Observation: same-name run resolution diverges from SQLite after a late
  metric arrives on an older run.
  Evidence: the Doris provider orders the last event timestamp, whereas the
  established SQLite behavior groups by run ID and orders the run's first
  timestamp. Reads, rename, or delete can therefore target different runs
  depending on the selected engine.

- Observation: the candidate schema bootstrap is not forward-safe.
  Evidence: `CREATE IF NOT EXISTS` is followed by an unconditional schema
  version write. No negotiation rejects a database created by a newer binary,
  and no ordered migration path distinguishes initialization from upgrade.

## Decision Log

- Decision: implement Doris behind an explicit Trackio storage-provider
  contract, not in `trackio/database.py` as another SQLite-compatible driver.
  Rationale: the current driver boundary assumes SQLite connection behavior and
  leaks SQLite-only functionality. A logical provider keeps the public Trackio
  API stable while allowing correct engine-specific SQL.
  Date/Author: 2026-07-26 / Codex and user.

- Decision: keep `SQLiteStorage` intact during the additive implementation and
  select the provider through a new factory.
  Rationale: SQLite and Turso are proven paths. An additive provider permits
  conformance comparison and a safe rollback before any cleanup refactor.
  Date/Author: 2026-07-26 / Codex.

- Decision: use one Doris database for Trackio with `project_id` included in
  every table's key rather than creating a database per project.
  Rationale: project identity is a logical tenancy boundary. Including it in
  keys makes cross-project administration, backup, migration, and Observatory
  access practical without weakening API-level isolation.
  Date/Author: 2026-07-26 / Codex.

- Decision: use application-generated stable string identifiers for rows that
  currently depend on SQLite autoincrement values.
  Rationale: deterministic or UUID identifiers survive retries, migration, and
  concurrent writers. API responses must not expose an engine-dependent
  identity.
  Date/Author: 2026-07-26 / Codex.

- Decision: use Unique Key merge-on-write tables with stable application event
  IDs for both immutable event series and mutable records.
  Rationale: metrics and system samples retain every logical occurrence because
  each has its own stable event ID, while retries of the same occurrence remain
  idempotent. Configs, traces, alerts, artifact aliases, and lineage links use
  their corresponding stable logical IDs.
  Date/Author: 2026-07-26 / Codex.

- Decision: do not expose unrestricted `query_project` SQL for Doris in the
  first release.
  Rationale: that endpoint currently validates SQLite syntax and table layout.
  Stable Trackio query APIs are the engine-neutral contract. A future
  explicitly read-only Doris query surface may be added after authorization,
  resource limits, and cross-project isolation are proven.
  Date/Author: 2026-07-26 / Codex.

- Decision: migrate by copy, reconcile, and cut over; do not dual-write in the
  first release.
  Rationale: application-level dual writes introduce ambiguous partial-success
  states. A short write pause plus an idempotent copy and exact reconciliation
  produces a bounded, reversible transition for this local deployment.
  Date/Author: 2026-07-26 / Codex.

- Decision: deploy the uncommitted implementation as a separately addressed
  qualification candidate with a source-tree receipt, not as the shared
  Trackio image.
  Rationale: the receipt binds the candidate wheel to the base commit, source
  diff digest, status, and wheel digest. Port 7860 and `trackio.lan` remain on
  the published SQLite image until backup/restore and retained-project
  reconciliation pass.
  Date/Author: 2026-07-26 / Codex.

- Decision: operate one Doris-backed Trackio server replica for the first
  release.
  Rationale: all required workload concurrency occurs through that server, and
  the candidate passed four concurrent client writers. Multi-replica metadata
  allocation is a separate HA concern and is not silently claimed.
  Date/Author: 2026-07-26 / Codex.

- Decision: content reconciliation, provider-semantic parity, and forward-safe
  schema negotiation are release gates, not post-publication hardening.
  Rationale: a successful 15-update candidate and count reconciliation prove
  the basic transport path, but cannot prove that migration preserved the
  authoritative database or that SQLite and Doris select the same logical run.
  The fork remains unpublished until a stable SQLite snapshot, non-mutating
  verification, per-table content digests, same-name run parity, and schema
  version refusal/migration tests pass.
  Date/Author: 2026-07-27 / Codex.

- Decision: raw project SQL and multi-writer artifact publication are not
  silently emulated on Doris.
  Rationale: SQLite raw SQL is scoped by a project-local database, while Doris
  uses shared project-keyed tables. The current artifact workflow interleaves
  reads and writes, and Doris transactions cannot contain those queries.
  Version allocation is therefore safe only in the declared single-server
  topology. A future HA release needs a scoped query contract plus staged or
  optimistic artifact coordination rather than an autocommit toggle.
  Date/Author: 2026-07-27 / Codex.

## Outcomes & Retrospective

Native Doris support is implemented and serves the shared qualification
endpoint. The SDK/HTTP path, artifact lineage, restart persistence, concurrent
writers, native backup/restore, retained-project count reconciliation, and
deployed Observatory source health passed against real Doris. The previously
considered projection sidecar was removed and is not part of the architecture.
The hashed SQLite snapshot and rollback composition remain available.

Those results establish a useful candidate, not yet a published database
engine. The 2026-07-27 source audit's P0 findings are now corrected in source:
migration snapshots committed WAL data, verification is non-initializing and
content-based across the complete authoritative table set, artifact history is
preserved, same-name run selection matches SQLite, and schema startup fails
closed. All four real-Doris cases now pass both against the dedicated candidate
database and a freshly bootstrapped disposable database, including content
reconciliation and verify-only. Fork publication, immutable consumer pinning,
deployment of the corrected service, retained-run reconciliation, and
Observatory readback remain mandatory.
The service also remains single-server until artifact version allocation is
safe across Trackio replicas.

## Context and Orientation

Trackio is the tracking backend used by the post-training framework.
`../trackio/trackio/server.py` exposes the self-hosted HTTP API and currently
calls static methods on `SQLiteStorage`.
`../trackio/trackio/sqlite_storage.py` owns both SQLite/Turso persistence and
much local filesystem behavior. `../trackio/trackio/database.py` is a narrow
SQLite-compatible driver selector. `../trackio/trackio/api.py` provides local
and remote read objects. The remote path already talks to the server and must
remain independent of its database choice.

In this repository, `packages/tracking-trackio` is the framework adapter and
`apps/observatory` is the read-only evidence product. Neither may import a
Doris client or know Doris table names. They continue to consume Trackio's
logical HTTP/query API.

The deployment repository is `../ai-infra`. Its Trackio container currently
sets `TRACKIO_DATABASE_ENGINE=sqlite`; its Apache Doris service is already a
separate qualified service. Deployment changes inject Doris connection
settings into Trackio through secret-backed environment variables. No password
or connection secret belongs in Git, a job package, a run config, or an
execution log.

## Plan of Work

First, in `../trackio`, introduce `trackio/storage.py`. It defines a
`get_storage()` factory and returns the existing `SQLiteStorage` for `sqlite`
and `turso`, or a new `DorisStorage` for `doris`. The factory validates all
required Doris settings at server startup and reports a redacted actionable
error. Change server-owned imports to use the selected provider. Keep local
client and Spaces-only file synchronization explicitly on `SQLiteStorage`
until their engine-neutral behavior is separated.

Define the provider surface from actual server calls, not from every private
SQLite helper. The first contract must cover project/run discovery; run config;
metric, system-metric, trace, and alert writes and reads; batch reads;
rename/delete; tab and summary queries; project metadata; artifact metadata,
aliases, manifests, consumers, and lineage links. Dataset/Parquet import,
bucket database-file sync, SQLite pragmas, local file paths, and guarded raw
SQLite SQL are declared unsupported capabilities rather than silently
approximated.

Add `trackio/doris_storage.py` and `trackio/doris_schema.py`. Connect to Doris
FE using a maintained Python MySQL-protocol driver, TLS where configured, a
least-privilege Trackio database user, and bounded connect/read/write timeouts.
Create schema through versioned, idempotent
migrations. Every table contains `project_id`; every query binds it.

Use application-assigned IDs and explicit idempotency keys. Metrics and system
metrics retain each occurrence, keyed by project and stable `log_id` when
available and otherwise by an application-generated event ID. Traces preserve
the existing `(project_id, run_id, trace_type, external_id)` retry contract.
Alerts use `alert_id`. Artifact name/version, alias, manifest digest, and
run-artifact links keep their existing logical uniqueness. Writes are batched
per Trackio flush and use one atomic Doris statement or transaction.

Add provider conformance tests that execute the same logical scenarios against
SQLite and Doris and compare public results rather than physical row IDs or
ordering accidents. Unit tests use fakes only for connection error mapping and
SQL construction. All storage semantics require a real Doris integration test.

Add a `trackio storage migrate --from sqlite --to doris` command. It reads a
stable snapshot of every project, writes deterministic IDs in bounded batches,
and records a local migration receipt with source paths, source hashes, target
schema version, per-table counts, and timestamps. Re-running the command must
be idempotent. A `--verify-only` mode compares project/run identities, table
counts, artifact manifest digests, sampled metric values, and trace external
IDs without writing.

In `../ai-infra`, deploy a candidate Trackio service configured for Doris
without replacing the current SQLite service. Run the full conformance and
failure suite. Then pause Trackio writes, snapshot the current SQLite directory,
run migration and verification, switch the service environment to `doris`,
restart, and run a canary. Retain the SQLite snapshot until the acceptance
window closes. Rollback stops the Doris-backed service, restores the previous
environment and image, and resumes the unchanged SQLite snapshot.

Finally, point current-source Observatory at the Doris-backed Trackio URL and
verify known historical and new canary runs. Update the fork ledger,
`docs/tooling/trackio/README.md`, the exact Trackio dependency pin, `uv.lock`,
deployment image digest, and the append-only execution log with measured
evidence.

## Concrete Steps

From `../trackio`, run focused development tests first:

    uv sync --extra dev --extra spaces
    TRACKIO_DATABASE_ENGINE=sqlite uv run pytest tests/unit -q
    TRACKIO_DATABASE_ENGINE=turso uv run pytest tests/unit -q
    uv run pytest tests/integration/test_doris_storage.py -q
    uv run ruff check .

Start a disposable Doris-backed Trackio server with secrets injected by the
operator environment:

    TRACKIO_DATABASE_ENGINE=doris \
    TRACKIO_DORIS_HOST="$TRACKIO_TEST_DORIS_HOST" \
    TRACKIO_DORIS_PORT="$TRACKIO_TEST_DORIS_PORT" \
    TRACKIO_DORIS_DATABASE=trackio_test \
    TRACKIO_DORIS_USER="$TRACKIO_TEST_DORIS_USER" \
    TRACKIO_DORIS_PASSWORD="$TRACKIO_TEST_DORIS_PASSWORD" \
    uv run trackio server --host 127.0.0.1 --port 7860

Run the migration in verify-first form:

    uv run trackio storage migrate \
      --source /srv/ai-control/trackio \
      --receipt /srv/ai-control/migrations/trackio-doris.json \
      --dry-run

After the dry run reports no unsupported records, run the same command without
`--dry-run`, then run it with `--verify-only`. Commands that touch the shared
service must be documented and executed from `../ai-infra`; secrets must be
redacted from captured output.

## Validation and Acceptance

The engine-selection test starts Trackio three times with `sqlite`, `turso`,
and `doris`. Each valid value starts successfully. An unknown value fails at
startup with the supported values and no secret content.

The real Doris conformance test creates two projects and overlapping run names.
It writes configs, at least 100 scalar and structured metric rows, system
metrics, standard and Verifiers traces, alerts, a two-version artifact with an
alias, and input/output lineage. It reads them through the Trackio HTTP API,
renames one run, deletes another, restarts Trackio, and reads the retained
records again. Project A must never return Project B records.

The retry test sends duplicate `log_id`, `alert_id`, trace external ID,
artifact manifest, alias, and lineage operations. Public counts and identities
remain unchanged. The concurrency test uses at least four writers on distinct
runs plus two writers retrying one run and proves no lost acknowledged batch.
An interrupted or timed-out write is retried and reconciled before the client
reports success.

The artifact test uploads and downloads a bounded real blob through Trackio.
Doris contains only its metadata, manifest digest, alias, and lineage; the byte
store contains the verified bytes. Trackio returns the original SHA-256 after
a server restart.

Migration acceptance requires exact project and run identity sets, exact
config/trace/alert/artifact/lineage counts, exact artifact manifest digests,
and metric/system-metric counts per run. It also compares deterministic samples
from the beginning, middle, and end of every time series. Any mismatch blocks
cutover.

Observatory acceptance uses its normal Trackio reader, not a Doris-specific
adapter. A migrated training run and a new canary run both resolve to their
job-aware views with metrics, system metrics, traces, artifacts, and lineage.
This proves Observatory is reading the remote Doris-backed Trackio server.

Backup acceptance takes a Doris backup or snapshot according to the deployed
cluster procedure, restores it into an isolated database, starts a disposable
Trackio instance against the restore, and repeats the read-only conformance
checks.

## Idempotence and Recovery

Schema migrations use a version table and `CREATE TABLE IF NOT EXISTS`.
Application writes use stable identifiers so retrying cannot create duplicate
logical records. Migration receipts are append-only and source snapshots are
immutable during a migration.

Cutover is reversible until the operator explicitly retires the retained
SQLite snapshot. A failed candidate deployment leaves the current service
unchanged. A failed migration may be retried into a clean test database or
replayed idempotently into the same target after the cause is fixed. Do not
delete SQLite databases, Doris data, Trackio artifact bytes, or migration
receipts during qualification.

## Artifacts and Notes

The initial audit found that adding `doris` only to
`trackio/database.py:ENGINE` would be incorrect. That module exports SQLite
error types and connection behavior, while `SQLiteStorage` performs SQLite
pragmas, uses filesystem locks, and exposes SQLite-specific raw queries. The
implementation must therefore proceed at the logical storage-operation level.

## Interfaces and Dependencies

In `../trackio/trackio/storage.py`, provide:

    def get_storage() -> type[TrackioStorage]: ...
    Storage = get_storage()

`TrackioStorage` is a typed protocol grouped into run evidence, queries,
mutation, project metadata, and artifact metadata capabilities. The selected
class exposes the existing static-call shape during migration so server routes
can move without changing their request handlers.

In `../trackio/trackio/doris_storage.py`, provide `DorisStorage` implementing
that protocol. In `../trackio/trackio/doris_schema.py`, provide versioned schema
migration functions. In `../trackio/trackio/cli.py`, add the
`trackio storage migrate` command and verification mode.

The Doris client dependency must support Python 3.10+, MySQL prepared
statements or safely bound parameters, TLS configuration, timeout control, and
pooling. Add it as a compatible range in `../trackio/pyproject.toml` and lock
its exact resolution in `../trackio/uv.lock`.

Revision note (2026-07-26): Created after correcting the requirement from a
Trackio-to-Doris projection to native Doris engine support. The plan separates
database metadata from artifact bytes, preserves SQLite/Turso, and requires a
real migration plus Trackio/Observatory end-to-end proof.
