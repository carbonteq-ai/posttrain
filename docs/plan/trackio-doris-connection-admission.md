# Bound Trackio Doris connections and protect run finalization

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Trackio currently lets HTTP readers and background evidence importers independently open Doris connections. Under retained-trace ingestion and Observatory traffic, the production Trackio user reached its 100-connection limit exactly when a completed training run attempted to commit final artifact metadata. Training reached step 180, but evidence finalization failed and the provider reported the job as failed. After this work, one Trackio process will reuse a small bounded set of Doris connections, reserve admission for artifact finalization, reject overload predictably instead of exhausting Doris, and expose configuration that keeps production below a 32-connection database safety ceiling.

The change spans three repositories. `/home/hammad/projects/trackio` owns the generic Doris connection manager. `/home/hammad/projects/ai-infra` owns production worker counts and the Doris user ceiling. This Posttrain repository owns the immutable Trackio package pin, fork ledger, release notes, and framework release. The frozen product baseline does not change: `docs/post-training/06-observation-and-lineage.md` already requires immutable artifacts, tracking health, typed failures, and durable evidence. This plan changes implementation reliability only.

## Progress

- [x] (2026-08-24) Confirmed the incident from workload, Trackio, and Doris logs: step 180 completed; `/artifact_log` failed while the Trackio Doris user was at 100 of 100 connections.
- [x] (2026-08-24) Created clean worktrees from Trackio dev18, Posttrain `origin/main`, and ai-infra `origin/main`; existing dirty checkouts remain untouched.
- [x] (2026-08-24) Implemented a bounded reusable Doris connection manager with reserved control admission, physical-age recycling, and deterministic shutdown/reset behavior.
- [x] (2026-08-24) Added focused concurrency, timeout, broken-connection, configuration, physical-age recycling, and artifact-reservation tests.
- [x] (2026-08-24) Ran Trackio unit/static/build validation and exercised local Trackio plus local Observatory against isolated real Doris and artifact services under concurrent load. Production reads remain blocked by the already-saturated old deployment until cutover.
- [x] (2026-08-24) Committed and pushed Trackio, published immutable dev19 artifacts with hashes, published to the development index, qualified, and promoted the identical files to stable.
- [ ] Update ai-infra declarations and Posttrain's exact package/source/hash pins, validate both repositories, merge, and publish the next Posttrain release without a GPU qualification job.

## Surprises & Discoveries

- Observation: The failure was not an optimizer, GPU, disk, or dstack failure. The final training metric reported epoch 7.5, which maps to step 180, before Trackio artifact flushing raised.
  Evidence: The workload traceback ends in `trackio.Run.flush_artifacts()` and the provider changed to failed only after the process exited with status 1.
- Observation: Production remains susceptible after a container restart.
  Evidence: Current Trackio logs repeatedly contain Doris error 1203, `Reach limit of connections. Total: 1024, User: 100, Current: 100`, plus error 2013 query timeouts.
- Observation: Production concurrency is not coordinated with database capacity.
  Evidence: ai-infra configures 64 API workers and 8 inbox workers, while `DorisStorage._connection` creates one new PyMySQL connection per operation and Doris permits 100 connections for the Trackio user.
- Observation: Application workers do not need to equal database connections once admission is bounded.
  Evidence: the production-shaped local test uses 32 API workers and 16 inbox workers over a 16-connection pool; only 14 ordinary operations can hold a connection while 2 slots remain reserved for artifact finalization.
- Observation: The old production Trackio deployment cannot currently serve as a clean pre-cutover load target.
  Evidence: local Observatory readiness against the production Doris user returned error 1203 with 100 of 100 connections already consumed before the candidate generated traffic.
- Observation: A four-connection stress configuration with a two-second checkout timeout intentionally surfaced backpressure under 16 concurrent Observatory callers.
  Evidence: 18 of 80 requests timed out at pool admission. The exact production configuration (16 pool, 2 reserve, 10-second timeout) completed 80 of 80 requests with p50 0.622 seconds, p95 1.605 seconds, and max 1.977 seconds.
- Observation: Sixteen inbox workers are useful without widening database concurrency.
  Evidence: 16 simultaneous clients uploaded independent 20-step fragments while Observatory completed 160 of 160 concurrent requests (p95 2.84 seconds, max 3.37 seconds); all 16 fragments materialized within 30 seconds and server logs contained no importer error.
- Observation: Artifact finalization retains admission during read pressure.
  Evidence: a real S3-backed `v0` artifact version committed while Observatory completed 80 of 80 concurrent requests; p95 was 2.62 seconds and max was 3.635 seconds.
- Observation: Doris retains timed-out sessions in per-user accounting after the client has discarded its socket.
  Evidence: after production cutover the new process never owned more than 16 pooled sockets, but repeated 30-second import timeouts left 29 Doris process-list records and consumed most of the 32-session safety budget.
- Observation: Stale pre-cutover sessions must be reconciled once when lowering an already-exhausted user limit.
  Evidence: the old server left 90 disconnected production sessions and 14 candidate sessions. They were terminated by exact Doris user and disconnected state before the final restart; no database, table, run, or artifact data was deleted.
- Observation: Large native-trace inserts amplify connection occupancy.
  Evidence: Doris audit records at the incident window include roughly 19 MB compressed insert fragments and about 101 MB peak FE memory per insert.

## Decision Log

- Decision: Build Trackio dev19 from the immutable dev18 commit currently consumed by Posttrain, not from the older fork `origin/main` tip.
  Rationale: Dev18 is the deployed and framework-pinned behavior. Starting from the older fork main would silently discard released trace, artifact, and bounded-read capabilities.
  Date/Author: 2026-08-24 / Codex
- Decision: Use a small application-owned connection pool plus admission reservation; do not treat Doris's user limit as a pool.
  Rationale: Reuse reduces handshake churn, bounded checkout creates backpressure, and reserved control capacity prevents bulk reads or trace imports from starving artifact finalization.
  Date/Author: 2026-08-24 / Codex
- Decision: Keep the production Doris user ceiling above the Trackio pool but far below 100.
  Rationale: The database limit is a final safety boundary and must leave capacity for qualification and operator access. It is not normal operating concurrency.
  Date/Author: 2026-08-24 / Codex
- Decision: Run 16 durable inbox workers behind the shared pool.
  Rationale: More workers can decode and queue independent fragments without increasing physical Doris concurrency; the pool, not thread count, owns the 14 ordinary plus 2 reserved connection budget.
  Date/Author: 2026-08-24 / Codex
- Decision: Give Doris reads and writes 120 seconds while retaining a 10-second connect timeout.
  Rationale: retained fragments can legitimately take longer than 30 seconds to materialize under load. Keeping the pooled socket alive avoids rapid disconnect/replacement churn in Doris user accounting; connection establishment should still fail quickly when the service is unreachable.
  Date/Author: 2026-08-24 / Codex
- Decision: Skip GPU release qualification for this framework release.
  Rationale: The user explicitly allowed skipping qualifying jobs; acceptance instead requires real Doris, artifact, and Observatory load verification for the changed boundary.
  Date/Author: 2026-08-24 / Codex

## Outcomes & Retrospective

Trackio source commit `b0f2ceb042dc741b458634efb5981604ead97702` is published as `carbonteq-v0.31.5.post14.dev19`. Source validation records 427 unit tests passing with 6 skips, 5 real Doris integration tests passing, Ruff passing, a successful frontend production build, and successful wheel/sdist construction. The wheel SHA-256 is `e92436100adc657993f7fc1e51008a4c5f6a63786c97afb2899962c872c8d5ea`; the sdist SHA-256 is `b310e9ed1220ce1327229bf8370b90efb6cbee578c42866435cb8c10c7c99501`. Development publication workflow `32739171972` and stable promotion workflow `32739265154` both succeeded.

ai-infra PRs 8 and 9 merged as `f00d04907fdccf3231d61b7016b701b0b99a5240` and `991176a96f6aa1b8e5d0a9908631f407144fc39d`. Both production and candidate now report Trackio dev19 and the declared 32 API workers, 16 write workers, 16-connection pool, 2-slot reserve, 10-second connect timeout, and 120-second read/write timeouts. The production scalar plus S3 artifact round-trip passed with producer `a4220baf6f794175b1f8d5a0a3809f7e`, consumer `8e1688cf062541e589859d5bf2857a47`, two indexed rows, and byte-identical artifact readback. Posttrain v0.3.24 publication remains in progress.

## Context and Orientation

`trackio/doris_storage.py` contains `DorisStorage._connection`, the context manager used by Doris reads and writes. It currently calls `pymysql.connect` for every operation and closes the socket afterward. `trackio/server.py` runs the durable inbox scanner and its write workers. `trackio/asgi_app.py` runs HTTP API calls in a separate thread executor. A connection pool is a bounded owner of reusable database sockets; checkout means temporarily borrowing one socket for an operation and return means making it idle for another operation.

Artifact upload bytes already use Trackio's configured local or S3-compatible object store. Doris stores artifact identities, manifests, and run-lineage links. `DorisStorage.commit_artifact_version` is therefore a control-plane operation: it makes already-verified artifact bytes visible as durable run evidence. It needs reserved admission even when ordinary readers or trace writers are busy.

The ai-infra production declarations are `ansible/roles/control/files/compose.yml` for Trackio worker and pool environment variables and `ansible/playbooks/trackio-doris-candidate.yml` for the production and candidate Doris users. Posttrain records the exact maintained-fork package in `packages/tracking-trackio/pyproject.toml`, `release/forks.toml`, generated runtime profiles/locks, `uv.lock`, `docs/tooling/trackio/README.md`, and release documentation.

## Plan of Work

In Trackio, introduce an internal connection manager in `trackio/doris_storage.py` or a narrowly owned sibling module. It will lazily create PyMySQL connections up to a configured maximum, reuse healthy idle connections, discard connections that fail health checks or operations, recycle old connections, bound checkout waiting, and close all idle sockets during process shutdown or test reset. A shared condition or semaphore will cap all checked-out connections. Ordinary operations will stop below the cap by a configured reserve; artifact commit/finalization operations may use the reserved slots. Defaults will be conservative for a single Trackio process and environment variables will validate impossible combinations at startup.

Trackio tests will prove that concurrent callers never exceed the physical maximum, ordinary callers cannot consume reserved slots, control callers progress while ordinary work is queued, timed-out checkout fails with a typed retryable error, dead connections are replaced, returned connections are reused, and settings changes/reset do not leak sockets. Existing Doris provider tests must continue to pass.

In ai-infra, configure the pool and reservation explicitly, reduce API and importer concurrency to values that cannot overwhelm the pool, and set the Doris production user maximum to a protective ceiling of 32. Candidate and production users remain distinct. Add or update infrastructure tests so the relationship between worker counts, pool size, reserve, and database ceiling is executable rather than prose-only.

Local integration will run the candidate Trackio server and Posttrain Observatory on loopback while using the existing Doris database and artifact endpoint. It will issue concurrent bounded metric, system-metric, trace-summary, and artifact-metadata operations. Acceptance requires successful artifact finalization while general requests saturate their admission lane, no Doris 1203 errors, a measured connection high-water mark at or below the configured pool maximum, and responsive bounded Observatory requests. This is a non-destructive read/load test except for a uniquely named disposable Trackio qualification project; any synthetic records will be removed through Trackio's project-scoped preview/apply interface after receipts are retained.

After validation, commit and push Trackio first. Build wheel and sdist once, record SHA-256 hashes, create an immutable `carbonteq-v0.31.5.post14.dev19` GitHub prerelease, publish those exact assets to `carbonteq/dev`, clean-install and qualify them, then promote the same hashes to `carbonteq/stable`. Only then update and commit Posttrain and ai-infra. Posttrain will use its protected candidate/final workflows, with the GPU job skipped only through the documented release input and with exact-SHA Quality evidence retained.

## Concrete Steps

Work in `/home/hammad/projects/worktrees/trackio-doris-pool`:

    uv sync --extra dev --extra spaces
    uv run pytest tests/unit/test_storage_provider.py tests/unit/test_doris_schema.py tests/unit/test_doris_metric_reads.py -q
    uv run ruff check .
    uv run pyright
    uv build

Work in `/home/hammad/projects/worktrees/ai-infra-doris-pool`:

    uv sync --locked
    uv run pytest -q
    uv run ansible-playbook --syntax-check -i ansible/inventory/generated.yml ansible/playbooks/site.yml

Work in `/home/hammad/projects/worktrees/posttrain-doris-pool` after the immutable Trackio assets exist:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Exact publication commands and workflow run IDs will be added here after the dev19 source commit and hashes exist; publication must never rebuild accepted assets.

## Validation and Acceptance

Source acceptance requires all focused tests plus repository static checks to pass. Integration acceptance requires a fresh local Trackio/Observatory pair to read the production-backed Ambient run under concurrent metric and system-metric traffic, while a disposable project completes an artifact metadata commit. Doris `SHOW PROCESSLIST` and Trackio pool diagnostics must prove the configured ceiling. Trackio and Doris logs must contain neither error 1203 nor a pool-capacity breach during the test window.

Publication acceptance requires the GitHub release assets, development index, stable index, Trackio import version, Posttrain lock, fork ledger, and runtime profiles to agree on version, source revision, filenames, and SHA-256 hashes. Framework publication is complete only when the final GitHub release is non-draft and non-prerelease, the stable private index exposes the exact candidate bytes, and the final tag points to merged source.

## Idempotence and Recovery

Worktrees and test projects have unique names. Re-running source tests and builds is safe. A failed Trackio publication must use a new immutable version unless the repository's retained-candidate retirement procedure proves no accepted or stable bytes exist. A failed Posttrain candidate is never overwritten after acceptance. ai-infra deployment must retain the prior compose file and Trackio image digest so its guarded playbook can roll back without changing Doris data.

## Artifacts and Notes

Incident evidence retained for this plan:

    Doris user property: max_user_connections = 100
    Trackio error: (1203, 'Reach limit of connections. Total: 1024, User: 100, Current: 100')
    Provider transition: running -> failed at 2026-08-24T07:47:50Z
    Training terminal metric: epoch 7.5, reward 0.8438

## Interfaces and Dependencies

The pool remains private to Trackio; it does not change the public SDK or storage-provider contract. Configuration names will use the `TRACKIO_DORIS_` prefix and will be documented in `CARBONTEQ_FORK.md`. PyMySQL remains the database driver. If a new pooling dependency is considered, it must be justified against a small internal implementation and locked in Trackio's package metadata; the preferred implementation avoids a new runtime dependency.

Revision note (2026-08-24): Created this plan from live incident evidence and current main/dev18 repository state so implementation and publication can proceed without relying on chat history.
