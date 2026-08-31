# Trackio fork and maintenance

The platform uses [`carbonteq-ai/trackio`](https://github.com/carbonteq-ai/trackio),
an additive fork of upstream Trackio. Workspace packages keep the normal
`import trackio` API. The current framework dependency is
`carbonteq-trackio==0.31.5.post14.dev19`, built from immutable fork commit
`b0f2ceb042dc741b458634efb5981604ead97702`. Its wheel
(`e92436100adc657993f7fc1e51008a4c5f6a63786c97afb2899962c872c8d5ea`) and
sdist (`b310e9ed1220ce1327229bf8370b90efb6cbee578c42866435cb8c10c7c99501`)
were released manually as `carbonteq-v0.31.5.post14.dev19`, published
unchanged to `carbonteq/dev` by Posttrain workflow `32739171972`, then promoted
byte-for-byte to `carbonteq/stable` by workflow `32739265154` after real-Doris
read, write, and artifact-finalization qualification. The development suffix is
part of the immutable version; it does not make the stable-index publication mutable.
This release adds generic typed trace facts: Posttrain supplies the versioned
scalar projection from native Verifiers records, Trackio persists facts on the
trace row plus a dynamic reward-component relation, and readers request only
approved aggregate dimensions and measures.
The release also bounds and reuses Doris connections, reserves two pool slots
for artifact metadata finalization, and turns pool pressure into retryable
backpressure instead of exhausting the Doris service-user connection limit.
It also repairs stale completed resumable-upload receipts: when retention has
removed the referenced content-addressed blob, the server reopens the
idempotent session and accepts the bytes again instead of failing completion
with HTTP 409.
`0.31.5.post4` on
`pypi.lan` is permanently skewed (metadata post4, import post3) and must not be
installed.

The shared Doris database changes directly from schema v1 to v2 through an
explicit backup-gated migration; every deployed Trackio server must be upgraded
as part of that separate service operation. Fork workflows do not build or
publish releases: each fork release is manually tagged, its assets are checked,
and the exact bytes are manually promoted before Posttrain updates its pin.

Dev4 extends the read contract to reward-component contribution, score, and
weight. A request may target one exact component or expand across component
name/source-kind groups; component coverage remains distinct from the count of
matching traces. Scalar and component aggregates are separate requests, which
prevents multi-component traces from multiplying scalar metrics.
The producer-side `Run` exposes the same aggregation contract, so a job can
verify its retained facts without using a Trackio internal read object.

Dev8 adds the client ordering boundary for an asynchronously queued native
trace followed by synchronous fact enrichment. It sends the queued source log
batch before the dependent upsert while holding the client lock; retry remains
only a defense for server-side visibility, not the normal ordering mechanism.

Dev9 completes the required service-side boundary: the Doris service commits a
batch containing a native `traces/...` value before returning its write
acknowledgement. Scalar-only metric batches remain asynchronous, so rollout
throughput is not globally serialized.

Dev10 completes the client-side boundary for bridges that explicitly call
`Run.flush()` before enrichment. A remote flush now sends its queued metric
batch to the service rather than only checkpointing it in the local retry
buffer, so a trace-fact update cannot overtake its native parent trace.

Dev17 supersedes the normal-training behavior of Dev8--Dev10. The old
source-before-fact repair turned every trainer enrichment into synchronous
Doris I/O, so a slow write could fail training at the 60-second remote-client
timeout. In async-Doris mode, native trace batches now use the durable server
inbox just like ordinary telemetry. Later trace-fact updates use the separate
`/enqueue_trace_facts` inbox API; the importer applies source metrics before
facts and leaves an early fact durable for retry. Explicit `Run.flush()` and
`Run.upsert_trace_facts()` retain their synchronous read-after-write meaning
for maintenance callers. Posttrain's training adapter uses only the new
enqueue operation, so optimizer progress is not coupled to Doris latency.

Dev18 adds storage-applied metric step windows
and named JSON-field projection. Observatory can request only the selected
metric names and logical step interval in bounded pages; Doris no longer sends
the full per-step JSON object to Trackio for those reads. Requested system
metrics use the same named-field projection while Trackio retains its existing
3,000-sample run-wide bound. The opt-in `drop_empty` history argument applies a
requested-key existence predicate before pagination, so sparse selected-series
pages contain observations rather than unrelated timestamp/step-only rows.
Posttrain preserves resumed/replayed points by
resolving their recorded logical `source_step` before presenting a requested
window. The fork release and stable package publication are complete; service
deployment and production timing remain separate operational gates.

Dev11 adds the bounded generic `/bulk_upsert_trace_facts` endpoint. Posttrain
uses it only for one already-projected historical page at a time; all facts
remain individually trace-keyed, idempotent, and receipt-backed. This avoids
one network round trip per retained trace without making Trackio interpret
Verifiers payloads or model templates.

Dev12 executes that validated page in one Trackio storage transaction and
resolves its trace keys in one bounded query. The public fact schema and
per-trace receipts are unchanged; the change makes large retained backfills
fit comfortably inside the request window.

Dev13 uses set-oriented Doris writes for new historical source projections,
while existing or replacement projections retain their conservative transition.
That makes the large retained backfill practical without weakening replay
semantics.

Dev14 recognizes the empty projection identity emitted by the deployed Doris
schema as absent, so these historical rows reliably take that fast path.

Dev15 batches the fresh Doris scalar-row updates that remain after component
staging. The retained-data qualification used these exact candidate bytes to
project 21,572 historical GRPO traces into 100 optimizer-step buckets and 12
historical OPD traces into one optimizer-step bucket. The payload-free model
inventory observed `models/qwen3.5-2b-sft-10k-json@lora-v0` for that GRPO run,
`models/gemma4-e2b-it@bf16` for that OPD run, and `Qwen/Qwen3.5-0.8B` for the
16-trace evaluation qualification. No null or unsupported model bucket was
present in those retained populations; thinking-token support remains governed
by Posttrain's versioned projector rules rather than by Trackio model policy.

Post10 keeps `local` as the safe default and adds a generic S3-compatible
artifact backend. With the S3 backend, Trackio issues short-lived multipart
URLs and the producing client writes parts directly to the configured bucket;
Trackio completes and verifies the object by SHA-256 before accepting the
artifact manifest. The endpoint must be reachable by producing clients, and
storage credentials remain server-only. The canary and migration gates passed
on 2026-08-07; the shared service now uses the
`trackio-artifacts/production` prefix. The original local CAS remains retained
for rollback, and its deletion is a separate operator-approved retention gate.

A 2026-08-08 two-step DAPO diagnostic exposed one remaining mixed-version
compatibility defect. A post8 job client used the legacy resumable upload route
against the S3-backed post10 service; that route verified a 11,952,582-byte
Verifiers trace blob into the server's local CAS, while artifact-manifest
validation correctly queried RustFS. The run therefore failed during evidence
publication after both optimizer updates had completed. The exact blob was
copied and SHA-256-verified in RustFS for preservation. The unreleased fork
repair makes the legacy completion route publish through the configured
artifact store and recover already-completed local sessions. A new client
release in every job image is still required before the next production run.

Post8 exposes authenticated digest-bound run and project purge. Exact provider
run ids preview with consumer-aware blockers, and apply accepts only the
returned SHA-256 digest. The provider transaction removes selected runs,
run-artifact links, unlinked artifact versions, and unreferenced CAS blobs with
equivalent SQLite and Doris semantics. Stale previews return actionable HTTP
400 messages through the remote client. Disposable live SQLite/Doris checks and
the framework's three-run cross-plane interruption/resume fixture passed on
2026-08-02; sanitized receipts live in
`release-evidence/cross-plane-purge/`.

## Unreleased inbox-throughput repair

The current candidate work addresses a production lag mode in the Doris
fragment importer. `TRACKIO_ASYNC_DORIS_WRITES=true` means durable JSONL
fragments plus background threads; it is not asyncio-based concurrent Doris
I/O. The repair changes startup to serve HTTP immediately, uses one scanner to
claim bounded batches, groups records before synchronous Doris writes, and
gives scalar step/reward/event fragments a separate priority lane from large
rollout traces. The complete native Verifiers trace artifact remains the replay
authority. Do not describe this repair as deployed until its immutable Trackio
commit, wheel, real-Doris backlog replay, and Observatory readback are
qualified.

## Background artifact publication

Post12 adds an opt-in bounded client-side artifact queue. A producer may call
`Run.log_artifact(..., background=True)` and receive an artifact with a stable
submission id and `pending`/`uploading`/`committed`/`failed` state. The producer
must call `wait()` or `Run.flush_artifacts()` before using the version. The
existing synchronous API remains the default; Posttrain opts in only after the
client capability is present and still performs an explicit final drain.

## Distribution transition

The currently deployed fork builds as `carbonteq-trackio==0.31.5.post1`,
preserving the
`trackio` import package and console command. Its wheel and source distribution
build successfully, a clean Python 3.12 environment reports matching
distribution/import versions, 33 focused deployment and storage tests pass,
the CPU-safe unit suite reports 326 passed and 2 skipped, and a
`carbonteq-v*` tag workflow is ready for PyPI Trusted Publishing. Five
hardware tests remain a separate GPU release gate because the fork development
environment does not install PyTorch or `nvidia-ml-py`.

The distribution is not yet published. Until its fork changes are committed,
pushed, and accepted by PyPI, framework metadata must retain the immutable Git
pin and consumers must not be told that the registry artifact is reproducible.
The packaging candidate is committed and pushed at
`c47bcc0e0a15030ec6f20cdc7d294a820ab617b2`; it still requires review, merge,
and registry publication through
[`carbonteq-ai/trackio#3`](https://github.com/carbonteq-ai/trackio/pull/3).
The PR merged as `c5072198b3b1556d31ed96ffc246a03f65418ab8`. The repository
has the `pypi` GitHub Actions environment and the tag-only release workflow is
present on `main`. After PyPI pending-publisher configuration and successful
publication, update `posttrain-tracking-trackio`, the root lock, and the
external-consumer test in one change, then remove the consumer's explicit Git
requirement.

The published candidate is `carbonteq-trackio==0.31.5.post2` at immutable
commit `e2784c1536b20832f3937d7589c10bce76df4b43`. It adds a self-hosted
resumable artifact transport without changing the public `log_artifact` or
`use_artifact` APIs. Artifact blobs are sent in bounded 8 MiB chunks,
acknowledged idempotently, resumed after interruption, and verified by complete
SHA-256 before an artifact version may be committed. A legacy server remains
compatible for files up to 32 MiB; larger uploads fail closed with an upgrade
message.

The focused qualification currently includes separate client and server
processes, interrupted-client resume, server-app restart, digest corruption,
authorization, incomplete-session expiry, and a 512 MiB round trip. The
512 MiB gate completed in 64 chunks with a measured server RSS increase of
approximately 15 MiB, demonstrating that the server did not buffer the whole
file. This evidence is local to the uncommitted candidate until its exact
commit is published and deployed.

The next fork release is `carbonteq-trackio==0.31.5.post3`, retaining the same
upstream base and the `post2` resumable artifact transport while adding native
Apache Doris as a first-class storage engine. It is source- and
integration-qualified but not yet committed or published. Migration reads a
SQLite backup snapshot including committed WAL rows, verify-only is
non-initializing, reconciliation compares canonical count and SHA-256 evidence
for all ten authoritative tables, artifact versions and lineage timestamps are
preserved, same-name run selection matches SQLite, and schema startup refuses
partial, older, and newer managed schemas.

Qualification now includes 93 focused storage, schema, migration, artifact, and
provider-parity tests, 376 non-hardware unit tests with three skips,
repository-wide Ruff, a correctly named `post3` wheel, all four real-Doris
cases against the existing candidate database with exact synthetic-row
cleanup, and all four cases against a freshly bootstrapped disposable database
and user followed by complete teardown. The parity pass covers metric time
windows and filter precedence, canonical run-record authority and ordering,
nullable legacy identities, tab payload and trace classification, artifact
relog identity preservation, and legacy-link folding and deduplication.
Publication, immutable framework pinning, corrected-service deployment,
retained-run reconciliation, and Observatory readback remain open.

The same release candidate also moves the bundled dashboard to Vega 6.3.1,
Vega-Lite 6.4.3, and Vega Embed 7.1.0, updates Svelte/Vite/Vitest within their
supported lines, emits Vega-Lite v6 schemas, preserves explicit canvas
rendering, and escapes generated axis-label lookup expressions. Both the full
and production-only npm audits report zero vulnerabilities; 48 frontend tests,
full frontend lint, a warning-free production build, a clean `npm ci` wheel
build, and 17 Python frontend/distribution tests pass. Two hostile-label tests
compile and execute representative chart values through Vega's non-DOM
renderer. Browser pixel comparison remains a deployment gate rather than a
source-security claim.

## Repository contract

```text
origin    git@github.com:carbonteq-ai/trackio.git
upstream  https://github.com/gradio-app/trackio.git
```

The initial extension commit is
`a79040fd9cecbb5881cda8d4c1961a55aeb7600f`, based on upstream commit
`438cb28d2c82c7b7d42431e45d5677a8cc90eb77`.
The current Turso and Verifiers UI implementation is pinned at
`9cf451c020dd8efafc1b518168f807923522b3a8`.

The fork adds `trackio.VerifiersTrace`, additive trace persistence fields,
Turso as the default embedded SQL driver, a separate Verifiers rollout UI, and
resumable self-hosted artifact transport. Apache Doris is implemented and live
as a first-class database engine candidate. The corrected `post3` source is not
yet released from an immutable fork commit or deployed.
It does not replace Trackio's SDK, HTTP API, artifact implementation, Parquet
interchange, standard `Trace`, or standard trace UI. Trackio's artifact store
defaults to its local CAS, but the server can select any S3-compatible endpoint.
New clients receive presigned multipart part URLs and upload large blobs
directly; Trackio verifies the final SHA-256 before committing metadata. Set
`TRACKIO_DATABASE_ENGINE=sqlite` when the stdlib SQLite fallback is required.

## Updating from upstream

In a sibling clone of the fork:

```bash
git fetch upstream
git switch main
git pull --ff-only origin main
git switch -c maintenance/upstream-YYYY-MM-DD
git merge --no-ff upstream/main
uv sync --extra dev --extra spaces
uv run pytest tests/unit -q
git push -u origin maintenance/upstream-YYYY-MM-DD
```

Open a pull request into the protected fork default branch. Resolve conflicts by
preserving upstream behavior and reapplying the additive Verifiers contract.
Before merge, run focused serialization, migration, idempotence, filtering,
search, pagination, and Parquet tests in Turso mode; run the compatibility suite
in SQLite mode; and run frontend unit/build tests plus the Playwright Verifiers
UI flow. GPU hardware tests require the optional CUDA test dependencies and a
visible device and are reported separately when unavailable.

Update `CARBONTEQ_FORK.md` in the fork with the new upstream base and
distribution version for every release. After merge and registry publication,
update `packages/tracking-trackio/pyproject.toml`, this page, and the root
lockfile.

## Ownership boundary

- Trackio fork: database-driver boundary, trace types, persistence, query API,
  migrations, Parquet, standard UI compatibility, Verifiers rollout UI, and
  server-managed artifact transport.
- `packages/eval`: Verifiers validation, JSONL tailing, batching, final retry,
  and synchronization health metrics.
- Verifiers native output: canonical trace record and replay/debug authority.
- `apps/observatory`: read-only consumer of Trackio's normalized run, trace,
  and artifact evidence through `posttrain-tracking-trackio`.

## Project developers (artifact handoff)

After a train run publishes a model artifact, pin the immutable Trackio `vN`
as a project catalog `ModelVariant` (`artifact.kind: trackio`) and bind that
id on the next work package. There is no in-YAML `from_job` wire.

How-to: [getting-started §9](../../getting-started.md#9-pass-one-jobs-model-into-the-next) ·
DX: [trained model handoff](../../developer-experience.md#trained-model-handoff-produce--pin--rebind) ·
Storage: [operations/dstack-trackio/object-storage.md](../../operations/dstack-trackio/object-storage.md).

Rust, Tokio, and direct job access to object storage remain deferred. The
native Doris engine passed real provider, content-reconciled migration,
clean-schema bootstrap, backup/restore, shared-endpoint, and existing
Observatory qualification. Its exact evidence and remaining immutable-release
and corrected-deployment gates are recorded in
`docs/plan/trackio-apache-doris-engine.md`. Trackio continues to own its backing
artifact bytes and credentials. The first Doris release supports one Trackio
server process: cross-process artifact version allocation requires a new
staged or optimistic protocol before high availability may be claimed.
Unscoped raw project SQL also remains deliberately unsupported because a shared
multi-project Doris database cannot safely provide SQLite's project-local SQL
semantics.
