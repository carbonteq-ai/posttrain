# Add a verified RustFS artifact backend to Trackio

This plan is a living execution document. It follows the execution-plan
requirements in `docs/templates/PLAN.md`; keep `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current as the
implementation advances.

## Purpose / Big Picture

Trackio currently stores content-addressed artifact blobs on the Trackio
server's `/data` volume. RustFS is already deployed as an S3-compatible service,
but no Trackio artifact operation uses it. After this change, an operator can
select any S3-compatible endpoint explicitly for the Trackio server. Trackio
will authorize an upload, issue short-lived presigned multipart URLs, let the
producing client upload bytes directly to the bucket, and commit metadata only
after the server verifies the completed object. Downloads may likewise use a
short-lived presigned URL so large bytes do not traverse Trackio. Jobs still
receive only Trackio credentials; S3 credentials remain server-side.

The first production proof is a small artifact containing a model/config file:
it must commit metadata in Doris, place its blob in the configured RustFS
bucket, download byte-for-byte through Trackio, survive a Trackio restart, and
leave no final blob under the local artifact directory. A migration proof must
report source/target counts and digests before local deletion is permitted.

## Progress

- [x] (2026-08-07) Confirmed the live Trackio service has no RustFS/S3 artifact
  settings and writes approximately 5.1 GB under the mounted local CAS.
- [x] (2026-08-07) Confirmed RustFS is healthy and currently contains only the
  separate `execution-results` qualification data.
- [x] (2026-08-07) Mapped the Trackio CAS, upload, download, purge, and Doris
  storage seams that must become backend-neutral.
- [x] (2026-08-07) Add a Trackio artifact-store interface with local and S3-compatible
  implementations, preserving local behavior as the default.
- [x] (2026-08-07) Add a provider-neutral direct-transfer protocol: Trackio creates a
  multipart session, returns presigned part URLs, accepts only completion
  metadata, verifies the object, and commits the manifest; downloads can return
  a presigned GET URL.
- [x] (2026-08-07) Route legacy proxy uploads, direct uploads, presence checks, download,
  and purge operations through the selected artifact store.
- [x] (2026-08-07) Add provider tests using a fake S3 client and a direct-session
  verification test; validate idempotency, digest checks, and failure recovery.
- [x] (2026-08-07) Add a migration/verification command that copies existing local CAS blobs
  to RustFS without deleting the source and emits a machine-readable receipt.
- [x] (2026-08-07) Qualify the generic S3 store and presigned multipart transfer
  against the live RustFS endpoint with a temporary object that was deleted
  after byte/digest verification.
- [x] (2026-08-07) Wire the control deployment template to accept an explicit
  S3-compatible endpoint, bucket, prefix, region, and server-only credentials;
  the default remains `local`.
- [x] (2026-08-07) Add abort-on-client-failure and stale-session cleanup for
  direct multipart uploads so abandoned provider multipart sessions do not
  accumulate indefinitely.
- [x] (2026-08-07) Add an optional worker-reachable presign endpoint so the
  Trackio server can use a private S3 endpoint while producers use a public or
  separately routed endpoint in the signed URLs.
- [x] (2026-08-07) Published `carbonteq-trackio==0.31.5.post10` from commit
  `f717ef438df88bc9cca20ea7e28752f618a8af49`, pinned the internal consumer,
  and deployed the candidate service with a dedicated S3 bucket/prefix while
  the shared service remained on its local backend.
- [x] (2026-08-07) Qualified the canary with direct multipart upload, part
  acknowledgement, candidate restart/resume, server-side SHA-256 verification,
  presigned download, SDK artifact-manifest commit in Doris, and receipt-bound
  project purge. Temporary canary objects were removed afterward.
- [ ] Migrate existing projects, verify artifact downloads and purge behavior,
  then remove local final blobs only after an explicit retention decision.
- [x] (2026-08-07) Commit and publish the Trackio fork, update the exact
  consumer pin and deployment source, and record live qualification evidence.

## Surprises & Discoveries

- Trackio's Doris engine stores artifact metadata and manifests but computes
  storage bytes by walking the local CAS. Evidence: `trackio/doris_storage.py`
  calls `cas.blob_path` and `_directory_bytes`.
- The resumable upload implementation stages chunks under the local artifact
  directory and atomically exposes a local CAS file. The provider seam must
  preserve resumability while allowing finalization to target RustFS.
- The HTTP download route directly returns a local `FileResponse`; it must
  stream from the selected provider without exposing a signed RustFS URL to the
  job or browser.
- Existing purge code derives deleted digests from manifests and unlinks local
  files. RustFS deletion must use the same retained-manifest closure and must
  never delete a blob still referenced by a retained version.
- RustFS is currently used by infrastructure qualification scripts and is not
  yet the authoritative Trackio artifact store. This change promotes it only
  after canary and migration evidence, not merely because the service exists.

## Decision Log

- Decision: Keep `local` as the default backend and require an explicit
  `s3`/RustFS selection on the Trackio server.
  Rationale: Existing SQLite users and unconfigured deployments must retain
  behavior; an accidental provider change could make existing artifacts
  unavailable.
  Date/Author: 2026-08-07 / Codex.
- Decision: Use a server-side S3-compatible client rather than exposing direct
  RustFS URLs or credentials to jobs, but issue short-lived presigned object
  URLs when the selected backend supports them.
  Rationale: The public Trackio API and framework artifact references remain
  provider-neutral, while large producers and consumers avoid sending blob
  bytes through the Trackio control plane. Only the URL and its narrow object
  permission leave the server; long-lived S3 credentials never do.
  Date/Author: 2026-08-07 / Codex.
- Decision: Use S3 multipart upload for direct transfers and keep the existing
  Trackio resumable API as a compatibility/proxy path for local backends and
  older clients.
  Rationale: Multipart URLs work for large model artifacts and third-party
  workers, while the compatibility path makes rollout backwards-compatible.
  Date/Author: 2026-08-07 / Codex.
- Decision: Verify the completed object by streaming it through the Trackio
  server and recomputing SHA-256 before committing the manifest.
  Rationale: S3 ETags are not reliably content SHA-256 values, and trusting a
  client-supplied digest would allow a corrupt object to become a durable
  artifact. The verification pass does not proxy the upload and is bounded to
  one post-upload read.
  Date/Author: 2026-08-07 / Codex.
- Decision: Use content-addressed keys under a project/prefix namespace and
  retain the existing SHA-256 manifest as the integrity authority.
  Rationale: Existing manifests, deduplication, aliases, lineage, and purge
  plans already operate on SHA-256 digests.
  Date/Author: 2026-08-07 / Codex.
- Decision: Migration is copy-and-verify first; deleting local bytes is a
  separate explicit operation after a successful receipt and canary download.
  Rationale: A storage cutover must be recoverable and must not turn a partial
  upload or provider outage into data loss.
  Date/Author: 2026-08-07 / Codex.
- Decision: Allow a separate S3 presign endpoint from the server endpoint.
  Rationale: A Trackio server may have private network access to a bucket while
  a third-party producer can reach only a public or worker-routed address. The
  signed URL must use the producer-reachable address without exposing provider
  credentials or changing server-side storage operations.
  Date/Author: 2026-08-07 / Codex.

## Outcomes & Retrospective

Milestones 1 through 5 are implemented, including the canary deployment and
live qualification, while production remains on local storage. The discovery
phase established the current behavior and the required ownership boundary;
the implementation now keeps that boundary while moving large bytes off the
Trackio control plane. At the migration milestone record test counts, the RustFS
bucket/prefix used (never credentials), migration counts/bytes, and any
remaining local storage. At completion compare the result with the purpose:
provider-neutral Trackio API, verified RustFS bytes, safe migration, and safe
garbage collection.

The production cutover remains intentionally incomplete: the shared Trackio
service has not been switched, no existing artifact has been copied or deleted,
and the current 5.1 GB local CAS remains the rollback source. The canary used
`trackio-artifacts-canary/post10-canary-20260807` and left no retained test
objects after qualification.

## Context and Orientation

The repository `/home/hammad/projects/trackio` owns the generic Trackio server
and client. `trackio/cas.py` defines SHA-256 blob paths and atomic local writes;
`trackio/server.py` implements artifact API endpoints; `trackio/asgi_app.py`
serves a blob by digest; `trackio/resumable_uploads.py` stages chunked uploads;
`trackio/artifact.py` downloads a manifest through the Trackio endpoint; and
`trackio/doris_storage.py`/`sqlite_storage.py` persist artifact metadata and
perform reference-aware cleanup. The Trackio package currently has no S3
runtime dependency.

The repository `/home/hammad/projects/ai-infra` owns deployment. RustFS runs on
`ai-storage` with an S3 endpoint at port 9000. The Trackio service runs on
`ai-control` with `/srv/ai-control/trackio` mounted as `/data`; its compose
environment currently contains Doris and Trackio worker settings but no RustFS
endpoint, bucket, or credentials. Add only server-side variables to this
deployment and keep secrets in the existing protected environment mechanism.

The repository `/home/hammad/projects/rl` owns the framework contract and
operational plan. The canonical post-training observation document requires
provider-neutral artifact references and permits the selected backend to change
behind Trackio. Do not modify the frozen product baseline for this implementation
unless a test exposes a contradiction; update the non-frozen operational and
tooling documentation instead.

## Plan of Work

### Milestone 1: provider-neutral storage seam and transfer sessions

In `trackio/artifact_storage.py`, define a small synchronous server-side
interface for `has`, `put_file`, `get_stream`, `presign_get`,
`begin_multipart`, `presign_part`, `complete_multipart`, `verify`, `delete`,
`iter_project`, and `stat`, plus a configuration loader that validates `local`
and `s3` settings.
Implement `LocalArtifactStore` using the existing CAS paths and
`S3ArtifactStore` using a lazily-created boto3 client with path-style
addressing, endpoint URL, bucket, and an optional key prefix. Keys must be
deterministic from project and digest. Multipart sessions use an opaque
server-side session record, an S3 multipart upload id, and a bounded part count;
the response contains only the object key, part numbers, short-lived presigned
PUT URLs, and required signed headers. Do not make boto3 import-time mandatory
for the local backend.

Add a bounded stream wrapper for S3 downloads and a presigned GET response so
the ASGI route/client can choose direct or streamed delivery without loading
model bytes into memory. Keep provider errors typed so transient S3 failures
become retryable API errors rather than 404s.

### Milestone 2: route all server operations through the seam

Change `server.py` artifact presence checks and legacy bulk uploads to use the
selected store. Add direct-transfer endpoints next to the existing
`artifact-upload` endpoints: create session, return presigned part URLs,
complete with the client-observed ETags, and return a verified completion record.
The completion handler calls S3 `CompleteMultipartUpload`, streams the object
once to recompute SHA-256 and size, and deletes the unverified object on
failure. Existing `resumable_uploads` remains the local/proxy compatibility
implementation. Change `asgi_app.py`'s blob route to return a local file,
provider-backed stream, or a short-lived redirect/JSON URL according to the
client capability. Keep reference entries
(`s3://`, `hf://`, and other external references) unchanged; they are not
Trackio-owned blobs.

Change both SQLite and Doris purge/project-delete paths to delete through the
selected store. Preserve the existing retained-manifest calculation and only
delete `deleted_digests - retained_digests`. Storage-byte reporting must use the
provider's `stat`/listing result when the backend is S3. Local database behavior
and public artifact API response shapes must remain compatible.

### Milestone 3: tests and failure behavior

Add unit tests for key derivation, local parity, S3 idempotent overwrite,
presigned multipart request shape, ETag collection, digest and length mismatch,
missing-object handling, provider timeout classification, stream cleanup, and
purge retention. Use a fake client object rather than a network mock framework
so tests remain deterministic. Add an integration test marker that creates a
uniquely named temporary RustFS bucket/prefix, uploads a small artifact directly
using the returned URLs, completes it through Trackio, downloads it directly,
and removes only its test prefix.

Run the Trackio unit suite, ruff, and wheel build. The real integration test is
required for the deployment qualification command and skips clearly when the
RustFS endpoint or protected credentials are unavailable.

### Milestone 4: migration and receipts

Add `trackio migrate-artifacts` (or the repository's established CLI command
style) with `--backend s3`, `--project`, `--prefix`, `--dry-run`, `--verify`,
`--delete-local`, and `--receipt`. It must enumerate local CAS files, derive
project/digest from their paths, compare against manifests when available,
upload idempotently, re-read or HEAD each object, and write a receipt containing
source path, target key, digest, size, status, and timestamp. `--delete-local`
must refuse to run without a prior successful verification receipt matching the
same backend, bucket, prefix, and source digest set.

Run dry-run and copy/verify for one small canary project first. Then run all
projects with local bytes retained. Only after Trackio can download every
retained artifact through RustFS and the receipt is archived may an operator run
the deletion phase.

### Milestone 5: deployment and cutover

Add a dedicated RustFS bucket/prefix and server-only variables to
`ai-infra/ansible/roles/control/files/compose.yml` and its protected template.
Build a new immutable Trackio image from the published fork commit, deploy a
canary endpoint, and run the real integration/qualification script. Verify
health, artifact upload/download/restart, Doris metadata, purge retention, and
that workers receive no RustFS credentials. Then migrate existing artifacts,
switch the production service, and retain the local directory until the agreed
recovery window expires.

Update `docs/operations/dstack-trackio/object-storage.md`,
`docs/tooling/trackio/README.md`, and the Trackio fork ledger with the actual
backend configuration boundary, migration receipt location, and rollback
procedure. Update the exact Trackio dependency pin in the framework only after
the fork commit and image are published.

## Concrete Steps

Work in `/home/hammad/projects/trackio` for Trackio code and tests, in
`/home/hammad/projects/ai-infra` for deployment, and in `/home/hammad/projects/rl`
for this plan and operational documentation. Preserve the existing unrelated
dirty changes in all three repositories.

The initial local validation commands are:

    cd /home/hammad/projects/trackio
    uv run ruff check trackio tests
    uv run pytest tests/unit -q
    uv build

The deployment qualification command will be added with the migration tool and
must report a successful upload, HEAD/GET digest match, Trackio download match,
and cleanup of its temporary prefix. Never print access keys or secret values in
logs or receipts.

## Validation and Acceptance

The local backend must pass its existing test suite unchanged. With the S3
backend selected, `check_artifact_blobs` must report a blob present after upload,
`artifact_log` must commit the manifest only after all referenced objects exist,
and the public blob URL must return the exact original bytes. A Trackio restart
must not change the result. A failed or truncated upload must not create a
committed manifest. A purge must remove an unreferenced RustFS object while
retaining a shared object consumed by another artifact version.

The migration command must be idempotent: running it twice reports the second
copy as already present and does not duplicate bytes. Verification must compare
SHA-256 and size, not merely object existence. Local deletion is accepted only
when the receipt proves every retained manifest digest is readable from RustFS.

The production cutover is accepted only when the live Trackio container has the
RustFS endpoint/bucket configuration but no credential in worker launch
envelopes, the RustFS bucket contains the migrated artifact keys, the local
final CAS contains no required blob after the retention decision, and
Observatory still reads artifact metadata and run lineage through Trackio/Doris.

## Idempotence and Recovery

All writes are digest-addressed and safe to retry. A provider outage leaves
local temporary upload sessions and no committed manifest; retrying completion
must reuse the same upload id. Migration never deletes the source during copy or
verify. If the canary fails, switch the backend setting back to `local`, restart
Trackio, and keep the original local CAS untouched. If a post-cutover object is
missing, restore the local source or rerun migration for that digest before
attempting any cleanup.

## Artifacts and Notes

The migration receipt is the durable evidence for a cutover. It must include a
redacted backend identity (endpoint host, bucket, prefix), source digest set,
object count/bytes, verified count/bytes, skipped/already-present count, failure
list, and the command/configuration revision. It must never include access keys,
secret keys, signed URLs, or raw job prompts.

## Interfaces and Dependencies

The Trackio server-side seam should expose typed operations equivalent to:

    class ArtifactStore(Protocol):
        def has(self, project: str, digest: str) -> bool: ...
        def put_file(self, project: str, digest: str, source: Path) -> None: ...
        def open(self, project: str, digest: str) -> BinaryIO: ...
        def delete(self, project: str, digest: str) -> None: ...
        def stat(self, project: str, digest: str) -> ArtifactObjectStat: ...

The S3 implementation uses the existing RustFS S3-compatible API through boto3,
with no client-side credential exposure. The provider configuration is selected
only by the Trackio server process. The client uses standard HTTP PUT against
presigned URLs and does not need boto3. The existing `cas` module remains the
client-side hashing/path utility and local compatibility implementation; it is
not used as the server's provider abstraction after Milestone 2.

### Plan revision note

2026-08-07: Initial plan created after live/configuration inspection. It
explicitly separates RustFS service qualification from Trackio backend
selection, includes resumable uploads and purge semantics, and defers deletion
of the current 5.1 GB local artifact store until verified migration.

2026-08-07: Revised after clarifying that the target is any S3-compatible
backend and direct producer uploads. The plan now makes presigned multipart
transfer the primary path, keeps proxy upload for compatibility, and requires
post-upload server-side SHA-256 verification before metadata commit.
