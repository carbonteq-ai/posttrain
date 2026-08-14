# LAN release runner

**Status:** the LAN runner and Posttrain candidate/final workflows are
implemented. The dependency-to-platform promotion contract below is proposed
by [ADR 0014](../decisions/0014-attested-release-promotion-graph.md) and must be
implemented before the next production release is accepted.

Posttrain releases require access to `pypi.lan` and `registry.lan`, while the
source review, approval, and release record live in GitHub. GitHub therefore
acts as the release control plane, and a dedicated self-hosted runner on the
private LAN acts as the execution plane. The runner produces immutable release
candidates on the development channel, lets maintainers repair the release
branch without consuming a final version, and publishes the final version only
after a candidate has passed.

This architecture does not require a public IP address. A GitHub Actions
self-hosted runner opens an outbound HTTPS connection to GitHub and polls for
work. GitHub does not open an inbound connection to the runner.

## Services, authority, and release evidence

A release crosses several systems, but no system is authoritative for another
system's state. The release materialization connects their signed-off receipts;
it does not replace them.

| Service or repository | Release role | Authoritative state | Required receipt |
| --- | --- | --- | --- |
| GitHub / Posttrain | Review and release control plane | Source commit, required checks, approvals, tags, retained bundles | Exact source SHA and successful workflow identity |
| Maintained dependency repository, initially Trackio | Builds and qualifies its own release | Dependency tag, distributions, service image, compatibility guarantees | Version, source SHA, artifact hashes, image digest, deployment and canary result |
| `pypi.lan` | Python distribution plane | Bytes visible from `carbonteq/dev` and `carbonteq/stable` | Index readback hashes for every expected file |
| `registry.lan` | OCI distribution plane | Immutable runtime and service image digests | Registry readback digest, labels, and cold-pull result |
| Trackio service | Observation and artifact write plane | Deployed server version, storage configuration, manifests and blobs | Sanitized deployment identity plus artifact round-trip evidence |
| dstack | Remote execution plane | Submitted job, selected worker, lifecycle and terminal state | Capacity snapshot, provider run identity, terminal status and cleanup result |
| Observatory | Read-only evidence product | Job-aware projection of provider evidence | Deployed version/digest and readback of the candidate run |
| `ai-infra` | Private execution and deployment authority | Runner, service deployment, network and credential policy | Runner/deployment health and sanitized configuration receipts |

## Responsibilities

GitHub owns the reviewed source commit, required checks, manual release
approval, workflow logs, retained release bundle, final tag, and GitHub Release.
It does not need network access to the private package index.

The LAN runner owns deterministic staging, distribution builds, artifact
inspection, publication to the development index, clean-install qualification,
promotion to the stable index, and readback verification. It resolves the
Python-index URL and credentials from Posttrain's machine configuration. No
credential value is committed to the repository or passed as a workflow input.

The internal index owns two distinct states:

- `carbonteq/dev` is the qualification index. A candidate run publishes the
  authored final version, such as `0.3.17`, so an accepted wheel can be promoted
  without rebuilding or renaming it. Failed bytes are retained by the workflow
  receipt. They may be removed from development only as a complete coordinated
  version after proving stable is empty and every indexed hash matches that
  failed receipt. Accepted candidate bytes are never removed or overwritten.
- `carbonteq/stable` is the non-volatile consumer index. A candidate reaches it
  only by server-side promotion after qualification. It is never rebuilt or
  uploaded independently.

The OCI registry is a separate artifact plane. A protected release-candidate
workflow builds runtime images only when their inputs changed, pushes directly
to `registry.lan/carbonteq`, and qualifies the resulting digests. The generated
`published.toml` and image receipts are release outputs under a dedicated
materialization directory. They are never inferred from uncommitted source-tree
state. The final workflow verifies accepted image receipts without rebuilding.

## Dependency-to-platform promotion graph

The release is a graph with fail-closed edges, not a single build job:

```text
dependency source and tests
  -> dependency distributions and service image
  -> internal-index and registry readback
  -> deployed dependency identity
  -> dependency compatibility canary
  -> Posttrain source and locks
  -> Posttrain runtime images and registry readback
  -> release materialization
  -> candidate distributions and development-index readback
  -> clean consumer install
  -> packed dstack job
  -> Trackio artifact round trip
  -> Observatory readback
  -> exact final build and development-index canary
  -> stable promotion and readback
  -> final tag and GitHub Release
```

An edge may verify earlier evidence but must not repair it. For example, the
Posttrain candidate may verify the deployed Trackio receipt, but it must not
publish Trackio packages, deploy Trackio, or upload a missing artifact blob as
part of qualification.

### Maintained dependency gate

Each maintained dependency publishes independently before a Posttrain
candidate starts. Trackio's gate must cover its Python client, server image and
live storage path together. At minimum, its dedicated canary uses a release-only
project and object-store prefix to prove direct multipart upload, SHA-256
verification, manifest commit, download, restart behavior, stale completed
session recovery, and retention/purge behavior. The receipt records the exact
client version, server image digest and deployed configuration identity.

The Posttrain lock is accepted only when package metadata, `uv.lock`, runtime
image profiles and generated image locks resolve the same dependency version.
CI rejects drift rather than allowing the wheelhouse and runtime image to use
different Trackio versions.

## Release materialization contract

Immutable committed source and generated release evidence are different input
classes. `posttrain-release stage` begins with `git archive` of the exact source
commit. OCI publication writes `published.toml`, per-image receipts and a
materialization receipt outside the checkout. Staging accepts that receipt
through an explicit argument and projects only the declared, hash-verified
files into the staged tree.

The materialization receipt binds:

- source repository and commit;
- target and candidate version;
- `uv.lock` and generated dependency-lock digests;
- accepted maintained-dependency receipt digests;
- runtime-image manifest hash and immutable OCI digests;
- built Python filenames and SHA-256 hashes after the build step;
- qualification run, Trackio project and Observatory readback identities.

Arbitrary dirty files are never copied into a release. A candidate retry may
reuse an immutable receipt whose inputs still match, but any changed source,
lock, dependency deployment, image or distribution allocates a new candidate.

## Trust boundary

The runner must not share a host with `ai-control`, the package index, Trackio,
or other authoritative services. A repository workflow executes code from the
checked-out commit, so placing the runner on a service host would give release
code an unnecessarily broad blast radius.

The durable deployment is a small, dedicated Ubuntu VM managed by `ai-infra`.
It runs under an unprivileged service account and carries a repository-scoped
runner registration with labels such as `self-hosted`, `linux`, `x64`, and
`lan-release`. It needs:

- outbound HTTPS to GitHub for runner control, logs, and artifacts;
- LAN DNS, HTTPS, and the private CA needed to reach `pypi.lan`;
- scoped write access to `registry.lan/carbonteq` and readback access to the
  accepted digests;
- rootless BuildKit storage owned by the runner account, never a host Docker
  socket or privileged container runtime;
- no inbound Internet route, public address, GPU, or SSH access from GitHub;
- no execution of pull-request workflows or code from forked revisions.

The runner keeps two independent caches with different ownership. Rootless
BuildKit retains OCI layers in its managed state root and applies the bounded
GC policy from `ai-infra`; the release workflow records `buildctl du` before and
after image work. Python build metadata uses the runner's persistent
`UV_CACHE_DIR` when one is configured, while local invocations fall back to a
temporary release-local cache. Neither cache is release evidence: the
wheelhouse receipt and immutable registry digests remain the source of truth.

Only manually dispatched, protected tag/candidate workflows may target a LAN
release runner. Access is repository-scoped: Posttrain workflows cannot use
Trackio's publication credentials, and Trackio workflows cannot execute on the
Posttrain runner merely because both need LAN access. `ai-infra` provisions a
runner group or equivalent protected execution path for each approved
repository and limits credentials to that repository's release environment.
Candidate work never runs from a `pull_request` event or fork. Final Posttrain
publication verifies that the selected commit is the current merged
default-branch commit and that the final tag does not already name another
commit.

## Candidate and final-release sequence

The release is a two-phase state machine. The target version is authored once
in `release/manifest.toml`; for example, `0.3.1`. The candidate workflow derives
PEP 440 versions such as `0.3.1rc1` without rewriting that manifest. Every
candidate is immutable: a repair produces `rc2`, not replacement bytes under
`rc1`.

1. Dependency releases finish first. Posttrain verifies their internal-index,
   registry, deployment and compatibility receipts without mutating them.
2. The release PR updates `release/manifest.toml`, the changelog, the single
   dependency lock and intended runtime-image inputs. Exact-SHA CI qualifies
   committed source.
3. A maintainer dispatches **Prepare candidate**. The GitHub run identity is
   the next RC identity. The runner builds and registry-verifies changed runtime
   images and writes all generated image evidence under the release
   materialization directory.
4. The staging command combines exact committed source with that declared
   materialization, validates every binding and builds candidate distributions
   once. The receipt records version, source, dependency and lock receipts,
   filenames, hashes and OCI digests.
5. The exact candidate files are uploaded to `carbonteq/dev`, read back by
   hash, and installed in a clean environment with workspace and Git sources
   disabled.
6. The clean consumer packs and executes the smallest representative dstack
   job on an explicitly qualified worker. Acceptance requires expected image
   digests, terminal job success, a Trackio artifact round trip, finalization,
   cleanup and Observatory readback of that same run. Changed training,
   evaluation or serving kinds require their corresponding bounded canary;
   untested kinds are reported as unqualified.
7. A failure creates no final tag and never writes to stable. Its workflow
   artifact remains the immutable failure receipt. If corrected source keeps
   the same authored final version, the old development version must pass the
   audited whole-version retirement gate before the next candidate uploads.
8. After one candidate passes, the release PR is merged. The accepted
   materialization remains a retained workflow artifact; generated OCI state
   does not need to be committed to bridge the two workflows. A maintainer
   dispatches **Publish release** for the exact merged default-branch commit and
   accepted candidate receipt.
9. The final workflow verifies the accepted candidate wheelhouse, dependency
   and OCI receipts, merged source tree, and development-index hashes without
   rebuilding, reinstalling, redeploying, or running a second GPU canary.
10. It promotes the exact candidate Python files server-side from
    `carbonteq/dev` to `carbonteq/stable`, then verifies stable readback hashes.
11. Only after stable readback succeeds does the workflow create annotated tag
   `v0.3.1` and a GitHub Release containing the final bundle and receipt.

The build-once invariant applies independently to every candidate and to the
final version. A receipt is the byte-level contract connecting each build to
its index files, qualification evidence, source commit, and release record.

## OCI qualification

An OCI build is not qualified merely because BuildKit pushed it. Qualification
has three layers and runs only for changed image inputs:

1. Build qualification runs the repository validators and the image builder's
   dependency/parent checks for every selected runtime variant. The current
   workflow does not claim a complete changed-kind matrix; it records the
   bounded canary explicitly and keeps matrix expansion as a separate release
   change.
2. Registry qualification pushes the candidate, reads its digest and labels
   back from `registry.lan`, and cold-pulls by digest so a warm builder cache
   cannot hide a broken registry artifact.
3. Runtime qualification first records a sanitized dstack capacity receipt and
   requires the selected 96-GiB worker to be active, healthy, reachable,
   unsliced, and idle. It then packs a tiny actual-job image on top of the
   selected kind digest and executes the smallest representative operation through
   dstack. The initial release workflow selects the transformation job because
   it exercises packaging, the private registry, provider submission, and
   terminal cleanup without starting an additional evaluation experiment.
   Acceptance requires the expected digest, terminal success, retained
   Trackio evidence, and Observatory readback. A future changed-kind matrix
   should add the corresponding bounded smoke for evaluation, serving, and
   training variants before claiming those kinds are release-qualified.

The base and kind images intentionally contain dependencies rather than the
framework worker and project source. That is why a real packed job is required
in addition to import smokes. When the base changes, all derived kinds repeat
their applicable gates; when one kind changes, only that kind is rebuilt and
requalified.

## Failure and retry semantics

Every failure is assigned to one edge before a retry:

| Failure class | Examples | Recovery owner |
| --- | --- | --- |
| Dependency publication | Missing wheel or wrong index hash | Dependency repository release |
| Dependency deployment | Server digest or configuration differs from receipt | `ai-infra` and dependency owner |
| Materialization | Generated manifest absent, stale or mismatched to source/lock | Posttrain release tooling |
| Distribution | Wheel metadata, contents or index readback mismatch | Posttrain candidate build |
| OCI | Build, registry readback, labels or cold pull fail | Runtime-image release step |
| Execution | Capacity, packing, scheduling or job lifecycle fail | Posttrain/dstack owning layer |
| Tracking | Artifact upload, manifest commit, finalization or cleanup fail | Trackio client/service owning layer |
| Read product | Observatory cannot retrieve or project the same run | Observatory query/deployment layer |

Before a candidate upload, a failed workflow is side-effect free. After an
upload, the workflow artifact is immutable. Matching files may be reused. A
repair creates a new candidate run and normally advances the version; before
stable publication it may instead retire the failed coordinated development
version after exact receipt verification. Qualification failure never creates
a final tag. A failed OCI digest remains an unreferenced candidate and may be
garbage-collected by registry policy; it is never written into the accepted
manifest.

Promotion is resumable package by package. A retry skips stable files whose
hashes already match the receipt and blocks on any conflicting file. Because
`carbonteq/stable` is non-volatile, a conflicting or incorrect stable artifact
cannot be repaired in place; the manifest version must advance.

If the final `0.3.1` files fail before stable promotion, stable and the final
tag remain untouched. A source or artifact correction returns to the candidate
loop; it does not mutate an accepted version. If stable promotion succeeds but
GitHub tag or Release creation fails, retry only finalization against the
retained receipt and exact source commit. Never rebuild or re-upload.

## Implementation ownership

The framework repository owns the Posttrain release workflow, materialization
and receipt schemas, staging and verification commands, and maintainer
documentation. Each maintained dependency repository owns its build,
publication and compatibility receipt. The sibling `ai-infra` repository owns
repository-scoped runner execution, service deployments, network policy,
private-CA installation, machine configuration and health checks. Observatory
owns job-aware readback qualification; it never becomes a writer or recovery
authority. Registration tokens are short-lived bootstrap material, and index,
registry and service credentials remain protected machine state that must never
appear in workflow inputs or logs.

## Revision history

- 2026-08-09: Reframed the runner as one execution node in an attested
  dependency-to-platform promotion graph; added release materialization,
  dependency deployment, Trackio compatibility and Observatory readback gates.
