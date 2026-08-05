# LAN release runner

**Status:** implemented and live-qualified on the release branch; activation
requires merging the protected workflows and configuring the GitHub
`release-candidate` and `release` environments.

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

- `carbonteq/dev` is the qualification index. Candidate versions such as
  `0.3.1rc1` and `0.3.1rc2` are never overwritten even though the index itself
  is operationally volatile. The final `0.3.1` files also land here for their
  last clean-install canary before promotion.
- `carbonteq/stable` is the non-volatile consumer index. A candidate reaches it
  only by server-side promotion after qualification. It is never rebuilt or
  uploaded independently.

The OCI registry is a separate artifact plane. A protected release-candidate
workflow builds runtime images only when their inputs changed, pushes directly
to `registry.lan/carbonteq`, qualifies the resulting digests, and writes those
registry-read identities into
`packages/runtime-images/src/posttrain/runtime_images/published.toml`. The
final workflow verifies accepted image receipts without rebuilding.

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

Only the manually dispatched candidate and final-release workflows may target
`lan-release`. Both require a protected GitHub environment. Candidate work is
restricted to an internal release branch in this repository and never runs
from a `pull_request` event or fork. Final publication verifies that the
selected commit is the current merged default-branch commit and that the final
tag does not already name another commit.

## Candidate and final-release sequence

The release is a two-phase state machine. The target version is authored once
in `release/manifest.toml`; for example, `0.3.1`. The candidate workflow derives
PEP 440 versions such as `0.3.1rc1` without rewriting that manifest. Every
candidate is immutable: a repair produces `rc2`, not replacement bytes under
`rc1`.

1. The release PR updates `release/manifest.toml`, the changelog, generated
   dependency locks, and intended runtime-image inputs. Normal CI qualifies the
   source.
2. A maintainer dispatches **Prepare candidate** for the internal release
   branch. The runner allocates the next unused `rcN`, stages that candidate
   version, builds its wheels and source distributions once, and writes a
   receipt containing the target version, candidate version, source commit,
   filenames, SHA-256 hashes, dependency-lock identity, and OCI input identity.
3. The exact candidate files are uploaded to `carbonteq/dev` and installed in a
   clean environment with workspace sources disabled. Package metadata,
   independent-consumer behavior, packing, and a bounded dstack canary must
   pass.
4. When runtime-image inputs changed, the same workflow builds and pushes the
   affected images with rootless BuildKit, reads their immutable digests back,
   and retains the generated image receipt and manifest. The current release
   gate runs one bounded packed transformation canary through dstack; expanding
   this to a changed-kind matrix is a follow-up gate, not an assumption hidden
   in the first runner rollout. Unchanged images are verified and reused. The
   canary is explicitly placed on `carbonteq-ai-workstation.lan`, the idle
   96-GiB RTX PRO worker; a generic 24-GiB selector is not sufficient because
   dstack does not observe unrelated host GPU processes.
5. A passing candidate may receive an immutable GitHub prerelease tag such as
   `v0.3.1-rc.2`. A failure creates no final tag and never writes to stable. The
   release branch is repaired and the next candidate is built.
6. After one candidate passes and its generated OCI manifest is committed, the
   release PR is merged. A maintainer dispatches **Publish release** for the
   exact merged default-branch commit.
7. The runner stages final `0.3.1`, builds those final distributions once,
   uploads them to `carbonteq/dev`, and performs an index-only install plus a
   final dstack canary. Candidate distributions are not renamed or promoted as
   the final version because their embedded metadata names `0.3.1rcN`.
8. The final workflow verifies the candidate-qualified OCI receipts and
   registry digests without rebuilding. It promotes the exact final Python
   files server-side from `carbonteq/dev` to `carbonteq/stable`, then verifies
   stable readback hashes.
9. Only after stable readback succeeds does the workflow create annotated tag
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

Before a candidate upload, a failed workflow is side-effect free. After an
`rcN` upload, that candidate is immutable: matching files may be reused, but a
repair allocates `rcN+1`. Qualification failure leaves evidence in
`carbonteq/dev` and never creates a final tag. A failed OCI digest remains an
unreferenced candidate and may be garbage-collected by registry policy; it is
never written into the accepted manifest.

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

The framework repository owns the release workflow, receipt schema, staging
and verification commands, and maintainer documentation. The sibling
`ai-infra` repository owns the runner VM, service account, network policy,
private-CA installation, machine configuration, runner lifecycle, and health
checks. The registration token is short-lived bootstrap material; package-index
credentials remain local machine state and must never appear in workflow logs.
