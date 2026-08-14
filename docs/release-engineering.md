# Release engineering

This page is for **framework maintainers** who prepare, qualify, and publish
Posttrain releases. Project teams installing a release should read
[install.md](./install.md) instead; the day-to-day publishing checklist is in
[publishing.md](./publishing.md).

Distribution infrastructure: CarbonTeq-managed projects install from
`pypi.lan`, and job images resolve from `registry.lan/carbonteq`. GitHub
provides source review, release orchestration, and an auditable copy of the
accepted release bundle; it is not a second build plane or the framework's OCI
registry. The source repository is published at
[`carbonteq-ai/posttrain`](https://github.com/carbonteq-ai/posttrain).

## Candidate and final workflows

Two protected workflows run on a dedicated LAN-connected self-hosted runner.

**Prepare candidate** consumes the exact-source readiness receipt produced by
required Quality, publishes the authored final version only to
`carbonteq/dev`, qualifies changed OCI digests plus one real packed job, and
retains the wheelhouse and evidence. A candidate run is the release-candidate
identity; the distributions use the authored final version so the accepted
bytes can be promoted unchanged. Development files are normally immutable. A
failed, never-promoted candidate may be retired as one coordinated version only
when its retained receipt matches every development-index file, stable has no
file for that version, and the deletion receipt names the failed run. An
accepted candidate, a partial or unexplained development version, and every
stable version remain immutable and require a new framework version.

**Publish release** runs only after a candidate passed and the release PR
merged. It verifies that the candidate source is an ancestor of, or has the
same Git tree as, the merged source; rechecks the retained wheelhouse and
development-index hashes; promotes those unchanged bytes to
`carbonteq/stable`; and creates the final tag last. It does not rebuild,
reinstall, recheck the registry, or run a second packed GPU job. The final
workflow therefore requires both the merged source SHA and the successful
candidate receipt.

External Verifiers environments, including `automationbench-v1`, resolve from
the immutable commits in the bundled constraints file instead of being copied
into the framework bundle.

A release remains reproducible because the receipt binds the merged source
commit, framework version, distribution filenames and hashes, dependency-lock
identity, maintained-dependency receipts, and accepted OCI manifest. The
internal indexes, registry and GitHub Release must expose those exact
identities.

### Release-specific qualification inputs

Algorithm, model and environment qualification is release input rather than a
reason to rerun every expensive experiment during publication. The release PR
must link the living plan and exact retained evidence for every capability it
claims. For example, the 0.3.2 Gemma qualification is retained in
[`docs/plan/gemma4-0.3.2-support-and-release.md`](./plan/gemma4-0.3.2-support-and-release.md),
while the corrected DAPO and 0.3.3 release work is tracked in
[`docs/plan/sft-dapo-256-experiment-and-framework-release.md`](./plan/sft-dapo-256-experiment-and-framework-release.md).

Each linked result names the source, model/environment revisions, execution
target, image digest, run identity and evidence location. Mutable model or
image tags are not accepted. A failed candidate is repaired as a new RC; it
never mutates the target stable version or reuses an old run as proof for a
different source or configuration. Because the candidate already contains the
authored final Python version, “new RC” means a new workflow run and receipt,
not a PEP 440 `rcN` distribution that would need rebuilding for promotion.

## Release artifact graph

Publish in dependency order. A Posttrain candidate does not build or publish a
maintained dependency on its behalf. Each earlier layer must have its own
readback and compatibility receipt before the next layer starts.

| Order | Artifacts | Purpose |
| --- | --- | --- |
| 1 | Trackio and other maintained dependency distributions/service images | Prove independent publication, deployment and live compatibility |
| 2 | Posttrain runtime-image digests and generated image receipts | Bind executable dependencies before building consumer distributions |
| 3 | Posttrain source plus declared generated inputs | Create one hash-addressed release materialization |
| 4 | `posttrain-*` distributions | Publish reusable contracts, adapters, CLI and applications as one coordinated version |
| 5 | Clean consumer and packed job | Prove index-only installation, OCI selection and remote execution |
| 6 | Trackio artifact round trip and Observatory readback | Prove the supported write and read products against the same run |
| 7 | Exact final files promoted to stable | Make accepted bytes consumable without rebuilding |

Use one coordinated pre-1.0 framework version for first-party distributions.
Fork and environment distributions may follow their own upstream-derived
versions, but framework metadata and release notes must name the exact
accepted versions.

### Maintained-fork preflight

The candidate must enumerate every maintained fork selected anywhere in the
executable graph, not only direct Python dependencies. This includes forks
selected by a runtime-image lock, an environment package, or the deployment
plane. The receipt records one of three publication forms:

- a Python distribution: immutable GitHub Release, wheel and source archive
  hashes, development readback, accepted qualification, and stable readback;
- a source-backed runtime: an immutable source release and retained build
  artifact selected by the runtime lock; a moving branch or bare Git checkout
  is not a release; or
- a component release: one source revision binding all server images and
  runner/shim binaries, their digests, deployment readback, and live
  qualification.

Posttrain does not build any of these dependencies during its candidate
workflow. A missing release, an unpublished local fork commit, or a production
deployment still running a different component revision blocks candidate
creation. Upstream dependencies pinned directly by full SHA are recorded as
third-party source inputs, but are not misrepresented as CarbonTeq forks.

### Actual-job image delta contract

The universal and job-kind images are release artifacts. An actual-job image
must inherit the selected job-kind image by digest and add only the resolved
job lock, environment/framework wheels, selected source, configuration, and
materialized dataset files. It must not reinstall CUDA, PyTorch, vLLM, TRL, or
veRL, and it must not include model weights or checkpoints.

Before job publication, mirror the base and kind images by digest into the
same registry prefix used for actual jobs. The image exporter must preserve
the parent layer descriptors (`force-compression=false`); recompressing an
existing parent changes every heavy layer digest and turns the first job for a
kind into a multi-gigabyte upload. Candidate qualification compares the child
manifest with its parent and requires the complete ordered parent layer list
to be unchanged. The receipt reports the count and compressed bytes of only
the new job layers so transfer cost is visible. A missing parent manifest or
blob is a registry-retention failure, not permission to rebuild or recompress
the ancestry.

### Transfer-budget contract

Release and job planning must report expected network work before opening a
builder session. Report logical object size and missing bytes separately for
the client-to-controller, controller-to-builder, builder-to-registry,
registry-to-registry, and registry-or-artifact-cache-to-worker links. The
completion receipt records the same links with observed bytes when the
underlying service exposes them. A matching digest is zero payload work even
when the referenced image is several gigabytes.

Multi-gigabyte builds, registry copies, model/checkpoint staging, and dataset
staging should run on the LAN side of the VPN when a separate developer
job-build service is deployed. The protected `ai-release` builder remains
release-only and must never accept end-user contexts or credentials. Without a
developer job-build service, the supported path is the developer's own local
BuildKit: its cold parent pull and job-layer upload are shown before execution,
and its warm cache is reused. Model weights and large datasets are
content-addressed worker artifacts, not runtime or actual-job layers.
Concurrent requests for one content key share one producer, and a worker pulls
only the selected kind plus missing job/model/data blobs rather than
pre-pulling every supported runtime.

The measured transfer matrix and first-install, warm-cache, partial-registry,
one-kind-change, and base-change scenarios are maintained in
[`portable-runtime-image-supply-chain.md`](./plan/portable-runtime-image-supply-chain.md#transfer-budget-matrix).

## Registry release contract

A release uses two manually dispatched transactions. GitHub records approvals
and evidence; a self-hosted runner on the private LAN performs build and
registry operations. The runner requires outbound HTTPS but no public IP or
inbound connection.

The candidate transaction:

1. verifies accepted receipts for maintained dependencies, including the exact
   Trackio client/server/storage combination;
2. verifies an internal release branch and the exact-source readiness receipt
   generated by required Quality; the same `posttrain-release readiness`
   command is available for a maintainer to run locally before dispatch;
3. uses the authored `X.Y.Z` from `release/manifest.toml` as an immutable
   development-only candidate version;
4. rebuilds OCI images only when their runtime source, relevant lock, parent
   digest, or installed CA bundle changes. A framework version bump alone
   reuses registry-verified digests, then records the same digests in the new
   release manifest; it never rebuilds layers just to relabel them;
5. writes `published.toml` and image receipts outside the source checkout,
   then creates a release materialization binding those generated inputs to
   the source, dependency receipts and locks;
6. stages from exact committed source plus that declared materialization,
   builds wheels and source distributions once, and records their hashes;
7. uploads the exact candidate files to `carbonteq/dev`, verifies readback and
   performs an index-only consumer install with workspace and Git sources
   disabled;
8. packs and runs the bounded dstack canary on an explicitly qualified worker;
9. proves artifact upload, manifest commit, finalization and cleanup through
   the deployed Trackio service, then reads that same run through the deployed
   Observatory;
10. retains the complete candidate materialization and classified gate results.

A failed candidate never reaches stable. If its source changes after its final
version occupied development, either advance the framework version or perform
the audited whole-version retirement described above. Retirement is allowed
only before any stable file or accepted candidate exists; it preserves the
failed candidate artifact and writes a deletion receipt before replacement.
After one candidate passes, its accepted materialization is retained and the
release PR merges. The final transaction:

1. validates the exact merged source, dependency receipts, locks, accepted
   candidate readiness receipt, and candidate materialization;
2. materializes the retained candidate wheelhouse and rechecks its hashes;
3. verifies development-index readback, writes a promotion receipt binding the
   candidate source/tree to the merged tag target, and retains it;
4. promotes the unchanged files server-side to `carbonteq/stable` and verifies
   stable readback hashes;
5. creates the tag last and attaches the same bundle and receipt to the GitHub
   Release.

Do not upload a later dependency layer until the previous layer can be
installed from the development index. Never overwrite an accepted candidate,
replace a stable version, or reinterpret an OCI digest. A repair before final
publication creates a new candidate run; it may keep the authored stable
version only through verified whole-version retirement. A repair after stable
publication advances the framework version. Detailed trust, network, and retry
semantics are in the [LAN release runner architecture](./architecture/lan-release-runner.md).

## Checkout validation before a release

Before a registry release, use the workspace lock:

```bash
mise install
uv sync --all-packages --locked --python 3.13
uv run --package posttrain posttrain doctor
uv run --package posttrain posttrain catalog validate
cd apps/lab && uv run --package posttrain posttrain work-package validate \
  .posttrain/work_packages/foundation_screen.yaml
```

For the release-specific local preflight, run the same deterministic command
that required Quality uses. It is optional acceleration for a maintainer, not
a substitute for CI: the protected candidate downloads and verifies Quality's
receipt for the exact source branch.

```bash
uv run --no-sync posttrain-release readiness --destination .release/readiness.json
uv run --no-sync posttrain-release readiness-check .release/readiness.json
```

The installed-wheel acceptance builds distributions, creates a clean Python
environment outside this repository, executes a deterministic CPU work
package, writes a terminal run through local Trackio, and reads the same run
through Observatory:

```bash
uv run pytest -q tests/consumer/test_wheel_project.py
```

The fixture is the smallest executable consumer example:
[`tests/consumer/fixture`](../tests/consumer/fixture).

Before stable release, one documented remote GPU gate must execute a supported
training or evaluation work package, record evidence, and retrieve it through
Observatory from the same provider. The candidate dispatch chooses one named
qualification profile, not an arbitrary host or memory override: the default
The `rtx-pro-96gb` release profile uses the RTX PRO 6000. The retired
`rtx4090-24gb` profile must not be selected even if stale scheduler inventory
still advertises it. The retained capacity
receipt records the exact selected host and hardware.

## Remaining release gates

The framework is feature-rich but not release-complete:

- The primary CLI performs composition-level work-package validation; concrete
  first-party job-definition preflight and `posttrain work-package run` remain.
- Trackio `carbonteq-v0.31.5.post12` is published to the internal index and
  deployed. Its manual compatibility receipt proves scalar read/write and a
  cache-independent S3 artifact round trip; a Trackio-owned automated release
  workflow remains operational follow-up rather than a 0.3.3 blocker.
- Other maintained forks and external environments need the same independent
  receipt whenever a clean consumer cannot resolve them from the internal
  index.
- License, security/contact policy, changelog, package metadata, compatibility
  window, and upgrade policy need an explicit owner decision and repository
  files.
- The Posttrain LAN runner is live. `ai-infra` still needs a protected,
  repository-scoped Trackio release path for later unattended releases;
  Posttrain remains a verifier and consumer, not Trackio's publisher.
- Release tooling still needs the explicit materialization receipt and stage
  input; copying the checkout's generated `published.toml` is temporary and is
  not an accepted production boundary.
- The Observatory image, deployment receipt, authentication boundary and
  production readback gate remain.
- Installation and task guides still need CI-executed examples.
- A clean CarbonTeq project and a remote GPU machine must pass the release
  candidate gate.

The living execution record is
[`docs/plan/dx-release-engineering.md`](./plan/dx-release-engineering.md).
