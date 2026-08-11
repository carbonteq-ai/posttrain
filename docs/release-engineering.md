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
retains the wheelhouse and evidence. The development index is an immutable
staging channel: if a different artifact already occupies the final version,
the release must be repaired with a new version rather than overwritten.

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
different source or configuration.

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

A failed candidate never reaches stable. If its source changes after an
artifact has occupied the development version, a new framework version is
required. After one candidate passes, its accepted materialization is retained
and the release PR merges. The final transaction:

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
installed from the development index. Never overwrite an RC, replace an
accepted stable version, or reinterpret an OCI digest. A repair before final
publication increments the RC number; a repair after stable publication
advances the framework version. Detailed trust, network, and retry semantics
are in the [LAN release runner architecture](./architecture/lan-release-runner.md).

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
Observatory from the same provider.

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
