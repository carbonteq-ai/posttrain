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

**Prepare candidate** derives immutable prerelease versions such as
`0.3.2rc1`, publishes them only to `carbonteq/dev`, qualifies changed OCI
digests plus real packed jobs, and lets maintainers repair the release branch
without consuming the final version.

**Publish release** runs only after a candidate passed and the release PR
merged. It builds final `0.3.2` once, downloads the candidate's generated
image manifest before building the final wheelhouse, qualifies those exact
files through `carbonteq/dev`, promotes them unchanged to `carbonteq/stable`,
and creates the final tag last. The final workflow therefore requires both the
merged source SHA and the successful candidate run ID.

External Verifiers environments, including `automationbench-v1`, resolve from
the immutable commits in the bundled constraints file instead of being copied
into the framework bundle.

A release remains reproducible because the receipt binds the merged source
commit, framework version, distribution filenames and hashes, dependency-lock
identity, and committed OCI manifest. The internal indexes and GitHub Release
must contain those exact bytes.

### 0.3.2 Gemma qualification gate

The 0.3.2 candidate must include the Gemma 4 dense matrix and the paired
assistant MTP path. Before dispatching the candidate workflow, the release
review must link the successful dstack/Trackio evidence from
[`docs/plan/gemma4-0.3.2-support-and-release.md`](./plan/gemma4-0.3.2-support-and-release.md):

- E2B, E4B, 12B Unified, and 31B each pass the model-neutral text-generation
  smoke on `targets/carbonteq-rtx-pro-6000-96gb`.
- The 12B TRL GRPO run has MTP enabled, complete non-truncated traces, reward,
  and speculative draft/accepted plus KV-cache metrics.
- The release candidate is built from the exact merged commit, publishes only
  to `carbonteq/dev`, and runs the final packed canary from the candidate
  wheelhouse. No mutable model tag or image tag is valid release evidence.

These are product qualification inputs, not a request to run the full Gemma
matrix on every ordinary pull request. A failed candidate is repaired as a new
RC; it never mutates the target stable version or reuses an old run ID.

## Release artifact graph

Publish in dependency order. A release candidate must install from its
published artifacts with workspace sources disabled before later layers are
uploaded.

| Order | Artifacts | Purpose |
| --- | --- | --- |
| 1 | CarbonTeq Trackio fork, `automationbench-v1`, other maintained fork/environment distributions | Replace transitive Git or unpublished path dependencies |
| 2 | `posttrain-common`, `posttrain-data`, `posttrain-eval`, `posttrain-serve`, `posttrain-tracking`, `posttrain-train`, `posttrain-work` | Reusable contracts and capabilities |
| 3 | `posttrain-catalog`, `posttrain-tracking-trackio`, `posttrain-tracking-wandb` | Versioned selections and provider adapters |
| 4 | `posttrain`, `posttrain-lab`, `posttrain-observatory` | User-facing command and application distributions |
| 5 | Observatory OCI image | Immutable deployed read product |

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

1. verifies an internal release branch and green normal CI;
2. allocates the next unused `X.Y.ZrcN` from the target version in
   `release/manifest.toml`;
3. builds wheels and source distributions once and records their hashes;
4. uploads the exact candidate files to `carbonteq/dev`;
5. runs an index-only consumer install, job packing, and a bounded dstack
   canary;
6. builds OCI images only when their inputs changed, pushes directly to
   `registry.lan`, verifies registry readback, and executes one bounded packed
   transformation canary through dstack on the explicitly verified idle
   `carbonteq-ai-workstation.lan` RTX PRO worker. A changed-kind real-job
   matrix is a follow-up gate, not an implicit property of the first runner
   rollout;
7. retains Trackio and Observatory evidence and generates `published.toml`
   only for accepted image digests.

A failed candidate is repaired as the next RC and never reaches stable. After
one candidate passes and the generated image records merge, the final
transaction:

1. validates source and lock state;
2. builds wheels and source distributions;
3. inspects wheel metadata and hashes;
4. installs into a clean environment with workspace sources disabled;
5. runs the independent-consumer test against those exact artifacts;
6. runs package, import-boundary, type, and documentation checks;
7. restores the generated `published.toml` from the successful candidate run,
   verifies that its image revision is an ancestor of the merged source and
   that its framework version matches the final version, then builds the final
   wheelhouse with those exact image digests;
8. uploads the exact final files to `carbonteq/dev`, qualifies installation
   from that index, and retains the final receipt and cache evidence as a
   GitHub Actions artifact;
9. promotes the unchanged files server-side to `carbonteq/stable` and verifies
   stable readback hashes;
10. creates the tag last and attaches the same bundle and receipt to the GitHub
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
- The Trackio and AutomationBench forks are merged and pinned by immutable
  public GitHub commits. Their renamed distributions can therefore travel as
  direct Git dependencies in the retained release bundle, but must still be
  published before a Git-free internal-index install is claimed.
- License, security/contact policy, changelog, package metadata, compatibility
  window, and upgrade policy need an explicit owner decision and repository
  files.
- The former tag-triggered GitHub-hosted workflow has been replaced on the
  release branch by the protected LAN-runner candidate/final workflows. The
  runner is live-qualified in `../ai-infra`; merge plus GitHub environment
  reviewer configuration is still required before the first production
  dispatch.
- The Observatory image, deployment configuration, authentication boundary,
  and production readback gate remain.
- Installation and task guides still need CI-executed examples.
- A clean CarbonTeq project and a remote GPU machine must pass the release
  candidate gate.

The living execution record is
[`docs/plan/dx-release-engineering.md`](./plan/dx-release-engineering.md).
