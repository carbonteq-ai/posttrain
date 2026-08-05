# Release and consumption

This page defines how CarbonTeq projects consume the post-training framework and
how maintainers publish it. The intended users are CarbonTeq-managed projects;
the internal Python and OCI registries are the supported distribution
infrastructure. GitHub provides source review, release orchestration, and an
auditable copy of the accepted release bundle; it is not a second build plane
or the framework's OCI registry.

The source repository is published at
[`carbonteq-ai/posttrain`](https://github.com/carbonteq-ai/posttrain). Team
projects install from `pypi.lan`, while job images resolve from
`registry.lan/carbonteq`. The commands below distinguish source-checkout use,
the internal-index contract, and the GitHub release bundle retained for audit
and recovery.

## What a project installs

CarbonTeq-managed projects install from the internal index with the release
constraints file. That is the supported consumer path; see
[consumer-setup.md](./consumer-setup.md) for the full walkthrough (trust, env,
local Docker, and dstack).

```bash
uv venv --python 3.13 .venv
VIRTUAL_ENV=.venv uv pip install --system-certs \
  --index-url https://pypi.lan/carbonteq/stable/+simple/ \
  --constraint github-constraints.txt \
  "posttrain[observatory,trackio,trl]"
```

Add the `dstack` extra when submitting remote GPU jobs. Obtain
`github-constraints.txt` from the framework release (repository
`release/github-constraints.txt` or the wheelhouse attachment); it is not
served by the index today.

Most projects start with the primary command distribution `posttrain`. It
supplies project initialization, diagnostics, catalog inspection, and
work-package commands. It depends on the reusable catalog and composition
packages; it does not make `posttrain-lab` a runtime requirement.

Capability extras and packages (`trl`, `verifiers`, `vllm`, Trackio, W&B) are
selected by what the project executes. Checkout developers may still use
`uv add` against a workspace clone; that is a maintainer path, not the team
install contract.

Every consuming project commits:

- `.posttrain/project.toml`
- `.posttrain/catalog/` overlays
- `.posttrain/work_packages/`
- `pyproject.toml` and `uv.lock`

It does not copy the framework base catalog. That catalog is a versioned
resource inside `posttrain-catalog`. Machine-local scratch, caches, recovery
files, and local Trackio storage live under `.posttrain/state/` or an explicit
state-directory override. Durable artifacts are published through the selected
tracking/artifact backend.

## Current checkout workflow

Before a registry release, use the workspace lock:

```bash
mise install
uv sync --all-packages --locked --python 3.13
uv run --package posttrain posttrain doctor
uv run --package posttrain posttrain catalog validate
uv run --package posttrain posttrain work-package validate foundation_screen.yaml
```

Create a separate project with:

```bash
uv run --package posttrain posttrain init ../my-posttrain-project \
  --project-id my-posttrain-project
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

## GitHub release bundle

Two protected workflows run on a dedicated LAN-connected self-hosted runner.
**Prepare candidate** derives immutable prerelease versions such as
`0.3.2rc1`, publishes them only to `carbonteq/dev`, qualifies changed OCI
digests plus real packed jobs, and lets maintainers repair the release branch
without consuming the final version. **Publish release** runs only after a
candidate passed and the release PR merged. It builds final `0.3.2` once,
qualifies those exact files through `carbonteq/dev`, promotes them unchanged to
`carbonteq/stable`, and creates the final tag last. External Verifiers
environments, including `automationbench-v1`, resolve from the immutable
commits in the bundled constraints file instead of being copied into the
framework bundle.

### 0.3.2 Gemma qualification gate

The 0.3.2 candidate must include the Gemma 4 dense matrix and the paired
assistant MTP path. Before dispatching the candidate workflow, the release
review must link the successful dstack/Trackio evidence from
`docs/plan/gemma4-0.3.2-support-and-release.md`:

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

Ordinary projects should use `pypi.lan`. The attached bundle is an exact
offline installation and recovery surface for the already accepted release:

```bash
gh release download <release-tag> \
  --repo carbonteq-ai/posttrain \
  --pattern 'posttrain-wheelhouse-*.tar.gz'
mkdir posttrain-wheelhouse
tar -xzf posttrain-wheelhouse-*.tar.gz -C posttrain-wheelhouse
uv venv --python 3.13
uv pip install --python .venv/bin/python \
  --constraint ./posttrain-wheelhouse/github-constraints.txt \
  --find-links ./posttrain-wheelhouse \
  posttrain posttrain-observatory
```

Install `posttrain-lab` from the same wheelhouse when a project needs the
reference composition host. Published Trackio and AutomationBench forks resolve
through the internal index; the bundled constraints retain only dependencies
that still require immutable Git URLs. A release remains reproducible because
the receipt binds the merged source commit, framework version, distribution
filenames and hashes, dependency-lock identity, and committed OCI manifest. The
internal indexes and GitHub Release must contain those exact bytes.

For framework development or a complete reference checkout, clone the exact
tag and use the checked-in lock:

```bash
git clone --branch <release-tag> --depth 1 \
  https://github.com/carbonteq-ai/posttrain.git
cd posttrain
uv sync --all-packages --locked --python 3.13
```

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
versions, but framework metadata and release notes must name the exact accepted
versions.

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
7. uploads the exact final files to `carbonteq/dev`, qualifies installation from
   that index, and retains the final receipt and cache evidence as a GitHub
   Actions artifact;
8. promotes the unchanged files server-side to `carbonteq/stable` and verifies
   stable readback hashes;
9. creates the tag last and attaches the same bundle and receipt to the GitHub
   Release.

Do not upload a later dependency layer until the previous layer can be
installed from the development index. Never overwrite an RC, replace an
accepted stable version, or reinterpret an OCI digest. A repair before final
publication increments the RC number; a repair after stable publication
advances the framework version. Detailed trust, network, and retry semantics are in the
[LAN release runner architecture](./architecture/lan-release-runner.md).

## Remote project workflow

A remote CPU or GPU server receives a project repository, not this framework
monorepo:

```bash
git clone <carbonteq-project>
cd <carbonteq-project>
uv sync --locked --python 3.13
uv run posttrain doctor
uv run posttrain catalog validate
uv run posttrain work-package validate <name>.yaml
```

The committed project lock selects exact framework, fork, environment, and
accelerator artifacts. Secrets and provider endpoints arrive through the
server's secret manager or environment, never through `.posttrain/` source.
Large scratch or recovery state may use an absolute state-directory override.

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
