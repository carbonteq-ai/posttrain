# Release and consumption

This page defines how CarbonTeq projects consume the post-training framework and
how maintainers publish it. The intended users are CarbonTeq-managed projects;
public GitHub Releases, PyPI, and GHCR are transport and installation
infrastructure, not a commitment to support arbitrary third-party plugins or
backends.

The source repository is published at
[`carbonteq-ai/posttrain`](https://github.com/carbonteq-ai/posttrain). GitHub is
the initial team distribution channel; PyPI remains a later convenience. The
commands below distinguish source-checkout use, GitHub wheelhouse releases, and
the eventual registry contract.

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

## GitHub-first team release

Each `v*` tag runs the full validation ladder, builds every first-party wheel
plus `automationbench-v1`, records SHA-256 hashes, and attaches a wheelhouse
archive to a GitHub Release. A team project can install that exact release
without waiting for PyPI:

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
reference composition host. The bundled constraints repeat the immutable
Trackio and AutomationBench Git URLs because uv requires transitive URL
dependencies to be declared by the consumer. A release remains reproducible
because the framework tag, wheel hashes, and fork commits are immutable.

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

A release tag builds artifacts once. CI then:

1. validates source and lock state;
2. builds wheels and source distributions;
3. inspects wheel metadata and hashes;
4. installs into a clean environment with workspace sources disabled;
5. runs the independent-consumer test against those exact artifacts;
6. runs package, import-boundary, type, and documentation checks;
7. publishes the validated wheelhouse through GitHub Releases and, once
   configured, Python artifacts through PyPI Trusted Publishing;
8. builds Observatory from the tagged source, tests that image, and publishes
   semantic-version and commit tags to GHCR;
9. records the Python artifact hashes and OCI digest in the release notes.

Do not upload a later dependency layer until the previous layer can be
installed from the staging registry. Never replace an accepted PyPI version or
OCI digest; increment the prerelease version and rebuild.

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
  direct Git dependencies in the GitHub-first release, but must still be
  published before a Git-free PyPI install is claimed.
- Package names must be reserved and ownership configured in PyPI before the
  first release.
- License, security/contact policy, changelog, package metadata, compatibility
  window, and upgrade policy need an explicit owner decision and repository
  files.
- Tag-driven Trusted Publishing and GHCR workflows are not implemented.
- The Observatory image, deployment configuration, authentication boundary,
  and production readback gate remain.
- Installation and task guides still need CI-executed examples.
- A clean CarbonTeq project and a remote GPU machine must pass the release
  candidate gate.

The living execution record is
[`docs/plan/polished-framework-release.md`](./plan/polished-framework-release.md).
