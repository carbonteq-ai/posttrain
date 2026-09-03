# Installing Posttrain

This page is the single source of truth for installing the framework. It is
written for **project teams** consuming a released version. If you are working
on the framework itself, use a workspace checkout instead — see
[contributing.md](./contributing.md). If you are publishing a release, see
[release-engineering.md](./release-engineering.md).

There are two supported ways to install a release:

1. **Internal index** (`pypi.lan`) — the standard path on the CarbonTeq
   network. Requires the internal CA to be trusted first.
2. **GitHub release wheelhouse** — an exact offline copy of the accepted
   release, attached to the GitHub Release. Works without the internal
   network; also the audit and recovery surface.

Both install the same bytes: the release process qualifies the wheelhouse
files and promotes them unchanged to the index. Whichever path you use, the
release **constraints file is required, not optional** — some dependencies are
maintained forks pinned to immutable Git commits, and uv will not resolve a
transitive direct URL unless it is also constrained at the top level.

## Prerequisites

- Python 3.13 and [`uv`](https://docs.astral.sh/uv/)
- For the internal index: network access to `pypi.lan` and the internal CA
  trusted on this machine ([Getting started §1](./getting-started.md#1-trust-the-internal-certificate-authority))
- For the wheelhouse: the GitHub CLI (`gh`)
- Docker with `buildx` if you will pack or run jobs locally
- An NVIDIA GPU if you intend to train locally

## Install from the internal index

```bash
uv venv --python 3.13 .venv
VIRTUAL_ENV=.venv uv pip install --system-certs \
  --index-url https://pypi.lan/carbonteq/stable/+simple/ \
  --constraint github-constraints.txt \
  "posttrain[observatory,trackio,trl]"
```

Add the `dstack` extra when submitting remote GPU jobs:

```bash
VIRTUAL_ENV=.venv uv pip install --system-certs \
  --index-url https://pypi.lan/carbonteq/stable/+simple/ \
  --constraint github-constraints.txt \
  "posttrain[observatory,trackio,trl,dstack]"
```

Obtain `github-constraints.txt` from the framework release you are installing:
it is `release/github-constraints.txt` in the framework repository, and the
release workflow attaches it to the published wheelhouse. It is not served by
the index today.

The internal index also mirrors PyPI, so it is the only index you need.

## Install from the GitHub release wheelhouse

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

When a generated project must resolve the unpublished team release (for
example after `posttrain init`), point uv at the same wheelhouse:

```bash
UV_FIND_LINKS="$(pwd)/posttrain-wheelhouse" \
UV_CONSTRAINT="$(pwd)/posttrain-wheelhouse/github-constraints.txt" \
posttrain init my-model-project --template sft --project-id my-model-project
```

## What to install

Most projects start with the primary command distribution `posttrain`. It
supplies project initialization, diagnostics, catalog inspection, and
work-package commands. It does not make `posttrain-lab` a runtime requirement.

Capability packages and extras (`trl`, `verifiers`, `vllm`, Trackio, W&B) are
selected by what the project executes — see
[Choose capabilities](../README.md#choose-capabilities) for the package table.
Install `posttrain-lab` only when developing the framework or adapting one of
its qualification workflows.

## What a project commits

Every consuming project commits:

- `.posttrain/project.toml`
- `.posttrain/catalog/` overlays
- `.posttrain/work_packages/`
- `pyproject.toml` and `uv.lock`

It does not copy the framework base catalog — that catalog is a versioned
resource inside `posttrain-catalog`. Machine-local scratch, caches, recovery
files, and local Trackio storage live under `.posttrain/state/` or an explicit
state-directory override. Durable artifacts are published through the selected
tracking/artifact backend.

## Install on a remote server

A remote CPU or GPU server receives a **project repository**, not the
framework monorepo:

```bash
git clone <carbonteq-project>
cd <carbonteq-project>
uv sync --system-certs --locked --python 3.13
uv run posttrain doctor
uv run posttrain catalog validate
uv run posttrain work-package validate <name>.yaml
```

The committed project lock selects exact framework, fork, environment, and
accelerator artifacts. Secrets and provider endpoints arrive through the
server's secret manager or environment, never through `.posttrain/` source.
Large scratch or recovery state may use an absolute state-directory override.
See [remote-gpu-qualification.md](./remote-gpu-qualification.md) for the GPU
qualification flow.

## Framework checkout (contributors)

For framework development or a complete reference checkout, clone the exact
tag and use the checked-in lock:

```bash
git clone --branch <release-tag> --depth 1 \
  https://github.com/carbonteq-ai/posttrain.git
cd posttrain
mise install
uv sync --all-packages --locked --python 3.13
```

Then follow [contributing.md](./contributing.md). Checkout developers may use
`uv add` against the workspace clone; that is a maintainer path, not the team
install contract.

## Version compatibility

Job images need **posttrain ≥ 0.2.1** for the trust merge. Re-run
`posttrain job pack` for any image packed before that release. See
[UPGRADING.md](../UPGRADING.md) for the upgrade policy and
[COMPATIBILITY.md](../COMPATIBILITY.md) for the support window.

## Next steps

Continue with [Getting started](./getting-started.md) to configure the
machine, create a project, and run your first job end to end.
