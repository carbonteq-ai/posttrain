# Installing Posttrain

This page is the single source of truth for installing the framework. It is
written for **project teams** consuming a released version. If you are working
on the framework itself, use a workspace checkout instead — see
[contributing.md](./contributing.md). If you are publishing a release, see
[release-engineering.md](./release-engineering.md).

Every release is a versioned wheelhouse attached to its
[GitHub Release](https://github.com/carbonteq-ai/posttrain/releases). It holds
every Posttrain distribution plus the pure-Python wheels of the maintained
forks it pins (Trackio, renderers, TRL), so the wheelhouse together with public
PyPI is a complete install source. The release **constraints file is
required, not optional**: some dependencies are maintained forks pinned to
immutable Git commits, and uv will not resolve a transitive direct URL unless
it is also constrained at the top level.

## Prerequisites

- Python 3.13 and [`uv`](https://docs.astral.sh/uv/)
- Docker with `buildx` if you will pack or run jobs locally
- An NVIDIA GPU if you intend to train locally

## Install from the release wheelhouse

```bash
VERSION=0.4.16
curl -LO "https://github.com/carbonteq-ai/posttrain/releases/download/v${VERSION}/posttrain-wheelhouse-${VERSION}.tar.gz"
mkdir posttrain-wheelhouse
tar -xzf "posttrain-wheelhouse-${VERSION}.tar.gz" -C posttrain-wheelhouse

uv venv --python 3.13
uv pip install --python .venv/bin/python \
  --constraint ./posttrain-wheelhouse/github-constraints.txt \
  --find-links ./posttrain-wheelhouse \
  posttrain posttrain-observatory
```

Wheelhouses from 0.4.16 onward bundle the fork wheels; earlier wheelhouses
also needed a private package index. Verify the download against the release's
`release-SHA256SUMS` asset if you need to audit the bytes.

When a generated project must resolve the same release (for example after
`posttrain init`), point uv at the wheelhouse:

```bash
UV_FIND_LINKS="$(pwd)/posttrain-wheelhouse" \
UV_CONSTRAINT="$(pwd)/posttrain-wheelhouse/github-constraints.txt" \
posttrain init my-model-project --template sft --project-id my-model-project
```

### GPU training extras

The CUDA build of `causal-conv1d` used by the GPU training profiles is a
250 MB platform wheel, so it is not bundled. Download it from its
[public release](https://github.com/carbonteq-ai/causal-conv1d/releases) into
the wheelhouse directory before installing a GPU extra:

```bash
curl -L -o posttrain-wheelhouse/causal_conv1d-1.7.0+cu130torch2.13-cp313-cp313-linux_x86_64.whl \
  "https://github.com/carbonteq-ai/causal-conv1d/releases/download/carbonteq-v1.7.0%2Bcu130torch2.13/causal_conv1d-1.7.0%2Bcu130torch2.13-cp313-cp313-linux_x86_64.whl"
```

## Install from a private package index (optional)

Organizations can mirror a release into their own PEP 503 index; the release
process qualifies the wheelhouse files and promotes the same bytes unchanged.
Point uv at the mirror and keep the release constraints file:

```bash
uv venv --python 3.13 .venv
VIRTUAL_ENV=.venv uv pip install \
  --index-url https://<your-index>/simple/ \
  --constraint github-constraints.txt \
  "posttrain[observatory,trackio,trl]"
```

Add the `dstack` extra when submitting remote GPU jobs. If the index uses a
private certificate authority, trust it first
([Getting started §1](./getting-started.md#1-trust-the-internal-certificate-authority)).
`github-constraints.txt` is `release/github-constraints.txt` in the framework
repository and is included in every wheelhouse.

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
git clone <your-project-repository>
cd <your-project-repository>
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
`uv add` against the workspace clone; that is a maintainer path, not the
release install contract.

## Version compatibility

Job images need **posttrain ≥ 0.2.1** for the trust merge. Re-run
`posttrain job pack` for any image packed before that release. See
[UPGRADING.md](../UPGRADING.md) for the upgrade policy and
[COMPATIBILITY.md](../COMPATIBILITY.md) for the support window.

## Next steps

Continue with [Getting started](./getting-started.md) to configure the
machine, create a project, and run your first job end to end.
