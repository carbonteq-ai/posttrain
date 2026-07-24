# Developer environment setup

Use this page when working from a Posttrain source checkout. Product projects
created by `posttrain init` own a separate `.venv` and `uv.lock`; they should
follow the generated project workflow instead.

## Prerequisites

All developers need:

- Git;
- [`mise`](https://mise.jdx.dev/) or an existing Python 3.12 installation;
- [`uv`](https://docs.astral.sh/uv/);
- access to the private or organization-owned Git dependencies selected by
  `uv.lock`.

GPU development additionally needs Linux, a visible NVIDIA GPU, and a driver
new enough for the CUDA 13 runtime selected by the lock.

```bash
git --version
mise --version
uv --version
nvidia-smi                    # GPU profiles only
```

Do not install TRL, Transformers, PyTorch, vLLM, or Verifiers separately with
`pip`. The workspace extras and lockfile select compatible versions and exact
fork commits together.

## 1. Clone and install the core workspace

```bash
git clone git@github.com:carbonteq-ai/posttrain.git
cd posttrain

mise install
uv sync --all-packages --locked --python 3.12
```

This is the default profile for CLI, catalog, contracts, documentation,
Observatory, and CPU tests. It includes the root development group, so Ruff,
Pyright, pytest, and import-linter are available.

Verify the checkout:

```bash
uv run --package posttrain posttrain doctor
uv run pytest -q tests/consumer
uv run python -c "import sys; print(sys.version)"
```

## 2. Select the backend profile you are developing

The profiles are workspace extras on `posttrain-lab`. Use `--all-packages` so
the development tools and applications stay installed while the backend
dependencies are added.

| Work | Command |
| --- | --- |
| Core/CPU development | `uv sync --all-packages --locked --python 3.12` |
| Verifiers evaluation with vLLM | `uv sync --all-packages --extra gpu-eval --locked --python 3.12` |
| TRL trainer work: SFT, DPO, trainer-side tests | `uv sync --all-packages --extra gpu-train --locked --python 3.12` |
| Agentic training: GRPO, DAPO, SAMPO, distillation | `uv sync --all-packages --extra gpu-posttrain --locked --python 3.12` |

`gpu-posttrain` installs the pinned CarbonTeq TRL fork, PyTorch CUDA 13,
vLLM, Verifiers, the CUDA compiler packages, `ninja`, and the local
AutomationBench environment. It is the complete TRL-side integration profile.
After selecting a GPU profile, use `uv run --no-sync ...` for profile-specific
commands. A plain `uv run` may synchronize back to the default profile because
the root project does not select those optional extras.

The active TRL fork revision is recorded in
`packages/train/pyproject.toml`, `uv.lock`, and
[`docs/tooling/trl/README.md`](../trl/README.md). Treat those files as the
authority instead of copying a revision into a local install command.

## 3. Verify a GPU profile

For TRL without vLLM:

```bash
uv run --no-sync python -c \
  "import torch, transformers, trl; print(torch.cuda.is_available(), torch.__version__, transformers.__version__, trl.__version__)"
```

For the complete agentic profile:

```bash
uv run --no-sync python -c \
  "import torch, trl, verifiers, vllm; print(torch.cuda.is_available(), torch.__version__, trl.__version__, verifiers.__version__, vllm.__version__)"
```

Expect `torch.cuda.is_available()` to be `True` on a GPU developer machine.
These import checks prove dependency compatibility; they do not replace a
bounded training qualification.

If vLLM cannot locate the wheel-provided CUDA runtime, add its library directory
for the current shell:

```bash
export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Do not commit an absolute checkout path or credentials in `.env`. The repository
ignores `.env`; provider secrets belong in the developer's secret manager or
local shell.

## Hugging Face access

Model downloads require a Hugging Face token when the selected repository is
gated:

```bash
uv tool install huggingface_hub
hf auth login
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
```

Never commit the token or the Hugging Face cache.

## veRL uses a separate environment

Do not add veRL to the root workspace environment. The supported veRL stack has
different Transformers and vLLM constraints. A veRL training binding must name:

- an absolute `backend_options.python_executable` inside the isolated veRL
  environment;
- an absolute `backend_options.working_directory` for the veRL checkout;
- the checkout's complete `backend_options.source_revision`;
- the isolated lock digest for lineage.

Follow [`docs/tooling/verl/README.md`](../verl/README.md) for the qualified
checkout and runtime contract. The framework launcher verifies the checkout
revision before starting Ray.

## Normal validation

Run the smallest affected tests during development, then the repository ladder:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run lint-imports
uv run pytest
git diff --check
```

Tests that need GPU, network, or Docker are marked and remain separate release
gates. A skipped integration test is not evidence that its backend is
qualified. After installing a GPU profile, invoke its focused tests with
`uv run --no-sync pytest ...`; rerun the desired `uv sync` command when
switching profiles.

## Common setup failures

| Symptom | Check |
| --- | --- |
| Wrong Python or resolver churn | Run `mise install`, then sync with `--locked --python 3.12` |
| `torch.cuda.is_available()` is false | Check `nvidia-smi`, driver/runtime compatibility, and that a GPU profile was installed |
| `libcudart.so.13` is missing | Export the wheel CUDA library path shown above |
| TRL/Transformers/Verifiers conflict | Remove hand-installed packages and resync the selected locked profile |
| vLLM build or first-use kernel failure | Confirm the `gpu-posttrain` profile installed CUDA compiler packages and `ninja` |
| veRL import conflict | Use the isolated veRL interpreter; never install veRL into `.venv` |

More detail: [workspace layout](./workspace.md), [TRL](../trl/README.md),
[veRL](../verl/README.md), [Verifiers](../verifiers/README.md), and
[vLLM](../vllm/README.md).
