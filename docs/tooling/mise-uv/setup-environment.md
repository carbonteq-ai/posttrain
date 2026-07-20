# Environment setup (uv workspace)

Machine: **RTX 3070 Ti (8 GB)**, i9-12900K, ~62 GB RAM, Pop!_OS / Ubuntu 22.04-class.

## Prerequisites

- Working `nvidia-smi` (driver/userspace aligned — a past mismatch was fixed with reboot + `nvidia-driver-580-open`)
- `mise` + `uv` on `PATH`

```bash
nvidia-smi
mise --version
uv --version
```

## Bootstrap

```bash
cd /home/hammad/projects/rl
mise install                    # Python 3.12 from mise.toml
uv sync --all-packages --python 3.12
```

Optional extras (see conflicts below):

```bash
uv sync --package eval --extra verifiers --python 3.12
uv sync --package serve --extra vllm --python 3.12
uv sync --package serve --extra sglang --python 3.12
uv sync --package train --extra vllm --python 3.12
```

Root `.env` sets `LD_LIBRARY_PATH` for `libcudart.so.13` (vLLM) and `LAB_TRACKIO_PROJECT=lab`.

## Verify

```bash
uv run --package train python -c \
  "import torch, trl; from common import WORKSPACE_ROOT; print(WORKSPACE_ROOT, torch.cuda.is_available(), trl.__version__)"
uv run --package common profile-resolve --help
```

Expect: CUDA `True`, `trl` `1.8.0` from fork commit
`935060f640f5195fe62f1acc300c16db327a32b9`, workspace root
`/home/hammad/projects/rl`. The `train[vllm]` variant must report vLLM `0.25.1`
without a TRL compatibility warning.

## Dependency conflicts (declared in root `pyproject.toml`)

| Pair | Why |
| --- | --- |
| `serve[vllm]` vs `serve[sglang]` | Backends pin incompatible Torch/Transformers stacks |
| `train` vs `eval[verifiers]` | TRL and the pinned Verifiers revision require incompatible `datasets` versions |

Sync the engine variant being developed; do not force resolver-driven downgrades across incompatible stacks.

## Hugging Face

```bash
uv tool install huggingface_hub
huggingface-cli login   # if needed
export HF_HOME="$HOME/.cache/huggingface"
```

## What not to do

- Do not recreate legacy `trl/` / `verifiers/` / `inference/` single-project trees
- Do not put the training loop in Docker
- Do not use 128k–262k context defaults on this GPU

More: [workspace.md](./workspace.md) · [overview](../../overview.md).
