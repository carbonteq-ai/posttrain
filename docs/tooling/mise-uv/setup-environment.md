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

Optional application/runtime extras:

```bash
uv sync --package posttrain-lab --extra gpu-eval --python 3.12
uv sync --package posttrain-lab --extra gpu-train --python 3.12
uv sync --package posttrain-lab --extra gpu-posttrain --python 3.12
```

Root `.env` sets `LD_LIBRARY_PATH` for `libcudart.so.13` (vLLM) and `LAB_TRACKIO_PROJECT=lab`.

## Verify

```bash
uv run --package posttrain-lab --extra gpu-posttrain python -c \
  "import datasets, torch, trl, verifiers; print(torch.cuda.is_available(), datasets.__version__, trl.__version__, verifiers.__version__)"
```

Expect CUDA `True`, `datasets` `4.6.1`, and `trl` `1.8.0` from fork commit
`b31dc19ad82b0f8fcba77ee1bdf7bd03986a193d`.

The CarbonTeq TRL compatibility pin removes the former TRL/Verifiers datasets
conflict. Serving engines remain optional runtime choices and should only be
installed when that backend is being exercised.

The vLLM extra includes the CUDA compiler packages and `ninja`. TurboQuant may
route sampling through a FlashInfer kernel that is compiled on first use; NVCC
without `ninja` is not a complete runtime for that profile.

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
