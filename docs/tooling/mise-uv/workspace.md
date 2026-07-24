# uv workspace

The root project is a Python 3.12 uv workspace containing every reusable
package and application under `packages/*` and `apps/*`.

## Ownership

```text
packages/common       framework-neutral contracts
packages/data         datasets and rollout data
packages/train        SFT, DPO, GRPO, DAPO, SAMPO, and distillation
packages/eval         endpoint-neutral Verifiers evaluation
packages/serve        inference and serving adapters
packages/jobs         standard job definitions
packages/work         work-package composition
apps/cli              primary posttrain CLI
apps/lab              reference qualification application
apps/observatory      read-only evidence product
```

The workspace has one root `.venv` and one `uv.lock`. Inter-package
dependencies use `{ workspace = true }`. PyTorch and torchvision resolve from
the explicit CUDA 13 index in the root configuration.

## Common commands

```bash
# Core framework and developer tools
uv sync --all-packages --locked --python 3.12

# TRL training dependencies
uv sync --all-packages --extra gpu-train --locked --python 3.12

# TRL + vLLM + Verifiers agentic training dependencies
uv sync --all-packages --extra gpu-posttrain --locked --python 3.12

uv run pytest
uv run lint-imports
```

Use `uv run --no-sync ...` for backend-specific commands after selecting an
extra; otherwise uv may synchronize the environment back to the core profile.

Do not create per-application environments inside this checkout. veRL is the
intentional exception: it runs from a separate checkout and interpreter because
its backend dependency stack conflicts with the locked TRL environment.

See [developer environment setup](./setup-environment.md) for prerequisites,
profile selection, verification, and troubleshooting.
