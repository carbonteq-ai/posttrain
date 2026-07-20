# uv workspace

Root project `lab` is a **uv workspace** (Cargo-style monorepo).

## Members

```text
packages/common   # library
packages/train    # reusable TRL training engine
packages/eval     # reusable Verifiers/evaluation engine
packages/serve    # reusable vLLM/SGLang inference engine
```

- One root `uv.lock` + `.venv`
- Inter-package deps: `{ workspace = true }` on `common`
- PyTorch CUDA wheels: index `pytorch-cu128` (explicit) in root + engine-package sources
- Python: `>=3.12,<3.13` (pin with `--python 3.12`)

## Common commands

```bash
uv sync --all-packages --python 3.12
uv run --all-packages --group dev pytest -q
uv run --all-packages --group dev lint-imports
```

Install heavy engine variants only when needed:

```bash
uv sync --package serve --extra vllm --python 3.12
uv sync --package serve --extra sglang --python 3.12
uv sync --package eval --extra verifiers --python 3.12
```

## Conflicts

See root `pyproject.toml` `[tool.uv] conflicts`. Package-level conflicts are **experimental** in uv (may warn unless `--preview-features package-conflicts`).

The prototype applications, task package, local catalog, executable config tree, and normalized result store were removed. They are not workspace members or compatibility surfaces.
