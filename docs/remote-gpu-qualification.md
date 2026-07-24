# Remote GPU qualification

Every stable Posttrain release must install and execute from its published
wheelhouse on a clean remote NVIDIA GPU host. A local source checkout or
workspace environment does not satisfy this gate.

The gate uses the project under `examples/gpu-qualification`, the primary
`posttrain work-package run` command, the lab qualification project entry
(`posttrain_lab.entry:configure` from `project.toml`), local Trackio storage on
the remote machine, and Observatory readback. No `--host` factory flag is
required for the work package; SSH `--host` below is only the remote machine.

## Run the gate

The coordinator requires `gh`, `ssh`, `scp`, and `sha256sum`. The remote host
requires `uv`, Python 3.12, a compatible NVIDIA driver, access to Python package
indexes and model weights, and enough disk space for the model and vLLM.

```bash
uv run python tools/qualify_remote_gpu.py \
  --host <ssh-host> \
  --release <release-tag>
```

The coordinator:

1. Downloads and verifies the tagged GitHub wheelhouse.
2. Creates a new `/tmp/posttrain-gpu-qualification.*` directory remotely.
3. Installs the release into a clean Python 3.12 environment.
4. Records the GPU model, memory, and driver.
5. Validates and executes the bounded Qwen 3.5 2B serving screen.
6. Records a terminal Trackio run and retrieves the same run through
   Observatory.
7. Writes a local evidence summary under
   `.posttrain/state/qualification/`.

The remote directory is intentionally retained for diagnosis. It contains the
environment, logs, downloaded release, local Trackio state, and JSON output.
Remove that specific temporary directory only after the release evidence is
accepted or copied to the release record.

## Acceptance

The gate passes only when the wheelhouse and every wheel checksum validate, the
work package and its benchmark job succeed, a run ID is present, and
Observatory returns that same run ID. The evidence summary must be attached to
the release or referenced by the release checklist before removing the remote
state.

## Local merge gate (no remote GPU)

From the repository root, composition can be checked without CUDA:

```bash
uv run --package posttrain posttrain work-package validate \
  .posttrain/work_packages/foundation_screen.yaml
```

That uses the repo-root qualification `project.toml` entry. Full GPU execution
remains a release gate.
