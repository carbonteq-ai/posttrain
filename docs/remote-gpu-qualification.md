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

Use the primary CLI on a clean remote host after installing the tagged
wheelhouse. Do not rely on ephemeral `tools/` helpers.

```bash
# On the remote host, after installing the release wheelhouse:
uv run --package posttrain posttrain work-package run \
  examples/gpu-qualification/.posttrain/work_packages/<gate>.yaml \
  --job <job-id>
```

Retain the remote evidence directory until the release checklist accepts or
copies the summary under `.posttrain/state/qualification/`.

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
