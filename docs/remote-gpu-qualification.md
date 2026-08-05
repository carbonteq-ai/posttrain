# Remote GPU qualification

Every stable Posttrain release must install and execute from its published
artifacts on a clean remote NVIDIA GPU host. A local source checkout or
workspace environment does not satisfy this gate.

The supported consumer path is documented in
[install.md](./install.md) and [getting-started.md](./getting-started.md):
install from the internal index with
`github-constraints.txt`, trust the private CA, configure dstack storage, and
submit with `posttrain job run --provider dstack`. The gate uses a project under
`examples/` (or an equivalent consumer project), remote Trackio for evidence,
and Observatory readback.

## Run the gate

The coordinator requires `gh`, `ssh`, `scp`, and `sha256sum`. The remote host
requires `uv`, **Python 3.13**, a compatible NVIDIA driver, access to the
internal index / OCI registry / tracking service, and enough disk for the model
and runtime images. Workers must already have `/etc/posttrain/trust/internal-ca.pem`
and the storage paths named in `[providers.dstack.storage]`.

On the remote host, after installing the release (see
[install.md](./install.md)):

```bash
posttrain job run .posttrain/work_packages/<gate>.yaml \
  --provider dstack \
  --target <target-id>
posttrain run reconcile --last
```

Retain the remote evidence until the release checklist accepts or copies a
summary under `.posttrain/state/qualification/`.

## Acceptance

The gate passes only when the installed distributions match the release, the
job succeeds, a run ID is present, `run reconcile` reports consistent retained
evidence, and Observatory returns that same run ID. Attach or reference the
evidence summary on the release checklist before removing remote state.

## Local merge gate (no remote GPU)

From the repository root, composition can be checked without CUDA:

```bash
uv run --package posttrain posttrain work-package validate \
  .posttrain/work_packages/foundation_screen.yaml
```

That uses the repo-root qualification `project.toml` entry. Full GPU execution
remains a release gate.
