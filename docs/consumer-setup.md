# Setting up as a developer

Every step here was executed against a real deployment before it was written.
The commands come from a machine with no framework checkout anywhere on it.

## What you need

- Python 3.13 and `uv`
- Docker with `buildx`
- Network access to the internal index, the OCI registry, and the tracking
  service
- An NVIDIA GPU if you intend to run training locally

## 1. Trust the internal certificate authority

The internal services present certificates from a private CA, and it has to be
installed in two places. They are read by different things: host tools such as
`uv`, `curl`, and the Docker daemon consult the machine's own store, while a
job container inherits nothing from the host and is given the authority
separately.

```bash
sudo cp /path/to/ai-infra/.state/certs/caddy-local-root.crt \
  /usr/local/share/ca-certificates/internal-ca.crt && sudo update-ca-certificates
```

```bash
sudo install -D -m 644 /path/to/ai-infra/.state/certs/caddy-local-root.crt \
  /etc/posttrain/trust/internal-ca.pem
```
The second path is where the framework looks by default, so a machine prepared
this way needs no trust configuration in any project. That file holds the
internal authority **alone** — never a hand-assembled union with the system
store. The job image merges it with the authorities it already has, and a
union assembled here would instead pin every job to whatever the machine that
submitted it happened to trust.

`posttrain doctor` reports which authority jobs will be given and warns if it
reaches jobs but is missing from this machine's own store.

If you run a VPN that captures DNS, the `.lan` names will not resolve. Check
with `getent hosts pypi.lan` before assuming the service is down.

## 2. Install the framework

The framework is served by the internal index, which also mirrors PyPI, so it
is the only index you need. Some dependencies are maintained forks pinned to
immutable commits; uv will not resolve a transitive direct URL unless it is
also declared at the top level, so the constraints file is required rather than
optional.

```bash
uv venv --python 3.13 .venv
```

Obtain `github-constraints.txt` from the framework release you are installing.
It is `release/github-constraints.txt` in the framework repository, and the
release workflow attaches it to the published wheelhouse. There is currently no
way to fetch it from the index, which is a gap: the constraints are required to
install, so they should travel with the distributions rather than beside them.

```bash
VIRTUAL_ENV=.venv uv pip install --system-certs --index-url https://pypi.lan/carbonteq/stable/+simple/ --constraint github-constraints.txt "posttrain[observatory,trackio,trl]"
```

For remote GPU through dstack, include the extra:

```bash
VIRTUAL_ENV=.venv uv pip install --system-certs --index-url https://pypi.lan/carbonteq/stable/+simple/ --constraint github-constraints.txt "posttrain[observatory,trackio,trl,dstack]"
```

Job images need **posttrain ≥ 0.2.1** for the trust merge. Re-run
`posttrain job pack` for any image packed before that release.
## 3. Set the environment

Write `posttrain.env` and source it before every command. Keep it `chmod 600`:
it carries a write token.

```bash
cat > posttrain.env <<'EOF'
# The framework and its forked dependencies are served by the internal index,
# which also mirrors PyPI, so this is the only index a developer needs.
UV_INDEX_URL=https://pypi.lan/carbonteq/stable/+simple/
UV_CONSTRAINT=/absolute/path/to/github-constraints.txt

# The internal services present certificates from a private CA that is already
# in the system trust store. uv uses rustls by default and reads this variable.
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Where this project publishes its own actual-job images. This is the project's
# registry, not the framework's release registry: the framework publishes its
# base and job-kind images once per release, and a project pushes the job
# images it builds on top of them here.
POSTTRAIN_REGISTRY=registry.lan/carbonteq

# The job publishes run evidence to the internal tracking service. Both names
# are forwarded into the job container, and both are needed to read the
# evidence back afterwards.
POSTTRAIN_TRACKIO_SERVER_URL=https://trackio.lan
TRACKIO_WRITE_TOKEN=...
EOF
chmod 600 posttrain.env
```

```bash
set -a; . ./posttrain.env; set +a
```

## 4. Create a project

`--template` scaffolds a runnable starter and builds its environment. The
available templates are `sft` and `grpo`.

```bash
posttrain init my-project --template sft
```

## 5. Local execution provider

`posttrain init` writes `.posttrain/state/execution.toml` (mode 0600) with
`providers.local.canonical_hostname` set from this machine's hostname. That
identity is how the admission ledger serializes local Docker jobs across
projects on the same GPU. Edit it only if the hostname is wrong.

Nothing about certificates belongs here. The authority installed in step 1 is
found automatically. Override it only for a one-off, either with
`POSTTRAIN_TRUST_BUNDLE` in the environment or `trust_bundle` under
`[providers.local]`; a path named that way must exist, because silently
substituting a different authority would be worse than refusing.

## 6. Run on dstack

Remote GPU jobs use dstack as the placer. posttrain submits a resource ask;
dstack owns offers, placement, startup, and cancellation. Local Docker still
uses the machine admission ledger (`posttrain workers`); dstack runs do not
take a host lock inside posttrain.

Install with the `dstack` extra (step 2). Then extend
`.posttrain/state/execution.toml`:

```toml
[providers.dstack]
project = "main"
python = "/absolute/path/to/dstack-venv/bin/python"
# environment_file = "/absolute/path/to/dstack.env"   # optional

[providers.dstack.storage]
run_root = "/var/lib/posttrain/runs"
model_cache = "/var/lib/posttrain/cache/huggingface"
# compile_cache = "/var/lib/posttrain/cache/compile"  # optional
```

`python` is the **client** interpreter that talks to the dstack server, not the
job image. Storage paths are what the **workers** mount; Ansible usually owns
those directories and the well-known CA at `/etc/posttrain/trust/internal-ca.pem`.
Your laptop only needs the system CA (step 1) plus this client binding.
`trust_bundle` under `[providers.dstack]` is rarely required when workers
already have the well-known path.

Targets declare capacity (`device_class` / `memory_gb`). That is enough for
dstack to place the run on any matching worker. posttrain does **not** lock a
hostname for dstack jobs; affinity is optional and lives only in the catalog
target’s `placement`, which is forwarded to dstack as a soft pin.

**Default (no pin)** — capacity only:

```yaml
target:
  targets/any-cuda-24gb:
    revision: "1"
    device_class: nvidia-cuda
    memory_gb: 24
    placement:
      world_size: 1
```

**Optional soft pin** — prefer one known worker when you care which box runs
the job (debug, local qualification, a machine with a warm cache). Use a list
of objects with `hostname`, not bare strings (dstack treats a string as a
different selector shape and will not pin the host you meant):

```yaml
target:
  targets/carbonteq-rtx-pro-6000-96gb:
    revision: "1"
    device_class: nvidia-cuda
    memory_gb: 96
    placement:
      world_size: 1
      instances:
        - hostname: carbonteq-ai-workstation.lan
```

Omit `instances` unless you need that pin. `fleets` may be listed the same way
when you want a fleet selector instead of a single host. If a VPN captures DNS,
`.lan` names will not resolve and placement fails — check with
`getent hosts` before debugging dstack itself.

```bash
posttrain job plan .posttrain/work_packages/sft.yaml --provider dstack --target <target-id>
posttrain job pack .posttrain/work_packages/sft.yaml --provider dstack --target <target-id>
posttrain job run  .posttrain/work_packages/sft.yaml --provider dstack --target <target-id>
```

`--target foo@1` is accepted when the catalog revision is `1`. After submit:

```bash
posttrain run status --last
posttrain run logs --last --follow
posttrain run reconcile --last
posttrain run cleanup --last
```

`reconcile` joins provider state to retained tracking evidence. A failed or
cancelled run settles without waiting for artifacts that will never arrive, so
admission (for local) or the next dstack submit is not blocked forever.
`cleanup` releases provider resources after evidence is retained.
`cancel` asks the provider to stop; use `recover-cancelled-tracking` when
tracking was left open by a hard cancel.

Fork and ops notes live under [tooling/dstack](./tooling/dstack/README.md).

## 7. Check readiness

```bash
posttrain doctor
```

Every line should read `OK`. `registry` and `runtime_images` report `WARN` when
`POSTTRAIN_REGISTRY` is unset, which means you did not source the environment.

You can confirm the framework's published images independently:

```bash
posttrain runtime images verify
```

## 8. Run a job

`plan` resolves without building. `pack` builds and publishes the actual-job
image. `run` does both and submits.

```bash
posttrain job run .posttrain/work_packages/sft.yaml
```

`--job` is optional when the package has exactly one enabled job. Then follow
it and join the provider's result to the retained evidence:

```bash
posttrain run status --last
# or: posttrain run status <run-id-or-prefix>
```

```bash
posttrain run logs --last --follow
```

```bash
posttrain run reconcile --last
```

A finished job reads `Reconciliation: consistent` with `Missing required
roles: none`. Use `posttrain workers` to see who holds a local GPU placement.

## Things that will bite you

- **Hand-authored catalog YAML must be listed in `layer.yaml`** or
  `doctor` warns and the selection is invisible.
- **A pack can fail with `package manifest key differs from PACKAGE_KEY`.**
  This is usually a stale BuildKit cache. Retry the pack; if it persists, run
  `docker buildx prune -af`.
- **A newly mirrored package can 404 once** before the index caches it. Retry.
- **A failed run holds its machine's admission slot** until you run
  `posttrain run reconcile <run-id>` (or `--last`). Later local runs sit at
  queue position 1 until you do. `posttrain workers` names the holder.
- **Cancel does not finish tracking by itself.** After
  `posttrain run cancel --last`, run `recover-cancelled-tracking` when needed,
  then `reconcile`, then `cleanup` once evidence is retained.
- Use `--traceback` when an `error:` line is not enough.