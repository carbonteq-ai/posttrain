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

The internal services present certificates from a private CA. Install it into
the system trust store so `curl`, `uv`, and the CLI all accept them:

```bash
sudo cp carbonteq-local-ai-caddy.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates
```

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

```bash
curl -sSLO https://pypi.lan/carbonteq/stable/+f/github-constraints.txt
```

```bash
VIRTUAL_ENV=.venv uv pip install --system-certs --index-url https://pypi.lan/carbonteq/stable/+simple/ --constraint github-constraints.txt "posttrain[observatory,trackio,trl]"
```

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

## 5. Configure the local execution provider

Two settings are required before any job can run locally, and neither is
scaffolded. They live in `<project>/.posttrain/state/execution.toml`, which
must not be readable by group or others.

```bash
cat > my-project/.posttrain/state/execution.toml <<EOF
schema_version = 1

[providers.local]
# Identifies this physical machine so the admission controller can hold one
# lease per host across local and dstack access paths.
canonical_hostname = "$(hostname)"

# The container does not inherit the host trust store, so the bundle is mounted
# explicitly. It must be the COMPLETE set of certificate authorities the job
# should trust: it becomes SSL_CERT_FILE inside the container, replacing the
# image's own bundle rather than adding to it. A private CA alone leaves
# huggingface.co untrusted, which surfaces much later as a model download
# failure rather than as a trust problem.
trust_bundle = "/absolute/path/to/combined-ca.crt"
EOF
chmod 600 my-project/.posttrain/state/execution.toml
```

Build the combined bundle by concatenating the system store with the internal
CA:

```bash
cat /etc/ssl/certs/ca-certificates.crt internal-ca.crt > combined-ca.crt
```

## 6. Check readiness

```bash
posttrain doctor
```

Every line should read `OK`. `registry` and `runtime_images` report `WARN` when
`POSTTRAIN_REGISTRY` is unset, which means you did not source the environment.

You can confirm the framework's published images independently:

```bash
posttrain runtime images verify
```

## 7. Run a job

`plan` resolves without building. `pack` builds and publishes the actual-job
image. `run` does both and submits.

```bash
posttrain job run .posttrain/work_packages/sft.yaml --job train
```

Then follow it, and join the provider's result to the retained evidence:

```bash
posttrain run status <run-id>
```

```bash
posttrain run reconcile <run-id>
```

A finished job reads `Reconciliation: consistent` with `Missing required
roles: none`.

## Things that will bite you

- **`--job` is required** even when the recipe has exactly one job.
- **Unknown subcommands print a traceback** rather than a usage error, and
  there is no flag to get a traceback when you actually want one.
- **A pack can fail with `package manifest key differs from PACKAGE_KEY`.**
  This is a stale build cache, not a real mismatch. Retry the pack; if it
  persists, run `docker buildx prune -af`.
- **A newly mirrored package can 404 once** before the index caches it. Retry.
- **A failed run holds its machine's admission slot** until you run
  `posttrain run reconcile <run-id>`. Later runs sit at queue position 1 until
  you do.
