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
sudo cp internal-ca.pem /usr/local/share/ca-certificates/internal-ca.crt && sudo update-ca-certificates
```

```bash
sudo install -D -m 644 internal-ca.pem /etc/posttrain/trust/internal-ca.pem
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

One setting is required before any job can run locally, and it is not
scaffolded. It lives in `<project>/.posttrain/state/execution.toml`, which
must not be readable by group or others.

```bash
cat > my-project/.posttrain/state/execution.toml <<EOF
schema_version = 1

[providers.local]
# Identifies this physical machine so the admission controller can hold one
# lease per host across local and dstack access paths.
canonical_hostname = "$(hostname)"

EOF
chmod 600 my-project/.posttrain/state/execution.toml
```

Nothing about certificates belongs here. The authority installed in step 1 is
found automatically. Override it only for a one-off, either with
`POSTTRAIN_TRUST_BUNDLE` in the environment or `trust_bundle` under
`[providers.local]`; a path named that way must exist, because silently
substituting a different authority would be worse than refusing.

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
