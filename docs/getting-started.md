# Getting started

This is the first-day walkthrough for a **project developer**: someone using
Posttrain to prepare data, train, evaluate, and serve models — not someone
working on the framework itself (that path is [contributing.md](./contributing.md)).

By the end you will have: the internal services trusted, the framework
installed, this machine configured once for every project on it, a runnable
project scaffolded from a starter template, one job executed (locally or on a
remote GPU through dstack), and that job's trained model handed to a follow-up
evaluation package. Every step here was executed against a real deployment
before it was written; the commands come from a machine with no framework
checkout anywhere on it.

If a term is unfamiliar — work package, catalog, binding, admission, evidence
— the [glossary](./glossary.md) defines all of them.

## What you need

- Python 3.13 and `uv`
- Docker with `buildx`
- Network access to the internal index, the OCI registry, and the tracking
  service
- An NVIDIA GPU if you intend to run training locally

The services this guide points at — the Python index at `pypi.lan`, the OCI
registry at `registry.lan`, the tracking server at `trackio.lan`, and the
dstack server — are operated from the `ai-infra` repository, not from this
one. If a name does not resolve or a service is down, that repository owns it;
its `docs/operations/` runbooks cover the index, worker enrollment, the image
builder, and workstation trust. This guide assumes they are already running.

If something misbehaves along the way, check
[Things that will bite you](#things-that-will-bite-you) at the end — it lists
the known first-week traps.

## 1. Trust the internal certificate authority

This comes first because nothing else works without it: the package index,
registry, and tracking server all present certificates from a private CA, so
until it is trusted, installs and job submissions fail with TLS errors.

The CA has to be installed in two places because they are read by different
things: host tools such as `uv`, `curl`, and the Docker daemon consult the
machine's own store, while a job container inherits nothing from the host and
is given the authority separately.

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

Follow the [installation guide](./install.md) — it is the single source of
truth for the install commands, covering both the internal index
(`pypi.lan`, the standard path on the CarbonTeq network) and the GitHub
release wheelhouse, plus the required release constraints file.

After this step you should have a `.venv` whose `posttrain` command runs, for
example installed as `posttrain[observatory,trackio,trl]` — add the `dstack`
extra if you will submit remote GPU jobs (§6).

## 3. Initialize this machine

Machine defaults are shared by every project and load automatically; they do
not depend on the shell that launched the CLI. Initialize them once:

```bash
posttrain machine init \
  --trackio-endpoint https://trackio.lan \
  --python-index-url https://pypi.lan/carbonteq/stable/+simple/ \
  --job-registry registry.lan/carbonteq
```

This writes the non-secret `$XDG_CONFIG_HOME/posttrain/config.toml` (normally
`~/.config/posttrain/config.toml`) and mode-0600 files beneath
`~/.config/posttrain/credentials/`. Put only the values this machine uses in
their scoped files:

```bash
printf '%s\n' 'TRACKIO_WRITE_TOKEN=...' > ~/.config/posttrain/credentials/trackio.env
printf '%s\n' 'HF_TOKEN=...' > ~/.config/posttrain/credentials/huggingface.env
chmod 600 ~/.config/posttrain/credentials/*.env
```

The config stores endpoints and credential *names*, never token values. The
framework injects each source only into its consumer: a dstack token is not
part of a job's runtime map, and Trackio credentials are not sent to dstack's
client process merely because both are configured on the same machine.

## 4. Create a project

`--template` scaffolds a runnable starter and builds its environment. The
available templates are `sft` and `grpo`.

```bash
posttrain init my-project --template sft
posttrain machine project add "$PWD/my-project"
```

You may instead pass one or more `--project` options during the first
`machine init`. Project registration is idempotent.
`posttrain.env` remains an ignored, mode-0600 project override for values that
genuinely differ by project. It is auto-loaded and never needs to be sourced.
For a one-off job-image destination, make the deviation explicit instead of
changing the shell environment or the shared machine default:

```bash
posttrain job run .posttrain/work_packages/sft.yaml --registry registry.example/team
```

## 5. Local execution provider

Local execution derives the canonical hostname at runtime unless the operator
explicitly passed `--machine-name`. Its admission ledger is machine-scoped, so
local Docker jobs from every registered project serialize against the same
physical resources. Mutable run and cache paths under `[storage]` resolve
beneath `$XDG_STATE_HOME/posttrain`, not beneath the configuration directory.

Trust belongs in the machine config's `[trust]` table. Projects cannot replace
the machine trust root. A named path must exist because silently substituting
a different authority would be worse than refusing.

## 6. Run on dstack

Remote GPU jobs use dstack as the placer. posttrain submits a resource ask;
dstack owns offers, placement, startup, and cancellation. Local Docker still
uses the machine admission ledger (`posttrain workers`); dstack runs do not
take a host lock inside posttrain.

Install with the `dstack` extra (§2), then initialize the client binding
with `posttrain machine init` or add this to the existing machine config:

```toml
[providers.dstack]
project = "main"
python = "/absolute/path/to/dstack-venv/bin/python"
credentials = "dstack-default"
# Persist a pre-start no-capacity task in dstack for up to one day.
capacity_wait_seconds = 86400

[credentials.dstack-default]
file = "credentials/dstack.env"
```

`python` is the **client** interpreter that talks to the dstack server, not the
job image. The referenced credential file must be mode 0600. Worker storage is
not a developer-machine setting: the execution-dstack contract and ai-infra
Ansible deployment own `/var/lib/posttrain/runs`, model cache, compile cache,
and the worker CA. Your laptop needs only the system CA (§1) plus this
client binding. `capacity_wait_seconds` is a server-side
dstack queue retention window. It retries only `no-capacity` before the job
starts; interruption and runtime errors remain fail-fast so user code is never
repeated under the same framework attempt. This dstack release defaults an
omitted retry duration to one hour and has no unbounded value.

Targets declare capacity (`device_class` / `memory_gb`). That is enough for
dstack to place the run on any matching worker. posttrain does **not** lock a
hostname for dstack jobs; affinity is optional and lives only in the catalog
target's `placement`, which is forwarded to dstack as a soft pin.

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

**Optional exact pin** — require one known worker when you care which box runs
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
posttrain run queue
posttrain run list
posttrain run logs --last --follow
posttrain run reconcile --last
posttrain run cleanup --last
```

`reconcile` joins provider state to retained tracking evidence. A failed or
cancelled run settles without waiting for artifacts that will never arrive, so
admission (for local) or the next dstack submit is not blocked forever.
`cleanup` releases provider resources after evidence is retained.
`run queue` separates framework admission from provider capacity waiting and
shows the requested target/host, assigned hostname (once present), provider
state, and provider run identity.
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

## 9. Pass one job's model into the next

Jobs do **not** auto-wire outputs. Two jobs in the same recipe share the same
catalog bindings; the eval job does not see the train job's new weights unless
you pin those weights as a catalog model and bind them on a later package.

Do this today:

1. **Run the producer** (for example SFT). It should publish a loadable model
   directory to Trackio and record a produced-artifact edge on the run.
2. **Copy the immutable identity** from the run / Observatory / Trackio UI:
   project, artifact name, and version (`vN`). Aliases such as `candidate` are
   planning helpers only — bind a concrete `vN`, never `latest`.
3. **Register a project overlay** `ModelVariant` that points at that Trackio
   artifact. List the file in `.posttrain/catalog/layer.yaml`.
4. **Author a consumer work package** whose `bindings.model` uses the new
   catalog id, then `plan` / `run` that package.

```yaml
# .posttrain/catalog/models.yaml
model:
  models/my-agent-2b@sft-v3:
    artifact:
      kind: trackio
      project: my-trackio-project
      name: ambient-agent-2b-sft
      version: v3
    form: peft-adapter          # or merged / full-finetuned — match the files
    weight_precision: bf16
    family: qwen3.5
    parameters: 2000000000
    instruction_tuned: true
    renderer_contract: qwen3.5-tools@1
    # Required today: Hub base the adapter/descendant was trained from
    base:
      kind: hub
      repo_id: Qwen/Qwen3.5-2B
      revision: 15852e8c16360a2fea060d615a32b45270f8a8fc
    capabilities:
      modalities: [text, image]
      native_context_window: 262144
      mtp: true
    parent: models/qwen3.5-2b@bf16   # optional catalog lineage pointer
    provenance:
      producer_run: "<run-id>"
```

```yaml
# .posttrain/catalog/layer.yaml  (must list the overlay file)
schema_version: 1
layer_id: my-project-v1
files:
  - settings.yaml
  - models.yaml
```

```yaml
# .posttrain/work_packages/qualify-eval.yaml  (consumer)
bindings:
  model:
    type: ref
    family: model
    id: models/my-agent-2b@sft-v3
```

```bash
posttrain doctor
posttrain job plan .posttrain/work_packages/qualify-eval.yaml --job eval
posttrain job run  .posttrain/work_packages/qualify-eval.yaml --job eval
```

After a gate passes, move a Trackio alias such as `qualified` to that exact
`vN` if your team uses aliases for humans. The next work package still binds
the versioned catalog id (or a new overlay pinned to the same `vN`).

Lineage is the run's consumed/produced edges (and optional `parent`), not the
order of work-package files. Product rules:
[03 · Work and evidence](./post-training/03-work-and-evidence.md),
[06 · Observation](./post-training/06-observation-and-lineage.md).
Ops detail for Trackio blob storage:
[operations/dstack-trackio/object-storage.md](./operations/dstack-trackio/object-storage.md).

## Things that will bite you

- **Hand-authored catalog YAML must be listed in `layer.yaml`** or
  `doctor` warns and the selection is invisible.
- **Passing train → eval/GRPO is not automatic.** Pin the Trackio `vN` as a
  project overlay model (`kind: trackio`) and bind that id on the next package
  (see §9). Same-recipe jobs share seats; they do not chain outputs.
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
