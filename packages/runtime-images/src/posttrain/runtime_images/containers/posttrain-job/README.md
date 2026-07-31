# Actual-job OCI image

This directory defines the framework-owned third image level. The fixed
Dockerfile derives from one digest-pinned job-kind image and consumes a
content-addressed context produced by `JobPackService`. Infrastructure supplies
BuildKit and a registry; it does not choose or assemble the job contents.

## Staged context contract

The named BuildKit context `job-context` has this exact top-level layout:

```text
package.json                         # JobPackageManifest
locks/
  runtime.requirements.txt           # compatibility copy of the control closure
  runtime.control.requirements.txt   # Python 3.12 control closure
  runtime.backend.requirements.txt   # veRL-only Python 3.13.12 closure
  code.requirements.txt              # local framework/project source paths
wheels/
  environments/                      # wheels built from locked environment Git sources
sources/
  framework/                         # deterministic framework source snapshot
  project/                           # deterministic project source snapshot
config/
  resolved.json                      # resolved, non-secret job configuration
datasets/                            # immutable snapshots named by manifest-relative paths
```

Runtime requirements are resolved across every selected environment and every
external project dependency. Ordinary profiles produce the Python 3.12
`runtime.control.requirements.txt`; `runtime.requirements.txt` remains a
compatibility copy during the manifest-v1 migration. The
`online-rl-verl-py313` profile additionally produces an independently resolved
Python 3.13.12 `runtime.backend.requirements.txt`. It is installed only into
`/opt/posttrain-verl`, while the control closure is installed into
`/opt/posttrain/venv`.

Every active requirement carries a SHA-256 hash. Environment wheels are
referenced through `./wheels/environments/...`. The compiler expands every
transitive dependency into each applicable interpreter-specific lock and
records packages already supplied by the immutable kind image. The image
installs every explicit lock line with `--require-hashes --no-deps`; this
prevents wheel metadata from reintroducing an unhashed kind-provided dependency
without omitting any compiled dependency.

`code.requirements.txt` contains only normalized local paths beginning with
`./sources/framework/` or `./sources/project/`. The runtime lock has already
resolved their complete external dependency closure, so this later source
installation uses `--no-deps` without weakening environment resolution. It
also uses `--no-build-isolation --no-sources`; every selected source tree's
build backend must therefore be included in the hash-locked runtime closure
instead of being downloaded implicitly while installing code, and
checkout-only `tool.uv.sources` overrides cannot leak into the isolated image.

Activations whose tasksets require runtime network access must declare
`qualification: deferred`. Packaging rejects those activations by default;
the explicit `--allow-deferred-qualification` CLI waiver makes the live job,
rather than the offline image smoke, own the `Taskset.load()` gate.

The Dockerfile derives a minimal top-level uv workspace from the hashed
`code.requirements.txt` immediately before installing source. This keeps the
framework and project snapshots in separate namespaces while the install
explicitly ignores their checkout-only source overrides. The derived workspace
adds no new input: its sorted members are exactly the local paths already
covered by `code_requirements_digest`.
Named package indexes are projected from the framework root `pyproject.toml`,
which is retained under `sources/framework` and covered by
`framework_source_digest`; this lets package-level `tool.uv.sources` entries
remain valid without permitting an unrecorded build-time index.

Dataset `package_path` values in `package.json` are relative paths beneath
`datasets/`. Base-model weights, mutable checkpoints, final models, run IDs,
attempts, providers, targets, mounts, credentials, and secrets are not staged.
The launch envelope and execution environment provide run-specific values;
model weights use external immutable references and persistent worker caches.

`kind_profile` remains the logical dependency family (`online-rl`, `eval`, and
so on). `runtime_variant` is the exact compatible profile within that family,
for example the TRL single-environment runtime or a veRL two-process runtime.
It is part of the package identity and selects the exact constraint profile and
kind-image digest; it is not a registry publication setting.

Framework and project distributions install into the kind image's inherited
Python 3.12 control environment at `VIRTUAL_ENV`. For
`online-rl-verl-py313`, the exact selected environment-wheel lock is also
installed into `/opt/posttrain-verl`, because the isolated Ray workers
reconstruct the portable Verifiers bridge. The framework distributions remain
Python 3.12-only; the actual-job layer instead projects the content-addressed
`posttrain.common`, `posttrain.data`, and `posttrain.train` namespace sources
under `/opt/posttrain-verl/projection`. The launcher prepends that directory to
the isolated Python 3.13 process only. Project source is never projected into
the backend environment.

The fixed Dockerfile is its own small build context. The staged job directory
is attached as a named context rather than copying a generated Dockerfile into
each package.

## Publication and qualification

The packing service supplies every variable below from the immutable package
plan and publication plan:

```bash
STAGED_CONTEXT=/absolute/path/to/staged-context \
POSTTRAIN_KIND_IMAGE=registry.lan/carbonteq/posttrain-kind-online-rl-trl-py312@sha256:<digest> \
PACKAGE_KEY=<sha256> \
JOB_KIND=train.grpo \
RUNTIME_VARIANT=online-rl-trl-py312 \
FRAMEWORK_SOURCE_DIGEST=<sha256> \
PROJECT_CONFIG_DIGEST=<sha256> \
PROJECT_SOURCE_DIGEST=<sha256> \
RESOLVED_INPUTS_DIGEST=<sha256> \
RUNTIME_DEPENDENCIES_DIGEST=<sha256> \
CODE_REQUIREMENTS_DIGEST=<sha256> \
RESOLVED_CONFIG_DIGEST=<sha256> \
IMAGE_REPOSITORY=registry.lan/carbonteq/posttrain-job \
IMAGE_TAG=<publication-key> \
docker buildx bake \
  --file containers/posttrain-job/docker-bake.hcl \
  posttrain-job
```

The publication target requests zstd-compressed OCI media types, maximal
BuildKit provenance, and an SBOM. The non-publishing `posttrain-job-smoke`
target deserializes the package manifest and checks the installed worker
entrypoint.

Run the repository and staged-context validator before invoking BuildKit:

```bash
uv run python containers/posttrain-job/validate.py \
  --context /absolute/path/to/staged-context
```

The image entrypoint remains `posttrain-runtime`; its default command points at
`/opt/posttrain/job/package.json`. The runtime package must complete the
planned migration from the legacy run-bearing manifest before this image is
eligible for a real submission.
