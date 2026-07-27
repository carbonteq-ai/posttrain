# Framework runtime image hierarchy

These definitions are the framework-owned first two levels of the job-image
hierarchy:

1. `posttrain-base` supplies Python 3.12, the system certificate store, and the
   CUDA-enabled PyTorch version locked by the workspace.
2. `posttrain-kind-*` adds stable backend dependencies for one runtime
   variant. Supervised training, evaluation, serving, and model transformation
   currently each have one variant. Online RL distinguishes
   `online-rl-trl-py312` from `online-rl-verl-py313`; a logical job kind is not
   a Python compatibility boundary. Kind images contain no framework or
   project source.
3. The job packer derives from one immutable kind-image digest and adds the
   exact framework worker, project code, environment wheels, dataset snapshot,
   resolved selections, and job manifest.

Concrete Verifiers environments are deliberately absent from levels 1 and 2.
The kind layer contains only Verifiers core. Environment repositories may be
different for each job, and multiple pinned environments may be included in one
job image without changing the shared online-RL or evaluation image.

The published TRL online-RL variant runs in the universal Python 3.12 control
environment. veRL is different: the framework control process must remain in
`/opt/posttrain/venv` on Python 3.12, while the backend worker runs from an
isolated `/opt/posttrain-verl` Python 3.13 environment. The candidate veRL
definition and release gate live under `verl-py313/`. It is intentionally not
present in the Bake publication graph while the CarbonTeq veRL fork is dirty
and unpublished. The current research lock is also ineligible because it uses
an editable checkout and includes concrete GSM8K and AutomationBench packages.

The large dependency layers are shared across actual jobs. A framework or
project source change therefore invalidates only the actual-job code layer,
while a kind lock change deliberately invalidates the shared dependency layer.
The generated constraint files retain the versions and hashes resolved by
`uv.lock`; `transform.lock.txt` comes from the separately maintained
`tools/quantization/uv.lock`. `build-tools.lock.txt` is a separate
hash-locked closure for Hatchling and its build-time dependencies. It is
installed in every kind image because actual-job source installation disables
build isolation and must never fetch an implicit build backend.

The publication targets emit zstd-compressed OCI media types and request
maximal BuildKit provenance plus an SBOM. A release invocation must pass an
immutable `POSTTRAIN_BASE_IMAGE`, the source revision, the source-lock digest,
and release metadata:

```bash
docker buildx bake \
  --file containers/posttrain-job-kinds/docker-bake.hcl \
  --set '*.args.POSTTRAIN_BASE_IMAGE=registry.lan/carbonteq/posttrain-base@sha256:<digest>' \
  --set '*.args.SOURCE_REVISION=<git-commit>' \
  --set '*.args.LOCK_DIGEST=<sha256-of-lock-inputs>' \
  --set '*.args.VERSION=<release-id>'
```

Build the non-publishing qualification stages with:

```bash
POSTTRAIN_BASE_IMAGE=registry.lan/carbonteq/posttrain-base@sha256:<digest> \
  docker buildx bake \
  --file containers/posttrain-job-kinds/docker-bake.hcl \
  smoke
```

`validate.py` checks the target graph, lock-derived pins, attestations,
compression policy, and the prohibition on concrete environment packages
without building or pushing an image. It also checks that the blocked veRL
candidate cannot be mistaken for a published target. A veRL release
additionally requires:

```bash
python containers/posttrain-job-kinds/verl-py313/release_gate.py \
  --release \
  --source-checkout /path/to/clean/carbonteq-verl \
  --verify-remote
```
