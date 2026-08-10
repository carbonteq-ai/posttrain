# Posttrain release and consumption

This guide describes how a project team consumes a published Posttrain release
and how to verify that the installed framework, job image, and observation
services are the same release family. It is the README embedded in the release
wheelhouse, so the commands below work when this file is read outside the
source checkout.

## Install an accepted release

Use the attached wheelhouse when installing on a remote or offline machine. It
contains the exact distributions qualified by the release workflow and the
constraints that pin maintained fork dependencies.

```bash
gh release download <release-tag> \
  --repo carbonteq-ai/posttrain \
  --pattern 'posttrain-wheelhouse-*.tar.gz'
mkdir posttrain-wheelhouse
tar -xzf posttrain-wheelhouse-*.tar.gz -C posttrain-wheelhouse

uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python \
  --constraint ./posttrain-wheelhouse/github-constraints.txt \
  --find-links ./posttrain-wheelhouse \
  posttrain posttrain-observatory
```

On the CarbonTeq network, the stable internal index is the equivalent
installation source. Use the matching `github-constraints.txt` from the
release; do not mix constraints from another release.

```bash
uv pip install --system-certs \
  --index-url https://pypi.lan/carbonteq/stable/+simple/ \
  --constraint github-constraints.txt \
  'posttrain[observatory,trackio,trl]'
```

Add the `dstack` extra only for projects that submit remote jobs. Install
additional capability extras (`verifiers`, `vllm`, or W&B) only when the
project's work package declares them.

## Verify the consumer environment

Run these checks from the consuming project, not from the framework checkout:

```bash
uv run posttrain version
uv run posttrain doctor
uv run posttrain catalog validate
uv run posttrain work-package validate <name>.yaml
```

`doctor` checks the Python/runtime and provider configuration. It does not
replace a job qualification: a release is usable only after the project's
packed image and execution target have been validated on the intended host.

## Project and job boundary

A consuming project commits its project manifest, catalog overlays,
work-package definitions, `pyproject.toml`, and `uv.lock`. It does not copy the
framework's base catalog or store credentials in source. Pack jobs from that
project with the release-selected wheelhouse and immutable runtime image
digests. A mutable `latest` image, an unpinned fork, or a workspace checkout is
not release evidence.

The normal lifecycle is:

1. validate the project and its catalog/work package;
2. pack the job and record the image and dependency lock digests;
3. run the job on the selected local or remote execution target;
4. read metrics, traces, and artifacts through the configured Trackio and
   Observatory endpoints;
5. retain the run receipt and recovery checkpoints with the project evidence.

Trackio is the tracking/artifact provider; Observatory is the read-only,
job-aware evidence surface. Their service versions and endpoint configuration
must be checked independently of the Python package version.

## Upgrade and recovery

Upgrade a project by changing the framework version and its constraints as one
reviewed change, then regenerate and commit `uv.lock`:

```bash
uv lock --upgrade-package posttrain
uv sync --locked --python 3.13
uv run posttrain doctor
```

Re-pack jobs after an upgrade. Existing runs remain readable, but a resumed
training job must use a compatible checkpoint, optimizer state, tokenizer, and
job configuration. For LoRA jobs, retain and resume from the adapter weights
and trainer state; do not substitute a full base-model checkpoint for the
adapter artifact. If no compatible checkpoint receipt exists, start a new run
and record the reason.

## Release evidence

The wheelhouse `SHA256SUMS`, Python release receipt, dependency constraints, and
the published runtime image digests are the release evidence. Keep them with
the project qualification record. The release workflow promotes the exact
candidate bytes after the candidate gates pass; rebuilding distributions on a
consumer machine is not equivalent evidence.

For the complete installation and remote-server procedures, see
`docs/install.md`, `docs/getting-started.md`, and
`docs/remote-gpu-qualification.md` in the source repository.
