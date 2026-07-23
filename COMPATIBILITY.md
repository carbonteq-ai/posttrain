# Compatibility policy

Posttrain is currently a pre-1.0 framework for CarbonTeq-managed projects. The
table below describes tested support, not every combination that may happen to
install.

| Surface | Supported baseline |
| --- | --- |
| Python | CPython 3.12 |
| Operating system | Linux |
| Package transport | Versioned GitHub Release wheelhouse |
| Project layout | `.posttrain/project.toml` schema version 1 |
| Local tracking | CarbonTeq Trackio fork pinned by the release constraints |
| Managed tracking | W&B through `posttrain-tracking-wandb` |
| Evaluation | Verifiers at the immutable revision selected by the package |
| Training | CarbonTeq TRL fork; qualified veRL paths remain explicitly gated |
| Inference | vLLM on a compatible NVIDIA CUDA host |
| GPU evidence | Only the targets and recipes named in release qualification |

macOS and Windows may support project authoring and CPU-only contract tests, but
they are not qualified execution platforms. Accelerator compatibility depends
on the selected PyTorch, CUDA, driver, vLLM, model, and execution-target
combination; installation alone is not qualification.

## Versioning

First-party distributions use one coordinated framework version. Before 1.0:

- A release-candidate suffix marks an artifact that has not passed every stable
  release gate.
- Patch releases contain compatible fixes and documentation corrections.
- Minor releases may change public APIs, catalog schemas, or project
  configuration with documented migration instructions.
- Removed behavior receives a documented compatibility window unless retaining
  it would create a security or correctness risk.

Maintained forks and environment packages may use their own upstream-derived
versions. Every framework release records their immutable versions or commits
in its constraints and release evidence.

## Backend equivalence

Tracking providers implement the same logical run, metric, event, trace,
artifact, and outcome contracts; their physical storage is not expected to be
identical. Trainer and inference adapters preserve the selected job meaning,
inputs, and evidence requirements, but numerical identity across backends or
hardware is not guaranteed.

## Support boundary

The supported project surface is the documented public Python API, CLI, project
manifest, catalog schema, work-package schema, and release artifacts. Private
backend modules, generated Observatory schemas, local state layout, scratch
paths, and provider storage internals are not compatibility interfaces.

See [UPGRADING.md](./UPGRADING.md) before changing a framework release.
