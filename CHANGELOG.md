# Changelog

All notable changes to Posttrain are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a coordinated
version across first-party distributions.

## Unreleased

### Added

- Developer how-to for handing a trained model to a later work package:
  publish to Trackio, pin `artifact.kind: trackio` in a project overlay, bind
  that catalog id on the next package
  ([consumer-setup §9](docs/consumer-setup.md#9-pass-one-jobs-model-into-the-next)).

## 0.2.2 - 2026-07-28

Machine-scoped local GPU admission, Observatory/Trackio listing performance,
consumer and ops documentation, then a required runtime-image republish so
LAN digests match the Trackio workspace lock.

### Added

- Shared local GPU admission across projects on one machine (`posttrain
  workers`); dstack placement no longer takes a host lock inside posttrain.
- Soft affinity via catalog target `placement.instances: [{hostname: …}]`
  (optional; capacity-only placement remains the default).
- CLI DX: clearer job resolve / run-id / follow paths and reduced setup
  friction for `job plan|pack|run` and run inspection.
- Trackio `0.31.5.post4` (`dc55020d…`) with bulk `run_configs` /
  `run_lifecycles` so Observatory can list runs without an N+1 history fetch.
- Consumer and ops docs: trust as a machine property, dstack affinity,
  contributing, publishing, and service ownership (`ai-infra` vs this repo).

### Changed

- Observatory qualification uses available runs rather than four fixed named
  runs; Observatory image builds on the interpreter the app requires.
- `published.toml` pins `registry.lan/carbonteq` kind images to lock digest
  `c93d274e…`. `0.2.1` on the index still carried prior GHCR digests / lock
  hash, which the manifest loader correctly refused until this republish.
- Publish tooling streams Buildx progress and defaults to faster push
  settings.

### Note

Install `posttrain==0.2.2` from the internal index (or the GitHub wheelhouse
when attached). Runtime images must match this release’s lock digest;
`posttrain doctor` / `runtime images verify` report drift.

## 0.2.1 - 2026-07-28

The framework was qualified from a library consumer's seat for the first
time: installed from an index, with no checkout on the machine. Eleven
defects stood between that developer and a finished job, none of which the
test suite could see, because it runs from a checkout where the source tree
exists, every package is one version, and the build definitions are inside
the project.

First stable line after `v0.1.0-rc.2`. Requires **Python 3.13**.

### Added

- An internal package index as the supported distribution channel, with the
  maintained forks constrained explicitly because uv does not resolve a
  transitive direct URL implicitly.
- Release-pinned portable runtime images (base + job-kind variants) shipped
  as package data, with registry resolution and drift refusal.
- `posttrain job diff`, explaining why two packed job packages differ.
- A `trust` readiness check reporting which certificate authority reaches
  jobs, and warning when it is absent from the machine's own store.
- `docs/consumer-setup.md`, written from steps that were executed rather
  than imagined.
- Primary-CLI work-package execution through an explicit project host.
- Apache-2.0 licensing, private security reporting, compatibility policy, and
  upgrade guidance.
- A reproducible remote GPU release-gate workflow.
- Durable execution lifecycle, deterministic job packing, provider adapters,
  and job capsule CLI paths needed for pack/run without a checkout.

### Changed

- Framework packages pin each other exactly. Declared by bare name, they
  allowed `posttrain` to be upgraded while every sibling stayed behind, which
  was individually satisfiable, matched no release, and was packed into job
  images as though coherent.
- A job image obtains framework code as built distributions when there is no
  checkout, rather than requiring twelve source directories to copy.
- Additional certificate authorities are merged with those the job image
  already trusts, and resolved from `/etc/posttrain/trust/internal-ca.pem`
  when nothing is configured. Execution providers no longer set
  `SSL_CERT_FILE`, which replaced the trust store rather than extending it.
- Runtime floor moved to Python 3.13; images and scaffolding follow.

### Fixed

- A run that died before opening a tracking run held its machine's admission
  placement permanently, and cancel, cleanup, and reconcile each refused it
  for a different reason.
- Tracking evidence was written to the configured server but looked for in a
  local one, so a succeeded run reconciled as pending with no artifacts.
- Writing `execution.toml` for the local provider's hostname discarded
  `POSTTRAIN_REGISTRY`.
- A failed provider submission reported only that its outcome was unresolved
  and to retry, naming nothing to act on.
- `release/github-constraints.txt` omitted `trl` and `verifiers`, so the
  documented install could not resolve.

### Note

Versions 0.1.1 through 0.1.13 were development builds published while
qualifying this work, because packing downloads framework wheels from the
index and a fix could not be tested until it was published. They are not
supported releases. Development builds now stage on a separate index.

## 0.1.0-rc.2 - 2026-07-24

Makes the framework usable from a normal project while adding explicit DAPO
and multi-turn SAMPO training contracts.

### Added

- `posttrain init` creates and installs a complete project with `.posttrain/`
  configuration, catalog overlays, work packages, and a project-local
  environment.
- Primary CLI owns dataset/environment materialization, work-package
  validation and execution, run inspection, and `posttrain observatory up`.
- Standard SFT, DPO, GRPO, DAPO, SAMPO, distillation, evaluation, serving, and
  model-transform jobs from `posttrain.jobs` without requiring `posttrain-lab`
  on the common path.
- DAPO as a first-class GRPO algorithm selection (TRL and veRL) with asymmetric
  clipping, token-level loss, and related bounded-sampling contracts.
- SAMPO as a separate multi-turn tool-agent operation with hierarchical
  episode/turn advantages.
- Developer environment profiles, remote GPU qualification guidance, and
  upgrade/compatibility documentation.

### Note

Real multi-turn SAMPO GPU qualification and larger GPU release gates remained
open; this prerelease did not claim production-qualified training quality for
those paths.

## 0.1.0-rc.1 - 2026-07-23

### Added

- Portable `.posttrain` project discovery, initialization, catalog overlays,
  work packages, and ignored machine-local state.
- The `posttrain` CLI for diagnostics, project inspection, catalog inspection,
  and composition validation.
- Versioned framework catalog resources and reusable data, train, evaluation,
  serving, tracking, and work-composition packages.
- Trackio and W&B tracking adapters with provider-neutral evidence contracts.
- Observatory Python, HTTP, MCP, report, and frontend surfaces.
- Installed-wheel consumer acceptance with real local Trackio persistence and
  Observatory readback.
- GitHub Release wheelhouses with immutable fork constraints and SHA-256
  checksums.

[Unreleased]: https://github.com/carbonteq-ai/posttrain/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/carbonteq-ai/posttrain/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/carbonteq-ai/posttrain/compare/v0.1.0-rc.2...v0.2.1
[0.1.0-rc.2]: https://github.com/carbonteq-ai/posttrain/compare/v0.1.0-rc.1...v0.1.0-rc.2
[0.1.0-rc.1]: https://github.com/carbonteq-ai/posttrain/releases/tag/v0.1.0-rc.1
