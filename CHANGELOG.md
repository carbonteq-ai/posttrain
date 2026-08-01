# Changelog

All notable changes to Posttrain are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a coordinated
version across first-party distributions.

## 0.3.0 - 2026-08-01

This release starts the project-owned developer-experience redesign. Static
job meaning, execution configuration, catalog composition, environment source,
and image publication are now separated so projects can be planned and packed
without silently inheriting the submitting shell or a provider connection.

### Added

- Installable `posttrain-project` and `posttrain-environment` packages with
  public project discovery, provider-free job intents, execution-setting
  provenance, portable environment activation contracts, and project-path
  environment sources.
- Deterministic catalog-family discovery through entry points. Resolved family
  provenance is locked into package identity; duplicate, absent, or undeclared
  providers fail before catalog decoding.
- Local OCI job-image export, selected transitive catalog closure staging,
  project environment scaffolding, and declared dataset builders with
  input-sensitive cache identity.
- Runtime qualification for staged activation resources, taskset loading, and
  frozen JSONL datasets before an actual job image is published.
- Manifest-only release preparation, release-neutral workspace metadata,
  staged static wheel metadata, and a single generated catalog dependency-lock
  table through `posttrain-release`.
- Durable project/control and provider locators, a foreground lifecycle
  controller, joined run views, and safe state migration/cache classification.

### Changed

- Project-root `posttrain.env` is loaded automatically and authoritatively;
  ambient shell variables no longer override project runtime configuration.
- `posttrain job plan` reports provider-free job intent. Publication and
  launch settings are selected by `job pack` and `job run` respectively.
- Read-only `--last` resolution is strictly chronological. Mutating run
  commands require the complete canonical run id.
- Machine defaults can configure local-container DNS without placing machine
  topology in project configuration. Managed inference bindings carry a
  versioned startup budget that is retained in resolved run evidence.

### Release qualification

- An external consumer installed all 24 coordinated 0.3.0 framework wheels.
  The Lab data-preparation gate and managed Qwen 3.5 2B GSM8K evaluation both
  executed from packed immutable images, reconciled provider exit 0 against
  retained Trackio artifacts, and reported complete required telemetry.
- The bounded evaluation synchronized both native Verifiers traces: 2/2
  completed successfully with mean reward 1.0 and no failed or truncated
  rollouts. Its resolved evidence records the 600-second managed-inference
  startup budget used to cover cold model/kernel initialization.

## 0.2.5 - 2026-07-31

This patch hardens high-concurrency native Verifiers GRPO and makes
Observatory discover Trackio projects dynamically.

### Fixed

- Concurrent policy turns are batched and arrivals are drained safely while a
  trainer update holds the lock.
- The TRL backend exposes Liger loss compilation as an explicit, validated
  setting.
- Verifiers rollout groups execute concurrently, with a bounded compatibility
  patch for the current MCP harness dependency.

### Changed

- Observatory discovers available Trackio projects instead of relying on a
  fixed project list.

## 0.2.4 - 2026-07-30

The maintained veRL and vLLM runtime becomes a first-class published job kind.
This release also makes dstack placement durable and visible, and carries the
runtime fixes found while qualifying two-step DAPO, SAMPO, and distillation on
the 24 GB and 96 GB workers.

### Added

- Published `online-rl-verl-py313` runtime with immutable CarbonTeq veRL and
  vLLM fork revisions.
- Persistent dstack capacity waiting plus `posttrain run queue`, requested and
  assigned worker hostnames, and worker-capacity inspection.
- Two-step DAPO, SAMPO, and distillation qualification packages, including
  MTP-only variants retained as explicit selections.
- CUDA toolkit activation and vLLM compatibility checks needed by packaged
  remote jobs.

### Fixed

- veRL distillation aligns dense and jagged teacher log probabilities to the
  exact response tokens and safely ignores fully masked padding microbatches.
- Checkpoint-free terminal model export no longer fails retention after the
  disposable checkpoint root is removed.
- Parallel Verifiers harnesses no longer race while installing container
  prerequisites.
- Colocated Qwen 3.5 rollout uses eager vLLM execution, avoiding the observed
  CUDA-graph illegal-memory-access path.
- Actual-job BuildKit targets no longer race through a mutable named context.

### Qualification

- `verl-distill-shared-pool-retentionfix-20260730` completed two optimizer
  steps on `carbonteq-ai-workstation.lan`, retained 16 native Verifiers traces,
  the trained adapter, summary, and retention manifest, and reconciled dstack
  exit `0` with no missing required artifact roles.
- Baseline DAPO and SAMPO completed two optimizer steps; dstack also proved
  concurrent capacity-based placement across the RTX 4090 and RTX PRO 6000
  workers.

### Documentation

- Produce → pin → rebind how-to for trained model handoff between work
  packages
  ([consumer-setup §9](docs/consumer-setup.md#9-pass-one-jobs-model-into-the-next),
  [developer-experience](docs/developer-experience.md#trained-model-handoff-produce--pin--rebind),
  [tooling/trackio](docs/tooling/trackio/README.md#project-developers-artifact-handoff)).

## 0.2.3 - 2026-07-28

Pin Trackio to `0.31.5.post5` (`703be380…`) so import and distribution
versions match. Do not install `0.31.5.post4` from the index — that wheel is
skewed. Public PyPI Trusted Publishing is still unconfigured, so the Git pin
remains.

### Changed

- Trackio workspace, kind-profile, and constraint pins move to `703be380…`.
- Republished kind images against the refreshed workspace lock digest
  (`bedcf309…`).

## 0.2.2 - 2026-07-28

Machine-scoped local GPU admission, Observatory/Trackio listing performance,
public developer documentation, then a required runtime-image republish so
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

### Documentation

Public developer-facing guides (no private ops required to start):

- [consumer-setup](docs/consumer-setup.md) — trust, index install, local and
  dstack providers, doctor, plan/pack/run, workers
- [developer-experience](docs/developer-experience.md) — project layout,
  catalog overlays, standard jobs, datasets/envs
- [tooling/dstack](docs/tooling/dstack/README.md) — client binding, soft
  affinity, placement vs local admission
- [tooling/trackio](docs/tooling/trackio/README.md) — fork pin and project
  artifact ownership boundary
- [contributing](docs/contributing.md) — framework checkout validation ladder
- [publishing](docs/publishing.md) — cutting a release without drifting images
- Trust as a **machine** property (well-known CA path), not project config;
  service ownership (`ai-infra` operates `.lan` services; this repo is the
  framework)

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
- A reproducible remote GPU release-gate workflow.
- Durable execution lifecycle, deterministic job packing, provider adapters,
  and job capsule CLI paths needed for pack/run without a checkout.

### Documentation

Public project-developer surfaces introduced with the consumer path:

- [consumer-setup](docs/consumer-setup.md) — install from the internal index,
  trust the CA, run local or dstack jobs (executed steps, not aspirational)
- [release-and-consumption](docs/release-and-consumption.md) — how releases
  are consumed and gated
- [remote-gpu-qualification](docs/remote-gpu-qualification.md) — remote GPU
  release-gate workflow
- [UPGRADING](UPGRADING.md), [COMPATIBILITY](COMPATIBILITY.md),
  [SECURITY](SECURITY.md), Apache-2.0 [LICENSE](LICENSE)
- Frozen product baseline under [docs/post-training/](docs/post-training/README.md)
  (workflow → primitives → work/evidence → framework → APIs → observation)

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
