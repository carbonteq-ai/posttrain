# Changelog

All notable changes to Posttrain are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a coordinated
version across first-party distributions.

## 0.3.11 - Unreleased

### Fixed

- Running GRPO and OLMo 3 jobs now publish the retained reward spread and
  zero-variance-group signals at each completed optimizer step.  Observatory
  can therefore distinguish a genuinely missing signal from a job that is
  still collecting rollouts; terminal trace replay remains the recovery path
  for interrupted jobs.

## 0.3.10 - 2026-08-12

This release makes OLMo 3 active sampling auditable as a distinct rollout
population, including older runs that emitted the native metrics before their
resolved selection snapshots contained the algorithm field.

### Added

- Observatory now shows active-sampling generation rounds, retained fraction,
  and the reserved, generated, retained, and unused candidate-row populations
  without mixing those counts with rollout outcome totals.
- OLMo 3 runs owe explicit active-sampling evidence, so a missing retention or
  candidate-window metric prevents the run from presenting as fully evidenced.

### Fixed

- Run snapshots retain the resolved GRPO algorithm, prompt-group shape,
  sampling mode, clipping, advantage scaling, and bounded dynamic/active
  sampling settings needed to interpret a training run independently.
- Per-candidate TRL rollout population metrics now carry their candidate scope
  and ordinal, so readers can distinguish refill waves from an optimizer step.

## 0.3.9 - Unreleased

This release prevents a veRL training selection from being packaged against a
different backend runtime than the one it declares.

### Fixed

- veRL job packing now requires the selected source repository, full commit,
  and dependency-lock digest to match the digest-pinned runtime image.
- Runtime-image verification checks the veRL provenance labels in the registry,
  not just the shared framework dependency lock.
- The immutable two-step veRL capsule is retained in Lab's qualification
  inventory as an explicit experimental candidate.

## 0.3.8 - 2026-08-12

This release makes terminal online-RL evidence observable while rollout work
is still running and advances the maintained veRL runtime used by Posttrain.

### Added

- Terminal Verifiers traces are retained before trainability checks, including
  harness errors, inference failures, unscorable attempts, and truncated
  completions.
- The veRL parent process tails native rollout evidence from its isolated
  worker, submits complete records through the configured observer, and retains
  a compact synchronization receipt on success or failure.
- Online-RL population evidence distinguishes requested rollouts, terminal
  traces, and attempts that ended without terminal evidence.

### Changed

- The maintained TRL runtime advances to `1.9.2.post2`, including raw
  sampler/actor parity checks for native-LoRA rollouts; public CI consumes the
  matching hash-verified release wheel.
- The `online-rl-verl-py313` runtime selects CarbonTeq veRL `0.9.0.dev1` at
  immutable source revision `a6fe39c22719ec981ed8544ad8feffd59995cc13`.
- Observatory presents requested-versus-terminal rollout coverage without
  treating missing evidence as a failed or zero-reward rollout.

### Fixed

- Execution failures are no longer classified as length truncation unless the
  native trace records an actual output boundary.
- Failed or unscorable traces no longer fabricate reward spread or enter
  advantage, active-sampling, clipping, entropy, or optimizer calculations.
- TRL failure finalization retains bridge-derived rollout counters instead of
  applying successful-batch metric exclusions.

## 0.3.6 - 2026-08-10

This release tightens local job lifecycle handling, selected-input packaging,
and online-RL evidence delivery.

### Added

- Bounded cache inspection, explanation, lease-aware pruning, compact package
  records, and receipt-backed image reuse for job packing.
- Completion-time rollout observation with asynchronous Trackio delivery while
  preserving crash-recoverable native trace artifacts.
- Job-local cleanup that removes disposable provider workspaces only after
  terminal evidence reconciliation.
- The published Reasoning Gym environment revision with bounded-reasoning
  termination guidance.

### Changed

- Job packing includes only datasets reachable from the selected job's resolved
  seats; unrelated project data is rejected before materialization.
- Live-streamed traces are not replayed during finalization, while aggregate
  trace-derived metrics still cover the complete local spool.
- Local daemon image tags and temporary build material have explicit ownership
  and terminal cleanup boundaries.

## 0.3.5 - 2026-08-09

This release carries checkpoint-scoped model artifacts and the validated
Trackio `0.31.5.post12` runtime through the same immutable Python and OCI
release inputs.

### Added

- Job-scoped checkpoint inspection, verification, selection, and model-artifact
  views for recovery, evaluation, and continuation across training jobs.
- Adapter-only model artifacts for LoRA and QLoRA runs, paired with complete
  recovery state for resumable training.

### Release integrity

- The published runtime lock and OCI manifest are committed and verified
  together, including the exact Trackio post12 wheel hash.

## 0.3.3 - 2026-08-09

This release strengthens online-RL correctness and makes the maintained
Trackio and TRL distributions reproducible inputs to Posttrain jobs.

### Added

- A named `algorithm: olmo3` GRPO recipe backed by TRL's
  `Olmo3GRPOConfig`, including zero-gradient filtering, bounded active refill,
  token-level loss normalization, asymmetric clipping, mean-only advantages,
  zero KL, and truncated importance sampling.
- Portable active-sampling settings and rollout telemetry for accepted,
  rejected, replacement, exhausted, and usable prompt groups.
- DAPO advantage diagnostics covering magnitude, sign balance, group reward
  spread, zero-variance groups, scoreability, truncation, and importance-ratio
  clamping.
- Deterministic runtime-lock materialization for internally published
  dependencies. Release candidates retain the generated lock together with the
  immutable OCI manifest before merge.

### Changed

- DAPO reward scaling is configurable instead of hardcoded. Truncated
  completions can be excluded from group statistics as well as from the loss,
  preventing masked samples from changing another completion's advantage.
- LoRA rollout synchronization and trainer configuration use the maintained
  TRL `1.9.2.post1` distribution rather than an ambient Git checkout.
- Trackio advances to `0.31.5.post11`, including the S3 artifact recovery path
  used by remote jobs and Observatory evidence readers.
- Runtime images consume the same exact Trackio and TRL wheel receipts as the
  Python workspace, while GitHub-only consumers retain hash-verified public
  release fallbacks.

### Release safety

- Pull-request validation distinguishes authored dependency changes from
  candidate-generated OCI evidence. The stale-image guard remains strict after
  materialization and for every merged/default-branch build.
- Candidate artifacts include both `workspace.lock.txt` and `published.toml`,
  so final distributions cannot silently package an earlier runtime image
  graph.

## 0.3.2 - 2026-08-05

This release adds the Gemma 4 dense support matrix and qualifies the TRL
paired-assistant MTP path on the RTX PRO dstack target.

### Added

- Immutable Gemma 4 E2B, E4B, 12B Unified, and 31B model variants with exact
  checkpoint provenance and shared family-level rendering.
- TRL MTP assistant validation and snapshot materialization for the Gemma 4
  12B GRPO path, including speculative acceptance and KV-cache evidence.
- Declarative Gemma serving, SFT, and TRL qualification work packages with
  tracked dstack evidence.

### Qualification

- E2B, E4B, and 31B serving smokes returned non-empty text on the RTX PRO
  target.
- The 12B TRL run completed two non-truncated rollouts and one optimizer step
  with reward 1, MTP acceptance 0.937888, and KV-cache metrics.
- The protected LAN release transaction published the final wheelhouse and
  completed the packed dstack canary before creating tag `v0.3.2`.

## 0.3.1 - 2026-08-05

This release makes evaluation meaning part of immutable run evidence, expands
the maintained Verifiers environment library, and completes the reviewable
cross-plane purge workflow introduced after 0.3.0.

### Added

- Explicit `run purge` and `project purge` planning across provider, OCI,
  tracking, and local state, with dependency closure, digest-bound previews,
  confirmation gates, resumable receipts, and retained verification evidence.
- Independently packaged GSM8K, AutomationBench, MMLU-Pro, IFEval, Reasoning
  Gym, and Math Python Verifiers environments with immutable source/data
  revisions, reproducible subset selection, and Lab qualification packages.
- Versioned evaluation contracts that snapshot the selected population,
  success predicate, reward and metric namespaces, task facets, and structured
  compound breakdowns such as problem type by difficulty.
- Evaluation-first Observatory Overview, Compare eligibility, performance
  distributions, schema-driven reward/verifier columns, pass/fail outcomes,
  and chat-style tool-aware trace inspection.
- Reproducible Python dataset materialization and package-owned serving
  workload definitions carried forward from the post-0.3.0 release branch.

### Changed

- Evaluation images install Verifiers, environment wheels, and runtime
  dependencies during image construction. Job startup only resolves the
  snapshotted configuration and executes the worker; it performs no package
  installation or upgrade.
- Tool-using environments declare portable inference capabilities. The Qwen
  renderer contract selects the compatible vLLM reasoning/tool parser while
  subprocess or MCP transport remains environment-owned.
- Project run listings exclude foreign admissions and successfully purged runs
  by default; `--include-purged` exposes labeled retained history for audit.
- Trackio advances to the maintained post8 lifecycle API, and affected runtime
  images resolve exclusively from the CarbonTeq OCI registry.
- Release publication now runs through the protected LAN runner, private
  `pypi.lan`/`registry.lan` channels, immutable candidate receipts, and a
  verified idle RTX PRO dstack canary; GHCR and public PyPI are not part of the
  release path.

### Qualification

- Real Qwen3.5-4B thinking evaluations qualified IFEval, Reasoning Gym, and the
  full 200-task AutomationBench Simple population with native MTP, retained
  Verifiers traces, and Observatory projections. The live Math Python schema-v3
  run additionally demonstrated the frozen 500-task population, configured
  success predicate, compound problem-type-by-difficulty reporting, subprocess
  Python tools, MTP, and concurrency eight. It remains active for terminal
  reconciliation and is not stopped by this release.
- The source validation ladder passes Ruff lint and format, Pyright, all eight
  import contracts, 1,030 Python tests with 18 expected skips, 32 Observatory
  frontend tests, and the production frontend build.

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
  ([getting-started §9](docs/getting-started.md#9-pass-one-jobs-model-into-the-next),
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

- [getting-started](docs/getting-started.md) (formerly consumer-setup) —
  trust, index install, local and dstack providers, doctor, plan/pack/run,
  workers
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

- [getting-started](docs/getting-started.md) (formerly consumer-setup) —
  install from the internal index, trust the CA, run local or dstack jobs
  (executed steps, not aspirational)
- [install](docs/install.md) and
  [release-engineering](docs/release-engineering.md) (formerly
  release-and-consumption) — how releases are installed and gated
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
