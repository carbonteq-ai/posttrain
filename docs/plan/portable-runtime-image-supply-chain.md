# Make the runtime image supply chain portable and release-pinned

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

This document must be maintained in accordance with `docs/templates/PLAN.md`.

## Purpose / Big Picture

Today the `posttrain` framework cannot be used from an installed wheel. A
developer who runs `pip install posttrain` and then `posttrain job run ...`
fails immediately, because the code that packs a job walks the filesystem
looking for a `containers/` directory that exists only in the framework's own
Git checkout. Everything about the container image supply chain — the
Dockerfiles, the dependency locks, the dependency profiles, the BuildKit bake
definitions, and the resolved image digests — lives either in that checkout or
in a machine-local file that is deliberately excluded from version control.

After this work, three things are true that are not true today.

First, an installed framework carries its own runtime image definitions and
dependency locks as ordinary package data, so a consumer who never clones the
framework repository can still pack and run a job.

Second, the framework publishes its universal base image and its per-variant
job-kind images to a public registry once per release, and the wheel carries
the exact digests it was built against. A consumer pulls those images by
digest. They do not build them. Publishing the project-specific actual-job
delta still requires either the supported local BuildKit path or the optional
developer job-build service described below.

Third, when a consumer's configured job-kind image does not match what the
installed framework expects, `posttrain doctor` says so loudly and
`posttrain job run` refuses to submit, naming the expected and the found
identity.

Fourth, a site can run a dedicated developer job-build service beside its OCI
registry. The client materializes the exact same content-addressed job package
as the local path, asks which context blobs are missing, uploads only those
bounded blobs, and receives a verified immutable actual-job image. The service
is not a general remote Docker daemon and is not the protected framework
release builder. Local publication remains a complete fallback, so service
availability never changes job meaning or makes the framework unusable.

The concrete failure this prevents has already happened. The on-policy
distillation release gate failed repeatedly with `grad_norm=nan`. The root
cause was a bug in the pinned TRL fork. Correcting the framework's TRL pin was
not sufficient, because `containers/posttrain-job-kinds/profiles/supervised.txt`
and `containers/posttrain-job-kinds/locks/workspace.lock.txt` independently
carried the old TRL commit, and the already-published
`posttrain-kind-online-rl-trl-py312` image had that old commit baked into its
layers. Nothing in the framework noticed. The stale image was found only by a
human manually inspecting the published image's OCI labels. Any GPU
qualification run against that image was silently invalid.

You can see the fix working by deleting your machine-local execution
configuration, setting one environment variable, and running a job. It works,
and it uses exactly the images the installed framework was released with.

## Progress

- [x] Milestone 1 — Amend the frozen product baseline (`05-apis.md`,
  `04-framework.md`) for the new `runtime` command noun, the
  `POSTTRAIN_REGISTRY` environment variable, and the new
  `posttrain-runtime-images` package row. Landed: `05-apis.md` gains four
  `runtime images` lines, a `--build-missing` flag on `job run`, and two
  paragraphs defining the `runtime` noun and `POSTTRAIN_REGISTRY` as a
  location rather than a credential; `04-framework.md` gains the
  `packages/runtime-images` row and extends rule 13 with per-release digest
  pinning, pull-by-default, and the fail-on-drift obligation.
- [x] Milestone 2 — Create the `posttrain-runtime-images` distribution and move
  the three container definition trees into it as package data, preserving the
  current uncommitted worktree edits. Landed: all 46 files moved with `git mv`
  (renames detected, the five modified files preserved as `RM`, untracked veRL
  files carried along); both lock digests byte-identical after the move, so no
  drift was introduced. `validate.py` re-rooted without weakening any assertion.
  The legacy `posttrain-job-runtime` tree was removed. The veRL gate test moved
  out of the shipped tree into `packages/runtime-images/tests/` and is now in
  `testpaths`. `_bake_file` resolves from package data. Verified against a built
  wheel: definitions present, tests absent, and the workspace lock inside the
  wheel hashes to `e8a833bf24f5…`, the published image identity. Full ladder
  green (684 passed, 17 skipped).
- [x] Milestone 3 — Add the release image manifest (variant to repository,
  digest, and lock digest) as package data. Landed: `published.toml` records
  the base plus all five released job-kind images, with `constraint_lock` and
  `provided_packages` migrated off the machine-local file;
  `posttrain.runtime_images.manifest` parses it into frozen values and
  integrity-checks every recorded lock digest against the shipped lock bytes at
  load time. Verified end-to-end: appending one byte to the shipped workspace
  lock makes `load_manifest()` fail with the exact republish instruction, and
  the lock restores to `e8a833bf24f5…`. Ladder green (700 passed, 17 skipped).
  Regenerating the manifest is an owner operation and moved to Milestone 5b.
- [x] Milestone 4 — Resolve the registry from `POSTTRAIN_REGISTRY`, make
  `[registry].kind_images` and `[registry].constraint_profiles` optional and
  derived from the installed manifest, and keep the existing file working.
  Landed: `POSTTRAIN_REGISTRY` is read in exactly one place, declared entries
  override per variant rather than replacing the whole set, and
  `constraint_profiles` derive from the published image including
  `provided_packages`. Added `[registry].mirror_prefix` so a site can relocate
  framework images without the project registry silently becoming a mirror.
  Verified end to end: with `execution.toml` moved aside entirely and only
  `POSTTRAIN_REGISTRY` set, `job plan` resolves
  `posttrain-base@sha256:66c71936…` and
  `posttrain-kind-supervised@sha256:5c247c81…` — the same digests the machine
  file used to carry by hand — and `doctor` passes. Ladder green (709 passed).
- [x] Milestone 5a — Add the consumer `posttrain runtime images` command group:
  `list`, `verify`, `mirror`, and `build`. Landed: `list` is a pure local read
  and works offline; `verify` compares the published `lock-digest` label per
  variant; `mirror` copies by digest and re-reads the destination, failing if
  identity changed; `build` validates definitions without `--push` and reports
  any image not matching the pinned digest as UNVERIFIED. `LOCK_DIGEST` is now
  computed in Python from the shipped lock, deleting the hand-typed `--set`.
  Added `RuntimeImageInspector` plus an `ImageInspector` protocol so identity
  checks are testable without a registry, and
  `BuildKitRuntimeBuilder.check()` for definition-only validation.
- [x] Milestone 5b — Add the owner-only `apps/release` distribution. Landed:
  `posttrain-release images publish` builds every variant, pushes to the
  release registry, reads each digest back from the registry rather than
  predicting it, and regenerates `published.toml`. Manifest rendering is a pure
  function so the exact shipped bytes are testable without a registry. A test
  asserts `posttrain` does not depend on this package, in required or optional
  dependencies, so release authority cannot reach a consumer environment.
- [x] Milestone 6 — Introduce a real `doctor` check abstraction and add the
  registry-configured and image-drift checks. Landed: a frozen `Check` with
  `ok`/`warn`/`error`, exiting non-zero only on `error` so a missing registry is
  reportable without breaking a pipeline. `registry` warns when unset;
  `runtime_images` errors when a configured job-kind image is not the digest
  this release pins. Both are local comparisons and make no network request;
  label verification stays in `runtime images verify`.
- [x] Milestone 7 — Make `posttrain job run` verify and fail closed, with an
  opt-in `--build-missing`. Landed: `_require_verified_kind_image` runs before
  packing and before any provider object exists, on both `job pack` and
  `job run`. A missing or drifted image aborts with the variant, both lock
  digests, and the remedy. An unreachable registry aborts even with
  `--build-missing`, so a network failure can never be silently converted into
  a local rebuild.
- [ ] (2026-08-14) Milestone 8 — Preserve parent blobs during actual-job
  publication and prove the pushed delta. The exporter and both shipped Bake
  definitions now use `force-compression=false`. Publication compares the
  parent and child OCI manifests and rejects rewritten or reordered ancestry;
  it reports inherited layers, new layers, and job-only compressed bytes.
  A machine that lost its local receipt now recovers a matching deterministic
  remote tag after verifying package labels and parent layers, without a build.
  A live registry qualification remains open. The historical kind digest in
  the committed manifest no longer resolves at `registry.lan`, so this proof
  must use the next materialized candidate rather than treating an old tag as
  evidence.
- [ ] (2026-08-14) Milestone 9 — Make the complete lifecycle selectively
  incremental. Profile files are copied into only their owning Docker stage,
  and release source identities are now per variant, so a supervised-profile
  edit does not invalidate serve, eval, transform, or veRL. Registry mirroring
  skips a destination tag that already names the expected digest. Focused
  validation is green (`218 passed, 1 skipped`, with two unrelated fork-state
  assertions deliberately deselected). The base now has a 29-package,
  hash-locked closure and every non-transform kind has its own generated
  profile closure. `posttrain-release lock-runtime-dependencies` first exports
  `workspace.lock.txt` directly from the exact root `uv.lock`, materializes
  internal wheel receipts, then projects each profile's Linux/Python 3.13
  transitive closure from that already-resolved graph. It never invokes a
  second dependency solver, so a generated kind lock cannot admit a newer
  artifact or a transitive package absent from the release resolution.
  Transform remains governed by `tools/quantization/uv.lock`. Focused static
  validation, package-data inspection, closure containment, and repeatable
  generation are green.
  `posttrain-release images plan` now emits a pure, topologically ordered
  desired-versus-observed decision for base and every kind:
  `reuse-remote`, digest-preserving `copy`, `build`, or explicit `blocked`.
  Publication executes that same plan, so a base change is an observable
  kind-image fan-out instead of an accidental cascade. The first read-only
  registry probe correctly found the pre-migration manifest absent and planned
  a new base plus its dependent kinds; it made no registry mutation. The
  Distribution reader now walks OCI indexes/manifests and probes the
  destination repository's config/layer blobs, so a cross-registry copy plan
  reports the exact missing-blob bytes when the registry permits those HEAD
  checks. A read-only live plan against `registry.lan/carbonteq` found no
  compatible candidate base image, so it correctly selected `build` for the
  base and all six kinds: this is the intentional parent-digest fan-out, not a
  cache miss or a fork-release failure. Remaining: materialize that candidate,
  then run first-install/partial-registry live scenarios against it.
- [ ] (2026-08-14) Milestone 10 — Add the optional developer job-build
  service. The architecture and acceptance contract are now specified in this
  plan: a new app owns the authenticated HTTP/queue boundary; the existing
  `JobImagePublisher` port remains the application seam; a new remote adapter
  performs content-addressed missing-blob upload; and the existing BuildKit
  publisher remains the only OCI build implementation. The normal publisher is
  now a machine-configuration setting with an
  explicit per-command `--builder local|remote` override, and the exact
  manifest-first/missing-blob context transfer is specified. The frozen
  framework/API baseline amendment is landed. The first implementation slice
  now provides provider-neutral sealed-context/transfer values and a private
  mode-0700 filesystem content store with per-publication records, digest and
  size verification, missing-blob planning, and sealing. The private v1 HTTP
  admission surface now authenticates opaque bearer tokens against protected
  token hashes, derives the only permitted project repository, and implements
  capabilities, plan, bounded blob upload, seal, status, and cancellation.
  A durable FIFO claim reconstructs only the fixed layout plus verified
  declared blobs, then invokes the existing `JobImagePublisher` port and
  persists a safe image result. The `posttrain-execution-job-builder` adapter
  implements manifest-first HTTP publication and local protected receipts.
  The CLI now resolves the machine-owned builder setting, defaults to local,
  supports the explicit `--builder local|remote` override on plan/pack/run,
  and forces `--local` exports through BuildKit even on a remote-configured
  machine. `job plan --builder` is metadata-only: it reports the effective
  developer builder, its selection source, and safe endpoint without reading
  credentials, creating context, or changing package identity. The mode is
  intentionally only for developer actual-job images: it is not, and cannot
  become, a fork-release runner. Production service
  composition/deployment and the
  two-builder reproducibility prototype and Milestones 8/9 live registry gates
  remain hard prerequisites to enabling a networked builder. The ai-infra
  inventory also confirms that `ai-release` remains a protected framework
  release surface; developer-builder deployment requires a separately scoped
  host or VM and must not be placed on that runner.
- [x] (2026-08-15) Consumer readiness check — Ambient Agent now pins
  `posttrain==0.3.17` and `posttrain-train==0.3.17`; its existing
  SFT-initialized OLMo 3 one-step GRPO canary passes detached work-package
  validation and `job plan --builder local`. `job plan --builder remote`
  correctly fails before any transfer because no machine-local
  `[services.job_builder]` binding exists. This is not a client defect: the
  current source has the admission API and worker library but no production
  composition, deployment role, endpoint, or `job-builder.lan` DNS record.
  The next Milestone 10 slice must establish that separate service first;
  configuring a speculative endpoint or reusing `ai-release` is prohibited.

## Surprises & Discoveries

- Observation: the published 0.3.17 client is ready, but the network service
  is not deployable from the current infrastructure checkout.
  Evidence: `/home/hammad/projects/rl/apps/job-builder/` contains
  `http.py`, `store.py`, and `worker.py`, but no production composition root;
  `/home/hammad/projects/ai-infra` has no job-builder Ansible role, playbook,
  inventory group, endpoint, or DNS record. On 2026-08-15,
  `posttrain job plan ... --builder remote` returned
  `--builder remote requires machine [services.job_builder] remote configuration`.
  `ai-release` remains reserved for protected framework releases.

- Observation: the actual-job Docker stages were thin, but the registry output
  was not. A retained eval job had 35 layers and 4,205,518,162 compressed bytes,
  while its job-specific local layers were about 55 MB uncompressed. The
  publisher and job Bake file set `force-compression=true`; BuildKit defines
  that option as applying the selected compression to every layer, including
  already-existing layers. Two jobs sharing one kind reused the rewritten
  ancestry after the first upload, but the first job for each new parent paid
  the entire roughly four-gigabyte transfer.
- Observation: the current registry contains actual-job manifests whose
  recorded kind-image manifest has already been garbage-collected, and the
  checked-in `0.3.16rc8` published manifest also references image digests that
  no longer resolve. Actual jobs remain runnable only because their manifests
  retain a complete rewritten layer set. Parent retention must be repaired and
  verified before the next candidate; forced recompression is not a valid
  retention mechanism.
- Observation: the first live desired-versus-observed plan for the current
  candidate selected a base build and six dependent kind builds, with zero
  known transferable bytes and unknown build-output bytes.
  Evidence: `uv run posttrain-release images plan --repository-root
  /home/hammad/projects/rl --registry registry.lan/carbonteq --receipt-root
  <empty-temporary-directory>` returned `blocked: false`, `base: build` with
  `no compatible published image is available`, and every kind with `parent
  base requires a new immutable digest`.
  Interpretation: the planner is distinguishing identity change from local
  cache state correctly. The next candidate must materialize its base before a
  selective kind-reuse or actual-job parent-prefix claim can be qualified.
- Observation: a direct 2026-08-14 registry read confirms that the committed
  `posttrain-kind-online-rl-trl-py312@sha256:6c8e…` digest is absent. This is
  not a build failure and must not be repaired by weakening manifest checks;
  it is the concrete reason the next candidate must materialize, push, and
  read back its own base/kind graph before an actual-job delta can be proved.
- Observation: the retained veRL kind is 6.741 GiB compressed compared with
  3.958 GiB for the TRL online-RL kind and 2.608 GiB for the universal base.
  It creates an independently locked backend virtual environment and currently
  resolves released veRL plus the CarbonTeq vLLM fork from Git. This cost is
  paid once per changed kind image, not per job, but individually released
  wheels and a reviewed shared-heavy-package boundary are the next material
  release-time optimization.
- Observation: the release publisher already handles more incremental cases
  than the workflow-level path suggested. It verifies and reuses a committed
  base when source, lock, trust bundle, and remote digest match; verifies and
  reuses every unchanged kind; seeds a changed build from the previous remote
  digest after a local-cache wipe; and builds a missing kind independently.
  The missing cases were remote actual-job receipt recovery and variant-local
  source identity, both now implemented.
- Observation: the universal base installs only CUDA-enabled PyTorch, but it
  copies and identifies itself with the complete workspace constraint lock.
  Consequently, changing Trackio, TRL, Verifiers, or another package that the
  base does not install still invalidates the 2.608-GiB compressed base and,
  because every kind pins its parent digest, every job-kind image. This is the
  largest remaining avoidable rebuild boundary.
- Observation: the kind Dockerfile previously copied the entire `profiles/`
  directory into `kind-common`. A one-line change to `serve.txt` therefore
  changed the filesystem ancestry of supervised, eval, transform, and TRL even
  if the release planner computed a per-kind digest. Each target now copies
  only `common.txt` and its own profile.
- Observation: the current developer path already uses a local builder, not
  the release host. Ambient Agent selects `buildx_builder =
  "posttrain-builder"`; `docker buildx inspect posttrain-builder` resolves to a
  Docker-container BuildKit worker on the developer's `unix:///var/run/docker.sock`
  with an 80 GB cache policy. This explains the cold-VPN behavior precisely:
  source and data originate locally, but the same local machine must first pull
  a 2.8-6.7 GiB kind and then push the job suffix. Moving only the bounded job
  context to a LAN-side developer service removes both heavy VPN legs without
  borrowing release authority.
- Observation: the current configuration puts `buildx_builder` on
  `RegistryBinding`, but the automatically loaded machine configuration already
  has a `MachineServicesBinding` for shared endpoints and credential references.
  A remote publisher is a service choice, not a property of an OCI repository;
  placing its mode/endpoint under `[registry]` would couple transport to image
  identity and make the existing registry table own two unrelated concerns.
  The plan therefore adds `[services.job_builder]` and leaves
  `[registry].buildx_builder` as the local BuildKit adapter setting.
- Observation: `job pack` and `job run` already expose `--build-missing`, but no
  command currently selects a `JobImagePublisher`; `_publisher()` constructs
  `BuildKitJobImagePublisher` directly in `execution_planning.py`. The CLI flag
  must enter at that composition point and must be threaded through `job plan`,
  pack, and run without entering `JobPackageManifest` or
  `ImagePublicationSpec`.
- Observation: accepting only a framework-owned Dockerfile is necessary but
  not sufficient isolation. The actual-job Dockerfile runs `uv pip install`
  over project/framework wheels and source, so package build hooks and project
  code can execute during image construction. The developer job-build service
  therefore needs a rootless, separately scheduled BuildKit worker with no host
  mounts, no release credentials, explicit CPU/memory/disk limits, and a
  restricted egress policy. It cannot safely share the `ai-release` VM merely
  because users do not supply Dockerfiles.
- Observation: `posttrain.execution_pack.JobImagePublisher` is already the
  provider-neutral seam. `BuildKitJobImagePublisher` implements local build,
  remote receipt recovery, parent-prefix verification, and publication. The
  service should not replace or duplicate this logic: the remote client
  implements the same port, while the server invokes the existing BuildKit
  publisher after reconstructing a verified staged context.
- Observation: a remote service must support more than one installed client
  release without accepting build definitions from clients. The safe boundary
  is a server-installed, allowlisted set of Posttrain runtime-definition
  bundles keyed by release-manifest and build-definition digests. A request for
  an unknown bundle is `blocked`, not an instruction to download and execute
  arbitrary Python or Docker content.
- Observation: content-addressed context upload is useful only if identity is
  reproducible across local and remote builders. `publication_key` already
  excludes local paths, but builder-specific timestamps or attestations could
  still change the published OCI index digest. Milestone 10 begins with a
  two-builder prototype against fresh repositories; byte-identical platform
  manifests are a gate. If provenance makes only the index differ, provenance
  must move to a separately attached artifact rather than weakening the rule
  that one publication key names one immutable runtime image.
- Observation: `PackedJobContext.context_digest` covers its complete local tree,
  including empty directories created by the packer. The v1 transfer protocol
  intentionally carries file descriptors plus implicit directories, so it
  cannot honestly recompute that local tree digest. The sealed transport
  identity is therefore `JobContextManifest.digest`; the original packed-tree
  digest remains receipt provenance. A worker reconstructs only the fixed
  staged layout plus declared regular files and must verify each descriptor
  before invoking BuildKit.
- Observation: all required maintained-fork release tags were read back from
  their fork remotes before this developer-builder work resumed. Trackio,
  TRL, veRL, vLLM, and AutomationBench each have an immutable tag and GitHub
  release assets whose published SHA-256 values match `release/forks.toml`.
  Fork releases remain locally built and manually published; the new service
  is exclusively for developer actual-job image publication.

- Observation: an installed wheel cannot pack a job at all today. This is a
  strictly larger gap than "cannot build kind images".
  Evidence: `apps/cli/src/posttrain_cli/execution_planning.py:522-532`
  discovers the framework root by walking the parents of `__file__` looking for
  `containers/posttrain-job/docker-bake.hcl` alongside
  `packages/execution/pyproject.toml`. From `site-packages` that walk finds
  nothing and raises `framework source checkout could not be discovered;
  configure registry.framework_source_root`.

- Observation: `BuildKitRuntimeBuilder` is a complete, tested implementation of
  exactly the build-push-verify-receipt behavior this work needs, and it has no
  production caller.
  Evidence: grepping `BuildKitRuntimeBuilder`, `RuntimeBuildRequest`, and
  `digest_runtime_sources` across `apps/` and `packages/` returns hits only in
  `packages/execution-buildkit/tests/test_builder.py`. The class already
  accepts `bake_file`, `context`, `target`, `repository`, `lock_digest`,
  `base_image`, `builder`, and `variables`, already performs
  `bake --call check` then `bake --push` with provenance and SBOM, already
  reads the pushed digest from the BuildKit metadata file, already re-verifies
  the digest remotely with `imagetools inspect`, and already writes an atomic
  mode-0600 receipt.

- Observation: the `LOCK_DIGEST` label on base and kind images is not computed
  by any code. It is typed by a human at build time.
  Evidence: `packages/execution-buildkit/src/posttrain_execution_buildkit/builder.py:294-305`
  passes `LOCK_DIGEST` for the *runtime* layer only. For base and kind images
  the bake variable defaults to the empty string
  (`containers/posttrain-base/docker-bake.hcl:1-19`,
  `containers/posttrain-job-kinds/docker-bake.hcl:5-7`) and the only documented
  producer is the manual
  `--set '*.args.LOCK_DIGEST=<sha256-of-lock-inputs>'` in
  `containers/posttrain-job-kinds/README.md:51-53`. That same hash is then
  transcribed a second time by hand into
  `[registry.constraint_profiles.*].sha256` in
  `.posttrain/state/execution.toml`. Two independent manual transcriptions of
  one value is precisely how the distillation drift survived.

- Observation: the container build context is almost empty, which makes package
  data far more tractable than the `824K` directory size suggests.
  Evidence: `containers/posttrain-base/Dockerfile:29` and
  `containers/posttrain-job-kinds/Dockerfile:6-8,60` are the only `COPY`
  instructions that read the build context, and they read only
  `containers/posttrain-job-kinds/locks/` and
  `containers/posttrain-job-kinds/profiles/`. The bake files set
  `context = "."` only because those `COPY` paths are written
  repository-root-relative. The shippable payload is about 190 KB, of which
  `locks/workspace.lock.txt` is 118 KB and `locks/transform.lock.txt` is 41 KB.
  The 824 KB figure is dominated by `verl-py313/release/`, a separate,
  release-blocked variant.

- Observation: `containers/posttrain-job-kinds/validate.py` cannot become a
  consumer-side check, and should not try to be one.
  Evidence: it sets `ROOT = Path(__file__).resolve().parents[2]` and validates
  every profile pin against `ROOT / "uv.lock"` and
  `ROOT / "tools" / "quantization" / "uv.lock"`. Neither lockfile exists in an
  installed wheel. It is inherently a framework-repository consistency check
  comparing container profiles to the workspace resolution. The new
  consumer-side check — published image identity versus installed framework
  expectation — is a genuinely different check. Nothing about `validate.py`
  needs to weaken; it only needs re-rooting.

- Observation: `containers/posttrain-job-kinds/tests/test_verl_release_gate.py`
  is never executed by the repository's validation ladder.
  Evidence: root `pyproject.toml` `[tool.pytest.ini_options] testpaths` lists
  only paths under `apps/` and `packages/`. Moving the container definitions
  into a package therefore also brings this test into `uv run pytest` for the
  first time. That is a desirable side effect but it must be verified rather
  than assumed to pass.

- Observation: consumers need the *contents* of `workspace.lock.txt` on every
  pack even if they never build an image, so package data is mandatory rather
  than merely convenient.
  Evidence:
  `packages/execution-buildkit/src/posttrain_execution_buildkit/environment_dependencies.py:120-125`
  compiles each environment's dependency lock with
  `--constraint <workspace.lock.txt>` so that environment wheels resolve
  against the exact dependency set the kind image already provides, and
  `apps/cli/src/posttrain_cli/execution_planning.py:128-135` aborts if that
  file's hash changed since configuration load.

- Observation: the single-prefix registry convention is already followed in
  practice, so introducing `POSTTRAIN_REGISTRY` invalidates no existing digest.
  Evidence: the current machine-local `.posttrain/state/execution.toml` uses
  `registry.lan/carbonteq/posttrain-base`,
  `registry.lan/carbonteq/posttrain-kind-<variant>`, and
  `registry.lan/carbonteq/posttrain-job`. The kind bake file already declares
  `variable "REGISTRY" { default = "registry.lan/carbonteq" }`.
- A fourth container tree existed that the plan had not accounted for:
  `containers/posttrain-job-runtime/`. It predates the base/kind/actual-job
  hierarchy and installs framework packages directly into a base image. Nothing
  in the codebase referenced it; the only mention anywhere was a historical
  image digest in `docs/plan/dstack-execution-provider.md:81`. It was removed
  rather than shipped. With the other three trees moved, `containers/` at the
  repository root no longer exists at all.
- `validate.py` needed a narrower change than expected. Its `parents[2]` root
  still resolves the `containers/` tree correctly at the new depth, and `ROOT`
  turned out to be used for nothing except the two `uv.lock` lookups. Splitting
  `DEFINITIONS` (the shipped tree) from `ROOT` (the framework checkout that owns
  resolved locks) left every existing assertion untouched. Both validators pass
  from the new location.
- The veRL release-gate test was already failing and nobody could have known.
  At `HEAD` both the actual-job Dockerfile and the test asserted
  `"/opt/posttrain-verl/bin/python" -s -c`; the working tree's uncommitted edit
  added `-B` to the Dockerfile but not to the test. Because the test lived
  outside `testpaths` it had never been collected, so the drift was invisible.
  This is the same failure shape as the distillation gate that motivated this
  plan — a check existed but was not wired into anything that runs.
- Separate `as_file` calls per bake file would have been a latent correctness
  bug. `bake` resolves a target's `dockerfile` relative to its `context`, so
  both must come from one extraction; independent `as_file` calls may yield
  different temporary directories for a non-filesystem distribution. The
  accessor module therefore exposes a single `definition_root`, plus a
  process-lifetime `cached_definition_root` because `docker buildx` reads the
  extracted paths long after any `with` block would have closed.
- `framework_source_root` was overloaded and had to be split. It was serving
  both "where the image definitions live" and "which framework source gets
  packed into the job image". Only the first is now package data; the second
  still genuinely requires a checkout, because the actual-job image installs
  framework code from real package roots. An installed consumer therefore still
  cannot pack framework source — a gap this plan does not yet close, and one
  that should be resolved by installing framework wheels in the job image
  rather than copying source.
- Adding a workspace member changed the root `uv.lock` digest, which six
  catalog entries in `packages/catalog/src/posttrain/catalog/base/training.yaml`
  record as `dependency_lock_sha256`. This coupling means any workspace
  membership change invalidates committed catalog values; it is pre-existing
  and was updated in place, not redesigned.
- Two registries were being conflated, and separating them changed the design.
  The framework's release registry (`ghcr.io/carbonteq-ai`) is a property of
  the framework: identical for every consumer, recorded in the distribution,
  never configured. A project's registry (`POSTTRAIN_REGISTRY`) is per-site,
  usually private, and holds that project's own actual-job images; it may also
  hold mirrored base and kind images. An earlier draft of `published.toml`
  recorded `default_prefix = "registry.lan/carbonteq"` — the local private
  registry — which made a site's private registry look like a framework release
  channel. `default_prefix` is now `ghcr.io/carbonteq-ai`, and a mirror prefix
  is passed explicitly rather than being read from the manifest.
- The publish operation does not belong in the consumer CLI. `posttrain` is the
  surface a project developer uses; building and pushing a framework release is
  an owner operation with entirely different authority. Publishing therefore
  lives in a separate `apps/release` distribution that `posttrain` does not
  depend on, so it cannot reach a consumer environment. `list`, `verify`,
  `mirror`, and `build` remain consumer operations.
- The digests currently in `published.toml` are not yet reachable under
  `default_prefix`; they were built and verified against the local registry and
  have not been pushed to the public one. They record the exact images this
  release consists of, and because digests are content-addressed a digest-level
  mirror preserves them across registries. `verify` fails until the release is
  published, which is correct behaviour rather than a defect.
- Moving the definitions invalidated the machine-local `bake_file` binding,
  which still named the deleted `../../containers/posttrain-job/`. It was
  removed from `.posttrain/state/execution.toml` (mode 0600 preserved) so the
  actual-job definition resolves from the installed distribution, which is the
  intended behaviour; setting it again is now only for building against an
  edited working copy.

- Deriving variants from the manifest superseded a config-level invariant. The
  rule that `kind_images` and `constraint_profiles` must declare identical
  variant sets previously caught a declared image with no matching lock. Now
  both sides are pre-filled from the release, so declaring only an image for a
  published variant inherits that variant's lock, which is the desired
  override-one-thing behaviour. The invariant still fires for variants the
  release does not publish, which is the case that genuinely cannot be
  resolved. Two tests encoded the old premise and were rewritten rather than
  deleted: one now asserts partial declaration overrides only itself, the other
  uses an unpublished variant name so the invariant is still exercised.
- A CLI test asserted the old failure. `test_grpo_plan_is_static_and_selects_
  online_rl_runtime` declared only `supervised` and expected planning a GRPO job
  to exit 1 with "runtime variant online-rl-trl-py312 is not published". Under
  the release-pinned model that plan now succeeds, which is the improvement
  this milestone exists to deliver, so the assertion was rewritten to check the
  resolved reference equals the manifest's rather than to expect an error.

- `--build-missing` had to be narrowed during implementation. Treating every
  non-`ok` verification as buildable would have turned an unreachable registry
  into a silent local rebuild, which is the failure mode this plan exists to
  prevent wearing different clothes. An unreachable registry now aborts
  regardless of the flag: absence and unreachability are different facts and
  only the first is safe to remedy by building.
- An image carrying no `lock-digest` label is treated as drift, not as an
  unknown. An unlabelled image cannot be shown to match the framework, and the
  distillation failure is precisely a case where an unverifiable image was used
  as though it were fine.
- The inspector became a protocol rather than a concrete class. Identity
  verification is the most important behaviour in this work and it would
  otherwise have been the least tested, because exercising it meant reaching a
  registry. `ImageInspector` lets the drift, missing, unreachable, and
  unlabelled paths all be tested with no network at all.
- `doctor` deliberately makes no network request. Comparing configured digests
  against the release manifest catches a pinned stale image locally and keeps
  `doctor` fast enough to run habitually; confirming the registry's copy still
  carries the expected label is what `runtime images verify` is for. Splitting
  them avoids a diagnostic command that fails because the network is down.
- Two existing CLI tests had to be adjusted for the new fail-closed behaviour.
  `test_doctor_reports_readiness_and_missing_project` asserted every check was
  `ok`; a fresh project now correctly reports `warn` on `registry` and
  `runtime_images` while still exiting 0, which is the documented acceptance
  criterion. `test_job_pack_publishes_actual_image_without_opening_a_provider`
  packs against a fixture registry holding no real images, so verification is
  stubbed there and exercised directly in `apps/cli/tests/test_runtime_images.py`.

- Confirmed with the framework owner (2026-07-27), after implementation, that
  the three defensible-but-arguable calls made while building milestones 5-7
  should stand: an unreachable registry aborts even under `--build-missing`; an
  image with no `lock-digest` label is drift rather than an unknown; and
  `doctor` keeps two separate warnings on a fresh project. The third was posed
  as redundant noise and deliberately kept, on the grounds that a silent or
  `ok` runtime-images check would read as "images verified" when no image
  identity had been checked at all. Its message was rewritten to say that
  explicitly rather than "nothing to compare".

- A reachability audit after milestone 7 separated two opposite cases that both
  looked like "unused code". `BuildKitRuntimeBuilder` was pre-existing,
  complete, and tested but had no production caller; it was kept and wired,
  because deleting it would have meant reimplementing the same
  build/push/verify/receipt sequence and discarding its coverage. It now has
  three callers. Four symbols added during milestones 5-7 had no caller at all
  and were removed: `drift_failures`, the `IMAGE_LEVEL_LABEL`,
  `REVISION_LABEL`, and `VERSION_LABEL` exports, and the
  `RemoteImageFacts.runtime_variant` property. Speculative surface on a new
  module is worse than on an old one, because it reads as deliberate to the
  next reader and silently acquires callers. The test count was unchanged at
  729 across the removal, which is the evidence that nothing depended on them.
- That removal was then partly reversed, and the reversal found a real hole.
  Challenged on whether the label constants were worth keeping for release and
  identity work, the check that mattered turned out to be missing rather than
  speculative: every level of the image hierarchy is built from the same
  workspace lock, so `published.toml` records an identical `lock_digest` for
  the base image and for every job-kind image. A base image pinned into a
  job-kind slot therefore passed lock-digest verification perfectly while being
  incapable of running a job. `IMAGE_LEVEL_LABEL` is now read first in
  `verify_variant` and rejects any image that is not `job-kind`, and
  `REVISION_LABEL` is surfaced in both success and drift messages so the
  framework commit behind an image no longer has to be recovered by a human
  reading OCI labels — which is exactly how the distillation drift was
  originally found. `VERSION_LABEL` is retained as the documented composite
  identity. The distinction that survives is not "is it referenced" but "does
  it have a reader that would catch something": `drift_failures` and
  `RemoteImageFacts.runtime_variant` stayed deleted because nothing was waiting
  for them.

- Actual-job image versioning turned out to need no new mechanism. Asked how a
  hyperparameter or dataset change should avoid overwriting an image, the
  existing model already answers it: `package_key` hashes a payload containing
  `resolved_inputs_digest`, `resolved_config_digest`, `project_config_digest`,
  both source digests, the dependency closure, the dataset locks, and
  `kind_image`; `publication_key` derives from it and becomes `IMAGE_TAG`.
  Overwriting is therefore impossible by construction. Verified empirically:
  two work packages differing only in their settings id produced different
  `plan_key` and `publication_plan_key` while resolving the same job-kind
  image, which is also the correct layering. A semantic version would be
  strictly weaker, since it depends on a human remembering to bump it and
  permits two configurations to share an identity — the same class of failure
  as the stale job-kind image.
- The real gap was legibility, and `posttrain job diff` closes it.
  `compare_job_packages` attributes a repack to specific fields with a
  human explanation per field. Running it against the 62 retained packages in
  this working tree immediately exposed two defects in its own output: a
  nested `backend_runtime` printed as two long JSON blobs where only
  `projection_digest` differed, and a list absent from an older manifest dumped
  raw. Both forced the reader to diff by eye, which is the work the module
  exists to remove, so comparison now recurses into nested mappings and
  summarizes composites by entry label. `05-apis.md` was amended for the new
  `job diff` verb and to state that job image identity is content-derived
  rather than versioned.

## Decision Log

- Decision: publish base and kind images to a public registry once per
  framework release, and pin their digests in the wheel, rather than expecting
  each consumer to build them.
  Rationale: base and kind images are a pure function of framework inputs
  (`workspace.lock.txt`, `profiles/*.txt`, two Dockerfiles). They contain no
  project code, no datasets, and no environments. For a given release there is
  exactly one correct image per variant. The distillation drift was therefore
  not fundamentally a detection failure but a distribution failure: the TRL pin
  and the image baking it in were maintained independently and reconciled by
  hand. Shipping them as one release artifact makes that divergence
  unrepresentable rather than merely detectable.
  Date/Author: 2026-07-27 / user and Claude.

- Decision: publish to `ghcr.io/carbonteq-ai`.
  Rationale: it matches the repository home already declared in the package
  `[project.urls]`, costs nothing for public images, requires no new
  infrastructure, serves immutable digests, and permits anonymous pulls. Docker
  Hub was rejected because anonymous pull rate limits are hostile to GPU
  workers repeatedly pulling multi-gigabyte images.
  Date/Author: 2026-07-27 / user and Claude.

- Decision: `POSTTRAIN_REGISTRY` is a single prefix holding registry host and
  namespace, for example `registry.lan/carbonteq`, and the framework derives
  `<prefix>/posttrain-base`, `<prefix>/posttrain-kind-<variant>`, and
  `<prefix>/posttrain-job` by convention.
  Rationale: this is already the de facto layout in both the machine-local
  configuration and the kind bake file's `REGISTRY` default, so adopting it as
  the contract renames nothing and invalidates no published digest.
  Date/Author: 2026-07-27 / user and Claude.

- Decision: `POSTTRAIN_REGISTRY` scopes to actual-job images only. Base and
  kind images resolve from the installed release manifest by default.
  Rationale: actual-job images contain the consumer's project code and
  materialized datasets, so they must go to a registry the consumer controls.
  Base and kind images are framework artifacts. Separating the two means a
  first-time consumer needs no registry write access and no BuildKit builder at
  all until they submit their first job.
  Date/Author: 2026-07-27 / Claude.

- Decision: registry configuration is an environment variable, not a CLI
  subcommand and not committed project configuration.
  Rationale: the registry is an operator and machine concern that travels with
  the person, not with the repository. A committed declaration would either
  leak one team's internal hostname into a portable project or require a second
  override mechanism anyway. The hostname is non-secret, so it does not need
  the mode-0600 protection that `.posttrain/state/execution.toml` enforces;
  credentials remain in the Docker keychain and `job.env`, untouched by this
  work.
  Date/Author: 2026-07-27 / user.

- Decision: `posttrain init` is not modified, and gains no `--runtime` flag.
  Rationale: `init` already performs the heavy install step, shelling out to
  `uv sync --python 3.12`
  (`apps/cli/src/posttrain_cli/scaffolding/init_project.py:382-397`). Folding
  registry setup into it would make project creation fail, or silently no-op,
  for every developer without a reachable registry, and would couple project
  scaffolding to infrastructure that `docs/post-training/04-framework.md:170`
  explicitly places outside the framework's ownership. Project creation stays
  offline-capable.
  Date/Author: 2026-07-27 / user.

- Decision: `posttrain job run` verifies and fails closed by default; an
  explicit `--build-missing` flag opts into an inline build.
  Rationale: `job run` today builds only the actual-job image, a thin fast
  layer. Base and kind images carry torch, CUDA, and vLLM. Building them
  implicitly would turn a routine submit into a multi-gigabyte, tens-of-minutes
  rebuild at the least convenient moment, possibly on a machine with no
  configured builder. Failing closed with the expected and found identity makes
  drift loud, which is the entire point of this work, rather than papering over
  it with a silent rebuild.
  Date/Author: 2026-07-27 / user and Claude.

- Decision: the mirror-into-my-own-registry path is first-class from day one,
  not a deferred escape hatch.
  Rationale: airgapped and private-registry-policy sites are real consumers,
  and the framework's own developers need the rebuild path every time they
  change a lock. Treating it as secondary would leave it untested precisely
  where correctness matters most.
  Date/Author: 2026-07-27 / user.

- Decision: preserve all parent layer descriptors when exporting actual-job
  images; compress only newly created job layers.
  Rationale: a digest-pinned kind already fixes the heavy CUDA, PyTorch,
  vLLM, TRL, and veRL bytes. Recompressing that ancestry creates different
  blobs without changing runtime behavior, adds multi-gigabyte first-job
  uploads, consumes BuildKit/OCI disk, and prevents the registry mirror from
  doing its job. The release gate must compare manifests, not merely inspect
  Dockerfile stages.
  Date/Author: 2026-08-14 / user and Codex.

- Decision: image work is selected from immutable input identities, never from
  the fact that a new Posttrain version exists or a local cache is empty.
  Rationale: a release number does not change CUDA, PyTorch, or a trainer
  closure. The desired graph is base identity, then one independent identity
  per kind, then one publication identity per actual job. Local receipts are
  accelerators only; remote manifests are recoverable authority and must be
  consulted before any build.
  Date/Author: 2026-08-14 / user and Codex.

- Decision: the next lock migration will introduce generated base and
  per-kind locks rather than manually trimming the workspace lock.
  Rationale: narrower locks are the only way to prevent an unrelated Python
  dependency from invalidating CUDA/PyTorch or every kind, but an incomplete
  hand-authored constraint closure can silently resolve different artifacts.
  The generator must derive hashes from `uv.lock`, test closure completeness,
  and migrate one image boundary at a time.
  Date/Author: 2026-08-14 / user and Codex.

- Decision: add an optional developer job-build service as a third trust
  domain; keep both the per-developer local builder and the protected release
  builder.
  Rationale: the local builder is the universal fallback and preserves offline
  setup, while a LAN-side service removes multi-gigabyte parent pulls and job
  pushes from slow VPN links. The release builder holds broader publication
  authority and must remain unavailable to unreviewed project contexts. These
  are three different workloads, credentials, availability expectations, and
  blast radii; combining them would optimize bandwidth by weakening release
  isolation.
  Date/Author: 2026-08-14 / user and Codex.

- Decision: retain local/manual fork publication as a separate workflow; use
  the remote builder only when a Posttrain developer elects to publish a
  project-specific actual-job image.
  Rationale: fork releases establish reusable framework dependencies and must
  remain reviewable on the maintainer workstation. Developer jobs instead
  contain unreviewed project context and benefit from a bounded LAN-side
  transport without receiving release credentials or authority. The two paths
  have intentionally different trust, artifact, and operational boundaries.
  Date/Author: 2026-08-14 / user and Codex.

- Decision: select local versus remote actual-job publication through the
  operator-owned machine configuration, with an explicit CLI override.
  Rationale: the service endpoint, trust, credential reference, and ordinary
  default belong to the developer machine just like the registry and other
  internal services; they are not portable project meaning. The precedence is
  `--builder local|remote`, then machine `[services.job_builder].mode`, then the
  compatibility default `local`. Endpoint/token environment variables may
  supply connection values but never turn remote mode on implicitly. The
  selected transport is recorded in the plan and transfer receipt, but excluded
  from package/publication identity.
  Date/Author: 2026-08-14 / user and Codex.

- Decision: transfer a canonical file manifest first and only its missing
  content-addressed file blobs; do not stream the repository, mount a client
  filesystem, or upload an unconditional tar archive.
  Rationale: `execution-pack` has already selected and materialized the exact
  staged context on the developer machine. Hashing those files once lets the
  service return zero payload for an existing publication, deduplicate
  unchanged source/wheels/config across jobs, reject over-budget content before
  transfer, and reconstruct the exact named BuildKit context. The submitter is
  still the source of private project bytes; the service does not invent or
  fetch them from Git.
  Date/Author: 2026-08-14 / user and Codex.

- Decision: the job-build service accepts a sealed Posttrain context manifest
  and missing content blobs, never Dockerfiles, LLB, arbitrary registry
  destinations, registry credentials, or model checkpoints.
  Rationale: the service can recompute package/publication identity, enforce a
  bounded allowlist, reuse identical source and wheel blobs across retries, and
  select a server-installed framework definition. Exposing raw BuildKit would
  create a remote-code-execution service with release-like infrastructure but
  no framework-level policy enforcement.
  Date/Author: 2026-08-14 / user and Codex.

- Decision: keep `JobImagePublisher` in `execution-pack` as the logical port;
  add a separate remote HTTP adapter and a separate `apps/job-builder`
  composition host, while reusing `BuildKitJobImagePublisher` on the server.
  Rationale: package materialization and job identity stay provider-neutral;
  local BuildKit and the remote service must return equivalent
  `PublishedJobImage` values. HTTP/auth/queue state is application
  infrastructure, and OCI building remains in the existing concrete BuildKit
  adapter. This avoids teaching the CLI, common contracts, or training/eval
  packages about service internals.
  Date/Author: 2026-08-14 / Codex.

- Decision: make the first production service single-node and durable rather
  than prematurely distributed.
  Rationale: one rootless BuildKit pool beside one registry eliminates the VPN
  bottleneck. A mode-0700 filesystem content store plus atomic request/receipt
  records and per-publication locks gives restart recovery, missing-blob
  deduplication, and single-flight behavior without introducing a database or
  object-store dependency. `JobContextStore` remains a protocol so a qualified
  object-store implementation can replace it if capacity or high availability
  later requires that change.
  Date/Author: 2026-08-14 / user and Codex.

- Decision: server authorization derives the actual-job repository from the
  authenticated principal and project namespace; clients cannot submit an
  arbitrary repository.
  Rationale: project code and selected small datasets are private. Opaque
  bearer tokens mapped to project scopes are sufficient for the first private
  deployment when stored only as hashes server-side and supplied to clients by
  environment variable. The service alone holds project-scoped push
  credentials; the token, Docker credentials, and other secrets never enter a
  build context or layer. Because repository is already part of
  `ImagePublicationSpec` and therefore the publication key, the CLI resolves
  the same policy-derived repository before materialization and the service
  requires an exact match; local fallback then targets the same repository and
  preserves identity.
  Date/Author: 2026-08-14 / user and Codex.

- Decision: create a new distribution `posttrain-runtime-images` exposing the
  module `posttrain.runtime_images`, rather than adding the package data to
  `posttrain-execution-buildkit`.
  Rationale: the resource package is pure data plus a thin accessor. `apps/cli`
  needs to read the release manifest for `doctor` without pulling in the
  BuildKit adapter machinery, and
  `docs/post-training/04-framework.md:125` charters `execution-buildkit` as an
  adapter, not a data owner. `posttrain` is already an implicit namespace
  package spread across distributions (`posttrain.catalog`, `posttrain.common`,
  `posttrain.data`), so `posttrain.runtime_images` follows the established
  convention.
  Date/Author: 2026-07-27 / Claude.

- Decision: physically move the container definition trees into the resource
  package; do not copy them.
  Rationale: a copy would create two files that must be kept in agreement by
  hand. That is the exact failure mode this plan exists to eliminate. There
  must be exactly one `workspace.lock.txt` in the tree.
  Date/Author: 2026-07-27 / Claude.

## Outcomes & Retrospective

Milestones 1-7 established installable image definitions, release-pinned base
and kind identities, registry/mirror operations, doctor checks, and fail-closed
actual-job packing. Milestone 8's code now preserves parent descriptors and can
recover deterministic remote publications; its live registry proof remains
open. Milestone 9 has narrowed per-kind invalidation and mirror work but still
needs generated lock boundaries and the executable image action plan.

The main retrospective finding is that release efficiency and developer job
efficiency are related by OCI ancestry but owned by different systems. Release
builders should reuse immutable base/kind nodes and publish a candidate
manifest once. Developer-local or developer-service builders should consume
those nodes and add only actual-job deltas. Sharing one builder would reduce
some transfers but collapse credentials and failure domains; the job-build
service instead moves the bounded context to the LAN while preserving that
separation.

Milestone 10 is design-ready but unimplemented. Its highest-risk unknown is
cross-builder byte reproducibility, so the prototype is deliberately the first
gate. The plan is not complete until each implemented milestone records its
actual validation, transfer receipts, surprises, and any changed decisions in
this section.

## Context and Orientation

The repository root is `/home/hammad/projects/rl`. It is a Python 3.12 `uv`
workspace whose members are `apps/*` and `packages/*`.

Some terms used throughout, defined plainly.

An **OCI image** is a container image. A **registry** is the server that stores
them. A **digest** is a `sha256:...` content hash naming one exact immutable
image; a **tag** is a mutable human label that can be repointed at any time.
Everything in this framework pins by digest, never by tag alone.

A **label** is a key-value string baked into an image at build time and
readable afterwards without downloading the image's filesystem layers. This
framework writes labels prefixed `org.carbonteq.posttrain.` and standard ones
prefixed `org.opencontainers.image.`.

**BuildKit** is Docker's build engine. **`docker buildx bake`** builds several
images described by a declarative `docker-bake.hcl` file. A **bake target** is
one named image in that file. A **build context** is the directory tree the
build may read files from.

The framework has a **three-level image hierarchy**:

The **universal base image** (`containers/posttrain-base/`) supplies Python
3.12, CUDA-enabled PyTorch, trusted certificates, and common system packages.

The **job-kind images** (`containers/posttrain-job-kinds/`) add the stable
dependency set for one class of work. Five are published today: `supervised`,
`online-rl-trl-py312`, `eval`, `serve`, and `transform`. A sixth,
`online-rl-verl-py313`, exists as a release-blocked candidate. The word
**variant** in this plan means one of these names.

The **actual-job image** (`containers/posttrain-job/`) adds the selected
project code, resolved configuration, materialized datasets, and environment
wheels for one specific job. It is built per publication key, either by the
consumer's local BuildKit worker or by the optional developer job-build service.
Both paths use the same framework-owned definition and must produce the same
immutable runtime image.

The **developer job-build service** is an optional private application beside a
site registry. It receives only a sealed content manifest plus missing blobs,
reconstructs a bounded staged context, runs the existing actual-job publisher
in an isolated rootless BuildKit worker, and returns a publication receipt. It
does not release Posttrain, build base/kind images, schedule GPU jobs, store
model weights, or expose a general container-build API.

A **publication key** is the SHA-256 derived from the `JobPackageManifest` and
image-publication settings. It is the idempotency and single-flight key for
both local and remote actual-job publication. A **context blob** is one regular
file in the staged context named by its SHA-256 and byte size. A **sealed
context manifest** maps safe relative paths to those blobs and is immutable once
accepted. Directories are implicit; absolute paths, `..`, symlinks, devices,
sockets, setuid bits, and undeclared files are rejected.

A **profile** (`containers/posttrain-job-kinds/profiles/*.txt`) is a plain list
of pinned requirements selecting which packages a variant installs. A **lock**
(`containers/posttrain-job-kinds/locks/*.txt`) is the fully resolved dependency
set. `locks/workspace.lock.txt` is the important one: it is used both as a
build input to the images and, at job-pack time, as a `--constraint` when
compiling each environment package's dependencies.

**Package data** means non-Python files shipped inside an installed Python
distribution. The precedent to follow is `packages/catalog`, which ships YAML
under `packages/catalog/src/posttrain/catalog/base/` with nothing more than
hatchling's default `packages = ["src/posttrain"]`, and reads it at runtime via
`packages/catalog/src/posttrain/catalog/__init__.py:109`:

        return as_file(resource_files("posttrain.catalog").joinpath("base"))

`.posttrain/` is tracked project configuration. `.posttrain/state/` is
gitignored machine-local state, and `.posttrain/state/execution.toml` is the
machine-local execution configuration, parsed by
`apps/cli/src/posttrain_cli/execution_config.py`. Its `[registry]` table is
represented by `RegistryBinding` at `execution_config.py:92-101` and currently
requires that `kind_images` and `constraint_profiles` declare identical variant
sets (`execution_config.py:483-487`). The file is required to be unreadable by
group and others (`execution_config.py:156-159`).

The automatically loaded operator-owned machine configuration is the correct
home for the job-builder default. `MachineServicesBinding` in
`apps/cli/src/posttrain_cli/execution_config.py` already owns shared internal
service locations such as the Python index and actual-job registry. Milestone
10 extends it with a typed `JobBuilderBinding`; it does not move builder choice
into tracked `.posttrain/` project configuration. A project can produce the
same package on two machines even when one publishes locally and the other uses
the remote service.

The intended machine configuration is:

        [services]
        job_registry = "registry.lan/posttrain-projects"

        [services.job_builder]
        mode = "remote"                 # "local" remains the default
        endpoint = "https://job-builder.lan"
        credentials = "job-builder"     # named protected credential source
        request_timeout_seconds = 900
        upload_concurrency = 4

The credential name resolves through the existing machine credential-source
mechanism; it is not the bearer token itself. `POSTTRAIN_JOB_BUILDER_URL` may
replace a missing endpoint and `POSTTRAIN_JOB_BUILDER_TOKEN` supplies an
ephemeral token when no named credential is configured. Neither variable
selects remote mode. The mode precedence for `job plan`, `job pack`, and
`job run` is explicit `--builder`, configured mode, then `local`. `--local` OCI
export and local-daemon execution ignore remote mode because they do not
publish a remotely runnable actual-job image. Project identity comes from the
resolved job package and the actual-job repository continues to come from the
existing registry binding; the service verifies that the authenticated
principal is authorized for both instead of introducing a second project-name
configuration.

Note before starting: the working tree currently has uncommitted modifications
under `containers/`, including `posttrain-job-kinds/verl-py313/Dockerfile`,
`posttrain-job-kinds/verl-py313/profile.toml`,
`posttrain-job-kinds/tests/test_verl_release_gate.py`,
`posttrain-job/Dockerfile`, and untracked files including
`posttrain-job-kinds/profiles/online-rl-verl-py313-control.txt` and
`posttrain-job-kinds/verl-py313/release/{pyproject.toml,uv.lock}`. These edits
must be preserved through the move in Milestone 2. Do not revert them and do
not stash them away and forget them.

### Lifecycle invariants and scenarios

The release and job paths use one desired-versus-observed model. The desired
state comes from immutable source digests, generated lock digests, parent image
digests, trust-bundle bytes, backend fork identities, and publication settings.
The observed state comes from local receipts and OCI registry manifests. A
receipt may avoid a registry lookup, but loss of a receipt must never force a
rebuild when the exact remote object can be verified. Conversely, a receipt
cannot make a garbage-collected or drifted remote object valid.

The invariants are:

1. Every CarbonTeq fork selected by a Python package, runtime lock, external
   environment package, or deployed execution component has its own immutable
   release before Posttrain candidate work starts. Posttrain consumes those
   releases; it does not build forks as a side effect.
2. The universal base changes only when its own Docker/Bake inputs, generated
   base dependency lock, upstream parent digests, or trust-bundle bytes change.
   A framework version by itself is not a base input.
3. A job-kind changes only when its selected base digest, its shared definition
   inputs, its own profile and generated lock, or its backend identity changes.
   A veRL source or lock change cannot invalidate the TRL kind, and a serve
   profile change cannot invalidate supervised.
4. An actual-job image inherits its kind layers byte-for-byte. It may add only
   locked environment wheels, framework/project code, resolved configuration,
   and materialized datasets. CUDA, PyTorch, vLLM, TRL, and veRL are never
   installed per job.
5. Every published object is verified at the destination. A successful build,
   copy, or cached receipt is insufficient until the destination reports the
   expected digest, labels, and, where applicable, ordered parent layers.
6. Candidate and final promotion use the same image bytes. Final promotion
   restores the candidate-generated manifest and publishes Python artifacts
   that contain it; it does not rebuild OCI images under a final label.
7. Registry garbage collection is observable absence. The system may copy an
   exact retained object or rebuild from identical inputs, but it cannot point
   the manifest at an absent parent or treat a child containing duplicated
   ancestry as parent retention.
8. Every networked action has a transfer plan before execution and a transfer
   receipt afterward. Both distinguish logical image size from missing-blob
   bytes, registry-to-registry bytes, builder-context bytes, and bytes crossing
   the submitter's VPN. A cache hit is reported as zero payload bytes rather
   than as a successful multi-gigabyte "copy".
9. Release work and developer job publication use separate builders. The
   protected `ai-release` builder is never an end-user service. Today a
   developer's local `posttrain-builder` owns actual-job publication, so its
   first cold build may pull the selected kind over VPN. The target optional
   job-build service runs next to the registry, accepts only a verified bounded
   Posttrain context and framework-owned build definition, and is isolated from
   release credentials and state. Registry mounts or server-side copies are
   preferred when both endpoints support them.
10. Project build contexts are allowlisted, inspected, and budgeted before a
    builder session starts. Model caches, checkpoints, generated graphs,
    `.posttrain/state`, and undeclared datasets cannot enter generic project
    source. An unexpected context-size increase is a planning error, not an
    acceptable slow upload.
11. Models and checkpoints are immutable artifact references outside runtime
    images. Workers cache them by content/revision digest on shared storage;
    jobs transfer only missing shards. A runtime image or job image never
    embeds model weights merely to make one run self-contained.
12. Small, explicitly selected datasets may be materialized into a job delta;
    large datasets are content-addressed artifacts staged near the worker and
    mounted or fetched by digest. The planner must show which path was selected
    and its byte cost before publication.
13. Publication is single-flight per content key. Concurrent clients requesting
    the same base, kind, job, model, or dataset wait for or reuse one producer;
    they do not duplicate builds and uploads.
14. Platform is part of identity and cost. The release publishes only qualified
    target platforms; adding a second architecture is an explicit additional
    manifest, build, storage, and transfer decision rather than hidden fan-out.
15. Secrets and machine-local credentials are session inputs only. They never
    affect filesystem layers, content keys, logs, or transfer receipts.
16. Local and remote publication share one package key, publication key,
    Dockerfile, bake settings, labels, qualification policy, and parent-prefix
    verifier. Transport selection cannot change job meaning. A remote service
    version that cannot reproduce the requested definition reports `blocked`.
17. A service request names no arbitrary output repository. Authentication
    resolves one principal and its allowed project namespaces; the service
    derives the actual-job repository and uses only that namespace's scoped
    push credential.
18. Context admission finishes before BuildKit starts. The service recomputes
    every blob digest, manifest digest, package key, publication key, framework
    definition digest, kind identity, total bytes, file count, and largest-file
    limit. Large datasets, model/checkpoint files, secrets, special files, and
    unsafe paths fail admission rather than relying on a `.dockerignore` later.
19. Service state is durable and idempotent. `planned`, `uploading`, `queued`,
    `building`, `published`, `failed`, `cancelled`, and `expired` are explicit
    states; process restart replays nonterminal requests, and one terminal
    receipt exists per principal/project/publication key. Cancellation never
    deletes a remotely verified publication.
20. The service reports expected and observed bytes for client upload, parent
    fetch, BuildKit cache, and registry publication separately. Cache eviction
    may make a build slower, but it cannot change identity. Current-release
    parents and active uploads are protected from garbage collection; all other
    cache retention is bounded and observable.

The lifecycle cases are summarized below.

| Situation | Required work | Work that must not happen |
| --- | --- | --- |
| First setup; canonical release registry populated | Install Posttrain, read its manifest, pull base/kind layers lazily, and build only the first actual-job delta | Rebuild base, CUDA, or every kind |
| First setup with a private mirror | Inspect destination tags, copy the base and selected kinds by digest when absent, set `mirror_prefix`, then build the job delta | Compile released dependencies from source |
| Local receipts/cache deleted; registry retained | Recover base/kind from the release manifest and recover an actual job from its deterministic remote tag and labels | Rebuild merely because local state is empty |
| Destination partially populated | Reuse matching destination digests and copy/build only missing objects | Re-copy every manifest or upload existing blobs |
| New Posttrain version; runtime inputs unchanged | Reuse all base/kind digests and render a manifest with the new framework version | Relabel or rebuild image filesystems |
| One kind profile/backend changed | Reuse base and every unrelated kind; build, smoke, and publish only the affected kind | Rebuild all variants |
| Base Dockerfile, trust bundle, PyTorch/CUDA closure changed | Build base once, then rebuild each kind because its parent digest changed | Hide the fan-out or reuse kinds with the old parent |
| Project code/config/data changed | Produce a new package/publication key and push only new actual-job layers | Re-upload or recompress inherited kind layers |
| Retry after interrupted push | Re-read atomic receipt and remote tag; reuse completed content-addressed blobs; retry only missing publication work | Delete broad caches or start the release from scratch |
| Remote service enabled; parent cold on service | Upload only missing bounded context blobs; fetch the exact kind over LAN; publish the job suffix | Pull the kind through the developer VPN or invoke `ai-release` |
| Remote service enabled; identical job already published | Resolve and verify the deterministic remote publication; return its receipt | Upload context blobs or open a BuildKit session |
| Remote service has some context blobs | Upload only the digest set difference, seal the manifest, then build/reuse | Re-send an archive containing every unchanged source and wheel file |
| Remote service unavailable before upload | Show the failure and permit an explicit retry or local-builder fallback with the same plan | Silently switch builders after partial external state or change identity |
| Remote service restarts while building | Recover durable request state, inspect the deterministic remote tag, then requeue only if no verified image exists | Assume failure and overwrite a matching immutable tag |
| Client release unsupported by service | Return `blocked` with accepted release/definition digests and use local fallback or operator upgrade | Download a client-provided Dockerfile or dynamically install unreviewed code |
| One project token attempts another namespace | Reject before blob admission and emit metadata-only audit evidence | Accept a client-supplied repository or reuse another project's credentials |

### Transfer budget matrix

The following numbers are a 2026-08-14 registry snapshot, not estimates from
Dockerfile contents. They are the sum of compressed Linux/amd64 layer
descriptors in the newest retained build manifest for each repository. A kind
total already contains the universal base, so a cold worker pulls the `Total`
column, while a worker that has the exact base needs at most the `After base`
column. Pull time is an ideal lower bound that excludes latency, TLS, manifest
requests, decompression, disk writes, and VPN loss.

| Runtime object | Total compressed | After exact base is cached | 20 Mbps cold | 50 Mbps cold | 100 Mbps cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Universal base | 2.608 GiB | 0 | 18m 40s | 7m 28s | 3m 44s |
| Eval kind | 3.898 GiB | 1.290 GiB | 27m 54s | 11m 10s | 5m 35s |
| TRL online-RL kind | 3.954 GiB | 1.346 GiB | 28m 18s | 11m 19s | 5m 40s |
| veRL online-RL kind | 6.742 GiB | 4.134 GiB | 48m 16s | 19m 18s | 9m 39s |
| Serve kind | 3.815 GiB | 1.207 GiB | 27m 19s | 10m 56s | 5m 28s |
| Supervised kind | 2.867 GiB | 0.258 GiB | 20m 31s | 8m 13s | 4m 06s |
| Transform kind | 2.829 GiB | 0.221 GiB | 20m 15s | 8m 06s | 4m 03s |
| All six kinds, unique layer union | 10.667 GiB | n/a | 76m 21s | 30m 33s | 15m 16s |

The all-kind union is 10.667 GiB, not 23.1 GiB, because OCI registries and
worker content stores deduplicate the shared base and any other identical
layers by digest. A normal worker should not pre-pull this union: it should pull
only the kind selected by the job.

The end-to-end matrix below separates logical payload from the path on which it
moves. `Metadata only` means manifests, configs, labels, and HEAD requests,
normally kilobytes; it does not mean that the object has zero logical size.

| Lifecycle action | Logical payload or observed example | Payload crossing submitter VPN | Efficient required behavior |
| --- | ---: | ---: | --- |
| Verify an existing base/kind/job digest | Metadata only | Metadata only | HEAD/GET manifests; do not pull layers |
| Reuse an existing destination tag | 0 new blob bytes | Metadata only | Compare expected and observed digest and report `reused` |
| Current developer's first local TRL build | 3.954 GiB parent plus context | Up to 3.954 GiB down plus new job layers up | Correct current behavior, but expensive over VPN |
| Current developer's warm local TRL build | Context plus missing job layers | Normally bounded context and job delta | Reuse the developer's local BuildKit parent/cache |
| Optional LAN developer job-build service | Bounded context in; parent and job publication stay on LAN | Context only, 16 MiB in the observed bounded Ambient Agent case | Separate rootless service/API; never the release builder |
| First remote request for a 16 MiB context | 16 MiB plus metadata | At most 16 MiB up | Plan first; upload only missing regular-file blobs |
| Repeat remote request; all context blobs retained | Metadata only | Metadata only | Seal by manifest digest; return an existing publication or reuse the blob store |
| One 2 MiB source or wheel blob changes | 2 MiB plus metadata | About 2 MiB up | Upload only the changed digest; unchanged blobs stay content-addressed |
| Unsupported framework definition | Metadata only | Metadata only | Return `blocked` before context upload or BuildKit allocation |
| Cancel before BuildKit starts | Uploaded missing blobs, if any | No additional payload | Mark cancelled; expire unreferenced blobs under bounded garbage collection |
| Cold worker starts TRL job | 3.954 GiB kind plus job delta | 0 when worker and registry are on LAN | Worker pulls directly from registry |
| Warm worker starts same TRL parent | Only missing job layers | 0 when worker and registry are on LAN | Content-store digest set difference |
| Sample actual-job suffix (17 layers after its 19-layer ancestry) | 24.0 MiB compressed suffix observed | Expected 24.0 MiB after parent preservation is live-qualified; otherwise unproven | Upload new blobs; inherit all 19 parent layers byte-for-byte |
| Same job under the removed forced-recompression path | 3.973 GiB logical child observed | Up to 3.973 GiB on first publication for that rewritten ancestry | Forbidden; parent-prefix verification fails publication |
| Old Ambient Agent project context | 2.0 GiB filesystem payload observed | Up to 2.0 GiB with a remote builder | Forbidden by source allowlist and context budget |
| Current bounded Ambient Agent context | 16 MiB filesystem payload observed | Up to 16 MiB with a remote builder | Send only declared source; builder deduplicates unchanged files |
| Qwen3.5-0.8B local model cache | 1.7 GiB observed | 0 from submitter; missing shards move registry/cache to worker | Worker-shared content-addressed model cache |
| Qwen3.5-2B local model cache | 4.3 GiB observed | 0 from submitter; missing shards move registry/cache to worker | Same; never bake weights into job image |
| Qwen3.5-9B local model cache | 19 GiB observed | 0 from submitter; missing shards move registry/cache to worker | Same; pre-stage near GPU fleet when practical |
| Large selected dataset | Exact artifact size, currently unmeasured by planner | 0 from submitter when artifact store is LAN-side | Digest-addressed stage/mount; no generic source copy |
| First private mirror, all runtime kinds | 10.667 GiB unique worst-case between registries | Metadata only when copy/mount executes server-side | Copy each missing digest once; verify destination |
| New Posttrain version with unchanged runtime inputs | 0 new blob bytes | Metadata and Python release artifact only | Reuse immutable base/kind digests |
| One kind changes | Only missing layers of that kind | Metadata only when build runner is LAN-side | Rebuild/smoke/publish one node; reuse all others |
| Base changes | New base layers plus each kind's changed child layers | Metadata only when release runner is LAN-side | Show explicit fan-out and missing-byte total before build |

The matrix intentionally leaves two quantities dynamic: a changed kind's layer
delta and a selected dataset's artifact size. Guessing either from source files
would be misleading. The desired-versus-observed planner must compute OCI
missing blobs from descriptor digests, inspect the source context before opening
the builder session, and obtain model/dataset artifact byte counts from their
content manifests. It then records expected and observed bytes per link:
`client -> controller`, `controller -> builder`, `builder -> registry`,
`registry -> registry`, and `registry/artifact cache -> worker`.

The current `docker buildx imagetools create` mirror implementation now avoids
work when the destination digest is already correct, but it does not prove that
an absent cross-registry copy is server-side. Until registry-native replication
or observed transfer accounting proves otherwise, a first private mirror must
be scheduled on a site-operated LAN administration host or mirror service and
budgeted as up to the complete missing-blob set. It must use neither a developer
laptop over VPN nor the protected framework release builder.

The optional developer job-build service is not a raw shared BuildKit socket.
Direct BuildKit access would let a user submit an arbitrary Dockerfile and is
too broad a trust boundary. Its API accepts a content-addressed Posttrain job
context and publication key, enforces project namespace, byte, concurrency and
platform budgets, selects the framework-owned Dockerfile by installed release,
uses project-scoped registry credentials, and returns the verified image digest
and transfer receipt. A small installation can omit this service and keep the
current per-developer local builder; the CLI transfer plan then makes the cold
VPN cost explicit before the user starts the build.

The optimized request flow is plan-before-transfer. The client sends package,
publication, release-manifest, build-definition, kind-image, context-manifest,
file-count, and byte-count identities. The service first verifies whether the
deterministic output already exists; if it does, the response is `reused` and
no context moves. Otherwise it returns only missing blob digests. Each blob is
an idempotent whole-file upload with a digest and length check; large datasets
and models are already forbidden from this channel, so resumable multi-gigabyte
archives are unnecessary. The client then seals the context, the service
recomputes every identity, and only then may a queued BuildKit worker start.

The developer machine remains the source of project data. `execution-pack`
first materializes the same staged directory used by today's local named
BuildKit context: `package.json`, hash-locked `locks/`, selected `wheels/`,
bounded `sources/framework/` and `sources/project/`, resolved `config/`, and
only datasets that the existing pack plan explicitly materialized under
`datasets/`. Credentials, `.posttrain/state`, model caches, checkpoints, local
logs, and undeclared repository files never enter this directory. Remote mode
does not clone the project on the server and does not assume the server can read
the developer's filesystem or Git working tree.

After materialization, the client walks only that staged directory and creates
a canonical manifest sorted by relative path. Each regular-file entry records
path, safe mode, uncompressed byte size, and SHA-256. The context digest hashes
the canonical entries, not filesystem timestamps, inode numbers, absolute
paths, a tar stream, or upload order. The initial `:plan` request carries this
manifest and the already-created package/publication metadata over authenticated
HTTPS, but no file contents. This lets the service reject an unsupported
release, unauthorized repository, unsafe path, excessive file count, excessive
single file, or excessive total bytes before project payload crosses VPN.

If the exact image is not already published, the service compares the declared
file digests with its content store and returns the missing set. The client
uploads those files directly from the staged directory with at most the
configured bounded concurrency (four by default). Each HTTPS PUT names one
SHA-256, declares `Content-Length`, writes to a request-scoped temporary file,
hashes while streaming, and atomically promotes only a complete match. Retry
resends only that failed file. Upload ordering is irrelevant, and the API has no
download operation, so deduplication cannot be used to read another project's
content. Text may use ordinary HTTP content compression later, but v1 hashes
and accounts the uncompressed file bytes and does not recompress already-zipped
wheels merely to save a small amount of VPN traffic.

Once every blob is present, `:seal` revalidates the complete manifest. The
service reconstructs a private read-only staged tree with filesystem reflinks
where supported and ordinary copies otherwise; it does not hard-link writable
request paths into the shared content store. That tree becomes the existing
`STAGED_CONTEXT` named context for `BuildKitJobImagePublisher`. The selected
kind image is then pulled from the nearby registry into the service's BuildKit
cache, and the actual-job suffix is pushed back to that registry. Neither the
2.8-6.7 GiB parent nor the produced OCI layers cross the submitter's VPN. The
client receives only state/receipt metadata and the verified digest.

Small explicitly materialized datasets therefore travel exactly like other
context files and are included in the transfer budget. A dataset over the
configured context threshold is rejected and must use the separate
content-addressed artifact staging/mount path; splitting or hiding it inside an
archive is also rejected. On success the client keeps its normal local package
and publication receipts, while the server retains content blobs only under its
lease/retention policy. Local builder mode skips this protocol entirely and
passes the staged directory directly to local BuildKit.

Builder selection behaves as follows:

| Machine mode | CLI option | Effective behavior |
| --- | --- | --- |
| unset or `local` | omitted | Use the current local `posttrain-builder` path |
| `remote` | omitted | Use the configured service and missing-blob protocol |
| any | `--builder local` | Force local publication for this command |
| any | `--builder remote` | Force remote publication; fail before transfer if endpoint/auth/scope is incomplete |
| `remote` | `job pack --local` | Produce the existing local OCI export; do not contact the service |
| any | `job pack --local --builder remote` | Reject the contradictory explicit options |

`job plan` resolves and displays the effective mode without contacting the
service. `job pack` and `job run` materialize the context, call remote `:plan`,
show the exact missing-byte total, and then transfer. `posttrain doctor` is the
explicit live readiness check for endpoint, authentication, project/repository
scope, accepted release definition, and service limits.

The service cache has three independent policies. The context blob store is a
bounded content-addressed cache with active-request leases. BuildKit cache is
bounded by bytes and last use, but exact parents used by queued work cannot be
evicted. Runtime-parent warming is policy-driven: a site may pin the current
release's enabled kinds (10.667 GiB for all six in the measured snapshot), pin
only common kinds, or stay lazy. It must not prewarm every historical release.
The output registry is authoritative after verification; neither context nor
BuildKit cache retention is required to run an already published job.

The current broad workspace lock violates the intended second invariant even
though it remains correctness-safe: the base copies the all-package lock while
installing only PyTorch. Milestone 9 therefore keeps this behavior explicit
until generated `base.lock.txt` and per-kind locks are qualified. It is better
to rebuild too much temporarily than to reuse an image whose dependency closure
was not fully represented in its identity.

## Plan of Work

### Milestone 1 — Amend the frozen baseline

The canonical product documents under `docs/post-training/` are a frozen
baseline. `AGENTS.md` requires that when implementation needs a different
product meaning, the baseline is amended first, narrowly, and only then does
code change. Two amendments are needed, and they must be committed before any
code in later milestones.

In `docs/post-training/05-apis.md`, the primary project CLI block at lines
121-142 enumerates the public command surface. There is no `runtime` noun.
Add to that block:

        posttrain runtime images list
        posttrain runtime images verify
        posttrain runtime images build [--variant VARIANT] [--push]
        posttrain runtime images mirror --to PREFIX

and add a short paragraph after the existing paragraph about `job` and `run`
noun ownership, stating that `runtime` owns framework-owned image identity:
listing what the installed framework expects, verifying what a registry
actually holds, and building or mirroring those images into a registry the
operator controls. State that `runtime` never touches project code, datasets,
or environments, and never builds actual-job images — `job pack` owns those.

Also in `05-apis.md`, document `POSTTRAIN_REGISTRY` as the one environment
variable the CLI itself reads, holding a registry-and-namespace prefix, used to
name actual-job images. State explicitly that it is non-secret and that
credentials are not read from it.

In `docs/post-training/04-framework.md`, add a row to the package table around
line 124 for the new package:

        | `packages/runtime-images` | Framework-owned runtime image
        definitions, dependency locks, and the per-release published image
        manifest, as installable package data | Building images, registry
        operation, provider scheduling, or job semantics |

Rule 13 at line 170 already says the framework owns all job-image semantics and
publishes a universal base, job-kind images, and actual-job images. Extend that
rule with one sentence: the framework distributes its base and job-kind images
as released artifacts pinned by digest in the installed distribution, and
consumers pull rather than rebuild them by default.

Acceptance for this milestone is purely documentary: the two files change, and
`git diff --check` is clean.

### Milestone 2 — The `posttrain-runtime-images` distribution

Create `packages/runtime-images/` following the shape of `packages/catalog`.
Its `pyproject.toml` declares distribution name `posttrain-runtime-images`,
`requires-python = ">=3.12,<3.13"`, `[tool.uv] package = true`, hatchling as
the build backend, and `[tool.hatch.build.targets.wheel] packages =
["src/posttrain"]`. It must have no dependency on any other workspace package;
it is leaf data.

Move, with `git mv` so that history and current uncommitted content are
preserved, the three definition trees to sit beneath a `containers/` directory
inside the resource package:

        packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-base/
        packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/
        packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job/

The nested `containers/` level is deliberate and load-bearing. The Dockerfiles
copy from paths like `containers/posttrain-job-kinds/locks/workspace.lock.txt`,
and the bake files declare `dockerfile =
"containers/posttrain-job-kinds/Dockerfile"`. Preserving that exact relative
layout means **no Dockerfile and no bake file needs editing**, and the
materialized resource directory can be handed to `buildx` directly as the build
context.

Because `git mv` does not move untracked files, move
`profiles/online-rl-verl-py313-control.txt` and
`verl-py313/release/{pyproject.toml,uv.lock}` with plain `mv`. Verify after the
move that `git status` shows renames rather than delete-plus-add for the
tracked files, and that the modified files still carry their modifications.

Add `packages/runtime-images/src/posttrain/runtime_images/__init__.py` exposing
a small accessor mirroring the catalog precedent:

        def definition_root() -> AbstractContextManager[Path]:
            """Yield the directory holding the framework's container definitions."""
            return as_file(resource_files("posttrain.runtime_images").joinpath("containers"))

plus helpers `base_bake_file()`, `kind_bake_file()`, `job_bake_file()`, and
`workspace_lock_path()` that resolve within it. These return context managers
because `importlib.resources` may need to materialize from a zip; callers must
use `with`.

Re-root `validate.py`. It currently computes `ROOT =
Path(__file__).resolve().parents[2]` and reads `ROOT / "uv.lock"` and
`ROOT / "tools" / "quantization" / "uv.lock"`. After the move it sits deeper in
the tree, so the workspace root is further up. Compute the repository root by
walking parents for a directory containing both `uv.lock` and `pyproject.toml`,
and raise a clear error if not found, since this script is only meaningful
inside the framework repository. Do not remove, relax, or narrow any of its
assertions — the profile version pinning check, the full-Git-revision check,
and the forbidden-environment check all stay exactly as strict as they are.

Update `apps/cli/src/posttrain_cli/execution_planning.py`. `_default_framework_root`
at line 522 and `_bake_file` at line 535 must stop walking the filesystem for
`containers/posttrain-job/docker-bake.hcl` and instead resolve the actual-job
bake file through `posttrain.runtime_images`. Keep `registry.bake_file` as an
explicit override so an operator can still point at a checkout. Add
`posttrain-runtime-images` to the dependencies of `apps/cli` and of
`packages/execution-buildkit`.

Add `packages/runtime-images/tests` to `testpaths` in the root
`pyproject.toml`, and move
`containers/posttrain-job-kinds/tests/test_verl_release_gate.py` into it. This
test has never run under `uv run pytest`. Run it and fix what it reports rather
than assuming it passes.

Acceptance: `uv run pytest` collects and passes the relocated veRL release-gate
test; `python -c "import posttrain.runtime_images"` works; building the wheel
with `uv build --package posttrain-runtime-images` produces an artifact
containing `locks/workspace.lock.txt`, verifiable with
`python -m zipfile -l dist/*.whl | grep workspace.lock`.

### Milestone 3 — The release image manifest

Add package data at
`packages/runtime-images/src/posttrain/runtime_images/published.toml`
recording, for each variant, the repository, the exact digest, and the lock
digest that image was built from. Shape:

        schema_version = 1
        framework_version = "0.1.0"
        default_prefix = "ghcr.io/carbonteq-ai"

        [base]
        repository = "posttrain-base"
        digest = "sha256:..."
        lock_digest = "..."

        [kinds.supervised]
        repository = "posttrain-kind-supervised"
        digest = "sha256:..."
        lock_digest = "..."
        constraint_lock = "locks/workspace.lock.txt"

        [kinds.online-rl-trl-py312]
        ...
        provided_packages = ["verifiers"]

Note that `provided_packages` and `constraint_lock` migrate here from
`[registry.constraint_profiles.*]` in the machine-local file. They are
properties of the published image, not of the machine, and this is what stops
them being hand-maintained.

Define the lock digest as the SHA-256 of the referenced constraint lock file's
bytes. This is the same value that `[registry.constraint_profiles.*].sha256`
holds today (currently `e8a833bf...` for `workspace.lock.txt` and `1f8927a7...`
for `transform.lock.txt`), so the definition is unchanged; only its authorship
moves from a human to code.

Add a module `posttrain.runtime_images.manifest` parsing this into frozen
dataclasses `PublishedImage` and `PublishedManifest`, with a
`expected_lock_digest(variant)` accessor that computes the digest from the
shipped lock file and asserts it equals the recorded value. That assertion is
the in-wheel integrity check: it fails at import time if someone edits a lock
without regenerating the manifest.

Acceptance: a unit test asserts that for every variant in `published.toml`, the
recorded `lock_digest` equals the SHA-256 of the shipped lock file. This test
fails if a lock is edited without regenerating, which is the drift class that
caused the distillation failure.

### Milestone 4 — Registry resolution

Add `POSTTRAIN_REGISTRY` reading to `apps/cli`. This is the first environment
variable the CLI itself consumes, so add it in one place —
`execution_config.py` — rather than scattering `os.environ` reads.

Resolution order for the actual-job repository: explicit `[registry].repository`
in `execution.toml` wins if present, else `POSTTRAIN_REGISTRY` suffixed with
`/posttrain-job`, else a clear error naming both remedies.

Resolution order for base and kind images: explicit `[registry].universal_image`
and `[registry].kind_images.<variant>` win if present, else the digest from the
installed `published.toml` prefixed by its `default_prefix`. This preserves
every current setup exactly, including the release-blocked
`online-rl-verl-py313` entry, while making the file optional for new consumers.

Relax `RegistryBinding` accordingly: `kind_images` and `constraint_profiles`
become optional, and the invariant at `execution_config.py:483-487` requiring
identical variant sets applies only to explicitly declared entries. Derive
`constraint_profiles` from the manifest when not declared. Keep the existing
validation that a declared constraint file exists and hashes to its declared
value — that check is correct and should now also run against manifest-derived
entries.

Acceptance: with `.posttrain/state/execution.toml` deleted and
`POSTTRAIN_REGISTRY=registry.lan/carbonteq` exported,
`posttrain --json job plan` resolves the same kind image digests currently
listed in that file. With the file present, behavior is byte-identical to
today.

### Milestone 5 — `posttrain runtime images`

Add `apps/cli/src/posttrain_cli/commands/runtime.py` registering a `runtime`
group with an `images` subgroup, following the existing uniform
`register(app: typer.Typer) -> None` convention and wiring it in `app.py`
alongside the others.

`runtime images list` prints what the installed framework expects: every
variant, its repository, digest, and lock digest, from `published.toml`. Pure
local read, no network.

`runtime images verify` queries the configured registry for each variant and
compares the published `org.carbonteq.posttrain.lock-digest` label and the
resolved digest against the manifest. Reports per-variant `ok`, `missing`, or
`drifted`, and exits non-zero if any is not `ok`.

`runtime images build [--variant] [--push]` builds base and kind images from
the package-data definitions. Implement this by materializing
`definition_root()` into a temporary directory and constructing
`RuntimeBuildRequest` values for `BuildKitRuntimeBuilder`, which already
performs check, build, push, remote verification, and receipt writing. This is
the milestone that gives that orphaned class its first production caller.

Critically, compute `LOCK_DIGEST` in Python from the shipped lock file and pass
it as a bake variable. This deletes the hand-typed
`--set '*.args.LOCK_DIGEST=...'` step documented in the kind README, and it is
the single change that most directly prevents a recurrence of the distillation
drift, because the label can no longer disagree with the lock it claims to
describe.

`runtime images mirror --to PREFIX` copies the released digests into another
registry prefix, for airgapped and private-registry-policy consumers. Implement
with `docker buildx imagetools create`, which copies by digest without pulling
layers locally, then re-verify the destination digest.

Add a release-time regeneration path so `published.toml` is written from a real
build rather than by hand. This is what closes the loop: the release process
builds, pushes, reads back digests, and rewrites the manifest in one operation.

Acceptance: `runtime images list` works offline with no registry configured.
`runtime images verify` against the current `registry.lan` reports `ok` for the
five published variants. A deliberately corrupted `lock_digest` in a test
fixture makes `verify` report `drifted` and exit non-zero.

### Milestone 6 — `doctor` checks

`doctor` currently has no check abstraction: `commands/doctor.py:37-106` builds
a `list[dict[str, str]]` inline in the command body, with only `ok` and `error`
statuses. Adding two more checks by appending literals would work but would
make this the fifth hand-rolled entry and would offer nowhere to express "the
registry is not configured, which is fine for validation but blocks
submission".

Introduce a minimal abstraction: a frozen `Check` dataclass with `name`,
`status`, and `message`, a `CheckStatus` literal widened to `ok`, `warn`,
`error`, and a list of callables that each take the CLI state and return
`Check | None`. Port the five existing checks unchanged in behavior and output.
Keep the JSON payload shape `{"ok": ..., "checks": [...]}` and the text format
`f"{status.upper():5} {name}: {message}"` exactly as they are, so existing
consumers and `tests/test_cli.py:573-588` continue to pass. Exit non-zero only
on `error`, so `warn` is reportable without breaking anyone's pipeline.

Then add two checks. `registry` reports `ok` with the resolved prefix when
`POSTTRAIN_REGISTRY` or an explicit repository is set, and `warn` otherwise
with the message that job submission requires it. `runtime_images` compares
each configured or derived kind image against the installed manifest and
reports `error` on any drift, naming variant, expected digest, and found
digest.

The second check is the one that would have caught the distillation failure. It
must be written so that it does so: its test should construct a configuration
whose kind image lock digest differs from the installed manifest and assert
that `doctor` exits 1 with a message naming the variant and both digests.

Acceptance: `posttrain doctor` on a fresh `posttrain init` project with no
registry configured reports all `ok` except a `warn` on `registry`, and exits
0. With a drifted kind image it exits 1.

### Milestone 7 — `job run` fails closed

In `commands/job.py`, before packing, verify that the registry is configured
and that each required kind image is present with a matching lock digest. On
failure raise a `ContractError` naming the variant, the expected digest, the
found digest, and the exact remedy command.

Add `--build-missing` to `job run` and `job pack`. When passed, absent or
drifted images are built and pushed inline via the Milestone 5 code path before
packing proceeds. Without it, the run aborts. Default is abort.

Acceptance: a job run against a drifted kind image exits non-zero before any
image is packed or any provider object is created; the same command with
`--build-missing` rebuilds, pushes, and proceeds.

### Milestone 8 — Verify actual-job deltas and recover remote publications

Set every actual-job exporter to preserve parent compression. After a push,
read the OCI manifests for each requested platform, require the complete parent
layer sequence as the child prefix, and report only the additional layer count
and bytes. Before building, inspect the deterministic
`<repository>:<publication-key>` tag. Reuse it only when its package, kind,
runtime-variant, and image-level labels match and its parent layer sequence is
intact; then recreate the protected local receipt. A mismatched deterministic
tag is corruption and fails rather than being overwritten.

### Milestone 9 — Plan and execute only changed image nodes

Give each kind a variant-local runtime source digest and ensure its Docker
stage copies only its own profile. Preserve shared Docker/Bake inputs as shared
invalidation boundaries. Make mirror operations inspect the destination tag
first and report `reused` when it already names the desired digest. Next, add a
pure release image plan containing one action per node (`reuse-remote`,
`copy`, `build`, or `blocked`) and make publication execute that plan. Finally,
generate a narrow base lock plus per-kind locks from `uv.lock`; prove that each
lock is a complete hash-pinned closure before switching one image at a time.

### Milestone 10 — Optional developer job-build service

This milestone introduces a new public service and therefore changes the
frozen product baseline. Before implementation, narrowly amend
`docs/post-training/04-framework.md` and `05-apis.md`. The amendment must keep
the existing meaning: `job pack` owns actual-job materialization and identity;
the new service is only an optional publication transport. Add an
`apps/job-builder` row owning authenticated request lifecycle, context
admission, queueing, and deployment. Add a
`packages/execution-job-builder` row owning the concrete remote client adapter.
Keep `execution-pack` as owner of `JobImagePublisher` and request/result values,
and keep `execution-buildkit` as the only BuildKit implementation. Document the
service API, machine `[services.job_builder]` binding,
`--builder local|remote` override, context-transfer evidence, and the local
fallback. Do not add the service to `common`, train, eval, serve, Observatory,
Lab, or the release application.

Implement in the following gates. Do not skip the first gate merely because a
remote build appears to work.

**Gate 10.1 — reproducibility prototype.** Build the same minimal staged
context with two independent local BuildKit workers, using the exact packaged
Dockerfile/Bake definition, parent digest, platform, compression settings,
publication spec, and a fresh repository for each worker. Compare package key,
publication key, platform-manifest digest, config digest, ordered parent
descriptors, job-layer descriptors, labels, and final OCI index digest. The
platform image must be byte-identical. If provenance/SBOM attestations make the
index builder-specific, attach them as separate referrer artifacts or otherwise
make their inputs deterministic; never permit one publication key to resolve
to different runtime bytes. Record the result in `Surprises & Discoveries`
before creating a network API.

**Gate 10.2 — contracts and local content store.** Extend
`posttrain.execution_pack` with provider-neutral values for a sealed context
manifest, file-blob descriptor, transfer plan, transfer receipt, request state,
and service capability. A manifest entry contains a normalized POSIX relative
path, SHA-256, exact byte size, and safe mode bits. Only regular files and
implicit directories are supported in v1. Derive the manifest from the already
materialized `PackedJobContext`; do not rescan the original repository or
create a second source-selection policy.

Define a `JobContextStore` protocol in the service application boundary and a
single-node filesystem implementation first. Store blobs at
`blobs/sha256/<prefix>/<digest>`, request records under
`requests/<principal>/<project>/<publication-key>/`, and receipts separately.
Use mode 0700 directories, mode 0600 atomic JSON, per-publication locks, and
lease files for active uploads/builds. A startup reconciliation pass inspects
every nonterminal request, checks the deterministic remote tag first, and
requeues only work without a verified publication. Garbage collection removes
only unleased blobs with no retained manifest reference after the configured
retention period. Keep the protocol narrow enough for a later object-store
adapter, but do not require RustFS/S3 for the first deployment.

**Gate 10.3 — authenticated plan-before-upload API.** Add
`apps/job-builder/`, distribution `posttrain-job-builder`, with these v1
operations:

        GET  /health/live
        GET  /health/ready
        GET  /v1/capabilities
        POST /v1/job-publications:plan
        PUT  /v1/job-publications/{publication_key}/blobs/{sha256}
        POST /v1/job-publications/{publication_key}:seal
        GET  /v1/job-publications/{publication_key}
        POST /v1/job-publications/{publication_key}:cancel

`capabilities` returns accepted release-manifest/build-definition digests,
platforms, context/file limits, queue availability, and API schema versions.
The plan request carries the complete `JobPackageManifest`, publication spec,
project identifier, sealed-context manifest identity and descriptors, but no
file bytes. Repository remains in the publication spec because it is already a
publication-key input. The service authenticates first, derives the one allowed
repository from principal/project scope, rejects any mismatch, recomputes the
package and publication keys, checks its installed release bundle and exact
kind digest, verifies an existing deterministic publication, then returns one
of `reused`, `upload-required`, `queued`, `building`, or `blocked`. An
`upload-required` response contains only missing blob digests and expected VPN
bytes.

Each blob PUT is idempotent and verifies authorization, declared membership,
`Content-Length`, maximum size, and SHA-256 before atomically exposing the blob.
Because this channel forbids large datasets, models, checkpoints, and arbitrary
archives, v1 retries one bounded whole-file blob instead of adding a multipart
upload protocol. Seal succeeds only when every declared blob is present and
the reconstructed tree and all keys revalidate. It transitions exactly once
to `queued`. GET returns state, timestamps, safe error code, verified image when
published, cache decisions, expected/observed transfer counters, parent/job
layer counts, and receipt digest. It never returns build logs containing source
or environment values. Cancel is idempotent; it cancels an upload/queue/build
where possible but returns the verified publication if one won the race.

For the first private deployment, use opaque bearer tokens whose SHA-256 values
and project scopes are stored in a protected server configuration. The client
reads the token only from `POSTTRAIN_JOB_BUILDER_TOKEN`; the endpoint may come
from `POSTTRAIN_JOB_BUILDER_URL` or machine-local execution configuration.
Neither value is committed. VPN reachability is not authentication. The
service derives a project-scoped repository such as
`registry.lan/posttrain-projects/<principal>/<project>/posttrain-job`; it
accepts the repository only when it exactly matches that derived value and
never accepts a client Docker credential. Audit events contain
principal, project, keys, state transition, byte counts, and result code, not
source paths/content, tokens, registry secrets, or full build logs.

**Gate 10.4 — isolated build worker and queue.** Run the service on a dedicated
developer-build VM/pool beside the registry, not on `ai-release` and not on a
GPU worker. Use rootless BuildKit with no host Docker socket, no host filesystem
mounts, no release credentials, bounded CPU/memory/PIDs/disk, and restricted
network egress to the configured registry and approved Python index only. The
service selects one server-installed `posttrain-runtime-images` definition
bundle by exact manifest/build-definition digest, reconstructs the staged
context into a private temporary directory, and calls the existing
`BuildKitJobImagePublisher`. It must not accept Dockerfile, Bake, LLB, build
args outside the typed publication contract, extra contexts, entitlements, or
secrets from the client.

Use one durable FIFO queue with configurable global, per-principal, and
per-project concurrency. Single-flight is keyed by
`(principal, project, publication_key)` and checks the registry before
allocation. A queue request has a maximum wait and build duration. On timeout
or worker failure, preserve verified remote content and request receipts, clean
only the request's temporary reconstruction, and allow safe retry. Configure
separate byte limits for total context, file count, largest file, retained
context store, BuildKit cache, and audit/receipt data. The initial platform
allowlist is `linux/amd64`; multi-platform fan-out is a later explicit
qualification.

**Gate 10.5 — remote client and CLI selection.** Add
`packages/execution-job-builder/`, distribution
`posttrain-execution-job-builder`, implementing the existing
`JobImagePublisher` port over the v1 API. It plans before transfer, uploads only
missing blobs with bounded retries, seals, polls with server-provided retry
intervals, verifies the returned publication key/image/kind identity, and
writes the same protected local publication record shape used by package
history. Do not put HTTP in `execution-pack`.

Change the CLI composition point in
`apps/cli/src/posttrain_cli/execution_planning.py` so the resolved
`MachineServicesBinding.job_builder.mode` selects the current
`BuildKitJobImagePublisher` for `local` and the new adapter for `remote`. Add a
frozen `JobBuilderBinding` in `execution_config.py` with `mode`, `endpoint`,
named `credentials`, `request_timeout_seconds`, and `upload_concurrency`.
Parse it from `[services.job_builder]` in the automatically loaded machine
configuration; keep tracked project configuration free of this machine
selection. Remote configuration uses the existing resolved project identity
and actual-job repository, and `doctor` checks that the authenticated service
authorizes both. Default to `local` for compatibility.

Add `--builder local|remote` to `posttrain job plan`, `job pack`, and `job run`
as an explicit one-command override and show the configured source, effective
builder, endpoint host, and expected VPN bytes in plan/pack output. The flag
overrides configured mode but does not alter `JobPackageManifest`,
`ImagePublicationSpec`, package key, publication key, or output repository.
Reject explicit `--builder remote` with `job pack --local`; configured remote
mode alone does not block local OCI export. Never
silently fall back after a remote request reaches `uploading`, `queued`, or
`building`; the user may explicitly retry remote or rerun local with the same
publication key. `pack_local` and `pack_local_daemon` always remain local.
`posttrain doctor` checks service reachability, authentication, accepted
release definition, project scope, limits, and registry readiness when remote
mode is configured.

**Gate 10.6 — cache and deployment qualification.** Provide a service
configuration and ai-infra deployment role with an explicit data volume,
backup exclusion (cache/blob bytes are reconstructible; receipts/configuration
are not), TLS/private ingress, token provisioning, resource limits, health
checks, log rotation, context/BuildKit garbage collection, and alert thresholds
for queue age, failure rate, disk pressure, and rejected context budgets.
Support `prewarm = none | selected | enabled`; default to selected site-common
kinds. Prewarming is asynchronous and reads only released parent digests. A new
Posttrain release adds an allowlisted definition bundle side-by-side, warms its
selected parents, passes readiness, and only then removes an old bundle after
no active request references it. It never rebuilds framework base/kind images.

Acceptance requires equivalent local and remote logical results, not merely a
successful HTTP response. Unit tests cover path safety, digest/size mismatch,
unsupported release, unauthorized namespace, quota rejection, state
transitions, cancellation races, startup recovery, blob/request garbage
collection, and redaction. Contract tests run the same `JobImagePublisher`
suite against local BuildKit and an in-process remote service. A real
integration starts the isolated service and registry, publishes one job from a
cold service cache, repeats it with no context upload/build, changes one small
source file and uploads only that blob, restarts during a queued request, and
confirms recovery. The published child must preserve its parent layer prefix;
the transfer receipt must prove that the submitter sent only admitted context
bytes. Finally, run one TRL and one veRL local-executor job using remote-built
images and verify their retained job-package evidence. GPU execution validates
runtime compatibility; it is not used to validate the upload protocol itself.

## Concrete Steps

Run everything from `/home/hammad/projects/rl`.

Before starting, record the current state so the move can be checked:

        git status --porcelain > /tmp/pre-move-status.txt
        sha256sum containers/posttrain-job-kinds/locks/workspace.lock.txt

That hash must equal `e8a833bf24f5fe5459ee69eb04d26a9ea5cfc49bd0b6dd8dc3b678c310fcfbbd`,
the value currently recorded in `[registry.constraint_profiles.*].sha256`. If
it does not, stop: the working tree already carries undetected drift, and that
must be understood before proceeding.

After each milestone, run the ladder:

        uv sync --all-packages --locked --python 3.12
        uv run ruff check .
        uv run pyright
        uv run lint-imports
        uv run pytest
        git diff --check

Milestone 2's move, in order:

        mkdir -p packages/runtime-images/src/posttrain/runtime_images
        git mv containers/posttrain-base packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-base
        git mv containers/posttrain-job-kinds packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds
        git mv containers/posttrain-job packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job
        git status --porcelain | grep '^R' | wc -l

Then confirm the uncommitted edits survived:

        git diff --stat -- packages/runtime-images

This must still list `verl-py313/Dockerfile`, `verl-py313/profile.toml`,
`tests/test_verl_release_gate.py`, and `posttrain-job/Dockerfile` as modified.
If it does not, the modifications were lost — recover with `git stash list` and
restore before continuing.

Verify the shipped wheel actually carries the locks:

        uv build --package posttrain-runtime-images
        python -m zipfile -l dist/posttrain_runtime_images-*.whl | grep -E 'workspace.lock|Dockerfile|docker-bake'

Verify the portability claim end to end, which is the whole point:

        mv .posttrain/state/execution.toml /tmp/execution.toml.bak
        export POSTTRAIN_REGISTRY=registry.lan/carbonteq
        uv run posttrain doctor
        uv run posttrain --json job plan .posttrain/work_packages/<package>.yaml --job <job> --provider local
        mv /tmp/execution.toml.bak .posttrain/state/execution.toml

The plan output's kind image digest must equal the one recorded in the
backed-up file for that variant.

For Milestone 10, execute in this order so service work cannot conceal an
unreleased fork or an unqualified image hierarchy:

1. From each maintained sibling fork selected by `uv.lock` and the runtime
   locks, record branch, commit, remote, tag/release state, and clean/dirty
   status. Publish forks manually according to `docs/tooling/forks.md`; do not
   add or invoke fork release runners. Update their `CARBONTEQ_FORK.md` and the
   matching `docs/tooling/<tool>/README.md`, then update exact Posttrain pins.
2. Complete Milestones 8 and 9's remaining live registry gates. In particular,
   prove parent-prefix preservation, remote tag recovery, partial-registry
   reuse, per-kind invalidation, and generated-lock closure before using those
   behaviors behind a service.
3. Land the narrow canonical baseline amendment, then the two-builder
   reproducibility prototype. Stop if one publication key produces different
   platform bytes.
4. Implement and validate provider-neutral context/transfer contracts, the
   filesystem store, and service state machine without HTTP.
5. Add the HTTP app and remote adapter, then run the shared publisher contract
   suite and an isolated registry integration.
6. Add the separate ai-infra developer-builder role and qualify it on the LAN.
   Do not modify the `ai-release` role except to assert that the new service is
   absent from it.
7. Publish a Posttrain release candidate to the development channel and
   materialize its exact framework base/kind images through the protected
   Posttrain release process. This is framework-release work, not use of the
   developer job builder. Run one TRL and one veRL canary against those pinned
   images; their project-specific actual-job image may use the developer's
   selected local or separately qualified remote builder. Retain the
   transfer/job-package receipts. Final promotion restores and verifies those
   exact candidate manifests and Python artifacts; it performs no image
   rebuild. Promote only after the fork ledger, candidate evidence, cleanup
   receipts, and release audit all pass.

During development, the narrow service test ladder is:

        uv run pytest packages/execution-pack/tests
        uv run pytest packages/execution-job-builder/tests
        uv run pytest apps/job-builder/tests
        uv run pytest apps/cli/tests/test_execution_config.py apps/cli/tests/test_cli.py -k 'job_builder or builder'
        uv run pytest packages/execution-buildkit/tests/test_job_image.py
        uv run ruff check packages/execution-pack packages/execution-job-builder apps/job-builder
        uv run pyright
        uv run lint-imports
        git diff --check

The live transfer qualification must retain machine-readable receipts for all
three paths: cold local builder, cold remote service, and warm remote service.
Compare actual VPN bytes rather than elapsed time alone. The expected shape is:
local cold includes the selected kind download and job upload; remote cold
includes only missing admitted context; remote warm is metadata-only for an
identical publication.

Retain one redacted transfer transcript with this shape so a future reader can
distinguish metadata planning from payload transfer:

        builder source=machine mode=remote endpoint=job-builder.lan
        context files=... total=16.0 MiB digest=sha256:...
        remote publication state=upload-required missing=... bytes=...
        uploaded files=... bytes=... retries=0
        remote publication state=published image=...@sha256:...
        vpn client bytes=... parent fetch bytes=... registry push bytes=...

Repeat immediately and expect `state=reused`, `missing=0`, `uploaded files=0`,
and `vpn client bytes=0` apart from HTTP metadata. Redact tokens, full local
paths, source names that expose private content, and registry credentials.

## Validation and Acceptance

The work is accepted when a developer with no framework checkout can do the
following, and each step is independently observable.

Install the framework, set `POSTTRAIN_REGISTRY` to any OCI registry they can
push to, and run `posttrain doctor`. It reports Python, project, catalog, work
packages, registry, and runtime images. Registry shows the prefix. Runtime
images shows five variants resolved from the installed manifest.

Run `posttrain runtime images list` with no network. It prints five variants
with digests, offline.

Run `posttrain runtime images verify`. It queries the registry and reports each
variant `ok`.

Run a job. It packs and submits without ever consulting a `containers/`
directory on disk.

With no job-build service configured, that job uses the existing local builder
and the plan reports the possible cold-parent VPN cost. With remote mode
configured, `doctor` proves the service accepts the installed definition and
the selected project scope. Planning an already-published job returns `reused`
without uploading context. Planning a new bounded job reports the exact missing
blob set and byte count before transfer.

Exercise the complete selection precedence. With no machine setting, all three
job commands report `builder=local`. With machine mode `remote`, they report
`builder=remote` and its source as machine configuration. `--builder local`
overrides that mode, while `--builder remote` overrides configured local mode.
Setting only `POSTTRAIN_JOB_BUILDER_URL` or a token must not enable remote mode.
An explicit remote override without endpoint or credential fails before
materialization/upload. `job pack --local` under configured remote mode remains
local, while combining it with explicit `--builder remote` is a clear argument
error.

For a cold remote build, capture a receipt showing client-upload bytes, LAN
parent-fetch bytes, registry-push bytes, inherited parent layers, and new job
layers. Repeat the identical job and require zero client payload and no
BuildKit allocation. Change one regular source file and require the second plan
to request only that file's new blob. Reject a 2 GiB accidental project
context, a model checkpoint, a symlink escape, a mismatched blob digest, an
unknown definition digest, and an unauthorized project before BuildKit starts.

Stop and restart the service once while a request is queued and once after the
registry push but before the client receives the response. In both cases the
client must converge on the same verified publication rather than creating a
second build. Cancel a queued request and a running request; neither path may
delete a publication that completed concurrently. Remove the local client
receipt and require the service/registry to recover the result without upload.

Inspect the admitted transfer itself. The context manifest must enumerate only
the materialized `package.json`, locks, wheels, bounded sources, config, and
selected small dataset files. The sum of uploaded `Content-Length` values must
equal the service's missing-byte plan and the receipt's observed client bytes.
No OCI parent/output descriptor, model/cache/checkpoint file, credential, or
unselected repository path may appear in the manifest or network capture.

For one representative job per runtime variant, read the parent and child OCI
manifests from the target registry. Every parent layer digest must appear in
the child in the same order, and the receipt must report only the additional
job-layer count and compressed bytes. A new kind may require workers to pull
its heavy layers once; submitting another job against that kind must not
recompress, rebuild, or upload CUDA/framework layers.

Corrupt the picture deliberately: point `[registry].kind_images.supervised` at
an older published digest whose `lock-digest` label differs. `posttrain doctor`
now exits 1 naming the variant and both digests, and `posttrain job run` aborts
before creating any provider object. Passing `--build-missing` rebuilds from
package data, pushes, and proceeds. This is the exact scenario that silently
invalidated the distillation qualification, and it must now be impossible to
walk past.

The repository ladder must pass, and `uv run pytest` must now include the
relocated veRL release-gate test that previously never ran.

`containers/posttrain-job-kinds/validate.py` must still pass with every
assertion intact. Confirm by running it directly and by diffing it against its
pre-move version to show that only the root computation changed.

## Idempotence and Recovery

`runtime images build` is content-addressed and safe to repeat: the existing
receipt mechanism in `BuildKitRuntimeBuilder` reuses a prior publication when
the build key matches, and re-verifies the digest remotely, rebuilding only if
the registry no longer holds it. `runtime images mirror` copies by digest and
is naturally idempotent. `runtime images verify`, `list`, and `doctor` are
read-only.

The Milestone 2 move is the only destructive step. It is recoverable with
`git restore --staged --worktree containers packages/runtime-images` provided
it is done before committing, but the uncommitted worktree edits under
`containers/` are the real risk. Commit or stash-with-a-named-message those
edits first so they can be recovered by name if the move goes wrong.

Deleting `.posttrain/state/execution.toml` during validation is safe as long as
it is backed up first, as shown above. It is machine-local and gitignored, so
it cannot be recovered from version control.

If `published.toml` is ever wrong, `runtime images verify` detects it and the
release-time regeneration path rewrites it from real registry state. Never edit
digests in it by hand — that reintroduces exactly the manual transcription this
plan removes.

The remote publication lifecycle is also content-addressed. Repeating `plan`
for a terminal publication verifies and returns the existing image. Repeating
a blob PUT either verifies the retained blob or replaces only a quarantined
partial file. Repeating `seal` returns the current state. A service crash leaves
atomic request records and leased blobs; startup reconciliation first inspects
the deterministic registry tag, then requeues only when no matching image is
present. A user may explicitly switch to the local builder after a remote
failure because both paths share identity, but must not do so automatically
while the remote request is nonterminal.

Service cleanup is scoped and recoverable. Cancelling one request removes only
its temporary reconstructed tree and lease; shared content blobs remain until
unreferenced retention expires. BuildKit garbage collection cannot delete OCI
registry content. Deleting a context blob merely causes a future plan to ask
for it again. Deleting a request receipt is an operator action and is safe only
after the registry publication and audit receipt have been verified. Never run
broad recursive cleanup against the service data root or BuildKit root.

If the remote service is unavailable, the supported recovery is an explicit
local build using the already materialized plan. If the client release is not
allowlisted, the recovery is either local build or an operator-installed
definition bundle; the server never fetches a definition from the request. If
the service approaches disk limits, it stops admitting new uploads before it
evicts active requests, current-release parents, or terminal receipts.

## Interfaces and Dependencies

In `packages/runtime-images/src/posttrain/runtime_images/__init__.py`:

        def definition_root() -> AbstractContextManager[Path]: ...
        def base_bake_file() -> AbstractContextManager[Path]: ...
        def kind_bake_file() -> AbstractContextManager[Path]: ...
        def job_bake_file() -> AbstractContextManager[Path]: ...
        def workspace_lock_path() -> AbstractContextManager[Path]: ...

In `packages/runtime-images/src/posttrain/runtime_images/manifest.py`:

        @dataclass(frozen=True, slots=True)
        class PublishedImage:
            variant: str
            repository: str
            digest: str
            lock_digest: str
            constraint_lock: str
            provided_packages: tuple[str, ...] = ()

        @dataclass(frozen=True, slots=True)
        class PublishedManifest:
            schema_version: int
            framework_version: str
            default_prefix: str
            base: PublishedImage
            kinds: Mapping[str, PublishedImage]

            def image_ref(self, variant: str, *, prefix: str | None = None) -> str: ...
            def expected_lock_digest(self, variant: str) -> str: ...

        def load_manifest() -> PublishedManifest: ...

In `packages/execution-pack/src/posttrain/execution_pack/remote_publication.py`
(names may move within that distribution, but ownership must not):

        @dataclass(frozen=True, slots=True)
        class ContextBlob:
            path: str
            sha256: str
            size: int
            mode: int

        @dataclass(frozen=True, slots=True)
        class SealedJobContext:
            schema: str
            context_digest: str
            package_key: str
            file_count: int
            total_bytes: int
            blobs: tuple[ContextBlob, ...]

        class JobPublicationState(StrEnum):
            PLANNED = "planned"
            REUSED = "reused"
            UPLOAD_REQUIRED = "upload-required"
            UPLOADING = "uploading"
            QUEUED = "queued"
            BUILDING = "building"
            PUBLISHED = "published"
            BLOCKED = "blocked"
            FAILED = "failed"
            CANCELLED = "cancelled"
            EXPIRED = "expired"

        @dataclass(frozen=True, slots=True)
        class JobBuildTransferPlan:
            publication_key: str
            state: JobPublicationState
            missing_blobs: tuple[str, ...]
            expected_client_bytes: int
            retry_after_seconds: int | None

        @dataclass(frozen=True, slots=True)
        class JobBuildTransferReceipt:
            publication_key: str
            image: RuntimeImageRef
            client_bytes: int
            parent_fetch_bytes: int
            registry_push_bytes: int
            inherited_layers: int
            job_layers: int
            receipt_digest: str

In `apps/cli/src/posttrain_cli/execution_config.py`:

        @dataclass(frozen=True, slots=True)
        class JobBuilderBinding:
            mode: Literal["local", "remote"] = "local"
            endpoint: str | None = None
            credentials: str | None = None
            request_timeout_seconds: int = 900
            upload_concurrency: int = 4

        @dataclass(frozen=True, slots=True)
        class MachineServicesBinding:
            python_index_url: str | None = None
            python_index_credentials: str | None = None
            job_registry: str | None = None
            job_builder: JobBuilderBinding = JobBuilderBinding()

        def resolve_job_builder(
            config: LocalExecutionConfig,
            *,
            cli_override: Literal["local", "remote"] | None,
            environ: Mapping[str, str] | None = None,
        ) -> ResolvedJobBuilder: ...

`ResolvedJobBuilder` records effective mode and `SettingSource` so JSON plans
can explain whether the CLI or machine configuration selected it. Remote mode
requires a valid HTTPS endpoint plus one resolved named credential or
`POSTTRAIN_JOB_BUILDER_TOKEN`; local mode neither resolves nor validates remote
credentials. `POSTTRAIN_JOB_BUILDER_URL` fills only a missing endpoint and does
not select mode.

`SealedJobContext.context_digest` hashes the sorted canonical blob entries,
not a tar stream or local path. `package_key` remains the authority for job
meaning; context digest proves transport bytes. The remote adapter must still
return the existing `PublishedJobImage`, with the transfer receipt retained as
additional evidence rather than creating a competing image contract.

In `apps/job-builder/src/posttrain_job_builder/store.py`:

        class JobContextStore(Protocol):
            def missing(self, context: SealedJobContext) -> tuple[str, ...]: ...
            def put(self, *, digest: str, size: int, chunks: Iterable[bytes]) -> None: ...
            def seal(self, context: SealedJobContext) -> Path: ...
            def lease(self, publication_key: str) -> ContextManager[Path]: ...
            def collect(self, policy: RetentionPolicy) -> CollectionReceipt: ...

The `Path` returned by `seal` is a service-private reconstruction and exists
only behind a lease. A future object-store implementation must produce the same
logical behavior and tests; it cannot change the HTTP contract or package key.

In `packages/execution-job-builder`, `RemoteJobImagePublisher` implements
`JobImagePublisher.publish()` and `.resolve()`. Its constructor receives only
endpoint, token provider, timeout/retry policy, local receipt root, and an HTTP
transport protocol for tests. It has no BuildKit dependency. Conversely,
`apps/job-builder` depends on `execution-pack`, `execution-buildkit`, and
`runtime-images`, but not on the CLI, Lab, Observatory, release app, or the
remote client package.

`posttrain-runtime-images` depends on nothing in the workspace.
`posttrain-execution-buildkit` and `posttrain` (the CLI) gain it as a
dependency. The CLI additionally depends on
`posttrain-execution-job-builder`. Add import-linter contracts proving that
`execution-pack` does not import an HTTP implementation, the remote adapter
does not import BuildKit, and no reusable package imports `apps/job-builder`.
Run `uv run lint-imports` after every boundary change.

Reuse rather than reimplement:
`packages/execution-buildkit/src/posttrain_execution_buildkit/builder.py`
already provides `BuildKitRuntimeBuilder`, `RuntimeBuildRequest`,
`RuntimeBuildResult`, `BuildxCli`, and `RemoteImageNotFoundError`. Milestone 5
wires these; it does not write a second builder.

Do not add training, evaluation, serving, or dataset logic to any of this.
`runtime` owns framework-owned image identity only; `job pack` owns actual-job
identity; the service owns only remote publication lifecycle.

Revision note (2026-07-27): created. The plan was materially reshaped during
authoring, before any code was written. The original framing — ship container
definitions as package data so each consumer can build their own images, and
add a label-comparison check to detect drift — was replaced by publishing base
and kind images per release with digests pinned in the wheel. The reason is
recorded in the first Decision Log entry: the distillation drift was a
distribution failure rather than a detection failure, and pinning the images to
the release makes the divergence unrepresentable instead of merely detectable.
Package data remains required, but for the sharper reason discovered during
research and recorded in Surprises: `workspace.lock.txt` contents are a build
input to every job pack via `--constraint`, independent of whether the consumer
ever builds an image.

Revision note (2026-08-14): expanded the plan from image publication into the
complete first-install, partial-registry, selective-change, retry, and final-
promotion lifecycle. This records the invariants behind remote actual-job
recovery, idempotent mirroring, variant-local source identities, and the
remaining generated-lock split so future releases do not rediscover the same
cache boundaries through trial and error.

Revision note (2026-08-14): folded the optional developer job-build service
into the same lifecycle rather than creating a disconnected infrastructure
plan. The service is now defined as a third trust domain with plan-before-upload
deduplication, project-scoped publication, isolated BuildKit execution, durable
single-flight state, explicit transfer receipts, local fallback, and candidate-
to-final release gates. This revision also makes byte reproducibility across
builders a prerequisite instead of assuming that a shared publication key is
enough.

Revision note (2026-08-14): made remote publication an operator-owned machine
setting with an explicit `--builder local|remote` command override and a
documented precedence rule. Expanded context transport from the phrase
"missing blobs" into the complete observable path: local materialization,
canonical safe-file manifest, metadata-only admission, missing-file HTTPS
uploads, sealed read-only reconstruction, LAN-side parent pull/job push, and a
transfer receipt. This preserves the fact that private project bytes originate
on the developer machine while keeping base/kind and produced OCI layers off
the VPN.
