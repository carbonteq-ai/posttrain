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
digest. They do not build them, and they need no registry write access and no
BuildKit builder to run their first job.

Third, when a consumer's configured job-kind image does not match what the
installed framework expects, `posttrain doctor` says so loudly and
`posttrain job run` refuses to submit, naming the expected and the found
identity.

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

## Surprises & Discoveries

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

Not yet started. To be written at the completion of each milestone and
summarized at the end.

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
wheels for one specific job. It is built per job, by the consumer.

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

Note before starting: the working tree currently has uncommitted modifications
under `containers/`, including `posttrain-job-kinds/verl-py313/Dockerfile`,
`posttrain-job-kinds/verl-py313/profile.toml`,
`posttrain-job-kinds/tests/test_verl_release_gate.py`,
`posttrain-job/Dockerfile`, and untracked files including
`posttrain-job-kinds/profiles/online-rl-verl-py313-control.txt` and
`posttrain-job-kinds/verl-py313/release/{pyproject.toml,uv.lock}`. These edits
must be preserved through the move in Milestone 2. Do not revert them and do
not stash them away and forget them.

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

`posttrain-runtime-images` depends on nothing in the workspace.
`posttrain-execution-buildkit` and `posttrain` (the CLI) gain it as a
dependency. No new import-linter contract is required, but run
`uv run lint-imports` to confirm the existing eight still hold.

Reuse rather than reimplement:
`packages/execution-buildkit/src/posttrain_execution_buildkit/builder.py`
already provides `BuildKitRuntimeBuilder`, `RuntimeBuildRequest`,
`RuntimeBuildResult`, `BuildxCli`, and `RemoteImageNotFoundError`. Milestone 5
wires these; it does not write a second builder.

Do not add training, evaluation, serving, or dataset logic to any of this.
`runtime` owns framework-owned image identity only.

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
