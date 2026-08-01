# A public application service and a humane authoring surface

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

Source findings: `docs/dx-improvements/v0.2.5/README.md` findings 10, 12, 18,
plus the code-organization observations recorded alongside them (private CLI
application layer, environment types owned by `eval`, duplicated catalog
family registries, the stalled `WorkPackageHost*` rename). This plan is
self-contained.

## Purpose / Big Picture

Everything a developer does with posttrain today goes through the `posttrain`
CLI, and the CLI is where the application actually lives: project discovery,
catalog composition, planning, packing, provider construction, and
reconciliation are assembled in private modules under
`apps/cli/src/posttrain_cli/` (~4,250 lines). There is no supported Python
path from "open this project" to "plan and submit this job"; a future
controller or a notebook must import private CLI internals. Meanwhile the
authoring surface is harder than its own model requires: adding one catalog
family means editing seven parallel literal lists, environment types live in
`posttrain.eval` even though training and packing depend on them, the
documented `ProjectEntry`/`JobRuntime` names are aliases over the real
`WorkPackageHost*` classes, and continuing from one job's output model to the
next package means hand-copying artifact metadata into a large YAML overlay
entry.

After this plan, `from posttrain.project import Project` is real and the CLI
is a thin renderer over it; catalog families are composed deterministically;
environment types have one domain owner; project extensions register
definitions without constructing a host runtime; and a produced model can be
bound by run-output identity or pinned into the catalog with one command
instead of transcription.

## Progress

- [x] (2026-08-01) Plan authored from the v0.2.5 release-scoped critique.
- [x] (2026-08-01) Cross-plan architecture review completed; application,
      environment ownership, extension, registry, and overlay decisions revised.
- [x] (2026-08-01) Follow-up review added a frozen installed-family lock, loud
      absence handling, and a versioned overlay migration/exclusion contract.
- [ ] Milestone 1: public `posttrain.project` service; CLI delegates
      (in progress: provider-free `Project.open()` / `jobs.plan()` and the
      `work-package plan` CLI seam).
- [x] (2026-08-01) Added installable `posttrain-project` with
      `Project.open()` / `Project.discover()` and a provider-free `JobIntent`;
      migrated `work-package plan` and its `job plan` alias to obtain static
      job meaning through that service. Focused tests, pyright, and import
      boundaries pass.
- [x] (2026-08-01) Moved execution override values, launch/package boundaries,
      and precedence/provenance resolution to `posttrain.project`; the legacy
      CLI configuration module now re-exports those public types while it
      retains machine-binding file parsing.
- [x] (2026-08-01) Moved tracked project source-selection for pack planning to
      `posttrain.project`; the CLI loader is now a tested compatibility
      re-export.
- [ ] Milestone 1 remaining: move execution-setting, pack-plan, and other
      command-family application logic out of `apps/cli` while preserving the
      current public intent seam.
- [ ] Milestone 2: dedicated `posttrain.environment` contracts and
      deterministic catalog-family assembly.
- [x] (2026-08-01) Extracted `posttrain-environment`; evaluation preserves
      one-release re-exports while catalog, jobs, packing, work composition,
      and the runtime worker depend on the shared domain directly. Catalog
      family registry composition remains the next part of this milestone.
- [x] (2026-08-01) Replaced core catalog-family literal lists with an explicit
      `FamilyRegistry`: core descriptors plus installed
      `posttrain.catalog_families` entry points compose in stable order,
      duplicate origins fail loudly, unavailable source families fail before
      decoding, and every opened catalog retains its `FamilyRegistryLock`.
      Package-identity and project-declared plugin requirements remain.
- [x] (2026-08-01) Bound the full family lock into the semantic package plan
      and digest-protected worker config. A project may declare required plugin
      distributions in `[catalog_plugins]`; CLI, `doctor`, public project
      opening, and the worker all reject an absent provider before decoding.
      Staging reachable provider distributions into the image remains.
- [ ] Milestone 3: replace the host-factory hook with a public registration
      extension API and compatibility adapter.
- [ ] Milestone 4: deterministic overlay discovery, `catalog explain`,
      run-output bindings, and `artifact pin`.
- [ ] Milestone 5: JSON Schema export, recipe-ref starter templates, and
      authoring generators.

## Surprises & Discoveries

- Observation: moving environment contracts into `common` would remove one
  dependency edge by turning the framework-neutral package into a domain
  dumping ground.
  Evidence: environment activations and resources are consumed broadly but
  remain a coherent domain with their own schema and compatibility lifecycle.
- Observation: one mutable registry populated by imports is still several
  sources of truth at runtime.
  Evidence: the in-progress `remote-evaluation` family currently requires
  edits across multiple literal lists; deterministic entry-point composition
  removes those edits without making import order part of catalog identity.
- Observation: the existing image-plan adapter needs local execution
  configuration even though static job meaning does not.
  Evidence: the first adapter extraction raised `NameError: local_config is
  not defined`; restoring it inside the adapter kept `JobIntent` free of
  provider, registry, and credential state while the CLI regression suite
  passed.
- Observation: an environment contract can move without changing evaluation
  semantics when compatibility imports retain object identity.
  Evidence: direct `posttrain.environment` imports and the legacy
  `posttrain.eval` re-exports resolve to the same binding and schema objects;
  the catalog, jobs, packer, work runner, and runtime worker tests all pass.
- Observation: type-level literal families had leaked into CLI argument
  generation, so replacing the registry alone left the application unable to
  start when the family annotation became extensible.
  Evidence: Typer rejected the widened alias as an unsupported parameter type;
  catalog commands now accept strings and validate them against the opened
  registry instead.
- Observation: recording a registry lock only in a host-side plan would still
  let an image rediscover a different installed set on the worker.
  Evidence: the lock is now in `JobPackSpec` (therefore plan identity), in the
  digest-protected resolved config, and compared to the worker's opened
  registry before the job is prepared.

## Decision Log

- Decision: `posttrain.project` is a new package (`packages/project`)
  depending on `catalog`, `work`, `jobs`, `execution`, `execution-pack`, and
  `tracking` contracts — not a module inside `apps/cli`.
  Rationale: the CLI, the controller, and Python automation must share one
  service; anything under `apps/` reads as private.
  Date/Author: 2026-08-01 / plan author.
- Decision: environment contracts move to a dedicated
  `posttrain.environment` package, not to `posttrain.common`.
  Rationale: environments are shared by eval, train, jobs, and packing, but
  their activation/resource schemas are domain contracts. `common` must remain
  framework-neutral and free of Verifiers-specific growth.
  Date/Author: 2026-08-01 / architecture review (supersedes the original plan
  decision).
- Decision: catalog families are assembled explicitly from core descriptors and
  sorted package entry points; registration never depends on import order.
  Rationale: import-time mutation makes validation and pack identity depend on
  which optional package happened to import first.
  Date/Author: 2026-08-01 / architecture review.
- Decision: the complete discovered family registry is a frozen resolution
  input, not an unrecorded property of the current Python environment.
  Rationale: deterministic ordering does not make two different installed sets
  equivalent. The project snapshot and package identity must bind the exact
  family names, schema revisions, entry points, distributions, and versions
  used during resolution; runtime consumes that lock instead of rediscovering.
  Date/Author: 2026-08-01 / architecture review follow-up.
- Decision: replace `configure(request) -> JobRuntime` with a registration
  extension instead of merely renaming the old host factory.
  Rationale: project extensions should declare definitions; the framework host
  should construct and validate execution runtime so policy is not inverted or
  privately duplicated.
  Date/Author: 2026-08-01 / architecture review.
- Decision: begin the CLI migration with a provider-free `JobIntent` rather
  than exposing image registry, credentials, or worker bindings through
  `Project.jobs.plan`.
  Rationale: project opening and static job meaning are reusable by a
  controller or notebook; image materialization and provider submission remain
  separate phases to migrate within this milestone.
  Date/Author: 2026-08-01 / implementation.
- Decision: retain `posttrain_cli.execution_config` imports as compatibility
  re-exports while moving the underlying execution-setting types and resolver.
  Rationale: command modules and external callers can migrate at their own
  pace, while the sole implementation of precedence is already public.
  Date/Author: 2026-08-01 / implementation.

## Outcomes & Retrospective

- Planning review outcome: the application service, environment domain,
  extension API, and catalog discovery now have separate owners. Implementation
  outcomes remain pending.
- First implementation outcome: `posttrain.project` now opens a project and
  statically plans one enabled job without a CLI state object, image registry,
  provider, or credentials. The current CLI routes its plan command through
  that service and keeps OCI-specific planning as a compatibility adapter.
- The execution-setting resolver is now shared by public Python callers and
  CLI compatibility imports; 67 focused tests, pyright, and import-boundary
  checks passed after the move.
- Milestone-2 partial outcome: the portable environment contract now has one
  owner (`posttrain-environment`) rather than being structurally coupled to
  evaluation. The catalog-family registry remains to be made explicit and
  frozen before the milestone is complete.
- Family-registry outcome: catalog composition no longer depends on which
  optional package imports first. The resolved catalog exposes a stable,
  serializable lock recording every installed family contributor; the next
  packing slice must bind that lock into package identity and runtime replay.
- Registry-lock outcome: package identity and worker replay now bind the
  complete discovered family set, including unrelated installed providers.
  Project-declared provider requirements produce `catalog_family_unavailable`
  with the missing distribution and installed family providers. Image staging
  of the selected provider distributions remains the final Milestone-2 gap.

## Context and Orientation

Key facts a novice needs:

- The real application logic sits in `apps/cli/src/posttrain_cli/`:
  `execution_planning.py` (806 lines), `execution_config.py` (1,060),
  `execution_provider.py`, `work_runtime.py`, `tracking_config.py`,
  `pack_config.py`, `run_resolve.py`, `overlay_write.py`.
- Public composition lives in `packages/work/src/posttrain/work/` — but its
  documented names are aliases: `ProjectEntry = WorkPackageHostFactory`,
  `JobRuntime = WorkPackageContext`, `ProjectExecutionRequest =
  WorkPackageHostRequest` (see `packages/work/src/posttrain/work/__init__.py`).
  The runner, the lab app, and the CLI still speak the old names.
- The project entry hook is documented as `configure(runtime)` but is really
  `configure(request: ProjectExecutionRequest) -> JobRuntime` — it *builds*
  the runtime (compare `apps/lab/src/posttrain_lab/entry.py`). The policy
  that loads and validates an entry (`load_project_entry`,
  `validate_standard_definitions`) is private in
  `apps/cli/src/posttrain_cli/work_runtime.py`.
- Catalog families are duplicated literal lists. Adding the in-flight
  `remote-evaluation` family touched `packages/common/src/posttrain/common/
  selections.py`, `.../common/catalog.py`, `packages/eval/src/posttrain/eval/
  catalog_schema.py`, `packages/catalog/src/posttrain/catalog/files.py`,
  `packages/work/src/posttrain/work/contracts.py`, `packages/jobs/...`, and
  `apps/cli/src/posttrain_cli/constants.py`.
- `EnvironmentSource`/`EnvironmentBinding` are defined in
  `packages/eval/src/posttrain/eval/requests.py`; `packages/train` maintains
  a structural mirror (`EnvironmentSourceSelection` Protocol in
  `src/posttrain/train/integrations/verifiers.py`) to avoid importing eval,
  while `packages/jobs` and `packages/execution-pack` import `posttrain.eval`
  directly.
- Continuation is manual: after SFT produces a model, the developer copies
  the Trackio artifact `vN` from a UI and hand-writes a full `ModelVariant`
  overlay entry plus a `layer.yaml` line (`docs/consumer-setup.md` §9).
- Overlay membership: a catalog YAML file not listed in the overlay's
  `layer.yaml` is silently ignored (a `doctor` WARN at best).

## Plan of Work

Milestone 1 creates `packages/project` and first migrates project opening,
configuration resolution, catalog composition, and provider-free job planning.
It establishes the application boundary needed by the configuration and
lifecycle plans before moving every CLI command. The final public journey is:

    from posttrain.project import Project

    project = Project.open(".")
    intent = project.jobs.plan("sft")
    run = project.jobs.run(intent, provider="dstack")
    for update in run.watch():
        ...
    run.cancel()

Move the logic (not the Typer wiring) of `execution_planning.py`,
`execution_provider.py`, `work_runtime.py`, and the config loaders into it,
leaving `apps/cli` as argument parsing plus rendering. Do this incrementally:
one command family at a time (`job plan` first), with the CLI calling the
service and its tests asserting the CLI path and service path produce identical
results. `Project.open()` and `jobs.plan()` are read-only. The convenience
`jobs.run()` composes materialize, publish, and submit while retaining those
phases as separately callable APIs.

Milestone 2 creates `packages/environment` (distribution
`posttrain-environment`) to own `EnvironmentSource`, `EnvironmentBinding`,
activation/resource types, and their schemas. `posttrain.eval` re-exports the
old names for one release; train deletes its structural mirror; jobs and
execution-pack import the new owner. In parallel, define immutable
`CatalogFamilyDescriptor` values. `posttrain.project` builds a `FamilyRegistry`
explicitly from built-in descriptors plus installed
`posttrain.catalog_families` entry points sorted by distribution and entry-point
name. Duplicate family names are hard errors with provenance. No module mutates
a global registry during import. A resolved plan records the contributing
family name, schema identity/revision, entry-point name, distribution name, and
installed version for every core and extension descriptor in a sorted
`FamilyRegistryLock`. The project/run snapshot and package identity include the
whole lock. The packer includes distributions for families reachable from the
selected job; runtime consumes the frozen resolved catalog and lock rather than
composing a new registry from the worker's installed extras.

If project files reference a family absent from the registry, planning fails
before catalog decoding with `catalog_family_unavailable`, naming the family,
the installed family set, and—when the project declares it—the missing plugin
distribution or extra. Add an optional tracked requirement block to
`.posttrain/project.toml` so a project can state required family providers and
`doctor` can diagnose them before planning. Absence must never look like an
empty family or silently ignored YAML.

    [catalog_plugins]
    required = ["acme-posttrain-catalog>=1,<2"]

Milestone 3 replaces the misleading entry hook. Define a public
`ProjectPlugin.register(registry: ProjectRegistry) -> None` extension that may
register job definitions and catalog-family descriptors. The host-owned
application service constructs `JobRuntime`, enforces standard-definition
protection, and resolves an execution request. Adapt legacy
`WorkPackageHostFactory`/`ProjectEntry` hooks behind a one-release compatibility
adapter with a targeted warning; do not make the old factory semantics the new
canonical names. `init` scaffolds an optional worked plugin example, and the
docs describe registration rather than the inverted `configure` signature.

Milestone 4 adds the lineage-continuation tools. A run-output binding in
work-package YAML:

    bindings:
      model:
        type: run-output
        run_id: train.sft-01J...
        role: trained_model

resolves at planning to one immutable artifact version. `posttrain artifact
pin --run RUN --role trained_model --as models/my-agent@sft-v3` writes the
catalog overlay entry by deriving base model, renderer, form, digest, and
producer run from retained evidence, and appends the file to `layer.yaml`
itself when operating on a legacy explicit-file layer. For new layers, all YAML
files below the overlay root (except `layer.yaml`) are discovered in sorted
path order; duplicate ids are errors and provenance records the source path.
`layer.yaml` remains the authority for layer id, revision, precedence, and
provenance. Version-1 manifests retain their explicit file list unchanged.
Version-2 manifests add deterministic discovery:

    schema_version: 2
    id: memory-agent
    revision: 4
    discovery:
      include: ["**/*.yaml"]
      exclude: ["drafts/**", "generated/**"]

`layer.yaml` itself is always excluded. Includes and excludes are evaluated
relative to the layer root and sorted before decoding. The resolved snapshot
records layer id/revision, manifest digest, included file paths/digests, excluded
patterns, and entry provenance. Add `posttrain catalog migrate-layer PATH
--check`: its first migration writes an exact include list equivalent to the
version-1 membership, preserving behavior; opting into the broad glob is a
separate reviewed edit. `posttrain catalog explain FAMILY/ID` reports origin
file, layer, revision, links, and known consumers. Do not add `catalog include`
as a permanent command merely to preserve an accidental loader mechanism.

Milestone 5 rounds out authoring ergonomics: `posttrain schema export`
writes JSON Schemas for work packages, catalog overlay files, and
`layer.yaml` (enabling editor completion); starter templates switch from
inline recipes to `recipe: {type: ref, ...}` so the seat map is stated once;
and `posttrain work-package new --stage STAGE --recipe ID` plus
`posttrain catalog add FAMILY` generate skeletons instead of asking the
developer to transcribe shapes from documentation.

## Concrete Steps

Work from the repository root.

    uv run pytest packages/work/tests packages/jobs/tests packages/common/tests -q
    uv run pytest apps/cli/tests -q

Milestone-1 proof: a test that opens the fixture project through
`Project.open`, plans the starter job, and asserts the intent equals the one
produced through the CLI code path. Milestone-2 proof: install a tiny fixture
distribution exposing a `posttrain.catalog_families` entry point and show its
family appears in catalog validation and work-package binding checks without
editing a framework literal list; prove duplicate entry points fail with both
origins named. Run the same fixture with that distribution absent and prove a
referencing project fails with `catalog_family_unavailable`, rather than
producing an empty family. Install an unrelated fixture family and prove the
`FamilyRegistryLock` and package key change even though selected bindings do
not. Also retain grep evidence that the parallel literal lists are gone:

    grep -rn "remote-evaluation" --include="*.py" packages apps/cli/src
    # expected: the registry definition, schema registration, and tests only

Milestone-4 proof: run the SFT starter to completion against the fake
tracking backend, then `artifact pin` its output and `job plan` a consumer
package binding the pinned id — no hand-written YAML in between. Exercise
`catalog migrate-layer --check` on a version-1 fixture and prove the migrated
version-2 exact include list resolves the same entries, layer id/revision, and
provenance. Then add `drafts/**` to the exclusion list, place a conflicting
draft entry there, and prove it is excluded while an equivalent conflict in an
included path fails loudly.

## Validation and Acceptance

- The critique's Python journey runs verbatim from an installed wheel:
  `Project.open(".")`, `jobs.plan`, `jobs.run`, `run.watch()`,
  `run.cancel()`.
- `apps/cli` contains no planning, provider, or config-resolution logic —
  measured as: every module in `posttrain_cli` either parses arguments,
  renders output, or delegates to `posttrain.project`.
- Adding a catalog family is demonstrably one descriptor plus one package entry
  point; discovery order does not change the composed registry. The full
  discovered registry is present in the resolved snapshot and package identity;
  a referenced missing family fails with `catalog_family_unavailable`.
- A produced model reaches the next work package via `run-output` binding or
  one `artifact pin` command, with zero copied metadata.
- A new-style overlay discovers tracked YAML deterministically without a
  second membership edit; duplicate ids fail with both source paths. Legacy
  explicit-file layers continue to validate unchanged. Migration preserves
  layer id, revision, membership, and snapshot provenance, while version-2
  include/exclude patterns make draft and generated paths explicit.
- Work-package and catalog YAML get completion in an editor configured with
  the exported schemas.

## Idempotence and Recovery

Every rename and move keeps a deprecated re-export for one release, so
external projects upgrade on their own schedule; deprecation warnings name
the new import path. The CLI-to-service migration is done command family by
command family with both paths tested equal before the old path is deleted —
never a big-bang swap. Generators (`work-package new`, `catalog add`,
`artifact pin`) refuse to overwrite existing files unless `--force` is given.
`catalog migrate-layer` writes to a temporary file, validates equivalent
resolution, and atomically replaces the manifest; rerunning it is a no-op.
Version-1 manifests remain readable for the full deprecation window.

## Artifacts and Notes

Keep the milestone-4 transcript (train → pin → consume) here as indented
evidence once it exists.

## Interfaces and Dependencies

New package `packages/project`, distribution name `posttrain-project`,
depending on `posttrain-common`, `posttrain-catalog`, `posttrain-work`,
`posttrain-jobs`, `posttrain-execution`, `posttrain-execution-pack`, and
tracking contracts. Core surface:

    class Project:
        @classmethod
        def open(cls, root: str | Path) -> "Project": ...
        @property
        def jobs(self) -> JobService: ...        # plan / pack / submit
        @property
        def runs(self) -> RunService: ...        # list / view / cancel / reconcile
        @property
        def catalog(self) -> Catalog: ...        # composed, resolve-only

    class JobService:
        def plan(self, job: str, *, work_package: str | None = None) -> JobIntent: ...
        def materialize(self, intent: JobIntent, *, frozen: bool = False) -> MaterializedJobPackSpec: ...
        def publish(self, spec: MaterializedJobPackSpec, *, destination: str) -> PublishedJob: ...
        def submit(self, published: PublishedJob, *, provider: str) -> RunHandle: ...
        def run(self, intent: JobIntent, *, provider: str) -> RunHandle: ...

In `packages/catalog`, define:

    @dataclass(frozen=True)
    class CatalogFamilyDescriptor:
        name: str
        schema_loader: Callable[[], type[BaseModel]]
        origin: str

    class FamilyRegistry:
        @classmethod
        def compose(cls, core: Iterable[CatalogFamilyDescriptor], entry_points: Iterable[EntryPoint]) -> "FamilyRegistry": ...

        def lock(self) -> FamilyRegistryLock: ...

`FamilyRegistry.compose` is explicit, deterministic, and rejects duplicate
names. It does not inspect already-imported modules. Environment contracts and
their schema loader live in `packages/environment`; `posttrain.common` retains
only framework-neutral selection identities and references.

`FamilyRegistryLock` is a sorted, serializable tuple of descriptor records with
family name, schema identity/revision, core-or-extension origin, distribution,
distribution version, and entry-point name. It participates in project/run
snapshots and package identity.

## Revision Notes

- 2026-08-01: Architecture review made provider-free planning explicit,
  introduced a dedicated environment-contract package, replaced import-time
  family mutation with deterministic entry-point composition, replaced the
  inverted runtime-building hook with declarative plugin registration, and
  chose deterministic overlay discovery instead of a permanent `catalog
  include` command.
- 2026-08-01: Follow-up review froze the complete installed family set into a
  `FamilyRegistryLock`, added loud missing-family behavior, and specified a
  behavior-preserving `layer.yaml` v1-to-v2 migration with explicit include /
  exclude patterns and retained provenance.
