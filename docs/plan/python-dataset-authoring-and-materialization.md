# Implement Python dataset authoring and reproducible materialization

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

Maintain this document in accordance with `docs/templates/PLAN.md`. A reader should be able to execute the plan from a checkout of this repository without access to prior conversations or unpublished context.

## Purpose / Big Picture

Posttrain projects can currently describe simple datasets in YAML, but a dataset that needs custom processing is represented as a YAML entry pointing at an arbitrary Python file. The file is executed with `runpy`, and the serving package maintains a separate corpus generator, manifest format, console script, and `--check` workflow. This divides one developer task across unrelated mechanisms.

After this plan is complete, a developer can register dataset selections in YAML or typed Python through the same catalog, materialize custom Python-built datasets through a framework-owned API, and verify the exact output without maintaining a standalone script. Catalog validation remains free of data downloads and builder execution. Job packing records builder code, declared inputs, canonical output, and content digest as part of the immutable package identity.

The visible proof is a project fixture containing one YAML dataset and one Python-authored dataset. `posttrain catalog list --family dataset` shows both, `posttrain dataset materialize` builds the Python selection, a second invocation reuses its cache, and `posttrain dataset verify` detects changed output without writing tracked files. The same generic machinery must reproduce the existing `general-serving-v1` corpus exactly before the package-specific generator is removed.

## Progress

- [x] (2026-08-02 11:00Z) Reviewed the canonical dataset, catalog, workload, package, API, and detached-planning contracts; inspected the current YAML loader, `DatasetLoadPlan`, `python-file` builder, CLI, execution packager, and serving corpus generator.
- [x] (2026-08-02 11:00Z) Wrote the proposed human-facing architecture in `docs/post-training/dataset-management.md` and resolved the serving corpus as a workload-owned record collection rather than a training dataset seat.
- [x] (2026-08-02) Amended the frozen framework and API baseline narrowly for explicit Python catalog providers, typed builder references, materialize/verify commands, and package layout conventions.
- [x] (2026-08-02) Implemented catalog layer schema version 2 and pure Python catalog providers while retaining schema version 1 compatibility.
- [x] (2026-08-02) Replaced new arbitrary in-process Python-file execution with typed builder references, isolated materialization, complete build keys, and versioned manifests; the old form remains a warned compatibility path.
- [x] (2026-08-02) Integrated Python-built selections with job planning, immutable execution packing, runtime resolution, and the primary CLI.
- [x] (2026-08-02) Migrated `general-serving-v1` to the owning package convention with byte-for-byte parity and generic workload commands.
- [x] (2026-08-02) Removed the standalone serving materializer entry point, reconciled overlapping plans, and ran the focused validation ladder plus the full repository suite.

## Surprises & Discoveries

- Observation: Typed Python authoring is already part of the canonical API stance, but project catalog layers currently load only manifest-listed YAML files.
  Evidence: `docs/post-training/05-apis.md` says YAML and typed Python validate into the same models; `packages/catalog/src/posttrain/catalog/files.py` restricts layer files to `.yaml` and `.yml`.

- Observation: Custom dataset processing already exists, but it is a stringly typed escape hatch rather than a package API.
  Evidence: `packages/data/src/posttrain/data/catalog.py` accepts `source.kind: built`, executes a project-relative file with `runpy.run_path`, invokes a named value with no context, and fingerprints only the builder file plus manually declared input files.

- Observation: The serving corpus is data-like to build but is not a training dataset selection.
  Evidence: `docs/post-training/02-primitives.md` limits public dataset seats to supervised and preference training while the `Workload` section owns a versioned prompt-corpus identity. `packages/catalog/src/posttrain/catalog/base/workloads.yaml` binds `general-serving-v1` inside the representative serving workload.

- Observation: Two active plans currently assign incompatible long-term owners to the same builder behavior.
  Evidence: `docs/plan/dx-packing-environments-datasets.md` milestone 7 retains `python-file` builders in an isolated process, while `docs/plan/dx-repository-and-qualification-cleanup.md` treats the package-specific serving corpus console script as its final owner. This plan supersedes those two narrow decisions after parity is implemented; their unrelated milestones remain intact.

- Observation: The current `posttrain dataset validate` command already materializes data, so its name hides a side effect.
  Evidence: `apps/cli/src/posttrain_cli/commands/dataset.py` calls `materialize_dataset` and reports either `Materialized` or `Validated cached`.

## Decision Log

- Decision: Keep one composed catalog and add explicit YAML and Python sources to a catalog layer.
  Rationale: Developers should not need to choose between registries, and every resolved selection must retain the same overlay and source-layer provenance.
  Date/Author: 2026-08-02 / Posttrain maintainers

- Decision: Python catalog providers return complete typed entries and perform no materialization.
  Rationale: Catalog validation and detached planning must remain lightweight, deterministic, and safe. Field-level merging across YAML and Python would make resolved values difficult to explain.
  Date/Author: 2026-08-02 / Posttrain maintainers

- Decision: Keep generic record build and manifest support in `posttrain.data` rather than creating a new public catalog family or a global asset registry.
  Rationale: `posttrain.data` already owns source adaptation and materialization and has a lightweight default dependency set. `posttrain.serve` can reuse that lower layer without importing training semantics, while a prompt corpus remains owned by its workload.
  Date/Author: 2026-08-02 / Posttrain maintainers

- Decision: Represent builders as importable module-level references and execute them only during explicit materialization in a child process.
  Rationale: An import reference can be locked, packaged, and activated in the selected project environment. Lambdas, closures, notebook cells, and arbitrary files cannot be reproduced reliably.
  Date/Author: 2026-08-02 / Posttrain maintainers

- Decision: Track selection identity, build key, and output content digest separately.
  Rationale: A catalog revision describes meaning, a build key determines cache reuse, and a content digest identifies the exact bytes consumed. Conflating them permits code or input drift to hide behind an unchanged version string.
  Date/Author: 2026-08-02 / Posttrain maintainers

- Decision: Use domain-facing CLI commands over one materialization service.
  Rationale: Developers should use `posttrain dataset materialize` and `posttrain workload verify`, while capability packages expose importable APIs. A permanent package-specific console script is unnecessary.
  Date/Author: 2026-08-02 / Posttrain maintainers

- Decision: Preserve catalog schema version 1, `python-file`, and `dataset validate` for one compatibility window.
  Rationale: Additive migration allows current projects and in-progress packing work to remain usable until the new path has catalog, cache, pack, and corpus parity coverage.
  Date/Author: 2026-08-02 / Posttrain maintainers

## Outcomes & Retrospective

Delivered behavior includes explicit YAML/Python catalog sources, typed child-process
dataset builders, content-addressed manifests, immutable package-lock fields,
`posttrain dataset materialize|verify`, and workload-owned serving corpus
materialization. Schema version 1 and `python-file` remain warning-emitting
compatibility paths; `dataset validate` remains a replacement-emitting alias
until a release migration decision removes them. The focused suites and the
full repository suite pass: 986 passed and 18 skipped. The serving wheel was
built and checked to contain the relocated corpus resources without the old
materializer module.

Packing verification: a typed builder now materializes into the actual-job
context under `datasets/<seat>-<identity>/<content-sha256>/`, and its lock carries
`build_key`, `materializer_schema_version`, `builder_target`,
`code_snapshot_digest`, `dependency_lock_digest`, and the output digest. The
staged manifest is checked against those values before the package is retained.
The execution-pack, execution, and jobs suites pass with 161 tests.

## Context and Orientation

This repository is a Python 3.13 `uv` workspace. `packages/common` owns framework-neutral selection and catalog values. `packages/catalog` loads the framework base catalog and project overlays. `packages/data` owns canonical supervised and preference data plus source adapters and local materialization. `packages/execution-pack` materializes selected datasets into immutable actual-job images. `packages/jobs` resolves dataset seats for standard jobs. `apps/cli` distributes the `posttrain` command. `packages/serve` owns serving workloads and the packaged representative prompt corpus.

A catalog layer is one named group of entries. The framework base layer is packaged in `posttrain-catalog`; project overlays live under `.posttrain/catalog/`. A `CatalogRef` contains a family and ID. Resolving it returns a typed selection plus the layer that supplied it. An overlay may replace a base entry with the same ID, but duplicate IDs inside one layer are invalid.

A build key is a SHA-256 digest over all facts that can change a materialization: the normalized selection, declared input identities, builder reference, project or distribution code snapshot, dependency lock, and materializer schema. A content digest is SHA-256 over the canonical output bytes. The build key chooses a cache entry; the content digest identifies the data actually consumed.

The current catalog loader in `packages/catalog/src/posttrain/catalog/files.py` reads `layer.yaml` with `schema_version: 1` and a list of YAML filenames. `packages/catalog/src/posttrain/catalog/__init__.py` converts those mappings into `CatalogLayer` values using family decoders. `packages/common/src/posttrain/common/catalog.py` already accepts a typed `CatalogLayer`, so Python providers can converge before normal catalog composition without changing catalog lookup.

The current dataset implementation in `packages/data/src/posttrain/data/catalog.py` combines source schemas, decoding, fetching, canonicalization, caching, and builder execution. `DatasetLoadPlan` stores an untyped source mapping. A built source names a project-relative Python file and callable; `_source_rows` executes it in the CLI process with `runpy`. `_plan_json` adds SHA-256 values for the builder and manually declared file inputs, but it does not identify imported helper code, the project package snapshot, the dependency lock, external source licenses, or a typed builder contract.

The current immutable packager in `packages/execution-pack/src/posttrain/execution_pack/datasets.py` calls `materialize_dataset`, verifies the manifest, copies the canonical file under `datasets/`, and records a `DatasetPackageLock`. Planning discovers dataset seats in `packages/execution-pack/src/posttrain/execution_pack/planning.py`. Standard job resolution occurs in `packages/jobs/src/posttrain/jobs/runtime.py` and `packages/jobs/src/posttrain/jobs/definitions.py`. These paths must consume the new selection and manifest without inventing a second build path.

The serving corpus generator currently lives in `packages/serve/src/posttrain/serve/benchmarks/materialize_corpus.py`, is exposed as `posttrain-serve-materialize-corpus` by `packages/serve/pyproject.toml`, and writes package resources below `packages/serve/src/posttrain/serve/benchmarks/resources/corpora/`. `packages/serve/src/posttrain/serve/prompts.py` loads and verifies those resources. The migration must retain the current 128 records, digest `9a9467fd8a5e744968d09a4d8fd6f4d92a089c50a84e1e6e7e5c5520a9f4e50e`, source revisions, categories, licenses, and model-visible fields.

The canonical product meaning does not change: dataset selections remain public training inputs, while serving prompt populations remain part of workloads. The implementation does change the documented package and API surface, so the frozen framework and API documents must be narrowly amended before code claims the new behavior.

## Plan of Work

### Milestone 1: Amend the public dataset authoring contract

Begin by updating the product documentation so implementation has one explicit contract. Add an amendment to `docs/post-training/README.md` linking `docs/post-training/dataset-management.md` and this plan. Update the package table and catalog DX in `docs/post-training/04-framework.md` to state that project layers may contain YAML documents and explicit pure Python providers, and add the package layout convention for dataset definitions and builders. Update `docs/post-training/05-apis.md` with catalog layer schema version 2, the typed dataset source and builder reference, the new materialize and verify commands, and the compatibility spelling of `dataset validate`.

Do not change `docs/post-training/02-primitives.md`: a dataset selection still means an exact semantic training input, and a prompt corpus still belongs to a workload. The amendment only makes the already-stated dual authoring approach executable and gives custom processing a reproducible lifecycle.

Update `docs/contributing.md` and the dataset section of `docs/developer-experience.md` so a new contributor sees the source layout and commands without reading an implementation plan. Retain historical evidence in older plans but add revision notes to `docs/plan/dx-packing-environments-datasets.md` and `docs/plan/dx-repository-and-qualification-cleanup.md` stating that this plan owns Python builder and serving-corpus migration. Do not modify their unrelated milestones.

This milestone is accepted when all public examples agree on the same nouns, commands, layer manifest, package layout, and serving-corpus boundary, and `rg` finds no current guide that recommends creating a new root script for a dataset builder.

### Milestone 2: Add pure Python sources to catalog layers

Change `packages/catalog/src/posttrain/catalog/files.py` so `CatalogLayerManifestSchema` accepts both the existing schema version 1 `files` form and a schema version 2 ordered `sources` form. Define strict source models for `kind: yaml` with a local filename and `kind: python` with an importable `module:callable` provider. Reject absolute paths, path traversal, duplicate YAML paths, duplicate provider references, unknown fields, and an empty provider target.

Add `packages/catalog/src/posttrain/catalog/providers.py`. Define a frozen `CatalogEntries` value that holds complete typed selections indexed by `CatalogRef`, plus a `load_python_catalog_provider(reference: str) -> CatalogEntries` function. The loader imports the target, invokes it without arguments, validates its return type, and wraps import or invocation failures with the layer and provider reference. The provider API must not accept a mutable catalog or global registry.

Refactor `load_catalog_layer` to return a typed `CatalogLayer` rather than an untyped mapping once all sources have been decoded, or add a new `load_catalog_layer_entries` boundary and retain the mapping function as a compatibility wrapper. YAML entries continue through the installed family decoders. Python entries are already typed selections. Detect duplicates across all YAML and Python sources before composing the layer with the base catalog. Preserve source ordering for diagnostics, but do not allow later sources in the same layer to overwrite earlier entries.

Update `packages/catalog/src/posttrain/catalog/__init__.py` and `packages/catalog/README.md` with the public provider types. Add `packages/catalog/tests/test_providers.py` and extend `packages/catalog/tests/test_files.py` and `packages/catalog/tests/test_project.py`. Tests must prove that YAML-only schema version 1 remains valid, a schema version 2 layer can mix YAML and Python, both entries resolve through one catalog, duplicates fail, and importing a provider does not import its builder target. Use a fixture provider whose builder module raises if imported to prove registration is inert.

Do not add Python entry-point discovery. The only Python providers loaded are those explicitly listed by the selected layer manifest or supplied as an explicit typed `CatalogLayer` by a direct library caller.

Run from `/home/hammad/projects/rl`:

    uv run pytest packages/catalog/tests -q
    uv run ruff check packages/catalog
    uv run pyright packages/catalog

Acceptance is a fixture project where `posttrain catalog list --family dataset` reports one YAML entry and one Python entry with the same overlay ID, while a counter or sentinel in the builder module proves the builder was not imported.

### Milestone 3: Introduce typed build plans and isolated materialization

Split the responsibilities currently concentrated in `packages/data/src/posttrain/data/catalog.py`. Add `packages/data/src/posttrain/data/definitions.py` for frozen dataset selection, source, input, and builder-reference values. Add `packages/data/src/posttrain/data/materialization.py` for build keys, cache paths, canonical output, manifests, and verification. Add `packages/data/src/posttrain/data/builder_runner.py` as the internal child-process entry point. Keep catalog decoding and compatibility conversion in `packages/data/src/posttrain/data/catalog.py`.

The target public values are:

    @dataclass(frozen=True, slots=True)
    class PythonDatasetBuilder:
        target: str

    @dataclass(frozen=True, slots=True)
    class BuiltDatasetSource:
        builder: PythonDatasetBuilder
        inputs: Mapping[str, DatasetBuildInput]
        expected_content_sha256: str | None = None

    @dataclass(frozen=True, slots=True)
    class DatasetSelection:
        id: str
        revision: str
        kind: Literal["supervised", "preference"]
        split: str
        schema_version: str
        provenance: DatasetProvenance
        access: DatasetAccessPolicy
        source: DatasetSource
        format: str

    @dataclass(frozen=True, slots=True)
    class DatasetBuildContext:
        inputs: Mapping[str, ResolvedDatasetBuildInput]
        workspace: Path

    @dataclass(frozen=True, slots=True)
    class DatasetMaterialization:
        selection_id: str
        selection_revision: str
        build_key: str
        content_sha256: str
        path: Path
        manifest_path: Path
        examples: int
        created: bool

Use exact final field names consistently across code and docs; if existing public compatibility makes `DatasetLoadPlan` the better retained name, record that decision here and keep `DatasetSelection` as a documented alias. Do not leave two independent models in active use.

The builder runner accepts a serialized build request path and result path, imports exactly one `module:callable`, constructs a `DatasetBuildContext`, and writes either a structured success result or a structured failure. It rejects non-module-level targets and non-iterable or non-mapping records. The parent process owns source resolution, timeout, cancellation, temporary directories, canonical serialization, record validation, and atomic cache promotion. Builder stdout and stderr are captured and included in actionable failures without being treated as the data result.

Compute the build key from canonical JSON containing the selection, every resolved input identity, the builder target, the selected project or distribution code snapshot digest, the applicable lockfile digest, and a materializer schema version. Reuse the source snapshot rules in `packages/execution-pack/src/posttrain/execution_pack/source_snapshot.py`; do not invent a second interpretation of `[tool.posttrain.pack]`. For a direct local materialization outside job packing, create a small shared source-identity service or pass a typed code identity into the data API from the project runtime.

Write a versioned manifest with selection identity, kind, schema, source provenance, input digests and licenses, builder and code identity, transformation settings, canonical format, record count, byte size, build key, and content digest. Write to a temporary sibling directory, fsync or close all files, validate by reading them back, and rename atomically into the cache. A failed or cancelled build leaves no completed cache marker.

Retain simple Hugging Face, JSONL, Parquet, NeMo, and packaged fixture behavior. Translate the old `python-file` source into a deprecated compatibility plan and emit a focused warning. Do not remove `runpy` until a compatibility test proves that existing entries receive the warning and still materialize during the declared window. New tests must never use the old form except for this compatibility case.

Extend `packages/data/tests/test_catalog.py` and add `packages/data/tests/test_materialization.py` and builder fixtures below `packages/data/tests/fixtures/`. Cover deterministic output, cache reuse, rebuild after input or code changes, expected digest mismatch, undeclared input access, invalid rows, builder import failure, child-process failure, interrupted build cleanup, manifest round-trip, and unchanged static sources.

Run:

    uv run pytest packages/data/tests -q
    uv run ruff check packages/data
    uv run pyright packages/data
    uv run lint-imports

Acceptance requires two identical materializations to share a build key and cache path, an edit to an imported project helper to change the code snapshot and build key, and `expected_content_sha256` mismatch to fail before the result can be packaged.

### Milestone 4: Integrate planning, packing, jobs, and CLI

Update `packages/execution-pack/src/posttrain/execution_pack/contracts.py`, `planning.py`, and `datasets.py` to carry the new dataset selection and materialization manifest. Planning records the inert builder reference and declared inputs but does not import or run the builder. Materialization receives the exact project source snapshot and dependency lock used by the actual-job package. Extend `DatasetPackageLock` with build key, materializer schema version, builder target when present, code snapshot digest, and content digest. Ensure `job diff` distinguishes selection, builder code, declared input, and output-content changes.

Update `packages/jobs/src/posttrain/jobs/runtime.py` and `definitions.py` so standard data and training jobs accept the one chosen dataset selection type and receive the same canonical supervised or preference data objects as before. The runtime must not rematerialize a dataset already staged and verified inside the actual-job image.

In `apps/cli/src/posttrain_cli/commands/dataset.py`, add `dataset materialize` and `dataset verify`. Keep `dataset validate` as a deprecated alias that performs materialization and prints a replacement message. `materialize` reports source layer, build key, content digest, record count, cache path, and created-versus-reused state. `verify` uses a temporary output and never changes catalog files, package resources, or the reusable cache. Both commands support deterministic `--json` output.

Update `apps/cli/src/posttrain_cli/materialize.py`, `execution_planning.py`, state layout, doctor output, and command documentation to use the new service. Add CLI tests in `apps/cli/tests/test_cli.py` or a focused `test_dataset_commands.py`. Extend `packages/execution-pack/tests/test_datasets.py`, `test_planning.py`, and `test_service.py`, and add a job-runtime regression in `packages/jobs/tests`.

Exercise the integrated fixture from a temporary installable project:

    posttrain catalog validate
    posttrain catalog list --family dataset
    posttrain dataset materialize datasets/python-reviewed@1 --json
    posttrain dataset materialize datasets/python-reviewed@1 --json
    posttrain dataset verify datasets/python-reviewed@1 --json
    posttrain job plan .posttrain/work_packages/train.yaml --job train
    posttrain job pack .posttrain/work_packages/train.yaml --job train

The first materialization reports `created: true`; the second reports `created: false` with the same build key and digest; verification reports a match and leaves the cache modification time unchanged; planning completes without importing the builder; packing includes the manifest and output digest in the package lock.

### Milestone 5: Migrate the representative serving corpus

Create `packages/serve/src/posttrain/serve/benchmarks/general_serving/` with `definition.py`, `build.py`, and `resources/`. Move the reviewed first-party prompt inputs out of the generator source into a reviewable package resource. Keep prompt record and workload semantics in `posttrain.serve`; reuse only the source resolution, child-process build, canonical output, manifest, and verification support from `posttrain.data`.

The workload remains `workloads/general-serving-32k-sweep@1` in `packages/catalog/src/posttrain/catalog/base/workloads.yaml`. Extend its typed workload request value or a serve-owned corpus definition so it carries an inert build specification alongside the packaged corpus identity. Catalog loading must not import `general_serving/build.py`. Add `posttrain workload materialize` and `posttrain workload verify` commands in `apps/cli`, implemented through a serve-owned operation that delegates generic record materialization to `posttrain.data`.

Port the exact behavior from `packages/serve/src/posttrain/serve/benchmarks/materialize_corpus.py`: pinned GSM8K and HumanEval revisions, NFC and newline normalization, lowest-SHA-256 deterministic selection, 64 reasoning records, 32 code records, and 32 reviewed first-party records split evenly among chat, extraction, structured output, and tool use. Preserve source record keys and license notices. Exclude answers, canonical solutions, tests, and other non-prompt fields.

Before deleting anything, run both implementations and compare the JSONL and manifest byte for byte. The required content digest is `9a9467fd8a5e744968d09a4d8fd6f4d92a089c50a84e1e6e7e5c5520a9f4e50e`, with 128 records and category counts `reasoning: 64`, `code: 32`, and eight each for `chat`, `extraction`, `structured-output`, and `tool-use`. Add focused parity tests in `packages/serve/tests/test_prompts.py` and a no-import registration test in the catalog or serve suite.

After parity passes, remove the `posttrain-serve-materialize-corpus` entry from `packages/serve/pyproject.toml` and delete the standalone argument parser module. Update package resources and any active documentation that names the old command. Ordinary serving benchmark tests must continue to load the packaged resource without network access or a required Hugging Face dependency.

Run:

    uv run --with 'datasets>=4.6.1,<4.7' posttrain workload verify workloads/general-serving-32k-sweep@1
    uv run pytest packages/serve/tests packages/catalog/tests apps/cli/tests -q
    uv run ruff check packages/serve apps/cli
    uv run pyright packages/serve apps/cli

Acceptance is the generic command reproducing the exact corpus and a clean search showing no package-specific materializer command or permanent dataset behavior under the repository root.

### Milestone 6: Finish migration and enforce package conventions

Update `apps/cli/src/posttrain_cli/scaffolding/init_project.py` only where needed to make an installable project package and explicit Python provider easy to add. YAML remains the default starter path. Add a documented Python example fixture rather than forcing an empty provider into every project. If a scaffold command is introduced, keep it narrow, such as `posttrain dataset new ID --python`, and ensure it creates `definition.py`, `build.py`, resources, tests, and the explicit layer source without overwriting existing files.

Add a repository ownership check or extend the existing one so permanent dataset builders under root `scripts/` or `tools/` are rejected with a message naming the expected owning-package layout. Do not ban those directory names categorically; the check should identify package behavior by maintained entry points or known builder patterns and allow legitimate repository-wide release tooling in its owner application.

End the compatibility window only after at least one release or explicit migration decision. When removal is authorized, delete `python-file` and `runpy` support, remove the `dataset validate` alias, require catalog layer schema version 2 for layers containing Python providers, and update migration notes. If removal is deferred, leave the compatibility tests and warnings in place and record the remaining work here rather than claiming completion.

Reconcile the two overlapping plans by updating only their affected entries: mark `docs/plan/dx-packing-environments-datasets.md` milestone 7 as superseded by the completed milestones here, and revise `docs/plan/dx-repository-and-qualification-cleanup.md` so the generic workload materializer, rather than a package console script, is the durable owner. Update `docs/plan/serving-capacity-screen-and-observatory.md` maintainer commands without altering its retained benchmark evidence.

Run the full repository validation ladder from `/home/hammad/projects/rl`:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Record exact results in `Progress` and `Outcomes & Retrospective`. Do not include unrelated dirty changes in a commit or cleanup operation.

## Concrete Steps

Work from `/home/hammad/projects/rl`. Before each milestone, inspect `git status --short` and preserve unrelated changes. The worktree already contains active cleanup edits in `packages/serve/pyproject.toml`, the serving corpus generator, several Lab qualification modules, runtime-image locks, and other plans. Rebase the implementation on the current working tree rather than reverting or recreating those edits.

Implement in the milestone order because catalog provider loading depends only on typed definitions, while materialization and packing require the provider contract to be stable. Use focused tests after each file group, then the full ladder at the end. Update the `Progress`, `Surprises & Discoveries`, and `Decision Log` sections whenever an implemented constraint changes a later step.

For the end-to-end project fixture, create temporary directories through pytest's `tmp_path` or `mktemp -d`. Do not place generated test projects in the repository root. Tests that fetch Hugging Face revisions carry the existing `network` marker; unit tests use local fixtures and fake source resolvers. The real corpus verification is a documented release gate and may skip in the ordinary suite when the dependency or network is unavailable, with a clear reason.

## Validation and Acceptance

The change is accepted only when all of the following behaviors are observable.

A mixed catalog layer resolves YAML and Python-authored entries through the same `Catalog.resolve` API and reports one overlay ID. Catalog validation imports the provider but neither imports nor executes the builder. Duplicate IDs across authoring sources fail with the layer, family, and ID in the error.

A Python-built dataset declares all inputs, materializes in a child process, writes canonical records and a versioned manifest atomically, and reuses the cache when the selection, inputs, code snapshot, lock, and materializer version are unchanged. Changing any of those changes the build key. Changing output behind a locked expected digest fails verification.

Detached job planning records the inert build plan without loading heavy dependencies. Packing materializes or reuses the dataset, includes the output and manifest in the actual-job image, and binds the build and content identities into the package key. Runtime resolution consumes the staged snapshot without rebuilding it.

`posttrain dataset materialize` and `posttrain dataset verify` provide readable and JSON output. Verification performs no persistent write. During the compatibility window, `posttrain dataset validate` works and explains its replacement.

`general-serving-v1` is reproduced with the exact existing digest, count, categories, source revisions, provenance, and licenses. The package-specific console script is absent after parity, ordinary serving execution remains offline, and the workload still owns the corpus identity.

The complete validation ladder passes. Network-dependent and GPU-dependent tests may skip only under their existing markers; their required release-gate commands and latest results must be recorded before claiming the corresponding integration complete.

## Idempotence and Recovery

Catalog registration changes are additive. Schema version 1 remains readable during migration, and schema version 2 rejects ambiguous or duplicate sources before composing a catalog. Re-running catalog validation does not change the filesystem.

Materialization writes to a temporary directory and atomically promotes only a verified result. A failed build can be retried safely. A cache entry without its completion marker or with a mismatched manifest is treated as incomplete and rebuilt; it is never trusted or packaged. Verification always uses a temporary output and removes it on success or failure.

Do not delete the old builder or corpus generator until parity tests pass. If the generic corpus output differs, keep both implementations, compare the first differing record and manifest field, record the discovery here, and fix the generic path. Do not update the expected digest merely to make the test pass; a content change requires explicit review and a new corpus revision.

If job package identity changes unexpectedly, use `posttrain job diff` to distinguish catalog, code snapshot, input, materializer, and output changes. Preserve old content-addressed job images and dataset caches until the new path is qualified; they remain useful recovery evidence and do not need to be overwritten.

## Artifacts and Notes

Keep concise evidence here as implementation proceeds. At minimum retain:

    Catalog valid: <base release>, <N> base entries, 2 project entries
    dataset datasets/python-reviewed@1 created build_key=<sha256> content_sha256=<sha256> examples=<N>
    dataset datasets/python-reviewed@1 reused build_key=<same> content_sha256=<same> examples=<N>
    dataset datasets/python-reviewed@1 verified content_sha256=<same>
    workload workloads/general-serving-32k-sweep@1 verified corpus general-serving-v1@1 sha256=9a9467...

Also record one `posttrain job diff` excerpt demonstrating that a builder-code edit is named separately from an input-data edit.

## Interfaces and Dependencies

`posttrain.catalog` depends on selection decoders from capability packages but must remain free of builder execution. Its Python provider loader uses the standard library `importlib`; no plugin framework or entry-point discovery is added. `CatalogEntries` contains already-typed selections and can be converted directly to the `CatalogLayer` value in `posttrain.common`.

`posttrain.data` retains `posttrain-common` as its only required framework dependency. Hugging Face input resolution remains in the existing optional `huggingface` extra. The builder runner uses the selected project's Python environment and standard-library process APIs. Do not add Trackio, W&B, TRL, vLLM, Verifiers, or YAML imports to `posttrain.data`.

`posttrain.serve` may add a required dependency on lightweight `posttrain-data` materialization contracts. `posttrain.data` must not import `posttrain.serve`, so dependency direction remains one way. Extend the import-linter contract in `pyproject.toml` if needed to state this direction explicitly.

`posttrain.execution-pack` owns code snapshot identity and immutable staging. The materializer must receive that identity through a typed argument rather than rediscovering or hashing an arbitrary subset of the project. `apps/cli` obtains the same identity through project discovery and packing configuration for local materialization.

The final implementation must expose one stable set of names from `packages/data/src/posttrain/data/__init__.py` and document compatibility aliases. It must not expose `runpy`, child-process transport payloads, temporary cache markers, or serving-specific corpus types from `posttrain.data`.

## Revision Notes

- 2026-08-02: Initial plan written after reviewing the frozen post-training baseline and the active dataset, catalog, packing, serving-corpus, CLI, and cleanup implementations. It replaces the planned long-term `python-file` builder and package-specific corpus command with explicit Python catalog providers and one reproducible materialization lifecycle while preserving dataset and workload semantics.
