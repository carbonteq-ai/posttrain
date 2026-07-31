# Posttrain v0.2.5 developer experience critique

- **Status:** release-scoped critique, revision 4
- **Release:** [Posttrain v0.2.5](https://github.com/carbonteq-ai/posttrain/releases/tag/v0.2.5)
- **Published:** 2026-07-31 11:21:42 UTC
- **Tag commit:** `dc1c5e1c8ad5ba27ce1ccc3a356caf89a0ae1579`
- **Assessment date:** 2026-07-31
- **Evidence boundary:** source and documentation at tag `v0.2.5`

This review deliberately excludes uncommitted changes made after the release.
Those changes may be useful candidate fixes, but they are not behavior a
developer receives by installing v0.2.5. At assessment time the working tree
already contained candidate fixes for parts of finding 5 (an authoritative
`environment_file` setting in `execution.toml`, `posttrain.env` added to the
source-snapshot forbidden names, and `--registry` as an explicit one-off
override) and an in-progress remote-evaluation surface from
`docs/plan/external-endpoint-verifiers-screening.md`. They are acknowledged
here so they are not re-proposed from scratch, but they are not release
behavior and do not close any finding.

## Executive assessment

Posttrain v0.2.5 has strong low-level contracts for immutable job packaging,
provider-neutral execution, durable tracking evidence, admission, cancellation
intent, reconciliation, and evidence-gated cleanup. Its main developer
experience problem is that these contracts are exposed as an operating
procedure.

A developer who submits one job must understand three independent state
systems—local Posttrain control records, the execution provider, and
Trackio/W&B—and manually advance the lifecycle between them. The missing
component is not merely a monitoring daemon. It is a unified project control
surface backed by an idempotent lifecycle controller.

Revision 2 adds a second axis: the authoring surface. An ordinary project is
guaranteed to be a Python repository that owns harness code, one or more
Verifiers environment packages, and dataset preparation alongside its catalog
overlays and work packages — and the release gives that repository no
organizational contract. Environments can only be packed from a pushed git
commit, activation configuration can reference files that are never packed,
image qualification stops at imports, and dataset preparation happens outside
the system entirely. The most common project shape therefore iterates slowest
and fails latest.

The release should be treated as a strong framework substrate with an
incomplete application experience.

## Architecture decisions after plan review

Revision 5 reviewed the five implementation plans as one system and chose the
following boundaries. These decisions replace narrower proposals elsewhere in
earlier revisions of this document.

- Admission stores a complete, immutable prepared submission plus a locator for
  its owning control store. Pumping a queued run must not reopen the project and
  reinterpret current configuration.
- `RunView` is a read-only projection. A separate pure reconciliation reducer
  compares observations and returns explicit idempotent actions for the
  controller to persist and execute.
- Machine-wide provider, tracking, trust, and storage settings belong to named
  site profiles in `$XDG_CONFIG_HOME/posttrain/config.toml`; a file named
  `providers.toml` is too narrow for that responsibility.
- Training and evaluation require a durable tracking backend by default, but
  core code does not assume that backend is Trackio. A site profile may choose
  Trackio or W&B, and `none` is an explicit development waiver.
- Project resources are declared by logical name and source path. Their hashes
  are computed into the materialized pack lock; users do not hand-maintain
  derived SHA values for files already captured from the project tree.
- Environment contracts belong to a small `posttrain.environment` package, not
  to the framework-neutral `posttrain.common` package and not to `eval`.
  Catalog-family discovery is explicit and deterministic rather than an
  import-time mutable registry. The complete discovered family set is frozen
  into project/run snapshots and package identity; a referenced absent family
  is a planning error.
- `layer.yaml` continues to own layer id, revision, precedence, and provenance.
  Deterministic file discovery is introduced through a versioned manifest with
  explicit include/exclude patterns and a behavior-preserving migration from
  existing explicit membership.
- Release metadata has one version in a generated release manifest. A release
  command expands that value into package metadata and CI verifies the
  expansion. The tag verifies the already-qualified release commit; it is not
  required in order to build that commit.

## What the release gets right

- `JobExecutionService` persists submit and cancel intent before contacting a
  provider.
- Job images use immutable identities and separate semantic package identity
  from provider launch identity.
- Provider termination is not automatically equated with retained logical
  evidence.
- Reconciliation joins provider state with Trackio/W&B run status and required
  artifact roles.
- Cleanup is gated by retained evidence and preserves bounded diagnostics for
  failures that never opened a tracking run.
- The catalog distinguishes a versioned framework base from project overlays
  and records selection provenance.
- Observatory remains a read-only evidence product rather than becoming an
  execution control plane.

These primitives should be retained. The improvement is to compose them behind
a coherent project-level API and CLI.

## Severity model

`P0` findings can target the wrong run, use the wrong project configuration, or
leave execution capacity indefinitely blocked. `P1` findings make the normal
workflow unreliable or require internal framework knowledge. `P2` findings are
important consistency and automation problems after correctness is restored.

## Findings

| Priority | Finding | Developer impact |
| --- | --- | --- |
| P0 | Machine-wide admission entries do not retain their owning project locator | Releasing project A's local placement can attempt to submit project B's waiting plan through project A's provider factory and receipt store. |
| P0 | `--last` is sorted by admission-state priority, not only time | A mutating command such as `posttrain run cancel --last` can select an older waiting run instead of the newest run. |
| P0 | Terminal completion and cancellation repair are manual | Local runs can remain `terminal_pending_evidence`, retain a GPU placement, and block later work until a developer reconciles them. |
| P0 | Provider reconstruction uses current project configuration | Moving or changing a dstack binding after submission can make status, cancellation, or collection query a different provider scope. |
| P0 | Project runtime configuration is shell-dependent | `posttrain.env` must be sourced, ambient variables can change behavior, and a future background process will not necessarily inherit the submitting shell. |
| P0 | Environment activation can reference files that are never packed | A project-relative `taskset.data_path` survives planning, packing, publication, and image qualification, then fails with `FileNotFoundError` at the first rollout of a submitted run. |
| P0 | In-repo environment packages must be committed and pushed before every pack | `EnvironmentSource` accepts only an HTTPS repository plus a full commit SHA, so iterating on a project-owned Verifiers environment requires a git push per attempt; the working tree can never be packed. |
| P1 | `.posttrain/state` mixes disposable cache, persistent configuration, and authoritative control receipts | Deleting what looks like scratch state can remove provider handles, intent, recovery information, and machine binding. |
| P1 | Run inspection is split across provider status, reconciliation, and Observatory | There is no default command that answers “what is the current logical state of this run?” |
| P1 | Read-only job planning requires a publication registry | A developer cannot fully inspect a semantic plan before configuring where a later job image will be pushed. |
| P1 | Local execution also assumes registry publication | A first local Docker run requires remote publication infrastructure even when a content-addressed local image would suffice. |
| P1 | Packing includes every project overlay file rather than the selected transitive closure | Editing an unrelated catalog selection can invalidate an otherwise unchanged job image. |
| P1 | The semantic pack plan does not bind the exact staged project-configuration digest | Project configuration can change between plan and pack without producing the same explicit “source changed after planning” boundary used for code. |
| P1 | Required job secret names are not derived from selected seats | Developers must understand and repeat `--env` names that are already known by an environment or external-service binding. |
| P1 | Producer-to-consumer artifact handoff is handwritten | Continuing from SFT to GRPO/evaluation requires copying Trackio artifact identity and reconstructing a large `ModelVariant` overlay entry. |
| P1 | Actual-job image qualification stops at imports | The smoke stages verify package imports and manifest parsing only; environment activation, `Taskset.load()`, and staged-dataset opens are first exercised after provider submission. |
| P1 | The project repository has no organizational contract | `init` scaffolds one `src/` package; a project with several environment packages, dataset preparation, and harness code must invent its own layout and hand-maintain `[tool.posttrain.pack]` to match it. |
| P1 | Dataset preparation is invisible to the framework | Dataset sources are static files; a generator script has no declared home, no execution hook, and no freshness check between the script and its committed output. |
| P1 | Tracking policy and the site backend are configured repeatedly per project | Every project re-enters a server URL and token configuration by hand, and the no-op-observer path carries special admission semantics that ordinary managed runs do not use. |
| P2 | `job` and `work-package` expose overlapping execution commands | The distinction between composition inspection and execution is harder to learn than necessary. |
| P2 | `doctor` is project-wide rather than intent-specific | It warns about infrastructure irrelevant to validation while failing to explain whether one exact job is launch-ready. |
| P2 | JSON output has no stable error envelope | Automation receives structured success values but plain stderr strings for failures. |
| P2 | The high-level application service lives in private CLI modules | Python automation and a future controller would need to import `posttrain_cli` internals or duplicate composition. |
| P2 | Starter work packages duplicate the seat map | The inline recipe declares seats and the bindings restate every seat as a three-field ref, with agreement enforced only at resolve time. |
| P2 | Install bootstrap depends on out-of-band artifacts and per-command flags | `github-constraints.txt` must be fetched beside the index, `--system-certs` must be repeated on every `uv` invocation, and the CA is installed in two places by hand. |
| P2 | The veRL runtime variant cannot pack from installed distributions | A project that packs SFT/TRL jobs from wheels hard-fails on veRL without a framework source checkout. |
| P2 | Release versions and digests are hand-edited across the repository | One version bump touches ~27 files, restates the version literal ~80 times, and hand-updates derived lock digests inside catalog YAML; any missed edit ships silently. |

## Detailed critique

### 1. Admission has no durable project owner

`packages/execution/src/posttrain/execution/admission.py` stores one
machine-scoped queue. An entry contains the encoded plan, evidence source, and a
provider-binding fingerprint, but not the project root or project state root.

`apps/cli/src/posttrain_cli/execution_provider.py` constructs
`ExecutionAdmissionService` with closures over one `ProjectLayout`. When
`acknowledge_reconciled()` releases a placement, the admission service selects
the next waiting entry and pumps it through whichever project's service factory
performed the acknowledgement.

The v0.2.5 cross-project test proves that two project factories share one host
lock and that project B waits behind project A. It stops before releasing A and
therefore does not prove that B is submitted with B's configuration and receipt
store.

The admission record needs a mode-protected, secret-free control locator and a
complete prepared submission:

    project_id
    control_store_uri
    prepared_submission
      launch_plan
      provider_source
      evidence_source
      package_identity

The controller must execute the frozen prepared submission and write receipts
through its recorded control store. It must not reload a mutable project root
when a slot becomes available. If the control store is unavailable, the entry
becomes an attention state rather than being submitted through another
project's factories.

### 2. “Last” can target the wrong run

`apps/cli/src/posttrain_cli/run_resolve.py` first sorts runs by timestamp and
then performs a stable sort by admission priority. Waiting runs sort ahead of
submitted, terminal, completed, and cancelled runs. `--last` returns the first
element of that state-prioritized list.

This behavior may be useful for an attention queue, but it is not the meaning
of “last.” It is unsafe on `cancel`, `cleanup`, `retry-submit`, and tracking
recovery.

Mutating commands should require the complete canonical run id. Prefixes and
`--last` remain read-only conveniences; even an unambiguous prefix can change
meaning between resolution and action. Read-only `--last` becomes strictly
chronological, while a separate `--needs-attention` selector expresses the
current priority behavior honestly.

### 3. Reconciliation is a manual release barrier

The intended evidence barrier is correct: a provider exit is not sufficient to
declare a successful Posttrain run. The developer experience is not.

The v0.2.5 cancellation path is:

    posttrain run cancel RUN_ID
    posttrain run recover-cancelled-tracking RUN_ID   # when stranded
    posttrain run reconcile RUN_ID
    posttrain run cleanup RUN_ID

A successful local run similarly requires `run reconcile` before admission is
released. If the developer closes the terminal, forgets the command, or the
submitting process dies, later local work can remain queued behind a provider
run that already finished.

The routine path should be automatic. Manual commands should remain guarded
repair tools.

### 4. Provider identity is less durable than evidence identity

An `ExecutionSubmission` stores provider name, provider id, idempotency key,
image, and evidence source. The evidence source is sufficient to reconstruct
the exact tracking destination after project settings change.

Provider construction does not have an equivalent locator. The CLI reads the
current ignored `execution.toml`, forces the recorded provider kind, and creates
a new adapter from the current dstack project, executable, credential file, and
storage settings. A configuration edit can therefore make an old run
unqueryable even though its provider id remains present.

Add an `ExecutionProviderSource` beside `ExecutionEvidenceSource`. It should
store only secret-free stable identity: provider kind, profile id, dstack
project or endpoint scope, credential-file identity, and binding fingerprint.
Secret values may rotate behind the same stable profile.

### 5. Configuration authority is fragmented

The release has three overlapping configuration paths:

- committed `.posttrain/project.toml` execution defaults;
- ignored `.posttrain/state/execution.toml` machine binding;
- ambient process variables, commonly populated by sourcing `posttrain.env`.

This produces hidden precedence and makes a background service unreliable. It
also puts persistent structural configuration below a directory documented as
ignored runtime state.

The target contract should be:

| Configuration | Location | Authority |
| --- | --- | --- |
| Project identity, catalog paths, work packages, non-secret defaults | `.posttrain/project.toml` | Git-tracked project intent |
| Project secret values, site-profile selection, explicit project runtime values | root `posttrain.env` | Ignored, mode-0600, automatically loaded |
| Named site profiles: provider, tracking endpoint, storage roots, trust and credential-file locations | `$XDG_CONFIG_HOME/posttrain/config.toml` | Operator-owned machine configuration |
| Submit/cancel intent, handles, journals, reconciliation and cleanup receipts | `.posttrain/state/executions/` | Local control records |
| Datasets, build contexts and downloaded assets | `.posttrain/state/cache/` | Disposable cache |

`posttrain init` should create a protected empty `posttrain.env`, ignore it, and
create a tracked `posttrain.env.example` containing variable names only. No
manual pointer inside `execution.toml` and no `source posttrain.env` step should
be required.

Ambient shell variables do not participate in project runtime configuration,
including as a fallback for missing values. An explicit `--env-file PATH` is a
suitable one-command override. Variables that locate the process itself, such
as `XDG_STATE_HOME`, remain machine process configuration and are not project
runtime values.

### 6. Run state has no unified read model

`posttrain run status` reports provider/admission state. `posttrain run show`
requires the optional Observatory application and reports tracking-derived
evidence. `posttrain run reconcile` is the joined view, but also persists state
and advances admission.

The project needs one provider-neutral `RunView` with separate dimensions:

    logical_state: finalizing
    control: terminal_pending_evidence
    provider: {state: succeeded, observed_at: ...}
    evidence: {state: running, required_roles_pending: [trained_model]}
    reconciliation: {state: pending, next_retry_at: ...}
    controller: {state: healthy, last_sweep_at: ...}

Do not collapse these into a false single “succeeded” flag. A provider can be
done while evidence is still finalizing, or evidence can remain active after a
provider cancellation.

The same `RunView` should back CLI human output, CLI JSON, the Python API, and
read-only Observatory integration. Reading the view must never persist a state
transition. A pure reconciliation reducer consumes a view plus observations
and returns explicit actions; only the manual reconciler or lifecycle
controller executes those actions.

### 7. Planning and publication are coupled

`apps/cli/src/posttrain_cli/execution_planning.py` resolves the registry while
building a `PlannedJobPackage`, even for `job plan`. The semantic job plan needs
release image identities, source/config digests, data/environment locks, target,
and runtime variant. It does not need the destination repository for a project
job image.

Separate:

- `JobPackSpec`: immutable job meaning;
- `JobPublicationPlan`: where and how those bytes are published;
- `JobLaunchPlan`: provider, policy, mounts, and run id.

The registry coupling is not only a planning concern. The BuildKit publisher
always builds with `type=image,push=true` and then verifies the result with
`docker buildx imagetools inspect` against the remote repository; there is no
load-only path, so even a fully local Docker run requires a reachable OCI
registry round-trip for an image that will execute on the same machine.

Local Docker should support a content-addressed local-image publication adapter.
dstack can require an OCI registry at submission preflight. `posttrain job plan`
should remain useful before either publication path is configured.

The v0.2.5 consumer documentation also demonstrates `posttrain job pack
--provider dstack`, while the `job pack` command does not accept a provider.
That inconsistency follows from the same unclear boundary.

### 8. Packing is broader than the selected job

The project source snapshot is bounded, and the configuration bundle already
narrows work packages to the single selected file — the closure validator
rejects a bundle containing sibling work packages. Catalog overlays get the
opposite treatment: `_project_config_bundle()` globs every file below every
configured overlay directory wholesale. The selected work package may reference
only one model and one training setting, yet editing an unrelated evaluation
entry changes the staged project configuration and therefore the package key.

Packing should retain:

- the project manifest and project brief;
- the selected work package;
- the transitive catalog entries needed to resolve its enabled job;
- source files selected by `[tool.posttrain.pack]`;
- the exact environment/data/runtime locks derived from those entries.

The semantic plan must include the digest of that closed configuration before
materialization. Pack should fail with “project configuration changed after
planning” just as it already does for source bytes.

The framework source origin must also be explicit. Silently packing a checkout
because the installed CLI happens to live inside one is useful for framework
development but surprising for an ordinary project. Default to installed
released distributions; require an explicit development-source selection for
a checkout. The `online-rl-verl-py313` runtime variant inverts this rule: it
refuses to pack from installed distributions at all and demands a framework
source root, so a project that runs its TRL jobs from wheels discovers only at
veRL pack time that its installation mode is insufficient. That divergence
should be a documented capability of the variant, surfaced by `doctor` and
plan-time preflight, not a pack-time error.

### 9. Runtime requirements should flow from selections

Execution accepts environment variable names without embedding secret values,
which is the right security boundary. The set of names is assembled from job
defaults, project defaults, and repeated CLI `--env` flags.

Selections already know some requirements. For example, an external evaluation
service can declare the API-key variable name. Environment packages can also
declare required credentials. Add a secret-free
`required_runtime_variables` contract to the relevant selection/job types.

Planning should produce:

    Required runtime variables
      OPENROUTER_API_KEY       present in posttrain.env
      TRACKIO_WRITE_TOKEN      present in posttrain.env

    Unused project variables
      HF_TOKEN

Only declared job variables are forwarded. dstack client credentials remain
provider-process-only and must never be forwarded into the job container.

### 10. Catalog mechanics leak into the user workflow

The catalog's base-plus-overlay model is a good reproducibility boundary. Its
current authoring path exposes loader mechanics:

- a hand-authored YAML file is silently ignored unless listed in `layer.yaml`;
- `catalog show` cannot explain the originating file and transitive links;
- training settings and training runtime bindings share one broad family;
- output artifacts must be transcribed into a complete new model entry.

Keep the catalog for reusable, versioned, secret-free selections. Do not use it
for provider handles, secrets, run state, or every one-off hyperparameter.

Add:

    posttrain catalog explain FAMILY/ID
    posttrain artifact pin --run RUN --role trained_model --as MODEL_ID

`catalog explain` should report origin file, layer, revision, links, and known
consumers. New overlay layers should discover YAML files recursively in sorted
path order, excluding `layer.yaml`, and reject duplicate ids with both source
paths. `layer.yaml` remains authoritative for layer identity, revision,
precedence, and provenance; version 2 adds explicit include/exclude patterns.
Legacy explicit-file layers remain behaviorally unchanged for a deprecation
window, and a migration command first writes an equivalent exact include list
before any developer opts into broad discovery. Resolved snapshots retain the
manifest digest, selected file digests, and entry provenance. Do not add a
permanent command merely to maintain two declarations of file membership.

For one-off continuation, introduce an explicit run-output binding:

    bindings:
      model:
        type: run-output
        run_id: train.sft-01J...
        role: trained_model

Planning resolves this to one immutable artifact version. `artifact pin`
promotes the same output to a reusable catalog `ModelVariant` while deriving
base model, renderer, form, digest, producer run, and artifact version from
retained evidence. This remains explicit lineage; it does not make jobs
silently wire their outputs.

### 11. CLI success and failure are asymmetrical

Normal JSON output is structured. Expected failures pass through the top-level
exception handler and become a plain `error: ...` line on stderr. Automation
cannot reliably distinguish invalid configuration, unavailable provider,
pending evidence, incompatible selection, or an ambiguous submission.

Define a stable error envelope:

    {
      "code": "evidence_pending",
      "phase": "reconciliation",
      "message": "tracking run is not terminal",
      "retryable": true,
      "suggested_command": "posttrain run watch <run-id>"
    }

Use distinct documented exit statuses only where shell automation benefits;
the JSON `code` is the durable API.

### 12. The CLI is currently the application layer

Project discovery, catalog composition, semantic planning, packing, provider
construction, submission, tracking-source construction, and reconciliation are
assembled under `apps/cli/src/posttrain_cli`. A background controller or Python
automation client would have to import private CLI modules.

The one public extension seam that does exist is misdocumented and half
private. `ProjectEntry.configure` is documented as `configure(runtime)` but
actually receives a `ProjectExecutionRequest` and must build and return the
`JobRuntime` itself, and the policy that loads and validates an entry
(`load_project_entry`, `validate_standard_definitions` in
`apps/cli/src/posttrain_cli/work_runtime.py`) is private to the CLI, so an
alternative host cannot reuse the rules for what an entry may and may not
override. There is also no public `Project` object at all: the highest-level
public value is the resolved-paths `ProjectLayout` dataclass.

The target packages are:

- `packages/execution`: provider-neutral lifecycle state, reconciliation
  decisions, idempotent actions, locks, and `RunView`;
- a public `posttrain.project` application package: project discovery,
  intent/materialize/publish/submit, run listing, cancellation, and
  dependency-injected provider and tracking adapters;
- a `posttrain.environment` package: shared environment sources, activations,
  resources, and schemas without making `common` Verifiers-aware;
- `apps/cli`: Typer argument parsing and rendering only;
- a controller application: foreground loop and service integration;
- `apps/observatory`: read-only durable evidence and analysis.

### 13. Environment activation resources escape the pack boundary

An activation is either a `module:callable` reference or an opaque
Verifiers-config mapping. `_activation_lock` copies that mapping verbatim into
the `EnvironmentActivationLock`, checks only JSON serializability, and digests
it as a string. Nothing inspects `taskset.data_path`, nothing stages a
referenced file, and there is no resource concept in the pack layout.

This is not hypothetical: the Ambient Agent DAPO packaging failure recorded in
`docs/feedback/verifiers-environment-data-packaging.md` is exactly this defect.
A `data_path` that existed in the developer checkout passed planning, packing,
publication, and image qualification, then killed the submitted run with
`FileNotFoundError` before the first rollout. Worse for reproducibility: the
referenced bytes sit outside the pack digest, so two packages with identical
package keys can behave differently depending on the submitting machine's
filesystem.

Adopt the resource contract from that feedback document as release behavior:

- named `activation.resources` declarations on the `EnvironmentBinding`, with
  explicit resource references inside the otherwise opaque activation config;
- resources staged below a deterministic environment-resource directory and
  included in the pack identity, with size and digest computed into the
  materialized lock;
- declared names resolved to staged paths before the native Verifiers
  configuration is constructed;
- unresolved project-relative paths rejected at detached planning for portable
  execution, while explicit local paths remain valid for in-process
  experimentation.

### 14. Project-owned environments cannot iterate

`EnvironmentSource` requires an HTTPS repository URL and a full 40-character
commit SHA; `subdirectory` is the only optional field. There is no local-path
and no wheel source kind. Yet the guaranteed shape of an ordinary project is a
repository that contains its own Verifiers environment package(s) next to its
harness code — the framework's own repo keeps them under `environments/`.

The consequence is a broken edit loop. Project source is snapshotted from the
working tree by content digest, but the environment package sitting in the same
repository must be committed **and pushed** before every `job pack`, and the
error arrives only when planning tries to derive the wheel request. Every
environment iteration costs a synthetic commit, and uncommitted environment
code is simply unpackable.

Add a project-path environment source:

    environment:
      envs/episode-qa@dev:
        source:
          kind: project-path
          path: environments/episode_qa

Identity comes from the same source-tree digest the packer already computes for
project code, the wheel is built from the staged tree exactly as it is built
from a git checkout today, and immutability is preserved because the package
key is content-addressed either way. The full-commit git source remains the
correct form for publishing an environment across projects; `project-path` is
the correct form for developing one.

### 15. Image qualification stops at imports

The kind-image smoke stages are pure import checks (`import verifiers, vllm`),
and the actual-job smoke stage runs `posttrain-runtime --help`, parses
`package.json`, and (for veRL) resolves projected modules. No stage activates
the selected environment, calls `Taskset.load()`, or opens a staged dataset.
Combined with finding 13, the first execution of the developer's actual
configuration happens on the provider, after admission, image publication, and
GPU allocation.

Actual-job qualification should execute through the real runtime entry point
inside the built image with network disabled, no runtime secrets, a temporary
writable directory, and a timeout. It should exercise, per selected binding:
environment activation construction, `Taskset.load()` for the selected
activation, and an open-and-parse pass over each staged dataset package. A
portable environment whose load step needs network access or credentials must
split that work from taskset construction; silently deferring qualification to
the GPU is not an acceptable default.

### 16. The project repository has no organizational contract

A real project owns, at minimum: harness/glue code, one or more environment
packages, dataset preparation, catalog overlays, and work packages. The
release's answer to "where does each live?" is one scaffolded
`src/<project_package>/` directory and a hand-maintained
`[tool.posttrain.pack]` table. There is no scaffolded or documented convention
for multiple environment packages, no dataset-script location, and nothing that
keeps `project_packages` / `source_includes` consistent with the actual layout
as packages are added.

Adopt and scaffold one standard layout:

    <project>/
      pyproject.toml            # pack table derived, not hand-edited
      src/<project_package>/    # harness and glue code
      environments/
        <env_name>/             # installable Verifiers package (own pyproject)
      datasets/
        <builder>.py            # declared dataset builders (finding 17)
      .posttrain/
        catalog/  work_packages/  state/

Derive discovery from the convention: every `environments/*` directory
containing a `pyproject.toml` is eligible as a `project-path` source, but only
the environments reachable from the selected job are staged. Declared dataset
builders and their inputs join the selected closure. `posttrain env new NAME`
should scaffold a minimal Verifiers package in place. Explicit
`[tool.posttrain.pack]` entries remain as overrides, not as the primary
interface.

### 17. Dataset preparation is invisible to the framework

Dataset source kinds are static: `fixture`, `huggingface`, `jsonl`, `nemo`,
`parquet`. A project whose data is produced by a script — the common case for
environment-derived or synthesized training data — must run that script by
hand, commit the output, and hope the two stay in sync. The framework neither
runs the script, records which inputs produced the file, nor detects staleness;
a forgotten regeneration is silently packed as authoritative data.

Add a declared builder to project dataset entries:

    dataset:
      datasets/kg-extract-sft@3:
        source:
          kind: built
          builder:
            kind: python-file
            path: datasets/build_kg_extract.py
            callable: build
          inputs: [datasets/build_kg_extract.py, data/raw/episodes.jsonl]

Semantic planning records the builder and input digests without executing
project code. Materialization executes the builder idempotently into the state
cache, automatically rebuilding on a cache miss, and records the input and
output digests in the materialized pack lock. A `--frozen` mode fails when the
lock is missing or stale for CI and exact replay. The normal edit loop rebuilds
instead of asking a developer to detect staleness manually. Static kinds remain
for genuinely static data.

### 18. Dual authoring is asymmetrical

The design stance is "YAML + schema or typed Python; both validate into the
same models." The release delivers neither half completely. Composition is
YAML-only: there is no public `Project` object and no supported Python path
from "open this project" to "plan and submit this job" (finding 12). Extension
is Python-only: adding a custom job definition requires the undocumented entry
hook with the inverted signature, whose loading policy is private to the CLI.

The YAML that does exist is more verbose than its own model requires. The
starter work package inlines its recipe, so the seat map is stated twice —
once as `recipe.seats`, once re-spelled in `bindings` as three-field refs —
with agreement between the two enforced only at resolve time. A `recipe: ref`
form that collapses this exists but is not what the scaffold teaches.

The fixes largely follow from findings already made: ship the public
`posttrain.project` service (finding 12) as the typed Python half; ship starter
templates that bind a catalog recipe instead of an inline copy; and replace the
inverted runtime-building hook with a documented `ProjectPlugin.register(...)`
extension while keeping a one-release compatibility adapter for existing
`ProjectEntry` implementations.

### 19. Tracking should be required and supplied by the site profile

The product docs treat the tracking backend as a per-project selection with a
no-op observer as a first-class alternative — one that even carries its own
special admission-release semantics. Deployment reality is the opposite: every
site runs a provisioned Trackio server, and no ordinary project runs untracked.
The cost of modeling tracking as optional is paid on every setup: each project
hand-enters `POSTTRAIN_TRACKIO_SERVER_URL` and a write token, `run show`
depends on optional Observatory, and the evidence half of the lifecycle is
configured rather than inherited.

Make durable tracking required for managed training and evaluation, while the
named profile in `$XDG_CONFIG_HOME/posttrain/config.toml` selects the deployed
backend and endpoint. The current site profile chooses Trackio; another site
may choose W&B without changing lifecycle rules. `init` inherits the profile,
leaving only the selected backend's secret token in `posttrain.env`; `doctor`
verifies reachability; and `run show` reads tracking directly per finding 6.
The no-op observer remains an explicit development waiver rather than a branch
every ordinary lifecycle rule must accommodate.

This changes the frozen product baseline. Before implementation, explicitly
unfreeze it, add a dated amendment to `docs/post-training/README.md`, revise the
no-op admission rule in `03-work-and-evidence.md` and dependent API/observation
language, review the migration, and refreeze. If that amendment is not
accepted, the implementation may supply site defaults but must keep tracking
optional.

### 20. Bootstrap depends on out-of-band artifacts

Installation requires `github-constraints.txt`, which the release attaches to
the wheelhouse but the index cannot serve — the consumer guide itself calls
this a gap. The internal CA must be installed twice by hand (system store plus
`/etc/posttrain/trust/`), `--system-certs` must be repeated on every `uv`
invocation, and the `environment_file` pointer is written relative to
`.posttrain/state/`, an ignored directory, by hand. Each item is small; the sum
is a first hour spent on transport rather than on the project.

The preferred fix is to publish maintained fork wheels with immutable versions
to the same internal index and express their exact pins in normal package
metadata, so a consumer does not need a file before it can install Posttrain.
Ansible should install the CA into system trust and configure uv's native-TLS
setting once at the machine level. `init` owns project configuration; `doctor`
only diagnoses host bootstrap and prints the operator repair command.

### 21. Release versions and digests are authored, not generated

The v0.2.5 release bump (`9e65b2ba`, merged by the tag commit) touched 27
files. Twenty-six `pyproject.toml` files each bump their own `version` and
re-pin every first-party dependency `==0.2.5`, so the version literal appears
roughly eighty times. The same commit hand-updates `dependency_lock_sha256`
values embedded in three entries of the base catalog's `training.yaml`,
regenerates `uv.lock`, and edits a lab test that hardcodes the version string.
Adjacent derived values live in still more places: pinned image digests in
`runtime-images/published.toml`, fork pins in `release/github-constraints.txt`.

Every one of these is a derived value. A release performed by hand-editing
derived values has no invariant protecting it: a missed pin, a stale lock
digest, or a forgotten test literal ships silently and surfaces as drift a
consumer debugs. The framework already applies the correct discipline to job
packages — content-addressed, generated, verify-after-write — but not to its
own release metadata.

The target contract is that **a release commit is 100% machine-generated**:

- **One release manifest.** `release/manifest.toml` contains the sole release
  version. `posttrain-release prepare X.Y.Z` updates only that manifest, and
  `posttrain-release stage DESTINATION` expands every package version and exact
  first-party pin in an isolated tree. Source pyprojects remain release-neutral;
  static wheel and sdist metadata remain inspectable. The eventual `vX.Y.Z` tag
  must match the manifest and point at the already-qualified commit.
- **Digests as generated lock tables.** Catalog entries reference a named lock
  (`lock: trl-fork@<commit>`); one generated file owns the name→sha256 table.
  Deterministic dependency locks are regenerated and diffed. Published image
  digests are different: they are captured once from signed build receipts and
  verified by immutable digest, never reconstructed later from mutable tags.
- **Consumer-complete index.** Maintained fork wheels are published to the
  internal index, and the generated package metadata pins them. The constraints
  file may remain a release-build input but is not a consumer prerequisite.
- **Curated release PR.** Small changelog fragments or PR labels feed a
  machine-generated release PR. Requiring every development commit to follow a
  changelog grammar is unnecessary. Changed images are built and pushed once,
  their receipts are committed to the release PR, the merged commit is
  re-verified, and only then is the matching tag created.
- **No hand-edited repetitions.** Tests read installed metadata; CI forbids
  unmanaged version and digest literals outside the manifest, generated files,
  changelog, and lockfile.

`posttrain-release check` therefore does not merely compare package files to
one another. It reads the expected version from `release/manifest.toml`, proves
that it is the only authored version source, and checks every generated package
version and first-party pin against that value.

## Source-of-truth model

| Source | Authoritative question | Retention |
| --- | --- | --- |
| Git project files | What did the project intend to run? | Version controlled |
| `posttrain.env` | Which project runtime values and job secrets are selected? | Local/project managed, protected |
| Named site profile | Which provider, tracking backend, trust, and storage identities does this machine supply? | Operator managed |
| Local control receipts | What was requested, submitted, cancelled, reconciled, or cleaned? | Retain until recoverable/imported |
| Machine admission | Which local placement is held and what waits behind it? | Compact active state plus terminal receipts |
| dstack or local Docker | What is the provider doing now? | Live and potentially ephemeral |
| Trackio/W&B | What logical outcome, metrics, traces, and artifacts were retained? | Durable evidence |
| Observatory | What can be computed from retained evidence? | Rebuildable read views |

The unified run view is derived from these sources. It is not another mutable
database that can disagree with them.

## Proposed lifecycle controller

The public product concept should be a **lifecycle controller**. A daemon is one
deployment form of that controller.

For each registered project, the controller should:

1. Load the named site profiles and active control-store locators.
2. Inspect only active or attention-required admission entries.
3. Deliver persisted cancellation intent idempotently.
4. Observe provider state with bounded exponential backoff and jitter.
5. When a provider is terminal, reconcile the exact recorded evidence source.
6. Perform tightly guarded cancelled-evidence recovery only when the selected
   tracking adapter exposes that capability and exact identities match.
7. Release admission only when reconciliation is settled.
8. Pump the next waiting entry from its frozen prepared submission and write
   through its recorded control store, without reopening project configuration.
9. Retain inconsistent, lost, or unresolved submissions as attention states.
10. Apply cleanup only under an explicit retention policy.

Safe automatic actions are observation, idempotent cancellation delivery,
reconciliation, guarded cancellation recovery, admission release, and later
policy-driven cleanup. An ambiguous submission must remain manual unless the
provider adapter implements a reliable `resolve_submission()` capability.

The controller should support:

    posttrain controller run                 # foreground and CI
    posttrain controller enable              # optional systemd user service
    posttrain controller status
    posttrain controller logs
    posttrain controller disable

An always-on production deployment should run the same controller core as a
system service installed through Ansible. A developer-laptop service cannot
guarantee reconciliation while the laptop is suspended. Observatory must not
absorb this responsibility.

## Target developer journey

### Project setup

    posttrain init memory-agent --template sft
    cd memory-agent
    $EDITOR posttrain.env
    posttrain preflight .posttrain/work_packages/sft.yaml --provider dstack

No environment sourcing and no manual pointer from ignored state to
`posttrain.env` should be required. Tracking configuration is inherited from
the machine profile; only the write token lands in `posttrain.env`.

### Author environments and datasets in place

    memory-agent/
      pyproject.toml
      src/memory_agent/
      environments/
        episode_qa/            # installable Verifiers package, own pyproject
        retrieval_qa/
      datasets/
        build_kg_extract.py
      .posttrain/
        catalog/  work_packages/

    # .posttrain/catalog/environments.yaml
    environment:
      envs/episode-qa@dev:
        source:
          kind: project-path
          path: environments/episode_qa
        activation:
          kind: verifiers-config
          config:
            taskset:
              id: episode-qa-v1
              mode: extract
              split: train
              data_path: { $resource: task_data }
          resources:
            task_data:
              source: { kind: project-path, path: data/kg_extract/train_env.jsonl }

`posttrain job pack` resolves the path source by content digest, builds the
wheel from the staged tree, stages declared resources under the pack identity,
and calls `Taskset.load()` for every selected activation during image
qualification. No git push is required to iterate; publishing the environment
for other projects still uses a full-commit git source.

### Submit and follow

    posttrain job run sft --provider dstack --follow

Expected output:

    Run: train.sft-01J...
    Package: 8f19...
    Provider: dstack/pt-73ab...
    State: running
    Target: carbonteq-ai-workstation.lan
    Evidence: active in Trackio
    Controller: healthy

When the provider terminates:

    State: succeeded
    Provider: done, exit 0
    Evidence: consistent, 2 required artifacts retained
    Admission: released
    Cleanup: retained for 24h

The developer does not run `reconcile` in the normal path.

### Cancel

    posttrain run cancel train.sft-01J...
    posttrain run watch train.sft-01J...

Expected result:

    Cancellation requested
    Provider: cancelled
    Tracking: cancelled
    Evidence: consistent

### Inspect source disagreements

    posttrain run show train.sft-01J...
    posttrain run inspect train.sft-01J... --source control
    posttrain run inspect train.sft-01J... --source provider
    posttrain run inspect train.sft-01J... --source tracking

`run show` must work without installing Observatory. The `inspect` variants are
expert diagnostics.

### Python automation

    from posttrain.project import Project

    project = Project.open(".")
    intent = project.jobs.plan("sft")
    run = project.jobs.run(intent, provider="dstack")

    for update in run.watch():
        print(update.logical_state, update.provider.state, update.evidence.state)

    run.cancel()

The CLI and controller must call this same public service rather than
reimplementing it.

## Recommended delivery order

1. Make mutations require exact run ids, and add release-metadata drift checks.
   Both are small safety nets that protect all later work.
2. Establish the minimal public `posttrain.project` application service, the
   `posttrain.environment` contract package, and deterministic catalog-family
   assembly. Later configuration, packing, CLI, and controller work use these
   seams instead of adding more logic to private CLI modules.
3. Add named activation resources whose digests are computed into the pack
   lock, then run offline `Taskset.load()` and staged-data qualification through
   the actual runtime entry point. This closes the production failure first.
4. Make `posttrain.env` authoritative and introduce named site profiles in
   `$XDG_CONFIG_HOME/posttrain/config.toml`. Move control receipts and caches to
   explicit subdirectories before a controller depends on them.
5. Freeze complete prepared submissions in admission, add immutable provider
   locators, and prove cross-project release-and-pump without reloading project
   configuration.
6. Add read-only `RunView`, the pure reconciliation reducer, and typed errors;
   migrate run CLI commands to the shared application service.
7. Add the foreground controller, qualify it against fake and real provider /
   tracking combinations, then deploy the same core as an Ansible-managed
   system service. A systemd user unit is optional workstation convenience.
8. Derive runtime-variable requirements from selected seats and add exact-job
   preflight. Require a configured evidence backend by default without
   hard-wiring Trackio into framework contracts.
9. Separate intent, materialization, publication, and launch; add local-image
   publication and hash only the selected transitive configuration closure.
10. Add project-path environments, selective convention-based discovery, and
    declared dataset builders with automatic cache rebuilds plus frozen replay.
11. Add run-output bindings, artifact pinning, deterministic overlay discovery,
    catalog explanation, schemas, and authoring generators.
12. Complete generated releases from one release manifest, generated dependency
    locks, captured image receipts, consumer-installable fork wheels, and a
    curated release-PR flow.
13. Deprecate manual happy-path reconciliation and compatibility aliases only
    after the automatic path has live provider and tracking qualification.

## Implementation plans

The findings are divided into five delivery categories, each with a
self-contained ExecPlan under `docs/plan/` authored per
`docs/templates/PLAN.md`:

| Category | Findings | Plan |
| --- | --- | --- |
| Run lifecycle and control | 1, 2, 3, 4, 6, 11, controller | [dx-run-lifecycle-and-control.md](../../plan/dx-run-lifecycle-and-control.md) |
| Configuration and bootstrap authority | 5, 9, 19, 20 | [dx-configuration-authority.md](../../plan/dx-configuration-authority.md) |
| Packing, environments, and datasets | 7, 8, 13, 14, 15, 16, 17 | [dx-packing-environments-datasets.md](../../plan/dx-packing-environments-datasets.md) |
| Public API and authoring surface | 10, 12, 18, code organization | [dx-public-api-and-authoring.md](../../plan/dx-public-api-and-authoring.md) |
| Release engineering | 21 | [dx-release-engineering.md](../../plan/dx-release-engineering.md) |

The delivery order above sequences work *across* the categories; each plan
sequences work *within* its category and states its own acceptance.

Cross-plan gates are explicit:

| Before this work | Required earlier milestone |
| --- | --- |
| Authoritative resolver or controller composition | Public API milestone 1: minimal `posttrain.project` application service |
| Activation resources and project-path environments | Public API milestone 2: `posttrain.environment` contracts and deterministic family assembly |
| Required-tracking implementation | Explicit unfreeze and amendment of the frozen post-training baseline |
| Lifecycle controller | Configuration milestones 1–2 and lifecycle milestones 2–4 |
| Dataset builders | Packing milestone 4: explicit materialization boundary |
| Constraint-free consumer install | Release milestone 3 and configuration milestone 5 |

These are dependency gates, not a requirement to complete an entire sibling
plan before returning to the current category.

## Acceptance for a later release

A later release can claim this critique materially addressed only when all of
the following are demonstrated from an installed external project:

- Two projects sharing one local worker hand off admission without using the
  wrong project config or receipt store.
- Mutating run commands require a complete canonical id; `--last` and prefixes
  are rejected.
- A submitted job reaches a reconciled logical terminal state after the
  submitting shell exits.
- Cancellation finalizes provider and tracking state without a manual recovery
  sequence in the ordinary case.
- `run show` joins control, provider, evidence, and reconciliation state.
- Local execution works without a remote project-image registry.
- `job plan` works before publication configuration and detects config changes
  before pack.
- The controller behaves identically whether or not the submitting shell had
  additional exported variables.
- One produced model can be consumed by exact run-output identity and promoted
  to the catalog without manually copying artifact metadata.
- The resolved snapshot and package identity contain the complete discovered
  catalog-family lock; a project referencing a missing family fails before
  decoding with a stable error.
- CLI JSON errors are stable enough for automation.
- An environment package living inside the project repository packs from the
  working tree by content digest, without a git commit or push.
- A declared activation resource has its digest generated into the pack lock
  and is staged, offline `Taskset.load()` succeeds during image qualification,
  and an undeclared project-relative `data_path` fails detached planning with
  an actionable error.
- A project with two in-repo environment packages and one declared dataset
  builder scaffolds, packs, and submits without hand-editing
  `[tool.posttrain.pack]`.
- A version-1 `layer.yaml` resolves identically through migration, version-2
  manifests support explicit excludes, and run snapshots preserve layer id,
  revision, manifest digest, selected files, and entry provenance.
- A dataset builder whose inputs changed rebuilds automatically during normal
  materialization and fails only under explicit frozen replay.
- A fresh `init` on a machine with a site profile reaches `job run` and `run
  show` without entering any tracking configuration beyond the selected
  backend's write token.
- A release is produced by changing only the release manifest and structured
  release inputs; generated metadata and captured receipts are reviewed, and CI
  fails when committed derived data does not match regeneration or receipts.

## Revision history

- 2026-07-31, revision 1: Created the release-scoped critique from the v0.2.5
  tag. Recorded lifecycle, configuration, state ownership, CLI/API, packing,
  catalog, artifact-continuation, and controller proposals. No implementation
  behavior was changed.
- 2026-07-31, revision 2: Added the authoring-surface axis from code-level
  review at the same tag: activation-resource packing (findings 13, 15, from
  the Ambient Agent production failure), project-path environment sources
  (14), project layout and pack-configuration derivation (16), dataset
  builders (17), dual-authoring asymmetry (18), Trackio-present defaults (19),
  and bootstrap transport (20). Strengthened findings 7 (push-always
  publisher), 8 (overlay glob vs work-package narrowing; veRL source
  requirement), and 12 (entry-hook signature and private loading policy).
  Extended the target journey, delivery order, and acceptance accordingly.
  Noted the uncommitted candidate fixes present in the working tree at
  assessment time. No implementation behavior was changed.
- 2026-07-31, revision 3: Added finding 21 (release versions and digests are
  authored, not generated) from analysis of the v0.2.5 release bump commit
  `9e65b2ba`, with a machine-generated-release target contract, a delivery-order
  item, and an acceptance criterion. No implementation behavior was changed.
- 2026-08-01, revision 4: Divided the findings into five delivery categories
  and authored one ExecPlan per category under `docs/plan/` (run lifecycle
  and control; configuration and bootstrap authority; packing, environments,
  and datasets; public API and authoring surface; release engineering).
  Added the "Implementation plans" section mapping findings to plans. No
  implementation behavior was changed.
- 2026-08-01, revision 5: Reviewed the five plans as one architecture. Chose
  frozen prepared submissions over project reload at admission time; exact ids
  for mutations; read-only run projections plus a separate reconciliation
  reducer; named site profiles; backend-neutral required tracking; generated
  resource locks; a dedicated environment-contract package and deterministic
  family assembly; and a generated release manifest instead of tag-derived
  builds. Reordered delivery around those dependencies. No implementation
  behavior was changed.
- 2026-08-01, revision 6: Recorded the installed catalog-family set as an
  immutable resolution and package input with loud missing-family failures;
  added a versioned, provenance-preserving overlay discovery migration and
  explicit excludes; made required tracking conditional on a frozen-baseline
  amendment; and tightened release drift checking around the sole authored
  version in `release/manifest.toml`. No implementation behavior was changed.
