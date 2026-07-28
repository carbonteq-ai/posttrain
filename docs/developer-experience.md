# Developer experience brief

**Status:** locked — 2026-07-23  
**Supersedes:** [developer-experience-audit.md](./developer-experience-audit.md)  
**Authority:** subordinate to [docs/post-training/](./post-training/README.md). This brief is the DX working authority for vocabulary and golden path. It does not change frozen primitive or job-kind meanings. Implementation follows [docs/plan/project-developer-experience.md](./plan/project-developer-experience.md). Baseline amendments are required before renaming public API types that appear in [05 · APIs](./post-training/05-apis.md).

## Intent

A developer with a model, data, and a use case should create a project, install the framework, run one tracked fine-tuning job, and open Observatory — all through `posttrain`, without importing `posttrain_lab` and without learning framework DI vocabulary.

The framework remains a construction kit underneath. The product surface hides that kit.

## Persona

One primary persona: **project developer**.

They care about project layout, catalog bindings, work packages, jobs, runs, and artifacts. They do not author “hosts,” “factories,” or composition roots on the happy path.

Framework maintainers and qualification authors are secondary. Their surface is `apps/lab`, not the quickstart.

## Vocabulary

| Prefer | Meaning | Avoid |
| --- | --- | --- |
| **project** | Repository with `.posttrain/` identity, overlays, and work packages | host, composition host |
| **project entry** | Optional Python hook that registers custom datasets/environments or overrides defaults | host factory, `create_host` |
| **job runtime** | Resolved registry used to execute one job: standard definitions, tracking, scratch | host context, `WorkPackageContext` as a developer term |
| **standard jobs** | Framework-shipped seat→operation definitions (`train/trl-sft@1`, …) | “copy definitions from lab” |
| **catalog** | Versioned selections (models, data, inference, targets, …) | trainer config files as the starting point |
| **work package** | Stage-scoped composition of bindings + jobs | pipeline, experiment |
| **job** | One executable unit; one run | “run the whole YAML” as recovery |
| **run** | One recorded execution of one job | provider-native run id as the product identity |
| **artifact** | Immutable published value with lineage | directory ancestry as lineage |
| **reference project / qualification suite** | What `apps/lab` is | reference host, required dependency |

Internal docs may still say that the CLI injects `RunContext`. Developer docs should not say “write a host.”

## Surfaces

| Surface | Owns | Does not own |
| --- | --- | --- |
| `packages/*` | Contracts, operations, adapters, **standard jobs**, **global/base catalog** publication | Project policy, thresholds, GSM8K/AutomationBench-only decisions |
| Project repository | Overlays, work packages, datasets/environments (when custom), accept/revise/reject | Framework internals, trainer types |
| `apps/lab` | Reference examples, backend integration tests, hardware qualification | Being imported by ordinary projects |
| `apps/cli` (`posttrain`) | Project init **including install**, diagnostics, plan/run/show, **Observatory bring-up** | Capability semantics |
| `apps/observatory` | Read-only evidence and analysis | Execution |

```text
packages (ops + standard jobs)
        ↓
project (.posttrain + optional Python)
        ↓
posttrain CLI (init+install / plan / run / show / observatory up)
        ↓
tracking → Observatory
```

`apps/lab` sits beside that path as CarbonTeq’s qualification project, not above it.

## Composition model

**Standard job definitions ship from the framework packages** (for example
`train/trl-sft@1`). They map catalog seats to typed package operations such as
`posttrain.train.sft`. Projects do not copy them from lab.

**Data and environment integration is framework-owned.** Popular dataset sources
(Hugging Face Hub, local JSONL/Parquet, and other adapters in `posttrain.data`)
and Verifiers environment bridges (in `posttrain.train` / `posttrain.eval`) are
wired **inside standard jobs**. A project registers a dataset or environment in
the catalog and binds it; the job resolves adapters and bridges automatically.
Project developers do not reimplement GSM8K-style source classes or
lab-only training bridges to use SFT or GRPO. If that glue stays in lab, the
framework has failed its purpose.

**Projects compose with YAML first; Python is the exception:**

| Path | What you write | When |
| --- | --- | --- |
| **YAML** | Catalog entries (model, dataset, env, settings, target) + work package bindings to a standard definition id | Default for SFT, DPO, GRPO, eval, serve |
| **Python (rare)** | Custom loader / env only when no shipped adapter or Verifiers env package covers it | Exotic formats or unpublished envs |
| **Python (custom definition)** | New versioned id that still calls package ops | Rare; different seats or glue |
| **Python (direct API)** | Call `posttrain.train.sft(...)` with no work-package definition | Scripts, notebooks, one-offs |

Customize a standard run by changing **bindings** (model, dataset, settings),
not by rewriting the definition. If wiring must differ, publish a **new**
definition id — do not shadow `train/trl-sft@1` with a different meaning.

## Where project knobs live

A **standard job** does not freeze learning rate, dataset, or hardware. It only
names seats and calls the package operation (including automatic data/env
bridges). Project developers register and tune values in the project catalog and
point at them from the work package.

| Concern | Where | Notes |
| --- | --- | --- |
| Register model / dataset / env / GPU | `.posttrain/catalog/` overlay YAML | Dataset entries declare source (HF/JSONL/…) and mapping; env entries declare Verifiers package + factory + sampling. Filenames are free; **family** matters. Trained descendants use `artifact.kind: trackio` — see **Trained model handoff** below |
| LR, steps, batch, LoRA / update plan | Catalog entries in family `training` | Prefer clear names like `sft_settings.yaml` |
| “Run SFT / GRPO with these picks” | `.posttrain/work_packages/*.yaml` bindings | Refs to catalog ids + standard definition id |
| Load HF/JSONL → trainer examples | **Inside** standard SFT/DPO jobs via `posttrain.data` | Not project Python |
| Env → rollouts / distill trajectories | **Inside** standard GRPO/distill/eval jobs via Verifiers bridges | Not lab bridges |
| How seats map to `train.sft` / `train.grpo` | Standard job id (framework) | Rarely edited by projects |
| Feed job A’s weights into job B | Overlay `ModelVariant` + new work package binding | Not an in-YAML `from_job` wire; see handoff below |

Prefer versioned catalog ids so runs stay comparable. Inline selections are
allowed for one-offs.

### Target catalog shapes (illustrative)

See **Dataset and environment layout** below for where files live versus where
catalog entries are declared.

```yaml
# work package — only picks; jobs supply adapters/bridges
bindings:
  dataset: { type: ref, family: dataset, id: datasets/support-sft@1 }
  # or for GRPO:
  # environment: { type: ref, family: environment, id: support-tool-use-grpo }
jobs:
  - id: train
    kind: train.sft
    definition: train/trl-sft@1
```

## Dataset and environment layout

Two different things must not be confused:

1. **Where the bytes / package live** (Hub, project `data/`, or an installed
   Verifiers env package).
2. **Where the project declares the selection** so the catalog and jobs can
   resolve it (always under `.posttrain/catalog/`).

### Datasets

| Kind | Where data lives | Where you register it | What the job does |
| --- | --- | --- | --- |
| Hugging Face (popular default) | Remote Hub repo at an **immutable** revision; local download/cache under `.posttrain/state/` (ignored) | `.posttrain/catalog/datasets.yaml` (family `dataset`) | Standard SFT/DPO job uses `posttrain.data` to load via the adapter |
| Local JSONL / Parquet | Tracked project files under `data/` (recommended) | Same catalog file; `source.path` relative to project root | Same — adapter reads the path |
| NeMo JSONL | Tracked project files under `data/` in NeMo row layout | Same catalog file; `source.kind: nemo` + relative `path` | Routes through `supervised_from_nemo` / `preferences_from_nemo` |
| Already-materialized snapshot | Tracking/artifact backend (digest), not hand-copied into catalog as lineage | Catalog may reference a published dataset artifact id when continuing a pipeline | Prepare/train consume the snapshot |

Recommended project tree for data:

```text
support-agent/
  data/                              # tracked SOURCE files only (optional)
    support_sft/
      train.jsonl                    # committed if not using Hub
      README.md                      # schema notes for humans
  .posttrain/
    catalog/
      datasets.yaml                  # REQUIRED registration for catalog.resolve
    state/                           # gitignored — HF cache, extracts, scratch
      cache/
      scratch/
```

Example registrations:

```yaml
# .posttrain/catalog/datasets.yaml

dataset:
  # Popular remote dataset — nothing under data/
  datasets/support-sft@1:
    revision: "1"
    source:
      kind: huggingface
      repo: org/support-conversations
      revision: abcdef0123456789…     # immutable
      split: train
    format:
      kind: messages   # or auto | prompt-completion | alpaca | sharegpt

  # Local files — path is relative to project root
  datasets/support-sft-local@1:
    revision: "1"
    source:
      kind: jsonl
      path: data/support_sft/train.jsonl
    format:
      kind: messages   # or auto | prompt-completion | alpaca | sharegpt

  # NeMo JSONL — same path rule; routes through NeMo adapters
  datasets/support-nemo-prefs@1:
    revision: "1"
    kind: preference
    source:
      kind: nemo
      path: data/support_prefs/nemo.jsonl
    format:
      kind: nemo-ranked   # or auto; supervised NeMo uses messages | auto
```

Rules:

- Catalog YAML is the registry the work package refs (`family: dataset`, id).
- Do **not** put large Hub downloads or recovery checkpoints in `data/`; those
  belong under `.posttrain/state/` or the tracking artifact store.
- Prefer Hub + immutable revision for shared/team datasets; use `data/` for
  small proprietary or fixture files that should be reviewed in Git.
- `posttrain dataset validate datasets/support-sft@1` checks the catalog entry
  and samples the source before a GPU job.

### Environments (Verifiers)

Environments are **not** folders of JSON under `.posttrain/`. A Verifiers
environment is a **Python package** (tasks, tools, rewards). The project only
**binds** a versioned use of that package in the catalog.

| Layer | Where | Who |
| --- | --- | --- |
| Env implementation | Published package (PyPI/git) or path dep, e.g. `my-env-v1` in `pyproject.toml` | Env authors / framework; same pattern as `automationbench_v1` / `gsm8k_v1` |
| Env binding (how this project uses it) | `.posttrain/catalog/environments.yaml` | Project developer |
| Shared bindings | Framework base catalog | Framework release |
| Bridge to GRPO/distill/eval | Inside standard jobs (`posttrain.train` / `posttrain.eval` integrations) | Framework — not lab copies |

Recommended project tree for envs:

```text
support-agent/
  pyproject.toml                     # depends on my-env-v1@pinned
  .posttrain/
    catalog/
      environments.yaml              # EnvironmentBinding entries
    work_packages/
      grpo.yaml                      # bindings.environment → catalog id
  # NOT: .posttrain/environments/implementations
```

If you are **authoring** a new Verifiers env for reuse, keep it as its own
package (sibling repo or workspace member like `environments/my_env_v1/`),
publish/pin it, then register bindings in projects. Do not nest the env
implementation inside `.posttrain/catalog/`.

Example binding (project overlay):

```yaml
# .posttrain/catalog/environments.yaml
environment:
  support-tool-use-grpo:
    revision: "1"
    category: agentic-tool-use
    source:
      package: my-env-v1
      repository: https://github.com/org/my-env
      revision: <immutable-commit>
    factory: my-env-training           # entry exposed by the env package
    sampling:
      max_tokens: 2048
      temperature: 1.0
    num_tasks: 8
    num_rollouts: 8
    parameters:
      # env-specific knobs (domains, seeds, …)
```

Work package:

```yaml
bindings:
  environment:
    type: ref
    family: environment
    id: support-tool-use-grpo
jobs:
  - id: grpo
    kind: train.grpo
    definition: train/trl-grpo@1
```

Rules:

- **Declare/register** the environment for a project in
  `.posttrain/catalog/environments.yaml` (or another overlay file with family
  `environment`).
- **Install** the env package via the project’s dependencies (`pyproject.toml`).
- **Implement** new envs as Verifiers packages, not as ad-hoc project scripts.
- Standard GRPO/distill/eval jobs load the binding and apply the framework
  Verifiers bridge automatically.

### Trained model handoff (produce → pin → rebind)

There is **no** work-package field that says “take the output of job `train`.”
Jobs in one recipe share **input** seats. To run eval/GRPO/qualify on a
trained descendant:

1. Run the producer package; retain the Trackio artifact `name` + immutable
   `version` (`vN`) from the run evidence.
2. Add a project catalog model with `artifact.kind: trackio` and a pinned Hub
   `base` (required today even for adapters).
3. List that YAML file in `.posttrain/catalog/layer.yaml`.
4. Bind `models/…@…` on a **new** work package and run that package.

```yaml
# .posttrain/catalog/models.yaml
model:
  models/my-agent-2b@sft-v3:
    artifact:
      kind: trackio
      project: my-trackio-project
      name: ambient-agent-2b-sft
      version: v3
    form: peft-adapter
    weight_precision: bf16
    family: qwen3.5
    parameters: 2000000000
    instruction_tuned: true
    renderer_contract: qwen3.5-tools@1
    base:
      kind: hub
      repo_id: Qwen/Qwen3.5-2B
      revision: 15852e8c16360a2fea060d615a32b45270f8a8fc
    capabilities:
      modalities: [text, image]
      native_context_window: 262144
      mtp: true
    parent: models/qwen3.5-2b@bf16
```

```yaml
# consumer work package
bindings:
  model: { type: ref, family: model, id: models/my-agent-2b@sft-v3 }
```

Step-by-step with CLI commands:
[consumer-setup §9](./consumer-setup.md#9-pass-one-jobs-model-into-the-next).
Storage and alias rules:
[ops/dstack-trackio/object-storage.md](../ops/dstack-trackio/object-storage.md).

### Full layout (datasets + envs + knobs)

```text
support-agent/
  pyproject.toml                 # posttrain* + optional env package pins
  data/                          # optional tracked local dataset files
    support_sft/train.jsonl
  .posttrain/
    project.toml
    catalog/
      layer.yaml                 # overlay list if used
      models.yaml
      datasets.yaml              # register dataset selections
      environments.yaml          # register env bindings (GRPO/eval)
      targets.yaml
      sft_settings.yaml
      lora.yaml
    work_packages/
      sft.yaml
      grpo.yaml
      qualify.yaml
    state/                       # gitignored caches and scratch
  # escape hatch only:
  # support_agent/datasets.py
```

How the catalog “picks it up”: `project.toml` lists `catalog_overlays` (default
`catalog/`). `posttrain` / `open_catalog` loads base + those overlays. Work
package bindings use `type: ref` + `family` + `id`. Jobs resolve the selection
and run adapters/bridges — they do not scan `data/` unless the catalog entry
points there.

## Catalog layers: global hub vs project

Projects should not reinvent shared models, datasets, environments, recipes, or
hardware targets. The composed catalog is layered:

```text
Global / published catalog(s)     ← install with the framework (and optional hubs)
        +
Project overlay (.posttrain/catalog/)  ← only what this project owns or overrides
        =
One logical catalog for resolve / list / run
```

| Layer | What it is | How a project gets it | Typical contents |
| --- | --- | --- | --- |
| **Global catalog (hub)** | Versioned published selections, starting as the packaged `posttrain-catalog` base and evolving into a richer CarbonTeq (and later multi-hub) distribution | Installed as a dependency; **automatically** composed by `open_catalog` / `posttrain catalog list` — no copy into `.posttrain/` | Shared models, HF dataset bindings, Prime/Verifiers env bindings, inference defaults, targets, recipes, baselines |
| **Project catalog** | Tracked overlay under `.posttrain/catalog/` | Authored in the project repo | Proprietary datasets, project env bindings, experimental settings, overrides of a global id |
| **Work package** | Bindings to ids from either layer | `.posttrain/work_packages/` | Which global/project ids fill seats for this stage |

**Discoverable:** `posttrain catalog list` / `show` lists global + overlay entries
and records `source_layer` (`base` / hub id / `overlay`). Developers browse the
hub through the CLI (and later a Hub UI) without vendoring YAML into every
repo.

**Usable:** a work package may ref a global id directly:

```yaml
bindings:
  model:
    type: ref
    family: model
    id: models/qwen3.5-2b@bf16          # from global catalog
  dataset:
    type: ref
    family: dataset
    id: datasets/ultrachat-sft@1        # from global catalog (or project overlay)
```

No need to re-declare that model in the project overlay unless you override it.
Overlay **wins** on the same id when the project must diverge; runs snapshot
which layer supplied each seat.

### Global declare → local resolve (first run)

A global catalog entry is a **pointer** (ids, revisions, package pins, format
metadata). It is not the bytes or the installed env on the developer machine.
The first time a project **validates or runs** a job that binds that id, the
framework must **resolve it locally** into this project’s environment and state.

| Global entry points at | Local resolution on first use | Where it lands |
| --- | --- | --- |
| HF dataset (repo + revision) | Download/materialize via `posttrain.data` | `.posttrain/state/` cache (gitignored) |
| Local-path dataset in an overlay | Open path under project root | Already local |
| Prime / Verifiers env package | Ensure pinned package is installed (or install from documented Hub URL / lock) | Project `.venv` / lockfile |
| Model weights on Hub | Download / snapshot per train/serve adapters | Cache / artifact backend policy |
| Inference / settings / target | Pure config — no fetch | Resolved selection in memory + run snapshot |

Rules:

1. **Discover** can list a global id before anything is fetched
   (`posttrain catalog list`).
2. **Validate / plan / first run** must perform local resolution (or fail with a
   clear “not materialized / not installed” error and a fix command).
3. Resolution is **idempotent** — later runs reuse `.posttrain/state/` and the
   project env; they do not re-copy catalog YAML into the overlay.
4. Run snapshots record the global id, revision, and provenance
   (`source_layer=base|hub`) plus digests/paths of what was materialized.
5. Offline / air-gapped machines need a prior materialize step or vendored
   cache; `doctor` should say when a bound global id is not yet local.

Illustrative flow:

```bash
posttrain catalog show dataset datasets/ultrachat-sft@1
# listed from global catalog — may not be on disk yet

posttrain dataset validate datasets/ultrachat-sft@1
# first resolve: fetch HF revision into .posttrain/state/…

posttrain work-package run .posttrain/work_packages/sft.yaml --job train
# uses local cache; no re-declare in project overlay
```

Optional explicit command (product intent): `posttrain catalog materialize`
(or materialize on `doctor --fix`) to pull everything referenced by a work
package before GPU allocation.

**Publishing to the global hub** (framework / platform owners):

- Add versioned entries to the published catalog distribution (today:
  `packages/catalog` base resources shipped as `posttrain-catalog`).
- Bump the catalog package version; projects pick it up by depending on that
  release (via `posttrain init` / lockfile).
- Later: additional hub packages (for example `posttrain-catalog-carbonteq`)
  declared in `project.toml` and composed automatically; optional remote Hub
  index remains a distribution concern, not a second resolve API.

**Project authors** still register local-only assets in `.posttrain/catalog/`
(see Dataset and environment layout). Prefer **consuming** global ids for
popular HF datasets and known Hub envs once they are published globally;
register locally only when the selection is project-specific or not yet in the
hub.

```text
posttrain catalog list --family model
posttrain catalog list --family dataset
# shows global hub entries + this project's overlay
```

## References, install, and discovery

Nothing is usable from a work package until it is **registered in some catalog
layer** (global hub and/or project overlay). Install makes code/bytes available;
the catalog makes them **selectable**. Discovery commands tell you what is
registered and whether it can resolve.

### Three layers (always)

```text
1) Install / fetch     → package, Hub revision, or HF bytes available
2) Catalog register    → global published catalog and/or .posttrain/catalog/
3) Bind in work package → family + id ref on a seat
```

Prefer registering popular shared assets once in the **global catalog**;
projects only overlay what they own. Global registration still requires
**local materialization** on first validate/run (see above).

| Question | Answer |
| --- | --- |
| Is it installed? | Dependency / Hub cache / `import pkg` |
| Is it registered? | `posttrain catalog list --family dataset\|environment` |
| Can jobs use it? | `posttrain catalog show …` + `dataset validate` / work-package validate |
| Is it bound for this run? | Work package `bindings` |

### Hugging Face datasets

1. **No project copy required** for Hub data.
2. **Register** in `.posttrain/catalog/datasets.yaml`:

```yaml
dataset:
  datasets/ultrachat-sft@1:
    revision: "1"
    source:
      kind: huggingface
      repo: HuggingFaceH4/ultrachat_200k   # Hub id
      revision: <immutable-commit-or-tag> # required for reproducibility
      split: train_sft
    format:
      kind: messages   # or auto | prompt-completion | alpaca | sharegpt
```

3. **Bind** `datasets/ultrachat-sft@1` in the work package.
4. At run/validate time the job uses `posttrain.data`; cache lands under
   `.posttrain/state/`.
5. **Know it works:** `posttrain catalog show dataset datasets/ultrachat-sft@1`
   and `posttrain dataset validate datasets/ultrachat-sft@1`.

Moving `revision: main` is not acceptable for qualification; pin an immutable
revision.

### Prime Environments Hub (Verifiers / PrimeRL ecosystem)

Hub envs are **installable Verifiers packages**, not catalog entries by
themselves.

1. **Install** into the project env (pin a version):

```bash
prime env install owner/env-name@1.2.3
# or uv add with the Hub wheel URL / documented pin
```

Record that pin in `pyproject.toml` / lockfile so CI and teammates reproduce it.

2. **Register** a binding in `.posttrain/catalog/environments.yaml` that points
   at the installed package + factory the env exposes:

```yaml
environment:
  hub/owner-env-name-grpo@1:
    revision: "1"
    source:
      kind: prime-hub            # or equivalent documented source kind
      owner: owner
      name: env-name
      version: "1.2.3"           # same pin as install
      package: env_name          # importable Python package after install
    factory: <factory-id>        # as documented by that env
    sampling: { max_tokens: 2048, temperature: 1.0 }
    num_tasks: 8
```

Until `kind: prime-hub` exists in decode, projects may use today’s
`source.package` / `repository` / `revision` shape **as long as** the installed
distribution matches. The DX rule is the same: Hub install ≠ catalog registration.

3. **Bind** that environment id in a GRPO/eval work package.
4. **Know it works:**

```bash
posttrain catalog list --family environment
posttrain catalog show environment hub/owner-env-name-grpo@1
python -c "import env_name"   # package importable
posttrain work-package validate .posttrain/work_packages/grpo.yaml
```

`posttrain doctor` should report missing env packages (catalog references an
import that is not installed).

### Custom environments that “exist in code”

Custom env **code** lives in a Python package (path dep, workspace member, or
published wheel)—not only as loose files the catalog scans.

| Situation | What you do |
| --- | --- |
| Env package in this repo (`environments/my_env_v1`) | Path-depend it from the project `pyproject.toml` |
| Env only in your app package | Expose a Verifiers factory/entry; depend on that package |
| Shared with others | Publish (Hub or internal index), then treat like Hub |

Then **register** the binding in `.posttrain/catalog/environments.yaml` with
`source.package` + `factory` (+ git revision if applicable). Until that YAML
exists, `catalog list --family environment` will **not** show it — code on disk
alone is invisible to work packages.

**Linked** means all of:

1. Package installable/importable in the project env.
2. Catalog entry present with matching `package` / `factory`.
3. Work package (or eval plan) refs that catalog id.
4. Validate/doctor pass (resolve + import check).

### How to see what the catalog has

```bash
posttrain catalog list
posttrain catalog list --family dataset
posttrain catalog list --family environment
posttrain catalog show dataset datasets/ultrachat-sft@1
posttrain catalog show environment support-tool-use-grpo
posttrain catalog validate
posttrain doctor
```

List = registered selections (**global hub + project overlay**). It does **not**
list every Hub dataset or every Prime env in the world—only what the installed
global catalog(s) and this project have declared. To add something new globally,
publish a catalog release; to add something project-local, write overlay YAML
(or use add helpers), then list/show/validate.

### Optional helpers

These land with the DX CLI:

- `posttrain dataset add hf|jsonl|nemo …` → writes a catalog dataset entry
- `posttrain environment add local …` → writes a Verifiers environment binding
- `posttrain catalog materialize [--work-package PATH]` → materializes datasets and
  preflights environments referenced by work packages
- `posttrain doctor --fix` → readiness plus the same materialize/preflight pass
- `posttrain job plan PATH` → alias for `work-package validate`
- `posttrain job run PATH --job ID` → alias for `work-package run`
- `posttrain run recover-cancelled-tracking RUN_ID` → narrowly repair a Trackio
  run stranded as `running` after its exact framework submission is already
  provider-terminal `cancelled`; writes a protected audit receipt, then requires
  an ordinary `run reconcile`
- `posttrain run show RUN_ID [--source SOURCE_ID]` → Observatory run view for the
  project’s tracking source (defaults to `{tracking}-{project_id}`)

Canonical nouns remain `work-package` and Observatory. The `job` / `run` commands
are thin aliases only; they do not introduce a second product vocabulary.
Recovery is not part of ordinary cancellation or reconciliation. It fails
closed unless provider handle, canonical run, project, Trackio provider run id,
and original start time all match, and it is idempotent only for an already
cancelled exact Trackio run.

## Code extension ladder

| Need | Use |
| --- | --- |
| Popular HF / JSONL / Parquet dataset | Catalog dataset entry; standard job uses `posttrain.data` |
| Published Verifiers environment | Catalog environment entry; standard GRPO/distill/eval job uses framework bridge |
| Same shapes, different hyperparams / model / GPU | Catalog YAML + work-package bindings |
| Format or env the framework does not ship yet | Prefer contribute adapter/env package upstream; else temporary project factory via `entry` |
| Different seat layout / glue | New versioned definition id |
| Ad-hoc script / notebook | Direct package API call |
| New algorithm for everyone | Framework package change |

Project `datasets.py` / `environments.py` are **escape hatches**, not the default
onboarding path.

## Golden path

```bash
posttrain init support-agent --template sft
cd support-agent
# init already created the layout and installed framework + project extras

posttrain dataset validate datasets/posttrain-sft-smoke@1
posttrain work-package validate sft.yaml
# CUDA release gate:
posttrain work-package run sft.yaml --job train

posttrain observatory up
```

`posttrain init` is bootstrap in one step: write `pyproject.toml`, lock or
constraints for the chosen release, selected extras (train/eval/serve/tracking),
`.posttrain/` layout, **and** create the project environment and install those
dependencies (wrap `uv sync` / documented wheelhouse install). There is **no**
separate `posttrain sync` command. Developers do not assemble the dependency
matrix by hand. If dependencies change later, use the project’s own package
manager commands (`uv sync` in the generated project) or re-run documented
install; do not add a second Posttrain install verb.

`posttrain observatory up` starts the read product for the current project
(local process or compose). It reads tracking settings from `project.toml`,
prints the URL, and does not require a separate `posttrain-observatory`
incantation on the happy path. Remote/shared Observatory URLs remain
configurable; `up` is the local default.

### Serving-capacity project

A serving screen has three independent configuration surfaces:

| Concern | Project location | Examples |
| --- | --- | --- |
| Product acceptance envelope | `.posttrain/project.yaml` | required context, minimum sustained aggregate output TPS, p95 TTFT/TPOT, failure rate |
| Comparable request population | catalog `workload` | representative corpus digest, fixed output budget, concurrency sweep, warmup/repetition and saturation policy |
| Backend and hardware search point | catalog `inference` + `target` | vLLM scheduler, MTP, KV-cache format, memory reservation, maximum sequences, GPU profile |

Do not put the 50 TPS example threshold into a workload or vLLM binding. It is
project policy. Conversely, concurrency is not a product threshold: the
workload sweeps it to find how a specific inference binding uses a specific
target.

The framework-shared `workloads/general-serving-32k-sweep@1` workload identifies
the checked-in `general-serving-v1` corpus by revision and content digest. It
contains GSM8K questions, HumanEval prompts, and reviewed first-party chat,
extraction, structured-output, and tool-use messages. Public answers, canonical
solutions, tests, and rewards are excluded because `serve.benchmark` measures
capacity rather than correctness. Exact-token synthetic requests remain a
separate controlled diagnostic cohort and are not merged into representative
capacity evidence.

One `serve.benchmark` job:

1. loads the selected inference engine once;
2. warms and measures each configured concurrency in order;
3. records per-request token and timing traces plus direct run/backend
   counters;
4. retains a typed terminal resource, unsupported, or failure boundary after
   successful lower points; and
5. writes one schema-versioned `serving-result` artifact without eligibility or
   a selected winner.

Observatory owns the calculator. Its run view chooses the highest-throughput
point satisfying context, latency, reliability, and evidence-completeness
constraints, then checks that point against the project throughput minimum. A
final valid point whose throughput is still rising is `unsaturated`, not
eligible. The work-package view only compares runs with the same requirements
digest, execution target, workload, corpus digest, cohort, and calculator
version. Mismatched runs remain visible as incomparable. Strict Pareto
membership maximizes aggregate output TPS while minimizing p95 TTFT and peak
VRAM among comparable eligible contenders.

The repository example is validated and opened with:

```bash
posttrain work-package validate .posttrain/work_packages/foundation_screen.yaml
posttrain work-package run .posttrain/work_packages/foundation_screen.yaml --job benchmark
posttrain observatory up
```

The run Overview is the place to inspect the full concurrency table, average
and p95 response length, request coverage, failure boundary, TTFT/TPOT,
throughput curves, GPU/VRAM/KV-cache telemetry, phase timing, and redacted
backend-native settings. Open the parent work package to compare contender
states and the strict Pareto frontier. Historical single-point runs are labeled
as compatibility evidence and never silently reconstructed into a canonical
multi-point sweep.

For a small Qwen 3.5 0.8B real-vLLM release gate:

```bash
POSTTRAIN_RUN_SERVE_GPU_INTEGRATION=1 \
POSTTRAIN_SERVE_GPU_VARIANT=mtp \
uv run --no-sync pytest packages/serve/tests/test_vllm_capacity_integration.py -q
```

The variant may be `standard`, `mtp`, `turboquant`, or `mtp-turboquant`.
The gate requires CUDA, the vLLM extra, and access to the immutable model
revision; it skips with a reason during the ordinary CPU test suite.

Properties:

1. Generated project runs from installed wheels before the developer reads `apps/lab`.
2. Default tracking and standard jobs come from the framework + `project.toml`, not from copied lab modules.
3. One command executes one job and creates one run.
4. Custom Python is an escape hatch for unshipped adapters/envs — not required for popular datasets or published Verifiers environments.
5. Lab is never on the import path of the starter template.
6. Setup and Observatory are primary CLI commands, not side manuals.
7. Standard jobs automatically apply `posttrain.data` adapters and Verifiers bridges; projects do not reimplement that glue.

### Project layout (happy path)

See **Dataset and environment layout** above for the full tree and rules.
Minimal SFT starter:

```text
support-agent/
  pyproject.toml
  data/…                         # only if local files (not required for Hub)
  .posttrain/
    project.toml
    catalog/
      datasets.yaml
      models.yaml
      targets.yaml
      sft_settings.yaml
      lora.yaml
    work_packages/
      sft.yaml
    state/                       # gitignored
```

### When Python is required

| Need | Module noun | Not |
| --- | --- | --- |
| Popular HF / JSONL / Parquet / shipped format | catalog `dataset` entry only | `datasets.py` |
| Published Verifiers env | catalog `environment` entry only | lab bridge copy |
| Truly custom format or unpublished env | temporary `datasets.py` / `environments.py` via `entry`, then upstream | host.py / lab fork |
| Override tracking | `[tracking]` in `project.toml` | hand-rolled observer wiring |
| Non-standard job definition | rare; new definition id or contribute upstream | fork lab |

If an entry point is needed:

```toml
# .posttrain/project.toml
[project]
id = "support-agent"
entry = "support_agent:configure"
```

```python
def configure(project):
    project.datasets(...)
    # standard jobs and tracking already applied from defaults
```

No `--host` on the common path. The CLI discovers `entry` from `project.toml`. An override flag, if kept, is `--entry MODULE:ATTR`.

## What `apps/lab` is for

Lab remains valuable as:

- Concrete SFT / GRPO / distillation / screen compositions used for release qualification
- Backend wiring proofs (TRL, vLLM, Verifiers, Trackio, W&B)
- Hardware-specific smoke and regression scenarios
- End-to-end tests that exercise real GPU paths

Lab is not:

- A dependency of CarbonTeq product projects
- The only way to execute training
- The home of generic work-package contracts (those stay in `posttrain.work`)
- The home of **standard** job definitions once those are extracted
- The production scheduler

Success metric: a new project completes the SFT golden path without importing `posttrain_lab`.

## Rename map (`WorkPackageHost*`)

Developer-facing rename first; code rename follows after a short compatibility window. Do not change job-kind or selection meanings.

| Current symbol / flag | Proposed | Notes |
| --- | --- | --- |
| `WorkPackageHostRequest` | `ProjectExecutionRequest` | Absolute project root, state dir, catalog, work-package path |
| `WorkPackageHostFactory` | `ProjectEntry` | `Callable[[ProjectExecutionRequest], JobRuntime]` |
| `WorkPackageContext` | `JobRuntime` | Definitions + catalog + executor/tracking binding |
| `--host MODULE:FACTORY` | discover `project.entry`; override `--entry` | Drop required `--host` on `work-package run` |
| `create_host` / `create_foundation_host` | `configure` / `posttrain_lab.entry:configure` | Lab host shim removed; use project entry |
| `posttrain_lab.host` | `posttrain_lab.entry` | Deleted after DX alignment |
| docs: “reference host” | “reference project” / “qualification suite” | Update `04-framework` via baseline amendment |
| docs: “command host” | “CLI” | `apps/cli` is the `posttrain` distribution |
| docs: “Host SDK” | do not use | Ship **standard jobs** + **project defaults** instead |
| fixture `tests/consumer/fixture/host.py` | `project.py` + `configure` | Done |

Compatibility: keep old type aliases and `--host` as deprecated synonyms for one release of the pre-1.0 line, then remove.

### What does *not* rename

- `JobDefinition`, `WorkPackage`, `Recipe`, job kinds (`train.sft`, …)
- `RunContext` (already the right runtime injection name for operations)
- Capability package public operation names
- Observatory product naming

## Non-goals

- Remote scheduler / multi-tenant execution platform
- Automatic checkpoint promotion or automatic “winners”
- Replacing Observatory with CLI-only analysis
- Moving GSM8K or AutomationBench **policy** into reusable packages (bindings and
  bridges are framework-owned; scenario thresholds stay project/lab)
- Rebuilding `posttrain.data` adapters or Verifiers bridges from scratch — wire
  the existing ones

## Locked decisions

- This brief is the DX working authority; the long audit is historical only.
- Remove “host” from developer vocabulary; keep a thin optional **project entry**.
- Ship **standard job definitions from packages**; projects use YAML first.
- **Dataset and Verifiers bridges live inside standard jobs** (`posttrain.data` +
  train/eval integrations). Catalog declares HF/JSONL/… datasets and Verifiers
  environments; jobs resolve them automatically. Lab-only bridges are not the
  product path.
- Project knobs (model, data, env, hardware, LR, LoRA) live in **catalog overlays**; work packages bind ids; standard jobs map seats **and** apply adapters/bridges.
- Code extension is for unshipped formats/envs or new techniques — not for ordinary dataset bring-up.
- Customize via bindings first; custom definitions get new ids; direct package APIs remain available.
- **`posttrain init` owns bootstrap:** template layout **and** install of framework and project extras (no separate `sync` command).
- **`posttrain observatory up`** brings up the read product for the current project.
- Prefer consuming **global catalog** ids for shared models/datasets/envs; use
  project overlay for proprietary or not-yet-published selections.
- Global catalog entries are pointers; **first validate/run materializes them
  locally** (state cache / project env) without copying YAML into the overlay.
- Format kinds in catalog YAML should align with existing adapter literals
  (`messages`, `prompt-completion`, `alpaca`, `sharegpt`, `auto`) rather than
  inventing a parallel vocabulary.
- Prove DX with external starters for SFT **and** at least one
  environment-backed job path (GRPO or distill/eval) without `posttrain_lab`.
- Keep lab as qualification/reference project.
