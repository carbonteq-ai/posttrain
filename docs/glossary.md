# Glossary

One-stop definitions for every term of art used across the README and docs.
Each entry links to the document that owns the full contract; if this page and
a baseline document ever disagree, the
[baseline](./post-training/README.md) governs.

**Admission** — The machine-scoped ledger that serializes local jobs against
physical resources (GPUs). Every registered project on a machine shares the
same ledger; `posttrain workers` shows who holds a placement. dstack runs do
not take a local admission lock. Contract:
[03 · Work and evidence](./post-training/03-work-and-evidence.md).

**Binding** — A work package's assignment of a catalog selection to a recipe
seat, usually by reference (`type: ref`) rather than by copied values. See
[02 · Primitives](./post-training/02-primitives.md).

**Catalog** — The layered store of versioned selections (models, datasets,
environments, inference profiles, training methods, workloads, execution
targets). The framework ships a base layer in `posttrain-catalog`; a project
adds overlay layers with only its own selections and overrides, and the
resolved composition is inspectable with `posttrain catalog list/show/validate`.
See [02 · Primitives](./post-training/02-primitives.md).

**Catalog layer / overlay** — One versioned set of selection files declared by
a `layer.yaml`. Project overlays compose on top of the framework base layer;
they never copy it. See [Developer experience](./developer-experience.md).

**Controller** — `posttrain controller run`, a reconciling loop that submits
queued work when capacity frees, delivers cancellations, finalizes tracking
evidence, releases settled admissions, and writes recovery receipts — so jobs
outlive the submitting shell.

**dstack** — An external GPU orchestrator used as Posttrain's remote execution
provider. Posttrain submits a resource ask; dstack owns offers, placement,
startup, and cancellation. Setup: [Getting started §6](./getting-started.md#6-run-on-dstack).

**Environment** — A versioned, independently installable package of executable
evaluation/RL tasks (a Verifiers package), such as `gsm8k-v1` or
`automationbench-v1`. Projects can own their environments
(`posttrain environment new`). See [02 · Primitives](./post-training/02-primitives.md).

**Evidence** — The retained, immutable record of a run: metrics, native
traces, configured success criteria, errors and truncations, and consumed/
produced artifact edges. Evidence is explored through Observatory and its
meaning never changes after the fact. Contract:
[06 · Observation and lineage](./post-training/06-observation-and-lineage.md).

**Host** — *Legacy term.* Older docs use "host" for the application embedding
the framework (and "reference host" for `apps/lab`). Current vocabulary:
a thin optional **project entry** replaces the host concept, and `apps/lab` is
the **reference project** and qualification suite. The term still appears in
[05 · APIs](./post-training/05-apis.md); read it there as "the embedding
application or work-package runner."

**Job / job kind** — One executable unit inside a work package, identified by
a kind such as `train.sft`, `eval.general`, or `model.transform`. Standard
definitions ship in `posttrain-jobs`. Contract:
[03 · Work and evidence](./post-training/03-work-and-evidence.md).

**Lineage** — The provenance graph formed by runs' consumed/produced artifact
edges (plus optional catalog `parent` pointers) — how a model variant traces
back to the exact run, data, and base weights that produced it. Contract:
[06 · Observation and lineage](./post-training/06-observation-and-lineage.md).

**Model variant** — An immutable catalog entry describing concrete weights (a
Hub snapshot, adapter, merged fine-tune, or quantized transform) with form,
precision, renderer contract, and provenance. See
[02 · Primitives](./post-training/02-primitives.md) and
[Getting started §9](./getting-started.md#9-pass-one-jobs-model-into-the-next).

**MTP (multi-token prediction)** — A serving acceleration where the model
proposes several tokens per step, natively or through a paired assistant
(draft) model; runs record speculative draft/accepted and KV-cache metrics.
See the MTP amendment in the [baseline](./post-training/README.md).

**Observatory** — The read-only evidence product (`posttrain observatory up`):
coverage, pass rates, rewards, latency, distributions, traces, comparisons,
Pareto frontiers, and lineage over recorded runs. It never mutates evidence.
See [04 · Framework](./post-training/04-framework.md).

**Plan / pack / run** — The three-step job lifecycle. `job plan` resolves all
selections without contacting a provider; `job pack` builds a
content-addressed OCI image from the plan and fails if any input drifted;
`job run` submits the packed job. `job diff` explains identity differences
between two packed jobs.

**Project** — An installable repository created by `posttrain init`, owning
its `.posttrain/` configuration: `project.toml` identity, catalog overlays,
and work packages. Policy stays in the project; the framework stays
replaceable underneath it.

**Project entry** — An optional `entry` hook in `project.toml` that registers
unshipped job definitions for a project without redefining standard ids. The
modern replacement for the legacy "host" concept. See
[Developer experience](./developer-experience.md).

**Qualification** — Two related but distinct uses. (1) The **qualify stage**
of the workflow: deciding whether a trained variant is ready for its intended
use ([01 · Workflow](./post-training/01-workflow.md)). (2) **Release/hardware
qualification**: the evidence-backed gates a release candidate or a
model/target combination must pass before promotion
([Release engineering](./release-engineering.md)).

**Recipe** — The ordered composition of jobs a work package executes, with the
seats those jobs require. Standard recipes ship with the framework; `inline`
recipes are declared in the work package itself.

**Reference project** — `apps/lab`: the framework's own Posttrain project,
carrying qualification scenarios and backend release gates. Formerly called
the "reference host." Installed as `posttrain-lab`.

**Run** — One tracked execution of a job: resolved identity in, evidence out.
Runs are provider-neutral (local Docker or dstack) and durable — `posttrain
run status/wait/logs/cancel/reconcile` operate on them after the submitting
shell exits. Contract: [03 · Work and evidence](./post-training/03-work-and-evidence.md).

**Screen** — Two distinct uses. (1) The **screen stage** of the workflow:
deciding whether a model/runtime combination is worth pursuing
([01 · Workflow](./post-training/01-workflow.md)). (2) In Observatory and DX
docs, a configured UI view, as in "serving screen"
([Developer experience](./developer-experience.md)).

**Seat** — A named slot a recipe requires (for example `model`, `dataset`,
`target`) that a work package fills through its bindings. Naming contract:
[05 · APIs](./post-training/05-apis.md).

**Selection** — One versioned entry in a catalog family, addressed by id and
revision (for example `models/qwen3.5-2b@bf16`, `targets/local-cpu` at
revision `"1"`). Work packages reference selections; they do not copy their
values. See [02 · Primitives](./post-training/02-primitives.md).

**Stage** — One of **screen**, **train**, **qualify** — the three phases of
the workflow a work package belongs to. See
[01 · Workflow](./post-training/01-workflow.md).

**Trackio** — The default local tracking backend (a CarbonTeq-maintained
fork), recording runs, metrics, traces, and artifacts behind a
provider-neutral contract; W&B is the alternative backend. Notes:
[tooling/trackio](./tooling/trackio/README.md).

**Training methods** — `SFT` supervised fine-tuning; `DPO` direct preference
optimization; `GRPO` group-relative policy optimization (online RL); `DAPO` a
GRPO algorithm variant selected within `train.grpo`; `SAMPO` a multi-turn
method run through `train.sampo`; on-policy **distillation** through
`train.distill`. Backends are TRL or maintained veRL profiles. See
[techniques/](./techniques/).

**Verifiers** — An external library and packaging convention for versioned,
executable evaluation/RL environments; Posttrain's environments are Verifiers
packages, and the same environment serves evaluation and online RL. Its own
concepts (Taskset, Harness, EnvConfig) are upstream vocabulary. Notes:
[tooling](./tooling/).

**Wheelhouse** — The release bundle attached to a GitHub Release: the exact
wheels plus `github-constraints.txt` pinning maintained forks to immutable
commits. An offline, auditable install surface for one release. See
[install.md](./install.md).

**Work package** — The central unit of work: one decision-making question,
its ordered jobs, the catalog selections they bind, and the evidence needed to
understand or reproduce the outcome. Authored as YAML under
`.posttrain/work_packages/`, validated with `posttrain work-package validate`.
Contract: [03 · Work and evidence](./post-training/03-work-and-evidence.md).

**Workload** — A versioned serving-benchmark population (prompts and decode
settings), such as `general-serving-v1`; materialized and verified with
`posttrain workload materialize/verify`. See
[02 · Primitives](./post-training/02-primitives.md).
