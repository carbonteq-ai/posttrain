# Agent guide

This repository is the primary workspace for the post-training framework. Agents
working here must treat the post-training product documents as the authority,
keep framework packages independent of concrete ML and tracking backends, and
leave enough written context that another agent can resume from the repository
without relying on chat history.

## Read this first

Start with `docs/post-training/README.md`. The six documents under
`docs/post-training/01` through `06` are the canonical product and design
baseline. They override older architecture documents, ADRs, prototype code, and
historical plans when those sources disagree.

Read only the additional canonical documents relevant to the task, in this
order:

1. `docs/post-training/01-workflow.md` for the screen, train, and qualify flow.
2. `docs/post-training/02-primitives.md` for reproducible selections.
3. `docs/post-training/03-work-and-evidence.md` for project, work-package, job,
   and run boundaries.
4. `docs/post-training/04-framework.md` for package ownership.
5. `docs/post-training/05-apis.md` for public names and contracts.
6. `docs/post-training/06-observation-and-lineage.md` for metrics, traces,
   artifacts, views, and lineage.

The baseline is frozen. If implementation work requires a different product
meaning, record the decision, make a narrow amendment to the canonical baseline
first, and then update the plan and code. Do not silently use current code as the
contract. Documents under `docs/architecture/` and older records under
`docs/decisions/` may be stale; use them for history and evidence only unless the
canonical docs explicitly delegate to them.

## Repository and package map

The current repository is a Python 3.12 `uv` workspace.

- `packages/common` owns framework-neutral identities, selections, artifact
  values, `RunContext`, and the smallest observation protocols.
- `packages/data`, `packages/serve`, `packages/eval`, and `packages/train` own
  reusable capability APIs and private backend adapters.
- `apps/observatory` is the dedicated read-only evidence product. It owns the
  job-aware query service, Python analysis, report exports, HTTP, MCP, and
  frontend surfaces over provider-neutral tracking readers.
- `packages/reports` is legacy migration input for Observatory and must be
  removed after its useful behavior and tests are ported. Do not add new report
  logic or preserve it as a parallel product.
- `apps/lab` is the reference composition host. It owns work-package execution
  and concrete observer wiring; it is not the only supported host.
- `catalog/` currently contains the framework base catalog.
  `.posttrain/catalog/` and `.posttrain/work_packages/` contain project
  overlays and work-package configuration; `.posttrain/state/` is ignored
  machine-local runtime state.
- Package-specific tests live beside their owning package under
  `packages/<name>/tests` or `apps/<name>/tests`.

Keep reusable capability packages independent. In particular, train, eval, and
serve must not import one another, `posttrain.common` must not import Trackio,
W&B, TRL, Verifiers, or vLLM, and no reusable package may import `apps/lab`.
Run `uv run lint-imports` after changing boundaries.

## Related repositories

Related source repositories are normally checked out as siblings of this tree.
Resolve their exact branch and commit before planning or changing them; the
immutable dependency pins in this repository and `uv.lock` are the executable
authority.

- `../trackio` is the CarbonTeq Trackio fork. Its `origin` is
  `carbonteq-ai/trackio` and its `upstream` is `gradio-app/trackio`. Put generic
  Trackio storage, query API, MCP, trace, and dashboard improvements there. Put
  post-training-specific job views in this repository. After a Trackio change
  is committed, update the exact Trackio pin and `uv.lock` here.
- `../trl` is the CarbonTeq TRL fork. Its `origin` is `carbonteq-ai/trl` and its
  `upstream` is `huggingface/trl`. Put generally reusable trainer/runtime fixes
  there; keep project selections, job policy, and reporting here. Derive the
  active fork SHA from `packages/train/pyproject.toml` and `uv.lock` because
  prose documentation can lag a pin update.
- `../automationbench` is the CarbonTeq AutomationBench compatibility fork. Its
  `origin` is `carbonteq-ai/AutomationBench` and its `upstream` is
  `zapier/AutomationBench`. Keep the Python compatibility declaration and any
  generally reusable benchmark fixes there; keep environment selection,
  category budgets, GRPO policy, and trace presentation in this repository.
  The Verifiers adapter is maintained in the external
  `carbonteq-ai/verifiers-environments` repository under
  `environments/automationbench_v1`; pin its full repository commit in the
  framework catalogs, dependency constraints, and lockfiles. Do not recreate
  the adapter under this framework repository.
- `../verl-upstream` is the current veRL candidate-fork checkout. Generic
  trainer, FSDP, rollout-lifecycle, and model-runtime fixes belong there;
  Qwen-only qualification policy, environment selections, and run evidence
  belong in this repository. Treat it as unpublished until a CarbonTeq remote
  and immutable fork commit replace the current dirty upstream checkout.
- `../dstack` is the current graceful-cancellation candidate fork. Generic
  server/runner stop-duration propagation and its regression tests belong
  there; framework admission, run policy, tracking finalization, and live
  qualification evidence belong in this repository. Treat the checkout as
  unpublished until a CarbonTeq remote and immutable fork commit replace the
  current dirty upstream-tag checkout.
- Verifiers and its environment packages are external Git dependencies pinned
  by `packages/eval/pyproject.toml` and `packages/data/pyproject.toml`. Their
  native traces remain the replay authority. Do not copy environment ownership
  into this workspace.
- Other upstream repositories and examples, including NeMo RL, are research
  inputs rather than framework contracts unless a plan records and validates a
  deliberate adoption.

Do not mix uncommitted changes across repositories implicitly. A multi-repo plan
must name the repository for each edit, define the order of commits and pin
updates, and provide validation commands for every affected repository.

All maintained forks follow `docs/tooling/forks.md`. Update the consumer page
at `docs/tooling/<tool>/README.md` and the fork's root `CARBONTEQ_FORK.md` in the
same logical change. The fork ledger owns its upstream base, maintained delta,
regression tests, compatibility constraints, rebase procedure, and published
commit; the consumer page owns framework selection, operating configuration,
qualification evidence, and remaining release gates. Commit and push the fork
before updating an immutable dependency pin or describing the fork as
reproducible.

## Creating and maintaining plans

Use `docs/templates/PLAN.md` for every implementation plan. Save active plans as
`docs/plan/<short-descriptive-name>.md`; create `docs/plan/` when it is absent.
Historical plans outside that directory may be referenced for evidence, but a
new plan must remain understandable on its own.

Before writing a plan:

1. Read this file and the complete plan template.
2. Read the canonical post-training documents governing the change.
3. Inspect the current code, tests, dependency manifests, lockfiles, and any
   sibling repositories the work will touch.
4. State whether the work changes the frozen product baseline. If it does,
   include the baseline amendment before implementation.
5. Resolve prototype-versus-production intent, external service requirements,
   migrations, rollout constraints, and validation expectations in the plan.

Every plan is a living execution document for a reader with no chat history. It
must include and maintain `Progress`, `Surprises & Discoveries`, `Decision Log`,
and `Outcomes & Retrospective`; define unfamiliar terms; name exact files and
interfaces; give commands with working directories; describe observable
acceptance; and record safe retry or recovery steps. Update the same plan as
implementation proceeds instead of creating disconnected status notes.

Plans that introduce an external backend must include at least one real
integration path in addition to fakes. Record credentials and network
assumptions without storing secrets. When two backends implement one contract,
test equivalent logical results rather than identical provider storage.

## Implementation and validation defaults

Preserve the user's existing dirty worktree and do not revert unrelated edits.
Use `apply_patch` for hand-authored changes. Prefer additive migrations with a
short, explicit compatibility window, then remove the old path once all callers
and tests use the new contract.

From the repository root, the normal validation ladder is:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

During development, run the smallest package-specific tests first. Mark tests
that require network, Docker, or GPU with the existing pytest markers. A test
that requires credentials must skip with a clear reason when they are absent,
but its documented release-gate command must be run before claiming that the
integration is complete.

Use compatible dependency ranges in `pyproject.toml` and exact resolutions in
`uv.lock`. Git dependencies must use immutable commits. JavaScript applications
must commit their lockfile and validate both tests and a production build.
