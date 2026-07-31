# One authoritative configuration model, from install to job secrets

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

Source findings: `docs/dx-improvements/v0.2.5/README.md` findings 5, 9, 19,
and 20. This plan is self-contained.

## Purpose / Big Picture

Posttrain behavior today depends on which shell you are standing in. Project
runtime values live in an ignored `posttrain.env` file that historically had to
be sourced; a manual pointer inside ignored state (`.posttrain/state/
execution.toml`, key `environment_file`) tells the framework where it is;
machine concerns (dstack client interpreter, storage roots, trust bundle) sit
in the same ignored file as authoritative control receipts; and job secret
*names* must be repeated by hand on the command line even when the selected
environment already declares which variables it needs. Separately, tracking is
modeled as optional and configured repeatedly even though managed deployments
already provision a site tracking backend, so each project re-enters the server
URL and token.

After this plan, `posttrain init` produces a project where no sourcing, no
manual pointer, and no repeated `--env` flags are needed; a background process
resolves exactly the same configuration as an interactive shell; the site's
tracking endpoint is inherited from a named site profile; and `posttrain job
plan` prints which runtime variables the selected job requires and which are
already satisfied.

## Progress

- [x] (2026-08-01) Plan authored from the v0.2.5 release-scoped critique.
- [x] (2026-08-01) Cross-plan architecture review completed; runtime, site
      profile, tracking policy, and install transport decisions revised.
- [x] (2026-08-01) Follow-up review identified the required-tracking proposal
      as a frozen-baseline amendment and added the governance gate.
- [x] (2026-08-01) Milestone 1: `posttrain.env` loads automatically and
      authoritatively. The dstack bridge injects declared job variables only
      from the resolved project map, never from its parent shell; a legacy
      `environment_file` pointer remains readable with a warning for one
      release before its scheduled removal.
- [x] (2026-08-01) Added public redacted runtime-environment resolution;
      `posttrain.env` now wins over shell values and the legacy pointer for
      local configuration and registry derivation. `init` writes protected
      `posttrain.env`, tracked `posttrain.env.example`, and ignores the former.
- [ ] Milestone 2: named site profiles at `$XDG_CONFIG_HOME/posttrain/config.toml`
      and the state-directory split.
- [ ] Milestone 3: required tracking with a site-selected backend.
- [ ] Milestone 4: runtime variables derived from selections.
- [ ] Milestone 5: install transport (constraints file, trust, blessed
      commands) folded into the distribution and `doctor`.

## Surprises & Discoveries

- Observation: the uncommitted working tree already prevents the registry from
  falling back to ambient variables once an `environment_file` is configured,
  but it retains the pointer under `.posttrain/state/execution.toml`.
  Evidence: the current diff passes an explicitly loaded environment mapping to
  `derived_registry()` and adds `posttrain.env` to scaffolded `.gitignore`.
- Observation: a profile named only for providers cannot honestly own tracking,
  trust, and storage defaults too.
  Evidence: v0.2.5 `execution.toml` already combines all four concerns, which
  is the mixed authority this plan is intended to remove.
- Observation: making durable tracking required changes a frozen product rule,
  not merely a configuration default.
  Evidence: `docs/post-training/03-work-and-evidence.md` currently defines
  terminal provider evidence as the admission barrier when a project selects
  the no-op observer, and `docs/post-training/README.md` requires unfreeze →
  baseline update → plan/code for behavior changes.
- Observation: seven execution-configuration tests encoded ambient
  `POSTTRAIN_REGISTRY` fallback rather than a project runtime source.
  Evidence: after the resolver stopped reading `os.environ`, those fixtures
  returned no registry until their values were moved to `posttrain.env`; a
  dedicated conflicting-shell regression test now covers the intended rule.
- Observation: dstack's isolated SDK process deliberately inherits parent
  variables so its client can authenticate, but it previously reused that
  process map to resolve job variables.
  Evidence: `sdk_bridge.py` translated the requested `env` name list directly
  from `os.environ`; a regression now proves its private runtime payload wins
  over a conflicting shell token and fails closed when the project map lacks a
  declared name.

## Decision Log

- Decision: ambient shell variables must not override project runtime values;
  the only override is an explicit `--env-file PATH`.
  Rationale: a controller or CI run must behave identically to the submitting
  shell; silent shell precedence is the root cause of "works on my terminal".
  Date/Author: 2026-08-01 / plan author.
- Decision: ambient process variables do not fill missing project runtime
  values either.
  Rationale: fallback still makes behavior depend on the launching shell. CI
  writes a protected env file or passes `--env-file`; process variables such as
  `PATH` and `XDG_STATE_HOME` remain process configuration, not job runtime.
  Date/Author: 2026-08-01 / architecture review.
- Decision: use one named site profile in `config.toml`, not a provider-only
  file that also accumulates tracking, trust, and storage settings.
  Rationale: provider identity and tracking identity are independent concerns,
  but an operator needs one coherent machine profile that selects both.
  Date/Author: 2026-08-01 / architecture review.
- Decision: `posttrain.env` is a string-to-string runtime value map, not a
  second structured configuration format.
  Rationale: typed structure remains in TOML/YAML schemas; environment values
  must work identically for YAML-authored and Python-authored jobs and are
  validated only when bound to a declared runtime variable.
  Date/Author: 2026-08-01 / architecture review.
- Decision: require durable tracking for managed jobs without hard-wiring
  Trackio into core contracts.
  Rationale: the current deployment chooses Trackio, while the framework also
  supports W&B. Policy should require evidence; the site profile chooses the
  conforming backend.
  Date/Author: 2026-08-01 / architecture review.
- Decision: no milestone-3 implementation may land until the frozen baseline is
  explicitly unfrozen, amended, reviewed, and refrozen.
  Rationale: silently changing the scaffolded default and admission policy from
  an implementation plan would invert the repository's documented authority.
  Date/Author: 2026-08-01 / architecture review follow-up.
- Decision: keep `execution.toml.environment_file` readable for one release,
  but select project-root `posttrain.env` first and emit a deprecation warning
  only when the legacy pointer is actually used.
  Rationale: existing projects remain runnable during migration without
  letting their ignored state file outrank the tracked project layout.
  Date/Author: 2026-08-01 / implementation.
- Decision: send resolved project runtime values to dstack only in a private
  bridge payload, immediately before the isolated SDK process runs.
  Rationale: the SDK process may still inherit host credentials, but plan
  output, durable submissions, and gateway-visible configuration must retain
  variable names—not values—and the job must never source a declared value from
  that parent environment.
  Date/Author: 2026-08-01 / implementation.

## Outcomes & Retrospective

- Milestone 1 outcome: fresh projects load root `posttrain.env` without shell
  sourcing; `--env-file` is the only one-off override; local and dstack
  execution consume that resolved map only. The one-release legacy-pointer
  reader is the sole compatibility exception and warns on use.
- Planning review outcome: project runtime, site configuration, and process
  environment now have non-overlapping authority. Tracking is required by
  policy but backend-neutral. Later implementation outcomes remain pending.

## Context and Orientation

The target authority table, which this plan implements:

    Project identity, catalog paths, work packages   .posttrain/project.toml      (tracked)
    Runtime endpoints, job secrets, provider profile  posttrain.env (repo root)   (ignored, 0600, auto-loaded)
    Named provider, tracking, storage, trust profile  $XDG_CONFIG_HOME/posttrain/config.toml
    Submit/cancel intent, handles, receipts           .posttrain/state/executions/
    Materialized datasets, build caches               .posttrain/state/cache/

The concrete machine and project selection shape is:

    # $XDG_CONFIG_HOME/posttrain/config.toml
    [profiles.rtx96]

    [profiles.rtx96.provider]
    kind = "dstack"
    project = "posttrain"
    python = "/opt/dstack/venv/bin/python"
    credentials_file = "/etc/posttrain/dstack.env"

    [profiles.rtx96.tracking]
    kind = "trackio"
    endpoint = "https://trackio.example.internal"

    [profiles.rtx96.trust]
    ca_bundle = "/etc/ssl/certs/ca-certificates.crt"

    # <project>/posttrain.env
    POSTTRAIN_PROFILE=rtx96
    TRACKIO_WRITE_TOKEN=...

`POSTTRAIN_PROFILE` is read from the selected env file, never from the ambient
shell. The profile stores no token values; it stores stable identities and
protected credential-file locations.

Key files today:

- `apps/cli/src/posttrain_cli/execution_config.py` (~1,060 lines) — loads
  `.posttrain/state/execution.toml`, which currently mixes the machine binding
  (local canonical hostname, dstack client python, storage paths, trust
  bundle) with the `environment_file` pointer. The uncommitted working tree
  already makes a configured `environment_file` authoritative over shell
  exports; this plan supersedes the pointer entirely.
- `apps/cli/src/posttrain_cli/tracking_config.py` — resolves the tracking
  backend per project.
- `apps/cli/src/posttrain_cli/scaffolding/init_project.py` — `initialize()`
  writes the project layout and `.posttrain/state/execution.toml`.
- `docs/consumer-setup.md` — documents today's manual steps (hand-written
  `posttrain.env`, the relative `environment_file` pointer resolved from
  `.posttrain/state/`, `--system-certs` on every uv call, fetching
  `github-constraints.txt` out of band). Update it as each milestone removes a
  step; the doc's own rule is that every command was executed for real before
  being written.
- `docs/post-training/README.md` and
  `docs/post-training/03-work-and-evidence.md` — frozen product authority. The
  latter explicitly defines the no-op-observer admission barrier; they must be
  amended before milestone 3 changes default tracking policy.
- Job secret handling: execution accepts variable *names* (never values) via
  job defaults and repeated `--env NAME` flags; selections such as an external
  evaluation service or an environment package already know which variable
  names they need but do not declare them.

## Plan of Work

Milestone 1 makes the root `posttrain.env` a first-class, automatically
discovered file: `init` creates it empty with mode 0600, adds it to
`.gitignore`, and writes a tracked `posttrain.env.example` listing variable
names only. Configuration loading reads it whenever the project root contains
it — no pointer, no sourcing. Precedence becomes: explicit `--env-file` flag,
else project `posttrain.env`, else no project runtime values. Ambient process
variables neither override nor fill the project runtime namespace. Remove the
`environment_file` key after a deprecation release that warns when it is set.

Milestone 2 introduces named site profiles in
`$XDG_CONFIG_HOME/posttrain/config.toml`, owned by the operator. Each profile
selects provider identity, provider executables (dstack client Python), storage
roots, trust-bundle location, and a tracking backend/endpoint. A project selects
the site profile by id from `posttrain.env`; provider and tracking locators are
still stored separately on each submission.
`.posttrain/state/` is reorganized into `executions/` (control receipts —
protect and retain) and `cache/` (safe to delete); `execution.toml` shrinks to
machine-binding leftovers and is then folded into the profile. Provide a
`posttrain state migrate` command that performs the split idempotently.

Milestone 3 makes durable tracking the default policy for managed training and
evaluation. The selected site profile supplies either Trackio or W&B plus its
endpoint; the fresh project needs only that backend's write token in
`posttrain.env`. `doctor` verifies reachability with an actionable failure. The
no-op observer becomes an explicit development waiver (`tracking = "none"` in
`project.toml`) rather than a branch every lifecycle rule accommodates. Add
`tracking = "site"` as the scaffolded project default; retain existing explicit
`trackio` and `wandb` selections through migration, then document them as
project-level backend constraints rather than repeated endpoints.

Milestone 3 begins with a product-governance gate, before code: explicitly
unfreeze the post-training baseline; add a dated amendment to
`docs/post-training/README.md`; update the no-op-observer rule in
`docs/post-training/03-work-and-evidence.md` so durable evidence is required for
managed jobs and `none` is a deliberate development waiver with its explicit
provider-terminal barrier; reconcile affected language in `04-framework.md`,
`05-apis.md`, and `06-observation-and-lineage.md`; review and refreeze the
baseline. If that amendment is declined, keep tracking optional and implement
only site-supplied defaults—the plan must not smuggle required tracking into
code.

Milestone 4 adds a secret-free `required_runtime_variables` contract to the
selection and job-definition types that already imply requirements (external
service descriptors, environment bindings, tracking backends). Planning
resolves the union for the selected job and prints a table of required
variables with present/missing status plus unused project variables; only
declared variables are forwarded into the job container; provider client
credentials are never forwarded.

Milestone 5 fixes install transport at its source. Publish maintained fork
wheels with immutable versions to the internal index and generate exact normal
package dependencies, so consumers do not need `github-constraints.txt` before
Posttrain itself can be installed. Keep the constraints file as a release-build
input until all forks are indexed. Ansible installs the internal CA into system
trust and configures uv native TLS once at machine scope. `doctor` verifies
those conditions and prints the operator repair command; it does not mutate
system trust. Update `docs/consumer-setup.md` to one blessed install command and
delete superseded per-command flags.

## Concrete Steps

Work from the repository root.

    uv run pytest apps/cli/tests -q

Milestone 1 needs tests proving: conflicting or missing values in
`posttrain.env` are never supplied by exported shell variables; `--env-file`
replaces the project file explicitly; and a scaffolded project contains
`posttrain.env` (0600, ignored) and `posttrain.env.example` (tracked).
Milestone 4 needs a test that a work package selecting an external
evaluation service reports its declared `API_KEY`-style variable as required,
and that an undeclared variable is not forwarded.

For milestone 2, verify the migration is idempotent:

    posttrain state migrate
    posttrain state migrate   # second run: reports "nothing to do", exit 0

Before any milestone-3 implementation commit, perform and record this sequence:

    1. Open the baseline amendment with finding 19 and the existing no-op
       admission text as evidence.
    2. Mark the baseline explicitly unfrozen for this decision.
    3. Update README, 03, and any dependent 04–06 contracts.
    4. Obtain review of the resulting product behavior and migration policy.
    5. Refreeze the baseline, then implement `tracking = "site"`.

The plan's `Progress`, `Decision Log`, and `Artifacts and Notes` sections must
link the accepted amendment before milestone 3 is marked in progress.

## Validation and Acceptance

- A fresh `posttrain init x --template sft` followed by editing
  `posttrain.env` reaches `posttrain doctor` all-OK with no `source`, no
  pointer edit, and no exports.
- The same command sequence run under `env -i PATH=...` (an empty
  environment) resolves identical configuration — proving a future controller
  inherits nothing from a shell.
- On a machine with the site profile installed, a new project reaches
  `job run` and `run show` entering only the selected tracking backend's write
  token.
- The accepted frozen baseline contains a dated amendment for required managed
  tracking, retains the explicit semantics of the no-op development waiver,
  and was merged before the first implementation change to `tracking = "site"`.
- `posttrain job plan` output contains a "Required runtime variables" table
  with per-variable presence, and submission forwards exactly that set.
- A clean-machine install succeeds from the internal index with ordinary
  package metadata, no out-of-band constraint file, and no repeated TLS flag.

## Idempotence and Recovery

All milestones are additive with deprecation windows: the `environment_file`
pointer keeps working for one release with a warning; `state migrate` is
re-runnable and never deletes receipts. It first copies and validates the new
layout, writes a migration receipt, and only then leaves compatibility links or
read fallbacks for the old layout. Rollback selects the old reader while the
copied receipts remain intact; it does not ask a developer to move control
files by hand.

## Artifacts and Notes

Capture, once milestone 4 lands, one real `job plan` transcript showing the
required-variables table, as an indented block here.

## Interfaces and Dependencies

Define one typed resolver as the single entry point in the public
`posttrain.project` package created by
`docs/plan/dx-public-api-and-authoring.md` milestone 1. Do not create a
temporary resolver in `packages/execution`:

    @dataclass(frozen=True)
    class ResolvedConfiguration:
        project: ProjectSettings          # from .posttrain/project.toml
        runtime: RuntimeValues            # from posttrain.env / --env-file
        machine: MachineProfile           # selected from config.toml
        source_report: list[SourceEntry]  # where every value came from

    def resolve_configuration(project_root: Path, env_file: Path | None = None) -> ResolvedConfiguration: ...

Every consumer — CLI commands, packing, submission, the controller — obtains
configuration through this function and nothing else. `doctor` prints the
`source_report` so precedence is inspectable instead of folklore.
`RuntimeValues` must redact its representation and serialization by default;
`source_report` records only variable names, presence, and source file, never
values. Only the execution boundary may request the declared subset as a plain
mapping for job injection.

Selections that need runtime variables implement:

    class RequiresRuntimeVariables(Protocol):
        def required_runtime_variables(self) -> frozenset[str]: ...

## Revision Notes

- 2026-08-01: Architecture review removed ambient-variable fallback, replaced
  `providers.toml` with named site profiles in `config.toml`, made durable
  tracking backend-neutral but required for managed jobs, moved clean-machine
  installation toward indexed fork wheels and machine-level TLS bootstrap, and
  made the state migration copy-and-verify rather than a manual move.
- 2026-08-01: Follow-up review added the mandatory unfreeze → baseline
  amendment → review/refreeze gate before required-tracking implementation and
  preserved optional tracking as the fallback if that product amendment is
  declined.
