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
resolves exactly the same configuration as an interactive shell; the machine's
tracking endpoint and scoped credentials load automatically; and `posttrain job
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
- [x] (2026-08-01) Milestone 2: operator-owned machine configuration at
      `$XDG_CONFIG_HOME/posttrain/config.toml` and the state-directory split.
      Top-level defaults now load automatically, relative storage resolves
      beneath `$XDG_STATE_HOME/posttrain`, and local execution needs no
      provider block. `posttrain machine init` creates the non-secret config
      and protected named credential sources; `posttrain machine project add`
      updates the controller project set idempotently. The dstack block holds
      only its client locator and named credential source; execution-dstack
      and Ansible own worker storage beneath `/var/lib/posttrain`.
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
- Observation: automatically loaded machine defaults made tests outside the
  CLI package sensitive to the developer's real `$XDG_CONFIG_HOME`.
  Evidence: the full suite initially failed four nested Lab packing tests on
  this workstation's obsolete profile schema. A reviewed root `conftest.py`
  now isolates config and state homes for every package, and the full suite
  passes independently of the host installation.

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
- Superseded decision: use one named site profile in `config.toml`, not a provider-only
  file that also accumulates tracking, trust, and storage settings.
  Rationale: provider identity and tracking identity are independent concerns,
  but an operator needs one coherent machine profile that selects both.
  Date/Author: 2026-08-01 / architecture review. Superseded later the same day
  by the top-level machine-config decision below.
- Decision: supersede both the named-profile layer and the
  one-provider-per-profile shape. The settings live at the top level of
  `config.toml`, with no machine name in any key, and they are the defaults
  every project on that machine inherits. The file also carries `projects` and
  `default_provider`; the run keeps choosing a provider with the existing
  `--provider` flag, and the controller is deployed per machine.
  Rationale: the file already lives under one user's `$XDG_CONFIG_HOME` on one
  machine, so naming a profile after that machine restates what the file's
  location already says, and `POSTTRAIN_PROFILE` then makes every project
  select the only thing on offer. The implemented schema compounded this by
  taking `provider.kind` as a single `"dstack" | "local"`, which forced two
  profiles — `pop-local` and `pop-dstack` — to describe one workstation, and
  made the provider a property of a config file rather than of a run. Three
  things already in the tree contradict that: `job pack`/`job run` already
  accept `--provider`, admission keys are host-scoped (`"host:" + hostname` in
  `packages/execution/.../admission.py`), and this program's own lifecycle plan
  states the controller pumps queued work across projects. Nothing enumerates
  project roots today, so a systemd unit can only be pointed at one project
  while the ledger it reconciles is machine-wide. The superseded
  `execution.toml` had the provider part right: it declared `[providers.local]`
  and `[providers.dstack]` together.
  Date/Author: 2026-08-01 / framework maintainer.
- Decision: local execution requires no provider configuration, and dstack is
  an optional block that only exists where a dstack server already does. The
  framework stores a dstack client locator and never installs, configures, or
  supervises the server.
  Rationale: someone running the framework needs tracking, trust, storage, and
  their project list; they do not need dstack. Putting `[providers.dstack]`
  beside `[providers.local]` in the baseline presents an optional remote
  dependency as part of setup. dstack is always a remote server that jobs are
  submitted to, and `ai-infra` owns its lifecycle through Ansible; the
  framework's concern ends at the client locator.
  Date/Author: 2026-08-01 / framework maintainer.
- Decision: a project inherits the machine settings and may narrow only
  `defaults`; provider, storage, trust, and tracking remain operator-owned and
  are not overridable from the project.
  Rationale: inheritance is wanted, but override in the other direction would
  reopen exactly what this plan closed. If a project can override
  `trust.ca_bundle`, `provider.credentials_file`, or `storage.run_root`, then a
  job's trust root and identity depend on who submitted it rather than on the
  machine it ran on. `defaults` are execution overrides with no security
  content, so narrowing them per project is safe.
  Date/Author: 2026-08-01 / framework maintainer.
- Decision: `posttrain.env` is a string-to-string runtime value map, not a
  second structured configuration format.
  Rationale: typed structure remains in TOML/YAML schemas; environment values
  must work identically for YAML-authored and Python-authored jobs and are
  validated only when bound to a declared runtime variable.
  Date/Author: 2026-08-01 / architecture review.
- Decision: require durable tracking for managed jobs without hard-wiring
  Trackio into core contracts.
  Rationale: the current deployment chooses Trackio, while the framework also
  supports W&B. Policy should require evidence; the machine config chooses the
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
- Milestone 2 outcome: `load_local_execution_config()` automatically resolves
  one machine file with no project selector. Named mode-0600 sources supply
  only the credentials allowed for Trackio/W&B, Hugging Face, the Python
  index, or dstack; tokens never enter the non-secret TOML. Project
  `posttrain.env` remains the authoritative project override. Legacy local
  `execution.toml` remains a one-release compatibility reader. Required-
  tracking policy and machine-aware doctor checks remain open.

## Context and Orientation

The target authority table, which this plan implements:

    Project identity, catalog paths, work packages   .posttrain/project.toml      (tracked)
    Project-specific runtime overrides and secrets   posttrain.env (repo root)   (ignored, 0600, auto-loaded)
    Machine defaults: providers, tracking, storage,
    trust, and the project list the controller serves  $XDG_CONFIG_HOME/posttrain/config.toml
    Machine credentials, scoped by consumer           $XDG_CONFIG_HOME/posttrain/credentials/*.env (0600)
    Submit/cancel intent, handles, receipts           .posttrain/state/executions/
    Materialized datasets, build caches               .posttrain/state/cache/

The concrete machine and project selection shape is:

    # $XDG_CONFIG_HOME/posttrain/config.toml
    # This file already belongs to one user on one machine, so its settings sit
    # at the top level and no machine name appears in any key.  Everything here
    # is a default that every project on this machine inherits.
    schema_version = 1

    # Project roots this machine's controller reconciles.
    projects = ["/srv/posttrain/foundation", "/srv/posttrain/lab"]

    # Which provider a run uses when it does not say. `--provider` overrides it.
    default_provider = "local"

    [tracking]
    kind = "trackio"
    endpoint = "https://trackio.example.internal"
    credentials = "trackio-default"

    [services]
    python_index_url = "https://pypi.example.internal/simple/"
    job_registry = "registry.example.internal/posttrain"

    [huggingface]
    credentials = "huggingface-default"

    [trust]
    ca_bundle = "/etc/ssl/certs/ca-certificates.crt"

    [storage]
    run_root = "runs"
    model_cache = "cache/huggingface"
    compile_cache = "cache/compile"

    [credentials.trackio-default]
    file = "credentials/trackio.env"

    [credentials.huggingface-default]
    file = "credentials/huggingface.env"

That is the complete baseline. Local execution normally needs no provider
block: `canonical_hostname` defaults to this machine's hostname. A machine
whose Docker bridge cannot reach the host's internal DNS may add literal,
machine-owned resolvers without leaking them into a project:

    [providers.local]
    dns_servers = ["192.0.2.53"]

The resolved server list is part of the provider binding snapshot, so a retry
does not silently adopt changed machine defaults. Hostnames are rejected here
because resolving a DNS server through the broken resolver would be circular.
A person running the framework must not have to declare a provider, and must
not need dstack at all.

dstack is an optional addition, present only where a dstack server already
exists. The framework never installs, configures, or supervises that server —
it is always remote, and `ai-infra` owns it through Ansible. What the framework
stores is a client locator and nothing more:

    # optional, only where a dstack server already exists
    [providers.dstack]
    project = "posttrain"
    python = "/opt/dstack/venv/bin/python"
    credentials = "dstack-default"

    [credentials.dstack-default]
    file = "credentials/dstack.env"

The TOML stores no token values; it stores stable identities and named
protected credential-file locations. Each consumer receives only its scoped
variables, and nothing is read from the ambient shell.

A machine may therefore offer more than one provider. The provider for a run is
chosen by the `--provider` flag that already exists on `job pack` and `job run`,
falling back to `default_provider`. It is never a property of which
configuration file happened to be in place.

`projects` is what makes the controller's deployment scope match the ledger it
reconciles. Admission keys are already host-scoped, and the controller is
specified to pump queued work across projects, but nothing enumerates project
roots, so a service unit could only ever be aimed at one. One machine runs one
controller; it reads this list, and for each run it loads whichever provider
the run's own frozen locator names.

A project inherits these settings and may narrow only `defaults`. Provider,
storage, trust, and tracking are operator-owned and are not overridable from
project files, because otherwise a job's trust root and storage identity would
depend on who submitted it rather than on the machine that ran it.

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
  writes only the portable project layout and protected project override.
- `apps/cli/src/posttrain_cli/scaffolding/init_machine.py` — writes machine
  defaults, scoped credential sources, and project registrations.
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

Milestone 2 introduces the operator-owned machine configuration at
`$XDG_CONFIG_HOME/posttrain/config.toml`. It describes one machine at the top
level of the file: storage roots, trust-bundle location, a tracking
backend/endpoint, `default_provider`, and the list of project roots whose runs
this machine's controller reconciles. Local execution needs nothing further.
An optional `[providers.dstack]` block adds the client locator — project,
client Python, and named credential source — where a dstack server already
exists; the framework never manages that server or its worker storage. A project inherits these
settings and selects a provider per run with `--provider`; provider and
tracking locators are still stored separately on each submission.

Do not keep the named-profile layer or the single `provider.kind` shape that
shipped in the first slice of this milestone. The profile name restates the
file's own location, and `provider.kind` makes the provider a property of a
configuration file, which forces one workstation offering two execution paths
to be described as two unrelated profiles and contradicts the `--provider`
flag, the host-scoped admission key, and the cross-project controller that all
already exist. Remove `POSTTRAIN_PROFILE` with them.

`.posttrain/state/` is reorganized into `executions/` (control receipts —
protect and retain) and `cache/` (safe to delete); `execution.toml` shrinks to
machine-binding leftovers and is then folded into the machine configuration.
Provide a `posttrain state migrate` command that performs the split
idempotently.

Milestone 3 makes durable tracking the default policy for managed training and
evaluation. The machine config supplies either Trackio or W&B plus its
endpoint and named credential source. `doctor` verifies reachability with an actionable failure. The
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
- On a machine initialized with `posttrain machine init`, a new project reaches
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
- 2026-08-01: Milestone 2 begins with profile resolution. It uses the agreed
  `[profiles.<id>.provider]`, `storage`, `trust`, and `tracking` shape,
  projects only the selected secret-free tracking endpoint into runtime
  configuration, and never reads `POSTTRAIN_PROFILE` from the ambient shell.
- 2026-08-01: Milestone 2 superseded that transitional profile slice with one
  automatically loaded machine config, named scoped credential sources,
  separate project and machine initializers, state-root-relative caches, and
  execution-dstack-owned worker storage.
