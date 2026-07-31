# Make the run lifecycle correct, observable, and automatic

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document in accordance with
`docs/templates/PLAN.md`.

Source findings: `docs/dx-improvements/v0.2.5/README.md` findings 1, 2, 3, 4,
6, 11, and the "Proposed lifecycle controller" section. This plan is
self-contained; the critique is background, not required reading.

## Purpose / Big Picture

Posttrain submits GPU jobs (training, evaluation, serving benchmarks) to an
execution provider — local Docker or a dstack server — and records the outcome
as durable evidence in a tracking backend (Trackio or W&B). Today a developer
who submits one job must manually advance its lifecycle: run
`posttrain run reconcile` after the provider finishes (or a local GPU slot
stays blocked forever), run recovery commands after a cancel, and mentally
join three state systems (local control records, the provider, tracking) to
know what happened. Worse, some commands can act on the wrong run or the wrong
project entirely.

After this plan, a developer can submit a job, let the submitting shell exit,
and later run `posttrain run show <id>` to see one truthful joined view;
cancellation is one command; a finished run releases its GPU slot without
human action while the controller host is running; and a mutating command can
never silently target a different run than intended.

## Progress

- [x] (2026-08-01) Plan authored from the v0.2.5 release-scoped critique.
- [x] (2026-08-01) Cross-plan architecture review completed; ownership,
      projection, targeting, and controller deployment decisions revised.
- [x] (2026-08-01) Milestone 1: chronological read selectors and canonical-id
      mutation. Read selectors order strictly by timestamp; cancel, retry,
      reconcile, recovery, and cleanup reject prefixes and `--last`.
- [ ] Milestone 2: frozen prepared submissions and control-store ownership.
- [ ] Milestone 3: durable provider locator (`ExecutionProviderSource`).
- [ ] Milestone 4: provider-neutral `RunView` and idempotent reconciler.
- [ ] Milestone 5: stable CLI JSON error envelope.
- [ ] Milestone 6: foreground controller, then service packaging.
- [x] (2026-08-01) Admission entries now persist and validate the owning
      control-store URI. Review found that the shared pump does not consume the
      locator yet: it still captures the reconciling project's provider and
      store factory. Milestones 2 and 3 therefore remain release blockers; the
      locator field alone is not recorded as their completion.

## Surprises & Discoveries

- Observation: the admission ledger already stores the encoded `ExecutionPlan`;
  the cross-project defect is caused by the project-capturing service factory,
  not by a total absence of frozen launch intent.
  Evidence: v0.2.5 `ExecutionAdmissionService.enqueue()` persists `plan`,
  `evidence_source`, and `provider_binding`, while
  `execution_admission_service()` closes over one `ProjectLayout`.
- Observation: prefix resolution is shared with mutations, so fixing only the
  sort order would leave a second time-of-check/time-of-use targeting risk.
  Evidence: v0.2.5 `resolve_run_id()` returns a unique prefix match for every
  caller.

## Decision Log

- Decision: order milestones so every correctness fix (1–3) lands before the
  automation (6); an automatic controller built on ambiguous targeting would
  automate the defects.
  Rationale: the controller pumps queued work across projects; findings 1 and
  4 make that unsafe today.
  Date/Author: 2026-08-01 / plan author.
- Decision: queue a complete `PreparedSubmission` instead of reopening the
  project when admission becomes available.
  Rationale: project files and machine defaults may move while a run waits. The
  known launch intent must not be reinterpreted later.
  Date/Author: 2026-08-01 / architecture review.
- Decision: mutating commands accept only the complete canonical run id.
  Rationale: an unambiguous prefix can become ambiguous between resolution and
  action; convenience selectors belong on read-only commands.
  Date/Author: 2026-08-01 / architecture review.
- Decision: `RunView` is read-only and reconciliation is a separate pure
  reducer that returns actions.
  Rationale: observing a run must not mutate it, while controller decisions
  must remain deterministic and testable.
  Date/Author: 2026-08-01 / architecture review.

## Outcomes & Retrospective

- Planning review outcome: the controller is no longer expected to repair
  ambiguous project ownership. It operates only after exact targeting, frozen
  submissions, provider locators, and a read/action split exist. Implementation
  outcomes remain to be recorded milestone by milestone.

## Context and Orientation

Key vocabulary. A *run* is one execution of one job. *Admission* is a
machine-wide ledger that serializes local-Docker runs so two projects on one
GPU box do not collide; it lives under a machine root (default
`/var/lib/posttrain` or `$XDG_STATE_HOME/posttrain`), implemented in
`packages/execution/src/posttrain/execution/admission.py`. *Reconciliation*
joins the provider's terminal state with retained tracking evidence and is the
barrier that releases an admission slot. *Evidence source* is a stored,
secret-free description of where a run's tracking evidence lives, sufficient
to find it again after settings change.

Key files, all repository-relative:

- `packages/execution/src/posttrain/execution/admission.py` — machine-scoped
  admission ledger. An entry today stores the encoded plan, evidence source,
  and a provider-binding fingerprint, but not the owning control store or a
  complete frozen provider locator.
- `apps/cli/src/posttrain_cli/execution_provider.py` — constructs the
  admission service with closures over one project's `ProjectLayout`; when a
  slot is released, the next waiting entry is pumped through whichever
  project's factories performed the release. This is finding 1: releasing
  project A's slot can submit project B's plan through A's configuration and
  receipt store.
- `apps/cli/src/posttrain_cli/run_resolve.py` — resolves `--last`. It sorts
  by timestamp, then stable-sorts by admission-state priority so waiting runs
  outrank newer terminal runs; `posttrain run cancel --last` can therefore
  cancel an older waiting run instead of the newest run (finding 2).
- `apps/cli/src/posttrain_cli/execution_config.py` — reads the current
  project's ignored `.posttrain/state/execution.toml` and rebuilds a provider
  adapter from *current* settings even for old runs, so editing the dstack
  binding can make an existing run unqueryable (finding 4).
- The CLI's top-level exception handler prints `error: <message>` to stderr;
  JSON-mode failures have no structured envelope (finding 11).

## Plan of Work

Milestone 1 changes `run_resolve.py` so read-only `--last` is strictly
chronological and gates every mutating command (`cancel`, `cleanup`,
`retry-submit`, `recover-cancelled-tracking`) on the complete canonical run id.
Prefixes remain read-only lookup conveniences. The old priority behavior moves
behind a new read-only `--needs-attention` selector so the useful "what needs
my eyes" query survives honestly. Introduce a typed selector error carrying
`exact_run_id_required`; milestone 5 later renders that code in the common JSON
envelope.

Milestone 2 replaces the factory closure with two durable values. A
`ProjectControlLocator` identifies the owning receipt store (`project_id`,
versioned `control_store_uri`). A `PreparedSubmission` freezes the semantic
plan and package identity, launch plan, provider source, evidence source, and
configuration fingerprint. When a slot releases, admission executes that
prepared submission and writes through its control locator; it does not reopen
the project or resolve current settings. Extend the existing cross-project
test so it releases project A, changes B's current defaults while B is waiting,
and proves the original frozen B submission is launched and its receipt is
written only to B's store. An unavailable store becomes an attention state
instead of falling back to A.

Milestone 3 adds `ExecutionProviderSource` beside the existing evidence
source on each submission record: provider kind, profile id, dstack
project/endpoint scope, credential-file identity, binding fingerprint — stable
identity only, never secret values. Status, cancel, and collection reconstruct
the adapter from the recorded locator, not from current project config.

Milestone 4 introduces a provider-neutral `RunView` value in
`packages/execution` with separate dimensions — `logical_state`, `control`,
`provider`, `evidence`, `reconciliation`, `controller` — and a pure
`reduce_reconciliation(view, observations)` function that returns a
`ReconciliationDecision` containing explicit actions. `posttrain run show`
only builds and renders the view; it never executes actions. Manual
`reconcile` and the controller execute the same decision through an idempotent
action runner. Do not collapse dimensions into one boolean: a provider can be
done while evidence is still finalizing.

Milestone 5 defines the JSON error envelope
(`code`, `phase`, `message`, `retryable`, `suggested_command`) raised as a
typed exception family and rendered by one handler, so automation can
distinguish invalid config from pending evidence from provider unavailability.

Milestone 6 adds `posttrain controller run` — first with `--once`, then as a
foreground loop. It scans active control locators, delivers persisted cancel
intent idempotently, polls provider state with bounded backoff, executes
reconciliation decisions, releases admission, and pumps frozen prepared
submissions. Ambiguous submissions stay manual. The production deployment is
the same controller core as an Ansible-managed system service on the always-on
control host. A systemd user unit is optional workstation convenience, not the
reliability boundary. Manual `reconcile` remains a guarded repair tool.

Milestones 4–6 require the minimal public application service from
`docs/plan/dx-public-api-and-authoring.md` milestone 1. Milestone 6 also
requires the authoritative configuration resolver and named site profiles from
`docs/plan/dx-configuration-authority.md` milestones 1–2. Do not recreate
either concern inside the controller.

## Concrete Steps

Work from the repository root. Run the focused test suites per package as you
go:

    uv run pytest packages/execution/tests -q
    uv run pytest apps/cli/tests -q

Expected: all green before and after each milestone; each milestone adds
tests that fail before its change. For milestone 1, add a test that creates
runs in the order waiting-then-terminal-then-newest and asserts read-only
`--last` returns the newest, while `cancel --last` and `cancel <prefix>` are
rejected with the domain error's `exact_run_id_required` code. Before milestone
5 the human CLI may render it as text; after milestone 5 JSON mode exposes the
same code in the common envelope.

For milestone 6, validate the controller against the fake provider used by
existing execution tests: submit through the fake, kill the submitting
process, run `posttrain controller run --once`, and observe the run reach a
reconciled terminal state and the admission slot release.

## Validation and Acceptance

Acceptance is behavioral, demonstrated from an installed external project:

- Two projects sharing one local worker hand off admission without using the
  wrong project's config or receipt store (extend the cross-project test to
  cover the release-and-pump step).
- `posttrain run cancel --last` and `posttrain run cancel <prefix>` refuse with
  `exact_run_id_required`; the complete canonical id targets exactly one run.
- A submitted job reaches a reconciled logical terminal state after the
  submitting shell exits, with the controller running.
- `posttrain run show <id>` prints control, provider, evidence, and
  reconciliation dimensions without Observatory installed.
- A JSON-mode failure emits a parseable envelope with a stable `code`.

## Idempotence and Recovery

Every controller action must be safe to repeat: cancellation delivery,
reconciliation, and admission release are idempotent by design; a crashed
sweep re-runs from persisted state. Milestone changes are additive — old
admission entries without an owner locator are treated as attention states,
never auto-pumped. If a migration of the admission ledger format is needed,
write new-format entries beside old ones and retain old terminal receipts
unchanged.

## Artifacts and Notes

Record the `RunView` rendering of one real dstack run and one local run in
this section once milestone 4 lands, as indented transcripts.

## Interfaces and Dependencies

In `packages/execution/src/posttrain/execution/views.py`, define:

    @dataclass(frozen=True)
    class RunView:
        run_id: str
        logical_state: str          # queued|submitted|running|finalizing|succeeded|failed|cancelled|attention
        control: str                # admission/control record state
        provider: ProviderObservation      # state + observed_at
        evidence: EvidenceObservation      # state + required_roles_pending
        reconciliation: ReconciliationState  # pending|settled|failed + next_retry_at
        controller: ControllerHealth | None

In `packages/execution/src/posttrain/execution/provider_source.py`, define
`ExecutionProviderSource` (frozen dataclass; secret-free fields only). The
reducer lives in `packages/execution` and depends only on immutable observations
and value types. The action runner depends on provider and tracking protocols,
never on dstack or Trackio SDKs directly, so both are testable with fakes. The
CLI and controller call them through the public application service in
`docs/plan/dx-public-api-and-authoring.md`; do not create a temporary second
application service in `packages/execution`.

Also define `ProjectControlLocator` and `PreparedSubmission` as versioned,
secret-free records. `PreparedSubmission` contains enough immutable launch
intent to submit after the project root changes, but contains no secret values;
`ExecutionProviderSource.profile_id` resolves rotating credentials from the
named site profile at action time.

## Revision Notes

- 2026-08-01: Architecture review replaced project reload at queue-pump time
  with frozen prepared submissions, made exact canonical ids mandatory for
  mutations, split read-only `RunView` from reconciliation actions, and made
  the Ansible-managed system controller the production deployment boundary.
