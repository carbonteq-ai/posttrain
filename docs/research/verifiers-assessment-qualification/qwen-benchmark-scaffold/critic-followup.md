# Independent follow-up review

Reviewed 2026-10-03 against the current isolated environment candidate at
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`.
This is a read-only source review, not a production eligibility qualification.
The root agent is running the focused eligibility, frozen-verification and
benchmark suites. A duplicate review-side run was stopped after one test to avoid
repeating the same expensive replay work; it supplies no suite-level result.

## What the fixes establish

The previous executor-trust defect is repaired in the inspected code.
`EligibleBenchmarkPool.verify` now checks structural consistency without reading
files or invoking subprocesses. Its explicit `validate_inventory` method requires
an accepted redesign and independently supplied executor and reassessment
registries. `eligibility._derive_proof` compares the retained revision with the
independently accepted revision before selecting an approved frozen verifier.
It no longer uses the artifact's own binding as its approval.

The previous self-consistent-label defect has a sound verification seam now.
`ReassessmentVerifier.replay` receives a detached native episode and frozen task.
The admission path compares rederived source snapshots, observation views,
expected targets and findings with the retained batch. Credit requests and
contributions are compared separately, with parent identifiers resolved to the
semantic content of their assessments. The deliberately volatile generated IDs
are omitted; native execution coordinates remain exact. The test fixture derives
facts from the original world rather than accepting caller-supplied task evidence,
and a resealed fabricated-evidence test exercises rejection.

The Simple publisher now separates assessed outcome from credited action.
`ReviewedSimpleTask.assessment_subject` assesses the trace. Its
`credit_recipient` rederives the supported action from source material and emits
no action recipient for neutral recording diagnostics, failed goals or an
already-correct state. The four actual retained Simple trajectory tests check
trace subjects, execution recipients, duplicate-batch handling and native reload.
This addresses the architecture issue without changing the native API.

## Remaining gates and defects

### 1. Eligibility is checked only before collection, not before later starts

`collect_benchmark` calls `manifest.pool.validate_inventory` once before taking
the collection lock. Its refill loop then persists starts and dispatches tasks
without checking whether the accepted source or selected proof changed while
earlier attempts were running. There is no full terminal closure check either.
Consequently, a valid initial admission can outlive the source revision it
qualified. This is a correctness gap for long-running, resumable collection,
even without adversarial input.

Add a private validation session owned by the run: initial independent approval,
freshly hashed selected proof/source closure immediately before each persisted
start, and a full closure verification before publishing the terminal summary.
Cache expensive replay only under the actual freshly hashed approved closure.
Do not cache by path, modification time or a caller-provided verified flag.

Required test: a fake runner changes the accepted source or selected assessment
artifact after the first dispatch. Subsequent starts must be rejected before
dispatch. Restore the source in test cleanup. Also test a mutation after the last
dispatch but before terminal publication.

### 2. An interrupted result still lacks independently retained justification

`benchmark._load_result` now rederives `summary_json` and ordinary terminal
status from native evidence. However, it explicitly exempts any
`status == "interrupted"` from status reconciliation. The result model requires
only a nonempty `error_type`; the loader does not reconcile that field either.
A resealed result can therefore relabel a completed or truncated native episode
as an interruption with an arbitrary explanation.

There is a valid reason for some interruptions to be outside the native episode:
the controller can crash after committing a completed episode but before its
result marker. Preserve that episode unchanged, but retain an explicit controller
recovery receipt explaining the original missing result and consumed start.
Validate normal errors against native errors and recovery dispositions against
that receipt. Do not solve this by forcing native episode mutation.

Required tests: reseal a completed result as interrupted with an invented reason;
reseal a failed result's error type; ensure both are rejected. Retain the existing
post-episode/pre-result crash recovery test and make its independent receipt the
reason the interrupted disposition remains acceptable.

### 3. Same-task foreign discarded traces are not excluded

Both `_load_result` and `_recover_interrupted` validate a discarded trace's task
data and artifact hashes, but do not check its episode ownership against the
current occurrence. A discarded trace from another rollout of the same task can
pass those checks. This contaminates failure/retry evidence even if the principal
episode remains correct.

Define the runtime's allowed discarded-trace ownership contract and verify its
native episode or explicit execution lineage. Do not assume task equality proves
occurrence membership. Add a same-task, different-episode discarded-prefix test;
keep a valid discard test to avoid rejecting real runtime behavior.

### 4. Actual deterministic Task.score replay is not implemented yet

In the inspected production package, `ReassessmentVerifier` is a protocol.
The only `replay` implementation found is `FixtureVerifier` in tests. This tests
the admission seam, but does not qualify the AutomationBench producer. The next
implementation needs to reconstruct the actual configured reviewed task from
the approved frozen task, rescore a detached native trace through `Task.score`,
and return its native batch and credit assignments without trusting retained
findings or source transformations.

The Simple producer also has goal and recording-coverage findings but no closed
whole-task guard contract. It therefore cannot supply the three independent
goal/guard/coverage checks required by `TaskRewardContract`. Keep real Qwen pool
admission closed until actual replay and explicitly authored guard contracts are
qualified. The structural three-purpose check alone cannot prove that an
arbitrary guard or coverage signal means the right thing.

## Additional tests before real collection

- A per-attempt timeout retains its native prefix while another slot completes.
- Controller cancellation consumes started attempts and does not resubmit them
  on resume.
- An expired collection deadline on resume starts no new attempts.
- Missing native usage remains unavailable rather than a verified zero.
- Fresh native stop-condition names are exercised against the real Qwen runtime;
  declared runner limits alone do not establish enforcement.
- Change each retained source/view/finding/credit-recipient field while resealing
  all local hashes and verify actual deterministic replay rejects it.

The inspected changes meaningfully improve trust boundaries and action
attribution. They remain a scaffold until these gates and the broader reward
contracts are implemented and independently checked.

## Reward-domain review

Read-only review of `support_rules.py`, `support_evidence.py`,
`notification_evidence.py`, `marketing_evidence.py`, `effect_index.py`,
`simple_evidence.py` and the updated eligibility guard contract. These are bounded
properties, not a qualification of whole-task reward closure. Domain workers own
the implementation and have received the counterexamples below.

### Sound architectural choices

- `GuardSetDeclaration` and `TaskRewardContract` version 2 explicitly distinguish
  a reviewed empty task-specific guard inventory from an unresolved inventory.
  Goal and coverage remain mandatory; declared guard keys must exactly match
  guard checks. This avoids invented prohibitions and constant pseudo-rewards.
  An empty task-specific inventory does not certify general safety.
- The Simple evidence adapter accepts recursively immutable mappings and tuples,
  reconstructs serialization from revisions, requires initial revision zero and
  reconciles the terminal world. Snapshot reuse does not deduplicate executions.
- Jira review effects join returned IDs to new persisted simulator action records.
  Domain obligation identity deduplicates accomplishment while preserving every
  created record. This does not imply a duplicate penalty.
- Hiver checks truth at the logging prefix. Delivery truth is bounded separately
  from counts, classification, thresholds and policy applicability.
- Current Q1 suppression authority can be selected deterministically from the
  public request, explicit applicability, authorized sender and task logical
  clock. An obsolete Q4 directive is not automatically a conflicting current
  authority. Genuinely co-applicable conflicts must remain unresolved; a generic
  newest-message rule is not justified.
- Per-address unresolved policy preserves independently proven harm for another
  resolved address, while keeping whole-task no-harm closure unavailable.

### Reproduced counterexamples and repair status

1. **Support prose can invert a purported positive purpose.** An authored
   `purpose_literals=("refund review",)` and otherwise matching FIN action record
   yields accomplishment 1 for both `Refund review is not requested` and
   `Do not open a refund review`. The current narrow immediate-negation regex
   does not cover those forms. Use reviewed finite affirmative forms or explicit
   structured purpose. Unsupported negation and surrounding prose must remain
   unavailable rather than become positive credit.

2. **Suppression note substrings can invert policy meaning.** `_candidates`
   classifies `Not pre-migration; not invalidated by system reset` as protected
   with no unresolved candidate, because both protected substrings occur. This
   can turn an otherwise allowed archival into a false harm finding. Prefer
   exact reviewed note literals or a closed grammar with defined parameter
   semantics. Do not grow an ad hoc general prose-negation parser.

3. **A hypothetical can pass as a completion report.** `_report` accepts
   `If we proceed: Archived 1 addresses: a@example.com`. Its completion regex is
   unanchored. A quoted or conditional clause must not qualify an actual report
   merely because its numerical values match the effects. Bind an authored
   completion form and keep unsupported prose unavailable.

4. **Unknown later capture rewrites an initial prefix fact.** A completed Hiver
   log at acknowledged revision zero yields premature-claim value 1. Extending
   the index with an unavailable occurrence lacking a revision changes that
   same log's finding to unavailable. `_prefix` checks all occurrences for known
   revisions before handling revision zero. Authoritative revision zero proves
   no earlier persisted simulator action; it must survive irrelevant later
   capture gaps. Later logs need source-qualified boundaries and revision
   linkage, not tuple order. Existing late-extension coverage exercises only a
   qualified later send and does not catch this case.

These counterexamples were executed directly against the current Python
predicates with synthetic public data and native fixture captures. They require
no model, credentials, hidden assertions or source modification.

Follow-up: the marketing owner replaced broad protected-note substring/prefix
classification with exact reviewed literals and anchored the full-message report
format. An independent run of `tests/test_notification_evidence.py` and
`tests/test_marketing_evidence.py` passed 31 tests in 2.00 seconds, including the
negated-note and hypothetical-report regressions. Support purpose and later-gap
prefix repairs remained pending at this review checkpoint.

### Remaining coverage edge

An already-correct Simple task with no actions can satisfy its goal but currently
cannot close recording coverage, because the empty effect index deliberately
returns `empty_occurrence_inventory`. Add a separate zero-transition proof only
when complete invocation inventory, authoritative initialization/finalization,
revision zero and initial/final equality establish that no simulator transition
occurred. An empty receipt list alone is insufficient. This issue is separate
from a reviewed empty guard inventory and from an empty credit plan.

### Additional counterexamples to qualify

- Duplicate or missing IDs in Gmail populations must be unavailable. The
  notification adapter's comparison dictionary currently collapses duplicate
  IDs, so an unknown operation can appear unchanged despite a hidden duplicate
  record mutation. Validate population identity before comparison.
- A sent message with matching recipient and subject but wrong digest counts
  establishes delivery, not correct content or whole-task completion.
- Known harm should remain observed after a later repair. A failed local action
  with acknowledged mutation must not be treated as a verified no-effect action.

### Native credit planning context: implementation and proof

The native candidate now exposes `CreditPlanningContext` and `Task.plan_credit`
and `Env.plan_credit`. The context carries the exact source identity, assessment
runs planned for this scoring invocation (including run, attempt and invocation
identities), and previously retained credit assignments. Assessors still receive
raw source material and may construct their own judge input; the planning context
does not constrain that input or interpret a judge's raw response.

The runtime validates source ownership and current-run identity uniqueness before
planning. Default planners delegate to the existing `credit_requests` hook, so
there is no implicit global deduplication or newest-success selection. A concrete
environment can intentionally decline previously assigned contributions and must
select this call's runs rather than substitute a successful historical attempt
when the current assessor fails.

Root's final native test run covered `tests/v1/test_judges.py` and
`tests/v1/test_trace.py`: 75 passed in 0.36 seconds. The extended integration cases
exercise both trace-level and episode-level planning, repeated scoring without
duplicate assignments under an explicit planner, failed current assessment with
older success retained, serialization, duplicate run rejection, and foreign
source rejection. Legacy planner behavior remains covered. Scoped Ruff and diff
checks passed. Candidate-aware scoped Pyright retains one pre-existing callable
union diagnostic in the assignment dispatcher; this is not a zero-error typing
qualification or a release qualification.

The earlier support findings were subsequently repaired by the support owner:
review purpose uses finite authored summary/description templates, and revision
zero facts survive unknown later capture. The owner reports 25 support tests
passing. Notification population identity validation and baseline-aware
marketing attribution were also added; the domain owner reports 36 tests
passing. These counts are owner-reported checkpoints, not a new independently
sealed whole-source qualification.

### Finance review: two executed counterexamples

1. An exact numeric report with a wrong ending balance loses its known harm when
   the sender appends `Please file this forecast.` An exact prohibited excluded
   vendor line also loses its disclosure finding with that same suffix. The
   whole-message unknown-context branch removes every harm. Numeric assertions
   need a bounded scope distinction so quoted or hypothetical reports still
   abstain; the public prohibition against naming excluded items can establish
   disclosure independently of unfamiliar surrounding prose.
2. Initial SENT mail to the CFO with a correct body and subject `Last year
   forecast` is treated as satisfaction of the new next-30-days obligation.
   A later wrong current report consequently retains ending goal 1 with no
   recipient, although its wrong-number harm remains observed. Public task/window
   binding must qualify baseline delivery; arbitrary historical matching content
   should remain a delivery diagnostic rather than current accomplishment.

Both cases were executed with real local Gmail simulator effects and the public
fixture. They were communicated to root and the finance evidence owner before
publisher qualification. The arithmetic uses explicit Decimal precision and
supports signed values coherently; no unsupported exchange-rate or rounding rule
was found in this review. Further useful tests are signed AR/AP amounts, missing
earlier capture versus a qualified current send, exact system-policy authority
(including quoted or negated policy text), and duplicate contradictory claims.

Finance repair review: independent simulator replay confirmed that the original
wrong-number report plus the reviewed filing appendix retains numeric harm, and
an excluded name plus unfamiliar prose retains disclosure harm. Initial undated
mail no longer satisfies the task; the scanner now requires the explicit
logical-clock forecast window. Guard authority is bound to the canonical system
prompt, and numeric reports require the acknowledged send's source projection
to match the initial forecast data.

Three further executed edge cases were sent to root and the evidence owner:

- The filing-appendix exception accepts any title beginning with the forecast
  title. A title ending `is fictional and not an actual forecast` consequently
  retains wrong-number harm. Use the exact reviewed title grammar in this
  exception too; an unknown qualifying title must remain unavailable.
- A native Gmail draft created with an additional SENT label and a matching
  dated body is treated as initial delivery. Its DRAFT label and draft-wrapper
  membership must exclude it from baseline satisfaction.
- Changing an included AR amount before sending an excluded vendor's name
  suppresses disclosure harm through the numeric source-basis early return.
  Numeric basis and excluded-name authority are distinct: a public excluded
  candidate unchanged at the send boundary can still establish prohibited
  naming, even when an unrelated included amount changed. Changed or ambiguous
  excluded-candidate authority must remain unavailable.

These are follow-up findings against the repaired source, not a claim that the
owner's 24 focused tests failed. No broad integration suite was duplicated while
root ran the integrated domain qualification.

Final finance repair checkpoint: independently reran those three exact simulator
reproductions after the owner's repairs. The fictional title now yields
unavailable numeric claims without false numeric harm; the initial DRAFT/SENT
message no longer supplies baseline satisfaction and the current wrong report
retains its harm; unrelated included-AR drift leaves numeric basis unavailable
but preserves the unchanged excluded-vendor disclosure and guard failure. All
three assertions passed. The owner reports 28 focused tests passing; root's
separate integrated batch remains the authority for broader qualification.

### Renewal acceptance and scalable follow-on work

Renewal publishing is a bounded predicate consumer, not whole-task acceptance.
Before treating its current task as accepted, qualify current-period baseline
processing, exact initial/live source authority at each send, and unambiguous
vendor/template/signer identities. A same-template historical envelope does not
alone establish that the present renewal was already processed. A multi-signer
agreement with unresolved semantics must not supply the single-vendor goal.
Wrong-template harm applies to a resolved task vendor; unrelated sends using
other templates remain outside this predicate. Excluded or already-processed
sends remain observed harm after voiding. Procurement delivery stays distinct
from summary correctness. Captured service history does not by itself prove a
complete relevant invocation inventory or whole-guard compliance.

After access routing, batch by reusable action family rather than writing one
monolithic evaluator for every successful task. The current adapters already
provide a basis for shared public row joins, authored policy selection,
eligibility predicates, native effect identity, exact content forms, baseline
period binding, prerequisites, and independent coverage axes. Compose reviewed
contracts for create-and-attach, route-or-deny, notify-and-record, conditional
mutation, numeric reporting, and approval/hold-gated actions. A task contract
supplies its public selectors, literals, grammar and declared supported scope;
an unknown semantic rule remains unavailable.

Freeze five to ten contracts from one action family, exercise the shared
adversarial matrix and each retained real trace, then qualify native publishing
and reload together. The matrix includes duplicate identities, historical
baseline, self-created progress, wrong recipients/targets, failed mutated
actions, unknown capture before/after the decision, source edits/restoration,
and repair after harm. Representative tests do not qualify untouched tasks:
the inventory must retain each task's goal, guard, prerequisite and unresolved
semantic coverage. This reduces duplicate implementation without introducing an
automatic prose-policy interpreter or a catch-all model judge.

### Shared record updates and access routing review

The first ten Salesforce contracts reuse one direct-state evaluator. It validates
population IDs, normalizes native Opportunity fields, keeps calendar-date intent
explicit, rejects booleans/non-finite values as numeric comparisons, and binds
supported updates to operation, response identity and persisted requested fields.
Initial satisfaction and completion followed by damage/restoration do not earn
fresh accomplishment credit. The reviewed scope does not establish general
no-harm guards over unrelated record fields. Two narrow follow-ups were sent to
root: the public `from Needs Analysis` stage request needs an explicit reviewed
baseline interpretation, and known Zapier success responses should require the
native `success=True` grammar rather than accept missing/null success. Salesforce
API PATCH's deliberately empty response remains a separate adapter contract.

Two access-routing counterexamples were executed before publisher qualification:

- An Asana task with notes prefixed `DO NOT provision this user; this is a
  cancellation record.` followed by the four exact identity fields earned both
  creation and section goals. Identity fields establish the candidate, not the
  task's provisioning purpose. Finite reviewed title/notes forms must qualify
  accomplishment; unsupported contextual clauses remain unavailable.
- An exact supported denial format sent to an approved requestor falsely
  describing a Director of Engineering as insufficient left the no-prohibited-
  route check and effect scope at 1. Approved-recipient decision messages were
  ignored. A supported incorrect denial should yield its own harm; unknown
  access-decision content cannot justify full content scope.

These cases used real native Asana/Gmail handlers and were communicated to the
access owner and root. The owner's 18 passing focused tests preceded these
counterexamples; broader publisher acceptance must include their repairs.
