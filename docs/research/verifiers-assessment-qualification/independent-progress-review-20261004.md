# Independent review of the AutomationBench manifest expansion

Reviewed 2026-10-04. This is an analysis of current source and evidence, not a
release approval or a change to the implementation plan. No environment/native
source was changed and no model calls were made during this review.

The other agent has moved the work from a small set of components to drafts
across the complete 105-task sample. The main remaining work is now adversarial
qualification and action credit, rather than another large authoring pass.
Two shared evidence-admission defects, a temporal-proof limitation and several manifest false positives
need resolution before treating candidate coverage as reliable learning signal.

## Current state

### Correction: the sample is not the Luna full-success pool

Following the user's correction, I inspected `reward-candidate/cross-category-selection.json`
and rehashed all 105 referenced episode files: all source hashes match. The
selection explicitly prioritizes family and observed native-tool-operation
coverage, not full Luna success. Its recorded official outcomes are **23 full,
67 partial, and 15 zero**. Only 11 of the 90 non-Simple selections have a recorded
full score; the selected Support tasks contain no full-score episode.

This sample was declared as development API-gap testing. It can expose missing
capabilities and harmful actions, but cannot substitute for the Luna-solvable
pool used to prioritize successful reference examples and downstream Qwen work.
The 81 manifest candidates are not 81 tasks solved by Luna. Official full score
also remains distinct from redesigned guard compliance and qualified complete
output accounting within the original 16,384-token budget.

The earlier recommendation to focus solely on qualification of existing drafts
was incomplete. First reconcile the complete retained development inventory
with its exact recorded outcomes, then report overlap of full-score tasks with
existing manifests. Prioritize successful reference coverage from that pool;
retain the broader 105-task sample as a separate API-gap and failure-test corpus.
Do not discard its shared infrastructure or failed traces, silently replace its
selection, pad categories with unsuccessful tasks, or relax scoring to turn a
reference failure into eligibility. Final student eligibility still requires
current reward, guard, source-version and budget qualification.

Environment HEAD: `0b0eef5cd660170b1c14aa6b5cb9ab71c7efe945`, branch
`wip/automationbench-reward-redesign-2026-10-04`. Native Verifiers HEAD:
`84ab782391bbfe1ac4f4ca32fa612e56d01b5b81`, with dirty changes. Framework HEAD:
`1ec6741f`, with a modified plan. Manifest work includes untracked files and
dirty revisions; a commit alone does not identify the complete reviewed tree.

| Measure | Current source census | What it establishes |
| --- | ---: | --- |
| Selected tasks with a draft | 105 / 105 | Authoring breadth, not acceptance |
| Tasks with a review | 104 / 105 | `sales.linkedin_event_promotion` lacks its review |
| Reviews marked qualified candidate | 81 | Author-declared coverage under the deterministic scope |
| Reviews marked not qualified | 23 | Explicit incomplete declarations |
| Obligation rows marked expressed | 911 | Declared checks; not 911 independently qualified checks |
| Obligation gaps | 28 | Author-reported gaps |
| Out-of-scope obligation rows | 323 | Excluded under the revised scope |
| Reviews recording known gaming | 59 | Known weaknesses remain in many drafts |
| Catalog entries | 28 | Includes previous partial declarations and five newly installed tasks |
| Independently reviewed installed tasks | 5 | Pinned reviewed bytes and native replay tests; bounded deterministic scope |
| Drafts with action-credit rules | 18 | Rule presence, not original-token qualification |
| Outcome-only drafts | 87 | Findings are not yet localized action rewards |

Drafts cover fifteen tasks in every category. Candidate reviews by category:
finance 12, HR 11, marketing 12, operations 12, sales 11, Simple 13, support 10.

The five installed tasks are `finance.escrow_tracking`,
`hr.airtable_learning_path_assignment`, `operations.zoom_training_setup`,
`operations.calendly_equipment_inspection` and
`simple.email_zendesk_ack_reply`. Their credit lists are currently empty.

The meaning of qualification changed in the updated plan: only deterministic
task-specific obligations count. Wording/judgment checks and the shared
no-clarification/exclusion-narration rules are separate. The plan records this
as user-requested. Respect that scope, but label it explicitly; these counts do
not establish complete verification of the original public task and system rules.

## Substantive progress

The prepaid aggregate report check is installed and qualified within its scope.
It computes current policy totals rather than hardcoding an answer, binds the
recipient and total to the same send, and rejects the recorded $4,300 report
against the corrected $4,900. Existing official episode scores remain unchanged.

Shared engine work now supports candidate-relative selections with explicit
unknown/tie handling, decided lookup absence, generated-effect joins,
cross-channel guard alternatives, generic acknowledged record writes,
composite identities, exact time/date formats, business-day arithmetic with
declared holidays, fact mentions and anti-hedging constraints. Sheets and Slack
read adapters expand observation evidence. These capabilities stay inside the
environment rather than introducing a competing framework API.

The agent also repaired simulator task data and tooling, including missing Jira
projects. Old Luna initial states and repaired task data must remain different
versions; genuine alternatives against repaired data cannot silently replace
the old trace's source contract.

Independent reviewers found gaming in each of the first five candidate tasks
and the coordinator applied fixes before installation. Parallel test execution
and digest/contract caching improved iteration speed. Local source commits and
separate worktree merges now make engine increments easier to inspect.

## Evidence independently checked here

On the current environment/native candidates:

```text
tests/test_qualified_manifests.py
10 passed in 20.40s

tests/test_manifest_sheet_reads.py
tests/test_manifest_effect_joins.py
tests/test_manifest_record_writes.py
tests/test_manifest_selections.py
tests/test_handler_scope.py
101 passed in 4.04s
```

Commands ran from
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`
with `PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src`
and `.venv/bin/python -m pytest ... -q --tb=short --strict-markers`.

The installed tests pin package bytes to the reviewed drafts, reproduce exact
findings on SHA-bound Luna episodes, preserve episode bytes and official scalar
scores, and repeat on rescoring. They are useful compatibility/replay evidence;
passing them does not establish resistance to the new adverse probes below.

The documented 3,493-pass/seven-skip full-suite checkpoint predates integration
round 6. This review did not run or claim a current full-suite gate.

## Shared evidence defects and ordering limits

### 1. Successful reads can disappear when nothing was persisted

`effect_evidence.py::_unpersisted` filters terminal returned calls with
`state_persistence=unchanged|not_attempted`, no conflict and no write ACK.
`gmail_observations.py` and `slack_reads.py` now use `persisted_transitions`.
That is a write-oriented inventory rule: no persisted mutation does not mean no
returned information.

An independent native-admitted Gmail probe retained a successful full-message
return but used a valid unchanged/no-ACK receipt. Gmail observation changed from
one qualified read to **complete inventory with zero reads**. WireTrace admission,
sealed source capture and the factual adapter accepted the probe. Sealing protects
source identity; it cannot correct a bad interpretation of the sealed evidence.

Relevant locations:
`src/automationbench_v1/effect_evidence.py:132`,
`src/automationbench_v1/contracts/gmail_observations.py:156`,
`src/automationbench_v1/contracts/slack_reads.py:149`.

Required repair: keep read-return inventory independent of persisted-write
inventory. Qualify a read from authenticated return evidence where supported;
otherwise retain unavailable read coverage. Do not certify absence by dropping it.

### 2. Initial reconciliation tolerates invented deterministic content

`contracts/service_hydration.py::_agrees` compares only supplied nested record
keys and treats all omitted fields as unconstrained generated defaults. Some
defaults are nondeterministic UUID/time values, but others are deterministic
business content.

With public Gmail `{messages: [{id: 'm', subject: 'S', body_plain: 'B'}]}`, the
normal hydrated `body_html` is empty. Replacing observed `body_html` with
`<p>invented initial content</p>` still passes `public_service_matches`. An
independent zero-call WireTrace/sealed-source probe then yields closed Gmail
inventory despite the invented content.

Relevant locations: `contracts/service_hydration.py:36` and `:82`.
Required repair: distinguish audited nondeterministic generated fields from
deterministic defaults, validate the complete native schema, and compare every
deterministic default. Sparse public state is not permission to ignore arbitrary
nested values or silently erase supplied fields.

### 3. Revision ordering is not read-return-before-dispatch

`contracts/joins.py:69` defines before/not-after through applied versus expected
state revisions. That establishes storage order, not returned information or
the model's opportunity to use it.

A native-admitted Sheets-read/Slack-post probe has event order
`read.dispatch → post.dispatch → read.returned → post.returned` and valid state
revisions. Both adapters close, and the `before` join matches. The read committed
before the post's state snapshot, but its response did not precede post dispatch.

Required repair: preserve storage-order joins for their legitimate purpose and
add an explicit authoritative return-before-dispatch prerequisite for public
read-before-action requirements. Do not imply model conditioning; abstain if
event/model provenance cannot establish the stronger order.

## Manifest correctness gaps

Generic record-write evidence covers both canonical records and action-log
records. Creating an audit entry is not proof of a retained business object.
Likewise, `values_text` may contain pre-existing fields as well as written ones:
matching its words does not prove that this action supplied or changed them.
Positive credit must use requested/changed fields and a verified accomplishment,
not merely matching text in the after-state.

Independent current-predicate probes found:

- `marketing.lead_enrichment` accepts “Added 6 contacts” through its count
  predicate without verifying six actual creations. Its comparison uses initial
  eligibility counts rather than accomplished additions. See
  `manifest-drafts/tasks/marketing.lead_enrichment/draft.json:1073`.
- `finance.monthend_journal_entries` lets the prohibited VOID $2,750 audit entry
  evade both guards under abbreviated labels (“Dec audit adjustment” and
  “Dec audit fee adj.”). The record guard still relies on the full item name.
  See that task's `draft.json:3658`. These are direct predicate probes, not newly
  collected model rollouts.
- The installed HR learning-path qualification review explicitly retains a
  counterexample: assigning the sabbatical employee path/courses while including
  “Learning Sabbatical” receives the positive finding and no harm. See
  `hr.airtable_learning_path_assignment/qualification-review.md:78`.
  A status/deferral phrase must not blanket-exempt a contradictory structured
  assignment; otherwise describe the check as partial rather than enforcing
  the no-assignment requirement.

Required-read manifests usually require `cell_values_returned=True`, so they
reject metadata-only and zero-row responses. However, this is only proof that
some cell was returned. The safety-walk prerequisite does not require the actual
policy rows/fields needed by its public procedure. See
`operations.asana_safety_walk_log/draft.json:815`. Match required source identities
and content coverage rather than treating any cell in the worksheet as sufficient.

The `sole` anti-hedging sweep was kept only where Luna replay did not worsen.
That is a useful regression signal, but cannot be the acceptance oracle:
incorrect Luna behavior must be allowed to lose score, and genuinely valid
alternatives—not just Luna's exact phrasing—must remain acceptable.

## Review and status freshness

`manifest-drafts/LEDGER.md` still reports 45 drafts. The current tree has 105.
Of 104 reviews, 45 declare a top-level draft hash: 37 match current bytes and
eight differ. Fifty-nine omit that hash. A mismatch may reflect a justified fix,
but then the review is not proof for the new bytes. The five installed manifests
have separately pinned package/draft tests, which are stronger than those stale
review metadata fields.

Many drafts remain explicitly gameable. The earlier three-task spot check found
11 of 14 attempted gaming paths worked. This is a diagnostic sample, not a
population failure rate; 59 reviews with recorded gaming is likewise not a count
of 59 uniformly severe defects. Both observations rule out treating 81 candidate
labels as 81 accepted manifests.

## Recommended next work

1. Repair the three shared evidence boundaries above with retained native
   counterexamples. Recheck affected read/write closure and ordering consumers.
2. Independently qualify candidate manifests against genuine alternatives and
   harmful executions. Close known gaming or explicitly retain partial status;
   prioritize installed rules and report counts based on actual accomplishments.
3. Refresh every review's exact draft/source revision, complete the missing sales
   review and regenerate the ledger from the filesystem.
4. Run and retain a source-frozen current integration-r6 regression and affected
   native replay/reload/rescore gates before broader installation or publication.
5. Advance action-credit coverage separately. Eighteen declarations have credit
   rules; 87 are outcome-only. Do not claim turn/token learning signal from those
   outcome checks until attribution and original-token consumer alignment are
   independently qualified.

No new Luna collection is required for these repairs. The environment sample
still has to meet the revised fifteen-per-category gate and then full-library
coverage. Qwen benchmarking, curricula, efficiency gating and the final 4B bank
remain downstream work.
