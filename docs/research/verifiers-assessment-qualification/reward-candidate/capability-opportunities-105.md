# What to build next for the 105 reviewed tasks

The next shared addition is a small deterministic count/sum operation. It
answers: **which eligible records contributed, and what is their exact total?**
Existing per-record arithmetic cannot establish that a total includes every
required record exactly once. This is a concrete gap worth closing before
adding a larger query language.

This note supports [Milestone 4](../../../plan/automationbench-teacher-calibration-and-action-rewards.md).
The companion `capability-opportunities-105.json` records task membership and
public evidence anchors. Its counts describe opportunities for partial checks,
not completed tasks. Public requirements define the rules; recorded Luna paths
provide examples and counterexamples, rather than prescribing a solution path.

## What already works

The current environment supports typed initial collections, exact chained
lookups, per-member eligibility, exact decimal/date expressions, retained-record
outcomes, and bounded Jira/HubSpot creation evidence. Several older review notes
predate those additions. Refreshing a declaration is often the next step; a
stale gap list is not evidence that a new service adapter is needed.

The 105 tasks have been reviewed, with fifteen from each category. Thirteen have
installed partial declarations. Ten other catalog entries are Simple Opportunity
pilots outside this sample. No task has yet passed whole-task qualification.

Update 2026-10-04: aggregates now run inside native occurrence obligations, and
`finance.prepaid_amortization` revision `public_batch01_schedule_report_v4` adds
an outcome-only controller `Total amortization: $X` line component. Counts are
unchanged (13 installed partial declarations, 0 whole tasks); prepaid still
lacks entity-name and journal-line coverage. Next authoring: the environment repo's `manifest-drafts/` (see `LEDGER.md`).

## The first implementation

`contracts/aggregates.py` now recaptures one declared initial collection or sheet
population from the raw source. A finite specification selects `count` or `sum`,
an existing eligibility predicate, a declared unit and, for sums, an existing
value expression. Results retain contributing identities, source paths, raw
numeric values, and source/selector/specification digests.

A closed empty population can yield zero. A missing population cannot. Unknown
eligibility, incomplete membership, duplicate native identities or an invalid
selected value make the total unavailable. Excluded records do not need numeric
fields that the calculation never uses. Repeated writes do not create additional
members. The unit is a declaration, not an automatic currency conversion.

The standalone evidence boundary passes 59 owner tests, 31 independent boundary
tests and a 286-case root combined regression. Ruff, formatting and Pyright are
clean for the new files. The actual public-input replay and its source hashes
are retained under `.posttrain/state/verifiers-assessment-qualification/aggregate-prepaid-public-01/`.
Next connect it through existing manifest compilation and native assessment
preparation, recapture and publication. Do not attach credit merely because the
evaluator computed a total; the agent must still perform the requested action,
and the appropriate action evidence must establish its contribution.

## A real task that exposes the gap

`finance.prepaid_amortization` asks for a journal email containing “Total
amortization: $X”, with X equal to the sum of entries. Its public policy supplies
the straight-line formula and a newer February insurance multiplier. Under the
reviewed February interpretation, the insurance entry is $4,000 and the hosting
entry is $300. Those two entries sum to $4,300. A controller's Slack correction
overrides the software row's earlier skip note and corrects its total to $7,200:
the software entry is $600. Together these entries sum to $4,900. The lease has
not started, and the marketing retainer has no remaining balance.

Those values are examples derived from the public inputs, not a fixed expected
count or a production identity whitelist. Changing a contributing amount,
eligibility or membership must change the result through the same operation.
The three differently calculated populations can use separate specifications;
this example does not justify adding conditional-expression machinery.

This arithmetic proof would still leave schedule-update correctness, the
controller delivery, journal facts and authored-output guards to their own
checks. If the raw population cannot establish closure, the aggregate must
remain unavailable even when a human can see a plausible total.

## Separate increments that remain necessary

| Requirement | Small shared addition | What it must not assume |
| --- | --- | --- |
| Reconcile both sides | A full-outer key relation with matched, missing-side and ambiguous groups | Missing-side absence from a truncated query; first/last duplicate wins |
| Pick the most urgent eligible item | Filtered extrema with explicit ties and unknown competitors | Row order as a tie-break; an inferred severity policy |
| Read required policy before acting | Exact returned-source scope and read-completion-before-action-start evidence | Source availability means the agent read it; state revisions prove non-overlap |
| Verify a multi-service workflow | Qualified retained objects and identity-bound relationships | An audit action is a persisted object; a generated ID alone proves the requested relationship |
| Verify a report or message | Exact fact/set coverage where structured evidence suffices; bounded semantic interpretation only where necessary | Correct destination means correct content |

Payment reconciliation is a particularly clear next relation case: its public
request explicitly defines amount mismatch, bank-only and QuickBooks-only
categories. The typed QuickBooks initial population already exists. A relation
operation would expose duplicate ambiguity instead of silently resolving it.

Conversion tracking illustrates a different gap: a complete source-bound deal
population and qualified Google Ads effects are required. Its failed recorded
Salesforce search cannot establish either the public policy or an environment
defect. Buyer email alone cannot merge a main deal and upsell with different
gclids.

## How the next batch is chosen

Use the public-requirement matrix to choose ten tasks covering conditional
selection, exclusions, aggregation, reconciliation and multiple services.
Simple tasks remain useful regression fixtures. Required output guards remain
visible on every task, but their shared uncertainty must not erase useful
deterministic partial work or dominate every capability ranking.

Record declaration refresh, missing mechanics and unresolved policy separately.
Replay the original traces and genuine simulator alternatives, then preserve
unsupported obligations in each acceptance entry. Whole-task qualification
requires all public obligations and applicable guards, native findings/credit,
reload/rescore and independently reviewed failures and alternatives.
