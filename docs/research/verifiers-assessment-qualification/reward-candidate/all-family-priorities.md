# Efficient implementation across the full development inventory

Advisory, 2026-10-03. The living authority remains
[the calibration plan](../../../plan/automationbench-teacher-calibration-and-action-rewards.md),
especially Milestone 4. This note proposes implementation order; it does not
replace acceptance gates or reduce the required scope.

The retained development index contains **700 traces across 46 family buckets**.
The development design inventory also contains nine unavailable traces. The 53
Astra task contracts are representative cases, not exhaustive definitions for
every task in a bucket. A working example must not mark its whole family complete.

## Next parallel implementation

| Lane | Concrete next work | Actual retained trace | Shared work reused |
|---|---|---|---|
| Finance | `finance.cash_flow_forecast`: per-customer probability arithmetic, per-vendor amounts, ending balance, truthful CFO report | Official full; three sheet reads and a send. New predicate tests exercise this exact SHA-bound trace. | Sheet population identities, qualified Gmail sends, `EffectIndex`, baseline-aware delivery attribution |
| Operations contracts | `operations.contract_renewal_pipeline`: eligible manual contracts, renewal template/signers, sent envelope and summary | Official full; three envelope creations, template lookup fallback and send | New root-owned DocuSign send adapter plus existing sheet/Gmail helpers |
| Operations access | `operations.access_request_validation`: email/department join, manager rank, Asana creation plus required section, status and denial | Official full; three task creations, three section moves, five writes, two emails | Asana action-record abstraction from Simple; exact row identities and Gmail effects |

With four slots, use root for shared integration, two workers for disjoint domain
files and the critic for independent policy/counterexample review. Do not let
multiple workers edit the publisher, registry or source qualification code.

Three high-value cases should advance through public-policy review and native
counterexamples rather than another reference-model run. Repair a capability
only after reproducing its failure against the declared public source:

- `marketing.conversion_tracking`: its selected trace stops after an empty
  Salesforce query, official zero. The public fixture supplies closed deals in
  Gmail, so this is evidence of wrong-source behavior, not a demonstrated query
  defect. Review the actual public conversion procedure,
  source service, exclusion identities and native Google Ads upload abstraction.
  Execute eligible and excluded upload counterexamples; this trace supplies a
  failure case, not a positive completion demonstration.
- `sales.weighted_priority`: selected official-zero trace performs discovery and
  an Intercom lead query. Qualify the complete intended Hot Salesforce population,
  weighted arithmetic and tie rules with public fixture facts and independent
  native executions before attributing a winner or outreach.
- `operations.conference_room_booking_conflicts`: selected official-zero trace
  sees empty calendar discovery despite fixture events. Fix/qualify public event
  access and candidate coverage first, then interval overlap and rescheduling
  identity. An empty discovery response is not a no-conflict proof.

The access, renewal and ten Opportunity-update batches now have native bounded
publishers. The next conditional lanes are conversion tracking and weighted lead
selection. Current Lead models retain the opened-email, activity-age and score
fields; do not assume the older suspected schema gap is still present. Preserve
original-zero traces as failure evidence and execute correct alternatives locally.

## Reuse to build once

1. **Scope and prefix qualification.** Share source inventory, initial projection,
   revision linkage and terminal/prefix reconciliation. Keep recording coverage,
   public authority and effect-model support independent. Domain code still owns
   candidate membership and guard meaning. Cache decoded snapshots per trace;
   do not repeat JSON decoding or whole-pool official replay inside each rule.
2. **Native effect adapters.** Gmail send/draft/reply, scoped sheet row mutation,
   DocuSign sent envelopes, Asana task/section records, calendar events and exact
   CRM record changes cover many tasks. Their acknowledged operation/result/object
   joins are reusable; business success is not inferred from a generic mutation.
3. **Baseline-aware obligation attribution.** Fix initial obligations before
   rollout, preserve initial satisfaction, credit one real accomplishment, record
   every harmful execution and avoid rewarding break/restore cycles. Reuse this
   rule explicitly rather than rediscovering it in each domain.
4. **Quantitative checks.** Exact decimal values, declared percentage denominators,
   identity joins and logical clocks support finance, selection and calendars.
   Rounding, exchange rates, time windows and title hierarchies remain authored
   task rules, not generic inferred defaults.

Simple expansion is worthwhile as a secondary lane using these effect adapters,
not as the primary demonstration of conditional reward redesign. Direct record
creation/update tasks can use reviewed parameterized contracts; do not generate
definitions mechanically from hidden assertions or assume all tasks in a broad
bucket share one requested operation.

## Breadth followed by closure

Build a task-level matrix from the immutable development index and public task
definitions. For every family, list its actual tasks, sources, known meaningful
goal/guard components, effect adapters, executed counterexamples, native scorer
integration and unresolved public/capability gaps. Tie records to implementation
and evidence revisions. Distinguish “representative case tested”, “task partially
supported” and “all required task predicates qualified”. Returning unavailable
for an entire unexamined task does not count as family implementation.

After the next conditional cases, group remaining work by actual adapter needs:
CRM/record mutations; notifications and report arithmetic; envelopes/calendars;
cross-platform migration/privacy; publication/conversion effects. Keep a separate
repair lane for missing public facts and capability defects. Examples include
Salesforce currency/freshness in reconciliation and Zendesk description/history
redaction in GDPR purge. A judge cannot supply these missing records or mutations.

Use the saved development episodes for cheap real replays. An official-zero or
truncated trace is still useful negative evidence. Execute valid alternative
paths and harmful variants directly against fresh native simulator worlds; then
score, serialize, reload and reassess through the production producer path.
Preserve official metrics and never overwrite original collection artifacts.
No extra Luna collection is needed for these first implementation waves.

Run domain tests during local changes. Run shared HR/Simple/marketing/support
regressions after shared integration, and source-sensitive calibration checks in
one coordinated stable window. A final freeze needs the full native and library
qualification gates; smaller passing suites do not substitute for them.

The wider environment migration remains after tested reward design. Qwen pool
admission still requires accepted native reassessment, required goals and guards,
qualified original budget/source evidence and the approved verifier identities.
Do not start student collection merely because the next three case slices pass.
