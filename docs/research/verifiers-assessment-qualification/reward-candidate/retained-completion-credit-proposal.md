# Credit for completing a retained row goal

Status: proposal, 2026-10-04. The retained outcome check is implemented; this credit policy is not. Scope is AutomationBench Sheets, using shared capabilities and manifest data. It introduces no task-specific evaluator or native Verifiers API.

The policy `retained_completion_once@1` assigns one positive contribution to a qualified execution that completes an initially unmet row goal, provided the finalized row still satisfies that goal. The retained assessment proves the outcome. Transition evidence identifies an allocation recipient. Neither fact alone is enough for credit.

## Keep the existing pieces

Reuse `RetainedRowCheck`, `SheetRetentionEvidence`, qualified `SheetEffectSource(kind="update")` facts, current-attempt assessment receipts, `CreditPlanningContext.prior_assignments`, and native execution recipients. Initial table evidence and terminal row evidence already have independent source recapture. No separate state store is needed.

Add one solo credit policy with a nonempty, unique `goal_fields` list:

```json
{
  "check": "schedule-balance",
  "policy": "retained_completion_once@1",
  "channel": "schedule-completion",
  "goal_fields": ["Amortized to Date", "Remaining"]
}
```

The referenced check must be a `RetainedRowCheck` backed by a Sheets update source. Validate that each declared goal field is read by `retained_when`, including paths inside typed derived expressions. Require the predicate to read retained row fields. This prevents a constant predicate or an unrelated metadata field from authorizing completion credit. Field relevance still requires public-policy review: schema validation cannot prove that the author chose a useful goal.

Existing credit policies must reject this new field. This keeps their meanings unchanged. Append/creation completion is a later policy with its own baseline proof; do not silently treat append as update.

## Eligibility and witness selection

For each retained goal instance:

1. Require the exact current complete assessment attempt, intact public source bindings, an authenticated outcome receipt, `required=True`, `status=valid`, and terminal value 1. Failed or unavailable current attempts cannot fall back to older successful assessments.
2. Resolve the original native row ID from qualified initial evidence. Evaluate the same `retained_when` predicate with the original initial cells as `retained`, retaining the original request and lookup contexts. It must be known false. An initially true or unknown goal receives no credit. Do not let later source edits redefine the expected value or applicability.
3. Consider source-qualified update facts on that exact native ID and selected worksheet. Each must have the acknowledged execution, strict unit revisions, requested fields, changed fields, and before/after cells already established by the Sheets adapter.
4. Evaluate the predicate against those before and after cells, with the same frozen request and lookup contexts. Before must be known false and after known true. Unknown is not false. Require a nonempty intersection of requested fields, changed fields and declared goal fields.
5. Select at most one completion witness: the latest **observed qualified** false-to-true completion by applied revision. Equal revisions belonging to different executions are ambiguous; abstain from allocation rather than breaking the tie by an arbitrary ID. The finalized original row must independently satisfy the goal.

This supports split writes. An update to the first financial field can leave the conjunction false; the update to the remaining field can complete it. It rejects Notes-only updates after somebody else completed the balances, no-ops, and formatting-only writes to an already correct numeric goal. Typed predicates supply numeric equivalence instead of imposing raw text equality.

Choosing the latest observed completion lets a qualified repair be the recipient when there is no prior contribution. It does not establish that this execution was the only cause of the final state. Missing later receipts can leave action inventory coverage unavailable while a known qualified completion and independently finalized outcome remain usable. Publish that limitation; do not call the witness the last actual write or infer uninterrupted retention.

## Once-only consumption and native projection

Use the existing required-effect planning pattern, with a separate rule identity and explicit stable consumption record. Its key contains:

- episode ID;
- contract digest and check ID;
- original native record ID, initial candidate instance key and population/retention selector digests;
- channel and signal identity.

The key excludes snapshot ID, assessment run/attempt/invocation IDs, witness execution, witness effect ID and current row position. Reassessment, reload, a different qualifying repair, or later row relocation must not create another contribution. The initial instance key remains stable even when the final row moves. A new record at the old position cannot consume or satisfy the original obligation.

The rule configuration separately records the selected allocation witness, required/initial/terminal predicate results, goal fields, contract/source/selector bindings and retained parent assessment ID. Validate the receipt against recomputation before planning. Bind the projected target to the exact execution occurrence and source membership; never accept an arbitrary native execution merely because it exists in the trace.

Prior valid contributions with the same consumption key intentionally produce no new request. Conflicting signal/value/channel metadata fail closed with a named conflict. Missing or failed prior assignments do not count as consumption. Multiple current records for the same instance need explicit conflict handling, using the existing current-attempt rules.

Use one positive unit contribution, coarse execution/turn allocation and an explicit transformation such as `retained_completion_identity@1`. Do not infer token masks, causal weights, or distribution across all earlier writes. If several row goals select one execution/channel, use an explicit aggregation/joint-credit contract or fail with a named aggregation requirement; do not silently sum or discard them.

An earlier prefix contribution stays historical. If the row is later damaged, the new retained outcome becomes 0 and cannot mint a new positive contribution. Repair can restore outcome 1, but the same consumed key prevents a second contribution. This policy does not retroactively rewrite historical credit or manufacture a negative contribution. Harm assessments remain independent.

## Acceptance tests

Use actual native Task scoring, serialized reload and repeated scoring alongside operator fixtures:

1. Combined and split financial updates reach the same terminal goal; one recipient is selected.
2. Notes-only, successful no-op and numeric reformatting receive no completion credit.
3. Initial correct state followed by break/restore has outcome 1 and no new credit.
4. Correct completion followed by damage has terminal outcome 0 and no new contribution.
5. Damage then repair restores outcome 1; the qualified repair can receive the first contribution, but reassessment or another repair cannot multiply it.
6. Original record deletion or same-position replacement fails the goal; same native ID relocation preserves it unless the manifest explicitly constrains location.
7. Missing ACK never creates a witness. Unknown before predicate, unknown initial baseline, unresolved original ID and ambiguous revision ordering suppress allocation.
8. Known terminal success can survive missing unrelated action receipts; unfinished or missing terminal state cannot authorize credit.
9. Typed derived predicates retain frozen public values and admit numeric-equivalent formatting. Public policy/source changes require reviewed rebinding.
10. Coherent retained receipt forgery, substituted raw inputs, copied-model type forgery, foreign current attempt, arbitrary target retargeting and changed witness consumption all fail or produce the intended no-op.
11. One execution completing multiple goals requires explicit channel/aggregation handling. No duplicate contribution is emitted through separate instances or histories.
12. The original scalar scorer is unchanged. Outcome, credit availability and action-inventory coverage are reported independently.

## Limits and alternatives

The simplest alternative is outcome-only publication, which is already useful and remains valid when allocation is uncertain. Keep that fallback. Crediting every write to a relevant field would reward partial or damaging edits; crediting the final write would reward Notes or formatting; crediting every historical successful state would multiply repaired goals. A bounded false-to-true completion allocation avoids those failures without claiming a general causal explanation.

This policy does not implement dense reward for each partial improvement, partial-field credit, training advantages, retrospective credit correction, transaction rollback or policy interpretation. Those require separate, explicitly reviewed semantics. It provides one reusable completion recipient for an authenticated retained row goal.
