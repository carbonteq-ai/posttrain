# Manifest authoring: batch 01 public-input review

Reviewed 2026-10-04. Ten public task inputs reviewed; **two partial component manifests proposed, zero accepted, zero whole tasks qualified**. The JSON companion contains the actual schema-shaped proposals, each task's goals, prohibitions, public authorities, unresolved interpretations, capability needs and adversarial cases. No catalog or task Python changed.

Batch: `/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/manifest-authoring-105/batch-01.json`
SHA-256: `d865aacbaac51c83441f852941b06116303d18bf3973bc1c8b265144c1c8e50d`

Only public prompts, initial worlds and named tools supplied policy. Retained episode references are copied for later qualification; their contents and hidden assertions/answers were not used. Positive unit-test results establish the operator boundaries described below, not learned behavior.

## Concrete candidates

| Task | Proposed component | What remains open |
| --- | --- | --- |
| sales.multi_hop_lookup | Set the uniquely identified Meridian Corp Platform Deal to Closed Won using existing record checks and once-only verified transition credit. | Routing, latest hierarchy/FX, notification content and support escalation scope. |
| simple.email_airtable_customer_welcome | Observe a fresh acknowledged Gmail send addressed to the customer identified in the onboarding email. **No credit declaration.** | Welcome intent, Airtable creation, correct record fields and whole-task completion. An unrelated invoice or cancellation to that recipient must never count as a useful welcome action. |

The delivery proposal currently measures the explicit `to` recipient field. Its zero is only absence of that bounded observation, not proof that the user task failed. A review may widen this observation to other recipient roles, but must preserve role visibility rather than silently merge them.

The proposals use frozen public source bindings as admission guards. A digest proves identity, not a correct interpretation of policy. These first proposals bind an array element by index and therefore conservatively reject a reordered source; ID-addressed public-source binding should replace this scaffolding for reusable variants. No expected recipient or entity name belongs in Python. The customer address is an authored fact directly cited from the public email, not an inferred answer from a recorded run.

## Review of all ten tasks

| Task | Public interpretation and decisive gap |
| --- | --- |
| finance.prepaid_amortization | Controller straight-line policy, explicit February insurance multiplier, and newer Software License correction must compose. Decimal arithmetic, period eligibility, schedule writes and aggregate journal validation are missing. Do not freeze calculated amounts into per-row predicates. |
| hr.job_posting_distribution | Approved-only requisition population is clear. Recruitee posting, Slack announcement and Gmail **draft** need distinct adapters. The careers-list address is absent; do not invent it. |
| marketing.product_launch_channel_plan | Channel approval windows are in **nested Slack channel messages**. LC-004's edit does not explicitly revoke its otherwise timely approval; no blog publishing tool is listed. Need date checks, publication/coordination evidence and an explicit capability decision. |
| operations.linkedin_abm_outreach | Tier1 inactivity, seniority, partnership/legal holds and blocklist compose over accounts and contacts. Invites are per contact; AE tasks per account. Need registered populations, bounded joins, date/title checks and several service effects. |
| sales.multi_hop_lookup | Closed-Won component is ready for qualification. Latest tier and FX are derived from dated keyed records. Public routing text does not establish whether a **parent's** Critical case triggers the target account's escalation notice. |
| simple.email_airtable_customer_welcome | Recipient is stated in the message body, unlike the onboarding sender. Current delivery observation is factual and partial; no welcome-purpose credit. Airtable and source-bound text extraction remain gaps. |
| support.reamaze_cross_platform_dedup | Require same customer **and** same issue, with alias mapping. Same customer/different issue and same issue/different customer are counterexamples. Need service populations/writes and qualified issue equivalence; do not encode four expected fixture pairs as the rule. |
| finance.expense_split_allocation | Compose method policy, item override, exclusions and updated headcounts. Need expense×department obligations, decimal proportional arithmetic, conservation and logs. Residual-cent rules are unspecified. |
| hr.calendly_manager_office_hours | Supplied procedure requires owner cancellation requests rather than direct cancellation. Latest org change leaves Platform alias/Product replacement unresolved. Generated scheduling URLs must join to later notifications; event-type creation alone may not prove weekly availability. |
| marketing.social_content_calendar | Platform mapping, manager exception, director hold and item restrictions conflict. Operative Slack timestamps are **Jan28**, after the **Jan27** request. Next-week planning may permit the Feb3 embargoed item after release. Do not copy an old fixture exclusion list or assume latest message always wins. |

All tasks inherit the system requirement to summarize only acted-on items where that instruction is present. That is a distinct output-content constraint, not permission to hide assessment coverage. Source amounts quoted verbatim are distinct from derived amounts that require a declared arithmetic/rounding policy.

## Small implementation sequence

1. Qualify the two concrete component proposals through native scoring, reload/rescore and tamper cases. Keep their coverage denominators separate from whole-task success.
2. Add reusable Recruitee/Airtable action-record adapters, Gmail draft and Slack post evidence, plus bounded required-fact inclusion. Compose HR/simple manifests using source projections. No per-task checking function or forced entire-message wording.
3. Add Sheets append/update evidence and effect-output joins; then decimal arithmetic, date windows, latest-as-of and finite sums. These unlock multiple finance/marketing tasks with the same operators.
4. Extend registered record populations and bounded joins. Introduce optional generic semantic assessment only where issue equivalence or personalization genuinely requires it. Its confidence/disagreement must remain distinct from deterministic facts.

The JSON lists capability requirements at a finer grain so the next implementation can choose one bounded increment; it is **not** a commitment to build all operators now or a new generic framework. Each missing adapter needs installed-schema and ACK/result/state qualification. Each expression operator needs bounded semantics, missingness and adversarial variants. Unresolved public authority is a manifest review issue, not a missing task-specific Python escape hatch.

## Required qualification

- Source and selector binding, native IDs, complete finite populations, duplicate/ambiguous identity, and empty-versus-unavailable evidence.
- Correct effect and wrong effect; missing ACK/result; repair and later incomplete capture; initial satisfaction; duplicate actions; once-only credit across reload and rescoring.
- Variants changing names, dates, amounts, ordering and public policy without reusing per-fixture outcome lists.
- Component truth separate from task completion and useful-action credit. Purpose-free recipient observations carry no reward credit.
- Native replay against the retained episode SHA only after public policy review. Replay behavior is evidence about execution, never the authority for desired outcomes.

## Operator work handed off

The shared `contracts/obligations.py` evaluator and its test file are ready for parent integration: 40 obligation cases pass; obligations, populations and predicates total **124 passing tests**. Scoped Ruff and Pyright pass. This includes genuine simulator Gmail send/initial-record population fixtures and a SHA-bound Luna access replay of raw capture evidence (no heavyweight native rollout replay).

Default occurrence semantics require observed initially-unsatisfied evidence for action credit. Explicit `new_occurrence` semantics require a fresh episode action, reject an initial-discharge predicate, preserve initial status as not applicable, and select at most one qualified witness per stable obligation. Neither mode claims retained terminal state. Missing initial evidence is never replaced with a literal false assertion.

These unit checks do not qualify the two new task manifests or resolve the policy ambiguities above.

Detailed source and proposal data: [manifest-authoring-batch-01-review.json](manifest-authoring-batch-01-review.json).
