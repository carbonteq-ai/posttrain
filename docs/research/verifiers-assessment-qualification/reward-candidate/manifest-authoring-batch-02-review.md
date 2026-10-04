# Manifest authoring: batch 02 public-input review

Reviewed 2026-10-04: **10 tasks, one partial component proposal, zero accepted manifests in this review, zero whole tasks qualified.** No catalog, implementation or recorded episode was changed.

The [JSON companion](manifest-authoring-batch-02-review.json) contains the actual ContractSpec candidate plus each task's goals, explicit prohibitions, goal constraints, evidence requirements, public sources, unresolved interpretations and adversarial variants. Deferred task blueprints are authoring specifications, not executable operators.

## Candidate ready for native qualification

**sales.docusign_void_resend — CRM amount only.** The unread Apex request and the opportunity's negotiation note both specify $175,000. Existing `record.fields_equal@1` and `verified_transition_once@1` can check that exact target update. The proposal binds the prompt, request, target opportunity and note.

This does not qualify voiding, resending, template selection, three-year terms, premium support or the confirmation email. No public instruction requires a Closed Won stage update. The candidate must not invent one. Initial satisfaction and break/restore remain subject to the existing transition-credit policy.

## Decisions across the ten tasks

| Task | Public interpretation and blocking capability |
| --- | --- |
| operations.fleet_vehicle_maintenance | Convert mileage, apply the explicit odometer correction and exclude Out of Service/HOLD vehicles before scheduling. The month/year of “the 14th” is absent; no task clock is supplied. Need numeric/unit decisions and calendar effects. |
| sales.deal_escalation | Score only the identified thread. Spreadsheet weights produce 8; the analyst proposal produces 12. Both route to leadership plus Zoom, but the adopted score policy is not explicit. Need thread grouping, derived score, content and Slack/Zoom effects. |
| simple.mailchimp_email_request | Sender address and Newsletter list_001 are explicit. This is a small shared Mailchimp subscription adapter target. No confirmation email is requested. |
| support.gorgias_refund_processing | Exact order identity, thresholds, category rules and explicit overrides matter. Repeat-refunder consequences, threshold equality, unworn evidence and qualifying-tag rules are not fully stated. Do not copy hidden expected classifications. Need joins, derived decisions, draft/Jira/log effects. |
| finance.wave_client_statement | Aggregate unpaid balances by customer; the $100 threshold is explicit. Aging basis/buckets are absent. A random email to a low-balance client is not a prohibited statement. Need invoice grouping and statement-content evidence. |
| hr.exit_interview_scheduling | Specific involuntary exclusion overrides general SOP. The direct roster-forwarding request conflicts with supplied organizational policy; record that authority conflict before promotion. Business-day calendar, overdue interview slots and missing manager handling remain open. |
| marketing.newsletter_sponsor_invoicing | Apply finance hold and Ridgeway's same-day exception; preserve invoice/audit codes and source amounts. The system's silent-exclusion instruction overrides the audit email's request to list skipped sponsors. Need date decisions, Sheets effects and grouped content checks. |
| operations.facility_incident_triage | Filter holds/resolved incidents before selecting one most-critical incident; Monday's contact is distinct from Sunday's despite the same manager. Severity ordering is an assumption, and tied minima need an explicit policy. Need unique selection and Jira/SMS/Slack effects. |
| sales.docusign_void_resend | CRM amount is independently expressible now. Remaining envelope workflow needs service effects, source-bound template authority and linked terms/signers. Beta's public “do not void” restriction remains separate. |
| simple.email_hs_create_contact | Create the contact from sender and supplied name/company/role/phone. No outreach or subscription is requested. Need qualified HubSpot creation and source-fact projection; duplicate/upsert behavior is unspecified. |

Publicly justified arithmetic examples are recorded as review examples, not production outcome lists. A changed amount, identity, date or policy must flow through shared operators or require a new source-bound authoring decision. It must never select new task-specific Python.

## Minimum shared work

1. **Service effects:** finish the shared draft/Slack/Sheets work from batch 01; add Mailchimp subscriptions and HubSpot contacts as small, direct-policy cases. Extend to Calendar/Zoom, Jira/Twilio and DocuSign only with their own installed-schema and ACK/result/state qualification.
2. **Derived decisions and bounded populations:** decimal/unit parsing, explicit dates, closed sums, exact joins, thread counts and unique eligible minima. Ties and incomplete populations stay unresolved. Business calendars and aging buckets require authored parameters.
3. **Content and composition:** source-bound facts, verbatim inclusion, purpose evidence, generated-output joins and grouped summaries. Recipient-only evidence does not justify useful-action credit or privacy harm. Factual occurrence remains distinct from retained terminal state.
4. **Policy authoring:** bind rules and exceptions to public sources and record authority conflicts. No universal sender hierarchy, “latest message wins” rule or rigid task prose parser.

These are six grouped capability families in the JSON, not a proposal to build a new framework or all capabilities at once. The smallest immediate acceptance path is the CRM amount component and the two simple service-adapter tasks.

## Provenance and limits

Pack: `/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/manifest-authoring-105/batch-02.json`  
Exact SHA-256: `eea8b69e67adae6bbef920168d57006e1d4f761896d09f7492907ae3c2e80222`

The exact pack hash and task order match the batch index. The index's selection hash covers exact selection-file bytes; the pack's selection hash covers canonical JSON contents. Both were independently recomputed from `cross-category-selection.json`: `13d326e...` for bytes and `4a793692...` for canonical contents. They are distinct hash schemes, not a selection mismatch.

Only public prompts, initial worlds and named tools supplied task policy. Hidden assertions/answers, task-specific predicates and recorded episode contents were not used. Episode references are retained solely for later exact-SHA replay.

The candidate validates against the current ContractSpec, and all four public source bindings match. Its array-position bindings conservatively reject source reordering; stable-ID binding is preferable for variants. Schema and source-binding validation do not establish native task qualification, whole-task coverage or learning.
