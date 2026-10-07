# Manifest authoring: batch 10 public-input review

Reviewed 2026-10-04: **10 tasks, one executable guard proposal, zero accepted manifests and zero whole tasks qualified.**

The [JSON companion](manifest-authoring-batch-10-review.json) contains the complete proposed ContractSpec and all ten public-policy inventories, ambiguities, deferred blueprints and counterexamples.

## Supported candidate

**finance.po_email_logging — prohibit appending an initially logged PO number.** The request explicitly requires checking for duplicates and skipping an already-present PO. The manifest uses the initial Purchase Order Log population keyed by PO Number and a qualified append to that same log, matched by exact business key. It contains no fixed PO outcome list or email parser.

Available task tools permit reading and appending, with no log deletion/update. This bounded guard therefore protects initial entries. It does not cover a second append of a PO first introduced during the current run; that requires dynamic duplicate consumption. It also does not prove new PO content, retained completion, all duplicate avoidance or useful positive credit. Missing/ambiguous keys remain unavailable.

**One ContractSpec admits, compiles and roundtrips. Five public bindings match; five independently altered bindings reject.** Initial table capture is qualified and closed with two complete PO keys. No native Task.score or recorded replay was run.

The duplicate Slack instruction conflicts with the higher-priority system instruction to handle exclusions silently. The proposed harm check depends only on avoiding duplicate writes; it does not reward reporting skipped duplicates.

## Task decisions

| Task | Public meaning and next shared requirement |
| --- | --- |
| support.helpscout_hubspot_deal_alerts | Critical requires closing within 30 days AND amount >10000; only critical alerts require AE email and Salesforce Task. Join actual contact/deal relationships, preserve amounts and exclude closed deals. Need service populations, grouped joins and notification fact coverage. |
| finance.payment_terms_tracking | Pinnacle terms are corrected to Net20; Legal clears Metro's dispute; a procurement-channel message finalizes CloudHost. Missing Slack author-role metadata leaves an authority question. Need QB Vendor state and source-bound effective updates. |
| hr.bamboohr_promotion_update | Approved/effective promotion facts differ from Pending sign-off and future approved dates. A scheduled April1 change is not an immediate March20 mutation. Need BambooHR state/effective-date semantics; do not invent a requirement to publish salaries in Slack. |
| marketing.conversion_tracking | Current closed-deals message, gclid, raised 10000 threshold and typed exclusions govern sends. Distinct Acme gclids are distinct deals. Need message-derived fact populations and shared Google Ads source registration; the task clock has no timezone. |
| operations.calendly_equipment_inspection | Highest overdue risk points to HVAC Unit3 and available certified Mike Chen. Book the exact UTC instant, then correlate Airtable/Notion/email/Slack. Need tie-aware ranking, certification joins and service evidence. |
| sales.apply_project_label | Exact spreadsheet code/HOLD rules conflict with a later VP scope expansion and hold release. Preserve that authority question. Mixed-code messages and policy emails complicate population definition; no hidden expected count. Need label membership and bounded token semantics. |
| simple.zendesk_resolve_email | Small full-task target: retained ZD-501 solved state plus a meaningful password-reset resolution email to Elena. Recipient matching alone cannot qualify the notice. Need Zendesk status evidence and generic message facts. |
| support.helpscout_reamaze_migration | User says active while playbook includes pending; review scope explicitly. Preserve all conversation content and map status/category, reuse contacts, add internal notes and report actual progress. Do not invent source closure. |
| finance.po_email_logging | Initial duplicate append guard is supported. Full task needs source-bound PO extraction, within-run duplicate handling and retained row content; system exclusion silence constrains duplicate reporting. |
| hr.buddy_assignment | Same department, sufficient tenure, longest-tenure selection and capacity one; Sales needs department-head fallback. Leave notes and near-boundary tenure wording matter. Need finite allocation, tie rules and meaningful paired/fallback notices. |

## Smallest next shared work

1. Complete the Zendesk resolution workflow through a reusable ticket state adapter and generic resolution-message facts. Verify the same requested ticket/issue and actual terminal state; do not substitute a recipient-only send.
2. Add source-bound message fact populations for POs and closed deals, with reusable field extraction and explicit ambiguity. Keep normative policy separate from generated expected outcomes.
3. Add bounded joins, dynamic duplicate consumption and capacity-aware selection. Unknown populations cannot prove absence or trigger a fabricated no-candidate fallback.
4. Preserve policy amendments, author authority and effective dates as evidence. A newer message, a named worksheet or an API acknowledgment alone cannot settle all eligibility or completion questions.

## Provenance

Pack: `/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/manifest-authoring-105/batch-10.json`
Exact SHA-256: `a99c6aa6fca047d97221b28679aec5811c74e9c7e651d352c8b5f4161d53e42c`

Pack SHA and ordered identities match the index. All ten public-input hashes pass. Exact-byte and canonical-JSON selection hashes were independently recomputed under their documented schemes.

Only public prompts, initial worlds, tools and shared schemas supplied policy/review input. Hidden answers/assertions, task-specific scorers and recorded episode contents were not consulted. Replay references are copied solely for later qualification. Only these two review files were edited; no source, catalog, ledger, plan or earlier review changed. Draft authoring and source admission do not establish installed coverage, native qualification or learning.
