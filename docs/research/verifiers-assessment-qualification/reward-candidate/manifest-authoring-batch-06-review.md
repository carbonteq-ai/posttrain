# Manifest authoring: batch 06 public-input review

Reviewed 2026-10-04: **10 tasks, one executable guard proposal, zero accepted manifests and zero whole tasks qualified.**

The [JSON companion](manifest-authoring-batch-06-review.json) contains all ten policy inventories, data-only blueprints, ambiguities, adversarial cases and the proposed ContractSpec. Deferred blueprints are not executable declarations or counted reward coverage.

## Supported candidate

**hr.docusign_nda_collection — preserve already Signed tracker status.** The public request explicitly says to keep already signed employees marked Signed. A generic Sheets guard can detect a qualified update that changes NDA Status away from Signed on the same native initial record. Matching requires the declared tracker scope, row identity, native record ID, requested field and actual changed field. No employee-specific list or task function is needed.

The proposal covers this harmful update occurrence only. No-op Signed writes and unrelated edits do not qualify; missing native identity or capture remains unavailable. Replacement rows must not inherit the original employee's identity. Repair does not erase the historical violation. Deletion and every possible tracker mutation are not claimed covered.

This earns no positive completion credit. An unsigned employee's tracker can be set to DocuSign Sent without sending anything; the full goal needs a qualified NDA send, matching recipient/document, temporal linkage and retained tracker outcome.

One ContractSpec validates and roundtrips. **Four public bindings match and four independently modified bindings reject.** Native Task.score and recorded-rollout qualification were not run. The retained Sheets increment is being qualified separately.

## Task decisions

| Task | Public meaning and reusable next requirement |
| --- | --- |
| hr.docusign_nda_collection | Preserve Signed; send then mark DocuSign Sent. The guard above is supported. Need shared DocuSign send evidence and effect-to-tracker linkage for useful completion. |
| marketing.social_scheduling | Apply CMO's two-per-channel weekly cap and Sunday blackout, plus surviving lead instructions and holds. Several eligible posts compete for capacity; do not choose a hidden preferred subset. Need Buffer evidence, grouped capacity and compositional text/date checks. “Exactly 2 images minimum” and suffix ordering remain ambiguous. |
| operations.hubspot_churn_prediction | Compose three exact signals and VIP override after protections. Migration excludes all workflows; renewal text says outreach, so do not flatten their scopes. Need HubSpot evidence and a reviewed mapping from requested CSM task to available ticket tool. |
| sales.sheets_multi_channel_campaign_router | Explicit ranked policy chooses Zoom first, LinkedIn second, Email fallback. Email preference does not override a VP title's LinkedIn rule. DNC blocks outreach. Need ranked predicates and qualified actions before marking the route/date. |
| simple.email_zendesk_ack_reply | Create a Legal Team ticket for the contract inquiry and acknowledge on the original email thread. Need Zendesk creation and Gmail reply adapters plus flexible content facts; send-only evidence is not automatically reply evidence. |
| support.zendesk_gdpr_purge | Legal hold protects account and tickets; all requester tickets include solved ones. Need deletion-safe identity, complete ticket scope, qualified absence and real anonymization. Prefixing a subject alone is insufficient; log-everything and system silence need a recorded interpretation. |
| finance.invoice_reconciliation | Join invoice identity across both exports, resolve an explicit accounting correction, and report actual amount/presence differences. No public rule excludes all disputed invoices. Need correction-aware joins, union/absence and report coverage; customer-name variants are not amount mismatches. |
| hr.equipment_provisioning | CFO specifies department laptop, common peripherals and remote monitor; contractors get no equipment/tickets. Existing shipped equipment affects baseline. Need Jira/content evidence and explicit IT delivery authority; no home addresses or IT email are supplied. |
| marketing.ad_platform_audit | Pause low CTR only after fourteen days; deletion requires spend under 5000 or written Finance approval. Need registered Google Ads effects and explicit approval/average semantics. Refused-deletion reporting conflicts with system silence; preserve the conflict. |
| operations.hubspot_lead_qualification | Apply source-bound exclusions before Hot/Warm/Cold rules and aggregate eligible budgets once. Public tiers leave some size/budget combinations uncovered; no invented catch-all Cold. Need HubSpot effects, deduplication, totals and content checks. |

## Shared work worth doing next

1. Complete a small linked workflow: Zendesk acknowledgment or NDA sending. Add reusable service evidence and correlated outcomes before more narrow recipient/status checks.
2. Add bounded policy composition: ranked rules, explicit precedence, uncovered classifications and reviewed text facts. Preserve genuine ambiguity rather than encoding exact fixture prose.
3. Add finite union/anti-join, deduplication, correction resolution and grouped counts/sums. Summaries need explicit source populations and denominators.
4. Keep deletion/absence, baseline satisfaction, historical harms and retained outcomes distinct. Service absence is not permission to fabricate an empty initial inventory.

## Provenance

Pack: `/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/manifest-authoring-105/batch-06.json`
Exact SHA-256: `fa9c668bea1afdf6f9cfc96e24a8c183015b9e22fa7827a83e220c654c617f02`

Exact pack bytes and task order match the index. All ten public-input hashes pass. Selection identity was independently verified under both the exact-byte and canonical-JSON schemes.

The review used public requests, initial worlds, named tools and installed shared schemas. Hidden answers/assertions, task scorers and recorded episode contents were not consulted. Replay references are retained solely for later immutable qualification. No code, catalog, ledger, plan or earlier review was changed. Schema validity does not establish native replay correctness, whole-task success or learning.
