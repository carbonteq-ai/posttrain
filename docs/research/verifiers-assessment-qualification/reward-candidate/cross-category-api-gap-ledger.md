# Cross-category API gap ledger

The next gate uses 105 development tasks: 15 from each of AutomationBench’s seven domains. It checks whether the new assessment and credit API can carry real task evidence across varied workflows. Selection is complete; this document does not claim these tasks have qualified reward manifests.

The source inventory contains 709 development tasks. Seven hundred have an indexed, hash-verified episode still present on disk. The other nine are support collection gaps. Every selected episode was rehashed against the review index; all 105 matched. No reserved task payload was used.

## How tasks were selected

The selection first takes one representative of every recorded family within each domain. It then fills the domain to 15 tasks by maximizing previously unseen native tool operations. Ties favor an existing public-policy review, then the task name in lexical order. API calls are grouped by method, so changing a record ID does not create fake diversity. Ordinary Zapier operations use the captured underlying tool name. Unsupported capture entries contribute no operation label. The HR index has one broad family, so HR diversity comes from actual operations rather than that family label.

The selection algorithm is `family_then_native_operation_greedy@1`. Its exact description and SHA-256, the two input file hashes, and every task’s episode path/hash are recorded in [cross-category-selection.json](cross-category-selection.json). Re-running selection against changed input creates a different source binding; do not silently append tasks to the frozen gate.

## Inventory and selected evidence

| Domain | Development tasks | Retained candidates | Selected | Families selected | Distinct operation labels | Full / partial / zero |
|---|---:|---:|---:|---:|---:|---|
| finance | 98 | 98 | 15 | 7 | 32 | 3 / 6 / 6 |
| hr | 100 | 100 | 15 | 1 | 43 | 2 / 11 / 2 |
| marketing | 89 | 89 | 15 | 7 | 40 | 2 / 11 / 2 |
| operations | 83 | 83 | 15 | 5 | 68 | 1 / 13 / 1 |
| sales | 88 | 88 | 15 | 8 | 66 | 3 / 11 / 1 |
| simple | 164 | 164 | 15 | 9 | 33 | 12 / 1 / 2 |
| support | 87 | 78 | 15 | 9 | 62 | 0 / 14 / 1 |

The 105 tasks expose 195 distinct captured operation labels across domains. Twenty-three have a current outcome marked verified in the index. Failed and partial rollouts are valuable API-gap inputs: they expose harmful actions and missing evidence. This selection does not grant Qwen training eligibility, which still requires the separately defined verified Luna success gate. Support has no full outcome verified in this index; that is an observed collection result, not a proof that its tasks are unsolvable.

## What counts as passing this gate

1. Replay each selected episode through native wire parsing and source capture, verify the file hash, and preserve the source bytes. Record any unsupported representation as a named gap. Do not skip it or invent evidence.
2. Transport independently declared findings, including unavailable findings, through native requests, output receipts, serialization and reload. Preserve the official episode reward. A transport probe passing does not qualify task policy or attribution.
3. For each supported mechanism, execute a fresh simulator success and a harmful or incomplete alternative. Check exact acknowledgements, persisted effects, immutable initial populations, and declared keys.
4. Check that harmful action credit survives later repair, unknown evidence stays unknown, and repeated scoring does not consume the same event twice. Test wrong source/selector/instance, stale attempts, missing acknowledgements, and conflicting current results.
5. Report results per task and mechanism. Fifteen transport probes per domain satisfy only the breadth requirement. Reward coverage requires a reviewed manifest and qualified evidence adapter for each declared check.

## Shared capabilities to test first

Fixture correction, 2026-10-04: current canonical Jira creation exposed a public
Gorgias refund prerequisite absent from its initial world. Its explicit FIN
destination is now declared as initial project data, with API/Zapier and original
assertion compatibility tested. Old retained worlds are not hydrated with that
project and remain bound to their earlier fixture. This does not qualify the
whole refund manifest or change the frozen 105-task historical selection.

| Capability | Examples in this selection | Smallest useful genuine check | Remaining API gap |
|---|---|---|---|
| Finite tables, exact lookup and policy precedence | conversion tracking, email suppression, prepaid amortization, multi-hop lookup | Freeze source rows; duplicate/missing keys stay unavailable; independently known prohibited effects survive another row’s gap | Policy-bearing Gmail sources and arithmetic/value transforms need explicit typed capabilities |
| Native record updates and linked creation | Salesforce contact/account updates, HubSpot deals, Airtable customer welcome | One acknowledged mutation with matching persisted fields; already-correct and break/restore cannot mint progress | Record adapter currently qualifies Opportunity only; other object/service schemas need reviewed extensions |
| Notifications, drafts and replies | Gmail invoice email, Zendesk acknowledgement, support draft responses | Native send/reply acknowledgement plus persisted recipient/content; a draft or label edit cannot masquerade as send | Generic manifest notification evidence adapter and explicit literal content checks |
| Task creation and routing | Asana sprint section, Jira bugs/accessibility, Trello support/vendor hold | Separate creation and routing witnesses; wrong target is harmful only under declared policy | Asana is supported; Jira/Trello/Monday action adapters and typed routing joins remain |
| Calendar scheduling, conflicts and follow-up | Zoom calendar sync/conflict, Calendly manager hours/no-show, demo scheduling | Bound event identities and logical interval; no host clock fallback | Calendar/time-window operators and service-specific acknowledgement semantics |
| Document lifecycle | DocuSign void/resend, NDA collection, contractor offboarding | Separate acknowledged send/void from actual signature; preserve historical harm | Generic manifest DocuSign lifecycle adapter and public effective-period contracts |
| Financial and commercial calculations | expense splitting, reconciliation, tax remittance, billing | Exact source amounts, reviewed decimal/rounding policy, independent excluded-record guard | Arithmetic transforms, money/unit types and deterministic summaries |
| Support migration, retention and deletion | cross-platform dedup, GDPR purge, HelpScout/Reamaze migration | Preserve source identity and required history; partial destination creation is not full migration | Generic support record/action adapters and explicit retention/deletion evidence |
| Model-assisted/semantic workflows | proposal customization, feedback analysis, brand mentions, call transcription | Separate deterministic identities/effects from bounded semantic assessment | Judge integration stays optional; neither vague prose nor model output invents deterministic authority |

Cheap pure checks should run synchronously over one source-bound capture. Build exact-key indexes once and reuse them. Share one input view and one evaluation per check; native outputs select declared instances. Parallel work should split adapter ownership by service, while one owner maintains schema/engine integration. Real simulator tests need no new model collection.

## Selected tasks

### finance

- `finance.prepaid_amortization` — full; finance.budget-forecast-and-analytics
- `finance.expense_split_allocation` — partial; finance.expenses-and-reimbursements
- `finance.wave_client_statement` — zero; finance.invoicing-billing-and-receivables
- `finance.xero_vendor_onboard` — zero; finance.payment-and-vendor-controls
- `finance.monthend_journal_entries` — full; finance.reconciliation-and-close
- `finance.sales_tax_remittance` — partial; finance.tax-and-statutory-reporting
- `finance.escrow_tracking` — zero; finance.unresolved-workflows
- `finance.contract_billing` — partial; finance.invoicing-billing-and-receivables
- `finance.invoice_reconciliation` — zero; finance.invoicing-billing-and-receivables
- `finance.qb_sales_receipt_batch` — partial; finance.invoicing-billing-and-receivables
- `finance.subscription_billing` — partial; finance.invoicing-billing-and-receivables
- `finance.deferred_revenue_tracking` — zero; finance.budget-forecast-and-analytics
- `finance.payment_reconciliation` — partial; finance.payment-and-vendor-controls
- `finance.payment_terms_tracking` — zero; finance.payment-and-vendor-controls
- `finance.po_email_logging` — full; finance.payment-and-vendor-controls

### hr

- `hr.job_posting_distribution` — partial; hr.all-development
- `hr.calendly_manager_office_hours` — partial; hr.all-development
- `hr.exit_interview_scheduling` — partial; hr.all-development
- `hr.candidate_research_dossier` — partial; hr.all-development
- `hr.trello_recruiting_event_coordination` — partial; hr.all-development
- `hr.benefits_open_enrollment_processing` — zero; hr.all-development
- `hr.break_schedule_processing` — partial; hr.all-development
- `hr.docusign_nda_collection` — full; hr.all-development
- `hr.equipment_provisioning` — full; hr.all-development
- `hr.intern_cohort_onboarding` — partial; hr.all-development
- `hr.intern_program_coordination` — partial; hr.all-development
- `hr.zoom_orientation_sessions` — partial; hr.all-development
- `hr.airtable_learning_path_assignment` — partial; hr.all-development
- `hr.bamboohr_promotion_update` — partial; hr.all-development
- `hr.buddy_assignment` — zero; hr.all-development

### marketing

- `marketing.product_launch_channel_plan` — partial; marketing.campaign-launch-and-coordination
- `marketing.social_content_calendar` — partial; marketing.content-seo-and-editorial
- `marketing.newsletter_sponsor_invoicing` — partial; marketing.customer-lifecycle-and-data
- `marketing.google_ads_high_intent_list_from_hubspot` — partial; marketing.paid-advertising-and-attribution
- `marketing.brand_mention_analysis` — partial; marketing.research-and-monitoring
- `marketing.linkedin_speaker_outreach` — partial; marketing.social-and-community
- `marketing.editorial_calendar` — zero; marketing.unresolved-workflows
- `marketing.social_scheduling` — partial; marketing.social-and-community
- `marketing.ad_platform_audit` — full; marketing.paid-advertising-and-attribution
- `marketing.email_blast_suppression` — full; marketing.customer-lifecycle-and-data
- `marketing.content_repurpose` — partial; marketing.content-seo-and-editorial
- `marketing.lead_enrichment` — partial; marketing.paid-advertising-and-attribution
- `marketing.twitter_influencer_followup` — partial; marketing.social-and-community
- `marketing.conversion_tracking` — zero; marketing.paid-advertising-and-attribution
- `marketing.instagram_approved_asset_publish` — partial; marketing.social-and-community

### operations

- `operations.linkedin_abm_outreach` — partial; operations.customer-and-marketing-workflows
- `operations.fleet_vehicle_maintenance` — partial; operations.facilities-inventory-and-scheduling
- `operations.facility_incident_triage` — full; operations.incident-security-and-access
- `operations.docusign_contractor_offboard` — partial; operations.procurement-vendors-and-contracts
- `operations.zoom_training_setup` — partial; operations.safety-training-and-compliance
- `operations.trello_vendor_hold_email` — partial; operations.procurement-vendors-and-contracts
- `operations.asana_safety_walk_log` — partial; operations.safety-training-and-compliance
- `operations.hubspot_churn_prediction` — partial; operations.customer-and-marketing-workflows
- `operations.hubspot_lead_qualification` — partial; operations.customer-and-marketing-workflows
- `operations.canva_asset_management` — partial; operations.customer-and-marketing-workflows
- `operations.mailchimp_ecommerce_sync` — zero; operations.facilities-inventory-and-scheduling
- `operations.salesforce_escalated_customer` — partial; operations.customer-and-marketing-workflows
- `operations.calendar_airtable_gmail_maintenance_notice` — partial; operations.facilities-inventory-and-scheduling
- `operations.calendly_equipment_inspection` — partial; operations.safety-training-and-compliance
- `operations.chatgpt_feedback_analysis` — partial; operations.customer-and-marketing-workflows

### sales

- `sales.multi_hop_lookup` — partial; sales.conditional-crm-selection
- `sales.deal_escalation` — partial; sales.crm-record-mutations
- `sales.docusign_void_resend` — full; sales.document-contract-workflows
- `sales.slack_channel_for_new_account` — partial; sales.email-and-customer-followup
- `sales.event_to_opportunity_pipeline` — partial; sales.meeting-and-event-workflows
- `sales.chatgpt_proposal_customization` — partial; sales.model-assisted-pipelines
- `sales.full_sales_cycle_orchestrator` — partial; sales.pipeline-analytics
- `sales.sheets_multi_channel_campaign_router` — partial; sales.unresolved-workflows
- `sales.calendly_no_show_followup` — partial; sales.meeting-and-event-workflows
- `sales.create_contact_for_account` — partial; sales.crm-record-mutations
- `sales.docusign_deal_workspace` — partial; sales.document-contract-workflows
- `sales.linkedin_event_promotion` — partial; sales.meeting-and-event-workflows
- `sales.zoom_calendar_conflict` — full; sales.document-contract-workflows
- `sales.apply_project_label` — full; sales.unresolved-workflows
- `sales.call_transcription` — zero; sales.unresolved-workflows

### simple

- `simple.email_airtable_customer_welcome` — full; simple.airtable-records
- `simple.mailchimp_email_request` — full; simple.email-communication
- `simple.email_hs_create_contact` — full; simple.hubspot-crm
- `simple.zoom_calendar_sync` — full; simple.meeting-scheduling
- `simple.asana_sprint_section_task` — full; simple.project-issue-work
- `simple.email_sf_contact_account_update` — zero; simple.salesforce-crm
- `simple.slack_dm_meeting_reminder` — full; simple.slack-communication
- `simple.email_zendesk_ack_reply` — partial; simple.support-tickets
- `simple.feature_launch_slack` — full; simple.unresolved-workflows
- `simple.trello_urgent_support_card` — full; simple.project-issue-work
- `simple.email_sf_contact_assistant_update` — full; simple.salesforce-crm
- `simple.hs_create_deal_with_contact` — full; simple.hubspot-crm
- `simple.jira_accessibility_audit` — full; simple.project-issue-work
- `simple.zendesk_resolve_email` — full; simple.support-tickets
- `simple.gmail_invoice_email` — zero; simple.email-communication

### support

- `support.reamaze_cross_platform_dedup` — partial; support.cross-platform-migration-and-sync
- `support.gorgias_refund_processing` — partial; support.gorgias-workflows
- `support.helpcrunch_engagement_scoring` — partial; support.helpcrunch-workflows
- `support.helpscout_jira_bugs` — partial; support.helpscout-workflows
- `support.hiver_inbox_report` — partial; support.hiver-workflows
- `support.intercom_demo_scheduling` — partial; support.intercom-workflows
- `support.reamaze_knowledge_routing` — partial; support.reamaze-workflows
- `support.zendesk_gdpr_purge` — partial; support.zendesk-workflows
- `support.zoho_account_health` — partial; support.zoho-workflows
- `support.zendesk_hubspot_org_sync` — partial; support.cross-platform-migration-and-sync
- `support.gorgias_inventory_routing` — partial; support.gorgias-workflows
- `support.helpcrunch_zoho_desk_bridge` — partial; support.cross-platform-migration-and-sync
- `support.helpscout_hubspot_deal_alerts` — partial; support.helpscout-workflows
- `support.helpscout_reamaze_migration` — partial; support.cross-platform-migration-and-sync
- `support.intercom_auto_response_drafts` — zero; support.intercom-workflows

## Execution ownership and status

Latest checkpoint: ten first-batch public inputs are reviewed. Two bounded
components are cataloged and pass seven actual-replay/adversarial tests; 103
selected tasks remain unregistered. All 105 transport probes pass again with
1,727 executions. Neither component establishes full-task eligibility. Earlier
counts below describe the original transport-only checkpoint. See
[updated results](cross-category-api-probe-batch01-results.json) and
[semantic checkpoint](manifest-positive-and-batch01-checkpoint.md).

### Semantic acceptance after transport

For each selected task, retain a reviewed public-request/policy binding and a
complete goal/guard inventory. Plan expected obligations from the initial
population before checking outcomes. Explicitly reviewed empty guards are
allowed; missing guard definitions are not equivalent to an empty inventory.
Keep unknown policy, incomplete populations and unsupported operations visible.

Actual hash-bound episodes must exercise `Task.score`, native serialization,
reload and rescoring with factual findings and exact execution recipients.
Each authored goal also needs a positive and failed witness; each guard needs a
forbidden-effect witness. Shared adversarial tests cover already-satisfied
state without invented action credit, missing acknowledgements, repaired harm,
duplicate/ambiguous identities, source tampering and idempotent consumption.
Manufactured witnesses must be labeled separately from observed Luna behavior.
Tokenless sources qualify semantic execution attribution only, not token masks.

A bounded check can pass while whole-task qualification stays open. The fifteen
per category gate requires meaningful reviewed findings/credit coverage; an
unregistered manifest or uniformly unavailable output does not satisfy it.
Whole-task eligibility additionally requires every required goal and guard to
be covered under the accepted reward revision.

### Shared capabilities exposed by public requests

These are inspected public requests in the candidate's
`src/automationbench/domains/<domain>/tasks.py`. They illustrate mechanism gaps;
they are not task-specific evaluator instructions or completed declarations.

| Category example | Public requirement | Shared capability still needed |
| --- | --- | --- |
| `finance.prepaid_amortization` | Calculate policy-based per-item recognition and email the total. | Bounded monetary/date expressions, aggregate checks, schedule updates and required notification evidence. |
| `hr.job_posting_distribution` | Post approved requisitions, announce on Slack and draft a careers email. | Approval population, record creation, Slack effects and draft evidence distinct from sending. |
| `marketing.conversion_tracking` | Upload conversions under the standard tracking policy. | Bind actual public policy/deal populations and preserve conversion-upload batch occurrences; do not substitute an unrelated empty query. |
| `operations.facility_incident_triage` | Handle only the single most critical unresolved incident and contact Monday's on-call person. | Deterministic rank/tie handling, dated lookup, Jira/SMS/Slack effects and forbidden-extra-action checks. |
| `sales.multi_hop_lookup` | Route a won deal using account tier, FX rates, escalation state and latest routing policy. | Cross-source joins, bounded currency conversion, record transition and conditional required sends. |
| `simple.email_airtable_customer_welcome` | Read onboarding data, create a customer record and send a welcome email. | Public source-field extraction, Airtable creation and positive send obligations. Existing Gmail evidence alone is partial coverage. |
| `support.intercom_auto_response_drafts` | Match open conversations to templates, draft responses and track the batch reference. | Conversation population, left joins, exact template fields, draft/note and tracking effects. |

Implement population-based positive effect obligations first, then qualify a few
selected tasks sharing that mechanism before expanding adapters and deterministic
operators. No Python branch may select policy by task name. The installed catalog
remains ten tasks; this ledger does not silently enroll the selected 105.

Selection, source hash verification and generic native transport probes are
complete. The final test batch passes 113 cases in 53.64 seconds: 105 recorded
tasks plus eight guard-boundary regressions. The selected traces retain 1,727
exact native execution occurrences. Source and episode roundtrips, scalar
replay, immutable identities and original bytes pass. See
[structured results](cross-category-api-probe-results.json).

All 105 selected traces are tokenless, so this evidence cannot qualify original
token masks. At that original checkpoint none of these workflows had an installed reward manifest;
the existing ten Opportunity cases have separate qualification. Thus only the
raw transport portion of the breadth gate has passed. Adding reviewed task
declarations, shared effect/check capabilities, native findings and credit tests
remains necessary. No new task evaluator or qualified reward policy was created
by this probe. Root owns subsequent schema/adapter decisions and plan updates.
The source index and this selection remain development evidence; reserved tasks
remain outside fitting.

Files used: `luna-development-review-coverage.json`, `luna-development-policy-contracts.json`, their referenced native episode files, and the installed AutomationBench native operation captures. The selection JSON binds their exact hashes.
