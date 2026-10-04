# Manifest authoring: batch 03 public-input review

Reviewed 2026-10-04: **10 tasks reviewed, zero executable component proposals, zero accepted manifests, zero whole tasks qualified.** No catalog, implementation or recorded episode was changed.

The [JSON companion](manifest-authoring-batch-03-review.json) contains ten concrete deferred blueprints: populations, policy inputs, decisions, outputs, ambiguities and adversarial cases. They are authoring specifications, not executable ContractSpec objects. Public input hashes bind every reviewed task.

## What this batch adds

The new Sheets row-write and derived decimal/date capabilities are available in the schema. This batch still needs service evidence, bounded composition or policy resolution before those primitives can express a useful goal. None of the tasks requests a supported Opportunity update or Asana action. Salesforce Account is a separate, currently unsupported capability. Arbitrary recipient-only emails or unrequested worksheet writes would inflate coverage without establishing the requested work.

| Task | Public interpretation and next boundary |
| --- | --- |
| support.helpcrunch_engagement_scoring | Apply source weights and tiers to captured activity. Some customers have only future-dated events relative to February 7; fixture-looking IDs are not an exclusion rule. Need nested sums, HelpCrunch tag/event evidence, significant-change semantics and grouped reports. |
| finance.xero_vendor_onboard | Existing CloudNine has a different billing email, so approval to onboard is not authority to modify it without a change form. Other visible requests are incomplete or flagged. Need Xero identity/baseline evidence and Slack thread confirmation. System silent exclusions override procedural rejection narration. |
| hr.candidate_research_dossier | Apply withdrawal and internal-transfer updates before dossiers. Internal research restriction and coordinator-only background checks are separate guards. Finalists are not identified; a draft brief is not automatically a send requirement. Need source facts, panel expansion and access-event evidence. |
| marketing.google_ads_high_intent_list_from_hubspot | Internal policy requires score >=80, no opt-out, no demo request and scoped exclusions. External 70+ recommendation is not an adopted change. Missing properties cannot silently become false; need HubSpot populations and Google Ads membership evidence. |
| operations.docusign_contractor_offboard | Filter eligibility before selecting one contractor by date then clearance. Alex Rivera is the derived public example, not a production ID list. Need unique ordered selection and linked DocuSign/Trello/Notion/Slack effects. |
| sales.slack_channel_for_new_account | Derive the channel name and account team from the exact Account and guidelines. Similar account names and escalation-only contacts must not leak into the team. Need Slack channel/membership effects and a Salesforce Account reference to the generated channel. |
| simple.zoom_calendar_sync | Topic, date, local start, duration and host are explicit; timezone is not. Need two qualified creation effects and field consistency. Missing initial Zoom service does not mean a qualified empty service. |
| support.helpscout_jira_bugs | Exact domains/tags and scoped overrides matter. “500 error” matches two priorities; keyword scope and precedence need review. Need HelpScout/Jira effects and generated issue-key joins. |
| finance.monthend_journal_entries | Separate verbatim source amounts from policy-derived journal amounts. Direct no-rounding wording and SOP rounding need an explicit interpretation; half-dollar ties are unresolved. No journal destination is named. Existing arithmetic and append tools do not resolve these meanings. |
| hr.trello_recruiting_event_coordination | Venue is already publicly confirmed; do not reward a redundant booking. Resolve the swag request against the supplied freeze and handle the recruiter on leave. Card IDs in a sheet do not prove cards exist; available tools update/find/comment rather than create. |

## Smallest useful next capabilities

1. **Slack channel and membership evidence, then Account references.** The account-channel task provides clear naming/team rules and a narrow generated-ID join. The CRM field representation still needs a source-backed decision.
2. **Zoom and Calendar creation.** A small paired-effect workflow with a singleton public-request population. Resolve timezone before admitting an exact schedule.
3. **Finite policy decisions.** Reuse decimal/date expressions; add closed population selection, unique ordered minima, scoped rule precedence and the necessary service populations. Ties and unknown candidates stay explicit.
4. **Source-grounded output and baseline evidence.** Grouped summaries, dossiers, retained state and access/purchase events cannot be replaced by exact teacher wording or recipient-only checks.

Each adapter remains a reusable service capability. Each task's identities, thresholds, rule choices and declared outputs remain data. A capability being expressible does not settle disputed policy.

## Validation and provenance

Pack: `/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/manifest-authoring-105/batch-03.json`  
Exact SHA-256: `689535bbc1dec2ab553ceae4f7ae60f1eb4e88e2f356a64c59e10b3b21f5ff15`

Exact SHA and task order match the batch index; all ten public-input digests match. Both selection hash schemes were independently checked: the index hashes exact file bytes and the pack hashes canonical JSON contents.

A live ContractSpec schema inspection confirms Sheets row-write registration and derived decimal/date values. Account source admission rejects with `record_object_capability_unimplemented`; the named service adapters above are not registered. There are **zero executable proposals to schema-validate**, and no native task scoring was run for this review.

Hidden assertions, hidden answers, task-specific reward predicates and recorded episode contents were not used. Episode references are copied only for later exact-SHA replay. Qualification must exercise changed inputs, ambiguous identity, missing ACKs, repair, repeated scoring and retained native evidence before any component is promoted.
