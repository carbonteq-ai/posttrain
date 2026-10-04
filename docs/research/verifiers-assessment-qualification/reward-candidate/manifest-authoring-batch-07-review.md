# Manifest authoring: batch 07 public-input review

Reviewed 2026-10-04: **10 tasks, one executable guard proposal, zero accepted manifests and zero whole tasks qualified.**

The [JSON companion](manifest-authoring-batch-07-review.json) contains the complete proposed ContractSpec and all ten public-policy inventories, deferred blueprints, ambiguities and counterexamples. Deferred blueprints are not executable manifests or counted coverage.

## Supported candidate

**support.zoho_account_health — no dashboard append for an excluded account.** The public request identifies `ws_new_accounts` as the exclusion list, and `ws_dashboard` declares an Account ID column. The proposal uses the validated exclusion population and a qualified Sheets append matched by that explicit business identifier. It needs neither a fixed list of excluded accounts nor an invented onboarding-age cutoff.

The unconditional guard predicate applies only to members of that declared exclusion population. Missing or ambiguous Account ID correlation stays unavailable. This append guard does not cover Salesforce flags, updates, alternate dashboard representations, score correctness or all excluded-account processing. Removing the row later does not erase the witnessed append. No positive reward is proposed.

One ContractSpec validates and roundtrips. **Seven public bindings match and seven independently altered bindings reject.** Native Task.score and recorded replay were not run. Shared native admission/retention work is being qualified separately; these schema checks do not establish native correctness.

## Task decisions

| Task | Public meaning and next shared requirement |
| --- | --- |
| sales.calendly_no_show_followup | The report says today, but the matching events are future, cancelled or five days earlier. Event resolution is genuinely uncertain. Priority uses any open opportunity >=50000, not total pipeline >=50000. Need event joins, exists/sum and Salesforce Task evidence. |
| simple.feature_launch_slack | Announce launch in product and create the exact Asana task in ws_prod; either order can be valid. Need a request singleton and Slack content facts. Missing initial Asana service cannot be silently treated as empty. |
| support.zoho_account_health | The bounded exclusion guard above is supported. Full scoring needs grouped tickets, a declared resolution-rate denominator and Salesforce flag evidence. No public score cap or thirty-day exclusion cutoff is supplied. |
| finance.qb_sales_receipt_batch | Completed transactions only; join supplied State tax rates, create receipts and report exact sums/count. Need QuickBooks receipt evidence and grouped content checks. Repeated customers must not merge distinct transactions. |
| hr.intern_cohort_onboarding | Standard process prohibits intern cards and direct HR email provisioning. Manager names are not written approval, and sending an IT request does not prove an account exists. Need approval/prerequisite semantics and truthful partial completion. |
| marketing.email_blast_suppression | Current compliance supersedes old unrestricted archiving; external vendors cannot override it. Deduplicate subscriber identity, preserve holds and exact addresses, and report actual archives only to ops. Need Mailchimp transitions and summary purpose/content evidence. |
| operations.canva_asset_management | Require approved AND quarter tags, with HOLD override. Link completed PNG export to actual Drive artifacts and destination; count designs, not files. Need export lifecycle, generated-ID joins and retained Drive evidence. |
| sales.create_contact_for_account | Resolve actionable new notifications, exact account, duplicate email, restrictions and seniority. Read state alone does not prove prior processing; dates are absent on the key messages. SOP skip reporting conflicts with system silence. Need Contact/Account evidence and reviewed message facts. |
| simple.trello_urgent_support_card | User explicitly requires listing lists first. Preserve that causal prerequisite, resolve actual list ID rather than action-wrapper ID, then correlate the created card and urgent label. Need request population and Trello composition. |
| support.zendesk_hubspot_org_sync | One-to-many domain matches need closest-name ID selection plus churn checks across ALL matches. Preserve holds, notes and tags. Sheet inventory includes IDs absent from Zendesk; do not fabricate organizations. Need registered service adapters and ambiguity-aware set joins. |

## Smallest next shared work

1. Use the small launch and Trello tasks to qualify a public-request population and bounded composition. Keep genuine procedural requirements, such as first listing lists, without imposing an arbitrary sequence on every workflow.
2. Add finite grouped counts/sums, existential predicates and one-to-many joins. Event/name ambiguity must survive into results rather than disappear through first-match selection.
3. Add the chosen workflow's shared service facts and generated-output links: export jobs/files, Contacts/Tasks or organization/company creation. Retained outcomes and occurrence evidence remain separate.
4. Add source-bound approval/hold facts and meaningful notification content. An arbitrary message to the right recipient is still insufficient useful-action evidence.

## Provenance

Pack: `/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/manifest-authoring-105/batch-07.json`  
Exact SHA-256: `6913ff37ab6ef7a4fd6297312a746a305330a350ada61bfc46c4e19efd2385a1`

Pack hash and ordered identities match the index. All ten public-input hashes pass. Both exact-byte and canonical-JSON selection digests were independently recomputed.

Only public prompts, initial worlds, tools and shared capability schemas supplied review input. Hidden answers/assertions, task-specific scorers and recorded episode contents were not consulted. Replay references are copied for later immutable qualification only. No source, catalog, ledger, plan or previous review was changed. This is authoring progress, not installed coverage, whole-task qualification or evidence of learning.
