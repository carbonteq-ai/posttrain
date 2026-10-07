# Manifest authoring: batch 05 public-input review

Reviewed 2026-10-04: **10 tasks, one executable component proposal, zero accepted manifests and zero whole tasks qualified.**

The [JSON companion](manifest-authoring-batch-05-review.json) contains the ContractSpec proposal, public authority and policy inventory for every task, deferred data blueprints, ambiguity decisions and adversarial acceptance cases. Deferred blueprints are not executable contracts or counted coverage.

## Supported candidate

**sales.full_sales_cycle_orchestrator — advance the named Opportunity to Proposal.** The Sales Manager's public post-demo playbook explicitly names this stage. The proposal binds the user prompt, manager identity, channel, playbook and target Opportunity identity, then uses the existing record check and verified-transition credit rule. It does not infer a generic sales stage from the request.

The initial public record uses `stage: Demo`, while the installed native schema declares `stage_name`. The manifest requires final `stage_name: Proposal` and binds name/account identity; it does not fabricate an observed normalized baseline. Native replay must establish the actual baseline before assigning transition credit. A final match alone cannot supply causal attribution.

One ContractSpec validates and roundtrips. All **13 public bindings** match, and **13 independently altered binding values** are rejected. No native Task.score or recorded rollout qualification was run. Contract drafting, the no-send prohibition, scheduling, prep, talking points, approval scope and Slack remain separate open requirements.

## Task decisions

| Task | Public meaning and next shared requirement |
| --- | --- |
| simple.email_sf_contact_account_update | David Kim's public email says Nova Horizon. Extend the reusable record adapter to Contact, preserving identity and account-name scope. Do not require updating email, title or account_id merely to increase coverage. |
| support.intercom_demo_scheduling | Join conversation/contact/company, apply exact tags and policy, then deduplicate by contact. Five employees is eligible; demo-requested is not demo-request. Always Schedule precedence and decline replies versus the system's silent exclusions remain explicit policy ambiguities. Need Intercom and Calendar evidence. |
| finance.escrow_tracking | All conditions must be Met; Not Met must not pass a substring check. Disputes route to legal. Need reviewed structured condition facts and purpose-qualified notices. Available tools cannot independently prove an external disbursement; a Released cell is a record fact. |
| hr.break_schedule_processing | Compose request age, reconfirmation, shift length, meal minimum, fifth-hour deadline and 2–4 PM blackout. Accept valid alternative break times. Shift timezone and relative manager-message date need interpretation; the Calendar event independently establishes the same-day blackout. |
| marketing.editorial_calendar | Current director policy supersedes older manager guidance, and agency suggestions do not waive it. Assignment evidence, hold start and recent-theme horizon are underspecified. A competitor's 48-hour hold is not a permanent ban. The tools support a plan email, not a sheet-write requirement. |
| operations.asana_safety_walk_log | Filter Annex/pending holds before selecting the next Main/Q1 inspection. The current rows imply Warehouse on March 2, but that derived result must not become a fixture outcome list. Need bounded minimum, date offset, tag evidence and generated-ID/retained composition. |
| sales.full_sales_cycle_orchestrator | Proposal stage is executable as the bounded component above. Buyer and technical lead are different contacts; DocuSign must remain draft. Preserve the channel's second-level-approval note as unresolved scope for the broader workflow. |
| simple.slack_dm_meeting_reminder | A public-request singleton and Slack DM content facts can cover the small task directly. Require Sarah, tomorrow at 2 PM, Globex and both materials with flexible wording. Recipient-only contact is not reminder completion; no absolute date is supplied. |
| support.reamaze_knowledge_routing | Billing API, shipping invoice and API SSO each match multiple mapped domains. Do not choose the first sheet row without a precedence rule. Resolved-conversation scope and general fallback are also unclear. The unused Batch_Reference field does not create a mandatory output requirement. |
| finance.contract_billing | Reuse exact arithmetic for the stated formula and Active eligibility. Need qualified QuickBooks invoice/customer/period evidence, then purpose-bound notice content. Emailing an excluded customer is not itself proof of forbidden billing. No hard-coded expected invoice list. |

## Smallest next shared work

1. Complete the two small tasks through a Contact record capability and the already proposed public-request population plus Slack DM evidence/content. These are clearer opportunities for complete useful coverage than adding recipient-only checks to larger workflows.
2. Add bounded selection and ambiguity-aware joins for finite populations, with explicitly reviewed text/condition facts. Do not convert unresolved policy into a guessed enum or fixture-specific predicate.
3. Add the selected workflow's service evidence and generated-ID/retained composition. The newly registered retained Sheets check is not treated as natively qualified by this review.
4. Preserve flexible message meaning, unknown policy, alternate valid schedules, and the distinction between records, occurrences, retained outcomes and action credit.

## Provenance and validation

Pack: `/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/manifest-authoring-105/batch-05.json`
Exact SHA-256: `8f93b47f1584ee58c553a50db4c800966f4e74ebb28d3600df2695d86de590e1`

Exact pack hash and ordered task identities match the frozen index. All ten public-input hashes pass. The exact-byte selection hash and canonical-JSON selection hash were independently recomputed and match their respective records.

Public prompts, initial state and available tools supplied task authority. Hidden assertions, hidden answers, task-specific scorer code and recorded episode contents were not consulted. Episode references are copied only for later exact-SHA qualification. No source, catalog, ledger, plan or previous review artifact was edited. This review establishes authoring progress, not installed reward coverage, whole-task success or learning.
