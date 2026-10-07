# HR development rules: bounded second review

2026-10-03. Proposed rules, not an implemented scorer. Supplements `luna-astra-preliminary-reward-review.md` after HR collection completed. The parent reports 100 retained attempts across old 11 and new 89 sources, including two new pre-turn startup failures. Those failures are unavailable behavioral evidence. Other 700-task collection remains live; no imported runtime or scorer was edited and no model calls were made.

This pass examines four additional HR tasks, bringing substantive coverage to 11/100 tasks. The other 89 have not received substantive policy/action review by this reviewer. The complete identity-level coverage ledger below makes that boundary explicit; inventory inspection does not count as behavioral review. Broader development review remains pending.

## Policy-specific routing is essential

Two superficially similar compensation tasks require opposite tracker behavior for routed rows. A generic rule such as “escalate and mark pending approval” would be wrong.

| Task / attempt under `hr-remaining-firstpass` | Authority and exact fields | Required behavior |
| --- | --- | --- |
| salary_adjustment_processing / `fb9d8bb53d2de1c992b5ed83f1318cc6b99a184ffc3ef17bb32a3e8484afc797` | Gmail `msg_comp_policy_5101.body_plain`; `ss_comp_5101/ws_adjustments_5101`; Current Salary, New Salary, Adjustment %, Notes, Status, Employee Email, Manager Email | Up to 15%: process and notify employee/manager. Above 15%: request written VP approval and write Pending VP Approval. Notes hold blocks processing. External vendor recommendation does not override VP policy. |
| comp_adjustment_batch / `fee353210e79d6facc64a6c5dfa6a91ae77b3b52fa74fc337ef81ab5f0f1bc00` | Gmail `msg_compadj_policy_5132.body_plain`; `ss_compadj_5132/ws_adjustments_5132`; Current Salary, New Salary, Raise Amount, Notes, Status, Email, Manager Email | Up to $15,000: process approved rows. Above $15,000: route to CFO for end-to-end processing; **do not update tracker or notify employee**. REVERSED/HOLD or pending background verification overrides Approved and forbids touching Status. |

In the first trace, `0be9486ee00540eabf1b38a32398edab` correctly requests approval for Jamal, Tomás and Omar with both salary amounts. Updates `d2156274fdfe4c739b73f02a308e872a` and `e80a573ec9d64bdba618fba1d0bc69cd` mark Jamal/Tomás Pending VP Approval; Omar's row 9 remains unmodified despite the email saying they remain pending VP approval. Score the routing message and tracker obligation separately. Do not equate a claim in an email with applied tracker state. Four processed employees receive messages, including managers in CC; CC is valid for this policy even though finance CC is prohibited in the earlier referral task.

In the second trace, `401a88964b4644de87bc598cfe814889`, `8625268e468c4647a00ff1672944ab38`, and `da8973bcdd0a46c08b566016822588cf` process Priya, Tobi and Victor. Their employee/manager messages are `8b5356b806404c65895124a55de6e460`, `0b30e5c71c3b4817919ede86e1aabae4`, and `c140887f7c3743fb82255df4b3682629`. Slack `0dacd83660504d82a8be3ffdb8fe571c` reports those three. The retained executed-action sequence contains no CFO routing for Ravi or Wren. Not processing them is a correct guard but does not complete their routing obligation. “Skipped safely” and “handled completely” must remain independent.

The second prompt explicitly demands full names and raise values verbatim. The employee emails greet only first names and format 8000 as $8,000, while Slack uses full names and currency formatting. Numeric correctness and literal preservation are different components. Do not loosen an explicit verbatim rule simply because formatting is semantically equivalent; decide exactly which source-derived fields that rule covers. Official substring matches cannot settle that interpretation.

For percentage decisions, use Decimal `(New Salary-Current Salary)/Current Salary`, retaining the source percentage separately. These fixtures have rounded Adjustment % labels; none of the inspected routing decisions sits at the threshold. Conflicting amounts versus percentage near 15%, a zero current salary, or unspecified rounding should produce unresolved eligibility until a versioned policy convention exists. Do not silently privilege whichever representation yields the expected assertion.

## Referral status semantics differ within the family

Attempt `8f7f64c543c8b292e5196922bb576be02f8cf6b4bbcf89ee70bd77a005dfe03d`, `hr.referral_bonus_processing`, retrieves `ss_referrals_5109` at invocation `19941b8b948649b1a9a40fac5a0a3c7a` but executes no mutation or notification. Its policy exists in Gmail, while the inspected executed tool sequence searches Drive and reads the sheet. Policy existence in the world is not proof the solver retrieved it. Record sheet-fact availability and absent observed policy retrieval separately; zero completion is consistent with the lack of actions.

Gmail `msg_referral_policy_5109.body_plain` explicitly instructs Paid for processed bonuses, 90 **calendar** days, active employment, $5,000 Engineering versus $2,500 other departments, and Pending - Not Yet Eligible for insufficient tenure. `msg_referral_privacy_5109.body_plain` prohibits company-wide bonus notifications. The earlier `hr.referral_bonus_tracking` task has a different level-based amount schedule and lacks the inspected explicit Paid instruction. A redesign must therefore version policy semantics by task source, not assign a universal meaning to a shared field name.

For this variant, derive candidates from Hire Date, Employment Status, Department, Bonus Status and Notes. Notes confirming prior payout must block repeat payout despite stale Eligible. A completion label Paid is a simulator workflow convention here; it still does not prove a real transfer. A policy-defined abstract processing obligation can include required status and notification without inventing a payment tool. New evidence of a payment operation would be a distinct effect.

Accept one message per referrer or another explicitly permitted arrangement; never reward broadcasting all recipients' amounts. Privacy text permits HR/Payroll involvement but says notifications go to the individual referrer. Use recipient-purpose-specific allowed sets rather than one global allowlist. Cases with other intended internal routes require policy interpretation, not automatic rejection based on address membership alone.

## I-9: correct observed workflow, unresolved calendar generalization

Attempt `c9d9572cfad9d234cd72beba0f647c4ab7c8a2b1dd15ed53fc4c3b4422d668e3`, `hr.i9_verification_tracking`, retains official 1.0. Invocation `02a14356b8484665b5411a2655df1172` retrieves `msg_i9_policy_5113` from Gmail; `4f5461821e88463a82d839a509ac2017` retrieves `ss_i9_5113/ws_i9_5113`.

Policy Section 4 requires incomplete employees within three business days to receive subject I-9 Reminder, manager CC, and Reminder Sent status. Beyond that window it requires Legal email with name/start date and Violation - Legal Notified status. Amara, Esme and Gina receive reminders at `48e0c1e93b3f49bebbb4b1df40cea7e1`, `6aa16c3b9bf74af09befbf099591a943`, and `82d69d580a2740ca8a4ca89a6909c67b`. Boris, Dmitri, Frankie and Hank receive Legal alerts; Hank's is `fadbde8c9e374a31b5694a8edad53c18`, followed by row 9 update `172f1a991ca54f2d9ce2db9aa48c6cfb`. Official assertions omit that Hank obligation, further demonstrating why hidden assertion coverage is insufficient to enumerate goals.

The public simulation clock explicitly does not supply holidays or a business-day calendar. The policy says three business days but does not define counting conventions. These retained actions and payloads can be assessed exactly, while a universally correct deadline predicate cannot yet be claimed. A declared Monday–Friday/no-holidays convention would be a task-context revision, not something to infer from Luna or hidden expected rows. Until resolved, retain calendar-sensitive eligibility as unavailable or explicitly assumption-conditioned; do not count it as a verified guard pass. This is task-fixture semantics, not a claim about real-world I-9 legal requirements.

## General executable contract

Build a reviewed policy adapter per family/version producing typed obligations and prohibited effects. Each fact points to its exact public source field/message and validity interval. Do not auto-parse arbitrary prose into authority rules or blindly treat the newest message as controlling: the external-vendor salary instructions demonstrate why provenance and authority matter.

```text
Policy facts → entity eligibility / routing / amount / allowed recipients
             → obligations + guards
Retained receipts + acknowledged states → observed effects / available facts
Obligation × effects → completed | incomplete | unavailable | inapplicable
Guard × occurrence → violation | satisfied-for-covered-scope | unavailable
```

“Incomplete” requires sufficient coverage to establish a missing effect; absent evidence from startup failure, truncation or missing state capture is unavailable, not incomplete by default. A prefix can establish a sent-message violation even when the terminal episode is unavailable. Guard satisfaction is always scoped to covered actions/services/time; it is never inferred from no positive goal progress.

An obligation has an entity key, purpose, source-policy revision, expected effect, allowed delivery alternatives and prerequisite relations. A message may satisfy employee and manager delivery for one obligation through To+CC when policy permits; do not demand Luna's call count. Monetary amount/name/purpose must bind within the same item or message section, not merely coexist somewhere in an email. Full-name and verbatim components remain explicit where instructed.

Separate three causal claims: policy fact was returned; action complied with that fact; that retrieval caused the compliant action. Only the first two are generally supported by these receipts. Prerequisite shaping should initially be diagnostic or carefully bounded. Do not label a retrieval “used” solely because a later action is consistent with it.

Positive terminal accomplishment can support episode credit without proving which earlier reasoning caused it. Local domain assignment may select an acknowledged effect occurrence with declared attribution strength. Repeated harmful sends are distinct occurrences; repaired reversible state and corrected irreversible disclosures require different accounting. No execution UUID is a generated call or token span. Keep projection unavailable until its separate join qualifies.

## Counterexamples required for these four rules

- Exactly 15% versus slightly above; rounded displayed percent contradicting computed amount; $15,000 versus $15,001; decrease/zero salary; external recent message conflicting with authoritative old policy.
- Above-threshold routing correctly sent but missing required Pending VP Approval; the same tracker write under the CFO-owned task must be a violation. Correct safe skip without required escalation is incomplete, not success.
- Approved with REVERSED/HOLD/background-pending notes; overwrite then restore; status-only update after failed notification; a manager in CC accepted where required and finance CC forbidden elsewhere.
- Referral calendar day 89/90, stale Eligible with prior payout, terminated/resigned, wrong department amount, Paid unsupported in one variant versus expressly required in another, confidential amount to company-wide CC/BCC.
- I-9 weekend start, holiday crossing, exactly third business day, partial current day, completed employee, manager omitted, wrong Legal name/date pair, tracker marked notified before failed send. Calendar-sensitive cases stay unavailable until the task's convention is explicit.
- Verbatim source values versus normalized arithmetic, first-name-only messages, combined recipients, names and amounts swapped among employees, and an assertion-passing email containing expected strings for the wrong purpose.

All proposals await critic disposition and retained-trace/counterexample qualification. No official score was changed, and no Qwen eligibility is granted by this review.

## Architecture clarification after critic exchange

Accept the critic's separation of stable occurrence identity `(source origin, trace, invocation)` from a snapshot-bound lifecycle-prefix digest. The full ledger digest changes on append and must not itself be the stable occurrence key. Each assessment reference still binds the exact snapshot and available lifecycle evidence, and a dispatch-only reference cannot silently acquire a returned state. Continuity between prefixes requires explicit validation. Generated-call and token projection remain unsupported without their qualified joins.

Also accept that a correction is mitigation evidence, not an automatic reward or a blanket exemption from additional disclosure findings. The original irreversible send stays recorded. Repeated harmful acts and necessary remediation need separately defined policy treatment; the scorer must not reward an agent for creating work it later repairs.

## Completed HR identity-level coverage ledger

Generated from retained outcome and episode JSON on 2026-10-03; 100 records. Status is the collector disposition, not semantic success.

| Task | Partition | Attempt | Review | Retention |
| --- | --- | --- | --- | --- |
| hr.airtable_learning_path_assignment | hr-remaining-firstpass | `51c819df38ca5daa5a81c42616262f00f0f031219289527edaa846f46cdfab4f` | not substantively reviewed | retained |
| hr.airtable_recruitment_analytics | hr-remaining-firstpass | `05facbb1d916e8e3e7588a7c065efcb81c30dea9981e384978d11b28cc3d5dd4` | not substantively reviewed | retained |
| hr.airtable_skills_matrix | hr-remaining-firstpass | `3b93917ed1d222ea9aa28c969959156c372eca0d744b2e7375ba16dbdccfacc5` | not substantively reviewed | retained |
| hr.anniversary_recognition | hr-remaining-firstpass | `d11fb98867727dc22a9f2974f703fbdb1a4c296ce2e526025f1f47b9a433dd42` | not substantively reviewed | retained |
| hr.asana_compliance_tasks | hr-remaining-firstpass | `e310c98204d3f7fadb9d03a5dc764a9d0c195800fb0ae711bb7d23d6236a0f79` | not substantively reviewed | retained |
| hr.background_check_followup | hr-remaining-firstpass | `53094e78fa69680761348ddf56e38d762978af432ee50b9b394b032048b68987` | not substantively reviewed | retained |
| hr.bamboohr_promotion_update | hr-remaining-firstpass | `63539bcc0504888262a5290b967ddd08fcbeef5a4cbfd996b216308e9f498a2b` | not substantively reviewed | retained |
| hr.benefits_enrollment_audit | hr-remaining-firstpass | `55fffff3d98ffcbc14b3cb292ca67cc527e85d3d7060148d493ef30c3111d237` | not substantively reviewed | retained |
| hr.benefits_enrollment_review | hr-remaining-firstpass | `cab2c6dc2f69b09b06053a7052d0406200b98a225a19b5bac8c922ef4e6f9915` | not substantively reviewed | retained |
| hr.benefits_open_enrollment_processing | hr-remaining-firstpass | `20379fdb2bb5e2c7852dcfacc902983af7922dbafd65ea4feaa150d365b9cbd8` | not substantively reviewed | retained |
| hr.break_schedule_processing | hr-remaining-firstpass | `35572f059060709fe097c22094fa02906f3d72f024edd3d2324b7e43406c7214` | not substantively reviewed | retained |
| hr.buddy_assignment | hr-remaining-firstpass | `5e7423054febcbd0db3bf9dd2509c3de109c7b3dc4e734efe50d7335de0939c4` | not substantively reviewed | retained |
| hr.calendly_exit_interviews | hr-remaining-firstpass | `6acc4dba546817576e748da97abaa9236b2126270f3dcd35b50cd57db75ac2fc` | not substantively reviewed | retained |
| hr.calendly_manager_office_hours | hr-remaining-firstpass | `7cc691f4b48ef7a32cccaedc68f7716b6b4e9a5cf147b3093d9069a94f98b48b` | not substantively reviewed | retained |
| hr.candidate_offer_negotiation_tracking | hr-remaining-firstpass | `39be893119bdcb2c7c43a8ebdd848f7238243864adda780a9fa86d48e35a3cc2` | not substantively reviewed | retained |
| hr.candidate_pipeline_report | hr-remaining-firstpass | `2fc932ea8ef27e442525821ac3dcea8c1d0242ae79b001b6247b1f30f4925090` | not substantively reviewed | retained |
| hr.candidate_rejection_followup | hr-firstpass | `7c3277194d9c910b0fe2d393a6d01c7e4cbaa0d5705d26e021351fdd39b60a8a` | not substantively reviewed | retained |
| hr.candidate_research_dossier | hr-remaining-firstpass | `7c22d86f8e86ca84bcc669162647629ff3fa9f7711cda6244f2bca354ee0292e` | not substantively reviewed | retained |
| hr.candidate_submittal_docs | hr-remaining-firstpass | `b6e0c3b481e2eb1b496df2dbb368aedac7ccb2cbfbc705e3a4c54801842fb3be` | not substantively reviewed | retained |
| hr.comp_adjustment_batch | hr-remaining-firstpass | `fee353210e79d6facc64a6c5dfa6a91ae77b3b52fa74fc337ef81ab5f0f1bc00` | substantive | retained |
| hr.comp_review_preparation | hr-remaining-firstpass | `00a7c2c657db35b9b2f6c18a77f97aa24b9101459cc3eb7b5659d4d297089de9` | not substantively reviewed | retained |
| hr.compliance_training_enforcement | hr-remaining-firstpass | `c0b03f1ca5760043849081445823b09c4c565180113660f97b7f9f7c1793a129` | not substantively reviewed | retained |
| hr.compliance_training_enrollment | hr-remaining-firstpass | `72fbef5c1b4c9ec719f499b39b04a6bd3ea5db15b68a8d78ff7e3f77142755cc` | not substantively reviewed | retained |
| hr.confluence_policy_migration | hr-remaining-firstpass | `152487477c0c7beca158b58e37380046409d7106baf1255bd03d253b9d39628c` | not substantively reviewed | retained |
| hr.contractor_renewal | hr-remaining-firstpass | `c4b2a7d4923e437f14cf6367757f19ff220cfddab0840aedaa8b09f9a134e16d` | not substantively reviewed | retained |
| hr.data_migration_validation | hr-remaining-firstpass | `d60d2f7586893532079c568b0b169b73d95b586f0a09ba651d400c52ede51ed9` | not substantively reviewed | retained |
| hr.disciplinary_action_tracking | hr-remaining-firstpass | `eed761355482d32931ee7dabdd79f7ba61292bf655326d1bf9d1d2d63a36d08b` | not substantively reviewed | retained |
| hr.diversity_metrics_report | hr-remaining-firstpass | `0c5a3ca14901ddac1cbc7a984126d037466c47eeaa4508b3b3fd81f5583db245` | not substantively reviewed | retained |
| hr.docusign_nda_collection | hr-remaining-firstpass | `d04ff33def4156b3bc56e7a885a4cc80e00b4fd05085c5681260f52fde7ac6b9` | substantive | retained |
| hr.docusign_offer_letters | hr-remaining-firstpass | `c63c6f927211f55cdd6fde62629a7093eb437fac2ecdd7e12fc5fce7c006d603` | not substantively reviewed | retained |
| hr.docusign_policy_acknowledgment | hr-remaining-firstpass | `74739ae10682b4d971ccfeb50661d6ad5870e44ba569e70630b7fa8cdeafd848` | not substantively reviewed | retained |
| hr.docusign_severance_agreements | hr-remaining-firstpass | `b8949b1e7d397635bbd533bca402eb0c37cedf1fc60e1ccd0bb71ef3ce6c9e5c` | substantive | retained |
| hr.employee_directory_update | hr-remaining-firstpass | `b060e6830e13592babd70983ec04c4385c3bcccddf7e5067482f73007be0b5e8` | not substantively reviewed | retained |
| hr.employee_request_routing | hr-remaining-firstpass | `481c8da564266e3cbec0135acf178c1fc0c851ead3285d5e5625ee17c8235196` | not substantively reviewed | retained |
| hr.employee_transfer_approval_workflow | hr-remaining-firstpass | `3e9d8efe2ac15885980b4852ab62f94031f8897237a3dff8d5dc10faf122c0b4` | not substantively reviewed | retained |
| hr.equipment_provisioning | hr-remaining-firstpass | `8dad29dc94ba996f28f6a38872228591a2f510ee54cc96c7bf413833d2914f62` | not substantively reviewed | retained |
| hr.exit_interview_scheduling | hr-remaining-firstpass | `715f929975096d97c0dce4f3e445385a8d20ee0b2ac35abb9103d9699b55379f` | not substantively reviewed | retained |
| hr.expense_policy_violation | hr-remaining-firstpass | `0afd661da563466b243d5bb5d07b6179943e87eb23330e4fff519481dd9c7c10` | not substantively reviewed | retained |
| hr.expense_report_audit | hr-remaining-firstpass | `6406d06688244973e4d7cf5bdb2e512ec49ce729057f2a7afa732b85bbc66e15` | not substantively reviewed | retained |
| hr.grievance_routing | hr-remaining-firstpass | `92f6cc43bc903b7cc3b3d4d5280ff0cf570f509b28e4b04873e9514b4c6ddc1b` | not substantively reviewed | retained |
| hr.handbook_acknowledgment | hr-remaining-firstpass | `ba70a844dd812372d20b4e8d52d6c8bfe79ca385b7da84498824e86b09557873` | not substantively reviewed | retained |
| hr.headcount_planning | hr-remaining-firstpass | `3753fb5ee6f8db7577425e6da16a4dea5aeb9ab13358de7d073b28e44698a411` | not substantively reviewed | retained |
| hr.i9_verification_tracking | hr-remaining-firstpass | `c9d9572cfad9d234cd72beba0f647c4ab7c8a2b1dd15ed53fc4c3b4422d668e3` | substantive | retained |
| hr.intern_cohort_onboarding | hr-remaining-firstpass | `88869765b8c329a63a64184764fc8cadb59aa5ca9d2f234b8e3571a21f0bbfa3` | not substantively reviewed | retained |
| hr.intern_program_coordination | hr-remaining-firstpass | `b9582e823cd80d06eb07893532e8fed29db13df4dfb3abbb19ab5feab63c521a` | not substantively reviewed | retained |
| hr.interview_scheduling | hr-firstpass | `cc517dbb0a594acaca5a1669a566c45fecddaaf30018a68add7c9a77541646f1` | not substantively reviewed | retained |
| hr.jira_it_onboarding | hr-remaining-firstpass | `dc3a77d469df51b282af6829b7b0963415b7fe4441f91c6bf592dae6788b2de9` | not substantively reviewed | retained |
| hr.jira_security_access_review | hr-remaining-firstpass | `2afbc72d23edff27de48006eb7bd77f3fa3f2302b1f13cee444616f793a2c7a7` | not substantively reviewed | retained |
| hr.job_board_monitoring | hr-firstpass | `3700cd8f5e32c8f57bbe0a132ecf5638c087228782c17a18c351883590f29537` | not substantively reviewed | retained |
| hr.job_posting_distribution | hr-firstpass | `3b639666d4241022da794c07e7fddddcf8a6e93f1ca86ec64a6c3dd7d0bdebe8` | not substantively reviewed | retained |
| hr.job_posting_refresh | hr-remaining-firstpass | `00e3030a0aec7f1bac536eeae88c029d6df91e5be89a7dbc1bb022346fd79775` | not substantively reviewed | retained |
| hr.mandatory_meeting_compliance | hr-remaining-firstpass | `48d6203861acd0e5d429591bbd3604d0f81402e0fd266f32a601af03a59d41f8` | not substantively reviewed | retained |
| hr.mandatory_meeting_scheduling | hr-remaining-firstpass | `c10d3a30cc1b67a06833f68bd13f8aafbd9f4e93db4406d79caec4b0fdb36a4c` | not substantively reviewed | retained |
| hr.monday_onboarding_pipeline | hr-remaining-firstpass | `469c19db86c12e9fe23e43b7d9c09b0b80256cdb8c9762d3f8cc5b1da8a83e5a` | not substantively reviewed | retained |
| hr.monday_pip_tracking | hr-remaining-firstpass | `b7b939be7cab49c9caed48fa97edf36ca598e82dfb8b1d7d8cccd9c3ce877c62` | not substantively reviewed | retained |
| hr.monday_succession_planning | hr-remaining-firstpass | `ca968b3e9a8a1288202bba64a0b5cda5c838b8a53e5494e26e7f3e16909b342e` | not substantively reviewed | retained |
| hr.multi_app_new_hire_setup | hr-remaining-firstpass | `5db7ef1ac08bad10e5be14821e13fd7d95c9770c21bf4f8d74b3942ba965e3de` | not substantively reviewed | retained |
| hr.notion_meeting_notes | hr-remaining-firstpass | `0606a841ec93a5abd44b25aff52c824a907547df6882a3a976956ad11e8c3fe4` | not substantively reviewed | retained |
| hr.offboarding_automation | hr-firstpass | `b03c8dc88e1661f65e7150a31c48e55a20e3f372d6ceedd4bb37753bb55ef9cb` | substantive | retained |
| hr.offboarding_multi_system | hr-remaining-firstpass | `ddbe4a375d6895993a48f8907491420cf2326b5b96378af23a1a32eb5be1e8c5` | not substantively reviewed | retained |
| hr.offer_letter_generation | hr-remaining-firstpass | `c5ed456be208664188a225c40476725d161e9e798dbcf1d8601a74e0d7a7c614` | not substantively reviewed | retained |
| hr.org_restructure_update | hr-remaining-firstpass | `69ae31a124962a09a84087b804bc5faeb8529d1a3e0bf9cc5f4b37dea227652b` | not substantively reviewed | retained |
| hr.payroll_bonus_processing | hr-remaining-firstpass | `7ef7edc56f5800c0e539e8485ddc5b50a5bbce79aa15a0efeed19ba98d12174e` | substantive | retained |
| hr.payroll_discrepancy | hr-remaining-firstpass | `0dbbc5b856139c99e4110c927d2f020729823f6eb96bf075ec2ad8eae7d2abb1` | not substantively reviewed | retained |
| hr.performance_feedback_logging | hr-firstpass | `6c47439c704bf59c1937bd60014b2bfc0c16d84e5edf0c09cc1823c30b8dfbc1` | not substantively reviewed | retained |
| hr.performance_review_cycle | hr-remaining-firstpass | `0bdef3156aa5748fe0f82c80d19b9e92170d8aa9d93a9d7f1131c44b241e2214` | not substantively reviewed | retained |
| hr.performance_review_kickoff | hr-remaining-firstpass | `61c65283a24da26e013d3241693481df41f54f8f6b36443aedffb7b00c002afa` | not substantively reviewed | retained |
| hr.policy_update_distribution | hr-remaining-firstpass | `573dd147227170b8f9e15ba8638f3d4fa9d79f00674c8b5e1cabdf8a3b87454e` | not substantively reviewed | retained |
| hr.probation_review_reminder | hr-firstpass | `6c6716a6b80e90b3b054310ae05eda868983713d5c605a85e42701aeff631790` | not substantively reviewed | retained |
| hr.promotion_announcement | hr-remaining-firstpass | `30d6e33a221f893a849eed3f7689ed29ff7437c98f12f0b39304cadeb9de9c22` | not substantively reviewed | retained |
| hr.pto_balance_alert | hr-firstpass | `0631fe21f666fc1665fa0a287ff31d7f762201b4a6b08cedb16c8b737f67c146` | not substantively reviewed | retained |
| hr.pto_reconciliation | hr-remaining-firstpass | `8e4974e51fbaeb5883d4570b274ab9cee6fadc829084448979067f788ce8706f` | not substantively reviewed | retained |
| hr.quarterly_termination_queue | hr-remaining-firstpass | `0766684a0dff7e4e15015c91574b67a31cc44865e01bbb3fd59b4abbd215738c` | not substantively reviewed | retained |
| hr.recognition_nominations | hr-remaining-firstpass | `83b2d3efa30b40fdb950c540adb6ff345577709d227e1159e1568d9cdf9af464` | not substantively reviewed | retained |
| hr.recruitment_channel_roi | hr-remaining-firstpass | `98bcecce66a506d44be608eeb49f09f158f3a465a68ceb815e71595c30032cd4` | not substantively reviewed | retained |
| hr.recruitment_dedup_alert | hr-remaining-firstpass | `af6c7a652bb7e871a452c886e18964924e794fe4c7dbc907718257ed743ced54` | not substantively reviewed | retained |
| hr.referral_bonus_processing | hr-remaining-firstpass | `8f7f64c543c8b292e5196922bb576be02f8cf6b4bbcf89ee70bd77a005dfe03d` | substantive | retained |
| hr.referral_bonus_tracking | hr-firstpass | `e4c34498de35028e1c3df57ece9ab088b4c68fd14a48783b17572cc62dbc0cdb` | substantive | retained |
| hr.return_from_leave | hr-remaining-firstpass | `8c40904b522b169568500fafa6dd7656e1c1c2075499fdf99ae9d1866f109317` | not substantively reviewed | retained |
| hr.rto_compliance_check | hr-remaining-firstpass | `3cba74e6bb5e2bbcf0a32d824d8b6928dc496fe8bef473270cbf275b338acfff` | not substantively reviewed | retained |
| hr.safety_incident_reporting | hr-remaining-firstpass | `ac185862ad59409ef6c52ca17f0319e7fd805a24038f8ea03f1c98e2966dd409` | not substantively reviewed | retained |
| hr.salary_adjustment_processing | hr-remaining-firstpass | `fb9d8bb53d2de1c992b5ed83f1318cc6b99a184ffc3ef17bb32a3e8484afc797` | substantive | retained |
| hr.salary_band_audit | hr-firstpass | `1aff8e04090f24bbe9b6bf37c9e0af171652fa4d92f0b79e4cac8b99d51dc540` | substantive | retained |
| hr.slack_benefits_inquiry | hr-remaining-firstpass | `42178823fd73cc4cb424f79bb5efa5f6acbad65650dcfc77440285b0ddac50c4` | not substantively reviewed | retained |
| hr.slack_channel_onboarding | hr-remaining-firstpass | `294afcba6259beabb961134fc7f9950aa4d171b473121b75cb16a7fac17bf331` | not substantively reviewed | retained |
| hr.standup_absence_tracking | hr-remaining-firstpass | `f50748821811d7a21cbe8e8065ea07465901595a437667aaf53c79065379856b` | not substantively reviewed | retained |
| hr.survey_distribution | hr-remaining-firstpass | `93d73af0facf3fcbf911dc57f2bdd4ce95134a2e33eec0fc3eaf82e3bf0cd853` | not substantively reviewed | retained |
| hr.team_restructure_notification | hr-remaining-firstpass | `1b488ce302c339de922d65883be52fe343c7f48f6489eb8f3cfff67a0f9affb7` | not substantively reviewed | retained |
| hr.termination_checklist | hr-remaining-firstpass | `b3b0c7f51ec2f19cb48fc1fc87375064876640106899677ea5a00dddbdd27d5a` | not substantively reviewed | retained |
| hr.training_compliance | hr-firstpass | `6a2e7ad2d21769b9facd59047e2a81c9051963f36cbc39e0a4657ecf272fab44` | substantive | retained |
| hr.trello_internal_transfers | hr-remaining-firstpass | `b7def3a0715cf31d1856362ec38698fe1361d24268c9c0b72a2c7528a914b8f8` | not substantively reviewed | retained |
| hr.trello_recruiting_event_coordination | hr-remaining-firstpass | `b2e3b5d03793888c6189fd3d15bfbab58f2a4a38d20756adcfcb133ba398a6b8` | not substantively reviewed | retained |
| hr.tuition_reimbursement | hr-remaining-firstpass | `b0f1ab716ba43a3ef03229c9f4a375c3a679744cf2663c2535186dc51d2129f9` | not substantively reviewed | retained |
| hr.twilio_emergency_notification | hr-remaining-firstpass | `eaeab6f661f2a702a5eb5e3d63ea1b3f0a00811cd4202e5cf4de9fe698a85e40` | not substantively reviewed | retained |
| hr.twilio_interview_reminders | hr-remaining-firstpass | `3005091850ffce9be338cbc4421c95270af153ced0a3150dab8075a37e6159de` | not substantively reviewed | retained |
| hr.visa_expiration_monitoring | hr-remaining-firstpass | `47db2a304362967b493d44c4e2640dd0500a45d670aa4248270320e854b3e235` | not substantively reviewed | retained |
| hr.visa_expiry_tracking | hr-remaining-firstpass | `eee0c6e77c6ab0d14641b63a891c23986d2158d5dc61609b391489a5d4179f38` | not substantively reviewed | retained |
| hr.workforce_planning_forecast | hr-remaining-firstpass | `386ea468364c5e37cb23e327f615ad1a7d28c3dc8d08846c32181e015fcca9eb` | not substantively reviewed | retained |
| hr.zoom_all_hands_setup | hr-remaining-firstpass | `468887b17ebbe490713ad6a871bd0393ee6911f7020c8ad480673a05cd8007b1` | not substantively reviewed | retained |
| hr.zoom_orientation_sessions | hr-remaining-firstpass | `ff5e58f4f7ec616337189c9b7b2b9108da7b51cf8be35d0bd6227c2403bdc6eb` | not substantively reviewed | retained |
