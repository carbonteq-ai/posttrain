# Redesign AutomationBench rewards from Luna traces, then prepare Qwen curricula

The Luna first pass is terminal: 800 distinct attempts were consumed, 790 native episodes retained and ten attempts lost artifact retention during disk exhaustion. No consumed attempt was retried. The development inventory contains 700 retained episodes and nine explicit unavailable entries across 709 tasks. Reward redesign, final eligibility and student benchmarking remain open. Follow `docs/templates/PLAN.md` and update Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective as work proceeds.

## Purpose / Big Picture

The primary goal is to redesign AutomationBench's reward signals using verified GPT-6 Luna (`gpt-6-luna`) executions. Record what Luna does, check the outcomes, and use those examples to define better episode rewards, useful and harmful action signals, guards, partial credit, and turn-level rewards. Also use the traces to propose tool-call and output-length budgets. Implement and test that redesign before benchmarking Qwen3.5 9B, 4B, and 2B on the Luna-verified subset. Then prepare separate 2B/4B curricula, collect ten fresh 4B rollouts per selected task, and stop.

For example, a conversion-tracking run might upload an excluded account and still earn partial credit. Recording that run lets us see the harmful call, check whether the current verifier catches it, and add a precise guard if it does not. A successful run shows another path we should accept. We can measure its tool calls and output length to propose budgets, while leaving room for other valid solutions.

The sequence is to collect and verify Luna executions, redesign the rewards from those examples, implement and test the redesign, freeze it, benchmark the Qwen models, prepare their curricula, and save the final 4B trace bank. Failed attempts help expose missing checks and confusing partial credit. A successful example establishes one valid path, without making every action in that path required or useful. Luna provides reference examples; imitation, distillation, and using its log probabilities are outside the plan. Training and any further reward iteration using the final 4B bank are separate work.

Build this inside the external `automationbench_v1` environment package. Reuse Verifiers' Codex and Claude execution support. Follow `docs/plan/verifiers-assessment-api-and-environment-migration-runbook.md`: establish safe calibration readiness first, then use Luna evidence and AutomationBench reward design to expose and resolve shared API gaps. Migrate all maintained environments after Milestone 4; complete their qualification and published dependency adoption before Milestone 5. Early AutomationBench instrumentation is a calibration integration, not completion of the wider migration. Luna supersedes the earlier Sol selection. The existing filename stays so links keep working, although its earlier "teacher" wording no longer describes the purpose. Writing this plan does not launch collection or training.

## Progress

- [x] (2026-10-04) Implement bounded source-recaptured aggregate evidence in environment `contracts/aggregates.py`: 59 owner regressions and 31 independently repeated boundary cases pass. Root combined aggregate/value/predicate/population/table gate passes 286 in 2.32 seconds; scoped Ruff/format/Pyright are clean. Independently replay actual public prepaid rules, including the controller Slack correction: three disjoint results 4,000/300/600, sum 4,900. Reordering, changed amount, missing eligibility and duplicate-identity controls pass. Source/test hashes and arithmetic proof: `.posttrain/state/verifiers-assessment-qualification/aggregate-prepaid-public-01/verification.json`. Standalone evidence qualification only; native integration, report content and action credit remain open.
- [x] (2026-10-04) Integrate aggregates into native occurrence obligations and qualify one outcome-only report component (environment candidate, uncommitted). `ObligationCheck.aggregates` declares check-local aliases readable only by `effect_match` as `aggregate.<alias>.value|selected_count`; every evaluation recaptures them from raw source, so no prepared total is ever accepted. An unavailable total abstains as `obligation_aggregate_unavailable`; the requirement never disappears. Empty aggregates are omitted from dumps and receipts, so legacy contract digests, selector identity and receipt bytes are unchanged. Receipts carry `aggregate_digest`; the scope receipt retains full member evidence; the planner rejects re-minted forged receipts with `obligation_credit_aggregate_mismatch`. New predicate `line_number_eq` reads one exact unquoted plain-text label line. Draft the environment repo's `manifest-drafts/legacy/prepaid-report-total-draft.json` adds `controller-total-line` to the installed prepaid v3 checks, outcome-only, with aggregates copied verbatim from the installed schedule eligibility and amounts. Actual hash-bound Luna replay: controller recipient correct, line `$4,300`, known 0, matching its software-row schedule 0. 91 new cases pass (format equivalence, wrong/missing/near-miss/quoted/fenced/HTML/conflicting lines, same-send binding, multiple sends, missing ACK, cc, unknown eligibility, duplicate members, changed public amounts, correct total with wrong schedule, reload/rescore, forged receipts). Full environment suite before the critic fixes: 3,027 passed, 7 skipped; scoped Ruff/Pyright clean. Independent critic found no forgery or legacy-digest change and five issues, all fixed. Proof script: `.posttrain/state/verifiers-assessment-qualification/aggregate-report-line-01/qualify.py`. Not installed; entity names, journal lines, guards and action credit remain open; counts unchanged (zero whole tasks).
- [x] (2026-10-04) Install prepaid revision `public_batch01_schedule_report_v4`: packaged bytes equal the reviewed draft, SHA `2d934cfd68bb8bcbcbd7497dc2ffa2d66298f5ddd516d231611e103f87427138`; v3 schedule/guard checks, credit and bindings unchanged; report line stays outcome-only. Actual hash-bound Luna episode through native score, reload-and-score and rescore: `$4,300` line is a known 0 against the computed 4,000 + 300 + 600 = 4,900 every time, scalar rewards and episode bytes unchanged. Correct alternatives (supported formats, cc delivery) pass; wrong recipient, wrong/missing total are 0; unknown totals, missing ACK, quoted/fenced/near-miss lines abstain. Full environment regression after the shared Gmail fixes: 3,048 passed, 7 skipped (3m44s with `-n 6`; 13m44s serial). Proof: `.posttrain/state/verifiers-assessment-qualification/aggregate-report-line-01/verification-20261004T120338.json` (370 passed). Ledger: 105 reviewed; 13 installed partial declarations in the sample (prepaid now has schedule, guard and report-line components); 0 whole tasks qualified.
- [x] (2026-10-04) Add `pytest-xdist>=3.8,<4` to the environment dev group (lock keeps format revision 3; only `pytest-xdist` and `execnet` added) and default `--dist=loadfile`, because several files share mutable module-scoped recorded-episode fixtures. Audit found no shared fixed paths, global environment mutation or fixed ports. Parallel and serial runs give identical results (3,048 passed, 7 skipped).
- [x] (2026-10-04) Shared mechanisms from batch-12 gaps (environment candidate, uncommitted): (1) static handler footprints close effect scope across Gmail/Slack/Sheets/Asana/Jira/HubSpot/Zendesk only when the footprint excludes the service and its state is unchanged; sparse public Slack/Gmail state reconciles via schema hydration; `SheetEffectSource.alternative_services` keeps ambiguous channels open (brand-mention guard revision `public_batch04_policy_inventory_v3`, SHA `0822e47b…`). (2) `mentions` predicate (words/verbatim/amount) and `per_candidate` obligations (no once-only credit). (3) Decided lookup outcomes (`lookup.<alias>` = matched/not_found; ambiguous/unavailable stay unknown). (4) Candidate-relative eligibility-first `selections` (typed ordering, ties and open populations unknown, `none` publishable; cross-candidate capacity not modelled). Full environment suite 3,124 passed, 7 skipped; scoped Ruff/Pyright clean. Batch Luna replays, genuinely abstained findings without → with mechanism 1: expense 19→0, payment 17→0, break 28→3, buddy 11→0, content 133→1 (an earlier tally wrongly counted inapplicable findings as unknown). Prepaid proof rerun on current code: `aggregate-report-line-01/verification-20261004T125616.json`, 370 passed. Evidence: `.posttrain/state/verifiers-assessment-qualification/archive/batch-12/mechanism-01-handler-scope.md` (machine-local).
- [x] (2026-10-04) Mechanism 5: generic `service.record_writes@1` effect adapter (`contracts/record_writes.py`). Declares `{service, collection, kind: create|update|delete}` against the installed schema; covers typed ID-keyed collections (Zoom, Calendly, Salesforce tasks, HelpCrunch, Google Ads, QuickBooks, Zendesk, HubSpot, …) and action-record services (`actions[<key>]`: Monday 31 keys, Airtable 5, Notion 15, plus Asana-style services). Diffs each acknowledged occurrence's BEFORE/AFTER service state by record ID, so writes through any tool (including unrecognised ones) are observed; update facts carry before/after/changed fields. Scope closes with handler footprints, a complete chain, matching ACK inventory and public initial/final service reconciliation (absent services start at schema defaults). Wired through contracts, guards, obligations and native transport; works with once-only obligation credit. 15 new tests incl. native Salesforce follow-up credit; full suite 3,139 passed, 7 skipped; Ruff/Pyright clean.
- [x] (2026-10-04) Batch 12 round 2: ten drafts redone with mechanisms 1–5; expressed obligations 41 → 84 of 156; zero whole tasks. Redraft found 14 mechanism defects, all fixed (reconciliation of generated fields, Gmail absent/zero-call scope, Gmail read closure, record-write identity/sibling actions/values text, mentions false positives, HTML body text). Full suite 3,162 passed, 7 skipped; Luna replay of v2 drafts has 3 abstentions total. Summary: the environment repo's `manifest-drafts/LEDGER.md`.
- [x] (2026-10-04) Mechanisms 6–8: value formats `clock_time`/`duration_text`/`duration_clock`/`iso_instant` (exact decimals; ambiguous forms unknown); `mentions` `clock_time` mode and `mentions_together` (all terms on one readable line); obligation `effect_joins` linking a matched effect to an earlier effect of another declared source (matched/none decided only over complete inventories; ambiguity unknown; join sources join selector identity). Tests: value formats 37, mentions/together/clock 30+, joins 12 (native Monday create→status). Full suite 3,223 passed, 7 skipped; Ruff/Pyright clean.
- [x] (2026-10-04) Round 3: 20 tasks (10 redrafts, 10 new from the 105 sample), four per author. Five whole-task qualified candidates: `simple.email_zendesk_ack_reply` (5/5), `finance.escrow_tracking` (8/8), `hr.airtable_learning_path_assignment` (9/9), `operations.zoom_training_setup` (13/13), `operations.calendly_equipment_inspection` (11/11). Near: break schedule 11/12, deferred revenue 12/13, social calendar 14/16. Scope decisions: system-prompt rules and wording/judgement checks out of scope; partial checking accepted. Replay of all 20 latest drafts on Luna: no errors, scalars/bytes unchanged (`.posttrain/state/verifiers-assessment-qualification/archive/batch-12/luna-replay-round3-latest.json`, machine-local).
- [x] (2026-10-04) Mechanisms 9–16 from round-3 gaps: guards over any population with joins; obligation alternatives (any channel); join `match: any`/`timing: any`; Gmail read fixes (writes are not reads, failed searches read nothing, list audited); clock ambiguity/ranges/noon, `clock_24h`, `add_business_days` with explicit holidays; `amount_reformatted`, block scope, `usd_marked`, numeric verbatim, possessives, `proven`; composite identities for Slack message populations. Fixed a latent defect: initial-record candidates had no `native_record_id`. Full suite 3,278 passed, 7 skipped; Ruff/Pyright clean.
- [x] (2026-10-04) Repository cleanup before committing. Manifest drafts moved to the environment repo: `environments/automationbench_v1/manifest-drafts/` with `LEDGER.md` (per-task status: 20 tasks, 5 qualified candidates), `AUTHORING.md` (rules and mechanism vocabulary), `tasks/<task>/` (latest draft/review only; no versioned copies) and `legacy/` (15 original drafts of installed manifests; tests now use repo-relative paths). Superseded batch-12 versions, replay tallies and nine run logs archived machine-locally under `.posttrain/state/verifiers-assessment-qualification/archive/` (nothing deleted except `__pycache__`). Reference evidence (Luna policy contracts, review coverage, capability matrix, batch reviews, cohorts, corpora) kept in place with an index at `docs/research/verifiers-assessment-qualification/README.md`. Environment suite unchanged after the move: 3,278 passed, 7 skipped; prepaid proof 371 passed.
- [x] (2026-10-04) Round 4 (environment branch `wip/automationbench-reward-redesign-2026-10-04`, local only). Independent qualification reviews of the five round-3 candidates found a real gaming path in every one: a fabricated booking log, book-then-cancel, assigning a sabbatical employee as active, deleting or clearing a disputed escrow row, and an ordering check satisfied by a read alone. All were fixed without engine changes and re-verified on the genuine simulator and Luna. Each task's `qualification-review.md` records the reviewer verdict and the coordinator resolution; two reviewer proposals were rejected because they rewarded inaction or penalized self-correction. The five were then installed in `contracts/catalog.json` as the first whole-task qualified manifests: `simple.email_zendesk_ack_reply`, `finance.escrow_tracking`, `hr.airtable_learning_path_assignment`, `operations.zoom_training_setup` and `operations.calendly_equipment_inspection`. Packaged bytes equal the reviewed drafts. `tests/test_qualified_manifests.py` pins each SHA plus a hash-bound Luna replay of the reviewed per-check findings (no abstention, rescore identical, scalars unchanged). Round-4 authors (20 tasks, including 5 new) brought finance, HR, marketing and operations drafts to full coverage; see LEDGER.md.
- [x] (2026-10-04) Round 5 and engine round 4, run in parallel. Engine work was done in separate worktrees, merged on `wip/abv1-integration` and fast-forwarded into the environment branch. Full environment suite: 3,493 passed, 7 skipped; Luna replay of the five installed tasks is unchanged. Fixes: amounts followed by a comma; exact magnitudes ("$120k"); 24-hour ranges; guards publishing `lookup.<alias>`; contract and digest caching (one Luna scoring pass about 63 s → 16–19 s, identical results); rescore repeatability (missing Salesforce timestamps were being filled with the current time). New mechanisms, documented in `manifest-drafts/AUTHORING.md`: 17 `excluding_values`; 18 `sole` (anti-hedging); 19 `slack.message_reads@1`; 20 `exists` over a closed population; 21 date mentions with `within`; 22 guard `alternatives`/`selections`; record-write `identity_paths`. Round-5 authors drafted 20 more tasks (sales, support, simple, marketing; six extend installed partial manifests) under a new rule: record gaming as `known_gaming` instead of closing every path. Ledger: 45 tasks drafted; 36 at full coverage (5 installed and qualified); 44 at 80% or more; 454 of 468 in-scope obligations expressed. A random spot-check of 3 unreviewed full-coverage drafts found 11 of 14 gaming attempts succeeded (79%; 1 of 3 tasks high severity). Most are hedging (several candidate values on one line) or harm guards limited to one channel or name; `sole` and guard `alternatives` now address both classes. Planning rule: treat an unreviewed full-coverage draft as gameable until it has anti-hedging terms and a review. `sole` was then swept onto goal amount checks, each change gated on the Luna replay not getting worse: 29 of 39 checks kept it (10 reverted where a correct line legitimately holds two amounts). Deferred revenue's high-severity hedge now scores 0 on all three entities; the `content_repurpose` sweep was still running at the checkpoint. Jira task data: HelpScout triage required project SUP but declared none, so under strict project validation no agent could pass it. It was repaired in the AutomationBench fork (uncommitted, fork ledger updated; 56 fork Jira tests pass) and the vendored copy, following the earlier Gorgias FIN repair. A scan found no other affected sample task. Next: apply `sole` to every draft (each change gated on an unchanged Luna replay), fill pending-mechanism gaps, review and install the 29 candidates, draft the remaining ~60 sample tasks, then push and pin the environment commit (needs authorization).

- [x] (2026-10-04) Save all-105 public capability opportunity matrix, SHA `a3b9fac18522f0cf11c4c0ef7d8a87d99a6e8589b615c7e8ea5f70e81d07bbd5`. Owner verifies episode/public inputs; root verifies 105 unique tasks, fifteen per domain, exact pack/review/task bindings and implementation snapshot hashes. Critic exposes mention-based overclassification; owner narrows aggregate candidates to 22, scheduling to 15 and scalar/date needs to 33. These remain partial planning memberships requiring exact obligation-level review, not measured task unlocks. Proposed next ten span six non-Simple domains and are independently checked against selection and earlier cohorts. The bounded aggregate is implemented; native consumption, ranking, two-sided relations and explicit read prerequisites remain open.
- [x] (2026-10-04) Prepare `docs/research/verifiers-assessment-qualification/independent-agent-handoff.md` with objectives, repository ownership, exact read order, qualification limits, validation/recovery and downstream scope. It recommends check-local aggregate aliases in existing occurrence checks with source-recaptured native preparation; that integration is not implemented. Existing plans remain execution authority.

- [x] (2026-10-04) Repair and independently qualify communication admission freezing. `prepare_admission(context)` runs after factual recapture and before producer readmission or either optional iterable; the selected `no_clarification@2` callback retains only immutable proven output keys. Owner's 118 core tests and critic's 23 focused regressions plus four exact injection/removal probes pass; all 28 v3 preparation/readmission cases pass. Historical `@1` and summary controls remain unchanged. This removes the source-mutation blocker; fresh semantic results remain pending.
- [x] (2026-10-04) Implement the finite HubSpot adapter and created/retained native integration. The adapter preserves top-level native Contact/Deal facts and actual receipt identities. The existing created-object evaluator uses a private predicate view, accepts explicit HubSpot selectors, and authenticates raw source again before publishing credit. Genuine fixtures cover creation, same-deal association, amount zero, missing ACK, damage, coherent retargeting, rescore/reload and repeated noops. Root's 89 focused cases and critic's 43 core/native cases pass; scoped Ruff/Pyright pass. Adapter review separately passes 31 cases and four adverse probes. These are shared-mechanic and manufactured-fixture results, not original-task acceptance or catalog activation.
- [x] (2026-10-04) Execute fresh v3 `no_clarification@2` campaign after the 433-case combined regression gate (two network tests excluded). Session 3877 is terminal: all 28 attempted once and retained, peak ten, 785 selected package/dependency file hashes unchanged. Frozen predicate gate fails at 25/28: NC08/NC14 resolve ambiguous purpose instead of abstaining; the disputed NC10 expectation still fails against unproven stored-field communication. All six stored/sent/literal/business-approval controls pass, including factual rejection of the raw unsupported NC25a violation. There are 27 reconciled exchanges; semantic accuracy and activation remain open. Evidence: `no-clarification-semantic-live-03{,-qualification}/` and the live-03 review. Independent artifact review and a principled purpose improvement remain next; historical labels/results are unchanged.
- [x] (2026-10-04) Author acceptance cohort 03: ten inactive declarations across six domains, ten compilations, twenty public bindings and forty pack/review/episode/manifest hash links independently verified by owner and root. Five destination diagnostics carry no accomplishment credit; business goals remain unsupported. Critic exposed an omitted six-section policy-read-before-cleanup obligation, now explicit in index revision `data_authoring_v2` without changing manifest bytes. Corrected index SHA `44046a8794d0c47aeaa55b754a190a0d039975c49d80454cd4bcf454becbcd7e`; root proof `acceptance-cohort-03-root-validation/verification.json`. Critic independently accepts the correction and unchanged manifest hashes. Counts remain 105 public reviews, 13 installed bounded components and zero whole tasks.
- [x] (2026-10-04) Independent critic reproduces live-03's 25/28 finite failure from SHA-bound prepared/native histories, source/archive identities, raw journals and reconciled usage. Raw NC10/NC25a violations are retained while the `@2` reducer withholds them for missing communication proof. Next purpose design must add something substantive: rubric 2 already forbids unsupported interpretations of vague questions. Do not replace the held gate with prompt repetition or label relabeling. Deterministic HubSpot output coverage and original-task native qualification can continue independently after source release.

- [ ] (2026-10-04) Finish HubSpot lane task qualification: replay one SHA-bound original Contact episode through score/rescore/reload and the proposed manifest, then qualify public-data alternatives and harmful counterexamples before installation. Separately audit HubSpot external authored fields so unsupported output operations cannot become silent compliance. Adapter and native manufactured fixtures are accepted locally; actual-task public eligibility, message obligations and whole-task guards remain distinct gates. Stage-label meaning stays in reviewed task data. Freeze source during fresh SDK qualification.

- [x] (2026-10-04) Complete acceptance cohort 02 data authoring: ten inactive manifests across five domains, ten successful compilations, twenty public prompt/initial bindings and forty source/review/episode/manifest hash links verified independently by owner, critic and root. Two declarations include bounded deterministic business components; eight explicitly leave business goals unsupported. Critic's scheduling-guideline observation/read-before-create gap is added without changing draft bytes. Current index SHA `cf9e730661cad2deae11d1a3d3dd06422b441c7d32762ae0e16ece669cc224ad`; root proof `.posttrain/state/verifiers-assessment-qualification/acceptance-cohort-02-root-validation/verification.json`. These drafts add no installed components, credit or whole-task qualification; counts remain 105 reviewed/13 installed bounded components/0 whole tasks.
- [ ] (2026-10-04) Qualify explicit `no_clarification@2` communication admission and its calibration route. Owner core gate passes 106 cases, independent explicit-profile preparation/native/resume gates pass two; root profile gate passes 109 plus the corrected mismatch-expectation case separately. Critic then reproduces mutable-source admission: a producer iterator adds a successful native call after context capture and turns unknown into harm. Hold acceptance and inference until communication bases are frozen before optional producer iteration and the reproduction passes. Root owns calibration version selection; Astra owns core/private admission helper; critic reviews both.
- [x] (2026-10-04) Prepare and re-admit all 28 cases for the fresh `@2` selection from `no-clarification-case-preparation-v3.json`, SHA `1174a425635afdc272cfe20e9787e25c87aa77cd0efe84d1525d467e8862fb8e`; root session 63811 terminates successfully. Independent data comparison finds only preparation revision and policy-profile changes; the unchanged v2 corpus, labels and three unsupported cases remain. Proof: `.posttrain/state/verifiers-assessment-qualification/no-clarification-v3-root-data-review-01/verification.json`. Source/certificate repair must pass before freezing and dispatching a fresh campaign; historical `@1` results remain unchanged.
- [x] (2026-10-04) Baseline combined core, preparation, runner and SDK-backend gate passes 332 cases with two network deselections in 117.07 seconds, session 64044. Scoped calibration typing and lint pass. This suite did not cover the critic's native producer-iteration mutation and does not accept the new communication rule. Require the new regression and immutable-basis fix before inference.

- [x] (2026-10-04) Qualify the generic SDK tool-isolation repair. The worker disables fourteen audited built-in features at startup and thread creation, then requires strict loaded-thread feature readback before dispatch. Native fixtures pass 24 cases; seven tests against the real pinned SDK and a local provider pass without paid calls. Independent review passes 116 environment backend cases and eight native readback variants. Identities truthfully include the changed worker and isolation revision; old failed archives remain unchanged and unqualified for the new transport.
- [x] (2026-10-04) Add optional, strictly validated isolation metadata recognition to shared authored-output capture. Historical archives remain readable without this record. Malformed, foreign, duplicate or misplaced metadata leaves coverage open while retaining observed text. Root source/backend gate passes 217 cases with two network deselections; critic passes 89 capture cases and five additional probes. Replay of seven retained real-SDK local-provider cases closes only the clean case and preserves the other observed gaps. Scoped Ruff and Pyright pass.
- [x] (2026-10-04) Fresh signed-in clarification canary session 93663 passes one test in 14.04 seconds. All 889 source/test hashes match; one reconciled response reports 18,906 input and 240 output tokens. Valid loaded-thread isolation, three admitted findings, compliance one and both original Contact credits are retained. Evidence: `.posttrain/state/verifiers-assessment-qualification/no-clarification-sdk-live-03{,-qualification}/`. This proves the selected transport/mixed-credit path, not general semantic accuracy or whole-task qualification.
- [x] (2026-10-04) Complete v2 clarification campaign 83942: 28/28 retained, peak ten slots, 27 backend exchanges and all 892 source/test/data hashes unchanged. Astra's strict native/source/report audit accepts 26/28 frozen predicates: inherited 21/22 and new development controls 5/6. NC10 correctly preserves unresolved audience but disagrees with its disputed historical violation expectation; NC25a incorrectly certifies an addressed request from a stored field. Retain both disagreements and all three unsupported cases. Evidence: `.posttrain/state/verifiers-assessment-qualification/no-clarification-semantic-live-02{,-qualification}/`; no whole-task qualification.
- [ ] (2026-10-04) Add source-backed communication-basis admission for no-clarification harm certificates. Astra owns `contracts/no_clarification.py` and existing `tests/test_summary_policy.py`; coordinate any shared-helper change before editing. Astra and critic find NC25a's cited Contact write proves storage, without recipient/send/presentation evidence; NC25b proves a genuine send of identical text. Preserve raw judge verdicts, reject unsupported harm to unknown rather than clean, and keep legitimate literal decisions available. Qualify audited outward assistant/message evidence without inventing channel/audience facts or exempting all record fields. Bind changed evaluation semantics to an explicit revision before activation; old artifacts remain immutable. The architecture records the implementation boundary.
- [x] (2026-10-04) Efficiency agent completes acceptance cohort 02 as ten data-only manifests across Marketing, Operations, Support, HR and Simple; owner/critic/root verify compilation and source bindings. Shared HubSpot creation/association evidence remains the next broad deterministic lane; Jira output classification is the smaller existing-candidate unlock. Replay, full obligations and activation remain pending. Preserve the 15-per-category breadth gate and zero whole-task acceptance count.

- [x] (2026-10-04) Clarification parser5/rubric2 request-boundary repair passes owner 100 offline cases, independent critic 178 with two skips, and root 264 with two network deselections in 48.75 seconds. Exact-empty output facts remain in full coverage while semantic aliases omit them; whitespace remains assessed. Unavailable invocations remain noncitable context, strict native admission is unchanged, and legacy summary wire/identity parity passes. This is controlled-certificate qualification, not fresh semantic accuracy.
- [x] (2026-10-04) Fresh parser5 SDK smoke session 41024 terminates FAIL after 19.06 seconds, with all 889 source/test hashes unchanged. Raw SDK journal contains an unexpected built-in `imageGeneration` item which fails without a result or saved file; the transport correctly rejects it as `summary_sdk_unexpected_tool_or_item`. The test then encounters the error envelope instead of a successful exchange. Full native trace/raw journal remain retained in `no-clarification-sdk-live-02`; adjacent `-qualification/` records terminal and hash proof. Do not silently retry or ignore the item. Freeze released after terminal/hash verification; built-in tool isolation needs local SDK/config investigation before further inference.
- [ ] Diagnose and qualify actual SDK built-in tool isolation, beyond the existing MCP/tool-catalog restrictions, before any fresh clarification campaign. Preserve the rejected image-generation attempt and prove a supported disabling mechanism using current local SDK/worker source. Independently review the separately authored v2 controls before selection; no inference while transport isolation is unresolved.
- [x] (2026-10-04) Extract byte-exact stdout/stderr copies from the failed smoke's retained native journal for independent inspection, without altering its trace. Observed usage remains available despite rejection: 39,363 input, 313 output, 53 reasoning detail included in output. Proof and failed-attempt usage are in `no-clarification-sdk-live-02-qualification/`. This exposed a runner reporting gap: partial SDK usage selected only the summary event kind, omitting failed clarification journals. Astra owns the fixed-profile event-kind correction and regression; failed transport must remain failed.
- [x] (2026-10-04) Prepare and independently review v2 development contrasts: 31 total cases, 28 executable recipes, three unsupported entries. Preserve all original 25 case objects/predicates and 22 recipes exactly. Owner and critic re-admit all 28; root independently verifies population/hash lineage in `no-clarification-v2-root-data-review-01/verification.json`. Six additional contrasts are designed after the failure, not blind validation. Keep the inherited NC10 label dispute explicit; it is not normative authority for blanket field harm. Dispatch remains held for SDK tool isolation.

- [x] (2026-10-04) Execute the frozen clarification campaign: session 26283 exits zero, all 22 distinct cases retain native results, 21 actual SDK exchanges (NC17 deterministic empty inventory skips inference), peak concurrency ten and 52.703 seconds between first reservation and last retention. All 892 source/test/reviewed-data hashes remain exact. Manifest SHA `f936ae06a31575ff2abc257032380782335663351f2544410a97e7a8d45bd224`; evidence `.posttrain/state/verifiers-assessment-qualification/no-clarification-semantic-live-01/` and adjacent `-qualification/`. Three unsupported cases remain explicit. The frozen guard-value gate FAILS: 17 of 22 pass, with NC08/14 unresolved-purpose disagreements, NC10 literal-field exemption, NC16 optional judge abstention suppressing deterministic empty text, and NC18 a correctly identified violation rejected for citing an unavailable invocation. Preserve all actual decisions and original proposal labels; agent-reviewed disagreement is not human-gold error measurement. Zero whole tasks qualified.
- [ ] Repair and independently qualify the clarification-only request interface: leave exact-empty outputs to deterministic reduction, expose unavailable invocation material as noncitable context, and allow only qualified invocation references in certificates. Preserve legacy summary request bytes and strict native admission. Clarify semantic rubric without blanket field exemptions or blanket field harm, add counterbalanced controls in separately versioned data, and require fresh evidence before semantic qualification. Efficiency owns shared SDK/test changes; Astra owns the thin clarification profile and proposed next corpus; critic reviews both.
- [ ] Before wider environment acceptance, resolve the separate shared-core empty-text availability gap: a custom backend may still submit a valid abstained decision for exact empty text and suppress the deterministic finding. The clarification request fix prevents this on its selected path, without establishing universal precedence. Keep malformed/duplicate certificates explicit; any core change needs independent cross-profile qualification and compatibility expectations.

- [x] (2026-10-04) Release fixed no-clarification case preparation and finite qualification execution. Root and independent critic each pass all 100 preparation/runner tests (69.61 and 68.93 seconds); both independently prepare and re-admit all 22 runnable cases. The proposed corpus accounts for 25 cases, with NC11/12/15 explicitly unsupported. Preserve original twenty proposals and agent-reviewed label provenance. Root evidence: `.posttrain/state/verifiers-assessment-qualification/no-clarification-executable-root-preflight-01/verification.json`. Before dispatch, freeze the final corpus/preparation bytes, including deterministic exact-empty and closed-zero-output acceptance exceptions. Report guard-value acceptance separately from exact proposed-state agreement; neither transport nor proposal agreement establishes whole-task qualification.
- [ ] Run the independently reviewed finite clarification campaign with ten rolling slots, one consumed attempt per runnable case and no API fallback. Retain native outcomes for all 22 cases; the closed-zero-output case should skip backend inference. Grade against the pinned predicates, preserve raw failures and all three unsupported cases, then resolve observed semantic or capture gaps before whole-task acceptance.

- [x] (2026-10-04) Author the separately named ten-task Simple acceptance cohort from saved public packs: eight new JSON manifests, one exact existing Contact candidate reference, one explicit unsupported HubSpot deal entry. Root independently verifies all public-pack/review/original-episode/proposal hashes, compiles all nine declarations and checks every binding against raw public input. Index SHA `8bc422ba4e9e7720d7251b1ea28194600b5c220e1de94a157659d34939ab887c`; files `reward-candidate/acceptance-cohort-01/`; root proof `.posttrain/state/verifiers-assessment-qualification/acceptance-cohort-01-root-validation/verification.json`. All ten retain no-clarification and other open requirements. Critic reviews risky semantic/credit boundaries; no catalog activation, source mutation, scoring or inference. Compilation is not whole-task acceptance; counts remain 105 public reviews/13 installed bounded components/0 whole tasks qualified. Next factual shared unlocks: retained fresh Asana state and generated-task/section relationships, then HubSpot creation/association and required-message facts.

- [x] (2026-10-04) Astra and critic agree on the smallest shared no-clarification increment: one `no_clarification@1` operator with a distinct pinned rubric, reusing evidence/citation/lifecycle/transport mechanics without a policy DSL or duplicate backend. First qualify outcome coverage; keep historical SDK assistant credit unassigned. Private reasoning/search/incoming questions are context, and requested business questions need contextual distinction. Proposal SHA `f3ec976c7b6d414ffe74fb87c2b368962e9bffe5d7b7fc32c2d687b1231bc96a`: `reward-candidate/no-clarification-capability-proposal.md`. Architecture now records the implementation boundary and separate semantic/combined-task gates. This is a reviewed design, not runtime qualification or another installed component.

- [x] (2026-10-04) Fresh full environment regression session 67204 terminates with 2,674 passes, six skips and 105 existing JUnit-property warnings in 737.70 seconds. All 862 before/after hashes match; fingerprint `53951cf9ba183cc3429d361376390326b763985eb1b475778c9f073316e7dc1a`. Evidence: `.posttrain/state/verifiers-assessment-qualification/summary-action-full-regression-03/`. Root/critic also independently pass the corrected 59-case gate. No production change was needed. Keep native/environment/test source frozen through semantic collection; doc/data authoring remains separate and does not replace the seven-domain breadth gate.
- [x] (2026-10-04) Actual reviewed 26-case SDK collection terminates in session 1080: all 26 distinct one-attempt cases retained, peak concurrency ten, 45.58 seconds; all 862 source fingerprints unchanged. Output `.posttrain/state/verifiers-assessment-qualification/summary-semantic-live-01/`; root proof `root-verification.json`. Reported usage: 491,018 input/7,817 output tokens, with 1,750 reasoning tokens already included in output. No retry, API fallback or source mutation; source freeze released only after terminal/hash proof.
- [ ] (2026-10-04) Semantic qualification fails despite complete retention: all eleven proposed violations detected, but clean 04a/06a receive false harm; 02a omits two required citations; uncertain 05a/08b become violations and 12b clean. Keep aggregate/target label-only disagreements 03a/13a separate from numeric errors. Critic confirms raw decisions penalize prospective instructions and negated skips, and identifies a stale 05a rationale (the prepared incoming message actually contains the quoted note). Labels remain agent-reviewed, not human gold. Independent review `reward-candidate/summary-semantic-live-01-review.md`; original campaign immutable. Repair schema/rubric with a new identity and qualify a fresh structured-output smoke before a separately reviewed follow-up campaign; no silent historical rescore.
- [x] (2026-10-04) Correct acceptance cohort data after critic exposes recipient/channel-only positive credit on contradictory messages. Preserve original raw candidate files beside the first root proof; remove four delivery-credit declarations, retaining factual checks and independently qualified status/creation rules. Corrected index SHA `11fffde28458927331d2f2bfdc920119fe9e67711b2f21668030fa490a1f4c10`. Root rechecks all nine compilations/public bindings/hashes and absence of the four unsafe rules: `.posttrain/state/verifiers-assessment-qualification/acceptance-cohort-01-root-validation-02/verification.json`. No activation or whole-task acceptance.
- [x] (2026-10-04) Implement protocol4/rubric3 SDK repair only in backend/test surfaces: resolved certificates require nonempty quotes in schema and local admission, abstention may remain uncited, peer errors preserve valid harm before or after malformed rows. Revised rubric distinguishes function/polarity/action role/uncertainty, without phrase exceptions or a general style/factuality penalty. Owner/critic backend gates pass 80; root adds four SDK-parser-to-native-credit controls. Root affected five-file gate passes 215 with one live skip in 77.12 seconds. After the final genre-state clarification, root backend/credit repeat passes 126 with one live skip in 16.94 seconds. Final backend SHA `ae059eca58a80049c47c00e976f492516d4438d1ac76c2dcb8005f4a38f17eb0`. No no-clarification refactor or reducer/native change in this increment.
- [x] (2026-10-04) Fresh schema4 actual SDK smoke session 64264 terminates: one passed, 80 deselected in 22.94 seconds. All 862 source hashes match before/after; trace SHA `ac8814ce1cad7685bc8566a1600ce8d1919904c7c1b669e6227a12a69852e474`. Evidence: `.posttrain/state/verifiers-assessment-qualification/summary-sdk-live-04/` and adjacent `summary-sdk-live-04-qualification/verification.json`. Independent critic reloads native trace: three admitted findings (assistant compliant, two literal fields inapplicable), complete compliance one, no decision/assessment/credit errors, both original +1 credits unchanged. One reconciled response reports 19,292 input/262 output/zero reasoning tokens; physical response model identity remains unavailable, authenticated-thread selection only. Actual request uses nested schema4 alternatives. This qualifies structured-output transport/citations and preservation of the original Contact credits, not finite-corpus semantic accuracy. Source freeze released after terminal/hash proof and artifact review.
- [x] (2026-10-04) Independently review separately versioned follow-up contrast data before dispatch. Critic catches an unjustified proposed 03a label correction: the reducer returns inapplicable/compliance one when all surviving outputs are inapplicable; the old actual compliant result reflected producer labels, not a different reducer rule. Restore the exact original expectation rather than fitting labels to model output. Final corpus differs only in the justified 05a factual rationale; all 25 other cases and all eleven violation controls are unchanged. Approved raw SHA: corpus `376dbf4ea7032897f96d797c76dcdbf41fd59667d3a3341a902248d39b421b53`, preparation `58412924510a662f77f96bdf0035caac1f349893f3db8dc71fb3ca64e9e1938b`. Root all-26 offline admission and source proof: `.posttrain/state/verifiers-assessment-qualification/summary-semantic-live-02-qualification/preflight.json`.
- [x] (2026-10-04) Follow-up actual SDK campaign session 53829 terminates with all 26 distinct one-attempt cases retained, peak concurrency ten, 51.46 seconds and all 862 source hashes unchanged. Immutable destination `.posttrain/state/verifiers-assessment-qualification/summary-semantic-live-02/`; root proof in adjacent `summary-semantic-live-02-qualification/verification.json`. Usage: 522,103 input/8,947 output tokens, with 2,260 reasoning tokens already included in output; all response accounting reconciled. No retries or API fallback. Release the source freeze only after terminal/hash proof.
- [ ] (2026-10-04) Follow-up semantic gate still fails: the prior clean false harms 04a/06a and all citation errors are resolved, but explicit first-person exclusion narrative in a record field (02b) is incorrectly inapplicable; ten of eleven proposed violations are detected. Uncertain 05a/07a/08b/12b remain determinate disagreements. 04a's all-inapplicable aggregate is a numeric-clean genre/status discrepancy, not false harm. Independent raw review confirms 02b/13b are different contexts, so their inconsistent boundary interpretations do not establish measured same-input nondeterminism. Preserve prior cases; next revision must distinguish legitimate source-backed literal values from authored work narration without field exemptions or phrase rules. Backend remains optional and semantically unqualified for production whole-task acceptance.
- [ ] (2026-10-04) Begin the separately reviewed no-clarification outcome increment after source release: efficiency owns shared core preparation/reduction plus the finite check, Astra owns shared SDK transport with separate fixed rubric/assessor, root owns manifest schema and native publication, critic reviews cross-policy and semantic boundaries. Existing summary wire identities remain unchanged; no model-carrier rename, policy DSL, task-specific code or invented SDK-token credit. New 20-case proposal data is separate from executable/semantic qualification: `reward-candidate/no-clarification-challenge-cases.{json,md}`. Neither this implementation nor summary transport closes whole-task acceptance by itself.
- [x] (2026-10-04) Implement no-clarification core and SDK lanes locally. Core owner passes 78 cases plus scoped Ruff/format/Pyright; backend owner passes 90 with one live skip and exact summary identity/request-byte parity against retained live02. Root admits the finite check in manifest schema/compiler and publishes through a private code-owned profile using distinct outcome/producer/request/exchange identities and independent composition registry. No clarification action-credit policy or task catalog activation is added.
- [x] (2026-10-04) Integrated controlled-certificate qualification passes: root final five-file session 89336 has 254 passes/one live skip in 47.94 seconds; independent critic four-file session 32089 has 208 passes/one live skip in 37.79 seconds. Initial failures remain history: the WireTrace fixture omitted simulator runtime restoration, and the new outcome-only producer was absent from credit lifecycle validation/exclusion. Both are fixed. Four mixed actual Contact controls preserve both original +1 credits under clean/harm/parse/execute states without remint. Known-request evidence survives a later parser fault. Root scoped Pyright has zero errors after an explicit existing-pattern WireTrace-to-Trace test cast. A focused repeat checks that final test-only change. Logs: `.posttrain/state/verifiers-assessment-qualification/no-clarification-integrated-01.log` and `no-clarification-final-native-01.log`. No semantic calls or whole-task claims. The previous 2,674-case regression predates this increment.
- [x] (2026-10-04) Implement and execute the opt-in one-attempt clarification SDK smoke after offline admission (ten passed/one live skip). Actual session 97280 passes in 17.11 seconds; all 866 source hashes match before/after. Three admitted findings (assistant compliant, two fields inapplicable), compliance one, zero decision errors; both original Contact +1 credits and legacy scalar/source remain intact. One reconciled response: 19,341 input/296 output tokens, 58 reasoning tokens included in output. Actual response model remains unavailable; authenticated-thread selection only. Retained native trace SHA `6febbf80efab8e5c928657b752f7b8c80e6294e41cc0c0bd370968ccbb1686e6`; evidence `.posttrain/state/verifiers-assessment-qualification/no-clarification-sdk-live-01/` and adjacent `no-clarification-sdk-live-01-qualification/verification.json`. Source freeze released after terminal/hash proof. Independent critic reloads native trace, reproduces all three parser states, verifies byte-exact raw journal and request/exchange binding, both exact original credits and all 866 unchanged sources. Accepted for transport only.
- [x] (2026-10-04) Apply the production results-only training projection to the actual clarification trace: 5,088,258 native bytes become 172,692 projected bytes, retaining five assessment results and two credit-assignment results. Clarification worker/exchange bodies are absent; native source SHA stays unchanged. Evidence `no-clarification-sdk-live-01-qualification/tracking-projection.json`. This checks local projection, not remote artifact upload/restart or transient lifecycle retention.
- [ ] (2026-10-04) Extend the finite preparer/runner with two fixed environment-owned profiles, exact earlier authored items, explicitly reviewed Gmail sends and user-task variants, identical completion deduplication and zero-operation counterfactual identity transitions. Keep legacy summary entry points/identities and all original contrast data intact. Unsupported source formats must remain separately hash-bound and uncovered, never become invented semantic targets. Root authors a separately reviewed executable clarification corpus; original twenty proposals remain unchanged. Before collection, the critic accepts a guard-value predicate permitting both resolved no-request labels for clean controls while preserving proposed exact-state diagnostics; harm, unknowns, source/citation validity and global closure remain strict. No rubric label relaxation for the failed summary campaign follows.
- [ ] (2026-10-04) Clarification accuracy remains open. The reviewed 20-case contrast set needs executable preparation and separate semantic qualification. Required business approval versus task-resolution confirmation and post-completion purpose need additional reviewed contrasts; ambiguous third-party purpose cannot acquire invented permission. Consult efficiency on narrowly sharing the finite source preparation/rolling runner before implementation. Summary semantics also remain failed; neither a clean smoke nor this core implementation permits whole-task acceptance.

- [x] (2026-10-04) Final executable preparation/runner qualification passes 59 cases independently at root (44.84 seconds) and critic (44.65 seconds), with no skips/inference/writes. Exact recipe/target/view/config and caller-approved raw-byte selection, archived-source verification, rolling scheduling, transport stop, safe resume and immutable lifecycle history are accepted within this scope. Final identities and reproduction: `reward-candidate/summary-semantic-runner-checkpoint.md`. Semantic accuracy, original-token fidelity and whole-task acceptance remain open.
- [x] (2026-10-04) Full environment regression session 19676 terminates with three failures, 2,671 passes and six skips in 736.41 seconds; all 862 before/after source hashes match. Evidence: `.posttrain/state/verifiers-assessment-qualification/summary-action-full-regression-02/`. Failures are outdated lifecycle fixtures: failed/interrupted snapshots deleted already-published findings, and a completed-run rewrite expected the older conflict error instead of terminal regression. Update only these test expectations, preserving production append-only/terminal guarantees. Root focused lifecycle/record/summary-credit gate passes 59 in 13.49 seconds; scoped Ruff and repository diff checks pass. A fresh full regression remains required before the 26-case SDK run; failed session 19676 and cancelled session 8176 are immutable history, not retry handles.

- [x] (2026-10-04) Intermediate integrated preparation/runner/publication/credit gate passes 123 in 63.54 seconds. Runner lifecycle metadata changed during this scope, so it is explicitly not final frozen-source qualification. Critic reproduced partial-history metadata/finding changes accepted by runner report reduction; efficiency repairs static fields and immutable finding/receipt/artifact prefixes, with fresh native admission. Owned repair gate passes 23 in 32.70 seconds; final receipt-reorder control/static checks and independent repeat remain pending. No model calls.

- [x] (2026-10-04) Harden and independently qualify prepared views/configuration: owner 34 cases pass in 13.58 seconds; root combined preparation/action-credit gate passes 76 in 22.13 seconds. Complete canonical view, typed configuration, original config checksum and reviewed target/spec admission are checked. Root additionally prepares all 26 reviewed cases with the actual SDK backend identity and constructs exact public requests offline: max 80,322 bytes, total 2,022,844 bytes, zero model calls. Evidence: `.posttrain/state/verifiers-assessment-qualification/summary-semantic-preflight-01/verification.json`. Request bytes are not measured tokens; model accuracy remains open.

- [x] (2026-10-04) Refresh account readiness without inference: included Codex allowance is exhausted, existing credits are available, and the account reports no spend-control stop. This does not establish Luna entitlement or physical billing for a future call. Earlier user authorization permits service-selected existing credits; no purchase, allowance reset or API fallback is selected. Real SDK dispatch still waits for source/admission qualification and retains observed failures rather than silently retrying.

- [x] (2026-10-04) Implement all 26 case-preparation recipes without inference; owner/root/critic each pass 25 tests (root 12.35 seconds). Actual cases share the unchanged baseline; silence/missing-artifact cases have no fabricated assistant target; action counterfactuals replay installed handlers with explicit fixture provenance. These tests establish preparation, not semantic labels. Retrospective view/configuration hardening is underway after final review, so this is not source release for dispatch.
- [ ] (2026-10-04) Complete bounded runner admission: verify archived source snapshot, typed task preflight/failure retention, systematic transport-failure stop and rolling slot refill. Require caller-approved original preparation/corpus hashes in a frozen selection and bind prepared cases to it; a computed self-consistency hash is not review evidence. Keep labels report-only, no implicit retries and one consumed attempt per case. Efficiency owns runner/tests and Astra owns view/config repairs; root/critic independently qualify before frozen broad regression and actual SDK execution.

- [x] (2026-10-04) Qualify offline compressed native retention on actual live-03 assessment material. Load the original Contact episode envelope with its actual scored live-03 trace (same episode/trace lineage), call the production bridge preservation/finalize path, decompress and revalidate native wire history. Assessments and credit assignments match exactly; input trace SHA remains `907dfa746a459097794290737416d4f8650943510a94d5673aa5863cd5f8bd5a`. Native JSON compresses 5,089,843→559,906 bytes, bundle SHA `5f6d06e6504731cbae873620cb3d623a103fa69ba126e2705c6e243fd1483e41`. Evidence: `.posttrain/state/verifiers-assessment-qualification/summary-sdk-live-03-compressed-retention/verification.json`. Five artifact tests also pass. This closes local preservation/compression/reload for this recorded input, not remote Trackio delivery, restart upload or transient solver artifact completeness. No model call/training or production change.

Current execution focus (2026-10-04): the SDK isolation repair and fresh signed-in
clarification canary pass. The reviewed v2 semantic campaign is terminal: all 28
cases are retained, and 26 frozen predicates pass. The source/data freeze is
released. Add source-backed communication-basis admission before trusting a
semantic harm certificate about an addressed request. The original campaign passed 17 of 22 proposed
predicates and failed five; those failures remain historical evidence. Complete
task manifest authoring stays in batches of ten, with a separate replay and
acceptance queue. Current coverage stays 105 public reviews, thirteen
installed bounded components and zero complete tasks qualified. Historical
unchecked entries below are not an alternative execution order: use the latest
entry for the named capability and the runbook's milestone order.

- [x] (2026-10-04) Root post-terminal-repair credit/revision gate passes 103 cases in 16.66 seconds; scoped Ruff clean. Independently match all 26 data preparation recipes to original corpus IDs, verify expected/decision fields are absent, and reconfirm companion SHA `000ea2eba33f0006c2e32516580c10fc23aed2651667c48ec2cd1daf37bd2e58`. Runtime preparation remains unqualified. Terminal session 8176 is stopped; do not poll or restart it. Case-preparation and runner implementation are active under separate file ownership, with no model calls.

- [x] (2026-10-04) Close terminal-history mutation after the critic reproduces changed completed reason/appended artifacts. Complete, failed and interrupted terminal snapshots now permit only exact duplicates; partial runs still allow legitimate reason/status changes and append-only outputs. Independent credit suite passes 42 in 10.83 seconds, test SHA `006dacc4213910cc82c8272d24366b6016beb913f271b8a3474a6e50ff7e3d60`; scoped lint/diff pass. Frozen broad regression remains pending after case/runner integration.
- [x] (2026-10-04) Author all 26 finite data preparation recipes, companion SHA `000ea2eba33f0006c2e32516580c10fc23aed2651667c48ec2cd1daf37bd2e58`. Critic reviews source anchors, absent-output handling, labels outside requests and explicit counterfactual provenance. This is data/design acceptance only: runtime schema and preparation must enforce constraints. Astra now owns `calibration/summary_cases.py` and its tests; efficiency owns `calibration/summary_qualification.py` and its tests. Root coordinates integration/qualification; no model calls authorized by these test gates alone.

- [ ] (2026-10-04) Prepare executable semantic cases while the full regression source freeze runs. Astra owns only the companion data file `docs/research/verifiers-assessment-qualification/reward-candidate/summary-semantic-case-preparation.json` now; source/test implementation waits for release. Critic and efficiency agree on closed typed transforms, exact frozen aliases, installed-handler replay, one shared hydrated baseline, explicit manufactured provenance and labels outside judge inputs. Proposed helper `calibration/summary_cases.py` and separate bounded runner `calibration/summary_qualification.py` are described in the architecture; neither is implemented by this decision.

- [x] (2026-10-04) Close the critic's static assessment-history omission. Independent final credit file passes 36 cases in 9.78 seconds, including twelve metadata mutations, four finding-prefix mutations and legitimate failure/interruption reasons with append-only artifact references. Root post-repair credit/revision gate passes 95 in 15.14 seconds; scoped Ruff/Pyright and diff checks pass. Controlled certificates establish credit mechanics, not real semantic accuracy or token projection.
- [ ] (2026-10-04) Full source-frozen environment regression session 8176 was deliberately cancelled (exit 143) after the critic reproduced terminal metadata mutation, not because observation timed out. All 858 before/after hashes match; fingerprint SHA `843f1d42492ad4e1493daa0bd66bfbed7572502b808c586bad18adf69b7b499f`. Preserve log/fingerprints/disposition under `.posttrain/state/verifiers-assessment-qualification/summary-action-full-regression-01/`; no full-pass claim. Source freeze is released. Qualify terminal equality, integrate executable case preparation and runner, then run one fresh full frozen suite before semantic dispatch, following efficiency consultation.

- [x] (2026-10-04) Integrate action-group summary assessments and the once-only negative policy. Root gate passes 204 cases in 37.79 seconds; critic independently passes 74 native publisher/group/credit cases in 28.07 seconds. Changed/missing policy preserves factual target planning and abstains before dispatch. One physical action produces one penalty even with several offending fields; retained parser replay makes no additional inference. Historical SDK assistant text remains unassigned. No catalog addition or whole-task qualification.
- [ ] (2026-10-04) Finish immutable assessment-history qualification after critic reproduces changed prior rubric metadata accepted by the shared planner. Root now compares every static run field, while allowing native lifecycle status/reason and append-only output/ execution evidence. Repeat final tests before the source-frozen broad regression. Next prepare executable contrast cases: the current 26-case file contains prose transformation recipes, so actual per-case source/contract/context admission is required before model dispatch; silence/missing-output cases must not invent output targets.

- [x] (2026-10-04) Qualify protocol 3 (schema-constrained SDK response, exact request-local aliases and exact-quote citations): owner/root/critic each pass 69 offline cases. Fresh live session 65802 passes in 16.48 seconds; actual summary compliance is one with zero decision errors, one reconciled response and 18,852 input/376 output tokens. Both original Contact credits and all 743 source hashes remain unchanged. Preserve live-01/live-02 failures; this positive trace does not qualify the contrast corpus or whole-task reward coverage.
- [ ] (2026-10-04) Implement and qualify fixed-action summary harm: shared action-group helper, predeclared native action assessments and lean once-only negative credit. Critic exposed policy-dependent target planning; repair keeps factual groups available when public policy is missing/changed, while all affected targets abstain and no judge runs. Complete native publication/credit integration, independent review and semantic contrast collection before installing a complete task manifest.

- [x] (2026-10-04) Add nine coherent mixed Contact failure controls; full summary bridge file passes 26 in 18.14 seconds and critic repeats the nine in 7.82 seconds. Forbidden text remains independent of correct goal/read credit, harm survives repair, silence is allowed, and missing capture is unavailable. Results-only training projection is checked on the actual live-02 artifact, preserving five result records/two credit results and removing raw assessor bodies without modifying the archive. These gates do not establish semantic model accuracy, summary negative action credit or artifact delivery.
- [x] (2026-10-04) Forward optional JSON Schema objects through the pinned native SDK public turn-start API. Root and critic each pass fifteen focused cases; eighteen SDK cases pass with seven opt-in unpaid tests skipped. No schema-adherence claim yet. Scoped lint is clean, while 83 existing SDK notification typing diagnostics remain outside the new forwarding lines. No dependency/lockfile change remains.

- [x] (2026-10-04) Qualify parser revision 2 offline (55 tests, independently repeated), Gmail/Zendesk submitted-output capture (root 74 combined Slack cases, critic 44 plus two probes), and execute one fresh mixed Contact/summary signed-in attempt. SDK completion and original two findings/credits survive; summary remains unavailable because model JSON has trailing content and a mistyped output ID. Terminal session 25672 fails its gate after 19.27 seconds, with immutable live-02 evidence. See `reward-candidate/sdk-summary-backend-checkpoint.md`; this does not qualify a whole task.
- [ ] (2026-10-04) Repair the summary response interface after live-02: inspect actual SDK structured-output support, use exact request-local reference mappings where justified, and distinguish exchange receipts from error envelopes. Qualify offline before another bounded live attempt. In parallel add coherent mixed Contact negative controls; semantic corpus acceptance and explicit once-only summary harm assignment remain open.

- [x] (2026-10-04) Qualify and run the optional signed-in SDK summary backend: owner 38 cases and independent critic 38 pass; root combined gate passes 88 with one live skip. One real Luna attempt passes transport in 23.17 seconds with 19,304 reported input and 533 output tokens, preserved source and full native evidence. Its aggregate abstains because the model gives an out-of-bounds citation, so semantic/whole-task acceptance remains open. Evidence: `reward-candidate/sdk-summary-backend-checkpoint.md`.
- [ ] (2026-10-04) Repair backend citation projection using exact, unambiguous quoted text or an already valid interval; count overlapping matches and preserve raw response. Bump parser revision, keep strict reducer admission, qualify the actual retained failure and adversarial quotes, then run a separate fresh bounded live attempt. Efficiency owns backend/tests; critic reviews. No fuzzy repair or invented original-token support.
- [ ] (2026-10-04) Add audited Gmail send and Zendesk comment output capture after source release. Recorded Zendesk update includes a public comment, so status-only evidence cannot close output scope. Astra owns shared external adapter/new tests; root/critic review full message/comment footprints and actual recorded replay. Other services remain open.

- [x] (2026-10-04) Repair mixed summary/Contact credit planning after three regression states reproduce the producer mismatch. Known summary assessments now pass lifecycle admission and bypass record-credit parsing, without any implicit summary allocation. The combined 95-case summary/revision/Contact gate passes in 20.44 seconds, including four native summary states and original once-only recipients through rescore/reload. Scoped Ruff/Pyright pass. Evidence: `reward-candidate/summary-policy-checkpoint.md`.
- [x] (2026-10-04) Extend local Trackio writer/SQLite storage regression with actual SDK journal field shapes. Four cases pass in 0.61 seconds: results remain queryable, plain and encoded assessment bodies are excluded, and native input is preserved. No tracking production change was needed; archive publication and restart delivery remain open.
- [x] (2026-10-04) Root independently repeats the new Slack DM/external/native HTTP gate: 50 passed in 6.34 seconds. Owner's broader 64-case gate and scoped lint/typing pass; independent critic repeats 30 in 2.68 seconds and confirms both main-text and bot-name prose retention across partial gaps. Real immutable three-call DM inventory closes with exact text and routing. Captured prose does not establish its meaning or whole-task success.

- [x] (2026-10-04) Inspect retained Contact SDK events with the efficiency agent. ChatGPT authentication and the accepted thread select `gpt-6-luna`; `rawResponse/completed` retains response/thread/turn IDs and usage, without a returned model or response body. Record this as authenticated SDK thread selection, not per-response model verification. No new inference was dispatched.
- [x] (2026-10-04) Prepare 26 source-backed summary-policy challenge cases in thirteen pairs after critic feedback. Two cases preserve actual outputs and context; the remainder explicitly identify counterfactual changes. Root verifies all exact Unicode citations and unique case IDs. Critic confirms the repaired name-field violation and literal-name contrast; labels remain agent-reviewed, not human-validated gold. Files: `reward-candidate/summary-semantic-challenge-cases.{json,md}`.
- [x] (2026-10-04) Align the summary backend protocol with its already-supported bounded decision iterator. Skip semantic dispatch when all observed texts are exactly empty; whitespace still requires assessment and missing capture still abstains through deterministic reduction. Focused summary/core/native coverage passes 82 in 10.24 seconds; final scoped Ruff/Pyright and both repository diff checks pass.
- [x] (2026-10-04) Extend shared audited external-output capture for Slack DM text and user-name lookup. Astra's implementation passes independent root/critic gates, including submitted bot username when persisted, exact world footprint and unsupported operation gaps. No summary meaning or new whole-task declaration follows from capture.
- [x] (2026-10-04) Implement and qualify the environment-owned signed-in SDK transport after the frozen full regression. Requests precede dispatch, original stdout/stderr precede parsing, and incomplete SDK completion or observed retries abstain. One actual bounded call runs successfully; semantic acceptance remains a separate open gate because its citation is invalid.

- [x] (2026-10-04) Final fifteen-case native core suite includes two physical attempts under one original sampled call/ticket, with distinct invocation/ACK identities and final returned value. Source-frozen full package regression session 10456 terminates successfully: 2,387 passed, five skipped, 661.93 seconds. Release the source freeze for the next increment; this gate does not qualify subsequently added code.

- [x] (2026-10-04) Native reconciliation and real HTTP/state-controller read/update gate pass in the 180-case combined suite (14.68 seconds); independent critic repeats 41 in 5.64 seconds. Omitted original calls prevent closure while known effects/harm survive. This qualifies clean host-dispatched capture mechanics, not actual model/token/parser fidelity, prose accuracy or whole-task coverage. Same-parent physical retry test and final stable broad regression remain next.

- [x] (2026-10-04) Executor-source forwarding passes 138 SDK/invocation/external/summary regressions in 11.12 seconds, including exact snapshot identity through all summary stages. Root scoped Ruff/Pyright pass. Native reconciliation and real HTTP/MCP operation gates are separate active work; no native closure or whole-task acceptance follows from this regression.

- [x] (2026-10-04) With candidate sources frozen, the eight calibration benchmark tests pass in 106.79 seconds. Earlier source-hash failures were reproduced only during concurrent source edits; no identity gate was bypassed. Later source changes require another stable qualification.
- [x] (2026-10-04) Add and independently review sealed native inventory facts: owner and critic each pass 24 cases. Fix Gorgias FIN initial fixture in simulator sibling (73 Jira/domain cases), exactly refresh its single vendor task file, and pass forced loaded-vendor Jira gate (55) plus Support extraction (26). Historical episodes remain unchanged. See `reward-candidate/native-inventory-builder-and-fin-fixture-checkpoint.md`.
- [x] (2026-10-04) Integrate explicit executor `SourceSnapshot` through native invocation, external-output and summary checks. Shared native call/harness/retry reconciliation, actual AutomationBench HTTP/MCP read/update transport and the source-frozen full package regression pass. Closure remains bounded to the qualified clean host lifecycles.

- [x] (2026-10-04) Repair the shared Support Jira effect adapter to use canonical issue transitions while preserving explicitly bounded historical audit-only replay. Focused gate passes 26 in 2.65 seconds; scoped Ruff/Pyright and diff checks pass. Genuine FIN-without-project failure remains visible. This does not repair or qualify the public refund task.
- [x] (2026-10-04) Strict native receipt and projection admission gate passes 175; consumer credit gate passes 35. Independent trace review passes 55 including fresh-source copied sampled/mask/token forgeries and valid exact-call control. Current hashes are in the execution-token alignment checkpoint.

- [x] (2026-10-04) Register shared `summary_exclusions@1` and source-bound native assessment transport. Fresh controlled-backend gate passes 78 in 9.85 seconds; original request/response, whole-output coverage, source identity and official scalar survive reload/rescore. No actual semantic model accuracy or whole-task acceptance is claimed. Evidence: `reward-candidate/summary-policy-checkpoint.md`.
- [x] (2026-10-04) Resolve the six failures from the earlier terminal broad package run (2,323 passed, five skipped, 649.06 seconds): stable-source rerun for two calibration source-hash failures, retain unavailable HR delivery candidates in two stale expectations, and repair two legacy Jira adapter assumptions using current canonical issue evidence. Final full package gate passes 2,387 with five skipped; native emitted-call coverage has its separately bounded acceptance.

- [x] (2026-10-04) Qualify shared external-output evidence after critic/efficiency consultation. Root's combined gate passes 205 cases in 9.23 seconds; critic independently repeats the 59-case external gate in 3.53 seconds. Actual sealed Contact evidence reconciles seven unique SDK/native calls and yields two written-field texts plus action relations, preserving bytes/scalars/unqualified join flags through reload. Exact outer/inner handlers, controller server anchors and missing/ambiguous call gates pass; scoped Ruff/Pyright and diff checks pass. No declaration is added. Native-only coverage, summary meaning and whole-task acceptance remain open. Evidence: `reward-candidate/external-output-checkpoint.md`.
- [ ] (2026-10-04) Next implement manifest-bound conditional-summary assessment and qualify native call provenance for Qwen traces. Sampled-call/harness links exist, but MCP IDs are independent and retries may multiply effects. Consult critic/efficiency on coverage-only reconciliation versus a real host-owned parent execution/attempt link; never invent historical links. Use actual bundled Null-harness/MCP tests. Keep optional summary semantics separate from capture and deterministic read/update signals; register shared checks and exact public policy/context bindings before task acceptance.

- [x] (2026-10-04) Qualify the shared consumed-revision ledger and factual assistant-output projection after critic/efficiency consultation. Final root combined gate passes 146 cases in 7.48 seconds; critic independently repeats 85 output/source cases including exact retained SDK bytes and its false-closure reproducers. Intermediate broader regression passes 1,363 in 62.78 seconds before the final pure-adapter repairs; final targeted gates cover those repairs. Scoped Ruff/Pyright and diff checks pass. No native public API, product-baseline amendment, catalog addition or publication is implied. Exact hashes and evidence: `reward-candidate/manifest-ledger-and-authored-output-checkpoint.md`.

- [x] (2026-10-04) Install reviewed Contact assistant revision `public_contact_read_v2` after exact draft-byte comparison, original Luna read/update credit, seventeen native alternatives and independent critic repetition. Post-install focused gate passes 196 in 6.98 seconds; broad manifest/core/retained/public regression passes 1,230 in 49.27 seconds. Scoped Ruff/Pyright and both repository diff checks are clean. Existing component count remains thirteen, zero whole tasks qualified. Preserve the conditional-summary projection gap, legacy unused Contact timestamp hydration drift and other credit policies' mixed-revision migration audit as open work.
- [ ] (2026-10-04) Next qualify the conditional-summary guard across assistant text and externally sent messages/free-form records. The raw SDK artifact exists and the prepared source now projects its exact bytes; factual extraction is being qualified separately from policy meaning. Mixed-revision consumption now has a shared all-six-family boundary. Do not treat missing capture as silence/compliance or require an unrequested summary. Consult critic and efficiency, ground names in verified action relations and values, then revisit whole-task qualification and the ten-task acceptance queue.

- [x] (2026-10-04) Implement shared Gmail returned-field observations and root manifest/native obligation wiring. Adapter gate passes 26 cases, root schema/obligation gate passes 136, combined core/native transport regression passes 225 in 29.48 seconds; scoped lint/typing are clean. Fix generic once-only mixed-revision remint and retain valid partial credit across genuine failure/cancellation. Public Contact draft/native qualification remains pending; thirteen bounded components and zero whole tasks remain installed. Evidence: `reward-candidate/gmail-read-checkpoint.md`.

- [ ] (2026-10-04) Implement and qualify shared original-message read evidence in `contracts/gmail_observations.py`, then extend the reviewed Contact assistant manifest with its explicit Find obligation. Consult efficiency and critic before implementation. Reuse occurrence/native transport; test find/get/API alternatives, ID/snippet-only returns, strict original identity/body, missing receipts and reload/rescore. Full returned content proves successful tool retrieval, not exact model conditioning. Audit applicable public instructions before claiming whole-task coverage; do not invent a mandatory summary or extra notification.

- [x] (2026-10-04) Install Zendesk credited revision `public_batch10_credit_v2` after its data-only draft exactly matches the actual-replay tested contract. Broad manifest/core/public regression passes 1,161 cases in 45.10 seconds; root post-install gate passes 217 in 8.37 seconds, with scoped lint/typing clean. The ledger remains thirteen bounded declarations and zero whole tasks. Full outcome-only review evidence is preserved as history; current original-update credit and lifecycle qualification are recorded in `reward-candidate/record-completion-credit-checkpoint.md`.

- [ ] (2026-10-04) Next complete-task qualification lane: audit public obligations of the frozen Contact assistant/account tasks, then add shared Gmail read-observation evidence if the explicit “Find the email” process clause needs it. Efficiency's source audit finds no mandatory summary, additional notification or blanket unrelated-action prohibition in those tasks; do not invent those as completion gates. Both requested Contact terminal fields are already covered. The assistant's recorded run genuinely reads the requested full message and updates both fields; critic independently checks remaining public obligations before implementing the read adapter.

- [x] (2026-10-04) Implement the shared Zendesk status-update evidence and earliest observed record-completion core/native policy. Adapter qualification passes 26 cases, root completion/schema gate passes 124 before the final added tie case (now 19 core cases pass), and native source/outcome gate passes 65. Critic reproduced a valid-yield-then-cancellation duplicate contribution; repaired consumption now includes valid partial/interrupted/failed histories. Real cancellation and failure tests plus independent repeats verify no remint through rescore/reload. The actual SHA-bound Luna gate passes 32 native-credit cases and gives the original ticket update `a19b0f39ad6f4d81ae756b8b93d2daa0` one positive contribution, with unchanged bytes/scalars and incomplete unrelated-action coverage preserved. Public credited declaration/catalog upgrade and broad regression remain pending.

- [x] (2026-10-04) Qualify and install the public Zendesk same-original-ticket status outcome. Shared core/population gate passes 100 cases; independent critic repeats 82 core/public alternatives and verifies authority/source hashes. Astra's full 34-case gate includes exact recorded Luna score/rescore/reload, preserved bytes/scalars and zero action credit. Broader `test_manifest_*.py` regression passes 1,043 cases in 39.17 seconds. Catalog and batch ledger now contain thirteen bounded development components, zero whole tasks qualified. Root post-install catalog/full-Zendesk gate passes 133 cases in 2.93 seconds; mutation credit and resolution email remain open.

- [ ] (2026-10-04) Next useful-action increment: add the exact Zendesk status-update adapter and separate `records_retained_completion_once@1` policy. Efficiency consultation recommends adapter/core parallel ownership and requires explicit earliest acknowledged completion, initially false same-ID baseline, current terminal success and requested/changed goal-field overlap. Keep initial/final goal projections identical in this increment, and reuse native raw recapture/current-parent/once-consumption without changing existing Sheets selection semantics. Qualify real handlers, damage/repair, initially correct/noop, missing ACK, malformed baseline and tied revisions before enabling any contribution.
- [ ] (2026-10-04) Execute the retained-record increment in three coordinated lanes: efficiency agent owns `contracts/retained_records.py` and core tests; root owns contract schema/compiler and `manifest_record_retained_assessments.py` dispatcher integration; Astra owns the public Zendesk component review/draft/native alternatives. Critic independently checks identity, terminal evidence and strict field admission. Public Zendesk review confirms solved-ticket state and a separate meaningful resolution email; no action order or extra priority/comment requirement is invented. No new manifest is installed before the combined gate.

- [x] (2026-10-04) Wire outcome-only typed record assessments into the native dispatcher and register strict manifest source/check references. Existing contract/Sheets-retention regression passes 119 cases; the expanded schema and Zendesk small-native alternatives pass 129 with one recorded replay deselected. Root files pass scoped Ruff/Pyright. Static admission rejects undeclared projections, invalid scalar tails, unknown candidate metadata, cross-collection joins and unsupported completion credit. Shared strict field admission and the original Zendesk replay remain pending; no new declaration is installed.

- [ ] (2026-10-04) Next useful-signal increment: generic typed retained-record outcomes, then one qualified Zendesk status-write family and public solved-ticket component. Architecture now names `contracts/retained_records.py`, `manifest_record_retained_assessments.py`, schema/dispatcher integration and separate `contracts/zendesk_effects.py`. First prove terminal same-initial-ID outcomes without causal credit; then qualify acknowledged false-to-true completion before enabling an action contribution. Public message-purpose requirements, other services and whole-task coverage remain distinct open work.

- [x] (2026-10-04) Install independently reviewed excluded-dashboard and protected-mention Sheets guards. Both full native/core suites pass 37 cases in 12.53 seconds, including actual Zoho source/scalar/reload and native harm recipients. Packaged contracts match drafts; catalog/core checks pass 127 cases with three already-qualified native cases deselected. Historical Zoho escalation remains unavailable to the Sheets PR-ticket guard. Ledger now reconciles twelve bounded declarations and zero whole tasks qualified.

- [x] (2026-10-04) Broader manifest regression passes all 988 cases in 37.96 seconds after native proof reuse, strict coordinate admission and structural population expansion. This covers every `tests/test_manifest_*.py` file; installed NDA/Jira/content task-specific replays are covered separately by the 190-case integrated gate and exact content measurement. The fifteen-per-category whole-task coverage gate, budget acceptance and later publication/student campaign remain open.

- [x] (2026-10-04) Final integrated installed-manifest source check passes 190 cases in 12.61 seconds across catalog/schema, expanded populations, NDA, Jira, content authority, Slack and native requests. The critic confirms seven frozen native file hashes and the 153-test JUnit digest. Record the performance evidence boundary: baseline session 8715 did not archive dirty-source hashes or peak RSS, so 195.00→55.91 seconds is a local matched-input measurement, not immutable before/after build qualification.

- [x] (2026-10-04) Qualify execution-owned intrinsic proof reuse. Native suite passes 153 cases without failures/skips, independent critic passes 30 focused intrinsic/archive-prefix cases, scoped Ruff/Pyright pass. The exact recorded content score/rescore/reload test passes in 55.91 seconds versus 195.00 seconds before (3.49 times faster); wall time 56.77 seconds, maximum RSS 576,748 KiB. Original episode SHA/scalar/negative action credit and repeated assignments remain checked. This is one local replay result, not training throughput or a memory-reduction claim. Archive loading stays fully validated. Native source hashes and measurement boundary: `reward-candidate/native-intrinsic-proof-qualification.json`.
- [x] (2026-10-04) Independent critic repeats all 51 structural population cases, with no hydration/alias/schema/initial-membership blocker. Zoho's 14 core cases and Board PR-ticket's 20 core cases pass; native candidate source is now released for their remaining action-credit/reload gates. Root integrated installed catalog/population/NDA/Jira/Slack/request suite is live in terminal session 62658.

- [x] (2026-10-04) Replace the two-collection initial-population allowlist with structural admission from the installed AutomationBench `WorldState` schema. Only direct `list[BaseModel]` collections with canonical `id: str` qualify. Fifty-one source/population cases pass, including eight newly admitted services/collections, immutable public Contact input, schema drift, missing/empty distinction and explicit raw identity; scoped Ruff/Pyright pass. This expands context/lookups, not service effects, table-only guards or created-object populations. Critic independent final review remains open.
- [ ] (2026-10-04) Qualify private native validation proof reuse after code release: exact successful intrinsic SourceSnapshot/ObservationView validation scoped to trusted execution owners, strict Python coordinate admission, bounded memory and task isolation. Cold/warm forgery, eviction, cancellation and current-plan checks precede matched recorded replay against the 195.00-second content baseline. Archive loading remains uncached in this increment. Hold Zoho/PR-ticket native replay until native source release; their manifest authoring/core checks proceed independently.

- [x] (2026-10-04) Install the independently reviewed NDA Signed-status preservation guard revision `public_batch06_native_identity_v2`; packaged bytes match draft SHA `e87867b5011cea00c905976966039724f3410a36e5d7a5c84d68ea2b62dd2588`. Final 16-case suite passes in 30.47 seconds, including unchanged actual Luna score/rescore/reload and semantic equivalence of fresh run waves. Guard follows scoped native identity rather than row position; repair cannot erase earlier harm. Ledger now has ten bounded development declarations and zero whole tasks qualified. Efficiency owns native proof-scope implementation; Zoho guard authoring proceeds without native replay until source release.

- [x] (2026-10-04) Install retired-content queue guard revision `public_batch08_guard_v2` after full native recorded score/rescore/reload passes in 195.00 seconds, independent critic's one-pass actual replay and seven real-handler/authority alternatives. Combined catalog/authority suite passes 100 cases; scoped Ruff/Pyright pass. Ledger now reconciles nine installed components and zero whole tasks. NDA's neutral actual first score passes; its test incorrectly demanded unchanged assessment history on rescore, so lifecycle semantics are being independently checked before repairing and rerunning the gate.

- [x] (2026-10-04) Native recorded retired-content guard replay passes in 195.20 seconds: exact prohibited append recipient, one negative contribution, unchanged scalar reward/episode bytes and preserved rescore/reload assignments. This first run used the historical batch-eight proposal. Root now tests the revised whole-policy-bound draft in live terminal session 8715; do not install it based only on the prior contract. Three revised authority-inventory mutation checks already pass. Astra owns the next NDA guard draft/native qualification, critic reviews policy/admission, efficiency audits safe performance options read-only.

- [x] (2026-10-04) Install the independently reviewed Accessibility Audit retained-issue component from batch nine. Packaged contract matches the public draft, seven actual/current-handler alternatives pass, and combined catalog/schema/public-Jira suite passes 100 cases. Ledger now has eight installed development components and zero whole tasks qualified. This is local candidate source, not published dependency readiness.

- [x] (2026-10-04) Integrate shared fresh-created/retained object assessments and once-only completion credit into `ManifestAssessmentTask`. Independent critic reproduced 73 passing Jira/core/native cases. The public Accessibility Audit draft passes all seven recorded/current-handler alternatives; the original action-log-only episode receives outcome zero and no new credit, without changing its bytes or scalar reward. Draft remains uninstalled pending final catalog acceptance; seven installed components and zero whole tasks remain the ledger state.
- [x] (2026-10-04) Recognize the installed `google_sheets_append_row` alias with its actual argument precedence, retaining receipt/native-row validation. All 64 adapter cases pass, including hash-bound recorded Zoho and content trajectories. Their five and three appends are now observable; unsupported actions still prevent claims of complete guard coverage. Native retired-content replay is running; its stronger declaration binds whole Gmail/Slack authority inventories before admission.
- [x] (2026-10-04) Repair unavailable public-request population canonical JSON reload. Sorted field inventory survives serialization while missing/extra fields still reject. Request/native-created/public-Jira combined qualification passes 69 cases. Preserve explicit initial Jira identity membership as partial evidence when hydration has conflicting captures; do not infer missing object fields or freshness.

- [x] (2026-10-04) Implement the shared source-bound request population adapter: 39 focused tests and 72 request/population tests pass, with scoped lint and typing clean. Native obligation/schema integration remains open.
- [x] (2026-10-04) Profile actual prepaid completion replay: one recorded-episode test passes, 35.22 seconds profiled wall time and 401,528 KiB peak RSS. Repeated native source/view validation dominates; domain completion evaluation is 0.368 cumulative seconds. Preserve exact-wire/type-sensitive executor-local reuse as a reviewed candidate, not an implemented speedup. See `reward-candidate/completion-validation-profile.md`.
- [ ] Continue authoring creation manifests in the existing ten-task packs using the now-qualified request integration; keep unsupported service/content obligations explicit.
- [x] (2026-10-04) Integrate authored requests into schema/native obligations. Worker 238 focused tests and independent critic 31 native cases pass; a coherent retarget to an actual unrelated effect/recipient is rejected. Original scalar, reload and once-only credit remain preserved.
- [x] (2026-10-04) Install the reviewed prepaid schedule-and-guard declaration: 38 schedule/guard tests pass, including actual saved Luna replay and added competing-policy cases. Journal, aggregate and reporting obligations remain open.
- [x] (2026-10-04) Install the exact-name/workspace Asana creation-occurrence component from batch seven. Seven root tests and two independent critic tests pass, including actual Luna execution recipient, same-effect conjunction, missing ACK and reload/rescore. Seven development components are now cataloged; whole-task qualification remains zero. Slack announcement remains open.
- [ ] Add Slack channel-message/DM occurrence evidence as the next shared breadth capability; root integrates schema/dispatch/native fixtures after standalone adapter review. Delivery alone cannot establish required message content.
- [x] (2026-10-04) Add and independently review Slack occurrence evidence. Forty-nine adapter cases include two actual SHA-bound Luna traces and raised/error capture rejection. Root registers sources in guard/obligation transport; all six native delivery/harm/reload cases now pass. The wider six-file run passed 199 with one fixture-membership failure; correcting its missing worksheet inventory yields the six-case pass, not a claim that the original wider command was green. Scoped lint/type checks pass. Production free-form content semantics remain open.
- [ ] Complete Jira issue reward evidence and native qualification after the simulator/vendor repair. Historical action-only traces must remain unchanged.
- [x] (2026-10-04) Repair Jira create/read/status-update state in the sibling: 195 tests pass, including 51 new Jira cases. Independent critic found and verified five atomic-admission repairs. Root refreshed exactly three hash-recorded vendor files and forced loaded-vendor imports; all 51 Jira cases pass. Evidence: `reward-candidate/jira-vendor-refresh.json`.
- [ ] Add shared Jira transition/final-inventory evidence and a retained-new-object check with separate completion selection, then qualify the source-bound Accessibility Audit declaration. Existing matching issues and historical audit-only logs cannot satisfy fresh creation.

- [x] (2026-10-04) Finish all eleven public-input review packs: 105 development tasks, fifteen per category, reviewed in batches of ten plus the last five. Root independently verified batch ten and eleven hashes/bindings. Six bounded components are cataloged locally, including two requested Contact state goals; zero whole tasks qualified. This closes public review, not the fifteen-per-category native qualification gate.
- [x] (2026-10-04) Replay the prepaid schedule draft against actual saved Luna evidence: Insurance and Hosting each receive one exact completing-action contribution; Software remains required but fails and gets none. Preserve original bytes/scalars and assignments after immediate rescore and serialized reload. Qualify optional initial-context supported_when so unknown accounting interpretations keep required obligations visible without numeric score or action credit. Draft v2 passes actual replay and alternatives; no journal or whole-task claim.
- [x] (2026-10-04) Add shared Contact record capability and two full-prompt/public-population-bound state-goal declarations. Saved Luna assistant goal passes; Account rename fails the Contact company-change goal; valid native Contact alternatives pass both. Independent critic reproductions exposed and then verified repairs for raw numeric coercion and moved original-row harm. Broad manifest run: 932 passed, three historical direct-context tests failed; all eight boundary cases now pass through the actual authenticated executor. Scoped lint/type checks pass.
- [ ] Complete supported-domain prepaid declaration review, install its schedule component only after acceptance, and profile meaningful completion/retention workload before claiming dense performance. Shared public-request population and actual Jira issue-state semantics are the next authoring bottlenecks. Maintain the full library coverage and later campaign gates.
- [x] (2026-10-04) Qualify the local retained-completion credit implementation with the combined manifest/core/native regression suite: 348 passed, one dense guard replay excluded (separately measured previously). Changed completion files pass Ruff. Stable once-only consumption, authenticated-source recomputation and cross-planner overlap rejection are covered. Actual recorded positive-action replay and whole-task declarations remain open.
- [x] (2026-10-04) Independently verify batch nine's immutable pack, ten public-input hashes and eight blueprint source bindings. Ninety tasks reviewed; four accepted bounded components and zero whole tasks qualified. Batch ten authoring is active. Batch nine needs shared service adapters and contributes no executable manifest.
- [x] (2026-10-04) Independently verify batch eight's ten public hashes, retired-content queue guard schema and fifteen bindings. Eighty sample tasks reviewed; accepted components remain four and whole-task qualification zero. Batch nine's public review is active. Completion-action credit ownership/implementation recommendation is reviewed with efficiency and critic agents.

- [x] (2026-10-04) Independently verify batch seven's ten public hashes, excluded-account dashboard guard schema and seven bindings. Seventy sample tasks reviewed; accepted components remain four and whole-task qualification zero. Batch eight's public review is active. See `reward-candidate/manifest-authoring-batch-07-review.{json,md}`.

- [x] (2026-10-04) Repair coherent input-source substitution across record/guard/occurrence/retained publishers. Native `AssessmentContext.retrospective_source()` anchors the executor snapshot without permitting prefix/future access. All publishers authenticate raw projected input before evaluation/result-cache lookup, and an immutable-wire bounded cache preserves fresh working copies. Root final integrated suite passes 281 tests, one dense case excluded; dense replay passes separately in 75.07 seconds with 470,188 KiB peak RSS. Critic independently verifies four substitutions, four cache boundaries and native visibility; scoped source lint/typing pass. Broader scale/native release and positive completion-credit gates remain open.
- [x] (2026-10-04) Independently verify batch six: ten public hashes, NDA Signed-status preservation guard schema and four bindings pass. Sixty tasks reviewed, four bounded components installed, zero whole tasks newly qualified. Batch seven public review is active. See `reward-candidate/manifest-authoring-batch-06-review.{json,md}`.

- [x] (2026-10-04) Implement a distinct terminal-row outcome check, `sheets.retained_when@1`, following original native identity and rederiving both projections from raw source. Correct-then-damaged writes fail, repairs can restore outcomes, replacement rows cannot steal obligations, and missing terminal evidence abstains. Retained/schema admission passes 119 tests; the earlier combined integration suite passes 195. Scoped lint/typing and seven independent critic reproductions pass. Evidence: `reward-candidate/retained-row-checkpoint.md`.
- [x] (2026-10-04) Qualify native retained-outcome publication through the original SHA-bound prepaid Luna replay, sealed per-member inputs/receipts, current complete runs, archive/reload and zero-credit rescoring. Root integrated suite passes 222 tests in 17.35 seconds with scoped Ruff clean. This is development-only ineligible-balance preservation, not full amortization qualification or a new installed task manifest.
- [ ] (2026-10-04) Add separately reviewed completion-action credit using `reward-candidate/retained-completion-credit-proposal.md`. Outcome checks alone cannot allocate positive contributions. Root owns shared semantics/schema; efficiency owns native planning/projector, critic owns independent qualification. Batch nine's ten-task review is active.
- [x] (2026-10-04) Complete and independently verify batch five's ten public-input hashes and source-bound CRM-stage proposal. Fifty tasks are reviewed, four bounded components installed, zero whole tasks newly qualified. The new proposal has thirteen matching public bindings but needs native baseline qualification because public `stage` and native `stage_name` differ. See `reward-candidate/manifest-authoring-batch-05-review.{json,md}`.

- [x] (2026-10-04) Independent critic completes the initial identity, audited operation-scope and calendar boundary review: 27 focused cases plus foreign-origin, missing-ACK, unknown-operation and leap-century reproductions pass. Native world-transition ACK already enforces tool-server origin. Preserve the installed-handler trust boundary and unknown/custom operation gaps. Evidence and exact commands: `reward-candidate/prepaid-native-guard-checkpoint.md`.

- [x] (2026-10-04) Install the source-bound prepaid ineligible-recognition guard with original native-ID matching. Twelve native cases pass, including actual retained Luna replay with closed declared compliance, no invented harm, unchanged scalar rewards, reload/rescore and unchanged source bytes. Simulator alternatives verify harm, metadata corrections, noops, repairs, missing ACKs and replacement at the same row position. Combined native/schema/identity/Sheets suite passes 223 cases with scoped lint/typing clean. Catalog now has fourteen entries, including four bounded components from the sample; no new whole tasks are qualified.
- [x] (2026-10-04) Add explicit calendar-month derivations with a mandatory invalid-day policy (unavailable or clamp), integer-month and year-range checks, and declared interval endpoints. Eleven additional edge/coverage regressions contribute to 154 passing value/predicate/obligation cases. This is calendar arithmetic, not an inferred fiscal/business calendar or approval of a task's coverage policy. Retained-row positive checks and source-bound singleton/paired effects remain the next substantive capability work.

- [x] (2026-10-04) Bind original Sheets native record IDs from acknowledged revision-zero before snapshots while preserving the public initial fixture. Fifty-five table cases and 101 combined focused cases pass; independent critic runs eight additional boundary mutations and four replay/replacement/tamper cases successfully. The actual prepaid trace binds all five rows. Complete native prepaid guard qualification is active: prohibited/allowed/repair/ACK cases pass, while recorded compliance exposed an additional audited operation-scope and raw-versus-hydrated initial reconciliation gap being repaired before catalog admission.
- [x] (2026-10-04) Complete batch four's ten public reviews. Forty tasks are reviewed. A BoardMemberBlog PR-ticket prohibition has schema validation, eight public bindings and six simulator checks; native qualification/catalog admission remain open. The review identifies source-bound singleton requests and generated-ID joins as the next cheap shared primitive for complete Simple workflows.

- [x] (2026-10-04) Re-run the dense SHA-bound Luna access replay after exact raw guard-input recapture. It passes in 71.14 seconds pytest / 71.96 seconds process elapsed with 469,436 KiB peak RSS. This measured slice remains comparable to the earlier 71.44-second / 468,672-KiB source/view-pooling checkpoint; broader CPU qualification is still open. The prepaid guard proposal has seventeen valid public bindings and twelve simulator mutation checks, but is not installed: critic demonstrated that reusing a deleted row's number can falsely match its old policy. Native record identity binding is the next shared evidence repair.

- [x] (2026-10-04) Register shared Sheets append/update effects in native guard/obligation paths. Preserve factual noop occurrences while withholding progress credit; accept uniquely inferable omitted sheet/tab targets. Revalidate retained guard primitives against raw source once per view and avoid duplicate guard/obligation source inventories. Independent critic confirms coherent forged table/effect labels reject. Add reserved typed `candidate.identity` so guards match native rows even after cells are renamed; alias shadowing rejects. The combined Sheets/guard/obligation/schema suite passes 206 cases with scoped lint/typing clean. Dense retained Luna replay is running to measure the added source-validation cost.

- [x] (2026-10-04) Complete the third ten-task public review, recording source hashes, concrete deferred declarations and shared service/policy gaps without inventing unrelated partial rewards. Thirty sample tasks are now reviewed. The exact-value/derived-predicate and batch-two replay suite passes 322 cases; independent critic reproductions confirm strict derived-field admission repairs. Evidence: `reward-candidate/exact-values-and-batch02-checkpoint.md` and `manifest-authoring-batch-03-review.{json,md}`. Complete redesign remains open.

- [x] (2026-10-04) Review batch two's ten public inputs and install its source-bound negotiated CRM amount component. Actual retained Luna replay preserves scalar rewards, native findings, exact execution credit, reload and once-only rescoring. The catalog now has thirteen entries, including three bounded components from the 105-task sample; whole-task qualification remains zero. DocuSign void/resend/template/confirmation obligations remain open. Exact arithmetic/date derivations and explicitly typed predicate operands pass focused tests; critic corrections reject malformed timestamp offsets and enforce derived-field context/admission boundaries. Qualification commands and final counts are recorded in the next capability checkpoint.

- [x] (2026-10-04) Complete public-policy review of the first ten-task batch and install two bounded components. Seven actual/adversarial cases pass: sales Closed-Won receives exact verified transition credit, customer recipient delivery receives no credit until purpose is checked. Catalog has twelve entries; zero new whole tasks are qualified. All 105 transport cases pass again (two registered bounded, 103 unregistered); combined run passes 240 plus two separately corrected catalog fixtures. Native obligation transport passes twenty cases including critic-reproduced scope/retarget repairs. Evidence: `reward-candidate/manifest-positive-and-batch01-checkpoint.md`.
- [x] (2026-10-04) Implement and register the first two shared capability lanes: Sheets write/retention evidence and exact decimal/date derivations, with critic-reviewed fixes and focused native tests. Batch two public review is complete. Full first-batch goal/guard coverage and original-row identity binding remain open.

- [x] (2026-10-04) Require installed manifests to bind public prompt/initial-world authority; reject missing bindings and final-outcome authority. The schema/record engine/credit/authoring batch passes 179 cases with scoped Ruff clean. Standalone proposals remain loadable for review and labeled manufactured tests; catalog admission is stricter.
- [x] (2026-10-04) Close critic-reproduced native obligation receipt gaps: recompute scope coverage and bind the chosen witness so a valid request cannot be retargeted. Preserve stable consumption across witnesses/prefixes. Twenty source-stable cases pass; critic independently confirms both reproduced defects now reject.

- [x] (2026-10-04) Implement public-input authoring batches in environment-owned `calibration/manifest_authoring.py`. Nine tests cover frozen hashes, development-only membership, exact identities, hidden-answer exclusion and strict batch limits. Generate eleven batches for all 105 selected tasks with size limit ten and category round-robin order; first batch spans all seven domains. Durable index: `reward-candidate/manifest-authoring-batches.json`; full public packs remain under `.posttrain/state/verifiers-assessment-qualification/manifest-authoring-105/`. These are proposals, not accepted manifests.
- [x] (2026-10-04) Register initial-record populations and required-effect check/credit schema capabilities, including mismatched-policy rejection. Schema/record engine/credit tests pass 168 cases. Required-action dispatcher wiring preserves the single registered assessment hook; 115 existing native record/notification/schema tests pass. New positive native transport tests and independent review remain in progress.
- [x] (2026-10-04) Implement shared occurrence obligations with explicit reviewed `new_occurrence` semantics and safe baseline default. Forty operator cases and 124 combined obligation/population/predicate cases pass in the worker's qualification. Population strict copied-model re-admission fixes boolean position/status tampering; 33 population cases pass. No initial-false evidence is invented to enable useful-action credit.
- [x] (2026-10-04) Review the first ten public inputs, record proposed goals/guards and shared capability gaps, independently review and install two tested bounded declarations. Native required-action transport is qualified by twenty focused cases. No first-batch task is whole-task qualified; broader declarations remain separate work.

- [x] (2026-10-04) Fix critic-reproduced native source schema-version coercion at raw and copied-model boundaries. Nine new regressions pass; native scoring/trace/judge batch now passes 123 cases. Fresh Gmail/guard boundary/lifecycle batch passes 41 cases; Posttrain adapter/bridge/artifact batch passes 76 with eight native-extra skips, and all nine import contracts pass.
- [x] (2026-10-04) Complete shared initial-population and required-occurrence capabilities, registration and native publishing. Later entries record the 33 population, forty operator and twenty native transport cases, plus independent receipt-boundary repairs. This closes the earlier parallel implementation increment; complete task policy coverage remains open.

- [x] (2026-10-04) Pool exact native sources and inline views per trace/episode without reducing assessor context. Native scoring/trace/judge regressions pass 114 cases. The same dense Luna replay passes in 71.44 seconds with 468,672 KiB peak RSS, compared with approximately 7.5 GB before normalization; repeated validation CPU remains open. Qualify shared Gmail send evidence and native guard registration: 21 adapter and ten native notification cases pass, with 415 combined fast environment regressions plus the separate dense replay. Training export retains compact pool identities and full inputs only in native artifacts. See `reward-candidate/native-archive-and-notification-checkpoint.md`.
- [ ] (2026-10-04) Turn the fixed 15-per-category sample into semantic reward tests. Implement shared required-effect obligations, author reviewed declarations from public task policy, and test actual findings, uncertainty and action credit. Transport-only or uniformly unavailable results do not satisfy this gate; complete task reward coverage remains separate from a bounded check's acceptance.

- [x] (2026-10-04) Replay the frozen cross-category source/transport sample: 105 actual development task probes pass, covering 1,727 native execution occurrences; combined with eight boundary regressions, 113 pass in 53.64 seconds. All 105 traces are tokenless and all selected tasks currently lack installed reward manifests. This closes raw transport breadth only; reviewed findings, credit and full-task coverage remain open. Results: `reward-candidate/cross-category-api-probe-results.json`.
- [x] (2026-10-04) Qualify native guard publication and negative credit: 384 focused tests plus one full-fidelity recorded Luna replay pass; scoped Ruff/Pyright/diff checks pass. One manifest assessment dispatcher handles record/guard checks, exact subjects and current output receipts are validated, and failed/interrupted assessor attempts cannot authorize partial credit. Installed catalog remains ten Simple tasks. See `reward-candidate/manifest-native-guard-checkpoint.md`.
- [x] (2026-10-04) Freeze the user's breadth sample: 105 development tasks, 15 in each of seven domains, spanning 46 recorded families and 195 native operation labels; all source hashes verified and no reserved payloads used. Selection and capability ledger: `reward-candidate/cross-category-selection.json` and `cross-category-api-gap-ledger.md`. Selection and transport tests do not establish reviewed manifest reward coverage.
- [ ] (2026-10-04) Complete dense scale qualification after the source/view pooling fix. The six-row/fourteen-effect recorded guard slice generates 85 instance/scope runs; exact archive normalization reduces peak RSS to 468,672 KiB, while replay still takes 71.44 seconds. Preserve independent findings and recovery; improve repeated validation without suppressing unknowns or aggregating away distinctions.
- [x] (2026-10-04) Implement the shared conditional guard components inside AutomationBench; 345 combined tests pass with scoped lint/typing clean. Adapters expose finite rows and acknowledged effects; manifest expressions own policy and exact matching forms. Retained Luna access supplies six action witnesses, while simulator cases qualify harm persistence, negative selection and unique action-to-candidate matching. See `docs/research/verifiers-assessment-qualification/reward-candidate/manifest-guard-components.md`.
- [x] (2026-10-04) Publish component findings through native manifest guard assessment/credit hooks and qualify the bounded processed-row access slice. Existing schema is extended and scalar/record regressions pass. Full-task access declarations and broader catalog expansion remain open; bounded guard acceptance does not establish whole-task reward redesign or eligibility.

- [x] (2026-10-04) Complete the focused manifest record-update checkpoint: 215 tests pass with ten original Luna replay cases, scalar preservation, native persistence, no duplicate credit, explicit joint attribution, and adversarial evidence/credit boundaries. Scoped lint/typing pass. See `docs/research/verifiers-assessment-qualification/reward-candidate/manifest-record-checkpoint.md`. Conditional guards, full-task coverage, efficiency, consumer qualification, publication, Qwen benchmarks and curricula remain open.

- [x] (2026-10-04) Implement and qualify the first manifest-driven record-update checkpoint. Root owns native bridge/credit/taskset selection; the schema agent owns manifests and qualification tests; the independent critic owns source-bound evidence and deterministic admission fixes. Architecture records actual module responsibilities, per-check runs, retained output receipts, explicit joint credit, and generic input-revision bindings. Conditional guards and multi-step obligations remain subsequent work; do not extend bespoke task evaluators.

- [x] (2026-10-04) Execute the first AutomationBench-only manifest increment after architecture review: validated packaged task data, reusable record/source checks, evidence-aware results, independent credit selection and neutral native publishing. Efficiency advised allocation; critic reviewed the shared native subject/signal seam, resolved with one run per check. No cross-environment engine, new collection or source-sensitive eligibility resealing during edits.

- [x] (2026-10-03) Fold the latest feedback and actual progress into the architecture before implementation. The engine is AutomationBench-only, inside `automationbench_v1/contracts/`, with a small schema/operators, evidence-aware results and separate credit rules. The architecture names proposed modules, native hook flow, example task data, migration increments and planned test commands. Other-environment proof and a new Verifiers contracts package are removed from this engine's requirements. Existing native contracts and later campaign requirements are preserved; no manifest-engine implementation is claimed.

- [x] (2026-10-03) Draft the manifest architecture from Astra's source review and the user's no-task-specific-code requirement. `docs/research/verifiers-manifest-assessment-architecture.md` includes a requirement-preservation matrix covering assessor freedom, hybrid checks, temporal guards, partial credit, multi-recipient assignment, original-token alignment, tracking, replay and the unchanged downstream campaign. This is a design artifact; engine implementation and capability parity remain unproven.

- [ ] Qualify every row of the manifest architecture's preservation matrix before replacing the existing authoring path. A smaller pilot does not remove earlier requirements. Record an explicit migration gap for unsupported policy/population evolution, input preparation or attribution rather than silently approximating it. Correct known broken old checks instead of treating their outputs as mandatory parity.

- [ ] (2026-10-03) Take the user's requested architecture review before extending task-specific authoring. Astra is reviewing the actual native API and AutomationBench candidate read-only: determine stable shared evidence/check/credit primitives, environment adapters and task declarations; compare manifest-selected registered strategies against Python-only authoring; challenge exact-fixture overfitting. Domain workers were interrupted by user steering and their partial changes are not accepted implementations. No unified manifest or new shared API will be adopted solely from the earlier recommendation. Record Astra's alternatives and qualification requirements before committing to that direction.

- [ ] (2026-10-03) Repair independently reproduced HR delivery misclassification: a native draft carrying a SENT label was treated as a delivered offboarding notification. Require shared qualified operation/result/persisted-effect evidence instead of label-only delivery inference. Critic owns the HR adapter and regression tests; root will verify native compatibility. Earlier HR test results do not close this newly demonstrated attribution gap.

- [ ] (2026-10-03) Extend conditional coverage with conversion tracking and weighted lead selection. Compile authority and frozen candidate populations from public emails/sheets/CRM facts, then qualify selective action goals and persistent harm using original failed traces plus fresh simulator alternatives. The conversion trace queried Salesforce although its authoritative deals were in Gmail; this does not establish an environment defect. Existing Lead models retain the fields required for weighted selection. Reproduce any capability gap before changing tool behavior. Efficiency agent owns conversion predicates/effects; native integration and independent critique remain required.

- [x] (2026-10-03) Factor identical marketing/finance transport and credit rederivation into `ReviewedEffectTask`, leaving policy and recipient meaning in domain evaluators. Twenty-seven shared marketing/finance/HR regressions pass; Ruff clean. Integrate renewal through this same publisher with explicit taskset opt-in: 24 renewal/helper/publisher tests pass, including actual Luna official-score parity, native reload, three agreement plus one procurement delivery contributions and repeated-scoring no-op. Summary correctness remains abstained; no whole-task eligibility claim.

- [x] (2026-10-03) Repair renewal history/population ambiguity using explicit reviewed period membership for initial envelopes and frozen initial public eligibility. Unknown historical envelopes grant no invented progress; live source drift suppresses accomplishment attribution while preserving independently observed forbidden sends. Twenty-two predicate/helper tests pass before publisher integration. Named history/source diagnostics remain neutral. Evidence: `conditional-slices-qualification.md`; source revision resealing remains pending.

- [x] (2026-10-03) Implement and integrate ten exact Simple Salesforce Opportunity contracts through one record-update strategy. All ten source-hashed Luna episodes pass native score/reload, preserve official scores, produce one qualified action contribution and avoid duplicate credit on rescoring. Baseline-correct states, break/restore, invalid acknowledgements and mismatched starting stages are independently tested. The combined record/access/operations batch passes 96 tests in 23.03 seconds; checked shared and domain publishers pass typing. Broader CRM schemas and full family coverage remain open.

- [x] (2026-10-03) Finish the cash-flow critic repairs and coordinated integration batch: 28 finance predicate cases plus native publisher/support/marketing/DocuSign/renewal cases pass all 57 tests in 21.30 seconds. Critic independently confirms fictional-title abstention, initial draft rejection and preserved excluded-name harm despite unrelated numeric source drift. Cash-flow publisher and DocuSign helper pass focused Pyright/Ruff; no new model calls. Evidence: `reward-candidate/conditional-slices-qualification.md`. Arbitrary prose and broader family acceptance remain outside this bounded result.

- [x] (2026-10-03) Save a source-backed coverage inventory for all 46 family buckets and 53 selected Astra contracts, joined to verified original episode hashes. Nine selected bounded cases plus three HR slices touch ten families; no family is complete. Historical Simple eligibility proofs need current-source resealing. Evidence: `reward-candidate/reward-coverage-current.{md,json}`. Use the matrix and `all-family-priorities.md` to batch broader implementation; do not reduce final scope to tested cases.

- [x] (2026-10-03) Integrate access-request validation using acknowledged Asana creation/section effects and public email/department/manager joins. Eight bounded goals on the actual Luna episode pass native score/reload with unchanged official scoring; processing writes remain diagnostics. Twenty-seven operations/access/Asana tests pass. Independent criticism confirms cancellation abstention, wrong-manager ambiguity and supported wrongful-denial harm. Renewal is also integrated with explicit historical-period and frozen population rules. Whole-task semantic gaps, whole-family coverage and downstream gates remain open.

- [x] (2026-10-03) Independently replay the renewal predicate/helper batch: 17 tests pass. Public registry/policy compilation yields three supported vendor obligations from the retained noisy population; sent envelopes retain prefix evidence, and forbidden sends remain violations after voiding. Boundary ambiguity and unsupported notes abstain, while procurement delivery is distinct from unqualified summary semantics. These are bounded predicates ready for native integration, not a complete task eligibility proof.

- [x] (2026-10-03) Qualify current-attempt planning at both native trace and episode levels: all 75 native judge/trace tests pass, including compatibility, repeated-scoring no-op and failed-current assessment without stale-success fallback. Shared type is `CreditPlanningContext`; both `Task.plan_credit` and `Env.plan_credit` default to legacy delegation. No runtime-selected assessment or generic silent dedup is introduced. Use the existing environment interpreter and isolated pytest plugin path; an exploratory `uv` invocation created an empty original Verifiers environment pointer, with no installed native runtime found or source/artifact loss. Production dependencies and pins remain unchanged.

- [x] (2026-10-03) Qualify the shared DocuSign send extractor with five tests: actual SHA-bound Luna renewal trace has three acknowledged template sends; draft-to-send, later void, repeated send and missing acknowledgement are distinguished. The extractor preserves send-prefix evidence and never claims a signature. Ruff/Pyright/diff checks pass. Renewal public-policy predicates are now assigned independently using this helper.

- [x] (2026-10-03) Finish bounded cash-flow acceptance after independent critique. Native publisher/loader retains seven completed action contributions on the actual Luna report, unchanged official scores, native reload and repeated-scoring no-op. Unsupported-appendix harm erasure, historical forecast/draft false completion, fictional-title handling and independent exclusion evidence are repaired; the final coordinated batch passes 57 tests. This qualifies the bounded report forms and transport, not arbitrary semantics or whole-family eligibility.

- [x] (2026-10-03) Independently pass 43 support-publisher/marketing/notification tests after current-attempt planning integration. Bounded support publishing requires an explicit source-bound contract: absent or development-proposal contracts abstain without credit; reviewed bounded contracts do not establish whole-task refund/digest success. Seven publisher cases include actual source-hashed Gorgias/Hiver scorer and reload paths. Worker typing/cleanup qualification remains pending; whole-task public-policy gaps remain open.

- [ ] (2026-10-03) Implement the next substantive deterministic slice, `finance.cash_flow_forecast`, from its public arithmetic and source-hashed Luna report. Efficiency agent owns new finance predicate/tests; root owns later native integration. Reuse existing notification and effect evidence. Keep unsupported prose separate from deterministic amounts and recipients, and do not infer policy authority from hidden assertions.

- [x] (2026-10-03) Consume the new native credit-planning context in the reviewed environment publisher. Select only the explicitly current assessment run/attempt/invocation, preserve all retained history, and intentionally skip already valid assignments for the same source/rule/recipient/channel/signal/value. Conflicting prior values fail explicitly. Forty-two marketing/HR/Simple regressions pass; the actual marketing scorer can run twice and retain new assessments without adding any assignments or errors. Native API qualification remains with the critic; missing/failed current results must not fall back to earlier success.

- [x] (2026-10-03) Repair critic-reproduced self-created marketing progress: an initially archived permitted contact retains goal completion after unarchive/rearchive but receives no positive action recipient. Initially unsatisfied repeated archives count once. Marketing/notification suite now passes 36 tests. Shared publisher rejects ambiguous independent assessment runs and conflicting retained records; 42 marketing-publisher/HR/Simple regressions pass, Ruff and diff checks clean.

- [x] (2026-10-03) Resolve the generic planning gap: a failed current reassessment cannot fall back to older successful evidence and issue duplicate credit. Explicit current assessment-attempt and prior-assignment context now reaches trace and episode planning, preserving legacy compatibility and raw history; the reviewed producer selects current attempts intentionally. Native qualification passes 75 tests. This source change invalidates previously sealed candidate closure for future admission until rebuilt; the four historical proof results remain inspectable evidence only.

- [x] (2026-10-03) Finish source-stable calibration qualification: eligibility, frozen original verification, benchmark admission/recovery and explicit guard inventory pass all 31 tests in 202.56 seconds. Four actual Simple references pass original scorer replay, regenerated native `Task.score` findings and independently audited eligibility proofs under the original 16,384-token budget. Their reviewed guard inventories are explicitly empty for the exact task-specific requests, not for unreviewed environment-wide policies. Evidence: `reward-candidate/simple-eligibility-qualification.{md,json}`. These proofs are candidate evidence tied to their source revision; later source edits require rebuilding them before use.

- [x] (2026-10-03) Integrate the reviewed marketing suppression producer through the native scorer and reload path: two publisher tests pass, including the source-hashed actual Luna trace, unchanged official score and exactly three action contributions. The underlying marketing/notification checks pass 34 tests; bounded support checks pass 25 tests after purpose-negation and prefix-truth repairs. A broader domain/native integration batch passes 130 tests. Support native publishing, full-family policy coverage and whole-task qualification remain open.

- [ ] (2026-10-03) Continue broader reward implementation with support native publishing and independent review of self-created progress, cached attribution and suppression public authority. Consult critic and efficiency agents before the next slice. Do not freeze a curriculum around the four qualified Simple examples or launch Qwen before the wider migration/runtime gates.

- [x] (2026-10-03) Fix native optional credit planning: empty plans and subsets of registered rules are valid, unknown rule names still reject before dispatch. Keep HR recording diagnostics as assessments without neutral learning contributions. Fifty native tests and 44 HR/Simple/reassessment tests pass; scoped Ruff/diff checks pass. Critic's benchmark suite passes eight tests before final recovery fixes and three affected tests afterward. Full integrated qualification awaits source-stable domain producers; see `reward-candidate/native-reassessment-and-index.md`.

- [x] (2026-10-03) Add a concrete deterministic `Task.score` reassessment adapter and revision-aware ephemeral effect index in the isolated environment candidate. Five actual replay/async-bridge/artifact-preservation tests pass across all four Simple tasks; twelve index tests and twelve effect tests pass. The earlier combined batch passes 65 tests before the additional three replay cases. Scoped adapter Ruff and candidate-aware Pyright pass. Evidence: `reward-candidate/native-reassessment-and-index.md`. This qualifies the seams, not whole-task guards or Qwen eligibility. Critic's benchmark source-closure/status/lineage fixes are under independent tests; domain integration remains open.

- [x] (2026-10-03) Separate four Simple trace-level outcomes from independently derived execution credit. Neutral recording diagnostics, failed goals, already-correct states and missing attribution generate no action credit. Require accepted producer/rubric revisions and exact signal/subject contracts. Fix duplicate credit planning from native progress snapshots by selecting terminal runs and deduplicating assessment identities. The combined 51 tests and expanded 16 Simple tests pass, including four actual development traces and native reload. Whole-task guards and independent eligibility replay remain open; see `reward-candidate/simple-slice-qualification.md`.

- [ ] (2026-10-03) Correct independent critic findings before admitting any Qwen pool: schema deserialization must not execute replay; externally accepted verifier/reward identities must precede execution; producer facts must be derived from native evidence; and resumed summaries/statuses must match native stops/errors/usage. The worker is implementing the frozen-original verifier seam and these fixes. Efficiency review recommends explicit per-closure trusted validation, not repeated all-pool replay during parsing. Evidence: `qwen-benchmark-scaffold/critic-review.md`.

- [x] (2026-10-03) Finish the original 800-task collection with honest storage-loss accounting. The authorized fourteen never-started tasks completed; no consumed start was retried. Final remaining-700 verification checks 690 retained episodes/budgets, 671 archived source files and unchanged sources. Root independently rehashes all 700 development episodes and verifies unique development-only membership. Final index: `development-traces-index-700-with-verification-a97042bf9ce7.json`; full coverage: `development-coverage-final-9d0b173c29e7.json`. Two exact-state replay exceptions are Mailchimp set ordering with identical assertion results; one belongs to development. Original artifacts are unchanged. This supersedes historical running/storage-recovery entries below; ten lost artifacts remain unavailable, not behavior failures.

- [x] (2026-10-03) Astra completes the final development review basis: all 700 retained source hashes verified, 53 selected policy/action cases plus three execution failures across all 46 families, 644 structural-only cases and nine artifact-loss gaps. Save 53 proposed contracts and eight bounded implementation contracts. Review inventory completeness is not 709 substantive adjudications or scorer acceptance. Evidence: `luna-development-reward-review.md`, `luna-development-policy-contracts.json`, `luna-development-policy-dispositions.md`.

- [x] (2026-10-03) Implement the next four direct-request predicates and opt-in native scorer integration. Combined Simple/HR/effect suite: 51 passed, including four source-hashed actual scorer/reload replays and normal loader selection. Critic-requested exact prompt binding, typed hydration projection, malformed-container handling and outcome/recipient distinctions are addressed. Evidence: `reward-candidate/simple-slice-qualification.md`. Whole-trace goal subjects, accepted guards and complete task eligibility remain open.

- [x] (2026-10-03) Qualify isolated three-model benchmark scaffolding and source-bound eligibility mechanics: 133 calibration tests, focused Ruff/Pyright and diff checks pass. Actual retained HR accounting qualifies but historical scorer drift deliberately rejects eligibility. Evidence: `qwen-benchmark-scaffold/README.md`. The separate frozen official-verification seam is now being implemented; actual model deployment, eligible pool, runtime budgets, curricula and final bank remain open.

- [x] (2026-10-03) Qualify a shared deterministic world-evidence adapter in the isolated candidate (12 tests including actual retained HR receipts, focused Ruff/Pyright clean). It preserves occurrences and exposes qualified snapshots/ACKs separately from domain success; no guard coverage is inferred from absence. Add a future-source Mailchimp set serializer with two tests across four hash seeds, without sorting ordered fields or rewriting the historical mismatch. Evidence: `reward-candidate/shared-effect-evidence.md`. Integration into broader task predicates remains open.

- [ ] (2026-10-03) Apply the user's deterministic-first clarification to every reward contract. Use existing public facts, returned records and acknowledged effects directly; judge wiring does not make a judge mandatory. Implement deterministic coverage checks over declared candidate sets, relevant services, receipt/capture completeness, acknowledged revisions and initial/final reconciliation. Separate recording gaps from policy ambiguity; repair public fixture/authority contracts where necessary rather than ask a judge to invent missing facts.

- [x] (2026-10-03) Qualify a separate reward-redesign identity in the isolated candidate: nine tests, Ruff and focused Pyright pass. Actual adapter/native sources, lock, canonical configuration and runtime are bound without changing historical Luna identities. Evidence: `reward-candidate/redesign-identity.md`. Strict source-bound eligibility remains in progress.

- [x] (2026-10-03) Qualify the first three-task HR slice through actual `Task.score` and WireTrace reload: offboarding timing, referral privacy and NDA accomplishments. Previously listed root review issues are corrected; 23 new tests plus 25 existing regressions pass, with focused Ruff/Pyright clean. Evidence: `reward-candidate/first-slice-qualification.md` and `first-slice-replay.json`. This supersedes pending first-slice entries below; full redesigned coverage remains open.

- [ ] (2026-10-03) Recover from a confirmed filesystem-full collection stop. Untimed process 25305 is terminal: 479 committed attempts, ten started without retained artifacts and fourteen never started (776 retained overall). Preserve all raw evidence and controller homes; do not resubmit the ten starts. Reclaim unused caches and local Posttrain job images, verify the committed bank including any replay mismatches, then resume only never-started tasks with a storage admission check. Final complete development coverage remains unmet.

- [ ] (2026-10-03) Root reviews the first HR rule implementation and requires fixes before acceptance: a payroll-address/bonus-keyword heuristic misses direct referrer payout notifications with forbidden finance CC/BCC; raw name substrings can alias entities; missing captures need explicit coverage findings; sheet/worksheet/row identities must not collide; and known nonfuture departures must not become unavailable future-notification findings. Coverage diagnostics also need their own signal semantics rather than being labeled harm. Worker is adding independent counterexamples and actual loader opt-in qualification; predicates alone do not pass the scorer-path gate.

- [x] (2026-10-03) Save a bounded operational accounting checkpoint for the terminal current-source HR89 and timed197 partitions. Of 286 retained attempts, 284 normal completed-response budgets reconcile; the two HR pre-turn failures remain unavailable. None of those 284 reaches the 16,384 threshold. HR output median/p95/max: 862/2,185/3,571; timed: 905/2,451/3,366. Reported cumulative input is about 86% cached and includes repeated contexts; it is not a per-response context requirement or physical token proof. Evidence: `sdk-probes/luna-reference-campaign-01/operational-accounting-checkpoint.json`. This excludes the old 11 and live untimed bank, sets no new budgets, and makes no task-solvability or learning claim.

- [ ] (2026-10-03) Broader Astra development review is active. Root prepares 445 source-verified inputs across 41 task families with hidden assertion lists omitted and explicit unreviewed status. `prepare_development_review.py` is an offline review aid, not a scorer or native trace replacement; preparation does not count as behavioral review. The final required design inventory is 709 development tasks, with 91 reserved tasks excluded from fitting. Source/retry/isolation checks: `reward-candidate/review-preparation.json`; final coverage, recommendation dispositions and all-family implementation remain open.

- [x] (2026-10-03) Verify all 197 timed non-HR attempts: exact frozen source/archive, all episode/outcome hashes, all completed-response budgets, complete observed receipts and zero output overshoots; 28 official full outcomes. Evidence: `sdk-probes/luna-reference-campaign-01/remaining700/timed-verification.json`. Original owner 44914 is terminal at the expected untimed ownership handoff, while child 25305 continues the remaining 503-task bank without duplicate dispatch. These outcomes are not redesigned reward/guard acceptance.

- [ ] (2026-10-03) Implement the first reward slice in the isolated external-environment candidate, with public-policy predicates separate from native evidence integration. Offboarding notification timing, referral recipient privacy and NDA per-entity completion must pass retained development replay and independent counterexamples before acceptance. `reward-candidate/source-copy.json` records the explicit current-source copy and repository base; imported collection code, official scoring, dependency pins and raw traces remain unchanged. Full bank review and all-task-family redesign remain open beyond this slice.

- [x] (2026-10-03) Complete HR first-pass coverage: 100 distinct tasks/100 retained attempts, 17 official full outcomes across two preserved source versions. Current-source 89 contains 87 normal completed-response budgets and two unavailable pre-turn controller failures; the original 11 includes one unavailable schema finalization. No failed attempt was retried. `sdk-probes/luna-reference-campaign-01/hr100-coverage.json` binds both manifests; redesigned rewards/guards and Qwen eligibility remain unqualified.

- [ ] (2026-10-03) Remaining-700 collection now runs timed 197 (session 44914) and untimed 503 (session 25305) concurrently, ten slots each. The explicit untimed handoff preserves the original manifest, inventory and exclusive journal ownership; it supersedes the sequential arrangement recorded below. Latest owner checkpoint: 190 timed retained/seven draining, 159 untimed retained/ten active, no observed execution errors. Finish both queues and verify all source/outcome/budget records; keep loaded sources frozen.

- [ ] (2026-10-03) HR current-source queue stopped after 60 retained attempts on two pre-turn controller RPC startup failures; 58 completed normally, all 58 completed-response budgets reconcile, and eight official full outcomes are verified. All source/archive/episode/outcome hashes validate. Account remains available with no reached spend/quota limit; failed startup budgets stay unavailable. Root explicitly reviews and resumes only the 29 never-started tasks under the same manifest (live session 78265), keeping the first 60 consumed. Other-700 queue remains live/healthy; aggregate concurrency 20. Evidence: `sdk-probes/luna-reference-campaign-01/hr-remaining-firstpass/first60-verification.json`. Finish and verify the full banks before task classification or source edits.

- [x] (2026-10-03) Save Astra's preliminary review from seven HR development attempts, with 27 resolved native execution references and exact task-policy evidence. Findings include prohibited payroll notification, confidential finance CC/PIP disclosure, missing assertion coverage and semantic score false positives. It proposes concrete deterministic tests and a generic execution-occurrence API; it does not implement rewards, inspect held-out records or complete final bank coverage. Report: `docs/research/verifiers-assessment-qualification/luna-astra-preliminary-reward-review.md`.

- [ ] (2026-10-03) Launch the remaining 700 non-HR tasks alongside the live HR batch: transport session 44914, concurrency 10; root observes ten actual starts. Aggregate concurrency is 20 while both queues run. The frozen exact-700 bank has 197 declared-clock tasks followed by 503 untimed tasks, all initial round-trips valid. Reviewed prospective splits: 609 development, 64 reward-test and 27 evaluation; earlier unenumerated historical exposure remains unqualified. Model/budget/retry settings unchanged. Source, composition script hash, manifests and explicit aggregate-20 authorization are retained under `.posttrain/state/verifiers-assessment-qualification/luna-reference-campaign-01/remaining700/`. Finish and verify both queues before source edits or final task classification.

- [x] (2026-10-03) Save a real signed-in budget checkpoint for 23 completed current-source HR attempts: all response usage reconciles, zero public errors/retries, reported output 204–2,710, rolling refill 5–14ms. Read-only host inspection supports the second ten-way queue. Evidence: `sdk-probes/luna-reference-campaign-01/hr-remaining-firstpass/usage-checkpoint.json`.

- [ ] (2026-10-03) Use a preliminary user-requested Astra subagent to review completed HR development traces during collection. Critic's source-backed examples identify public workflow/status ambiguities, useful prerequisites and irreversible guard gaps, including a full-score notification followed by a correction. No runtime rewards are changed during collection. Final post-collection Astra coverage, implemented redesign and student eligibility remain open.

- [ ] (2026-10-03) First HR dispatch retained 11 attempts and halted on LinkedIn's own serialized `company_size` field being rejected by its legacy-only validation alias. Freeze and verify that old version, then deliberately apply the tested field-level alias repair to the actual vendor after the process terminates. Sibling domain regressions: 18 passed; root loaded-vendor frozen HR world round-trips: 100 passed. All 800 initial worlds were audited and only six instances of this same defect found. New remaining-89 batch is live in root session 70018 at concurrency 10/16,384 threshold; ten attempts started, all earlier 11 excluded. Complete and verify this batch before further source edits.

- [ ] (2026-10-03) Launch and monitor the actual first 100-task HR reference batch with signed-in Luna, concurrency 10 and the original 16,384 output threshold. Root exec session 25093 is live; inventory, source archive, manifest, execution policy and attempt journal are persisted at `.posttrain/state/verifiers-assessment-qualification/luna-reference-campaign-01/hr-firstpass/`. Root confirms retained attempts without infrastructure failures and rolling refill. One attempt/task, no retries, 600/660-second SDK/attempt limits, 24-hour deadline; all 100 tasks use valid declared-world time and remain development. Loaded source edits are stopped. Remaining: finish/verify this batch and collect the other 700 without resubmitting HR.

- [x] (2026-10-03) Make reported SDK errors/retries explicitly disqualify the budget clause because failed partial output can lack usage; reject empty terminal turn identities. Environment tests: 45 passed, Ruff/Pyright clean. Native error retention has a real unpaid recovered-retry reproduction; final combined SDK qualification and full-campaign family review remain pending before freezing sources.

- [x] (2026-10-03) Add conservative completed-response budget reconciliation in `calibration/budgets.py` and qualify native raw-response retention with six unpaid SDK executions. Root checks six distinct raw-journal hashes, the exact worker/test source hashes and archive integrity, then independently replays all six through the environment: two completed cases reconcile, the missing-usage case stays unavailable, and three interrupted cases stay unavailable. Scoped environment tests: 41 passed; static checks clean. Evidence: `sdk-probes/sdk-usage-accounting-02/`. Known retry/error handling and full campaign preparation remain pending; this is reported-counter qualification, not physical token completeness.

- [ ] (2026-10-03) Prepare one full 800-task first-pass campaign after critic/efficiency review: exhaustive timed/untimed partitions, family-disjoint splits, aggregate concurrency two, one attempt per task, no automatic retries, a 24-hour deadline and verification between bounded tranches. Native completed-response usage retention, conservative environment reconciliation and campaign orchestration are being implemented in parallel. No campaign inference has started. Retain failures for reward design; do not confuse collection admission with final student eligibility.

- [x] (2026-10-03) Complete the expanded signed-in smoke: twelve retained attempts across all seven domains, five full successes, four partial results and three zero-score results. Full successes: NDA, offer letters, contract renewal, HelpScout/Jira and Simple. Root independently checks both 670-file source archives, unchanged loaded source identity, episode/outcome hashes and all twelve official outcomes. All 169 observed native invocations have complete retained evidence. Evidence: `sdk-probes/luna-expanded-smoke-01/{summary,verification}.json`. Both owned processes are terminal; old manifests and traces remain unchanged. These are versioned debug samples, not an unbiased success-rate estimate or Qwen eligibility.

- [x] (2026-10-03) From expanded traces, reproduce and repair grouped Gmail-query semantics and future-dated marketing noise in the sibling fork, then deliberately refresh both vendor files after the batch terminates. Task messages/assertions and official scoring remain unchanged. New grouped semantics alone do not retrieve a policy absent the query terms. The noise audit across 100 tasks changes only 390 generated date fields and preserves initial assertion results; all 800 tasks and 9,863 initial assertions were checked. Actual forced-import vendor qualification: 102 passed, one unsupported legacy-runner test excluded. Hashes: `sdk-probes/automationbench-fixes-01/query-and-noise-refresh.json`.

- [x] (2026-10-03) Complete four fresh targeted attempts after these additional fixes, with a new source/scorer manifest, unchanged 16k threshold, timed-world context, concurrency two and no retries. Scores: NDA 1.0, I-9 0.0, conversions 0.0, advertising 0.5. All four traces retained; all 43 observed invocations have complete evidence. Root independently verifies the 670-file source archive, unchanged loaded sources, episode/outcome hashes and recomputed official outcomes. I-9 retrieved the correct Gmail compliance guide but still used empty BambooHR data and stopped; the zero episode score obscures that useful retrieval. Conversions again selected HubSpot; remaining workflow failures are not explained by the repaired parser/noise defects. Evidence: `sdk-probes/luna-targeted-smoke-02/{summary,verification}.json`. Both smoke generations remain separate, with single-attempt/stochastic comparison limits.

- [x] (2026-10-03) Repair reproduced benchmark issues in the sibling fork and deliberately refresh the actual loaded vendor: public `DocuSign Sent` status vocabulary for NDA/offer letters, simulated DocuSign action timestamps, and the previously omitted 25-file committed tool-fidelity delta. Before-copy checks prove each old tool file matched published base; new helper files are included. Forced actual-vendor qualification changes from 73 failing fidelity cases to 86 passing domain/fidelity cases (one legacy runner import case excluded). Actual adapter suite: 144 passed, five opt-in tests skipped. Root also fixes task-name filtering before world-time validation; selected tasks no longer fail on unrelated unselected clocks. No assertions/scorer changes or publication. Evidence: `sdk-probes/automationbench-fixes-01/`.

- [x] (2026-10-03) Execute the user-requested expanded Luna debugging batch: twelve explicit tasks across seven domains, including the four earlier workflows, one attempt each, 16,384 observed output threshold, no retries/fallback. Eleven timed tasks use world-time context; the untimed Simple task has a separate manifest. The timed collector uses concurrency two; the independent Simple collector uses one and overlaps it. Loaded sources remain unchanged until both terminal integrity checks pass. Results and subsequent fixes are recorded above.

- [x] (2026-10-03) Finish five actual no-auth pinned SDK accounting executions: completion below threshold, terminal crossing, active interruption, exact 16,384 and 16,385 with one-token overshoot. Root verified five distinct retained journals and their hashes. Repair the narrowly identified terminal race without swallowing other RPC failures; preserve actual terminal status and excess usage. Native config suite passes (32 ordinary cases; five opt-in executions tested separately). Evidence: `docs/research/verifiers-assessment-qualification/sdk-probes/sdk-usage-accounting-01/`. This qualifies executable cumulative counters and observed interruption, not signed-in reporting completeness, hard pre-generation admission, token alignment or Docker execution. Public-contract audit is retained under `sdk-probes/luna-hard-smoke-01/public-contract-audit.json`.

- [x] (2026-10-03) Implement opt-in `AutomationBenchTaskConfig.world_time_context` in the external adapter's `taskset.py`. Public system context exposes only the explicitly declared simulated timestamp, preserves offsets and discloses unspecified timezone; missing/invalid clocks fail rather than using host time. Original user messages and default upstream prompts remain unchanged. Existing environment tests: 24 passed; scoped Ruff/Pyright passed. This is a new task-context version, not a correction to old evidence; new manifests must bind it. Critic accepted the context contract. Tool-generated timestamps remain a separate clock-consistency gap.

- [x] (2026-10-03) Add explicit CLI `--world-time-context` selection and reject changed context on resume before dispatch. Two CLI tests pass with the current native source candidate; environment plus frozen-inventory tests: 27 passed. Scoped typing/lint and diff checks pass. No additional signed-in calls were made for this change.

- [x] (2026-10-03) Reproduce and repair the terminal usage race with the no-auth pinned app-server fixture. Initial tests showed `turn/interrupt` can fail with “no active turn” after completion; the five repaired executions above now pass. Signed-in usage completeness and hard-cap admission remain unqualified.

- [x] (2026-10-03) Finish all four signed-in hard-workflow probes with concurrency 2 and unchanged upstream prompts. Official partial scores: NDA 0.6, I-9 0.0, conversions 0.0, advertising 0.5. All completed normally; no failed MCP calls. Independently verify all 668 archived source files, episode/outcome hashes, sealed SDK journals and recomputed official scores. Correct verification to accept native `agent_completed`, require SDK evidence for SDK-routed manifests, and compare canonical JSON rather than Python boolean/integer equality. Scoped runner/collector tests: 45 passed; Pyright clean. Evidence: `docs/research/verifiers-assessment-qualification/sdk-probes/luna-hard-smoke-01/summary.json` and `verification.json`. These are retained failures, not qualified solutions.

- [ ] Audit public task information and replay the actual discovery queries before bulk collection. The independent adviser recommends checking NDA's hidden exact status, the I-9 world/SDK date mismatch and irrelevant tool-search results; preserve advertising's observed conditional-action failures separately. All four stopped far below 16k, so increasing budgets is unsupported. Fix only reproduced contract defects under a new version, then collect fresh matched probes; retain original evidence. Complete output-accounting semantics and Docker execution still need qualification.

- [x] (2026-10-03) Preserve native state artifacts that JSON trace export excludes. Collector writes immutable named-byte envelopes for retained and discarded traces, binds their hashes in episode evidence, and checks them during recovery/verification. SDK completion verification now requires sealed journal bytes to match the mutable info copy. Runner/collector tests: 41 passed, including missing/divergent SDK sources and artifact corruption; scoped Pyright/Ruff/diff checks pass. Legacy traces without envelopes remain available but cannot establish the new sealed-source check.

- [x] (2026-10-03) Prepare and complete four fresh signed-in hard-workflow probes: `hr.docusign_nda_collection`, `hr.i9_verification_tracking`, `marketing.conversion_tracking`, `marketing.ad_performance_review`. Adopt the efficiency adviser's rolling collector concurrency 2, one attempt/task, no retries, SDK timeout 600s, attempt timeout 660s and collection timeout 1500s. Keep upstream task prompts unchanged; defer the proposed 32-turn prompt cap until actual trace-budget calibration. One agent per episode is separate from task concurrency. Freeze package sources/scorer and all task worlds before dispatch, then preserve successes and failures without editing sources midrun. Results and independent verification are recorded above.

- [x] (2026-10-03) Qualify one manifest-bound signed-in Luna attempt through the normal native environment and durable collector. Official reward 1.0; seven complete MCP invocations; reported output 358 tokens. Recovery/reopening submitted no extra attempt. Root independently rechecked frozen inputs, all declared source-file hashes, episode/outcome digests and assertion scoring. Preserve the original scorer-file-SHA manifest rather than rewriting it; future manifests use the combined fingerprint. Evidence: `docs/research/verifiers-assessment-qualification/sdk-probes/luna-native-collection-01/summary.json` and `verification.json`. Hard-workflow smoke, complete output-accounting qualification and wider collection remain open.

- [x] (2026-10-03) Run the user-authorized signed-in Luna debugging task. `simple.email_sf_contact_phone_update` passed its sole assertion with reward 1.0 in 29.75 seconds; seven MCP calls and 14 native receipt events retained, with zero ordinary API fallback attempts. SDK reports 318 output tokens. Root independently recomputed retained assertion semantics against the final world and hashed initial capture. See `docs/research/verifiers-assessment-qualification/sdk-probes/luna-simple-debug-01/summary.json` and `verification.json`. The native environment/manifest-bound collection path and final budget contract remain separate gates.

Before implementing each milestone, follow the parent's execution-efficiency consultation protocol. Review parallel ownership, early route checks and minimal meaningful validation; retain the decisions in the parent runbook.

- [x] (2026-10-03) Add provisional offline verification of retained attempts against the frozen task, committed artifact/outcome hashes, current scorer source identity and final world assertions. Collector/runner tests: 25 passed, including malformed SDK completion evidence, nested task/manifest mutation, mismatched attempt identity and a failed excluded assertion with official score 1. Scoped Pyright and Ruff pass. This establishes current final-state outcomes only; Qwen eligibility remains pending redesigned action guards and complete 16,384-token accounting.

- [x] (2026-10-03) Align implementation milestones with flexible assessor authoring, strict published findings, native domain assignment and separate training alignment.
- [ ] Implement AutomationBench assessment and assignment rules through the qualified native API; validate full-context grading, arbitrary judge-output interpretation and recipient provenance in Milestone 4.

- [x] (2026-10-03) Inspect environment ownership, task/scorer definitions, current pins, Codex harness, and native evaluation storage.
- [x] (2026-10-03) Implement host-private task inventory save/load and actual-data instantiation with content validation, initial-score failure retention and overwrite protection. Three focused tests pass; the seven-domain offline smoke loads and scores 800 initial tasks successfully. No model calls.
- [x] (2026-10-03) Add opt-in raw state/action capture to all three tool interfaces and task setup/finalization. Combined inventory, environment and capture suite: 24 passed; scoped Ruff passed. Tests preserve actual results/worlds with controlled test clocks and reproduce failure-evidence loss through native MCP's state wrapper. This does not complete transport qualification.
- [x] (2026-10-03) Implement versioned collection manifests and durable attempt journals. Manifests bind frozen tasks, family/split labels and explicit limits; journals enforce exclusive ownership, physical execution versus logical attempt/retry accounting, persisted deadlines and crash-tail handling. Combined suite: 33 passed. Actual route, split authority, cost reservations and native artifact verification remain collector admission work.
- [ ] Complete neutral action capture through actual MCP, including failure-independent retention and concurrent-call evidence, then qualify hidden-state isolation and the real Luna route. Direct/local state records alone do not satisfy collection readiness.
- [x] (2026-10-03) Specify separate reference-collection and action-evidence stages, with validation and learning gates.
- [x] (2026-10-03) Review native assessment, token-projection, and consumer boundaries with an independent critic; record the generic API proposal in `docs/research/verifiers-assessment-api.md`.
- [x] (2026-10-03) Reframe Luna as a reference-trace generator for verifier and budget calibration, with no imitation or distillation objective.
- [x] (2026-10-03) Rewrite the milestones in plain language, explaining the work and expected result before implementation details.
- [x] (2026-10-03) Set the final stages to a Luna-subset Qwen benchmark, separate 2B/4B curricula, and ten fresh 4B rollouts per selected task; stop before subsequent reward iteration or training.
- [x] (2026-10-03) Restore Luna-based reward redesign as the primary goal in Milestones 3 and 4; keep training and further redesign after the 4B bank outside this plan.
- [ ] Pass the parent runbook's calibration readiness check, including the actual Codex/Luna route and retained execution/state evidence.
- [ ] Resolve a candidate environment revision including the existing local fixes; qualify the collection harness and exact GPT-6 Luna route.
- [ ] Freeze provisional success/guard coverage and capture the execution ledger before collecting reference traces.
- [ ] Implement calibration inventory, collection manifest, and native-trace verification in the environment package.
- [ ] Complete bounded reference collection, independent confirmation, and task classification; retain the coverage report and evidence-backed selection.
- [ ] After Luna verification, run the parent runbook's Astra rollout review and record decisions on its reward-design recommendations.
- [ ] Redesign episode/turn rewards, partial credit, and guards from Luna examples; document their meanings and test cases.
- [ ] Implement, test, and freeze the redesigned environment rewards before Qwen benchmarking.
- [ ] Resolve shared API gaps exposed by reward design, then complete the parent's six-environment migration, combined qualification and published pin adoption before Milestone 5.
- [ ] Freeze the Luna-verified subset and benchmark Qwen3.5 9B, 4B, and 2B on the same eligible tasks.
- [ ] Prepare and validate separate evidence-backed curricula and candidate budgets for 2B and 4B.
- [ ] Collect and verify ten fresh 4B rollouts per selected development task, save the bank and handoff, and stop.

## Surprises & Discoveries

The Gmail send adapter (`gmail.messages@1`) could never prove a send was
missing on a real trace. Every non-Gmail action opened send scope, and initial
reconciliation compared the sparse public task state with the hydrated native
snapshot as whole worlds, which always differ. Positive witnesses still worked,
so the defect was invisible to installed positive-only checks. Repaired with an
audited no-send handler allowlist (mirroring `_NO_SHEET_WRITES`) and Gmail-only
schema hydration that must equal the native BEFORE Gmail service and keep every
public key. Defaults drawn from uuid/now() fail closed. The aggregate proof's
`Notes == ""` eligibility filter was not public policy; the critic caught that
it diverged from the reviewed schedule rules under perturbed inputs.

The initial prepaid example missed a controller correction stored in Slack and
computed only the insurance/hosting subset, 4,300. Independent review caught the
omission before any task activation. Complete reviewed public-rule arithmetic
adds corrected software amortization of 600, producing 4,900. Review must cover
all relevant policy channels; passing arithmetic cannot certify policy coverage.

The earlier Simple completion queue does not span the conditional workflow
requirements in the 105-task sample. Recorded operation labels also miss tools
that unsuccessful runs should have used. Shared-work selection must use public
requirements, not just observed calls. Several old review gaps are now stale:
typed initial collections, retained-record checks, chained exact lookups, exact
decimal/date calculations and HubSpot creation/association already exist.

A read's returned data and its ordering are different facts. An increasing
state revision does not prove that a read finished before a concurrently
dispatched mutation. Any explicit read prerequisite must use comparable,
authoritative read-completion and mutation-start evidence, or remain unknown.

The first communication-basis implementation reread mutable native generation
material inside certificate admission, after the producer's decision stream had
been consumed. A critic-provided iterator added a successful call after context
capture and manufactured a violation tied to the original source digest. Freeze
or precompute the communication basis before consuming producer iterables. A
passing semantic fixture suite did not cover this source-lifetime boundary.

The v2 clarification campaign improves the inherited proposal gate from 17/22
to 21/22, but a new counterbalanced pair exposes an unsupported causal inference.
Identical request-shaped words in a Salesforce field and a genuinely sent Gmail
message both receive violation decisions. Only the send supplies addressed-use
evidence. The existing rubric already warns that storage alone is insufficient,
so another wording change is not the preferred repair. Add evidence admission
for communication basis and retain the raw semantic decision separately.

2026-10-04: The whole Contact public-policy audit finds a separate prohibition
on clarifying questions. The summary-exclusion rubric does not cover it, so
state/read/summary success cannot establish every public obligation. Keep this
shared guard capability open rather than introduce a manual task exception.
Approximate turn guidance and parallelism preferences do not define invented
hard counts. A proposed full-initial binding is verified against raw declared
data in three original sources (SHA `94932633f369d465faf0f5f3afaf4d328a9892d9241417eca23b4907f045b48f`);
private timestamp hydration debt does not affect that raw binding. Runtime and
final state must remain mutable.

2026-10-04: The critic verifies that native assessment execution retains valid
partial findings through failure/cancellation and the credit catalog admits
their exact records. AutomationBench's failed-run exclusion is consumer policy.
No current summary-path loss is demonstrated because it records the complete
evaluation before returning records. Preserve that policy now; explicitly
qualify target-local admission if future streaming introduces yield-before-fail.

2026-10-04: Multitarget assessment runs legitimately append partial findings,
but comparing only their configuration and target set misses changed earlier
rubric/producer/snapshot metadata. Independent mutation reproduced acceptance
of an altered prior rubric. The repaired planner compares all static run fields
and immutable evidence prefixes; native lifecycle reason/status and appended
output artifacts remain permitted. Contrast-corpus preparation also remains a
real prerequisite: written mutation recipes are not coherent admitted sources.

The second summary SDK probe exposed a response-interface problem after clean
model completion: a trailing `</final>` tag and one mistyped long output ID.
Strict parsing correctly abstained. The evidence journal also records a parse
error under the exchange kind, so counting records cannot establish how many
model calls occurred. Reconciled SDK usage proves one observed response. Both
original Contact findings and action credits remained valid despite this gap.

The existing SDK worker changes process environment variables while isolating
its configuration. Concurrent assessments must therefore launch isolated worker
subprocesses rather than call that worker in the shared environment process.
The retained Contact artifact proves authenticated Luna thread selection, but
its raw-response notification has no returned model or body. Preserve that
limitation explicitly. The 16,384-token limit is an observed cumulative output
threshold with possible overshoot. The worker can observe provider retries;
disabling outer retries does not disable internal provider retries.

The next semantic backend gate requires actual execution provenance. Existing
`vf.Judge.complete` preserves text/parsed output/usage but drops returned model,
finish and raw response metadata. A declared model label alone cannot satisfy
the summary exchange's observed backend identity. Prefer an environment-owned
adapter to the already-qualified signed-in SDK route for the user's subscription
execution policy; keep `SummaryBackend` generic and do not relax `SdkOnlyClient`.
Native transport, semantic accuracy and summary token credit remain separate.

The current public `support.gorgias_refund_processing` task requires escalation
to Jira project FIN but supplies `jira.projects=[]` and no project-creation tool.
The current installed handler rejects FIN, so this is an environment compatibility
gap, not a model failure. Positive effect-adapter fixtures now declare their
Finance project explicitly; that manufactured world does not qualify the public
task. Resolve the simulator/task contract from original benchmark semantics
before accepting this workflow, and preserve failed historical attempts.

The summary reducer can retain a valid harmful finding even when a later custom parser iterator fails. Empty observed text is distinct from missing capture. The broad regression also exposed stale expectations about unavailable delivery candidates and Jira audit IDs after canonical issue persistence changed. Source-frozen calibration tests correctly fail if implementation sources change during their run; a stable-source rerun is required, not bypassing the identity check.

Exact inner handler checks did not protect outer discovery dispatch: replacing
`search_tools` with a same-signature function still yielded false read-only
coverage. Both outer meta-tools now require exact loaded-callable admission.
Native harness and MCP invocation IDs are independently generated; retries
prevent treating call count/order as native attribution.

The real Contact episode has seven unique SDK/native coverage matches, with only
the installed search default needing expansion. Its archive still declares the
SDK/native execution join unqualified. Native lifecycle accounting and finite
population reconciliation can prove bounded coverage without rewriting that
provenance. Existing per-service adapter `complete` flags cannot close all
external authored channels.

The authored-output adapter's synthetic gates did not cover the retained SDK
collector's startup wrappers. Actual archive replay exposed that compatibility
gap. Independent probes also found false closed-empty inventories for unfinished
streams, malformed started items and foreign stream identities. Known text can
remain available while coverage stays incomplete. Accepting collector metadata
requires auditing its producer and reconciling its stream IDs; an unknown event
must not establish silence.

The recorded content workflow used the real installed Sheets append alias, which the evidence adapter had omitted. Recognizing it exposes a concrete prohibited queue append rather than improving the model or changing the historical trajectory. Canonical JSON also sorts public-request field keys; an insertion-order-dependent unavailable-field validator caused changed instructions to fail planning instead of publishing an unavailable assessment. Both fixes are shared evidence mechanics, not task-specific policy.

- Observation (2026-10-04): binding selected original policy messages alone permits an added competing policy without invalidating the old rule. The prepaid v3 declaration now binds both complete relevant message collections; independent public-hash review and added-policy tests pass.
- Observation (2026-10-04): obligation credit previously retained source identity while trusting selected witness configuration more than completion credit. Native request integration now carries full authenticated source and rederives the allocation; a coherent actual-effect/recipient substitution is rejected.

Recapturing internally consistent evidence from a transformed view did not prove
that it described the executor's source. The critic produced a stored false
positive even though later credit planning rejected it. Source authentication
must precede publication, not only allocation. The repaired native source anchor
and all four environment publisher admissions pass root integrated and
independent substitution/cache tests.

Final state can remain known when action ACKs are missing. Requiring a closed
action inventory for a terminal outcome would unnecessarily discard that fact.
Conversely, the original native ID must be known: replacing an object at the
same row position does not discharge its obligation. Seven independent critic
reproductions confirm those boundaries. Duplicate logical names are a separate
public-policy concern rather than a reason to reject all per-record checks.

- (2026-10-04) Installed tool metadata can overstate mutation scope: `google_sheets_lookup_row` advertises search-or-write but its inspected implementation only searches. Exact audited handler semantics, acknowledged unit-step capture and unchanged selected objects now close scope for known unrelated discovery/Gmail/Drive/Slack calls. Unknown/custom APIs stay open even if final endpoints match.

- (2026-10-04) Original public Sheets fixtures omit generated native IDs. The actual prepaid trace's revision-zero before snapshot supplies them, with exact matching public cells and unique scope. Rehydrating public data would regenerate IDs and is not evidence. Native prepaid guard testing also exposes unrelated known tool operations and hydrated-default differences as preventable scope gaps, requiring audited handler semantics and the same source-bound initial proof before compliance can be established.

- (2026-10-04) A typed Sheets row number is only a position. The critic executed deletion/replacement at the same number and a qualified update with a different native record ID. The proposed prepaid guard therefore needs original native identity as well as position; seventeen public bindings and twelve mutation checks alone do not settle this attribution boundary.
- (2026-10-04) Standard datetime parsing normalizes invalid offset minutes, and JSON model serialization can turn a copied boolean path index into an integer. Explicit offset grammar and Python-mode fresh schema admission now reject these cases. Derived expression fields must also participate in initial-context validation; the critic independently confirms the repaired boundaries.

Recipient delivery is a factual component of a requested welcome message, but
an unrelated invoice or cancellation email to that person also passes the
recipient check. Do not enable useful-action reward from recipient evidence
alone. Purpose/content coverage stays explicit until a suitable deterministic
or separately qualified semantic check exists. Native review additionally
reproduced forged scope receipts and execution retargeting; both require fixes
before positive transport acceptance.

The selected public requests expose mechanism gaps that transport tests cannot
resolve: required drafts differ from sends, single-incident handling needs
ranking, and financial schedules require policy-derived arithmetic. The next
shared slice is positive occurrence obligations plus frozen initial-record
populations. Partial check coverage must not stand in for full-task eligibility.
Native review also found that literal schema-version validation accepted booleans
and floating values; exact raw and copied-model checks now have nine regressions.

The 15-per-category sample reveals two distinct gaps. All 105 SDK traces are
tokenless, although they retain 1,727 exact native executions, and none of its
varied workflows is covered by the ten installed Opportunity manifests. Native
source/scalar/reload compatibility passes; semantic reward declarations and
student token alignment must be qualified separately. Dense guard replay also
reveals source/view duplication across 340 lifecycle batches, rather than slow
policy evaluation, as the major resource cost.

Independent review found that restoring an initially satisfied archival state could manufacture positive action credit despite zero net progress. The repaired producer keeps final completion separate from solver-induced accomplishment. The same review found that distinct assessment attempts with fresh generated IDs could multiply credit; a source-scoped one-attempt policy prevents that case, but failed-current-attempt fallback also requires explicit planning context. This is a shared API concern, not a reason to discard retained assessment history.

Full-campaign admission exposed a split gap: the smoke CLI assigns every task its own family and labels everything development. The benchmark supplies individually named task factories, without authoritative family metadata. Full collection must require an explicitly reviewed mapping of every task to a related-workflow family and a split; already inspected smoke families stay in development. A proposed grouping is not validated merely because its hash is stable.

The adapter's vendored HR task and DocuSign files matched sibling HEAD, but its other tool implementations still matched the older published base. Running sibling tests by path can silently import the sibling package and therefore miss this. Explicitly importing and asserting the actual vendor path before invoking pytest reproduced 73 fidelity failures; copying the exact maintained 25-file tool delta resolves them. Future loaded-source qualification must prove the import path, not only a green sibling suite. Historical traces used the old tools and remain valid evidence of that version; do not pool them silently with refreshed runs.

The hard smoke distinguishes transport success from task success. All four completed normally, but none passed all required assertions. Deterministic replay of all five retained I-9/conversion searches matches the 549-tool registry exactly: the initial I-9 workflow queries give its Gmail/Sheets/Slack tools zero BM25 score, and the conversion query ranks policy/data tools 36–113 outside its requested top ten. Service-specific queries can discover them. This is evidence of a discovery usability gap and observed model navigation failures, not evidence that tools are absent. Replay: `docs/research/verifiers-assessment-qualification/sdk-probes/luna-hard-smoke-01/discovery-replay.json`.

The NDA task's public prompt and tracker do not specify the asserted exact status `DocuSign Sent`; Luna sent the correct envelopes but wrote `Sent`. Audit and version the public contract instead of silently accepting Luna's string. The I-9 prompt lacks a simulated date while its frozen world says April 15; SDK context uses October 3. Date context cannot alone explain the empty BambooHR result, because the unfiltered employee list is empty too. DocuSign also stamps actions with host time (`automationbench/tools/zapier/docusign/envelope.py` uses `datetime.now(timezone.utc)`), so exposing world time does not unify every tool clock. Advertising received real policy/protected-row evidence and still made harmful conditional actions; retain that substantive failure independently.

Earlier exploration identified an OpenRouter route and implemented a guarded gateway. The user subsequently selected this Codex installation's signed-in account and authorized existing credits plus subscription allowance. OpenRouter is optional historical preparation, not the campaign route or its readiness prerequisite. Preserve logical model identity separately from the resolved SDK/runtime selection. Verify actual account access, exact Luna selection, effective tool isolation and native evidence retention during a bounded smoke; login status alone does not pass readiness. Never substitute another model or fall back to API billing automatically.

Native receipt corrections now preserve primary exceptions and bind applied claims to their invocation's persisted write acknowledgement. Focused receipt/reload tests pass (41); the broader native v1 suite passes (153, with 76 integration cases deselected). An actual AutomationBench MCP subprocess test also passes, covering failures, concurrent local-index collisions, authenticated transport, snapshot hashes, write acknowledgements and native reload. These are local source candidates, not published dependency adoption. Old selected runtimes retain only explicitly unqualified local capture. No receipt implies mathematical progress, task success or generated-token attribution.

The earlier plan described environments as supplying labels while algorithms owned all credit assignment. The accepted architecture now distinguishes native domain responsibility/allocation from trainer return and advantage estimation. Both are necessary; token coordinates alone do not decide which action deserves credit.

The selected Verifiers version has Codex and Claude Code harnesses, but the current Codex ACP adapter forces an injected gateway and does not implement the user's selected signed-in Codex route. The official Codex SDK documentation now includes the stable Python `openai-codex` package and `AsyncCodex`, controlling local app-server over JSON-RPC. Evaluate that SDK as the signed-in execution adapter while preserving native Verifiers task, MCP, episode and assessment ownership. Reuse compatible existing capture and persistence rather than assuming either that ACP already supports this route or that an entirely separate trace system is needed. Resolve and pin the SDK/runtime pair, verify `gpt-6-luna` availability without substitution, and qualify tool events, interruption, native replay and observable budget limits before the campaign. Source: https://learn.chatgpt.com/docs/codex-sdk (checked 2026-10-03).

Each task already includes a prompt, tools, a starting world, and assertions describing what to check. Some negative assertions have explicit markers. The current scorer also protects initially satisfied assertions through its free-assertion rules, so those markers alone do not identify every guard. Checks explicitly excluded from scoring must stay excluded.

The current partial-credit score does not show each useful change or guard violation separately. A prohibited conversion upload can execute successfully and still break a task requirement.

The external adapter has uncommitted changes to judging, tool descriptions, spreadsheet discovery, and prompt budgets. Review those changes before including them. The current RL catalog selects `11f4d712806d292c6c6a752af046f4e16c4f037e`; the adapter checkout is at `a6d779fc1fdfde23f86e297125b3381b140cec2f`. The tooling README names a different version from the catalog. Reconcile them before adoption; the executable selection determines which code runs.

## Decision Log

- Decision (2026-10-04): From round 5, authors record gaming paths as `known_gaming` instead of closing each one during drafting (user direction). Gaming is measured by random spot-check and closed with shared mechanisms, not per-task effort. Rationale: per-task hardening stalled coverage, while the spot-check showed most holes fall into two shared classes (hedging, narrow harm channels).

- Decision (2026-10-04): Reject reviewer fixes that reward inaction or penalize self-correction. Escrow: a "disputed row retained" goal paid a run that did nothing, so it was replaced with harm guards on deleting or clearing the row. Calendly: a duplicate-booking guard also fired on cancel-and-rebook corrections, so it was not applied; duplicate active bookings remain a documented gap that needs final-state guards. Airtable: the sabbatical note check accepts a wider list of deferral words ("on hold", "postponed", "on leave") to avoid false zeros, and new guards penalize course codes sent without any deferral wording.

- Decision (2026-10-04): Engine work runs in separate git worktrees while authors draft on the current engine, and is merged only after a green full suite. Rationale: editing source mid-run previously broke author and calibration runs; worktrees let both tracks proceed in parallel.

- Decision (2026-10-04): The checks are RL training feedback, not only eval scores; authoring and policy-interpretation calls are made by the agent and recorded here (user direction). Criterion: does the signal reward correct behaviour, penalize harm, resist reward hacking, and stay unknown rather than guess? Applied: (a) escrow ES7 — releasing a disputed escrow is harm (policy routes disputed items to legal); (b) escrow ES8 — overwriting escrow amounts/terms is harm (the instruction is a status update; collateral edits destroy source data); (c) expense split E13 — one check per expense × department pair counts as expressed, because each task's public data is fixed and hash-bound, so per-pair checks give a correct, specific signal; a data-general form adds no training value. Author: Claude.

- Decision (2026-10-04): Only deterministically checkable obligations are in scope. In scope: exact facts the public instructions require (named entities, amounts, IDs, dates/times, recipients), lines or values the prompt requires verbatim, and created/updated/preserved records. Out of scope: wording or judgement about message content (covers a topic, acknowledges, explains, summarizes, tone) — no word-list or phrase-list checks, because they reject correct paraphrases or accept wrong text. Tasks with judgement-dependent parts stay in the pool and are partially checked: their deterministic obligations are expressed and earn partial credit, and only the wording obligations are out of scope. "Fully qualified" means every deterministic task-specific obligation is expressed and verified. Requested by the user (revised the same day: partial checking is acceptable because credit is partial). Author: Claude.

- Decision (2026-10-04): Exclude the two shared system-prompt communication rules (`no_clarification` — do not ask clarifying questions; `summary_exclusions` — list only items acted on) from whole-task qualification. A task is fully qualified when every task-specific public obligation (prompt and public initial state) is expressed by deterministic checks and verified. The two rules remain separately reported, model-judged signals with their own semantic gate (currently 25/28); they neither block nor count toward the 15-per-category gate, and their results must never be merged into the deterministic task outcome. Rationale: they require semantic judgement, and every task inherits them, so leaving them in the gate would hold all qualification hostage to one non-deterministic check. Requested by the user. Author: Claude.

- Decision (2026-10-04): Report totals reuse the installed schedule checks' `required_when` and recognition amounts verbatim over the same `schedule` source, rather than the earlier standalone proof specs. Rationale: one reviewed definition of which rows are affected; a separate eligibility rule can drift. `line_number_eq` is known false only when no line mentions the label's words; near-miss formatting, quotes, fences, indentation, conflicting values and HTML-only bodies are unknown. Identical repeats count once. "Email to" accepts any delivered recipient role; address case is not normalized (known limitation). One correct send among several satisfies the occurrence; conflicting extra reports are not penalized here. Author: Claude after independent critic review.

- Decision (2026-10-04): Use public capability coverage to select the next ten-task acceptance batch; retain Simple tasks as fixtures rather than the primary roadmap. Start with a finite closed filtered COUNT/SUM operator, with authenticated source recapture, tri-valued eligibility and retained contributing identities. Do not introduce a grouping/ranking language or change the native Verifiers API for this increment. Two-sided absence/duplicate handling and temporal prerequisites are separate work. Rationale: existing scalar math does not prove complete totals; a narrow aggregate addresses an actual shared gap without inventing business policy. Author: Codex after Astra, critic and efficiency consultation.

- Decision (2026-10-04): Version the new communication admission explicitly as `no_clarification@2`; default `@1` and summary semantics remain unchanged for historical reading/replay. Calibration requires caller-selected `@2` preparation and fresh frozen identities, rejecting cross-version resume before dispatch. Rationale: changing certificate admission must not silently reinterpret previously retained judgments. Precompute basis from the same recaptured source/context before optional producer iteration. Author: Codex after critic and efficiency review.

- Decision (2026-10-04): Require source-backed communication basis before admitting a no-clarification harm certificate. A stored imperative is neither automatically an addressed request nor automatically a harmless literal. Missing basis yields unknown while preserving the raw judge output. Rationale: actual NC25a/NC25b results distinguish semantic request wording from observable communication, which the rubric alone failed to respect. Implement this in the existing environment operator/certificate path, without a new general assessor pipeline. Author: Codex after Astra and critic review.

Decision (2026-10-04): Attribute summary harm once per original physical
execution/check/channel, preserving every field finding as evidence. Plan
action membership independently of semantic policy applicability, so changed
policy abstains declared targets without inference. Reuse one retained exchange
and parse it once for planning; authenticate recipients against the native
accepted catalog. Explicit sum allows distinct physical retries sharing token
support; it does not allow duplicate action consumption. Source freezes include
tests during broad regression. Actual assistant-token mapping remains open.

- Decision: qualify a reliable summary response interface before further live
  semantic checks, preserving both failed attempts unchanged. Inspect actual
  SDK schema support and consider trusted request-local short references;
  unknown references and arbitrary trailing text remain errors.
  Rationale: copying long source IDs adds avoidable failure without adding
  semantic evidence. Deterministic reference resolution must preserve exact
  identity and must not manufacture a successful historical verdict.
  Date/Author: 2026-10-04, root after critic and efficiency consultation.

- Decision: summary harm assignment uses one negative contribution per physical
  originating action, check and channel; retain all offending output fields as
  evidence rather than multiplying penalties by field count.
  Rationale: the training target is the harmful action. This also avoids native
  assessment-target collisions when several fields share one execution subject.
  Predeclare the aggregate trace target and unique action targets from recaptured
  factual output provenance before dispatch, using one semantic exchange.
  Date/Author: 2026-10-04, root and independent critic.

Decision (2026-10-04): implement the optional summary backend inside
`automationbench_v1/summary_backends/codex_sdk.py`, using the existing pinned
native SDK worker as an isolated subprocess. Keep `SummaryBackend` generic and
the reference collector's `SdkOnlyClient` unchanged. Composition supplies the
protected authentication-file path; manifests supply policy and assessor
selection, never credentials or client construction. Journal complete actual
inputs before awaiting execution and preserve worker events before parsing.
Record requested model, authenticated selection and accepted thread model
separately, without inventing a returned response model. Reject observed retry,
refusal, forbidden execution, missing/regressing usage, budget overshoot or
incomplete terminal capture. This cannot guarantee suppression of provider
internal retries; a stricter controller promise needs separate qualification.
No new product baseline amendment is needed: this uses accepted environment
assessment preparation and native evidence ownership.

Decision (2026-10-04): initially close native invocation coverage only for clean
allowed before/dispatch/after lifecycles with all physical attempts observed and
source-bound. Rejected, raised and interrupted controls remain open until their
host semantics are independently qualified. Preserve already verified effects
and harmful findings despite that gap; no task rule or compliance threshold is
weakened. Final acceptance includes a real HTTP/state-controller path separately
from deterministic node/receipt fixtures.

Decision (2026-10-04): repair the public Gorgias initial fixture by declaring the
FIN project already required by its public request. Original upstream creation
accepted symbolic project parameters, whereas current canonical persistence
requires an existing project; this is a compatibility regression. Keep strict
unknown-project rejection rather than creating arbitrary projects implicitly.
The simulator sibling owns the fixture change; qualify actual public creation,
API/Zapier parity and original assertion compatibility before exact vendor
refresh. Existing archive snapshots remain unchanged.

Decision (2026-10-04): qualify summary transport with controlled certificates before using a real semantic backend. Keep request preparation and arbitrary output parsing caller-owned, retain raw exchanges before parsing, and require independently declared complete-output coverage. This proves the API boundary without pretending test certificates establish prose accuracy. Native output closure remains a separate emitted-call inventory gate.

- Decision (2026-10-04): Separate invocation accounting from external authored-field classification. Start with audited operations actually occurring in reviewed tasks; preserve unsupported operations as gaps. Keep unconstrained Contact string values as authored record facts and structured action relations, including no-op submissions, rather than assume typed field names exclude summaries. No new native schema, cache or product-baseline amendment is needed.

- Decision (2026-10-04): Preserve the user's ten-task authoring cadence. Codex drafts source-bound manifests, independent review and recorded replay admit supported checks, and a per-task acceptance entry keeps gaps visible. The first 105 public reviews are complete; this is not full task qualification. Further library batches use the same process without named-task evaluators.
- Decision (2026-10-04): Freeze any episode ledger containing valid manifest credit to one canonical manifest digest across all six current policy families. A revised reward design needs a separate replay ledger. Valid partial credit remains consumed after cancellation or failure; a new selected recipient does not create a new obligation. No semantic equivalence resolver or new public native API is needed.

- Decision (2026-10-04): Keep task manifests as data and qualify reviewed ten-task batches against recorded trajectories plus valid alternatives. For created objects, require proven episode-initial absence and finalized retained existence; select completion credit separately. Preserve known identity facts when richer hydration fails. Bind full relevant policy-message inventories before enabling a compiled rule. This prevents newly appended instructions from silently leaving stale policy active.

- Decision: Install the independently reviewed supported prepaid schedule checks together with the existing harm guard, and the feature-launch Asana occurrence component; keep unrelated obligations explicitly open.
  Rationale: These declarations have public-source authority, actual saved rollout replay, valid alternatives, adversarial cases and native rescore/reload evidence. No whole-task claim follows from component success.
  Date/Author: 2026-10-04 / Codex with independent critic acceptance.
- Decision: Close exact reviewed policy inventories by binding their entire relevant message collections when no general precedence resolver exists.
  Rationale: Added authority must invalidate a stale compiled interpretation. Exact version admission is deterministic and does not pretend to interpret new prose.
  Date/Author: 2026-10-04 / Codex following critic reproduction.

- Decision: Separate obligation applicability from supported calculation semantics. Retained checks optionally evaluate supported_when against frozen initial context; a required but unsupported/unknown instance stays present and abstains. Existing receipts and completion selection reuse that result; no extra native API or store is added.
  Rationale: A missing cap or ambiguous rounding rule must not disappear as an excluded row or earn a numeric label. The same mechanism handles other source-defined calculation domains through manifest data.
  Date/Author: 2026-10-04 / Codex, following efficiency and independent critic review.
- Decision: Keep batches of ten as the authoring unit, with a separate acceptance queue. Allow the next public-input batch to be drafted during qualification, but install only independently reviewed and replay-tested rules. Missing evidence adapters become shared work, not task-specific evaluators.
  Rationale: Batch reviews are exposing reusable gaps; counting them as working manifests would hide the remaining reward-design work.
  Date/Author: 2026-10-04 / Codex, following the user's batch-authoring instruction.
Decision (2026-10-04): add retrospective-only access to the executor's privately
anchored native snapshot and share one environment admission helper across all
manifest publishers. Preserve unrestricted preparation of working views, but
authenticate claimed raw projections before results are emitted. Keep prefix
visibility and serialized native source authority unchanged. Cache exact admitted
wire strings, not mutable policy results or unverifiable source labels.

Decision (2026-10-04): implement terminal retention as a distinct outcome check,
without changing occurrence semantics or inventing action credit. Root owns
the shared check/schema, efficiency owns native publication, critic owns final
identity-admission repairs and independent tests. This separates known final
facts from the later allocation policy and allows parallel implementation.

Decision (2026-10-04): bind omitted original IDs only through acknowledged unit-step revision-zero before evidence, with exact supplied fields/cells, unique native/positional identities and agreement across captures. Preserve missing/conflicting evidence as unavailable. Audit exact installed handler semantics for adapter operation scope; a tool's name, metadata label or unchanged endpoints alone cannot prove absence for an unknown mutation.

Decision (2026-10-04): expose reserved candidate identity separately from request cell values, and require source-backed native record matching for entity-specific row-update guards. Missing original identity remains unavailable. Do not install the prepaid proposal using positional equality alone. Keep actual retained initial-state evidence as the authority and investigate a shared initial binding rather than introducing task-specific row IDs.

Decision (2026-10-04): keep exact derivations as typed manifest operands with explicit formats and rounding rules. Use exact rational intermediates and preserve unavailable results; do not add ambient precision, host clocks, silent amount/date coercion or inferred business policy. These choices address public arithmetic rules while leaving joins, policy interpretation and aggregation as separately qualified capabilities.

Decision (2026-10-04): retain exact assessor views in owner-local native pools
rather than requiring compact environment projections. Preserve standalone and
legacy inline records and validate resolved references before native prefix
checks. Training export keeps compact results while full pools remain artifacts.
The next semantic capability is population-based required-effect obligations;
prohibited-effect checks cannot stand in for required sends, drafts or updates.

Decision (2026-10-04): use the fixed 105 development cases as an intermediate
API-gap and manifest qualification gate, followed by the whole task inventory.
Retained failures and partial attempts are valid test evidence but cannot grant
Qwen eligibility. Implement shared missing capabilities and data-only task
declarations; distinguish transport, bounded check coverage and full task
coverage in every result. Repair native evidence retention before scaling dense
guard publication. This preserves the existing product meaning and stopping
point; no new cross-environment authoring engine is introduced.

Decision: Implement the manifest engine only within AutomationBench for now, and
update the architecture before editing implementation code. Keep three priorities:
small manifest/operator vocabulary, evidence-aware results and separate credit
assignment. Reuse the existing native API; resolve only demonstrated bridge gaps.
Cross-environment generalization, visual editors, query optimization and elaborate
templates are outside this implementation. Broader existing native compatibility
work is a separate workstream and does not require applying this engine to other
environments. Date/author: 2026-10-03, user.

Decision: New tasks must be defined through generic manifests, not handwritten
task-specific evaluator code. Manifests declare task goals, prohibitions,
conditions, populations, evidence requirements and credit recipients using
reusable operations. Python may implement shared operations and environment
capability adapters, but must not hide task-name, exact-prompt or fixture-specific
branches behind a manifest-selected plugin. Existing bounded evaluators are
migration and qualification inputs, not the scalable final authoring system.
Astra is reviewing the smallest compositional contract and its validation gates
under this constraint before implementation. This changes the authoring approach,
not native findings, credit/alignment ownership, collection evidence or the final
benchmark/curriculum/trace-bank scope. Date/author: 2026-10-03, user.

Decision: Use reviewed action-family contracts and shared transport for the next implementation batches. The shared publisher only handles source-bound rederivation, subjects and contribution membership; it cannot invent candidate populations, policy or causal responsibility. Start with record updates and route-or-deny workflows, keeping per-task acceptance explicit. Rationale: native evidence mechanics repeat across tasks, while public policy and valid alternatives remain domain concerns. Date/author: 2026-10-03, Codex with critic and efficiency-agent consultation.

Decision: Expose current assessment attempts and pre-existing assignments through `CreditPlanningContext` and `Task.plan_credit`, with the default hook delegating to legacy `credit_requests`. The reviewed deterministic producer selects current results and treats repeated valid domain assignments as an intentional no-op. Rationale: retained history is evidence, not an instruction to sum every assessment or fall back after failure. Native runtime stays neutral about domain selection and independent algorithm reuse. This refines the accepted native assignment architecture without changing scalar reward or trainer advantages. Date/author: 2026-10-03, Codex and critic.

Decision: Require an explicitly reviewed guard inventory, which may be empty, rather than a synthetic guard reward on every task. The candidate `TaskRewardContract` version 2 binds inventory status, exact guard keys, policy digest, review revision and task-specific versus public-environment scope. Omitted, unresolved or unreviewed inventories cannot establish eligibility. Rationale: direct requests may contain no task-specific prohibition; inventing a reward does not prove safety. Old candidate version 1 contracts must be rebuilt rather than silently upgraded. Date/author: 2026-10-03, Codex and critic.

Decision: Honor the user's updated minimum concurrency ten and launch the actual 100-task HR collection through the existing tested CLI now. Rationale: the SDK seven-case qualification and scoped runtime checks have passed; optional full-campaign orchestration and split refinement must not delay evidence collection. Treat all HR as development to avoid later held-out leakage. Keep the original 16,384 threshold and stop new scheduling on infrastructure/account failure. This supersedes concurrency-two recommendations for future dispatch, while preserving historical manifests. Date/author: 2026-10-03, user and Codex.

Decision: Prepare one frozen first pass over all 800 tasks, with aggregate concurrency two and one attempt per task, after unpaid usage/campaign qualification. Keep final student eligibility pending reward redesign, action guards and complete reported usage; keep failures rather than looping until success. Rationale: the completed smoke runs establish actual collection transport, and a broader trace bank is required to redesign rewards. Stop new dispatch on account or infrastructure failure and inspect bounded tranches before continuing. Date/author: 2026-10-03, Codex after critic and efficiency consultation.

Decision: Expand bounded real Luna debugging instead of launching all 800 tasks or waiting for full reward redesign. Keep the current loaded sources frozen while collecting; develop isolated sibling-fork fixes concurrently, refresh the actual vendored source afterward, and rerun affected tasks. Rationale: the user explicitly asked for more runs and AutomationBench fixes, and broader actual failures can expose issues the four-task smoke missed. The adapter currently imports its vendored `src/automationbench`, not the sibling checkout; both candidate files were checked byte-identical to sibling HEAD before changes. Date/author: 2026-10-03, user and Codex.

Decision: Add explicit simulated-time context only through an opt-in task configuration and new frozen manifest. Preserve the exact declared timestamp, disclose a missing offset and never synthesize time from host defaults. Rationale: relative dates need observable task context across all harnesses; scorer targets and required actions remain private. The critic revised its initial UTC-normalization recommendation because changing offsets can shift the local business date. Date/author: 2026-10-03, Codex and critic.

Decision: Do not describe explicitly initialized service namespaces as an exhaustive connection catalog. They prove fixture data presence, while a valid action may target an initially empty service. A future opt-in discovery context may disclose initialized services non-exhaustively, or use a separately authored public connection manifest. Never expose assertion-derived `allowed_services` as public discovery evidence. Rationale: improve navigation without inventing account availability or leaking the verifier's required tools. Date/author: 2026-10-03, Codex and critic.

Decision: Wider environment migration follows AutomationBench reward design. Rationale: real action-level rules may expose missing API capabilities before other environments adopt the contract. Safe calibration capture and route qualification precede Luna collection; the full migration and dependency qualification precede Qwen benchmarking. Date/author: 2026-10-03, user and Codex.

Decision: AutomationBench assessors may transform available raw evidence and interpret arbitrary judge output using ordinary Python and optional shared helpers. Published findings and native domain assignments must satisfy the common typed contract. Assessment targets, observed context, cited evidence and credit recipients remain distinct. Rationale: grading one turn can require an entire workflow, while rewards still require dependable semantics and provenance. Date/author: 2026-10-03, user and Codex.

Decision: Build calibration under `automationbench_v1/calibration`, without importing Posttrain. Rationale: this package already owns the tasks and scoring rules, and the user chose this location. Date/author: 2026-10-03, Codex and user.

Decision: Use checked reference traces to develop rules and budgets. Start recording intermediate state before collecting them. Rationale: without that state, we may be unable to explain what an action changed. A successful path is an example, not a required sequence to imitate. Date/author: 2026-10-03, user and Codex.

Decision: Attempt every listed task and export a selection with verified solutions, while keeping the full inventory. Rationale: this supports the requested initial filter. Luna failing a task does not prove it is impossible or tell us how difficult it is for the student. Hold task selection fixed when comparing rewards. Date/author: 2026-10-03, Codex.

Decision: Try each task once, then collect additional runs for confirmation, selected failures, or budget measurements. Rationale: extra runs should answer a specific question. This saves attempts on tasks where more examples add little, but the resulting counts cannot be treated as an unbiased success-rate estimate. Date/author: 2026-10-03, Codex.

Decision: Use an actual signed-in Luna attempt on one Simple task to debug the SDK integration, then iterate on observed failures. The user explicitly requested this probe. Keep one attempt at a time, retained native evidence, the 16,384-token threshold, no API fallback and no automatic retries. Tool exposure and controller credential isolation must be checked; campaign-wide reward and curriculum qualification need not precede this debugging attempt. A debugging success is not full campaign qualification. Date/author: 2026-10-03, user and Codex.

Decision: Permit the SDK's MCP resource discovery/read helpers for this debugging route only after the configured benchmark server proves empty resource/template lists and rejects unregistered resource URIs. These helpers are scoped to configured MCP servers; they are not domain actions or controller filesystem tools. Disable goal-management, agent-management, execution and question capabilities. Rationale: the pinned SDK unconditionally exposes MCP resource administration when a server is configured; a second dynamic-tool bridge would add complexity without improving the qualified empty-resource boundary. The critic independently supported this narrower admission. Date/author: 2026-10-03, Codex.

Decision: Test rules against harmful examples and valid alternatives, and save their results in the common Verifiers assessment format. Rationale: replay alone does not prove that a scorer catches mistakes. Reusing the common format avoids a second validation and storage system. Date/author: 2026-10-03, Codex.

Decision: Keep the official benchmark score and give additional training signals their own names. Rationale: we need to compare benchmark results and inspect progress and harm separately. Date/author: 2026-10-03, Codex.

Decision: Luna-based reward redesign is the primary deliverable, implemented and checked before benchmarking Qwen. Rationale: the reference traces should produce a better reward system, not only a collection of examples. Training remains outside the plan, and the final 4B bank ends execution. Date/author: 2026-10-03, user and Codex.

Decision: Qwen3.5 9B, 4B, and 2B may attempt only tasks with a currently valid verified Luna solution. Freeze this pool before benchmarking, and derive both curricula from it. Rationale: the user wants to study smaller-model performance within an established solvable subset. Date/author: 2026-10-03, user and Codex.

Decision: After benchmarking and curriculum preparation, generate ten fresh 4B attempts per selected task and stop. Rationale: the next reward-design round needs a stable bank of successes and failures; tuning during collection would mix scoring or policy versions. The ten-per-task interpretation makes the requested rollout groups explicit. Date/author: 2026-10-03, user and Codex.

## Outcomes & Retrospective

The new aggregate evidence operation is implemented and independently accepted
within its standalone scope. Real public prepaid arithmetic includes the newer
Slack correction, and adverse membership/value cases pass. Native manifest
integration and checking the agent's actual report remain the next work.

Current scope clarification: the 105 tasks have public-input reviews, not 105
working manifests. The installed catalog has 23 entries: thirteen belong to
that selection and ten are separate Simple Opportunity pilots. Whole-task
qualification remains zero. Three acceptance cohorts cover thirty distinct
tasks, often through unsupported or partial drafts. The new opportunity matrix
will distinguish declaration refresh, reusable code and unresolved public policy;
its coverage counts will not be presented as completed tasks.

Ten additional manifest drafts now compile with independently verified public
bindings. Their two deterministic components remain partial checks, and eight
business outcomes still need adapters/composition. The new explicit communication
revision and calibration route are implemented; an uncovered producer-iteration
mutation holds their acceptance despite the 332-case baseline pass. Offline
preparation of all 28 fresh-revision cases is complete. No new model calls,
activation or whole-task acceptance has followed that finding.

The generic SDK isolation gap is repaired and qualified on fixtures, the actual
pinned SDK/local provider and a fresh signed-in canary. The v2 semantic campaign
retains all 28 cases with 26 accepted proposal predicates, not a passing whole
guard gate. The remaining new error requires communication evidence admission;
the other disagreement preserves a disputed historical proposal. Manifest
authoring advances to a ten-task cross-category second cohort independently of
that repair. Publication, whole-task qualification and downstream model work
remain open.

2026-10-04 action-credit increment: candidate code now publishes predeclared
action harm and once-only penalties, with controlled semantic integration and
independent history-integrity regressions. No additional task is installed or
qualified. The full environment gate is running, and actual negative semantic
examples still need coherent case preparation followed by bounded SDK execution.
This advances reward implementation without substituting integration tests for
semantic accuracy, full-task coverage or training qualification.

The latest increment qualifies shared Gmail/Zendesk output capture and proves
that an optional failed summary assessment preserves the installed Contact
read/update credit. It does not yet supply a reliable semantic guard or summary
harm assignment. The first ten-task whole-acceptance queue and the broader
fifteen-per-category gate remain incomplete; reviewed/component/whole-task
counts remain 105/13/0.

The semantic preparation milestone now has 26 proposed challenge cases and an
audited subscription SDK execution path. Neither establishes model accuracy.
Independent label review, the actual backend, retained exchange tests and a
bounded live semantic qualification remain open. Keep ten-task manifest
authoring moving through shared capabilities; semantic assessment is required
only where public prose meaning cannot be resolved deterministically.

Latest native acceptance (2026-10-04): the 180-case combined gate and independent
41-case HTTP/core review pass; final native core adds a same-parent retry case
and passes fifteen. The stable full package suite passes 2,387 cases with five
skipped in 661.93 seconds. This qualifies the source-frozen pre-backend increment.
The semantic challenge set
must cover exclusions, legitimate field values, quoted/negated wording,
incidental reads, incomplete capture and injection attempts. Controlled labels
are not evidence of a model judge's accuracy, and whole-task outcome acceptance
must not imply token credit for tokenless SDK outputs.

Native source integration checkpoint (2026-10-04): the public FIN prerequisite
is repaired in sibling and exact vendor, the factual native builder is reviewed,
and the executor snapshot now reaches external/summary checks explicitly. The
138-case SDK regression is green. Core reconciliation and actual HTTP/MCP
operation qualification proceed in parallel with distinct ownership. All
source-frozen calibration cases passed before this new source increment; rerun
them after it stabilizes. No published pin or whole-task count changes.

Latest local increment (2026-10-04): shared summary-exclusion transport passes
78 controlled tests, strict native receipt/token admission passes 175, consumer
overlap passes 35, and the corrected HR suite passes 23. Catalog counts remain
105 public reviews, thirteen bounded declarations, zero fully qualified tasks.
The broad package run is not yet green. The next native inventory projection
is a factual source builder only until source integration, full call/retry
reconciliation and live AutomationBench operation tests pass. Prose accuracy,
success-gated budgets and later campaign milestones remain open.

Historical shared-capability increment (2026-10-04): native fresh-object integration and installed Sheets append evidence are qualified within their declared scope. At that checkpoint, the real Jira draft passed seven tests but was not yet cataloged; strengthened content guard native replay was in progress. The current checkpoint below supersedes those counts and statuses.

Current checkpoint (2026-10-04): all 105 public reviews are complete, thirteen
development declarations are installed and zero whole tasks are qualified.
All six manifest credit families now share the consumed-revision boundary;
assistant-authored text is recoverable from exact archived SDK bytes through an
independently checked factual source. Final combined qualification passes 146
cases. External-output evidence is now locally qualified for the initial
audited SDK operations: combined gates pass 205, with independent 59-case review.
Native host-issued parent links and the shared summary operator now have local transport qualification. Native-only environment population closure and actual summary-policy interpretation remain open.
The Jira creation and content guard increments each pass their combined 100-case
catalog suites; full revised content score/rescore/reload passes in 195.00 seconds
with independent one-pass confirmation. Native fresh-object critic qualification
passes 73 cases and Sheets append evidence passes 64. Source candidates
remain unpublished. Generic retained typed-record outcomes and the public Zendesk
status outcome are qualified locally. Narrow acknowledged Zendesk completion
credit is qualified and installed in revision `public_batch10_credit_v2`; native
proof reuse has passed its measured and broad regression gates. Full library coverage, budgets, migration/publication,
student benchmarks, curricula and the final 4B bank remain required.

Typed record outcome evidence and exact qualification hashes are recorded in
`docs/research/verifiers-assessment-qualification/reward-candidate/typed-retained-record-checkpoint.md`.
The subsequent Zendesk mutation adapter and once-only action-credit qualification
are recorded in `reward-candidate/record-completion-credit-checkpoint.md`.
The outcome-only declaration is historical. The current declaration credits
the verified original ticket update; its separate resolution-email obligation
remains open.

Historical batch-authoring checkpoint (2026-10-04): ninety of the 105 development
tasks have recorded public-input reviews. Four bounded reward components are
accepted; zero whole tasks are qualified. Batch ten is being authored. Local
retained-completion credit now passes the combined 348-case suite, but this does
not establish actual recorded positive-action coverage or library-wide reward
readiness. The next steps are replay of a useful recorded completion, shared
adapter work identified by the batches, and continued manifest qualification.

Later checkpoint (2026-10-04) supersedes that count: all 105 public task reviews
are complete. Six local declarations cover bounded goals or guards, while whole
task qualification remains zero. The actual prepaid rollout proves that useful
completion credit can distinguish correct Insurance/Hosting actions from stale
Software state without changing the original scalar. Optional support predicates
now expose unresolved accounting interpretations explicitly. The two Contact
requests also have tested full state-goal declarations, including a real failed
path. Evidence and commands are in
`docs/research/verifiers-assessment-qualification/reward-candidate/completion-credit-and-contact-checkpoint.md`.
The next real coverage work is shared capabilities and native task qualification,
not another round of public review or a new teacher collection campaign.

Latest source-admission and terminal-outcome checkpoint: 281 integrated tests
and the separately measured dense recorded Luna replay pass on the repaired
source revision. Eighty development tasks are publicly reviewed, with four
bounded components installed and no new whole-task qualification. Completion
allocation is the next shared capability; broader redesign and campaign gates
remain intact.

The terminal-row core now distinguishes earlier progress from retained success.
Native publication and original Luna replay pass the 222-test integrated suite.
Positive completion allocation remains open. Full reward
redesign, whole-task manifest qualification, efficiency budgets, other-environment
migration, consumer qualification/publication and the Qwen campaign remain open.


Latest increment (2026-10-04): forty tasks have public-policy reviews, four bounded sample components are installed, and the actual prepaid guard now has source-backed identity and closed declared observation scope. Its twelve native alternatives and 223 combined regressions pass. No sample task is newly whole-task qualified. The next positive-signal gates are retained-row outcomes (so successful writes followed by damage cannot pass), source-bound request populations and verified generated-ID joins. Journal aggregation/content, complete library coverage, efficiency, consumer qualification, publication and student benchmarking remain open.

Current increment (2026-10-04): thirty development tasks have public-input reviews across three batches. Three bounded components are installed from the sample, with zero new whole tasks qualified. Exact values and batch-two native replay pass 322 focused cases; shared Sheets/native identity-context checks pass 206. Dense retained guard replay remains about 71 seconds and 469,436 KiB after source recapture. Native initial-row identity is the next shared correctness gap before prepaid guard installation. Complete library coverage, efficiency gates, migration/publication and the frozen Qwen/curriculum/4B bank sequence remain required.

The native conditional-guard increment is qualified locally and the 105-task
raw transport sample passes. Exact archive pooling and shared Gmail send
evidence are also qualified locally. This is useful progress, not completion of the
redesign: broad task manifests, shared service/check capabilities, dense
validation efficiency, success-gated budgets and downstream campaign
gates remain. The cross-category sample supplies concrete evidence for the next
architecture and implementation decisions rather than assuming the Simple
pilot generalizes.

Additional qualification (2026-10-03): explicit current-attempt credit planning now covers both native trace and episode APIs, with 75 tests passing. Support publishes bounded source-bound findings under explicit review contracts (eight publisher tests), and the shared DocuSign extractor qualifies three real renewal sends (five tests). Cash-flow native integration works but two independently reproduced semantic gaps are being repaired before acceptance. Full-family reward design and all downstream migration/student gates remain open; none of these counts qualifies the final curriculum.

Current outcome (2026-10-03): the first-pass collection is terminal and four exact Simple requests now have actual source-bound candidate eligibility proofs. Original output counts are 111, 123, 132 and 193 tokens; no additional model calls were needed. The final calibration batch passes 31 tests. Marketing suppression is integrated into native assessments and credit; support predicates are qualified only within their declared contracts and native publishing is next. Earlier paragraphs below record historical milestones and do not describe the current collection state. Full reward coverage, accepted final source revision, six-environment migration/publication, Qwen benchmarks, curricula and the final ten-attempt 4B bank remain open.

The sixteen post-fix debug attempts are complete and inspectable. The next deliverable is a frozen first-pass campaign, not another success-seeking smoke loop. Native response accounting and explicit family splits are still being qualified; the campaign has not launched. Keep all current official scores separate from redesigned reward claims, transient-action safety and Qwen eligibility.

The implementation plan now covers assessor authoring and native assignment as separate work in Milestones 3–4. This is a planning update, not implemented reward behavior. Safe calibration readiness precedes collection; wider migration follows reward design and any resulting API fixes. No campaign or training is launched by this revision.

Local Milestone 1 implementation includes frozen tasks, collection manifests, a durable attempt journal, artifact-first bounded collection, receipt-backed action inspection, host-state retention, and a conservative Responses cost guard. The combined source-candidate suite passes 63 tests, including actual AutomationBench MCP subprocess failures/concurrency and native reload. Critic fixes bind episodes and dispositions to attempts and verify discarded retry artifacts. Unpaid Codex isolation/request compatibility, exact composition limits, price/model provenance, the joined unpaid fixture and paid smoke remain pending. No Luna reference campaign has run. Completion still requires a tested reward redesign backed by Luna examples, benchmark evidence from all three Qwen models, both curricula, and ten fresh 4B attempts per selected task. Training and further reward changes using the final bank remain separate work.

## Context and Orientation

The plan is stored in `/home/hammad/projects/rl`. Environment implementation belongs in `/home/hammad/projects/verifiers-environments/environments/automationbench_v1`, under `src/automationbench_v1/`. Its `taskset.py` owns task loading, `tools.py`, `api_tools.py`, and `limited_tools.py` own execution, and `scoring.py` wraps benchmark verification. Vendored benchmark definitions are in `src/automationbench/domains/<domain>/tasks.py`; assertions are implemented in `src/automationbench/rubric/assertions/` and registered in `rubric/registry.py`.

Generic harness and trace changes, only if necessary, belong in `/home/hammad/projects/verifiers`, principally `verifiers/v1/harnesses/codex/harness.py`, `verifiers/v1/harnesses/claude_code/harness.py`, and native trace/runtime surfaces. That checkout HEAD was `84ab782391bbfe1ac4f4ca32fa612e56d01b5b81` when inspected and has unrelated dirty work. The selected dependency revision is `e6a3d9bbfe6959b97878f451fc721793a232cd5f`; inspect and test the selected revision, not only the newer checkout.

Posttrain selects the environment through `packages/catalog/src/posttrain/catalog/base/environments.yaml` and reads its reward evidence. `packages/train/src/posttrain/train/reward_evidence.py` validates that evidence. The training algorithm and backend adapter decide how it changes the policy. An advantage is the signed coefficient used to encourage or discourage sampled behavior; an environment label still needs that training step.

The canonical baseline amendment dated 2026-10-03 gives environments reward meaning and native domain assignment rules, Verifiers reusable assignment and original-token alignment infrastructure, and trainers return/advantage estimation and optimizer objectives. Native traces remain replay authority. This plan implements environment findings and assignments without building a trainer. New loss meanings still require their own named plan and any necessary baseline amendment.

## Plan of Work

### Milestone 1: Prepare the environment and make sure we can record what happens

Before asking Luna to solve tasks, we need a working environment and a reliable way to record its actions. Otherwise we could collect a successful run and later discover that the intermediate state needed to explain its success was never saved.

List the actual tasks in all seven domains: simple, sales, marketing, operations, support, finance, and HR. Save each task's prompt, starting world, available tools, and existing checks. Record the exact versions used. If the loader creates random IDs or other generated data, save the resulting task as well as the seed and generator version.

Decide which task families we will use to develop the rewards, which we will use to test those rewards, and which we will reserve for the final student evaluation. Keep related variants together so that one version of a task does not leak its answer into another split. We can collect traces for every task, but examples reserved for evaluation must stay out of reward design, task selection, and budget fitting. If we use an evaluation example to change the design, we can no longer call it untouched.

Write down what the current verifier considers a complete solution. Include partial credit, strict completion, known guards, checks excluded from scoring, and what happens when a check fails to run. A guard is a condition the agent must protect, such as leaving excluded accounts untouched. Record gaps in the current checks. A trace can pass those checks while still revealing a missing requirement; adding that requirement later means updating the scorer version and checking the saved traces again.

For every tool execution, save its arguments, result or error, execution order, and the relevant state before and after it. Keep a record of generated calls as well as calls that actually executed, and record whether the run finished cleanly. These records should point back to the original Verifiers trace. Save state snapshots or changes that let us reconstruct the state later, using native artifact references to avoid copying the whole world after every call.

Recording must leave tool behavior and execution order unchanged. Begin by recomputing the required checks from the saved state. We can later skip checks that could not have changed, but only after tests show that this gives the same results as full recomputation.

Use the existing Codex harness with the explicit model ID `gpt-6-luna`. This model was checked against official OpenAI documentation on 2026-10-03: https://developers.openai.com/api/docs/models/gpt-6-luna. Keep that selection fixed throughout a collection. Confirm which endpoint or Codex authentication route the harness actually uses and which model it reports. Record credential variable names without their values. A model appearing in the desktop app does not prove that the collection harness can use it. If the requested route is unavailable, record the failure rather than substituting another model.

Check the existing local changes before adding new work: judge prompts and context limits, tool descriptions, spreadsheet discovery, and the number of turns promised in the prompt. Changes to prompts or tools create a different task version from historical runs. Keep optional judge calls disabled during basic collection unless a particular unresolved check needs one. Setting their reward weight to zero does not stop those calls from costing money.

Luna should see the task, its legitimate tools and results, and the execution limits. Keep hidden assertions, expected target IDs, source answers, host files, and previous solutions outside its reach. Use the harness's container runtime and inspect its built-in tools. If shell access cannot be disabled, restrict filesystem and network access and record how the interface differs from the student's. A solution using extra tools establishes solvability under that interface; claiming it works under the student interface requires checking the actual tool path.

Try a small set of tasks before full collection: simple execution, spreadsheet discovery, I-9 policy retrieval, conversion exclusions, and conditional campaign actions. Confirm the model, fresh starting state, isolation, saved usage, timeout/truncation handling, and clean finalization. Save and reload the traces and execution records, then reconstruct their states and check results. Where reconstruction is unavailable, say so. Replaying the recorded calls is required for paths we later use to test action attribution; terminal results can still be useful without replay.

This milestone produces a checked collection setup, a fixed task list and split, and example traces showing that we retain the evidence we need.

### Milestone 2: Collect task executions and check the results

Run Luna on each task and save what it does. Check each result against the current verifier and applicable guards. Keep successful and failed runs so we can study both useful actions and mistakes.

Start with one attempt per task, with two tasks running at a time after the small setup test passes. As soon as one finishes, start the next. Check results as they arrive. Run successful candidates once more from a fresh world to obtain an independent example.

Initial reference-attempt output budget: 16,384 tokens total across all model calls in one task attempt (user decision, 2026-10-03). Do not grant that amount again at every tool step. Confirm whether the selected SDK/runtime usage includes generated reasoning; retain separate reported fields and explicitly unavailable components. Qualify whether this is a provider-enforced cap or an event-observed interruption threshold with possible overshoot. Record exhausted budgets and overshoot rather than treating truncated attempts as verified solutions.

The Qwen-eligible reference subset requires a fully verified Luna solution, all required guards passing, and confirmed complete output usage within this 16,384-token attempt contract. Missing or invalid usage does not pass the budget check. Retain over-budget and larger-budget rescue traces for reward analysis, but do not use them to establish eligibility. If no qualifying solution was observed, report "not verified within the 16k budget" rather than asserting impossibility. Reverify both scoring and the budget clause before freezing the subset.

Calculate input budgets from the resolved instructions, tool schemas, public task prompt and retained conversation, using a verified model tokenizer/counting interface where available. Keep three distinct measurements: initial request input, peak request context, and cumulative input across all requests (including repeated history and reported cached input). Include tool-result/history growth and space for the remaining output within the documented context limit. Byte counts are size diagnostics, not exact token counts. If the SDK does not expose complete provider requests or a qualified counter, record the visible-input estimate and its method separately from provider-reported usage; do not invent hidden-token counts. Fit subsequent per-task budgets from retained successes and failures under this initial contract; a changed rescue budget creates a separate recorded attempt class.

For failures, first work out what failed. A broken tool or verifier needs a reproducer and a fix. A model mistake or exhausted interaction budget may justify another attempt. Choose those extra attempts to answer a specific question, and initially allow no more than six ordinary and rescue attempts per task. Further runs used to study budgets need their own recorded limits. Stop automatic retries when the same infrastructure or scoring defect keeps recurring.

Set limits for the whole collection as well: tasks, attempts, elapsed time, and concurrency. The user now authorizes existing Codex credits and subscription allowance through the signed-in account, superseding the earlier subscription-only restriction. Let the service determine consumption order; no credits-first switch has been established. Observe account usage, but do not equate polling with atomic spend admission. No automatic API/OpenRouter fallback, credit purchases or quota resets. Subscription usage is not an enforceable API-dollar ceiling. Infrastructure retries count against a separate retry limit and the overall limits. If we run out of resources before finishing, report which tasks remain incomplete.

Keep runs with larger rescue budgets separate from runs with the original budget. Likewise, choosing extra runs based on earlier outcomes does not give us an unbiased success-rate estimate. We can report how many successes we observed; estimating reliability requires a separate set of tasks with a fixed number of repeats.

Save native `traces.jsonl`, configuration, logs, and a collection record with their hashes. The record must identify the tasks, environment, Verifiers and Codex versions, model route, sampling settings, timeouts, token/turn limits, repeat policy, concurrency, and total attempt ceiling. Reuse the native evaluation runner to collect these traces.

Verification means checking the final result and all applicable guards under the recorded scorer version. A score of one is insufficient if required checks are missing or failed. Require a fresh starting world and clean finalization. Where replay is supported, execute the recorded calls again and compare the assertion results. Where it is unavailable, keep that limitation alongside the trace.

Give each task one summary state. Two independent verified solutions make it `repeatedly_solved`; exactly one makes it `verified_solved`. With no verified solution, use `environment_invalid` for a reproduced environment defect, `collection_incomplete` for missing required attempts or unresolved scoring, and otherwise `attempted_no_verified_solution`. Keep individual attempt errors separately, and record the version of the success checks. Results invalidated by a scorer change must be checked again before counting as verified. Luna failing a task does not prove that the environment is broken.

Implement this in `calibration/models.py`, `inventory.py`, `collect.py`, `verify.py`, and `cli.py`, with tests in `tests/test_calibration_*.py`.

This milestone produces a collection of checked executions and a report showing which tasks have verified solutions, which failed, and why. The totals must account for the task inventory, scheduled and completed attempts, valid results, and tokens/tool calls in each domain. Every selected task must link to a complete verified episode. Name the selection "tasks with a verified reference solution under this contract," retain the generator identity, and show excluded families. Keep the original evaluation tasks. Failed traces remain available for reward design even when their tasks are outside the initial selection.

### Milestone 3: Redesign rewards from the verified Luna examples

For each proposed rule, separately specify the assessed subject, raw context needed, interpretation of model output if used, published signal meaning, and credit recipient. A whole episode may support a finding about one turn; a tool result may justify assigning harm to an earlier call. Define the native assignment rule, allocation, gate and overlap behavior independently of token availability. Preserve episode outcome, useful progress and guard costs as named channels until an explicit composition rule combines them. Do not treat exact alignment as proof of responsibility.

Reuse attribution strategies across task families. A verified state-change obligation can credit its qualifying acknowledged transition; a prohibited send/write can penalize that occurrence even after correction; a multi-action obligation requires an explicit contribution rule rather than rewarding every preceding action. Verified prerequisite acquisition can receive its own named finding, without assuming that a later decision used the information. Already-satisfied initial obligations retain outcome success without action accomplishment credit. Each task still needs reviewed policy parameters, identities, applicability, evidence scope and obligation deduplication. Luna traces expose cases and test these rules; a successful trajectory does not make every action useful. Token projection, return propagation, advantages and normalization remain separate consumer operations.

Before finalizing the reward design, run the Astra analysis step in the parent runbook. Spawn `gpt-6-astra` against the actual development Luna traces, verification results, task/scorer definitions, and saved execution state. Retain `astra-luna-rollout-review.md` and `astra-reward-recommendations.json`; record which proposals are accepted, changed, rejected, or unresolved in `reward-redesign-report.md`. Astra guides the design through concrete examples; the executing agent remains responsible for implementation and tests. Held-out evidence stays outside this review.

Inspect successful and failed Luna executions to decide what the environment should reward, penalize, or leave unknown at episode and turn level. Review partial credit, useful progress, harmful actions, required investigation, and completion. Measure how much work valid solutions take. This milestone produces the reward design itself; the rules must also accept other valid ways to solve the task.

For each proposed signal, explain which observed behavior it describes, which Luna examples exposed the need for it, how it is checked, and whether it is an episode outcome, a turn/action effect, a diagnostic, or an unavailable judgment. Specify the meaning and scale, when the rule applies, how call results become turn results, and which effects are already counted in the episode outcome. Keep useful progress and guard harm separate so one cannot hide the other in partial credit. Deterministic checks are the primary implementation: use available task/world/policy facts directly. Model assessment is optional for a precisely named property the evidence cannot decide; do not replace exact facts with model opinions or use a judge to resolve missing fixture authority.

Check evidence availability deterministically too. For each task guard, declare its candidate population, relevant effects/services and observation window. Verify native execution terminal records, action snapshots, state acknowledgements and applicable revision continuity; reconcile initial and final state under explicit service semantics. A recorded failed command can establish no effect without making every other channel unavailable. If that closed scope is complete and contains no prohibited effect, the guard passes. Otherwise report exactly which completeness or policy condition is unresolved. Missing recording, conflicting policy and a genuinely unknown semantic property are separate cases; unavailable is not a substitute for implementing the checks.

Make this a reproducible coverage result for each guard, with the checked scope and failed completeness conditions attached. Distinguish state guards (for example, a protected record remains unchanged) from action guards (for example, never send an email to a prohibited recipient). Initial/final equality can establish the former only under its declared semantics; it cannot establish the latter because a forbidden action might have happened and then been reversed. For action guards, inspect the complete relevant effect history. An observed prohibited effect establishes a violation even if another part of the record is missing; absence establishes compliance only when the relevant history is complete. Missing evidence in an unrelated service must not invalidate an otherwise closed guard scope.

Qualify coverage checks with executed cases for an acknowledged rejected action with no effect, a missing relevant receipt, a missing unrelated receipt, out-of-order receipts with valid revision linkage, a revision gap, and a prohibited effect followed by a corrective action. Report recording completeness, effect interpretation and policy authority independently; a complete record alone does not qualify the domain rule.

Begin with the existing assertions and their markers. Review conditions that are already true at the start, inverse tasks, conditional requirements, and checks excluded from scoring. A condition being true initially does not by itself make it a guard. Build reusable rules for task families, with task-specific parameters where needed, and record which rules apply to each task and which questions remain unresolved. Additional checks get a new scorer version and remain distinguishable from the official benchmark result.

For each rule, record a stable ID, task identity, what it checks, when it applies, how it is checked, the evidence it uses, and whether it is `candidate`, `qualified`, or `rejected`. Goal rules describe required changes. Guard rules describe protected conditions or forbidden actions. Prerequisite rules describe information needed to make a decision. Keep informational checks outside training credit. These definitions belong to the verifier; hidden targets must not enter Luna's or the student's prompt.

Test proposed rules against harmful examples and alternative valid paths. For example, change the target of a write, skip a condition, repeat a forbidden action, reorder independent reads, use an equivalent authoritative source, or add an irrelevant successful call. Execute those changes in a fresh environment: editing the text of a trace does not establish that the new path is possible. Where an alternative cannot yet be constructed, record that limitation.

Replaying the same scorer can confirm reproducibility, but it cannot establish that the scorer detects every mistake. Check disputed cases against the task requirements, leave uncertain labels unavailable, and measure false positives and false negatives on the reserved reward-validation tasks. Use a narrower rule with reliable labels when broader labels would be guesses.

For each path, record assistant turns, executed calls, reads and writes, failed calls, runtime, and model input/output/reasoning usage where available. One assistant turn containing four calls counts as one turn and four calls. Keep cached input usage, generated output, and unavailable usage separate. Visible text does not reveal private reasoning or original sampled-token coordinates.

We need two kinds of budget. Hard limits stop a run to protect resources. Soft budgets charge for excess work only after the task is complete. Reference traces help propose both, but the shortest successful trace is not a proven minimum.

With at least five verified successes, we can propose a soft budget from the observed range. The initial candidate is 1.5 times the empirical 90th percentile, using a recorded quantile method. Five examples alone do not make that estimate reliable. Save the sample count and spread, then test the fixed proposal on independently collected valid paths. Keep efficiency disabled when support is weak, valid branches are missing, or valid solutions routinely exceed the proposal. Use a fixed collection protocol for budget fitting; mixing selected rescue successes under changing limits would distort it.

Tool-call budgets transfer only when the reference and student tool interfaces are equivalent. Token budgets also need comparable measurement. Luna's token count and retokenized visible text provide estimates; Milestones 5 and 6 provide successful Qwen measurements to check those proposals. Hidden reasoning and harness overhead must not be treated as visible output. Use a family-level budget only where branch equivalence has been demonstrated; otherwise keep generous limits and leave the profile provisional. Preserve low-support flags rather than adding unplanned runs after the final bank. Efficiency penalties stay disabled throughout the current collection plan.

Implement `calibration/rules.py`, `budgets.py`, and `export.py`. Export `task-selection.json`, `task-rules.json`, `budget-profiles.json`, and `calibration-report.md`. Add `reward-spec.json` and `reward-redesign-report.md` with the proposed episode/turn signals, definitions, source examples, comparison with the old scorer, known gaps, and required tests. Each export records its format/version, source hashes, sample support, uncertainty, and split; the original episodes remain the source of truth.

This milestone produces a reward specification supported by concrete Luna examples and counterexamples, plus proposed budget profiles. For each task, a reader can inspect successful paths, valid alternatives, guards, required sources, partial-credit changes, turn-level signals, unresolved cases, and budget measurements. A change to the task, tools, or scorer invalidates the affected profile. Budgets fitted from held-out tasks stay outside the curriculum export.

### Milestone 4: Implement and test the redesigned episode and turn rewards

The ten installed Salesforce manifests are a pilot, not library-wide coverage.
Expand the catalog across the complete AutomationBench task inventory before
calling this milestone complete. First publish a coverage ledger with one entry
per task: declared goals, guards, authoritative sources, supported shared checks,
credit policy, and explicit unresolved capabilities. Separate a valid declaration
from replay-qualified coverage; neither official score one nor a passing Simple
pilot establishes complete redesigned verification.

Before expanding to the entire library, qualify a fixed cross-category sample:
at least 15 distinct development tasks from each category (all tasks if fewer
than 15 are available). Choose varied service, conditional, multi-step, guard,
and evidence patterns rather than the first 15 task IDs. Record the selection
and source hashes before implementation. Each case must exercise the declared
manifest, native assessment execution, retained findings, credit recipients,
reload and rescoring, and failure/ambiguity handling. Report coverage by check
and task: a partial slice, schema-only acceptance, or generic transport test
does not establish complete task reward coverage. Produce an API-gap ledger
from these cases and resolve shared contract gaps before library-wide authoring.

Implement reusable missing capabilities in the environment, then add task
declarations as manifest data only. Group work by shared evidence/check pattern
so one capability unlocks many tasks; do not add named-task Python evaluators.

Catalog expansion uses a repeatable authoring pipeline. Public-input batch
preparation is implemented; proposal generation, acceptance and catalog expansion
remain in progress. Review batches contain ten tasks (the final remainder may
be smaller), following the user's batch-of-ten instruction. Enumerate the native task inventory and group tasks by workflow
pattern and required service capabilities. Build reusable declaration templates
for those patterns; instantiate task parameters and source bindings from the
public request, initial world and public policy messages or documents. Preserve
the source path and content hash for each normative rule. Existing structured
task data can be translated mechanically; free-form policy may need an agent to
propose a declaration, followed by independent review. Generated proposals do
not become accepted reward rules merely because they parse or match one trace.
Luna examples expose missing mechanisms and counterexamples; their action path
and hidden answer fields cannot supply invented normative requirements.

For each batch of ten, Codex first reviews every task's public instructions,
initial state and policy sources, then proposes manifest goals, guards and
credit rules. Save the proposals and unresolved questions before installing
anything. Independently review the declarations, compile them, replay the
recorded traces and test counterexamples. Install only the supported parts and
record precisely which obligations remain open. Finish with a batch summary:
tasks reviewed, checks accepted, whole tasks qualified and shared capabilities
still needed. Carry unsupported tasks forward in the ledger rather than
silently dropping them or writing task-specific Python. The final batch may
contain fewer than ten tasks.

The next batches are manifest-authoring batches, not another pass over the
already reviewed 105 tasks. Reuse their saved public-input review packs, check
them against the current shared capabilities, and author ten task declarations
at a time. Keep a stable task list and per-task disposition so interrupted
batches resume without duplicating work. Advance to another batch after saving
its declarations, replay/counterexample results and remaining gaps; unsupported
tasks need not hold up unrelated supported tasks. Activation still requires
passing the affected shared-code regressions and task acceptance checks.

Each batch produces a review pack, proposed manifest files and an acceptance
record with one entry per task. Each entry lists the requested outcomes,
applicable guards, evidence sources, useful and harmful action rules, and any
unsupported requirement. Codex authors declarations from public task inputs;
recorded Luna traces provide replay examples and expose gaps. Counterexamples
must include plausible failures and valid alternative ways to solve the task.
The acceptance record distinguishes reviewed, partially supported and fully
qualified tasks. Only fully covered tasks count toward the fifteen-per-category
gate. An unsupported requirement stays visible until a shared capability can
verify it; the author must not substitute named-task Python code or weaken the
task to fit the current adapters.

After the first 105 public reviews, finish whole-task declarations in a separate
ten-task acceptance queue while preserving their original public review packs.
The historical Simple queue below remains useful for regression fixtures and
bounded completion work. It no longer determines the order of shared capability
implementation. Select the next batch from the source-backed opportunity matrix,
favoring conditional requirements and service diversity. Preserve the
fifteen-per-category gate and carry every unsupported reviewed task forward.

| Task | Original review batch | Remaining work to verify |
| --- | --- | --- |
| `simple.email_sf_contact_assistant_update` | 08 | Conditional summary meaning and complete authored-output coverage |
| `simple.jira_accessibility_audit` | 09 | Jira submitted-field capture and conditional summary coverage |
| `simple.email_sf_contact_account_update` | 05 | Original introduction read/resolve obligation and summary coverage |
| `simple.zendesk_resolve_email` | 10 | Correct recipient and same-issue resolution message, send capture and summary |
| `simple.feature_launch_slack` | 07 | Retained requested Asana task and product announcement, complete output capture |
| `simple.slack_dm_meeting_reminder` | 05 | Correct recipient and requested meeting/material facts, output capture and summary |
| `simple.asana_sprint_section_task` | 04 | Fresh retained task with requested fields and section membership of that same task |
| `simple.hs_create_deal_with_contact` | 09 | Fresh retained deal with all requested fields and the correct contact association |
| `simple.email_hs_create_contact` | 02 | Introduction read and fresh retained Contact populated from that source |
| `simple.email_airtable_customer_welcome` | 01 | Fresh retained customer and meaningful welcome message to the correct recipient |

Recheck gaps against current code before implementation: the old public reviews
are historical capability snapshots. Extend shared audited output capture in parallel;
Gmail, Slack, Jira and Asana must preserve all unconstrained submitted text and
qualify relevant lookup operations, not just the main message body. Shared Asana
retention/relationship evidence and HubSpot creation/association evidence can
proceed independently. Message-purpose checks are separate from the negative
summary rule. Do not infer current objects from historical action-only handlers
or accept a delivered message as proof of its requested meaning.

The first new deterministic capability is a finite, versioned aggregate in the
environment-owned `contracts/aggregates.py`. Its input is one declared initial
typed collection or sheet population and a filter written with existing
predicates. Recapture the population from the raw source and selector; do not
accept a caller's closure flag as proof. Count distinct native member identities
or sum one existing exact value expression for each eligible member. Retain the
source, selector and specification digests, contributing identities and paths,
and raw numeric evidence. Incomplete inventories, unknown eligibility, duplicate
identities and invalid selected values leave the aggregate unavailable. A closed
empty population may legitimately yield zero; absent data may not. Excluded
members need not supply irrelevant numeric fields. This output is evidence,
not action credit or proof that an email contains a correct report.

Qualify the standalone arithmetic/membership boundary first, then connect the
result through the existing manifest compiler, prepared assessment views and
source recapture. Replay a real prepaid-amortization public input and test missing
rows, duplicate identities, unknown exclusions, exact fractional totals and
valid alternative record order. Only after native replay/reload/rescore and
independent review may a task declaration consume the new result. Add two-sided
relation status separately for reconciliation; do not silently select the first
duplicate or parse correction prose into an invented precedence rule. Add
tie-aware ranking separately, preserving unspecified ties. Explicit observation
prerequisites need read-return-before-action-start evidence and may abstain on
historical captures lacking it. No broader framework primitive is required by
this environment-only increment.

Keep authoring and acceptance as separate queues. A Codex author may prepare the
next ten public-input proposals while the previous batch is tested, but catalog
installation waits for independent review and replay. Stop expanding a workflow
family when its declarations expose a missing shared evidence adapter; implement
and qualify that adapter once, then revisit the affected drafts. Do not let
reviewed-task counts stand in for working reward coverage.

Complete summary action credit with a shared `summary_action_negative_once@1`
manifest policy, not named-task logic. In the environment package,
`contracts/summary_policy.py` retains each exact field finding;
`manifest_summary_assessments.py` predeclares aggregate compliance and distinct
action-harm targets from deterministic output capture, then uses one exchange
to publish their results. Native target identity is subject plus signal and
revision, so several fields on the same execution must share one action target.
`contracts/models.py` admits this policy only for a summary harm check.
`manifest_summary_credit.py` will validate the current assessment and original
physical execution subject, then allocate minus one once per episode, contract,
check, action and channel. `manifest_assessments.py` owns dispatch and lifecycle
wiring; the existing consumed-revision ledger remains the only deduplication
authority. These are implementation instructions, not completed behavior.

An admitted violation in any attached output establishes action harm even when
another field or a later capture is unavailable. Absence of action harm requires
complete capture and decisions for every relevant output of that action;
whole-trace compliance still requires all declared inventories to close. Group
membership comes from recaptured execution provenance, never from a judge's
chosen invocation references. Historical SDK assistant messages may establish
trace-level violations but cannot receive action credit without an original
qualified native subject. Character citations do not create token spans.
Separate physical retry actions remain separate recipients; any shared-token
projection must use an explicit overlap policy. Test two bad fields on one
action, clean plus unknown fields, repair, later gaps, coherent recipient
retargeting, cancellation and rescore/reload before installing the policy.

Run strict schema/capability compilation over every proposed declaration, then
review the goal/guard inventory and test recorded executions and simulator
counterexamples. Missing operators create shared capability work rather than
per-task Python patches. After fifteen varied tasks per category pass semantic
qualification, apply the same templates and checks to the rest of each family
with bounded parallel queues. Review exceptions explicitly and reconcile every
task in a coverage ledger with separate drafted, schema-valid, bounded-check,
whole-task-qualified and blocked states. Do not exclude failed Luna examples
from reward-design testing; eligibility for later student runs is a separate
verified-success gate. Install accepted declarations in the packaged catalog
only after this review, retaining source and test identities for regeneration.

Replay every retained development Luna episode against its declaration and use
simulator alternatives for harmful actions, valid alternative solutions,
duplicates, repair, and incomplete evidence. Missing/lost episodes remain
explicitly unavailable. Freeze rules before applying the reserved validation
set; do not fit rules or budgets to reserved outcomes.

The library-wide gate requires inventory reconciliation, schema validation for
every declaration, explicit coverage gaps, reviewed goal/guard meaning, native
finding and credit persistence, unchanged separately reported official scores,
and qualified success-gated efficiency rules. A task with an unresolved required
guard cannot enter the frozen Luna-verified Qwen subset. Unsupported tasks remain
visible in the ledger rather than disappearing from the denominator. The parent
runbook's later migration to six environment packages is a separate operation.

Implement the reward specification in the environment so it scores new executions automatically. Emit redesigned episode components and useful/harmful turn or action components through the common API. Keep the official benchmark outcome available separately. A useful action and a harmful action can occur in the same turn, so preserve both results.

For example, a rollout reads an exclusion sheet. In a later assistant turn, it uploads two conversions: one allowed and one excluded. The read can establish that the required information was retrieved. The allowed upload can satisfy a goal. The excluded upload violates a guard even if the tool reports success. The turn should retain all of that, so the final score does not hide the harmful call.

Use the common Verifiers API from `docs/research/verifiers-assessment-api.md`, checked through the parent runbook. Its records can refer to an episode, trace, turn, call, or span of a message. They record who produced the assessment, what evidence it used, what its value means, and whether it is valid or unavailable. Deterministic checks and model assessments can use this same format.

Implement `action_evidence.py` and `reward_components.py` over the execution records saved in milestone 1. Cover Zapier, restricted Zapier, and API tools. Associate every result with its original trace, node, call, actual execution order, and before/after evidence. Retain exceptions and failed finalization. When calls requested together execute in sequence, record that actual sequence. If concurrent writes cannot be separated reliably, assess their joint effect at turn level or leave call attribution unavailable. The final state alone cannot tell us every intermediate state.

Keep assessor implementation in `action_evidence.py` and domain signal composition in `reward_components.py`; add `credit_rules.py` for versioned recipient and allocation rules using the qualified native assignment API. These are proposed environment modules under `src/automationbench_v1/`, not new framework packages. Use optional native source/message/receipt helpers independently. A model assessor may summarize or reorder the complete available context, make several calls, and parse arbitrary output. Preserve immutable originals and record the actual inputs, raw outputs, transformations and interpretation revision. Prefix-only access is selected explicitly. Model output becomes a published assessment only after common validation; invalid output remains unavailable. Deterministic checks use the same publication boundary without inventing prompts or model usage.

Qualify these modules with environment tests covering the same turn graded from full context and an explicitly restricted prefix, a multi-call judge, deterministic/model composition, arbitrary raw response parsing, and findings assigned to a different action than their evidence source. Check exact call support where available, explicit coarse turn support where selected, and tokenless semantic assignments. Preserve masks separately from broadcast/fixed-total allocation and turn-boundary rewards. Unknown alignment or attribution must not become zero. Retain assignments and rule identity in the final Qwen bank alongside findings; training Trackio exports remain results-only with full records in native compressed artifacts.

Report goal gains and reversals, new guard violations, restoration, tool errors, acquisition of required information, and episode completion separately. A read with no immediate state change may still be useful. A successful write may still be harmful. Keep "no effect" distinct from "we cannot tell." When providing turn-level totals from call-level results, record the aggregation rule so the trainer does not count each result twice.

Decide how each guard behaves over time. An irreversible forbidden action remains a violation even if a later action hides it. A repairable condition may be satisfied at the end while the earlier harmful action remains recorded. Do not blame the policy for a problem that already existed in the starting world. Check harmful executions directly as well as changes in assertion truth, because a second forbidden call can be harmful even when the guard was already broken. Repair must not create a reward that can be earned repeatedly by breaking and restoring the same condition.

Count progress against a fixed task-defined set of goals, with explicit handling for conditional requirements. Record reversals as well as gains. Test repeated gains and reversals under the trainer's actual discounting: equal positive and negative totals can still reward an early gain. Either award progress only once or use a bounded potential-based rule, which measures changes in a defined progress value. In either case, specify what happens at the end and test it. Adding the terminal outcome again through summed progress would double-count completion.

A retrieval check can establish that an authoritative source was returned. Establishing whether the model used it later requires examining the later execution and may remain uncertain. Reward verified acquisition under a named rule, not any Gmail search or reproduction of Luna's reasoning. Start with deterministic evidence where it can answer the question. Use a judge for a named unresolved property, retaining its method, instructions, observed scope, availability, and evidence. A model opinion remains a model opinion.

Keep official `partial_credit` and strict-completion metrics intact. A separately named candidate episode score may include goal outcomes and guard costs. Step components keep their own meanings; the selected trainer recipe decides their weights and how they change the policy. Store results alongside native traces.

A useful assessment can exist even when exact sampled tokens are unavailable. Applying it to a token mask requires a separately checked mapping to the student's original tokens. With exact call spans we can target individual calls. With only turn support, we must declare that broader scope.

This milestone produces a tested reward implementation, not only annotated reference traces. Test correct and excluded conversions, repeated excluded calls, mixed useful/harmful turns, initially protected conditions, excluded checks, inverse tasks, reversals, irrelevant reads, authoritative retrieval, repaired guards, scoring failures, concurrent writes, and truncation. A forbidden call must remain identifiable when the tool succeeds or the episode's partial credit increases.

Compare old and redesigned outputs on retained Luna executions and executed counterexamples. Report which harmful behaviors are now detected, which valid paths remain accepted, and which turn labels remain unavailable. Check that the same saved evidence yields the same results, invalid scorer execution cannot become a valid zero, and deterministic/model combinations preserve their inputs. Freeze the accepted reward revision before Milestone 5. Reverify the Luna solutions against its required completion and guard checks before fixing the Qwen task subset. These checks demonstrate reward correctness; they make no claim that a model has learned from the rewards.

### Milestone 5: Benchmark Qwen3.5 9B, 4B, and 2B, then build separate curricula

We now need to see what the Qwen models can do on tasks that Luna has already demonstrated are solvable. Run Qwen3.5 9B, 4B, and 2B on the same fixed subset and save all their traces. Their successes and failures will tell us which tasks are useful practice for 2B and which are useful practice for 4B.

A task enters this subset only if Luna has at least one complete solution verified against the current success checks and applicable guards. Freeze the task list and its evidence links before any Qwen benchmark starts. A changed scorer can invalidate the earlier Luna result, so recheck it before admitting that task. Tasks Luna has not verifiably solved remain in the reference-collection report and are not sent to any of these three Qwen models. A later Luna success creates a new subset version rather than silently expanding a running benchmark.

Apply this restriction within each previously declared split. Benchmark the same eligible development tasks for all three models, and keep any eligible held-out results separate from curriculum design. Report how many tasks and task families Luna excluded; this benchmark describes performance on the Luna-verified subset, not the whole AutomationBench inventory.

Resolve and record exact model revisions, tokenizer and chat-template identities, inference route, precision, sampling settings, tools, and limits. The current base catalog has 2B and 4B selections; resolve and add an immutable 9B selection before execution. Use each model's correct native template with equivalent task and tool access. Differences in template, reasoning mode, or runtime settings must remain visible in the comparison.

Use a fixed number of fresh attempts per task for each model, with no success-driven early stopping. The initial proposed benchmark setting is three attempts per task and model, recorded before collection; changing it means changing the benchmark manifest. Each attempt starts from a fresh world. Use the same task set, scorer, sampling policy, and comparable execution ceilings across model sizes. Keep successful, partially successful, guard-violating, truncated, and failed runs. Infrastructure errors remain separate from model mistakes and must not silently remove difficult tasks from the common comparison.

For each task and model, record complete safe solutions, raw partial credit, goal progress, guard violations during the run and at the end, retrieval and conditional-action failures, tool errors, output tokens, tool calls, turns, and truncation. Include every attempt in the report. Three samples provide a first picture, not a reliable estimate of rare success; keep small-sample uncertainty visible. Qwen9B is a comparison model, not a replacement for Luna's task-eligibility check.

Use these measurements to prepare two curricula: one for Qwen3.5-2B and one for Qwen3.5-4B. A task that 4B solves consistently but 2B only partly completes should not have the same starting priority for both. For each target model, group tasks by observed behavior: consistently complete, partial progress or mixed success, no complete solution yet, and unresolved execution/scoring. Use goal and guard details to distinguish a promising incomplete workflow from repeated harmful execution.

Each curriculum should specify its task IDs, the evidence behind their placement, and starting sampling weights. Include a mix of successful tasks for practice, tasks with useful progress and remaining mistakes, and a bounded allocation for harder tasks. Hard-task exploration stays inside the Luna-verified subset. Do not equate one successful sample with mastery, or remove a task solely because all three initial Qwen attempts failed. Unresolved environment or scoring defects require repair before a task enters the collectable curriculum.

Choose the initial weights from these benchmark results and record the choice; this plan does not establish a universal percentage in advance. Retain domain and task-family coverage and show which eligible tasks are included, held for repair, or reserved for evaluation. Both curricula must be subsets of the frozen Luna-verified pool. The 9B traces can help explain whether a smaller model is missing a workflow behavior, but they do not supply imitation targets or permission to add Luna-unsolved tasks.

Use successful Qwen paths to check whether proposed tool-call and output budgets are appropriate for each model. Preserve model-specific measurements and low-support flags. The 2B and 4B curricula may have different starting budgets where their valid paths require different amounts of work. This stage records candidate budgets; it does not enable an efficiency penalty or update model weights.

Implement generic benchmark summaries and curriculum exports in `calibration/benchmark.py` and `calibration/curriculum.py`, using the existing evaluation runner and configured inference services. Save `qwen-benchmark-report.md`, a machine-readable per-task/per-model summary, `curriculum-qwen35-2b.json`, and `curriculum-qwen35-4b.json`, with source trace links and fixed model/task/scorer identities.

This milestone produces the three-model benchmark and two evidence-backed curricula. Check that every Qwen task has a currently valid Luna solution, each model was tested on the same declared tasks, all attempts are accounted for, and each curriculum decision can be traced back to the recorded results. No student training is part of this milestone.

### Milestone 6: Collect ten fresh 4B rollouts per selected task, then stop

Once the two curricula are prepared, freeze the selected development task list from the 4B curriculum and generate a new trace bank with Qwen3.5-4B. Collect ten fresh rollouts per selected task. These are ten additional attempts per task after the benchmark, not ten rollouts total or a reuse of the benchmark samples.

Use the same fixed 4B model revision and a recorded collection configuration. Keep the environment, task versions, scoring rules, and sampling settings fixed throughout the bank. Do not enable new efficiency shaping, tune reward weights, update the policy, or change the curriculum while collecting it. This gives us a stable set of varied executions to inspect in the next reward-design round.

Number each task's logical attempts from 1 through 10. Start each from a fresh world and retain every outcome, including failures, harmful actions, and truncations. Do not replace a model failure with another sample to obtain ten successes. Bound and record infrastructure retries separately; they must not create hidden extra logical attempts. If the global resource limits are reached, leave the bank incomplete rather than claiming ten attempts were collected for every task.

Save native traces, execution order, tool arguments/results, relevant before/after state, assertion results, current episode and action assessments, generated-token accounting where available, and model/renderer identities. Retain the original sampled tokens and their source links where the runtime provides them. If exact token support is unavailable, keep the trace useful for turn-level analysis and mark finer masking unsupported. Visible transcript text cannot reconstruct missing sampled-token provenance.

Keep the ten-rollout bank separate from the benchmark collection and held-out evaluation. Use the frozen 4B curriculum's collectable development tasks; held-out tasks and tasks awaiting repair are excluded. Retain the 2B curriculum for later use without launching an equivalent 2B trace bank in this stage.

Save `qwen35-4b-ten-rollout-bank.json` and `qwen35-4b-ten-rollout-report.md`. The manifest must account for exactly ten logical attempts per selected task, link every retained native episode, identify retries and errors, and record missing state/usage/projection capabilities. The report should show variation in completion, partial credit, useful actions, guard violations, tokens, and tool calls within each task's group.

This milestone ends the current execution plan. Its final output is a checked 4B trace bank scored with the Luna-based redesign completed in Milestone 4. Stop after saving and validating it. Training and any additional reward redesign using this bank are separate work, with no follow-on command in this plan. Offline trace analysis alone must not be described as a model-learning improvement.

## Concrete Steps

During implementation, first inspect repository status, package manifests, and full pins in all affected checkouts. Do not include existing local changes implicitly. Use a dedicated candidate branch/worktree and an explicit patch inventory if needed. Validate the existing candidate fixes before adding calibration.

Existing native collection commands, from the environment package with the selected dependencies installed, are:

    cd /home/hammad/projects/verifiers-environments/environments/automationbench_v1
    uv run eval @ configs/calibration/smoke.toml --dry-run
    uv run eval @ configs/calibration/smoke.toml

Create `configs/calibration/smoke.toml` and `reference.toml` in milestone 1. Resolve the native config schema against the pinned runtime, including the exact model route, taskset domains/task_names, Codex version, container runtime, limits, and output location. These configuration files do not exist yet; do not treat these commands as currently runnable.

The planned environment-owned CLI contract is:

    uv run python -m automationbench_v1.calibration inventory --config configs/calibration/reference.toml --output artifacts/calibration/inventory.json
    uv run python -m automationbench_v1.calibration plan --config configs/calibration/reference.toml --inventory artifacts/calibration/inventory.json --output artifacts/calibration/collection.json
    uv run python -m automationbench_v1.calibration collect --manifest artifacts/calibration/collection.json
    uv run python -m automationbench_v1.calibration verify --manifest artifacts/calibration/collection.json
    uv run python -m automationbench_v1.calibration export --manifest artifacts/calibration/collection.json --output artifacts/calibration/derived

Implement `calibration/__main__.py` to run these commands. It should report missing dependency versions, a wrong model, malformed traces, changed tasks, missing success rules, or a collection request without limits. `plan` saves a fixed collection manifest, meaning a record of the task list, settings, identities, and limits; it makes no inference calls. `collect` follows that record and saves scheduling decisions and attempt IDs without overwriting valid episodes. Additional runs get a new manifest linked to the earlier collection. These commands become usable only after implementation and testing.

Milestones 3 and 4 add the reward specification and implementation. The proposed command for checking that implementation against the retained Luna evidence is:

    uv run python -m automationbench_v1.calibration rescore --manifest artifacts/calibration/collection.json --reward-spec artifacts/calibration/derived/reward-spec.json --output artifacts/calibration/redesign-check

`rescore` produces a new native assessment run linked to the original traces and the redesigned scorer version. It compares old and new results without changing the saved collection manifest or overwriting earlier assessments. Missing state or unresolved attribution stays unavailable. New judge calls require explicit bounded settings; the command must reject unbounded inference. Counterexample execution and the fresh-rollout scorer tests remain required in addition to rescoring. Use the accepted revision for the later Qwen runs and reverify the eligible Luna subset.

In Milestone 5, create `configs/calibration/benchmark-qwen35-9b.toml`, `benchmark-qwen35-4b.toml`, and `benchmark-qwen35-2b.toml`. Each selects the same frozen Luna-verified task pool and fixed repeat count, with its own exact model/runtime binding. Supply the existing configured inference service; the environment package must not launch a second model-serving system. A benchmark request outside the verified pool fails before collection.

The planned benchmark and curriculum commands, from the same environment package, are:

    uv run python -m automationbench_v1.calibration benchmark --config configs/calibration/benchmark-qwen35-9b.toml --eligible artifacts/calibration/derived/task-selection.json --output artifacts/calibration/qwen35-9b
    uv run python -m automationbench_v1.calibration benchmark --config configs/calibration/benchmark-qwen35-4b.toml --eligible artifacts/calibration/derived/task-selection.json --output artifacts/calibration/qwen35-4b
    uv run python -m automationbench_v1.calibration benchmark --config configs/calibration/benchmark-qwen35-2b.toml --eligible artifacts/calibration/derived/task-selection.json --output artifacts/calibration/qwen35-2b
    uv run python -m automationbench_v1.calibration curriculum --benchmark artifacts/calibration/qwen35-9b/benchmark.json --benchmark artifacts/calibration/qwen35-4b/benchmark.json --benchmark artifacts/calibration/qwen35-2b/benchmark.json --eligible artifacts/calibration/derived/task-selection.json --output artifacts/calibration/curricula

The benchmark commands save native episodes and a per-model `benchmark.json` manifest. `curriculum` reads only those explicitly supplied manifests and their development split; it must not scan unrelated or held-out collections into its inputs. It writes both curricula and the task-placement report. These commands and files are proposed until implemented.

In Milestone 6, create `configs/calibration/qwen35-4b-bank.toml` with ten fresh logical attempts per selected task and no automatic follow-on work. The planned final command is:

    uv run python -m automationbench_v1.calibration bank --config configs/calibration/qwen35-4b-bank.toml --curriculum artifacts/calibration/curricula/curriculum-qwen35-4b.json --rollouts-per-task 10 --output artifacts/calibration/qwen35-4b-bank

After the bank passes its coverage and retention checks, save the handoff and end execution. This command must not launch reward tuning, additional samples, or training as a completion hook.

## Validation and Acceptance

From the environment package run `uv lock --check`, `uv sync --locked --python 3.12`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`, `uv run pytest`, and `uv build --wheel`. Add focused tests under `tests/test_calibration_inventory.py`, `test_calibration_verify.py`, `test_calibration_budgets.py`, `test_action_evidence.py`, and `test_reward_components.py`. Run them first during development, then the full package suite. Record actual counts, never planned counts as passed evidence.

From the external environment repository root run its documented boundary check, `uv run --python 3.12 python scripts/check_boundaries.py`. If Verifiers changes are necessary, run its native harness/MCP/trace tests plus a real container smoke run. Generic runtime support is not established by mocks alone.

For Posttrain selection or export changes, run the relevant package tests, `uv run lint-imports`, typing/lint checks, and `git diff --check`, then complete broader release checks before adoption. Generic API and consumer qualification belongs to the parent runbook. This child plan implements the environment reward redesign, benchmarks fixed models, and collects traces; it does not run optimizer or learning experiments. This planning request launches no GPU runs.

Add `tests/test_calibration_benchmark.py`, `tests/test_calibration_curriculum.py`, and `tests/test_calibration_trace_bank.py`. They must reject Qwen tasks without a current verified Luna solution, preserve a common three-model comparison set, keep held-out evidence outside curriculum fitting, and verify that each curriculum is a subset of its frozen eligible pool. Test that a scorer change invalidates affected eligibility and that model failures do not disappear from reports. The final bank must account for ten logical attempts per selected task, including failures and truncations, with retries recorded separately. An incomplete bank remains incomplete.

The reward tests must demonstrate the primary deliverable: redesigned episode and turn/action signals run in the actual environment scorer path, preserve official benchmark metrics, detect the declared harmful examples, accept valid alternatives, and distinguish unavailable evidence from valid zero. Retain an old-versus-new report linked to Luna traces and executed counterexamples. Freeze the verified implementation before any Qwen benchmark; an unimplemented specification or a set of manual labels does not pass Milestone 4.

## Idempotence and Recovery

Identify each attempt by its collection manifest, task hash, and attempt number. Resume incomplete attempts without overwriting completed ones. Keep failed task attempts and record infrastructure retries separately. A scorer or prompt change needs a new manifest and a fresh check of affected selections and budgets. Leave old saved configurations intact.

Store large outputs under an ignored artifact directory or existing durable artifact storage, and retain source references and checksums in exported reports. Native scoring internals remain inaccessible to the policy. Existing adapter/Verifiers dirty work and the SAMPO report remain intact. Reverting a candidate selection means selecting the prior immutable environment and recipe, not deleting traces.

## Artifacts and Notes

Save the fixed task inventory and collection settings, native Luna episodes, the Luna-verified subset, reward specification and redesign report, tested rule implementation and revision, budget profiles, three-model benchmark traces and summaries, both curricula, and the ten-rollout 4B bank. Reports must account for every task and logical attempt, including exclusions, failures, and uncertainty. The final handoff identifies the completed Luna-based redesign, ends collection, and leaves training or further bank-based reward iteration to separate work. It makes no learning-improvement claim.

Update `docs/tooling/automationbench/README.md` when adopting a candidate; reconcile its pin with the executable catalog. Environment package changes remain framework-neutral. Publish generic Verifiers changes first if needed, then select their immutable commit in the environment package and lock. Publish the environment revision next, then update Posttrain catalog selections and lock/package references where applicable. The external environment repository prohibits commit/tag/publication without explicit user authorization; local implementation and validation can proceed, and any publication decision should use the tested diff and artifacts. No pin may describe uncommitted patches as reproducible.


## Interfaces and Dependencies

Use Verifiers' existing evaluation runner and trace APIs. In `calibration/models.py`, define versioned records that can be saved and loaded. `CalibrationManifest` describes the source, runtime, model, tasks, splits, and limits. `TaskCalibrationResult` records what the task achieved and whether execution and scoring worked. `TaskRuleContract` describes when a rule applies, what it checks, and what evidence it uses. `BudgetProfile` describes the usage measurement, supporting samples, quantile and headroom, and whether the budget is ready for efficiency training.

Add `ModelBenchmarkResult`, `CurriculumProfile`, and `TraceBankManifest` in the same module. Benchmark results bind model and runtime identity to task-level attempts and evidence. A curriculum records its target model, Luna-verified pool, split, task placement, sampling weights, candidate budgets, and reasons. The bank manifest records the fixed 4B identity, selected development tasks, ten logical occurrences per task, retained native episode references, errors, retry counters, and collection completeness. Keep these records usable without a Posttrain import; model serving is supplied by composition through the existing inference clients.

Save results through Verifiers' `Assessment`/`AssessmentBatch` records in `action_evidence.py`. An additional `ActionEvidence` transport format would duplicate that API. Put AutomationBench rule IDs, before/after truth, execution details, and uncertainty in the evidence attached to those records. Identify the assessed subject separately from raw context, actual observed inputs and cited evidence. Status can be `valid`, `inapplicable`, `abstained`, or `failed`. A valid zero is an observed result; missing evidence has no numeric substitute. `credit_rules.py` emits native assignments referencing accepted findings, semantic recipients, rule/configuration identity, allocation, gates and validity. Alignment separately maps those recipients to original sampled tokens when available. The environment supplies findings and domain assignments without importing a trainer or constructing advantages.

Code and a model can assess different properties of the same action. For example, code can prove that a write hit a forbidden target while a model assesses whether the decision was grounded. Keep both results and record the rule used to combine them; a positive model opinion must not average away a verified violation. If an optional model call fails, independent code-based results remain usable. A combined result that needs the missing model output stays unavailable. Reuse a cached model result only when the source, actual input view, judge instructions, provider identity, and prior evidence match exactly. Apply limits to uncached calls and retries. Add model scoring only for a stated question where it has demonstrated a benefit.

Revision note, 2026-10-03: initial plan records the requested GPT-6 Sol collection first, calibration inside the environment package, and subsequent action-signal development. It deliberately separates verified solvability, budget calibration, reward correctness, and demonstrated learning.

Revision note, 2026-10-03: an earlier version included efficiency activation and learning checks. Those sections have been removed from the current scope; the user will handle reward redesign and training separately.

Revision note, 2026-10-03: changed the active teacher selection to the user's requested latest Luna, verified as `gpt-6-luna`. Earlier Sol wording in the initial revision note is historical; all operative selections now use Luna. Model-route qualification and no-fallback requirements remain in force.

Revision note, 2026-10-03: the user clarified that Luna supplies reference executions for better verifiers, turn rewards, and length budgets, not teacher supervision. Removed demonstration training, moved evidence capture before collection, made sampling adaptive and bounded, added scorer counterexamples and split protection, required budget-transfer validation, and reused the generic assessment schema. The initial all-environment-migration prerequisite was subsequently superseded by the reward-first ordering below.

Revision note, 2026-10-03: wider migration now follows Milestone 4's tested reward design and generic API refinements. Safe calibration readiness precedes Luna collection; full migration and published dependency qualification precede Milestone 5. Consult the efficiency agent before each milestone to identify safe parallel work and early prerequisite checks.

Revision note, 2026-10-03: rewrote the plan after the user found its language hard to follow. Each milestone now explains the work, why it is needed, and what it produces. Technical requirements, code references, and commands remain in place; the implementation scope and completion conditions are unchanged.

Revision note, 2026-10-03: the user changed Milestone 5 to benchmarking Qwen3.5 9B, 4B, and 2B only on the Luna-verified subset and preparing separate 2B/4B curricula. Milestone 6 now collects ten fresh 4B attempts per selected development task and stops. Reward iteration and training comparisons are deferred, including the previous efficiency-activation milestone. This stopping point supersedes earlier revision notes describing training as an active deliverable.

Revision note, 2026-10-03: removed the detailed future-work section and efficiency recipe as requested. Reward redesign and training have no milestones, commands, objectives, or acceptance conditions in this plan.

Revision note, 2026-10-03: the user clarified that Luna-based reward redesign is the primary goal and belongs here. This supersedes the previous scope exclusion: Milestones 3 and 4 now specify, implement, and test episode/turn rewards, partial credit, and guards from Luna examples. The redesigned scorer is frozen before Qwen benchmarks and the final bank. Training and additional redesign after that bank remain outside the plan.

Revision note, 2026-10-03: added the requested Astra review after Luna collection and verification, before reward-design finalization. The parent runbook contains the launch brief and evidence/report requirements.

Revision note, 2026-10-03: propagated the accepted architecture into implementation order, owning modules, interfaces, and acceptance tests. Flexible assessor internals publish validated findings; native domain rules assign recipients; alignment and algorithm estimation remain separate. Preserved campaign ordering, qualification gates, and the final ten-attempt 4B stopping point. No runtime changes or collection were performed in this revision.

Revision note, 2026-10-03: advance from completed post-fix debugging toward one frozen full first pass. Record the pending response-accounting and family-split corrections, bounded dispatch/recovery policy and independent implementation ownership. Broader collection remains unstarted; redesign and student eligibility remain separate gates.

Historical revision note, 2026-10-04: at the fifty-task checkpoint, terminal-row
core qualification was separate from native publication and completion-action
credit. Later Progress entries supersede that checkpoint.

Historical revision note, 2026-10-04: validated batch nine and advanced the review ledger to
ninety tasks, started batch ten, and recorded the 348-case combined completion
credit regression result. Kept draft, accepted component and whole-task coverage
separate; actual positive-action replay and whole-library qualification remain
required.

Revision note, 2026-10-04: finished all 105 public reviews, installed two qualified
Contact state-goal declarations, and recorded actual prepaid completion credit
and supported-domain qualification. Updated remaining work to shared service
capabilities and native coverage. Preserved whole-task, budget, migration,
publication and Qwen/curriculum/4B-bank gates; no new model calls were made.

Revision note, 2026-10-04: install reviewed prepaid schedule and Asana creation
declarations, qualify authored request/native credit and Slack delivery transport,
and refresh the independently qualified Jira simulator files. Build the candidate
wheel and verify its new manifest/adapter resources; this is resource packaging
evidence, not published/native-pin qualification. Continue shared Jira evidence
and retained-new-object credit next, with the broader campaign gates unchanged.

### Execution increment: authored requests and Jira fidelity (2026-10-04)

The request worker owns the environment source/schema and native obligation
capture/restore integration plus relevant tests. Use the existing
`contracts/requests.py` interface; accept `public.request@1` only for
`new_occurrence`, without an invented initial baseline. Public bindings must be
revalidated, and missing authority must preserve an unavailable obligation.

The simulator worker independently owns `/home/hammad/projects/automationbench`
files `automationbench/schema/jira.py`,
`automationbench/tools/zapier/jira/actions.py`,
`automationbench/tools/api/impl/jira.py`, new
`tests/test_jira_state_fidelity.py`, and `CARBONTEQ_FORK.md`. Preserve existing
dirty changes. Follow the architecture's canonical issue versus audit identity,
unique project resolution, supported updates and legacy loading rules. From that
repository run `uv run pytest tests/test_jira_state_fidelity.py -q`, scoped Ruff,
and `git diff --check`; record any runtime/dependency limitation rather than
claiming a fake-only integration. Root then refreshes exact qualified vendor
files into the environment and records hashes; no live sibling import.

From the environment package, run request/schema/obligation/native replay tests
with `PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src`
and its `.venv/bin/python -m pytest`, plus scoped Ruff/Pyright and diff checks.
Acceptance requires authored request membership, public authority, acknowledged
actual effects, replay/reload stability and scalar preservation. Jira whole-task
qualification additionally requires actual persisted issue evidence; action logs
alone cannot satisfy it. Fresh replay never upgrades historical state.

This changes simulator fidelity and local environment capabilities, not the
frozen framework product meaning. Candidate changes remain unpublished. Later
publication order is simulator commit/push, environment vendor provenance and
commit/push, then framework immutable consumer pins and lockfile updates, with
each repository's fork ledger and consumer documentation kept in sync. Until
then the framework still selects its existing immutable environment/native pins.

Revision note, 2026-10-04: recorded the request integration and separately owned
Jira state repair before implementation. Parallelism follows repository/file
ownership; native evidence qualification follows simulator qualification.

The parallel Slack worker owns only new `contracts/slack_effects.py` and
`tests/test_manifest_slack_effects.py`. Root subsequently owns union/dispatcher
integration. Qualify canonical channel/timestamp identity, unique user/channel
resolution, ACK/result/persisted-message agreement, failed writes, missing ACK,
later edit/deletion, ambiguous identities and explicitly unsupported scheduling.
Use real installed handlers. This unlocks delivery evidence, not free-form
announcement/reminder meaning or whole tasks. Track each missing public goal and
guard in the 105-task inventory with a named capability and acceptance case.

After the Jira simulator/vendor gate, the evidence worker owns only new
`contracts/jira_effects.py` and `tests/test_manifest_jira_effects.py`. Root owns
the shared retained-new-object evaluator, completion projection and native
integration after evidence review. Capture true issue/audit linkage, before/after
canonical fields and a closed-or-unavailable final issue inventory. Qualify
missing ACK, ACK without insertion, insertion without ACK, legacy actions,
wrong project/type, later damage, supported repair, repeated creations and archive
reload. The creation predicate must establish fresh native identity and final
requested fields; do not use first-occurrence credit as a shortcut for repair
attribution or retained success.

The shared check worker owns only new `contracts/created_objects.py` and
`tests/test_manifest_created_objects.py`, coordinating the Jira evidence API
with the evidence worker. The same worker owns new
`manifest_created_assessments.py` and `tests/test_manifest_created_assessments.py`
for native receipt/credit transport. Root owns source/check/credit unions and
`ManifestAssessmentTask` dispatcher/planner wiring. The operator is
`objects.created_and_retained@1`; predicates use
authored request and canonical retained fields. The completion policy explicitly
selects earliest qualified completion, with revision ties unavailable. Creation
requires verified birth; updates require known false-to-true predicates plus
requested/changed goal-field overlap. Consume once by authored request identity,
not by generated object. Multiple distinct correct new objects satisfy an
existential goal; do not add a duplicate prohibition absent public policy.

Freshness means absence from the episode's initial global ID membership,
independently of project filtering. Initial public IDs or authenticated
revision-zero schema binding can establish this fact. Later qualified creation
can prove before-birth absence but cannot replace unknown episode-initial
membership. Keep that case unavailable rather than inferring freshness from
UUID generation. Raw finalization must be true for any terminal outcome.

Revision note, 2026-10-04: make episode freshness, finalization and occurrence
evidence distinct in the retained-new-object API. Add explicit earliest
completion selection and task dispatcher ownership; preserve scalar and legacy
archive behavior.

Revision note, 2026-10-04: begin the typed retained-record implementation in
coordinated core, native integration and public Zendesk lanes. Record the focused
129-case gate without treating the pending recorded replay or useful-action
credit as complete. Strict schema projection admission prevents malformed data
and vacuous named-target populations from masquerading as successful rewards.

Revision note, 2026-10-04: specify the Codex batch-of-ten deliverables and
per-task acceptance states. Keep partial requirements visible and require full
coverage before counting a task toward the category gate. Update the current
outcome to distinguish installed Zendesk action credit from its historical
outcome-only revision and outstanding email obligation.

Revision note, 2026-10-04: begin shared Gmail read evidence and the Contact
whole-task acceptance audit. Separate successful returned-message verification
from token-conditioning evidence. Preserve public Find requirements without
inventing output steps or a fixed tool sequence. This remains within the
accepted environment/native API ownership and needs no product amendment.

Revision note, 2026-10-04: qualify factual assistant and external-output evidence
separately from summary meaning. Consumed credit freezes one manifest revision;
controller-bound SDK/native population matching proves only coverage. Preserve
record-field submissions and action relations through gaps, repair and noops.
Record the native harness/MCP identity gap before Qwen use; no task, publication
or whole-workflow success is implied by the 205-case local gate.

Revision note, 2026-10-04: the first native parent-link increment now passes real
local MCP dispatch/retry/reload gates. Host-issued tickets link physical effects
to exact original sampled calls; historical SDK evidence stays unlinked and
generated-token alignment remains open. Qualification evidence is in
`docs/research/verifiers-assessment-qualification/reward-candidate/native-parent-link-checkpoint.md`.
Keep the user's ten-task Codex manifest authoring cadence already specified
above. Current reviewed/component/whole-task counts remain 105/13/0.

Revision note, 2026-10-04: native execution recipients now reuse exact retained
generated-call spans with distinct retry contributions. Final native gate:
158 cases passed; projector Ruff/Pyright pass. Fixture coordinates qualify
transport, not production Qwen parser versions. Continue shared summary-policy
coverage before whole-task acceptance; an optional summary must cite acted-on
relations and values, never a flat name whitelist or a required Luna wording.
Evidence: `docs/research/verifiers-assessment-qualification/reward-candidate/execution-token-alignment-checkpoint.md`.

Revision note, 2026-10-04: register shared summary-exclusion outcomes and native
transport with 78 controlled-backend tests. Preserve independent semantic
accuracy, native call-population closure and whole-task gates. The broad suite
is terminal with six failures; resolve adapter/expectation drift and rerun
source-frozen calibration tests only after all source writers release their
files. Authoring remains in batches of ten, with counts 105/13/0 unchanged.

Revision note, 2026-10-04: record the exact observed SDK provenance and process
isolation limits before implementing the optional summary backend. Add the
26-case reviewed proposal set without calling it a model evaluation or gold data.
The source-frozen regression has now completed successfully; subsequent code
requires its own qualification.

Revision note, 2026-10-04: close the stable full package regression with 2,387
passing cases and five skips. Begin the isolated SDK backend and audited Slack
DM capture in disjoint files, preserve bounded partial parser findings and
avoid semantic calls for exact empty output. Add a ten-task whole-declaration
completion queue without replacing the cross-category coverage gate.

Revision note, 2026-10-04: reproduce and fix summary checks breaking existing
Contact credit planning. Preserve independently proven contributions through
unavailable or failed summaries without assigning unimplemented summary credit.
Record local Trackio exclusion of the new raw SDK journal format and the first
independent Slack DM capture gate; whole-task and model accuracy gates remain
open.

Revision note, 2026-10-04: record the first actual SDK assessment independently
from its semantic outcome. Transport works, but a correct quote with invalid
character arithmetic yields abstention. Fix span derivation in the versioned
backend parser while retaining strict source/citation checks. The next shared
output increment must include Zendesk comments as well as Gmail sends.

Revision note, 2026-10-04: qualify the fixed clarification preparation/runner
and execute its separately reviewed finite campaign. Preserve the 17/22 failed
semantic gate and all raw decisions. Separate deterministic exact-empty handling
and qualified certificate references from uncertain prose interpretation;
record the shared-core residual and independent repair lanes before fresh
inference. Ten-task manifest authoring and the full downstream sequence remain
unchanged.

Revision note, 2026-10-04: record the generic SDK isolation repair, optional
metadata capture support, independent tests and successful fresh canary. Begin
the reviewed v2 semantic campaign without changing historical labels, retries,
whole-task counts, publication gates or the ten-task authoring cadence.

Revision note, 2026-10-04: the v2 campaign is terminal and independently audited,
with 26/28 frozen predicates accepted. Record communication-basis admission as
the next implemented evidence boundary and assign it separately from the next
ten-task, cross-category data-authoring batch. Preserve raw decisions and policy
revision lineage; this does not declare the clarification guard or whole tasks
qualified.
