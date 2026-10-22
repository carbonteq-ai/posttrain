# Run five adaptive ten-update guidance SDPO experiments (MTP discontinued)

Latest review follow-up: user requested four full teacher contexts combining
improved guidance with actual failed-attempt feedback BEFORE restarting. Offline
candidate artifacts are in docs/research/evidence/combined-teacher-four-20261019-r1/.
All four use real experiment03/update1 failed episodes. Guidance is DeepSeek-derived
and manually corrected; exact historical feedback JSON and rendered history outside
the supplement are preserved. No model call, native rerun, optimizer update or
training restart occurred. These candidates are for inspection, not qualified
training inputs. See deepseek-teacher-context32.md for provenance and reproduction.

Latest user-authorized follow-up, October19: separately generated32 teacher contexts
using OpenRouter deepseek/deepseek-v4.1-flash-1017, without Luna traces. This
supersedes no-external/no-LLM prose only for that finite authoring batch; it does
not restart training or renew old experiment clocks. See
[the authoring plan](deepseek-teacher-context32.md). All32 completed; API-reported
cost USD0.091739508. All32 passed structural coverage, but7 copied private IDs
and all5 independently source-reviewed cases had semantic/capability defects.
Do not admit these contexts into training. Full outputs and source reviews:
docs/research/evidence/deepseek-context32-20261019-r1/context-review.html.

This living project-research plan follows docs/templates/PLAN.md. Maintain
Progress, Surprises & Discoveries, Decision Log and Outcomes & Retrospective.
Worktree: /home/hammad/projects/worktrees/rl-rft, codex/rft-warm-start.
No frozen product-baseline amendment is required: this is an opt-in experiment,
not a change to framework training semantics. Preserve all existing dirty work.

## Purpose / Big Picture

The latest user goal authorizes five MORE experiments, each ten optimizer
updates, inspired by the SFT guidance pipeline and previous SDPO failures. This
new 50-update campaign supersedes the unlaunched two-arm guidance20 proposal.
No updates from that proposal have occurred. Use local pinned Qwen3.5-4B for
guidance preparation, self-review, student and EMA teacher. No27B or external
model calls. The user explicitly said "skip the mtp" after the runtime failures.
Use standard non-speculative vLLM decoding for future attempts; preserve all MTP
attempts as historical evidence and do not pool their baselines with plain runs.
Keep concurrency40, batch32,
32K context/4K replies,12288 episode output,8 turns for training. No truncation.

## Progress

- 2026-10-19 13:22UTC: compact r14completed exit0,16unique native episodes,
  all scored and repeatable; semantic qualification FAILED. Base success/partial
 75%/75%; guided teacher50%/56.25%. Sales success0->100%, HR100->0%,
  recruitment100->0%, marketing100->100% (destination-only assessment).
  Native tool-mistakes mean0.5->4.875. Independent audit saved as
  compact-guidance-four-v1-native-audit.json; no S/V symbol copied into tool
  arguments. Actual failures were guessed spreadsheet identifiers and display
  labels repeatedly passed as IDs, plus bare channel-name handling. No training
  restart. New compact v2adds deterministic capability-derived retrieval
  prerequisites: Drive discovery when no public spreadsheet ID, returned bare
  IDs and observed worksheet IDs, no unchanged failed-ID retries, exact public
  #channel name when tool contract supports names. Source conditions/checks/
  private-value boundary unchanged; no generated procedure or gold values.
 13compact tests pass (55with prior focused suite), Ruff/bash/diff pass.
  Accepted r15same16episode zero-update diagnostic at13:19:49UTC,
  Docker20bc21a2018fe60f5b08b831b68842ce1e3edddbd3c7bc063875a9e6420fc57d,
  capsule sdpo-guidance-five-20261019-experiment04-compact-four-r15,
  sourceb92d6531323dae48d10d8b6aae557c7a8f73318babd73d2109c3ad450ed5ac54.
  Current fresh base collection active. Same04deadline13:47:48UTC;
  main04/05not submitted, zero new optimizer updates. New worker awaits rollout
  cleanup explicitly. Runtime receipt alone never passes semantic/gradient gate.

- 2026-10-19 13:10UTC: user authorized deterministic compact guidance, inspect
  same four tasks, qualify before training. Owned r13preparation user-stopped
  and preserved exit143 at13:02:31UTC; zero optimizer updates/main04not submitted.
  New sdpo_compact_guidance.py compiles typed conditions/effects, explicit row
  ownership, joins, polarity/cardinality and exact headers without LLM paraphrase.
  Hidden IDs/answer literals live only in private audit; teacher sees stable
  references and wildcard schema locations to retrieve, not answers. Public
  request bindings are resolved rather than instructing invisible request.target
  reads. Historical context is deduplicated read-tool names only; no Luna args,
  outputs or generated workflow. Unsupported operators reject. Marketing's
  manifest covers delivery only, so this sheet cannot prove report quality;
  public task and retrieved policy remain necessary. Unresolved references are
  explicit and may still confuse the teacher; CPU compilation is not behavior
  qualification.54focused CPU tests/Ruff/diff pass, including actual four-source
  bindings/leakage, negation/joins, teacher wrapper and phone format provenance.
  Native qualification planned on8fresh baseline +8compact-guided TEACHER
  episodes, same frozen4training tasks/pinned base/greedy profile; zero updates,
  not SDPO learning evidence. Independent actual tool/source/outcome audit and
  teacher/actor token/gradient integration gates required before fitting.
  Same04deadline13:47:48.785806UTC;05window unstarted. No automatic extension.

- 2026-10-19 12:59UTC: audit-driven r13preparation accepted, Docker
  5993789b778db2aa3ea66d359e5e2311d675b1aa8378cbb8fb9d6d5548da9294,
  capsule sdpo-guidance-five-20261019-experiment04-balanced-preparation-r13,
  source-manifest f537e64d485db78b25f554d7ddb2e92170893c285acb603ed7c501bc86610b66.
  Deterministic source/population binding now explains request.id as the current
  population-record ID, not an email/API request ID. Private interpretation
  retains the exact source; text abstraction receives ownership meaning only.
  Cosmetic dataset->records normalization touches generated instruction prose
  only and records raw/normalized output before the unchanged validator. No
  source/check/ID edits or waived semantic gates.42focused tests pass, Ruff and
  diff checks pass; pinned tokenizer CPU preflight passed.14guide candidates
  re-reviewed and6regenerated; exactsame20data/core/deadline, fresh80native still
  mandatory. Identical-material CPU replay receipt referenced explicitly from
  r12, not newly claimed. No main04submission or updates;05unstarted.

- User-requested four-task source/output audit of r12is saved in
  teacher-guidance-four-task-audit-r12.json. Observed preparation, not learning:
  Sales misinterprets contacts-population request.id as an email request ID;
  HR weather invents an opt-in field and an extra employee email; marketing
  omits policy-email retrieval and verifies Slack via Gmail SENT; recruitment
  is blocked twice by the cosmetic word dataset. Some model-critic accusations
  are false/overbroad (public ss_timesheets allegedly must first be discovered;
  policy-supported channel called invented). Reviews are evidence to inspect,
  not correctness oracles. All20dependency blocks the feedback-only control
  as well as treatment; no optimizer gradient has occurred in these new arms.
  No running-source edit, task replacement or gate waiver made during audit.

- 2026-10-19 12:53UTC: r11exited1 at12:42:58UTC with15accepted5rejected
  guides and zero fitting updates. Source inspection found financial-close
  recipients absent from the public request, initial data and available read
  tools; historical Luna directly supplied hidden addresses. Perfect replay is
  not source observability. Before any update, replace only that Finance task
  with finance.year_end_audit_prep, whose public request supplies recipients.
  Protected-set exclusion and source SHA verified; all20actual pinned CPU
  invoke/capture/official-score replays pass on the amended cohort. Both arms
  must use this identical final20. Prior protocol preserved; exact evidence in
  balanced-finance-source-feasibility-amendment.json. No reward-based selection.
  Fresh full80native qualification is mandatory; original-cohort r10component
  cannot be inherited. r12staged immutable, source-manifest
  cfb1185ac652be2a0ed9c58839db133208c822fa451046574d77dff1aef8a137,
 13accepted guide candidates re-reviewed and7regenerated with stricter source
  feasibility/scope wording. Accepted r12Docker
  ad707fd2cc697e0c7b8d3606fc8cd45ef0b4166147d8f3263bf9af62eb4f2edc
  started12:51:52UTC; source verification/CPU preflight passed, model loading.
  Same04deadline13:47:48.785806UTC;05unstarted.
  No main04submission or optimizer updates. Fitting plus full readouts may not
  fit the remaining clock; no extension or shortened benchmark is authorized.

- 2026-10-19 12:40UTC: r10finished exit0 at12:36:32UTC and its complete
 40greedy+40stochastic native student component passed independent composition
  audit. Its guide qualification was nevertheless MANUALLY REJECTED: overtime
  harm scope became all at-risk names (contradicting the summary goal), Sales
  invented a note title/ContactId-in-email premise, HR allowed guessed locations
  or known-address fallbacks. Automatic approvals are not semantic proof.
  Rejections retained in r10/evidence/manual-guidance-qualification.json; zero
  fitting updates. No main04submission.
  New r11uses exact private source-record/policy provenance to interpret entity
  roles; every reused17guide receives a new grounded review and3are regenerated.
  Duplicate full-contract input was removed only where the same intact typed
  check/source definitions are explicitly present. All checks/source predicates
  and full-guide review remain.71focused tests pass. Entire unchanged r10native
 80component may be referenced only with actual source/model/data/decoding and
  file-hash proofs; no selected successful episodes, no inherited guide approval,
  and zero new episode claims. Composed main gates re-audit every referenced
  episode. This saves redundant student qualification after teacher-only edits.
  Accepted r11Docker b1ff37204d4237f0c6a43cff43b51db92cfa1acb7f0346b58e67788feda16a1f,
  capsule sdpo-guidance-five-20261019-experiment04-balanced-preparation-r11.
  Same04deadline13:47:48.785806UTC;05unstarted. Manual source review of final
  guides remains required before fitting even if every model reviewer approves.

- 2026-10-19 12:25UTC: resumed after interruption by inspecting retained state;
  r10was already accepted, never resubmitted. Docker
 293ec8847623faef1a0170a1d0855d89f4df9307a1da4405ca0a25ba5a11f192
  started12:23:39.259UTC, capsule
  sdpo-guidance-five-20261019-experiment04-balanced-preparation-r10,
  source-manifest a6e5676a0dacc893e194dad28b28a676079b5726cbb976ab8a7e10f8d158df63.
  r9exited1 at12:19:52UTC with18approved2workflow rejections: one lexical
  bookkeeping word and one private numerical policy shortcut/invented reads.
  r10reuses18exact approved bundles and regenerates those2; lexical feedback
  names offending words, privately observed numerical examples are hidden from
  procedural synthesis while public dates/IDs and full-source review remain.
 66focused tests pass. No guide/content or task-score gate was waived. All20
  approval plus80native qualification remain mandatory before fitting.
  Same04deadline13:47:48.785806UTC, zero updates/main04not submitted;05unstarted.

- 2026-10-19 12:16UTC: r8exited1 at12:13:43UTC with16approved4rejected
  guides, zero updates. Remaining failures still copied instance phrases or
  unlisted ws_/ss_ identifiers through abstraction/repair. New r9removes those
  strings from TEXT-only abstraction/repair inputs (not from typed first-stage
  predicates or full-source reviews). Public supplied IDs remain untouched;
  placeholders must become actual retrieval relationships, not guide text.
 65focused tests pass. Reuses16approved exact bundles with source checks;
  no task dropped, no threshold weakened, all20plus80native qualification still
  required. Accepted r9Docker
 03db70a53cf25727c609e5f3c4c8ca75b959b20b23c7048fdd3e83460efb1bce,
  capsule sdpo-guidance-five-20261019-experiment04-balanced-preparation-r9.
  Same04deadline13:47:48.785806UTC; main04and05not submitted. Preparation
  consumes04budget; no promise that10updates plus complete readout will fit.

- 2026-10-19 12:10UTC: r7exited1 at12:08:33UTC with15guides approved,
 5rejected and zero fitting updates. Manual source review additionally rejected
 2approved guides for invisible mailroom-summary binding/public channel-ID
  assumption; their tasks remain in the20and will be regenerated.13approved
  guide bundles are reused byte-identically with current source/model/manifest
  checks, never rejected outputs. New r8re-renders mentions predicates with
  explicit polarity and preserves effect_joins alongside operational dependencies;
  hidden binding prose now rejects.64focused tests pass. This is preparation
  repair, not reward/algorithm/data improvement. Actual new80native episodes
  and all20guide acceptance are still mandatory.
  Accepted r8Docker2c2574df682c8bf99ad36b0cbf6fac4c78cf80094fb97349dfe7532ef476c3ba,
  capsule sdpo-guidance-five-20261019-experiment04-balanced-preparation-r8.
  Same original04deadline13:47:48.785806UTC;05unstarted. Monitor exact owned
  preparation first, never submit main04before new-task gates pass.

- 2026-10-19 12:05UTC: r6common preparation exited1 at12:02:22UTC,
  zero accepted guides/updates. Retained66briefs,42rewrites,41check reviews and workflows
  show repeated copied instance literals;3complete-guide reviews were also
  rejected, including inconsistent approved=true with nonempty substantive
  issues. Gates remained strict. Diagnosis saved as
  experiment04-preparation-r6-diagnosis.json (exact stage outputs/source hash).
  New immutable r7 introduces source-limited abstraction after private typed
  interpretation; workflow receives approved abstract requirements and public
  tools/request, not raw references. Full-source semantic reviews remain and
  explicitly require approved=>issues=[]; no checks are deleted or waived.
 62focused tests pass. Accepted r7Docker
 9ddb9fae2bc5e35422ca99fe63835349151b19ae7af346aec82753422ddccd0b,
  capsule sdpo-guidance-five-20261019-experiment04-balanced-preparation-r7.
  Same20data/04deadline13:47:48.785806UTC; no main04updates/submission yet.

- Independent03native audit closed1100unique IDs across14populations, including
 310referenced shared-control IDs and790new episodes. All readouts passed their
  native repeatability gates. Independent saved-file hashes and496Adam parameter
  states verify moment10/student/teacher/EMA10. Read-only initial-state assertion
  audit finds heldout preservation-breaking episodes36->44of140applicable,
  broken checks71->94of920; unknown initial checks0. This is preservation of
  initially satisfied native assertions, not full harm taxonomy. Actual heldout
  tool error responses151->270 and length-limited generations5->11 increased.
  Higher success/partial credit therefore does not establish a clean overall
  improvement. Durable raw/reconciled audits live in experiment03-plain-r3/.
  No score zero-fill or frozen-environment repair was performed; transport retry
  count is unknown. These heldout outcomes do not change prespecified04/05.

- 2026-10-19 11:58UTC: balanced04common preparation r5 exited1 after all20
  guides were rejected; copied private instance IDs/answers survived attempted
  repair. All rejected stages/generations remain intact, no fitting/native
  qualification ran. New immutable r6 strengthens instance-to-retrieval wording
  and privately identifies copied strings for repair, retaining the same literal,
  semantic and complete-guide gates. No task/data/runtime changes; same04clock.
  Accepted r6 Docker0fc07ed5afc7452c07004941da240a460317aefed680ebcb8ffd1f1fefa0da4e,
  capsule sdpo-guidance-five-20261019-experiment04-balanced-preparation-r6.
  Never duplicate or mutate it. Main04remains feedback-only;05uses the final
  byte-identical accepted guide.05window unstarted. Source hashes/receipts are
  saved before launch; registry tracks preparation phases separately from fitting
  attempts, all within04's original13:47:48.785806UTC deadline.

- 2026-10-19 11:48UTC:03closed exit0 at11:39:46UTC,10updates and310post
  heldout episodes. Saved outcome curve reports success58.06->61.61%,paired
  task-bootstrap delta95CI[-2.26,+10.00]points;partial82.47->84.37%,deltaCI
  [+0.18,+3.98]points. Native identity/guard/checkpoint audit still pending;
  success improvement is not statistically resolved.04window started at actual
  local staging11:48:48.785808UTC, deadline13:47:48.785806UTC;05unstarted.
  New20 CPU replay r1 exposed helper assumptions: pinned Salesforce read tools
  return results/count or error without success. Actual pinned source confirmed;
  tool-specific strict parsing now preserves raw payloads and rejects malformed
  counts/unknown responses.9replay tests pass. Fresh CPU-only r2 preserved the
  same deadline/data and replayed all20 successfully with native capture and
  official scoring. No training update/model call in these replay attempts.
  Shared guide preparation plus fresh80native qualification is next, inside04.

- 2026-10-19 11:33UTC heartbeat: exact owned03 Docker remains running,
  not OOM-killed, with284 persisted post10heldout episodes at the follow-up
  check. The population is still open; no full heldout result is reported.
  Fitting remains10actual updates.04/05windows remain unstarted while03owns
  the GPU; its original11:44:00.822UTC deadline remains unchanged.

- User authorized a clean matched04/05context comparison on a better balanced
  cohort. Both drafts now explicitly select guidance_context feedback-only for04
  and taskwide for05; old20-task CPU receipts are labeled historical, not new-data
  qualification. Candidate source SHA/public-request/tool/manifest checks are
  complete for the proposed20; all canonical historical states equal current
  states. Native replay/model qualification/final training freeze remain pending.
  Common guide generation occurs inside04's deadline and is withheld from04's
  teacher, reused byte-identically by05. No further adaptive05algorithm change.
  Explicit historical quality/failed-assertion evidence goes only into private
  preparation and its literals enter output screening; non-private paths get
  quality counts only.54focused tests/Ruff/bash pass; current source hashes in
  candidate-new-task-qualification.json. Actual04/05capsules/windows unstarted.
  Owned03is evaluating post10heldout62 (30episodes observed, not a score/readout),
  checkpoint10 student/teacher/optimizer/RNG/manifest files exist. Do not interrupt
  it to start04 or infer job completion from completed fitting.

- Prelaunch CPU review caught an actual sparse-wire boundary in the future
  native qualifier: Agent.collect returns WireTaskData without initial_state in
  native-feedback mode. It now reads the persisted native archive and requires
  native_task_bindings exact identity/user/task reconciliation before state
  normalization. Native replay now uses collect_local_evidence and retains full
  capture envelopes, requiring count/tool/index/result agreement with each replay
  call; retained evidence is never overwritten. The Asana-specific workspace_info
  example is scoped only to tasks exposing asana_create_task, avoiding unrelated
  example copying into other tasks. General canonical/applied-field rules remain.
 49focused tests, Ruff and shell syntax pass; refreshed source hashes are in
  candidate-new-task-qualification.json. These are future-only changes, not actual
  new-task model qualification.03source remains untouched and live at7updates,
  step8collection.04/05windows unstarted. Previous goal turn was progress plus a
  verified wait on the exact live03Docker; current heartbeat resumes that same
  owned process, not a replacement submission.

- Owned03now6actual updates/moments/EMA, loss0.192464 at elapsed3376.01s,
  collecting step7. Previous goal turn was progress (native step5 diagnosis,
  tested new-cohort preparation path), not a blocker. No final gain claim.
- Changed-cohort composition is implemented separately in
  scripts/qualification/compose_sdpo_prepared_qualification.py. It requires
  unchanged inherited Python runtime sources (only explicitly named preparation
  helpers may differ), actual fresh80episode evidence, full40+40score coverage,
  original policy0/initial adapter/sampling, exact greedy repeatability and
  recomputed source/canonical-state binding. Old population labels cannot stand
  in for new tasks. It copies inherited capacity as inherited capacity and replaces
  only the separately proven native-task component, recording all provenance.
  CPU tests reject old tasks, partial populations, trained/other weights and
  changed runtime sources;45focused checks passed with Ruff/shell syntax.
  Future04preparation rules now explicitly preserve canonical parameter semantics
  and distinguish success status from applied fields, based on03TRAIN Asana.
  Both workspace and workspace_info are allowed schema keys; the failure is an
  incorrect semantic field assignment, NOT a schema-invalid tool call. The
  public x-automationbench-contract already lists results[].workspace as the
  applied field. This prompt refinement is an explicit context change.

  After03is terminal and the GPU is free, start04's window at actual staging,
  then run the isolated candidate replay IN THE PINNED OFFLINE CPU IMAGE:
  python /guidance/driver/replay_sdpo_training_candidates.py --materials
  /guidance/training-materials.json --tasks simple.gmail_reply_meeting_request
  simple.zendesk_billing_ticket --output /evidence/candidate-replay.json.
  Reject unexpected errors; successful historical paths are still only fixture
  compatibility. Then run preparation with native_task_qualification=true,
  parallel_checks=true, preparation_request_concurrency=40 and new cohorts.json.
  After successful preparation, mount immutable03read-only as /previous and run
  the composition helper in the pinned CPU image with --old-capsule /previous
  --preparation /guidance --output /evidence/composed. It writes bound guidance,
  normalization and an explicitly composed native/runtime gate receipt. Run
  preflight_sdpo_guidance_native.py on preparation/task-qualification/step-0000/
  online_train using its adjacent task-qualification/receipt.json as the bootstrap
  authority, bound guidance and exact new cohort; require all20original-prefix/
  action-ID/full-context and32distinct-action admission checks before fitting.
  Stage a new immutable main with those receipts and exact current sources;
  own initial/changed-adapter checks remain mandatory. Everything shares04's
  single deadline. These commands are a future procedure, not completed replay,
  generation or fitting evidence.04/05windows remain unstarted.

- Current03plain run reached5actual optimizer/moment/EMA updates at elapsed
  3077.39s; step6native collection is live. Step5loss0.206498 is a diagnostic,
  not reward improvement. Study20actual confirmed updates (01ten,02five,03five).
  Saved step5 original-decision/token/Adam review and native Asana receipts under
  experiment03-plain-r3/step5. Failed Asana replica used workspace_info instead
  of workspace in create_task; success:true response persisted that field but
  required workspace assertion failed. Its selected actions were later section
  placement and final text, with no risk flags. The other replica used workspace
  and passed. Current risk sampler is therefore incomplete; do not claim it
  fixes every failed decision. Competitor summary still includes excluded entities
  and teacher/Adam directions remain mixed; no local correctness qualification.
  Future05 may test a recorded supported semantic-parameter sampling mechanism
  after04 TRAIN evidence; do not silently add it to current03 or call04context-only.
- Future-only preparation extension now supports opt-in fresh native qualification
  for a changed20-task cohort within the SAME preparation server: all40greedy and
  40stochastic episodes, exact greedy repeatability, pinned initial adapter identity,
  current copied-constructor normalization checked against every task/replica.
  It uses64K serving like the training runtime while preparation requests remain
  capped32K/4K. No optimizer updates. Preparation launcher requires this mode for
  04/05; no old20-task native receipt may be relabeled. Native and actor capacity
  proofs remain distinct. New isolated replay helper uses actual invoke/capture
  and native deterministic state; expected bad search followed by valid recovery
  is retained explicitly.38focused CPU tests and Ruff/bash syntax passed; these
  are NOT pinned-runtime generation/replay qualification. No GPU job/capsule was
  started, and04/05windows remain unstarted. Actual new-cohort replay/preparation/
  qualification will count inside04's own2h window after03 releases the GPU.

- 2026-10-19 10:02 UTC: plain03-r3's own full evaluation-serving gate
  passed65536context,57344prefix,8192generated tokens and exact batch
  invariance with40peak requests, plus native tool/result/final continuation.
  It is now collecting its fresh TRAIN baseline. No optimizer update or
  learned outcome confirmed; actual32Kactor fitting gate is separate.
  Source manifest reverified unchanged after local future-preparation edits.
  Actual initial adapter independently matched the shared-control expected
  `e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c`.
  Initial serving receipt saved under `experiment03-plain-r3`.

- Future preparation optimization CPU-validated while03runs: independent
  check compilation is now opt-in parallel, with a GLOBAL40request semaphore
  across all tasks/stages, stable source-order results and every child awaited
  before rejection. Individual reviews, literal screens and final complete-
  guide contradiction rejection remain mandatory. Real TRAIN materials have
  42goal/18harm checks plus one excluded diagnostic; serial-per-task execution
  previously admitted at most20requests despite40server slots. A finite20task/
  60check mock witness reached40requests, completed60reviews/180totalrequests
  and returned to zero in-flight. This proves scheduling only, not actual GPU
  speed or generated quality.27focused tests, Ruff and shell syntax passed.
  Parallel-mode CPU tokenizer preflight now requires every check's first-stage
  input, not only the first check per task. Future04draft selects this option;
  its window and model generation remain unstarted. Evidence and source hashes:
  `candidate-parallel-preparation.json`. No accepted capsule was edited.

- CPU literal-boundary audit disproved a suspected date-field screening bug:
  the real Asana material accepts a declared dueDate instruction. One Sales
  task has source values also present in public API examples; that overlap
  alone does not license choosing a private policy criterion. No blanket
  whitelist was added. Manifest check coverage is not complete task-semantic
  or native guard coverage: for example Finance's manifest checks Slack
  delivery but public requirements and legacy outcome tests additionally cover
  email/report content. Exact60-check preservation cannot prove a correct
  generated guide. Actual guides/teacher decisions still require inspection.

- 2026-10-19 plain03-r3 ACCEPTED after02 was authoritatively terminal and
  real GPU compute empty. Run `sdpo-qwen35-4b-guidance-five-exp03-20261019-r3`,
  Docker `f640562727534b7923de4973e598a5e32608ac62e67f3a0c06615cf22d1c2fcc`,
  capsule `/var/lib/posttrain/qualifications/sdpo-guidance-five-20261019-experiment03-plain-r3`,
  source-manifest `900e40121f01b4c339cc7d4fbc46b30243f5a5746f1725f48c47572a83377d7a`.
  Its own fresh window began09:45:00.822UTC; deadline11:44:00.822UTC
  (16:44PKT), including actual staging/qualification/readouts. Standard
  decoding,40slots,batch32,10updates target; no new optimizer update confirmed
  at acceptance. CPU preflight on02's actual first online population passed
  40episodes/all20tasks,195complete actions,32distinct selected,5risk actions,
  unchanged original student/teacher tokens and no truncation. All272inherited
  runtime Python file names/bytes and the shared-base student fingerprint
  matched; an initial metadata inspection incorrectly assumed269root-level
  files and was corrected to full recursive equality without changing runtime.
  Shared plain base is bound; no repeated baseline generation claimed.
  Receipts under `experiment03-plain-r3`. Never duplicate or resume02.

- 2026-10-19 09:43:30.388UTC: plain02-r4 EXIT124 at its external cap,
  no OOM. Exactly5confirmed optimizer/EMA updates, moments1through5; no final
  checkpoint directory or poststudent greedy evaluation. Last loss0.256147;
  loss decrease is not established. Native audit found589unique episodes:
  TRAIN baseline40, heldout baseline310, five closed online40populations and
  39episodes from open step6. No native errors among these retained episodes.
  All five admitted batches had32unique actions/all20tasks and no truncation.
  Step6 has no closed outcome/admission/Adam record; do not infer a sixth
  update or score its unfinished population. Preserve as incomplete, no
  matched learning-gain estimate or exact crash resume. Terminal source/logs,
  artifact hashes and full population audit: `experiment02-plain-r4/terminal-audit.json`.

- 2026-10-19 09:28 UTC: owned plain02-r4 is still running with3confirmed
  optimizer/EMA updates and moment3; loss sequence0.229636,0.178121,0.200409.
  Step4 is collecting; no step10 student readout exists. Warm update intervals
  468.74s/427.42s leave insufficient room for the full ten-step target at this
  pace before09:43:59UTC. Keep the authorized cap; do not manufacture completion
  or reset it. Native collection wall time on first two online populations was
  357.7s/291.5s, versus median agent85.7s/76.4s; slower Marketing ROI episodes
  dominate those collections. This is a TRAIN timing audit, not a comparison
  across runtimes or evidence of reward gain. Prospective plain03 recipe frozen
  in `experiment03-plain-protocol.json` from prior TRAIN evidence only; actual
  preparation/window not started. Replaced the15-minute automation's stale
  historical status stack with a complete current sequential execution protocol,
  exact shared-control binding and remaining budget rules.

- 2026-10-19 09:16 UTC: plain02-r4 confirmed optimizer1/EMA1, batch32,
  loss0.229635864, gradient norm0.414398283, moment step1, LR1e-6. Online step2
  is collecting. First update admitted195complete actions; two whole exclusions,
  32distinct actions/all20tasks, no training truncation. Audited seven decisions
  from four TRAIN tasks and saved exact original action tokens/teacher advantages
  and actual Adam deltas in `experiment02-plain-r4/step1/decision-review.json`.
  Signals are mixed: the first embargoed-name occurrence in a bad Marketing
  write has negative teacher advantage but near-zero Adam movement; an unwanted
  adjacent-competitor name has positive teacher advantage and positive movement.
  This does not establish corrected task behavior. Exact selection replay
  reproduced all32 historical identities, then first-risk-frontier changed
  three slots: earlier Sheets typo, structured-error action and Asana create
  with undeclared due_date instead of later section placement. Proposed risk
  actions2to5 is a selection diagnostic, not reward or per-action credit.
  This independently supports03's predeclared selection comparison; do not
  alter the running02 control. Evidence: `step1/risk-selection-replay.json`.

- 2026-10-19 09:12 UTC: verified accepted MAIN plain02-r4 remains running,
  collecting fresh stochastic online training at step1; no optimizer update
  confirmed. Fresh TRAIN baseline closed40/40 (20tasks twice), zero errors,
  success50%, partial65.5556%; full heldout baseline closed310/310 (62tasks
  five times), zero errors, success58.0645%, partial82.4736%. All3396 heldout
  repeatability comparisons matched. These are untrained controls, not gains.
  Exported and independently revalidated common plain control at
  `/var/lib/posttrain/qualifications/sdpo-guidance-five-20261019-shared-plain-base02-r4/reference.json`,
  SHA256 `4ad42c5d5a10a35ca66d75acfeadd0823f6b55dd341c059ccd643b50355fc2b0`.
  Later arms may reuse only after exact student/runtime/decoding/initial-adapter
  fingerprint checks; referenced episodes are not new per-arm collection.
  Local baseline summaries and absolute task-bootstrap intervals saved under
  `experiment02-plain-r4`. Updated active15-minute queue with these facts.
  MAIN02 deadline stays09:43:59UTC (14:43:59PKT); remaining03–05 each receive
  their own fresh2h window starting at preparation, with10updates target,
  MTP disabled and40concurrency. No implicit retry renewal or duplicate launch.

- [x] At08:45:36UTC partial heldout baseline identity audit covered80completed
  records/16tasks:all names inside sealed62fixture,no duplicate episode IDs,
  no task above5repetitions. Actual fixture SHA
 5e492c7c2a3103db621f4b474e8b91b406ec68605479efc2b046717cdfe98a34
  matches the pinned launch input. Partial-tail handling explicit; no score or
  full-population/runtime qualification inferred. Evidence:
  experiment02-plain-r4/heldout-baseline-identity-progress.json.

- [x] Live observation08:41:13UTC(13:41:13PKT):accepted02-r4still running,
  heldout62baseline45/310native records, no closed baseline summary yet and
  zero optimizer updates. Do not export/reuse this control or publish a score
  until310identities/counts/coverage/native repeatability are reconciled.

- [x] At08:36:02UTC fresh MAIN02TRAIN baseline closed40episodes/all20tasks,
 40success scores, no native errors, success50%. Saved outcomes and native
  repeatability report under experiment02-plain-r4/train-baseline-*.json.
  These are baseline diagnostics, not update effects. Job remains running;
  full heldout62baseline precedes updates. No loss/reward-gain claim yet.

- [x] At08:33:01UTC(13:33:01PKT), accepted MAIN02-r4passed its own full
  evaluation-budget gate and entered step0 TRAIN greedy native collection.
  Saved full-evaluation-budget-qualification.json confirms65536context/8192reply,
 57344near-limit prefix, native tool continuation and exact40inflight batch
  controls. No optimizer update recorded yet. Do not confuse this capacity
  gate with closed heldout62 outcome evidence; fresh baselines precede training.

- [x] Bound the accepted02actual initial adapter to r17qualification:SHA
  e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c
  matches exactly. Actual saved inputs confirm speculation=none, explicit plain
  authorization,10steps/batch32/concurrency40. Actual fresh-base gate reports
  max_delta0 and allBmatrices zero. Evidence:
  experiment02-plain-r4/actual-qualification-binding.json. No updates.json yet;
  job remains live in initial long-context checks. No running-source edits.

- [x] CPU-only review of fresh plain r15greedy TRAIN cases saved with raw native
  episodes and input/output hashes in plain-four-train-baseline-{cases,analysis}.json.
  Asana used undeclared due_date while advertised schema/source assertion use
  dueDate; API responses succeeded but native scores0. Sales currently exhausted
  eight retrieval attempts with no write, unlike the old wrong-phone-write
  diagnosis; final repeated attempt has no response and is not a second observed
  empty retrieval. Competitor posts succeeded but score0/partial0.5, with distinct
  per-channel/conditional output requirements; Gmail complaint retained success1.
  These are pre-update TRAIN diagnostics only, not heldout adaptation or learning
  gains.02frozen control remains unchanged. Later guidance should preserve public
  API parameter names, positive/negative requirement polarity and output
  dependencies, while concrete runtime identities remain tool-retrieved.

- [x] MAIN plain02-r4ACCEPTED, run
  sdpo-qwen35-4b-guidance-five-exp02-20261019-r4, Docker
  a139b661d2714f216b0044b2cab2a71507583e0ba7de895892640f6f9b9f3436,
  capsule sdpo-guidance-five-20261019-experiment02-plain-r4, source
  402879e8cdac61e13c11ef77ed69609dcb8e3e275dc8e28a5bbd07925efd8819.
  10updates/batch32/concurrency40/standard decoding; feedback-only teacher
  context and first-error-frontier frozen control. Fresh native baselines run
  before optimization. No optimizer updates confirmed at acceptance.02deadline
  remains09:43:59.853UTC(14:43:59PKT); qualification/preparation time not reset.
  Monitor exact owned job; no duplicate/replacement or running-source edits.
- [x] Independent r16capacity component exited0 at08:15:08.862UTC. CPU-only
  r17composition passed after recomputing native84pair audit, reconciling80
  unique episodes/20tasksx2 each population/all success+partial scores/no native
  errors, same initial adapter e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c,
  and269identical runtime Python files. Native r15exit124 remains explicit
  provenance; r16real32Kbackward/unchanged weights supplies the capacity component.
  Exact source manifests/raw population hashes bound into r17receipt. Eight
  rejection tests cover wrong weights/sources/image, failed capacity, missing
  scores, forged repeatability and invalid gradients;42combinedfocused checks
  passed with Ruff. Composed qualification is not learning or heldout evidence.

- [x] Capacity-only r16ACCEPTED, Docker
  b4c87bc91cadad71d481040923567af0dff11e475f7f897e35a073c25848da82.
  Monitor this exact owned job; r15is terminal and must not be resubmitted.
  Component-binding audit/receipt remains required before any training launch.

- [x] r15terminal08:05:56.504UTC exit124(noOOM), zero updates:both40episode
  populations closed/scored without native errors,84greedy comparisons passed,
  long-prefix parity0.12501049<0.15.850second sub-timeout occurred before
  capacity/backward receipt; preserve r15as INCOMPLETE, never label it passed.
- [x] Independent capacity component r16prepared, source
  1f4760af788a72ba8791a2d7fa14aa07bb7c144b48bc6ed0080abd82626800ed.
  It uses r15's exact runtime driver snapshot, same pinned image/model/adapters,
  seed18/LoRA16/alpha32,plain compiled40slots/cacheoff/unchunked. Repeat the
  full28,672prefix+4096target rollout/40batch controls and32Kfinite nonzero
  backward with unchanged weights.600second sub-cap stays within02's ORIGINAL
  renewed deadline09:43:59.853UTC. This is zero-update component composition,
  not a checkpoint restart, gate relaxation or new experiment/window.
  Before accepting a combined qualification, independently verify r15native
  IDs/tasks/scores/repeatability, matching initial adapter hash and runtime
  source files against r16and both exact non-speculative operating points.
  Preserve component states explicitly; no global r15pass claim is permitted.

- [x] Plain native r15greedy TRAIN closed40episodes/all20tasks,40success-score
  coverage,no native errors,success50%; native repeatability passed84pairs.
  Stochastic collection is live. These are fresh-base diagnostic results, not
  learning gains. Raw outcomes/requested/worker diagnostics/repeatability copied
  to plain-native-r15 local evidence. Backward and complete receipt still pending.

- [x] Plain native r15live at07:54:50UTC with22/40greedy episodes recorded;
  stochastic/backward checks still pending. Prepared experiment02-plain-r4-draft
  from its frozen feedback-only/first-error control, explicitly speculation=none
  and renewed window; no learning submission or fabricated qualification.
- [x] Later private-checkwise preparation now explicitly supports standard
  decoding in both generation module and launcher. New preparation.json must
  bind logical04/05, approved window and hashed genuine plain gate request;
  launcher passes --speculation none/--plain-decoding-authorized, generates
  fresh private-checkwise guidance rather than reusing old accepted contexts,
  and recomputes remaining GPU time after bounded CPU preflight.33focused tests,
  Ruff and shell syntax passed. No edits to running r15capsule. New snapshots
  must include these local preparation changes when04is actually frozen.

- [x] Plain mixed r14completed exit0 at07:50:47.652UTC:all16full8192mixed
  token/logprob comparisons exact,40inflight, native call/result/final passed;
  actual requested speculative_config=null. Raw receipt/requested/worker
  diagnostics and compact audit retained in plain-mixed-r14. This is not yet
  full native/backward qualification or any learning result.
- [x] Accepted plain native qualification r15 at07:51:45.907UTC, container
  a8c103636d8332f82ee6cd06451dbc4f065405a707c5005909ff55741a988d10,
  capsule sdpo-guidance-five-20261019-plain-native-r15, source
  18ab8448ff62e714cdf27179e005e47b5e262a1f1eb96a7fb0d6ee90f7000ab2.
  It requires40greedy+40stochastic frozen TRAIN episodes and32Kbackward under
  plain compiled Triton40slots/cacheoff/unchunked.850second soft cap and02's
  inherited fresh deadline. Monitor this owned run; no duplicate qualification.
- [x] Existing15minute heartbeat monitor updated/reactivated for the authorized
  plain four-experiment queue; no unrelated automation changed. User budget
  decision recorded and retry-window guards passed27focused tests. No new
  optimizer updates yet; historical01still accounts for10/50actual updates.

- [x] NEW USER AUTHORIZATION supersedes prior budget blocker:remaining02/03/04/05
  each receive a fresh2h window; qualify without MTP, then10updates each. Keep
  five learning experiments/50actual-update ceiling and all numerical/native
  checks. No reinterpretation of zero-update failures as completed experiments.
  Explicit renewal code is limited to remaining plain-decoding arms; retries
  inherit their renewed deadline.27focused tests/Ruff/shell syntax passed.
- [x] First fresh window started2026-10-19T07:44:59.853UTC(12:44:59PKT),
  bounded deadline09:43:59.853UTC(14:43:59PKT), leaving the existing safety margin
  inside7200seconds. Qualification counts in02's window.03–05windows start only
  on their own recorded preparation/qualification, not simultaneously.
- [x] Accepted plain mixed-prefix qualification r14, container
  fdfe136b31c5fbedc5ba435e7da8a59860e7a1f238c0bb65c465ebfaa2ed1705,
  source3d75b42d4c2a7504140552c81698a1ae55e1ef86ed22ae03ea46dca3ea57bd71.
  Non-speculative compiled Triton40slots/cacheoff/unchunked, exact retained
  four8192reply prefixes plus16heterogeneous background prefixes. No training
  updates. Monitor this owned job, then require actual40greedy/40stochastic and
  full32Kbackward before02training. Historical MTP controls are not plain bases.
  Keep the old blocker entries below as history; they no longer require input.

- [x] User amendment:skip MTP. Prepared explicit standard-decoding support in
  the guidance-five profile and launcher, with opt-in user authorization and
  original requested-config evidence verifying speculative_config=None.
  MTP receipts cannot qualify plain decoding. Twenty-one focused tests passed,
  plus Ruff and launcher shell syntax. CPU preparation only: no new GPU run or
  training update. Historical plain r3passed only short/tool gates and is not
  full native/32Kbackward or matched-heldout qualification. Remaining time-window
  permissions are not inferred from the instruction to remove MTP.

- [ ] Execution blocked awaiting explicit budget decision after repeated strict
  runtime failures and exhausted02/03windows. Last remote inspection05:26:47UTC:
  r13terminal exit1; no running containers or GPU compute processes. Actual
  campaign remains10/50updates; five completed learning experiments are NOT
  established. Requested one additional bounded2h MTP repair window and fresh
  original2h windows for zero-update02/03 after qualification, retaining all
  numerical/repeatability gates, five-experiment ceiling and50update ceiling.
  No approval received. This supersedes the older request for a noise-control
  protocol as the recommended next action; that alternative remains disabled.
  Do not consume04/05as unrecorded runtime extensions or claim failed zero-update
  qualification attempts completed the learning goal. Resume only within the
  user's actual decision; keep every failed capsule and source/score evidence.

- [x] r13FAILED exit1 at05:25:26.204UTC, zero updates. Eager MTP2 gives
  exactly the same actor AND rollout selected-token scores as eager MTP1 r12
  on the16token long probe: max discrepancy0.4999885559nats. Compiled MTP2 r9
  had identical actor scores but max discrepancy0.1250095367. Thus speculative
  depth alone does not repair this probe; eager execution is implicated by
  this historical comparison, but its internal numerical cause is unproven.
  Compiled MTP2 still failed full-native repeatability, so returning to it
  does not qualify the experiment. Preserve long-parity-r13/comparison-audit.json,
  terminal files and all failed gates. No thresholds changed. Original03window
  is effectively exhausted; do not renew it or consume a later learning arm
  as an unrecorded diagnostic extension. Pending user time/protocol decisions
  remain unresolved. Campaign completion remains10actual updates, not50.

- [x] Cross-runtime long probe audit shows actor scores bit-identical between
  compiled MTP2 r9 and eager MTP1 r12, while rollout values differ. This narrows
  the mismatch to rollout-side operating points; speculative depth and eager
  execution are confounded. No causal MTP-only conclusion is warranted.
- [x] Accepted120second zero-update long-parity diagnostic r13, container
  ed592b73114883ea4117ff766b219239db109494d28709873339b8aa42251b56,
  sourceedc50d51028641f79fc8ae7b70fa1b2e1225fa2327be93b3382c6518aa9e4922.
  It tests MTP2/eager/Triton on the same28,672prefix and16target tokens as r12,
  with65536engine context/40slots/cacheoff/unchunked and the same fresh adapters.
  Existing0.15nat gate remains unchanged. It performs no native populations,
  learning or global qualification.120seconds+30kill stays inside03deadline.
  New local diagnostic source passed Ruff/byte compilation; running capsules
  remain immutable. Monitor r13exactly and preserve its outcome.

- [x] r12FAILED exit1 at05:21:36.298UTC, noOOM, zero optimizer updates.
  Both native populations closed40episodes each; greedy audit passed80pairs.
  It then failed the near-limit28,672-token-prefix selected-token parity check:
  max actor/rollout discrepancy0.4999885559nats exceeds unchanged0.15limit.
  The32Kbackward check was never reached, and no complete qualification receipt
  exists. Preserve long-parity-v0.json, terminal log/inspect and native outcomes
  under mtp1-native-r12. Finite r11pass does not override this failed gate.
  Thus neither current MTP2full-native repeatability nor MTP1long-prefix scoring
  is fully qualified. Do not silently relax thresholds or start actor training.
  Original03deadline remains05:26:50UTC; fresh02/03windows and an exploratory
  repeated-control protocol are still pending user choices, not approved.

- [x] r12greedy TRAIN closed40episodes by05:19:04UTC; native repeatability
  passed80same-input comparisons. Stochastic40episode collection started.
  Raw audit/requested/outcomes copied to local mtp1-native-r12 evidence;
  backward and complete receipt still pending. No new optimizer updates.

- [x] CPU-only launcher preparation permits explicit MTP depth1 or2, with
  depth1 requiring hashed original requested-config artifacts for both mixed
  and native/backward qualification. It rejects mismatched depth/concurrency/
  context/cache/chunk/eager settings rather than treating depth2 evidence as
  depth1 qualification. Ten focused speculation tests, Ruff and shell syntax
  passed. No existing immutable capsule was changed; this enables a future
  explicitly recorded depth1 runtime treatment only after its gates pass.
- [x] r12was live at05:15:43UTC and had entered greedy native TRAIN collection;
  short parity/batch and native tool continuation artifacts exist. Actual
  trajectory and backward qualification still pending, zero optimizer updates.

- [x] r11completed exit0 at05:12:39UTC:16full8192 mixed comparisons had
  identical tokens/logprobs, peak40inflight; native call/result/final gate passed.
  This is finite-case evidence, not the full310episode operating point.
- [x] Accepted next strict zero-update native MTP1 qualification r12,
  container7a01490cececa591017fbaa5c643035585f0f23fea386303ab97dbfd958355ca.
  Source736125e8956bf6ae6e42a95935275e83a2ef08e7aa3a208a4a1dafc42a035524.
  It requires actual40greedy+40stochastic frozen TRAIN episodes and32Kbackward,
  with Triton/eager/cacheoff/unchunked matching r11. Its650second soft cap plus30
  kill is inside the existing03deadline; no renewed window or gate relaxation.
  Learning still totals10updates. Monitor this owned run, never duplicate it.

- [x] Accepted strict zero-update native MTP1/Triton/eager qualification r11
  at05:08:39UTC, container6123f2a6bcec6b380813583503b5043b788c9076990d335942b0ed8d64b8dc6c.
  Four retained failing prefixes6772/6391/9608/10236 tokens retain8192reply
  budgets with16heterogeneous background prefixes,40inflight. Source
  ada191619f30254d9375b705d188a3916d92b55a23390723e5a37460926c6362.
  This tests speculative depth1 after depth2failed full native repeatability;
  it changes no hard gate, performs no updates and renews no time window.
  Bounded650seconds plus30kill stays inside03's05:26:50UTC deadline.
  Monitor this exact owned container; finite qualification is not full native
  repeatability or learning evidence. Pending user choices remain unanswered.

- [x] Completed CPU-only support/Adam audit of all320selected transitions in
  historical01's10actual updates, joined by original task/generation identities
  and verified moment/EMA counts. Native tool-token teacher mass on the
  student's top20averages99.4561%; no importance weights hit the upper2cap.
  Nominal transition-normalized target coefficient mass is55.28%reasoning,
 29.97%native tool action,14.32%final answer,0.44%closing marker; this is NOT
  actual KL magnitude or gradient share.35selected actions had factual API
  errors. On their native tokens, teacher sampled advantage averages-0.03179
  and actual Adam logp change-0.00002576; these local averages do not establish
  valid repairs or improved trajectories. Rare correction-token bottlenecks
  remain possible despite high aggregate support. Evidence:
  ten-step-teacher-support-audit.json plus source/hash execution receipt.
- [x] Independently checked the actual pinned TRL loss file SHA
 530b571c0001bbf85bb0c78725d32645a4d7155660e22e41580fab5c9f7041e4
  on CPU:alpha1matches KL(student||teacher),alpha0matches the forward direction,
  and student-topk-plus-tail reverse KL and gradients match manual calculations
  on the small tested tensors(max gradient discrepancy3.33e-16). No loss
  direction/sign error found in this check. This does not qualify teacher
  semantics, every tensor regime or native inference. Receipt:
  pinned-loss-algebra-cpu.json. No new model calls/optimizer updates/pin changes.
  Protocol/time questions remain unanswered; no additional GPU launch made.

- [x]03-r2terminal FAILED04:49:10.404UTC,exit1,noOOM,zero updates. All310
  baseline episodes scored, but repeatability failed47/3165pairs across1814
  complete calls. Among mismatched pairs39share the final success value and8
  differ; pairs are correlated, not a causal error-rate estimate. Two of62tasks
  have mixed success across their five replicas. All raw artifacts preserved;
  no strict qualification or learning gain. Source:
  experiment03-r2/zero-update-runtime-outcome-dispersion.json and terminal files.
- [ ] User decisions requested and pending:retain token-identical hard stop or
  explicitly exploratory matched repeated-control studies; allow fresh2h
  windows for still-zero-update02/03 or retain existing deadlines. This replaces
  the earlier02-only time question. No answer means no authorization to change
  the gate or reset a window. No new learning job submitted while these choices
  are pending. Existing windows/failures remain preserved; campaign still10/50.
- [x] Concrete candidate protocol saved in exploratory-noise-control-proposal.json:
  two complete independent matched310episode zero-update controls,10updates,
  full310episode student readout. No runtime failure is promoted to a strict
  pass. CPU-only descriptive analyzer keeps control drift, comparisons against
  both controls, absolute uncertainty and missing-outcome bounds separate. It
  requires matched runtime/base weights, disjoint930readout episode IDs, exact
 62x5coverage and joint-complete sensitivity; baseline-pair uncertainty remains
  independent of missing student scores. Four focused analysis tests pass;
  eight combined analyzer tests and Ruff pass. No observation-mode driver or
  GPU execution was enabled; protocol/budget approval is still required.

- [x] Verified an actual future-preparation input gap:historical match scope
  was stored in each cohort row but absent from material passed to the model.
  Future checkwise preparation now receives current-source-hash-validated,
  typed per-reference match metadata at every generation/review stage; false
  and unknown remain distinct. Every example remains procedure-only, even when
  states match; provenance must not become an actor instruction. Binding is
  recorded in preparation stages/output lineage and never supplied to student
  rollouts. All20frozen source bindings validate(12state matches/8mismatches).
 34focused preparation/context tests and Ruff pass. No running source changed.
- [x] Actual pinned-tokenizer CPU preflight of the new private-checkwise inputs
  PASSED all20tasks with verified reference-scope metadata at first stage;
  no model calls/updates/GPU allocation. Receipt:private-reference-cpu-r3/preflight.json,
  source e0342b26371f5d42a1e8c57111296cd5ca47811be8ce3479cab6a9685895db3e,
  Docker bbdc9ed36bbe9ddb8766074cfd7c6eabfe2ead3d2a56a2c3b16e476c0d771355,
  exit0. This qualifies input/schema/context bounds, not generated guide quality.
 04draft explicitly identifies a context-package treatment, not an isolated
  causal ablation of each preparation mechanism.

- [x]03-r2eager TRAIN baseline closed40episodes/all40scored/no errors.
 186complete greedy calls/80identical-input pairs/zero differing outputs.
  Fresh untrained control success47.5%,partial63.5443%; use this runtime's
  matched base, never treat changes from rejected compiled runs as learning.
  Actual MTP acceptance89.3565%(57290/64114draft tokens),162.3805s interval;
  acceptance is not a matched speedup claim. Heldout baseline now collecting.
  Receipt:experiment03-r2/train-baseline-closed.json. Zero new updates confirmed.
- [x] Posthoc teacher-direction audit now includes available actor-versus-
  rollout logp differences and importance-weight saturation separately by
  original reasoning/native-action/final regions. Missing vectors stay unknown,
  not zero. Seven direction/alignment tests and Ruff pass, including a test
  that excludes reasoning/marker tokens from native-action discrepancy stats.
  This is outside all running capsules and changes no optimizer behavior.

- [x] Final-comparison analyzer now independently recomputes native runtime
  evidence and verifies saved audits. Failed, missing or undeclared readout
  gates remain explicit; qualified paired benchmark comparisons are withheld
  when either arm lacks qualification. Raw native metrics/missingness remain
  inspectable; execution completion and runtime-qualified readouts are separate.
  Nine focused analyzer/shared-control tests and Ruff pass. Real terminal03-r1
  replay confirms350closed episodes/zero updates, with only the heldout baseline
  runtime rejected(70differences); no benchmark completion claim. Evidence:
  experiment03-r1/runtime-aware-native-audit.json. No running source changed.

- [x]03-r2full evaluation-budget gate PASSED with65536context/8192reply,
  native tool/result/final continuation and exact near-limit concurrency
  witness. Fresh TRAIN baseline collection has begun(4open native episodes at
  observation); no optimizer updates confirmed. Receipt:
  experiment03-r2/long-gate-and-stage.json. Actual Docker timeout arguments
  --signal=TERM --kill-after=30 4172 independently confirm retained logical
  deadline enforcement rather than a fresh two-hour timer. Continue this
  exact owned job through complete populations and pre-update backward gates.

- [x]03-r2actual initial configuration verified from saved inputs and worker
  diagnostics:enforce_eager=True,zero captured graphs. Native original-prefix
  and tool/result/final gates pass, actor/rollout discrepancy0.02213nats.
  Receipt:experiment03-r2/initial-runtime-gates.json. Full near-limit gate still
  live at latest observation; no update/pass of full populations inferred.

- [x] Fresh03-r2 ACCEPTED after eager r10 passed and exact current-source CPU
  admission passed198rows/32distinct selected/all20tasks. Run
  sdpo-qwen35-4b-guidance-five-exp03-20261019-r2,
  Docker d2372486440bc07e0bceb81f06184bad4991cf1ea207e99e47d516c11a0466a7,
  capsule sdpo-guidance-five-20261019-experiment03-r2, source
 7382b99ef7701705d75991007fc060f948ac357d93dbec2f1c1f9d59930306a2.
  Existing05:26:50UTClogical deadline inherited and enforced, no budget reset.
  Eager execution is the only runtime change; learning/data/feedback-only
  treatment and risk sampler remain03's frozen recipe. Fresh complete native
  baselines, near-limit serving,32Kbackward and changed-adapter gates are still
  mandatory in this job; r9actor receipt is historical reference and r10is a
  finite mixed/native witness, not a replacement for those own checks. Zero
  optimizer updates at acceptance. Preserve03-r1 and monitor this exact run.
- [x] r10 terminal exit0:all16full8192mixed comparisons exact, actual native
  continuation passed, actor/rollout parity max0.02213nats,40inflight reached,
  zero captured CUDA graphs,zero updates. Receipts copied under native-eager-r10.
  This is a bounded numeric counterexample repair; universal/native-full
  repeatability remains unproven until03-r2completes its full populations.

- [x] Precise argument-direction auditor now supports native Qwen XML parameter
  names as well as JSON keys, checks exact tokenizer roundtrip/offsets, ignores
  reasoning-region mentions and refuses per-tool attribution if multiple calls
  own the same field. Six focused direction/alignment tests and Ruff pass.
  CPU replay on actual01/update1/32rows exited0; r1's JSON-only unavailable
  mapping remains preserved, r2 maps native sheettitle tokens. At its title
  subtoken the teacher sampled logp advantage is-1.6629 but actual full-batch
  Adam delta is+0.01473; other field tokens differ in direction. This local
  mixed direction is not proof of a coherent repaired call or task outcome.
  Native schema declares spreadsheet,title,spreadsheet_id,drive; a pointwise
  preferred token cannot be presented as a valid whole alternative. Evidence:
  argument-diagnostics-cpu-r2/teacher-directions-step1.json. No model calls,
  GPU allocation or new optimizer updates in these retrospective diagnostics.

- [x] Native-eager-r10 ACCEPTED, Docker
 1fe27a5842a42060671980ffa75425e60a4cd71b88f84c99b69de879266110e0,
  run sdpo-qwen35-4b-guidance-five-native-eager-20261019-r10,650s cap within03's
  unchanged deadline. Source c836847f7577896c8b3a978ffa56a99c3e478a97dfe154c03c80675cd8e41a3a.
  New numeric-only witnesses7584/8826/5931/8719tokens retain8192reply caps;
 16background prefixes565–23554tokens retain64token diagnostic probes.
  Fixture14271a843bcb33de48d37743baaf48a8f8c4496536ca13ce7f5f3904a7a3e5d6.
  Exact owned process running; never duplicate. No optimizer updates or native
  qualification pass yet. Source/config receipts under native-eager-r10.

- [x]03-r1 terminal FAILED04:07:24.709UTC,exit1,noOOM,zero optimizer updates.
  All310baseline native episodes were retained, but full greedy repeatability
  failed70/3189identical-input pairs across1958complete calls. Cacheoff and
  unchunked prefill are insufficient. Preserve every artifact; this is a runtime
  rejection, not teacher learning evidence. No qualified shared control exists.
  Terminal state/log and full audit copied under experiment03-r1.
- [ ] Evidence-backed zero-update native-eager-r10 repair planned under03's
  existing05:26:50UTCdeadline (650s cap, no reset): retain MTP2,GDN Triton,
 40slots,BF16,pins,cacheoff,unchunked and full8192critical replies; disable only
  CUDA-graph execution. Reproduce four fresh numeric-only failure witnesses
  under heterogeneous40request pressure, unchanged parity/exactness gates.
  Passing this finite gate would NOT authorize calling the full runtime fixed
  or skipping complete-native/actor/changed-adapter qualifications. No learning
  restart or source mutation to a retained attempt.

- [x] CPU historical sampling replay reproduced all320actual selected
  identities across01's10saved rounds before comparing the new selector.
  On those same saved admission populations first-risk-frontier changes16slots
  across five tasks, increasing selected risk-cue actions24to39. Seven changed
  slots select earlier Asana create calls with undeclared due_date instead of
  later section actions. These cues remain non-causal metadata, and changed
  earlier learning would itself change later trajectories: this is NOT an
  estimate of counterfactual reward, teacher correctness or gradient gain.
  Evidence:sampling-counterfactual-r1.json and its source-hashed execution
  receipt. New audit requires exact saved admission and old-selection replay;
  two negative provenance tests and Ruff pass. No GPU/model/update calls.

- [x] Future context-candidate student-source fingerprint exactly matches
 03's sealed runtime/student methods, initial adapter hash, native decoding,
  task budgets and message/population helpers. CPU evidence:
  status-candidate-student-compatibility.json. This can support a later shared
  untrained control only after03closes and passes its full native audit; no
  open/rejected control was exported or reused. Teacher-side changes remain
  an explicit treatment, not part of student0context.
- [x] Ruled out a proposed tool-attribution repair on the actual retained r9
  stochastic TRAIN data:207/207native tool messages have names and call IDs.
  No speculative attribution change added.03exact container remains live,
  heldout baseline71/310closed native episodes at latest bounded observation,
  no applied updates confirmed. Continue owned run, never duplicate.

- [ ]04context draft saved as experiment04-context-draft.json; not frozen or
  submitted. It proposes same4Bprivate-checkwise guidance, intact source-check
  interpretation and whole-bundle review while retaining03's risk sampler and
  factual hindsight. It requires actual03TRAIN/teacher evidence before recipe
  selection, source-separated generation/admission, native-state binding and
  an explicit own total deadline including preparation. Heldout semantics and
  outcome scores are not used to choose its context.05remains adaptive.
- [x] Original02deadline03:48:05UTC elapsed without a fresh-time approval.
  No02restart was submitted. Its zero-update failures stay preserved; this
  does not prevent the independently authorized03–05work. Latest03live poll
  confirms the exact owned container running, baseline heldout34/310native
  episodes still open, zero applied updates confirmed; no partial-score claim.

- [x]03closed TRAIN baseline:40native episodes and native greedy repeatability
  passed195complete calls/84identical-input pairs/zero differences. First actual
  native MTP counter interval is available:34471drafts,68942draft tokens,
 60706accepted draft tokens, acceptance88.0537%,161.3235seconds for this baseline
  collection interval. This is measured acceptance, NOT an end-to-end speedup
  or learning gain; no matched non-MTP native timing control exists. Evidence:
  experiment03-r1/train-baseline-runtime.json and train-baseline-outcomes.json.
  Fresh heldout baseline has started; zero optimizer updates confirmed.

- [x]03's full evaluation-budget qualification now PASSED:65536context,
 8192reply,native call/result/final continuation and the exact batch witness.
  Its original native-prefix gate also passed. Live closed-file observation
  subsequently saw38baseline TRAIN episodes still collecting, zero optimizer
  updates; do not treat that partial population as a score or learning result.
  Receipt copied to experiment03-r1/initial-qualified-stage.json. The exact
  owned run remains running; fresh full native heldout baseline follows.

- [x] Status-only candidate now has an explicit future-only context selector
  feedback-status across teacher construction, CLI and CPU preflight. Actual
  zero-GPU replay passed on all40retained stochastic qualification episodes:
 198admitted transitions, one whole reply-limit exclusion,32distinct selected
  actions covering all20tasks, maximum teacher+target15263tokens. Student
  prefix/action identities and task coverage remain exact. Capsule
  status-projection-admission-cpu-r1, source
  cb9120e2597b556b40309ab28bf29d5d73e4f9928e027fe84d7248664d420bf8,
  Docker a667f3422b68ca2488745ee976a966ddfc9bf69d7a59e7b05a47449478d1fd67,
  exit0; receipt status-projection-admission-cpu-r1/preflight.json. No model
  calls or optimizer updates. Native findings keep their original missingness;
  only raw observation payloads/assertion parameters/excluded rows are projected.
  No03source change and no future learning arm accepted from this candidate.
- [x] Future same4Bprivate-checkwise preparation can explicitly select the
  qualified cacheoff/unchunked runtime instead of silently returning to the old
  defaults. New CLI flags are opt-in; historical defaults stay intact.32focused
  preparation/context tests and Ruff pass. No GPU preparation or generation
  has occurred for this newer recipe; source-format tests are not guidance
  correctness evidence. Freeze the next actual arm only after03TRAIN evidence.

- [x] Further CPU-only review preserves four actual consumed TRAIN decisions
  from experiment01/update1, with exact student prefixes, generated actions and
  teacher context:teacher-context-four-training-cases.json. The companion
  teacher-context-four-training-audit.json finds post-action result strings not
  present in original messages in three cases (including newly created Asana
  IDs); Sales also contains nine excluded legacy assertions in its teacher
  payload. All four have zero decided native goal/harm instances. These are
  context-design observations, NOT evidence that the model copied those values
  or that a whole-episode failure makes a selected action incorrect. The
  selected Sales phone update itself succeeded; its missing note is a separate
  decision. Experiment01 runtime variability still limits causal conclusions.
- [x] Prepared a separate, unselected status-only hindsight candidate in
  scripts/qualification/sdpo_hindsight_projection.py. It removes entity payloads
  and assertion parameters, omits excluded assertions while recording their
  count, and distinguishes reported error, reported success and successful
  empty retrieval without inventing action-correctness labels. Four focused
  CPU tests and Ruff pass. It is not wired into03 or any submitted job; local
  validation proves data boundaries only, not useful teacher gradients or
  improved rollouts. Consider it as a separately declared future context test
  after inspecting actual03training decisions. No running source changed.
- [x] Live03recheck: exact Docker remains running with initial native prefix
  receipts present and real actor/vLLM GPU processes. Full evaluation-budget
  qualification is still in progress; no optimizer update confirmed. Campaign
  actual applied total remains10/50.02fresh-time request remains unanswered.

- [x]03 ACCEPTED03:34:21.587UTC for the already-authorized third ten-update slot while
 02's fresh-time question is pending. Run
  sdpo-qwen35-4b-guidance-five-exp03-20261019-r1,
  Docker282cc2d167dad9f2ba9c7d4dd92eb2e9ae07d8ae541c06ba4e1295a774d7818c.
  Loading confirmed; zero new optimizer updates observed. Do not duplicate.
  Only training-row selection changes to
  first-risk-frontier; feedback-only teacher shell,student prompts,tokens,loss,
 20tasks,batch32,core hyperparameters and qualified runtime remain matched.
  It is motivated by01's nine missed silent Asana calls, not nonexistent02
  learning results or any heldout outcomes. No isolated03-vs02 causal claim is
  possible until a qualified02learning control exists; own fresh-base comparison
  remains valid if all native runtime gates pass. Source
  c2353c209c626c51c1c186be6e2bc19280d98f187baf941f5c049beb4727a5d0,
  capsule sdpo-guidance-five-20261019-experiment03-r1. Deadline05:26:50UTC
  includes its dedicated CPU preparation plus a shutdown margin, no reset.
- [x]03 bootstrap CPU preflight PASSED with explicit r9 qualification receipt
  binding:198admitted,one whole reply-limited exclusion,32unique selected/all20
  tasks,unchanged student/teacher tokens and admission. One selected decision
  changes in this bootstrap:email campaign Sheets query with undeclared snippet
  instead of a later decision. Two selected risk cues; Asana has no risky call
  in this particular sample. Cues are not correctness labels. Evidence:
  risk-admission-cpu-r1/preflight.json. Source35ee46098b4d7bcb9f23cf475ceed31d2a023dcd23ac076a62f2526dc589af1b.
  Tests reject implicit qualification/training relabeling and wrong receipt/task
  binding. No updates in CPU replay. No02budget exception authorized.

- [x] r9 PASSED exit0:80native episodes(40greedy+40stochastic),84greedy
  repeated-input pairs exact, fresh32768token rollout/backward finite with
  nonzero gradients and unchanged weights. All qualification steps are zero
  optimizer updates. Receipts:unchunked-native-r9/receipt.json and capacity files.
- [x] Consistent-source CPU admission r2 PASSED on the actual40stochastic
  native episodes:198complete transitions admitted, one whole reply-limited
  transition excluded,32distinct selected examples/all20tasks,max teacher plus
  target16079tokens,zero training truncation. Prior CPU r1 failed with a mixed
  old teacher-context helper; its source/failure are preserved. r2 snapshots
  the complete current project-source set, not a mixed old runtime driver.
  Actor objective/scoring and runtime/capacity/parser functions match the GPU
  qualified source exactly:unchunked-control-source-compatibility.json.
- [ ] Repaired logical02restart is concrete but NOT submitted. Proposal:
  experiment02-repaired-proposal.json, same10steps/batch32/core recipe,
  feedback-only control,cacheoff,unchunked,MTP2,Triton,40slots. Fresh full native
  baseline required; previous02-r3 is ineligible for common-control reuse.
  Runtime repairs used the existing02window; deadline03:48:05UTC is retained.
  Less than24minutes remain versus~31minutes for prior updates1–10alone plus
  startup/collections/evals. Request explicit fresh time before exceeding it.
  No budget exception authorized yet; other requested arms remain available.

- [x] r9 real40episode greedy training check passed:195complete calls,
 84identical-input pairs,0different outputs. Stochastic40collection and full32K
  backward remain pending at03:17UTC. Exact owned r9 stays live; do not duplicate.
- [x] Optional MTP counter reader now has a real pinned-library CPU integration
  receipt:mtp-native-counter-cpu-integration.json. Instantiates native vLLM
  SpecDecodingProm in a private registry with injected counts and verifies
  snapshot deltas. GPU hidden for that auxiliary process, no model calls or
  running-source edits. This is collector wiring evidence, NOT measured
  rollout acceptance or speedup; do not quote injected75%as performance.

- [ ] Zero-update unchunked native/capacity r9 ACCEPTED under original02deadline:
  sdpo-qwen35-4b-guidance-five-unchunked-native-20261019-r9,
  Docker098b922e481c2ff2c8ede06be444f51ec665bd02589f56cdb899c623ebd79fde.
  Capsule sdpo-guidance-five-20261019-unchunked-native-r9,evidence/mtp; source
  f6d3fd01d16de48c1932d5300a1ea53344b106cd1f3243a558a9f2ac6c6ab465.
 40greedy+40stochastic frozen training episodes and new32Krollout/backward.
  Same pins,cacheoff,unchunked,MTP2,Triton,40slots.850s cap, no training updates.
  This complements r8's full8192mixed numeric witness; neither replaces the
  future full heldout collection. No fresh learning attempt accepted yet.

- [x] r8 UNCHUNKED numeric full-budget gate PASSED exit0. All16critical
  full8192reply comparisons are token/logprob exact against same-candidate
  serial references in heterogeneous40request forward/reverse orders. Native
  tool/result/final witness and short actor parity/batch gate also pass.
  Actual engine chunked-prefill=False was checked; cacheoff,MTP2,Triton remain.
  Evidence:native-unchunked-r8/receipt.json,native-mixed-qualification.json.
  This repairs the demonstrated witness, not proof of global determinism.
  Require actual greedy/stochastic training collection and32Kbackward next.

- [x]02-r3 FAILED before updates at03:02:22UTC,exit1,noOOM. It completed310
  heldout baseline episodes/all310scores/no native scoring errors, then the
  actual repeated-input audit rejected37different complete outputs among3500
  pairs (1967complete calls,393duplicate-input groups). All differed with
  zero cached tokens; cacheoff alone is insufficient. No learning gain or
  eligible shared control. Terminal state/logs and numeric-only failure audit
  preserved under experiment02-r3. No online admission/optimizer updates.
- [ ] Zero-update runtime repair r8 ACCEPTED,
  sdpo-qwen35-4b-guidance-five-native-unchunked-20261019-r8,
  Docker451a86c26b91b42646f5d2cf73c9511078ac5917bbdb9c93a9f956cf447294b7.
  Capsule sdpo-guidance-five-20261019-native-unchunked-r8,evidence/mtp.
  Disable only chunked prefill while
  retaining cacheoff,MTP2,Triton,40slots and model/dependency pins. Four failing
  numeric prefixes retain original8192reply cap;16varied background prefixes
  provide40request mixed prefill/decode pressure in forward/reverse orders.
  Heldout numeric data are runtime-only, never teacher/training inputs; no
  decoded task/answer semantics or success outcomes used for recipe selection.
 650s cap remains inside original logical02deadline03:48:05UTC. Source manifest
  0b02f91e8a9946f6a03d1e20d2257fc53ac36a7a00fb2e272701aa32cd526158.
  Fixture69ec910c202dfae976b69745b4d06e93defcb55106989d909b836ae50a327a3b.
  Eleven focused mixed/prefix/shared-control tests and Ruff/bash checks pass.
- [x] Shared-control optimization is CANDIDATE ONLY and cannot use02-r3.
  It checks source policy/collection semantics, weights, fixture, raw native
  files and recomputed native repeatability before reuse. Complete score
  coverage alone never qualifies it. Seven shared-control/native audit tests
  now preserve missing scores and separate own versus referenced episode IDs;
  rejected repeatability cannot export. No GPU collection is skipped yet.
  The earlier code-compatibility-v2 preflight predates the chunked-prefill
  repair and is historical evidence, not permission to reuse a different runtime.

- [x] Rechecked older SDPO training/development evidence with source hash and
  explicit limitations:historical-training-lessons.json. Slot5's apparent
  development success gain is parser-confounded; schema-bound slot6 starts at
 50%. Higher LR7 and cosine8 have partial-credit changes without sustained
  development success gains. Offline Luna treatment9 never started, so it is
  not evidence that Luna context failed. No new/old heldout failure trajectories
  used to choose later contexts.03still awaits actual02learning evidence.

- [x]02-r3 fresh TRAIN baseline reconciled40episodes/20tasks, all scored,
  zero errors:success52.5%,partial72.371031746%. Exactly matches separate r7
  cacheoff baseline. Across both executions390complete greedy calls form
 111duplicate-input groups and531unordered identical-input pairs; zero different
  complete outputs. Both own audits have84pairs/0differences. Same initial
  adapter hashes checked. Evidence:cache-off-cross-execution-repeatability.json,
  experiment02-r3/baseline-training-outcomes.json. This is repeatability evidence
  for a finite operating point, not learning gain or universal cache causality.
  Heldout baseline collection follows before any optimizer update.

- [x] Accepted02-r3 now passed its own full64K serving gate (not just r7's
  training-capacity gate):65536context,8192reply,57344token near-limit prefix,
  native tool/result/final continuation, exact40inflight batch witness.
  Receipt:experiment02-r3/full-evaluation-budget-qualification.json. Still zero
  optimizer updates at this stage; fresh train/heldout baselines follow.
- [x] Later-arm launch settings with risk/error selectors and optional MTP
  counters validated in four read-only configurations against exact frozen02
  input artifacts: seven settings, no GPU calls/source changes. Evidence
  candidate-launch-observation-preflight.json. Metrics missing-dependency path
  is recorded rather than failing training; three focused metrics tests pass.

- [x] Candidate later-arm native MTP measurement prepared outside accepted02:
  opt-in sdpo_mtp_metrics.py reads actual pinned vLLM Prometheus draft/drafted-
  token/accepted-token counters, recording per-population deltas and elapsed
  wall time. Missing counters remain unavailable, zero drafts have undefined
  acceptance, and counter resets are rejected as measurement evidence. Two
  focused tests pass; combined sampling/risk/metrics suite7tests and Ruff pass.
  Pinned source confirms the AsyncLLM manager installs PrometheusStatLogger and
  exact spec_decode counter names. No live counter values or speedup measured
  yet; no running source edited. Optional launcher flag is explicit in config.

- [x] Candidate next-arm launcher now takes the declared first-error-frontier
  or first-risk-frontier selector from configuration and rejects profile/config
  disagreement before submission. This changes no running capsule. Read-only
  exact launcher-config preflight on accepted02inputs proves all six settings
  and both selector values; five sampling/risk tests and bash syntax pass.
  Evidence:candidate-selector-launch-preflight.json.03remains unfrozen and
  unsubmitted pending actual02teacher/Adam/native student results.

- [x] r7 completed exit0:40greedy and40stochastic native episodes, native
  call/result/final witness, exact88prefix probes, full32768token actor backward
  with finite nonzero student gradients and unchanged weights. Evidence
  native-cache-off-r7/receipt.json,capacity-qualification.json. Cacheoff actual
  engine configuration verified. No optimizer updates in qualification.
- [x] Fresh logical02repair r3 ACCEPTED02:29:24.935UTC,
  sdpo-qwen35-4b-guidance-five-exp02-20261019-r3,
  Docker405f906f829f00cd58cc31f259d7b7ae821f92e5dd9ed0a4dec8af907dc08848.
  Source/evidence capsule sdpo-guidance-five-20261019-experiment02-r3,
  evidence/run. Model loading confirmed; no new optimizer update yet. Outer
  timeout4690seconds enforces original logical02deadline with shutdown margin.
  Same feedback-only control,
  base/model/data/loss/token budgets/MTP2/Triton/40slots. New cacheoff runtime and
  explicit initial/changed-adapter/native greedy guards; old01cacheon outcome
  comparison is exploratory. Deadline remains03:48:05UTC. Source manifest
  c3aaac47077bc31788ebc21a95bd020a915e2d4e74efb94393a1c29f6c11d834.
  r2 was NEVER GPU submitted: literal newline serialization was caught before
  launch and its staging retained. r3 JSON/hash manifest/bash preflights pass.
  Main step1 native witness now explicitly greedy and restores online temperature;
  no stochastic training sampling change. All running/old sources untouched.

- [x]02:21UTC r7 cache-OFF actual40episode TRAIN baseline passed native
  repeatability:195complete greedy calls,84duplicate-input pairs,0different
  complete token outputs. Earlier within-run controls differed8/74and6/71.
  This supports the proposed cache-off operating point, not universal
  determinism or a clean one-setting causal experiment (new execution/history).
  At this intermediate observation, stochastic40episode collection and full32K
  rollout/backward were pending; r7 subsequently passed as recorded above.
  Retained native-cache-off-r7/
  native-greedy-repeatability.json and baseline-training-outcomes.json.

- [x]02:08:41UTC native-prefix cache-ON r6 exited0. All88exact-token/logprob
  checks passed at40inflight on four retained native prefixes. This simplified
  static gate did NOT reproduce full native-history variability; cache causality
  remains unproven. Receipt retained in native-prefix-cache-on-r6/receipt.json.
- [x] Historical r7 submission receipt (now completed exit0): ACCEPTED02:16:09UTC,
  Docker1260dba15d611b7bf1ace95e10648d055767505b223731ae908ed836b092489f,
  retaining MTP2/Triton,
 40slots/model/pins/budgets. Test actual40greedy native episodes for duplicate-input
  token repeatability, then40stochastic episodes and new full32K rollout/backward.
 850second cap remains inside logical02deadline03:48:05UTC; no budget reset.
  Source manifest02f8b5e6f4fd9121c780821cdefe8e104834cc4a0fedfaf5ab434fcf417793a9.
  Five focused repeatability/prefix tests pass; Ruff/bash checks pass. Exact
  logprob checks on static probes remain unchanged; full native auditor reports
  probability variation separately from token disagreement. No learning claim.
  Candidate restart plumbing (outside running r7) now forces static prefix
  probes greedy and restores online temperature even on failure; initial and
  changed-adapter checks plus actual greedy-population checks are available.
 32focused tests/Ruff/bash validation pass. This receipt qualified the later
  accepted02-r3; the earlier staging status is superseded above.
- [x] New complete-native-call auditor replayed retained baseline populations:
 01has8different complete output pairs among74within-run same-input pairs;
 02has6among71. Combined all unordered pairs has41differences among457pairs,
 108duplicate-input groups. Earlier23 used a narrower pair selection. Evidence
  native-repeatability-retained-replay.json; this is not cache causal proof.
  At02:20UTC r7 passed88static native-prefix checks and the native tool witness;
  real40episode greedy baseline was collecting. No new optimizer update.

- [x]02-r1 STOPPED before any optimizer update at02:00:59UTC,exit143, after
  stronger native invariance evidence. Guard verified no updates.json or online
  training-admission files.20first-turn prefixes/outputs/logprobs match01exactly,
  but108identical-input control groups include23different COMPLETE greedy output
  pairs at later native prefixes with the same actual adapter and token budget.
  Differing cached-prefix lengths correlate; causality is NOT established.
  Source/stop evidence retained under experiment02/baseline-identical-input-
  variability.json,baseline-first-prefix-comparison.json,pre-update-stop-receipt.json,
  stopped-state.json. Earlier short exact40gate did not cover this operating
  point. Do not treat01learning deltas as clean causal effects or resume02r1.
  Qualify real native prefixes/cache states before further fitting; MTP remains
  required. Logical02deadline03:48:05UTC (first start plus7140s) is retained,
  including qualification/repair work; zero updates spent by02,10/50total.

- [x] Candidate first-risk-frontier is now an explicit opt-in selector outside
  sealed01/02. It prioritizes factual undeclared-argument/error-object cues in
  failed episodes, keeping source actions, teacher/student tokens, loss and
  native execution unchanged.22focused tests/Ruff pass. Exact pinned CPU replay
  of01online steps3and10 (40episodes/20tasks each) verifies admission and all
  teacher/student tokens unchanged;32unique actions/all20tasks retained.
  It selects Asana creation24instead of later assignment72 atstep3, and
  creation38instead of assignment71 atstep10. Both retain terminal selections;
  observed_tool_error staysfalse, so no fabricated API error/reward.
  Evidence:native-risk-step3.json,native-risk-step10.json. Source
  f2e96c3778bb56ff27688654495a03606beb07f820f99c26dd07e6b71633dbab,
  capsule sdpo-guidance-five-20261019-risk-sampling-cpu-r1. Zero model calls or
  optimizer updates. This qualifies selection mechanics, NOT gradients/rewards;
 03is not frozen/submitted, pending02training evidence. Reserve55CPU seconds
  against any future learning arm that uses this candidate.

- [x] Historical02-r1 acceptance (subsequently stopped at zero updates):
  sdpo-qwen35-4b-guidance-five-exp02-20261019-r1,
  container2712c945f2437876450e5e921523348c85ee60355d9df6e8fd9af0349431f7ed,
  started01:49:05UTC. Source/config remain frozen
  d10ac2b743abcf4a3fd96762e0f7dc24bb896ed1f751b8e8457ae4cf18d58e1c.
  Evidence:experiment02/launch-receipt.json. Exact02container/execution absence
  and realGPUcompute-free state verified before the one accepted submission.
  Fresh base,10updates,batch32,MTP2/Triton,40slots,matched settings,guide payload
  null. Startup only; no actual02updates confirmed. NEVER resubmit or mutate it.
- [x]01-r3 exited124 at01:47:11UTC under original time cap, no OOM. Preserved
  logs/state, ten-update checkpoint and EMA proofs. Complete790episode closed
  populations plus243post-heldout episodes=1033unique retained episodes;
  67heldout episodes missing, no partial-file fragment or scoring errors in the
  retained243. Full post-heldout mean unavailable. Success identification bounds
  41.94–63.55%,partial62.08–83.70%; these are NOT confidence intervals or an
  observed-only headline. Do not zero-fill or pool attempts. Pending user
  evaluation-only extension question remains unanswered; no extra01eval launched.
  Evidence:experiment01-r3/terminal-native-audit.json,
  partial-heldout-terminal-audit.json,terminal-state.json,terminal-run-timestamped.log.
- [x] Actual adapter lineage verified for all4382generation calls in completed
  native populations against11distinct retained student exports,versions0–10.
  Unfinished heldout population explicitly excluded from this closed-population
  proof. Evidence:experiment01-r3/checkpoint10-native-policy-lineage.json.

- [x]01checkpoint10 matched greedy TRAIN readout reconciled40/40scores:
  success55%to50%,partial72.0468%to69.0538%. Paired20task success delta-5points
  CI[-15,0];partial delta-2.993points CI[-16.396,+8.885]. Asana Sprint task
  regresses from2/2to0/2; no task gains binary success. Some partial goals improve
  (campaignROI/content performance), others regress. Legacy tool_mistakes.05to0
  does not establish task correctness; actual observed tool-error responses26to24,
  repeated exact calls7to2, empty retrievals32to39. Heldout later timed out;
  its terminal missingness is recorded above.
- [x] Actual regression trace:baseline uses declared Asana dueDate and returned
  task contains it. Checkpoint10 sends undeclared due_date; API returns success
  but task has no due date. Training exposure audit finds9such creation calls
  across20generated creations;NONE selected. Only2correct creation calls were
  selected; sampler generally selected later section assignments. Explicit-error
  frontier therefore still misses silent semantic failures. Evidence:
  experiment01-r3/asana-training-action-exposure.json and
  checkpoint10-training-case-review.json. No heldout failure trajectory inspected.
- [x] Candidate factual risk cue added OUTSIDE01/02:undeclared argument fields
  and error-only response objects. An undeclared field with additionalProperties
  allowed is a sampling cue, not an invented API error or per-action reward;
  successful empty searches remain neutral.4tests/Ruff and actual20creation-call
  CPU audit detect all9missed risky calls. No integration/new training yet.
  Candidate evidence:candidate-asana-risk-cue-audit.json. Keep02matched control
  unchanged; choose any later sampling treatment from its training evidence.

- [x]01-r3 completes all10actual optimizer/EMA updates and saves checkpoint10.
  Final loss0.2648843,gradnorm0.511892; fresh batches differ so this is not a
  controlled loss trend. Saved optimizer moments10, student/teacher finite and
  nonzero; all7checkpoint metadata file hashes reconcile. Independent CPU EMA
  recurrence across all10student exports matches the final teacher with zero
  squared error. The teacher update itself is working; this does not prove its
  contextual supervision is semantically correct or that rewards improved.
  Evidence:experiment01-r3/checkpoint10-files-audit.json,
  checkpoint10-hash-audit.json,checkpoint10-independent-ema.json.
  Final greedy training40/heldout310 collection is running, not complete yet.

- [x] Independent EMA recurrence auditor added outside running capsules.
  It reads the initial student, all ten saved student policy exports and final
  teacher; compares recomputed EMA under dtype/depth precision bounds, including
  tiny LoRA-B updates that a fixed absolute tolerance could hide. Frozen-teacher,
  student-only-teacher and nonfinite controls pass3tests/Ruff. CPU source staged
  as sdpo-guidance-five-20261019-ema-audit-r1,
  SHA4284b459012fd7e0b166ee9f561495f42de8340971b25c6b27c3dc6e91dbee32.
  Actual final recurrence is NOT yet audited; require10closedupdates/parityv10.

- [x]01-r3 through update7 independently reconciled:630unique native episodes,
  nine completed populations, zero native scoring errors,224distinct selected
  transitions with all20tasks per update and correct recorded policy versions.
  Evidence:experiment01-r3/mid-training-audit-step7.json. Fresh stochastic
  training outcomes are not matched greedy reward gains; final readout pending.

- [x]02control source/config frozen from actual01training decision evidence
  before any01post-heldout outcomes. ONLY intended treatment:null prepared-guide
  payload, with identical hindsight/prompt shell/objective/model/data/budgets.
  This is not paper-prompt reproduction or a no-context teacher. Capsule
  sdpo-guidance-five-20261019-experiment02-r1, source
  d10ac2b743abcf4a3fd96762e0f7dc24bb896ed1f751b8e8457ae4cf18d58e1c.
  Byte-identical CPU-qualified control implementation and guide reuse40episode/
  20task preflight; input/source/config and shell checks pass.55CPU seconds
  reserved plus7100GPU seconds/30kill grace remains below7200total. NO GPU
  submission yet; wait for01terminal/audit and verify no retained02run/GPU use.

- [x]01-r3 actual update5 confirmed,loss0.2848263,gradnorm0.550142,moment/EMA5.
  Four warm intervals200.23/204.18/221.12/230.77seconds show variable fresh
  collection/fit cost. Final full evaluation may not fit before the original
  logical01deadline01:47:40UTC, which includes zero-update failed setup attempts.
  Asked user asynchronously for at most30extra evaluation-only minutes if
  needed, keeping10updates. NO extension is authorized until answered; active
  timeout/source remain unchanged. Continue permitted work meanwhile. No
  automatic crash resume, episode pooling or shortened readout is permitted.

- [x] New private-context CPU-r2 first-stage preflight passes all20tasks with
  the intact-contract candidate. Maximum prefix22229tokens plus4096reserved
  output fits32768 without clipping. Sourcee320f93daa30c90bfd345b822a599ce84965d66f87e791d577ca959dbed0965e.
  Evidence:private-preparation-input-preflight-r2.json. Zero model calls,
  optimizer updates or GPU allocation; full generated-bundle context size and
  semantic quality are still unqualified. No second learning arm submitted.

- [x] Deterministic preparation-input audit found typed expression metadata
  corruption in the historical observation-bound source representation: nested
  literal.kind can become a retrieval-description object. This is an input
  semantics defect, not proof of its causal share in model failures. Candidate
  private-checkwise now reads a deepcopy of the intact source contract before
  interpreting checks, then validates every compiled instruction against the
  original private-literal/metadata output boundary. Default/01-r3 remain
  unchanged. Typed-predicate preservation plus actual schema dispatch and
  rejection tests pass17checks/Ruff. Evidence:candidate-contract-input-audit.json.
  Previous private first-prefix preflight is stale for this new input version;
  rerun it before any model generation. No new learning arm is frozen/launched.

- [x]01-r3 reaches actual update2,loss0.2466656,moment/EMA2,gradnorm0.477226.
  First observed warm update interval200.23seconds is not a stable runtime
  estimate. Original logical01deadline remains01:47:40UTC, including preserved
  zero-update repairs. Do not reset the deadline or shorten final populations
  to manufacture completion; preserve any terminal partial evidence honestly.

- [x] Candidate private-checkwise preparation now ends with a whole-guide
  source-contract review, including cross-check target/decoy polarity and
  contradictory requirements. A rejection stops admission instead of silently
  dropping a check or repairing only the procedure. Default preparation and
  sealed01-r3 are unchanged. Actual generation schema dispatch and rejection
  paths pass16focused tests/Ruff. No model generation or semantic qualification
  of this new final review has occurred; earlier first-prefix CPU evidence
  does not validate its larger assembled-review context.

- [x]01-r3 actual optimizer update1 confirmed:batch32,loss0.2504119,
  gradnorm0.432321,LR1e-6,moment/EMA1. Full32768token/4096target actor
  backward passed before fitting. All20tasks represented by32distinct complete
  actions;185admitted,3whole reply-limited generations excluded,no truncation.
  Changed-adapter parityv1 max0.056344nats, exact40batch gate and native
  tool-result/final-content continuation pass; step2 collection is now active.
  Saved step1-runtime-receipts.json and training-admission-step1.json.
- [x] Audited all32actual selected-prefix/teacher/Adam records and retained four
  training decision reviews in experiment01-r3/step1-four-decision-review.json.
  Email analytics batches an unobserved worksheet lookup with a wrong parameter;
  teacher discourages some wrong tokens, but not every actual Adam token change
  follows that direction. News digest malformed quoted Gmail query receives a
  negative mean sampled-action advantage and negative mean sampled Adam delta.
  Competitor message API calls succeed but terminal content checks fail;
  prepared guide also invents incompatible status/date filtering. Sales phone
  update PASSES; missing audit note causes the episode failure. Its prepared
  guide both requires a note on the updated contact and forbids that same
  relationship. Never label the successful phone call a wrong action merely
  because the episode failed. This provides actual context/sampling evidence,
  not post-update native improvement. No heldout failure trajectories inspected.
  Preferred teacher tokens outside studenttop20 are0% in the selected email,
  digest and Sales tool-action regions and0.72% for competitor; these cases
  do not establish missing top20 support as the dominant cause.

- [x]01-r3 complete native baseline reconciled independently:350unique scored
  episodes,40train plus310heldout, zero native scoring errors. Train success55%,
  partial72.0468%; heldout success57.7419%,partial83.3582%. Separate absolute
  task-bootstrap success intervals:train35–75%,heldout45.48–69.68%. These are
  fresh pre-update controls only; no gain or learning claim. Evidence:
  experiment01-r3/baseline-native-audit.json. Previous failed-attempt controls
  remain separate and are not substituted or pooled.

- [x]01-r3 live native audit at00:53UTC reconciles its own complete40episode,
  20task training baseline:55%success,72.0468%partial credit, no scoring errors.
  These are pre-update controls, not improvement. Baseline heldout collection
  remains active; no updates yet. Saved experiment01-r3/live-native-audit.json
  covers completed populations only, not the partially written heldout cohort.
  Matched context/absolute uncertainty/actual-direction unit checks pass14tests.

- [x] Candidate matched no-prepared-task-guide control added outside active
  capsule. It sets guide payload to null and preserves identical prompt shell,
  retrospective findings and final directives; it is not a paper-prompt
  reproduction or a no-context teacher. Native pinned CPU replay verifies all
  20tasks/40episodes and unchanged original student tokens,0model calls/updates.
  It is a possible next arm, not a frozen/submitted learning experiment.
- [x] Verified pinned SDPO loss SHA530b571c0001bbf85bb0c78725d32645a4d7155660e22e41580fab5c9f7041e4:
  top20 support is selected by the student; remaining teacher token mass shares
  one tail bucket. Added region-specific teacher-top1 outside-top20/top100
  fractions to the read-only direction audit. This is a mechanism to inspect,
  not a proven explanation for native performance or a changed running loss.

- [x]01-r3 full ten-update attempt HAS BEEN ACCEPTED: Docker
  f2ace6c0657d292981671837af7bca9fceaeae27db8408378a56d2c32e9ba11e,
  run sdpo-qwen35-4b-guidance-five-exp01-20261019-r3,
  capsule sdpo-guidance-five-20261019-experiment01-r3. Native input CPU replay
  again passes40episodes/20tasks/199complete actions;2whole incomplete outputs
  excluded. Worker trace0 is explicitly enforced and qualification receipts
  require native stochastic40 plus actual32768token backward. No updates yet
  observed at acceptance. Original logical01deadline and overall50update cap
  remain unchanged; failed attempts are separate, never resumed or pooled.

- [x] Stochastic gate-r5 SUCCEEDED exit0:40native temperature1 episodes,
  engine alive; full32768token/4096target selected-loss backward finite/nonzero,
  weights unchanged,0optimizer updates. Actor peak27289522176bytes,
  actual backward13.4855seconds. Worker confirms periodic0/fatal tracing enabled.
  This qualifies the operational configuration, not a causal timer diagnosis,
  actual changed-adapter readiness or reward improvement.
- [x] Experiment01-r3 source sealed after the successful stochastic/backward
  gate. Same guide/algorithm/data/budgets/numerical checks, original logical01
  deadline retained. Sourcea731867367939a12a5c4da68c35136456245a0ba0da814f1d33796dd07581c57.
  Rechecking all native fixture bindings before launch. Neither failed attempt
  is resumed or pooled; new matched baseline will be collected in r3.

- [x]01-r2 exited1 before any optimizer update: EngineCore died during first
  temperature1 online collection.40episodes retained,5ProviderErrors; no
  training admission or actor-capacity receipt. Baseline350episodes remain
  diagnostic/control evidence for this failed attempt, never pooled as new
  run results. Full timestamped logs and terminal Docker state retained.
- [x] Root cause unproven. No kernel OOM/Xid evidence found; a worker's45second
  periodic faulthandler dump ends in Triton driver launch immediately before
  engine loss. Correlation is not causal proof. Initial gates had been greedy,
  so actual stochastic/native readiness must be tested before another costly
  full benchmark. No dependency/model/budget/numerical threshold changed.
- [x] Targeted zero-update stochastic gate-r5 ACCEPTED, Docker
  39989bf8be72d9b9a7ebb7e8cc6e6c3570f14d15a24516928ab5d03efb53587b,
  capsule sdpo-guidance-five-20261019-stochastic-gate-r5,
  source61bdacc4e6b61a5f290d9a97a6b8151393e52d85bdfb31a55738b60708dda0fd.
  600second cap and original logical01deadline retained. Same MTP2/triton40slots;
  periodic dumps disabled, fatal traces and child exit-code capture enabled.
  It runs40native stochastic episodes and actual32K zero-update backward using
  the byte-matched previously measured rollout-capacity reference. Monitor it,
  never duplicate or restart full training before its receipts qualify.

- [x]01-r2 baseline350native episodes reconciled:40train and310heldout62,
  exact62ordered task keys/five attempts/40concurrency and frozen budgets verified
  from persisted resolved eval config. Benchmark baseline success57.0968%,
  partial82.1763%,zero native/scoring errors. No optimizer update at this point.
- [x]32K rollout capacity passed4096generated tokens at full32768total context;
  first online training population collecting. Actor backward gate deferred
  until the complete batch is admitted, before optimizer step1.
- [x] Added read-only actual Adam audit separating original reasoning/tool/final
  token regions and verifying serialized teacher context. It never concatenates
  pointwise top1 tokens into a purported teacher trajectory. Separate absolute
  uncertainty is now reported per population, with incomplete-task intervals
  explicitly conditional and full-population intervals absent when missing.
  Analysis capsule sdpo-guidance-five-20261019-audit-r2 is CPU-only, separate
  from immutable running source. These new audits still require applied data.

- [x] Repaired01-r2 fresh native training baseline complete40/40episodes:
  success55%,partial71.4218%,no scoring errors. Actual r2 teacher task bindings
  pass40/40 against compiled native state representation. Its own baseline,
  not preserved r1's69.7551%partial, controls its checkpoint comparison. Eval
  baseline310episodes is in progress; no optimizer update observed yet.

- [x] Pre-update semantic spot audit identifies unsupported status/date filtering
  in competitor guidance and unjustified latest-email selection in renewal
  guidance. Recorded before any learning result. Binding/self-review approvals
  are not semantic certification. Running01 remains the model-approved noisy
  taskwide-guide treatment; do not relabel its results as clean-context evidence.
- [x] Added opt-in private-checkwise preparation candidate: same4B preparation
  sees retained raw offline references and source contract, while output still
  passes no-private-literal/metadata and retrieval-only checks. Default/running
  sources unchanged. Fourteen focused preparation tests pass. Pure CPU pinned
  tokenizer/schema preflight covers all20 first-stage prompts without clipping;
  no model calls/updates. Next learning treatment is not yet frozen or submitted.

- [x] Experiment01-r1 baseline40episodes scores55%success/69.7551%partial;
  no scoring errors. No optimizer updates. Runtime audit reproduces16teacher
  binding failures across8tasks because the pinned WorldState constructor
  flattens Sheets nested records in-place. r1 stopped/preserved exit143 before
  fitting, not a learning failure or a completed experiment.
- [x] Pinned constructor on deep copies exactly matches all40native initial
  states. Receipt normalizer SHA3536a92a18db7a8275f2750591b97c01412a31e7eb4583e16c1e80f0f4565851.
  Recompiled guidance stores both original and native representation hashes;
  no task values, instructions, checks, pins or student prompts changed.
- [x] Expanded pinned native CPU preflight covers40episodes/20tasks/201calls,
  admits199complete actions and excludes2whole reply-limit outputs. Maximum
  teacher context+target23534tokens. All student prefixes/action IDs unchanged.
- [x] Repaired experiment01-r2 ACCEPTED: Docker
  1651423d49835abc126cf7f5e3d24bb6f94de4c6929283c172c97c8b60d31cfb,
  capsule sdpo-guidance-five-20261019-experiment01-r2,
  source21bbf6df177ca9dded40a7b2cbf686577a2cad415a6b6fb14eca940d9342c531.
  Same logical experiment01 deadline retained from r1; no extra50update or
  two-hour budget reset. No training updates yet observed. Never duplicate.
- [x] Native declared-signal audit finds missing goal/harm decisions in all40
  baseline episodes. Legacy success/partial and final factual assertions are
  available; absence of native guard findings is not harm-free evidence.
  Jira SALES resource rejection is separately under task-feasibility review.

- [x] Experiment01 long serving receipt qualified:57344prompt tokens plus8192
  generated tokens reach65536total; exact long-prefix batch invariance through40
  and native tool continuation pass. Synthetic generation hit its deliberate
  reply cap, remains capacity-only evidence, and is never a training example.
  Native baseline collection follows; full32K actor backward still pending.

- [x] Experiment01 actual initial short-probe MTP batch invariance passes the
  latest reordered4-request witness through40inflight. Native tool/result/final
  witnesses pass at32768/4096 and65536/8192 with separate reasoning. Long-context
  near-limit serving and actor-backward remain pending; no optimizer update yet.

- [x] Preparation r8 exited0; all20contexts generated/accepted. Spot review now
  confirms complaint read-before-reply and original-message IDs, not sent-reply
  ID equality. All20bundles pass exact current state/user/requirement bindings.
- [x] Latest native CPU fixture replays2episodes/6complete actions with unchanged
  student prefixes/actions, teacher-only context and no exclusions/truncation.
- [x] Experiment01 HAS BEEN ACCEPTED: run
  sdpo-qwen35-4b-guidance-five-exp01-20261019-r1, Docker
  3ae28482e1ac17c4e3dc89dc5d7cbe6838d7d714ecc87ecdd229ec5da78e5c56,
  capsule sdpo-guidance-five-20261019-experiment01-r1,
  source14b8f9b08b7a9a9f8c67c92f4a437b17e0ea98cb8bb667b303c6f6292e78db27.
  Ten updates authorized; none yet observed at acceptance. Monitor owned run,
  never duplicate or treat its GPU occupancy as competition. Initial65K,
  full32K backward and real updated-adapter gates still execute inside this run.

- [x] Preparation r7 generated20/20 approvals, but source audit rejected the
  complaint context: original-message read event was described as sent-reply ID
  equality, and a prompt example induced unrelated batch-summary language.
  Preserve its source/outputs; approval alone is insufficient.
- [x] Preparation r8 accepted, Docker
  2605e2e366bf3151c0f68c3ce1ca92954d4cce9b0d917fb606cd7c29e2d1b1b9,
  source159304372812f93e6e09e9851fc3818d6748ceed415c78b845803c72483d4cf4.
  Retains19contexts and regenerates only the complaint context with explicit
  native event-kind scoping and no task-specific prompt exemplar. Zero updates.
- [x] Added source-bound native training finding auditor; missing/invalid signals
  remain unknown, and absent guard findings never become a harm-free claim.
  Latest focused suite62tests passes, Ruff and diff check pass.

- [x] Preparation r6 exited1 with19 model-approved contexts and1 rejection;
  manual/source validation found spaced `request key` selectors in six contexts.
  They are invalidated, not admitted as runtime requirements. No updates occurred.
- [x] Added separator-aware bookkeeping rejection, explicit selector-omission
  prompts and readable membership predicates. Twenty-two focused CPU tests pass.
- [x] Preparation r7 accepted as Docker
  4c775a343fab8d98e308a41cdce708dd78e740b370ef8b2f736248aa513c1b31,
  source542b9dddf0b3501f6032b8922efcc0c8197e1b707f01725880167acab1e8162c.
  Retains13 bound contexts and regenerates7, all20 tasks retained. Monitor exact
  capsule sdpo-guidance-five-20261019-preparation-r7; no learning submission yet.

- [x] Live SSH confirmed free GPU/no running Docker containers.
- [x] Read prior SFT generation and SDPO code/results. Historical task audit:
  89/176 candidates pass current authority/tool/request checks, four categories.
- [x] Added explicit speculative config support and a true40-request invariance
  probe; three focused CPU tests pass. Old capsules remain unchanged.
- [x] Froze cohort-v1:20tasks,40online trajectories/update,11Simple/6Marketing/
  2Sales/1Finance; exact protected-set overlap empty. Twelve historical initial
  states match current states; eight are procedural variants only.
- [x] MTP gate r1 reached40inflight with identical generated tokens, but failed
  exact target-logprob invariance: maximum discrepancy0.049195528nats. Zero
  optimizer updates. Retained Docker d96ad80187e4133bf3ea0cecb0a49e18ebb1edeca581613a1f4b8dfe77e8d446.
- [x] Accepted one-token MTP gate r2, Docker
  8d23160291c2ce391e831d98745532830909e177eb6434a0d9368b6ba9e1db73,
  source77f6ecd5ddb0de2c324be69cadd89d7943e5132bb1d57bc24d9e54eb4e5a1806.
  Terminal exit1: identical tokens through40 but target logprob drift up to
  0.048253298nats. Zero updates; retain mtp-gate-r2 local evidence.
- [x] Accepted matched plain-decoding40-slot control r3, Docker
  fe7b01a461d3acc9efb6c63650b4d0bab962f63971880c67e7eaf8685fa12300,
  source6419c119e2900e3aadc1675b6c000c29d58330c8608808d9aaa9c40cac4049df.
  Capsule sdpo-guidance-five-20261019-plain-gate-r3. Diagnostic only; never
  silently substitute plain decoding into the requested MTP learning campaign.
- [x] Plain r3 exited0: exact40-slot invariance, native call/result/final passed,
  actor/rollout short-probe maxdelta0.137607574nats within unchanged0.15 gate.
- [x] MTP2 with existing VLLM_GDN_DECODE_KERNEL=triton r4 exited0: same unchanged
  parity/invariance/native gates passed. Docker
  ea247ba11025a44c2f02540f512931307cee4ae7e524f4c6f26e99dd68653bc3,
  capsule sdpo-guidance-five-20261019-mtp-triton-gate-r4,
  source cf451d8b390427d35e2737fe1720e9a50651e1ea94c6e4cedbfa4c7300cf5258.
- [x] Preparation r1 failed before generating any examples: Transformers5
  apply_chat_template returned a structured encoding, not a list of integer
  IDs. Preserved container01b2bfe2869de9bfdb69a7cbef983423d8b008d5ab0a977bd42d622aed87175f.
- [x] Fixed preparation with the existing pinned explicit token helper. New r2
  CPU preflight passed all20real task prefixes/schemas, maximum8085prefix tokens.
  Preparation r2 accepted: Docker
  de71d9b23e5fcb4f7eb866a5babfdb373b7b9887a501a408b4f93d2fdd512bf0,
  source b0de4ea7209a0b6003067b53bd5d41fce83120242ea53a58513d4ea8f9228f81.
  Capsule sdpo-guidance-five-20261019-preparation-r2. Monitor it; never duplicate.
- [x] Added opt-in ten-step profile, teacher-only binding/rendering,4B preparation
  driver and CPU tests. Native evidence/obligation audit and launch integration
  remain to finish before a learning run; passing CPU tests is not qualification.
- [x] Forty focused CPU tests pass, including native-assessment dedup/missingness,
  legacy-pass/native-goal-failure frontier selection, explicit new-profile caps,
  protected-set selection and teacher/student separation. Ruff/diff checks pass.
- [x] Expanded suite65focused tests passes. Added opt-in schema-first API argument
  roles and value-free query structures based on identified SFT defects; old
  preparation variants retain behavior. Native signal IDs now come from the
  manifest, not guessed task/check names. Retained Sales failure replay detects
  valid0 audit_note_on_target despite legacy success1; missing signals remain
  explicit. Compact teacher findings avoid repeating hundreds of candidate IDs.
- [x] Preparation r2 generated82stages:8approved/12rejected. Preparation r3
  generated110stages:9approved/11rejected. Both preserved, neither admitted as
  a complete20task training-context population. No student updates occurred.
- [x] Real retained native replay passed:2Trello episodes,6complete original
  actions, zero exclusions/truncation, original student prefix/action IDs and
  teacher-only context. Receipt native-cpu-r1.json. This is protocol evidence,
  not20task rollout/learning qualification.
- [x] Checkwise preparation r4 accepted: Docker
  7ed18b23f446ec7137eadf5ff8cb0c7af9db0794473313456d3308ea8851514a,
  source9d58b9867fa229eb8d56a86f1dd7f48e4540cc8d68f0cc7f76904c4dfb064566.
  Capsule sdpo-guidance-five-20261019-preparation-r4. Latest verified running
  after graph capture; no completed guidance at that observation. Monitor this
  exact owned run before any new GPU launch.
- [x] Checkwise r4 terminated after154generated stages:16approved contexts,
  4rejected tasks. Rejections were metadata prose and an exact-string false
  positive on JSON composed entirely of public Airtable fields/values. All
  rejected outputs/stages remain in preparation-r4 local evidence.
- [x] Preparation r5 accepted: Docker
  cddac8c2dd4a5ae53448b1421736e905fc3c87e15d4436ab642fbf2a25ef2ac9,
  source3ff4e23f4ae7414b56000603f3f84e32b6a9a87cef738c5bc352f96724c8ff15.
  Capsule sdpo-guidance-five-20261019-preparation-r5. Carries16individually
  approved, unchanged same4B/checkwise contexts after exact model/manifest/
  signal/state checks and regenerates only4rejected tasks. Monitor this owned
  run first. All20tasks must still be present before any training launch.
- [x] Current focused suite72tests passes; Ruff and diff checks pass.
- [x] Run zero-update native MTP initial-LoRA/parity/40-inflight/tool gate.
  MTP Triton r4 and native stochastic/backward r5 receipts retained above.
- [x] Freeze protected-set-disjoint training/evaluation identities and current
  manifests. Same4B-only guidance generated with retained stages/rejections.
  Twenty tasks cover four domains, not all seven; admitted guidance is known
  semantically noisy and is the treatment under investigation, not certified truth.
- [ ] Qualify full-context backward, changed-adapter refresh, MTP comparative
  throughput and native episode admission. No silent non-MTP fallback.
- [x] Experiment01: ten updates, compact source-separated task-wide guidance.
  Final native evaluation and full experiment reconciliation still pending.
- [ ] Experiment02: ten updates; matched feedback-only control is frozen from
  actual01training evidence. Cache-off repair r3 is accepted and running; inspect
  latest Progress and its exact owned container before any further action.
- [ ] Experiment03: ten updates; adapt from earlier native evidence. The initial
  retrieval-dependency-focused proposal is a candidate, not an accepted recipe.
- [ ] Experiment04: ten updates; adapt from earlier native evidence. Source-
  grounded recovery context remains a candidate until relevant evidence exists.
- [ ] Experiment05: ten updates; best supported combination plus retention,
  selected without using heldout failure trajectories or outcome-based tuning.
- [ ] Reconcile all five native outcome/coverage/gradient curves and report.

## Surprises & Discoveries

- Explicit API-error prioritization is insufficient when tools silently accept
  undeclared arguments. In01, nine Asana due_date creations were never selected
  for fitting, while later successful assignment calls were. A schema risk cue
  is not proof of a failed effect: additionalProperties may allow extra fields,
  and declared aliases may still differ under frozen legacy scoring. Preserve
  this distinction instead of inventing a per-action penalty or changing tools.
- Actual update1 Sales trajectory selected a correctly completed phone update
  from an episode whose audit note was missing. Whole-episode failure therefore
  cannot serve as a negative label for the selected tool call. The task-wide
  guide simultaneously requires and prohibits a note on that updated contact;
  individual same4B check/workflow approvals did not establish mutual consistency.
  Candidate whole-guide review now rejects contradictions before admission,
  but real semantic effectiveness remains to be tested on generated contexts.
- Native baseline source audit confirms Jira SALES cannot resolve: isolated
  pinned tool replay on initial world has0projects and returns
  jira_project_not_found. Available task tools cannot create a project. Preserve
  this feasibility limitation separately from model syntax/content mistakes.
- Preserved r1 baseline tool audit v2 counts36observed error responses:
  14success=false,10additional error-only JSON and12text exceptions. The active
  sampler already detects text exceptions but misses those10SOQL JSON errors.
  Running sources remain unchanged. This is a candidate later sampling
  treatment, not evidence of improvement or permission to relabel empty reads.
- Four source-backed baseline failure reviews retained in
  experiment01/baseline-four-failure-review.json: checklist search drift,
  unsupported SOQL/recovery, unavailable Jira project, and successful API calls
  with incorrect competitor-summary content. Actual teacher/Adam evidence must
  be joined from new online episodes after fitting; baseline hints are not it.

SFT successes sometimes copied hidden policy or invented note-title constraints.
Reviewer approval and legacy task completion did not guarantee correct behavior.
Previous SDPO sampling could select an already-correct early search instead of
the later failed decision. A task-level failed goal is not per-action credit.
The existing batch probe never sent40 requests even when asked for40; fixed
by constructing a population containing the requested concurrency explicitly.
The89 eligible Luna candidates contain finance1,marketing6,sales2,simple80;
do not claim seven-domain balance. Historical state variants are procedural
references, not current-state solutions or verified current gold calls.

## Decision Log

- Latest HUMAN AUTHORIZATION supersedes the unlaunched04guide/05adaptive design:
  use a better balanced SAME20-task mix for04feedback-only and05compiled teacher
  guidance,10steps each. Fresh base/EMA each; loss, LR, sampler, batches, budgets,
  thinking, concurrency40, frozen heldout62 and source pins matched. No MTP.
  Names/source-quality/exclusion proofs are prespecified in
  balanced-context-pair-protocol.json; final training freeze requires native replay.
  Candidate domains: Finance3,HR3,Marketing3,Operations3,Sales3,Support1,Simple4.
  This covers all7categories but is not equal-quota balance: only one Support
  candidate clears current source/historical-tool availability. It has0.9091Luna
  partial/0success and2known failed assertions (trend classification and unnecessary
  email); retain these as preparation counterexamples, never gold. Nineteen other
  references are historically perfect on scored assertions; excluded historical
  assertions remain explicit, not full goal/harm proof. Do not select using new
  student or heldout outcomes. Replacements may only address pretraining native
  incompatibility using documented eligible same-domain backups; after04starts
  preserve exact same final20for05.
  Prepare/review common20guides once INSIDE04's own2h window, freeze before04
  updates;04teacher receives null guide,05teacher receives identical accepted bytes.
  No fake model-preparation metadata, free preparation window or clock reset.
  No additional05algorithm/sampling/LR change: this is the direct context test.
  Fresh TRAIN40 before/after each; same full heldout62x5 readouts. Analyze05versus04
  paired task outcomes plus both versus matched fresh base, missingness/preservation
  and native qualitative decisions; do not rank cross-cohort training means.

- Shared CPU inventory found the earlier89/176screen was overly restrictive:
  manifest checks ran BEFORE the actual pinned WorldState constructor normalized
  Sheets. The exact original hash checks pass after real constructor normalization
  for sampled Finance/HR/Operations/Support sources. Opt-in --normalize-sources in
  prepare_sdpo_guidance20.py records original/canonical data and actual constructor
  source hash, then applies UNCHANGED authority digest checks and current-tool/
  public-request checks. Full176-source read-only audit now admits123candidates:
  Finance9,HR6,Marketing8,Operations7,Sales4,Simple88,Support1. Evidence:
  balanced-task-source-audit.json. No predicates/digests rewritten, environment
  pins/fixtures changed, model/tool calls or optimizer updates, or04capsule staging.
  Recovered metadata is not native replay qualification; compare fresh native
  generated states before fitting. Native replay now also records official pinned
  score_world outcomes/failed/excluded assertions, so API execution success cannot
  masquerade as semantic task success.52focused CPU tests/Ruff/bash passed.

- Current03completed10actual updates/moments/EMA at elapsed4816.44s and is now
  evaluating post10heldout62. Its fresh TRAIN40readout is closed/error-free:
  success50%to50%;partial65.5556%to70.5556%, paired20-task95%delta interval
  [-1.25,+14.375]points. Legacy tool-error counters stay0despite known API errors.
  This is training-only and includes known frozen Jira/Buffer issues; no heldout
  improvement claim. Store: experiment03-plain-r3/checkpoint10-training-outcomes.json.
  Study25actual updates total (01ten/02five/03ten), remaining25ceiling. Never start
  04until03terminal and real GPU free.04/05windows remain unstarted.

- Human authorized replacing the two replay-broken training tasks for04–05:
  `simple.new_lead_sf_jira` and `simple.buffer_company_award`. Keep20tasks,
  current03source/cohort unchanged, and exact heldout62 unchanged. Metadata-screened
  candidates are Gmail meeting reply and Zendesk billing ticket; both have eligible
  historical perfect demonstrations and clear current protected-set checks.
  Slack CEO announcement was rejected because it belongs to a protected fixture.
  These are NOT yet native-replay-qualified or frozen. Require pinned invoke/capture
  full dependency replay, current contract/canonical-state binding and new20-task
  native qualification before fitting; old20-task receipts cannot be relabeled.
  All experiment-specific qualification/preparation belongs to04's own unstarted
  two-hour window. See replacement-task-decision.json for source/exclusion proofs.
  Later03-versus04 comparison is a context-and-data bundle, not context-only.
  Latest owned03check confirmed4actual optimizer/moment/EMA updates, step5collection;
  total study19actual updates. No final student reward readout exists yet.

- CPU source review during02baseline collection:existing feedback-status mode
  sets prepared guidance to null and projects raw hindsight to status facts.
  It must not be silently described as compressed prepared guidance plus status
  feedback. Any later comparison retaining prepared guidance while removing raw
  private assertion/response payloads requires a separately named explicit mode,
  original-token/privacy tests and a hypothesis frozen from TRAIN evidence.
  This does not preselect05or alter running02; decide after actual earlier-arm
  training/guidance quality evidence, never from heldout task semantics.

- Decision: Freeze02guide-removal control using01's actual five-update training
  diagnostics and four original teacher/Adam decision reviews, before post-
  heldout results. Contradictory guide instructions and a successful sampled
  call within a failed episode motivate isolating guide effects first. Keep
  retrospective feedback, prompt shell and every learning/runtime knob matched;
  do not silently label this plain SDPO or train on heldout trajectories.
  Date:2026-10-19.
- Decision: New authorization is five independent fresh-base ten-update runs,
  50 updates maximum. Proposed mechanisms02–05 are hypotheses, frozen only
  after preceding TRAINING diagnostics; no adapting on evaluation outcomes.
  Rationale: user requests learning from previous experiments. Date:2026-10-19.
- Decision: Prefer native MTP with two speculative tokens; initial runtime gate
  makes zero optimizer updates. DSpark investigation is retained as history,
  superseded by explicit MTP instruction for this campaign. No pin changes.
- Decision: After r1 failed invariance, qualify one MTP token under r2 with the
  same threshold. Also fixed a missing minimal-feedback argument in the native
  gate harness; r1 failed before reaching that branch. Neither gate trains.
- Decision: Select MTP2/GDN Triton after r4 passed exact thresholds. The pinned
  CUDA path dispatches pure speculative batches to a fused recurrence while
  mixed batches use a different recurrence; Triton removes that dispatch
  difference without a weight/pin/source change. This is source-backed causal
  evidence plus one controlled gate, not a universal numerical proof.
- Decision: Correct teacher wording asks for the next native assistant action,
  not an assessment or critique. Preserve actual student prefix/action IDs.
  Native WireTaskData drops initial_state; use the existing ID-joined retained
  native task reader for prepared guidance, rather than guessing missing state.
- Decision: After staged r3 failed11tasks, use per-check generation/review and
  a concise linear native-tool procedure (sdpo_checkwise_preparation.py). Keep
  every frozen goal/harm/task and one bounded repair per rejected check/workflow;
  no action-ID DAG or pooling earlier approvals. Same4B generates and reviews.
  Rationale: actual rejections showed DAG-ID confusion, review demands for
  invisible evaluator bindings, and4K review truncations. These are preparation
  revisions, not completed ten-update experiments. Date:2026-10-19.
- Decision: Recognize explicitly public date normalization and public-word query
  composition, with an audit. ISO2026-03-05 is public information when the task
  says March5,2026; hidden IDs, private policy titles and unrelated dates remain
  rejected. Runtime invariance thresholds are unchanged.
- Decision: Permit explicit JSON composition only when its fields and every
  value are public; reject any substituted hidden value. Give check/workflow
  literal-validator failures one model repair with the same validator afterward.
  This fixes false-positive representation checks and metadata prose, without
  accepting a failed review or changing native scoring/runtime tolerances.
- Decision: Reuse individually approved r4 contexts in r5 after source binding
  validation; do not regenerate correct work or drop failed tasks. Pin the exact
  accepted-guidance input in the new source manifest and record the parent file.
  This is versioned context-dataset assembly, not pooling failed training runs
  or benchmark episodes. No r2/r3 contexts are included. Date:2026-10-19.
- Decision: All ordinary student prefixes/action IDs remain intact. Teacher-only
  guidance distinguishes public requirements, historical procedure, current
  observations, required retrieval and preservation. No raw feedback CE or
  corrected-call gold CE silently added. Any algorithm hybrid is named explicitly.
- Decision: Each experiment has at most7200seconds TOTAL including startup,
  diagnostics, training and evaluation, and at most10 actual optimizer updates.
  Failed partial attempts consume their applied updates and retain artifacts;
  no automatic crash resume or budget reset. No full-training/LFM queue resumes.

## Context and Orientation

scripts/qualification/sdpo_agent_ablation.py owns actual on-policy collection,
original-token association, TRL top20-plus-tail reverse KL and EMA. The optimized
runtime is self_feedback_vllm.py, pinned vLLM0.29.1.dev4 with block128. Native
Qwen MTP weights are in target revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
qualify_sdpo_mtp.py gates actual LoRA/token-score/native call-result-final output.
sdpo_speculation.py provides explicit MTP configuration and probe populations.
luna_solver_stages.py/luna_solver_separation.py/luna_policy_guidance.py provide
source separation and model-generated preparation. Never reuse27B cached stages.

## Plan of Work

First qualify the requested runtime, independently of learning. Prepare an
immutable stage capsule and record source hashes/container receipt immediately.
Screen current audit against exact heldout and protected identities. Select all
nine eligible nonSimple tasks plus eleven tool-diverse Simple tasks if their
guidance passes source-grounding checks; explain actual coverage. Freeze before
training. Use20tasks x2 online trajectories,32 distinct complete transitions per
update, failure/terminal selection with successful retention. Audit native goal
findings separately from legacy success, preserving task-level causal limits.

Each run starts from the same BF16 base,LoRA16/alpha32/dropout0,LR1e-6 constant,
reverseKL1,top20+tail,EMA update rate.01 unless an evidence-backed later arm
explicitly changes one named mechanism. Native thinking mode stays matched.
Freeze evaluation before launching01. Use full frozen heldout62 research
monitoring at0/10 with62x5,20turns,8192reply,32768episode,65536context,matched
greedy thinking mode. This benchmark was repeatedly inspected previously and is
not a pristine final test. Evaluation findings must not select later context.
MTP serving at65K needs a separate capacity gate; do not shorten evaluation.

## Concrete Steps

Current run is already accepted:inspect/monitor it rather than executing its
launcher again. After its own step0 heldout62outcomes AND successful native
repeatability files exist, the following **CPU-only** command on the workstation
exports an audited common plain base for later arms. Do not execute on an open
population or overwrite an existing destination; inspect retained reference first.
The exporter recomputes310unique episode IDs/62tasksx5, native summary, policy
identity and runtime gate, retaining missing scores and provenance. A later
consumer must pass the independent source/runtime/initial-adapter fingerprint.

    sudo -n env PYTHONPATH=/var/lib/posttrain/qualifications/sdpo-guidance-five-20261019-experiment02-plain-r4/driver python3 - <<'PY'
    from pathlib import Path
    from sdpo_shared_base import export
    cap = Path('/var/lib/posttrain/qualifications/sdpo-guidance-five-20261019-experiment02-plain-r4')
    export(cap / 'evidence/run', cap / 'driver', cap / 'heldout62.json',
           Path('/var/lib/posttrain/qualifications/sdpo-guidance-five-20261019-shared-plain-base02-r4'))
    PY

This is read-only with respect to the live job. It does not create a new model
run or extend any window. Never export historical MTP data as a plain control.

From the worktree:

    PYTHONPATH=scripts/qualification .venv/bin/python -m pytest -q scripts/qualification/test_sdpo_speculation.py
    .venv/bin/ruff check scripts/qualification/sdpo_speculation.py scripts/qualification/qualify_sdpo_mtp.py

SSH: timeout25 ssh -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes
-o IdentityAgent=none -o GSSAPIAuthentication=no -o ConnectTimeout=8
-i /home/hammad/projects/ai-infra/.state/ssh/dstack_worker dstack@192.168.110.12.
New capsules: /var/lib/posttrain/qualifications/sdpo-guidance-five-20261019-*.
Runtime readonly: self-campaign-2048-20261017-r5 under the same parent.
Image sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337
from registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job.

## Validation and Acceptance

Require actual optimizer moment steps1–10,EMA and changed rollout adapter hashes,
native task/episode identity and score coverage, no student privileged context,
whole-transition exclusions, call finishes/repeats/empty results and native
goal/harms evidence. Compare matched checkpoint student outcomes with paired
task-bootstrap uncertainty and missing-score bounds. Inspect3–4 failure decisions
per arm and actual teacher/logprob/Adam direction there. Loss decline, selected
token alignment or provider exit alone does not prove learning. Five executed,
audited experiments and honest final report are required for completion.

## Idempotence and Recovery

Check retained owned Docker state and real GPU processes before EVERY launch.
Never overwrite a capsule, compete with another workload, duplicate a submission,
mutate running source, change dependency pins, quantize, commit/push or publish
local scoring fixes. A failed essential MTP gate preserves evidence and remaining
updates; diagnose an in-scope repair under a fresh identity. Do not claim MTP
speedup or learning qualification from a startup-only gate.

## Outcomes & Retrospective

Experiment01-r3 trained10updates with independently verified saved moment10 and
final EMA recurrence, then ended at the original time cap with243/310post-heldout
episodes. Training success55%to50%,partial72.05%to69.05%; full heldout score
unavailable. Earlier r1/r2 remain preserved zero-update setup/runtime failures.
Experiment02-r1 was stopped before any update. Cacheoff qualification r7 passed
its smaller scope, but02-r3's full heldout repeatability failed before updates.
Historical runtime repairs are terminal. User subsequently discontinued MTP
and authorized fresh two-hour windows for remaining02–05. Compiled plain
decoding passed composed native/capacity qualification; MAIN02-r4 subsequently
ended at its cap with5confirmed updates and no poststudent readout. MAIN03-r3
is now accepted/loading with no new update confirmed. Five complete experiment
results remain unproven; campaign has15confirmed actual updates at acceptance.
Arms03–05 require sequential training-evidence-based decisions.
Training decision audits exposed inconsistent prepared requirements and a
selected successful action from a failed episode. Candidate preparation now
interprets intact source predicates before output redaction and reviews the
complete guide;17CPU checks and all20actual pinned-tokenizer first-stage inputs
pass, but new generated quality remains unqualified. The original01timeout was
enforced; no evaluation-only extension is authorized until the user answers.

MTP CUDA r1/r2 failed real numerical gates, not task performance. Plain r3 and
MTP Triton r4 passed initial runtime gates, zero updates. Preparation r8 exited0
with20bound contexts after retained semantic rejections and repairs. Toy40request,
16token probe timings were0.4131s plain versus0.2392s MTP (about1.73x), but no
native long-trajectory throughput or changed-adapter speedup is established.

Current next action:monitor accepted MAIN plain03-r3 Docker
f640562727534b7923de4973e598a5e32608ac62e67f3a0c06615cf22d1c2fcc
through10actual updates and post-readouts within its renewed11:44UTC deadline.
Composed native/capacity qualification, exact source compatibility and CPU
admission have passed; shared plain baseline is bound. Never rerun historical
gates by rote, duplicate accepted03, or resume incomplete02. Because02has no
poststudent readout, no matched10update03-versus02 learning comparison exists.
User authorized fresh02–05windows and removed MTP; no further time/protocol
decision is pending. The older MTP run/deadline history below is retained
evidence, not current launch authority.
03-r2 Dockerd2372486440bc07e0bceb81f06184bad4991cf1ea207e99e47d516c11a0466a7
is terminal exit1/zero updates after47/3165native repeatability failures.
Never duplicate/restart/mutate its source.03-r1 is terminal exit1/zero updates
after70native repeatability failures. r10eager numeric gate is terminal exit0,
but03-r2subsequently failed its full native gate before actor training. r8/r9 are historical
terminal exit0.02-r3 is terminal, not a live training
job or eligible shared control. Do not weaken invariance gates or reset the
new plain02 deadline09:43:59UTC. Historical02 deadline03:48:05UTC and03
deadline05:26:50UTC applied to preserved MTP attempts. Each future plain03–05
receives its separately authorized window when its own preparation begins;
retries keep that deadline. Qualified shared plain control is the complete
310-episode02-r4 baseline, subject to exact source/runtime/input fingerprint
checks. Decide04from TRAIN diagnostics, not heldout outcomes. No full256/LFM
queue may resume.
Stopped02 Docker2712c945f2437876450e5e921523348c85ee60355d9df6e8fd9af0349431f7ed
is retained at zero updates; never resume it or mutate its source. Reconcile
fresh baselines, actual updates, native teacher/Adam evidence and full readouts.
The existing01checkpoint is preserved, not resumed. No01evaluation outside its
original budget without an answer. No full256/heldout-sweep/LFM queue resumes.

## Artifacts and Interfaces

Evidence: docs/research/evidence/sdpo-guidance-five-20261019. Persist an explicit
ledger with experiment number,hypothesis,falsifier,source/config hashes,run ID,
caps,start/end,actual updates and results before each submission. Keep native
Verifiers traces authoritative; no model/judge calls outside this workstation.

Revision2026-10-19: supersedes the unlaunched guidance20 plan with the user's
five ten-update experiments and explicit native MTP request.


2026-10-19 completed four-context qualification: two distinct one-update diagnostic arms and read-only tool-gradient audit finished. Combined student failure success25% (base25%, control12.5%); partial70.45% (base56.25%, control42.12%), retention100%. Blog regressed and campaign forbidden-token signal still points the wrong way. Not qualified for training restart; automation remains paused. See docs/plan/combined-teacher-context-qualification.md and docs/research/evidence/combined-teacher-qualification-20261019-r1/qualification-verdict.json.
