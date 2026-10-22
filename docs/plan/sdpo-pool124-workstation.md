# Run SDPO on the complete 124-task training pool

This living plan follows docs/templates/PLAN.md. Work in
/home/hammad/projects/worktrees/rl-rft on codex/rft-warm-start. Maintain Progress,
Surprises & Discoveries, Decision Log and Outcomes & Retrospective. This is a
project research experiment; the frozen product baseline and dependency pins
do not change. Do not commit, push, edit accepted capsules or use external models.

## Purpose / Big Picture

Correct the mistaken carryover of eight ablation tasks into a long campaign.
Use every task in automationbench-train-pool-luna07-v1 (124 identities), choose
32 distinct tasks per optimizer update, run a fresh 50-update campaign, and
compare checkpoint50 against a fresh matched base on the separate heldout62.
Batch32 means one complete assistant transition per selected task, averaged
over microbatch1/accumulation32. Two fresh trajectories per task provide native
feedback and optional successful sibling context; they are not extra tasks.

## Progress

- [x] User reaffirmed evaluation every 10 optimizer updates. The full-pool
  profile enforces evaluate_every=10 and save_every=10: matched fresh base at0,
  then full heldout62 evaluations at10/20/30/40/50. Each is62tasks x5attempts,
  with unchanged64K context,8192 reply,32768 episode tokens and20turns.
  Evaluations0/10/20/30 completed;40/50 are absent because the run failed at38,
  not because evaluations were skipped. This cadence remains required for any
  subsequently authorized long campaign; it does not authorize restarting a
  failed attempt. Separately capped six-update ablations evaluate at0/final6.
- [x] TERMINAL FAILURE supersedes all earlier active-run notes: full50r1 exited1
  at2026-10-18T12:26:46Z (17:26:46PKT), after38logged optimizer updates while
  collecting online population39. vLLM AsyncLLM reported EngineDeadError;16native
  ProviderError500episodes followed and the strict collector stopped. Exact
  engine-death trigger is unconfirmed. DockerOOMKilledfalse; bounded privileged
  kernel journal/ring checks did not corroborateOOM orGPUXid. No stale-policy
  mismatch appears in retained calls. Preserve the container/logs; do not guess
  a kernel cause, restart or claim a completed50update experiment.
- [x] Terminal native audit:3992unique episodes/47closed populations,38actual
  updates,1216selected actions across124tasks (9or10visits).39collected64episodes,
  only48scored and16transport-error scores missing;39had ZEROoptimizer updates.
  Never zero-fill/drop these failures. Whole exclusions before applied updates:
 168replylimit/110overlong/67context;zero training truncation. Last physically
  saved/evaluated checkpoint30;40/50and a saved38adapter do not exist. Native
  GPUcompute and Dockeractive workload checks show capacity released.
- [x] Rawlogs/state/failure summary preserved in full-run-engine-failure38-r1.log,
  full-run-terminal-state-r1.json and full-run-terminal38-summary-r1.json. Auditor
  R2rejected the failed population because native persistence omits traceback:null
  while the in-memory summary includes it. Read-only analyzerR3 canonicalizes
  ONLYthat unset optional field, keeping every error/score/denominator unchanged;
 18targeted tests pass. OriginalfailedCPUaudit retained. R3manifest
  e1e778bf3b0eb5e17047ca7ad49aef6759fe98cff970ae8fcc79b6ed5ddb3e74,
  remote/tmp/sdpo-pool124-analysis-20261018-r3, exact-image terminalaudit exited0.
  Evidence full-run-terminal38-native-audit-r3.json and
  failed-population39-summary-comparison.json. Accepted training source unchanged.
- [x] Full50continuation is INCONCLUSIVE/unqualified because the study stopped
  early; checkpoint30interim partialCIcrosseszero and4newpreservation identities
  remain. Do not replace the missing50endpoint with the best earlier checkpoint,
  reconstruct/resume38fromlogs or startLFM/full250automatically. The separately
  authorized bounded09/10experiments remain feasible fresh runs; audit this
  failure and use their strict in-attempt GPUgates before any model-learning claim.

- [x] User authorized pool124/32distinct tasks per update/50-update outcome gate.
- [x] Requested stop and retained evidence for obsolete eight-task campaign at
  177 logged updates, last saved checkpoint175; Docker exited143, GPU free.
- [x] Implemented coverage/per-task admission/profile guards and launcher;
  27targeted tests and Ruff passed. Exact pinned-image CPU objective/gradient,
  support/Adam diagnostics and same-mode baseline preflight exited0.
- [x] Qualify actual batch32 with four optimizer/EMA updates, visiting all124
  training identities, plus full64K-serving capacity/native gates.
  Accepted qualifier sdpo-qwen35-4b-pool124-b32-qualify-20261018-r1,
  Docker ca434e41bbc3ba09b7a7c3267df58f0c78d5e934e757eaaa454860976458a1f5,
  started2026-10-18T08:18:02Z and exited0. All4actual moments/EMA,576native
  episodes,124distinct selected task identities, finite changed saved adapters,
  native scoring and64K/32slot gates passed CPU audit.
- [x] Fresh50 run accepted after qualification: sdpo-qwen35-4b-pool124-b32-50-20261018-r1,
  Docker8c02afc6ebe2dfd83e38f373db9efe0e0eaf1b877b1cf880a8e2af0d21e4e3c1,
  started2026-10-18T08:55:08Z. Startup/step0 baseline, zero full-run updates
  confirmed at handoff; qualification updates are separate, never pooled.
- [x] All124 identities verified in the actual fresh50 campaign through10updates:
  32distincttasks/update,320selected transitions,100whole-transition exclusions,
  zero training truncation. Native/task audit independently reconciles1388episodes.
  First two actual updates verified32distinct tasks each/64different identities
  in total. Update2 logged loss0.2988537,moment2/EMA2,LR1e-5; changing task
  populations mean raw per-update loss is not a controlled reward comparison.
- [x] Matched fresh-base heldout62 completed310/310episodes without errors:
  success51.6129%, partial81.1947%. First actual full-run update confirmed:
  step1/batch32/moment1/EMA1,loss0.2248746,LR1e-5,grad norm1.18743.
  First update selects32distincttasks;10whole transitions excluded,0truncated
  training sequences. Qualification's4updates are separate, never pooled.
- [ ] Run matched fresh-base/checkpoint50 heldout62 evaluation and native audit.
  Step10heldout evaluation completed310/310scores/noerrors. Full success increases
  51.6129->58.0645% (+6.4516points, paired62-task95%CI[1.6129,12.9032]points),
  partial81.1947->80.7679% (-.4269points,CI[-4.1271,3.0489]points).
  Four previously failed Finance/Operations tasks now succeed; no previously
  fully solved task regressed. Partial outcomes are mixed, not an overall win.
- [ ] Continue only after outcome gate and qualified optimizer/EMA continuation.

## Surprises & Discoveries

The previous job used8training/4development tasks, not the intended broad pool.
It is retained separately and is not evidence about training124tasks. Its stop
receipt is evidence/user-stop-for124-receipt.json inside the old capsule
/var/lib/posttrain/qualifications/sdpo-campaign-b32-250-20261018-r1. Logged update177
is a lower bound on applied work if interruption fell inside an unlogged update;
do not claim exact checkpoint177 continuation or pool attempts.

All previous24-task development and research-reserve subsets lie inside124.
They are explicitly retired as independent tests for this new campaign. The
original62-task split remains disjoint from training. Once used for the50-step
continuation decision, it is a monitoring/selection set for this campaign, not
an untouched final proof. Prior base-model benchmark results also exist.

## Decision Log

- Decision: fresh base, same nonthinking SDPO recipe, data breadth changes.
  Rationale: isolate breadth rather than silently mix stopped adapter history,
  Luna, thinking or corrected-call CE. User authorization2026-10-18.
- Decision: deterministic hashed round-robin across124;32unique tasks/update.
  Rationale: covers the entire pool within four updates, balances visit counts
  to within one, and never substitutes an easier task after admission failure.
- Decision: two stochastic trajectories per selected task; one selected action
  per task. Preserve failure-frontier action selection and whole-transition
  exclusion. A task without any admitted action is an explicit shortfall; never
  duplicate another task or truncate to obtain a batch.
- Decision: heldout62 evaluation at0/10/20/30/40/50, as explicitly requested.
  Use the same population and fresh-base control; only the predeclared50-step
  gate determines whether to continue. Repeated use makes this monitoring,
  not a pristine final benchmark claim.
- Decision: keep32fixed training diagnostic tasks for inexpensive progress;
  never call these independent validation. Baseline/every10 diagnostics use two
  greedy repetitions; actual generalization check uses all62separate tasks.
- Decision: heldout comparison uses identical native nonthinking greedy settings
  for fresh base and candidate, frozen environment a0c7378b179e2b267382b8125192876502d82f89,
  ordered task digest ed81cf1fe88f68bb8420bb46c5fdf3afde24660c7043812da680bb1c7bc9196f,
  62tasks x5,20turns,8192reply,32768episodeoutput,65536context. Qualify serving
  capacity and native adapter/tool continuity before that separate evaluation;
  no shortened benchmark budgets or comparison to historical thinking-mode runs.
- Decision: predeclare partial credit as the50-step learning signal, success and
  native preservation as safeguards. Continue only for positive paired taskmean
  partial-credit gain with95%paired task-bootstrap lower bound>0, success not
  lower and no added broken native preservation identities/counts. Missing
  scores stay missing: require conservative gain bounds>0 plus separately
  labelled complete-task sensitivity; inadequate guard/score evidence is
  inconclusive. No tuning on individual heldout failures. No automatic crash
  resume: qualify saved optimizer/EMA/RNG continuation before extending toward
  the prior250update ceiling. Otherwise retain and report the50step result.

## Outcomes & Retrospective

Qualification passed and full50 is active after checkpoint10. Complete matched
heldout evaluation shows four new Finance/Operations successes, mixed partial
outcomes and five new preservation failures despite fewer total violations.
This is an exploratory result, not the50-step continuation decision. Original bounded
SDPO slots09/10 remain unsubmitted. They must not use the now-trained reserve
as independent evidence after pool124 training; revise their reserve comparison
before launch. This campaign does not count as a bounded-search ablation slot.

## Context and Orientation

scripts/qualification/sdpo_agent_ablation.py owns the project SDPO loop and native
teacher/token alignment. sdpo_long_campaign_config.py owns existing bounded/long
guards; new sdpo_pool124_config.py owns exact pool membership, task cycling and
per-task selection. Agent.collect retains native episodes and scores without
coercing missing scores to zero. No external scorer or Luna calls are used.
The124/62 fixtures in scripts/qualification/fixtures are source authority.

## Plan of Work

First implement pure pool selection/profile guards and test full50step coverage,
uniqueness, exact membership, heldout exclusion and admission shortfalls. Add
explicit pool124 qualification/train profiles without relaxing old ablation
caps. Stage all sources/configs/hashes in a new immutable capsule. Qualify the
native tool bridge, actor/rollout parity,32slots and four actual batch32 updates;
audit moments, EMA, selected task identities and native outcomes. Then launch50.
The same process switches request budgets for full-budget heldout evaluations
without changing optimizer/EMA state or competing for GPU capacity.

## Concrete Steps

From /home/hammad/projects/worktrees/rl-rft:

    PYTHONPATH=scripts/qualification .venv/bin/pytest -q scripts/qualification/test_sdpo_pool124.py scripts/qualification/test_sdpo_long_campaign.py scripts/qualification/test_sdpo_agent_config.py
    .venv/bin/ruff check scripts/qualification/sdpo_pool124_config.py scripts/qualification/sdpo_agent_ablation.py
    git diff --check

SSH dstack@192.168.110.12 with key
/home/hammad/projects/ai-infra/.state/ssh/dstack_worker, -F /dev/null,
BatchMode=yes,IdentitiesOnly=yes,IdentityAgent=none,GSSAPIAuthentication=no,
ConnectTimeout=8 and bounded outer timeout. Inspect exact run identities and
real GPU compute before every submit. Record exact capsule/run/container/hash
receipts here before claiming submission or updates.

## Validation and Acceptance

CPU tests must prove32distinct tasks/update, one transition each, all124visited
in four updates, visit counts12or13after50, no heldout overlap, no truncated or
duplicated task fallback. GPU qualification must prove four actual optimizer
moment/EMA steps, changed finite adapters, correct native call/result/final
channels, exact batch invariance and selected task/episode provenance. Training
expects3200online native episodes plus384training diagnostics and1860heldout
episodes at0/10/20/30/40/50, subject to no
silent population changes; missing scores/errors and exclusion causes remain
explicit. Fifty-update completion alone is not an improvement claim.

## Idempotence and Recovery

Retain stopped177-update attempt and all old ablations. Use fresh run IDs;
refuse existing Docker name/evidence, busy GPU or source-hash mismatch. Never
edit accepted capsule, alter environment pins, weaken gates or quantize. On
failure retain logs/inspect/native artifacts, diagnose before any new attempt.
No automatic exact crash resume; no unrelated workload/automation changes.

## Artifacts and Notes

Evidence belongs under docs/research/evidence/sdpo-pool124-20261018/ locally and
the new capsule evidence directory remotely. Runtime source is the read-only
/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5 capsule. Pinned
image registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337.

## Interfaces and Dependencies

Same Qwen3.5-4B851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a BF16, optimized vLLM
0.29.1.dev4 alignedblock128 and pinned TRL SDPO loss. LoRA16alpha32dropout0,
reverseKL1/top20tail/EMA.01/AdamWdecay.01/clip1. LR1e-5updates1–10, cosine11–20
to1e-6, constant thereafter. Training16Kcontext/2048reply/12288episode/8turns,
32episode/server slots16GiBKV. No raw feedback CE, corrected gold CE or teacher
labels in student prompts. Teacher alone receives native hindsight/reference.

The single serving engine has65536maximum context throughout this campaign;
training admission and request limits remain16384/2048/12288/8turns. During
heldout evaluation only, requests switch to65536/8192/32768/20turns and5replicas,
then restore training limits before collecting the next update. Qualification
must validate native continuation at both reply caps and near-limit64K serving
with32in-flight short probes; that does not claim32simultaneous forced8K replies.

Revision2026-10-18: created for corrected full-pool authorization; supersedes
the eight-task campaign and retires overlapping internal evaluation reserves.

Immutable capsule /var/lib/posttrain/qualifications/sdpo-pool124-20261018-r1,
source-manifest SHA2569ce6cab7ab3afacf24d5681bcc1acfff1166f1ee827b064ed60b5e3c749bde8f.
Host queue PID2945938 launched the qualifier and will audit all4updates/576native
episodes, source hashes and file states before accepting the50step run named
sdpo-qwen35-4b-pool124-b32-50-20261018-r1. Never rerun the queue or qualifier if
either identity is retained; inspect exact owned states first. Queue log and
submission receipts are in capsule/evidence. A queue PID/log alone is not proof
of GPU progress; monitor the exact Docker state and actual optimizer evidence.

Final qualification receipts are retained locally in
docs/research/evidence/sdpo-pool124-20261018/qualification-gate-r1.json,
qualification-checkpoints-r1.json and qualification-native-r1.json. The selected
call/result audit independently checks128selected actions across124identities;
no selected tool call lacks its native responses. This is provenance, not a
proof that each action is correct. Full-run accepted receipt is
training-submission-r1.json. The host queue may exit normally after submission;
monitor accepted Docker8c02afc6... rather than restarting the queue.

Read-only checkpoint analysis is implemented outside accepted training source in
scripts/qualification/analyze_sdpo_pool124.py. It independently reconciles native
episode/task identities, shifting32-task training populations, one selected action
per task, actual optimizer/EMA counts, policy versions, native scores and declared
completion populations. It separates strict means, missing-score identification
bounds and paired task-bootstrap sensitivity; tasks missing in both arms remain
explicitly excluded, and resampling retains all5repetitions per heldout task.
Ten focused analysis/config tests and Ruff passed. No GPU/model calls were used.

The exact analyzer was staged at /tmp/sdpo-pool124-analysis-20261018-r1 (outside
the immutable training capsule), source-manifest SHA256
d2394d013babd2914494ad4eebe2712f9025d7442344fb2dc977f9176db868df.
Pinned-image CPU native audit exited0 and independently reconciled576unique
qualification episodes/124selected task identities. Durable output is
qualification-independent-audit-r1.json. Full-run checks require5444episodes,
62completed populations and all124selected task identities; live unfinished
populations are not treated as completed denominators. Separate native
initial-state preservation/guard auditing remains required, not inferred from
missing assessment metrics. Current full run was verified running at startup,
zero updates, with no completed baseline populations at2026-10-18T08:59:22Z.

First real full-run audit reconciles438unique native episodes:64training
diagnostics,310heldout base and64pre-update stochastic training episodes.
The online pre-update scores are not learning gains. Baseline training diagnostic
success40.625%,partial63.5258% refers to32tasks/short training budgets; it is not
comparable to the old8-task cohort or heldout62's larger evaluation budgets.
Native initial-state preservation audit at base has920applicable heldout checks,
158violations across79episodes, unknown0; training262checks/35violations,unknown0.
These are benchmark preservation assertions, not a complete independent harm
taxonomy. Retain identities as well as counts for the50-step continuation gate.
See baseline-native-preservation-r1.json and full-run-early-audit-r2.json.

AnalyzerR1 had floating-point cancellation in an identification-bound upper
endpoint (~1e-16) when no scores were missing. R2 uses math.fsum and adds the
missing count before division; eleven focused tests passed, including exact
collapsed-bound ordering. Model/data/optimizer source was unchanged. New CPU
source /tmp/sdpo-pool124-analysis-20261018-r2 has manifestSHA256
3c967feb07b848adeea780ac1ca5d9a876f0e381b42189fcb853879d51e0b6ed;
its read-only pinned-image audit exited0 on438episodes/32selected tasks/1update.
R1 artifacts remain retained; R2 is authoritative for subsequent bounds.

Checkpoint10 saved-file audit passed: actual optimizer moment10, LR1e-5,
finite changed student/teacher tensors, RNG/optimizer/adapter file hashes match.
Read-only helper scripts/qualification/audit_sdpo_pool124_checkpoint.py and three
counterexample tests were added outside accepted source (14focused tests passed,
Ruff clean). SourceSHA256a21ae90eaae1ad40133bfa4a31002493c32067468f635b572699a0c6989b6307;
remote /tmp/audit_sdpo_pool124_checkpoint-20261018-r1.py. It audits a single
closed10step checkpoint, never demanding future files or claiming exact resume.

Step10native preservation totals improve from158to125violations/920checks,
79to55affected episodes among140applicable heldout episodes, unknown0. However,
five NEW broken check identities appear across four tasks: HRpromotion wrong
employee update and wrong recipient, HRequipment unwanted Jira issue,
HRworkforce-plan forbidden email and marketing contest disallowed Sheets row.
Each occurs in all5replicas. No existing broken identity worsens, but these
new identities violate the predeclared no-new-preservation-failure safeguard.
Training diagnostics preserve262checks with35->38violations (13->14episodes).
Initial-state variants94match32train+62heldout tasks; no mismatched task state
explains these changes. These are native preservation constraints, not a
complete independent harm taxonomy. At10 the50-step continuation gate is not
satisfied; continue the fixed authorized50-step study unchanged and decide at50.

Four new complete tasks are finance.xero_expense_claim_review,
operations.contractor_badge_expiration, operations.google_ads_budget_alert and
operations.holiday_coverage_planning. Largest partial regressions are marketing
social-crisis response (83.33->27.27%) and equipment lease processing
(55.56->0%). Partial comparison95%CI crosseszero; no definitive aggregate
partial-credit regression or clean overall improvement is claimed. This is one
exploratory monitoring checkpoint; the50-step primary decision remains fixed.
Durable evidence: full-run-audit-through10-r2.json,
native-preservation-through10-r1.json, checkpoint10-comparison-summary.json,
checkpoint10-files-r1.json and training-coverage-through10-r1.json.

Remaining bounded ablation preparation (CPU only, 2026-10-18): original GPU slots
remain8/10. A broader offline Luna inventory inspected961saved episode files;
160 belong to the124training identities. There are67recorded-success sources
across62tasks with exact task/state/assertion/user bindings and currently
available domain tools. Domain coverage is50Simple,5Operations,4HR,2Sales,
1Finance,0Marketing,0Support. These are candidates, not admitted references;
the other62training tasks must not be dropped to make Luna coverage appear full.
InventoryR2 corrects a misleading output-field label fromR1; source admission is
unchanged. Evidence: luna-pool124-offline-inventory-r2.json.

New read-only review_sdpo_luna_pool124.py verifies source and snapshot byte hashes,
retained state-chain continuity, bound initial state and recorded endpoint before
current native rubric admission. Four focused tests and Ruff pass. CPU review
source manifest0ea9e087afaf858ff99f17881084d3bd6572c56717a898144c8029cb4872dddd
is retained at /tmp/sdpo-luna-pool124-review-20261018-r1 on workstation. The first
CPU container exited2 because the image's default entrypoint treated timeout as
a runtime subcommand; retained unchanged. Corrected entrypoint-only CPU wrapper
sdpo-luna-pool124-cpu-review-20261018-r2 exited0 and audited67sources, admitting
ZERO:58initial-boundary comparisons differ and9snapshot maps are serialized JSON
strings. No rubric pass is claimed because these gates precede rubric execution.
Evidence: luna-pool124-current-rubric-review-r1.json. A bounded current-image
diagnostic on HRoffboarding localizes differences to generated Google Sheets row
IDs and meta.allowed_services. Raw task rows omit id, so independently loading
the raw state can generate a different identity; runner.py also initializes
allowed_services. This is an audit-initialization issue to resolve against the
pinned runtime, not evidence that Luna actions failed. Do not discard these
fields or weaken admission before proving the actual initialization contract.
The serialized-map reader also needs an explicit regression case. No GPU slot,
external call, accepted-source edit or model update was used for this preparation.
At last direct observation the accepted full-pool run remains running with15
actual optimizer/EMA steps; next heldout population is scheduled at20.

CPU reference review follow-up: the pinned runner sourceSHA256
6d64f3b9580c2b5ba675d1cfa64a710d55d13a2bae066ba1298fc80835964f50
initializes services and factory-created IDs. Importing its full legacy runner
fails on installed verifiers.types; CPU wrapperr3exit1 is preserved. The audit
instead executes the exact four pure initialization AST definitions from that
pinned file without importing its old execution API. Only omitted factory-created
id fields may bind to captured IDs; seeded IDs, record values, collection lengths
and all other initialized state remain strict. Sources whose tool arguments use
these generated historical IDs are rejected, without remapping actions. Tests
also verify legacy module top-level code is never executed by helper loading.

CPU wrapperr4exit0 (source manifest860cd0ddb8213279c77aadf30550ba5c8db7c23a9cdd6befa3d4f717a41ca7a6)
admits five task references,45current native checks all passing: HRoffboarding,
HRteam-restructure,HRsafety,HRrecruitment analytics and Financeoverdue-invoice.
Other53retained-state candidates remain rejected at initial-state binding.
One inspected Simple example differs in an unseeded Gmail thread_id and clock;
do not assume historical relative-time arguments are valid in today's task.
Nine original candidate snapshots are archive placeholders, not JSON state maps;
inventoryR3 rejects missing payloads explicitly (13of160training-source entries
overall). Updated inventory has58bound/tool-compatible sources/tasks, still not
58admitted references. Earlier inventories/reviews remain retained.

Evidence: luna-pool124-current-rubric-review-r3.json and derived strictly passing
luna-pool124-admitted-references-r1.json. The derived bundle's complete-check flag
applies ONLY to its five admitted references;119training tasks need native-feedback
fallback. Fifteen focused reader/binding/context tests pass; Ruff clean. All five
rendered references use tools available to their current native tasks; serialized
observations are5453–27505characters. Token-length/live prefix/teacher gradient
and GPU native improvements remain unqualified. This is CPU preparation for
bounded09, not a ninth GPU launch, full Luna coverage, or a learning result.

Checkpoint20 was actually reached and its independent read-only CPU file audit
passed: saved optimizer moment20, finite student/teacher tensors, exact adapter,
optimizer and RNG hashes, LR1.0000000000000002e-6 at the intended floor. Receipt
checkpoint20-files-r1.json. The logged update20loss is0.2582213,gradientnorm1.04743;
fresh training batches and the moving teacher prevent treating this scalar as
a fixed-example validation loss. Stage20training diagnostics closed64episodes;
full310episode heldout evaluation is still collecting. An expired950second
read-only SSH observer did not indicate failure: fresh exact Docker inspect
confirmed running, with148native heldout episodes persisted and fresh file times
at2026-10-18T10:55:56Z. A second read-only observer monitors the same owned run.
Do not calculate a partial-population checkpoint20headline or restart the job.

Checkpoint20 evaluation now CLOSED:310episodes/62tasks/5each, all success/partial
scores present, no scoring errors. Independent CPU native audit covers2402episodes,
26closed populations and640selected transitions across124tasks (5or6visits each).
Whole-transition exclusions:88reply-limited,61overlong,33context-limited;0training
truncation. Heldout has5max_turns stops,1incomplete generation and0context refusals;
these stop/finish facts are distinct from score correctness.

Heldout success54.8387% versus51.6129%base (+3.2258points;paired62task95%CI
[-3.2258,+9.6774]); partial83.5103% versus81.1947%base (+2.3156points;CI
[-0.6721,+5.6142]). Both intervals crosszero. Step10was58.0645%success/80.7679%
partial:success falls since10 while partial rises. Relative to base three full
task successes are gained (FinanceMonday billing,FinanceXero expense review,
OperationsGoogleAds budget) and Operationsrole-based access audit loses success.
Preservation violations125/920 versus158base remain lower in aggregate, but
7NEWbroken check identities (35replicas) appear across4HR/Sales tasks. Affected
episodes65/140applicable versus79base;unknown0. Initial-state variants94match the
same32train+62heldoutpopulation. This is preservation evidence, not comprehensive
harm measurement. The continuation gate is NOT satisfied at20; finish fixed50
unchanged and evaluate30/40/50. No adaptation to specific heldout failures.

Training32sentinel at20 success39.0625%,partial63.1457% versus40.625%/63.5258%base;
training-only diagnostics are not heldout evidence. Full artifacts:
checkpoint20-files-r1.json,full-run-audit-through20-r2.json,
native-preservation-through20-r1.json,checkpoint20-comparison-summary.json.

Checkpoint30 CLOSED and independently audited:310heldout episodes fully scored,
no scoring errors; saved moments/EMA30,LR1e-6,finite adapters and exact file hashes
pass. Native audit3416episodes/38closed populations,960selected transitions across
124tasks (7or8visits each), whole exclusions130replylimit/79overlong/49context,
zero training truncation. Heldout10max_turns stops,1incomplete generation and
0context refusals; initial-state variants remain94 and guard unknowns0.

| Checkpoint | Heldout success | Heldout partial credit | Preservation violations /920 | New broken identities vsbase |
| --- | ---: | ---: | ---: | ---: |
| Base | 51.6129% | 81.1947% | 158 | 0 |
| 10 | 58.0645% | 80.7679% | 125 | 5 |
| 20 | 54.8387% | 83.5103% | 125 | 7 |
| 30 | 56.4516% | 81.5370% | 100 | 4 |

At30success gain+4.8387points, paired62task95%CI[0,+11.2903]; partial+0.3423points,
CI[-3.1668,+3.7709]. Partial evidence remains inconclusive; success lower bound
toucheszero. Three new full successes versusbase (FinanceMonday,FinanceXero,
OperationsGoogleAds), no lost base full successes at30. Native preservation
improves in aggregate (100violations/45affected among140applicable episodes vs
158/79base) but four NEWbroken identities remain acrossHRpromotion,HRworkforce
andOperationstemporary-worker-placement (20replicas), no existing broken identity
worsens. Aggregate improvement cannot erase these new failures or imply broad
harm-freedom. The50continuation gate is not satisfied at30. Fixed50continues
unchanged with next evaluations40/50, no tuning on the named heldout failures.

Training32sentinel success40.625%,partial62.5350% at30 vs40.625%/63.5258%base,
training-only evidence. Receipts checkpoint30-files-r1.json,
full-run-audit-through30-r2.json,native-preservation-through30-r1.json and
checkpoint30-comparison-summary.json. All point estimates above use310/310score
coverage; uncertainty uses tasks as clusters, not310independent observations.
