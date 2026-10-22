# Find a useful bounded SDPO configuration for native tool agents

This living execution plan follows docs/templates/PLAN.md. Maintain Progress,
Surprises & Discoveries, Decision Log and Outcomes & Retrospective. Work in
/home/hammad/projects/worktrees/rl-rft on codex/rft-warm-start. This is an
unpublished project research prototype, not a public train operation. The frozen
product baseline is unchanged: public train.distill requires a frozen teacher;
this experiment explicitly investigates a moving self-teacher outside that API.

## Purpose / Big Picture

The user authorizes a new SDPO investigation after a completed negative SELF
investigation. Find whether the same model, conditioned on native hindsight
feedback, produces useful dense supervision for its own fresh tool trajectories.
Choose configurations by actual native success, partial progress, preservation
checks and complete workflow behavior, not declining distillation loss. SELF's
raw response-prediction auxiliary and repaired gold argument spans are absent.

## Progress

### Supplied research considered: verified bootstrap before compression

On2026-10-18 the user supplied Pasted text.txt from attachment069523f5-4165-409e-a3a3-0f06df431478
and explicitly allowed considering changed or combined algorithms. Its cited
primary papers were read. This expands the research design, not the completed
ten-attempt budget or the deferred LFM launch. No new GPU run is submitted.
The proposed order here supersedes the earlier research-only arm ordering below.

The supplied [OPSDC link](https://arxiv.org/abs/2603.05433) now resolves to CRISP,
version7, revised3July2026. The current abstract reports up to56% reasoning
compression on MATH-500 and38% on AIME24 for Qwen3-14B, with up to3.3points
MATH-500 improvement and AIME24 within about one point. Do not propagate the
pasted57–59%/41%/+10point figures as current-version results. The mechanism is
still useful: an extra conciseness instruction conditions a same-model teacher,
whose reverse-KL signal is applied on student prefixes. This is compression
research, not demonstrated long-horizon action correction.

[BIRD](https://arxiv.org/html/2607.15736v1),17July2026, directly addresses a
different limitation: prefix-local distillation may remain on off-track student
trajectories. It first samples concise solutions, filters by correctness and
performs prompt-switch SFT, then applies on-policy distillation. Qwen3-8B
MATH-500 results are86.2->92.0% accuracy and3099->1115 tokens. Its SFT-only
and cold-start controls are essential. Our proposed adaptation uses verified
native workflow traces to improve initial plans; it is not a math reproduction.
Its published training used an epoch and eight A800 GPUs, not our20-step budget.

[Rethinking self-distillation for thinking models](https://arxiv.org/html/2607.05184v1)
finds harmful privileged-context supervision around high-entropy branching and
self-correction positions. [Does self-distillation improve reasoning?](https://arxiv.org/html/2610.18613v1),
13October2026, reports weak privileged-information effects and weak or adverse
alignment with reward gradients in its reproductions. These are reasons to
include same-mode teachers with correct context versus absent/shuffled context
in diagnostic scoring, and to check actual repaired actions and fresh rollouts.
They do not prove our failures have identical causes or rule out all hybrids.

[AutoL2S](https://aclanthology.org/2026.findings-acl.831/) learns a switching token
using verified long/short reasoning and then a GRPO-style objective. It motivates
adaptive reasoning length but needs additional supervision and routing changes.
[CODI](https://arxiv.org/abs/2502.21074) aligns explicit and latent reasoning
states, with its headline3.1x result at GPT-2 scale. [DART](https://arxiv.org/abs/2506.11752)
adds a separate silent-thought pathway/module. Neither is a drop-in objective
change for the currently qualified autoregressive tool-agent runtime.

#### Revised preferred combination and necessary controls

Prefer a BIRD-inspired verified bootstrap followed by task-grounded SDPO/OPCD
before adding compression. Curate complete native trajectories for the same
32-task mixture: original task prompt to reasoning/actions and tool results,
with every scored goal and preservation check passing under the exact frozen
state/tools. Where no verified path exists, the candidate remains unqualified;
do not manufacture gold from historical Luna scores or sampled tokenwise argmax.
Existing Luna examples have no guaranteed hidden reasoning; never invent it.
Use local same-model generation or already verified offline sources, with no
external model/judge calls. SFT learns under the clean student prompt, without
teacher-only hints or goals leaking into deployment inputs.

The first candidate hybrid is sequential: a small verified SFT warm start, then
fresh online distillation from a context-conditioned teacher using verified
workflow evidence. It is a supervised hybrid, not plain SELF or plain SDPO.
Keep a control that stops after the same SFT warm start, so its gain cannot be
misattributed to subsequent self-distillation. Also compare cold-start SDPO on
the exact same curated32. A budget-compatible proposed split is4 SFT updates
plus at most16 online updates, not20+20; this is a pilot allocation, not a tuned
paper hyperparameter. All GPU sampling, gates and evaluation still share the
two-hour total cap; no currently exhausted budget is silently reopened.

Only after correctness and retention qualify, add a CRISP-inspired concise
teacher instruction on already successful trajectories. Preserve useful
verification, evidence retrieval and recovery. Do not give shorter failures a
reward, shorten tool-result evidence, or suppress backtracking simply because
it adds tokens. Compare the same warm start with and without this component.
If information-sensitive credit remains wrong, CREDIT or an outcome-anchored
policy-gradient hybrid is a separate next arm, not another simultaneous change.

Do not revive SELF's raw tool-response prediction CE as a generic correctness
penalty. Predicting an error string is not learning to avoid the error, and our
previous auxiliary-loss ablations did not qualify that combination. Keep that
term zero initially; retain a frozen-base/warm-start KL anchor as an explicit
separate retention control rather than choosing an untested coefficient by fiat.

The proposed minimal comparison is cold-start SDPO, bootstrap-only, bootstrap
plus grounded distillation, then that hybrid plus correctness-gated compression.
Record bootstrap and online update counts separately, along with action/target
token exposures and elapsed time. The bootstrap-only arm has fewer updates;
report that rather than pretending it is a compute-matched baseline. A further
continued-SFT control is needed if claiming on-policy distillation is better
than spending the same additional training on supervised updates.

Primary measures remain heldout task success, preservation and error recovery.
Measure reasoning tokens and total generated tokens per successful episode,
calls and wall time; preserve failure/timeout rates separately to prevent an
efficiency metric improving by dropping hard tasks. Inspect action-critical
argument and recovery decisions, not only whole-action mean likelihood or
the frequency of verbal markers such as 'wait'. No combined recipe is qualified
until verified native improvements and successful-action retention are observed.

### New direction after the closed ten-attempt search: curated32 and reasoning research

The user deferred the separately prepared LFM20 run before GPU submission.
Direct owned-state inspection confirmed no LFM container, train directory or
submission receipt. Retain its CPU preparation without launching it. The original
ten-attempt search remains complete; no budget is reopened by this new research.

On2026-10-18, the user asked to reconsider prior configurations on a handpicked,
more diverse32-task training mix and inspect Connected Papers for reasoning-mode
evidence. Task quality is a hypothesis, not an isolated causal finding: earlier
controls changed corpus size, exposure frequency and thinking mode, while actual
Adam diagnostics still showed some failed actions becoming more likely.

The candidate fixture is scripts/qualification/fixtures/sdpo_curated32_candidate_v1.json:
four tasks each from Finance, HR, Marketing, Operations, Sales and Support, plus
eight Simple tasks. All32 belong to the124 pool and none overlap heldout62. The
labels describe intended skills, not verified student competence. The known stale
sales.linkedin_job_change binding is excluded. Retained training-only source
metadata and a real CPU invocation of the pinned native initial-state assertion
registry are in docs/research/evidence/sdpo-curated32-20261018. All32 tasks have
readable initial states and no unknown assertion checks. This does not prove a
correct solution exists, offered tools can reach it or the teacher can teach it.

An important measured imbalance: each Simple task has1–2 initially-unsatisfied
checks; the four Support tasks have23–27. Check counts are not counts of required
API calls, and initially-satisfied checks are preservation constraints, not new
goals. Eight turns can contain parallel calls, so check count alone does not prove
budget infeasibility. Do not mistake domain balance for difficulty balance.

Before fitting, hand-review each prompt, source-state identity, available schema,
positive goals, preservation checks and an actually executed successful path.
Test the teacher's extra context at the original failure state: it must improve
the verified decision, not merely produce a fluent rationale. Label unsupported
repairs and repeated empty searches explicitly; never invent a successful path.
The proposed difficulty quotas are8 solved retention tasks,16 mixed-success tasks
and8 hard tasks with a verified teacher rescue. Assign these only after a fresh
base screen; historical Luna partial credit does not establish these strata.
Replace an unqualified candidate transparently before freezing the cohort,
keeping every exclusion and source hash. No substitutions after launch.

Use32 distinct task identities per batch, not32 repetitions of a smaller cohort.
Expose every task equally; account for reasoning/API/terminal target tokens
separately so long reasoning or a large task rubric cannot silently dominate.
One chosen action per task can still miss the first causally relevant decision.
The current failure-frontier sampler preferentially picks the last executed tool
action of a failed episode on nonterminal slots; it is not goal-aware causal
assignment. Inspect and compare first-error/goal-linked decisions explicitly,
and keep successful terminal behavior in retention diagnostics.

#### Connected Papers and original-paper findings

The live Connected Papers graph was inspected, including OPSD, OPCD, CREDIT,
SDPG and the reasoning-degradation follow-up:
[SDPO graph](https://www.connectedpapers.com/main/89e83ee3eb95843292b1f75f0e50b89e7667b828/Reinforcement-Learning-via-Self%20Distillation/graph).
Graph proximity is a discovery aid, not validation. The following claims were
checked against original papers, not graph summaries.

[SDPO](https://arxiv.org/html/2601.20802v2), Table12, explicitly sets Thinking=False
for its main experiments. Its scientific-reasoning and coding gains therefore
do not directly establish a thinking-enabled recipe. Its question batch is32,
with8 rollouts/question; this is distinct from32 selected action transitions.

[OPSD](https://arxiv.org/html/2601.18734v3), AppendixB/Table5, chooses a thinking
teacher and non-thinking student. It supports privileged reasoning context,
not the claim that its main student learns to emit explicit thinking. Its
limitations identify teacher comprehension of difficult examples as a bottleneck.

[OPCD](https://arxiv.org/html/2602.12275v1), Section4.2, explicitly trains thinking
Qwen3 students for math and uses thinking teacher/student on Frozen Lake.
Qwen3-8B math accuracy rises75.0->79.7% in its table. It consolidates experiential
knowledge/context through on-policy distillation. Our proposed counterpart is
short verified workflow guidance supplied only to the teacher; reproducing math
results on AutomationBench remains unproven.

[CREDIT](https://arxiv.org/html/2605.11613v1), Section4/Figures5–6, reports an
8.8-point gain from enabling thinking for CREDIT in its coding setting. It also
finds that thought-only teacher context can improve early then collapse, while
solution grounding stabilizes it. A contrastive teacher baseline reduces generic
feedback correlations. This is the strongest directly inspected positive
thinking-mode ablation relevant to our signal problem, but coding is not our
long-horizon workflow benchmark. Actual token direction and rollout evidence
would be required for our adaptation.

[Why self-distillation can degrade reasoning](https://arxiv.org/html/2603.24472v1)
tests thinking-enabled models and finds that confident, shortened traces can
coexist with worse unseen-task reasoning. Its task-coverage and fixed/moving
teacher ablations make diversity, uncertainty retention and frozen-teacher
controls relevant. It does not prove our regression has the same mechanism.

[I-SDPO](https://arxiv.org/html/2608.12957v1), dated13August2026, is within the
last three months. It routes all-incorrect groups to privileged distillation and
any-success groups to GRPO. Its scientific-reasoning results motivate reducing
teacher reliance as competence grows; do not label its reasoning-task gains as
an explicit thinking-on/off ablation.

[SDPG](https://arxiv.org/html/2606.04036v1) combines verifier policy gradients,
privileged full-vocabulary distillation and a reference-policy anchor; its math
results motivate a later outcome-anchored hybrid control. It is a different
objective, not a parameter adjustment to plain SDPO. [SGCD](https://arxiv.org/abs/2606.12634)
addresses long-horizon tool credit directly, but its reported method uses an
external LLM reference summarizer, which conflicts with the user's no-external
calls constraint. An offline deterministic variant would be a new adaptation.

#### Proposed ablation order; none launched

First establish a reviewed32-task non-thinking control with the existing best
recipe. For isolating curation, compare to a predeclared uncurated32-task control
matched by domain, exposure and budgets; changing8 to32 alone cannot isolate
quality. Then compare non-thinking versus thinking students using identical
32Kcontext/4Kreply budgets for both and the same verified teacher information.
An OPCD-style concise verified context arm and a CREDIT-style contrastive signal
arm follow only if the teacher demonstrates a real correction advantage. Keep
loss weights, learning rate, seeds and initial base matched unless the declared
mechanism changes them. Frozen teacher versus EMA is a separate controlled test.
Outcome-anchored I-SDPO/SDPG hybrids are lower-priority implementation work after
the basic signal audit, not a bundle silently added to the thinking arm.

Four online trajectories/task is a proposed sampling option, giving128 episodes
per32-task collection with32 concurrent slots; it must be identical across
matched arms and fit the existing two-hour/20-update ceilings. It is not an
already-authorized extension beyond the closed ten attempts. Preserve whole
transition exclusions without truncation. Record actual unique tasks, selected
actions, failed-versus-retained tasks and update/token exposure rather than
equating tasks, trajectories, batch examples and updates.

Freeze an internal development selection outside the new32 before fitting;
prior full124 training means it is not globally untouched. Select one primary
from development before another descriptive heldout62 sweep at0/10/20. Heldout
remains disjoint from updates but has been repeatedly inspected already. No
fresh holdout or independent generalization claim is manufactured by relabeling.
Success, preservation and exact failure-linked tool decisions are acceptance
measures; declining loss, shorter thoughts and fewer raw errors are diagnostics.

### Final status: all ten budgeted attempts terminal and audited

- [x] Final requirement audit saved as completion-audit-r5.json. Six targeted
  analyzer/checkpoint/EMA/lineage/observation test modules passed20 tests in1.74s;
  git diff --check passed. Only monitor-self-pilot-every-15-minutes was paused
  through the app after reporting; its persisted PAUSED status was verified.
- [x] Final direct Docker inspection confirms all ten accepted attempts exited,
  all within their 7200-second total runtime caps. Accepted source manifests and
  every listed source file still match. See ten-attempt-final-container-state-audit-r5.json
  and ten-attempt-final-source-verification-r5.json under the evidence directory.
- [x] Slot10 exited0 at2026-10-18T15:18:54.863Z after5978.42seconds,
  six actual updates and1024 unique native episodes. All124 training tasks were
  covered; all native populations scored without errors. Runtime qualified,
  learning qualification rejected: heldout success58.06->54.84%, paired95% delta
  interval[-9.35,+1.29]points; reference-training retention also regressed.
- [x] Ten attempts total: eight completed runtime receipts, two failed attempts,
  124 actual optimizer updates and4335 retained episode records, including45
  records from09's incomplete final evaluation. Do not pool these populations.
  Luna treatment never started. No eleventh attempt or automatic full/LFM restart.
- [x] The authoritative consolidated comparison is the table in Outcomes &
  Retrospective below. Earlier Progress entries and tables are dated execution
  history and are superseded by this final status.

### Historical progress: R5slot09 terminal, slot10 accepted

- [x] Slot10 ACCEPTED2026-10-18T13:39:16Z (18:39:16PKT), run
  sdpo-qwen35-4b-agent10-pool124-thinking-20261018-r5, Docker
  53968d4e36522a3dff14ef542bfc213d13ade1f325c0d3cdb99b11b1c2371edd.
  All10original slots have now started;9terminal,10loading models at acceptance,
  no updates confirmed. Immutable capsule/sourceR5 above, exclusive shared
  deadline started once; never duplicate/reset/restart. Launch receipt
  experiment10-launch-r5.json includes actual Docker state/config/budget.
  Source hashes, empty directGPU/activeDocker/retained10states and protected
  dstackmain were checked before launch. Monitor native thinking and channels,
  full-context gates, actual updates and full native evaluation before reporting
  any learning. Final completion still requires10terminal/audited and report.
- [x]10native transport/channel witnesses passed at32K/4K and64K/8K: correct
  arguments, actual synthetic tool result consumed, finalcontent correct,
  reasoning present/separate and no leaked think markup. Durable receipt
  experiment10-native-channel-gates-r5.json. At13:44Z job stillrunning,0updates;
  this is not yet full-context actor-backward or native learning evidence.
- [x] Completion audit preparation: direct Docker inspection found all10exact
  accepted containers,9terminal and10stillrunning. Durations of01–09 are all
  below7200seconds; original source-manifest hashes and every listed file in
  all10capsules still match their accepted receipts. Durable evidence:
  ten-attempt-container-state-audit-r5.json and
  ten-attempt-source-verification-r5.json. These prove attempt identity and
  preservation, not completion or useful learning. Remaining completion items:
 10terminal state, native/optimizer/guard/signal reconciliation, final unified
  report and schedule pause. Goal remains active.
- [x]10full64K serving gate passed:57344prompt+8192generated tokens and exact
  batch invariance through32inflight16token probes; the neutral synthetic
  continuation hit its length cap and is NOTtraining data. Receipt
  experiment10-full-serving-gate-r5.json. Baseline reference-training10episodes
  closed with all10scores/noerrors:40%success,75.8333%partial,0.6toolmistakes;
  offboarding now succeeds in THINKING freshbase. This is0-update inference-mode
  behavior, not learned improvement over09nonthinking20%/47.1429%. Full310heldout
  fresh thinking baseline is collecting; no optimizer update yet confirmed.
  A read-only local observer (exec session84277) polls exact Docker/evidence,
  writes experiment10-live-observations-r5.jsonl and never submits GPUwork.
- [x] Read-only audit memory repair: thinking baseline generation log exceeded
 90MB with903calls before the population finished, versus134MB for09's entire
  baseline. Stream every call rather than retain all large token arrays; keep
  stale-policy rejection, malformed-record rejection, limit/finish counts and
  all native score denominators unchanged.17focused audit tests/Ruffpass.
  Pinned-image CPU replay of09terminal exited0 and produced BYTE-IDENTICAL audit
  JSON(SHA2564d9589fbc5429eb5c533a4afb53b89e99f3667061a7af5d968d2c6f5d13a8478),
  including714closed episodes/6updates. New analysis-only source
  /tmp/sdpo-pool124-analysis-20261018-r4 manifest
  e9fe3633e0dbf0185d25cbef2847166099df5a3d6b7ec8f712479c0009f02054.
  Receipt streaming-native-audit-equivalence-r4.json. Accepted GPU sources are
  unchanged. Snapshot summary also preserves the broad analyzer's paired_vs_base
  intervals separately from absolute metrics and missing-score bounds.
- [x]09independent exact-token lineage CPUaudit passed all192selected actions
  across6applied updates. Re-rendered native student messages/tools reproduce
  original student token IDs, action IDs and rollout logprobs exactly; teacher
  prefixes re-render with feedback appended only there, without truncation.
 100selected actions use a verified successful native same-task sibling/own
  reference; other92use factual feedback. No offline Luna used. These are
  provenance checks, not correct-gradient proof. Receipt
  experiment09-terminal-r5/selected-token-lineage.json, executed auditor copy
  executed-lineage-auditor-r1.py.txt SHA256
 4169c4a9b7d2896e3418486051a25f23ffa8c85bb3bb9ff7e0e9143a8471adbc;
  remote /tmp/sdpo-selected-lineage-analysis-20261018-r1. Native preservation
  identity comparison also confirms no new/worsened/resolved reference-training
  broken identities (same3identities/6replicas), separate from missing heldout.
- [x]10fresh THINKING baseline fully closed and independently CPUaudited:
 320native episodes(310heldout+10reference-training),allscored,noerrors,0updates.
  Heldout58.0645%success/83.34345%partial,0.16129toolmistakes;4length-limited calls,
 0context refusals. Reference-training40%/75.8333%,no incomplete calls. These
  are pre-training inference results, NOTdistillation gains over09nonthinking.
  Receipt experiment10-baseline-audit-r5.json (streaming analyzerR4,pinnedCPUimage).
  Full32K4Krollout/actor capacity and first applied update remain separate gates.
- [x]10pre-training preservation audit passed coverage(67state variants,
 0unknown checks):heldout110broken/920checks,45affected/140applicable episodes;
  reference-training12broken/32checks,4affected/10. This is not a clean safety
  improvement just because success/partial increased. Relative to09fresh
  nonthinking base, heldout has4NEWbroken identities while14resolve; reference
  has3NEWbroken identities/6replicas and none resolve. These are inference-mode
  differences, not learned harms. Native preservation is not a full independent
  harm taxonomy. Evidence experiment10-baseline-native-preservation-r5.json and
  fresh-base-thinking-preservation-comparison-r5.json. Training improvement must
  be compared against10's own thinking base, including its constraint behavior.
- [x]10first ACTUAL optimizer update completed:step1,batch32/32distinct tasks,
  loss0.4123053,finite gradientnorm0.83067,LR1e-5,optimizer moment1/EMA1;
  elapsed2514.34sec fromdriver start. Native64online episodes closed before it.
  Full32K/4K zero-update actor-backward gate passed(student/teacher32768tokens,
 4096target tokens,finite nonzero gradients,weights unchanged);peakallocated
 27,291,502,592bytes,actor12.91sec. First batch admits304complete transitions,
  excludes5reply-limited+7overlong whole transitions,selects32unique tasks/actions,
 12terminal actions and0observed-tool-error actions;maxselected context+target
 26567tokens,no training truncation. Second fresh rollout collection is active.
  Receipt experiment10-actor-capacity-r5.json. No post-training reward result yet.
- [x] Independent saved checkpoint1 CPUaudit exited0:actual moment1, applied
  LR1e-5,finite changed student/teacher adapters,RNG and saved-file hashes pass.
  Receipt experiment10-checkpoint1-r5.json. Separate manifest audit confirms
  all124train/62heldout names disjoint, exact ordered heldoutSHAed81cf1fe88f68bb8420bb46c5fdf3afde24660c7043812da680bb1c7bc9196f
  and frozen environmenta0c7378b179e2b267382b8125192876502d82f89;
  receipt experiment10-manifest-audit-r5.json. This does not imply a reward gain.
- [x] Independent EMA recurrence audit of09passed all496adapter tensors at
  every one of6updates,0max absolute error: saved teacher equals
  previous_teacher.lerp(current_student,.01), starting from actual initial adapter.
  Receipt experiment09-terminal-r5/ema-recurrence.json. New CPU-only helper
  /tmp/sdpo-ema-analysis-20261018-r1/audit_sdpo_ema_recurrence.py SHA256
 21dca8578b34d77e14dd2020a9a86593936fe18623962735820e375042f21194;
  test rejects a nonzero but stale/hard-copied teacher, Ruffpasses. This verifies
  teacher updating, not teacher correctness; repeat on10terminal saved states.
- [x]10token-lineage auditR1 failed closed on message re-render, while original
  student prefix/action IDs still matched rollout exactly. Preserved failed CPU
  container13638e485ad4bb074c36f66e98264e3526a979b17aa6bbcaf7bbb65b573af228.
  Diagnosis proved the pinned ChatCompletionRequest constructor renames
  reasoning_content to reasoning INPLACE after native token rendering. For all
 22affected step1histories, restoring ONLYthat string-field alias recovers
  original student tokens exactly. Teacher tokens stay IDENTICAL after the
  same restoration because the native added-feedback user turn already applies
  the template's historical-reasoning treatment. No run/teacher/student edits.
  Evidence prompt-rerender-diagnosis.json/reasoning-alias-diagnosis.json and
  diagnose-thinking-*.py in the evidence directory.
- [x] Strict read-only lineage auditorR2 validates all96selected actions over
 3actual thinking updates:96actions contain native reasoning,29554reasoning
  target tokens,54verified successful native own/sibling teacher contexts.
  Exact original student tokens/logprobs, native history, teacher feedback-only
  append and teacher tokens all match; no truncation. Aliases must be strings,
  conflicting aliases fail and any student/teacher mismatch still fails.
  CPU native replay exited0; alias/EMA tests and Ruffpassed. Receipt
  experiment10-token-lineage-through3-r5.json. New read-only helper
  /tmp/sdpo-selected-lineage-analysis-20261018-r2/audit_sdpo_selected_token_lineage.py
  SHA256a0d0d18d0c1de175371a7f3dd16709063b53a4399e62bb8cd54ace054f5d8e46.
  This proves reasoning is in training targets, not that it is correct or improves.
- [x]10reached4applied updates and full124task coverage:128selected actions,
 32distinct tasks/update matching exact hashroundrobin;4tasks visitedtwice,
 120once. Cumulative whole exclusions13reply-limited+10overlong,0truncation.
  Losses.4123053/.4711981/.3922149/.3157972; elapsed2514.34/2826.84/3274.29/3672.25sec.
  Receipt experiment10-applied-coverage-through4-r5.json counts only selections
  for logged applied updates, not an in-progress future population. Fresh-data
  losses are not matched validation; fixed post-update evaluation stillpending.
- [x]10all6actualupdates completed atelapsed4544.90sec; losses
 .4123053/.4711981/.3922149/.3157972/.3446340/.3968962,LRfloor1e-6.
  Pinned-image CPU saved-evidence audit exited0:actualmoments1..6,checkpoint6
  RNG/hash/finitechangedadapters,independent496tensor EMA recurrence(0error),
  exacttokenlineage all192selected actions/192withreasoning,53139reasoning target
  tokens,105verified successful own/sibling contexts. Source/runtimevalidity
  does not imply correct teacher preferences or task reward improvement.
  Receipts experiment10-saved-evidence-final-r5/.
- [x]10post-reference TRAININGdiagnostic closed10/10scored/noerrors:
  success40%->20%,partial75.8333%->66.9048%,toolmistakes.6->1.2.
  Airtable successretained;offboarding1->0(partial1->.85714),safety partial
 .66667->.41667,restructure.625->.57143,finance.5unchanged. Small5task
  slice,notgeneralization; nevertheless successful-action retention regressed.
  Full310heldout post-fit remainsrunning; do not stop merely to hide a bad arm,
  extendtraining,rerun or make a fullpopulation claim before its closure.
- [x] Inspected four actual trained failure actions with native response feedback
  and saved full-batch Adam scores (four-thinking-failure-signals.json): guessed
  missing Slack channels after empty Gmail results, nonexistent Payment worksheet,
  Jira project-not-found (correct replacement not established), and QuickBooks
  currency-formatted line_amount strings causing Decimal ConversionSyntax.
  All four teacher contexts used native feedback without successful sibling
  rescue. Teacher tokenwise argmax often keeps rejected argument values; it is
  not an autoregressive corrective call. Full-action fixed-prefix logprob moved
  upward in3of4actual Adam updates, downward in1; this is local conditional
  evidence,not proof of rollout causality or per-action gradient correctness.
- [x] Inspected four fixed reference-task before/after pairs (rep0,rawIDs/hashes
  retained in four-reference-task-before-after.json). Offboarding lost row4
  Status=Scheduled despite valid calls/no explicit tool errors. Safety still
  confuses display labels with spreadsheet IDs; rep0 loses all6positivechecks
  while the other repetition partially improves, so taskmeanpartial falls.
  Restructure repeats worksheet/display-label errors and loses the Platform
  announcement, while resolving a forbidden Revenue announcement; native
  partial denominator changes8to7,so compare assertion identities,not justcounts.
  Airtable retains complete success/no tool errors. This shows semantic goal
  completion and retention problems beyond transport validity. Do not attribute
  these outcomes causally to one inspected optimizer action.

- [x] Slot09 exited1 at2026-10-18T13:33:53Z (18:33:53PKT). The control
  completed6actual updates, but its3500second phase watchdog expired during
  final heldout collection. Luna NEVER STARTED:0Luna updates, no paired result.
  Do not retry09, extend/reset its deadline or call this evidence against Luna.
  Exact-image CPU terminal audit exited0:714closed native episodes,192selected
  actions across all124tasks, saved moments1..6, finite changed student/teacher
  adapters and checkpoint6 hashes/RNG verified. Final heldout has only45unique
  task/repetition episodes across10tasks,265missing and no closed outcomes;
  no final heldout headline is identified. Reference-training diagnostics stay
  at20%success/47.1429%partial, not validation. Durable files:
  experiment09-terminal-native-audit-r5.json and experiment09-terminal-r5/.
- [x] Additional09CPU native preservation audit exited0. Reference-training
  stays6broken checks/32applicable checks at0/6,4affected episodes/10,0unknown.
  Heldout baseline158broken/920; no finalheldout guard comparison is available.
  Four actually selected error-bearing actions were decoded and joined to
  native feedback and their actual full-batch Adam changes (see below). Teacher
  context did not consistently produce the needed argument repair. CPU-only
  snapshot summarizer now distinguishes reference-training from heldout and
  absent Luna/live10 from zero gains;5targeted tests and Ruff passed. Snapshot
  ten-attempt-progress-snapshot-r5.json counts10accepted,7runtime successes,
  one failed qualifier,one timeout and one still running; not10completed.
- [x] Decision before10: native32K/4K runtime/backward and6control updates worked;
  the paired half-budget was insufficient. Keep10's separate single-arm7200sec
  cap, exact6update/124task/batch32/32K4K recipe and own fresh THINKING baseline.
  Its budget is not split into3500second phases. This tests within-thinking
  acquisition/retention and transfer; a complete matched nonthinking post-result
  is unavailable from09. Thinking GPU capacity/channels remain unqualified until
  real gates pass. No source changes, deadline reset or smaller benchmark.
  Before launch, protected dstackmain shows no unfinished runs and direct GPU
  compute/retained agent10 containers are empty. Source/evidence guards must
  pass at launch. Slot10 is planned, not counted as started until accepted.

- [x] Slot09 ACCEPTED2026-10-18T12:35:31Z (17:35PKT), run
  sdpo-qwen35-4b-agent09-pool124-luna-paired-20261018-r5, Docker
  ee06564519e01aa93c964d04ad31ea8cf5ccee6f174d07a0607f8a00e7bdb5b9.
  Count9/10actual attempts started; no updates yet confirmed at launch. Monitor
  this exact owned run, never resubmit/rebuild its source. Control starts first;
  Luna starts only after control completes and capacity/native gates pass.
  Receipt experiment09-launch-r5.json and remoteevidence/submission.json.
  The prior separatefull50attempt exited1after38updates during39collection;
  its3992episode terminal audit and raw failure logs are preserved, lastsaved30.
  EngineDeadError trigger remains unknown, not provenOOM/kernel/scoring failure.
  Fresh short-run GPU readiness gates remain mandatory inside09. No fullresume.
- [x] 09control fresh baseline closed and independently CPU audited320episodes
  (10reference-training+310heldout), all scored/noerrors,0optimizer updates at
  baseline. Heldout success51.6129%,partial81.1947%, exactly the priorfull50fresh
  base values under the same64K/20turn greedy evaluation. Reference-training
  acquisition baseline20%success/47.1429%partial over5tasksx2; this is not heldout
  evidence. Native32K/4K and64K/8Ktool-call/result/final witnesses and64Kserving
  batch probes produced receipts; actualfull32K/4Kactorbackward remains a separate
  first-update gate. Receipt experiment09-control-baseline-audit-r5.json.
- [x] 09control actual32K/4Kactorcapacity gate PASSED:32768student and teacher
  tokens,4096scored targets,finite nonzero student gradients,student weights
  unchanged and0optimizer updates within this gate. Peak allocated CUDA
  27,291,502,592bytes (~25.42GiB),reserved30,352,080,896bytes;actor gate29.43sec,
  rollout capacity probe156.45sec. Receipt experiment09-control-capacity-r5.json.
  This is real selected-loss/backward capacity evidence, not task acquisition
  or reward improvement; no actual optimizer update was confirmed at this check.
- [x] 09control first REALupdate confirmed: batch32,32distinct tasks/actions,
 11terminal actions,2observed tool-error actions,4whole exclusions and0truncation;
  loss0.2246826,gradientnorm1.38147,LR1e-5,optimizer/EMA1. Independent exact-image
  CPUaudit confirms actualsaved moment1, appliedcosine LR, finite changed
  student/teacher tensors and metadata hashes. Receipt experiment09-control-
  checkpoint1-r5.json. The outside-capsule single-checkpoint auditor now supports
  strict6step bounded profiles without weakening full50checks;11targeted tests
  andRuffpass. Remote /tmp/audit_sdpo_pool124_checkpoint-20261018-r2.py,
  SHA256253d38c6e78d07f2e49eca08be35860e02ed80e7ae15020d491667f877f16a49.
  No runningGPUsource edit. This is execution evidence, not native reward gain.

- [x] Rebalanced from measured execution time, not reward selection. Both branches
  now fit6updates; slot09uses6fresh control+6fresh Luna updates (12combined),
  slot10uses6native student+teacher thinking updates with its own fresh baseline.
  Each original experiment retains7200seconds TOTAL;09phase3500seconds each,
  parent7100+60CPU+30kill grace. Earlier preparation consumed noGPUslot;09now
  consumes its authorized slot after actual acceptance,10remains unsubmitted.
- [x] Preserve ALL124training identities,32distinct tasks/update,192selected
  transitions per branch (all124covered in the first4updates),32K/4Ktraining,
  unchanged12Kepisode/8turns and full62x5heldout64K/8192reply/32768episode/20turns.
  Replace the redundant fixed32training-sentinel collections in bounded
  experiments with just5predeclared reference-covered training tasks x2 at0/6.
  This measures native acquisition/retention on the only five Luna-covered
  tasks; all arms share the same five and these are training diagnostics only.
  Evaluate fresh baseline and final6; the separate50attempt evaluated every10
  before its38update runtime failure. Per branch384online+620heldout+20targeted=1024episodes/
  10populations; slot09total2048. No baselines, adapters or optimizer steps pooled.
- [x] Match cosine1e-5to1e-6 over6updates, batch32, top20RKL/EMA.01,
  failure-frontier and native schema parser across09control/Luna and10thinking.
  Five admitted teacher-only offline references,119fallbacks; no external calls.
  Short-horizon absence of improvement cannot establish failure after longer
  training. Native outcome/retention/guard gates remain unchanged.
- [x] R5passes27targeted tests, Ruff/bash checks, exact-image124task/five-reference
  CPU replay, pinned loss/gradient proof and thinking-channel replay. Sources:
 09 /var/lib/posttrain/qualifications/sdpo-agent09-20261018-r5,manifest
  46ca890b457d5d80aff8f66555bcd4e855d14011a589a9573bea58a927ce2542;
 10 /var/lib/posttrain/qualifications/sdpo-agent10-20261018-r5,manifest
  f3203e3820c0293a3046d9aae45dc601ddd459417ec0021814876f0964697344.
  Receipts prepared-experiment09-r5.json/10-r5.json and broad124-*-cpu-preflight-r5.json.
  Driver files are byte-identical across both capsules; CPU proof checks exact
  declared5training-reference/62heldout populations as well as data admission.
- [x] Current50attempt terminal/audited; protected dstackmain client showed no
  unfinished runs, realGPUcompute and activeDockerworkloads empty, no retained
  09/10container acrosssuffixes or09run evidence. ExactR5hashes validated before09.
- [ ] Audit09 before10; inspect ALLretained10identities and realGPU before
  its submission. Launch ONLYR5
  with bash driver/run_sdpo_agent_ablation.sh from its capsule, persist accepted
  Docker ID and budget immediately. Audit09before deciding/recording10. Native
 32K/4K/64K transport,32slot and full32Kbackward gates execute inside each actual
  bounded attempt; no already-qualified GPU learning claim. The R5budget model
  forecasts~3180seconds plus full-context backward-gate overhead per branch;
  plus the small targeted diagnostic overhead; it is not a guarantee. Preserve
  timeout/failure without weakening checks. R4unsubmitted preparation is retained;
  its omission of covered-task post-update measurement would have made Luna
  acquisition uninterpretable despite a valid heldout transfer comparison.

### Historical R3 preparation and the resolved timing hold

- [x] Pre-launch budget review required for sealed R3 preparations; do not launch
  them unchanged. Measured current-run full heldout durations are700.75seconds
  atbase and678.93seconds at10, plus~170seconds per32task training diagnostic.
  Current-run startup/native gates took~452seconds, and native collection plus
  update takes~210–230seconds. A fresh10step branch with two full heldout and two
  training diagnostics is therefore forecast around68–74minutes before larger
  context/actor qualification overhead; two branches exceed the120minute cap.
  This timing evidence, not rewards or heldout task failures, requires a new
  prospective preparation: preserve all124tasks/32distinct actions/full62x5
  benchmark and token budgets, remove redundant fixed training diagnostics in
  bounded arms and choose a shorter matched fitting horizon that fits. Fewer
  updates are explicitly permitted by the original20update upper cap. Retain
  R3 as unsubmitted evidence, keep slot count8, and update exact configuration,
  tests, manifests, plan and automation before launching a replacement.

- [x] Prepared remaining09/10 R3 on all124training tasks with32distinct tasks per
  update; no GPU submission and original count remains8/10. R2eight-task sources
  are superseded and retained. See prepared-experiment09-r3.json and10-r3.json.
  Both use32Ktraining/4Kreply,2fresh replicas per task, batch32,10updates per branch,
  cosine1e-5to1e-6, existingfailure-frontier/top20RKL/EMA.01/schema-bound parser.
  Sampling is held fixed rather than combining the historical first-error change
  with Luna/thinking; its CPU result remains evidence, not a validated learning fix.
- [x] Slot09 has two independent fresh10update branches, control then Luna,
  within ONE20update/7200second budget (3400seconds/branch;7100GPU cap+60CPU
  preflight+30kill grace). Only5task-bound offline references are admitted;119tasks
  use native/own-sibling feedback fallback. Report reference-covered and full-pool
  outcomes separately; never call this full Luna demonstration coverage.
- [x] Slot10 is10updates of native student+teacher thinking, no Luna, matched to
  slot09control's horizon/data/objective with its OWN fresh thinking baseline.
  The unresettable parent7200second budget remains; old reserve phase is retired
  because those tasks now belong to training. No new external model calls.
- [x] Both broad profiles collect full heldout62 baseline and step10 evaluation:
  62x5,64K/8192reply/32768episode/20turns,32slots. Train still8turns/12288episode.
  Per branch:640online+128training diagnostic+620heldout=1388native episodes,
  14closed populations,320distinct-task selected actions covering all124tasks.
  Slot09total2776episodes; never pool its optimizers, baselines or updates.
- [x] Local21targeted tests/Ruff/bash pass. Exact pinned CPU image confirms all124
  identities and all5Luna references exercise current native action admission;
  max teacher+action20796tokens, original student token parity, no truncation.
  Loss/gradient instrumentation and native thinking/parser structural checks pass.
  Receipts broad124-native-cpu-preflight-r1.json, broad124-loss-cpu-preflight-r1.json,
  broad124-thinking-cpu-preflight-r1.json. These are NOT GPU capacity/learning proof.
- [ ] After the accepted full-pool50 run terminates and its results are audited,
  inspect retained states and actual GPU before09. Capsule09
  /var/lib/posttrain/qualifications/sdpo-agent09-20261018-r3, manifest
  a7e2e28a038000a0ed15bfd34f9d750a3adbb5b9df154109689ce6cfc6f94e15;
  capsule10 /var/lib/posttrain/qualifications/sdpo-agent10-20261018-r3, manifest
  33af2498b767e1055794f33f83053040875f6f60b8324beea7e37fd203edc99a.
  Exact launch from either capsule: bash driver/run_sdpo_agent_ablation.sh.
  Do not launch both blindly: audit09native outcomes/teacher directions first,
  record the next decision, then10 if still informative. GPU readiness gates
  native32K/4K and64Ktool continuity, actualfull32K/4Kbackward and32in-flight
  batches occur INSIDE each bounded attempt. Failures count, no weakened checks.
  Original10attempt goal remains open; full50extension decision is separate.

### Historical R2 preparations, superseded by full-pool R3 above

Everything in this subsection records the prior narrow-corpus preparation.
Do not launch R1 or R2, reuse its retired reserve, or infer these sources ran.
R3 above and the current ledger are authoritative.

New r2 configurations prospectively test first-error-frontier sampling for both
09control/Luna and10thinking. The separate250run is unchanged. Through20ofthat
run's collections, only2of35failedHRtrajectories' first actualtoolerrors were
selected; all353selectedtool-bearing actions had genuine native responses.
The amendment prioritizes each failedtrajectory's earliest admitted observed
toolerror, otherwise uses the existingfrontier; terminal/successcoverage remains.
It is not per-goal credit and does not treat an empty result as an error.

Pinned CPU replay passes unchanged selectedloss/gradient, thinking/tokenizer
bridge, own/sibling andLuna nativeadmission (67rows/16distinctselected,all8tasks,
0exclusions/truncation). On the SAME16episodefixture across20selectoroffsets,
firsterrorselections change2->20, with16distinct/twopertask maintained. Those
20replays add no nativeepisodes oroptimizerupdates and prove coverage only,
not native learning improvement.53targeted tests pass; latest stage-lineage
override test also passes. Core CPU source manifest:
e70e6e009791658fa9610def260d1f190d89d4833c3db8e227e7b0e6daf6dce4.

Current09capsule /var/lib/posttrain/qualifications/sdpo-agent09-20261018-r2,
run sdpo-qwen35-4b-agent09-luna-paired-20261018-r2,source
d6331de3fc73af0fbf03453b87f4857bbf77cc152b0c79db843070d2b45b27fd.
Current10capsule /var/lib/posttrain/qualifications/sdpo-agent10-20261018-r2,
run sdpo-qwen35-4b-agent10-thinking-20261018-r2,source
5661ced5bbf4d2723c9495d966b033e71aa3dc54f0b14f87829646c81a150449.
Neither is GPU submitted; slotslaunched remains8. r1preparations are retained
unused and superseded, not failed GPUattempts or extra budget. Do not launchr1.
Before each launch inspect all owned09/10container names acrosssuffixes, not
just the chosen r2name, and reconcile ledger to prevent an extra attempt.

Limits remain09fresh10control+10Luna,32K/4K,3100sperbranch/6400GPU/600CPU+30kill;
10thinking10updates,4300sfitting and600sCPUreservation within original7200s
shared fit/review/selection/validation deadline.10validationrun is
sdpo-qwen35-4b-agent10-reserve-20261018-r2; stage helper reads this explicitname.
Thinking matches amended09control. The control bundles larger capacity and the
sampling amendment versus08; do not attribute that bundled difference to one
factor. Only09controlvsLuna isolates teacherreference at matchedrecipe. Later
GPU-native outcomes/gradient behavior/retention remain mandatory; CPUcoverage
does not qualify the method. See first-error-CPU-* andprepared-experiment09/10-r2*.
Long250campaign has queuepriority; heldout/LFM remainON HOLD.

After that run is terminal/audited, noowned09attempt exists under anysuffix, and
the actualGPU is free, the current09launch command is:

    bash /var/lib/posttrain/qualifications/sdpo-agent09-20261018-r2/driver/run_sdpo_agent_ablation.sh

After09native/gradient/retention audit andfreeGPU, current10fitting command is:

    bash /var/lib/posttrain/qualifications/sdpo-agent10-20261018-r2/driver/run_sdpo_agent_ablation.sh

Only after10completes and a primaryis frozen, within10's originaldeadline:

    python3 -B /var/lib/posttrain/qualifications/sdpo-agent10-20261018-r2/driver/stage_sdpo_reserve_phase.py --fitting /var/lib/posttrain/qualifications/sdpo-agent10-20261018-r2 --primary <frozen-primary.json> --adapter <audited-default-adapter-dir> --destination /var/lib/posttrain/qualifications/sdpo-agent10-reserve-20261018-r2
    bash /var/lib/posttrain/qualifications/sdpo-agent10-reserve-20261018-r2/driver/run_sdpo_reserve_eval.sh

These are gated futurecommands, not submitted jobs. No new timebudget is allowed
for the validationphase and no unqualified resume/replacement is authorized.

Latest user authorization2026-10-18: separately requested a fresh250update
campaign with batch32 and highLRfirst10/cooldownto20/constantfloor thereafter.
See sdpo-batch32-250-workstation.md for exact source, runtime gates and queue.
It takes priority after08;09Luna/10thinking remain pending, preserving their
20update/twohour limits. Do not race the active host queue with another GPU job.
08last observed16updates;checkpoint15 devpartial54.7619%vs50%base is exploratory.

Eight of ten GPU experiment slots have launched; all eight are terminal and audited.
Each future experiment is capped at20 optimizer updates and7200seconds total.
The two runtime qualification attempts count toward the explicitly declared budget.

### Reviewed numeric comparison

The following final metrics come from the per-arm native audit snapshots, not
from provider success or loss values. Each before/after comparison uses that
arm's own fresh base. All measured arms use eight training and four development
tasks, two greedy episodes per task/checkpoint. This is an adaptive small-cohort
study, with no independent reserve outcome yet.

Historical table superseded by the single consolidated comparison in Outcomes & Retrospective.

After the parser is repaired,06–08 do not improve final development full-task
success above their own50% base.08's temporary partial gain comes from four
correct HubSpot writes, with the next call blocked by the16K context guard.
Base and final support responses instead hit the2K reply cap. Larger capacity
must therefore have a matched fresh-base control before any improvement is
attributed to Luna or thinking. Current training full-task success remains87.5%
in every measured post-update population; the partial gains are concentrated in
one unfinished HR task. These observations support further controlled tests,
not a claim of a generally successful SDPO configuration.

The separate user-authorized250step/batch32 campaign is excluded from this table
and from the bounded search's primary selection. Seven of the eight terminal
attempts have successful runtime receipts; the failed first qualification is
preserved, not described as successful training. Metrics after failed/unrun
attempts remain null. Paired task uncertainty is retained in each audit rather
than inferred from these rounded rates.

Machine-readable comparison and source digests:
[unified-reviewed-results.json](../research/evidence/sdpo-vllm-20261018/unified-reviewed-results.json).
Rebuild from the repository root with:

    PYTHONPATH=scripts/qualification .venv/bin/python scripts/qualification/summarize_sdpo_search.py docs/research/evidence/sdpo-vllm-20261018 --output docs/research/evidence/sdpo-vllm-20261018/unified-reviewed-results.json

The snapshot preserves per-arm checkpoint curves, separate paired09branches
(never averaged or pooled), parser regimes, exact source hashes, missingness,
and the optional independent zero-update reserve result. Three CPUtests verify
that a failed qualification does not receive invented post scores, incomplete
coverage cannot masquerade as a full-population rate, and Luna/control rates
remain separate. The current snapshot measures through08's actual completion;
preparations are not additional measured experiments.

2026-10-18 checkpoint15 CPU review:08 development partial50->54.7619% comes
from support.zendesk_hubspot_org_sync,0/21->4/21positive target checks per
replica. The model actually creates four required HubSpot companies after reading
the same configuration/inventories. All15initiallyheld support checks remainheld
in bothreplicas; fulltask stillfails and the assistant terminates before remaining
updates. HRtraining2/7->4/7 completesStatusProcessed andITemail targets butstill
hits8turns. Four original native episodes reviewed (base/15firstreplica forHR
andSupport), matchingstatehashes/assertionidentities, nonewfalse finalchecks;
separate48episode base/15guard audit has0unknownchecks andnoaddedviolations.
This is specific partial task progress, not fullsuccess or heldout generalization.
See experiment08-four-episode-review.json andcheckpoint15-native-guards.json.

Future09/10 CPU proof refreshed against current campaign-compatible source:
manifest a98335fc55e08825dc1e8d51feb41c301a1bfae253046d5a044776edb7352f12,
bounded90scontainer exited0. Same selected-loss gradients, realtokenizer reasoning/
tool bridge and Luna native admission pass:67rows,16distinctselected/all8tasks,
49reference/18fallback,0exclusions/truncation. Actual32K/4KGPUgate remains pending;
do not reuse historical source receipts after code changes. Longcampaign retains
queuepriority; this preparation allocates no GPU and submits no new ablation.

Historical table superseded by the single consolidated comparison in Outcomes & Retrospective.

**Monitor08 first:** sdpo-qwen35-4b-agent08-cosine20-20261018-r1,
Docker1c683eafca10e0f9ac01987e714e944183b5394cce57bf1371ccb2413411f03f,
capsule /var/lib/posttrain/qualifications/sdpo-agent08-20261018-r1,
sourceb736f1c282194594bd3f4cd84b59f859a4e1ab1a7600228165ffed959b05f188.
Accepted2026-10-18T00:54:45.226993782Z. Never duplicate/restart/mutate this source.
Expected440episodes/20updates; live updates.json confirms11 optimizer/EMA steps
at2026-10-18 01:19Z, loss.32543, elapsed1381.55s. This is not final outcome
evidence. The actual update file is updates.json, not updates.jsonl. All07
native/guard/policy/checkpoint audits passed and real GPU was idle before08
submission. Sealed08 source predates future09/10 edits: never rebuild from the
evolving working tree.01-07 are terminal, preserve rather than resume them.

Final08supersedes the historical live notes above: exited0 at01:35:29.663Z,
receipt2430.07s;20actual saved optimizer moments/EMA counters and finite changed
student/teacher files audited. Reconciled440unique episodes/30populations, all
scores/noerrors.1851actual generated-call adapter hashes match21policy receipts;
two additional zero-action context refusals at15development have no generation
hash and are explicitly separated.21short probes pass(max.1376076), exactbatch
invariance reaches32.11incomplete call records including two contextrefusals;
one wholetraining transition exclusion, no truncation. All7original solved train
tasks retained. Train guards0throughout; dev2/0/0/2/2at0/5/10/15/20,unknown0,
no added broken identities versus base. Lossfirst.35919,last.33917 is not a
monotone reward improvement. Trainpartial final94.6429% vs91.0714%, paired8task
deltaCI[0,10.7143]points. Developmentpartial peak54.7619%at15vs50%, paired4task
deltaCI[0,14.2857]points, but final50%; success remains50%development/87.5%train.
Evidence experiment08-audit/native-guards/checkpoint-files/policy-provenance,
updates/receipt/outcome-curve/batch-invariance and native behavior/budget reviews.

Support budget audit corrects causal interpretation of the native agent_completed
labels: base and final support responses each hit2048generated tokens with
finish_reason=length, not a completed workflow. At15 the same11707token prefix
produces1951complete tokens includingfour company creations, followed by a
zero-generation context refusal(14357prefix+2048requested>16384). Both replicas
agree. Four previously unsatisfied checks become true with all15preservation
checks held, but neither token efficiency nor the stop label proves fulltask
ability. Keep matched32K/4K control before attributing any future Luna gain.

The user-authorized separate longrun qualifier is accepted and running:
sdpo-qwen35-4b-b32-qualify-20261018-r1, Docker
6081ccfec7f606c5237b4edc14397c3a4a1d9e280fe3f8d8fb4d898606c04ea0,
accepted01:35:51Z by owned host queuePID2682629. Monitor rather than resubmit.
No full250update training receipt yet.09/10 remain pending behind that campaign.

Historical r1preparations (superseded by the currentr2section above; do notlaunch):
Prepared09and10 immutable source capsules exist on the workstation, with no
GPU submissions and no10deadline started.09runname
sdpo-qwen35-4b-agent09-luna-paired-20261018-r1, capsule
/var/lib/posttrain/qualifications/sdpo-agent09-20261018-r1, source manifest
ebb6bf208fc40f9c68ed8dd248ce10e7d4b8af27beb587db14fece3d7b007b0e.
10runname sdpo-qwen35-4b-agent10-thinking-20261018-r1, capsule
/var/lib/posttrain/qualifications/sdpo-agent10-20261018-r1, source manifest
d26367b1ce9d9d8b4528af7279251d346572439bda54c16b4f066a0cd575519a.
Both use the CPU-tested campaign-compatible runtime snapshot, same frozen
cohorts and exploratory cosine1e-5->1e-6 over10updates per branch.09pairs
fresh10control+10Luna,3100s perbranch/6400sGPU/600sCPU/30sforcedkill allowance.
10fits10thinkingupdates with4300sGPUcap and600sCPUreservation; afterfitting,
hostreview/selection/validation share its original nonresettable7200sdeadline.
Neither is claimed optimal;08's development gain did not persist.

stage_sdpo_reserve_phase.py prepares10's optional zero-update validation after
actual fitting completion. It verifiesparent source hashes, Dockerexit0, all10
recorded moment/EMAupdates, originalbudgetidentity/deadline, frozenprimaryadapter
hash andreservetaskhash; copies candidate and originalbudget into a new phaseB
capsule, then rechecks remainingtime. Existingcapsules are never overwritten.
Four new CPUtests reject alteredsource/tamperedadapter/incompleteupdates and
prove byte-identical budget preservation. Run it onhostwith python3 -B to avoid
creating bytecode in accepted source. No reserveprimary is selected yet.
Afteractual10fittingaudit and development-first freeze, exact invocation is:

    python3 -B <10capsule>/driver/stage_sdpo_reserve_phase.py --fitting <10capsule> --primary <frozen-primary.json> --adapter <audited-selected-default-adapter-dir> --destination /var/lib/posttrain/qualifications/sdpo-agent10-reserve-20261018-r1
    bash /var/lib/posttrain/qualifications/sdpo-agent10-reserve-20261018-r1/driver/run_sdpo_reserve_eval.sh

The first command onlystages; the second submits GPU after retainedstate/source/
deadline checks and parentexit0. If noeligibleprimary or insufficienttime, report
that finalvalidation could notcomplete; never create a fresh time budget. Pending
GPUcommands for09and10 respectively are bash<their-capsule>/driver/
run_sdpo_agent_ablation.sh, only afterthelongcampaign/audit andfreeGPU. Record
acceptedIDs/receipts immediately and inspectownedstates before every launch.

- [x]06 completed20updates/440unique native episodes/30populations, scoring errors0,
  all scores present, policy versions0–20. All1904generation adapter digests match
  active receipt;21probe actor/vLLM parity checks pass with maxdelta.13761nats,
  exact batch invariance reaches32. Actual optimizer states1–20 and finite changed
  student/EMA checkpoint files verified.11incomplete generation calls,9whole
  transition exclusions, no truncation. Receipt2544.59s/container2558.94s.
  Training partial91.0714->92.8571% at10/15/20, paired8task deltaCI[0,5.3571]points;
  full success87.5% unchanged. Dev goals success/partial50%/50% unchanged from
  repaired-parser base. Train guards broken0allcheckpoints; dev2at0/10,0at5/15/20,
  unknowninitial0; no added broken identities. This is small training efficiency
  evidence, not demonstrated development goal transfer. See experiment06-*.
- [x]Read06 final evidence and accepted07 after exact retained name/evidence absent,
  GPU idle and sealed source verification. Native guards and solved-task retention
  remain mandatory. Earlier micro-margin interpretation has the documented mode
  limitation;07 is an exploratory LR control, not a claimed numeric repair.

- [x]05 reconciled440unique episodes/30populations,20actual saved optimizer moment
  states, finite nonzero student/EMA adapters at evaluation checkpoints and policy
  versions0–20. Native scoring errors0. Eleven incomplete generation calls;
  training excludes1reply-limited and2whole overlong teacher-context transitions,
  never truncates. Dev success/partial25->50% at15/20, paired task deltaCI[0,75]
  points across4tasks. All7original solved training tasks retained. Greedy train
  success87.5%,partial91.0714% unchanged; train preservation violations0throughout,
  dev2->0 at10/15/20, unknowninitial0. Exact receipts/audits experiment05-*.
  Runtime2449.80s in receipt,2464.40s container wall. Provisional within05 best15
  by declared earlier-step tie-break; no primary frozen and reserve untouched.
- [x]06 CPU qualification/source receipts verified; GPU and exact retained names
  checked idle/absent before one accepted launch. See experiment06-launch.json.
- [x]06 fresh-base control: train87.5%success91.0714%partial, dev50%/50% with
  complete24scores/no scoring errors, before optimization. Actual first Salesforce
  prefix IDs AND generated IDs are byte-identical to05freshbase; only parsed
  integer500001 becomes schema-allowed string500001. Both06native Case episodes
  pass with0toolmistakes, versus05base failures. Parser repair alone reproduces
  the earlier best task-success gain. See experiment06-transport-baseline.json.
  Report subsequent06learning relative to its50%base, not old25%broken-parser base.
- [x]06 checkpoint5 native scores complete/no errors: train87.5%success91.0714%
  partial, dev50%/50%, unchanged from repaired base. Six actual optimizer/EMA/moment
  updates observed; no beyond-transport reward gain established yet.
- [x]06 checkpoint10 native HR inspection supersedes the no-gain-at5 observation:
  both HRoffboarding replicas now satisfy the required row2StatusProcessed check,
  positive target checks2/7->3/7 with same7scored denominator. Training partial
 91.0714->92.8571%, success87.5% unchanged; all7original solved tasks retained.
  This is a narrow training-task gain above repaired base, not full HR success or
  development/generalization proof. Evidence experiment06-hr10-native-audit.json;
  full guard/final-run audit pending.
  Ordered native calls explain it: base spends3calls failing sheet lookups,
  including repeating the compound title in spreadsheet_id. Checkpoint10 drops
  that redundant third attempt and uses ss_offboard directly after2errors. The
  saved turn lets it execute row2StatusProcessed as its eighth call. This is
  observed recovery/efficiency, not evidence of a new broad reasoning skill.

- [x] Read papers and pinned implementation; frozen product baseline unchanged.
- [x] Qualified optimized native16K/2048reply/32slot runtime, original token
  prefixes, typed-null handling, actor/rollout parity, optimizer/EMA and refreshes.
- [x] Reconciled01–04 separately; full IDs/digests/receipts in experiment-ledger.json
  and experimentNN-* evidence. No attempts pooled or missing scores zero-filled.
- [x]04 exited0 in2446.21seconds. Greedy train87.5%success91.0714%partial and
  dev25%/25% unchanged at5/10/15/20. Finaldev tool mistakes2.25->2.375.
  Dev preservation violations2->0 persist, no train violations, unknowninitial0.
  Ten incomplete dev calls never enter training; training exclusions/truncation0.
- [x]05 sealed native replay selects the exact failed Slack summary (generation29)
  instead of working search(generation5):16episodes65eligible16unique7terminal,
  no exclusions/truncation.12CPU tests passed for this sealed source.
- [x] Future local online-group2/4/8 support and reserve-evaluation draft prepared
  separately;13CPU tests pass. They are not changes to05 or qualified GPU arms.
- [ ] Verify05 actual selected actions, teacher/Adam correctness, native checkpoints
  and full outcome/guard/coverage audit before choosing06.
- [x] Live check after update7: optimizer moment/EMA counts agree through7.
  Greedy checkpoint5 has complete scores: train16episodes success87.5%,
  partial91.0714%; development8episodes success25%, partial25%, all unchanged
  from matched step0. No native scoring errors in these checkpoint populations.
  Sampling replay qualification therefore does not yet demonstrate reward gain.
- [x] Detailed checkpoint10 development audit supersedes an overly broad live
  status inference: partial25->37.5%, success25% unchanged, all8scores present.
  Both HR handbook replicas now pass Carol/Hugo HR Director escalation checks;
  Bob/Femi checks still fail. Leave-exempt Elena is no longer emailed, so the
  partial denominator changes5->4; this gain also contains2newly passed positive
  checks, not only a denominator change. Paired task deltaCI[0,37.5]points across
  only4development tasks. Training success/partial unchanged. Native artifacts
  and exact checks in experiment05-checkpoint10-hr-audit.json. The completed
  development0/10 initial-state guard audit verifies38preservation checks at each
  checkpoint, broken2->0, unknowninitial0; newly satisfied positive checks4->8
  across8episodes. See experiment05-dev10-native-guards.json. Full final-run audit
  remains pending; do not claim full success, statistical certainty or generalization.
- [x]05 first consumed batch has16unique actions,65eligible, no exclusions/truncation,
  seven terminal contributions. HR selected last executed farewell plus a distinct
  earlier erroneous lookup; teacher/Adam both lower the lookup's sampled-action
  logprob on average. Latest executed frontier can still be a valid partial step
  in an unfinished workflow, so it is a heuristic, not causal credit. Evidence
  experiment05-first-consumed-batch.json; inspect checkpoint5 before reward claims.
- [ ] Execute/adapt/audit06–10; freeze one eligible primary before any new reserve.
- [x] Checkpoint15 native inspection: both Salesforce development replicas now
  pass Case status In Progress after2invalid numeric-ID calls then a quoted-string
  XML ID. Base repeats invalid numeric IDs7times and fails. Dev success25->50%,
  partial25->50%, tool mistakes2.25->1.0; HR partial gain at10 does not persist.
  Training16scores unchanged, all7original solved tasks retained. See
  experiment05-checkpoint15-salesforce-audit.json. This is learned recovery around
  the reproduced parser bug; compare repaired-parser fresh base06 before claiming
  broader task-learning ability. Full guard audit/final05 outcome still pending.
- [ ] Produce unified experiment table, uncertainty and honest final verdict;
  heldout62/full256/LFM remain on hold.

Superseded04r1 broader16task capsule was CPU-only preparation, never GPU submitted:
do not launch /var/lib/posttrain/qualifications/sdpo-agent04-20261018-r1.
02r1 was also an unsubmitted CPU-preflight source. Both remain preserved.
Earlier progress snapshots are archived in progress-history-through-experiment04.json.
Results and distinct methodological discoveries below remain current.

## Surprises & Discoveries

07 final: exit0, started00:08:23.886Z, finished00:52:22.117Z, receipt2623.87s.
20actual saved optimizer moment steps, finite changed student/EMA files,
440unique native episodes/30populations all scored,0native scoringerrors.
11incomplete generation calls; training admission excluded1reply-limited action,
no truncated sequences.1855native generation adapter digests match21policy
versions; actual32batch exact invariance and21probe parities pass,max.1376076.
All7greedysolved training tasks retained. Train partial91.0714% at0/5,
92.8571% at10,98.2143% at15,94.6429% at20; success87.5% throughout. HR required
checks2/7->6/7 at15 then4/7 at20. Actual15 first action searches Drive for the
departure spreadsheet, then uses observed ss_offboard, avoiding baseline's
3invalid sheet lookups; these saved turns enable notifications/row updates.
One required Scheduled row remains missing at15. All6HR preservation checks
hold in inspected replicas. This is more than a lower token loss, but remains
training-task learning. Development success/partial stay50/50. Native train
guard violations0allcheckpoints;dev2at0/5/15/20,1at10,unknown0, no added broken
guard identities versus baseline. Final loss.21024 vsfirst.34480 does not imply
monotone task rewards; late training partial regression motivates08cooldown.
Evidence experiment07-audit/native-guards/checkpoint-files/policy-provenance,
receipt/updates/outcome-curve and HR15native audit. Old preferred-token mixed-mode
caveat still applies07. No reserve/full-training/heldout/LFM result inferred.

Luna interface audit found one additional incompatibility AFTER state replay:
finance.po_email_logging reference uses google_sheets_lookup_row, absent from
its current task tools. Do not translate, omit the required domain call, or
teach unavailable actions. Per-transition reference tool availability now gates
teacher admission; finance uses the original own/sibling/native feedback fallback,
as does Mailchimp for missing source. Thus7state-qualified sources but6usable
references; keep all8tasks. Evidence luna-reference-tool-coverage.json and a
regression test rejecting unavailable tools. Re-run exact final09CPU replay to
record actual reference/fallback counts before sealing; earlier63/4CPU counts
predate this compatibility gate and are historical, not final coverage.

Future09/10 code now implements32K/4K CLI limits, explicit paired fresh10+10
subprocess branches with shared total20update/time caps, current SDPO full-context
capacity/backward helper, and actual native tool/random-code continuation gate.
Capacity uses near-limit generation,32in-flight16token batch probes and full32K
scoring/backward while vLLM is suspended; no optimizer step in capacity and no
claim that32requests each generated4096tokens. Native tool probe's random result
must appear in final content and never be confused with reasoning. These are
new project prototype paths, NOT GPU qualified yet. Future thinking mode uses
native tokenizer enable_thinking=True for BOTH student and teacher and the pinned
Qwen3 reasoning adapter extract_reasoning API (extract_reasoning_content does not
exist here), then the schema-bound tool parser. Preserve full raw thinking-plus-
action target IDs; unclosed reasoning is incomplete and excluded. Verifiers'
pinned dialect/types retain reasoning_content in assistant history. CPU synthetic
thinking/tokenizer/channel replay passes, including old nonthinking prefix parity
and no marker leakage; actual thinking generations/learning remain unverified.
Record observed reasoning lengths, not merely the enabled flag. The combined
slot10 deadline controller and explicit reserve mode/token-profile checks are now
implemented with CPU tests, including real subprocess deadline termination.
Fitting, host audit, selection and reserve validation share one non-resettable
CLOCK_BOOTTIME deadline with preflight time reserved; no fresh two-hour phase.
These changes still require actual GPU qualification and phase orchestration.
Final future-source CPU proof is /tmp/sdpo-final-future-cpu-20261018-r2,
with evidence under final-future-CPU-* in this plan's evidence directory.
Actual compatible Luna admission is49 reference rows and18 fallback rows out
of67 admitted actions,16 selected across all8 tasks, zero exclusions/truncation;
the historical63/4 counts predate the tool-availability gate.

### Latest authorization: larger token limits and a thinking arm

User explicitly chose tokens only:32K context/4K replies for Luna and thinking,
keeping2hours and at most20optimizer updates per experiment. Keep the episode
output ceiling12288 and8turns; no authority to raise either.07 and08 remain the
original16K/2K controls and their immutable sources must not be changed.

To avoid attributing extra rollout capacity to Luna context, implement09 as a
paired experiment: fresh matched32K/4K no-Luna control10updates and fresh Luna
context10updates,20total with separate optimizer/EMA/adapters/native populations.
Same rate recipe, sampler, seed and task cohort; no pooled learning curves.
Choose the rate recipe after07/08 native audits, before09 fitting. Larger native
profile must pass real capacity/backward/tool-continuation gates inside09's
total execution budget. Preliminary CPU preparations are not GPU qualification.

Use10 for the newly requested thinking arm, targeting10updates matched to09's
no-Luna control, leaving time inside its total2hour cap for the predeclared
independent comparison. Do not create an11th fitting slot or grant a free,
unbounded evaluation exemption. Implement a phase-level budget before launch;
freeze primary adapter/mode/profile from development before reserve outcomes.
Final comparison must load a fresh base with the primary's matching thinking
mode and token profile. A timeout/failure is retained and reported as incomplete
evidence, never silently replaced. Student AND teacher thinking mode, exact
native reasoning/tool parsing, complete thinking-plus-action token targets and
reply-limit exclusions require separate qualification. Full training/heldout62/
LFM remain on hold. This is a local algorithm extension: SDPO paper Table12
reported thinking=False, so do not call10 a paper recipe reproduction.

08 staged immutable capsule:
/var/lib/posttrain/qualifications/sdpo-agent08-20261018-r1,
run sdpo-qwen35-4b-agent08-cosine20-20261018-r1,
source-manifest SHA256b736f1c282194594bd3f4cd84b59f859a4e1ab1a7600228165ffed959b05f188.
Prelaunch qualification snapshot (08 is now accepted; see Progress): exact
new-source pinned CPU loss/gradient/same-mode diagnostic,
parser bridge and16episode replay gates pass:67admitted,16distinct selected,
no exclusions/truncation. Proofs experiment08-*.json/source-sha256.txt. Future
local source has advanced for09; launch08 from its sealed capsule, never rebuild
from the evolving working tree. Inspect07 terminal state/audit before launch.

09 engineering gates caught two issues before any GPU submission. First,
WireTaskData drops initial_state/assertions even from trace.task; recover them
from persisted native episodes joined by exact unique episode IDs plus task/user
instruction equality, then enforce source state/assertion/user hashes. Second,
raw SDK captures exceeded16K for HR(36021–39359), mobile(37622–38511), and
finance(18645–21584). Preserve these failed CPU source snapshots and logs under
/tmp/sdpo09-luna-cpu-*-v2 and sdpo09-luna-admission-diagnosis*; these were not
GPU submissions or optimizer updates.

Explicit reference projection keeps every retained domain-tool execution with
complete arguments/results, removes tool-discovery schemas already supplied in
the live tool prompt, removes duplicate execute_tool routing and decodes nested
JSON. No domain result fields or training sequences are cut. Raw sources remain
unchanged. The resulting context-format/source bundle is the treatment; do not
attribute differences solely to Luna's model quality. CPU-v3 passes16episodes,
67eligible actions,16distinct selected,all8tasks,63Luna-reference actions and
4explicit Mailchimp fallbacks, longest teacher+target11654tokens, zero exclusions
or truncation, original student prefix/action IDs unchanged. Current-rubric
review passes all7candidate final states,7preservation checks total,zero broken
and14newly satisfied checks; source system/clock hints are omitted from teacher
observations. This is state-check replay, not native tool re-execution or proof
that every demonstration call is correct.29local CPUtests/Ruff/bash/diff pass.
32K/4K runtime code is parameterized locally but not GPU qualified. Need fresh
exact-source CPU replay after final paired driver/source is sealed.

### User-directed remaining fitting arms: decay and offline Luna context

The latest user requests supersede earlier candidate JSD/broader-data ideas for
slots08/09. Keep total10 experiments, <=20 updates and <=7200s total execution
per arm;07 remains immutable and slot10 remains the independent comparison.

08 is a matched schedule test against07: same original8/4 tasks, fresh base,
schema-bound-v1 parser, top20 reverseKL/EMA.01, batch16 and selection/context.
Use20 applied update rates following cosine decay from1e-5 at update1 to1e-6
at update20, no warmup. Its average rate is5.5e-6. This changes the cumulative
step size as well as its time distribution versus constant1e-5; do not claim
it isolates ordering alone. Log the rate actually applied to every Adam update.
Pure endpoint/monotonicity/default-compatibility tests pass. New same-mode
preferred-token/full importance diagnostics are measurement-only changes;
qualify their exact source in the pinned CPU image before sealing08.

Evidence: SDPO Table12 (https://arxiv.org/html/2601.20802v2#SxE2) uses constant
1e-6 for rich feedback, so decay is an explicit local ablation. The WSD paper
(https://arxiv.org/abs/2410.05192) supports investigating high-rate progress
followed by cooldown, primarily in pretraining. It does not establish our
20-update cosine recipe as best. No WSD-reproduction claim:08 has no stable
phase or warmup. Judge native trajectories/retention, not loss alone.

09 tests offline Luna reference context, compared with a declared matched
no-Luna arm using the rate recipe selected from07/08 development/diagnostics
before09 begins. Keep original student prompts/action targets and SDPO loss;
only teacher-side reference context changes. Label this an external-demonstration
context variant, not pure own-success SDPO or corrected-call SFT. No new Luna,
judge or external model calls. Verify task prompt, initial state, assertions,
recorded outcome and preservation evidence before admitting each saved trace.
Preserve intermediate errors; a successful endpoint does not make every call
correct. Never silently truncate a reference or the training transition to fit.
Record exact per-task reference coverage and explicit fallback for missing or
incompatible sources; do not remove tasks from the cohort to hide missingness.

Initial CPU discovery found local recorded full-success Luna episodes for7/8
training tasks in luna-development-review-coverage.json. All7 source episode
hashes, initial states and assertions match recorded/current native populations.
Five task prompts match exactly; finance/HR historical system prompts additionally
contain simulated-clock hints, which require explicit compatibility review.
Mailchimp was absent from this index. SDK episodes have no native message nodes;
executed-call/result captures and SDK events exist, with native call alignment
and failed/concurrent capture coverage explicitly unqualified. Do not describe
these as complete token-aligned gold trajectories. Native guard replay and full
teacher-context length checks remain pending before09 can be qualified.

Local schedule/config integration tests:24 passed, Ruff/bash/diff checks pass.
Historical preparation snapshot: neither08 nor09 was submitted then. Current
Progress supersedes this snapshot:07completed and08accepted;09not submitted.

Future diagnostic instrumentation is CPU-qualified outside accepted07: compute
teacher logits once, score original and teacher-preferred token IDs together in
eval mode before/after Adam, then restore student train mode before the unchanged
loss. This uses the same number of model forwards as before. Also record every
sampled-token actor/rollout difference, importance weight and student rank rather
than only the recoverable95.75%subset. Pinned-image synthetic CPU tests verify
loss/gradient equivalence and expose a deliberate train/eval-logit difference,
so the new preferred-token baseline avoids the old mixed-mode ambiguity.
Proof/source hashes in diagnostic-cpu-equivalence.json. No actual model/GPU
updates at that historical point.08 is now sealed/accepted;07 remained unchanged.
Qualify exact new-source native replay before any future GPU use.

07 observation-only SSH watcher exited255; fresh exact Docker inspection showed
the same container healthy with2actual optimizer/moment/EMA updates. Monitoring
reconnected without resubmitting or restarting the model job. Treat observer
failure separately from authoritative execution state.

Selected immutable TRL config defaults dont_reprompt_on_self_success=True;
actual trainer excludes current rollout as its own demonstration but can still
reprompt a successful sample using a sibling/environment feedback. It does not
simply mask all successful samples. Our own-success-first reference is an
explicit custom context choice, not a full paper-trainer reproduction. Actual
runtime config/trainer hashes in selected-trl-reference-policy-audit.json; the
sibling repository HEAD differs and was not used as executable authority. This
is an available later reference-policy hypothesis, not a new launched arm.

06 native loss-input audit covers28,694/29,969selected positions (95.75%) where
generated token equals logged teacher-top1. Actual HFtraining/vLLM sampled logp
difference meanabs.00473nats,maxabs.45784nats; no importance ratio reaches cap2
in that subset. This is broader-context evidence than the short .15nat probes,
but not full-distribution/argument-token parity. The remaining positions are not
recoverable from old diagnostics. See experiment06-native-importance-subset.json;
do not claim all native tokens meet the probe tolerance. No runtime gate changed.

05 selected training exposure is small and success-heavy:320distinct native action
identities but262distinct student-prefix/action token sequences from229source
episodes across8tasks;31,905target tokens. Student/teacher context+target forward
exposures1,079,844/1,476,078tokens are compute counts, not distinct training targets.
Only44/320selected actions belong to failed tasks and19have observed tool errors;
284use verified successful context and140are terminal actions. Exact counts in
experiment05-training-exposure.json. This supports a later harder/broader data
hypothesis; it is not a model-size token-saturation ablation or an SDPO scaling law.

Final-slot CPU preparation, no reserve selected/read: reserve driver now records
zero trainable parameters and requires frozen explicit transport mode, identical
for base/candidate. `run_sdpo_reserve_eval.sh` is guarded for slot10/zero updates,
sealed primary adapter hash, duplicate/GPU checks, normal pinned offline image and
7100s+30s hard timeout. `analyze_sdpo_agent.py` now audits the separate96episode
zero-update protocol, actual parity values, policy/adapter digests, missingness
and preselection. Six synthetic CPU audit tests cover reused IDs, missing scores,
wrong adapter/transport and false parity claims; existing05audit is unchanged.
Only CPU/Ruff/shell checks passed: this path is NOT GPU integration qualified,
not staged/submitted, and no primary or reserve outcome has been inspected.

05 four-consumed-failure audit (experiment05-four-consumed-failure-audit.json)
shows the sampling fix is not universal causal credit: finance step7 selects a
valid final spreadsheet read and failed Slack step12 selects a valid final channel
listing in an unfinished episode. Trello step6 does select the actual wrong
`March 5, 2026` date call. Its successful sibling teacher now strongly disfavors
sampled `March` (teacher-minus-student logp -7.9656), prefers `2` for ISO date,
already student rank1. The saved preferred-minus-original delta difference is
negligible (~-3.7e-9), but its preferred baseline uses training mode and after uses
eval mode. It is not independently verified as a same-mode Adam margin, nor an
optimizer-bug proof. Original sampled-token before/after deltas are eval/eval.
Greedy Trello already succeeds;
this is a rare stochastic error. See experiment05-trello-token-audit.json.
These residuals motivate testing a larger-step/different-divergence paper recipe
after parser control, rather than claiming top20 omitted the correct token here.
07 is now conditionally CPU-prepared as recorded in Decision Log; read final06
before acceptance. No change to accepted06 or sealed07 source for diagnostics.

Pinned-parser CPU reproduction now confirms a transport bug, separately from
sampling/teacher quality: native Qwen XML row `1` is coerced into Python integer1
although the advertised row schema permits string/object/array/null, not integer.
The container/array coercion branch accepts any json.loads result without checking
its type. Actual installed utils SHA256 is
fd5153158235fbb5844a85f3081cab8757fdb75684c4416fd51da6d5bdff7d34.
`schema-parser-preflight.json` pins the saved actual call, reproduces native versus
guarded arguments, checks six typed controls, and verifies identical bridge
prefix/action IDs and budgets. Native/installed source remains untouched.

Preparation record before accepted06: same05 algorithm/profile with an explicit
project-local schema-bound-v1 parser subclass. See experiment06-prepared.json.
It only rejects coercion into undeclared types; it never invents IDs or changes
model tokens/schema. The generic eventual fix belongs in the vLLM fork, but no
fork commit/pin update is authorized here. This isolated research adapter changes
no frozen public product meaning. Guarded fresh-base results must be reported as
transport evidence, separately from within-arm SDPO checkpoint improvements.
05 continued unchanged;06 was launched only after05 terminal audit/GPU idle check.

The accepted06 source is sealed at
`/var/lib/posttrain/qualifications/sdpo-agent06-20261018-r1`, manifest SHA256
d68952d05ba48be085eb3430d4e59f33f867ce2834022557ede61c3c763280c2.
Bounded60second combined CPU preflight passed both native parser/bridge controls
and typed Episode rendering/admission:16episodes65admitted16unique7terminal,
no exclusions or truncation. Receipts experiment06-schema-preflight.json and
experiment06-native-preflight.json;17CPU tests, Ruff/shell/diff checks pass.
The following launch recipe was verified before its accepted submission. Intended
run sdpo-qwen35-4b-agent06-schema20-20261018-r1 uses20updates and440episodes,
same05 recipe except explicit schema-bound-v1 transport. After05 terminal native/
guard reconciliation, inspect exact retained06 name/evidence and real GPU before
executing `bash /var/lib/posttrain/qualifications/sdpo-agent06-20261018-r1/driver/run_sdpo_agent_ablation.sh`.
Launcher refuses duplicates/occupied GPU and caps7100seconds+30second kill grace;
combined CPU preflight cap60 keeps total bounded below7200seconds. Capture actual
container ID/manifest immediately in ledger and automation. Do not rebuild the
sealed source or attribute fresh-base parser improvements to SDPO optimization.

The successful04HR reference at12 passes7/7goals while containing3validation
errors from numeric row_id, then repairs all3with strings. Exact native episode
and hash in experiment04-successful-reference-errors.json. Full task-passing
reference is not a clean list of correct actions. Current prompt's "Correct
solution" label must not be interpreted as per-action correctness; consider
explicit error annotations or conservative error-filtered context as a future
separate treatment if observed teacher directions warrant it. Keep05sampling
comparison context unchanged; no silent edit to running04 or sealed05.

Greedy and stochastic policy evidence answer different questions. Original8
training tasks have a greedy7/8 ceiling before updates; stochastic online
trajectories still fail on otherwise solved tasks. Do not conclude stochastic
reliability is unchanged solely from flat greedy scores, or infer improvement
from unpaired online fluctuations. A later explicitly budgeted zero-update
matched stochastic base/candidate diagnostic on already-open train/dev cohorts
could use more repetitions before any sealed-reserve decision. No such run is
selected/launched; reserve remains untouched and greedy controls remain common.

Future CPU preparation: online rollout groups2/4/8 are now explicit; default2
and greedy train/dev/reserve2 are unchanged. A group4 arm on8tasks would use all
32 episode slots and increase chances of a successful sibling while retaining
batch16. No such arm has been selected/launched;13CPU tests pass. Current04 and
sealed05 sources are unchanged. This is an alternative to broader task count.

Draft scripts/qualification/sdpo_reserve_eval.py implements a zero-update fresh
base/selected-adapter comparison for the final reserve slot, with frozen primary,
exact adapter hash/pin/LoRA checks, native actor/rollout parity, original profiles,
two48episode populations and zero optimizer construction. Only Ruff/compile
checks and the later six synthetic audit tests have passed; a guarded launcher
is now prepared. No reserve outcomes read, no primary selected, no GPU submitted
or integration qualified.
Launcher plus native integration and final paired audit still required before
describing this path as ready. Any no-gain search can retain reserve sealed and
use an explicitly different final diagnostic instead.

04step2 failedSlack has a successful sibling, so richer context does reach an
actual failure. However selectedaction is an already-effective Gmail search;
the causal-looking native discrepancy is later unsupported Slack summary text.
Teacher discourages capitalized sender/max_results5 in favor of reference10;
changing those search details does not fix the summary. Preserve exact trace,
teacher input and token preferences in experiment04-slack-sibling-audit.json.
This motivates an explicit failed-episode executed-action frontier sampler,
not just transport-error priority. Such a sampler is a heuristic recipient
selection treatment, not verified per-action goal credit. Qualify separately
after04 checkpoint outcomes; do not change running04 or infer task gains yet.

04step1 teacher mass on student top20 support is approximately98–100% peraction,
teacher top1 outside support generally0–2%. This weakens missing support as a
primary explanation for this batch, not a proof that its teacher is correct.
Successful-context supervision initially reaches successes only; HR has no
successful sibling. A paper full-model LR transferred to LoRA may also produce
small changes: consider an explicit LR-only comparison after04 outcomes rather
than attributing everything to data quantity/topk. Do not predeclare it a rescue.

CPU-only future sampling treatments prepared after the03 audit: explicit
legacy-balanced (unchanged control), distinct-balanced (avoid same action twice
when alternatives exist), and native-error-priority (prefer actions with actual
success:false or tool validation errors, then retain other/terminal behavior).
Empty results and semantically wrong but transport-success calls are not labelled
transport errors. Terminal text does not inherit its previous tool's error flag.
Eleven CPU invariants/Ruff/shell syntax pass. These changes are outside04's
immutable capsule, not a new launched arm or a claim of native improvement.
Any unavoidable duplicate if only one eligible action exists must remain visible
in unique_selected_actions. Sampling repair is a separate treatment from context.

Future-snapshot diagnostics now additionally retain teacher top1 token IDs,
their rank under the student, teacher mass on student top-k support and actual
full-batch Adam change in teacher-top1 log probability on the original sampled
prefix. Computed detached within existing forward passes; objective unchanged.
Ruff/compile pass; this instrumentation is not yet GPU-qualified. Do not infer
semantic correctness solely from teacher top1; decode alternatives and inspect
native outcomes. Running03 capsule remains unchanged.

CPU-only preparation while03 runs: successful-reference selection now prefers
the rollout's own verified success, otherwise a verified successful sibling,
otherwise native feedback. Original implementation chose first successful
sibling even for successful rollouts. Native replay on02 step1:16episodes,
66admitted,16selected/7terminal,50reference actions (46own), no exclusions or
truncation, exact student prefix/action parity. This is not GPU learning evidence
and not a new launched experiment. Eight pure tests pass. Initial CPU container
import omitted self_failure_diagnosis; rerun included the exact02 helper and
passed. Running03 source is unchanged. Evidence:solution-context-cpu-preflight.json.

Further CPU preparation supports explicit wider training cohorts with batch16:
32 tasks rotate across two updates, and tool/terminal action preference alternates
across full coverage cycles. Existing8-task controls retain their exact selection.
Nine tests pass, including full32task coverage without changing batch size.
cohorts-train32-prepared.json freezes the original training-only32 tasks, disjoint
from dev/new reserve/previous reserve/heldout62. This is not yet an assigned or
launched arm; native replay/admission and results-driven selection remain gates.
Train ordered digest:d8058c54e6e0b54db1c9afa3a7751f4b390b7b026fa882144cd1762f7799bcb3.

Prepared a smaller16task coverage treatment that fills32 rollout slots with
two trajectories each while maintaining batch16: original8 plus finance credit
review/intercompany recon, marketing industry tracking, operations expiry and
personalized outreach, support Gorgias/escalation, and simple Salesforce closed
case. These extra8 failed in prior training-only baseline; it is selection
evidence, not a fresh matched score. Preserve source diagnostics as
prior-training32-selection-diagnostics.json. Cohort artifact is
cohorts-train16-prepared.json, ordered digest
0e33bd3510eb967cef5b58906c971f4f726bda15c42fd484ed4b23d6ab0562c7.
No16task GPU run submitted yet. A new matched base is mandatory. Broader32task
artifact remains preparation, not an automatically scheduled additional arm.

The strongest reported setting is task dependent. SDPO arXiv2601.20802v1 Table12
uses reverse KL, top20 and EMA update rate0.01 with LR1e-6 for rich feedback;
its tool-use/no-rich-feedback setting uses Jensen–Shannon, top100, EMA rate0.05
and LR1e-5. Neither is established best for our multi-turn environment.

The previous SELF work found positive teacher-forced repair margins could
coexist with wrong native action prefixes and worse end-to-end performance.
The old reserve24 has been inspected and is not a new final test. A new24-task
research reserve is frozen from the44 tasks outside old pilot32/dev24/reserve24.
Novelty beyond the historical Oct11 audit is unknown. Heldout62 stays sealed.

The inherited agent system text advertises50 turns while the harness allows8.
The new request bridge changes that exact phrase to8 for every arm and its fresh
base control, and records the actual rendered prompts. Historical results are
not matched controls for this experiment.

A CPU review found a potential admission exception if an online call is rejected
before generation: such a record has no sampled_logprobs. The local next-snapshot
driver now excludes incomplete calls before association/rendering. The accepted
01 capsule is unchanged; retain its exact source and any resulting failure.
This is admission bookkeeping, not permission to train truncated actions.

Experiment01 hit a separate valid typed-message shape: tool_calls=None on final
text messages. Native persisted JSON omits optional fields; the actual runner's
Episode.model_dump includes them as None. The regression preflight therefore
rehydrates the pinned Episode type before exercising actual token rendering and
admission. The02r1 CPU-preflight source incorrectly expected explicit nulls in
sparse persisted JSON and was never submitted to GPU. Its source is preserved;
02r2 fixes the fixture reconstruction and passed the exact native regression.

## Decision Log

- Decision (2026-10-18, latest bounded-budget authority): revise unsubmitted R3
  to R5with6updates per fresh branch and five covered-task diagnostics instead
  of32task training-sentinel reruns,
  retaining all124training tasks/32distinct examples and full62x5benchmarks.
  Measured11–12minute full evaluations made the paired10+10fit exceed2hours;
  this revision uses timing only, not heldout outcomes to choose a winner.
  The original20updates is an upper bound, not a reason to exceed7200seconds.
  Both Luna and thinking have the same6step LR/sampling/data horizon;9has12total
  applied steps in separate fresh optimizers,10has6. Short-horizon conclusions
  must remain limited. R3/R4are retained unsubmitted, not failedGPUarms. The five
  diagnostic tasks come from offline source qualification, not heldout failures;
  covered-task acquisition and heldout transfer are separate outcome questions.

- Decision (2026-10-18, supersedes narrow R2 and reserve-slot10 provisions):
  preserve the remaining two bounded experiments, but use all124training tasks
  and32distinct tasks/update as the user's latest data directive. Luna09uses two
  independent10update branches (20total); thinking10uses10updates to match09's
  control horizon and its own thinking-mode baseline. Each still has the original
 7200second total ceiling. Keep failure-frontier unchanged so Luna/thinking are
  not silently bundled with the unqualified first-error sampling hypothesis.
  Current main50run remains immutable and has GPU priority. The old purported
  reserve lies within the now-trained124pool, so retire its independent-test
  claim and phase rather than silently reusing contaminated tasks. Full-budget
  heldout62 monitoring at0/10 is explicitly repeated selection evidence.
- Decision: use only five current-state/rubric-qualified Luna references;
  preserve all124tasks with119native/own-sibling fallbacks. Do not repair absent
  archived snapshots, change historical arguments, ignore unseeded clock/thread
  differences or call all58tool-compatible sources qualified. Report the small
  reference-covered subset as well as full-cohort results. Native learning and
  successful-action retention are required despite passing CPU token checks.

Conditional07 CPU preparation is recorded in experiment07-prepared.json while06
runs. Intended sole treatment is LR1e-6->1e-5 with identical06 sealed runtime,
parser, data and loss. Evidence:05Trello teacher correctly favored ISO token but
saved relative-margin diagnostic barely moved (with the mode limitation above);
this is an exploratory step-size hypothesis,
not proof of a numeric error or missing top-k support. Read06 final native/guard/
teacher evidence before deciding to submit07; no GPU overlap or automatic grid.
Preparation does not increment slots launched.07 would still use20updates and
the independently enforced7200second total execution budget.
07 was CPU-sealed before its accepted launch at /var/lib/posttrain/qualifications/sdpo-agent07-20261018-r1,
manifest a63685f4b9660fd075839c6c2eb9e556be74042241385c59bc7f071df5e8a2b2.
All06driver files and cohorts are byte-identical; experiment LR1e-5 differs.
Combined60second pinned CPU schema/bridge controls and actual06step1 Episode
replay passed:16episodes67admitted16unique7terminal, no exclusions/truncation.
Receipts experiment07-schema-preflight.json/native-preflight.json. That source
was accepted only after reading06 final evidence and inspecting owned states/GPU;
see current Progress/experiment07-launch.json. Do not launch it again.

Decision: Prioritize failure-frontier sampling as experiment05 after04 completes,
using same successful-reference context, original8tasks, LR1e-6/top20/RKL/EMA.01,
batch16 and20update cap. Explicit change is recipient sampling: latest executed
tool action from failed episodes plus distinct selections/terminal retention.
Replay proves inspected Slack training recipient moves from working Gmail search
to unsupported Slack summary. A heuristic recipient does not prove causal goal
credit or correct teacher direction; native rollout and retention gates remain.
Do not change or stop accepted04; it is the matched context-only control.
Date:2026-10-18.

Decision: Actual03 trajectory audit supersedes the tentative broad-data04
launch. Preserve sealed04r1 as CPU-only preparation, never GPU submitted.
At consumed Trello step3, native assertion requires due2026-03-05 but successful
transport storedMarch5,2026. Teacher sampled-token logp advantage for March is
+0.9449568 (~2.57x probability), actual full-batch Adam delta-4.77e-7. Native
feedback therefore does not reliably make teacher preferences correct. Prioritize
verified own-success/sibling solution context on original8tasks as04r2, retaining
same LR/top20/RKL/EMA and comparing common10updates;20update endpoint descriptive.
No new GPU submission yet. Wider data remains a later explicit treatment.
Audit artifact:experiment03-teacher-trajectory-audit.json contains four exact
native episodes, teacher feedback, actions and sampled-token update directions.
HR is selected twice with same generation index in each first5updates, yielding
15unique actions in16contributions; hold this known sampling behavior constant
for the context comparison, then test sampling repair separately. Development
Salesforce errors never enter updates; do not misreport their feedback as ignored
training examples. Date:2026-10-18.

Decision: Latest user authorization raises the per-experiment optimizer cap to20
updates;10 experiment slots and7200seconds total per experiment remain unchanged.
Accepted03 remains immutable at10updates as a clean duration control. Future
launchers must explicitly support and enforce20 rather than silently passing
the old10 assertion. Existing32 concurrent rollout slots can support broader
training coverage; concurrency is collection parallelism, not optimizer batch
size. Any expanded training cohort must be preregistered from eligible training
tasks, exclude development/old inspected reserve/new frozen reserve/heldout62,
and have fresh matched controls and per-task coverage audits. Preserve20update
and two-hour ceilings independently; more data is not automatically better.
Date:2026-10-18.

Decision: Extend the qualified rich-feedback recipe to10 updates before changing
context or objective. Two updates establish runtime behavior, not adequate
learning duration. This is a duration extension with fresh matched base, not a
claim that native feedback is the best configuration. Preserve02 unchanged.
Four inspected failures show display-label versus ID recovery errors, repeated
malformed Salesforce calls, unfinished handbook goals, and premature sync
completion. Actual sampled-token sign alignment alone cannot adjudicate these.
All32 selected02 actions had executed responses; potential unexecuted-action
admission hardening is deferred rather than mixed into03. Date:2026-10-18.

Decision: Fresh budget up to10 SDPO experiments, each at most10 optimizer updates
and7200 seconds total container execution including startup, diagnostics,
collection, training and evaluation. Qualification uses slot01, not an unlimited
exemption. Prior SELF slots remain exhausted and separate. Date:2026-10-18.

Decision: Start original pinned Qwen3.5-4B through a newly initialized zero-effect
LoRA; never start from damaged SELF checkpoint64. Use native feedback first,
not external calls, Luna labels or hand-repaired gold actions. Date:2026-10-18.

Decision: First experiment qualifies two updates of rich-feedback SDPO with
actual native rollouts. Each update collects new stochastic student trajectories
and consumes its selected transitions once. Repeated epochs over old bad-prefix
fixtures are not SDPO qualification. Date:2026-10-18.

Decision: Do not allocate the remaining nine slots in a fixed grid. Diagnose
feedback informativeness and native behavior, then vary one mechanism where
possible. Candidate comparisons include teacher feedback vs verified successful
siblings, regularized teacher rate, divergence/top-k support, LR, thinking and
reward alignment. Name hybrids explicitly rather than attributing them to plain
SDPO. A full training/heldout/LFM restart is not authorized by a local pass.

Decision: Reserve the final available slot for a zero-update fresh-base versus
preselected-candidate comparison on the new research reserve, if there is a
candidate worth that test. Use at most slots02–09 for further adaptive fitting
and diagnosis. Do not spend every slot tuning and then claim transfer without
an independent comparison. Date:2026-10-18.

## Outcomes & Retrospective

Final audit summary (2026-10-18): all10 original attempts terminal and audited.
Eight completed runtime receipts;01 and09 failed. Counts include the zero-update failed qualifier01; successful
runtime execution is not a positive learning result. All01–08diagnostics used
8training/4development tasks, so those small-cohort scores cannot be compared
directly with new124training/62heldout arms. Exact configs, sources, native
audits and run identities are in experiment-ledger.json.

| Slot | Actual updates | Treatment / diagnostic | Measured result | Status |
| --- | ---: | --- | --- | --- |
| 01 | 0 | Native feedback,LR1e-6 | Native tool_calls=null reader error before learning;40episodes preserved | Failed qualifier |
| 02 | 2 | Same recipe,typed-null handling repaired | Train success87.5->87.5%;development25->25% | Audited |
| 03 | 10 | Longer native-feedback fit,LR1e-6 | Train success87.5->87.5%,partial91.07->91.07%;development25->25% | Audited |
| 04 | 20 | Own/successful-sibling teacher context | Same greedy goal/partial results as03; contexts alone did not rescue decisions | Audited |
| 05 | 20 | Failure-frontier sampling | Development success25->50%,but recovery exploits quoted-string IDs around parser bug; not broad learning proof | Audited,parser confound |
| 06 | 20 | Same sampler with repaired schema parser | Fresh development base already50%;training partial91.07->92.86%,development unchanged50% | Audited |
| 07 | 20 | Matched10xLR,1e-5constant | Training partial91.07->98.21%at15->94.64%final;development50%unchanged;solved tasks retained | Audited |
| 08 | 20 | High initialLR with cosine decay to1e-6 | Training partial91.07->94.64%final;development partial54.76%at15->50%final,success unchanged50%;no added guard identities | Audited |
| 09 | 6control,0Luna | Full124,batch32;planned paired6control+6Luna | Control phase timeout during finalheldout;45/310persisted, no closed post-result. Reference-training20%success/47.14%partial unchanged. Luna never started | Terminal, comparison inconclusive |
| 10 | 6 | Full124,native student+teacher thinking | Heldout success58.06->54.84%;partial83.34->80.52%. Reference-training40->20%success;offboarding retention lost | Audited; runtime qualified, learning rejected |

The separate full-pool50campaign is not an eleventh original ablation or a
substitute for09/10. It failed after38updates with lastsaved/evaluated30;
40/50are missing, gate inconclusive. No fullresume or automaticLFM/250run.
Broader heldout monitoring is repeated selection evidence, not pristine final
validation. The ten budgeted attempts are complete; this means eight completed
runtime experiments and two preserved failures, not ten passing learning results.

### Student checkpoint success on the larger evaluation

The strongest observed trained student is checkpoint10 of the separate full-pool
batch32 campaign:180/310 successful episodes,58.06%, versus160/310,51.61% at
its matched non-thinking base. Checkpoint20 scored170/310,54.84%; checkpoint30
175/310,56.45%. The campaign failed after38 updates, so40/50 do not exist.
The thinking base also scored180/310 before fitting; thinking checkpoint6 scored
170/310. These modes require their own controls. Earlier06–08 development success
was50% at every evaluated step5/10/15/20 on only four development tasks.
Experiment07's training partial-credit peak is not the best heldout checkpoint.
Selecting the maximum after repeated heldout inspection is descriptive, not an
independently validated selection protocol. Checkpoint10 also added preservation
violations, so it is not a clean overall qualification winner.

### Final thinking-arm uncertainty, signal and preservation audit

For310/310 scored episodes at both base and checkpoint6, success delta was
-3.23points, paired62-task bootstrap95% interval[-9.35,+1.29]; partial delta
-2.82points, interval[-7.24,+0.93]. Both include zero: improvement was not
established, and population-wide harm is not statistically established either.
Absolute success intervals were[45.48,70.32]% and[41.94,67.74]%, separately
from paired uncertainty. Native initially-satisfied assertion checks broke
110/920 at each endpoint, but affected episodes rose45/140->50/140, with five
new broken identities, six resolved and one existing identity worsened. This is
a preservation proxy, not a complete independent harm taxonomy.

All192 selected actions retained exact original student prefixes/action IDs and
included reasoning targets. All six optimizer moment steps and checkpoint hashes
passed; the496-tensor EMA recurrence matched exactly at every step. Training
excluded23 reply-limited and10 overlong whole transitions, without truncation.
No authoritative transport retry counter exists; retry count remains unknown.
Native assessment batches were absent, so batch deduplication was inapplicable;
native final assertions and scored episodes are the measurement authority.

Four inspected selected failures show the remaining learning problem: native
hindsight did not supply verified repaired arguments for Slack channels, a
Payment worksheet, a PLAT project or currency strings. Three of these four
failed original actions increased in conditional log probability after the
actual full-batch Adam update. This is local conditional evidence, not proof
that any individual gradient caused an entire rollout failure. Lower loss and
fewer raw API errors therefore did not establish better task completion.
See experiment10-saved-evidence-final-r5/four-thinking-failure-signals.json,
experiment10-terminal-r5/final-summary.json and the unified reviewed results.

No reliable general learning winner was established in this adaptive, single-seed
search. The short thinking fit does not disprove SDPO generally; the Luna arm
remains untested because its control exhausted its phase deadline. The concrete
outcome is a completed bounded investigation with preserved failures and enough
evidence to reject an automatic long-training or LFM continuation.

### Four selected error-action checks from09control

Read-only CPU extraction joins original task/episode/generation IDs, complete
action tokens, native observed error, teacher token scores and actual full-batch
Adam before/after scores. Evidence experiment09-terminal-r5/four-error-action-signals.json;
source review_failure_signals.py. No new gradients, fitting, external calls or
accepted-source changes. The teacher argmax is tokenwise under the original
action prefix, not an autoregressively generated corrective example.

| Actual selected failure | Teacher evidence | Actual Adam evidence | Supported interpretation |
| --- | --- | --- | --- |
| Fire testing, step1: repeated Jira call returns project-not-found | FIRE argument span remains teacher argmax | Span summed logprob changes-0.00185 | No demonstrated repaired call; FIRE is requested by task, so an unknown-project error alone cannot establish a correct replacement |
| Offboarding, step1: empty spreadsheet ID; tool reports known ss_offboard | Empty-ID markup span has negative mean sampled advantage-0.475 | Span summed logprob changes-0.00519 | Bad span is suppressed locally, but no verified ss_offboard repair or task completion is established |
| Equipment inspection, step2: guessed meetings/demo event type rejected | Invalid value remains teacher argmax, mean sampled advantage+4.221 | Invalid span summed logprob changes-0.0571 | Teacher favors the failed value; full-batch Adam moves differently. Neither direction alone proves a correct alternative |
| Benefits, step2: full title used instead of spreadsheet ID | Full title remains teacher argmax despite native not-found feedback | Failed span summed logprob changes+0.00896, despite negative mean teacher advantage | Shared update can increase a failed argument; richer context is not a correctness guarantee |

All four source trajectories have native success0. Local changes are neither
independent per-action gradients nor causal task-reward estimates. These cases
support continued skepticism about automatic correct-call supervision, not a
claim that Luna failed (it was never used) or that thinking has improved learning.

Experiment01 failed before updates and is retained. Its exact name is
sdpo-qwen35-4b-agent01-rich-qualify-20261018-r1 and capsule is
/var/lib/posttrain/qualifications/sdpo-agent01-20261018-r1. It began at
2026-10-18 02:15:40 PKT (2026-10-17 21:15:40 UTC), with source-manifest SHA256
ab43cf507143f401228feaa1d4f75a6f2b8a7ebdb71bfe837aabf79c50e633dd.
Experiments02–04 completed and were independently audited:2/10/20optimizer
updates and80/232/440native episodes, respectively. Both native-feedback and
verified-success-context recipes left matched greedy success and partial credit
unchanged. Native preservation violations decreased in development; report this
separately from goal gains and comprehensive safety. Some stochastic HR attempts
completed, but no matched stochastic reliability gain has been established.

The strongest qualitative finding is recipient selection. A failed Slack episode
had a working email search and a later unsupported summary; legacy sampling
trained on the search. Native replay verifies the new frontier sampler selects
the summary and16distinct actions. Another finding is teacher reliability: final
checks did not prevent a wrong-format date from receiving positive teacher
preference, and a task-passing reference can include intermediate tool errors.
These findings motivate controlled sampling and context studies, not a claim
that distillation itself has been qualified as effective.

Experiment05 completed20updates/440episodes and provides the first positive
native development success result:25->50% at15/20, with complete guard coverage
and original solved-task retention. The observed Salesforce improvement is
recovery via quoted-string IDs around the reproduced parser bug, not proof of
broader reasoning/task knowledge. Four-task uncertainty remains large and reserve
has not been evaluated.06 then showed the repaired parser alone yields50%dev
success; actual identical first model tokens now execute correctly.06 SDPO adds
one HRtraining goal through fewer wasted lookups, partial+1.7857points with
8task deltaCI[0,5.3571]points, but no development goal gain. Its artifact/native
guard audit is complete. Experiment07 was the matched LR control:
sdpo-qwen35-4b-agent07-lr10x20-20261018-r1, capsule
/var/lib/posttrain/qualifications/sdpo-agent07-20261018-r1. Exact source/container
receipts are in experiment-ledger.json and experiment07-launch.json. Both07and08
are now terminal and audited; eight of ten slots are launched. The15-minute heartbeat monitor-self-pilot-every-15-minutes
is ACTIVE on this SDPO protocol and must not execute obsolete SELF instructions.
Complete the remaining adaptive search and report all attempts. No optimal
configuration or generalization gain has yet been demonstrated. The separately
authorized full124/50campaign now monitors heldout62; oldSELF256/LFM remain on hold. Native outcome evidence is required;
CPU replay and local gradient checks alone do not establish learning.

## Context and Orientation

SDPO means Self-Distillation Policy Optimization: current student actions are
scored by the same model under extra training-only feedback, and the student
learns from the difference. A teacher is an exponential moving average (EMA):
each update moves teacher adapter weights toward student weights by a stated
rate. Gradients never update the teacher through its scoring branch.

scripts/qualification/sdpo_agent_ablation.py owns this prototype. It reuses the
native Verifiers HTTP bridge from self_campaign.py and the optimized engine from
self_feedback_vllm.py. sdpo_agent_config.py owns cohort separation, teacher
feedback construction, whole-transition admission and task-balanced selection.
The actual selected TRL loss is imported from
trl.experimental.sdpo.loss_utils.compute_topk_self_distillation_loss; it uses
student top-k support, the teacher on that support and a normalized tail bucket.
Alpha1 is reverse KL; alpha0.5 is Jensen–Shannon. No teacher gradients flow.

Each update uses16 complete actions, two per training task. Where present one
is a native final answer, preserving stopping behavior; another is a native
tool action. If absent, no terminal target is invented. Every action retains
its original generated token prefix; no corrected branch replaces it.
The teacher receives native assertion facts and the current action's executed
tool response, explicitly marked as hindsight. It is not full per-action goal
credit assignment. Failed/successful native checks may contain privileged
instance facts; transfer must therefore be measured separately from memorization.
Teacher prompts and selected row identities are saved. Students never see them.

All primary evidence is native traces and actual generation token IDs; summaries
are rebuildable. Each experiment writes input selections, fresh-base gate,
actor/rollout parity, batch-invariance evidence through32, per-update admission,
teacher-vs-student sampled-token advantages, full-batch actual Adam probability
changes, native train/dev outcomes, adapter exports, optimizer states and final
receipt. Local gradient direction is diagnostic, not proof of correct actions.

## Plan of Work

Milestone1 qualifies the actual pinned runtime and two native updates from fresh
base. Greedy matched baseline and final evaluations use eight training tasks and
four development tasks, two episodes each. Online collection is temperature1,
two episodes per training task, refreshed before every optimizer update. Context
16384, reply2048, episode generated ceiling12288, eight assistant turns,32
episode/server slots and16GiB KV remain explicit common settings. This is an
agent adaptation, not a replication of the paper's8192-token single responses,
eight rollouts/question or full fine-tuning. No training sequence is truncated.

Milestone2 diagnoses the first outcomes and selects the next controlled change.
The new ledger records hypothesis, falsifier, exact config, source digest, IDs,
time/update caps, acceptance, actual results and reason for the next choice.
No new entry is launched before the prior owned run exits and real GPU compute
is free. Two-hour timeouts retain all partial evidence and consume their slot.

Milestone3 compares promising configurations against their fresh base and matched
controls using task-cluster bootstrap intervals with missing coverage explicit.
Use development for selection. Freeze one primary before inspecting the new
research reserve; reserve results do not select a replacement winner. Keep native
preservation/harm applicability separate from positive goals and unknown checks.
No performance superiority is established by these small diagnostics alone.

Primary selection protocol, declared before further-arm/reserve outcomes:
eligible checkpoints must have complete native score/initial-guard coverage,
no per-task development success/partial regression, no added preservation violations,
no regression on the common initially successful training tasks (including all
seven initially greedy-solved original tasks, not only the four named retention
tasks), and an observed
native success/partial improvement within their own train or development cohort
with an inspected example of improved behavior. Matched stochastic evidence can
support eligibility only if an explicitly budgeted equal-temperature base/control
comparison is implemented and audited; unpaired online fluctuations cannot.
Rank eligible candidates by development success, development partial, fewer
development tool mistakes, earlier saved checkpoint, then earlier experiment.
Do not rank absolute training means across different training cohorts. A candidate
with training gains only is exploratory task-learning evidence; reserve is then
an independent transfer test, not confirmation promised in advance. If none is
eligible, keep reserve sealed and use final slot for an explicit mechanistic or
sampling control, then report that this search did not find a useful candidate.
Freeze the exact selected adapter/file digest and rationale in a sealed selection
receipt before launching any reserve comparison. Do not reopen selection after
reserve outcomes. Heldout62/full-training/LFM remain on hold regardless.

## Concrete Steps

From the worktree run:

    PYTHONPATH=scripts/qualification .venv/bin/pytest -q scripts/qualification/test_sdpo_agent_config.py
    .venv/bin/ruff check scripts/qualification/sdpo_agent*.py scripts/qualification/test_sdpo_agent_config.py
    bash -n scripts/qualification/run_sdpo_agent_ablation.sh
    git diff --check

SSH uses dstack@192.168.110.12, key
/home/hammad/projects/ai-infra/.state/ssh/dstack_worker with -F /dev/null,
BatchMode=yes, IdentitiesOnly=yes, IdentityAgent=none, GSSAPIAuthentication=no,
ConnectTimeout=8 and outer timeout25. Default SSH agent/config previously hung.
No credential values belong in evidence. On host use python3, not python.

The selected immutable runtime capsule is
/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5 (read only).
New capsules use /var/lib/posttrain/qualifications/sdpo-agentNN-20261018-r1.
The new driver and supporting source hashes live in each capsule. The launcher
checks source hashes, retained container identity and actual GPU processes.
It launches a network-none container with an outer7200s timeout. Record Docker
receipt immediately. Do not run the launcher from the local host; stage first.

## Validation and Acceptance

Require a byte-identified selected TRL loss and source snapshot, fresh-base
active-vs-disabled adapter parity, native-format requests, passing selected-token
actor/vLLM parity, exact batch invariance through32, two actual optimizer moment
steps and EMA updates, fresh policy exports and finite nonzero gradients.
Reconcile all native episode/task IDs, missing scores/errors, repetition counts,
length limits, context exclusions, repeated calls and native preservation checks.
Never zero-fill missing scores or equate agent_completed with task success.
Inspect actual calls for3–4 failures to interpret the dense teacher signal.
Qualifying the runtime is distinct from qualifying learning improvement.

## Idempotence and Recovery

Never mutate a running capsule, duplicate an accepted run, automatically resume
a crash or overwrite failure evidence. A repair requires an evidence-backed new
identity and its own remaining budget slot. No unrelated GPU interruption,
dependency-pin edits, quantization, paid cloud, commits or pushes. The old SELF
capsule and all unsuccessful attempts remain immutable. Abort launches on busy
GPU and monitor an owned accepted job rather than competing with it.

## Artifacts and Notes

New ledger/cohorts/receipts belong under
docs/research/evidence/sdpo-vllm-20261018/. Research reference is
https://arxiv.org/abs/2601.20802 and authors repository lasgroup/SDPO at
7c457fc1b1f636ae794eb0362ba37d4743b06fbc. Literature alternatives were reviewed
before authorization: OPSD uses a frozen initial teacher/reference solution;
SDAR/SRPO/OPDVR combine or gate reward signals and are separate algorithms.

## Interfaces and Dependencies

Exact Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, BF16,
LoRA16/alpha32/dropout0, all text linear layers; environment revision
a0c7378b179e2b267382b8125192876502d82f89. Selected TRL source
290c03d8e1553db509b0d033907f60fc1cb47b82, loss_utils.py SHA256
530b571c0001bbf85bb0c78725d32645a4d7155660e22e41580fab5c9f7041e4.
Pinned vLLM0.29.1.dev4 fork f09e4479123d348cee87d217c695a22f9b2daacc.
Runtime image registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@
sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337.
No sibling repository edits or public API changes are planned.

Revision note2026-10-18: created fresh plan after research and explicit renewed
execution authorization; preserves the completed negative SELF investigation.
