# Run a fresh batch32 SDPO campaign for250 updates

This living execution plan follows docs/templates/PLAN.md. Work only in
/home/hammad/projects/worktrees/rl-rft on codex/rft-warm-start. Maintain Progress,
Surprises & Discoveries, Decision Log, and Outcomes & Retrospective. This is an
unpublished project research prototype; it does not change the frozen public
train.distill contract, package boundaries, or dependency pins.

## Purpose / Big Picture

The user explicitly requested a longer Qwen SDPO run using the current eight
training tasks,32 examples per optimizer update instead of16, and a high initial
learning rate followed by a short decay and a constant floor. Test whether the
short-run training progress survives longer optimization and transfers to the
four development tasks. Loss decline is not evidence of reward improvement.
Experiment08 is an exploratory predecessor, not a successful generalization gate.
Its last live observation was12/20updates; development success remained50%.

## Progress

USER-SUPERSEDED2026-10-18: the requested broad corpus was not implemented by this
eight-task campaign. User authorized all124 training tasks,32distinct tasks per
update,50updates with evaluations every10. This owned run was stopped/preserved
exit143 at177logged updates and checkpoint175. Inflight unlogged work is possible;
do not claim exact checkpoint177 resume. Follow docs/plan/sdpo-pool124-workstation.md
and its new immutable qualifier/campaign. Never resume this run or count it as
full-pool evidence. Native checks through175 remain valid as narrow diagnostics.

- [x] Read user authorization, current code, pinned runtime and live08state.
- [x] Implement explicit campaign profiles without relaxing20update ablation caps.
- [x] Implement hold10/cooldown20/flat250 rate, accumulated batch32, strict distinct
  transition admission and25update checkpoint/evaluation cadence.
- [x] Exact-source pinned CPU native replay and selected-loss checks passed;
  source manifestb1f66e8333c0d6f2c35202d03283961b5c25671f619a40d8d08904962dec8c3f.
  CPU replay67complete actions,32distinct selected/fourper task, no exclusions.
  Four campaign tests and40related protocol tests passed;07analyzer regression
  reconciled20updates/440episodes unchanged.
- [x] Sealed capsule staged and host queue accepted2026-10-18 01:28Z,
  PID2682629, waiting_for_experiment08; this is not a GPU training receipt.
- [x]08completed20updates/440episodes, exited0 at01:35:29Z and finalnative/
  checkpoint/guard/policy audits passed. Devpartial gainat15reverted at20.
- [x] Queue accepted GPU qualification01:35:51Z:
  sdpo-qwen35-4b-b32-qualify-20261018-r1,
  Docker6081ccfec7f606c5237b4edc14397c3a4a1d9e280fe3f8d8fb4d898606c04ea0.
- [x] Qualification exited0; actual2saved moment/EMA updates,136unique native
  episodes/8populations with complete scores/noerrors,32distinct/fourpertask per
  update and32teacher/Adam diagnostics. Native tool/result/nonce continuity,
  parity/batch32 invariance and finite changed student/teacher files audited.
  Trainingadmission1:131admitted/1excluded;2:132admitted/0excluded, no truncation.
  Greedytrain/development outcomes unchanged through two qualification updates;
  this is runtime qualification, not rewardgain.
- [x] Fresh250update campaign accepted01:47:55Z:
  sdpo-qwen35-4b-b32-250-20261018-r1,
  Docker7bf3dfabdee5c2eb060f45d71a9799d71c4f7cad67332de69c847c759ce862a5.
  Fullrun startup, zero optimizer updates confirmed at this handoff. Never
  duplicate/restart/mutate the source; queuefinished its submission duty.
- [ ] Monitor actual updates/checkpoints, audit final curves and report.

First fullrun update is now confirmed: step1/batch32,LR1e-5,loss.3593589,
gradnorm2.02599(preclip),moment/EMA1,elapsed340.58s. Its own fresh-base gate
confirms zero B matrices and active/base parity. This is separate from the two
qualification updates. See first-update.json in the campaign evidence directory.
Do not infer rewardgain or stable warm-step timing from this first update.

Update10 monitoring: first10actualupdates nowverified, all at1e-5 with exact
moment/EMA counters1–10, finite losses,32distinct selected transitions perupdate
(four per each ofeighttasks), zero wholetrainingexclusions and zero truncation.
Warm intervals aftercoldstep1 have median115.39s,min101.27s,max141.05s(nine
intervals). Roughremainingtime8–10hours includes anticipated periodic evaluation
overhead; it is not a completion guarantee or cross-concurrency throughput claim.
Source warm-timing-step10.json. No post-update development score is yet available;
the next planned evaluation is25.11–20must nowdecay and20–250remainat1e-6.

Update20 is observed at1e-6,loss.26500,moment/EMA20. Existing stochastic training
collections provide diagnostics before the scheduled greedycheck: firstfive
policies0–4success84.375%/partial91.7857%, lastfivepolicies15–19success88.125%/
partial95.4464%,160episodes perwindow. All640episodes throughcollection20have
complete success/partial scores and no scoringerrors. These windows compare
changing policies onthe sameeight trainingtasks attemperature1; they are not
greedy validation, independent-task generalization, or a causal schedule result.
Losses are measured after each collection onselected32actions; outcomes belong
to policyversion stepminus1. Source online-through20.json. Keep this distinction
when comparing reward andloss trends; development25isstillpending.

The online data contains one fullysuccessful HRtrajectory throughpolicies0–19:
one of four HRepisodes atpolicy16(collection17). Native metrics7/7requiredchecks
pass, and all retained native assertion records pass; the trajectory contains11
toolcalls within the unchanged8turn budget. Allfour selectedHRactions atupdate17
actually receive that verified own/sibling success as teacher-only context.
Source hr-own-reference-step17.json preserves its native filehash/referenceID,
actual calls, nativecheckresults and consumedteachertext. This demonstrates the
reference path is exercised, not that such a rare stochastic event proves learning
above the fresh base. It also explains why offlineLuna's complete HR reference
remains an informative matched intervention when native successes are sparse.

Response-provenance audit independently joined all640selected transitions
through20updates to rawnative traces. All353selected tool-bearing actions have
native toolresults for every generated toolcallID; errorresponses remainvalid
feedback, not correctness labels. This checks actual feedback alignment rather
than trusting an aggregate counter. Source native-response-provenance-through20.json.
The successfulHRreference itself emits11calls, ofwhich10have recorded toolresults;
a final confirmationread is unexecuted atmax_turns. Its fullnative score therefore
qualifies the endstate, not every emitted step as executed/gold. That pendingcall
is not a selected no-response training action, but remains visible inreference
context. Treat this as a reference-quality limitation; do not call the whole
reference a validated executable demonstration. Source reference-execution-audit.json.

Checkpoint25native evaluation completed:16train/8developmentepisodes, complete
scores/noerrors. Train success87.5%unchanged,partial91.0714->92.8571%; development
success/partial50%unchanged. All7previouslysolvedtrainingtasks retained. Raw
checkpoint25student/EMA tensors arefinite/nonzero, storedoptimizer moments25and
actualLR1e-6 verified, allcheckpointmanifest hashesmatch. Nativeinitialguard
audit has14trainingcheckswith0violations atbase/25,38developmentcheckswith2base
violations and0at25,unknown0. These nativepreservationchecks are not a full
independentharmtaxonomy. See checkpoint25-files.json,checkpoint25-native-guards.json
andcurve-through25.json. No claim of a development goalgain at25.

Checkpoint50 audit: train success remains87.5%; partial credit rises from
91.0714% at base to94.6429%, a3.5714point paired task mean gain with95%
bootstrap interval[0,10.7143]points across8tasks. Development success and partial
remain50%. Native preservation audit reports14training checks/0violations and
38development checks/0violations, unknown0. Saved student/EMA tensors are finite
and changed; optimizer moment step50, LR1e-6 and all manifest hashes verified.
See checkpoint50-files.json, checkpoint50-native-guards.json and curve-through50.json.
The live container was subsequently confirmed running at53actual updates, loss
0.1877465, moment/EMA counts53 and LR1e-6. This is not development improvement.

Across50updates, the logged actor/native selected-token audit covers177255
gradient-bearing tokens: mean absolute log-probability difference0.004899nats,
p95 0.030155, maximum0.916839; mean importance weight1.000004 and one position
at cap2. All values are finite. This checks selected positions, not full-vocabulary
parity or corrected-call probability; it does not establish reward improvement.
Source importance-through50.json. Read-only audit_saved_checkpoint.py is retained
beside the evidence for subsequent saved checkpoint audits without GPU allocation.

Checkpoint75 completed and audited:75actual optimizer/EMA updates; saved moment
step75, LR1e-6, finite changed student/teacher and all checkpoint hashes verified.
Loss0.1988107, grad norm1.16996. The16training/8development greedy episodes have
complete scores and no errors. Train success87.5%, partial94.6429%; development
success/partial50%, matching checkpoint50. Train partial gain vsbase remains
3.5714points, paired8-task95%CI[0,10.7143]points, not a demonstrated development gain.
Native guards: training0/14violations; development2/38violations, unknown0. Both
development violations are the SAME handbook-acknowledgment email-preservation
assertion/recipient as base, after being absent at25/50; no new broken identity
relative tobase. Thus the earlier preservation improvement has not persisted.
Evidence checkpoint75-files.json, checkpoint75-native-guards.json and
curve-through75.json. The live campaign remains authorized and unmodified;
bounded Luna09/thinking10 remain staged/unsubmitted behind it. Previous goal
turn was a verified wait followed by these new checkpoint audits, not completion
of the ten-ablation objective.

Checkpoint100 completed and audited after a verified live wait from75. Saved
optimizer moment100, EMA count100, student/teacher finite tensors and all file
hashes pass; actual LR1e-6, logged loss0.1709651 and grad norm1.29007. All24greedy
native episodes scored without errors: train success87.5%, partial94.6429%;
development success/partial50%. The partial gain is confined to training and has
not increased since50. All7solved training tasks remain solved. Preservation
audit has0/14training and0/38development violations, unknown0; the handbook
violation at75 is absent again, so this check fluctuates rather than proving a
sustained safety improvement. These are native preservation checks, not an
independent harm evaluation. Sources checkpoint100-files.json,
checkpoint100-native-guards.json and curve-through100.json. Continue the accepted
250campaign unchanged;09/10 remain unsubmitted, and only8bounded slots are audited.

Checkpoint125 (halfway) completed and audited after a verified live wait from100.
All saved checkpoint hashes match; actual optimizer moment125, EMA125, finite
student/teacher tensors and LR1e-6 verified. Logged loss0.1660979, grad norm1.58173.
All16train/8development greedy native episodes scored without errors. Train
success87.5%, partial94.6429%; development success/partial50%, unchanged since50.
All7solved training tasks retained. Native preservation violations0/14train and
0/38development, unknown0. No development learning gain established at halfway.
Evidence checkpoint125-files.json, checkpoint125-native-guards.json and
curve-through125.json. This audit does not complete either remaining bounded
ablation; continue250, then09and10 under their own previously authorized caps.

Checkpoint150 audit after verified live wait: saved optimizer moment150/EMA150,
finite student/teacher tensors, LR1e-6 and all checkpoint hashes pass. Logged
loss0.1670974, grad norm1.44411. Complete16train/8development native scores/noerrors;
train success87.5%, partial94.6429%; development success/partial50%. No goal gains
since50. All7solved training tasks retained. Native preservation violations
0/14train and2/38development, unknown0: same handbook recipient/assertion as
base/75, absent at100/125. This recurring failure prevents claiming sustained
preservation improvement. Checkpoint150-files.json, checkpoint150-native-guards.json
and curve-through150.json retain raw audit results. Accepted source unchanged;
the ten-ablation objective still has09/10 unsubmitted, not replaced by this run.

Checkpoint175 audited after a verified live wait: actual saved optimizer
moment175/EMA175, LR1e-6, finite changed student/teacher tensors and all file
hashes pass. Logged loss0.1368632, grad norm1.55140. All24greedy native episodes
scored without errors. Train success87.5%, partial94.6429%; development
success/partial50%, unchanged since50. All7solved training tasks retained.
Native preservation violations0/14train and2/38development, unknown0: same
handbook assertion/recipient asbase/75/150, no newly broken identity relative
to base. Lower loss still does not demonstrate a development outcome gain.
Sources checkpoint175-files.json, checkpoint175-native-guards.json and
curve-through175.json. Continue the accepted250campaign;09/10 remain staged only.

Sampling diagnosis (current250run unchanged): among80HRtrainingepisodes through
20collections,35failed trajectories contain realtool errors, but only2first-error
actions enter selectedbatches. All353selectedtoolactions do have genuine native
responses; the issue is decision-point coverage, not fabricated feedback. The
next unsubmitted control/Luna/thinking recipes will test first-error-frontier:
prioritize the earliest admitted observederror in failed trajectories, fall back
to existing last-actionfrontier when noerror exists, retainterminal andsuccess
coverage. This is a hypothesis, not a proven learning fix or per-goalcredit.
Source first-error-coverage.json. The longrun's accepted failure-frontier source
remains immutable; newer policy tests must be separately named and source-pinned.

## Surprises & Discoveries

The old driver hardcoded16 in selection, loss accumulation and metrics; changing
only rollout concurrency would not produce batch32. Four online trajectories per
training task now produce32episodes per update, from which32distinct complete
assistant transitions are selected. They are not32new tasks or32whole episodes
backpropagated end to end. Training tokens remain untruncated; whole incomplete
or overlong transitions are excluded. Insufficient distinct actions stop the run
rather than silently repeat a transition.

A naive32-action selection would have changed the old tool/terminal balance:
only the first task cycle selected ordinary/frontier actions and the later three
favored terminal answers. campaign_batch now performs two disjoint16-action
selections, keeping the old mixture where actual complete terminal actions exist.
No artificial terminal action is created when an episode did not finish.

## Decision Log

Decision2026-10-18: interpret the user's schedule as applied LR1e-5 atupdates1–10,
cosine decrease over11–20 to1e-6, then exactly that floor through250. This retains
the prior high/normal rate meanings. Effectivebatch32 is microbatch1 with32
accumulated transition losses averaged before a single AdamW update. Start fresh
from the original base, not an unqualified continuation of08optimizer/EMA.

Decision2026-10-18: keep08's16Kcontext/2048reply/nonthinking and native
solution-or-feedback teacher. The separately authorized32K/4K applies to Luna09
and thinking10, which remain pending. No Luna demonstrations, additional
manifest-goal/harm label provider, external model/judge calls or heldout training
enter this campaign. The existing rich teacher feedback does contain native
final assertion types/parameters/pass facts for unsuccessful trajectories;
successful own/sibling trajectories provide reference context otherwise. This
is teacher-only hindsight. Do not claim the teacher receives no goal information;
student prefixes and original action IDs retain their native observations only.

Decision2026-10-18: the new explicit250update authorization is a separate long
campaign, not a claimed≤20update/two-hour ablation. Use a24hour hard safety limit,
with no automatic crash resume. Preserve all prior attempts. It takes queue
priority after08; retain09/10 plans and budgets, and keep heldout/LFM on hold.
The long run and its qualification do not replace either remaining bounded
ablation or enter that search's frozen-primary reserve selection. Report its
curves separately:250updates/batch32 is a different exposure budget from the
10–20update/batch16 comparisons. Do not pool native episodes across these runs.

Decision2026-10-18: evaluate train16episodes and development8episodes at0 and
every25updates through250, with two replicas/task. Save student/EMA adapters,
actual optimizer and RNG files at25/50/75/100/125/150/175/200/225/250. Every
update retains actual moment/EMA counters, loss, applied LR and teacher/Adam
directions; adapters are refreshed for rollout after every update. No silently
selected heldout winner. Qualification saves/evaluates after each of two updates.

## Outcomes & Retrospective

Batch32 runtime qualification passed two actual updates and the separately fresh
250step campaign is accepted, not completed. No fullrun optimizer update is yet
confirmed. Native/file gate receipts are durable under
docs/research/evidence/sdpo-campaign-b32-20261018/qualification-*.json, with
queue-state.json retaining the training acceptance. Experiment08checkpoint15
development partial credit54.7619% versus50%
base is exploratory evidence on four tasks, not established generalization.
Final08developmentpartial reverted to50%. Support baseline/final responses hit
2048reply tokens;15fitsfourwritesin1951tokens then cannotadmit nextcall because
14357prefix+2048reply exceeds16K. The longcampaign preserves16K/2K for comparison,
so these evaluation limits remain explicit. Only the pending matchedlarger-profile
experiments can separate token-capacity effects from teachercontext effects.

## Context and Orientation

Pinned Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a BF16,
fresh LoRA16/alpha32/dropout0, all text linear layers, visual modules excluded.
SDPO uses teacher-only native hindsight plus verified own/sibling successful
trajectory context, top20+tail reverseKLalpha1, EMA update rate.01. AdamW decay.01,
gradient clip1; no raw-feedback CE and no gold-call CE. Student/teacher
nonthinking; schema-bound-v1 tool parser preserves declared string arguments.
32episode/server slots,16GiBKV,16384context,2048reply,12288generated tokens per
episode and8turns. Source environmenta0c7378b179e2b267382b8125192876502d82f89.
Model/dependency pins and runtime image exactly match08; no sibling repo edits.

The frozen cohorts.json contains train:hr.offboarding_automation,
simple.email_sf_contact_mobile_update,simple.gmail_reply_meeting_request,
simple.slack_ceo_announcement,simple.mailchimp_add_subscriber,
simple.sf_lead_revenue_update,simple.trello_catering_contract,
finance.po_email_logging. Development:hr.handbook_acknowledgment,
simple.email_sheets_update_reply,simple.sf_case_in_progress,
support.zendesk_hubspot_org_sync. Reserve24 and heldout62 never enter updates.

## Plan of Work

scripts/qualification/sdpo_long_campaign_config.py owns strict execution
profiles and campaign_learning_rate. sdpo_agent_ablation.py uses those profiles
for selection/gradient averaging/checkpoint/evaluation cadence, preserving its
default bounded ablation behavior. run_sdpo_long_campaign.sh separately guards
qualify/train submissions, native/source receipt checks and GPU occupancy.
analyze_sdpo_agent.py and audit_sdpo_checkpoint_files.py understand both bounded
ablations and the explicit campaign profile. New tests verify schedule boundaries,
distinct balanced32transition selection and actual selected-loss accumulation32
gradient equality. No generic capability API changes are made.

## Concrete Steps

From /home/hammad/projects/worktrees/rl-rft:

    PYTHONPATH=scripts/qualification .venv/bin/pytest -q scripts/qualification/test_sdpo_long_campaign.py scripts/qualification/test_sdpo_agent_config.py
    .venv/bin/ruff check scripts/qualification/sdpo_long_campaign_config.py scripts/qualification/sdpo_agent_ablation.py scripts/qualification/test_sdpo_long_campaign.py
    bash -n scripts/qualification/run_sdpo_long_campaign.sh
    git diff --check

Stage only tested source plus frozen cohorts/config/hashes in a new capsule
/var/lib/posttrain/qualifications/sdpo-campaign-b32-250-20261018-r1.
Use SSH dstack@192.168.110.12, key
/home/hammad/projects/ai-infra/.state/ssh/dstack_worker, -F /dev/null,
BatchMode=yes,IdentitiesOnly=yes,IdentityAgent=none,GSSAPIAuthentication=no,
ConnectTimeout=8 and outertimeout25. Never print credentials. Check exact Docker
names, source hashes and real nvidia-smi compute occupancy before submissions.
When08has finished and GPU is idle, run the capsule driver launcher with qualify;
after exit0 and native/file qualification-gate.json, run it with train. Each
accepted container is immutable; Docker name/existing evidence prevent duplicates.

## Validation and Acceptance

Qualification must exit0 with exactly2saved moment steps and EMA updates1/2,
batch32with32distinct selected transitions at each update, policyversions0/1/2,
native tool-call/result/final-content gate, exact batch invariance through32 and
finite actor/rollout parity. Reconcile136unique episodes:72greedy diagnostics
(baseline and two checkpoints) plus64fresh online episodes. All task identities,
scores/errors, exclusions, argument/channel behavior and adapter receipts must
be coherent. Preserve failures, never weaken checks or automatically retry.

Full run expects8000online episodes (250*8*4) plus264greedy diagnostics
(11checkpoints*24), total8264native episodes and8000transition exposures. Check
actual counters instead of assuming planned totals. Audit scores/missingness,
guard identities and successful-action retention, repeated/empty tool calls,
length limits and exclusions; paired task-bootstrap intervals are training or
development diagnostics, not heldout proof. Native guard absence is not safety.

## Idempotence and Recovery

Never interrupt08or unrelated workloads, submit duplicates, alter accepted source,
mix attempts, publish local fixes, commit/push, or automatically resume from a
checkpoint. On failure retain Docker inspect/logs/native evidence and report the
cause. A new attempt requires evidence-backed repair and a separate identity.
The launcher refuses busy GPU and existing run/evidence. Keep unrelated edits.

## Artifacts and Notes

Durable receipts/config/CPU proofs live under
docs/research/evidence/sdpo-campaign-b32-20261018/. Remote immutable runtime
/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5 is mounted read
only; its pinned image is registry.lan/carbonteq/posttrain-lab/posttrain-lab/
posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337.
No external registry/publication or paid capacity is needed.

## Interfaces and Dependencies

Selected TRL loss source290c03d8e1553db509b0d033907f60fc1cb47b82,
SHA530b571c0001bbf85bb0c78725d32645a4d7155660e22e41580fab5c9f7041e4;
optimized vLLM0.29.1.dev4, aligned block128. New code remains project-only
qualification scripts, without reusable package imports from apps/lab or changes
to external forks, canonical product meaning, lockfiles or dependencies.

Revision2026-10-18: created for explicit user request for250updates/batch32 and
hold10/cooldown20/constant-floor schedule, superseding the hold on this specific
Qwen long-run experiment only.
