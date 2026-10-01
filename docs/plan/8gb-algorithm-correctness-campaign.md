# Audit and experimentally qualify training mathematics on the local 8GB GPU

This ExecPlan follows `docs/templates/PLAN.md`. Progress, Surprises & Discoveries,
Decision Log, and Outcomes & Retrospective are living records. The full user goal
is a broad correctness investigation, starting with SAMPO, using real Qwen 0.8B
and LFM 1.2B models and AutomationBench tasks to explain poor training outcomes.
Passing a smaller fixture does not complete this campaign.

User priority (2026-09-30): BF16 and FP16 are the primary supported execution
targets. FP32 model runs serve only as diagnostic references. Qualification
must cover actual scaled optimizer updates, not only unscaled derivatives;
retain existing operation-specific FP16 restrictions unless deliberately changed.

## Purpose / Big Picture

Establish which objective, gradient, data projection, and update invariants hold
in the current Posttrain workspace, reproduce failures, repair confirmed bugs,
and quantify their effect under controlled training. Keep correctness evidence
separate from recipe suitability and task quality. The canonical documents under
`docs/post-training/` remain authoritative; testing changes no frozen product
meaning. Any future semantic change requires its own recorded baseline amendment.

## Progress

- [x] (2026-10-01) Independently check LFM eager attention and unfused causal convolution outputs and reverse derivatives:54 paired BF16/FP16/FP32-diagnostic cases plus three native-head-dimension normalized controls. Six scalar finite differences agree within5.69e-11. All masks/cache-state/prefix-causality checks pass; convolution cached/full outputs match exactly. Stress attention gradients expose precision/saturation sensitivity, with FP32 rounded-input controls isolating it; normalized32-query/8-KV-head,64-dimension errors max0.335% BF16/0.0416% FP16. Do not claim trained-model or fused-kernel qualification; actual activation ranges and nonlinear block Jacobians remain open.

- [x] (2026-10-01) Collect four matched training-seed episodes after the fresh-group update: successes2/4→3/4, truncations3/4→2/4 and sampled tokens2775→2908. Seed6200 recovers the tool task; the other three generated-token paths remain identical. All four exact native/projection/independent episode/turn/token-credit audits pass. This is small training-sample recovery, not held-out generalization or a production throughput claim.

- [x] (2026-10-01) Connect native step-one LFM adapter collection to a fresh-process checkpointed FP16 update. Four observed episodes have rewards `[1,0,0,1]`, three truncations and exact token/projection/independent hierarchical-credit checks. Adapter equality is exact; optimizer advances1→2 and restored scaler512/tracker1→1024/tracker0. Four independent loss/score/mask checks,24 matrix gradients across four microbatches,192 scalar dots and AdamW(max2.12e-9) pass at2.54GiB peak Torch allocation. No first-forward policy clipping: fresh current/old ratios equal1. Full production admission/refill and other-family/precision continuity remain open.

- [x] (2026-10-01) Corrected native LFM step2 handoff matches scores within4.77e-7; fresh matched seeds still give1/4 success versus2/4 at step0, with all four new native token/mask/projection/credit audits passing. Publish scaler lifecycle tests atdc08945ddecf0693ea42bfb1bf0b11312aa2a0b4: eight tests pass, including exact CPU/CUDA scaler/AdamW restore after overflow and two-rank Gloo shared skip/backoff. Corrected update3 retains a positive policy derivative, applies successfully and passes72 matrix/288 scalar-dot checks across all three updates. Fresh corrected-step3 succeeds0/4 with four truncations and zero group credit; all native trace/projection audits pass. Fixed-token positive conditional likelihood ratio grows18.32→153.35 after updates2→3 while the geometric ratios are1.003932→1.006815. Preserve the adverse sample and distinguish length normalization, surrogate clipping and full trajectory likelihood.
- [x] (2026-10-01) Close single-rank native LFM FP16 checkpoint state omission: publish sourced0d7804795dc1254f7309916fce69898387a2a8a. Persist enabled scaler with optimizer/scheduler/RNG; reject missing enabled-scaler state before tensor restore; preserve model-only/BF16/disabled-scaler compatibility. The34-test checkpoint/cleanup/scaler slice passes. Actual LoRA-only/full-optimizer native save/load and fresh-process replay match next-update parameters, moments, scaler, scheduler and RNG exactly at deliberately nondefault scale512/tracker1. Omitted-binding control fails only scaler restore. Raw checkpoint/receipts remain external; runtime pins are unchanged.
- [ ] Complete fresh-collection/update continuity with native checkpointed optimizer/scaler state across fresh groups; verify other-family/BF16/full-weight/distributed native replay and production adoption. Single-rank LFM FP16 replay does not close these gates.

- [x] (2026-10-01) Native LoRA chain-rule audit exposed a timing-sensitive CPU-offload unscale race in Torch2.13: delayed scalar-only control multiplies instead of divides in3/4 plain CPU cases, while synchronous controls pass. Publish veRL `CPUOffloadShardedGradScaler` source269fde84d1769469f6b02b186170f420ec353d9e: five CPU/GPU regressions pass, native BF16/FP16 each apply2/2 updates,96 total matrix gradients match exactly and384 scalar dot checks pass. Correct the earlier unconditional FP16 qualification; post-fix fresh learning, distributed overflow/recovery and production adoption remain open. No runtime pin changes.

- [x] (2026-10-01) Reproduce LFM adapter3's held-out failure exactly; intermediate adapters1/2 already produce identical failing512-token outputs. Larger1,024-token budget preserves the first512 IDs/logprobs, exposes unknown-tool error at625 tokens and still fails after a second truncated response. Native trace audits pass. Matched temperature ratio differences stay below0.000151 with identical eight clipping classifications;32 independent softmax checks stay below2.17e-7. The first-update behavior change precedes reuse clipping; retain full fresh-loop, parameter-Jacobian and broader algorithm gates.

- [x] (2026-10-01) Export native LFM FP16 adapters at steps0–3, verify exact parameter handoff and isolate teacher-forced score differences to temperature precision: matched native arithmetic agrees within4.77e-7 across eight row/state checks. Fresh matched-seed AutomationBench collections give2/4 versus1/4 success and2/4 versus3/4 truncations before/after training. All eight native token/mask/projection/independent-credit audits pass. Preserve this adverse small sample and the open shared FP32-temperature/backward, intermediate-step, reproducibility and full fresh-update/refill gates; raw tools and exports remain external.

- [x] (2026-10-01) Extend native LFM collection to BF16/FP16 and a1024-token BF16 budget control. All six native episodes pass exact token/log-probability/mask/projection checks; preserve reasoning truncations and a completed invalid tool call. Use the unchanged mixed FP16 reward group for native FSDP2 CPU-offload/checkpointed updates at1018 prompt/872 response tokens: BF16 applies2/2, FP16(scale1024)3/3; ten independent loss/score checks and AdamW references pass. All sampled tokens clip on FP16 update3 while optimizer momentum still moves parameters. Fresh updated-policy generation/admission and cross-backend trajectory parity remain open.

- [x] (2026-10-01) Execute actual TRL/adaptive admission method bodies on observed equal-reward native group: both discard two rows despite104 nonzero discounted-credit tokens. Check298 native Qwen sampled tokens through actual TRL temperature scoring, direct softmax and30 independent BF16/FP32 selected-position checks. Cached BF16 replay matches every sampling score exactly; teacher-forcing discrepancy falls from weighted mean0.00713942 BF16 to1.947e-6 FP32. Keep sampling-correction ratios separate from optimizer policy ratios and native worker/refill learning open.

- [x] (2026-10-01) Collect fresh Qwen BF16 AutomationBench episodes through native Verifiers null harness/MCP and native training-client token transport. Scripted controls pass twice; real evaluation episodes expose duplicate-post reward blindness. Native training-client episodes project into Posttrain as133/165 sampled tokens and2/3 turns. Derive sparse-return SAMPO credit at discount0.95 versus1; preserve external adapter/serialization failures and keep full learning/admission acceptance open.

- [x] (2026-10-01) Measure native LFM partition accumulation with identical initial adapters: BF16/FP16 microbatch1 gradients exactly equal separate-row sums. Ten native contribution/layout/FP32 diagnostic arms apply updates and pass13 score checks. Batch2 versus microbatch1 changes total gradients29.06% BF16/6.51% FP16 versus0.00732% FP32; component cancellation conditions are43.65/42.51. Preserve measured Adam coordinate amplification and avoid inferring a dropped gradient, production cause or full independent model Jacobian. Tools/tensors remain external.

- [x] (2026-10-01) Exercise native LFM1.2B with FSDP2 CPUOffloadPolicy, preserving FP32 trainable masters and optimizer moments. Four BF16/FP16 arms at8/128 tokens apply8/8 updates and pass16 score/mask checks; FP16 retains scale1024 without skips. Independent native-gradient AdamW error stays below2.13e-9 and peak Torch GPU allocation is1.00GiB. Full parameter-derivative/accumulation, rendered LFM and fresh-task parity gates remain open. Tools/receipts remain external.

- [x] (2026-10-01) Repair GDPO weighted-aggregate overflow, distinct from component normalization. Four of nine finite-weight probes fail despite finite independent credit. Reject a common-scale candidate that erases a tiny residual after large-component cancellation; use precision1600 Decimal only on weighted-overflow fallback. Five new regressions fail against exact prior source and pass after repair;46 related tests,86 independent existing cases,9 new oracle cases, scoped Pyright/Ruff and9 boundary contracts pass. Retain extreme-coefficient scope and external-only runners/receipts.

- [x] (2026-10-01) Execute native veRL on controlled rendered multi-turn Qwen conversations with tool/header masks and unequal42/45-token responses. Four final BF16/FP16 precision arms pass16 independent loss/score checks and exact zero prompt/excluded/padding gradients. Three successful arms apply six updates; ordinary FP16 skips2/2. Reused-population clipping activates at10/27 sampled tokens for ordinary BF16 and17/27 for both FP32-region arms. Preserve measured probability differences and fresh-task/native initial-state parity gates. All tools and receipts remain external.

- [x] (2026-10-01) Run external native Qwen context extension: BF16 at128 tokens and corrected FP16(scale1024) at128/256 tokens apply six updates total; uncorrected FP16 at128 skips both attempts. All16 independent loss/score/context checks pass, applied AdamW errors stay below2.25e-9, and peak Torch allocation is5.05GiB. Keep runners/receipts in the external archive, not Git. Supplied-token results do not close the full rendered/fresh-rollout gates.

- [x] (2026-10-01 user correction) Remove experimental correctness runners and raw receipts from unpublished Posttrain commits. Preserve them byte-for-byte in `/home/hammad/experiments/posttrain-correctness/2026-10-01`, with a verified Git bundle and dirty-work snapshot. Keep written findings and actual production fixes/regressions in the repository; ignore known experimental paths to prevent recommitting them. Future experiments run from the external archive.

- [x] (2026-10-01) Separate GDN precision components: promotion-only and autocast-disable-only each skip2/2 native Qwen updates, while the combined control applies2/2. Add independent token-state recurrence, all six input derivatives and reference directional finite differences; FP32-region grids pass24/24 fixtures on each of Transformers5.16.1 and immutable upstream5.19. Autocast controls miss the FP32-accuracy gate; preserve these as precision measurements rather than treating normal half rounding as an algorithm bug.

- [x] (2026-10-01) Trace native Qwen FP16(scale1024) to the gated-delta output/norm boundary. Independent actual-input RMSNormGated derivative predicts67 FP16 overflows, matching67 native nonfinite elements. An explicit research-only FP32 fallback/norm region applies2/2 updates on both Transformers5.16.1 and immutable upstream5.19; all8 score checks pass, output-gradient hooks remain finite and AdamW errors stay below2.4e-9. Matched upstream baseline skips2/2. Preserve production, long-context, fused-kernel and cross-backend adoption gates.

- [x] (2026-10-01 user follow-up) Exercise native veRL model loading, single-rank FSDP1 accumulation, loss wrapper and AdamW on Qwen0.8B. Repair optional FlashAttention indexing dependency and avoid installing a legacy packed Qwen forward in the padded path; publish veRL8778c5d6e2ddd847d5098a24f4dc882f11ac57b4. Fifteen padding/forward regressions pass. Final BF16 and FP16(scale1) arms each apply two updates, pass four independent loss/score-gradient/context-mask checks, and match independent AdamW updates within2.3e-9. FP16 scales128/1024 remain overflow controls, not successes. Current and upstream Transformers both pass this small fixture at scale1; do not attribute that success to the upstream normalization repair.
- [ ] Broaden native veRL acceptance to the same full rendered traces and initial adapter/Adam states as TRL; add LFM/Gemma, packed/long context and distributed checks. Publish runtime assets and adopt pins only after those gates. The new native Qwen token probe closes the previously unexecuted-engine gap, not all backend equivalence or precision issues.

- [x] (2026-10-01) Close finer sampled-k3 transition cancellation in both forks: 268/3880 matched represented-input checks fail at the prior 0.05 boundary; tenth-order expansion through 0.25 passes the expanded grid. Publish TRL 5d4f9ad3c5f5d51b1ea50b827d82fdd379231dbf and veRL 10ad6babade57546139554a67bc8469118627748 without changing production pins. TRL focused tests pass92; veRL math/hierarchy slice passes87. Preserve the independent Decimal80 oracle and 1e-6 gate.
- [x] (2026-10-01) Extend matched source kernels through real model scores and scaled LoRA parameter gradients. Initial six arms (Qwen0.8B, LFM1.2B, tiny Gemma4; BF16/FP16) apply12 updates and pass24 microchecks. TRL applies updates; native veRL engine/distributed qualification remains open. Qwen FP16 scale1024 negative control applies0/2 updates; scale128 applies2/2. Final-source repeats are recorded in the matched-kernel evidence report.

- [x] (2026-10-01) Implement an unpublished veRL parity candidate in isolated `/home/hammad/projects/verl-posttrain-parity`, based on runtime SHA ef1c37715fa75de5973ae5b3c398383cd7e0093d. Preserve native GSPO; add opt-in `sampo_token_credit`, stable small-delta k3 arithmetic and early PPO-wrapper masking. 130 fork CPU checks and48 existing Posttrain parity checks pass. Original-wrapper negative control fails14/18 mask cases; candidate passes18/18. Half rounding exposed three further value failures near the series transition; extending the stable region to0.05 closes them. Posttrain mapping/pins, the analogous TRL transition sweep and actual model qualification remain open.

- [ ] (2026-09-30 user scope) Enforce TRL/veRL agreement using Posttrain selections as the semantic authority. Compare resolved settings, masks, credit, KL, clipping, reductions, sampler correction and update schedules on identical data, including numerical boundaries and shared real-model fixtures. A backend-native default must not silently change the selected Posttrain objective.
- [x] Refresh exact veRL identities: current runtime profile pins ef1c37715fa75de5973ae5b3c398383cd7e0093d/post8; existing parity fixtures explicitly select ce8e0430018204b03c009b72bfba3b58968696c7. Current TRL math source candidate is416b8978053d56bf4bc8674748c652f464999fac; production wheel adoption remains separate. The first attempted environment skipped; actual pinned-ce8 imports in the retained CPU parity environment run48 existing parity tests successfully.
- [x] Measure18 boundary cases against exact runtime-pinned veRL source and TRL candidate using Posttrain SAMPOSettings: ordinary case agrees,14 gates fail across capped large sequence ratios, excluded-score poisoning and near-zero KL. Retain independent scalar/Decimal checks; do not label ordinary parity as complete.

- [x] (2026-09-30 scope expansion) Add immutable Gemma4 tiny architecture fixture to direct and native harnesses:all14 native BF16/FP16 arms pass42 updates/84 microchecks; fifteen direct branches in both precisions retain SPPO-hard scaler/value gates. Complete144 attempts/143 applied/140 strict step passes including controls.
- [ ] Complete pretrained Gemma coverage. Gemma3-270M-it asset download returned gated401; authenticated access or a local checkpoint was requested. Gemma4-E2B full weights are10.25GB and cannot fit the local8GB device unquantized. Do not present tiny architecture fixtures as pretrained-scale qualification.

- [x] (2026-09-30) Extend the direct preference harness to FP16 base weights and dynamic scaling:270 baseline attempts across15 branches, both models and scales1024/128/1; BF16 controls pass12/12. Qwen skips135/135; LFM applies131/135. FP32 query/key normalization ablation restores Qwen45/45 applied updates; retain42/45 strict passes and three residual gates.
- [ ] Qualify and adopt the upstream query/key normalization correction through a reproducible runtime source/release boundary; repeat Qwen FP16 native policy runs and long/padded sequence controls. Research monkeypatches do not change production behavior.

- [x] (2026-09-30) Extend independent preference checks to all fifteen upstream DPO branches and four divergence transforms:306 matched cases,69 failures before/13 after; retain residual value misses. Complete66 BF16 SFT/preference updates plus24 matched alpha-near-one control updates on both models; repair confirmed cancellation, retaining29 passing focused/native trainer tests.
- [x] (2026-09-30) Extend native accumulated gradient checks to scalar GRPO/DAPO and upstream GSPO, Dr. GRPO, BNPO and LUSPO on both BF16 models and LFM FP16. Ordinary matrix:15/18 pass; Dr. GRPO precision misses and LUSPO scaler skip preserved. Check reward centering/std and initial denominators independently.
- [x] (2026-09-30) Complete 33 numerical arms including repeats/diagnostics:93 attempted /86 applied updates,25 pass /8 fail. FP64-loss and FP32-model controls localize sensitivity; Qwen FP16 GRPO passes at128, DAPO retains a strict miss. Trace-identified signed reward controls pass. No generic scale/default change.
- [ ] Qualify OLMo3 sampler/refill, full fresh generation/public recipe mapping, intended contexts/modules and remaining preference branches; preserve standard-loss Dr. GRPO/Qwen FP16 precision gates.
- [x] (2026-09-30) Complete three authorized research reviews of update schedules, LoRA rank/alpha/LR, and BF16/FP16 recipes, including primary ablations and paper/code discrepancies.
- [x] (2026-09-30) Repeat fresh Qwen simulated AutomationBench with stable KL source: all three BF16 score and parameter-gradient checks pass. This closes the earlier short-run discrepancy for these traces, not full runtime qualification.
- [x] (2026-09-30) Parameterize native fixture for one/two/three frozen-group updates; BF16 one/two controls pass on both models with matching first steps and zero excluded-token score gradients. Second-update clip averages are 29.17% Qwen /58.52% LFM.
- [x] (2026-09-30) LFM FP16 scale1024 one/two arms pass with no skipped updates, matching first-step controls and zero scaler-aware optimizer error; Qwen FP16 remains unqualified.
- [ ] Qualify fresh-rollout learning across seeds before selecting a production recipe; complete intended all-linear/context FP16 gates.
- [x] (2026-09-30) Extend native probes to actual framework FP16 precision helpers and dynamic scaling; rerun both BF16 controls. BF16 and LFM FP16 pass whole-update checks. Qwen FP16 default scale skips 3/3; scale 128 applies 3/3 but retains one strict independent-gradient miss. Preserve both failures.
- [x] (2026-09-30) Derive scaler-aware and exact-score optimizer references; verify Qwen's scale-128 optimizer implements the coded derivative exactly. FP64 loss-only diagnostic on an FP16 model passes the independent gradient gate. No production recipe or default was changed.
- [x] (2026-09-30) Reproduce 22/28 near-zero KL value/gradient failures with Decimal80; correct cancellation without changing the estimator, and publish TRL source `9f0825046ae3509a6be804d74a93fb89d5dc695e`. All 28 after cases, 82 focused tests, and 248 prior policy cases pass.
- [x] (2026-09-30) Execute real Trainer loops with supplied multi-turn traces on both models in BF16 and FP32. Twelve after-correction updates and 24 derivative checks pass; three baseline BF16 misses disappear. Fresh native collection and full Trainer recovery remain open.
- [x] (2026-09-30) Derive 86 independent GDPO/CAPO cases; correct extreme finite-value normalization overflow while retaining ordinary credit semantics.
- [x] (2026-09-30) Execute twelve actual-model GDPO/CAPO updates with exact parameter-gradient and accumulation agreement; four direct adapter/Adam save-and-replay cases pass. Native Trainer recovery remains open.
- [x] (2026-09-30) Execute 60 independent DPO/SFT-branch cases and twelve final real-model SFT/DPO updates; reproduce and correct half-probability and near-certainty gradient errors in TRL.
- [x] (2026-09-30) Publish DPO source candidate `18e89c58bee70d25f1231cbe0dc4a865540d6fe5`; final 10 focused and 21 broader trainer cases pass. Wheel/runtime gates remain open.
- [x] (2026-09-30) Repair LFM renderer SFT assistant-header targets in published source candidate `1aafe24595a7f2d2f31d24afb4b1bb7a6c6dd076`; 59 renderer and 15 framework cases pass. Native rollout masks and wheel/runtime qualification remain open.
- [x] (2026-09-30) Extend independent policy loss/gradient grid to 248 cases; reproduce six excluded-behavior-score failures and repair ratio arithmetic in TRL.
- [x] (2026-09-30) Derive 66 independent distillation cases, repair FP16/BF16 sampled IW-OPD probability precision, and execute nine real-model updates in fresh groups of two.
- [x] (2026-09-30) Push source candidate `780bdd3edea61a993151c7dc1f02336b74a399d9`; retain wheel/production qualification as open work.
- [x] (2026-09-30) Revalidate dirty worktree, active goal, GPU availability, model caches, and executable TRL pin.
- [ ] Build independent loss/gradient oracles and a resumable case registry, starting with SAMPO.
  - SAMPO short-run Python float64 value/analytic-gradient oracle is implemented; broader registry and algorithms remain.
- [ ] Execute real-model SAMPO cases on Qwen3.5-0.8B and LFM2.5-1.2B.
  - Initial direct-loss/scoring slice executed for both models in FP16 and BF16; full matrix remains pending.
- [ ] Compare pre-correction and corrected SAMPO updates, retaining exact initialization and inputs.
- [ ] Check sampled action masks, causal label shifts, chat headers, EOS, truncation, padding, and temperature.
- [ ] Check accumulation, unequal lengths, clipping thresholds/signs, zero credit, KL value/gradient, sampler correction, and finite precision.
- [ ] Check LoRA rank/alpha scaling, frozen base/reference identity, optimizer state, gradient checkpointing, and save/resume equivalence.
- [ ] Extend to GRPO, GSPO, DAPO, OLMo 3, GDPO, CAPO, SFT, DPO, and on-policy distillation; include exposed alternative loss branches separately.
- [ ] Reconcile TRL/veRL mathematical parity and distributed/fused limitations without using dependency skips as evidence of success.
- [ ] Execute bounded AutomationBench collection and controlled task-learning comparisons on both models.
  - Three-round groups of two executed for Qwen and LFM using real simulated task builders/tools/assertions in a direct-loss harness. Native Verifiers/full Trainer and multi-turn comparisons remain.
- [ ] Repair each confirmed issue in its owning repository, validate, publish any required fork, and update immutable consumers.
- [ ] Produce detailed progress/results analysis with explicit coverage, failed invariants, uncertainty, and practical recommendations.
- [ ] Audit the full original objective against authoritative artifacts before marking the goal complete.

## Surprises & Discoveries

Saturated attention can show100% relative query-gradient error while the
reference gradient norm is only2.15e-8. A different stress fixture has a
material0.429 BF16 coordinate error; both magnitude and conditioning must be
reported. Shape-correct unit-weight Q/K normalization reduces the measured
gradient error to0.335% BF16 and0.0416% FP16. Actual trained activations remain
unmeasured, so stress arithmetic is not a diagnosis of current-run failure.

A successful AutomationBench episode can truncate after the correct tool call:
the fresh step-one group has two successes but three truncated episodes.
Reward, completion and sampled-token work must therefore be reported separately.
Fresh-data ratios reset to1 even when restoring nonzero AdamW moment history;
this distinguishes optimizer continuity from old-policy reuse.

The independent linear-gradient audit catches an upstream scaler ordering race
that score-loss and conditional AdamW oracles miss. An inverse-scale scalar
read synchronizes CUDA and hides it. The delayed ordinary-CPU control removes
model/DTensor semantics from the reproduction. First-step native adapters are
identical before/after the fix; later states differ, so earlier held-out behavior
must remain tied to its exact exported state rather than a broad FP16 verdict.

The LFM held-out behavior change appears after the first unclipped training
update and repeats exactly. More rollout budget reveals an invalid tool name
without rescuing the task. Absolute temperature-related score differences do
not change the measured directional clipping classifications on this fixture.
These observations constrain the explanation; they do not establish general
reward regression or a defect in the SAMPO equations.

- (2026-10-01) LFM's zero executed-tool failures can hide invalid or unfinished calls: all three collection arms retain actual reward/truncation separately from harness completion. A real mixed task population activates clipping under frozen reuse; the third FP16 update has zero policy score gradient on all1253 sampled tokens yet Adam's stored state still produces a verified step. This is compatible with the optimizer equations, not evidence that clipping failed.

- (2026-10-01) A fixed Qwen model has nontrivial BF16 cached-versus-teacher-forced score differences without any update. Exact cached replay rules out lost sampling log probabilities on this fixture; FP32 greatly reduces the execution gap. Matching temperature does not remove half-precision path differences. Observed sampler/trainer ratios must not be reported as PPO update clipping.

- (2026-10-01) Real native task success can hide duplicate actions: the office-closure assertion rewards three identical posts and one post equally. Equal terminal rewards have zero episode credit, while discount0.95 still produces length-dependent initial turn credit; terminal-reward variance admission would reject that group. Native evaluation traces lack token evidence, so the training client is a separate acceptance gate. Default Verifiers JSON rounds policy floats; Posttrain already explicitly retains full precision.

- (2026-10-01) Native accumulation can be exactly correct while batch-layout precision differences substantially alter a cancelling gradient. The half-precision row sum is exact; changing layout yields small errors relative to component norms but large errors relative to their remaining signal. FP32 reduces that sensitivity. Near-zero coordinates around Adam epsilon can also produce large coordinate-update differences despite a small full-gradient error.

- (2026-10-01) FSDP1 actor CPUOffload is explicitly disabled because of accumulation limitations; FSDP2 CPUOffloadPolicy provides a separate executable native LFM path. Its8/128-token BF16/FP16 probes preserve FP32 CPU masters/moments and use about1GiB peak Torch GPU allocation. Passing AdamW arithmetic does not independently prove accumulation through every model parameter.

- (2026-10-01) Stable component normalization does not prevent the subsequent GDPO weighted sum from overflowing. Simple global rescaling is insufficient when equal large opposing signals cancel and a tiny third component determines the credit. Preserve represented component values and original epsilon in an exceptional high-precision aggregate/whitening path; do not extrapolate this extreme-weight issue to current-run learning failures.

- (2026-10-01) Clipping activates in the native engine on the second update of a frozen rendered population, with opposing token credit selecting the appropriate clipped branch. Corrected BF16/FP16 both clip17/27 sampled tokens, but their probability ratios differ; ordinary BF16 clips10/27. Passing objective derivatives does not prove identical parameter trajectories or choose an optimal recipe. Prompt/tool/header/padding score gradients are exactly zero after native jagged conversion.

- (2026-10-01) At128 tokens, uncorrected FP16 still passes every loss/score check while applying no updates. Masked objective correctness and native scaled optimizer execution are distinct acceptance requirements. The combined precision control continues to apply updates at128 and256 tokens within the local GPU budget.

- (2026-10-01) A tensor's FP32 storage does not guarantee FP32 recurrence compute under CUDA autocast. Promotion-only removes the norm overflow but leaves internal delta-rule backward overflow. Both precision safeguards are needed on the measured native fixture. Independent recurrent equations validate chunk outputs, final states and q/k/v/decay/beta/initial-state derivatives before half-boundary casts.

- (2026-10-01) The short native Qwen failure is a gradient range problem inside gated normalization even after upstream q/k promotion. Its true scaled gradient is about1.095million, exceeding FP16's65504 maximum. FP32 fallback output plus autocast-disabled core avoids all traced nonfinite gradients at scale1024. This control does not separately ablate those two precision changes; retain that distinction from the earlier zero-vector normalization bug.

- (2026-10-01) Native veRL initialization succeeds but padded execution initially fails without flash_attn, then with a legacy Qwen forward referencing missing causal_conv1d_fn attributes in Transformers5.16.1. Reusing existing Torch padding helpers and retaining native forwards in the padded path closes both failures. A full upstream Transformers d6c1e71/5.19.dev runtime does not close high-scale overflows on the eight-token fixture; current and upstream runtimes both apply updates at scale1. This short fixture does not isolate the independently established zero/small-vector normalization defect.

- (2026-10-01) Matching backend kernels shared residual cancellation just outside the 0.05 series boundary: 268/3880 strict typed-input checks fail despite parity. A tenth-order interval through 0.25 passes the expanded oracle sweep. Qwen FP16 scaling also separates applied updates from attempted updates: scale1024 skips both, scale128 applies both on the controlled traces. Exact scaled backward is required to distinguish rounding from a loss derivative error.

2026-09-30 Qwen FP16: lowering scale to1 still skips all45 preference/SFT
attempts. Fallback l2norm computes rsqrt in half before promoting q/k; the
zero-vector derivative should be finite (1000 times the cotangent) but is NaN.
Independent48-case scalar Jacobian grid finds8 baseline FP16 failures and none
with FP32 work; BF16 controls pass. Module traces first become nonfinite at
layer22 linear-attention qkv projection; FP32-norm controls eliminate all
nonfinite traced module outputs and apply all45 Qwen updates. Upstream main
d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a already promotes q/k before normalizing.
The locked Transformers5.14.1 wheel source retains the faulty order, as does
the research5.16.1 runtime. This is model fallback math, not a universal recipe
scale recommendation. AOT's remaining0.15% gate miss persists with FD h1e-6;
actual raw logits match the rounded double-probability control while the
FP32 probability oracle differs by one FP16 subnormal increment.

2026-09-30 f-DPO extension: subtracting one from exp near alpha=1 produces
1.9–2.3% actual BF16 parameter-gradient discrepancies on both requested models.
Using expm1 closes all six matched repaired preference steps. This is a
non-default DPO transform and does not establish the cause of SAMPO learning
failure. Remaining13 grid misses and four expanded model-step misses are
strict absolute loss-value failures with matching parameter gradients.

2026-09-30 native reduction matrix: independent248-case kernel checks remain
green while real backward finds Qwen BF16 Dr. GRPO accumulated error1.1529%
and LFM FP16 Dr. GRPO errors0.10747%/0.04194%. Exact-score controls match;
loss-only FP64 closes these cases. Direction controls retain nearly parallel
gradients (worst cosine.99993367), limiting any claim about poor learning.
LFM FP16 LUSPO skips at1024 but passes at512/128: its length-weighted loss
produces much larger backward magnitude before norm clipping. Qwen FP16 GRPO
passes at128 while DAPO retains a strict miss, closed only by FP64-loss diagnostic.
These are recorded numerical boundaries, not reasons to declare a blanket fix.

2026-09-30 recipe research: author SAMPO uses turn rows and steps inside
optimizer-minibatch loops; one epoch can mean several policy updates. Our
full-episode accumulation is a different schedule and reduction. SAMPO
paper/script clipping and normalization disagree, and Precision-RL's paper
rank32 differs from its public script rank1. Neither is an exact recipe to copy.
Holding alpha/r constant also does not normalize Adam updates across ranks;
the proposed rank rule depends on initialization and alpha convention.

2026-09-30 supported precision: Qwen FP16 unscaled gradient checks pass while
actual scale-1024 backward overflows and skips every update. Scale 128 permits
all three updates, but the independent scaled parameter reference misses by
0.8168% at one step. Exact-score/scaler VJP reproduces the real optimizer
gradient exactly. A loss-only FP64 diagnostic closes the independent miss,
supporting loss-rounding sensitivity through FP16 backward. Unscaled FP16
VJP is not by itself the right scaled optimizer oracle: scaling changes
underflow survival. LFM FP16 passes at 1024; both BF16 optimizer controls pass.

2026-09-30 native Trainer/KL: the first supplied-trace arms expose three BF16
parameter-gradient misses (0.61–1.36%) despite score-derivative errors <=7.45e-9.
Exact-score VJP controls match; FP32 parameter references pass. An independent
KL-only Decimal80 grid finds cancellation even in expm1(d)-d: tiny terms and
derivatives become zero. A differentiable sixth-order series near zero fixes
all 28 cases and eliminates the three BF16 misses in matched native loops.
The reused frozen group reaches real clipping. These are controlled traces,
not fresh environment collection or proof of improved learning.

2026-09-30 structured rewards: all 80 ordinary interleaved-population cases
already pass; extreme finite offsets/norms and a large epsilon expose five
numerical failures. Rescaling values and epsilon together restores all 86
cases. The first actual-model accumulation arm was invalid because the harness
used evaluation mode; training mode restores exact full/microbatch gradients.
Fresh behavior scores give ratios one and constant loss despite real updates;
post-update geometric mean ratios reach 1.214 on Qwen CAPO and 1.203 on LFM
GDPO. These fixed synthetic-credit fixtures do not demonstrate task learning.

2026-09-30 LFM masking repair: the dev2 renderer's empty sampled mask made SFT
fall back to all assistant-attributed tokens, including injected headers and
separators. Exact generation-prompt boundaries now define sampled spans through
the turn stop, preserving input IDs. LFM2.6B prefills `<think>` but 1.2B Thinking
samples it; the initial correction failed eight existing 2.6B tests, then the
family-aware boundary repair restored all 59 focused tests. Twenty new cases
fail on the old wheel, and 15 Posttrain integration cases pass. Matched LFM
SFT targets change 7 to 3 tokens; per-token losses consequently cannot be
interpreted as an improvement comparison. Real gradients still agree, and
the DPO control retains identical update results. Native rollout extraction
and a new wheel/pin remain separate gates.

2026-09-30 preference/SFT follow-on: all 40 half-precision kernel cases fail
before promotion; all 60 cases pass after repair. Qwen DPO still differs by
about 2% in parameter gradients after promotion alone. The first discrepancy
is an approximately 8e-8 raw-logit gradient error; a near-certain-token
fixture reproduces cancellation in selected-logit minus logsumexp. Stable
row-wise log-softmax is closer to double-probability gradients and restores
all three model update comparisons. LFM and Qwen SFT direct-loss math passes,
but pinned LFM renderer SFT masks include injected assistant headers in three
multi-turn selection fixtures. System/user/tool body content remains excluded.
This is a separate open renderers-fork repair, not a reason to declare SFT
qualified based on decreasing loss. Details in `preference-and-supervised-results.md`.

2026-09-30 follow-on: excluded old-policy scores can poison importance ratios
before final masking, independently of the already-corrected KL guard. Six
independent cases failed; neutralizing excluded log ratios before exponentiation
restores all 248 cases. Unequal-length BNPO differs under microbatch-local
normalization by design, unlike the tested globally normalized accumulation paths.
Sampled IW-OPD half log-softmax loses precision before loss casting; two cases
failed and both regressions fail on the original post13 wheel. Promoting half
logits restores 66 independent checks. The sampled surrogate can be negative;
the trainer smoke-test positivity assertion was corrected to finiteness.
Nine actual-model distillation updates match independent parameter-gradient
credit exactly. LFM completes 0/6 at 32 tokens and 5/6 at 256, so a short budget
can make numerically correct updates useless for final-answer behavior.
See `docs/research/evidence/correctness-matrix/loss-kernels-and-distillation.md`.

2026-09-30 first real-model slice: LFM FP16 and both BF16 models retain
opposing sampled-logp gradients [-0.5, 0, 0.5] and finite LoRA movement.
Qwen FP16 has correct sampled-logp gradients but nonfinite parameter gradients
in the direct-loss harness. This bypasses full Trainer precision handling;
localize the first nonfinite operation before claiming a production bug.
BF16 scoring differs from FP32 log-softmax of the same logits by 0.017236
for Qwen and 0.044215 for LFM. Both full and chunked paths agree in this
discrepancy. Source keeps BF16 logits through temperature/log-softmax;
distinguish expected quantization from unacceptable ratio/clip error with
matched old/new forward experiments. Do not loosen the declared tolerance
after observing failure.

The two-action KL derivation reproduces 0.400 and 0.1977502119602598 with
identical value 0.3680642071684971. This is before beta and at fixed context,
not a measured adapter gradient. Legacy flag provenance is compatibility
preservation, not demonstrated task quality. Upstream PR6503 explicitly
qualifies unbiasedness to token importance sampling; no automatic SAMPO
recipe flip is justified by that statement alone.

The local RTX 3070 Ti has 8GB total; desktop processes consume approximately
1.4GB, leaving 6.4GB. Do not assume the full nominal VRAM is available.
Cached complete weights include Qwen3.5-0.8B at
`2fc06364715b967f1860aea9cf38778875588b17`, LFM2.5-1.2B-Thinking at
`95053d21d8e0b7ca99421a2127ae39c64f685ff3`, and LFM2.5-1.2B-Instruct at
`0f604ada3f766f9f257460c4c9f0b5d6f69d431b`. The first two are catalog selections.
Qwen's cache is a conditional-generation model; text-only extraction must retain
the correct language backbone and LM head rather than guess architecture names.

Previous investigation published TRL post13, release source
`d97a2cf619f94c7076a720116d75f85dfd62382b`, fixing token-local sequence credit,
masked/tiny KL arithmetic, and refill sampled-token denominators. Those fixes
have synthetic mathematical evidence, not broad model/task qualification.
Posttrain currently selects that candidate locally. Existing run images retain
their previous immutable runtimes. The prior work remains dirty and must be
preserved, including unrelated `.claude/` and `.release/` files.

## Decision Log

- Decision: use explicit scalar grouped-attention and causal-convolution equations for expected derivatives, validate the scalar reference with finite differences, retain paired rounded-input controls and report absolute as well as relative errors. Check native LFM head dimensions and normalization separately; do not modify production precision based on artificial saturated logits. Date/Author:2026-10-01/Codex.

- Decision: continue from the native checkpoint with fresh observed task groups, retain the original reward and truncation evidence, and audit all four microbatches using the restored optimizer/scaler. Keep collection/updating orchestration external and do not equate this bounded continuity probe with the production worker loop or general task improvement. Date/Author:2026-10-01/Codex.

- Decision: bind the FSDP engine's optional GradScaler to `FSDPCheckpointManager` and persist it in per-rank extra state. Validate required FP16 scaler state before loading model/optimizer; old checkpoints remain usable for explicitly model-only loading, while full FP16 restore fails clearly if scaler state is absent. This preserves the existing exact-recovery meaning and changes no frozen baseline. Source edits belong in `/home/hammad/projects/verl-posttrain-parity` at `verl/utils/checkpoint/fsdp_checkpoint_manager.py`, `verl/workers/engine/fsdp/transformer_impl.py`, checkpoint CPU regressions and the fork ledger. Commit/push the fork before consumer documentation; no pins change. Validate focused checkpoint tests plus an external actual LFM native save/load/replay with LoRA-only model state, full optimizer/extra state and a deliberately nondefault loss scale. Require exact restored scaler state and next-update parameter/moment equality; retain negative control and GPU/multi-rank limitations. Date/Author:2026-10-01/Codex.

- Decision: stage inverse-scale and overflow scalars synchronously on CPU only when gradients have CPU storage, in veRL's FSDP engine scaler subclass. Rationale: PyTorch's nonblocking CUDA-to-CPU scalar copies can be consumed by a host foreach kernel before completion; an inverse-scale Python read hides the failure. Preserve native device-only scaling and distributed overflow reduction. This changes no frozen product meaning. Edit `verl/utils/sharded_grad_scaler.py`, `verl/workers/engine/fsdp/transformer_impl.py`, `tests/utils/test_sharded_grad_scaler.py` and `CARBONTEQ_FORK.md` in `/home/hammad/projects/verl-posttrain-parity` (base8778c5d6e2ddd847d5098a24f4dc882f11ac57b4); then commit/push that fork, then commit Posttrain's consumer/evidence/plan documentation. Keep production pins, lockfiles and unrelated dirty work untouched. Run the three new CPU/GPU regressions using `/tmp/posttrain-native-verl-update-runtime/bin/python -m pytest -c /dev/null -p no:cacheprovider tests/utils/test_sharded_grad_scaler.py -q` with the fork on `PYTHONPATH`; run external native collected-LFM chain probes serially in FP16/BF16. Require manual linear gradients to match unscaled preclip native gradients, finite applied updates, score/AdamW oracles and external hash receipts. Restore the previous fork commit to discard the candidate; raw failures remain external. Date/Author:2026-10-01/Codex.

- Decision: preserve native observed LFM rewards and token spans, compute framework SAMPO credit without relabeling outcomes, and include the selected per-token sampler correction before testing two/three frozen-population native updates. Use beta0 matching the default for this slice; make no new KL claim. Separate collection and veRL research dependency environments after a confirmed ANTLR grammar conflict. Date/Author: 2026-10-01 / Codex.

- Decision: measure score discrepancies using identical native token paths and an independent cached replay before attributing them to temperature, optimizer movement or a loss bug. Preserve FP32 solely as a diagnostic and treat equal-reward discounted-credit rejection as a measured recipe tradeoff. Keep runners/receipts external and retain production inference and native lifecycle gaps. Date/Author: 2026-10-01 / Codex.

- Decision: exercise the pinned native chat harness and MCP task tools locally with an external HF token provider. Preserve native traces and exact sampling evidence, distinguish evaluation transport from training transport, and retain duplicate-action reward limitations without changing the environment's reward contract. No frozen product amendment or production pin change is required for these experiments. Date/Author: 2026-10-01 / Codex.

- Decision: use Decimal precision1600 only after GDPO weighted aggregation overflows, preserving the ordinary float path and component normalization semantics. Commit the generic framework repair and meaningful regressions, while keeping exploratory candidates, plugins and receipts external. This changes no frozen product meaning or training recipe. Date/Author: 2026-10-01 / Codex.

- Decision: correctness runners and raw traces are local research material, not product source. Consolidate unpublished root commits to remove them from the publishable history, while preserving the original history and receipts externally. Retain source repairs, regression tests and concise reports. This follows the user's explicit request. Date/Author: 2026-10-01 / Codex.

- Decision: require an independent state-equation oracle and split precision ablations before treating the combined Qwen control as understood. Scope the derivative grid to its FP32 region and retain native scaled-update evidence for boundary behavior. Do not infer a math defect solely from ordinary half rounding against an FP32 gate. Date/Author: 2026-10-01 / Codex.

- Decision: retain an opt-in research-only FP32 GDN region as a measured correction candidate; do not silently adopt it as a production monkeypatch or change the loss-scale recipe. Keep the failed baseline, analytical derivative and native optimizer receipts alongside the successful controls. Broaden rendered/context/backend gates before publishing runtime adoption. Date/Author: 2026-10-01 / Codex.

- Decision: preserve native Qwen forwards when remove-padding, sequence-parallel and fused paths are all disabled; retain the specialized packed path and vision fixes. Use real single-rank FSDP1 and native AdamW, with independent score and optimizer oracles, rather than calling a kernel-only comparison native-engine qualification. Record high-scale overflows and production/packed-path adoption separately. Date/Author: 2026-10-01 / Codex.

- Decision: widen sampled-k3 stable arithmetic with a tenth-order polynomial through absolute delta0.25 in both forks; retain the estimator and derivative convention. Require immutable-source Decimal80 boundary checks plus final-source model repeats. Record scaled and unscaled FP16 references separately. This is source qualification, not production adoption or a recipe decision. Date/Author: 2026-10-01 / Codex.

- 2026-09-30 next cross-backend repair ownership: isolate a veRL fork worktree from ef1c37715fa75de5973ae5b3c398383cd7e0093d. Generic masked ratio and stable sampled-k3 fixes belong in `verl/trainer/ppo/core_algos.py` and masked KL admission in `verl/workers/utils/losses.py`, with CPU regressions and `CARBONTEQ_FORK.md`. Preserve native GSPO's intentionally capped recipe; provide a distinct uncapped token-credit sequence-ratio loss for the Posttrain SAMPO contract if required, then map it in the Posttrain worker with an explicit fork compatibility gate. Do not silently redefine GSPO or adopt veRL's cap as a Posttrain default. Publish the fork before pins, and stage only owned root mapping changes after accounting for existing dirty work.

- 2026-09-30: The user names Posttrain as the normalizer between veRL and TRL. Interpret this as algorithm settings and admitted logical data defining semantics; backend configuration names are translations. Preserve settings meaning rather than choosing one backend's defaults. Posttrain owns shared resolution, mapping, parity fixtures, manifests and evidence; generic kernel repairs belong in isolated maintained backend forks with ledger/consumer updates and fork publication before pins. Existing dirty mappings/tests are prior work and must remain unstaged unless separately owned. This changes no frozen product meaning.

- 2026-09-30: User explicitly adds Gemma architectures to the ongoing campaign. Root research harnesses gain a shared model/renderer/factory selection seam. Keep existing Qwen/LFM fixture IDs and settings intact. Gemma4 loads through the existing multimodal auto factory and targets language-model q/v LoRA only; unused audio/vision parameters must not enter text gradient comparisons. No production catalog/support change is intended. Pretrained-scale coverage requires a fitting accessible immutable checkpoint or separately qualified memory strategy.

- 2026-09-30: FP16 preference experiments are research canaries, not an expansion of framework operation support. The root harness owns dtype/scaler controls and evidence; no production adapter, fork or pin change is needed. Compare gradients after scaling the independent score coefficients through the actual model Jacobian and dividing FP32 adapter gradients by the same scale. Record exact coded-logit VJP and scaler backoff separately to distinguish loss arithmetic from finite-range overflow.

- Decision: check all DPO branches with Python probability calculations and finite differences, then repair only independently reproduced divergence arithmetic problems in the isolated TRL fork. Existing sigmoid framework defaults stay fixed.
  Ownership/order: Posttrain owns `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/preference_loss_grid.py`, real-model harnesses, evidence and consumer documentation; isolated `/home/hammad/projects/trl-sampo-local-credit` owns reusable DPO arithmetic, focused tests and `CARBONTEQ_FORK.md`. Start from clean published source9f0825046ae3509a6be804d74a93fb89d5dc695e, preserve primary TRL checkout, validate and commit/push the fork before recording a reproducible candidate here. No wheel/pin or frozen baseline change is intended.
  Validation: original five-branch grid remains a control; all fifteen branches, beta/length-discount variations, FP16/BF16 probability inputs and FP64 diagnostics, allowed divergence combinations and targeted small-ratio regressions. Preserve precision misses rather than silently widening tolerances.
  Date/Author: 2026-09-30 / Codex.
- Decision: extend the supplied-trace native runner with an explicit objective flag and reuse the independent Python policy oracle, including the configured Dr. GRPO maximum-length denominator.
  Rationale: prior loss grids do not prove real model gradients or optimizer accumulation. Scalar outcome credit is separate from SAMPO's opposing turn credit. Additional upstream loss branches are research coverage, not new framework operations. No frozen product meaning changes.
  Validation: three updates, six microbatch checks per arm, independent normalized [0,1] reward check, sampled-only masking and scaler-aware optimizer references. Use both BF16 model families and the currently passing LFM FP16 target; retain any failures instead of relaxing tolerance.
  Date/Author: 2026-09-30 / Codex.
- Decision: prefer a two-update frozen-group SAMPO candidate against one update, changing only schedule first. Keep ordinary LoRA r4/alpha8, existing KL convention and repaired credit/masks; treat BF16 as control and FP16 as a separate correctness gate.
  Rationale: primary research justifies bounded reuse and precision experiments but does not select our optimal recipe. Minibatching, turn packing, adapter geometry and loss scaling require separate controlled arms. No frozen baseline or production selection changes in this research slice.
  Ownership: authorized subagents own three research documents under `docs/research/evidence/correctness-matrix/`; parent owns runner, synthesis and evidence.
  Date/Author: 2026-09-30 / Codex.
- Decision: center remaining qualification on BF16/FP16 and record applied updates, scaler skips, and whole-optimizer gradient agreement separately from microbatch loss checks.
  Ownership: research runner and evidence here, using existing framework precision helpers. Preserve operation-specific support limits. Use FP32 models and FP64 loss only for diagnostic controls.
  Validation: six supported/diagnostic arms, actual callbacks after clipping/unscaling, independent scaled references, exact-score controls, and 21 precision tests in the ML runtime. Keep Qwen FP16 standard-loss strict failure open; do not alter production defaults based on one short fixture.
  Date/Author: 2026-09-30 / Codex, reflecting direct user steering.
- Decision: use a stable k3 series for |reference-policy logprob difference| <= 0.01; preserve existing estimator and bias-correction configuration.
  Ownership: TRL `trl/trainer/grpo_trainer.py`, focused regressions, and `CARBONTEQ_FORK.md`; Posttrain owns independent harnesses, evidence, and consumer qualification notes. No frozen baseline amendment or recipe change.
  Order: prove baseline failure, validate source plus real Trainer controls, commit/push fork source, then update consumer evidence. Wheel publication and immutable pins remain pending. Do not stage preexisting dirty consumer changes with this slice.
  Date/Author: 2026-09-30 / Codex.
- Decision: repair only reproduced normalization overflow in the framework reward module; retain nearby-origin statistics on ordinary inputs and scale epsilon consistently in the overflow fallback.
  Ownership: `packages/train/src/posttrain/train/reward_advantages.py`, its regression tests, and independent research harness/evidence here. No fork change or frozen product baseline amendment is required.
  Validation: Decimal100 grid, actual model gradient and accumulation controls, direct save/replay, 52 reward tests, scoped Ruff, and import contracts. Leave native Trainer and runtime pin gates open.
  Date/Author: 2026-09-30 / Codex.
- Decision: implement LFM sampled masks in isolated `/home/hammad/projects/renderers-lfm-mask`, branch `codex/lfm-sampled-mask`, based on renderer ledger commit `d1458bf1a665278b05ac6e9ed0611abc78f953a8` (behavior matches pinned release `6f712616fa88073919827a695af4f318b8c2d24e`).
  Ownership: only `renderers/catalog_models.py:LFM25Renderer.render`, focused mask regressions, and `CARBONTEQ_FORK.md`; update Posttrain consumer/evidence here. Preserve other renderer families and the primary checkout.
  Contract: derive sampled assistant spans from the exact generation-prompt prefix, include the emitted turn stop, exclude injected headers, separators and appended generation prompts; reject inconsistent template prefix boundaries rather than silently train scaffolding. No frozen product meaning changes.
  Validation/order: prove old pinned failure, execute actual cached tokenizers and focused renderer regressions, repeat real LFM SFT updates, commit/push fork source, and leave release/pin changes pending the remaining qualification gates.
  Date/Author: 2026-09-30 / Codex.
- Decision: repair DPO half scoring and stable selected-probability gradients in the TRL fork; leave the reproduced LFM sampled-mask failure explicitly open for its own renderers-fork change.
  Rationale: independent probability/finite-difference references and actual-model backward controls reproduce both DPO numerical failures. Renderer semantics must be fixed at their source, without hiding headers in the train adapter.
  Ownership/order: validate and publish TRL source first. For the renderer repair, resolve an isolated renderer worktree from the pinned release, update renderer ledger plus `docs/tooling/renderers/README.md`, validate actual Qwen/LFM masks and existing renderer regressions, then commit/push before release/pin updates. No product baseline change is intended.
  Date/Author: 2026-09-30 / Codex.
- Decision: repair masked importance-ratio arithmetic and sampled IW-OPD half probability math in the owning TRL fork; keep gamma and KL estimator settings unchanged.
  Rationale: independently reproduced numerical failures preserve their existing exact-arithmetic contracts when corrected. Source candidate is published, but no wheel or production pin is changed.
  Validation: 248 policy cases, 66 distillation cases, 30 SAMPO/precision regressions, 73 IW-OPD tests with one optional Liger skip, two old-wheel negative controls, and nine finite real-model updates. Full integration gates remain open.
  Date/Author: 2026-09-30 / Codex.
- Decision: add three-round, group-of-two simulated AutomationBench experiments and a frozen-group replay arm.
  Rationale: fresh single-use groups cannot demonstrate clipping after an update; bounded actual tool behavior reveals zero-spread and truncation failures. Replay is a diagnostic recipe, not a production schedule change.
  Date/Author: 2026-09-30 / Codex.
- Decision: repair BF16 score/loss precision in the TRL fork as an unpublished candidate, without flipping the KL estimator flag.
  Rationale: promoting half logits before temperature/log-softmax restores the FP32 probability calculation; BF16 spacing can exceed the configured 0.003/0.004 clipping band. These corrections preserve exact-arithmetic objective meaning.
  Ownership/order: edit and validate `/home/hammad/projects/trl-sampo-local-credit` GRPO/RLOO helpers and regression tests, update its ledger and this consumer page, commit/push the fork, then publish and update immutable Posttrain pins only after qualification. Do not silently repoint to a dirty checkout.
  Validation: fork `PYTHONPATH=/home/hammad/projects/trl-sampo-local-credit:/tmp/trl-math-peft /home/hammad/projects/trl-gdpo-capo/.venv/bin/python -m pytest tests/test_grpo_float16_loss.py tests/test_sampo_precomputed_advantages.py -q`; real-model candidate scoring and three-step harness commands below.
  Date/Author: 2026-09-30 / Codex.

- Decision: use cached catalog Qwen3.5-0.8B and LFM2.5-1.2B-Thinking as primary models, with Instruct as an additional LFM control when useful.
  Rationale: exact weights are available and family differences expose hybrid recurrence, padding, and projection errors.
  Date/Author: 2026-09-30 / Codex.
- Decision: execute GPU cases serially with short contexts, LoRA, and explicit memory measurements before increasing task budgets.
  Rationale: the GPU shares memory with the desktop; correctness needs no paid worker or disruption to another run.
  Date/Author: 2026-09-30 / Codex.
- Decision: independently derive reference losses, rather than copy the trainer branches into tests.
  Rationale: matching implementations can share the same bug. Compare values, sampled-logp gradients, parameter gradients, and optimizer movement.
  Date/Author: 2026-09-30 / Codex.
- Decision: preserve the full goal across continuations, and report each experiment as pass, fail, blocked, or untested.
  Rationale: fixture success, skipped dependencies, or a running process cannot certify an entire algorithm.
  Date/Author: 2026-09-30 / Codex.

## Outcomes & Retrospective

2026-10-01 nonlinear-kernel milestone:57 paired/native-shape attention and
convolution cases extend the audit beyond LoRA linear layers. Independent
derivatives, masks, cache state and causality hold in these slices, while
attention stress sensitivity is quantified rather than hidden behind finite
outputs. Full nonlinear blocks and actual trained activation checks remain open.

2026-10-01 fresh-group continuity milestone: native LFM FP16 collection,
Posttrain SAMPO credit and the resumed native update agree with independent
references. Optimizer/scaler continuity holds on changed data, beyond replaying
the same fixture. Broader model mathematics, backend trajectory agreement and
production fresh-learning behavior remain incomplete.

2026-10-01 LFM collected-population milestone: six fresh native episodes expose
reasoning-budget and tool-grammar failures, while all retained token/projection
invariants hold. Five native updates on real task traces validate masks,
hierarchical credit, sampler correction, clipping and Adam arithmetic within
the8GB budget. Updated-policy task learning and matched native backend
trajectories remain unproven.

2026-10-01 admission/scoring milestone: both admission method bodies reject
the observed reward-constant group. Actual TRL temperature scoring agrees with
independent softmax math; cache/teacher-force precision sensitivity is measured
on298 real sampled tokens. This narrows the diagnosis without establishing
production causality or completing fresh task learning.

2026-10-01 native collection milestone: actual multi-turn task execution is now
observed through native Verifiers, including task scoring and sampled-token
projection. A duplicate-action reward blind spot and a discount/admission
interaction deserve controlled experiments. Full native collection-to-update
learning, LFM native collection and production inference parity remain open.

2026-10-01 accumulation milestone: a direct separate-row invariant rules out
dropped accumulation on the controlled LFM fixture. Layout/precision and Adam
sensitivity are measured instead. Current-run causality, rendered populations,
model-layer independent derivatives and task outcomes remain unqualified.

2026-10-01 native LFM milestone: both requested primary model families now
have actual native veRL model/optimizer evidence. LFM executes BF16/FP16
through FSDP2 CPU offload without changing master precision; Qwen's prior
evidence uses FSDP1. Rendered LFM, fresh tasks and full initialized-backend
trajectory comparison remain separate unproven requirements.

2026-10-01 structured-reward milestone: weighted-aggregate overflow and
cancellation now have independent failing controls and a bounded source repair.
Ordinary behavior remains covered; current-run performance causality and the
broader native/fresh-rollout campaign remain unproven.

2026-10-01 rendered-native milestone: Qwen masking and applied optimizer
checks now cover real renderer token serialization, injected tool messages,
unequal response lengths and padding in native veRL. Controlled frozen reuse
also exercises clipping. Fresh AutomationBench learning and matched native
backend initialization remain unqualified; this is not their substitute.

2026-10-01 context milestone: the native Qwen precision candidate extends
from8 to256 supplied tokens with verified optimizer updates and context masks.
The failed128-token FP16 baseline remains visible. Raw evidence and scripts
remain external under the user's repository-hygiene requirement.

2026-10-01 recurrence milestone: the two-component correction is now supported
by failing native single-component controls and independent recurrence
derivatives across two runtime sources. Full rendered rollout and broader
algorithm qualification remain open; this precision work does not replace them.

2026-10-01 precision milestone: the high-scale native Qwen failure now has an
actual-input mathematical explanation and a working isolated correction.
Both current and upstream runtimes apply two updates with the FP32-region
control; a matched upstream baseline applies none. This closes the cause and
bounded experiment gap, while production integration and broader qualification
remain open. Details and hashed receipts are in native-verl-results.md.

2026-10-01 native-engine milestone: Qwen now runs through actual veRL FSDP1
loading, microbatch accumulation, score extraction, masked SAMPO loss and
AdamW in BF16 and FP16(scale1). Four updates and eight independent derivative
checks pass, including zero context-score gradients. Native optimizer movement
matches a detached scalar-form AdamW reference within2.3e-9. The8GB allocation
gate passes at about4.20GiB of Torch allocation. Neither high-scale FP16
support nor identical native TRL/veRL adapter trajectories is inferred.
See `native-verl-results.md`.

2026-10-01: Ordinary backend agreement was insufficient even after the first
KL repair. A finer represented-input sweep finds shared residual errors just
outside the 0.05 polynomial boundary. Expanding the stable arithmetic to
tenth order through 0.25 closes the tested errors without changing estimator
or penalty gradients in exact arithmetic. Model-kernel checks now compare
score gradients and scaled LoRA VJPs on identical real model outputs across
three architectures and two production precisions. They do not establish native
veRL model loading, distributed optimizer agreement, fresh task learning or
published runtime adoption. See `matched-kernel-results.md` for receipts and
the retained FP16 scale failure.

2026-09-30 Posttrain-normalized parity milestone:48 real existing tests cover
settings/correction/advantage/loss/sampling/curriculum/schedules on pinned-ce8
veRL and the current TRL candidate. The18-case independent boundary extension
on the exact post8 runtime source finds14 failures despite that ordinary
suite passing. Root Posttrain mappings include existing unstaged work, so
these are current source evidence, not a packaged dual-backend release claim.
Repair, published-pin validation and matched actual-model optimizer cases
remain open. See `posttrain-cross-backend-parity.md`.

2026-09-30 Gemma architecture extension: the third family now exercises sliding
and full attention, per-layer input embeddings, output softcap and the text path
of a multimodal loader. Fourteen native BF16/FP16 arms pass;15 direct preference
branches each have3-step results in both precisions. Initial Gemma tool fixture
grammar failure was corrected with a matching assistant call, preserving all
previous Qwen/LFM serialization. SPPO-hard retains an overflow skip and loss
value gates; lower-scale control applies all updates but does not relax gates.
The report `gemma-family-extension.md` retains144 attempts and explicit tiny
versus pretrained boundaries. Accessible pretrained Gemma remains pending;
the expanded user objective and broader campaign are incomplete.

2026-09-30 FP16 preference/model-fallback milestone:357 retained model attempts,
206 applied updates,195 strict passes, including repeats and diagnostic arms.
The broad FP16 matrix reproduces Qwen135/135 skips independent of tested
initial scale. Forty-five normalization-control updates apply with42 strict
passes; all six traced correction updates have finite module gradients.
Two independent48-case grids include the locked5.14.1 helper and local5.16.1
helper; both show8 ordinary half failures closed by FP32 normalization work.
The report `preference-fp16-and-qwen-normalization.md` retains source receipts,
padding positions, null-valued undefined comparisons and residual AOT/value
gates. Actual dependency adoption, native/fused policy replay and task learning
remain unqualified; the active campaign is not complete.

2026-09-30 preference extension: all fifteen loss branches now have independent
kernel checks; ten additional branches have three direct BF16 steps on each
model. All66 expanded parameter comparisons match; four SPPO-hard values miss
the unchanged gate. The new report `preference-branches-and-divergences.md`
retains eight raw artifacts, matched failure controls and source hashes.
Production pins and public recipes remain unchanged. Native/fused preference
paths, FP16 model breadth, clamp stress and held-out behavior remain open.

2026-09-30 native loss-branch milestone: both BF16 models pass GRPO/DAPO
three-update loops, scalar GSPO is distinguished from SAMPO token-local credit,
and additional upstream reductions have explicit supported-precision results.
All new fixtures exclude tool/padding gradients and normalize [0,1] rewards
correctly. Failed standard-loss/scale arms remain alongside passing controls.
Full evidence, methodology and transfer limits live in
`docs/research/evidence/correctness-matrix/native-policy-branches.md` and its
33 raw JSONs plus summary. The campaign remains active: real environment
collection, full recipes, broader preference kernels and intended long contexts
are not certified by supplied-trace tests.

2026-09-30 research synthesis: three source-backed reviews now identify a
bounded-reuse candidate and specific controls. A fresh corrected Qwen
simulated-task run passes all three independent BF16 parameter checks, with
rewards [0,1], [1,0], [1,1], no truncations and no tool execution errors.
One unsuccessful response in each of the first two groups claims completion
without calling a tool. This distinguishes semantic failure from parser errors.
It is six samples, not evidence of a learning trend. Qwen FP16 remains open.
Six native schedule arms now pass: one/two updates for both BF16 models and
LFM FP16. Matched inputs and first-step controls isolate the schedule; later
updates reach clipping. The research runner exposes the comparison without
changing production selections. See `recipe-selection.md` for candidate,
reproduction, primary evidence and remaining learning/deployment gates.

2026-09-30 short-rollout milestone: both requested models ran three-round
experiments on two public simulated AutomationBench tasks. The corrected
Qwen harness produces mixed rewards and finite updates; LFM exposes reasoning
truncation and zero-spread groups. Frozen-group replay reaches real clipping.
The independent float64 SAMPO loss/logp-gradient baseline agrees. Deterministic
attention resolves repeatability failures, while later BF16 Qwen parameter
gradients still miss the strict tolerance and FP16 nonfinite gradients remain
open. The BF16 score and token-ratio clip precision correction passes 29 fork
regressions, with two negative controls failing on the isolated original wheel.
Detailed mechanisms, controls, and evidence are in
`docs/research/evidence/correctness-matrix/short-rollout-results.md`.

Campaign active. Real-model slices now cover SAMPO, sampled distillation,
SFT/DPO, and GDPO/CAPO, with explicit limits in the correctness-matrix reports.
The structured-reward slice adds twelve finite updates and four exact direct
recovery controls. This is partial numerical qualification; unresolved native
runtime, precision, task-learning, and recipe gates prevent full certification.
The subsequent real Trainer slice adds twelve supplied-trace updates and
corrects a reproduced near-zero KL defect. It establishes clipping under
frozen-group reuse and passing combined gradients on both model families.
The supported-precision follow-on distinguishes healthy BF16/LFM FP16 updates
from Qwen FP16 scaler skips and a remaining strict gradient sensitivity case.
The full report is `docs/research/evidence/correctness-matrix/bf16-fp16-results.md`.
Earlier evidence is in `docs/plan/sampo-post-update-measurement.md`.

## Context and Orientation

Framework advantage construction lives in
`packages/train/src/posttrain/train/sampo_advantages.py` and
`reward_advantages.py`. Private adapters under `backends/trl/` and
`backends/verl/` own translation and runtime integration. Verifiers' native trace
tokens and masks are replay authority; new text retokenization cannot prove their
alignment. AutomationBench ownership remains in its external pinned adapter.

The retained wheel is installed in `/tmp/trl-post13-wheel-install`. A usable
Torch 2.13 CUDA 13.0 / Transformers 5.16.1 Python is
`/home/hammad/projects/trl-gdpo-capo/.venv/bin/python`; isolated PEFT 0.21.1 is
in `/tmp/trl-math-peft`. Record these versions as research environment identity;
they differ from some production runtime pins and cannot certify those pins.
Resolve production-equivalent environments as a later explicit gate.

## Plan of Work

Milestone 1 creates `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/online_rl_matrix.py`: a read-only model
loader and serial case runner, an independent objective oracle, and atomic JSON
results. Start with known-action score matrices and actual SAMPO `_compute_loss`;
then reuse model-generated scores for parameter-gradient comparisons. Include
historical trainer source as a controlled negative case, without changing the
published wheel or an existing run. Observable acceptance is both agreement on
valid cases and detection of the known opposing-turn failure in the old code.

Milestone 2 exercises exact native multi-turn inputs, padding/chunking/checkpoint
paths, accumulation, KL, clipping, and LoRA. Track errors against float64 math,
FP32 model scoring, and half-precision execution separately. Finite difference
checks must hold behavior-policy scores fixed; recomputing them changes the
function being differentiated. Tolerances must be declared before outcomes.

Milestone 3 extends independent references to the public algorithm surface and
actual optimizer/recovery paths. A loss returning zero can have nonzero correct
gradients, and equal losses can hide wrong gradients. Verify sampler correction,
reference adapter identity, and reward population before interpreting learning.

Milestone 4 selects bounded AutomationBench tasks using the external environment,
retains native task identity and trajectories, and runs controlled correction
comparisons. No external judge calls or remote training allocation are required
for the mathematical gates. Any paid judge integration follows existing bounded
bindings and is unnecessary for independent correctness tests.

## Concrete Steps

The native runner now accepts `--iterations 1`, `2` or `3` (default3).
It also accepts `--objective sampo|grpo|gspo|dapo|dr_grpo|bnpo|luspo`.
Default SAMPO semantics are preserved. Other objectives use native scalar
reward normalization and an independent reduction-specific oracle. The report
`native-policy-branches.md` gives exact commands, environment identities,
precision controls, failed gates and artifact hashes. FP64-loss or FP32-base
arms must be labeled diagnostic; a successful process exit is not a passing
qualification unless its JSON `qualification_status` says pass.
Use the native command below with distinct output paths for each arm, keeping
all other flags fixed. The one-update native path omits old logprobs and uses
current detached scores; its independent oracle must use the same mathematical
single-use definition. Reuse arms freeze old logprobs. Require identical input
hashes, matching first-step controls, zero excluded-token gradients and passing
scaler-aware accumulated gradients; do not rank task learning from this fixture.

For native supplied-trace and tiny-KL probes, use TRL source
`9f0825046ae3509a6be804d74a93fb89d5dc695e` and renderer source
`1aafe24595a7f2d2f31d24afb4b1bb7a6c6dd076`, with isolated PEFT/renderer/OpenAI
dependencies. Run model arms serially, verify each process is terminal first:

    PYTHONPATH=/home/hammad/projects/renderers-lfm-mask:/home/hammad/projects/trl-sampo-local-credit:/tmp/trl-math-peft:/tmp/trl-math-renderers-deps:/tmp/posttrain-mathdeps /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_multiturn_run.py --model qwen08 --dtype bfloat16 --output .posttrain/state/correctness/qwen08-native-multiturn-bfloat16-after.json
    PYTHONPATH=/home/hammad/projects/trl-sampo-local-credit:/tmp/trl-math-peft /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/small_kl_grid.py --output .posttrain/state/correctness/small-kl-after.json

Repeat the native command for `lfm12` and `--dtype float32`. Before arms consume
source `18e89c58bee70d25f1231cbe0dc4a865540d6fe5`. Preserve before artifacts;
never overwrite them with after results. Full runtime selection remains post13.

Working directory `/home/hammad/projects/rl`:

    nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
    PYTHONPATH=/tmp/trl-post13-wheel-install:/tmp/trl-math-peft /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/online_rl_matrix.py --model qwen08 --output .posttrain/state/correctness/qwen08-sampo.json
    PYTHONPATH=/tmp/trl-post13-wheel-install:/tmp/trl-math-peft /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/online_rl_matrix.py --model lfm12 --output .posttrain/state/correctness/lfm12-sampo.json

For bounded task experiments, include `/tmp/posttrain-mathdeps` and the pinned
benchmark source path recorded in `short-rollout-results.md` on `PYTHONPATH`.
Use the original wheel for controls and corrected source commit
`d1d298bc3c3528b57c183be4f956ddc921a92c07` for candidate experiments:

    /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/short_rollout_run.py --model qwen08 --reuse-rollouts --score-fp32 --deterministic --output .posttrain/state/correctness/qwen08-frozen-group-candidate-deterministic.json
    /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/short_rollout_run.py --model lfm12 --score-fp32 --deterministic --max-tokens 384 --output .posttrain/state/correctness/lfm12-short-candidate-deterministic.json

The runner will retain source versions, model revision, seed, optimizer settings,
case-level value/gradient errors, status, and memory. Add exact test/recovery/task
commands here as their executable path is established.

## Validation and Acceptance

Each algorithm needs independently checked mathematical invariants, real-model
gradient/update evidence, and relevant data/optimizer integration evidence.
BF16 and supported FP16 execution are the primary acceptance paths. An FP32
diagnostic pass cannot substitute for either. FP16 acceptance must include
autocast, scaler state, applied-versus-skipped updates, and a scaler-aware
optimizer reference; retain numerical misses as failures even if loss is finite.
SAMPO must retain opposite local turn credit, exclude context from direct loss,
apply its selected trajectory ratio, and clip only the correct advantage sign.
Accumulated updates must match their declared logical objective for unequal
lengths and excluded tokens. Checkpoint recovery must reproduce uninterrupted
updates within a declared numerical tolerance. Completion requires both named
model families and bounded AutomationBench learning experiments, plus a detailed
coverage report for almost all supported algorithms, not just SAMPO.

## Idempotence and Recovery

Write each result atomically and identify it by model/code/config/input hashes.
Resume only cases with no terminal artifact; do not rerun a still-live process
because a polling timeout expired. An OOM is a recorded resource failure, not an
algorithm correctness failure: release the process, verify GPU memory, and retry
with a smaller documented context. Do not shut down the desktop or another run.
Do not overwrite historical results or paid-service configuration.

## Artifacts and Notes

Machine-local raw runs go under ignored `.posttrain/state/correctness/`.
Inspect and summarize non-sensitive aggregate evidence under
`docs/research/evidence/correctness-matrix/`. The campaign plan and final report
must distinguish proven failures, confirmed repairs, qualified paths, and open
coverage. Do not include raw prompts or credentials in published artifacts.

## Interfaces and Dependencies

The case runner depends on Torch, Transformers, PEFT, and the exact TRL wheel.
Framework-neutral numerical contracts remain in their owning train package.
Generic trainer repairs belong in the TRL/veRL forks, with isolated worktrees,
ledgers, focused tests, immutable releases, and consumer pin updates in that order.
Do not use the primary dirty sibling fork as an implicit production dependency.

Revision 1: initial full-scope campaign and authoritative inventory, 2026-09-30.
Revision 2: bounded actual-rollout experiments, independent float64 oracle,
diagnostic frozen-group replay, and candidate BF16 precision correction;
retain the unqualified broad campaign scope, 2026-09-30.
Revision 3: independent excluded-token and distillation grids, two source
corrections, signed-loss test repair, and fresh group-of-two distillation
experiments with a token-budget control; broader campaign remains open.
Revision 4: independent preference loss grid, twelve real SFT/DPO updates,
double-probability localization of near-certain gradient cancellation, source
fix publication, and an explicitly open LFM renderer assistant-header mask
failure. Preserve the broad correctness and rollout-integration objective.
Revision 5: published LFM sampled-mask source repair, family reasoning-prefill
edge cases, old-wheel negative controls, framework integration regressions,
and matched token/target evidence; runtime qualification remains incomplete.
Revision 6: three authorized research reviews, provisional bounded-reuse
candidate, parameterized schedule controls and fresh stable-KL rollout evidence.
Research separates recipe hypotheses from validated correctness and learning.
Revision 7: scalar native policy branches, independently checked reward/length
normalization, 33-arm numerical evidence and reproduced rounding/scaler limits.
Keep failed precision gates visible; do not infer task learning from these probes.
Revision 8: fifteen preference branches, stable divergence arithmetic,
matched BF16 parameter-gradient repair, residual value gates and90 actual
model updates. FP64 remains a reference, not a supported production model dtype.
Revision 9: broad direct FP16 preference matrix, dynamic scaler-aware gradients,
independent query/key normalization Jacobians, locked dependency source receipt,
module/padding localization and controlled repair. Keep production adoption open.
Revision 10: user-requested Gemma family extension, shared research model seam,
two-precision native/direct architecture evidence and a separate gated
pretrained checkpoint acceptance item. No production support or pin expansion.
Revision 11: user establishes Posttrain as cross-backend semantic normalizer;
refresh pins, run actual parity suite, add independent numerical boundary
comparisons and define isolated fork/root repair ownership without committing
preexisting dirty work.
Revision 12: finer typed-input KL boundary controls, published tenth-order
corrections in both forks, immutable score-wrapper comparisons, and matched
model-score/LoRA-gradient evidence across Qwen, LFM and tiny Gemma4. Preserve
native veRL, pretrained Gemma and production adoption gates.
Revision 13: user asks to close the native-engine gap; add real Qwen FSDP1
training, repair optional padding dependency and padded-forward compatibility,
check score/context gradients and native AdamW independently, and retain
scale/architecture/packed/distributed/runtime adoption limits.
Revision 14: native FP16 backward tracing, actual-input gated norm oracle and
matched FP32-region controls on two Transformers runtimes; preserve failures,
the executed runner snapshot and production integration limits.
Revision 15: split native GDN precision ablations and independently check
chunked recurrence outputs, final states and every input derivative against
token-state equations and reference finite differences on two runtimes.
Revision 16: externalize experimental tools and raw receipts, reduce the
publishable diff, preserve the unpublished history and unrelated dirty work,
and make local evidence/reproduction locations explicit in the reports.
Revision 17: external native context-extension experiments in BF16/FP16,
separate successful loss checks from applied updates, retain the failed baseline,
and record GPU allocation without committing runners or raw data.
Revision 18: native rendered multi-turn masking/padding and active clipping
checks across four precision arms; retain actual probability sensitivity,
failed FP16 attempts and external-only experimental artifacts.
Revision 19: GDPO aggregate overflow/cancellation controls, rejected scale-only
candidate, rare high-precision framework fallback, before/after regressions
and explicit separation from current-run performance attribution.
Revision 20: native LFM FSDP2 CPU-offload BF16/FP16 checks at8/128 tokens,
FP32 master/moment preservation, bounded independent optimizer evidence and
explicit parameter-gradient/accumulation limitations.
Revision 21: native partition/row-gradient measurements, FP32 diagnostic,
cancellation conditioning and Adam coordinate sensitivity; preserve the exact
accumulation result separately from precision stability and production causality.
Revision 22: fresh native AutomationBench transport, real Qwen tool behavior,
reward blind spots and sparse-return SAMPO credit; preserve exact token evidence,
runner failures and full learning/admission gates. Experiments remain external.
Revision 23: observed-group admission replay, independent temperature scores,
exact cached replay and FP32 path-sensitivity control; keep sampler correction
and optimizer policy clipping distinct, with full native learning still open.
Revision 24: native LFM BF16/FP16 collection, budget/grammar observations and
actual collected-population updates with sampler correction and active clipping;
retain optimizer-history effects and fresh-learning/backend parity gates.
Revision 25: exact native adapter export, isolated temperature-rounding scores
and matched fresh pre/post behavior; retain the adverse small-sample outcome,
full learning-loop and shared numerical-policy gates.
Revision 26: intermediate-state, exact-repeat and larger-budget controls;
independent softmax denominator and matched temperature clipping ratios;
separate the first-update tool error from later reuse and budget truncation.
Revision 27: native linear chain-rule checks reveal a CPU-offload scalar-copy
race; publish the synchronous-host-staging source fix and five regressions,
qualify BF16/FP16 matrix gradients, and narrow earlier FP16 conclusions.
Revision 28: scaler lifecycle/two-rank skip regressions, corrected third-step
gradient and fresh rollout outcomes, geometric versus full conditional
likelihood ratios, and the remaining native checkpoint/scaler continuity gate.
Revision 29: bind/persist the native scaler in checkpoint extra state, retain
legacy compatibility boundaries, and prove same-engine/fresh-process replay
against the actual uninterrupted next update with a failing omission control.
Revision 30: collect a fresh native LFM group from the checkpointed adapter and
verify its resumed four-microbatch update, independent credit/linear/Adam math,
overlapping success/truncation and the initial fresh-policy ratio of1. Repeat
the training seeds after updating and record bounded task recovery alongside
the increased sampled work and held-out/production limitations.
Revision 31: independently audit eager attention and unfused convolution,
retain paired precision controls and scalar finite differences, quantify
saturation versus absolute gradient magnitude, and test normalized native head
dimensions without claiming actual trained activation or fused-kernel coverage.
