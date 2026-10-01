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

- [x] (2026-10-01) Repair native entropy cancellation in `../verl-posttrain-parity/verl/utils/torch_functional.py`, published sourcec1e477d7d83badb4be6742c9efde483956698f01. Independent BF16[10,11] error0.0197 and uniform high-offset collapse reproduce;24/36 CPU baseline regressions fail. Normalized log probabilities, half-input FP32 promotion and safe zero-probability terms pass36 CPU plus six CUDA controls; related utility slice53 passes. Seventeen scalar cases max repaired error1.87e-8. Actual Qwen BF16 controller scoring method bodies/private queue/GPU worker pass causal shift, mask, old/current actor, initial-base reference and restoration checks under Posttrain seq-mean-token-mean aggregation; scalar entropy metric error5.41e-9. Three updates/step0–3 scores/parameters/preclip gradients match pre-repair control bitwise. Native entropy changes only1.19e-7 on this trace. Preserve unchunked8GB OOM; existing chunk64 setting passes. Fork ledger/consumer docs updated, runtime pins unchanged. This numerical implementation repair changes no frozen product meaning; full controller admission/Ray dispatch/fresh learning remain open. Commands: from the fork, `PYTHONPATH=. /tmp/posttrain-native-verl-update-runtime/bin/python -m pytest -c /dev/null -p no:cacheprovider tests/utils/test_entropy_precision_on_cpu.py tests/utils/test_torch_functional.py -k 'not distributed' -q`; scoped Ruff and both-repo `git diff --check` pass. Sources/raw receipts are external except the production fix and regression tests.

- [x] (2026-10-01) Extend genuine queue/native-worker GPU transport to Qwen FP16 and LFM BF16/FP16. The completed four-arm matrix applies12 updates total (nine new in this slice), passes24 independent losses,288 matrix checks and1,152 scalar dots. Every step0–3 parameter/score and all three preclip gradients match corresponding local-worker controls bitwise. Sixty successful inferences and four intentional reference-loss exceptions restore policies exactly. FP16 scale1024 persists without skips. PeakQwen3.992GB/LFM2.440GB; maxima loss5.83e-8, score derivative1.13e-10, worker loss1.04e-7, Adam4.04e-9. Preserve LFM fully-active-row empty audit reduction failure; correct the external oracle to0 for an empty excluded set, with no product change. Reused-batch diagnostic clipping (.003/.004) reaches104/298 Qwen BF16,88/298 Qwen FP16,1024/2341 LFM BF16 and2341/2341 LFM FP16 at update3. Actual full controller/Ray GPU dispatch, admission/refill, fresh learning, default-scale and broader algorithm/family qualification remain open. All tools/raw evidence external.

- [x] (2026-10-01) Execute private real Ray/SimpleStorage TransferQueue and native bridge: BF16/FP16/FP32 CPU scores, masks and opposing credit retain exact row values/dtypes/order/tags. Execute Qwen BF16 actual native worker/model/optimizer through the queue for three applied updates; six loss,72 matrix and288 scalar checks pass. Steps0–3 parameters/scores and all three preclip gradients match local-worker controls bitwise; peak3.991GB, loss2.20e-10, score derivative8.63e-12, Adam4.02e-9. Fifteen successful inference calls plus one intentional reference-loss exception restore actor/base exactly. Preserve socket-path, API/equality and padded-layout harness failures. The queue reconstructs dense row columns as nested tensors; correct unpadded response fields and remove padded max_response_len hints before transport. No product fix inferred from the invalid harness layout. Full controller/Ray worker dispatch, FP16/LFM model transport, admission/refill and fresh learning remain open; tools/raw receipts stay external.

- [x] (2026-10-01) Execute actual local TrainingWorker constructor/reset/infer/train/bound-loss methods with real Torch/NCCL/FSDP2/model/optimizer on Qwen/LFM×BF16/FP16. Twelve applied updates pass24 losses,288 matrices,1,152 dots; every step0–3 parameter/score and three preclip gradients match direct-engine controls bitwise. Independent worker aggregate loss max1.04e-7, Adam4.04e-9, peakQwen3.992GB/LFM2.440GB. Sixty successful native inference calls plus four intentional reference-loss exceptions verify CPU return, defaults/flag consumption and exact base/actor restoration; training deliberately ignores inference-only flags and reproduces bound-loss updates. Preserve initial failed zero-inference-loss expectation/source/log; native sentinel1 per microbatch sums2, retry checks2. Both classic/v1 TQ scoring controllers discard placeholder loss by source inspection, not live queue proof. No product math bug in this bounded path. Cached Qwen fixture bypasses equal-reward admission deliberately; no fresh/refill/quality claim. Runtime/queue/distributed/TRL-equivalence/algorithm/family gates remain open; tools/raw evidence external.

- [x] (2026-10-01) Isolate bias-correction counter and update magnitude with both Adam moments retained. Counter-only reset keeps moments bitwise, yields L2update0.038342/cosine0.999994 with retained; reduced-LR4.9154e-5 control yields0.010938270, matching first-moment-clear norm within2.44e-7 relative while retaining direction(cosine0.99999999985). Both first fresh gradients/losses and step0–3 policies are exact controls. Eight applied updates(six replays/two fresh) pass16 losses,192 matrices,768 dots and independent temporal Adam(max2.17e-9), peak3.006GB. Four fresh episodes: both new arms[0,0], both seeds truncate2048;14 independent episode world checks across seven arms pass. Thus global norm/counter alone does not recover the first-moment-clear/full-reset outcomes. No product change. Inspect actual TrainingWorker construction/reference/train source for the next integration slice; live execution remains unproven. Broader algorithms/precision/family/worker/refill/runtime gates remain open; tools/raw evidence external.

- [x] (2026-10-01) Separate first/second Adam moments while retaining counter3 at the common second-group boundary. Exact policy/loss/first-gradient controls pass; eight applied updates include six replays and two fresh updates, with16 losses,192 matrices,768 dots, peak3.006GB. Independent temporal Adam errors2.13e-9/2.20e-7. Four fresh episodes: first-moment-only clear[0,0.5], second-only[0.5,0.5], versus retained[0,0] and prior full reset[0.5,0.5]. Displacement L2first0.010938/second6.406456 versus retained0.022253. Largest second-clear coordinate moves5.034128 with current gradient0 and carried momentum divided solely by epsilon; scalar reference agrees2.20e-7. Exact adapters/transport and10 final-state checks pass; second-clear equal rewards hide email-only versus Sheets-only successes. Correct omitted renderer PYTHONPATH in preflight; no dependency mutation. Revise momentum-only hypothesis, do not adopt resets; counter/size/generalization/full-worker gates remain open. Tools/raw artifacts external.

- [x] (2026-10-01) Isolate Adam history at the second fresh FP16 LFM group: exact step0–3 parameter/score replay, then clear moments/counter only. First fresh loss checks and preclip gradients remain bitwise identical. Five applied updates pass10 losses,120 matrices,480 dots; independent saved-gradient Adam reconstruction across both arms matches10 parameter steps within2.15e-9, all norm caps inactive. Six new matched fresh episodes complete one/two-update controls: retained[0,0] with both truncating after either count; reset[0.5,0.5] with neither truncating after either count. Exact adapters, token transport and10 independent world checks pass. Retained first-moment history/current contribution norm ratio3.8176; first displacement cosine0.4391 and relative difference161.26%. This implicates optimizer-state sensitivity in this two-seed fixture, not broken Adam or a justified reset default. Both reset episodes still fail Sheets; equal next-group rewards require refill. Record findings only; tools/raw evidence external, broader gates open.

- [x] (2026-10-01) Advance native FP16 LFM SAMPO onto its second fresh weekly-report group[0,0.5]. Replay step0–3 parameters/scores exactly, retain live Adam counter/moments and original base reference, refresh old-policy scores, then apply updates4/5. Five finite applied updates pass10 loss checks,120 LoRA matrices,480 dots and Adam(max2.15e-9); fresh loss/score errors1.72e-8/2.10e-11, peak3.006GB. New-group ratios reset1 while reference k3 remains nonzero; updates4/5 remain unclipped. Exact adapter/score transport and four independent final-state rewards support a negative same-seed behavior result:[0,0.5]→[0,0], both post-update episodes truncate2048. Next population has zero credit and is rejected. BF16 equal-reward population was not forced through admission. No product bug confirmed in this slice; optimizer-history/reuse/precision causation, held-out quality and full worker/queue gates remain open. Keep all tools/raw evidence external.

- [x] (2026-10-01) Execute eight exact TrainingWorker inference/training bodies with real TensorDict helpers and an instrumented engine, plus one exceptional-exit control. Reference inference consumes its adapter flag; training always uses the bound loss and enabled adapter. Fresh training inputs get training microbatch defaults; explicit same-input reuse retains supplied inference engineering fields. Revalidate the production mapping mismatch: SAMPO still selected GSPO while native qualification used dedicated sampo_token_credit. Repair Posttrain launcher/worker to select that loss, admit published d8e472db822f2916ed81a408b8d28192be95e678 explicitly, and reject old post8 before planning/again before native startup. New legacy rejection regression fails before repair;120 backend tests pass(8 dependency skips), two actual candidate TRL/veRL score-gradient parity cases pass; targeted Ruff/Pyright and nine import contracts pass. Native parity setup required correcting dependency ordering and adding the environment workspace path, not dependency mutation. No frozen baseline amendment: unsupported implementations must already reject. Runtime pins/images stay old and cannot run corrected SAMPO until adoption; no broad goal completion.

- [x] (2026-10-01) Add eight matched2048-token fresh LFM episodes against previous1024 controls; all eight pairs preserve first1024 IDs/logprobs exactly and verify native adapter scores/token transport/independent world rewards. BF16 starting/trained rewards at2048 are[0,0]/[.5,.5]; FP16 [1,0]/[0,.5], preserving an adverse trained pair. Peak3.461GB. Per-turn and derived total output budgets change together; no optimal/default/efficacy claim. Six exact-source native reference route/projection CPU cases check unequal lengths, next-token shift, retained excluded coordinates and both worker routes with transport doubles. Fix only external empty-mask dtype fixture, preserve failure. No optimizer/product updates; real queue/worker, fresh iterative training and broader gates remain open.

- [x] (2026-10-01) Compare eight fresh LFM weekly-report episodes under matched seeds39400/40400, starting versus three-update adapters in BF16/FP16. All four loaded adapters reproduce their recorded native scores exactly; token/logprob/excluded-mask and independent final-state reward checks pass. All eight remain reward0, truncate at1024 inside unclosed thinking, and make no tool action; changed first-turn prefixes establish inference sees the updates, not efficacy. Actual native FSDP2 LoRA-disable probes in both precisions recover initial base scores exactly and restore trained scores exactly after normal and exceptional exit; no optimizer steps in those probes. Full reference-worker/TransferQueue transport remains open. Next paired budget/reasoning controls and real fresh iterative worker behavior, preserving broader algorithm/family gates.

- [x] (2026-10-01) Collect12 fresh native training-client episodes across Qwen/LFM and two harder AutomationBench tasks, with exact token/logprob/branch/excluded-mask projection and per-task independent credit audits. Independent plain-JSON world-state checks reproduce all12 task rewards. Initial512-token LFM episodes all truncate; follow-up1024-token weekly-report pair has observed rewards[1,0], but seeds differ so no isolated budget-effect claim. Preserve an audit identity-mismatch failure and correct only its external requested task IDs. Three deterministic native SAMPO updates per precision on the complete LFM pair pass12 loss checks,144 matrices,576 scalar dots and Adam(max2.13e-9), peak2.440GB. Third-update clipping reaches both FP16 populations and BF16's failure row. No live refill, post-update task-quality, production pin/default or full-worker qualification. Next fresh adapter task comparisons and actual worker reference/admission transport, alongside broader algorithm/model gates.

- [x] (2026-10-01) Revalidate the earlier TRL admission result and add exact-source veRL active-group/Posttrain native-statistic seam agreement across seven observed/controlled cases. Equal-success Qwen has104 credited tokens but rejects under the canonical episode-spread rule. Observed LFM [1,0] retains by default; conditional truncation masking leaves one finite reward and rejects. This is a contract-consistent policy tradeoff, not a new loss defect or full worker/refill qualification. Correct a superseded source-prediction sentence; preserve runners/receipts externally. Next obtain fresh admitted populations with task headroom and qualify reference/evidence/refill in the real worker.

- [x] (2026-10-01) Qualify nonzero beta0.02 sampled-k3 on native SAMPO/GDPO/CAPO×Qwen/LFM×BF16/FP16:36 updates,72 loss/mask checks,864 matrices,3,456 scalar dots and Adam(max4.13e-9), excluded reference NaNs neutralized. Actual TRL replay matches independent value/gradient within2.14e-15/1.30e-17 and native losses within6.79e-8. Add12 deterministic Qwen SAMPO updates at beta0/.02 with identical initial and first updated states/gradients. Common-state step2 parameter gradients differ116.46% BF16/125.53% FP16, updates61.22%/81.01%; final cached k3 proxy reduces40.22%/48.62%. Score-space smallness did not imply parameter-space irrelevance after clipping. No source defect, production beta/pin change or task-quality/active-run verdict; all48 updates/tool receipts external. Next real structured/admission/reference worker paths and harder fresh-task behavior, preserving broader algorithm/model gates.

- [x] (2026-10-01) Trace native sparse-CAPO FP16 overflow to layer22 gated-norm input: ideal scaled maxima347,692/173,846/86,923 at1024/512/256 each predict one overflow matching actual;128 yields43,461 and applies. Preserve tracing OOM from vocabulary-wide gradient indexing; bounded trace keeps144 norm/3,232 module records within3.991GB. Six actual-class BF16/FP16 CPU cases distinguish avoidable intermediate overflow from final-range overflow, with12 oracle finite differences(max6.90e-9 relative). Existing FP32 delta-rule control applies three updates at1024 with108 finite norm references(max1.17e-7), still carrying348,280 in FP32. Native forward changes(mean/max sampled-score delta0.000718/0.01656); no backward-only or quality claim/adoption. Four additional applied updates; raw sources/receipts external. Next broader nonzero-KL/worker qualification, residual precision and harder fresh-task behavior.

- [x] (2026-10-01) Extend native veRL model/optimizer coverage to GDPO/CAPO on both recorded Qwen/LFM task pairs in BF16/FP16. Actual outcomes plus explicitly controlled effort/spans normalize within4.45e-16 of Decimal references. Final eight-arm matrix applies24 updates in27 attempts,54 loss/mask checks,576 matrices,2,472 scalar dots and Adam(max4.10e-9), peaks3.992GB Qwen/1.450GB LFM. Actual TRL loss replay at common scores matches scalar loss/derivatives within1.42e-15/3.47e-18; native loss difference max6.79e-8. Qwen sparse CAPO credit−8.55 triggers FP16 overflow; preserve failed audit assertion, correct only external instrumentation to allow native skip/backoff, verify three exact unchanged skips1024→512→256 then three applied updates at128 with Adam counters1→2→3. Separate extra zero-update control from final matrix. No new production repair/default/pin; live evidence/admission, nonzero KL, precision repeatability/native TRL optimizer parity and full worker/fresh learning remain open.

- [x] (2026-10-01) Capture native deterministic SDPA layer23 forward/cotangents in both FP16 scales/BF16, preserving original adapters/scores/gradients bitwise. Independent full causal/GQA NumPy references cover27,159,552 Q/K/V coordinates and18,106,368 output coordinates; full-reference finite differences agree within6.24e-11, selected actual-query checks within7.94e-17. Native query errors reach0.13367% FP16/0.90953% BF16, beyond final-half rounding; diagnostic saved-output reduction lowers them. Three all-six-block FP32 reverse-equation updates preserve forward/represented layer23 upstream cotangents and approach local final-rounding floors, but full FP16 gradient scale gap remains0.29553% versusnative0.29532%; update gap3.357%→2.901%. BF16 update changes12.15%, without quality evidence. Six total new updates remain within3.99GB; tools/raw archives external. Next isolate residual numerical sensitivity, broaden native algorithms and harder task/fresh-worker qualification.

- [x] (2026-10-01) Run15 native Torch SDPA arms/21 applied updates on both primary precisions:42 loss checks,504 matrix checks,2,016 scalar dots and Adam(max4.05e-9), peak3.99GB. Uncontrolled first-update repeats differ0.23553%/1.82980% in FP16/BF16 gradients and2.11023%/10.26202% in updates despite exact initial weights/scores. Global deterministic controls yield bitwise paired repeats in FP16 scales1024/65536 and BF16. Deterministic profiler controls retain actual Torch FlashAttention and reproduce unprofiled gradients/adapters/scores exactly. Repeat-controlled FP16 scale gap remains0.295324% gradient/3.357149% update with89 sign changes, independently explained by first-step AdamW within3.71e-11. Preserve failed profiler repeatability gate and avoid attributing all noise to attention. Next independently audit fused attention derivatives, harder tasks and wider campaign gates.

- [x] (2026-10-01) Qualify six three-update native/candidate trajectories in FP16(two scales)/BF16:18 applied updates,36 loss checks,432 matrix checks,1,728 scalar dots and Adam(max4.11e-9), exact earlier step0/1 replay. Retained FP16 endpoint scale gaps2.896%→2.062%→1.877% versusnative7.948%→5.760%→5.144%. At third reuse all104 credited positions clip in four FP16 arms/candidate BF16, leaving zero current gradients but valid momentum updates; native BF16 retains row2 signal. Twelve fresh conditional-Qwen episodes under exact native adapters/scores pass all trace/projection/credit audits, each arm2/2 reward/no truncation/errors. Baseline ceiling prevents a quality claim. Restore exact environment11f4 outside Git after import failure, verify499 Git blobs. Next native kernels/harder tasks/residual precision and wider campaign gates.

- [x] (2026-10-01) Run seven all-six-attention-block ablation updates: FP16 three policies×two scales and BF16 retained-intermediate control. Initial scores/weights remain bitwise native;14 loss checks,168 matrix checks,672 scalar dots and Adam(max1.34e-9) pass at3.95GB. Retained FP32 intermediates reduce scale gaps to0.26843% gradient/2.89572% update/86 sign flips versusnative0.84449%/7.94874%/295; FP32 products with half intermediates remain0.84378%/8.12159%. Native-products explicit-VJP control remains0.85151%/8.15353%. BF16 gradient/update changes1.9848%/11.0602%. Exact class passes18 independent primary-precision staged-reference cases(max0.02562% FP16/0.23137% BF16). Intermediate casts causally implicated; multi-update/fresh-rollout and supported-kernel/quality qualification remain open.

- [x] (2026-10-01) Capture144 internal cotangent records in blocks19/23 with exact original-gradient/adapter/score and prior-boundary replay. Attention query scale gaps0.293%/0.915% exceed output gaps0.039%/0.158%. Two native-forward-preserving FP32 internal attention-VJP updates pass four loss checks,48 matrix checks,192 scalar dots and Adam(max1.34e-9) at3.95GB; full LoRA scale gap decreases0.84449%→0.76569%, update gap7.9487%→7.2013%, sign flips295→256. Exact candidate independently checked in six CPU FP16/BF16 causal GQA cases plus four ideal-equation finite differences. Partial causal reduction only; no quality verdict or production adoption. Next remaining full-attention blocks and operation-specific backward ablations/native BF16.

- [x] (2026-10-01) Trace all24 Qwen decoder-output gradients over both complete trajectories in native FP16 scales1024/65536 (96 boundary records). Original unscaled LoRA gradients, adapter steps0/1 and scores reproduce bitwise. Four loss checks,48 matrix checks,192 scalar dots and Adam(max1.34e-9) pass. Boundary discrepancy grows0.01015% atlayer23→0.16947% at19→0.82099% at0; largest jumps traverse full-attention blocks23/19, without isolating their attention/MLP/norm components. Prefix hidden gradients are correctly nonzero upstream despite zero prompt-score gradients. Preserve the rejected FSDP-class instrumentation lookup; next inspect internal boundaries of23/19.

- [x] (2026-10-01) Expand native projection capture to all104 credited Qwen tokens in FP16(scale1024/65536)/BF16. Three updates pass six loss checks,72 matrix checks,288 scalar dots and Adam(max1.34e-9); original adapters/scores and previous six-position captures remain bitwise exact. Full population scale discrepancy is0.0007697% raw-logit→0.0090455% hidden versus0.84449% LoRA. Four-column/full-vocabulary references cover1,248 scalar dots(max2.97e-21); native projection differs from exact-dot final-half rounding at66/115/102 coordinates. FP16 complete hidden-zero tokens39→16, but lost ideal four-column norm is only0.0008896%→0.00000551%. Keep quality attribution and accumulation/downstream-cancellation causes open.

- [x] (2026-10-01) Capture actual Qwen vocabulary-projection backward in three native FP16/BF16 updates, reproducing original step0/1 adapters/scores bitwise. All six loss/score checks,72 LoRA matrix checks,288 scalar dots and Adam(max1.34e-9) pass. Independent full-vocabulary projection references over four hidden coordinates pass72 scalar fsum controls(max1.70e-21). Six-position scale discrepancy increases0.0002676% raw-logit→0.0041967% hidden L2; saturated-coordinate projection underflow has large relative but tiny absolute errors. Peak3.95GB. Expand to all104 credited positions before explaining global LoRA sensitivity; no production correction adopted.

- [x] (2026-10-01) Execute stable selected-score backward through three complete native Qwen SAMPO updates (FP16 scales1024/65536, BF16). Initial scores/weights remain bitwise native; six loss/score/mask checks,72 matrix checks,288 scalar dots and Adam(max1.34e-9) pass at2.44GB peak. FP16 scale gradient gap remains0.87175%/296 sign flips and update gap7.902%, rejecting the control as a demonstrated scale-sensitivity remedy. BF16 gradients differ2.046% and updates10.526% from native, not a quality verdict. Next isolate frozen vocabulary projection accumulation and residual model backward.

- [x] (2026-10-01) Capture six real credited Qwen token positions at FP16 scales1024/65536. Identical logits/cotangents, native head-gradient bitwise replay and exact original adapter controls isolate local underflow/rounding. Independent finite differences agree within2.24e-14. A saturated token is entirely zero at1024 but recovers six coordinates at65536; FP32 cast removal still exhibits target cancellation. Stable FP32 non-target-mass derivative agrees with the F64 oracle within1.97e-7 relative L2. Full-model corrective backward, BF16 companion and quality attribution remain open; no production precision policy changes.

- [x] (2026-10-01) Run six native TRL/scorer-controlled Qwen SAMPO loops,12 applied updates on complete tasks, with loss/score/Adam maxima1.06e-9/1.26e-11/2.31e-9. Actual scorer trajectories differ; exact veRL helper/full head plus common next-forward weights gives bitwise-identical native gradients at both updates in BF16/FP16. Native capture replays match all original adapters,96 matrix/384 scalar-dot checks pass. CPU/CUDA first-step rounding crosses4 BF16 or22 FP16 boundaries in corresponding arms. Native scale65536 baseline/FP32-delta-region arms apply four finite updates;144 actual gated-norm references pass, maxordinaryFP16 error0.03925%, maxderivative35597<65504. Repair external diagnostic OOM with bounded reductions; untraced repeat matches step0–2 exactly. Loss-scale1024→65536 causes0.8445% normalized-gradient L2 difference,295 sign flips and7.95% first-update L2 difference, with Adam sensitivity prediction error3.37e-11. Underlying half-backward scale sensitivity, default-scale TRL, wider algorithms and quality remain open.

- [x] (2026-10-01) Publish singleton FSDP sync repaird8e472db822f2916ed81a408b8d28192be95e678. A complete Qwen task population first fails in FSDP2 no-sync backward; eight minimal controls isolate unused-branch plus deferred-sync interaction. Six new regressions fail before repair; all12 sync tests pass afterward, including CUDA BF16/FP16 offload/device accumulation and two-rank Gloo equivalence. Three full collected-Qwen SAMPO arms apply six updates: ordinary FP16 atscale1024, FP32-delta diagnostic FP16 and BF16, maxloss9.93e-10/scoregrad1.36e-11/Adam2.24e-9,3.67GiB peak. Real traces preserve rewards[1,1] and104 nonzero local credits; direct math bypasses production constant-reward admission. LFM FP16 two-update regression passes48 matrix/192 scalar-dot checks andAdam2.14e-9. Pins unchanged; multi-rank unused branches, full worker/admission, fresh learning and broader algorithms remain open.

- [x] (2026-10-01) Run12 further native LFM scoring/head/alignment diagnostic updates. Matched BF16 scoring closes GRPO/DAPO parameter differences to1.86e-9. FP16 actual veRL row-wise helper plus full head gives bitwise-identical first gradients; CPU/CUDA AdamW differ only2.91e-11 but17 coordinates cast differently to FP16, zero to BF16. Optimizer-only replay reproduces each backend bitwise. Enforcing common native weights before second forward restores bitwise-identical FP16 gradients and1.86e-9 update difference while retaining own moments. These controls isolate numerical path and rounding-boundary causes, not a new formula bug or task-quality gain; wider model/algorithm/default recipe gates remain open.

- [x] (2026-10-01) Execute actual TRL scoring/loss, HF backward and AdamW for collected LFM GRPO/DAPO in BF16/FP16: eight applied updates,16 independent loss/score/mask checks, maximum errors2.28e-8/8.78e-11/2.13e-9. Native backend score arithmetic differs. FP16 matched-arithmetic controls reduce initial score difference below4.77e-7 but first preclip gradients still differ0.125–0.128% L2;18–22 small sign flips produce3.36–3.73% update L2 differences. Independent first-step Adam sensitivity predicts all parameter differences within3.76e-11. Full-head projection does not resolve residual backward differences. Native capture rerun matches all adapter snapshots exactly. All tools/raw receipts remain external; broader families, BF16/DAPO isolation and full lifecycle/quality remain open.

- [x] (2026-10-01) Extend native collected-LFM updates to GRPO/DAPO in BF16/FP16: all8 final updates apply;16 independent loss/score/mask checks,192 LoRA matrix gradients and768 scalar dots pass. Maximum loss/score/Adam errors2.23e-8/6.85e-11/2.13e-9. First updates do not clip; second reused-population token clipping12.146–19.531%. Global token versus row normalization is checked on unequal741/512 sampled lengths. Actual TRL loss replay on exported scores agrees within2.99e-8 across16 cases. Initial adapters match exactly; native TRL model/optimizer trajectories, live advantage production and full worker/default-recipe qualification remain open.

- [x] (2026-10-01) Isolate the output head in two full LFM BF16/FP16 arms: exact half-weight values promoted into a separate frozen FP32 head, native embedding/base preserved,24 more shared-coordinate finite differences. Finite analytical gradients shift0.368–0.445% FP16/0.624–2.12% BF16, but FD agreement improves only at some coordinates and remains unstable overall. BF16 upper crossings7→3/12; FP16 stays2/12. Exact baseline repeats and3.25/3.31GiB peaks hold. Reject head-only promotion as a demonstrated remedy; upstream precision, larger coordinate coverage and broader native backend algorithms remain open.

- [x] (2026-10-01) Execute full pretrained LFM backward on the real1018-token prompt plus8 sampled tokens, with FP32 LoRA masters and gradient checkpointing. Check four shared q-projection A/B coordinates with three finite-difference sizes in FP32/BF16/FP16. Best FP32 errors per coordinate are0.0284–0.0647%, all12 range0.0284–1.092%; diagnostic peak5.05GiB fits8GB. Half analytical gradients differ from FP32 by1.90–3.33% FP16/0.514–6.18% BF16, but native half finite differences do not give stable slopes. Unperturbed no-grad/grad repeats are exact. Preserve half rounding/boundary crossings and failed initial probes; full-coordinate Jacobians, LM-head isolation and broader native algorithms remain open.

- [x] (2026-10-01) Run the actual1,890-token LFM task branch through native BF16 with identical FP32 LoRA weights, then validate six layers/18 full-key-support arms against independent reverse equations. Native BF16 gradient relative error max1.60%; reference finite differences within2.14e-12. Paired native FP16 capture repeats bitwise. With common FP32 temperature arithmetic,128 selected-token log probabilities differ by mean absolute0.0122932/max0.142080; this is not KL or quality evidence. Native BF16 forward is now measured on this branch; complete nonlinear model backward and optimizer/backend trajectory agreement remain open.

- [x] (2026-10-01) Independently audit full gated LFM short-conv forward and all parameter/input derivatives in60 bias/mask/precision cases. Seven finite differences agree within5.29e-11; native FP32 absolute error max5.37e-7; masked input gradients are exactly zero. Capture all six attention layers on an actual1,890-token FP16 task path, then check18 masked subsets and18 selected-query/full-key-support arms against scalar/NumPy reverse equations. Full-support reference finite differences agree within2.35e-12; maximum gradient relative differences are1.57% BF16 recast,0.214% FP16,7.29e-7 FP32 diagnostic. Native BF16 capture, full nonlinear model backward, packed/fused boundaries and task-learning consequences remain open.

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

- Native controller old-score inference requested full-vocabulary entropy and exceeded8GB memory on the complete Qwen traces; existing64-token chunking retained the actual fixture and passed. An independent entropy probe then reproduced avoidable common-offset cancellation, including ordinary BF16[10,11] logits and chunked FP32 high-offset uniform logits. The correction is real, but its actual metric effect on this Qwen trace is only1.19e-7; do not conflate the synthetic failure magnitude with the observed run effect.
- The genuine unpadded LFM queue fixture includes a response with every position active. Its excluded-gradient set is empty; the external oracle's unguarded max reduction failed despite the native loss succeeding. Defining the audit maximum as0 when there are no excluded positions repairs the harness and preserves exact local-worker updates. Dense padding had previously hidden this audit edge case.
- Real SimpleStorage materializes per-row tensor columns as nested tensors even when the supplied column was dense. A local padded fixture cannot be passed unchanged into the nested controller contract: padding becomes apparent response length. The first Qwen transport training attempt rejected inconsistent sequence offsets before optimizer application. Trimming all response-aligned fields to actual lengths and dropping the local padded width hint restores exact local-worker agreement. This is a reproduced harness-layout issue, not evidence that the production controller sends invalid lengths.

- Real local worker execution, including metric reduction and exceptional reference contexts, introduces no parameter/gradient differences versus direct-engine controls. Its forward-only loss is an engine sentinel, not a measured objective; expecting0 was an external harness mistake. Native controller scoring methods discard it. Training ignores inference-only adapter/loss flags even when explicitly supplied, now verified with actual updates rather than an instrumented engine. Local worker success cannot prove transport/refill or release readiness.

- Equal global displacement norms can yield different task behavior: first-moment clearing preserves an email-only success, while an ordinary retained update scaled to the same norm loses it. Counter clearing amplifies nearly the same retained direction and also fails. This narrows the single-task explanation toward the distribution of parameter changes without establishing a general recipe or a production arithmetic defect. Native worker construction uses environment metadata and an existing Torch process group, offering a real local-method path beyond previous fake-engine/AST seams; GPU behavior still requires execution.

- Second-moment clearing yields an enormous but finite, independently correct update and still improves narrow partial rewards; numerical displacement is not a task-quality verdict. At the maximum coordinate, current gradient0 coexists with a5.0 change because carried momentum survives while cleared variance leaves epsilon as denominator. First-moment clearing preserves one partial success, so both optimizer components matter; direction/size and counter effects remain coupled. Equal0.5 rewards hide different assertion successes.

- Identical current gradients do not imply similar updates under different Adam histories. Retained history dominates the new first-moment contribution and the independently correct displacements differ strongly. One fresh update already loses the partial-success episode; clearing history instead produces partial successes on both seeds after one/two updates. Reset changes both moments and counter, so first-momentum-only causation and general recipe suitability remain unproven.

- Local math checks survive a genuinely fresh second group, but matched two-seed behavior regresses to all-zero truncations. Old-group clipping clears when old-policy scores refresh, while base-reference KL remains nonzero; their different anchors explain this combination. Do not diagnose missing clipping from the first update on fresh data, or infer efficacy from correct losses/Adam. With truncation masking, the second group would not be admitted; with default unmasked selection the next all-zero group is rejected.

- Worker training ignores inference-only loss/adapter flags as intended. Engineering microbatch fields persist only when explicitly reusing the same input object; a fresh input resolves training defaults. More materially, production SAMPO selection had not adopted the qualified dedicated objective. Historical GSPO mapping was only an intermediate repair and preserves an extra cap; correct the normalizer and fail closed on incompatible released runtimes rather than claiming probe-only agreement.

- Identical1024-token prefixes can hide different later actions: larger budgets reveal trained BF16 partial reward and an adverse trained FP16 pair. Thus a truncation-floor evaluation conceals useful behavioral distinctions while correct local updates remain insufficient for robust quality. Reference projection preserves excluded token coordinates and applies a one-token shift; the tested routes agree with independent coordinate labels, without validating live queue/concurrency.

- Longer-task adapters transfer to inference exactly but do not improve the two fresh seeds: all1024 generated tokens remain inside unclosed thinking blocks. This is an observed rollout bottleneck, not proof of a symbolic gradient defect or optimal budget. Native adapter disabling separately recovers the unchanged base reference and restores the actor even after intentional exception exit, closing a context-lifecycle gate without claiming live queue/worker qualification.

- Native rollout `ok` and empty harness errors do not mean task success: Qwen's report tools return invalid-cell payloads while traces remain `ok`. Equal partial-credit pairs are also filtered despite nonzero discounted turn credit. An independently checked mixed-outcome LFM group now exercises longer3248-token native trajectories without changing rewards; clipping appears by its third reuse in both precisions. Larger-budget follow-up changed seeds, so its success cannot be attributed to budget alone.

- Nonzero hierarchical credit is insufficient for current SAMPO admission by design. Both backend seams reject the observed equal-success Qwen pair; optional truncation exclusion can also remove the reward spread from LFM's mixed-outcome pair. This limits what the frozen-fixture optimizer results say about production updates. The earlier TRL result already existed; this turn extends backend/boundary coverage rather than counting it as a newly discovered defect.

- Beta0.02 remains an active score derivative when policy credit clips to zero, and in deterministic common-state controls changes actual LoRA gradients by more than the beta-zero gradient norm. Initial and first updated states match exactly, so step2 isolates the coefficient effect; later clipping/history diverge. Cached sampled-k3 reductions are40–49%, but neither those proxies nor small score-space gradients establish true KL or production-run quality.

- The live CAPO fixture's first norm overflow is expected from its ideal scaled derivative, unlike a synthetic intermediate-cast case whose ideal final derivative fits FP16. One non-finite norm-input coordinate propagates widely upstream. FP32 delta-rule state permits the still-large derivative and1024-scale updates, but also changes forward/sampler weights, so it is not an isolated backward remedy. Large-logit tracing itself can exceed8GB via boolean-index materialization; bound trace shapes and retain separate score-gradient audits.

- Sparse CAPO whitening can amplify four controlled errors among298 sampled tokens to−8.5514, despite equal actual outcome rewards. Qwen FP16 overflows at scales1024/512/256 but applies at128; BF16 and both LFM precision paths apply normally. An audit assertion initially intercepted overflow before the native scaler could handle it. Fixing the external harness reveals safe native skips with exact unchanged adapters/scores and unadvanced Adam counters; it does not prove a symbolic CAPO defect or justify a recipe change.

- Full-coordinate native FlashAttention errors exceed final-half derivative rounding at layer23, especially BF16 query gradients. Replacing the ideal softmax reduction with a diagnostic native-rounded-output reduction explains much of the discrepancy. Corrective FP32 reverse equations approach the local rounding floor yet leave the full FP16 gradient scale gap unchanged; local fidelity alone cannot support a model-level stability or quality claim.

- Native SDPA's uncontrolled repeat noise is material: BF16 first updates differ10.26% relative L2 with405 gradient sign changes. The earlier profiler mismatch also occurs without profiling, so profiling alone cannot explain it. Deterministic flags make paired repeats exact but leave FP16 scale sensitivity. These are separate issues; finite local seam checks do not prove ideal nonlinear derivatives or learning stability.

- Three reuses of the tiny frozen population exhaust all credited FP16 gradient signal despite partially clipped sampled-token metrics; zero-advantage tokens make that metric misleading. Adam momentum still moves weights. Fresh adapters remain valid, but the easy task already has2/2 baseline success, so numerical consistency gains cannot be called learning gains.

- Across all six full-attention blocks, FP32 products alone leave scale sensitivity essentially unchanged; retaining FP32 intermediate cotangents reduces it substantially. Native half-products plus explicit softmax-VJP control retains the original-scale gap, strengthening the intermediate-cast explanation while preserving softmax-kernel rounding differences.

- Query-gradient sensitivity rises inside full attention, while corresponding query-normalization and post-rotary gradients remain nearly aligned in relative gap. FP32 internal attention backward at two blocks reduces the global scale discrepancy modestly without changing forward scores; remaining blocks and other backward operations still contribute.

- Complete boundary gradients locate progressive scale divergence inside model backward, with large jumps traversing blocks23/19. These contain full attention plus MLP/norm/residual paths; component-level attribution remains unproven. Hidden prefix gradients reflect legitimate conditioning and must not be confused with prompt-score masking failures.

- Capturing every credited token confirms that the head-level relative scale gap is much smaller than the final LoRA gap. Many FP16 hidden derivatives vanish, but their four-column ideal gradient mass is tiny. Native projection outputs also differ from exact-dot final rounding, motivating an accumulation control without declaring a formula bug.

- Actual output projection amplifies the scale discrepancy at the six captured positions, while projection-only relative errors can be dominated by subnormal hidden gradients around1e-13. Already-zero input gradients must be distinguished from an accurately computed nonzero derivative. Selected-token evidence cannot explain the whole-population LoRA gap.

- A stable selected-softmax derivative alone leaves the full FP16 scale gap essentially unchanged. Local target cancellation is real but insufficient to explain the global discrepancy; replacing it also changes BF16 gradients substantially despite identical forward scores.

- Actual Qwen output-head derivatives expose both half-gradient underflow and FP32 selected-target cancellation. Promoting logits alone can worsen local derivative error. A direct non-target-mass complement avoids cancellation, but the highest relative errors occur at extremely small absolute gradients; their contribution to full-model sensitivity is unproven.

- (2026-10-01) Qwen BF16 also crosses CPU/CUDA first-step rounding boundaries; the earlier LFM BF16 result does not generalize across models. Matched-state native SAMPO gradients can agree bitwise despite later naturally divergent trajectories. Default-scale65536 is finite on these real traces, yet lowering it to1024 changes normalized gradients by0.8445% and flips295 coordinates. Exact traced/untraced replay excludes the diagnostic hook as the cause; finite updates alone do not establish loss-scale invariance or recipe quality.

- (2026-10-01) Complete1,119-prompt-token Qwen AutomationBench FP16 traces apply two scale1024 updates without the FP32 delta-region control, despite earlier short supplied-token failures at that scale. Overflow evidence is data-dependent; length or successful native score checks alone do not predict it. The distinct runtime failure is unnecessary singleton no-sync accumulation with unused sharded branches, reproduced independently of CPU offload and trainable/frozen status.

- (2026-10-01) Matching forward scores is insufficient for FP16 backward parity: actual veRL row-wise log-softmax plus full-head projection is required in this control to match first gradients bitwise. Even then, CPU/CUDA optimizer FP32 differences as small as7.28e-12 straddle FP16 rounding boundaries. Seventeen cast differences change the next forward and clipped gradient population. Native-weight alignment removes this discrepancy; BF16 has no such cast differences for this measured first step.

- (2026-10-01) Closely matching initial token scores does not guarantee identical half-precision gradients. In FP16 GRPO,18–22 sign flips among159,744 LoRA coordinates occur despite0.125–0.128% gradient L2 discrepancy. With eps1e-8, first-step Adam nearly normalizes gradients larger than epsilon to their sign, explaining near2*LR maximum coordinate differences. Completion-only versus full-head projection changes few flips but barely reduces the discrepancy; preserve this failed isolation hypothesis.

GRPO and DAPO legitimately yield different native updates on unequal-length
rows: DAPO/GRPO token weights are1.18276 on the longer successful row and
0.817239 on the shorter failed row before sampler correction. AdamW can turn
that changed contribution mixture into approximately2e-4 coordinate separation
after one LR1e-4 update. Both updates match their independent arithmetic.

Promoting only the output head improves one late-layer FP16 coordinate's best
FD error56.4%→6.35% but worsens early-layer errors. The model backward changes
less than0.45% at those FP16 coordinates. A single output-precision ablation
cannot explain the full native finite-difference staircase or justify a
production recipe change.

A fixed tiny parameter perturbation is not a reliable full-model derivative
oracle: weak LoRA A coupling and forward rounding can overwhelm its loss
change. Function-sized FP32 perturbations agree much better; native half
slopes plateau/jump even with exact unperturbed repeats. Positive-credit
upper-bound crossings change the clipped objective, while lower-bound
crossings alone do not. Native rounded-map differences must be distinguished
from the continuous derivative propagated by the backward implementation.

Even after controlling temperature division, native BF16/FP16 model forwards
differ by up to0.142 in selected-token log probability on the paired task
branch. This combines model-weight and activation rounding, not only the
attention-kernel derivative discrepancy. A repeated FP16 capture is bitwise
exact here; bounded precision differences must not be called execution drift.

Bias-enabled short-conv padded positions need not output zero despite exact
zero masked-input gradients; later biases still act. The current pretrained
model is bias-free. Actual task-path attention has Q/K RMS below2 but sharp
heads (maximum selected-query probability0.9722). Full key support is needed
for a meaningful softmax denominator; a small key subset is only an auxiliary
kernel check and must not stand in for the actual attention distribution.

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

- (2026-10-01) Put the generic entropy value/gradient repair and regressions in the maintained veRL fork. Publish source before consumer documentation; retain unchunked half-output FP32 promotion as an explicit compatibility change, keep runtime pins unchanged, and test actual Posttrain-normalized controller aggregation rather than a convenient alternative. Use existing chunking to fit the real traces, and report the small observed metric change separately from synthetic stress failures.
- (2026-10-01) Treat local TensorDict and controller TransferQueue layouts as distinct representations of the same logical batch. Validate row values and lengths plus independent updates, preserve failed attempts, and do not classify harness contract violations as product defects. Qualify transport separately from worker Ray dispatch, controller admission and rollout synchronization.

- Replace fake-engine/AST evidence for this path with the actual registered TrainingWorker and real model/optimizer while keeping controllers/queues outside the claim. Preserve the failed external inference-loss assumption and retry under new source/output names; no fork change because sentinel behavior is documented by implementation and excluded by controller selection. Own only plan/findings. Reproduce external run_actual_worker_matrix_retry1.py and analyze_actual_worker_matrix.py using recorded native runtime/PYTHONPATH; actual singleton worker uses RANK/WORLD_SIZE metadata with an existing file-rendezvous process group. Retry safely into new artifacts after hash/idle checks. Next live TransferQueue/controller lifecycle, other algorithm paths/native TRL equivalence and runtime/family gates; do not count cached equal-reward Qwen updates as normal admission or fresh learning.

- Hold both moments fixed for counter-only and reduced-LR controls, record actual LR/counter at each update, and require matched parameter magnitude/direction measurements before interpreting behavior. Own plan/findings only; no baseline amendment, fork edit or pin update. Reproduce external run_fresh_counter_size_controls.py, run_fresh_counter_size_quality.py and analyze_fresh_counter_size_controls.py with recorded runtime/PYTHONPATH; retry into new artifacts after hash/idle checks. Do not adopt counter resets or infer a lower-LR remedy from failed controls. Next actual native TrainingWorker reference/train GPU execution, Qwen/BF16 companions and broader algorithm gates instead of treating this one-task suite as the full objective.

- Isolate moment components with the bias-correction counter retained and verify the untouched component exactly. Preserve extreme updates and partial-success distinctions rather than classifying numerical amplification as observed task collapse. Own only this plan/findings; no product or fork edit, baseline amendment or pin update. Reproduce external run_fresh_moment_controls.py, run_fresh_moment_quality.py, analyze_fresh_moment_controls.py and audit_fresh_second_moment_max_coordinate.py with the recorded native runtime/PYTHONPATH. Require finite exported scores before sampling; rerun safely into new names after hash/idle checks. Next counter-only and displacement-size controls, followed by broader task/precision/family coverage.

- Use optimizer-state clearing as a controlled diagnostic, not a product default. Hold weights, scaler, LR, reference, sampler correction, credits, precision and matched sampling seeds/budgets constant, require identical first gradient, and verify all parameter steps independently from saved gradients. Own this plan and short-rollout findings only; no frozen baseline amendment, sibling edit or pin adoption. Reproduce with external native_verl_fresh_reset_adam_run.py, run_fresh_adam_quality_matrix.py and analyze_fresh_adam_matrix.py using the documented runtime/PYTHONPATH. Retry into new artifacts after hash/idle checks. Next isolate moments/counter/decay and update size, then broader tasks/seeds before recommending a recipe.

- Preserve optimizer/reference continuity at the fresh-group boundary and require exact replay of the policy that sampled it. Run only the admitted FP16 group here; do not force equal-reward BF16 data into training. Record the adverse fresh outcome and use controlled update-count/optimizer-history ablations before changing recipes. No baseline amendment or production edit; own this plan and short-rollout findings only. Reproduce with external native_verl_fresh_iteration_run.py and run_fresh_iteration_quality.py using the documented native runtime/PYTHONPATH; five updates include three replay controls and two fresh-group updates. Safe retry uses new output names, verifies prior source/receipt hashes and checks the GPU has no compute process.

- Repair Posttrain selection/gating now without advancing runtime pins or silently redefining GSPO. Canonical SAMPO semantics and reject-unsupported requirements already authorize this repair; no product baseline change. Own launcher/worker, their SAMPO regressions and consumer documentation. The generic fork loss is already published at d8e472db822f2916ed81a408b8d28192be95e678; no sibling edit/commit or pin update in this slice. Preserve unrelated release edits when staging; use focused root backend tests and actual fork score-gradient parity plus static/import checks. Recover failed parity collection by runtime path ordering, never by weakening assertions.

- Retain both positive and adverse fresh outcomes, require exact prefix controls for budget attribution, and report that per-turn and total output allowances change together. Expand seeds/tasks before adopting budget or precision settings. Source-body routing/projection checks with explicit transport doubles complement native GPU context checks, but cannot replace real worker/queue integration.

- Require adapter identity and exact cached-score reproduction before interpreting fresh before/after behavior. Match seeds, initial prompts, attention, deterministic flags, precision and budgets within pairs. Keep reward-floor outcomes descriptive; do not label them lack of task headroom or infer quality from changed text. Reference-only native probes execute zero optimizer updates and must be reported separately.

- Keep task populations separate during credit construction and use authoritative example IDs; retain the failed external audit as evidence that identity guards work. Verify sampled transport and final world-state rewards before using a fresh task fixture. Treat screening and fixed-group native updates as separate from held-out/post-update quality, and retain current admission/truncation policy while testing a group it would admit by predicate.

- Keep SAMPO's frozen episode-spread admission contract unchanged during this audit. Any token-credit-based admission proposal needs a baseline amendment and controlled fresh-task evidence, including whether discounted length preferences are useful. Default unmasked LFM admission and conditional masked admission are separate selections; do not attribute either to an uninspected active run.

- Validate the current uncorrected sampled-k3 definition and its independent masking/aggregation/sampler-weight placement without relabeling it full-vocabulary KL. Treat broad native/TRL loss replay as common-score logical agreement only. Use deterministic paired common-state controls before attributing beta effects, and keep production settings unchanged pending fresh/held-out and real worker qualification.

- Keep representability, premature intermediate-cast overflow and symbolic objective correctness distinct. Use norm references at actual inputs/cotangents to explain the captured native failure, and retain the FP32 delta-rule control as external evidence with measured forward changes. Preserve failed wide tracing and bound instrumentation to smaller tensors. Do not adopt a production norm patch, fixed loss scale or FP32 island solely from this fixture.

- Broaden native coverage using production-selected structured token clipping and0.2 bounds, retaining actual recorded outcome rewards and labeling extra effort/spans as controlled audit evidence. Test Posttrain normalization independently and replay actual TRL loss at common native scores. Report skipped attempts separately from applied updates. Preserve the original failed-hook arm, permit native backoff in a distinct external runner, and keep repeatability/live worker/native TRL model parity as separate gates.

- Retain the native-forward/FP32-backward SDPA control only as an external diagnostic. It qualifies the observed causal/zero-dropout GQA equations and captured layer23 cotangents, not every supported mask/path, all nonlinear layers, candidate repeats or learning quality. Native BF16 updates change materially; no production kernel/default adoption. Continue residual-operation isolation and wider native algorithm coverage instead of presenting local rounding-floor agreement as campaign completion.

- Use global deterministic controls for native SDPA precision comparisons after preserving the failed repeat gate and uncontrolled repeats. Record actual native dispatch and forward-score changes; SDPA versus eager is not a forward-preserving backward ablation. Do not adopt production determinism/defaults/pins or call a quality improvement from these controls. Keep all executed tools and raw receipts external.

- Report first-step scale error separately from later different-weight trajectories, normalize relative comparisons consistently, and distinguish momentum-only updates from fresh policy gradients. Use harder tasks/native kernels next; restore immutable environment dependencies outside the dirty sibling and require exact adapter-score qualification before behavioral comparisons.

- Treat all-block retained intermediates as a promising numerical control, requiring repeated updates/fresh behavior and native supported-kernel qualification. Do not adopt the diagnostic custom-autograd attention implementation, extrapolate eager-path findings to fused kernels, or equate improved scale consistency with learning quality.

- Keep the two-block FP32 VJP as a corrective experiment, not a production replacement. Its attention-weight diagnostic output lacks a gradient contract, and its partial scale-gap reduction does not establish learning quality. Extend operation/block ablations and native BF16 before recommending policy.

- Capture attention, normalization and MLP boundaries in blocks23/19 under exact native-forward/update controls before selecting a numerical correction. Keep activation-gradient and parameter-gradient norm comparisons distinct; no source formula defect is established by their different relative gaps.

- Inspect projection accumulation and downstream parameter-gradient cancellation/aggregation over the complete credited population. Report norm-space limitations and gradient mass beside zero counts; do not attribute poor task quality to large relative errors on negligible derivatives.

- Expand capture to all104 nonzero-credit token positions with bounded hidden-coordinate oracles. Preserve native forward and bitwise instrumentation controls; do not infer whole-model causation from six selected positions or adopt FP32 projection accumulation without testing it.

- Reject stable selected-score backward as a demonstrated loss-scale remedy. Retain its native-forward controlled evidence and move to output-projection accumulation; no production defaults/pins are changed by an isolated local improvement.

- Retain the stable FP32 distribution derivative as an independent reference/candidate. Require real projection/model backward and BF16/FP16 validation before changing scoring or gradient policy. Keep the original represented temperature fixed when comparing equivalent-forward derivatives.

- Decision: preserve loss-scale sensitivity as an unresolved half-backward numerical issue, separately from matched-state backend objective agreement and optimizer-device rounding. Compare actual conditional Qwen SAMPO model/optimizer paths, validate native defaults on real traces, and keep failed diagnostic OOM/source plus bounded trace recovery external. Do not declare a universal precision fix or prefer the lower scale because it avoids overflow elsewhere. No production source, pins or frozen baseline change. Date/Author:2026-10-01/Codex.

- Decision: fix unnecessary single-rank synchronization deferral in the published veRL candidate, based on a real collected-Qwen FSDP2 backward failure and eight minimal native GPU controls. Four synchronized unused-branch cases pass; the same four deferred-sync cases fail with missing `_unsharded_param`, independently of frozen/trainable status or CPU offload. Ownership: `/home/hammad/projects/verl-posttrain-parity` edits `verl/workers/engine/fsdp/transformer_impl.py`, `tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py` and `CARBONTEQ_FORK.md`; Posttrain maintains this plan, native evidence and `docs/tooling/verl/README.md`. Base source isd0d7804795dc1254f7309916fce69898387a2a8a. Keep sync enabled for data-parallel size1, retain multi-rank deferral. Validate CPU context cases, actual unused-branch GPU accumulated gradients and existing two-rank Gloo equivalence, then repeat complete1,119-token-prompt Qwen task updates serially. Preserve baseline failure and raw tools externally; no pins or frozen product meaning change. Commit/push the validated fork before recording reproducible consumer source. Rollback is the prior fork commit; multi-rank conditional-unused behavior remains an explicit gate. Date/Author:2026-10-01/Codex.

- Decision: distinguish matched-state gradient correctness from unconstrained backend trajectory reproducibility. Preserve actual helper/head and CPU/CUDA Adam controls plus a deliberately weight-aligned causal test; do not call tiny optimizer-device rounding a formula error or adopt weight snapping in product code. Keep precision-policy/default-recipe changes pending broader evidence. No baseline or dependency-pin change. Date/Author:2026-10-01/Codex.

- Decision: retain backend score-temperature differences and native trajectory discrepancies as measured qualification gaps. Use external matched-arithmetic/full-head controls and independent first-step Adam equations to locate causes; do not alter product precision policy merely to force parity. Both native optimizers pass their own mathematical references. No frozen baseline or dependency-pin change. Date/Author:2026-10-01/Codex.

- Decision: extend the published native veRL engine probe to Posttrain's token-clipped GRPO/DAPO actor mappings, using explicit group-mean/std-disabled episode credits and unchanged native task masks. Audit each normalization with global counts and replay actual TRL loss methods on exported scores; do not call that a native TRL model/optimizer or live admission qualification. Preserve exact initial adapter equality, exploratory receipts and external-only tools. Date/Author:2026-10-01/Codex.

- Decision: test head-only FP32 arithmetic using a separate frozen head copied from native half weights; preserve the tied token embedding's dtype and all LoRA weights. Record adverse coordinate results and clipping crossings rather than presenting isolated improvement as qualification. Do not change production head precision from this experiment. Date/Author:2026-10-01/Codex.

- Decision: check full-model LoRA coordinates using a common real prompt and fixed-old interior SAMPO surrogate; retain three perturbation sizes and shared coordinates, calibrate by functional loss change, record repeatability and directional clipping crossings, and do not treat failed tiny half finite differences as proof of a backward defect. Keep FP32 diagnostic and all experiment sources/receipts external. Date/Author:2026-10-01/Codex.

- Decision: close the native BF16 forward gap with a paired actual task branch and exact FP32 adapter handoff; isolate temperature arithmetic with common FP32 division and retain selected-token score differences separately from KL, generation quality and full-model backward claims. Preserve original receipts by saving parameterized runner snapshots under new names. Date/Author:2026-10-01/Codex.

- Decision: check the whole gated-convolution reverse chain, then capture actual task-path Q/K/V and preserve every key for selected-query attention checks. Use independent NumPy reverse equations to keep full support affordable; distinguish BF16 recasts from a native BF16 forward and kernel gradients from full-model parameter gradients. Retain experiment failures and keep all runners/receipts external. Date/Author:2026-10-01/Codex.

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

The first live TransferQueue slice now covers a real private queue and actual Qwen BF16 worker/model/optimizer. Three updates and fifteen inference calls reproduce the previous local worker bitwise; reference exception restoration also passes. CPU transport additionally preserves FP16/BF16/FP32 row values and metadata ordering. This closes a bounded transport gap, while FP16/LFM GPU transport, controller orchestration, admitted fresh groups, vLLM synchronization and the broader algorithm/family campaign remain incomplete.

- The local native-worker model/optimizer/reference gate now has actual four-arm GPU evidence and exact direct-engine agreement. No confirmed math defect appears here; the isolated harness expectation was repaired without changing production. Real transport/refill, held-out/fresh quality, distributed paths, compatible runtime adoption, broader algorithms and Gemma remain incomplete. Keep the full objective active; tools/raw receipts are never committed.

- The observed regression survives a substantially smaller retained update and a counter-only reset, while equal-norm first-moment clearing preserves one partial success. Tested calculations remain independently consistent. This is useful causal narrowing, not proof of all math or a justified default change. Leave broad correctness, full worker/queue/refill, production runtime adoption and held-out/family/algorithm requirements active. No correctness tools/raw receipts are committed.

- The optimizer-history explanation is now more precise: both moments alter matched-seed behavior, and independently valid Adam can move a zero-gradient coordinate through stored momentum. Neither symbolically correct updates nor partial rewards prove a sound general recipe. Keep state clearing diagnostic, investigate counter/size components, and retain full worker/refill, runtime adoption and broad coverage gates. The broad goal stays active and correctness tools/raw receipts remain outside Git.

- The matched control now identifies a concrete behavioral sensitivity to optimizer history while rejecting an Adam arithmetic defect within tested coordinates/steps. Reset removes these two truncations but does not solve Sheets or justify production changes. Full worker/refill, causal moment/size ablations, runtime adoption, BF16/Qwen/Gemma companions and broader algorithms remain open. The original goal stays active; correctness tooling/raw receipts remain external.

- A second sampled group now has uninterrupted native optimizer-state evidence. No tested loss, mask, linear-gradient or Adam invariant fails, yet the two-seed fresh reward drops and the resulting group cannot train without refill. This distinguishes correctness evidence from recipe quality without resolving the larger concern. Full native worker/refill, controlled causal ablations, runtime adoption, held-out and broader algorithm/family checks remain incomplete; the broad goal stays active. No correctness tools/raw receipts enter Git.

- The normalizer now concretely selects qualified SAMPO math and rejects declared legacy sources early. This closes a configuration gap beyond the isolated native probes, while intentionally exposing incompatible default runtime pins. Two real-library loss-gradient parity cases support the selection; eight worker-body metadata cases support flag lifecycle only. Full runtime adoption, actual worker/queue integration, fresh iterative learning and broader algorithm/model gates remain open. Correctness tools/raw receipts remain external; the broad goal stays active.

- Eight longer-budget episodes change the behavioral evidence: some previously truncated runs act later, yet trained quality is mixed across precisions. Six CPU projection/routing seams pass and introduce no confirmed product fix. Exact transport, rewards and updates narrow several hypotheses without proving global mathematical correctness or explaining an active production run. The broad goal stays active; tools/raw evidence remain external.

- Eight fresh episodes provide a negative quality result despite verified update transport; no new math defect appears in the tested seams. Both native reference contexts pass exact base/actor restoration checks. Reasoning-budget mismatch warrants controlled follow-up, but full worker/reference transport, fresh iterative training, held-out quality, nonlinear/distributed derivatives and broad algorithm/model coverage remain unresolved. Tools and raw receipts remain external; no product default or dependency adoption.

- Twelve fresh baseline episodes and six native updates broaden behavior evidence beyond the office-closure ceiling. Measured failures include clarification instead of action, invalid tool payloads and truncation; tested transport/credit/reward/update seams pass. This narrows some hypotheses without proving that math is globally correct or explaining the active production run. Full worker/fresh iterative training, post-update quality and broad algorithm/family coverage remain incomplete; all tools/raw evidence stay external.

- Seven source-seam admission checks establish current cross-backend logical agreement and an important evidence boundary: cached objective updates may use groups that active sampling would reject. Full worker/refill, admitted harder tasks, held-out learning, reference transport and broader algorithm coverage remain open. No product repair is justified by this contract-consistent result; correctness tools remain external.

- Forty-eight further native updates cover combined KL/policy/mask/Adam seams in both model families/primary precisions and deterministic beta controls. No new symbolic seam defect is found. KL measurably changes cached-fixture updates and reduces its reference proxy despite a small score-gradient impression, but worker reference transport, real admission, fresh harder tasks, held-out quality and full algorithm/architecture coverage remain incomplete. Correctness tools/raw artifacts remain external; the goal stays active.

- Native CAPO's layer22 overflow now has a concrete mathematical explanation: the scaled ideal derivative exceeds FP16 range before upstream propagation. A wider delta-rule/norm intermediate path applies three1024-scale updates without skips and matches local references, while changing forward arithmetic. This resolves the captured range question, not the full objective, current-run quality, universal precision stability or production release gates. Nonzero KL, repeatability, live worker/evidence/admission and fresh harder-task consequences remain open.

- Native GDPO/CAPO now have24 applied updates across both families/primary precisions and common-score TRL/veRL logical-loss agreement. CAPO sparse-credit scaling exposes a real FP16 backward limitation handled by existing native scaler backoff; scale128 permits three updates after three safe skips. Only experimental instrumentation changes, with all sources/raw evidence external. First overflowing nonlinear operation, default-scale qualification, nonzero KL, full structured-evidence worker/judge transport and fresh held-out learning remain unproven. The wider campaign is still incomplete.

- Six further native/corrective SDPA updates pass local seam checks. Full layer23 derivatives now have independent full-coordinate references and capture-neutrality proof; a forward-preserving backward improves local derivatives without resolving the global scale-sensitive gradient. The finding narrows a numerical mechanism while contradicting a general-remedy claim. Production defaults/pins remain unchanged; tools and raw receipts stay external. Full-model derivatives, harder-task learning and broad native algorithm/worker gates remain open.

- Native SDPA controls separate uncontrolled repeat noise from remaining FP16 loss-scale sensitivity. All three deterministic repeat pairs are bitwise equal; scale1024/65536 still differ0.2953% in gradients and3.3571% in the first displacement, which the independent Adam equation reproduces. Full fused-attention derivative truth, harder-task learning consequences and broad native algorithm qualification remain unproven. Existing published repairs and production pins are unchanged.

- Candidate consistency improvement persists across bounded update trajectories, and exact native adapters produce valid independently audited fresh rollouts in both primary precisions. Constant task-success ceilings and frozen-population clipping limit quality conclusions; supported-kernel/full-worker, harder-task, broader-algorithm/family and residual-derivative requirements remain open.

- The ablations identify a substantial causal precision effect over the full native model/population while preserving forward scores. They support a concrete corrective direction, but residual backward sensitivity, full-model derivative references, multiple-step behavior and the broader algorithm/family/quality requirements remain open.

- Internal traces and a forward-preserving corrective control provide causal evidence for part of the FP16 attention-backward sensitivity. Independent candidate equations pass bounded primary-precision references; the larger campaign still needs remaining-backward isolation, multiple algorithms/families and quality qualification.

- Decoder-boundary evidence moves the scale-gap investigation beyond the output head and narrows the next measurement to two high-amplification blocks. Exact native-gradient replay confirms instrumentation fidelity; an independent full-model derivative reference and a demonstrated corrective control remain open.

- Complete credited-token evidence closes the selected-population coverage gap for head scale sensitivity. It strengthens the next causal question—where the larger LoRA discrepancy arises—while leaving full-model derivative truth, corrective controls, broader algorithms and learning quality unqualified.

- Independent actual-value projection references establish local rounding/underflow and scale amplification, with exact-repeat instrumentation evidence in both primary precisions. The next available action is full credited-population capture and controlled accumulation/backward experiments, not a production recipe change.

- Three full native candidate updates narrow the causal explanation: all downstream linear/optimizer references pass, but head-score stabilization does not resolve scale sensitivity. Full frozen-projection and model-backward references remain necessary, alongside broader algorithms and held-out quality gates.

- The loss-scale gap is now localized partly to the actual output head, with bitwise native replay and independent derivative evidence. This narrows the next experiment to corrective backward controls; it does not close the broad correctness campaign or establish a task-quality remedy.

Matched-state native SAMPO model gradients now agree bitwise across TRL/veRL
in both Qwen precisions on complete task traces. Both actual TRL scorer arms
apply updates and pass independent references, while scoring/device-rounding
differences produce different trajectories. Default native FP16 scale65536
also applies finite updates and matches actual norm derivatives. A new measured
loss-scale sensitivity remains: identical initial weights/source/data/optimizer
device but0.8445% unscaled gradient difference and295 sign flips. This supports
continued precision isolation rather than a blanket correctness or quality claim.

The native Qwen gate now covers complete observed tool-task traces through
FSDP2 CPU offload in BF16/FP16, with six applied updates and12 independent
loss/score/mask checks. A confirmed singleton runtime failure is repaired and
published, with LFM model-gradient and distributed-equivalence regressions.
Reward-constant worker admission is deliberately bypassed, so this does not
qualify production collection/refill or establish learning quality. FP16 overflow
on prior synthetic contexts remains valid rather than generalized to all Qwen runs.

Matched-state native FP16 GRPO gradients can now agree bitwise across two
updates when the scorer/head and forward weights match. The previous second
update discrepancy is reproduced by CPU/CUDA optimizer rounding crossing17
FP16 boundaries, and removed by common-weight alignment. BF16 GRPO/DAPO matched
scoring trajectories differ by only1.86e-9. This closes these particular causal
questions; it does not prove broader numerical stability or task-learning quality.

Native TRL model/optimizer coverage now includes collected LFM GRPO/DAPO in
BF16/FP16, beyond the earlier score-only replay. Eight applied updates pass
independent loss/mask/Adam checks, but backend trajectories differ. FP16 gradient
capture explains first-update parameter sensitivity quantitatively; full-head
projection does not resolve the underlying small backward discrepancy. This
does not establish task-quality improvement or full framework-worker parity.
Exact runners and receipts stay outside Git, with source hashes and preserved
initial exploratory controls.

2026-10-01 broader native-objective milestone: GRPO/DAPO losses, sampled-token
normalization, matrix gradients and AdamW pass actual LFM engine updates in
both supported precisions. Loss-interface backend agreement holds on the same
native scores. Fresh learning, native TRL optimizer trajectories and remaining
algorithms still need qualification; no production pin or recipe changed.

2026-10-01 head-isolation milestone:24 additional full-model finite differences
bound the output head's contribution. Selected FP16 slopes improve, but the
control does not rescue overall agreement; broader upstream rounding and
backend trajectory checks remain necessary. Experimental tools remain external.

2026-10-01 full-backward milestone: selected full-model FP32 derivatives agree
with independent outer-objective finite differences, extending beyond local
matrix/kernel checks. BF16/FP16 backward remains finite but numerical finite
differences expose rounding and boundary sensitivity; no blanket native-half
Jacobian qualification follows. No production precision or recipe changed.

2026-10-01 native precision milestone: actual BF16 and FP16 model forward
captures now exist for the same task branch. Independent full-support attention
gradients hold within measured rounding differences; FP16 repeatability holds.
The last128-token score comparison quantifies precision sensitivity, while
full-model backward, live learning behavior and backend update parity stay open.

2026-10-01 actual-activation milestone:60 full gated-convolution cases and
36 attention arms on captured task-path activations extend the numerical
audit. Source equations, masks and gradients agree in these slices within
measured precision differences. Actual sharp attention does not reproduce
the catastrophic artificial stress case here; this does not close native
BF16 full-model backward or prove that accumulated numerical error is harmless.

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
Revision 32: check the full gated-convolution reverse chain and padding/bias
semantics; capture real FP16 task-path attention and compare selected queries
with full causal key support, preserving native-BF16/full-backward limitations.
Revision 33: capture a native BF16 forward, check its full-key-support attention
derivatives, repeat FP16 exactly, and quantify paired chosen-token score
differences under common temperature arithmetic without interpreting them as KL.
Revision 34: compare four full-model LoRA derivative coordinates over three
perturbation sizes and three precisions; preserve native half rounding/crossing
failures, functional calibration and exact baseline-repeat evidence.
Revision 35: isolate FP32 head arithmetic while preserving the native embedding
and weight values; retain improved/adverse coordinate slopes and reject the
head-only control as a demonstrated general remedy.
Revision 36: run native GRPO/DAPO updates on collected LFM task traces in
BF16/FP16, check unequal-length normalization and active token clipping, and
replay actual TRL losses on native scores while preserving model/optimizer gates.
Revision 37: execute actual TRL model/optimizer arms, isolate half-temperature
scoring, capture preclip native gradients and test full-head projection; explain
first-update Adam sign sensitivity without treating bounded checks as full parity.
Revision 38: isolate actual row-wise scoring and full-head projection, reproduce
CPU/CUDA Adam rounding independently, and test half-boundary causation with
matched next-step weights; preserve unconstrained trajectory and quality gates.
Revision 39: reproduce and repair singleton FSDP2 no-sync unused-branch failure,
validate complete collected Qwen task updates and an LFM gradient regression,
publish the fork before documenting candidate qualification without pin adoption.
Revision 40: compare native TRL Qwen SAMPO gradients, isolate device half-cast
boundaries, qualify default loss-scale gated-norm derivatives and retain a
measured scale-sensitive gradient/update gap with exact instrumentation controls.
Revision 41: capture native Qwen output-head gradients, separate half underflow
from saturated-target cancellation, validate a stable FP32 distribution reference
and retain full-model/precision/quality qualification before adopting a remedy.
Revision 42: execute stable selected-score backward in native Qwen FP16/BF16
updates, quantify its Adam-amplified changes and reject it as a demonstrated
scale-gap remedy before isolating output-projection accumulation.
Revision 43: measure actual full-vocabulary projection derivatives at selected
credited tokens in FP16/BF16, quantify local scale amplification and require
whole-population capture before attributing the global LoRA discrepancy.
Revision 44: capture all104 credited-token head/projection gradients, validate
full-vocabulary selected-coordinate oracles, quantify lost gradient mass and
retain accumulation/downstream-cancellation isolation before adopting changes.
Revision 45: trace complete decoder-boundary cotangents, identify backward
amplification across blocks23/19, distinguish legitimate prefix conditioning
from masked prompt scores and require internal-block isolation before correction.
Revision 46: capture full-attention internals, isolate query-gradient sensitivity
and test native-forward FP32 internal VJP at blocks19/23 with independent
primary-precision references and a measured partial global-gap reduction.
Revision 47: ablate all six full-attention blocks, separate FP32 products from
retained intermediate cotangents, validate native BF16 and18 independent
staged-reference cases, and require multi-update/fresh-kernel qualification.
Revision 48: run matched three-update trajectories, identify fully clipped
credited populations versus momentum updates, qualify exact conditional-adapter
handoffs and twelve fresh native episodes, and retain baseline-ceiling limits.
Revision 49: run native SDPA controls, preserve failed repeatability checks,
measure uncontrolled BF16/FP16 noise, establish exact deterministic repeats
and quantify the remaining repeat-controlled FP16 gradient/update scale gap.
Revision 50: capture full native layer23 fused-attention derivatives, check
independent causal/GQA equations and finite differences, isolate rounded-output
reduction effects, and reject local FP32 reverse-equation improvement as a
demonstrated global scale-sensitivity or training-quality remedy.
Revision 51: qualify native GDPO/CAPO BF16/FP16 updates on both recorded model
families, independently normalize structured credits and replay TRL loss,
preserve a failed overflow audit hook, and verify native sparse-CAPO scaler
backoff with exact skip safety and three subsequent optimizer updates.
Revision 52: locate native sparse-CAPO gradient range overflow at layer22,
preserve/fix bounded tracing, distinguish intermediate-cast versus final-range
failures with independent references, and qualify a forward-changing FP32
delta-rule control without claiming production or learning-quality adoption.
Revision 53: qualify nonzero sampled-k3 native updates and common-score TRL
agreement, test excluded reference NaNs, isolate deterministic beta effects
from a common state, and distinguish cached reference proxies from true KL
and fresh-task or production-run conclusions.
Revision 54: extend prior admission replay to veRL/native-reward statistic
seams and observed LFM truncation-policy effects, preserve the frozen
episode-spread contract, and clarify cached-objective versus live-update evidence.
Revision 55: screen harder real tasks, independently verify transport/group
credit/final-state reward, qualify longer native LFM updates in both precisions,
and preserve seed, admission, clipping and fresh-quality evidence boundaries.
Revision 56: verify exact fresh adapter transport, record matched reward-floor
behavior and unclosed-thinking truncation, and execute native reference-context
disable/restore checks without inflating them into optimizer or worker proof.
Revision 57: compare matched token budgets with exact sampled prefixes, preserve
mixed precision-specific fresh quality, and extend native reference routing/
projection seams without adopting a recipe or claiming live worker qualification.
Revision 58: exercise worker inference/training flag lifecycle, repair the
production SAMPO objective selection with early compatibility rejection,
validate actual candidate-library parity and preserve runtime adoption gates.
Revision 59: retain live Adam/reference state across a second fresh SAMPO group,
verify exact replay and independent derivatives/updates, and preserve an adverse
matched-seed fresh outcome before controlled recipe/optimizer-history ablations.
Revision 60: isolate Adam history and one/two-update behavior with identical
first fresh gradients, independent temporal optimizer reconstruction and matched
episodes; preserve partial-success, admission and generalization limitations.
Revision 61: separate first/second moments with counter retained, reconstruct
zero-gradient extreme displacement, preserve differing partial successes and
revise the momentum-only interpretation without adopting a reset recipe.
Revision 62: isolate counter age and match retained-update magnitude, verify
independent temporal arithmetic and failed behavior controls, and advance toward
actual native worker execution and broader coverage beyond the single task.
Revision 63: execute actual native worker/model/optimizer/reference methods
across both models/precisions, verify exact direct-engine agreement and exception
restoration, preserve the inference-sentinel correction and transport limits.
Revision 64: execute real private queue/native bridge and Qwen BF16 model updates,
verify bitwise local-worker agreement, preserve failed padded-layout expectations,
and retain full controller/other precision/family/fresh-learning gates.
Revision 65: complete Qwen/LFM BF16/FP16 native queue transport matrix, preserve
the fully-active-row audit correction, verify exact controls and measured clipping,
and retain full controller/default-scale/fresh-quality and broader coverage gates.
Revision 66: execute actual normalized controller score-storage methods, preserve
unchunked entropy OOM, reproduce/repair/publish entropy cancellation with CPU/CUDA
derivative regressions and native replay, and report the small observed trace effect.
