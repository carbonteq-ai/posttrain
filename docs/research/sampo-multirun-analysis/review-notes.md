# Independent buddy review: training evidence and causal limits

Reviewed 2026-10-02. This review reads the actual training captures and scalar
histories, historical TRL source, and native training transcripts. It does not
launch training or alter live runs. Training exports were still progressing when
these notes were written; all rollout-level numbers below use the three named
complete captures. Recent FP16 history series are complete even where their
rollout export is still progressing.

## Assessment

The strongest established defect is loss of local SAMPO credit in the historical
sequence-ratio implementation. Active sampling and VORTEX select and correct
training data; they do not repair that differentiation path. This is a confirmed
mechanism, not a measured percentage attribution of task failure. There is no
matched corrected-trainer task run in the inspected population.

The broad claim that the model did not learn is too strong. Policies moved,
some training tasks improved, and failures differ by task. The report should
lead with actual training populations, distinguish generated candidates from
optimizer contributions, and use held-out evaluation as supporting evidence.

## Exact source and loss interpretation

Historical TRL revisions inspected:

- `4950b99d457faacbec856cbd5305732e7b3cf7b0` (post10).
- `c4d0db051a7839fe1b1d587185fac33ba88c784f` (post12).

Read with `git -C /home/hammad/projects/trl-sampo-local-credit show
<revision>:trl/trainer/grpo_trainer.py`. Relevant functions are
`_prepare_active_sampling_inputs` and `_compute_loss`. Both historical losses
differentiate a single sequence importance ratio broadcast over token-aligned
advantages. For an unclipped trajectory with fixed token correction weights
`w_t`, the policy contribution is `-r * mean_t(w_t * A_t)`. Each token
participates in the derivative of the same sequence ratio. Distinct local
advantages therefore influence a trajectory aggregate rather than their own
token derivatives. Exact opposing-credit cancellation requires cancelling
weighted credit; it is not guaranteed for every real opposing pair. The KL term
also contributes separate gradients.

The existing unit-weight CUDA canary demonstrates exact cancellation under the
old loss and nonzero local updates under post13. It does not measure historical
task lift or certify FP16 production behavior. See
`docs/research/evidence/sampo-opposing-credit-cuda-canary.json` and
`docs/plan/sampo-post-update-measurement.md`.

Zero logged clipping is expected with one pass and detached current actor
scores: ratios are one at loss evaluation. This does not mean the ensuing
optimizer update cannot move the policy beyond clipping bounds.

## Active sampling and VORTEX were operating

Work-package descriptions explicitly declare yield-first curriculum with active
refill. Histories include curriculum selection and per-round admission metrics.
The compact top-level resolved settings do not contain every curriculum option;
the description and runtime telemetry should be distinguished from a complete
serialized selector configuration.

Mean training metrics:

| Run (Trackio ID prefix) | Updates | Generated rows/update | Variance-retained rows/update | Variance-retained fraction | Generation rounds |
|---|---:|---:|---:|---:|---:|
| Original SAMPO `c49136f7` | 150 | 72.373 | 64.000 | 0.89061 | 2.033 |
| Continuation from120 `aab30e6e` | 43 | 154.744 | 144.000 | 0.93324 | 1.930 |
| Fixed-tools continuation `fbcdd791` | 39 | 156.000 | 144.000 | 0.92564 | 1.923 |
| FP16 base `476f5dd9` | 100 | 169.860 | 160.080 | 0.94347 | 1.050 |
| FP16 higher LR `444abb71` | 60 | 170.000 | 159.200 | 0.93790 | 1.050 |
| FP16 H100 no truncation penalty `ccd2fa43` | 50 | 171.720 | 159.240 | 0.92934 | 1.100 |
| FP16 continuation `2fbe8ca6` | 100 | 141.840 | 131.040 | 0.92517 | 1.070 |

Despite their names, historical `active_sampling_candidate_groups_generated`
and `...retained` count **episode rows**, not prompt groups. Divide by configured
generations. With oversampling, retained means variance-qualified **before**
surplus trimming: the three FP16 base arms optimize 144 rows/24 groups and the
continuation optimizes120 rows/30 groups. A high retained fraction cannot be
reported as the optimizer-used fraction of generated data.

VORTEX prioritizes useful reward spread, not guaranteed semantic discovery.
It can select partial-credit variation, and later environment versions allow
tool-mistake penalties to produce spread while task partial credit is constant.
Among complete non-Simple candidate groups with native reward variation:
65/1010 (6.4%) in `aab30e6e` and37/901 (4.1%) in `fbcdd791` have constant task
partial credit. The original run has0/2273. These are candidate-population
statistics; do not label them exact retained optimizer percentages.

## FP16 is not supported as a zero-update explanation

All four substantial FP16 runs log zero skipped optimizer steps at every
observed update, and loss scale stays1024 throughout. Mean gradient norms are
0.004717,0.004805,0.003867,0.004156 respectively for the four FP16 rows above.
Sampler/trainer log-probability gap means are about0.0010, versus0.01728 in the
original SAMPO run. Importance-correction cap fractions are zero in base100
and at most approximately4e-8 in the other run means. There is no evidence that
VORTEX's cap suppresses a material fraction of these updates.

Incoming SAMPO turn-credit shares are0.2932,0.2932,0.3064,0.2747; informative
turn-advantage fractions0.5349,0.5329,0.5390,0.4836; singleton-anchor fractions
0.4017,0.3947,0.4066,0.4506. These are prepared-credit diagnostics, not evidence
that the historical loss delivered the credit locally.

Existing checkpoint rescoring in
`docs/plan/sampo-post-update-measurement.md:410` confirms policy movement between
step20 and40 for both FP16 learning rates. Its530 policy tokens from four short
first turns are deliberately selected off-policy probes: they do not estimate
population KL, actual historical clipping, or convergence.

## Actual training behavior challenges a pure exploration explanation

Native transcripts inspected include these training traces:

- Fixed-tools continuation, step18,
  `8d3a003b15334b37a263471edf196730`, `finance.wire_transfer_approval`: reads
  Gmail and sheet, scores5/6, but marks the15k transfer `Sent` instead of
  `Pending Approval`. This is a conditional decision after retrieval.
- Same run, step27, `e99f47fd17cd496a80442973b43e2071`: writes prohibited
  row4 `Sent`, then `Pending Approval`, and misses a required row5 update;
  scores4/6. More actions are not uniformly better actions.
- Same run, step6, `b2e32f28d2d74335bd721e8c799db1bf`,
  `marketing.google_ads_pause_low_ctr`: scores4/5 but pauses the protected
  New Product Launch campaign.
- Same run, step37, `d142dac174724f1cb654c3fd586a8b80`: pauses two protected
  campaigns and misses a required campaign; scores0.5.

These first/last/highest-reward case selections are purposive and cannot
estimate error prevalence. They establish that retrieval and substantial
positive credit occur alongside selective-rule mistakes during training.

In the original SAMPO run, non-Simple generated-candidate mean partial credit
rises from0.32486 in steps1–20 (1464 episodes/117 tasks) to0.37193 in steps81–100
(1348 episodes/109 tasks). Truncation falls25.48% to10.61%. Adaptive populations
differ, so neither change alone estimates a causal policy improvement.
Across the complete150-step run, wire-transfer candidate means improve
0.625 to0.7222 in the first/last20-step windows, while ads means fall
0.3042 to0.2306. This supports mixed task-specific learning rather than a single
universal stagnation story.

## Report checks still required

### Follow-up: small advantages and normalization

The user's concern about spreading credit over many episodes and tokens is
partly supported, but two different quantities must be kept separate.
Historical post12 logs `advantages/abs_mean` for its internally computed scalar
episode advantages **before** replacing them with `precomputed_advantages`.
Consequently, recorded `train/rl/advantage_abs_mean` means0.7220,0.7174,0.7114,
and0.7020 in the four FP16 runs are not the magnitude of the actual combined
SAMPO token advantages. The existing hierarchy record reports episode absolute
means0.12600,0.12184,0.10898,0.11523 and turn absolute means0.05456,0.05225,
0.05054,0.04564, respectively. These are prepared candidate-population
hierarchy diagnostics, not a reconstruction of final retained policy tokens.

The first-update history contains one separate framework record with41 fields
and another normalized TRL record with54 fields, a few microseconds later.
The hierarchy metrics and generic advantage metric have100/60/50/100
observations, with no duplicate per-step values. Current worktree additions
`SAMPOAdvantages.policy_credit_evidence` and its callback wiring are uncommitted
and absent from those historical hierarchy records. Reading current source
must not retrospectively assign the new metric meaning to old observations.

With per-device microbatch1 and gradient accumulation144 (or120 in the last
continuation), the historical SAMPO `grpo` loss is an average of trajectory
means: `L = (1/B) sum_i (1/T_i) sum_t loss_it`, where `T_i` counts sampled action
tokens under the policy mask. For unclipped sequence ratios and fixed VORTEX
correction weights, a token-score derivative is
`-r_i * mean_t(w_it A_it)/(B T_i)`. Under the corrected local derivative it is
`-r_i * w_it A_it/(B T_i)`. A local credit change confined to `q` action tokens
contributes proportionally to `q/T_i` in the trajectory average; the old loss
also broadcasts its aggregate to other tokens, obscuring opposing local signs.
These are derivatives with respect to log probabilities, not a direct
prediction of an Adam parameter update or eventual task success.

Increasing batch size does not automatically make the expected update smaller:
the number of averaged contributions also increases. A behavior's frequency,
sign agreement, context-specific gradients, trajectory length, optimizer,
learning rate, and competing KL all matter. These data demonstrate small
prepared advantages and long episodes; they do not alone establish that
gradient dilution is the primary cause of weak learning. The selected SAMPO
`mean` advantage normalization subtracts the group mean; it does **not** divide
by the mean or standard deviation. A mean absolute advantage is not the
gradient norm, and a centered near-zero scalar loss is not a zero gradient.

The historical refill `num_items_in_batch` counting-tool-output bug affects
global-token-normalized losses such as DAPO. It does not alter this SAMPO
`grpo` trajectory-mean denominator and must not be listed as a SAMPO causal
normalization defect. Source: exact post12 `_compute_loss` normalization
branches; `policy_config.py` selects `loss_type='grpo'` for SAMPO; the existing
objective audit already records this distinction.

### Capture reconciliation found nonrandom gaps

Independent per-step reconciliation finds the original run logs10856 generated
rows but stores10787 traces. Gaps occur only at step41 (68 expected/39 stored),
step83 (68/60), and step87 (64/32). Missing rows interrupt multiple prompt
groups within the first candidate batches; they are not simply complete
zero-variance groups omitted by filtering.

The fixed-tools continuation logs6084 generated rows but stores5940 traces.
All144 missing rows belong to its step11 first candidate batch; only batch2
(18 rows) and batch3 (6 rows) appear. Step11 logs one admission-rejected group,
one failed rollout,162 completed rows out of168 attempted, and6 admission-missing
rows. Those counters explain one excluded six-row group, but not the absence
of the entire144-row batch from the projected trace store. A publication or
batch-observation path is plausible, but its exact failure is not established
by the read-only evidence. There are no trace-sync scalar keys in these cached
histories. Do not assume the missing episodes are random or replace them with
zeros; final completeness must distinguish all stored from all generated.

1. Reconcile every final training capture against stable start/end counts and
   report incomplete groups, errors, truncations, and omitted launches.
2. Report native candidate reward, task partial credit, and optimizer telemetry
   separately. Trainer shaping/admission cannot always be reconstructed from
   projected native episode reward alone.
3. Use matched-task windows and disclose adaptive sampling selection bias.
   Repeated tasks/rollouts are not independent experiment seeds.
4. Define assertion identities and positive/negative classification explicitly.
   Five missed samples are finite-support evidence, not a proven exploration
   ceiling. Generalization can change unsampled behavior probabilities.
5. Do not claim LoRA rank4 is sufficient without a controlled rank comparison.
   Information-per-episode arguments and unrelated benchmarks do not establish
   that conclusion for this environment.
6. Preserve the distinction between original SAMPO's checkpoint100,
   FP16 base100, and the separate GRPO/VORTEX100 run.

Sources are the read-only cache under
`.posttrain/state/analysis/sampo-multirun/`, Trackio project `posttrain-lab`,
recorded work-package descriptions/configurations, exact TRL revisions above,
and the cited existing objective evidence. No new experiment was conducted.

## Review of the report builder before final capture

The dilution example in `build_report.py` is numerically correct:20 tokens
with0.05 isolated credit over4000 tokens produce mean0.00025; dividing by
144 episodes and4000 tokens gives4.34e-10 per-token log-probability derivative
in the historical loss, versus8.68e-8 at credited tokens with the corrected
local derivative. The report labels this an illustration, retains the Adam
and gradient-direction caveats, and correctly separates generic historical
advantage telemetry from supplied SAMPO credit.

Recommended wording repairs sent to the report author:

- The wire-transfer transcript changes sheet status to `Sent` and sends
  notifications. It does not establish actual execution of a bank transfer.
- A rising score over changing adaptive candidate populations is not by itself
  proof of policy learning. Use separately observed policy movement and
  matched-task results to support that conclusion.
- Smaller sampler/trainer gaps in FP16 arms do not isolate precision as their
  cause; other operating changes are confounded.
- Call the objective problem a confirmed credit-assignment defect; its task
  outcome contribution remains unquantified.
- The post13 local-derivative repair automatically activates for token-aligned
  advantages (`advantages.size(1)>1`), rather than requiring an opt-in flag.

Two reproducibility issues also require repair before final delivery:

1. `analyze.py` requires cached `all-run-configs.json`, but the listed capture
   commands do not create it. The exporter must capture it, or analysis must
   fall back to the configs already retained in run metadata/durable evidence.
2. The purposive training exporter currently skips whenever an output JSONL
   exists, even without its completion metadata. Interrupted or stale exports
   must be validated against their source capture and repaired rather than
   silently reused.

These are report/capture workflow issues, not evidence that historical training
or task scores were wrong. Final numerical aggregates remain contingent on
completing the remaining FP16 captures and rerunning analysis.

## Final population review

All compact training captures are now complete. Independently reconciled
74,006 stored episodes across eight nonempty SAMPO runs and two zero-episode
launches. Stable start/end provider counts hold. The four substantial FP16
matched-task first/final20-step means independently reproduce the analysis:

| FP16 arm | Matched non-Simple tasks | Early | Late |
|---|---:|---:|---:|
| Base100 | 116 | 0.345551 | 0.362915 |
| Higher-LR60 | 114 | 0.366268 | 0.372955 |
| H100 no-penalty50 | 116 | 0.376758 | 0.393446 |
| Continued100 | 118 | 0.381031 | 0.394565 |

The H100 run additionally has three missing projected traces at step38:
204 logged generated versus201 stored. This leaves three incomplete groups
containing15 stored episodes, excluded from group statistics. The interrupted
four-rollout-step H100 attempt has672 stored episodes versus504 rows counted
in completed optimizer-step sampling telemetry; the extra candidate collection
does not imply an extra completed optimizer update. These grains remain distinct.

### A sampled success did not become reliable behavior

The higher-LR run's ads task provides a direct training example:
step26 trace `1ee1fe7e66154fa69d61458765377f12` satisfies every task assertion,
but it is the only full success in that six-attempt group (group mean0.25),
and the only full success among72 stored task episodes in the run. Task partial
credit is1.0; native reward is0.92 because four tool mistakes remain. It reads
worksheet rows and Gmail, then pauses the four required campaigns.

The30 task episodes at later revisits (steps30,41,52,55,59) contain no full
success. The step59 selected trace `d7914ef23b0a4eeaba46ffe05ef488e7` reads
campaigns/metadata but skips worksheet rows, pauses the protected launch and
misses Mid-Market Awareness; partial credit0.6. This demonstrates a correct
branch in sampling support alongside later wrong branches. It does not prove
the model first learned a reliable skill and then forgot: the step26 group's
other five attempts already failed, and subsequent group means0.383,0.483,
0.325,0.300,0.467 exceed the step26 mean. Best-episode comparisons exaggerate
apparent improvement followed by regression.

The original run's ads first20-step mean0.304 versus steps81–100 mean0.1625
is a suggestive task-specific decline, but represents only three versus four
small stochastic prompt groups. It should not be presented as a precise
forgetting estimate. The mechanism conclusion is narrower: desired branches
were sampled but weakly and unreliably consolidated; historical loss locality
and coarse/competing supplied credit are plausible contributors with different
levels of evidence.

### Final report and reproducibility checks

The generated report retains appropriate observational, reward-grain and
single-seed caveats. Its advantage-dilution maths remains correct. The exporter
now creates `all-run-configs.json`, and purposive transcript reuse now verifies
both source-capture and output hashes, addressing the two earlier workflow
issues.

Two final presentation corrections were sent to the author: escape or replace
the pipe characters in advantage-table headers, and label the step26 value as
task partial credit1.0 rather than native reward1.0. Final source/gradient
interpretations otherwise agree with the independent review. Held-out
comparability is bounded to matched task manifests and inspected adapter/scorer
code; it does not make training environments, budgets, ancestry or seeds equal.

## Competing integrated explanation and a falsifiable proposal

My explanation is **unstable consolidation of partially correct branches**.
The stack supplies useful comparisons, but the selected comparison, supplied
label and applied gradient each discard information needed for reliable
procedure execution. This differs from an absolute exploration ceiling.

The elements have different evidence levels:

| Element | Evidence level and qualification |
|---|---|
| Successful and failed branches coexist at the same actor checkpoint | Directly observed: step26 ads has1/6 complete solutions. The correct branch was never reliable at that checkpoint. |
| Partial successes can receive positive relative episode credit | Confirmed by centered group-relative construction. A0.6 incomplete solution can rank above other failures; this does not mean it outranks a1.0 complete solution. |
| Full behavior has weak empirical support on particular tasks | Observed in ads1/72; opportunity counts and behavior probabilities differ across tasks. The successful episode can have a strong relative advantage despite being rare. |
| VORTEX optimizes supply of nonconstant comparisons rather than mastery | Confirmed selector/admission semantics. This is useful, but a reward-varying task can continue producing heterogeneous incomplete solutions. |
| Missing local comparators and coarse turn assessments | Observed singleton rates and construction. Multiple parallel calls can share one turn assessment; no measured individual-call blame is available. |
| Positive episode credit can offset a bad turn's negative credit | Confirmed addition rule; prevalence specifically on forbidden-call tokens has not been reconstructed. |
| Historical gradients mix different turn locations | Confirmed post10/post12 defect. Current post13 repair fixes this path; it does not retroactively change these runs or repair incorrect supplied labels. |
| Cheap, frequent action patterns dominate shared updates | Plausible frequency/alignment explanation, not measured gradient attribution. There is no general reward rule preferring cheap incomplete execution to a full solution. |
| Cross-task interference or drift destroys a learned skill | Plausible and supported by policy movement, but not identified causally. Later bad samples can reflect continuing stochasticity rather than forgetting. |

The causal hypothesis is therefore: frequent partial branches get reinforced
relative to weaker siblings, while rare successful procedures have insufficient
consistent local evidence to become the default branch. Singleton anchors,
parallel-call aggregation, episode/local-credit mixing and the historical
derivative compound that difficulty. Cross-task updates and drift may further
move probabilities, but are additional hypotheses rather than established
causes of each reversal.

### Minimal staged proposal after the loss repair

Do not combine all changes in a single new recipe. Begin with the current
post13 objective and otherwise preserve the selected environment/scorer,
rank/alpha, precision, optimizer, sampled-episode budget and task inventory.
Historical runs are controls for context, not an interchangeable baseline when
other runtime changes differ. Record any differences explicitly. No new run is
authorized or launched by these review notes.

1. **Measure credit delivery before adding another knob.** For actual training
   episodes, retain the applied token advantages, masks, sampler corrections,
   before/after actor scores and exact native turn/call outcomes. Distinguish
   candidate, admitted and optimizer-used rows. Audit sampled complete
   solutions, partial solutions with broken guards and no-action failures.
   Report the fraction of forbidden-action spans with positive supplied credit,
   the actual policy-gradient contribution by turn, and changes in likelihood
   of correct versus wrong actions under the same frozen observation prefix.
   A corrected-objective real trajectory must preserve the intended local
   derivative; passing only the synthetic fixture is insufficient.

2. **Test consolidation with fixed sentinel contexts.** Alongside native
   training population metrics, repeat a predefined set of complete procedure
   tasks across checkpoints with enough samples to estimate full-success and
   branch probabilities. Keep temperature and sampling settings fixed. Compare
   task means and success rates, not each checkpoint's best episode. If a
   checkpoint is reliably successful and a later checkpoint is reproducibly
   worse, regression is established. If both checkpoints mix successes and
   failures at similar frequencies, the apparent reversal is sampling noise or
   persistent unreliability. Check task-specific and shared-task gradients
   separately before attributing a drop to interference.

3. **Only then test the dominant remaining evidence gap.** Use separate arms:
   (a) explicit guard-cost/full-completion scoring; (b) more sibling attempts at
   fewer distinct tasks per update, holding total sampled episodes fixed; or
   (c) verified procedure retrieval support. Choose among them from the credit
   audit rather than changing every knob. A turn-weight or reward-scale change
   alone cannot repair parallel-call credit granularity; a larger sibling
   count alone cannot create an absent retrieval behavior.

Concrete predictions and falsifiers:

| Hypothesis | Minimal controlled comparison | Expected result if material | What would weaken or falsify it |
|---|---|---|---|
| Historical gradient mixing blocks consolidation | Old versus repaired loss on identical real supplied-credit fixtures, then matched on-policy training arms | Correct local score gradients; increased correct-branch probability and stable task success | Real gradients already align locally, or repaired arms retain the same branch distribution under adequate matched evidence |
| Positive episode credit rewards forbidden local decisions | First audit signs on forbidden-call spans; then adjust turn weight or separately scored guard costs with frozen other settings | Fewer positive labels on forbidden actions, improved guard compliance conditional on attempted positive goals | Forbidden spans already carry negative labels reliably, or compliance does not respond despite effective local updates |
| Partial-score plateau is a reward problem | Fixed-positive-denominator progress plus explicit newly broken-guard cost/full-completion bonus, retaining unmodified task-success reporting | Complete branches replace high-partial unsafe branches | Full success does not improve, or apparent guard improvement is entirely abstention/no-action behavior |
| Too few comparable siblings weaken local evidence | More attempts per task, fewer distinct tasks per update at equal sampled-episode budget | Lower singleton fraction, more useful late-turn comparisons, improved reliable procedure execution | Singleton/informative coverage does not change, or it changes without task benefit |
| Missing procedure access is the bottleneck | Provide only the real relevant procedure text versus ordinary discovery, keeping model/reward fixed | Conditional action success rises substantially | Failures persist after grounded procedure access, implicating reasoning/state binding or credit rather than discovery alone |
| Cross-task interference causes forgetting | Fixed-context checkpoint repeats plus gradients/updates from relevant versus other task groups | Measurable correct-branch likelihood decline after other-task updates; balanced rehearsal reduces it | Similar failure mixtures already exist before those updates, or no reproducible probability decline occurs |

An explicit guard cost should be tested rather than chosen as an arbitrarily
large universal penalty. It must not give reward merely for doing nothing:
report positive-goal completion, full success and guard compliance jointly.
Likewise, a full-success bonus can increase variance without improving support;
its success must be measured rather than presumed. Increasing turn weight can
amplify noisy or mixed-call labels. More siblings trades prompt coverage for
comparison density and must preserve same-step task uniqueness.

For single-call attribution, executing parallel calls serially is not enough:
they still came from one assistant turn and may retain one combined reward.
A one-call-per-turn policy or qualified call-addressed evidence would change
the credit unit and possibly the turn/model-call budget. Treat it as a separate
documented experiment, not a free recipe switch or a reason to retroactively
claim the existing labels were per-call.

Success means a previously intermittent correct branch becomes reliably
preferred across fresh native rollouts without more forbidden actions or
abstention, then holds across later checkpoints and additional seeds. A higher
partial mean or a single impressive rollout is insufficient. Rank and optimizer
schedule comparisons remain worthwhile alternatives after the credit/reward
audit; this evidence does not prove those factors irrelevant.

### Independent review of the episode-credit conflict audit

`credit_conflict.py` now supplies a measured proxy for one element of the
integrated hypothesis. I independently recomputed every nonempty run's
numerator/denominator from native captures and reproduced all results:
guard-violating episodes above their own native group mean are1810/3858 in the
original run,1134/2245 and1015/2052 in its continuations,2792/5420 in FP16 base,
1630/3217 in higher LR,1612/3276 in H100, and2699/5735 in FP16 continuation.
The interrupted attempt reproduces105/215. No included group has nonfinite
reward in this capture.

The audit restricts to complete same-task/same-step non-Simple groups with no
truncated sibling. This removes the selected SAMPO truncation-shaping confound;
DAPO overlong shaping is not selected for SAMPO. Native episode reward already
contains configured mistake penalties. The centered-native value is therefore
a meaningful **pre-admission episode-credit proxy**. It is not the applied
combined token advantage: turn credit, sampler weighting, group admission,
surplus trimming and the differentiated objective remain separate.

Checked native examples with no truncation or recorded harness error:

- Higher-LR step1 `sales.email_zoom_fuzzy`, episode
  `2b9a851d48cc44d4b278dd1e405240f5`: native reward0.666667, group mean0.144444,
  centered proxy+0.522222 while a prohibited Zoom registrant exists.
- FP16 base step1, same task, episode `8d8212f7134f4fa1a68404878ce70632`:
  reward0.5, group mean0.294444, proxy+0.205556 with the same violated guard.
- FP16 continuation step1 `hr.asana_compliance_tasks`, episode
  `bd104ce40cb5422d9995ba003c31482a`: reward0.48, group mean0.382857,
  proxy+0.097143 despite three forbidden Asana actions.

These establish that violating a guard does not preclude above-mean episode
credit across arms. They do not establish that full success receives lower
reward, that the forbidden call itself had positive combined credit, or that
approximately50% is intrinsically a defective rate. Ordinary relative policy
gradients can positively rank flawed trajectories above worse ones.

There is also an important counterpoint to the small-advantage story: the
rare fully successful higher-LR ads episode `1ee1fe7e66154fa69d61458765377f12`
has native reward0.92 and group mean0.176667, yielding a **large+0.743333**
episode-credit proxy. Its rarity does not mean its episode label was small.
The claim needing further measurement is whether that strong episode signal
reinforced the decisive procedure locally and consistently, rather than
chiefly reinforcing other tokens or being offset by competing updates.

### Final critique of the integrated proposal

The new hypothesis/current-source/proposal sections correctly distinguish the
historical defect, current post13 selection, local anchor changes and unmeasured
quality effects. The reward proposal makes newly broken guards explicit, keeps
positive-goal progress, and requires checking abstention and actual supplied
signs. Those are appropriate protections against misleading partial-score gains.

Four control details need explicit treatment before implementation:

1. A30-update pilot with144 used episodes per update fixes4320 **used** episodes.
   Actual generated candidates vary with reward spread and the adaptive
   controller. Keeping VORTEX unchanged cannot generally fix both actual
   generated counts and used updates identically across arms. Use common
   candidate/probe caps and report actual generation, used data and compute;
   compare equal generated budgets separately if needed. Candidate-cost
   differences can be an outcome of the derivative change.
2. Isolate the historical derivative inside the otherwise identical current
   runtime. Comparing whole post12 and post13 wheels would also change other
   numerical behavior, including KL handling. Holding controller parameters
   and seed fixed does not hold realized adaptive task selections fixed after
   the policies diverge. Report allocations as mediators of the end-to-end
   intervention, or use a separate fixed-task mechanism diagnostic.
3. Define the25% practice reservation over **candidate selections** unless a
   different admission contract is intended. Constant all-success and
   all-failure practice groups are still filtered by SAMPO and may produce
   no update. Record retained practice contribution, candidate cost and
   fallback behavior. Guaranteeing an optimizer-used practice quota would
   require an explicit change, not mere resampling.
4. Historical exact supplied token advantages and frozen trainer scores were
   not retained in this analysis. A real-trajectory comparison may use exact
   native context/token/mask evidence and explicitly recomputed credit under
   a versioned anchor/credit definition, then hold that fixture identical
   between derivative branches. It must not claim an exact reconstruction of
   the historical applied update.

The proposal should remain a staged decision process. Current source inclusion
is not proof of deployment; a qualified image must resolve immutable versions.
An unchanged recipe need not mean unchanged sampled data once feedback changes.
Passing local gradient fixtures does not establish improved full procedure
success; that is the purpose of the subsequent controlled pilot and repeated
checkpoint tests.

### Offline replay review (2026-10-02)

The reward sweep preserves native sibling groups, excludes incomplete and
truncated groups, and centers candidate rewards inside each group. Its
mixed-safety fraction counts only violating episodes with a safe sibling;
its all-violating-episode fraction has a different denominator. The latter
remains essential: a uniform binary violation penalty cancels exactly when
every sibling violates a guard. Positive affine credit rescaling preserves
that problem and every advantage sign. These are reward-credit diagnostics,
not evidence that a policy will learn new branches.

The selected conservative counterfactual is binary cost 1.125, completion
bonus 0, and old-arm mean-absolute-credit scale 0.47867817442098065. Both
selection criteria use older arms only; the four FP16 arms are held out from
selection but are correlated continuation/model histories, not independent
seed replications. In the current script both criteria operate on the same
44-entry refined grid, so a separate historical coarse-grid result is not
reproducible from that script alone. Negative penalized rewards must not be
fed unchanged into a controller whose beta reward-sum model requires [0,1].
The proposed optimizer reward and controller selection statistic therefore
need explicit separate, versioned contracts or a declared affine mapping.

`replay_turn_credit.py` captures 14 purposive complete native groups across
seven substantial arms: the first mixed-safe/violating group and first hard
full-success group, with high-LR ads step 26 forced. Native sampled nodes
retain token IDs and masks, so sampled-mask counts can supplement unit-turn
and JSON-character span proxies. The replay recomputes CURRENT bundle@2
anchors and explicit native turn reward discounted at gamma 0.95; it does
not reproduce historical supplied advantages or optimizer admission. The
local/shared derivative comparison is an analytic ratio-one, unit-importance,
unclipped mean-loss diagnostic without KL or parameter Jacobians. Negative
immediate assertion_progress is not necessarily a negative-guard failure.
These purposive cases cannot establish population prevalence or the learning
effect of an anchor, normalization, weight, or sampler change.

The bounded capture completed: 76 distinct native episodes, 476 sampled turns,
and 298 singleton bundle@2 anchors (62.6%). All captured graphs are linear
parent chains and all sampled nodes align exactly to assistant-turn assessment
IDs. Native sampled masks are retained, but sampled nodes' supplied advantages,
trainer log-probs and loss weights are null. Native sampler log-probs alone do
not recover the optimizer gradient. Full/compact identity, reward and assessment
equality checks pass; the checked-in evidence records raw-cache and source hashes.

The 76 episodes contain no negative native assertion_progress component, so
their negative-progress/positive-credit fraction has denominator zero. This is
purposive sample absence: scanning all stored components in the seven substantial
arms finds negative counts 865/78567, 599/44385, 352/37721, 1375/121221,
644/71789, 922/65886 and 1026/102363 respectively. These counts include truncated
candidates. Pinned environment checkout `0bad6187f40e23666150234ade99efd35ac567a0`,
`environments/automationbench_v1/src/automationbench_v1/turn_rewards.py:119`,
defines assertion_progress as current minus previous credit, without clipping.

There are separately 25 turns with negative immediate native turn_reward in
the bounded capture. Seven receive positive combined credit at turn weight 0;
five do at weights 1, 2 and 4, under both mean and mean_std. The five remaining
turns belong to FP16-base onboarding episode
`977c581b028b49848948770f16274d57`, sampled turns 1, 2, 3, 6 and 8. All five
have singleton anchors, zero centered turn credit and mean-centered episode
credit +0.0188888889. With these anchors no global turn-weight increase can
change those five signs. This is a current-code local-penalty example, not a
guard-failure claim or measurement of the historical applied gradient.

At native-mask span weighting, mean normalization gives local/shared-gradient
sign disagreements on 2/395 nonzero-credit turns at weight 1, 11/395 at weight 2
and 10/395 at weight 4. Mean_std gives 16/395, 18/395 and 23/395. Weight 0 gives
zero disagreements, as expected for trajectory-constant advantages. These
counts quantify the specified ratio-one loss fixture, not parameter changes.

Current `adaptive_curriculum.py:679` rejects the entire feedback group if any
finite reward is outside [0,1]. Native rewards can be negative, so a live
invalid-feedback claim requires proving which reward field the controller
actually receives. Variance-based active admission is a separate contract.

### Enlarged bounded turn-credit replay

The replay now adds each arm's first complete, nontruncated hard group with a
negative assertion_progress component in any sibling. This explicitly targeted
extension captures 21 groups, 114 distinct native episodes and 775 sampled
turns. All earlier selection roles remain recorded; high-LR ads step 26 is
still forced. There are 477 singleton anchors (61.5%). The enlarged selection
is deliberately diagnostic and cannot estimate population rates.

Fifteen sampled turns have negative assertion_progress. All 15 have singleton
anchors, and all occur in episodes with a final explicit guard failure. Seven
of the 15 receive positive combined credit at each weight 0, 1, 2 and 4 under
both mean and mean_std. Their relative turn credit is exactly zero. Forty-two
turns separately have negative immediate native turn_reward: 14 receive
positive combined credit at weight 0, and 12 at weights 1, 2 and 4. Of those
42, 32 are singleton anchors; all 12 remaining positive cases are singleton.
Sixteen of the 42 occur in episodes with final guard failures, including seven
positive cases at every weight. End-of-episode guard status is not a per-turn
assertion-causality label; negative tool-mistake rewards remain separate.

One directly inspected concrete case is FP16-base vendor payment approval,
step 2, episode `b668657dab574811be86d2df0714464b`. Sampled turn 3 sends Gmail
to `ar@globallogistics.example.com`, the recipient named by the failed final
`gmail_message_not_sent_to` assertion. Its native assertion_progress is
-0.0333333333, anchor group size 1, turn advantage 0, and combined mean credit
+0.1404761905 at every tested weight. Both native episode reward and task
partial credit are 0.5. High-LR episode `c4e93efcba8a4ccd917ebd35af83d6fb`
similarly sends to that recipient at turn 3 and updates the prohibited row at
turn 4; both have singleton anchors and +0.0238095238 mean combined credit.
The final recipient and row predicates corroborate those actual actions.
This establishes a present-reconstruction credit conflict on actual training
actions, without claiming these were the historical applied advantages.

The enlarged native-mask derivative fixture has 694 nonzero-credit turns.
Mean normalization gives 9, 23 and 25 local/shared sign disagreements at
weights 1, 2 and 4; mean_std gives 31, 34 and 45. Both give zero at weight 0.
Increasing turn weight also increases this historical shared-sign mismatch;
it cannot repair singleton turn credit. Per-episode task partial credit,
full-success flag, native reward and final guard status are now retained in
each replay for separate strategy experiments. Final local-cache rerun passed
all identity/reward/assessment checks; own script passes Ruff format/check.

### Final proposal/report review

Independently checked `simulation-report.md` and `reward-and-mask-design.md`
against current artifacts and scripts. The 44 reward variants cover 44,262
complete, finite, nontruncated non-Simple candidate episodes. The controller
artifact contains 168 allocation simulations and common-task inventories of
100–137 tasks. The local-cost artifact contains 14 sweeps of the same 21 groups
and 114 episodes, and its input hash matches the latest turn-credit artifact.
Its reported 42 cost turns, 198 positive-progress turns, 105 complete-episode
turns and every subtract/override table entry agree with the artifact.

The proposed mask system appropriately distinguishes changed surrogate credit
from an estimator of the original reward, coarse turn support from call-span
attribution, future labeling from earlier policy conditioning, and offline
prepared-credit checks from causal learning. The zero positive-cost-span result
for disjoint override is a sign guarantee by construction; it is not evidence
of improved action probabilities or learning. The recorded fixture does not
rule out a useful and harmful parallel call sharing the same coarse mask.

Suggested standalone prose corrections sent to the parent: qualify the opening
coverage claim as a frozen-bank allocation result, rather than a general law
that maximizing yield sacrifices coverage; remove "pilot described below"
unless that standalone document actually includes the pilot description.
