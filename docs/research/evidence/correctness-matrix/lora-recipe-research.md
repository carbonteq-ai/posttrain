# LoRA recipe research: rank, alpha, learning rate, and precision

Research date: 2026-09-30. This is a candidate-selection note, not a claim that
the local models have a validated optimal recipe. No training selection, runtime,
or source implementation was changed for this research.

The first candidate should remain ordinary LoRA with explicit initialization,
all-linear coverage where memory permits, zero dropout, and an LR sweep measured
in actual policy movement. Rank changes must specify an alpha rule. Increasing
rank while keeping alpha/r constant can require reducing LR; that is different
from the fixed-alpha experiments behind common rank-independent LR advice.
LoRA+, rsLoRA, and newer normalized adapters are useful comparison arms, but the
available evidence does not establish one as the winner for SAMPO tool agents.

## What our inspected configurations actually do

Repository snapshot: `b59e0e136119058014851a8c853e567cafbe94f4`.

| Inspected selection | Rank / alpha / multiplier | Coverage | LR and update context |
| --- | --- | --- | --- |
| `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_multiturn_run.py` | 4 / 8 / 2 | q_proj, v_proj | 1e-4; two examples, accumulation 2, three iterations on the frozen fixture; controlled clipping thresholds 0.003/0.004 |
| `apps/lab/.posttrain/catalog/precision-qualification.yaml`, LFM2.5-2.6B SAMPO `...r4-lr6e-5...` | 4 / 8 / 2 | all-linear | 6e-5; 24 prompts × 6 generations, accumulation 144, constant schedule, beta 0.01 |

The catalog binding requests FP16, float32 logits, and initial loss scale 1024.
It is checked-in configuration, not evidence that a live job currently uses it.
The smaller harness uses Qwen0.8B/LFM1.2B, different coverage, much shorter traces,
different batch structure, and a diagnostic loss regime. Its LR cannot establish
that the LFM2.6B production selection is too high or too low.

`packages/train/src/posttrain/train/backends/trl/common.py` constructs PEFT
`LoraConfig` from rank, alpha, dropout, and targets. It does not select rsLoRA or
LoRA+ here. Actual research PEFT 0.21.1 `layer.py` sets scale to alpha/r, or
alpha/sqrt(r) for rsLoRA; default A is Kaiming-uniform and B is zero. Its
`optimizers/loraplus.py` puts A at the supplied LR and B at LR × ratio (also
handles embeddings and one-dimensional parameters). A factor-ratio experiment
must inspect the optimizer groups, not merely set a field. These are the actual
implementations, with public API context in [PEFT LoRA documentation](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora).

Inspected isolated implementation hashes:

```text
PEFT 0.21.1 peft/tuners/lora/layer.py
9a2688af69fd16cef94fdb08ab0e973b37d7b2a672e64ba2057a01df7e98b7d8
PEFT 0.21.1 peft/optimizers/loraplus.py
435d46aca87a2648224c2bdf8e54e5af5513ab1a6728e263a1e4e84e7057077b
```

## Primary evidence and its limits

**Original LoRA (Hu et al., 2021; ICLR 2022).** Establishes the low-rank
parameterization and zero-initialized branch. It is foundational evidence for
the adapter mechanism, not proof of a modern online-RL learning rate or a
rank-independent policy trust region. [Paper, arXiv 2106.09685](https://arxiv.org/abs/2106.09685).

**rsLoRA (Kalajdzievski, 2023, v1 inspected).** Replaces alpha/r with
alpha/sqrt(r). Llama2/OpenOrca experiments use 20,000 examples, AdamW LR 5e-5,
all-linear adapters, and ranks 4, 8, 32, 128, 512, 2048. Gradient-norm and
perplexity comparisons favor stabilized scaling at large ranks; ablations cover
SGD, initialization-only rescaling, attention-only coverage, and an LR sweep at
rank 4. This supports testing a rank-aware multiplier, not forcing large ranks
for a small tool-agent RL task. The theorem concerns asymptotic adapter signal
stability; it does not give a universal finite-rank AdamW LR. [Paper revision](https://arxiv.org/html/2312.03732v1).

**LoRA+ (Hayou, Ghosh, Yu, 2024, v1 inspected).** Uses a larger B-factor LR.
Llama7B Flan-v2 tests use r64/alpha16, all-linear coverage, BF16, AdamW epsilon
1e-6, and two-seed comparisons: best asymmetric rates improve MMLU by about
1.3 percentage points over the best equal rates. FP16 MNLI uses r8/alpha16 and
finds equal rates nearly optimal. Thus the favorable ratio is task dependent;
BF16/FP16 were both exercised, but there is no SAMPO or Qwen3.5/LFM qualification.
The published grid reaches B LR 2e-3 for Flan-v2. It is not evidence to multiply
our current shared LR by that ratio without controlling initial movement.
[Paper](https://arxiv.org/html/2402.12354v1), [authors' implementation](https://github.com/nikhil-ghosh-berkeley/loraplus).

**Thinking Machines, LoRA Without Regret (2025-09-29).** Sweeps ranks 1–512
and LR, including math RL. Uses fixed PEFT alpha32, random A/zero B, equal factor
rates. Optimal LR varies by less than 2× between r4 and r512; LoRA matches full
finetuning even at low RL ranks. The all-linear/MLP placement ablation beats
attention-only coverage even at matched parameter budgets. These findings
support low rank and broad coverage, not our q/v-only diagnostic as a sufficient
learning recipe. The reported approximately 10× LoRA/full-FT LR relation is
empirical. The article exposes Adam parameterization invariances but does not
provide a pinned executable training recipe for our architectures.
[Primary research article](https://thinkingmachines.ai/blog/lora/).

**Maximal-Update Adaptation, μA (2026-02, v1 inspected).** Distinguishes the
full BA multiplier from PEFT alpha. For random A/zero B, equal factor rates and
Adam-like SignSGD, fixed multiplier requires LR proportional r^-1/2; multiplier
1/r makes LR rank-independent; multiplier 1/sqrt(r) implies r^-1/4. Assumptions
include bounded signals, single-sample analysis, and no Adam momentum. Its RLVR
ablation uses Llama3.1-8B, 500 GSM8K training/50 test problems, group 8,
temperature 1, length 1024, DAPO loss, no KL, and zero weight decay. It uses
BF16 mixed precision, decoder MLP-only coverage, and Kaiming-normal A variance
1/n, versus PEFT's default uniform variance 1/(3n). Rank/LR sweeps support the
trends, but initialization, coverage, the small split, and different objective
limit transfer. Its mapping is a hypothesis, not a finite-rank AdamW law.
[Paper revision](https://arxiv.org/html/2602.06204v1).

**LoRA-alpha (2026-06, v1 inspected).** Reports alpha approximately C sqrt(r)
with C near 256 in its Tulu3 calibration, and compares alpha scaling to LR
scaling. Its 5-layer synthetic MLP ablation shows LR128× divergence while
alpha256× remains stable. This is strong motivation to distinguish the knobs;
it does not establish that alpha512 at r4 is safe for our FP16 tool training.
The paper also includes reasoning-RL comparisons. Without reproducing the
implementation and precision behavior, this remains a later challenger arm.
[Paper revision](https://arxiv.org/html/2606.12883v1).

**NoRA (authors' project page inspected 2026-09-30).** Normalizes A across the
rank dimension. Reports DeepSeek-R1-Distill-Qwen1.5B DAPO benchmark average
41.0 base, 42.8 LoRA, 44.4 NoRA; spectral-init baselines degrade badly. This
motivates checking factor geometry. It does not supply a verified universal
advantage for our models, precision, or tool tasks. The code is published, but
no immutable revision or execution was qualified here, so it is not selected
as the default. [Authors' results](https://spherelab.ai/NoRA/), [code](https://github.com/Joluck/NoRA).

## Independent mathematics: what can actually be normalized

Write the trainable change as D = s BA, where s = alpha_PEFT/r. With the
loss derivative G = dL/dD, the exact factor derivatives are

```text
dL/dB = s G A^T
dL/dA = s B^T G
delta_D = s(delta_B A + B delta_A + delta_B delta_A).
```

At zero B initialization, the first A gradient is zero. For plain SGD the first
effective matrix update is exactly -eta_B s^2 G A^T A. For Adam's first step
without weight decay, delta_B is -eta_B g_B/(abs(g_B)+epsilon), after moment
bias correction. Its scale is approximately linear in s, not s^2, when epsilon
is negligible; gradient direction, rank correlations, clipping, and factor
initialization still matter. This derivation explains why “effective LR equals
LR × alpha/r” is not a general equivalence between recipes.

At fixed width and the μA assumptions, a useful multi-step comparison coordinate
is eta sqrt(s r) = eta sqrt(alpha_PEFT), while holding initialization, factor-rate
ratio, optimizer, and coverage fixed. This is a hypothesis about the leading
update term, not a KL estimate or an exact optimizer invariance.

| Rank change from our r4/alpha8/LR1e-4 arm | Multiplier s | μA-matched candidate LR | Reason |
| --- | --- | --- | --- |
| r16, alpha8 | 0.5 | 1e-4 | Fixed PEFT alpha absorbs rank dependence |
| r16, alpha32 | 2 | 5e-5 | Keeping alpha/r at 2 changes rank contribution; fourfold rank predicts half LR |
| r16, rsLoRA alpha4, anchored to r4 rsLoRA alpha4 with s=2 | 1 | approximately 7.07e-5 | Fixed rsLoRA alpha predicts r^-1/4 |

The rsLoRA anchor is constructed to equal the ordinary baseline multiplier at
r4. Merely enabling rsLoRA with alpha8 at r4 doubles that multiplier and
confounds the comparison. Candidate rates need an LR bracket, not just one
theory-derived point. Changing q/v to all-linear changes the total function
update and clipping population; no rank-only conversion makes those equivalent.

## Candidate selection and experiments on the 8 GB GPU

Choose an ordinary-LoRA reference first: random A/zero B, r4/alpha8, dropout0,
AdamW with FP32 adapters/moments where supported, weight decay0, and recorded
optimizer epsilon/betas. Use BF16 and FP16 as separate qualification arms.
FP32 is a numerical oracle only. Our current FP16 Qwen issue must not be
declared fixed by shrinking LR: overflow can occur during scaled backward
before the optimizer uses LR.

Run paired two-to-three-step tests initially, serially by model/precision:

1. Freeze the same rendered token traces, masks, credits, old-policy logprobs,
   and KL reference. Compare the current q/v arm with all-linear r4/alpha8
   at LR {3e-5, 6e-5, 1e-4}. Report trainable modules and parameter count.
2. With coverage fixed, compare r4/alpha8 against r16/alpha8 and r16/alpha32;
   bracket each predicted LR by 0.5× and 2×. Nest shared A rows when possible
   and use multiple seeds, because different ranks otherwise change features.
3. Test LoRA+ ratios 4 and 16 by keeping B LR equal to the winning shared LR
   and reducing A LR. This matches initial movement more closely than keeping
   A LR and multiplying B LR; test that second interpretation separately.
4. Compare the rsLoRA anchored pair from the table. Add high-alpha or NoRA only
   after the ordinary arms have measured drift and stable precision behavior.
5. Take surviving arms to fresh bounded AutomationBench groups, recording
   task/tool errors, truncation, reward variance, and retained groups. A
   controlled-gradient pass does not demonstrate task learning; three fresh
   steps only screen recipes, so any performance selection needs repeated
   seeds and held-out tasks with matched rollout-token budget.

For every applied update measure per-layer A/B norms, unscaled gradients,
Adam step norms, effective D increment, sampled-token delta logp distribution,
geometric sequence ratio, positive/negative-credit clipping, KL versus both
behavior and reference, loss-scale/skipped updates, and peak VRAM. Use fixed
probe prefixes for actual full-vocabulary KL where feasible. Fresh sampled KL
estimates can be noisy or signed, so their interpretation must specify the
sampling distribution and estimator. Select the smallest stable recipe that
improves held-out behavior; clipping frequency alone is not the target.

Unresolved: none of these publications validates Qwen3.5's hybrid attention
or LFM's architecture in this exact agentic setting. A paper can justify a
candidate and its control experiment. It cannot replace our numerical and
behavioral qualification or establish that the active poor run is caused by
rank, alpha, or LR.
