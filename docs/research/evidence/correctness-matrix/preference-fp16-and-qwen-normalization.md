# FP16 preference updates and Qwen normalization

2026-09-30. Qwen's Torch fallback query/key normalization has a reproducible
FP16 backward defect. Lowering the loss scale does not solve this fixture:
all 135 baseline Qwen attempts skip. A research-only FP32 normalization
control restores 45/45 applied updates, with 42 passing the original strict
gates. This is stronger evidence of incorrect numerical behavior than a
failure to learn, and it occurs inside the model before the optimizer.

The retained evidence (`preference-fp16-and-qwen-normalization/summary.json`, local archive)
contains direct model runs, scaler controls, module traces, independent
normalization grids and a hashed receipt for the locked Transformers wheel.
Across the retained model arms and repeated diagnostics there are357 attempts,
206 applied updates and195 strict passes. Counts include baseline failures,
repeats and controls; they are not357 distinct recipes or fresh task episodes.
The research correction is not installed in production; the runtime adoption
gate remains open. No model was trained in FP64.

## Supported-precision experiments

Each matrix arm executes three SFT steps and three steps for each of fourteen
preference branches, covering fifteen unique loss kinds. DPO names the
ordinary sigmoid branch. Models use FP16 base weights and FP32 rank4/alpha8
q/v LoRA, LR1e-4, dropout0, fixed supplied pairs and AdamW without weight decay.
Branches reset adapters, optimizer and dynamic scaler. These are direct-loss
experiments, not native Trainer, full recipe or task-learning qualification.

The independent scalar loss produces score derivatives. The oracle scales
those coefficients through the actual model Jacobian and divides resulting
FP32 adapter gradients by the same scale used by backward. A second control
uses the trainer's exact scaled logit derivative through the model. This
distinguishes score arithmetic from model backward and optimizer behavior.
Both BF16 SFT/DPO controls pass all six updates, twelve in total, after the
scaler extension. Original gates remain value error1e-6, parameter relative
error1e-4, finite gradients and nonzero movement.

| Model | Initial scale | Attempts | Applied | Strict passes |
| --- | ---: | ---: | ---: | ---: |
| Qwen0.8B FP16 | 1024 | 45 | 0 | 0 |
| Qwen0.8B FP16 | 128 | 45 | 0 | 0 |
| Qwen0.8B FP16 | 1 | 45 | 0 | 0 |
| LFM1.2B FP16 | 1024 | 45 | 41 | 40 |
| LFM1.2B FP16 | 128 | 45 | 45 | 41 |
| LFM1.2B FP16 | 1 | 45 | 45 | 44 |
| Qwen0.8B FP16, FP32 norm control | 1 | 45 | 45 | 42 |

Qwen's unscaled loss and logit gradients are finite, but model gradients are
nonfinite even at scale1 and subsequent backoffs. LFM has four overflow skips
at1024: three SPPO-hard attempts and the first IPO attempt. Lower-scale
controls apply all45 steps. Remaining LFM strict misses are loss values,
with matching independent parameter gradients. This does not select a
universal scale or justify relaxing operation-specific FP16 restrictions.
Peak allocation remains below3.82 GiB on Qwen and2.76 GiB on LFM.

## Independent normalization math

For a vector x, let s=sum(x²)+epsilon and y=x/sqrt(s). For an incoming
cotangent v, the derivative is

`grad_x = v/sqrt(s) - x*dot(x,v)/s^(3/2)`.

At x=0 and epsilon1e-6 this is exactly1000*v, finite for the tested v.
The fallback graph instead evaluates reciprocal square root in FP16.
The derivative of rsqrt(s) at epsilon is -5e8, beyond FP16's finite range.
At zero, an intermediate zero-times-infinity produces NaN; at small nonzero
vectors the graph can yield infinities although the final mathematical
derivative is finite. Large vectors also overflow the half-precision sum
of squares. A larger loss scale cannot repair these operations.

The independent grid uses quantized inputs and cotangents, Python scalar
norms and analytical Jacobian-vector products, not another Torch copy of
the normalization graph. It covers FP16/BF16, widths4/128 and magnitudes
0,1e-4,.01,1,100,1000. Predeclared relative gates are .002 for FP16 and .02
for BF16, with denominator floor1. Each48-case grid includes both ordinary
and FP32-work variants. Eight ordinary FP16 cases fail; all24 FP32-work
cases and all ordinary BF16 controls pass.

The result repeats for the installed research Transformers5.16.1 helper
and for the actual locked5.14.1 wheel's helper, isolated without replacing
the environment. Its source hash and verified PyPI wheel hash are retained
in the source receipt. The source-only extraction reports the executing
runtime version5.16.1 separately; it is not a5.14.1 package integration run.

## Localizing and correcting model backward

Output-gradient traces record273 module outputs per SFT/DPO step.
Before the control, the first recorded nonfinite output gradient is
`layers.22.linear_attn.in_proj_qkv`;245 traced outputs become nonfinite.
The gated branch projections immediately preceding that boundary remain
finite. All six baseline steps skip. With FP32 normalization work, all
six matched traced updates pass and every recorded output gradient is finite.
An additional baseline trace distinguishes token positions: the first qkv
boundary has five nonfinite padding/two nonpadding tokens on SFT and seven
padding/four nonpadding tokens on DPO. Nonfinite model intermediates can thus
poison backward even when those padding positions have no loss credit.
These output hooks localize a boundary; the isolated scalar reproduction
and controlled substitution establish the normalization defect.

The ablation promotes half inputs before normalization and returns normalized
values in the original dtype. Base weights remain FP16. This deliberately
small control fixes the failure without replacing the entire gated-delta
implementation. Current upstream Transformers source (`https://github.com/huggingface/transformers/blob/d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a/src/transformers/models/qwen3_5/modeling_qwen3_5.py`, local archive)
promotes query/key tensors to FP32 before normalizing in its chunked and
recurrent fallbacks. Its implementation has additional changes; the
research substitution is not qualification of that complete upstream commit.

## Residual gates and evidence limits

The corrected45-step Qwen arm has two absolute value misses, IPO/SPPO-hard,
and one AOT parameter-gradient relative miss .00150097. Repeat AOT controls
with finite-difference h1e-4 and1e-6 reproduce the same mismatch. Raw scores
are retained; their sorted gaps are far larger than the perturbations, so
crossing a sorting tie does not explain it.

On that AOT step, actual raw logit gradients match the independently computed
double-probability gradient rounded to FP16 exactly. The FP32 probability
oracle differs in two elements by5.96046e-8, one FP16 subnormal increment;
the exact coded-logit model VJP matches parameter gradients exactly. This
supports rounding sensitivity in the model-Jacobian comparison, rather than
another demonstrated incorrect AOT formula. Keep the original failure visible.

Raw older model artifacts contain NaN for undefined gradient comparisons;
curation encodes these as null, records affected fields and preserves original
artifact hashes, skipped flags and failures. Some baseline arms predate runner
hash instrumentation. Their model/renderer/DPO hashes are present, but exact
intermediate harness snapshots were not retained. Later trace and control
artifacts include the runner hash. No failed comparison is encoded as zero.

## Reproduce and adopt safely

Run from `/home/hammad/projects/rl` with research Python
`/home/hammad/projects/trl-gdpo-capo/.venv/bin/python` and the same renderer,
isolated TRL, PEFT and renderer-dependency PYTHONPATH used by the prior
preference report. Run `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/short_preference_run.py --model
qwen08 --dtype float16 --initial-scale 1 --two-pairs --loss-types dpo hinge
ipo sigmoid_norm exo_pair nca_pair robust bco_pair sppo_hard aot aot_unpaired
apo_zero apo_down discopop --output PATH`; repeat for LFM and scales128/1024.
Add `--stable-qk-norm` for the Qwen research control and `--trace-backward`
on the short SFT/DPO arm. `--score-fd-step` controls the AOT oracle sensitivity
check. Execute GPU experiments serially.

Run `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/qwen_qk_normalization_grid.py --output PATH` for the
CPU grid. `--source PATH` extracts only the trusted source's l2norm function
without installing that package. The locked wheel receipt supplies its
immutable URL/hash; the package install and production lock remain unchanged.

The next adoption step must select a released or maintained immutable source
with FP32 query/key normalization, respect current Transformers compatibility
constraints and validate native policy/preference, fused fallback selection,
padded and long sequences, save/recovery and runtime image installation.
Do not hide the bug by training in FP64 or choose a loss scale from this small
fixture. Framework preference/SFT still deliberately reject the online-RL-only
FP16 option; these experiments do not change that public support contract.
