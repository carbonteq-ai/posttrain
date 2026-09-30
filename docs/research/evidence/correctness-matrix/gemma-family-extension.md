# Gemma family correctness coverage

2026-09-30. Gemma is now included in the ongoing multi-architecture audit.
The cached immutable Gemma4 testing checkpoint passes all seven native
policy branches in BF16 and FP16, including SAMPO tool masks and accumulated
optimizer gradients. Fifteen direct preference/SFT branches also have both
precision results. This is architecture evidence; pretrained Gemma coverage
remains a separate, explicitly open gate.

## Architecture and checkpoint boundaries

| Family | Current checkpoint | Architecture exercised |
| --- | --- | --- |
| Qwen | Qwen3.5-0.8B, existing immutable fixture | Linear and full attention |
| LFM | LFM2.5-1.2B-Thinking, existing immutable fixture | Convolution and full attention |
| Gemma | tiny-Gemma4ForConditionalGeneration, testing checkpoint | Sliding and full attention, per-layer input embeddings, logit softcap and text path of multimodal loader |

The Gemma checkpoint is `trl-internal-testing/tiny-Gemma4ForConditionalGeneration`
at `0dc1746b7f9f623b748e735ed5a4302eb4baf346`. It has two text layers,
hidden width16, vocabulary262144 and about25.4MB of stored weights. Its
manifest (`gemma-family-extension/checkpoint-manifest.json`, local archive) records the complete
config and verified cached weight hash. The full multimodal model is loaded,
but these experiments supply text only. LoRA targets language-model q/v
projections; unused vision/audio projections do not enter gradient comparisons.
Images, audio and pretrained-scale behavior are not qualified here.

Google Gemma3-270M-it is a fitting pretrained candidate: its immutable
`ac82b4e820549b854eebf28ce6dedaf9fdfa17b3` weights occupy536,223,056 bytes.
The actual download returned gated401; no credential retries were attempted.
User input was requested for an authenticated local session or downloaded
checkpoint. Gemma4-E2B-it has10,246,621,918 bytes of full weights, so it cannot
fit unquantized on this8GB GPU. Do not silently substitute a tiny fixture or
quantization and call that pretrained BF16/FP16 qualification.

## Results

The aggregate (`gemma-family-extension/summary.json`, local archive) links all retained artifacts:
144 optimizer attempts,143 applied updates and140 strict step passes,
including the initial smoke and lower-scale follow-up. Native runs additionally
check84 microbatch gradients. These are supplied-data numerical experiments,
not fresh task samples or learning curves.

| Native objective | BF16, three updates | FP16 scale128, three updates |
| --- | --- | --- |
| SAMPO | Pass | Pass |
| GRPO | Pass | Pass |
| GSPO | Pass | Pass |
| DAPO loss branch | Pass | Pass |
| Dr. GRPO | Pass | Pass |
| BNPO | Pass | Pass |
| LUSPO | Pass | Pass |

All14 native arms apply3/3 finite, nonzero updates. Independent parameter
gradient and scaler-aware optimizer comparisons match exactly. Excluded
tool/padding score gradients remain zero. Peak allocation is0.58GiB.
The native fixture uses microbatch1/accumulation2, one supplied multi-turn
group, three updates, q/v LoRA rank4/alpha8, LR1e-4, beta.01 and the earlier
diagnostic clipping bounds. Only SAMPO uses opposing token-local credit;
other branches use independently checked scalar reward normalization.
The DAPO loss branch is not qualification of its complete sampling recipe.

The renderer initially rejected the old synthetic trace because Gemma tool
observations must follow a matching assistant tool call. The Gemma fixture
now includes that explicit call and matching tool name/id. Qwen/LFM fixtures
retain their original serialization. This was an invalid test trace, not a
trainer math defect. Injected tool content remains outside sampled targets.

| Direct15-branch matrix | Attempts | Applied | Strict passes |
| --- | ---: | ---: | ---: |
| BF16 | 45 | 45 | 44 |
| FP16, scale128 | 45 | 44 | 43 |

Branches are SFT, ordinary sigmoid DPO, hinge, IPO, sigmoid_norm, EXO-pair,
NCA-pair, robust, BCO-pair, SPPO-hard, AOT, AOT-unpaired, APO-zero, APO-down
and DiscoPOP. Each branch resets adapters and Adam, uses two fixed preference
pairs, and compares independently derived loss/score derivatives through
the actual model Jacobian. Peak allocation is1.50GiB. Gates are unchanged:
absolute loss1e-6, relative parameter gradient1e-4, finite gradients and movement.

BF16's sole miss is SPPO-hard step3 loss error1.006e-6; its parameter gradients
match. FP16 skips SPPO-hard's first attempt at scale128 and later misses its
value gate by2.254e-6. A scale64 control applies all three SPPO-hard updates,
with one value miss2.254e-6 and matching parameter gradients. Its SFT controls
also pass. Preserve the original failed arm; this control is not a universal
scale recommendation. No other matrix branch has a parameter-gradient miss.

## Integration and remaining gates

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/model_fixtures.py` centralizes research checkpoint identity,
organization, renderer, auto factory and LoRA targeting. Direct preference and
native policy harnesses use that seam. Existing Qwen/LFM IDs, factories,
renderer settings and targets are preserved and checked. This does not add a
public Gemma algorithm selection or change the production catalog or pins.

Reproduce from `/home/hammad/projects/rl` with the established research Python
and renderer/TRL/PEFT dependency paths. Run `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/short_preference_run.py --model
gemma4tiny --two-pairs --dtype bfloat16 --loss-types dpo hinge ipo sigmoid_norm
exo_pair nca_pair robust bco_pair sppo_hard aot aot_unpaired apo_zero apo_down
discopop --output PATH`, then repeat FP16 with `--initial-scale 128`.
Run `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_multiturn_run.py --model gemma4tiny
--dtype bfloat16 --objective sampo --output PATH`; repeat the seven objective
choices in the table and FP16 scale128. Execute GPU arms serially. Runtime
setup remains research infrastructure, not a published production release.

Open Gemma gates: accessible pretrained model, model-sized numerical stress,
intended contexts/modules, fresh environment trajectories, held-out behavior,
family-specific extreme-logit/normalization boundaries and production runtime
qualification. The ongoing broad audit remains active; passing tiny architecture
fixtures is useful evidence but does not close those gates.
