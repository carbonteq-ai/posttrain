# Recipe candidate for the correctness campaign

2026-09-30. Choose **two optimizer updates over one frozen rollout group** as
the first SAMPO schedule candidate, against a one-update control. This is an
experiment selection, not a validated production winner. The objective is to
learn from useful trajectories more effectively while preserving correct
credit and measuring the resulting policy movement.

Three authorized research reviews checked primary papers, author code and
ablations: [update schedules](update-recipe-research.md),
[LoRA rank/alpha/LR](lora-recipe-research.md), and
[BF16/FP16](precision-recipe-research.md). Their evidence and transfer limits
are recorded separately so future recipe changes have an inspectable rationale.

## Candidate and controls

Keep ordinary LoRA, rank4/alpha8, dropout0, FP32 trainable adapters and Adam
state. Hold target modules, initialization, LR, reward projection, reference
identity, KL coefficient/convention, masks and truncation policy fixed when
comparing one versus two updates. Freeze behavior logprobs throughout both
passes. Use microbatch1; accumulation defines the complete optimizer batch,
not extra updates. First run controlled traces, then 2 prompt groups × 4
episodes with fresh tasks. Compare equal fresh-rollout budgets and equal
applied optimizer-step budgets separately, across at least three seeds and
held-out tasks. A half-LR two-update arm follows the fixed-LR comparison.

Use BF16 as the passing control. Qualify matched FP16 training/rollout as a
separate candidate; measure scale history and skipped updates. LFM's short
FP16 fixture passes at initial scale1024. Qwen skips three of three at1024;
scale128 applies updates but retains one strict gradient miss. Lowering LR
does not prevent scaled-backward overflow before the optimizer uses LR.
FP64 loss-only arithmetic remains a diagnostic, not the selected recipe.

The learning candidate should use the intended module coverage. Current
production selections use all-linear while our small correctness fixture uses
q/v adapters; qualify all-linear independently before transferring results.
Keep coverage fixed within schedule comparisons. For later rank comparisons,
the μA hypothesis maps r4/alpha8 to r16/alpha8 at the same LR, or r16/alpha32
at half LR. This depends on optimizer, initialization and coverage; measure
merged adapter increments and policy drift rather than treating the formula
as an exact normalization.

## Why this candidate

[ARLArena/SAMPO](https://arxiv.org/html/2602.21534v3) is closest to multi-turn
tool agents. Its released actor takes steps over turn minibatches, so one epoch
can contain several updates. Our accumulated full episodes differ in both
schedule and weighting. Two whole-group updates isolate reuse without also
changing the objective to turn rows. The paper and script disagree on some
clipping/normalization settings, preventing an exact copied recipe claim.

[GSPO](https://arxiv.org/html/2507.18071v2) motivates sequence control and
token-local credit; it does not establish an ideal reuse count for our LoRA
models. The repaired SAMPO gradient must stay in the control. Clipping becoming
nonzero is a mechanics check, not a learning target.

[DAPO](https://arxiv.org/html/2503.14476v2) has useful progressive ablations,
but its single-turn math setting does not justify rejecting a multi-turn group
solely for equal final rewards. Audit actual valid-token advantages in rejected
groups before changing filtering. Truncation, tool syntax, tool execution and
task assertions must be reported separately.

[Precision-RL](https://arxiv.org/html/2510.26788v1) supports testing matched
FP16 engines. Its paper and script use different ranks; it supplies no universal
initial loss scale. [μA](https://arxiv.org/html/2602.06204v1) and
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) explain why
rank/alpha conventions matter. Neither directly qualifies these model families,
our initialization or SAMPO tool learning. Test ordinary LoRA first, then LoRA+
or rsLoRA as separate challenger arms.

## Acceptance and deployment boundary

The controlled one/two-update BF16 probes both pass for Qwen0.8B and LFM1.2B.
Each model's paired arms have identical input hashes and identical recorded
first-step gradient/update controls. Qwen's second update has a 29.17%
clip-region microbatch average; LFM's has 58.52%. Accumulated independent
gradient relative errors are at most 9.89e-8 and zero, respectively. All
excluded-token score gradients are zero. These supplied traces and synthetic
opposing turn credits demonstrate working mechanics, not improved task success.
One-update native TRL omits old scores and uses current detached scores; the
independent oracle accounts for that. Two-update TRL freezes old scores.

Artifacts: Qwen one (`qwen08-recipe-one-bf16.json`, local archive),
Qwen two (`qwen08-recipe-two-bf16.json`, local archive),
LFM one (`lfm12-recipe-one-bf16.json`, local archive),
LFM two (`lfm12-recipe-two-bf16.json`, local archive). All use the published isolated TRL
candidate `9f0825046ae3509a6be804d74a93fb89d5dc695e` and renderer
`1aafe24595a7f2d2f31d24afb4b1bb7a6c6dd076`; JSON retains source hashes.
The runner was formatted between the BF16 arms; its CLI layout changed only,
so runner hashes differ while the scientific configuration remains fixed.
The initial one-update probe failed in the harness because it assumed old
scores were always present. Handling native current-detached scores repaired
the oracle; no trainer source was changed for this schedule comparison.

LFM FP16 also passes both arms at initial scale1024, with no skipped updates,
matching first-step controls and zero independent scaler-aware optimizer
gradient error. Second-update clipping is 58.52%. The unscaled reference differs
by 0.1469% at that update; the scaled oracle matches exactly, confirming why
half-precision controls must reproduce actual loss scaling. Evidence:
LFM FP16 one (`lfm12-recipe-one-fp16.json`, local archive) and
LFM FP16 two (`lfm12-recipe-two-fp16.json`, local archive). Qwen FP16's earlier failed precision
gate remains open; BF16 schedule success does not resolve it.

Require independent loss/score/accumulated-gradient checks, finite applied
updates, unchanged excluded-token gradients, and exact input/seed/source
identities. Then choose using held-out task success, tool errors, truncations,
behavior/reference drift, update cost and rollout-token cost. Count skipped
FP16 attempts separately. Instrumented fixture runtime is not throughput.

`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_multiturn_run.py --iterations 1|2|3` implements
controlled schedule probes only. Framework settings do not currently expose
this schedule; a deployable change needs the public setting, adapter mapping,
buffer/recovery semantics and integration qualification. Production selections
and immutable runtime pins have not changed in this research slice.

Reproduce from `/home/hammad/projects/rl`, serially on the free local GPU:

    PYTHONPATH=/home/hammad/projects/renderers-lfm-mask:/home/hammad/projects/trl-sampo-local-credit:/tmp/trl-math-peft:/tmp/trl-math-renderers-deps:/tmp/posttrain-mathdeps /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_multiturn_run.py --model qwen08 --dtype bfloat16 --iterations 1 --output .posttrain/state/correctness/qwen08-recipe-one-bf16.json

Repeat with `--iterations 2` and a separate output filename, then with model
`lfm12`. Use FP16 only with explicit scale and the precision gates above.
The isolated runtime is Torch2.13.0+cu130, Transformers5.16.1 and PEFT0.21.1.
Temporary dependency paths require reconstruction on another machine; these
probes do not certify the production lockfile.

The full campaign remains active, including native Verifiers collection,
long-context FP16 and other algorithm objectives. A short passing probe or a
paper ablation cannot replace those gates.
