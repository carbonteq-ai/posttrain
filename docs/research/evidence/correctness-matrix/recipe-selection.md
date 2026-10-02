# Recipe candidate for the correctness campaign

Updated2026-10-01. **Use published agentic-RL ablations and author implementations
as the primary basis for the recipe.** Small-card runs qualify implementation
and estimate local numerical effects; they cannot reproduce the learning
evidence of large training runs. The earlier two-update proposal is a bounded
mechanics probe, not a research-established optimal production schedule.

Three authorized research reviews checked primary papers, author code and
ablations: [update schedules](update-recipe-research.md),
[LoRA rank/alpha/LR](lora-recipe-research.md), and
[BF16/FP16](precision-recipe-research.md). Their evidence and transfer limits
are recorded separately so future recipe changes have an inspectable rationale.

## Candidate and controls

### 2026-10-01 correction: research selects; local checks qualify

The user clarified that recipe selection must rely primarily on research and
ablations, not small-card learning comparisons. Apply the following evidence
hierarchy: task-relevant controlled ablations, complete reported recipes and
author code, broader framework examples, then local implementation checks.
Distinguish a measured ablation result from a published setting with no isolated
ablation, and from our transfer assumption. A production-scale local replication
is not a prerequisite for recommending a research-backed recipe.

The recommended direction is SAMPO's sequence-level control with fine-grained
environment credit and its documented agentic training schedule. Preserve
behavior scores across optimizer minibatches, compute credit over complete
groups before splitting, and take an optimizer step after each complete
optimizer minibatch. Accumulation is a memory mechanism, not a substitute for
that schedule. Author turn-row training and our packed episodes have different
ratios/reductions: adopting the former requires an explicit objective amendment,
not silently translating its row counts into episode counts.

Revision 24 (2026-10-02) adds that explicit amendment and `sampo-turns@1`,
selected through `policy_updates.objective_variant: turn-rows`. It combines
turn-wide geometric ratios with equal-turn token-mean reduction and local token
derivatives, retaining complete-group Posttrain SAMPO credit. Episode-wide
`sampo@1` remains unchanged. This targets the inspected ARL-Arena collection/GSPO
kernel shape, not full author-run reproduction: the available script names a
SAMPO estimator absent from the inspected Python registry, and its GSPO kernel
caps log weights at 10 whereas our engine rejects nonfinite uncapped ratios.
Clipping coefficients, credit normalization and filtering retain their separate
provenance and qualification requirements below.

Use group8, gamma.95 and reference beta.01 as author-recipe starting values for
agentic tasks. These are published choices, not independently proven optima for
our LoRA setup. The inspected actor uses one PPO epoch containing multiple
minibatches; no reviewed reuse-count ablation establishes exactly two whole-
population passes. Clipping and advantage normalization remain provenance gaps:
the WebShop script selects .003/.004 and mean normalization, while the paper's
table lists .03/.04 and mean/std normalization. Preserve both sources explicitly
rather than presenting one as an unambiguous paper default.

The research rationale is stronger for sequence control than for stronger KL:
ARLArena's ALFWorld tolerant-clipping ablation reports SAPO success25.16%,
48.05% with KL.05, and76.92% with sequence masking. This is an ablation of
SAPO stabilization, not proof of an optimal SAMPO KL or optimizer batch size.
The paper also finds filtering's format-learning effect depends on credit
design. Retaining equal-return groups with nonzero local credit is a Posttrain
transfer proposal, not an author-validated filtering rule.

| Dimension | Recommended first comparison | Reason and boundary |
| --- | --- | --- |
| Update schedule | One versus two complete optimizer updates per frozen rollout population | Preserve whole-episode SAMPO credit/reduction and freeze behavior scores for both passes. This isolates reuse; it does not reproduce the author's turn-row objective. |
| Group size | Four episodes per prompt for the small pilot; test eight separately | Increase the chance of mixed outcomes without confusing a two-episode numerical fixture with a learning batch. Hold episode and prompt budgets explicit. |
| Admission | Existing episode-spread gate versus a separately defined valid-token-credit gate | Equal final returns can retain nonzero local SAMPO advantages. Record rejected credit before proposing a contract change; include format-invalid and truncated cases separately. |
| LR and LoRA | Keep rank, alpha, scaling, target modules and LR identical initially; then test half LR for two updates | Two Adam steps are not equivalent to one step at twice LR. Compare adapter increments and policy drift; do not normalize by rank alone. |
| Clipping | Retain the selected sequence bounds initially | Do not copy DAPO token bounds or tune LR merely to produce a clipping percentage. Measure useful unclipped credit and post-update behavior drift. |
| Reference KL | Retain the selected coefficient and estimator initially; then compare beta zero against the retained coefficient | Separate policy and penalty gradients. Avoid conflating a changed KL convention with a schedule experiment. |
| Precision | BF16 first; FP16 as a separately measured candidate with explicit scaler history | Preserve raw numerical alerts, but no universal 0.01% full-model gradient threshold establishes a framework defect or a safe training recipe. |
| Response budget | Collection-only budget check before learning comparisons | Determine whether tool syntax and task completion fit. Changing the budget or dropping truncated episodes changes the learning population. |

Fresh primary-source review strengthens the admission concern.
[ARLArena section4.3 and Appendix G](https://arxiv.org/html/2602.21534v3#S4.SS3)
find that dynamic filtering can impair format learning with GRPO, while its
interaction with richer GIGPO advantages is more favorable. This is evidence
to test admission carefully, not evidence to remove filtering universally.
The author's reported effects involve their reward/format penalties and models;
identically zero Posttrain advantages supply no policy-gradient signal.
[DAPO Table1](https://arxiv.org/html/2503.14476v2#S4.T1) shows gains along a
cumulative math-recipe ablation; those gains do not independently validate each
change for multi-turn tool learning.

The comparison table above now defines optional local checks and transfer
diagnostics, not a learning leaderboard or prerequisites for recipe selection.
Do not use four-episode pilots or two/three-step clipping rates to overrule the
published group8 recipe or claim a winning schedule. Local acceptance checks
are correct masks/credit, frozen behavior scores, intended optimizer-step count,
finite scaled updates and agreement between normalized backend settings.
Calculations can estimate useful-signal retention, reuse exposure, ratio drift,
memory and cost. Their scope must remain explicit. Subsequent production
monitoring should track tool/task quality and cost; recommending the research
recipe does not require proving its superiority on8GB first. Admission or
objective changes require the canonical baseline amendment and public/backend
normalization before production adoption.

This follow-up introduces no CLI setting, production pin, recipe default or
new experiment result. Older precision controls below remain historical evidence
for their exact fixtures; they are not the current universal FP16 verdict.

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

## Research rationale and historical mechanics probes

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

## Historical local evidence and implementation boundary

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
