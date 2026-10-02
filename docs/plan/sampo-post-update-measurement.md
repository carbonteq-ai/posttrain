# Measure SAMPO movement after the optimizer update

This living plan follows `docs/templates/PLAN.md`. Keep Progress, Surprises &
Discoveries, Decision Log, and Outcomes & Retrospective current.

## Purpose / Big Picture

Fix objective correctness first, then measure policy movement and compare update
schedules. The frozen product meaning does not change: SAMPO retains one
trajectory-level ratio with token-aligned turn credit. Shared loss arithmetic
must also exclude masked tokens before KL exponentiation.

Zero policy clipping in our SAMPO runs does not tell us whether the optimizer
moved the policy outside the selected clipping interval. The loss is evaluated
before the sole optimizer update, with the current actor's detached scores as
the old scores. Add an optional bounded observation that rescoring the same
sampled tokens after the update can measure. The probe changes observation only;
the separate TRL correction restores the existing credit contract.

## Progress

- [x] (2026-09-30) Verify pinned TRL schedule and retained run evidence.
- [x] (2026-09-30) Trace author scripts, worker imports, turn packing, and minibatch normalization.
- [x] (2026-09-30) Add bounded capture/rescore observation and validate callback, optimizer, masks, signs, and skip behavior.
- [x] (2026-09-30) Record recipe provenance, research comparison, and measurement limitations.
- [x] (2026-09-30) Replay real LFM adapters and native first-turn tokens on the local GPU, independently of the busy workstation.
- [ ] Qualify on real LFM SAMPO tokens and checkpoints when an idle worker is available.
- [x] (2026-09-30) Correct token-local sequence gradients, masked KL arithmetic, and refill token denominators in the maintained TRL fork; publish and select the immutable development candidate.
- [ ] Audit system/user/tool masks and run local 8GB correction comparisons after the objective fix.
- [x] (2026-09-30) Audit one retained native branch's role masks and run a paired local CUDA LoRA correctness canary.
- [ ] Extend the mathematical audit across GRPO/GSPO, GDPO/CAPO, DAPO, and the shared KL path; distinguish correctness from recipe choices.

## Surprises & Discoveries

The shared GRPO loss exponentiated reference-policy differences before masking.
An excluded tool token with policy logp -114 and reference logp -2 overflowed
float32 and produced NaN in sequence GRPO, token DAPO, and token CISPO. All three
failures were reproduced in the actual trainer loss before the fix. The fork now
sets excluded deltas to zero before exponentiation. It also uses `expm1(delta) -
delta`, preserving the approximately 5e-9 k3 value for deltas +/-1e-4 that the old
float32 expression rounded to zero. Neither change alters the selected estimator.
The combined fork regression passes 36 tests, with four PEFT dependency skips.

The next audit found two refill paths recalculating `num_items_in_batch` from
`completion_mask` instead of the sampled-action mask. Both active and dynamic
sampling counted eight completion tokens for a retained batch with only four
sampled actions. Before correction, both regression cases failed. The fork now
uses `completion_mask * tool_mask`, just as initial scoring does. This affects
token-normalized objectives such as DAPO and OLMo 3; SAMPO's per-trajectory mean
does not use that denominator. The extended fork suite passes 59 tests with four
PEFT dependency skips; 42 isolated installed-wheel regressions also pass.

The consumer pins TRL `c4d0db051a7839fe1b1d587185fac33ba88c784f`, not the
sibling checkout's HEAD. With one iteration and aligned gradient accumulation,
the old actor scores are the current scores detached from the gradient graph.
The ratio is therefore exactly one during loss evaluation, even if the ensuing
optimizer update would cross the clipping bounds.

The SAMPO authors' repository at `a25a2a229c85431b421ac785fa5f375a99b2072a`
packs each active agent turn as a row. Its environment rollout group count is
eight while veRL's separate rollout count is one. Do not multiply the actor
minibatch by eight a second time. The actor worker imports the generic
`verl.workers.actor` implementation, which uses frozen old scores when there
is more than one minibatch. Script defaults are not evidence of an actual
executed author's run. The WebShop script references a missing entrypoint.

The workstation was occupied when checked in this session: a running job held
83,585 MiB. The completed high-LR workspace has no local adapter/checkpoint
files; its model and recovery artifacts remain registered in Trackio at steps
20, 40, and 60. Native sample traces have sampler scores, but no frozen trainer
scores. Those cannot reconstruct an exact adjacent-step actor comparison.

## Decision Log

Decision: the user requested the confirmed fixes before further masking and
recipe experiments. Apply the GSPO token-local derivative when sequence
importance sampling receives token-aligned advantages. Scalar episode advantages
retain their existing branch. The numerical sequence ratio and loss value stay
the same; the gradient now retains each token's supplied credit. Rationale:
the frozen SAMPO contract already requires both trajectory-level ratios and
token-aligned turn credit, so this is a correctness repair and does not amend
the baseline. Date: 2026-09-30.

The fork checkout for this repair is `/home/hammad/projects/trl-sampo-local-credit`,
branch `codex/sampo-local-credit`, based on the exact consumer c4d0db commit.
The primary TRL checkout and all its existing worktrees are preserved. Commit
and publish the fork first, create retained release distributions, publish the
exact bytes through Posttrain's repository-owned development workflow, and only
then update this repository's candidate pin and runtime locks. Stable promotion
remains a separate qualified release gate; the current consumer already selects
the development-channel candidate. Fork changes own the trainer regression and
root ledger; Posttrain changes own selection, lockfiles, consumer documentation,
and integration checks. No external reward service is needed for local math tests.

Decision: add optional `post_update_probe_rows` in TRL backend options, default
zero, with a maximum of sixteen rows per optimizer step. Capture the existing
training forward scores, IDs, attention mask, policy mask, and advantages on
CPU, then rescore after the optimizer update. Rationale: bounded observation
adds no extra backward pass or reward call and does not pretend that sampled
rows cover the full rollout population. Date: 2026-09-30.

Decision: initially accept only text, single-process, ordinary GRPO-loss paths
with dropout disabled. Reject incompatible configurations explicitly. Rationale:
distributed collective ordering, multimodal slicing, and dropout noise require
separate qualification. Leave the maintained TRL fork unchanged; this is private
Posttrain adapter observation, not a reusable trainer algorithm change.

Decision: do not change the default schedule solely to make clipping nonzero.
Rationale: compare one versus two passes with frozen old scores, keeping sampled
population, rank, alpha, LR, KL, and optimizer-step budget explicit. A second
pass changes the amount of optimization per environment interaction.

## Outcomes & Retrospective

Objective fixes are committed and pushed to the TRL fork on
`codex/sampo-local-credit`, release commit
`d97a2cf619f94c7076a720116d75f85dfd62382b`, tag
`carbonteq-v1.12.0.post13`. The retained wheel SHA-256 is
`d5530d28b16a4a16ffed1f786356caa9c68a2ad40bdfa6c1b2ab67d03d0d355a`;
sdist SHA-256 is `bf075f003c5134240bbddf7195b93c71ee01ff95f142b68ce200146c20e03510`.
Development publisher run: https://github.com/carbonteq-ai/posttrain/actions/runs/36727590142.
The publisher succeeded with verified readback and clean installation. Consumer
metadata, uv.lock, catalog lock, runtime profile/constraint locks, release ledger,
and CI wheel mirror now select those exact bytes. Existing run containers retain
their recorded objective.

The user-authorized local CUDA correction comparison also ran. The reproducible
tool `tools/check_online_rl_objective_math.py` constructs the same seeded tiny
GPT-2 LoRA, rank 4, alpha 8, dropout zero, LR 1e-4, temperature 0.8, beta zero,
and opposing token advantages in both arms. The only baseline code replacement
is exact post12 `grpo_trainer.py`; all surrounding dependencies match. Both
arms have scalar loss zero, but baseline logp gradients and adapter updates are
zero, while post13 produces [-0.5, 0, 0.5], adapter gradient norm 0.160868, and
maximum adapter change 9.999998e-5. Peak CUDA allocation is 17.1 MB. This
float32 synthetic LoRA fixture isolates correctness; it does not qualify our
float16 LFM task, KL, or production throughput. Aggregate evidence:
`docs/research/evidence/sampo-opposing-credit-cuda-canary.json`.

### Broader objective audit and recipe decisions

| Path | Inspected result | Interpretation |
| --- | --- | --- |
| SAMPO sequence ratio with token advantages | Opposite local credit collapsed before correction | Confirmed bug, fixed in post13 |
| veRL SAMPO selection | Custom sequence_clip reproduced the same cancellation | Worker now selects existing GSPO token-local path in the unchanged pinned runtime |
| GRPO scalar sequence advantages | Same ratio, value, and gradient after correction | Compatibility regression passes |
| Shared non-Liger KL | Masked overflow and cancellation at tiny gaps | Confirmed numerical bugs, fixed in post13 |
| DAPO/active-sampling token denominator | Refill counted tool output | Confirmed bug, fixed in post13 |
| GDPO | Independent component normalization, weighted aggregation, whole-population rollout whitening | Matches selected documented profile; no new construction bug demonstrated |
| CAPO | Outcome-minus-error token rewards, pooled sampled-token normalization per prompt group, token ratios | Matches selected documented profile; no new construction bug demonstrated |
| KL gradient correction | Adapter explicitly sets `use_bias_correction_kl=False` | Intentional legacy recipe with materially different gradient; compare before adoption |
| Loss-length weighting | SAMPO/GRPO average within trajectory; DAPO uses global sampled-token mean | Different objectives, not interchangeable normalizers |
| Distributed/fused losses | No production proof from single-process tests | Audit and qualification remain open |

Sources: [GDPO Eq. 6](https://arxiv.org/html/2601.05242v1),
[CAPO Eq. 2](https://arxiv.org/html/2508.02298v1), pinned TRL
`grpo_config.py:420` and `grpo_trainer.py::_compute_loss`, and the framework
`reward_advantages.py`. Framework construction regressions pass 33 tests with
one dependency skip. These are finite fixtures, not judge-quality or convergence
qualification.

For a fixed context and actions sampled from the actor p, let q be the reference,
z the sampled log-probability, and d = log(q) - z. The logged k3 value is
`exp(d) - d - 1`; its expectation is KL(p||q). Differentiating that value while
holding sampled actions fixed gives `1 - q/p`. Averaging its logit gradient
under p equals the gradient of KL(q||p), not KL(p||q). Multiplying by the
differentiable token actor/old ratio restores the KL(p||q) gradient at the
collection policy. Exact float64 categorical enumeration gives:

| p(action 1), q(action 1)=0.5 | Legacy logit gradient | KL(p||q) logit gradient |
| --- | --- | --- |
| 0.70 | 0.200000 | 0.177933 |
| 0.90 | 0.400000 | 0.197750 |
| 0.99 | 0.490000 | 0.045492 |

This holds with fixed context; it does not establish unbiased full-trajectory
gradients when states are sampled from an older autoregressive policy. A geometric
sequence ratio is also not the exact token action ratio. Do not silently enable
the shared sequence correction and claim a mathematically exact token KL update.
The current beta's effective strength cannot be judged from its numeric value
alone. A KL-recipe experiment must preserve rank 4, alpha 8, LR, temperature,
reference identity, sampled population, and update budget.

Cross-backend correction: Posttrain's veRL SAMPO worker selected the maintained
fork's `sequence_clip`, intentionally mirroring TRL's old derivative. Executing
the exact pinned loss and aggregation bodies confirms zero gradients for opposing
credit and incorrect credit spreading when one sign is clipped. The existing
native `gspo` path produces local gradients matching corrected TRL within the
tested finite-ratio range. Select it without modifying or republishing veRL.
Introduce a separately named `sampo_hierarchy` compatibility capability so
preflight still rejects older forks lacking hierarchical credit; upstream GSPO
registration alone is insufficient. Worker mapping validation passes 119 tests
with eight dependency skips. Full Ray/V1 KL integration remains unqualified.
Native GSPO caps sequence log ratios at 10, so extreme-ratio parity is not claimed.
The reproducible read-only tool `tools/check_verl_sequence_credit.py` extracts
and executes actual pinned core/helper functions, not a rewritten loss. It
records `docs/research/evidence/verl-sequence-credit-math.json`.

The bounded native-mask audit uses the retained high-LR run's step 18 linear
branch, trace `8813246ba499464f8c8a35da75ff73f0`. Its 1,228 native tokens contain
635 system, 51 user, 200 tool, and 342 assistant tokens. System/user/tool have
zero direct loss positions. Each assistant turn excludes four injected header
tokens; all three sampled `<|im_end|>` tokens are included. The prompt split is
690 tokens, the completion span is 538, and the actual scored population is 330.
This demonstrates why a completion-token denominator can materially dilute a
multi-turn objective: on this row 330/538 = 0.613, or approximately 39% dilution
if those extra 208 context tokens were counted. This example is not a measured
whole-update dilution. Source inspection confirms full attention keeps tool
context while the loss multiplies the completion and sampled-action masks.
There is no new direct system/user loss-mask bug demonstrated by this branch;
multi-branch, truncation, population, and fused paths need further qualification.
Evidence: `docs/research/evidence/sampo-native-action-mask-audit.json`.

Final focused consumer validation after pin/CI-ledger synchronization: 185 tests
pass with eight dependency skips; veRL mapping tests pass 119 with eight skips.
The new tools and modified worker pass targeted type checking and linting. Runtime
constraint materialization is current; all nine import contracts pass. Posttrain
changes remain reviewable in the working tree; the TRL release and ledger commits
are pushed. No existing run or deployed image was modified.

Recommended next schedule experiment: after the correctness canary, compare the
current one-pass update with two passes over the same 144 complete trajectories,
freezing actor old scores before the first pass. Keep microbatch one and
accumulation 144. Two passes give two optimizer steps and approximately twice
the actor forward/backward work per collection. A 72+72 split instead gives two
updates with the original amount of actor scoring, plus optimizer/sync overhead,
but decoupling collection and update size needs explicit supported configuration.
Current Posttrain does not expose `num_iterations` through its adapter; these
are proposed settings, not an implemented or launched recipe. Keep the objective
repair separate from either schedule and from KL-recipe changes.

Research identifies an inherited schedule rather than a documented SAMPO
schedule experiment. A real checkpoint-to-checkpoint replay is implemented in
`tools/measure_sampo_checkpoint_drift.py` and has run successfully. Exact
single-update production measurement remains pending; do not present replay
over twenty updates as that measurement. Nine new probe tests pass; combined
adapter/precision validation passes 92 tests with seven dependency skips. The
changed probe and adapter files pass targeted type checking. Full-repository
type checking reports an existing optional-member error in
`packages/train/tests/test_adaptive_curriculum.py:71`, unrelated to these edits.
All nine import contracts pass.

The investigation also exposes a correctness problem in how varying turn
advantages differentiate through our full-trajectory sequence ratio. Resolve
that before interpreting a schedule ablation as a faithful SAMPO comparison.

## Context and Orientation

`packages/train/src/posttrain/train/backends/trl/policy_telemetry.py` wraps the
trainer and owns actor timing. `policy_optimization.py` assembles callbacks;
`policy_config.py` resolves backend settings. A microbatch is a forward/backward
memory unit; accumulated microbatches do not change parameters until the
optimizer update. A policy ratio compares the probability of the same sampled
action under two actor states. For SAMPO it is the exponential of the mean
masked token log-probability difference, rather than a product over tokens.

## Plan of Work

Add `PostUpdateProbe` in a new private `policy_probe.py`. Its CPU snapshots
contain at most the selected rows across an entire update. The trainer wrapper
captures scores inside `_compute_loss`; the actor callback finishes the probe
at `on_step_end`, after optimizer execution. Use the accelerator autocast
context and no gradients for the second scoring pass. If the optimizer was
skipped, emit coverage and skipped status without policy movement numbers.
Namespace observations under `train/rl/post_update/*`, independently of native
loss-time clipping and reference-model KL.

## Concrete Steps

From `/home/hammad/projects/rl` run:

    uv run --with torch pytest packages/train/tests/test_policy_probe.py
    uv run pytest packages/train/tests/test_api.py packages/train/tests/test_trl_precision.py
    uv run ruff check packages/train/src/posttrain/train/backends/trl/policy_probe.py packages/train/src/posttrain/train/backends/trl/policy_telemetry.py packages/train/src/posttrain/train/backends/trl/policy_optimization.py packages/train/tests/test_policy_probe.py
    uv run pyright
    uv run lint-imports
    git diff --check

For a future measured campaign, select `post_update_probe_rows: 4` in a new
versioned TRL training binding. Build a new actual-job image and launch a new
run using normal `posttrain` admission. Do not alter an existing container or
historical run. Compare probe coverage and ratio tails at equal optimizer
steps; do not substitute sampler log probabilities for old actor scores.

## Validation and Acceptance

Tests must demonstrate zero clipping before a real optimizer update and a
nonzero out-of-interval fraction after that update, using the same tokens.
They must also demonstrate tool-mask exclusion, advantage-sign clipping,
bounded row capture, cleanup, and skipped-update behavior. Disabled probing
must preserve the existing path. Production acceptance requires at least one
real LFM SAMPO update with nonempty coverage, finite score differences, and
recorded timing. This gate remains explicit until it is actually run.

## Idempotence and Recovery

The option defaults off. Dropping it restores the previous observation cost.
Each completed or skipped update releases snapshots. A failed probe fails the
run visibly rather than emitting invented zeros. Existing native artifacts
and historical metrics are read-only throughout this investigation.

## Artifacts and Notes

This plan records source links, formulas, run IDs, measured existing counters,
and limits on retrospective reconstruction. Machine-readable replay results
live in `docs/research/evidence/sampo-lr1e4-checkpoint20-40-drift.json` and
`docs/research/evidence/sampo-lr6e5-checkpoint20-40-drift.json`.
Untracked `.claude/` and `.release/` directories are unrelated and preserved.

## Interfaces and Dependencies

`PostUpdateProbe.capture(inputs, old_logps)` stores bounded detached rows.
`finish(trainer, skipped=False)` rescores through the trainer's existing
`_get_per_token_logps_and_entropies` method and emits through `RunContext`.
Torch is imported lazily. No new train/eval/serve dependencies are introduced.

Revision 2026-09-30: created to make the clipping diagnosis measurable without
silently changing SAMPO's optimization schedule.

## Research findings and recipe provenance

Our July 24 SAMPO introduction, commit `676ff4d7`, selected the existing TRL
GRPO loss, sequence importance sampling, and precomputed token advantages. It
did not explicitly set `num_iterations`, so it inherited TRL's one-pass default.
The original plan `docs/plan/sampo-agentic-training.md` discusses reuse of GSPO
and token advantages but records no controlled experiment choosing one optimizer
update per trajectory population. Subsequent small-GPU qualification demonstrates
that training executes; it does not establish the optimal schedule.

There is another explicit compatibility decision: `policy_config.py` pins
`use_bias_correction_kl=False` to preserve the prior sampled k3 KL gradient when
upstream changed its default. The recorded reason is behavioral compatibility,
not a SAMPO quality ablation. This choice cannot explain the policy ratio being
identically one, since that ratio compares current and old actors, not the
reference model.

The [authors' turn collector](https://github.com/WillDreamer/ARL-Arena/blob/a25a2a229c85431b421ac785fa5f375a99b2072a/agent_system/multi_turn_rollout/rollout_loop.py#L251)
adds each active action to the actor batch separately. Their
[WebShop script](https://github.com/WillDreamer/ARL-Arena/blob/a25a2a229c85431b421ac785fa5f375a99b2072a/examples/shop_agent_trainer/Qwen3_4B/train_sampo.sh)
selects sixteen tasks, eight trajectories per task, actor minibatches of 128
turn rows, one PPO epoch, and clip bounds 0.003/0.004. Actor rollout `n` remains
one; environment rollout `n` is eight. For mean trajectory length L, there are
approximately 128L active turn rows and ceil(128L/128) optimizer minibatches,
subject to padding and distributed splitting. The script's missing final
entrypoint means these defaults alone cannot certify a reproduced author run.
The published ALFWorld script uses 256-row minibatches, but its available trainer
does not wire the `sampo` estimator; do not call it an executed SAMPO recipe.

The available WebShop trainer maps SAMPO to GiGPO advantages plus GSPO loss.
Its generic actor imports resolve to `verl.workers.actor.dp_actor`; that code
holds old actor scores fixed when there are multiple minibatches and updates
parameters inside the minibatch loop. This differs from our 144 full trajectories
accumulated into one optimizer update, with each next population freshly sampled.

Public practice has no single required schedule. The
[PPO paper](https://arxiv.org/abs/1707.06347) describes repeated minibatch
optimization over collected samples. [veRL configuration](https://verl.readthedocs.io/en/latest/examples/config.html)
distinguishes collection batch, optimizer minibatch, and PPO epochs, so one
epoch can still contain multiple optimizer updates. By contrast,
[DeepSeekMath](https://arxiv.org/html/2402.03300) documents one-update GRPO, and
[TRL](https://huggingface.co/docs/trl/grpo_trainer) defaults to one iteration
while supporting reuse. One update is therefore a legitimate algorithm choice;
it is not evidence that our agentic turn-credit implementation matches SAMPO.
These are public implementations, not claims about undisclosed company recipes.

## Gradient mismatch: a separate correctness finding

Let z_t be the current actor log probability, z_old,t the frozen old score, and
T the number of scored policy tokens in one full trajectory. Our pinned TRL
sequence branch computes r = exp(mean_t(z_t - z_old,t)). Before clipping binds,
the policy loss is -r * mean_t(A_t), with A_t the token-aligned advantage. Thus
its derivative with respect to every z_t is -r * mean(A) / T. With detached
sampler corrections w_t, replace mean(A) with mean(w A); the shared-gradient
problem remains. The KL term is separate and does not restore the intended
turn-specific policy credit.

The authors' [GSPO implementation](https://github.com/WillDreamer/ARL-Arena/blob/a25a2a229c85431b421ac785fa5f375a99b2072a/verl/trainer/ppo/core_algos.py#L1056)
uses exp(stop_gradient(log r) + z_t - stop_gradient(z_t)). Its numerical ratio
is still r, but its derivative preserves local advantages: -r * A_t / T in
the analogous token-mean case. Moreover, the authors pack separate turns,
while our advantage calculator explicitly assigns different values to different
turn spans within one trajectory. Plain sequence GRPO with a constant advantage
can hide this distinction; varying advantages cannot.

PyTorch reproduced the consequence with two scored tokens and A=[+1,-1]:

    our trajectory sequence-ratio policy gradient: [0.0, 0.0]
    author GSPO token-local policy gradient:       [-0.5, +0.5]

This establishes gradient mixing in the selected implementation. Its quantitative
effect on AutomationBench success still requires a corrected-objective ablation.
It does not establish that all observed small gradient norms are caused by this
one issue.

Corrective work belongs in the maintained CarbonTeq TRL fork, with an explicit
opt-in GSPO local-credit derivative or a carefully qualified turn-row packing
path. Read `docs/tooling/forks.md`, resolve the fork at the pinned c4d0db commit,
create an isolated checkout, add a nonconstant-advantage regression there, and
update its `CARBONTEQ_FORK.md`. Commit and publish the fork first, then update
`packages/train/pyproject.toml`, `uv.lock`, and `docs/tooling/trl/README.md` here.
Do not silently change vanilla GRPO's sequence gradient. No sibling repository
has been edited by this measurement task.

Revision 2026-09-30: record the discovered local-credit gradient mismatch and
the limits of the authors' published scripts; distinguish runtime qualification
from mathematical fidelity and from quality evidence.

## Retrospective measurement result

The higher-LR run is Trackio ID `444abb71c7194b83a2703a5c8f504402`, the
lower-LR run is `476f5dd9a2fb4097aeb48055e9c62bcd`. Both use rank four,
alpha eight, the standard alpha/r scaling, and actor temperature 0.8. Compare
their exact step-20 and step-40 model-adapter artifacts, not run labels alone.
Higher-LR adapters are version four; lower-LR adapters are version three. Full
manifest digests and trace IDs are retained in the two JSON evidence files.

Both replays use the same 530 original policy tokens from four distinct tasks
in the higher-LR run's first update: examples 32, 114, 74, and 22. They include
one complete first assistant turn per task, no tool tokens, with no retokenization.
The selection is deliberately bounded to at most 2048 total tokens and 256
policy tokens per row so the local 8GB GPU can coexist with the desktop. An
earlier attempt including long contexts exhausted GPU memory; it ended without
changing any training job. The final diagnostic filters complete short turns
rather than truncating their contents. It is not a random or representative
sample of training, and the higher-LR source tokens favor neither matched
on-policy interpretation nor an unbiased KL estimate at step 20.

For LR 1e-4, the first-turn geometric policy ratio over twenty updates ranges
from 0.87898085 to 1.01005669, and two of four turns lie outside [0.997,1.004].
For LR 6e-5, it ranges from 0.99086982 to 1.00523667, also with two of four
outside. The maximum absolute individual token log-ratio is respectively
6.536726 and 0.711777. The unweighted k3 statistic on these off-policy probe
tokens is explicitly named `unweighted_probe_k3`; do not present it as measured
KL or compare it with the run's reference KL series. Same-checkpoint repeat
controls yield maximum absolute log-probability difference zero in both runs.

A useful mathematical constraint is that .997^20 = .94167961. The observed
minimum first-turn ratio .87898085 is below this. Therefore at least one
intervening update must have moved that fixed first-turn probe by more than the
lower bound, if evaluated under the same scorer at each intermediate state.
Its geometric average step ratio is at most .99357115. This does NOT establish
clipping on an actual sampled training batch or on our full-trajectory ratio;
neither the intermediate actor states nor their exact training populations
are retained. The result is cumulative policy movement, not a reconstructed
historical clip fraction and not proof that LR 1e-4 is too high.

From the repository root, reproduce the higher-LR measurement with:

    uv run --with torch --with 'transformers>=5.2' --with peft python tools/measure_sampo_checkpoint_drift.py --run-id 444abb71c7194b83a2703a5c8f504402 --output /tmp/sampo-high-checkpoint-drift.json --device cuda

Reproduce the lower-LR measurement on those same tokens with:

    uv run --with torch --with 'transformers>=5.2' --with peft python tools/measure_sampo_checkpoint_drift.py --run-id 476f5dd9a2fb4097aeb48055e9c62bcd --probe-run-id 444abb71c7194b83a2703a5c8f504402 --output /tmp/sampo-low-checkpoint-drift.json --device cuda

The helper uses Trackio's private manifest hydration calls only to download
through its checksum-verifying artifact implementation, without opening a
writer run or adding lineage edges. It needs read access to Trackio artifacts
and the exact base revision in the local Hugging Face cache. CPU is supported
via `--device cpu`. Token payloads remain in memory, not the saved report.

Revision 2026-09-30: add actual checkpoint replay evidence, matched-token LR
comparison, temperature normalization, deterministic control, and the precise
boundary of the per-update inference from cumulative movement.
