# Online-RL update recipes: research and controlled candidates

2026-09-30. This is research for the 8 GB correctness campaign, not a
production recipe change or a claim about the active workstation run. BF16
and FP16 are the supported targets; FP32/FP64 are diagnostic references.

The strongest next candidate is **two bounded optimizer updates per frozen
behavior population**, compared with the existing one-update control. Keep
the repaired token-local SAMPO gradient, rank, alpha, reward projection and
mask fixed. Measure what changes before changing clipping, KL, reward shaping
or turn packing. Published evidence supports investigating reuse and
sequence clipping; it does not establish that adding epochs fixes our runs.

## What our checked-in selections actually do

Inspected the current adapter and catalog files directly. These are selections,
not verified live-run settings:

| Selection | Population and optimizer | Other relevant settings |
| --- | --- | --- |
| `apps/lab/.posttrain/catalog/precision-qualification.yaml`: 2.6B SAMPO r4 | 24 prompts × 6 episodes; microbatch 1, accumulation 144; LR 6e-5 constant | beta .01, base reference, gamma .95, mean advantages; initial oversample 4/refill 5; truncation penalty .1, truncated loss retained |
| `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml`: 1.2B local SAMPO | 2 prompts × 4 episodes; microbatch 1, accumulation 8; LR 5e-5 constant | beta .005, gamma .95, mean advantages; reply budget 3072; truncation penalty .2, truncated loss retained |
| Current native precision fixture | Frozen group of 2 trajectories; microbatch 1, accumulation 2; 3 iterations; LR 1e-4 | rank 4, alpha 8, q/v adapters, beta .01, sequence bounds .003/.004; synthetic supplied traces |

`packages/train/src/posttrain/train/backends/trl/policy_config.py` does not
select `num_iterations`: the ordinary path inherits one iteration. It sets
`use_bias_correction_kl=False`, sequence importance sampling for SAMPO, and
precomputed token credit. One full accumulated population produces one
optimizer step. Accumulation makes the population fit memory; it creates no
intermediate updated policy. Fresh actor/old equality therefore explains zero
pre-update clipping, independently of how far the eventual optimizer moves.

The [native fixture](native-multiturn-results.md) already demonstrates clipping
under frozen-group reuse. Its BF16 step-2/3 clip-region averages are
29.17%/29.17% for Qwen and 58.52%/46.02% for LFM. These are microbatch averages,
not a quality ranking or a recommended clipping target.

## Primary research and implementation evidence

### 2026-10-01 follow-up: defaults are not paper recipes

Fresh inspection of upstream configuration and the immutable ARL-Arena source
distinguishes a dataset epoch from a pass over one frozen rollout population.
An optimizer step follows each optimizer minibatch, after its microbatches
have accumulated. Neither finishing the task inventory nor finishing the
rollout-population epoch is a prerequisite for that step.

| Source | Checked setting | Interpretation |
| --- | --- | --- |
| [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) | `num_iterations=1`; `steps_per_generation` defaults to gradient accumulation | Default generation population corresponds to one effective optimizer batch; larger populations and reuse are configurable. |
| [veRL actor defaults](https://github.com/verl-project/verl/blob/main/verl/trainer/config/actor/actor.yaml) | `ppo_epochs=1`, `ppo_mini_batch_size=256` | One pass can still contain multiple optimizer steps; count also depends on the rollout batch and backend sample-count normalization. |
| [NeMo RL 1B example](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/grpo_math_1B.yaml) | 32 prompts ×16 generations; global training batch512 | Configured population fits one optimizer batch. `max_num_epochs=1` is an outer dataset limit. |
| [NeMo RL 8B example](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/grpo_math_8B.yaml) | 64 prompts ×32 generations; global training batch512 | Configured population partitions into four optimizer batches. This is a shipped example, not a universal NeMo default. |
| OpenRLHF CLI defaults (`https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/train_ppo_ray.py`, local archive) | rollout batch1024, samples per prompt1, global train batch128, training epochs1 | Eight configured effective training batches at default sample count. The generic PPO entrypoint is not itself a complete GRPO recipe; additional generations change the count. |

The [ARLArena paper Appendix B](https://arxiv.org/html/2602.21534v3#App2)
reports16 prompts and group8 for ALFWorld/WebShop, with minibatches256/128;
Sokoban uses32 prompts/group8/minibatch64 and TIR Math512 prompts/group5/
minibatch128. It does not state a universal number of optimizer minibatches
per agentic rollout population or PPO reuse epochs. Its total-epoch counts
must not be read as repeated optimization of one frozen population.

At author commit `a25a2a229c85431b421ac785fa5f375a99b2072a`, active turns
become separate training rows. The actor defaults to one PPO epoch and steps
inside the minibatch loop. FSDP normalizes the configured minibatch by
`actor_rollout_ref.rollout.n` and data-parallel size; the inspected environment
recipes keep actor rollout `n=1` while setting separate `env.rollout.n=8`.
Thus, before padding and distributed balancing, WebShop's128 episodes with
mean L active turns supply approximately L optimizer minibatches; ALFWorld's
256-row setting supplies approximately L/2. These are inferred counts, not
measured author-run counts. The WebShop script's missing final entrypoint
continues to limit reproduction. Whole-episode packing in Posttrain cannot
use these turn-row minibatch sizes without changing ratio/reduction meaning.

For Posttrain normalization, define rollout population, optimizer minibatch
size, passes per frozen population, row unit, and old-score lifetime explicitly.
Compute complete reward/anchor groups before optimization splits. A group need
not remain in one gradient minibatch after its advantages are fixed, but a
split must never recompute group statistics or refresh behavior scores.

### SAMPO / ARLArena: the closest task family, with provenance gaps

[ARLArena v3](https://arxiv.org/html/2602.21534v3) evaluates SFT-initialized
Qwen3-4B agents, plus 8B checks, rather than our tiny LoRA policies. Table 3
reports ALFWorld success 62.36% GRPO, 78.61% GSPO, 92.72% SAMPO. Table 4 shows
that sequence masking improves tolerant SAPO from 25.16% to 76.92%, versus
48.05% with KL .05. This supports sequence control over simply increasing KL;
it is not a SAMPO reuse-count ablation. Table 3 and its following prose disagree
on SAMPO WebShop success (77.73% versus 74.08%); retain that uncertainty.

Appendix B uses group 8, gamma .95, environment KL .01 and training temperature
1.0. It lists WebShop GSPO bounds .03/.04 and `mean_std_norm`. The pinned
[WebShop SAMPO script](https://github.com/WillDreamer/ARL-Arena/blob/a25a2a229c85431b421ac785fa5f375a99b2072a/examples/shop_agent_trainer/Qwen3_4B/train_sampo.sh)
instead selects .003/.004 and `mean_norm`, with 16 prompts, 8 episodes per
prompt, 128-row optimizer minibatches, 8-row microbatches, LR 1e-6 and KL .01.
It references an unavailable `main_shop_agent_final` entrypoint in the
checked source. These are inspectable choices, not a reproduced author run.

The author's turn collector (`https://github.com/WillDreamer/ARL-Arena/blob/a25a2a229c85431b421ac785fa5f375a99b2072a/agent_system/multi_turn_rollout/rollout_loop.py`, local archive)
creates active turn rows. The [actor worker](https://github.com/WillDreamer/ARL-Arena/blob/a25a2a229c85431b421ac785fa5f375a99b2072a/verl/workers/actor/dp_actor.py#L354)
splits rows into optimizer minibatches, zeroes gradients for each, accumulates
its microbatches, and steps inside the minibatch loop. It substitutes current
detached scores only for a single minibatch and one epoch; otherwise old scores
stay frozen. Thus one epoch can contain multiple optimizer steps. With 128
episodes of mean L turns and 128-row minibatches, roughly L steps occur,
subject to row padding and distributed batching. The
actor configuration (`https://github.com/WillDreamer/ARL-Arena/blob/a25a2a229c85431b421ac785fa5f375a99b2072a/verl/workers/config/actor.py`, local archive)
defaults to one PPO epoch. Dataset epochs are a separate quantity.

**Transfer limit:** changing our iteration count alone preserves whole-episode
ratio/reduction and does not reproduce turn-row optimization. Changing the
unit to turns changes the objective: equal turn means and equal trajectory
means weight long episodes differently. Treat author-faithful turn packing as
a separate experiment after the current packed objective is qualified.

### GSPO: sequence clipping and local token credit

[GSPO v2](https://arxiv.org/html/2507.18071v2), section 4.3, explicitly defines
GSPO-token for multi-turn token advantages. Its forward ratio is the geometric
mean sequence ratio, but its derivative is token-local. Scalar credit recovers
GSPO; mixed-sign credit requires the local path. This supports retaining our
gradient repair rather than reverting it to make recipes match superficially.

The paper compares GRPO and GSPO on Qwen3-30B-A3B and emphasizes multiple
minibatches of collected rollouts. Its clipping fractions differ by roughly
two orders of magnitude; equal percentages or equal epsilon values are not
equivalent across token and sequence ratios. It does not supply a small-model
LoRA BF16/FP16 optimum or prove the normalized sequence ratio is an unbiased
trajectory importance weight. The length-normalized geometric ratio is a
chosen stabilization surrogate, distinct from the full likelihood product.

### DAPO: useful ablations, weaker direct transfer to agentic credit

[DAPO v2 Table 1](https://arxiv.org/html/2503.14476v2) gives the actual
progressive Qwen2.5-32B AIME24 avg@32 ablation:

| Added cumulatively | Score |
| --- | ---: |
| Naive GRPO | 30 |
| Overlong filtering | 36 |
| Clip-higher | 38 |
| Soft overlong punishment | 41 |
| Token loss | 42 |
| Dynamic sampling | 50 |

These are conditional gains along one path, not independent causal effect
sizes. DAPO uses token bounds .2/.28, a batch-token denominator, no reference
KL, and graded length penalties near the response cap. Copying .28 into SAMPO
would change a very different ratio regime. Rejecting all identical-outcome
groups is justified for scalar group-centered outcome credit; it need not be
safe for nonzero turn/process credit. The large single-turn math setting does
not establish that masking incomplete tool trajectories helps AutomationBench.

### Multi-turn KL evidence is conditional, not a universal coefficient

[A Practitioner's Guide v1](https://arxiv.org/html/2510.01132v1), Table 10,
reports Qwen1.5B TextWorld success .78 at KL .01, temperature .7,
actor/critic LR 5e-7/5e-6, versus .66 at KL .005 with those same other listed
settings. KL .001 gives .43. The .90 best configuration also increases both
learning rates; it does not isolate KL. These are PPO actor/critic results,
not LoRA SAMPO evidence. Its sparse-reward GRPO setting is beta .001, four
generations, batch 16. Keep the distinction when arguing for beta .01.

[Current TRL documentation](https://huggingface.co/docs/trl/grpo_trainer)
defaults to one iteration and beta zero while exposing reuse. Multiple public
recipes legitimately coexist. Our zero clipping is therefore a schedule
observation; it is not proof of an invalid optimizer or universal industry
practice. Clipping compares actor to behavior, whereas reference KL compares
actor to an anchor policy. Neither bounds every post-step parameter move.

## Candidate and experiments to select it

These are proposed research arms, not newly implemented CLI flags. Start with
short correctness gates, then longer behavioral experiments. Use both models
in BF16 and FP16, trainable parameters FP32, explicit logits/logprob promotion,
and record initial scale, scale history and applied optimizer steps. Current
[FP16 evidence](bf16-fp16-results.md) prevents treating Qwen's scale-1024
three-step fixture as three learned updates: all three are skipped. Scale128
applies updates but retains one strict gradient discrepancy. Resolve or label
that gate before interpreting a recipe ranking.

| Arm | Change | Question and required control |
| --- | --- | --- |
| S0 | One full optimizer step per fresh group | Existing schedule baseline; retain exact frozen scores for measurement |
| S1, preferred first candidate | Two full optimizer steps over the same group, old scores frozen | Does bounded reuse improve reward-directed movement without invalid gradients or excessive clipping? Same credit and denominator as S0 |
| S2 | Three full optimizer steps | Sensitivity/control only; tests whether step 3 becomes mostly clipped or adds useful movement |
| M1 | One epoch split into two optimizer minibatches | Distinguish reuse from smaller optimizer populations. Preserve full-population reward grouping before splitting; state per-minibatch denominator explicitly |
| T1, later | Turn rows, turn geometric ratios and author reduction | Test the paper's optimization unit; this is an objective/packing comparison, not just S1 with another name |

First replay **one identical sampled population** across S0/S1/S2 from identical
adapter and Adam states. A matched third backward pass without stepping tests
numerical repeatability. Count fresh rollout batches, backward token exposure,
Adam steps, scale skips, update norms and wall time separately. A three-step
budget is enough to establish derivative/sign/clipping mechanics; it cannot
select a learning recipe.

For learning, start 2 prompt groups × 4 episodes (8 episode rows), microbatch1.
Keep dropout zero, rank4/alpha8 and target modules fixed while comparing
schedules. S0 accumulation8 versus S1 accumulation8 with two passes; M1
accumulation4 with two minibatches is a different update budget. Compare both
equal fresh-episode budgets and equal **applied** optimizer-step budgets, with
at least three seeds and held-out fixed tasks. Keep the existing LR initially
to identify schedule effects; a second arm halves S1 LR, while acknowledging
Adam makes two half-rate steps mathematically different from one full-rate
step. Report alpha, rank, scaling convention and adapter update norms; raw LR
alone cannot normalize a cross-paper comparison.

Initially preserve beta and its derivative convention. Then test beta0 versus
.01 on the winning qualified schedule, measuring penalty gradients separately
from policy gradients. Do not silently flip the sequence KL correction flag:
the current legacy convention and a token-corrected KL need separately derived
and independently checked objectives. Prefer a fixed base anchor for a
comparable pilot, recording its exact identity. Reference resets/adaptive
coefficients would add moving-target confounds without a local ablation.

Before dynamic filtering, record actual valid-token advantage norm in every
rejected group. Equal final episode returns can coexist with differing
intermediate returns and nonzero anchor-state credit; no scalar-spread test
alone proves zero SAMPO gradient. For groupsize2 and binary independent success
probability p, mixed outcomes occur with probability 2p(1-p), only .095 at
p=.05. Groupsize4 yields 1-p^4-(1-p)^4=.18549. This explains expensive refill
without requiring an optimizer bug. Changing group size changes both coverage
and credit statistics, so measure it separately.

For truncation, first distinguish budget exhaustion from task failure and
parse failure. Re-run identical prompts at 128/256/512 response budgets in a
collection-only pilot; allow the budget LFM needs to finish before comparing
learning. Longer later responses are not identical samples even with the same
random seed. Then compare retained truncated credit, whole-completion masking,
and the existing graded penalty one at a time. Retain completed earlier turns
in a separate defined arm if only the final turn truncates; masking the entire
episode is another objective choice. Avoid an invented universal 10% threshold.

Required telemetry: valid tool format/errors, truncations, completion reason,
assertion progress, reward/component spread, rejected groups' credit norm,
per-sign sampled-token ratio tails, turn and episode geometric ratios,
behavior drift after each step, base-reference KL with estimator definition,
sampled-only entropy, unclipped/clipped gradient norms, nonfinite counts,
optimizer state changes and held-out task success. Weight rate aggregates by
their actual denominators; report group/episode/turn/token averages separately.

## Currently runnable correctness controls

From the repository root, use the isolated runtime described in the campaign.
These commands exercise the existing frozen three-update fixture, not S0/S1:

```bash
PYTHONPATH=/home/hammad/projects/renderers-lfm-mask:/home/hammad/projects/trl-sampo-local-credit:/tmp/trl-math-peft:/tmp/trl-math-renderers-deps:/tmp/posttrain-mathdeps /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_multiturn_run.py --model qwen08 --dtype bfloat16 --output .posttrain/state/correctness/qwen08-recipe-control-bf16.json
PYTHONPATH=/home/hammad/projects/renderers-lfm-mask:/home/hammad/projects/trl-sampo-local-credit:/tmp/trl-math-peft:/tmp/trl-math-renderers-deps:/tmp/posttrain-mathdeps /home/hammad/projects/trl-gdpo-capo/.venv/bin/python ${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/native_multiturn_run.py --model lfm12 --dtype float16 --initial-scale 1024 --output .posttrain/state/correctness/lfm12-recipe-control-fp16.json
```

Run serially on the shared 8 GB GPU. Qwen FP16 diagnostic substitutes
`--model qwen08 --dtype float16 --initial-scale 128`; it retains the documented
strict failure. S0/S1/M1 need a parameterized runner with independent references
before execution. At research completion the script hardcoded three iterations;
the parent subsequently added `--iterations 1|2|3` for supplied-trace probes.
See [candidate synthesis](recipe-selection.md) for current controls. No experiment
or configuration change was run by this research agent.

The recommendation remains provisional: qualify bounded reuse first, then
decide with held-out learning and throughput evidence. Public ablations justify
this experimental order; they do not establish our best recipe in advance.
