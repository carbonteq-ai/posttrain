# BF16 and FP16 recipe research

Research checked on 2026-09-30. This is a recommendation for controlled
experiments, not a production recipe change. BF16 and FP16 remain the primary
qualification targets; full FP32/FP64 runs are diagnostic controls.

## Recommendation

Keep BF16 as the currently passing control. Qualify a matched FP16 training
and rollout candidate with dynamic loss scaling, FP32 trainable LoRA weights
and optimizer state, and FP32 probability/loss arithmetic. Test the initial
scale separately for each model and context length before spending a rollout
budget. There is credible research support for trying FP16 in online RL, but
no inspected ablation establishes that scale 128 is generally optimal, that
1024 is generally wrong, or that our Qwen0.8B/LFM1.2B task behavior improves.

The local evidence presently supports different numerical conclusions for
the two models. LFM's three-update fixture passes at scale 1024. Qwen's
fixture skips all three updates at 1024 and applies three at 128, with one
strict independent scaled-gradient comparison still failing. Exact coded
derivative controls match the optimizer; FP64 loss-side diagnostics close
that comparison. These are precision-sensitive backward observations,
not proof that a production FP64 loss is warranted. See the full
[local precision qualification](bf16-fp16-results.md).

## Recent paper and its actual implementation

[Precision-RL, arXiv:2510.26788v1](https://arxiv.org/html/2510.26788v1),
30 October 2025, is directly relevant. It compares precision combinations
with vLLM rollouts and FSDP training. Both FP16 gives nearly 100% training
accuracy on its filtered solvable MATH dataset; BF16 training with FP32
inference is stable but almost three times slower in inference. Its offline
analysis reports approximately 24 times smaller sequence mismatch with FP16.
The LoRA comparison uses Qwen2.5-Math-1.5B, rank 32, alpha 64, and LR 4e-5;
BF16 collapses around 600 steps while FP16 remains stable. These are paper
results, not guarantees for our architectures, small groups, or tasks.

The paper's solvable-task filtering and long learning curves are useful
experimental designs. A three-update math probe cannot reproduce its learning
claim or establish an algorithm as fundamentally unreliable.

The [authors' code](https://github.com/sail-sg/Precision-RL/tree/5b0448596e65840423f9627307745af7b2c2f340)
was inspected at commit `5b0448596e65840423f9627307745af7b2c2f340`
(14 November 2025). Two distinctions matter:

- The [veRL patch](https://github.com/sail-sg/Precision-RL/blob/5b0448596e65840423f9627307745af7b2c2f340/verl_fp16.patch)
  aligns actor and rollout dtype to FP16, retains FP32 gradient reduction and
  buffers, uses `ShardedGradScaler(growth_interval=400)`, scales backward,
  and unscales before gradient clipping. It does not explicitly choose an
  initial scale. Growth interval 400 is an implementation choice, not an
  inspected ablation winner.
- The [published LoRA script](https://github.com/sail-sg/Precision-RL/blob/5b0448596e65840423f9627307745af7b2c2f340/oat/scripts/lora/fp16_grpo_tis_lora.sh)
  actually specifies rank 1 / alpha 2, LR 4e-5, beta 0, constant scheduler,
  one PPO epoch, eight responses per prompt, and token-TIS cap 2. Its run
  name says BF16 despite FP16 flags. The paper/script rank discrepancy
  prevents treating this script as exact reproduction of the reported LoRA
  result. Its Oat dependency is installed without an immutable pin.

This warrants testing precision consistency, not copying all flags. Our
diagnostic rank 4 / alpha 8 shares alpha/rank=2 with both variants, but that
alone does not normalize effective Adam updates across rank, targets,
initialization, batch, objective, or model. The configured LFM recipe uses
all-linear targets; the short precision fixture uses q/v targets. Keep that
scope difference visible.

## Established runtime practice

[PyTorch AMP documentation](https://docs.pytorch.org/docs/2.14/amp.html)
documents FP16 backward underflow, scaler defaults of 65536 with growth
factor 2, backoff 0.5, and growth interval 2000. It also warns that BF16-trained
models can overflow FP16's range and that the scale can legitimately fall
below one. These are API behavior and numerical guidance, not RL recipe
ablations. This online reference is version 2.14; our isolated runtime is
Torch 2.13.0+cu130, so installed behavior remains authoritative.

[PyTorch accumulation examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)
require a constant scale through all microbatches in one effective batch;
unscale and clip only after accumulation; call scaler step/update once for
that effective batch. These are correctness constraints. Changing scale
mid-accumulation mixes incompatible gradients. Likewise, clipping the scaled
gradient implements the wrong cap.

[PEFT's dtype guidance](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting)
keeps trainable adapters FP32 by default over BF16/FP16 bases. Lower-precision
adapters have greater overflow/underflow risk and usually little total memory
benefit. Our FP32 adapter guard agrees with that choice. The frozen base
does not need a trainable FP32 master copy in LoRA; this does not imply all
intermediate backward tensors are FP32.

The current framework chooses initial scale 1024 explicitly in
`packages/train/src/posttrain/train/precision.py`. Its runtime helper preserves
an already initialized scaler, and its monitor detects skips using both
optimizer metadata and scale backoff, including the fused-optimizer case.
These observations explain current behavior; they do not establish the
original empirical rationale for 1024.

## Mathematics that determines the experiments

In exact arithmetic, with score derivative vector c and model Jacobian J,
`Jᵀc = Jᵀ(Sc)/S`. Half-precision backward violates this equality through
rounding, underflow and overflow. A useful scale must keep relevant
intermediate gradients inside the usable range. Lowering S can remove
overflow while discarding small components; increasing S can preserve those
components while introducing overflow elsewhere. Final FP32 adapter gradients
alone do not bound hidden FP16 intermediates. Scaling cannot repair a
nonfinite forward activation or an already rounded probability loss.

For a toy intermediate gradient g, avoiding representational overflow requires
`S|g| < 65504`; retaining subnormal magnitude requires roughly
`S|g| >= 2^-24`, subject to kernel flush behavior and rounding. These are
illustrative necessary range considerations, not sufficient whole-model bounds.
The model's largest and smallest relevant intermediate components can make
the admissible scale interval narrow or empty.

Train/rollout mismatch is a different axis. For identical weights, define
`delta_t = log p_train(token_t) - log p_rollout(token_t)`. Sequence log-ratio
is `sum(delta_t)`; a sequence geometric mean uses `mean(delta_t)` instead.
Both must be recorded because one can look small while the other grows with
length. Upcasting a rounded BF16 head output prevents additional probability
rounding, but cannot recover information already lost in the head computation.
Therefore a probability-kernel fix and a matched FP16 engine experiment test
different hypotheses.

## Concrete small-GPU experiments and selection gates

1. **Scale adaptation, same traces.** For each model, sweep FP16 initial
   scales 32, 128, 512 and 1024 against BF16. Keep rank/alpha/targets, LR,
   optimizer, masks and input tokens fixed. Use 2–3 attempted updates first;
   extend surviving candidates to 8–12 applied updates. Include longer traces
   from the earlier Qwen failure. Record attempted/applied/skipped updates,
   scale before/after, finite activation boundaries, independent scaled score
   and parameter gradient errors, clip norm and actual parameter displacement.
   Report both attempts and applied updates instead of extending secretly
   until a favorable result appears. Scale 128 is a local candidate only.
2. **Engine precision, identical weights.** On fresh bounded AutomationBench
   trajectories, obtain sampler logprobs and teacher-forced trainer logprobs
   in matched BF16 and matched FP16. Hold temperature/top-p and sampled-token
   provenance fixed; for a pure raw-policy comparison use temperature 1 and
   top-p 1. Report delta quantiles, summed and mean sequence deltas, mismatch
   by response length, probability ratio tails, tool-call failures and
   truncation. Distinguish sampling-policy changes from numerical mismatch.
3. **Production targets after q/v control.** Repeat passing arms with each
   configured target set, including LFM all-linear. Keep the loss-side FP64
   control diagnostic; compare FP32 formulations before adopting extra
   precision. Check memory and time separately from instrumented backward
   probes. Add scaler checkpoint/restore and fused-optimizer skip tests.
4. **Learning evidence after correctness.** Select a small set of tasks
   with mixed initial success, retaining the selection rule and held-out
   evaluations. Compare rewards/tool errors/truncation per applied update
   and per rollout token budget across multiple seeds. Group size two is
   adequate for some derivative tests but a weak learning comparison.
   Stop short of declaring a precision winner from a three-step reward change.

An FP16 candidate should be selected only when it applies finite nonzero
updates, passes independent scaler-aware gradient controls on its intended
target/context configuration, and demonstrates acceptable skip cost and
engine mismatch. BF16 remains a valid supported candidate even if this
particular paper favors FP16. None of these research findings justify
disabling overflow fencing or counting skipped optimizer attempts as learning.
