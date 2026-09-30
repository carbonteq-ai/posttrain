# Initial real-model SAMPO correctness results

Executed locally on 2026-09-30, RTX3070Ti8GB, Torch2.13.0+cu130,
Transformers5.16.1, TRL post13 source d97a2cf, PEFT0.21.1. Runner:
`${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/online_rl_matrix.py`. Exact source/model/input identities
are in the adjacent JSON artifacts. These are direct-loss tests, not complete
Trainer, task-learning, distributed, or production runtime qualification.

| Model / base precision | FP32 scoring reference max error | Opposing local logp gradient | Adapter update |
| --- | ---: | --- | --- |
| Qwen3.5-0.8B FP16 | 0.000000954 | correct | nonfinite gradients; unresolved |
| Qwen3.5-0.8B BF16 | 0.0172361 | correct | finite |
| LFM2.5-1.2B-Thinking FP16 | 0.000000477 | correct | finite |
| LFM2.5-1.2B-Thinking BF16 | 0.0442148 | correct | finite |

Full and chunked scoring have the same reported error. Tolerance 0.00002 was
declared before execution and retained. BF16 discrepancies need investigation
of temperature and log-softmax precision, followed by matched policy-drift
and clipping measurements; an absolute score error is not itself a ratio error.
Qwen's FP16 failure needs localization and comparison with full Trainer
precision handling. Never step an optimizer with nonfinite gradients.

The scored-token derivative [-0.5, 0, 0.5] excludes the middle tool token.
Zero numerical loss still produces opposing local credit. This fixture does
not establish native system/user message projection or full multi-turn masks.

## Where the KL numbers come from

For actor p=[0.9,0.1], reference q=[0.5,0.5], and first-action logit z=ln(9),
set d=ln(q)-ln(p) and k3=expm1(d)-d. Hold sampling weights p_old=p fixed.
Differentiate sum(p_old*k3): the derivative is p[0]-q[0]=0.400.
Differentiate sum(p_old*(p/p_old)*k3): it is
p[0]*(1-p[0])*ln(9)=0.1977502119602598, matching KL(p||q).
Both values equal 0.3680642071684971 at this point. Multiply the derivatives
by beta to obtain this example's penalty contribution. These are logit
derivatives in an enumerated fixed context, not measured LoRA gradients.

Posttrain commit e9287d411 (PR107) explicitly preserved
`use_bias_correction_kl=False`. The rationale is recorded in
`docs/plan/gdpo-capo-dual-backend-support.md`: retain the historical sampled
k3 gradient across an upstream upgrade. There is no quality ablation in that
decision. Upstream changed the default in commit185f7382 / PR6503 and
qualifies its unbiased-gradient claim to token importance sampling:
https://github.com/huggingface/trl/pull/6503.
Switching the flag for sequence weighting remains an unqualified recipe
change. The flag was not changed by this experiment.

Next gates: localize numerical failures, compare independent parameter
gradients, expand clipping/KL/accumulation cases, qualify save/resume and
native masks, then run bounded AutomationBench learning on both families.
