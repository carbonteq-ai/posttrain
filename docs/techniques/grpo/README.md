# GRPO

Group Relative Policy Optimization (online RL with group-relative advantages).

- Target engine: `packages/train` with TRL `GRPOTrainer`
- Rewards and task behavior come from a referenced Verifiers environment.
- Public operation: `posttrain.train.grpo(context, request)`
- Lab smoke entrypoint: `gsm8k-qwen-grpo-smoke`

## Contents

- [Single-GPU optimization](./single-gpu-optimization.md)
- [Adaptive OLMo 3 curriculum: interim 14-step results](./olmo3-adaptive-curriculum-interim-results.md)
- [heuristics.md](./heuristics.md)
- [recipes/](./recipes/)
