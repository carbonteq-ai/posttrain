# GRPO

Group Relative Policy Optimization (online RL with group-relative advantages).

- Target engine: `packages/train` with TRL `GRPOTrainer`
- Rewards and task behavior come from a referenced Verifiers environment.
- Public operation: `posttrain.train.grpo(context, request)`
- Lab smoke entrypoint: `gsm8k-qwen-grpo-smoke`

## Contents

- [heuristics.md](./heuristics.md)
- [recipes/](./recipes/)
