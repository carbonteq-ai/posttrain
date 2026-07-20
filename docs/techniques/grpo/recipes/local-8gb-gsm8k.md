# Recipe target: local 8 GB QLoRA GRPO

Status: pending implementation against the new profile and Verifiers-environment contracts.

The reusable recipe will be a typed definition shipped with `packages/train`.
It will consume a model profile plus a qualified Verifiers environment package
reference. There is no local reward-callback compatibility layer.

Required validation before publishing the recipe:

- the environment package can run independently through Verifiers;
- the TRL bridge declares which task semantics it supports;
- bounded rollouts fit the local RTX 3070 Ti;
- reward components and native traces are retained;
- the resulting adapter is logged with its exact parent artifact.

No command is documented until the rebuilt `packages/train` entrypoint and first published environment exist.
