# Recipe target: local 8 GB QLoRA SFT

Status: pending implementation against the new profile contract.

The reusable recipe will be a typed definition shipped with `packages/train`
and will define memory-safe QLoRA defaults for the local RTX 3070 Ti. A job
supplies its model profile, dataset reference, explicit overrides, and promotion
decision.

Required validation before publishing the recipe:

- one Gemma 4 E2B profile and one Qwen3.5 2B profile complete a bounded smoke run;
- model-specific LoRA targets are explicit;
- resolved profile/config snapshots are retained;
- the selected adapter is logged as a descendant artifact;
- recovery checkpoints do not automatically become derived profiles.

No command is documented until the rebuilt `packages/train` entrypoint exists.
