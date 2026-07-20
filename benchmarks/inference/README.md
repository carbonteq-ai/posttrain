# Inference benchmark data

This directory owns versioned workload definitions and representative prompt
data. It is consumed by `packages/serve`; it is not an evaluation environment
and does not assign capability scores.

- `suites/` defines controlled input/output/context/concurrency matrices.
- `corpora/` stores canonical message records, never model-rendered strings.
- model-native templates and supported reasoning controls are declared by the
  model profile and applied by `serve.prompts`.

Controlled suites use exact token IDs and forced output lengths to measure the
system. Representative corpora use the tokenizer's native chat template and
natural stopping to measure realistic serving behavior. Do not combine those
measurements into one cohort.
