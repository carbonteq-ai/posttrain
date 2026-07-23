# ADR 0011 — Canonical post-training data and framework adapters

## Status

Accepted.

## Context

SFT and preference data must be reusable across TRL, NeMo, future trainers,
Hugging Face datasets, and demonstrations derived from Verifiers traces. These
systems overlap around OpenAI-style messages but do not share one complete
schema. TRL commonly uses `messages`, `prompt`/`completion`, and
`prompt`/`chosen`/`rejected`; NeMo additionally uses ranked
`context`/`completions`; older Hub datasets use Alpaca or ShareGPT columns.
Verifiers does not own an SFT dataset format: its authoritative values are
typed `TaskData` inputs and graph-shaped rollout traces.

Putting canonical records in `posttrain.train` would make dataset builders and
converters depend on one execution engine. Making a Hugging Face `Dataset` the
domain object would couple identity and validation to one storage library.
Using every external row shape throughout training would spread format
detection, lineage rules, and loss-target assumptions across backends.

## Decision

- Create the independently versioned `posttrain-data` package under
  `packages/data`, imported as `posttrain.data`.
- Own three immutable canonical snapshots: supervised conversations,
  preference pairs, and task-neutral rollout inputs. Every snapshot has a
  stable dataset ID, source revision, schema version, unique example IDs, and
  JSON-safe metadata.
- Represent conversations as OpenAI-compatible message records and JSON
  function definitions. A supervised example additionally records explicit
  trainable message indices. A preference example records an explicit shared
  prompt, chosen continuation, rejected continuation, optional scores, and
  optional trace lineage.
- Treat Hugging Face Dataset, JSONL, Parquet, and future object stores as
  containers or transports rather than canonical domain types.
- Implement explicit adapters for common Hugging Face/TRL, Alpaca, ShareGPT,
  Tulu, and NeMo row layouts. Avoid a global plugin registry; callers select an
  adapter or use bounded structural detection.
- Project retained Verifiers trace branches into supervised examples through a
  dedicated adapter. Native traces remain authoritative. Selection policy
  controls errors, truncation, and minimum reward, while sampled message
  provenance determines the training targets.
- Keep renderer-specific tokenization, chat-template controls, exact token
  loss masks, packing, and trainer tensors in `posttrain.train`. Pretokenized
  data is a derived cache tied to a model and renderer revision, never the
  canonical dataset.
- Make `posttrain.train` depend on `posttrain.data`; `posttrain.data` may depend
  only on lightweight common contracts and optional boundary libraries. It
  never imports train, eval, serve, TRL, vLLM, or Trackio.

## Consequences

- One curated dataset can feed TRL now and NeMo or another trainer through an
  adapter without changing job code or its lineage identity.
- Environment packages remain native Verifiers packages and are not forced
  into an SFT schema. Successful or human-approved traces can still become SFT
  datasets reproducibly.
- Tool definitions, multi-turn messages, target attribution, preference scores,
  and trace references survive framework conversion.
- External adapters require compatibility tests as upstream schemas evolve.
- The canonical package owns semantic validation but does not own artifact
  publication, Trackio logging, data governance, or model rendering.

## Alternatives Considered

### Use Hugging Face Dataset as the canonical abstraction

Rejected because it combines storage/execution mechanics with domain identity
and makes every consumer depend on Arrow schema behavior.

### Keep data classes inside the training package

Rejected because curation, conversion, evaluation-derived demonstrations, and
other trainer implementations should be reusable without importing training
execution.

### Use the TRL row schema directly everywhere

Rejected because TRL does not capture all canonical provenance and target-mask
semantics, while NeMo and Verifiers expose different native structures.

### Define one universal environment and supervised-data record

Rejected because executable environment tasks contain behavior, resources,
timeouts, rewards, and runtime state that do not belong in static SFT examples.

### Build a global format-adapter registry

Rejected for the MVP because explicit functions are easier to discover, type,
test, and replace. A registry is warranted only if independently installed
third-party adapters become a real requirement.

## Implementation Notes

- Canonical models: `packages/data/src/posttrain/data/models.py`.
- Reusable derivations: `packages/data/src/posttrain/data/sources.py`.
- Hugging Face/TRL and legacy Hub adapters:
  `packages/data/src/posttrain/data/adapters/huggingface.py`.
- NeMo adapters: `packages/data/src/posttrain/data/adapters/nemo.py`.
- Verifiers trace projection:
  `packages/data/src/posttrain/data/adapters/verifiers.py`.
- Training rendering remains in
  `packages/train/src/posttrain/train/rendering.py` and preserves tools plus
  explicit trainable-message attribution.
- The workspace import contract prevents `posttrain.data` from importing
  execution packages or observation backends.

## Revision History

- 2026-07-20: Accepted the independent canonical data package, explicit
  external-format adapters, and Verifiers-trace-to-SFT projection boundary.
