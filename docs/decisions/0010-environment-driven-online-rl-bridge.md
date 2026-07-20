# ADR 0010 — Environment-driven online-RL bridge

## Status

Accepted.

## Context

The first GRPO integration let TRL generate one assistant completion and then
called a Verifiers task only for post-generation scoring. That was sufficient
for a GSM8K smoke, but it did not execute a native Verifiers episode. Tool
calls, harness turns, user simulation, environment stop conditions, group
scoring, and trace construction were either unavailable or reconstructed after
the fact. The adapter also mixed prompt projection, scoring, trace creation,
and evidence preservation.

Verifiers v1 already owns the complete environment lifecycle and records exact
per-branch token IDs, sampling masks, logprobs, rewards, errors, and termination
state. TRL already supports custom rollout output and owns the policy weights,
generation engine, loss, and optimizer. The platform needs a reusable boundary
between those responsibilities without loading a second policy or making TRL,
Verifiers, vLLM, or Trackio the public cross-project abstraction.

## Decision

- Make `posttrain.train.OnlineRLBridge` the public environment-driven rollout
  contract accepted by `GRPORequest`.
- Inject a `PolicyGenerator` into the bridge. A generator performs one model
  turn against the trainer's already-loaded policy and returns structured
  output plus exact prompt/completion tokens, token attribution, logprobs, and
  finish reason.
- Share max output tokens, temperature, and top-p through typed
  `PolicySampling`; the TRL adapter rejects drift between the environment and
  the loaded generator instead of silently using different sampling policies.
- Implement `VerifiersOnlineRLBridge` with native
  `Environment.episode(...)`. Verifiers owns task setup, harness/runtime,
  tools, user simulation, stopping, finalization, per-rollout and group
  scoring, and the authoritative trace.
- Translate one final Verifiers trace branch into `TrainingRollout`: split the
  initial prompt from the trajectory at the first sampled token, retain all
  later model and environment tokens, and set `env_mask=True` only for
  model-sampled tokens.
- Keep `TrlPolicyGenerator` and the TRL `rollout_func` translation private to
  `posttrain.train.backends.trl`. Transformers and colocated vLLM remain
  selectable implementations of the same policy-generator responsibility.
- Pass aligned raw dataset rows into TRL custom rollouts so stable task identity
  never has to be encoded in model-visible prompts. Accept native truncation
  state from custom rollouts rather than inferring it only from the last token.
- Route the returned `TraceObservation` through `ExecutionContext`; the bridge
  does not import Trackio. Preserve native `traces.jsonl` as the authoritative
  artifact independently of the observer's queryable trace copy.
- Reject zero-branch and multi-branch traces in the MVP. A trainer example must
  not silently select one of several trainable branches.

## Consequences

- The same Verifiers environment semantics can be used for evaluation and
  online RL, including linear multi-turn and tool-use trajectories.
- Verifiers does not load model weights, and the trainer does not reimplement
  environment execution or trace semantics.
- Tool, harness, and simulator tokens remain part of the model context while
  being excluded from policy loss through `env_mask`.
- A future trainer, vLLM server, or SGLang implementation can supply another
  `PolicyGenerator` without changing environment packages or job code.
- A future environment implementation can supply another `OnlineRLBridge`
  without changing the TRL adapter.
- The pinned TRL fork carries two additive custom-rollout fields—aligned inputs
  and authoritative truncation—until an upstream interface provides equivalent
  behavior.
- Native episode execution may make several sequential generation calls. Batch
  scheduling and multimodal sidecars require explicit extensions; they are not
  hidden behind the current contract.

## Alternatives Considered

### Keep the post-generation reward callback

Rejected because it reduces Verifiers to a scorer and cannot correctly execute
or observe interactive environments.

### Let Verifiers call a separately served policy

Rejected as the default local path because it duplicates policy residency or
requires a second lifecycle. Remote/server generation remains possible through
a future `PolicyGenerator` implementation.

### Put Verifiers types directly in `GRPORequest`

Rejected because it would make one framework the public training API and make a
future environment or trainer replacement invasive.

### Let TRL own tools and environment state

Rejected because TRL's tool callback model is not the same as a native
Verifiers task, harness, runtime, user simulator, and scoring lifecycle.

### Correlate tasks by prompt text

Rejected because prompts are not stable unique identities. The TRL fork passes
the aligned dataset rows to the custom rollout function instead.

## Implementation Notes

- Public contracts: `packages/train/src/posttrain/train/online_rl.py`.
- Native Verifiers adapter:
  `packages/train/src/posttrain/train/integrations/verifiers.py`.
- TRL policy and rollout adapters:
  `packages/train/src/posttrain/train/backends/trl/online_rl.py` and
  `packages/train/src/posttrain/train/backends/trl/grpo.py`.
- Job composition:
  `apps/lab/src/posttrain_lab/environments/gsm8k_grpo.py` and
  `apps/lab/src/posttrain_lab/jobs/gsm8k_posttraining.py`.
- TRL fork support was merged in
  [`carbonteq-ai/trl#10`](https://github.com/carbonteq-ai/trl/pull/10) and is
  pinned at `a0b4bca78eeeb02abb050abfa04624f952d5f633`.
- Contract tests cover exact turn tokens, native multi-turn masks, reward and
  trace projection, task identity, and native artifact finalization. Trackio
  run `07984dfc3feb44e1b34dcd5b92e2d850` from clean revision `e7babfc` passed
  the one-step GPU runtime gate with two native traces and no additional
  Verifiers-owned policy load. Colocated TRL/vLLM still uses its intentional
  training and rollout representations.

## Revision History

- 2026-07-20: Accepted the environment-driven bridge, replacing the partial
  post-generation Verifiers scoring callback.
- 2026-07-20: Recorded the clean one-step colocated-vLLM GPU acceptance run.
