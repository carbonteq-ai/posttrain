# Qualify Qwen 3.5 thinking rollouts before release

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` current while the work proceeds.
It follows `docs/templates/PLAN.md` and the frozen product baseline under
`docs/post-training/`.

## Purpose / Big Picture

Ambient Agent's Reasoning Gym OLMo 3 canary must preserve genuine reasoning,
finish bounded tasks, produce scorable reward variance, and reach an actor
update. The current 16K canary does not satisfy that contract: one retained
completion consumed the full 16,384-token budget by repeating a short phrase,
three sibling rollouts produced no model result before the rollout deadline,
and the batch failed before optimization.

This plan fixes the generic online-RL sampling and reasoning-mode contracts,
qualifies the result from a locally built Posttrain package on the RTX 4090,
and promotes a release only after a two-prompt-group by four-generation run is
healthy. A release is an output of qualification, never a prerequisite for
discovering whether the change works.

The work does not change the frozen product baseline. `InferenceBinding`
already owns rollout sampling and renderer identity; this work makes those
resolved selections reach every online-RL generator without silent loss and
makes their reasoning-mode provenance truthful.

## Progress

- [x] (2026-08-11) Re-read the canonical API and observation contracts and
  traced the Ambient canary through its packed selections, Verifiers bridge,
  TRL policy generator, retained native trace, and trainer argument builder.
- [x] (2026-08-11) Confirmed the failed canary used 2 prompt groups, 4
  generations, a 16,384-token completion budget, temperature 1.0 and top-p
  0.95. The retained trace contains one 16,389-token sampled assistant node,
  56,645 characters of output, no final answer, and repeated derivation text.
- [x] (2026-08-11) Confirmed that Posttrain currently forwards only
  `max_tokens`, `temperature`, and `top_p` from the inference selection into
  the Verifiers policy bridge. TRL already supports `top_k`, `min_p`, and
  `repetition_penalty`, but Posttrain drops them. Verifiers' pinned
  `SamplingConfig` deliberately permits provider-neutral extra fields.
- [x] (2026-08-11) Merged the independently green failed-run artifact
  finalization repair as PR #48. No new framework release has been started.
- [x] (2026-08-11) Added typed sampling fields and end-to-end propagation, including
  strict environment/generator drift checks and veRL parity.
- [x] (2026-08-11) Proved in focused tests that the training renderer
  supplies `enable_thinking: true`; do not invent a duplicate model variant for
  the same weight bytes.
- [x] (2026-08-11) Added an additive SFT-backed Ambient 16K canary selection
  without mutating historical work packages; local source composition resolves
  the intended SFT LoRA, 2 prompt groups, 4 generations, and complete sampling
  policy. Exact packing and task-identity inspection remain in the run gate.
- [ ] Run the 2x4 RTX 4090 canary from exact local package bytes and inspect all
  eight traces, termination, rewards, advantages, truncation, TIS, clipping,
  actor update, and LoRA-only artifacts.
- [ ] Iterate on evidence-backed configuration or implementation faults until
  the canary passes. Do not release a merely buildable or queued state.
- [ ] After the good run, complete the release checks, publish one exact
  version, update Ambient's dependency and lock, and repeat the acceptance
  canary from released bytes.

## Surprises & Discoveries

- Observation: The inference selection names `qwen3.5-tools@1`, whose default
  reasoning mode is off, while the training binding explicitly selects the
  `thinking` mode. The TRL policy adapter renders with the training renderer
  and passes `enable_thinking: true` to TRL. The two selections describe
  different responsibilities: the model-compatible wire contract and the
  job's explicit training render mode. This is not the cause of the failure.
  Evidence: `packages/train/src/posttrain/train/backends/trl/online_rl.py`
  constructs its renderer from `TrainingBinding`; the retained completion
  begins with `Thinking Process:` even though it contains no separately parsed
  `reasoning_content`.
- Observation: The retained failure is not ordinary insufficient context. The
  model had already derived the task structure and then repeated instead of
  terminating. Raising the cap alone would make the failure slower and more
  expensive.
- Observation: The current failure-artifact repair was required to preserve
  the native partial trace. Without it, this diagnosis would have depended on
  transient worker state.
- Observation: Sampling is originally selected by the environment and must be
  realized by the rollout inference binding. Treating either copy as silently
  dominant allows drift. The request builder now resolves both into one typed
  value and rejects any mismatch before creating the native bridge.
- Observation: Ambient's released virtual environment correctly rejects the
  new sampling fields because it contains the old schema. Local qualification
  composes the unreleased environment/train/work/eval packages into Ambient
  without changing its stable dependency pin.

## Decision Log

- Decision: Preserve thinking mode and the 16K budget during qualification.
  Rationale: the goal is reasoning training, and the observed failure is a
  repetition/termination fault rather than evidence that reasoning should be
  disabled.
  Date/Author: 2026-08-11 / Codex.
- Decision: Treat the environment declaration and rollout inference binding as
  two representations of one sampling policy and require exact equality after
  defaults are resolved. Propagate that value into Verifiers, TRL and veRL.
  Rationale: the environment owns requested episode behavior while inference
  owns engine realization; neither may silently override the other.
  Date/Author: 2026-08-11 / Codex.
- Decision: Keep provider-specific generation escape hatches behind the
  backend adapter, but promote commonly shared controls (`top_k`, `min_p`,
  repetition and presence penalties) into the backend-neutral online-RL turn
  contract.
  Rationale: these affect the sampled behavior policy, its log probabilities,
  and therefore TIS evidence. They cannot remain undocumented vLLM-only state.
  Date/Author: 2026-08-11 / Codex.
- Decision: Block release promotion until a local 2x4 canary reaches a finite
  actor update with scorable, nondegenerate evidence.
  Rationale: local exact-package qualification is faster than publishing
  speculative versions and separates code readiness from distribution.
  Date/Author: 2026-08-11 / Codex.

## Outcomes & Retrospective

Not complete. The accepted outcome must name the exact Posttrain and, if
changed, TRL commits; the exact Ambient package key; the local provider and
Trackio run identities; the eight rollout termination/reward summaries; actor
update evidence; and the final release/pin only after both local and released
canaries pass.

## Context and Orientation

`packages/train/src/posttrain/train/online_rl.py` owns the backend-neutral
`PolicySampling` value passed between an environment and its policy generator.
`packages/train/src/posttrain/train/verifiers_requests.py` converts an
`InferenceBinding` into a Verifiers rollout bridge.
`packages/train/src/posttrain/train/integrations/verifiers.py` injects those
controls into native Verifiers episodes and projects individual turns back to
the trainer. TRL generation is adapted in
`packages/train/src/posttrain/train/backends/trl/online_rl.py`, while its
trainer arguments are built in `packages/train/src/posttrain/train/backends/trl/grpo.py`.
The equivalent veRL turn bridge is
`packages/train/src/posttrain/train/backends/verl/agent_loop.py`.

The project configuration lives in `/home/hammad/projects/ambient-agent`.
The immediate work package is
`.posttrain/work_packages/k1_reasoning_gym_olmo3_sft_2b_4090_canary16k_1step.yaml`.
Historical selections and runs remain immutable; corrected selections receive
new ids or revisions.

## Plan of Work

First, define one complete `PolicySampling` value and validation semantics.
The inference binding's numeric values are decoded once, included in the
Verifiers native `SamplingConfig`, reconstructed at each environment request,
passed to TRL/veRL generation, and compared against the already loaded
generator before tokens are sampled. The comparison includes defaults so that
omitted values have deterministic meaning.

Second, preserve the existing reasoning ownership and prove it. The training
binding selects `thinking`, the model renderer declares that this mode is
supported, and the trainer arguments must contain `enable_thinking: true`.
There is no need to duplicate one immutable model weight state merely to change
a default that the training job already overrides explicitly.

Third, validate the exact Ambient package locally: catalog resolution, job
plan, dataset/task selection, packed manifest, renderer arguments, sampling
arguments, and dependency/source digests. The job keeps two prompt groups,
four generations, 16K completion budget, OLMo 3 active sampling, mean-only
advantages, asymmetric clipping, token TIS, truncation masking, and LoRA
updates. Concurrency remains bounded for the 24 GiB card.

Finally, run one logical update. A good run needs eight completed rollout
attempts; bounded generation that either stops or cleanly reaches a declared
limit; at least one scorable group with reward spread; centered positive and
negative advantages; finite entropy and gradients; TIS and clip telemetry;
one completed actor update; and adapter-only model/recovery artifacts. Any
failure is diagnosed from retained traces before changing another knob.

## Concrete Steps

Work in `/home/hammad/projects/rl` unless a command names Ambient Agent.

1. Add focused unit tests for the full sampling value, catalog-to-bridge
   translation, Verifiers round trip, TRL drift detection/arguments, and veRL
   request parameters. Make those tests fail before implementation.
2. Implement the smallest backend-neutral contract and adapter changes. If
   presence penalty requires a maintained TRL change rather than its supported
   `generation_kwargs`, make that change in `/home/hammad/projects/trl`, test
   it there, commit/publish the fork first, and only then update Posttrain's
   immutable dependency pin.
3. Add a regression proving that the Reasoning Gym training path resolves and
   forwards thinking-mode template kwargs. Keep ordinary non-thinking bindings
   valid.
4. Run focused tests, Ruff, Pyright, import contracts, package tests and
   `git diff --check`. Build a local wheelhouse from the exact candidate tree.
5. In Ambient, add versioned thinking-compatible model/inference/work-package
   selections. Run `posttrain catalog validate`, `posttrain job plan`, and
   `posttrain job pack`; inspect `package.json` rather than trusting YAML.
6. Submit the exact locally built package to the RTX 4090 and wait through the
   first logical update. Reconcile provider and Trackio evidence before calling
   it terminal.
7. Query the native rollout artifact and trainer metrics. Record per-rollout
   task identity, output length, stop reason, answer, reward, error and retained
   status, then the update-level advantage, entropy, TIS, clipping, active
   sampling, truncation, gradient and artifact evidence.
8. Only after acceptance, prepare the release PR/candidate from these exact
   commits, publish stable bytes, update Ambient's dependency/lock, and rerun
   the same canary without changing its experiment contract.

## Validation and Acceptance

Local source acceptance requires focused tests for every new field and drift
condition, all affected package tests, Ruff, Pyright, import contracts and a
clean diff check. Packed-job acceptance requires the manifest to resolve the
thinking renderer with `enable_thinking: true`, temperature 1.0, top-p 0.95,
top-k 20, and the chosen anti-repetition controls identically in the environment,
bridge and trainer. `reasoning_parser` is not an acceptance field for this
colocated raw-token training path; the training renderer, not an OpenAI response
parser, owns reasoning mode here.

Run acceptance requires a completed optimizer step, not merely a running or
terminal provider job. No rollout may enter an unbounded repetition loop; all
errors and truncations are retained; at least one four-generation prompt group
has nonzero reward variance; advantage mean is approximately zero with both
signs present; TIS is finite and clamping is reported; asymmetric clipping is
reported; entropy and gradient norm are finite; and the output contains only a
LoRA model view plus an adapter/trainer recovery checkpoint, never full model
weights.

Release acceptance requires the same experiment to pass from exact released
bytes after stable package and OCI readback. A local pass alone authorizes the
release process; it is not a substitute for the released-byte canary.

## Idempotence and Recovery

All Ambient selection changes are additive. Every pack produces a
content-addressed image, and every execution uses a new run id. Failed local
runs remain evidence and are not relabeled or overwritten. A one-step run has
no resumable state until its first complete checkpoint; if it fails earlier,
fix the cause and create a new attempt. Release workflows are not dispatched
until local acceptance, so repeated local iteration cannot publish accidental
versions.

## Artifacts and Notes

Retain the failed native trace digest
`86b38943113485752a223653f938a3d42fb792e8259f9dcad5632dfec587c41d`
as the before-fix evidence. Retain each candidate's source commit, wheel hashes,
package JSON/key, local provider id, Trackio id, native rollout artifact,
metrics query and LoRA manifest. Do not store tokens, credentials, signed URLs
or private registry authentication in the plan.

## Interfaces and Dependencies

The public product nouns remain `InferenceBinding`, `TrainingBinding`,
`GRPOSettings`, `EnvironmentBinding`, run, trace and artifact. The maintained
external dependencies are the pinned Verifiers revision
`284a868d6a9022109b749710672a0460e8a996d4` and CarbonTeq TRL. Verifiers extra
sampling keys are allowed by its pinned `SamplingConfig`; TRL source and tests
remain authoritative for which controls affect colocated vLLM and Transformers
generation. Any TRL edit must follow `docs/tooling/forks.md`, update both fork
and consumer ledgers, and be committed before Posttrain pins it.

Revision note (2026-08-11): created after the 16K Reasoning Gym canary exposed
that resolved sampling controls were being dropped. Investigation then proved
the training renderer already selected thinking mode; the failure was an
unbounded repetition/termination pathology, not a disabled-thinking path.
Release promotion was explicitly moved after a successful local 2x4
qualification.
