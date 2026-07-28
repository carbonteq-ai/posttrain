# Qualify Policy Prism Gemma 4 on-policy distillation

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` current in accordance with
`docs/templates/PLAN.md`. The canonical product authority is
`docs/post-training/README.md` and the six documents it indexes.

## Purpose / Big Picture

Extend the existing `train.distill` path so a project can train independent
BF16 LoRA adapters for `google/gemma-4-E4B-it` from an unquantized local
`google/gemma-4-31B-it` teacher on one H200. Policy Prism owns prompt-only task
selection, staged JSON schemas, native Verifiers traces, and sealed before/after
evaluation. PostTrain owns model loading, live student rollouts, exact-token
teacher scoring, optimization evidence, artifacts, and Observatory views.

The shortest supported route is a standard `train.distill` job using
`train/trl-distill@1`, launched in process on a user-provisioned RunPod. This
plan does not add a custom Lab trainer, a RunPod provider, offline teacher
answers, or any quantization.

Scope and recovery never share an adapter. The first paid experiment is scope
only: a one-step smoke, an eight-step qualification, and a 64-step pilot.
Recovery remains CPU-qualified but its catalog packages and GPU run are deferred.

## Progress

- [x] (2026-07-28 17:00Z) Verified the clean worktree
  `exp/policy-prism-gemma4-distill` at PostTrain v0.2.3 commit
  `5522b5ea84669401d8f217d7016afe04a80bb381`.
- [x] (2026-07-28 17:15Z) Read the canonical product documents, plan template,
  existing Verifiers distillation plan, pinned TRL fork implementation, train
  adapters, catalogs, work packages, tests, and Observatory telemetry.
- [x] (2026-07-28 17:25Z) Confirmed the Policy Prism branch has no distillation
  environment at `c4cc8618e796b48e34a8768bf1967cbc0b82e623`; a final pin does not
  yet exist.
- [x] (2026-07-28 17:14Z) Implemented and CPU-tested the Gemma 4 renderer,
  conditional-generation loading, local frozen BF16 teacher loading, and
  language-model-only LoRA targeting.
- [x] (2026-07-28 17:14Z) Implemented and CPU-tested JSON Schema and per-stage
  sampling propagation through
  Verifiers and the TRL policy generator.
- [x] (2026-07-28 17:14Z) Enforced prompt, flattened-completion, and total
  trajectory budgets before
  teacher scoring.
- [x] (2026-07-28 18:15Z) Verified the exact pinned E4B and 31B tokenizer files
  are byte-identical and recorded the shared ordered-vocabulary/special-token
  fingerprint `1ab787c816b67a0936e8d1c9ff20e6cf5bd8b77faabfe6ada5905bd2c433b413`.
- [x] (2026-07-28 18:25Z) Added independent project selections for both Gemma
  models, one H200 target, colocated E4B vLLM rollout, local BF16 31B teacher,
  text-only rank-8 LoRA, and shared 1/8/64-step settings.
- [x] (2026-07-29 00:25Z) Added and validated the three scope Policy Prism environment bindings and
  standard work packages pinned to Policy Prism commit
  `bfa7802f4e8250803f11fdba242608fb419acc8d`.
- [x] (2026-07-28 17:14Z) Extended per-step distillation telemetry,
  checkpoint runtime phases, and Observatory charts, evidence groups, and
  health alerts. Focused validation has 84 common/train tests with Verifiers
  active and 45 non-HTTP Observatory tests passing. The default package suites
  have 16 common and 138 train tests passing, with expected optional skips.
- [x] (2026-07-28 19:05Z) Completed focused common, train, Verifiers bridge,
  Observatory, catalog, work-package, Ruff, changed-file Pyright,
  import-boundary, and diff validation.
- [x] (2026-07-28 19:20Z) Committed and pushed the independent PostTrain
  support as `b5396d0f58113d85f48be8db61ed795313c10cd5`.
- [x] (2026-07-29 00:25Z) Clean-built Policy Prism 0.4.0 and verified exact
  catalog equality, 1/8/64 scope task counts, source-only fields, native JSON
  Schema, stage limits, and independently conditioned scope/recovery rows.
- [x] (2026-07-29 00:25Z) Passed 168 common/train tests, 12 catalog tests, 45
  focused Observatory tests, 13 work-package tests, three static package
  validations, changed-file Ruff/Pyright, all import contracts, project doctor,
  and diff checks.
- [x] (2026-07-29 00:35Z) Committed and pushed the scope integration support as
  `8f91a333b62ca2f9a851dd9a3216a89e0dfb8682`.
- [x] (2026-07-29 00:55Z) Provisioned a USD 3.59/hour Community H200, installed
  the locked Python 3.13/Torch 2.11/CUDA 13 stack, verified real H200 compute,
  cached both exact Gemma revisions, and loaded 1/8/64 scope tasks.
- [x] (2026-07-29 01:05Z) Passed a clean one-step H200 smoke at PostTrain
  `e09550d6c557b8259b16436df9f7568cc74ee1e6`: finite loss/reverse KL and
  gradient norm, 198 scored tokens, zero teacher failures, 104.78 GiB peak GPU
  memory, four retained traces, and complete adapter/checkpoint artifacts.
- [x] (2026-07-29 01:30Z) Diagnosed the first eight-step qualification stop
  from its native Trackio trace. Policy Prism completed the trajectory with no
  errors and `agent_completed`, while its final structured stage ended at the
  environment-owned token cap. Narrowed distillation admission so this explicit
  environment-success state is trainable while provider errors, incomplete
  episodes, and framework-limit truncations remain blocked.
- [ ] Rerun and pass the eight-step H200 qualification, then apply the measured
  timing/budget admission gate before starting the 64-step pilot.

## Surprises & Discoveries

- Observation: v0.2.3 already implements backend-neutral `train.distill`, a
  native Verifiers rollout bridge, consume-once batch lineage, a local
  Transformers teacher, and colocated vLLM student rollout.
  Evidence: `packages/train/src/posttrain/train/backends/trl/distillation.py`
  and `docs/plan/verifiers-on-policy-distillation.md`.

- Observation: the pinned TRL fork converts a mapping in
  `vllm_generation.generation_kwargs["structured_outputs"]` into vLLM
  `StructuredOutputsParams` for every colocated generation call. This provides
  the required extension point without a TRL fork change.
  Evidence: pinned TRL commit `6e7739b8ec741d21ecd79c0c212694cd15ff20d8`,
  `trl/generation/vllm_generation.py::VLLMGeneration.generate`.

- Observation: the Policy Prism branch currently contains no training
  environment, so pinning `c4cc8618...` would produce a reproducible but
  non-functional
  training selection.
  Evidence: `git status --short` in the Policy Prism checkout lists modified
  harness, program, schema, and tests.

- Observation: Transformers 5.14 exposes cumulative non-padding tokens to
  callbacks as `num_input_tokens_seen`, not `num_tokens`.
  Evidence: the pinned `TrainingArguments`/`Trainer` implementation and the
  focused callback test. The adapter now normalizes that native key to the
  framework-owned `train/num_tokens` name.

- Observation: Observatory already projects model loading, rollout, teacher
  scoring, actor update, and artifact export phases. Checkpoint saves were the
  missing interval.
  Evidence: `apps/observatory/src/posttrain_observatory/runtime_phases.py` and
  the new observed distillation-trainer checkpoint test.

- Observation: repository-wide Pyright has existing v0.2.3 re-export errors in
  Lab, Work, Execution, Tracking, and unrelated tests. Every file changed by
  this plan type-checks with zero errors; all eight import-linter contracts and
  repository-wide Ruff pass.
  Evidence: focused Pyright command over the changed-file set, `lint-imports
  --cache-dir /tmp/posttrain-policy-prism-import-linter-cache`, and
  repository-wide Ruff. No unrelated type errors were edited.

- Observation: the Observatory HTTP test module blocks on its first existing
  `TestClient` request in this restricted worktree, while all 45 service,
  projection, settings, and execution-target tests pass.
  Evidence: bounded test runs of `apps/observatory/tests/test_http.py` and the
  four remaining Observatory test modules.

- Observation: tokenizer JSON and tokenizer configuration files are
  byte-identical for the pinned E4B and 31B revisions. Canonicalizing the
  ordered vocabulary, added-token special flags, and special-token
  configuration produces the same fingerprint for both models.
  Evidence: exact Hugging Face files at revisions
  `ee0ef6023621cff504d758262d4e04895a5af4a2` and
  `842da3794eaa0b77d5f08bae87a17459d91ff475`.

- Observation: Verifiers-enabled catalog/work-package tests that activate
  `gsm8k-v1` or `automationbench-v1` require those external environment
  packages. The implementation-specific bridge tests and all work-package
  tests not requiring those packages pass.
  Evidence: 69 Verifiers/TRL tests and eight focused work-package tests pass;
  one GSM8K and one AutomationBench activation test are explicitly deselected.

- Observation: Policy Prism's staged program issues each stage as an
  independent model call, so Verifiers exposes three disjoint trace branches
  for scope and two for recovery rather than one conversational branch.
  Evidence: the real `bfa7802...` smoke environments and the cross-repository
  bridge test.

- Observation: Verifiers supplies the rollout-level sampling cap separately
  from the harness request. PostTrain must preserve a smaller positive
  per-call `max_tokens` from the staged request while keeping the rollout cap
  authoritative.
  Evidence: the real Policy Prism bridge requests 512/1536/768 tokens for
  scope and 512/2048 for recovery.

- Observation: vLLM 0.25.1's XGrammar backend rejects semantic JSON-Schema
  keywords including `uniqueItems`. The first real smoke therefore produced a
  completed Verifiers trace whose stop condition was `error`; the prior
  distillation adapter still consumed its sampled error tokens.
  Evidence: H200 smoke run `c3813db3-9451-4d7f-b075-690ba5ff0f76` reported a
  `HarnessError` with `Grammar error: Unimplemented keys: ["uniqueItems"]`,
  one failed rollout, and 123 scored tokens.

- Observation: the corrected smoke produced three healthy independently
  conditioned stage rows, but their copied Verifiers payloads retained one
  original trace ID. Trackio uses that payload ID as its storage identity and
  therefore retained only one of the three observations.
  Evidence: smoke run `85505897-7f83-4780-aa72-c461690c0db5` optimized 198
  scored tokens with zero teacher failures, while Trackio stored one trace
  whose metadata reported `branch_count=3`.

- Observation: Policy Prism deliberately treats both `stop` and `length` as a
  completed staged rollout when the harness inventory finishes. On the first
  qualification attempt, several student stages exhausted their local
  512/1536/768-token limits; their native traces nevertheless had
  `is_completed=true`, `stop_condition=agent_completed`, no provider errors,
  and usable sampled-token sequences. Exact-token reverse-KL distillation can
  score those student tokens even when the final structured text is incomplete.
  Evidence: qualification run `a9613e08-83fb-4b48-808d-27860b8a4f3d`, its
  native Trackio traces, and
  `PolicyPrismDistillTask.finalize` at Policy Prism `bfa7802...`.

## Decision Log

- Decision: No frozen product-baseline amendment is required.
  Rationale: the existing baseline already assigns live environments to
  project selections, optimization to `train.distill`, and evidence queries to
  Observatory. This work uses those extension points without changing their
  meaning.
  Date/Author: 2026-07-28 / Codex.

- Decision: Do not create a runnable Policy Prism environment catalog entry
  until the external wheel is committed and its full SHA is known.
  Rationale: a placeholder or the pre-feature SHA would violate immutable
  selection and preflight guarantees.
  Date/Author: 2026-07-28 / Codex.

- Decision: Apply per-turn structured decoding by temporarily overriding the
  pinned TRL generator under the adapter's existing async lock and restore it
  in `finally`.
  Rationale: staged environments require different schemas and token limits;
  shared mutable generation state must never leak between turns or failures.
  Date/Author: 2026-07-28 / Codex.

- Decision: Keep the pinned TRL fork unchanged.
  Rationale: its colocated vLLM generator already accepts per-call structured
  output mappings and supports LoRA-only weight synchronization; the framework
  adapter can use those supported hooks directly.
  Date/Author: 2026-07-28 / Codex.

- Decision: Train scope and recovery as separate adapters with 1/8/64-step
  smoke, qualification, and pilot settings.
  Rationale: the tangents have different stage contracts and must be evaluated
  independently; combining them would obscure which policy improved.
  Date/Author: 2026-07-28 / Codex.

- Decision: Run paid GPU qualification for scope only in this execution.
  Rationale: one retained adapter and a roughly USD 20 Community H200 cap are
  the current goal; recovery remains a separate future experiment.
  Date/Author: 2026-07-29 / Codex.

- Decision: For distillation only, project independent root branches as
  separate exact-token training rows and expand each trainer input through an
  explicit source-index mapping. Continue rejecting multi-branch GRPO/SAMPO
  and forked rather than independent distillation traces.
  Rationale: every Policy Prism stage must contribute trainable tokens while
  teacher and student logits retain that stage's actual prompt conditioning.
  Date/Author: 2026-07-29 / Codex.

- Decision: Accept Policy Prism's approximately 14-token dependency-label
  budgeting undercount for the selected source data.
  Rationale: the selected rows have headroom and PostTrain enforces the actual
  rendered-token limits before teacher scoring.
  Date/Author: 2026-07-29 / User and Codex.

- Decision: Keep only identities required for correct execution: exact model
  revisions, shared tokenizer token-ID fingerprint, final Policy Prism commit,
  and the automatically emitted adapter identity.
  Rationale: wheel hashes, lock hashes, dataset hash chains, and new comparison
  reports add experiment overhead without being required by these runtime
  contracts.
  Date/Author: 2026-07-28 / Codex.

- Decision: Strip XGrammar's documented unsupported semantic constraints only
  from the temporary vLLM generation copy, while retaining Policy Prism's
  canonical schema for its local Draft 2020-12 validation. Reject explicit
  provider errors, incomplete episodes, error stop conditions, and
  framework-limit truncations before teacher scoring. Admit a token-limited
  rollout only when the environment explicitly records `is_completed=true`,
  `stop_condition=agent_completed`, and no errors.
  Rationale: an infrastructure failure must never become distillation
  supervision, while Policy Prism intentionally exposes length-bounded student
  samples for exact-token on-policy teacher scoring.
  Date/Author: 2026-07-29 / Codex.

- Decision: Give each independently projected stage branch a payload ID equal
  to its unique framework external ID while retaining `original_trace_id` in
  attributes and the native JSONL replay.
  Rationale: training rows and Observatory traces must have the same one-to-one
  identity; otherwise the tracking backend legitimately deduplicates them.
  Date/Author: 2026-07-29 / Codex.

## Outcomes & Retrospective

The reusable framework slice, independently conditioned staged-row bridge,
scope catalog, and three scope work packages are implemented. The final CPU
gate passes: 168 common/train tests, five clean-wheel Policy Prism tests, 12
catalog tests, 45 Observatory tests, and 13 work-package tests. Ruff,
changed-file Pyright, all eight import contracts, static package validation,
project doctor, and diff checks pass. The first H200 smoke correctly exposed
an XGrammar incompatibility and a failed-rollout admission gap; their focused
regressions pass locally. A clean GPU smoke, qualification, pilot, and local
adapter export remain.

## Context and Orientation

`packages/common/src/posttrain/common/variants/` owns built-in renderer
contracts and exact foundation variants. `packages/train/src/posttrain/train/`
owns backend-neutral turn requests and private TRL adapters.
`packages/train/src/posttrain/train/integrations/verifiers.py` translates the
native Verifiers request/response protocol. The project overlay belongs under
`.posttrain/catalog/`; standard work packages belong under
`.posttrain/work_packages/`. `apps/observatory` remains read-only.

Policy Prism supplies paired scope and recovery tasks. Each staged task may
request an OpenAI-compatible `response_format` of type `json_schema` and a
stage-specific `max_tokens`. The complete trajectory limits are 4,096 prompt
tokens, 12,288 flattened completion tokens, and 16,384 total tokens. Oversized
trajectories are rejected, never truncated.

## Plan of Work

First add a tokenizer-native Gemma 4 renderer contract and exact E4B/31B model
facts. Add one private model-factory resolver that reads pinned Transformers
configuration with `trust_remote_code=False`, prefers a declared locally
available `*ForConditionalGeneration` class, and otherwise uses
`AutoModelForCausalLM`. Reuse it for both the trainable student and frozen local
teacher. For Gemma 4 LoRA, validate resolved module names before PEFT wrapping
and reject any vision/audio match.

Next add immutable `JsonSchemaResponse` to the public train contract and carry
it on `PolicyTurnRequest`. Parse only OpenAI `json_schema` response formats in
the Verifiers bridge. In `TrlPolicyGenerator`, allow a turn token limit up to
the global maximum, temporarily apply that maximum and the schema to the
colocated vLLM generation arguments, use bounded Gemma JSON whitespace, and
restore all state even on an exception.

Then validate every completed rollout against the configured distillation
limits before returning it to TRL for teacher scoring. The rejection names the
example, tangent, three observed lengths, and their limits.

The project overlay independently selects exact Gemma variants, BF16 rank-8
LoRA, local BF16 teacher, a 2-GiB colocated vLLM KV cache, and an H200 141-GB
target. Copy Policy Prism's exact serialized environment activations and add
scope smoke, qualification, and pilot work packages using the existing
1/8/64-step settings. Recovery packages are deferred, while its real bridge
remains CPU-tested.

Finally expose objective, optimizer, supervision/runtime, and teacher metric
groups in Observatory. Treat non-finite objectives, teacher failures, zero
scored tokens, missing required metrics/traces, and failed/partial status as
errors. Wrap native trainer checkpoint saves as `checkpointing` runtime phases
so system metrics can be correlated with every required distillation phase.

## Concrete Steps

Run all commands from `/home/ali-awais-safdar/Post-Train/posttrain-policy-prism`.

    uv run pytest packages/common/tests/test_model_variants.py \
      packages/train/tests/test_trl_online_rl.py \
      packages/train/tests/test_verifiers_grpo_bridge.py \
      packages/train/tests/test_api.py
    uv run pytest apps/observatory/tests
    uv run ruff check packages/common packages/train apps/observatory
    uv run pyright
    uv run lint-imports

After focused tests pass, run package-scoped Ruff, Pyright over changed files,
`uv run lint-imports`, catalog/work-package validation, and `git diff --check`.
Do not expand into long unrelated suites unless focused validation exposes a
cross-package risk.

On the H200, validate selections, run the scope one-step smoke, inspect its
Verifiers trace and adapter artifact, then run the scope eight-step
qualification before one independent 64-step pilot. Stop if teacher failures are
non-zero, scored tokens are zero, an
objective is non-finite, or available VRAM drops below the qualification
headroom gate.

## Validation and Acceptance

CPU acceptance requires tests proving renderer resolution, conditional-loader
selection/fallback, multimodal LoRA exclusion, first/subsequent-turn schema
preservation, temporary generation-state restoration, pre-teacher overlength
failure, normalized per-step metrics, and deterministic health alerts.

GPU acceptance requires a successful H200 scope smoke with one native Verifiers
trace, finite loss/reverse-KL/gradient norm, non-zero scored tokens, zero
teacher failures, and retained adapter plus summary. Qualification requires
eight scope training tasks. Pilot acceptance requires 64 scope training tasks
and no error alert. Improvement is determined only by Policy Prism sealed
before/after evaluation, not monotonic training loss.

## Idempotence and Recovery

All source and catalog edits are additive. The work packages use immutable
selection IDs and can be retried with a new run identity. Failed runs preserve
their traces and recovery checkpoint. Never replace an external SHA in place;
publish a new selection revision. Do not reuse a partial Trackio directory as
evidence for a new run.

## Artifacts and Notes

Required execution identity is limited to the final Policy Prism full commit,
both exact model revisions, the shared tokenizer token-ID fingerprint, and the
adapter identity emitted by training. Existing Observatory evidence remains
available, but this plan adds no wheel/lock hashes, dataset hash chain,
comparison tooling, or reproducibility report.

## Interfaces and Dependencies

The completed public train interface includes:

    @dataclass(frozen=True, slots=True)
    class JsonSchemaResponse:
        name: str
        schema: Mapping[str, JsonValue]
        strict: bool

    @dataclass(frozen=True, slots=True)
    class PolicyTurnRequest:
        ...
        response_format: JsonSchemaResponse | None = None

The implementation stays on PostTrain v0.2.3's immutable TRL, Verifiers,
Transformers 5.14, renderers 0.1.8, and vLLM 0.25.1 dependency set unless a
focused test proves a fork change is unavoidable.

Plan update note (2026-07-28): created the implementation plan after verifying
the clean worktree, current framework extension points, exact pinned TRL hook,
and the external Policy Prism branch state.

Plan update note (2026-07-28): completed the reusable Gemma, structured-turn,
trajectory-budget, checkpoint-phase, and distillation-observability slices.
Recorded focused test evidence and retained the immutable external-pin gates.

Plan update note (2026-07-28): verified shared tokenizer identity, added the
independent Gemma/H200/LoRA/1-8-64 catalog selections, separated the scope and
recovery adapters, and reduced new lineage requirements to runtime-mandatory
identities only.

Plan update note (2026-07-28): completed the independent CPU validation ladder.
The two deselected work-package activations require unrelated external GSM8K
and AutomationBench taskset packages; no Policy Prism or changed-code failure
remains.

Plan update note (2026-07-29): pinned Policy Prism `bfa7802...`, added the three
scope packages, and changed staged distillation projection from an unsafe
synthetic concatenation to independently conditioned rows. The clean wheel and
all CPU release gates pass; only commit/push and the budget-capped H200 run
remain.
