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
qualifies the result from a locally built Posttrain package, first on one
complete 32-prompt by eight-generation cold-start update, and promotes a
release only after that update is healthy. A release is an output of
qualification, never a prerequisite for discovering whether the change works.

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
- [ ] Run the 32x8 RTX PRO cold-start gate from exact local package bytes and
  inspect task mix, termination, rewards, active-sampling retention,
  advantages, truncation, parity, TIS, clipping, actor update, and LoRA-only
  artifacts.
- [x] (2026-08-11) Audited the broader 32x8 RTX PRO attempt before update one:
  800 rollouts covered 100 prompts across ten generators, but 67.9% truncated,
  Graph Color's structured-answer scorer raised on every row, and active
  sampling retained only 16 of the required 32 prompt groups after masking.
- [x] (2026-08-11) Repaired the Reasoning Gym cascade locally so tasks with a
  metadata-native oracle do not enter string fallbacks, and repaired
  Observatory's lightweight trace projection to use the already returned
  `is_truncated` attribute without hydrating full transcripts.
- [ ] Qualify the additive cold-start task binding on one exact update, then
  promote the same environment/sampling contract to the longer campaign.
- [x] (2026-08-11) Built exact local framework wheels, validated the Ambient
  catalog with current source, and packed a network-isolated qualified OCI
  image. Package key `d886422c6bfa35a358164cb434920494a118fedd6ca11cca34b5f2256a81f651`
  contains 512 rows across only the four selected generators and no unrelated
  datasets.
- [x] (2026-08-11) The first one-update cold-start gate
  `ambient-k1-reasoning-gym-coldstart-v1-rtxpro-g8-1step-20260811-r1`
  (provider `pt-bc82d1171bf5309f1b8040dc`, image digest
  `sha256:6514843ab6b189a720275ebfec6cdc86dcf7e7d6c22d3fc95d798f60fb753527`)
  produced all 256 traces but failed before optimization because the policy
  parity gate compared processed sampling log probabilities against raw actor
  log probabilities.
- [x] (2026-08-11) Repaired the maintained TRL fork in
  `/home/hammad/projects/trl-1.9-upgrade`: raw teacher-forced policy parity now
  has an independent metric and gate, while processed sampling deltas remain
  the TIS signal. Focused parity, generation, and OLMo 3 tests pass.
- [x] (2026-08-11) Corrected canary `r5` ran as provider
  `pt-de83fd53d94069fd4429767a` from package key
  `a2033b7e3eefecc18159782e3c439df47bf205d660e5be84a43cbc9593ff52d1`
  and actual-job image
  `sha256:f7700e89b386c3370d81e2871d2ae6cba92726d3c3a1d899905e597e5f056bb4`.
  Raw policy parity no longer caused a false failure, but the four-task v1
  curriculum exhausted four active-sampling rounds with 232/256 retained rows,
  so the trainer correctly refused to update.
- [x] (2026-08-11) Hardened actual-job builds so the BuildKit named-context
  identity includes the package key. Independent manifest and file-digest
  checks remain in place. This follows two locally caught stale-context
  failures and removes the need for developers to prune builder cache between
  unlike packages.
- [x] (2026-08-11) Added durable normalization for the raw policy-parity
  mean, maximum, and token-count metrics instead of treating absence of a
  runtime failure as sufficient evidence. The focused framework ladder passes
  200 tests with 4 skips; Ruff, Pyright, import contracts, and diff checks pass.
- [x] (2026-08-11) Replaced the v1 cold-start mixture with an additive v2
  selection over syllogism, letter counting, and number sorting. The 6,144-row
  pool defers leg counting because its truncation cost displaced usable groups.
- [x] (2026-08-11) V2 canary
  `ambient-k1-reasoning-gym-coldstart-v2-rtxpro-g8-1step-20260811-r1` is
  provider `pt-8443e5ce7bd3a658dba2136c` from package key
  `9e42da02d311b8903d5e8af9773ba748eaed4c72775abdf4fe2be520d93379cc`
  and actual-job image
  `sha256:89e965d77c9053ea972ed3554957915d1101866b965697b324da7d6d8326a520`.
  The package build is also the first live qualification of the package-keyed
  BuildKit named-context path. It retained only 14/32 groups in round one and
  9/18 in round two, so it was cancelled before spending two further rounds on
  an inefficient initial mixture.
- [x] (2026-08-11) Signal-first v3 canary
  `ambient-k1-reasoning-gym-coldstart-v3-syllogism-rtxpro-g8-1step-20260811-r1`
  completed as provider `pt-753fe23aa473bdf3a9e3b828` from package key
  `6ab2298b55027f5e0eecda6363226076ab3e546624e6c95fa6a03a16dbb1fd4f`
  and actual-job image
  `sha256:28753ab3b4c4d9ec93a8e94a8095337fba0ba5e427fec81557aa103a17706d11`.
  Packing this unlike package without pruning also live-qualified the
  package-keyed named-context cache boundary a second time. It completed one
  optimizer step after 328 generated rollouts and three candidate rounds.
- [x] (2026-08-11) Audited the completed v3 update. Advantage mean was
  `7.70e-09` with 36.95% positive and 35.23% negative values; raw policy-parity
  mean delta was `0.01354` over 16,384 tokens; TIS mean was `0.97636` with
  0.516% clamped; entropy was `0.55826`; policy loss was `0.08637`; gradient
  norm was `0.03637`; and lower/upper clipping fractions were
  `4.17e-06`/`0`. The step took 949.27 seconds, including 579.70 seconds across
  three rollout waves and 303.97 seconds for the actor update.
- [x] (2026-08-11) Verified the step-one checkpoint pair. The model view is a
  53,692,780-byte `model-adapter`; the 121,324,580-byte recovery view carries
  LoRA, trainer, optimizer, scheduler and RNG state. Metadata and paired view
  digests verify. Provider-side deep blob verification reports `unsupported`,
  not success, and remains a release gate rather than being silently accepted.
- [x] (2026-08-11) Recomputed every v3 reward from the retained answer and
  oracle under strict terminal correctness. Only 6/32 first-wave groups, 0/5
  second-wave groups and 2/4 third-wave groups retain two-sided correctness
  signal. The prior native decimal scorer converted incidental oracle text and
  response length into partial reward: 99 correct verbose answers and 86
  incorrect/ambiguous answers changed classification under strict replay.
- [x] (2026-08-11) Repaired the local Reasoning Gym candidate so binary
  syllogisms require an unambiguous terminal decision and added a reusable
  `boxed_exact` reward mode. That mode extracts one boxed answer, invokes the
  selected generator's native verifier on only that answer, and reduces native
  partial scores to exact zero/one. Seven package tests and focused Ruff/diff
  checks pass; candidate wheel SHA-256 is
  `8b55fbcc50d6182f9fc9b9ea3b2c545986b110daa9fde607be85d8291e9b96b5`.
- [x] (2026-08-11) Added deterministic MATH level/type filtering to the local
  `math-python-v1` candidate so policy-relative Levels 2-4 can be selected
  before type-balanced ordering. The package's lock, static analysis, tests,
  wheel build, and diff check pass. A local wheel digest is intentionally not
  treated as the candidate identity: it varies with build packaging metadata.
  The immutable external source commit and the job-package digest will be the
  reproducible identities after publication.
- [x] (2026-08-11) Confirmed the exact bounded MATH candidate population:
  4,628 valid train rows across Levels 2–4 and seven problem types. The prompt
  length is compact (p95 491 characters; maximum 2,197), and a 4,096-row,
  type-balanced selection is unique and loads without a malformed source row.
  The taskset now excludes malformed rows before ordering; eleven focused
  package tests pass. The production-shaped 4,608-row selection has 36 exact
  128-group candidate pools, all unique. Repeat wheel builds differ in archive
  metadata, so the external source commit plus the packaged job digest—not a
  locally rebuilt wheel hash—will identify the runnable candidate.
- [x] (2026-08-11) Replayed the exact boxed ground truth through the symbolic
  verifier for a deterministic eight-task sample from each of the 21
  Level×type strata (168 tasks). All returned reward 1.0; verifier latency was
  0.70 ms p50, 12.18 ms p95, and 93.80 ms maximum. This validates the
  source-answer-to-verifier path, not model-policy correctness.
- [x] (2026-08-11) Rendered all 4,608 MATH prompts with the pinned Qwen3.5-2B
  chat template and thinking enabled. Prompt-token p50/p95/p99/max are
  133/295/471/1,184; none reaches the 2,048-token input cap. The cap is
  therefore a genuine protection against malformed future inputs, not a
  hidden filter on this selected corpus.
- [x] (2026-08-11) Made declared environment facets part of the generic
  Verifiers training bridge. Selected scalar task fields now flow to rollout
  dataset metadata, live trace attributes, and recovered trace evidence. The
  focused bridge test runs against the real Verifiers types, so the MATH gate
  can report pass@8, truncation, reward spread, and retention by `level` and
  `problem_type` rather than only as pooled run values.
- [x] (2026-08-11) Added those four active-sampling population metrics to the
  maintained TRL candidate and normalized them through Posttrain. With a maximum
  of four candidate rounds, a 32-group update reserves 128 groups; v3 generated
  only 41. The new reserved/generated/retained/unused evidence preserves that
  bounded refill behavior rather than silently changing the iterator. Added an
  opt-in `shuffle_prompts` GRPO selection field, realized consistently by TRL
  and veRL from the recorded loop seed. The focused cross-backend train suite
  passes 130 tests; existing selections remain fixed-order by default.
- [ ] Publish neither candidate yet. First create immutable fork revisions,
  bind a bounded 32x8 qualification over type-balanced MATH Levels 2-4, and
  retain only strata with current-policy pass@8 between 10% and 90% and
  truncation below the promotion guard. Use the eight-family procedural bank
  as a coverage probe, not as the primary initial training source.
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
- Observation: The failed broad RTX PRO attempt was not starved of prompt
  diversity. It sampled 100 unique prompts across all ten requested generators
  over four candidate rounds. The failure was retention quality: only 16
  groups survived truncation masking and zero-gradient filtering.
- Observation: Lightweight Trackio trace pages already retain
  `attributes.is_truncated`; Observatory discarded that field because its
  summary projection inspected only the bounded payload. Full transcript
  hydration is neither necessary nor desirable for population summaries.
- Observation: The repaired Observatory projection was exercised against the
  real 800-row broad run and returned 543 truncated, 720 scored, 80 failed, and
  mean reward 0.37484697 through summary-only pages. This matches the direct
  attribute audit and proves the fix without loading transcript bodies.
- Observation: Policy-relative cold-start quality differs sharply by task.
  After excluding truncated samples, syllogism supplied reward variance in
  8/12 observed groups, letter counting in 4/12, number sorting in 2/9, and leg
  counting in 1/7. Products supplied 0/8; countdown, shortest path, graph color,
  and zebra puzzles truncated at 98.6-100%; knights-and-knaves truncated 87.5%.
- Observation: The first four-task update produced 256 retained traces with
  mean reward 0.6649, but 25% truncated. Reward variance was concentrated in
  syllogism and leg counting; letter counting and number sorting were easier
  and more often unanimous. OLMo 3 active sampling should therefore retain the
  harder variable groups and refill from the same balanced 512-row source.
- Observation: The corrected four-task v1 gate proved that the balanced source
  still could not fill a 32-group update. Across successive candidate rounds it
  retained 15/32, 10/17, 3/7, and 1/4 variable groups; TRL stopped with 232 of
  256 rows retained. Leg counting produced 4 usable groups across the first 14
  complete groups while spending 57 of 112 rollouts at the 4K cap. Syllogism
  produced 12/14 usable groups; letter counting 5/14; number sorting 7/14.
- Observation: TRL's sampled log probabilities are post-processed by
  temperature, top-p, top-k, repetition, and presence controls. They are valid
  for TIS but not for a raw actor/sampler synchronization gate. vLLM prompt
  log probabilities are teacher-forced and bypass sampling processors, making
  them the correct parity evidence.
- Observation: BuildKit v0.31.2 reused files from a prior local named context
  across two package identities. The package manifest and resolved-config
  digest barriers stopped both attempts before provider submission. The durable
  fix gives each package a distinct context name, then aliases it through an
  initial Dockerfile stage.
- Observation: The successful v3 actor update proves the repaired OLMo 3
  mechanics, but not the curriculum. Reasoning Gym's inherited decimal scorer
  awards `len(oracle) / len(response)` when an oracle substring occurs anywhere
  in the answer. That made brevity differences look like correctness variance
  and allowed strings containing both labels to earn reward.
- Observation: Under strict correctness, syllogism is already too easy for the
  SFT policy: 21/32 first-wave groups were unanimously correct and only 6/32
  had a usable mixed outcome. A generator can therefore be a useful held-out
  check while being a poor training source for the same policy.
- Observation: Skywork OR1 uses the same model-relative principle at larger
  scale: its published recipe removes difficulty extremes for each model rather
  than treating all 119,112 rows as one training pool. Its 1.5B difficulty is
  a useful prior for our 2B model, but cannot replace pass@8 measured with our
  exact renderer, sampler and verifier.
- Observation: Active sampling reserves the maximum candidate pool in the
  trainer dataloader before reward outcomes are known. With 32 target prompt
  groups and four bounded rounds, this is 128 task rows per logical update.
  V3 generated 41 groups before filling the update, leaving 87 already
  selected task rows ungenerated. A 4,628-row MATH pool therefore supplies 36
  full candidate pools before a later trainer epoch reshuffles it. This is not
  duplicate generation inside an update, but it is a real data-consumption
  cost that the current metrics hide.

## Decision Log

- Decision: Preserve thinking mode, but use a 4,096-token completion cap for
  the measured cold-start mixture. Keep the 16K failure as immutable evidence
  rather than making it the initial training shape.
  Rationale: the goal is reasoning training, but cold-start tasks should fit a
  bounded cap that makes pathological repetition and truncation observable
  without spending the whole update on dead completions.
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
- Decision: Block release promotion until the local 32x8 cold-start gate
  reaches a finite actor update with scorable, nondegenerate evidence.
  Rationale: local exact-package qualification is faster than publishing
  speculative versions and separates code readiness from distribution.
  Date/Author: 2026-08-11 / Codex.
- Decision: Use syllogism, letter counting, number sorting, and leg counting as
  the initial cold-start mixture. Preserve the broad ten-generator selection
  as immutable evidence and reintroduce tasks only after their scorer and
  policy-relative pass/truncation gates are met.
  Rationale: These four generators produced two-sided signal among untruncated
  completions under the current SFT policy. The choice is based on measured
  retention, not a claim that excluded tasks are intrinsically unsuitable for
  reasoning RL.
  Date/Author: 2026-08-11 / Codex.
- Decision: Supersede the initial four-task v1 mixture with an additive v2
  cold-start selection over syllogism, letter counting, and number sorting;
  expand the deterministic pool from 512 to 6,144 rows and defer leg counting.
  Rationale: Active sampling cannot manufacture signal from masked or
  unanimous groups. The live v1 population showed that leg counting consumed
  disproportionate completion capacity and still left the update short, while
  the retained three tasks jointly preserve logical, string, and numerical
  reasoning with materially better usable-group yield.
  Date/Author: 2026-08-11 / Codex.
- Decision: Keep summary reads lightweight. Provider-neutral trace attributes
  may supply bounded summary facts such as truncation, while transcript content
  remains a detail-only fetch.
  Rationale: This fixes correctness without restoring the old 11 MB/5,000-row
  overfetch path.
  Date/Author: 2026-08-11 / Codex.
- Decision: Separate raw policy parity from behavior-policy TIS evidence in
  TRL. Enforce actor/vLLM synchronization only on a bounded teacher-forced raw
  probe; retain processed sampling deltas for TIS and diagnostics.
  Rationale: sampling controls intentionally change behavior log probabilities,
  so comparing them to raw actor logits creates a false failure exactly when
  the selected policy is realized correctly.
  Date/Author: 2026-08-11 / Codex.
- Decision: Name the BuildKit package context with `PACKAGE_KEY` and reference
  it through a build-argument-selected initial stage.
  Rationale: no global cache prune or builder restart should be part of normal
  developer workflow, and digest checks should detect corruption rather than
  serve as the only way to escape it.
  Date/Author: 2026-08-11 / Codex.
- Decision: Supersede substring/length-sensitive Reasoning Gym reward for the
  next online-RL qualification with `boxed_exact`; preserve `native` as an
  explicit compatibility mode.
  Rationale: online RL must optimize verified task correctness. Answer parsing
  and formatting are observable gates, but response length must not silently
  become a reward component through substring matching.
  Date/Author: 2026-08-11 / Codex.
- Decision: Use type-balanced MATH Levels 2-4 as the primary initial cold-start
  candidate, then filter by this exact policy's measured pass@8. Probe the
  procedural families `basic_arithmetic`, `chain_sum`,
  `fraction_simplification`, `base_conversion`, `prime_factorization`,
  `simple_equations`, `propositional_logic`, and `calendar_arithmetic` for
  coverage; do not assume all eight belong in training.
  Rationale: MATH already provides substantive reasoning problems and symbolic
  boxed verification. The procedural bank supplies deterministic capability
  coverage, while policy-relative retention prevents easy unanimous and
  impossible/truncated groups from displacing learning signal. Skywork OR1 is
  the next data expansion after an immutable adapter and model-specific
  difficulty measurement, not an unfiltered initial dependency.
  Date/Author: 2026-08-11 / Codex.
- Decision: Preserve OLMo 3's bounded active-sampling behavior for the first
  MATH qualification, but make candidate-pool consumption a first-class metric
  before deciding whether a lazy refill buffer is warranted.
  Rationale: a lazy buffer can improve data efficiency, but changes dataloader
  and checkpoint-resume semantics. The immediate, low-risk fix is accurate
  evidence: expose all reserved, generated, retained, and unused prompt-group
  counts; use that evidence plus MATH pass@8 to size the longer campaign.
  Date/Author: 2026-08-11 / Codex.
- Decision: The MATH campaign will opt into seeded prompt shuffling and use a
  4,608-row type-balanced Level 2–4 population (36 exact 128-group candidate
  pools), not the arbitrary 4,096-row round number.
  Rationale: 4,608 admits the whole-candidate-batch sampler without a partial
  final chunk and uses all but 20 valid selected source rows. Shuffling changes
  only task order across later data epochs; task identities, source revision,
  filters, row digests, and run seed remain retained. The rebuilt local wheel
  is a test artifact, not an identity to copy into a published binding.
  Date/Author: 2026-08-11 / Codex.

## Outcomes & Retrospective

Not complete. The accepted outcome must name the exact Posttrain and, if
changed, TRL commits; the exact Ambient package key; the local provider and
Trackio run identities; the eight rollout termination/reward summaries; actor
update evidence; and the final release/pin only after both local and released
canaries pass.

## Proposed MATH cold-start binding

The next candidate is deliberately a one-update qualification, not a
100-update production run. It starts from
`models/qwen3.5-2b-sft-10k-json@lora-v0`; the earlier syllogism update is
mechanics evidence only because its inherited reward was invalid.

```yaml
environment:
  id: math-python-train-coldstart-l2-l4-g8-concurrent256
  source: math-python-v1@<published-candidate-commit>
  taskset:
    id: math-python-v1
    split: train
    num_tasks: 4608       # must equal the binding population below
    order_seed: 20260811
    balance_by_type: true
    levels: [Level 2, Level 3, Level 4]
  num_tasks: 4608
  num_rollouts: 8
  max_concurrent: 256
  sampling: {max_tokens: 4096, temperature: 1.0, top_p: 1.0}
  observation:
    primary_metric: math_reward
    pass_rate_metric: symbolic_correctness
    facets:
      - {field: level, dimension: level, label: MATH level}
      - {field: problem_type, dimension: problem_type, label: Problem type}

settings:
  algorithm: olmo3
  loop: {max_steps: 1, max_length: 6144, per_device_batch_size: 2,
         gradient_accumulation_steps: 128, learning_rate: 1.0e-5,
         lr_scheduler_type: constant, checkpoint_steps: 1, seed: 42}
  num_prompts_per_step: 32
  num_generations: 8
  max_prompt_length: 2048
  max_completion_length: 4096
  shuffle_prompts: true
  beta: 0.0
  advantage_scaling: none
  clip_epsilon_low: 0.2
  clip_epsilon_high: 0.272
  importance_sampling_mode: token_truncate
  importance_sampling_clip_min: null
  importance_sampling_clip_max: 2.0
  active_sampling: {max_candidate_batches: 4}
  mask_truncated_completions: true
```

The package must bind the taskset and outer population to the same 4,608 rows;
otherwise the bridge's deterministic finite-task selection would resample a
different population and erase the type-balanced order. The qualification
report must contain the standard OLMo 3 measurements plus candidate groups
reserved/generated/retained/unused and a level×type pass@8, reward-spread, and
truncation table. Promotion to 100 steps requires at least 10% and at most 90%
pass@8 in retained strata, two-sided reward groups, acceptable truncation, and
the observed candidate-population consumption rather than a presumed reuse
rate.

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
arguments, and dependency/source digests. The cold-start gate uses 32 prompt
groups, eight generations, a 4,096-token completion budget, OLMo 3 active
sampling, mean-only advantages, asymmetric clipping, token TIS, truncation
masking, and LoRA updates. Environment and vLLM concurrency are 256 on the
RTX PRO qualification target.

Finally, run one logical update. A good run needs 32 retained prompt groups
after active-sampling refill; bounded generation that either stops or cleanly
reaches a declared limit; scorable groups with reward spread; centered positive
and negative advantages; finite entropy and gradients; raw actor/sampler
parity plus independent TIS and clip telemetry; one completed actor update; and
adapter-only model/recovery artifacts. Any failure is diagnosed from retained
traces before changing another knob.

Before that update, add a versioned cold-start environment that round-robins
syllogism, letter counting, number sorting, and leg counting. Keep G=8 and the
32-group update, use the existing 4,096-token cap, and realize the complete
sampling policy (`temperature=1`, `top_p=0.95`, `top_k=20`, and a mild
anti-repetition control) identically in the environment and vLLM binding. Do
not mutate the failed broad selection or silently turn off thinking.

## Concrete Steps

Work in `/home/hammad/projects/rl` unless a command names Ambient Agent.

1. Add focused unit tests for the full sampling value, catalog-to-bridge
   translation, Verifiers round trip, TRL drift detection/arguments, and veRL
   request parameters. Make those tests fail before implementation.
2. Implement the smallest backend-neutral contract and adapter changes. If
   presence penalty requires a maintained TRL change rather than its supported
   `generation_kwargs`, make that change in `/home/hammad/projects/trl-1.9-upgrade`, test
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
6. Submit the exact locally built package to the RTX PRO qualification target
   and wait through the first logical update. Reconcile provider and Trackio
   evidence before calling it terminal.
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
errors and truncations are retained; retained eight-generation prompt groups
have nonzero reward variance; advantage mean is approximately zero with both
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
Release promotion was explicitly moved after a successful complete cold-start
update.
