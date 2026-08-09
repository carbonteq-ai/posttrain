# Publish the RL framework and qualify SFT-backed OLMo 3 GRPO

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` current after every milestone.
The repository has no `.agents/PLAN.md`; this file follows the contract in
`docs/templates/PLAN.md` and the frozen baseline in `docs/post-training/README.md`
and documents the missing repository-local plan instructions explicitly.

## Purpose / Big Picture

This work answers two related but separate questions in a deliberate order.
First, it turns the reviewed framework, tracking and Observatory changes into a
reproducible release that an independent project can install and use without
this checkout. After that exact stable release is installed into Ambient Agent,
it establishes whether the named OLMo 3 GRPO recipe can learn from the existing
SFT LoRA adapter under the high-concurrency RTX PRO target, using a
small run that is cheap to retry and rich enough to detect a broken learning
signal.

The current 256-concurrency run is not the final algorithm experiment: its
selected project training settings enable DAPO and truncation handling but do
not explicitly set `advantage_scaling: none`, so it still uses the compatibility
default group scaling. It must not be presented as evidence for the corrected
DAPO recipe. The release is also not allowed to start from the current dirty
checkout: release checks must pass on one intentional commit, then the
candidate and final protected workflows must build, qualify, and publish exact
bytes.

This work does not change the frozen post-training product meaning. It makes
the existing artifact, observation, execution and read-product responsibilities
enforceable at release time and changes implementation only after the proposed
ADR is accepted.

## Progress

- [x] (2026-08-09) Read the canonical post-training baseline and plan template;
  confirmed the product boundary is frozen and `train.grpo` selects DAPO through
  `GRPOSettings.algorithm`.
- [x] (2026-08-09) Audited the active experiment inputs. The SFT adapter,
  environment revision, framework source, image digest, and remote run IDs are
  recorded below; the active settings omit `advantage_scaling: none`.
- [x] (2026-08-09) Audited the release state. `release/manifest.toml` remains
  `0.3.2`, tag `v0.3.2` exists, `main` is at `116b1fd1`, and the working tree
  contains 69 dirty paths.
- [x] (2026-08-09) Ran `uv run --no-sync posttrain-release check`; it fails on a
  stale `packages/catalog/src/posttrain/catalog/base/locks.toml` digest.
- [x] (2026-08-09) Added a dedicated project settings revision with `algorithm: olmo3`, while
  retaining scalar verifier rewards, asymmetric clipping, truncation masking,
  zero-gradient filtering with active refill, and the materialized SFT LoRA
  input.
- [x] (2026-08-09) Classified the prior 256-concurrency DAPO probe as rejected
  historical evidence because it did not select the intended advantage
  semantics; it will not be reused as OLMo 3 qualification.
- [x] (2026-08-09) Ran the corrected one-step OLMo 3 probe and verified actor/sampler parity,
  reward spread, advantage statistics, truncation exclusion, optimizer update,
  checkpoint/LoRA artifact output, and tracking finalization. The accepted r5
  gate used a 2,048-token completion cap and published the retained LoRA adapter
  and recovery checkpoint.
- [x] (2026-08-09) Inventoried the dirty framework tree and created one intentional release
  commit containing only reviewed framework changes and regenerated release
  inputs; leave unrelated edits untouched.
- [x] (2026-08-09) Chose release `0.3.3`, updated `release/manifest.toml`,
  regenerate dependency locks, and pass the complete local quality ladder.
- [x] (2026-08-09) Prepared the `0.3.3` release branch and draft PR, passed
  the local quality ladder and exact-SHA quality CI, and resolved the Trackio
  post10 wheelhouse asset required by the protected workflow.
- [x] (2026-08-09) Published Trackio
  `carbonteq-v0.31.5.post11` from exact source `7a2b885b...`, including the
  missing-completed-upload recovery fix and complete package/frontend tests.
- [x] (2026-08-09) Stopped release retries after candidate `31281418857`
  reached the real dstack canary and exposed a deployed Trackio artifact
  compatibility failure. Recorded the cross-system cause and proposed
  [ADR 0014](../decisions/0014-attested-release-promotion-graph.md).
- [x] (2026-08-09) Manually published and deployed Trackio
  `0.31.5.post11`, then qualified scalar history and a cache-independent S3
  artifact upload/download against the live service. Producer run
  `50f9a44eda0e4588b8b5ac4d88c8d8c8` and consumer run
  `829d2776062a4825bbf3c228dfceac89` read back artifact SHA-256
  `bf6e71abcbd1631441dbc2b45610f9d4c59648c6dc1039dc0106d930d7fc143c`.
- [x] (2026-08-09) Manually published TRL `1.9.2.post1` from source
  `a82ecebc0fa081efd58302a34a553445fc73271d`, retained wheel and sdist
  hashes, and changed Posttrain to consume the stable package instead of a Git
  checkout.
- [x] (2026-08-09) Added the two-phase runtime-lock materialization boundary.
  Pull-request CI validates authored dependency receipts while keeping the last
  published OCI manifest strict; the protected candidate materializes the
  internal wheel URLs, rebuilds affected images, then applies strict validation
  and retains both generated lock and manifest.
- [x] (2026-08-09) Passed the complete source ladder: release metadata and
  repository audits, Ruff lint/format, Pyright, eight import contracts, 1,081
  tests with 21 expected skips, and `git diff --check`. Installed the exact
  published TRL `1.9.2.post1` extra and passed the focused TRL compatibility
  suite (8 passed, 1 environment-dependent skip).
- [x] (2026-08-09) Repaired the public clean-consumer bridge exposed by
  exact-SHA CI `31289633742`: GitHub CI now downloads and SHA-256 verifies both
  maintained Trackio and TRL release wheels before constructing an offline
  starter-project wheelhouse. The previously failing SFT starter acceptance
  passes locally against those exact two assets.
- [x] (2026-08-09) Diagnosed candidate `31289887803` at the registry boundary:
  the veRL image completed and three sibling images published, but the registry
  invalidated its resumable upload with `invalid content range`. Added one
  classified BuildKit retry that preserves content identity and reuses completed
  layers; regression tests prove transient recovery and a bounded persistent
  failure.
- [x] (2026-08-09) Proved candidate retry `31290415458` was not another
  transport-only failure: the bounded retry exposed `no space left on device`
  in the private registry. Purged 27 terminal GRPO smoke runs with at most two
  reward-bearing logical steps across all Trackio projects, retaining exact
  plans and receipts and excluding an indeterminate nonterminal run.
- [x] (2026-08-09) Reclaimed the registry safely. Deleted 86 exact superseded
  runtime manifest digests after proving no overlap with 13 retained stable and
  candidate digests; native garbage collection removed 367 unreachable blobs
  totaling 74.49 GiB. Registry storage fell from 154 GiB to 80 GiB, root free
  space rose from 1.9 GiB to 77 GiB, the service restarted healthy, and all 13
  retained digests passed readback. Machine-local plans, receipts, and GC logs
  are under
  `/home/hammad/.local/state/posttrain/operations/grpo-short-run-purge-20260809/`.
- [x] (2026-08-09) Candidate `31291348503` proved the repaired registry,
  generated all seven `0.3.3` runtime images, published `0.3.3rc5`, and passed a
  clean index-only consumer install. Its dstack gate then received malformed
  `--json` output and passed it directly to `jq`, even though the RTX PRO target
  was idle and the same check passed immediately afterward. Added bounded
  capacity-read retries, explicit JSON validation, sanitized failure receipts,
  and regression coverage for malformed, persistently invalid, and valid-but-
  busy responses.
- [x] (2026-08-09) Re-ran the protected release candidate using the accepted manual Trackio
  and TRL receipts. Require dev-index/OCI readback, clean install, packed dstack
  job, Trackio artifact round trip, and Observatory readback before merge.
- [x] (2026-08-09) Merged release PR #34 and dispatched the final workflow for exact merged
  SHA and accepted candidate materialization. Create no tag before stable
  readback succeeds.
- [x] (2026-08-09) Updated the local Ambient Agent project to exact stable Posttrain
  release, replace its direct legacy TRL/Trackio pins with the release-resolved
  dependency graph, and revalidate its SFT-backed online-RL work packages.
- [x] (2026-08-09) Added a new versioned Ambient Agent `algorithm: olmo3` setting and work
  package after that dependency update. Preserve the historical DAPO settings
  and runs; do not relabel them as OLMo 3 evidence.
- [x] (2026-08-09) Packed the resolved OLMo 3 job and ran one SFT-LoRA-backed optimizer step on
  RTX PRO 6000 with rollout and verifier concurrency 256. Qualify reward spread,
  recomputed advantages, truncation exclusion, clipping/TIS, actor-sampler
  parity, LoRA-only checkpoint artifacts, and restart metadata.
- [x] (2026-08-09) Diagnosed the first released OLMo 3 gate: the SFT LoRA
  rollout binding omitted Qwen 3.5's `language_model.` namespace, causing a
  systematic actor/sampler log-probability delta near `0.25`. Corrected all 19
  project LoRA rollout bindings; the second gate passed parity and entered
  active sampling.
- [x] (2026-08-09) Diagnosed the second gate before its optimizer update:
  Posttrain opened actor-update telemetry after every candidate rollout, so an
  OLMo 3 refill looked like an overlapping update. Moved phase start to the
  trainer's post-preparation boundary, where all candidate/refill rollouts have
  completed and the retained batch is about to enter forward/backward. Focused
  and broader TRL tests pass.
- [x] (2026-08-09) Published Posttrain `0.3.4` with the actor-update boundary repair, updated
  Ambient Agent to that exact stable graph, and repeat the one-step gate.
- [x] (2026-08-09) After the r5 gate passed, submitted 200 optimizer steps and monitored the
  first five completed steps before leaving the campaign unattended. The
  campaign ran on the RTX PRO 6000 as
  `ambient-k1-olmo3-sft10k-rtxpro-c256-l2048-200step-20260809-r1`. A later
  health check through step twelve remained finite and exploratory. Step ten
  had reward `0.6913`, advantage standard deviation `0.1458`, entropy
  `0.2464`, TIS mean `0.9996` with no clamping, `1.5625%` zero-variance
  candidate groups, and controlled upper clipping. Step twelve showed an
  isolated `13.28%` candidate-batch truncation spike, but the authoritative
  run population remained 39 truncated of 1,788 completed rollouts (`2.18%`),
  below the five-percent campaign guard. Truncated rows remained zero-loss,
  TIS remained unclamped, and usable advantage spread remained present. Both
  asymmetric clipping sides were exercised across the first twelve steps.
  Training remained healthy through the full-LR transition and step 25. The
  exact mounted `checkpoint-25` was then validated in the packaged training
  environment: trainer global step 25, 372 rank-8 LoRA tensors matching 372
  optimizer states, scheduler state, CPU/CUDA/NumPy/Python RNG state, and no
  full-model weight file. The checkpoint adapter is not a copy of the starter:
  all 372 corresponding tensors changed across 8,409,600 compared elements,
  with aggregate L2 delta `0.09549` and maximum absolute element delta
  `0.0001261`; its SHA-256 differs from the materialized SFT adapter. This is
  direct weight evidence that the actor updates are being applied, independent
  of the metric trend. The run then completed step 26 after that checkpoint
  with reward `0.8114`, advantage standard deviation `0.0998`, positive and
  negative advantage fractions `0.4654` and `0.3664`, entropy `0.2170`, zero
  truncation, TIS mean `0.9998` with `0.000004814` clamping, and both
  asymmetric clipping directions exercised. This proves the checkpoint write
  did not stall or destabilize the next update.
  Across all 26 completed updates, five-to-six-step windows show mean reward
  increasing from `0.6011` at steps 1--5 to `0.6410`, `0.7256`, `0.7612`, and
  finally `0.7815` at steps 21--26. Over the same windows advantage standard
  deviation narrowed from `0.3084` to `0.1205`, entropy declined gradually
  from `0.2592` to `0.2189` without collapse, and truncation fell from `1.72%`
  to `0.43%`. TIS clamping remained effectively zero and asymmetric clipping
  stayed small but active. This is evidence that the online updates are
  learning while retaining usable exploration; it is not a final campaign
  qualification until the run finishes and held-out evaluation is compared.
  Steps 27 and 28 then completed with rewards `0.8420` and `0.7856`, advantage
  standard deviations `0.0863` and `0.1378`, entropy `0.2076` and `0.2231`,
  zero truncation, unclamped TIS, and both clipping directions active. The
  post-checkpoint full-learning-rate continuation therefore remains healthy
  through step 28. Step 29 remained finite at reward `0.7968`, advantage
  standard deviation `0.0879`, entropy `0.1981`, truncation `1.45%`, unclamped
  TIS, and two-sided clipping. Entropy is now about 24% below the opening
  five-step mean; this is not a collapse, but entropy and zero-variance group
  fraction are the primary watch metrics for the next monitoring window.
  The 29-step invariant audit found maximum absolute advantage mean
  `1.16e-8`; all 29 updates contained both positive and negative advantages;
  mean truncation was `1.44%`; TIS mean stayed within `0.0004` of one and only
  steps 21 and 26 clamped approximately `0.00048%` of tokens. Lower clipping
  activated on 23 steps, upper clipping on 24, and both sides on 20. The last
  five-step reward mean is `0.8029` versus `0.6011` initially. Zero-variance
  groups increased from `9.0%` to `18.8%`, but active refill retained at least
  `76.2%` of candidates and every update kept usable two-sided advantage
  signal. Continue watching that curriculum saturation signal rather than
  treating the higher reward alone as proof of final generalization.
  Steps 30--32 then remained healthy at full learning rate. Their rewards were
  `0.8033`, `0.7123`, and `0.8336`; advantage standard deviations were
  `0.1283`, `0.0736`, and `0.0971`; and entropies were `0.2135`, `0.2057`,
  and `0.2033`. Every advantage mean remained within `1e-8` of zero and every
  update retained both positive and negative advantages. TIS means remained
  within `0.0003` of one with zero clamping, both asymmetric clipping
  directions remained active, and active sampling retained `76.2%`--`82.1%`
  of generated candidates. Only step 32 truncated completions in this window,
  at `1.04%`; those completions remained excluded from the advantage and loss.
  The run is therefore healthy through 32 completed optimizer updates. This is
  an interim online-training verdict, not the final held-out qualification.
  Step 33 then persisted to Trackio with reward `0.8009`, centered advantage
  mean `-4.24e-10`, advantage standard deviation `0.1220`, entropy `0.2273`,
  zero truncation, TIS mean `0.9998` with no clamping, and both clipping
  directions active. Its zero-advantage row fraction rose to `39.8%`, so active
  sampling needed five generation rounds and retained `68.1%` of 188 generated
  candidates. Both positive and negative advantages remained present. This is
  not an algorithm failure, but it is the strongest curriculum-saturation
  warning so far and makes retained fraction plus zero-advantage share the
  primary monitoring pair through the next checkpoint.
  Step 34 showed that the warning was not a monotonic collapse: reward was
  `0.8104`, advantage mean `1.35e-9`, advantage standard deviation `0.0871`,
  entropy `0.2191`, truncation `0.20%`, and step time `323.7` seconds. TIS
  remained unclamped and both clipping directions remained active. The
  zero-advantage row fraction fell to `23.9%`; active sampling used four rounds
  and retained `69.6%` of 184 generated candidates. Retention remains below the
  earlier run average, so continue watching the pair, but the update preserved
  centered, two-sided learning signal and normal runtime behavior.
  Step 35 further recovered active-sampling retention to `82.1%` of 156
  generated candidates while completing in `247.6` seconds. Reward was
  `0.8798`, advantage mean `3.49e-9`, advantage standard deviation `0.1270`,
  zero-advantage rows `28.1%`, entropy `0.1969`, and truncation zero. TIS
  remained unclamped and both clip directions were active. Comparing the first
  five updates with steps 31--35, mean reward rose from `0.6011` to `0.8074`,
  advantage standard deviation narrowed from `0.3084` to `0.1014`, entropy
  declined from `0.2592` to `0.2105`, truncation fell from `1.72%` to `0.25%`,
  and active-sampling retention moved from `93.8%` to `75.6%`. This is a
  coherent learning signature with reduced but still usable relative signal;
  the rising zero-advantage share (`9.2%` to `24.2%`) and lower retention are
  curriculum-saturation guardrails, not yet stopping conditions.
  Step 36 remained stable in `290.8` seconds: reward `0.8664`, centered
  advantage standard deviation `0.1288`, entropy `0.1992`, zero truncation,
  TIS mean `0.9998` without clamping, and both clip directions active. Active
  sampling retained `72.7%` of 176 generated candidates across four rounds;
  zero-advantage rows were `26.2%`. The paired saturation indicators therefore
  oscillate within a usable range rather than worsening monotonically.
  Step 37 completed in `289.0` seconds with reward `0.8419`, centered
  advantage standard deviation `0.1055`, entropy `0.2179`, `0.52%`
  truncation, TIS mean `1.00004` without clamping, and both clip directions
  active. Zero-advantage rows fell to `21.0%`; active sampling needed three
  rounds and retained `74.4%` of 172 generated candidates. This second
  consecutive recovery in zero-advantage share confirms that the step-33 spike
  was prompt-batch variation rather than a monotonic loss of learning signal.
  Step 38 completed in `373.1` seconds with reward `0.7758`, centered
  advantage standard deviation `0.1444`, `0.26%` truncation, TIS mean
  `0.99993` without clamping, and both clip directions active. Zero-advantage
  rows fell to `11.9%` and active sampling retained `82.1%` of 156 candidates,
  so relative learning signal strengthened even though reward varied downward.
  Entropy reached a new single-step low of `0.1924`; the steps 34--38 mean is
  still `0.2051`, versus `0.2592` initially, so this is not entropy collapse.
  Entropy is nevertheless the sharper guardrail now that reward and advantage
  spread remain healthy.
  Step 39 was the strongest curriculum-saturation event so far. It completed
  in `407.7` seconds with reward `0.8752`, centered advantage standard
  deviation `0.0528`, entropy `0.1945`, zero truncation, TIS mean `0.99992`
  without clamping, and both clip directions active. Zero-advantage rows rose
  to `59.9%`; active sampling needed eight rounds and retained only `60.4%` of
  212 generated candidates. Positive and negative advantages still remained
  (`23.6%` and `16.5%`), so this is not a zero-gradient update or an algorithm
  failure. Step 40 is now a confirmation gate: a similar result would indicate
  sustained curriculum saturation, while recovery would classify step 39 as
  another prompt-batch spike.
  Step 40 confirmed an elevated saturation regime but not collapse. It
  completed in `331.8` seconds with reward `0.8338`, centered advantage
  standard deviation `0.1131`, entropy `0.2086`, zero truncation, and TIS mean
  `0.9998` without clamping. Both advantage signs remained; only lower clipping
  activated on this batch, while the surrounding window exercised both sides.
  Zero-advantage rows recovered to `38.9%`, and active sampling needed five
  rounds with `68.1%` retention of 188 generated candidates. Comparing steps
  31--35 with 36--40, mean reward rose from `0.8074` to `0.8386` and advantage
  standard deviation held at `0.1014` versus `0.1089`, but zero-advantage rows
  rose from `24.2%` to `31.6%`, retention fell from `75.6%` to `71.5%`, rounds
  rose from `3.8` to `4.6`, entropy moved from `0.2105` to `0.2025`, and mean
  step time increased from `297.7` to `338.5` seconds. OLMo 3 active sampling
  is still preserving a full, two-sided update batch; it is paying a growing
  rollout-cost premium as the current prompt distribution becomes easier.
  Step 41 recovered toward the healthier end of that regime in `322.6`
  seconds: reward `0.8239`, centered advantage standard deviation `0.0963`,
  entropy `0.2006`, `0.20%` truncation, TIS mean `0.99996` without clamping,
  and both clip directions active. Zero-advantage rows fell to `25.6%`, and
  active sampling retained `78.0%` of 164 candidates across four rounds. The
  rollout premium is therefore variable rather than steadily worsening.
  Steps 42--45 preserved the same usable but increasingly saturated regime.
  Step 42 set a single-step entropy low of `0.1860` with `40.8%`
  zero-advantage rows, but step 43 immediately recovered to entropy `0.2040`,
  advantage standard deviation `0.2012`, and `17.4%` zero-advantage rows.
  Step 44 reproduced the step-39 high-saturation shape (`59.1%` zero rows,
  eight rounds, `60.4%` retention, advantage standard deviation `0.0456`),
  then step 45 recovered to `26.0%` zero rows, four rounds, `72.7%` retention,
  and advantage standard deviation `0.1112`. Across steps 41--45, mean reward
  was `0.8706`, entropy `0.1957`, advantage standard deviation `0.1077`,
  zero-row share `33.8%`, retention `71.0%`, rounds `4.8`, truncation `0.09%`,
  and step time `312.7` seconds. OLMo 3 continues to preserve informative
  two-sided batches, but the campaign is now clearly in a higher-reward,
  lower-entropy, more rollout-expensive phase.
  Step 46 extended that regime rather than invalidating the update: reward was
  `0.8985`, centered advantage standard deviation `0.0751`, entropy `0.1693`,
  zero-advantage rows `43.8%`, zero truncation, and TIS mean `0.99991` without
  clamping. Active sampling retained `54.2%` of 236 candidates across six
  rounds; both advantage signs and both asymmetric clipping directions were
  present. Across all 46 updates, maximum absolute advantage mean remains
  `1.17e-8`, every retained update is two-sided, maximum TIS-mean displacement
  from one is `0.00037`, and the maximum clamped TIS fraction is
  `0.000004814`. The live evidence therefore still shows correct advantage,
  TIS, and clipping mechanics. It also makes the late-run curriculum pressure
  explicit: the next four steps and the step-50 recovery checkpoint are the
  next qualification boundary rather than treating one low-entropy batch as
  either a collapse or a success.
- [x] (2026-08-09) At the user's requested stopping point, gracefully cancelled
  the production campaign after optimizer step 51 instead of aborting its
  provider process. The latest complete checkpoint is `checkpoint-50`: its
  `trainer_state.json` reports step 50; it contains 372 LoRA tensors, 372
  optimizer states, scheduler state, and CPU/CUDA/NumPy/Python RNG state; and
  it contains no full-model weight file. Trackio retained it as the run's
  `recovery` artifact (121,449,914 bytes, content-tree SHA-256
  `0da9494ddc4cd886d571e69b60f662418e3ae660997e159b3e007cca9cce8839`).
  An independently copied controller fallback under
  `/home/hammad/.local/state/posttrain/recovery/ambient-k1-olmo3-sft10k-rtxpro-c256-l2048-200step-20260809-r1/checkpoint-50`
  has the same byte count and tree digest. Provider state is terminal
  `cancelled`; reconciliation reports provider and Trackio evidence consistent,
  tracking status `cancelled`, and admission `completed`. A future invocation
  can use `posttrain job run ... --resume-from-run <source-run> --run-id
  <new-run>` to materialize the retained state under a new run identity.
- [x] (2026-08-09) Found and locally corrected a recovery-lineage metadata bug:
  interrupted publication previously labeled the latest complete checkpoint
  with the live trainer step, so this step-50 payload was advertised as step
  51. The generic TRL adapter now derives the published step from the retained
  checkpoint's own `trainer_state.json` and rejects malformed state. Focused
  recovery tests, Ruff, Pyright, and diff validation pass. This correction is
  not part of the already released `0.3.4`; the actual retained checkpoint
  remains safely resumable at its authoritative step 50.
- [x] (2026-08-09) Purged the failed r3 smoke run. Added shared-image ownership
  protection after its first preview showed that the released planner could
  delete a manifest still referenced by r4. The r1, r2, and superseded r4
  cleanups are durably deferred on the occupied exact worker; registry,
  Trackio, and local deletion remain blocked until workspace cleanup is
  verified. A transient five-minute user timer reapplies only those three exact
  immutable plans and stops itself after all receipts exist. Its first
  automatic reconciliation completed successfully at 11:11 PKT, reused the
  same three provider tasks without duplicates, and kept all downstream
  actions deferred. A subsequent code review found that project-local image
  ownership was insufficient for a registry-global manifest deletion. Purge
  previews now combine the opened and registered project submission stores
  with current and archived machine-admission image references, retire owners
  only through complete unblocked purge receipts, and block registry deletion
  if any registered ownership source is unreadable. Live inspection proves
  the three queued smoke digests each have exactly one remaining owner; the
  already-purged r3 is correctly retired from r4's shared digest. The final
  planner also revalidates the complete machine ownership snapshot immediately
  before apply, so a newer run that acquires the same digest after preview
  blocks deletion instead of becoming a time-of-check/time-of-use victim. A
  live apply of the existing r4 plan passed this ownership gate and then
  returned the expected provider `deferred` result without advancing registry
  or Trackio mutation. New purge plans, journals, and receipts now use the
  machine-scoped admission state root, matching their cross-project ownership;
  `show`, `apply`, completed-plane recovery, and run-list filtering retain
  read compatibility with existing project-local plans. The live r4 legacy
  plan resolves through that compatibility path and the five-minute timer
  continues to defer it safely. The final local cleanup implementation passes
  200 focused execution, dstack, and CLI tests plus the complete repository
  suite of 1,109 tests with 18 expected skips. Ruff, full Pyright, all eight
  import contracts, and `git diff --check` also pass.
  The transient reconciler now checks the production provider state before
  reapplying any plan. While production is nonterminal it writes one bounded
  status line and exits; the three provider-native no-capacity cleanup tasks
  continue retrying independently. Live dstack inspection confirms each
  current task is `pending` with status message `retrying`, exact
  `retry.on_events: [no-capacity]`, a 24-hour retry duration, and a bounded
  five-minute cleanup execution once placed. The production forecast leaves
  several hours of margin before those durable retries expire. A manual cycle
  proved this path added no
  purge journal or provider mutation; the scheduled 13:18 PKT cycle then
  produced the same single nonterminal status line automatically. This
  eliminates redundant five-minute apply traffic without delaying cleanup once
  the worker is released.
- [ ] Record final experiment and release receipts, update this plan’s outcome,
  and report the tag, release URL, package/image digests, and remaining gates.

## Surprises & Discoveries

- Observation: OLMo 3 candidate/refill metrics and retained actor-batch metrics
  have different populations when active sampling needs more than one
  generation round.
  Evidence: steps one through four used multiple generation rounds, so native
  reward and advantage summaries aggregate candidate rounds. Step five used
  exactly one round, generated and retained 128/128 rows, and is therefore an
  exact retained-batch audit: reward `0.6723 ± 0.3878`, advantage mean near
  zero with standard deviation `0.3023`, positive/negative fractions
  `0.6614/0.3386`, zero-advantage fraction `0`, and truncation `0.7812%`.
  Clipping is computed during the retained actor loss and is not affected by
  this candidate-population distinction.
  Date/Author: 2026-08-09 / Codex.

- Observation: the first fourteen production steps show an encouraging but
  non-qualifying early reward movement without an entropy collapse.
  Evidence: the unadjusted mean reward increased from approximately `0.625`
  over steps 1-7 to `0.674` over steps 8-14, while mean entropy moved only from
  approximately `0.258` to `0.251`. Step 14 retained finite gradients and
  usable advantage spread, TIS mean `1.0` without clamping, both asymmetric
  clip sides, and `0.7812%` truncation. These batches mix prompts and, when
  refill occurs, candidate populations, so this is a health observation rather
  than evidence of final policy improvement.
  Date/Author: 2026-08-09 / Codex.

- Observation: live Observatory omits reward standard deviation and
  zero-variance groups while a TRL run is active because the adapter suppresses
  those trainer aggregates in favor of a bridge replay that only occurs during
  finalization.
  Evidence: provider logs contain `reward_std` and `frac_reward_zero_std` for
  each step, while the live run view reports both canonical series missing.
  The eventual correction must distinguish candidate rollout population from
  the retained actor-update population rather than simply duplicating the same
  metric name.
  Date/Author: 2026-08-09 / Codex.

- Observation: an exact-worker cleanup can be valid but temporarily
  unschedulable when a production run occupies the same single-slot dstack
  worker.
  Evidence: the initial r4 cleanup failed placement with `no offers`. The
  corrected provider path now retains a 24-hour dstack no-capacity retry,
  journals the purge action as `deferred`, and prevents registry, tracking, or
  local deletion until the exact workspace reports verified cleanup.
  Date/Author: 2026-08-09 / Codex.

- Observation: a normal TRL checkpoint is retained in the run workspace at its
  configured save step but is published to Trackio only during successful
  finalization or interruption handling.
  Evidence: the production run exposes its exact host volume inside the task at
  `/opt/posttrain/run/ambient-k1-olmo3-sft10k-rtxpro-c256-l2048-200step-20260809-r1`.
  A read-only dstack attach before step 25 found only the immutable
  `scratch/.../inputs/model_adapter/adapter_model.safetensors` and no
  `checkpoint-*` directory, as expected. The monitoring heartbeat now checks
  this mounted workspace at or after step 25 and distinguishes the input
  adapter from adapter-only recovery state.
  Date/Author: 2026-08-09 / Codex.

- Observation: Active sampling may invoke the rollout bridge several times for
  one optimizer step; a rollout return is therefore not an actor-update
  boundary.
  Evidence: the corrected LoRA canary passed parity and failed when its second
  candidate batch tried to reopen step-one actor telemetry. TRL's
  `_prepare_inputs` owns generation, reward scoring, filtering and all refills,
  and returns only after selecting the retained batch.
  Date/Author: 2026-08-09 / Codex.

- Observation: persisted Verifiers traces from online-RL runs were not
  joinable to optimizer-step metrics or to the active-sampling candidate batch
  that produced them.
  Evidence: the Trackio trace `step` is a provider ingestion/index value; the
  sampled records exposed `task_index` and environment identity but no logical
  optimizer step. A request for traces with `step=39` returned a trace observed
  near run startup, not the rollout population that produced optimizer update
  39. The local generic TRL adapter fix now uses upcoming optimizer step
  `trainer.state.global_step + 1` consistently for rollout phases, population
  metrics, and the bridge, and adds `optimizer_step` plus a per-step
  `rollout_batch_ordinal` to every trace. The ordinal resets when the optimizer
  step changes. All 47 train API tests, the complete 161-test train suite, and
  the 1,109-test repository suite pass; full Ruff, Pyright, import contracts,
  and diff hygiene also pass. This
  improves future runs only; it does not retroactively relabel this live run.
  Date/Author: 2026-08-09 / Codex.

- Observation: The active 256-concurrency package is built from the current
  framework source and the SFT adapter, but its selected settings do not enable
  the intended advantage fix.
  Evidence: the packaged project catalog contains `algorithm: dapo`,
  `mask_truncated_completions: true`, and DAPO clip bounds, but no
  `advantage_scaling`; the corrected `advfix` settings entry does contain
  `advantage_scaling: none`.
- Observation: The runtime image is content-addressed and contains the current
  framework source, so changing the project catalog or settings requires a new
  actual-job package digest even when the framework release image is reused.
  Evidence: current run image
  `registry.lan/carbonteq/posttrain-job@sha256:5a158389834c089edc4eb5743b559324a9b238e5eca71a13c258c33c39f676b4`;
  embedded `grpo.py` matches the working-tree source SHA
  `8277ce5cc5853db4048fbbaa64b1ccb6f35bf96c797658253b553e9a8efefae7`.
- Observation: The framework release checker currently fails before build or
  publication because one generated catalog lock digest is stale.
  Evidence: `posttrain-release check` reports `locks.toml` digest
  `0f122dfc...` but expects `2c482c10...`.
- Observation: The release source is not clean enough to tag safely.
  Evidence: `git status --porcelain` reports 69 modified or untracked paths,
  including train, Observatory, tracking, docs, packaging, and release tests.
- Observation: Prior SFT-backed DAPO runs have both positive and negative reward
  trajectories, so a negative short run is not by itself proof that the SFT
  adapter is unusable.
  Evidence: completed two-step Trackio runs recorded reward means
  `+0.647602 → +0.667955` and `−0.118092 → −0.212115` from the same SFT source
  tree. The corrected probe must therefore verify loading, parity, and learning
  telemetry before interpreting reward direction.
- Observation: The first protected `0.3.3rc1` build reached distribution
  receipt verification but stopped while assembling the wheelhouse because the
  repository did not contain the developer-facing `docs/release-and-consumption.md`
  file that the release script embeds as its README.
  Evidence: candidate workflow `31279099084` failed at the `cp` step in
  `scripts/release/build-python-distributions`; the guide is now present and
  release tests/checks pass locally.
- Observation: After the guide fix, the candidate exposed a second release
  pipeline defect: OCI publication happened after wheelhouse staging. The
  consumer virtualenv therefore installed a wheelhouse containing the old
  runtime-image manifest even though the registry had newly built images.
  Evidence: candidate `31280341495` refused the transform canary because the
  installed manifest expected lock `d26cadd4…` while its pinned registry image
  still carried `a9a2f0c8…` / framework `0.3.2`.
- Observation: Moving OCI publication before wheelhouse construction was not
  sufficient because the distribution builder deliberately stages from
  `git archive HEAD`, which excludes the generated `published.toml` mutation.
  Evidence: candidate `31280782204` published fresh transform digest
  `c26e6451…`, but the clean consumer still packed old digest `c772968f…` and
  the dstack canary reported framework `0.3.2` / lock `a9a2f0c8…`.
- Observation: The correct ownership boundary is a narrow overlay of the
  generated runtime manifest after immutable source staging, not a broad dirty
  checkout copy. The overlay is covered by a release regression test and will
  be requalified in the next protected candidate.
- Observation: The narrow checkout overlay is a useful regression repair but
  not a sufficient release architecture. It does not attest who generated the
  manifest, which dependency deployment the canary exercised, or whether the
  runtime-image dependency lock matches the Python wheelhouse.
  Evidence: candidate `31281418857` successfully consumed the refreshed image
  manifest and launched the packed job, then failed while Trackio committed an
  artifact manifest whose referenced blob was absent from the deployed server.
- Observation: a dedicated Trackio-owned release workflow is not required to
  unblock this release because the operator completed the publication and
  deployment manually and retained live readback evidence. Automation remains
  a follow-up; Posttrain consumes and verifies Trackio rather than publishing it.
  Evidence: the live post11 producer/consumer qualification succeeded only
  after deleting the producer cache, proving the artifact was served by the
  deployed S3-backed path rather than local residue.
- Observation: Trackio dependency versions currently have more than one source
  of truth across package metadata, `uv.lock` and runtime-image profiles/locks.
  A framework release can therefore build a client wheelhouse and an execution
  image against different Trackio versions unless CI rejects the drift.
- Observation: Observatory is named in the supported release workflow but is
  not yet qualified as a deployed read product with an immutable image/config
  receipt. A local import or package test cannot prove production readback.
- Observation: Ambient Agent cannot select `algorithm: olmo3` before the
  Posttrain release is installed. Its current project metadata still resolves
  `posttrain==0.3.2`, Trackio post10 and the legacy TRL Git commit `91b0ce...`,
  none of which contains the named Posttrain OLMo 3 selection contract.
  Evidence: `/home/hammad/projects/ambient-agent/pyproject.toml` and `uv.lock`
  retain those exact constraints. Changing only the project YAML would fail
  catalog decoding or execute the wrong trainer implementation.
- Observation: candidate `31289887803` did not fail compilation or dependency
  resolution. The veRL image finished exporting, then the registry rejected its
  push with `unknown: invalid content range`; eval, TRL online-RL and serve
  images had already pushed successfully. This is a resumable registry-upload
  failure and must be retried at the BuildKit boundary, not interpreted as an
  invalid runtime image or handled by restarting the entire release manually.
- Observation: candidate retry `31290415458` showed that the registry's
  resumable-upload symptom masked storage exhaustion. Its second classified
  attempt reached the primary filesystem error: `/var/lib/registry` had no
  space left. Deleting small actual-job manifests alone reclaimed little
  because their layers were shared; superseded tagged runtime images owned the
  reclaimable capacity.

## Decision Log

- Decision: Use step five as the exact retained-batch algorithm audit and treat
  earlier multi-round reward/advantage summaries as candidate-population
  diagnostics.
  Rationale: step five required one generation round and retained all 128
  generated rows, eliminating the active-refill population ambiguity without
  stopping or relabeling the production campaign.
  Date/Author: 2026-08-09 / Codex.

- Decision: Defer exact-worker smoke cleanup instead of bypassing provider
  verification or deleting downstream evidence first.
  Rationale: the active production run owns the worker. A durable provider
  retry preserves exact-host/path verification and keeps image, Trackio, and
  local evidence available until cleanup actually succeeds.
  Date/Author: 2026-08-09 / Codex.

- Decision: Do not cancel the healthy production run merely to release its
  worker for deferred smoke cleanup, even after checkpoint 25 exists.
  Rationale: the runtime has a cooperative SIGTERM/SIGINT finalizer and the
  framework can resume a new run through `job run --resume-from-run`, but the
  live dstack deployment still uses upstream `0.20.29`. The maintained dstack
  consumer ledger explicitly says that deployment has not qualified the
  greater-than-ten-second stop-duration propagation needed to guarantee
  Trackio checkpoint publication before forced removal. The local checkpoint
  proves recoverability of the retained workspace, not safe interruption and
  remote publication. Let production continue; keep exact cleanup queued.
  Date/Author: 2026-08-09 / Codex.

- Decision: Treat the earlier pre-release 256-concurrency diagnostic as
  rejected DAPO evidence; it was superseded by the released OLMo 3 campaign
  whose artifact metadata records `advantage_scaling: none`.
  Rationale: the earlier comparison confounded the algorithm fix with the
  previous group-normalized behavior, while the accepted r5 gate and current
  production package use the corrected named algorithm contract.
  Date/Author: 2026-08-09 / Codex.
- Decision: Keep the DAPO reward scalar and retain component rewards only as
  diagnostic telemetry.
  Rationale: this is DAPO, not GDPO; splitting objectives would change the
  experiment rather than repair the scalar learning signal.
  Date/Author: 2026-08-09 / Codex.
- Decision: Use the existing SFT LoRA adapter as the policy input and require a
  LoRA adapter/checkpoint output, not a full model export.
  Rationale: the question is whether DAPO can improve the trained policy, and
  the recovery contract must preserve adapter, optimizer, scheduler, trainer,
  and RNG state.
  Date/Author: 2026-08-09 / Codex.
- Decision: Use one 32 prompt groups × 4 generations optimizer step before the
  200-step campaign. Keep environment and serving concurrency at 256 for the
  capacity test, but retain global training batch 128 and the existing safe
  microbatch / accumulation contract. Submit 200 steps only when the one-step
  evidence passes, then supervise its first five completed steps.
  Rationale: one complete rollout and actor update is sufficient to verify the
  algorithm contract without producing another disposable multi-step smoke run;
  it is a gate, not a production-learning claim.
  Date/Author: 2026-08-09 / Codex.
- Decision: Release only from one clean, reviewed commit and use the existing
  protected candidate/final workflows. Do not hand-build a tag from a dirty
  tree or publish an unqualified wheelhouse.
  Rationale: the release workflow is designed to preserve exact source,
  dependency, image, and distribution receipts and to create the final tag last.
  Date/Author: 2026-08-09 / Codex.
- Decision: Make the release-consumption guide a first-class source document
  and wheelhouse README rather than teaching consumers from a workflow-only
  implementation detail.
  Rationale: the release script intentionally embeds this guide in every
  wheelhouse; omitting it makes a technically built release unusable to an
  independent consumer and caused the candidate build failure.
  Date/Author: 2026-08-09 / Codex.
- Decision: Publish changed OCI inputs before building the Python wheelhouse in
  the candidate workflow.
  Rationale: `published.toml` is packaged inside `posttrain-runtime-images`; a
  clean consumer can only select the newly qualified image digests when the
  wheelhouse is built after image publication. The canary must fail closed on
  drift rather than bypassing the check with `--build-missing`.
  Date/Author: 2026-08-09 / Codex.
- Decision: Keep `git archive` as the release source boundary, but explicitly
  overlay the generated runtime-image manifest into the staged tree before
  building distributions.
  Rationale: this preserves protection against arbitrary dirty state while
  ensuring the wheelhouse and OCI registry carry identical image digests.
  Date/Author: 2026-08-09 / Codex.
- Decision: Supersede the checkout-copy mechanism as the production boundary
  with an explicit release materialization receipt. `git archive` remains the
  committed-source input; only hash-declared generated inputs are projected
  into the staged tree.
  Rationale: a generated file is valid release input, but it is not source and
  must not enter the build through ambient working-tree mutation.
  Date/Author: 2026-08-09 / Codex.
- Decision: Require maintained dependencies to publish, deploy and qualify
  themselves before the Posttrain candidate begins. Posttrain verifies their
  receipts and never publishes Trackio as an incidental workflow step.
  Rationale: package availability, server deployment and live compatibility are
  distinct states with different owners and credentials.
  Date/Author: 2026-08-09 / Codex.
- Decision: Treat Trackio artifact round trip and Observatory readback as
  first-class candidate gates against the same packed run.
  Rationale: a remote job that reaches terminal provider state is not a
  supported Posttrain result if its artifacts cannot be committed or its
  evidence cannot be read through the product surface.
  Date/Author: 2026-08-09 / Codex.
- Decision: Finish the framework release before updating Ambient Agent or
  submitting further DAPO runs.
  Rationale: experiments must consume the exact stable framework and dependency
  graph being qualified; running from the release branch would recreate the
  provenance ambiguity this release is intended to remove.
  Date/Author: 2026-08-09 / Codex.
- Decision: Use the named `Olmo3GRPOConfig` recipe for the next post-release
  Ambient Agent learning canary rather than approximating it with another DAPO
  settings combination.
  Rationale: active refill and token-level TIS are trainer/runtime behavior, not
  aliases for `loss_type`; selecting `algorithm: olmo3` binds the complete
  recipe and its invariant checks while retaining the common `GRPOTrainer`.
  Historical DAPO runs remain valid comparisons under their original labels.
  Date/Author: 2026-08-09 / Codex.

## Outcomes & Retrospective

Posttrain `0.3.4` is published from release commit
`6b64b067` and merged main commit `5a311a5`. Ambient Agent consumes that stable
release with Trackio post11 and TRL `1.9.2.post1`, and its named OLMo 3 work
packages retain the SFT LoRA lineage. Registry cleanup restored 77 GiB of
host-root capacity without losing any protected release image.

The accepted r5 gate corrected both the Qwen 3.5 LoRA namespace and the
actor-update observation boundary, completed one optimizer step on RTX PRO
6000, and published a LoRA-only model artifact plus recovery checkpoint. Its
first 1,536-token version was rejected for excessive truncation; revision 2
raised the completion cap to 2,048 and reduced truncation to `0.7812%`.

The 200-step production campaign is running from the materialized r5 adapter.
Five monitored steps completed with finite loss and gradients, centered
advantages, nonzero reward spread, stable entropy (`0.2446` to `0.2714`), mean
TIS ratio near one with no clamping, and both asymmetric clip boundaries
exercised during warmup. The exact step-five retained batch had no zero-gradient
groups. This is sufficient to accept algorithm execution and continue the
campaign; five steps are not sufficient to claim a reward trend or final model
improvement.

Remaining operational work is bounded: the three superseded smoke workspaces
are queued for exact-worker cleanup after production capacity becomes
available, and the live Observatory population semantics for retained versus
candidate reward/advantage summaries need a follow-up implementation before a
future release. The r5 lineage starter and production run remain protected.

## Context and Orientation

The framework is a Python 3.12/3.13 `uv` workspace. `packages/train` owns the
backend-neutral training settings and TRL adapter, `packages/jobs` owns the
standard job definitions, `packages/catalog` owns framework selections and
dependency-lock metadata, `packages/runtime-images` owns framework image
definitions and the generated published-image manifest, and `apps/observatory`
reads provider-neutral evidence. The `posttrain-release` package in
`apps/release` is framework-owner tooling and is intentionally not installed by
consumer projects.

The experiment lives in the sibling project `/home/hammad/projects/ambient-agent`.
Its work package binds the SFT adapter, a Verifiers environment, a training
selection, an inference binding, and an RTX PRO execution target. A run is one
execution of that package; Trackio and dstack IDs are evidence references, not
the work-package identity.

The active diagnostic run is:

    framework run: ambient-k1-dapo-rtxpro-sft-g4-seq256-2step-20260809
    dstack provider: pt-7cfc0eebe7f3a2ff28b2bd85
    Trackio: 1823843611f24ad4a71efdeb66240e5d
    image: registry.lan/carbonteq/posttrain-job@sha256:5a158389834c089edc4eb5743b559324a9b238e5eca71a13c258c33c39f676b4

The corrected run must preserve the SFT model variant and environment revision,
record the framework/project/configuration digests in its snapshot, and emit
the DAPO telemetry required by `docs/post-training/06-observation-and-lineage.md`.

## Plan of Work

First, finish the current release from the accepted manual dependency receipts.
Trackio post11 and TRL 1.9.2.post1 are immutable inputs; the protected candidate
must materialize their exact wheel URLs into the runtime constraint lock before
publishing images. Retain the generated lock and `published.toml`, apply both to
the release branch, rerun strict CI, and only then merge.

The corrected candidate must pass exact-SHA CI, registry and development-index
readback, clean index-only installation, a bounded packed dstack job, Trackio
artifact finalization and Observatory readback. The final workflow uses the
exact merged SHA and accepted materialization, verifies identical stable bytes,
and creates `v<version>` only after all gates pass. Update Ambient Agent to that
stable version, remove its old direct Posttrain/TRL/Trackio resolution, and
revalidate the SFT model, environment, inference and training bindings before
scheduling work.

Then add an additive, versioned Ambient Agent setting with `algorithm: olmo3`.
It must resolve the released `Olmo3GRPOConfig`: zero-gradient filtering with
bounded active refill, global token-level loss normalization, beta `0`, clipping
`0.20/0.272`, token-level TIS with no lower cap and upper cap `2.0`, and
mean-only group advantages. Keep the selected batch, generation, length,
learning schedule, truncation policy and active-refill budget explicit in the
project catalog. Do not mutate or rename an existing DAPO selection.

Bind the new setting to the retained 1,938-update SFT LoRA model variant and its
matching vLLM inference binding. Run exactly one logical optimizer step before
the 200-step campaign. Retain reward components, group spread, active-sampling
rounds, generated/retained rows, advantage statistics, truncation semantics,
TIS ratios, clip fractions, entropy, rollout/runtime telemetry, actor/sampler
parity, optimizer completion, recovery checkpoint and LoRA artifact.
Independently recompute advantages and require a floating-point tolerance match
before a longer run is considered.

## Concrete Steps

Work from `/home/hammad/projects/rl` unless a command names the ambient-agent
checkout.

1. Refresh release state without mutating authoritative systems:

       git status --short
       uv run --no-sync posttrain-release check
       git show --stat --oneline HEAD

   Confirm the exact release branch, source SHA, candidate receipts and current
   deployed dependency identities. Do not infer deployment from a GitHub tag.

2. In the Trackio and `ai-infra` repositories, implement the protected,
   repository-scoped internal publication and deployment transaction. Produce a
   receipt binding post11 source, wheel/sdist hashes, internal-index readback,
   service image digest, deployed configuration identity and the dedicated S3
   artifact compatibility canary.

3. In Posttrain, add the materialization model and explicit `stage` input,
   migrate the candidate/final workflows to it, enforce the single dependency
   lock, and remove the temporary Posttrain-owned Trackio publisher. Add focused
   receipt, tamper, stale-input and dependency-drift tests before changing the
   workflow.

4. Push the reviewed release branch and wait for exact-SHA quality CI. Dispatch
   **Prepare release candidate** only after the dependency receipt is accepted.
   Inspect every gate result independently; a candidate is rejected if Trackio
   writes or Observatory readback fail even when dstack reaches terminal state.

5. Merge the release PR only after the candidate passes. Dispatch **Publish
   release** with the exact merged SHA and accepted materialization, verify
   stable readback, and create the tag/GitHub Release last. Never rebuild or
   repair a dependency inside final qualification.

6. Update `/home/hammad/projects/ambient-agent` to the exact stable framework
   release. Update `pyproject.toml` and `uv.lock` so Posttrain, Trackio and TRL
   resolve through the released framework graph rather than the current
   `posttrain==0.3.2`, Trackio post10 and direct legacy TRL Git pin. Validate the
   catalog and save the dependency receipt before editing the job selection.

7. Add a new Ambient `algorithm: olmo3` catalog entry and new work package. Bind
   it to `models/qwen3.5-2b-sft-10k-json@lora-v0`, the matching SFT-policy vLLM
   inference binding, the selected verifier environment and an effective batch
   equal to prompt groups times generations. Verify the resolved trainer config
   is `Olmo3GRPOConfig` and that the immutable recipe fields match the release.

8. Pack and run the one-step OLMo 3 canary. Save the package JSON, resolved
   inputs, dstack and Trackio identities, telemetry and final evidence query.
   Recompute mean-only advantages independently, verify actor/sampler parity,
   active-refill behavior and TIS bounds, and confirm both the recovery
   checkpoint and LoRA artifact before considering a longer run.

9. If and only if the canary passes, create a distinct 200-step work package
   with the same SFT input and algorithm contract, submit it to RTX PRO 6000,
   and supervise the first five completed optimizer steps. Stop and diagnose
   architecturally if reward spread, advantages, clipping/TIS, parity,
   checkpoints, runtime health, or Trackio finalization violates the gate.

## Validation and Acceptance

The experiment is accepted only when the run snapshot resolves the SFT adapter,
the intended environment and target, and `algorithm: olmo3`; the resolved
configuration must prove active sampling, mean-only advantages, token-level
global normalization, zero KL, `0.20/0.272` clipping and token-level TIS capped
at `2.0`. The logical step completes rollout and actor-update phases;
independent advantage recomputation matches telemetry within the documented
tolerance; truncated
completions contribute neither group statistics nor loss; actor and sampler
weights are equal at the parity gate; a recovery checkpoint and LoRA adapter
artifact are linked to the run; and Trackio finalization succeeds.

The release is accepted only when the maintained-dependency receipt proves the
exact internally published and deployed Trackio version; the materialization
binds source, locks and generated image evidence; the full quality ladder
passes; a clean environment installs exact index bytes without workspace/Git
sources; OCI digests match registry readback; the dstack, Trackio artifact and
Observatory readback gates succeed against one run; stable contains the exact
final files; and tag `v<version>` points to the verified merged commit. A
one-step probe remains diagnostic evidence, not a production-learning claim.

## Idempotence and Recovery

All catalog and plan edits are additive and versioned. Repacking after a config
change creates a new content-addressed actual-job image. Re-running the one-step
probe creates a new run identity and never overwrites prior evidence. If the
probe fails before its first checkpoint, record that no safe resume point exists;
otherwise resume only from its immutable training checkpoint and retain the
original run as the parent attempt.

Release staging is isolated and can be repeated against any manifest version.
If candidate qualification fails, publish no final tag and allocate the next RC
after the fix. If final promotion succeeds but GitHub finalization fails, retry
only tag/Release creation from the retained receipt. If the release-scope audit
cannot separate user edits safely, stop before commit and report the exact paths
instead of using `git reset`, `git clean`, or a broad destructive cleanup.

## Artifacts and Notes

Retain the corrected run’s package JSON, resolved catalog snapshot, dstack and
Trackio IDs, independent advantage-recompute transcript, checkpoint and LoRA
artifact references, and a concise metrics receipt. Retain the release branch
commit, `release/manifest.toml`, generated lock diff, wheelhouse receipt,
published image manifest, candidate workflow URL, final workflow URL, stable
readback hashes, and GitHub Release URL. Redact tokens, registry credentials,
and private certificate material from every report.

Current evidence anchors:

    current framework source grpo.py sha256:
    8277ce5cc5853db4048fbbaa64b1ccb6f35bf96c797658253b553e9a8efefae7
    SFT adapter source tree sha256:
    79b17299c7808b373eaa67f3c34153ab513e27d2f28105f0ef633c58cccaa7b7
    published Posttrain release commit:
    6b64b067 (v0.3.4)
    merged main commit containing the release:
    5a311a5
    stable release:
    https://github.com/carbonteq-ai/posttrain/releases/tag/v0.3.4

## Interfaces and Dependencies

The experiment uses `posttrain.train.GRPOSettings.algorithm = "olmo3"`, the
released TRL `Olmo3GRPOConfig`, and the shared adapter in
`packages/train/src/posttrain/train/backends/trl/grpo.py`; the project catalog
and work-package files remain in `/home/hammad/projects/ambient-agent/.posttrain`.
The run lifecycle uses dstack through `packages/execution-dstack` and evidence
through the provider-neutral Trackio adapter. The release uses
`apps/release/src/posttrain_release`, `release/manifest.toml`,
`packages/catalog/src/posttrain/catalog/base/locks.toml`,
`packages/runtime-images/src/posttrain/runtime_images/published.toml`, and the
protected workflows `.github/workflows/release-candidate.yml` and
`.github/workflows/release.yml`.

The framework release must preserve the canonical package boundaries: common
contracts remain backend-neutral; train, eval, and serve remain independent;
Observatory reads providers only through tracking adapters; and framework
runtime images are distinct from private actual-job images. Dependency pins are
immutable and the lockfile is regenerated, not hand-edited.

Revision note (2026-08-09): created this combined living plan after discovering
that the active 256-concurrency run did not select the explicit advantage fix
and that the framework release checker failed on a stale generated lock digest.
Revised it after the OLMo 3 recipe was implemented so the next experiment first
updates Ambient Agent to the stable framework, then adds an additive OLMo 3
selection, runs one SFT-LoRA-backed optimizer step at concurrency 256 on RTX PRO
6000, and schedules 200 steps only after the algorithm evidence passes.
