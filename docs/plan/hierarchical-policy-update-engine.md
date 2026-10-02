# Implement a general policy update engine with explicit algorithm contracts

This ExecPlan is a living document maintained under `docs/templates/PLAN.md`.
Revision 136, 2026-10-02. Update Progress, Surprises & Discoveries, Decision Log and
Outcomes & Retrospective at every implementation stopping point. This revision
replaces the earlier GSPO/SAMPO-only plan with the reviewed general-engine scope.
Implementation is authorized by the active thread goal. Native qualification
remains open; pure contract tests do not certify a backend.
The planning skill's suggested .agents/PLAN.md is absent; the repository template
and canonical product documents govern.

## Purpose / Big Picture


A developer will be able to collect trajectories, calculate credit on complete
declared populations, select episodes, turns or semantic spans for an optimizer
update, and fit that update into GPU packs without silently changing the
algorithm. Process reward models can supply step-addressed evidence independently
of which regions receive policy or KL loss. Posttrain defines the resolved
meaning; TRL and veRL execute that meaning using their supported native machinery.

The engine is general, not universal. Versioned algorithm implementations declare
supported credit, masks, ratios, derivatives and reductions. There is no arbitrary
objective-language compiler or promise that every combination is valid.

A visible acceptance example has two episodes containing three and two assistant
turns. Packs [A1,A2], [A3,B1], [B2] produce one optimizer update matching the
unsplit reference. A second example makes two optimizer minibatches from the
same frozen population and shows two applied updates with refreshed current
scores and unchanged old scores. A third attaches scores to reasoning steps
and independently selects reasoning policy loss and declared KL support.
These prove semantics, not recipe quality.

SAMPO's author implementation already supports multiple optimizer minibatches
through veRL. Our current configuration constrains that capability. Implementation
must reuse native scheduling where it is faithful and add generic fork mechanisms
only for demonstrated gaps such as cross-record ratio dependencies or replay.
Small-GPU qualification checks correctness; research and ablations guide recipes.

## Progress

- [x] Revision136 (2026-10-02) closes multi-GPU as out of scope (user decision)
  and prepares release 0.4.14: public guard rejects resolved selections with
  `world_size != 1` (tests in `packages/train/tests/test_policy_update_settings.py`),
  `posttrain-release prepare 0.4.14`, CHANGELOG entry, and local publication of
  the four kind images whose inputs changed (supervised, online-rl-trl-py312,
  online-rl-verl-py313, eval). Stale unchecked boxes below record superseded
  intermediate revisions; their final state is in Outcomes.

- [x] Revision134 TRL post14 consumer adoption. Following `docs/tooling/forks.md`
  steps 11-13: `packages/train/pyproject.toml` pins `trl==1.12.0.post14` with
  `[tool.posttrain.trl]` tag `carbonteq-v1.12.0.post14`, source
  `09312dd1d96603587d1a24c3b714836cf67400f1`, wheel `ef77a1f0...dc40`, sdist
  `ed6a99c7...bac2dd` (both indexes serve those exact hashes); `uv lock` moves
  only trl post13->post14; `release/forks.toml`, the CI wheel URL/hash in
  `.github/workflows/quality.yml`, `profiles/supervised.txt`, release tests and
  the TRL parity test name post14. `posttrain-release lock-runtime-dependencies`
  regenerates workspace/supervised/online-rl-trl-py312 locks; the base catalog
  `trl-fork@current` lock takes the generator-reported digest
  `82201d74...0d691`. `posttrain-release check --allow-pending-runtime-lock`
  passes; `fork-ledger` resolves trl to 09312dd1. Locked sync installs post14.
  The 21 lab training bindings that reference `trl-fork@current` (comparison,
  precision-qualification and vortex-parity catalogs) carried stale post12
  backend/revision/digest facts since the uncommitted post13 bump; they now
  name `trl@1.12.0.post14`, 09312dd1 and the lock digest, which equals
  `sha256(uv.lock)`. Full workspace pytest: 2562 passed, 97 skipped; remaining
  failures are only the three known renderer mask cases.
  Kind images still need a candidate rebuild before jobs execute post14.
- [x] Revision134 consumer validation: ruff, import contracts (9 kept), pyright
  (0 errors after fixing three pre-existing issues: typed empty-rank anchor in
  `policy_update_distributed_execution.py`, which previously passed integer 0
  instead of a tensor or None, and two TRL test imports through lazy `trl`
  exports) and `git diff --check` pass. Train suite: 958 passed; the only
  failures are the three known LFM renderer SFT header-mask cases
  (`test_rendering.py`), which do not involve TRL.
- [x] Revision134 admits SAMPO active collection at the native veRL guard:
  `policy_native.py::validate_native_selection` previously rejected every
  `active_sampling`, while `contracts.py` requires it for any SAMPO manifest, so
  the resolved native SAMPO path could not run end to end. It now admits
  active sampling only for `SAMPOSettings` (native entrypoint always installs
  `NativeActiveCollectionBuffer` with host context); adaptive curriculum,
  dynamic sampling and GRPO-family active sampling stay rejected and the public
  `requests.py` guard is unchanged. New regression
  `test_native_selection_admits_recorded_sampo_active_collection_only`; native
  veRL cohort (collection/driver/capabilities/inventory/runtime) passes 85 tests
  with real veRL `../verl-posttrain-parity`.
- [x] Revision134 veRL 0.9.0.post9 released and adopted: release commit
  8e513f3b (post8 plus the scoped-arithmetic candidate through c45392d2 and its
  records; all additions opt-in; 310 post8-relative CPU regressions pass),
  receipt 6b3ceef7, wheel 13930e0f..., sdist 726f02c6..., published to dev by
  run 36996695048. `release/forks.toml`, the online-rl-verl-py313 release
  pyproject/lock/constraints and profile digests, release tests and
  `_FORK_NATIVE_NAME_REVISIONS` select post9; `posttrain-release check` and
  `fork-ledger` resolve post9. Lab veRL bindings keep the published post8
  image until the rebuilt kind image is published.
- [x] Revision134 pushed branch `codex/hierarchical-policy-update-engine`
  (commit 005bfae5, built from a temporary index so the shared checkout stays on
  main with its dirty tree) and dispatched runtime-image candidate run
  36997130090 for post14/dev3/post9 kind images.
- [x] Revision134 TRL resolved active collection (code + CPU gates):
  `update_active_rounds.py::ActiveRoundPlan` ports TRL post11 round arithmetic
  and metrics, matching veRL `ActiveSamplingRounds` exactly in four parametrized
  parity cases; `policy_rollouts.py::collect_active_resolved_population` runs
  rounds over one reserved distinct-task pool with the shaped-reward zero-epsilon
  spread rule, publishes content-addressed `training-collection` evidence
  (shared `backends/policy_collection_evidence.py`, also used by veRL) for every
  reserved candidate before admission, admits only selected groups and emits
  active-sampling metrics. `ResolvedTRLJob` reserves
  `num_prompts_per_step*max_candidate_batches` tasks per update; the resolved
  guard admits SAMPO active sampling only; native TRL refill arguments are
  removed under a resolved selection. New `test_trl_active_collection.py`
  passes with real veRL parity; full ladder 2574 passed, 0 failed. GPU
  qualification R135 is next.
- [x] Revision134 semantic reasoning/answer spans (code + CPU gates). The
  environment-owned producer `verifiers_conditioning.py::native_reasoning_partition`
  splits each sampled assistant call's original eligible actions at the
  renderer's `usage.reasoning_tokens` (thinking is the leading sampled run),
  rejecting missing accounting or `completion_tokens` that disagree with the
  retained sampled actions (revision `verifiers.renderer-reasoning-prefix@1`).
  `update_spans.py::reasoning_answer_spans` addresses those indices as
  `SemanticSpan`s in the population's exact action coordinates (multi-interval
  runs). Admission derives them from the retained traces when
  `objective_variant: semantic-spans` supplies no spans. `ActionSelection` gains
  `mode: roles` (`roles: (reasoning, answer, ...)`) so catalogs select by role
  across populations; policy and KL selections stay independent. TRL resolved
  SAMPO and native veRL SAMPO now advertise/admit `sampo-spans@1`; the public
  request guard stays closed. Regressions: 7 environment partition cases, 3
  admission/selection cases on the real admission fixture. Full ladder 2577
  passed, native veRL cohort 96 passed. GPU qualification of span-selected
  updates is open.
- [x] Revision134 resolved TRL capabilities omitted `sampo@1`, so resolved TRL
  SAMPO would have been rejected at plan resolution; it is now advertised and
  exercised by the R135 CPU preflight (two `sampo@1` updates resolved).
- [x] Revision134 granular process credit (code + CPU fixture gates):
  `update_process_credit.py` adds the composition-injected `SpanScorer`
  protocol, `assess_population_spans` (every selected retained span scored
  exactly once by span ID, matching scorer revision, observation scope and
  retained evidence) and `ExternalSpanCreditEstimator`, an explicitly
  identified estimator mapping complete assessments to detached per-span
  advantages (zero elsewhere; overlapping assessed spans rejected without a
  declared combination estimator) that `prepare_credit` validates like any
  estimator. The planned fixture
  `packages/train/tests/fixtures/policy_updates/two-episode-reasoning.json`
  (two native episodes with renderer reasoning accounting) drives admission,
  derived reasoning spans, a fake transport scorer, a known external estimator
  and `sampo-spans@1` resolution; scorer evidence returned as credit, partial
  or foreign-revision assessments and incomplete estimator output are
  rejected. Live gate still requires a real injected local scorer.
- [x] Revision134 fixes missing renderer reasoning accounting on in-process
  training paths: `PolicyTurnResult.reasoning_tokens` now carries the renderer
  parse count from the TRL and veRL generators into Verifiers call `usage`.
  Previously every TRL/veRL training trace recorded `reasoning_tokens=None`
  (R135 evidence), leaving thinking-token facts unsupported and blocking span
  extraction. Regression asserts a real native Verifiers episode records it.
- [x] Revision134 R135b qualifies TRL resolved SAMPO active collection on
  LFM2.5-2.6B with installed post14 + renderers dev3 (every installed source
  verified against pinned commits), ActiveGroupSampling(6) over six
  historically informative tasks. BF16: rounds discard [1,1,1,1] and [0,0,0,0],
  select [1,1,1,0], three candidates unused; 2537/2537 nonzero advantages,
  nonzero KL at update 2, independent FP64 objective/gradient/clip checks pass,
  32 adapters change, active metrics equal snapshot counts. FP16: five rounds
  discard three all-success and one all-fail group, select [0,0,1,0], one
  unused; 3103/3103 nonzero advantages. Checkpoint1 continuation matches exactly
  in both precisions (FP16 164 tensors, difference 0) without recollection.
  Receipts: `/var/lib/posttrain/qualifications/corrected-trl-20261002-r135b/`.
- [x] Revision134 R135 TRL active qualification attempts (all evidence kept):
  r135-failed-precomputed-advantages (legacy `use_precomputed_advantages`
  without rollout_func; fixed by `resolved_native_arguments`),
  r135-failed-preflight-source-match (qualifier text check; replaced by a
  behavior check), r135 (preflight accepted; fresh collection ran three rounds
  with complete evidence and correctly failed: all 12 episodes reward 0, as in
  R124's two uniform tasks). Superseded by R135b above.
- [x] Revision134 native veRL data-parallel resolved updates (code + CPU gates):
  `ResolvedVeRLPopulation` assigns each update's original contexts to ranks
  (least-loaded by token count; partitions may be unequal or empty), pads every
  rank to the same number of native forwards (FSDP gathers per forward),
  all-gathers owned old/reference/current scores as exact CPU tensors so every
  rank holds identical global scores and adjoints, and replays only owned
  contexts with carriers scaled by the data-parallel size (native FSDP averages
  gradients); padding forwards contribute exact zeros. Checkpoints: every rank
  saves its native shards, rank 0 alone seals the population with all ranks'
  files and `world_size=N`, so resume with a different size is rejected by
  identity. `single_actor_result` accepts N ranks only when their global state
  and metrics are identical; the worker observer journals only on rank 0; the
  native guard admits one node with `target.world_size == devices_per_node`.
  Real two- and three-process gloo DDP runs reproduce the single-rank objective
  and every parameter exactly (three ranks over two contexts include an empty
  rank). Execution packs remain one context per rank in data-parallel mode.
  GPU qualification R137 (two-GPU pod on RunPod) is OPEN: attempts on
  2026-10-02 hit RunPod two-GPU capacity failures, three uniform-reward
  collections (samples verified distinct; no defect), and three qualifier
  staging mistakes (two-context packs rejected by the data-parallel guard, a
  dropped variable, an empty manifest hash). On the user's cost decision the
  run was stopped (RunPod spend for these attempts about $5; no pods remain).
  The single-host data-parallel path stays proven only by the real gloo
  two/three-process tests; no in-house two-GPU host exists. A later attempt on 2x RTX PRO 4500 (EU-RO-1, $1.44/h) failed at pod
  start (`waiting_instance_limit_exceeded`; the dstack agent never connected,
  about $0.50 billed). R137 is recorded OPEN with no further paid retries
  without user approval; multi-node and distributed TRL are out of scope. TRL has
  no multi-process launcher in Posttrain; distributed TRL remains unsupported
  and explicitly rejected.
- [x] Revision134 R136 BF16 qualifies native veRL SAMPO active collection on
  Gemma 4 E2B (google/gemma-4-E2B-it 3e22461f, text-only, 50 language q/v LoRA
  targets, rank 8): rounds discard [0,0,0,0] and [1,1,1,1], select [1,1,0,1],
  three candidates unused; 394/394 nonzero advantages; all 100 adapter tensors
  change. Update 2 clips every selected action (clip fraction 1.0, gradient
  norm 0.0031): correct clipping, but the recipe's learning rate is aggressive
  for this small Gemma population. The production launcher still rejects
  `gemma4`; the qualifier bypasses that family check for the pinned policy.
  BF16 resume twice failed at Ray node registration before any training
  (`raylet` not registered within 30 s; attempts preserved as
  `*-failed-ray-startup*`); a diagnostic rerun with
  `RAY_raylet_start_wait_time_s=120` failed the same way. Root cause: the
  workstation root disk was 100% full, so the raylet plasma store got 0 bytes
  and aborted (raylet.out). Space was recovered without data loss: R134
  checkpoints were relocated to the development machine
  (`~/experiments/posttrain-correctness/2026-10-01/evidence/r134-checkpoints`,
  106 files sha256-verified) and checkpoint-publications that were
  byte-identical (sha256) to retained checkpoints were removed, with notes in
  each run directory. The BF16 resume then matched uninterrupted boundary 2
  exactly (2356 tensors, difference 0, no recollection), so Gemma 4 E2B BF16 is
  qualified end to end. FP16 is UNSUPPORTED for Gemma 4 E2B: the first forward
  pass at the untouched base policy (old-score freezing, before any update or
  checkpoint) produced non-finite logits, and the resolved engine correctly
  rejected scoring (`resolved veRL scores require finite logits`). This is FP16
  activation overflow in the model family, not an engine defect; the production
  launcher still rejects `gemma4`, and any future Gemma admission must exclude
  FP16.
- [x] Revision134 fixes renderer reasoning attribution on in-process training
  paths: TRL and veRL generators now pass the rendered prompt to
  `parse_response(prompt_ids=...)`. R135b evidence showed every LFM2.5-2.6B
  assistant message carried its thought and `</think>` inside `content`, with
  `reasoning_tokens=0`, because the template prefills `<think>` in the
  generation prompt and the parser never saw it opened. Regression uses the
  real cached 2.6B tokenizer/renderer.
- [x] Revision134 real local scorer (code + CPU gate):
  `apps/lab/src/posttrain_lab/scorers/likelihood.py::LikelihoodSpanScorer`
  (composition-owned) scores each retained span by the injected model's mean
  log-probability of its original tokens given their exact prefix, recording
  scorer revision, model snapshot, observed input digest and prefix scope. Its
  test checks every span score against an independent computation on the
  reasoning fixture and drives scorer -> external estimator -> prepare_credit
  -> `sampo-spans@1` resolution. GPU run on a real retained population with
  nonzero reasoning spans (R137 traces carry the prompt fix) is open.
- [x] Revision134 R138 qualifies semantic reasoning spans and the real local
  scorer on LFM2.5-2.6B (TRL resolved SAMPO, post14 + dev3, ActiveGroupSampling(6),
  BF16). Fresh traces: 24/24 calls report renderer reasoning (5505 tokens; the
  prefill fix works on GPU) with no think markers in content. Retained spans
  exactly partition every sampled call (1521 reasoning, 684 answer actions);
  each `sampo-spans@1` term's policy actions equal the reasoning spans and KL
  actions equal all actions; independent FP64 objective and equal-episode
  weights match; selected group [0,1,0,1]; 2205/2205 nonzero advantages; 32
  adapters change; checkpoint1 continuation exact (164 tensors). The
  composition-owned `LikelihoodSpanScorer` (base LFM2.5-2.6B 654f9463, FP32,
  weight SHA-256s recorded, prefix scope) assessed every reasoning span by ID,
  matched an independent recomputation within 3.9e-8, and the explicitly
  identified `group-centered-likelihood@1` estimator credited exactly the 1521
  reasoning actions; resolution yields `sampo-spans@1`. Receipts:
  `/var/lib/posttrain/qualifications/corrected-trl-20261002-r138/`. FP16 pair
  also passes: 25/25 calls report reasoning (8874 tokens), spans partition
  exactly (1977 reasoning / 741 answer actions), objective weights match in both
  updates, 2718/2718 nonzero advantages, exact continuation (164 tensors with
  scaler). TRL SAMPO `semantic-spans` is added to the public qualified matrix. Training with process credit inside a job (estimator selection
  in settings) is not wired; the scorer gate is evidence, credit and resolution.
- [x] Revision134 process credit inside training jobs (code + CPU gates):
  `PolicyUpdateSettings.credit_estimator` (catalog schema too) names the
  explicit external estimator whose detached credit replaces the algorithm's
  own. The host composition injects `SAMPORequest(process_credit=provider)`;
  request construction and admission both require a provider whose
  `estimator_id` equals the selection (no silent fallback to SAMPO credit).
  `AdmittedNativePopulation.from_rollouts` prepares credit through the
  provider on the retained population reader and re-resolves; resolved TRL
  collection paths pass it through. `ScoredSpanCreditProvider` (train) and the
  lab factory `likelihood_process_credit` with
  `group_centered_likelihood_estimate` (`group-centered-likelihood@1`) are the
  first provider. Recovery restores frozen credit from the checkpoint without
  rescoring. Tests: fixture admission replaces credit with exact values and
  rejects missing/mismatched providers; lab admission matches independent
  group-centered values; the public guard keeps `credit_estimator` closed until
  GPU qualification R139 (workstation) passes.
- [x] Revision134 R139 BF16 trains with composition-injected process credit
  (workstation, TRL resolved SAMPO semantic spans, LFM2.5-2.6B, post14 + dev3).
  Admitted credit is `group-centered-likelihood@1` (not SAMPO), assessed by the
  injected `LikelihoodSpanScorer` (base 2.6B, FP32, on the training GPU) over 8
  reasoning spans; 1647 credited actions; selected group [1,1,1,0]; 32 adapters
  change; the objective terms use that credit digest; independent recomputation
  (float32 forward, float64 probabilities) matches assessments within 2.9e-7
  and credit within 3.1e-7; checkpoint1 continuation restores the frozen credit
  without calling the scorer and matches exactly (164 tensors). First recompute
  attempt failed only in the qualifier (LFM kernels reject float64 forward;
  preserved as `*-failed-recompute-dtype`). FP16: first draw exhausted six
  uniform groups (preserved as `fp16-r139-uniform-draw`); the rerun passes:
  credit `group-centered-likelihood@1` over 2475 actions, selected group
  [1,0,0,0], recomputation within 1.2e-6, exact continuation (164 tensors).
  The public guard now admits exactly (TRL, SAMPO, semantic-spans,
  `group-centered-likelihood@1`); other estimators/backends stay closed.
- [x] Revision134 lifts the public `policy_updates` guard for qualified
  selections only. `requests.py::_resolved_selection_problem` admits the GPU-
  qualified matrix: TRL GRPO/DAPO (R124-R132) and SAMPO with active collection
  (R135b) on the `algorithm` objective variant with transformers generation,
  and native veRL SAMPO active collection (R134/R136). Everything else
  (semantic spans and turn rows until their GPU gates, GDPO/CAPO, vLLM-rollout
  TRL, other backends) is rejected with the reason; each backend's own
  admission still rejects narrower unqualified settings (distribution beyond
  qualified sizes, masks, curriculum). Legacy execution stays for absent
  `policy_updates` for one release. Matrix regression covers nine cases.
- [x] Revision134 observation gate (Observatory): metric catalog entries and
  run aggregation for resolved counters (applied/attempts max, selected
  actions sum, nonzero-advantage mean, loss/kl mean, loss scale last, skipped
  sum), computed `skipped_optimizer_attempts`, resolved-updates chart and
  conditional evidence requirement, training-collection/checkpoint artifact
  kinds; 188 backend and 120 frontend tests pass, build passes. Open: veRL
  emits no `resolved_policy_update_applied` denominator/digest event (TRL
  does); scored-vs-gradient counts, mask coverage, replay drift and dependency
  cost are not emitted by either backend; collection snapshot contents need an
  `ArtifactLink` metadata/content reader contract in `packages/tracking`.
- [x] Revision134 R134 native veRL SAMPO active collection passes on real
  Ray/TransferQueue: LFM2.5-2.6B 654f9463, R124's three tasks, one group x4,
  ActiveGroupSampling(3), two applied updates per run, image f0d32115, fork
  c45392d2, sealed source archive 7150c634 (363 files). BF16: round 1 group
  uniform [0,0,0,0] discarded as surplus evidence, round 2 [0,0,0,1] selected,
  third candidate recorded unused; 2567/2567 nonzero advantages, KL loss
  3.2e-6 at update 2, all 32 adapter tensors changed, frozen population/credit/
  scores/correction, collection receipts equal population traces/episodes.
  FP16: same pattern ([0,0,0,0] surplus, [0,1,0,1] selected), 3516/3516 nonzero
  advantages, 32 adapters changed. Checkpoint1 continuation without
  recollection matches uninterrupted boundary2 exactly for both precisions
  (399 tensors, max difference 0.0). Receipts:
  `/var/lib/posttrain/qualifications/full-verl-20261002-r134/r134-*.json`;
  qualifiers outside Git under
  `~/experiments/posttrain-correctness/2026-10-01/working/full-verl-r67/r134/`.
  Public SAMPO admission stays guarded: `requests.py` lifts per backend only
  after release gates, and veRL c45392d2 is not yet the released consumer fork
  (`release/forks.toml` selects post8).
- [x] Revision134 R134 exposed and fixed two production defects before any GPU
  time: (1) the worker emits `algorithm.active_sampling.metric=group_reward`
  while the driver accepted only `seq_reward`, and without a curriculum the
  native observe metric defaulted to `seq_reward`, which the collection
  recorder rejects; the driver now accepts `group_reward` and the worker sets
  `data.prompt_selector.metric=group_reward` when no curriculum is selected
  (curriculum keeps observing `seq_reward`), with the shared driver fixture now
  composed from the worker's real overrides; (2) `_model_path` failed offline
  because huggingface_hub 1.31 `snapshot_download` requires every repository
  file (README.md) even for an immutable pinned snapshot; a pinned commit with
  local `config.json` now resolves directly (regression
  `test_verl_offline_pinned_hub_model_uses_partial_local_snapshot`). Failed
  preflight attempts are preserved as `full-verl-20261002-r134-failed-*`.
- [x] Revision134 fixes the LFM renderer SFT mask failure. The unreleased fork
  fix `1aafe24` (exclude injected headers) regressed the 2.6B reasoning-history
  offsetless case because the template rewrites assistant turns preceding a
  later user message; renderers `7fe5d06` masks such turns in the rewritten
  form (new fork regression; full fork suite 11,834 passed). Released
  `carbonteq-v0.1.12.post1.dev3` (wheel 57dcc6f8..., sdist 650954a1...),
  published to dev by run 36996039659, adopted in pyproject/uv.lock/forks/CI/
  runtime locks/veRL release lock and profile digests, catalog and lab
  `trl-fork@current` digest db3b92d0 = sha256(uv.lock). Full ladder: ruff,
  pyright 0 errors, import contracts, diff check and pytest 2567 passed,
  0 failed.

- [x] Revision133 R132 installed-development-asset native suite is terminal
  exit0. Both BF16/FP16 branches pass two applied updates, independent objective/
  gradient/clipping verification and full checkpoint continuation. Final receipt
  `/var/lib/posttrain/qualifications/r132-suite-readback.json` declares complete.
  No new rollouts; original informative R124 evidence remains the authority.
- [x] Revision133 stable retained-asset promotion run36990000110 completes success:
  `https://github.com/carbonteq-ai/posttrain/actions/runs/36990000110`.
  It verifies development bytes, transfers unchanged files server-side and proves
  byte-identical stable readback. Stable consumer manifests/locks have NOT yet
  been updated: they still select TRL post13. Next adoption must update exact
  post14 commit09312dd1, release tag and retained hashes together, using the
  repository-owned lock generators and then consumer integration checks.
- [x] Revision133 adds private veRL `policy_collection.py::NativeActiveCollectionBuffer`
  around existing native dispatch/observation hooks. Native buffer retains round
  sizing, spread predicate, selection and eviction ownership. Content-addressed,
  fsynced collection artifacts record reserved/dispatched/finished/failed/selected
  candidates and original episode receipts; episode artifacts are submitted
  before native eviction. Uniform, surplus and unused candidates remain inspectable.
  Driver installs the recorder when actual host context is supplied; existing
  public SAMPO qualification guard remains unchanged. No new frozen-baseline
  meaning or veRL fork changes are required for this observation adapter.
- [x] Revision133 actual native CPU buffer/driver/capability/inventory cohort
  passes68 tests in5.95s (in-memory TQ transport, not real Ray/GPU collection).
  Covers refill, failed groups, surplus, unused reservation, publication before
  eviction, corrupt evidence/observer failure, task/group/policy membership,
  complete receipt coverage and actual driver factory wiring. Scoped Ruff and
  Pyright pass; import contracts9 kept/0 broken. Test sources are maintained
  regressions, while experimental qualifiers remain outside Git.
- [ ] Revision133 next native gate: qualify the observation adapter with real
  SAMPO Ray/TQ collection, informative optimizer updates and recovery before
  enabling public active SAMPO. TRL active collection, semantic/process credit,
  distributed/family/Gemma coverage and broader release checks remain open.
  R132 is terminal; do not restart it. Recheck GPU availability before any new
  launch and preserve other active runs. Follow the detailed acceptance scope
  below; these completed gates do not narrow the original engine goal.

- [x] Revision132 retained-asset publisher run36988812519 completes success:
  `https://github.com/carbonteq-ai/posttrain/actions/runs/36988812519`.
  Exact GitHub bytes are verified, stored in development, independently fetched
  and installed by the Posttrain-owned workflow. Stable promotion stays open.
  Fork release ledger commit `092a8a01` is pushed separately; source tag remains
  fixed at09312dd1 and is not moved for documentation updates.
- [x] Revision132 CPU installed-asset check in the exact qualification image
  installs post14 from development with no dependency changes, verifies trainer
  files against qualified R129 source, and passes all5 regressions in5.80s.
  Raw receipt `/var/lib/posttrain/qualifications/r131-index-runtime.log`.
- [ ] Revision132 final installed-asset native precision/recovery suite is
  launched after direct GPU idle verification (RTX PRO6000,41MiB,0%, no compute
  processes), and confirmed running with installed-source identity passing.
  Handle `posttrain-r132-index-native-precision-resume`, container
  `9d389465bd181f401a50cff452bed8b7500fb981ae3835fb09d20cb3218e9532`;
  remote `/var/lib/posttrain/qualifications/corrected-trl-20261002-r132-index`.
  Same image digest f0d32115d6e3bdca1dda76aad0ff8b9605e9869d4413ea65f76f4bc942cc85eb,
  frozen362-file R129 framework snapshot and R124 informative8340-action population.
  Installs released post14 from dev, verifies every installed TRL Python source
  against the qualified archive, then runs BF16 and FP16 two-update execution,
  independent objective/gradient/clipping verification and full continuation.
  Existing R129 packing/schedule comparisons remain the authority; no redundant
  fresh rollout collection. Network host permits only needed index installation;
  model/HF inputs remain pinned/offline/read-only. Observe this handle; do not
  restart on an observation timeout. Raw qualifier scripts stay outside Git.

- [x] Revision131 commits and pushes only the six owned generic TRL fork files
  at `09312dd1d96603587d1a24c3b714836cf67400f1`, branch
  `codex/hf-sampled-policy-scores`. Rebuilds from that clean commit; twine checks
  both distributions and exact trainer-source comparisons pass. Publishes GitHub
  prerelease `carbonteq-v1.12.0.post14`; GitHub independently reports retained
  wheel digest `ef77a1f07154d3958d9f32eb90d2ce01e16048ff5d7e33bbf5be9e215666dc40`
  and sdist `ed6a99c71f950af7147367ec582fda65f7cf22933460c86b73b7aadeabbac2dd`.
  Earlier revision130 archives were precommit builds and are not release assets.
- [ ] Revision131 dispatches Posttrain's retained-asset candidate publisher on
  remote main with the exact release digests. Remote workflow bytes match the
  reviewed local workflow (SHA25688f22b761c0be8b1cce439dbebd71039571bc78d9dc44ac87958732bda992a9a).
  Observe the exact GitHub run before claiming development-index publication.
  Stable promotion, candidate runtime qualification and consumer pins remain open.

- [x] Revision130 R129 is authoritatively terminal exit0. All four retained-
  evidence branches pass two applied updates, independent objective/gradient/
  clipping verification and full checkpoint1-to2 continuation: BF16 records2/
  budget6, BF16 records1/budget6, FP16 records2/budget6, and BF16 records2/budget12
  with two epochs. BF16 packing differences are exactly0 at both boundaries.
  The final full-population continuation compares164 tensors exactly, including
  optimizer/scheduler/RNG/frozen scores and sealed population/credit/correction.
  Final suite scope is retained evidence; fresh FP16 collection remains open.
  Raw final receipt: `/var/lib/posttrain/qualifications/r129-suite-readback.json`.
- [x] Revision130 local unpublished post14 wheel/sdist builds complete outside
  Git. Both archives contain byte-identical candidate trainer sources and correct
  version metadata. Wheel SHA256 `0a8c5698328907745bc4ba3c043cea7df55d66a35bbb9dd22bd3de63f250a00c`;
  sdist SHA256 `9f95d17b2de27180b5fe319a5caf0316789b7b804b210dd89dabf04beafb8182`.
  Build validity is not publication: fork commit/push, retained release assets,
  candidate-index qualification and stable promotion precede consumer adoption.

- [x] Revision129 R128 first two BF16 packing branches each pass2 informative
  updates, sealed/correction/mask checks and independent objective/gradient/
  clipping references. Each resume rejects before training because original
  sampler identity differs from current executor. Keep the existing suite live
  for FP16 and schedule evidence; do not modify its sealed source on disk.
- [x] Revision129 fixes Posttrain restore's provenance comparison: the sampler
  identity comes from authenticated retained evidence and frozen correction;
  current/old/reference executor versions remain independently exact. This
  implements existing separate sampler/old/current contracts, without a baseline
  amendment.14 focused tests pass, covering distinct sealed sampler preservation
  and rejection of changed old/current/reference versions, template and recipe.
  Ruff, source Pyright and diff checks pass.
- [ ] R129 corrected362-file source suite staged externally and remotely under
  `corrected-trl-r129`; launch only after R128 is authoritatively terminal and
  GPU idle. Same informative R124 source, combined four branches with complete
  resume under each new source identity. No source or checkpoint identity rewrite
  is applied to existing R128 artifacts.
- [x] Revision129 R128 is confirmed terminal exit1. All4 execution/verifier/
  independent-math stages pass; all4 resumes reject the same sampler provenance
  comparison before training. CPU comparator shows exact0 adapter difference
  between BF16 records1/records2 at both boundaries. Suite/readback receipts
  preserved outsideGit; total suite remains failed because resume is required.
- [ ] Revision129 after R128 terminal confirmation and GPU idle, launch the
  corrected post14 suite: `posttrain-r129-corrected-provenance-post14-suite`,
  container `c14e46881ccffd732eaa7e3d7ef8a2f32e30595f85bd21df89b3d44f63faf2af`,
  `/var/lib/posttrain/qualifications/corrected-trl-20261002-r129`.
  CPU preflight accepts; all5 post14 fork regressions pass in5.77s before launch.
  Native first BF16 branch is running; observe this handle and per-stage/full
  checkpoint comparator before accepting strengthened-identity recovery.
- [x] Revision129 first BF16 records2/budget6 branch now passes native2-update
  verification, independent math, continuation execution and exact full-state
  comparator under the strengthened source identity. Remaining3 branches of
  the same live suite still require results before accepting its overall gate.
- [x] Revision129 complete repository Ruff passes. Full Pyright initially finds
  14 errors in owned test fixtures (optional narrowing, intentional negative
  immutable-map mutation, stub declarations/casts); corrected without changing
  production semantics. Full Pyright now reports721 files/0 errors/0 warnings.
  All9 import contracts and both-repository diff checks pass. Native veRL cohort
  runs against clean3cf5e16db7591d62f611ff4ed37b39ab574d1bc7 checkout:57 pass.
  Initial incomplete-PYTHONPATH attempt had73 pass/33 skip/3 missing-veRL failures;
  explicitly adding `/home/hammad/projects/verl-posttrain-parity` resolves the
  native dependency, rather than suppressing those tests. This is CPU native
  buffer/driver/capability evidence, not distributed Ray/GPU qualification.
- [x] Revision129 R128 FP16 replay also passes2 informative applied updates and
  all independent objective/gradient/clipping checks. All32 adapters change;
  native gradient norms .0569926/.0375527; scaler128 remains unchanged with
  growth trackers1/2 and attempts matching applied counters (no skipped update).
  Both BF16 packing branches match observed gradient norms .0697015/.0311401.
  Their exact saved-weight packing comparator subsequently passes at both boundaries.
- [x] Revision129 selects unpublished fork candidate version1.12.0.post14 after
  confirming remote releases end at post13. Rebuilds and rehashes the staged
  R129 source to qualify its final candidate version with the provenance fix.
  Origin/upstream remain carbonteq-ai/trl and huggingface/trl; no release,
  distribution publication, stable promotion or consumer pin adoption yet.

- [x] Revision128 diagnoses R127 external replay harness omission: beta.01
  enabled but replacement collector did not call the native reference provider.
  Framework rejects missing reference before optimizer mutation. Own invalid
  suite stopped; container1d039b918bb7882f3e3f32054a004f0e2cfb507c1074da58395cc762d7e259ec
  confirmed terminal137 and GPU idle. Failure artifacts preserved, no production
  relaxation. External R128 collector now calls job._reference_scores exactly
  as the ordinary native collector does. Same sealed framework/fork archives.
- [ ] R128 replaces R127 as the live combined suite, remote
  `/var/lib/posttrain/qualifications/corrected-trl-20261002-r128`, name
  `posttrain-r128-informative-precision-packing-resume`, container
  `1170d8d0e960f29d4b05393ccc6816d8af9939bb86200d05f458f7eea0ecf44e`. Four same-evidence
  2.6B branches and full checkpoint1-to2 continuation per branch, as described
  below. Qualifier scripts rehashed before staging; retained R124 source remains
  read-only. Observe this handle; do not restart on a polling timeout.
- [x] Revision128 executes all5 focused fork regressions with pytest in the
  exact qualification image (generation normalization BF16/FP16/FP32, continuous
  batching rejection, mixed root/reference checkpoint layout):5 pass in6.18s.
  Offline pytest modules copied from the local development runtime, no dependency
  or source changes to the GPU suite. CPU test container has no GPU access.

- [x] Revision127 stages authenticated current362-file framework source and the
  repaired TRL fork, including inherited-source recovery identity. CPU preflight
  accepts replay settings before GPU launch. Raw tools/manifests remain outside
  Git in `working/corrected-trl-r127` under the external correctness directory.
- [ ] R127 historical combined native replay/continuation attempt: container
  `1d039b918bb7882f3e3f32054a004f0e2cfb507c1074da58395cc762d7e259ec`,
  `posttrain-r127-informative-precision-packing-resume`, remote
  `/var/lib/posttrain/qualifications/corrected-trl-20261002-r127`.
  Source R124 checkpoint1 is mounted read-only. Four branches reuse its exact12
  episodes/8340 actions/credit: BF16 records2/budget6, BF16 records1/budget6,
  FP16 records2/budget6, BF16 records2/budget12/epochs2. All branches run2
  updates with beta.01 and exact-model2.6B654f9463ce32b05d0429d76fe1f580b27d4c1ac0.
  Each then resumes checkpoint1 to2 under its own strengthened identity; compare
  policy/reference adapters, optimizer/scheduler/RNG/scaler, frozen scores,
  native evidence/credit/correction. One coordinated job, no fresh collection.
  GPU idle verified immediately before launch. Observe stage receipts and final
  suite outcome before closing any gate; packing parity is within BF16 only.

- [x] Revision126 identifies R125's loader defect: native Transformers5.14.1
  loads `ref/` and skips PEFT's trainable root `default` adapter. Resumed
  update2 ratios are all1/KL0, unlike uninterrupted R124. Same CPU native
  checkpoint-layout regression passes with the TRL supplement and exercises
  exact saved policy/reference weights and trainability. Generic fix lives in
  sibling `trl-engine-generation/trl/trainer/base_trainer.py`; no publication
  or dependency-pin update yet.
- [x] Revision126 strengthens `policy_job.job_identity` to hash inherited
  native trainer source as well as the GRPO file. Thirteen focused composition
  tests pass, including changed inherited loader rejection; Ruff passes.
- [x] R126 diagnostic continuation uses sealed R124 framework source plus the
  repaired fork, preserving the historical runtime identity for this bounded
  comparison only. Container
  `4cb35a85f6f4dccbe7306dc9fcdd05da02dfe6a737adbaa8b63605a52e64a13e`,
  `/var/lib/posttrain/qualifications/corrected-trl-20261002-r126`;
  external local `working/corrected-trl-r126`. GPU idle verified before launch.
  No fresh collection; checkpoint1 to2. Terminal exit0; native update gradient
  norm returns to .03114. Exact comparator passes132 tensors (policy weights,
  optimizer, scheduler, RNG and frozen scores); population/credit/native evidence
  and correction match exactly. External native-resume-comparison.json retained.
  Separate authenticated CPU comparison also passes32 reference-adapter tensors
  and the full independent observed-score measurement for update2, including2
  clipped actions; reference-objective-resume-comparison.json retained outsideGit.
  Final recovery qualification requires a new matched pair under the stronger
  inherited-source identity; R126 alone cannot establish that release gate.

- [x] Revision125 repository-noise audit: generated SAMPO JSON, renders and
  scratch Python tools account for 759,994 of 780,836 untracked lines. Added
  scoped `.gitignore` entries plus local Claude configuration and release image
  receipts. Visible status entries fall from193 to140; untracked lines fall to
  20,734. Authored Markdown, engine modules and maintained tests remain visible.
  No files deleted or removed from the index; no tracked file matches ignore rules.
- [ ] R125 continuation container
  `bc59a7415c2490e5ab1affe1a155471597250aa610ee427f07a1015dd1b69a13`
  is terminal exit1: native continuation ran, but exact saved-adapter comparison
  failed at layer13 q_proj lora_A. Resume qualification remains open; preserve
  both checkpoints and investigate before accepting this gate.

- [x] Revision124: matched LFM2.5-2.6B native BF16 job applies two informative
  updates over12 fresh episodes/8340 actions with beta.01. Trello group rewards
  [0,1,0,0]; Sheets and Airtable groups remain all0. Independent observed-score
  references match both policy/KL values and score derivatives (max1.61e-11)
  and clipping decisions. Update2 ratios range0.7349524-1.5336192 with2 actively
  clipped actions out of3654. Nonzero sampled KL0.0003947754, weighted3.947754e-6.
- [x] Revision124: original external verifier rejects its hardcoded24-adapter
  assertion after native training succeeds; container exit1 reflects that check.
  Separate CPU verify_generic.py passes:17 sealed components at both boundaries,
 8340 correction weights independently exact, credit/old scores/correction
  frozen,96 finite optimizer tensors, both native gradient norms nonzero
  (.0697015/.0311401), all32 architecture-correct adapters change. No training
  rerun or relaxation of objective/gradient/informative checks was needed.
- [ ] Revision124: after GPU idle confirmation launch full native checkpoint1
  continuation with no fresh collection: posttrain-r125-native-continuation-no-rollouts,
  containerbc59a7415c2490e5ab1affe1a155471597250aa610ee427f07a1015dd1b69a13,
  /var/lib/posttrain/qualifications/corrected-trl-20261002-r125. Same sealed
  R124 framework/fork source and task/model/precision settings; source mounts
  read-only. Exact comparison covers model/optimizer/scheduler/RNG, frozen
  old/reference scores, native evidence, population/credit and correction.
  Observe terminal state and comparator before claiming recovery closure.

- [x] Revision123 (terminal): CPU preflight of selected three-task profile passes. Current
  task user prompts/assertions equal archived definitions and tools preserve
  the archived set. GPU idle before launch; R123 fresh12-episode BF16 diagnostic
  runs under sealed R117 source, beta.01, reply2048/context8192, episode budget6.
  Container71d9fb8d86451ae9705fbd5b5fd91b2bd7f1888e21b894703c252a2900a85b03,
  name posttrain-r123-selected-three-task-population. Native job completes two
  boundaries, then qualifier exits1 because all12 rewards/advantages are zero.
  Independent reference authenticates17 components and30348 action corrections
  exactly at both checkpoints; no adapters change. Retain negative evidence.
- [x] Revision123: resolved TRL generator now uses the existing Verifiers
  train-client named-call admission, matching native veRL; legacy generator
  strict behavior remains. Regression covers nonconforming executable calls and
  exact raw evidence. Focused generator/message cohort29 passed, Ruff passed,
  production generator pyright0 errors. This preserves existing native training
  semantics; no frozen product baseline change or fork/pin update is required.
- [x] Revision123: external independent score-derivative hook tested on clipped
  token-GRPO with nonzero KL: policy difference1.50e-8, KL7.39e-12, maximum
  gradient difference4.69e-9; two clipped actions match. Native graph remains
  differentiable. This fixture check is not an observed-model qualification.
- [x] Revision123: R124 source/qualifier staged separately with362 framework
  hashes, exact dedicated TRL candidate base d97a2cf, corrected admission and
  historical LFM2.5-2.6B revision654f9463ce32b05d0429d76fe1f580b27d4c1ac0.
  CPU preflight passes. Qualifier-manifest.json now hashes external scripts,
  profile and source manifest before GPU launch. Wait for R123
  authoritative terminal state and inspect its actual native traces; do not
  overlap GPU attempts. R124 adds independent objective/gradient measurement
  to the same fresh training pass. Raw qualifier files remain outside Git.
- [x] Revision123: R123 terminal second boundary and failed informative gate
  inspected before R124; first step includes404.1 seconds fresh collection.
- [ ] Revision123: after direct GPU idle check launch
  posttrain-r124-matched-model-combined-qualification under
  /var/lib/posttrain/qualifications/corrected-trl-20261002-r124, container
  e7d7b67a356d51492ad30a320463ac9feb2cc645a3c08ebdec41faf6cd141914.
  Last inspection is running. Observe that exact
  handle. Matched2.6B model, corrected named-call admission/finish reasons,
  three groups4, two updates, beta.01, independent values/score derivatives,
  tool/context masks, checkpoint/correction/gradient/finite optimizer checks
  belong to this one job. Qualifier scripts are hashed before execution.
- [ ] Revision123: external verify_generic.py removes the earlier1.2B-only
  hardcoded24-adapter assumption by authenticating rank and A/B pairs against
  adapter_config.json. It is a separately staged CPU verifier, not a mutation
  of the sealed live qualifier. Use on retained2.6B artifacts if the original
  verifier rejects that architecture-specific assertion; no fresh training is
  required to repair this measurement. compare_native_resume.py is staged for
  exact authenticated model/optimizer/scheduler/RNG/old/reference/correction
  and population comparison after native continuation; not yet run.

- [x] Revision122: inspect archived training exports in
  .posttrain/state/analysis/sampo-multirun, exclude duplicate message exports,
  group by native group_id, retain source hashes/trace IDs outside Git. Select
  simple.trello_manager_feedback_from_email (22 mixed training groups),
  simple.sheets_log_sales_call (17) and simple.email_airtable_lead (6).
  Exact example groups/rewards/tool counts/token lengths are in the external
  selected-task-profile.json and historical-training-task-groups.json under
  /home/hammad/experiments/posttrain-correctness/2026-10-01/working/corrected-trl-r117.
  Manually inspect assertion and turn-reward summaries plus native task definitions.
- [ ] Revision122: qualify current task-definition/runtime compatibility, then
  collect three complete groups of four in one fresh population. Retain it for
  all compatible diagnostics/comparisons. Historical spread is not guaranteed
  for the current policy; do not replace measured rewards or weaken admission.

- [x] Revision121: consolidated retained-evidence TRL job exits0, container
  ef2f16850594064e1ca972515e045189915dce2bed911af3d4fcd162524cd24d,
  posttrain-r121-retained-evidence-suite. Four stages each apply two informative
  updates: BF16 records2/budget2, BF16 records1/budget2, FP16 records2/budget2,
  BF16 records2/budget4 with two epochs. All reuse authenticated R111 population
  with4476 actions and unchanged original contexts/actions/prepared credit.
  All stage correction/checkpoint verifiers pass. CPU saved-state comparator
  proves identical initial adapters within precision and zero adapter difference
  between BF16 pack layouts at both boundaries. Raw r121-suite.json and
  r121-controlled-comparison.json remain under corrected-trl-20261002-r117,
  outside Git. This is replay execution, not fresh TRL collection or recovery.
- [ ] Revision121: select next fresh tasks using historical training traces and
  explicit mixed-reward group IDs, tool activity and token budgets, following the
  user's direction. Do not infer suitability from a task's assertion count alone.

- [x] Revision120: R119 exits0 but all four calls truncate at1024, reward0,
  no tools or errors. Independent verification authenticates15 components and
  4096 exact action corrections; all24 adapters remain unchanged. Informative
  update gate rejects. R119 is terminal, not live.
- [x] Revision120: private SAMPO driver reuses native ActiveSamplingReplayBuffer
  with one validated task reservation and native reserved-round dispatch. Reject
  config mismatches and duplicate candidate task identities before generation.
  Driver/runtime/runner/inventory cohort:42 passed; production driver pyright:
  zero errors. The round/eviction test uses stub transport, not real Ray/TQ/GPU.
- [ ] Revision120: consolidate remaining qualification into the retained-evidence
  suite below. External fixed-evidence TRL prototype is staged but unlaunched;
  preserve original action/context/credit provenance. Stop separate fresh rollout
  attempts for individual diagnostics. Native SAMPO integration and public gates
  remain open; no dependency pins or public admission guards changed.

- [x] Revision119: R118 reply768 exits0, all four native traces finish length with
  reward0 and no tool/error records. Independent verifier authenticates15 components,
 3072 frozen action corrections exactly; all24 adapters unchanged and zero prepared
  advantages. Informative-update gate rejects as intended. R117 success control
  separately authenticates3468 exact frozen corrections. Neither closes informative
  native-update qualification; both retained raw receipts remain outside Git.
- [ ] Revision119: after direct GPU idle checks launch native multi-assertion
  task simple.buffer_webinar_dual_post using external check_multi_assertion.py,
  posttrain-r119-corrected-ordinary-trl-bf16-multi-assertion, container
  75d490501f833e5248b1e4feabe1ea343a3d7430ceb6bcbb74d89c7ef7d7b655.
  Same hashed R117 source, group4/episode budget2/records2, reply1024 and
  prompt3072 within loop4096. Task has two native action assertions; no scorer
  or reward changes. Environment identity names the task. Inspect actual traces,
  run independent verifier and qualify continuation only if informative gate passes.
- [x] Revision118: corrected TRL BF16 R117 exits0 with two native optimizer steps,
  but independent verifier rejects informative-update gate: all four actual
  AutomationBench episodes score1, no errors, eight calls finish tool_calls/stop,
  prepared advantages and gradient norms0, all24 adapters unchanged. Manual native
  traces explain this expected GRPO constant-reward behavior. Retain as lifecycle/
  correction evidence only; do not claim productive or informative qualification.
- [x] Revision118 (terminal, informative gate failed by Revision119): after direct GPU idle checks launch separate reply768 profile
  posttrain-r118-corrected-ordinary-trl-bf16-bounded, container
  9700e6a218f029137340efe4cf34f46dfb15d9a90efdc4390244dc0060174183.
  Same R117362-file framework and dedicated TRL candidate; new run bf16-r118-bounded.
  Native benchmark rewards stay unchanged. Follow exact handle, inspect actual
  completed/truncated groups and independent verifier before informative update/
  resume claim. This bounded profile is correctness evidence, not a recipe choice.
- [x] Revision117: corrected FP16 R115 continuation exits0. Independent CPU
  comparator authenticates boundary2 checkpoints and matches all248 saved
  tensors, model/optimizer/scheduler/RNG/scaler, native data.pt, frozen old scores,
  population and correction exactly (maxdiff0, applied=attempts=next-update2).
- [x] Revision117 (terminal, informative gate failed by Revision118): after direct GPU/container idle checks launch corrected TRL
  BF16 posttrain-r117-corrected-ordinary-trl-bf16, container
  10d782d22eadd5a1f187e0109b0482b17cfb14d0283d00548baa148e52988642.
  Source R117362 framework files plus hashed dedicated unpublished TRL candidate
  ond97a2cf, ordinary HF sampler/group4, episode budget2, execution records2,
  reply1024, two maximum applied updates. Follow exact handle under
  corrected-trl-20261002-r117; verify real native corrections/updates before
  checkpoint1 continuation and FP16. External verify.py staged; no completion
  claim yet. R116 remains rejected audit record.
- [x] Revision116: reject staged TRL R116 before GPU launch: siblingc9af78c1 is
  not the plan's HF sampled-score candidate. Retain its CPU preflight as admission
  evidence only, not native capability. Stage current362 framework files plus
  /home/hammad/projects/trl-engine-generation based ond97a2cf, including dirty
  candidate delta and new fork test, with archive and per-file hashes in R117.
  CPU preflight exits0, installed metadata1.12.0.post13, verifies native
  return_generation_logprobs signature and ordinary host admission. Unpublished
  candidate publication/adoption and actual corrected TRL updates remain open.
- [x] Revision115: reject resolved veRL inventories with missing/duplicate task
  identities or fewer distinct tasks than prompt groups before native dataset
  writing. Preserve legacy cycling for unselected policy_updates. Six focused
  inventory/legacy/curriculum checks, scoped Ruff/Pyright,9 import contracts and
  diff checks pass. This local edit is excluded from both staged R111 and R116.
- [x] Revision115: R114 corrected FP16 exits0. Independent CPU verification
  authenticates16 components/publications,4750 frozen action corrections exactly
  against Torch FP64 reference, unchanged credit/population and all24 changing
  adapters. Physical packs[2,1], costs4274/2137; saved scaler scale128,
  growth_tracker1/2, applied=attempts1/2, skipped0. Objective clipping0 both steps.
- [x] Revision115 (closed by Revision117): after direct GPU/container idle checks launch corrected FP16
  checkpoint1 continuation posttrain-r115-corrected-ordinary-fp16-resume, container
  865b9ae6965789aa8e17ba0bbe678cc896a0cf6855602ca60b91d05fa17efcb0.
  Same1592-file R111 tree; follow fp16-r115-resume and compare complete native
  state including scaler/correction using compare_corrected_resume_r115.py after
  terminal. Do not restart based on observation timeout.
- [x] Revision114: corrected BF16 R113 continuation exits0. Independent CPU
  comparator authenticates both boundary2 checkpoints and compares all248 saved
  tensors, optimizer/scheduler/RNG, native data.pt, frozen scores, resolved
  population and correction JSON exactly (maxdiff0, applied=attempts=next-update2).
  R111 general verification also confirms all24 adapters change and physical
  packs[2,1] at both updates, including real partial tails.
  Current local correction/job/overflow/recovery/veRL driver cohort76 passes;
  this includes the shared recipe defaults but does not qualify GPU TRL wiring.
- [x] Revision114 (rejected lineage by Revision116): stage separate current corrected TRL source362 files and clean
  forkc9af78c1 under corrected-trl-20261002-r116; archives and framework hashes
  verified before isolated install. CPU ordinary-job preflight accepts BF16,
  episode schedule budget2 and execution records2 with native environment factory.
  Candidate fork metadata is1.8.0 versus image1.12.0.post11 and selected
  dependency1.12.0.post13; this preflight does not resolve release identity or
  qualify runtime execution. No GPU launch while R114 remains live.
- [x] Revision114 (closed by Revision115): after direct GPU idle checks launch corrected FP16 actor/scaler
  posttrain-r114-corrected-ordinary-fp16, container
  2477336ba59920278998d9b52851e2c33656c2909b7604865d4120adba5e6e24.
  Same1592-file R111 tree; follow fp16-r114-corrected under the R111 qualification
  root. Authenticate applied updates, correction, scaler and checkpoints before
  checkpoint1 continuation. Rollout inference remains effective BF16.
- [x] Revision113: corrected BF16 R111 exits0 with2 applied updates. Independent
  CPU reference authenticates16 components and matching publications at both
  boundaries, compares4476 actual action correction weights with Torch FP64
  episode exp(sum(old-sampler))/clamp(.1,3) exactly (maxdiff0), and verifies weights/
  old scores remain unchanged. Episode weights.1/1.19586368/.34753443/1.08040997;
  update2 clips3 of2735 selected actions. These are correctness measurements,
  not recommendation of sequence correction over SAMPO's token correction.
- [x] Revision113 (closed by Revision114): after direct GPU/container idle checks launch corrected
  checkpoint1 continuation posttrain-r113-corrected-ordinary-bf16-resume, container
  be17c982cbff6ad8491c7018ccb034ff855c40141238800e511b0da34b089c94.
  Same1592-file staged R111 source and publishedc45392d2; current local shared
  recipe/TRL wiring changes are excluded. Follow this handle and compare full
  native model/optimizer/scheduler/RNG/data/population/correction at boundary2
  using external compare_corrected_resume_r113.py after terminal.
- [x] Revision112: share recipe_sampler_correction_weights between TRL and veRL.
  TRL's supported ordinary collector captures authenticated native sampled scores,
  prepares correction after old-score freezing before first loss, retains the map
  through resume and rejects missing correction for the selected recipe. Model
  export excludes the correction JSON.59 related native/overflow/job/recovery/
  veRL tests pass;18 pure correction references include GRPO/SAMPO/GDPO/CAPO
  defaults. Scoped Ruff/Pyright,9 import contracts and diff checks pass.
  TRL's vLLM/filtering/mask production guards remain; no native TRL corrected-job
  or cross-backend GPU parity claim is made. R111 staged source is unchanged.
- [x] Revision112 (closed by Revision113): R111 corrected BF16 remains live after saving two native
  checkpoints. Independent verify_corrected_ordinary_r111.py is staged externally
  and compares actual native sampler/frozen-old episode-product correction against
  Torch FP64 exp(sum)/clamp(.1,3), authenticates both published boundaries and
  frozen weights/scores. Wait for authoritative terminal state, run it on CPU,
  then resume checkpoint1 on this identical1592-file tree.
- [x] Revision111: NativePopulationInputs captures compact native sampled scores
  on admitted assistant nodes and exposes sampling_log_scores with original
  coordinates, finite/alignment checks and mutation detection. Ordinary veRL
  collection selects recipe correction; a one-use preparation callback runs after
  complete old-score freezing inside first update, before objective evaluation.
  Complete detached weights persist through updates/retries/checkpoints.90 input/
  admission/native/recovery/runtime tests, scoped Ruff/Pyright,9 import contracts
  and diff checks pass. A fresh population still starts without reused old scores.
- [x] Revision111 (BF16 closed by Revision114): stage362 framework files plus publishedc45392d2, verify1592
  hashes including native source marker. After direct GPU/container idle checks
  launch posttrain-r111-corrected-ordinary-bf16, container
  db4988fda7d04a23b9c6f8b6c45609f0f546d125e63e226dc9e47b6cfbdac46c.
  Fresh group4 episodes,2 episode minibatches,records2, selected GRPO sequence
  correction bounds.1/3. Follow this handle under full-verl-20261002-r111;
  authenticate correction values from native sampler/frozen old scores, seals,
  updates and resume before FP16. Default SAMPO active collection stays guarded.
- [x] Revision110: R108 ordinary dense FP16 checkpoint1 continuation exits0.
  Independent CPU comparison authenticates boundary2 checkpoints and matches
  all248 saved tensors plus optimizer/scheduler/RNG/scaler/native data and
  resolved population exactly (maxdiff0, applied=attempts=next-update2).
  Ordinary BF16/FP16 recovery gates close for this single-rank LFM profile;
  correction changes were not part of the immutable run source.
- [x] Revision110: broader recovery/native regression cohort83 passes; focused
  Ruff/Pyright and9 import contracts pass. Tighten pure sampler helper to plain
  finite numerical log scores/bounds so tensors or strings cannot accidentally
  enter a declared detached calculation. No release/pin adoption is claimed.
- [x] Revision110 (wiring closed by Revision111): next wire fresh correction from authenticated native sampled
  log probabilities and population-frozen initial old scores before the first
  objective. Retain/restore its weights through the new sealed component. Then
  stage new source and qualify corrected ordinary BF16/FP16/resume separately.
  Active SAMPO dispatch/observe/uniqueness and full remaining scope remain open.
- [x] Revision109: shared policy_update_recovery.py seals optional complete
  detached corrections in posttrain-sampler-correction.json with component role
  sampler-correction. load_sampler_correction verifies all components before
  parsing, exact support/digest/finiteness and returns immutable weights; legacy
  uncorrected identity/file layout remains unchanged. Native veRL session restore
  consumes the verified map instead of replacing it with None.22 focused sampler/
  score-recovery tests pass, including exact continued loss and gradients and
  tampered-file rejection; scoped Ruff/Pyright pass. Fresh correction production
  and native corrected-job qualification remain open; R108 tree is unchanged.
- [x] Revision108: R106 ordinary dense FP16 exits0. Independent CPU verifier
  authenticates15 components/publications at both checkpoints; unchanged
  population/credit/4420 old scores and all24 changing adapter tensors. Physical
  packs[2,2], token costs[5954,5954]. Saved FP16 scaler scale128 at both steps,
  growth trackers1/2, skipped0, applied=attempts=1/2. Fresh FP16 population differs
  from BF16; this is not an exact cross-dtype comparison.
- [x] Revision108 (closed by Revision110): after fresh GPU/container idle checks launch ordinary FP16
  checkpoint1 continuation posttrain-r108-ordinary-dense-fp16-resume, container
  20ef980d7cc1d76bb986b135e32aeb33c5d1a9937a6f54eb5bdbd6fc5cae2f85.
  Same sealed1591 files/publishedc45392d2, separate writable dataset cache.
  Follow this handle and compare boundary2 native model/optimizer/scaler/RNG/data
  and resolved population with compare_ordinary_resume_r108.py after terminal.
  Shared sampler helper remains outside this sealed qualification tree.
- [x] Revision107: audit native collection at published veRL sourcec45392d2
  (current3cf5e16d is ledger-only) and TRL checkoutc9af78c1. Native
  ActiveSamplingReplayBuffer owns round dispatch/observe/eviction; resolved driver
  currently replaces it and unconditionally pre-dispatches. Preserve the existing
  qualification guard until corrected integration, fresh uniqueness, filtering
  evidence and sampler correction are qualified. No framework error is attributed
  to the native implementations.
- [x] Revision107: add framework-neutral update_sampler_correction.py detached
  weights for explicit token/episode truncate or mask modes on complete original
  action support. Sequence correction is exp(sum(old-sampler)), separate from
  normalized PPO ratio/clipping. Selected bounds precede exp to avoid overflow
  without implicit caps; missing/nonfinite support rejects.14 independent reference
  tests, scoped Ruff/Pyright pass. This helper is not wired or native-qualified;
  current sealed R106 source is unchanged.
- [x] Revision107: external retained BF16 measurement over4427 actions finds
  mean absolute log gap.01391847, p99.14774896, max.38760614; token correction
  weights min.67867959, mean1.00055819, max1.36539913, none above2. The ordinary
  resolved host currently passes sampler_correction=None. Unwarped sampling does
  not establish identical numerical actor/sampler probabilities; do not claim
  this run qualifies the selected SAMPO correction.
- [x] Revision107 (closed by Revision108): R106 FP16 remains live after two reported applied updates
  with scale128 and skipped0 at both steps. Independently authenticate checkpoints,
  frozen scores/parameters and overflow/scaler state after terminal, then resume
  checkpoint1 on the same immutable tree. Native metric rows alone are insufficient.
- [x] Revision106: R105 ordinary dense BF16 checkpoint1 continuation exits0.
  Independent CPU comparison authenticates both boundary2 seals and finds exact
  equality in248 saved tensors across model, optimizer, extra_state (scheduler/
  RNG), old scores and native data.pt, plus exact resolved-population payload.
  Applied/attempt/next-update counters all2; maximum difference0. Raw comparison
  receipt stays outside Git under full-verl-r67 / remote R103 qualification root.
- [ ] Revision106: fresh direct GPU idle check (CPU verifier alone running,
 120926MiB available RAM), launch posttrain-r106-ordinary-dense-fp16, container
  6b3e3ac877d95afb64ed9b81cc80248866b1591ec1c1ff4dbf36eb68f9ebf73f.
  Same immutable source; selected FP16 with initial scale128 and separate writable
  dataset cache. Verify fresh rollout/overflow/applied updates and native checkpoint
  state before checkpoint1 continuation. BF16 success does not certify FP16.
- [x] Revision105: ordinary R103c BF16 exits0 with two applied updates. Independent
  verification authenticates15 checkpoint components at each boundary and matching
  publications, unchanged population/credit/4427 old scores, and changes in all24
  adapter tensors. Update2 clips2 of2767 selected actions. Native traces exclude
  system/user/tool nodes; assistant masks total4427. Three benchmark assertions
  pass; one fails and its final call reaches length1024. External R105 mask audit
  authenticates native evidence SHA and exactly matches all4427 action identities,
  token IDs/masks/logprobs/graph parents, conditioning refs and context lengths.
  Independent teacher-forced probability recomputation remains separate.
- [x] Revision105 (closed by Revision106): after direct GPU/container idle checks launch ordinary
  checkpoint1 continuation posttrain-r105-ordinary-dense-bf16-resume, container
  8be5dc24d47957f2fdf728a48fda08f43b030d97689edd354b636962dde55270.
  Same sealed1591-file source and publishedc45392d2; distinct writable dataset
  cache only. Observe this handle, compare complete native optimizer/model/RNG/
  scheduler and frozen population state at continued boundary2, then qualify FP16.

- [x] Revision104: R103b source/config checks pass, then exits1 before training
  because datasets tries to create its lock under the read-only /hf model-cache
  mount. No optimizer update is claimed. Keep weights read-only and select
  writable HF_DATASETS_CACHE=/qualification/runtime-cache/r103c/datasets.
- [x] Revision104 (closed by Revision105): after terminal R103b and direct GPU/container idle checks
  launch posttrain-r103c-ordinary-dense-bf16, container
  0cefbb99b8b5ecb74a91abb7675b1bc7500a1f3a2fabd258aac77e56d331f130.
  Preserve source361 framework files/publishedc45392d2/1591 sealed hashes;
  fresh dataset cache changes environment setup only. Observe this handle before
  any further GPU launch. Authenticate actual updates/checkpoints and continuation,
  then qualify ordinary FP16 and one-record comparison on retained evidence.

- [x] Revision103: implement policy_capabilities.py layout/profile admission,
  policy_native.py pre-dispatch/effective actor capability wiring, and optional
  validation in resolved_workers.py before native engine allocation. Explicit
  backend_options.resolved_context_layout selects dense-population; legacy ragged
  records1 retains its admission. worker.py emits the selected BF16 native mixed
  policy explicitly with FP32 reductions/buffers. All99 focused native transport,
  capability/runtime/factory/driver/runner/checkpoint tests pass; focused Ruff,
  Pyright,9 import-boundary contracts and diff checks pass.
- [x] Revision103: stage361 current framework files and publishedc45392d2;
  all1590 source hashes verified. Real request/config preflight accepts records2,
  episode schedule budget2 in BF16. Initial ordinary attempt exits1 before
  training because the archived fork lacks the supported source-revision marker.
  Verify all files and seal via .posttrain-source-revision;1591 hashes now bind
  the marker. Retain temporary genuine Git metadata outside the fork source.
- [x] Revision103: after terminal first attempt and fresh GPU/container idle
  checks launch posttrain-r103b-ordinary-dense-bf16 on the same maintained source,
  targeting fresh AutomationBench generation and two applied updates. This
  attempt is terminal before training; remaining qualification is Revision104.
  Both setup failures remain retained, never reported as training evidence.

- [x] Revision102: R100 FP16 exits0. Independent saved comparisons find all24
  gradients/adapters exactly equal across all3 layouts at both applied updates.
  This completes the maintained scoped BF16/FP16 retained actor packing gate.
- [x] Revision102: integrate explicit backend_options.resolved_context_layout
  into ordinary native capabilities. Preserve ragged records1 defaults; admit
  dense-population only under declared qualified numerical settings, rejecting
  undeclared multi-record or dense-pack selections. Validate resolved config
  before Ray and effective engine config before model construction through the
  existing worker factory seam. Wiring is complete; fresh ordinary job and
  checkpoint qualification remain open under Revision104.

- [x] Revision101: R100 maintained BF16 exits0 without external torch backend
  overrides or input-padding wrappers. Independent comparisons find all24
  gradients/adapter tensors exactly equal across all3 layouts at both applied
  updates. Each layout restores172 named parameters and4482 old scores;
  continuation exactly matches gradients, parameters, optimizer, scheduler,
  loss and counters. Update2 clips one action in each layout.
- [x] Revision101: after terminal BF16 and direct GPU/container idle checks,
  launch posttrain-r100-native-arithmetic-resume-fp16 on the same1589-hash
  tree and publishedc45392d2; container
  9639e6dec2dc3a93812d4ca48239d3c3ba39cdbb530152baef41a28994fdd545.
  Observe this handle, independently compare tensors, then qualify ordinary
  multi-record admission through the native driver. Do not relaunch on timeout.
- [x] Revision101: register publishedc45392d2's inherited objective-name source
  compatibility after verifying7cf68728 ancestry. This does not admit previously
  unqualified execution profiles. The current frozen R100 source remains unchanged;
  include this worker mapping in the next ordinary-job source snapshot.

- [x] Revision100: R98 FP16 exits0; independent saved comparisons find all24
  gradient/adapter tensors exactly equal across all three layouts at both applied
  updates. All three resume exactly, restoring172 parameters and4482 old scores,
  optimizer/scheduler/scaler/loss/counters. Scale128, no skipped steps. Together
  with BF16 this closes the retained physical-budget/v3 recovery gate.
- [x] Revision100: implement native scoped full_precision_matmul and math_sdpa;
 14 new CPU cases and46 focused native tests pass, including independent/default
  controls, actual backward hooks, nesting, concurrent cooperating scopes and
  entry/body/cleanup failures. Ruff/diff pass. Publish source
  c45392d22df0c1ab2258e9c0675d3a03093094af to
  origin/codex/resolved-engine-worker, including the fork ledger. Consumer
  page updated; immutable pins unchanged. No raw correctness tools committed.
- [x] Revision100: add two framework recovery regressions binding each scoped
  native configuration field to runtime identity. All16 native runtime tests,
  focused Ruff/Pyright,9 import-boundary contracts and diff checks pass. The
  initial test mutation was corrected to dataclasses.replace because native
  settings are frozen; no production field mutability was introduced.
- [x] Revision100: stage1589 verified files at
  /var/lib/posttrain/qualifications/full-verl-20261002-r100; CPU preflight passes
  all3 layouts. After direct idle checks launch
  posttrain-r100-native-arithmetic-resume-bf16, container
  63601f3ff25cd0ec189e43f36517a4bfd90326b8fcdcb5c8905553aa3aafb6e3.
  This probe removes every external torch backend override and uses published
  native settings. Observe terminal state, independently compare saved tensors,
  then qualify FP16. Ordinary driver admission remains a separate later gate.

- [x] Revision99: R98 BF16 exits0. Independent saved comparisons across
  records1/8192, records2/8192 and records2/4000 find zero differing gradient
  or adapter tensors at both applied updates. All three layouts restore172
  named parameters and4482 frozen old scores; update2 resumes exactly.
  Physical costs are2079 per row and4158 for a two-row pack. The maintained
  padding adapter and strictv3 recovery run without an external padding wrapper.
- [x] Revision99: after direct GPU/container idle verification, launch
  posttrain-r98-native-layout-resume-fp16, container
  a5087e8cf8750cc944e1a990b33bd41f8544ba2ae4fd3f7807429b9d09d40217.
  Observe this handle, then compare saved gradients/parameters across all three
  layouts. Do not relaunch based on an observation timeout.
- [x] Revision99: replace the probe's external global arithmetic setup with
  independently default-off native FSDP settings for full-precision GEMM
  accumulation and math SDPA. Scope settings over native train/eval contexts,
  including backward; restore caller state on exceptions and nested contexts.
  Qualify the maintained path before ordinary job admission. Frozen product
  meaning and objective contracts remain unchanged.

- [x] Revision98: implement physical context_layout costs in update_plan.py;
  v3 strict transport with boundedv1/v2 ragged reconstruction in
  update_transport.py; preserve legacy ragged execution digests and compare
  canonical decoded records in policy_update_recovery.py. Native veRL resolves
  matching pack sizes and padding through its engine input adapter, checking
  physical budget before large population padding and before model forward.
  All98 focused planner/transport/recovery/native runtime/driver/runner tests,
  focused Pyright/Ruff,9 import-boundary contracts and diff checks pass.
- [x] Revision98: stage360 current framework files plus publishedc334d02a,
  verifying1587 total hashes and fork archive digest. CPU native preflight on
  the actual population passes all3 layouts: records1/8192→[1,1,1],
  records2/8192→[2,1], records2/4000→[1,1,1] at both optimizer updates.
- [x] Revision98: after direct GPU/container idle checks launch
  posttrain-r98-native-layout-resume-bf16. Use maintained physical-cost/padding
  code without the external prepare_model_inputs wrapper, with high/low budgets
  and strictv3 checkpoint recovery. Observe this handle before launchingFP16.

- [x] Revision97: implement declared context_layout on ExecutionCapabilities:
  ragged (sum real context lengths), dense-pack (rows times pack maximum) and
  dense-population (rows times maximum sampled context in the frozen population).
  plan_packs and native veRL transport must use the same physical cost. Native
  population padding moves into the resolved engine input adapter with pre-forward
  budget guards. Preserve original conditioning/action positions and update digests.
  Write strict population transportv3; readv1/v2 only as the existing ragged
  contract for one release, preserving legacy recovery execution digests.

- [x] Revision96: R94 BF16 exits0; independent saved tensor comparison finds
  all24 gradients/adapter parameters exactly equal across both layouts at both
  applied updates. Boundary1 restores172 model parameters and4482 old scores;
  resumed update2 exactly matches gradient, parameter, optimizer, scheduler,
  scaler (absent inBF16), loss and counters. Update2 has one clipped action in
  both layouts. Both R94 handles are terminal; no GPU qualification remains live.
- [x] Revision96: bind GEMM accumulation modes and all activeSDPA backend/
  math-reduction switches into native_job_identity. Eight new sensitivity
  regressions plus the existing identity case pass; all14 native runtime tests,
  focused Pyright/Ruff,9 import-boundary contracts and diff checks pass. R94 uses
  its immutable staged source; this current identity change is separately CPU
  validated and must be included in the next ordinary-job runtime qualification.

- [x] Revision95: maintained R94 FP16 exits0. Both layouts apply2 native
  updates with scale128/no skips. Independent saved tensor comparison finds
  all24 gradients and adapter parameters exactly equal across layouts at both
  updates. Boundary1 recovery restores172 named parameters and4482 old scores;
  resumed update2 exactly matches gradients/adapters, optimizer, scheduler,
  scaler, loss and counters in each layout. Sourcec334d02a,1587 verified files.
- [x] Revision95: after R94 FP16 terminal and GPU/container idle verification,
  launch posttrain-r94-native-rowwise-resume-bf16 on the same immutable inputs
  for the BF16 applied-update/recovery counterpart. Observe existing handle.
  This is the retained native actor path; ordinary driver admission, physical
  padding budget accounting, other families and distributed gates remain open.

- [x] Revision94 continuation: rowwise sourcec334d02aa7950b46ee7544c53edb43d2b401a953
  committed/pushed with maintained helpers,13 CPU regressions and ledger.
  Consumer pins are unchanged. Separate R94 source staging verifies1587 file
  hashes and exact fork archive digest; raw probes stay external.
- [x] Revision94 continuation: after direct idle verification launch
  posttrain-r94-native-rowwise-resume-fp16. Use only published native precision,
  layout and rowwise settings. For each layout apply2 updates, save complete
  native/population boundary1, restore all parameters/optimizer/frozen scores,
  and replay update2 comparing gradients, adapters, optimizer, scheduler,
  scaler, loss and counters exactly. Observe handle before launchingBF16.

- [x] Revision94: R93 FP16 exits0. Independent saved tensor comparison finds
  all24 adapter parameters and gradients exactly equal across records1/2 at
  both actual applied updates. Initial score drift0, scale128, no skips, native
  scheduler/applied counters2. This closes the external control's two-step
  discrepancy for this exact retained single-rankLFM profile, not all families.
- [x] Revision94: rowwise native candidate passes40 focused CPU regressions
  and Ruff/diff. Next qualify its maintained source with applied updates and
  native model/optimizer/scaler plus population recovery, in both precisions.

- [x] Revision93: R92 FP16 rowwise adapter backward exits0. All24 gradient
  tensors exactly match across both layouts; same-layout repeats are exact.
  This supports the specific FP32 adapter GEMM shape hypothesis at the shared
  updated boundary. Native candidate adds independently default-off
  lora_rowwise_compute, requiringFP32 LoRA, with13 focused CPU cases passing.
- [x] Revision93: after R92 terminal and direct GPU/container idle checks,
  launch posttrain-r93-rowwise-packs-fp16 for two actual native applied updates
  per layout from a common restored checkpoint. Retain handle. Independently
  compare saved gradients, parameter updates, scores, losses and counters.
  Maintained GPU configuration and exact checkpoint resume remain later gates.

- [x] Revision92: R90 FP16 block14 trace exits0. Real-context activations
  match except tinyFP32 v_proj.lora_B outputs (maximum1.455192e-11) erased
  by downstreamFP16 casting. lora_B cotangents match; q/v lora_A cotangents
  differ inFP32 by at most2.619345e-10/5.820766e-11. Operator-normalization
  cotangents differ at4 FP16 entries. This identifies adapter input-backward
  arithmetic as a remaining candidate, not an objective/mask change.
- [x] Revision92: stage publishedc62c4468 in a separate source tree with1587
  verified file hashes and fork archive digest. R91 native BF16 controls exit0,
  with24 FP32 leaves/117 hooks and packing comparison exactly matching R86.
  No external class mutation or MixedPrecision patch remains in that run.
- [x] Revision92: after R91 terminal and direct idle verification, launch
  posttrain-r92-rowwise-adapter-fp16. External-only prototype evaluates each
  conditioning row'sFP32 adapter Linear independently to stabilize GEMM shape.
  Native backbone and scheduler/reduction ownership remain unchanged. Observe
  handle; do not adopt before evidence and maintained tests/qualification.

- [x] Revision90: fork sourcec62c4468635ac52e70598c1ae0c546676c203347
  committed and pushed to origin/codex/resolved-engine-worker. Only maintained
  helpers/config/integration, regression tests and ledger are committed; raw
  probes remain external. Consumer immutable pins are unchanged.
- [x] Revision90: R88 shared FP16 layer trace exits0. Every original-context
  forward activation is identical. Layer14 output cotangents are identical;
  layer13 output differs in one FP16 entry (maximum9.536743e-7), with differences
  spreading through lower layers. Same-layout repeats are exact throughout.
- [x] Revision90: R88 terminal and GPU/container idle verified. Launch
  posttrain-r90-shared-block-fp16 to trace block14 Linear, attention, MLP and
  normalization outputs/cotangents at this same shared updated state. Retain
  handle; native maintained-control GPU qualification follows separately.

- [x] Revision88: R86 FP16 shared post-update backward exits0, scale128,
  unchanged parameters and exact repeats. Packing gradient relativeL2
  7.467330e-4 (0.0746733%), maximumabsolute6.237067e-6. BF16 counterpart
  3.562957e-7 is much smaller. The FP16 residual contradicts attributing all
  R84 continuation differences solely to parameter perturbation.
- [x] Revision88: implement default-off native Linear arithmetic controls in
  the maintained veRL fork, preserving initialized Parameter identity, native
  synchronization and checkpoint keys. Ten focused CPU cases and 37 total
  precision/microbatch/determinism/accumulation cases pass with Ruff/diff.
  GPU validation and fork publication remain open; consumer pins are unchanged.
- [x] Revision88: after terminal R86 and direct GPU/container idle verification,
  launch posttrain-r88-shared-layer-fp16. Trace actual-context layer activations
  and cotangents at the shared updated parameters to locate remaining packing
  sensitivity. Observe this handle before launching another GPU probe.

- [x] (2026-10-02, revision78) GPU/container idle and host memory verified;
  launch posttrain-r78-layer-backward-bf16. On the unchanged native model,
  capture original-context activations and cotangents at all16 decoder layers
  plus the language head under the same controlled layouts and repeat. No
  optimizer update applies. Compare earliest forward/backward differences
  before choosing a numerical acceptance contract; raw tensor traces stay
  outside Git. Observe the existing handle rather than restarting on timeout.
  Terminal exit0; independent CPU comparison restricted to original context
  lengths shows all real activations exactly match. Language-head and layer15
  output cotangents match; first backward discrepancy appears at layer14
  output, after traversing block15, and grows toward lower blocks. Large raw
  forward differences were padded states, not original conditioning.
- [x] (2026-10-02, revision79) After R78 terminal state and idle GPU/container
  verification, launch posttrain-r79-final-block-backward-bf16 to trace final
  block submodules and replay its frozen w2 input-gradient GEMM. Retain current
  handle and diagnose any hook failure; no production code or tolerance changes.
  Exit0: frozen w2 replay isBF16 layout-exact. Submodule trace instead locates
  first cotangent difference at operator_norm output, while conv.in_proj output
  cotangents and other downstream submodule cotangents match exactly.
- [x] (2026-10-02, revision80) After idle verification, isolated frozen
  in_proj GEMM replay exits0. Identical contiguous single/packed operands give
  identicalBF16 results and match native packed gradients, but differ from
  native single-record gradients (5674 entries). This narrows the remaining
  control to operand layout rather than objective derivatives or weight values.
- [x] (2026-10-02, revision81) After R80 terminal/idle verification, launch
  posttrain-r81-strided-in-projection-bf16 preserving actual captured row strides
  in the isolated projection. Observe this handle before any production fix.
  Exit0: captured strides[1,2079] reproduce native singleton derivatives
  exactly. Contiguous strides[6144,1] reproduce packed derivatives exactly.
  Changing only stride produces5674 differingBF16 entries, maximum4.76837e-7.
- [x] (2026-10-02, revision82) After R81 terminal/idle verification, launch
  posttrain-r82-contiguous-backward-bf16. External native-backward prototype
  makes Linear output gradients contiguous while preserving their values;
  native model/FSDP/reduction still execute. Observe this existing handle, then
  test combined FP32-adapter precision only if needed. No production adoption.
  Exit0:117 Linear hooks preserve values and unchanged parameters; packing
  gradient gap falls to0.2343%, maxabsolute6.10352e-5. Same-layout repeat exact.
- [x] (2026-10-02, revision83) After R82 terminal/idle verification, launch
  posttrain-r83-fp32-contiguous-backward-bf16 combining nativeFP32 adapter
  wrapping with contiguous output gradients. Observe this existing handle;
  qualify real applied updates/resume before any generic native capability.
  Exit0: combined nativeFP32 adapters and contiguous cotangents reduce packing
  gradient relativeL2 to2.67064e-7 (0.0000267%), maximumabsolute7.45058e-9,
  with unchanged parameters and exact same-layout repeats. This validates the
  isolated correction combination for this retainedBF16 profile, not release.
- [x] (2026-10-02, revision84) After R83 terminal/idle verification, launch
  posttrain-r84-hybrid-packs-bf16: two actual native applied updates per layout
  under the combined controls, with model/optimizer checkpoint restoration and
  independent saved-gradient/adapter comparison. NativeFP32 LoRA initialization
  is preserved rather than rounded as in the diagnostic reference. Observe this
  handle; requireFP16, resume, budget accounting and maintained implementation
  before opening ordinary multi-record admission. BF16 exits0: two applied
  updates/layout, first gradient gap2.64198e-7 relativeL2 and adapter maxdelta
  6.90161e-8. Second-step gradient gap0.07907% and adapter maxdelta9.07350e-5
  remain despite equal objective loss and clipping support. Do not classify
  this as complete multi-update packing equivalence.
- [x] (2026-10-02, revision84 continuation) After BF16 terminal/idle state,
  FP16 counterpart exits0 with two applied updates/layout, scale128/no skips
  and zero initial score drift. Second-step losses/scores differ. Independently
  compare saved tensors; no admission or consumer pin change follows. Comparison
  first-step gradient gap2.65705e-7, adapter maxdelta2.80679e-7; second-step
  gradient gap0.8812%, adapter maxdelta0.000144264. Retain continuation failure.
- [x] (2026-10-02, revision85) After R84 FP16 terminal and idle verification,
  launch posttrain-r85-shared-boundary-bf16. Freeze initial old scores, apply
  exactly the same real R84 singleton step1 adapter values to both layouts,
  and intercept the next native optimizer before application. Compare update1
  backwards/repeats on unchanged shared parameters; native scheduler identity
  is advanced to the corresponding boundary. This is a derivative diagnostic,
  not full optimizer/job checkpoint recovery qualification. Observe this handle.
  Numerical acceptance, maintained capability, resume, physical budget accounting
  and ordinary-job qualification remain open.
  R85 exits1 in cleanup because the external generator accidentally removed
  contiguous-gradient hook setup while removing reference rounding. Its
  0.8271% result is not a combined-profile qualification result. Retain the
  failed handle/source; no production code changed.
- [x] (2026-10-02, revision86) Restore layout hooks, assert their receipt flag
  before execution and preserve scaler128 for the futureFP16 counterpart.
  After R85 terminal/idle verification launch posttrain-r86-shared-boundary-bf16
  in a separate directory. Observe this existing handle; do not reuse R85 as
  evidence for the combined profile.
  Verified terminal exit0: shared post-update BF16 parameters give packing
  gradient relativeL2 3.562957e-7 (0.0000356%), maximum absolute3.259629e-9,
  with exact same-layout repeats. FP16 shared-boundary verification remains open.
- [x] (2026-10-02, revision76) R75 FP16 unchanged-parameter probe exits0:
  gradients computedFP16, storedFP32, packing gap0.1068%, exact same-layout
  repeats and no applied updates. Strict R76 reference additionally disables
  both matmul and cuDNN TF32 and exits0; gap remains0.0011274%, maximum
  absolute2.18395e-7 and exact repeats. Thus TF32 does not explain that gap.
- [x] (2026-10-02, revision76) Independent CPU analysis of R73 first-step
  gradients finds271 opposite-sign entries inBF16 and38 inFP16 across319488
  entries. Maximum affected gradient magnitudes7.62939e-5 /2.86102e-6;
  none exceeds1e-4 BF16 or1e-5 FP16. Preserve these rather than interpreting
  small total relative-L2 differences as insignificant Adam update changes.
- [x] (2026-10-02, revision77) After strict reference terminal state and
  direct GPU/container idle verification, launch
  posttrain-r77-fp32-adapter-backward-bf16. External prototype uses native
  FSDP per-module precision exclusion and separate wrapping of24 LoRA Linear
  leaves, explicitly disabling autocast on those leaves while keeping the
  backboneBF16. It retains native gradient synchronization ownership. Observe
  this handle before production API/source work; tests do not adopt this policy.
  It exits0: unchanged parameters and exact repeat, but packing gradient gap
  remains0.8360%, maxabsolute0.000171833. FP32 adapter computation alone fails
  the repair hypothesis. Independent reference comparison reports approximately
  4.9% relative-L2 for both ordinaryBF16 andFP32-adapter variants versus the
  whole-modelFP32 reference on equal rounded weights. Do not adopt the prototype.
- [ ] Revision77 continuation: locate upstream backbone activation/cotangent
  sensitivity and define a numerically justified qualification contract.
  Ordinary multi-record admission remains closed. All above handles are
  terminal; no live GPU diagnostic remains at this stopping point.
- [x] (2026-10-02, revision 74) Verify GPU/container idle and launch
  posttrain-r74-fixed-backward-bf16. Intercept the native optimizer immediately
  after backward, with no optimizer/scheduler application. Compare layouts
  [1,1,1], [2,1], then [1,1,1] repeat on one unchanged model, with full model
  equality assertions and gradient dtype hooks. Use the established arithmetic
  controls. An explicit FP32 computation reference on identically BF16/FP16
  rounded parameter values is diagnostic only, not a training recommendation.
  First attempt exits1 on external output-path concatenation before model
  creation. Corrected retry exits0: unchanged parameters, no applied updates,
  0.8654% packing gradient difference and exactly zero same-layout repeat
  difference. Initial parameter hooks give no dtype events; do not claim they
  measured compute dtype. Capture dtype at native step and actual adapter
  forward views in the next probe instead.
- [x] (2026-10-02, revision 74) FP32 reference exits1 inside FSDP
  _use_unsharded_views on a tied-parameter Tensor/Parameter assertion. Retain
  its failure; it establishes no reference-gradient result and does not imply
  a defect in the supported BF16/FP16 recipe.
- [x] (2026-10-02, revision 75) GPU/container idle verified; launch
  posttrain-r75-fixed-backward-bf16-reference. External-only reference clones
  the frozen output embedding before FSDP while asserting equal values and
  frozen status, removing storage aliasing without changing fixed-model math.
  Improved dtype probes attach to actual adapter forward weight views and
  capture native stored gradient dtype before context cleanup. Observe this
  same handle; no production source or model selection is changed. Reference
  exits0 with unchanged parameters/no applied updates, relative gradient gap
  0.0011274%, maximum absolute2.18395e-7 and exact same-layout repeats. Adapter
  gradient hooks and native stored gradients areFP32. This reference sets
  float32 matmul precision highest but does not separately disable cuDNN TF32;
  do not treat it as a fully controlled numerical acceptance oracle yet.
- [x] (2026-10-02, revision75 continuation) After observing the reference
  terminal and GPU/container idle, launch posttrain-r75-fixed-backward-bf16
  with corrected adapter-view hooks and pre-cleanup dtype capture. Observe this
  same handle. It exits0, reproducing the0.8654% unchanged-parameter gradient
  gap and exact same-layout repeats. Actual adapter weight gradients areBF16
  at hooks, while native stored gradients areFP32.
- [ ] Revision75 continuation: posttrain-r75-fixed-backward-fp16 is the live
  handle after BF16 terminal state and GPU/container idle verification. Observe
  it and finish strict TF32-controlled reference before adopting numerical
  bounds or production precision changes.
- [x] (2026-10-02, revision 70) Reproduce loss of LoRA target topology using
  a real PEFT/HF model with a language q_proj and an unrelated vision q_proj.
  Preserve full trained module paths in native veRL adapter export, pass 18
  focused export tests, update fork/consumer ledgers and publish 3a4812cb.
- [x] (2026-10-02, revision 70) CPU-only posttrain-r70-gemma4-adapter-reload
  exits0 against the actual R66 checkpoint and R69 exported Gemma base model.
  All 50 language targets and 100 adapter tensors match the exported dtype;
  zero-adapter text logits equal the base exactly. This closes this topology
  repair's reload check, not informative Gemma training qualification.
- [x] (2026-10-02, revision 71) After direct GPU/container idle verification,
  launch posttrain-r71-fixed-scores-bf16 over the unchanged R67 source and R59
  evidence. Compare identical contexts/actions without restore or optimizer
  updates, with repeat controls, full GEMM accumulation and math SDPA. Observe
  this handle before launching another GPU diagnostic; packing remains gated.
  Initial attempt exits1 before model creation due an incomplete external
  PYTHONPATH. Retry with the existing verified all-package bootstrap exits0.
- [x] (2026-10-02, revision 71) Fixed-parameter BF16 and FP16 probes finish
  exit0, verify every model parameter unchanged and same-layout repeats exact.
  Math SDPA plus full GEMM accumulation makes both update subsets exactly
  layout-equivalent in both precisions. Population maximum score differences
  remain 0.226462603 BF16 / 0.027493596 FP16; independent row attribution
  confines these to the 1705-token context paired with the 2079-token context.
- [x] (2026-10-02, revision 72) Launch posttrain-r72-fixed-width-bf16 after
  confirming GPU/container idle. External wrapper pads all model-input rows to
  the frozen population's 2079-token width, retaining original masks/positions,
  and compares math SDPA with local versus fixed padding. Observe the existing
  handle; no production precision policy or multi-record admission is changed.
  Terminal exit0: fixed width plus math SDPA/full GEMM accumulation makes all
  4482 population action scores and both update subsets exactly equivalent
  across layouts, with unchanged model parameters and exact repeats.
- [x] (2026-10-02, revision 73) After verifying GPU/container idle, launch
  posttrain-r73-controlled-packs-bf16 for two actual native updates per layout
  with the isolated math-SDPA/full-accumulation/fixed-width controls. Observe
  this same handle and independently compare saved gradients and adapters.
  BF16 exits0 with two applied updates per layout and exactly zero initial
  old/current score drift. Independent CPU comparison still finds gradient
  relative-L2 differences 0.8654% at step1 and 11.8095% at step2. Adapter maximum
  differences remain 0.000199891 and 0.000392250. Keep packing unqualified.
- [x] (2026-10-02, revision 73 continuation) GPU/container idle verified after
  BF16 terminal state; posttrain-r73-controlled-packs-fp16 is the existing live
  handle. It subsequently exits0: two applied updates per layout, scale128,
  no skips and zero initial score drift. Independent CPU comparison finds
  gradient relative-L2 differences0.1068% /0.9022%, with adapter maximum
  differences0.000198614 /0.000238253. Packing remains unqualified.
- [ ] Revision73 continuation: isolate fixed-parameter backward arithmetic
  and mixed-precision adapter accumulation. All diagnostic handles above are
  terminal. Production memory/performance contracts remain gates.
- [x] (2026-10-02, revision 69) R66 ends exit1 (OOMKilled false) in native
  final export: torch.cat rejects a scalar. Independent CPU inspector exits0,
  authenticates both checkpoint/native-publication pairs with 16 components
  each, verifies 100 finite LoRA tensors and zero checkpoint-1-to-2 changes.
  Both native updates have zero prepared credit/gradient. No informative pass.
- [x] (2026-10-02, revision 69) After direct idle GPU verification, execute
  real BF16 FSDP comparison on published 7cf68728 over frozen mixed-reward R59
  evidence: two updates each under [1,1,1] and [2,1]. Both complete exit0, but
  gradient discrepancies are 8.879% at step1 and 32.756% at step2. Retain this
  as failed equivalence evidence; do not open ordinary multi-record admission.
- [x] (2026-10-02, revision 69) Reverify GPU idle and launch
  posttrain-r69-native-packs-fp16 with initial-adapter equality assertions and
  old/current-score drift instrumentation. Preserve the earlier BF16 artifacts
  in their own output directory. The model/optimizer remain native-owned.
- [x] (2026-10-02, revision 69) Find the exact Gemma export fix already merged
  upstream in PR #7610, commit ddb96db19850bf820fe0aa13e4cb71b21f285869.
  Backport its source and regressions, update the fork/consumer ledgers, pass
  four tests plus Ruff/diff, then publish source 62db2a41e048c21b69d8d98ae67da55448f88dd7.
- [x] (2026-10-02, revision 69) Stage the export backport separately and
  launch CPU-only posttrain-r69-gemma4-export from the existing checkpoint2.
  Do not retrain the zero-signal group or modify historical source snapshots.
- [ ] Revision 69 continuation: observe the existing FP16 packing and CPU
  export handles; measure score/gradient discrepancies and fix their cause
  before accepting native larger packs. Verify Gemma exported model/adapter
  reload preserves target topology; informative family and recovery checks
  remain open along with default SAMPO and broader release gates.
- [x] (2026-10-02, revision 69 continuation) FP16 layouts both finish exit0,
  scale128, no skips and two applied updates. Initial restored adapters match
  exactly. Independent gradient comparison still differs 0.9467% at step1
  and 2.0629% at step2. Packed pre-update scores drift from frozen old scores
  (maximum absolute log-ratio 0.034660816), while single-record drift is zero.
  Keep larger packs unqualified and investigate native batch arithmetic.
- [x] (2026-10-02, revision 69 continuation) CPU export from Gemma checkpoint2
  finishes exit0 on backport62db2a41. Real exported-config/meta-model reload
  fails: target_modules was reduced to q_proj/v_proj, losing the language-only
  paths and selecting unsupported Gemma4ClippableLinear vision projections.
  Retain this topology failure; a successful save is not usable-model proof.

- [x] (2026-10-02, revision 68) Verify R66 native metrics at both applied
  boundaries: policy loss, prepared advantages and gradient norms are exactly
  zero. Both select 72 sampled actions. All four real episodes pass task
  assertions. Record this as an uninformative group, not successful learning.
- [x] (2026-10-02, revision 68) Prepare actual FSDP packing comparison using
  R59's retained mixed-reward native evidence. Real isolated CPU preflight
  resolves the same two episode minibatches under layouts [1,1,1] versus [2,1]
  for each update. Both runs will restore the same initial native checkpoint.
  Qualifier, source and outputs remain in external r67 storage, outside Git.
- [x] (2026-10-02, revision 68) Launch independent CPU Gemma checkpoint
  inspector posttrain-r68-gemma4-checkpoint-inspector against both sealed
  native/publication pairs; preserve the active R66 GPU job and finalization.
- [ ] Revision 68 continuation: observe the existing R66 and inspector
  handles. When the GPU is idle, run check_native_packs.py on published
  7cf68728 in BF16 and FP16, compare retained gradients/adapters/optimizer
  tensors and full counters. Informative Gemma and recovery remain gates.

- [x] (2026-10-02, revision 67) Extend native veRL prepare_micro_batches with
  optional declared ordered micro_batch_sizes. Validate full row coverage,
  positive sizes, group integrity, collective count agreement and requested
  count limits; retain native backward accumulation and optimizer ownership.
  Fourteen native helper cases pass, including a real two-process Gloo check.
- [x] (2026-10-02, revision 67) Publish source/test/ledger commit
  7cf687284ba054e836b8ce34475a3e695b3dd22b on the maintained fork branch.
  Publish ledger follow-up d3ca05a4. Consumer page and explicit source mapping
  identify the candidate; immutable dependency pins remain unchanged.
- [x] (2026-10-02, revision 67) Framework adapter constructs declared packs
  under record/context capacities, checks them against plan_packs and verifies
  exact callback coverage before native stepping. Remove private population
  and collection-host single-record restrictions, keeping ordinary admission
  and public guards pending GPU qualification. Twelve adapter tests pass,
  including objective/parameter equivalence for two execution layouts.
- [x] (2026-10-02, revision 67) Scoped Pyright reports zero errors; Ruff,
  nine import contracts and diff checks pass. Expanded cohort: 29 pass and one
  previously recorded installed post8/native-name metadata mismatch fails.
  Do not suppress that release reconciliation gate.
- [x] (2026-10-02, revision 67) Stage the published 7cf68728 fork and 360
  hashed current framework files in a separate immutable external r67 root.
  R66 keeps its original source. Manually inspect its four current traces:
  each has task_completed_correctly=1 and no recorded errors. This may yield
  uninformative group credit; inspect saved prepared credit before qualification.
- [ ] Revision 67 continuation: qualify actual native GPU pack equivalence on
  the newly published source, retaining old/new identities separately. Observe
  existing R66 Gemma job and inspect credit/parameter changes and recovery;
  checkpoint saving alone is insufficient proof of informative optimization.

- [x] (2026-10-02, revision 60) Observe existing R59 strict BF16 ordinary
  job to exit 0 (OOMKilled false). It applies three informative updates,
  saves checkpoints, merges the model and completes host observation replay:
  seven events, five artifacts, four traces and three canonical metric records.
- [x] (2026-10-02, revision 60) Independent CPU inspector authenticates all
  three native/publication pairs (15 components each), applied/attempt/cursor
  boundaries 1/2/3, and 24 finite changed LoRA tensors from checkpoint 2 to 3.
  Maximum adapter delta 0.00010035664308816195. Raw receipts outside Git.
- [x] (2026-10-02, revision 60) Manually inspect all four real R59 traces:
  rewards [1,0,0,0], two reasoning-only truncations and one empty-argument
  tool failure followed by truncation. System/user/tool and the unsampled
  assistant rewrite have zero policy mask. Assistant supports are
  723/1024/1024/1711 sampled tokens; first three reconcile to native metrics.
- [x] (2026-10-02, revision 60) Execute the new native determinism regression
  against the parent helper in an external in-memory control: it fails on
  warn-only mode as expected. Parent source files remain unchanged.
- [x] (2026-10-02, revision 60) Reverify workstation GPU idle after R59 exits;
  launch posttrain-r60-strict-bf16-resume from published checkpoint 1 with
  unchanged r59 source/settings. Prepare independent step-2/step-3 comparisons.
- [x] (2026-10-02, revision 61) Ordinary strict BF16 R60 continuation exits
  0 (OOMKilled false). Independently authenticate checkpoints 2 and 3 and
  compare 248 tensors plus driver state at each boundary with uninterrupted
  R59. Model, optimizer, RNG, scheduler, frozen scores and driver match exactly:
  no differing tensors, maximum absolute difference 0.0 at both boundaries.
- [x] (2026-10-02, revision 61) Confirm GPU compute list empty after R60
  terminates and launch ordinary posttrain-r61-strict-fp16-kl on the same
  published source, initial scale 128, beta 0.05 and adapter-disabled frozen
  base reference. Prepare native/publication inspection and checkpoint-1
  continuation (R62) without changing the source or runtime identity contract.
- [x] (2026-10-02, revision 62) R61 strict FP16/base-KL ordinary job exits
  0 (OOMKilled false), applies three updates and completes native saves/merge
  and host finalization. Independent CPU inspector authenticates three
  native/publication pairs (15 components each), 24 finite changed LoRA tensors
  and maximum checkpoint-2-to-3 adapter delta 0.00010035886953119189.
- [x] (2026-10-02, revision 62) Verify all 3994 frozen old/reference scores
  remain identical across three checkpoints; initial old scores equal the
  adapter-disabled base reference. Saved scale is 128 at each boundary,
  growth trackers 1/2/3, no skips, and policy/KL supports 671/895/1024. Weighted
  KL losses are 0 / 0.000004557688043860253 / 0.000008141176294884644.
- [x] (2026-10-02, revision 62) Reverify GPU idle after R61 terminal state;
  launch posttrain-r62-strict-fp16-kl-resume from published checkpoint 1 with
  unchanged source/settings/reference contract. Keep local FP16 raw artifacts
  in their own subdirectory so earlier BF16 evidence is preserved.
- [x] (2026-10-02, revision 63) R62 ordinary FP16/base-KL resume exits 0.
  Independently authenticate and compare checkpoints 2 and 3 against R61:
  249 tensors plus driver state match exactly at both boundaries, including
  optimizer/scaler/RNG/scheduler/frozen-old/reference state and counters.
- [x] (2026-10-02, revision 63) Inspect real FP16 traces: rewards [1,1,0,1],
  sampled supports 671/895/1024/1404 sum to the frozen 3994-action population.
  Three successful tool episodes and one reasoning-only truncation; every
  unsampled/system/user/tool coordinate is masked. Retain external inspection.
- [x] (2026-10-02, revision 63) Offline native Gemma 4 factory check selects
  AutoModelForImageTextToText/Gemma4ForConditionalGeneration for pinned E2B
  config 3e22461f65e89153144f8adb70e3b8c2cc9845a7. Check cached config,
  storage (548 GiB free) and idle GPU; launch ordinary
  posttrain-r63-gemma4-strict-bf16 for two applied updates, G=4, text-only
  off-reasoning renderer, LoRA rank8/alpha16, beta0 and the same source/image.
- [x] (2026-10-02, revision 63 continuation) Initial R63 attempt exits 1
  before model initialization: the launcher family guard admits only LFM/Qwen.
  Retain this failure. Reverify idle state and launch separate
  posttrain-r63b-gemma4-candidate-bf16 with an external exact-policy family-gate
  bypass, in addition to the already-declared public constructor bypass.
  Production family support stays guarded; no source allowlist is expanded.
- [x] (2026-10-02, revision 65) R63b ends before model initialization because
  the pinned offline Hub snapshot lacks README.md and .gitattributes. R64's
  local-artifact hypothesis fails the foundation variant validator before
  native allocation. Retain both failures; do not weaken foundation identity.
- [x] (2026-10-02, revision 65) A real CPU check reproduces the incomplete
  snapshot even with local_files_only=True. Fetch exactly the two missing
  files at the original immutable Hub revision; unrestricted offline
  snapshot_download then resolves all nine cached files successfully.
- [x] (2026-10-02, revision 65) Confirm GPU compute list and running Docker
  list empty, then launch posttrain-r65-gemma4-cache-complete-bf16 with the
  unchanged r59 framework/fork source and original pinned foundation model.
  The existing exact-policy qualification bypass remains external only.
- [x] (2026-10-02, revision 66) R65 loads all 1951 Gemma weight tensors, then
  exits 1 (OOMKilled false) during PEFT injection: suffix q_proj/v_proj also
  selects Gemma4ClippableLinear vision/audio projections. A real meta-device
  model preflight selects 50 language Linear projections and successfully
  injects 100 trainable LoRA tensors, all under language_model.
- [x] (2026-10-02, revision 66) Retain the language projection receipt outside
  Git, reverify idle GPU/container lists, and launch
  posttrain-r66-gemma4-language-lora-bf16 using the same pinned foundation,
  framework/fork/image and bounded two-update check with explicit targets.
- [ ] Revision 66 continuation: observe the existing R66 Gemma 4 handle and
  qualify informative credit, original context masks, finite parameter changes,
  checkpoints and continuation. If groups are uninformative, retain that result
  without calling it optimizer qualification. Distributed/default SAMPO/native
  semantic-span, observation/release/public-admission gates remain open.

- [x] (2026-10-02, revision 57) Revalidate workstation idle state directly:
  no GPU compute processes and no running Docker containers. Launch the bounded
  external posttrain-r57-backward-probe against the unchanged r55 source and
  deterministic BF16 checkpoint 1. Instrument native loading to save immediate
  restored state, then repeat native backward three times before any optimizer
  update. Raw probe and bootstrap remain outside Git.
- [x] (2026-10-02, revision 58) R57 ends with its expected pre-optimizer
  diagnostic exception (exit 1, OOMKilled false). Independent CPU container
  verifies 247 restored model/optimizer/extra-state tensors match checkpoint 1
  exactly. Three native backward passes under unchanged weights/RNG produce
  different gradients with no buffer changes. Raw receipts retained outside Git.
- [x] (2026-10-02, revision 58) Identify native PyTorch warning: SDPA Flash
  Attention backward remains nondeterministic with warn_only=True. The veRL
  enable_full_determinism helper explicitly sets that mode. Reverify idle GPU
  and launch posttrain-r58-strict-backward-probe, changing only the external
  diagnostic's backward mode to warn_only=False after authenticated restore.
- [x] (2026-10-02, revision 59) Strict R58 repeats produce exactly equal
  gradients across all 24 trainable tensors, norm 0.15649279096790697 each time,
  with no buffer changes. Expected diagnostic abort exits 1, OOMKilled false.
  In R57, 22/24 gradients differ in each repeat; maximum difference 0.00006485.
- [x] (2026-10-02, revision 59) Correct the owning veRL helper's opt-in
  full_determinism mode to warn_only=False; add native CPU regression proving
  that the helper clears warn-only mode. One test, fork Ruff and diff checks
  pass. Update fork ledger and framework consumer page together. Normal
  training defaults are unchanged; unsupported strict operations must fail.
- [x] (2026-10-02, revision 59) Commit and push strict-determinism source
  8f0de2365f1041954b67f74df5a14c7ba0532755 to
  origin/codex/resolved-engine-worker. Record it in the consumer page and
  inherited native-name source capability mapping; public request guard remains.
- [x] (2026-10-02, revision 59) Stage immutable fork source 8f0de236 and
  360 hashed framework source/resource files in the new external r59 root.
  Ledger publication follow-up 83c35675 is pushed. Verify idle GPU and launch
  ordinary posttrain-r59-strict-bf16 (no diagnostic instrumentation) for three
  applied updates through full save/observer/model-publication paths.
- [ ] Revision 59 continuation: observe posttrain-r59-strict-bf16, verify
  checkpoints and strict arithmetic evidence, then run unchanged-source
  checkpoint-1 continuation and FP16 full-job qualification.
  Instrumented backward equality alone does not qualify full-job recovery.

- [x] (2026-10-02, revision 56) Ordinary r55 FP16 job exits 0 through native
  fit, three checkpoint saves, final model merge, host metric replay and five
  artifact observations. Host retains all three canonical metrics with native
  applied/attempt counts 1/1, 2/2, 3/3 and scale 128; no optimizer skips.
  Negative/negative/positive credit supplies informative applied updates.
- [x] (2026-10-02, revision 56) Independently authenticate all three FP16
  native checkpoints and publication copies (15 components per boundary).
  All 24 LoRA tensors are finite and change between updates 2 and 3, maximum
  delta 0.00010034973092842847. Receipt remains outside Git in
  full-verl-20261002-r55/three-update-verification.json.
- [x] (2026-10-02, revision 56) Confirm FP16 job exited and GPU has no compute
  processes before launching posttrain-full-verl-r55-bf16-deterministic on the
  same r55 source/image with native actor fsdp_config.full_determinism=true.
- [x] (2026-10-02, revision 56) Correct resolution, score-recovery and admission
  test typing using concrete settings constructors, TypedDict boundaries and
  non-null frozen-evidence assertions. Scoped Ruff/Pyright pass; resolution
  tests pass 13, native dependency admission/recovery tests pass 14.
- [x] (2026-10-02, revision 56) Finish fixture typing without excluding tests
  or weakening production annotations: explicit test-double boundaries, concrete
  settings arguments, typed recovery/admission inputs and non-null evidence
  assertions. Full repository Pyright now passes with zero errors; Ruff and
  git diff --check pass. Affected native tests pass 143, plus separate resolution
  and admission/recovery checks. Optional TRL async tests now skip clearly when
  TRL is absent; full pytest collects 2472 tests and is running, with a catalog
  failure requiring diagnosis. This does not qualify skipped native backends.
- [ ] Revision 56 continuation: observe the same deterministic BF16 handle;
  qualify its unchanged-source checkpoint-1 continuation, complete FP16 resume
  and KL-enabled ordinary jobs, and finish the remaining type/test release gates.
  Current live resume handle is posttrain-full-verl-r56-bf16-deterministic-resume
  under full-verl-20261002-r55; compare_deterministic_resume.py is staged there.
- [x] (2026-10-02, revision 56) Deterministic ordinary BF16 job exits 0;
  independent CPU inspection authenticates all three native/publication pairs,
  all counters/cursors and 24 finite changed adapter tensors (maximum step-2/3
  delta 0.00010035348532255739). Launch unchanged-source checkpoint-1 resume
  posttrain-full-verl-r56-bf16-deterministic-resume only after idle GPU verification.
- [x] (2026-10-02, revision 56) Full base pytest finishes 2360 passed,
  32 failed, 106 skipped. Diagnose catalog mismatch separately; rerun the
  affected optional-ML files with real detached d97a2cf TRL, published veRL,
  Torch/Hydra/Verifiers/renderers dependencies: 317 pass. Retain the external
  native-cpu-regressions.log and dependency/path declaration; no failures are
  suppressed. This cohort does not replace the complete native release suite.
- [x] Revision 56 full CPU native-dependency suite finishes via external
  check_cpu_full_suite.py; stdout/stderr are retained in native-cpu-full-suite.log.
  CUDA visibility is disabled for this CPU suite; remote native resume is
  preserved. Result: 2625 passed, 25 skipped, seven failures and 21 errors.
  Besides the catalog mismatch, the legacy local distribution reports TRL post2
  while the consumer pin is post13, and candidate veRL post8 metadata exposes
  sampo_token_credit instead of the recorded sampo_hierarchy registry name.
  Four failures plus all 21 errors concern local Ray startup timeouts. These
  must be reconciled under the final published runtime and native release gate;
  do not claim a passing complete suite from the 317-case cohort.
- [x] (2026-10-02, revision 56) Deterministic BF16 resume exits 0. Independently
  authenticate checkpoints 2/3 and compare 248 tensors plus driver state at
  both boundaries. Frozen scores/RNG/driver/counters match, but adapter and
  optimizer states still differ: maximum adapter differences 0.00014805783575866371
  at step 2 and 0.00023292191326618195 at step 3. Native full_determinism alone
  does not close recovery equality; preserve both negative receipts outside Git.
- [x] (2026-10-02, revision 56) After verifying no GPU compute processes remain,
  probe the actual causal_conv1d CUDA primitive and PyTorch convolution at
  batch=1/channels=1024/length=2048/width=3 under deterministic CUDA settings.
  In BF16 and FP16, all three repeats have bitwise-equal forward values, input
  gradients and weight gradients for both paths. This bounded negative result
  does not attribute whole-model drift to that convolution kernel.

- [x] (2026-10-02, revision 55) Same-source r54 BF16 resume restores checkpoint
  1 and completes native updates 2 and 3, saves and final merge. Outer host
  fails on the historical duplicate metric alias, as expected. Independently
  authenticate both final checkpoints and recursively compare 248 tensors plus
  driver data.pt: frozen scores, driver state, RNG and counters match; all 24
  adapter tensors and optimizer moments differ. Maximum adapter absolute
  difference is 0.0002171425148844719. Aggregate adapter drift L2 is
  0.005904951815619802 versus 0.0670279967424202 for the uninterrupted two-update
  change: 8.809679689983511 percent. Recovery equality remains unqualified.
- [x] (2026-10-02, revision 55) Stage current framework source (360 hashed
  resource/source files) with published veRL 77fe49a in immutable external
  full-verl-20261002-r55. After confirming the resume container exited and no
  GPU compute processes remained, launch posttrain-full-verl-r55-fp16 using
  the same digest-pinned image, real AutomationBench group and scale 128.
- [ ] Revision 55 continuation: observe the same FP16 container and verify
  complete host metrics/artifacts. Diagnose BF16 continuation drift with native
  full_determinism controls; distinguish checkpoint restoration from backward
  repeatability before attributing discrepancies to a framework defect.
- [x] (2026-10-02, revision 55) Run repository-wide Ruff (passes) and import
  boundaries (nine kept). Full Pyright reports 258 errors, predominantly test
  fixtures; full pytest stops at collection for absent optional TRL/Torch.
  Correct the distributed engine's optional-reference narrowing and skip the
  new native-runtime test clearly when Torch is unavailable. Scoped source
  Pyright passes; affected tests in the native dependency environment pass 13
  with five dependency/runtime skips. Adding the published fork source to that
  environment runs all six native-runtime tests successfully, including actual
  native CLI/Hydra validation and Ray runner serialization. Full type/test
  release gates remain open.
- [x] (2026-10-02, revision 55) Recheck full Pyright after source correction:
  257 remaining errors, all in tests. Correct the largest transport fixture
  cluster (85 errors) using concrete constructor arguments, algorithm-specific
  payload construction, explicit minimal test-double casts and non-null envelope
  assertions. That file now passes Pyright/Ruff and all 11 tests without
  changing its selected algorithms or weakening its behavioral assertions.

- [x] (2026-10-02, revision 54) Ordinary r53-b native fit completes three
  informative BF16 updates, all native saves, queue cleanup and final model
  merge. Native worker succeeds; outer host replay fails on duplicate
  canonical/actor policy-loss names. Producer now emits one canonical loss;
  native summary accepts evaluated total loss and normalization preserves
  resolved counters/support/nonzero-credit and native scale diagnostics.
  Actual actor RPC metrics round-trip through normalization in regression tests.
- [x] (2026-10-02, revision 54) Independently verify all three native
  checkpoint/publication pairs, 15 sealed components each and counters/cursors
  1/2/3. All 24 LoRA tensors remain finite and change between updates 2 and 3;
  maximum absolute delta 0.00010035588638857007. Retained external receipt:
  full-verl-20261002-r53b/three-update-verification.json.
- [x] (2026-10-02, revision 54) Manually inspect all four r53-a native traces:
  sampled policy support 725/1024/1024/1711, zero user/tool/unsampled assistant
  masks, one successful Slack episode and three reply-token-limit endings.
  Retain raw traces and manual review outside Git. Generic agent_completed
  metadata does not establish absence of truncation.
- [ ] Revision 54 continuation: observe posttrain-full-verl-r54-bf16-resume,
  restoring published checkpoint 1 under the unchanged r53-b source/runtime.
  Compare frozen evidence/scores, remaining updates, final native parameters,
  optimizer/RNG and counters. That historical source retains the outer metric
  replay failure; fresh current-source full-job qualification is still required.

- [x] (2026-10-02, revision 53) Observe r52-g terminal failure during native
  LoRA creation: veRL received q_proj,v_proj as one PEFT regex, while TRL
  split the same Posttrain CSV selection. Move the existing parser to neutral
  bindings.py and reuse it in both adapters. Actual Hydra config tests preserve
  CSV names, whitespace, all-linear and regex selections. veRL tests pass
  125 with seven existing dependency skips; TRL common tests pass 25.
- [ ] Revision 53 continuation: posttrain-full-verl-r53-bf16-a is the existing
  failed job retained for analysis. Two native optimizer updates committed and
  both full checkpoints/publication copies verify, but empty-batch cleanup
  failed afterward. Observe replacement posttrain-full-verl-r53-bf16-b on
  published fork 77fe49a; complete job/resume qualification remains open.
- [x] (2026-10-02, revision 53) Independently verify both failed-job checkpoints
  and staged copies: 15 sealed components each, applied/attempt counters 1/1
  and 2/2, frozen population cursors 1 and 2. All 24 finite LoRA tensors change
  between updates; maximum absolute delta 0.00010013514838647097.
- [x] (2026-10-02, revision 53) Publish native V1 cleanup guard for steps with
  no newly allocated TransferQueue keys. Actual fit-loop regression preserves
  two saves/logs/counter increments and clears only the allocated batch; all
  15 native trainer-base CPU tests pass. Published implementation 076072b9,
  ledger head 77fe49a, before any framework pin/runtime adoption.

- [x] (2026-10-02, revision 52) Launch ordinary native veRL qualification on the
  idle RTX PRO 6000 using published fork ef5aac6, exact existing GPU image,
  pinned LFM Thinking, real AutomationBench collection, P=1/G=4 and three
  selected updates. Preserve all setup failures outside Git. Fix our CLI's
  missing native validator arguments and Ray runner capture of a live
  RunContext; serialize the actual Ray actor class in the regression test.
  Focused native/backend suite: 169 passed, seven dependency skips.
- [ ] Revision 52 continuation (superseded by revision 53's job): qualify
  through native model initialization, collection, three applied updates and
  checkpoint publication. It uses a writable datasets cache and remains an
  unqualified attempt until actual optimizer/checkpoint evidence is inspected.

- [x] (2026-10-02, revision 51) Stage complete verified job checkpoints in
  separate publication directories before emitting recovery artifacts. Rotation
  removes only complete superseded native sources with independently verified
  identical stages. Publication copies remain host-owned while asynchronous
  observation delivery proceeds. Native model-only pruning is disabled during
  save; the driver owns sealed rotation. checkpoint_steps=0 preserves the
  existing no-retained-recovery policy while allowing the terminal model merge.
- [x] (2026-10-02, revision 51) Route explicit selected manifests through the
  resolved native module in the ordinary isolated worker, validate selection
  before dataset/runtime activation, and discover only sealed terminal saves.
  Native Hydra overrides select V1 sync, dense one-context scoring and disable
  stock admission retries that would mutate populations before normalizer
  admission. Public request guards remain until real native job qualification.
- [x] (2026-10-02, revision 51) Check direct workstation state over SSH: RTX PRO
  6000 has 41 MiB/97887 MiB allocated, 0 percent utilization and no running
  containers at inspection. No GPU job was launched in this revision.

- [x] (2026-10-02, revision 50) Add private policy_native.py entrypoint composing
  resolved actor/session, trainer and native TaskRunner before Ray dispatch.
  Validate typed host/settings, single-host/rank, native update budget and
  unwarped sampler selection. Expose a CLI using native Hydra configuration and
  native run_ppo; the ordinary worker/public request branch stays guarded until
  checkpoint publication/retention and real job qualification are complete.
- [x] (2026-10-02, revision 50) Add actual native runtime/template identity
  derivation and adapter-disabled base-reference provider in policy_runtime.py.
  Bind selected payload, native source, framework sources, installed versions,
  effective engine/optimizer/model configuration and arithmetic controls.
  Exclude output/resume locations. Base-reference scores use the native adapter
  disable context and are detached population-frozen values; alternate reference
  ownership rejects instead of fabricating a replacement.

- [x] (2026-10-02, revision 49) Add resolved_task_runner_type in
  backends/verl/policy_runner.py retaining native manager construction,
  TransferQueue lifecycle and logger completion. Native TaskRunnerV1 was already
  a Ray actor and could not be subclassed; expose plain TaskRunnerV1Base in the
  maintained fork while preserving the default Ray wrapper. Publish tested
  source 70baba82c0b2b0a8981089ca62c5ab474efe1816 on
  origin/codex/resolved-engine-worker before any consumer pin adoption.
  Fork ledger and consumer page are updated. This is not full-job qualification.
- [x] (2026-10-02, revision 49) Make resolved native fit termination use the
  declared applied-update budget independently of dataset length/population
  reuse. Keep native dataloader cursor for fresh collection/recovery; do not
  report its synthetic fit-loop epoch as a resolved objective epoch. Require
  native total_training_steps to match the selected loop budget.

- [x] (2026-10-02, revision 48) Add actor_session_from_manifest binding the real
  typed RunContext and selected settings to the existing actor host, with sealed
  raw-native evidence restoration and no credit recomputation. Reject absent
  context/settings and NullObserver. Native composition must still supply and
  qualify actual runtime/template/score identities and frozen reference scoring.
- [x] (2026-10-02, revision 48) Add credential-free ResolvedWorkerObserver in
  backends/verl/policy_observer.py. It fsyncs observation JSONL before returning;
  the parent launcher polls through the existing AppendOnlyJsonlTailer and
  forwards typed observations to the actual host observer. Failed publication
  does not acknowledge its record; wrong run/schema rejects. Successful resolved
  jobs must synchronize a nonempty journal. Native worker/TaskRunner installation
  is still pending, so this is transport qualification, not a launched job.

- [x] (2026-10-02, revision 47) Add complete native driver checkpoint seals in
  backends/verl/policy_checkpoint.py and wire native synchronous trainer save
  and load hooks. Reuse the existing update-recovery schema to bind actor
  components and actor seal, mandatory data.pt and additional driver files.
  Auto recovery ignores unfinished saves and rejects corrupted committed saves;
  explicit recovery authenticates all bytes before native model/dataloader load.
  Exercise actual native PPOTrainerSync save/load with fixture actor evidence
  and native torch dataloader serialization; this is not GPU resume evidence.
- [x] (2026-10-02, revision 47) Transport the actual host RunContext identity
  through typed VerlRunContext at launcher execution, with framework identity
  and absolute workspace validation. Detached plans may omit it. Actor session
  factory and durable observer transport still need composition.

- [x] (2026-10-02, revision 46) Add actor-owned resolved RPC sessions and a
  synchronous native V1 trainer composition in backends/verl/policy_session.py,
  policy_driver.py and resolved_workers.py. Collection runs only at exhausted
  population boundaries and tags the already-applied actor version; retained
  minibatches reuse original receipts without the flattened PPO pipeline.
  Native initialization, replay storage, weight synchronization and fit remain
  native responsibilities. The real PPOTrainerSync constructor is exercised;
  this is not a launched native GPU job.
- [ ] Revision 46 continuation: connect the typed job manifest/RunContext and
  retained artifact publication to the actor session factory and TaskRunner;
  seal native driver data.pt alongside actor evidence, authenticate it before
  resume, and implement retention after the complete job seal. Retention RPCs
  currently reject configured pruning instead of silently ignoring it. Resolve
  dataloader epoch/update counting before enabling ordinary veRL launch.

- [x] (2026-10-02, revision 45) Separate restoration from backward repeatability
  using external named-parameter/gradient captures on ordinary BF16 Thinking
  jobs. All 24 pre-update adapter tensors and their parameter order match after
  resume. Nondeterministic backward has relative gradient L2 difference
  0.009307596249859723; only 2/24 resulting tensors match. With deterministic
  algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8, all 24 pre-update tensors,
  gradients and post-update tensors match exactly. This qualifies the diagnostic
  mode's retained second occurrence, not subsequent fresh sampled populations.
- [x] (2026-10-02, revision 45) Expose TRL's native full_determinism backend
  option, configure its controls before CUDA model loading, pass it to native
  Trainer and observation, and bind effective deterministic/TF32/workspace
  controls into resolved runtime identity. Defaults retain existing behavior.
  51 focused regressions, scoped Ruff/Pyright and all nine import contracts pass.
- [x] (2026-10-02, revision 45) Qualify production full_determinism on ordinary
  BF16 Thinking with one fresh complete four-episode native group. Three updates
  have nonzero credit; resumed updates 2/3 match all 24 adapter tensors, optimizer
  state and frozen evidence exactly, and checkpoint seals verify. Update 2 clips
  2/976 selected actions. This uses native :16:8 workspace controls, distinct
  from the diagnostic :4096:8 setting. The equivalent FP16 group finishes with
  four successful episodes and zero credit; informative FP16 continuation is
  not claimed for that group. The bounded eight-episode FP16 pair completes
  with five successes/three length failures and three nonzero-gradient updates.
  Resumed updates 2/3 match all 24 weights, optimizer, scheduler, scaler and
  counters exactly, with identical frozen sidecars and verified checkpoint seals.
  FP16 selected credit includes positive 0.7244288643548162 and negative
  -1.2073814405913603; native scaler remains 128 with growth trackers 2/3.
- [x] (2026-10-02, revision 45) Two-episode BF16/FP16 controls and resumes
  finish, matching retained zero-credit update 2 exactly. Update 3 collects a
  different population and therefore is not a fixed-evidence equality test:
  masks/old-score/schedule sidecars differ. Both BF16 first assistant nodes have
  identical token IDs, sampled logprobs and messages across fresh groups, while
  timestamps differ. Keep later-population parameter differences as separate
  coverage, not evidence of a retained-population restoration failure.

- [x] (2026-10-02, revision 44) Add native TrainingWorker.create_engine factory
  extension and framework recipe-worker specialization, preserving pre-dispatch
  construction and native lifecycle. Publish source candidate
  cad959e97471600376203d1a1d860259d32dc64d on codex/resolved-engine-worker;
  source starts from acad5211619aef5ed80b25e9657262f08a436b03. Two native worker
  constructor tests and six framework factory tests pass; scoped quality checks
  and import contracts pass. Fork/consumer ledgers record candidate publication;
  runtime pins remain unchanged pending ordinary worker qualification.
- [x] (2026-10-02, revision 44) Fresh ordinary BF16 Thinking job at 512 tokens
  completes three slots with zero policy credit; manual native trace inspection
  confirms four length endings and no completed tool call. At 1024 tokens,
  the first complete group has rewards 0/1 (malformed rejected call versus
  successful tool episode), producing two nonzero policy-gradient updates. The
  second update reports 2/646 clipped selected actions; all 24 adapter tensors
  change between checkpoint 1 and 2. Third slot has zero credit on a new 1/1
  group. All three checkpoint seals verify. Nonzero-gradient continuation
  completes but fails adapter equality: only 2/24 tensors match at checkpoint 2,
  maximum absolute difference 0.00014228280633687973. Loss, clipping and three
  retained population/score sidecars match; optimizer steps agree but moments
  differ in 22/24 entries. The cause is not established. BF16/FP16 matched fresh evidence,
  veRL production, distributed and full release gates remain open.

- [x] (2026-10-02, revision 43) Carry complete typed normalizer settings in
  resolved veRL manifests, discriminated by GRPO/SAMPO/GDPO/CAPO operation.
  Preserve original identities, loop values, numerical profiles, filtering and
  objective selections rather than reconstruct settings from lossy native
  algorithm fields. Manifest validation checks corresponding population,
  beta/clipping and native loop/warmup fields; missing settings or divergent
  launch values reject. Distillation rejects this unqualified selection.
  Transport/backend suite: 132 passed, seven skipped. Native worker composition
  and all other release gates remain open.

- [x] (2026-10-02, revision 42) Add ResolvedVeRLCollectionHost binding native
  DataProto receipt columns to durable shared admission and ResolvedVeRLRun.
  Keep one engine/optimizer, collect only at exhausted population boundaries,
  publish native artifacts before execution and delegate native checkpoint
  operations. Restoration re-resolves the selected contract from frozen credit
  and rejects changed settings, versions, template, scoring or retry policy.
  Native source acad5211619aef5ed80b25e9657262f08a436b03: nine BaseEngine
  tests pass, including four applied updates/two populations/five attempts after
  one overflow. Scoped Ruff/Pyright and nine import contracts pass. Ordinary
  worker startup, native model/Ray/GPU integration and public enablement stay open.

- [x] (2026-10-02, revision 41) Add veRL NativeEpisodeReceipt JSON transport
  retaining typed EnvironmentRollout metadata and a digest-verified original
  replay artifact. Opt-in agent-loop collection retains this receipt before
  returning flattened rows and supplies explicit prompt-occurrence/session
  identities for all resolved algorithms. Verify original conditioning and one
  synchronous sampler version; reject flattened/mixed-version inputs and
  modified artifact bytes. Add admit_episode_receipts to authenticate and merge
  original JSONL envelopes into one durable replay artifact, then delegate
  complete-group credit/objective resolution to AdmittedNativePopulation.
  Selected duplicate trajectories, mixed formats and incomplete groups reject.
  Native resolved-host wiring and GPU qualification remain open.

- [x] (2026-10-02, revision 40) Preserve typed PolicyUpdateSettings in the veRL
  isolated-worker algorithm contract for GRPO/DAPO, SAMPO and structured RL.
  Resolved launch plans validate the explicit legacy-loop compatibility contract
  instead of forcing population size to equal one optimizer minibatch. JSON
  round trips preserve schedule, execution hard limits, independent span
  selections, denominator and empty-support semantics. The existing main_ppo
  worker rejects this selection before runtime loading or dataset creation;
  it cannot silently train the legacy objective. Native resolved worker
  composition and public enablement remain open. Isolated manifest/backend
  regressions: 128 passed, seven skipped; scoped Ruff/Pyright pass and all nine
  import contracts remain intact. Skips are not native integration evidence.

- [x] (2026-10-02, revision 39) Correct ordinary resolved TRL observation:
  emit evaluated total, policy and beta-weighted KL losses separately at each
  applied boundary; emit divergence only when beta enables reference scoring.
  Selected-action advantage statistics use frozen prepared credit. Digest and
  denominator events retain evaluated meaning. Suppress native windowed total
  loss from the legacy policy-loss normalizer on this branch. Duplicate applied
  observations are idempotent; resume initializes its observation cursor from
  the authenticated applied counter. Ten composition tests, scoped Ruff/Pyright
  and nine import contracts pass; 38 composition/recovery/execution tests pass.
  Fresh ordinary native FP16 verification completes three updates and reports
  exactly three policy-loss records (all zero), separate positive KL losses and
  zero prepared-credit statistics. No duplicate legacy policy-loss values occur.

- [x] (2026-10-02, revision 37) Resume the ordinary FP16 Instruct job from its
  native checkpoint 1 to global step 3. All 24 final adapter tensors match the
  uninterrupted job exactly; checkpoint 2 retains byte-identical native evidence,
  population/credit and frozen scores. Native scaler equals scale 128; only two
  new episodes are collected after exhausting the restored population. This is
  zero-advantage continuation, not a nonzero-gradient resume qualification.
- [x] (2026-10-02, revision 38) Seed resolved job initialization before PEFT
  model loading, using Transformers set_seed. Native Trainer seeding occurred
  after adapter construction. Keep legacy job behavior unchanged. Scoped quality
  checks and 80 selected API/composition regressions pass; nine import contracts
  remain intact. New runtime source identity requires new qualification receipts.
- [x] (2026-10-02, revision 38) Fresh ordinary FP16 job continues the already
  qualified Instruct adapter from revision-26 native checkpoint 1, selects beta
  .04 with base KL reference, and completes three finite nonzero-gradient updates.
  Losses are 3.411e-7, 6.706e-6 and 2.322e-6; reported gradient norms .0004047,
  .002077 and .0006663. This is initially a KL-reference transport/regularization
  control; inspect retained credit and independently check its first loss before
  treating it as a qualified objective result. Start-reference and natural policy
  credit, new-source resume and all other native/release gates remain open.

- [x] (2026-10-02, revision 36) Exercise fresh ordinary TRL jobs on
  AutomationBench simple.slack_office_closure using LFM2.5 Thinking selected
  model commit 95053d21d8e0b7ca99421a2127ae39c64f685ff3, candidate HF sampler
  receipts, rank 8/alpha 16, LR 1e-4, two-episode complete groups and three native
  optimizer slots. External runner bypasses only the public request release
  guard. BF16 Thinking attempt d and FP16 Instruct attempt e both finish three
  optimizer slots and save three sealed checkpoints. All advantages/gradients
  are zero: Thinking truncates every episode before tools; Instruct succeeds in
  every episode. This establishes lifecycle transport, not productive learning.
- [ ] Obtain informative natural complete groups through ordinary jobs, then
  qualify nonzero updates, reference KL and native continuation with the new
  self-contained recovery artifacts. Uniform outcomes cannot close this gate.
  Revision 45 closes ordinary TRL LFM policy-only informative continuation in
  BF16/FP16 under full_determinism; combined reference-KL and broader algorithm/
  model coverage remain separate gates.

- [x] (2026-10-02, revision 35) Identify missing regular Transformers sampled
  probabilities in selected TRL post13. Create clean fork candidate worktree
  /home/hammad/projects/trl-engine-generation on codex/hf-sampled-policy-scores,
  based on d97a2cf619f94c7076a720116d75f85dfd62382b. Add opt-in native generation
  receipts and framework adapter opt-in only for resolved collection. Four fork
  tests pass; pinned native container passes all 21 fork/framework tests, including
  the renderer case missing from the local isolated runtime. New sampler-warp
  rejection tests subsequently pass locally; the container archive predates those.
- [ ] Publish the fork after fresh native GPU qualification, then update immutable
  consumer pins, lock/runtime and both ledgers in the required order. Existing
  pins remain unchanged. Fresh ordinary jobs must use the candidate source until
  adoption; selected post13 cannot provide the new opt-in HF receipt capability.

- [x] (2026-10-02, revision 34) Compose ordinary synchronous TRL execution with
  ResolvedTRLJob in backends/trl/policy_job.py and policy_optimization.py. Reuse
  native loading, optimizer, precision, cancellation and checkpoint callbacks;
  bind retained population restoration to the selected runtime, scoring,
  template and recipe. Preserve behavior-policy versions through synchronous
  collection, admission retries and native episode/trace retention. Exclude
  resolved recovery sidecars and raw rollout evidence from serving exports.
- [x] (2026-10-02, revision 34) Run the actual selected native Verifiers bridge
  and production collector on retained LFM episodes in the pinned container:
  two episodes, three contexts, 1253 actions, two resolved updates, applied
  offset 3 and attempt offset 5. Verify native episode and trace policy stamps
  both retain step 3. This CPU replay does not generate or optimize a model.
- [ ] Qualify this ordinary TRL branch with fresh collection, native optimizer
  execution, reference scoring and checkpoint continuation. Public launch stays
  guarded; distributed, vLLM correction, filtering/refill/curriculum and selected
  semantic/truncation masks still reject in this candidate job composition.

- [x] (2026-10-02, revision 33) Retain original evidence bytes on
  NativePopulationInputs and verify their digest during admitted-population
  construction. Shared save_population_recovery copies those exact bytes as
  posttrain-native-population.bin with the native-rollout-evidence component
  role before sealing. from_checkpoint without an external resolver requires
  that sealed component. Relocation restores identical native inputs; modified
  evidence rejects before decoding. Legacy callback-only checkpoints retain
  explicit resolver compatibility. Focused isolated native suite: 46 passed;
  root admission/input/collection tests: 23 passed, two optional skips.
  Job-level runtime/collection/resume composition is still open.

- [x] (2026-10-02, revision 32) Add TRL collect_resolved_population and an
  explicit retain_native collection mode in policy_rollouts.py. Reuse ordinary
  rollout execution, group admission and rollout observations; return original
  native rollouts before legacy credit/flattened-row projection. Retain, verify,
  admit and submit the replay artifact before returning the population.
  Reject missing conditioning/artifact support, implicit selections, distributed
  collection and native active refills before collection; asynchronous resolved
  collection also rejects. Focused root tests: 25 passed, two optional skips.
  Selected legacy/native handoff tests in the isolated ML runtime: 11 passed.
  The ordinary optimizer job still needs to invoke this collector through
  ResolvedTRLRun; this is not a completed production launch gate.

- [x] (2026-10-02, revision 31) Add
  AdmittedNativePopulation.from_retained_artifact: verify local retained bytes
  against their declared digest, require native replay metadata, decode with
  native schemas and bind logical artifact names instead of machine paths.
  Preserve policy_update_context_contract through VerifiersBridgeSnapshot
  serialization/reconstruction. Focused root tests: 22 passed, two optional
  Torch skips. Real CPU retained LFM checks for SAMPO/GRPO/DAPO preserve 1,253
  sampled coordinates and three contexts; both native backend factories receive
  equal resolved contracts and independently checked credit. Ordinary job
  collection, promotion/resume resolution and execution wiring remain open.

- [x] (2026-10-02, revision 30) Implement decode_native_population using native
  Verifiers Episode/Trace schema models for uncompressed population JSONL.
  Reject missing/duplicate trace identities before native default generation.
  Eleven focused tests pass; scoped Ruff/Pyright and nine import contracts pass.
  Real retained LFM evidence restores three original contexts and 1,253 sampled
  actions through each native artifact format in the pinned CPU runtime.
  The runner, frozen source archive and result hashes are outside Git. Ordinary
  job artifact resolution/admission/execution is still not connected.

- [x] (2026-10-02, revision 29) Add Verifiers bridge retain_population and
  integrations/verifiers_population_artifact.py: snapshot complete original
  episode envelopes verbatim for admitted trace IDs before optimizer execution.
  Preserve sibling native traces without adding loss samples. Content-addressed
  JSONL publication uses fsync and an atomic exclusive link; retries verify
  existing bytes, missing/duplicate/incomplete records reject before publication.
  Seven focused durability/membership tests pass, including the actual bridge
  method preferring native episodes over derived observation rows. Scoped Ruff,
  Pyright (zero errors), all nine import contracts and git diff --check pass.
  This supplies the collector's
  artifact boundary; production job handoff and native artifact decoding are
  still open and the public launch guard remains.

- [x] (2026-10-02, revision 28) Implement matched native forward rounds in
  update_distribution.py:resolve_score_rounds and graph-retaining distributed
  execution in backends/policy_update_distributed_execution.py. Unequal/empty
  ranks pad with admitted context graphs carrying zero objective weight. Verify
  native inputs and shared old/reference/correction identities before model
  collectives; keep global reported objective separate from native backward
  carrier. Each rank performs the same number of native forwards and one
  backward per update. Native trainer/FSDP admission stays closed.
- [x] (2026-10-02, revision 28) Real two-process CPU DDP with forward-buffer
  synchronization, full causal contexts, FP32 trainable parameters and FP32,
  BF16/FP16 activations: 24 cases per rank, two SGD transitions per case across
  four objectives and unequal/empty ownership. Rank gradients have identical
  hashes; losses/gradients/transitions pass independent same-parameter references.
  Missing native artifact on rank 1 makes both reject before forward. Retain 137
  sources, exact runner, log and two rank receipts outside Git. Focused native
  tests: 33 passed; root distribution tests: 10 passed, two optional Torch skips.
  Scoped Ruff/pure Pyright, nine import contracts and diff checking pass.
- [ ] Current completion gate: connect durable fresh collection, resolved
  scheduling/execution and artifact-based resume to ordinary Posttrain TRL and
  veRL job entrypoints. Requests still reject all explicit policy_updates;
  private fixture adapters are not a production launch path.
- [ ] Current completion gate: qualify actual native distributed trainer/model
  paths, synchronized FP16 overflow/scaler and applied counters, interruptions,
  rank ownership in checkpoint seals and distributed resume. CPU DDP seams do
  not qualify native TRL/veRL FSDP multi-GPU execution.
- [ ] Current completion gate: finish the explicit algorithm/backend capability
  matrix and live native coverage for reasoning masks and real granular scorer
  credit; obtain informative Gemma 4 optimizer evidence and natural SAMPO groups.
  Reject combinations without required statistics or verified execution paths.
- [ ] Current completion gate: wire resolved update/counter/mask/denominator/
  score-version/dependency-cost evidence through RunContext and Observatory,
  with immutable membership artifacts and no duplicated global reports.
- [ ] Current completion gate: finish required fork release/publication and
  consumer adoption in dependency pins, lockfiles, runtime images and ledgers;
  resolve installed TRL metadata and candidate veRL release identity mismatches.
- [ ] Current completion gate: run the full validation/release ladder and real
  integration commands in the resolved runtime, reconcile skips/failures, retire
  compatibility paths and remove public guards only for qualified selections.
- [x] (2026-10-02, revision 27) Add pure exclusive original-context score
  ownership in update_distribution.py and private distributed score collectives
  in backends/policy_update_distribution.py. Gather detached FP32 scores after
  coordinated identity/coverage checks, preserve global ratio dependencies and
  denominators, and route complete-objective adjoints to owned replay under DDP
  gradient averaging. Empty ranks require native graph anchors; finite decisions
  use a shared collective. Native TRL/veRL distributed admission stays gated.
- [x] (2026-10-02, revision 27) Actual two-process CPU Gloo/DDP qualification:
  16 cases per rank across GRPO, DAPO, SAMPO and sequence-coupled objectives,
  BF16/FP16 sampled scores, unequal 1/4 ownership and empty 0/5 ownership.
  Global loss and gradients match independent full-objective references; maximum
  gradient absolute error is zero. Single-rank malformed coverage makes both
  ranks reject; one-rank nonfinite decisions propagate to both. Retain 136 source
  files and two raw rank receipts outside Git. Focused native math suite: 38
  passed; root distribution tests: 7 passed, 2 optional Torch skips. Scoped Ruff,
  pure Pyright, nine import contracts and diff checks pass.
- [x] (2026-10-02, revision 26) Qualify private sampo-turns@1 native
  mid-population recovery for LFM on TRL and candidate veRL, BF16 and FP16.
  Four three-update controls checkpoint after update 1; four fresh processes
  resume the remaining two updates. All remaining events and 24 final adapter
  matrices match exactly within each backend/precision, including old scores,
  counters and final scaler. Twenty physical updates apply with no skips.
  Verify 135 staged sources, 139 copied artifacts and four typed turn-row
  checkpoint payloads. Focused native suite: 42 passed; nine import contracts
  and diff checking pass. Both consumer tooling pages now record the bounded
  evidence and retained release gates. Public launch,
  distributed execution, informative Gemma and runtime release gates stay open.
- [x] (2026-10-02, revision 25) Resolve the bounded LFM BF16 native
  discrepancy through observed scores and exact adapter initialization, without
  changing production math. Two unmatched three-update arms reproduce the
  discrepancy; two matched arms give identical scores, ratios, clipping, losses
  and all 24 adapter matrices at every boundary. Both matched arms clip
  [0,741,0]. Five native turn-row unit tests pass, including JSON transport.
  Retain all diagnostic tools and raw receipts outside Git. Broader native
  qualification, new-variant resume and production wiring remain open.
- [x] (2026-10-02, revision 24) Amend canonical primitives/APIs and ADR 0020
  before adding explicit SAMPO turn rows. Reinspect clean author ARL-Arena
  a25a2a229c85431b421ac785fa5f375a99b2072a: rollout_loop.py:227 flattens
  active agent steps; core_algos.py:1015 uses geometric row ratios, local token
  derivatives and seq-mean-token-mean. Add sampo-turns@1 and opt-in
  policy_updates.objective_variant=turn-rows, combining turn-wide ratios with
  equal-turn reduction. Preserve existing episode objectives and complete-group
  credit. Reject other algorithms and semantic masks for this new selection.
- [x] (2026-10-02, revision 24) Independent unequal-length row references
  compare loss, BF16/FP16 gradients and replay adjoints, demonstrate episode
  cancellation versus turn clipping, and verify partial-turn dependency closure
  and typed transport. Expanded native suite: 92 passed before the additive
  transport assertion; that assertion subsequently passes the root 15-test
  resolution slice. Scoped Ruff/Pyright and diff checks pass.
- [x] (2026-10-02, revision 24) Native LFM TRL/veRL BF16/FP16 turn-row
  objective campaign completes eight three-update full/split arms: 24 applied
  updates. All 135 staged sources and 24 raw artifacts verify. Full FP16 arms
  clip every action at update 2, producing zero policy gradient while Adam
  momentum continues to change parameters. BF16 full arms clip 741 (TRL) versus
  945 (veRL) actions: matched numerical/parameter audit remains necessary before
  asserting native cross-backend parity. Workstation idle afterward.

- [x] (2026-10-02, revision 23) Add private shared host composition in
  backends/policy_update_admission.py. AdmittedNativePopulation reads back
  retained native artifact bytes, resolves complete groups, validates all
  native inputs and checks each projected sampled token against the original
  graph. Fresh sampling must match the current applied boundary. Checkpoint
  admission verifies the selected native seal before reading original evidence
  and restores frozen credit without a new estimator. Both backend factories
  consume the same admitted contract and offsets. Seventy-one isolated native
  tests pass; scoped Ruff/Pyright, nine import contracts and diff checks pass.
- [x] (2026-10-02, revision 23) Enforce canonical fresh-task uniqueness in
  update_evidence.py: different complete prompt groups cannot repeat the same
  example_id in one population. Correct test fixtures to use distinct task IDs
  for distinct groups; retain a direct duplicate-task rejection test. Declared
  frozen update reuse and legacy collection behavior are unchanged.
- [x] (2026-10-02, revision 23) Exercise current shared admission on retained
  real native LFM AutomationBench evidence: SAMPO, GRPO and DAPO each verify
  1253 sampled actions over three turns and produce two identical resolved
  occurrences for the TRL/veRL factories. Credit agrees with independent
  scalar references or the existing native SAMPO estimator. All 133 staged
  sources and transferred result/log hashes verify. This is CPU native
  projection/admission coverage, not fresh collection or a new GPU optimizer run.

- [x] (2026-10-02, revision 22) Qualify typed mid-population restoration on
  native LFM TRL and candidate veRL, BF16 and FP16. Four three-update controls
  and four fresh-process two-update resumes from checkpoint 1 complete twenty
  applied updates. Resume reconstructs sealed credit/occurrences without
  rebuilding that population's credit. All 24 adapter matrices and remaining
  update receipts match each same-backend/precision control exactly; old scores,
  scheduler/counters and FP16 scaler state match. Every arm clips 512 actions at
  update 2. All 134 staged sources and 139 retained artifacts verify; GPU idle.
- [x] (2026-10-02, revision 22) Broaden train-package tests in the isolated
  native environment: 856 pass, 23 skip, four failures are inspected. Repair
  the stale KL-reference legacy fixture to exclude the newly additive absent
  policy_updates field; all 15 KL-reference/reward-recovery tests then pass.
  Remaining full-suite failures concern missing renderers, installed TRL
  distribution metadata and historical veRL release-name selection; the full
  release validation gate remains open.

- [x] (2026-10-02, revision 21) Add framework-neutral update_transport.py
  typed reconstruction of retained populations, credit, objective contracts,
  update occurrences and packs. Backend load_retained_population verifies the
  native recovery seal and selected identity before returning the contract;
  recovery does not recompute credit. Native TRL CPU interruption/resume tests
  now reconstruct retained records rather than calling the credit builder.
  Twenty-two isolated native tests pass; 56 expanded root tests pass.
- [x] (2026-10-02, revision 21) Verify typed reconstruction against eight
  existing native TRL/veRL BF16/FP16 GPU checkpoints, including offset 2
  populations. Native component seals, credit and ordered update digests all
  verify. This is existing-artifact validation, not a fresh GPU resume run.

- [x] (2026-10-02, revision 20) Add shared
  backends/policy_update_inputs.py: NativePopulationInputs authenticates retained
  native artifact bytes, preflights every original context and sampled action,
  and revalidates mutable graphs on every scoring/replay read. Hosts supply
  native-format decoding; no transcript reconstruction or parallel trace store.
  Eleven tests cover input identity, digest tampering, action/context references,
  unsupported positions, boolean graph indices and mutable graph changes.
- [x] (2026-10-02, revision 20) Exercise that reader on real retained LFM
  traces and native model updates in TRL and candidate veRL, BF16 and FP16.
  All four arms apply three optimizer updates. Within each backend/precision,
  all update receipts and 24 adapter matrices equal the previous callback
  campaign exactly. This does not qualify cross-backend optimizer equivalence,
  fresh collection or checkpoint-host composition. GPU is idle afterward.

- [x] (2026-10-02, revision 19) Retain and independently verify the completed
  native LFM veRL continuous-population campaign: four three-update BF16/FP16
  full/split arms and two fresh-process one-update continuations. All 346 staged
  source files and 47 artifact files verify. Both precision resumes reproduce
  all 24 adapter matrices, final update receipts and scaler state exactly.
  Old scores remain frozen within populations and refresh at applied offset 2.
  Full arms clip 512 actions at their second update; split arms clip none.
  This qualifies the published acad521 candidate under its private image and
  recorded deterministic profile, not the selected production runtime.
- [x] (2026-10-02, revision 19) Add native GRPO/DAPO scalar-group credit before
  optimizer scheduling. Preserve complete-group means, sample standard
  deviations, group/batch/no scaling, epsilon 1e-4 and existing reward shaping;
  retain raw native rewards unchanged. Explicit native context projection
  supplies turn coordinates beyond SAMPO. Independent references plus selected
  veRL estimator checks: 33 root tests pass (3 native skips); all 11 scalar
  tests pass in the isolated actual veRL runtime. Public requests remain gated.
- [x] (2026-10-02, revision 19) Run the current generic native bridge and
  GRPO/DAPO scalar resolution against retained real LFM traces: two episodes,
  three turns, 1,253 sampled actions, all three normalization modes. Independent
  reward references agree and masks/context records remain unchanged. This is
  CPU native-trace composition evidence, not new optimizer qualification.


- [x] (2026-10-02, revision 18) Add shared
  backends/policy_update_lifecycle.ResolvedPolicyRun collection/restoration
  boundaries. Native TRL's dataloader carries run-global slots; collection occurs
  at loss evaluation after prior applied work, not during prefetch. Keep one
  Trainer.train call and one optimizer/scheduler across populations. Restore the
  unfinished retained population before native model/state loading. The host
  still owns native-input reconstruction; production host wiring remains open.
- [x] (2026-10-02, revision 18) Native TRL CPU tests cover four updates across
  two populations and resume at both the population boundary and midway through
  the second population. Model parameters, Adam state, scheduler and old scores
  match uninterrupted execution exactly; unfinished work is not recollected.
  ResolvedVeRLRun consumes the same collection contract on one retained engine;
  BaseEngine protocol tests preserve optimizer identity and overflow attempt
  offsets and reject wrong prior counters before scoring. Combined native,
  protocol, score/recovery and overflow slice: 21 passed.
- [x] (2026-10-02, revision 18) Recheck direct SSH workstation GPU occupancy;
  preserve all existing processes (none present). Run actual LFM2.5-1.2B LoRA
  TRL continuous fixed-evidence replay: BF16/FP16 by full/split schedules, three
  applied updates per arm (12 total), plus two physical fresh-process resume
  updates from shared checkpoint-2. Each precision's final 24 LoRA matrices,
  final update receipt and scaler state match its native branch bit for bit.
  All processes are terminal and GPU is idle. Verify and retain 107 raw files
  outside Git. This qualifies this private continuous TRL seam, not production
  collection, native continuous veRL GPU, distributed or Gemma optimizer paths.

- [x] (2026-10-02, revision 17) Separate local population cursors from prior run
  applied-update/attempt offsets in both native adapters and recovery identity.
  Seal v2 recovery/population sidecars; accept legacy v1 only at zero offsets.
  Recovery/native/overflow/protocol slice: 38 passed. Continuous TRL dataloader
  resume across populations remains unqualified; offsets alone do not close it.
- [x] (2026-10-02, revision 17) Recheck the reported historical SAMPO loss,
  telemetry and observation-anchor issues against selected d97a2cf source and
  native trajectory evidence. Preserve sequence clipping with the already-fixed
  token-local derivative. Suppress discarded native GRPO advantage statistics
  when supplied credit is active and emit sampled-token SAMPO credit means,
  magnitudes and sign fractions with token counts. Do not average population
  standard deviations into a purported run-wide standard deviation.
- [x] (2026-10-02, revision 17) Read pinned AutomationBench adapter 11f4d712
  and sibling e193bce code before retaining UUID normalization. Hash complete
  ordered preceding observation bundles with observation-bundle@2 provenance.
  Count native validation-error text and JSON error objects as failed tool
  episodes alongside success:false. This remains an observation proxy, not a
  recovered environment-state identifier. Frozen older evidence is unchanged.
- [ ] Qualify this anchor recipe on naturally informative native SAMPO groups
  and retain native environment-state evidence before claiming GiGPO state
  equivalence or improved training quality. The historical step-50 percentages
  supplied in chat have not been independently reproduced in this revision.

- [x] (2026-10-02) Read the complete plan template, inspect the revised proposal,
  relevant canonical contracts, consumer settings, reward projection and backend
  entry points, and exact sibling checkout identities.
- [x] (2026-10-02) Recheck author SAMPO's veRL minibatch/optimizer loop and separate
  existing native capability from Posttrain's present scheduling constraint.
- [x] (2026-10-02) Specify contracts, migration, incremental milestones, native
  reuse decision gates, precision, recovery and release acceptance in this plan.
- [x] (2026-10-02) Amend canonical README/02/04/05/06 and record ADR 0020 before
  implementation, preserving existing algorithm identities and native replay.
- [x] (2026-10-02) Add pure action/context/span/population records, span assessments,
  external detached-credit validation, schedule resolution and capacity-checked
  pack planning. New and existing focused regressions: 80 passed.
- [x] (2026-10-02) Inspect pinned TRL d97a2cf and dereferenced veRL post8
  ef1c377 source for native scheduling and precision hooks; see audit below.
- [x] (2026-10-02) Add closed objective formula definitions, explicit independent
  policy/KL reductions and selectors, sequence value/derivative supports, a private
  Torch evaluator and current-score version validation. Add original-row adapters
  delegating to existing SAMPO/GDPO/CAPO estimators. Isolated Torch CPU suite: 39
  passed, including BF16/FP16 tensors; framework regression slice: 91 passed,
  seven Torch-dependent skips, subsequently executed in the isolated runtime.
- [ ] Establish compatibility references and algorithm/backend capability matrix.
- [x] (2026-10-02) Add strict opt-in catalog settings for optimizer schedules,
  execution budgets and semantic selectors. Preserve absent-selection validators
  and reward-contract digests. Keep launch fail-closed until native qualification.
- [x] (2026-10-02) Preserve original native conditioning paths, sampled-node to
  completion mappings and exact context digests. Assemble complete named groups
  into action populations and original credit rows. Revalidate native inputs
  before causal logit projection. Focused framework/environment slice: 115 passed;
  isolated Torch objective/scoring slice: 30 passed, including BF16/FP16 gradients
  at score temperatures 1.0 and 0.7 with the independent 1/temperature derivative.
- [x] (2026-10-02) Verify reconstructed inputs against retained real AutomationBench
  provider requests: five native calls and 298 sampled actions match exactly.
  External receipt recorded below; this does not qualify an optimizer backend.
- [x] (2026-10-02) Add full-context model scoring, detached population-bound old/
  reference scores and a graph-retaining resolved-loss bridge for native adapters.
  CPU operator tests execute two parameter transitions under one/two packs for
  GRPO/SAMPO in FP32/BF16/FP16, compare loss at identical parameters and independent
  gradient/state histories. Combined scoring/objective/execution suite: 39 passed.
  This is a callable loss seam, not a qualified native trainer executor.
- [x] (2026-10-02) Connect resolved fixed populations to native TRL's existing
  Trainer.train/backward/optimizer/scheduler lifecycle without native generation
  repetition. Four native CPU loop tests pass at exact consumer d97a2cf.
- [x] (2026-10-02) Run actual Qwen3.5-0.8B LoRA native TRL updates on retained
  AutomationBench inputs in BF16/FP16, with one versus two optimizer minibatches.
  Each arm applies two updates. Full-population BF16 initially OOMs in backward;
  expandable allocator segments permit the same graph/objective to complete.
  Initial successful arms total eight applied updates; an additional BF16 control
  adds per-step state/ratio/frozen-score receipts. Broader qualification remains open.
- [ ] Implement trajectory views, spans, named populations and credit transport.
- [x] (2026-10-02) Add resolve_rollout_population to connect complete admitted
  native rows to existing SAMPO/GDPO/CAPO estimators and shared resolution.
  Native SAMPO credit now owns raw-reward shaping before normalization; callers
  cannot bypass it through the lower-level estimator. Version its identity as
  sampo-credit@2. Regression tests retain original traces/rewards, compare full
  versus split credit, and preserve CAPO process positions. Unsupported completion
  truncation masking and truncated structured inputs fail explicitly.
- [ ] Implement deterministic update resolution and scheduling.
- [x] (2026-10-02) Add update_resolution.resolve_policy_population as the shared
  typed-selection boundary. It binds prepared credit, preserves supported
  GRPO/DAPO/SAMPO/GDPO/CAPO objective coefficients, admits SAMPO semantic spans
  explicitly, and capacity-checks every occurrence before returning execution.
  Eight new tests plus the focused contract/credit/evidence/settings slice pass
  (42 tests). This is composition machinery, not public native launch wiring.
- [ ] Implement explicit objective/reduction contracts and reference gradients.
- [ ] Integrate qualified native TRL execution, then veRL execution.
- [x] (2026-10-02) Add from_resolved constructors on both native adapters,
  consuming the same validated typed objective/credit/schedule/execution value.
  Preserve execution budgets and capabilities on ResolvedPolicyPopulation and
  revalidate every occurrence's objective and pack before returning it. Reject
  duplicate native veRL occurrences and invalid TRL score contracts/temperatures
  at construction. Actual native TRL CPU scheduling tests and veRL BaseEngine
  protocol/retry tests execute through the shared resolver: 28 tests pass.
- [x] (2026-10-02) Add complete-score adjoints and per-context replay carriers,
  preserving coupled objective derivatives under native microbatch backward.
  Ten CPU tests cover FP32/BF16/FP16, clipping and replay drift rejection.
- [x] (2026-10-02) Add an explicit private veRL resolved-population adapter over
  native BaseEngine inference/train_batch/scaler/scheduler. Three native protocol
  tests pass, including bounded overflow retry without advancing the cursor/LR.
  The small operator fixture is not a native FSDP/model qualification.
- [x] (2026-10-02) Exercise Posttrain selected-action scoring through the native veRL
  model-output hook; retain full conditioning while avoiding unused probability
  normalization. Initial stock-scoring FSDP attempts exceed local 8 GB capacity.
  Four fixed-evidence BF16/FP16 schedule arms subsequently apply eight updates.
  Independent real-model gradient and broader release qualification remain open.
- [x] (2026-10-02) Run two native Qwen3.5 FSDP applied updates in each of BF16
  and FP16 with two episode minibatches over the retained population. Selected
  action scoring retains full native model contexts, FP32 trainables and exact
  score replay. Frozen old scores agree with the earlier TRL candidate digests
  within each precision. This does not establish backend optimizer equivalence.
- [ ] Implement distributed normalization, atomic recovery and observation.
- [x] (2026-10-02) Add content-bound update_recovery.py seals and private frozen
  score/population transport. Native checkpoint bytes, prepared credit, objective,
  schedule, correction, runtime, world size and applied cursor are bound together.
  Twelve pure tests cover corruption, identity changes and interrupted publication;
  the frozen-score roundtrip rejects pending updates and preserves old scores.
- [x] (2026-10-02) Integrate native TRL checkpoint save/load hooks and verify a
  two-update CPU continuation equals uninterrupted native training after stopping
  at update 1. Five native TRL tests pass, including this checkpoint continuation.
- [ ] Qualify native GPU checkpoint continuation for BF16/FP16 and both backends.
- [x] (2026-10-02) Native candidate veRL Qwen3.5 FP16 shared-boundary resume
  equals same-engine continuation bit-exactly across all 24 LoRA matrices, old
  scores, objective/clipping, scaler and counters. The selected post8 source
  remains unqualified for FP16 resume until the native scaler fix is adopted.
- [x] (2026-10-02) Native TRL Qwen3.5 FP16 shared-boundary resume equals
  same-engine continuation bit-exactly across 24 LoRA matrices, objective,
  clipping, old scores, native/resolved counters and scaler growth tracker.
- [x] (2026-10-02) Native candidate veRL BF16 shared-boundary resume under an
  explicitly recorded deterministic CUDA profile equals continuation bit-exactly
  across 24 LoRA matrices, loss, clipping, frozen scores and counters. Default
  BF16 numerical margins and broader qualification remain open.
- [x] (2026-10-02) Native TRL BF16 shared-boundary resume under the same explicit
  deterministic profile matches all 24 matrices, gradients, objective, clipping,
  frozen scores and counters exactly. This does not establish cross-backend parity.
- [x] (2026-10-02) Add bounded native TRL scaler retry inside the optimizer
  boundary. CPU GradScaler injection checks exact parameters, same-scale
  gradients and frozen scores with dropout enabled; exhaustion leaves the
  occurrence pending, weights/LR/applied counters unchanged. Fourteen optional
  native/score-recovery checks pass, including the existing veRL protocol suite.
- [x] (2026-10-02) Native Qwen FP16 overflow run skips eight attempts, then
  applies two updates. It matches a same-initial-state, accepted-scale control
  exactly across 24 matrices, successful gradients, loss/clipping, frozen scores
  and scaler. Fresh-process post-retry resume also matches exactly, restoring
  ten attempts, two applied updates and scale 65536. Broader gates remain open.
- [x] (2026-10-02) Native LFM2.5-1.2B TRL fixed-evidence checks complete two
  applied updates in all four BF16/FP16 and full/split minibatch arms. Initial
  parameters and frozen old scores match across schedules within each precision.
  Fresh-process shared-boundary resume matches all 24 LoRA matrices, final
  events, counters and scaler exactly in both precisions. This covers the current
  TRL adapter guards; it does not establish fresh collection or backend parity.
- [ ] Complete fixed-evidence and real-rollout BF16/FP16 qualification.
- [x] (2026-10-02) Collect two fresh native Gemma 4 E2B AutomationBench episodes
  in each of BF16 and FP16 on the idle remote RTX PRO 6000. All succeed without
  truncation. Four contexts per precision match provider bytes; all 143 action
  positions exclude prompt/tool context. Complete-group SAMPO advantages are
  all zero, retained as negative coverage rather than productive-update evidence.
- [x] Finish informative Gemma 4 native optimizer checks (R136: BF16 informative
  updates and exact resume; FP16 rejected before update on non-finite logits).
- [x] Publish necessary fork changes, qualify retained artifacts, update consumer
  pins and runtime locks (TRL post14, veRL post9, renderers dev3; kind images
  rebuilt for release 0.4.14). Removing the temporary compatibility route is
  deferred one release per Milestone 7.

## Surprises & Discoveries

Revision134: the 68 native collection tests reached `NativeActiveCollectionBuffer`
through the driver factory but never called `validate_native_selection`, so they
could not reveal that the worker-side guard rejected the very selection SAMPO
manifests require. Pyright also exposed a latent distributed bug: an empty rank
with no anchors passed integer 0 where the carrier requires a graph tensor.

Revision133: native SAMPO already performs the required active rounds and surplus
eviction. The missing framework integration was durable evidence for discarded
groups and final candidate accounting. Existing native hooks support the adapter
without a second scheduler or generic fork change. CPU tests prove those hooks
and failure boundaries; real distributed transport and model updates are still
unqualified. R132 source snapshots predate this new observation adapter.

Revision132: development installation in the existing native qualification image
produces exactly the previously qualified trainer sources and passes the same
five regressions. The installed-asset native suite is still required before
stable adoption; neither uploaded assets nor source identity alone proves its
native optimizer/recovery behavior.

Revision131: normal TRL source distributions omit tests and the fork ledger;
those remain accessible at the immutable Git commit. Both distribution trainer
files match the committed sources exactly. Rebuilding produces new artifact
hashes, so publication binds the final retained bytes rather than an earlier
precommit build. No experimental correctness tools enter the fork commit.

Revision130: the same sealed R124 population now passes all four matched R129
continuations after preserving sampler provenance separately from executor
identity. Changing pack size gives exact saved weights; changing optimizer
schedule is a different update path and does not require identical final weights.
These results do not establish fresh all-FP16 rollout behavior or active SAMPO.

Revision129: Posttrain restore compared every population version to the current
executor's default collection versions, incorrectly conflating sampling
provenance with execution identity. The source seal authenticates the original
sampler and all8340 sampled probabilities; retained correction binds them to
the executor's frozen old scores. Restore must preserve this sampler rather
than inventing a new version or requiring that historical producer to have the
executor's source hash. Old/current/reference and full runtime identity checks
remain strict. The reference adapter used as initial state is included in the
authenticated source seal, as verified directly against its component ledger.

Revision128: replacing collection in an external replay harness must retain
reference-score preparation as well as sampler correction. A beta0 prototype
did not require reference scores; enabling beta.01 exposed that omitted step.
Production admission correctly rejects it. R128 adds the missing native
reference-provider call; this is a harness correction, not a framework defect.

Revision127: precision changes may round random adapter initialization before
PEFT's FP32 upcast. To isolate executor precision, the FP16 replay explicitly
restores the authenticated R124 starting adapter from its frozen reference
checkpoint. Fresh BF16 initialization is independently compared against that
same source. Original actions/context/credit remain unchanged; executor policy
versions and population digest are explicitly rebound. BF16 sampler evidence
used by an FP16 actor is deliberate replay with measured correction, not an
all-FP16 fresh rollout qualification or a recipe-quality comparison.

Revision126: PEFT's mixed root/subdirectory save convention exposes a native
Transformers loader branch that silently skips trained policy weights whenever
a named reference subadapter is present. This is reproduced independently of
SAMPO loss math. Existing recovery source identity hashes grpo_trainer.py but
not its inherited loader, permitting different loader source under one identity.
That omission is now corrected locally; old checkpoints are deliberately not
accepted under the changed identity.

- Revision125: the reported approximately800k changes are predominantly lines
  in local generated research exports, not800k changed files. Existing ignored
  runtimes were not tracked. R125 restores and applies an update but does not
  reproduce the uninterrupted adapter exactly; successful execution alone is
  insufficient evidence of correct continuation.

Revision124: known historical2.6B model produces informative credit and genuine
native clipping in the existing episode-minibatch schedule. Two other groups
remain uniform; preserve them honestly rather than force successes. The adapter
count is32 for this architecture, so the old24-tensor verifier assertion was
invalid. Config-authenticated A/B pairing checks replace that measurement
assumption; production training and saved artifacts are unchanged.

Revision123: first observed Sheets group in R123 has all rewards0, three stopped
malformed Pythonic calls and one length finish. Trello traces execute1-2 tools
but also have malformed calls/truncations and no successful assertions so far.
Current new HF generator uses strict admission, whereas veRL and the existing
Verifiers train client execute named nonconforming calls. The adapter mismatch
is verified in source and corrected with a regression test; it does not prove
that the observed malformed syntax has a recoverable name or causes every zero
reward. Historical example runs actually use LFM2.5-2.6B, not1.2B-Thinking;
R124 resolves that exact historical cached model instead of assuming transfer.

Revision122: historical training groups supply better selection evidence than
task complexity alone. Trello email example step99/batch1/group8 has rewards
[0,0,1,1],2-3 tool calls and538-662 completion tokens; successful episodes gain
assertion progress at assistant turn1. Sheets example step38/batch1/group11 has
[0,0,1,0], one tool call each,503-665 tokens. Airtable example step6/batch1/group33
has [0,1,1,1],0-4 tools,168-822 tokens. These are native groups, not aggregates
over unrelated steps. Sprint retro shows evaluation spread but no qualifying
mixed training group in this archive, so it is not the first fresh choice.

Revision121: retained informative evidence gives nonzero gradients and changes
all24 adapters in both precisions, unlike uniform fresh controls. All148 frozen
base parameters match the authenticated source after precision conversion.
BF16 records1 versus records2 yields exactly equal saved adapter values at both
boundaries. FP16 correction differs from BF16 because old-score execution differs;
each independently matches its own FP64 correction reference exactly. Beta0
means these stages do not qualify a nonzero KL penalty.

Revision120: R119 also has uniform zero reward. Repeating fresh collection for
each check adds cost without providing a controlled comparison. One informative
retained population can support many diagnostics and state-forked comparisons.
The staged replay prototype has not qualified TRL execution or recovery; its
original renderer provenance must survive replay rather than be rewritten.

Revision119: bounded single-action task saturates both ends: reply1024 gives four
successes, reply768 gives four truncated failures. Native group normalization
correctly yields zero advantages in both. Do not keep changing token limits to
chase a nonzero gradient. The next native task has two separately evaluated
actions and therefore natural partial-credit support; actual variation is still
unproven until its traces and checkpoint evidence are inspected.

Revision118: zero corrected TRL gradient is explained by uniform successes, not
masking or loss failure. All four task rewards1 make group-relative advantages0;
reply lengths vary, but lengths do not supply reward credit. Original first calls
have820/714/661/484 tokens, each followed by successful final replies. A separate
reply768 profile exercises bounded completion/truncation without fabricated
rewards. The same native group and public recipe semantics remain unchanged.

Revision117: corrected FP16 continuation retains complete native scaler and
correction state exactly, not merely matching adapter output. This bounded
single-rank LFM profile has informative corrected updates/resume in both actor
dtypes; rollout remains BF16 and broader model/distributed qualification is open.

Revision116: general sibling TRL checkoutc9af78c1 is a different source lineage
from pinned post13 and lacks the dedicated HF receipt delta. Earlier CPU R116
admission did not check this native capability. The selected candidate is the
existing dirty trl-engine-generation worktree ond97a2cf; R117 hashes its exact
delta and verifies the required native signature. R116 must never be GPU qualified
as that candidate. R117 includes the latest local inventory guard; R111 does not.

Revision115: legacy veRL dataset cycling can synthetically repeat one task to
fill multiple prompt groups. The resolved contract requires distinct tasks,
so reject deficient/duplicate inventories at materialization instead of waiting
for rollout population admission. This is an early Posttrain admission correction,
not an upstream sampler defect; native active-refill uniqueness remains open.
Corrected FP16 fresh collection has4750 actions and four episode weights
.67036008/.76670951/1.04142212/.43902348, each exactly matching the independent
reference. It differs from BF16's4476-action population; clipping0 versus3 actions
does not establish a dtype or recipe effect. Both exercise real partial tails.

Revision114: corrected continuation preserves the complete boundary2 state,
including the new sealed correction map, exactly. The external comparator now
asserts exact_match after writing its receipt so a mismatch cannot exit0.
R111 partial physical packs[2,1] demonstrate native tail execution, separate from
the schedule comparison and broader mask/process/distributed qualification.
Separate TRL CPU preflight passes selected admission but installation reports
candidate source metadata1.8.0. Retain this discrepancy as a release blocker;
do not confuse a source-qualified candidate with the consumer's published wheel.

Revision113: actual corrected population has4476 actions and four episode weights
.100/1.196/.348/1.080. One weight reaches the selected lower bound even though
per-token gaps can be modest: sequence correction sums unnormalized log gaps.
This is separate from SAMPO/GSPO's length-normalized current/old clipping ratio.
Three actions hit objective clipping at update2. Fresh population differs from
earlier diagnostics, so changes in losses or weights are not a controlled recipe
ablation. Current independent correction calculation is exactly equal in FP64.

Revision112: correction retention adds private checkpoint JSON that must be
excluded from TRL's model-only export. The explicit export test now includes this
file. TRL and veRL previously duplicated correction preset selection; a shared
recipe helper prevents default sequence/token scope and bounds from drifting.
This does not remove backend collection admission guards or authenticate a new
sampling backend by itself.

Revision111: pre-freezing during collect initially violates the shared lifecycle's
fresh-population invariant. Keep admission score-free and prepare correction in
the first actor update immediately after original old-score freezing; retries and
reuse retain the derived map. Tests independently compare complete episode-product
weights and their stability through simulated overflow and population reuse.
Also clarify dtype evidence: R106/R108 qualify FP16 actor/scaler execution while
their effective vLLM rollout dtype is bfloat16 (native launch log). Do not label
these jobs as all-FP16 inference/training; the new correction addresses this
possible mismatch along with other sampler/trainer numerical differences.

Revision110: ordinary FP16 continuation reproduces complete saved state, including
scaler state and native data.pt, exactly. The live process briefly emitted a
DataLoader-worker-killed message after the update; inspect full export/host
completion and authoritative terminal state rather than infer failure from one
tail. The container exits0, native export completes, and sealed state comparison
passes. This is not evidence for native multi-GPU recovery or corrected SAMPO.

Revision109: correction identity was hashed, but ordinary actor reconstruction
still needed externally supplied values; there was no retained correction
component. Optional sealed JSON closes this transport gap without adding a new
backend storage primitive or changing uncorrected checkpoint identities. It
does not by itself qualify correct sampled-score extraction or selected modes.

Revision108: fresh FP16 emits4420 original selected actions versus BF16's4427.
Both preserve their own frozen populations across updates, but independent
generation does not produce an exact same-evidence dtype comparison. FP16 has no
clipped actions here; no missing-clip bug is inferred. Its saved scaler growth
tracker advances1→2 with scale128 and no skipped updates.

Revision107: the sampler/trainer gap is measurable even with an unwarped sampler;
the raw token weights span.679–1.365 on the retained BF16 population. Normalized
PPO ratios current/old and rollout correction old/sampler serve different roles.
Native TRL computes token correction from old minus sampled scores, sequence
correction from their masked sum; veRL rollout_corr_helper exposes matching
token/sequence aggregation with configurable bounds. The resolved ordinary veRL
host currently supplies no correction map. Default SAMPO must remain guarded
until this selected recipe is integrated and sealed for recovery.

Revision106: full ordinary dense BF16 native resume is exact, not only close in
reported loss/gradient/clipping. All248 compared saved tensors and complete
population payload match uninterrupted boundary2. The failed fourth rollout's
tool result is a real validation error: message=False where the tool requires
a string. It then reaches a length-truncated final assistant call. This is a
policy failure in the benchmark; no masking or scheduler correction is implied.

Revision105: ordinary fresh dense BF16 reproduces meaningful clipping without
external arithmetic or padding wrappers:2 clipped actions in update2. Native
trace stop_condition=agent_completed does not imply absence of truncation; one
final model call has finish_reason=length and1024 sampled tokens. Role masks
exclude system/user/tool and three serialization tokens on each assistant node.
Their sampled support agrees in count with4427 frozen actions; count agreement
alone does not establish exact token/conditioning provenance. The subsequent
external R105 audit closes token-coordinate provenance for this population:
all4427 actions match exactly, native token/mask/logprob/parent fields match the
observed traces, and all8 conditioning references have exact prefix lengths.
Native episode serialization omits default fields, so raw serialized equality
with the separately emitted trace is not expected; compare declared fields and
authenticate the full retained evidence digest instead.

- Revision104: actor-only qualification needs no datasets writes, while ordinary
  launch materializes Parquet through Hugging Face datasets. Read-only HF_HOME
  therefore needs a distinct writable HF_DATASETS_CACHE; R103b fails on its lock
  before model training. Keep source/runtime issues separate from loss/gradient
  evidence and preserve each failed setup attempt.

- Revision103: native worker already supports immutable source snapshots through
  .posttrain-source-revision, and rejects snapshots retaining Git metadata.
  Actor-only probes did not exercise this normal-launch ownership check. R103
  initial attempt lacks the marker and exits before training; no framework fix
  is needed. Verified published archive plus marker follows the existing runtime
  contract. BF16's native default mixed policy is mathematically suitable but
  must be emitted explicitly for the checked profile; raw Hydra cannot override
  framework-selected precision. Both actor and engine compile settings matter.

- Revision102: ordinary native entry still hard-rejects records>1 and always
  advertises ragged cost. Native workers retain micro_batch_size_per_gpu=1 as
  a legacy fallback, but resolved actor transport already supplies explicit
  ordered micro_batch_sizes including partial tails. Reuse that scheduling;
  layout capability and numerical-profile admission are the missing wiring.

- Revision101: replacing global probe setup with scoped native configuration
  preserves the BF16 two-step/budget/layout result exactly, including recovery.
  The arithmetic controls do not remove clipping: update2 has one clipped action
  in each layout. This closes the retained BF16 profile only; ordinary native
  admission and additional model/distributed contracts remain separate gates.

- Revision100: R98 FP16 independently matches both row-count and budget-driven
  packing layouts exactly at both updates, with scale128/no skips and exact
  recovery. Native engine settings are frozen values: recovery tests must create
  a changed configuration with dataclasses.replace rather than mutating it.
  The new arithmetic context holds through backward and restores even when
  native entry or offload cleanup raises; CPU regression covers each path.

- Revision99: native physical-budget BF16 qualification preserves exact updates
  across both row-count and budget-induced layout changes, including checkpoint
  continuation. Independent comparison reports relativeL2=0 and zero differing
  tensors for all four variant/step comparisons. This is retained single-rank
  LFM evidence; ordinary jobs still depend on undeclared external arithmetic
  setup, so this result alone cannot close production admission.

- Revision98: actual fixed population width2079 means2 packed rows cost4158
  slots. Budget4000 must split even where summed original lengths would fit.
  Preflight observes the split, and CPU tests prove over-budget population width
  rejects before allocating the padded native inputs.

- Revision97: existing context budgeting sums original lengths. Two contexts
  of3 and9 tokens cost12 ragged tokens but18 dense tokens. R94's global-width
  control therefore needs explicit physical accounting rather than an external
  input-padding wrapper. Padding geometry changes execution identity, not credit,
  objective support, logical populations or optimizer scheduling.

- Revision96: native recovery identity trackedTF32 and deterministic settings
  but omitted reducedGEMM accumulation and selectedSDPA kernels. These affect
  arithmetic and must invalidate recovery when changed. New identity tests
  demonstrate each independent switch changes job identity without changing
  tokenizer renderer identity.

- Revision95: the maintained opt-in controls reproduce the exact FP16 two-step
  result and exact native/population checkpoint continuation. The correction
  survives model/optimizer/scaler reload rather than only repeated derivatives.

- Revision94: the rowwise correction also removes actual two-update FP16
  trajectory divergence in this retained population: both saved derivatives and
  applied parameters are exactly equal, rather than merely smaller differences.

- Revision93: fixed per-conditioning-row FP32 adapter GEMMs eliminate the
  retained FP16 shared-state discrepancy exactly, without changing backbone
  precision or objective semantics. This does not prove every pack size or
  multi-update trajectory; those remain separate qualification requirements.

- Revision92: evenFP32 adapter GEMMs show small batch-shape rounding changes.
  With nonzero post-updateB weights, their input cotangents can straddleFP16
  rounding boundaries. R90 locates this sensitivity; a fixed per-row GEMM
  control will test the causal hypothesis rather than loosening acceptance.

- Revision90: the residual shared FP16 discrepancy first appears across
  block14 backward even while all retained forward activations match. Exact
  repeated execution does not establish packing-invariant derivatives. The
  trace localizes arithmetic sensitivity but does not yet identify its operation.

- Revision88: shared-parameter FP16 still has a0.0746733% gradient gap despite
  contiguous Linear cotangents, nativeFP32 adapters and common padded widths.
  Thus the BF16 trajectory-sensitivity interpretation cannot simply transfer to
  FP16. Preserve exact repeat evidence separately from packing invariance.

- Revision87: R86 BF16 shared post-update boundary has near-FP32 packing
  agreement under combined external controls. The larger R84 continuation gap
  is therefore not reproduced when both layouts use identical updated parameters.
  This supports trajectory sensitivity as an explanation; it does not qualify
  multistep equivalence or prove the corresponding FP16 behavior.

Revision85's external source generator removed the layout hooks along with
reference weight rounding, so its output does not test the intended combined
profile and cleanup fails. Revision86 restores the controls and asserts their
presence. Failed experimental setup must not be diagnosed as a native-model
failure or used to set precision tolerances.

Revision84's FP16 independent tensors show the first-step gradient difference
2.65705e-7 and parameter difference2.80679e-7, followed by0.8812% second-step
gradient difference. Revision85 removes that parameter-history difference by
using the same real updated adapter tensors for both layouts, while retaining
initial old-policy scores. It distinguishes local packing sensitivity from
trajectory sensitivity; it does not substitute for native checkpoint recovery.

Revision84 confirms that a close first derivative is insufficient to claim
identical multi-step dynamics:BF16 first-step adapter maxdifference6.90e-8
is followed by second-step gradient gap0.07907% and adapter maxdifference9.07e-5.
FP16 also has continuation differences despite no skips. These are distinct
from the causally diagnosed first-step layout effect. Test both layouts from
one shared updated checkpoint before attributing the second difference to
packing or promising a repair.

Revision83's combined control removes both demonstrated sources to approximately
FP32 accumulation scale:2.67064e-7 relative gradient difference,7.45058e-9 maximum
absolute difference. This is far smaller than either isolated control and has
exact repeatability. It supports value-preserving cotangent layout plus explicit
adapter precision as a coherent candidate, but real updates can amplify tiny
gradient differences and must still be independently compared.

Revision82's contiguous-output-gradient control reduces actual nativeBF16
packing gradient sensitivity from0.8654% to0.2343%, with exact repeats and no
parameter changes. Remaining low-precision adapter weight-gradient reductions
are a separate candidate cause. Revision83 combines the independently tested
controls; until qualified, do not describe either as a complete repair.

Revision81 causally reproduces the first native mismatch by changing only
input-gradient operand strides. Transposed singleton rows and materialized
packed rows select numerically differentBF16 GEMM paths. This is a tensor-layout
numerical effect, not evidence of incorrect objective/mask math. Revision82
tests whether contiguous Linear output cotangents remove that source through
the whole native backward; any remaining adapter reduction rounding remains
a separate precision control.

Revision79 excludes the final feed-forward w2 multiplication and locates the
first discrepancy at the convolution input projection's input derivative.
Revision80 shows a contiguous GEMM matches the packed native path exactly but
not the native singleton path, even with equal cotangents and frozen weights.
The candidate cause is stride-dependent GEMM dispatch when batch flattening
materializes a transposed gradient. Revision81 tests this directly; until it
passes, retain that explanation as a hypothesis rather than a diagnosed bug.

Revision78's valid-context trace excludes a loss-adjoint/forward difference in
this controlled first update: all decoder activations and head outputs match
exactly, and head/final-layer output cotangents match. The first trace difference
is below the final block, then accumulates backward through earlier blocks.
Raw padded activation changes must not be interpreted as original-context
changes; independently restricting the retained tensors proves this distinction.

Primary research reinforces the distinction between repeat determinism and
batch invariance: [Thinking Machines, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
uses fixed kernel reduction strategies to align inference, and demonstrates
aligned sampler/trainer training. It does not establish backward invariance for
our LFM/FSDP profile. [FP16 mismatch study](https://arxiv.org/html/2510.26788v1)
reports precision-sensitive RL mismatch; [optimization follow-up](https://arxiv.org/html/2602.01826v1)
also observes FP16 collapse and optimization-dependent effects. These support
separate numerical and recipe qualification, not a universal tolerance or an
automatic FP16/default-LR change. Small runs here validate math, not ablations.

Revision78 expands the diagnostic from aggregate adapter gradients to original
context-indexed backbone activations/cotangents. Same selected log probabilities
do not establish that all intermediate activations or their derivatives match.
The trace therefore tests where disagreement starts, while preserving masks,
sampled action derivatives, native backward and frozen model parameters.

Revision77 disproves the adapter-only repair hypothesis: native per-moduleFP32
LoRA computation leaves0.8360% packing sensitivity, compared with0.8654% before.
Both variants remain approximately4.9% from the whole-modelFP32 reference.
That reference also changes activation arithmetic, so this cannot isolate a
particular backward kernel or prove an objective defect. Preserve the result
as evidence that blanket adapter promotion is insufficient, rather than adding
a production setting that claims to solve packing invariance.

Revision76 confirmsFP16 computation/FP32 storage and stable packing-sensitive
backward. Strict TF32 exclusion leaves the reference unchanged. Gradient sign
analysis explains a route by which small relative errors can materially alter
Adam's first update: its normalized step is approximately learning-rate times
gradient sign when magnitude exceeds epsilon. This is a mathematical mechanism,
not proof that every measured adapter difference is explained by it. Revision77
tests whether native FP32 LoRA leaves reduce the sensitivity without promoting
the whole backbone; per-module FSDP wrapping retains native communication.

Revision75's frozen-embedding storage split lets the native reference complete.
Its gradient packing gap is0.0011274%, versus0.8654% BF16, with exact repeats.
The corrected hook location measuresFP32 adapter gradient computation/storage
in this reference. The source also reveals a remaining oracle control: setting
float32 matmul precision highest does not independently disable cuDNN TF32.
Retain that limitation. The nativeBF16 probe subsequently exits0, measuringBF16
adapter weight gradients withFP32 storage. Optimizer changes are excluded and
same-layout repeats remain exact. FP16 counterpart is now live.

Revision74 fixed-parameter BF16 backwards reproduce the0.8654% packing gap
without optimizer/restoration state changes; same-layout repeat is exactly
equal. The first dtype hook attempt is ineffective because it attaches before
actual FSDP forward views, and post-context grad fields have been cleaned up.
No dtype conclusion is drawn from empty events. The FP32 reference hits a tied
embedding FSDP assertion; the reference-only retry splits frozen storage while
preserving values. This is not adopted as a production model change.

Revision74 source inspection confirms native FSDP MixedPrecision sets the
forward/backward parameter dtype globally, while gradient reduction usesFP32.
An FP32 stored gradient therefore does not prove FP32 gradient computation.
PEFT normally promotes adapter storage toFP32, but that alone does not establish
the effective FSDP/autocast precision. The new hook probe measures the latter
before making claims or changing a trainer policy.

Revision72's fixed-width BF16 control removes the last score difference:
all4482 population actions and both update subsets match exactly across
layouts. This establishes a causal padding-shape control for this retained
LFM profile, not a general model-family guarantee. Actual gradient accumulation
and optimizer transitions still require independent comparison. Population-wide
padding may cost memory and must not bypass declared execution budgets.
Revision73's actual updates show that exact forward scores are insufficient:
first-step gradient discrepancy falls to0.8654%, but after adapter updates
diverge, the second-step discrepancy is11.8095%. This does not establish a loss
bug; fixed-parameter backward and accumulation precision still need isolation.
The matching FP16 update control also exits0 with no overflow or skips. Gradient
differences fall to0.1068% /0.9022%, but nonzero adapter differences remain.

Revision71 isolates score differences without any checkpoint restore or
optimizer change. Both precisions repeat exactly for the same layout, but
default native batch layouts differ. Disabling reduced GEMM accumulation makes
the equal-length pair match; additionally selecting math SDPA makes both update
subsets exact. One population row still differs when padded from1705 to2079.
This narrows the remaining control to padding shape; it does not yet establish
a universally invariant model kernel or certify gradient equivalence.

Revision70's real regression demonstrates that leaf-only LoRA export can add
an untrained vision adapter, not merely fail to reload Gemma. Full paths fix
the regression and actual Gemma reload. The actual checkpoint has zero learned
signal, so identical base logits cannot establish informative training quality.
Revision71 isolates packing with one retained model and no optimizer or restore;
this avoids attributing numerical drift to a loss implementation prematurely.

The revision69 CPU export succeeds but real exported-config/meta-model adapter
injection fails because language-only full target paths were reduced to leaf
names. FP16 is closer than BF16, but gradients and pre-update scores still
differ. Initial adapter equality passes; the evidence suggests batch-dependent
score arithmetic but does not yet locate its kernel or exclude every other
restore-state difference. Both diagnostic handles are now terminal.

Revision 69's actual model check contradicts the CPU fixture's sufficient
equivalence: changing execution layouts changes native BF16 gradients by
8.879% on the first update and 32.756% after the second. At the first packed
update an action clips despite an unchanged starting policy; old-score scoring
and current scoring use different context compositions. This requires direct
score/batch-arithmetic investigation, not relaxed tolerances. Independent Gemma
checkpoint inspection confirms exactly zero adapter change across its two
zero-credit updates. The final export failure has an exact merged upstream
repair (#7610), so reuse that repair rather than implement another scalar fix.
The exporter also infers LoRA targets from leaf names; exported language-only
Gemma topology must be checked before treating a successful save as usable.

Revision 68 confirms that the text-only Gemma target correction permits two
native applied updates and checkpoint publication, but the complete easy task
group gives no policy gradient. Both native metrics report zero credit and
zero gradient norm, so parameter transitions from optimizer regularization
would not prove informative policy learning. The new isolated CPU packing
preflight reconstructs R59's real native evidence with both layouts, including
a partial final pack. The actual native ActiveSamplingReplayBuffer already
implements bounded rounds/oversampling and observation callbacks; default
SAMPO integration should reuse that constructor path with typed setting
validation, while preserving complete groups and explicitly addressing sampler
correction evidence. It should not introduce a competing refill scheduler.

Revision 67 adds execution partition transport rather than another optimizer
loop. The native helper retains its default scheduling behavior when the
optional micro_batch_sizes field is absent. A real two-rank CPU collective
allows different local sizes with equal pack counts and rejects unequal counts
or incomplete coverage on every rank. Framework replay verifies declared
context indices and objective dependency coverage before the optimizer step.
These checks certify the CPU seam, not native GPU packing or distributed model
normalization. The broader test cohort still encounters candidate post8 metadata
versus sampo_hierarchy registry identity; this is a release blocker, not a reason
to weaken the registry check. R66 reaches native checkpoint 1 saving after real
Gemma generation; informative credit and final lifecycle are not yet verified.

Revision 66 exposes a selection issue, not a demonstrated PEFT math defect:
broad projection suffixes select multimodal wrapped layers as well as language
layers. The real Gemma configuration has 50 matching language Linear modules;
PEFT injection into their explicit full paths succeeds on the meta model and
creates 100 language-only trainable tensors. This is construction evidence,
not a native optimizer pass. The native veRL prepare_micro_batches static path
also asserts exact divisibility by micro_batch_size_per_gpu times force_group_size.
Simply raising records from one to two cannot faithfully execute variable
Posttrain packs (for example two/two/one); native planned-boundary transport
must precede removing the framework's current packing guards.

Revision 65 disproves the proposed private-adapter offline flag fix: the real
installed Hub client already honors offline mode, and explicit local-only
resolution also rejects the incomplete snapshot. Missing files are documentation
and Git attributes, not model weights. Completing these cache entries fixes
resolution without production changes. Replacing a foundation variant's artifact
with a local path violates its canonical pinned-base identity; R64 correctly
rejects this before model initialization.

Revision 63 closes single-rank LFM ordinary FP16/base-KL continuation exactly
at both later native boundaries (249 tensors plus driver state). These runs
use different fresh rollout populations from BF16, so their success rates are
not a precision or training efficacy comparison. The pinned Gemma 4 config is
multimodal and the native factory correctly chooses its conditional-generation
class; the new ordinary text-only check must still prove model/optimizer
behavior, not merely tokenizer collection or factory availability. Initial
R63 is correctly rejected by the separate launcher family guard before native
model construction. The R63b external candidate permits only the exact pinned
policy through that gate; this must not be presented as ordinary public Gemma
support. The first attempted launch should have inspected that guard sooner.

Revision 62's real FP16/base-KL run is informative and successful: positive
prepared advantages at updates 1/2 are 0.49990001999600087 and
0.4999000199960008, while update 3 uses -1.4997000599880024. Gradient norms are
0.24550622701644897 / 0.16815118491649628 / 0.49053671956062317 and clipping
fractions 0 / 1/895 / 1/1024. Initial KL is exactly zero and becomes positive
after policy updates. Independently frozen reference vectors and scaler
states verify the ordinary path actually uses the declared base reference and
FP16 scale128. This does not establish an effective regularization recipe or
close the separate ordinary continuation gate.

Revision 61 closes the strict ordinary BF16 continuation discrepancy: both
subsequent checkpoints match uninterrupted native state exactly across 248
tensors plus driver state. This confirms the correction for this exact single-
rank LFM BF16 profile, not all families, FP16, KL or distributed execution.
The first resumed gradient norm is now exactly 0.15685229003429413, unlike the
historical warn-only controls. R61 adds nonzero base-policy KL and FP16 scale128
as the next ordinary profile; fresh collection is still required and may yield
uninformative groups, so launch itself is not qualification.

Revision 60's strict ordinary BF16 job is informative and successful end to end.
At updates 1/2/3, selected action counts are 723/1024/1024, gradient norms
0.6346007585525513 / 0.15685229003429413 / 0.1662064641714096, and clipping
fractions 0 / 1/1024 / 2/1024. Policy losses are -1.4997000694274902 /
0.5004739165306091 / 0.5006630420684814. Beta is zero: this does not qualify KL.
Actual trace inspection distinguishes task failure from missing policy masking:
two episodes spend the entire sampled-token budget reasoning without a tool
call; another sends empty arguments and receives Channel '' not found. The
unsampled assistant rewrite remains masked. These four episodes establish
informative native credit, not a task-quality or recipe-efficacy conclusion.

Revision 57 separates restoration from backward using the ordinary actor path.
The external instrumentation resets the callback's coverage bookkeeping and
the same Python/NumPy/Torch RNG states between three native backward passes;
it records buffer changes and gradient tensors before any optimizer step.
Production source, recovery identity checks and numerical tolerances remain
unchanged. R57 shows exact immediate native load-save equality across 247
tensors, but non-repeatable whole-model backward. The native log specifically
warns that Flash Attention defaults to a nondeterministic algorithm unless
torch.use_deterministic_algorithms(True, warn_only=False) is requested.
The native helper sets warn_only=True. The strict R58 control makes all three
backwards exactly equal, including all 24 trainable gradients, with no buffer
changes. Correct the owning helper's opt-in mode rather than checkpoint state
or objective math. This is a kernel-selection issue; it does not demonstrate
an error in the popular framework's policy loss formula. The fresh R59 ordinary
job uses a new source identity; it does not reuse the historical warn-only
identity or injected diagnostic mode.

Revision 56's ordinary FP16 job succeeds end to end after the canonical metric
fix. Native loss at updates 1/2/3 is 0.4999000132083893 / 0.4995693266391754 /
-1.4995520114898682, with sampled action counts 1024/1024/1162 and prepared
advantages -0.4999000199960008 / -0.4999000199960008 / +1.4997000599880024.
Clipping fractions are 0 / 1/1024 / 0; gradient norms are
0.17041023075580597 / 0.1693841814994812 / 0.5106959939002991. The actual host
observations retain these values plus scale 128 and zero skipped steps.
This is beta=0 policy qualification; it does not qualify nonzero KL behavior.
An inspector run via docker exec ends with 137 when the successful job container
terminates (job exit 0, OOMKilled false). Rerun the CPU-only inspector in a
separate digest-pinned container; it independently verifies all saved bytes.
Full local validation now passes Pyright (zero errors), Ruff and import contracts,
but the catalog regression exposes a real selection inconsistency: existing
lfm26-automationbench-comparison.yaml bindings declare TRL c4d0db while their
trl-fork@current lock declares published d97a2cf. Similar historical bindings
exist in precision-qualification.yaml and verl-vortex-parity.yaml. Resolve these
source/version/lock selections in the required fork-publication and consumer
adoption step; do not relax the reproducibility assertion or relabel historical
qualification evidence as current engine qualification.
Native full_determinism=true does not remove this resume discrepancy: the first
resumed loss matches while grad norm is 0.15650798380374908 versus
0.15652132034301758, and adapter/optimizer differences are independently
confirmed. Transformers LFM2 automatically selects causal_conv1d when available,
but a direct same-process BF16/FP16 primitive probe is exactly repeatable in the
tested shape. That hypothesis is not established. Next separate actual native
whole-model backward repeatability from immediate loaded model/optimizer parity;
do not loosen numerical margins or checkpoint identity in response to the drift.

Revision 55's historical BF16 continuation has exact frozen scores, driver state
and final RNG but unequal model/optimizer states. Update 2's evaluated loss is
exactly 0.5007587671279907 in both jobs, while gradient norms are
0.1831698715686798 versus 0.18312399089336395. Update 3 losses diverge to
0.49967318773269653 versus 0.49952104687690735, and clipping is 2/1024 versus
1/1024. This localizes the first observed discrepancy to backward/update
execution; it does not prove the cause or excuse recovery qualification.
Independent adapter-only analysis compares drift against the actual two-update
change rather than total base-model magnitude: relative L2 drift is 0.08809679689983511.
This is material for this bounded control; do not label it harmless BF16 noise.
The native engine exposes full_determinism (historical profile is false),
including deterministic CUDA settings and disabling nondeterministic fused CE.
Use that existing control for a separately identified repeatability experiment.
Raw comparison tools and receipts remain outside Git under
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/full-verl-r53;
remote receipt is full-verl-20261002-r53b/resume-comparison.json.

Revision 54 distinguishes terminal states: native r53-b performs all three
updates and returns through final merge, but the host rejects both canonical
train/rl/policy_loss and actor/policy_loss mapping to one metric. The compatibility
alias was introduced for native summary parsing; summary now reads the canonical
total loss instead. Additional resolved counters/support metrics were missing
from canonical passthrough, and scale metrics used a noncanonical prefix.
These observation gaps are corrected without changing the loss. Historical
recovery qualification must retain its old source identity despite that outer
replay defect. A DataLoader worker-killed warning appears during Ray teardown;
it is not the outer failure cause and its cleanup behavior remains to audit.

Revision 53 exposes a real backend translation mismatch, not a PEFT/model bug:
TRL's private parser understood comma-separated target_modules but the veRL
Hydra override passed the original string unchanged. The resulting PEFT regex
matched no modules. A shared neutral parser preserves the existing selection
meaning; single strings remain sentinels/regexes, CSV becomes a list. Native
r53-a actor initialization now succeeds before vLLM startup.
Actual ordinary r53-a execution collects rewards [1,0,0,0], commits two updates
and saves both sealed boundaries. V1 fit unconditionally calls TransferQueue
kv_clear even for the empty batch returned by a retained-population update;
TransferQueue rejects empty keys. This is a recipe lifecycle gap rather than
an objective/optimizer failure. Native cleanup now checks allocation existence.

Revision 52 ordinary execution exposes two framework composition defects missed
by captured-dispatch tests: native validate_config requires explicit
need_reference_policy/need_critic results, and a runner class closing over a
trainer class closes over the live RunContext's thread lock. Constructing the
native recipe/context inside the runner fixes serialization without changing
objective meaning. Earlier staging omitted Jinja resources and used a lambda
environment factory; the qualifier now includes package resources and uses
NativeVerifiersEnvironmentFactory. The subsequent Ray runner reaches dataset
initialization, where HF_HOME under a read-only model-cache mount requires a
separate writable HF_DATASETS_CACHE. None of these failures reached an optimizer.
Attempt f reaches native actor construction but the launcher selects fsdp2 while
the resolved worker admits only previously exercised fsdp. The selected branch
now explicitly selects fsdp and one-record ppo_micro_batch_size_per_gpu; legacy
selection remains unchanged. FSDP2 remains an explicit qualification gate.

Revision 51's real Hydra-composition test found that the native infer microbatch
dataclass field is absent from the YAML, requiring an additive override prefix.
It also found legacy TRL-parity override generation enabling native admission
retries even for explicit resolved populations. The selected branch removes
that legacy override and supplies zero retries, preserving complete receipts for
normalizer admission; legacy jobs retain their existing behavior. Asynchronous
host artifact publication requires a separate stable source copy: emitting a
native directory then rotating it could otherwise race provider delivery.

Revision 50's initial fixture entrypoint checks rejected correctly because the
legacy fixture advertised two training devices. The explicitly single-rank
composition test now selects one rank/device; production admission was not
weakened. Runtime identity tests verify reproducibility at unchanged settings,
output-directory independence and changes on effective precision or template
changes. The native reference API provides disable_adapter, so base-policy
ownership can reuse that context rather than constructing another model.

Revision 49's first TaskRunner subclass test failed because the exported
TaskRunnerV1 is a decorated Ray ActorClass, not a plain Python class. Ray forbids
inheriting that object. Exposing a plain native base preserves recipe reuse
without private __ray_metadata__ access. Native fit also has a dataset-derived
epoch termination guard, even with explicit total_training_steps; repeated
optimizer minibatches could therefore stop before the selected update budget.
The resolved dataloader hook uses one complete optimizer window for that guard
while native collection retains its real iterator state.

Revision 48 catches a launch-context injection defect introduced in revision
47: strict dataclass settings cannot be revalidated from model_dump() Python
dicts. Context injection now validates VerlRunContext independently and copies
it into the already-validated immutable launch manifest. The worker JSON
roundtrip reconstructs strict typed settings correctly, and the factory test
exercises that path. The observer tailer requires record IDs; the transport uses
fresh UUIDs and preserves its failed-delivery cursor for retry.

Revision 47 native _save_checkpoint writes latest_checkpointed_iteration.txt
before the complete resolved job seal can exist. Recovery now discovers sealed
global_step directories and treats that native pointer as advisory, so a crash
between the dataloader write and final seal cannot admit a model-only boundary.
Native _load_checkpoint previously warned and restarted when data.pt was absent;
the resolved branch rejects that incomplete recovery before loading anything.
Checkpoint pruning, remote promotion and async saves are explicitly rejected
until the complete sealed-job artifact publication owner is connected.

Revision 46 confirms two native lifecycle contracts that need explicit
composition. V1 fit increments global_steps before collecting, so generation
must temporarily use the committed actor version rather than the pending
optimizer slot. Native SAMPO replay construction also enables failed-group
refill even when the sampler option is false. The resolved trainer constructs
the existing native ReplayBuffer without filtering or implicit refill, leaving
complete-group admission to Posttrain. Native filter/active-sampling/custom
sampler, skip, critic/teacher and legacy rollout-dump combinations are rejected
by this initial composition. The public guard remains in place.

Revision 45: identical adapter restoration does not imply deterministic backward.
The nondeterministic diagnostic resumes with 24/24 identical pre-update weights
and identical parameter ordering but 22/24 different gradients. Explicit
deterministic algorithms remove the retained-occurrence discrepancy entirely.
No loss/credit formula or optimizer mapping correction is justified by this
evidence. Prior ordinary runtime identity omitted effective arithmetic controls;
new checkpoints bind those controls to prevent a silent recovery-mode change.
The first native full_determinism groups contain two successful episodes and
zero credit in both precisions. A complete four-episode BF16 group contains three
successful tool episodes and one zero-reward response without executed tools;
it gives informative fixed-evidence continuation and active clipping. New groups
are separately collected evidence, even when their first assistant content is
identical; their timestamps and resulting population identities differ.
The FP16 four-episode group is uniformly successful. Its eight-episode extension
produces five successes and three length endings: all three selected updates
have nonzero credit, including a negative truncation update. Both remaining
updates reproduce exactly after checkpoint-1 restoration. No FP16 clipping or
overflow is observed in that bounded arm; these outcomes are not manufactured.

Revision 44 continuation failure: identical resumed scalar loss and retained
evidence do not establish equal backward or optimizer behavior. Checkpoint 2
has matching optimizer parameter shapes and step 2 in both runs, but 22/24 first
moments differ; the final two entries match. This is not yet evidence of incorrect
optimizer parameter mapping or an upstream defect. External diagnostic runs
capture named trainable parameters before each update and gradients after native
backward to separate restoration, backward arithmetic and repeatability.

Revision 44: Thinking's 512-token budget cut off reasoning or an unfinished call,
not an optimizer failure. At 1024, the zero-reward response stops after a malformed
Python call with an unterminated text string; native provider_state marks
malformed_structure and no tool executes. The successful sibling makes one tool
call and a final response. This naturally informative group supports measuring
policy credit and clipping without fabricated rewards; it is not an optimal
recipe ablation or a performance comparison between response budgets.

Revision 43: the legacy veRL launch payload does not retain every original
TrainingLoop/algorithm field (including max_length and warmup_ratio). A resolved
worker must consume the full typed selection instead of inferring defaults.
Native V0/V1 driver loops count rollout/training populations; actor minibatching
and native engine scheduler steps have their own boundaries. Wiring resolved
applied updates into those loops therefore requires explicit driver/actor
coordination and policy synchronization; merely replacing the actor loss or
changing ppo_mini_batch_size would not satisfy the contract. This is an intentional
native lifecycle distinction, not an upstream math defect.

Revision 42 verifies selected veRL agent_loop.py collects all extra_fields as
one-dimensional numpy object arrays in DataProto.non_tensor_batch. No fork
change is needed to carry JSON episode receipts. The earlier local veRL test
skip was an import-path issue, not missing source/dependencies: explicitly adding
the exact sibling candidate source to PYTHONPATH runs all nine BaseEngine tests.
Optional torchtitan/veomni/automodel/megatron engines remain unavailable and emit
warnings; none is required by this tested BaseEngine path.

Revision 41: ProducedArtifact freezes metadata as MappingProxyType; Pydantic's
default serializer cannot reliably encode it. The receipt explicitly serializes
local artifact fields and converts other retained immutable mappings to objects;
unsupported objects still fail. Typed JSON round trips retain native coordinates,
while artifact content is authenticated separately. Existing agent-loop group
identities were explicit only for GDPO/CAPO, so the resolved collection option
must also request uid/session identities for GRPO/DAPO/SAMPO.

Revision 40 finds that the existing veRL algorithm-payload builders omitted
policy_updates, and _plan enforced the legacy full-population batch equality
even for explicitly resolved scheduling. Public request guards prevented normal
users from reaching this incomplete path, but internal candidate launch would
lose the selection. Preserve the normalizer's contract across the process
boundary before binding the native resolved lifecycle. No upstream veRL math
defect is inferred from this Posttrain adapter gap.

Revision 39 confirms an observation defect rather than a loss-math defect:
the native windowed total loss was normalized under policy_loss. In the retained
revision-38 KL-only first update, the independent 80-digit calculation yields
3.4113379867831256e-7, versus native 3.4113378433175967e-7 (relative difference
about 4.21e-8); all prepared advantages are zero. Revision-39 first qualification
attempt exits before model loading because its external source archive omitted
the environment package. Preserve the failed log; stage all workspace source
packages and retry in a fresh output directory, without changing job semantics.

Revision 37 first resume attempt used floating beta 0.0 while the original
external fixture supplied integer beta 0. The complete serialized selection
fingerprint differs; strict runtime admission rejects before restored native
work. Preserving the original fixture's exact beta representation succeeds.
This is a selection-identity sensitivity, not evidence of native loss failure.
Revision 38 finds PEFT initialization occurred before native Trainer seed setup;
the resolved path now seeds earlier. Old-source checkpoint continuation is
qualified using retained revision-36 source, without weakening source identity.

Revision 36 preserves three qualification setup failures before optimizer work:
attempt a receipt serialization missed datetime; attempt b used a mismatched
model revision/cache root; attempt c explicitly pinned a placeholder agent
client, so native Verifiers intentionally preferred it over the supplied policy
context. Reading the selected native env.py confirms that pinned agent fields
override context fields. Attempt d leaves policy model/client unpinned so it
inherits the host-managed client. These are runner corrections, not demonstrated
framework or Verifiers math errors. Failed native episodes are retained.
Completed fresh traces explain the degenerate credit: all four Thinking
episodes terminate at length 256 with only reasoning and reward 0; all four
Instruct episodes perform the Slack tool action then finish with reward 1.
Both groups therefore have exactly zero centered GRPO advantages. Do not label
these successful lifecycle runs as learning evidence or a BF16/FP16 comparison:
the model variants and outcomes differ. Saved native checkpoints confirm zero
prepared advantages (512 actions/two contexts for Thinking; 240 then 267
actions/four contexts for Instruct). All six checkpoint component seals verify
after relocation. Informative credit and resume remain open.

Revision 35 source inspection finds that _generate_single_turn returns None
for regular Transformers log probabilities, while native trace projection
requires token-aligned scores. Current upstream GRPO source has the same behavior;
an upstream source/issue/PR search did not locate a directly reusable opt-in path.
This is an integration capability gap, not an upstream loss-math error. The
candidate requests output_scores and return_dict_in_generate only when asked,
normalizes actual processed sampling scores in FP32, and applies the token EOS
mask to scores. It does not substitute a later actor forward for sampler evidence.
Holding vocabulary scores per generation step adds memory cost requiring bounded
real-model qualification. Continuous Transformers batching remains unsupported.
The job also rejects nucleus/top-k/min-p and repetition/presence warpers until
sampler correction is implemented; omitting correction is not a valid shortcut.

Revision 34 finds that synchronous collection supplied no behavior-policy span
to the modern native bridge. Strict engine admission would reject freshly
collected episodes. RolloutBatch now carries that optional span; retries preserve
it, the synchronous collector checks the unchanged native applied boundary, and
the bridge stamps it before retaining native records. The actual native CPU
replay verifies both episode and trace stamps. Legacy callers retain None.
The native trainer already owns reference models and PEFT reference adapters;
the candidate host reuses those paths rather than inventing another KL reference.
That scoring path still needs real native qualification. The task selector uses
an explicit applied-step/seed digest and distinct complete task groups; it is
not claimed to reproduce legacy dataloader ordering.

Revision 33 optimizer-job audit finds a missing cross-machine recovery input:
the native checkpoint sealed scores and population sidecars but did not carry
the original rollout artifact bytes. Private recovery callers supplied those
bytes externally. A transferred checkpoint could therefore lack its original
conditioning graph despite valid native model/optimizer state. Checkpoints from
admitted NativePopulationInputs now seal those original bytes alongside native
state; generic callback-only fixtures and historical checkpoints keep their
existing external resolver path. This adds no new transcript representation or
product meaning. The filename is .bin because the host decoder owns the native
format, which need not be JSONL for every admitted source.

Revision 32 ordinary TRL collection already performs group-atomic admission
and streams all attempted trace evidence. Preserve it rather than constructing
a second collector. The new raw-return branch occurs before legacy structured or
SAMPO advantage computation; the resolved engine computes credit from original
complete groups. Root legacy rollout selection initially reports five passed
and one failure because accelerate is absent, not a demonstrated behavior
regression. The same selected legacy tests plus new handoff/reference tests pass
in the isolated ML runtime (11 passed). Its source tests precede the final added
artifact-corruption case; root final handoff tests cover that case explicitly.

Revision 31 finds that VerifiersBridgeSnapshot omitted the bridge's
policy_update_context_contract, so a portable worker reconstruction silently
returned to legacy collection without native conditioning records. Carry that
optional field through both snapshot writing and create; None remains the
legacy default. Two serialization regression cases exercise both selections
without activating the native environment. Admission from a retained artifact
uses the artifact's logical name; using its absolute workstation path would make
cross-machine recovery impossible. Artifact promotion is a separate host duty.

Revision 30 verifies that the actual retained LFM envelopes validate directly
through Verifiers Episode.model_validate, and legacy trace rows through
Trace.model_validate. No chat reconstruction is necessary. The developer uv
runtime lacks Verifiers; the isolated local ML runtime also lacks renderers.
Therefore the real schema qualification used the retained workstation container,
not mocked native classes. Mock classes in four focused tests only prove invalid
identity rejection happens before schema defaults; they do not qualify decoding.

Revision 29 collection audit: the existing Verifiers bridge appends untouched
native episodes before projection, but finalize publishes the cumulative gzip
only at run completion. A resolved update cannot reference that still-changing
run file for mid-population recovery. retain_population freezes matching entire
envelopes without reconstructing chat or copying derived trace payloads. The
host must publish its returned ProducedArtifact and resolve that reference before
AdmittedNativePopulation.from_rollouts; this host handoff is not implemented yet.

Revision 28 finds that correct score ownership alone does not provide native
model-call coordination. DDP forward-buffer synchronization is itself a
collective; a rank with fewer or no contexts cannot simply skip forwards.
resolve_score_rounds revalidates the ownership mapping against the original
update and schedules one original context per rank per round. Short/empty ranks
use an admitted context/action as an anchor whose carrier weight is zero.
Anchors never enter gathered objective scores. Actual gradient contribution
remains limited to owned score derivatives, with global reduction preserved.
This trades extra zero-weight forwards for a defined dense execution path;
other sharding/packing strategies require separate native qualification.

compute_distributed_graph_loss validates capacities and old/reference contracts,
materializes original inputs before model collectives, and coordinates preflight
failure across ranks. Shared identity includes actual frozen score bytes,
correction values, temperature, execution budget and capabilities. It retains
owned model graphs, gathers global detached scores, prepares existing global
adjoints, adds zero-weight graph anchors and returns both the reported objective
and a separate native backward carrier. Model wrapping, optimizer steps, scaler
transactions, checkpoint seals and population collection remain native caller
responsibilities. Native communication errors are not replaced by a custom
optimizer loop or silently accepted. Local preflight exception causes remain
available on the rejecting rank.

Actual CPU Gloo/DDP checks use a small cumulative causal model with synchronized
forward-count buffers, three native context views, six sampled actions, and a
mixed-sign external prepared-credit fixture. Ownership is 1/2 contexts or 0/3;
both ranks perform two or three model forwards respectively before one backward.
GRPO, DAPO, SAMPO and sequence-coupled cases use complete global weights and
ratios. Each dtype/definition/layout case applies two SGD transitions. Direct
references calculate causal sampled log probabilities, geometric ratios,
clipping and global 1/8 versus 1/4 (or DAPO 1/6) weights without production
objective helpers. The second update exercises four clipped actions in some
cases; initial updates have ratios one. All actual rank gradient hashes agree.

FP32 maximum gradient absolute error is 2.08616e-7, parameter-transition error
1.19209e-7 and minimum gradient cosine .99999988. BF16 maxima are .00390625 and
.000390619, minimum cosine .99997944; FP16 maxima are .00048828125 and
.0000488460, minimum cosine .99999976. These observed errors belong to this
small CPU autocast/DDP fixture. They are not universal backend tolerances.
Partitioned half-precision backward rounds before FP32 rank reduction, unlike
the one-graph monolithic reference. The initial -a diagnostic compared continuing
trajectories after their parameters had already differed slightly; its second
loss comparison failed by 2.90424e-5. The corrected -b and final -c checks compare
each transition at identical current parameters, retaining the independent
gradient and parameter-transition differences instead of declaring exact parity.
No tolerance was widened to make that loss comparison pass. Source/diagnostic
roots are retained separately; -a is a failed check, not native success evidence.

Final evidence root is
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/distributed-graph-r28-c.
Its source manifest binds 137 retained framework/fixture files and runner SHA;
artifact-manifest.json verifies runner.py, run.log and two raw rank receipts.
Torch is 2.13.0+cu130; execution is CPU Gloo with FP32 trainables and autocast
FP32/BF16/FP16 activations. Earlier -a/-b roots remain outside Git. The exact
final runner is retained inside -c; repeats must use a fresh evidence root.
All processes are terminal. No GPU, pretrained model family, real scorer,
native TRL/veRL trainer, synchronized scaler retry or distributed checkpoint
is qualified by this fixture. These remain the current completion gates above.

Revision 27 separates score ownership from loss normalization. Multiplying a
weight by world size alone does not prove that actions are owned exactly once,
that an episode ratio sees scores from another rank, or that an empty rank
participates in backward. resolve_score_ownership binds every required original
conditioning context to exactly one rank. The complete global term is retained;
even unselected ratio dependencies remain assigned. Rank partitions are distinct
from optimizer minibatches and cannot introduce local denominators.

The private gather_current_scores seam first gathers plan/version/validity
metadata on every rank. Missing, extra or nonfinite local scores make all ranks
reject before the score tensor collective. Exclusive owned scores are placed in
a common ordered FP32 vector and summed, not averaged. Existing
prepare_score_adjoints then evaluates the complete original objective on detached
global score leaves. distributed_score_carrier routes only owned derivatives,
multiplying by the number of averaging ranks so native DDP's average yields the
original global gradient. Empty ownership requires a finite differentiable
native model anchor multiplied by zero; a detached manufactured zero is rejected.
all_ranks_finite computes a shared MIN decision but does not itself update an
optimizer or scaler. Native adapters still own replay coverage, matching forward
and backward collective schedules, parameter continuity and commit transactions.

Two actual CPU processes use Torch 2.13.0+cu130 Gloo and native Torch DDP over
a small linear score model. Independent references construct clipping, episode
means, token-local derivatives and 1/6 versus 1/4 (or DAPO 1/5) weights directly,
without using the production evaluator or reducer. Each rank runs 16 cases:
four definitions by BF16/FP16 sampled-score casts by unequal or empty ownership.
Both ranks' reduced model gradients match the monolithic reference exactly.
Non-unit ratios exercise one or two clipped actions. Eight rank-0 cases have
no owned scores and still complete native backward. All 16 coordinated bad
coverage checks reject on both ranks; shared finite checks propagate a false
decision from rank 1. These are real CPU collectives and model backward, not
native TRL/veRL multi-GPU training, synchronized scaler retries or checkpoint
qualification. No GPU was launched or occupied for these checks.

Final current-source evidence is retained at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/distributed-scores-r27-c.
source-manifest.json binds 136 retained framework/fixture source files and the
external runner SHA; files are copied under source/ and rechecked against live
files after execution. artifact-manifest.json verifies both raw rank receipts.
Earlier roots distributed-scores-r27 and distributed-scores-r27-b remain separate:
the first precedes added identity checks, the second binds current critical
sources but has narrower source retention. The -c campaign repeats those same
cases with complete four-owner source retention. All processes are terminal.
The external runner check_distributed_scores_r27.py stays outside Git. From
/home/hammad/projects/rl, use PYTHONPATH=packages/train/src:packages/common/src:
packages/data/src:packages/environment/src with
/tmp/posttrain-native-verl-update-runtime/bin/python and that runner's absolute
path. Choose a fresh evidence root before repeating; existing receipts must not
be overwritten. Production regression tests live in test_update_distribution.py;
they cover ownership, global coupled derivatives, replay drift, foreign scores,
rank identities and real versus invalid empty anchors. The existing isolated
native pytest command selecting that file, test_update_objectives.py,
test_update_turn_rows.py and test_update_replay.py yields 38 passed.

Revision 26 reuses the existing native checkpoint lifecycle without a new fork
patch. The explicit sampo-turns@1 objective and complete prepared credit survive
strict typed checkpoint decoding. All four checkpoint-1 sidecars resolve the
turn-row definition, two frozen occurrences, 1253 sampled actions, zero global
offsets and the sealed credit identity; their next occurrence is 1. The restore
callback consumes these retained records rather than running build() or a new
credit estimator to reconstruct the checkpoint. The external fixture still
prepares its original reference population at process startup and later creates
the next declared replay at offset 2; this is not fresh production collection
and does not qualify an artifact-only public composition host.

For each backend/precision, branch controls apply three updates and retain a
checkpoint after the first. Fresh resume processes use a different initialization
seed (935 versus 71), and FP16 starts with scale 32 before native restoration of
checkpoint scale 128. Both resumed events match the corresponding last two
control events exactly, including loss, gradient norm, clipping, parameter delta,
old-score digests, population/version identity and attempts. Final adapter tensors
match all 24 matrices exactly. Final FP16 scaler scale is 128 and tracker 3 in
both backends; native growth intervals remain backend-specific (2000 TRL,
400 veRL). Resume preserves those native settings rather than equating them.
Both resume processes restore population offset 0, retain its old scores for
the reused update, then collect the next declared replay at applied offset 2.
Final applied, attempt and scheduler counters are all 3; no overflow retry occurs.

BF16 controls clip [0,741,0] on TRL and [0,945,0] on veRL; FP16 controls clip
[0,1253,0] on both. This campaign deliberately compares each native backend
with its own continuation using its ordinary initialization. It does not replace
revision 25's exact-initialization cross-backend evidence or diagnose the old
default-runtime Qwen discrepancies. Fully clipped FP16 update 2 still changes
parameters through retained optimizer momentum and resumes exactly.

The local evidence root is
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/turn-row-resume-r26;
the remote root is /var/lib/posttrain/qualifications/turn-row-resume-20261002-r26.
source-manifest.json verifies 135 source files, and transfer-manifest.json verifies
139 checkpoint/receipt/log/adapter files remotely and after copying. summary.json
contains exact continuation comparisons; checkpoint-definition-audit.json records
the four independently decoded objective/credit identities. Model LFM2.5-1.2B
revision df58c174f05ff733f83f8cae10ea9298224c8006, TRL
d97a2cf619f94c7076a720116d75f85dfd62382b, candidate veRL
acad5211619aef5ed80b25e9657262f08a436b03 and image
sha256:f0d32115d6e3bdca1dda76aad0ff8b9605e9869d4413ea65f76f4bc942cc85eb
match the preceding qualification profiles. Rank 8, alpha 16, LR 1e-4,
weight decay .01, temperature .8, deterministic algorithms and CUBLAS workspace
:4096:8 are bound in each runtime receipt. Evidence SHA is
d3c83210e2ffdab0088a3eff6ef1284c4f6dae07e51f2e99133398f6c67c6fea.
Resume peak allocation is about 3.820 GB TRL and 5.669 GB veRL.

External controllers prepare_turn_row_resume_r26.py, run_turn_row_resume_r26.py
and retain_turn_row_resume_r26.py remain in that root's parent outside Git.
Run the retainer with /tmp/posttrain-native-verl-update-runtime/bin/python to
verify hashes and exact comparisons; it requires direct SSH access described
below. Creation/run scripts must use new roots for repeats, preserve all old
receipts and check an idle GPU before every process. All eight processes are
terminal; final GPU state is 41 MiB and 0 percent utilization. From repository
root, the isolated native pytest command documented below with
test_update_turn_rows.py, test_update_transport.py, test_update_admission.py,
test_update_native.py and test_update_verl.py reports 42 passed. Warnings are
missing optional unrelated veRL engines. Diff checking passes.

Revision 25 corrects the initial interpretation of revision 24's clipping
counts: a net count difference of 204 is not one disagreeing 204-token turn.
Actual addressed scores show 820 disagreements: TRL clips a 308-token turn
that veRL does not, while veRL clips a 512-token turn that TRL does not.
The shared 433-token turn clips in both. More importantly, equal random seeds
did not produce identical FP32 adapter initialization across the two loading
paths. All 12 LoRA B matrices initially match; none of the 12 A matrices do.
Initial adapter difference L2 is .00983886265, maximum absolute difference
.0000610332936. Initially A matrices round to identical BF16 values, but after
update 1 their BF16 representations differ by up to .0001220703125. Initial
scores agree exactly because the initial B matrices are zero; identical initial
scores alone therefore cannot prove identical initial trainable parameters.

A controlled rerun supplies the same retained FP32 adapter values to both
native paths before scoring or optimizer work. All 24 matrices then match
exactly at initialization and after each of three updates. All 1253 current and
old scores, all three turn ratios, losses and addressed clipping decisions
match exactly. The second-update ratios are 1.00714313984 (433 actions),
1.00482082367 (308 actions), and .99714809656 (512 actions). The first two
turns clip; both arms report [0,741,0]. Changing only initial adapter values
eliminates this fixture's discrepancy. This supports an initialization cause,
not a universal precision tolerance or broad backend certification. No
production objective, clipping threshold or fork source was changed.

Retained evidence roots under
/home/hammad/experiments/posttrain-correctness/2026-10-01/working are
native-bf16-audit-r25-b (unmatched) and native-bf16-matched-r25-c (matched).
Their remote counterparts under /var/lib/posttrain/qualifications are
native-bf16-audit-20261002-r25-b and native-bf16-matched-20261002-r25-c.
Source manifests bind 135 and 136 inputs respectively; the extra matched input
is canonical-initial.safetensors. Each transfer manifest verifies 20 raw
score/receipt/log/adapter artifacts both remotely and after copying locally.
Both analyses verify hashes before examining evidence. The matched analyzer
also asserts exact score/parameter/loss equality, three applied updates and
zero retries in both backends. Model, image, fork identities, evidence digest,
rank/alpha/LR/temperature and deterministic settings are unchanged from
revision 24. Four successful processes apply twelve updates. A preceding
diagnostic process applies one update before its callback fails because its
args parameter shadows parsed runner arguments; retain that failed source and
three artifacts separately at native-bf16-audit-r25 and remote
native-bf16-audit-20261002-r25. Its failure is diagnostic logging, not a passed
training gate. Corrected campaigns use fresh paths and never overwrite it.

From /home/hammad/projects/rl, use the isolated native Python and PYTHONPATH
documented below to run packages/train/tests/test_update_turn_rows.py (5 passed).
To verify retained diagnostics without a GPU, run the external
analyze_native_bf16_audit_r25.py and analyze_native_bf16_matched_r25.py with
/tmp/posttrain-native-verl-update-runtime/bin/python from the working root
above. Repeating GPU campaigns requires fresh source/output roots and a new
idle GPU check; do not rerun their creation scripts over existing receipts.
The final workstation check reports 41 MiB used and 0 percent utilization.

Revision 24 shows that the user's requested turn-row work requires two linked
objective changes: turn-local geometric ratio support and equal-turn weighting.
Changing only the ratio preserves an episode-weighted objective different from
the inspected row kernel. A five-turn/two-episode fixture with turn lengths
3,1,1,2,1 assigns each turn one fifth of an update's policy weight, regardless
of episode membership. Complete-group credit still precedes minibatch selection.
The new objective is explicit; no existing run switches automatically.

Author-source limits remain explicit: the inspected script requests a SAMPO
estimator missing from its Python registry, so this does not reproduce the full
author training stack. Its inspected GSPO kernel clamps log weights at 10;
Posttrain's named row variant retains the qualified uncapped finite-ratio
contract and rejects nonfinite ratios. Clipping thresholds, advantage
normalization, group filtering and observation-proxy credit remain separate
recipe choices with the provenance caveats recorded in recipe-selection.md.

Raw revision-24 sources/receipts live outside Git at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/native-turn-rows-r24
and /var/lib/posttrain/qualifications/native-turn-rows-20261002-r24.
source-manifest.json and transfer-manifest.json bind 135 source files and 24
raw receipt/log/adapter artifacts; summary.json verifies initial parameter and
old-score equality between schedules within each backend/precision, unchanged
old scores within a population, refresh at applied offset 2, finite events,
parameter changes and exact three-applied-update counters. Model/image/fork
identities, rank 8, alpha 16, LR 1e-4 and temperature .8 match revision 22.

With the selected default SAMPO .003/.004 clipping band, full-population turn
ratios clip [0,1253,0] in FP16 on both backends. The second update's policy
gradient is exactly zero, but parameter delta L2 is .023481 because native Adam
retains momentum from update 1. The initial analyzer incorrectly required
strictly positive gradients; it now accepts zero only at that fully clipped
reused occurrence and records it explicitly. BF16 clips [0,741,0] in TRL and
[0,945,0] in veRL. Split episode-minibatch arms clip [0,0,0] in both precisions;
this is different optimizer work over different subsets, not evidence for an
optimal schedule. Native gradient norms are reported before versus after norm
clipping in the two runners and must not be directly compared. No recipe-quality
or cross-backend optimizer-equivalence claim follows from these checks.

Revision 23 finds two collection-admission omissions: complete groups alone
did not prevent the same task entering another group, and valid native view
coordinates alone did not prove flattened completion tokens agreed with those
coordinates. Shared admission now checks both, plus sampler/applied-boundary
coherence. Pure tests reject changed context, permuted sampled tokens, stale
sampling, malformed offsets and foreign input readers. Native CPU tests verify
recovery never calls fresh population resolution and rejects altered native
checkpoint components before invoking the artifact reader.

Current real native admission receipts are outside Git at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/native-admission-r23-b
and /var/lib/posttrain/qualifications/native-admission-20261002-r23-b.
source-manifest.json binds 133 staged files; result.json records the three
algorithm checks and transfer-sha256.txt verifies the result and raw run.log.
The prior native-admission-r23 staging is retained separately: the final staging
adds the explicit sampling/current-boundary guard before the recorded final
check. Evidence digest and exact image/fork identities remain those in revision
22. No GPU was allocated for this native projection replay.

Revision 22 GPU evidence lives outside Git at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/typed-recovery-r21/gpu-v3,
with raw remote artifacts at
/var/lib/posttrain/qualifications/typed-recovery-20261002-r21-v3.
summary.json independently compares all four native resumes against controls;
transfer-manifest.json verifies every retained native checkpoint and receipt.
The exact image is sha256:f0d32115d6e3bdca1dda76aad0ff8b9605e9869d4413ea65f76f4bc942cc85eb;
TRL d97a2cf619f94c7076a720116d75f85dfd62382b, veRL
acad5211619aef5ed80b25e9657262f08a436b03, LFM
df58c174f05ff733f83f8cae10ea9298224c8006 and retained native evidence
d3c83210e2ffdab0088a3eff6ef1284c4f6dae07e51f2e99133398f6c67c6fea
are unchanged. The deterministic profile, rank 8, alpha 16, LR 1e-4,
temperature .8, source/runner hashes and precision are bound in each receipt.
Resume peaks are 3,819,942,912 bytes for TRL and 5,669,426,688 for veRL.
This qualifies recovery over fixed original evidence, not fresh collection.

Two initial experiment callbacks omitted required explicit world_size and
sampler_correction arguments. Both failed before optimizer-state restoration;
failed campaigns remain separate. Corrected sources use a fresh campaign.
The controller later stopped on a transient utilization reading (41 MiB,
10 percent) after a completed process. Its verified terminal receipts were
preserved, sources reverified, and only missing arms continued after GPU idle.

Broader validation first failed root collection because optional TRL was absent,
then isolated collection because only four workspace package paths were supplied.
Supplying every workspace package source yields 856 passes, 23 skips and four
failures. The KL-reference fixture incorrectly retained policy_updates=None in
its claimed pre-engine schema; production already omits that absent field to
preserve older checkpoint identities. The other failures reflect the isolated
runtime: no renderers package, TRL metadata 1.12.0.post2 with selected post13
source, and candidate veRL version 0.9.0.post8 whose registry differs from the
published historical post8 table. Do not treat this isolated runtime as a
release installation or weaken the immutable fork/source gates.

Revision 21 removes another experiment-specific recovery callback: retained
credit and update identities can be decoded directly from the existing sealed
population sidecar. Trusted dataclass decoding must resolve Python TypeAliasType
annotations and reject booleans as integer coordinates. Both were covered before
the final native tests passed. The sidecar schema and serializer remain unchanged;
the decoder validates normative objectives, digests and reconstructed packs.
Existing GPU artifact checks and their source hashes live outside Git at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/typed-recovery-r21/results.json.
Public host wiring and fresh GPU use of this decoder remain separate gates.

Revision 20 identifies a production composition gap: native qualification
callbacks previously hand-selected trace records. The new shared reader verifies
the full source SHA-256 before decoding, then checks actual physical graph paths,
input digests, context lengths, sampled-token eligibility and artifact references
for every view before any update. Native mutable objects are rechecked on reads.
This is an ephemeral reader over existing authoritative native artifacts, not
an alternate durable trace format. It preserves the already-approved contract;
no frozen baseline amendment is needed. Public collection/recovery composition
must still supply durable artifact bytes and faithful native decoding.

Raw revision-20 GPU evidence lives outside Git at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/native-reader-r20
and its trl subdirectory. Verified staged source counts are 193 (veRL) and 194
(TRL); six raw result files per backend are transferred and SHA-256 verified.
Native veRL acad5211619aef5ed80b25e9657262f08a436b03 and TRL
d97a2cf619f94c7076a720116d75f85dfd62382b run using the same private r19 image,
LFM snapshot, retained sampler evidence, rank/alpha/LR and deterministic profile
documented above. Three applied updates and three attempts occur per arm;
clipped-action counts remain [0,512,0]. Old scores refresh at population offset
2. All 24 trained matrices and per-update events match their own previous
backend/precision campaign exactly. These arms use the full two-episode schedule;
earlier full/split and resume gates remain separate evidence. Current scalar
credit is not thereby model/optimizer qualified: the GPU arms still use SAMPO.
Root reader/credit/evidence/resolution slice: 42 passed; scoped Ruff/Pyright,
nine import contracts and diff checks pass. Expanded real-trace CPU checks cover
all six GRPO/DAPO scaling combinations using this authenticated reader.

Revision 19 veRL continuous GPU evidence is stored outside Git under
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/continuous-verl-r19.
The exact published candidate is acad5211619aef5ed80b25e9657262f08a436b03;
private qualification image is
sha256:f0d32115d6e3bdca1dda76aad0ff8b9605e9869d4413ea65f76f4bc942cc85eb,
derived from cached base RepoDigest
sha256:92883712e8a23beda85f3c2ab04acb5c16e30ae057781d350686c9c58b98f846.
Torch 2.13.0+cu130, Transformers 5.14.1, PEFT 0.19.1, tensordict 0.10.0;
LFM model snapshot df58c174f05ff733f83f8cae10ea9298224c8006. Native sampler
evidence d3c83210e2ffdab0088a3eff6ef1284c4f6dae07e51f2e99133398f6c67c6fea
is explicitly reused; no fresh on-policy collection is claimed. Full versus
split exposures are 3,759 versus 1,994 actions, so this is not matched-work
recipe evidence. Full peaks at 9,973,769,728 bytes and split at 5,668,122,624;
the full configuration exceeds 8 GB. Initial Docker build failed because a
bare image ID was parsed as a repository; using the cached qualified RepoDigest
resolved setup, with both logs retained. No production dependencies changed.
GPU receipts bind staged revision-18 sources; they do not qualify subsequent
scalar-credit or generic bridge changes. FP16 resume restores scale 128 from
the shared native checkpoint. All arms terminate at global applied/attempt/LR
cursor 3; one final declared occurrence remains unapplied at the requested cap.
The source and artifact transfer manifests and independent analyzer retain
these boundaries. GPU is idle after the completed campaign.

Revision 19 scalar-credit integration initially lacked a valid unsupported
OLMo3 test fixture: its existing fixed recipe and active sampling guards
rejected construction before the intended engine gate. The corrected fixture
preserves those guards and proves the scalar adapter rejects that recipe.
Root optional skips are supplemented by real candidate veRL estimator tests;
the added estimator is a Posttrain composition capability, not an upstream
bug fix or a recipe change.
The actual native LFM scalar projection check initially set legacy batch size 2
alongside an explicit execution selection, then used only 8,192 statistic bytes.
Both configurations correctly fail their existing preflight guards. The external
fixture now leaves legacy batching at 1 and declares 1 MiB statistics capacity;
all six algorithm/scaling combinations pass. Both failed logs and current source
manifest remain at the external working/scalar-native-r19 evidence directory.
The original native evidence hash matches the continuous GPU campaign, but the
current scalar adapter source is newer and is not retroactively GPU-qualified.
Scoped Ruff/Pyright, nine import contracts and diff checks pass; the expanded
root update/credit/objective/SAMPO slice passes 79 tests with 10 optional skips.
The same expanded slice passes all 89 tests in the isolated actual TRL/veRL
runtime, eliminating those optional skips. This does not supply missing native
model/optimizer qualification for the newly introduced scalar path.

Revision 18 closes the demonstrated TRL global dataloader skip problem by
retaining run-global slots across populations rather than restarting training
with a population-local dataset. Existing optimizer, scheduler and native retry
hooks remain authoritative. The shared lifecycle rejects collection after an
incomplete population, wrong applied/attempt offsets, reused old scores and
immediate regeneration of a completed population identity. Restored current
population inputs are reconstructed by the host and then checked against the
sealed native recovery identity before loading model/optimizer state.

The real-model campaign uses explicit replay of retained AutomationBench native
evidence, SHA256 d3c83210e2ffdab0088a3eff6ef1284c4f6dae07e51f2e99133398f6c67c6fea,
on LFM model snapshot df58c174f05ff733f83f8cae10ea9298224c8006. Old-score snapshots
refresh at the declared next population; sampler evidence stays explicitly
reused with correction disabled, so no fresh on-policy collection is claimed.
Both full arms clip 512 of 1,253 sampled actions on update two; both split arms
clip zero. Full arms process 3,759 sampled-action exposures versus 1,994 for
split arms. The three-update cap leaves one occurrence unapplied in the final
population; receipts expose local and global counters rather than claiming that
final population completed. Peak allocated memory is about 3.82 GB full and
3.78 GB split. Within each precision the initial trainable state and first frozen
old-score digest agree across schedules. This is not matched-work recipe evidence.

Exact TRL source is d97a2cf619f94c7076a720116d75f85dfd62382b; immutable runtime image
is sha256:92883712e8a23beda85f3c2ab04acb5c16e30ae057781d350686c9c58b98f846.
Framework source bundle f1b7a9adc0a4243c3d10b9c72b2089bbe7d8ee550ae70abd3887e3001c8accb0
contains 344 individually verified files. TRL adapter source hash is
2ab42d55e3ac425693944697b2e99558e492632ed4d7d5b2342c37aeacf2f755;
shared lifecycle hash is cdbb39bc211490171a935bcd1c5fe7891a6979d0882d079e37cf703f28d2e774.
External evidence root:
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/continuous-native-r18.
summary.json independently verifies transfer-manifest.json and all 107 files,
then compares retained safetensors, final update receipts, offsets, scores and
scaler. Docker-created private files initially blocked SCP; only this completed
qualification directory's artifact ownership was corrected, then hashes were
verified against the remote manifest. No global cache permissions were changed.

Revision 17 replays retained native traces rather than assuming UUIDs are
meaningless or meaningful. AutomationBench has generated entity UUIDs. The
historical 908db2a response builder merges templates containing fixed
invocation_id/response_uuid fields; the inspected sibling e193bce response
builder discards those templates. Retained older traces do contain the template
fields, so their presence must not be inferred for newer runs. Broad UUID
normalization remains the existing recipe. An initial attempted removal was reverted after the user requested
source and real-run analysis. Normalized complete observation bundles address
the demonstrated parallel-result omission without removing UUID normalization.
In lfm26-vortex-v5-150-dspark-opt-20260926-r1, the retained file contains 1,892
episodes and 7,667 sampled assistant turns. Within recorded step/batch/task
populations, 3,608 turns follow multiple tool observations; 487 last-response
anchor groups hide different complete bundles. Manual cases include the same
final empty result after either finding or not finding a contact, and the same
last email result after a spreadsheet error versus successful search.
Raw last-response matching yields 3,996 cross-episode comparable turns;
UUID-normalized last-response matching yields 4,029; normalized bundles yield
3,007. Mixed-position matches change from 1,546 to 507. Reduced matching is
not automatically better credit and repeated states may occur at different
positions; no position-based grouping restriction is added. These are OLMo3
training traces used for observation replay, not a SAMPO optimizer ablation.
The old failure detector flags 69 episodes; adding validation-error text alone
flags 178; also including native JSON error objects flags 863. This is an
episode fraction, never a per-call failure rate. Native single-turn TRL traces
(96 episodes) and fresh Gemma4 collection (2 episodes, 4 turns) provide negative
controls with no parallel bundles. Gemma4 still has zero reward variance.
External audit runner and hashed-source results live under
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/
as audit_sampo_anchor_replay.py and sampo-anchor-replay-results.json; they are
not repository assets. Actual retained Qwen/LFM SAMPO advantage autograd replay
at ratio one matches independent token-local derivatives within 1.2e-10 in
FP32; these four episodes contain no mixed-sign nonzero credit within an
episode, so they cannot demonstrate the historical sign reversal. Opposite-turn
native fork regressions separately cover that failure. Results are retained as
sampo-actual-credit-gradient-replay.json, not new optimizer/model qualification.

Follow-up source and actual-run check (2026-10-02): rerunning the retained-trace
audit reproduces these counts. An independent serializer/hash check against the
current production `_observation_bundle_key` matches all 7,667 real turn bundles
with zero mismatches. UUID normalization adds 33 cross-episode comparable turns
over raw last-response keys in this population; this is evidence of an effect,
not proof that every new match is a valid state equivalence. Manually reviewed
turn-1 histories share the same final two-email response while their preceding
spreadsheet result is an error in one episode and a successful match in another.
Reviewed turn-2 histories share a final empty search, while one preceding search
found a contact and the other did not. The complete-bundle fix separates these
cases while retaining UUID normalization. Zoom meeting UUIDs also serve as
lookup keys in AutomationBench; do not conclude that all UUIDs are semantically
irrelevant. Neither raw-ID equality nor UUID blanking alone establishes true
environment-state equality. This follow-up makes no new optimizer qualification
or training-quality claim. `test_sampo.py`: 27 passed. External runner and report:
`working/verify_anchor_real_projection.py` and
`working/anchor-real-projection-results.json` under the same evidence root.


Revision-16 native composition now consumes ResolvedPolicyPopulation directly
through ResolvedTRLPopulation.from_resolved and ResolvedVeRLPopulation.from_resolved.
The value retains execution and capability contracts, not only resulting packs.
Its construction checks all occurrence objective/credit identities and exact
pack plans; replacement of a later objective or pack cannot reach native training.
Both native constructors independently validate all objective terms before
mutation, including direct callers that do not use the typed factory.

The native TRL scheduling regressions use typed SAMPO settings, shared resolution,
then the actual GRPOTrainer CPU loop. veRL tests use the same resolution through
real BaseEngine protocol methods with an explicitly small causal-model fixture.
The isolated source-selected suite test_update_native.py, test_update_verl.py,
test_update_native_overflow.py and test_update_resolution.py passes all 28 tests;
this includes native scheduling, checkpoint continuation, retry and shared
recipe transport. It is not FSDP/GPU/distributed or fresh public-host qualification.
Scoped Ruff/Pyright, all nine import contracts and diff whitespace checks pass.

Remaining production lifecycle gap: the private native adapters still own one
frozen population. Their checkpoint native/global counters currently coincide
with that population's applied count. A continuous collector must preserve a
run-global applied offset and a population-local occurrence cursor, seal current
native evidence/credit/scores, and resume without collecting a replacement for
unfinished occurrences. Do not repeatedly reset Trainer.train or remove the
launch guard merely because fixed-population factories now compose correctly.

Revision-15 SAMPO audit confirms a private adapter mismatch: production TRL's
policy_rollouts shapes rewards before compute_sampo_advantages, while direct
SampoCreditEstimator calls previously consumed unshaped rewards. Equal native
rewards with one truncated episode and penalty 0.5 wrongly yielded zero credit.
Move shape_online_reward into SampoCreditEstimator on copies of original rows,
and remove shaping from the higher-level assembly to prevent double application.
The regression now gives sampled credits 0.4875/0.5 and -0.4875/-0.5. This fixes
the new engine seam; the production TRL shaping path already behaves correctly.
Version the estimator contract because prepared-credit/checkpoint meaning changes.

The new rollout assembly uses population_from_rollouts before estimator calls,
so groups remain complete and credit is frozen before optimizer slicing. CAPO
process evidence stays aligned to native sampled coordinates. Structured native
inputs must be nontruncated, matching the legacy TRL admission contract. The
resolved objective currently has no qualified legacy completion-truncation mask
variant: reject that setting rather than silently optimize all original actions.

Audit exact source, not the unrelated ../trl HEAD: selected TRL d97a2cf already
contains uncapped token-local derivatives for token-aligned SAMPO sequence credit.
The clean published veRL candidate acad5211619aef5ed80b25e9657262f08a436b03
registers sampo_token_credit; selected legacy post8 cannot be treated as that
candidate. Actual native loss/advantage parity tests execute at those candidate
sources in the isolated CPU runtime: both normalization cases pass (2 tests),
covering shaped rewards, masks, correction, clipping and k3 KL. These execute loss
code, not matched native model/FSDP optimizer trajectories or runtime adoption.
The root suite reports 47 passed and one optional module skip; that skipped module
is then executed separately above. Scoped Ruff/Pyright, nine import contracts and
diff whitespace checks pass. Existing runtime/fork gates remain open.
An expanded isolated CPU run executes test_update_credit.py,
test_update_resolution.py, test_verl_sampo_parity.py and test_update_objectives.py:
40 tests pass without skips, including independent objective/gradient references.

Revision-14 Gemma 4 collection uses the catalog-pinned AutomationBench adapter
11f4d712806d292c6c6a752af046f4e16c4f037e, not the unrelated HEAD of the local
checkout. Export all tracked package resources with git archive; the 500-file
bundle is independently verified in the cached runtime before model launch.
Use AutoModelForMultimodalLM and Gemma4RendererConfig(enable_thinking=False) with
native TrainClient, NullHarness and SubprocessRuntime. No Qwen text parser is
used on this route; the inherited local-qwen transport label is only an endpoint
alias. Model identity remains the explicit Gemma snapshot
3e22461f65e89153144f8adb70e3b8c2cc9845a7.

Initial attempts a/b fail before provider requests because the unprivileged
container cannot write /.cache; this pinned Verifiers source derives CACHE_DIR
from Path.home(), so XDG_CACHE_HOME does not redirect it. A writable results
mount at /.cache resolves that failure. Attempt c then fails during native
PEP-723 harness provisioning in the offline container: the pinned helper installs
uv and resolves script dependencies. A CPU-only real prepare_uv_script preflight
passes with Docker bridge networking and a writable /.local results mount. The
subsequent attempt d completes native collection without monkeypatching the
harness. Models remain offline, AutomationBench tools are simulated, and no paid
inference endpoint is used. Retain the terminal setup failures; they are runtime
prerequisite failures, not evidence of a model/math defect.

BF16 receipt gemma4-bf16-native-train-d.json has SHA-256
0e0ab943c132b82ad9bf76450764233de69022ef12f391260438c3396f1578c3.
Its four context lengths are 662, 828, 662 and 827; sampled lengths are 59, 13,
59 and 12. Both native episode rewards are 1.0, both have two turns, and the
existing complete-group SAMPO estimator returns only zero advantage. Do not
fabricate rewards to qualify productive updates. Identical calls in separate
episodes must be compared as a multiset, not assumed to have unique token bytes.
gemma4-bf16-collection-checks-d.json retains the exact input/mask/credit diagnostic.
External tools and receipts are under campaign working/gemma4-native-r13;
remote results are /var/lib/posttrain/qualifications/gemma4-native-r13/results.
The collection runner SHA-256 is
4f1e9efeedde9e8e204174ca6bbcc81e0695be90eca79df1728ae70370db157f.
FP16 attempt e completes successfully after another idle-GPU check. Its two
episodes also produce four calls, 143 actions, the same context lengths, no
truncation and zero SAMPO credit. FP16 receipt SHA-256 is
dc04d12e31d04506e204696dcc45d1efa6864114e97d1436c8e6cf1f227b826c;
gemma4-fp16-collection-checks-e.json retains the context/credit diagnostic.
Both training collection handles (90037 BF16, 50786 FP16) and their analysis
handles are confirmed terminal. Check current GPU occupancy before the next run.

Revision-13 LFM results: each full-population arm makes two passes over 1253
sampled actions (2506 exposures); each split arm makes one pass (1253 exposures).
Both schedules apply two updates. The full-population second update clips
512/1253 actions in both precisions; split arms clip zero. These schedules have
different reuse and update histories, so neither their losses nor clipping counts
establish a recipe preference. Peak allocated memory is 3.811 GB for full arms
and 3.728 GB for split arms on the remote RTX PRO 6000. The initially unwritable
Triton cache attempt applied zero updates; explicit writable result-directory
cache paths permit the retained source and objective to complete.

The final native adapter source hash is
e1a7b618133809854a6cb1990367e90bba5b28e4bbc92f871f57703f44f3ba60;
the staged 556-file source manifest hash is
761b451885f1c57897afc50b2165e2d84d5796f9c5c916e58a5ea18ef6e099ed.
LFM model snapshot is df58c174f05ff733f83f8cae10ea9298224c8006. External
receipts live under the campaign results/native-collection/engine-remote-r11d
directory: lfm-four-arm-summary.json, lfm-resume-comparison.json, and the raw
arm/control/resume JSON, logs and safetensors. The twelve branch/resume artifact
copies have additionally been hash-verified against the authoritative remote
files; lfm-resume-transfer-verification.json retains those hashes outside Git.
BF16 and FP16 initial states differ; within-precision equality is the evidence.

Public integration audit: requests._validate_online_rl still rejects explicit
policy_updates. Production policy_optimization._run_online_rl creates a legacy
GRPOTrainer with policy_rollouts.rollout_function and lets native generation
schedule collection. The private ResolvedTRLPopulation instead consumes one
frozen population. A production collection-to-population lifecycle must connect
these paths, retaining native traces, prepared credit, sampler correction,
reference scores and recovery before removing the launch guard. Merely swapping
trainer classes would leave collection and resolved occurrences unsynchronized.
The new update_resolution.py boundary now replaces hand-authored objective and
schedule assembly for supported typed settings. It consumes already-prepared
credit; it must not derive advantages from a minibatch or treat a scorer's raw
quality values as advantages. Caller-supplied capabilities remain qualification
claims to validate against the selected native runtime. The legacy launch guard
is retained until collection, native lifecycle and recovery call this boundary.

Revision-12 family preflight: the retained LFM BF16 two-episode group supplies
no informative SAMPO credit, while its FP16 group supplies 1253 actions in three
original conditioning views and advantages -0.975, 0.5 and 0.975 under the
existing estimator. Reuse that exact FP16 evidence for both training precisions;
this avoids changing tasks or fabricating rewards between arms. It does not
qualify fresh on-policy collection. The external engine_family_population.py
is a source-parameterized copy of the retained fixture assembler.

A remote source bundle must include native package resources, not only Python
modules. Missing Verifiers judge text and TRL chat-template files initially
prevented import. Preserve the first terminal training log lfm-bf16-full.log;
that attempt exited before model load/update. The corrected bundle verifies 556
file hashes and imports GRPOTrainer from the exact overlaid source path
/qualification/trl-source/trl/trainer/grpo_trainer.py. Installed metadata still
reports post11; source identity is independently d97a2cf, not inferred from that
version string. The fixed-evidence path does not use vLLM generation, so its
installed-version warning is not vLLM integration qualification.

Remote staging lives at
/var/lib/posttrain/qualifications/hierarchical-engine-20261002-r11-a on
carbonteq-ai-workstation.lan. Local archives and manifests are in the external
campaign working directory. Base engine-remote-r11-a.tar.gz has SHA-256
aa00e3368e87fd0fcb8982ea3c9b3f922487fcf2a014221fd132642ccdc0dded;
resources-b adds Verifiers text with SHA-256
127f8a0bc2ba2eeb0e01017499cc376211df1d0609903fb4b505940f4ce457f2;
resources-c adds tracked TRL resources with SHA-256
31e265d156b56d41dac169fb965130b85c2a32d06b458817b049d304a59673ec.
Retain source-manifest-a.json and -b.json and the current source-manifest.json.
The current engine_native_trl_family.py runner uses CLI-selected model/evidence,
the manifest's exact source commit and selected native loader. All runtime/model
mounts are read-only, only qualification results are writable, and network is
disabled. A fresh two-update BF16 full-population control is launched as Docker
container hierarchical-engine-lfm-bf16-r11c; confirm its actual state before any
subsequent launch and preserve results/lfm-bf16-full-c.log and eventual JSON.

Container r11c terminates before an update because Triton attempts to create
/.triton under the container's unprivileged UID. Retain its failed log. Retry
under fresh container hierarchical-engine-lfm-bf16-r11d, with explicit
TRITON_CACHE_DIR=/qualification/results/triton-cache-d and
TORCHINDUCTOR_CACHE_DIR=/qualification/results/inductor-cache-d. Keep the source
and model mounts read-only and caches inside this qualification's writable results;
do not change host-wide cache permissions. Its log/receipt suffix is full-d.

Revision-11 retry validation caught an adapter error: recording random state
before the initial old-score freeze was too early when dropout was enabled.
That freeze consumes randomness once, whereas retries reuse its detached
scores. Capture stochastic state immediately before current scoring, after
freezing old scores. With this correction, a CPU native GradScaler skip followed
by retry matches control parameters and same-scale gradients exactly over two
updates with dropout enabled. Comparing inverse-scaled gradients from different
FP16 loss scales is a different numerical question and is not required to be
bit-exact; the test compares the actual accepted scale instead.

The initial real Qwen FP16 scale-1024 control applies two updates without a
skip. Keep engine-trl-fp16-overflow-retry.json as negative coverage evidence;
its name does not qualify retry. A fresh high-scale run uses 2^24 and an explicit
17-retry ceiling to exercise native scaler overflow while retaining two applied
updates. Runners and receipts remain outside Git. Match the eventual control to
the retry's accepted scale, not to its overflowing startup scale.

The high-scale run succeeds after eight native skips: two applied updates use
ten backward attempts, while global_step and scheduler.last_epoch are both two.
The accepted scale is 65536. Its same-scale control and fresh-process resume
match all 24 LoRA matrices exactly, as well as losses, clipping, frozen scores,
successful gradients and native scaler state. Resume starts with scale 32 and
a different seed, then restores the saved scale/counters. Receipts under the
external campaign results/native-collection directory are
engine-trl-fp16-overflow-{high,control,restored,comparison,resume-comparison}.json.
Runtime identity is
52de873f8558fae9cfd140f30b4dd454ee6aad58838fe090b4100f1be042400a.
The failed first-attempt gradient norm in the raw high-scale receipt is NaN;
it is undefined, not a usable gradient measurement. The strict finite comparison
receipt compares successful gradients only. Preserve that raw failure evidence.
Later source guards reject retry preparation without a completed skip and reject
unscaled nonfinite gradients before mutation; their native CPU regressions pass.
These later guards change source identity, so do not relabel GPU receipts as a
final release qualification of the subsequently edited source.

Direct SSH preflight (2026-10-02) verifies carbonteq-ai-workstation.lan at its
pinned host key, with dstack and the existing ai-infra/.state/ssh/dstack_worker
identity. The RTX PRO 6000 reports driver 595.91.07, 97887 MiB total, 41 MiB
used and no compute process. Recheck before launch; this observation is not a
reservation. Cached immutable snapshots include Gemma 4 E2B
3e22461f65e89153144f8adb70e3b8c2cc9845a7 and LFM2.5-1.2B-Instruct
df58c174f05ff733f83f8cae10ea9298224c8006 under
/var/lib/posttrain/cache/huggingface/hub. Cached runtime image digest
sha256:92883712e8a23beda85f3c2ab04acb5c16e30ae057781d350686c9c58b98f846
has Torch 2.13.0+cu130, Transformers 5.14.1, TRL post11, PEFT 0.19.1,
Accelerate 1.14.0, Verifiers dev93 and renderers post1.dev2; it has no installed
Posttrain, veRL or AutomationBench adapter. Therefore cached models and GPU
capacity do not yet qualify either engine family path. Supply a separately
identified source/runtime bundle rather than changing the running worker or
silently claiming this older image implements the selected dependencies.

Revision-10 inspection found that the private checkpoint writer retained the
resolved population, but the neutral seal validator required only native files
and frozen scores. Require all three evidence roles so another writer cannot
publish a resume boundary without its original masks, credit and schedule.
Private preflight validation includes the population role before writing sidecars.
Sixteen pure recovery tests pass, including missing-role and changed-population
rejection. Native GPU receipts above bind the pre-hardening source bytes;
subsequent source qualification is recorded separately rather than relabeling them.

The locked Transformers 5.14.1 trainer loop invokes on_optimizer_step before
LR scheduling and global-step advancement. It skips LR scheduling on overflow
but increments global_step afterward even for a skipped step. The revision-10
Posttrain callback raised first, preserving the resolved cursor, without retry.
The revision-11 retry finishes inside that boundary, reruns the
same original contexts and frozen scores with restored stochastic state, reuses
native training/backward/clipping/scaler hooks, and exposes attempts separately
from applied updates. Merely clearing the pending marker and continuing the
outer dataloader would lose an occurrence. This is an intentional native counter
contract to adapt, not evidence of an upstream framework arithmetic defect.

Revision-9 audit: selected veRL post8 does not persist its FP16 scaler in native
checkpoints. Published candidate acad5211619aef5ed80b25e9657262f08a436b03 already
passes the scaler to its checkpoint manager and saves/restores it in extra state.
The candidate checkout is clean and origin/codex/posttrain-math-parity resolves
to the same SHA. Reuse that candidate for native FP16 resume qualification; no
duplicate fork fix or consumer pin change is justified. Candidate publication
does not satisfy runtime asset and stable consumer adoption gates by itself.

Revision-8 native veRL attempts distinguish setup from objective failures.
The isolated runtime had drifted to Transformers 5.16.1/tokenizers 0.23.2;
the repository runtime locks select 5.14.1/0.22.2. The newer Qwen wrapper failed
on an absent convolution attribute. An external version-pinned target overlay
restored the selected dependency versions without modifying existing virtualenvs.
With these versions, the actual native model reached probability computation
but exhausted 8 GB in full-context log_softmax. Native activation offload also
failed at that point. Neither attempt applied an optimizer update. Earlier TRL
candidate receipts did not bind Transformers/tokenizer versions and must be
rerun against the locked runtime before release qualification.

The veRL protocol fixture initially used plain TensorDict slicing, which cannot
slice jagged tensors on the batch axis. Using native index_select_tensor_dict
fixes the fixture; this is not a newly discovered veRL training defect.
The isolated pytest environment also needs PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
because an unrelated installed plugin metadata entry names a missing module.

The public GRPOSettings and SAMPOSettings validators in profiles.py currently
require per_device_batch_size multiplied by gradient_accumulation_steps to equal
num_prompts_per_step multiplied by num_generations. This ties collection size to
execution accumulation. GDPO/CAPO instead have a divisibility check; their
compatibility behavior must be inspected separately rather than generalized
from SAMPO.

Posttrain's veRL build_hydra_overrides sets actor_mini_batch from prompts per
step and describes one optimizer update over the generated rows. This is an
adapter policy, not absence of native veRL minibatching. Author ARL-Arena at
a25a2a229c85431b421ac785fa5f375a99b2072a splits collected turn rows using
ppo_mini_batch_size, loops over ppo_epochs and calls its optimizer step inside
the minibatch loop. Its WebShop script selects minibatch 128 and microbatch 8.
Native row units and world-size/generation multiplication must be resolved before
translating those numbers. The source has an unavailable script entrypoint;
inspection does not certify reproduction of the author's full run.

The experimental TRL checkout already routes sequence-ratio values through local
token derivatives when advantages are token-aligned. That behavior must be
preserved and independently tested; this plan does not treat it as a newly found
upstream error. Scalar losses can agree while gradients differ.

Canonical APIs explicitly defer semantic segmentation. Supporting reasoning
spans therefore requires a baseline amendment; silently broadening
assistant-turns@1 would break an existing projection identity.

Implementation audit, 2026-10-02: veRL post8 is an annotated tag. Its tag object
3de3d2c7ae5eb3f4868679dac48c373de680d757 is not the source commit; dereferencing
`carbonteq-v0.9.0.post8^{commit}` yields ef1c37715fa75de5973ae5b3c398383cd7e0093d.
The ADR skill's suggested docs/technical/adr/ADR.md is absent; the actual repository
template docs/templates/ADR.md and current docs/decisions records govern.

The inspected native CISPO definitions differ: TRL caps detached weights only
above an absolute bound; veRL's inspected function uses two-sided bounds and a
log-ratio cap. `cispo-upper@1` explicitly names the former formula. It must not be
mapped to veRL's existing CISPO name without a native equivalence gate or required
generic implementation. This is a definition difference, not an upstream bug.

Empty selected support needs explicit reduction-domain evidence. ContributionRef
now retains original domain actions independently of selected actions and scoring
dependencies. Otherwise an episode with no selected policy tokens would disappear
before `zero` versus `omit` could determine equal-episode weighting. Domain records
need not receive loss or be scored. Tests distinguish weights 1/2 and 1 for those
policies and independent policy/KL support in different episodes.

Native Verifiers physical parent paths preserve each original call's context,
including roots replaced by summaries or rolling windows. The existing terminal
training branch may flatten selected assistant siblings; it is not proof of later
calls' conditioning. The opt-in projection follows original parent links instead
and retains native coordinates without adding a token store. Materialization
recomputes the path, eligibility and digest before exposing ephemeral inputs.
Only causal-text@1 is implemented; custom positions/attention and multimodal
inputs require separate qualification.

SAMPO's existing estimator requires contiguous prompt groups. The new assembler
groups rows by their explicit first-seen prompt identity, preserving within-group
order, before delegation. This avoids conflating same-example independent groups
or accidentally centering interleaved groups together. Episode IDs and rollout
occurrence IDs are distinct: reward evidence is checked against the stamped
posttrain_rollout_id, not posttrain_episode_id. Legacy structured projections
currently stamp branch 0; explicit branch binding remains a native integration
gate, not evidence of arbitrary-branch process-credit qualification.

The neutral root runtime lacks optional Torch/Accelerate/TRL dependencies.
A broader API slice had three dependency failures (one Torch, two Accelerate);
those three passed in the existing isolated ML runtime. Full train/environment
collection also failed on two async-TRL modules unavailable in the root runtime.
These are unresolved full-ladder environment gates, not passing skips or native
qualification. No ML dependency was added to the neutral root to hide them.

An initial two-transition FP16 fixture compared loss values from independently
evolved half-parameter models with an FP32-level tolerance. Second-update errors
were 3.07e-6 (GRPO) and 7.60e-6 (SAMPO). Separate episode graphs and one shared
graph accumulate rounded half gradients differently. Tests now compare loss
arithmetic at identical parameters (rtol/atol 1e-6) and retain separate gradient/
parameter-history checks: BF16 0.005, FP16 0.0007, FP32 1e-6. These are explicit
fixture tolerances, not industry-wide error guarantees. Native LoRA qualification
must check trainable parameter/optimizer dtype; the existing FP16 runtime requires
FP32 trainable parameters. No production precision setting was changed.

The exact pinned TRL dataloader repeats generation rows and slices buffered
completions. Already-resolved updates must bypass that repetition, while retaining
the native optimizer lifecycle. The private adapter uses one opaque update payload
per sequential dataloader batch and delegates to native training_step. It rejects
native gradient accumulation/num_iterations, distributed execution, column removal
and Liger's loss bypass when those would change resolved meaning. The public
request gate remains closed; this fixed-evidence adapter is not live collection,
overflow retry, checkpoint recovery or distributed support.

Real full-population Qwen BF16 graph retention OOMed during backward with the
default CUDA allocator despite roughly 1.89 GiB reserved-but-unallocated memory.
PYTORCH_ALLOC_CONF=expandable_segments:True allowed the identical graph/objective
to complete, peaking at 5.215 GB allocated. Keep the failed log and allocator setting
in qualification evidence; do not infer a universal 8GB capacity guarantee.
FP16's second full-population update clipped 104 actions and had zero gradient,
but Adam's retained moments produced a real 0.0309985 L2 parameter transition.
A zero gradient is therefore not proof that a native optimizer applied no update.
Empty/omitted objectives and overflow require separate transaction semantics.

## Decision Log

Decision (revision136, 2026-10-02/user): multi-GPU execution is out of scope,
including single-host data parallelism. R137 is closed as out of scope rather
than open. The data-parallel veRL code and its gloo multi-process tests stay,
but the public guard in `packages/train/src/posttrain/train/requests.py` now
rejects any resolved selection whose `world_size` is not 1, so no unqualified
multi-GPU run can be admitted. Release 0.4.14 ships the single-device engine.

Decision (revision134, 2026-10-02/user): multi-node distributed execution and
distributed TRL are out of scope for this plan. Single-machine multi-GPU on
small workstation-class cards (e.g. 2x RTX PRO 4500) is in scope for the native veRL
data-parallel qualification; prefer the free workstation for everything else.

Decision (revision134, 2026-10-02/Claude): lift the public guard per
(backend, algorithm, objective variant) from a declared qualified matrix rather
than per backend wholesale, so newly built paths (spans, turn rows, GDPO/CAPO,
vLLM TRL rollouts, distributed TRL) stay closed until their own receipts exist.

Decision (revision134, 2026-10-02/Claude): add role-based `ActionSelection`
(`mode: roles`). Span IDs are population-specific, so a reusable catalog
selection of reasoning or answer support needs roles; this selects supplied
annotations only and preserves the canonical rule that extraction stays
plugin-owned (here the environment's renderer-accounting producer).

Decision (revision134, 2026-10-02/Claude): resolved TRL collection runs TRL
post11 active rounds in Posttrain (`ActiveRoundPlan`) because the resolved path
collects explicit reserved rows, bypassing the dataloader-driven native refill;
its arithmetic is parity-tested against veRL's native implementation and the
native TRL refill is disabled under a resolved selection to avoid nesting.

Decision (revision134, 2026-10-02/Claude): adopt TRL post14 through the existing
`carbonteq-dev` source convention used by every fork, since post14's identical
bytes are on both dev and stable. Admit only SAMPO active sampling at the native
guard; keep curriculum/dynamic filtering and the public request guard closed
until R134 real collection and recovery pass.

Decision (revision133, 2026-10-02/Codex): retain native active-sampling control
flow and add a private evidence adapter rather than implement another sampler.
Reject changed receipt membership/coverage and failed evidence publication before
eviction or optimizer admission. Keep public SAMPO guarded until real integration
and recovery pass. Stable TRL promotion does not enable broader engine features.

Decision (revision132, 2026-10-02/Codex): combine final shipped-asset BF16/FP16
execution and recovery in one retained-evidence job; reuse established R129
pack/schedule comparison evidence rather than collect another population.
Keep stable promotion and broader native SAMPO/semantic/distributed/family
qualification open until their own authoritative gates pass.

Decision (revision131, 2026-10-02/Codex): publish the tested generic TRL fixes as
a candidate prerelease, with a fixed source tag and retained artifact hashes.
Adopt only after Posttrain-owned development readback and candidate qualification,
then stable promotion. Broader engine qualification guards stay in force.

Decision (revision130, 2026-10-02/Codex): accept the retained-evidence TRL
precision/packing/schedule/recovery gate only within the final receipt's scope.
Keep native active SAMPO, semantic/process-credit integration, distributed and
model-family qualification and publication/adoption gates explicitly open.

Revision129: retain original sealed sampling provenance during restoration;
only expected sampler provenance is taken from the sealed population. Keep
every executor/old/reference/template/recipe/recovery identity check and frozen
correction requirement. New native verification uses a new source-bound pair,
not silent migration of historical checkpoints. Preserve R128's useful math/
packing/precision evidence despite its failed recovery gate.

Revision128: stop only the owned known-invalid R127 suite after recording its
failure; wait for terminal state and GPU idle before R128. Retain both source
archives unchanged and independently hash the corrected harness. Preserve
nonzero KL qualification instead of setting beta0 to avoid the failed gate.

Revision127: consolidate strengthened-identity precision, packing, schedule and
full resume checks over the known informative2.6B native population. Compare
equal-objective packing branches within precision; record schedule differences
without demanding equal final parameters. Independent observed-score value,
derivative and clipping references run at every update. Native two-update
continuations use sealed checkpoints and must not collect new data.

Revision126: add the generic mixed-layout supplement to TRL's shared native
trainer lifecycle, where all inherited trainers receive the same correction.
Preserve native optimizer/RNG/adapter APIs, active selection and trainability;
leave distributed/full-model loading untouched. Keep the historical R126
diagnostic distinct from final qualification and reject changed ancestry in
future source-bound recovery. Do not relax exact saved-state comparison.
Validation from the framework root uses the isolated native runtime and existing
source PYTHONPATH with pytest `--import-mode=importlib -c /dev/null
-p no:cacheprovider packages/train/tests/test_trl_policy_job.py -q`:13 pass.
`uv run --no-sync pyright packages/train/src/posttrain/train/backends/trl/policy_job.py`
reports0 errors; scoped Ruff, both repository diff checks and all9 import
contracts pass. The remote image lacks pytest, so the maintained offline fork
fixture is invoked directly using runpy and a TemporaryDirectory in the same
Torch/Transformers/PEFT runtime: repaired loader passes, unmodified native loader
fails the identical fixture. The small model uses random local configuration,
never network weights. No correctness runner is added to framework Git.

- Revision125: ignore scratch tools and generated evidence by their specific
  local directories/extensions; retain written findings and package-owned
  correctness tests. Preserve raw artifacts on disk for investigation. Do not
  weaken the exact continuation acceptance criterion because R125 failed it.

Revision124: preserve the failed original qualifier and use CPU verification of
the same saved artifacts to repair architecture assumptions. Keep model choice
at historical2.6B and proceed to native recovery using the retained population.
Do not collect new data for checkpoint continuation. Native score derivatives
are compared to independent analytic FP64 references without training in FP64.

Revision123: align resolved native TRL admission with native veRL and prior
Verifiers train-client semantics without repairing sampled arguments. Keep raw
rejected syntax visible and unknown/nameless calls rejected. Preserve the live
R123 source as a diagnostic; stage a new sealed source for the correction.
Use the exact historical2.6B model for the next informative fresh population.
Independent objective measurement observes the same training pass and returns
the unchanged evaluation; it must not fabricate advantages or alter gradients.
The user explicitly requests a known better model and continued engine work;
the selected2.6B revision is the model resolved by the historical positive-reward
runs. Finish inspecting the live combined job before changing model or restarting.

Revision122: select Trello email, Sheets sales-call and Airtable email together
for multi-turn and short-action coverage. Prefer archived within-group semantic
reward spread greater than0.1 over tiny mistake-penalty differences or uniform
success. Preserve historical evidence as selection rationale; fresh correctness
still requires actual current-model/runtime receipts. Longer tasks with thousands
of completion tokens are deferred from the first bounded population.

Revision121: keep one grouped native job for precision, packing and schedule
checks, plus CPU artifact comparison. Preserve source renderer and sampler
provenance. Initial states must match within precision; schedule and precision
variants are not required to produce equal final parameters. Audit historical
native training groups before choosing the next fresh collection tasks.

Revision120: follow the user's consolidated-run direction. Gather diagnostics
in one native pass and fork the same evidence and initial states for necessary
comparisons within one coordinated job. Small runs qualify implementation;
research and ablations still determine recipe recommendations. No fabricated
reward spread and no relaxation of informative-update assertions.

Revision119: retain uniform success/failure controls and their failed informative
gates. Use a multi-assertion native AutomationBench task without changing reward
meaning, algorithm or model. Keep original check.py unchanged for exact recovery
of earlier profiles; separate checker and bootstrap name the new task and admit
its larger3072 prompt budget within the existing4096 total context.

Revision118: preserve R117 zero-credit result as a rejected informative gate;
do not relax assertions or invent heterogeneous reward. External verifier writes
all authenticated observations before asserting informative_update_gate so failed
controls remain inspectable. Try a separate bounded reply profile and inspect
native behavior manually; no inference about best training recipe follows.

Revision117: close corrected veRL continuation only after authenticated complete
native-state equality. Proceed to real corrected TRL ordinary jobs using the
dedicated dirty candidate, preserving source hashes and publication gate. Native
TRL source preflight is not sufficient to admit public policy_updates.

Revision116: use the dedicated sampled-score candidate named in the plan instead
of the general sibling checkout. Keep incorrect R116 snapshot/preflight as a
rejected audit record and create separate R117; never overwrite evidence or call
dirty source published. Verify native generation-score capability during CPU
preflight, in addition to ordinary Posttrain admission.

Revision115: enforce distinct effective example_id values for resolved dataset
inventory before launch, while retaining established legacy cycling semantics.
Keep frozen R111/R116 qualification sources untouched by this local worker edit.
Independent CPU receipts remain required before reporting corrected FP16 qualified.
R114 receipts now pass; launch R115 on the same immutable source and use full
native state comparison, including saved scaler, before closing continuation.

Revision114: retain the immutable R111 source for corrected FP16 and resume,
separately from newer local TRL/shared-preset changes. Qualify correction and
scaler together before opening another native recipe or production guard.
The BF16 result closes this bounded single-rank LFM continuation gate only.
Stage corrected TRL separately with exact clean fork SHA and362 framework hashes.
Use the existing native environment factory and ordinary collector; only the
external qualifier bypasses final public request construction. No production
guard removal follows from CPU preflight.

Revision113: close actual corrected BF16 score/weight/checkpoint-component
qualification only after independent native score reduction and publication
authentication. Keep initial weights frozen for resume and compare native state
and the new correction component against uninterrupted boundary2. Preserve the
R111 source and distinguish it from local subsequent TRL/shared-preset refactoring.

Revision112: keep Posttrain as the recipe normalizer for correction mode/bounds,
including structured recipe token cap3. Wire TRL's existing supported ordinary
collector with the same authenticated sampled scores and first-loss freeze
boundary as veRL. Preserve the vLLM correction guard until actual native TRL
qualification. Keep current shared-helper refactor outside immutable R111 source
so its corrected BF16 resume tests use exactly the original staged implementation.

Revision111: use native node log probabilities authenticated by retained population
bytes, not unauthenticated receipt score metadata. Derive complete correction
once from initial frozen old scores; do not relax fresh admission to accept
pre-scored populations. Corrected ordinary recovery rejects missing correction
maps. Source hashing in native_job_identity already changes runtime identity when
these implementation files change. No frozen product meaning changes are needed.

Revision110: close the ordinary single-rank LFM FP16 resume gate using complete
state equality, not matching metrics or a successful exit alone. Keep the new
correction checkpoint changes separate from the sealed legacy qualification.
Next collection wiring must authenticate sampler log scores against original
native evidence; typed receipt metadata alone is insufficient probability proof.

Revision109: retain detached correction weights as an optional authenticated
component of the existing native checkpoint. Verify complete original action
support and the existing correction digest before exposing weights. Never infer
them from a resumed current actor or silently supply unit weights. Native veRL
restore reads this component; new collection must still derive its weights from
initial frozen old scores and authenticated sampled scores before the first
objective evaluation. Raw qualification tools remain external.

Revision108: continue FP16 from its own authenticated checkpoint1 with unchanged
source; compare full state at boundary2, including saved scaler. Keep fresh-job
BF16/FP16 checks separate from retained same-evidence packing arithmetic gates.
Do not conflate absent clipping in one naturally sampled population with broken
clipping or recipe quality.

Revision107: use a shared pure detached-weight helper for both backend adapters,
with exact complete native action support independent of update/pack selection.
Do not enable active/default SAMPO merely by removing guard checks. Reuse native
ActiveSamplingReplayBuffer and its dispatch/observe/refill callbacks, prevent
pre-dispatch duplication, prove task uniqueness across discarded/refill groups,
and preserve sampled/old score provenance and correction values across resume.
This changes private qualification machinery, not the frozen product baseline.

Revision106: close ordinary BF16 native recovery only at authenticated boundary2
after comparing model/optimizer/extra/data/old scores and resolved payload. Move
to fresh ordinary FP16 on identical sealed source rather than infer dtype parity
from BF16 or retained actor probes. Preserve all runs and raw external receipts.

Revision105: retain the successful ordinary BF16 source unchanged for checkpoint1
continuation. Verify the full saved native state against uninterrupted boundary2
before claiming resume equivalence. Keep the failed benchmark episode in the
population under the selected mask_truncated_completions=False recipe; do not
silently drop it or infer recipe quality from this correctness run.

- Revision104: preserve the read-only model cache and reuse the existing datasets
  cache environment setting with a qualification-owned writable path. No core
  framework or native source change is warranted for this staging constraint.
  Retry only after verified terminal state and idle workstation. Date/Author:
  2026-10-02, Codex.

- Revision103: reuse immutable-snapshot launch identity rather than weakening
  source validation or building another collector/scheduler. Preserve the failed
  attempt and retry only after authoritative exit/idle confirmation. Declared
  dense layout is checked before Ray and again through the native create_engine
  seam before model weights; native one-record fallback remains while explicit
  resolved pack sizes execute the selected population. Date/Author: 2026-10-02,
  Codex.

- Revision102: keep physical layout an explicit backend selection, independent
  of optimizer schedule/objective. backend_options.resolved_context_layout
  supports the qualified ragged/one-record and dense-population paths; generic
  planner support does not imply native dense-pack qualification. Dense-population
  requires causal SDPA, FSDP1, FP32 initialization/reduction, deterministic
  dropout-free LoRA, published arithmetic/rowwise/gradient-layout flags, and no
  compiled/fused/dynamic/quantized path. Check effective configs before native
  model allocation. No frozen product amendment is needed: these are private
  backend capability/capacity gates, not new objective or collection semantics.
  Date/Author: 2026-10-02, Codex.

- Revision101: close only the maintained retained BF16 arithmetic-profile gate;
  observe the FP16 counterpart before qualifying ordinary multi-record entry.
  Add published source compatibility without changing immutable dependency pins
  or removing admission guards. Date/Author: 2026-10-02, Codex.

- Revision100: publish scoped controls before staging GPU source; retain defaults
  and consumer pins. Qualify settings with no external global overrides, then
  use only the actually qualified profile in ordinary multi-record admission.
  R98 closes physical padding/recovery, not ordinary collection or algorithms.
  Date/Author: 2026-10-02, Codex.

- Revision99: keep full-precision GEMM and math SDPA as independent opt-in native
  numerical settings rather than inferring them from packing or algorithms.
  Apply and restore them for the entire native train/eval scope, including
  backward and checkpoint recomputation. Serialize cooperating native scopes
  because PyTorch backend switches are process-global. Uncoordinated external
  threads remain outside this contract. Preserve default runtime settings.
  Date/Author: 2026-10-02, Codex.

- Revision98: retain v1/v2 only for one release as their original ragged cost
  contract; they cannot declare new layout fields. Default ragged execution
  identity omits the new field to preserve existing seals. Dense layout changes
  execution identity while keeping logical population/objective/update digests.
  Native dense multi-record adapters require an explicit dense layout capability.

- Revision97: no frozen product meaning changes: existing execution budgets
  are hard capacity limits and packing must preserve objective semantics.
  Add explicit backend-neutral layout capability and strict transport migration.
  Preserve ragged defaults/legacy digests; qualify dense-population natively
  before ordinary multi-record admission. Unsupported adapter/layout combinations
  must reject rather than silently reinterpret the budget.

- Revision96: close the retained single-rank BF16/FP16 actor packing and
  checkpoint-recovery gate for publishedc334d02a, while preserving ordinary
  driver, physical padded-token budgeting, algorithms, families, distributed
  and immutable release gates. Arithmetic controls belong in runtime identity.

- Revision95: close the exact retained single-rank FP16 native actor packing
  and boundary-recovery gate. Keep ordinary admission closed until physical
  padded-work budgets/runtime identities and the actual driver path are qualified.

- Revision94 continuation: use published native rowwise configuration, not
  probe replacement, for full applied-update/recovery qualification. Retain
  complete boundary identities and assert restoration before the resumed update.

- Revision94: qualify rowwise controls through the maintained native engine
  with exact model/optimizer/scaler/population resume before ordinary admission.
  Preserve BF16, budget-accounting, family and distributed gates independently.

- Revision93: add independently selectable rowwise adapter compute, default-off,
  as a native control rather than modifying default arithmetic. State kernel
  launch cost and3D/equal-width scope explicitly. Test actual applied updates
  before treating the shared-boundary result as multistep evidence.

- Revision92: qualify the published controls natively before consumer pins;
  BF16 derivative diagnostic passes. Retain separate FP16 and full-update/resume
  gates. Test fixed adapter GEMM shapes externally before expanding configuration.

- Revision90: publish independently tested native arithmetic capabilities as a
  candidate while keeping framework admission closed. Trace the updated-state
  FP16 residual before adopting a numerical tolerance or asserting equivalence.

- Revision88: expose the two independently justified arithmetic controls as
  default-off native engine capabilities rather than shipping probe monkeypatches.
  Require maintained GPU qualification before framework admission. Do not widen
  tolerances merely because the FP16 residual is smaller than its previous gap.

- Revision87: retain R86 as a derivative diagnostic only. Require FP16 at the
  shared boundary and maintained configuration, budget accounting, applied-update
  and resume qualification before enabling ordinary multi-record execution.

- Decision (2026-10-02, revision86): invalidate R85 combined-profile claims
  and rerun the corrected diagnostic from a new path after terminal/idle checks.
  Preserve failure evidence, verify active arithmetic flags before measuring,
  and keep production qualification/admission unchanged.

- Decision (2026-10-02, revision85): compare the second objective at a common
  real parameter boundary before judging multi-step numerical conditioning.
  Retain first-step old scores and reject any claim of job recovery from this
  optimizer-intercepted derivative diagnostic.

- Decision (2026-10-02, revision84 continuation): keep the combined profile
  unqualified for multi-update equivalence until shared-boundary packing and
  perturbed-boundary sensitivity are separated. Record improvements and residual
  failures without claiming general invariance or relaxing tolerances.

- Decision (2026-10-02, revision84): carry the causally verified layout and
  adapter precision controls into native applied updates with ordinaryFP32
  adapter initialization. Preserve frozen old scores and all native counters;
  do not adopt external monkey patches or call the profile released before
  maintained source/config/tests, both dtypes and recovery qualify.

- Decision (2026-10-02, revision83): test combined nativeFP32 adapter precision
  and contiguous cotangents after isolated controls; preserveBF16 backbone and
  native synchronization. Do not add production settings based on aggregate
  improvement without native updates, resume and both-dtype qualification.

- Decision (2026-10-02, revision82): test value-preserving gradient-layout
  normalization using the native backward before implementing a generic opt-in
  capability. Do not alter loss derivatives, masks, weighting or native gradient
  synchronization. Qualify actual updates and performance before adoption.

- Decision (2026-10-02, revisions79–81): replay actual frozen projection
  weights and cotangents with their original strides before proposing a native
  gradient-layout correction. Do not adopt the adapter-only prototype or blame
  objective/mask normalization for the isolated numerical difference.

- Decision (2026-10-02, revision79): restrict intermediate comparisons to
  declared original contexts; treat padding states separately. Trace the final
  block before naming a kernel defect. Preserve fixed-objective evidence and
  keep numerical precision decisions separate from research recipe adoption.

- Decision (2026-10-02, revision78): inspect original-context layer values
  and cotangents before attributing the residual to arbitrary framework math
  or silently weakening packing requirements. Keep diagnostic hooks outside
  maintained source; their qualification coverage is explicitly bounded.

- Decision (2026-10-02, revision77 continuation): do not adopt the external
  FP32-adapter prototype as a packing correction. Retain its failed hypothesis
  and continue backbone numerical analysis; any acceptance bound must explain
  operation/dtype error and Adam sensitivity, not simply fit observed gaps.

- Decision (2026-10-02, revisions76–77): investigate FP32 adapter computation
  as a named potential precision capability, not a silent recipe change. Reuse
  native FSDP per-module precision controls and maintain distributed ownership.
  Require actual gradients and update/resume evidence before adopting it; no
  tolerance is relaxed based only on reference proximity or smaller FP16 errors.

- Decision (2026-10-02, revision75 continuation): reference sensitivity is
  evidence for a precision-dependent backward cause, not a complete tolerance
  calibration. Require effective adapter dtype and separate convolution TF32
  controls before changing acceptance or shipping a precision profile.

- Decision (2026-10-02, revision 75): keep the FP32 reference explicitly
  diagnostic and preserve frozen embedding values when removing its storage
  alias assertion. Do not broaden supported precision or declare a framework
  defect from an unsupported reference path. Correct hook placement before
  treating gradient dtype telemetry as evidence.

- Decision (2026-10-02, revision 74): intercept optimizer application and
  compare native backwards on unchanged parameters before diagnosing a
  reduction/objective bug. Use FP32 computation only as a numerical reference,
  retaining BF16/FP16 as the supported training qualification targets.

- Decision (2026-10-02, revisions 72–73): qualify actual native gradients and
  updates under the isolated arithmetic controls before selecting production
  settings. Keep global-width padding external until its memory accounting,
  runtime identity and model-family behavior are explicitly contracted.
  The BF16 result remains unqualified despite exact starting score agreement;
  do not relax acceptance or infer backward correctness from forward equality.

- Decision (2026-10-02, revisions 71–72): retain negative native packing
  qualification and isolate input padding after attention/GEMM controls.
  Exact fixed-score equality is diagnostic evidence; require renewed native
  gradient/update qualification before adopting a scoring precision contract.

- Decision (2026-10-02, revisions 70–71): preserve exact trained LoRA module
  topology. Diagnose fixed-parameter score drift before any acceptance-margin
  change or multi-record admission. Keep experimental probes outside Git and
  use unchanged published native source for the arithmetic comparison.

- Decision (2026-10-02, revision 69 continuation): preserve explicit language
  module topology in exported adapters before qualification. Do not broaden
  target suffixes or silently ignore missing modules. Keep the measured FP16
  discrepancies as negative evidence and investigate unchanged-policy score
  drift before choosing any precision-dependent acceptance margin.

- Decision (2026-10-02, revision 69): mark the native BF16 layout comparison
  unqualified despite exit0 and finite updates. Preserve score/gradient/adapter
  receipts, instrument FP16 drift and initial adapter restoration, and keep the
  existing admission guard. Reuse merged upstream #7610 for scalar export;
  retry export directly from the retained Gemma checkpoint on CPU. No duplicate
  upstream PR, new training run or consumer pin change is required for that
  diagnostic. Native larger-pack correctness remains a required full-goal gate.

- Decision (2026-10-02, revision 68): retain Gemma's zero-signal execution
  result without calling it model-gradient qualification. Isolate the native
  packing comparison from collection by reusing informative original evidence
  and restoring one native initial model/optimizer/scaler/RNG boundary for each
  layout. Wait for authoritative GPU idle state before dispatching that check.
  Audit reuse of native active sampling before implementing default SAMPO.

- Decision (2026-10-02, revision 67): adopt the narrow opt-in native declared
  boundary seam for CPU candidate integration, publish its fork before runtime
  selection, and preserve existing static/dynamic defaults. Keep public and
  ordinary multi-record admission guarded until actual native GPU qualification.
  Native GPU/distributed acceptance cannot be inferred from a helper or model
  fixture test. Preserve the active R66 run and its immutable older source.

- Decision (2026-10-02, revision 66): qualify text-only Gemma with explicitly
  inspected language projection paths using the existing LoRA CSV contract.
  Preserve multimodal tower semantics and public family guards. For native
  packing, retain the native forward/backward/optimizer lifecycle and add a
  narrowly scoped planned-microbatch seam only if native scheduling cannot
  preserve both record and context-token capacities, including partial tails.
  Do not duplicate the native optimizer loop or treat packing as minibatching.

- Decision (2026-10-02, revision 65): preserve the Hub foundation selection,
  complete the two missing files at its immutable revision, and retry without
  changing production loaders or validators. Keep failed experiments outside
  Git and do not call successful cache resolution family qualification.

- Decision (2026-10-02, revision 63): qualify FP16/base-KL recovery from full
  state equality at two boundaries and retain actual masks/outcomes separately
  from efficacy claims. Move next to the exact cached Gemma 4 E2B variant with
  its dedicated off-reasoning renderer and native multimodal model factory;
  do not substitute a tiny random model or redefine informative optimizer
  qualification as successful collection. Keep the ordinary job bounded to
  two updates within the requested two-to-three-update scope.
- Decision (2026-10-02, revision 63 continuation): retain the family-guard
  failure, leave the production supported-family list unchanged, and use an
  exact pinned-policy external qualification bypass to obtain native candidate
  evidence before considering a supported-family amendment. All remaining
  native model, context, objective and checkpoint checks remain active.

- Decision (2026-10-02, revision 62): verify nonzero KL through addressed
  support, finite loss, frozen reference tensors and saved scaler state rather
  than relying on the selected beta alone. Do not infer recipe efficacy from
  short bounded KL values. Compare full unchanged-source FP16/KL continuation
  next and preserve each profile's raw artifacts in distinct local locations.

- Decision (2026-10-02, revision 61): accept single-rank LFM BF16 ordinary
  recovery only after independent exact comparisons at both resumed native
  checkpoints. Qualify FP16 plus nonzero base-policy KL next using existing
  reference ownership and counters. Keep historical negative controls and
  avoid extrapolating this profile's success to the remaining native gates.

- Decision (2026-10-02, revision 60): verify checkpoint publications in a
  separate CPU container after the training job becomes terminal. Keep the
  same r59 source for ordinary resume and compare full native/driver state at
  both subsequent boundaries. Use real trace policy masks and executed tool
  outcomes, rather than trace completion flags or tool-schema counts, to
  describe rollout behavior. Retain parent-failure regression evidence outside
  Git without modifying parent checkout files.

- Decision (2026-10-02, revision 59): implement strict PyTorch kernel selection
  in veRL's existing full_determinism helper because the actual native control
  eliminates the observed backward variation. Keep it opt-in and let unavailable
  deterministic operations raise. Require new ordinary job/resume evidence
  under the resulting published identity; historical warn-only checkpoints do
  not qualify the new strict training profile.
  Archive published source 8f0de236 for native qualification; the subsequent
  83c35675 ledger-only commit records publication without changing runtime code.

- Decision (2026-10-02, revision 58): test strict PyTorch determinism in the
  external backward diagnostic before selecting a fork correction. Do not
  label native checkpoint loading defective when its immediate restored state
  matches exactly; do not silently expand equality tolerances for drift.

- Decision (2026-10-02, revision 57): use an explicitly instrumented external
  diagnostic to isolate native checkpoint loading and backward repeatability.
  Keep its expected pre-optimizer abort separate from ordinary job success,
  and independently compare its saved loaded state rather than assuming that
  authenticated input checkpoint bytes prove correct in-memory restoration.

- Decision (2026-10-02, revision 56): keep independent checkpoint inspection
  outside the lifespan of its training container. Verify published copies and
  native bytes in a CPU-only container without occupying the GPU. Use veRL's
  existing actor full_determinism setting for a separate identified control;
  do not change historical recovery identities or infer nonzero KL validation
  from a beta=0 job.
- Decision (2026-10-02, revision 56): record the unsuccessful native deterministic
  control and the repeatable convolution primitive as separate evidence. Do not
  infer a checkpoint defect or a custom-kernel defect from matching forward
  loss and unequal backward gradients; require an actual native restore-state
  and whole-model repeatability comparison before selecting a correction.

- Decision (2026-10-02, revision 55): do not mark native recovery qualified
  from matching counters or frozen scores alone. Retain the measured adapter
  and optimizer discrepancy, qualify native deterministic execution separately,
  and preserve historical source/runtime identity. Current-source FP16 job
  qualification proceeds independently of diagnosing historical BF16 drift.

- Decision (2026-10-02, revision 54): eliminate producer aliases rather than
  weakening duplicate-metric rejection. Preserve total/policy/KL terms and
  resolved lifecycle metrics through the canonical normalizer. Qualify recovery
  with unchanged historical source and record host finalization failure
  separately; do not weaken recovery identity to accept revised source.

- Decision (2026-10-02, revision 53): make the existing TRL CSV interpretation
  backend-independent and reuse it without importing one backend into another.
  Do not change LoRA rank, alpha, target names, model family or objective to
  bypass the failure. No fork or frozen product-baseline change is needed.
- Decision (2026-10-02, revision 53 continuation): put the empty-batch cleanup
  guard in native V1 fit; preserve its saves/logging/counter lifecycle. Do not
  fabricate TransferQueue keys, regenerate evidence or bypass native fit.
  Publish this generic maintained-fork change before qualifying a new source
  snapshot. Old-source recovery artifacts remain bound to their actual runtime.

- Decision (2026-10-02, revision 52): transport only the serializable manifest
  and trainer factory through Ray; construct process-local context/observation
  objects in the owning process. Reuse native configuration predicates and the
  existing portable environment factory. Keep setup receipts and tools outside
  Git, keep model cache read-only, and place generated dataset cache in the
  qualification directory. No product-baseline or recipe change is required.

- Decision (2026-10-02, revision 51): stage full copies, independently verify
  their actor/driver seals, fsync files/directories and rename atomically before
  sending recovery observations. Do not interpret observer enqueue as completed
  provider promotion. Keep staging until the host's artifact lifecycle releases
  it; rotate only native source directories with verified corresponding copies.
  Honor checkpoint_steps=0 without producing retained recovery artifacts.
  Ordinary selected manifests may use the candidate native composition while
  public operation admission stays guarded pending actual GPU qualification.

- Decision (2026-10-02, revision 50): derive score/runtime identities from the
  effective initialized native engine instead of fixed qualification labels.
  Use adapter-disabled native inference for the declared base reference only;
  separate checkpoint/start references remain unsupported until qualified.
  Retain public launch guards while the private native entrypoint supplies the
  complete actor/trainer/runner composition needed for full-job qualification.
  Unwarped sampler support still needs actual vLLM/native score reconciliation;
  passing the selection guard is not distribution-parity evidence.

- Decision (2026-10-02, revision 49): add a narrow generic plain TaskRunnerV1Base
  extension to the fork rather than copying agent-loop manager construction or
  accessing Ray-private class metadata. Preserve default TaskRunnerV1 decoration
  and all native manager interfaces. Publish source and ledger before immutable
  consumer adoption. Dataset traversal and objective schedule epochs do not
  govern driver update termination; the explicitly selected optimizer budget
  does. Full native GPU/Ray job execution remains a qualification gate.

- Decision (2026-10-02, revision 48): keep host tracking credentials outside
  native workers. Use the existing append-only tailer for durable observation
  transport over the current single-host shared workspace, preserving canonical
  ProducedArtifact and observation values. Original native artifact files remain
  replay authority. Actor metrics remain evaluated by the actor and logged by
  the driver; do not add a second global metric emission path. Runtime identity,
  template identity and frozen reference scoring remain explicit native
  composition requirements rather than fabricated defaults.

- Decision (2026-10-02, revision 47): extend the existing resolved recovery seal
  to the native synchronous job directory rather than creating a parallel
  checkpoint format. Retain the native actor save and dataloader serialization;
  authenticate the complete enclosing job before invoking native restore. Never
  fall back silently from corrupt committed evidence. Transport real runtime
  identity only at launch; do not fabricate RunContext values during planning.

- Decision (2026-10-02, revision 46): keep the optimizer and population host in
  the actor worker; the driver passes durable receipt JSON through native
  ONE_TO_ALL RPCs and never receives a callback from inside an actor RPC.
  Compose PPOTrainerSync locally rather than altering global trainer registries.
  Require one actor result and exactly one committed update per driver step.
  Reused populations reclaim rollout replica memory without collecting again;
  metrics describe the evaluated objective and prepared selected credit.
  Reject checkpoint pruning until the driver can own a fully sealed job save.

- Decision (2026-10-02, revision 45): reuse native Transformers full_determinism
  for opt-in reproducibility, without changing default kernels or algorithm
  semantics. Configure before loading weights and record effective arithmetic
  controls in checkpoint identity. Do not accept numerical tolerances merely
  because loss agrees; distinguish nondeterministic continuation from checkpoint
  corruption. This is backend operating configuration, not a frozen-baseline
  product-meaning change.

- Decision (2026-10-02, revision 44 continuation): keep informative ordinary
  resume unqualified after tensor inequality. Diagnose with the same archived
  source and exact retained population before proposing a fix or tolerance;
  preserve failed receipts and diagnostic runners outside Git.

- Decision (2026-10-02, revision 44): expose a narrow native engine-construction
  seam and use existing actor_worker_cls selection. Preserve native ownership
  rather than mutate an initialized engine or a global registry. Publish the
  independently tested source before any dependency-pin adoption. Extend the
  bounded Thinking response budget only after inspecting actual truncated
  outputs; interpret the result as correctness evidence, not recipe selection.

- Decision (2026-10-02, revision 43): add a typed complete-settings envelope
  alongside existing native launch fields, and reject disagreements. Preserve
  original values across the isolated process instead of reconstructing settings
  IDs, warmup ratios or algorithm profiles. Legacy manifests without a resolved
  selection remain readable; public/worker guards stay until lifecycle wiring
  and native qualification are complete.

- Decision (2026-10-02, revision 42): bind existing DataProto and native
  BaseEngine lifecycle rather than replace native stepping with a standalone
  optimizer. Worker-owned initialization, rollout policy synchronization and
  authenticated restoration remain explicit host dependencies. A passing CPU
  fixture does not permit removing the ordinary-worker/public release guards.

- Decision (2026-10-02, revision 41): carry typed collection metadata as JSON
  in native agent-loop extra_fields, alongside the retained native artifact
  reference. Do not infer original contexts or process credit from response_ids
  or response_mask, and do not treat a single-episode receipt as an admitted
  complete population. Legacy collection keeps its current behavior by default.

- Decision (2026-10-02, revision 40): carry the existing framework-neutral
  PolicyUpdateSettings value in the typed worker manifest. Do not translate
  schedule units into native PPO minibatch fields or derive masks from flattened
  agent-loop rows. Until the resolved host is wired, fail explicitly at worker
  admission rather than consuming the selection through legacy main_ppo.

- Decision (2026-10-02, revision 39): native Trainer's total loss cannot be
  reported as policy loss for resolved objectives. Use the actual evaluator at
  the committed boundary, separately label weighted KL and measured divergence,
  and do not call declared support or nonzero advantages applied learning.

- Decision (2026-10-02, revisions 37–38): preserve exact selected/runtime identity
  on continuation and keep old-source receipts distinct from new seeded-source
  receipts. Establish a nonzero base-reference KL control using a previously
  qualified trained adapter, not synthetic parameter perturbations or altered
  rewards. Uniform task rewards cannot establish policy-credit learning even
  when a valid KL regularizer produces nonzero gradients.

- Decision (2026-10-02, revision 36): use the actual cached selected Thinking
  model and ordinary _run_online_rl entry point with fresh native AutomationBench
  episodes. Keep each failed runner/version and output directory; inherited
  policy-context configuration preserves Verifiers' deliberate pinned-client
  semantics. Do not count model loading or a running container as applied work.

- Decision (2026-10-02, revision 35): put the generic opt-in generation receipt
  capability in the TRL fork, not a copied generator in Posttrain. Preserve the
  default native path and require the new capability before resolved generation.
  Validate and publish the fork first; consumer pins/runtime adoption follow.
  Candidate work touches the new fork worktree plus framework online_rl.py,
  policy_rollouts.py and job preflight; regression ownership stays with each repo.

- Decision (2026-10-02, revision 34): compose the resolved lifecycle inside the
  ordinary TRL job function while retaining the public guard until fresh native
  jobs pass. Unsupported production handoffs reject before model loading.
  Inspect sealed recovery components before native decoding, then independently
  compare selected runtime, recipe, masks, template, scoring and retry contracts.
  Recovery evidence belongs to checkpoints, not serving-model exports. Native
  CPU replay establishes transport only; it cannot substitute for fresh training.

- Decision (2026-10-02, revision 33): make new admitted native checkpoints
  self-contained for original rollout bytes before wiring production resume.
  Seal an additional native-rollout-evidence component and verify it before
  decode. Retain byte-identical authority rather than serializing mutable
  reconstructed trace objects. A legacy checkpoint without it must supply a
  verified external artifact resolver; never silently recollect on recovery.

- Decision (2026-10-02, revision 32): production native collection receives
  complete logical collection rows independently of resolved dataloader slots.
  Preserve ordinary group admission and trace observations, then retain and
  validate native evidence before artifact submission/handoff. Reject currently
  unqualified async/distributed/native-active-refill composition explicitly;
  keep their full qualification requirements open rather than approximating them.

- Decision (2026-10-02, revision 31): retain context selection across worker
  transport, and verify a native artifact's declared content before resolving
  population credit. Frozen native references use logical artifact names.
  Local durability does not imply completed remote promotion; recovery hosts
  must resolve retained or promoted bytes and preserve the same digest.

- Decision (2026-10-02, revision 30): decode population snapshots with native
  schemas and preserve graph topology. Import Verifiers lazily so detached
  framework planning remains independent of environment activation. Missing
  original trace IDs must reject before a schema supplies random defaults.

- Decision (2026-10-02, revision 29): reuse the bridge's existing native episode
  authority for immutable per-population snapshots. Do not invoke finalize each
  optimizer update or replace native envelopes with observed derived trace rows.
  Keep native decoding and artifact promotion in host composition, and reject
  incomplete retained membership before optimizer mutation.

- Decision (2026-10-02, revision 28): use matching native context-forward
  rounds and zero-weight admitted graph anchors for this dense distributed path,
  retaining actual native collectives. Separate global objective reporting from
  the carrier passed to backward. Compare independent loss/gradient/transition
  references at identical parameters; do not interpret accumulated trajectory
  differences as a loss error or broaden margins without diagnosing the cause.
  Native trainer integration, sharding, scaler and recovery are still separate
  gates. No public settings or frozen product meaning changes in this slice.

- Decision (2026-10-02, revision 27): partition original-context score/replay
  work, then evaluate one complete global objective and route its score adjoints.
  Do not split the objective into independently normalized rank losses, which
  loses cross-rank ratio dependencies. Reuse existing bounded adjoints and native
  DDP averaging; add no independent optimizer loop. This implements the accepted
  distributed/global-weight contract without a public API or baseline change.
  Native distributed admission stays closed until the adapters qualify matching
  collective schedules, empty rank execution, synchronized overflow and recovery.
  Distributed ownership identity must enter the native checkpoint seal before
  any distributed resume can be admitted; single-rank checkpoint schemas remain
  unchanged in this slice.

- Decision (2026-10-02, revision 26): qualify each backend's turn-row
  checkpoint continuation against its own uninterrupted trajectory and keep
  cross-backend initialization equivalence as a separate gate. Preserve the
  sealed prepared credit, resolved objective, population-frozen scores and
  native optimizer/scaler state; do not infer distributed recovery, fresh
  collection or public readiness from these single-rank fixed-evidence checks.

- Decision (2026-10-02, revision 25): require exact initial trainable tensors
  for cross-backend optimizer comparisons, in addition to frozen scores and
  source/runtime identities. Equal seeds and initial scores are insufficient
  when zero-initialized LoRA B hides differences in A. Preserve the native
  initialization paths for ordinary jobs; the controlled fixture supplies one
  immutable initialization solely to isolate equivalence. Do not alter loss
  math or expand tolerances to hide initialization differences. This closes
  the observed bounded discrepancy, while broader release gates stay open.

- Decision (2026-10-02, revision 24): implement the user's requested
  turn-scoped SAMPO work as the named turn-rows variant, coupling ratio scope
  with equal-turn reduction while retaining original credit. Preserve legacy
  episode semantics and recovery identities. Scheduling by turn is independently
  available for author-shaped optimizer row units; scheduling by episode may
  still select several turns for an update. Do not change defaults or tune the
  clipping band from tiny checks. Investigate the observed BF16 native deviation
  and complete new-variant native recovery/public wiring gates before adoption.

- Decision (2026-10-02, revision 23): keep artifact persistence and native
  decoding host-owned, but share durable input admission and frozen recovery
  between backends. Fresh populations must obey the already-approved task
  uniqueness and synchronous sampling contracts; explicit frozen reuse is a
  different lifecycle operation. This implements the amended canonical baseline
  without changing objective formulas or introducing another trace store.
  Public launch remains gated until production collection/restore wiring uses
  the admitted path and all remaining backend/runtime release gates are met.

- Decision (2026-10-02, revision 22): qualify typed restoration from inside
  a population, where remaining occurrences must use retained old scores and
  credit, rather than only from an exhausted population boundary. Keep failed
  experiment sources/logs and verify unchanged sources before continuation.
  Preserve historical recovery schema tests; correct their pre-engine fixture
  instead of adding absent new fields to legacy production identities.

- Decision (2026-10-02, revision 21): restore frozen credit and occurrence
  identities from authenticated retained evidence, never by rerunning a credit
  estimator. Require the selected complete recovery identity and explicit
  sampler correction; typed decoding does not replace native checkpoint seals.
  Preserve the existing schema and product baseline.

- Decision (2026-10-02, revision 20): authenticate complete native input views
  at population admission and revalidate on replay. Share this reader across
  backends; native-format decoding and durable artifact ownership remain host
  responsibilities. Keep public execution guarded until actual production
  collection and recovery use this verified composition.

- Decision (2026-10-02, revision 19): keep native GRPO/DAPO scalar credit
  separate from SAMPO and structured credit. Normalize on complete admitted
  prompt groups before optimizer scheduling; do not normalize each minibatch.
  Leave public production integration gated until collection, recovery and
  algorithm-specific native qualification are complete.

- Decision (2026-10-02, revision 18): share native population lifecycle rules
  between TRL and veRL, while each backend retains its own optimizer/precision
  hooks. Keep run-global dataloader slots and collect only during actual native
  loss evaluation; do not call Trainer.train separately per population or reset
  optimizer state. The host restores retained native inputs, and backend recovery
  verifies their resolved identity before native state loading.
- Decision (2026-10-02, revision 18): retain explicit fixed-evidence reuse and
  unapplied final occurrences in bounded campaign reports. Keep unequal-exposure
  schedule comparisons as correctness evidence only. Continuous GPU veRL,
  productive Gemma4, actual collection composition, cross-backend matched native
  gradients, distributed normalization and release/pin gates remain open.

- Decision (2026-10-02, revision 17): keep UUID blanking pending domain-aware
  normalization research; preserve complete observation bundles and explicitly
  version proxy keys. Do not claim repeated tool text proves identical world
  state. Keep generated traces/runners outside Git. Scope replay findings to the
  recorded run/population and keep informative SAMPO native qualification open.
- Decision (2026-10-02, revision 17): recovery binds prior applied/attempt
  offsets independently of the current population cursor. Reject inconsistent
  Trainer global_step or veRL scheduler position before updates/checkpointing.
  Legacy zero-offset recovery remains supported; continuous native collection
  and dataloader resume still require implementation and real qualification.

- Decision (2026-10-02, revision 16): native adapters consume the shared resolved
  value rather than reconstructing objective/schedule settings independently.
  Validate all later occurrences before the first optimizer update; carry the
  execution/capability contract alongside the plan. Keep one-population versus
  run-global recovery semantics explicit before implementing continuous collection.

- Decision (2026-10-02, revision 15): make SampoCreditEstimator own raw native
  reward shaping and increment its contract version. Keep the existing legacy
  production formula and public launch gate. The confirmed adapter discrepancy
  does not justify changing the algorithm, beta, clipping recipe or a published
  framework merely because earlier qualified evidence is incomplete.

- Decision (2026-10-02, revision 14): preserve native harness preparation rather
  than replacing it with a scripted chat loop or reusing Qwen rollout tokens.
  Resolve writable cache/network prerequisites in the external qualification
  container, preflight on CPU, and retain all-zero real groups as negative coverage.
  This narrows evidence claims without weakening the required Gemma optimizer gate.

- Decision (2026-10-02, revision 13): place typed objective/schedule resolution
  in backend-neutral update_resolution.py and validate all occurrence capacities
  before handing work to a native adapter. Retain the public launch gate until
  production collection and recovery consume this resolved value. This prevents
  hand-authored recipe drift and late capacity failures after a partial population
  has already updated weights. Prepared credit remains an input, so complete-group
  normalization is not repeated on optimizer minibatches.


Revision-11 decision: adapt the native on_optimizer_step boundary for bounded
single-process TRL retries instead of replacing the trainer loop or adding a
fork patch. Native Trainer training_step still owns precision/backward; its
clip/scaler/optimizer hooks own each attempt. Resolved applied counters and LR
advance only once on success. Keep the occurrence pending after exhaustion,
require explicit nonnegative retry bounds, and bind bounds plus execution
capabilities into checkpoint recovery identity. No public launch is certified
by this private native seam. The frozen product baseline does not change.

Revision-10 decision: retain the deterministic CUDA profile as an explicitly
qualified reproducibility control. Do not silently make it a production default
or reinterpret default BF16 discrepancies as accepted numerical tolerance.
Rationale: both native backends pass exact same-boundary continuation under this
profile, but the default-kernel cause and performance implications are unmeasured.

Decision: native checkpoint files are saved first and bound by a final atomic,
fsynced resolved-update seal. No seal means no usable engine recovery boundary.
Native model/optimizer/scheduler/RNG/scaler ownership stays in the backend;
Posttrain retains population/credit/occurrence metadata and frozen old/reference
scores. Applied/native cursors and runtime identities must agree before resume.
Rationale: a cursor sidecar alone could resume changed optimizer or evidence.
Date/Author: 2026-10-02/Codex.

Decision: center the engine on a resolved optimizer update, with native trajectory
records, named populations, explicit objective terms and backend execution.
Rationale: overlapping statistical groups are not storage parents, and native
schedulers already supply part of the required machinery. Date/Author:
2026-10-02/Codex, following user review.

Decision: general support consists of qualified algorithm definitions and
supported combinations, not freely composable mathematical switches.
Rationale: ratio, credit, derivative and reduction choices interact.
Date/Author: 2026-10-02/Codex.

Decision: preserve existing algorithm formulas and defaults during migration;
adding schedules does not select a recommended recipe or change group admission.
Rationale: author settings are evidence, not universal ablations.
Date/Author: 2026-10-02/Codex.

Decision: replay is an optional qualified execution strategy, not a prerequisite
for all updates or a universal memory guarantee. Reuse existing native scheduling
first. Rationale: one long context or full-distribution statistic may still exceed
capacity. Date/Author: 2026-10-02/Codex.

Decision: consume process evidence and qualified external credit without owning
a reward-model training lifecycle. Rationale: score interpretation and inference
composition already have separate owners. Date/Author: 2026-10-02/Codex.

Decision: expose explicit settings additively but reject launching them until a
backend's resolved native executor is qualified. Rationale: accepting a catalog
selection must not silently execute the legacy schedule. Existing absent
selections retain their path and recovery identity. Date/Author: 2026-10-02/Codex.

Decision: the common graph-retaining bridge forms one objective after all score
dependencies are present; native trainers own backward and optimizer transitions.
Rationale: per-pack normalization or stepping would change sequence-ratio support
and update boundaries. Explicit sampler correction must be supplied or explicitly
absent, never dropped by the bridge. Date/Author: 2026-10-02/Codex.

## Outcomes & Retrospective

Revision136 closes the plan's in-scope work for release 0.4.14. Multi-GPU is out of
scope by user decision (R137 closed, not open); the public guard admits only
single-device resolved selections. Old execution route retirement remains
deferred one release per Milestone 7.

Revision135 completes the in-scope engine gates on real models: TRL post14
adoption; native veRL SAMPO active collection (R134); TRL resolved active
collection (R135b); renderer-derived reasoning spans with role selection and a
composition-owned real scorer (R138); training with injected process credit
(R139); Gemma 4 E2B BF16 (R136, FP16 unsupported for that model); veRL post9
and renderers dev3 released and pinned; public admission opened only for the
GPU-qualified matrix. Every GPU gate passed in BF16 and FP16 with exact
checkpoint continuation unless noted. Open: R137 two-GPU single-host native
veRL data-parallel GPU qualification (code proven by real gloo multi-process
tests; RunPod attempts failed on capacity/startup; awaiting approval to spend
again). Out of scope by user decision: multi-node and distributed TRL. Old
execution route retirement remains deferred one release per Milestone 7.

Revision134 closes TRL post14 consumer adoption and qualifies native veRL SAMPO
active collection (R134: BF16/FP16 real Ray/TQ collection, informative updates,
exact checkpoint continuation), fixing three production defects found on the
way (metric mismatch, offline pinned model resolution, LFM history-turn mask)
and releasing renderers dev3. The full consumer ladder passes with no failures.
Still open: kind-image rebuild for post14/dev3, veRL fork release adoption
before lifting public SAMPO admission, TRL active collection, semantic/process
credit, distributed normalization, multiple-family/Gemma 4 qualification,
observation wiring and release gates.

Revision133 closes installed-asset TRL BF16/FP16 math/update/recovery and stable
artifact promotion gates, but consumer pin adoption remains undone. Adds native
SAMPO candidate evidence retention with68 native CPU regressions passing. Real
SAMPO collection/recovery, semantic/process credit, distributed/family coverage
and complete release validation remain required. This entry supersedes previous
R132-live and promotion-pending observations and provides the current handoff.

Revision132 closes candidate development-index storage/readback and installed-
asset CPU regression gates. Final installed-asset native BF16/FP16 continuation
is confirmed live; no success is inferred yet. No stable pin changes or goal
completion is claimed. The original full engine scope remains unchanged.

Revision131 makes the generic TRL fixes reproducible at a pushed immutable
source commit and retained GitHub candidate release. Development-index publisher
is dispatched; no index success or stable adoption is inferred from dispatch.
All remaining engine qualification requirements retain their original scope.

Revision130 closes the matched retained-evidence TRL suite: terminal exit0,
four independently verified two-update branches and four exact continuations,
with exact BF16 packing parity. Candidate distribution archives build and match
their source, but remain unpublished. This supersedes preceding live R129 status;
the full engine goal remains active with native SAMPO, semantic/process-credit,
distributed/family coverage and release checks still required.

Revision129 corrects a Posttrain recovery restriction exposed by real retained
evidence, with14 regression cases passing and unchanged objective mathematics.
R128 remains live for remaining precision/schedule branches; R129 is prepared
for final matched continuation under the corrected provenance contract.
R128 subsequently terminates with all4 math/update branches passing and exact
BF16 packing parity, but4 rejected resumes. R129 is launched and confirmed live.
The complete static-check gates now pass;57 actual native veRL CPU tests pass.

Revision128 corrects a failed replay harness boundary without changing engine
semantics. R127 remains terminal negative evidence; R128 is the live matched
precision/schedule/packing/resume qualification and is not yet a passed gate.

Revision127 advances final recovery qualification from a historical-loader
diagnostic to a matched source-bound suite, presently live. Replay and fresh
collection are separate gates; original R124 trace evidence remains authoritative.
Native SAMPO active selection, semantic/process-credit integration, distributed
family coverage, publication/adoption and broader release checks remain open.

Revision126 supplies a reproduced checkpoint restoration correction and closes
the inherited-source identity omission, with13 focused Posttrain tests and a
real offline PEFT restoration regression. R126 exits0 with132 tensors exactly
matching uninterrupted training and no fresh rollouts. This is a bounded
historical-identity diagnostic; final strengthened-identity BF16/FP16 recovery, fork publication,
native SAMPO selection, semantic/process credit, distributed/family coverage and
remaining release checks stay open.

Revision125 removes approximately97% of untracked line noise from Git's view
without deleting evidence or hiding production changes. `git check-ignore`
coverage/visibility checks and `git diff --check` pass. R125 resume comparison
is a recorded failure requiring investigation; the engine goal remains active.

Revision124: fresh informative ordinary TRL BF16 collection, nonzero KL,
actual clipping, both applied boundaries, finite optimizer state and sampled
correction now have direct2.6B evidence. Continuation is launched but unverified.
Full goal remains open: corrected FP16 on informative fresh/replayed evidence,
native SAMPO active selection, semantic/process credit, Gemma/multiple-family
and distributed qualification plus fork/public release gates remain required.

Revision123: selected tasks now have current-definition compatibility evidence,
and an actual rollout probe exposes malformed syntax, truncation and an adapter
admission mismatch. Source fix passes29 focused tests; independent measurement
hook passes a fixture. R123 has terminal uniform failures; fresh informative
qualification remains unproved. R124 launches only after R123 terminal/GPU idle
verification, with matched historical model and separate sealed source.

Revision122: task selection now uses inspected past training groups and actual
reward/turn evidence. A12-episode three-task population is selected, not launched.
It will support one combined fresh qualification instead of independent jobs for
each diagnostic. Partial-credit/error cases, current task compatibility and full
remaining engine gates stay open. No experimental tooling added to Git.

Revision121: informative native TRL replay, two applied updates, BF16/FP16,
frozen correction and BF16 pack parity now have direct evidence. The full goal
still requires retained replay recovery, fresh informative TRL collection,
nonzero KL/masks/process credit, active SAMPO, other families/Gemma and distributed
qualification plus publication/release gates. External qualifier code remains
uncommitted. The launched suite used its original orchestration script; expanded
CPU comparator is separately retained, not claimed as part of that initial script.

Revision120: private SAMPO reservation/round wiring passes42 local tests, but
real native active collection remains unqualified. Corrected TRL controls check
correction and lifecycle with zero credit; they do not qualify productive updates.
Qualification now follows the consolidated suite below. R119 is closed and no
new standalone GPU attempt has been launched. The full engine goal remains open.

Revision119: actual corrected TRL collection, detached sampler correction and
zero-credit behavior are independently checked, but informative native updates
are not yet qualified. Multi-assertion R119 is live; no completion or recipe-quality
claim follows. Corrected TRL recovery/FP16 and full broader engine scope remain open.

Revision118: corrected TRL actual job completes but informative qualification is
open because constant successful rewards produce no policy gradient. R118 bounded
profile is live; source/framework/fork remain unchanged and benchmark scoring is
native. Complete corrected TRL recovery/FP16, active SAMPO, schedule comparison,
masks/process/Gemma/distributed/observation/public release gates remain open.

Revision117: corrected ordinary veRL BF16/FP16 actor/scaler full-state continuation
passes exact independent comparisons. Corrected TRL BF16 is live on its separate
candidate source; correction/native-update/recovery qualification remains open.
No cross-backend objective, active-SAMPO or broader release completion is inferred.

Revision116: correct TRL candidate staging now passes CPU preflight and native
generation-score signature check with post13 metadata. No actual corrected TRL
applied update or continuation claim yet. R115 corrected FP16 continuation remains
live on unchanged R111 source. Full remaining scope is preserved.

Revision115: resolved inventory admission is stricter and validated locally.
R114 actual corrected FP16 actor/scaler updates and independent correction/state
authentication pass. R115 corrected continuation is live; complete comparison
pending. Full scope remains active, including native SAMPO refill/uniqueness and
public release gates. Neither snapshot includes the local inventory guard.

Revision114: corrected BF16 full-state continuation passes independent comparison.
Corrected FP16 R114 is launched; no applied-update completion claim yet. Native
TRL correction, default SAMPO active collection, same-evidence schedule comparisons,
semantic/process credit, informative Gemma, distributed execution, observation and
release/adoption gates remain open. No raw correctness tools or consumer pins
are committed.
Current local cohort76 passes. Corrected TRL source is separately retained and
CPU preflight accepted; BF16/FP16 applied updates/recovery remain unqualified for
that snapshot. The candidate TRL version mismatch remains explicit.

Revision113: corrected ordinary BF16 has real applied/checkpoint evidence and
exact independent correction weights. Corrected full native resume is live,
not yet verified. FP16 correction, native TRL correction/sampling, active SAMPO
and full remaining model/mask/process/distributed/release scope remain open.

Revision112: shared correction preset and TRL persistence/export wiring pass
focused validation. Actual corrected BF16 R111 is still live; two saved updates
are not yet an authenticated completion claim. Native corrected TRL/vLLM,
active-SAMPO, family/distributed/masks/process and release gates remain open.

Revision111: collection-to-first-update correction wiring passes90 related tests.
The fresh corrected BF16 job is live on a distinct immutable1592-file snapshot.
Its actual native correction/clipping/checkpoint/resume remains unverified;
prior ordinary exact resume gates refer to the uncorrected diagnostic source.
Keep all active-SAMPO/semantic/process/family/distributed/release gates open.

Revision110: both ordinary dense BF16 and FP16 resume gates pass exactly for the
qualified LFM profile. New correction persistence passes83 related regressions,
but fresh corrected collection/training is not qualified. No raw correctness
tools, consumer pins or release assets were committed. Continue correction/native
SAMPO, masks/process credit, family, distributed and release gates unchanged.

Revision109: correction persistence and uncorrected compatibility pass focused
references, including identical next loss/gradients over restored old scores.
Corrected native training, active SAMPO and broader backend/release gates remain
open. R108 FP16 continuation is live on the unchanged1591-file tree; current
local correction changes are not credited to that native job.

Revision108: ordinary dense FP16 applied/checkpoint/scaler gate passes for the
single-rank LFM profile. FP16 continuation is live; complete recovery comparison
remains open. Correction helper14 tests, scoped Ruff/Pyright and9 import contracts
pass. Active SAMPO/native correction and full remaining goal gates remain open.

Revision107:14 new correction reference tests pass; native wiring remains open.
Measured BF16 mismatch supplies evidence that unit correction cannot be assumed.
Fresh FP16 has reported2 applied attempts, finite gradients, scale128/skipped0;
terminal/checkpoint verification and resume are pending. Keep source qualification
scopes distinct: R106 runs the existing1591-file sealed source, not the new helper.

Revision106: ordinary dense BF16 applied-update, exact native resume and native
trace-mask provenance gates are closed for this single-rank LFM profile. R106
FP16 is live. This does not close SAMPO active/dynamic collection, semantic spans,
granular scorer credit, informative Gemma4, native multi-GPU or release adoption.

Revision105: ordinary two-record BF16 now has independently authenticated applied
updates, checkpoint publications, frozen credit/scores and adapter changes.
Native role masks and exact saved-population correspondence are verified;
ordinary resume and FP16 remain unverified. R105 continuation is live. The full
SAMPO/mask/process-credit/family/distributed/observation/release gates remain open.

- Revision104: explicit ordinary capability wiring and99 focused tests remain
  green; two setup failures precede any applied update. The corrected ordinary
  BF16 job is live with immutable weights/source and writable dataset cache.
  Ordinary GPU, resume, FP16 and all broader algorithm/family/distributed/release
  requirements remain open.

- Revision103: ordinary capability/profile wiring and99 focused checks pass.
  Fresh normal-driver BF16 qualification is live after correcting snapshot
  metadata. No applied ordinary update is claimed yet. Scoped retained BF16/FP16
  gates remain closed; all broader algorithm/mask/family/distributed/release
  requirements remain open. Consumer pins are unchanged; no raw tools committed.

- Revision102: maintained numerical profile passes BF16/FP16 real retained
  updates and exact resume over physical budget/row-count variations. Next
  replace ordinary entry's records1 restriction with explicit checked capability
  selection and prove the fresh normal-driver path, preserving all broader gates.

- Revision101: published scoped numerical settings now pass BF16 real native
  applied-update/recovery qualification with exact cross-layout gradients and
  adapters. FP16 is live;16 framework runtime tests and46 native CPU regressions
  pass. Next ordinary-driver qualification must use declared settings and the
  maintained physical-budget path. The full goal is still active.

- Revision100: retained BF16/FP16 physical-budget and strictv3 checkpoint gates
  pass with exact saved cross-layout derivatives/updates. Native arithmetic
  controls are published and CPU-tested; maintained BF16 qualification is live.
  Ordinary admission, SAMPO native collection/correction, masks/process credit,
  informative Gemma, multi-GPU and final release integration remain open.

- Revision99: maintained BF16 physical packing/budget/recovery gate passes on
  all three retained layouts. FP16 counterpart is live on immutable R98 source.
  Next close the declared-arithmetic configuration gap and qualify ordinary
  driver admission. SAMPO collection/correction, granular credit, informative
  Gemma, distributed execution and publication/release gates remain open.

- Revision98: physical layout and migration are implemented with98 passing
  focused tests. Real-evidence preflight validates the budget-dependent pack
  boundaries, and maintained BF16 applied-update/recovery qualification is live.
  Ordinary profile admission and the remaining full-goal gates remain open.

- Revision97: physical layout accounting is the next production gap. Numeric
  controls and native recovery are already qualified only for the retained R94
  profile; budget/profile integration remains incomplete pending implementation,
  regression tests and a native run without the external padding wrapper.

- Revision96: two native applied updates and exact complete-boundary recovery
  now pass across records1/2 in both major precisions. Native controls are
  published;40 fork CPU tests and14 framework native-runtime tests pass.
  Raw receipts stay outside Git. Next resolve physical padding costs and native
  profile admission before ordinary multi-record jobs; the full goal is active.

- Revision95: native FP16 applied-update and exact checkpoint-recovery packing
  qualification passes for this LFM population; BF16 counterpart is live.
  Full engine release is incomplete, with remaining gates unchanged.

- Revision94 continuation: published rowwise native capability is staged
  reproducibly and FP16 native update/resume qualification is live. Its result
  is not yet known; the full goal remains active.

- Revision94: external FP16 multi-update packing discrepancy is corrected;
  native rowwise capability has40 passing focused tests. Full release and
  qualification remain open, and no experimental probes are committed.

- Revision93: the shared-state FP16 residual has an experimentally verified
  correction. Native rowwise capability/tests are implemented; publication and
  native GPU adoption remain open. Actual two-step FP16 packing comparison is
  live, and the full goal remains active.

- Revision92: published native controls reproduce the BF16 diagnostic result
  without monkeypatches, and the FP16 residual is localized to small adapter
  arithmetic differences. Rowwise FP16 control is live. Full goal remains active.

- Revision90: native capability candidatec62c4468 is published,37 focused CPU
  tests passed, R88 exits0 and block14 tracing is live. Neither the candidate's
  native GPU initialization nor applied-update/resume profile is qualified yet.

- Revision88: completed shared-boundary probes in both major precisions and
  implemented native opt-in controls with37 passing CPU regressions. FP16
  residual tracing is live; multistep numerical acceptance, maintained GPU
  profile, recovery, release publication/pins and broader goal gates remain open.

- Revision87 status: R86 BF16 exits0 with the restored117 layout hooks and24
  FP32 adapter modules. The goal remains active: production adoption, broader
  native integration and release gates remain incomplete despite this numerical
  diagnostic success. Raw probes and tensors remain outside Git.

Revision85 is terminal and invalid for its intended profile due a diagnostic
generator error. Revision86's corrected shared-boundaryBF16 handle is live.
All mathematical and release gates remain open; no production source changed.

Revision84 comparisons are independently retained for both precisions and
continue to show multi-step differences. Revision85's common-boundaryBF16
derivative probe is live after verified idle hardware. No production precision,
tolerance, pins or admission change; the full goal remains active.

Revision84 nativeBF16 andFP16 comparisons both finish exit0 with two applied
updates/layout. The demonstrated precision/layout controls greatly improve the
first derivative but do not prove multi-step equivalence; continuation remains
open. All diagnostic handles are terminal. Full goal, ordinary packing and
maintained release gates remain open.

Revision83 establishes a successful fixed-parameter correction combination for
retainedBF16 evidence. Revision84's actual native update comparison is live.
Production settings, tolerances, pins and admission remain unchanged; full goal
and broader algorithm/family/distributed/release gates remain open.

Revision82 reduces native backward packing discrepancy while preserving math
and model parameters. Revision83's combined precision/layout control is live.
The full goal and production admission remain open; raw probes stay outside Git.

Revision81 proves the first stride-dependent native mismatch in isolation.
Revision82's value-preserving layout prototype is live. Production precision,
tolerances, packing admission and consumer pins stay unchanged.

Revisions79–80 identify the first differing projection and a stride-sensitive
replay hypothesis. Revision81 is live. Native model parameters/objective remain
unchanged; no tolerance relaxation or production packing admission follows yet.

Revision78 narrows the residual to backward through the final decoder block,
with exact valid forward values and starting adjoints. Revision79 starts its
submodule/GEMM trace. Primary research is recorded with scope limits; production
precision, tolerance, masks, objective and packing admission remain unchanged.

Revision78 starts the native fixed-parameter layer trace after verifying idle
hardware. Its handle is live; no production precision/tolerance/admission or
consumer pin change. The full implementation goal remains active.

Revision77 completes the FP32-adapter control and independent reference
comparison, disproving adapter-only repair. All diagnostic handles are terminal.
The numerical qualification contract and native packing remain open; no
production adapter precision, tolerance, dependency pin or admission change.

Revision76 completes strict reference controls and FP16 dtype measurement,
and quantifies gradient sign sensitivity. Revision77's external BF16-backbone/
FP32-adapter probe is live. Consumer pins, production precision and packing
admission remain unchanged. The full goal stays active.

Revision75 reference completes with a substantially smaller packing gradient
gap and correctedFP32 dtype evidence. Native BF16 dtype probe subsequently
exits0 and confirmsBF16 gradient computation despiteFP32 storage. FP16 is live.
The reference's cuDNN TF32 control and FP16 counterpart remain open, as do
production packing admission and all broader implementation/release gates.

Revision74 establishes deterministic packing-sensitive backward at unchanged
parameters. Its FP32 reference fails before yielding gradients. Revision75
launches the explicitly storage-split frozen reference with improved dtype
instrumentation; this handle is live. All release gates remain as before.

Revision74 starts an unchanged-parameter native backward probe after all prior
handles are terminal and GPU idle is verified. Full goal and packing gates stay
open. No backend precision selection, tolerance or consumer pin is changed.

Revision72 closes the fixed-model BF16 score diagnostic with exact equality
across both layouts. Revision73 starts native applied-update comparison under
those controls after observing the prior handle terminal. Production packing
admission is unchanged; gradient/update, FP16 and broader gates remain open.
The BF16 native update comparison subsequently exits0 but still fails gradient
equivalence:0.8654% first-step and11.8095% second-step relativeL2. Controlled
FP16 is now live. Exact forward equality narrows the next diagnostic to backward
arithmetic; it does not close packing or justify a production precision change.
FP16 subsequently completes and independent comparison confirms0.1068% first-
step and0.9022% second-step gradient differences. Both controls are terminal;
fixed-parameter backward is the next investigation, with packing still gated.

Revision71 verifies that optimizer/restoration changes are unnecessary to
reproduce packing drift. Fixed-model probes on both precisions locate most
drift in attention/GEMM arithmetic and leave one padding-sensitive row. All
parameters remain unchanged and same-layout repeats are exact. Revision72's
fixed-width BF16 control is live; no consumer pin or admission change follows
from these diagnostic results. The release goal remains active.

Revision70 closes actual Gemma adapter topology reload: 50 language targets,
100 matching adapter tensors at export dtype, and exact zero-adapter base
logits. Source3a4812cb is published; main consumer pins remain unchanged.
Revision71 has launched a fixed-parameter BF16 diagnostic after confirming
idle hardware. Packing, informative Gemma, default SAMPO, distributed/native
advanced-contract and full release gates remain open. The full goal is active.

Revision 69 produces strong negative evidence and a reusable export correction.
The native BF16 qualifier completes but its two layouts fail equivalence at
the model-gradient level. Gemma's ordinary run applies two zero-credit steps
and publishes sealed checkpoints, then fails model export; the independent
inspector verifies both publications and no adapter change. The exact upstream
replicated-buffer repair is backported, tested and published as 62db2a41.
FP16 packing and CPU Gemma export subsequently finish exit0; neither closes its
gate. FP16 gradients differ across layouts, and the exported Gemma adapter
fails topology reconstruction. Both handles are terminal. Public and ordinary
multi-record gates and dependency pins remain
unchanged. Raw tools/receipts remain outside Git; the full goal remains active.

Revision 68 establishes a negative qualification result: R66's two policy
updates have zero credit and gradients. Checkpoints 1 and 2 are published, but
the GPU job remains active during finalization and an independent CPU inspector
is checking sealed state. A CPU-only real-runtime preflight succeeds for two
optimizer minibatches under native execution layouts [1,1,1] and [2,1] over
R59's frozen mixed-reward evidence. The prepared GPU qualifier does not alter
ordinary public admission, and its source/receipts are outside Git. Actual GPU
packing, informative Gemma, native recovery and all broader goal gates remain
open.

Revision 67 publishes the native ordered-partition source candidate and passes
14 native helper regressions plus 12 framework adapter tests. Two-process CPU
validation covers rank disagreement; the framework fixture covers unchanged
objective and parameter updates across packs. Scoped types, Ruff, import
boundaries and diff checks pass. An expanded 30-case cohort has 29 passes and
the existing candidate version/native-name mismatch failure. R66 remains the
verified live Gemma handle at this point; checkpoint-1 native saves are present,
but final metrics and informative optimization remain unproven. No dependency
pins or public admission changes occur. Default SAMPO, actual GPU packing,
distributed model execution and release gates remain open.

Revision 66 moves Gemma beyond the cache blocker and identifies its next concrete
construction failure. The meta-model/PEFT inspection succeeds and its receipt
gemma4-language-lora-preflight.json is retained locally and remotely outside Git.
R65 is terminal exit1 without OOM; R66 is the next running candidate. No
production source changes, pins or public support declarations are needed for
these cache and target-selection corrections. Native packing inspection finds
an exact-divisibility constraint requiring explicit handling before its gate
can be removed. Remaining full-goal gates are unchanged.

Revision 65 resolves the concrete offline-cache blocker with a successful real
snapshot_download check. The R64 invalid local-artifact experiment is retained
as gemma4-failed-attempts/r64-foundation-contract.log outside Git. R65 is launched
under the existing immutable r59 source root, with a distinct output directory;
its native model/optimizer outcome is pending. No consumer pins or supported
family declarations change. Default SAMPO, packing, distributed and release
gates remain open.

Revision 63 independently verifies R62/R61 exact checkpoint-2 and checkpoint-3
continuation. Both CPU comparison containers exit 0 with 249 matching tensors
plus driver state; the ordinary resume exits 0. Local external receipts are
fp16-kl-resume-step2-comparison.json and fp16-kl-resume-comparison.json.
The Gemma 4 factory preflight exits 0 and its receipt is retained outside Git.
Initial posttrain-r63-gemma4-strict-bf16 is terminal exit1 from the family guard.
Separate posttrain-r63b-gemma4-candidate-bf16 is launched after another idle
check; this is the next live handle, not public or completed family support. No source,
fork or consumer-pin/public-admission changes occur in this revision.

Revision 62 qualifies ordinary strict LFM FP16/base-KL three-update execution
on source 8f0de236 with authenticated checkpoint publications and finite adapter
transitions. The independent inspector exits 0 and additionally verifies frozen
3994-action old/reference vectors plus saved scale/growth state. Raw FP16
result/metrics/observations/traces are retained under working/full-verl-r59/
fp16-r61-strict-kl; fp16-kl-three-update-verification.json is retained separately.
The R62 ordinary resume is launched and is the next live handle. Public
admission remains guarded; no source, fork or consumer pin changes occur here.

Revision 61 independently verifies exact R60/R59 continuation at steps 2 and 3;
both CPU comparison containers exit 0 and report 248 matching tensors plus
driver state. Ordinary strict BF16 recovery is now qualified for the retained
single-rank LFM Thinking profile. R61 ordinary FP16/base-KL job is launched on
the idle GPU and its existing handle is the next active check. Independent
comparison receipts are resume-step2-comparison.json and resume-comparison.json
under the external r59 root; no raw tools enter Git. No source, fork, immutable
consumer pin or public-admission changes occur in this revision.

Revision 60 closes the fresh strict ordinary BF16 three-update gate on published
veRL source 8f0de236: job exit 0, native/published checkpoint authentication,
finite changed adapters, canonical observations and manual native trace review.
CPU inspector exits 0. The unchanged-source R60 resume is launched and remains
the next recovery gate; exact continuation is not yet claimed. Local raw
result.json, observations.jsonl, verl-metrics.jsonl, traces.jsonl,
manual-rollout-inspection.json and parent-determinism-regression.log live under
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/full-verl-r59.
Remote source/results remain /var/lib/posttrain/qualifications/full-verl-20261002-r59.
No production code, fork commit or consumer pin changes occur in this revision.

Revision 57 starts the targeted native restore/backward diagnostic on the idle
workstation. Revision 58 inspects its terminal expected diagnostic abort and
exact restored-state roundtrip. Backward gradients differ despite restored RNG
and unchanged buffers; native warnings identify the warn-only Flash Attention
backward path. A docker-exec CPU inspector terminates with 137 when the probe
container exits; an independent CPU-only container reruns the comparison and
succeeds. Container posttrain-r58-strict-backward-probe is now launched after
direct idle verification. Retained source identity is veRL 77fe49a with 360 hashed framework
files. Scripts and resulting receipts live outside Git under
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/full-verl-r55;
remote root /var/lib/posttrain/qualifications/full-verl-20261002-r55.
Revision 59 closes the strict backward control: three gradient snapshots match
exactly. The one-case native CPU regression, fork Ruff and both diff checks pass.
The owning helper correction is published as 8f0de2365f1041954b67f74df5a14c7ba0532755.
Durable fork/consumer ledgers and inherited native-name capability mapping track
that source. The recovery
blocker remains open until fresh ordinary job/resume qualification under the
new published source identity. R59 is staged at
/var/lib/posttrain/qualifications/full-verl-20261002-r59 with raw local scripts
under the corresponding working/full-verl-r59 directory. The existing ordinary
  posttrain-r59-strict-bf16 handle is running; do not restart it on observation
  timeouts. Framework worker Ruff passes and three targeted source/capability
  admission tests pass (129 deselected). No consumer pins change in this revision.

Revision 56 closes ordinary veRL FP16 full-job finalization on the retained r55
source: three informative updates, canonical host metrics, complete native saves
and final adapter publication succeed. Independent checkpoint/parameter checks
pass. Deterministic BF16 qualification is running; continuation equality, KL,
distributed/family/algorithm and dependency/public-admission gates remain open.
No raw qualifier tools were added to Git and no dependency pins changed.
Full type/lint/import checks now pass after fixture corrections, and 143 affected
native checks pass. Full pytest is running; the observed catalog source/lock
inconsistency remains a release blocker until consumer adoption is reconciled.
The deterministic BF16 continuation succeeds as a job but still fails equality.
Convolution repeats are exact in the bounded primitive control; the whole-model
cause remains unresolved. Raw evidence is retained outside Git. Full native CPU
suite finishes with 2625 passed, seven failures and 21 errors (25 skips).
Remaining failures distinguish catalog/source/version adoption and local Ray
startup from the successful native GPU host jobs. Locked uv sync succeeds
(303 resolved packages, 111 checked). No complete test/recovery/release gate is
claimed; no live qualification containers remain at this stopping point.

Revision 55 completes the historical native BF16 continuation and yields direct
evidence of backward/update drift despite matching frozen and driver state.
No tolerance or identity guard was weakened. Current-source ordinary FP16
qualification is running in posttrain-full-verl-r55-fp16; full-job success,
deterministic native recovery and broader algorithm/family/distributed/release
gates remain open. Experimental tools and raw model/checkpoint receipts remain
outside the repository; no consumer pins or main-repository commits changed.
The full local validation attempt also exposes unclosed type/test gates. Base
workspace dependencies cannot collect direct TRL tests; the new runtime test
now follows the existing optional-Torch convention. This is a collection
correction, not a substitute for running the native release environment.

Revision 54 r53-b native metrics at updates 1/2/3 report loss
-1.4997000694274902 / 0.5007587671279907 / 0.49967318773269653,
selected policy actions 719/1024/1024, nonzero advantage fraction 1 throughout,
applied/attempt counters 1/1, 2/2, 3/3. Clipping is 0, 2/1024, 2/1024.
The group rewards are [1,0,0,0]; only one population is collected and reused.
This is BF16 policy-only correctness evidence, not KL or recipe-quality
qualification. Complete outer operation fails during metric replay after native
final merge. Affected actor/observation/backend tests pass 161 with seven
existing optional skips; scoped Ruff/Pyright, nine import contracts and diff
checks pass. Resume container posttrain-full-verl-r54-bf16-resume uses checkpoint
/qualification/bf16-r53b/model/checkpoint-publications/global_step_1 within
full-verl-20261002-r53b and remains running at inspection. Independent checkpoint
verification and comparison scripts/receipts, original traces and manual review
remain outside Git. No fork or pin changes were made in this revision.

Revision 53 retains r52-g's failure in its existing external directory and
stages 360 framework source/resource files under workstation
/var/lib/posttrain/qualifications/full-verl-20261002-r53. The same exact image
and published fork are used. Job posttrain-full-verl-r53-bf16-a loads the real
LFM Thinking weights and initializes the native actor/LoRA FSDP engine; vLLM
startup is in progress at inspection. This does not yet establish collection,
an applied optimizer update, parity or checkpoint recovery. veRL adapter tests:
125 passed/seven existing dependency skips; TRL common: 25 passed. Scoped Ruff,
Pyright, nine import contracts and diff checks pass. Raw tools remain outside
Git; no fork/pin change or main-repository commit was made.
Continuation: r53-a fails only after its second applied update at native queue
cleanup. External committed-state-verification.json independently verifies
source/publication equality for checkpoints 1 and 2 and finite movement in all
24 LoRA tensors. Update 1 has loss -1.4997001886367798, selected advantage
1.4997000599880024, 725 policy actions, grad norm 0.6178939342498779 and clipping
0. It is policy-only (beta=0); this does not qualify KL. Native source fix
076072b92baf336c2e18e9f38bf7434e4a5f3cb7 and ledger followup
77fe49a9de909f036aa72d8568cc957d226b1e7c are published on
origin/codex/resolved-engine-worker. The fresh replacement is
posttrain-full-verl-r53-bf16-b, root full-verl-20261002-r53b, source 77fe49a;
it is confirmed running through model/rollout initialization. Framework focused
native/backend/TRL-common suite: 198 passed, seven existing optional dependency
skips. Raw sources, logs, receipts and inspection tools remain outside Git.
Full-job completion, resume, FP16 and other original gates remain unqualified.

Revision 52 source archives and raw qualifiers are outside Git under
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/full-verl-r52*
and workstation /var/lib/posttrain/qualifications/full-verl-20261002-r52*.
GPU image sha256:f0d32115d6e3bdca1dda76aad0ff8b9605e9869d4413ea65f76f4bc942cc85eb
contains Torch 2.13.0+cu130, Transformers 5.14.1, Ray 2.56.1, vLLM 0.29.1.dev4,
TransferQueue 0.1.8 and PEFT 0.19.1. The published fork is installed without
dependency replacement and its exact SHA recorded separately from inherited
package version 0.9.0.post8. Attempts a/b/c/d/e respectively expose omitted
templates, an unpickleable fixture lambda, validator arguments, captured context
serialization, and read-only datasets cache. All are terminal before optimizer
execution. Attempt f with a writable cache reaches actor construction and exposes
the fsdp2/fsdp selection mismatch; attempt g uses the corrected selected override.
Six runtime and four runner
tests pass; full focused suite passes 169 tests with seven existing optional
integration skips. Scoped Ruff/Pyright and nine import contracts pass. No pins,
fork commits or main-repository commit changed. Full ordinary veRL qualification,
distributed/family/SAMPO/release gates remain open.

Revision 51 exercises three sealed checkpoints rotated to one native source;
all three independent publication copies remain valid. Missing/corrupt stages
prevent any source deletion. Host publication failure retains the source and
staged recovery boundary. Generated ordinary-worker overrides compose against
the real native Hydra config and install the actor/trainer/runner recipe before
captured Ray dispatch. This is not native model/GPU execution. Focused retention,
driver, runtime and existing backend checks pass. The full focused/native backend
suite passes 168 tests with seven existing Renderers/Verifiers dependency skips;
those integrations remain unqualified. Scoped lint/types, all nine
import contracts and diff checks pass. Published fork source recognition is
extended to 70baba82c0b2b0a8981089ca62c5ab474efe1816 and its ledger-only head
ef5aac6ff92d5a69f72cfe222f0a409af4220314. No fork/pin changes or main-repo commit
were made. Real full-job GPU qualification and remaining family/distributed/
observation/release gates remain open.

Revision 50 native composition tests install the actual worker and TaskRunner
types before a captured run_ppo dispatch, without starting Ray or CUDA workers.
Native BaseEngine fixtures verify adapter-disabled reference scores remain exact
and detached after actor parameter changes. Effective runtime identity binds
precision/template changes and ignores output relocation. Four new focused
tests pass; the broader focused native/backend suite passes 164 tests with the
same seven missing Renderers/Verifiers dependency skips. Scoped Ruff/Pyright,
all nine import contracts and diff checks pass. Full native GPU job execution,
ordinary worker branch, checkpoint
artifact publication/retention, distributed/family qualification and release
gates remain open. No GPU was launched, dependency pin changed or main-repo
commit made in this revision.

Revision 49 runs four runner lifecycle cases: success, TransferQueue init
failure, trainer init failure and fit failure. All retain native manager
dispatch and close TransferQueue; initialized tracking finishes with the correct
exit status. Runner/driver plus fork factory/runner regressions pass 20 tests.
The broader focused native and existing backend suites pass 160 tests with the
same seven Renderers/Verifiers dependency skips; no skipped integration is
claimed complete.
Scoped lint/types, all nine import contracts and diff checking pass. Fork source
70baba82c0b2b0a8981089ca62c5ab474efe1816 is published; consumer pins/lockfiles
remain unchanged. Runtime/template identity derivation, frozen reference scoring,
the actual worker entrypoint branch, sealed artifact publication/retention and
native GPU job qualification remain open. No GPU was launched and no main-repo
commit was made.

Revision 48 passes 156 tests across the focused native engine/driver/checkpoint/
observer modules and existing veRL backend suite. Seven existing skips remain:
six lack Renderers and one lacks Verifiers in the isolated runtime. Scoped Ruff
and Pyright pass. Tests cover actual typed manifest JSON roundtrip, rejection of
missing host context or publishing observer, artifact/metric/event forwarding,
host-publication retry without cursor advancement, and wrong-run/schema
rejection. The native actor factory, TaskRunner installation, runtime/reference
identity derivation, epoch accounting and sealed artifact retention still need
full job composition before native GPU qualification. No GPU was launched,
fork changed, dependency pin updated or main-repository commit made.

Revision 47 passes native driver save/load tests: dataloader position survives
the existing torch serialization; an idempotent repeated save avoids rewriting
committed actor state; data.pt corruption prevents both actor load and dataloader
deserialization. Missing data, mislabeled applied step, altered frozen scores,
actor seal or selector state are rejected. A newer unsealed directory and stale
latest-step pointer do not replace the last complete job boundary. Scoped lint
and type checks, all nine import contracts and git diff --check pass. The three
focused native modules plus test_verl_backend.py pass 151 tests with seven
skips: six lack Renderers and one lacks Verifiers in this isolated environment.
These skips do not satisfy the affected integration gates. No GPU was launched
and no fork or pin changed. Full job
context/observer/session composition, retention publication, epoch accounting,
native GPU qualification and all remaining release gates stay open.

Revision 46 adds native driver/actor lifecycle composition, not production
qualification. Focused validation uses the isolated native veRL runtime and
the published sibling source at 8ae3500d80c1ec48179d786766afe6b4931bacc0.
The two focused test modules pass 22 tests. They exercise collection at actor versions 0 and 2 across four updates,
receipt-only TransferQueue reads, retained-population no-collection updates,
counter rejection, collection interruption boundary restoration, complete
receipt validation, native SAMPO trainer construction and actor engine identity.
Scoped Ruff and Pyright pass; all nine import contracts and git diff --check
pass. No GPU was launched, fork edited, pin changed or main-repo commit made.
Manifest/context composition, driver checkpoint sealing/retention, native job
qualification and the remaining distributed/family/release gates remain open.

Revision 45 diagnostics are retained under the revision-44 external root in
diagnostic-comparison.json, resume-optimizer-comparison.json and native model
diagnostic-before/gradient-2.pt files. The deterministic diagnostic has zero
gradient and parameter discrepancy; the unqualified default-mode discrepancy
remains documented. Production-option runners and archived framework sources
live outside Git at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/fresh-deterministic-r45,
remote /var/lib/posttrain/qualifications/fresh-deterministic-20261002-r45.
comparison.json verifies ordinary BF16 four-episode checkpoints 2/3: 24/24
weights, optimizer state and all three frozen sidecars equal, with seals verified.
The three selected advantages have magnitude 0.4999000199960008; update 2 clip
fraction is 0.0020491803278688526. Both two-episode controls have zero initial
credit and produce new-population differences at step 3; their informative
retained-population resume gate is not claimed. FP16 four-episode qualification
finishes with four successes and zero credit. The bounded eight-episode FP16
pair finishes and verifies informative continuation: checkpoints 2/3 match all
24 matrices, optimizer/scheduler/scaler, native counters and frozen sidecars.
The selected advantages are +0.7244288643548162, +0.7244288643548162 and
-1.2073814405913603. Clipping is zero in this FP16 arm. Scaler 128 is retained
with growth trackers 2/3. This closes this ordinary LFM retained-population
continuation check in both major precisions; public launch, veRL production,
distributed, Gemma and full release gates remain open.
Two additional native determinism translation regressions pass
(14 KL/configuration tests total), beyond the earlier 51-case focused run.
Fork/pin adoption, ordinary veRL integration and broader release gates stay open.
Revision 45 retained.tar.gz SHA256 is
c1f7c8b984f370b4a37ad56cf86f3f8805e4525aa95d1a80a79d772656dd7fba.
After transfer and safe extraction, all 25 retained checkpoint seals verify.
All 24 adapter matrices change at each informative control update 2/3 in both
precisions; maximum update deltas are 0.00010013496648753062 /
0.00010036007734015584 for BF16 and 0.00010013505379902199 /
0.00010035948071163148 for FP16. retention-verification.json records the hashes,
checkpoint names and changes outside Git. The earlier diagnostic archive SHA256
is 1749df224364538c3b1942b29ba8e8cd579f58522e5d3ec18f4d0143b965e0f8;
all ten diagnostic checkpoint seals verify. All native jobs are terminal; final
direct workstation check reports 41 MiB / 0 percent GPU use and no containers.
No experimental runners, receipts or checkpoints are staged or committed.

Revision 44 resumed BF16 Thinking run exits successfully but fails the numerical
continuation gate. At checkpoint 2 only 2/24 trainable tensors equal the control;
checkpoint 3 maximum difference is 0.00025120005011558533. Frozen population,
schedule and score bytes agree at checkpoint 2; the resumed second-step gradient
norm is 0.28944605588912964 versus control 0.2895035743713379. External
resume-optimizer-comparison.json records equal optimizer step counters and
shapes with unequal moments. Informative resume remains open; the goal is active.

Revision 44 raw sources, native receipts and manual analyses live outside Git at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/fresh-thinking-r44;
remote /var/lib/posttrain/qualifications/fresh-thinking-20261002-r44. Pinned native
image remains f0d32115d6e3bdca1dda76aad0ff8b9605e9869d4413ea65f76f4bc942cc85eb.
BF16 Thinking selected model commit 95053d21d8e0b7ca99421a2127ae39c64f685ff3,
rank 8/alpha 16, LR1e-4, temperature .8, two complete episodes, episode minibatch1,
three native slots. Sessions 34552 (512) and 14563 (1024) both exit0. Archive
SHA256s: 886e08105121bec39ce655e51307bd09c15fa2cd20834d2299f738724a62eccc
(512), 72297a42c5f3ab5921063b7bf0cd292b3123fac70c8f98eb699a6d3b7e85bcab
(1024). 1024 first two losses -0.7070068120956421 and 0.7061316967010498;
advantages +/-0.7070067953266834, second clip fraction 2/646. Maximum checkpoint
1-to-2 adapter delta 0.00010013512655859813 across 24 changed tensors.
New worker source publication and native-runtime adoption remain distinct. The
fork worktree is clean after code publication and its ledger follow-up commit;
no experimental runners or raw GPU receipts are committed.

Revision 43 full-settings round trips cover GRPO/DAPO, SAMPO, GDPO and CAPO.
The complete launcher manifest preserves the original GRPO selection; mutated
learning rate and missing typed settings reject. 132 transport/backend tests
pass, seven skip; scoped Pyright has zero errors. No GPU is launched. Next work
is actual driver/actor composition with applied-update counters, native policy
synchronization and checkpoint boundaries; no ordinary veRL resolved training
completion is claimed by this contract change.

Revision 42 command (cwd /home/hammad/projects/rl): add
/home/hammad/projects/verl-posttrain-parity and every packages/*/src directory to
PYTHONPATH, then run /tmp/posttrain-native-verl-update-runtime/bin/python -m pytest
-c /dev/null --rootdir=/home/hammad/projects/rl --import-mode=importlib
-p no:cacheprovider packages/train/tests/test_update_verl.py -q. Nine pass.
The collection-host test uses real native DataProto/BaseEngine with explicit
fixture model/decoder: collected boundaries (applied, attempts) are (0,0) and
(2,3), final counters 4/5, two published artifacts, one retained optimizer and
changed trainable parameters. Changed clipping contracts reject restoration.
No GPU is launched and no ordinary-worker lifecycle or fresh native schema
qualification is claimed. The next composition work is native worker startup,
policy synchronization and selected-settings/runtime reconstruction.

Revision 41 adds policy_rollouts.py in the veRL adapter and opt-in agent-loop
configuration. Two receipt transport/negative tests plus the manifest/backend
suite pass (130 passed, seven skipped); a subsequent agent-loop wiring test
also passes (three receipt tests total). Scoped Ruff/Pyright pass before that
final test addition. These fixtures exercise actual Posttrain loop code with
native protocol fakes; they do not qualify Ray transport or native GPU training.
The assembler now authenticates and merges receipts before shared complete-group
credit resolution; four focused tests pass, including complete versus incomplete
groups and duplicate rejection. Its fixture decoder is explicit: this does not
qualify native schema integration. The host must publish the returned artifact
and bind the admitted population to ResolvedVeRLRun. Public and legacy-worker
resolved guards remain in force.

Revision 40 changes only framework contract/launcher/worker admission and its
transport regressions. The ordinary veRL resolved job gate remains open:
native agent-loop evidence must reach complete-population admission, then the
qualified ResolvedVeRLRun/native engine lifecycle and sealed recovery. No GPU
run or native runtime qualification is claimed by these JSON contract tests.
The additional test_update_verl.py attempt skips at import because the local
/tmp/posttrain-native-verl-update-runtime interpreter no longer exposes a verl
module; pytest exits 5 with no collected tests. Retain this as an unexecuted
native regression gate, not a passing run or evidence of a trainer defect.

Revision 39 source and fresh native runner remain outside Git at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/fresh-trl-telemetry-r39;
remote /var/lib/posttrain/qualifications/fresh-trl-telemetry-20261002-r39. The
workstation was idle (41 MiB, 0%, no containers) before starting the three-slot
FP16 warm-adapter base-reference control. Session 48635 exits before model load
due to the external archive omission. Corrected session 50784 exits 0: three
updates, runtime 18.6694s, total losses 1.875095847481134e-7,
8.813490239845123e-6 and 9.417807405043277e-7, each equal to weighted KL loss;
policy losses and prepared advantages are all zero. Inspection verifies three
resolved records and exactly three policy_loss records, with no legacy duplicate.
Retained archive SHA256 is
78676f8d9f7d9e3c0f5769ee0d8e20d264bf544a05e36e94d3be90d27bf5c80c;
telemetry-analysis.json retains the inspected values outside Git. GPU returns to
41 MiB/0%. This qualifies ordinary-job reporting for this KL-only control,
not productive policy learning or a matched comparison with revision 38.
This reporting change does not close distributed observation, immutable
membership publication or the other production/release gates.

Revision 37 receipts and relocated checkpoints live outside Git at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/fresh-trl-job-r37;
remote /var/lib/posttrain/qualifications/fresh-trl-job-20261002-r37. Failed resume
and successful resume-b logs remain separate. Successful session 78424 exits 0:
global_step 3, runtime 13.0677s, all 24 adapter tensors exact, three retained
checkpoint-2 sidecars byte exact, scale 128 exact, two newly collected episodes.
Archive SHA 852b50138463b83552155447f702968c5eeae51e1b438f8385b7dd9858339726.
Revision 38 runner/current-source archive and native GPU receipts live at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/fresh-trl-kl-r38;
remote /var/lib/posttrain/qualifications/fresh-trl-kl-20261002-r38. Base-reference
session 35245 exits 0 at step 3, runtime 18.4241s, train_loss
3.123146920340029e-6. Initial adapter comes from
/var/lib/posttrain/qualifications/turn-row-resume-20261002-r26/results/trl-float16/checkpoint-1
over exact LFM Instruct df58c174f05ff733f83f8cae10ea9298224c8006. Both launches
follow idle GPU checks (41 MiB/0%, no containers). Independent KL analysis and
reference variants are pending; no policy-credit or release completion claim.

Revision 36 qualification runner and staged framework archive live outside Git
at /home/hammad/experiments/posttrain-correctness/2026-10-01/working/fresh-trl-job-r36;
remote /var/lib/posttrain/qualifications/fresh-trl-job-20261002-r36. Candidate
fork source is mounted from native-hf-scores-20261002-r35/fork. The same pinned
image is used, with offline HF cache at /hf/hub. Attempt a/b/c failure logs and
native episodes remain distinct. BF16 attempt d (session 14641) and FP16 attempt
e (session 72581) are terminal with exit 0, global_step 3, loss/gradient norm 0,
runtime 21.0556 and 18.2451 seconds respectively. The Instruct variant uses exact
cached commit df58c174f05ff733f83f8cae10ea9298224c8006. All GPU launches followed
a 41 MiB/0% occupancy and empty-container check. The new focused regression
suite passes 135 tests. Checkpoint archives, run logs, runner versions and
analysis.json remain outside Git. Some sealed root-owned files could not be
read directly over SCP; a terminal-container read-only archive preserves them
without altering original permissions. Verified relocated archives:
BF16 SHA 3f51032cde77e2c19469dee61ca32605df0c6a37cff188d8e5cf9996c3a96c8b;
FP16 SHA e37afe9f8d0b3babc1ec61cc29cf5d80c3b2f58544ead7f7c2023abe21b3c4df.
No productive gradient, new native resume or KL-reference gate is claimed.
Public launch, fork publication and all other release gates remain open.

Revision 35 fork CPU tests pass four cases; framework local focused tests pass
21 with one existing renderer-dependent case deselected. The pinned native
container with ephemeral pytest 8.4.2 installed passes all 21 tests in the earlier
archive (four fork and seventeen framework cases), including native renderer
coverage. An initial container attempt lacked pytest and is retained as tests.log;
the successful retry is tests-with-pytest.log. New fork/framework archives and
receipts remain outside Git under
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/native-hf-scores-r35,
remote /var/lib/posttrain/qualifications/native-hf-scores-20261002-r35.
Scoped Ruff/Pyright and nine import contracts pass. Before any GPU launch the
workstation reports 41 MiB/0% and no running Docker containers. No fresh native
generation or optimizer is established by these mock generation-score regressions.
Fork candidate remains uncommitted/unpublished and consumer pins unchanged.

Revision 34 adds ordinary TRL job composition, behavior-policy transport and
selected-contract recovery. The isolated native-runtime focused suite passes
130 tests (test_api.py, test_trl_policy_job.py, test_trl_resolved_collection.py,
test_update_admission.py, test_update_inputs.py, test_update_recovery.py,
test_update_score_recovery.py and test_reward_admission.py). Run from the repo
root with all packages/*/src on PYTHONPATH, using
/tmp/posttrain-native-verl-update-runtime/bin/python -m pytest -c /dev/null
--rootdir=/home/hammad/projects/rl --import-mode=importlib -p no:cacheprovider
and those paths. Scoped Ruff/Pyright and import boundaries pass. Root API tests
encounter missing Torch; the isolated runtime closes that focused dependency gap.
The native CPU replay receipts, source archive, runner and SHA manifest remain
outside Git at /home/hammad/experiments/posttrain-correctness/2026-10-01/working/native-job-collection-r34;
remote /var/lib/posttrain/qualifications/native-job-collection-20261002-r34.
Image sha256:f0d32115d6e3bdca1dda76aad0ff8b9605e9869d4413ea65f76f4bc942cc85eb
loads retained selected Verifiers/TRL sources under the revision-11 qualification
mount. Input evidence SHA d3c83210e2ffdab0088a3eff6ef1284c4f6dae07e51f2e99133398f6c67c6fea;
new retained artifact SHA db9f0e35eedb8b4899d0f99e08d35ae42b66c28293301df1d35a9433aa533f0b.
No GPU launched. Fresh native ordinary jobs, veRL host composition, distributed
trainers, production refill/correction, observability and release adoption remain
open. Candidate runtime source hashing is not published dependency qualification.

Revision 33 completes original-evidence retention for admitted checkpoints in
the shared TRL/veRL save path. The regression relocates the complete checkpoint,
restores the typed population without an artifact callback, and confirms that
modified evidence fails before the decoder executes. It also saves a legacy
callback-only population, verifies resolver-free recovery rejects, and checks
explicit resolver recovery still works. This is CPU transport/identity coverage,
not another native GPU optimizer or fresh production job qualification. No GPU
was launched. Ordinary job composition, reference scoring, refill ownership and
version/capability selection remain open; public policy_updates stays guarded.
From /home/hammad/projects/rl set PYTHONPATH to packages/train/src,
packages/common/src, packages/data/src and packages/environment/src (colon
separated), then /tmp/posttrain-native-verl-update-runtime/bin/python -m pytest
-c /dev/null --rootdir=/home/hammad/projects/rl --import-mode=importlib
-p no:cacheprovider packages/train/tests/test_update_admission.py
packages/train/tests/test_update_inputs.py packages/train/tests/test_update_recovery.py
packages/train/tests/test_update_score_recovery.py -q (46 passed).
An initial command referenced nonexistent test_update_native_recovery.py and
collected no tests; the corrected command above is the validation evidence.
Root uv run pytest admission/input/collection tests: 23 passed, two Torch skips.
Scoped Ruff, pure-module Pyright (zero errors), nine import contracts and
git diff --check pass. New raw-evidence components require a real native GPU
checkpoint/recovery refresh before production release claims.

Revision 32 connects the existing TRL rollout collector to retained artifact
admission through collect_resolved_population, with a corruption check proving
failed admission does not submit an artifact or return optimizer work. The
tests inject generation/native decoding to isolate composition; they do not
qualify a fresh native job or vLLM synchronization. The ordinary
policy_optimization.py path still needs ResolvedTRLRun composition, selected
capabilities, collection rows, version/template identities and resume artifact
resolution. Public policy_updates guard remains in force. No GPU was launched.
From /home/hammad/projects/rl run uv run pytest
packages/train/tests/test_trl_resolved_collection.py
packages/train/tests/test_update_admission.py
packages/train/tests/test_verifiers_population_artifact.py -q (25 passed, two
Torch skips), scoped Ruff and Pyright on policy_rollouts.py (zero errors),
uv run lint-imports (nine kept), git diff --check. Root test_api.py selection
using -k 'rollout_adapter or sampo_rollout' yields five passed/one missing
Accelerate failure. In /tmp/posttrain-native-verl-update-runtime/bin/python use
all packages/*/src on PYTHONPATH, pytest -c /dev/null --rootdir=/home/hammad/projects/rl
--import-mode=importlib -p no:cacheprovider with those three test files and
test_api.py, -k 'rollout_adapter or sampo_rollout or collector or retained_artifact or portable_bridge'
(11 passed, 84 deselected before the additional final corruption case).

Revision 31 establishes a real retained-artifact-to-admission-to-backend-factory
path for SAMPO, GRPO and DAPO. This is CPU projection/credit/context validation,
not a model optimizer or ordinary job run. It retains the public launch guard.
External evidence is under
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/native-admission-r31
(check.py, source.tar.gz, result.json, manifest.sha256); remote copy is
/var/lib/posttrain/qualifications/native-admission-20261002-r31. Runtime image is
sha256:f0d32115d6e3bdca1dda76aad0ff8b9605e9869d4413ea65f76f4bc942cc85eb,
with retained r11 Verifiers/TRL source mounts and r19 candidate veRL source.
Input digest remains d3c83210e2ffdab0088a3eff6ef1284c4f6dae07e51f2e99133398f6c67c6fea;
retained population digest is ce51f1911bfe1b8847e53bfb9ebddc866c6d24ababf98d66302fcb96f2c9e1c9.
All three algorithms resolve two episode occurrences over three contexts/1,253
actions and match independent SAMPO or scalar-group credit references. The
native schemas ran in the real container; root artifact tests inject a decoder
only to isolate digest/reference checks. From repository root run uv run pytest
packages/train/tests/test_update_admission.py
packages/train/tests/test_verifiers_population_artifact.py -q (22 passed, two
optional Torch skips); scoped Ruff on the changed files, uv run pyright
packages/train/src/posttrain/train/backends/policy_update_admission.py (zero
errors), uv run lint-imports (nine kept), and git diff --check. No GPU was
requested. Production collection/execution and release qualification stay open.

Revision 30 validates CPU native decoding and materialization for two actual
LFM episodes in both episode/trace formats, each restoring three contexts and
1,253 action coordinates. This closes decoding evidence for the retained fixture,
not fresh production collection or optimizer wiring. No GPU was requested.
External evidence: /home/hammad/experiments/posttrain-correctness/2026-10-01/working/native-population-r30
contains check.py, source.tar.gz, result.json and manifest.sha256. Remote copy:
/var/lib/posttrain/qualifications/native-population-20261002-r30. Container:
sha256:f0d32115d6e3bdca1dda76aad0ff8b9605e9869d4413ea65f76f4bc942cc85eb;
native Verifiers source is the retained e6a3d9bbfe6959b97878f451fc721793a232cd5f
under hierarchical-engine-20261002-r11-a/verifiers-source. Input digest:
d3c83210e2ffdab0088a3eff6ef1284c4f6dae07e51f2e99133398f6c67c6fea.
Snapshot digests: episodes 623ba140c58cd5e2a82607be9a2e87d2dcae52736b60e48522dd4565ecc2eee4;
traces 8bb4ee337038a69213b0dadd29da7aa0308c31afb07f4747817fe53dca097da5.
From repository root run uv run pytest packages/train/tests/test_verifiers_population_artifact.py -q
(11 passed); uv run ruff check packages/train/src/posttrain/train/integrations/verifiers_population_artifact.py
packages/train/tests/test_verifiers_population_artifact.py; uv run pyright
packages/train/src/posttrain/train/integrations/verifiers_population_artifact.py
(zero errors); uv run lint-imports (nine kept); git diff --check. Retained fixture
qualification binds the archived source; subsequent test-only import formatting
does not change its executed decoder. The goal remains active.

Revision 29 closes the collector's immutable snapshot boundary with seven focused
tests. Ordinary TRL/veRL jobs still need collection/admission/execution wiring;
no production launch, native GPU run or distributed release gate is claimed.
The snapshot retains exact source lines and survives subsequent source growth;
changed existing content-addressed artifacts fail without being overwritten.
Run from /home/hammad/projects/rl: uv run pytest
packages/train/tests/test_verifiers_population_artifact.py -q (seven passed),
uv run ruff check packages/train/src/posttrain/train/integrations/verifiers_population_artifact.py
packages/train/src/posttrain/train/integrations/verifiers.py
packages/train/tests/test_verifiers_population_artifact.py;
uv run pyright packages/train/src/posttrain/train/integrations/verifiers_population_artifact.py;
uv run lint-imports; git diff --check. No GPU was launched in this revision.

Revision 28 extends shared distributed execution from detached score arithmetic
to coordinated full-context native DDP forwards, backward and optimizer
transitions, including empty ranks and pre-forward artifact rejection. Thirty-three
focused native tests pass. Observed BF16/FP16 rounding differences remain visible;
only rank-reduced gradients are exact across the two participating ranks. The
completion audit still finds six major open production/qualification/release
gates recorded in Progress. The goal remains active and public launch stays
gated; no experimental tool or receipt is placed in this repository.

Revision 27 supplies reusable distributed ownership, score gathering, routed
adjoints and a shared finite decision, with actual CPU collectives validating
unequal and empty ranks against independent global gradients. Thirty-eight
focused native math tests pass. This advances distributed execution beyond
arithmetic-only weighting, but does not close native TRL/veRL multi-GPU execution,
collective scheduling, synchronized scaler transactions or distributed recovery.
Production launch, informative Gemma coverage, runtime adoption and full release
checks also remain open. Experimental runners/raw receipts stay outside Git;
the goal remains active.

Revision 26 closes the private turn-row mid-population native recovery gate for
the retained LFM fixture in both supported major precisions and both backends.
Twenty physical applied updates and verified checkpoint artifacts show exact
within-backend continuation through clipping and population rollover. Forty-two
focused native tests, nine import contracts and diff checking pass. Both consumer
pages record the bounded evidence. Production collection/launch wiring, distributed
execution, informative Gemma updates, candidate fork/runtime adoption and full
release checks remain open; the full goal remains active. No experimental
runner, raw receipt, dependency pin or generic fork edit enters this repository.

Revision 25 resolves the observed LFM turn-row BF16 discrepancy with a matched
initialization counterfactual and exact evidence at every update boundary.
The existing two backend paths execute the same bounded objective and optimizer
work once initialized identically. This is stronger evidence than comparing
aggregate clipped counts. Five native turn-row tests pass and diff checking
passes; experiments remain outside Git. Public collection/execution wiring,
distributed execution, informative Gemma updates, new-variant native resume,
fork/runtime adoption and full release checks remain open. The goal remains
active; no whole-engine completion or general recipe superiority is claimed.

Revision 24 implements and exercises the explicit turn-row objective in both
native backends and major precisions, rather than leaving it as a recommendation.
Independent gradients and actual updates support its ratio/reduction mechanics.
Fully clipped FP16 updates demonstrate functioning clipping and expose the
importance of optimizer momentum in interpreting parameter movement. The BF16
backend difference is unresolved and remains a qualification gate. New-variant
native resume, production wiring, distributed execution, informative Gemma,
runtime adoption and full release checks remain open; the goal remains active.

Revision 23 supplies a common durable-evidence handoff to both native backends
and strengthens collection provenance at actual sampled-token and task-group
boundaries. Seventy-one isolated tests pass and three real native projection
arms validate current admission. The host must still persist authoritative
artifacts, select complete distinct tasks, and wire collection/restoration into
the production online launch. Distributed execution, informative Gemma coverage,
runtime adoption and full release validation remain open; goal stays active.

Revision 22 closes the private native mid-population typed recovery gate for
LFM in both backends and major precisions. Twenty applied updates produce exact
same-engine resume equality, including active clipping. Import boundaries and
diff checks pass; focused recovery tests pass after repairing one stale fixture.
The complete suite is not green in the isolated runtime. Public production
collection/recovery wiring, distributed execution, informative Gemma coverage,
runtime adoption and full release validation remain open. Goal stays active.

Revision 21 provides typed retained-population recovery and validates its native
CPU composition plus eight previous GPU artifacts. Scoped Ruff/Pyright, import
boundaries and diff checks pass. No new GPU execution is claimed. Production
collection/recovery wiring, distributed execution, informative Gemma coverage,
runtime adoption and release checks remain open; the goal remains active.

Revision 20 adds and qualifies shared native input admission/replay rather than
relying on experiment-specific callbacks. Twelve real model optimizer updates
show unchanged behavior in both backends and major precisions. This advances
production input composition without closing its still-missing collection,
checkpoint restoration or distributed execution gates. Informative Gemma,
runtime adoption and release validation remain unfinished; goal stays active.

Revision 19 closes the private LFM native veRL continuous-population and
deterministic shared-boundary resume checks in both major precisions. Fourteen
physical optimizer updates complete with source-bound raw evidence outside Git.
It adds independently checked GRPO/DAPO scalar credit and native turn projection
for explicit engine selections. Production collection-host integration,
distributed execution, informative Gemma optimizer evidence, candidate runtime
adoption and full release validation remain open. Keep the overall goal active.


Revision 18 adds continuous private native execution without restarting TRL's
optimizer lifecycle. Native CPU recovery proves model and Adam state agreement
across two populations. The GPU campaign applies 12 primary and two resumed
LFM updates across BF16/FP16, verifies exact shared-boundary resume, and confirms
both clipping and refreshed population old scores. Scoped Ruff/Pyright, all nine
import contracts and diff checks pass. Experimental tools/receipts remain outside
Git. Public production collection and the remaining native/backend/distributed
release gates are not complete; leave the overall goal active.

Revision 17 fixes remaining Posttrain telemetry and parallel-observation
projection gaps after source inspection and manual review of real trajectories.
UUID normalization is preserved. The selected TRL already contains the
token-local sequence-clipping correction; no switch to token ratios or formula
reversion was made. Selected native fork, Posttrain SAMPO/credit/resolution,
probe and actual TRL/veRL loss-parity tests pass (73 tests). Recovery/native
offset tests separately pass (38). Scoped Ruff/Pyright and all nine import
contracts pass. Public engine activation, continuous native recovery, full
environment-state equivalence and recipe-quality ablations remain open.

Revision 16 connects typed selection resolution to both private native adapters
and closes late objective/pack validation gaps. Twenty-eight source-selected CPU
native/protocol/resolution tests pass. The production multi-population collection
and checkpoint lifecycle remains the next integration requirement; GPU receipts
remain bound to their earlier exact adapter sources and are not silently promoted.

Revision 15 fixes the confirmed SAMPO reward-shaping discrepancy in the resolved
native adapter and adds admitted-rollout assembly with process-credit transport.
Focused framework regressions and actual candidate native loss parity pass.
Earlier raw receipts retain their exact source and credit identities; they are
not relabeled as qualification of the changed @2 credit implementation. Public
collection/recovery composition, matched native model gradients, distributed
execution and release adoption still require completion.

Revision 14 establishes fresh BF16 Gemma 4 native collection and exact original
context/action-mask transport, followed by equivalent FP16 collection coverage.
It does not qualify productive Gemma updates: both complete groups supply zero
credit. Public collection lifecycle and independent native model
gradients remain open; successful collection alone cannot close those gates.

Revision 13 records eight native LFM applied updates across four schedule/precision
arms and six additional applied updates in shared-boundary control/resume checks.
Both fresh-process resume comparisons are bit-exact across all 24 trainable
matrices, final events and native/resolved/scaler counters. These are fixed-evidence
within-backend checks using the current TRL guard implementation. Public lifecycle,
independent matched native gradients, distributed execution, fresh process credit,
Gemma 4 and veRL family coverage, and release assets remain incomplete.
The typed resolution boundary additionally passes 42 focused tests, scoped Ruff
and Pyright, all nine import contracts, and diff whitespace checks. Regression
coverage includes independent recipe coefficients, structured-credit transport,
reasoning selections retaining episode-wide dependencies, and rejection of a
later occurrence's oversized context before any executable population is returned.
No public launch, fork pin, or runtime asset is promoted by this change.

Revision 11 adds native TRL overflow retry and explicit failed-attempt ownership.
Sixteen optional native/score-transport tests pass after the final local guards;
131 framework/environment tests, scoped Ruff/Pyright, nine import contracts and
diff checks pass. Actual Qwen FP16 retry plus post-retry resume comparisons are
bit-exact at their recorded source identities. Public launch remains gated;
these results do not close distributed, independent native model-gradient,
fresh collection/process-credit, multi-family or final runtime/pin adoption.
Direct SSH and immutable remote cache/runtime preflight establish the next
available larger-model execution route without launching or disturbing a job.

Revision 10 completes the deterministic BF16 same-boundary GPU continuation
control on both native backends. Together with revision 9's FP16 controls, both
precisions have bounded within-backend resume evidence for Qwen3.5-0.8B at the
recorded source/runtime identities. This is not completion of the engine: public
composition, independent native model-gradient parity, distributed normalization,
fresh rollout/process credit, additional model families and release adoption
remain open. The strengthened neutral recovery validator requires retained
population metadata; all 16 pure recovery tests and 12 optional native
TRL/veRL/score-transport tests pass afterward. Scoped Ruff/Pyright and diff checks
pass. No experimental runner, native receipt or model file was added to Git.

Revision 9 adds update_recovery.py and backends/policy_update_recovery.py and
integrates native checkpoint hooks in the private TRL and veRL adapters. The
manifest binds component paths, sizes and hashes. Atomic metadata publication
cannot expose an incomplete seal, and identical repeated commits validate
without rewriting native files. Recovery rejects source/runtime/world-size,
evidence, schedule, correction or counter changes before restoring the cursor.
Population/credit/spec/objectives/occurrences are retained once per checkpoint,
alongside original native evidence references, without a new token store.
Unsealed partial directories are not automatically deleted or resumed.

Native CPU TRL continuation stops at update 1 and resumes update 2 through the
actual model/optimizer/scheduler/RNG loading path. Final parameters match the
uninterrupted control at rtol/atol 1e-6; population-frozen old scores match exactly.
Native TRL plus score-recovery tests: six passed. Twelve pure seal tests pass.
These do not certify GPU, FP16 scaler, distributed or production recovery.
The veRL adapter rejects saving an incomplete update, a changed scheduler cursor
or a scaler not owned by its native checkpoint manager. A native exception leaves
the pending occurrence set, requiring recovery before another run_update call.

The external candidate runner working/engine_native_verl_resume.py is retained
under /home/hammad/experiments/posttrain-correctness/2026-10-01. It uses the clean
published acad521 candidate, locked Transformers/tokenizers and separate
interrupted/resume/control processes. Resume intentionally changes startup seed
and FP16 startup scale; native checkpoint restoration must recover retained
weights and scale. FP16 interrupted/save and separate-process continuation both
complete. The retained scale 128 is restored from startup scale 32, along with
LoRA-only weights, optimizer, scheduler, RNG, old scores and next occurrence 1.
The second applied update has 104 clipped actions, zero new gradient and nonzero
Adam-momentum parameter motion. Repeated save validates without rewriting native
files. Peak save allocation is 5.829 GB.

The uninterrupted control exposes a failure of strict parameter agreement:
24 matrices differ by up to 3.2963e-4 (L2 0.005457), while second loss, clipping,
old scores, applied/attempt/scheduler counters and scaler states match. Initial
control/interrupted LoRA hashes agree. This is not a passing GPU resume gate.
The checkpoint reload probe next retains parameters immediately after native
load, before another forward/update, to distinguish restoration from independent
first-update variability. Do not dismiss the difference with the CPU fixture's
tolerance or claim a native-framework defect before identifying its source.
Receipts: engine-verl-fp16-resume-{interrupted,restored,control,comparison}.json;
raw logs/checkpoint state stay under results/native-collection.

The reload probe now finds zero weight error across all 24 matrices immediately
after loading, and repeated continuation from that same checkpoint is bit-exact.
Retained receipts are engine-verl-fp16-restore-exactness.json and
engine-verl-fp16-resume-repeatability.json; the probe instrumentation hash is
recorded separately. Thus weight restoration is verified, while the independent
first-update discrepancy remains a numerical reproducibility investigation.
Resume equality must use a shared prefix: run update 1, save a boundary and keep
the same engine running through update 2, then resume a new process from that
exact saved boundary. Do not relax tolerances to hide independent-prefix drift.
The external runners now expose --mode branch for that control. Original runner
bytes are retained as *_resume_v1.py so older checkpoint identities remain
reproducible. The FP16 veRL shared-boundary comparison is bit-exact across all
24 matrices, loss, clipping, old scores, scheduler/applied/attempt counters and
scaler growth tracker. This qualifies that candidate's tested resume path;
independent-run numerical reproducibility remains a separate open gate.
Receipts are engine-verl-fp16-branch-{control,restored,comparison}.json. No source
or consumer pin changed to explain away the independent-prefix discrepancy.

The parallel next backend path is the external engine_native_trl_resume.py
runner (executed serially on the GPU). Its FP16 interrupted first update saves
native model/optimizer/scheduler/RNG/scaler and the resolved seal successfully.
Separate-process continuation/control comparison is next. Production launch,
native GPU resume equality and the remaining release gates remain open.

Native TRL's FP16 shared-boundary continuation also passes with zero error across
24 matrices, exact objective/clipping/old scores, native/resolved/scheduler
counters and scaler growth tracker. Its saved scale 128 overrides resume startup
scale 32. Receipts are engine-trl-fp16-branch-{control,restored,comparison}.json.
The first veRL BF16 branch used a mistyped overlay path and actually ran
Transformers 5.16.1/tokenizers 0.23.2. It completed two updates, but must not count
toward locked 5.14.1/0.22.2 qualification. Its second-update clipping count 88
differs from the earlier locked-runtime BF16 controls, reinforcing the need to
bind runtime identities before comparisons. Retain this alternate-runtime receipt
and checkpoint; the new locked retry uses fresh -locked paths. No attempt is
overwritten or mislabeled as a lock-qualified success.

Locked BF16 shared-boundary comparisons complete on both backends but remain
open for parameter agreement: maximum matrix errors are 6.2473e-5 (veRL) and
6.7978e-5 (TRL). Loss, clipping (88), old scores and all applied/native counters
match; veRL gradient-norm difference is 0.0447%. These are observed errors, not
accepted tolerances. Receipts are engine-verl-bf16-branch-locked-comparison.json
and engine-trl-bf16-branch-comparison.json. Checkpoint restoration alone cannot
prove the next low-precision backward is numerically reproducible.

The independent Adam readback (working/engine_adam_prefix_audit.py) uses retained
FP32 moments and the optimizer's actual beta/epsilon/decay to reconstruct native
first B updates (12 matrices) and zero-new-gradient second updates (24 matrices).
Maximum errors are 2.18e-11 and 2.91e-11. Separate first prefixes have gradient
relative L2 difference 0.562% with 323 sign-flipped coordinates; the largest
gradient at a flip is 4.43e-7 versus epsilon 1e-8. This explains how small
gradient changes can become larger Adam parameter differences. It does not prove
the underlying model-gradient/kernel cause. Retain engine-adam-prefix-audit.json.

The explicit deterministic CUDA control sets
torch.use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG=:4096:8,
retaining both in runtime identity. Original runner bytes are preserved as
*_resume_v2.py; deterministic paths use fresh checkpoint/receipt names. Native
veRL BF16 control and fresh-process resume now match all 24 LoRA matrices exactly
(maximum error and difference L2 both zero), second loss, 88 clipped actions,
old scores and counters. Both gradient norms are 0.0010011331178247929.
Receipts are engine-verl-bf16-deterministic-{control,restored,comparison}.json in
the external campaign results/native-collection directory; runtime identity is
31db90269cce09da5da78d503c0ca2a33f91ada9731f1174e5e43ccee1446f26.
TRL's equivalent deterministic control and fresh-process resume also match all
24 matrices exactly, gradient norm 0.0009883747708201434, loss
-0.006738131865859032, 88 clipped actions, old scores and counters. Receipts are
engine-trl-bf16-deterministic-{control,restored,comparison}.json; runtime identity
is c68187bd93deffee666aa4cccfbb4f390548e00b30999485f4f502ab18326e7d.
Within each backend the restored process starts with a different initialization
seed and resumes from the exact same saved first-update boundary. No across-
backend initial-state parity is claimed. This is a correctness/reproducibility
control, not a recipe or performance recommendation; default-runtime margins
remain a release gate.

Revision 8 adds private backends/policy_update_replay.py and
backends/verl/policy_updates.py, with test_update_replay.py and test_update_verl.py.
Complete objective derivatives are computed over detached current-score leaves;
native backward replays each original context and contracts its score derivatives
with the frozen adjoints. The scalar carrier is not the reported objective.
Replay rejects parameter-version changes and score drift. Public launch remains
closed; deterministic real-model replay, distributed execution and recovery
still require qualification. Native protocol plus replay tests: 13 passed before
the selected-output hook was added; its independent value/gradient tests are next.
Scoped source Ruff/Pyright and nine import contracts pass at that checkpoint.

Native FSDP runners and failure logs remain outside Git at
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/engine_native_verl_run.py
and results/native-collection/engine-verl-qwen-*.log. Preserve the missing padding
helper, incompatible Transformers, tokenizers import, stock-scoring OOM and
activation-offload OOM attempts. Successful two-minibatch receipts are
engine-verl-qwen-bf16-two-minibatches-selected.json and
engine-verl-qwen-fp16-two-minibatches-selected.json in results/native-collection.
Each applies two updates in two attempts; peak allocation is about 5.048 GB.
BF16 second update has zero clipped actions; FP16 has 16, zero new gradient norm,
and a nonzero parameter transition from Adam's retained moments. These are
observations of this fixed small population, not recipe-quality conclusions.
The adapter uses an explicit Posttrain selected FP32 tempered score contract,
through an engine subclass's existing prepare_model_outputs seam; it does not
claim use of the stock veRL probability kernel. Native full-context model,
backward, optimizer, scaler and scheduler remain unchanged. No fork source or
consumer dependency pin has changed.

Selected-output value/gradient tests pass independently against full vocabulary
FP32 log_softmax for FP32/BF16/FP16 inputs. Combined objective, scoring,
execution, replay and veRL protocol slice: 55 passed. Neutral environment and
contract regressions: 115 passed. Runtime version receipts bind Transformers
5.14.1, tokenizers 0.22.2, Torch 2.13.0, PEFT 0.21.1 and TensorDict 0.10.0.
veRL uses native default AdamW weight_decay 0.01; the earlier TRL runner leaves
its default untouched. Explicitly match optimizer settings, initial trainable
weights and runtime identities before claiming cross-backend update parity.
Full-population/two-epoch native veRL checks also pass, bringing this candidate
matrix to eight applied updates in eight attempts. Peak allocation is 5.048 GB
for both schedules. Second-update clipped-action counts are:

| Precision | Two episode minibatches, one epoch | One full-population minibatch, two epochs |
| --- | ---: | ---: |
| BF16 | 0 | 0 |
| FP16 | 16 | 104 |

These schedules have 298 versus 596 action exposures; do not interpret them as
matched-work speed or quality comparisons. Full-population receipts are
engine-verl-qwen-bf16-full-population-selected.json and
engine-verl-qwen-fp16-full-population-selected.json. They additionally bind
production source hashes and FP32 trainable storage. Failed allocator mapping
warnings may appear in successful logs; terminal exit zero, applied counters and
receipts prove these attempts completed. Real gradient references, exact initial
state/optimizer parity, resume, fresh rollout and distributed qualification remain
the next gates. Public policy_updates launching stays closed.

From /home/hammad/projects/rl, run the native candidate with the external
Transformers overlay and /home/hammad/projects/verl-vortex prepended to the
revision-7 runtime PYTHONPATH (omit the detached TRL source for this runner).
Use PYTORCH_ALLOC_CONF=expandable_segments:True and
VERL_DISABLE_FLASH_ATTN_CE=1, then invoke
/tmp/posttrain-native-verl-update-runtime/bin/python with
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/engine_native_verl_run.py
and --dtype bfloat16 or float16, --minibatch 1 or 2 and a fresh --output path.
Retain stderr/stdout to a separate fresh .log; check GPU compute processes first
and wait for the previous handle's terminal exit before another launch. Do not
retry by overwriting prior receipts or by treating allocator warnings as terminal.
Optional native tests use the same runtime/source prefix and
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, then `python -m pytest -c /dev/null
--rootdir=/home/hammad/projects/rl --import-mode=importlib -p no:cacheprovider`
with test_update_verl.py and test_update_replay.py selected (16 passed).

After the candidate runs, separate native replay carrier scalars from the true
objective in the adapter result: loss/policy_loss/kl_loss report complete resolved
arithmetic; replay_carrier_losses retains native backward diagnostics. This
observation repair does not change gradients or optimizer transitions. Its
protocol regression passes; full release manifests still require current-source
qualification rather than reusing a pre-edit receipt hash.

Implementation has begun with the canonical amendment and pure contracts in
update_records.py, update_credit.py and update_plan.py, plus additive
SpanAssessment. Focused new and existing tests report 80 passed; targeted source
type checking reports zero errors and all nine import-boundary contracts pass.
The scheduler reproduces two episodes/five turns/three packs/one resolved update,
records reuse occurrences and discarded final contributions, and rejects
unqualified cross-pack dependencies. These are CPU contract results only.

Revision 4 adds update_objectives.py and private backends/policy_update_math.py.
Independent references demonstrate equal forward values but different coupled
versus token-local sequence gradients, both clipped sign branches, detached
CISPO weights and near-zero sampled-k3 derivatives. CPU BF16/FP16 logit gradients
match an explicit monolithic formula under one/two/five-record execution packs.
Pure distributed weighting checks unequal and empty partitions under assumed
gradient averaging; no real distributed collective has run. Built-in credit
transport delegates to existing estimators without changing their arithmetic.

Revision 5 adds typed catalog decoding, original native conditioning projection,
population_from_rollouts and private sampled_logprobs. Complete-group assembly
delegates SAMPO credit while preserving native observation exclusions. Retained
real provider inputs agree with native path materialization for five calls and
298 actions. Pure tests reject context mismatch, forged coordinates, mixed
sampler versions, incomplete groups and truncated logit outputs. Focused slice
115 passed; isolated Torch objective/scoring 30 passed. Scoped Ruff/Pyright pass.
The guarded settings are not yet executable support.

Revision 6 adds score_actions, FrozenPopulationScores and freeze_population_scores
in the private scoring module, plus compute_resolved_loss in private
backends/policy_update_execution.py. It checks population/version/temperature/
score-contract identity, complete dependency-pack coverage and context footprints,
then evaluates one loss with all model graphs retained. Old/reference scores remain
detached through two transitions; current scores refresh and context embeddings
retain gradients. Combined CPU suite: 39 passed. This operator fixture does not
certify native TRL/veRL, a pretrained architecture, GPU execution, or a checkpoint.

Live built-in estimator wiring, semantic extraction/composition,
OLMo3 recipe-contract resolution, live native execution, veRL integration,
distributed collectives, checkpoint transactions and full native gradient/model
qualification remain incomplete. Basic fixed-evidence TRL GPU transitions now
pass, as detailed below. No fork edits, publication, consumer pins or active
production-run changes have occurred. The local GPU was most recently observed
at about 1397 MiB desktop usage of 8192 MiB; availability must be rechecked before
training. Source inspection alone does not qualify scheduling equivalence.

Revision 7 native TRL fixed-evidence results use Qwen3.5-0.8B checkpoint
2fc06364715b967f1860aea9cf38778875588b17, TRL d97a2cf, native Verifiers e6a3d9b,
Torch 2.13.0+cu130, LoRA rank8/alpha16/dropout0, FP32 trainable parameters,
LR1e-4, score temperature0.8, beta0 and unchanged SAMPO clip settings. The retained
AutomationBench source has two episodes, five calls and 298 eligible actions;
group metadata is explicitly derived for this fixed-evidence check, rewards are
unchanged, and existing SAMPO computes credit. Sampler correction is explicitly
absent for this actor-objective check; sampler-gap qualification remains open.

| Precision | Schedule | Applied updates | Current action exposures | Clipped actions, second update | Peak allocated GB |
| --- | --- | --- | --- | --- | --- |
| BF16 | two episode minibatches, one pass | 2 | 298 | 0 | 5.160 |
| FP16 | two episode minibatches, one pass | 2 | 298 | 16 | 5.186 |
| BF16 | one full-population minibatch, two passes | 2 | 596 | 88 | 5.215 |
| FP16 | one full-population minibatch, two passes | 2 | 596 | 104 | 5.215 |

Old-score digests remain unchanged within each arm and agree across schedules
within each precision. Native global_step, attempts and applied_updates equal2.
The BF16 diagnostic repeat adds two further updates (ten successful applied
updates total including the initial four arms). It records per-step L2 transitions
0.046686 and0.037584. FP16 full-population records0.046443 and0.030998 despite the
second gradient being zero. This validates basic optimizer/counter continuity,
not independent real-model gradient equivalence, resume, quality improvement,
precision equivalence, fresh collection, distributed behavior or veRL parity.
An initial BF16 gradient norm differs from its diagnostic repeat by about7.3e-6;
component-level reproducibility/gradient qualification remains open.

## Context and Orientation


Work from /home/hammad/projects/rl. The planning snapshot is framework HEAD
c81622ae5d1fa3803ff5ecfb286878948709ab3a with substantial unrelated dirty work.
The reviewed proposal is docs/research/hierarchical-policy-update-engine-proposal.md;
its review-branch revision is fd424001d3cd5bcfd23acb9e942c301df83de8fa.
This plan embeds its required decisions rather than relying on chat history.

packages/train owns public policy settings, evidence-to-credit arithmetic and
private adapters. profiles.py defines TrainingLoop, GRPOSettings, SAMPOSettings,
GDPOSettings and CAPOSettings. catalog_schema.py serializes those selections.
requests.py and api.py validate and dispatch operations. bindings.py owns backend
execution choices. Preserve SFT, DPO and distillation behavior unless an explicit
later milestone extends their contracts.

reward_evidence.py defines RewardValue, RewardEvidence and ProcessCredit;
turn_rewards.py defines TurnTokenMap, TurnAssessment, native_turn_map and
validate_turn_assessments. reward_projection.py defines versioned projections.
sampo_advantages.py computes group-relative episode and anchor-state turn credit;
reward_advantages.py computes GDPO/CAPO credit. Reuse these implementations behind
explicit adapters rather than creating a parallel reward pipeline. CAPO's
ProcessCredit.error_mask unions overlapping error spans; preserve that rule.

backends/trl/policy_config.py builds native arguments and runtime evidence.
policy_rollouts.py bridges sampled trajectories, policy_optimization.py constructs
trainers, precision_runtime.py manages numerical choices, and cancellation.py
retains checkpoints at complete update boundaries. backends/verl/launcher.py
builds serialized launch plans, contracts.py validates isolated-worker payloads,
worker.py translates them to Hydra options, agent_loop.py bridges environment
records, and metrics.py normalizes native telemetry.

packages/environment/src/posttrain/environment/verifiers_runtime.py and
verifiers_evidence.py own native runtime compatibility and trace evidence.
Generic native-record compatibility belongs there. Train-owned span annotations
remain derived references to native traces, not a second trajectory store.
packages/common must remain free of Torch, TRL, veRL and tracking SDKs. Train,
serve and eval must not import one another; composition injects scorer clients.

At inspection, packages/train/pyproject.toml and uv.lock select TRL
1.12.0.post13, source d97a2cf619f94c7076a720116d75f85dfd62382b. release/forks.toml
selects veRL release carbonteq-v0.9.0.post8; the veRL runtime profile records
upstream_revision 483b8a009ba3a97563edee3a19887e4862b8094a, which is an upstream
base and must not be substituted for the published fork identity. Resolve the
actual fork commit from retained release records before implementation.

Sibling checkouts inspected on 2026-10-02 are ../trl at c9af78c1c2ea04ad271e95b26b93dfadf8b9fca1
on codex/bounded-vllm-waves; ../trl-sampo-local-credit at
255aa649636410f0c06b6ecfb53e941b40357ccf on codex/sampo-local-credit;
../verl-upstream at 78266f97dc0a57c922f7fd8fc14f6a4b8ac91028 on
codex/verl-rollout-execution; and ../verl-posttrain-parity at
acad5211619aef5ed80b25e9657262f08a436b03 on codex/posttrain-math-parity.
All inspected origins point to their CarbonTeq forks; upstream remotes point to
Hugging Face TRL or verl-project. These checkout identities are evidence, not
consumer pins. Recheck them and sibling AGENTS.md before changes.

An episode is a native task trajectory, possibly containing multiple branches.
A turn is a sampled assistant response. A span is one or more half-open original
token intervals; [start,end) includes start but excludes end. Named populations
are memberships used by estimators, such as same-prompt episodes or turns sharing
an anchor observation. A contribution is an objective term over selected actions.
A dependency closure is all records/statistics required for its values and
derivatives, including records without independent selected loss. A minibatch
determines one optimizer update. A pack determines one bounded execution portion.
An epoch here is one declared pass over a frozen rollout population, not a pass
over an environment dataset.

## Product Baseline and Scope


Before code, record a decision under docs/decisions following existing numbering,
then narrowly amend docs/post-training/README.md, 02-primitives.md, 04-framework.md,
05-apis.md and 06-observation-and-lineage.md. Define public schedule and execution
controls, derived semantic spans, named population completeness, objective
identity, new evidence and recovery counters. Explicitly remove the segmentation
deferral for qualified supplied annotations while leaving automatic semantic
extraction algorithm/plugin-owned. Do not add a framework-owned rubric.

Amend task uniqueness to apply to fresh collection/refills within one population.
Deliberate reuse of its frozen contributions is not duplicate generation.
Admission rules remain algorithm-specific; this plan does not replace SAMPO's
episode-spread gate with a new credit-aware gate.

Initial production scope is synchronous first-order autoregressive text-policy
updates. Preserve GRPO/DAPO/OLMo3, SAMPO, GDPO and CAPO. New engine qualification
also exercises GSPO-token-style derivative routing, detached-weight CISPO,
semantic policy/KL selection and supplied step-level credit. Standalone public
GSPO/CISPO selections are separate additions only after their identity and native
qualification are recorded. No arbitrary objective compiler, multimodal
qualification, asynchronous replay correction, higher-order optimizer,
differentiable simulator, or online reward-model/critic training is included.
Teacher-distribution and critic-based PPO examples are design pressure tests:
they must either declare sufficient inputs or be rejected without approximation.

## Resolved Contracts and Interfaces

### Native reuse audit (source inspection, 2026-10-02)

| Selected source | Existing capability | Required equivalence gate |
| --- | --- | --- |
| TRL d97a2cf619f94c7076a720116d75f85dfd62382b | `num_iterations`, `steps_per_generation`, buffered generation slices and frozen old-score path | Resolve optimizer accumulation versus generation slicing; preserve old scores across multiple applied updates and complete-population advantages |
| Same TRL source | Sequence ratio with token-local derivative routing for token-aligned advantages | Compare mixed-sign gradients independently; do not classify the existing routing as an upstream defect |
| veRL ef1c37715fa75de5973ae5b3c398383cd7e0093d | engine_workers minibatch size/count, epochs, data-parallel division and per-minibatch train_batch | Confirm expanded native row units, shared ordering and episode dependencies; native scheduler marks LR advancement on the final minibatch, requiring explicit applied-update semantics |
| Same veRL source | FSDP ShardedGradScaler, unscale-before-clipping and skipped-step evidence | Qualify synchronized overflow, bounded retry and cursor/LR behavior with actual native execution |

These observations support reuse, not blanket capability registration. No fork
is changed until independent objective and native optimizer evidence shows a gap.
Pure `ObjectivePopulation` currently represents internal already-resolved atomic
contributions; it is not exported as a freely composable user objective DSL.
`ExecutionCapabilities` is supplied by tests only so far; native capability
resolution must bind actual qualified source/runtime identities.


Create train/update_records.py, update_plan.py, update_objectives.py and
update_credit.py. These modules use pure Python immutable records; tensor storage
and autograd remain private to adapters. Export only stable user settings and
qualified evidence values through train/__init__.py. Do not expose every internal
record as a catalog primitive.

In update_records.py define ActionRef(episode_id, branch_id, turn_id, token_index),
ConditioningView(id, native_ref, token_ids_ref, attention_ref, positions_ref,
template_revision, digest), SemanticSpan(id, role, projection_revision,
action_intervals), PopulationRelation(id, kind, members, completeness),
PolicyVersions(sampler, old_score, current, reference), and PopulationSnapshot.
A snapshot references native evidence, admitted records, annotations, versions,
frozen credit/selector state and their digests. It does not copy raw trajectories
into another store. Store backend tensors behind opaque adapter handles.

The bridge must retain the exact conditioning view for each sampled action:
rolling context, summaries, branch continuations and tool truncation included.
Never reconstruct from final transcript text. If legacy records cannot prove
that view, admit them only through a qualified legacy algorithm path or reject
the new feature. Policy eligibility excludes system/user/tool-result/padding
positions independently of context attention.

Extend reward_evidence.py additively with SpanAssessment: evidence_ref, span_id,
component values/statuses, scorer revision, observed_input_ref, observation_scope
(prefix/current-step/full-trajectory), semantic_kind and scorer_snapshot.
Do not reinterpret ProcessCredit as a floating PRM reward. update_credit.py
adapts existing estimators and validates external credit. A score, reward, value
and advantage are different typed meanings. Missing values remain unavailable.

A qualified external estimator can return PreparedCredit with original ActionRef
values, estimator identity, required complete relations, component weights,
normalization, observation scope and evidence digests. Initially accept supplied
advantages only with a declared detached provenance and validation contract;
provide an integration fixture with an explicit external estimator rather than
inventing a recommended PRM algorithm. Existing SAMPO discounted-return and
CAPO/GDPO adapters provide real built-in credit preparation. An external scorer
returning step quality without a supported estimator fails before optimization.
Future estimators must declare temporal boundaries, discount unit, bootstrapping
and whether hindsight inputs are valid for their interpretation.

In update_plan.py define PolicyUpdateSchedule(unit, budget, epochs, order,
seed, final_policy, max_applied_updates) and PolicyExecutionBudget(records,
context_tokens, statistic_bytes, oversized_policy). Schedule unit is episode,
turn or selected-token budget; contributions remain atomic according to the
objective definition. Token budgets may admit the final atomic contribution
above a soft schedule budget, but never truncate it. GPU budgets are hard capacity
checks. Default final_policy is include; drop must identify discarded contributions
and error must reject the incomplete schedule. Empty objective reductions are
resolved before scheduling, not hidden as an empty GPU batch.

Define ContributionRef, ResolvedObjectiveTerm, ResolvedUpdate, ExecutionPack,
UpdateCursor and UpdateOutcome. A ResolvedUpdate identifies contribution
occurrences, credit and objective versions, reductions, dependency closure,
selector refresh policy and parameter version. Reuse creates explicit occurrence
identities over the same original actions rather than duplicating action records.

The pure entry points must be:

    resolve_updates(snapshot, schedule, objective) -> tuple[ResolvedUpdate, ...]
    plan_packs(update, execution_budget, capabilities) -> tuple[ExecutionPack, ...]
    prepare_credit(snapshot, estimator) -> PreparedCredit
    validate_update(update, capabilities) -> None

Schedule resolution does not import ML libraries or fetch models. Statistics
whose selectors depend on current parameters are resolved at update execution
under the declared scope, then frozen across that update's score/backward passes.
Population-frozen selectors retain their original values across reuse. World-size
and GPU packing cannot change logical selection unless a separately named
microbatch-dependent objective intentionally declares that behavior.

In update_objectives.py define a closed initial registry of ObjectiveDefinition
implementations. Each specifies identity/revision, required statistics, contribution
granularity, ratio formula, ratio value support, derivative routing, sampler
correction, policy/KL selectors, independent reductions, allowed credit meanings,
and supported scheduling/execution combinations. Validation must reject an
unsupported combination before model training begins; no backend-options escape
hatch may override resolved semantics.

The adapter-owned tensor interface accepts the same definition and update:

    score(update, parameter_version) -> ScoreBundle
    evaluate(update, scores) -> ObjectiveEvaluation
    backward(update, evaluation, packs) -> GradientEvidence
    apply_update(update, cursor) -> UpdateOutcome

ScoreBundle advertises actual statistics, such as sampled log probabilities,
reference scores or predictive entropy. ObjectiveEvaluation carries the true
loss and derivative recipe. A private implementation may use autograd or detached
adjoints, meaning objective derivatives with respect to model outputs. Arbitrary
tensor callbacks are not a public serialization contract.

Preserve existing reductions exactly. For equal-episode mean, explicitly resolve
E, each selected set S_e, weights w_t and denominator d_e:

    L = (1 / len(E)) * sum(sum(w_t * loss_t for t in S_e) / d_e for e in E)

A selected-token denominator differs from original eligible-token count.
Each term chooses reject, omit with declared renormalization, or zero contribution
for empty selection. Overlap set operations do not define credit arithmetic:
CAPO error masks retain union semantics; other credit projections declare whether
scores combine before clipping or independent terms are added afterward.

For a geometric sequence ratio use exp(mean(current_logp-old_logp)) over its
declared eligible support. Distinguish this from a likelihood-product ratio.
Token-local derivative routing can use a detached sequence value multiplied by
exp(current_logp-stop_gradient(current_logp)); a fully coupled sequence derivative
is another definition. With fixed +1/-1 token advantages and inactive clipping,
the coupled shared-ratio average has zero gradient whereas local routing gives
s*(grad(logp1)-grad(logp2))/2. Maintain an independent reference for that case.
Masks must prevent non-finite excluded-token operations before reduction, not
multiply infinity by zero afterward.

## Plan of Work

### Consolidated qualification over retained evidence

Use one coordinated qualification job, with one retained informative native
rollout population per model/tokenizer family. In each native backend/precision
branch, collect policy, context, system/user/tool and semantic-region masks;
credit and denominators; old/current/sampler scores and correction; KL and
clipping; finite gradients and parameter deltas; applied/skipped counters; and
checkpoint authentication together over two or three updates. Do not launch a
fresh rollout job for each diagnostic. Keep all raw tools and receipts outside Git.

Within the job, fork the same initial model, optimizer, scheduler and RNG state
and reuse identical evidence for pack-layout and optimizer-schedule comparisons.
Packing comparisons must preserve objective and update boundaries. Different
optimizer schedules intentionally change boundaries; compare each against its
declared reference, not against equal final parameters. BF16 and FP16 require
separate native precision/scaler states, but are stages of this same suite rather
than independent fresh collections. Derive recovery checks from its checkpoint1
artifacts and compare against its uninterrupted boundary2 results.

Authenticate retained source artifacts before decoding. Preserve original token
coordinates, conditioning, sampled scores, prepared credit, model lineage and
renderer provenance. Any replay rebinding must be explicit and evidenced. The
external check_fixed_evidence.py/bootstrap_fixed_evidence.py prototype under
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/corrected-trl-r117
is staged, unlaunched, and lacks qualified continuation; complete the grouped
runner and provenance validation before treating it as qualification.

Different model/tokenizer families need their own native evidence, but can be
stages of the suite. A source change needs a newly sealed manifest, not necessarily
new rollouts if replay compatibility is proved. Frozen replay qualifies math and
execution only: fresh ordinary-host collection, native active SAMPO selection,
process/mask integration, distributed execution and public release remain their
own acceptance gates. Group compatible gates into the same collection/training
job whenever possible. Do not count stub transport tests as native integration.


### Ordinary veRL physical-layout qualification


The private adapter in packages/train/src/posttrain/train/backends/verl/
policy_capabilities.py defines native_context_layout, validate_native_execution_profile
and native_execution_capabilities. A training binding may set
backend_options.resolved_context_layout to dense-population explicitly; the
default remains ragged with one record. Dense-pack and implicit multi-record
selection reject. policy_native.py checks the composed config before Ray and
passes profile validation through resolved_workers.py's create_engine seam,
checking the effective model/engine before native model allocation. Both actor
and FSDP engine compilation must be disabled. The selected profile requires
dropout-free LoRA, causal SDPA, FSDP1 with original FP32 parameters, FP32
reduction/buffers, strict determinism and the independently declared FP32 LoRA,
rowwise adapter, contiguous gradient, full GEMM and math SDPA settings. Native
explicit ordered micro_batch_sizes still owns execution; no new scheduler or
collector is added. worker.py emits BF16 mixed precision explicitly for this
layout while retaining precision-override protection.

Run tests from /home/hammad/projects/rl using the isolated native environment
and PYTHONPATH procedure recorded above, including test_update_verl_capabilities.py,
test_update_verl_runtime.py and test_verl_resolved_workers.py. The current broader
native transport/driver/runner/checkpoint cohort has99 passing tests. Real
qualification uses the published fork and digest-pinned image recorded below,
fresh local AutomationBench slack_office_closure generation, four episodes,
episode budget2, records2 and physical token budget8192 for two applied updates.
The archived source must be verified and sealed with the existing
.posttrain-source-revision marker; Git metadata cannot coexist with that marker.
Model cache remains mounted read-only; set HF_DATASETS_CACHE to a writable
qualification-owned path. Inspect the exact current container before any retry.
Acceptance requires both applied boundaries, informative finite parameter changes,
complete groups, original policy/tool/context masks, physical pack budget evidence,
checkpoint authentication/publication and exact retained-population continuation.
Fresh generation success alone does not certify those conditions.


### Milestone 1: Freeze compatibility and amend the canonical contract


First inspect source at consumer pins and the author SAMPO commit, including
native collection rows, epoch/minibatch loops, old-score refresh and reductions.
Add maintained test fixtures in packages/train/tests/fixtures/policy_updates/
containing small synthetic trajectory views, group/anchor memberships and
expected compatibility semantics. Record existing algorithm formulas, batch
translations, KL reference/derivatives and precision in a concise manifest.
Do not copy experimental correctness runners or raw research archives into Git.

Make the narrow canonical amendments described above and add optional
policy_updates settings to online profiles in profiles.py and catalog_schema.py.
Absence resolves to the exact existing schedule and execution semantics.
Explicit scheduling decouples population size from execution accumulation;
remove equality checks only for that path. Reject conflicting legacy accumulation
and explicit budget settings with actionable errors. Keep TrainingLoop.max_steps
as the legacy update limit; explicit max_applied_updates must agree if both are
supplied. Record population, attempt, applied and epoch counters separately.
SFT/DPO/distillation decoders must remain unchanged.

Acceptance is CPU round-trip tests proving old catalogs resolve identically,
new schedules decode without backend imports, and unsupported combinations fail
during detached planning. Record the consumer-pinned and author native reuse
matrix before choosing fork edits. SAMPO multiple minibatches must be recognized
as an existing native capability.

### Milestone 2: Preserve evidence, actual context and granular credit


Implement update_records.py and update_credit.py, extend existing reward records,
and wire native projection through policy_rollouts.py and veRL agent_loop.py.
Only generic native compatibility changes go into packages/environment.
A bridge-produced semantic annotation is validated against original eligible
actions and immutable extraction revision. Scorers return span IDs, not arbitrary
replacement masks. Support multi-interval and overlapping annotations.

Reuse SAMPO/GDPO/CAPO estimators behind adapters preserving their population and
normalization. Validate supplied external step credit independently from reward
evidence. Add test_update_records.py and test_update_credit.py for rolling windows,
branch continuations, tool context, malformed annotations, missing scores,
scorer observation scope, overlapping spans and unchanged legacy projection.

Add a local integration fixture under packages/train/tests/fixtures/policy_updates/
with two episodes and reasoning steps. Its fake scorer demonstrates transport;
an independently specified external estimator provides known detached advantages.
The later live gate must replace the fake with a real injected local scorer over
those exact span IDs. This is not a claim to implement PRIME or choose a PRM recipe.

Acceptance is reconstructing the exact original action/context coordinates from
native references, unchanged existing estimator results, and explicit rejection
of quality scores masquerading as ready advantages.

Implemented transport seam: environment/verifiers_conditioning.py owns native
physical-path projection and materialize_native_conditioning; train/online_rl.py
retains conditioning_records, selected_branch_id and conditioning_completion_indices.
train/update_evidence.py exposes population_from_rollouts and returns a frozen
PopulationSnapshot plus NativeCreditRows. Private backends/policy_update_scoring.py
projects selected actions from preceding causal logits, normalizing selected
half-precision positions in FP32. Score temperature is required explicitly and
tested independently; it does not reconstruct top-p/top-k filtering. It does not
certify model attention, native byte authentication or parameter provenance;
the actual native scorer/executor must supply those guarantees.

### Milestone 3: Resolve schedules and objective mathematics


Implement update_plan.py and the initial registry in update_objectives.py. Keep
population preparation separate from selecting contribution occurrences.
Support partial episodes and named group completeness. Dependency closure adds
scoring/gradient dependencies without duplicating selected loss weight.
Deterministic order and seeds must reproduce the same plan digest.

Implement private Torch reference tests in test_update_objectives.py. A monolithic
reference computes the whole declared objective directly; do not build the oracle
by calling the production reducer. Preserve current algorithms first, then test
GSPO-token routing and CISPO detached clipping as separate definitions. Reasoning
selection is an explicitly versioned variant, not a silent change to SAMPO.
Full-distribution objectives without required statistics reject deterministically.

Add test_update_plan.py proving episode/turn/token budgets, final policies, zero
selection, repeated passes, overlapping relations and complete dependency sets.
The five-turn example yields one update regardless of pack layout. A two-minibatch
schedule yields two updates; tests must not expect equivalence to one update.

Acceptance includes the mixed-sign gradient example, unequal episode lengths,
different policy/KL masks and denominators, clipped positive/negative advantages,
and a contribution whose ratio uses unselected sibling turns. These tests establish
mathematics before native backend modification.

### Milestone 4: Native TRL integration with explicit capability gates


In Posttrain add backends/trl/policy_updates.py to translate resolved updates,
statistics and evidence. Wire it through policy_config.py and policy_optimization.py.
Resolve capabilities from the actual selected fork revision. Existing native
num_iterations/steps_per_generation and sampler behavior may implement a faithful
schedule; verify rather than assume their counters match ours.

Generic gaps belong in the maintained TRL fork, primarily
trl/trainer/grpo_trainer.py and a new private trainer/policy_updates.py helper.
Add tests/test_policy_update_execution.py and preserve
tests/test_sampo_precomputed_advantages.py. Do not make TRL import Posttrain:
its reusable protocol consumes primitive IDs, memberships, declared execution
data and callable internal loss definitions, while Posttrain owns selection.

First qualify graph-retaining/native accumulation for supported closures.
Prototype score-and-replay only where it is needed to reduce retained activation
memory. At fixed parameters, compute objective adjoints and replay vector–Jacobian
products; report the original objective separately from the gradient carrier.
Direct parameter-dependent terms require a qualified direct backward contribution.
Preserve intended context gradients; detached KV reuse is forbidden by default.

Capabilities declare context/dependency/statistic capacity, supported model outputs
and faithful stochastic replay. Oversized work must use a qualified checkpointing,
offload or sharding path or fail before update. Initial replay supports deterministic
execution/dropout zero; nonzero dropout requires per-record state equivalence,
not just a global seed. Deterministic MoE is not categorically rejected.

Acceptance is actual native model forward/backward plus optimizer transitions
matching the independent reference across two pack layouts in BF16 and FP16.
If the replay prototype fails, keep correct graph-retaining execution with an
honest capacity bound and record the failure; do not truncate dependencies.

### Milestone 5: Native veRL integration and equivalent resolved meaning


Wire the same resolved update into backends/verl/launcher.py, contracts.py and
worker.py; store its schema/digest in the isolated launch manifest. Inspect veRL
workers/engine_workers.py minibatch orchestration, workers/engine/base.py
BaseEngine.train_batch, workers/engine/fsdp/transformer_impl.py optimizer_step,
and trainer/ppo/core_algos.py registered policy losses. This checkout uses engine
workers; do not assume the author's older workers/actor/dp_actor.py exists here.

Reuse native minibatch/epoch scheduling and native objective hooks when the
fixed-evidence gate proves equivalence. Generic fork changes belong in these
veRL modules, with a reusable workers/engine/policy_updates.py helper only if
necessary. Add tests/trainer/ppo/test_policy_updates_on_cpu.py and
tests/workers/test_policy_update_execution.py in the fork. Update actor config
only for required stable native contracts; raw Hydra options cannot override
Posttrain semantics. Prevent both framework and native loops scheduling the
same passes twice.

Acceptance is test_verl_trl_settings_parity.py and new
test_update_backend_parity.py comparing one serialized logical plan, fixed
trajectories, initial parameters and optimizer states. Both native engines must
execute forward/backward and step. Comparing detached logits or native name
registration alone is insufficient. World-size multiplication, turn-row
expansion and generation counts must appear in resolved evidence.

### Milestone 6: Distributed execution, atomic updates and observation


Add update_recovery.py and adapter integration around existing checkpoint
publication/cancellation hooks. An update is a transaction: its parameter and
optimizer state transition, schedule cursor and evidence commit together.
Checkpoint the admitted population/reference digests, prepared credit, selectors,
permutation and occurrence cursor, policy versions, model/optimizer/scheduler,
RNG and FP16 scaler. An incomplete accumulation is discarded and restarted from
the previous committed boundary. Resume rejects changed algorithm, evidence,
scorer/estimator, template or source identities; it does not recollect lost
populations as if they were identical.

In FP16 keep one loss scale across the whole effective update, unscale/check/clip
after accumulation, and coordinate finite checks across ranks. Overflow changes
the scaler but applies no parameter update and does not advance current-policy
version or learning-rate schedule. Retry the same occurrence up to an explicit
bounded limit, then fail with retained evidence. An all-zero or omitted update
does not count as an applied optimizer update merely because a step hook ran.
Model/rollout synchronization occurs only after committed applied updates;
fresh collection uses its recorded version.

Distributed execution must reproduce declared global weights while accounting
for gradient averaging. Empty local contributions still participate in required
collectives. Sharding or token balancing cannot change membership or denominator.
If world-size changes on resume are not qualified, reject them explicitly.

Current implementation seams are update_distribution.py:resolve_score_ownership
and private backends/policy_update_distribution.py:gather_current_scores,
distributed_score_carrier and all_ranks_finite. Their score plan binds the global
update/term digests and exclusive context ownership. Native adapters must consume
that plan, verify score/replay coverage, and bind its digest into recovery before
lifting their single-rank capability gate. DDP empty anchors must use an admitted
native context without introducing loss weight. FSDP forward/backward collectives
may require aligned empty-rank work; qualify the actual model path rather than
assuming the CPU DDP fixture provides it. Keep global objective telemetry distinct
from local gradient carriers and do not count replicated global reports twice.

Matched dense graph execution is now available through
update_distribution.py:resolve_score_rounds and private
backends/policy_update_distributed_execution.py:compute_distributed_graph_loss.
The returned DistributedGraphEvaluation holds global evaluation, native carrier,
ownership digest and real/padded forward counts. The trainer must backward the
carrier while reporting evaluation, qualify its actual collective/model path,
and integrate coordinated finite/step/recovery transactions before admission.

Use RunContext and existing normalized telemetry in policy_telemetry.py,
update_totals.py and veRL metrics.py. Record population/epoch/minibatch,
attempt/applied/skipped counters, selected versus scored versus gradient actions,
denominators, mask coverage, score versions, replay drift and dependency cost.
Identity-rich membership and span evidence is an immutable artifact, not metric
tags. Add supported definitions to apps/observatory/src/posttrain_observatory/
metric_catalog.py, telemetry.py and configuration.py with tests beside that app.
Observatory displays resolved meaning and costs; it does not recompute training
credit or introduce another correctness database.

Acceptance requires unequal two-rank partitions, empty-rank behavior, synchronized
overflow and forced interruption before/after update commit. Native checkpoint
loading must restore the next occurrence and produce the same continuation.

### Milestone 7: Real integration, release and compatibility retirement


Add packages/train/tests/test_update_native.py for fixed-evidence native runs
and test_update_rollout_integration.py for real rollout/score ingestion.
Use existing gpu/network markers; register a distributed marker in pyproject.toml
if introduced. Qualification runs are serial on the shared 8GB card after checking
it is idle. Larger contexts, Gemma 4 and two-rank cases require a suitable target;
their gates remain incomplete until run on real native models. Tiny architecture
stubs test shapes only and cannot qualify a model family.

Run a local real scorer through existing composition injection on original
reasoning-span IDs. Retain its model revision, observation scope and result
provenance. The scorer may run sequentially with the policy on an 8GB card;
scorer lifecycle is composition-owned. No external paid endpoint is required.
Optional paid paths obey existing cost ceilings and require configured credentials.
Test absent access with clear skips, but a skipped gate never counts as release
success. Run two or three applied updates on Qwen and LFM fixtures in both dtypes,
and native Gemma-family/Gemma 4 paths where capacity permits. These runs prove
loss, clipping, masks and counters, not research recipe superiority.

Publish only necessary generic fork deltas, following docs/tooling/forks.md.
Commit/push each fork and its CARBONTEQ_FORK.md first. Create retained release
assets and record hashes; publish candidates to carbonteq/dev through repository
owned retained-asset workflows, qualify the exact candidate runtimes, then promote
the unchanged bytes to stable. Update packages/train/pyproject.toml, uv.lock,
release/forks.toml, catalog locks and affected runtime-image locks only afterward.
Update docs/tooling/trl/README.md and docs/tooling/verl/README.md in the same
logical consumer change. Name each repository, source commit and test command.

Keep the old execution route for one release behind absent explicit settings.
Remove it only after all shipped online-policy catalogs use resolved plans and
native gates pass. Legacy decoding may remain as an explicit compatibility
translator; duplicate execution loops must not become permanent products.
Do not commit raw correctness tools or untracked experimental environments.

## Concrete Steps

Revision 4 verification in /home/hammad/projects/rl (actual commands):

    uv run --locked pytest packages/train/tests/test_update_credit.py packages/train/tests/test_update_objectives.py packages/train/tests/test_update_plan.py packages/train/tests/test_update_records.py packages/train/tests/test_sampo.py packages/train/tests/test_reward_advantages.py packages/train/tests/test_reward_projection.py
    uv run --locked pyright packages/train/src/posttrain/train/update_objectives.py packages/train/src/posttrain/train/update_plan.py packages/train/src/posttrain/train/update_credit.py packages/train/src/posttrain/train/backends/policy_update_math.py
    uv run --locked lint-imports
    git diff --check

The framework environment deliberately has no Torch: 91 passed/seven skipped.
The existing isolated runtime has Torch 2.13.0+cu130 but lacks Pydantic. Supplying
the existing framework site-packages on its path permits CPU formula tests
without installing ML packages into the neutral workspace. Initial collection
without that path failed with missing Pydantic; no training was attempted.
The corrected command completes 39 tests, with no skips:

    PYTHONPATH=/home/hammad/projects/rl/packages/train/src:/home/hammad/projects/rl/packages/common/src:/home/hammad/projects/rl/packages/data/src:/home/hammad/projects/rl/packages/environment/src:/home/hammad/projects/rl/.venv/lib/python3.13/site-packages /home/hammad/projects/trl-gdpo-capo/.venv/bin/python -m pytest -c /dev/null --rootdir=/home/hammad/projects/rl --import-mode=importlib -p no:cacheprovider packages/train/tests/test_update_records.py packages/train/tests/test_update_credit.py packages/train/tests/test_update_plan.py packages/train/tests/test_update_objectives.py

Scoped Ruff passes; targeted Pyright has zero errors; all nine import contracts
pass. These commands qualify formula/transport behavior, not a native model,
optimizer, distributed runtime or published artifact.


During implementation start in /home/hammad/projects/rl, preserve dirty work and
record identities. These commands are read-only except fetch, which updates remote
references but not files:

    git status --short
    git rev-parse HEAD
    git fetch origin main
    git -C ../trl-sampo-local-credit status --short
    git -C ../trl-sampo-local-credit rev-parse HEAD
    git -C ../verl-posttrain-parity status --short
    git -C ../verl-posttrain-parity rev-parse HEAD

Inspect sibling instructions and consumer release records. Work in isolated
codex/ branches/worktrees from deliberate source revisions; do not treat the
dirty workspace HEAD as the production baseline. Record chosen checkout paths
here before editing forks, then use those paths for the commands below.

From the framework root, after the named tests exist:

    uv run pytest packages/train/tests/test_update_records.py packages/train/tests/test_update_credit.py
    uv run pytest packages/train/tests/test_update_plan.py packages/train/tests/test_update_objectives.py
    uv run pytest packages/train/tests/test_sampo.py packages/train/tests/test_reward_advantages.py packages/train/tests/test_reward_projection.py
    uv run pytest packages/train/tests/test_update_backend_parity.py packages/train/tests/test_verl_trl_settings_parity.py
    uv run lint-imports
    git diff --check

Each command should pass without GPU/network for pure tests. Before implementation
the new test paths do not exist; do not count their absence as validation.
Record actual passed/skipped counts when milestones complete rather than inventing
a fixed count now.

In the chosen TRL fork's qualified environment:

    python -m pytest tests/test_policy_update_execution.py tests/test_sampo_precomputed_advantages.py

In the chosen veRL fork's qualified environment:

    python -m pytest tests/trainer/ppo/test_policy_updates_on_cpu.py
    python -m pytest tests/workers/test_policy_update_execution.py

Fork test environments need their pinned Torch/backend dependencies. Do not
install them into the framework's backend-neutral environment to make imports pass.

From the framework root with each isolated native runtime available:

    uv run pytest packages/train/tests/test_update_native.py -m gpu
    uv run pytest packages/train/tests/test_update_rollout_integration.py -m "gpu and network"
    uv run pytest packages/train/tests/test_update_native.py -m "gpu and distributed"

The harness must run backend=trl/verl and dtype=bf16/fp16 as explicit parametrized
cases, show capability skips with reasons and write raw receipts outside Git.
Distributed cases require two available devices; serial single-GPU receipts
do not satisfy that command's release gate. Record immutable model revisions and
the runtime's actual interpreter command in this plan when provisioned.

Before shipping, from the framework root:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

The all-package suite must pass or retain separately diagnosed pre-existing
failures. Do not repeatedly broaden tests after passing unless new changes
justify it.

## Validation and Acceptance


Completion means a user can select explicit schedule/execution controls and
qualified span/credit variants through Posttrain catalogs, observe identical
resolved meaning on TRL and veRL, and resume at complete optimizer boundaries.
The two-episode/five-turn receipt must show:

    selected episodes: 2
    selected turns: 5
    execution packs: 3
    optimizer attempts: 1
    applied updates: 1
    objective/credit/reduction digests: unchanged after repacking

Unequal lengths, partial episodes, overlapping spans and empty selections must
match independently calculated weights. Mixed-sign local credit must distinguish
coupled from routed gradients. Non-unit ratios must exercise both clipping
branches; tests at ratio one alone are insufficient. Excluded user/tool/padding
tokens must not contribute direct policy loss, while their intended context
remains available and differentiable where required. Separate KL support must
produce the declared regularization contribution.

Numerical qualification compares objective statistics, unscaled gradients,
parameter changes and optimizer-state changes under identical inputs and state.
Verify exact initial trainable tensor values, not only random seeds or initial
log probabilities: zero-initialized LoRA B can conceal differing A values.
Use absolute and relative error plus norm/direction evidence; relative error
alone is unstable near zero. Establish per-operation/dtype tolerances from
same-backend references and repeated pack layouts before assessing cross-backend
results. Do not relax tolerances after observing a failure without documenting
a numerical cause. BF16/FP16 targets may use FP32 sensitive arithmetic; FP64 is
neither a production requirement nor a GPU recipe.

A live integration must ingest native sampled actions and real scorer evidence,
execute actual model backward/optimizer steps, and publish resolved counters.
Overflow and cancellation receipts must prove no partial update and correct
cursor advancement. Capability rejection must identify the unavailable statistic,
context capacity or objective combination before optimization. Registration,
schema tests, synthetic logits and similar reward curves cannot substitute for
native integration.

## Idempotence and Recovery


Plan resolution is deterministic from immutable evidence, definition revisions,
seed and schedule. Cache keys include actual context, statistic requirements,
selector snapshot and parameter version. Never reuse current-score caches after
an applied update. Retry transport/scoring without changing admitted evidence;
a changed scorer result creates a new evidence identity.

Checkpoint publication uses the existing atomic artifact boundary. If accumulation
fails, discard partial gradients and reload the last complete checkpoint. If
publication is interrupted, resolve the committed checkpoint identity before
re-running an update; do not infer success from a partially written directory.
Keep failed receipts. Retry FP16 overflow only under the declared bounded policy.

Preserve unrelated worktree edits. Select files explicitly in every commit and
inspect staged diffs. Raw logits, GPU scripts, downloaded model data and native
receipts belong under /home/hammad/experiments/posttrain-correctness or ignored
.posttrain/state, not maintained source. Reverting a new capability means choosing
the retained compatible selection/runtime; never force-load an incompatible
checkpoint or silently substitute another algorithm.

## Artifacts and Notes


Primary source anchors used for design are the author SAMPO veRL actor and
WebShop script at a25a2a229c85431b421ac785fa5f375a99b2072a, GSPO v2 section 4.3
(arxiv 2507.18071), and the selective-objective review already retained in
docs/research/selective-policy-objectives-research.md. Required behavior is
embedded above; external links are evidence rather than implementation instructions.
Paper/script disagreement, incomplete author entrypoints and transfer limits
must remain visible in recipe profiles.

A milestone receipt should include framework/fork/runtime/model identities,
objective and schedule digests, evidence coverage, tested precision/capability,
expected versus observed applied updates, gradient/update comparison, peak
memory, additional dependency work, skips and failures. Publish concise results
through existing artifacts and consumer tooling pages; raw data stays outside Git.

Revision note, 2026-10-02 (revision 2): reconciled the general-engine proposal and accepted
independent critique; added actual-context provenance, named populations,
ratio value versus derivative support, granular external credit, explicit empty
reductions, bounded optional replay, native reuse gates, distributed/precision
transactions and a self-contained release path. Corrected the native SAMPO/veRL
minibatching distinction. That revision was planning only.

Revision note, 2026-10-02 (revision 25): resolve the bounded LFM BF16
turn-row discrepancy by capturing original scores and adapter tensors, then
repeating both native paths with exact initial adapter values. Record twelve
successful applied updates, a separately retained diagnostic callback failure,
twenty verified artifacts per successful campaign, exact matched results and
five passing native tests. Require exact initial tensors for future parity
checks. No production math or recipe change was necessary; remaining production,
distributed, family, recovery and release gates remain open.

Revision note, 2026-10-02 (revision 26): verify native checkpoint-1 turn-row
continuation on both backends in BF16/FP16, retaining twenty physical updates,
139 verified artifacts and strict objective/credit decoding. Record 42 passing
native tests and the fixture-versus-production boundary. Reuse native recovery
without changing forks or pins; public, distributed and broader release gates
remain open.

Revision note, 2026-10-02 (revision 27): add exclusive context score ownership
and shared distributed score/adjoint/finite seams; qualify unequal and empty ranks
with actual CPU Gloo/DDP and independent BF16/FP16 score references. Retain 136
source files and two rank receipts outside Git, and record 38 passing focused
tests. Leave native distributed admission and all remaining production/release
gates open until their actual integration and transaction checks pass.

Revision note, 2026-10-02 (revision 28): add matched context-forward rounds and
graph-retaining distributed execution, test actual CPU DDP backward and two
optimizer transitions with unequal/empty ownership, and preserve measured
half-precision rounding differences. Retain exact sources/runner/log/receipts
outside Git, record 33 passing focused tests, and consolidate the six current
completion gates after the user's status request. Native trainer, live coverage,
observation and release adoption remain open.

Revision note, 2026-10-02 (revision 3): implementation authorized by the active
goal; canonical amendment and ADR 0020 landed locally before pure records,
credit and scheduling code. Recorded 80 passing focused tests and source-level
native reuse audit. Native qualification and release gates remain open.

Revision note, 2026-10-02 (revision 4): added explicit objective and reduction
contracts, private tensor evaluation and independent derivatives; retained zero
reduction domains, validated current-score versions and delegated built-in credit
to existing estimators. Recorded actual CPU test commands and CISPO definition
differences. Catalog/native/recovery/distributed/release gates remain open.

Revision note, 2026-10-02 (revision 5): added guarded explicit catalog selections,
original native context/action maps, complete-group assembly and causal score
projection. Retained provider-input equivalence receipt lives at
/home/hammad/experiments/posttrain-correctness/2026-10-01/results/native-collection/engine-native-context-receipt.json;
the external runner is working/engine_native_context_receipt.py under that campaign
root. Run from /home/hammad/projects/rl with
`PYTHONPATH=packages/environment/src uv run --locked python /home/hammad/experiments/posttrain-correctness/2026-10-01/working/engine_native_context_receipt.py`.
The source SHA-256 is in the receipt; rerun only against identical retained source
bytes and preserve failed receipts. Focused commands are
`uv run --locked pytest packages/environment/tests packages/train/tests/test_update_records.py packages/train/tests/test_update_evidence.py packages/train/tests/test_policy_update_settings.py packages/train/tests/test_update_credit.py packages/train/tests/test_update_plan.py packages/train/tests/test_reward_recovery.py -q`
(115 passed) and, in the existing isolated ML runtime,
`PYTHONPATH=/home/hammad/projects/rl/packages/train/src:/home/hammad/projects/rl/packages/common/src:/home/hammad/projects/rl/packages/data/src:/home/hammad/projects/rl/packages/environment/src:/home/hammad/projects/rl/.venv/lib/python3.13/site-packages /home/hammad/projects/trl-gdpo-capo/.venv/bin/python -m pytest -c /dev/null --rootdir=/home/hammad/projects/rl --import-mode=importlib -p no:cacheprovider packages/train/tests/test_update_scoring.py packages/train/tests/test_update_objectives.py -q`
(30 passed). Native optimizer/GPU/recovery/distributed/release gates remain open.

Revision note, 2026-10-02 (revision 6): add the private full-context score and
graph-retaining loss seams and independent two-transition CPU operator checks.
Repeat the revision-5 isolated command with test_update_execution.py added
alongside test_update_scoring.py and test_update_objectives.py (39 passed).
No native scheduling replacement, fork edit/publication or consumer pin change
was made. Next wire this seam through the actual TRL trainer update lifecycle,
then veRL, enforcing population-frozen old/reference scores and applied-update
transactions. Both native gates remain open.

Revision note, 2026-10-02 (revision 7): added private
backends/trl/policy_updates.py and optional test_update_native.py. The adapter
reuses native TRL training_step and optimizer hooks, bypassing only generation
repetition for pre-resolved payloads. Four native CPU tests pass, including cursor
preservation on overflow; actual retry/recovery still deliberately fails closed.
Tests run from /home/hammad/projects/rl using the revision-6 isolated environment
with the exact detached source checkout
/home/hammad/experiments/posttrain-correctness/2026-10-01/dependencies/trl-engine-d97a2cf
prepended to PYTHONPATH and test_update_native.py selected.

Real runner and receipts stay outside Git under
/home/hammad/experiments/posttrain-correctness/2026-10-01:
working/engine_native_trl_run.py and results/native-collection/engine-trl-qwen-*.json/.log.
Preserve engine-trl-qwen-bf16-full-population.log (terminal default-allocator OOM)
and the first two-minibatch log (runner argument-name failure before model load).
Successful full-population receipts have the -expandable suffix. BF16 two-minibatch
diagnostic repeat also has that suffix. FP16 two-minibatch receipt has no suffix.
Later receipts include runner and production-source hashes; earlier successful
arms have weaker source receipts and must not certify the release manifest.

Run serially only after nvidia-smi reports no compute process. Use
/tmp/posttrain-native-verl-update-runtime/bin/python with PYTHONPATH containing
/tmp/posttrain-native-verifiers, /home/hammad/projects/renderers-lfm-mask,
/tmp/trl-math-renderers-deps, the exact detached TRL source, the train/common/data/
environment src directories and the root .venv/lib/python3.13/site-packages.
Invoke working/engine_native_trl_run.py with --dtype bfloat16 or float16,
--minibatch1 or2 (with a space between flag/value), and a fresh --output JSON path;
retain stdout/stderr in a fresh corresponding log. Set
PYTORCH_ALLOC_CONF=expandable_segments:True for full-population graph retention.
The immutable model is cached locally; no paid endpoint or fresh generation is used.
No fork source edits or dependency pin changes were required for this adapter.
Next integrate native veRL and public collection/transaction wiring, then complete
independent real-model gradient, replay-if-needed, overflow retry, resume, process
scorer, multi-family and distributed gates before removing public launch guards.

Revision note, 2026-10-02 (revision 10): record deterministic native BF16 resume
equality on TRL and veRL, keep default-runtime discrepancy margins open, and
require resolved-population evidence in the neutral checkpoint seal. Update the
consumer pages with the exact qualification boundary. This improves recovery
evidence completeness without adopting a training recipe or changing fork pins.

Revision note, 2026-10-02 (revision 11): implement native bounded TRL scaler
retries, preserve stochastic replay after old-score freezing, bind retry limits
and capabilities into recovery identity, and prevent failed attempts from
becoming checkpoint boundaries. Record CPU stochastic/exhaustion/nonfinite
regressions, real FP16 skip/control/resume equality and direct remote GPU/runtime
preflight. Public composition and broader qualification remain open.

Revision note, 2026-10-02 (revisions99–100): close retained BF16/FP16 physical
packing, budget and strictv3 recovery on three layouts, then publish independent
native scoped arithmetic settings atc45392d22df0c1ab2258e9c0675d3a03093094af.
Their46 focused CPU checks pass. Stage1589 hashes for R100 and qualify without
external backend overrides; BF16 container is live at this stopping point.
The immutable runtime image remains
sha256:f0d32115d6e3bdca1dda76aad0ff8b9605e9869d4413ea65f76f4bc942cc85eb;
LFM2.5-1.2B-Thinking snapshot95053d21d8e0b7ca99421a2127ae39c64f685ff3
and frozen native population183b03b5dffd1e683de7c68d25de12e41eeee030a82370f5f83dfa950b4a50d3
are unchanged. Raw builders/scripts/receipts remain outside Git under
/home/hammad/experiments/posttrain-correctness/2026-10-01/working/full-verl-r67.
Poll the named container; do not restart on an observation timeout. Once terminal,
run compare_native_arithmetic_all_r100.py bf16 inside the same image on CPU,
verify GPU idle, and launch the FP16 counterpart on the same immutable tree.
Ordinary multi-record admission must follow declared native settings qualification,
not bypass the current records1 guard on the basis of actor probes alone.

Revision note, 2026-10-02 (revision101): R100 scoped native BF16 exits0 and
independent comparison confirms exact all24 gradients/adapters at both updates
across all three layouts, with exact boundary1 continuation and one clipped
action at update2. Its raw result is retained as r100-bf16-result.json in the
external root above. Ledger follow-updb485f4f records evidence; qualification
still uses sourcec45392d2. After GPU idle checks launch the FP16 counterpart
posttrain-r100-native-arithmetic-resume-fp16 (container9639e6dec2dc3a93812d4ca48239d3c3ba39cdbb530152baef41a28994fdd545).
Observe terminal state and run compare_native_arithmetic_all_r100.py fp16 on
CPU. Framework16 runtime tests, focused Ruff/Pyright and9 boundary contracts
pass. Worker compatibility adds the published source after verified ancestry;
the immutable R100 source is not modified. Ordinary-driver admission remains
the next implementation gate after FP16 qualification, with full remaining
algorithm/mask/family/distributed/release scope preserved.

Revision note, 2026-10-02 (revisions102–104): close R100 maintained scoped FP16
packing/recovery, then implement explicit ordinary dense-population admission in
policy_capabilities.py/policy_native.py/resolved_workers.py and selected BF16
mixed-policy emission in worker.py. All99 focused native tests, Ruff/Pyright,
9 import-boundary contracts and diff checks pass. R103 stages361 framework files
plus publishedc45392d2. The first two ordinary attempts stop before training
on missing snapshot marker and read-only datasets cache. Reuse native immutable
marker, verify1591 hashes, preserve those failures, and set writable
HF_DATASETS_CACHE=/qualification/runtime-cache/r103c/datasets. The corrected
posttrain-r103c-ordinary-dense-bf16 container
0cefbb99b8b5ecb74a91abb7675b1bc7500a1f3a2fabd258aac77e56d331f130
is live at this stopping point; native actor and vLLM GPU processes are present.
Read /var/lib/posttrain/qualifications/full-verl-20261002-r103/bf16-r103c-ordinary/model/verl-native.log
and the actor observation journal, then authenticate applied boundaries/masks/
packing/parameters/checkpoints before continuation or any FP16 launch. Never
restart on polling timeout. Raw script builders and receipts remain external
under full-verl-r67; no consumer pin changes or raw correctness tools committed.

Revision104 stopping observation: posttrain-r103c-ordinary-dense-bf16 is confirmed
running. Native actor initialization journal binds runtime
resolved-verl-job/2499c81e2d4404d3e82014b504e1ac46d5cf0f9d73e16a998e574d3a06e31830
and renderer/4e5417d125346991e56c39d3ec5d81d8e93567cfce401d1a6d97f4adf6abc20a.
Native actor and vLLM GPU processes are present; agent workers and file logger
are initialized. No complete rollout/checkpoint or applied update has yet been
verified. Follow this exact handle and native log; do not restart based on an
empty journal or missing not-yet-produced trace/checkpoint.

Revision note, 2026-10-02 (revisions105–106): the R104 live observation above is
historical. R103c exits0 and independent verification authenticates15 components
per checkpoint, frozen4427 old scores, all24 changing adapter tensors and2 clipped
actions at update2. External check_ordinary_masks_r105.py authenticates the native
episode digest and exactly matches all4427 action identities to assistant masks,
retained token IDs/logprobs/parents and8 conditioning context references. R105
ordinary checkpoint1 continuation exits0; compare_ordinary_resume_r105.py verifies
boundary2 seals and exact248 saved tensors plus resolved population payload
(maxdiff0). After direct GPU idle/RAM checks launch ordinary FP16 R106 container
6b3e3ac877d95afb64ed9b81cc80248866b1591ec1c1ff4dbf36eb68f9ebf73f;
follow this handle, native log and journal under
/var/lib/posttrain/qualifications/full-verl-20261002-r103/fp16-r106-ordinary.
Same1591 sealed files/sourcec45392d2; no recipe/math changes, consumer pins or raw
correctness scripts committed. Finish actual FP16 applied/overflow/checkpoint
qualification and resume, then continue the full remaining plan gates.

Revision note, 2026-10-02 (revisions107–108): ordinary FP16 R106 exits0 and
independent checkpoint verifier authenticates15 components per boundary, frozen
4420 old scores, all24 changing adapter tensors, explicit two-record physical
packs and applied=attempts2. Separate saved-scaler audit confirms scale128,
growth_tracker1/2 and no skipped steps. Fresh FP16 uses a different generated
population from BF16; retained same-evidence controls remain the arithmetic
comparison authority. After direct GPU/container idle checks launch ordinary
checkpoint1 resume posttrain-r108-ordinary-dense-fp16-resume, container
20ef980d7cc1d76bb986b135e32aeb33c5d1a9937a6f54eb5bdbd6fc5cae2f85.
Observe /var/lib/posttrain/qualifications/full-verl-20261002-r103/fp16-r108-resume;
after terminal run /qualification/compare_ordinary_resume_r108.py on CPU in the
same pinned image. No source changes to this1591-file qualification tree.

Audit reveals resolved ordinary host correction=None despite default SAMPO's
selected token_truncate cap2. External measure_sampler_gap_r107.py finds actual
BF16 token weights.679–1.365, mean1.000558. New maintained
packages/train/src/posttrain/train/update_sampler_correction.py implements
complete-support token/episode truncate/mask detached weights;14 reference tests,
scoped Ruff/Pyright,9 import contracts pass. It is not wired/native-qualified and
not included in R108 source. Next collection integration must reuse native
ActiveSamplingReplayBuffer dispatch/observe/eviction, avoid unconditional initial
batch submission, preserve fresh uniqueness including discarded/refill groups,
and retain sampler/old/correction provenance and weights across checkpoint
recovery before admitting default SAMPO. Raw controls stay outside Git; consumer
pins and release guards remain unchanged. Full remaining scope is preserved.

Revision note, 2026-10-02 (revisions109–110): ordinary FP16 R108 exits0 and
/qualification/compare_ordinary_resume_r108.py independently verifies complete
boundary2 equality:248 saved tensors, optimizer/scheduler/RNG/scaler, native
data.pt and resolved population, maxdiff0. CPU receipt remains under the R103
qualification root. Both ordinary dtype resume gates close for this single-rank
LFM source profile, not for corrected SAMPO or the full engine.

New shared correction transport uses optional
posttrain-sampler-correction.json, component role sampler-correction, existing
identity correction digest and exact complete action support. Read through
load_sampler_correction(checkpoint, identity) only after component authentication;
veRL actor reconstruction now consumes its immutable map. Uncorrected checkpoints
retain their existing identities/file layouts.83 recovery/native regression
tests, scoped Ruff/Pyright,9 import contracts and diff checks pass. The next
fresh-collection step must authenticate sampled log probabilities from the native
evidence, freeze initial old actor scores, derive selected correction before
objective evaluation, then qualify new source with actual BF16/FP16 applied
updates and recovery. R108 does not exercise these new files. Preserve full
active-SAMPO/semantic/process/family/distributed/observation/release scope.

Revision note, 2026-10-02 (revision111): add authenticated compact sampled scores
to NativePopulationInputs, freeze old actor scores once in ResolvedVeRLPopulation,
and prepare selected correction before the first objective through a one-use
callback. Admission retains its score-free fresh-population invariant; no relaxed
lifecycle rule is introduced. Ordinary collection selects the existing recipe's
token/sequence truncate/mask settings; structured recipes retain their existing
token cap3. Checkpoint restore reads sealed weights and rejects absent correction
for this corrected ordinary path.90 related tests and scoped Ruff/Pyright,
9 import contracts/diff checks pass. Source hashes change native runtime identity.

Stage362 current framework files plus publishedc45392d2 at
/var/lib/posttrain/qualifications/full-verl-20261002-r111, verify1592 hashes and
the immutable marker. After direct GPU/container idle checks launch corrected
BF16 posttrain-r111-corrected-ordinary-bf16, container
db4988fda7d04a23b9c6f8b6c45609f0f546d125e63e226dc9e47b6cfbdac46c.
Follow /qualification/bf16-r111-corrected/model/verl-native.log and actor journal;
authenticate native sampled logps→initial old scores→selected correction and
frozen checkpoint component, then exact resume, before corrected FP16. Raw
builder/checker/receipts remain external under full-verl-r67; no consumer pins or
raw correctness tools committed. Prior R106/R108 FP16 means FP16 actor/scaler
with effective BF16 vLLM rollout, not all-FP16 execution. Active SAMPO and full
remaining goal scope stay open.

Revision note, 2026-10-02 (revisions112–114): corrected BF16 R111 exits0.
Independent CPU verification authenticates16 sealed components/publications,
4476 frozen action scores/corrections and exact Torch FP64 episode-product weights.
General verification confirms all24 adapters change, physical packs[2,1] with
partial tails and3 clipped actions at update2. Current local TRL/shared correction
normalizer changes pass59 related tests plus18 pure correction references; they
are excluded from the immutable R111 tree and not native GPU qualified.

R113 corrected checkpoint1 continuation exits0; independent comparison verifies
all248 native saved tensors plus optimizer/scheduler/RNG/data/population/correction
exactly at boundary2 (maxdiff0). Receipt is
/var/lib/posttrain/qualifications/full-verl-20261002-r111/r113-corrected-ordinary-resume-step2-comparison.json.
After direct GPU idle checks launch corrected FP16
posttrain-r114-corrected-ordinary-fp16, container
2477336ba59920278998d9b52851e2c33656c2909b7604865d4120adba5e6e24.
Follow this exact handle and fp16-r114-corrected/model/verl-native.log in the R111
root. Verify native applied updates, sampler correction, FP16 scaler and complete
checkpoints, then checkpoint1 continuation using the same1592-file source. Actor
is FP16; vLLM rollout remains effective BF16. Preserve all broader SAMPO, masks,
process-credit, Gemma, distributed, observation, public-entrypoint and release
gates. Raw tools and receipts remain external; no pins or source commits made.

Revision114 continuation preparation: corrected TRL362-file snapshot and clean
forkc9af78c1c2ea04ad271e95b26b93dfadf8b9fca1 are staged separately at
/var/lib/posttrain/qualifications/corrected-trl-20261002-r116. bootstrap.py verifies
archive/framework hashes then installs the fork in an isolated container.
CPU --precision bf16 --attempt r116-preflight --preflight exits0 and accepts the
ordinary host admission. check.py uses NativeVerifiersEnvironmentFactory,
group4, episode minibatch budget2, execution records2,2 maximum applied updates;
GPU launch must use a new attempt after R114/R115 terminal and direct GPU check.
The candidate metadata1.8.0 differs from installed image1.12.0.post11 and selected
dependency1.12.0.post13; immutable runtime/release adoption remains unresolved.
The preflight directory already exists and cannot be reused as a fresh attempt.

Revision note, 2026-10-02 (revision115): local veRL dataset materialization now
rejects missing/duplicate/undersized task inventories for explicit policy_updates,
preserving legacy cycling. Six focused tests, Ruff/Pyright,9 import contracts and
diff checking pass. This source change is outside both staged R111 and R116.
R114 corrected FP16 exits0; independent CPU verifiers authenticate16 components,
4750 frozen corrections exactly, native scaler scale128/growth_tracker1/2,
all24 changing adapters and physical packs[2,1]. Clipping0 both steps; fresh
population differs from BF16, so this is no controlled dtype comparison.
After direct GPU idle checks launch posttrain-r115-corrected-ordinary-fp16-resume,
container865b9ae6965789aa8e17ba0bbe678cc896a0cf6855602ca60b91d05fa17efcb0.
Follow fp16-r115-resume/model/verl-native.log under the R111 root. After terminal
run compare_corrected_resume_r115.py on CPU in the same pinned image; it compares
native model/optimizer/scheduler/RNG/scaler/data/population/correction boundary2.
Then proceed to separately staged corrected TRL real BF16/FP16 qualification and
the full remaining native SAMPO/masks/process/Gemma/distributed/release gates.

Revision note, 2026-10-02 (revision116): R116 staged TRL is rejected before GPU
launch because general siblingc9af78c1 is not the plan's dedicated HF score
candidate. Keep it for audit only. Correct source is
/home/hammad/projects/trl-engine-generation, base
d97a2cf619f94c7076a720116d75f85dfd62382b, dirty GRPO generation-score delta plus
ledger/new fork test. R117 retains all tracked fork files plus test with per-file
hashes and362 current framework files (including the local inventory guard) at
/var/lib/posttrain/qualifications/corrected-trl-20261002-r117. CPU preflight
--precision bf16 --attempt r117-preflight --preflight exits0 and checks native
return_generation_logprobs signature as well as ordinary job admission. Installed
metadata is1.12.0.post13. R117-preflight directory exists; use a new attempt for
real training after R115 terminal/comparison and direct GPU idle check. Do not
mutate R111 or imply dirty candidate publication. No pins/source commits made.

Revision note, 2026-10-02 (revision117): R115 corrected FP16 exits0 and CPU
compare_corrected_resume_r115.py verifies all248 saved tensors, optimizer,
scheduler/RNG/scaler/data/population/correction exactly at boundary2. Receipt
r115-corrected-ordinary-resume-step2-comparison.json remains in the R111 root.
After direct GPU idle checks launch posttrain-r117-corrected-ordinary-trl-bf16,
container10d782d22eadd5a1f187e0109b0482b17cfb14d0283d00548baa148e52988642.
Follow docker logs and /qualification/bf16-r117-corrected under
/var/lib/posttrain/qualifications/corrected-trl-20261002-r117. External verify.py
compares actual sampled/frozen-old correction independently in FP64, authenticates
both native checkpoint seals and applied counters, verifies unchanged correction
and changed adapters. It must run only after authoritative terminal state, then
full checkpoint1 continuation and corrected FP16. R116 is rejected wrong lineage;
do not reuse its source. No public guards, consumer pins or commits changed.

Revision note, 2026-10-02 (revision118): corrected TRL BF16 R117 exits0 but all
four native episodes succeed with reward1 and no errors. Actual calls finish
tool_calls then stop. Prepared advantages/gradient norms0 and all24 adapters
unchanged; independent verifier rejects informative update. Retain this control,
do not describe it as training qualification. Raw traces/observations were fetched
outside Git for manual analysis. The verifier retains its receipt before rejecting
informative_update_gate; mathematical/correction checks remain unchanged.
After direct GPU idle check launch separate reply768 profile
posttrain-r118-corrected-ordinary-trl-bf16-bounded, container
9700e6a218f029137340efe4cf34f46dfb15d9a90efdc4390244dc0060174183,
under the same R117 root as bf16-r118-bounded. Observe this exact live handle;
authenticate checkpoint/correction values and inspect actual native rollout
completion/truncation before resume or FP16 launch. No benchmark rewards altered,
no public guards/pins/commits changed. Full active goal scope remains intact.

Revision note, 2026-10-02 (revision119): R118 exits0 with four length768 calls,
all rewards0,3072 exact independent corrections, unchanged adapters. R117 has
four successful episodes,3468 exact corrections, unchanged adapters. Both verify
15 sealed components/counters/frozen correction and reject informative_update_gate.
Native traces were inspected manually; zero gradients follow group normalization.
After direct GPU idle check launch native simple.buffer_webinar_dual_post,
posttrain-r119-corrected-ordinary-trl-bf16-multi-assertion, container
75d490501f833e5248b1e4feabe1ea343a3d7430ceb6bcbb74d89c7ef7d7b655,
run bf16-r119-multi-assertion under R117 root. Same sealed source; separate
check_multi_assertion.py/bootstrap_multi_assertion.py retain original checker
for recovery. Native task has two independently scored posts, group4, episode
budget2, records2, prompt3072/reply1024 within4096 context. No rewards fabricated.
Observe exact handle, authenticate terminal state, inspect actual traces and run
verify.py on CPU. Only then qualify full continuation and FP16; preserve remaining
SAMPO/schedule/masks/process/Gemma/distributed/observation/public release scope.
