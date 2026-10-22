# Evaluate LFM2.5 8B and 24B MoEs on heldout62

This living execution plan follows `docs/templates/PLAN.md`. Maintain Progress,
Surprises & Discoveries, Decision Log, and Outcomes & Retrospective. The frozen
product baseline is unchanged: this is an additive model-screening experiment.

## Purpose / Big Picture

Compare LiquidAI/LFM2.5-8B-A1B and LiquidAI/LFM2.5-24B-A2B against the existing
LFM2.5-2.6B and Qwen3.5 baselines using the report's exact 62-task benchmark.
Queue the two GPU jobs sequentially after the current Occupancy Engine training
run releases the workstation. Preserve the reward-redesign checkout and the
running training job.

## Progress

- [x] Final publication complete. Page sequence 3 preserves the report and
  numerical audit attachment; six-bundle coverage and reverse-pair bootstrap
  symmetry verified. Changed Python passes Ruff/Pyright, two targeted smoke
  tests pass, import contracts kept and diff whitespace check passes. Monitor
  paused after reporting; no further GPU runs scheduled for this experiment.

- [x] (2026-10-16 03:40Z) 24B r2 completed successfully and reconciles with
  native evidence: all 310 unique episodes, five per each of 62 tasks, ordered
  manifest verified; 305 scored, zero HarnessError timeouts, five LinkedIn
  finalization TaskErrors on sales.milestone_celebration. Resolved native config
  confirms max_concurrent=8. Artifact v1 digest
  `33e64d3e5d444f80dfcdfb3b35fb9b7a7669dafc1867e99c75ff82ee193603f9`.
- [x] Final report and complete numerical audit saved in
  `docs/research/lfm-moe-heldout62-evaluation-2026-10-16.md` and
  `docs/research/lfm-moe-heldout62-results-2026-10-16.json`. Six native bundles
  checked, including original 24B r1; no pooling or silent score replacement.
- [x] Published a child Page under the existing OPD research Page:
  https://chatgpt.com/space/page_3ef91b283c188191bbbcd97e648145bd
  and attached the numerical JSON with confirmed file access. Readback verifies
  headings, tables and attachment. Repository analysis, source artifacts and
  prior failed attempts remain intact. Final interpretation uses the common
  61 fully scored tasks; complete 62-task headline remains unqualified because
  of the frozen environment bug, not hidden by imputation.

- [x] (2026-10-16 03:25Z) Full 24B r2 continues normally: seven/eight active
  requests, zero waiting, recent HTTP 200 completions, GPU 100%, 84,267 MiB.
  Native final bundle remains pending; no additional submission.

- [x] (2026-10-16 03:10Z) Full 24B r2 still running normally: eight active
  requests, zero waiting, HTTP 200 completions, GPU 100%, 84,267 MiB. Final
  evidence pending; no duplicate or additional workload submitted.

- [x] (2026-10-16 02:55Z) Full 24B r2 remains healthy: HTTP 200 completions,
  GPU 100%, 83,729 MiB, eight active requests and zero waiting at successive
  samples. Await native completion/reconciliation; no further GPU submissions.

- [x] (2026-10-16 02:40Z) 24B full r2 is running normally: eight active
  requests, zero queued at latest sample, HTTP 200 completions, GPU 100%,
  83,729 MiB. The admission repair has removed the previously observed large
  server queue; episode timeout/scoring outcomes still require final evidence.

- [x] (2026-10-16 02:25Z) 24B full eval r1 completed, exit 0 and retained
  artifact reconciled. Native bundle in `/tmp/lfm-moe-heldout62-results/24b/`:
  all 310 identities/five attempts, exact ordered manifest verified. Only
  298 scored: ten HarnessError timeouts at 1800s and two TaskError LinkedIn
  finalization failures. Preserve r1 and its full-population missingness.
  Evaluation artifact digest `9641f9be6f617953b3af1d75e3b575071fae891af75dc3cd113f1c0f96d2a1dc`.
- [x] Diagnosed a capacity mismatch: environment admits 64 concurrent episodes
  but server runs eight sequences. Healthy logs repeatedly show 50+ queued
  requests. Timed-out episodes spent 1552–1790s in model calls; seven of ten
  produced only 4293–7135 completion tokens. This supports a queueing hypothesis,
  not proof of model-only inability. No OOM/provider failure observed.
- [x] Added opt-in Lab job definition `eval/lfm-moe-managed-general-c8@1` with
  `EvaluationBudget(max_concurrent=8)` and 24B job `evaluate_c8`. Only episode
  admission changes from 64 to eight; server stays eight sequences and all
  frozen source/task/model/sampling/context/token/turn/1800s budgets stay exact.
  Product baseline unchanged. Keep original `evaluate` definition and r1 for
  audit; report operating policies separately and do not mix episodes across runs.
- [x] Submitted full 24B retry `lfm25-24b-a2b-heldout62-20261016-r2` using
  `--job evaluate_c8`, normal LAN packaging path; log
  `/tmp/lfm25-24b-heldout62-eval-r2-submit.log`. No new GPU smoke is needed:
  model, renderer, reasoning/tool parsers and serving settings are unchanged.
  Final report pending retry; frozen schema error may still leave missing scores.
  Ruff/Pyright pass, strict five-seat/two-job composition validates, all nine
  import boundaries kept and `git diff --check` passes. The retry uses the same
  qualified inference snapshot semantics; no upstream backend or environment
  source was changed to obtain this repair.
  Accepted provider `pt-d2e129d7ec91785df1cff97e`, image
  `registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337`.

- [x] (2026-10-16 02:10Z) Full 24B eval r1 remains running with HTTP 200
  completions, GPU 100%, 84,939 MiB. Eight active requests and 17 waiting;
  native final evidence remains pending. No duplicate or competing submission.

- [x] (2026-10-16 01:55Z) Full 24B eval r1 remains healthy on its owned
  provider: GPU 100%, 84,571 MiB, eight active requests and 56 waiting; recent
  completions HTTP 200. No additional submission. Final episodes/coverage
  reconciliation remains pending.

- [x] (2026-10-16 01:40Z) Full 24B eval r1 continues serving normally on its
  owned provider: GPU 100%, 83,985 MiB, eight active requests and 53 waiting,
  recent HTTP 200 completions. Completion/reconciliation pending; no duplicate
  submission or additional GPU workload started.

- [x] (2026-10-16 01:25Z) Full 24B eval r1 remains running on its owned
  provider, GPU 100%, 83,739 MiB allocated. Server responses are HTTP 200;
  eight active requests and 55 waiting, with recent diagnostic generation
  throughput 352 tokens/s. No new job submitted; final native evidence pending.

- [x] Deeper 8B error audit corrects the earlier shorthand: all three missing
  score episodes contain `TaskError` during task finalization. WorldState
  validation rejects extra LinkedIn PostContent article fields
  (`article_url`, `article_title`, `article_description`). Their stop condition
  is also max_turns, but missing scores are an environment finalization failure,
  not a demonstrated semantic failure. Preserve frozen environment/pins and
  report this limitation; no silent rescore or zero fill. Model transport retry
  counters are not authoritative in the native schema; do not equate zero
  observed provider-error nodes with proof of zero retries.

- [x] (2026-10-16 01:10Z) 24B smoke r2 succeeded and native artifact review
  qualified tool transport/continuation. First generation: correct
  lookup_code/readiness, 35 tokens, 84 reasoning characters, empty content.
  Final: READY-2047 in content, 52 tokens, 159 reasoning characters, no extra
  tool calls or leaked think markup. Evidence `/tmp/lfm-moe-24b-smoke-r2-evidence/`.
  Preserve the earlier incorrect-argument r1 failure alongside this result.
- [x] Submitted full 24B eval r1 normal fresh package after checking
  unused ID, no active builds, empty GPU and no live workload. Log
  `/tmp/lfm25-24b-heldout62-eval-r1-submit.log`. Inspect process and retained
  state before any retry; full 8B r1 remains completed and must not be rerun.
  Accepted provider `pt-cb77402130f07fd1d8ab6553`, image
  `registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:029a99643097b372c888fde5c5fe96ef181e665e78b854d44530e56814b8ea20`.
  Wait for completion and native reconciliation before reporting scores.

- [x] (2026-10-16 00:55Z) 24B native smoke r1 failed the strict argument gate.
  It emitted a valid parsed lookup_code call but chose `key_readiness` instead
  of `readiness`; the retained reasoning also refers to that wrong key. This
  is a model argument-selection failure, not demonstrated parser corruption.
  Evidence retained in `/tmp/lfm-moe-24b-smoke-r1-evidence/`. No full 24B eval
  has been submitted.
- [x] Diagnostic-only repair makes the smoke request specify the exact JSON
  `{ "key": "readiness" }`, keeping schema, gate, result and continuation
  checks intact. Preserve r1 as failed. Retry r2 qualifies transport and
  tool-result consumption under an explicit request; it does not erase the
  original argument-selection error or establish benchmark capability.
  Neither benchmark prompts, native template, weights nor budgets changed.
- [x] Submitted 24B smoke r2 under new suffix; submission log
  `/tmp/lfm25-24b-native-smoke-r2-submit.log`.
  Provider `pt-82d70f86bc2e5b08957b9310`; image
  `registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:bb3050a3adeaa2cbf9a6854afa39560efd68630393db36574547b38e308f6203`.
- [x] Reducer now includes per-domain complete-task sensitivity. Ruff/Pyright
  pass; manual checks confirm absent evidence stays missing, duplicate findings
  count once, and manifests explicitly declaring no guards contribute zero
  applicable harms. Goal-rate absence includes episodes with no decided
  obligations and must not automatically be described as an assessment failure.

- [x] Added reproducible native reducer
  `scripts/qualification/analyze_lfm_moe_heldout62.py`: enforces exact 62-task,
  five-repetition identity coverage, rejects duplicate episodes/conflicting
  assessment IDs, deduplicates lifecycle findings, preserves absent metrics,
  and reports complete-task paired-bootstrap sensitivity separately from
  absolute intervals and bounded missing-score identification ranges.
  Reviewed no-guard manifests are zero applicable harms, distinct from absent
  guard evidence. Baseline audit found success coverage 305/310 (2.6B),
  306/310 (Qwen4B), 310/310 (Qwen9B). Historical zero-filled summaries must not
  be presented as strict complete-score results. Initial 8B success comparison
  on 61 fully scored tasks is lower by 14.4 points vs 2.6B (paired 95% interval
  -20.7 to -8.5 points); this is conditional sensitivity, not the 62-task headline.
  Reducer output currently `/tmp/lfm-moe-heldout62-results/audit.json`.
  8B retained manifest order exactly matches fixture and its newline-joined
  names hash `ed81cf1fe88f68bb8420bb46c5fdf3afde24660c7043812da680bb1c7bc9196f`.
  Ruff, Pyright and diff whitespace validation passed for the reducer.

- [x] 24B native smoke r1 accepted as provider `pt-d4086ba91523f150e0bcedce`,
  image `registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:148332104570b68d302e946b61a5b38f992168fddc676a7d02a89f029bdc1d2d`.
- [x] 8B native bundle downloaded to `/tmp/lfm-moe-heldout62-results/8b/`.
  310 episodes, 62 task identities, exactly five repetitions each. 307 are
  completed/scored; three `sales.milestone_celebration` repetitions (1,2,4)
  hit max_turns with empty metrics/rewards. These are retained missing scores,
  not coerced failures or dropped observations. Strict headline is unavailable
  until scoring coverage is resolved; preserve this limitation in comparisons.

- [x] (2026-10-16 00:40Z) Full 8B eval r1 completed successfully (provider
  exit 0, Trackio succeeded, evaluation artifact retained; reconciled at
  00:29:47Z). Native 310-episode verification and scoring comparison pending.
  Evaluation artifact digest `6884e023c4c8f5ef53d9d57a04b4d3a61d707bd0e618cf0ffab0604d361bf85b`.
- [ ] Started 24B dedicated native smoke r1 packaging/submission after confirming
  empty GPU, no containers, no active pack and unused ID. Log
  `/tmp/lfm25-24b-native-smoke-r1-submit.log`; inspect before any retry.

- [x] (2026-10-16 00:25Z) Full 8B eval r1 remains running on its owned
  provider. GPU 100%, EngineCore 83,948 MiB; native server is serving HTTP 200
  responses with eight active requests and 37 waiting. Recent generation
  throughput 683.5 tokens/s is diagnostic only, not a cross-model comparison.
  Completion and episode reconciliation remain pending; no second job launched.

- [x] (2026-10-16 00:10Z) 8B native smoke r4 succeeded and artifact review
  qualifies channel separation. First generation: valid lookup_code/readiness
  call, 240 tokens, 992 reasoning characters, empty content. Continuation:
  correct READY-2047 final content, 29 tokens, no reasoning or extra tool calls.
  No leaked think markup. Actual DelegatingParser direct/explicit stream
  preflight also passed. Evidence: `/tmp/lfm-moe-smoke-r4-evidence/`.
- [x] Full 8B evaluation r1 fresh packaging/submission restarted after checking
  unused run ID, no active pack, empty GPU processes and no running containers.
  Log `/tmp/lfm25-8b-heldout62-eval-r1-submit.log`; inspect process/state before
  any retry. This supersedes the stopped pre-qualification CPU build below.
  Accepted as provider `pt-7ea8da9d3f146679427bb1ee`, immutable image
  `registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:85fae6fd661ab3ea8be1ef054cd6fc8beaf08cefb3d56fee46da6e780bc05d8f`.
  All 310 episodes are pending; provider acceptance is not benchmark success.

- [x] Deeper r3 artifact review supersedes its scalar smoke success: explicit
  think blocks leaked into content on both calls. Final answer itself was
  correct, but channel separation was not qualified. Preserve r3 as a
  provider-successful, qualification-rejected attempt. Evidence downloaded to
  `/tmp/lfm-moe-smoke-r3-evidence/`. Stopped the owned full-eval CPU build before
  submission; full-eval r1 has no submission receipt and remains unused.
- [x] Evidence-backed repair: the pinned DelegatingParser bypasses extraction
  when the prompt reports reasoning ended. Optional parser now waits for output
  disambiguation, then transitions on a parsed closing marker or visible plain
  content. Added actual DelegatingParser token-stream preflight and rejection
  of leaked think markup in native smoke. Ruff/Pyright and 2 smoke tests pass.
- [x] Submitted fresh native smoke r4 before the full evaluation;
  log `/tmp/lfm25-8b-native-smoke-r4-submit.log`. No duplicate submission or
  GPU competition was observed before this retry.
  Accepted provider `pt-37066f278e40d55b2461d23d`, image
  `registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:a98a2397a39f8c414a474e31a04d4e20c194a66fb2fdf3fbcf8b59a6c20a6933`.

- [x] (2026-10-15 23:55Z) 8B native smoke r3 succeeded. Provider exit 0 and
  Trackio outcome reconcile consistently; both generation bundles and serving
  log are retained. The strict tool-result continuation gate passed after the
  actual pinned parser's streaming and nonstream preflights. Workstation has
  no GPU compute processes; only the completed run's cleanup container remains.
- [x] Historical pre-qualification CPU build stopped before submission for full 8B evaluation
  `lfm25-8b-a1b-heldout62-20261016-r1`; log
  `/tmp/lfm25-8b-heldout62-eval-r1-submit.log`. No prior submission exists for
  this ID. Inspect this process and retained state before any retry.

- [x] Live status follow-up: 8B native smoke r2 failed its continuation check;
  GPU is idle and the full 8B evaluation has not been submitted. Both inference
  artifacts were fetched read-only into `/tmp/lfm-moe-smoke-r2-evidence/`.
- [x] Correct the native 8B reasoning/content partition before any further
  benchmark submission; preserve native prompt behavior and the strict gate.
- [x] Added an opt-in Lab plugin `lfm2_optional_thinking`, selected only by the
  new 8B inference revision `@2`. It starts in content and enters reasoning on
  an explicit think marker; it does not change tokens, native template or
  thought generation and never replaces the global qwen3 parser. The 24B
  forced-think binding remains on qwen3. No dependency pins were changed.
- [x] r3 native smoke submitted as `pt-ae4c6a5366e86be19cf4b9bd` under
  `lfm25-8b-a1b-heldout62-smoke-20261016-r3`; inspect
  `/tmp/lfm25-8b-native-smoke-r3-submit.log` and retained execution state before
  retrying. This job also tests the actual pinned parser on direct and explicit
  thinking answers, in streaming and nonstream modes, before allocating a model.
  Image digest: `24024c20209a18f5ae1fe0f777f21c1b6924066a521aceb9a940ef08b2572ffb`
  in the experiment's LAN repository.

- [x] (2026-10-15 23:34Z) Workstation free. First 8B smoke r1 failed before
  model loading with `job package environment activations differ from registered
  selections`. Its failed receipt and diagnostic.log are preserved.
- [x] Repaired the project composition: separate serving-only work packages
  `lfm25_8b_moe_native_tool_smoke.yaml` and `lfm25_24b_moe_native_tool_smoke.yaml`
  bind only inference. Full eval packages now contain only the evaluate job.
  Added Lab-native tool smoke operation and two fragment/argument gate tests
  (2 passed); changed Python passes Ruff and Pyright.
- [x] 8B native smoke r2 submitted as `pt-f04b2c18422674dba4c3c744`; image
  `registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:c2a48e0e751a33dd0805b93c1874b94fd1f2003968e84f9a7e95f5bdb9194cde`.
  Submission log:
  `/tmp/lfm25-8b-native-smoke-r2-submit.log`. Inspect execution state and this
  process before retrying. New ID: `lfm25-8b-a1b-heldout62-smoke-20261016-r2`.
- [x] (2026-10-15 23:38Z) r2 running on the workstation. vLLM 0.29.1.dev4
  resolves Lfm2MoeForCausalLM, BF16, TP=1, context 65536, qwen3 reasoning and
  native lfm2 tool parsing. EngineCore occupies about 17 GB while loading;
  tool-call/continuation result still pending. This passes the old environment
  verification failure point but is not yet a successful GPU smoke.
- [x] Historical CPU preparation of refreshed 8B full-eval image completed; log
  `/tmp/lfm25-8b-eval-split-topology-pack.log` (check active pack process before
  duplicating). Full evaluation must not be submitted before r2 succeeds.

- [x] (2026-10-15 22:15Z) Read the supplied Claude report and located its code via
  Claude session `06f4fe1b-c6e2-45ea-9bb2-d0b2a04f9815`.
- [x] Located benchmark in clean worktree `/home/hammad/projects/worktrees/rl-rft`,
  branch `codex/rft-warm-start`, starting commit `16edeacacb11b7ee96ad08334c9112d01778e637`.
- [x] Verified 62 unique evaluation tasks, 124 training-pool tasks, zero overlap,
  and 310 native episodes for each of the eight historical baseline models.
- [x] Fetched immutable model revisions and ran CPU AutoConfig/AutoTokenizer
  tool-conversation rendering probes for both new models using Transformers 5.14.1.
- [x] (2026-10-15 22:20Z) Added pinned model/inference selections, native-tokenizer
  renderer, project-layer registration, two work packages and candidate entries.
  Both packages pass CLI validation. Both full-eval intents resolve successfully.
- [x] (2026-10-15 22:20Z) Common model and serving binding tests: 32 passed;
  changed Python files pass Ruff and Pyright; nine import contracts kept;
  `git diff --check` passes.
- [x] (2026-10-15 22:20Z) Scheduled thread heartbeat
  `run-lfm-8b-and-24b-heldout62-evaluations`, active every 15 minutes, to gate on
  actual workstation availability and execute the two arms sequentially.
- [x] (2026-10-15 22:38Z) All four corrected images built, passed packaged
  qualification, and published through registry.lan. Verified HTTP 200 and
  exact manifest digest for every image from the workstation itself.
- [x] (2026-10-15 22:39Z) First 8B smoke accepted by dstack as
  `pt-04029b4076cb58f1b3f5d1e2`; Posttrain state `queued`, provider state
  `pending`, requesting `carbonteq-ai-workstation.lan`. Dstack reports no
  offers while the Occupancy job still occupies the only matching worker.
- [x] Retain GPU tool-call smoke evidence before each full evaluation.
- [x] Run both 310-episode evaluations and reconcile native traces.
- [x] Report task-mean capability, harms, coverage, uncertainty, and operating limitations.

## Surprises & Discoveries

- r2 artifact evidence narrows the failure to channel parsing: request 1
  successfully emitted `lookup_code` with `{key: readiness}` and finish reason
  `tool_calls` (240 output tokens). After the tool result, request 2 stopped
  normally after 29 tokens, with no further tool calls, empty content and
  107 characters in the reasoning field. The expected READY-2047 value is in
  that reasoning field. Therefore the model did consume the tool result;
  the current qwen3 parser classified its direct final answer as reasoning.
  Do not call this model failure or bypass the gate by accepting arbitrary
  reasoning as final content. Investigate optional-thinking support with the
  unchanged native 8B template. The 24B forced-think prefix is different.

- GPU handoff exposed a composition mismatch, not a model failure: the old
  combined smoke/eval recipe retained evaluation environment seats when
  serving packaging declared no environment activations. The runtime's
  `_verify_environment_selections` correctly rejected this disagreement.
  Separate inference-only smoke packages avoid the inconsistency without
  weakening runtime verification. Their custom `serve/lfm-moe-native-tool-smoke@1`
  operation keeps one model server alive through a streamed Pythonic call,
  synthetic tool result and final answer containing the supplied code.
  Both generation bundles are retained through normal serving evidence.
  Old publication receipts below describe the previous topology; normal
  `job run` must package current selections and record fresh receipts.

- The current reward-redesign branch has an 85-task suite, not the report's
  heldout62 suite. Its scorer and budgets differ. Never substitute its results.
- The report's clean benchmark is committed in the sibling rft tree. No merge is
  needed to evaluate there. User explicitly authorized working in that tree.
- All eight historical models have 62 tasks with exactly five episodes each in
  the downloaded native artifacts. The report's approximately three-point
  intervals describe some paired differences, not absolute model uncertainty:
  base success has a task-bootstrap interval approximately 0.365–0.606.
- Native assistant generation prefixes differ: 8B ends with an assistant header;
  24B also prefills `<think>`. Reusing the dense 2.6B packaged chat template would
  erase that distinction. Use a separate renderer that delegates to each pinned
  tokenizer's native template and retains the Pythonic tool-call protocol.
- Workstation access recovered. It is occupied by
  `occupancy-synthetic-qwen35-4b-from-opd9b-step250-opd250-chunk1024-workstation-20261015-r7`.
  Latest inspection saw 219 logged updates of 250 and a saved step-200 checkpoint.
  GPU utilization was high; do not launch a second model on it concurrently.
- Broader Lab registry tests report three failures and six passes because nine
  pre-existing RFT/heldout50/heldout62 work packages lack registry entries. The
  new MoE packages are classified. This older inventory debt is not a failure
  of the new model composition and was left unchanged.
- In this checkout, `job plan` resolves intent only. Passing `--provider dstack`
  to plan is rejected. Provider selection belongs on `job run`; `job pack`
  materializes the immutable image separately.
- Readiness inspection caught an unsupported reasoning-parser name inherited
  from an older LFM binding. The pinned vLLM revision registers `qwen3`, not
  `lfm2`, for reasoning. Both new bindings now use `qwen3` for reasoning and
  `lfm2` for Pythonic tool calls, matching the official LFM2.5 vLLM recipe.
  Earlier images containing the wrong reasoning-parser selection are obsolete
  and were never submitted. Rebuild all four jobs from the corrected catalog.
- The public registry hostname resolved locally to Cloudflare and returned
  HTTP 405 during a 24B blob upload after successful build/qualification. Both
  the client and workstation resolve `registry.lan` to `192.168.110.53`, and
  its registry API returns HTTP 200. Publish and launch this experiment with
  the per-command override `--registry registry.lan/carbonteq/posttrain-lab`.
  Do not change global DNS or the Occupancy run. A previously published public
  8B image exists, but use the final LAN receipts for all four jobs.

## Decision Log

- (2026-10-16) Retain 24B full r1 as the 64-episode admission operating-point
  result and run a separate r2 at eight concurrent episodes after observed
  1800s timeouts. This changes only the declared execution concurrency through
  a project-owned job definition, not benchmark semantics or the frozen source.
  Do not combine successful repetitions across r1/r2 or compare their throughput
  as though concurrency were controlled. Keep environment-finalization failures
  missing and publish coverage with any conditional estimates.

- Decision: Work in rl-rft; leave `/home/hammad/projects/rl` unchanged.
  Rationale: exact benchmark and baseline work packages already exist here;
  merging unrelated runtime and training changes would add unnecessary risk.
  Date/Author: 2026-10-16 / user and Codex.
- Decision: Evaluate post-trained LFM2.5 models in BF16, 8B first, then 24B.
  Rationale: these are the requested agent candidates. Pretrained `-Base`,
  quantized weights, older LFM2 generations, and speculative drafters are
  separate experiments. Quantization must not confound the primary comparison.
  Date/Author: 2026-10-16 / Codex.
- Decision: Primary comparison uses the existing LFM sampling: temperature 0.5,
  top-k 50, top-p 1, repetition penalty 1.1. Keep the exact benchmark budgets.
  Rationale: answers how the new models behave under the report's operating
  policy. Vendor-recommended 8B settings (0.2/80/1.05) and 24B settings
  (0.1/50/1.05) can be separate follow-ups if matched-budget truncation is high.
  Do not tune against heldout62 and then present it as untouched evaluation.
  Date/Author: 2026-10-16 / Codex.
- Decision: Start with eight scheduled model sequences and one GPU per server.
  Rationale: MoE active parameters describe compute, while resident weights
  scale with total parameters. Leave memory for long prompts and hybrid state.
  Increase capacity only after measurement; do not copy the dense model's
  64-sequence engine setting. The environment may queue 64 episodes while the
  model server schedules at most eight sequences.
  Date/Author: 2026-10-16 / Codex.
- Decision: Enable batch-invariant inference on both new bindings.
  Rationale: the CLI warns that batching otherwise changes logits. Historical
  baselines did not necessarily enable this, so the new comparison is a model
  and serving-policy comparison; rerun a baseline with this flag if a narrow
  model-only conclusion is needed. Native cache precision remains unchanged.
  Date/Author: 2026-10-16 / Codex.

## Outcomes & Retrospective

Completed 16 October 2026. Native tool continuation qualified for both models;
8B full r1 and 24B full r1/r2 all completed with their 310 native episodes
retained. The capacity repair removed observed 24B timeouts (10 to zero) while
leaving all task/model/token/timeout budgets intact. Frozen LinkedIn WorldState
finalization errors still leave 8B 307/310 and 24B r2 305/310 scored. Complete
62-task qualification is explicitly unavailable; no scores were imputed.

On the common 61 fully scored tasks, success is 35.4% for 8B, 53.8% for 24B,
49.8% for dense LFM2.6B, 55.1% for Qwen4B and 55.4% for Qwen9B. 24B's success
gain over dense LFM is uncertain, but partial credit improves 8.0 points
(paired 95% interval +4.3 to +11.8). 24B does not establish a success advantage
over Qwen; 8B materially underperforms. Capabilities and guard harms are separate.
The report contains all cohorts, domain slices, uncertainty and operating limits.

Durable report and numerical audit are under docs/research and attached to the
new child Page https://chatgpt.com/space/page_3ef91b283c188191bbbcd97e648145bd.
Page readback verifies structure and attached source; native visual opening was
queued by the desktop, so no rendered visual QA is claimed. No further GPU
retry is needed for this frozen experiment. Fixing the known environment error
requires a separately versioned comparison, as recommended in the report.

## Context and Orientation

The report is https://claude.ai/artifact/WDjeDT4tkeEZzLntrRthGq, especially Line 4.
Its exact fixture is `scripts/qualification/fixtures/automationbench_heldout62_v1.json`.
The ordered task-name digest is
`ed81cf1fe88f68bb8420bb46c5fdf3afde24660c7043812da680bb1c7bc9196f`.
The environment revision is
`a0c7378b179e2b267382b8125192876502d82f89` in carbonteq-ai/verifiers-environments.
The catalog is `apps/lab/.posttrain/catalog/automationbench-manifest-steps.yaml`:
environment `automationbench-manifest-heldout62-eval-64k20t-t05-v1` and evaluation
plan `automationbench-manifest-heldout62-64k20t-t05`. It declares strict missing
coverage, equal task means, and `task_completed_correctly == 1` success.

The benchmark has finance 6, HR 5, marketing 5, operations 8, sales 3, simple 30,
and support 5 tasks. It deliberately selects tasks a strong reference solved;
it does not estimate performance on all 800 AutomationBench tasks. No claim
of uncontaminated vendor pretraining follows from exclusion from our training.

Historical task means from retained analysis (success / partial credit /
harmful actions per episode): LFM base 0.4903 / 0.7249 / 1.0613;
Qwen3.5-4B 0.5419 / 0.8133 / 1.1387; Qwen3.5-9B 0.5452 / 0.8372 / 1.0032.
The LFM training variants have no significant partial-credit improvement in
the report's paired task bootstrap; several increase harms. Do not infer
training success from reward or distillation loss alone.

Model selections, checked against the official Hub on 2026-10-16:

| Model | Immutable revision | Total / active | BF16 weight floor | Native context |
| --- | --- | --- | --- | --- |
| LFM2.5-8B-A1B | `5dd22602c2e9f6a097b1de4c4efe0658b605015c` | about 8.47B / 1.5B | 15.77 GiB | 128000 |
| LFM2.5-24B-A2B | `1769516ac588c4c558089c94f27d96b0dc67b183` | about 23.97B / 2B | 44.65 GiB | 131072 |

Both use `Lfm2MoeForCausalLM`; 8B has 32 experts and 24B has 64, both top-4.
The weight floors exclude attention caches, convolution state, activations,
CUDA graphs, allocator overhead, and runtime workspaces. Single RTX PRO 6000
96 GB is the target; the local 8 GB card cannot host these BF16 experiments.
Use TP=1, without expert parallelism, offloading, or a colocated trainer.
At 64K, a full attention-cache allocation for a 24B sequence is approximately
1.25 GiB before other state; eight such sequences need substantial headroom.

## Plan of Work

### Milestone 1: Prepare exact selections

Add a native-tokenizer MoE renderer in
`packages/common/src/posttrain/common/variants/lfm25.py` and register it in
`variants/__init__.py`. Add project overlay
`apps/lab/.posttrain/catalog/lfm25-moe-heldout62.yaml` with both pinned BF16
variants and conservative vLLM bindings. Reuse the existing heldout62
environment and measurement plan without edits. Add the two model-specific
work packages and candidate entries in
`apps/lab/src/posttrain_lab/qualification/gates.toml`.
Register the overlay in `apps/lab/.posttrain/catalog/layer.yaml`. Both work
packages contain an explicit `smoke` job using `serve/vllm-generation-smoke@1`
and an `evaluate` job. The generation smoke verifies load and text output;
it does not replace the subsequent bounded tool-call continuation check.
Success means both packages resolve through the primary CLI with correct
revision, renderer, sampling, environment and target selections.

### Milestone 2: Gate and schedule GPU execution

Record run IDs before submission. Use the normal packaged dstack execution path
and existing protected execution binding. The worker must be genuinely free:
prior dstack run terminal and no training/inference compute process remaining.
Do not stop or alter Occupancy training. Run a native-tool conversation smoke
(call, tool result, continuation, final answer) before each full job. Failure
to load architecture, preserve thinking boundaries, parse calls, or fit memory
stops that arm before benchmark execution. Prefer an existing serve.smoke job
or bounded direct API probe in the same pinned runtime; retain responses and
server startup logs as evidence. Never treat CPU template success as GPU support.

Run 8B then 24B sequentially. Both use 65536 context, 8192 per model reply,
32768 episode output tokens, 20 turns, five attempts per task: 310 planned
episodes each, 620 in total. If direct queue submission cannot yet be completed,
use one thread heartbeat to check capacity and dispatch these concrete packages;
avoid duplicate queue submissions by consulting retained run state first.

### Milestone 3: Reconcile and compare

Retain resolved configuration, evaluation manifest, native traces, environment
source pin, model revision, renderer, runtime/image digest, GPU and concurrency.
Read task identity and scores from native traces. Check all 62 task identities,
five valid repetitions each, error and ending types, retry counts, and trace
sync completeness. Missing evidence makes the headline unavailable under strict
measurement; never silently drop tasks, insert zeros, or expand repetitions by
counting retries. Deduplicate assessment IDs across lifecycle batches before
counting harms. Goal rate uses obligation signals, not inverse guard compliance.

Report task-mean strict success, partial credit, goal rate, harmful actions,
harm-free rate, and truncations by reason. Add domain and simple-versus-multistep
slices with task counts. Use paired bootstrap resampling of tasks, retaining
all within-task repetitions, for differences against each baseline; show
absolute uncertainty separately. Existing baseline scores may be reused only
after verifying environment, scorer and budgets match. Historical throughput
is not comparable to a different concurrency or runtime.

## Concrete Steps

All project commands run in `/home/hammad/projects/worktrees/rl-rft`.

```bash
git status --short
.venv/bin/posttrain --project-root apps/lab work-package validate lfm25_8b_moe_heldout62_eval.yaml
.venv/bin/posttrain --project-root apps/lab work-package validate lfm25_24b_moe_heldout62_eval.yaml
.venv/bin/python -m pytest packages/common/tests/test_model_variants.py packages/serve/tests/test_vllm_bindings.py -q
.venv/bin/lint-imports
git diff --check
```

After protected dstack/image configuration resolves, use CLI `job plan`
with `--job evaluate --explain` to inspect intent, then `job pack` for immutable
package selections. Submit with `job run --provider dstack`
and `--registry registry.lan/carbonteq/posttrain-lab`
using unique run IDs `lfm25-8b-a1b-heldout62-20261016-r1` and
`lfm25-24b-a2b-heldout62-20261016-r1`. Record exact accepted commands and receipts
here once resolved; do not guess a runtime profile or registry.

The automation is `run-lfm-8b-and-24b-heldout62-evaluations`. It checks every
15 minutes, stays quiet while the workstation is occupied, and records accepted
run receipts before advancing. Its smoke IDs are
`lfm25-8b-a1b-heldout62-smoke-20261016-r1` and
`lfm25-24b-a2b-heldout62-smoke-20261016-r1`. Inspect retained state before every
submission. It is a scheduled dispatcher, not evidence of native queue admission.

Read-only capacity command:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -i /home/hammad/projects/ai-infra/.state/ssh/dstack_worker dstack@carbonteq-ai-workstation.lan 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

## Validation and Acceptance

Local configuration validation is necessary but insufficient. GPU acceptance
requires successful native tool-use continuation and 310 reconciled episodes
per arm. Show results even if worse than the baseline; capability gains do not
override harm regressions. Operational acceptance includes no OOM, no parser
failure, explicit truncation counts, and no interference with the prior job.
Run package-specific checks first; if modifying Python contracts, run Ruff,
Pyright and import-boundary validation and record any pre-existing failures.

## Idempotence and Recovery

Before scheduling/submitting, check retained state for both unique run IDs.
Do not submit an already queued, running, or completed ID. Restart only failed
arms with a new suffix and preserve previous evidence. A capacity wait is not
a model failure. If capacity fails, retain strict budgets and reduce sequence
concurrency; do not silently shorten context, quantize, or drop tasks.
Never print or copy execution credentials into plans or artifacts.

## Artifacts and Notes

### Historical publication receipts (superseded topology and parser)

The following table and launch commands are retained history, not current
readiness. In particular, full-eval packages no longer define smoke jobs.
Use the dedicated native-tool smoke packages and the latest Progress receipts.

First accepted run: `lfm25-8b-a1b-heldout62-smoke-20261016-r1`, provider ID
`pt-04029b4076cb58f1b3f5d1e2`. Its retained receipt is
`apps/lab/.posttrain/state/executions/lfm25-8b-a1b-heldout62-smoke-20261016-r1/submission.json`.
It is queued/pending for the workstation, while the Occupancy training provider
run `pt-a1b721bdb256850d47485792` remains running at high GPU utilization. Do not
resubmit this smoke or treat it as unsubmitted based on earlier plan text.

All images use repository
`registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job`; the CLI appends
the project name to the selected registry prefix. Keep the exact per-command
prefix below so cached publication resolution stays consistent. Each receipt
is under `apps/lab/.posttrain/state/publications/`.

| Model / job | Image digest (sha256) | Receipt filename |
| --- | --- | --- |
| 8B evaluate | `f05ae3c6859edc4ec7c3c427e827b6795bde0747282567ab5f639b8534e9597b` | `be79b34e73568cee7166987cae01d15efeff82dd7e249c7317c2718a5665f324.json` |
| 24B evaluate | `1bf11960db20c30cccb4ff82930123f83e419ee5464fa8a16a60db802eb6c750` | `dbf9a654c1b3ff36c6bf386d0a5eb0dbae15e0ac97ff52b8fded3ec4bad05122.json` |
| 8B smoke | `4a5d725f7a54f53f7ca18b17d9477df5b3ac2768caaeea01011147c999ba5530` | `c24353db033600e24cc14db072b1edc130d0b772fbffc8951df28cb65a8c3763.json` |
| 24B smoke | `82136e61f4cd1cc5a0bdabc6a20d868e611fa213836410863424988062dc73f2` | `8952b76a50c359ebbcca36d3ec0ac6ace8391352d4f8a10ecbd8810d63cb77df.json` |

All four manifests returned HTTP 200 and a matching Docker-Content-Digest to
read-only HEAD requests issued on the workstation. Worker disk has 633 GB free;
Docker and dstack-shim are active. These checks establish package accessibility,
not yet model loading, GPU compatibility or benchmark performance.

Use the following commands only after inspecting existing state for each ID.
Do not resubmit an accepted run. The first command was invoked during readiness
preparation; confirm its receipt/status before any retry. Each subsequent command
requires successful completion of its predecessor and a free workstation. After
each generation smoke, retain a bounded native tool-call/result/continuation
probe before dispatching the corresponding full evaluation.

```bash
cd /home/hammad/projects/worktrees/rl-rft
.venv/bin/posttrain --project-root apps/lab job run lfm25_8b_moe_heldout62_eval.yaml --job smoke --provider dstack --registry registry.lan/carbonteq/posttrain-lab --run-id lfm25-8b-a1b-heldout62-smoke-20261016-r1 --timeout-seconds 3600
.venv/bin/posttrain --project-root apps/lab job run lfm25_8b_moe_heldout62_eval.yaml --job evaluate --provider dstack --registry registry.lan/carbonteq/posttrain-lab --run-id lfm25-8b-a1b-heldout62-20261016-r1 --timeout-seconds 86400
.venv/bin/posttrain --project-root apps/lab job run lfm25_24b_moe_heldout62_eval.yaml --job smoke --provider dstack --registry registry.lan/carbonteq/posttrain-lab --run-id lfm25-24b-a2b-heldout62-smoke-20261016-r1 --timeout-seconds 3600
.venv/bin/posttrain --project-root apps/lab job run lfm25_24b_moe_heldout62_eval.yaml --job evaluate --provider dstack --registry registry.lan/carbonteq/posttrain-lab --run-id lfm25-24b-a2b-heldout62-20261016-r1 --timeout-seconds 86400
```

Historical native bundles and paired analysis currently reside in
`/tmp/claude-1000/-home-hammad-projects-rl/06f4fe1b-c6e2-45ea-9bb2-d0b2a04f9815/scratchpad/`:
`eval-artifacts/heldout62-*/traces.jsonl.gz` and `heldout62-analysis.json`.
These are inspection locations, not durable publication. Retain final bundles
through normal Posttrain tracking and artifact storage before temporary files
are removed. The report and stored historical summaries are secondary to traces.

Official model sources:
https://huggingface.co/LiquidAI/LFM2.5-8B-A1B and
https://huggingface.co/LiquidAI/LFM2.5-24B-A2B.
Liquid's published AutomationBench score uses different turn and sampling
budgets; it cannot be compared numerically with heldout62.

## Interfaces and Dependencies

Use existing `ModelVariant`, `RendererContract`, `InferenceBinding`,
`EvaluationPlan`, and managed Verifiers evaluation job definitions. Native
Verifiers traces remain replay authority. No capability-package imports across
train/eval/serve are introduced. This tree pins vLLM
`f09e4479123d348cee87d217c695a22f9b2daacc` and Verifiers
`bc70a7deaf64c8f8e0b41e39c00ac2d1ea1d7e0b`; verify runtime/image manifests
actually contain those selections before making reproducibility claims.
No dependency/fork changes are planned. If GPU compatibility requires one,
stop that arm, record evidence, and follow the fork/pin publication workflow.

Revision note (2026-10-16): Initial plan incorporates the recovered benchmark,
native-template probes, busy-workstation constraint, and user's scheduling request.
Updated after configuration validation to record the active dispatcher, corrected
CLI semantics, batch-invariance policy, local check results and pending packaging.
Readiness revision: corrected the unsupported reasoning-parser selection,
avoided the public registry's HTTP 405 route with a per-job LAN registry override,
published and worker-verified all four immutable images, and began first-smoke
provider admission. Neither full evaluation is claimed as running or complete.
