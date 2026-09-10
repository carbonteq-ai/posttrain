# Implement and qualify GDPO and CAPO on TRL and veRL

Revision 39 — 2026-09-10. Exact localhost request capture from R7 proved that
the active AutomationBench judge still combined its former turn-level default
rubric with the episode-level response schema. Spark consequently interpreted
rubric instructions as task requirements and expanded them until the response
budget was exhausted. This was a product-contract defect, not stale tracking
data or insufficient context. `automationbench-v1` 0.4.0 now exposes only
`AutomationBenchEpisodeJudge` and `EpisodeQualityConfig`, with one versioned
whole-episode rubric and one compact wire schema. Integer evidence indexes are
validated and normalized to stable native message IDs before reward admission;
turn scope, prefix/retrospective selection, and turn annotation namespaces are
removed from the public runtime. The candidate is committed and published at
`carbonteq-ai/verifiers-environments@b14dfe0ba9d60184f36d78786a543242fabfb765`;
its 21 tests, Ruff, formatting, Pyright, lock check, and wheel build pass.
On 14 exact captured requests, the corrected prompt changed Spark from 13/14
length stops and 1/14 schema-valid responses in 85.13 seconds to 14/14 normal
stops and 14/14 valid responses in 36.39 seconds. This closes the contract and
truncation defect, not semantic calibration: successful traces still saturated
at 1.0 while failed controls differentiated. A fresh immutable GPU replay must
demonstrate defect-sensitive, nonuniform episode scores before GDPO training
qualification resumes.

Revision 38 — 2026-09-10. R6 was cancelled during judge startup after live
process inspection showed that its Spark server still used the old 24,576-token
context with the new 16,384-token response cap. A mechanical catalog edit had
changed similarly named fields in unrelated LFM rollout and Gemma judge
bindings rather than Spark's engine block; those unrelated changes are restored
and Spark now resolves to a 32,768-token model and batch-token ceiling. This
also exposed a missing generic admission invariant. Managed native judges now
fail before model startup when their declared input budget plus selected output
budget exceeds the selected self-hosted inference context. The focused native-
judge suite passes 9 tests, including the new negative case. R6 produced no
rollout, reward, or optimizer evidence and must not be resumed; use a fresh
immutable identity after its graceful teardown.

Revision 37 — 2026-09-10. R5 confirmed that 12,288 was only the Spark judge's
response allowance, not its whole episode-assessment budget. The judge retained
a separate 12,288-token input allowance inside a 24,576-token serving window.
After 36 completed requests, 25 still ended by length and only 11 ended by
normal stop; no complete reward group or optimizer update was retained. R5 was
cancelled. The next attempt raises the judge response allowance to 16,384 and
the serving window and aggregate batch-token ceiling to 32,768, while preserving
the 12,288-token input allowance. This leaves 4,096 tokens of configured context
headroom beyond the maximum input-plus-output envelope. It does not alter the
policy episode budget, task population, reward projection, or algorithm.

Revision 36 — 2026-09-10. The first real Spark no-thinking training attempt R4
proved that the inference service and continuous-batching boundary work, but it
did not admit a training group. Thinking was explicitly disabled and the judge
served up to 16 requests concurrently at roughly 1.3–1.4K aggregate generated
tokens/second with 100% GPU utilization. Of the first 69 completed judgments,
54 exhausted the 8,192-token response allowance, 15 stopped normally, and none
failed at transport. R4 was therefore cancelled before any optimizer update;
this is output-budget/model-behavior evidence, not GDPO qualification. The
Spark judge response allowance is now 12,288 tokens at both the Verifiers
request and inference-binding boundaries, while its input allowance remains
12,288 inside the existing 24,576-token context. Fresh immutable R5
`lfm26-gdpo2-spark4b-nothink-judge12k-20260910-r5` uses provider run
`pt-8edc8af984db5f61da17bb28`; it must still prove valid differentiated rewards
and two optimizer updates. MTP remains deferred.

Revision 35 — 2026-09-10. The immediate bounded GDPO qualification replaces
the cancelled Gemma R3 judge with pinned `XHToken/Spark-X2.5-4B` at revision
`5e10fcc0286756aebf7c41dc52c1e42d95c70281`, using its vendor vLLM plugin at
commit `3e1040e63a5907e4c748a485d6795d739dcd5bd6`. MTP is deferred. Thinking is
disabled explicitly in both the typed renderer selection and request body.
Frozen-trace replay already produced 8/8 schema-valid differentiated judgments
in 20.42 seconds at 465.35 completion tokens/second, while the thinking-enabled
cell produced 0/8 valid judgments in 132.77 seconds. This replay admits the
model/runtime/profile combination but does not qualify training. The fresh
two-update run keeps the LFM2.5-2.6B policy, AutomationBench mix, 8x4 population,
12,288-token episode budget, generic eight-component GDPO projection, and local
RTX PRO 6000 target unchanged.

Revision 34 — 2026-09-10. The 0.45 Gemma judge passed the earlier KV-cache
admission boundary in replacement R2, which then exposed an algorithm-type bug
before its first trajectory. The OLMo refill validator read
`settings.active_sampling` directly even though that capability exists only on
GRPO settings; GDPO and CAPO intentionally omit it. The validator now uses
capability-safe access while preserving complete logical batches for GDPO/CAPO
and relaxed complete-group refills only for active-sampling OLMo. A direct GDPO
regression plus the focused rollout/admission suites pass. R2 is failure
evidence and must not be resumed; use a fresh identity after publication.

Revision 33 — 2026-09-10. GDPO submission
`lfm26-gdpo2-gemma12-local-12k-v2-20260910-r1` failed before rollout when the
revision-1 32K Gemma judge exposed 10.41 GiB of KV cache but vLLM required
10.50 GiB to admit one full-context request. This is serving admission evidence,
not an algorithm or reward result. A revision-2 judge binding raises
`gpu_memory_utilization` from 0.40 to 0.45, preserving the 32,768-token context,
MTP-2, concurrency 16, and all scoring budgets with deliberate headroom. The
failed immutable run remains evidence; qualification requires a fresh identity.
The replacement is `lfm26-gdpo2-gemma12-local-12k-v2-20260910-r2` /
`pt-12273d933d5ce86bd3747b16`, packaged from the revision-2 binding and
submitted to the same exact RTX PRO target.

Revision 32 — 2026-09-10. The immediate matched local qualification is reduced
to two optimizer updates per arm so it proves initial collection, one
changed-weight recollection, learner updates, and final adapter publication
without paying for an unnecessary third collection. New revision-2 OLMo3 and
episode-GDPO selections preserve the current 12,288-token episode budget,
4,096-token per-turn cap, 24,576-token actor context, and 8 prompt groups by 4
generations. The older two-step GDPO revision remains reproducible but is not a
valid comparison arm because it used 16x4 and a 3,072-token completion limit.
The paired revision-2 work packages share one comparison family; only GDPO adds
the generic eight-component projection and self-hosted Gemma 4 12B judge. The
immutable submissions are OLMo3
`lfm26-olmo3-2-local-12k-20260910-r1` / `pt-43bcd322a64d6e334f2ffa57`
and GDPO `lfm26-gdpo2-gemma12-local-12k-v2-20260910-r1` /
`pt-77dabd78ca3c8a1aa028e52a`. After the immutable 16K OLMo R3 completed two
finite changed-weight updates without the refill ValueError, the queued 12K
OLMo submission was cancelled before allocation as redundant; GDPO remains the
required two-update qualification. The superseded queued three-update
submissions were also cancelled before allocation. Eleven terminal zero-update LFM2.6B
AutomationBench workspaces have exact-host cleanup tasks pending behind the
active workload; the twelfth had never created a workspace and was finalized
immediately. Retained Trackio failure evidence is intentionally preserved.

Revision 31 — 2026-09-10. Future matched OLMo3 and episode-GDPO qualification
uses a 12,288-token cumulative policy-output ceiling, selected from the retained
R2/R3 distribution rather than the single 4K-turn Airtable tail. The per-turn
cap remains 4,096 because R3 independently produced a second 4K call. The Gemma
retrospective input allowance is reduced in lockstep to 12,288 so it is never
smaller than the policy episode budget; its separate 16,384-token response
allowance and 32,768-token serving context remain unchanged. Already packaged
or running 16K jobs remain immutable historical evidence.

Revision 30 — 2026-09-10. The first 16K-budget OLMo3 run proved the complete
native rollout path (32/32 episodes, no failed/replaced admission, nonzero group
reward variation) but exposed a trainer-side active-sampling refill invariant
before optimizer update one. OLMo3 legitimately refills only the rows missing
after zero-variance groups are filtered; those refills remain complete prompt
groups but need not equal the original 8x4 batch. Posttrain now accepts that
bounded OLMo3 shape while keeping full logical batches mandatory for ordinary
GRPO, GDPO, and CAPO. R2 is retained as failure evidence; a fresh immutable R3
must complete changed-weight optimizer and checkpoint gates before OLMo3 is
qualified.

Revision 29 — 2026-09-10. The matched three-update OLMo3 and episode-GDPO
qualification now uses an explicit 16,384-token generated-output budget for
each complete AutomationBench episode. A separate 4,096-token per-turn limit
keeps individual policy calls bounded without conflating one call with the
multi-turn episode. The 24,576-token actor context admits the observed initial
prompts plus substantially longer trajectories, while the remaining-context
guard makes any late-turn clipping explicit. The retrospective Gemma
judge receives a 16,384-token compacted episode envelope and a separate 16,384-
token response allowance inside its 32,768-token context. Tool calls and policy
outputs remain authoritative; oversized tool observations may be compacted by
the native judge projection. These are new v3 environment selections so the
older 8K comparison evidence remains reproducible. Both arms still use the
same 8 prompt groups by 4 generations, three optimizer updates, and one final
adapter checkpoint.

Revision 28 — 2026-09-09. Before submitting the matched local scalar
comparison, re-audit LFM rollout capacity rather than treating a valid engine
configuration as an efficient one. The previous explicit 512 MiB KV-cache cap
held only about 32K attention KV tokens for LFM2.5-2.6B, or roughly 1K resident
tokens per request at concurrency 32. Both GRPO and OLMo3 now share a 4 GiB
cache cap (roughly 262K KV tokens) and a 32,768-token aggregate continuous-
batching budget while preserving the 12,288-token sequence ceiling, 2,048-token
per-turn generation budget, 8,192-token episode budget, 8x4 population, and
microbatch one. Exact pinned-TRL inspection confirms ordinary GRPO normalizes
by each sequence's selected-token mask and OLMo3 normalizes by global active
tokens, so multi-turn completions are not divided by the per-turn generation
cap. LFM has no native MTP head. Its separate DSpark drafter is currently a
documented SGLang path and remains outside this matched vLLM qualification;
TurboQuant remains an explicit unqualified recommendation. A generic derived
KV-capacity recommendation still needs a model-memory contract and is not
silently approximated by a hardware-only heuristic.

The optimized R9 jobs were packaged locally from the successful exact
Posttrain 0.4.0 candidate wheelhouse and passed offline environment smoke
qualification before dstack submission. GRPO run
`lfm26-ab-grpo20-dev-v040-r9-capacity` is provider job
`pt-2b821d61e66ff2bccbf0f290`, using actual-job image
`sha256:1cf8ee203417b09b704cc6e3dfff6ee1ebcf4cf522184f3a89158185a2c159bd`.
OLMo3 run `lfm26-ab-olmo3-20-dev-v040-r9-capacity` is provider job
`pt-c845d0da6e8eaa55c8c175a8`, using actual-job image
`sha256:870168d2aae50e1b1a2da10c3a46cc4dbdd451f1bd2f01abfae221c70d734f75`.
At submission, both were waiting for the single named RTX PRO 6000 behind
unrelated occupancy-evaluation job `pt-3ad781ed40bcccfa703ca622`; this is
capacity waiting, not optimizer progress or failure. Do not preempt that job.

Revision 27 — 2026-09-09. Live inspection of release-candidate run
`34340433050` established that GPU scheduling was not the delay: the protected
8-vCPU release runner was concurrently rebuilding the TRL and veRL runtime
images. The base image's source and dependency-lock digests were unchanged, but
the workflow had supplied the runner's complete 122-certificate system bundle
as additional image content. An ordinary host CA refresh therefore changed the
base identity and correctly fanned out a rebuild to every child image. Keep
host-side clients on system trust, but give BuildKit only the provisioned
single-certificate CarbonTeq private root that the runtime needs in addition to
its distribution CA store. Release-candidate and manual runtime publication
now retain an explicit immutable image plan, fail before publication when it is
blocked, and separate lock materialization, planning, publication, and registry
verification into visible steps. The broad branch-diff shortcut and its dead
legacy publication block are removed; immutable input identity, rather than an
unrelated catalog or release-file diff, decides image work. The already-running
candidate remains untouched and retains its exact evidence. These changes are
release-path efficiency and observability improvements, not algorithm or
qualification evidence.

Revision 26 — 2026-09-09. The matched twenty-update GRPO and OLMo3
qualification remains on the retained `carbonteq-ai-workstation.lan` RTX PRO
6000 because one worker is presently healthy and idle; the two canonical runs
will be submitted together and serialized by dstack's single worker block. A
hardware-planning audit found that both LFM comparison training bindings still
declared the superseded TRL post5 source after the runtime dependency advanced
to post6. Both bindings now select `trl@1.12.0.post6` and retain the generated
current lock's exact source revision and digest. Catalog regression coverage
compares the active bindings with `locks.toml` so a later fork upgrade cannot
silently leave this campaign on stale provenance. The unused RTX PRO 4500
on-demand training variant is deferred; its target remains available for a
separate measured 32 GiB qualification instead of being substituted into this
matched run without evidence.

Revision 25 — 2026-09-09. The complete external environment repository now
selects the same immutable CarbonTeq Verifiers revision as Posttrain. Published
environment commit `d994073b9632e73c96a57865683133d7a6ebc4bf` removes mixed
upstream/fork dependency URLs across all six packages and passes their
individual and combined-wheel gates. Posttrain catalogs, generated starters,
release constraints, comparison overlays, and CI select that commit. This is a
dependency-closure correction; it does not alter environment reward semantics
or satisfy the remaining GPU qualification matrix.

Revision 24 — 2026-09-09. The next framework release target is `0.4.0` and
GDPO/CAPO are part of its intended scope. This does not close the qualification
matrix: the release notes preserve the four backend cells, packaged scorer,
checkpoint/reload, exported inference, maintained-fork publication, and final
OCI gates as explicit blockers. Existing numerical and five-step toy evidence
cannot be promoted into a general support claim.

Revision 23 — 2026-09-08. Status: the independently governed CarbonTeq
Verifiers distribution now rebases its host-client seam onto upstream main
`e3bcbcbe` (71 commits after v0.3.1) at published CarbonTeq commit `90055c11`.
The compatible external environments are published at `12ff5e1a`; Posttrain's
catalogs, dependency locks, native replay projection, MCP host and runtime
profiles are migrated to the latest contracts. Focused fork, environment,
train/eval, Lab, CLI, Observatory, Job Builder and runtime-image tests pass.
Artifact publication, an immutable release tag, the full release ledger gate,
and production GPU qualification remain open. The earlier portable LFM and
scalar-comparison evidence remains retained within its recorded scope.
Maintain this plan using docs/templates/PLAN.md. It is self-contained; Progress,
Surprises & Discoveries, Decision Log, and Outcomes & Retrospective are living
sections. Repository root: /home/hammad/projects/rl.

## Purpose / Big Picture

Revision 12 also lets a project select a different judge model without changing
its rubric plugin or training algorithm. DSpark (draft-assisted decoding) and
MTP (multi-token prediction) are inference implementation concerns only. A judge
consumes a compatible endpoint; orchestration supplies and owns that connection
where appropriate. Qualifying inference, judging quality and training algorithms
are separate activities with separate evidence. This continuation supersedes
historical instructions that restrict the judge to the original 2B toy profile.

Add two published algorithms as reusable Posttrain capabilities on both TRL and
veRL. GDPO exercises independent reward components and their normalization.
CAPO exercises generative critique, error-span localization, token-level reward,
and nonuniform advantages. Together with existing SAMPO, these must demonstrate
that the framework supports distinct credit-assignment methods without
project-local trainer patches.

Success is four qualified cells: GDPO/TRL, GDPO/veRL, CAPO/TRL, CAPO/veRL.
Each must pass numerical contract tests, an actual masked optimizer update,
Verifiers integration, checkpoint/resume, and exported-model inference. A
registered name, fake backend test, or successful process exit is insufficient.

The initial deliverable is supported synchronous, text-only, parameter-efficient
training on a bounded Qwen3.5 policy profile, followed by the research pilot's
exact 4B policy and 12B judge configuration. Do not infer support for every model,
precision, distributed topology, kernel, or environment from those cells.
The first research use is outcome + whole-trajectory thinking quality + local
feedback, but faithful GDPO and CAPO qualification precedes a combined custom
objective. This plan does not claim benchmark reproduction or superiority.

## Progress

- [x] (2026-09-10) Remove the dual turn/episode AutomationBench judge contract.
  The external v0.4 package now has one episode rubric, one schema, and one
  exported judge. Exact captured-request replay eliminates the pathological
  output expansion, and the immutable environment commit is published. Update
  every Posttrain source pin and active judge code revision to that commit/blob;
  retain historical turn evidence only as read-only compatibility input.

- [ ] (2026-09-10) Complete the two-update Spark-no-thinking GDPO qualification.
  The new work package is
  `apps/lab/.posttrain/work_packages/lfm26_automationbench_gdpo_episode_2_spark_nothink_local.yaml`.
  Static composition and catalog validation pass; 40 focused common, catalog,
  and vLLM-binding tests pass. The actual-job dependency closure pins the Spark
  architecture plugin and records the current `uv.lock` digest. Acceptance
  requires two finite optimizer updates, changed-weight recollection, nonuniform
  retained judge dimensions, native task credit, and final adapter, trace,
  summary, and model artifacts. The frozen replay is only serving/judge evidence.
  Pre-submit image inspection found that the first actual-job image omitted the
  vendor architecture plugin: declaring it only in `posttrain-serve[vllm]` did
  not place it in the shared vLLM kind closure. Attempt R1 was cancelled while
  pulling and consumed no training step. The plugin is now an explicit pinned
  root of `vllm-common`; regenerated online-RL, eval, and serve locks and the
  Spark parser mapping are covered by 68 focused tests. Rebuild the kind image,
  prove the installed plugin entry point from the immutable actual-job image,
  and use a fresh run identity. R2 then failed before model load because the
  catalog's `enable_prefix_caching` setting was not represented by
  `VllmEngineConfig`; the framework now forwards that supported vLLM option in
  both Python and CLI forms with regression coverage. R2 retained zero traces
  and performed no GPU work or optimizer update. R3 exposed the same omission
  for `trust_remote_code`, also before model load or GPU work. Both supported
  fields now exist in the typed engine contract, and the Lab catalog test
  resolves the complete Spark judge through the vLLM adapter so every selected
  engine key is checked together before submission. R4 then reached live
  continuous-batched judging but 54 of its first 69 completions exhausted the
  8,192-token judge response cap; it was cancelled before reward admission or
  optimizer work. R5 raised that response cap to 12,288 but still produced 25
  length completions versus 11 normal stops across its first 36 requests, with
  no retained reward group or update, so it was also cancelled. The 16,384-token
  response / 32,768-token context replacement validates statically and requires
  a fresh run identity. Do not count R4/R5 serving throughput as judge-quality
  or algorithm qualification.

- [x] (2026-09-10) Correct shared late-turn context admission before the
  matched three-update runs. OLMo attempt
  `lfm26-olmo3-3-local-20260910-r1` was stopped after 24 episodes/18 traces
  exposed the same `prompt + 3072 > 13312` failures retained by the completed
  20-update GRPO run. GRPO did not prove the configuration healthy: group
  admission merely tolerated 30 structured overflow errors. The generic TRL
  loopback endpoint now caps an admitted turn to remaining policy context and
  preserves a normal length/truncation result, independent of algorithm.
  GDPO attempt `lfm26-gdpo3-gemma12-local-20260910-r1` was cancelled while
  queued and never consumed GPU. Superseding runs must use fresh identities.

- [ ] (2026-09-10) Run matched bounded OLMo3 and self-hosted-Gemma GDPO
  qualifications before the remote two-machine topology. The exact work
  packages are
  `apps/lab/.posttrain/work_packages/lfm26_automationbench_olmo3_2_local_v2.yaml`
  and
  `apps/lab/.posttrain/work_packages/lfm26_automationbench_gdpo_episode_2_gemma_local_v2.yaml`.
  Each performs two optimizer updates over eight prompt groups by four
  trajectories (32 trajectories per update, 64 nominal trajectories per arm)
  with the same LFM2.5 2.6B LoRA policy, AutomationBench task mix and rollout
  service. OLMo3 retains native active sampling. GDPO additionally uses the
  pinned Gemma 4 12B thinking judge with MTP-2 and the generic eight-component
  episode reward projection. All compute roles resolve to the CarbonTeq RTX PRO
  6000 96 GB server; no OpenRouter/OpenAI credential or request path is
  permitted. Run OLMo3 and GDPO serially because they target one GPU. Acceptance
  requires two finite optimizer updates per arm, changed-weight rollout
  collection after update one, valid retained reward evidence, one
  final LoRA checkpoint per arm, and complete summary/model/trace artifact
  publication. For GDPO, retain the seven independent judge dimensions plus
  native task partial credit and reject uniform/suspicious scoring as a quality
  finding rather than binding the projection to Gemma identity. This is a
  colocated development qualification, not evidence for the still-pending
  independently provisioned production judge topology.

- [x] (2026-09-08) Consolidate Verifiers worktrees. Current edit targets are
  `/home/hammad/projects/verifiers` on `codex/carbonteq-verifiers-latest` and
  `/home/hammad/projects/verifiers-environments` on `codex/verifiers-latest-support`.
  Historical `*-latest`, `*-turns`, and `verifiers-gdpo-capo` directory references
  below are no longer active targets. Follow the verified checkout table and
  mandatory preflight in [the execution architecture plan](async-continuous-rollout-workers.md#mandatory-checkout-preflight).

- [x] (2026-09-08) Record the requested harness execution architecture in
  [Asynchronous rollout requests, continuous batching, and environment workers](async-continuous-rollout-workers.md).
  That scoped plan owns native worker integration, fixed-policy async inference,
  bounded admission, cancellation, and qualification. It adds no instrumentation
  and does not change this plan's algorithm or reward contracts; implementation
  and GPU acceptance for that workstream remain pending.

- [x] (2026-09-08) Correct retained-group trainer alignment before fresh GRPO
  and OLMo submissions. The user explicitly prioritizes algorithm correctness
  and fresh runs before optimizing the Verifiers harness. The TRL candidate
  worktree is `/tmp/trl-parity-probe.WIQjjv`, based on published post4 commit
  `19e6c89a18617f1bd6e6385212705a67f5434962`. Validate complete source-row groups,
  align reward inputs, and preserve GRPO accumulation using zero-credit tensor
  padding after scoring plus retained-row loss normalization. OLMo retains its
  own informative-group selection. Publish the tested fork before updating the
  consumer pin, locks and runtime; submit fresh R8 identities afterward.
  Completed as TRL post5 `b9f3a09369d9cfa21950feef3e110e1fdf779c54`;
  retained-asset publisher `34224729623` passed. Runtime digest is
  `sha256:8230413ea572158e59e3f4099b218474d339869fb3eb1676ebaf23e35d35d03d`.
  43 focused fork tests pass, including actual two-update tiny-model runs for
  GRPO, OLMo's active-sampling path and an empty candidate round. 118 affected
  framework/release checks pass; import boundaries and changed-file typing pass.
- [ ] (2026-09-08) Complete fresh R8 GPU qualification. GRPO R7
  `pt-8e1775fe007e8e61b6ade36b` failed at 12:03:49 UTC, before optimizer
  update one, on actor/vLLM raw parity: mean delta 0.271201 exceeds 0.05 across
  16,384 selected tokens. The admission correction does not establish a fix for
  this separate mismatch. R8 submission commands use the new runtime and retain
  the same parity gate. Do not claim successful training until it passes.
- [x] (2026-09-08 12:20 UTC) Submitted both R8 jobs through dstack for the
  RTX PRO 6000 target. GRPO `lfm26-ab-grpo20-verifiers-latest-r8` is
  `pt-6c377125189495278506fc9e`, image
  `sha256:02c11c015a15e2ba8e9a43892de9783b2ac0f5f0f5c11806be25a2cee2424f17`.
  OLMo `lfm26-ab-olmo3-20-verifiers-latest-r8` is
  `pt-3ce6e57869d9f5f5447ac7c7`, image
  `sha256:78eadccb119b515386cadbabcd500a7f0c6baefc2cc9541f7b66229cfa96120e`.
  Both use 20 steps, 8x4 samples, concurrency 32, microbatch 1, no failed-episode
  replacement, and unchanged raw-policy parity. Submission is not an optimizer
  update or successful qualification. OLMo still owns bounded active sampling.
  At 12:20:40 UTC OLMo was provisioning/pulling on the workstation; GRPO was
  pending/retrying placement with `failed_to_start_due_to_no_capacity` because
  the one specified host was occupied. This is capacity waiting, not a GRPO
  training failure; no R8 optimizer update was yet observed.
- [x] (2026-09-08) Advanced the independent CarbonTeq Verifiers distribution
  from the v0.3.1 base to upstream main `e3bcbcbe5c55297a07a5d1038e37c2408b4a3dbd`
  while retaining host-provided async client injection. Published fork commit
  `90055c11896954fac429bb9120245caa6dc1dd59` and compatible environment commit
  `12ff5e1abfab369b8dec4df3ce83c5984f55ad34`. Migrated latest nested agent
  configuration, MCP 2 hosting, dependency locks and exact-float native replay.
  Fork v1 tests, 27 AutomationBench tests, 10 focused train/eval integration
  tests, 16 Lab catalog tests, 49 CLI tests, 114 Observatory/Job Builder tests,
  and 66 runtime-image tests pass. Full release readiness remains open because
  the new distribution has not yet been tagged/built and the existing release
  ledger also carries an older TRL tag than the currently selected TRL pin.
- [x] (2026-09-08) Chose CarbonTeq Verifiers as an independently maintained
  platform. Its repository owns environment/harness execution, native episodes,
  traces and scorer plugins; Posttrain consumes exact immutable revisions through
  adapters and retains training admission/credit ownership. Upstream review is a
  synchronization input, not a promotion requirement.
- [ ] (2026-09-08) Finish the low-concurrency RTX 3070 Ti canary. Actor and
  colocated vLLM loaded within the 8 GB device and valid AutomationBench traces
  were retained, but the first run exhausted bounded group replacement before
  optimizer update one. Correct the generic candidate-admission boundary and
  require an observed optimizer update plus exported adapter before closing it.

- [ ] (2026-09-08) Run the matched local twenty-update scalar comparison. The
  stale occupancy-engine Gemma 4 12B GRPO task had completed all fifty updates
  but retained the RTX PRO worker after Trackio artifact upload returned 404;
  its finalization container was stopped without deleting mounted run storage.
  Added local GRPO and OLMo3 work packages, both using the seeded 160-task
  population. R1 was cancelled
  before its first update after live process inspection showed the environment
  still capped execution at 32. Corrected R2 images are GRPO
  `sha256:9c48a3e7cb09abc3c04d94ee53150fd700a9a65848c0ee2c63afa4e81e6a3162`
  and OLMo3
  `sha256:68723edfd74c5b2a06187ace4933c5b5e257433cbc2c3bebadf354f46f20d0c5`.
  OLMo3 run `lfm26-ab-olmo3-20-local-r2`
  (`pt-8164c5dcb4090c21fd3aa7e3`) was admitted first and live inspection observed
  exactly 64 AutomationBench subprocesses. It retained 61 successes but failed
  before optimizer update one because three episodes did not produce exactly
  one trainable native trace and scalar training lacked group admission. The
  obsolete GRPO R2 was cancelled before training. Revision 20 adds shared,
  bounded complete-group admission, changes the local update to eight prompt
  groups by four candidates, sets environment and vLLM concurrency to 32, and
  uses micro-batch one with effective/global batch 32. OLMo preserves
  `max_candidate_batches: 10`; invalid-episode admission allows one replacement
  round for only the affected four-sample groups. Fresh R3 submissions exposed
  an independent provider-safety bug: the default local provider accepted the
  RTX PRO hostname constraint on `pop-os.lan` and both attempts failed while
  loading LFM on its 8 GB RTX 3070 Ti. No optimizer update occurred. Local
  admission now rejects that mismatch before submission (19 admission tests
  pass). The corrected dstack submissions are GRPO
  `lfm26-ab-grpo20-local-r4` (`pt-5e7c1e380340436ecf713933`, actual-job image
  `sha256:49076b9a00c951576dda6f7debb4aa526635576e149c6721e4b977b6bf002dfa`)
  and OLMo `lfm26-ab-olmo3-20-local-r4`
  (`pt-715c5ca6e0afb3697caca53b`, actual-job image
  `sha256:340ae9e8117195aed940710956e871944edf9019882c5ccb960b42992faafbe7`).
  Dstack assigned GRPO to `carbonteq-ai-workstation.lan`; OLMo is queued for
  the same worker. Runtime image verification also passes against TRL kind
  digest `sha256:416fc8eeb0b4116850fcc4a40c82a32adadc04f8bc7c8622defc0a18f739380c`.
  The release-gate audit then found and repaired stale common and veRL backend
  Verifiers constraints; 68 runtime-image/builder tests pass, the manifest is
  republished, and veRL now resolves as
  `sha256:d2428f73ebf82d8a495ea4faff82026b391a0f7b65637602b2e9eaea43bcd244`.
  Live R4 evidence at optimizer step one shows 21 of 32 native traces retained:
  five prompt groups are complete and a sixth has one candidate. Completed
  episodes span about 508--611 seconds with no errors so far; their scalar
  rewards include both zero and one. This is rollout progress, not yet an
  optimizer update.
  R5 moved the comparison to the latest independently published Verifiers and
  environment revisions. Real job packaging exposed an environment-wheel
  compatibility bound that isolated tests had missed: AutomationBench required
  `rich<15` while the shared runtime resolves Rich 15. The environment now
  accepts `rich<16` at published commit `12ff5e1abfab369b8dec4df3ce83c5984f55ad34`;
  its 27 tests and a Rich 15 isolated install pass. The republished TRL runtime
  is `sha256:0e3f731c6c959124b1759d5e4aee3143d83a385754ce892fb86008686f569abe`.
  Fresh matched submissions are GRPO
  `lfm26-ab-grpo20-verifiers-latest-r5` (`pt-480d23ab9e417a21be9c99ff`)
  and OLMo3 `lfm26-ab-olmo3-20-verifiers-latest-r5`
  (`pt-7bec6e5d3c1fab585b999cc3`). R5 retained 19 native traces, then failed
  before optimizer update one while TRL's one-time actor/vLLM raw-policy parity
  check recomputed log probabilities over the full padded 32-row rollout batch.
  The actor occupied about 94.63 GiB and could not allocate another 108 MiB.
  OLMo3 was cancelled before placement because it shares that actor-scoring
  path. This was diagnostic-probe memory, not evidence that a GRPO update or the
  selected 32-way rollout concurrency intrinsically exceeds the GPU.

  The generic correction is published as TRL `1.12.0.post3`, fork commit
  `db85b262d9bfe1fc12bff173e2d81e8a54e3f2f7`: the raw parity check now builds
  a distinct selected-row probe whose prompt-plus-completion length is capped
  at 4,096 tokens, while rollout and optimizer-update tensors remain unchanged.
  Ten focused fork tests pass; retained publication workflow `34217148045`
  passed exact-byte readback and clean installation. Posttrain exposes and
  records the cap through the backend-neutral training binding, adapter tests
  pass, the exact internal package is locked, and the republished TRL runtime is
  `sha256:6e37b01ebb0fbb259cf9163dbb7e713bf09d731eb4ad14ddb4ce3528f58fe243`.
  Fresh R6 submissions are GRPO `lfm26-ab-grpo20-verifiers-latest-r6`
  (`pt-6ba56b85a3ca7bb3624db7e7`) and OLMo3
  `lfm26-ab-olmo3-20-verifiers-latest-r6` (`pt-a80be5557417d061f47779ce`).
  GRPO is running on the requested RTX PRO 6000; OLMo3 is queued behind it.
  The item remains open until both runs
  complete, retain checkpoints/evidence, and pass the common evaluation.
- [x] (2026-09-08) Diagnosed false RunPod on-demand A100 admission to the
  maintained dstack adapter's offline-catalog fallback, not the dynamic fleet
  or Posttrain target translation. Published dstack candidate
  `fcf257da683879bd93d863c1d870ae8b549ff8fb` enables live on-demand GPU
  discovery and invalidates cached offers after provider capacity rejection.
  The two queued comparison arms subsequently acquired capacity on the old
  server: GRPO began training, while the OLMo arm failed at global step zero
  because its trainer received no samples. Dstack deployment, OLMo input
  repair, and a fresh OLMo run remain open.

### Secure Cloud execution inventory and placement — 2026-09-08

- [x] Verify the live dstack RunPod backend is restricted to Secure Cloud. Its
  stored configuration explicitly sets `community_cloud: false`, enables only
  named Secure Cloud data-center regions, and retains managed run storage.
- [x] Inventory current on-demand capacity without allocating a GPU. RunPod
  advertises one L40 48 GB at $0.82/hour, three A100 80 GB offers at
  $1.59/hour, four RTX PRO 6000 Server Edition 96 GB offers at $2.09/hour, and
  two RTX 4090 24 GB offers at $0.74/hour. L40S and A6000 have no current
  on-demand offer in the configured Secure Cloud regions. Availability and
  price are snapshots and must be rechecked at submission.
- [x] Add an immutable Secure Cloud on-demand A100 80 GB target for vanilla
  GRPO, OLMo 3 and the GDPO policy/trainer process. It selects the independently
  managed `runpod-secure-ondemand-workers` dynamic fleet, which scales from zero
  to two matching workers, and retains one GPU, managed run storage, a $1.75/hour
  ceiling, and the same 12-hour job bound. Do not use
  the 48 GB L40 for the full comparison until this exact 12,288-token,
  micro-batch-one colocated trainer/rollout shape passes a production-size
  optimizer-update memory gate there.
- [ ] Add an immutable Secure Cloud on-demand RTX PRO 6000 96 GB target for the
  Gemma 4 12B judge service. Keep MTP-2 and 16 serving sequences as inference
  settings; they do not change GDPO reward semantics.
- [ ] Extend the remote execution composition so GDPO attaches to a separately
  provisioned judge service. The current `train/gdpo-episode-judged@1`
  definition starts a `ManagedInferenceService` inside the training task; a
  judge `InferenceBinding.target` is provenance but does not by itself allocate
  a second dstack worker. Prove service readiness, exact model/engine identity,
  scoped credentials, consumer attachment, and owner-only teardown on RunPod
  before launching the 50-update GDPO arm.
- [ ] Qualify a corrected Secure Cloud target before returning the comparison
  to A100 on-demand capacity. GRPO `lfm26-ab-grpo50-v2-r1`
  (`pt-7e81ab96d45acac8cb363769`) and OLMo3
  `lfm26-ab-olmo3-v2-r1` (`pt-dab35215fe3f94f5a469bf9a`) both reached the
  same A100 host and failed before training because the host's NVIDIA driver was
  too old for the packaged PyTorch/CUDA runtime. Do not retry that target until
  scheduling expresses and enforces runtime/driver compatibility. The local
  twenty-update comparison above is the current execution path. After accepted
  scalar checkpoints exist, launch GDPO with independently owned policy and
  judge services; preserve run identities so no failure repeats a completed arm.

### LFM2.5-2.6B fifty-update GRPO/GDPO comparison — 2026-09-07

The user replaced the earlier Qwen toy policy with the post-trained agentic
`LiquidAI/LFM2.5-2.6B` policy and expanded the comparison from five to fifty
optimizer updates. Use the immutable upstream revision
`654f9463ce32b05d0429d76fe1f580b27d4c1ac0`. This is a new model-profile
qualification as well as an algorithm comparison: the 2.6B checkpoint's
official template differs materially from the existing 1.2B Thinking template,
so it receives a separate package-owned renderer contract rather than silently
reusing the old wire format.

- [x] Resolve the exact post-trained 2.6B artifact, model metadata, license and
  official chat template. Preserve explicit reasoning content and Pythonic tool
  calls through a distinct `lfm2.5-tools-thinking@2` renderer contract.
- [x] Validate the new common/catalog identity and exact renderer behavior,
  then run a one-rollout preflight against the selected vLLM/TRL runtime before
  spending the fifty-update budget. Twenty focused model/contract tests pass;
  the RTX PRO 6000 preflight loaded the pinned LFM checkpoint, completed two
  rollouts and optimizer execution, exported/reloaded the exact zero-signal
  adapter, and completed 32 base-model evaluation rollouts. Zero native reward
  correctly produced no weight update, so this proves compatibility rather than
  learning.
- [x] Add a true vanilla GRPO control that consumes only AutomationBench native
  scalar partial credit and does not launch or invoke an LLM judge. Do not call
  single-component GDPO “vanilla GRPO.”
- [x] User selected a third OLMo 3 GRPO arm. It uses the framework's explicit
  OLMo 3 recipe: scalar native reward, no advantage standardization, 0.2/0.272
  asymmetric clipping, token-truncated importance sampling bounded above at 2,
  and bounded active replacement of constant-reward groups. Report it separately
  from vanilla GRPO.
- [x] Freeze one comparison manifest: identical policy revision, renderer,
  training task slice, seed, sampling, token budgets, LoRA update, optimizer,
  batch shape, hardware and fifty-update limit. The deliberate difference is
  scalar GRPO versus episode-level eight-component GDPO with the qualified
  Gemma 4 12B judge and recorded weights. The executable task is
  `scripts/qualification/lfm26-three-arm-comparison.dstack.yml`; dstack run
  Initial run `posttrain-lfm26-three-arm-r1`
  (`faf36419-50e3-465b-b760-87e23582c335`) stopped before replay or training
  because the calibration command passed the replay directory instead of the
  materialized JSONL file. Gemma loaded and stopped cleanly. The corrected
  fresh-output R2 job `posttrain-lfm26-three-arm-r2`
  (`92bfb761-c066-4148-a373-f1c256196166`) supersedes it. Its fail-closed
  Gemma gate passed 8/8 heterogeneous retained episodes with zero invalid
  outputs, only one all-perfect episode, and nonconstant scores in every one of
  the seven judge dimensions. R2 was stopped during vanilla GRPO after its base
  evaluation exposed an invalid 8,192-token policy context ceiling: 19/40
  trajectories reached `context_length` and two reached provider output length.
  One support task required 7,010 initial prompt/tool-schema tokens, or 9,058
  tokens after reserving the unchanged 2,048-token per-turn response budget.
  Preserve R2 as diagnostic evidence; do not compare its 42.5% exact-success
  result with corrected runs.
- [ ] Pass the R3 budget/throughput preflight before another fifty-update run.
  R2 preflight `60d4aff9-b943-47f1-bdec-b63281d7250d` proved the 12,288-token
  policy context removes context-length failures at 32-way evaluation, then
  completed five small-shape GRPO updates; its only terminal failure was a
  post-run audit bug that discarded the step-less final runtime metric before
  looking it up. The corrected production-shape R3 preflight
  `posttrain-lfm26-budget-speed-r3`
  (`e44580bd-6b31-4121-8816-122f0b133e13`) is running on the RTX PRO 6000. It
  retains the 8,192-token episode and 2,048-token per-turn generation budgets,
  evaluates with 64 concurrent LFM requests, runs sixteen prompt groups per
  update with four same-prompt candidates in each group (64 trajectories in one
  policy wave), provisions 32 concurrent Gemma judge sequences, and performs
  one real seven-component GDPO update plus provenance, numerical, update and
  adapter-reload audits. R4 passed its 20-episode evaluation with zero failed
  or context-limited trajectories, then exposed that one native episode failure
  aborts the whole 64-row collection before group admission can retain healthy
  groups. Across three collections, 159 episodes had valid judge evidence, 24
  exceeded the five-minute scoring deadline, and nine exhausted judge evidence
  admission; seven of the latter invented a `reasoning_content` suffix on an
  otherwise valid message ID. The bridge now retains successful occurrences and
  retries only the complete four-response group containing a failed occurrence;
  missing/invalid evidence still never becomes zero reward. The judge deadline
  is 900 seconds and the prompt explicitly forbids evidence-ID suffixes. At the
  user's direction, the exact Gemma paired assistant is enabled with MTP-2 for
  the next run; this changes inference performance only, not reward semantics.
  The fifty-step jobs save checkpoints every ten updates.
  Full comparison R3 failed before model loading because the external judge
  schema still bounded its timeout at 300 seconds. The schema now admits the
  explicitly bounded 900-second episode-judge deadline, its focused contract
  tests pass. Full comparison R4
  (`f4347491-1dc6-4176-8748-ee6dc633d770`) was stopped during judge replay
  calibration because the requested comparison was to begin directly with
  training, not rerun calibration and base evaluation. Corrected R5
  (`53d025a5-3593-4f98-98cf-64d1af8d4fd8`) begins with vanilla GRPO, then
  OLMo 3 and GDPO; the final report reuses the completed, matching R4 preflight
  baseline. The actual GDPO command, not merely the removed calibration command,
  now enables the pinned Gemma paired assistant with MTP-2 and the 900-second
  judge deadline. Measure acceptance and latency rather than assuming MTP-2 is
  faster, because vLLM warns that repeated use of one MTP layer can reduce
  acceptance above one speculative token.
  R5 was stopped at optimizer step 22 after its durable evidence showed all 384
  sampled episodes had zero native reward, zero advantages and zero gradient.
  The policy emitted valid LFM Pythonic calls inside message content, but the
  renderer did not project that model-native form to structured tool actions,
  so AutomationBench never executed a tool. The generic policy-message bridge
  now recognizes only the exact declared `lfm2_pythonic` block, accepts only
  declared function names with keyword-only literal arguments, and preserves
  the original sampled tokens as replay authority. It also accepts both OpenAI
  nested tool definitions and Verifiers' normalized flat tool records.
  Qualification additionally selects the worker's preinstalled `uv`, mounts
  the worker's system CA bundle, and tells `uv` to use system certificates;
  missing public harness dependencies fall back to PyPI without disabling TLS.
  One-update smoke R8 (`d0202da9-6733-4943-92b3-9bd804085724`) completed four
  real AutomationBench trajectories with native reward 1.0 and exported a
  checkpoint. Its zero gradient is expected because all four candidates solved
  the same simple task, not evidence of the former no-action defect. Full R6
  (`dc78535c-83b7-456b-948c-cbaf36174403`) then exposed a second model-family
  boundary before optimizer step one: full-history rendering changed the final
  BPE boundary of earlier sampled LFM tool calls, so Verifiers correctly retained
  both the sampled and retokenized branches and rejected 16 of 64 trajectories.
  The LFM renderer adapter now extends tool cycles from the retained prompt and
  completion IDs, appending only the template's newline and newly observed tool
  messages. It never retokenizes earlier policy output. Read-then-write smoke R9
  (`a090be4a-e884-485a-b604-f1e8bfa8b858`) completed all four multi-turn
  trajectories, executed both tools, returned native reward 1.0, completed an
  optimizer step and exported its checkpoint. Full R7
  (`b297a69c-7b27-4f38-ae14-8249679ac336`) completed its first 64-trajectory
  vanilla-GRPO rollout but failed before optimizer update one while computing
  actor log probabilities. The 96 GiB worker had about 950 MiB free and the
  forward pass requested another 1,008 MiB. No checkpoint or training artifact
  exists. The Gemma judge was not running: scalar GRPO bypasses
  `bind_native_judges`; the resident vLLM process was the colocated LFM policy
  rollout engine, selected with sleep during optimization. The replacement
  uses per-device micro-batch one and preserves the 16 prompt groups, four
  generations per group and 64-way rollout concurrency.
- [x] Replace the workstation-only comparison launcher with portable Posttrain
  job packages before R8. The current direct dstack YAML mounts a workstation
  virtualenv, model cache and evidence directory, so changing its fleet would
  not produce the same logical job on RunPod. Publish the exact AutomationBench
  fork revision, bind each training/evaluation arm as an independent work-package
  job, package it through the immutable online-RL job-image path, and select
  either the local RTX PRO target or
  `targets/runpod-rtx-pro-6000-spot-qualification`
  from the same work package. Target resolution deliberately produces a distinct
  package identity because execution selections are part of the digest; source,
  task snapshot, algorithm settings and dependency closure must remain equal.
  RunPod must use managed run storage and exact
  checkpoint recovery; the already-completed R4 base evaluation remains an
  external comparison input and is not rerun.
  The Lab host now registers backend-neutral judged SAMPO, GDPO and CAPO job
  definitions with an explicit `judge_inference` seat. This keeps the
  environment-owned rubric separate from the host-owned managed endpoint and
  makes each composition packable instead of relying on the qualification
  script's process-local launcher. GRPO and OLMo3 remain scalar native-reward
  jobs and therefore do not allocate a judge service.
  The former TRL `grpo.py` adapter is now the small
  `policy_optimization.py` lifecycle facade. Runtime/config translation,
  rollout collection plus credit projection, and actor telemetry live in
  `policy_config.py`, `policy_rollouts.py`, and `policy_telemetry.py`
  respectively. This reflects that the adapter serves GRPO, OLMo3, SAMPO,
  GDPO and CAPO rather than making those methods appear to be GRPO branches.
  The three training arms and shared held-out evaluation are now independent
  work packages over the same immutable catalog selections. The environment is
  pinned to `carbonteq-ai/verifiers-environments` commit
  `7c2ba218a2368b2fd2dd45ea29e6511b4aea7f41`, Verifiers is pinned to
  `8e8f3042481c0996a58c3de0f86d55406725b6c1`, and the published TRL runtime
  parent is
  `sha256:c7f78d028066c46a55d3b4aae913a9b2e2309c452127d5655395e1a11d193f8e`.
  Offline package qualification exposed the Verifiers 0.3 `Environment` to
  abstract `Env` migration; the framework now recognizes both names and uses
  Verifiers' official `load_environment` factory. The vanilla GRPO package
  `fd35c26ad5771197a57e95cefb3eb8de3ce6b563babcabd708b9e4ed77eff191`
  passed offline Taskset loading and was published as actual-job image
  `sha256:df7503ca5a8b8b881dff9e10a7b7caa78f412e2d3a3f3159b480ce21e2d035b3`;
  its compressed delta over the shared parent is 8,437,458 bytes. Live run
  `lfm26-ab-grpo50-r10` is the first sequential arm. During submission, the
  framework correctly rejected an upstream dstack client that could not encode
  event-specific retry policy. The ai-infra client is now pinned to maintained
  fork commit `88f5b319b54ddfb8c2415da8c93f0fc3ab6a7e96`, matching the deployed server,
  rather than degrading interruption/capacity semantics to a legacy boolean.
  The remaining packages also pass offline taskset qualification and are ready
  without allocating GPUs: OLMo3 package
  `b18ab757e4a6ee5cc5a9f5ac807159c5ce2f15dd71d8827081ed528f240929f0`
  at image `sha256:7414e9117876ab537cad7abd1168908d7f39baaf588322c67ef9f73ced21fcec`,
  episode-GDPO package
  `e2300060c3040aa36c8f7b3530bdb7d0d8c7c7ae24f1ef559dfd03b1b72f6091`
  at image `sha256:b3dd86e2fbd9a18cab1346cb7d410484baa08f1964c891416547dbb177901562`,
  and held-out evaluation package
  `f44402f97de087ce19a2f9d2aaf6a44444f4edc932bf73d3ddce79ab00e4aef6`
  at image `sha256:9842b436f9c50d3225e5be7845bf5e3d865bebab79e1f6621cea947ecf67daa2`.
  Runtime-image planning now maps all policy-optimization kinds (`train.grpo`,
  `train.sampo`, `train.gdpo`, and `train.capo`) to the online-RL family; this
  removes the last legacy GRPO-only enumeration from portable packaging.
  Live admission then exposed a separate infrastructure bound: the durable
  publication controller serializes recursive OCI graph copies, while dstack's
  RunPod readiness precondition allowed only fifteen minutes. R10 incurred no
  provider allocation, timed out at that precondition, and entered its bounded
  no-capacity retry. The ai-infra RunPod configuration now sets the maintained
  dstack contract's one-hour maximum, with validated bounds and a static release
  test. The server recovered healthy after deployment; R10 provider submission
  2 has a fresh `2026-09-07T13:42:51Z` readiness deadline and remains queued at
  zero GPU allocation while the controller drains its earlier image backlog.
  R10 subsequently reached RunPod worker `198.13.252.37` and initialized the
  policy runtime, but failed before its first rollout because the immutable
  environment still installed upstream Verifiers
  `b2e4e8157783b2c0dffc7821044c87f29f1c3ccf`; that release's
  `Env.serving()` cannot accept the host-owned policy client required by the
  modern bridge. The local workstation had exercised an unpublished candidate,
  so the earlier local success did not establish portable dependency closure.
  Upstream main `e3bcbcbe5c55297a07a5d1038e37c2408b4a3dbd` still lacks
  the seam. CarbonTeq's maintained fork now publishes the tested implementation
  at `8e8f3042481c0996a58c3de0f86d55406725b6c1`; all three real
  server/static/elastic interception cases pass. AutomationBench pins that fork
  in published environment commit
  `7c2ba218a2368b2fd2dd45ea29e6511b4aea7f41`. Framework package,
  catalog and runtime locks select the same immutable pair. R10 remains failed
  with no rollout, optimizer update or checkpoint and must not be admitted.
  The replacement TRL runtime is published at
  `sha256:7f7e645d6f431752f146098a58e03f0213d4746b08634cdb05ecec7e563a3a27`.
  Its complete candidate override retains `verifiers` as a parent-provided
  package, so the environment lock contains only the selected AutomationBench
  wheel and its hash. The resulting actual-job image is
  `sha256:db336bdc2a44963556b0e70f39aa2e67bd3821fad49b48d948899cdcb70e1dc8`;
  BuildKit verified 20 inherited layers, 18 new layers, and an 8,438,337-byte
  compressed delta before publication. Fresh run `lfm26-ab-grpo50-r11` was
  submitted as dstack run `pt-c92a251730a83fb7e54f22f0` and is queued for the
  RunPod RTX PRO 6000 spot target. No rollout or optimizer evidence exists yet.
  R11 later acquired worker `157.157.221.177` and collected 40 usable
  trajectories from its first 64-response update, but 24 native episodes ended
  unsuccessfully, so no optimizer update or checkpoint was admitted. Failure
  finalization exposed a second independent defect: Trackio's convenience
  metadata summed legacy scalar reward values while Verifiers v1 serializes
  weighted `{score, weight}` records. The maintained Trackio fork fixes that
  query projection without mutating native replay data;
  `carbonteq-trackio==0.31.5.post14.dev20` at
  `1f7ceba9f5728259de6cb8801a7fd8a5dcd49752` was published by workflow
  `34138466438` and promoted byte-for-byte by `34138583808`. All three
  comparison environments retain a bounded Verifiers episode-retry ceiling;
  R12's captured agent configuration used zero agent retries and no failed
  episode was replayed.
  Candidate runtime parent
  `sha256:ad4ec14d3b6da272fb7921761bfeae5c7bd66382c86e1b3c36385e76b56f99d7`
  and actual-job image
  `sha256:13cc04cf1e66186167d01f6c4bd893d35ec19c67676ab3d336c6960ef6d7eec8`
  passed offline Taskset qualification. Run `lfm26-ab-grpo50-r12` acquired a
  RunPod RTX PRO 6000 Server Edition, retained all 64 native episodes, and
  failed before optimizer update when 24 episodes exhausted the 1,800-second
  agent budget under 64-way rollout concurrency. Each failed episode contained
  exactly one trainable trace and recorded `HarnessError: agent timeout`; the
  native replay artifact digest is
  `45ca007c388d74c88ea97322b0c64dd7dded88e8cb9a8e56bf6d9e3de6a62ba2`.
  The matched comparison now caps policy environment and vLLM concurrency at
  32 and Gemma judge sequences at 16. The rebuilt GRPO capsule passed offline
  Taskset qualification and was published at
  `sha256:6183ee51a731890701aa2d688fd6883874bd3b9951569dfbf17456b32d024d62`;
  run `lfm26-ab-grpo50-r13` was submitted as dstack run
  `pt-b3897c6613f86276f04280ab`. It acquired Secure Cloud RTX PRO 6000 workers
  twice, but both spot instances were interrupted by provider capacity after
  approximately ten minutes and three minutes respectively. It is terminal
  pending evidence with no accepted optimizer update and is superseded by the
  Revision 18 on-demand placement plan.
- [x] Replace the two-task toy selection with a frozen, explicit twenty-task
  training mix and disjoint twenty-task evaluation mix. Each contains eight
  easier `simple` tasks spanning one/two assertions and one-to-four tools, plus
  lower- and middle-complexity examples from sales, marketing, operations,
  support, finance and HR. The task plugin now accepts exact `task_names`,
  rejects duplicates/unknown names, and retains original dataset indices.
- [ ] Measure the untrained base and all three exported adapters on the same frozen
  evaluation slice, kept out of optimizer updates. Report native partial credit
  and exact task completion first, with trajectory counts and uncertainty;
  report judge dimensions and inference cost as diagnostics. Training reward
  curves alone are not evidence of task improvement.
- [ ] Audit finite/nonzero advantages and updates, checkpoints, exact export
  reload, invalid-judge fail-closed behavior, and retained trace provenance.
  Fifty updates remain a bounded engineering comparison, not a statistically
  powered claim of general algorithm superiority.

### Episode-level multidimensional comparison — 2026-09-07

Prompt-correction gate (2026-09-07): attempt o proved that valid structured
output is not sufficient scorer evidence. Nanbeige assigned 1.0 to all seven
dimensions even when a calendar trajectory visibly replaced an all-day event
with a timed event, added an unrequested attendee, and falsely reported success.
Do not rerun training with that scorer. The external plugin now uses the
domain-general `general-agent-episode@1` contract: a system-priority instruction,
a requirement ledger before scoring, independent dimension anchors, exact
task-selected tool schemas, message-addressed evidence, and validation that
forbids a perfect relevant score when the ledger records a violation. Native
assertions, rewards, reference answers, and hidden end state are intentionally
withheld from the judge; trainer/GDPO contracts and component weights remain
unchanged.

- [x] Added an offline replay tool over retained native trace JSONL, preserving
  original messages and losslessly minified tool observations. It deduplicates
  by trace identity and can take a stable round-robin sample across task/outcome
  strata before any provider call.
- [x] Preserved 118 unique episodes from 12 prior runs and materialized a
  24-episode historical sample spanning positive, zero-reward, rejected-action,
  and tool-error strata for both AutomationBench tasks. Native outcomes remain
  out-of-band and are never included in judge inputs.
- [x] Added five frozen, domain-diverse engineering controls for all-day action
  mismatch, wrong arithmetic, honest external failure reporting, a minimal
  exact answer, and contradictory tool state. These are calibration controls,
  not few-shot prompt examples and never training data.
- [x] User narrowed prompt calibration to Gemma12B. On RTX PRO 6000, task
  `posttrain-general-episode-prompt-r12-t` produced valid differentiated scores
  for all five controls. The initial harness reported 4/5 only because it
  redundantly required the calendar requirement ledger to tag grounding after
  separately requiring a non-perfect grounding score; Gemma scored grounding
  0.0 and correctly identified both action violations. Removing that redundant
  explanation-shape constraint yields 5/5 without changing the prompt, model,
  scores, or substantive acceptance. No optimizer or policy rollout ran.
- [ ] Replay the accepted Gemma prompt over the 24 historical episodes, review
  score distributions and requirement ledgers against out-of-band native
  outcomes, and reject suspicious saturation before restarting five-step GDPO.
- [x] Historical smoke replay covered one episode from each of eight strata.
  Under contract v4, five verdicts were admitted and three failed closed because
  the free-form assessment mapping omitted dimensions. The five admitted
  verdicts had zero uniformly perfect episodes and scores spanning 0 through 1.
  Contract v5 replaces that mapping with seven fixed required schema fields;
  replaying the three formerly invalid episodes then admitted 3/3 with
  differentiated scores. Across both diagnostics, native-positive episodes
  scored materially better than zero/tool-error/rejected-action examples. These
  are two scorer-contract revisions, so they validate the correction but are
  not one homogeneous eight-episode qualification corpus.
- [ ] Rerun the homogeneous historical sample under contract v5, then use only
  that exact prompt/schema/scorer digest for the next five-update GDPO run.

Capacity recovery (2026-09-07): attempt n failed before allocation with no
offers, rather than remaining queued, because this recipe selected `retry: false`.
After confirming m stopped and the RTX PRO 6000 was idle, resubmit as
`posttrain-episode-judges-r12-o` with fresh `episode-gdpo-*-o` directories.
Select `retry: {on_events: [no-capacity], duration: 1h}` so temporary capacity
shortages remain retryable for a bounded hour. Execution errors and interruptions
remain nonretryable; no partial training is automatically restarted. The model,
budget, weights, rubric and five-step settings are unchanged.

User-requested clean restart (2026-09-07): stop comparison attempt m and
resubmit as `posttrain-episode-judges-r12-n`, using fresh
`episode-gdpo-nanbeige-n` / `episode-gdpo-gemma-n` evidence directories.
Preserve prior outputs. No configuration changes beyond run/output identities;
the increased 12,288-token judge output allowance remains selected.

Budget retry (user authorized 2026-09-07): attempt l exited with bounded reward
admission exhausted before completing the first update. A retained assessment
hit `LengthFinishReasonError`; its successful retry used 5,985 output tokens,
including 5,046 reasoning tokens, against 6,144 allowed. Increase judge output
to 12,288 and context to 24,576, keeping input at 12,288 and rollout output at
8,192. Both judge profiles receive the same budget. Attempt m uses fresh
`episode-gdpo-nanbeige-m` / `episode-gdpo-gemma-m` output directories; retain l
unchanged. Weights, rubrics, model revisions and five-update settings are unchanged.
This retry addresses output capacity, not the separate native-date/judge-score
disagreement, and is not yet a completed qualification.

This continuation supersedes the collapsed turn-quality profile for the new
judge comparison, not the existing generic turn reward API or historical runs.
The existing canonical ownership remains unchanged: external AutomationBench
owns seven episode rubrics; composition owns judge inference; GDPO owns credit.

- [x] User approved independent whole-episode planning, logical correctness,
  grounding, verification/self-correction, progress/efficiency, action quality,
  and answer quality, alongside native task partial credit.
- [x] User retained explicit relative weights: native partial credit 0.50;
  planning 0.05; logic 0.05; grounding 0.05; verification 0.03; efficiency 0.07;
  action 0.15; answer 0.10. The project-owned qualification profile aligns these
  by name and rejects schema drift. They weight normalized advantages; they are
  not percentages of actual gradient influence or guaranteed task dominance.
- [x] External candidate plugin preserves seven scores and exact prompt strings,
  validates message evidence, and rejects incomplete/inapplicable assessments
  without fabricating neutral rewards. Package: Ruff/Pyright pass, 24 tests pass,
  wheel builds. No fork commit or consumer pin publication is claimed.
- [x] Four priority-profile tests and ten GDPO numerical tests pass with
  `PYTHONPATH=scripts/qualification uv run --offline pytest scripts/qualification/test_episode_reward_profile.py packages/train/tests/test_reward_advantages.py -q`.
- [ ] Run identical five-update TRL selections on RTX PRO 6000 with Nanbeige3B
  and Gemma12B thinking judges, preserving identical policy, weights, seed,
  task selection and budgets. These are exploratory runs, not a statistical
  winner or a substitute for the remaining backend/release gates.
- [ ] Independently audit all eight components, normalized advantages, exact
  judge prompts, native token provenance, updates and export/reload; compare
  native completion first, then component variation and inference cost.

Candidate launch: `scripts/qualification/automationbench-episode-comparison.dstack.yml`,
task `posttrain-episode-judges-r12-l`. Runs execute sequentially and fail closed
before the second run if the first fails its audit. Evidence directories on the
worker are `/var/lib/posttrain/qualifications/nanbeige-r12-20260906/episode-gdpo-nanbeige-l`
and `episode-gdpo-gemma-l`. External judge source blob is
`b2e712091423e5580b86735126a141401cba3ceb`; candidate wheel SHA-256 is
`6f8d40a4763a70d3212e8d8b1953099173e3e19e9ee6cf602709b191f9cc2ed3`.
The k attempt stopped before training because the trainer virtualenv's older
vLLM did not register Nanbeige. The retry explicitly selects the image inference
executable at verified version `0.26.1rc1.dev1235+g62f6de733`, using the existing
injected serving factory/lifecycle. Trainer dependencies stay isolated; no
remote-code trust bypass, model substitution or framework API change is used.
The version and executable are retained in the training selection. Both judges
use this same runtime. Four exact-prompt audit regressions pass, including
tampered prompt, digest and verdict rejection; all eight import contracts pass.

Implementation files: `scripts/qualification/episode_reward_profile.py`,
`automationbench_native_judges.py`, and `audit_structured_toy.py`; external
`/home/hammad/projects/verifiers-environments-turns/environments/automationbench_v1/src/automationbench_v1/judge.py`.
Use fresh output directories on retries and retain failed assessments. Older
turn-based attempt `posttrain-gdpo-trl-gemma-r12-j` completed five updates but
failed its prompt reconstruction audit; it does not close this checklist.

- [x] 2026-09-07: Fixed silently discarded non-OK renderer tool calls in both TRL and veRL through shared `packages/train/src/posttrain/train/policy_messages.py`. Rejected spans are decoded from original sampled IDs into non-executable message content and carry typed `posttrain.rejected_tool_call` provider state with parser status and exact token span; accepted tool calls and reasoning retain separate channels. Missing span provenance fails explicitly. Replayed the first three affected turns from `training-gdpo-trl-gemma-e` with the installed Qwen renderer/tokenizer: all attempts preserved; thinking prompt and reasoning-channel retention verified.
- [x] 2026-09-07: Enabled policy thinking in `scripts/qualification/automationbench_native_judges.py`, increased per-turn generation to 2,048 tokens and reduced prompt budget to 6,144 within the existing 8,192 total limit. Gemma judge thinking stays enabled. This changes the next selection, not the already-exited run.
- [x] 2026-09-07: Managed run `posttrain-gdpo-trl-gemma-r12-h` completed five optimizer steps with policy and judge thinking enabled, lossless JSON-minified tool observations and the 8,192/12,288/6,144/18,432 rollout-input-output-context budget contract. It exposed a second semantic defect rather than qualifying the scorer: the plugin asked for one coarse quality scalar and did not retain five rubric dimensions; a rejected calendar call was therefore rated 1.0 because parser failure status was unavailable to the judge. Preserve this diagnostic run and do not call its semantic rewards qualified.
- [x] 2026-09-07: Changed the task-owned AutomationBench judge schema to return five explicit per-turn dimension scores. The plugin alone computes their arithmetic mean as the existing `quality` component; GDPO, reward projection and Verifiers framework contracts remain dimension-agnostic. Independent audit code now verifies raw dimensions, the stored evidence panel and scalar reduction agree exactly, and reports within-group quality variation.
- [ ] Run `posttrain-gdpo-trl-gemma-r12-i`, the fresh five-update thinking-enabled package qualification with explicit dimensions and rejected-call provider state; require exact judge-input replay, scalar-reduction parity, advantage parity, a nonzero learning signal, export reload and cleanup before accepting it.
- [x] 2026-09-06: User approved the inference/judge/orchestration boundary; inspected the existing serving and native-judge code and extended the same plan. No R12 implementation or GPU jobs launched.
- [x] R12.1: Frozen Nanbeige target/draft/runtime identities and qualified standalone standard, TurboQuant, and DSpark inference on `carbonteq-ai-workstation.lan` (NVIDIA RTX PRO 6000 Blackwell, 96 GB); retained the measured DSpark+TurboQuant incompatibility as an explicit unsupported capability for this runtime revision.
- [x] R12.2: Implement generic named managed/attached inference dependencies, per-service isolation, explicit resource placement and safe sharing; migrate the native-judge helper.
- [x] R12.3: Bind task-owned judge plugins to resolved connections without duplicated selection or acceleration logic; test failures, credentials and artifact isolation.
- [ ] R12.4: Compare Nanbeige and the pinned Gemma 4 12B judge on frozen, reviewed trajectories; qualify DSpark separately at inference level and retain all failed assessments.
- [ ] R12.5: Connect a qualified judge profile to real training, preserve deterministic correctness gates, and finish existing release prerequisites.
- [x] 2026-09-06: Pinned Nanbeige target `3384e426066d1a49c3aea90a7190b81260a6533f`, draft `5f12c792dabbfcdc4a0cf504e75f7216706d9590`, and candidate vLLM branch HEAD `62f6de733d7ae63b759329993bc209e67afdf431`; added the target model/renderer contract and immutable host-materialized DSpark draft configuration. These are source identities, not runtime qualification.
- [x] 2026-09-06: Added explicit Nanbeige judge inference profiles for standard, TurboQuant, and DSpark and allowed an immutable draft checkpoint to be materialized by exact Hub revision or supplied as an absolute runtime path. A composed profile was initially represented, then removed after live evidence proved that this runtime cannot combine the two accelerators. Focused common/catalog/serve validation passes with Ruff, scoped Pyright, and import contracts clean.
- [x] 2026-09-06: User clarified that “RTX 9000” meant the RTX PRO 6000 Blackwell worker. Live inspection resolved the exact host and 98,304 MiB dstack resource. A prior dstack task retained the sole block after its main runtime exited because leaked environment-service children kept the container alive. Graceful cleanup completed with the provider run terminal as `stopped_by_user`, the worker idle, and mounted run artifacts preserved.
- [x] 2026-09-06: Added named managed/attached service resolution with readiness/model checks, per-service workspaces, reverse-order owned cleanup, non-owned attachment semantics and non-secret evidence. Standard SAMPO/GDPO/CAPO composition can map multiple judge plugins to one explicit service name.
- [x] 2026-09-06: Isolated policy/judge/draft artifact materialization by seat-specific input names and fixed speculative benchmark labels, including composed `mtp-turboquant` and native `dspark` labels. Focused result: 98 tests passed; jobs-only result after composition migration: 30 passed; scoped Pyright and all eight import contracts pass.
- [x] 2026-09-06: Published Nanbeige candidate runtime `registry.carbonteq.com/carbonteq/posttrain/nanbeige-vllm@sha256:bfbeee151fb59c687f7e27d5e5e287716a7c1c6dd58ebf0a4ffe99b13702a3aa`. Initial dstack attempt `posttrain-nanbeige-r12-matrix` failed before GPU/model work because the vendor image has `python3` but no `python` alias; it produced no matrix directory and released the worker. Attempts `matrix-a` through `matrix-c` retained the composed failure, raw response shape and output-budget corrections. Final task `posttrain-nanbeige-r12-matrix-d`, evidence `matrix-4`, passes standard, TurboQuant and DSpark thinking-off/direct plus thinking-on/parsed-final probes. All three return GPU memory to the 303 MiB baseline. The composed profile is retained as unsupported, not counted as a passing runtime.
- [x] 2026-09-06: Provisional plugin-owned comparisons ran the same five cases twice against exact Nanbeige 3B and Gemma 4 12B selections. With judge thinking disabled both scored 8/10 and missed the same intentionally wrong timezone arithmetic twice, so neither direct profile qualifies. With thinking enabled both passed 10/10. Nanbeige used 31,009 completion/29,698 reasoning tokens; Gemma used 5,120 completion/4,127 reasoning tokens. Gemma is the provisional five-step integration choice because it produced the same accepted labels with about one-sixth the generated tokens. This engineering fixture remains separate from the planned held-out comparison, and the comparison used pre-lint-fix plugin blob `59dd00e20b42ee83d2e76f845d339185e51b3bc7`.
- [x] 2026-09-06: Migrated all six standalone environment packages to immutable Verifiers v0.3.1 commit `8e8f3042481c0996a58c3de0f86d55406725b6c1`, including native loader/trace/reward contracts and Math Python's task-scoped toolset construction. `automationbench-v1` is now a 0.3.0 release candidate. Every package passes Ruff, format, Pyright and its tests; the six wheels build, install together, and activate through the native loader. The current candidate AutomationBench wheel SHA-256 is `370a43b88e2a925714cc656a45d152f1d4f59b1d021925add14d8534e4e70782`; publication still requires an immutable repository commit and clean-clone rebuild.
- [x] 2026-09-06: The first managed Gemma/GDPO attempt stopped before optimizer step one after bounded admission retained four episodes. One valid episode proved four distinct turn ratings; three failed safely because an empty assistant action elicited abstention or a 12-turn trajectory exhausted judge output length. No missing assessment became zero. The task-owned judge now sends a bounded, source-digest-preserving projection of bulky tool observations while leaving native traces untouched, and its rubric explicitly makes empty assistant actions assessable. The small-policy rerun also selects AutomationBench's task-filtered toolset and colocated vLLM policy generation so required behavior log-probabilities are present.
- [x] 2026-09-06: Extended judge calibration to accept an external frozen fixture, validate complete per-turn expectations for multi-turn trajectories, and retain the canonical fixture SHA-256 plus judge token usage. Added `automationbench-turn-quality-candidate-v1`: eight cases, 17 assistant turns, two repetitions, digest `d28606a76e988db541292b9e0f4e48c01a034633e67ce1ba220675c48bf8c340`. It is a candidate for human review, not accepted held-out evidence.

- [x] 2026-09-06: User approved deterministic scoring for algorithm qualification. The failed 2B judge calibration is a separate scoring-profile gate, not a reason to pause controlled optimizer tests.
- [x] Added qualification-only synthetic turn fixture and dual-backend AutomationBench runner; 26 fixture/turn-contract tests pass, including original mask holes and complete turn coverage.
- [x] Rechecked 77 framework fixture/turn/veRL adapter tests and 57 fork numerical/recovery/replay tests; full Ruff, Pyright and all eight import contracts pass.
- [x] Independently audited attempt b inputs: eight trajectories, 17 assistant turns, all four synthetic cases, with exact native Episode token/mask/logprob parity. This is input evidence only, not a completed optimizer qualification.
- [x] veRL GDPO `gdpo-deterministic-d` and CAPO `capo-deterministic-a` each completed five updates. Decimal aggregate-advantage audits pass; native provenance passes for 10/12 traces and 24/44 turns respectively. CAPO retained a rejected max-turns group.
- [x] Fixed generic veRL LoRA export target collisions; 17 merger tests pass. Re-exported unchanged step-5 checkpoints into separate directories and verified every adapter key/tensor plus CPU generation for both algorithms. Original exports remain retained as unqualified.
- [ ] Run and independently audit five deterministic updates on each GDPO/CAPO backend, including checkpoint/resume and export/reload. A runner or fixture test does not close a live cell.

### Revision 7 — retained execution checklist

This checklist and the Revision 7 continuation supersede stale open items in
the historical record below. Revisions 11/12 above govern current results and
the new inference/judge work. Prototype results do not close production gates.

- [x] 2026-09-06: Agreed on turn rewards first, with task-owned rubrics in custom Verifiers judge/scoring plugins; extended this plan without changing code or publishing artifacts.
- [x] TRL GDPO 0.8B/2B toy completed five updates; independent audit confirms original-token mask alignment, Decimal advantage-mean parity, adapter change and exported-model reload.
- [x] TRL CAPO toy completed five updates and exported an adapter.
- [x] Independently audited CAPO and reloaded its export: five nonzero-gradient updates, Decimal advantage-mean parity, exact masks and adapter changes. Native task success is 0/10 admitted attempts, not evidence of improvement.
- [x] Retained both toy bundles outside `/tmp` under `outputs/qualification/automationbench-2026-09-06/`; hashes and local artifact paths are in `docs/tooling/trl/evidence/2026-09-06/automationbench-retention.json`. Remote publication remains open.
- [x] Narrowly amended the canonical baseline and ADR 0018 before code changes.
- [x] Added Episode-to-SFT projection with episode lineage, non-policy-agent exclusion and streaming JSONL reads; native v0.3.1 suite passes in its isolated locked runtime. Legacy runtime reader regressions pass.
- [x] Shared fact projection now recognizes modern explicit `ok=false` even without saved exceptions; calculator version advanced to v4.
- [x] Added exact native assistant-turn IDs, original token/mask provenance validation, complete generic assessments, explicit GDPO mean/sum reductions, SAMPO direct-turn selection and CAPO error-turn-to-token union projection. No rubric names were added to algorithms.
- [x] Re-ran GDPO and CAPO with actual 0.8B policy / 2B judge turn ratings for five updates each. Both independent Decimal/mask audits and exported-adapter reloads passed. These still use the legacy-selected Verifiers runtime and TRL candidate, not final modern packages.
- [x] Created isolated native Verifiers client-injection candidate; three real local harness tests cover static, elastic and server interception and adapter closure. Modern framework train integration preserves exact sampled IDs and native Episode artifacts.
- [x] External AutomationBench migration passes eleven tests, including optional native judge-plugin discovery and bounded invalid-output retry. Real modern bridge integration executes a task tool, receives native reward 1.0 and transports two distinct SAMPO turn rewards while excluding observations.
- [x] Modern evaluation uses native nested configuration and single-argument runner; real loopback endpoint integration preserves an Episode and emits a lineage-linked trace view with one-attempt population accounting.
- [x] Modern native-Episode GDPO and CAPO toys completed five optimizer updates each, with Decimal/mask audits and exported-adapter reloads. These use the unpublished client-injection/environment source overlays and local prototype judge, not final production composition.
- [x] Added `posttrain.jobs.bind_native_judges`: the host binds explicit inference selections to existing native judge names and owns startup/cleanup; plugin model/sampling mismatch fails before loading models. Recovery identity retains engine/model/renderer/target settings, never API-key values. Structured job factories accept optional named judge inference seats. Six lifecycle/preflight tests pass.
- [x] Native and derived bridge JSONL now share cross-process append locking. Two native AutomationBench integrations cover both limited and default Zapier toolsets.
- [ ] Complete mixed-version native replay, live train/eval API migration and external AutomationBench compatibility; then switch qualified runtime pins. Reader support alone does not close R7.2.
- [ ] Implement generic turn evidence and custom judge/scoring-plugin composition with injected inference, detached planning, bounded failures and immutable provenance.
- [ ] Qualify SAMPO explicit-turn rewards, CAPO turn-error projection and GDPO declared turn-to-trajectory reduction without changing their algorithm contracts.
- [ ] Complete four GDPO/CAPO backend cells, SAMPO regressions, judge calibration and installed-package/OCI qualification.
- [ ] Publish remaining fork deltas, qualify development artifacts, promote unchanged assets, update stable pins and release with evidence-backed support claims.

### Historical implementation record — Revisions 1–6

- [x] User reduced the toy profile from 2B policy / Gemma 12B judge to 0.8B policy / 2B judge. Select catalog-pinned Qwen3.5 checkpoints for this toy; the larger pilot remains a later gate.
- [x] Verified both pinned model snapshots exist locally. Live fleet inspection: 8 GB `pop-os.lan` idle; 96 GB `carbonteq-ai-workstation.lan` busy. Do not interrupt the busy worker or assume its capacity is available.
- [x] Fixed actual-job package and CLI runtime routing for `train.gdpo` and `train.capo`; 18 focused package/CLI tests passed.
- [x] User confirmed AutomationBench. The selected native adapter is `automationbench_v1` at `b7bcb591facfcd2b073802f6d7496b24ab9c479e`, using the catalog's Zapier simple-task selection.
- [x] Added the local qualification runner `scripts/qualification/automationbench_structured_toy.py` and independent raw-evidence auditor `scripts/qualification/audit_structured_toy.py`. Native partial credit and binary task success remain separate from judge annotations. The projection pins the actual scorer digest, including model revision, rubric, generation and retry semantics.
- [ ] Execute and audit five optimizer steps per selected algorithm/backend, recording reward components, critique outputs, exact masks, advantages, gradients and parameter changes. A zero-signal batch is diagnostic evidence, not permission to invent rewards.

- [x] Standard `train/gdpo@1` and `train/capo@1` jobs, work kinds and explicit reward-projection catalog decoder implemented.
- [x] Real two-rank Gloo admission passes: one-rank annotation failure replaces the complete shared group before global normalization.
- [x] Installed TRL post2 passes masked CUDA GDPO and CAPO fixtures (two nonzero-gradient updates each, beta=0.1), matching checkpoint resume and exported inference. Reproducer and JSON receipts are retained in the consumer repository; live bridge/judge, LoRA, vLLM and pilot gates remain open.
- [x] Recovery guards retain projection/environment/numerical-setting identities in TRL and native veRL V1 checkpoints. An external scorer changed behind an unchanged environment revision remains outside this guarantee; freeze its actual selection before live qualification.
- [x] veRL numerical/recovery/replay/V1 transport suite: 69 passed. Main train/jobs/work suite in upgraded dependency environment: 324 passed, 3 skipped, 1 intentionally deselected old-pin assertion. Eight import boundaries kept.

- [x] Candidate runtime build `34007498425` succeeded; authentic generated locks and image receipts restored to `/tmp/posttrain-fork-upgrade.ouJ7s4`. Runtime/release tests: 120 passed, 1 skipped. These are candidate artifacts, not stable promotion.
- [x] Explicit serializable reward projections, native trace identity stamping before enrichment, asynchronous enrichment, complete-group bounded TRL admission, and veRL V1 structured metadata transport implemented locally.
- [x] veRL structured GDPO/CAPO numerical profiles and separate ordinary token-clipped loss/unclipped k3 KL added without replacing legacy behavior. Eleven independent numerical/gradient tests pass.
- [ ] Finish standard-job/catalog, checkpoint identity, distributed admission and native bridge regression qualification. New veRL source remains unpublished; main dependency pins remain unchanged.
- [ ] Resolve the frozen CAPO judge model revision/inference binding with the user; no endpoint or immutable 12B model is specified by the pilot. This gates the real judge run, not deterministic implementation.

- [x] 2026-09-06: User prioritized upgrading underlying TRL/veRL versions while retaining all maintained features; saved tested corrections as TRL `7fe16760` and veRL `c47f8e6a` before integration.
- [x] Integrate TRL v1.12.0 and veRL v0.9.0 in isolated candidate branches; resolve conflicts by behavior and run retained-feature regressions.
- [x] 2026-09-06: Integrated, tested, committed and pushed both upstream-base upgrades: TRL `6a5532e2f51e4e1cdc8a891582514a50f68a775a` (140 source/45 installed-wheel checks) and veRL `cec7e74c361bb973b641db8dfbb75a5544c33139` (151 CPU source/63 installed-wheel checks). Immutable GitHub candidate releases are retained with verified hashes.
- [x] Development-channel publication runs `34006220244` (TRL) and `34006221953` (veRL) succeeded, including clean-install/readback receipts.
- [x] CUDA recovery qualification exposed an upstream IW-OPD dataloader bug. Published corrected TRL `1.12.0.post2` at `95a787b6c04f91a5d485fd827d31b1e1fb67ae8e`; development publisher `34007394648` succeeded. Installed wheel passes two-update CUDA train/resume/export with accumulation 1 and 2. Twelve map/streaming recovery regressions pass.
- [ ] Clean framework candidate `codex/trl-verl-stable-base-upgrade` at `/tmp/posttrain-fork-upgrade.ouJ7s4`: new pins and veRL lock/constraints authored; main workspace GDPO/CAPO changes preserved. Complete runtime lock/build and GPU gates before merging candidate pins into main.
- [ ] Commit/push upgraded forks, retain immutable release assets, publish to the development index, qualify candidate runtime locks, then promote unchanged assets and update stable consumer pins.

- [x] 2026-09-06: Inspected framework contracts, adapters, and immutable backend source revisions.
- [x] 2026-09-06: Checked published method descriptions and official reference repository identities.
- [x] 2026-09-06: Ran focused existing tests: 57 passed, 1 skipped; Verifiers module unavailable.
- [x] 2026-09-06: Defined implementation slices, numerical contracts, and four-cell qualification gates.
- [x] 2026-09-06: User authorized end-to-end implementation, including correctness/precision improvements; amended the canonical baseline for the new operations and reward/scorer contracts.
- [ ] Complete reference-code/permission review and freeze golden numerical fixtures.
- [ ] Implement shared reward transport, identities, and validation.
- [x] 2026-09-06: Added structured evidence values, critique-mask validation, and complete-population GDPO/CAPO numerical constructors; 20 focused tests passed, including Decimal-oracle, mask, permutation, and missing-evidence cases.
- [ ] Implement GDPO on both backends and prove distributed parity.
- [ ] Implement CAPO judging/alignment and both backend paths.
- [ ] Qualify real Verifiers jobs, optimizer behavior, checkpoint/reload, and regressions.
- [ ] Publish required fork changes, update immutable pins, and qualify final packaged artifacts.
- [x] 2026-09-06: Audited newer stable releases against the maintained deltas; added isolated TRL Liger window-normalization and veRL REINFORCE++ observation-credit fixes. Nineteen TRL CPU tests (including native Liger and two-rank Gloo gradients) and 28 veRL native CPU tests pass. Candidate publication and GPU gates remain open.

## Context and Orientation

The inventory below records the initial inspection. Use the Revision 7
continuation for current checkout identities and remaining work; older source
descriptions are historical evidence, not today's dependency selection.

An episode is a complete task attempt. A rollout record contains its generated
tokens, actions, observations, and reward evidence. A prompt group contains
multiple attempts generated for one occurrence of a task in a learning batch.
Repeated occurrences of the same task are different groups. An advantage is the
numerical learning signal used in the policy loss. A token loss mask identifies
sampled assistant tokens and excludes tool observations, prompt context, and
padding. A critique's evidence span and the token positions it affects are
different representations and require an explicit mapping.

The framework was clean at inspection, HEAD
46455ea975bde00b61058d37cc5171ca206a2bfc. The authoritative package boundaries and
operations are docs/post-training/README.md, 04-framework.md, and 05-apis.md.
The baseline is frozen; this plan proposes amendments, not an implicit unfreeze.
Reward meanings remain environment/project-owned. Algorithms and their
normalization, masking, clipping, and backend realization belong in train.

Current source evidence:

| Surface | Observed implementation | Consequence |
| --- | --- | --- |
| online_rl.py | EnvironmentRollout has one scalar reward, env_mask, and optional AgenticTurn records | Add component and token-credit transport without changing legacy scalar semantics |
| integrations/verifiers.py::_project | Reads trace.reward, requires one branch, retains sampled token masks | Keep one trainable branch initially; components in trace facts do not automatically reach training |
| integrations/verifiers.py::_agentic_turns | Creates contiguous assistant spans and observation keys; no step_reward assignment | SAMPO span support is not CAPO token-credit ingestion |
| backends/trl/policy_optimization.py | One _bridge_reward callback; precomputed advantages enabled only for SAMPO; sequence ratio for SAMPO | GDPO needs separate reward channels; CAPO needs token-ratio loss, not SAMPO's sequence-ratio setting |
| pinned TRL GRPOTrainer | normalize_then_sum aggregation and precomputed token advantages exist | Reuse these seams; no need to implement a replacement trainer wholesale |
| backends/verl/contracts.py | Estimator literals limited to grpo/sampo | Public selection and isolated manifest reject GDPO/CAPO today |
| pinned veRL core_algos.py | GDPO and SAMPO estimators already registered | GDPO work includes connecting existing machinery, not merely adding an estimator |
| pinned veRL V1 TrainerBase::_compute_advantage | Retrieves extra_fields specifically for SAMPO | GDPO component metadata is not materialized through this path; its estimator also expects batch fields absent from this fetch |
| pinned veRL V1 utils | Session-final special handling applies only to GRPO | Define prompt-group/session identity for the new algorithms; do not infer it from transport-key formatting |

Key framework paths under packages/train/src/posttrain/train:
online_rl.py, profiles.py, requests.py, catalog_schema.py, api.py, results.py,
verifiers_requests.py, integrations/verifiers.py, sampo_advantages.py,
backends/trl/policy_optimization.py, backends/verl/contracts.py, backends/verl/launcher.py,
backends/verl/agent_loop.py, backends/verl/worker.py, backends/verl/reward_fields.py,
and grpo_observations.py. Standard jobs live in
packages/jobs/src/posttrain/jobs/definitions.py. Catalog defaults live in
packages/catalog/src/posttrain/catalog/base/.

The selected TRL is 1.9.2.post11, source
69cf80a7319079ec5523841553467e119ebc1cec, from packages/train/pyproject.toml.
The local /home/hammad/projects/trl HEAD is
c9af78c1c2ea04ad271e95b26b93dfadf8b9fca1, so inspection used git show of the pin.
Do not edit the unrelated current branch to implement this plan.

The selected veRL source is 808923d487aa2c524fda02cf5289110541b4221f,
carbonteq-v0.9.0.dev2. Its authority is
packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/verl-py313/profile.toml
and release/pyproject.toml plus release/uv.lock. The selected runtime separates
backend and control environments. The adjacent verl-upstream checkout is at
a35908ca3c9632859c58d6a2855d858918ae21dc with substantial unrelated dirty work.
Inspect pinned objects; create an isolated worktree for implementation after
reading that fork's AGENTS.md and CARBONTEQ_FORK.md. Never absorb that dirty tree.

## Published contracts and references

[GDPO paper](https://arxiv.org/html/2601.05242v1) and
[official reference](https://github.com/NVlabs/GDPO/tree/4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115)
define component-wise prompt-group normalization, weighted aggregation, and a
final batch normalization. Its reference implementations differ on whether the
last stage weights rollouts or response tokens.

[CAPO paper](https://arxiv.org/html/2508.02298v1) and
[official reference](https://github.com/andyclsr/CAPO/tree/98b25f8c67b5fa65e1158ecc5d51b03272b5edee)
define critique-derived erroneous-step sets, optional voting, token rewards
R_it = W_outcome * outcome_i - W_process * error_it, and normalization over
sampled tokens in the response group. The objective uses token probability
ratios and per-response token-mean aggregation. Outcome-dominant weights are
part of the initial reference configuration.

The reference repository HEADs above were resolved read-only on 2026-09-06.
Paper descriptions and READMEs were inspected, not every reference implementation
file. Milestone 1 must reconcile the exact implementation details, including
variance correction, epsilon placement, voting ties, segmentation, and KL.
Do not silently label a changed recipe as a reproduction.

The GDPO repository README declares NVIDIA Source Code License-NC. Do not vendor
its code into the framework without checking permission and compatibility.
Prefer reviewed reuse of the already selected backend implementations or an
independent implementation from the mathematical specification. Check CAPO's
license and copied-code provenance before reuse as well. This is a release gate,
not a recommendation to change project licensing.

## Interfaces and Dependencies

### Revision 12 service dependency contract


Keep `ModelVariant`, `InferenceBinding`, `ExecutionTarget` and native Verifiers
judge configuration as the existing selections. Do not add a scorer catalog
family, a DSpark judge class, or decoding fields to algorithm settings.

In `packages/jobs/src/posttrain/jobs/inference_services.py` introduce a host-level
`bind_inference_services(context, services, *, provisioner)` context manager.
Its input is a mapping of explicit service names to a managed service request or
an attached endpoint request. Both carry the expected inference selection; an
attached request also carries a scoped credential reference and endpoint address.
Its output maps those names to resolved connections with inference identity,
endpoint, readiness evidence and lifecycle ownership. These are composition
values, not new product primitives. Reuse existing endpoint and provider types
where sufficient rather than duplicating them. The provisioner is the host's
mechanism for starting a selected runtime, not part of the judge plugin.

In `native_judges.py`, map plugin name to service name separately from service
provisioning. Two plugins may share one named service. A multi-client plugin may
declare multiple named dependencies, but its native client interface must be
qualified before claiming that route supported. Launch once per explicit service
name, not once per rubric. Do not silently deduplicate services based on matching
URLs or model names. Continue using `bind_native_judges` as a compatibility
wrapper until its existing callers and tests migrate in this plan's release.

Judge plugins retain rubric, context scope, schema, bounded assessment policy
and named output annotations. Model/revision/connection and effective sampling
are resolved once from the selected service and materialized into native config;
reject conflicting user declarations before launch. Distinct rubric request
budgets may share a service only within its declared serving limits. Algorithms
consume the existing validated reward projection and never inspect DSpark/MTP.

Propose separate typed operations train.gdpo and train.capo, following train.sampo,
rather than overloading SAMPO settings or accepting arbitrary trainer kwargs.
Add GDPOSettings/GDPORequest and CAPOSettings/CAPORequest, their schema variants,
public exports, result technique identities, and standard job definitions.
These names are proposed until the baseline amendment is accepted.

GDPOSettings declares ordered component names, weights, epsilon, final
normalization population, group size, token-ratio clipping, KL policy, and loop
settings. CAPOSettings declares outcome/process weights, normalization epsilon,
group size, token-ratio clipping, KL policy, and loop settings. A separate
versioned scorer selection holds judge model/inference, rubric/projection
identity, segmentation, critique count/voting, retries, timeouts, and context
limits. Model inference is an execution binding, not an algorithm setting.
Preserve training-target versus rollout-target independence.

Extend the backend-neutral rollout contract additively with structured,
serializable reward evidence. Suggested implementation files are reward_evidence.py,
gdpo_advantages.py, and capo_advantages.py alongside online_rl.py. Proposed logical
fields are prompt_group_id, rollout_id, trace/branch identity, ordered named
component values/status, and sampled-token-aligned process evidence. Store
critic texts and annotations as derived native-trace evidence with immutable
references; do not create another trajectory database.

Group identity must identify the task occurrence and generation batch, while
rollout identity distinguishes responses within that group. Preserve both across
rank sharding, asynchronous completion order, candidate refill, and TransferQueue.
Never group by task name alone.

Each value has valid, inapplicable, abstained, or failed status. The initial
GDPO contract requires all selected components valid for every admitted rollout.
The initial CAPO contract requires a valid outcome and resolved critique.
Reject/replace incomplete groups with bounded attempts; never use nansum or
nan_to_num to turn missing required evidence into a successful score. An explicit
zero score remains valid data.

CAPO segmentation produces a deterministic step-to-original-token mapping.
Keep exact sampled policy tokens; do not regenerate or retokenize the response
and assume offsets survived. Judge text can be rendered with another tokenizer.
Store any text reconstruction and character/token alignment version. Handle
Unicode, repeated text, special tokens, tool-call serialization, multiple errors,
and overlaps. An error mask is the union of affected sampled positions, not a
count of how often a critic mentioned a span. No process credit applies to tools'
observation tokens or padding.

Versioned project/environment scorer composition owns the judge prompt and
deterministic outcome checks. Reuse Verifiers enrichers where their contract is
sufficient, but provide serializable scorer selection and generic validated
projection rather than embedding an unpicklable project closure in a launch
manifest. Scoring executes before group admission/advantage construction; it may
not mutate the sampled trajectory or overwrite verified task success.
train, eval, and serve must not import one another to host the judge.

Data Designer is a separate research-project dependency. It is not added to any
framework or backend environment. Existing policy/judge SFT and dataset support
remain the precursor capability specified in the pilot companion; this work
does not add veRL SFT or a numerical reward-model head.

## Algorithm semantics selected for qualification

### GDPO

The proposed default is rollout-weighted final normalization, consistent with
the inspected TRL path and paper-level response notation. For each group g and
component k, normalize its response values using that group's sample standard
deviation and a declared epsilon. Apply component weights after normalization.
Normalize the resulting scalar advantages once over the complete admitted
logical batch, counting each rollout once. Then broadcast to eligible sampled
tokens. Constant components contribute zero; entirely constant batches remain
finite and do not justify claiming a useful update.

Use epsilon 1e-4 for the initial TRL-aligned reference profile; freeze explicit
sample-variance and epsilon placement fixtures. This is a numerical profile
choice, not a universal optimal value. Preserve a separately identified
token-weighted final-normalization option only if reproducing the veRL/reference
variant is required. Both backends must implement the same selected option.

The pinned veRL GDPO path currently uses response-mask-weighted whitening after
broadcast. Unequal response lengths therefore make it different from TRL's
rollout-wise final normalization. It also reconstructs terminal positions from
prompts/attention_mask. Extend it to consume validated component vectors
directly, avoiding an unnecessary terminal-index reconstruction and a silent
single-reward fallback. Do not change existing fork defaults for other consumers
without a versioned option.

GDPO is not inherently a local-credit method. Aggregating local scores to a
trajectory component is a documented project reduction, not token attribution.

### CAPO

Start with binary verified outcomes, W_outcome=2, W_process=1, and token rewards
as stated above. Continuous AutomationBench partial credit is a later declared
adaptation: the outcome-priority argument for binary rewards cannot simply be
assumed for fractional outcomes. The first real method check should use a pinned
reasoning environment with deterministic final-answer verification.

Normalize over eligible sampled tokens from all responses in one prompt group,
not same-position tokens and not an unrelated collection of prompts. Longer
responses contribute more tokens to these statistics; preserve and test that
method choice. The normalization population excludes tools, padding, rejected
responses, and truncated samples selected for replacement. Fix sample variance
and epsilon in Milestone 1 against the chosen reference version; zero-variance
or fewer-than-two-valid-token groups must return a defined finite result or a
typed invalid-group error, identically across backends.

Use the standard token-ratio clipped policy loss, averaged over sampled tokens
within each response and then over responses. Do not reuse SAMPO's sequence
importance-ratio objective, reward-to-go computation, or DAPO token-global loss
merely because their plumbing accepts token tensors. Keep KL outside reward
normalization for the first profile and declare the same estimator/reduction on
both backends. Test beta=0 and a positive beta; do not double-apply KL.

First implement one deterministic critique; then qualify configurable multiple
critiques, intersection, and the reference's majority threshold (including
even-count ties). Invalid or timed-out critiques cannot masquerade as an empty
error set. Cache only by immutable trace, judge, rubric, projection, and sampling
identity. Stochastic samples retain individual identity and count toward cost.

CAPO on segmented reasoning is the reference capability. Native multi-turn
thinking/action traces exercise a separately named projection profile with
prefix-versus-retrospective assessment recorded. Recovery, harmless detours,
and rejected hypotheses must be audited before claiming useful agent training.
Do not force the general-purpose policy into numbered reasoning merely to
simplify alignment.

## Plan of Work

### Revision 12 continuation — inference services and judge comparison


This continuation is production-intent work preceded by isolated qualification,
not permission to replace the working trainer environment with a vendor fork.
The existing frozen ownership baseline already places decoding in inference,
composition in the host, and rubrics in Verifiers. No product-baseline meaning
change is needed for that separation. Before implementing new public endpoint
attachment or lifecycle contracts, document their concrete names and ownership
in canonical 04/05/06; if implementation requires new product semantics, record
and amend that narrow baseline first. Do not silently introduce a remote
scheduler, scorer store, or new job kind through this plan.

#### R12.1 — qualify the inference dependency independently


The linked repository `Nanbeige/Nanbeige4.2-3B-DSpark` contains draft weights, not
the standalone judge. The target is `Nanbeige/Nanbeige4.2-3B`, whose config names
`NanbeigeForCausalLM`. Its model card documents Nanbeige's vLLM `nanbeige42` and
SGLang `nbg42` branches. Start with vLLM because the existing managed adapter is
vLLM-specific; SGLang is an explicitly separate fallback qualification, not an
automatic runtime switch. Resolve branch and both model revisions to immutable
commits before installing anything. Review required custom code, build and
license constraints; retain runtime image/wheel hashes and source provenance.
The documented DSpark example uses method `dspark` and seven speculative tokens;
treat that as an initial inference profile, not a judge API default.

Extend `packages/serve/src/posttrain/serve/profiles/base.py` and
`backends/vllm/bindings.py` to represent and validate an immutable draft artifact
as part of backend-owned speculative configuration. Materialize target and draft
independently into the selected serving runtime. Correct the benchmark adapter's
current classification of every speculative method as `mtp`. Add the actual
Nanbeige model/renderer selection in `packages/catalog/src/posttrain/catalog/base`
and qualify its reasoning parser and chat template rather than reusing Qwen's
protocol by assumption. The judge must not need live tool execution; model tool
calling is not a prerequisite for rating tool-call text in supplied trajectories.

Run target-only inference first, then the same target with DSpark. Verify output
schema support with reasoning enabled/disabled as selected, context accounting,
stop conditions, timeout/cancellation and actual draft activation. Benchmark
representative judge-shaped prompts, including prefill-heavy long trajectories
and short JSON answers, with warm/cold measurements separated. Measure valid
responses, latency distribution, throughput, memory and draft acceptance where
available. Acceleration gains and distributional equivalence are not assumed;
same seeds do not promise identical outputs across decoding implementations.
Keep the target-only binding available if DSpark fails qualification. No judge
or trainer code changes are required to toggle acceleration.

#### R12.2 — resolve and own inference services explicitly


Implement the service contract above in jobs and the existing serve adapter.
`serve.launch` currently starts `vllm` beside `sys.executable`; `ExecutionTarget`
does not cause that subprocess to run remotely. A managed launch must execute
inside the provisioned runtime with its explicit GPU assignment. Validate local
placement against the granted resources, not merely a host label. For the first
remote comparison, provision the serving workload through the existing execution
path and attach the comparison job to its endpoint. If the existing provider
cannot supply service readiness/lifecycle handles, keep deployment and comparison
explicitly separate rather than inventing a hidden SSH scheduler. Inspect
`packages/execution`, `packages/execution-dstack` and
`apps/cli/src/posttrain_cli/execution_planning.py` before any provider edits;
integrated remote-managed lifecycle is not qualified by a local context manager.

Use a distinct run/service-instance directory for logs, templates and manifests;
current `serve.launch` reuses `vllm-server.log` and `chat-template.jinja` within a
workspace. Separate bind address from consumer-reachable address. Allocate ports
and credentials within a service instance, redact secrets from evidence, and do
not mutate process-wide credentials for overlapping jobs. Prefer client-scoped
or child-process-scoped injection; if native Verifiers only supports environment
credential references, isolate the consumer runtime rather than sharing mutable
global state. A model alias from `/models` is not proof of its checkpoint SHA:
retain deployment provenance, or label an external endpoint's revision unverified
and reject it from reproducible qualification.

The host owns resource reservation, service health, queues and shutdown. Budget
weights, draft weights, KV cache and engine overhead together. Do not interpret
independent GPU utilization fractions as a reservation or start two profiles at
their standalone 80% defaults on one GPU. Initially compare models sequentially
on the verified target using frozen traces, without a policy trainer competing
for memory. Managed cleanup drains/cancels pending calls and closes only owned
instances; attached endpoints are never stopped. Partial startup failure closes
previously started owned instances and retains diagnostics. Hard crashes require
provider cleanup/ownership fencing, not just Python `finally` blocks.

#### R12.3 — connect existing task-owned judge plugins


Update `packages/jobs/src/posttrain/jobs/definitions.py` to resolve inference
dependencies by seat and named service, then adapt native configs in
`native_judges.py`. Fix artifact selection before adding a second model: current
`_materialize_selected_model_variant` searches generic `model_weights` and
`model_adapter` input names. Judge and draft materialization must never reuse the
policy's checkpoint accidentally. Use explicit per-seat/per-artifact identities
and verify model role, repository/revision and digest. Add a regression with
distinct policy, judge and draft artifacts.

Keep the AutomationBench rubric and output parser in the external environment
plugin. It already calls a supplied native client and validates complete turn
coverage. Improve only generic binding or failure transport as necessary;
no Nanbeige/DSpark branches in the rubric. Readiness includes an authenticated
request with the selected JSON schema and reasoning configuration, not health
alone. Distinguish network failure, overload, timeout, output truncation, invalid
JSON/coverage, deliberate abstention and a valid low score. Retain attempts and
bounded retry budgets; never manufacture zero rewards, silently change the model,
or repeatedly retry deliberate abstention. Preserve original trajectory IDs and
tokens, and use distinct annotation namespaces for model comparisons.

#### R12.4 — compare reasoning-quality profiles without training confounds


Use the existing five-dimension rubric as task-owned policy; do not encode its
dimensions in orchestration or algorithms. Freeze a reviewed evaluation set of
native trajectories with target turn IDs, expected error marks, rating guidance
and content hashes before comparing outputs. Separate rubric-development examples
from held-out assessment. Include correct concise actions, plausible wrong
reasoning, tool failures, real corrections, redundant work and adversarial
instructions inside observations. Human-reviewed references establish the target;
the 12B model is a competitor, not ground truth. Prefix and retrospective context
are distinct profiles and must not be pooled. Supply the policy's reasoning as
quoted trajectory data; configure the judge's own thinking mode separately.

Compare Nanbeige and the exact `models/gemma4-12b-it@bf16` catalog selection after
verifying its immutable artifact and serving compatibility. Same examples,
rubric, schema and context policy are required; identical numeric token budgets
across different tokenizers do not imply identical semantic budgets. Freeze
adequate per-model budgets and report actual tokens and truncations. Report error
precision/recall, rating agreement, invalid/unavailable responses and repeated-call
stability separately from runtime cost. Retain paired examples and uncertainty;
do not declare a winner from the old ten-check engineering fixture. Before the
held-out run, record exact sample size, acceptance thresholds and comparison
rule in the frozen fixture manifest. Those choices remain an explicit gate,
not thresholds to adjust after seeing which model wins.

Extend `scripts/qualification/calibrate_automationbench_judge.py` for frozen
trace input and multiple endpoint profiles, or extract reusable qualification
helpers there without making scripts part of production algorithm semantics.
DSpark qualification belongs to R12.1; the judge comparison may consume an already
qualified accelerated binding but must not know its decoding implementation.
Only after a judge profile passes its declared quality gate run a bounded managed
training integration. Deterministic algorithm qualification remains independent.

#### R12.5 — repository sequencing and release


Framework HEAD at planning is `8f240ac5474ecaecb5f851751d52e37dbd0d3dee` with
existing dirty work. Current stable manifests still select vLLM 0.25.1, TRL
1.9.2.post11 and Verifiers `284a868d6a9022109b749710672a0460e8a996d4`;
candidate qualification did not update those pins. Build a separate serving
runtime under `packages/runtime-images` rather than overwriting trainer locks.
Resolve Nanbeige fork SHA and maintained repository ownership before source
changes or publication; neither is established by a moving vendor branch.

If native client changes are needed, use `/home/hammad/projects/verifiers-gdpo-capo`
at inspected HEAD `8e8f3042481c0996a58c3de0f86d55406725b6c1`; its publication
ownership is still unresolved. Task plugin changes belong in
`/home/hammad/projects/verifiers-environments-turns` at inspected HEAD
`b7bcb591facfcd2b073802f6d7496b24ab9c479e`. Recheck branch, diff and repository
instructions before edits. Publish tested generic fork changes first, then
environment packages, then exact consumer locks/images and framework selections.
Update each fork ledger with its `docs/tooling/<tool>/README.md` consumer page.
Preserve all existing unrelated edits. No new fork publication is authorized by
this planning turn. Existing algorithm resume, distributed, SAMPO, environment
migration and packaged release gates are not closed by judge comparison results.

Primary research inputs inspected on 2026-09-06 are the model card at
https://huggingface.co/Nanbeige/Nanbeige4.2-3B-DSpark, target config at
https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/config.json and vendor
runtime at https://github.com/Nanbeige/vllm/tree/nanbeige42. Their relevant
requirements are embedded above; resolve immutable revisions before execution.

### Revision 7 continuation — current implementation and release sequence

Execute this continuation over the existing Milestones 1–5 implementation.
Their numerical and backend gates remain applicable. This revision supersedes
within-turn semantic segmentation as an initial requirement and clarifies
scoring ownership. The initial release remains GDPO/CAPO on both backends with
native turn support and SAMPO regression qualification. SRPO is research input,
not another promised release algorithm: a future turn-based SRPO profile needs
explicit equations and independent qualification. The larger policy/12B judge
pilot is later work, not a gate for a narrowly qualified small-model release.

An assistant turn is one policy response in the native interaction, potentially
containing reasoning, visible text and tool calls. Tool observations are context,
not policy actions. A turn reward assesses that response; an advantage is the
algorithm's derived learning signal. A judge may score after episode completion,
but its selection must declare prefix-only versus retrospective context.

#### R7.1 — retain evidence and amend the baseline before code

Audit `/tmp/automationbench-capo-toy-20260906-b` with
`scripts/qualification/audit_structured_toy.py`, including `--reload-export`.
The GDPO audit at `/tmp/automationbench-gdpo-toy-20260906-e/audit.json` already
passes; its repository receipt is
`docs/tooling/trl/evidence/2026-09-06/gdpo-automationbench-toy.json`.
Retain raw native records, selections, judge output, logs and models through the
existing artifact backend, linking repository summaries to immutable bundles.
Do not rerun completed training merely to fill an audit gap. Audit failures
become scoped fixes and regression tests, not overwritten evidence.

Before implementation, narrowly amend `docs/post-training/README.md` and
`02-primitives.md`, `04-framework.md`, `05-apis.md`, `06-observation-and-lineage.md`:
this extension adds turn-addressed evidence and modern native episode envelopes.
Keep environment ownership, operation names and job/run boundaries unchanged.
Replace misleading “project scorer” terminology with custom Verifiers
judge/scoring-plugin configuration; do not create a new product primitive or
parallel evaluation database. This revision plans that amendment; it does not
silently change the frozen baseline.

#### R7.2 — qualify modern native Verifiers and environment compatibility

Current Verifiers is upstream `284a868d6a9022109b749710672a0460e8a996d4`,
not a CarbonTeq fork. Candidate baseline is released v0.3.1 at
`8e8f3042481c0996a58c3de0f86d55406725b6c1`; reverify identity before locking.
Inspected main `25debce78aff23fca201cefe9c6f72dc65176d06` is research evidence,
not a floating upgrade target. Modern native APIs wrap task and trace in Episode.
Migrate activation, generation, scoring, persistence and reload together in
`packages/train/src/posttrain/train/integrations/verifiers.py`,
`packages/data/src/posttrain/data/adapters/verifiers.py`, and
`packages/eval/src/posttrain/eval/backends/verifiers/{adapter,runtime,synchronization}.py`.
Inventory distillation and Observatory readers before editing. Keep a detected
legacy-artifact reader for one documented compatibility window; write only the
new native format and never rewrite old bundles in place. Qualify explicit local
execution and installed plugin discovery so changed environment-server/runtime
defaults cannot silently select remote services or require new credentials.

Native node advantages use sampled-position coordinates, while loss weights use
full-token coordinates. These are training annotations, not turn assessments.
Validate completeness before native flattening can fill absent advantages with
zero; never write raw judge ratings directly into advantages. Retain exact token
IDs and numerical streams; disable any native float-rounding option when writing
numerical replay authority, rather than using display-rounded records for audits.

AutomationBench remains the external `carbonteq-ai/verifiers-environments`
package at `b7bcb591facfcd2b073802f6d7496b24ab9c479e`, subdirectory
`environments/automationbench_v1`. Resolve and record its checkout, branch,
remotes and commit here before changing it. Put compatibility fixes and native
activation/replay tests there, not inside Posttrain. Prefer upstream extension
points and versioned native annotations; create a maintained Verifiers fork only
for a demonstrated generic gap that those mechanisms cannot handle.

#### R7.3 — generic turn transport and task-owned scoring

Extend `packages/train/src/posttrain/train/reward_evidence.py`,
`reward_projection.py`, `reward_recovery.py`, `online_rl.py` and the Verifiers
bridge. A typed turn-evidence value identifies native episode/trace/branch,
turn ID, generic reward name, finite value or explicit status, scorer/projection
digest and raw-assessment reference. No rubric dimension names belong in this
schema. The authoritative turn map preserves original sampled tokens, native
node/offset provenance and policy eligibility. Reject duplicate/unknown turns,
cross-episode references, missing required scores, invalid spans and unsupported
branches before complete-group admission. All-absent optional turn evidence
preserves legacy scalar behavior; partial required evidence is not zero-filled.

Custom judge/scoring plugins live in pinned external environment or project
packages. They prepare task-specific context/rubrics, call injected inference
clients, validate structured output and emit native annotations/rewards. Reuse
native Judge facilities; one plugin may perform both judging and score extraction,
without separate mandatory classes. Multiple judges may use different models for
turn quality and trajectory outcome. Runtime composition in
`packages/jobs/src/posttrain/jobs/definitions.py` and existing runtime factories
binds clients from serializable inference selections. Plugins do not launch
vLLM, load CUDA models, choose devices or import train/eval/serve. The toy runner's
local model-loading wrapper is not the production interface.

The five reasoning dimensions are only an example project rubric:
understanding/planning, logical correctness, evidence/state grounding,
verification/self-correction, progress/efficiency. A plugin can combine them
internally into one reward or expose selected generic outputs. Training selection
owns subsequent combination of separately exposed components. Preserve diagnostic
ratings without forcing them into algorithm inputs. Never assume a quality rating
is a calibrated probability of eventual success.

Scorer identity covers code, rubric, judge revision, inference settings, context
scope, output schema, reduction and bounded retry policy. Cache identity also
binds exact episode/turn input. Preserve timeout, abstention, inapplicability and
invalid-output statuses distinctly from valid zero; record retry attempts without
secrets. Detached planning must work without importing models or environments.
Exit with a project example whose rubric and judge can change without modifying
Verifiers or algorithm code, plus cache/resume mismatch rejection tests.

#### R7.4 — explicit algorithm mappings and useful diagnostics

Connect complete turn evidence to existing `AgenticTurn.step_reward` and
`sampo_advantages.py` on each supported backend. Preserve SAMPO anchor-relative
returns, episode component, sequence-ratio loss and sparse terminal fallback when
no turn rewards are selected. Declare whether the turn reward sequence includes
terminal outcome to prevent double counting. The historical field name does not
require semantic step extraction or an immediate public rename.

For CAPO, use versioned `assistant-turns@1` projection in `critique.py` and
`capo_advantages.py`. Error marks address known turn IDs and map to the union of
eligible sampled positions in each turn. A quality rating is not automatically
an error label: the plugin defines that interpretation. Preserve outcome/process
weights, token ratios and complete-group token normalization. Call this
turn-granularity CAPO, not reproduction of reference paragraph segmentation.
Defer semantic within-turn extraction; reasoning-only credit additionally needs
a verified native channel mask and is not proven by reasoning-disabled toys.

For GDPO, explicitly reduce selected turn scores to trajectory components before
existing normalization in `gdpo_advantages.py`. Pin reduction/missing-value rules:
sum and mean create different length incentives. Do not silently redefine GDPO
as independent per-turn normalization. Golden fixtures must show the distinct
algorithm results produced from the same raw turn evidence.

Update `grpo_observations.py` and backend adapters to report actual structured
advantages entering loss, coverage, failures, admission and separate task success.
Ordinary scalar-reward diagnostics must not falsely imply zero learning signal
on a structured path. Keep per-turn detail in native artifacts, not unbounded
metric names.

#### R7.5 — qualify behavior and release the exact tested artifacts

Extend `packages/train/tests/test_reward_evidence.py`, `test_reward_projection.py`,
`test_reward_recovery.py`, `test_critique.py`, `test_reward_advantages.py`,
`test_sampo.py` and `test_verifiers_grpo_bridge.py`. Add migration round trips in
data/eval tests and native-installed integrations, not just fakes. Prove turn IDs
and noncontiguous masks survive reordering/padding/sharding, observations receive
zero direct policy credit, incomplete evidence fails and changed scorers invalidate
cache/resume. Conflicting scores across several turns must catch accidental
episode-score broadcast. Exercise prefix-only and retrospective assessment.

Run all four GDPO/CAPO cells for five optimizer updates on the selected 0.8B/2B
AutomationBench profile through production composition. Existing toy results do
not replace reruns of changed production paths. Also exercise explicit-turn and
legacy sparse SAMPO modes on every backend claimed supported. Preserve two-rank
numerical/gradient parity and checkpoint/resume/export gates. Record exact work
packages, commands, model/scorer revisions, runtime, hardware and budgets here
before launching. Do not interrupt busy workers. Missing credentials, endpoints
or capacity block real integration gates rather than being hidden by fake tests.

Judge quality is a separate gate: retain a manually reviewed calibration fixture
with valid/invalid actions, state errors, corrections and redundant turns. Freeze
expected ratings/error marks and acceptance criteria before scoring; report
disagreements and repeated-call stability. Nonzero gradients and five updates do
not demonstrate trustworthy reasoning judgments or benchmark improvement.

Current framework main is `8f240ac5474ecaecb5f851751d52e37dbd0d3dee` with dirty
implementation. Candidate `/tmp/posttrain-fork-upgrade.ouJ7s4`, branch
`codex/trl-verl-stable-base-upgrade`, is
`0a356ad4b2d04bbf7c314b7d730b0cd0f88794fb`; main still pins TRL 1.9.2.post11.
TRL `/home/hammad/projects/trl-gdpo-capo`, branch
`codex/trl-1.12-retained-features`, HEAD
`63d67a968e06b64f70197eb6c8073e2d38531c96`, retains published post2 source
`95a787b6c04f91a5d485fd827d31b1e1fb67ae8e`. veRL
`/home/hammad/projects/verl-gdpo-capo`, branch `codex/verl-0.9-retained-features`,
HEAD `c4b2d65f90127f8b24c8e162797a0b23df9c2025`, retains published base
`cec7e74c361bb973b641db8dfbb75a5544c33139`; structured reward additions remain
dirty/unpublished. Reverify identities before commits; preserve unrelated work.

Commit/publish external environment compatibility and generic fork changes first,
with per-repository tests and fork ledgers. Follow `docs/tooling/forks.md`: retain
immutable assets/hashes, publish development artifacts, resolve candidate locks,
qualify installed wheels and actual-job OCI images, promote unchanged bytes, then
update stable pins. Update consumer pages for TRL/veRL/Verifiers, train/data/eval
manifests, catalog environment revisions, `uv.lock` and isolated veRL runtime
profile/constraints/release locks. Run retained-feature suites in each fork as
well as new tests. Release notes must name qualified algorithm/projection/backend
profiles, evidence, migration window and unsupported modes. A narrower release
than four qualified cells requires an explicit scope decision. Preserve prior
stable artifacts for rollback; do not publish as a side effect of this plan edit.

### Milestone 1 — Baseline amendment and mathematical fixtures

Amend the canonical API baseline before implementation once approved. Add the
two operations, their typed scorer/reward requirements, exact support boundaries,
and derived trace evidence to the relevant 02/04/05/06 documents and README
amendment log. Keep environments as owners of tasks and verified outcomes.

Write a small independent reference oracle and fixtures under
packages/train/tests/fixtures/reward_algorithms/, with tests in
test_gdpo.py and test_capo.py. These must not call the implementation they test.
Reconcile paper/reference numerical conventions and record them here, including
the normalization-population discrepancy. Inspect reference code and fork
licenses. Exit with explicit equations, expected arrays, and loss reductions,
not an unresolved choice hidden in an adapter.

### Milestone 2 — Reward transport, selection, and admission

Implement reward_evidence.py, extend online_rl.py and Verifiers projection,
then profiles.py, requests.py, catalog_schema.py, api.py, results.py,
verifiers_requests.py, and exports. Extend standard jobs and catalog discovery
so an ordinary project can select either algorithm on either backend.
Scorer identities, schemas, and failures must survive isolated serialization.

Keep dynamic/active scalar-variance filtering disabled in initial GDPO/CAPO
profiles. It is not a required part of these algorithms and can discard useful
signals. Bound retries for incomplete/error groups separately. If filtering is
later enabled, implement an explicit algorithm-aware admission rule and
recompute normalization over the retained logical batch, not candidate waves.
Never encode a fake scalar reward just to trick the existing filter.

Exit with an actual pinned Verifiers fixture transporting multiple rewards and
CAPO span annotations into the generic contract, preserving masks and identities.
Legacy scalar GRPO/DAPO and SAMPO sparse fallback remain unchanged.

### Milestone 3 — GDPO on TRL and veRL

In backends/trl/policy_optimization.py, reuse the existing multi-reward callback machinery:
provide one validated callback per declared component, set
multi_objective_aggregation=normalize_then_sum, and select token importance
ratios with the chosen per-response loss reduction. Do not continue forwarding
only _bridge_reward. The native trainer gathers rewards before normalization;
verify group order and complete logical batches across ranks. Keep original
component scores available for evidence. Reject unsupported Liger combinations
until their exact numerical behavior is proven.

For veRL, extend contracts.py, launcher.py, worker.py, agent_loop.py, and
reward_fields.py. Add metadata for ordered components and explicit group IDs.
In the maintained fork, update core_algos.py and V1 trainer_base.py so
_compute_advantage fetches/materializes the new fields and calls GDPO with them.
Audit ray_trainer.py and v1/utils.py to retain the intended group/session
population and avoid missing prompts/attention_mask dependencies. Support the
declared rollout-weighted normalization; keep legacy variant behavior explicit.

Exit with equal oracle advantages and loss gradients across backends for unequal
lengths, holes in masks, multiple prompt groups, and shard permutations.
Test raw component delivery through V1 TransferQueue, not only core_algos directly.

### Milestone 4 — CAPO judging, token credit, and both losses

Implement the reusable validated critique projection and token-credit builder.
Independent project scorer code supplies the rubric; runtime composition binds
the frozen judge without a train-to-serve import. Retain raw critiques and
resolved annotations, and inject fixed annotations for numerical tests.

TRL can reuse precomputed_advantages for single-process qualification, but
the callback's local computation must not normalize a split prompt group.
Implement a generic distributed raw-credit/group gather or a post-gather
advantage construction seam in the maintained TRL fork if required. It must run
after the final logical batch is known, before minibatch slicing, and preserve
row/token alignment. Carry credit metadata through padding, shuffle, buffering,
and gradient accumulation. Token-aligned advantages do not authorize SAMPO's
sequence ratio. Reject Liger on this path until supported.

Add CAPO estimator/config registration to the maintained veRL fork. Materialize
validated token credit through V1's queue, reconstruct its padded layout from
sampled-token ordering and masks, compute group statistics, and write nested
advantages back correctly. Do not assume training's repadded coordinates equal
original trace offsets. Reuse the standard token-ratio loss, not GSPO.
Add CAPO to framework manifests, job selection, metrics, and resume compatibility.

Exit with real critic integration plus identical oracle advantages, loss, and
gradients on fixed inputs for both backends. Add process-error injection cases
across assistant turns to prove turn-local differentiation. Within-turn semantic
segmentation is deferred by Revision 7, not a release requirement.

### Milestone 5 — Qualification, release, and handoff

Run all four cells through source tests, one-rank real updates, multi-rank
numerical transport tests, and bounded real jobs. Start with a small supported
policy for mechanism qualification, then repeat the intended 4B/12B pair on
profiled hardware before claiming pilot readiness. Use the same source task
population, frozen judge, token budgets, and update recipe when comparing
backends; backend-specific runtime differences are recorded.

Preserve immutable evidence: resolved selections, algorithm/rubric versions,
native traces, component scores, masked advantages or bounded checksums/samples,
loss/gradient summaries, clipping/KL statistics, checkpoints, exported-model
reload results, retries, judge tokens, and total runtime costs.
Measure integration correctness separately from held-out policy improvement.

Publish required TRL and veRL fork changes with their CARBONTEQ_FORK.md ledgers
before updating framework pins. Update docs/tooling/trl/README.md and
docs/tooling/verl/README.md, train manifests/uv.lock, and the isolated veRL
runtime profile/release lock/constraints through the repository's release
procedure. Then rebuild/package and repeat the qualified smoke cells against
those immutable artifacts. A source-tree pass is not release evidence.
No registry publication or paid run is authorized by this planning turn.

## Validation and Acceptance

Revision 12 acceptance requires the same native judge plugin and algorithm code
to work unchanged when switching compatible inference bindings. Test two rubric
plugins sharing one service, two services with isolated files/credentials,
policy/judge/draft artifact isolation, attached-service non-ownership, partial
startup failure, cancellation and resource/identity mismatch rejection. A local
test cannot certify remote placement. A successful response cannot certify JSON
schema, reasoning-parser or speculative compatibility without explicit probes.
End-to-end tests must retain the deployment receipt, checkpoint/draft/runtime
identities, effective requests, raw assessments and derived rewards without
secrets. No Nanbeige quality, DSpark speedup or RTX PRO 6000 compatibility claim is
made until measured. Existing unrelated correctness gates remain in force.

The following matrix is an implementation requirement, not a report of passed
new tests:

| Test class | Required cases | Observable acceptance |
| --- | --- | --- |
| GDPO oracle | Conflicting components; equal scalar totals; scale changes; zero weight/constant component; unequal lengths | Component distinctions survive where equations predict; declared final normalization matches exactly |
| CAPO oracle | Correct/wrong outcome × error/no error; multiple spans; duplicate critique references; zero variance | Correct raw hierarchy and token advantages; overlaps do not multiply penalties |
| Identity | Same task twice in a batch; interleaved completions; groups split across ranks; V1 sessions | No accidental group mixing or duplicate trajectory weighting |
| Masking | Assistant/tool/assistant; padding; EOS; truncation; process spans within a turn | Tools/padding have zero direct policy-loss contribution and are excluded from normalization |
| Missing evidence | Invalid JSON, unavailable component, wrong trace, NaN, timeout, partial vote | Typed failure or bounded group replacement; never silently scored as zero |
| Loss parity | Positive/negative advantages; clipped/unclipped ratios; beta 0/nonzero | Same declared objective, scalar loss, and per-token log-probability gradients |
| Distributed statistics | 1 vs 2 ranks; rank-order permutation; gradient accumulation/microbatch changes | Same logical-batch oracle within declared precision tolerance |
| Backend transport | TRL callback→gather→loss; veRL AgentLoop→queue→estimator→actor | Distinct scores/credits arrive intact, including noncontiguous trainable tokens |
| Lifecycle | Update, weight sync, export, reload, resume, scorer/config mismatch | Optimizer progress and usable artifacts; incompatible resume rejected |
| Regression | GRPO/DAPO/SAMPO, SFT masks, standard jobs, detached planning | Existing behavior unchanged for unselected algorithms |

Start oracle comparisons in float64 with tight tolerance, then float32
(atol=1e-6, rtol=1e-5 as initial test tolerances). Set measured BF16 runtime
tolerances explicitly rather than widening them to hide a normalization mismatch.
For masked loss, perturb logits at excluded positions in a controlled fixture;
do not assert that observation context has no influence on later predictions.

Distributed support requires tests of global counts, sums, and variances before
local slicing. GDPO final statistics cover the admitted logical batch, not a
rank or accumulation microbatch. CAPO statistics cover each complete prompt
group. Verify that gradient accumulation does not renormalize advantages or
change the effective loss denominator.

Initial live gates: at least two complete prompt groups with multiple attempts,
two optimizer steps to exercise weight refresh, and a complete checkpoint after
step one. Resume and execute the next step; compare with an uninterrupted
deterministic fixture within declared tolerances. Include a real pinned
Verifiers task and frozen judge in CAPO on both backends, not only fabricated
credit. No fixed GPU-hour estimate is claimed before profiling.

Qualification is per cell. A blocked cell remains unsupported/unqualified while
others may be published with narrow claims. Never mark both backends complete
from an isolated core function or one backend's run.

## Concrete Steps

### Revision 12 commands and launch prerequisites


Work in `/home/hammad/projects/rl`. Before implementation resolve target/draft
Hugging Face revisions and run `git ls-remote https://github.com/Nanbeige/vllm.git
refs/heads/nanbeige42`; retain the resulting SHA before building an isolated
runtime. Use the user-confirmed `carbonteq-ai-workstation.lan` dstack worker and
verify its `RTXPRO6000BlackwellWorkstationEdition` identity, 96 GB resource
record, free memory, endpoint routing and deployment authority immediately before
launch. A stale or conflicting reservation blocks launch even when `nvidia-smi`
reports free memory; release it through the owning provider lifecycle first.

After each framework slice, from the repository root run:

    uv run pytest packages/jobs/tests/test_native_judges.py packages/serve/tests -q
    uv run pytest packages/jobs/tests/test_inference_services.py -q
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    git diff --check

`test_inference_services.py` is a planned new test file and does not exist yet.
Add backend configuration tests under `packages/serve/tests` for immutable draft
dependencies and actual method labels. Before remote provider integration run
`uv run pytest packages/execution-dstack/tests packages/work/tests/test_execution_target_override.py -q`.
If editing the external native plugin, run its existing judge tests in the
documented native v0.3.1 candidate interpreter; do not pretend the old stable
Verifiers lock qualifies the new API. Follow the full repository validation
ladder with candidate dependency locks before release. Populate exact remote
launch/probe/comparison commands here after the runtime and host identities are
resolved; no placeholder shell command should be presented as a runnable GPU job.

### Revision 10 installed native-judge development qualification

The next run uses `scripts/qualification/automationbench_native_judges.py`, the
public training API, native `AutomationBenchTurnJudge`, and
`posttrain.jobs.bind_native_judges`. It is explicitly an **unpublished development
candidate**, not OCI/release acceptance. The original external-interpreter veRL
binding remains supported (`docs/tooling/verl/README.md`); no capsule isolation
check is removed or bypassed.

Hardware: idle RTX PRO 6000 Blackwell, 96 GiB, on
`carbonteq-ai-workstation.lan`. Task directory:
`/home/carbonteq-ai-workstation/posttrain-qualification.2iyErl`.
Interpreter: `verl-venv/bin/python` (Python 3.13.12). The candidate backend lock
installed Torch 2.11.0+cu130 and vLLM 0.25.2.dev2+g7817d8457.precompiled; the base
vLLM wheel SHA-256 is
`16fc7a28df1576eb6f7ca0455026551b8f9adb674c19c66059359ef3e964bd1e`.
CUDA matrix smoke passed. Candidate wheels from `/tmp/posttrain-turn-native-wheels`
were installed without dependency resolution; this is not a clean release install.
Native Verifiers requires renderers 0.1.10, prime-pydantic-config 0.4.3 and newer
prime-sandboxes; these are installed explicitly. Full final lock regeneration is
still required. The unused Trackio dependency is not yet installed in this runtime.

Policy: catalog Qwen3.5-0.8B BF16, LoRA rank 4/alpha 8; judge: pinned
Qwen3.5-2B `15852e8c16360a2fea060d615a32b45270f8a8fc`, hosted via managed vLLM
on loopback port 8123 (20% GPU budget). Policy rollout uses 25% GPU budget for
veRL; two generations, one prompt per update, five updates, 8192 context, 768
response tokens, at most 12 assistant turns. Checkpoints every two updates.
The judge source is frozen by Git blob
`59dd00e20b42ee83d2e76f845d339185e51b3bc7`; this is a source blob, **not a
published commit**. Rubric/retries/native sampling schema and inference selections
are saved before training. Keep all failed attempts and allocate new output dirs.

Working directory is the remote task directory. Run (through the existing scoped
SSH bootstrap key; no credential contents are retained):

    HF_HUB_CACHE="$PWD/models" OMP_NUM_THREADS=8 verl-venv/bin/python automationbench_native_judges.py --backend trl --algorithm gdpo --judge-code-revision 59dd00e20b42ee83d2e76f845d339185e51b3bc7 --output gdpo-managed-a

Then repeat for CAPO, and veRL with `--verl-source "$PWD/verl-source"` and
distinct output directories. The veRL checkout preserves HEAD
`c4b2d65f90127f8b24c8e162797a0b23df9c2025` plus the explicitly copied dirty
candidate delta. Preserve the launch manifest's source-state digest. Judge
calibration, SAMPO runs, resume/export and final package/OCI gates remain separate.

Modern prototype retained bundles:
`outputs/qualification/automationbench-2026-09-06/gdpo-native-b.tar.gz`
SHA-256 `ba4e13349498ba85b246372242ea61b24c99ca2825c0ff16b6ac2464c3f7a5bd`;
`capo-native-a.tar.gz` SHA-256
`ed458b095ad1d387b24e13f43dcafb8065b576582f5cd51cbc7891eaef76b181`.
GDPO: 13 traces/10 admitted, four nonzero gradient norms, adapter B norm
0.0241265952. CAPO: 14 traces/10 admitted, all five nonzero gradients, adapter B
norm 0.0235076261. Native success 1/10 and 0/10 respectively is not a learning claim.

Revision 8 executed reader qualification, from `/home/hammad/projects/rl`:

    uv sync --project /tmp/posttrain-verifiers-031.AZRgVw/source --locked --no-dev --python 3.13
    uv pip install --python /tmp/posttrain-verifiers-031.AZRgVw/source/.venv/bin/python pytest pytest-asyncio
    PYTHONPATH=/home/hammad/projects/rl/packages/data/src:/home/hammad/projects/rl/packages/common/src /tmp/posttrain-verifiers-031.AZRgVw/source/.venv/bin/python -m pytest packages/data/tests/test_verifiers_episodes.py -q -rs

Observed: 4 passed against real Verifiers v0.3.1. The locked upstream source is
`8e8f3042481c0996a58c3de0f86d55406725b6c1`; this test environment is not a
release runtime. The old installed runtime passes 64 data/environment/sync tests
with one expected modern-API skip, and 84 bridge/reward-projection/eval tests.
When using `/tmp/posttrain-fork-upgrade.ouJ7s4/.venv/bin/python` for regressions,
set PYTHONPATH to this checkout's train/eval/data/environment source directories;
otherwise editable installs resolve candidate code instead of current changes.
Scoped Ruff and Pyright pass; `uv run --offline --locked lint-imports` keeps all
eight boundaries. Full packaged qualification remains outstanding.

Revision 7 first command, from `/home/hammad/projects/rl`, using the existing
candidate environment (verify it still exists and that the local GPU is free):

    HF_HUB_OFFLINE=1 PYTHONPATH=/home/hammad/projects/rl/packages/train/src /tmp/posttrain-fork-upgrade.ouJ7s4/.venv/bin/python scripts/qualification/audit_structured_toy.py /tmp/automationbench-capo-toy-20260906-b --reload-export

Acceptance is an independently generated audit with exact-token alignment,
numerical checks, five recorded updates, parameter-change evidence and successful
export inference. Do not treat this planned command as a passed test.

After R7 implementation, from the framework root in its newly locked candidate
environment, run the focused suite, then the full validation ladder below:

    uv run pytest packages/train/tests/test_reward_evidence.py packages/train/tests/test_reward_projection.py packages/train/tests/test_reward_recovery.py packages/train/tests/test_critique.py packages/train/tests/test_reward_advantages.py packages/train/tests/test_sampo.py packages/train/tests/test_verifiers_grpo_bridge.py -q -rs
    uv run pytest packages/data/tests packages/eval/tests packages/jobs/tests -q -rs

From `/home/hammad/projects/trl-gdpo-capo`, using its qualified TRL environment:

    python -m pytest tests/test_liger_window_normalization.py -q

From `/home/hammad/projects/verl-gdpo-capo`, using the isolated veRL environment:

    python -m pytest tests/trainer/ppo/test_structured_rewards_on_cpu.py tests/trainer/ppo/test_reward_recovery_on_cpu.py tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py tests/trainer/ppo/v1/test_trainer_base_on_cpu.py -q

These are focused entry points, not substitutes for each fork ledger's complete
retained-feature and GPU commands. Record exact external environment migration
test commands after resolving its checkout in R7.2, before changing or publishing
that repository. If a candidate runtime is missing, rebuild its locked profile;
do not install incompatible TRL/veRL dependencies into a shared ad hoc runtime.

Existing baseline command, from /home/hammad/projects/rl:

    uv run --offline pytest packages/train/tests/test_sampo.py packages/train/tests/test_verifiers_grpo_bridge.py packages/train/tests/test_verl_backend.py -q -rs

Observed 2026-09-06: 57 passed, 1 skipped in 0.28s. The skipped module could not
import verifiers. This does not prove the Verifiers runtime or either new method.

After adding the proposed tests:

    uv run pytest packages/train/tests/test_gdpo.py packages/train/tests/test_capo.py packages/train/tests/test_reward_evidence.py
    uv run pytest packages/train/tests/test_verifiers_grpo_bridge.py packages/train/tests/test_trl_online_rl.py packages/train/tests/test_verl_backend.py
    uv run pytest packages/train/tests/test_api.py packages/train/tests/test_checkpoints.py packages/train/tests/test_grpo_observations.py
    uv run pytest packages/train/tests/test_rendering.py packages/train/tests/test_sft_validation.py
    uv run pytest packages/jobs/tests packages/runtime-images/tests/test_verl_release_gate.py

Dependency installation follows the locked repository/runtime profiles. Offline
baseline execution is deliberately not used as a substitute for missing runtime
dependencies. Test modules needing models, optional packages, or GPUs must use
the repository's existing markers and report skips explicitly.

Proposed fork regression targets, created in the relevant isolated worktrees:
TRL tests/test_gdpo_capo_contract.py and veRL
tests/trainer/ppo/test_gdpo_capo_on_cpu.py plus
tests/trainer/ppo/test_gdpo_capo_v1_transport_on_cpu.py. Run pytest against each
using its locked backend environment; add a two-rank distributed fixture with
torchrun --standalone --nproc_per_node=2 against a dedicated test driver.
These are planned files, not existing runnable commands.

Create versioned work-package files under apps/lab's existing qualification
layout for each algorithm/backend cell. Record the actual names, catalog IDs,
hardware, exact CLI command, and run IDs in this plan before launching them.
Use the existing posttrain work-package validation and job plan/run commands;
do not invent runtime-compatible bindings from a paper's shell script.

Before handoff, from the framework root:

    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Fork changes require their own focused tests and ledgers before pin updates.
New import boundaries, job identities, dataset/scorer serialization, checkpoint
compatibility, and packaged runtime activation are release gates.

## Idempotence and Recovery

R12 trials use fresh service-instance directories and immutable assessment
namespaces. Preserve failed attempts and original native Episodes. Never replace
the working TRL/veRL interpreter or install from a moving branch into it. Attached
endpoints are never stopped by the caller. Retry infrastructure only within a
bounded policy and the same frozen inference identity; changing target, draft,
runtime or rubric creates a new selection/profile, not a silent recovery.

Work on isolated branches/worktrees from the declared source bases. Preserve the
dirty verl-upstream checkout and do not move local branch pointers. Version
datasets, traces, scorer caches, and settings; never overwrite old evidence.
Retry judge calls under stable request identity, recording stochastic attempts.
Do not replay external side effects to regenerate missing annotations; score
retained traces or use isolated resettable environments.

Keep scalar fields for legacy callers and require explicit opt-in to structured
rewards. No data migration or deletion is needed. Restore only complete
compatible checkpoints. A changed algorithm, reward schema, normalization
population, scorer, rubric, or renderer requires an explicit compatibility
decision/new run. Keep the prior published backend versions available until
new artifacts pass qualification.

## Surprises & Discoveries

2026-09-10 exact judge replay: the apparent need for ever-larger Spark output
budgets was caused by incompatible instructions and structured output, not by
the episode input size. R7 requests carried a turn rubric demanding a `turns`
list while constrained decoding required `requirement_checks` and episode
assessments. The model tried to encode the rubric itself as task requirements.
One corrected episode contract reduced wall time by more than half and made all
14 captured requests structurally valid at a 2K response cap. Structural
validity is still separate from semantic discrimination: Spark no-thinking
distinguished failed controls but gave uniform perfect ratings to the successful
real traces, including cases with subtle unsupported claims.

2026-09-08 correctness/performance audit: the bridge returned
`retained_input_indices`, but post4 TRL never consumed them, leaving reward
inputs misaligned after dropped groups. GRPO also only enabled complete-group
retention for OLMo active sampling; setting `max_admission_attempts: 1` alone
therefore failed vanilla GRPO on one invalid group. Correct both seams before
relaunch. The separate synchronous `_generate_single_turn` call in
`backends/trl/online_rl.py` blocks its asyncio loop: a 500 ms fake generation
delayed a 50 ms callback until 500 ms. This proves event-loop blockage, not its
share of the observed rollout duration. Harness optimization is deferred by
explicit user direction; preserve this finding for the subsequent work.

2026-09-08 A100 deployment: dstack's authenticated offer query returned A100
80 GB on-demand inventory at $1.59/hour in EUR-IS-1, US-KS-2 and US-WA-1, but
RunPod rejected multiple subsequent Pod creation attempts as no capacity. The
dynamic fleet correctly released each failed pending node back to zero and the
two canonical runs remained queued under bounded capacity retry. Catalog
availability is therefore advisory; only a provisioned instance establishes
usable capacity.

2026-09-08 Secure Cloud inventory: changing an `InferenceBinding.target` does
not create another provider allocation. The current judged job materializes and
launches its `ManagedInferenceService` inside the enclosing training task, so a
single-GPU dstack run would still colocate LFM training, LFM rollout serving and
Gemma judging despite showing two target selections in resolved provenance. The
production decision already requires an independently owned attached judge;
the missing work is remote service orchestration and live lifecycle evidence,
not another catalog-only target substitution.

2026-09-08 capacity snapshot: the least expensive available Secure Cloud
on-demand GPU with at least 40 GB is one L40 48 GB at $0.82/hour. The next tier
is A100 80 GB at $1.59/hour with three offers; RTX PRO 6000 Server Edition 96 GB
has four offers at $2.09/hour. The earlier micro-batch-two actor forward reached
approximately 93.7 GiB on a 96 GiB worker. Micro-batch one and chunked logits
reduce that peak, but no production-size 48 GB optimizer update exists, so L40
is an optimization candidate rather than an admissible 50-update target.

2026-09-07 portability correction: the LFM three-arm qualification proved a
deployment-boundary defect in the qualification harness. The algorithm and
runtime APIs are packageable, but `scripts/qualification/lfm26-three-arm-comparison.dstack.yml`
directly mounted `/home/carbonteq-ai-workstation/...` and
`/var/lib/posttrain/...`. A direct fleet substitution therefore cannot run on
RunPod or preserve spot checkpoints. The standard actual-job image and managed
run-storage path already solve this for ordinary Posttrain jobs; the correction
is to move the qualification onto that boundary rather than add cloud-specific
mount aliases.

2026-09-07 R7 memory diagnosis: the failure occurred after rollout and before
optimizer update one in TRL actor log-probability computation. Scalar GRPO did
not enter the managed-judge context, so Gemma consumed no memory. The colocated
policy rollout engine was configured to sleep during optimization, leaving the
training micro-batch of two as the smallest experiment-preserving capacity
knob. Reducing it to one keeps the 64 sampled trajectories and effective update
batch unchanged through gradient accumulation.

2026-09-07: The prior episode schema was structurally valid but semantically
underconstrained. Role-collapsing the rubric and trajectory into one user
message, omitting exact tool schemas, and asking for scores without a preceding
requirement audit allowed a thinking-enabled judge to produce uniformly perfect
ratings over an obvious request/action/result contradiction. Structured output
validation must therefore cover cross-field semantics, not JSON shape alone.

2026-09-07: Equal step-one rewards in `training-gdpo-trl-gemma-e` exposed
lossy message projection, not generation truncation. Native traces
`6bb7832f4b2649319e98cf336f770619` and `40c250acaa5b4981a2ed75bf0e81ee88`
retain 57/55 sampled tokens for attempted Asana lookups with closing tool tags
and EOS, but their assistant messages were empty. The renderer retained
non-OK parsed calls; both trainer adapters filtered them away. Distinct judge
responses consequently assigned [1, 0] to both trajectories. A third affected
calendar call also survives replay through the corrected projection. Raw
evidence remains at `/var/lib/posttrain/qualifications/nanbeige-r12-20260906/training-gdpo-trl-gemma-e`
on the RTX PRO 6000 worker; do not overwrite it or count nonzero gradients as
proof that the judge saw complete evidence.

- 2026-09-06: Verifiers v0.3.1 is a repository-wide compatibility boundary for
  `carbonteq-ai/verifiers-environments`, not an AutomationBench-only pin. The
  combined-wheel gate caught five stale locks plus required loader, trace,
  reward-object and task-toolset migrations. Updating one package alone would
  have produced mutually incompatible wheels from a single repository revision.
- 2026-09-06: TRL's ordinary Transformers generation path deliberately returns
  no sampling log-probabilities, which is insufficient for the framework's
  exact structured-RL rollout contract. The managed qualification therefore
  uses the selected colocated vLLM inference capability, which returns sampled
  token log-probabilities, instead of manufacturing values in the bridge.

2026-09-06, R12 implementation: Hugging Face resolves the Nanbeige target to
`3384e426066d1a49c3aea90a7190b81260a6533f` and the DSpark draft to
`5f12c792dabbfcdc4a0cf504e75f7216706d9590`; the Nanbeige vLLM `nanbeige42`
branch resolves to `62f6de733d7ae63b759329993bc209e67afdf431`. The target tokenizer JSON
SHA-256 is `1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f`.
The native plugin still accepts credentials by environment-variable reference,
so the generic service layer is client-scoped and secret-free but the legacy
in-process Verifiers adapter retains a narrowly scoped environment bridge. Full
elimination requires consumer-process isolation or a native client credential
API; it is not hidden as completed service isolation.

The Nanbeige branch identifies itself after `v0.26.1rc0` and has no common Git
base with the current CarbonTeq v0.25.1 maintenance checkout available locally.
An initial narrow source search incorrectly suggested that it lacked
TurboQuant. Direct inspection of the branch's quantization modules, attention
backends, Triton/FlyDSL kernels, cache-dtype configuration, and history confirms
that it contains a later TurboQuant implementation with backend-owned KV-cache
specification. The older CarbonTeq reshape patch therefore cannot be transplanted
mechanically and appears structurally superseded. Retain the separately locked
Nanbeige runtime and qualify all four standard/TurboQuant/DSpark/composed cells
before considering fork convergence or changing the global vLLM line.

Live fleet inspection reports `RTX3070Ti` and
`RTXPRO6000BlackwellWorkstationEdition`. The user clarified that the latter is
the intended target: `carbonteq-ai-workstation.lan`, with 98,304 MiB in dstack's
resource record. Before qualification it was held by a stale `occupancy-engine`
task whose main `posttrain-runtime` process was absent while dozens of task-owned
environment-service descendants kept the container and reservation alive. The
stale task terminated gracefully, all task-owned descendants drained, the
worker returned to zero busy blocks, and retained run storage was not deleted.
This exposes a generic child-process lifecycle defect to address separately from
the Nanbeige inference matrix.

The first Nanbeige candidate-image build failed before publication because the
vendor Dockerfile binds `.git` into a Rust relink stage, while the source checkout
was a Git worktree whose `.git` is a pointer file. `setuptools-scm` therefore had
no repository metadata. Qualification preserves the exact source revision by
building from a temporary full clone of the same commit; it does not patch the
runtime, manufacture a version, or describe the failed build as an image.

The next publication attempt exposed that the docker-container BuildKit daemon
did not inherit the workstation's already-correct system CA bundle. The durable
fix lives in `../ai-infra`: its managed BuildKit image now copies the complete
host `/etc/ssl/certs/ca-certificates.crt`, the daemon configuration contains no
per-registry CA or insecure setting, and qualification checks byte identity plus
an ordinary verified TLS handshake to the canonical registry. The retained
builder cache/state volume was reused without pruning. Publication then succeeded
at OCI index digest `sha256:bfbeee151fb59c687f7e27d5e5e287716a7c1c6dd58ebf0a4ffe99b13702a3aa`;
the linux/amd64 manifest carries source revision `62f6de733d7ae63b759329993bc209e67afdf431`.

The first complete runtime matrix distinguishes probe failure from backend
failure. Standard, TurboQuant, and DSpark all became healthy and generated 128
reasoning tokens twice; each hit the deliberately small output limit before an
answer, so the original pass predicate had no visible content to validate.
DSpark+TurboQuant failed earlier during draft initialization: every available
attention backend rejects `turboquant_k8v4` for DSpark's non-causal draft
attention, including TurboQuant itself. This is a measured compatibility limit
of runtime revision `62f6de733`, not an algorithm or judge-quality result.

Final evidence under `matrix-4` passes all three supported cells. Startup was
70.06 seconds standard, 29.02 seconds TurboQuant, and 33.03 seconds DSpark from
the already-populated host cache. The reasoning probe stopped normally with
1,073, 1,211, and 1,015 completion tokens respectively; direct mode returned
exactly `READY`. Ready-state memory is dominated by the configured 80% engine
budget (78,837/78,969/78,937 MiB), so total allocated memory does not establish
TurboQuant savings. One sequential request per cell does not establish speedup
or output-distribution equivalence.

2026-09-06, R12 inspection: The DSpark link is a draft checkpoint. The managed
serve config cannot currently carry its model artifact; all speculative benchmark
variants are labelled MTP. `serve.launch` shares log/template filenames and uses
the local Python environment. The judge helper maps one plugin to one endpoint
and temporarily mutates global credential variables. Generic model input artifact
names in jobs can cross policy/judge roles if reused without per-seat resolution.
These are implementation gaps, not reasons to put runtime fields in judge APIs.

2026-09-06, Revision 11: The first deterministic veRL run
`gdpo-deterministic-a` loaded the actor and rollout server but performed zero
updates. Retained native Episodes identify `SamplingParams.min_p=None` as the
root failure, not missing judge evidence. After three failed candidate groups,
bounded admission terminated and GPU processes cleaned up. The framework veRL
policy adapter now omits unset sampling values, preserving explicit zero and
nonzero selections. A second issue affected only startup telemetry: non-GRPO
requests were labelled SAMPO. The event now uses the selected launch operation.
Both fixes have regression coverage in `test_verl_backend.py` (51 tests pass).
Fresh attempt `gdpo-deterministic-b` uses the corrected installed train wheel;
the first attempt is retained and is not a successful qualification cell.

Attempt b then exposed a toy-runner allocation error: its 768-token response cap
was too short for complete multi-turn responses including observations. Attempt
c failed preflight because the existing request contract requires the per-call
sampling cap to equal the response cap. Attempt d selects 1024 prompt plus 7168
response tokens, an aligned 7168 call cap and the unchanged 8192 total trajectory
cap. Separating call and trajectory budgets remains an API improvement to plan;
this qualification does not bypass validation or silently truncate trajectories.
`audit_deterministic_inputs.py` independently verified b's retained source
digest, synthetic values/error IDs and native token/mask/logprob parity over
eight traces and 17 turns. It explicitly reports `backend_qualified=false`.

2026-09-06, Revision 10: Installed native-judge integration found failures that
the fake-client tests could not expose. The first remote attempt failed because
old `connect-python` and new `connectrpc` share a module namespace; removing the
old distribution and reinstalling `connectrpc` resolved import failure. The next
attempt needed the virtualenv `bin` on PATH for `ninja`. The third exposed native
validators mutating nested judge configuration inside a supposedly stable
activation; detached JSON at the activation boundary fixed serialization and
recovery identity without teaching the identity encoder about native objects.

Managed GDPO attempt `gdpo-managed-d` reached three optimizer updates and saved
step-2 recovery state, then correctly exhausted complete-group admission when
judge explanations exceeded the schema limit and longer verdicts hit the
1024-token budget. Candidate `gdpo-managed-e` used native structured outputs,
4096 judge tokens, prefix/retrospective support and explicit unavailable statuses,
but failed admission on contradictory ratings plus `status=abstained`. All
attempts retained native Episodes, and their managed endpoints shut down.

The first provisional five-case/two-repeat calibration (`judge-calibration-a`)
failed all ten calls on the same contradictory status contract. Raw ratings also
misread arithmetic and honest permission recovery. This is evidence against the
selected **judge/rubric/runtime profile**, not evidence that turn rewards or the
optimizer are mathematically wrong, nor proof that a bigger model alone fixes it.
The rubric did not explicitly instruct `status=valid` for ordinary scored turns;
the revised candidate makes that branch explicit and puts status before ratings.
It adds generic arithmetic/observation checks, without putting rubric logic into
the bridge or algorithms. `judge-calibration-b` tests that revised profile before
any further training. These are engineering regression fixtures, not independent
held-out or human-reviewed calibration.

`judge-calibration-b` completed with **2/10** passing checks. The valid-status
ambiguity was reduced for the arithmetic cases, but the wrong equation
`15:00 - 05:00 = 12:00 UTC` received quality 1.0 / erroneous=false on both calls.
The other six tool-observation checks produced invalid verdicts. Do not claim
that changing syntax solved scoring quality. R10 paused further training here;
R11 supersedes that pause for algorithm qualification. Controlled synthetic
rewards qualify credit assignment independently of judge quality. Training that
claims useful semantic supervision still requires a qualified scoring profile;
fork publication still requires the maintained-fork repository decision.
No test-owned model remains running; the remote GPU returned to its three
pre-existing desktop processes.

Native Episode retention now stamps group/run/environment identity even on failed
attempts or episodes without policy output. Both native and derived JSONL use
the same inter-process file lock. One deliberate policy-failure integration
verifies the failed native artifact remains replayable and grouped.

2026-09-06, Revision 9: modern Verifiers' interception resolves clients from
serializable configurations and had no public host client factory. A generic
optional factory is implemented in the isolated fork, not a trainer-specific
monkeypatch. The native judge API already accepts a configured hosted endpoint;
its plugin does not need to own CUDA or a serving engine.

AutomationBench's limited-tools adapter overrode legacy `_register`, but modern
Toolset calls `register`. Static tool invocation tests passed while the real
harness reported an unknown tool. The adapter now implements the new entry point
and retains a compatibility forwarding method. The live test then passed,
including actual Salesforce state mutation. Modern native eval still names its
Episode JSONL file `traces.jsonl`; consumers must detect schema, not guess from
the filename.

Revision 8 (2026-09-06): CAPO audit now passes; all five recorded gradient norms
are positive, but all ten admitted task-success values are zero. Five excluded
traces remain retained. Native v0.3.1 imports fail in the older candidate runtime
because its Prime SDK lacks `validate_egress_lists`; installing v0.3.1 from its own
upstream lock in an isolated environment fixes this without changing stable pins.
The old `verifiers.v1.episode` module exists but has no WireEpisode, so tests must
check the actual API capability rather than module presence. Modern execution
standing can be false with an empty error list; ignoring it would fabricate valid
semantic evidence. The new failure-projection regression protects that boundary.

Revision 7 evidence (2026-09-06): GDPO's audit records 12 native traces, 10
admitted attempts, five groups, four nonzero gradient norms and one zero-signal
update. One admitted attempt has native binary success. CAPO has no audit.json
yet, despite completing training. These observations separate execution,
numerical correctness, judge quality and task performance.

Native Verifiers training fields do not settle scoring semantics: advantages
and loss weights have different coordinate conventions, and absent native
advantages can flatten to zeros. Coverage must be validated before transport.
The inspected API research is grounded in the native graph/judge sources at
`/tmp/verifiers-api-review.IHsLV6`; released v0.3.1 is the migration candidate,
while additional main-branch facilities are not assumed available in that release.

2026-09-06 AutomationBench: preflight attempts A/B corrected an explicitly
versioned inference binding and native sampling-dataclass replacement. Attempt C
made no updates: complete judge JSON ended with the model-config stop token,
not the tokenizer EOS. Attempt D corrected that guard and completed three GDPO
updates (two nonzero gradients), then exhausted bounded admission because the
judge added forbidden JSON fields. Repeating greedy decoding with identical
input could not repair these responses. Scorer revision 3 requests concise
verdicts and supplies validation feedback on the second attempt, retaining strict
schema validation and native failure evidence. Run E is a fresh run, not a resume
under a changed scorer identity. No failed output is converted into zero credit.

2026-09-06 CUDA recovery gate: the retained TRL post1 candidate completed fresh
IW-OPD training but failed resume with CPU inputs against CUDA weights.
Accelerate's skip-first-batches path cannot preserve the upstream custom
`_RepeatBatchDataLoader` wrapper. The fork correction moves repetition into the
sampler/iterable dataset before accelerator preparation. Source-overlay CUDA
checks at accumulation 1 and 2 now match uninterrupted final weights and export
a reloadable model; publish a new immutable post2, never replace post1 assets.
Cancelled superseded runtime build `34006619287`; restart with post2 after
installed-wheel validation and development publication. Framework candidate
commit `46a8d96b` also aligned release/fork ledgers and CI asset hashes; 53 release
tests pass. Broader jobs/runtime tests: 70 pass, 1 skip, 13 manifest failures
expected until the genuinely new image manifest is retained.

2026-09-06 upstream-upgrade candidate: the initial runtime build
`34006530811` stopped before image building because the supervised image's
root requirements still selected `liger-kernel==0.8.0`, absent from the new
workspace lock. Candidate commit `06b301ea` updates that explicit root to
0.8.2; the existing internal-pin synchronizer only manages retained internal
packages, not public kernel roots. The candidate lock changes TRL 1.9.2.post11
to 1.12.0.post1 and Liger 0.8.0 to 0.8.2; veRL's isolated lock changes only its
fork revision/version. Do not confuse changing an index URL via a CLI override
with the explicit index configuration: CLI replacement made public packages
appear internal, so generation was repeated with the configured explicit index
and that spurious diff removed.

Execution starts on main at 8f240ac5474ecaecb5f851751d52e37dbd0d3dee.
Other framework worktrees were removed at the user's request, retaining all
branches and archiving their untracked readiness files outside this checkout.
Sibling TRL and veRL working trees remain unrelated and must be preserved.

The CAPO reference repository carries Apache-2.0. Its published recipe routes
through RayPPOTrainer rather than a named CAPO estimator; the implementation
audit must distinguish paper semantics from the checked-in recipe defaults.
Specifically, `recipe/prm/config/CAPO_Qwen1_5B.yaml` inherits
`verl/trainer/config/ppo_trainer.yaml` with `adv_estimator: rloo`, and the
`not_to_sum` path in `core_algos.py::compute_rloo_outcome_advantage` constructs
a leave-one-response-out baseline with a padded-width denominator. This does
not implement the paper's declared eligible-token group normalization. Use
the paper-profile contract, explicitly identified as such; do not claim exact
reproduction of that checked-in recipe. Sample variance and epsilon 1e-6 are
the framework's explicit CAPO numerical profile choices.

Both pinned backends already have GDPO normalization machinery. The framework
collapses Verifiers rewards to a single callback and does not expose that
capability. This narrows implementation toward contract and transport work.

The pinned veRL V1 path does not fetch/materialize GDPO components, and the
estimator's existing prompt/attention indexing assumptions do not match its
fetched batch. Merely adding gdpo to the manifest enum is insufficient.

TRL uses rollout-wise final normalization while the inspected veRL estimator
uses token-mask whitening. The official GDPO README also shows this difference.
Variable-length parity must be designed and tested, not assumed.

Precomputed TRL advantages bypass computed scalar advantages only after normal
reward statistics are calculated. Its dynamic filter uses those scalar
statistics. Keep filtering disabled initially; do not discard useful CAPO/GDPO
signals via the existing scalar admission rule.

The local fork checkouts do not match selected release revisions; veRL also has
unrelated uncommitted work. Pinned-object inspection was necessary.

## Decision Log

2026-09-10: AutomationBench exposes one whole-episode judge API. Rubric detail
is task-owned scorer policy and does not change trainer algorithm semantics, but
scope and response schema must be one inseparable versioned contract. Remove the
turn-level selector instead of keeping a compatibility default that can silently
pair the wrong prompt and schema. Preserve old turn evidence only in offline
audit readers; new jobs cannot select or emit it. Do not raise output budgets to
mask prompt/schema mismatch, and do not admit uniform perfect scores as GDPO
qualification merely because they are schema-valid.

2026-09-08: fix algorithm correctness first, then submit matched jobs, then
optimize harness execution. A dropped group must never be partially normalized,
assigned fabricated zero reward, or aligned to another task's metadata. For
single-process ordinary GRPO, preserve the scheduled accumulation shape with
zero-loss tensor slots created only after scoring retained evidence; rescale
the sequence-mean loss to the retained population. Reject unsupported partial
distributed or alternate fused-loss paths explicitly until qualified. Keep
OLMo's own active sampling rather than converting it into vanilla GRPO.

2026-09-08: Take the next CarbonTeq Verifiers baseline from current upstream
main rather than v0.3.1, but preserve CarbonTeq ownership and the narrow
host-client seam. Compatibility is judged by native v1 behavior and consumer
tests, not by retaining the old dependency graph. Therefore the AutomationBench
adapter no longer depends on `openai-agents` (whose `mcp<2` constraint conflicts
with Verifiers' MCP 2); it constructs callable schemas locally and declares its
actual runtime dependencies directly. Posttrain advances only to immutable
CarbonTeq commits, never to a moving upstream branch.

2026-09-08: Maintain `carbonteq-ai/verifiers` as an independent CarbonTeq
distribution, not a temporary fork whose lifecycle depends on upstream
acceptance. Verifiers continues to own environment and harness execution,
native Episode/Trace records and scorer-plugin behavior. Posttrain owns typed
selection/composition, exact consumer pins, rollout admission, algorithm credit,
job lifecycle and evidence projection. Review upstream changes periodically and
port or merge them by behavior; an upstream PR is optional and never a release
gate. Each update requires an explicit delta ledger, immutable CarbonTeq commit,
fork regressions, Posttrain train/eval compatibility, and packaged runtime
qualification before consumer pins advance.

2026-09-08: An exact hostname in an execution target is a placement constraint,
not merely an admission-lock label. A local provider must compare it with the
machine's canonical hostname and fail before container submission on mismatch;
remote-host execution goes through a scheduler-backed provider. This prevents a
96 GB target from silently running on the caller's 8 GB GPU. For the current
comparison, explicitly use dstack to reach `carbonteq-ai-workstation.lan` while
keeping the job builder local. Image construction and execution placement are
independent machine-level choices.

2026-09-08: Treat partial environment failure as prompt-group admission for all
group-relative GRPO-family objectives, not as reward zero and not as permission
to train an incomplete group. Preserve completed groups, reject every candidate
belonging to a group with one failed occurrence, and regenerate only that full
group under a bounded algorithm setting. Structured GDPO/CAPO additionally
validate reward evidence; scalar GRPO/OLMo validate rollout identity. For the
local comparison use two total admission attempts, eight groups by four
candidates, environment/vLLM concurrency 32, and an effective/global update
batch of 32. This bounds the replacement tail while keeping group normalization
and cross-arm batch shape correct.

2026-09-08: Use one explicit dynamic Secure Cloud on-demand fleet with
`nodes: 0..2` and let each Posttrain target retain the exact GPU constraint.
Do not create static A100 hosts or duplicate provider-specific job YAML. The
fleet automatically provisions matching workers for parallel jobs and returns
to zero when idle; provider no-capacity keeps the canonical jobs queued rather
than creating replacement runs.

2026-09-08: Run the two scalar arms, vanilla GRPO and OLMo 3, on RunPod Secure
Cloud A100 80 GB on-demand capacity. Use the same A100 tier for the GDPO policy
worker and a distinct RTX PRO 6000 96 GB on-demand service for Gemma 4 12B.
This is the smallest presently defensible production placement: 24 GB is below
the colocated 2.6B training shape, 48 GB remains unqualified at the exact 12k
context/update shape, and 80 GB costs less than the 96 GB tier while retaining
substantial margin from the corrected micro-batch-one configuration. Preserve
the L40 as a later measured cost-optimization gate rather than risking another
failed 50-update run.

2026-09-08: Do not launch judged GDPO by merely changing its judge target. Add
and qualify remote attached-service orchestration first, because target
provenance without a separate provider task would silently colocate the judge
and trainer. The service owner launches and tears down Gemma; the GDPO job only
receives a ready, scoped endpoint whose immutable inference identity matches the
selected binding.

2026-09-07: Production remote judged training attaches to an independently
owned inference service; it does not interpret a judge `InferenceBinding` as an
extra GPU reservation inside the training job. Local managed child processes
remain available only for explicitly qualified colocated/partitioned profiles.
The service owner proves model and engine provenance, readiness, capacity and
teardown; the consumer binds a volatile endpoint and scoped credential without
letting GDPO/CAPO or Verifiers own deployment. ADR 0019 records the durable
boundary. Do not launch the 12B Gemma GDPO arm until this attachment path has
live RunPod lifecycle evidence.

2026-09-07: Treat local-to-production portability as an invariant of the
work-package and actual-job packaging path, not as a second qualification script.
Local and RunPod runs must share source, algorithm settings, dependency closure
and task snapshot. An execution-target override is a declared selection change
and therefore produces a different content-addressed package identity; it must
not require host-path mounts or procedural YAML edits. Split the three-arm
comparison into independently recoverable train and
evaluation runs so a later arm or spot interruption cannot force completed work
to repeat. The legacy monolithic dstack YAML remains diagnostic history and is
not the production launch surface.

2026-09-07: Keep the production episode prompt domain-general and keep frozen
examples outside it. Tool contracts and observable trajectory evidence are
runtime inputs; benchmark assertions, native rewards, hidden state and reference
answers are not. Use historical native traces as the primary comparison corpus,
with synthetic examples only as targeted preflight controls. This changes the
external scorer plugin and qualification evidence, not Verifiers, trainer, or
GDPO algorithm APIs.

2026-09-07: Preserve rejected tool syntax as non-executable assistant content,
decoded from validated renderer spans, rather than silently dropping it or
repairing/executing it. Use one private helper for both trainers. This fixes
evidence fidelity without adding a scorer primitive or changing algorithm
equations. Parser coercion policy and richer environment error/retry semantics
are separate questions; this patch does not claim to resolve them. Enable
policy thinking for the reasoning-quality trial as requested by the user.

2026-09-06, R12 provisional judge selection: Use thinking-enabled Gemma 4 12B
for the first bounded managed-judge training integration. Both Nanbeige 3B and
Gemma passed the same 10 engineering checks only when judge thinking was enabled,
but Gemma generated 5,120 completion tokens versus Nanbeige's 31,009. This is a
cost/throughput selection for the next gate, not a general quality or benchmark
winner. Nanbeige remains a supported inference selection; DSpark is qualified
independently and must be compared on the reviewed fixture before any default
change.

2026-09-06, R12 live qualification: Do not expose DSpark+TurboQuant as a valid
catalog selection for the pinned Nanbeige runtime. Reject the combination when
the selected model and exact backend revision are translated to the typed engine,
and retain it in the qualification matrix as an
explicit unsupported cell with the original failure evidence. Continue to
qualify standard, TurboQuant, and DSpark independently. A future runtime may
re-enable composition only under a new backend revision and a passing live
compatibility result. Runtime probes now exercise both thinking-disabled direct
answers and thinking-enabled answers, retain the complete OpenAI message, and
require a normal `stop` plus visible answer content.

2026-09-06, R12 implementation: Represent a speculative draft as immutable Hub
repo/revision provenance plus an absolute runtime-local materialized path. Only
the path is sent to vLLM; provenance remains in the typed engine/evidence value.
Require DSpark to declare this draft, while MTP remains valid without an external
draft. Service sharing is explicit by name and never inferred from equal URLs or
models. Attached endpoints are probed but never shut down; managed services are
owned by the enclosing composition context.

2026-09-06, Revision 12: Following the user's explicit correction, DSpark and MTP
remain exclusively inference concerns. Separate inference qualification from
judge-quality comparison and deterministic algorithm correctness. Named service
dependencies belong to composition; task rubrics remain native Verifiers plugins.
Extend this existing plan instead of creating a competing scorer-service plan.
The initial comparison uses frozen trajectories and sequential isolated services;
remote multi-service orchestration is qualified explicitly, never inferred from
an `ExecutionTarget` field. This revision records planned work only.

2026-09-06, Revision 11 export follow-up: Generic LoRA export ownership remains
in `/home/hammad/projects/verl-gdpo-capo`, branch
`codex/verl-0.9-retained-features`, HEAD
`c4b2d65f90127f8b24c8e162797a0b23df9c2025`. Edit
`verl/model_merger/base_model_merger.py` and
`tests/model_merger/test_output_validation_on_cpu.py`, with the fork ledger and
framework consumer page updated together. Leaf-only `proj` inference widened
the restored adapter to an untrained vision Conv3d; preserve exact trained
module paths instead. Commit/publish this generic fix with the other fork delta
before consumer pin updates. Do not rewrite original run outputs; regenerate
from the retained checkpoint into `reexport-exact-targets`. The qualification
reload script verifies the native conditional-generation wrapper and exact
saved tensor equality, since generation alone can succeed after missing-key
warnings. No optimizer or reward semantics change.

2026-09-06, Revision 11: Separate algorithm correctness from semantic scorer
quality. The user explicitly approved deterministic qualification. The fixture
in `scripts/qualification/deterministic_turn_fixture.py` emits signed turn
inputs and none/alternating/all/last error patterns selected by stable rollout
identity. It preserves native task reward and token IDs, records synthetic
provenance and source digest, and uses existing bridge enrichment/projection
interfaces without production rubric changes. This does not amend the frozen
product baseline or qualify a reasoning-quality model. No judge endpoint is
needed. Identical reward groups and zero-gradient updates must remain visible,
not be hidden by replacement merely to produce attractive results.

2026-09-06, Revision 10: Runtime composition uses existing `InferenceBinding`
and `ServeLaunchRequest`, not a new scorer service or algorithm-specific judge
class. `bind_native_judges` owns endpoint lifetimes and serializable inference
provenance; native plugins own scoring. SAMPO optionally selects direct turn
rewards through the same composition seam while its existing sparse job contract
is unchanged. The example plugin supports prefix-only and retrospective panels,
strict complete coverage, bounded transport/validation failures, and distinct
abstained/inapplicable statuses. No mutable score cache is introduced: reassess
under a fresh namespace or trace; recovery rejects changed scorer/configuration.

Do not repeat failed training merely to obtain five successful steps. Qualify the
judge contract against explicit fixtures first, retain failures, and keep judge
quality distinct from numerical and token-provenance qualification. Creating a
maintained Verifiers distribution is now resolved: use
`https://github.com/carbonteq-ai/verifiers` as the independent CarbonTeq-owned
platform and exact tested commits as consumer pins. This decision does not by
itself publish a new package revision or close runtime qualification.

2026-09-06, Revision 9: keep direct turn rewards and turn-error labels separate
from token masks. The plugin emits IDs and annotations. The bridge independently
constructs the authoritative map; SAMPO selects one named local score, GDPO
selects an explicit reduction, and CAPO maps explicitly selected error IDs to
eligible policy spans. A low quality score is not automatically an error mark.
The native Episode remains replay authority, even if it failed before producing
any trace; trace views are derived artifacts. Stable pins remain unchanged until
the new forks, environment packages and final runtime qualification are complete.

Decision (2026-09-06, Revision 8 implementation): stage read-side compatibility
and test modern native APIs in an isolated locked runtime before changing writers
or pins. Preserve old artifacts and exclude non-trainable judge-agent traces from
SFT even when callers explicitly retain errored episodes. Use the existing local
artifact path/digest representation for toy retention; do not imply remote
publication from a locally checksummed archive.

Decision (2026-09-06, user direction / Revision 7): support assistant-turn rewards
first, with custom Verifiers judge/scoring plugins and injected inference.
Rubric dimensions are task-owned diagnostics or explicitly selected outputs,
not framework fields. This avoids coupling algorithm APIs to a particular judge.

Decision (2026-09-06): preserve each algorithm's numerical contract and name its
turn projection explicitly. Do not equate rewards with advantages or promise
SRPO reproduction from replacing steps with turns. Semantic within-turn
extraction and the larger pilot remain deferred.

Decision (2026-09-06): migrate native episode APIs before production scoring;
amend the frozen baseline before code. Qualify real packaged cells and judge
assessments separately, then follow immutable fork publication/promotion order.

2026-09-06: Use the idle local 8 GB GPU sequentially with a Qwen3.5-0.8B
BF16 LoRA policy and frozen Qwen3.5-2B judge. This is a qualification-only
project scorer composed as a native trace enricher, not an environment fork or
a production distributed judge service. Use the upgraded installed TRL post2
candidate with this checkout's train source; stable pins remain unchanged.
Audit partial credit, binary outcome, judge quality and assistant-token process
credit independently. Small-judge judgments require review and are not ground
truth. A five-step toy does not establish benchmark improvement or veRL support.

2026-09-06: The latest user instruction supersedes the interim Gemma judge
choice for this toy. Use Qwen/Qwen3.5-0.8B at
`2fc06364715b967f1860aea9cf38778875588b17` for policy and
Qwen/Qwen3.5-2B at `15852e8c16360a2fea060d615a32b45270f8a8fc`
for judge. Both are existing immutable catalog selections with local snapshots.
Five optimizer steps are a mechanism test, not evidence of improved task quality.
Benchmark choice is not silently changed: the user said AgentBench, while the
maintained environment here is AutomationBench; ask before binding that seat.

2026-09-06: If AutomationBench is confirmed, CAPO outcome must select its
`task_completed_correctly` metric, not scalar `partial_credit`. Its taskset
exposes these separately. GDPO can explicitly select partial credit and a
versioned judge-quality component without changing native task success.

2026-09-06: Settings know the logical batch but not the training rank count.
Validate divisibility in settings and global accumulation equality in the
composed TRL request. Initial structured TRL admission requires TP size 1:
Gloo failure agreement is verified, but tensor-parallel zero-work generation
participation is not. Reject larger TP before model loading.

2026-09-06: Keep environment/scorer reward meaning out of algorithm dispatch.
`RewardProjection` explicitly selects retained scalar/contribution/metric fields;
the bridge transports evidence and original token coordinates. Critique voting
is a pure resolver over retained step IDs, with explicit even-vote tie policy.
See ADR 0018. Async enrichers are awaited and terminal native evidence survives
annotation failure/cancellation; unavailable critique is never valid zero.

2026-09-06: veRL legacy `vanilla` includes dual clipping and legacy k3 clamps
the KL estimator. New algorithms select `token_clip` and `k3_unclipped` to
match the declared objective, leaving existing algorithms unchanged. Reward
admission is bounded independently of scalar-variance filtering, which would
incorrectly discard useful multi-component GDPO evidence.

2026-09-06: Multi-repository edit order remains fork source/tests/ledger commit
and push, immutable development publication, candidate consumer pins/locks,
packaged numerical and real four-cell lifecycle qualification, then stable
promotion. Do not move main pins to unpublished structured-reward fork changes.

2026-09-06: User explicitly changed immediate priority from adding algorithms
to upgrading the underlying fork bases while retaining fixes, features, and
precision improvements. No additional frozen product amendment is needed:
this changes implementation dependencies, not operation meaning. Integrate from
stable upstream tags in `codex/trl-1.12-retained-features` and
`codex/verl-0.9-retained-features`, merging the retained fork history to avoid
silently discarding deltas. Resolve overlapping corrections once, not twice.
The latest stable-base commit must be an ancestor of the final fork commit.
Existing GDPO/CAPO work in the framework stays untouched during this prerequisite.
Commit order is TRL fork and veRL fork, then retained releases/development index,
candidate framework/runtime locks and qualification, then stable promotion and
framework pins. Never describe a source-only merge as a finished runtime upgrade.

2026-09-06: Expanded the implementation review at the user's request to compare
the maintained TRL/veRL deltas with newer stable upstream releases and retain
useful correctness and numerical improvements. GitHub's release API reports
TRL v1.12.0 (2026-08-26), commit
59c4a8e104413fa9f4ca1a54eaf2ff93c0f299be, and veRL v0.9.0
(2026-08-14), commit 483b8a009ba3a97563edee3a19887e4862b8094a.
TRL's v1.12.0 release notes explicitly identify it as an accidental duplicate
of v1.11.0. These are review candidates, not updated framework pins.

The retained TRL delta includes bounded DAPO/OLMo sampling, precomputed token
advantages, exact-token IW-OPD hooks and masks, post-generation token counts,
finite-logprob/overflow diagnostics, actor/sampler parity, bounded vLLM waves,
LoRA synchronization, and chunked logprob projection. The retained veRL delta
includes SAMPO equations and V1 metadata/identity transport, bounded failed
group refill, teacher-logprob alignment through nested microbatches, repaired
3-D position IDs, and Qwen/LoRA/entropy/MTP runtime fixes. Evaluate each against
upstream code and regression tests before dropping or porting it.

Concrete upstream review targets: TRL e1b2e219 (generation-window versus
accumulation-window DAPO/CISPO/VESPO normalization), 6297c477 (Liger parity),
185f7382 (new KL bias-correction default), c285dc17 (sleep-mode weight sync),
67290a07 (native vLLM server migration); veRL 8a1bf6d5 (REINFORCE++ carries
returns through observation spans), 19e5347f (non-merged LoRA receiver names),
and the v0.9.0 distillation normalization/checkpoint changes. Do not apply
REINFORCE++ reward-to-go semantics to CAPO's direct token credit.

Pin `use_bias_correction_kl=False` explicitly in the TRL adapter to preserve the
existing sampled k3 KL gradient across upstream upgrades. Positive-beta parity
must test gradients, not only forward values: a differentiable importance
ratio equal to one can still change the KL gradient.

Source comparison correction: the pinned TRL already contains the
generation-window/accumulation-window DAPO normalization formula from upstream
e1b2e219, via the maintained lineage. Release-note presence is not evidence of
a missing fix. Its three-line formula was checked in the pinned
`grpo_trainer.py` around lines 3640–3644. Do not duplicate this patch.
Upstream v1.12.0 still lacks the fork's precomputed-advantage, bounded dynamic
sampling, raw-parity, and chunked-projection controls. veRL v0.9.0 still lacks
the fork's SAMPO implementation and V1 metadata fetch. A bare version bump
would remove required capabilities; candidate rebases must preserve and test
these behaviors before any framework pin changes.

2026-09-06: Implement published GDPO and CAPO as independent capabilities on both
backends. Reason: stress distinct framework contracts without inventing a SAMPO
variant or conflating research algorithms.

2026-09-06: Default GDPO parity to rollout-weighted final normalization and make
other variants explicit. Reason: variable-length backend semantics currently differ.

2026-09-06: Disable optional scalar-variance filtering initially. Reason:
faithful algorithm support does not require DAPO admission, and scalar collapse
can remove useful multi-reward/token signals.

2026-09-06: CAPO uses token-ratio clipping and direct token credit, not SAMPO
returns/sequence ratios. Reason: an implementation must preserve objective
meaning, not just reuse a compatible tensor shape.

2026-09-06: Keep the initial method reference separate from agent adaptation and
the combined research reward objective. Reason: capability qualification,
algorithm reproduction, and policy quality are different claims.

## Outcomes & Retrospective

2026-09-10 judge-contract checkpoint: the production-request reproducer and
three-pair input/output artifact make the failure inspectable without rerunning
training. The v0.4 external environment candidate removes the old runtime path
and passes its complete local package gate. Posttrain pin promotion and a fresh
GPU calibration remain; no optimizer qualification is claimed from prompt-only
replay.

2026-09-08 current continuation: code inspection established an unconsumed
retained-row contract and a reproducible event-loop bottleneck. Retained-row
correction is published/tested as post5; the harness bottleneck is deferred.
Acceptance for the correctness change is exact source
identity after dropping a nonprefix complete group, equivalent retained-sample
gradients at multiple accumulation sizes, real trainer updates, immutable fork
publication, and fresh GRPO/OLMo GPU submissions. Run fork
`tests/test_rollout_admission.py`, framework `test_reward_admission.py`,
`test_trl_online_rl.py`, and the affected API tests before packaging.

2026-09-08 placement checkpoint: no new GPU was allocated. The live backend is
confirmed Secure-Cloud-only and on-demand inventory is sufficient for the
planned split. R13 is terminal after two spot interruptions. Scalar training can
move to A100 on-demand after target addition and repackaging; judged GDPO still
requires the already-decided remote judge attachment path to be implemented and
qualified before deployment. This prevents a misleading single-worker launch
from being treated as the intended two-service architecture.

2026-09-07 portability checkpoint: R7 produced no optimizer checkpoint, but its
failure separated judge lifecycle from policy-training memory. RunPod currently
advertises available 96 GiB RTX PRO 6000 spot offers while another packaged job
successfully uses the same managed-storage target. The comparison is not yet
submitted there: first package the exact dirty AutomationBench/environment and
framework candidates into immutable source revisions and an actual-job image.
This is a deliberate correction to deployment fidelity, not a capacity-only
retry.

The external environment candidate now passes the exact six-package CI matrix:
Ruff, formatting, Pyright, package tests and wheel builds all pass (59 tests
passed, two optional tests skipped), and a clean combined installation activates
all six wheels. The framework adds retained runtime evidence for rollout-engine
sleep and GPU-memory-utilization selections; 57 focused Lab/train tests pass.
Publication remains gated on explicit commit/push authorization because the
environment repository forbids agents from publishing implicitly.

2026-09-07 focused correction: message projection and next-run thinking settings
are changed locally; recorded-failure replay passes using the remote installed
renderer. A fresh packaged GPU run remains necessary. Existing judge evidence
cannot be repaired retroactively by changing the adapter; preserve its original
assessment and perform a new run. Focused validation from the repository root:
`uv run pytest packages/train/tests/test_policy_messages.py packages/train/tests/test_trl_online_rl.py packages/train/tests/test_verl_backend.py -q`;
also run scoped Pyright, Ruff, and `uv run lint-imports`.

Revision 12 implementation checkpoint: The first architecture slice is present
in common/catalog, serve and jobs. Nanbeige has a distinct tokenizer-backed
conversation contract with explicit thinking modes and vLLM parser mapping;
DSpark and TurboQuant are orthogonal serving selections with an explicit composed
profile, and algorithms/judge rubrics do not import or inspect either. Named
inference services now separate lifecycle from plugin binding, support safe model
sharing, and preserve compatibility evidence. Policy, judge and draft
materialization no longer share generic input lookup. Local focused tests, scoped
type checking and import-boundary checks pass. No Nanbeige runtime, TurboQuant
correctness/performance, DSpark acceleration, judge-quality comparison, RTX PRO
6000 job, dependency pin, fork publication or release has been claimed.

Repository-wide Ruff, Pyright, all eight import contracts and `git diff --check`
also pass at this checkpoint. The focused common/catalog/serve/jobs suite reports
139 passed and 5 expected skips.

Revision 12 planning outcome: A production-intent implementation sequence now
covers inference-owned Nanbeige/DSpark, generic service binding, safe sharing,
resource/credential/artifact isolation and a separate 12B judge comparison.
No R12 runtime was installed, no source code or stable pin changed, and no GPU
job launched. Exact vendor/model revisions, RTX PRO 6000 capacity and frozen
comparison thresholds are explicit pre-execution gates. R11 results below remain
historical measured evidence, not Nanbeige qualification.

Revision 11 live results: GDPO gradient norms are
`[1.8306781054, 0, 0.5686625242, 2.5349743366, 0]`; CAPO norms are
`[2.4023716450, 0.5578591228, 1.4519388676, 0.8970183134, 0]`.
`audit_verl_deterministic.py` independently reconstructs raw rewards from native
turns and compares Decimal expected advantage mean/min/max against all five
native optimizer-step metrics. This is aggregate evidence, not per-token actor
gradient proof. CAPO's one truncated group is excluded only under the retained
`max_turns` stop condition and listed explicitly in the audit. It is not hidden
as a successful trajectory. Both regenerated adapter exports pass exact key and
tensor equality and CPU generation (`ready.`); LoRA-B norms are 0.0343524479
and 0.0344342612. The exporter source SHA-256 is
`b4c5ddf49848a53be348bfe65a2f78741ce455ab32f91d1cde288d09ef69780d`.
No stable dependency pin or fork publication changed. The existing step-2/4/5
checkpoints are retained remotely, but no resumed training equivalence claim is
made. Complete that gate, deterministic TRL parity, explicit/sparse SAMPO,
distributed/per-token evidence and package/OCI promotion before release.

Retained local bundle:
`outputs/qualification/automationbench-2026-09-06/deterministic-r11-evidence.tar.gz`,
SHA-256 `fd118ab693cb9d5be2342a7cc6691346d28a5490c21993401f02d35a2e22cd0e`.
It contains successful and unsuccessful attempt evidence, original and repaired
adapters, audit/reload reports, candidate framework/fixture wheels and the exact
exporter source. Base weights and recovery checkpoints are excluded from this
lightweight archive; full runs remain at
`/home/carbonteq-ai-workstation/posttrain-qualification.2iyErl` on the workstation.
All task-owned GPU processes exited; only the three pre-existing desktop
processes remain. No worktrees or user artifacts were deleted.

Revision 11 in progress: the algorithm/judge gate conflation is corrected.
`scripts/qualification/automationbench_deterministic.py` runs the 0.8B policy
through real AutomationBench with independent synthetic process inputs. The
qualification-only fixture wheel must be installed in the external veRL
interpreter so serialized enrichers resolve without a source-path bypass.
Use a fresh output directory per attempt and retain unsuccessful attempts.
From the repository root, validate with:

    PYTHONPATH=scripts/qualification:packages/train/src:packages/common/src:packages/environment/src /tmp/posttrain-fork-upgrade.ouJ7s4/.venv/bin/python -m pytest scripts/qualification/test_deterministic_turn_fixture.py packages/train/tests/test_turn_rewards.py -q

After installing current candidate wheels on the documented workstation, run
`automationbench_deterministic.py --backend verl --algorithm gdpo --verl-source
<task>/verl-source --output <fresh-directory>` with its documented interpreter,
PATH and public model cache. Repeat for CAPO and TRL. Completion requires five
observed updates, independently checked numerical/mask evidence and real
resume/export checks; fixture tests alone are insufficient. LLM calibration,
multi-device evidence, all-environment migration and OCI/release gates stay open.

Revision 10 verification so far: 66 focused jobs/turn/recovery tests pass; full
framework Ruff and Pyright pass and all eight import boundaries remain intact.
Broader candidate train/jobs/environment/data/eval suite: 427 passed, 10 skipped,
one expected pin mismatch (installed TRL 1.12.0.post2 versus stable main
1.9.2.post11). Do not weaken that assertion before promotion. veRL structured
numerics/recovery/replay suite: 57 passed. Modern train bridge integrations:
five passed (including deliberate failure retention); native Verifiers host-client
interception: three passed. External AutomationBench: 15 passed, Ruff/format and
Pyright pass after migration type fixes. These are source/candidate checks,
not a final locked/OCI release suite.

Failed managed attempts, their saved step-2 checkpoint, both calibration fixtures
and raw assessments, executed scripts and the installed candidate wheel set are
retained at
`outputs/qualification/automationbench-2026-09-06/retained-managed-r10.tar.gz`,
SHA-256 `97260da4c7cc4864b26adcb75c2c31435a5811b6a6453e289cfdea5523e38d79`.
The revised judge module is identified by source Git blob
`59dd00e20b42ee83d2e76f845d339185e51b3bc7`. The qualification runner now also
verifies the declared source blob against the installed module and saves its
bytes before future runs; the earlier failed experiments predate this guard.
The fixture remains an engineering regression set, not independent calibration.

The two modern native-Episode **prototype** TRL runs still provide five-step
Decimal/mask/export evidence; managed production-composition attempts do not yet
close that gate. Judge qualification, four managed algorithm/backend cells,
explicit/sparse SAMPO GPU coverage, full migration/clean installation, publication
and stable pin/release gates remain open. No release-complete claim is justified.

Revision 9 execution evidence: GDPO and CAPO turn-rating runs at
`/tmp/automationbench-{gdpo,capo}-turns-20260906-a` completed five optimizer
updates each and passed `audit_structured_toy.py --reload-export`. GDPO retained
18 traces / 10 admitted attempts; CAPO retained 13 / 10. CAPO had five nonzero
gradient norms and one native successful task among the ten admitted attempts.
GDPO had one zero-gradient update; the other updates and changed adapter prove
the path was exercised, not benchmark improvement. These are development
results, not veRL or modern packaged runtime qualification.

The local retention bundles are
`outputs/qualification/automationbench-2026-09-06/gdpo-turns-a.tar.gz`
(SHA-256 `27ee95765efba216fe798efe4b0550a02a73353388011eb1be0b248bde7fe05e`)
and `capo-turns-a.tar.gz`
(`4e2113aa9cceba3178913da7e07e4788ad267d8036dbacd8dae981ce74d638ea`).
Each contains its selection, native traces, observations, checkpoints, export
and independent `audit.json`.

Revision 23 supersedes the former uncommitted v0.3.1 worktrees described below.
The current isolated fork worktree is `/home/hammad/projects/verifiers`, branch
`codex/carbonteq-verifiers-latest`, published at
`5304495a246e174683f5932377703e9a0a4a6926`. The current environments worktree
is `/home/hammad/projects/verifiers-environments`, branch
`codex/verifiers-latest-support`, published at
`b14dfe0ba9d60184f36d78786a543242fabfb765`. This environment revision aligns
all six standalone packages on the selected Verifiers fork and passes a
combined wheel activation gate. Neither development branch is a release tag;
package artifacts and the complete release-readiness receipt remain explicit
gates.

Modern integration command from the framework root uses the upstream-locked
Python interpreter at
`/tmp/posttrain-verifiers-031.AZRgVw/source/.venv/bin/python` and PYTHONPATH
entries for the new Verifiers fork, external AutomationBench src, framework
train/data/environment/common/eval src. Run
`uv run --no-project --python <interpreter> python -m pytest packages/train/tests/test_verifiers_modern_bridge.py packages/eval/tests/test_api.py -k modern -q`.
The legacy candidate interpreter at
`/tmp/posttrain-fork-upgrade.ouJ7s4/.venv/bin/python`, with framework src
overlays, passes 110 eval/turn/projection/bridge/SAMPO regressions before the
last additional CAPO cases. The native fork client-factory E2E subset passed
three cases; external AutomationBench suite passed eleven. Eight import
contracts remain kept. Re-run focused tests and type/lint checks after final
edits; no whole-repository or release qualification is claimed here.

### Revision 8 — first implementation slice verified

CAPO's missing independent audit is closed and both toy bundles are retained
outside temporary storage. The canonical boundary is amended, Episode-to-SFT
reader support is implemented and exercised against the real released native
API, and modern failure standing is preserved in shared facts. No stable pins,
live writers or algorithm losses changed. Continue with mixed-version replay,
train/eval live migration and external AutomationBench compatibility in R7.2;
then implement R7.3–R7.5. This is not completed production turn-reward support or
a release. Prior Revision 7 status below is historical.

### Revision 7 — planned continuation, not implementation completion

The plan now includes native turn evidence, external custom judge/scoring plugins,
modern Verifiers migration, algorithm-specific mappings and staged qualification
through release. Existing numerical implementations and toy evidence remain
useful foundations. The immediate next execution step is CAPO audit/export reload,
followed by the narrow baseline amendment and migration. Production turn scoring,
judge calibration, veRL live cells and stable promotion are not yet complete.

### Revision 6 AutomationBench execution

User confirmed the benchmark and authorized the run. Retain each attempt under
`/tmp/automationbench-gdpo-toy-20260906-{a,b,c,d,e}` and its adjacent `.log`;
do not overwrite failed evidence. Reproduce from the repository root using:

    HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/home/hammad/projects/rl/packages/train/src /tmp/posttrain-fork-upgrade.ouJ7s4/.venv/bin/python scripts/qualification/automationbench_structured_toy.py --algorithm gdpo --output /tmp/automationbench-gdpo-toy-NEW
    HF_HUB_OFFLINE=1 PYTHONPATH=/home/hammad/projects/rl/packages/train/src /tmp/posttrain-fork-upgrade.ouJ7s4/.venv/bin/python scripts/qualification/audit_structured_toy.py /tmp/automationbench-gdpo-toy-NEW

Repeat sequentially with `--algorithm capo` and a fresh output directory.
The candidate environment additionally installs only the pinned AutomationBench
plugin with `uv pip install --no-deps` from the external repository/subdirectory.
No credentials or external service calls are needed for the cached local models
and native simulated tool tasks. Re-run recovery/projection tests: nine passed;
targeted Pyright has zero errors. Completed-run audit receipts are still pending.

### Revision 5 toy-run preparation

The model sizes and five-step limit are recorded; execution has not started.
Actual-job packaging now selects the online-RL profile for both new job kinds.
The next gate is explicit benchmark identity, followed by bounded judge
composition and runtime preflight. Preserve native partial credit, binary
success and judge assessment as separate evidence. Recompute advantages from
retained raw rewards with the independent oracle; inspect sampled versus
observation masks and check finite gradients plus changed adapter tensors.
Require informative rewards/advantages for an algorithm-learning claim. If the
toy emits only zero-variance evidence, retain it and report that the learning
mechanism was not exercised instead of filtering until a desired result appears.

### Revision 4 implementation handoff

Implementation is locally tested, not fully qualified or published. Main's
stable dependency pins are unchanged. veRL's new structured source and both
fork ledger edits are uncommitted follow-on work; do not reuse the post1
version for a new build. Candidate runtime upgrade build `34007498425`
succeeded independently of these later algorithm edits.

Repeat framework checks with upgraded dependencies without replacing main's
environment:

    PYTHONPATH=/home/hammad/projects/rl/packages/train/src:/home/hammad/projects/rl/packages/jobs/src:/home/hammad/projects/rl/packages/work/src /tmp/posttrain-fork-upgrade.ouJ7s4/.venv/bin/python -m pytest packages/train/tests packages/jobs/tests packages/work/tests -m 'not gpu and not integration' -k 'not test_trl_is_installed_from_the_pinned_release' -q

CUDA receipts: `docs/tooling/trl/evidence/2026-09-06/gdpo-cuda-fixture.json`
and `capo-cuda-fixture.json`. Reproducer:
`scripts/qualification/structured_trl_lifecycle.py`, with `--algorithm` and a
fresh `--output` directory. Never overwrite an existing qualification output.

Remaining: freeze/integrate the actual CAPO judge, retain scorer configuration
and complete panels, qualify standard real Verifiers jobs and native veRL GPU
optimizer/resume/export, distributed generation/loss parity, scorer-change
recovery refusal, and packaged fork publication/pin promotion. The tiny TRL
fixture verifies optimizer plumbing, not this whole product path. Earlier
outcome entries below are historical, not the current implementation status.

Revision 3 latest status: framework candidate `0a356ad4` on
`codex/trl-verl-stable-base-upgrade` selects TRL `1.12.0.post2` and veRL
`0.9.0.post1`. Post2 source is `95a787b6c04f91a5d485fd827d31b1e1fb67ae8e`,
wheel hash `8b19cd7fad5a28cf9ced4a7fac93bfaaedaac154faf019a3d3dd6abeaaa45c26`,
sdist hash `0f6407d8280d59b433ef28ed5d007ddde95bb064cd28f12661fa8e59ef2e67c9`.
Its development publisher `34007394648` succeeded with byte readback. Candidate
framework validation: 262 passed, 3 skipped, 1 deselected; train Pyright zero
errors, eight import boundaries kept, Ruff and diff checks pass. Installed-wheel
CUDA lifecycle receipts live in the candidate's
`docs/tooling/trl/evidence/2026-09-06/`; the native tiny-fixture gate is explicitly
not vLLM/LoRA/veRL qualification. Corrected image build `34007498425` is running,
framework label `0.3.26.dev2026090602`. Retrieve its generated manifest, locks,
and receipts only after success; run real rollout/training lifecycle gates before
stable promotion/main adoption. Root main still preserves GDPO/CAPO WIP.

Revision 3 initial upgrade snapshot (superseded by post2 above): underlying fork source upgrades are complete
and retained GitHub/development-index candidates exist, not just backports on
old versions. TRL `1.12.0.post1` is tagged at
`6a5532e2f51e4e1cdc8a891582514a50f68a775a`; veRL `0.9.0.post1` at
`cec7e74c361bb973b641db8dfbb75a5544c33139`. Both publication workflows succeeded
with clean installs and byte readback. Framework candidate branch
`codex/trl-verl-stable-base-upgrade` has commits `fde973cc` and `06b301ea`,
while root main keeps unfinished algorithm work separate. Candidate training
tests: 209 passed, 3 skipped after installing the pinned GSM8K environment;
Pyright has zero errors and all eight import boundaries pass. Runtime build
retry `34006619287` was cancelled after the checkpoint bug was found.
Image qualification, GPU lifecycle runs, stable promotion and main pin adoption
are still open. The historical implementation snapshot below predates this
upgrade slice and must not be read as its current publication status.

Implementation is in progress on `main` at the user's requested checkout.
Canonical amendments, structured settings/requests, typed reward evidence,
complete-population numerical constructors, initial TRL precomputed-advantage
wiring, native Verifiers component projection, and veRL launch contracts exist.
The shared numerical/evidence and adapter tests pass, but this is not four-cell
qualification. CAPO critic selection/alignment/admission, veRL worker/V1
transport, distributed population recovery, public job/catalog composition,
real optimizer/checkpoint/export tests, and final GPU runs remain open.

At the user's request, newer stable releases were audited. Two correctness
improvements are implemented in isolated sibling fork worktrees rather than
mixed into unrelated local fork work. TRL's Liger 0.8.0 token-loss path now
uses the same generation-window normalization as the ordinary path: 19 tests
pass, including native Liger gradients and two-rank Gloo parity. The installed
pin fails 14 of the first 18 cases (four unchanged GRPO cases pass). veRL's
REINFORCE++ now preserves action-time returns across masked observations:
28 core/regression tests pass. These candidates are not published or selected
by framework pins. No full version upgrade, GPU launch, or publication occurred.

Fork worktrees and safe continuation:

- `/home/hammad/projects/trl-gdpo-capo`, branch
  `codex/gdpo-capo-upstream-parity`, base `69cf80a7319079ec5523841553467e119ebc1cec`.
- `/home/hammad/projects/verl-gdpo-capo`, branch `codex/gdpo-capo-support`,
  base `808923d487aa2c524fda02cf5289110541b4221f`.
- The earlier TRL branch and dirty `verl-upstream` checkout are unchanged.
- The existing framework worktrees were removed only after retaining branches
  and archiving their release-readiness files under
  `/home/hammad/projects/worktree-archives/2026-09-06/`.

Repeat source checks from `/home/hammad/projects/rl`:

    PYTHONPATH=/home/hammad/projects/trl-gdpo-capo .venv/bin/python -m pytest /home/hammad/projects/trl-gdpo-capo/tests/test_liger_window_normalization.py -q

The temporary veRL CPU harness at `/tmp/posttrain-verl-review-20260906`
isolates OmegaConf's ANTLR 4.9 runtime from Verifiers' ANTLR 4.13 requirement;
do not install the two stacks into one production environment. Its source test
command is:

    PYTHONPATH=/tmp/posttrain-verl-review-20260906/lib/python3.13/site-packages:/home/hammad/projects/verl-gdpo-capo:/home/hammad/projects/rl/.venv/lib/python3.13/site-packages /tmp/posttrain-verl-review-20260906/bin/python -m pytest /home/hammad/projects/verl-gdpo-capo/tests/trainer/ppo/test_core_algos_on_cpu.py /home/hammad/projects/verl-gdpo-capo/tests/trainer/ppo/test_reinforce_observation_credit_on_cpu.py -q

This temporary CPU harness is not an isolated GPU runtime lock or a publication
receipt. Rebuild the declared runtime and run the existing release gates before
changing consumer pins. Porting stable upstream is still open; blindly updating
either version would remove required fork capabilities.

Revision note: initial plan reflects immutable-source inspection and explicitly
supersedes the earlier suggestion of inventing a second diagnostic SAMPO-like
estimator. It does not supersede the pilot's two-channel research objective.

Revision 5 note (2026-09-06): Record the user's reduced 0.8B policy / 2B judge
and five-step toy scope, verified model snapshots and fleet availability,
correct GDPO/CAPO job packaging, and separate binary success from partial credit.
Benchmark identity remains a required choice before starting the run.

Revision 7 note (2026-09-06): Extend the existing execution plan following the
user's turn-reward decision. Keep rubrics inside custom Verifiers judge/scoring
plugins, separate inference composition from reward interpretation and algorithm
credit, migrate native episodes before production integration, and require
independent audits, calibration and packaged qualification before release.
Supersede finer-than-turn segmentation as a first-release requirement and
distinguish completed GDPO evidence from pending CAPO audit and backend gates.

Revision 8 note (2026-09-06): Begin implementation with independent CAPO audit,
local checksum-addressed retention, canonical/ADR amendment and a tested native
episode data-reader slice. Record the isolated dependency requirement and modern
execution-failure fix; keep live migration, turn-scoring and release gates open.

Revision 9 note (2026-09-06): Record the two additional five-step turn-rating
runs and retained audits, generic turn mapping and SAMPO/CAPO/GDPO projections,
the demonstrated native client-injection gap, live train/eval migration and the
external AutomationBench registration fix. Keep candidate code and endpoint
plugin tests distinct from final packaged runtime and release qualification.

Revision 11 note (2026-09-06): Correct the R10 training pause: deterministic
fixtures qualify algorithm correctness, while LLM calibration qualifies semantic
scoring separately. Add the qualification-only runner, source-addressed fixture,
tests and explicit unfinished live-run gates; no production rubric or algorithm
semantics change.

Revision 11 completion note: Added two actual veRL five-update results,
independent aggregate/input audits, retained failure evidence and exact corrected
export reloads. Recorded the sampling-null and telemetry fixes in the framework
and the generic exact-module export fix in the external veRL fork. Resume,
remaining matrix and release gates are explicitly unfinished.

Revision 12 note (2026-09-06): Extend the same plan after critical source review
and the user's inference-boundary correction. Keep DSpark/MTP configuration,
draft artifacts and acceleration qualification inside inference; add generic
managed/attached service binding, lifecycle and per-role artifact isolation in
composition. Plan frozen-trajectory Nanbeige/12B judge comparison separately from
algorithm correctness. Preserve R11 evidence and unresolved release gates; this
revision changes documentation only and authorizes no deployment or publication.

Revision 13 note (2026-09-07): Record the R7 pre-update OOM, prove that no Gemma
judge service ran during scalar GRPO, reduce only the training micro-batch, and
replace the workstation-mounted monolithic comparison with independently
packaged train/eval jobs derived from one portable work package for local dstack
or RunPod spot. Target-specific package identities are expected; host-path mounts
and procedural recipe drift are not.

Revision 18 note (2026-09-08): Replace the repeatedly interrupted spot launch
with a split Secure Cloud on-demand topology. Inventory current RunPod capacity,
select A100 80 GB for each LFM policy/trainer arm and RTX PRO 6000 96 GB for the
Gemma judge, and make remote attached-service orchestration an explicit gate.
No deployment was performed while revising this plan.

Revision 24 note (2026-09-08): publish post5 complete-group admission and
retained-gradient correction, record the distinct R7 raw-policy parity failure,
and submit R8 GRPO/OLMo qualification with that gate intact. The user deferred
Verifiers harness optimization until after these correctness changes and runs.
