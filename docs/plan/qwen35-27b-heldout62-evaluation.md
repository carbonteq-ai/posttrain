# Evaluate Qwen3.5-27B before using it for training-data generation

## Purpose / Big Picture

The user explicitly requested Qwen3.5-27B evaluation before considering it as a
local demonstration generator. Run a native tool-continuation smoke and then the
frozen heldout62 benchmark on the existing96GB workstation. Do not generate SFT
data or train any model as part of this baseline. This is separate from the
closed ten SDPO ablations and the deferred LFM20 experiment.

The frozen framework baseline is unchanged. This plan follows docs/templates/PLAN.md
and the canonical product documents under docs/post-training. New model/inference
selections belong to the lab overlay; project-owned aliases compose existing
serve/eval jobs without modifying reusable package boundaries or dependency pins.
Preserve the dirty worktree. No commits, pushes or scoring-fix publication.

## Progress

- [x] User cancelled further27B evaluation in favor of4B/9B SFT generation.
  Framework cancellation intent persisted; GPU process exited. Exact owned
  container72ab23daf38d then stopped with15second grace, exit137, OOMKilled=false.
  Preserve as USER-INTERRUPTED, not a complete benchmark or model failure.
  Snapshot82 complete native episode rows copied before cancellation to
  docs/research/evidence/qwen35-27b-20261018/interrupted-r1/traces-before-stop.jsonl.
  This is a partial snapshot, not proof of final executed episode count. Runtime
  scratch was already removed during shutdown; retained Trackio remains source.
  Do not restart this run or cancelled observable queue automatically.
- [x]2026-10-18: Direct workstation inspection confirms idle GPU/no active container.
- [x] Exact cached official Qwen/Qwen3.5-27B revision
  fc05daec18b0a78c049392ed2e771dde82bdf654 verified; all11 shards are present,
  55,562,872,800 weight bytes,27,781,427,952 tensor parameters, BF16/F32 tensors.
- [x] Add pinned model/inference overlay, smoke/evaluation packages and classified
  project gates. Eight admission slots match eight server sequences.
- [x] Both read-only job plans pass. Three existing entry/native-smoke tests,
  Ruff and diff checks pass. Static GDN batch-invariance warning retained;
  configured invariant mode is not27B empirical reproducibility qualification.
- [x] Smoke submitted as qwen35-27b-heldout62-smoke-20261018-r1,
  provider pt-f3faab8ac317e46dfb19b0a5. Actual image
  registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:b0df17c99ea7c88d74378f9e8dbcd0529f822621b817300e09c0d323b6564a40.
  First direct Docker observation0a60ce9c7094, running startup; no model outcome
  or GPU generation claimed. Never duplicate this accepted identity.
- [x] Publish current immutable LAN packages and retain exact image receipts.
- [x] R2 native smoke succeeded and artifact review qualified transport:
  lookup_code({"key":"readiness"}), 83 output tokens with separated reasoning;
  continuation returned READY-2047 in final content, 61 tokens, no extra calls
  or leaked think markers. Trackio run d2c5a78c3faa4e50bbf23301d1121ef1;
  original generation bundles/server log saved under local evidence/smoke-r2.
- [x] Smoke r1 failed before model load because VllmEngineConfig rejected the
  block_size engine knob. Failure log retained locally; zero model generations.
  Added optional positive-integer block_size to the owned serve profile and
  both CLI/embedded translations, preserving defaults.33 focused tests pass,
  scoped Pyright reports0 errors, Ruff passes. Fresh local framework wheels
  built under /tmp/qwen27-framework-wheels; every replacement pack/run must
  use --framework-wheelhouse /tmp/qwen27-framework-wheels so it contains the fix.
- [x] R2 smoke accepted and completed under new run identity
  qwen35-27b-heldout62-smoke-20261018-r2; inspect /tmp/qwen35-27b-smoke-r2-submit.log
  and retained state before any retry. Initial full-eval pack also failed during
  CPU resolution because pypi.lan was unresolved, with no submission. Verified
  pypi.lan=192.168.110.53 from the workstation and added it only to the scoped
  CLI hosts overlay. No environment/task/model/dependency-pin changes.
- [x] R2 provider pt-584ac45d7c200bfff93110d8 completed successfully; image
  sha256:f45242ed54d015bd4cbe074a992afb4bf849f41c117c29af468f0224b78302a3.
  Actual model GPU memory observed 52194MiB. Full-eval image published:
  sha256:d2ca1b3d7e584fe20d27ddfe83bd9032abde18a56b6f654dc9a78f811c124a10.
  Temporary marked BuildKit hosts line removed after publication; system hosts
  and DNS were never modified.
- [x] Full evaluation accepted as qwen35-27b-heldout62-20261018-r1,
  provider pt-7baaaedecc3f76db86edbdb3,
  after successful smoke review and direct idle GPU/container check. Inspect
  /tmp/qwen35-27b-eval-r1-submit.log and retained execution before any retry.
- [ ] Reconcile310 native evaluation episodes and report success/tool errors,
  preservation, coverage and paired uncertainty versus compatible baselines.

## Context and Orientation

Worktree: /home/hammad/projects/worktrees/rl-rft, branch codex/rft-warm-start.
Model catalog: apps/lab/.posttrain/catalog/qwen35-27b-heldout62.yaml.
Work packages: apps/lab/.posttrain/work_packages/qwen35_27b_native_tool_smoke.yaml
and qwen35_27b_heldout62_eval.yaml. apps/lab/src/posttrain_lab/entry.py adds
serve/native-tool-continuation@1 and eval/workstation-managed-general-c8@1,
aliases of existing native smoke and managed-general evaluation operations.
EvaluationBudget(max_concurrent=8) limits admitted episodes without changing
the environment source, tasks, sampling or episode token/turn/time budgets.

Use the existing qwen3.5-tools-thinking@1 renderer, qwen3 reasoning parser and
native Qwen tool parser selected by the renderer. Full-precision weights, text-only
serving, max_model_len65536, max_num_seqs8, max_num_batched_tokens8192,
batch-invariant mode and block_size128 on selected vLLM0.29.1.dev4. Qwen thinking
sampling is temperature1.0/top_p.95/top_k20, with8192 tokens per reply. Each
episode has20turns/32768 generated tokens;62tasksx5attempts=310episodes.

Frozen environment a0c7378b179e2b267382b8125192876502d82f89 and ordered names
digest ed81cf1fe88f68bb8420bb46c5fdf3afde24660c7043812da680bb1c7bc9196f
remain unchanged. This population has been inspected repeatedly already and
must not be described as pristine new validation. It remains excluded from
training/data generation. Known frozen LinkedIn finalization errors remain
explicit missingness; do not fix the environment, drop tasks or zero-fill scores.

## Decision Log

Decision2026-10-18: user clarified Qwen3.5-27B. Pin the complete cached official
revision, also verified on the official Hugging Face repository, rather than
silently fetching a newer state. BF16 is feasible in the96GB workstation by
weight size; actual memory and native transport still require live qualification.

Decision2026-10-18: use8 admitted episodes and8 server slots. Earlier24B evaluation
with64 admissions/8 server slots hit rollout timeouts from queueing. Do not
repeat that configuration or compare throughput to higher-concurrency baselines.
Historical Qwen4B thinking sampling is comparable in tasks/budgets, but runtime
differences and scoring coverage must be disclosed. Recent greedy/non-thinking
student checkpoints are not matched sampling/mode controls.

## Concrete Steps

From the worktree, read-only planning has no registry/provider override:

    .venv/bin/posttrain --project-root apps/lab job plan qwen35_27b_native_tool_smoke.yaml --job smoke
    .venv/bin/posttrain --project-root apps/lab job plan qwen35_27b_heldout62_eval.yaml --job evaluate

Check active packaging processes and retained executions before building or
submitting. Use the protected project configuration; never print credentials.
All pack/run invocations use --registry registry.lan/carbonteq/posttrain-lab.
The public registry routes through Cloudflare and previously rejected uploads.

    .venv/bin/posttrain --project-root apps/lab job run qwen35_27b_native_tool_smoke.yaml --job smoke --provider dstack --registry registry.lan/carbonteq/posttrain-lab --framework-wheelhouse /tmp/qwen27-framework-wheels --run-id qwen35-27b-heldout62-smoke-20261018-r2 --timeout-seconds 3600

Only after provider success plus retained tool-call/result/final-content review:

    .venv/bin/posttrain --project-root apps/lab job run qwen35_27b_heldout62_eval.yaml --job evaluate --provider dstack --registry registry.lan/carbonteq/posttrain-lab --framework-wheelhouse /tmp/qwen27-framework-wheels --run-id qwen35-27b-heldout62-20261018-r1 --timeout-seconds 86400

Prefix these commands with the process-scoped bwrap resolver invocation described
below while local WARP DNS remains broken. These identities are retained history;
inspect accepted state instead of executing them again.

SSH: timeout25 ssh -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes
-o IdentityAgent=none -o GSSAPIAuthentication=no -o ConnectTimeout=8
-i /home/hammad/projects/ai-infra/.state/ssh/dstack_worker dstack@192.168.110.12.
Inspect scheduler, exact owned container and real GPU compute before every launch.
Never interrupt/compete with unrelated workloads or enable paid cloud capacity.

## Validation and Acceptance

Both packages must compose strictly with correct model pin, renderer, sampling,
target and frozen environment. Run applicable native-smoke/composition tests and
git diff --check; unrelated pre-existing unclassified inventory debt is not a
reason to modify other packages. A smoke must actually load BF16 weights, parse
lookup_code(key=readiness), consume its synthetic result and return the code in
final content with separated reasoning and no leaked markers. Scalar provider
success is insufficient without artifact inspection.

Full-evaluation acceptance requires62 exact task identities,5episodes/task,
310 unique native episodes, original traces and deduplicated native assessments.
Report strict coverage, provider/scoring errors, authoritative retries where
available, truncation/limits, task success, partial progress and tool mistakes.
Initially satisfied native checks are a preservation proxy, not a complete
independent harm taxonomy. Paired task bootstrap95% delta intervals must be
separate from absolute uncertainty and missing-score identification bounds.
No score implies missing, not failure zero. Strong baseline performance alone
does not qualify its generated demonstrations for the SFT corpus.

## Idempotence and Recovery

Retained framework/provider IDs are authoritative. Never duplicate an accepted
run or rebuild/resubmit merely because monitoring times out. Preserve failures
and their exact image/source receipts. An evidence-backed configuration repair
gets a fresh suffix; do not change quantization, tasks or budgets silently.
Do not edit running source or change external dependency pins.

## Surprises & Discoveries

Local Cloudflare WARP resolvers failed registry.lan. CLI requests work inside a
process-scoped bwrap namespace with /tmp/qwen27-lan-hosts bound over /etc/hosts
and --dev-bind /dev /dev. The CPU BuildKit container also needed a temporary
192.168.110.53 registry.lan entry for publication. Its original hosts file is
saved under local evidence/builder-hosts-before.txt; remove only the marked
qwen27-evaluation-temporary line after both packages publish. System DNS/hosts,
TLS verification and unrelated services remain unchanged. Initial attempts failed
before submission. Two smoke identities were subsequently accepted: r1 failed
before load and r2 qualified. Preserve both attempts separately.

The official weights were already cached, so no download is needed. The current
smoke implementation is model-independent except an opt-in LFM parser preflight;
it can be reused through a clearly named generic job alias for Qwen.

## Outcomes & Retrospective

Native BF16 Qwen3.5-27B tool transport is qualified by successful r2 artifacts.
Full benchmark is accepted; no task-quality result is available yet.
The initial serving-config failure is preserved separately from model evidence.

### Interim four-episode failure analysis

Follow-up2026-10-18: user authorized full62-task observability audit and local
repairs. See docs/plan/heldout62-observability-audit.md. Repaired counterpart
qwen35-27b-heldout62-observable-20261018-r1 is accepted under provider
pt-66e62a0ecce949676f225c4d, separately labelled and queued behind this original
run. Preserve original frozen results; do not pool or rescore them.

User requested inspection while evaluation remains running. Four scored failures
were chosen, one per first available finance task type, not a representative
sample. Original native episodes and SHA256 index are preserved in evidence/
qwen35-27b-heldout62-20261018/failure-analysis. All four ended agent_completed,
with no native episode errors and no length-limited call; this is not evidence
of a generation-budget bottleneck. Tool validity is distinct from business correctness.

- Sales tax, episode 7a22f641552d425bb3b9441edc20a204: zero tool mistakes,
  3/4 assertions, native partial_credit .75. Reads all spreadsheet revenue/rates
  and sends NY $22,800, mathematically correct for $285,000 at8%. Scorer expects
  $16,800 after $75,000 exemption, evidenced by initial Gmail notice
  msg_exempt_notice. Native tools expose only Sheets/Drive and Gmail send;
  there is no email reader/search. The exemption is absent from returned sheet
  data and student prompt. This is a concrete information-access defect, not
  demonstrated arithmetic failure. Frozen score remains unchanged; do not train
  on hidden answer leakage or silently rescore this arm.
- Billing, episode 967dc6083c844e36b72ff2b70d728802: zero tool mistakes,
  8/10 assertions, partial .8. Correct billable-hours arithmetic:27,000/18,000/
  12,500 and differences7,000/-2,000/0. But emailed and summarized Juniper
  Paused Pilot in a skipped-project section despite explicit system instruction
  not to name skipped/excluded entities. Prohibited-effect guard is witnessed.
  This is an output-constraint failure, not broken tool transport or arithmetic.
- Budget variance, episode 2009d4cbc2a94f8995f679fb0998f984: zero tool mistakes,
  5/7 assertions, partial5/7. Never uses available Gmail search/read to retrieve
  Variance Reporting Guidelines. Substitutes a generic threshold (10% or10,000)
  for authored15% AND5,000; includes budget-frozen Legal in report and Sales
  in finance-alerts. Both prohibited effects witnessed. Engineering/Operations
  arithmetic and requested sends pass. Some checks are deliberately strict
  about any excluded name, even under an 'additional' heading.
- Prepaid amortization, episode797bfb01120b4138ac4ed32234efee8b:3/8 assertions,
  partial .375 and separate -.04 tool penalty. Query 'amortization rules prepaid
  expense' retrieves only standard rules; AND search terms miss the available
  'Insurance Amortization Acceleration' override. Model uses2,000 instead of
  February insurance4,000, misses hosting300, writes2,000 total instead of4,300,
  and names excluded items. Two integer row arguments fail validation before
  string row_id succeeds. Native clock is February1; student infers an earlier
  month from email timestamps. Clock is not explicitly in the observed prompt
  or retrieved results and no clock tool is exposed, an additional observability
  concern. Do not attribute that ambiguity solely to model reasoning.

Implication: failures span inaccessible policy, incomplete retrieval, constraint
enforcement and tool-schema recovery. A successful tool call only validates an
operation, not the evidence or business content it carries. Evaluate task
observability before admitting any teacher demonstration to SFT. Do not alter
this running frozen benchmark; future repaired variants require separate lineage.

## Artifacts and Notes / Interfaces and Dependencies

packages/serve/src/posttrain/serve/profiles/base.py now accepts block_size:int|None
and forwards it as --block-size or block_size in embedded vLLM kwargs. Its
regression test is in packages/serve/tests/test_vllm_bindings.py. This exposes
an existing backend tuning option; it does not change product meaning, package
boundaries or backend selection. No baseline amendment is needed.

Durable local evidence belongs in docs/research/evidence/qwen35-27b-heldout62-20261018.
Framework package receipts and retained executions remain under apps/lab/.posttrain/state.
Native Verifiers traces are replay authority. No new backend, migration or
framework public API is introduced by these additive project selections.
