# Build balanced, verified Luna workflow data with synthetic reasoning

This living plan follows docs/templates/PLAN.md and the canonical post-training
baseline. This is project-owned research/data preparation, not a framework
algorithm contract change. Preserve dirty work, existing jobs and cancelled queue.

## Purpose / Big Picture

Turn already saved, successful Luna workflow executions into candidates for
structured reasoning SFT. Balance unique task identities across Finance, HR,
Marketing, Operations, Sales, Support and Simple; validate native behavior and
causal information availability before treating a rationale as training data.
Historical perfect scores are a candidate filter, not a demonstration admission
proof. Reconstructed reasoning is synthetic annotation, never recovered Luna
hidden thought. No external model/judge calls or source publication. The latest
user request authorizes candidate generation and a gated small SFT improvement
test; earlier generation-only restrictions remain historical.

## Progress

- [x]Mixed-policy27B critic exited0: all three initial4B drafts rejected, one
  bounded requirements/assembly repair per task then all three approved. Manual
  guide inspection: Sales explicitly retrieves both HR request and separate
  matching guidelines, obtains Note title from observed policy, keeps domain and
  phone in body; no nonpublic title/ID/phone values forwarded. VIP draft contains
  a residual textual step-s1 reference, retained for qualitative review rather
  than silently edited. No solver success claim from critic approval.
- [x]Fresh4B control+mixed-guided solver accepted, container
  67f6d2fe8f7d1730df4d3f1ce07daa9638731024d75a01e4550aad839bed80fa.
  Same sealed sourcef294d0c862c0289c8ba3bc769212a2ecfbc35f47d2e1e0348f2aec43a65a2e96.
  Critic is stopped and its receipt checked before actor allocation; no duplicate.
- [x]Collect six fresh native episodes, inspect policy result before mutations,
  check title/body and manifest note goal, reconcile outcomes and provenance,
  compare with saved4B/27B diagnostics, update mixed-role viewer and report.
- [x]Mixed role result: fresh control3/3 and guided3/3 completion1/partial1,
  native Sales audit-note goal valid1 in both. Guided policy read occurs before
  update/note turns5/6; concrete title absent from initial prompt. Guided Sales
  seven calls/one empty versus control11/three. No success gain over passing
  control, no throughput comparison. Six unique episodes, no errors/length
  finishes/repeated calls/witnessed prohibited effects; sources/images verified,
  GPU free. Preserve qualitative failures: Simple reasoning cites guidance,
  Sales final response enumerates excluded records against native system rule.
  No SFT admission/scaling. Exact solver/critic context exports and report
  docs/research/luna-4b-solver-27b-critic-policy-2026-10-19.md retained.
- [x]Mixed-policy critic accepted after CPU native preflight, container
  0da510e39ec1598234ccd9dea86b0242754897e159a86594b876753b35d36fd6,
  sourcef294d0c862c0289c8ba3bc769212a2ecfbc35f47d2e1e0348f2aec43a65a2e96.
  41 focused CPU tests/Ruff/diff pass. Exact new phase receipt is in
  mixed-4b-27b-policy-20261019-r1/critic-launch.json; never duplicate it.
- [ ]2026-10-19 user authorized mixed4B solver/27B critic on same three tasks.
  Reuse saved4B requirements/workflow/initial assembly drafts;27B reviews and
  performs at most one requirements+assembly repair, then re-reviews. New opt-in
  policy variant maps actual title mentions predicates to existing policy email
  retrieval, strips private literals from critic-to-repair input, retains raw
  critic audit, and keeps private actor-value gates. The solver must search/read
  policy before choosing a Contact and derive required title from observations.
  Zero updates, no scaling.32K/4Kreply/16Kepisode/12turns/2episode8server slots
  unchanged, both BF16 pins unchanged. GPU free before staging. Two sequential
  immutable phases in luna-mixed-policy-20261019-r1, each cap3600seconds; no
  simultaneous model allocation. Run stage_luna_mixed_policy.py critic, inspect
  receipt and all three accepted guidance sets, then run its solver phase. Fresh
  4B control+guided six native episodes, reconcile errors/coverage and manifest
  obligations separately from legacy completion scores; inspect actual policy
  read, target selection, note title/body and reasoning. Stage/receipt failures
  remain preserved; no blind retry, gate weakening, SFT or old queue restart.
- [x]27B r1 exited0 after537.49seconds: five native episodes, both Simple tasks
  full in both arms, Sales control0success/0partial wrong decoy update/missing
  note. Sales guided was blocked by literal gate on in:inbox NexGen HR. Preserve
  r1; blocked is not zero score. This query is composed only of public request
  words plus the inbox operator. Six CPU prompt tests retain rejection of private
  names/numbers/email values. Narrow repair permits only that public syntax.
- [x]27B r2 accepted after native CPU preflight, container
  3711d3747ff0547f78347ee79fdc92c6274d0f4847d92e12c365076dda584de6,
  source25d10111ca973ffdb6446f4973cae4f41831e4955ea5d472d4fd9ecae28addc7.
  Actual material/native_data equality with4B verified for all three r1 tasks.
- [x]27B r2: repeat all six slots under separate immutable identity, sole source
  change public-query literal classification; all model/native/semantic gates
  unchanged. Do not pool r1/r2. The4B saved outputs remain a reference; their
  driver differs by the explicit gate repair, so label that in comparison.
  Launch with stage_luna_27b_comparison.py --public-query-repair, collect with
  collect_luna_27b_comparison.py --suffix r2. No further automatic variants.
- [x]27B comparison complete: r1/r2 each five scored episodes and one blocked
  guided Sales slot; no native errors or length finishes. r2 Simple4/4 full,
  Sales control0success/2/3partial correct update/missing note. Guided Sales
  critic correctly detected missing policy-title text, then copied nonpublic
  policy literal in repair; preserve legitimate block, no additional bypass.
  Saved4B Sales guided completion1/partial1 masks manifest audit_note_on_target
  valid0: retain separate signals.4B control goal valid1. Both27B sources/images
  verified; GPU free. Report docs/research/qwen35-27b-vs-4b-solver-generation-2026-10-19.md,
  viewer solver-27b-vs-4b-comparison.html. No admission or training restart.
- [x]2026-10-19 separate27B comparison accepted after CPU native preflight:
  container3227a320e53a6d504f514e4654635c0a7eddf66eae75c6a840f3e323acd7753a,
  source-manifest5647eb696cf9762a8ddd3e6aa06105c8e36f3d47f6da0963621ecc02271ace5c.
  Driver and inputs hashes match sealed4B opt05 exactly. No model updates.
- [ ]2026-10-19 authorized separate27B comparison: run the same three training
  tasks control+guided, with byte-identical sealed4B opt05 driver/inputs and
  scope-repair-natural-provenance-v5. Pinned BF16 fc05daec18b0a78c049392ed2e771dde82bdf654,
  same32K/4K/16Kepisode/12turns, greedy thinking actor/nonthinking same-model
  preparation,2episode/8server slots,16GiB KV. Zero updates, one-hour cap.
  Stage with scripts/qualification/stage_luna_27b_comparison.py from this tree;
  retained identity guards reject reruns. Inspect six native episodes, preparation
  rejection/missingness, actual reasoning/arguments/results, then combine saved4B
  evidence in solver-27b-comparison-20261019-r1 viewer. This is a new authorized
  comparison outside completed five-arm budget, not a training restart. GPU
  and Docker live check free before staging; no frozen product baseline change.
- [x] Further-five03: both Simple native episodes success1/partial1, no tool
  errors/empty responses or length finishes. The failing find/create case now
  performs lookup then creates the requested VIP record. Sales blocked after
  one repair because original retained requirements still require sender name
  instead of domain. Its repaired plan also copied a private domain example
  from raw critic context; it was not dispatched or admitted for SFT.
- [x] Further-five04: type sender-domain extraction without exposing its value,
  use observation-bound critic sources, screen known embedded private domains,
  and allow one requirements-plus-assembly repair. Three native episodes run;
  both Simple full, Sales success0/partial0 after selecting a similar-name decoy.
  Critic approved Sales despite remaining bad matching semantics; requirements
  also invented nexgen.com as an example. Retain as failed semantic qualification.
- [x] Selected03's simpler conservative repair before05: it blocks unresolved
  Sales instead of executing a wrong plan.05 accepted fresh4B control+guided,
  container40a844f2d7eac1bfd0bcb18a6792008c9b190da029c76aba98a27fc9ca056fa3,
  source-manifest6f2503f1ad5499923690d1a3fac407c80e3bcfb29ca0ad49b9c76a647f94f7f9.
  Fifth also uses neutral actor rendering and a novel bare-domain screen, based
  on03's private-scaffold reasoning and04's invented domain. This is a model/
  recipe bundle transfer check, not a causal single-variable estimate.
- [x] Finish05 native reconciliation, inspect actual reasoning, compare own4B
  base controls, summarize all five and stop at the authorized experiment cap.
- [x] Further-five05 completed: fresh4B control3/3 and guided3/3 full native
  success/partial; Sales uses5 tool-call turns/1 observed empty search guided,
  versus11/3 in its control. This is fewer native episode turns, not a measured
  end-to-end speedup (preparation costs extra calls). Controls already pass, so
  no reward gain or learning claim. All five reconcile14 unique native episodes
  (11 guided,3 fresh controls) and4 blocked guided tasks; zero native errors or
  reply-length finishes. No missing score zero-fill; retries remain unknown.
- [x]05 content review/CPU canonical export: two provisional Simple examples,
  five assistant targets, original thinking/calls/results and clean student
  prefixes retained. VIP rationale now refers to the public creation request
  rather than private harm/step labels. Create-contact rationale has minor
  string/object format wording despite schema-valid arguments; flag retained.
  Reject Sales rationale even with full scores: it invents a required sender-
  domain note title inherited from guidance. Export in adaptive-next-five/
  experiment05/compiled; training_ready=false pending native token/mask gates.
  Budget5/5 complete. GPU verified free; no additional generation or SFT queued.
- [x] Further-five01 completed, zero native episodes: scoped requirements
  distinguished createRecord from updateRecord, but a newly introduced variant
  prefix bypassed JSON constraints. Preserve as failed coverage.02 restored
  constrained output and corrected an existing schema mismatch where diagnostic
  checks were requested but validator expected goals/harms only; original check
  pointer indices are preserved.02 ran3/3 episodes: create-contact1, VIP creation0,
  Sales0/max_turns with four empty searches. Review correctly rejected the two
  bad plans; advisory diagnostic dispatch was explicitly retained in02.
- [x] For03, implement ONE bounded critic-guided assembly repair followed by
  fresh review. Rejected repair blocks solver dispatch; all literal, pointer,
  dependency and check coverage gates remain. CPU tests use the actual02 bad
  plan and require exactly one repair, including the rejection path. No running
  source was edited; all variants remain separate immutable capsules.
- [x]2026-10-19: User authorized FIVE more adaptive generation experiments.
  Ledger adaptive-next-five/ledger.json owns five accepted GPU attempts total,
  at most3600seconds each, zero optimizer updates. Choose each next arm after
  native review; no blanket queue, external calls or training restart. Start
  with9B on the same three bound tasks and explicitly reuse batch01-r2's fixed
  base controls. Model/seed/thinking/budgets remain matched. Blocked preparation
  is missing execution coverage, not a zero success. Other models/task subsets
  must be recorded as separate diagnostic comparisons, not matched replication.
- [ ] Experiment01: expose exact source collection, predicate and required versus
  prohibited effect scope to every preparation stage. Falsifier: Airtable still
  invents no-create-if-found or successful creation retention regresses. Source
  interpretation belongs in project research, not the native evaluator; no
  frozen product baseline or environment scoring changes.
- [ ] Experiments02–05: learn from actual stages/native episodes and record
  hypothesis, source/config/run receipts, time cap, quality failures and why
  each next intervention follows. Finish with an honest quality/admission
  comparison and inspectable raw traces before deciding on SFT readiness.
- [x]2026-10-19: Batch01 generation finished exit0: three controls and two
  guided episodes, one blocked guided task (Sales requirements repeated/omitted
  check IDs). Controls: Sales success0/partial0.6, both Simple success1/partial1.
  Guided create-contact success1/partial1; guided find-update success0/partial0.
  The requirements stage mistranslated the harm "do not overwrite existing
  contact" into "do not create when a matching contact exists"; assembly added
  an unsupported conditional and same-model review approved it. Native actor
  followed that condition, withholding the explicitly requested VIP creation.
  Preserve this as a pipeline regression, not a student inability.
- [x] Exported ONE content-reviewed guided episode through the existing installed
  posttrain.data.supervised_from_verifiers_jsonl adapter to
  sft-batch01-20261019-r2/compiled/supervised.jsonl. Two trainable assistant
  messages, with591/346 reasoning characters retained; clean native student
  system prompt, source episode/trace lineage and review hashes retained.
  Other guided episode rejected; no silent zero-filling of blocked Sales.
  Tokenization/no-truncation gate remains pending and training_ready=false.
  One easy task is insufficient for the planned balanced SFT improvement test;
  zero SFT updates launched. GPU verified free after generation. Do not substitute
  this singleton for a useful corpus or report student gains.
- [x]2026-10-19: User authorized generating SFT data and testing student
  improvement. Fresh CPU selection r1 screened perfect sources; r2 includes the
  previously authorized >0.9 sources,176 eligible identities total. Screened65
  identities before reaching two Simple slots; only sales.soft_matching and
  simple.airtable_create_contact/simple.airtable_find_update passed exact
  source-state/assertion/public-request/tool and manifest-authority bindings.
  Other categories have no admitted candidates under these checks. Preserve
  all rejection reasons; do not claim category balance or relax authority.
- [x]9B generation accepted as luna-sft-qwen35-9b-batch01-20261019-r2,
  container b260ce43b878b5fcca007904dbfb87c6d783c608b31fd0d2df8963c772256782.
  Exact CPU native preflight passed; source-manifest SHA256
  83547bb1bf9757e344cf54880cf0e93fffe208ed575cbcb8dc2d93e41f36512c.
  Capsule /var/lib/posttrain/qualifications/luna-sft-batch01-20261019-r2.
  Observe this owned run rather than duplicating it. No optimizer updates.
- [ ] Complete bounded9B control/guided native traces for these three candidates,
  retaining per-stage errors, critics, actual thinking and complete observations.
  Evidence root sft-batch01-20261019-r2; two-hour GPU hard ceiling, no updates.
- [ ] Audit candidate reasoning/causal availability/order/guards and native
  complete assessments; reject truncated/error/ungrounded targets. Compile only
  explicitly admitted native episodes via the existing data projection, preserving
  source lineage and clean student prompts. Critic opinion alone is not admission.
- [ ] If sufficient qualified data exists, qualify a bounded4B LoRA SFT pilot
  (at most50 updates, development every10) against a fresh matched base. Mask
  prompt/tool observations, preserve native reasoning and tool-token alignment,
  exclude whole overlong examples. Never use heldout62 for adaptive decisions.
  Otherwise report the actual data-quality shortfall before launching training.
- [x]2026-10-19: User requested the separate9B verdict and transfer of these
  lessons to SDFT teacher context. Added opt-in project candidate builder
  scripts/qualification/sdft_teacher_context.py and focused real-task tests.
  Retains unchanged student prompt, all manifest checks, observation-bound
  procedure and argument origins; rejects stale manifest authority, unavailable
  reference tools and non-user-ended prompts. It targets the task-level SDFT
  prompt/privileged_context interface, not SELF or SDPO action-prefix scoring.
  Candidate rows explicitly remain training_admitted=false. No trainer wiring,
  new GPU run, updates or SFT export; existing recipes unchanged. Examples and
  teacher template saved under teacher-context-v1 in the existing evidence root.
  Exact pinned SDFT integration, native tool continuity, token admission without
  truncation, teacher directions and matched student outcomes remain open gates.
- [x]2026-10-19: All five adaptive experiments completed sequentially on4B/9B:
  ten model runs,20 control and10 guided native episodes, zero model updates.
  All30 observed episodes scored fully and satisfied declared native checks;
  ten possible guided episodes were blocked, never zero-filled. Source receipts/configs/native populations and
  per-stage evidence are retained in adaptive-five/experiment01 through05.
  The complete live experiment ledger supersedes earlier single-pilot notes.
  No broader corpus or SFT export. Review findings remain distinct from native
  success:05/9B Salesforce is the strongest trace candidate;4B guard overclaims,
  read/mark-read ambiguity and proposed-plan dependency loss remain unresolved.
  GPU verified free after all runs. No sixth experiment or scaling queued.
  Report: docs/research/luna-source-separated-generation-five-experiments-2026-10-19.md.
- [x]2026-10-19: Execute five user-authorized adaptive source-separation
  experiments on the same two tasks with4B/9B. Each experiment uses fresh model
  instances, at most one hour/model, no updates, eight maximum native episodes.
  Historical attempts are outside this new five-experiment ledger; any new GPU
  attempt or evidence-backed retry consumes one experiment. No broad scaling.
  Learn from actual stage outputs/calls before configuring the next experiment.
  Ledger: docs/research/evidence/luna-ssr-sft-20261018/adaptive-five/ledger.json.
- [x]2026-10-19: Both fresh runs ended exit1 at requirements validation.
  Four control episodes completed/scored1 with zero tool mistakes; zero guided
  episodes.4B invented citation paths and manually leaked private expected values;
 9B cited real checks but failed literal-leak validation. Neither reached later
  preparation stages or HubSpot preparation. No silent retry or scale-up.
  Native9B control also planned create/post together and announced creation
  before observing its result. Preserve full-score outcome and semantic concern
  separately. Qualitative review and audit saved in solver-pilot2-20261019-r1.
  GPU verified free after exit. Revised input design/qualification remains open.
- [x]2026-10-19: User authorized revised two-task4B/9B execution. Fresh capsule
  /var/lib/posttrain/qualifications/luna-solver-two-task-20261019-r1 passed exact
  CPU native preflight. Source manifest7161297123d362abcd6a504aa732c790dba098e4c461e4fa8fb11ad37b2d7c9f.
 4B accepted container4ff59a602abeb4fbfc0c1a415be5555696039321578949f5a74e05f63adcfc9c;
 9B independently follows when GPU is free. No automatic retries or scaling.
  Launcher records model failures and still attempts the other requested model,
  retaining duplicate/GPU guards. One-hour hard cap/model remains unchanged.
  Local launch/experiment/source receipts: solver-pilot2-20261019-r1 in evidence.
- [x]2026-10-19: Implement same-model requirements, workflow, assembly and
  review stages in scripts/qualification/luna_solver_stages.py and wire them
  into luna_solver_pilot.py. Twelve focused CPU tests pass; Ruff passes.
- [ ] Repair the evidence-backed instruction-input issues and qualify all
  stages/native guided execution before scaling. Current fresh runs tested the
  stages but were rejected; preserve their immutable capsules and failures.
- [x] Prior two-task r2:4B completed four native episodes; legacy scores all1.
  Guided HubSpot nevertheless dispatched dependent read/create/post calls
  together, so those scalar scores do not qualify generated SFT.9B exited1
  at summary word-limit validation, before guided execution. Retain both.
- [x] User requested stopping annotation runs. Direct inspection found all
  annotation containers/queues already completed; no active GPU container.
  Preserve all completed evidence; no further annotation launches.
- [x] Re-read original paper2602.14469v2 and pinned public rcot_prompt.py.
- [x] Implement two-task fresh native solver pilot and four CPU prompt tests.
- [ ] CPU image preflight, then4B and9B control/guided native execution;
  inspect actual thinking, calls/results, score/guard coverage before scaling.
- [x] Original4B generation completed:14 skeletons,70 action annotations
  attempted after10 plan passes.20 annotations passed automated checks,
  36 length-limited;10 invalid/missing evidence references and4 explicitly
  unsupported/wrong-action rejections.36 of106 actions were not attempted
  because their plans failed. These are not native success counts.
- [x] Build self-contained interactive trace-comparison artifact with all14
  tasks, action selector, source-versus-annotation and original-versus-revised
  modes, model selector, source results, raw generation and pending states.
  Original normal execution is Luna input, not an unconditioned Qwen control.
- [x] Analyze first4B skeleton outputs:10/14 admitted by original checks,
  two length-limited, two invalid tags. Four qualitative examples expose tag
  misuse, overthinking and identifier copying despite some format passes.
  Evidence generation-r1/4b/qualitative-review-r1.json.
- [x] Implement an evidence-backed four-stage generation revision and25 CPU
  tests, including an end-to-end fake stage-flow test (not GPU quality proof).
- [ ] Finish baseline4B/9B annotations, audit both, then run ONE4B quality
  revision on identical14 cases; compare coverage and qualitative errors.
- [x] User-authorized27B cancellation persisted; owned container stopped,
  native82-row pre-stop snapshot preserved. Provider was still terminating at
  last poll; no GPU process remained before generation launch.
- [x]4B/9B immutable sequential generation queue launched.4B container
  bf4f04065cb4eb39c7d6348d45957bf2472b07fdfe753c501998d126b922348b
  confirmed running; model weights loaded and optimized vLLM compilation active.
  No generated skeleton/rationale confirmed at this observation.9B follows4B,
  not yet submitted. See generation-pilot14-r1-launch.json in evidence root.
- [x]2026-10-18: User proposed SSR-RCoT-16K-inspired Luna reasoning-data pipeline.
- [x] Read primary paper2602.14469v2, official dataset card and public prompt code.
- [x] Located both offline Luna campaigns and previous strict reference audit.
- [x] Fresh all-corpus inventory with model provenance, perfect-score filtering,
  split exclusion and domain counts; distinguish raw candidates from replay proof.
- [x] Build/test balanced selection and two-stage prompt/annotation contracts;
  eleven CPU tests pass, including nested evaluation-group exclusion.
- [x] Prepare14 real task cases, two/category, with106 action rationale templates,
  native tool definitions, reference quality and binding discrepancies explicit.
- [x] CPU native-tokenizer measurement and prompt scoping:22.2% fewer aggregate
  rationale input tokens; maximum prepared input14139 tokens. Generated skeleton
  and output lengths still require rechecking.
- [ ] Qualify local generator on small multi-domain set with native replay,
  then generate/validate a balanced corpus. GPU work waits for current owner.
- [ ] Export tokenizer-validated SFT with masked prompt/tool observations, retain
  supervised-only action versus reasoning controls. No training run yet.

## Context and Orientation

Worktree /home/hammad/projects/worktrees/rl-rft, codex/rft-warm-start. Offline
sources are /home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/
luna-reference-campaign-01 and luna-reference-campaign-02. These are historical
gpt-6-luna executions with capture events; some have no native message nodes or
hidden reasoning. Existing source/admission helpers live in scripts/qualification/
inventory_sdpo_luna_pool124.py and review_sdpo_luna_pool124.py. Their previous
strict audit admitted only five124-pool references, not a balanced seven-domain
corpus. Do not silently promote rejected reference attempts.

Task split authority: scripts/qualification/fixtures/automationbench_heldout62_v1.json,
automationbench_train_pool_luna07_v1.json and frozen SDPO cohorts. Exclude heldout62,
development24 and both inspected/new research reserve cohorts by task identity,
including every duplicate execution. Historical training tasks may be reused
only with explicit corpus scope; prior training novelty is not assumed.

The Qwen3.5-27B benchmark currently owns the workstation. The repaired evaluation
queue was user-cancelled; this request does not restart it or any SDPO/LFM queue.
Use CPU-only preparation during occupancy. The user selected the same pinned
Qwen3.5-4B, in thinking mode, to annotate data using teacher-only goals, verified
answers and Luna execution traces. It is not an SSR-format-trained teacher.
Generated annotations need native validation. Generator revision:
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. No generation has been launched.

## Research Basis and Adaptation

[Paper](https://arxiv.org/html/2602.14469v2), revised19June2026, separates an
abstract tagged skeleton from full reverse reasoning. It warns that simply
instructing a generator to hide a known answer does not remove answer anchoring.
SSR-D additionally trains an SSR-format teacher before producing student data;
prompting the local4B model is an SSR-inspired adaptation, not SSR-D
reproduction. The official16K dataset is general conversational reasoning, not
validated multi-step workflow/tool supervision. Its ArenaHard gain is not an
expected AutomationBench effect size. Official sources:
https://huggingface.co/datasets/Nanbeige/SSR-RCoT-16K and
https://github.com/viniferagy/SSR (inspected main tree
7527bd6049782568d051dccd912d94f23135bdff).

## Plan of Work / Milestones

1. Inventory and select: at least one recorded partial credit above0.9, no known episode,
   assessment or credit errors, nonempty complete capture, exact source hashes,
   eligible split. Inventory partials/missing evidence separately. Deduplicate
   task identities before selecting equal quotas, then stratify by service mix,
   action length and workflow skills. Do not let many Simple traces fill missing
   categories. If a category has insufficient qualified sources, report a
   shortfall instead of oversampling or weakening admission. Use own current
   model failures as a skill taxonomy only; never train on heldout task states,
   solutions or evaluation-specific repair examples.
2. Revalidate native actions: bind prompt/state/assertions/tool schemas to exact
   environment source, reconcile capture state chains, actually replay/solve
   using current tools, check all required goals and preservation constraints.
   Generated IDs or clocks cannot be silently ignored/remapped. Historical final
   assertions alone are weaker than fresh native execution; label each stage.
3. Generate an abstract full-trajectory skeleton first (PLAN/RETR/INFR/EVAL/
   SUMM plus purposeful BRCH/BTRK/RFLX). Then annotate one action at a time:
   original user/system prefix, tool schemas and observations available BEFORE
   the action, skeleton, target action. Do not include future tool results or
   privileged manifest expected answers. The generator knows the target for
   reconstruction, but factual claims must cite prefix evidence; this is a
   leakage-control adaptation, not evidence of recovered or faithful internals.
   Prefer short decision rationales naming needed checks, then native tool call.
4. Validate and export: exact action/argument equality, evidence references,
   schema parsing, no future IDs/results, complete target and length admission,
   fresh native task outcome. Citation presence is not semantic correctness;
   keep manual review and fresh replay as gates. Reject unsupported annotations
   rather than editing evidence to match them. Preserve real errors/backtracking
   or regenerate a fully qualified clean path; never make an executed failure
   look like a successful first attempt. Terminal answers are included only when
   backed by state. No tool-result tokens receive SFT loss. Render with target
   native thinking/tool template and verify exact assistant/action boundaries.

## Concrete Steps

Preparation code and tests live in scripts/qualification/luna_ssr_sft.py and
test_luna_ssr_sft.py. Evidence/output in docs/research/evidence/luna-ssr-sft-20261018.
All preparation commands must work without a model endpoint. Keep prompt text
versioned and input/source hashes in manifests; never overwrite accepted source
episodes. Any later generation uses existing local GPU/native model path after
checking occupancy, immutable request/output ledgers and bounded explicit token
budgets; no remote API model calls and no automatic new GPU queue now.

## Validation and Acceptance

Meaningful CPU tests must reject score-only admission, missing capture, duplicate
tasks filling quotas, split leakage, future-observation references and changed
tool calls. Real inventory reports domain counts and unmet quotas. No fabricated
CoT is emitted by CPU preparation. Generated rationales are not accepted SFT
until tokenizer/context and actual native outcome gates pass. Record examples,
source/tool states, cohort, teacher revision/prompts, reasoning/action tokens,
coverage, exclusions and heldout task identity independence.

First learning comparison when separately authorized: matched student base,
action-only verified SFT, skeleton/reasoning-augmented SFT on identical tasks,
then optional online distillation only after supervised improvement qualifies.
Measure clean task success, partial progress, tool errors and retention on frozen
development; freeze primary before heldout and never select by heldout answers.

## Idempotence and Recovery

Stable source hashes and per-request IDs prevent duplicates; write new output
directories. Missing/corrupt sources remain exclusions. Preserve prior failed
admissions and runtime attempts. No commits/pushes, paid capacity, external
judges, existing-run interruptions, shortened budgets or resumed cancelled jobs.

## Surprises & Discoveries

- 27B original comparison exposed a conservative screening false positive:
  a generated query composed of public NexGen/HR words and in:inbox syntax was
  rejected as a copied historical literal. No secret world value was involved.
  Separately, valid function signatures and zero tool-mistake metrics did not
  prevent a harmful wrong-contact update in unguided Sales. See native guard
  decoy_contact_not_updated, witnessed_declared_prohibited_effect, value1.
-05's ordinary4B control solves Sales with full native scores after broadening
  searches, resolving company/title/name and using a company/title SQL query.
  Its first guessed sender domain is unsupported and would need action-target
  admission review. The guided run also passes using fewer turns, but claims a
  domain-bearing note title is mandatory. Behavior, factual reasoning, and
  training target admission remain separate. Do not force enriched context on
  every task merely because native passes are available.
- Root cause still visible in05 workflow evidence: argument_provenance treats
  the API object_type value Contact as a substring extracted from an email
  subject, while masking SQL where_clause AND/field structure as generic runtime
  derivation. Workflow then suggests querying from the subject instead of
  preserving demonstrated company/title filters. API resource/field constants
  need schema-role classification before substring matching; preserve query
  operators and field paths while redacting only data literals. This is a
  source-inspected next repair, not a GPU-qualified fix in the five-run budget.

- Further-five02's critic correctly detects the unsupported absence condition,
  yet advisory dispatch reproduces the bad behavior.03's single assembly repair
  restores native creation but its successful reasoning still cites "harm
  requirement" and step labels absent from a clean student prefix. Native score
  improvement over a bad guided recipe does not establish exportable reasoning.
  Keeping every translated requirement is lossless only if the translation is
  correct: an incorrect sender-name requirement survives assembly repair.
-04's structured source normalization distinguishes sender domain from name but
  does not establish correct entity resolution. The solver picks a decoy contact;
  full tool API execution and reviewer approval remain insufficient. Screening
  only copied known private values also misses invented plausible domains.05's
  common-TLD bare-domain screen addresses the observed example conservatively;
  it is not a universal privacy or grounding guarantee.

- Batch01 demonstrates an actual causal context regression on a new task:
  baseline creates the requested VIP record, generated instructions invent a
  stop-if-found branch, and the guided actor stops after finding the existing
  contact. The manifest prohibited source is actions/updateRecord while the
  required source is actions/createRecord. A preservation constraint on updates
  must not cancel a required creation. The sources' generic kind field is create
  for both, so summaries need the concrete collection/effect predicate and public
  request, not a kind label alone. A cited check ID and same-model approval do not
  prove the paraphrase preserves its meaning. Broad scaling/training remains
  blocked by this quality problem despite passing transport/format checks.

- The local sibling TRL checkout c9af78c1c2ea04ad271e95b26b93dfadf8b9fca1
  has DemonstrationTeacherContextBuilder in trl/experimental/sdft/sdft_trainer.py.
  It composes a privileged teacher prompt and scores the completion under it;
  its conversational composer replaces the last user turn. This source review
  is not verification of the installed trl==1.12.0.post15 artifact. Restrict the
  candidate adapter to initial task prompts until native multi-turn integration
  is qualified. Privileged teacher evidence is intentional in SDFT; source
  separation is an explicit context variant, not a claim all SDFT must redact
  demonstration answers. The existing two-task redaction/provenance heuristics
  require broader validation before applying them to the full corpus.

Adaptive source-separation experiments: Removing private values did not suffice
for faithful instructions.4B translated static deal-stage provenance into a
fictional email-subject fact; its actor followed that hint and still received
full benchmark credit. Requirements were dropped by subsequent summaries.
Same-model reviews sometimes approved impossible instructions or demanded hidden
IDs instead of valid retrieval. A public quoted-name literal check was also too
strict; normalize genuinely public quoted values without allowing private ones.
Keeping typed parameter origins in one place corrected the demonstrated API
stage. It also exposed remaining opaque-handle and internal-join leakage.
Fifth-source compilation translates joins into ordering constraints, keeps
protocol field literals such as body_plain, and replaces opaque handles with
descriptions of values obtained from actual observed read fields. Full original
contracts and source traces remain the audit authority.

Native tool_empty_results metrics remained0 despite an observed empty Gmail
result in experiment04. The analyzer now reports parsed native empty/error
responses separately and retains original metrics. Neither value is rewritten.

The original solver preparation discarded manifest predicates and effect joins,
and discarded demonstrated argument origins.4B invented a search by a NEW mobile
number and referenced preprocessing metadata observed_success as if it were an
API field. Full source evidence now reaches earlier local-model interpretation
stages; only reviewed task instructions reach the actor. Same-model review can
share the generator's mistakes and is not independent semantic qualification.


- Historical Luna trace nodes may be empty: capture observations support actions,
  but do not provide recoverable hidden CoT or complete failed-call history.
- Prior124-pool perfect-score candidates were strongly Simple-skewed and
  current binding/state validation reduced admission further. Fresh inventory
  must report both source scores and actual qualification status.

## Decision Log

- Decision: Reuse saved4B drafts as the proposal source and use27B for critique
  and one bounded requirements/assembly repair, then load4B for native execution.
  Rationale: isolates the requested solver/critic roles without co-loading models
  or inventing new demonstrations. Typed policy retrieval contracts are derived
  from actual title predicates and existing policy messages, preserving mentions
  semantics rather than imposing exact equality. Raw critic input/output stays
  inspectable; private literals are removed only from repair context and never
  supplied to the actor. This changes project composition only, not the frozen
  framework baseline. Date/Author:2026-10-19/Codex.
- Decision: Preserve original27B r1 and run r2 after a narrow public-query gate
  repair, rather than bypassing validation or treating blocked Sales as failure.
  Rationale: satisfy the requested guided comparison while keeping all private
  value/semantic checks and exact model budgets. Six prompt regression tests,
  38 focused checks overall. This supersedes the self-imposed no-retry note for
  this specific evidence-backed repair; no additional exploratory arm authorized.
  Date/Author:2026-10-19/Codex.
- Decision: Run an authorized27B control/guided comparison against preserved4B
  opt05, cloning its sealed driver rather than incorporating pending fixes.
  Rationale: changing provenance or constraint handling now would confound the
  requested model comparison. The guided arm uses its own model for preparation,
  so any difference is end-to-end recipe/model evidence, not actor size alone.
  Date/Author:2026-10-19/Codex. Record source/receipts under
  docs/research/evidence/luna-ssr-sft-20261018/solver-27b-comparison-20261019-r1.
  Budget one3600-second GPU attempt, six potential native episodes, zero updates;
  failed/blocked slots preserved without zero filling. No automatic retry or SFT.
-2026-10-19: Reuse structured-provenance-v5 with9B for a three-task candidate
  batch; the two-task success evidence does not establish full pipeline quality.
  Candidate selection uses sources and bindings, not model scores on heldout.
  A stricter exact reference-state check excludes some procedures that could
  serve as cross-state hints, but this first SFT batch intentionally avoids that
  extra transfer assumption. Record the resulting category shortfall explicitly.
  New SFT authorization does not resume old SELF/SDPO or LFM queues.

-2026-10-19: Transfer the five experiments' provenance, ordering and guard-scope
  lessons into a separately versioned task-level SDFT context candidate. Keep
  teacher evidence out of student prompts and historical outcomes distinct from
  current observations. Do not transfer native full-score status or critic
  approval into blanket rationale correctness.9B experiment05 Salesforce is a
  candidate for review, not an admitted training target. No algorithm/default
  changes or further GPU execution are implied by this context preparation.

- Decision: In adaptive04/05, retain every translated requirement independently
  of the proposed action plan, rather than depending on another paraphrase to
  preserve it. Record any check uncited by the plan. Critic-negative plans may
  execute only as simulated diagnostics after privacy/structure checks, with
  explicit review_rejected_diagnostic_only evidence and sft_qualified=false.
  Rationale: Noisy critic judgements prevented native diagnosis and were not a
  reliable quality certificate. This is an experimental control-flow change,
  not a relaxed SFT admission rule or an improvement in model weights.
  Date/Author:2026-10-19/Codex.

- Decision: Run five bounded adaptive generation experiments, keeping both
  native tasks, exact4B/9B pins and actor budgets fixed. No training or larger
  corpus. Date/Author:2026-10-19/Codex, explicit user authorization.
  The first experiment separates nonpublic concrete source values into stable
  retrieval references, clarifies terminal-state requirements and constrains
  citations to check roots. This is a combined pipeline repair, not causal
  attribution to redaction alone. Record exact source manifests/configs and
  hypotheses before every launch in adaptive-five/ledger.json. Select2–5 only
  after observing the preceding model stage/native results. Failed attempts count.
  A failure on one task is retained without blocking the other task/model;
  guided native episodes are collected only for fully admitted instructions.
  Native control/guided scores have a ceiling on these easy tasks; instruction
  coverage, hidden-answer independence, actual observation-conditioned calls,
  prohibited changes and successful-action retention are separate criteria.
  Source aliases must not be mistaken for executable API argument values.
  Same-model reviews and literal screens cannot prove reasoning faithfulness.

- Decision: Make instruction construction an explicit same-model pipeline,
  rather than hand-authoring per-task hints or further compressing source labels.
  Rationale: The user wants the generator to understand what each task requires.
  Date/Author:2026-10-19/Codex.
  Stage1 translates every full goal/harm check to a requirement with a valid
  manifest JSON pointer. Stage2 interprets every demonstrated action, including
  failed/empty attempts, argument origins and earlier-result dependencies.
  Stage3 assembles ordered instructions with check/action provenance. Stage4
  checks the assembly against original sources with the same pinned local model.
  Each request uses that model's existing local server, greedy non-thinking
  preparation and a2048-token output cap. Solver execution remains thinking,
  greedy,32K context/4096 reply/16384 episode output/12turns. No external calls.
  Structural gates reject missing/duplicate/invented check IDs, wrong role or
  citations, invented tools, forward dependencies, endorsed empty/error actions,
  screened private literals and negative reviews. Output is never clipped or
  silently retried. These gates prove structure, not correct natural-language
  translation; actual native calls/manifest assessment still require inspection.
  All stage requests/responses and parsed source/output pairs are retained in
  hint-generations.jsonl and instruction-stages.jsonl. Clean actor inputs stay
  separate from privileged preparation evidence. This is research code under
  scripts/qualification; no frozen product meaning or dependency pin changes.


-2026-10-18: Solver r1 stopped on native manifest admission before model load,
  no model generations. The selected sheets_update_status has a stale full
  initial-state digest. Preserve run/container2be8c0203f13 and evidence/r1.
  A separate ad-hoc CPU probe also used typed Message objects instead of serialized
  dictionaries for authority paths; corrected preflight now executes the exact
  driver's serialized-material path. Replace only the rejected spreadsheet task
  with eligible simple.email_sf_contact_mobile_update; keep email_hubspot_deal_slack.
  CPU_NATIVE_PREFLIGHT_OK confirmed for both new native tasks and their bindings.
  New solver r2 capsule /var/lib/posttrain/qualifications/luna-solver-two-task-20261018-r2,
  source-manifest SHA2562978ec52843b60474b216dde381f5a915ca39447ff54de1fac6127179808ee4e.
  Launcher requires matching source-manifest CPU preflight receipt. New expected
  containers luna-solver-qwen35-{4b,9b}-two-task-20261018-r2. Sequential queue
  accepted PID3248997 (verify actual live launch receipt before relying on PID).
  No retry of failed r1 identity; no scoring gate weakened.
-2026-10-18: Latest user authorizes ONLY a two-task solver-generation pilot
  with4B/9B before scaling. Selected simple.email_hubspot_deal_slack and
  simple.sheets_update_status from the eligible Luna inventory; no heldout,
  development or reserve use. Each model runs one fresh control and one guided
  trajectory per task (4 native episodes/model,8 total), sequential models.
  Native task/tools/world execution and scoring use the installed immutable
  image, without publishing the local scoring/environment changes.
  Actor greedy thinking,32K context,4096 reply ceiling,16384 episode-output
  ceiling,12 turns,2 admitted native episodes,8 serving sequences,16GiB KV.
  Same limits and deterministic-world setting in both arms. One hour/model.
  No external inference/judge, training, adapters, quantization or automatic retry.
-2026-10-18: Compress guidance in two independent512-output-token, nonthinking
  calls to the same local model. Luna procedure input includes tool names,
  argument FIELD NAMES and success/error flags only, not historical values or
  results. Goal/harm input includes native public request and structural check
  declarations, not hidden expected-value payloads or full manifest. Validate
  word/item caps and nonpublic-literal/bookkeeping leakage before use; failure
  aborts the attempt instead of silently using invalid hints. Hints are plain
  operational instructions, not annotator bookkeeping. Actual and clean student
  request messages are saved separately. This is a privileged-guidance teacher
  diagnostic, not proof that conditioned reasoning is unconditioned/faithful;
  no SFT export until inspected and native goals/guards qualify.
-2026-10-18: Original SSR builds abstract skeletons then reverse reasoning
  from known answers; SSR-D trains a format teacher before final data generation.
  The public prompt encourages iterative deliberation, and the paper warns
  suppression does not remove answer anchoring. We adapt structural abstraction
  to online native tool execution; we are not reproducing SSR-D or relying on
  a promise to hide reference awareness. Native solver never sees full Luna
  traces, raw manifests or target calls; only summarized operational hints.
  Primary sources https://arxiv.org/html/2602.14469v2 and
  https://github.com/viniferagy/SSR/tree/7527bd6049782568d051dccd912d94f23135bdff.
-2026-10-18: User correctly identified retrospective annotation as a mismatch
  with the desired solver SFT demonstrations. Exact existing instructions say
  "Annotate the supplied next action" and "Return no replacement action";
  the driver supplies reference target calls and never executes newly generated
  calls. This is annotation research, not a fresh teacher trajectory pipeline.
  Treat prior outputs as diagnostics only, including quality r3; none can become
  whole-trajectory SFT solely by passing format/citation gates. No further
  annotation-only variants should be queued as a solution to this objective.
-2026-10-18: Live evidence confirms formatting deliberation loops:4B36/70
  rationale outputs length-limited,9B14/14 skeletons length-limited. Largest
  observed4B skeleton example repeats "wait"20 times. The revised nonthinking
  plan stage produced10/14 format-valid plans with zero length limits, but
  does not fix solver-versus-annotator ownership. Prompt overload is a plausible
  contributor, not an isolated causal result. Do not use a keyword ban or larger
  token ceiling alone as a correctness fix.
-2026-10-18: Next generation design must separate (1) reference-informed private
  planning, (2) genuine native actor execution and (3) external validation.
  Planner reads Luna/goals/harms and produces an abstract operational strategy,
  with no hidden entity IDs, answers or future outcomes. Actor receives normal
  task/tools and observed results; any supplied initial strategy must be an
  explicit synthetic assistant-plan TARGET in the exported demonstration, not
  an unrecorded privileged student input. Check strategy facts against the normal
  task context before admission. Actor chooses its own calls and parameters,
  executes them in a fresh native world and responds to actual results; do not
  ask it to annotate a known call or emit training/teacher/evidence bookkeeping.
  Native goal/guard evaluation remains outside actor context. Export only
  complete validated solver traces with normal student input, observed tools
  masked from loss and admitted assistant thinking/calls/final answer as targets.
  Do not fabricate hidden historical Luna CoT or correct results in place.
  This replacement execution path is NOT implemented/qualified/launched yet.
  Current bounded r3 annotation diagnostic may finish, but is not that path.
-2026-10-18: Quality queue r2 found an empty9B-exit-code.txt created by tee
  when docker wait started. Its existence check treated this as completed.
  The downstream running-container guard prevented a quality GPU submission;
  r2 queue exited and is preserved. Its baseline9B audit is premature/invalid.
  Corrected both queue/launcher to require a NONEMPTY exit-code receipt, while
  retaining stopped-container and idle-GPU guards. Fresh r3 source capsule
  /var/lib/posttrain/qualifications/luna-ssr-quality14-20261018-r3,
  manifest SHA256937444d7afd9e506074c44b1477478f25ffc63436215e8ed8a5839991f65528b,
  queue PID3246842 verified waiting while9B baseline runs. Expected new model
  container luna-ssr-qwen35-4b-quality14-20261018-r3; no quality GPU submission
  confirmed yet. Algorithm/cases/budgets unchanged from planned r2.
-2026-10-18:4B skeleton evidence motivated a bundled pipeline comparison,
  not a claim that extra stages always help. New run
  luna-ssr-qwen35-4b-quality14-20261018-r2, capsule
  /var/lib/posttrain/qualifications/luna-ssr-quality14-20261018-r2,
  source-manifest SHA25631acad7ea400132e779a2e7a7227291947ae0712ec27a248eae92566364866a7.
  Exact same14 cases and106 reference actions. Nonthinking workflow plan uses
  READ/DECIDE/ACT/VERIFY/PROTECT and1536 output tokens; prefix-only exact-quote
  evidence extraction1536; grounded per-action reasoning remains thinking4096;
  critique nonthinking1536.32K context, same BF16 model/seed/sampling/runtime,
  16 sequences and16GiB KV. Total quality container limit3600seconds. No
  training, truncation, external judge or native success claim.
  Queue waits for baseline9B terminal marker, audits both baseline outputs,
  then rechecks retained container identity and real GPU vacancy before launch.
  It submits this single distinct quality attempt, never automatic retries.
  Afterward it writes quality-4b-audit.json, with model critique labelled opinion.
  Each stage writes completed batches of16 so partial evidence survives timeout.
  Source capsules remain immutable after acceptance; old baseline is untouched.
-2026-10-18: User explicitly authorized quality-driven pipeline iteration,
  including breaking generation into additional stages. Preserve the running
  immutable4B/9B baseline. Inspect actual outputs before choosing a revised
  generation attempt; no student training. Compare the same task/action IDs,
  missingness, output limits, source compatibility, unsupported claims and
  qualitative examples, not just JSON parse rate or annotation length.
  Candidate additional stages are source/goal-gap review, prefix-only evidence
  extraction, abstract plan, per-action reasoning, separate critique, and bounded
  repair. Add a stage only when an observed failure motivates it. Every repair
  retains original output, criticism and revision; never change a real observed
  result to make a rationale fit. Source/native tooling repair remains separate.
-2026-10-18: Generator raw thinking cannot be exported as student reasoning:
  the generator saw privileged goal/reference/future context. Only separately
  reviewed prefix-grounded synthetic annotations can become thinking targets.
  A model critic is a diagnostic, not an independent correctness oracle. Native
  action/result checks and semantic review remain required before SFT admission.
-2026-10-18: User authorized stopping27B evaluation and generating pilot data
  with BOTH Qwen3.5-4B and9B. Same14 tasks, two-stage annotation prompts,
  thinking mode, temperature0.6/top_p0.95/top_k20/seed17,32K context,4096 total
  output tokens per request,16 model sequences,16GiB KV, BF16 pinned revisions.
  Run4B then9B sequentially, at most3600seconds per model including startup;
  no training, model/judge APIs or automatic retries. Format/reference checks
  are separate from semantic/replay admission; no ready SFT export claim.
  9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a verified in catalog/cache.
  Source capsule /var/lib/posttrain/qualifications/luna-ssr-generation14-20261018-r1,
  source-manifest SHA256 b333dd71fff45500e1a9625441ed7e76adaa3ab4ffcbb84ece4b6a2681af09f9.
  Expected containers luna-ssr-qwen35-{4b,9b}-pilot14-20261018-r1. Check exact
  retained state before submission; monitor accepted runs instead of duplicating.
-2026-10-18: User clarified current objective is optimization of CoT SFT DATA
  GENERATION, not student training. Authorized14-task generation pilot remains
  separate from training/old SDPO queues. CPU preparation completed while live
  GPU inspection found the original27B evaluation using100% compute and88.5GB.
  No new GPU allocation or model call occurred.
-2026-10-18: Hand-selected pilot spans distinct services and workflow skills.
  Simple email_hubspot_deal_slack and event_asana_calendar cover Gmail, HubSpot,
  Slack, Asana and Calendar. Thirteen sources score1; Hiver workload forecast
  scores0.9333 and deliberately exercises imperfect-reference review.
  Prefer verified outputs, not unconditional copying of reference actions.
-2026-10-18: Rationale prompts retain all prior observations and available tool
  names, but include full native schemas only for current/prior used tools.
  Missing schemas are explicit; high scores do not imply harm-free execution.
  No model-generated skeleton or rationale exists yet. Prepared templates carry
  a skeleton placeholder and are not ready for direct submission.
-2026-10-18: User expanded candidate eligibility to partial credit above0.9.
  Strict >0.9 (exactly0.9 excluded) inventory is revised-exclusions-r2/
  near-perfect-above09-r1.json:176 tasks, including28 pool124 tasks; Finance11,
  HR10, Marketing13, Operations13, Sales7, Support9, Simple113. Eleven additional
  identities relative to perfect-only filter. Prefer perfect executions; retain
  near-perfect sources separately and inspect failed goals/harms before repair
  and native qualification. Do not label an entire imperfect trajectory correct.
  This supersedes perfect-only candidate selection, not SFT admission gates.
  More Support candidates permit broader balanced subsets;14 is no longer a
  category-availability ceiling. Existing14 subset remains diagnostic only.
-2026-10-18: Keep SSR skeleton planning distinct from causal action annotation.
  Full-future reasoning copied before actions would teach unavailable information.
-2026-10-18: User selected the broader training-eligible Luna corpus, retaining
  eligible124-pool tasks. One verified perfect demonstration per task suffices;
  historical failures do not disqualify that task. Preserve attempt pass rates.
-2026-10-18: Protect nested difficulty-tiered evaluation groups as well as named
  heldout/development/reserves. Revised exclusion inventory supersedes the
  initial top-level inventory; initial artifacts remain preserved.
-2026-10-18: Prepare an initial14-task balanced validation cohort (two per domain)
  because Support has only two recorded-perfect candidates. This is a diagnostic
  subset, not a cap on the eventual corpus or a qualified SFT cohort.

## Outcomes & Retrospective

Mixed4B solver/27B critic completed: all six native tasks score1/1 and the
previously missed Sales note obligation now passes. The actor explicitly
retrieves policy before either mutation and uses its observed title. This
qualifies the local retrieval fix; the fresh control already succeeds. Guidance
still supplies procedural knowledge and has11 redundant bullets; Simple
reasoning remains dependent on guidance wording, and Sales output violates the
native instruction about excluded records. No clean-corpus/learning claim or
scaling follows. Full raw contexts were exported at the user's request. Both
GPU phases exited0; sources/images verified and GPU free.41 focused CPU checks
and Ruff/diff pass. Next work needs a user-authorized scope, not an automatic
training or broad-generation queue.
27B comparison:10 unique new native episodes across two preserved attempts,
two blocked guided Sales slots. Six saved4B references are reused across views,
not counted as fresh replications. No end-to-end27B improvement shown; critic
caught a real note-title constraint error, but repair copied a nonpublic policy
value and was rejected.4B legacy passing Sales score also conceals a valid0
manifest note obligation in the guided arm, confirming the need for separate
native obligation audits and qualitative admission. Greedy Sales control varied
between attempts (wrong-record harm versus correct update/missing note); no
determinism or model-ranking claim.38 CPU tests/Ruff/diff pass, sealed sources
and actual images verified, GPU free at final check. Both attempts remain
immutable; no SFT, heldout or old training queue started.
The additional five-experiment campaign is complete under adaptive-next-five.
03 repairs the concrete Airtable behavioral regression, retaining the creation
control.05 transfers the repair/natural-guidance bundle to4B and passes all three
native tasks, matching its fresh base control; Sales requires fewer tool turns
but preparation overhead prevents an end-to-end efficiency claim.14 native IDs
and all four preparation exclusions are reconciled. Two Simple episodes are
provisional reviewed SFT candidates, not a balanced corpus or training-ready
dataset. Sales reasoning remains rejected despite native full score. No model
updates, protected evaluation, external calls, dependency changes or additional
jobs occurred. The next engineering work is typed API/query provenance and
field-specific constraint validation, followed by broader verified data before
the earlier gated SFT improvement test. Do not launch a sixth experiment under
this authorization. Detailed comparison and raw snapshots are in the existing
research report and solver-next-five-experiments.html.

Batch01 produced five fully retained native episodes and one reviewed candidate
example, not a usable balanced SFT corpus. This falsifies readiness for the
user-requested improvement run under the current generation recipe. Repair
goal-versus-harm effect-scope translation and unsupported branch creation before
another data batch; require matched native execution to preserve the requested
creation while avoiding updates to the old contact. Keep the failing attempt and
all source bindings visible. No student training or improvement is claimed.

Completed the five-experiment budget, with no additional GPU attempt after05.
The final recipe executes both tasks on both models, but all fresh controls
already score1, so no model learning or reward improvement is established.
Same-model reviews are unreliable; critical order/guard semantics need explicit
validation beyond textual check coverage. Preserve the full native reasoning,
including unsupported verification claims, instead of treating good calls as
proof of good CoT. The complete audit is adaptive-five/summary.json; the local
interactive artifact is http://127.0.0.1:8852/solver-adaptive-experiments.html.

2026-10-19 instruction-stage integration: twelve tests pass, including a fake
four-stage flow, rejection cases and original two-task manifest input checks
preserving returned_before_dispatch and effect predicates. Reproduce from the
worktree root with PYTHONPATH=scripts/qualification .venv/bin/python -m pytest -q
scripts/qualification/test_luna_solver_stages.py
scripts/qualification/test_luna_solver_prompts.py. Run .venv/bin/ruff check on
luna_solver_stages.py, luna_solver_pilot.py and test_luna_solver_stages.py in
scripts/qualification. This is CPU integration evidence only. Native revised
generation, semantic instruction review and SFT admission remain outstanding;
no generation-quality improvement or trained-model gain is established.

Trace viewer: docs/research/evidence/luna-ssr-sft-20261018/trace-comparison.html.
Rebuild the snapshot with scripts/qualification/build_luna_ssr_trace_viewer.py
and the evidence directory argument after downloading fresh outputs. Its source
template is trace-comparison-template.html. The artifact is self-contained,
has no external requests and is a snapshot, not a live dashboard. Desktop/mobile
browser checks confirm model/revision/action controls, no JS errors and no390px
horizontal overflow. Raw JSONL remains the evidence authority; displayed raw
thinking is diagnostic and not SFT export. The unavailable9B/revision states
are explicit. Local preview: http://127.0.0.1:8852/trace-comparison.html.

Quality analyzer scripts/qualification/analyze_luna_ssr_generation.py reconciles
expected106 action IDs, parsing and finish reasons, explicit unsupported answers,
prefix-reference scope, potential copied target literals absent from the prefix,
source binding blockers and four prioritized task examples. Literal flags are
review heuristics, not proof of leakage; missing outputs remain missing. Eighteen
CPU tests pass across preparation, generation parsing and quality auditing.

Latest generation-preparation evidence: pilot14-r4/manifest.json and individual
task files under docs/research/evidence/luna-ssr-sft-20261018. Reproduce with:

    PYTHONPATH=scripts/qualification /home/hammad/projects/worktrees/abv1-heldout62-scoring-fix/environments/automationbench_v1/.venv/bin/python scripts/qualification/prepare_luna_ssr_pilot.py --project . --inventory docs/research/evidence/luna-ssr-sft-20261018/revised-exclusions-r2/near-perfect-above09-r1.json --output <fresh-directory>

The environment checkout is local and unpublished; its native loaded task state
and tool definitions are research inputs, not a new published environment pin.
All14 source user prompts/assertions match the current task versions. Only three
have byte-equivalent initial states; many current task tool lists omit tools used
by Luna, and several manifest bindings fail. These are generation/replay admission
issues, not grounds to claim a qualified14-task corpus. Inputs explicitly retain
the discrepancies so an annotator can flag unsupported actions. Do not undertake
unrelated environment repair or silently remap state just to pass this pilot.

Exact native thinking-template input audit: pilot14-r4/token-audit-r2.json.
Compared with all-schema pilot14-r2/token-audit-r2.json, aggregate rationale input
tokens declined822622 ->639959 (22.2%), including added compatibility metadata.
Largest prepared request14139 tokens. These measurements exclude actual generated
skeleton length/output and do not prove fit at generation time or GPU throughput.
Earlier token-audit.json files counted a BatchEncoding container rather than
token IDs and are INVALID, preserved for evidence; only token-audit-r2.json is
authoritative. CPU tests13 pass; Ruff passes. No generated SFT quality claim.

Scanned961 local episode files representing800 Luna task identities. With the
user-selected one-perfect-source rule and revised protected-split exclusions,
165 task candidates remain: Finance10, HR10, Marketing10, Operations13, Sales7,
Support2, Simple113. Of these28 belong to the124-task pool. Candidate source
hashes, historical failures and omitted assertion counts remain visible.

Authoritative inventory is docs/research/evidence/luna-ssr-sft-20261018/
revised-exclusions-r2/{inventory,summary,prompts}.json. Earlier root-level
inventory omitted nested evaluation groups and is superseded, not deleted.
Historical scores can omit assertions; in particular both Support candidates
have excluded legacy checks. Neither historical perfection nor valid generated
citations establishes complete current goal/guard coverage. No generated or
admitted CoT corpus, native replay qualification or learning gain claimed.
No new GPU job or external model request.

## Interfaces and Dependencies / Artifacts

Use existing native AutomationBench verification/replay and target tokenizer
interfaces. Keep project curation separate from reusable training packages.
Sources are immutable local JSON plus public paper/code reading; no new pin.
