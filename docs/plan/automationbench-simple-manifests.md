# Give AutomationBench Simple tasks a full share of the manifest-steps training mix

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

The manifest-steps runs train on a fixed task mix, `automationbench-manifest-luna-v1` (85 training and 20 evaluation tasks, fixture `scripts/qualification/fixtures/automationbench_manifest_luna_v1.json`). It holds only two training tasks from AutomationBench's Simple category, although the GPT Luna reference agent fully passed 153 Simple development tasks, and the other categories hold 9 to 19 training tasks each. After this change a new mix, `automationbench-manifest-luna-v2`, keeps every v1 task and adds Simple training tasks until Simple has 20, chosen by Luna's score the same way the other categories were. Every Simple task in it earns manifest step credit (the per-step part of the reward), including the one-field record-update tasks that v1 had to reject. The local 1.2B VORTEX check then trains on the Simple tasks only, which a small model can solve within its context.

Someone can see it working by loading the new environment and listing its tasks (20 `simple.*` training tasks), by the step-credit tests in the environments repository, and by a local run whose rollouts show passed record goals credited to the turn that made the write.

## Progress

- [x] (2026-10-06 04:30Z) Diagnosed why v1 has two Simple training tasks (see Surprises).
- [x] (2026-10-06 04:40Z) Clean environments worktree `/home/hammad/projects/worktrees/venv-simple-manifests`, branch `wip/automationbench-simple-manifests-2026-10-06`, from `da8bb32` (the revision the manifest-steps catalog pins).
- [x] (2026-10-06 06:10Z) Milestone 1: `manifest_outcomes` counts record goals, credited to the write `select_credit` names; scorer version 4 (environments `a107af4`). Three new tests on Luna's recorded `simple.sf_opp_*` episodes (the step-credit occurrence equals the native credit recipient for all ten, joint record goals count twice on one write, an unreachable expected value is decided and uncredited); 12 tests fail on the old reader. Environments suite 4,182 passed, 7 skipped.
- [x] (2026-10-06 07:20Z) Milestone 2: `docs/research/verifiers-assessment-qualification/reward-candidate/simple-manifest-replay.py` replayed all 200 Simple manifests (190 drafts, 19 installed; 164 have a Luna episode) into `simple-manifest-replay-20261006.json`; `simple-manifest-select.py` accepted about 130 by v1's rule and wrote `scripts/qualification/fixtures/automationbench_manifest_luna_v2.json` (103 training tasks, Simple 20, sha256 `4201d4a5...d9a2`).
- [x] (2026-10-06 07:45Z) Milestone 3: 17 manifests installed (14 rebound to the user message; `simple.sf_opp_type_update` was already installed), environments `cdcc4dd`, ledger path fix `0408bc0`, pushed to `wip/automationbench-simple-manifests-2026-10-06`. Suite 4,199 passed, 7 skipped; the final replay on the installed bytes selects the same 18 tasks, each with every required goal passed and credited to a step.
- [x] (2026-10-06 08:05Z) Milestone 4: catalog environments `automationbench-manifest-steps-luna2-4k16t-v1` (103 tasks) and `automationbench-manifest-steps-luna2-simple-vortex-local-v1` (20 Simple tasks, 3072-token replies, 8192 output tokens per episode) at `0408bc0`; `reward/automationbench-manifest-steps@2` (scorer version 4, digest `f221a532...48dd`); settings `lfm2.5-1.2b/automationbench-manifest-steps-luna2-simple-vortex-local-v1` (12k context, 8-group candidate pool); binding `inference/lfm2.5-1.2b-vllm-manifest-local-12k@1`. The local VORTEX work package now uses them; its earlier environment and settings, which never finished a run, are removed. The environments revision is pinned only in the catalog (no lockfile entry). Lab tests 105 passed.
- [x] (2026-10-06 04:45Z) Milestone 5, first pass: local run `manifest-steps-12-vortex-local-20261006-r5` (r3/r4 ran out of memory starting vLLM with 1 GiB and 768 MiB KV caches; 512 MiB fits) applied 8 updates over 4 collections with finite non-zero gradients (norms 0.07-0.30), KV peak about 50%. Collection evidence shows `candidate_source: adaptive_curriculum`, a different curriculum task pair per collection, zero-spread groups never retained, and learning-signal retention keeping the second candidate (0.375) over the first (0.25); the checkpoint holds `adaptive-curriculum-state.json`. Every collection filled in its first round, so the refill path was not exercised. It also showed manifest goal credit missing on most successful episodes (see Surprises).
- [x] (2026-10-06 05:40Z) Live finalization fix: environments `089b67d` (sources read `assessment_finalization_state`, not `trace.ok`); regression test reproduces live scoring on Luna's recorded episode; suite 4,201 passed. The v2 catalog environments pin `089b67d`.
- [ ] Milestone 5, confirmation: local run `manifest-steps-12-vortex-local-20261006-r6` shows decided goals and goal credit on live episodes.
- [ ] A 2.6B work package on `automationbench-manifest-steps-luna2-4k16t-v1` with reward `@2`, once the user chooses to move the 2.6B run to v2.

## Surprises & Discoveries

- Observation: in live training, manifest goals and guards mostly abstained (`obligation_scope_unavailable`, `obligation_occurrence_count_unavailable`, `created_terminal_finalization_unavailable`, `retained_record_finalization_unavailable`, `guard_compliance_scope_unavailable`), so manifest step credit and harm debits rarely applied, while rescoring the same saved episodes decided them. Verifiers (`verifiers/v1/rollout.py`) runs `task.score` after task finalization but sets `trace.ok` in its `finally` block after scoring, and the AutomationBench sources declared the task complete only when `trace.ok` was true. Replays of saved episodes, where `ok` is already true, never showed it. This affects every manifest-steps run on environments revisions before `089b67d`, including the 2.6B r10 run.
  Evidence: r5 `simple.hs_update_deal_stage` (partial credit 1.0) live findings `requested_write abstained obligation_occurrence_count_unavailable`; rescored `requested_write valid 1.0 obligation_occurrence_count_verified`, `manifest_outcomes` = (one credited write, 1 required goal).

- Observation: 115 Simple drafts bound the whole prompt, so only 45 of 164 replayed manifests bound the 16-turn task; rebinding the binding alone left their goals abstained (`request_field_authority_unbound`) because request fields name the whole prompt as their authority.
  Evidence: `simple.airtable_create_contact` original replay `required 1, passed 1`; binding-only rebind `required 0` with findings `abstained request_field_authority_unbound`; with authority narrowed `required 1, passed 1, witnessed 1`.
- Observation: abstained goal findings carry `required: null`, so a count of "required and undecided" misses them; the replay counts any abstained goal finding as undecided.

- Observation: the v1 selection excluded Simple by construction. Its candidate list was "Luna top-20 per non-Simple category" (`docs/research/verifiers-assessment-qualification/reward-candidate/luna-top20-per-nonsimple-category-selection.json`); Simple contributed only the 19 tasks that already had installed manifests, and the plan of record noted a request for few Simple tasks (`docs/plan/automationbench-manifest-reward-release-and-sampo.md`, Decision Log). The user now expects about 20 per category.
- Observation: 13 of those 19 Simple tasks were rejected as `no_required_goals` although Luna fully passed 12 of them. Ten are installed record-update manifests (`simple.sf_opp_*`) whose only goal is a `record.fields_equal@1` check: it verifies the final Salesforce record and is credited through `select_credit` (`contracts/credit.py`) to the write that made the record correct. Manifest step credit (`manifest_step_credit.py`, `manifest_outcomes`) reads only obligation and guard findings, so these goals never count, and the replay that fed the selection (`manifest-luna-replay.py`) counted the same way.
  Evidence: `manifest-luna-replay-20261005.json` rows `simple.sf_opp_amount_update ... official_full req 0`; `tasks/sf-opp-amount-update.json` check `requested-state`, role goal, operator `record.fields_equal@1`, credit `verified_transition_once@1`.
- Observation: the local VORTEX r1/r2 failures were not curriculum faults. All 16 r2 rollouts were truncated before any tool call (reply cap 1024 tokens, or a 6,078-token Support prompt against the 6k context), so every group had zero reward spread and the 4-candidate active-sampling pool ran out.

## Decision Log

- Decision: drafts that bind the whole prompt are rebound to the user message only when admissible: the old binding matches Luna's episode, the user message is identical in Luna's episode and the 14- and 16-turn tasks, the draft never reads the system message, and every request field authorized by the whole prompt is stated by the user message (its literal value appears in it, case-insensitively, or it is copied from inside it). Those fields' authority moves to the user message.
  Rationale: the system message states the turn budget, so whole-prompt bindings abstain at any budget but the authoring one; the engine does not check that an authorized value appears in its authority text, so the move is made only where it is verifiably true, and anything else keeps the old binding and is rejected by the selection.
  Date/Author: 2026-10-06, Claude.
- Decision: the selection scripts live with the research evidence (`docs/research/verifiers-assessment-qualification/reward-candidate/`), like v1's `manifest-luna-select.py`.
  Rationale: they record how a fixed mix was chosen; they are not maintained tools, and `scripts/qualification/` requires an owned exit criterion per script.
  Date/Author: 2026-10-06, Claude.
- Decision: the local VORTEX work package is retargeted rather than duplicated.
  Rationale: its previous environment (mixed domains, 1024-token replies, 6k context) could not produce a tool call with LFM2.5-1.2B-Thinking, and no run of it completed.
  Date/Author: 2026-10-06, Claude.

- Decision: fix step credit to read record goals rather than rewrite record-update manifests as obligations.
  Rationale: a record goal is a required goal with a verified witnessing write; teaching the reader fixes every record-goal manifest in every category at once, and attributing through `select_credit` keeps step credit identical to the environment's own credit assignment.
  Date/Author: 2026-10-06, Claude.
- Decision: the manifest step-credit scorer version becomes 4.
  Rationale: the same configuration now produces different rewards for tasks with record goals; the scorer digest must say so.
  Date/Author: 2026-10-06, Claude.
- Decision: the selection replay counts goals with `manifest_outcomes` itself.
  Rationale: the v1 replay had its own count, which drifted from what training reads; selection and training must use one function.
  Date/Author: 2026-10-06, Claude.
- Decision: Simple candidates are ranked by Luna's original score, ties broken by `sha256("20261006:" + task_name)`, not lexically.
  Rationale: 153 Simple tasks tie at a full pass; a lexical tie-break would pick 20 Airtable, Asana and Buffer tasks. A seeded hash is just as reproducible and spreads the choice across apps.
  Date/Author: 2026-10-06, Claude.
- Decision: v2 keeps every v1 training and evaluation task and only adds Simple training tasks.
  Rationale: evaluation stays comparable with v1 runs, and the two reserved Simple evaluation tasks stay out of training.
  Date/Author: 2026-10-06, Claude.

## Outcomes & Retrospective

None yet.

## Context and Orientation

Two repositories are involved. The environments repository (`carbonteq-ai/verifiers-environments`, worktree `/home/hammad/projects/worktrees/venv-simple-manifests`) holds the AutomationBench environment in `environments/automationbench_v1`. A manifest is a per-task JSON contract (`src/automationbench_v1/contracts/tasks/*.json`, listed in `contracts/catalog.json`) whose checks score an episode: goals (things the agent must achieve), harms (things it must not do) and diagnostics. Obligation goals publish `automationbench.manifest_obligation_result@1` findings with the witnessing tool invocation; record goals publish `automationbench.manifest_evaluation@1` evaluations whose credited write comes from `select_credit`. Drafts live in `manifest-drafts/tasks/<task>/draft.json`; installing one copies it into `contracts/tasks/`, adds it to the catalog and to `tests/installed_candidate_manifests.json`, whose test checks the packaged bytes equal the draft and that the manifest binds the 16-turn training task.

Manifest step credit (`src/automationbench_v1/manifest_step_credit.py`) adds `manifest_goal_share / R` to the turn whose tool call first witnessed each passed required goal, where `R` is the number of decided required goals, and debits verified harms on their issuing turn up to a cap. The configuration and its scorer digest are in `turn_rewards.py` (`AutomationBenchTurnRewardConfig`).

In this repository, the catalog `apps/lab/.posttrain/catalog/automationbench-manifest-steps.yaml` defines the manifest-steps environments (task names, pinned environments revision) and the settings and bindings of the runs; the mix fixture and its selection evidence live under `scripts/qualification/fixtures/` and `docs/research/verifiers-assessment-qualification/reward-candidate/`.

## Plan of Work

Milestone 1 changes `manifest_outcomes` in `manifest_step_credit.py`: besides obligation findings it collects the latest complete `manifest_evaluation@1` receipt per record check, loads the contract from that batch's view input (`material["contract"]`), counts each goal-role result with status `valid` as a decided required goal, and for passed results takes the witnessing occurrence from `select_credit` on the combined evaluation. `turn_rewards.py` sets manifest scorer version 4. `tests/test_manifest_step_credit.py` gains a record-goal test (a correct write credited on its turn, an initially satisfied record giving no credit, a damaged record giving none) and the digest test is updated.

Milestone 2 adds `docs/research/verifiers-assessment-qualification/reward-candidate/simple-manifest-replay.py` to this repository: for each Simple draft or installed manifest it scores Luna's recorded episode as `manifest-luna-replay.py` did, but reads goals and harms through `manifest_outcomes`, and writes `docs/research/verifiers-assessment-qualification/reward-candidate/simple-manifest-replay-20261006.json`. Acceptance per task is v1's rule (binds the 16-turn task, no assessment or credit errors, at least one decided required goal, agreement with Luna full passes, no harm on full passes, Luna full pass or partial score of at least 0.5). The 18 highest-ranked accepted tasks that are not v1 evaluation tasks join the training mix.

Milestone 3 installs the selected drafts, runs the environments suite, commits and pushes the branch.

Milestone 4 writes `scripts/qualification/fixtures/automationbench_manifest_luna_v2.json`, adds v2 environments to the catalog (the 2.6B comparison environment and a local Simple-only environment) pinned to the new environments commit, and updates the dependency pins and `uv.lock` files that name the environments revision.

Milestone 5 runs the local VORTEX check on the Simple-only environment with a reply budget of 4096 tokens and an 8192-token context, and records the curriculum and credit evidence in `docs/plan/vortex-resolved-engine.md`.

## Concrete Steps

Environments tests, from `/home/hammad/projects/worktrees/venv-simple-manifests/environments/automationbench_v1`:

    uv run pytest -q tests/test_manifest_step_credit.py
    uv run pytest -q -x

## Validation and Acceptance

The new step-credit test fails before Milestone 1 and passes after. The replay output lists at least 18 accepted Simple tasks outside the v1 evaluation split, with `required >= 1` for the `sf_opp_*` record-update tasks. Loading the v2 training environment yields 20 `simple.*` tasks. The local run's rollouts record `manifest_goal_credit > 0` on the turn that made a correct record write.

## Idempotence and Recovery

The replay is read-only and can be rerun. The environments work is on its own branch from `da8bb32`; the dirty campaign workspace at `/home/hammad/projects/verifiers-environments-reward-candidate-20261003` is left untouched. v1 environments and runs are unchanged; v2 is additive.

## Interfaces and Dependencies

`manifest_outcomes(trace) -> (goal_occurrences, required_count, harms)` keeps its signature. The environments revision used by v2 replaces `da8bb32` only in the v2 catalog entries and the pins they need.
