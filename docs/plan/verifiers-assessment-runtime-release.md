# Release the Verifiers assessment runtime-cost work with the manifest coverage campaign

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`; fork records follow `docs/tooling/forks.md`.

## Purpose / Big Picture

AutomationBench manifest tasks with hundreds of manifest assessments are slow to score and expensive to serialize. On the recorded 2.6B episode for `marketing.email_blast_suppression`, scoring takes about 12.5 seconds (the environment's own code is 0.3 seconds of that) and the episode record is 49 MB with 2,204 assessment batches. The trainer's worker pool also spends about 66 ms per episode converting episodes to records (`Episode.to_record` / `Trace.to_record`), almost all in the assessment archive encoder. After this release the same episodes score and serialize much faster with byte-for-byte identical findings, rewards, metrics and Posttrain turn rewards, and the release also ships the completed AutomationBench manifest coverage campaign (800 of 800 canonical tasks covered by reviewed drafts or installed contracts) and the task-metadata facet transport used for picking training and evaluation tasks.

Someone can see it working by rescoring the recorded episodes with the old and new Verifiers commits and comparing the outputs (identical), the scoring time and record size (lower), and by running `record_costs.py` on the r6 episodes (lower per-episode serialization cost).

## Progress

- [x] (2026-10-05 19:30Z) Surveyed every Verifiers, environment and rl checkout and branch; recorded what is unreleased (see Context). User decided scope.
- [x] (2026-10-05 19:40Z) Confirmed `codex/context-budget-clamp` (e9edbeb3) was never pinned or used by a run; excluded.
- [x] (2026-10-05 20:05Z) V1: archive-loading candidate committed as Verifiers b5026573 (scoring/trace/judges suites 187 passed).
- [x] (2026-10-05 21:40Z) V2: scoring, serialization and reply-wire changes committed as Verifiers d793cc6e with six new regression tests; native v1 suite 355 passed, 83 skipped (credentialed/opt-in). Serialization JSON schemas of SourceSnapshot, ObservationView, AssessmentBatch, WireTrace and WireEpisode are unchanged.
- [ ] Layer-1 equivalence: rescore recorded episodes, old vs new Verifiers, identical outputs; record timings and sizes.
- [ ] Serialization measurement with `record_costs.py` on r6 episodes; send numbers to the engine-redesign session.
- [ ] Update `CARBONTEQ_FORK.md`; commit and push the fork.
- [x] (2026-10-05 21:55Z) E1: campaign committed as environment f44ac80 on `wip/automationbench-manifest-coverage-release-2026-10-05` (81 copied, 1,780 new, 2 clean three-way merges, 1 import conflict resolved; scratch `controls.py` kept as `manifest-drafts/evidence/missing-ack-controls.py.txt`).
- [ ] E2: pin the new Verifiers commit in the environment; full environment suite.
- [ ] Layer-2 report: per-task findings/reward changes caused by E1, each tied to an engine fix.
- [ ] Push the environment branch.
- [x] (2026-10-05 22:10Z) R1/R3: campaign and task-selection evidence (115 untracked shared-checkout files except the facet test), the Verifiers README ledger notes, and nine machine-local selection files under `docs/research/verifiers-assessment-qualification/reward-candidate/selection-working-files-20261005/` committed on this branch, rebased onto a54759f9.
- [ ] R2: task-facet transport on its own branch for the engine-redesign session to merge.
- [ ] R4: pin bump only (packages/{train,data,eval}/pyproject.toml Verifiers and environment URLs, uv.lock, catalog environment revisions); the engine-redesign session runs the runtime-lock/image release chain on the merged result.
- [ ] Hand the pin-bump branch and the changed-task list to the engine-redesign session.

## Surprises & Discoveries

- Observation: the environment branches `wip/abv1-engine-r7` and `wip/abv1-merge-r7` are already released.
  Evidence: all 34 files on `wip/abv1-merge-r7` are byte-identical to environment commit 26f21b0, an ancestor of f587146.
- Observation: the scoring profile of `marketing.email_blast_suppression` (cProfile, 12.46 s) is dominated by Verifiers validation, not the environment.
  Evidence: `_validation_scope._typed` 3.29 M calls / 3.2 s cumulative from 11,577 intrinsic-proof lookups (4,413 source, 7,164 view); 868,908 pydantic `__eq__` calls / 1.33 s from `AssessmentBatch.verify.check_subject` (`subject.execution not in self.source.executions`); `SourceIdentity.verify_nodes` 11,035 calls / 1.5 s recomputing 266 k occurrence digests; environment code about 0.3 s. Source JSON is 262 KB with 24 executions; each assessment's single view input is about 251 KB. Batches: 551 each of queued, running, partial and complete.
- Observation: the trainer no longer restores assessment archives at admission; it decodes only nodes and calls. Archive loading still matters for the env client decode, evaluations and audits. Serialization (`serialize_assessment_archive` → `normalize_history` → `_same_wire`, about 51 M recursive calls over 160 r6 episodes) is the dominant remaining trainer-side Verifiers cost.
  Evidence: the engine-redesign session's cProfile of 160 r6 episodes (85 s total; `Episode.to_record` about 39 s, `Trace.to_record` about 30 s).

## Decision Log

- Decision: rl branches start from origin a54759f9; the pin-bump branch changes only pins (pyproject URLs, uv.lock, catalog environment revisions) and skips runtime locks, published images and the TRL lock digest; the task-facet change is a separate branch.
  Rationale: the engine-redesign session pushed its rewrite at a54759f9 (adds numpy to posttrain-train, two uv.lock edges) and will run the release chain once on the merged result, which also regenerates release constraints left stale by the 24c12379 repin (four failing tests on the branch).
  Date/Author: 2026-10-05, Claude with the engine-redesign session.
- Decision: pack env-server replies in the pooled archive form.
  Rationale: `serve/server.py` packed `model_dump(mode="python")`, which bypasses archive pooling, so every batch carried its full source and view: the email_blast_suppression reply was 1,174.5 MB for a 49.4 MB record, and an r6 episode with a 34.3 MB record produced a 1,772.8 MB reply (5.5 GB peak RSS in the decode harness). `EnvClient` already restores the pooled form; Posttrain reads replies only through `EnvClient` (confirmed by the engine-redesign session).
  Date/Author: 2026-10-05, Claude.
- Decision: elide repeated source/view objects with scope-token placeholders instead of memoizing their dumps.
  Rationale: pydantic re-serializes (copies) whatever a wrap serializer returns, so a memoized dict never reaches normalization as the same object; the memo version measured slower than the pin (790 vs 750 ms per heavy `Episode.to_record`). Placeholders remove both the repeated dumps and the deep `_same_wire` comparisons (391 ms).
  Date/Author: 2026-10-05, Claude.

- Decision: include the uncommitted archive-loading candidate (`/home/hammad/projects/verifiers-credit-candidate-20261003`, plan `docs/plan/verifiers-archive-loading-optimization.md`) as its own Verifiers commit, then revise its large-string proof-key change.
  Rationale: user asked to include all work. The candidate encodes every string over 4,096 characters to UTF-8 bytes on every proof lookup; on scoring workloads that is about 0.5 MB encoded and hashed per lookup, about 11.6 k lookups per heavy episode. Its purpose (stop cache eviction thrash under the 64 MiB accounting) is kept by accounting strings that the model already holds without re-encoding them.
  Date/Author: 2026-10-05, Claude (user approved scope).
- Decision: exclude `codex/context-budget-clamp` (e9edbeb3).
  Rationale: user decision; it was never pinned in rl, the environment or a catalog, only mentioned as an unchecked prototype item in plan notes.
  Date/Author: 2026-10-05, user.
- Decision: do not merge or archive the 2026-10-03 prototype dirty tree in `/home/hammad/projects/verifiers`.
  Rationale: user decision; its only content absent from 24c12379 is the LFM2/K2 renderer parsers, which moved to `carbonteq-renderers` deliberately. Its files are left untouched.
  Date/Author: 2026-10-05, user.
- Decision: include the manifest coverage campaign (uncommitted in `/home/hammad/projects/verifiers-environments-reward-candidate-20261003`) and the task-metadata facet transport (uncommitted in the shared `/home/hammad/projects/rl` checkout).
  Rationale: user decision. The campaign changes manifest evaluation on purpose, so it is proven separately (layer 2) from the reward-neutral Verifiers work (layer 1). The original dirty files stay untouched, as the campaign handoff requires.
  Date/Author: 2026-10-05, user.
- Decision: rl edits happen in `/home/hammad/projects/worktrees/rl-assessment-release` (branch `wip/assessment-runtime-release-2026-10-05`, from origin f2aa6d9f). The pin bump touches only pyproject, uv.lock, runtime-image locks/profiles, published.toml and catalogs; the task-facet change is a separate branch.
  Rationale: the engine-redesign session owns `packages/train/src/posttrain/train/update_*.py`, policy-update backends, `integrations/verifiers.py`, `integrations/verifiers_population_artifact.py`, `integrations/native_records.py` and `packages/environment/.../verifiers_conditioning.py`, and merges pin bumps and the facet change itself.
  Date/Author: 2026-10-05, Claude with the engine-redesign session.

- Observation: pydantic 2.13 runs a model's `after` validators even when an instance (not a dict) is supplied for a field, so every lifecycle batch re-ran `SourceSnapshot.verify` and `ObservationView.verify` on the same objects.
  Evidence: a minimal `Leaf`/`Par` model printed the leaf validator once per parent validation; email_blast scoring still made 2,204 source and 2,204 view proof lookups from `assessment_runtime.validate` after views were passed as instances.
- Observation: unprofiled heavy-episode serialization on the pin is about 570 ms of pydantic dumping and 215 ms of `normalize_history` per `Episode.to_record`; cProfile inflates `_same_wire` (31 M calls) to look dominant.
  Evidence: `t_split.py` on the 64.6 MB r6 episode (1,988 batches).

## Outcomes & Retrospective

Not started.

## Context and Orientation

Three repositories are involved. Verifiers (`carbonteq-ai/verifiers`, a maintained fork of PrimeIntellect Verifiers) provides the native assessment runtime: Pydantic models in `verifiers/v1/assessments.py` (`SourceSnapshot`, `ObservationView`, `AssessmentBatch`, `AssessmentRun`, `SubjectRef`, `ExecutionRef`), the executor in `verifiers/v1/assessment_runtime.py` (`execute_assessment_plan`, `execute_assessment`), the archive encoder/decoder in `verifiers/v1/assessment_archive.py` (`normalize_history`, `restore_history`, `_same_wire`) and the private proof cache in `verifiers/v1/_validation_scope.py`. An "assessment batch" is one immutable lifecycle record of one assessment attempt; `execute_assessment` appends queued, running, partial and terminal records to `trace.assessment_batches`. An "intrinsic proof" is a cache entry saying that a model with exactly these field values already passed its expensive self-consistency checks (canonical JSON, digest and identity checks); a "validation scope" is the bounded, task-owned lifetime of those entries. The "archive" is the wire form of a trace or episode in which full sources and views are stored once in `assessment_sources` / `assessment_views` pools and batches refer to them.

The environment (`carbonteq-ai/verifiers-environments`, package `environments/automationbench_v1`) pins Verifiers by commit in `environments/automationbench_v1/pyproject.toml` and lock. Its manifest engine lives in `src/automationbench_v1/contracts/` and the assessors in `src/automationbench_v1/manifest_*_assessments.py`. The current release branch is `wip/automationbench-manifest-release-2026-10-05` at f587146, which pins Verifiers 24c12379.

The consumer is this rl repository. It pins both by commit in `packages/{train,data,eval}/pyproject.toml`, `uv.lock`, `packages/catalog/src/posttrain/catalog/base/locks.toml`, the job-kind locks and profiles under `packages/runtime-images/.../posttrain-job-kinds/{locks,profiles}/`, `published.toml`, and catalogs under `apps/lab/.posttrain/catalog/` (notably `automationbench-manifest-steps.yaml`). Commit 842d5163 is a complete previous example of this pin transition.

Unreleased work found on 2026-10-05: the archive-loading candidate (dirty, base 959da638, an ancestor of 24c12379); the manifest coverage campaign (27 modified and 22 new engine/test files plus about 1,800 manifest-draft files, base 3d7ebc4; one import conflict with f587146 in `manifest_guard_assessments.py`); the rl shared checkout's task-facet change, three plans and 96 research evidence files; a machine-local task-metadata export at `/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/task-metadata/luna-selected120-metadata.json`; and a September environment checkout (`/home/hammad/projects/verifiers-environments`, branch `codex/evaluation-task-selection`) whose `jira/actions.py` carries a 611-line Jira change absent from f587146.

Recorded evidence for equivalence: `/tmp/claude-1000/-home-hammad-projects-rl/06f4fe1b-c6e2-45ea-9bb2-d0b2a04f9815/scratchpad/runs/r3/episodes.jsonl` and `.../runs/manifest-steps-26-g16x8-mean-20261005-r1/episodes.jsonl`, with single-episode extracts `.../mem/out/one-blast.jsonl` and `one-gorgias.jsonl`, and the harness `.../mem/equivalence.py`, `compare.py`, `profile_score.py`. Serialization evidence: `/home/hammad/projects/sim/data-r6/episodes.jsonl` (160 r6 episodes) and `/home/hammad/projects/sim/checks/record_costs.py`. The engine-redesign session's venv `/home/hammad/projects/sim/postcollect/venv` must not be modified.

## Plan of Work

Verifiers work happens in `/home/hammad/projects/worktrees/verifiers-assessment-cost` on branch `codex/native-assessment-runtime-cost` from 24c12379. First apply the archive-loading candidate unchanged and commit it (V1). Then V2: replace the recursive `_typed` key with an exact type-sensitive key that does not re-encode large strings and accounts shared strings once; reuse the already validated request views and source across one assessment's lifecycle records instead of re-dumping and re-validating them; make `AssessmentBatch.verify` membership tests set-based with identical equality semantics (falling back to the list when a value is unhashable); memoize `ExecutionRef.occurrence_id` per frozen instance; and make `normalize_history` avoid repeated deep `_same_wire` comparisons of the same pooled source/view (identity or digest short-circuits computed once per pooled item), plus avoid re-running full intrinsic verification while serializing already validated models. Each change gets a regression test in `tests/v1/` that pins its exactness property (tampered or type-changed inputs are still rejected) and, where measurable, its cost. The queued/running lifecycle records are kept unless both rl readers (`integrations/verifiers.py`, `posttrain_tracking_trackio/assessment_results.py`) and the environment's `plan_credit` are proven unaffected by a compact encoding; otherwise the decision and evidence are recorded here.

Environment work happens in a new worktree from f587146 on branch `wip/automationbench-manifest-coverage-release-2026-10-05`: copy the campaign's modified and new files from the candidate checkout, three-way merge the three files that also changed on f587146, check that new read captures honor ac10ccf's rule that manifest inputs ignore harness hook events, place the scratch `controls.py` under `manifest-drafts/` as evidence, then pin the new Verifiers commit.

rl work is the release worktree named in the Decision Log.

## Concrete Steps

Commands are added here as each milestone runs, with working directories and short expected transcripts.

## Validation and Acceptance

Layer 1 (no reward change): from the environment worktree at f587146 code, run `equivalence.py EPISODES asis OUT` once with Verifiers 24c12379 (installed in `/home/hammad/projects/worktrees/abv1-pin/environments/automationbench_v1/.venv`) and once with `PYTHONPATH=/home/hammad/projects/worktrees/verifiers-assessment-cost` prepended; `compare.py OLD NEW` must report every episode identical across error, rewards, metrics, turn_rewards, assessment_errors, credit_errors and findings. Report total scoring seconds and rescored trace bytes. Run `record_costs.py` on the r6 episodes with both Verifiers versions in a private venv and report per-episode `to_record` cost.

Layer 2 (intentional change): with the new Verifiers, rescore the same episodes with environment f587146 and with the campaign branch; list each task whose findings or rewards change and the engine fix responsible.

Verifiers: `uv run --python 3.13 python -m pytest tests/v1 -q` passes (credentialed cases skip), scoped Ruff and Pyright pass. Environment: the full `environments/automationbench_v1` suite passes with the new pin. rl: the AGENTS.md ladder (`uv sync --all-packages --locked --python 3.13`, ruff, pyright, lint-imports, pytest, `git diff --check`) and the runtime-image release check pass.

## Idempotence and Recovery

All work is on new branches and worktrees; the shared checkouts and candidate checkouts are never modified. Equivalence runs write to the session scratchpad and can be rerun. Benchmarks run one process at a time. If a Verifiers optimization breaks equivalence, revert that commit alone and rerun layer 1. Pins move only after the fork commit is pushed; the catalog is repinned only after confirming no 2.6B run is active.

## Artifacts and Notes

Filled in as milestones complete.

## Interfaces and Dependencies

No public Verifiers API changes. `verifiers.v1._validation_scope` stays private with the same `validation_scope`, `validation_owner`, `planned_validation_child` and `intrinsic_proof` names. Wire format of `trace.assessment_batches` and the archive pools is unchanged unless the Decision Log records otherwise.
