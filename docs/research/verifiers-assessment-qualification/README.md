# AutomationBench assessment qualification: research index

Working evidence for the AutomationBench reward redesign (living plan:
`docs/plan/automationbench-teacher-calibration-and-action-rewards.md`). The
plan's Progress, Decision Log and Outcomes sections are the authority for
status; this folder holds the inputs and evidence behind them. Files are kept
until the redesign is resolved because later work still uses them.

## Where things live

- **Manifest drafts, reviews and status** are in the environment repo
  (`verifiers-environments`, `environments/automationbench_v1/manifest-drafts/`):
  `LEDGER.md` (per-task coverage and status), `AUTHORING.md` (authoring rules and
  the shared mechanism vocabulary), `tasks/<task>/` (latest draft and review)
  and `legacy/` (original drafts of installed manifests).
- **Installed manifests** are in the environment package under
  `src/automationbench_v1/contracts/tasks/`.
- **Raw recorded data** (Luna reference episodes, public packs, proofs) is
  machine-local under `.posttrain/state/verifiers-assessment-qualification/`;
  superseded drafts, replay tallies and run logs are archived there under
  `archive/`. Git does not carry these.

## Current reference (used by ongoing work and tests)

| Path | What it is |
|---|---|
| `independent-agent-handoff.md` | Handoff: goal, boundaries, repositories, read order |
| `luna-development-policy-contracts.json`, `luna-development-policy-dispositions.md` | Public policy contracts per development task (read by environment tests) |
| `luna-development-review-coverage.json`, `luna-development-review-cases.json`, `luna-development-reward-review.md` | Development-set review coverage and cases (read by environment tests) |
| `HR-rule-spec.md` | HR family rule specification |
| `reward-candidate/capability-opportunities-105.{json,md}` | 105-task capability matrix: public requirements, capabilities, packs and episodes |
| `reward-candidate/cross-category-selection.json` | Task → SHA-bound Luna episode index (used by `recorded()` in tests) |
| `reward-candidate/manifest-authoring-batches.json`, `manifest-authoring-batch-NN-review.{json,md}` | Public reviews of the 105 tasks (11 batches) |
| `reward-candidate/acceptance-cohort-0{1,2,3}/` | Inactive acceptance-cohort declarations and their limits |
| `reward-candidate/reward-coverage-current.{json,md}` | Earlier reviewed-slice coverage snapshot (2026-10-03) |
| `reward-candidate/no-clarification-*`, `summary-semantic-*` | Semantic-judge corpora for the separately reported system-prompt rules |
| `qwen-benchmark-scaffold/`, `sdk-probes/`, `execution-occurrence-candidate/`, `frozen-official-verification-candidate/` | Scaffolds, probes and replay evidence for later benchmark/migration work |

## History

Dated `*-checkpoint.md`, `*-qualification.md`, `*-proposal.md`, `*-review.md`
and early `luna-*` review notes under this folder and `reward-candidate/`
record how decisions were reached. They are evidence, not current status;
prefer the plan and `LEDGER.md` when they disagree.
