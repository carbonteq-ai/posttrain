# Agent command: implement project DX plan (milestone loop)

Paste the **Command** block below into an agent that can set goals and work
autonomously. Authority for *what* to build remains:

- DX product: `docs/developer-experience.md`
- Execution plan: `docs/plan/project-developer-experience.md`
- Repo rules: `AGENTS.md`
- Frozen product baseline: `docs/post-training/` (amend before changing meanings)

Do not invent a second product plan. Execute the existing plan.

---

## Command (paste this)

```text
GOAL: Fully implement docs/plan/project-developer-experience.md in
/home/hammad/projects/rl until every required Progress checkbox for milestones
A, A1, B0, B, C, D, E, and F is [x], Validation and Acceptance in that plan
passes, and docs/developer-experience.md remains the DX authority.

OPERATING LOOP (mandatory — do not skip):

1. Read AGENTS.md, docs/developer-experience.md, and
   docs/plan/project-developer-experience.md (Progress + current milestone
   section + Concrete Steps + Validation).

2. Set / refresh goals as a stack. Top goal = current incomplete milestone
   (first unchecked among A → A1 → B0 → B → C → D → E → F). Optional helpers
   only after F or when explicitly cheap and non-blocking.

3. Implement ONLY that milestone’s scope from the plan. Prefer wiring existing
   posttrain.data adapters and VerifiersEnvironmentRolloutBridge; do not
   rebuild them. Do not import apps/lab from reusable packages. Do not add
   posttrain sync. Do not shadow standard definition ids.

4. After the milestone’s code/docs edits, run THAT milestone’s validation
   commands from the plan’s Concrete Steps (and any Acceptance bullets in the
   milestone section). From repo root, prefer:
     uv sync --all-packages --locked --python 3.12   # when deps/lock change
     then the milestone-specific pytest / lint / CLI commands

5. If validation fails: fix until it passes. Do not start the next milestone
   with red checks. Record surprises in the plan’s Surprises & Discoveries;
   record scope decisions in Decision Log; update Progress checkboxes with
   timestamps when a milestone is done.

6. Only when the current milestone Acceptance is met, mark it [x] in Progress,
   then advance to the next incomplete milestone and repeat from step 3.

7. After F, run the full Validation and Acceptance list in the plan. If
   anything fails, treat it as a fix loop on the owning milestone — do not
   declare done.

CONSTRAINTS:
- Preserve unrelated dirty worktree changes; do not revert others’ edits.
- Baseline amendment (Milestone A) before code that renames Host* types or
  claims standard jobs / declarative dataset decode as product contract.
- train / eval / serve must not import one another; posttrain.jobs may compose.
- Format catalog fields must use adapter literals (auto|messages|prompt-completion|alpaca|sharegpt).
- Commit only if the user asks; otherwise leave a clean summary of what passed.
- Update the living plan as you go (Progress, Surprises, Decision Log, Outcomes).

DONE WHEN: plan Progress shows A–F complete, consumer proofs exist for SFT and
an environment-backed path without posttrain_lab, and plan Validation and
Acceptance items 1–7 hold.
```

---

## Goal stack (for agents that take structured goals)

Use this ordered goal list. Complete each before unlocking the next.

| # | Goal id | Done when |
| --- | ---: | --- |
| 1 | `dx-A-baseline` | `04`/`05`/README amended; `git diff --check` clean on those docs |
| 2 | `dx-A1-global-catalog` | Empty-overlay project lists global model/dataset/env ids; materialize idempotent |
| 3 | `dx-B0-wire-adapters` | Catalog dataset→`posttrain.data`; env→Verifiers bridge without lab imports; validate commands pass |
| 4 | `dx-B-jobs-package` | `posttrain.jobs` ships SFT/DPO/GRPO/distill/serve/eval/transform; lab re-exports; lint-imports + jobs tests pass |
| 5 | `dx-C-cli-no-host` | `work-package run` without required `--host`; Host aliases; entry escape-hatch test |
| 6 | `dx-D-init-install` | `posttrain init --template sft|grpo` writes + installs; doctor + import jobs; no lab |
| 7 | `dx-E-observatory-up` | `posttrain observatory up` prints URL for project tracking |
| 8 | `dx-F-consumer-proof` | Wheel consumer tests for SFT + env-backed path; docs/quickstart/lab README updated |

Per-goal subroutine:

```text
implement → run milestone checks → fix until green → update plan Progress → next
```

---

## Milestone check cheatsheet

Run from `/home/hammad/projects/rl` unless noted. Expand using the plan if a
command drifts.

**A**

    git diff --check docs/post-training docs/developer-experience.md docs/plan/project-developer-experience.md

**A1 / B0** (after code)

    uv run pytest packages/data/tests packages/catalog/tests packages/train/tests/test_verifiers_grpo_bridge.py -q

**B**

    uv sync --all-packages --python 3.12
    uv run lint-imports
    uv run pytest packages/jobs/tests packages/work/tests apps/lab/tests/test_work_packages.py -q

**C**

    uv run pytest apps/cli/tests packages/catalog/tests -q

**D**

    rm -rf /tmp/posttrain-sft-demo
    uv run --package posttrain posttrain init /tmp/posttrain-sft-demo --template sft --project-id sft-demo
    /tmp/posttrain-sft-demo/.venv/bin/posttrain doctor

**E / F**

    uv run pytest tests/consumer -q
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

---

## Anti-patterns (stop and correct)

- Starting milestone N+1 while N’s Acceptance is red
- Rebuilding HF adapters or Verifiers bridges from scratch
- Adding `posttrain sync`
- Making lab a consumer dependency
- Skipping Milestone A before Host renames / public DX API claims
- Marking Progress complete without running the milestone checks
