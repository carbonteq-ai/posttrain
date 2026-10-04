# Release manifest rewards and run one corrected SAMPO comparison

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

AutomationBench now has whole-task reward manifests. These are deterministic checks: each one declares which business effects a task requires (goals) and which it forbids (harms), and the environment turns the matches into native findings and credit. After this plan:

- the environment that carries these manifests is released and pinned;
- the manifest reward is shown to work in live training, through 2–3 step correctness runs on LFM2.5-1.2B;
- one SAMPO training run on LFM2.5-2.6B compares the corrected, manifest-informed reward with the older system.

Success is observable as a run whose findings and step credit are recorded, plus a held-out evaluation compared directly with the last H100-era SAMPO run. More tasks are added after the release; this plan does not wait for them.

This plan does not change the frozen product baseline. It uses the 0.4.14 `policy_updates` engine and existing credit contracts. Admitting the CISPO objective in the resolved engine is a separate gate, recorded below.

## Progress

- [x] (2026-10-05) Commit the environment candidate on local branch `wip/automationbench-reward-redesign-2026-10-04`:
  - `26f21b0`: shared verification capabilities, engine round 7 and the unpersisted-read inventory. Full suite 3,738 passed, 7 skipped.
  - `d0b2c9c`: 204 manifest drafts, our 105 sample drafts plus about 99 from the Luna top-20 authoring wave. All drafts load and survive a canonical re-save.
- [ ] Release: commit and push the AutomationBench fork, publish the native Verifiers credit candidate, push the environment repo, then pin its commit in this repository (catalogs, `packages/eval` and `packages/data` constraints, `uv.lock`) and run the validation ladder.
- [ ] Choose the release task set and install its manifests with SHA plus Luna-replay gates.
- [ ] Wire manifest step credit into the trainer as process credit on whole steps (one step = one model turn: its reasoning tokens and its tool call).
- [ ] Correctness runs on LFM2.5-1.2B, 2–3 updates each, in parallel on the two machines.
- [ ] One SAMPO run on LFM2.5-2.6B with the last H100 run's settings, plus the held-out evaluation and comparison.

## Surprises & Discoveries

- Observation: every earlier SAMPO run spread turn credit across the whole trajectory, so mistakes inside a relatively good episode were reinforced.
  Evidence: `docs/research/sampo-multirun-analysis/report.md`. Across arms, about 47–52% of guard-violating episodes still scored above their group mean, and 7 of 15 score-lowering actions received positive credit.
- Observation: 0.4.14 admits the resolved `policy_updates` engine only for TRL GRPO/DAPO/SAMPO, TRL SAMPO semantic spans, and native veRL SAMPO. It requires `world_size` 1 and rejects TRL with vLLM rollouts.
  Evidence: `CHANGELOG.md`, the 0.4.14 entry.
- Observation: the CISPO math exists (`cispo-upper@1`) but is not admitted in the resolved engine.
  Evidence: `packages/train/src/posttrain/train/update_objectives.py` and `backends/policy_update_math.py`.
- Observation: only 23 of the 105 sample tasks, and none of the 5 installed ones, are full passes in the 709-task Luna run. That run had a mean partial credit of 0.56 and 222 full passes.
  Evidence: `docs/research/verifiers-assessment-qualification/luna-development-review-coverage.json`.

## Decision Log

- Decision: the episode reward stays the official AutomationBench `partial_credit`. Manifest findings add step-level credit.
  Rationale: this keeps runs comparable with the older system while the manifests supply the per-step signal the older runs lacked.
  Date/Author: 2026-10-05, user and Claude.
- Decision: manifest credit is applied to the whole step that produced the effect (a satisfied goal is positive, a verified harm is negative), never to the whole trajectory. Unknown findings give no credit.
  Rationale: the multi-run analysis shows trajectory-wide spreading reinforces mistakes. A step is the unit the model controls; a single tool-call token range would drop the reasoning that led to it.
  Date/Author: 2026-10-05, user and Claude.
- Decision: correctness runs use LFM2.5-1.2B for 2–3 updates, split across the two machines, checking correctness and gradient norm. The comparison run uses LFM2.5-2.6B with the last H100 run's settings, on our own hardware, not RunPod.
  Rationale: this is cheap correctness evidence first, then one apples-to-apples comparison. Only the settings are reused, not the cloud machine.
  Date/Author: 2026-10-05, user.
- Decision: the comparison is a single SAMPO run, compared with the older system.
  Rationale: one controlled comparison isolates the reward change. CISPO stays in the correctness runs, pending its admission gate.
  Date/Author: 2026-10-05, user.

## Outcomes & Retrospective

(To be written at each milestone.)

## Context and Orientation

- **Environment candidate:** `/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`.
  - Manifests are installed under `src/automationbench_v1/contracts/tasks/` and listed in `contracts/catalog.json`.
  - Drafts live in `manifest-drafts/tasks/<task>/`. Authoring rules and mechanisms 1–32 are in `manifest-drafts/AUTHORING.md`; status is in `manifest-drafts/LEDGER.md`.
- **Native Verifiers candidate**, required at runtime and currently unpublished: `/home/hammad/projects/verifiers-credit-candidate-20261003`. It owns findings, credit assignments and receipts.
- **AutomationBench fork** (task data and simulator fixes, uncommitted): `/home/hammad/projects/automationbench`, ledger `CARBONTEQ_FORK.md`.
- **Trainer credit inputs in this repository:**
  - `packages/train/src/posttrain/train/assigned_rewards.py` covers episode, turn and token projections of native assigned credit;
  - `update_process_credit.py` covers process credit;
  - `update_credit.py` and `update_objectives.py` complete the credit path.
- **Last H100 run (the settings source):** `lfm26-sampo-fp16-base-lr6e5-notrunc-6k14t-h100sxm-50-20260930-manual-r4`, work package `train/lfm2.5-2.6b/automationbench-sampo-turns-50-r4-notrunc-6k14t-fp16-os4r5-h100sxm-v1`. Its settings:
  - model LFM2.5-2.6B, bf16 weights, fp16 base, LoRA rank 4;
  - learning rate 6e-5, constant schedule, KL 1e-2, 50 updates, seed 42;
  - groups 24×6 (per-device batch 1, gradient accumulation 144), max length 28,672;
  - 14 turns with up to 6,144 tokens each, temperature 0.8;
  - oversampling os4r5, no truncation penalty;
  - environment pin `0bad6187`, task mix `lfm26-automationbench-mix-v2` (160 tasks), reward `partial_credit`.

  Full resolved config: `docs/research/sampo-multirun-analysis/evidence/run-configs.json`.
- **Held-out evaluation used by the older runs:** `eval/lfm2.5-2.6b/automationbench-heldout-matched-64k-v4-t05`.
- **Machines:** a local 8 GB card, and the workstation RTX PRO 6000 via dstack. Run one job per machine (see memory notes).

## Plan of Work

1. **Release.** Commit the AutomationBench fork changes with their ledger. Publish the Verifiers credit candidate. Push the environment branch. Update the immutable environment pin in this repository's catalogs, dependency constraints and `uv.lock`, then validate.
2. **Release task set.**
   - Start from the manifest-covered tasks in `lfm26-automationbench-mix-v2`, so the comparison keeps its task mix.
   - Install their manifests with gate tests (packaged bytes equal the reviewed draft; the hash-bound Luna replay reproduces recorded findings).
   - Report which mix tasks have manifests. Tasks without one receive only the episode reward.
3. **Step credit wiring.** Map each manifest credit contribution (goal-positive or harm-negative, keyed by tool invocation) to the model turn that issued that invocation, and project it as process credit over that turn's whole token span through the 0.4.14 process-credit path. Then combine it with the episode reward under the SAMPO advantage. Document the scale and the combination rule in the reward contract digest.
4. **Correctness runs**, LFM2.5-1.2B, 2–3 updates each, using the H100 settings except the model and update count:
   - workstation: SAMPO (native veRL, admitted);
   - local card: the same advantages with CISPO. This requires admitting `cispo-upper@1` as a qualification step, otherwise it runs as an explicitly unadmitted experiment. Use bf16 if the fp16 token-CISPO NaN recurs.
5. **Comparison run.** One SAMPO run on LFM2.5-2.6B with the H100 settings and seed, then the held-out evaluation.

## Concrete Steps

To be filled with exact commands as each milestone starts: the `posttrain job plan` and launch commands, dstack submission for the workstation, and local launch from a clean detached worktree of the pushed commit, per the memory note on clean-worktree launches.

## Validation and Acceptance

**Release:** the environment suite passes at the pinned commit (3,738 or more passed). This repository's ladder passes (`uv run ruff check .`, `uv run pyright`, `uv run lint-imports`, `uv run pytest`).

**Correctness runs, per arm:**
- no assessment or credit errors;
- the reward contract digest is recorded;
- on sampled steps, manifest credit lands only on the issuing turn's tokens;
- the gradient norm is finite, non-zero and within the range of the older runs;
- there are no NaNs and the loss changes.

**Comparison run:** it finishes 50 updates. Against the last H100 run on the held-out evaluation, report the official score, complete-solution rate and harm rate, plus the same metrics on the manifest-covered tasks.

## Idempotence and Recovery

Releases are by immutable commits, so a bad pin is reverted by restoring the previous pin. Training runs launch from clean detached worktrees and keep native evidence. The correctness runs are short and can be repeated.

## Artifacts and Notes

The run IDs, evaluation IDs and comparison table are recorded here as they are produced.

## Interfaces and Dependencies

- TRL 1.12.0.post14 and CarbonTeq veRL 0.9.0.post9 (0.4.14 pins).
- The native Verifiers credit candidate, to be published, and its commit pinned.
- The environment commit carrying the manifests, to be pushed and pinned.
- The manifest credit contract: `credit` entries with `required_effect_once@1` (goals) and `per_effect_negative@1` (harms) in installed manifests.
