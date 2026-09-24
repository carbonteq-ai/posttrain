# Prepare VORTEX v5: solvable HR tasks, real turn budget, truncation penalty, and a larger batch

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

The last three LFM2.5-2.6B VORTEX runs (`lfm26-vortex-v2-agentic-20260923-r1`, `...-v3-lr2e4-...`, `...-yield-first-v4-8k-...` in Trackio project `posttrain-lab`) wasted much of their rollout budget:

- Every AutomationBench HR task, and six marketing tasks, were unsolvable in the `limited_zapier` toolset.
- Up to 73% of rollouts were truncated, most often by one reply running out of its 4,096 tokens mid-thinking.

After this change a VORTEX v5 work package exists, ready to schedule. It trains on solvable tasks and tells the policy its real 12-turn budget. It uses a LoRA learning rate inside the documented range, penalizes truncated rollouts so they rank below finished ones, and uses 16 prompt groups per update instead of 10. Nothing is launched by this plan. The first run is scheduled separately.

This does not change the frozen product baseline in `docs/post-training/01`–`06`. The truncation penalty is a new optional GRPO setting, off by default, in the same family as the existing DAPO soft-overlong shaping described in `02-primitives.md`.

## Progress

- [x] (2026-09-24) Adapter: add `google_drive_find_multiple_files` to `limited_zapier` tasks whose spreadsheet IDs cannot otherwise be discovered (106 of 800 tasks). This is in `../verifiers-environments`, file `environments/automationbench_v1/src/automationbench_v1/taskset.py`, function `_with_spreadsheet_discovery`.
- [x] (2026-09-24) Adapter: optional `AutomationBenchTaskConfig.turn_budget`. It replaces the upstream "~50 tool-using turns" system sentence with the real budget and a brief-thinking instruction (`_with_turn_budget`). Adapter tests: 14 passed. Ruff and pyright are clean.
- [x] (2026-09-24) Framework: optional `GRPOSettings.truncation_penalty` (TRL only; veRL rejects it). Covered by the new `packages/train/tests/test_truncation_penalty.py` (9 passed). The train and work suites give 523 passed and 9 skipped; the one test deselected is listed under Surprises. `lint-imports` keeps 9 contracts.
- [x] (2026-09-24) Catalog, `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml`:
  - settings `lfm2.5-2.6b/automationbench-vortex-yield-first-64-local-v5`;
  - training binding `training/lfm2.5-2.6b-trl-lora-automationbench-local-g64@1`;
  - inference binding `inference/lfm2.5-2.6b-vllm-automationbench-rollout-local-c64-4k@1`.
- [x] (2026-09-25) Adapter committed on top of the pinned base 9bbd3116, not the stale local branch, in a separate worktree so other uncommitted judge edits stay untouched. Commit `acb3043ed7b8c24df62c5b8e5232b13fd92710e1`, pushed on branch `codex/automationbench-v5-hr-tools-turn-budget`. 30 adapter tests passed.
- [x] (2026-09-25) Environment binding `automationbench-lfm26-train-mix-v5`: train-mix-v3 activation plus `task.turn_budget: 12`, source revision acb3043, `max_concurrent: 64`. Only this entry moves; every other environment entry and eval program stays on 9bbd3116. Verifiers environment packages are pinned per catalog entry, not in `uv.lock`.
- [x] (2026-09-25) Work package `apps/lab/.posttrain/work_packages/lfm26_automationbench_vortex_yield_first_64_local_v5.yaml`. `posttrain job plan --explain` resolves 29 settings with no errors (configuration digest e746a45c...).
- [ ] Launch the 20-update run and watch its first two updates as the capacity probe (user decision 2026-09-25, replacing a separate one-update probe). Check that rollout collection holds 64 episodes without OOM or refill exhaustion, the truncation rate and the HR reward, then let it finish.

## Surprises & Discoveries

- **The HR tasks can never be solved in `limited_zapier`.**
  - Upstream `zapier_tools` lists for all 100 HR tasks, and six marketing tasks, contain only Sheets tools that need a spreadsheet ID. The prompt never names that ID, and no tool is offered to find one.
  - In the 50-trace sample, HR scored 0 on 8 of 8 rollouts with a 74% failed-call rate, and `google_sheets_find_worksheet` failed 144 of 148 calls.
  - Zapier upstream `main` has identical HR task data, so this is an upstream bug that our adapter now works around.
- **VORTEX (OLMo 3, TRL) keeps truncated rollouts, but the signal from them is weak.**
  - At every step truncated rollouts keep their task reward: admission, `mask_truncated_completions: false`, reward shaping, and advantages.
  - Active sampling (`_prepare_active_sampling_inputs`, installed TRL 1.12.0.post9) drops any group with zero reward spread. That discarded 35%, 49% and 29% of truncated rollouts in v2, v3 and v4.
  - In the kept groups, 42, 50 and 79 truncated rollouts scored above their group mean and were reinforced.
  - Evidence: Trackio traces, grouped by `optimizer_step`, `rollout_batch_ordinal` and `example_id`.
- **Truncations are real reasoning, not loops, and never cut a tool call.** All 74 cut-off replies sampled from v3 steps 12–20 stopped inside thinking. None stopped after `</think>`, and the median 8-gram repetition was 2%.
- **The existing DAPO soft-overlong shaping applies only to `algorithm: dapo`** (`profiles.py`, `shape_online_reward`). It is inert under OLMo 3.
- **Pre-existing test failures, not caused by this plan:**
  - `apps/lab/tests/test_catalog.py::test_peft_bindings_settings_and_quantization_load_from_filesystem_catalog`: the recorded `uv.lock` digest differs from the current dirty `uv.lock`.
  - `packages/train/tests/test_verifiers_worker_client.py::test_native_train_client_round_trips_exact_lfm_tokens_over_loopback`: `ModuleNotFoundError` inside the local policy engine.

## Decision Log

- **Decision:** Fix the HR tool lists in our Verifiers adapter, not in the AutomationBench fork or upstream.
  **Rationale:** User direction. The adapter owns tool exposure for `limited_zapier`, and scoring is unchanged because Drive file search is read-only.
  **Date/Author:** 2026-09-24, user and Claude.
- **Decision:** Keep truncated rollouts, with a flat penalty, rather than masking them.
  **Rationale:** Published evidence is against masking at this truncation level:
  - Skywork-OR1 (arXiv 2505.22312) found masking raised the truncation ratio;
  - SWE-Master (arXiv 2602.03411) saw collapse;
  - OLMo 3 (arXiv 2512.13961) saw no consistent gain.

  Small fixed penalties are common in agent RL: verl-agent/GiGPO −0.1, AgentRL −0.2, Kimi K2 budget penalty. 0.2 on the 0–1 partial-credit scale makes a truncated failure rank below a finished failure, so mixed failure groups keep reward spread.
  **Date/Author:** 2026-09-24, user and Claude.
- **Decision:** Do not force-end thinking mid-reply (ScaleRL-style interruption).
  **Rationale:** Verifiers' train client requires a finite model logprob for every completion token, and `sampling_mask` describes filtered candidates, not unsampled spans. Inserted tokens would therefore be trained as policy output, or would need Verifiers and renderers fork changes. The user judged that too complex.
  **Date/Author:** 2026-09-24, user.
- **Decision:** Keep all budgets unchanged: 4,096 tokens per reply, 12 turns, 12,288 episode output tokens, 24,576 context.
  **Rationale:** User direction. Earlier turn-limit hits came from other issues, chiefly the HR tool gap.
  **Date/Author:** 2026-09-24, user.
- **Decision:** LoRA learning rate 1e-4, constant.
  **Rationale:** The v3 catalog note scales Tinker's LoRA RL learning rates to 4e-5–1.6e-4 at this binding's alpha 8. 2e-4 was above that range, and the full-fine-tune 1e-5 does not apply to LoRA.
  **Date/Author:** 2026-09-24, user.
- **Decision:** 16 prompt groups × 4 generations per update.
  **Rationale:** The user reports spare RTX PRO 6000 capacity. The v4 rollout phase averaged 220 s with vLLM at a 5 GiB KV reservation.
  **Date/Author:** 2026-09-24, user.

## Outcomes & Retrospective

Not yet run. To fill in after the capacity probe and the first v5 run:
- truncation rate per step (target: stays below 15%);
- groups discarded by active sampling;
- HR domain reward (expected to rise above 0);
- held-out evaluation compared with the base model.

## Context and Orientation

- **VORTEX:** this repository's name for OLMo 3 GRPO (`algorithm: olmo3`) with active sampling and an adaptive curriculum, trained with TRL on one RTX PRO 6000.
- **Active sampling:** after scoring, drop prompt groups whose four rollouts share one reward, then generate replacement groups.
- **Truncated rollout:** one whose Verifiers trace stopped at a limit. Either the stop condition is `max_turns`, `max_output_tokens`, `max_total_tokens` or `context_length`, or the last model reply ended with finish reason `length`. See `packages/environment/src/posttrain/environment/verifiers_evidence.py`, `verifiers_trace_is_truncated`.

Files changed in this repository:
- `packages/train/src/posttrain/train/profiles.py`: `GRPOSettings.truncation_penalty`, its validation, and `shape_online_reward(..., is_truncated=...)`.
- `packages/train/src/posttrain/train/catalog_schema.py`: schema field.
- `packages/train/src/posttrain/train/backends/trl/policy_rollouts.py`: passes `rollout.is_truncated` into shaping before rewards reach TRL, so group statistics and active sampling see the penalty.
- `packages/train/src/posttrain/train/integrations/verifiers_async_groups.py`: same call-site update.
- `packages/train/src/posttrain/train/backends/verl/launcher.py`: rejects the setting.
- `packages/train/src/posttrain/train/api.py`, `.../backends/trl/policy_config.py`, `packages/work/src/posttrain/work/runner.py`: record the setting as a run attribute.
- `packages/train/tests/test_truncation_penalty.py`: new tests.
- `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml`: v5 settings, g64 and c64 bindings.

Files changed in `../verifiers-environments`:
- `environments/automationbench_v1/src/automationbench_v1/taskset.py`
- `environments/automationbench_v1/tests/test_environment.py`

## Plan of Work

The remaining work is the Progress items left unchecked, in order:
1. Commit and push the adapter.
2. Add the environment binding and pins.
3. Add the work package.
4. Run the capacity probe.
5. Schedule the run.

## Concrete Steps

From `../verifiers-environments/environments/automationbench_v1`:

    uv run pytest -q tests/test_environment.py        # expect 14 passed

From this repository's root:

    uv run pytest -q packages/train/tests/test_truncation_penalty.py   # expect 9 passed
    uv run lint-imports                                                 # expect 9 kept, 0 broken

## Validation and Acceptance

- Loading any HR task with `toolset: limited_zapier` lists `google_drive_find_multiple_files`. Searching "NDA compliance tracker" in `hr.docusign_nda_collection` returns `ss_nda_tracker`.
- With `turn_budget: 12`, all 800 system prompts state "You have a budget of 12 tool-using turns" and none mention "~50".
- With `truncation_penalty: 0.2`:
  - a truncated rollout scoring 0 is shaped to −0.2;
  - a finished rollout is unchanged;
  - combining the setting with `mask_truncated_completions: true`, or with the veRL backend, fails at construction.
- For the run itself: the per-step truncation rate is below 15%, and HR reward is above 0.

## Idempotence and Recovery

- The new settings default to off, so existing work packages and recorded runs are unchanged.
- The adapter options are also off by default, except the Drive tool, which applies only in `limited_zapier` mode.
- To roll back, remove the v5 catalog entries. Nothing has been launched.

## Interfaces and Dependencies

- `posttrain.train.profiles.shape_online_reward(settings: GRPOSettings, reward: float, completion_tokens: int, *, is_truncated: bool = False) -> float`
- `GRPOSettings.truncation_penalty: float | None` (finite, > 0).
- `automationbench_v1.taskset.AutomationBenchTaskConfig.turn_budget: int | None` (> 0).
