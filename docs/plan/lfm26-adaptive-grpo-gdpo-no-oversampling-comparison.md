# Run the matched LFM2.5 adaptive-GRPO and GDPO comparison

**Superseded as the primary comparison on 2026-09-20.** The completed GRPO run and failed GDPO attempts below remain historical evidence, not a GDPO-versus-SAMPO result. The corrected objective and remaining work are in [the VORTEX/GDPO/SAMPO plan](lfm26-vortex-gdpo-sampo-comparison.md). Do not submit another job from this plan's old two-arm design.

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

This experiment measures two LFM2.5-2.6B LoRA training policies on the same AutomationBench task mix after the Posttrain 0.4.4 release. One arm uses adaptive curriculum with ordinary GRPO and the other uses GDPO with a self-hosted Gemma judge. Each arm performs 50 optimizer updates, for 100 total planned updates. Every update collects eight prompts with four trajectories per prompt at inference concurrency 32. Neither arm may collect extra candidate batches. After both descendants exist, the same held-out evaluation is run against each adapter.

The first three optimizer updates of each arm are an operating gate. Logs and metrics must show finite optimization values, exactly 32 requested trajectories per update, correct LoRA policy-version advancement, and enough sustained GPU work to justify the selected concurrency. If the accelerator is materially under-filled, the run must be stopped rather than wasting the remaining budget, the inference policy adjusted, and the arm restarted from a new immutable run identity.

## Progress

- [x] (2026-09-20 12:00Z) Defined two matched 50-update training selections, c32 inference profiles, and one shared held-out evaluation package.
- [x] (2026-09-20 12:10Z) Corrected the no-oversampling arm from invalid OLMo3 to ordinary GRPO with adaptive initial-batch curriculum allocation.
- [x] (2026-09-20 12:12Z) Validated all three work-package compositions with the repository CLI.
- [x] (2026-09-20 12:35Z) Qualified the released runtime images and both job plans from a clean Posttrain 0.4.4 stable-index installation.
- [x] (2026-09-20 12:39Z) Passed 132 focused tests plus Ruff, committed the experiment as `480ece91`, and pushed its branch.
- [x] (2026-09-20 13:05Z) Qualified the attention compiler fix in attempt `20260920b`; startup then exposed the colocated memory-budget mismatch and revision 2 reduced the hard vLLM fraction to 9% without changing c32 or the 4 GiB KV cache.
- [x] (2026-09-20 13:22Z) Attempt `20260920c` proved native vLLM FA4 rejects SM120. Switched LFM rollout and eval to native FA2 and added a static job-compilation error for RTX PRO SM120 plus FA4.
- [x] (2026-09-20 13:42Z) Attempts `20260920d/e` exposed host contention rather than an LFM capacity defect. The retained `vllm-sm120-dev.service` held 80.8 GiB; it was stopped and the intended 4 GiB KV arena restored before the exclusive-GPU retry.
- [x] (2026-09-20 15:28Z) Adaptive arm `lfm26-adaptive-grpo-50-c32-20260920f` (`pt-1455ba03bb1be0920665b259`) succeeded with all 50 optimizer updates, 32/32 rollouts and zero failures on the final step, plus committed step-50 model, recovery, and training-summary artifacts. Its first-three-step operating gate recorded 96/96 rollouts, warm throughput of 1,086-1,708 tok/s, repeated 100% rollout intervals, and sustained 100% actor-update utilization.
- [ ] Complete the second 50-update arm and monitor at least three updates with the same evidence. Attempt `lfm26-gdpo-50-c32-gemma-mtp2-20260920b` (`pt-d0b98e60633ce736d8b0c4df`) failed Gemma KV admission. Attempt `lfm26-gdpo-50-c32-gemma-mtp2-20260920c` (`pt-3fa6e61cd9ed8a37a44743cf`) used the corrected revision-4 profile but failed later on judge context. A new immutable attempt is required after qualification.
- [x] (2026-09-20 17:42Z) Diagnosed attempt `20260920c` after 11 completed updates: it failed before update 12 because a judge request reserved 16,384 output tokens after at least 16,385 prompt tokens against a 32,768-token serving limit. Preserved the step-10 checkpoint and native traces; no restart has been submitted.
- [ ] Qualify a bounded judge request contract and a Gemma serving context that fits representative long episodes at the requested concurrency. Inspect structured-output length and actual prompt-token distribution, including copied tool contracts and observations. Re-run the reviewed rubric calibration gate before promoting a new scorer version.
- [ ] Decide and validate the inference path on the released vLLM fork: retain the currently selected Gemma MTP-2 path unless DSpark's separately measured speed and scorer calibration both qualify. Verify actual compilation/CUDA-graph and speculative counters rather than inferring them from `enforce_eager: false`.
- [ ] Run the common held-out evaluation once against each materialized adapter.
- [ ] Reconcile the comparison and record final run identities and evidence.

## Surprises & Discoveries

- Observation: OLMo3 is not a valid no-oversampling algorithm in this framework.
  Evidence: work-package validation returned `OLMo 3 requires active group sampling`; `GRPOSettings` enforces this contract, while adaptive curriculum independently supports the `initial_batch` sampling mode with ordinary GRPO.

- Observation: The first live attempt failed before GPU allocation because the framework emitted a nonexistent vLLM `AttentionConfig.backend_priority` field.
  Evidence: vLLM raised `ValidationError: backend_priority Unexpected keyword argument`. The released fork exposes scalar `AttentionConfig.backend`; compilation now resolves the ordered framework policy to that scalar runtime field.

- Observation: After the attention fix, vLLM reached GPU initialization but the old colocated utilization setting requested 18.99 GiB while the trainer left 9.48 GiB free.
  Evidence: vLLM reported `Free memory on device cuda:0 (9.48/94.97 GiB) ... less than desired GPU memory utilization (0.2, 18.99 GiB)`. No unrelated dstack training run was active. Revision 2 uses 9% with the same c32 scheduler and explicit 4 GiB KV cache.

- Observation: Native vLLM FA4 is not an SM120 backend even though SM120 is a Blackwell architecture.
  Evidence: attempt `lfm26-adaptive-grpo-50-c32-20260920c` passed the attention-field and memory gates, then repeatedly reported that FA4 supports compute capabilities 9.x, 10.x, and 11.x before failing attention initialization on compute capability 12.0. The earlier favorable SM120 FA4 result came from the separately extracted kernel, not native `FLASH_ATTN`.

- Observation: The apparent colocated KV-cache pressure in attempts d/e was caused by a retained standalone inference service, not the selected LFM profile.
  Evidence: host `nvidia-smi` identified `VLLM::EngineCore` in `vllm-sm120-dev.service`, serving K2-Horizon/Uno at 65K context and c32 while holding 80.8 GiB. It remained after each failed dstack container terminated. Stopping the reversible user service released the allocation; the comparison keeps its intended 4 GiB BF16 KV arena.

- Observation: The corrected adaptive arm saturates the GPU and its warm rollout path is substantially faster than cold startup.
  Evidence: the first three updates completed 32/32 rollouts each with zero failures and no refill rounds. Rollout throughput was 868.96, 1708.13, and 1086.20 tok/s; step times were 508.79, 429.92, and 404.80 seconds. Consecutive dstack samples showed repeated 100% generation intervals and sustained 100% actor updates; late-tail generation dropped toward 52-58% as requests completed. Peak KV usage was 76.3%, 80.5%, and 46.2%.

- Observation: Long-sequence GRPO correction reaches the configured sequence-level importance-ratio floor.
  Evidence: `train/rl/importance_sampling_ratio_mean` was 0.1 on all first three updates while mean token-logprob delta fell from 0.0507 to 0.0457. Rewards and gradients remained finite and reward mean rose from 0.4005 to 0.4663, so the run continues, but a token-level correction comparison is a recorded follow-up rather than a silent mid-run semantic change.

- Observation: The 4,096-token completion ceiling is a quality constraint even though the adaptive arm is operationally healthy.
  Evidence: 359 of 1,600 trajectories (22.44%) reached the completion cap across 50 updates. The run was not interrupted because optimization stayed finite and the operating contract forbids restarting a healthy run, but a larger completion budget is required before calling this the final best-quality profile.

- Observation: The first GDPO attempt reached Gemma initialization but its revision-3 judge profile narrowly underprovisioned full-context KV memory.
  Evidence: vLLM measured 9.92 GiB available versus 10.50 GiB required to admit one 32,768-token request (estimated limit 30,944). Revision 4 raises the judge fraction from 45% to 47%, about 1.9 GiB of card-level headroom, rather than reducing the declared context and silently changing judge semantics.

- Observation: The corrected GDPO attempt reached 11 optimizer updates but the nominal input budget did not bound rendered judge prompts.
  Evidence: `20260920c` failed at the next collection with vLLM HTTP 400: at least 16,385 prompt tokens plus a 16,384-token output reservation exceeded the configured 32,768-token context. The selected Verifiers judge declares `input_budget_tokens: 12288`, but its `score` method passes the full serialized episode and selected tool definitions to `complete` without token-count enforcement. Earlier updates retained fewer complete groups when individual judge calls failed; group-atomic admission already drops invalid groups when at least one complete group remains, but correctly refuses an all-invalid batch rather than fabricating rewards.

- Observation: The active GDPO profile did not use the separately benchmarked Gemma/DSpark judge path.
  Evidence: the work package selects Gemma MTP-2 on the released vLLM fork, Triton attention, `enforce_eager: false`, and 32K context. The retained DSpark qualification used an unreleased dirty vLLM runtime and showed 16/16 schema-valid verdicts but only 42.31% reviewed constraint pass and 10 false-perfect verdicts under its initial rubric; later prompt versions improved but did not establish a complete replacement gate. The 50-case two-pass diagnostic produced 2,435 output tokens on average (maximum 3,703), showing that a 16K-per-call reservation deserves remeasurement rather than automatic retention. This two-pass output distribution is not a direct-mode guarantee.

## Decision Log

- Decision: Interpret “100 steps total” as two matched 50-step arms rather than 100 steps per arm.
  Rationale: This preserves an equal comparison and matches the explicit total budget.
  Date/Author: 2026-09-20 / Codex

- Decision: Name the first arm adaptive GRPO, not VORTEX.
  Rationale: VORTEX combines the OLMo3 update recipe, active group refill, and adaptive curriculum. Removing oversampling removes active refill, and the typed framework contract correctly rejects OLMo3 without it. An honest ablation uses GRPO plus adaptive initial-batch allocation.
  Date/Author: 2026-09-20 / Codex

- Decision: Use native FA2 for LFM on the RTX PRO SM120 target and the qualified Triton MTP-2 path for the Gemma judge, subject to pre-submission policy compilation.
  Rationale: Live startup proved that vLLM native FA4 rejects compute capability 12.0; the earlier FA4 result used the separately extracted SM120 kernel and cannot be represented as native `FLASH_ATTN`. Gemma MTP-2 on Triton remains the retained correct path.
  Date/Author: 2026-09-20 / Codex

- Decision: Do not treat a missing or malformed rubric score as zero, and do not count a skipped all-invalid collection as an optimizer update.
  Rationale: GDPO's per-group component normalization needs valid reward vectors. The current admission layer already retains complete groups without oversampling; an all-invalid batch has no unbiased update. Fix request budgeting first, preserve failed trace reasons, and add an explicit bounded whole-batch recovery policy only if it can maintain the no-oversampling and 50-real-update contracts.
  Date/Author: 2026-09-20 / Codex

- Decision: Keep DSpark and actual CUDA-graph execution as release gates, not inferred performance claims.
  Rationale: the running selection was MTP-2 and `enforce_eager: false` is only an instruction allowing compilation, not proof that its target and proposer replayed compiled graphs. The DSpark speed result used an unpublished runtime and its rubric calibration was not yet acceptable.
  Date/Author: 2026-09-20 / Codex

## Outcomes & Retrospective

Configuration and live qualification of the adaptive-GRPO arm are complete. Run `lfm26-adaptive-grpo-50-c32-20260920f` succeeded with 50 updates and immutable model, recovery, and summary artifacts. GDPO attempt `20260920c` completed 11 updates and then failed on an unbounded judge request. It must not be presented as a 50-update result. Judge-budget repair, runtime qualification, a new immutable GDPO attempt, both held-out evaluations, and final comparison remain.

## Context and Orientation

`apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml` contains model, environment, training-setting, and inference selections. A selection is an immutable, named configuration value. The work packages under `apps/lab/.posttrain/work_packages/` bind those values into one train or evaluation job. `apps/lab/src/posttrain_lab/qualification/gates.toml` records the intended qualification and prevents an experimental result from being mistaken for released evidence.

The adaptive-GRPO arm chooses task classes before the initial batch is collected. It does not refill rejected or uniform-reward groups. The GDPO arm uses a Gemma 4 12B MTP-2 judge to produce component rewards. “Concurrency 32” means the inference scheduler may keep 32 sequences in flight; it does not mean 32 separate model processes. A “policy-version fence” prevents a rollout from using stale KV-cache state after a LoRA update.

The held-out evaluator is intentionally separate from training. It accepts a materialized training run through `--model-from-run`, so the exact trained adapter and lineage are preserved instead of evaluating a mutable path.

## Plan of Work

First, validate the catalog and compile both dstack job plans using Posttrain 0.4.4 installed from the stable package index. Confirm that the LFM rollout resolves to native FA2 with CUDA graphs and prefix caching, while the Gemma judge resolves to Triton attention with MTP-2. Reject any silent fallback that changes the intended operating point.

Second, run focused catalog, qualification-gate, training-policy, and work-package tests. Commit the configuration and plan so each submitted job has an immutable source revision. Submit the adaptive-GRPO and GDPO work packages with distinct run IDs. Because one RTX PRO 6000 may serialize jobs, queueing is acceptable; overlapping two memory-heavy training jobs on one GPU is not.

Third, observe each arm through at least three completed optimizer updates. Capture optimizer step, trajectory counts, truncation, rollout throughput, queue depth or scheduler occupancy, KV-cache utilization and hit rate where applicable, GPU utilization and memory use, speculative acceptance for the Gemma judge, LoRA/policy version, loss, KL, gradient norm, and reward statistics. A run is not healthy merely because it remains alive. If metrics indicate underutilization, preemption, cache thrash, invalid values, stale weights, or malformed rewards, stop the affected run, change only the evidence-supported bottleneck, recompile the job, and resubmit under a new run ID.

Before a GDPO retry, inspect the pinned `environments/automationbench_v1/src/automationbench_v1/episode_prompt.py` and `judge.py` in `../verifiers-environments` at the catalog's exact source revision. The current direct prompt copies the full role-separated episode, selected tool definitions, message IDs, and rubric labels; tool observations are JSON-wrapped but not semantically shortened. Introduce a tokenizer-measured input and output reservation that cannot exceed the selected server context, and preserve evidence-index validity if redundant fields are removed. Any prompt or schema change requires a new scorer/source revision, frozen replay calibration, and an immutable Posttrain pin. At least one long episode that previously exceeded 32K must complete or be explicitly marked unjudgeable before training begins. Do not silently truncate material tool results.

Qualify the Gemma runtime separately at 32 concurrent judge requests: verify the server's actual max context and KV admission, inspect compile/CUDA-graph logs, target and proposer graph coverage, MTP or DSpark draft/acceptance counters, GPU utilization, and valid structured outputs. The current profile permits compilation (`enforce_eager: false`) but does not prove it. If testing DSpark, use its pinned BF16 block-7 draft on a committed CarbonTeq vLLM fork, re-run the reviewed rubric label gates, and only then select it in a versioned inference binding. Do not substitute the smaller Spark-X2.5-4B judge; it is a different model and scorer. Keep the current MTP-2 profile as the comparison baseline.

Finally, after both 50-step descendants materialize, submit the same held-out evaluation twice using each training run as `--model-from-run`. Compare task-level means, completion and truncation, tool-call correctness, reward components, throughput, and resource efficiency. Keep training evidence and held-out evidence distinct.

## Concrete Steps

Run from `/home/hammad/projects/rl/apps/lab` unless stated otherwise:

    uv run posttrain work-package validate lfm26_automationbench_adaptive_grpo_no_oversampling_50_local_v1.yaml
    uv run posttrain work-package validate lfm26_automationbench_gdpo_episode_50_local_8x4_v1.yaml
    uv run posttrain work-package validate lfm26_automationbench_heldout_eval_c32_v1.yaml

Create a temporary clean Python 3.13 environment and install `posttrain==0.4.4` and `posttrain-lab==0.4.4` from `https://pypi.lan/carbonteq/stable/+simple/`. Use that CLI for `posttrain doctor`, `posttrain job plan`, and `posttrain job run`. Do not bypass a failed doctor check.

Run focused tests from `/home/hammad/projects/rl`:

    uv run pytest apps/lab/tests/test_catalog.py apps/lab/tests/test_qualification_gates.py packages/train/tests/test_adaptive_curriculum.py packages/train/tests/test_api.py packages/work/tests/test_validation.py -q
    uv run ruff check apps/lab packages/train packages/work
    git diff --check

For the judge repair, first confirm the pinned environment revision and local checkout state without editing the latter implicitly:

    git -C /home/hammad/projects/verifiers-environments rev-parse HEAD
    git -C /home/hammad/projects/verifiers-environments status --short

Run the environment's judge tests there after its prompt-builder change, then run `uv run pytest packages/jobs/tests/test_native_judges.py packages/train/tests/test_reward_admission.py apps/lab/tests/test_work_packages.py -q` from `/home/hammad/projects/rl` after the immutable pin update. Compare output lengths with `jq '[.cases[].completion_tokens]|{count:length,min:min,max:max,mean:(add/length)}' outputs/qualification/lfm26-oversample-training-sample-50-20260920/gemma4-dspark-v16-report.json`; this is the two-pass diagnostic, not a substitute for replaying the promoted direct or framed production protocol. Use `scripts/qualification/episode_judge_vllm_benchmark.py` on frozen retained inputs against a live versioned endpoint and verify its `structured_output_valid`, finish reasons, prompt/output tokens, and reviewed labels before submitting a new training run. The endpoint and GPU credentials are external preconditions, never stored in this plan.

Submit training only after both job plans resolve the requested backend policies. Use stable, descriptive run IDs containing the algorithm, 50-step budget, and date. Evaluation submission waits for its source training run to materialize.

## Validation and Acceptance

Static acceptance requires all three work packages to validate and the focused test suite, Ruff, and `git diff --check` to pass. Compilation acceptance requires both dstack plans to select the released runtime and compatible inference backends before submission.

Live acceptance for each arm requires three completed optimizer updates with finite loss, reward, KL, and gradient metrics; 8 prompt groups and 32 trajectories requested per step; no hidden candidate refill; changing policy versions after updates; and no malformed reward or stale-cache warnings. GPU acceptance requires sustained utilization during model-active windows rather than an average diluted by environment waits. If the runtime exposes scheduler occupancy, queue depth, KV-cache use, and token throughput, those values are the primary explanation of saturation. Overall GPU percentage alone is insufficient.

Judge acceptance additionally requires a deterministic long-episode replay whose exact rendered token count plus reserved output and safety margin is within the selected serving context; a 400 context response, absent component vector, or invented zero score fails this gate. A single failed judge may exclude its whole prompt group if at least one complete group remains; an all-invalid collection must produce an explicit bounded recovery/stop outcome, never a fictitious optimizer step. The selected runtime must show observed compile/graph and speculative acceptance evidence. Faster DSpark decoding alone is insufficient without calibration parity on reviewed labels.

Final acceptance requires 50 completed updates in each arm and two held-out evaluations derived from the immutable outputs. Any restarted attempt remains in lineage and is reported rather than overwritten.

## Idempotence and Recovery

Validation and job planning are read-only and safe to repeat. Every submitted attempt must use a new run ID. Never resume a run after changing configuration; create a descendant attempt so the changed inference policy is visible. If the only RTX PRO is occupied, leave the second arm queued instead of evicting unrelated work. Do not delete partial evidence: failed attempts are useful saturation and correctness records.

## Artifacts and Notes

The first validation exposed a useful contract boundary:

    error: OLMo 3 requires active group sampling

After changing the arm to GRPO with adaptive initial-batch allocation, all three compositions reported `Static composition validation: complete`.

## Interfaces and Dependencies

The experiment uses Posttrain 0.4.4, the released CarbonTeq vLLM `0.29.1.dev2` runtime, TRL LoRA training, the AutomationBench Verifiers environment, Trackio/Doris evidence storage, and dstack scheduling on the RTX PRO 6000. The LFM rollout inference selection is `inference/lfm2.5-2.6b-vllm-automationbench-rollout-local-c32-4k@2`. The Gemma judge selection is `inference/gemma4-12b-vllm-automationbench-judge-mtp2-local-32k@4`. The held-out evaluator uses `inference/lfm2.5-2.6b-vllm-automationbench-eval-local@2`.

Revision note (2026-09-20): Created this plan after static validation rejected an invalid OLMo3-without-active-sampling interpretation; the plan records the corrected adaptive-GRPO design and the live operating gates. Updated after live attempts found both a framework-only backend-priority field at the vLLM boundary and a pre-0.29 colocation memory fraction that exceeded actual free device memory. Revised after GDPO attempt `20260920c` failed at 11 updates to require tokenizer-measured judge budgeting, explicit all-invalid-batch behavior, and observed—not configured—compilation and speculative evidence before another retry.
