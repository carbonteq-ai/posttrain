# AutomationBench: evaluation and SFT

Post-training of Qwen3.5 on [AutomationBench](https://github.com/zapier/AutomationBench) business-workflow tasks.
This branch holds the evaluation runner and the supervised fine-tuning (SFT) pipeline. Two branches build on it:

| Branch / repository | What it adds |
| --- | --- |
| `feat/automationbench-sft` (this branch) | evaluation runner, SFT data pipeline, SFT training configs, framework fixes for Qwen3.5 adapters |
| `feat/automationbench-dpo` | DPO pair builder, chunked DPO loss and its config (`docs/automationbench/dpo.md`) |
| `feat/automationbench-sdpo` | on-policy self-distillation (SDPO) with written feedback (`docs/automationbench/sdpo.md`) |
| [carbonteq-ai/verifiers-environments#7](https://github.com/carbonteq-ai/verifiers-environments/pull/7) | the AutomationBench environment with the task fixes every run used (version 1.1, commit `28146f9`) |
| [carbonteq-ai/AutomationBench `fix/six-task-scored-requirements`](https://github.com/carbonteq-ai/AutomationBench/tree/fix/six-task-scored-requirements) | benchmark data fix that the environment vendors |
| [carbonteq/automationbench-posttrain-data](https://huggingface.co/datasets/carbonteq/automationbench-posttrain-data) | training data and raw evaluation records (private) |

Trained adapters (private Hugging Face repositories; each card gives its settings and how to load it):

| Adapter | test120 | guard10 |
| --- | --- | --- |
| Qwen3.5-4B, untrained | 28.3% | 71.2% |
| [carbonteq/qwen3.5-4b-automationbench-sft](https://huggingface.co/carbonteq/qwen3.5-4b-automationbench-sft) | 42.4% | 78.8% |
| [carbonteq/qwen3.5-9b-automationbench-sft](https://huggingface.co/carbonteq/qwen3.5-9b-automationbench-sft) | 46.3% | 75.0% |

## Pages

The four pages explaining the work are kept here as standalone HTML (open them in a browser), copied from their
published versions on 25 Nov 2026:

| Page | What it covers |
| --- | --- |
| [pages/handover.html](pages/handover.html) | what to load, where everything is, how to run it, pitfalls ([live](https://claude.ai/artifact/7C7CcKHGXQosEafj8zXPah)) |
| [pages/post-training-map.html](pages/post-training-map.html) | the story: starting point, how the sets were built, scoring, SFT, DPO, SDPO, lessons ([live](https://claude.ai/artifact/DGhMxqevpwUdBCbS18azuQ)) |
| [pages/task-deep-dive.html](pages/task-deep-dive.html) | every model on every test task ([live](https://claude.ai/artifact/4L3YcbakSzDMGQsDiDc4Dp)) |
| [pages/sft-story.html](pages/sft-story.html) | how the SFT data was made and checked ([live](https://claude.ai/artifact/2oCnh7zJDPNCN5iwUhnhXt)) |

## Layout

| Path | What it is |
| --- | --- |
| `scripts/qualification/sdpo_ab_runner.py` | the runner: rolls out the model on AutomationBench tasks in vLLM and scores each attempt (`--phase collect`) |
| `scripts/qualification/sdpo_ab_launch.sh` | submits one runner job to the GPU workstation as a dstack task |
| `scripts/qualification/sdpo_frozen_environment.py`, `sdpo_ab_execution.py`, `sdpo_ab_replay.py`, `sdpo_ab_world.py` | the environment pin, tool execution and world replay the runner uses |
| `scripts/qualification/fixtures/automationbench/` | task lists: `test120.json` (120 test tasks, 20 per domain), `guard10.json` (10 report-only test tasks), `sets-phase2-frozen.json` (all frozen sets) |
| `scripts/qualification/sftgen/` | SFT data pipeline: teacher attempts, row building, data checks, render check |
| `scripts/qualification/sft_stage_adapter.py` | copies a trained adapter out of a framework run into the evaluation volume |
| `apps/lab/.posttrain/catalog/qwen35-4b-automationbench-sft-v4.yaml` | SFT data, training settings and LoRA binding |
| `apps/lab/.posttrain/work_packages/qwen35_{4b,9b}_automationbench_sft_v4_e3.yaml` | the 4B and 9B SFT jobs |
| `packages/train`, `packages/data` | framework fixes: Qwen3.5 thinking renderer, chunked NLL loss, image-text model class (adapter weights load in vLLM) |

## Evaluate a model

The runner executes in the posttrain job image on the GPU workstation. `sdpo_ab_launch.sh` packs
`scripts/qualification` and the environment checkout, and submits them to dstack. It refuses to launch while another
dstack run or GPU process is active.

```bash
export SDPO_AB_ENV_REPO=../verifiers-environments   # checkout of the environment with the task fixes
export SDPO_AB_ENV_COMMIT=28146f9                     # environment version 1.1
export SDPO_AB_STAGE=/tmp/ab-stage SDPO_AB_WS=<ssh wrapper that runs a command on the workstation>
export SDPO_AB_WORLD_DATE=2026-10-28                  # every evaluation pins the tasks' world clock to this date
scripts/qualification/sdpo_ab_launch.sh sdpo-ab-eval-sft-test120 12h \
  /code/scripts/qualification/sdpo_ab_runner.py --phase collect --protocol ours --sampling card-thinking \
  --thinking on --reply-limit 32768 --context-limit 131072 --rollout-timeout 10800 --kv-cache-gib 48 \
  --concurrency 32 --lora-rank 16 --lora-alpha 32 \
  --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --tasks /code/scripts/qualification/fixtures/automationbench/test120.json --attempts 8 \
  --seed-salt sdpo-ab-20261112-p6-test120 --label sft-test120 \
  --adapter /evidence/<folder with adapter_model.safetensors> --output /evidence/sdpo-ab-eval-sft-test120/run
```

- **guard10:** `--tasks .../guard10.json --seed-salt sdpo-ab-20261112-p6-guard10`.
- **9B:** the 9B snapshot (`Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`) and
  `--kv-cache-gib 32`; 48 GiB runs out of GPU memory.
- **Untrained model:** leave out `--adapter`.
- **Volumes:** `/evidence` is the workstation's `/var/lib/posttrain/qualifications/sdpo-ab-runs`, and `/hf` its
  Hugging Face cache.
- **Duration:** on one RTX PRO 6000, one test120 read (960 attempts) took 2.8 hours for the 4B and 3.1 hours for
  the 9B; guard10 took 18 minutes.

The run folder gets `episodes.jsonl.gz` (one line per attempt with its score), `chains.jsonl.gz` (each attempt's
conversation), `outcomes.json` and `summary.json` (success, partial credit, harm and tool errors per task), and
`adapter-check.json`. That last file records how much the adapter changed the model's scores; a loaded adapter must
change them.

## Build the SFT data

The data is one solved Qwen3.5-397B attempt per training task, generated inside the same evaluation loop the 4B is
scored in (same tools, prompt, limits and thinking mode). `sftgen/settings.py` lists the folders and interpreters the
pipeline uses; set them through `SFTGEN_*` variables.

1. `teacher_proxy.py` serves an OpenAI-compatible endpoint that forwards to OpenRouter (`qwen3.5-397b-a17b`,
   provider pinned), with retries and a spend cap. It reads `OPENROUTER_API_KEY`.
2. `generate_rounds.py --tasks-json fixtures/automationbench/sets-phase2-frozen.json --set train-d204 --base-url
   http://127.0.0.1:PORT/v1 ...` runs `teacher_runner.py` on every training task (attempts in chunks). It then builds
   rows from solved attempts with `episode_rows.py`, and checks each row with `validate_row.py`. Test and guard tasks
   are refused.
3. `render_check.py` renders every row with the framework's Qwen3.5 thinking renderer and checks the token counts.
4. The admitted rows are SFT data version 3 (97 rows). It is `sft/data-v3.jsonl` in the dataset repository, along with
   every teacher attempt under `sft/teacher-traces/`.

## Train

Put the data at `apps/lab/.posttrain/datasets/qwen35-4b-ab-sft-v4-all.jsonl`; the catalog file's header shows how.
Then submit the job:

```bash
cd apps/lab
posttrain job run .posttrain/work_packages/qwen35_4b_automationbench_sft_v4_e3.yaml --job train \
  --provider dstack --timeout-seconds 43200 --run-id <run id>
```

| Setting | Value |
| --- | --- |
| LoRA | rank 16, alpha 32, dropout 0, all linear layers except the vision tower |
| Optimiser | learning rate 1e-4, cosine schedule, 5% warm-up, effective batch 16, 3 epochs (21 steps) |
| Run | 32,768-token rows, bf16, gradient checkpointing, seed 20261103 |
| Loss | reasoning, visible text, calls and reply; the prefilled `<think>` opener is masked |
| Evaluated adapter | epoch 3 |

For the 9B, use `qwen35_9b_automationbench_sft_v4_e3.yaml`. It has the same data and settings; only the model
binding differs. The framework writes the adapter under `/var/lib/posttrain/runs/<id>/scratch/<id>/training/sft/`.
`sft_stage_adapter.py` copies it into the evaluation volume, run as a short task through `sdpo_ab_launch.sh`.

`posttrain job run` packs the framework source of the checkout that holds the `posttrain` command. Run it from this
branch's environment, so the job carries these framework fixes.

## Pins

| Item | Value |
| --- | --- |
| Base models | `Qwen/Qwen3.5-4B` `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; `Qwen/Qwen3.5-9B` `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Job image | `registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337` |
| Environment | verifiers-environments `28146f9` (task fixes 1.1). Version 1.0 (`8699430`) differs only in one training task's check; test120 and guard10 are identical. |
| Evaluation code | this branch's runner. The 4B SFT adapter's own evaluation ran on an earlier runner commit with the same evaluation settings. |
