# AutomationBench: SDPO

On-policy self-distillation (SDPO) of the Qwen3.5-4B SFT adapter. Each round, the student makes fresh attempts on
the training tasks. A teacher, which is a running average of the student itself, then re-reads every attempt with the
task's outcome and, for failures, written feedback. The student is trained towards the teacher's next-token
distribution on its own tokens. This branch builds on `feat/automationbench-sft`. Evaluation, the environment and
the SFT adapter are described in [README.md](README.md).

| Adapter | test120 | guard10 | H2 (102 held-out training tasks) |
| --- | --- | --- | --- |
| [carbonteq/qwen3.5-4b-automationbench-sft](https://huggingface.co/carbonteq/qwen3.5-4b-automationbench-sft) (start) | 42.4% | 78.8% | 24.8% |
| [carbonteq/qwen3.5-4b-automationbench-sdpo](https://huggingface.co/carbonteq/qwen3.5-4b-automationbench-sdpo) (round 1) | 50.6% | 66.3% | 34.4% |

Qwen3.5-397B scores 49.6% on test120 with 2 attempts per task. The run trained 3 rounds; rounds 2 and 3 scored 31.7% and
29.9% on H2. Round 1 was chosen before test120 was run, by its H2 score.

## Layout

| Path | What it is |
| --- | --- |
| `scripts/qualification/sdpo_ab_runner.py --phase online-train` | the training entry: rounds of attempts, teacher contexts, loss, checkpoints |
| `scripts/qualification/sdpo_online.py` | the online rounds: builds each attempt's teacher context, marks the trained tokens, and runs the EMA teacher and the loss |
| `scripts/qualification/feedback_c.py`, `feedback_versions.py`, `feedback_z.py` | turn the engine's findings into the teacher's text (design `S1FUHWNLGOJYIV` is the one trained) |
| `scripts/qualification/sdpo_ab_engine.py`, `sdpo_a3.py` | load the feedback engine and build its two views: values hidden, and values shown for pointers |
| `scripts/qualification/abce_vendor/` | the feedback engine exactly as this run used it; the maintained version is [carbonteq-ai/automationbench-context-engine](https://github.com/carbonteq-ai/automationbench-context-engine), which gives identical feedback on this run's attempts |
| `scripts/qualification/fixtures/automationbench/sdpo-h1-train.json` | the 78 training tasks (H1) |
| `scripts/qualification/fixtures/automationbench/sdpo-h2-heldout.json` | the 102 held-out training tasks (H2) used to pick the round |

## How a round works

1. **Attempts:** the student makes 8 attempts per task, sampled with the evaluation's settings (thinking on, 50 turns,
   32,768-token replies).
2. **Teacher context:** the same conversation plus the task's outcome, built for every attempt.
   - A solved attempt gets an outcome line.
   - A failed attempt gets the engine's feedback. It names the checks the attempt missed: required actions not taken,
     forbidden actions taken, and wrong values. Each note sits at the turn where the attempt went wrong; values the
     attempt never saw stay hidden.
3. **Trained tokens:** reasoning, tool calls, and the token that decides between acting and replying. Free text
   inside tool calls is skipped. A failed attempt is trained up to its first wrong decision, a solved attempt in full.
4. **Loss:** reverse KL between student and teacher on the teacher's top 20 tokens plus the remaining tail. The
   importance weight is capped at 2.
   - Each update covers 16 attempts.
   - The optimiser is AdamW: learning rate 3e-5 constant, weight decay 0.01, gradient clip 1.0.
   - The teacher moves 1% towards the student after every update.

Round 1 admitted 622 of 624 attempts and took 39 updates. Two attempts had no engine feedback and were left out.

## Train

```bash
export SDPO_AB_ENV_REPO=../verifiers-environments SDPO_AB_ENV_COMMIT=28146f9 SDPO_AB_WORLD_DATE=2026-10-28
export SDPO_AB_STAGE=/tmp/ab-stage SDPO_AB_WS=<ssh wrapper that runs a command on the workstation>
scripts/qualification/sdpo_ab_launch.sh sdpo-ab-sdpo-train 24h \
  /code/scripts/qualification/sdpo_ab_runner.py --phase online-train --protocol ours --sampling card-thinking \
  --thinking on --reply-limit 32768 --context-limit 131072 --rollout-timeout 10800 --kv-cache-gib 48 \
  --concurrency 32 --lora-rank 16 --lora-alpha 32 \
  --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --fit /code/scripts/qualification/fixtures/automationbench/sdpo-h1-train.json \
  --adapter /evidence/<folder with the SFT adapter> \
  --online-rounds 1 --rollouts-per-task 8 --online-design S1FUHWNLGOJYIV --online-ema 0.01 --lr 3e-5 \
  --classic-batch 16 --loss-tokens reasoning+calls --online-decision-token --online-call-text skip \
  --seed-salt sdpo-close-sdpo-ab-20261122-f1-train --output /evidence/sdpo-ab-sdpo-train/run
```

The 50.6% run used these arguments with `--online-rounds 3`. Its round-1 checkpoint is
`run/online/checkpoint-1`.
- **Duration:** the 3 rounds took 8.4 hours on one RTX PRO 6000, about 2.8 hours per round.
- **Memory:** a round peaks near 27 GB of GPU memory.

To choose between rounds, run each checkpoint on H2 with the evaluation command in [README.md](README.md). Use
`--tasks /code/scripts/qualification/fixtures/automationbench/sdpo-h2-heldout.json` and
`--seed-salt sdpo-weekend-w6-train204`; one read takes 2.4 hours.

## Outputs

`run/online/round-N/` holds each round's attempts (`episodes.jsonl.gz`, `chains.jsonl.gz`, `outcomes.json`) and
`admission.jsonl.gz`, which records whether each attempt was trained and its teacher length. Training losses are in
`steps.jsonl`. `run/online/checkpoint-N/` holds the adapter after round N, and `run/online-summary.json` gives each
round's success, reasoning length, loss and peak memory.

## Code provenance

The training code on this branch is the code the 50.6% run trained with, minus the files it never imports. The
`online-train` phase and its runner were added after the evaluation commit, and both phases share
`sdpo_ab_runner.py`. The evaluations in the table ran its `collect` phase as it stood at that evaluation commit.
