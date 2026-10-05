# Policy-update engine scale harness

`simulate.py` replays the part of a resolved SAMPO training round that runs after episodes are collected, on real retained episodes, and times each stage. It exists so engine changes can be measured in minutes on a workstation instead of by submitting 2.6B training jobs. The plan it serves is `docs/plan/columnar-policy-update-engine.md`.

## What it runs

The harness loads the project catalog, builds the same Verifiers training bridge a `train/sampo-turns@1` job builds (environment, SAMPO settings, manifest-steps turn-reward projection, `causal-text@1` conditioning), projects every retained episode into an `EnvironmentRollout`, selects the first 16 complete prompt groups whose shaped rewards differ (as active collection does), seals the population, admits it (`AdmittedNativePopulation.from_retained_artifact`), builds the TRL population view and sampled scores, and computes collection telemetry. With `--train` it then freezes reference scores and runs every optimizer update through `ResolvedTRLPopulation.loss` and `backward` on the GPU.

In `--train` mode the transformer is replaced by a stand-in: `sampled_logprobs` returns one tensor per turn that depends on a single parameter. Everything else is the job's code, including one score tensor per sampled token, so the reported time is the engine's own CPU and synchronization overhead, not model compute.

## Setup

The harness needs the rl workspace with the Verifiers extra and the AutomationBench environment at the run's pins. From the rl worktree root:

    UV_HTTP_TIMEOUT=300 UV_PROJECT_ENVIRONMENT=/home/hammad/projects/sim/postcollect/venv uv sync --all-packages --locked --python 3.13 --extra verifiers
    uv pip install --python /home/hammad/projects/sim/postcollect/venv/bin/python -e <verifiers-environments checkout at the run's revision>/environments/automationbench_v1

The data directory holds the run's `episodes.jsonl` (copied from the job's `training/sampo/episodes.jsonl`). The first run projects the episodes and writes `population.pkl` and `episodes-projected.jsonl` next to it; later runs start from those (delete both to rebuild). For r6 the data directory is `/home/hammad/projects/sim/data-r6` (1.3 GB of episodes, not committed).

    /home/hammad/projects/sim/postcollect/venv/bin/python docs/research/policy-update-engine-scale/simulate.py \
        --data /home/hammad/projects/sim/data-r6 --project apps/lab --out /tmp/engine-scale --train

`--profile "<stage name>"` writes a cProfile file for one stage into the output directory.

## Baseline (r6, 2026-10-05)

Run `manifest-steps-26-sampo-100-g16x8-20261005-r6` (LFM2.5-2.6B, 16 turns, 4,096-token replies): 160 collected episodes; 128 selected (16 groups of 8); 806,662 completion tokens; 515,868 sampled tokens; 784 assistant turns; 4 optimizer updates per round. Times on the local workstation (one core, RTX 3070 Ti for `--train`):

    stage                         unpatched engine     with one-pass grouping and conditioning decode
    admit (resolve+plan)          > 23 min (cut off)   44.0 s
    trl population view                 -                6.9 s
    reference scores                    -               21.8 s
    update 0 loss / backward            -              116.3 s / 21.6 s
    update 1 loss / backward            -               97.8 s / 38.5 s
    update 2 loss / backward            -              100.0 s / 49.4 s

Process memory grew from 6.1 GB to 13.9 GB over three updates. Every selected episode's turn contexts form an exact prefix chain; scoring per turn forwards 6,362,187 tokens per pass where one pass per episode would forward 1,363,646.

For comparison, the older TRL SAMPO system (run `lfm26-sampo-cont100-g30x4-16t-lr6e5-kl1e2-20260930-r1`, 120 episodes per update) averaged 467 s per update: 198 s of rollouts and 219 s of actor update.
