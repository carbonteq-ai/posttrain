# Posttrain Lab

Lab is the framework's qualification project. It exercises concrete SFT, GRPO,
distillation, serving, evaluation, tracking, and hardware combinations before a
release. It is not a dependency of generated product projects and it is not the
home of standard job definitions.

## Product path (do this first)

For a product project, start with the primary CLI — not lab:

```bash
posttrain init support-agent --template sft
cd support-agent
.venv/bin/posttrain dataset validate datasets/posttrain-sft-smoke@1
.venv/bin/posttrain work-package validate sft.yaml
```

The generated project gets standard definitions and tracking from
`posttrain.jobs`. Use `--template grpo` for the environment-backed starter.
Neither starter imports `posttrain_lab`.

## Qualification project (this repo)

Lab is the qualification project at `apps/lab`; its tracked control tree lives
under `apps/lab/.posttrain/`. Its `project.toml` points at
`posttrain_lab.entry:configure`, which builds the standard `JobRuntime` and
attaches git source metadata. From the workspace root, validate YAML work
packages with the primary CLI and **no** `--host`:

```bash
uv run --package posttrain-lab posttrain --project-root apps/lab work-package validate \
  foundation_screen.yaml

uv run --package posttrain-lab posttrain --project-root apps/lab work-package validate \
  automationbench_zapier_grpo.yaml
```

Remote GPU release evidence uses the Lab-owned
`tests/fixtures/remote_gpu_project` and the
primary `posttrain work-package run` path on the remote host.

## Qualification gates

Lab owns the reviewed inventory of framework qualification gates in
`src/posttrain_lab/qualification/gates.toml`. During the additive migration,
the manifest classifies the work packages currently owned by the root
qualification project; it does not submit work or duplicate provider logic.

```bash
uv run --package posttrain-lab posttrain-lab qualification list --project-root apps/lab
uv run --package posttrain-lab posttrain-lab qualification list --project-root apps/lab --json
```

Every work-package YAML must occur exactly once in the registry. An entry
records its lifecycle, tier, selected job, expected job kind, and evidence
acceptance adapter. The two active release gates are the SFT data-preparation
and managed GSM8K evaluation qualifications. Nine active extended gates cover
distinct framework contracts. The remaining candidate experiments are retained
as evidence, not active Lab gates: each names an experiment family, hypothesis,
responsible maintainer area, and the condition that replaces or promotes it.
The DAPO and SAMPO acceleration variants are therefore two candidate matrices,
not eight independent qualification requirements. A candidate must be executed
through the normal `posttrain work-package` path and cannot silently satisfy a
release or extended qualification requirement. It is an actionable holding
state, not an archive: its replacement condition must name promotion,
replacement, retirement, or deletion. When that condition is met, update the
record first; a later source-removal milestone may then remove its YAML only
after the selected successor has demonstrated command and evidence parity.

## Lab scenario CLI

Use Lab when qualifying framework backends or the repository's reference
scenarios that are not yet expressed only as YAML:

```bash
uv run --package posttrain-lab posttrain-lab foundation-qwen-smoke --tracked
uv run pytest apps/lab/tests -q
```

`posttrain-lab` remains a scenario helper. Standard seats map through
`posttrain.jobs` and the framework Verifiers bridge. Scenario thresholds,
hardware policy, and optional GSM8K shaping stay here. Reusable capability
fixes belong in their owning package.

## Gemma 4 SkyRL-BIRD SQL experiment

This Lab-only experiment combines the SkyRL-SQL text protocol, the pinned
ReViSQL/BIRD-Verified population, and TRL GRPO. It expects one RTX PRO 6000
96 GB GPU and a persistent cache with roughly 21 GB available for the BIRD
SQLite archive. It does not require a SQL server or database credentials.

Set the pod environment without writing secrets into YAML:

```bash
export HF_TOKEN=...
export POSTTRAIN_SKYRL_BIRD_CACHE=/workspace/cache/skyrl-bird-sql
export POSTTRAIN_SCRATCH_ROOT=/workspace/posttrain-scratch
export TRACKIO_DIR=/workspace/trackio
```

Prepare the immutable ReViSQL files and BIRD databases once. Interrupted
preparation is safe to retry, and a valid cache is not downloaded again:

```bash
uv run --locked --project apps/lab/environments/skyrl_bird_sql_v1 \
  python -m skyrl_bird_sql_v1.assets prepare
uv run --locked --project apps/lab/environments/skyrl_bird_sql_v1 \
  python -m skyrl_bird_sql_v1.assets validate
```

Before allocating the training run, execute the two release gates. The first
checks all 2,462 pinned gold queries against the prepared databases; the second
loads the pinned Gemma weights, applies only language-model LoRA modules, runs
one update, and reloads the adapter:

```bash
POSTTRAIN_SKYRL_BIRD_INTEGRATION=1 \
  uv run --locked --project apps/lab/environments/skyrl_bird_sql_v1 \
  pytest apps/lab/environments/skyrl_bird_sql_v1/tests/test_integration.py -m network

POSTTRAIN_GEMMA4_GPU_TEST=1 \
  uv run --locked --package posttrain-lab --extra gpu-posttrain \
  pytest apps/lab/tests/test_skyrl_bird_sql_gpu.py -m gpu
```

Run the base evaluation, then the full-shape one-step canary. The canary uses
the same four-prompt, sixteen-generation, 64-trajectory geometry as the full
profile; there is no smaller pilot.

```bash
uv run --locked --package posttrain-lab --extra gpu-posttrain posttrain-lab \
  gemma4-skyrl-bird-base-eval \
  --tracked --tracking-backend trackio \
  --project skyrl-bird-sql --project-root apps/lab \
  --scratch-root "$POSTTRAIN_SCRATCH_ROOT"

uv run --locked --package posttrain-lab --extra gpu-posttrain posttrain-lab \
  gemma4-skyrl-bird-grpo-canary \
  --tracked --tracking-backend trackio \
  --project skyrl-bird-sql --project-root apps/lab \
  --scratch-root "$POSTTRAIN_SCRATCH_ROOT"
```

Only after the canary has finite loss/gradients, reward variance, no OOM or
truncation, and a reloadable adapter, launch the 516-step full run:

```bash
uv run --locked --package posttrain-lab --extra gpu-posttrain posttrain-lab \
  gemma4-skyrl-bird-grpo-full \
  --tracked --tracking-backend trackio \
  --project skyrl-bird-sql --project-root apps/lab \
  --scratch-root "$POSTTRAIN_SCRATCH_ROOT"
```

Evaluate a produced descendant by passing its explicit immutable Trackio
version. Mutable aliases such as `latest` are rejected:

```bash
uv run --locked --package posttrain-lab --extra gpu-posttrain posttrain-lab \
  gemma4-skyrl-bird-adapter-eval --adapter-version v0 \
  --tracked --tracking-backend trackio \
  --project skyrl-bird-sql --project-root apps/lab \
  --scratch-root "$POSTTRAIN_SCRATCH_ROOT"
```
