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

The monorepo root is itself a qualification project under `.posttrain/`. Its
`project.toml` points at `posttrain_lab.entry:configure`, which builds the
standard `JobRuntime` and attaches git source metadata. Validate YAML work
packages with the primary CLI and **no** `--host`:

```bash
uv run --package posttrain posttrain work-package validate \
  .posttrain/work_packages/foundation_screen.yaml

uv run --package posttrain posttrain work-package validate \
  .posttrain/work_packages/automationbench_zapier_grpo.yaml
```

Remote GPU release evidence still uses `examples/gpu-qualification` and the
primary `posttrain work-package run` path on the remote host.

## Qualification gates

Lab owns the reviewed inventory of framework qualification gates in
`src/posttrain_lab/qualification/gates.toml`. During the additive migration,
the manifest classifies the work packages currently owned by the root
qualification project; it does not submit work or duplicate provider logic.

```bash
uv run --package posttrain-lab posttrain-lab qualification list --project-root .
uv run --package posttrain-lab posttrain-lab qualification list --project-root . --json
```

Every work-package YAML must occur exactly once in the registry. A gate records
its lifecycle, tier, selected job, expected job kind, and evidence acceptance
adapter. The two active release gates are the SFT data-preparation and managed
GSM8K evaluation qualifications; extended and experimental entries remain
visible without silently becoming release requirements.

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
