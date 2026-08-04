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
.venv/bin/posttrain dataset materialize datasets/posttrain-sft-smoke@1
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
