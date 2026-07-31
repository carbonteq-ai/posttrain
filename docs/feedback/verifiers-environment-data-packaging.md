# Verifiers environment data packaging

Status: open framework feedback
Observed: 2026-07-31 in Ambient Agent production DAPO packaging

## Problem

An immutable actual-job image successfully installed and activated the
`episode-qa-v1` Verifiers wheel, but failed before the first rollout because
the activation contained:

```yaml
taskset:
  data_path: data/kg_extract_sft/train_env.jsonl
```

That path existed in the developer checkout but not in the final image.
Detached planning preserved the activation as serializable configuration,
while dataset packing only collected `DatasetLoadPlan` seats. The image smoke
test verified imports and the package manifest but did not call
`Taskset.load()`. The submitted run therefore failed with `FileNotFoundError`
before producing traces.

This violates the existing environment-only GRPO boundary: the selected
`EnvironmentBinding` owns task selection, and `job pack` is expected to
qualify every activation in the actual-job image. It does not require a new
public dataset seat for `train.grpo`.

## Desired developer experience

The normal environment package owns its task data:

```yaml
activation:
  taskset:
    id: episode-qa-v1
    mode: extract
    split: train
```

The taskset loads package resources and does not depend on the process working
directory.

The framework must also support large or private environment-owned resources.
They remain part of the `EnvironmentBinding`, not a second GRPO dataset seat:

```yaml
activation:
  taskset:
    id: episode-qa-v1
    mode: extract
    split: train
  resources:
    task_data:
      source:
        kind: jsonl
        path: data/kg_extract_sft/train_env.jsonl
      sha256: <digest>
```

YAML stays fully serializable. At runtime Verifiers may still receive a string
path, but that path is produced by resource resolution rather than copied from
an unchecked project-relative string.

## Framework work

1. Extend the environment activation contract with named, immutable resource
   declarations.
2. Include those resources in the job-pack identity and stage them below a
   deterministic environment-resource directory.
3. Resolve declared resource names to staged paths before constructing the
   native Verifiers configuration.
4. Reject unresolved relative file paths in portable execution. Keep explicit
   local paths available for in-process experimentation.
5. In the actual-job image, install the wheel and call `Taskset.load()` for the
   selected activation before provider submission.
6. Report activation and resource failures as packaging or preflight errors,
   never as a successfully admitted training run.

## Compatibility

- Package-owned data needs no external resource declaration.
- Existing explicit `data_path` continues to work for local execution.
- Portable execution may temporarily accept `data_path` only when it resolves
  inside an installed environment package or a declared staged resource.
- Environment-only GRPO keeps its current public seats.

## Acceptance

- A package-owned split loads with the project checkout absent and an unrelated
  working directory.
- A declared external JSONL is digest-locked, staged, and visible to the
  taskset in the final image.
- A missing or digest-mismatched resource fails before image publication or
  provider submission.
- An undeclared project-relative `data_path` fails detached planning with an
  actionable error.
- Image qualification exercises `Taskset.load()`, not only package import.
- Tests cover local package data, external data, missing data, and immutable
  job-package replay.

## Project mitigation

Ambient Agent now packages its small extract and retrieve splits inside
`episode-qa-v1`, selects them by `mode` and `split`, and retains `data_path` as
the explicit local override.
