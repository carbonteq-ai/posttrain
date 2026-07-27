# Move the framework runtime to Python 3.13

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

This document must be maintained in accordance with `docs/templates/PLAN.md`.

## Purpose / Big Picture

Every distribution in this workspace declares `requires-python = ">=3.12,<3.13"`,
and the universal base image is built on `python:3.12.11-slim`. veRL requires
Python 3.13. Those two facts are why the veRL job-kind image is not an ordinary
image at all: it carries a second, independently qualified 3.13 interpreter, a
projection of `posttrain.common`, `posttrain.data`, and `posttrain.train` into
that interpreter's import path, a `POSTTRAIN_VERL_PYTHONPATH` bridge variable,
and a block of actual-job Dockerfile logic that exists solely to reconcile the
two.

After this work the control plane and every backend run on one interpreter.
`online-rl-verl` becomes an ordinary single-interpreter job-kind image, the
projection bridge is deleted rather than maintained, and the release-blocked
status of that variant loses its structural cause.

The reason to believe this is cheap was established by experiment before any
code was written, and is recorded under Surprises below: the entire dependency
closure already resolves on 3.13 at identical versions, ships cp313 wheels, and
the framework's own test suite already passes on 3.13.

You can see this working when `posttrain doctor` reports Python 3.13, the veRL
capsule no longer projects framework packages across interpreters, and a GPU
qualification run reproduces its 3.12 result.

## Progress

- [x] Milestone 1 — Raise `requires-python` across the workspace, relock, and
  make the framework's own version assertions agree. Landed: all 23
  distributions moved to `>=3.13,<3.14`, ruff targets `py313`, pyright targets
  3.13, and `doctor` asserts 3.13. The workspace relocked and synced at
  3.13.12. The full ladder passes with 744 tests, the same count as on 3.12,
  and no assertion was relaxed to get there. `posttrain doctor` reports Python
  3.13 with every check green.
- [ ] Milestone 2 — Move the universal base image to a 3.13 interpreter,
  including the hardcoded cp312 Triton wheel, and republish the base.
- [ ] Milestone 3 — Rebuild and requalify the job-kind images on the new base,
  regenerating `published.toml` from real registry state.
- [ ] Milestone 4 — Collapse the veRL two-interpreter capsule into a single
  interpreter and delete the projection bridge.
- [ ] Milestone 5 — Run a GPU qualification on 3.13 and compare it against the
  retained 3.12 evidence before treating the move as proven.

## Surprises & Discoveries

- Observation: the dependency closure needs no version movement whatsoever to
  run on 3.13.
  Evidence: compiling `profiles/common.txt` + `supervised.txt` +
  `online-rl-trl-py312.txt` with `uv pip compile --python-version 3.13`
  resolved 866 packages and produced identical pins to the 3.12 profile —
  `vllm==0.25.1`, `transformers==5.14.1`, `torch==2.11.0`, `accelerate==1.14.0`,
  `peft==0.19.1`, and the same TRL and Verifiers Git commits.

- Observation: every package in that closure publishes a cp313 wheel, so the
  move requires no source builds.
  Evidence: `uv pip install --dry-run --only-binary :all:` against a 3.13
  virtual environment succeeded for the whole resolved set, including vLLM,
  flashinfer, bitsandbytes, liger-kernel, and xgrammar. Separately,
  `torch==2.11.0+cu130` from the PyTorch CUDA 13.0 index also resolves to a
  cp313 wheel.

- Observation: the framework's own code already runs on 3.13, so the risk is
  concentrated in GPU runtime behaviour rather than in the codebase.
  Evidence: in a scratch worktree with `requires-python` relaxed and the
  workspace locked and synced at 3.13.12, `uv run pytest` reported 742 passed,
  17 skipped, 2 failed. Both failures are artifacts of the experiment rather
  than incompatibilities: `commands/doctor.py:41` asserts
  `sys.version_info[:2] == (3, 12)`, and `apps/lab/tests/test_catalog.py`
  compares against a `dependency_lock_sha256` that changed because the lock was
  regenerated.

- Observation: three framework packages already execute on 3.13 in production
  today, which is why the codebase result above is unsurprising in hindsight.
  Evidence: `verl-py313/profile.toml` sets `backend_python = "3.13.12"` and
  `worker_projection_packages = ("common", "data", "train")`, so those packages
  are already imported by a 3.13 interpreter inside the veRL capsule.

- Observation: the interpreter suffix in job-kind variant names loses its
  meaning after this change.
  Evidence: the naming scheme `{kind}-{backend}-{py}` distinguishes
  `online-rl-trl-py312` from `online-rl-verl-py313`. Once both are 3.13, the
  suffix is constant across every variant and no longer separates anything.
  The decision to keep the interpreter in the name was taken before this
  experiment and should be revisited once the bump lands.

## Decision Log

- Decision: bump the whole workspace rather than give TRL its own 3.13 backend
  environment.
  Rationale: the two-interpreter pattern exists to bridge a version gap. Adding
  a second instance of it would double the bridge while leaving the gap in
  place. Raising the floor removes the gap and lets the existing bridge be
  deleted.
  Consequences: the base image, every job-kind image, and every actual-job
  package key change once. Nothing is yet published to the public registry, so
  the blast radius is the local registry and retained packages.

- Decision: no product baseline amendment.
  Rationale: `docs/post-training/` pins no Python version and does not describe
  the veRL two-interpreter mechanism; it names veRL only as a backend whose job
  meaning must be preserved. This change alters implementation, not product
  meaning.

- Decision: treat the test suite passing as necessary but not sufficient.
  Rationale: the suite does not exercise vLLM, Torch, or fused kernels on a
  GPU. Milestone 5 exists because a green suite on 3.13 says nothing about
  whether a GRPO rollout converges on 3.13.

## Outcomes & Retrospective

To be completed as milestones land.

## Context and Orientation

The universal base image installs a Python 3.12 interpreter and a
`triton-3.6.0-cp312` wheel by direct URL. Job-kind images layer runtime
profiles onto it. The actual-job image installs framework and project source
into the base interpreter's virtual environment, and — for veRL only — projects
a subset of framework packages into a second interpreter.

`requires-python` appears in 23 distributions: the workspace root, every
package under `packages/`, and every application under `apps/`.

The framework asserts its own interpreter version in
`apps/cli/src/posttrain_cli/commands/doctor.py`, and pyright is configured with
`pythonVersion = "3.12"` in the workspace root.

Changing the interpreter changes the resolved dependency closure, which changes
`runtime_dependencies_digest`, which changes every job package key. This is
correct behaviour rather than a problem: a job built against a different
interpreter is a different job.

## Plan of Work

Milestone 1 raises the floor. Every `requires-python` moves to `>=3.13,<3.14`,
the workspace relocks, `doctor` asserts 3.13, and pyright targets 3.13. The
catalog entries that record the workspace lock digest are regenerated. This
milestone is complete when the full ladder passes on 3.13 with no relaxed
assertions.

Milestone 2 moves the base image. `PYTHON_IMAGE` moves to a digest-pinned
3.13 slim image and the hardcoded cp312 Triton wheel URL moves to its cp313
equivalent. The base image is rebuilt and its digest captured.

Milestone 3 rebuilds each job-kind image on the new base and regenerates
`published.toml` from what the registry reports, using the release tooling
rather than by hand.

Milestone 4 collapses the veRL capsule. With one interpreter, the projection
path, the `POSTTRAIN_VERL_PYTHONPATH` variable, the worker projection package
list, and the actual-job Dockerfile bridge are removed, and the release gate is
updated to assert their absence rather than their correctness.

Milestone 5 qualifies on hardware. A GRPO or distillation qualification runs on
the 3.13 images and its evidence is compared against the retained 3.12 run.
Until this passes, 3.13 images must not be treated as qualified.

## Validation and Acceptance

The repository ladder passes on 3.13: `uv sync --all-packages --locked`,
`ruff check`, `pyright`, `lint-imports`, `pytest`, `git diff --check`.

`posttrain doctor` reports Python 3.13 and all checks pass.

`containers/posttrain-job-kinds/validate.py` and the actual-job `validate.py`
pass unchanged in substance.

The veRL release gate passes with the projection assertions replaced by
assertions that no cross-interpreter projection exists.

A GPU qualification run on 3.13 produces evidence comparable to the retained
3.12 run for the same work package.

## Idempotence and Recovery

Milestone 1 is a pure source change and is revertible by branch. The lock can
be regenerated at any time.

Image work is content-addressed and safe to repeat; a rebuilt image that does
not match is reported rather than silently accepted, and the 3.12 images remain
published and pinned by the previous release manifest until `published.toml` is
regenerated.

The whole change lives on `codex/python-313-runtime` and can be abandoned by
deleting the branch. Nothing published to the public registry is overwritten,
because that registry has no published release yet.

## Interfaces and Dependencies

No public API changes. The `runtime_variant` naming question is explicitly out
of scope here and is deferred until after the bump, when the interpreter
suffix's meaning can be reassessed.
