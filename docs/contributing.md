# Contributing to the framework

This is for working **on** the framework. Using it as a library is
[`consumer-setup.md`](./consumer-setup.md); cutting a release is
[`publishing.md`](./publishing.md). [`AGENTS.md`](../AGENTS.md) at the
repository root carries the conventions in more detail and is worth reading
before a first change.

## Setting up

The repository is a `uv` workspace requiring Python 3.13.

```bash
uv sync --all-packages
```

`--all-packages` is not optional. A plain `uv sync` installs the root project's
dependencies but leaves most workspace members absent, and the tree then fails
in ways that look unrelated to whatever you were doing — imports of
`posttrain.execution` disappearing, for instance.

## The validation ladder

Run all of it before proposing a change. Nothing here needs network, Docker, or
a GPU.

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run lint-imports && uv run pytest -q
```

`lint-imports` enforces the dependency contracts between packages. It is the
one people are surprised by: the layering is a contract, not a convention, and
a new import can break it without failing any test.

During development run the smallest relevant package's tests first — the full
suite is fast, but a focused run is faster.

## How the workspace fits together

`packages/` holds the libraries and `apps/` the executables. Two boundaries
matter more than the rest:

- **`apps/cli`** is the consumer surface. Framework-owner operations do not
  belong there; `apps/release` exists for those, and a test asserts `posttrain`
  does not depend on it so release authority cannot reach a consumer's
  environment.
- **`apps/runtime`** runs *inside* a job image and nowhere else. A consumer
  never installs it, which is why it is requested explicitly when a job image is
  packed rather than read from the packing environment.

## Things that will surprise you

**The catalog records the lock's hash.** `packages/catalog/src/posttrain/
catalog/base/training.yaml` stores `dependency_lock_sha256`, the SHA-256 of
`uv.lock`. Any change that re-locks — a bump, a new dependency — drifts it and
fails one test in `apps/lab` until you realign it. The check is deliberate: it
ties catalog bindings to a specific dependency closure.

**Some tests read machine-wide state.** Trust resolution consults
`/etc/posttrain/trust/internal-ca.pem` and admission consults a machine-scoped
ledger, so a test that resolves a provider can pass on a laptop and fail on a
worker. `apps/cli/tests/conftest.py` isolates both with autouse fixtures. If
you add a test that resolves providers, let those fixtures do their work rather
than pointing at real paths.

**Changing a third-party pin is a release-sized change.** A fork commit or
dependency bound changes the job-kind constraint lock, whose hash is a label on
every published runtime image. The manifest loader then refuses to load and
roughly fifty tests fail at once. That is the guard working; see
[`publishing.md`](./publishing.md) for what it costs to resolve.

**Do not rewrite plan, decision, or qualification records** to match current
behaviour. They describe what was true when written. Correct them by adding a
dated observation, not by editing history — the plan documents under
`docs/plan/` are living in their `Progress` and `Decision Log` sections and
immutable in their evidence.

## Writing tests

A test should say what breaks and why it matters, not restate the assertion.
The suite is full of examples: the reason a check exists is usually a defect
that reached production once, and recording that reason is what stops the check
being deleted as redundant later.

Mark tests needing network, Docker, or a GPU with the existing markers. A test
requiring credentials must skip with a clear reason when they are absent — but
its release-gate command still has to be run before the integration is called
complete.
