# CLI framework and package split

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

Maintain this document in accordance with `docs/templates/PLAN.md`.

## Purpose / Big Picture

`posttrain` grows as a multi-verb developer CLI. Putting every parser, dispatcher, scaffold template, and command body in one `cli.py` made aliases and helpers expensive and hard to review. After this change, the CLI uses Typer for typed command wiring, splits command groups into modules, and keeps domain logic in existing packages. Developers still run the same `posttrain …` argv surface; tests still call `main([...])`.

## Progress

- [x] (2026-07-24T00:15Z) Recorded Typer choice and target package layout.
- [x] (2026-07-24T00:40Z) Added `typer` to `apps/cli` and refreshed the lock.
- [x] (2026-07-24T00:55Z) Extracted context/output/errors/project/work_runtime/materialize/scaffolding and Typer command modules; `cli.py` is a thin `main()`.
- [x] (2026-07-24T01:05Z) Converted Typer parameters to `Annotated` so ruff B008 is clean.
- [x] (2026-07-24T01:10Z) Amended `05-apis.md` exit-code wording; CLI tests 22 passed; ruff/pyright clean for the CLI package.
- [x] (2026-07-24T01:15Z) Commit with prior DX leftover aliases on the ship branch.

## Surprises & Discoveries

- Observation: Typer is already resolved in `uv.lock` as a transitive dependency; adding it as a direct CLI dependency should not introduce a novel package family.
  Evidence: `uv.lock` contains `name = "typer"`.

## Decision Log

- Decision: Use Typer (not Cyclopts or raw Click) for this migration.
  Rationale: Strong DX, nested apps, shell completion, and already present in the workspace lock; Cyclopts remains a future option if Union-heavy command signatures become painful.
  Date/Author: 2026-07-24 / agent
- Decision: Keep `main(Sequence[str] | None) -> int` as the public test and console entry.
  Rationale: Existing CLI tests and scripts invoke `main([...])` and assert exit codes 0/1/2.
  Date/Author: 2026-07-24 / agent
- Decision: `job` / `run` remain thin aliases that call the same functions as `work-package` / Observatory helpers.
  Rationale: Product vocabulary stays canonical; aliases must not duplicate behavior.
  Date/Author: 2026-07-24 / agent

## Outcomes & Retrospective

The CLI is no longer a single argparse god module. Command nouns live under
`commands/`, scaffolding under `scaffolding/`, and shared emit/context helpers
are reusable. Typer owns wiring; product packages still own domain behavior.
Remaining follow-ups: optional shell-completion docs and any further thinning of
`install_starter` test monkeypatch indirection through `posttrain_cli.cli`.

## Context and Orientation

Today almost all CLI behavior lives in `apps/cli/src/posttrain_cli/cli.py` (~1400 lines) with one helper module `overlay_write.py`. The distribution entrypoint is `posttrain = posttrain_cli.cli:main` in `apps/cli/pyproject.toml`. Canonical CLI contract text is in `docs/post-training/05-apis.md` and `docs/developer-experience.md`.

Terms:

- **Command module**: a Typer sub-app for one noun (`catalog`, `dataset`, …).
- **CliState**: runtime object holding `--project-root`, `--json`, and lazy project/catalog accessors.
- **Scaffolding**: init templates and `uv sync` for starter projects, not runtime command orchestration.

## Plan of Work

Create the package layout under `apps/cli/src/posttrain_cli/`:

- `constants.py` — distribution name and catalog family literals
- `context.py` — `CliState`
- `output.py` — JSON serialization and human/JSON emit
- `errors.py` — safe error string helpers
- `project.py` — discover layout and open catalog
- `work_runtime.py` — work-package path load, host/entry runtime construction
- `materialize.py` — dataset/environment seat materialize/preflight
- `scaffolding/init.py` — project initialize + starter install
- `commands/*.py` — Typer command groups
- `app.py` — compose the root Typer application
- `cli.py` — `main()` only: invoke app, map exceptions to exit codes

Add `typer` to `apps/cli/pyproject.toml` dependencies. Update `05-apis.md` to say usage errors exit 2 without naming argparse. Preserve argv behavior for all existing `apps/cli/tests/test_cli.py` cases, including job/run aliases.

## Concrete Steps

From repository root:

1. Implement modules and Typer app.
2. `uv lock` / `uv sync --all-packages --locked --python 3.12` as needed after adding typer.
3. `uv run pytest apps/cli/tests/test_cli.py -q`
4. `uv run ruff check apps/cli`
5. `uv run pyright apps/cli/src/posttrain_cli` (or repo pyright if configured for the package)

## Validation and Acceptance

- `uv run pytest apps/cli/tests/test_cli.py` passes.
- `posttrain --help` and `posttrain catalog --help` show Typer/Click help.
- Invalid usage such as missing required options still exits 2.
- Contract failures still exit 1 with `error: …` on stderr.
- No business logic remains in a monolithic argparse dispatch ladder.

## Idempotence and Recovery

Re-running the migration is additive file writes followed by deleting obsolete functions from `cli.py`. If tests fail, keep `main(argv)` semantics and fix Click/Typer option names (`--json`, `--project-root`) before changing product behavior.

## Interfaces and Dependencies

- Dependency: `typer` (brings `click`).
- `posttrain_cli.cli.main(argv: Sequence[str] | None = None) -> int`
- `posttrain_cli.app.create_app() -> typer.Typer`
- `posttrain_cli.context.CliState` with `project_root`, `json_output`, `layout()`, `catalog()`
- Command handlers raise `ContractError` / `FileExistsError` / etc.; `main` maps them to exit 1.

## Artifacts and Notes

Prior DX leftovers (`dataset add`, `environment add`, `catalog materialize`, `doctor --fix`, `job plan|run`, `run show`) ship in the same commit as this structural fix so the branch does not land a second god-module bump.
