# AutomationBench compatibility fork

The post-training framework uses the CarbonTeq AutomationBench fork at
`https://github.com/carbonteq-ai/AutomationBench` so Zapier AutomationBench
1.0.5 can share the platform's qualified Python 3.12 trainer runtime.

## Distribution transition

The fork now builds as
`carbonteq-automation-bench==1.0.5.post1`, preserving the
`automationbench` import package and `auto-bench` command. Its wheel and source
distribution build successfully, clean Python 3.12 installation succeeds, the
75-test compatibility suite passes, and a `carbonteq-v*` tag workflow is ready
for PyPI Trusted Publishing.

The distribution is published on the internal stable index from merge revision
`908db2abd4a868acc37ab0850474bff653bea25c`. The adapter depends on the exact
registry version instead of repeating a transitive Git URL, so environment
packing can resolve and hash the complete portable dependency closure.

## Selected revision

The selected immutable revision is
`d54dbebabdba6c6eda201694aee8ddcf36ccfc51`, based on upstream Zapier commit
`a321764ace3cfbe42289e6a13abef2f0f4f56fad`. The maintained delta lowers the
declared Python floor to 3.12, regenerates the fork lockfile, and documents the
fork. It does not change benchmark tasks, tools, simulated application state,
routes, runner behavior, or scoring.

The executable fork pin lives in the external environment repository's
`environments/automationbench_v1/pyproject.toml` and `uv.lock`. Framework
catalog and evaluation source bindings pin the full
`carbonteq-ai/verifiers-environments` commit and the root `uv.lock` carries the
resolved wheel closure, so run lineage describes the code that actually loaded
the task population.

## Supported integration boundary

The Verifiers v1 adapter is published as the standalone `automationbench-v1`
package in `https://github.com/carbonteq-ai/verifiers-environments` at the
framework-pinned commit `017ac72f543f79f48400cbb4cb641d6df4c3adfa`, under
`environments/automationbench_v1`. There is no framework-local implementation;
the external repository owns its package lifecycle. GRPO and
evaluation select domain categories, deterministic sampling seeds, task and
rollout budgets, toolset, and interaction limits. Concrete task identities are
resolved by the environment and retained in native traces; they are not public
work-package inputs.

Install the integrated runtime with:

    uv sync --all-packages --all-extras --locked --python 3.12

## Validation

The fork's Python 3.12 domain, runner, rubric, and Zapier meta-tool suite passes
75 tests. The published adapter passes all ten package tests from its Python
3.12 lock.
The platform additionally constructs a real two-task deterministic bridge in
the Python 3.12 host and runs the complete workspace validation ladder.

From the external repository checkout, run:

    cd /home/hammad/projects/verifiers-environments/environments/automationbench_v1
    uv sync --locked --python 3.12
    uv run pytest -q

From `/home/hammad/projects/rl`, run the framework consumers:

    uv run pytest apps/lab/tests packages/eval/tests -q
    uv run lint-imports
    uv lock --check

Upstream's full test tree has pre-existing generated-tool import failures and
two Gmail route expectation failures. The two Gmail failures reproduce on both
Python 3.12 and Python 3.13; they are not regressions introduced by the fork.

## Update and retirement

Make compatibility changes in `/home/hammad/projects/automationbench`, update
its `CARBONTEQ_FORK.md`, commit and push the fork first, then update the
external `verifiers-environments` package and its immutable framework pin.
Rerun the fork, external package, and platform suites before accepting a new
revision. Retire the fork when upstream supports Python 3.12 and passes the
same relevant suites; move the pins to the immutable upstream commit rather
than a branch name.
