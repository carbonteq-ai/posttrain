# AutomationBench compatibility fork

The post-training framework uses the CarbonTeq AutomationBench fork at
`https://github.com/carbonteq-ai/AutomationBench` so Zapier AutomationBench
1.0.5 can share the platform's qualified Python 3.12 trainer runtime.

## Distribution transition

The fork builds as `carbonteq-automation-bench==1.0.5.post1`, preserving the
`automationbench` import package and `auto-bench` command. The manually
published
[`carbonteq-v1.0.5.post1`](https://github.com/carbonteq-ai/AutomationBench/releases/tag/carbonteq-v1.0.5.post1)
release is bound to commit `908db2abd4a868acc37ab0850474bff653bea25c` and
retains the wheel SHA-256
`bd80b4947fbdd60706d9545e79635b79931d89dfc294ed45b01df6886c1f1509` and source
distribution SHA-256
`04ccef85e2a83bd26777a10a08702b4fb6a47169352777ab8564fa1bbba9acf6`.
No fork release runner is retained or used.

The distribution is published on the internal stable index from merge revision
`908db2abd4a868acc37ab0850474bff653bea25c`. The adapter depends on the exact
registry version instead of repeating a transitive Git URL, so environment
packing can resolve and hash the complete portable dependency closure.

## Selected revision

The selected immutable revision is
`908db2abd4a868acc37ab0850474bff653bea25c`, based on upstream Zapier commit
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
framework-pinned commit `b7bcb591facfcd2b073802f6d7496b24ab9c479e`, under
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
