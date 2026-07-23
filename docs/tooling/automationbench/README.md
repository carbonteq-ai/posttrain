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

The distribution is not yet published. Keep the adapter's immutable Git
dependency until the fork changes are committed, pushed, and accepted by PyPI.
The packaging candidate is committed and pushed at
`6e3c50209731c0b06c3bc6d3dbb30bc7fdf10a38`; it still requires review, merge,
and registry publication through
[`carbonteq-ai/AutomationBench#1`](https://github.com/carbonteq-ai/AutomationBench/pull/1).
The PR merged as `908db2abd4a868acc37ab0850474bff653bea25c`, and the repository
now has the `pypi` GitHub Actions environment used by the release workflow.
After the PyPI pending publisher is configured and the release tag publishes
successfully, replace the adapter's direct reference with the exact registry
version, regenerate the environment and root locks, publish
`automationbench-v1`, and only then publish the lab GPU post-training extra.

## Selected revision

The selected immutable revision is
`d54dbebabdba6c6eda201694aee8ddcf36ccfc51`, based on upstream Zapier commit
`a321764ace3cfbe42289e6a13abef2f0f4f56fad`. The maintained delta lowers the
declared Python floor to 3.12, regenerates the fork lockfile, and documents the
fork. It does not change benchmark tasks, tools, simulated application state,
routes, runner behavior, or scoring.

The executable pins live in
`environments/automationbench_v1/pyproject.toml`,
`environments/automationbench_v1/uv.lock`, and the root `uv.lock`. Catalog and
evaluation environment sources use the same revision so run lineage describes
the code that actually loaded the task population.

## Supported integration boundary

`environments/automationbench_v1` owns the Verifiers v1 adapter. GRPO and
evaluation select domain categories, deterministic sampling seeds, task and
rollout budgets, toolset, and interaction limits. Concrete task identities are
resolved by the environment and retained in native traces; they are not public
work-package inputs.

Install the integrated runtime with:

    uv sync --all-packages --all-extras --locked --python 3.12

## Validation

The fork's Python 3.12 domain, runner, rubric, and Zapier meta-tool suite passes
75 tests. The adapter passes all six package tests from its Python 3.12 lock.
The platform additionally constructs a real two-task deterministic bridge in
the Python 3.12 host and runs the complete workspace validation ladder.

From `/home/hammad/projects/rl`, run:

    uv run --project environments/automationbench_v1 --python 3.12 \
      --with pytest --with pytest-asyncio \
      pytest -q environments/automationbench_v1/tests
    uv run pytest apps/lab/tests packages/eval/tests -q
    uv run lint-imports
    uv lock --check

Upstream's full test tree has pre-existing generated-tool import failures and
two Gmail route expectation failures. The two Gmail failures reproduce on both
Python 3.12 and Python 3.13; they are not regressions introduced by the fork.

## Update and retirement

Make compatibility changes in `/home/hammad/projects/automationbench`, update
its `CARBONTEQ_FORK.md`, commit and push the fork first, then move the immutable
pin and both locks here. Rerun the fork and platform suites before accepting a
new revision. Retire the fork when upstream supports Python 3.12 and passes the
same relevant suites; move the pins to the immutable upstream commit rather
than a branch name.
