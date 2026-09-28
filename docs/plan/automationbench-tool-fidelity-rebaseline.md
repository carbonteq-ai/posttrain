# Fix AutomationBench tools that mislead the model, then re-baseline held-out evaluations

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

The LFM2.5-2.6B training runs on AutomationBench were partly training the
model against broken tools. AutomationBench is a benchmark of workflow tasks in
which the model calls simulated SaaS tools (Google Sheets, Gmail, Salesforce,
Slack, Drive, Jira and others) that act on a simulated world. Several of those
tools do not do what their names and parameters promise: a Sheets row search
ignores the search and always returns the first ten rows, Drive search mixes
nine placeholder files into every answer, Salesforce search never matches
without a field name, and our own adapter release 0.4.3 turned list arguments
into corrupted strings. The model was rewarded or punished for outcomes the
tools, not the model, decided, and some episodes ended early because repeated
large results filled the 24,576-token context.

After this work the tools behave as documented, the adapter passes list
arguments correctly, and held-out evaluations of the base model and the
checkpoints we care about are re-run on the fixed environment so later training
runs are compared against numbers that mean what they say. A reader can see it
working by running the fork's and the adapter's regression tests, and by
opening the new evaluation runs in Observatory and comparing their scores and
tool-mistake counts.

This work does not change the frozen post-training product baseline in
`docs/post-training/`; it changes one benchmark fork, one environment adapter,
and lab catalog entries.

## Progress

- [x] (2026-09-28) Audits: 12 tool bugs reproduced on task data
  (`scratchpad/tool_audit/`), 210 training traces classified. Run notes on
  `lfm26-sampo-cont120-g24x6-lr5e5-kl1e2-t05-20260928-r1`.
- [x] (2026-09-28) Adapter list-argument fix, commit `fcd06f9` on
  `codex/automationbench-tool-fidelity` of `verifiers-environments`, with
  `tests/test_limited_tool_arguments.py`.
- [ ] AutomationBench fork: tool fixes on `codex/tool-fidelity-fixes` with
  `tests/test_tool_fidelity.py`, ledger updated.
- [ ] Push the fork branch; re-vendor it into the adapter
  (`environments/automationbench_v1/src/automationbench/`), bump the adapter
  to 0.5.0, lock, push.
- [ ] Posttrain lab catalog: held-out suites `automationbench-lfm26-heldout-mix-v4`
  and `...-v4-t05`, identical to the v3 suites (same sampling, budgets and
  tasks) except the fixed adapter, with their evaluation plans, work packages
  and gates.
- [ ] Re-baseline: base, SAMPO update 120, continuation updates 20 and 40 on
  both suites; record score, tool mistakes per episode and truncation.

## Surprises & Discoveries

- The fork's ledger said it "does not change ... tools"; every tool bug found
  is identical in upstream Zapier AutomationBench, so all earlier runs and
  evaluations in this project carried them.
- Our 0.4.3 adapter fix for FastMCP decoding (`fields_json`) re-encoded every
  decoded list as JSON, so comma-separated parameters received `'["a","b"]'`
  and the tool reported success. Before 0.4.3 such calls failed validation.
- Keeping earlier turns' reasoning in the prompt is Liquid's design for this
  model: the LFM2.5-2.6B chat template keeps `<think>` for every assistant
  message after the last user message (older 1.2B templates kept only the
  last one). Our renderer matches it, so it is not changed here.
- The temperature-0.1 held-out suite already samples with `top_k 50` and
  `repetition_penalty 1.1` (Liquid's `generation_config.json`); training and
  the temperature-0.5 suite used `top_k 0`, `repetition_penalty 1.0`.

## Decision Log

- Decision: fix tools in the CarbonTeq AutomationBench fork and re-vendor, not
  patch the vendored copy. Rationale: `AGENTS.md` puts generally reusable
  benchmark fixes in the fork; the adapter only vendors it. Date: 2026-09-28.
- Decision: do not change task data or graders; keep tool names and
  parameters. Rationale: scores must stay about the same tasks, and the model's
  tool schemas must stay compatible with its training. Date: 2026-09-28.
- Decision (user): re-baseline held-out evaluations on the fixed environment;
  new suites get new ids so the old numbers stay valid for old comparisons.
  Date: 2026-09-28.
- Decision (user): the re-baseline changes only the tools. Each v4 suite
  keeps its v3 sampling; the two suites differ only in temperature (0.1 and
  0.5). Liquid's `top_k 50` / `repetition_penalty 1.1` are not added to the
  temperature-0.5 suite or to training. Date: 2026-09-28.
- Decision: evaluation score stays partial credit, the benchmark's own
  measure; tool mistakes per episode, truncation and the training-penalised
  score are reported beside it (user noted the old evaluations did not
  reflect the mistake penalty). Date: 2026-09-28.

## Outcomes & Retrospective

Not yet complete.

## Context and Orientation

Three repositories are involved, all checked out under `/home/hammad/projects`.

`automationbench` is the CarbonTeq fork of Zapier's AutomationBench
(`origin` carbonteq-ai/AutomationBench). Tools live in
`automationbench/tools/zapier/<app>/`; `automationbench/tools/zapier/action_utils.py`
builds responses for template-based apps. Its ledger is `CARBONTEQ_FORK.md`.

`verifiers-environments-turns-v2` is a worktree of
carbonteq-ai/verifiers-environments. The adapter package
`environments/automationbench_v1` exposes AutomationBench to the Verifiers v1
runtime. It vendors (copies) the fork's `automationbench/` package under
`src/automationbench/`; `src/automationbench_v1/limited_tools.py` exposes only
the tools a task allows, through FastMCP, which decodes JSON-looking string
arguments into objects before the tool runs.

`rl-perf-guard` is a worktree of this Posttrain repository. Held-out
evaluation suites are environment entries in
`apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml` that pin
the adapter by full commit; `apps/lab/.posttrain/work_packages/` holds the
evaluation work packages; `apps/lab/src/posttrain_lab/qualification/gates.toml`
registers them.

## Plan of Work

First the fork: a branch `codex/tool-fidelity-fixes` from
`codex/python312-compat` fixes each audited tool and adds one regression test
per fix in `tests/test_tool_fidelity.py`, checks that graders of ~30 tasks
using the changed tools remain satisfiable, and records the delta in
`CARBONTEQ_FORK.md`. Then the adapter: copy the fork's `automationbench/`
tree over `src/automationbench/`, update the pinned fork commit in its README,
bump to 0.5.0, relock, run its tests, commit and push. Then Posttrain: add the
two v4 held-out suites pinned to the new adapter commit, their evaluation
plans, work packages and gates; validate; commit. Finally run eight
evaluations one at a time on the workstation GPU and record them in a run
note and in this plan.

## Concrete Steps

Fork (working directory `/home/hammad/projects/automationbench`):

    uv sync --locked --python 3.12 --all-groups
    uv run --python 3.12 pytest -q tests/test_domains.py tests/test_meta_tools.py tests/test_runner.py tests/test_rubric.py tests/test_tool_fidelity.py
    uv run --python 3.12 ruff check .

Adapter (working directory
`/home/hammad/projects/verifiers-environments-turns-v2/environments/automationbench_v1`):

    uv run pytest -q
    uv lock --system-certs

Posttrain (working directory `/home/hammad/projects/rl-perf-guard`):

    cd apps/lab && uv run posttrain catalog validate
    uv run posttrain job plan .posttrain/work_packages/<new work package>.yaml
    cd ../.. && uv run pytest apps/lab/tests/test_catalog.py apps/lab/tests/test_qualification_gates.py -q

## Validation and Acceptance

Each fixed tool has a test that fails on the old code and passes on the new.
On the Gorgias refund task, `google_sheets_find_many_rows` for order 4512
returns that order's row; Drive search returns no "Hello World Document";
`gorgias_update_ticket(tags=["coaching","agent_b"])` saves two tags. The eight
evaluations succeed, and each reports score, tool mistakes per episode and
truncation.

## Idempotence and Recovery

All catalog additions use new ids; old suites and their results are untouched.
A failed evaluation is re-run under a new run id suffix (`-r2`).

## Artifacts and Notes

Audit scripts and traces: session scratchpad `tool_audit/` and `trace_audit/`.
Penalised re-scoring of old evaluations: `eval_penalty/score.py`.

## Interfaces and Dependencies

The adapter pins the fork by vendoring; Posttrain pins the adapter by full
commit in the catalog. No Python dependency pins in Posttrain change.
