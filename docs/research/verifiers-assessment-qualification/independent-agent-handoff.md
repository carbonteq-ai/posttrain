# AutomationBench reward evidence and credit assignment: independent agent handoff

Snapshot: 2026-10-04. Read this with the living plans below; this is a handoff,
not a replacement implementation plan. Several older unchecked plan entries are
historical and have been superseded by newer checkpoints.

## What we are trying to achieve

Make AutomationBench explain what an agent accomplished and which actions
helped or caused harm, instead of exposing only a final episode score. Keep
that evidence trustworthy enough for several training algorithms to consume.

For example, a conversion task may require a policy lookup, eligible deal
selection and an upload. A final zero can conceal a useful retrieval. A final
partial score can conceal an excluded conversion. We want independently
verified outcomes, persistent guard violations and justified links to actual
actions. Later, training can use those links to focus learning on the original
generated tokens. A correct-looking trace or a final score does not prove that
every action was correct.

There are two connected pieces:

1. **Verifiers library foundation:** retain native trace identities, allow
   flexible assessment preparation, publish structured findings, assign domain
   credit to actual actions, and align recipients to original sampled tokens.
   Assessors can prepare their own context and use deterministic checks, judges,
   or both. They need not expose a rigid judge-message format.
2. **AutomationBench implementation:** declare public goals, guards, policy and
   credit rules in manifests, backed by reusable deterministic evidence
   operations. Task-specific parameters belong in data; named-task Python
   evaluators do not. Build this for AutomationBench now, rather than designing
   abstractions for unrelated environments.

The central boundary is:

**raw trace/state → authenticated evidence → assessment finding → domain action
credit → original-token alignment → training algorithm.**

Training owns returns, advantages, normalization and losses. Verifiers must not
pretend an assessment finding is already an algorithm advantage. Evidence about
a whole episode, a turn, an action and a token span is related but distinct.

## How the reference traces fit

The Luna traces are **verified reference executions**, not imitation targets.
They expose environment defects, useful and harmful actions, missing mechanisms
and realistic budgets. Public task instructions and available policy sources
define correctness; Luna's chosen path and hidden benchmark answers cannot
invent requirements. Successful alternatives must remain acceptable.

Reference collection is already finished: 800 attempts were consumed, 790
episodes retained and ten lost to disk issues. The development partition has
709 tasks, with 700 retained episodes and nine unavailable. Keep unavailable
data visible. Do not silently rerun tasks or assume success from transport.

**No new Luna collection is required for the present implementation.** Use
recorded episodes, public inputs and genuine simulator counterexamples. The
original collection used ten concurrent slots and a 16,384 observed-output-token
threshold. That threshold was not a proven physical decoding cap.

## Current progress and its limits

| Area | What exists | What remains |
| --- | --- | --- |
| Native assessment and credit API | Substantial source admission, prepared views, findings, domain credit and replay machinery in the candidate Verifiers fork | Full consumer/original-token qualification, publication and immutable adoption |
| Public task review | 105 unique development tasks, fifteen in each of seven categories, reviewed in eleven batches | Whole-task verification and eventual full-library declarations |
| Installed manifests | 23 catalog entries: thirteen within the 105 sample, ten separate Simple Opportunity pilots | These are bounded components; zero whole tasks qualified |
| Acceptance cohorts | Three cohorts cover thirty distinct tasks with public bindings and explicit gaps | Compiling an inactive draft is not activation or completion |
| Deterministic mechanics | Typed initial populations, chained exact lookups, rational decimal/date expressions, conditional effects, retained state, and bounded Gmail/Sheets/Slack/Zendesk/Jira/HubSpot evidence | Missing service/relationship families, two-sided relations, ranking, explicit read prerequisites and report fact coverage |
| New aggregates | Source-recaptured filtered COUNT/SUM, member provenance and refusal on unknown membership/eligibility | Native manifest integration and verifying the agent's actual report/action |
| Output guards | Authored-output/call coverage and explicit no-clarification revision 2 implemented | Semantic gate still fails; complete output coverage remains open |
| Training tracking | Local results-only projection excludes raw assessment bodies | Remote artifact upload/restart/end-to-end qualification |

Latest aggregate qualification: 59 owner tests, 31 independently repeated
boundary cases and a root combined 286-case gate pass. Scoped Ruff, formatting
and Pyright are clean. The actual prepaid public inputs, including the controller
Slack correction, compute 4,000 + 300 + 600 = 4,900. Reordering, changed amounts,
missing eligibility and duplicate identities were exercised. This proves fixed
public-rule arithmetic, not journal completion or action credit.

The latest no-clarification campaign retained all 28 cases and independently
accepted 25/28 frozen predicates. NC08, NC10 and NC14 remain failures. Source
mutation during producer iteration was reproduced and fixed by freezing
communication evidence before optional producer iteration. The semantic gate
remains open; do not repeat prompts or relabel failures as qualification.

An all-105 capability matrix is now saved and its public pack/review/source
provenance checked. The author narrowed aggregate candidates to 22, scheduling
to 15 and scalar/date needs to 33 after critic corrections. Other candidate
memberships are relationships 42, CRM 24 and ranked selection 10. These overlap
and describe possible partial benefit, **not tasks unlocked**. Remaining broad
tags still need exact obligation-level review before being used as measured
coverage or a firm implementation ranking.

## Read these files, in this order

All paths below are absolute. Read applicable AGENTS.md files before edits.

1. `/home/hammad/projects/rl/docs/post-training/README.md`, then relevant
   canonical `04-framework.md`, `05-apis.md` and `06-observation-and-lineage.md`
   in that directory. The frozen baseline and its narrow amendments are the
   product authority.
2. `/home/hammad/projects/rl/docs/plan/automationbench-teacher-calibration-and-action-rewards.md`
   — living AutomationBench plan, especially Milestones 3–4 and current Progress.
3. `/home/hammad/projects/rl/docs/plan/verifiers-assessment-api-and-environment-migration-runbook.md`
   — native API, integration, migration and publication sequence.
4. `/home/hammad/projects/rl/docs/research/verifiers-assessment-api.md`
   — assessment/credit/alignment contract and assessor freedom.
5. `/home/hammad/projects/rl/docs/research/verifiers-manifest-assessment-architecture.md`
   — environment-owned manifest engine, evidence admission and code boundaries.
6. `/home/hammad/projects/rl/docs/research/verifiers-assessment-qualification/reward-candidate/capability-opportunities-105.md`
   and `.json` — readable direction and exact task/source inventory. JSON SHA
   at this snapshot: `a3b9fac18522f0cf11c4c0ef7d8a87d99a6e8589b615c7e8ea5f70e81d07bbd5`.
7. `/home/hammad/projects/rl/docs/research/verifiers-assessment-qualification/reward-candidate/manifest-authoring-batches.json`
   and `manifest-authoring-batch-01-review.json` through `11-review.json`.
   Old gap lists are historical; check current code before implementing them.
8. `/home/hammad/projects/rl/docs/research/verifiers-assessment-qualification/reward-candidate/acceptance-cohort-01/`,
   `acceptance-cohort-02/`, `acceptance-cohort-03/` — drafts and their limitations.

Public inputs are in
`/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/manifest-authoring-105/batch-01.json`
through `batch-11.json`. Each task links its exact retained episode. Full
reference data is under
`/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/luna-reference-campaign-01/`.
Raw runtime evidence is ignored machine-local data: preserve it, and do not
assume Git transfers it to another host.

Aggregate proof and reproduction script:
`/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/aggregate-prepaid-public-01/{qualify.py,verification.json}`.
The proof includes source/specification/member evidence and six file hashes.

Latest semantic review:
`/home/hammad/projects/rl/docs/research/verifiers-assessment-qualification/reward-candidate/no-clarification-semantic-live-03-review.md`.

## Repositories and ownership

- **Framework/docs/consumer integration:** `/home/hammad/projects/rl`.
  Relevant consumer code is under `packages/train/src/posttrain/train/integrations/`
  (`verifiers.py`, `verifiers_credit.py`, `verifiers_generation.py`,
  `verifiers_assessment_artifacts.py`), `packages/train/src/posttrain/train/assigned_rewards.py`
  and `packages/tracking-trackio/src/posttrain_tracking_trackio/assessment_results.py`.
- **Native Verifiers candidate:** `/home/hammad/projects/verifiers-credit-candidate-20261003`.
  Detached base `84ab782391bbfe1ac4f4ca32fa612e56d01b5b81`, with dirty work.
  Native API improvements belong here, not in copied framework modules.
- **Environment candidate:** `/home/hammad/projects/verifiers-environments-reward-candidate-20261003`.
  Detached base `a6d779fc1fdfde23f86e297125b3381b140cec2f`, with dirty work.
  Package: `environments/automationbench_v1`; source: `src/automationbench_v1`.
  Manifest engine, calibration and business verification belong here.
- **Simulator compatibility fork:** `/home/hammad/projects/automationbench`.
  Observed branch `codex/tool-fidelity-fixes`, base
  `e193bce99af1ea7cca272644a2b3a1676a587b4a`, dirty. Verify live state before changes.

These candidates are unpublished and do not equal the framework's executable
pins. Root still selects Verifiers `e6a3d9bbfe6959b97878f451fc721793a232cd5f`
and environment `11f4d712806d292c6c6a752af046f4e16c4f037e` at this snapshot.
Read pyproject/lockfiles to confirm current selection. Preserve every unrelated
dirty edit. Do not commit, publish, change pins or dispatch GPU training without
authorization. Update fork ledgers and consumer pages together when appropriate.

## The immediate independent assignment

Integrate the accepted aggregate evidence into the existing native manifest
occurrence path, then prove one useful deterministic report component. Recheck
the proposal against code; it is an architectural recommendation, not an
implemented API.

The proposed minimal design adds optional check-local aggregate aliases to
`ObligationCheck`. Qualified values become available only to `effect_match`.
An unavailable total must make report verification unknown, not make the report
obligation disappear. Retain full aggregate evidence; no caller closure flags,
aggregate dependency graphs, synthetic populations or new native producer lane.

Likely change surface inside the environment package:

- `contracts/obligations.py`: alias admission, aggregate context and evaluation.
- `contracts/models.py`: resolve declared population dependencies.
- `manifest_obligation_assessments.py`: capture/restore aggregate evidence and
  thread it through native assessment and existing planner/projector recomputation.
- `manifest_guard_assessments.py`: include aggregate dependencies in selector
  identity, preserving legacy identity when no aggregates are present.
- `contracts/predicates.py`: if needed, one bounded numeric-line comparison
  using the existing exact numeric parser.

Compute identical aggregates once per prepared view, not for every candidate
or tool effect. Recapture before accepting prepared evidence or publishing a
result. Start with simple local reuse; introduce a cache only when necessary,
keyed by authenticated source/view/contract identities and admitted before use.

First public example: on one qualified Gmail send, verify both the requested
controller recipient and the requested `Total amortization: $X` line against
computed public-policy totals. Do not hardcode 4,900 or require the entire email
to equal a template. Preserve duplicate/conflicting line ambiguity, missing ACK,
quoted/code/HTML limitations, unknown totals and equivalent supported currency
formatting. A numeric-line component does not verify entity names, accounting
lines, schedule changes or all guards. Keep it outcome-only until a separately
justified action-credit rule is qualified.

After integration, continue manifest authoring in batches of ten. Proposed next
batch: `finance.expense_split_allocation`, `finance.payment_reconciliation`,
`hr.buddy_assignment`, `hr.break_schedule_processing`, `marketing.ad_platform_audit`,
`marketing.content_repurpose`, `operations.zoom_training_setup`,
`operations.calendly_equipment_inspection`, `sales.calendly_no_show_followup`,
`support.helpcrunch_engagement_scoring`. The list is unique, in the selection,
and disjoint from earlier cohorts. Recheck individual public gaps; it is not
a promise that all ten are implementable with aggregates.

Separate next capabilities include two-sided key relations with duplicate
ambiguity; eligibility-first extrema with explicit ties; retained generated-ID
relationships; and required-read evidence. Read-before-action needs authoritative
read completion before action dispatch. State revision ordering alone can
falsely accept overlapping calls. No public prerequisite means no invented read
requirement; historical captures lacking timing must abstain.

## Validation and recovery

From `/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`:

```bash
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_manifest_aggregates.py tests/test_manifest_values.py tests/test_manifest_populations.py tests/test_manifest_predicates.py tests/test_manifest_tables.py -q --tb=short --strict-markers
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python /home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/aggregate-prepaid-public-01/qualify.py
.venv/bin/ruff check src/automationbench_v1/contracts/aggregates.py tests/test_manifest_aggregates.py
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/pyright --pythonpath .venv/bin/python src/automationbench_v1/contracts/aggregates.py tests/test_manifest_aggregates.py
git diff --check
```

The first command passed 286 cases at this snapshot. Broaden only for affected
integration paths or new failures. Native-only changes use that fork's `uv run`
workflow and its existing tests; follow its AGENTS.md constraints.

Integration acceptance must include unchanged old manifests; genuine native
handlers; forged prepared totals; source/spec/selector drift; unknown eligibility;
duplicate members; same-send binding; missing ACK; multiple sends; malformed or
conflicting output; reload/rescore/current-attempt admission; and a correct total
with an incorrect schedule. Preserve legacy official scalar behavior. Save
immutable source hashes and proof artifacts; do not overwrite failed campaign
history or silently hydrate incomplete captures. No model calls are needed for
these deterministic acceptance cases.

All current delegated workers were completed at handoff preparation. Verify
live worktree/activity before editing. If another agent runs concurrently,
agree explicit file ownership before mutations. Use an independent critic and
consult an efficiency reviewer before implementing a milestone.

## Preserve the downstream scope

After reward design and AutomationBench qualification: migrate maintained
environments through the proven API, complete publication/adoption gates, then
benchmark Qwen3.5 9B/4B/2B only on Luna-verified solvable tasks. Use those student
traces to build separate curricula for 2B and 4B. Collect ten fresh 4B rollouts
per selected task, then stop. Training and further reward iteration are separate
work. Trace-based tool/output budgets become efficiency pressure only after full
base success and required guard compliance; shorter incorrect solutions must
not earn a benefit.

Do not reduce the complete-library goal or fifteen-per-category qualification
gate to a passing component. Keep four counts separate: reviewed tasks, valid
declarations, qualified components and fully qualified tasks.
