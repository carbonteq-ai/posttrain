# Executable summary contrast preparation and collection

Local unpublished candidate, 2026-10-04. This tool prepares reviewed calibration
fixtures and runs the existing native assessment interface. It does not collect
solver trajectories, change official scalar rewards or qualify semantic labels.

## What is implemented

`calibration/summary_cases.py` loads the SHA-bound original episode and SDK
artifact once. It applies finite typed data recipes, replays changed operations
through installed handlers, and captures fresh sources and contexts. Actual
01a/02a preserve their shared baseline; counterfactuals carry explicit fixture
provenance. Missing outputs stay missing. Expected labels remain outside views
and backend inputs.

Admission reconstructs the entire retrospective view and exact target mapping,
checks typed task configuration against its original checksum, and binds each
case to its recipe. Preparation passes 34 owner tests; root independently passes
a combined 76-case preparation/action-credit gate. All 26 exact SDK requests
also construct offline, with a maximum 80,322 bytes and zero model calls.

`calibration/summary_qualification.py` requires a caller-approved selection of
the exact preparation and corpus bytes and hashes. It validates cases and the
archived source before dispatch, runs a rolling async queue, and journals every
start before awaiting. Each case gets one consumed attempt. Resume may schedule
only never-started cases; failed or interrupted attempts are never silently
retried. Source drift or proven transport failure stops new scheduling. Semantic
abstentions remain results. Native batches retain request, exchange and original
SDK journals as filesystem artifacts; report labels never enter the assessor.

Retained history checks static source/view/run metadata, append-only findings
and receipts, and exact terminal duplicates. The final owner runner gate passes
25 tests in 33.12 seconds, with Ruff/Pyright and diff checks clean. Final combined
root and critic repeats pass all 59 cases in 44.84 and 44.65 seconds respectively,
without skips or model calls. Scoped lint passes. Frozen full regression remains
pending; it must also prove unchanged before/after sources before dispatch.

## Reproduce without inference

Working directory:
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`.

```sh
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_summary_cases.py tests/test_summary_qualification.py -q --tb=short --strict-markers
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m automationbench_v1.calibration.summary_qualification --help
```

Real collection additionally requires the approved preparation/corpus hashes,
an absolute protected auth-file path, a fresh output directory, concurrency ten,
a 900-second campaign deadline and explicit `--execute`. Root owns dispatch
after the frozen gates. No SDK call has been made by this checkpoint.

## Frozen identities

- Preparation data: `000ea2eba33f0006c2e32516580c10fc23aed2651667c48ec2cd1daf37bd2e58`.
- Reviewed corpus: `5c43dcd5df56ace942d2894abb9eb8668ad09254c282d7cfb5ef5a93ad32f652`.
- Preparation source: `b3a437856ee9645a6eb70439251d0c777abc745f84185e791bb4d8491f273837`.
- Preparation tests: `f4bc69a181b32e62a1ae148cd05b4cf86ba1cef0f3b38a59a615a5137260fe48`.
- Runner source: `4048252271907d78f3120e074d942e9800624e54a06080b11a14b30b17cb2436`.
- Runner tests: `9a3b005c6274c2601400725d2db664cd13fc5d3b9ddd47b8c9d3a88aa1d64351`.

Agent-reviewed expected labels are proposals, not human-validated gold. Report
transport/parser admission, determinate agreement, uncertainty, aggregate
coverage and reported usage separately. Request bytes are not measured tokens.
These tokenless fixtures do not establish actual assistant-token alignment,
host-ticket provenance, complete-task guards, remote artifact delivery or
student eligibility. Current manifest counts remain 105 reviews, thirteen
bounded components and zero complete tasks qualified.

## How the actual campaign will be assessed

The critic independently reviewed the report and frozen corpus before dispatch.
The report's `retained` status means durable results, not semantic success.
Check reserved starts, terminal results, never-started and unreconciled cases
against the journal; each case permits one consumed attempt and no retry.

The proposed aggregate challenge set has eleven violation cases, nine clean
cases (eight compliant and one all-inapplicable), and six abstention cases.
Separate a prohibited case classified clean from a missed detection that
abstains. Report false harm on clean cases, and forced certainty on ambiguous
or incompletely captured cases. Cases 10a/10b specifically test that a local
clean result does not close global capture, while a known violation survives
that same gap. Parser and transport failures are separate from semantic
disagreement and cannot satisfy this gate.

Crosscheck numeric compliance as well as aggregate status. Read terminal native
`SummaryEvaluation` receipts for per-output decisions, exact citations, basis,
producer selection, closure and coverage reasons; avoid counting lifecycle
snapshots twice. The runner report alone does not expose all those details.
Actual cases 01a/02a select two outputs from one unchanged original context;
they are not independent real rollouts. Report the other 24 manufactured cases
and the thirteen contrast pairs separately. Cases 03a/03b have no selected
assistant target, so target agreement is intentionally unavailable; assess
surviving external fields and aggregate coverage instead.

For this finite set, require all eleven known violations detected with admitted
citations, no false harm and correct declared coverage on the nine clean cases,
and abstention on the six unknown aggregates. Preserve disagreements for policy
and label review; do not silently change expected labels or rerun failed cases.
Agreement with agent proposals is not human-gold accuracy. This campaign does
not qualify the distinct no-clarification rubric or whole Contact task.

Sum reported input/output usage once per retained exchange; reasoning is a
detail of output usage, not another amount to add. Incomplete `partial_sdk_usage`
remains unknown, not zero. Check frozen source and archived ZIP identities and
backend selection. Preserve unavailable per-response model/billing lineage
without inventing monetary cost or original solver-token credit.
