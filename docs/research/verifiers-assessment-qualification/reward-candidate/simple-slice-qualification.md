# Deterministic direct-request slice

The isolated environment candidate now implements requested-state checks for
Salesforce case priority, HubSpot phone, Asana task creation and a calendar event.
It reuses the native assessment/credit publisher through task-specific evaluator,
signal and producer/rule revisions. `task.reviewed_simple_assessments=True`
selects only these four tasks; other tasks retain their normal class.

The predicates bind the exact approved user-request text. They check native record
IDs/fields, Asana's explicit simulator creation-record abstraction, and calendar
instants/duration/attendee. Equivalent timezone offsets are accepted; unspecified
extra attendees receive no invented violation label. Already-correct updates
produce an outcome without a pointless write or action accomplishment. Creation
requests require a new identity; duplicate creations retain both effects but
receive one accomplishment attributed to the first surviving qualifying creation.

Recording reconciliation checks every declared initial domain field against the
native-bound hydrated baseline, preserving scalar types, nulls and declared list
order/cardinality. Undeclared schema defaults may be present. Infrastructure
metadata is outside this direct-request projection. The retained chain must have
acknowledged effects, adjacent state revisions/world snapshots and an exact final
world match. This is recording reconciliation, not all-service policy coverage.
An otherwise supported final outcome can survive a coverage gap, but then has no
action recipient. Malformed service containers raise explicit adapter conditions
and become unavailable findings through the publisher.

Validation: **51 combined tests pass**: 16 Simple, 23 HR and 12 effect-evidence
tests. All four source-hashed actual development episodes run through `Task.score`
and WireTrace reload with unchanged official scalars. The normal loader selection,
wrong targets/fields, missing acknowledgements, equivalent offsets, malformed
states, hydration projections and duplicate creations are exercised. Scoped Ruff
and focused predicate/effect Pyright pass. No model calls or original-source edits.

The requested-state outcome and recording diagnostic now assess the whole trace.
Successful goal credit independently rederives the decisive native execution from
the source. Diagnostics, failed goals, already-correct states and missing action
attribution produce no action credit. The publisher validates the producer/rubric
revision and exact Simple signal/trace subject. Native progress snapshots are
deduplicated by assessment identity before requests are generated; they do not
multiply reward observations. Actual replay tests check trace subjects, execution
recipients, parent linkage, stale revisions, altered signal/subject rejection and
native reload. The 51-test combined batch and expanded 16-test Simple batch pass.

Subsequent qualification (2026-10-03): the Simple predicate suite now passes 17
tests, including revision-derived ordering and rejection of a purported complete
subchain beginning after revision zero. The four actual references also pass the
independent eligibility pipeline; see
[eligibility qualification](simple-eligibility-qualification.md). Each proof binds
a reviewed, explicitly empty guard inventory for its exact task-specific request,
two required checks (goal and recording coverage), original-score replay and the
original 16,384-token budget. This establishes candidate eligibility within that
declared scope; it does not close unreviewed environment-wide guards. Subsequent
source edits require resealing and revalidation before curriculum admission.

Remaining acceptance includes complete zero-action capture evidence, broader
concurrent capture-to-command checks and final accepted source closure. Initial
infrastructure metadata and undeclared hydrated populations are not independently
closed by the recording diagnostic; relevant clock/policy/population facts require
their own authored checks.

Working directory:
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`

```sh
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_simple_evidence.py tests/test_hr_assessments.py tests/test_effect_evidence.py -q
```
