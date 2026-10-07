# Native reassessment and revision-aware evidence

The isolated AutomationBench environment candidate now includes a concrete
`TaskScoreReassessmentVerifier`, rather than only a replay protocol and synthetic
fixture producer. Composition supplies a trusted deterministic task class and
accepted source digest. Artifact content cannot select an import or executable.

The adapter checks frozen task and producer identities, reloads a detached native
episode, restores the retained final simulator world and runtime artifact bytes,
and executes `Task.score`. It regenerates assessment context, findings and credit.
Incomplete native assessment/credit finalization fails explicitly. Semantic
comparison still belongs to the eligibility validator and preserves native
execution IDs, source/view identity, signals and parent linkage. Producer-generated
run and contribution UUIDs are excluded by the documented semantic comparison.

The synchronous interface supports invocation from an active asyncio collector
using one bounded worker for the offline scoring loop. This performs no model
inference. It blocks the caller during validation; admission replay should be
cached only under independently approved, freshly checked source closure. This
is a correctness bridge, not a claim of optimal bulk replay throughput.

Five tests replay actual SHA-bound Salesforce, HubSpot, Asana and calendar
development episodes, compare independently generated results, preserve original
bytes and a restored runtime artifact digest, check trace outcomes and execution
recipients, exercise the async bridge and reject a stale producer. All five pass;
the earlier combined Simple/HR/effect/index/replay batch passed 65 tests before
the additional three replay cases were added. Scoped Ruff and
candidate-aware Pyright pass. The placeholder test contract does not grant task
eligibility or claim that its guard has been implemented.

`EffectIndex` is an ephemeral environment helper over existing native-derived
world transitions. It caches recursively immutable worlds by SHA and keeps input
occurrence order distinct from a revision/world-linked serial chain. Read-only
calls retain their acknowledged revisions. Missing links, branching revisions or
world conflicts remain explicit. Scoped service history preserves intermediate
effects and does not treat unknown missing capture as unrelated.

Twelve new index tests plus twelve effect-adapter tests pass. An actual retained
offboarding trace has 16 occurrences but only 10 distinct snapshots; reversing
the transition input still reconstructs its revision-derived chain. Recording a
complete captured chain does not prove complete invocation capture, guard
compliance, policy authority or external service effects.

Remaining work includes production use in closed task eligibility, integration
of the index into domain predicates, qualified capture-to-command mapping and
complete per-task goal/guard/coverage contracts. The corrected frozen-verification
tampering fixture also passes its explicit revalidation test; this is distinct
from passing a whole eligibility suite. No reference inference, Qwen
benchmark or training was launched by these changes.

## Optional assignment and diagnostics

An independent probe reproduced a native API defect: an empty credit plan raised
`credit requests must account for registered hooks`. Hook registration is a
capability, and a task with no justified action attribution must not manufacture
credit. The isolated native candidate now permits empty plans and subsets of
registered rules; unknown hook names still fail before dispatch. HR recording
diagnostics remain assessments and no longer create neutral learning contributions.

Validation after this change: all 50 native judge/assessment/credit tests pass,
including empty-plan, unused-hook and unknown-hook tests; all 44 HR/Simple/native
reassessment tests pass. Scoped Ruff and native diff checks pass. The native
tests used the environment candidate runtime and the already installed RL
pytest-asyncio plugin exposed in an isolated temporary plugin path; no production
dependency or pin was changed. Initial runs without that plugin were test-runtime
failures and do not count as behavioral qualification.

## Current assessment attempts and assignment history

`CreditPlanningContext` supplies current declared assessment run/attempt identities
and assignments retained before this scoring call. It is ephemeral context over
existing native history, not a new trace store. Both `Task.plan_credit` and
`Env.plan_credit` default to the existing `credit_requests` hook for compatibility.
The runtime supplies boundaries without selecting the newest score or silently
deduplicating an algorithm's independent findings.

The reviewed environment publisher selects only current declared attempts and
skips an already valid assignment for the same source, rule, recipient, channel,
signal and value. Conflicting values fail explicitly. Failed current assessments
cannot fall back to earlier successful findings. Seventy-five native judge/trace
tests pass, including both trace- and episode-level repeated scoring and
failed-current-attempt cases. Actual marketing and cash-flow scorer tests retain
new findings on a second score call without multiplying assignments. Retained
running/partial/complete journal snapshots remain distinguishable from completed
assignments when reporting counts.

The test interpreter is the existing isolated AutomationBench environment's
`.venv/bin/python`, with the installed pytest-asyncio plugin exposed through
`/tmp/verifiers-credit-test-plugins-20261003`. No packages were installed for this
qualification. A mistaken exploratory `uv` invocation created an empty original
Verifiers `.venv` pointer; inspection found no installed native runtime removed.
Original sources, reference artifacts and production pins were preserved.

## Benchmark trust checks

The critic implemented fresh selected-proof/source checks before each dispatch
and all-source checks at terminal publication, native result/status reconciliation,
same-occurrence discarded-trace checks and independently recorded controller
interruptions. Recovery reuses an exact already-published interruption receipt
without overwriting original episode bytes. The full eight-test benchmark suite
passed before the final two recovery fixes; the three affected tests then passed.
Candidate-aware Pyright and scoped Ruff pass. The subsequent source-stable batch
passes all 31 eligibility, frozen-verification, benchmark and guard-contract tests
in 202.56 seconds. This completes the bounded scaffold regression; actual Qwen
runtime, model identity, budget enforcement and the final frozen pool remain open.
Further producer changes require the relevant source-closure qualification again.

To reproduce that batch, use the isolated environment working directory above:

```sh
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_calibration_eligibility.py tests/test_calibration_frozen_verification.py tests/test_calibration_benchmark.py tests/test_calibration_guard_contracts.py -q
```

Four actual Simple references separately pass source-bound candidate eligibility
with reviewed empty task-specific guard inventories. See
[the qualification report](simple-eligibility-qualification.md) for exact source
identities, original budgets and the independent sealed-file audit. Empty reviewed
inventory is an explicit policy decision, not evidence of universal safety.
