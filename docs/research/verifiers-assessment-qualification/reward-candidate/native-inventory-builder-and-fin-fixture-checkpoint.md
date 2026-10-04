# Native inventory facts and the public FIN fixture

2026-10-04. Local unpublished work. No model calls, training, commit or immutable
pin update. Whole-task counts remain unchanged.

## Native source builder

`native_invocation_source.py` derives a small factual projection from a sealed
native `SourceSnapshot`. It retains original emitted-call coordinates, generated
attempt ownership, model-call terminal metadata and server parents resolved by
the native public resolver. It does not duplicate token arrays or claim closed
action capture. SDK historical episodes remain unlinked.

Missing or malformed calls, conflicting generated coordinates, nonterminal model
calls and missing/duplicate committed sampled nodes produce unavailable facts.
The final owner gate passes **24 cases in 2.19 seconds**; the critic independently
repeats 24 in 2.21 seconds. Scoped Ruff/Pyright and diff checks pass.

Source SHA-256: `423b6159f456915e0283958806f8e8652d8903273a646c0ee70a212460911974`.
Test SHA-256: `b6341f5f9760d5e3bc15634e907bde54681201200c64c89425cf0711718f8c1a`.

Next integration passes the executor's original `SourceSnapshot` explicitly as
an optional keyword through invocation/external/summary APIs. A prepared view
cannot supply it. Native capture must rederive the facts, compare task/events/
writes with the sealed source, partition harness/server ledgers and reconcile
all original emitted calls and physical retries. This is pending implementation
and real AutomationBench MCP qualification; the builder alone proves no closure.

Forwarding qualification now passes **138 SDK/invocation/external/summary cases
in 11.12 seconds**, including a native scoring test that observes the same
executor snapshot object in all three summary bridge stages and matches its ID
to the assessment run. SDK behavior is preserved. Scoped root Ruff/Pyright pass.
Native reconciliation and actual HTTP/state-controller gates are still in
progress; this is not their acceptance. Current forwarding source SHA-256 values:

- `contracts/external_outputs.py`: `c87f4e1a1c5b242a290f1eb3053334fe4041517732c833fbe2baa46bf92703c1`
- `contracts/summary_policy.py`: `dee675db02677798c1996545b5b5b9f065599e7371a610444a7c8b6a5d15692a`
- `manifest_summary_assessments.py`: `28b7b50137e2fb56b38fe3d9d813a401749699040b25c9f4ea7b39668b16a865`
- `tests/test_manifest_summary_assessments.py`: `e6a27b3a93f91af4545256b64a08cbe9bd3c2c637f14f9abdb0e5ed6da012fa7`

## Fixture defect and repair

Public `support.gorgias_refund_processing` requires escalation to Jira project
FIN, offers `jira_create_issue` but no project creation tool, and originally
supplied only `jira.actions={}`. The newly faithful handler correctly returned
`jira_project_not_found`, with zero issues or audit records. Local upstream
comparison confirms its older handler accepted symbolic project parameters.

The task's initial state now explicitly declares project
`{id: fin, key: FIN, name: Finance}`. An AST-based complete task comparison proves
that removing this one collection restores exact original task equality:
prompts, tools, assertions and other state are unchanged. Unknown project
references still fail atomically. Historical episode snapshots are preserved.

Sibling Jira/domain qualification passes **73 cases**. The critic repeats the
four actual public API/Zapier success/rejection tests. The task-file refresh is
byte-identical in the environment vendor, whose forced-source Jira gate passes
**55 cases in 0.33 seconds**. Current task SHA-256:
`3318468070b90cc07a87bf4ecdddbd03792527f97a42653129321246a8a8e9e8`.
Sibling regression test SHA-256:
`da38f1ec3fa0fd3b82093c168bd4ab5f75899247af593a7019b551b6703d7ef2`.
The environment Support extraction gate passes **26 cases in 2.61 seconds**
after the refresh; a manufactured empty-project world still yields no creation.

## Stable-source regression

Final source-frozen full-package regression supersedes the pending status below:
session **10456**, PID **3517045**, terminates with **2,387 passed, five skipped
in 661.93 seconds**. From the environment package directory:

    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests -q --tb=short

No candidate source or test edits occurred during this run. The source freeze is
released for the next optional SDK summary-backend increment; these results do
not qualify that later code or semantic model accuracy.

Before the next source integration/vendor increment, all source writers released
their files and `tests/test_calibration_benchmark.py` passed **eight cases in
106.79 seconds**. This resolves the observed source-hash test failures caused
by concurrent edits. It does not qualify later edits or turn the earlier broad
run into a passing full-package gate. Repeat source-frozen acceptance only once
the next increment's writers have released their files.

Environment commands, from
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`:

    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_calibration_benchmark.py -q --tb=short
    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_native_invocation_source.py tests/test_support_evidence.py -q --tb=short

The forced-vendor gate imports the public task module from the environment's
`src/automationbench` before invoking the sibling's existing Jira test file.
Native inventory integration, summary model accuracy, fully qualified manifests
and the remaining campaign milestones remain open.

## Native reconciliation and real transport follow-up

The shared native branch now validates the supplied sealed snapshot and compares
its exact selected task/events/writes to working material. It rederives the
original emitted-call and terminal-generation populations, groups harness and
server events separately, checks clean allowed dispatch/after lifecycles,
authorized parent routes, distinct physical attempt identities, acknowledgements,
serial/final worlds and the final returned value. Native pairs retain original
node/emitted-call coordinates and physical retry identity. Missing native source
cannot establish native closure; SDK-declared episodes retain their historical
coverage-only route even when an optional snapshot is supplied.

Initial combined source/inventory/SDK/external/summary/live transport gate:
**180 passed in 14.68 seconds**. The critic independently repeats **41 native
core/SDK/live transport cases in 5.64 seconds**. Three downstream omission tests
pass: missing original execution keeps compliance unavailable while preserving
known field effects, and a controlled harmful certificate still produces zero
compliance. Controlled certificates do not qualify prose accuracy.

The real transport test starts an AutomationBench MCP subprocess over HTTP and
uses authenticated native state GET/PUT and receipt acknowledgements. A Gmail
read and two-field Contact update yield two physical invocations and explicit
host dispatch links. Their HTTP results agree with native receipt returns.
Sealed capture through `ManifestAssessmentTask`, wire reload and retained world
reconstruction preserve coverage. An added original call without execution opens
the inventory while the known Contact facts remain available.

This qualifies capture mechanics. Original sampled nodes/model-call metadata are
deterministic test inputs, and initial schema defaults are deliberately normalized
for the fresh fixture. It proves neither actual model/token/parser fidelity nor
whole-task or semantic summary compliance. Rejected/raised/interrupted control
lifecycles remain conservatively open in this slice. Same-parent multiple-physical
retry coverage subsequently passes in the final fifteen-case native suite: one
original sampled call/ticket has two actual simulator operations with distinct
invocations, unit-step acknowledgements and attempts zero/one, with the after
hook tied to the final return. Both qualified entries and attribution pairs
remain distinct; no SDK equality join is used. Final native test SHA-256:
`7d5eb16cd3d04273d5d4902c1a6496653c40b63aaf109006f5d31d1d785dbe77`.
Dense-scale performance
has not been measured; no new cache was added.

Production source SHA-256:
`165a657fc7d66024b7874175bab9d605b3c4b22c5f8df2af8d607e8ccc16a45a`.
Live transport test SHA-256:
`deae8bb64db6e35355e0e25518d38725075c619577d2ed155fadf5d4ac7a28ac`.
Downstream summary omission test SHA-256:
`b040b1cb93f63cefa4227f9a6c2812ba6cfb5c8fb00d861fc4dc1caf4c24dcec`.
