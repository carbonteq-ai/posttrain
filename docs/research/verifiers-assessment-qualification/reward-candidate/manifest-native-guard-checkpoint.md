# Native manifest guard checkpoint

2026-10-04. This qualifies the local AutomationBench guard transport increment,
not the complete reward redesign or a published dependency.

## What is implemented

The existing manifest schema now admits record, finite-table and acknowledged
action sources. A single registered native assessment hook dispatches to the
declared check type. Record goals and conditional guards share that authoring
path; there is no named-task checking branch.

Guard targets are fixed from the source inventory before outcomes are evaluated.
Each candidate/action instance and the overall compliance finding has a native
run. Captures are shared and evaluation is cached once per source, view, contract
and check. Each run retains a small source/selector/input-bound output receipt.
Strict configuration validation rejects malformed identities, boolean instance
counts, unexpected fields and mismatched targets.

An observed prohibited effect publishes harm=1 with lower-is-better semantics.
The separately declared `per_effect_negative@1` credit rule transforms that
finding into a contribution of -1 on a derived penalty signal. It identifies
the exact retained execution and uses turn-boundary allocation. This is domain
credit, not a computed trainer advantage or a qualified token mask.

Credit accepts current findings only. It verifies the retained input, output
receipt, published parent, execution, rule and signal. Prior consumption prevents
duplicate penalties on rescoring. A current assessor that ends failed or
interrupted cannot authorize credit from its earlier partial output; the output
remains in history. Independently completed findings remain usable.

## Verification

The combined focused suite passes in two partitions: 384 cases in 10.54 seconds,
plus the large recorded Luna native replay, one case in 82.89 seconds. Scoped
Ruff and Pyright pass; repository diff checks pass. The initial full run exposed
one outdated boundary fixture after execution subjects were added to the shared
view; the fixture was corrected and its eight boundary cases pass.

The 20 guard-native cases cover factual/penalty signal direction, scalar reward
preservation, distinct execution recipients, repair and later acknowledgement
gaps, unknown policy values, allowed actions, explicit instance budgets, stale
attempts, omitted/tampered receipts and findings, native reload, and idempotent
rescoring. Eight additional boundary cases check exact source membership and
strict configuration. Two lifecycle cases check partial-to-failed/interrupted
runs.

The recorded test uses the original hash-bound
`operations.access_request_validation` Luna episode. Its declared property is a
bounded literal provisioning guard for processed queue rows. It retains known
nonviolations, publishes overall compliance as unavailable because the broader
effect scope is not closed, preserves the official scalar result, and roundtrips
native findings without changing the original file. Harmful executions are
separately tested using genuine simulator mutations inside explicitly
manufactured native envelopes. No observed harmful Luna behavior is fabricated.

Run from
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`:

```bash
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_manifest_contracts.py tests/test_manifest_engine.py tests/test_manifest_credit.py tests/test_manifest_assessments.py tests/test_manifest_tables.py tests/test_manifest_predicates.py tests/test_manifest_effects.py tests/test_manifest_guards.py tests/test_record_assessments.py tests/test_record_update_evidence.py tests/test_manifest_guard_assessments.py tests/test_manifest_assessment_lifecycle.py tests/test_manifest_guard_boundaries.py -q
```

## Important limits and next gate

The installed catalog still contains ten Simple Salesforce manifests. Guard
fixtures and the bounded recorded slice do not establish full-task contracts for
Access or any other category. Prose policy extraction, other service adapters,
multi-step obligations, success-gated efficiency, and complete task coverage
remain open.

Dense publication has a material scalability gap. This Luna case has six queue
rows and fourteen effect facts, producing 84 instance runs plus compliance.
Queued, running, partial and complete lifecycle records retain the full source
and input repeatedly. The successful full-fidelity replay reached approximately
7.5 GB resident memory. The inventory cap prevents unbounded enumeration but
does not make this representation suitable for library-wide dense checks.
Resolve native retention/batching without losing independent findings, source
authority or journal recovery before accepting dense publication at scale.

The next breadth gate is the frozen [105-task selection](cross-category-selection.json):
15 development tasks in each of seven categories. Its [coverage ledger](cross-category-api-gap-ledger.md)
separates native transport from reviewed manifest reward coverage and lists
shared capability gaps. Do not treat transport success as task qualification or
Qwen eligibility. No publication, dependency pin change, fresh collection, GPU
benchmark or training is claimed.
