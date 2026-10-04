# Exact values and the second public-input batch

Local candidate checkpoint, 2026-10-04. Redesign remains in progress. No
publication, dependency adoption or whole-task qualification is implied.

Environment-owned `contracts/values.py` supplies explicit input formats, exact
rational arithmetic, declared rounding and deterministic date operations.
Missing inputs, malformed timestamps, zero divisors, unsupported formats,
unresolved ties and oversized expressions remain unavailable. It uses neither
the host clock nor ambient Decimal precision. Results retain raw values, source
paths and rounding choices.

Predicates accept a `derived` operand, with compatible typed derivations required
on both sides. Obligation validation discovers their observed field references
and rejects future-effect context in initial requirements. Ordinary strings do
not silently acquire amount or date semantics.

Batch two reviewed ten public inputs. Its negotiated CRM amount component for
`sales.docusign_void_resend` is installed as
`contracts/tasks/sales-docusign-crm-amount.json`. Four public-source bindings
support the target Opportunity amount of 175,000. Native replay checks the
finding, exact execution credit, unchanged scalar rewards, reload, once-only
rescoring and unchanged original episode bytes. Counterexamples cover wrong
targets, changed authority, duplicate/noop updates and damage followed by repair.
DocuSign envelope/template/terms/confirmation coverage remains open.

## Verification

**322 focused cases pass in 22.96 seconds**, with scoped Ruff and Pyright clean:

```sh
# cwd: /home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_manifest_obligations.py tests/test_manifest_values.py tests/test_manifest_predicates.py tests/test_batch01_manifests.py tests/test_manifest_contracts.py tests/test_manifest_engine.py tests/test_manifest_credit.py -q --tb=short
```

The independent critic reproduced invalid offset-minute normalization, derived
fields escaping context validation, and copied boolean path indices becoming
integers during JSON serialization. Fixes use strict offset grammar, a complete
derived-field walker and Python-mode fresh admission. Regression tests cover
each; the critic independently confirmed all three derived admission fixes.

The catalog has thirteen entries: ten original pilots plus three bounded
components from the 105-task sample. Batch three has also completed public review;
thirty sample tasks are reviewed, with no newly whole-task-qualified tasks.
Sheets native registration and the next financial guard proposal are concurrent
work. The earlier all-105 transport pass was measured with two sample components;
it is not a fresh all-105 qualification of this source revision.

Remaining gates include supported service evidence, complete task goals/guards,
semantic qualification of fifteen tasks per category, the rest of the library,
efficiency gating, consumer/token alignment, publication and frozen eligibility.
