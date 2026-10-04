# Manifest record-update checkpoint

Validated locally on 2026-10-04. This qualifies the first record-update slice,
not the complete reward redesign, original-token alignment, or a release.

The environment candidate is
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`,
based on `a6d779fc1fdfde23f86e297125b3381b140cec2f` with unpublished local changes.
It uses the existing native candidate at
`/home/hammad/projects/verifiers-credit-candidate-20261003`.

## What works

Ten packaged Opportunity manifests select record identities, fields, comparisons,
reviewed input bindings, and credit policies as data. Runtime code does not call
the previous named-task evaluators. A changed task, target, or desired value is
exercised by changing manifest/data only.

Shared evidence is captured once. Each check publishes a native assessment run
and retains its immutable evaluation receipt. The credit consumer validates the
source, manifest, selector inventory, published finding, current attempt, and
input receipt before selecting a recipient. A qualified completion can receive
one contribution; explicitly joint obligations can share one contribution with
both parent findings. Repeated scoring does not consume the accomplishment twice.

The native tests replay all ten original Luna episodes, checking their source
bytes against the retained development index. They preserve the original scalar
rewards, serialize and reload assessments and credit assignments, rescore without
duplicate credit, and leave the original files unchanged. Separate simulator
tests cover alternative generic/API tool paths and failure cases.

Counterexamples cover cross-source and cross-selector cache reuse, unrequested
field changes, omitted initial fields, repeated break/restore, malformed dates,
numeric booleans, missing acknowledgements, changed public-source bindings,
overlapping undeclared credit, conflicting history, and missing current runs.
Correct final state remains distinct from attributable action credit.

## Validation

From the environment candidate directory:

```sh
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_manifest_contracts.py tests/test_manifest_engine.py tests/test_manifest_credit.py tests/test_manifest_assessments.py tests/test_record_assessments.py tests/test_record_update_evidence.py -q --tb=short
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/pyright --pythonpath .venv/bin/python src/automationbench_v1/contracts src/automationbench_v1/manifest_assessments.py
.venv/bin/ruff check src/automationbench_v1/contracts src/automationbench_v1/manifest_assessments.py tests/test_manifest*
git diff --check
```

Final result: **215 tests passed in 8.09 seconds**, scoped Pyright zero errors,
Ruff clean, and diff check clean. This is a focused local qualification, not an
all-environment suite. The critic independently reran the original attribution
counterexamples, including an initially successful record damaged and restored
after its initial field was omitted.

## Remaining work

Add finite collections, exact lookups, conditional predicates, qualified effect
matching, and scope-aware absence checks for structured guard slices. Harm
findings must survive repair and partial capture; positive transition credit
must not be reused for them. Prose policy/purpose extraction remains a component
gap wherever structured public data cannot establish it.

Full families, efficiency signals, token alignment, archives/consumer gates,
publication and immutable pin adoption remain open. No model collection,
training, release, or dependency adoption occurred in this checkpoint.
