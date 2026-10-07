# Conditional guard components

Local checkpoint, 2026-10-04. The shared deterministic components now exist and
pass focused qualification. **They are not yet connected to manifest catalog
loading, native guard assessment runs, or native negative-credit publication.**
The live manifest task bridge still supports the previously qualified record
checks. No full access-task reward redesign is claimed.

Implementation remains in the isolated AutomationBench environment candidate:
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`.

## Implemented code

| Module under `src/automationbench_v1/contracts/` | Implemented responsibility |
| --- | --- |
| `base.py` | Shared immutable model base and identifiers, extracted without changing record-manifest semantics. |
| `tables.py` | Declared flat worksheet capture, stable scoped row identities, canonical cells, source/selector digests, population completeness, and exact-key left lookup. Missing collections differ from empty ones. Ambiguous matches cannot supply unique context. |
| `predicates.py` | Declarative field/literal comparisons, finite membership, bounded conjunction/disjunction/negation, explicit field types and allowed values, conditional applicability, and safe whole-text concatenation. Missing or out-of-domain fields stay unknown. |
| `effects.py` | Acknowledged Asana creation and section-placement facts with exact invocation/effect identities. Scope completeness is separate from individual positive witnesses. |
| `guards.py` | Conditional per-candidate/per-effect harm findings, distinct stable instance keys, scope-aware compliance, and explicit `-1` per-effect selection. Source and declared selector checks precede evaluation. This selector is a component utility, not yet native credit transport. |

The adapters recognize service schemas and installed simulator operations. They
contain no access-task policy, manager-title ranking, exact task prompt, or
provisioning-purpose classifier. Manifest predicates own those policy values and
whole matching forms. Text concatenation evaluates no format expressions and
does not coerce missing values into a name.

## Evidence

The final combined test command passed **345 cases in 8.73 seconds**. This
includes the previous record checkpoint and 130 new component cases. Scoped
Ruff, Pyright and diff checks passed.

The Asana adapter replays a retained Luna access episode whose bytes are checked
against the development index. It recovers three qualified creation facts and
three qualified section-placement facts. A separate bounded processed-request
check demonstrates data-driven matching while keeping whole compliance
unavailable because the mixed-operation scope is not qualified. It produces no
harm credit for that successful recorded slice. Simulator counterexamples supply
the harmful cases; do not describe them as failures observed in that Luna trace.

Counterexamples require harm to survive later deletion and a missing later
acknowledgement. Invalid field types and unknown statuses cannot select an
inapplicable branch. An ambiguous directory cannot erase an independently known
processed-request violation. Duplicate native row IDs or logical candidate keys
produce unknown for affected candidates while preserving unrelated identifiable
harm. A processed and pending request sharing the same matching email cannot
receive a speculative penalty. Changed sources, effects, or worksheet selectors
cannot consume another capture.

Unique declared row keys are not enough for attribution. An effect matching only
an email may still belong to either of two requests with different departments.
The default `match_cardinality=unique_candidate` requires exactly one definite
action match and no unknown potential matches. A whole matching form that includes
the distinguishing department can resolve the ambiguity. Broader per-candidate
rules require an explicit declaration and do not bypass credit aggregation.
Table evidence separates complete raw enumeration from fully qualified identity
scope so an unrelated duplicate ID can leave an independent positive witness
usable without manufacturing compliance.

One adapter boundary needed correction during review: the first native Asana
append can begin with an absent action bucket. The simulator's documented
`AsanaState.record_action` append semantics qualify this positive occurrence when
the service and actions map are explicit and the matching action, result,
persisted object and acknowledgement agree. That does not fabricate an initial
collection or establish complete episode scope. Missing services/actions remain
unavailable.

## Reproduce

Working directory: the environment candidate above. Use its existing runtime.

```sh
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_manifest_contracts.py tests/test_manifest_engine.py tests/test_manifest_credit.py tests/test_manifest_assessments.py tests/test_manifest_tables.py tests/test_manifest_predicates.py tests/test_manifest_effects.py tests/test_manifest_guards.py tests/test_record_assessments.py tests/test_record_update_evidence.py -q --tb=short
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/pyright --pythonpath .venv/bin/python src/automationbench_v1/contracts src/automationbench_v1/manifest_assessments.py tests/test_manifest_tables.py tests/test_manifest_predicates.py tests/test_manifest_effects.py tests/test_manifest_guards.py
.venv/bin/ruff check src/automationbench_v1/contracts src/automationbench_v1/manifest_assessments.py tests/test_manifest*
git diff --check
```

## Next implementation gate

Extend the single installed manifest contract and catalog to reference these
sources/checks. Connect the existing native bridge to stable guard instances and
validated retained output receipts. Plan targets before verdicts, including
unavailable populations and malformed instances, rather than publishing only
detected harm. Avoid evaluating the whole candidate/effect matrix for every
instance. Preserve the factual `harm=1` finding and publish explicitly transformed
negative credit with appropriate derived signal metadata. Distinct occurrences
remain distinct; overlapping findings require declared aggregation. Prove native
reload and prefix/rescoring deduplication using both recorded and simulated cases.

Efficiency review recommends one source-bound capture and one evaluation per
guard, with each request selecting a predetermined instance. Index lookup tables
once, keep pure deterministic computation synchronous, and publish a separate
compliance/coverage target even when no candidate instances can be resolved.
Bound the matrix explicitly; budget overflow must publish unavailable coverage,
never silently remove instances. Retain shared input once and avoid copying a
whole matrix into each output receipt.

Complete absence still requires a qualified operation scope. Current Asana
scope excludes unsupported calls and unknown/missing dynamic populations, so
mixed-workflow compliance frequently remains unavailable. Broader scope
qualification, policy/purpose extraction, positive access milestones, conversion
guards, efficiency signals, full-task eligibility, release/adoption and all Qwen
campaign gates remain open. The goal remains active.
