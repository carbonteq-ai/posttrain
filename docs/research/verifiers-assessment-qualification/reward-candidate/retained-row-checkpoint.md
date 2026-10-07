# Terminal-row verification checkpoint

Local, unpublished AutomationBench environment increment, 2026-10-04.
Native outcome publication is locally qualified; positive completion-action
credit remains a separate gate.

`contracts/retained.py` adds `sheets.retained_when@1`: select each original
initial row, evaluate a public eligibility predicate, then verify the same
native record in finalized terminal cells. Initial and final projections are
rederived from the exact raw source. A successful past write followed by damage
fails. Repair can restore the outcome. Closed absence fails; missing identity,
fields or finalization abstains. Same-ID relocation is accepted; replacement at
the same row position cannot satisfy the original record. This operator infers
no useful action or positive credit.

Native final ID duplication, mixed scope and copied identity tampering are
rejected by `sheet_effects.py`. Initial population admission requires the
initial source and exact same sheet/tab selector. Eligibility cannot read the
terminal or effect context. Scope completeness is independent of success;
unknown action history does not erase independently known terminal state.
Per-record checks do not implicitly prove that logical names are unique.

From `/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`:

    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_manifest_retained.py tests/test_manifest_contracts.py -q --tb=short

119 passed in 1.93 seconds. Before four extra selector-admission regressions,
the retained/schema/Sheets/native/prepaid-guard combination passed 195 cases in
12.97 seconds. These are distinct executed suites, not additive unique counts.
Scoped Ruff and Pyright pass. The critic independently reproduces damage,
repair, incomplete capture, missing final fields, missing ACK, replacement and
future-context rejection without finding an outcome-core blocker.

Native `manifest_retained_assessments.py` publishes one trace-addressed outcome
per initial member plus scope coverage. Configuration, source/view, candidate,
current complete attempt, receipt and parent finding are validated against
recaptured state. Failed or incomplete assessment runs cannot provide valid
completion. This publisher emits zero credit assignments.

The root independently ran:

    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_manifest_retained.py tests/test_manifest_retained_assessments.py tests/test_manifest_contracts.py tests/test_manifest_sheet_effects.py tests/test_manifest_sheet_assessments.py tests/test_prepaid_manifest_guard.py -q --tb=short

222 passed in 17.35 seconds. The 23 native cases include the original SHA-bound
prepaid Luna episode, using a development-only public ineligible-balance
preservation declaration: two required rows verify and three are inapplicable.
It preserves official scalar results and original source bytes; native reload
and rescoring retain outcomes without inventing credits. This is neither the
complete amortization goal nor a newly installed positive task manifest.
Scoped Ruff and integrated scoped Pyright pass with zero errors.

Independent review subsequently exposed a coherent source-substitution gap:
recapturing projections from a forged input was not proof that it matched the
executor's sealed source. The native candidate now provides retrospective-only
`AssessmentContext.retrospective_source()` and the retained assessor compares
the exact source projection before cached evaluation. The worker's 24 native
tests pass, and the critic's original reproduction now fails before publishing
any finding. The earlier 222-test result predates this repair; equivalent source
admission in record/guard/occurrence publishers and a fresh integrated run are
requirements before a shared source-authentication claim.

The shared repair is now source-stable. All four publishers call
`manifest_source.admit_manifest_source` before evaluating or entering a result
cache. A bounded four-entry cache stores immutable source/input wire strings,
checks exact bytes on hits, and decodes fresh working copies when needed.
Visibility and executor identity checks occur on every invocation. The worker's
final fast publisher suite passes 82 tests, excluding the one dense replay;
Ruff, explicit-interpreter Pyright and diff checks pass. The critic independently
passes all four original source-substitution regressions and four cache seams,
including changed bytes under a copied digest and retrospective-to-prefix scope.
Root dense performance replay passes on the repaired source revision:

    env PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src /usr/bin/time -f 'elapsed=%e peak_rss_kib=%M' .venv/bin/python -m pytest tests/test_manifest_guard_assessments.py::test_native_guard_actual_hash_bound_luna_access_replay -q --tb=short

One case passes in 75.07 seconds pytest / 75.91 seconds elapsed, with 470,188 KiB
peak RSS (about 459 MiB). The prior pre-admission repair measurement was 71.14
seconds / 469,436 KiB. These are operating measurements rather than controlled
speedup estimates: another short validation job overlapped the new run's early
phase. All-library CPU/scale qualification remains open.

Final root integrated command on the repaired revision:

    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_manifest_retained.py tests/test_manifest_retained_assessments.py tests/test_manifest_contracts.py tests/test_manifest_sheet_effects.py tests/test_manifest_sheet_assessments.py tests/test_prepaid_manifest_guard.py tests/test_manifest_assessments.py tests/test_manifest_guard_assessments.py tests/test_manifest_obligation_assessments.py -k 'not actual_hash_bound_luna_access_replay' -q --tb=short

281 passed, one dense replay deselected, in 25.84 seconds. The dense case passed
separately as recorded above. These results supersede the earlier pre-repair
publisher qualification; they still do not establish whole-task manifest coverage
or positive retained-completion allocation.

The selected native scoring/trace/judges tests and Ruff pass after the accessor
addition and three explicit native type-safety checks. The two affected native
source modules pass focused Pyright. An earlier manual invocation also included
the full scoring test file and reported 26 errors; three source-module diagnostics
are now resolved, while the test file's broader static cleanup remains open.
This is not a claim that all native candidate/release gates pass.

Historical remaining work at this checkpoint was completion-action allocation,
initial-success suppression and stable once-only consumption. The later
`completion-credit-and-contact-checkpoint.md` records their implementation and
actual prepaid replay, along with supported calculation domains and two Contact
state-goal declarations. Whole-task and release qualification remain open. No
dependency pins, publication, model collection or training changed in this slice.
