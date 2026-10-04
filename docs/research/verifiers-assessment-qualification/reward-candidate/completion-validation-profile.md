# Completion replay: where verification time goes

The measured cost is repeated native source/view verification, rather than the
prepaid calculation or credit rule. This supports testing a narrowly scoped
verification reuse mechanism. It does not establish a training throughput gain.

On 2026-10-04 the actual hash-bound prepaid Luna replay test passed under
`cProfile`. The test scores the original episode, rescores it and reloads its
archive for a third score. Findings, action recipients, original scalar and
episode bytes remain checked. No fresh model execution was used.

Working directory:
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`

```sh
env PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src /usr/bin/time -v .venv/bin/python -m cProfile -o /tmp/prepaid-completion-support.pstats -m pytest tests/test_prepaid_schedule_manifest.py::test_actual_hash_bound_luna_schedule_findings_recipients_reload_and_scalar -q --tb=short
```

Result: one test passed, pytest 34.23 seconds; process wall time 35.22 seconds;
maximum resident set 401,528 KiB (about 392 MiB). This includes profiler overhead
and is not an unprofiled benchmark. The temporary pstats file is machine-local;
the measurements below are retained here.

| Operation | Calls | Measured seconds | Meaning |
|---|---:|---:|---|
| JSON raw decode | 84,221 | 11.584 | Self time |
| JSON iteration/encoding | 394,606 | 11.112 | Self time |
| SourceSnapshot.verify | 888 | 13.832 | Cumulative time |
| validate_execution_refs | 888 | 8.199 | Cumulative time |
| ObservationView.verify | 1,281 | 12.274 | Cumulative time |
| execution prefix derivation | 891 | 4.400 | Cumulative time |
| native content digest | 298,592 | 9.073 | Cumulative time |
| manifest source admission | 87 | 1.383 | Cumulative time |
| retained completion evaluation | 8 | 0.368 | Cumulative time |
| retained row evaluation | 20 | 0.276 | Cumulative time |

Cumulative times overlap and must not be added. Different workloads, including
the earlier dense access-guard replay, cannot serve as matched before/after
measurements.

## Candidate optimization and acceptance

The independent critic recommends a private executor/plan-local admission scope.
Reuse only successful intrinsic source/view proofs for exact immutable JSON
strings plus complete, recursively type-sensitive Python metadata. IDs and
digests locate candidates; they cannot authorize a hit. A copied boolean index
can serialize as integer and compare equal to `1`, so JSON-mode metadata and
ordinary Python equality are insufficient.

Keep all relational checks active: source membership, execution prefixes,
accepted parents, expected inventory, request/source matching and source-access
visibility. Do not cache assessment batch validation or persist proofs. Archive
reload gets an isolated owner scope; child traces do not share parent proofs.
Outside the scope validation remains full. Scope exit, errors and cancellation
must release proofs; bounded eviction changes cost only.

Before implementation acceptance, cover changed body/digest/metadata, copied
bool/float/string coordinates, self-consistent foreign/future views, coherent
forged credit source, failed admission, separate owners/reloads, async isolation
and eviction. Repeat this same actual replay unprofiled before and after, with
identical findings/credit/scalar, then separately measure the dense workload.
No optimization has been implemented or qualified at this checkpoint.
