# Assessment reload: investigation and implementation strategy

Date: 2026-10-05. Status: local runtime implementation validated; publication and consumer adoption remain pending.

## Implementation results

Follow-up scored-archive case: DocuSign's 44,731,994-byte archive reloaded in
84.013 seconds in the worker and 87.142 seconds under root's profiler. A scoped
probe found one validation owner but 161 source misses and 161 view misses:
their exact proof keys accounted for 39,983,605 and 40,300,468 bytes, exceeding
the unchanged 64 MiB budget together. A non-BMP character can widen an entire
Python string to four bytes per codepoint, causing source/view eviction churn.

Large strings in private proof keys now use reversible UTF-8 with surrogate
preservation and the original string type tag. No hash-only shortcut,
normalization, enlarged cache or validation bypass is used. Final ordinary
reload takes 3.891 seconds and reserializes to exactly the original bytes and
SHA `398ed0bf258e280a411433579295b36260db522288838e123047f5b0bb65384a`.
The scoring test module passes, including bounded-cache, Unicode distinction
and tampering controls. Full native v1 validation passes with 83 external or
opt-in skips; scoped Ruff and diff checks pass. Results and
baseline profile are saved as `docusign-scored-reload-compact.json` and
`docusign-scored-reload-baseline.pstats` in the evidence directory. Scoring
runtime improvement has not been measured. An intermediate cache-priority
experiment reached 34.43 seconds but was removed; the final fix preserves the
existing eviction policy.

The native Episode and Trace wrap-validators now keep the existing scope open
through archive restoration and nested validation. Ordinary default loading,
without an external prototype scope, takes **2.076 seconds** for project label
and **0.260 seconds** for calendar conflict. Both match their original baseline
reserialized hashes, and both show one source/view intrinsic miss. Raw results
are `real-sales-default-optimized.json` and `real-zoom-default-optimized.json`
in the evidence directory. The profiler's mode is named `baseline` because it
adds no external scope; these files measure the changed default loader.

The extended native archive integration test passes for Python, JSON and
TypeAdapter inputs, with retained credit, nested owner reuse, exact source/view
verification counts and cleanup after invalid archive rejection. The full
native test command exits successfully: 348 passed and 83 skipped;
credential-dependent and opt-in SDK cases skip. All 118 selected environment
regressions pass in 7.81 seconds
without a test-added scope. Default-loader tampering controls also pass.

Scoped Ruff and diff checks pass. The repository-wide pre-commit command fails
on an existing Markdown heading-level issue and the private package index's
certificate trust during its `uv run --locked` hooks. Those failures are not
reported as passing checks. No new model calls or dependency pins were used.

## Finding

Give native Episode and Trace loading a validation scope lasting through the complete model-validation operation. Reuse the existing bounded, exact-content intrinsic proofs within that scope. Two real saved scored episodes support this as the first optimization:

| Saved Sales episode | Archive size | Assessment batches | Current load | Scoped load | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apply project label | 61.1 MB | 2,736 | 258.08 s | 2.09 s | 123.4× |
| Zoom calendar conflict | 11.3 MB | 60 | 7.26 s | 0.253 s | 28.7× |

For each episode, baseline and prototype produced exactly the same reserialized bytes, checked by SHA-256. Original files were unchanged. Reserialization adds defaults in both paths; output equality does not mean equality with the original raw file.

These are instrumented single measurements of archive loading, not training throughput. The deterministic reduction in repeated work is stronger evidence than the exact wall-time multiplier.

## Root cause

The archive already pools sources and views. Each measured episode has one pooled source and one pooled view. Restoration admits that material and places the admitted models into every assessment batch. Nested Pydantic validators subsequently run again on those models.

Without a validation owner spanning the load, each use repeats canonicalization, JSON decoding, digest checks and execution-ledger verification. The larger episode's source contains about 2.40 million characters and its view input about 2.55 million characters. Revalidating these thousands of times dominates loading even though the wire archive contains one pooled copy.

| Instrumented work | Project label baseline | Project label scoped | Calendar baseline | Calendar scoped |
| --- | ---: | ---: | ---: | ---: |
| Source intrinsic proof misses | 2,737 | 1 | 61 | 1 |
| View intrinsic proof misses | 2,737 | 1 | 61 | 1 |
| Large JSON decodes | 13,685 | 5 | 305 | 5 |
| Large JSON encodes | 13,685 | 5 | 305 | 5 |

JSON decoding and encoding accounted for approximately 215 seconds of the larger baseline's 258 seconds, including small nested values. These are wrapper timings, not a full CPU stack profile. The scoped path still validates each batch's contextual associations.

Peak RSS barely changed: approximately 769 MiB to 766 MiB for project label, and 308 MiB to 306 MiB for calendar conflict. This is primarily a CPU improvement; large episode memory remains a separate concern.

## Native implementation

Checkout: `/home/hammad/projects/verifiers-credit-candidate-20261003`. Its HEAD at investigation completion was `959da6381394942596cb15e0cbe78f4df8594c22`, after an external commit/merge during this investigation. This investigation made no runtime edits or commits.

| File | Proposed role |
| --- | --- |
| `verifiers/v1/episode.py` | Wrap archive restoration and the full model-validation handler in the existing scope. |
| `verifiers/v1/trace.py` | Own standalone Trace validation; borrow the Episode owner when nested. |
| `verifiers/v1/_validation_scope.py` | Reuse existing bounded intrinsic proofs and owner isolation. |
| `verifiers/v1/assessment_archive.py` | Preserve pool admission and conflict checks. |
| `verifiers/v1/assessments.py` | Preserve typed admission and all contextual batch checks. |

Replace the relevant restoration before-validator with a wrap-validator. The intended flow, adapted to each actual validator signature, is:

```python
with validation_scope(borrow=True):
    restored = restore_history(value) if isinstance(value, dict) else value
    return handler(restored)
```

Wrapping only `restore_history` is insufficient: that scope would close before nested field validators execute. The owner must span the full handler. Nested Trace loading should borrow an authorized owner; standalone Trace loading needs its own owner.

The existing scope holds at most 64 entries and 64 MiB of accounted proof data, clears on exit, and restricts reuse by execution owner. Keys include full type-sensitive contents. Never authorize reuse from a supplied digest, snapshot ID or object identity alone: frozen models can still be forged with `model_copy`.

A proof that source contents are intrinsically valid does not prove that a batch uses the right source, sees an allowed prefix, names the right subject or supports a credit assignment. Source/view/run bindings, observation scope, coverage, invocation associations, lifecycle and credit checks must remain intact on every relevant use.

## Validation performed

The prototype opens the existing scope around `WireEpisode.model_validate_json` without changing native source. Both real episodes loaded and matched baseline reserialized outputs.

A valid reduced real archive passed. Nine altered archives failed: noncanonical source/view JSON, wrong source/view digests, wrong snapshot identity, boolean execution count, incorrect view/run source bindings, and a conflicting inline source after pooled admission. Separate controls rejected a forged model copy after a proof hit and prevented an unplanned async child from reusing the owner's proof. Cleanup after errors was also checked.

Four existing environment test files ran with a fresh scope around each test call: `test_native_invocation_source.py`, `test_native_invocation_inventory.py`, `test_manifest_credit.py`, and `test_manifest_record_retained_credit.py`. **118 passed in 6.85 seconds.** This exercises useful provenance and credit regressions, but does not qualify the full native library or an implemented production loading boundary.

Both measured archives contain zero credit assignments. A real retained-credit archive remains an acceptance requirement.

## Selected strategy and acceptance gates

1. **Fix ownership first.** Add native Episode/Trace wrap-validators using the existing borrowing scope. Preserve pooling and every contextual check. No new public knob, global cache or wire version is needed.
2. **Qualify loading paths.** Test JSON and Python inputs, standalone Trace, nested Episode/Trace and supported TypeAdapter use; empty histories and legacy source versions; fresh and borrowed owners; exception/cancellation cleanup and unrelated async tasks. Entries exceeding the proof budget must safely fall back to full validation.
3. **Preserve rejection behavior.** Run the native suite and environment regressions. Include future prefixes, cross-trace references, invalid lifecycle order, linked-parent provenance, view conflicts and forged models. Add a real archive with retained credit. Assert output equality and operation counts rather than fragile wall-time thresholds.
4. **Measure the workflow after the fix.** Separate loading, scoring, rescoring, serialization and memory. Scoring already owns a scope through `task.py`, `assessment_runtime.py` and `credit.py`. The archive result cannot establish training/scoring speedup.
5. **Then address remaining measured costs.** For native execution-prefix views, consider a private per-validation parsed ledger serving all references. Preserve per-reference checks and returned-data ownership. Parse-once source verification is another candidate if first-use cost matters. Do not enlarge caches or change archive format before measuring these smaller remedies.

Concurrency alone is a poor first remedy: it multiplies redundant CPU work and large resident episodes. After the boundary fix, use a bounded work-conserving queue for independent validation jobs, choosing worker count from actual CPU, RSS and throughput measurements. Do not repeatedly load the same episode when its legitimate local validation lifetime can serve multiple required checks.

The large number of batches deserves separate analysis. Distinct obligations or evidence cannot safely be merged merely for speed. This prototype does not establish that aggregation preserves assessment meaning.

## Secondary opportunity

An earlier synthetic 32-reference experiment measured about 12.7× faster resolution with one private parsed ledger and equal results. It lacked sampled nodes and linked-parent provenance, limiting the conclusion.

The large Sales artifact uses `automationbench.manifest_inputs@1`, not the `native_execution_prefix_v1` loop responsible for repeated per-reference resolution. That optimization is therefore independent of this episode's measured bottleneck and should follow representative prefix-view profiling.

## Evidence and reproduction

Scripts and raw JSON results live in `assessment-reload-performance-data/` beside this report. Scripts: `profile_real_load.py`, `rejection_controls.py`, `scoped_regressions.py`. Nested JSON counter fields named `input_bytes` count string lengths: interpret these as characters. Top-level archive sizes are actual bytes.

Local original artifacts:

- `/tmp/automationbench-luna-sales-20261004/scored-roundtrips/sales.apply_project_label.json`
- `/tmp/automationbench-luna-sales-20261004/scored-roundtrips/sales.zoom_calendar_conflict.json`

From `/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`:

```sh
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src:tests uv run --no-project /home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1/.venv/bin/python /home/hammad/projects/rl/docs/research/verifiers-assessment-qualification/reward-candidate/assessment-reload-performance-data/profile_real_load.py /tmp/automationbench-luna-sales-20261004/scored-roundtrips/sales.apply_project_label.json --mode scoped --output /tmp/recheck-scoped.json
```

Use `--mode baseline` for the unscoped comparison; allow several minutes for the large archive. The original artifacts must still exist locally. No new model rollout is required.

Outcome: the main reload bottleneck is fixed locally and verified on two
recorded episodes plus the native suite and selected environment regressions.
Python, JSON and TypeAdapter paths are covered. Publication, installed consumer
adoption, real retained-credit archive performance and broader workflow
throughput qualification remain open. The proposal sections above describe
the reasoning behind the implemented change and the remaining measured-work
strategy; the implementation did not add the secondary ledger optimization.
