# Native execution credit to original call tokens

2026-10-04. Local unpublished candidate. No training, model call, pin change or
new whole-task qualification. The native parent-link checkpoint is the dependency.

## Implemented behavior

`project_subject` and `project_assignment` now align execution recipients through
their retained sampled dispatch. Server invocations use
`resolve_execution_parent`; harness executions use their retained dispatch.
The generated-attempt index selects the existing call projector, not the
emitted-call ordinal. Original node anchors, generation links, exact parser
spans, completion-token digests and sampled masks remain required.

The outer projection retains the execution subject ID and records the mapped
call projection as a member. Physical retries remain distinct contributions,
even when they point at the same tokens. Unlinked historical executions and
missing, joint or unqualified spans remain unsupported. There is no whole-turn
fallback. Domain values and trainer advantages are unchanged.

Selected native and parent receipt prefixes must still exist unchanged in the
current trace. Their original Python fields are validated before comparison,
rejecting copied boolean coordinates that Python would otherwise equate with
integers. Unrelated later history does not strengthen or erase a known prefix;
full trace admission remains separate from this bounded alignment check.

## Qualification and limits

The existing native `tests/v1/test_trace.py` now contains a real
`RolloutSession.handle_tool` ticket fixture with original token coordinates and
a explicitly manufactured qualified parser descriptor. It has an un-emitted
generated attempt 0, emitted attempt 1/call ordinal 0, and two physical server
retry invocations. Harness and both effects map to `[2, 4)` rather than the
whole sampled turn `[1, 5)`. Subject identities and reload are preserved.

This proves coordinate transport and admission under retained exact parser
evidence. It does not qualify production Qwen parser/tokenizer versions or
recover token coordinates from Luna SDK logs. Those remain separate gates.

Final native server/trace/scoring gate: **158 cases passed**, including 13 new
execution alignment cases in the 38-case trace suite. Scoped projector Ruff and
Pyright passed. The independent critic reproduced current receipt coordinate
forgeries, confirmed their rejection after repair and checked exact valid retry
projections. The final prefix-local check preserves earlier alignment even when
an unrelated later receipt is invalid; export correctly rejects that malformed
whole trace before JSON can normalize its fields. The terminal training overlap
qualification is recorded below. Production model/parser qualification remains
open.

After final trace-export changes, the AutomationBench invocation/external-output
compatibility gate was rerun: all 59 cases passed in 3.51 seconds. No new
manifest or whole-task acceptance is implied by that adapter regression.

Frozen SHA-256 values:

- `assessment_projection.py`: `6da9bc806345dc71e0758bee673b6f098a2447fa3d87923af2dd77918e2e57b5`
- `trace.py`: `d522340faa1236529e56876c45804251db33349b50aaeda5dcdb4a6ee293cc1a`
- `tests/v1/test_trace.py`: `a81b3ab3822f8f7527b605966319b1af9edb4fcaa8c71b858b650141f6da3927`

Run from `/home/hammad/projects/verifiers-credit-candidate-20261003`:

    uv run --no-sync pytest tests/v1/test_trace.py tests/v1/test_scoring.py -q --tb=short
    uv run --no-sync ruff check verifiers/v1/assessment_projection.py tests/v1/test_trace.py
    uv run --with pyright pyright verifiers/v1/assessment_projection.py

## Next acceptance

Admission repair, 2026-10-04: strict Python-mode re-admission now covers server
emission, session retention/hooks, trace state writes and harness/server events,
archive serialization and selected execution projection. It rejects copied bytes
before JSON can turn them into strings and copied boolean coordinates before
integer comparisons. Projection also requires `sampled is True`, exact boolean
mask members and exact nonnegative integer token IDs. An independent fresh-source
harness probe initially exposed these type forgeries; the new regression covers
them without relying on a stale source mismatch. Exact original strings and
the valid `[2, 4)` control remain accepted.

The native runtime interpreter used by UV is
`/home/hammad/.cache/uv/environments-v2/verifiers-cp3.13-4b47c758ed127f06/bin/python`.
It is not a candidate-local `.venv`. The consumer's 35-case gate was rerun after
the admission repairs and passed in 1.64 seconds. Earlier SHA values above describe
the previous checkpoint, not this repaired source revision.

Final repaired native server/trace/scoring gate: **175 cases passed**. The
independent critic's 55-case trace gate passes, with Ruff and diff checks clean.
Current repaired source SHA-256 values:

- `assessment_projection.py`: `113fca231ce5f47c355a7fe56437c52e85ad4237c95c7519f7a1639f62deaa8d`
- `trace.py`: `f8928a394387e6591a04cface0f5853cfc3380cf40f6e5ce047275333bd70b9d`
- `session.py`: `3dae3ba3c69dc982d2e98dcfc5de1cfbdc1bc726ca11552efaa95582f69ab3cd`
- `mcp/server.py`: `21a4d7de2f1231e4743cfc044be178a305e19154dec53dc9522b0768859b7a71`
- `tests/v1/test_trace.py`: `e26ff01a495098a03161a98004c3a7383b2c5e3101fef33ddd903ec663d79ffc`

Training overlap consumption is qualified locally: all 35 cases in
`packages/train/tests/test_update_credit.py` passed, including two new actual
native-ticket/physical-retry cases. Native assignment projection and reload feed
the existing `align_native_credit` bridge and `local_token_rewards` consumer.
Default overlap rejects; explicit `sum` combines opposite values while keeping
recipient/contribution IDs, raw values, evidence parents and assignment identity.
The official scalar remains 0.75 in both cases. No consumer production change
was needed. Test SHA-256:
`d9db3def2cf5c2224a603c3cca70a075b9b76559faa735560e130204acfaec6b`.

Consumer command, from the same native candidate directory (using its existing
UV interpreter; no dependency sync):

    task_pythonpath=$(find /home/hammad/projects/rl/packages /home/hammad/projects/rl/apps -maxdepth 2 -type d -name src -printf '%p:')
    PYTHONPATH="$task_pythonpath" uv run --no-sync pytest /home/hammad/projects/rl/packages/train/tests/test_update_credit.py -q

The trailing empty path component preserves imports from the candidate working
directory. The RL interpreter lacks an optional native dependency, so this gate
uses the existing native environment rather than changing either lockfile.

Next resume shared summary-policy coverage and whole-task manifest acceptance.
Thirteen AutomationBench development components remain installed; zero whole
tasks have complete qualified coverage. Other-environment migration,
publication, Qwen benchmarks, curricula and the final 4B trace bank remain open.
