# Manifest revision and authored-output evidence checkpoint

Local development work, 2026-10-04. No new published commit, dependency pin,
paid judge request, training run or task-catalog entry follows from this work.
The catalog remains thirteen bounded development components and zero whole
tasks qualified. The first 105 public reviews are complete; further library
authoring uses Codex batches of ten with separate review and acceptance queues.

## Consumed credit

The environment's `manifest_credit_history.py` checks all six current manifest
credit families before planning. A valid contribution freezes the episode's
ledger to one canonical manifest digest, including partial contributions from
failed or cancelled attempts. Changed check IDs, channels, source aliases or
policy grouping cannot mint a second contribution under a new manifest. A
changed reward design uses a separate replay ledger; do not erase consumed
history in place. Empty or invalid-only attempts do not freeze the revision.
Legacy fixed-record once credit also consumes the logical obligation when a
different execution is selected on a later attempt.

The 61-case native revision matrix covers all families, rescore/reload, genuine
yield-then-failure/cancellation, renamed checks and fresh unscored replay.

## Authored-output evidence

`authored_output_source.py` projects exact retained SDK bytes and native sampled
assistant message content into admitted task material. `contracts/authored_outputs.py`
rederives authored facts and capture coverage from that material. It excludes
prompt history, reasoning, tool returns and deltas as output facts. Missing,
malformed, conflicting or unfinished stream evidence cannot establish silence.
Known independent text can remain available with incomplete inventory.

The actual Contact assistant episode contains the final assistant message in its
sealed SDK artifact despite having no native message nodes. Episode SHA-256:
`05c81896501fa361b1d35d1388d1e7300eee468b1aacd8b2cced895dafeee037`.
Artifact-envelope SHA-256:
`a585ec5edee63890d73cfae792cec6fb8e24f3410aff36dd2cc00fd8a25eb012`.
Raw SDK artifact SHA-256:
`bb65881e6d7fec76cb38c370902ba67c774e379da4f353d469521d62a7aacd06`.
All original bytes and scalar scores are preserved; the replay restores sealed
artifacts and survives native serialization/reload.

## Verification record

An intermediate combined gate had 127 passes and one failure: the actual
artifact contained producer-emitted startup wrappers absent from the adapter's
synthetic fixtures. Their names and identities were audited against the SDK
collector. The repaired intermediate broad regression passed 1,363 tests in
62.78 seconds. Independent probes subsequently found unsupported wrapped
notifications and terminal item types still opening a false closed-empty path;
final repair and targeted qualification now pass. Root's final combined gate
passes 146 cases in 7.48 seconds; independent critic repeats all 85 output adapter
and source cases in 2.06 seconds, including its reproduced closure failures and
the actual archived artifact. Scoped Ruff/Pyright and diff checks pass. Only the
pure adapter and its tests changed after the broad gate; the final targeted
checks cover that change. The broad count remains an intermediate result.

Frozen source/test SHA-256 values:

| File in the external environment package | SHA-256 |
| --- | --- |
| `src/automationbench_v1/manifest_credit_history.py` | `9aac3c328b78b4354b8d744cdd4542b3f784fd7bf1d640dc1365fe74fc2b0949` |
| `src/automationbench_v1/manifest_assessments.py` | `d18b675c55a2666334dd91a6f9df96007ea68ecad78f83b9b710af72cda64a23` |
| `src/automationbench_v1/authored_output_source.py` | `ecf6cfb14d7b53012754c769d96506fd72b84b185ece231ab71666331796c867` |
| `src/automationbench_v1/contracts/authored_outputs.py` | `5c110997467a85e2bf7561aa42be64f7db021b4803a5651a2b39b7d132959179` |
| `tests/test_manifest_credit_revision.py` | `909ddc7b7a5037d22d75783d9416fc22740f1e3a93545087e6f92863f903cac8` |
| `tests/test_authored_output_source.py` | `32fca614fb9743f1bd6be70b89b02e5a871dcacfbce6c0b7de9a8c1c06637a96` |
| `tests/test_manifest_authored_outputs.py` | `de2b7ce6c0d9a07091594c15bc62ce4483d48ee2d39ec098687129f4a1520e0f` |

Run from the external environment package:

```bash
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_manifest_authored_outputs.py tests/test_authored_output_source.py tests/test_manifest_credit_revision.py -q --tb=short
```

## Remaining boundary

This source does not assess public summary-policy meaning, register a new
manifest guard, align token masks, or close external message/record text.
Conditional summary checks must stay optional where the task does not request a
summary. Ground references in action relations and field values: a summary of
Rachel's assistant update can legitimately mention Kevin. Public policy defines
the semantic question; do not invent a separate factuality rule. Unknown
external operations or incomplete execution capture keep whole-task coverage
unavailable. Existing legacy Contact unused timestamp hydration drift also
remains a reproducibility limitation.
