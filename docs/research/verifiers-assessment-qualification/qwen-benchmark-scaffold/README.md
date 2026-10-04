# Qwen benchmark and Luna task eligibility candidate

This is an isolated implementation candidate, not a benchmark result or a promoted
curriculum. Original live Luna sources, manifests, journals and environments were
not edited. No inference, serving, GPU changes, paid retries, commits, publication
or dependency pin changes occurred in this slice.

## What the candidate does

`calibration/eligibility.py` admits a task only when its original signed-in Luna
reference is bound to the frozen inventory, manifest, committed logical attempt,
native episode and sealed SDK journal. It recomputes fresh response accounting
and requires the original 16,384-token output budget, complete usage including
reported reasoning detail, and output within that budget. A success from a rescue,
retry, unknown accounting or a larger budget cannot establish eligibility.

The accepted redesign registers an exact reviewed task contract. That contract
requires separate whole-trace or whole-episode goal, guard and coverage findings.
Every required finding must come from the declared native producer/rubric/signal
revision, belong to the exact original native source, be available, and pass its
declared criterion. The proof stores an immutable journal terminal prefix so later
append-only collection does not invalidate that occurrence. Revalidation reads
the source closure again; a copied digest or an official scalar score is insufficient.
Hashes establish binding and integrity, not artifact authenticity or scientific
approval. Composition chooses the accepted revision and reviewed contract registry.

`calibration/benchmark_models.py` freezes the same eligible pool for Qwen3.5 9B,
4B and 2B, with three attempts per task per model: exactly nine logical attempts
per selected task. Every binding declares exact weights, tokenizer, template,
renderer, runtime, route and tools policy identities. Sampling and budgets are
shared. An invalid or stale proof, changed source/scorer, outside task, missing
model size or conflated serving alias is rejected before collection.

`calibration/benchmark.py` consumes an injected, already configured async runner.
It immediately refills free slots, creates fresh native task worlds, persists each
start before dispatch, retains native episodes and discarded traces, and consumes
failures, truncations and interruptions without automatic retries. Resuming never
reruns a consumed start. If a process stopped after native episode bytes were
committed but before the result marker, recovery preserves those bytes and records
the occurrence as interrupted. Cancellation drains active attempts into retained
prefixes. The driver does not start inference or select a fallback provider.

## Validation and evidence

The injected fixtures perform no authentication or model requests. Synthetic SDK
labels are deliberately fabricated test inputs. The Luna fixture executes the
real native task scoring/finalization and artifact/journal pipeline, then passes
those sealed bytes through the real eligibility builder. Its contract is a test
policy and carries no promotion authority.

The benchmark fixture demonstrates all three distinct bindings, a common task
selection, three consumed attempts per binding, slot refill while an earlier slow
attempt remains active, retained failure/truncation, success without early stop,
resume without resubmission, native episode crash-window recovery and artifact
tamper rejection. Eligibility tests reject output overshoot, missing response
usage, non-original budgets, harmful or unavailable guards, different native
materials, empty assessments, changed artifact bytes and scorer/revision drift.

Run from
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`
using its independent existing environment:

```bash
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src \
uv run --frozen --no-sync pytest \
  tests/test_calibration_benchmark.py tests/test_calibration_eligibility.py \
  tests/test_calibration_reward_revision.py tests/test_calibration_manifest.py \
  tests/test_calibration_journal.py tests/test_calibration_collector.py \
  tests/test_calibration_inventory.py tests/test_calibration_runner.py \
  tests/test_calibration_campaign.py tests/test_calibration_cost.py -q

PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src \
uv run --frozen --no-sync pyright --pythonpath .venv/bin/python \
  src/automationbench_v1/calibration/benchmark_models.py \
  src/automationbench_v1/calibration/benchmark.py \
  src/automationbench_v1/calibration/eligibility.py \
  tests/test_calibration_eligibility.py tests/test_calibration_benchmark.py
```

See `pytest.log`, `pyright.log`, `ruff.log` and `source-hashes.json` for the precise
qualification boundary. Other agents own the redesign revision and HR policy
implementation; those files were preserved.

`actual-development-reference.json` reads one explicitly named real development
reference, `hr.i9_verification_tracking`. Its sealed fresh SDK accounting qualifies
the original 16K budget. Current candidate official replay rejects it because the
historical scorer identity differs. This is an expected fail-closed result, not
admission of the task. No accepted closed redesign reassessment was supplied.

## Remaining gates

1. Independently qualify a frozen historical official-verification closure. Current
   `verify_retained` requires exact current scorer identity and cannot approve
   legitimate historical-source drift. Do not normalize arbitrary lists or ignore
   retained verification. The Mailchimp unordered-set issue remains a separate,
   versioned replay concern.
2. Supply and accept actual closed task policies plus deterministic and/or judge
   native findings under the current redesign. Successful Luna actions do not
   automatically become required actions. Gaps remain in multi-task reward design.
3. Bind actual immutable Qwen weight/tokenizer/template/runtime identities and
   qualify the real supplied inference clients, tool parser, loaded models,
   budgets, complete accounting, cancellation and callback behavior. A matching
   runner declaration is insufficient. The native adapter relies on composition
   for enforcement and client lifecycle. Missing native usage stays unavailable.
4. Run the actual three-model comparison on the admitted pool. Every result in
   this schema is explicitly `scaffold_only`; there are no real Qwen results here.
5. Use those results to propose 2B/4B curricula, then retain the requested ten
   4B rollouts per selected task as the next slice. Training and further reward
   iteration are outside this implementation slice.

Execution-occurrence recipients remain semantic evidence. Hosted SDK receipts do
not supply a qualified occurrence-to-generated-token alignment. This candidate
does not invent token masks or change the native unsupported projection result.
