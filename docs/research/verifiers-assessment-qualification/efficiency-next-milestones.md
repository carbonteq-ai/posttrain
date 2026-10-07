# Efficiency advice for the remaining assessment and reward milestones

2026-10-03. This is advisory feedback for the two living execution plans, not a
new execution plan or a qualification result. Inspection covered the isolated
environment candidate's calibration models, collectors, route validators and
scorer fingerprint; the first HR slice and broad development review; Posttrain's
assessment archive/export modules; and the migration/tracking gap reports.
No inference, queue changes, source changes, pin updates, publication, commits,
GPU changes or reserved raw-payload inspection were performed.

The critical path is **finish and verify collection → finish development policy
dispositions → implement/test the complete redesign → qualify shared API gaps →
six-environment migration and published adoption → Qwen benchmarks/curricula →
ten fresh 4B attempts per selected development task → stop**. Preparation can
overlap this path. Preparation must not be represented as passing its later gate.

## Concrete findings

- The 48-case HR qualification exercises the actual candidate Task.score and
  native wire reload on three retained executions. It covers three task revisions,
  not HR generally or the complete redesign. Its predicates, effect interpreter
  and native publisher are already separated into hr_rules.py, hr_evidence.py
  and hr_assessments.py; preserve that division when expanding domains.
- The broad review is 51 selected cases on a 445-episode immutable index. Its
  709-task inventory is coverage accounting, not 709 substantive adjudications.
  Process only newly available development sources into a new immutable review
  index, then review missing families and materially different policy/outcome
  branches. Do not repeatedly ask Astra to reread unchanged reviewed examples.
- calibration/models.py deliberately fixes CalibrationManifest.model_id to Luna.
  Both sdk_runner.py and runner.py validate Luna-specific routes/harnesses.
  The existing CLI is a signed-in reference collector, not the proposed generic
  benchmark CLI. Widening these validators risks weakening the live contract.
- calibration/collector.py already supports attempts_per_task, durable occurrence
  identity, immediate slot refill, retained failures and no implicit resubmission.
  Reuse this machinery after qualifying a Qwen binding; avoid another scheduler.
- calibration/verify.py::scorer_fingerprint hashes vendor automationbench Python
  files plus adapter/scoring.py. It omits hr_rules.py, hr_evidence.py,
  hr_assessments.py, native producer/assignment semantics and rule configuration.
  New eligibility needs an explicit accepted-redesign identity covering those
  inputs. Preserve original collection identities and add linked reassessments;
  do not silently reinterpret the current fingerprint as a redesign fingerprint.
- Posttrain retain_episode_artifacts/seal_assessment_artifacts explicitly leave
  external_references_resolved=False. Opaque retained bytes and a local tar.gz
  are not proof of downloadable complete evidence. This is a real Gate 2 gap.

## Safe parallel preparation now

All environment/native implementation below belongs in isolated candidates;
the loaded live sources stay frozen. Suggested ownership assumes explicit
coordination before edits, especially for shared modules.

| Lane | Exact ownership and smallest meaningful checks | Dependency/limit |
| --- | --- | --- |
| Qwen driver scaffold | New environment modules src/automationbench_v1/calibration/benchmark_models.py and benchmark.py; new tests/test_calibration_benchmark.py. Define frozen eligibility/reassessment references, exact model/runtime/renderer binding, common task pool, fixed repeats, logical attempts and injected native inference client. Test rejection of unverified tasks, stale redesign identity, task digest changes, unavailable accounting, extra/missing attempts and preserved failures using fake clients. | No edits to models.py, sdk_runner.py, runner.py, campaign.py or live CLI yet. No serving launch or inference. Fake tests establish schema/orchestration only. Integrate with collector and CLI through one owner after reward/API contracts settle. |
| Curriculum and final-bank schema | New calibration/curriculum.py and trace_bank.py; tests/test_calibration_curriculum.py and test_calibration_trace_bank.py. Use explicitly supplied development benchmark manifests, separate 2B/4B placement, frozen eligible pool, ten logical occurrences including failures, distinct benchmark/bank identities. | Synthetic fixtures only until real Qwen evidence exists. No inferred placement based on Luna alone, hidden task scan, automatic reward tuning or training hook. Coordinate benchmark_models ownership. |
| Tracking archive qualification | packages/train/src/posttrain/train/integrations/verifiers_assessment_artifacts.py and tests/test_verifiers_assessment_artifacts.py; tracking adapter artifact path/tests with separately assigned ownership. First build one native assessment-bearing episode with external artifact bytes and distinctive input markers; verify retained manifest closure locally. | A single later real publish → download → decompress → native reload probe is mandatory. Test publication failure and restart without rescoring. Preserve pending archive state. Local fixture success does not qualify remote storage. |
| Results presentation | apps/observatory/src/posttrain_observatory/models.py, traces.py and service.py; frontend trace result component plus trace-presentation.ts/tests. Start with bounded projected result fixtures from assessment_results.py and zero-child episode cases. | Avoid raw assessment querying. Display valid zero, abstained/failed/unavailable, selected attempts, coverage and archive availability independently of legacy scalar. Summary backend transport must retain these results before real UI acceptance. |
| Native gap inventory and migration fixtures | Native candidate scoring/trace/judge test owners; external environment package-local tests/test_contract.py and existing taskset.py seams, without migration writes yet. Inventory exact input/recipient/alignment requirements and scalar parity fixtures for all six packages. | Full migration waits for redesign/API acceptance. Gather existing dataset/container/network requirements early; do not launch unapproved infrastructure. Combined wheel install and native real panels remain mandatory. |

Trackio generic transport, bounded SQLite/Doris summaries and restart delivery
identity belong in the Trackio fork (trackio/sqlite_storage.py,
trackio/doris_storage.py, trackio/storage.py with their storage unit/integration
tests), not an Observatory workaround reading raw payloads. Resolve its selected
consumer revision before assigning edits: the inspected sibling checkout differs
from the framework-selected revision. Use one fixture matrix across export,
storage and Observatory; keep each owner responsible for its boundary.

## Reward implementation sequence

1. Finish reusable effect coverage and typed policy authority in the isolated
   candidate before multiplying family adapters. Resolve draft→sent transitions,
   changed existing objects, paginated candidate sets and final-state availability.
   Keep known irreversible prefix harm usable when terminal coverage fails.
2. Complete the remaining HR policies alongside simple exact-target/effect rules.
   Add task-specific public-policy adapters rather than treating three predicates
   as a universal classifier. Accomplishments deduplicate by obligation/entity;
   harmful effects retain actual execution occurrence identity.
3. Implement support refund/escalation and marketing approval/hold families next:
   these reuse recipient/effect/authority machinery and expose high-scoring harm,
   duplicate accomplishments and misleading logs. Counterexamples must include
   valid alternative routes, lifted holds, wrong entities, duplicate real writes
   and repair after disclosure. Do not require Luna's particular tool sequence.
4. Add sales/operations eligibility, joins and clocked obligations, then finance
   unit/currency/authorization/ledger families. Resolve task-policy conflicts,
   simulator Paid/Sent abstractions and business-calendar ambiguity before
   negative labels; unavailable interpretation is not a safe pass or numeric zero.
5. Qualify narrow semantic judging only where deterministic content predicates
   demonstrably cannot decide the required property. Keep the policy/effect guard
   authoritative and bounded auxiliary calls optional. Do not build universal
   prose classification before the deterministic families work.

For each slice: pure predicate counterexamples → receipt interpreter cases →
actual Task.score with native finding/assignment serialization → immutable
development replay → independently authored executed counterexamples/new scorer
execution. Replaying prerecorded episodes proves interpretation of retained
evidence; it does not establish new rollout behavior, real model conditioning,
original token alignment or learned improvement.

## Budget and validation choices

Keep the 16,384 Luna output contract fixed for eligibility. The SDK contract is
reported completed-response accounting and an observed threshold, not a physical
provider hard cap. A Luna budget does not automatically transfer to Qwen tokenizers,
reasoning or tool templates. Distinguish initial/maximum response input from
cumulative input; retain cached usage separately. Success-conditioned length
quantiles have survivor bias and one reference attempt cannot establish a tight
task budget. Provisional per-task budgets need floor/headroom, family fallback,
failure/truncation accounting and Qwen transfer evidence before use.

Before Qwen dispatch freeze repeats and total attempt ceiling: for N common tasks
and three repeats/model the benchmark target is 9N logical attempts; the final
bank adds 10M fresh 4B attempts for M selected development tasks. Select a bounded
resource envelope up front, retain incomplete coverage if reached, and report
small-sample uncertainty rather than adapting repeats to successes. Keep timeouts,
infrastructure retries and model failures distinct.

Avoid full source rehash/replay and full workspace tests on every progress poll.
During collection observe append-only journal terminal deltas; hash each immutable
episode once into the review index, then independently verify complete archives
at terminal boundaries. Run touched-module tests per slice; repeat broader suites
only after meaningful integration changes or a failure. Keep the full release
ladder, combined wheel installation, six real native panels, original-token
consumer qualification, real SQLite/Doris result parity and published archive
round trip as explicit unfinished gates. The 83 skipped integrations and
unsupported token projection cannot be converted into completion by rerunning
the same passing offline tests.

Publication order remains native Verifiers commit/push → environment commit/push
with native pin/locks/boundary constant → Posttrain immutable pins/lock/catalogs
and tooling ledgers. Prebuild the adoption change and commands now; do not change
live pins or describe candidate sources as published/reproducible. Consult this
adviser again with proposed ownership before the next implementation milestone.
