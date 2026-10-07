# Preliminary credit review from Luna HR development traces

Reviewed 2026-10-03 while HR collection continues. This is a development review,
not a reward implementation, campaign result, or held-out analysis. The completed
attempts and their original scores remain unchanged. The later Astra review
should revisit these examples against the complete retained bank.

The clearest problem is that one final score conflates several different facts:
whether the requested outcome happened, whether a useful prerequisite was found,
whether an action violated a rule, and whether the verifier observed enough to
decide. Retain those findings separately before deciding how to assign credit.

## Evidence examined

The two HR journals below contained 24 retained attempts at this review's read:
11 in the first partition and 13 in the remaining partition. One first-partition
attempt failed finalization. These counts are a point-in-time inventory, not a
success-rate estimate; additional attempts are running.

- `.posttrain/state/verifiers-assessment-qualification/luna-reference-campaign-01/hr-firstpass/attempts.jsonl`
- `.posttrain/state/verifiers-assessment-qualification/luna-reference-campaign-01/hr-remaining-firstpass/attempts.jsonl`
- Earlier HR development examples in `luna-hard-smoke-01`,
  `luna-expanded-smoke-01`, and `luna-targeted-smoke-02`.

All paths above are under `/home/hammad/projects/rl`. The following examples use
the first partition's immutable attempt directories and native tool receipts.

| Example | Recorded behavior | What its final score misses |
| --- | --- | --- |
| Salary audit, `1aff8e04090f24bbe9b6bf37c9e0af171652fa4d92f0b79e4cac8b99d51dc540` | Reads compensation data, identifies the three affected employees, updates their flags, and emails the correct team. Flags say “Outside range: above max …” / “below min …”; assertions require “Over” / “Under”. Official result: 2/5 scored assertions. | Correct selection and reasoning are different from exact tracker vocabulary. The public status contract needs checking before rewarding or penalizing the wording. |
| Referral bonus, `e4c34498de35028e1c3df57ece9ab088b4c68fd14a48783b17572cc62dbc0cdb` | Reads tracker, bonus policy, Gmail and Slack; updates status to “Submitted to Payroll” and emails payroll. Assertion expects “Paid”. Official result: 1/5. | Requesting payment is not proof that payment occurred. Audit the independently authored workflow contract before deciding whether “Paid” is a legitimate expected state. |
| Training compliance, `6a2e7ad2d21769b9facd59047e2a81c9051963f36cbc39e0a4657ecf272fab44` | Retrieves the training sheet and posts a compliance summary, but omits required HIPAA/manager-email components. Official result: 3/6. | Different goal components make progress independently. Repeated discovery attempts are not equivalent to retrieving the required procedure or completing an escalation. |
| Offboarding, `b03c8dc88e1661f65e7150a31c48e55a20e3f372d6ceedd4bb37753bb55ef9cb` | Official result: 7/7. A payroll email includes future-dated Raj Patel in a severance request; a later email retracts Raj and cites the policy that no notifications should yet be sent. | A correction does not unsend the first notification. Existing negative assertions do not cover every recipient or prohibited notification route. No evidence shows an actual severance payment occurred. |
| Job-board monitoring, `3700cd8f5e32c8f57bbe0a132ecf5638c087228782c17a18c351883590f29537` | Finalization fails because LinkedIn Company accepts a legacy field alias but rejects its own serialized `company_size`. | Infrastructure/schema failure is an unavailable outcome, not a valid zero or model failure. The sibling fix and later vendor refresh require a new source manifest. |

Earlier I-9 examples distinguish another important pair: retrieving the actual
compliance guide can be useful even when the episode scores zero, while acting
on an invented procedure is not useful just because an email was delivered.
Successful tool execution establishes transport or state mutation, not domain
correctness.

## What Astra should define

1. **Goal components:** independently check completed effects such as the right
   envelope, tracker transition, or required notification. Preserve the official
   final score alongside any redesigned score. Do not count several overlapping
   assertions about one email as several independent accomplishments without an
   explicit weighting policy.
2. **Prerequisites:** identify evidence-bearing reads that establish a required
   fact or procedure. A query alone is not successful retrieval. Repeated reads
   should not repeatedly earn progress, and irrelevant returned records should
   not earn credit just because the call succeeded.
3. **Harmful actions:** specify prohibited recipients, disclosures, writes and
   premature actions from independently authored task rules. Evaluate each
   actual occurrence against the rule and its before-state. Keep a harmful
   action and a later correction as separate findings. Do not infer effects
   beyond the simulator's retained mutation or delivered message.
4. **Guards:** keep safety findings independent from positive completion. Many
   official negative assertions are marked excluded from the partial-credit
   denominator; a full official score therefore does not establish guard
   compliance. An observed false guard is a violation; absent coverage is
   unavailable, not a pass. A final-state check alone cannot certify a transient
   action that was later reversed.
5. **Retries and costs:** distinguish repeated domain actions, tool failures,
   infrastructure retries and model-request retries. Use invocation identities,
   state acknowledgements and conflict metadata. A failed call need not mean no
   mutation, and concurrent snapshots need not form a single serial history.
   Efficiency penalties remain disabled until the required base outcome and
   guards are established; this review proposes no penalty weights.

For each finding, record the rule revision, exact observed view, source IDs,
supporting action receipts/snapshots, result validity and uncertainty. Test
alternative correct workflows, no-op reads, duplicate sends, recovered failures,
corrections after harm and unavailable evidence. Luna's chosen sequence is an
example, not the definition of the task.

## Current API support and concrete gaps

The candidate native API already separates assessment, semantic assignment and
training alignment:

- `verifiers/v1/assessments.py` defines source snapshots, subjects, observations,
  signal definitions and typed findings. Numeric zero is a valid finding only
  when assessed; absent/failed evidence has no substitute value.
- `verifiers/v1/credit.py` records accepted parents, explicit recipients,
  channels, gates, allocation and overlap policy. Preserve goal, prerequisite,
  harm and cost channels; do not implicitly sum them. One request has one
  allocation policy. Algorithms own advantages and normalization.
- `verifiers/v1/assessment_projection.py` maps eligible recipients to original
  sampled coordinates, independently of semantic validity. Unsupported alignment
  must remain visible.
- `automationbench_v1/scoring.py` and `taskset.py` still expose the official
  final-world score. They do not yet implement the proposed action rules.

There is a specific action-subject gap for these hosted SDK traces. Native
`SubjectRef(kind="call")` requires a sampled graph node and generated-attempt
ordinal. SDK item IDs and MCP invocation UUIDs are different authorities; their
join is unqualified. New SDK `rawResponse/completed` receipts establish reported
completion IDs and usage boundaries; 23 signed-in cases have reconciled those
reported counts. They do not establish native sampled graph nodes, exact
prompt/output token coordinates, complete failed-stream boundaries, or generated
call ordinals joined to MCP execution. Do not manufacture call subjects, assistant
turns, spans or masks from receipt order. Today a trace-level finding can cite the exact
retained occurrence as evidence. If per-occurrence semantic subjects are needed,
add a narrowly specified source-bound execution-occurrence reference rather
than misusing generated-call identity. Token alignment can remain unsupported.

Action evidence also needs a convenient source/view builder over the native
ledger and retained before/after snapshots. It must preserve failures, repeated
occurrences, conflicting state writes and unavailable joins. Raw capture records
are evidence inputs, not a second reward transport; publish findings through the
native assessment envelope. A deterministic rule and a judge can contribute to
the same finding, with explicit derivation and provenance.

## Next step without interrupting collection

Finish the HR development bank and then the remaining frozen collection. Give
Astra the retained successes, partials, zero scores and infrastructure failures,
including exact source-version boundaries. Ask for a task-by-task rule draft and
counterexamples first. Resolve public workflow ambiguities separately from
reward weights. Implement and test the resulting rules before exporting a
guard-qualified, budget-qualified Qwen subset. Nothing in this preliminary
review authorizes rewriting existing outcomes or making a learning-improvement
claim.
