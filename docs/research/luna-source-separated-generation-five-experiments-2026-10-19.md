# Five adaptive experiments in source-separated solver generation

Five sequential experiments completed with fresh Qwen3.5-4B and9B on the same
two training-eligible tasks. The final pipeline produced native execution on
all four model/task cases. It is the most complete generation candidate, but
neither successful execution nor the same-model reviewer qualifies an entire
reasoning corpus for SFT.

There were20 control and10 guided native episodes, all scored with full legacy
success and partial credit. All declared goal/guard checks in these observed
episodes were decided and satisfied. Ten possible guided episodes were blocked
by preparation gates; they are not scored failures or zero-filled observations.
Controls already scored fully, so this is not evidence of reward improvement.
Model weights did not change. No SFT export, external model calls, heldout
evaluation or larger corpus generation occurred.

## Experiment progression

Guided counts below mean episodes actually executed, out of two possible tasks
per model. In04/05, reviewer-negative plans could execute as explicitly marked
simulated diagnostics, retaining privacy/structure gates and no SFT approval.
Coverage therefore is not a controlled, constant-admission quality score.

| Experiment | Change learned from previous evidence |4B guided|9B guided| Main finding |
|---|---|---:|---:|---|
|01|Replace hidden source values with stable retrieval handles; clarify check roots and terminal state.|0/2|0/2|Literal-answer copying disappeared in observed translations; malformed dependencies, citations and dropped obligations blocked execution.|
|02|Constrain JSON output types and allowed identities.|0/2|1/2|9B Salesforce executes correctly. Assembly still drops guards;4B reviewer approves a fictional stage-from-email claim. Public quoted-name filtering was too strict.|
|03|Expose argument provenance, distinguish API conventions from observed data, normalize public quoted names and constrain list sizes.|1/2|1/2|9B uses the demonstrated API stage correctly.4B follows a false email-subject stage hint despite full native score; unsupported audit instructions remain.|
|04|Use one authoritative argument representation; retain all requirements alongside the plan; label reviewer-negative native execution diagnostic-only.|1/2|2/2|API stage corrected.4B Salesforce invents an address and is blocked. Opaque handles, extra audit demands and internal join names still enter instructions.|
|05|Replace handles with observed-tool-field descriptions; translate join aliases into ordering constraints; preserve protocol field literals and clarify effect-history scope.|2/2|2/2|No literal/handle gate failures. All four guided episodes execute successfully. Remaining reasoning/plan defects prevent blanket SFT admission.|

Experiment01's Salesforce search phrase `mobile phone` matched both the public
task wording and Luna's reference. It was not a demonstrated invented query.
The demonstrated invented provenance was the deal stage attributed to email
content. This corrects an earlier progress-note interpretation.

## Actual trajectory findings

The strongest sample is experiment05/9B on
`simple.email_sf_contact_mobile_update`. It searches Gmail, uses the returned
sender address to locate the Salesforce contact, updates the returned contact ID
with the number extracted from the email, and re-queries to verify the field.
Its reasoning explains those observations without claiming that a target-record
query proves global absence of other modifications. This is a candidate for
clean-prefix/tokenizer export validation, not an already exported SFT sample.

Experiment05/4B executes that same task correctly, but says that re-querying the
target verifies that no other records changed. That conclusion is unsupported
by the query. Its actual issued write was correctly limited to one record;
those are separate facts. Exclude the unsupported reasoning transition from
training unless an independently validated generation resolves it.

On HubSpot, experiment03/4B used `Partnership Opportunity` from an email subject
as a deal stage because the generated instructions told it to do so. Scoring
accepted it, but the derivation was false.04/05 use the demonstrated API option
`appointmentscheduled`, explicitly distinguished from an email-derived fact.
This demonstrates why task success cannot replace instruction inspection.

Experiment04/4B first queried `Vertex Labs INBOX` and received an empty result,
then recovered with `Vertex Labs`. The native `tool_empty_results` metric was0;
the saved tool response explicitly had `result_count:0`. The audit reports
observed empty/error responses separately, preserving native metrics.

Experiment05 guided HubSpot calls wait for a returned deal before announcing
creation. The fresh9B control sometimes creates and announces in one assistant
batch. This is a useful observation about result-conditioned execution on this
case, not a statistically established gain. The final9B HubSpot reasoning still
briefly conflates reading content with marking an email read.4B still discusses
a global duplicate audit before recognizing its single-creation action history.

## Remaining pipeline limitations

Retaining requirements prevents textual omission, but does not guarantee correct
planning. In05/9B the proposed HubSpot creation step lacked an explicit dependency
on the email read, even though actual native execution read first. Critical
ordering must be checked against a compiled dependency graph, rather than
trusted to another summary or to the actor's usual behavior.

The same-model reviewer has both false positives and false negatives. It can
approve unsupported API context fields, demand hidden identifiers instead of
valid retrieval, or treat one-record verification as a global audit. Its
approval is not a correctness label; its rejection is not automatically proof
that an executable plan is wrong. Raw reviews remain visible in every snapshot.

A CPU counterfactual test changes the hidden phone and record ID and confirms
identical separated initial preparation inputs. Protocol literals such as
`body_plain`, check identities and temporal ordering are retained. This tests
the input transformation on these cases, not arbitrary answer independence or
counterfactual native-world behavior. The paper's entropy/probability anchoring
metrics were not run. No SSR-format teacher was fine-tuned, so this is not an
SSR-D reproduction.

These are two reused Simple-domain training tasks, not an unseen benchmark or
a balanced category sample. The adaptive bundles were selected on these same
tasks. No uncertainty interval or generalization claim is justified here.

## Evidence and next decision

All30 native episode identities, source configs, exact model requests/responses,
stage outputs, complete assessment IDs and container receipts are retained under
`docs/research/evidence/luna-ssr-sft-20261018/adaptive-five/experiment01` through
`experiment05`. `ledger.json` records hypotheses and decisions before launch;
`summary.json` reconciles observed populations without dropping blocked arms.
Source hashes and exact run IDs are in each `launch.json` and
`source-sha256.txt`. Original source traces/contracts remain in `materials.json`.

Both model pins remain unchanged:4B
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,9B
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Each uses BF16,32K context,
4096 reply tokens,16384 episode output tokens,12 turns,2 admitted episodes and
8 serving sequences. Preparation uses the same model, non-thinking greedy
generation, with a2048-token reply cap; native actor reasoning is enabled.
One-hour hard cap per model includes startup and execution. Every accepted
capsule is immutable; no failed attempt was overwritten or resumed.

Use05 as the next engineering baseline and the9B Salesforce trace as a review
example. Before expansion, validate plan dependencies against the original
contracts, separate guard enforcement from global verification claims, and
filter reasoning for support in the actual prefix. Do not automatically rewrite
bad reasoning after seeing the answer and call it a faithful execution trace.
The five-experiment budget is complete; no sixth experiment or scaling is queued.

Reproduce the audit from the worktree root:

    .venv/bin/python scripts/qualification/summarize_luna_solver_campaign.py docs/research/evidence/luna-ssr-sft-20261018/adaptive-five

The interactive comparison is served locally at
http://127.0.0.1:8852/solver-adaptive-experiments.html.

## Follow-up: five further generation-quality experiments

The separate `adaptive-next-five` campaign addresses the new batch01 failures on
three training-eligible tasks: Sales soft matching, Airtable contact creation,
and Airtable find-then-create VIP. It does not pool results with the earlier
two-task campaign. The user authorized five additional adaptive experiments;
each accepted GPU attempt had a one-hour hard ceiling and zero optimizer
updates.01–04 use9B;05 uses4B with fresh same-model controls. Each next intervention
was recorded after reviewing the preceding native evidence. Models, image,
32K context,4096 reply tokens,16384 episode tokens,12 turns and inference
concurrency remain pinned. Exact configs, source hashes, accepted receipts,
native episodes and reviews are under `adaptive-next-five/experiment01`–`05`.

The table shows **success / partial credit** on each task. A dash means blocked
preparation, not a zero score. The9B control is the explicitly reused batch01-r2
control and is outside this campaign's14 native episodes. These are reused
training diagnostics with one episode per task/arm, not an unseen benchmark or
an uncertainty estimate.

| Run | Model | Create contact | Find then create VIP | Sales matching | Executed / intended tasks |
| --- | --- | --- | --- | --- | --- |
| Prior fixed control |9B|1 /1|1 /1|0 /0.6|3 /3|
|01: explicit effect scope |9B|—|—|—|0 /3|
|02: structured output/schema repair |9B|1 /1|0 /0|0 /0|3 /3|
|03: one assembly repair and re-review |9B|1 /1|1 /1|—|2 /3|
|04: typed extraction, normalized critic, full repair |9B|1 /1|1 /1|0 /0|3 /3|
|05: fresh control |4B|1 /1|1 /1|1 /1|3 /3|
|05: repair, neutral guidance, stronger domain gate |4B|1 /1|1 /1|1 /1|3 /3|

01 exposed a variant-name bug introduced in this follow-up: its label bypassed
JSON constraints, causing invalid check references. Its prose corrected the
create/update distinction, but no native episode qualified.02 repaired that
bug and an existing schema mismatch requesting diagnostic entries that the
validator excludes. Diagnostics remain native audit evidence; original goal/
harm pointer indices are preserved.02 nevertheless showed that assembly could
invent an absence precondition after correctly translating requirements. The
critic detected it, but advisory execution still reproduced the failure.

03 added one critic-guided assembly revision followed by re-review, with no
dispatch on unresolved rejection. This restored the actual lookup/create VIP
trajectory without harming the successful creation task. Sales remained blocked:
the requirement summary demanded sender name while the source check required
sender domain. Preserving all requirements had preserved an incorrect one.
Its raw-source critic also supplied a concrete private-domain example to repair.

04 made domain extraction explicit, used observation-bound critic inputs,
screened known embedded private domains and allowed one requirements/assembly
repair pass. Both Simple tasks passed again, but Sales chose a similar-name
decoy despite an approved review. It also invented `nexgen.com` as an example.
This falsifies reviewer approval as a sufficient admission criterion and shows
that screening copied answers alone misses newly invented identities.

Before05,03 was selected as the simpler conservative9B candidate because it
blocked unresolved Sales instead of executing a wrong plan.05 checked that
repair recipe on4B, with neutral actor guidance and a novel bare-domain screen.
The model and rendering changes make this a bundle transfer diagnostic, not a
single-variable causal ablation. All six native episodes passed. On Sales,
guided execution used5 tool-call turns and one empty result, versus11 turns
and three empty results in the control. This is fewer native turns on one task,
not measured end-to-end speedup: preparation incurs extra model calls.
The control already passed all three tasks, so this is not a reward gain.

Reasoning quality still limits SFT admission.03's successful VIP trace cites
private harm/step labels;05's VIP reasoning instead explains the public request
to create a separate record and grounds IDs in actual responses.05 Sales still
claims that the note title must contain the sender domain, an extra constraint
inherited from guidance. It is rejected for SFT despite full native scores.
The ordinary4B Sales control has a useful matching/recovery sequence, but starts
with an unsupported guessed email domain; it is not silently substituted as a
fully clean target. Correct recovery and speculative initial calls need distinct
target-admission decisions.

Two05 Simple episodes were exported as provisional content-reviewed examples
through `posttrain.data.supervised_from_verifiers_jsonl`, preserving five
assistant targets, actual reasoning, calls/results, clean student prefixes and
native episode/trace lineage. Create-contact has a retained minor format-wording
issue: rationale alternates string/object terminology while the actual argument
is a schema-permitted object. Neither example is claimed to prove reasoning
faithfulness. `compiled/admission.json` keeps `training_ready=false`: exact
native token rendering, masks and no-truncation admission remain unqualified.
No student training or balanced-corpus readiness is claimed.

All14 native episode IDs are unique;11 guided and3 fresh controls plus4 blocked
guided task slots account for all18 planned slots. Native score coverage is
complete for executed episodes, with no native errors and no length-finished
calls. The02 Sales max-turn stop is a turn-budget outcome, not a reply truncation.
Transport retry counts are unavailable. Native guard compliance is checked
separately: VIP has a valid compliance view with no offending update subject,
not an assumption that absent raw harm rows establish safety. No heldout,
development or reserve tasks were used, no pins were changed, and no sixth
experiment or SFT job is queued. GPU compute was verified free after05.

The next concrete source repair is visible in the saved workflow inputs:
`object_type="Contact"` is incorrectly treated as a value extracted from the
email subject because of substring matching, while SQL field/operator structure
is erased into generic runtime derivation. API parameter roles should precede
substring matching; query fields/operators should survive literal redaction.
Field-specific constraints must also distinguish note body from note title.
These are identified next changes, not fixes qualified by this completed sweep.

Inspect the complete follow-up at
http://127.0.0.1:8852/solver-next-five-experiments.html. The exported candidates
are `adaptive-next-five/experiment05/compiled/supervised.jsonl`; reviews and
critical rejections remain adjacent in `sft-reviews.json`.
