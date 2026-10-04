# HR reward recommendations: preliminary dispositions

2026-10-03. This dispositions review uses the completed HR development bank and
earlier development debugging, while the remaining Luna collection continues.
It does not approve reward weights, implement rewards, qualify Qwen selection,
or substitute for Astra's final whole-bank review. No held-out tasks, runtime
sources, or model calls were used or changed.

The assessment → assignment → alignment architecture is suitable. Accept its
separation of evidence, semantic accomplishment, harm and optimizer coordinates.
Change proposals that collapse task-specific policy, treat corrections as
undoing messages, or use a growing ledger digest as stable occurrence identity.

## Examined evidence

Reviewed Astra's `luna-astra-preliminary-reward-review.md`, the native assessment
API proposal and candidate records, and these frozen HR examples. All attempts
are under `.posttrain/state/verifiers-assessment-qualification/luna-reference-campaign-01/`
in `/home/hammad/projects/rl`. First-partition examples were independently
reviewed for the earlier critic report; this review rechecks policy distinctions
from each partition's retained `inventory.json`.

| Scope | Task | Attempt ID |
| --- | --- | --- |
| Policy and execution | offboarding_automation | `b03c8dc88e1661f65e7150a31c48e55a20e3f372d6ceedd4bb37753bb55ef9cb` |
| Policy and execution | salary_band_audit | `1aff8e04090f24bbe9b6bf37c9e0af171652fa4d92f0b79e4cac8b99d51dc540` |
| Policy and execution | referral_bonus_tracking | `e4c34498de35028e1c3df57ece9ab088b4c68fd14a48783b17572cc62dbc0cdb` |
| Policy and execution | training_compliance | `6a2e7ad2d21769b9facd59047e2a81c9051963f36cbc39e0a4657ecf272fab44` |
| Policy comparison only | referral_bonus_processing | `8f7f64c543c8b292e5196922bb576be02f8cf6b4bbcf89ee70bd77a005dfe03d` |
| Policy comparison only | salary_adjustment_processing | `fb9d8bb53d2de1c992b5ed83f1318cc6b99a184ffc3ef17bb32a3e8484afc797` |
| Policy comparison only | comp_adjustment_batch | `fee353210e79d6facc64a6c5dfa6a91ae77b3b52fa74fc337ef81ab5f0f1bc00` |
| Policy comparison only | i9_verification_tracking | `c9d9572cfad9d234cd72beba0f647c4ab7c8a2b1dd15ed53fc4c3b4422d668e3` |

The NDA, bonus and severance occurrence IDs in Astra's report are inputs to the
next replay test set, not independently re-reviewed executions here. No result
is generalized to all 100 HR tasks solely from the inspected examples.

## Dispositions

| Recommendation | Disposition | Concrete implementation boundary |
| --- | --- | --- |
| Preserve official score alongside redesigned components. | **Accepted.** | Keep original assertions/results immutable. Publish new findings with separate versioned signal IDs and policy sources. |
| Represent independent goal obligations rather than overlapping substring assertions. | **Accepted with qualification.** | Define entity/effect/target/policy keys from independently authored public rules. A delivered request is not an actual payment or access revocation. Goal deduplication is per obligation, not per equal JSON payload. |
| Salary semantic flags should accept the observed correct directional descriptions. | **Changed.** | The salary-audit public policy has no required Over/Under enum. A reviewed direction/entity/value predicate can accept semantically correct flags, including these examples, but a generic substring or an unconditional observed-output exception cannot. Ambiguous prose goes to a bounded judge or remains unavailable. Test misleading expected words and contradictory values. |
| Resolve referral Paid versus Submitted to Payroll semantics. | **Unresolved for tracking; family-wide replacement rejected.** | Astra correctly calls for disposition before scoring this vocabulary. Tracking's public Bonus Policy says submit to payroll and gives no Paid instruction. Processing's `msg_referral_policy_5109` explicitly requires Paid. Decide tracking's public workflow meaning before changing its tracker rule; preserve processing's declared vocabulary. Neither status proves real-world payment. |
| Reuse authority/approval rule machinery across salary tasks. | **Changed to require task-specific facts.** | This is a constraint on generalizing the proposal, not an allegation that Astra prescribed one threshold. `msg_comp_policy_5101` requires Pending VP Approval for adjustments over 15%. `msg_compadj_policy_5132` gives the CFO end-to-end ownership above $15,000 and forbids HR tracker writes/employee notifications. Preserve task-specific threshold units, authority and effects. |
| Give useful policy/data retrieval shaping. | **Accepted as diagnostic first; numeric shaping unresolved.** | Returned relevant facts establish availability, not understanding or necessity. Deduplicate fact revisions, distinguish missing policy from failed query, and measure exploitation before assigning positive weights. Do not gate success on Luna's particular discovery path. |
| Detect transient guards across all actual sends/writes. | **Accepted.** | Check To/CC/BCC, destination, entity and before-state from retained receipts and authored rules. Required uncovered service/effect means unavailable, not pass. Final-world satisfaction alone cannot establish no earlier harmful action. |
| A later correction cannot erase an irreversible notification. | **Accepted.** | Offboarding's initial payroll request included Raj more than 14 days before departure. Preserve that guard finding even after correction. Greg/Diana requests conditioned on legal review do not establish unauthorized payment. |
| Avoid penalizing a corrective message again. | **Changed.** | Link a correction to its prior occurrence and classify remediation separately. Do not grant a blanket exemption: a correction may repeat sensitive disclosure or reach additional unauthorized recipients. Whether it merits mitigation credit or another harm finding needs a policy-specific rule; never reward self-created harm just for correcting it. |
| Add reversal/debt to prevent repeated state progress rewards. | **Changed for first slice.** | Use one terminal accomplishment per obligation initially; record transition chronology and earlier violations separately. This avoids introducing an uncalibrated debt algorithm. Later shaping can use a versioned state machine with explicit reversal behavior and counterexamples. |
| Deduplicate repeated action effects. | **Accepted with qualification.** | Deduplicate accomplishment, not execution identity. Identical sends are distinct delivered occurrences and may be duplicate harm. Read retries remain separate costs; one failed return does not erase later successful completion. |
| Require complete guards, goals and reported budget evidence for Qwen eligibility. | **Accepted.** | Missing rule coverage or unavailable evidence blocks eligibility. A full official score is insufficient; many negative assertions are excluded from its partial-credit denominator. Preserve reported SDK budget proof separately from failed-stream and physical-provider limitations. |
| Add source-bound execution occurrence subjects. | **Accepted with the identity refinement below.** | Do not substitute MCP UUIDs for generated-call ordinals or create assistant nodes. Keep semantic assignment useful with unsupported token alignment. |
| Treat joint/conflicted transitions through grouped recipients. | **Changed.** | A group is not by itself causal proof. Accept jointly attributed credit only when the rule declares joint semantics and evidence supports the group effect; otherwise mark transition attribution unavailable. Never duplicate full credit across concurrent contributors. |
| Universal weighted scalar or shortest-Luna-path efficiency budget. | **Rejected.** | No inspected evidence supports a common weighting, causal credit rule or minimal budget. Base completion and guards must precede efficiency penalties; this milestone collects evidence and designs components. |

## Execution-occurrence identity and prefix safety

Use the existing native envelope, with a narrowly typed execution subject and
recipient rather than an AutomationBench-specific reward transport. The proposed
reference should distinguish:

- **Occurrence key:** execution origin (`harness` versus `tool_server`), trace
  identity and invocation/occurrence ID. This identifies the same local attempt
  across its lifecycle. Same arguments or provider IDs do not establish equality.
- **Evidence binding:** source snapshot ID plus a digest of that occurrence's
  exact retained lifecycle prefix, including event indices, phases, payloads,
  results/errors, state acknowledgements and conflict metadata available there.
- **Visibility:** a declared observation view/cutoff. A dispatch-only reference
  cannot read or cite a later result merely because the live ledger grew.

Do not use the whole growing ledger digest as the stable occurrence key. It
changes when an unrelated action is appended. It can be a snapshot integrity
anchor, while the occurrence-prefix digest binds the selected evidence. Nor
should an unscoped UUID alone certify source or lifecycle evidence.

An expanded snapshot creates a new evidence-bound reference. A resolver may
declare continuity only after comparing the old retained prefix exactly and
checking identity/source lineage; old findings remain bound to their old source.
Reject lifecycle conflicts, cross-trace references, unresolved occurrences and
silent prefix strengthening. Retransmitted identical receipt events are
idempotent; distinct retries are not.

SDK raw completion IDs and reported usage boundaries are legitimate evidence.
They do not establish an occurrence → generated attempt → native sampled node →
original-token map. Projection must return unsupported until that relation is
qualified. No tool receipt count becomes an assistant-turn count. Assessment
truth and semantic assignment must remain usable when projection is unavailable.

## First implementable slice

1. Implement and test the generic occurrence resolver/view builder with synthetic
   native records and the retained HR examples. Preserve current generated-call
   subjects and old journals; add no token support claims.
2. Author deterministic offboarding future-notification and referral finance
   recipient guards. Public policy explicitly supplies these rules. Test the
   offboarding 14/15-day boundary, payroll and public-channel routes, CC/BCC,
   correction-after-send, missing acknowledgements and incomplete coverage.
3. Add policy-specific goal components for the already disclosed NDA send/status
   relationship. Count each eligible employee once, distinguish draft from sent,
   preserve signed employees, and require a successful send before claiming its
   tracker-send status. Retain domain failure separately from transport return.
4. Publish prerequisites as diagnostics and terminal accomplishments as separate
   assessments. Use explicit native credit channels and allocation rules; do not
   yet aggregate weights or introduce progress-loop debt or efficiency penalties.
5. Replay exact retained sources plus independently authored counterexamples.
   Demonstrate a useful read in a zero-score episode and a guard violation in a
   full-score episode without changing the official score. Unavailable alignment
   and evidence must survive save/reload.

This work can proceed independently of the remaining 700 collection attempts.
It must not modify the loaded campaign source. Final task-family acceptance and
Qwen eligibility still require completed-bank review and implementation tests.

## Unresolved contracts to settle explicitly

- Tracking's intended status transition after submitting a referral payout;
  salary-audit semantic flag acceptance and ambiguous text handling.
- I-9 business-day calendar, holidays, start-day counting and deadline boundary.
  `msg_i9_policy_5113` specifies three business days but not a calendar convention.
  Do not reverse-engineer one from expected employee actions; boundary cases
  remain unavailable until independently defined.
- Which corrective disclosures are permitted remediation, and whether mitigation
  receives credit after self-created harm.
- Whether task messages implying direct payment refer only to simulator requests;
  distinguish request, authorization and confirmed execution per tool contract.
- Transient guard coverage across services, concurrent/conflicted writes, and
  partial failed-stream evidence. An execution receipt alone does not prove every
  relevant effect was observed.
- Numeric component weights, prerequisite shaping, duplication/reversal penalties
  and budget profiles. Do not choose these solely to improve the reviewed scores.

These are design decisions and evidence limits, not reasons to restart completed
attempts or pause the independently frozen reference collection.
