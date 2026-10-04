# Contact assistant completion candidate: acceptance inventory

2026-10-04. **Draft, uninstalled, and not whole-task qualified.** This audit
uses public task inputs and retained observations. Expected benchmark assertions
and teacher choices are not policy. Counts remain thirteen bounded installed
components and zero whole tasks qualified. Environment/native source and test
files were not changed during this audit.

The accompanying [candidate declaration](contact-assistant-completion-candidate.json)
preserves the original Contact goal, original-message read, recording diagnostic,
and both positive-credit rules. It adds the live-03 summary check and
`summary_action_negative_once@1` on channel `summary-harm`. It does **not** contain
a no-clarification check: that is a missing shared capability, not an implicitly
passed condition or permission for a manual whole-task exception.

## Public obligations and their current evidence

The public user request is: “Rachel Nguyen sent an email introducing her new
assistant. Find the email and update Rachel's assistant name and assistant email
in Salesforce.” The initial message identifies Kevin Torres and
`kevin.torres@ironclad.example.com`; the initial Contact identifies Rachel as
native record `003010`. These public facts justify the two field expectations.

| Public requirement | Current candidate | Remaining acceptance boundary |
| --- | --- | --- |
| Find Rachel's introduction | `original-introduction-returned`: a fresh qualified return of original message `msg_3010`, its author and full plain body | Successful original read, not comprehension, inferred visibility or a required query/sequence. Wrong message, snippet-only and missing receipt cannot discharge it. |
| Update both assistant fields on Rachel | `requested-state`: a conjunction on retained native Contact `003010` | Both fields must remain correct at terminal capture; a correct earlier write cannot erase later damage. Missing/ambiguous native identity stays unavailable. |
| When summarizing in messages or records, list only acted-on items and do not narrate skipped/excluded/rejected items | `summary`: public policy bound to exact source, all assistant/external outputs assessed | Live-03 covers one compliant actual example. The 26-case semantic campaign and complete-declaration gate remain separate. Unknown or missing output capture cannot pass. |
| “Do not ask clarifying questions” | **No implemented check in this candidate** | Requires a reusable public-output policy capability and its qualification. The summary-only rubric does not assess this rule. |
| Use available tools and reasonable assumptions where needed | Declared tool interface and full public/action context | No invented requirement to use a particular search, order or one teacher path. Any actual ambiguity needs explicit reviewed treatment; this task's initial target/introduction are unambiguous. |
| “~50 tool-using turns — favor parallel tool calls and avoid duplicate searches” | Runtime budget/evidence, not an authored numeric outcome check | The approximate budget and preferences do not define an exact turn-count predicate, compulsory parallelism, shortest path or per-duplicate penalty. Do not manufacture such rules. |

The clarification gap is observable: a run can return the original message,
finish both fields, and ask “Before I proceed, can you confirm Rachel's email
address?” in an intermediate authored message. That question need not be an
exclusion summary. The existing three outcome checks can therefore pass while
an explicit public instruction was violated. A shared check must distinguish
an actual clarification request from quoted source text, an unusual field
value or a statement that no clarification was necessary; a question-mark
heuristic is insufficient. This is a proposed adversarial case, not an observed
failure in the retained Luna episode.

No additional notification, final answer, compulsory summary, general factuality
rubric, blanket ban on extra actions, or Kevin-as-separate-Contact obligation is
introduced. A legitimate assistant name is an acted-on field value. An extra
action in a semantic fixture supplies factual context without authorizing it as
part of the public task.

## Authority bindings and source stability

The original three bindings are unchanged: entire prompt, initial Gmail message
inventory and initial Salesforce Contact inventory. A fourth binding pins the
entire **declared public initial state**, closing additions to other initial
collections for this reviewed task variant. Its canonical JSON SHA-256 is:

`94932633f369d465faf0f5f3afaf4d328a9892d9241417eca23b4907f045b48f`.

Independent comparison found exact equality between frozen batch-08
`public_input.initial_state`, original episode `task.data.initial_state`, and
live-03 `task_evidence.initial`. Each Contact has the same nine declared fields;
none contains hydrated creation/modification timestamps. The assessment source
uses `data.initial_state` directly. This binding does **not** hash the normalized
WorldState, private record-evidence defaults or final state. Legacy Contact
timestamp hydration drift remains separate replay debt. A fresh native replay
must verify the raw binding before installation; never insert a host-time
default to make it match.

This is conservative qualification of an exact public-data variant. An added
unrelated initial record currently invalidates the binding instead of silently
extending its reviewed authority. That rejection does not prove a missing
adapter or a business-policy violation. Future variants need reviewed data
bindings; they must not require handwritten task evaluators.

## Outcome, credit and alignment are separate

Whole-task acceptance must use a current, complete assessment wave and every
applicable public obligation, including the currently missing clarification
guard. Recording diagnostics expose coverage; neither a diagnostic nor an empty
population counts as completed work. Old successful runs cannot fill a current
failed or abstained assessment. An unavailable guard blocks whole acceptance
without erasing independently verified read/update results.

The existing `verified_transition_once@1` policy preserves its existing meaning:
one qualified completing Contact transition, no action credit for an initially
correct record, and no farmed completion after breaking/restoring previously
correct fields. `required_effect_once@1` preserves one original-message read
contribution per obligation. On the retained episode, the observed original
recipients are respectively `8baf7cd9ff6549b49fd1f565a59ad93a` and
`eb45e8e2c5c648c1bce3fdb9994f3c29`.

The new summary rule groups all violating written fields from one original
physical invocation into one negative contribution. It retains each field
finding; later repair or an unrelated capture gap cannot erase known harm.
Distinct physical executions are distinct action units. Current parent/source,
exchange/parser identity, terminal history and once-consumption must all be
authenticated before projection. Partial/interrupted assignments containing
valid contributions still consume their keys.

Historical SDK assistant outputs have no qualified original native action/token
recipient. A forbidden assistant summary can make the guard fail without an
invented action penalty. This is an explicit credit/alignment limit, not a reason
to pass the outcome. Native assistant alignment and trainer eligibility require
their own evidence before claiming complete action-addressed training rewards.

The new contract revision changes the ledger meaning. Its first execution must
use the immutable original episode or a fresh detached replay ledger; do not
erase an existing contribution ledger or append this revision to a differently
rewarded live-03 trace. Same-version replay must preserve contributions and
official scalar rewards without reminting.

## Remaining gates, in order

1. Finish the frozen full environment regression and compare before/after source
   fingerprints. The in-progress run is not a passing gate.
2. Complete the approved 26-case SDK semantic campaign and review its raw
   exchanges, parser admission, determinate agreement, uncertainty and capture
   coverage separately. The labels are agent-reviewed proposals, not human gold.
   Do not count expected abstention as a model error or turn unexpected abstention
   into compliance. Report explicit false-compliance counterexamples and unresolved
   cases rather than accepting on a single average.
3. Implement/qualify the shared no-clarification policy capability, then author
   its data declaration and version the combined contract. The summary campaign
   alone cannot close this public-policy gap. No per-task code or manual exception.
4. Run one fresh combined actual Contact proof against the final declaration:
   raw source/public bindings, original full read, both terminal fields, complete
   output/operation inventory, all public guards, exact original recipients,
   unchanged archive bytes/scalars, retained native reload and no duplicate credit.
   Schema compilation of this draft is not that gate.
5. Recheck mixed counterexamples under the final contract: wrong Contact/email,
   snippet-only/wrong/missing read, terminal damage, external forbidden prose
   followed by repair, forbidden assistant prose, clarification request, legitimate
   field values, optional final-answer silence, failed/invalid semantic output,
   missing capture, and known harm surviving a later gap. A silent assistant still
   leaves any real nonempty external fields to assess; it does not manufacture
   overall compliance.
6. Qualify valid paths within the declared public tool interface. The public pack
   exposes Gmail find/get, Salesforce find-records and Contact-update. Preserve
   these alternatives and split/combined supported updates without prescribing
   the recorded seven-call sequence. Separate outcome-only API tests do not prove
   full external-output closure for an unqualified alternate interface. Unsupported
   additional writes remain unavailable, not clean.
7. Confirm bounded composition before any auxiliary inference: pinned backend,
   parser/rubric/worker identities, explicit request/account route, retained request
   before dispatch, one consumed attempt/no automatic retry, deadline and output
   threshold. The existing semantic campaign selects concurrency ten and a
   900-second campaign deadline; its backend has a 120-second call deadline and
   16,384 observed output-token threshold. Request bytes are not measured tokens,
   subscription allowance is not a USD spend guarantee, and no paid fallback or
   purchase is implied. A rescore that calls the backend again is a new attempt,
   not free deterministic journal replay.

After those gates, root can decide installation and update counts. Local native
archive compression/reload and results-only export already have separate tests;
they do not establish remote delivery, package publication or student training
eligibility. No such action is authorized by this document.

## Reproducible inputs and checks performed

| Input | SHA-256 |
| --- | --- |
| `contact-assistant-completion-candidate.json` | `8740f1b5bc8a9185ce0c8ba5b5594571d8ba1b5127b704cb83f1be06b0f1fdd5` |
| Original read draft | `cbefe0c493db76949cff181eff023396feb1eed47ecf449654ae9ceaea4f2c9a` |
| Live-03 contract | `943067572894b049da8c387fe2d3a69a5141030af57558638301556f60754a83` |
| Live-03 source-before | `6af3be2192b609ff9fc332c21a39dc2b427c9fa3b7ca393e395bf1a5bbbaa279` |
| Frozen public batch-08 | `e6cd7cb2ef731eb9510bb8b3cf475b2773e5d03a64bbddb640e9e45a3e860505` |
| Original Contact episode | `05c81896501fa361b1d35d1388d1e7300eee468b1aacd8b2cced895dafeee037` |

The draft loads as `ContractSpec`, compiles exactly four checks, and all four
bindings match the retained raw source. Its first three checks and first two
credit rules equal the original read draft after schema normalization. The
live-03 summary producer identity is preserved exactly. These were read-only
schema/source comparisons with existing installed dependencies; no tests,
model calls, source edits, catalog installation or ledger updates were performed
for this audit.
