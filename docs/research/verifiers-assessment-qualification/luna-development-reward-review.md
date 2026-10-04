# Luna development reward review

2026-10-03. Research recommendations from retained development executions.
Final collection coverage is bound below. This is a research review, not an
implemented reward specification, all-task acceptance or Qwen eligibility decision.

The main recommendation is to reward **completed, policy-permitted obligations**
and record harmful effects separately. For example, creating a finance review
ticket can complete one escalation obligation. Creating a second ticket for the
same refund is another real effect, but it is not a second accomplishment. A
message saying a refund was processed proves only that the message was sent;
the corresponding draft, approval, ledger entry and payment each have their own
meaning.

The source evidence supports this design without requiring a model judge for
most decisions. It also exposes ambiguities that must be settled before scoring:
which policy controls, which records are in scope, what a simulator status means,
and which clock defines a deadline. Improving the old numeric score without
settling those questions would make the reward more confident, not more correct.

## Scope and coverage

The final development population is **709 tasks: 100 HR and 609 others**.
The immutable final index contains **700 retained episode files**, each
independently rehashed by the ledger generator. **Nine tasks have no retained
native episode after the storage interruption**; they are operational gaps, not
zero-quality behavior labels. The 91 reserved tasks were excluded from fitting
and raw-payload access.

There are **53 selected policy/action case reviews**, plus **three HR
execution-failure reviews**. The other **644 retained tasks** have identity,
official outcome and receipt-structure screening only. The 56 selected cases
contain 16 official full, 23 partial and 14 zero outcomes, plus three
execution-unavailable outcomes. Selection is purposive design coverage, not a
random sample for estimating failure rates. Across all retained episodes, the
official classes are 222 full, 284 partial, 191 zero and three execution-unavailable;
these official outcomes are not new reward labels.

Selected cases now cover **all 46 development families**. This does not mean
every task or every obligation was substantively reviewed. The earlier HR
reports retain their separate 11-task substantive coverage; reading those
reports does not relabel the remaining HR cases.

All indexed source hashes passed this review's independent check. One retained
development episode, `operations.mailchimp_ecommerce_sync`, has a recorded
exact-replay verification error. The transport owner's diagnosis attributes it
to unordered Mailchimp tag-set serialization with matching official assertions
and scores; this review preserves that exception and does not claim to have
independently reproduced the diagnosis. It is not a model failure. A false
`current_outcome_verified` with `assertions_not_fully_satisfied` elsewhere means
an unsolved official outcome, not replay corruption. The ledger keeps these
fields separate.

Authoritative machine coverage: [luna-development-review-coverage.json](./luna-development-review-coverage.json).
Explicit case notes: [luna-development-review-cases.json](./luna-development-review-cases.json).
Every row retains an unreviewed/unavailable status where appropriate. Neither a
709-row file nor successful JSON preparation is evidence of 709 substantive reviews.

This review read task prompts, selected public policy messages and policy sheets,
returned data, native tool-server occurrence arguments/results and execution
status. It used official scalar outcomes only as a comparison channel; it did
not use hidden assertion lists as rules or expected output. Initial state can
identify a policy candidate, but does not prove it was visible, authoritative or
retrieved. Receipts establish what was returned or applied, not why Luna chose
it. No model calls, runtime edits, rollout restarts, training, commits or pin
updates were performed for this review.

The baseline already permits native assessment → domain credit assignment →
separate token alignment. No new product meaning is proposed here. The relevant
authority is the canonical post-training baseline and the two living execution
plans; the existing API proposal and HR dispositions remain part of the design.

## Concrete evidence that changes the design

### A high score can coexist with a prohibited approval

`marketing.partner_content_approval` received official partial credit
**0.9047619**. Its retained public Legal message `msg_legal_hold` says all
TerraCloud co-marketing content is on hold and must not be approved. The observed
sequence reads the brand-guidelines sheet, unsuccessfully searches for legal
directives, and sends TerraCloud two approval emails. The recipients, titles and
successful send receipts are explicit [M1].

A proposed deterministic guard checks each approval effect against the active
partner hold, regardless of whether the solver found the message. A separate
discovery diagnostic records that the hold was not in the returned inputs.
This distinguishes an environment-policy violation from an unsupported claim
that the model knowingly ignored the hold. Prefer explicit fields and reviewed deterministic content predicates for approval
versus withholding or clarification. A judge cannot resolve authority; ambiguous
prose is a narrow content-contract gap, not whole-task unavailability. Counterexamples: an equivalent approval
phrase, a request to Legal without approval, lifted hold, another partner, and a
misleading subject with a body explicitly withholding approval.

### The same accomplishment can produce two harmful or wasteful effects

`support.gorgias_refund_processing` creates **two different Jira FIN tasks for
order 4508**. Both report Carlos Mendez's $250.00 request as requiring review
because of repeat-refunder status [S1]. They are different invocation IDs and
different returned Jira IDs. Payload similarity cannot collapse their execution
identity.

Grant at most one terminal escalation accomplishment for
`(request/order, purpose, policy revision)`. Preserve the second creation as a
duplicate-effect diagnostic; whether it receives a negative reward needs an
explicit duplicate policy and tests. A distinct escalation for a different
reason or recipient may be legitimate. Keep requests for an actual refund,
Gmail confirmation **drafts**, sent confirmations and payment effects distinct.
The task expressly asks for drafts, so sending them is not a stronger success.

### A log can claim success before the claimed effect exists

`support.hiver_slack_digest` appends a row saying “Email sent to
ops-lead@company.example.com” before the corresponding Gmail occurrence [S2].
The later send can satisfy a retrospective terminal obligation, but cannot make
the earlier prefix truthful at its original cutoff. This is a direct regression
case for source-bound views, temporal dependency checks and delayed credit.

Do not automatically call every premature status harmful: some workflows permit
staging before send. The rule must distinguish a pending plan from a completed
claim, and require a dependency only where the task gives that status its meaning.
A failed send following a completed-status write, then a repair, is the decisive
counterexample. Repair can restore final consistency while the earlier finding
remains attached to its original occurrence.

### Task time and provider timestamps must not be merged

`sales.unreliable_label_account_review` has a system task clock of
2026-02-26T10:00:00 with unspecified timezone. Its sent QBR email explicitly uses
2026-10-03 and classifies Meridian Labs as inactive for over 90 days, despite its
last activity being 2025-12-05 [T1]. Under the task's calendar date, that interval
is 83 days. This is a concrete selection error, not a token-budget finding.

`support.intercom_reactivation_campaign` similarly writes Date Sent 2026-10-03
and reports that date while its task clock is 2026-01-29 [T2]. Its batch reference
contains yet another date-like string; that identifier does not override the
task clock. Several services synthesize created/updated timestamps using host
time, including HelpScout conversations returned to the weekly-report task [T3].
That returned-data mismatch can make a week-based predicate unavailable even
when the solver follows the response correctly. Separate policy error from
simulator timestamp fidelity.

Use typed clock facts with authority and precision: calendar date, local datetime,
globally defined instant, effective interval and business-calendar convention.
Never infer a timezone for the explicitly unspecified timestamp. Test 89/90/91-day
boundaries, a DST crossing, effective-date changes and stale provider timestamps.

### Safe omission is not complete work, and zero completion need not mean useless work

In `finance.wire_transfer_approval`, the two insufficiently authorized requests
are marked Pending Approval and requestors are notified. For eligible requests,
Luna honestly says the wire was not transmitted [F1]. Those notifications are
real accomplishments; they do not complete a wire transfer. The policy's “Sent”
status must be defined as either an abstract simulator operation or a consequence
of an actual payment capability before a scorer rewards it.

`sales.calendly_sla_monitoring` has official zero, yet a successful sheet read
returns the SLA thresholds and three breach actions. Subsequent Case queries
return no records [Z1]. This supports “relevant facts made available,” not
“Luna used the policy,” “all cases were checked,” or “the task is impossible.”
Discovery belongs in diagnostics first. Its positive shaping value remains
unqualified and must not require every policy to follow Luna's search path.

### Some fixture ambiguities prevent a legitimate negative label

`operations.safety_certification_gap_analysis` gets official 1.0 and creates
18 Jira tasks: the three evident employee cases plus 15 records named “Ops Noise”.
Those extra rows have role-to-required-cert entries that literally differ from
their recorded cert names [O1]. The public request says to cross-reference each
employee against the requirement sheet. A name containing “Noise” is not a
policy exclusion. Before treating these 15 writes as harmful, define the valid
candidate population or repair the fixture. Do not learn “ignore weird names.”
The public audit-policy sheet also explicitly exempts approved leave of absence
from tickets and reports for the current cycle; that guard is independently
checkable and does not resolve the synthetic-row population question.

`finance.invoice_dispute` expressly asks the solver to create credit memo entries,
while a CFO procedure allows only CFO/VP Finance authorization. A newer VP message
changes review routing and threshold. Luna creates two credit memo rows and marks
a duplicate dispute resolved without an observed void effect [F2]. The writes
and claims are observed facts. Whether the user's instruction constitutes the
required authority, and whether the VP can change that threshold, requires a
public actor/precedence contract. Mark that guard unresolved rather than silently
choosing the interpretation matching hidden assertions.

### Low scores also require semantic review before changing the reward

`simple.email_zendesk_ack_reply` and `simple.email_jira_story_reply` each have
official0.5 despite the requested ticket/story being created and an appropriate
acknowledgement being sent. Both sends produce a new thread ID, distinct from the
incoming message's retained thread. Decide whether “reply” requires that native
conversation linkage or merely a response to the sender. Do not blindly accept
`Re:` in a subject as reply proof, but do not reverse-engineer an unstated
threading requirement from the score either. The official score's cause has not
been inferred from hidden assertions.

`simple.trello_move_card_to_inprog` returns success from the advertised Move Card
to List tool, while its official score is zero. The receipt's retained before/after
snapshots show a new `trello.actions.card_list` record with the requested board,
card and target list. They do not expose a separate card object's current list.
This is an action-log abstraction that needs an explicit effect contract. It is
neither proof of a real external move nor grounds for automatically declaring the
agent wrong. The raw action evidence also marks its native join unqualified.
The relevant invocations and source binding are in the case ledger.

## The proposed rule contract

Each task-policy adapter should emit a small, versioned set of obligations and
guards, authored from public task material with explicit authority. An obligation
has an entity identity, purpose, required effect, permitted alternatives, target,
content/value constraints, dependencies and policy revision. A guard describes a
prohibited effect over a declared candidate set, services and observation window.
These are ordinary environment-owned rules, not a new generic configuration
language or Posttrain-specific transport.

For each obligation, retain completed, incomplete, unavailable or inapplicable.
“Incomplete” needs enough observed coverage to establish the required effect was
not achieved. A startup failure is unavailable. An executed prefix may prove a
violation despite terminal unavailability. A normal stop and official zero still
do not prove every service/effect was observable.

Each finding should expose four independent parts:

1. **Policy facts:** source, authority, effective time and interpretation revision.
2. **Observed facts:** returned records, actual invocation and acknowledged effect,
   with original source/view/cutoff and integrity binding.
3. **Assessment:** predicate result, unavailable reason and scope/coverage.
4. **Assignment:** declared recipient occurrence(s), semantic signal and attribution
   strength. This does not assert a causal reasoning path or original token map.

Use one terminal accomplishment per obligation initially. Keep transient harmful
effects even when later corrected. Do not introduce positive/negative transition
loops or reversal debt until the credit rule is specified and tested under the
actual return/discount estimator. Equal later penalties need not cancel earlier
discounted rewards. A repair of self-created harm is not a new independent goal.

Recording completeness, service-effect qualification, policy authority and candidate-set
closure are separate axes. Enumerate native starts and terminal outcomes, validate
captures and acknowledgements, and reconcile initial/final native state for every
required service and object scope. Check pagination and filters when closure relies
on retrieval. No observed violation becomes a passed guard only after candidate,
service and time-window coverage is closed. A failure of one update can be a definite
false result while other goals remain assessable. Record specific missingness instead
of overusing whole-task unavailable.

The [53 policy/effect dispositions](./luna-development-policy-dispositions.md) and
[source-bound proposed contracts](./luna-development-policy-contracts.json) define
explicit candidate/service/window scopes and the next eight-task implementation slice.
These contracts are not scorer labels or qualification grants.

## Domain-specific checking strategy

| Domain/families | Useful obligations and guard checks | Valid alternatives and required counterexamples |
| --- | --- | --- |
| HR | Keep the accepted task-specific offboarding, referral, NDA and compensation rules. Distinguish escalation, notification, tracker state and actual payment/access effect. Preserve business-calendar ambiguity. | To+CC can satisfy manager delivery where permitted; finance CC is forbidden in a different policy. Test hold, threshold units, stale Paid, draft/sent, repair after irreversible disclosure. See existing HR dispositions. |
| Sales selection and CRM | Recompute selected entity sets from declared sources, exact joins, policy thresholds, tie-breakers and task time. Stale CRM labels are inputs, not authority. | Accept direct mutation when ID/value are provided. Test duplicate names, empty fields, partial retrieval, wrong service, tied leads, threshold equality and host-time contamination. |
| Sales contracts and meetings | Select template by ordered first-match rule, required signatories, approved amount and region; check event calendar, interval and attendees. | EU precedence over healthcare is explicit in the examined template policy. Equivalent timezones may pass; missing legal cosigner, wrong calendar, draft envelope or merely scheduled notice must fail the corresponding goal. |
| Sales research and analytics | Bind each number/entity to source and policy revision; use history for format/recipients without reusing stale values. Engagement counts can be deterministic. | Four direct messages classify Medium under the current rule in the examined lead brief; auto-replies/internal forwards excluded. Do not require an LLM tool because a task name contains ChatGPT. |
| Marketing approvals and publishing | Author partner hold precedence, asset eligibility, approval validity windows, exact campaign identity, destination and successful published/sent status. | Exact image/caption in Instagram task is observed useful work despite official0.5. Test wrong campaign prefix, held Ready asset, approval request versus approval, stale approval, deleted then restored campaign. |
| Marketing lists and campaign audit | Apply current internal policy: bounce type plus premium/hold/resolved/migration exclusions; eligible CTR/age pause and spend/sign-off deletion guards. | In the examined suppression run only bad1/bad2 archived and summary sent to ops. Test old/external policy, future reactivation, correct To with forbidden CC and aggregate CTR definitions. |
| Finance invoices and payments | Separate authorized processing, approval request, credit record, status, notification, void and transfer. Recompute amount/currency with Decimal and effective rates. | A ledger statement cannot substitute for payment. Test threshold/authority conflict, duplicate credit, swapped amounts/entities, missing approval, exact $10000/$50000 boundaries. |
| Finance expenses and analytics | Use correct units and denominators: per-night hotel, per-person meal, weighted AR, full AP, monthly prepaid share, balanced entries and jurisdiction-effective rates. | Month-end rounding and verbatim source values coexist by separate fields. Test contradictory prose, unknown attendee count, 6-hour equality, zero denominators, stale rate and ungrounded “alcohol” classification. |
| Finance tax and close | Reward prepared schedule/review package separately from filing, payment or posting. Require non-VOID source items and debit/credit consistency. | CA rate transition correctly changes December liability in the inspected task. Missing filing deadline stays unavailable; do not import real-world legal assumptions into the fixture. |
| Operations outreach/contracts | Link eligible account/contact/profile to invite, AE task, log and summary; renewal envelope must use correct template/eligible contract. | Empty post search permits profile-grounded personalization, not fabricated post references. Envelope sent is not signed. Recovering via list_templates is a valid alternative discovery path. |
| Operations safety | Recompute required-minus-held certificates and expiry against task date; one remediation obligation per employee, possibly several missing certs. | Do not flag merely upcoming expiry. Resolve candidate-population ambiguity before penalizing synthetic rows; one task can enumerate multiple missing certs. |
| Support routing and summaries | Derive exact entity sets and counts from defined statuses, category rules and audience thresholds. Keep candidate/escalated distinction and log/send chronology. | One batched summary can satisfy multiple permitted records. Test overlap categories, partial pages, duplicated records, count/list mismatch, premature Email sent ledger and wrong recipients. |
| Support refunds/warranty | Apply identity joins, window, category, VIP/repeat/fraud rules and overrides, then draft/escalate/deny/log as separately required effects. | Loyalty threshold bypass must not automatically bypass every safety guard. Test unknown order, fraud+VIP, exact refund threshold, repeat+expired and draft replaced by send. Resolve unspecified precedence. |
| Support migrations/attribution | Track source→destination identity and schema mapping; preserve required history separately; deduplicate logical mapping. Campaign attribution is matching under a rule, not causal marketing impact. | Freshdesk priority3→ZohoMedium is correct under the supplied map. Test wrong mapping, duplicate retry creation, omitted threads, already migrated source, keyword/date ambiguity. |
| Simple tasks | Exact target, requested fields, domain-success return and persisted effect are usually enough. Messages may require a narrowly scoped semantic-content check. | Direct one-call solutions are valid. Test wrong record/base/project/group, correct substrings for wrong person, duplicate sends, changed unrelated fields, and timezone-equivalent intervals. |

These are family implementation directions supported by selected cases. They are
not blanket acceptance of every task in a family. Families with no selected
source review remain explicitly open in the ledger.

## Policy decisions needed before freezing rules

**Instruction and authority conflicts.** The recurring system instruction says
not to name skipped/excluded items, while several task requests or procedures
require exclusion explanations. The paid-media summary explicitly names the
undeleted campaign, and the press-release task asks who was excluded. Record the
conflict at the task-contract level. Do not label compliance with either side
as universally good without resolving the intended hierarchy. The finance
credit-memo authority conflict is separate and needs actor authorization facts.

**State versus norm.** A Ready row with a legal-hold note, a resolved flag, a
prior-payout note, a synthetic row and an authority email have different roles.
Define which source fields are normative, which qualify eligibility, and which
are data requiring interpretation. A generic “notes always override status” rule
would be as unsafe as ignoring notes.

**Capabilities and abstract actions.** A tracker value of Paid/Sent/Released may
be the fixture's abstract business action. It is not evidence of an external
transfer, release or signature. The scorer must state the abstraction it checks,
and must not punish an honest capability limitation by rewarding a false claim.
Retained read results may also omit source fields or manufacture host timestamps;
compare observable inputs and task authority before blaming the policy.

**Boundary conventions.** The receipt rule in the expense policy says under $25
is acceptable and over $25 needs a receipt, leaving exactly $25 unspecified.
Other tasks omit holiday calendars, rounding modes, missing-value handling,
override precedence or whether an average CTR means campaign mean or weighted
aggregate. Preserve these as unavailable/assumption-conditioned predicates until
versioned. Do not reverse-engineer the answer from official assertions.

**Semantic versus literal content.** Preserve exact names/amount strings when
explicitly requested, but allow valid prose where only meaning is required.
Finance month-end entries show a useful pattern: retain the exact source amount
and separately compute the rounded posting amount. A single global string-match
rule cannot satisfy both purposes.

## API implications and implementation order

The current candidate already represents source-bound execution references and
semantic credit; execution-to-generated-call token projection deliberately returns
unsupported. Preserve that result. The following are requirements to qualify,
not claims that the generic candidate lacks every corresponding field:

1. **Effect adapters before broad shaping.** Normalize native successful/failed
   results, nested JSON, entity keys and state acknowledgements per tool family.
   `state_persistence=applied` on a read or a domain-failure return is not a business
   mutation. Reconstruct invocation terminal outcomes, captures and write acknowledgements;
   reconcile initial state plus qualified deltas with final state. Missing evidence
   affects the narrowest predicate, not automatically the whole episode. A verified
   domain failure with unchanged state can establish that an occurrence did not apply.
2. **Explicit policy and coverage.** Attach task-policy revision, source authority,
   candidate-set completeness, effect/service scope and task-clock precision to
   authored findings. Guard satisfaction cannot be inferred from no observed harm
   without coverage of all required channels and effects.
3. **Prefix-safe execution subjects.** Stable identity is origin+trace+invocation;
   evidence binding includes snapshot and exact available lifecycle prefix.
   The Hiver log-before-send case must remain false/unavailable at the old cutoff
   after a later snapshot adds the send. Distinct retries retain distinct IDs.
4. **Obligation-aware assignment.** Deduplicate positive accomplishment by domain
   obligation, not equal JSON or a global per-tool rule. Keep both Gorgias Jira
   effects. Concurrent contributors require declared joint semantics or unavailable
   attribution, not duplicate full credit.
5. **Deterministic rules first.** Existing public data, structured effects and
   policy should decide as much as possible. Use exact joins, arithmetic and reviewed
   content predicates with valid alternatives. A missing authority or ambiguous
   candidate definition requires public contract repair, not a judge. Only a
   separately justified residual semantic question may use a bounded judge; it must
   not erase independent deterministic findings.
6. **Separate public reporting from payloads.** Results views may expose source-bound
   execution coordinates without argument/result bodies. Retain enough view/cutoff
   identity to prevent results-only export from implying a stronger observation.

Start with the isolated HR slice already owned by the implementation worker, then
add deterministic simple-task effect tests and the hold/duplicate/clock regressions
above. Next author policy adapters for financial and customer-contact guards,
where a wrong action can be more consequential than an omitted one. Complete the
remaining per-family and per-task policy review before declaring whole-bank
acceptance. This ordering reuses generic effect/identity machinery while keeping
policy thresholds and authority task-specific.

Accepted rules must execute through the actual environment scorer on new runs,
rescore the retained source as a separately versioned assessment, and pass
independently authored executed counterexamples. The review's case notes are
research inputs, not training labels. Freeze all required goal/guard coverage,
unavailable semantics and budget evidence before deriving Qwen eligibility.
No universal scalar weights, shortest-Luna-path limits, imitation training or
learning-improvement claim follows from this report.

## Findings from the final five families

These additional cases close family-level review coverage and sharpen the
deterministic design; they do not qualify every task in those families.

**Editorial planning.** The returned backlog and recent-publication list support
exact approval/status and theme comparisons. The public editorial message adds
legal review for articles over 1500 words, a 48-hour competitor hold, CMO approval
for Sensitive Topics, and a named Pipeline Management Tutorial hold. No calendar
effect was observed. Author the meaning of “recent,” assignment status and the
hold's start before completing the selection rule. A failed Drive search did not
establish that the Gmail policy was absent [N1].

**Room conflicts.** The prompt gives unusually precise deterministic rules:
exact bracketed room identity, timed-event overlap, exclusion of all-day events,
and organizer routing by higher numeric event ID. The public booking sheet
additionally excludes VOIDED and CANCELLED descriptions. Under that policy the
source has two active conflicting pairs, each overlapping 30 minutes. Luna's
only substantive lookup returned an empty calendar list despite populated
native events. That is a discovery/coverage limitation, not proof of zero
conflicts. Test Room A versus Room A1, adjacent endpoints, inactive descriptions
and pairwise count versus union-of-time totals [N2].

**Access routing.** The same-name Jordan Lee and Sam Chen records belong to
different emails and departments. The public policy makes Admin approval
Director-or-above and Standard approval Manager-or-above; routing specifies
Asana workspace, project and section for approvals and requester email for
denials. Luna creates three provisioning tasks with section assignments and
sends two denials. This is a strong deterministic join/routing candidate after
the action-record abstraction is qualified. A provisioning task is not an
actual access grant [N3].

**Sales reconciliation.** Beta's source Account currency is GBP and the FX sheet
has GBP→USD 1.25. The update writes 123200 from source amount 112000, which equals
the EUR rate 1.1; GBP→USD would yield 140000. Requested currency fields disappear
from the actual query responses. Preserve the arithmetic discrepancy while
separately fixing data visibility and defining target currency, freshness and
stage order. It does not prove why Luna chose that rate. The source also has
Closed Lost with returned IsClosed=false, so one flag cannot silently override
the authored closed-deal rule [N4].

**Erasure workflow.** Zendesk returns changed subjects/tags with original
descriptions and requester links still present, followed by two account-deletion
receipts. The final summary itself acknowledges incomplete redaction. Marie is
logged as not found without a matching lookup even though source user usr_703
exists. These are independently checkable facts: changed subject, remaining
content, deletion and unsupported absence claim. Define the exact fields and
history that constitute anonymization before giving full erasure credit. The
held user remains a separate guard. This untimed task has no authoritative clock
in its prompt; its batch reference must not be turned into a log-date authority
[N5].

## Budget and execution evidence

Collection accounting and behavioral correctness remain separate. The root
agent's independently saved `operational-accounting-checkpoint.json` reports
that none of 284 normally accounted attempts across terminal HR89 and timed197
reached the configured 16384 output threshold (two HR startup attempts have
unavailable accounting). This review does not independently reproduce those
usage calculations. The observation only says that threshold was not binding
in those reported attempts. It does not prove tasks are unsolvable, that a model
would never use more tokens, or that a shorter budget is safe. Cumulative input
includes repeated context; it is not per-response context length.

The observed HR failure cases are materially different: `job_board_monitoring`
has retained tool receipts followed by a finalizer WorldState schema error;
`payroll_discrepancy` and `anniversary_recognition` have startup harness errors
and no tool-server occurrences. Keep their behavioral/terminal availability
separate. Neither becomes a valid zero model-quality assessment.

## Evidence appendix

The machine ledger resolves every task below to an exact attempt, episode path,
SHA256 and native trace. Invocation IDs identify native tool-server occurrences,
not assistant turns or sampled tokens. Selected receipts were decoded for review;
native bytes remain the replay authority. For multi-occurrence cases, all listed
IDs are checked for unique resolution within that task by the ledger generator.

| Ref | Task | Key invocation IDs / source facts |
| --- | --- | --- |
| M1 | marketing.partner_content_approval | `9b48d4954cb34234b3155229f244550f`, `ba2741e93f88493399af7b37ec98a20c`; initial Gmail `msg_legal_hold` |
| S1 | support.gorgias_refund_processing | `82d421b9e286497b9e5424597907b662`, `fb7512eac7cd4d43b482421f5c6c0c4f`; `ss_refund_policy/ws_thresholds`, `ws_overrides` |
| S2 | support.hiver_slack_digest | premature log `4e1753c92b624056b6193b9ff3c40823`; subsequent send `96155ea1f6fb472da45739abc3e5ffdc` |
| T1 | sales.unreliable_label_account_review | sent email `b995030548e04b85a62960e239375de0`; system prompt clock; `ss_activity/ws_activity` Meridian Labs row |
| T2 | support.intercom_reactivation_campaign | log `1fc98b96aafa469eb22e605496bc4a00`; summary `c0d3a54145c64ddc8bcfc1a371eb577b`; system prompt clock |
| T3 | support.helpscout_weekly_report | conversation response `7fb4ddd26c864e5cbcfb1245af69536a` |
| F1 | finance.wire_transfer_approval | Pending writes `15f246874e3f4fecaf1b9db3baede25b`, `dad8488363dc461293b74e9b403dc1fe`; honest no-transfer notice `d3ebd2c172524815bc8b967999b97ccb`; `msg_wire_auth` |
| F2 | finance.invoice_dispute | credit writes `0cfbce460cc247fda37e23f3be637720`, `bb069f5477794e999b02b382c22296d9`; duplicate status `b0c59eac72774b4594a5bed9877577df`; user prompt, `msg_dispute_policy`, `msg_vp_threshold` |
| O1 | operations.safety_certification_gap_analysis | first ordinary remediation `db12c679d8954373a47e858e45a2406e`; first synthetic-row remediation `08ac79f8699a4a728bd155b95eef8271`; summary `3f81b686563744aa92e3df8275274084` |
| Z1 | sales.calendly_sla_monitoring | useful policy read `19b6432363eb4872a99b7f1ecbc204ad`; empty Case query `7910810d761b41ebb227936dd2e129bc` |
| N1 | marketing.editorial_calendar | backlog `b72c1fdea8d34206aea5396f1bac6dd1`; recent posts `777abeb28b154cc4919d1320d3681a8d`; public Gmail `msg_editorial_policy` |
| N2 | operations.conference_room_booking_conflicts | empty calendar discovery `06fe0c27db494f55acac1eb64f54e459`; public `ss_room_booking/ws_booking_policy` and native events |
| N3 | operations.access_request_validation | provisioning `ea1e6590c0914762ac1190d7c1aa4c2d`; section `c28e9694adb342ee8efe854ff8a28544`; denials `e3a5fd229a904840a62d4f6c363fe84d`, `d27dad3a234341899bf017f9c28c3db5` |
| N4 | sales.sheets_reconciliation | opportunity query `dae5b9b598414df990c5e1069debd3ba`; account query `9635b8e710944a40a1d0e68f390d6e7c`; update `9ea892bb775a4a11b5ef7788702e2fc6` |
| N5 | support.zendesk_gdpr_purge | ticket edit `2115b16cf7864445940fb12257037472`; deletion `5abb7da45555492ba3531e38c5cc86ac`; Marie log `650cd07930c74ecdbde1e40bc2ad6104`; summary `50f8c84fed804cd08c0f7579281ee038` |

Other selected cases and their concrete rule implications are enumerated in
`luna-development-review-cases.json`; the coverage file supplies source paths
and digests for all indexed tasks, including unreviewed ones. This separation
keeps the readable report short enough to use without hiding evidence boundaries.

Rebuild the ledger from the repository root, using the frozen final index and
coverage paths below; no runtime import or inference is involved:

```bash
python3 docs/research/verifiers-assessment-qualification/review_development_coverage.py \
  docs/research/verifiers-assessment-qualification/sdk-probes/luna-reference-campaign-01/development-traces-index-700-with-verification-a97042bf9ce7.json \
  docs/research/verifiers-assessment-qualification/sdk-probes/luna-reference-campaign-01/development-coverage-final-9d0b173c29e7.json \
  docs/research/verifiers-assessment-qualification/luna-development-review-cases.json \
  docs/research/verifiers-assessment-qualification/luna-development-review-coverage.json
```

This review deliverable now has final all-709 coverage, source-bound selected
cases across all 46 families, and explicit recommendation dispositions. The
644 structurally screened cases remain substantively unreviewed and the nine
artifact-loss tasks remain unavailable. Whole-task reward qualification still
requires authored rules and executed tests; implementation acceptance and Qwen
eligibility are outside this report's ownership.
