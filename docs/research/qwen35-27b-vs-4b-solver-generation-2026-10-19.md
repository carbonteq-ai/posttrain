# Qwen3.5-27B versus 4B: native solver generation

This is a diagnostic on three reused, training-eligible tasks, one greedy
rollout per task and arm. It is not an unseen benchmark, a training experiment,
or proof of reasoning faithfulness. The user authorized fresh27B execution with
and without pipeline guidance and comparison with the saved4B opt05 run.
The pipeline prepares requirements, a historical workflow summary, an assembled
procedure and a same-model review; the solver receives only accepted compact
guidance, not the raw Luna trace or manifest. The critic is not an independent
verifier. Native calls and results remain the evidence authority.

## Matched settings and source

27B uses BF16 Qwen/Qwen3.5-27B revision
fc05daec18b0a78c049392ed2e771dde82bdf654. Saved4B uses revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Both are fresh bases with no optimizer
updates/adapters, the same native states/prompts/tool schemas/reference actions/
goal manifests,32K context,4096 reply tokens,16384 episode output tokens and12
tool turns. Actors use native thinking and greedy decoding; preparation uses the
same model, nonthinking greedy decoding,2048 reply tokens and structured JSON.
Both use2 episode slots,8 server sequences,16GiB KV,block128,optimized pinned
vLLM and image sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337.
All execution is offline on the existing workstation. No truncation or model
substitution was introduced. Comparing own-model preparation plus execution is
an end-to-end recipe comparison, not an isolated actor-size ablation. No matched
whole-pipeline cost or throughput conclusion is drawn.

Saved4B evidence: evidence/luna-ssr-sft-20261018/adaptive-next-five/experiment05.
New27B evidence: evidence/luna-ssr-sft-20261018/solver-27b-comparison-20261019-r1
and r2. The original5/5 adaptive budget remains complete and separate.

## Original attempt r1

The driver and inputs exactly match sealed4B opt05. Accepted container
3227a320e53a6d504f514e4654635c0a7eddf66eae75c6a840f3e323acd7753a exited0
after537.49seconds, but only five of six planned native episodes ran.
Source manifest5647eb696cf9762a8ddd3e6aa06105c8e36f3d47f6da0963621ecc02271ace5c.

| Task | 4B control | 4B guided | 27B r1 control | 27B r1 guided |
| --- | --- | --- | --- | --- |
| Create Airtable contact | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Find contact then create VIP | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Sales soft matching | 1 / 1 | 1 / 1 | 0 / 0 | Blocked before solver |

Entries are native success / partial credit. Blocked is missing execution,
not a zero score. All five completed episodes scored with no native errors or
length-finished model replies. No successful-action replay or SFT admission is
inferred from a passing score.

27B Sales control searched `from:NexGen HR contact update`, retrieving the HR
request but not the matching-policy email. It searched the exact supplied name,
then first/last names, and retrieved plausible decoys. It correctly noticed
their conflicting companies/titles, but did not obtain the policy or search
the alternate Kristen spelling/company-title combination. It eventually
treated Marketing Director as a sufficient match to VP Marketing and updated
Kristin O'Brien,003xx000004KRS1. It stopped without creating a note. The native
guard records a witnessed prohibited decoy update. Syntax-level tool mistakes
remain zero: this is a wrong-record decision, not an invalid function call.

The saved4B control initially guessed an unsupported email domain and got an
empty result, then recovered with a broader search that returned the policy.
It enforced all-three-factor matching, queried company/title, updated target
003xx000004KOC1 and created a note.11 tool-call turns versus27B's6 tool-call
turns/7 calls (one parallel batch); fewer calls did not mean a better outcome.

27B guided VIP performed the right find-then-create actions, preserving the
old record. However, its rationale quotes "Do not issue an update action on
the existing record" as though the user said it. This exact prohibition came
from guidance, not the original public user message. That attribution is a
qualitative SFT concern despite correct behavior. Saved4B guided VIP instead
grounds creation in the public request to create a new record.

Sales preparation was blocked on `in:inbox NexGen HR`: a historical query string
matched the literal blacklist although its terms are public and its prefix is
search syntax. This is a conservative gate false positive, not evidence that
27B cannot solve Sales with guidance. The rejected workflow remains preserved.

## Evidence-backed gate repair r2

The one source change permits only `in:inbox` followed by at least two alphabetic
words already present in the public request. Private names, digits, IDs, domains
and email addresses remain rejected.38 focused CPU checks pass, including the
captured false positive and rejection counterexamples. This does not repair
the known argument-provenance or note-field interpretation issues.

Container3711d3747ff0547f78347ee79fdc92c6274d0f4847d92e12c365076dda584de6
was accepted separately. Source-manifest
25d10111ca973ffdb6446f4973cae4f41831e4955ea5d472d4fd9ecae28addc7.
All six slots are repeated; r1 is preserved and never pooled with r2. The4B
reference is not regenerated; comparison must disclose the sole gate change.
It exited0 after580.26seconds. Five of six native episodes ran, all scored,
with no native errors, observed tool error responses, repeated identical calls
or length-finished replies. All task materials still exactly equal the saved4B
materials. All sealed files and actual container images were verified after
execution; the GPU and Docker active-job lists were empty. The attempted guided
Sales slot remains unexecuted, not a semantic zero.

| Task | 4B saved control | 4B saved guided | 27B r2 control | 27B r2 guided |
| --- | --- | --- | --- | --- |
| Create Airtable contact | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Find contact then create VIP | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Sales soft matching | 1 / 1 | 1 / 1, note goal unmet | 0 / 0.667 | Blocked before solver |

These are the recorded completion/partial-credit metric pair, not a declaration
that all manifest obligations passed. See the separate note-goal audit below.
One seed/rollout per task gives no reliable model ranking or uncertainty estimate.
No model selection or update uses heldout/development/reserve data.

27B r2 control differed from r1 despite greedy decoding. It continued past the
initial decoys, queried Contacts by the observed NexGen Solutions AccountId,
selected003xx000004KOC1 and updated the correct phone. It still stopped without
the note, receiving success0 and partial2/3. Six tool-call turns/seven calls,
two empty responses. Do not treat r2 as a selective replacement success or pool
it with r1. Identical budgets/greedy sampling do not guarantee identical native
multi-turn paths; tool-call IDs differ, and causal explanation of variation was
not qualified. Both repeats and all raw requests are retained.

## Preparation quality and the unexecuted guided Sales case

The public-query repair allowed the same generated query through.27B requirements
then incorrectly said the Note title must contain the new phone number. Its
critic identified that the title must instead contain the policy phrase
`Phone Update Processed`. The proposed assembly repair copied that nonpublic
literal directly from critic context; the existing literal gate blocked it.
Unlike the first query issue, this is a legitimate source-separation rejection:
the solver should retrieve the matching-guidelines email and obtain the policy
there. No secret-value exception or forced execution was added. This also shows
why a more capable critic alone does not guarantee an admissible solver plan.

The critic called the title an exact-match requirement. The manifest actually
uses a verbatim `mentions` predicate, so the phrase must occur in the title; the
critic's exact-equality wording is stronger than that check. Its main correction
(phone number in body; required policy text in title) is supported. A future
repair should instruct policy retrieval, not copy private constants.

All four executed Simple episodes pass again. The guided VIP reasoning again
quotes guidance as if it were the public user's own wording. The create-contact
reasoning says JSON string but emits the schema-permitted object form, as in4B.
These small cases do not establish that27B generates cleaner reasoning targets.

## Completion metrics versus manifest goal evidence

Sales legacy completion/partial scores exclude checks: the successful4B traces
show3 scored and9 excluded assertions; the manifest goal/harm assessments are a
separate channel. They must not be collapsed into an all-requirements pass.
The exact `audit_note_on_target` assessment is valid1/required-effect-witnessed
for4B control, valid0/required-effect-missing for4B guided, and valid0 for27B r2
control. Its coverage signal is valid1 in all three, so these are audited missing
obligations, not unknown coverage.4B guided wrote a domain-based title instead
of one containing `Phone Update Processed`, despite its legacy success1/partial1.
That trace remains rejected for SFT.27B r1 additionally has a witnessed prohibited
decoy update.27B r2 has no such witnessed harm, but fails the note obligation.

## Verdict and next decision

There is no end-to-end evidence here to replace4B with27B for generation.4B's
unguided Sales path completes both the record update and required audit note;
27B's two unguided attempts fail, and its guided Sales preparation is blocked.
27B's critic did catch a field-specific error missed by4B's critic, suggesting
a possible separately tested reviewer role. This is a hypothesis from one task,
not a validated production routing choice.

The immediate pipeline work remains typed argument provenance, field-specific
constraints and explicit retrieval of policy values. Increasing model size did
not remove those issues. No new SFT examples were admitted, no training began,
and no benchmark or old SELF/SDPO/LFM queue resumed.38 focused CPU checks, Ruff
and diff checks passed. Raw traces, stages, model calls, source hashes and both
attempts are inspectable through solver-27b-vs-4b-comparison.html (r2) and
solver-27b-vs-4b-original.html (r1); saved4B data is labeled historical reuse.
