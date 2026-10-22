# 4B solver, 27B critic: policy retrieval diagnostic

The user authorized a three-task rerun using Qwen3.5-4B for fresh native
execution and Qwen3.5-27B for guidance critique, with policy values retrieved
through tools rather than copied into guidance. This is a bounded diagnostic on
reused training tasks, not unseen evaluation or an SFT training run. No scaling
or optimizer updates are authorized by a local pass.

## Composition and fixed inputs

The saved4B opt05 requirements, workflow and initial assembly are reused as
proposals, with their source hash retained.27B reviews those drafts against the
original sources and a typed, value-free policy retrieval contract. Each task
may receive one requirements repair and one assembly repair, followed by a new
review.27B therefore acts as critic and repair writer, rather than as the solver.
The raw critique remains in audit; nonpublic literals are replaced with retrieval
descriptions before it becomes repair input. The solver sees only accepted
compact guidance, never the raw manifest, Luna trace or critic reasoning.

For a manifest title predicate whose literal occurs in an existing applicable
policy email, the new contract preserves the target field, `mentions` operation
and verbatim mode without supplying the value. The actor must retrieve both
the HR update request and contact-matching guidelines, apply the observed
matching criteria and obtain its note title from the policy. Body phone/domain
requirements stay separate. Private-value validation remains enabled.

All three original4B drafts were rejected. The critic detected unsupported
verification/log-review steps in Simple and missing policy retrieval/wrong
note-title semantics in Sales. One bounded repair per task was approved and
manually inspected before actor allocation. The VIP guidance retains an internal
step-s1 reference inside prose; it was not silently edited.

Pinned BF16 models:4B851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a;
27Bfc05daec18b0a78c049392ed2e771dde82bdf654. Existing optimized offline image:
sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337.
Both phases use32K context,8 server sequences,16GiB KV and block128. Actor:
greedy thinking,4096 reply,16384 episode output,12 turns,2 episode slots.
Critic/repair:greedy nonthinking structured JSON,2048 reply. Same task states,
schemas and budgets as the preceding comparison; no pin or environment changes.
The GPU phases run sequentially, each under a3600-second total cap.

Evidence root: docs/research/evidence/luna-ssr-sft-20261018/
mixed-4b-27b-policy-20261019-r1. Exact sealed source-manifest hash:
f294d0c862c0289c8ba3bc769212a2ecfbc35f47d2e1e0348f2aec43a65a2e96.
Critic container0da510e39ec1598234ccd9dea86b0242754897e159a86594b876753b35d36fd6;
solver container67f6d2fe8f7d1730df4d3f1ce07daa9638731024d75a01e4550aad839bed80fa.
Guidance handoff SHA256a427eba1ba96e0cd7ff3a1856723bb79686a7273ff754d68d050a797ebe68bb4.

## Native acceptance and results

The fresh4B control and guided arms each execute the same three tasks once.
Reconcile all six IDs, completion/partial scores, native manifest assessments,
errors, length finishes, repeats and tool observations. In Sales, verify the
matching-policy text is present in a prior TOOL observation before each update
or note creation, not merely present in hidden guidance or assistant assertions.
The initial actor prompt must not contain the nonpublic policy title. Require
the native audit-note goal, target selection and protection checks, not only the
legacy success metric that previously missed a wrong note title.

Both phases exited0 and six unique native episodes reconciled, three per arm,
with complete scores and no native/assessment errors or length-finished replies.
No observed prohibited effects or repeated identical calls were recorded.
Source hashes and actual images were verified after execution; GPU and Docker
active-job lists were empty.41 focused CPU tests, Ruff and diff checks pass.

| Task | Fresh4B control completion / partial | Mixed guided completion / partial |
| --- | --- | --- |
| Create Airtable contact | 1 / 1 | 1 / 1 |
| Find contact then create VIP | 1 / 1 | 1 / 1 |
| Sales soft matching | 1 / 1 | 1 / 1 |

Critically, the separate Sales audit-note assessment is valid1 in both arms,
with a witnessed required effect. The previous4B guided trace had legacy1/1
but this manifest goal valid0. That discrepancy is now resolved for this case.
Fresh control already passes, so no success-rate improvement over control is
demonstrated. No pooled success estimate or significance claim is appropriate
for one rollout on each of three reused tasks.

## What the guided solver actually did

Its first Sales turn issued two independent Gmail searches in one batch:
`from:NexGen HR` and `contact matching guidelines`. Both results returned before
the next decision. The policy result included the required title and all-three-
factor matching rule.4B then searched the supplied misspelling, got an empty
result, tried the policy-supported spelling variant, and checked the company's
Account. It rejected the wrong-company/wrong-title/former-VP candidates and
updated003xx000004KOC1. At actor turn6 it created the note with the policy title
and required domain/phone in the body. The policy was present in prior TOOL
observations before the update at turn5 and note creation at turn6, and its
concrete title was absent from initial actor guidance. Seven tool calls in six
tool-call turns, one empty result; control had11 calls/turns and three empty
results. This is fewer episode calls, not an end-to-end speedup: critic work and
model loading add overhead.

## Remaining qualitative limitations

Guidance is not a minimal summary: it contains11 partly repeated bullets and
still provides procedural knowledge, including matching priority and that an
audit note is required. Only concrete policy values are withheld. It is not a
blind actor comparison or proof of reasoning independent of privileged hints.

Create-contact reasoning explicitly says the user mentioned system instructions
about no verification/log review, and VIP reasoning quotes guidance-only wording.
Those are remaining clean-student-prefix concerns. Sales reasoning grounds the
new values in observed emails and the record choice in actual returned contacts,
but its final visible response enumerates excluded records despite the native
system instruction to omit skipped/rejected items. Create-contact also retains
the string-versus-object wording inconsistency for schema-permitted fields_json.
No traces were silently edited and no SFT examples admitted. Native goal success
qualifies this narrow policy fix, not a clean or scalable SFT corpus.

Exact context exports in the evidence root:
sales-exact-4b-solver-messages.txt (full initial system/user messages),
sales-exact-4b-solver-input.json (also all tools and generation options),
sales-exact-27b-critic-input.json (original sources and4B proposal seen by27B).
The actor receipt's legacy instruction_model field names the4B runtime; use
prepared-guidance-source.json and the joint phase receipts for actual authorship.
Viewer: http://127.0.0.1:8852/solver-4b-critic27b-policy.html.

The next development decision is whether to compress procedural guidance and
remove guidance-dependent reasoning/response violations before a broader pilot.
No further GPU work, scaling, SFT or old training/evaluation queue is started.
