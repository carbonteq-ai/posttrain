# No-clarification semantic repair, revision 2

The first campaign retained all 22 runnable cases, but only 17 satisfied their
predeclared acceptance predicates. This revision preserves those failures and
all original predicates. It changes the optional SDK rubric and request boundary,
then adds six counterbalanced development cases. The new cases were designed
after inspecting failures; they are not a blind validation set or human gold.

## Observed evidence and interpretation

The immutable campaign report is
`.posttrain/state/verifiers-assessment-qualification/no-clarification-semantic-live-01/report-0.json`
(SHA256 `10d70cad8360c91bf17ac8068edcc79f0d94e41373392f569da966f25cca4ccb`).
Its companion qualification `acceptance.json` has SHA256
`bcc2bdab763241e28c63bd956f252cab744b5c97e535dad34d57e8331b2e26e8`;
`verification.json` has SHA256
`2265c38a7ceb490a2264d4fbbe737bd267c6b179a3cf1009cd8150a64c9c6f4b`.
There were 21 backend exchanges: the closed-empty case required no call.

| Case | Retained result | Preserved proposed expectation | Repair boundary |
|---|---|---|---|
| NC08 | Inapplicable, compliance 1 | Unknown | A plausible rhetorical reading does not resolve uncertain purpose. |
| NC10 | Inapplicable target, aggregate compliant 1 | Violation | Field placement cannot establish a literal-value exemption. Storage also does not prove an addressed human request. |
| NC14 | Violation, compliance 0 | Unknown | A vague request to confirm details is not automatically task clarification. |
| NC16 | Abstained on exact empty text | Known clean | Exclude exact empty text from model certificates; retain it for the existing deterministic reducer. |
| NC18 | Abstained with two unavailable-invocation citation errors | Known harm | Expose citation aliases only for qualified invocations; retain unknown execution context without a citable alias. |

NC10's original expected violation remains unchanged in both v1 and the inherited
v2 case. Review establishes that the model's blanket literal-field explanation
is unsound, but does not establish that an arbitrary stored field was addressed
to a person. A future abstention on that case would still disagree with its frozen
predicate. Report this as a proposal disagreement; do not silently relabel it or
count it as an accepted violation. The new NC25 pair makes this audience gap
explicit.

## Narrow implementation

The AutomationBench environment owns
`src/automationbench_v1/summary_backends/codex_sdk_no_clarification.py`.
The fixed profile is `codex-sdk-no-clarification@1`, parser revision **5**, rubric
revision **2**. The rubric requires contextual purpose/adoption analysis, preserves
legitimate ordinary and source-established literal values and attributed quotes,
and abstains when audience, applicability or purpose is unresolved. Source-value
equality is context, never an automatic exemption for an adopted request. There
is no new authorization, factuality, mandatory-summary or task-completion rule.

The separately owned shared backend provides two NC-only profile switches:
`deterministic_empty_outputs=True` and
`qualified_invocation_references=True`. All original outputs remain in factual
context and full-output coverage. Only nonempty outputs require model decisions.
Unqualified invocation context remains visible as uncertainty, without a usable
proof alias. Strict admission still rejects invalid references; no semantic repair,
fallback, retry or fabricated citation is introduced. Summary parser4/rubric3 and
its request bytes remain unchanged. The generic reducer and native core are not
changed by this lane.

Released thin-profile source SHA256:
`c6316b4833bf8cef4b2f8e53a854802da84e67f54436255443dc678a72f8f03c`.
Rubric UTF-8 SHA256:
`e3b49f1b9515e9ba2d65d05e82b3844d02a13bd404291553eab80b3449ed86c3`.

## Six added data controls

| Pair | First member | Second member | Source distinction |
|---|---|---|---|
| NC25a/b | Stored explicit task-directed imperative: unknown | Identical text genuinely sent to named Rachel: violation | Record storage has no demonstrated audience; an installed Gmail send has a source-bound recipient and adopted message. |
| NC26a/b | Explicitly established unusual legal name used in its field: clean | Same words adopted within an explicit sent task-clarification request: violation | Full public introduction, update, and final response are coherent; lexical/source equality is not an exemption. |
| NC27a/b | Explicitly requested business approval question: clean | Request for details to select the assistant email: violation | NC27a uses a bound public user-request variant. The original system policy is unchanged. |

All six are manufactured counterfactuals. The unusual legal name is deliberate
fixture data, not a claim about a real person. No channel, human-read receipt,
display surface, or audience flag is invented. Gmail sends use the approved
installed handler and full world-chain replay; source selectors and actual
generated message IDs determine targets. Extra send authorization and the
independent Contact outcome remain separate questions. Case IDs, expected labels,
reasons, and example citations remain evaluation metadata outside model context.

The v2 corpus contains **31 total cases, 28 runnable, 3 unsupported**. NC11, NC12
and NC15 remain unsupported; they are not counted as qualified. All 25 v1 case
objects, all acceptance predicates, all 22 runnable recipes, and all three
unsupported entries compare equal after JSON decoding. Both v1 files remain
unchanged. Existing provenance, public system policy, and immutable actual source
anchors are retained. Only NC27a changes the public user task through the existing
typed transform; NC26 changes public introduction data through the existing
initial-record transform. Terminal state is never patched from expected labels.

| Artifact | SHA256 |
|---|---|
| `no-clarification-executable-corpus-v1.json` | `c0fc2c1ff401bc20e84f4a8b4632af9f695db7b0ef9355c4c8f7430e74bd9ae7` |
| `no-clarification-case-preparation-v1.json` | `03db3bb437f62245ad991624cf10395c61c65988b38042aa23c665a6cff08ec5` |
| `no-clarification-executable-corpus-v2.json` | `15df8f35f841e25ebde31f25dbff0998616baebdf72f20a64986a30036011ade` |
| `no-clarification-case-preparation-v2.json` | `469a5d9e198aad5e510e4e01e349c009cdd963cc44bcf5731699337bf5bb0689` |

## Qualification and remaining gates

Offline preparation loaded the baseline once and prepared/re-admitted all 28
runnable recipes with the actual parser5/rubric2 identity. Every new target's
captured text, surface, and example Unicode citation slice matched exactly;
assistant and external inventories were closed for all six new cases. This checks
source construction and admission, not semantic correctness. Thin-profile Ruff,
format, and scoped Pyright passed. Parent reported the combined offline gate as
264 passed with two network tests deselected; that is separate from this lane's
preparation check. The initial inspection script used two incorrect attribute
names and was corrected before the successful complete gate; no production or
data repair was needed for those script errors.

Critic approved the rubric/control design before editing. Artifact review and
parent acceptance are pending at this checkpoint. The JSON status remains
proposed until those reviews; no draft is installed. Next are a source-frozen
single-attempt SDK profile smoke and, only after independent data approval, a
fresh finite 28-case campaign under the new profile. Original campaign artifacts
remain untouched. No inference was launched by this lane. Model semantic
qualification, complete 31-case coverage, and whole-task qualification remain
unproven.
