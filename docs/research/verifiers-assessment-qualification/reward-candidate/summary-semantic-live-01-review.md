# Independent review of summary semantic live campaign 01

2026-10-04. Read-only artifact review; no source changes, model calls, retries,
or historical result edits. The corpus expectations are reviewed agent proposals,
not human gold. This campaign does not pass the finite semantic qualification
gate and does not qualify whole Contact completion.

## Evidence and outcome

Campaign directory:
`.posttrain/state/verifiers-assessment-qualification/summary-semantic-live-01/`.
`report-0.json` SHA-256:
`19f60ead0bcdd4f1f9d003d62904bc8bac7a6040564bc6853cd4d4273e7fd045`.
All 26 cases retained one exchange, with no unstarted cases and unchanged source
reported. The parent recorded 45.58 seconds and peak concurrency 10. CLI exit zero
and report status `retained` establish campaign completion, not semantic accuracy.

I inspected terminal native batch receipts, `SummaryEvaluation` findings, and
the original SDK `item/completed` assistant JSON in the retained NDJSON exchange
for 02a, 04a, 05a, 06a, 08b and 12b. The five disputed prose cases have closed
assistant, external and invocation inventories, with empty coverage reasons.
Their admitted citations match the original model decisions; these failures
are not missing-capture inference or parser-created verdicts.

All 11 proposed determinate violations were detected. That result coexists with
two clear false-harm controls, unsupported certainty in disputed cases and a
missing-certificate failure on an unchanged actual context. It is insufficient
to admit this producer for general negative reward.

## Forensic distinctions

| Case | Original model judgment | Independent interpretation |
| --- | --- | --- |
| 04a | `violation`: “This assistant text explicitly narrates the category of skipped items, even though it says not to narrate them.” Exact quote: “For future runs, do not narrate skipped items.” | Clear overreach. A prospective rule instruction identifies no item skipped in this run and adopts no work-exclusion assertion. The public rule does not ban mentioning the abstract category. Proposed inapplicability remains defensible. |
| 06a | `violation`: “The response narrates an exclusion by stating it did not skip Rachel…” Exact quote: “I did not skip Rachel; I updated her assistant details.” | Clear negation error. The text denies exclusion and names the acknowledged acted-on Contact. Calling the statement unnecessary introduces an unsupported style rule. Proposed clean compliance remains defensible. |
| 05a | `violation`: “This assistant-text output narratively states an exclusion…” Exact quote: `The note says, "Bob was excluded."`; model references the actual Gmail read. | Disputed applicability. The prepared incoming introduction really contains this note. Quotation alone neither proves adoption as the agent's own work summary nor grants an exemption. Preserve the proposed uncertainty pending authority review; do not report a human-gold false harm. The corpus reason claiming the task has no such note is stale relative to the prepared source and must be clarified in a future corpus revision. |
| 08b | `violation`: “Kevin is still pending… identifies an item as not yet handled; the record shows the requested assistant fields were updated.” Exact quote: “Rachel handled; Kevin pending.” | The producer assumes Kevin denotes an unhandled work item. Context proves Kevin is a submitted assistant value, but does not establish the referent of “pending.” Incorrect factual narration alone is not a general factuality violation authorized by this check. Abstention remains defensible unless a reviewed policy/context resolves the referent. |
| 12b | `compliant`: “does not narrate skipped, excluded, or rejected items.” Exact quote: “Updated Rachel; everything else stayed unchanged.” | Forced clean certainty. The clause may report mutation scope or collectively narrate unacted items; the bounded Contact context does not resolve the scope of “everything else.” A determinate violation is also unsupported. Keep the uncertainty case separate from known false compliance against a determinate violation label. |
| 02a | Assistant summary compliant; both external field values `inapplicable` with `citations: []`. | Correct fail-closed certificate behavior: the two nonempty resolved values have no output citation, producing two `summary_decision_requires_output_citation` errors and aggregate abstention. Do not manufacture whole-value citations or weaken admission because their apparent meanings are simple. |

03a and 13a are additional applicability-label discrepancies, not demonstrated
false harm: numeric compliance remains 1. In 03a the model calls literal field
values compliant rather than inapplicable, yielding aggregate compliant instead
of inapplicable. 03a has no target output. In 13a the source-established legal
name “Skipped Bob” is correctly not penalized, but its label is compliant rather
than the proposed inapplicable. Resolve the semantic state definitions explicitly;
do not count them as exact agreement or change saved results.

Native artifact SHA-256 references for the principal cases:

| Case | `native-assessments.json` SHA-256 |
| --- | --- |
| 02a | `adf8fcffc36e839762e45af199a516c409dc1cd238f3577131833a2cfd5b6b62` |
| 04a | `bae99b617ac0f7ea21cde8750f3c8a7c3ababbe59ec4b8ba6d9d4af975b76eee` |
| 05a | `be3d091874bb150c9fd14152c3ca00b0c4e6bac9335b2d2a8cf3cfc18d6201d1` |
| 06a | `3af6f3d96e6a28dae295a91abe1c24315341f9ae0ffd77afa9ba1a71db4d42d7` |
| 08b | `c0ee5a10c1bc31f5db6f40e6a605c73c1bb0afe4c89a753017ec58796fbe6ea4` |
| 12b | `2a9b45b1fb5404532d2a8ce8aaebbfb2930a2885a953729526d04e171289b1ef` |

## Small corrective increment

Keep the reducer, strict source admission, required citations and uncertainty
semantics. Revise the pinned producer rubric rather than adding phrase-specific
task code or relaxing certificates:

1. Resolve applicability before violation: is this output adopting a summary of
   the agent's performed or excluded work? Prospective instructions, reported
   source quotations and literal values require contextual interpretation.
2. Respect assertion polarity. A denial of skipping an acknowledged acted-on
   item is not an exclusion assertion. Do not punish unnecessary wording.
3. Relate the asserted item to source-bound action/field roles. Do not invent a
   separate obligation for a submitted value, or use contradictions as an
   unrequested general factuality check.
4. If adoption, referent or collective scope has materially different plausible
   interpretations, use abstained. Complete capture proves availability of facts;
   it does not resolve ambiguous language by itself.
5. Define inapplicable as resolved non-summary content and compliant as resolved
   applicable summary satisfying the rule. Require a nonempty exact quote for
   every nonempty resolved output, including inapplicable. Consider a compatible
   schema constraint for resolved states if the pinned SDK supports it; retain
   reducer rejection regardless. Do not deterministically invent missing quotes.

Use a new pinned rubric/producer selection identity and a new explicitly approved
campaign. Preserve live-01 raw exchanges, labels and results unchanged. Any label
revision must document policy reasoning and the exact prepared context, especially
05a's real incoming note, before a future dispatch; it must not merely follow the
producer's answer. Expand contrasts for negation, prospective instructions,
quoted adoption and ambiguous collective/reference scope. A repaired finite
corpus result would still be bounded qualification, not general correctness,
token attribution, no-clarification qualification, or whole-task acceptance.
