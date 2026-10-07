# Prepaid schedule goals: data-only rules and remaining shared primitives

Status: public-input proposal, 2026-10-04. **No executable positive manifest is promoted by this document.** The existing negative balance-movement guard remains a separate component.

The source is frozen batch01, task `finance.prepaid_amortization`, public-input SHA `ac47fd9382da85450c8ac73a8a28c60c7be0a1b502fc33f2356c7d5069a46c77`. Pack SHA: `d865aacbaac51c83441f852941b06116303d18bf3973bc1c8b265144c1c8e50d`. Only its public request, initial schedule, controller rules, VP exception and controller correction supplied policy. No hidden answers or recorded actions were consulted.

## Public rules to author as data

Use the initial `ss_prepaids/ws_prepaid_items` population with a unique Item key and qualified native record identity. The task's source-bound accounting date is February 1, 2026. Preserve these three branches as policy data; do not list expected row IDs or precomputed outcomes:

| Branch | Public authority | Effective calculation |
| --- | --- | --- |
| Standard | `msg_amort_rules` | Total / positive integral Term (Months), rounded to cents under a declared rounding interpretation. |
| Annual Insurance, February | `msg_vp_accel` | Double the standard monthly rate for this item and period only. |
| Software License correction | `slk_sw_corr` | Use corrected Total 7200 for the monthly rate beginning this month; preserve January's already recorded historical amount. |

The named exceptions and 7200 are explicit public policy facts, not task-specific executable logic. Their bindings must include identity, author, date and relevant text, as in the guard proposal. Replacing an item name or correction in the public source requires reviewed rebinding, not new Python.

Eligibility is **inside coverage period and positive initial Remaining**. Future-start and zero-remaining rows are excluded. Do not reuse the stale Software note “skip this month” after ignoring the explicit current-month correction. No blanket ban on notes or source-total corrections follows from this rule.

Let `T_effective` be the corrected or original total, `A_initial` initial Amortized to Date, and `R` the recognized amount. A proposed schedule consistency rule is:

```text
Amortized to Date_after = A_initial + R
Remaining_after        = T_effective - Amortized to Date_after
Total_after            = T_effective, when a source-total correction is required
```

The corrected-total balance relation is a **reviewed accounting interpretation**, not an explicit formula quoted in the procedure. It matters: subtracting the new monthly rate from the stale Software remaining balance would fail to reconcile the corrected invoice total. Preserve January's historical recognized value; do not silently recalculate prior months at the new rate. Validate this interpretation before installation.

Do not invent a cap at Remaining if the monthly rate exceeds it: the supplied rules do not specify that boundary. Mark such a variant unresolved until the policy is authored. Likewise, rounding the doubled rate before versus after multiplication can differ. Current public values divide exactly, but variants need an explicit interpretation. Use `unavailable_on_tie` for unspecified ties; that alone does not settle multiplication/rounding order.

## What current operators can express

Existing `values.py` supports strict USD and decimal parsing, exact arithmetic, explicit rounding and date comparisons. Existing predicates can express the three source-bound rule branches and compare qualified write fields to derived values. `SheetEffectSource(kind="update")` provides before/after cells, requested/changed fields, native record ID and an acknowledged persisted update witness.

These suffice for a **formula-correct update occurrence** once eligibility is independently proven. A matching update must affect the same initial native record and include a relevant requested, changed financial field. An unrelated Notes update after some other writer changed the balances is not itself amortization work.

Do not require all fields to change in a single call. One combined update and multiple legitimate field updates can reach the same requested schedule. Final-state verification and credit for the relevant contributing/completion execution must distinguish those valid paths. Exact string formatting should not override numeric equality unless the public instruction actually requires a verbatim copied value.

## Two shared gaps block a complete schedule goal

### Calendar-month coverage

Current date operators provide date difference and interval comparison, but cannot derive an end date by adding Term (Months). Checking only Start Date <= February 1 and Remaining >0 admits expired items. Using 30 days per month is incorrect.

The smallest addition is a bounded calendar-month coverage operation, or month-add value with explicit conventions:

- strict ISO start date and positive integral term;
- source-bound as-of date;
- declared start/end inclusivity;
- declared behavior for day 29/30/31 when the target month is shorter;
- no host-clock fallback.

The source does not specify arbitrary month-end conventions. For this first task, admit the public first-of-month starts under the explicitly reviewed interval `[start, start + term months)`; leave other conventions unavailable until authored. This is a generic operator restriction, not a list of accepted task rows. An equivalent bounded operator that validates the first-of-month condition is sufficient.

This proposal therefore does not ship a current-schema draft whose incomplete eligibility silently earns positive credit.

### Retained schedule outcome

`capture_sheet_retention` and `validate_sheet_retention` already exist in `contracts/sheet_effects.py`. They retain final cells, native record ID, source/selector digests and finalization status. `ObligationCheck` and `manifest_obligation_assessments.py` currently consume occurrence evidence without that final-row evidence.

Add a narrowly named retained-row predicate check over the existing initial population and that existing retention evidence. Its context contains the original request row and the same qualified native record's final cells. Expected values remain typed manifest expressions. No parallel state store or new native assessment API is needed.

Keep its semantics explicit:

- Final fields match: verified goal outcome, even if action attribution is unavailable.
- Final fields are known wrong or the original record is demonstrably absent: goal failure.
- Identity, finalization or required fields are unresolved: unavailable.
- An earlier correct update followed by damage does **not** remain a successful retained goal.
- Restoring the same record can restore the outcome; it does not erase separate harm or create repeated once-only credit.
- A new record reusing a row number cannot discharge the old row's obligation.

The positive credit rule consumes both the retained result and qualified relevant transition witnesses. A verified state alone does not invent an action recipient. Selecting a completion execution is a declared credit policy, not a claim that that execution alone caused every correct field. Stable same-episode obligation consumption and source/receipt validation remain mandatory.

Prefer this small explicit state check over changing every existing occurrence check's meaning. Reuse native planning/publication and retain one check instance per eligible initial member.

## Accounting summary stays a separate gap

Even a fully correct schedule does not qualify the entire task. The request also requires:

- a Gmail journal summary to controller@company.example.com;
- debit expense accounts and credit Prepaid Asset;
- affected names and the exact public `Total amortization: $X` line;
- a closed sum of eligible recognized entries;
- silent exclusion handling.

A recipient-only send or a number appearing somewhere in the body proves none of these together. Grouped arithmetic, source-bound line/account content and prohibited excluded-item narration need their own qualified evidence. Do not impose a full teacher-written body template.

## Acceptance before installing the schedule component

1. Standard, insurance and corrected-total Software paths derive their amounts from changed public values with no Python changes.
2. Expired, future, zero-remaining, invalid term and unsupported month-end cases distinguish inapplicable from unavailable.
3. A correction preserves prior recognized amounts while reconciling the new total; unresolved rounding and insufficient remaining stay explicit.
4. Combined and split field updates both satisfy the same final goal. Notes-only, no-op and numeric reformatting do not receive useful-action credit.
5. Correct update then damage fails retained verification; repair restores state without duplicate credit.
6. Renaming the same native row preserves identity; deletion and row-number reuse cannot steal the original obligation.
7. Missing ACK permits no invented action credit; incomplete final capture permits no retained-success claim.
8. Actual native Task.score, repeated score, serialized reload and variant fixtures preserve scalar compatibility, source bindings and once-only credit.

Implementation remains in the AutomationBench environment candidate. Proposed modules are a small addition to `contracts/values.py` for calendar coverage, a retained-row check reusing `contracts/sheet_effects.py`, and existing model/native publisher registration. This is a design increment, not a new framework.
