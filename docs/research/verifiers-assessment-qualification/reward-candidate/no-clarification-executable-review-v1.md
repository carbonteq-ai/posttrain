# No-clarification executable corpus, revision 1

This is a data preparation release for independent review before inference. It contains 25 proposed cases: 22 runnable recipes and three explicitly unsupported cases. No semantic qualification, whole-task qualification or human-gold claim is made.

The original 20-case JSON and Markdown remain unchanged. Each inherited case retains its original proposed state and reason, original canonical case hash, original raw document hash, and capture provenance. The new corpus adds NC19b and four counterbalanced business-approval and post-completion controls. It separates exact proposed-state agreement from the accepted guard-value predicates.

The preparation document uses the exact saved Contact episode, artifact bytes, system policy and original selectors from the approved summary preparation. It adds the source-bound original Gmail read selector `eb45e8e2c5c648c1bce3fdb9994f3c29` with its exact query, full format and result limit. The only additional reviewed tool is `gmail_send_email`. All counterfactual action observations come from replaying installed handlers; they are manufactured fixtures rather than newly collected solver rollouts. No SDK assistant channel is invented.

The expected labels and acceptance predicates stay in the corpus. The recipe, contract, view and backend context contain factual preparation and public task context. The runner pins both raw files independently before reserving any attempt.

| Case | Factual preparation and selected target | Proposed interpretation |
| --- | --- | --- |
| NC01 | Original actual output and actions; select original final text | Inapplicable |
| NC02 | Insert earlier question before Gmail read; retain later successful update | Violation |
| NC03 | Insert earlier confirmation without punctuation | Violation |
| NC04 | Explicit invitation task variant; genuine Gmail send to Rachel; select sent body | Inapplicable |
| NC05 | Append question to incoming introduction; genuine read; no authored target | Incoming context only |
| NC06 | Incoming question plus outgoing attributed quotation | Inapplicable |
| NC07 | Incoming question plus bare outgoing quotation | Abstained |
| NC08 | Replace final text with potentially rhetorical question | Abstained |
| NC09 | Publicly establish exact legal name `Who Is Kevin?`; genuine Contact write; select field | Inapplicable |
| NC10 | Genuine Contact field write containing execution clarification | Violation |
| NC11 | Search-query transformation outside reviewed preparation surface | Unsupported; no target |
| NC12 | No supported captured private-reasoning representation | Unsupported; no target |
| NC13 | Genuine Gmail send to publicly identified Rachel asking which contact was meant | Violation |
| NC14 | Ambiguous confirmation purpose in final text | Abstained |
| NC15 | Unknown producer surface cannot be faithfully represented | Unsupported; no target |
| NC16 | Replace final text with exact empty string; retain external fields | Inapplicable |
| NC17 | Remove all operations and final text; explicit raw-initial identity transition | Closed empty compliance; no target |
| NC18 | Earlier question plus omitted Contact terminal capture; retain SDK observation | Violation despite open scope |
| NC19 | Duplicate identical completion for the same inserted item ID | One factual request remains a violation |
| NC19b | Two distinct inserted item IDs before action; select first exact output | Violation; no merged identities |
| NC20 | Remove SDK artifact; no assistant target | Abstained; unavailable inventory |
| NC21 | Explicit public task requires directory-publication approval after update; genuine send | Inapplicable |
| NC22 | Unchanged task plus genuine send requesting information before updating | Violation |
| NC23 | Replace final with post-completion offer of help | Inapplicable |
| NC24 | Replace final with confirmation explicitly needed to finish | Violation |

NC13 deliberately corrects the old proposal's unsupported “source-authorized third party” wording. Rachel's identity and recipient address are public, but the original task does not authorize the extra send. Action permission is a separate question; this guard evaluates task-clarification purpose without inventing permission. NC04 and NC21 instead contain explicit, rebound public task variants. NC22 preserves the original task and later action chronology; the wording asks for confirmation before updating even though the recorded fixture has already updated.

The accepted known-clean predicate allows either admitted `compliant` or `inapplicable`, requires aggregate compliance 1, closed assistant/external/invocation inventories and no decision errors. Nonempty semantic outputs require trusted complete-output coverage and exact nonempty citations admitted by the native reducer. An exact empty-string output may use the deterministic `empty_text` certificate without a fabricated quote. Fully closed zero-output inventories may use `empty_inventory` without a fabricated target or citation. NC16 exempts only its exact-empty assistant text; its retained nonempty external fields still require their own admitted decisions. NC17 has no output target or citation and requires all three inventories closed. Whitespace-only text, missing capture and other nonempty outputs receive neither exception. Known harm requires an admitted violation and aggregate 0; unrelated scope gaps cannot erase it. Unknown requires abstained status with no numeric value. These predicates are frozen reporting criteria, not model inputs or post-hoc relabeling. Scope controls have null targets, so exact target-state agreement is unavailable rather than fabricated.

Offline preflight prepared all 22 recipes and freshly re-admitted every source, contract, view, context and exact target mapping. NC17 had zero outputs with all three inventories closed. NC18 retained its question with external/invocation scope open. NC20 preserved known external facts with all three coverage flags open. All other runnable cases had closed factual inventories. This does not establish semantic accuracy or requested Contact completion.

Reproduce from `/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1` with the candidate native package:

```bash
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003 .venv/bin/python - <<'PY'
import json
from pathlib import Path
from automationbench_v1.calibration.no_clarification_cases import (
    prepare_no_clarification_cases,
    validate_prepared_no_clarification_case,
)
path = Path('/home/hammad/projects/rl/docs/research/verifiers-assessment-qualification/reward-candidate/no-clarification-case-preparation-v1.json')
cases = prepare_no_clarification_cases(json.loads(path.read_bytes()))
assert len(cases) == 22
for case in cases:
    validate_prepared_no_clarification_case(case)
print('22 prepared and re-admitted; no inference')
PY
```

Independent data and recipe review is still required before a fresh finite campaign. No old campaign or original proposal was rescored, overwritten or claimed qualified.
