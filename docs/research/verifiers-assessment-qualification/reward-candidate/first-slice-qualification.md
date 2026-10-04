# First deterministic HR slice: isolated development candidate

2026-10-03. The candidate emits independent native findings and semantic credit
through the normal task scorer. It preserves AutomationBench's official scalar.
It is not integrated, committed, published or promoted into the running collection.
No model calls were made for this qualification.

## Implementation and opt-in

Environment candidate:
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`.
Native API candidate:
`/home/hammad/projects/verifiers-credit-candidate-20261003`.
The initial dirty-source copy is recorded in `source-copy.json` beside this report.

- `hr_rules.py`: frozen internal predicate inputs and findings for future offboarding
  notifications, individual referral payout privacy, and per-employee NDA goals.
- `hr_evidence.py`: immutable receipt/snapshot interpretation. Native write IDs,
  read/write revisions and conflicts are checked; state acknowledgement alone is
  insufficient to establish delivery. Effects require actual domain-world deltas.
- `hr_assessments.py`: `ReviewedHrTask` publishes native `Assessment` records,
  then separate identity-mapped `CreditContribution` channels. The normal taskset
  chooses this class when `task.reviewed_hr_assessments=True`.
- `taskset.py`: only the candidate opt-in field and task-class selection were added
  by this implementation. Existing copied modifications are preserved.

For new evidence collection, also select `capture_actions=True`. Without captured
native tool-server evidence, required coverage is unavailable. Unsupported tasks
produce an explicit unavailable coverage finding. They do not get fabricated
guard passes or zeros.

The published transport is native Assessment/AssessmentBatch and
CreditAssignment. The pure `Finding` dataclass is an internal working value, not
another persisted reward format. Identity assignment has no advantage, efficiency
penalty, learned weight or aggregation policy. Goal, harm and diagnostic coverage
channels remain distinct. Algorithms must choose channels deliberately; coverage
is not an optimization objective.

## Actual development evidence

`first-slice-replay.json` retains exact episode file digests, sealed source IDs,
actual imported module paths and findings for three development attempts:

| Task | Observed candidate findings | Official scalar |
| --- | --- | --- |
| offboarding_automation | Raj's first delivered notification remains a violation; the later corrective notification is separately retained under the same no-notification policy. Greg/Diana notifications are resolved nonfuture nonviolations. | Unchanged |
| referral_bonus_tracking | The individual Payroll referral message with Finance in CC violates the public privacy policy. | Unchanged |
| docusign_nda_collection | Three distinct employee accomplishments, including Mei-Ling; already-Signed Sarah remains preserved. Failed empty-argument tracker operation contributes no accomplishment. | Unchanged |

Each retained replay calls actual `Task.score`, obtains native findings and
assignments, and reloads them through WireTrace. Token projection remains
unsupported. An execution UUID is not a generated call, assistant turn or token
coordinate.

## Independent checks

The new test file covers the strict 14/15-day boundary, persistent harm after
correction, Finance in To/CC/BCC, direct-referrer payout delivery, display-name
recipients, failed sends, missing acknowledgements, conflicting writes, wrong
templates, status-before-send, duplicate accomplishments, signed-status damage
then restoration, exact sheet/worksheet/row identities, duplicate rows, entity
word boundaries, unknown services, missing capture envelopes and existing draft
delivery transitions. Three supported tasks and an unsupported task are also
loaded through the normal config-selected taskset and scored without inference.

From the environment candidate package:

```bash
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003 uv run --frozen pytest tests/test_hr_assessments.py tests/test_environment.py -q
uv run --frozen ruff check src/automationbench_v1/hr_*.py src/automationbench_v1/taskset.py tests/test_hr_assessments.py
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003 uv run --frozen pyright src/automationbench_v1/hr_rules.py src/automationbench_v1/hr_evidence.py src/automationbench_v1/hr_assessments.py
```

The final combined suite passed **48 tests** (23 HR candidate cases and 25
existing environment regressions). Ruff passes. The three new modules passed Pyright.
Candidate Git diff whitespace validation passed. No dependency or lockfile was
changed, and the package uses its own isolated virtual environment.

## Explicit limits and next acceptance

The referral purpose matcher recognizes a literal source-bound referrer/referred
pair, policy amount and recipient route. Other legitimate wording, wrong amounts,
missing entity identity and ambiguous purpose stay unavailable; they are not safe
passes. This is a reviewed bounded interpretation, not a general natural-language
classifier. Hidden assertions are absent from producer views.

New-object mail, Slack and envelope effects are supported. Changes to existing
mail/Slack/envelope objects, including draft-to-sent delivery, make coverage
unavailable in this slice. Positive NDA goals require applied sent/status effects
and a covered final status; irreversible message harm can be retained from a
known delivered prefix even when final coverage is unavailable.

Coverage means the selected retained, acknowledged action material is available.
It does not prove provider-internal completeness, every possible domain action,
actual model conditioning or a universal privacy/guard classifier. Action-world
snapshots are trusted environment-producer material; this does not independently
prove arbitrary external side effects from an MCP success wrapper.

Public policy adapters cover only these three task revisions. They do not settle
the referral tracking hidden Paid vocabulary, business-day calendar questions,
other 97 HR policies or cross-domain rules. Whole development-bank review, native
API final acceptance, broader semantic counterexamples, algorithm channel
selection and publication remain separate gates. The 91 reserved tasks were not
inspected.
