# SELF correction and transfer experiments

The ten bounded experiments are complete. The current goal-augmented SELF setup
and its tested correction variants do not qualify for a full training restart.
The preselected primary reaches 45.83% success on 24 reserved tasks, below the
original base's 58.33%; partial credit is 58.24% versus 78.90%. No evaluated
candidate exceeds the original base's success.

Selected tool errors are learnable: local failure success reaches 62.5% at the
primary checkpoint. However, correct conditional teacher probabilities and
falling loss do not reliably improve free tool calls or transfer. The matched
soft-distillation and SFT controls both score 33.33% reserved success. These
findings support repairing the training signal and data coverage before another
long run; they do not prove that every SELF or self-distillation method fails.

## Local repairs improve while important failures remain

All repair experiments start from the same retained Qwen3.5-4B student at
checkpoint64. That checkpoint had deteriorated during the stopped goal-augmented
SELF campaign using the goals-observations-v4 teacher context. It is not a clean
paper-faithful SELF replication. The four selected failures start at zero success and zero
partial credit. Recovering them therefore demonstrates repair of a damaged
policy, rather than an improvement over the original base model.

The table reports final step10 outcomes unless marked otherwise. Failure
outcomes average four tasks with two native episodes each; retention covers
four separate successful tasks, also twice each. Experiment03 uses a different
24-task transfer population and must not be compared directly with these
selected-task percentages. All percentages are native task scores, not losses.

| Experiment | Change or control | Updates | Failure success | Failure partial credit | Retention success | Main finding |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Automatic sweep, excluded from budget | Whole-action corrected CE 0 or 1 crossed with feedback CE 0, 0.05 or 0.2; reverse KL 1 | 4 per arm | 0% in every arm | Varies by arm | 100% | No arm qualified despite favorable selected gradients |
| 01 | Corrected argument SFT and successful-action retention | 10 | 25% | 42.86% | 100% | Slack recovered; HR partial progress; email tasks failed |
| 02 | Matched 01 plus original reverse KL 1, feedback CE 0.2 and EMA teacher | 10 | 25% | 40.63% | 100% | Same success as SFT; HR privacy violation lowers credit |
| 03 | Zero-update transfer comparison of 64, 01 and 02 | 0 | Separate population | Separate population | Not applicable | SFT 29.17% success versus 20.83% for both 64 and hybrid on previously trained pilot tasks |
| 04 | Teacher and actual minibatch gradient diagnosis | 0 | Not evaluated | Not evaluated | Not evaluated | Teacher preferences can be correct while batch updates oppose individual repairs |
| 05 | First repair token soft forward KL with task gradient balancing | 10 | 0% | 17.19% | 100% | Correct first-token ranks do not ensure correct full arguments; retention loops appear |
| 06 | Full corrected-argument soft forward KL with task balancing | 10 | 50% | 57.14% | 100% | Gmail and Slack recover; no known preservation violations; Salesforce fails |
| 07 | Balanced hard SFT control for 06 | 10 | 50% | 65.63% | 100% | Same success as soft teacher; higher HR credit accompanies privacy violations |
| 08 | 06 plus additional first-divergence token KL | 10 | 0% | 18.75% | 100% | Better measured token margins, worse native calls; reject this weighting |
| 09 | 06 with learning rate 3e-5 instead of 1e-5 | 10 | 50% | 67.86% | 100% | Step5 was better at 62.5% success and 71.43% partial; Salesforce still fails |
| 10 | Reserved transfer of seven fixed candidates including fresh base | 0 | Separate population | Separate population | Not applicable | Primary09 step5 scores45.83% versus58.33% base; not qualified |

The original six-arm automatic sweep is outside the ten-experiment budget.
Its six final outcomes are retained individually below. Every arm has reverse
KL weight1 and four updates; corrected CE applies to the complete corrected
action, unlike the later argument-span controls.

| Corrected action CE weight | Feedback CE weight | Failure success | Failure partial credit | Retention success | Qualified |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0% | 15.63% | 100% | No |
| 0 | 0.05 | 0% | 7.14% | 100% | No |
| 0 | 0.2 | 0% | 15.63% | 100% | No |
| 1 | 0 | 0% | 7.14% | 100% | No |
| 1 | 0.05 | 0% | 15.63% | 100% | No |
| 1 | 0.2 | 0% | 15.63% | 100% | No |

Experiments01–10 each have at most ten updates and a two-hour total deadline.
The seven updating experiments use 70 updates in total. Ten steps mean 160
transition exposures per updating experiment at batch16, not 160 unique
examples: the batch contains eight repair exposures from five distinct contexts
and eight retention exposures. The stopped original campaign remains separate.

Updating variants use microbatch1 with accumulation16, LoRA rank16/alpha32/
dropout0, AdamW with clipping1 and weight decay0, and learning rate1e-5 except
09's3e-5. Soft-teacher variants use EMA0.99; pure SFT controls do not update a
teacher. The ledger records each objective's weights and exact source identity.

## What four native episodes reveal

### Gmail needs the full sender address and a grounded reply

Full-span soft distillation06 and higher-LR09 retrieve the actual message from
`anika.patel@brightpath.example.com`, consume its `thread_4003` result and reply
with the requested meeting confirmation. The earlier first-token-only variant
could prefer a correct initial token while continuing an invented domain and
retrieving nothing. Complete arguments and result consumption explain the
native success more directly than the first-token loss.

### Slack improves once retrieval supplies the announcement

The repaired policy retrieves Nathan Drake's funding announcement and posts its
substance to `all-hands`. In09 step5, a shortened search suffix still retrieves
the actual message. This illustrates why literal argument matching is only a
diagnostic: the environment can accept a semantically sufficient alternative.
Conversely, a teacher-preferred token does not establish correct posting or
content preservation. Native final assertions pass in the inspected episode.

### Salesforce still searches the wrong domain

In09 step5 the first call searches
`from:marcus.rivera@brightpath.example.com`; at step10 it searches
`from:marcus.rivera@example.com`. The verified correction is bound to
`bluemtn.example.com`. The API returns success with an empty message list,
and the policy repeats variations until the turn budget is exhausted. The
contact mobile field remains unchanged.

By step10 the measured correct-versus-original-wrong token margin improves for
all four failures, including Salesforce. A CPU token audit establishes why that
does not certify the native call. The diagnostic conditions on a separate `@`
token and tests the next domain token. The native Salesforce call instead
generates the single token `@example` one position earlier, bypassing the
conditioned repair position. At step5 the action prefix matches and the domain
token is genuinely wrong: `bright` ranks first while the correct token ranks9.

The first native student prompt tokens match the training prompt. That rules
out a different tool schema or user prompt in this comparison, but does not
establish identical generated action prefixes. HR adds a natural-language
preamble, and the first Gmail call at step10 also emits `@example` before later
recovery. The correction objective supervises the changed argument span under
recorded prefixes; it does not cover every new prefix the updated policy can
generate. This is an observed coverage gap, not a measured causal rescue from a
new training intervention. The [token audit](evidence/self-vllm-20261017/adaptive09-action-prefix-audit.json)
retains both prefixes and exact token IDs for every inspected repetition.

### HR recovers the spreadsheet but can fail later schema and privacy checks

In09 step5, an inspected successful HR episode first requests `ss_departures`,
receives the factual error naming `ss_offboard`, retries successfully, sends
the required notifications and updates the relevant rows. Success is scored
even though its native stop label is `max_turns`; goal completion and clean
termination are distinct measurements.

At09 step10, an inspected episode recovers the correct spreadsheet but passes
integer row identifiers to a tool expecting strings, incurs repeated validation
errors, then misses the Diana and scheduled-departure row updates. Its partial
credit is 5/7. The corrected first retrieval alone cannot teach every later
schema-sensitive decision.

Other variants publicly announced Diana Reese's involuntary departure, breaking
a native privacy prohibition. The initially satisfied prohibition becomes a
scored failure when broken. This explains how similar positive-goal completion
can produce different partial-credit denominators. These are observed native
preservation constraints, not a comprehensive harm taxonomy.

## Why lower loss did not reliably improve behavior

The original feedback CE trains the model to predict the observed tool response.
It does not automatically penalize the action that caused an error or an empty
search. Predicting that response more accurately can reduce loss without
improving the next tool argument. The original stopped campaign showed falling
loss alongside deteriorating task success.

The teacher diagnosis finds correct repair tokens ranked first, but shared
parameter updates across the batch can oppose individual repairs. Initial task
gradient norms in05 differed by almost tenfold between HR and Salesforce.
Balancing task norms changes this interaction, but does not guarantee an Adam
update improves every task. Experiments measured both projected directions and
actual post-update token margins rather than treating them as interchangeable.

The initial full-span teacher audit ranks every corrected-span token highest, with
probabilities from about98.2% to nearly100%. Under those teacher-forced corrected
prefixes, forward KL closely resembles hard-label CE. The matched balanced SFT
control reaches the same 50% local success as06. This does not demonstrate a
distinct advantage from soft distillation. It also does not measure freely
generated teacher trajectories.

The first-token anchor08 makes all four recorded margins improve but eliminates
the Gmail and Slack successes. Higher LR09 restores those successes and improves
HR, but step10 falls below step5. Aggregate loss, a selected token margin and
additional steps are each inadequate checkpoint-selection rules on their own.

![Experiment09 repair loss and selected native outcomes](evidence/self-vllm-20261017/adaptive09-loss-and-outcomes.png)

The next algorithm hypothesis would therefore require fresh student action
prefixes and verified complete-call corrections, including the earliest
divergent token and subsequent result consumption. A fresh teacher prompt alone
does not resolve a loss mask that misses an earlier generated branch. This
hypothesis has not been run: the ten-slot budget is exhausted and a new training
experiment requires a separate decision after the reserved comparison.

There is also an input-to-target issue. The three email repair prompts name the
person but do not provide their address. The verified teacher targets provide
exact addresses from the manifest, such as Marcus Rivera's Blue Mountain
domain. This is legitimate privileged teacher information, but an exact-address
target can train fixture-specific recall rather than a general search strategy.
The base's successful name-search broadening on the reserved tasks shows that
those hidden addresses are not always necessary to solve the task. The
[input and target audit](evidence/self-vllm-20261017/adaptive09-supervision-observability.json)
establishes the information difference; it does not prove that replacing the
queries would improve learning. A future matched experiment should test
history-grounded retrieval and error-recovery supervision rather than assuming
that a state-valid exact call is the best transferable target.

## Retention and partial credit require separate checks

Successful task scores do not prove preserved behavior. In05, Mailchimp adds an
idempotent subscriber repeatedly, scores100%, and reaches the turn limit. The
retention batch contains first successful tool transitions rather than terminal
replies, leaving stopping behavior weakly supervised. Experiments06–09 preserve
observed retention stopping, but the data coverage limitation remains.

Partial credit averages fractional completion; success counts wholly completed
episodes. In01 versus06, Gmail changes from0 to100% while HR drops from71.43%
to28.57%. Success doubles from25% to50%, but partial credit rises by only14.29
points. Native check counts and preservation violations must accompany both
macro metrics, because tasks have different numbers of goals.

All compared policies inherit a system message advertising approximately50
tool-using turns while the diagnostic environment enforces8. This shared
mismatch can constrain complex workflows and stopping. It does not establish
the cause of differences between variants, and was not changed mid-comparison.
The local findings apply to this common diagnostic budget.

## Reserved transfer protocol

Experiment10 uses24 tasks frozen before06 outcomes, outside the original32
training tasks,24 development tasks, and every adaptive update example. The
cohort is disjoint from heldout62. Earlier use elsewhere after the Oct11 novelty
audit is not reverified, so this is reserved research transfer rather than a
universal unseen-task claim.

Seven candidates are fixed: original base, damaged64, unbalanced SFT01 step10,
soft06 step10, balanced SFT07 step10, soft09 step10 and primary09 step5.
The primary was selected from local repair/retention and preservation evidence
before reserve scores were available. It is exploratory because no candidate
meets all local qualification criteria. Other reserve comparisons are
descriptive; they must not be used to choose a new winner.

The original base uses a structurally zero-effect additive LoRA adapter: only
finite A/B weights, B exactly zero, no saved bias or extra modules. Exact loaded
state, active-versus-disabled Torch parity, selected-to-exported tensor parity
and actor-to-rollout token probabilities are checked. This avoids labelling
damaged checkpoint64 as a fresh base.

Each candidate runs48 native episodes under the same16K context,2048 reply,
12288 episode-output and8-turn limits, with32 rollout/server slots, non-thinking
greedy decoding and BF16 weights. Native episodes with no trainable tool calls
remain evaluation data. Native errors stop the comparison; no task dropping,
zero-filling, successful-episode pooling or replacement experiment is allowed.

### Final transfer results remain below the original base

All 336 episodes are reconciled, with 24 tasks and two episodes per candidate,
identical initial task states, complete success/partial-credit coverage and no
native evaluation errors. No scores were dropped or replaced. Every candidate
passed actual loaded-state, selected/exported tensor equality and the finite
actor/rollout probe. Maximum probe discrepancy was 0.11493 nats, below the
predeclared 0.15 tolerance; this is a selected-token probe, not a claim of exact
logit equality in every native context.

| Candidate | Success | Absolute success 95% interval | Partial credit | Success delta versus base with paired 95% interval |
| --- | ---: | --- | ---: | --- |
| Original base | 58.33% | 37.50% to79.17% | 78.90% | Reference |
| Goal-augmented SELF checkpoint64 | 33.33% | 16.67% to54.17% | 48.42% | −25.00 points [−41.67, −8.33] |
| Unbalanced SFT01 step10 | 29.17% | 12.50% to50.00% | 43.83% | −29.17 points [−50.00, −12.50] |
| Full-span soft06 step10 | 33.33% | 16.67% to54.17% | 44.66% | −25.00 points [−41.67, −8.33] |
| Balanced SFT07 step10 | 33.33% | 16.67% to54.17% | 44.52% | −25.00 points [−41.67, −8.33] |
| Higher-rate soft09 step10 | 50.00% | 29.17% to70.83% | 63.65% | −8.33 points [−25.00, +8.33] |
| Preselected primary soft09 step5 | 45.83% | 25.00% to66.67% | 58.24% | −12.50 points [−25.00, 0.00] |

The primary's partial credit falls 20.66 points versus the base, paired 95%
interval −36.74 to −6.23 points. It restores 12.50 success points versus damaged64,
interval 0.00 to 25.00, but loses three tasks the base solved and gains no new
fully successful task over the base. Hiver escalation partial progress does
improve from 0 to 43.48%, so this is mixed learning rather than a claim of no
behavioral change.

The higher-rate step10 row is descriptive. It was not selected as the primary
using reserve outcomes, and there is no matched higher-rate SFT control to
attribute its recovery specifically to soft distillation. All intervals resample
tasks while retaining their two greedy repetitions; they do not measure
variation across independent training seeds. The research reserve is smaller
than a final benchmark and has the historical novelty limitation above.

The [complete native audit](evidence/self-vllm-20261017/adaptive10-reserved-audit.json)
retains raw score coverage, candidate hashes, separate absolute/paired intervals
and missing-score identification bounds. With no missing scores, the primary's
gain bounds collapse to −12.50 success points and−20.66 partial-credit points;
these identification bounds are not confidence intervals.

### Preservation and repeated calls also fail the readiness check

Strict initial-state replay identifies 110 initially satisfied, non-excluded
preservation-check instances per candidate. They apply to 24 of 48 episodes,
covering 12 tasks; the Simple tasks have no such applicable checks. There are
zero unknown initial checks. This is native preservation evidence, not a
comprehensive independently labelled harm taxonomy.

| Candidate | Newly satisfied target checks out of306 | Broken preservation checks out of110 | Episodes with a violation out of24 applicable | Exact repeated tool calls |
| --- | ---: | ---: | ---: | ---: |
| Original base | 172 | 4 | 4 | 2 |
| Checkpoint64 | 98 | 10 | 8 | 106 |
| Unbalanced SFT01 step10 | 86 | 10 | 8 | 115 |
| Full-span soft06 step10 | 87 | 10 | 8 | 119 |
| Balanced SFT07 step10 | 88 | 12 | 8 | 120 |
| Higher-rate soft09 step10 | 140 | 7 | 3 | 32 |
| Preselected primary soft09 step5 | 138 | 11 | 7 | 51 |

Target-check totals pool obligations across tasks, so a task with many goals
contributes more checks. Native macro partial credit weights tasks equally and
also penalizes broken preservation checks. Therefore172/306 is not the base's
78.90% partial credit; both measurements are retained to show different aspects
of behavior.

The primary preserves all applicable checks in 17/24 episodes, 70.83%, versus
20/24, 83.33%, for the base. The paired violation-rate change is +12.50 points
with interval −8.33 to 37.50, so this small sample does not establish a precise
increase in general harm risk. It also does not meet an observed non-regression
criterion. The primary adds a forbidden CTO signer to a DocuSign envelope and
creates excluded marketing contacts/subscribers. Some base violations disappear;
the report retains both check counts and episode rates rather than hiding this
tradeoff in one safety number.

Factual empty retrieval responses rise from 18 to 82, and identical repeated empty
responses from 0 to 40, between base and primary. Explicit tool-error responses
are 22 in both. Thus a tool-error count alone misses the main regression. All
candidates have two length-limited generations; checkpoint64/SFT01/soft06/SFT07
also have two pre-generation context rejections each. Those native outcomes stay
in the population. Retry counts are unknown, not assumed zero.

The [assertion summary](evidence/self-vllm-20261017/adaptive10-assertion-summary.json)
links to the strict read-only audit and separates target progress, preservation
applicability and unknowns. Author-excluded checks stay excluded. No scoring
fix, replacement reward or model judge was introduced.

Four inspected reserved episodes show deterioration beyond the original repair
examples. The base is not error-free; its ability to recover is the meaningful
contrast in these cases.

| Reserved task | Original base episode | Checkpoint64 episode |
| --- | --- | --- |
| Create inbound HubSpot contact | Searches for Liam Foster by name, retrieves the RedRock message, creates the contact and completes | Guesses a Broadway Entertainment address and repeats empty retrievals |
| Set Salesforce case priority | Initially submits numeric ID500002 twice, then changes it to string `"500002"` and completes | Repeats the numeric-ID validation error through the turn limit |
| Update Salesforce lead description | An invented Sarah Johnson address fails; it broadens to a name search and retrieves the Crestline message | Repeats an invented `acme.com` address and never updates the lead |
| Reply to inbound HubSpot inquiry | Searches unread mail, retrieves Jordan Blake's inquiry, creates the contact and replies | Searches for unrelated Jeffrey Chen at an invented domain and exhausts the turn budget |

These [native call excerpts](evidence/self-vllm-20261017/adaptive10-controls-qualitative.json)
show loss of error recovery and grounding in observed messages. They do not
identify which training loss component caused each change. The separate matched
controls are needed for that attribution.

## Evidence and reproducibility

The exact model revision is
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Offline execution uses image digest
`sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337`.
The frozen environment revision is
`a0c7378b179e2b267382b8125192876502d82f89`; the reserve task-name digest is
`6e24042a565ad2accb71294556a215771c86c3165399d66394b09c36944a3dd6`.
Every retained capsule has an immutable source manifest. Initial student64 and
saved EMA teacher64, source/export hashes, optimizer update counts, native
traces and call evidence remain separate for every attempt. No external model
or judge is used. Corrected calls derive from verified manifest/native state,
not Luna demonstrations.

Sources:

- [Budget and experiment ledger](evidence/self-vllm-20261017/self-adaptive-experiment-ledger.json)
- [Final Docker states and execution deadlines](evidence/self-vllm-20261017/adaptive-budget-final-inspect.json)
- [Budget compliance and per-experiment runtimes](evidence/self-vllm-20261017/adaptive-budget-compliance.json)
- [Final immutable source and offline network checks](evidence/self-vllm-20261017/adaptive-source-final-inspect.json)
- [All928 native episode identities and source hashes](evidence/self-vllm-20261017/adaptive-global-native-inventory.json)
- [Local audits and native preservation evidence](evidence/self-vllm-20261017/)
- [Inspected step5 and step10 episode calls and failures](evidence/self-vllm-20261017/adaptive09-qualitative-episodes.json)
- [Frozen reserve manifest](evidence/self-vllm-20261017/adaptive-transfer-reserve24.json)
- [Primary selection before reserve outcomes](evidence/self-vllm-20261017/adaptive10-primary-selection.json)
- [Exact candidate adapter hashes](evidence/self-vllm-20261017/adaptive10-candidates.json)
- [Living execution and recovery plan](../plan/self-trl-workstation-experiment.md)

The loss figure is reproduced with an isolated local plotting environment:

    uv run --isolated --no-project --python 3.13 --with matplotlib python scripts/qualification/plot_self_repair_learning.py --evidence docs/research/evidence/self-vllm-20261017 --output docs/research/evidence/self-vllm-20261017/adaptive09-loss-and-outcomes.png

Plotting does not change the pinned training or evaluation environment.

## Decision and next research priorities

Do not restart256-step training, launch heldout62 or start LFM replication with
this setup. The primary fails success and partial-credit improvement versus the
original base, remains locally unqualified, and does not preserve the observed
constraint rate. The final comparison cannot be rescued by selecting another
reserve checkpoint after the fact.

The three priorities for a separately authorized research phase are:

1. Train on fresh student prefixes and verify the complete corrected decision,
   including earlier tokenization branches, rather than qualifying only a
   teacher-forced repair token. Retain a matched hard-SFT control and evaluate
   against a fresh base.
2. Distinguish history-grounded search/error recovery from fixture-specific
   hidden addresses and IDs. Same-state valid calls are necessary evidence, but
   are not sufficient evidence of a transferable supervision target.
3. Expand retention beyond four first successful tool calls to include terminal
   replies, schema recovery, search broadening and diverse tasks. Choose by
   development outcomes and guard checks, not falling loss, and keep a final
   test separate from adaptive decisions.

The ten-slot investigation used 70 optimizer updates and 816 native evaluation
episodes, plus 112 episodes in the excluded automatic sweep. All 928 episode IDs
are distinct. Each of the ten owned containers exited 0, used an offline network,
retained its original source manifest and stayed within its 7200-second execution
deadline; the longest took 24.32 minutes. Total container execution across the
ten experiments was 84.48 minutes. These are execution durations, not a claim
that engineering preparation or the entire research conversation took that long.
All files and failed/superseded attempts remain local. The budget is exhausted;
no additional experiment or full-training restart was submitted.
