# Author 32 Luna-free teacher contexts using DeepSeek Flash

## Purpose / Big Picture

Generate inspectable teacher-only task requirements and constraints through the user-authorized OpenRouter credential. DeepSeek receives the public request, goal/harm contract and available tool definitions; no Luna execution, model trajectory or initial world state is sent. This is context-authoring research, not training or proof of solver improvement. The frozen product baseline is unchanged. Work is confined to the rl-rft worktree on codex/rft-warm-start; no dependency, fork, commit or publication changes are needed.

## Progress

- [x] Verified live dated DeepSeek V4.1 Flash 1017 inventory and existing credential-file presence without printing secrets.
- [x] Implemented allowlisted inputs, exact check-coverage validation and framework CostLedger reservations.
- [x] Ran32 calls, all32 responses retained, all32 exact model/provider identities verified; reported cost USD0.091739508.
- [x] Inspected Finance, HR, Marketing and Sales, plus recruitment; all five reviewed contexts have substantive defects. Saved the complete HTML review and machine-readable source audit.
- [x] Follow-up user request: combined corrected guidance with full actual feedback on four failed experiment03/update1 episodes. Offline candidates, zero new model calls or optimizer updates; complete rendered prompts, original prompts, sampled actions and hashes saved under docs/research/evidence/combined-teacher-four-20261019-r1/.

## Surprises & Discoveries

Only one Support task is source-screened. The balanced cohort therefore uses Finance5, HR5, Marketing5, Operations5, Sales4, Support1 and Simple7. Source screening is distinct from fresh native replay. Typed manifests contain private expected values; the author may interpret these but is instructed to turn them into retrieval requirements rather than answers.

All32 responses passed schema/check-role coverage but7 copied private identifiers. Manual reviews found unavailable read-back capabilities, a hidden target falsely described as public, and an expected note title promoted into an instruction. Prompt instructions alone did not enforce the information boundary. API billing includes cache-adjusted cost; the cost guard conservatively accounted USD0.094715 from token prices, versus reported billed USD0.091739508. All32 report zero reasoning tokens because this finite authoring batch explicitly disabled reasoning.

## Decision Log

The October 19 user authorization supersedes the earlier no-LLM/no-external restriction for this finite authoring batch. Use dated deepseek/deepseek-v4.1-flash-1017 through Fireworks only, no fallback or automatic retry, temperature0, reasoning disabled and maximum8192 output tokens. Reuse the existing concurrency-safe CostLedger with a USD4.99 ceiling and live endpoint prices. No training queue is resumed. Selection prefers the previously balanced cohort then alphabetic eligible tasks, excludes protected evaluation/development/reserve identities and three known broken fixtures. No generated outcomes influence selection.

Follow-up decision: before any training restart, show four complete candidate teacher prompts by replacing only the teacher supplement inside saved historical rendered teacher prefixes. Preserve the real conversation outside that supplement, and preserve all historical hindsight fields as JSON. These four prompts use DeepSeek-derived guidance manually corrected by Codex, explicitly labelled as hand-reviewed prototypes. They do not demonstrate an automatic optimized authoring pipeline. Keep the original sampled action in a separate review file, never inserted into the teacher prefix. Date/author: October19, Codex following direct user request.

## Context and Orientation

scripts/qualification/author_deepseek_task_context.py owns this project-local authoring runner. Its task_input function is a strict four-field allowlist. The source is docs/research/evidence/sdpo-guidance-five-20261019/balanced-task-source-audit.json. luna_ssr_sft.exclusions supplies protected cohorts. Native environment dependencies remain frozen. The named local credential is stored outside the repository; only its runtime value enters the Authorization header.

## Plan of Work

Freeze 32 inputs and exclusion/source hashes, fetch live provider metadata, reserve the conservative UTF8-byte input upper bound plus protocol overhead and maximum output before each dispatch. First task also checks live authentication. Send the remaining tasks with concurrency4, retain request/response/usage/provider IDs and never overwrite an output directory. Validate every check appears exactly once with unchanged role. Manually inspect four contexts for population ownership, retrieval, scope and fabricated effects. Save an evidence report with cost coverage; do not treat schema compliance as semantic qualification.

## Concrete Steps

From /home/hammad/projects/worktrees/rl-rft run:

    .venv/bin/python scripts/qualification/author_deepseek_task_context.py --output docs/research/evidence/deepseek-context32-20261019-r1

The runner prints task names and status only. Read summary.json and four contexts afterward. Use .venv/bin/ruff check on the new runner and targeted pytest for input/validation/cost-boundary behavior before paid execution.

## Validation and Acceptance

Expect32 distinct protected-disjoint task inputs,32 retained paid response receipts or explicit errors, exact source/check provenance and actual billed usage where available. User can inspect each context.json and exact request.json. Four source reviews must explicitly separate remaining defects from usable guidance. No solver reward or student-learning claim is permitted from this batch.

## Idempotence and Recovery

An existing output directory is rejected. No retry after an ambiguous network failure: retain its cost reservation conservatively. Any future repair needs a new identity and explicit remaining budget accounting. Failed outputs stay failed rather than silently rewritten. Leave old GPU experiment clocks and paused automation unchanged.

## Artifacts and Notes

run.json records source SHA, dated model ID, full system prompt and live maximum batch cost. selection.json records all32 task identities and protected-source hashes. summary.json reconciles results, ledger and reported costs. Each task retains its complete request and response without credentials.

## Interfaces and Dependencies

Use existing httpx, standard-library concurrent futures and posttrain.jobs.providers.cost_control.CostLedger/TokenPrices. No package additions. task_input(material) returns only task/public_request/goal_harm_contract/available_tools; validate(context,input) returns structural errors. These are project research helpers, not new public framework APIs.

## Outcomes & Retrospective

Completed32/32 using deepseek/deepseek-v4.1-flash-1017 through Fireworks. Usage171842 input/35968 output tokens,207810 total. Actual reported inference charge USD0.091739508 (about9.17 US cents), aboutUSD0.002867/task. All32 costs present, no retry or orphan probe. The static context batch is NOT qualified for distillation: all five source-reviewed cases require correction. Seven private-ID-copy flags are not an exhaustive defect count, and unreviewed tasks are not passes. No solver episodes or updates were run, so there is no new reward estimate. The next recommended change is to separate private answer bindings from author-visible requirements and make tool observability explicit before another generation batch; this is a recommendation, not an additional submitted run. Historical compact guidance and training results remain separate.

Evidence lives under docs/research/evidence/deepseek-context32-20261019-r1/. context-review.html contains every actual context and original request; source-review.json contains five manual reviews, conservative automatic flags and token/cost totals. Focused tests3passed; Ruff passed. No training restart, commit, pin change or automation resume occurred.

Four-context follow-up: scripts/qualification/assemble_teacher_context_review.py uses reviewed-guidance-four.json plus the saved experiment03 step1 selected-transitions.json and exact retained tokenizer. It validates task-request equality, failed native status, at least one non-excluded failed assertion, check-ID/role coverage, workflow-tool membership and exact decoded historical-feedback binding. It preserves history outside the supplement and actual feedback JSON without omissions. Existing Qwen template rendering can strip earlier reasoning after appending a user message: preserve the recorded teacher rendering instead of assuming the student-prefix string can simply be concatenated. Both the original student prefix and original teacher prefix are retained for inspection.

All four offline candidate inputs plus original sampled actions fit32768 tokens: input counts4743 (content performance),11380 (competitor launch),10391 (email analytics),11695 (contact matching). Token counts use the retained tokenizer, not a live serving qualification. Historical feedback still contains privileged expected-value parameters; its retention is explicit and must not be mistaken for a newly leak-free prompt. Missing manifest signals remain unknown. No teacher inference, gradient check, current-state native requalification or solver reward measurement has occurred. Training remains on hold.

To reproduce into a NEW output directory from this worktree:

    .venv/bin/python scripts/qualification/assemble_teacher_context_review.py --selected docs/research/evidence/sdpo-guidance-five-20261019/experiment03-plain-r3/step1/selected-transitions.json --guidance docs/research/evidence/deepseek-context32-20261019-r1/reviewed-guidance-four.json --output /tmp/combined-teacher-four-new-review

The browser artifact is four-full-teacher-contexts.html. Each task has full-teacher-context.txt, teacher-supplement.txt, actual-feedback.json, provenance.json, historical-teacher-context.txt, original-student-prefix.txt and sampled-action-review-only.txt. Two new composition tests pass; existing three authoring tests remain green. No old training deadline or budget was renewed.


2026-10-19 completed four-context qualification: two distinct one-update diagnostic arms and read-only tool-gradient audit finished. Combined student failure success25% (base25%, control12.5%); partial70.45% (base56.25%, control42.12%), retention100%. Blog regressed and campaign forbidden-token signal still points the wrong way. Not qualified for training restart; automation remains paused. See docs/plan/combined-teacher-context-qualification.md and docs/research/evidence/combined-teacher-qualification-20261019-r1/qualification-verdict.json.
