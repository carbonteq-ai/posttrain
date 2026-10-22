# Distillation research: colleague handoff

Status: 2026-10-22. Start here; the chat is not required. This is private research,
not a qualified training recipe. Source commits preserve the investigation;
historical results retain their original source/runtime identities.

Publication update (2026-10-22, subsequent user authorization): all three handoff
branches are being published to GitHub. Start with the
[main research branch](https://github.com/carbonteq-ai/posttrain/tree/codex/rft-warm-start),
[benchmark repair branch](https://github.com/carbonteq-ai/AutomationBench/tree/codex/fix-six-task-data)
and [environment repair branch](https://github.com/carbonteq-ai/verifiers-environments/tree/codex/distillation-handoff-env-20261022).
Earlier local-only/no-push entries below and in historical plans describe the
initial handoff snapshot; this update supersedes that publication restriction.
Publishing source does not approve training, release a package or change pins.

## Objective and current answer

We want Qwen3.5-4B to learn better tool decisions from its own trajectories. The
student receives ordinary task instructions. A teacher initialized from the same
model receives additional task requirements and factual feedback on the student's
attempt. Distillation transfers the teacher's conditional token distribution to
the student without giving the student privileged context at inference time.

The current question is narrower: **can concise, correctly localized requirements
and failure feedback make the teacher recommend the right action AND make the
actual distillation gradient suppress the wrong action?** Neither teacher prose
quality nor falling loss alone establishes that result.

**Training is blocked.** The latest 124-task focused-context qualification has
zero optimizer updates and `training_approved: false`. Runtime and source
reconstruction passed, but teacher mistakes and inconsistent corrective gradient
directions remain. All old GPU windows have expired; this handoff authorizes no
new jobs. Earlier SELF/full-training/LFM queues must not resume automatically.

SELF below denotes the earlier feedback-prediction/distillation investigation;
SDPO denotes the later teacher-conditioned distribution-matching implementation.
Corrected-call CE/SFT hybrids are explicitly supervised variants, not evidence
that plain SELF worked. Generated reasoning text is not proof of faithful reasoning.

## Five most informative prior studies

These are reference studies, not five proven winners. Compare each student with
its OWN control. Different modes, task sets and environments prevent a global
causal ranking. Success is native task success, not `agent_completed` or training
credit. Percentage-point intervals below are paired task-bootstrap 95% intervals.

| Study / student checkpoint | Population and own control | Student outcome | Interpretation |
| --- | --- | --- | --- |
| Guidance-five experiment 03, step 10 | Thinking; 20 training tasks; heldout62, all 310 scored | Success **58.06 → 61.61%**, delta +3.55 points, CI [-2.26, +10.00]. Partial **82.47 → 84.37%**, delta +1.89, CI [+0.18, +3.98] | Highest completed heldout student percentage among these studies. Preservation violations rose 36→44/140 applicable episodes; API errors 151→270; length limits 5→11. Not a clean overall gain. |
| Pool124 batch32 campaign, step 10 | Non-thinking; full 124 pool; matched heldout62, 310 scored/arm | Success **51.61 → 58.06%**, delta +6.45, CI [+1.61, +12.90]. Partial **81.19 → 80.77%**, delta -0.43, CI [-4.13, +3.05] | Strongest own-control success gain here, with partial-credit tradeoff. Step20: 54.84% success/83.51% partial; step30: 56.45%/81.54%. Run crashed after 38 updates; no step40/50 readouts. Step10 is a retrospective descriptive peak. |
| SDPO08 cosine schedule, step20 | 8 training tasks, 4 development tasks; non-thinking | Train success 87.5→87.5%, partial 91.07→94.64%; development success **50→50%**, partial **50→50%** | No demonstrated development success gain. Earlier 25→50 comparisons were parser-confounded. Useful schedule/control evidence, not a benchmark winner. |
| SDPO10 pool124 thinking, step6 | Full pool; heldout62, all310 scored | Success **58.06 → 54.84%**, CI for delta [-9.35,+1.29]; partial **83.34 → 80.52%**, CI [-7.24,+0.93] | Thinking did not establish improvement. Preservation-affected episodes rose45→50/140. Do not pool with incomplete Luna/non-thinking attempts. |
| SELF adaptive09, preselected step5 | Supervised soft repaired-span hybrid initialized from damaged checkpoint64; reserved24 tasks×2 | Original base **58.33% success/78.90% partial**; damaged checkpoint33.33%/48.42%; selected student **45.83%/58.24%**. Versus original base: success -12.5 points CI[-25,0]; partial -20.66 CI[-36.74,-6.23] | Local four-failure success reached62.5%, retention100%, but transfer stayed below original base. Rescuing a damaged policy is not improving the base model. |

Primary source paths after evidence restoration:

1. `docs/research/evidence/sdpo-guidance-five-20261019/experiment03-plain-r3/`:
   `postrun-audit.json`, `postrun-guards.json`, `shared-base-guards.json`,
   `checkpoint10-independent-audit.json`. Plan: [guidance-five](../../plan/sdpo-guidance-five-mtp.md).
   Run `sdpo-qwen35-4b-guidance-five-exp03-20261019-r3`; source manifest
   `900e40121f01b4c339cc7d4fbc46b30243f5a5746f1725f48c47572a83377d7a`.
2. `docs/research/evidence/sdpo-pool124-20261018/`:
   `checkpoint10-comparison-summary.json`, corresponding20/30 files,
   `full-run-terminal38-native-audit-r3.json`. Plan: [pool124](../../plan/sdpo-pool124-workstation.md).
   Run `sdpo-qwen35-4b-pool124-b32-50-20261018-r1`; source manifest
   `9ce6cab7ab3afacf24d5681bcc1acfff1166f1ee827b064ed60b5e3c749bde8f`.
3. `docs/research/evidence/sdpo-vllm-20261018/unified-reviewed-results.json`,
   slot8; `experiment08-outcome-curve.json`, `experiment08-four-episode-review.json`.
   Run `sdpo-qwen35-4b-agent08-cosine20-20261018-r1`.
4. Same unified file, slot10; `experiment10-audit.json`.
   Run `sdpo-qwen35-4b-agent10-pool124-thinking-20261018-r5`.
   Plans for3/4: [SDPO study](../../plan/sdpo-workstation-ablation.md).
5. `docs/research/evidence/self-vllm-20261017/adaptive10-reserved-audit.json`,
   `adaptive10-primary-selection.json`, `adaptive10-selection-before-outcomes.json`,
   `adaptive09-*.json`. Narrative: [SELF results](../self-adaptive-ablation-results-2026-10-17.md).
   Run `self-qwen35-4b-adaptive09-lr3e5-20261017-r1`.

Experiment03 trained10 verified optimizer/EMA updates with496 Adam states at
moment10. Its student adapter SHA256 is
`ced7c07f68366082015475d90e700f2d6aa7ccb1aa16fe361f3f9f3dd02b795d`;
teacher adapter `bc77f513d132c9dff5c93c203389fc2b295ddbf0fbf978b59fd16d1e00ce5d82`.
Native artifacts and checkpoint receipts take precedence over stale intermediate
ledger statuses. Repeated inspection of heldout results limits final-validation
claims; none of these retrospective selections is a new independent test.

## Current focused teacher-context workflow

1. Keep the intended **124 training tasks / separate heldout62** split. Previous
   training use alone does not exclude a training task. Additional development or
   reserve exclusions need individual justification; do not inherit every old
   small-pilot exclusion automatically. Historical novelty after Oct11 is unknown.
2. Bind each task to its exact manifest, public request, native initial state and
   tool contracts. Fix environment defects in their owning fork; do not quarantine
   tasks or weaken assertions to obtain a pass.
3. Reconstruct a real student attempt and preserve its original action token IDs,
   prefix and tool observations. Localize the earliest supported failing decision,
   rather than selecting an unrelated successful read from a failed episode.
4. Build a focused teacher-only context: operation requirements, relevant policy
   already retrieved, observed resource handles, and precise failure facts.
   Do not dump all manifests or leak hidden answers/IDs. Preserve scope, timing,
   cardinality and population ownership. Unsupported business-policy inference
   remains unsupported, even with a deterministic compiler.
5. Replay actual teacher calls against the native environment, audit their effects
   and final assertions, then measure corrective directions on the **same original
   student tokens**. A positive whole-reference projection is not proof that
   a forbidden value is being suppressed.
6. Only after teacher quality, actual loss/Adam behavior and retention qualify,
   integrate the approved path into a source-bound training runner. Training and
   heldout evaluation need fresh approved windows and immutable receipts.

DeepSeek was an explicitly authorized earlier preparation comparison. It is not
used by the latest deterministic focused qualification. Luna traces are historical
research inputs; the current focused teacher path does not call Luna, DeepSeek or
an external judge. Do not silently add external preparation or gold-call CE.

### What the latest qualification actually found

Start with `docs/research/evidence/focused-pool-qualification-20261020-r1/qualification-gate.json`
and its `pool-context-review.html` (124 primary cases plus10 no-copy cases).
The gate rejects both teacher quality and corrective signal. Runtime qualification
means124 complete replies/source-bound replay pairs, not124 correct tool calls.
Whole-reference projections were positive for94 tasks and nonpositive for30.
Teacher API errors remain in recruitment analytics, social content calendar,
DocuSign NDA, fire-suppression testing and Freshdesk journey mapping.

Localization retained61 successful-episode decisions, identified25 earliest
linked prohibited writes,18 late omissions and5 direct errors;15 failures remain
unlocalized. Structural coverage of762 manifest checks is not semantic translation
of every arbitrary predicate. No task was removed to improve these counts.

Two concrete lessons:

- Including the full failed candidate prose caused the teacher to copy fabricated
  marketing percentages47/62/78 that were absent from the original history.
  Omitting candidate prose removed these copies, but introduced placeholder email
  content, wrong destinations/date windows and caption-versus-ID errors.
- Targeted no-copy probes improved HR timing-token suppression, but Marketing's
  bad-token projection remained positive and Sales was tiny/inconclusive. Teacher
  log-probability preference can oppose a bad token while the total-sequence
  gradient still favors it. These are zero-update infinitesimal-SGD diagnostics,
  not actual Adam learning or reward improvements.

The exact r7 probes retained original token IDs, seed18, dropout disabled and the
initial adapter. HR projection changed +0.00024069→-0.00043321;
Marketing +0.00096689→+0.00046276; Sales differing-ID digit
+0.00000435→+0.00001167. Positive here favors the selected bad-token score;
negative suppresses it. Magnitudes vary slightly across repeated numeric probes.

### Important integration gap

`scripts/qualification/progress_corpus_training.py` still imports the older
`progress_teacher_context.compose_corpus` / `semantic_selection` path. It does
**not** yet train with the new focused builder/localizer. Its source and gate
schema expect the older corpus-v6 qualification. Do not feed the new contexts
to the old launcher and describe them as qualified training inputs.

The next implementation should deliberately connect the focused builder and
selector, bind exact builder/context/native/gradient hashes, audit all124 tasks,
and verify original student prefixes plus no truncation before a training launch.
First diagnose loss contributions causing the Marketing/Sales interference;
do not assume a prettier context fixes the loss. Current plan:
[progress qualification](../../plan/progress-teacher-ten-qualification.md).

## Code and artifact navigation

- `scripts/qualification/`: private research runners, context builders, replay,
  localization, gradient diagnostics, analyzers, launchers and focused CPU tests.
  Search `focused`, `localiz`, `progress_teacher`, `sdpo`, `self` in filenames.
- `docs/plan/`: living experiment protocols, budget/authorization history, exact
  commands, source manifests and decisions. Latest dated entries override early
  intentions; plans are not completion receipts.
- `docs/research/`: analytical reports, ledgers and restored interactive examples.
- `apps/lab/.posttrain/`: historical model/catalog/work-package bindings; local
  environment variants are explicitly local, not published dependencies.
- Environment ownership is external: benchmark task data in AutomationBench,
  compatibility/adapter repairs in verifiers-environments; see fork receipt below.

The user's comparison page is
`docs/research/evidence/focused-deterministic-two-20261020-r1/comparison.html`.
Restore it before opening on a fresh clone. HTML files load their local assets;
retain the restored directory layout. Native traces remain the authority, with
original missing scores/errors; do not silently rescore old results.

## Restore the evidence and run CPU checks

From this repository root, with Python3.12+:

```bash
python scripts/qualification/materialize_experiment_evidence.py --list
python scripts/qualification/materialize_experiment_evidence.py --verify
python scripts/qualification/materialize_experiment_evidence.py --extract
```

Use `--study focused-pool-qualification-20261020-r1` to restore one study (repeat
the option for more). All4863 archived files retain their SHA256 and original
path:1,452,057,716 bytes compressed to186,497,921 bytes in27 archives. Raw originals
were not deleted. Extraction is idempotent and refuses to overwrite changed files;
it rejects unexpected members, traversal and escaping symlinks. Exact archived raw
directories are ignored in Git to avoid committing duplicate gigabytes. Future
evidence is not broadly ignored. The manifest is in `docs/research/experiment-archives/`.

```bash
PYTHONPATH=scripts/qualification .venv/bin/python -m pytest -q scripts/qualification/test_materialize_experiment_evidence.py
.venv/bin/python -m pytest -q apps/lab/tests/test_entry.py apps/lab/tests/test_lfm_moe_smoke.py packages/serve/tests/test_vllm_bindings.py packages/execution-buildkit/tests/test_environment_wheels.py
```

See `validation/` for actual test logs. The research sweep recorded521 passed,
1 skipped and8 collection errors from missing optional AutomationBench/Verifiers/
vLLM dependencies in the local environment. This is **not a clean full-suite pass**.
The changed framework subset passed49 tests; benchmark fixture/Jira subset65;
environment repair subset24. No GPU rerun was performed for this handoff.

## Runtime, lineage and colleague access

Historical model: `Qwen/Qwen3.5-4B`, BF16 revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; initial LoRA adapter SHA256
`e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c`.
Historical selected TRL source was
`290c03d8e1553db509b0d033907f60fc1cb47b82`, file SHA256
`530b571c0001bbf85bb0c78725d32645a4d7155660e22e41580fab5c9f7041e4`.
The clean sibling TRL checkout at`c9af78c1c2ea04ad271e95b26b93dfadf8b9fca1`
is not a substitute for that run-selected source. No dependency pins were updated.

Workstation `dstack@192.168.110.12` / `carbonteq-ai-workstation.lan` retains
immutable capsules under `/var/lib/posttrain/qualifications/`. In particular,
`self-campaign-2048-20261017-r5` contains the historical runtime/TRL/kernel
dependencies; source alone does not recreate this cached runtime. The source-bound
environment capsule `progress-teacher-124-20261020-r5` has source manifest
`4eaf090a4201f0d14b3c6efcb0c4b07541ffcc4e47224cc9f14fca427c019f85`.
Image:
`registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337`.
Model weights, adapters, private package/image access, SSH credentials and large
compiled runtime libraries are **not** included in Git. Arrange access separately;
never copy credentials into the handoff. Preserve unrelated workstation workloads.

The proposed future profile was10 updates,32 distinct tasks/update,
microbatch1/accumulation32, LoRA16/alpha32/dropout0, LR1e-6, top20+tail reverseKL1,
EMA update.01, thinking teacher/student,32K context/4K reply/12K episode output,
8 turns,40 rollout/server slots,16GiB KV. This is a **proposal behind a failed
gate**, not an accepted run. Exclude whole incomplete/overlong training transitions;
never truncate them to fit. Matched heldout evaluation keeps62×5 episodes,
64K context/8192 reply/32768 episode output/20 turns and the same inference mode.
Frozen heldout source `a0c7378b179e2b267382b8125192876502d82f89`, ordered-task
SHA256 `ed81cf1fe88f68bb8420bb46c5fdf3afde24660c7043812da680bb1c7bc9196f`.

## Ownership and safe next steps

Source-only fork commits and GitHub branch links are listed in `source-receipt.json`. Benchmark
fixtures must be committed before the matching vendored environment changes;
consumer pins remain unchanged until release qualification.
These commits do not retroactively change any historical result's environment.

1. Restore evidence and inspect four current failure contexts plus native effects.
2. Resolve corrective-loss interference and remaining localization/teacher errors.
3. Integrate the focused path into the training runner with explicit source/gate
   compatibility, then qualify the actual path and retention on the full pool.
4. Request a fresh runtime budget before any new GPU work. Inspect real GPU,
   retained container identities and launch receipts to avoid duplicates.
5. Run matched training/evaluation only if gates pass; report success/partial,
   harms/preservation, errors, exclusions and paired uncertainty together.

Transfer uses verified Git bundles outside this repository; their manifest records
the exact main and fork commits. Ali Safdar received the initial handoff summary
and local transfer details in Mattermost; the GitHub branches now provide remote
source/evidence access under the subsequent user authorization.
The initial bundles are an offline snapshot; GitHub branch tips include subsequent
publication documentation. Source and evidence are portable; live GPU reproducibility still needs the private
runtime access and qualification described above.
