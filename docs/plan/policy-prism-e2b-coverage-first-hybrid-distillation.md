# Build and qualify coverage-first hybrid distillation for Policy Prism

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current while executing it. Maintain it according to `docs/templates/PLAN.md`.

## Purpose / Big Picture

The completed Gemma 4 E2B from 12B IW-OPD experiment transferred a useful but narrow behavior: the student became more conservative and its returned rules were usually well supported, yet it omitted more required rules in scope interpretation and its recovery gain was concentrated in a small number of cases. The broader goal is different. The student must improve both complete legal-scope interpretation and exhaustive rule recovery, and those improvements must transfer across prompt wording, regulatory families, source lengths, and evidence positions.

This plan builds the next experiment around that goal. First, Gemma 4 E2B learns complete, independently validated scope and recovery trajectories through supervised fine-tuning. It then continues from that checkpoint with full-distribution on-policy distillation from Gemma 4 12B while complete expert examples remain in every update. Full-distribution means the loss compares all grammar-allowed next-token choices, not only the one token sampled by the student. The retained expert examples supply correct later states that an on-policy student never visits when it ends an answer too early.

The result is observable in four places. PostTrain exposes a typed `train.hybrid-distill` job with reproducible input and evidence contracts. Policy Prism packages a benchmark-independent capability curriculum and non-sealed selection evaluations. The accepted private LoRA adapter is published at an immutable Hugging Face revision. The final sealed scope and recovery evidence is written in Policy Prism's existing five-file `evaluation-runs` format and compared with base E2B non-thinking, the completed IW-OPD model, and Gemma 4 12B.

The plan does not promise that a particular loss will improve the model. It creates a controlled experiment that can establish that result without conflating model behavior with prompt, serving, tracking, or evaluation defects.

## Launch Verdict

Do not start another long training run from the completed IW-OPD recipe. Its runtime is valid, but its sampled-token objective and scope-only, selected-stage curriculum do not directly supervise missing rules or complete cross-stage behavior.

The next experiment may begin only after four implementation gates pass:

1. Amend the frozen PostTrain product baseline to introduce `train.hybrid-distill`; do not silently change the meaning of existing `train.distill`.
2. Add a memory-safe, generation-constrained, full-distribution loss that supports different student and teacher chat templates while preserving aligned completion positions.
3. Build and validate a new Policy Prism curriculum containing complete scope and recovery trajectories, independent training prompt profiles, output-cardinality balance, non-sealed family isolation, and complete expert targets.
4. Pass the offline numerical, schema, data-lineage, prompt, and checkpoint tests plus one production-shaped long-output integration canary. Do not repeat throughput, MTP, cache, batch, or teacher-strength sweeps that the completed experiment already settled.

The completed geometry `logical batch 12 / physical batch 1 / gradient accumulation 12` remains the serving and update baseline unless the new full-distribution loss cannot fit. Memory adaptation should first reduce vocabulary/position chunk size; it must not silently alter the scientific batch, loss, or curriculum.

## Progress

- [x] (2026-08-18) Complete and reconcile the 384-target, 32-update E2B from 12B IW-OPD experiment with zero teacher failures and complete native evidence.
- [x] (2026-08-18) Publish and byte-verify the private rank-16 IW-OPD adapter, then finalize its sealed scope and recovery evaluations.
- [x] (2026-08-19) Compare base E2B non-thinking, final E2B IW-OPD, and Gemma 4 12B across every available business KPI, engineering metric, Claude diagnostic, and retained model trace.
- [x] (2026-08-19) Identify the distinction between broad recovery improvement and case-concentrated micro improvement, including the 148-rule Ohio case.
- [x] (2026-08-19) Reconcile the behavioral evidence with the actual IW-OPD objective, position weighting, selected-stage curriculum, prompt profiles, constrained serving path, and current TRL implementations.
- [x] (2026-08-19) Review primary research on sequence distillation, on-policy distillation, position bias, student-teacher mismatch, KL direction, prompt transfer, and legal information extraction.
- [x] (2026-08-19) Freeze the next-experiment decision: coverage-first sequence SFT followed by constrained full-distribution OPD with continued expert replay; compare only balanced JSD and forward KL.
- [x] (2026-08-19) Write this self-contained implementation and experiment plan without modifying either completed evaluation or historical IW-OPD evidence.
- [ ] Create clean feature branches at the immutable bases recorded under `Repositories and immutable starting points`.
- [ ] Amend the canonical product baseline and add the new public operation, seats, settings, evidence, and telemetry contracts.
- [ ] Implement and release the generic constrained full-distribution and mixed-expert trainer behavior in the CarbonTeq TRL fork.
- [ ] Implement PostTrain's `train.hybrid-distill` adapter, catalog definitions, validation, packaging, checkpoint, and Observatory support.
- [ ] Implement the Policy Prism capability curriculum, expert-data preparation, non-sealed selection environments, and completion validators.
- [ ] Generate and independently validate the first 256-source expert dataset without sealed-family or answer leakage.
- [ ] Run the common Stage 1 sequence-SFT job and verify its adapter, checkpoints, training summary, and expert-data lineage.
- [ ] Run the two controlled Stage 2 branches: balanced JSD and forward KL, both starting from the exact same Stage 1 checkpoint.
- [ ] Select one candidate using only predeclared non-sealed development and transfer gates.
- [ ] Run sealed scope and recovery exactly once on the selected candidate, publish the accepted adapter, and finalize the five-file evaluation evidence.
- [ ] Record an explicit `accept`, `revise`, or `reject` decision and update this plan's retrospective with exact run, artifact, model, source, and image identities.

## Surprises & Discoveries

- Observation: there is no Gemma 4 E3B result in the finalized evidence.
  Evidence: the model named informally as “E3B IW-OPD” is the final Gemma 4 E2B adapter `gemma-4-e2b-policy-prism-scope-iwopd-r16-from-12b-v1`.

- Observation: the final IW-OPD run is mechanically valid and its scientific regression is real.
  Evidence: training consumed 384 unique targets in 32 finite updates with zero teacher failures. The completion gate verified model-native E2B and 12B prompts, identical per-position XGrammar allowed sets, aligned completion-token digests, finite loss and gradients, checkpoint state, and ledger state. Final scope and recovery produced 18/18 and 17/17 traces with all model stages ending normally, complete Claude packets, consistent reconciliation, and no selected truncation or whitespace loop.

- Observation: scope became more precise on the rules it returned but substantially less complete.
  Evidence: returned-rule source support rose from `45/49` for base E2B to `34/35` for IW-OPD, while expected-rule matches fell from `40/68` to `28/68`. Full-rule micro precision remained close (`0.524` to `0.514`) while micro recall fell from `0.378` to `0.264`. The model returned only 35 rules instead of 49.

- Observation: scope regression is concentrated in early, valid closure rather than invalid output.
  Evidence: on the X-ray, California building, and Ohio public-records scope cases, matched inventories changed from `5/7`, `6/7`, and `11/11` to `1/1`, `1/1`, and `8/8`. Each IW-OPD response was valid JSON with `finish_reason=stop`. It chose a smaller inventory and closed it normally.

- Observation: recovery's large pooled gain is not a broad gain of the same magnitude.
  Evidence: expected-rule matches rose from `223/460` to `337/460`, but one 148-rule Ohio case contributed 108 of the 114 added matches. Excluding that case, base E2B recovers `223/312` and IW-OPD `229/312`, a six-rule difference. Macro recall moves only from `0.639` to `0.650`.

- Observation: recovery still contains useful transfer, but it is uneven and incomplete.
  Evidence: required text rises from `238/376` to `376/376`, preserved legal meaning from `87/460` to `178/460`, and source assignment from `83/460` to `147/460`. At the same time, hard-gate pass rate falls from `41.18%` to `35.29%`, false positives per provision rise, and one compact 42 CFR case collapses from six matched rules to none.

- Observation: conditional improvements and end-to-end regressions can coexist without a benchmark bug.
  Evidence: node-span F1 and legal-detail error improve on the smaller set of surviving scope rules, while candidate count and recall fall. These conditional metrics answer “how good are the aligned rules that remain”; inventory metrics answer “did the model find every rule.” The model optimized the first behavior at the expense of the second.

- Observation: scope and recovery are not interchangeable tasks.
  Evidence: scope requires evidence segmentation, detailed legal effects, actors, conditions, exceptions, ambiguity resolution, qualifications, and a graph. Recovery supplies provision context and asks for a compact inventory. A local improvement in compact rule decomposition can therefore appear in recovery without transferring to complete scope interpretation.

- Observation: the completed objective could refine sampled behavior but could not directly teach an omitted continuation.
  Evidence: IW-OPD scored exact student-sampled tokens. If the student emitted one rule and then selected the JSON-closing token, rules it never emitted had no positive token target. The teacher could adjust the sampled rule and closing token, but it did not generate a replacement inventory for the loss.

- Observation: early-token weighting was poorly matched to set completion.
  Evidence: the completed objective used position weight `gamma=0.5`, emphasizing early output. For an inventory, later rules are separate required set members rather than disposable reasoning tails. The observed pattern—correct structure, plausible early rules, then early closure—is the expected risk of this weighting.

- Observation: the curriculum mainly optimized local rule emission rather than complete legal workflows.
  Evidence: 230 of 384 targets were rules, 179 of those were standalone, only 128 tasks were full trajectories, and only the selected stage received loss. The experiment was scope-only during training even though final qualification measured both scope and recovery. Evidence, rule, graph, and recovery completeness therefore did not receive a balanced joint signal.

- Observation: source-length balance did not guarantee output-inventory balance.
  Evidence: the prior plan balanced input quartiles but included inventories ranging from compact cases to more than one hundred rules. The Ohio recovery case could dominate a pooled metric and long inventories could dominate tokens, while prompt/source length appeared balanced.

- Observation: about forty percent of training targets used the exact benchmark prompt profile.
  Evidence: 154 of 384 targets used the canonical benchmark profile. Sealed source families stayed excluded, so this was not answer leakage, but it weakens the claim that improvement transfers independently of benchmark wording. The next curriculum keeps sealed prompt text evaluation-only.

- Observation: prompt and serving differences are comparison limitations, but they do not explain the final behavior.
  Evidence: base E2B and 12B historical runs used raw YAML prompt contracts and an older RunPod serving path, while final IW-OPD used semantic-only rendering and bounded-whitespace managed vLLM. Their compatibility hashes differ. Nevertheless, the retained final traces show normal stops and complete structured outputs; the main conclusions come from content-level omissions, not latency or transport comparisons.

- Observation: latency is not a controlled model-quality metric in the three-model comparison.
  Evidence: hardware, vLLM versions, prompt rendering, output length, and serving configuration differ. Keep latency in the evidence tables, but do not use it to infer that a model is more capable.

- Observation: no evaluated model solved qualification preservation or graph composition.
  Evidence: scope qualifications remain `0/44` for base and IW-OPD and `0/53` for 12B. Scope relationship attachments are `1/116`, `1/116`, and `9/122`. The next curriculum must target complete trajectories rather than treating rules alone as a proxy for the full capability.

- Observation: the strongest available teacher does not automatically create the strongest student.
  Evidence: Gemma 4 12B remains far above E2B IW-OPD on scope hard gates, full-rule correspondence, relationships, semantic coverage, and pipeline score. Capacity gap and student-state mismatch limit what an on-policy student can absorb from a larger teacher.

- Observation: full-distribution OPD is necessary for the missing-choice problem but insufficient by itself.
  Evidence: at a reached prefix it can put probability mass on every grammar-allowed alternative, including “start another rule” instead of “close JSON.” Once the student closes, however, no later prefix exists. Expert replay is therefore required to expose the student to the complete later states and correct cross-stage dependencies.

- Observation: the maintained TRL fork already contains the two algorithm families, but not the required combined product.
  Evidence: `trl.experimental.iw_opd.IWOPDTrainer` owns the sampled-token IW-OPD path. `trl.experimental.distillation.DistillationTrainer` computes full-vocabulary generalized JSD/KL over student-generated completions. The latter does not mix retained expert trajectories into the same update, does not implement the Policy-specific different-template constrained loss, and would materialize unsafe sequence-by-vocabulary logits for this workload if used unchanged.

## Decision Log

- Decision: preserve the completed `docs/plan/policy-prism-gemma4-e2b-12b-opd.md` as historical evidence and use this new plan for the next experiment.
  Rationale: rewriting the completed plan would make the prior run difficult to reproduce and would mix an observed result with a future hypothesis.
  Date/Author: 2026-08-19 / Codex.

- Decision: introduce `train.hybrid-distill` instead of broadening `train.distill` silently.
  Rationale: the frozen baseline defines `train.distill` as fresh, consume-once, fully on-policy student trajectories and explicitly places stored expert completions under data generation or SFT. The new job consumes both fresh student trajectories and an immutable expert dataset, so it has different seats, semantics, metrics, and evidence.
  Date/Author: 2026-08-19 / Codex.

- Decision: use two training stages with one common supervised checkpoint.
  Rationale: Stage 1 teaches complete legal answers and later completion states. Stage 2 adapts behavior on the student's own prefixes without allowing the model to forget those complete examples. This directly addresses the observed combination of high conditional precision and low coverage.
  Date/Author: 2026-08-19 / Codex.

- Decision: compare balanced Jensen-Shannon divergence and forward KL only.
  Rationale: balanced JSD interpolates student and teacher distributions and is the conservative primary candidate. Forward KL places stronger pressure on teacher-supported alternatives, including missing continuations. Reverse-KL-style and sampled-token IW-OPD objectives favor the student's existing modes and are not the appropriate primary comparison after the observed omission behavior.
  Date/Author: 2026-08-19 / Codex.

- Decision: retain complete expert examples during Stage 2 with initial loss weights `2/3 expert cross-entropy` and `1/3 on-policy divergence`.
  Rationale: this is a predeclared experimental starting ratio, not a claimed optimum. It makes coverage preservation the dominant signal while retaining meaningful student-state adaptation. Each component is normalized and logged independently so a future revision can change the ratio based on evidence rather than hidden scale differences.
  Date/Author: 2026-08-19 / Codex.

- Decision: remove position decay from the new full-distribution objective.
  Rationale: all rules and all stages are required. Use uniform token weighting within a completion, then equalize logical source contributions so early tokens and very long inventories do not silently dominate.
  Date/Author: 2026-08-19 / Codex.

- Decision: use complete scope and recovery trajectories in both the expert dataset and on-policy environment.
  Rationale: recovery improvement did not reliably transfer to scope, and scope-only selected-stage optimization did not train the end-to-end behavior. The next experiment must optimize the actual two capabilities rather than infer one from the other.
  Date/Author: 2026-08-19 / Codex.

- Decision: keep exact sealed benchmark prompts and families out of training and method selection.
  Rationale: the experiment must measure capability transfer, not familiarity with the evaluation wording. Four independently authored training profiles and one held-out profile cover prompt variation; sealed v9/v10/v6 prompts remain qualification-only.
  Date/Author: 2026-08-19 / Codex.

- Decision: begin with the predeclared 256-source rung and scale only after broad non-sealed transfer.
  Rationale: 256 sources was the first rung of the original capability plan. It is large enough to cover both tangents and required strata while keeping a two-objective comparison affordable. Increasing to 512, 1,000, or 2,000 before selecting a method would multiply cost without resolving the objective question.
  Date/Author: 2026-08-19 / Codex.

- Decision: use a stronger sequence-answer teacher and Gemma 4 12B as the probability teacher.
  Rationale: complete supervised targets should come from the strongest independently validated answer generator available in the finalized model evidence, with deterministic checks and review. Stage 2 retains Gemma 4 12B because it is a strong local model with the same ordered token mapping as E2B and its exact native prompt alignment is already understood. Raw teacher output is never treated as Gold without validation.
  Date/Author: 2026-08-19 / Codex.

- Decision: keep the qualified runtime geometry and runtime safety invariants.
  Rationale: `12/1/12` was the fastest sustained geometry. Model-native prompts, generation-only wire schemas, identical constrained probability spaces, bounded whitespace, output-feasible task plans, retries, capacity-aware reserves, concurrency-safe ledgers, and idempotent checkpoints are proven requirements. The new method changes the learning signal, not those protections.
  Date/Author: 2026-08-19 / Codex.

- Decision: use one production-shaped GPU canary after exhaustive offline validation.
  Rationale: another broad serving sweep would repeat settled work. The remaining live uncertainty is whether the memory-safe full-distribution plus expert-replay update fits and produces finite gradients on the longest admitted trajectories. One fixed canary should cover that boundary.
  Date/Author: 2026-08-19 / Codex.

- Decision: choose the method on non-sealed development and transfer evidence, then run sealed qualification once.
  Rationale: choosing between checkpoints or losses with sealed results leaks qualification evidence. The sealed benchmark is a final decision gate, not a tuning set.
  Date/Author: 2026-08-19 / Codex.

- Decision: keep existing Policy Prism business KPI definitions unchanged.
  Rationale: the user explicitly deferred KPI redesign. The plan may add training diagnostics and non-sealed selection summaries, but it must not rewrite prior KPIs or traces.
  Date/Author: 2026-08-19 / Codex.

## Outcomes & Retrospective

The design milestone is complete. This document reconciles the completed run, metric behavior, trace evidence, prompt curriculum, runtime implementation, and relevant distillation research into one next-experiment contract. No new training job has started and no prior evidence has been altered.

The completed experiment established three durable facts. First, the corrected E2B-from-12B constrained runtime works end to end. Second, sampled-token IW-OPD can improve conditional rule quality and a narrow recovery behavior without transferring complete scope interpretation. Third, a pooled recovery gain can hide substantial case concentration, so future model selection must require both micro and macro improvement across non-sealed families.

The next retrospective entry must record whether Stage 1 improves complete coverage, whether JSD or forward KL preserves that coverage while improving student-state behavior, whether gains transfer to the held-out prompt and family cells, and whether the accepted candidate improves both sealed tangents. If neither branch passes, retain all evidence and record `reject`; do not publish a model as improved merely because training succeeded.

## Context and Orientation

### Repositories and immutable starting points

Three repositories participate. Do not modify any `main` or `develop` branch directly.

`/home/ali-awais-safdar/Post-Train/posttrain` is the framework repository. The completed IW-OPD feature branch is `feat/gemma-policy-prism-opd-e2b-12b` at `8682aac`. It contains the corrected runtime, public materialization flow, completion validation integration, and historical plan. Create the next branch `feat/policy-prism-e2b-coverage-first-hybrid-distill` from that commit unless those changes have reached a newer audited release. If starting from a newer release, merge or port every invariant listed under `Runtime invariants that must not regress` and record the exact merge base here.

`/home/ali-awais-safdar/Policy Prism` owns legal prompts, sources, task assembly, admission, sealed evaluations, metrics, and final evidence. Its completed OPD branch is `feat/scope-opd-e2b-12b-environment-v1` at `2be72a1`. Create `feat/policy-prism-capability-distillation-v2` from that commit. Preserve the old scope-OPD environment and all finalized evaluation directories unchanged.

`/home/ali-awais-safdar/Post-Train/trl` is the CarbonTeq TRL fork. The completed constrained IW-OPD release is `3b3e1a6d1fc53f7e52807e676cc0cd9a020250a9`, tagged `trl==1.9.2.post12`. Create `feat/constrained-hybrid-full-distribution-distillation` from that commit. Generic full-distribution loss, expert replay, batching, and gradient logic belong in this fork; Policy-specific task semantics do not.

Commit and push the TRL fork first, update its `CARBONTEQ_FORK.md`, then update PostTrain's consumer page and immutable dependency pin. Commit and push Policy Prism before pinning its environment source in PostTrain project overlays. Never describe an unpushed fork commit as reproducible.

### Terms used in this plan

The **student** is the trainable Gemma 4 E2B model. The **probability teacher** is frozen Gemma 4 12B, which supplies a probability distribution at every completion position. An **expert trajectory** is a complete, validated sequence of stage outputs generated offline and stored as immutable supervised data. It is not the same as a teacher probability distribution.

**IW-OPD** is the completed importance-weighted on-policy method. It learns mainly from the exact tokens sampled by the current student. **Full-distribution OPD** evaluates all grammar-allowed next-token choices at every prefix reached by the current student. **Forward KL** is the divergence from teacher probabilities to student probabilities; it strongly penalizes teacher-supported choices that the student underweights. **Reverse KL** emphasizes choices already favored by the student and can be mode-seeking. **Jensen-Shannon divergence**, abbreviated JSD, compares teacher and student through a mixture distribution; at `beta=0.5` it balances both sides.

**Expert replay** means adding complete supervised examples to every Stage 2 update. It prevents the on-policy component from forgetting complete inventories and gives the student correct prefixes beyond the place where its own rollout would have stopped.

**Scope** is the complete evidence-to-detailed-rules-to-graph legal interpretation workflow. **Recovery** is the evidence-to-compact-rule workflow used to test exhaustive rule inventory recovery. A **sealed** source family or prompt is held out until final qualification. A **non-sealed transfer** cell uses unseen families and unseen prompt wording but may be used for method selection.

**Canonical schema** is the full JSON Schema used to validate the final answer. The **wire schema** is a generation-only copy transformed for XGrammar compatibility. Unsupported wire keywords may be removed without weakening canonical post-generation validation.

### Frozen model identities

The Stage 1 student starts from base Gemma 4 E2B revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. Stage 2 starts from the exact accepted Stage 1 model artifact, not a mutable branch or repository name. The probability teacher is Gemma 4 12B revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.

The ordered token vocabulary and special-token mapping fingerprint is `059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416`. Matching token IDs are necessary but not sufficient. Student and teacher chat-template fingerprints differ, so each model must render the same semantic messages through its own pinned template. Only identical completion token IDs cross the model boundary.

### Evidence corpus

The comparison uses these finalized Policy Prism runs:

| Role | Scope run | Recovery run |
| --- | --- | --- |
| Base E2B non-thinking | `gemma-4-e2b-it-bf16-runpod-a100-sxm-prompt-v2-v11-sealed-scope-20260803` | `gemma-4-e2b-it-bf16-runpod-a100-sxm-prompt-v2-v11-sealed-recovery-20260803` |
| Final E2B IW-OPD | `gemma-4-e2b-policy-prism-scope-iwopd-r16-from-12b-v1-v11-sealed-scope-20260818` | `gemma-4-e2b-policy-prism-scope-iwopd-r16-from-12b-v1-v11-sealed-recovery-20260818` |
| Gemma 4 12B non-thinking | `gemma-4-12b-it-bf16-runpod-a100-sxm-prompt-v2-v11-sealed-scope-20260803` | `gemma-4-12b-it-bf16-runpod-a100-sxm-prompt-v2-v11-sealed-recovery-20260803` |

Each run directory contains `manifest.json`, `traces.jsonl`, `business-kpis.json`, `engineering-metrics.json`, and `semantic-diagnostics.json`. Native Verifiers traces remain the replay authority. Derived KPI files are useful summaries but never replace the underlying traces.

The historical base and teacher runs share compatibility hashes `3fe4471b…` for scope and `72411755…` for recovery. Final IW-OPD uses `3a1c5338…` and `8714ddc1…` because prompt rendering and serving were corrected. Treat business-content comparisons as informative and trace-grounded; do not claim they are a perfectly controlled latency or adapter-only experiment.

## Verified Three-Model Result

### Scope business behavior

| Business KPI | Base E2B | Final IW-OPD E2B | Gemma 4 12B | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Operationally complete cases | 18/18 | 18/18 | 18/18 | Runtime reliability is not the cause. |
| Ambiguity handled | 12/18 | 12/18 | 16/18 | IW-OPD did not transfer teacher ambiguity skill. |
| Contract/source checks | 4/18 | 2/18 | 10/18 | End-to-end legal validity regressed. |
| Expected rules matched | 40/68 | 28/68 | 60/72 | The primary coverage objective regressed. |
| Field evidence grounded | 8.55% | 8.41% | 21.56% | No meaningful E2B improvement. |
| Fully conformant interpretations | 0/18 | 0/18 | 5/18 | Complete interpretation was not learned. |
| Relationships attached | 1/116 | 1/116 | 9/122 | Graph capability did not improve. |
| Returned rules source-supported | 45/49 | 34/35 | 69/72 | Conditional precision increased as coverage fell. |
| Qualifications preserved | 0/44 | 0/44 | 0/53 | This remains unsolved for all three. |
| Required legal text found | 32/33 | 32/33 | 31/33 | Source retrieval was already strong and unchanged. |
| Rule instruction captured | 0/68 | 0/68 | 0/72 | No model solved this field. |
| Rule subject captured | 16/68 | 14/68 | 27/72 | IW-OPD regressed. |
| Selected legal text relevant | 32/32 | 32/32 | 31/31 | Evidence selection remained reliable. |

The teacher denominator is 72 because its output selected different approved Gold alternatives. Base E2B and final IW-OPD retain the same 68-rule denominator and are directly comparable for the central regression.

### Scope engineering behavior

| Engineering metric | Base E2B | Final IW-OPD E2B | Gemma 4 12B | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Hard-gate pass rate | 0.222 | 0.111 | 0.556 | Complete scope quality regressed. |
| Evidence-segment F2 | 0.828 | 0.828 | 0.944 | Evidence extraction was unchanged. |
| Full-rule correspondence F1 | 0.309 | 0.276 | 0.562 | Inventory correspondence declined. |
| Full-rule micro precision | 0.524 | 0.514 | — | Precision was effectively flat. |
| Full-rule micro recall | 0.378 | 0.264 | — | Recall is the principal loss. |
| Node-span F1 | 0.285 | 0.345 | 0.436 | Surviving aligned nodes improved. |
| Legal-detail error rate, lower is better | 0.447 | 0.412 | 0.373 | Surviving rules contain fewer local errors. |
| Edge F1 | 0.0039 | 0.0039 | 0.100 | No graph transfer. |
| Exact scope parse | 0 | 0 | 0.278 | No complete exact E2B case. |
| Definition/recovery at Gold | 0 | 0 | 0.250 | No improvement. |
| Resolution/abstention accuracy | 0.400 | 0.400 | 0.667 | No improvement. |
| Scope pipeline score | 0.0710 | 0.0714 | 0.5608 | End-to-end capability is effectively unchanged. |
| Median latency | 9.75 s | 25.19 s | 29.18 s | Serving/hardware confounded. |
| p95 latency | 39.95 s | 138.02 s | 149.56 s | Serving/output confounded. |

Oracle and dependency-gap metrics are null because no qualifying joint population exists. Null means unavailable, not zero and not success.

Claude completed 49 new packets and reused 2 cached packets for base E2B, completed 47 new packets for IW-OPD, and completed 42 new plus 4 cached packets for 12B. All three runs record zero failed judge packets. The judge evidence therefore supports the source-entailment and semantic conclusions; it does not support blaming missing Claude results for the regression.

### Recovery business behavior

| Business KPI | Base E2B | Final IW-OPD E2B | Gemma 4 12B | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Operationally complete cases | 17/17 | 17/17 | 16/17 | IW-OPD is operationally sound. |
| Complete provision exact | 0/17 | 0/17 | 0/17 | No model fully solved a provision. |
| Context-position stability | 48.38% | 73.44% | 73.42% | Pooled position behavior improved. |
| Contract/source checks | 7/17 | 6/17 | 11/17 | IW-OPD slightly regressed. |
| Expected rules recovered | 223/460 | 337/460 | 339/460 | Large pooled but case-concentrated gain. |
| Returned rules source-supported | 293/303 | 417/428 | 378/420 | Conditional support stayed high. |
| Recovered legal meaning | 87/460 | 178/460 | 294/460 | Useful gain, still far below teacher. |
| Required legal text found | 238/376 | 376/376 | 364/376 | Major retrieval/coverage gain. |
| Correct source assignment | 83/460 | 147/460 | 215/460 | Improved, still below teacher. |
| Nonduplicate unmatched rules | 80 | 80 | 79 | Unsupported inventory did not improve. |
| Selected legal text relevant | 98.76% | 98.95% | 99.73% | Evidence relevance stayed high. |

### Recovery engineering behavior

| Engineering metric | Base E2B | Final IW-OPD E2B | Gemma 4 12B | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Hard-gate pass rate | 0.412 | 0.353 | 0.647 | Overall case quality regressed. |
| Compact-rule F1 | 0.415 | 0.464 | 0.587 | Compact rule matching improved. |
| Gold-rule macro recall | 0.639 | 0.650 | 0.793 | Broad case-level improvement is small. |
| Gold-rule micro recall | 0.513 | 0.728 | — | Pooled count is dominated by a large case. |
| Missing rules per provision, lower is better | 13.18 | 7.35 | 7.71 | Pooled omissions improved. |
| Maximum missing rules, lower is better | 148 | 52 | 78 | The largest failure improved. |
| False positives per provision, lower is better | 3.94 | 5.47 | 5.69 | Added coverage also added unsupported rules. |
| Evidence exact-set recovery | 0.529 | 0.529 | 0.647 | Evidence set selection was unchanged. |
| Evidence-segment F2 | 0.8660 | 0.8661 | 0.8612 | Evidence overlap was unchanged. |
| Per-segment count error, lower is better | 1.171 | 1.064 | 0.680 | Small improvement. |
| Coverage-ledger exact match | 0.0588 | 0.0588 | 0.3529 | No complete-ledger improvement. |
| Position recall, macro | 0.6428 | 0.6432 | 0.7772 | Broad positional recovery was flat. |
| Recovery pipeline score | 0.5369 | 0.5616 | 0.6607 | Small aggregate improvement. |
| Median latency | 11.82 s | 25.60 s | 27.66 s | Not a controlled capability comparison. |
| p95 latency | 174.37 s | 539.71 s | 656.03 s | Output/backend confounded. |

Claude completed 84 new packets and reused 17 cached packets for base E2B, completed 115 new packets for IW-OPD, and completed 112 new plus 19 cached packets for 12B. All three runs record zero failed packets. Excluding the 148-rule Ohio case, Claude-supported meaning changes from `87/223` for base E2B to `84/229` for IW-OPD. The pooled semantic gain therefore does not establish uniform semantic improvement.

### Reconciled scientific interpretation

The model did not simply become worse, nor did it broadly learn the teacher's recovery capability. It learned a narrower policy: return a conservative set of locally credible rules and, for some long provision shapes, decompose far more aggressively. That policy helped one exceptionally large recovery case and improved several conditional metrics. It hurt scope cases that required continuing a detailed inventory and coordinating the result across evidence, detailed rules, and graph stages.

The benchmark is behaving consistently. Micro metrics count every rule and therefore let a 148-rule case dominate. Macro metrics weight cases or families more evenly and show a much smaller gain. Conditional alignment metrics ignore omitted rules and can improve while end-to-end coverage falls. Future selection must inspect all three views together: pooled counts, macro/family results, and complete-pipeline gates.

## Root Causes and Ruled-Out Explanations

### Primary cause: missing outputs had no direct positive target

The final loss was computed on sampled student tokens. It could say that a returned actor, action, quote, or closing token was more or less teacher-like. It could not directly assign likelihood to an entire missing rule that never appeared. This is the strongest explanation because it follows from the exact loss implementation and matches the observed high precision, lower output count, and normal early closure.

### Amplifier: early-token weighting

Position decay further reduced the relative influence of later rules. Research on position bias in on-policy distillation explains why early states often carry cleaner teacher guidance, but Policy Prism's later inventory members are not optional. The next method uses uniform completion-token weight and source-balanced aggregation.

### Amplifier: selected-stage and tangent imbalance

Most optimized targets were standalone rules. A graph target did not optimize the evidence and rules that created its dependencies, and a rules target did not optimize graph consequences. Recovery was not a training tangent. This explains why compact rule behavior moved while scope relationships, qualifications, and full pipeline quality did not.

### Amplifier: output-cardinality imbalance

The curriculum stratified input length but not the number of required output objects. A method can look balanced across source quartiles while seeing too few examples of zero-rule abstention, small inventories, medium inventories, and safe 33-64-rule completion. The next plan treats output cardinality as a first-class hidden stratification variable.

### Amplifier: prompt-profile imbalance

The exact benchmark prompt appeared frequently in training and non-canonical variants were less detailed. This did not leak sealed answers, but it weakens prompt-transfer evidence and can encourage a narrow response policy. The next profiles are independently authored, semantically complete, and training-only.

### Ruled out: final XGrammar probability mismatch

Earlier attempts did mix behavior and scoring probability spaces, but the final run corrected that. Student rollout, teacher scoring, and current-student loss used the same allowed-token set and constrained normalization at every optimized position. The completion validator checked the evidence. The new full-distribution method must retain this invariant.

### Ruled out: teacher-native prompt mismatch in the final run

E2B and 12B have different chat templates. The final implementation rendered the same semantic prompt independently through each pinned template and aligned only the exact completion IDs. The new method must compare completion positions after separate native prefixes; it must never send E2B's integer prefix directly to 12B.

### Ruled out: malformed output or serving truncation

The checkpoint-96 evaluation had a whitespace-serving defect, but the final evaluation did not. Final scope and recovery selected outputs were valid, bounded structured responses with normal stops. The scientifically important omissions are genuine model choices.

### Ruled out: Claude outage or incomplete semantic evidence

Every required final judge packet succeeded. Claude evidence may have normal model-judge uncertainty, but missing judge results do not explain the difference.

### Ruled out: a simple KPI implementation error

Trace-level case inspection reproduces the headline coverage changes. Metric naming can always be improved, but the omitted rules and case concentration exist in the raw model outputs. Existing KPI definitions remain unchanged for continuity.

## Research Synthesis

Generalized Knowledge Distillation established the value of training on student-generated sequences because it exposes the teacher to states the student actually visits. It also permits alternative divergences when a student cannot represent the full teacher behavior. This supports retaining an on-policy phase, not treating it as sufficient for unseen inventory members. See [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649).

Sequence-level knowledge distillation established the complementary value of complete teacher sequences. The student receives the actual continuation and later states rather than only local probabilities on its own path. This supports the supervised cold-start stage and expert replay. See [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139/).

Recent analysis of position bias in on-policy distillation reports that early tokens can dominate useful learning and that later student states are harder to supervise. That finding explains why position decay can help generic reasoning yet conflict with an exhaustive legal inventory whose later elements remain required. See [On the Position Bias of On-Policy Distillation](https://arxiv.org/abs/2606.22600).

Research on teacher-student mismatch shows that student-deficit tokens may rarely be sampled, blocking their transfer, and that a larger teacher is not automatically the best teacher for a smaller student. These findings support explicit complete targets, non-sealed selection, and treating 12B as a ceiling and probability guide rather than assuming its capability will transfer automatically. See [Mismatch Matters: On-Policy Distillation Beyond Token Agreement](https://arxiv.org/abs/2608.09836) and [Towards the Law of Capacity Gap in Distilling Language Models](https://arxiv.org/abs/2311.07052).

Work on KL direction in practical language-model distillation shows that forward and reverse KL can emphasize different parts of the distribution during finite training, even though their asymptotic behavior is more nuanced than the usual “mean-seeking versus mode-seeking” shorthand. This supports the narrow JSD-versus-forward-KL comparison and rules out another unstructured loss sweep. See [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) and [Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2404.02657).

Prompt-paraphrase and contrastive-instruction research supports training with semantically equivalent but genuinely independent instructions rather than superficial suffix changes. Legal information-extraction work further supports evaluating downstream structure, not only local extraction accuracy. See [Paraphrase Types Elicit Prompt Engineering Capabilities](https://aclanthology.org/2024.emnlp-main.617/), [Contrastive Instruction Tuning](https://aclanthology.org/2024.findings-acl.613/), and [Connecting Symbolic Statutory Reasoning with Legal Information Extraction](https://aclanthology.org/2023.nllp-1.12/).

XGrammar research explains the efficiency and strictness of grammar-constrained decoding, but grammar correctness is not semantic completeness. A model can generate a perfectly valid one-rule JSON object where ten rules are required. The training signal and curriculum must teach completeness. See [XGrammar](https://arxiv.org/abs/2411.15100).

These papers inform the hypotheses; they do not prove this experiment will succeed. Only the controlled non-sealed comparison and final sealed qualification can establish the result for Policy Prism.

## Target Experiment

### Stage 1: complete-trajectory sequence SFT

Start from base E2B non-thinking. Train a rank-32 LoRA adapter on complete validated trajectories from 256 non-sealed source units. Use one epoch initially so every target is seen once without repeatedly memorizing the first data rung. Apply ordinary next-token cross-entropy to every assistant token in every required stage. Prompts and tool/environment text remain outside the loss.

Use 128 scope sources and 128 recovery sources. Each scope source contains complete evidence, detailed-rules, and graph outputs, creating 384 supervised stage outputs. Each recovery source contains evidence and compact-rules outputs, creating 256 supervised stage outputs. The first rung therefore contains 640 stage outputs while keeping the two capabilities balanced by source.

Aggregate Stage 1 loss in two levels. First average tokens uniformly within a stage output, with no early-position decay. Then give each logical source equal total weight, averaging its required stages internally. This prevents long inventories and three-stage scope trajectories from silently dominating compact recovery sources.

Stage 1 is successful only if non-sealed development and held-out prompt/family cells show better complete coverage than base E2B without a material support-precision or structural-reliability loss. Training loss alone is not an acceptance criterion.

### Stage 2: expert-retaining full-distribution OPD

Start both Stage 2 branches from the exact same accepted Stage 1 adapter artifact. For each update, obtain fresh student trajectories through the bound Policy Prism environment and separately sample immutable expert trajectories from the exact Stage 1 dataset selection. Do not regenerate expert text during training.

At each student completion position, independently render the shared semantic messages through the E2B and 12B native templates. Feed the same completion prefix to both models. Reconstruct the XGrammar allowed-token set from the canonical generation schema and completion prefix. Mask and normalize both models inside that identical allowed set before computing divergence.

The primary branch uses generalized JSD with `beta=0.5`. The controlled alternative uses forward KL with `beta=0.0`, where the teacher distribution is the target and the student distribution is penalized for missing teacher-supported alternatives. Do not run a new IW-OPD or reverse-KL branch; the completed run already supplies that evidence.

Each Stage 2 update computes two independently normalized terms:

    expert_loss = mean source-balanced cross-entropy on complete expert trajectories
    opd_loss    = mean source-balanced constrained full-distribution divergence on fresh student trajectories
    total_loss  = (2 / 3) * expert_loss + (1 / 3) * opd_loss

Log all three values, their token and source counts, and their gradient contributions. A change to the weights is a new experiment revision, not an undocumented tuning action.

Full-distribution OPD can teach “continue with another rule” rather than “close the object” at prefixes the student reaches. Expert replay covers later prefixes and correct stage dependencies that the on-policy trajectory may never reach. The two signals are complementary and must remain present throughout Stage 2.

Use exactly 24 optimizer updates at logical batch 12, for 288 logical source occurrences per branch. Schedule every one of the 256 sources once, then schedule 32 predeclared second occurrences balanced as 16 scope and 16 recovery sources across prompt profile, family, source-length, and output-cardinality strata. Pair each on-policy occurrence with the complete expert trajectory for the same logical source and profile. This keeps the qualified `12/1/12` geometry without dropping four sources or silently using a partial optimizer batch. Freeze the 288-occurrence schedule before either divergence branch and reuse it byte-for-byte.

### Provisional common configuration

These settings are hypotheses to be frozen before the one live canary. They are not claimed to be universally optimal.

| Dimension | Value | Reason |
| --- | --- | --- |
| Student | Gemma 4 E2B, thinking off | Direct comparison with the established baseline. |
| Probability teacher | Gemma 4 12B, frozen | Strong local teacher with compatible ordered tokens. |
| LoRA | rank 32, alpha 64, dropout 0 | More adapter capacity than the narrow rank-16 result; identical across all branches. |
| Logical / physical / accumulation | 12 / 1 / 12 | Fastest validated long-context geometry. |
| Maximum per-call sequence | 40,960 | Qualified Policy envelope; requests remain stage-clipped. |
| Stage 1 learning rate | `5e-5` | Conservative supervised LoRA starting point; one epoch. |
| Stage 2 learning rate | `1e-6` | Small continuation update to preserve Stage 1 coverage. |
| Warmup | 0 | Short first-rung experiment; identical branches. |
| Scheduler | linear | Existing qualified scheduler; identical branches. |
| Rollout sampling | temperature 1, top-p 1, top-k 0, one generation | Required for an unbiased broad on-policy distribution; grammar still constrains validity. |
| Student prefix cache | off | Weight synchronization invalidation remains an unnecessary correctness risk. |
| Teacher prefix cache | on | Teacher weights are frozen. |
| MTP | off | It was not required for correctness and is not part of the method question. |
| Stage 2 updates | 24 over 288 occurrences | Covers every source once plus 32 balanced repeats at logical batch 12. |
| Checkpoints | updates 6, 12, 18, 24 | Recovery evidence; selection uses non-sealed evaluation, never loss alone. |

If the full-distribution operator exceeds memory, reduce its position and vocabulary chunk sizes while proving numerical equivalence. Do not first reduce sequence coverage, batch cardinality, model-native alignment, constrained normalization, or expert replay.

### Scaling after the method decision

The 256-source experiment answers which learning method works; it is not assumed to be the final data scale. If one branch passes all non-sealed and sealed gates, freeze that method, loss weights, prompt contract, LoRA geometry, and evaluation protocol. Build nested 512-, 1,000-, and 2,000-source datasets by retaining every previously accepted source and adding new isolated families and strata. Revalidate every new expert trajectory and output envelope.

Use non-sealed development and transfer cells at each scale. Do not repeatedly inspect or tune against sealed scope/recovery. Run sealed qualification again only for a predeclared release candidate after the larger-scale method and configuration are frozen. After proving the two Policy Prism capabilities, add a separate model-family transfer experiment and an optional legacy planner-worker transfer check; neither is part of the primary 256-source acceptance gate.

## Policy Prism Curriculum and Prompt Contract

### Repository boundary

Create new modules and resources rather than modifying the completed scope-OPD environment or sealed evaluation prompts. The expected package layout is:

    packages/normative-verifiers/src/policy_prism_normative_verifiers/
        capability_distill_data.py
        capability_distill_prompts.py
        capability_distill_tasks.py
        capability_distill_admission.py
        capability_distill_teacher_data.py
        capability_distill_selection.py
        policy_prism_capability_distill_v1/__init__.py

Add explicit package resources for scope and recovery prompt profiles, immutable family splits, source plans, teacher-candidate receipts, expert-data manifests, and non-sealed evaluation selections. Update `constants.py`, `factories.py`, `harness.py`, `program.py`, `cli.py`, package exports, and `pyproject.toml` only through additive capability-distillation paths.

### Family and evidence isolation

Split complete regulatory-instrument families before prompt expansion. Every version of an instrument stays in one split. Exclude every family, source version, text hash, and semantic equivalent listed by the existing sealed-training exclusion manifest.

Create four disjoint cells:

1. `train`: source families used to construct the 256-source expert and on-policy curriculum.
2. `nonsealed_dev`: unseen families used for checkpoint and method selection.
3. `nonsealed_transfer`: unseen families plus a held-out prompt profile used to test transfer.
4. `sealed`: existing benchmark families, exact prompts, Gold, and cases; never resolved by a training or selection job.

Write a deterministic split receipt containing every family ID, source version, text hash, and split. A test must fail on any cross-cell family or text-hash overlap.

### Prompt profiles

Create four independently authored, semantically complete training profiles for each tangent and one held-out transfer profile. A profile is a complete set of stage prompts, not an emphasis paragraph appended to a common benchmark prompt.

Every scope profile must preserve evidence inventory, inherited parent meaning, actor and agency distinctions, legal effects, conditions, exceptions, quantities, deadlines, ambiguity, abstention, exact contiguous quotes, zero/one/many rule decomposition, unique IDs, relationships, and one closed non-repeating JSON object. Every recovery profile must preserve exhaustive compact rule decomposition, source identity, legal meaning, assignment, evidence quotes, duplicate prevention, and safe abstention.

Keep exact benchmark v9/v10/v6 prompt resources evaluation-only. Preserve prompt IDs, profile IDs, and version metadata in configuration and traces, but send only semantic fields to the model: role, objective, instructions, output rules, skeleton, and examples. Model-visible context IDs must be opaque and must not contain decision classes or split labels.

Snapshot-test the exact semantic messages for every tangent, profile, stage, and task shape. Tests must prove identical canonical schemas across semantically equivalent profiles and the absence of Gold, expected rule counts, hidden decision labels, benchmark IDs, and sealed text.

### Source and output distribution

Balance both inputs and expected outputs. Each tangent must cover source domain, regulatory actor, regulated actor, obligation, prohibition, permission, right, power, definition, constitutive effect, parent-child inheritance, positive and empty qualifications, conditions, exceptions, quantities, deadlines, ambiguity, abstention, source length, evidence position, and hard negatives.

Use hidden offline output-cardinality bins `0`, `1`, `2-4`, `5-10`, `11-32`, and `33-64`. These labels and counts never enter prompts. Each train and non-sealed cell must contain enough sources in every applicable bin to prevent one very large case from determining a method decision. No candidate whose conservative complete output envelope exceeds its effective sequence budget may be packaged.

For scope, every logical source must include one complete evidence-to-rules-to-graph trajectory. For recovery, every source must include one complete evidence-to-compact-rules trajectory. Do not use selected-stage-only loss in this curriculum. A structurally invalid dependency cannot feed a later stage, but a valid complete expert trajectory contains every required dependency by construction.

### Expert target generation and validation

Generate at least four candidate complete trajectories per source with the strongest available broad answer teacher selected from finalized evidence. The current evidence nominates GPT-5.6 Luna Pro for candidate generation; re-resolve its immutable provider identity before execution. This provider is separate from the Gemma 4 12B probability teacher.

Admission is stricter than the old source-only OPD admission because these outputs become positive supervised targets. Require:

- normal completion with exactly one canonical JSON object per stage;
- canonical schema, IDs, references, uniqueness, and dependency validity;
- exact source and quote resolution;
- no unsupported source IDs, invented text, repeated objects, or unresolved graph references;
- complete inventory review using available non-sealed annotations and cross-candidate comparison;
- an independent semantic review for actor, effect, qualification, ambiguity, source assignment, and omissions;
- human review for high-cardinality, candidate-disagreement, ambiguous, and incomplete-context cases.

Use deterministic checks first, a second model or Claude judge as supporting evidence second, and human adjudication at the risky boundary. A judge score never converts raw teacher output into Gold by itself. Store every candidate, rejection reason, selected target, reviewer decision, provider identity, prompt/schema hash, source hash, and content digest.

Package only accepted complete trajectories into a versioned supervised dataset. Do not allow provider calls during Stage 1 or Stage 2 data loading.

## Framework and TRL Design

### Required canonical amendment

Before implementation, amend the frozen baseline narrowly:

- `docs/post-training/README.md`: add hybrid distillation to the capability map without redefining existing distillation.
- `docs/post-training/01-workflow.md`: place `train.hybrid-distill` in the train phase and keep later qualification separate.
- `docs/post-training/02-primitives.md`: define its exact seats, expert-data selection, fresh-trajectory semantics, settings, and outputs.
- `docs/post-training/03-work-and-evidence.md`: define the two-source training evidence and lineage edges.
- `docs/post-training/04-framework.md`: assign public contracts to `posttrain.train`, generic trainer behavior to the TRL adapter/fork, and legal semantics to Policy Prism.
- `docs/post-training/05-apis.md`: add request/result types and public operation names.
- `docs/post-training/06-observation-and-lineage.md`: define expert and on-policy metric namespaces, native traces, checkpoints, summary, and model-artifact lineage.

The amendment must state that `train.distill` remains fully on-policy and accepts no stored expert completions. `train.sft` remains ordinary supervised training. `train.hybrid-distill` is the only operation that combines an immutable expert dataset with fresh current-student trajectories in one optimizer update.

### Public API

Add framework-neutral types in `packages/train/src/posttrain/train/api.py` or the module that owns the current request values:

    class HybridDistillationSettings:
        divergence: Literal["jsd", "forward_kl"]
        jsd_beta: float
        temperature: float
        expert_loss_weight: float
        on_policy_loss_weight: float
        teacher_prompt_alignment: Literal["model_native_prefix_exact_completion"]
        probability_space: Literal["generation_constrained"]
        token_weighting: Literal["uniform"]
        source_normalization: Literal["equal_logical_source"]

    class HybridDistillationRequest:
        student: ModelVariant
        teacher: ModelVariant
        expert_data: DatasetSelection
        environment: EnvironmentBinding
        settings: HybridDistillationSettings
        training: TrainingBinding
        rollout_inference: InferenceBinding
        teacher_inference: InferenceBinding
        quantization: QuantizationPlan | None

    def hybrid_distill(ctx: RunContext, request: HybridDistillationRequest) -> TrainingResult: ...

Validation must require positive loss weights summing to one, `jsd_beta=0.5` for JSD and `0.0` for forward KL, uniform token weighting, exact model/template/tokenizer identities, a supervised dataset with complete-trajectory metadata, a fresh Verifiers environment, one rollout per source occurrence, and aligned constrained schemas. Reject a backend that cannot prove these semantics.

Add `train.hybrid-distill` to work-package contracts, job definitions, execution-pack routing, catalog schemas, CLI planning, Lab composition, deployment qualification, telemetry, Observatory query/presentation, and tests. Do not make train import eval, serve, Trackio, Verifiers, or Lab directly.

### Generic TRL implementation

Implement generic behavior in the CarbonTeq TRL fork. Reuse the stable `DistillationTrainer` full-distribution semantics where possible, but do not use its current dense sequence-by-vocabulary path unchanged.

At each selected on-policy completion position:

1. Run the student on its native prompt plus completion prefix and obtain the hidden state required for the next-token distribution.
2. Run the teacher on its native prompt plus the same completion prefix and obtain the aligned hidden state.
3. Reconstruct the exact grammar-allowed token IDs for that completion prefix.
4. Project student and teacher hidden states through their language-model heads in vocabulary chunks.
5. Compute masked log-sum-exp normalization over the full allowed set for each model.
6. Accumulate either generalized JSD at `beta=0.5` or teacher-to-student forward KL without materializing a full sequence-by-vocabulary tensor.
7. Average uniformly across valid completion positions and then according to the source-level normalizer.

The implementation must support different native prompt lengths. Alignment is by semantic request identity, completion-token digest, and completion position—not absolute sequence index. It must fail closed if template fingerprints, completion IDs, canonical schema, wire schema, grammar state, or allowed-token masks differ.

For expert replay, collate complete supervised trajectories independently of on-policy rows. Compute ordinary causal cross-entropy on assistant completion tokens only, normalized by logical source. Combine independently reduced loss components only after global gradient-accumulation item counts are known. The final gradient must be mathematically identical whether a logical batch is processed physically as `1x12`, `2x6`, or `3x4` within numerical tolerance.

Expose explicit per-update diagnostics:

    train/hybrid/expert_loss
    train/hybrid/opd_loss
    train/hybrid/total_loss
    train/hybrid/expert_tokens
    train/hybrid/on_policy_tokens
    train/hybrid/allowed_vocab_mean
    train/hybrid/teacher_entropy
    train/hybrid/student_entropy
    train/hybrid/continue_vs_close_teacher_mass
    train/hybrid/continue_vs_close_student_mass
    train/hybrid/teacher_failures

Names may be refined during the canonical amendment, but their meanings must be fixed before a run. `continue_vs_close` is diagnostic only and must be computed from grammar-valid structural alternatives without Gold rule counts.

Checkpoint recovery must include model, optimizer, scheduler, RNG, sampler position for both expert and on-policy streams, Verifiers runtime state, and all content digests. Publishing the same logical checkpoint name with the same digest is idempotent; the same name with different bytes fails closed.

### Runtime invariants that must not regress

Retain all of the following from the completed experiment:

- immutable ordered-token and special-token fingerprints;
- immutable student and teacher chat-template fingerprints;
- separate native prompt rendering from the same semantic messages;
- exact completion-token alignment;
- canonical validation schema plus generation-only XGrammar wire transformation;
- one identical allowed-token set for rollout, teacher, and differentiable student distributions;
- bounded Gemma whitespace and grammar-owned stopping;
- output-feasible source plans and dynamic per-call limits;
- deterministic retry, safe fallback, and no invalid dependency propagation;
- concurrency-safe, checkpointed Policy runtime state;
- bounded rollout waves at logical 12 and physical training batch one;
- student prefix cache off, frozen-teacher prefix cache on;
- idempotent paired checkpoint publication and terminal tracking recovery;
- read-only artifact materialization with provider and content digest receipts;
- exact model-adapter selection in evaluation, never the base model by accident;
- sequential scope then recovery submission on the single dstack GPU;
- native Verifiers traces as replay authority and five-file Policy finalization as derived evidence.

## Milestones

### Milestone 1: freeze evidence and amend product semantics

Create the three feature branches at the recorded commits. Copy no source tree into another repository. Update the canonical PostTrain documents first and record the new operation's seats, exact semantics, output roles, and telemetry. Add failing API/contract tests that demonstrate why expert data cannot be passed to `train.distill` and why `train.hybrid-distill` requires it.

Acceptance is a reviewable baseline amendment and tests that distinguish `train.sft`, `train.distill`, and `train.hybrid-distill` without backend-specific types. No trainer code is changed before this contract exists.

### Milestone 2: implement memory-safe constrained hybrid loss in TRL

Add the generic chunked allowed-set full-distribution loss and expert replay to the TRL fork. Build tiny dense reference tests for JSD and forward KL. Compare loss and every trainable gradient against the chunked operator under different prompt lengths, grammar masks, sequence lengths, and gradient-accumulation partitions.

Add negative tests for mismatched completion IDs, masks, schemas, templates, missing expert rows, zero valid tokens, non-finite teacher logits, and checkpoint stream mismatch. Retain existing IW-OPD tests unchanged. Update `CARBONTEQ_FORK.md`, commit, push, tag an unambiguous post-release version, and record its wheel and source hashes.

Acceptance is numerical loss and gradient agreement with the dense reference, no dense sequence-by-vocabulary allocation in the production operator, and byte-consistent checkpoint/recovery of both data streams.

### Milestone 3: add PostTrain's public operation and evidence surfaces

Pin the released TRL fork. Add the public settings/request, adapter, catalog definition, work-package seats, execution-pack route, static validation, telemetry, Observatory view, checkpoint roles, and job tests. Add a generic fake backend to prove contract behavior and a real TRL adapter test to prove the selected backend semantics.

Update `docs/tooling/trl/README.md` with the exact fork revision, configuration, qualification evidence, and remaining gates. Rebuild the dedicated online-RL dependency closure without changing unrelated SFT or evaluation closures.

Acceptance is catalog and work-package planning for a minimal hybrid job, exact immutable dependency resolution, equivalent logical results across physical accumulation partitions, and a packed image that reports the intended PostTrain, TRL, torch, vLLM, XGrammar, and Policy source identities.

### Milestone 4: build Policy Prism's capability curriculum

Implement the new scope and recovery tasksets, independent prompt profiles, family splits, source/output stratification, teacher-candidate generation contract, expert-data validator, and non-sealed dev/transfer environments. Do not alter sealed prompt files or the completed scope-OPD taskset.

Generate plan receipts deterministically. Run exhaustive tokenizer rendering, output-envelope, schema compilation, prompt-semantic, split-isolation, task-distribution, and label-leakage checks over every candidate/profile/stage materialization.

Acceptance is a reproducible 256-source plan with exactly 128 complete scope and 128 complete recovery trajectories, all required strata represented, no sealed overlap, no model-visible hidden labels or counts, and no output-infeasible candidate.

### Milestone 5: prepare and validate expert data

Run a separate tracked `data.prepare` workflow to generate candidate trajectories. Retain raw candidates and validation decisions. Produce one immutable supervised dataset artifact containing only selected complete trajectories plus lineage metadata.

Acceptance is 256/256 sources with complete admitted trajectories, 640 stage outputs, zero unresolved structural/source/reference failures, complete candidate and reviewer receipts, and a fresh materialization whose content digest matches the catalog selection. If a source cannot produce a trustworthy complete target, replace it within the same predeclared stratum and update the selection receipt before training.

### Milestone 6: pass offline gates and one live canary

Run focused Policy, TRL, and PostTrain tests, then the normal repository validation ladders. Pack the exact Stage 1 and Stage 2 images and inspect them in isolation. Run one fixed production-shaped Stage 2 canary from a small deterministic Stage 1 fixture or checkpoint. It must include the longest feasible scope and recovery inventories, different native template lengths, all relevant stage schemas, and exactly one `12/1/12` finite optimizer update.

The canary must prove nonzero expert and on-policy loss, finite gradients, identical allowed sets, completion alignment, no teacher failure, no truncation, correct continue-versus-close diagnostics, memory headroom, one model/recovery checkpoint pair, and successful reconciliation. Cancel rather than broaden the smoke matrix if this one boundary test fails; diagnose the exact failed contract.

### Milestone 7: run common Stage 1 sequence SFT

Submit one complete SFT job from base E2B using the immutable expert dataset. Stay attached until the first finite update and first checkpoint. Monitor structural completion, loss, gradient, data cursor, memory, and artifact publication. Reconcile and validate every source/stage exactly once.

Run non-sealed dev and transfer evaluations on base E2B and the Stage 1 checkpoint. Stage 1 passes only if complete coverage improves across both tangents and both cells without structural or support-precision regression. If it fails, do not spend GPU budget on Stage 2; revise the expert curriculum.

### Milestone 8: run the two Stage 2 method branches

Launch JSD and forward-KL jobs sequentially from the exact same Stage 1 artifact, source plans, seeds, images, and batch geometry. Only the divergence selection differs. Monitor expert loss, on-policy loss, total loss, gradient norms, teacher failures, output cardinality by bin, complete trajectory admission, and continue-versus-close mass.

Reconcile each run and verify every checkpoint deeply. Evaluate the predeclared checkpoints on non-sealed dev and transfer cells. Select one candidate using the rule below; never use sealed data or training loss alone.

### Milestone 9: select, qualify, publish, and finalize

Rank candidates by a predeclared Pareto rule:

1. Both scope and recovery must exceed Stage 1 and base E2B on macro/family coverage in both non-sealed cells.
2. Operational reliability, source support, structural validity, and hard-gate pass rate may not regress beyond the predeclared tolerance.
3. Improvements must appear across output-cardinality bins and may not be driven by one source family or one case.
4. Complete scope pipeline and recovery pipeline measures take priority over conditional precision when they conflict.
5. If candidates are otherwise tied, choose the earlier checkpoint, then balanced JSD over forward KL because it is the more conservative update.

Run sealed scope once and then sealed recovery once for the selected candidate. Apply exact case-count, error, truncation, Claude, artifact, and reconciliation gates. Publish only an accepted or explicitly retained research adapter to the private CarbonTeq Hugging Face repository, resolve its immutable revision, download it freshly, and compare a complete SHA-256 file manifest.

Materialize each exact native evaluation artifact and finalize directly under Policy Prism's `evaluation-runs`. Validate the standard five files and catalog, update the all-model comparison report, commit and push the final Policy evidence, and record the explicit scientific decision in this plan.

## Concrete Steps

All commands below are templates for the future execution. Replace only identifiers explicitly marked with angle brackets, record the resolved values in this plan, and never paste secrets into a command transcript.

Start by verifying immutable repository state:

    cd /home/ali-awais-safdar/Post-Train/posttrain
    git status --short --branch
    git rev-parse HEAD

    cd "/home/ali-awais-safdar/Policy Prism"
    git status --short --branch
    git rev-parse HEAD

    cd /home/ali-awais-safdar/Post-Train/trl
    git status --short --branch
    git rev-parse HEAD

Create only the feature branches named in this plan after confirming each source worktree is clean. Push each branch before unattended work begins.

After implementation, run the normal PostTrain validation ladder from `/home/ali-awais-safdar/Post-Train/posttrain`:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Run the TRL fork's focused hybrid-distillation tests first, then its maintained quality suite according to `CARBONTEQ_FORK.md`. The focused command must include dense-versus-chunked loss and gradient tests for JSD/FKL, native-prefix alignment, constrained masks, expert replay, accumulation `1/12`, `2/6`, and `3/4`, and checkpoint resume.

Run Policy Prism's focused capability curriculum and expert-data tests, then Ruff, strict type checking, deterministic plan regeneration with `--check`, and packaged plugin discovery. Record exact commands and pass counts here once the CLI names exist.

Before any GPU job, configure the PostTrain CLI without printing secrets:

    export POSTTRAIN_ROOT=/home/ali-awais-safdar/Post-Train/posttrain
    export POLICY_ROOT="/home/ali-awais-safdar/Policy Prism"
    export POSTTRAIN_ENV_FILE="$POLICY_ROOT/.env.posttrain"

    cd "$POSTTRAIN_ROOT"

    pt() {
      UV_CACHE_DIR=/tmp/posttrain-uv-cache \
      uv run --no-sync --package posttrain posttrain \
        --project-root "$POLICY_ROOT" \
        --env-file "$POSTTRAIN_ENV_FILE" \
        "$@"
    }

Use the exact future work-package names recorded by Milestones 5-9. For every package, run:

    pt catalog validate
    pt work-package validate <work-package.yaml>
    pt job plan <work-package.yaml> --job <job-id>
    pt job pack <work-package.yaml> --job <job-id> --build-missing

Inspect the resolved JSON before submission. It must contain the expected model revisions, dataset content digest, environment source commit, settings revision, TRL version, actual-job image digest, and job kind. No package may resolve the old sampled-token IW-OPD settings.

Submit each training job with a unique descriptive run ID. Run Stage 1 once, then Stage 2 JSD and forward KL sequentially. Use a 24-hour provider timeout as a safety ceiling, not a duration prediction. Keep the controller or explicit reconciliation workflow consistent with the final package design.

For each run:

    pt run status "$RUN_ID"
    pt run logs "$RUN_ID" --follow
    pt run wait "$RUN_ID" --timeout-seconds 86400
    pt run reconcile "$RUN_ID"
    pt --json run show "$RUN_ID" > "$POLICY_ROOT/.posttrain/state/<experiment>/$RUN_ID/run-view.json"

Do not clean a run until its completion receipt, model/checkpoint pairs, native traces, summary, Trackio links, provider/content digests, and non-sealed results are independently verified.

Exact qualification, Hugging Face publication, and Policy finalization commands must be added here after the additive catalog and CLI surfaces exist. Do not leave placeholders at launch time.

## Validation and Acceptance

### Offline correctness gates

All gates below must pass before the one GPU canary:

- dense and chunked constrained JSD/FKL losses agree within declared floating-point tolerance;
- every trainable gradient agrees across dense/chunked and `1/12`, `2/6`, `3/4` accumulation partitions;
- student and teacher native prompt lengths may differ while completion positions and IDs remain exact;
- canonical and wire schemas are separated and every real static/dynamic schema compiles under the exact XGrammar version;
- allowed-token masks are byte-identical for student and teacher at every tested position;
- no dense sequence-by-vocabulary tensor is allocated in the production path;
- expert and on-policy stream cursors resume exactly from every checkpoint;
- one loss stream cannot be missing, empty, non-finite, or silently rescaled;
- every prompt profile is semantically complete and contains no sealed text, Gold, expected counts, split labels, or decision labels;
- every training source is family-isolated, source-hash isolated, output-feasible, and assigned to an output-cardinality bin;
- every expert target passes structural, source, semantic, and reviewer gates;
- every current runtime invariant listed above retains regression coverage;
- catalog, plan, package, image, model, data, environment, and dependency identities are immutable and mutually consistent.

### Live canary gates

The one production-shaped canary passes only when it completes one logical-12 update with:

- twelve unique on-policy logical sources and matching expert rows;
- complete accepted scope and recovery trajectories for the fixed boundary cohort;
- finite expert, divergence, total loss, and gradient norm;
- zero teacher failures and zero selected truncations, malformed objects, mask mismatches, or template mismatches;
- nonzero allowed-vocabulary and teacher/student entropy diagnostics;
- a measurable continue-versus-close comparison on long rule inventories;
- no CUDA OOM, KV-cache preemption, or unexplained sustained memory growth;
- one idempotent model/recovery checkpoint pair and successful reconciliation.

Passing the canary establishes runtime compatibility, not model improvement.

### Stage 1 gates

Stage 1 must consume each selected expert trajectory exactly once, produce finite loss and gradients, and publish complete checkpoints and a loadable adapter. Its non-sealed results must improve both scope and recovery complete-coverage measures over base E2B in development and transfer cells. A gain limited to pooled micro counts, one source family, or conditional precision is insufficient.

### Stage 2 completion gates

Each branch must complete its exact update plan with no teacher failures, non-finite values, data-stream drift, duplicate/missing logical sources, or artifact mismatch. Every update must contain both loss components. Completion evidence must prove exact expert and on-policy counts, output-cardinality distribution, tangent/stage distribution, checkpoints, source commits, image digest, and model identities.

### Non-sealed selection gates

Selection uses both macro and micro coverage, family-level results, complete-pipeline gates, conditional source support, structural reliability, and output-cardinality strata. Report confidence intervals or paired case differences where supported. A method cannot win solely through one large inventory.

Do not select a candidate if it improves recovery but regresses complete scope, improves scope but loses exhaustive recovery, or raises coverage through unsupported additions. Record the full Pareto comparison even when no candidate passes.

### Final sealed acceptance

The selected model must complete 18/18 scope and 17/17 recovery cases with zero errors/truncations, complete Claude evidence, exactly one native evaluation artifact per run, and consistent reconciliation. At minimum, acceptance over base E2B non-thinking requires:

- scope expected-rule matches above `40/68`, full-rule F1 above `0.309`, and no regression below base hard-gate, operational, or source-support levels;
- recovery macro recall above `0.639`, expected-rule matches above `223/460`, and no regression below base operational, contract/source, hard-gate, or source-support levels;
- improvements distributed across cases/families and output-cardinality bins rather than dominated by one case;
- no loss of required-text relevance, ambiguity behavior, source assignment, or structural validity that outweighs coverage gains;
- a recorded comparison with the 12B teacher ceiling without requiring the E2B student to equal it.

If the model fails any primary tangent gate, the scientific decision is `revise` or `reject`, even if training and publication are operationally successful.

## Idempotence and Recovery

Every generated plan, split, prompt bundle, dataset, model, and evaluation artifact must have a content digest and immutable provider identity. Re-running a builder with the same inputs must produce byte-identical receipts. The same logical artifact name with identical bytes is reused; the same name with different bytes fails closed.

Infrastructure interruption may resume only from the latest complete paired checkpoint. Restore model, optimizer, scheduler, RNG, expert sampler, on-policy sampler, Policy runtime state, and all digests together under a new run ID. Verify the resumed first batch identity before allowing a new optimizer step.

Do not resume after a deterministic schema, mask, prompt-alignment, data-lineage, non-finite-loss, systematic teacher, or objective error. Fix the code/configuration, version every affected selection, and restart that scientific branch from its declared starting model. Preserve failed evidence.

If Stage 1 fails scientific non-sealed gates, revise expert data or curriculum rather than proceeding. If one Stage 2 branch fails deterministically, do not mutate it in place; version the correction and rerun both method arms only if the change affects comparability. If a provider fails, resume the affected branch without rerunning a completed comparable branch.

Submit scope and recovery sequentially on the single dstack GPU. A failed scope gate blocks recovery for that candidate. Finalize directly into the permanent `evaluation-runs` root; do not manually move directories. Clean provider workspaces only after Hugging Face fresh verification and Policy evidence validation. Never delete Trackio, failed-run, checkpoint, or finalized evaluation evidence as part of routine cleanup.

## Artifacts and Notes

The completed IW-OPD model is retained as a valid research artifact, not an accepted scope improvement. Its private Hugging Face repository is `carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-lora-v1`; the completed historical plan records weight and card revisions and exact artifact digests.

The core observed behavior can be summarized without ambiguity:

    Scope matched rules:       base 40/68  -> IW-OPD 28/68
    Scope returned rules:      base 49     -> IW-OPD 35
    Scope source support:      base 45/49  -> IW-OPD 34/35
    Recovery matched rules:    base 223/460 -> IW-OPD 337/460
    Recovery excluding Ohio:   base 223/312 -> IW-OPD 229/312
    Scope relationships:       base 1/116  -> IW-OPD 1/116

This is the pattern the next experiment must change: retain high source support, increase complete inventory coverage across families, and improve the whole legal workflow rather than one recovery shape.

## Interfaces and Dependencies

The new PostTrain operation is `train.hybrid-distill`. Its reusable API belongs to `posttrain.train`; project-specific prompts and task selection remain in Policy Prism; generic loss and trainer mechanics belong in the CarbonTeq TRL fork. `posttrain.common` must not import TRL, Verifiers, vLLM, XGrammar, or Trackio. Train, eval, and serve remain independent packages, and no reusable package imports `apps/lab`.

The expert dataset is an immutable `DatasetSelection`. The on-policy curriculum is an immutable `EnvironmentBinding`. The two are separate seats because they have different generation, freshness, and lineage semantics. The Stage 1 model and each Stage 2 model are `model-adapter` artifacts. Recovery checkpoints are trainer state and are never published as Hugging Face models.

The exact required external services are the internal OCI registry, Trackio, dstack, the GPU workstation, gated Hugging Face model access, private CarbonTeq Hugging Face write access, and OpenRouter/Claude only for authorized teacher-data review and final semantic evaluation. Preflight credentials without printing secrets and record service reachability separately from model correctness.

Primary research references used to design this plan are:

- Agarwal et al., [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649).
- Kim and Rush, [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139/).
- [On the Position Bias of On-Policy Distillation](https://arxiv.org/abs/2606.22600).
- [Mismatch Matters: On-Policy Distillation Beyond Token Agreement](https://arxiv.org/abs/2608.09836).
- [Towards the Law of Capacity Gap in Distilling Language Models](https://arxiv.org/abs/2311.07052).
- Gu et al., [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543).
- [Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2404.02657).
- [XGrammar](https://arxiv.org/abs/2411.15100).
- [Paraphrase Types Elicit Prompt Engineering Capabilities](https://aclanthology.org/2024.emnlp-main.617/).
- [Contrastive Instruction Tuning](https://aclanthology.org/2024.findings-acl.613/).
- [Connecting Symbolic Statutory Reasoning with Legal Information Extraction](https://aclanthology.org/2023.nllp-1.12/).

Revision note (2026-08-19): created this plan after the completed final IW-OPD qualification. It reconciles all retained metrics, Claude diagnostics, case traces, objective behavior, curriculum structure, serving corrections, and primary research. It preserves the completed IW-OPD plan as historical evidence and defines a new coverage-first hybrid experiment rather than extending the same sampled-token objective.
