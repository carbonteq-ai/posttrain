# Run Policy Prism Gemma 4 E2B from 12B on-policy distillation end to end

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated as work proceeds. It follows `docs/templates/PLAN.md` and the frozen product baseline under `docs/post-training/`.

## Purpose / Big Picture

This work makes one reproducible, full-length Policy Prism on-policy distillation (OPD) experiment possible on the in-house RTX PRO 6000. Gemma 4 E2B is the trainable student and Gemma 4 12B is the frozen teacher. The student generates exact source-only legal-interpretation trajectories; the teacher scores the selected target-stage tokens; a sparse completion-only reverse-KL loss updates a rank-16 LoRA adapter. When complete, the adapter, checkpoints, native traces, training metrics, sealed scope and recovery evaluations, and immutable Hugging Face publication must all be recoverable from recorded lineage.

The user-visible proof is a reconciled PostTrain training run with 384 admitted updates, a private verified Hugging Face adapter, two reconciled domain-evaluation runs with 18/18 and 17/17 usable traces, and Policy Prism five-file evaluation directories that pass its normal validator.

## Progress

- [x] (2026-08-06) Read the canonical PostTrain workflow, primitives, work/evidence, framework, API, observation/lineage documents and the plan template.
- [x] (2026-08-06) Verify Policy Prism branch `feat/scope-opd-e2b-12b-environment-v1` is clean at `de5ca4ccd8de1cbdf1b5066ad034c815426c5b00`.
- [x] (2026-08-06) Preserve the obsolete uncommitted PostTrain prototype as backup ref `refs/backup/feat-gemma-policy-prism-opd-e2b-12b-pre-realign` and `/tmp/posttrain-opd-prototype-a78df9e.patch`, then realign only the authorized feature branch to current `origin/main` at `78d329fae89a3448cbe4f89b1744ae684e8e6358`.
- [x] (2026-08-06) Verify credentials exist without displaying secrets and verify approximately 136 GiB local disk is free.
- [x] (2026-08-07) Audit Policy Prism source discovery and produce an auditable independently reviewed ambiguity/incomplete registry without keyword inference or fabricated human-review claims.
- [x] (2026-08-07) Build the deterministic 384-primary/96-reserve production plan with the pinned E2B tokenizer, package it in Policy Prism, pass the deterministic check, 17 focused tests, Ruff, strict mypy, diff check, and wheel resource audit, then commit and push Policy Prism commit `874e90205ed407126a7777221cb87dd3b58ee09e`.
- [x] (2026-08-07) Implement generic structured rollout constraints, stable selected-target projection, sequential sampling/resume, memory-safe sparse OPD loss, and checkpoint publication in PostTrain.
- [x] (2026-08-07) Add exact E2B/12B tokenizer compatibility metadata, catalog entries, project overlay, and one production work package.
- [x] (2026-08-07) Run focused tests, Ruff, targeted Pyright, import-boundary checks, lock checks, and isolated job packaging/preflight; commit and push only `feat/gemma-policy-prism-opd-e2b-12b` through commit `8ce8872`.
- [x] (2026-08-07) Diagnose nine pre-optimization admissions. Correct the verified TRL scheduler ownership defect, record the intervening generation hypotheses without claiming they worked, and use raw Trackio calls plus the earlier successful branch to identify the missing XGrammar wire-schema transformation as the actual r4-r9 cause.
- [x] (2026-08-07) Stop the launch after the r9 failure, inspect raw Trackio call evidence across every live attempt, correct the earlier diagnosis, force-terminate only the orphaned r9 workspace, and verify all ten dstack attempts are terminal and the RTX PRO 6000 is healthy, reachable, idle, and has zero busy blocks.
- [x] (2026-08-07) Restore the proven generation-only XGrammar schema projection from `f9bbb59`, remove the unproven active EOS/min-token/global-whitespace changes, preserve canonical Policy Prism validation, and pass the focused TRL suite (54 tests), Ruff, lock, and runtime-image static checks.
- [x] (2026-08-07) Add Policy Prism recipe `@2`, neutral rollout binding `@3`, training binding `@2` with ledger/native-trace checkpoint sidecars, validate the 70-base/22-project catalog and composed production job, and verify all required credentials without displaying them.
- [x] (2026-08-07) Align the feature branch and online-RL runtime closure to the deployed Trackio `0.31.5.post10` artifact protocol; no Trackio naming, UI, server, or generic artifact redesign is part of this launch.
- [x] (2026-08-07) Publish and registry-verify the corrected `0.3.1+opd.e2b12b.20260807` runtime set from PostTrain revision `16dbdf3bcba75f83fb3307117f86834e69e10aef`; the OPD TRL image is pinned at `sha256:11d3c56b88785f27813a23b3af2ceabc1b6f6c6faa190e3fed9ea2f0040fba89` and its lock digest matches the framework manifest.
- [x] (2026-08-07) Package the clean Policy Prism commit `075c35159d4af4485940ec1f33bf45a560314361` into job image `sha256:384b373c56badff5148924e2299945331de8fe1d502526999102b327719619b9`, pass isolated qualification, and prove Trackio post10 artifact upload/download byte integrity from that exact image (`sha256:4afe25b65381339710925097f450a76b2db5bab8650bd3194915d3ac1d8241f7`).
- [ ] Submit the corrected full 384-update GPU job, monitor it, preserve milestones 96/192/288/384, reconcile all required evidence, and diagnose/retry safely if necessary. Failed attempts remain operational evidence only; none reached optimizer step one.
- [ ] Publish the final rank-16 LoRA adapter privately to `carbonteq/gemma-4-e2b-policy-prism-scope-opd-from-12b-lora-v1`, verify a fresh download, and record the immutable Hugging Face revision.
- [ ] Register the exact adapter, run sealed scope then recovery evaluations sequentially, apply scientific gates, and reconcile them.
- [ ] Materialize native evaluation artifacts, finalize them into Policy Prism `evaluation-runs`, validate compatibility and evidence, then commit/push final Policy Prism lineage.

## Surprises & Discoveries

- Observation: Policy Prism implements the runtime foundation but intentionally packages no production plan.
  Evidence: `resources/scope_opd_plan/README.md` requires 50 reviewed multiple-valid and 51 reviewed constructed-incomplete entries, while `review-registry.template.json` is empty and `scope_opd_tasks._load_configured_plan` raises if `candidates.jsonl` and `task-plan.jsonl` are absent.
- Observation: the pushed PostTrain branch was 108 commits behind current main and contained nine uncommitted prototype files.
  Evidence: the prototype was preserved before the authorized feature branch alone was reset to current `origin/main`; applying the patch wholesale now conflicts because current main already contains newer Gemma model support.
- Observation: current upstream TRL's external-teacher sparse top-1 path still materializes the student's full sequence-by-vocabulary logits before slicing completion tokens.
  Evidence: the installed `DistillationTrainer.compute_loss` obtains `student_outputs.logits`; at Policy Prism sequence lengths this is not a safe 96 GiB configuration even though teacher scoring is sparse.
- Observation: the earlier 41.5-hour estimate multiplied a mismatched E4B-from-31B H200 update time by 384. It does not model the smaller E2B student, 12B teacher, or this task's actual token distribution.
  Evidence: use an initial planning envelope of 6–11 training hours, most likely 7–9, and 8–14 hours end to end. Replace this estimate with observed rolling throughput from the live run; no separate smoke is authorized.
- Observation: the original randomized template generator could not guarantee the published rule decision/task-shape matrix or a compatible shared reserve for every exact stratum.
  Evidence: the production builder now uses exact primary and reserve matrices and an immutable solver-approved selection lock. A locked rebuild is byte-identical and yields 384 primaries, 96 reserves, 77/230/77 stages, 96 targets in each quartile, and 161/35/34 rule decisions.
- Observation: replacement candidates can legitimately come from another source domain while preserving the target stage, prompt profile, quartile, and decision class.
  Evidence: the execution ledger now balances reserve selection by accepted domain and atomically rejects any acceptance that would exceed 153 of 384 targets from one domain; the reviewed primary plan itself uses 106 eCFR targets.
- Observation: every packaged prompt has substantial headroom under the E2B runtime envelope.
  Evidence: the deterministic token audit reports maximum rendered prompts of 8,740 evidence, 10,940 rules, and 7,663 graph tokens, while preserving the full 2,048/16,384/8,192 output allowances under the 40,960 per-call cap.
- Observation: PostTrain already materializes immutable Trackio model inputs inside consumer jobs; it does not expose a standalone artifact-download CLI.
  Evidence: managed evaluation calls `RunContext.input_artifact` for Trackio-backed adapter models, while the getting-started workflow requires pinning the producer artifact as a catalog model before qualification. A new local materialization command is not a launch dependency.
- Observation: the first isolated pack attempt exposed two stale Python targets in environment dependency packaging.
  Evidence: the published online-RL image runs Python 3.13.12 and all current framework and Policy Prism packages require Python 3.13, but both `KindDependencyConstraints` and the non-veRL runtime-closure override selected Python 3.12. The compiler defaults and every control closure now match the immutable runtime interpreter.
- Observation: the repository-wide Pyright command currently reports 151 existing workspace import-resolution errors after a locked all-package sync, while targeted Pyright over every changed source and test file reports zero errors.
  Evidence: focused runtime tests passed before the canonical sync (62 tests), Ruff and all eight import-linter contracts pass, `uv lock --check` and `git diff --check` pass, and the targeted changed-file Pyright invocation is clean.
- Observation: PostTrain's local placement ledger does not reveal dstack jobs submitted by other users, so `pt workers` can report an idle placement while the shared workstation is occupied externally.
  Evidence: the first full admission `policy-prism-e2b-opd-12b-r16-v1` failed before assignment with dstack `no-capacity`, while a credential-safe dstack query showed another active run holding the sole RTX PRO 6000. The failed attempt ran no model code and created no Trackio training run.
- Observation: the first capacity-waiting replacement reached the freed worker but a transient mandatory Trackio readiness probe failed before tracking or model loading; an immediate host-side probe with the identical project, token, and CA then passed.
  Evidence: `policy-prism-e2b-opd-12b-r16-v1-r1` exited in `require_remote_trackio_ready`, and the exact non-mutating `check_artifact_blobs` probe subsequently returned `trackio-readiness-ok`. No training compute occurred.
- Observation: the next replacement initialized Trackio but exposed an overly literal sparse-loss qualification guard that compared the canonical catalog model ID with the unqualified family variant ID.
  Evidence: `policy-prism-e2b-opd-12b-r16-v1-r2` rejected `models/gemma4-e2b-it@bf16` while checking for `gemma4-e2b-it`, before model loading. The guard now qualifies the immutable hub model identity and family instead; 45 focused API/sparse-loss tests pass, including the canonical catalog-ID regression.
- Observation: the corrected replacement passed tracking and request validation but the pinned TRL runtime rejected two scheduler keys duplicated through `vllm_engine_kwargs` before either model loaded.
  Evidence: `policy-prism-e2b-opd-12b-r16-v1-r3` failed in `VLLMGeneration._init_vllm` because TRL owns `max_num_batched_tokens` and `max_num_seqs`. PostTrain now validates but omits those two engine kwargs; TRL derives one sequence from the batch schedule and uses chunked prefill for long prompts. The focused training suite passes 51 tests and Ruff/diff checks are clean.
- Observation: the first replacement to load both CUDA/vLLM and the E2B policy reached logical target zero, but vLLM accepted an immediate model stop before emitting any JSON token on both provider attempts for the primary and its only compatible reserve.
  Evidence: `policy-prism-e2b-opd-12b-r16-v1-r4` recorded two rejected candidates with two provider attempts each and zero accepted targets. The exact first prompt renders correctly to 2,813 tokens and ends at `<|turn>model\n`; its strict root schema requires `resolution`, `rules`, and `completion`. The installed vLLM accepts the schema dictionary, but `SamplingParams.min_tokens` defaults to zero and removes an immediate stop token from returned token IDs. Structured policy turns now require at least one non-stop token while retaining the exact grammar for all subsequent tokens.
- Observation: requiring one token did not solve the empty structured response because Gemma's repository generation configuration declares alternate stop IDs that vLLM merges after XGrammar has compiled the tokenizer contract.
  Evidence: `policy-prism-e2b-opd-12b-r16-v1-r5` again exhausted the primary and reserve at logical target zero. The pinned tokenizer declares EOS `1`, while the model/generation configurations additionally declare `106` (`<turn|>`) and `50` (`<|tool_response>`). vLLM 0.25.1 adds those IDs to `SamplingParams.stop_token_ids`, but XGrammar's tokenizer metadata knows only canonical EOS `1`; a whitespace token can therefore satisfy `min_tokens=1` before an alternate stop is accepted and stripped, leaving no JSON object.
- Observation: neutralizing repository generation defaults did not by itself make the first rules response enter a JSON object.
  Evidence: `policy-prism-e2b-opd-12b-r16-v1-r6` loaded the corrected image and exhausted the primary plus its compatible reserve before optimizer step one with `response is not one JSON object: Expecting value: line 1 column 1`. Both candidates were attempted twice. Trace finalization then reported a missing blob, so this run is operational evidence only and is not a scientific training result.
- Observation: the previous H200/RunPod Gemma distillation path contained an additional Gemma-specific XGrammar constraint that the current reimplementation omitted.
  Evidence: `origin/exp/policy-prism-gemma4-distill` defines `_GEMMA_JSON_WHITESPACE_PATTERN = r" ?"` and adds it to every Gemma JSON-schema request. Its E4B-from-31B eight-update qualification completed with finite loss and gradients. The current `_structured_outputs` passed the schema alone, allowing unbounded grammar whitespace before the root object. The old run is useful runtime evidence but not a scientific equivalent: it used E4B/31B, H200/RunPod, smaller output caps, and admitted length-bounded outputs that the current strict environment correctly rejects.
- Observation: restoring bounded whitespace exposed that vLLM's `generation_config="vllm"` is neutral only for sampling defaults, not special-token configuration.
  Evidence: `policy-prism-e2b-opd-12b-r16-v1-r7` again rejected the primary and reserve at target zero before optimization. The exact vLLM 0.25.1 source shows `try_get_generation_config()` still loads repository EOS fields in `"vllm"` mode. E2B declares EOS IDs `1`, `106`, and `50`, while XGrammar's tokenizer metadata recognizes only canonical EOS `1`. A verified `SamplingParams(ignore_eos=True, stop_token_ids=[1])` leaves `eos_token_id=None` and the operative stop list `[1]`, so repository-only turn/tool delimiters cannot stop an incomplete grammar while canonical EOS remains grammar-controlled.
- Observation: vLLM 0.25.1 accepts a request-level `whitespace_pattern` but its XGrammar backend ignores that field when compiling JSON schemas.
  Evidence: `policy-prism-e2b-opd-12b-r16-v1-r8` still exhausted the first primary and reserve after the canonical-EOS correction. Inspection of the exact installed vLLM source showed `backend_xgrammar.py` reads only the engine-wide `StructuredOutputsConfig.disable_any_whitespace`; `get_structured_output_key()` forwards the JSON schema but not the request-level whitespace pattern. The installed `EngineArgs` and TRL `VLLMGeneration` path accept and forward `structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": true}` to `vllm.LLM`.
- Observation: the compact-whitespace replacement exposed a separate wire-schema compatibility failure before generation, which Policy Prism's downstream admission summarized misleadingly as an empty JSON response.
  Evidence: the raw Trackio call evidence for `policy-prism-e2b-opd-12b-r16-v1-r9` contains four `ValueError` calls with `The provided JSON schema contains features not supported by xgrammar`, no sampled nodes, and no token usage. Its canonical rules schema still contains nested `uniqueItems`. The OPD environment cannot infer its backend from PostTrain's embedded policy-client endpoint, so its usual local-vLLM wire transformation did not run. The corrected backend must strip only XGrammar's documented unsupported constraint keys from a private wire copy while leaving the environment's canonical schema unchanged for admission.
- Correction: the r4-r8 EOS/whitespace explanations above were hypotheses derived from the downstream Policy ledger, not the root cause. Raw Trackio call evidence now proves that every one of r4, r5, r6, r7, and r8 failed before sampling with four identical `Grammar error: Unimplemented keys: ["uniqueItems"]` calls, zero sampled nodes, and no token usage. Those five fixes could not affect this error and must not be treated as validated runtime corrections.
  Evidence: direct raw `TrackioDataSource(...)._provider_run(run_id).traces(...)` inspection for r4-r8. The same query for r9 reports the newer vLLM wrapper text `The provided JSON schema contains features not supported by xgrammar`, again with four calls and zero nodes. Policy Prism's candidate program converted transport errors into empty stage text, and the ledger then reported `response is not one JSON object`; that downstream message hid the actual compiler error.

## Stopped launch incident report (2026-08-07)

### Verdict

No OPD optimization occurred. All ten admissions are failed operational attempts, not experiments: none reached optimizer step one, produced a checkpoint, or created a candidate adapter. The launch was stopped after r9. All corresponding dstack runs are now terminal, and the shared RTX PRO 6000 is released.

### Attempt ledger

| Attempt | Proven failure boundary | Result |
|---|---|---|
| `policy-prism-e2b-opd-12b-r16-v1` | dstack could not assign the occupied 96 GiB worker | Failed before container execution; no Trackio run |
| `...-r1` | mandatory Trackio readiness probe failed transiently | Failed before model loading; an immediate identical probe later passed |
| `...-r2` | sparse-loss qualification compared the catalog-qualified student ID with the unqualified family variant | Failed before model loading; corrected by canonical model qualification |
| `...-r3` | PostTrain forwarded `max_num_batched_tokens` and `max_num_seqs` although pinned TRL owns those constructor arguments | Failed during colocated vLLM construction; corrected by omitting the duplicate keys |
| `...-r4` | XGrammar rejected nested `uniqueItems` in the canonical rules schema | Four call errors, zero sampled nodes, zero optimizer updates |
| `...-r5` | Same `uniqueItems` compiler rejection | The added structured-output `min_tokens` did not run because compilation failed first |
| `...-r6` | Same `uniqueItems` compiler rejection | Neutral repository generation defaults did not run because compilation failed first |
| `...-r7` | Same `uniqueItems` compiler rejection | Canonical-EOS overrides did not run because compilation failed first |
| `...-r8` | Same `uniqueItems` compiler rejection | Request/global whitespace changes did not address schema compilation |
| `...-r9` | Same unsupported-schema rejection, surfaced through vLLM's generic wrapper | Four call errors, zero sampled nodes; main process failed and left one orphaned engine until audited force termination |

The Policy Prism ledger's `response is not one JSON object: Expecting value: line 1 column 1` and reserve-exhaustion messages were secondary symptoms. Its candidate subprocess received provider exceptions, represented each failed call as empty stage text, retried the primary and reserve, then reported that neither candidate was admissible. The authoritative cause lives in the native Verifiers call records, not the downstream ledger summary.

### What was tried and why it did not work

The changes after r4 attempted to prevent empty generation by requiring a token, neutralizing repository stop defaults, constraining EOS IDs, bounding Gemma JSON whitespace, and setting engine-wide XGrammar whitespace policy. Those mechanisms act only after a JSON grammar has compiled. r4-r9 never reached generation: XGrammar rejected `uniqueItems` while validating the request. These changes therefore supplied no evidence about EOS or whitespace behavior and should be removed or independently justified before restart.

The diagnostic process also failed initially: it relied on terminal logs and the Policy ledger instead of querying raw Trackio call payloads. The native trace retained `calls[].error.message`, but trace finalization's missing-blob error caused the normalized view to omit those calls. Future triage must inspect raw native calls before changing generation behavior.

### Why the earlier H200 experiment worked

The earlier branch `origin/exp/policy-prism-gemma4-distill` at `84dcb6f` had already implemented `_xgrammar_json_schema()`. It recursively removed XGrammar's unsupported assertion keys—including `uniqueItems`—only from the temporary vLLM generation schema. Policy Prism retained and locally validated the untouched canonical Draft 2020-12 schema. Its first H200 smoke encountered the same error, recorded explicitly as `Grammar error: Unimplemented keys: ["uniqueItems"]`, and commit `b5396d0` added the correct boundary transformation before the successful one-step smoke and eight-step qualification.

That run was otherwise easier and scientifically different: E4B/31B on a 141 GiB H200, short 512/1536/768 output caps, one-step and eight-step gates, and admission of environment-completed length-bounded samples. The present E2B/12B plan uses a 96 GiB RTX PRO 6000, strict `finish_reason=stop`, outputs up to 16,384 tokens, 384 targets, replacement accounting, and a memory-safe sparse loss. The H200 result proves the schema-boundary solution, not the end-to-end viability or throughput of the new configuration.

### Correct restart path

1. Retain the verified pre-generation fixes through commit `19bde5d`: canonical student identity and removal of TRL-owned scheduler keys.
2. Revert or separately re-justify the unproven r4-r9 generation changes (`8a62cc0` through `7f67c90`). They did not execute in any live attempt and alter generation semantics beyond the known cause.
3. Port the earlier branch's proven XGrammar wire-schema transformation into the current TRL backend: recursively strip only `multipleOf`, `uniqueItems`, `contains`, `minContains`, `maxContains`, `patternProperties`, and `propertyNames` from a private schema copy. Never mutate the environment request or canonical trace schema.
4. Add a regression using the complete Policy Prism rules schema, not a toy schema. Assert the wire copy contains no nested `uniqueItems`, the canonical schema still contains it, and the exact pinned vLLM/XGrammar version accepts the transformed schema in a CPU-only compiler test.
5. Fix native error precedence and trace finalization: provider errors must remain explicit and cannot be collapsed into empty-text JSON admission; raw calls must remain visible even if one blob upload fails.
6. Repackage and stop at offline/isolated compiler proof. Do not submit another GPU job until the user authorizes restart. At restart, use a new run ID and inspect the first native call before allowing continued execution; this is an early gate inside the single full job, not a separate training experiment.

### Teardown evidence

The exact r0-r9 dstack provider runs report only `FAILED` or `TERMINATED`. r9 provider `pt-7d95d4ebb8a11409bf33e796` required `stop(abort=True)` because its main Python process had exited while `VLLM::EngineCore` remained alive. After termination, fleet `local-gpu-workers` reported `carbonteq-ai-workstation.lan` as `idle`, `unreachable=false`, `health_status=healthy`, `busy_blocks=0`, `total_blocks=1`. PostTrain and Trackio records were deliberately retained; no cleanup or evidence deletion was performed.

## Decision Log

- Decision: keep all work on the two user-authorized feature branches and never modify or push either main branch.
  Rationale: this preserves main and makes the experiment independently reproducible.
  Date/Author: 2026-08-06 / Codex.
- Decision: rebuild PostTrain changes on current `origin/main` instead of reviving the old-release prototype wholesale.
  Rationale: current main already contains newer model/runtime capabilities, and the prototype conflicts with them and carries a stale tokenizer fingerprint.
  Date/Author: 2026-08-06 / Codex.
- Decision: do not bypass the tokenizer fingerprint gate. E2B and 12B must share the verified canonical token-ID fingerprint before exact-token OPD is admitted.
  Rationale: teacher and student probabilities are comparable only when each integer token has the same meaning.
  Date/Author: 2026-08-06 / Codex.
- Decision: do not label agent review as `human_approved`. Introduce an auditable review provenance supported by the Policy Prism contract and perform source-level legal-structure review.
  Rationale: the user authorized Codex to conduct the review, but scientific lineage must describe who and what actually reviewed each source.
  Date/Author: 2026-08-06 / Codex.
- Decision: run exactly one full training job, with no separate smoke job, but retain packaging, isolated runtime preflight, deterministic offline checks, and live early-step gates.
  Rationale: this honors the user's no-smoke requirement without removing reproducibility or runtime safety checks.
  Date/Author: 2026-08-06 / Codex.
- Decision: use rank 16, alpha 32, dropout 0, batch 1, accumulation 1, learning rate 1e-5, 20 warmup updates, and 384 optimizer updates unless a verified implementation constraint forces a recorded revision before submission.
  Rationale: OPD supplies dense token-level teacher signal; rank 16 limits overfitting and optimizer memory, while one update per admitted logical target preserves the planned source distribution.
  Date/Author: 2026-08-06 / Codex.
- Decision: record review provenance as `independent_approved` with `reviewer_type: codex`, immutable source hashes, and source-specific legal rationales.
  Rationale: the user authorized Codex to perform the source review; describing it as human review would make the lineage false.
  Date/Author: 2026-08-07 / Codex.
- Decision: package an immutable `selection-lock.json` and require exact digest agreement during every rebuild.
  Rationale: source allocation uses a constraint solver; the lock makes the production selection reproducible without relying on solver implementation or scheduling behavior.
  Date/Author: 2026-08-07 / Codex.
- Decision: use PostTrain's existing recorded producer/consumer artifact flow rather than add a standalone `pt artifact materialize` command.
  Rationale: evaluation jobs already materialize pinned Trackio adapters in their run workspace, checkpoint recovery is handled by the trainer checkpoint contract, and Hugging Face publication should remain a separately recorded consumer operation rather than an untracked local copy.
  Date/Author: 2026-08-07 / Codex.
- Decision: bind the 96 GB execution target to fleet `local-gpu-workers` and enable a 24-hour provider-native no-capacity wait instead of pinning the hostname with zero wait.
  Rationale: the 96 GB minimum still excludes the fleet's 24 GB RTX 4090, while dstack can safely queue for the RTX PRO 6000 rather than fail before assignment when another team temporarily owns it.
  Date/Author: 2026-08-07 / Codex.
- Decision: set the effective vLLM `min_tokens` to at least one only when a structured-output contract is active, preserving any stricter configured minimum and restoring the original generation settings afterward.
  Rationale: an empty completion can never satisfy `json_object` or strict `json_schema`; suppressing only the initial stop token prevents a false operational success without weakening, rewriting, or repairing the model's schema-constrained output.
  Date/Author: 2026-08-07 / Codex.
- Decision: select vLLM's neutral generation configuration for the E2B structured-rollout engine and validate this as a generic TRL engine option.
  Rationale: the renderer and structured-output grammar own the canonical tokenizer stop contract. `generation_config: vllm` prevents repository-specific alternate EOS IDs from bypassing XGrammar while preserving normal vLLM stopping and exact generated tokens; `ignore_eos`, arbitrary output padding, and schema repair would change the experiment semantics.
  Date/Author: 2026-08-07 / Codex.
- Decision: restore the proven bounded-whitespace XGrammar contract for Gemma strict JSON-schema rollouts while leaving other model families unchanged.
  Rationale: one optional space preserves valid JSON formatting but prevents the model from satisfying generation with an indefinite whitespace/control-token prefix. This is a generation constraint, not output repair, and matches the earlier live Gemma OPD implementation that successfully reached optimization.
  Date/Author: 2026-08-07 / Codex.
- Decision: for Gemma strict structured rollouts, ignore model-repository EOS injection and explicitly stop only on the tokenizer's canonical EOS ID.
  Rationale: XGrammar can mask canonical EOS until the schema is complete, but it cannot govern extra repository EOS IDs absent from its tokenizer stop metadata. This preserves natural valid completion and exact sampled tokens; it does not pad, repair, or rewrite student output. Non-Gemma and unstructured generation remain unchanged.
  Date/Author: 2026-08-07 / Codex.
- Decision: expose vLLM's structured-output compiler configuration as a validated colocated-rollout engine option and select XGrammar with arbitrary whitespace disabled for this Policy Prism OPD binding.
  Rationale: current vLLM owns whitespace policy at engine construction rather than per request. Compact JSON remains schema-valid and preserves the exact sampled token sequence; the framework option is backend-generic while the project catalog owns its use.
  Date/Author: 2026-08-07 / Codex.
- Decision: adapt unsupported JSON Schema constraints at the private TRL-to-vLLM XGrammar boundary without mutating the environment request.
  Rationale: the embedded policy client hides the concrete vLLM route from external environments. Removing only XGrammar-unsupported assertion keywords from the generated wire grammar permits token generation while Policy Prism continues to validate the exact sampled output against its untouched canonical schema, including `uniqueItems`.
  Date/Author: 2026-08-07 / Codex.
- Decision: withdraw live-validation claims for the r4-r9 EOS and whitespace changes and do not launch from the present branch head.
  Rationale: raw native call evidence proves none of those changes reached token generation. They must be reverted or independently qualified after the known schema compiler defect is fixed; retaining them as if live-proven would confound the restart.
  Date/Author: 2026-08-07 / Codex.
- Decision: restart from a minimal correction: strip only XGrammar-unsupported keywords from the temporary wire schema, retain Gemma's proven per-request `r" ?"` whitespace pattern, and remove the active `min_tokens`, manual EOS, repository generation-config, and engine-global whitespace selections.
  Rationale: raw r4-r9 calls prove grammar compilation was the first failing boundary; this exactly reconciles with the earlier working OPD implementation without adding unrelated runtime behavior.
  Date/Author: 2026-08-07 / Codex.
- Decision: use uniquely prefixed run IDs (`opda01`, `opdsc01`, and `opdrc01`) rather than modify Trackio naming.
  Rationale: the existing adapter and UI work; unique first-eight-character prefixes make the train and qualification runs distinguishable without framework or server changes.
  Date/Author: 2026-08-07 / Codex.

## Outcomes & Retrospective

The failed r0-r9 evidence remains retained and none reached optimizer step one. The user has now explicitly authorized the corrected end-to-end restart. The minimal XGrammar correction and catalog revisions are implemented; packaging, live artifact qualification, the uniquely named full run, publication, and sealed qualification remain in progress.

## Context and Orientation

PostTrain is the framework repository at `/home/ali-awais-safdar/Post-Train/posttrain`. Policy Prism is the environment repository at `/home/ali-awais-safdar/Policy Prism`. PostTrain packages source, model, training, tracking, and execution contracts into an OCI job image; dstack places that image on `carbonteq-ai-workstation.lan`; Trackio stores native run metrics, traces, and artifacts; Observatory reads that evidence without changing it.

OPD means on-policy distillation: the current student generates a response and the frozen teacher assigns token probabilities to that exact response. Only the target-stage completion tokens receive loss. The teacher is not asked to generate a replacement answer. A reverse-KL loss encourages the student distribution toward the teacher while using sparse teacher information: the teacher's top token, a sampled reverse token, and the remaining probability mass as a tail bucket.

The Policy Prism production environment owns 384 logical targets and 96 deterministic replacements. Its stage split is 77 evidence, 230 rules, and 77 graph targets. Full trajectories may generate prerequisite or diagnostic stages, but only the explicitly selected stage is optimized. Structurally unusable attempts consume compatible replacements; schema-valid legal mistakes remain trainable because correction is the purpose of distillation.

The exact model identities are `google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7` and `google/gemma-4-12B-it` at revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Their verified canonical token-ID fingerprint is `059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416`. The run target is one NVIDIA RTX PRO 6000 Blackwell Workstation Edition with 96 GiB memory.

Key PostTrain files are `packages/common/src/posttrain/common/variants/gemma4.py` for immutable model variants, `packages/train/src/posttrain/train/online_rl.py` for policy-turn contracts, `packages/train/src/posttrain/train/integrations/verifiers.py` for native environment bridging, and `packages/train/src/posttrain/train/backends/trl/distillation.py` for the training runtime. Project overlays and work packages live in Policy Prism under `.posttrain/catalog/` and `.posttrain/work_packages/`.

## Plan of Work

First, complete Policy Prism's production inputs. Resolve source units from the repository's real handover corpus with the existing deterministic 85/15 family split and sealed-family exclusions. Generate review packets containing source identity, complete segment hierarchy, candidate legal structure, hashes, and proposed withholding only. Review every proposed multiple-valid entry for genuinely inseparable legal alternatives and every incomplete entry for a genuine parent whose required child language is withheld. Record truthful reviewer provenance, rationales, and immutable hashes. Modify the registry type narrowly if needed to represent Codex review rather than falsely asserting human approval. Build and re-check the plan with the exact E2B tokenizer, then package `candidates.jsonl`, `task-plan.jsonl`, and `summary.json` with the environment wheel.

Second, implement the reusable PostTrain runtime. Extend the provider-neutral policy-turn contract with optional structured-response schema and per-turn prompt/sequence limits. The Verifiers bridge must pass those controls generically; it must preserve the canonical schema in native evidence while producing a vLLM-compatible wire schema. The TRL policy generator must render first, calculate the allowed completion from the actual prompt length, reject over-budget prompts, group only compatible calls, and return exact IDs and log probabilities.

Project the full native trace to the selected target stage using stable call identity and prompt/completion SHA-256 values recorded by Policy Prism. Keep the full trace as evidence. Fail closed if the marker is missing, ambiguous, reordered, or hash-inconsistent. Use a deterministic sequential sampler so global update N consumes logical slot N, and persist both trainer state and Policy Prism's allocation ledger so a checkpoint resumes the same replacement chain.

Implement a narrowly scoped memory-safe TRL trainer override for the exact supported configuration. It must run the Gemma transformer without requesting full logits, select completion hidden states in chunks of 16 positions, apply the model LM head and Gemma final-logit softcap only to those positions, and compute the existing exact sparse top-1 reverse-KL plus tail-bucket objective. It must reject unsupported beta, lambda, top-k, reverse-token mode, model class, or teacher mode rather than silently changing mathematics. Unit tests compare its loss and gradients against the existing full-logit calculation on small tensors.

Add checkpoint policy so update checkpoints are saved every 16 updates, only four rolling local checkpoints remain, and every checkpoint is published as a versioned Trackio artifact before local eviction. Milestones at 96, 192, 288, and 384 must remain named and materializable. Recovery and qualification consume immutable Trackio references through the existing job input-materialization contract; Hugging Face publication is a recorded consumer operation after training reconciliation.

Third, add catalog and work-package composition. Register exact student and teacher variants, inference bindings with 49,152 model context and conservative student/teacher GPU fractions, rank-16 LoRA training settings, Policy Prism environment revision, Trackio project, dstack target, and a single 384-update work package. Set prompt cap 32,768, per-call sequence cap 40,960, evidence/rules/graph output caps 2,048/16,384/8,192, job timeout 432,000 seconds, student vLLM memory utilization 0.18, and teacher utilization 0.35. Packaging must include the exact committed Policy Prism wheel and reviewed production plan.

Fourth, validate offline and package. Run focused package tests first, then Ruff, Pyright, import-boundary checks, relevant lock validation, and `git diff --check`. Validate the catalog and work package, plan the job, pack it with `--build-missing`, and require the isolated image check to load the environment, model profiles, tokenizer contract, reviewed task plan, checkpoint backend, and required credentials. This is not a model smoke and does not reserve the GPU.

Fifth, submit and monitor the full job. Use run ID `policy-prism-e2b-opd-12b-r16-v1`. Verify the target workstation, Trackio project `policy-prism-scope-opd-e2b-12b`, initial model loading, first accepted target, finite loss/gradient metrics, teacher latency, and checkpoint publication. Thereafter monitor provider and PostTrain state, rolling throughput, replacement pressure, failures, GPU memory, and milestone artifacts. Reconcile only after the provider process terminates and require all mandatory roles and 384 admitted updates.

Sixth, publish and qualify. Materialize the exact final adapter, generate a model card from recorded training lineage and limitations, upload it privately, fresh-download it, compare file hashes, and record the immutable Hugging Face commit. Register that immutable adapter in Policy Prism/PostTrain. Run sealed scope first and require 18 included, zero failed/truncated/error traces, then recovery and require 17 included under the same gate. Do not submit them concurrently on the single GPU. Materialize each native Verifiers artifact and use `policy-prism-verifiers finalize-run` with real serving metadata to write the established five-file directories directly under `Policy Prism/evaluation-runs`; validate them and compare only matching compatibility hashes.

## Concrete Steps

All exact commands will be recorded here immediately before they are executed, after the implementation fixes their final public names. The stable working-directory convention is:

    cd /home/ali-awais-safdar/Post-Train/posttrain

for PostTrain code, catalog, packaging, submission, monitoring, reconciliation, and artifact materialization; and:

    cd "/home/ali-awais-safdar/Policy Prism"

for source review, plan building, Policy Prism tests, evaluation finalization, and Policy Prism commits.

Secrets are loaded from existing permission-protected files and are never printed. The expected high-level lifecycle is `pt catalog validate`, `pt work-package validate`, `pt job plan`, `pt job pack --build-missing`, `pt job run`, `pt run status/logs/wait/reconcile`, artifact materialization, sequential domain jobs, and Policy Prism finalization.

## Validation and Acceptance

The production plan is acceptable only if a deterministic rebuild is byte-identical and reports exactly 384 primaries, 96 reserves, the exact stage/profile/decision/quartile quotas, no sealed family, no family leakage, no invalid review hash, no prompt above 32,768 tokens, and no static sequence violation.

The runtime is acceptable only if focused tests prove dynamic token limits, structured schema routing, exact-token preservation, stable selected-target projection, sequential sampling, ledger resume, sparse-loss numerical and gradient equivalence, Gemma logit softcap, checkpoint retention/publication, and fail-closed behavior for unsupported configurations. Import boundaries must remain valid: reusable train code cannot import Policy Prism, Verifiers details stay behind the integration adapter, and common code cannot import concrete backends.

The training run is acceptable only if it terminates successfully with 384 admitted optimized target stages, finite loss and gradient norm, no unresolved teacher failures, complete native trace sync, final and milestone adapter artifacts, complete lineage, and consistent reconciliation. Operational success without these scientific gates is insufficient.

Each sealed evaluation is acceptable only if provider outcome is succeeded, evidence completeness is complete, expected and included counts match 18 or 17, failures and truncations are zero, no trace contains an error, trace sync completed, and exactly one output `verifiers-evaluation` artifact exists. Finalized Policy Prism runs must contain `manifest.json`, `traces.jsonl`, `business-kpis.json`, `engineering-metrics.json`, and `semantic-diagnostics.json`, update `evaluation-runs/catalog.json`, pass `validate-runs`, and retain Claude judgments when configured.

## Idempotence and Recovery

Plan building uses atomic deterministic outputs and has a `--check` mode. Job packing is content-addressed and may be rerun. Never reuse a run ID after provider admission; retries receive a new suffix unless PostTrain has a proven retry-resume operation tied to the same immutable checkpoint lineage. Resume only from a materialized checkpoint whose adapter, optimizer, scheduler, trainer, sampler, Policy Prism ledger, and source-plan digests all validate. Keep failed run evidence until the replacement run and finalization are proven.

Only the authorized feature branches may be force-updated, and only with `--force-with-lease` against the previously observed remote head. Existing E2B/E4B SFT evidence, Trackio runs, evaluation directories, and Hugging Face repositories are never deleted. Cleanup is limited to new OPD provider workspaces and redundant local build layers after all artifacts have been verified remotely.

## Artifacts and Notes

Current immutable inputs:

    Policy Prism foundation: de5ca4ccd8de1cbdf1b5066ad034c815426c5b00
    Policy Prism plan:       874e90205ed407126a7777221cb87dd3b58ee09e
    PostTrain base:       78d329fae89a3448cbe4f89b1744ae684e8e6358
    E2B revision:        3e22461f65e89153144f8adb70e3b8c2cc9845a7
    12B revision:        707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
    Token fingerprint:   059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416

The old PostTrain prototype remains recoverable but is not authoritative:

    backup ref: refs/backup/feat-gemma-policy-prism-opd-e2b-12b-pre-realign
    patch:      /tmp/posttrain-opd-prototype-a78df9e.patch
    SHA-256:    948b6fc0bd94e004bd9a0afb1af06b2a4f7e552b5b5f6115fd31f8472e0b5e7d

Planning duration is 6–11 hours for training, most likely 7–9 hours, and 8–14 hours for packaging-through-final-evaluations. This is an engineering estimate, not measured proof. The live run's admitted tokens per second and seconds per update will replace it after enough representative stages have completed.

## Interfaces and Dependencies

`posttrain.train.online_rl.PolicyTurnRequest` will expose generic optional structured-response and token-budget fields. `posttrain.train.integrations.verifiers.VerifiersEnvironmentRolloutBridge` will translate environment-native request metadata into that contract and preserve native traces. `posttrain.train.backends.trl.online_rl.TrlPolicyGenerator` will enforce the resulting vLLM request and exact-token behavior. `posttrain.train.backends.trl.distillation` will select the verified sparse completion-only trainer for this configuration.

Policy Prism's `scope_opd_data.py` owns reviewed source selection and plan construction; `scope_opd_tasks.py` owns task projection; `harness.py`, `scope_opd_admission.py`, and `scope_opd_ledger.py` own execution, admission, replacement, and resume. PostTrain must consume these contracts structurally and must not import Policy Prism modules into reusable framework packages.

The external services are Hugging Face for gated models and final private adapter publication, the Live Kit OCI registry for content-addressed job images, dstack for the workstation placement, Trackio for native evidence and artifacts, OpenRouter for Claude qualification judging, and Observatory for read-only inspection. Their credentials remain external to source control.

Revision note (2026-08-06): created the living plan after canonical-document and repository-state audit; corrected the obsolete 41.5-hour estimate and recorded the production-registry and sparse-loss blockers discovered before implementation.
