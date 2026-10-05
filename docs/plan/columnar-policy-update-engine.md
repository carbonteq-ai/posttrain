# Make the resolved policy-update engine scale to full AutomationBench populations

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Posttrain trains LFM2.5 models with SAMPO (a multi-turn policy-gradient algorithm with episode-level and turn-level credit) through its own "resolved policy-update engine": the code that turns one collected batch of agent episodes into frozen evidence, credit (advantages), optimizer updates and a loss. The engine was qualified on populations of a few thousand sampled tokens. The planned 100-update LFM2.5-2.6B AutomationBench run collects 128 episodes per update round with about 516,000 sampled tokens across 784 assistant turns. At that size the engine spends tens of minutes per round in single-threaded Python while the GPU idles, so 100 rounds cannot finish inside the 24-hour job limit.

After this change the same run spends seconds, not minutes, of CPU time between collecting episodes and applying updates, and the GPU scores each episode once per pass instead of once per turn. The observable result is a local simulation, run on the real episodes of the cancelled run, that reports each stage's time before and after; and then the 100-update run itself, whose per-round time is at or below the older SAMPO system's (about 7.8 minutes per update for 120 episodes in run `lfm26-sampo-cont100-g30x4-16t-lr6e5-kl1e2-20260930-r1`).

Training semantics do not change: the same rewards, the same episode-level and turn-level (sequence-level) SAMPO credit, the same objective, the same evidence guarantees. Only the representation and the order of work change.

## Progress

- [x] (2026-10-05 18:00Z) Cancelled run `manifest-steps-26-sampo-100-g16x8-20261005-r6` after it spent more than 25 minutes on one CPU core after its first collection with zero updates.
- [x] (2026-10-05 18:20Z) Built the simulation harness (`/home/hammad/projects/sim/postcollect/simulate.py`, see Concrete Steps) on r6's real 160 episodes and cached the 128 selected rollouts.
- [x] (2026-10-05 18:40Z) Measured the current engine (with interim algorithmic patches) stage by stage; see Surprises & Discoveries.
- [x] (2026-10-05 18:50Z) Milestone 1: harness checked in at `docs/research/policy-update-engine-scale/` with the recorded baseline; interim fixes kept: one-pass grouping in `update_objectives.py` and `policy_update_inputs.py`, `plan_packs` computing its update digest once, and the conditioning-only decode.
- [x] (2026-10-05 20:10Z) Milestones 2 and 3 implemented together (all engine modules, TRL and veRL adapters, distribution, replay, transport v4, recovery): train package type-checks clean. Harness on r6: admission 7.4 s, reference scores 0.3 s, four updates' loss and backward 0.8 s, memory flat at 5.0 GB.
- [x] (2026-10-05 20:40Z) Admission: one conditioning derivation per trace, inputs built once from verified records, retained evidence hashed once (`RetainedEvidence`), conditioning decode parsed on a worker-process pool (identical output, 4.67 s serial versus 1.81 s warm). Admission 3.4 s.
- [x] (2026-10-05 21:20Z) Collection side: episode and trace replay records serialized on the shared worker pool (`integrations/native_records.py`, bridge `record_encoding="process"`); sealing proves indexed lines by SHA-256 instead of parsing. Main-process CPU to convert 160 episodes 68 s to 14 s (wall 67 s to 17 s); sealing 4.4 s to 3.1 s.
- [x] (2026-10-05 22:30Z) Train tests ported (five parallel agents, disjoint files) and two restore/empty-update bugs they exposed fixed; real-data parity against the previous engine on r6 matches advantages, sampler scores, weights, ratio groups, updates and packs exactly. Committed 12be9000.
- [x] (2026-10-06 00:20Z) Milestone 4: prefix covers (`update_plan.prefix_covers`, `ExecutionPack.covers`, `ExecutionCapabilities.prefix_sharing`, TRL on), cover scoring from final hidden states with a chunked FP32 head (`backends/policy_update_logprobs.py`). Done: unit parity (scores, entropies, gradients), real-model fp16 run on r6 (5.6x faster, chunked head equals model logits to 1.9e-6). fp32 parity on the real model (LFM2.5-2.6B on CPU, two-turn cover versus per-turn: max |diff| 5.7e-6, mean 1.5e-7). Harness on r6 with a real-vocabulary stand-in head: 32 packs per update (one per episode, was 155 to 238), reference scores 9.2 s and four updates about 32 s on the RTX 3070 Ti, GPU peak 0.72 GB, process memory flat at 5.2 GB.
- [ ] Milestone 4: one forward pass per episode for exact-prefix episodes and language-model head only at sampled positions (fp32 chunks), with score parity on a real model.
- [x] (2026-10-05 21:07Z) Milestone 4b, as a 2.6B smoke run on the workstation with Verifiers 58df1306 and environment da8bb32 (`manifest-steps-26-smoke-columnar-20261005-r2`, 8 updates, f2200d28): every update applied; gradient norms 0.0082 to 0.0132; KL 0 to 5.3e-4; first update 617 s after job start (start-up, collection, two scoring passes); later updates 32 to 57 s; round 2 took 401 s to its first update, so a full round is about 9 minutes (100 rounds about 15 hours). SAMPO telemetry shows both credit levels (episode advantage magnitude 0.15/0.13, turn 0.060/0.070, turn credit share 27%/32%, zero-spread groups 0%), trainer-sampler gap 0.001, importance weights at most 1.21. Rescoring round 1's 160 live episodes on da8bb32: 0 assessment or credit errors, manifest credit on the witnessing turn in 32/32 episodes. The first smoke run (r1) failed because EnvClient's parametrized generics could not be pickled for record workers; fixed in f2200d28.
- [x] (2026-10-05 21:10Z) Submitted the 100-update run `manifest-steps-26-sampo-100-g16x8-20261005-r7` (f2200d28, 24-hour limit).
- [x] (2026-10-05) r7 applied 4 updates, then failed when vLLM woke for round 2 (CUDA out of memory in its cumem allocator). Fixed in 2290ff52: frozen scores and ratios on the host, garbage collection before emptying the cache, device memory recorded before each wake. Harness on r6 unchanged (admission 3.4 s, reference scores 9.1 s, peak GPU 0.71 GB).
- [x] (2026-10-05) Submitted a fresh 100-update run `manifest-steps-26-sampo-100-g16x8-20261005-r8` (2290ff52, 24-hour limit; image sha256:bf7b4154…).
- [x] (2026-10-05) r8 failed at its first rollout batch: the new sampler-wake memory metric was recorded one step ahead (`step + 1` on a value that was already the collection's step), so the batch's own metrics went back a step and Trackio rejected them. Fixed in d2cbf5e6 with a regression test. Its one wake reading: 5.07 GiB held by the trainer, 89.19 GiB free.
- [x] (2026-10-05) Reward design checked on r6's 128 admitted episodes (784 turns) through the real admission engine (`sim/checks/credit_sign.py`, default, `--decompose` and `--sequence` modes). Episode level: the episode reward is partial_credit and the turns' progress parts sum to it exactly (max gap 5.6e-17). Turn level: each turn's credit is the episode advantage plus the anchor-centred discounted turn return; only 6% of turns lack a sibling at their state (the old runs: 40–51%). Harmful-write turns: turn part negative in 5/5, total credit negative in 4/5 (the old loss: 7 of 15 harmful actions positive); tool-mistake turns: turn part negative in 8/10; goal turns: total credit positive in 82%. Sequence level: in each of the 4 updates, sampo@1's ratio is episode-geometric with exactly one ratio segment per episode over all of its turns' tokens (32 segments for 32 episodes).
- [x] (2026-10-05) The Doris backend on `ai-doris` held 82 GB resident (77 GB untracked by Doris) and rejected queries; restarted with the user's approval (1.2 GB after restart). Old Doris backup snapshots and this workstream's failed and smoke runs were purged (r6, r7, smoke-columnar r1 and r2).
- [x] (2026-10-05) Submitted `manifest-steps-26-sampo-100-g16x8-20261005-r9` (d2cbf5e6, 24-hour limit). First 8 updates (2 collections) applied: first update 612 s after start, second collection's first update at 1153 s (541 s per collection, about 15 hours for 100); gradient norms 0.0076 to 0.0164; KL 0 to 5.3e-4; rollouts 160/160 per collection. Round 1/2: reward 0.449/0.414, episode advantage magnitude 0.142/0.137, turn 0.057/0.074, turn credit share 29%/33%, zero-spread groups 0%, trainer-sampler gap 0.0008 to 0.0012, importance ratio at most 1.18, clip fraction 0. Device memory before each sampler wake: start 5.07 GiB allocated / 89.19 free; after collection 1 8.12 allocated, 35.55 reserved, 56.57 free (update peak 76.59); after collection 2 7.51 / 33.63 / 58.49 (peak 71.61).
- [x] (2026-10-05) r9 failed in its eighth collection (after 28 updates, about 75 minutes): vLLM ran out of memory mid-generation ("Tried to allocate 686 MiB ... 565 MiB free"; the trainer process held 36.6 GiB, vLLM 57.7 GiB). Every wake report had shown the trainer at about 8 GiB live but 33 to 36 GiB reserved after `empty_cache`, even with expandable segments, so vLLM's 48 GiB KV cache plus weights and activations only fit while its own peaks stayed low. The engine itself keeps 24 MiB live after a round in the harness (`sim/checks/live_after_updates.py`), so the pinning comes from the real model and trainer stack. ecaa6d86 prints the allocator layout and the largest live tensors at each wake; a 2-collection diagnostic run (`manifest-steps-26-memdiag-20261005-r1`) measures it before the next 100-collection submission.
- [x] (2026-10-06) Root cause of r9's memory exhaustion found and fixed (9d7d6136). The diagnostic run (`manifest-steps-26-memdiag-20261005-r1`, ecaa6d86, succeeded) showed after one round 105 allocator segments, 33.1 GiB mapped, 8.3 GiB live and 24.8 GiB free-but-split, with full-context fp16 hidden-size tensors (1, ~22k, 2048) among the live ones. The TRL population view kept the last update's ObjectiveEvaluation with its loss still referencing the autograd graph, so every leaf of the update survived backward: with gradient checkpointing on a LoRA model these include each scored context's gradient-tracked input embeddings and their full-size gradients (about 3 GiB), scattered through the update's large segments. The view now returns the differentiable loss to the trainer and keeps a detached copy; `test_update_native.py` asserts the kept evaluation has no graph (fails without the fix). The harness's new `--model lfm2-layers` reproduces it on the 8 GB card: 2,508 MiB allocated after update 0 against 806 MiB of parameters before the fix, 822 MiB after.
- [x] (2026-10-06) Submitted `manifest-steps-26-sampo-100-g16x8-20261006-r10` (9d7d6136, 24-hour limit).
- [ ] Milestone 5: recovery, transport and telemetry on the new representation; remove the per-token path; full validation; 2-update smoke on the workstation; then the 100-update run.

## Surprises & Discoveries

- Observation: the r6 stall was not in decoding episodes but in pack planning. Each of the 784 execution packs recomputed `ResolvedUpdate.digest`, which serializes the whole population (every token record) to JSON and hashes it.
  Evidence: a live `py-spy` sample of r6's main thread showed 100% of samples under `plan_packs -> emit -> ResolvedUpdate.digest -> record_digest -> dataclasses.asdict / json.dumps`.

- Observation: several group lookups scan the whole population once per group, which is quadratic. `NativePopulationInputs.from_evidence` scanned 516k actions for each of 784 views; `objective_population`, `_reduction` and `resolve_objective_term` scanned all actions per episode or per turn.
  Evidence: interim patches that group in one pass cut admission (unprofiled) from more than 25 minutes to 44 s.

- Observation: even after those patches each optimizer update costs about two minutes of pure bookkeeping, before any model compute. The simulation replaces the transformer with a stand-in that returns one tensor per turn, so the time below is CPU and GPU-synchronization overhead only.
  Evidence (local RTX 3070 Ti, stand-in model):

      [stage] admit (resolve+plan)            44.04s
      [stage] trl population view              6.86s
      [stage] reference scores                21.81s
      [stage] update 0 loss                  116.26s   (includes old-score freezing)
      [stage] update 0 backward               21.56s
      [stage] update 1 loss                   97.81s
      [stage] update 1 backward               38.54s
      [stage] update 2 loss                   99.97s
      [stage] update 2 backward               49.39s
      [stage] update 3 loss                  122.46s
      [stage] update 3 backward               72.06s

  Total engine overhead per round is about 690 s (11.5 minutes). Process memory grew from 6.1 GB to 17.2 GB over four updates, and backward time grows with each update.

- Observation: every r6 episode is an exact prefix chain: each turn's conditioning context is the beginning of the next turn's. The engine nevertheless runs one full forward pass per turn over that turn's whole history, which is 4.67 times more tokens than one pass per episode.
  Evidence: over the 128 selected rollouts, the per-turn contexts sum to 6,362,187 tokens per scoring pass; the final contexts of the episodes sum to 1,363,646.

- Observation: two thirds of each saved episode's bytes are assessment archives that policy scoring never reads, yet the trainer validated the whole Verifiers `Episode` (including restoring those archives) to read the message graph.
  Evidence: in one r6 trace, `assessment_sources` 819 KB, `assessment_views` 482 KB, `assessment_batches` 194 KB, `nodes` 254 KB.

- Observation: without any interim patches, admission alone did not finish within the simulation's 25-minute cap (it was cut off after more than 23 minutes in admission).
  Evidence: `simulate.py --baseline` (digest cache disabled, run on the pre-patch grouping code) timed out after `retain population` with no `admit` line.

- Observation: the resolved scorer materializes full-vocabulary logits for every position of every forwarded context, then keeps only the sampled positions. The older TRL path computes log-probabilities from hidden states in fp32 chunks of 128 positions and never holds whole-sequence logits.
  Evidence: `backends/policy_update_scoring.py` `score_actions` calls `model(input_ids=...)` and indexes `output.logits`; the 2.6B binding sets `logits_chunk_size: 128` and `logits_float32: true` for the TRL path. One 24k-token context is about 3.2 GB of half-precision logits over LFM2.5's 65,536-token vocabulary.

- Observation: the harness does not measure GPU compute. Its stand-in replaces the transformer, so attention kernels (SDPA), the language-model head, backward through the LoRA model, gradient checkpointing and memory are not represented. Training updates do not use CUDA graphs (vLLM uses them only for rollout decoding), and variable sequence lengths make graph capture or `torch.compile` a separate later question.
  Evidence: `backends/trl/common.py` sets `attn_implementation="sdpa"`; the stand-in `sampled_logprobs` in `simulate.py`.

- Observation: after the engine rewrite, admission was dominated by evidence handling, not bookkeeping: JSON parsing of the retained episodes (3.3 s), hashing the same bytes four times (1.4 s), and deriving each turn's context twice while re-checking every earlier node per turn.
  Evidence: cProfile of `admit (resolve+plan)` on r6 after Milestones 2 and 3 (12.3 s profiled).

- Observation: converting each collected episode into a training rollout cost about 340 ms of main-thread CPU, almost all of it serializing the episode and its trace back into replay records (Verifiers `to_record`, dominated by assessment-archive normalization) and encoding them as JSON. Pickling an episode costs 1.6 ms, so the serialization can move to worker processes.
  Evidence: cProfile of `project episodes` (85.6 s profiled, of which `_native_record` 69 s and appends 16 s); `checks/record_costs.py`: `to_record` 34 ms (episode) and 32 ms (trace), `pickle.dumps` 1.6 ms.

- Observation: LFM2.5-2.6B has a 128,000-token vocabulary (the 1.2B models have 65,536), so full-vocabulary FP32 logits cost about 512 KB per scored token; one update of r6's population would retain tens of gigabytes of them for backward.
  Evidence: `config.json` vocab_size; an episode-sized forward with full kept-row logits ran out of memory on the 8 GB card while per-turn forwards fit.

- Observation: scoring all turns of an episode inside one forward of its longest context is exact in arithmetic but not bitwise in half precision: on two r6 episodes with LFM2.5-2.6B in fp16 the per-token log-probability difference from per-turn forwards was 7.7e-4 on average (max 3.7e-2), about a fifth of the trainer-sampler gap (0.0037 mean). Old, reference and current scores all use the same covers, so ratios stay consistent within a run.
  Evidence: `sim/checks/model_cover_parity.py 9000 2`: per-turn 12.1 s versus shared 2.1 s for 12 turns (64,121 versus 12,406 forwarded tokens); chunked head versus the model's own logits max |diff| 1.9e-6.

- Observation: the older SAMPO system (TRL's own trainer on flattened episode rows) took 467 s per update on average for 120 episodes: 198 s of rollouts and 219 s of actor update, about 13 hours for 100 updates.
  Evidence: `posttrain query --sql "select run_id, avg(update_seconds), avg(rollout_seconds), avg(actor_seconds) from updates where run_id like 'lfm26-sampo-cont100%' group by run_id"` from `apps/lab`.

- Observation: r7 applied four updates in round 1 and then failed at the start of round 2: `wake_up` returned "CUDA Error: out of memory" from vLLM's cumem allocator, so active sampling found no live sampler and exhausted its candidate pool. The new engine kept frozen old and reference scores (one float per population token) and dense ratios on the GPU for the whole round. They are small, but they outlive the update, so their caching-allocator segments stay reserved and `empty_cache` cannot hand those pages back before vLLM re-maps its 0.68 share of the card.
  Evidence: r7 job log at the round-2 wake; the old engine held these values in Python lists on the host.

- Observation: turns whose tool calls failed are not all penalized, by design: the environment fines only tool mistakes (rejected arguments, an unknown tool, an ID never returned), not empty searches. Turns that lowered partial credit and were followed by a recovery carry almost no turn credit (+0.004 on average), because the discounted return from that turn includes the recovery; their sign comes from the episode part.
  Evidence: `automationbench_v1/turn_rewards.py` (environment da8bb32); `credit_sign.py --decompose` on r6.

- Observation: after an update round, `empty_cache` leaves about 34 GiB reserved against 8 GiB allocated, so the sampler wakes with about 57 GiB free. The colocated vLLM woke successfully at that margin in r9's second and third collections, but caching-allocator fragmentation is the remaining memory risk if later populations run longer contexts (update peak 72 to 77 GiB).
  Evidence: r9 job log, "device memory before sampler wake" lines.

## Decision Log

- Decision: redesign the representation instead of keeping the interim patches.
  Rationale: every hot spot comes from one choice: each sampled token is an individual immutable Python object carrying its own identity, and each layer re-derives and re-checks the previous layer's output. Caching digests or grouping loops reduces the constant but keeps millions of objects, per-token tensors and a per-token autograd graph. The user asked for an architectural fix rather than workarounds.
  Date/Author: 2026-10-05, Claude with the user.

- Decision: keep the interim conditioning-only decode (`decode_native_population(..., content="conditioning")`) as the permanent design; drop the weak-reference digest cache and the compositional-digest change once Milestone 2 removes their callers.
  Rationale: decoding only the message graph is the right boundary (it is all scoring reads, and it uses Verifiers' own `Branch` schema). The digest cache only papered over repeated whole-population serialization.
  Date/Author: 2026-10-05, Claude.

- Decision: one forward pass per episode is allowed only when admission proves the episode's turn contexts form an exact prefix chain; otherwise the engine keeps per-turn scoring.
  Rationale: per-turn scoring exists so that each action is scored under the exact context it was sampled with. When contexts are exact prefixes, causal attention makes the single pass score every action under the same context, so the guarantee is preserved by proof, not assumed. Episodes with branching or rewritten history (for example a renderer that drops earlier reasoning) still get exact per-turn contexts.
  Date/Author: 2026-10-05, Claude.

- Decision: one pass per episode (Milestone 4) is required for the 100-update run, not an optional optimization.
  Rationale: at 4.67 times the scoring tokens, the actor phase would be roughly four to five times the older system's 219 s per update, so 100 rounds would take about 25 to 30 hours and exceed the 24-hour limit. With one pass per episode the GPU processes about 1.36M tokens per pass, close to what TRL's flattened trainer processed, and the expected round time is 7 to 9 minutes (12 to 15 hours for 100 rounds). This estimate assumes scoring only sampled positions as the old path does and is unmeasured until Milestone 4b.
  Date/Author: 2026-10-05, Claude.

- Decision: CPU-bound work that needs no trainer state runs on one shared pool of up to 8 worker processes started through a fork server (`integrations/native_records.py`): parsing retained records for admission and serializing episodes for preservation. The trainer process keeps every ordered or stateful step (appends and their byte spans, identity and schema checks, observation, projection).
  Rationale: the trainer runs standard CPython with the global interpreter lock; JSON parsing and pydantic serialization hold it, so threads do not help. A fork server starts workers from a fresh interpreter, so they never inherit the trainer's CUDA context, threads or locks. Bridges built directly (as tests do) keep thread encoding through an explicit `record_encoding` field, not a silent fallback.
  Date/Author: 2026-10-05, Claude.

- Decision: a contribution's dependencies are whole turns, not the selected subset of a turn's tokens.
  Rationale: a turn's forward pass scores all of its tokens anyway, so whole-turn dependencies change no scores, weights or ratios; they only make the statistic-capacity check count every position of a dependency turn (conservative). This removes the last per-token sets from scheduling.
  Date/Author: 2026-10-05, Claude.

- Decision: policy scores are formed from the decoder's final hidden states with a custom autograd function that computes the output head and FP32 log-softmax in row chunks and recomputes them in backward, instead of requesting logits from the model.
  Rationale: retaining full-vocabulary FP32 logits (or even model-dtype kept rows) for every scored token does not fit an update at r6's size; the chunked function retains only hidden states and one normalizer per token. It forms logits with the model's own linear op and applies the config's `final_logit_softcapping` when declared (the only post-head transform in current Transformers causal LMs, used by the Gemma family), so it reproduces model logits exactly; tests check values, entropies and gradients against full log-softmax autograd.
  Date/Author: 2026-10-05, Claude.

- Decision: internal digests and the recovery checkpoint format change version; checkpoints written by the old engine are not resumable by the new one.
  Rationale: identities become hashes of compact columns instead of JSON of token objects. The project starts fresh runs rather than resuming (standing user instruction), and no in-flight run depends on old checkpoints.
  Date/Author: 2026-10-05, Claude.

- Decision: no population-sized tensor outlives an update on the device. Frozen scores are moved to the host when frozen and moved to the device per pack, ratios are gathered on the host, the trainer collects garbage before `empty_cache`, and it records `train/rl/device_memory_*` before every sampler wake.
  Rationale: the colocated sampler can wake only into memory the trainer has truly released. Host copies cost microseconds per pack. Wake-time memory telemetry turns any later failure into a measured cause instead of a guess.
  Date/Author: 2026-10-05, Claude.

- Decision: telemetry state kept on a population between optimizer steps holds detached values only; the differentiable loss lives only in the trainer's step.
  Rationale: anything that keeps a graph alive keeps every leaf of the update alive, and those leaves (gradient-tracked input embeddings under gradient checkpointing, plus their gradients) are context-sized; on a colocated sampler they decide whether the next wake fits.
  Date/Author: 2026-10-06, Claude.

## Outcomes & Retrospective

The resolved engine went from unable to finish one round at r6's size (over 25 minutes on one core before any update) to about 9-minute rounds on the workstation with four optimizer updates each. The changes that mattered: a population as a table of turns with per-token arrays (no per-token objects, identities computed once), evidence verified once at admission, whole-tensor scores and loss, one forward per episode for exact-prefix chains, sampled-position log-probabilities from hidden states in chunks, and stateless CPU work on a fork-server process pool. A real-data parity check against the previous engine matched every credit, weight, ratio group, update and pack on r6.

Lesson: the simulation decoded episodes differently from the production client (plain classes instead of parametrized generics) and so missed a pickling failure that the first workstation smoke run hit immediately. Simulations must construct inputs through the same code paths as production; the harness now decodes episodes as `WireEpisode`.

## Context and Orientation

All paths are relative to the rl repository root. The engine lives in `packages/train/src/posttrain/train/`. Its pieces, in the order a training round uses them:

`backends/trl/policy_rollouts.py` (`collect_active_resolved_population`) collects episodes through the Verifiers bridge (`integrations/verifiers.py`), selects 16 prompt groups whose rewards differ, and asks the bridge to seal them (`retain_population`, writing a JSONL file of native episodes, the "retained evidence").

`backends/policy_update_admission.py` (`AdmittedNativePopulation.from_retained_artifact`) re-reads those bytes, checks their SHA-256 digest, and builds the frozen population. `update_evidence.py` (`population_from_rollouts`) creates one `ActionRecord` (defined in `update_records.py`) per sampled token. An action is identified by `ActionRef(episode_id, branch_id, turn_id, token_index)`. A "conditioning view" (`ConditioningView`) is one assistant turn's exact input context: the token ids of the path from the root of the message graph to that turn's node.

`update_credit.py` prepares credit. `SampoCreditEstimator` calls `compute_sampo_advantages` (in `sampo_advantages.py`), which already returns one advantage array per episode row; `NativeCreditRows.project` then explodes it into one `ActionCredit` per token.

`update_objectives.py` builds the objective population (one contribution per turn), and per update the `ResolvedObjectiveTerm`: per-token policy and KL weights and the "ratio support" (which tokens share one importance ratio: one token, one turn, or, for `sampo@1`, the whole episode, called episode-geometric).

`update_plan.py` schedules contributions into optimizer updates (`resolve_updates`; with budget 32 episodes, 4 updates per round) and splits each update into execution packs (`plan_packs`), groups of conditioning views sized to the GPU context budget.

`backends/policy_update_inputs.py` (`NativePopulationInputs`) materializes model inputs for a view from the retained bytes. `backends/policy_update_scoring.py` runs the model per view and returns one scalar tensor per token (`score_actions`, `freeze_population_scores`). `backends/policy_update_execution.py` (`compute_resolved_loss`) and `backends/policy_update_math.py` (`evaluate`) build the loss token by token. `backends/trl/policy_updates.py` (`ResolvedTRLPopulation`) plugs this into TRL's trainer; `backends/trl/policy_job.py` drives rounds.

Supporting modules that also carry per-token records: `update_sampler_correction.py` (vLLM sampler-mismatch correction weights), `update_telemetry.py` (metrics), `update_spans.py` and `update_process_credit.py` (optional span and process credit), `update_transport.py` and `update_recovery.py` plus `backends/policy_update_recovery.py` (checkpointing the population), `update_distribution.py` and `backends/policy_update_distributed*.py` (multi-GPU), and the veRL backend under `backends/verl/`.

Tests live in `packages/train/tests/`, mainly `test_update_*.py`, `test_resolved_telemetry.py`, `test_sampo*.py`, `test_trl*.py`, `test_verifiers_population_artifact.py`.

## Plan of Work

The core change is a column-based ("columnar") population. Instead of a tuple of token objects, a population is a table with one row per conditioning view (turn) and flat arrays with one entry per sampled token, in a fixed canonical order (views in admission order, tokens in ascending native index). Every later structure refers to positions in those arrays.

In a new module `packages/train/src/posttrain/train/update_population.py`, define `PopulationTable`. It holds the population identity fields that exist today (id, native evidence reference and digest, policy versions, selector digest), the view table (view ids, episode index, turn ordinal, context length, input digest, native coordinates), `token_offsets` (int64, one more than the number of views; view `v` owns tokens `token_offsets[v]` to `token_offsets[v + 1]`), `token_index` (int32 native index of each sampled token within its node), episode and branch identities per view, relations as arrays of view indices, and an `exact_prefix` flag per episode. Its `digest` is computed once in construction from the identity fields and the array bytes, and stored as a field. An `ActionRef` remains available as a derived, on-demand view of one position for error messages and the public telemetry vocabulary, never as storage.

`PreparedCredit` becomes an advantage array aligned with the token order plus its existing metadata; `NativeCreditRows.project` is replaced by a direct concatenation of the estimator's per-row arrays restricted to sampled positions. Selections (`policy_selection`, `kl_selection`, semantic spans) resolve to boolean masks. The `ResolvedObjectiveTerm` becomes arrays: policy weight, KL weight, and a ratio segment id per token (with the segment count). Scheduling keeps its current rules but works on episode and turn rows; an update is a sorted array of view rows plus its contribution identities. Packs are lists of view rows; when Milestone 4 lands, an exact-prefix episode's views form one "episode pack" scored by one forward pass.

Admission becomes the single place that checks evidence. It verifies the retained bytes' digest, decodes only the message graph (`content="conditioning"`), derives every view's conditioning record once, checks it against the rollout's frozen record, materializes the view's token ids once, proves or rejects exact-prefix chains, and builds the immutable table. Later layers receive the typed table and do not repeat those derivations; `ResolvedPolicyPopulation` and `ResolvedTRLPopulation` stop re-running objective resolution and pack planning as checks.

Scores become one float32 tensor per pass aligned with the token order (`FrozenScores.values`), validated with one finiteness check. `score_actions` scatters each pack's gathered log-probabilities into the population-ordered tensor. The loss in `policy_update_math.evaluate` is rewritten as whole-tensor operations: log-ratio as current minus old; segment means with `index_add_` over ratio segment ids; exponentiate; PPO clip; multiply by advantages, policy weights and sampler-correction weights; sum. The sampled-k3 KL term is computed with `torch.where` over both branches using inputs clamped so the unused branch cannot produce non-finite values or gradients. Clipping indicators and per-token ratios stay available for telemetry as tensors.

Recovery, transport and telemetry move to the table: checkpoints store the arrays (NumPy `.npy` inside the existing checkpoint component mechanism) with a bumped format version, and telemetry computes the same metric values from arrays.

## Milestones

Milestone 1 makes the measurement repeatable. The harness lives in `docs/research/policy-update-engine-scale/simulate.py` with a README that names the r6 data location, how to regenerate the cache, and the baseline table above. Acceptance: running the harness prints the stage table and writes `summary.json` with digests of updates and credit.

Milestone 2 introduces `PopulationTable` and moves credit, selections, objective terms, scheduling and packs onto it, keeping the existing per-token code temporarily as a reference implementation in tests only. Acceptance: new parity tests build both representations from the same fixtures (`test_resolved_telemetry.py` fixtures, the SAMPO fixtures, and a synthetic multi-turn population of 128 episodes) and assert equal advantages per token, equal policy and KL weights and denominators, equal ratio supports, equal update membership and equal pack membership; and the harness admits r6's population in under 5 seconds.

Milestone 3 moves scores and the loss to tensors. Acceptance: parity tests compare the vectorized loss, policy loss, KL loss, clipped-token set and gradient of a small real model's LoRA parameters against the token-by-token reference on fixtures, within 1e-6 relative for float32; and the harness with the stand-in model reports under 5 seconds of overhead per optimizer update and flat memory across updates.

Milestone 4 adds episode packs. Acceptance: with `LiquidAI/LFM2.5-1.2B` on the local GPU, per-token log-probabilities from one pass per episode equal the per-turn passes within bf16 tolerance (documented), on a handful of r6 episodes truncated to fit; and the harness reports scoring token volume reduced from 6.36M to 1.36M per pass.

Milestone 4 also changes how scores are computed from the model: run the transformer to get hidden states, gather the hidden states at the positions preceding sampled tokens, and apply the language-model head and an fp32 log-softmax to those rows in chunks (the chunk size comes from the binding's `logits_chunk_size`), so whole-sequence full-vocabulary logits are never materialized. Acceptance adds: the gathered-position scores equal the full-logits scores within fp32 tolerance on the 1.2B parity set.

Milestone 4b measures what the harness cannot: GPU time. A short workstation job (or a dstack task in the job image) loads LFM2.5-2.6B with the run's LoRA settings, kernels and precision, and runs the harness's reference-score pass and one optimizer update on r6's population with the real model, reporting forward and backward seconds per pass and peak GPU memory. It runs once on the current engine and once after Milestones 2 to 4. Acceptance: the after-run's actor time per round (old scores, reference scores and four updates) is at or below the older system's 219 s per update scaled to 128 episodes, and the measured numbers replace the estimate in this plan's Decision Log.

Milestone 5 finishes the migration: recovery and transport on arrays, telemetry from arrays, veRL and distributed adapters updated, the per-token reference code deleted, the full validation ladder green, a 2-update 2.6B smoke run on the workstation, and then the 100-update run.

## Concrete Steps

Simulation: setup, data layout and the recorded baseline are in `docs/research/policy-update-engine-scale/README.md`. From the worktree root:

    /home/hammad/projects/sim/postcollect/venv/bin/python docs/research/policy-update-engine-scale/simulate.py --data /home/hammad/projects/sim/data-r6 --project apps/lab --out /tmp/engine-scale            # admission stages
    /home/hammad/projects/sim/postcollect/venv/bin/python docs/research/policy-update-engine-scale/simulate.py --data /home/hammad/projects/sim/data-r6 --project apps/lab --out /tmp/engine-scale --train    # plus reference scores and every optimizer update
    ... --profile "admit (resolve+plan)"    # cProfile one stage

Release chain for the 2.6B run (from the rl-perf worktree root, after merging the Verifiers/environment pin branch). Verifiers 58df1306 and the environment release replace 24c12379/f587146; posttrain-train now also declares numpy, so uv.lock and its digest change:

    UV_HTTP_TIMEOUT=300 uv lock          # then restore by hand: `revision = 3` and vllm `version = "0.29.1.dev4"` (no `+precompiled`)
    UV_HTTP_TIMEOUT=300 uv lock --check
    .venv/bin/posttrain-release lock-dependencies
    .venv/bin/posttrain-release lock-runtime-dependencies     # job-kind locks and profiles for online-rl-trl-py312 and eval
    nice -n 5 .venv/bin/posttrain-release images publish --registry registry.lan/carbonteq \
        --receipt-root /home/hammad/projects/rl/.posttrain/state/release-receipts \
        --trust-bundle /usr/local/share/ca-certificates/carbonteq-local-ai-caddy.crt \
        --framework-version 0.4.15.dev1 --variant online-rl-trl-py312 --variant eval
    # then replace the old trl-fork lock digest with sha256(uv.lock) in apps/lab/.posttrain/catalog/*.yaml

Expected: `apps/lab/tests/test_catalog.py::test_peft_bindings_settings_and_quantization_load_from_filesystem_catalog` passes again. Four tests remain red from before this plan (release constraints, veRL release profile and CLI template still name Verifiers e6a3d9bb); they do not affect the online-rl-trl-py312 job kind.

Tests (from the worktree root): `nice .venv/bin/python -m pytest -q packages/train/tests/test_update_*.py packages/train/tests/test_resolved_telemetry.py packages/train/tests/test_sampo*.py packages/train/tests/test_verifiers_population_artifact.py packages/train/tests/test_trl*.py`, then the full ladder from `AGENTS.md`.

## Validation and Acceptance

The work is accepted when the harness on r6's population shows admission under 5 seconds, per-update engine overhead under 5 seconds, flat memory across updates, and scoring volume of 1.36M tokens per pass; when the parity tests pass; when the full ladder passes; and when a 2-update 2.6B smoke run on the workstation applies both updates with finite gradient norms, the SAMPO telemetry (episode and turn advantage magnitudes, turn credit share) in the same ranges as r6's rescoring analysis, and per-round wall time reported in Trackio. The 100-update run is then submitted with the 24-hour limit.

## Idempotence and Recovery

The harness writes only under its `--out` directory and the cache file; deleting `population.pkl` forces a rebuild. Engine changes are on the `wip/automationbench-reward-redesign-2026-10-04` branch in the `rl-perf` worktree and land in reviewed commits per milestone; any milestone can be reverted independently until Milestone 5 deletes the per-token path.

## Interfaces and Dependencies

No new third-party dependencies: NumPy and PyTorch are already present. The public request and settings types (`SAMPORequest`, `SAMPOSettings`, `PolicyUpdateSettings`) and all metric names stay unchanged. Everything below is internal to `posttrain.train`.

Token order. A population's sampled tokens have one canonical order: conditioning views in admission order, and within a view the native token indices in ascending order. A "position" is an index into that order. Every per-token value in the engine is an array indexed by position; nothing stores one Python object per token.

`update_records.py`:

    @dataclass(frozen=True, slots=True)
    class ConditioningView:            # one sampled assistant turn
        id: str                        # "<trace id>/node-<node index>"
        native_ref: str
        token_ids_ref: str             # JSON coordinates: trace_id, prefix_nodes, node_index
        attention_ref: str
        positions_ref: str
        template_revision: str
        digest: str                    # input digest of the exact context tokens
        context_tokens: int
        episode_id: str                # new
        branch_id: str                 # new
        sampled: tuple[int, ...]       # new: ascending native indices of eligible sampled tokens

    @dataclass(frozen=True, slots=True, eq=False)
    class PopulationSnapshot:
        id, native_evidence_ref, native_evidence_digest
        conditioning: tuple[ConditioningView, ...]
        spans: tuple[SemanticSpan, ...]
        relations: tuple[PopulationRelation, ...]   # members and expected_members are view ids
        versions: PolicyVersions
        selector_digest: str
        # computed once in __post_init__ (read-only arrays):
        offsets: np.ndarray            # int64, len(conditioning) + 1; view v owns positions offsets[v]:offsets[v+1]
        view_of: np.ndarray            # int32 per position
        episode_of: np.ndarray         # int32 per position, index into episodes
        episodes: tuple[tuple[str, str], ...]   # (episode_id, branch_id), first-seen order
        digest: str
        size -> int; view_index(view_id) -> int; action(position) -> ActionRef; positions(actions) -> np.ndarray
        select_roles(roles) / select_spans(span_ids) -> np.ndarray[bool]

`ActionRef` stays as the addressing type at boundaries (spans, error messages, tests); `ActionRecord` and `ActionCredit` are removed.

`update_credit.py`: `PreparedCredit.advantages: np.ndarray` (float64 per position, read-only) replaces `values`; `eq=False`; `digest` computed once from metadata plus the array bytes. `NativeCreditRows` keeps the rollout rows and, per row, the array of positions its sampled completion indices map to; `project(values)` gathers estimator rows into the position array.

`update_plan.py` and `update_objectives.py`:

    ContributionRef(id: str, view: int, dependency_views: tuple[int, ...])       # one per turn
    ObjectivePopulation(definition_id, credit_digest, contributions, policy: np.ndarray[bool],
                        kl: np.ndarray[bool], required_statistics, statistic_bytes_per_action, contract_digest)
    ResolvedUpdate(population, objective, schedule_digest, epoch, minibatch,
                   contributions: tuple[int, ...], occurrence_ids, views: tuple[int, ...], discarded_contributions)
        # views: sorted dependency views; digest computed once from the population and objective digests
    ExecutionPack(update_digest, index, views: tuple[int, ...], context_tokens, layout: "turn" | "episode")
    ResolvedObjectiveTerm(spec, update_digest, credit_digest, policy_weight: np.ndarray, kl_weight: np.ndarray,
                          ratio_segment: np.ndarray[int32, -1 outside support], segment_count: int,
                          policy_denominators, kl_denominators, zero_policy_episodes, zero_kl_episodes, parameter_version)

`backends/policy_update_scoring.py`: `score_positions(model, snapshot, views, *, read_input, device, score_temperature, chunk_size) -> torch.Tensor` returns a float32 tensor of length `snapshot.size` holding scores at the positions of the given views (zero elsewhere), scattered from per-pack gathers. `FrozenPopulationScores.values` is a detached float32 tensor of length `snapshot.size`.

`backends/policy_update_math.py`: `evaluate(term, credit, scores) -> ObjectiveEvaluation` computes the loss with whole-tensor operations; `ObjectiveEvaluation.ratios` and the clipped set become tensors and a boolean mask.

`update_sampler_correction.py`: correction weights are a float64 array per position; sequence modes sum log differences per episode with `np.add.at`.

`update_transport.py` and `backends/policy_update_recovery.py`: schema `posttrain.resolved-population.v4` stores views with their sampled indices, credit and correction as lists aligned with positions, and update view lists; v1 to v3 sidecars are rejected with a clear error (no in-flight run depends on them).

`decode_native_population(evidence, *, format, content="traces" | "conditioning")` in `integrations/verifiers_population_artifact.py` is the decode entry point; scoring uses `content="conditioning"`.
