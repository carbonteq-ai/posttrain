# Add a veRL backend for Qwen 3.5 GRPO and distillation

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This plan follows `docs/templates/PLAN.md`; the planning skill's usual `.agents/PLAN.md` entrypoint is not present in this repository.

## Purpose / Big Picture

After this change, callers can select a versioned `verl@...` `TrainingBinding` for the existing `train.grpo` and `train.distill` operations. The public operations, Verifiers environment bridge, exact token identities, response masks, native traces, artifacts, and observation contract remain backend-neutral. The backend is named generally as veRL, while the qualification notes and preflight validation state that this first supported slice is Qwen 3.5 only.

A developer can demonstrate the feature without a GPU by running adapter and dispatch tests that render an immutable veRL launch specification and reject unqualified model families. The release gate is a real GPU run in an isolated veRL environment: one Qwen 3.5 GRPO update and one Qwen 3.5 student/teacher distillation update using fresh Verifiers trajectories.

## Progress

- [x] (2026-07-22) Read the canonical post-training baseline, current TRL adapters, Verifiers bridge, catalog selections, and upstream veRL Qwen 3.5 GRPO/distillation code.
- [x] (2026-07-22) Confirm the change preserves the frozen product baseline: veRL is a private adapter for the already-defined operations and does not change job meaning.
- [x] (2026-07-22) Added backend dispatch for `train.grpo` and `train.distill` while preserving explicit runner injection in tests and hosts.
- [x] (2026-07-22) Added the isolated veRL launcher contract, Qwen 3.5 qualification checks, config translation, immutable Hub materialization, result ingestion, and unit tests.
- [x] (2026-07-22) Added portable Verifiers bridge state and the veRL agent-loop adapter that returns exact prompt ids, response ids, response masks, rollout log probabilities, rewards, and native trace identity.
- [x] (2026-07-22) Added support notes that expose a general `verl` backend but qualify only Qwen 3.5. A machine-independent catalog binding remains intentionally absent until the fork environment and lock are pinned because the binding requires absolute isolated-runtime paths.
- [x] (2026-07-22) Ran focused and package tests, Ruff, Pyright, import-boundary checks, and `git diff --check`.
- [x] (2026-07-22) Prototyped NF4 QLoRA in a local veRL branch and validated a Qwen 3.5 2B adapter backward pass, then parked it when the user chose a smaller unquantized model instead.
- [x] (2026-07-22) Standardized shared training-runtime names across adapters and added a separate `TrainingBinding.backend_options` escape hatch with protected selection paths.
- [x] (2026-07-22) Ran a one-step Qwen 3.5 0.8B integration pass, then rejected it as qualification evidence because equal zero rewards produced a zero gradient.
- [x] (2026-07-22) Audited the native Verifiers v1 AutomationBench port against Zapier AutomationBench `a321764` and restored the upstream default Zapier meta-tool contract (`search_tools` and `execute_tool`) while retaining API mode as an explicit alternate toolset.
- [x] (2026-07-22) Added upstream-compatible `limited_zapier` concrete-tool mode, restored the original 50-turn behavior, raised the environment context budget to 8192 tokens, and exposed the rollout wall-clock timeout as an environment-owned setting.
- [x] (2026-07-22) Proved 16 native limited-tool AutomationBench trajectories with within-group reward variance and multi-turn tool calls; two-way vLLM scheduling completed the batch in about two minutes on the local 8 GiB GPU.
- [x] (2026-07-22) Fixed veRL FSDP dense-input chunked entropy and added GPU regression coverage; old-logprob computation now passes at the 8192-token configured capacity.
- [x] (2026-07-22) Recorded the qualified single-GPU optimizations in `docs/tooling/verl/README.md`, added the workspace-wide maintained-fork documentation convention, and created the candidate veRL fork's root `CARBONTEQ_FORK.md` delta ledger.
- [x] (2026-07-22) Completed the replacement AutomationBench qualification: two prompt groups, eight generations per group, two optimizer steps, within-group reward variance, tool-using multi-turn evidence, non-zero gradients, changed adapter weights, and a step-two post-update rollout.
- [x] (2026-07-22) Corrected the required context target to 32768 tokens, added normalized `kv_cache_dtype` translation into veRL's vLLM engine kwargs, and documented TurboQuant K8V4 as a rollout-only memory optimization.
- [x] (2026-07-22) Re-locked a separate veRL runtime to PyTorch 2.11.0+cu130, Transformers 5.14.1, and vLLM 0.25.1; verified CPU synchronization APIs, K8V4 allocation/generation/sleep/wake, and matched normal-versus-K8V4 quality probes through 32700 input tokens.
- [ ] Run the 32K GRPO qualification with normal FP16 KV cache, including one trajectory beyond the old 8K boundary; K8V4 is excluded because it failed matched long-context recall.
- [ ] Run and record the full GPU GRPO and distillation release gates in the isolated veRL runtime.
- [x] (2026-07-22) Added normalized native-MTP rollout translation for veRL, corrected the Qwen 3.5 catalog method to `mtp`/one speculative token, and rejected other speculative methods before Ray starts.
- [x] (2026-07-22) Qualified Qwen 3.5 0.8B MTP-1 standalone at a 32K window: token-identical short outputs, successful 8K-through-32.7K recall, 88.67%-93.75% draft acceptance, and level-1 sleep/wake at a 0.65 rollout budget.
- [x] (2026-07-22) Completed two Verifiers GRPO updates with MTP enabled: 32 trajectories, non-zero gradients, checkpoints, adapter export and synchronization, and a post-update MTP rollout using a 0.55 rollout budget with 4096-token chunked prefill.
- [x] (2026-07-22) Added step-local aggregate MTP telemetry and qualified it across two complete GRPO steps: 79.03% acceptance before the first update and 81.67% after LoRA synchronization. Per-request attribution remains unavailable and is not required for the synchronous backend gate.
- [x] (2026-07-24) Completed shared checkpoint-policy conformance: veRL now maps `checkpoint_limit` to actor and critic recovery retention, applies explicit `resume_from`, disables implicit auto-resume for fresh runs, and rejects backend overrides that replace the selected checkpoint policy.
- [x] (2026-07-29) Published self-contained veRL fork commit `8aa0b356d462568a92dedab642bba54aae37475d`, moved bounded dynamic group replacement into core V1, removed the runtime `verl-recipe` checkout, and smoked candidate kind image `sha256:dca0d724352b47630dc041368d110643faa2318d50a2a1e5e1b6eb9705402b99`.
- [x] (2026-07-29) The first metadata follow-up `61d4cc18de5c70472f6a912985f8676bdcf150f9` proved insufficient on the locked 4090 run because V1 stores custom agent-loop values inside `extra_fields`, not as top-level TransferQueue fields.
- [x] (2026-07-29) Published corrected veRL follow-up `0c1bea25266e346526a314c0206e96c8010911c3`, extracting SAMPO turn spans, anchor-state keys, and step rewards from the queued per-row `extra_fields` object before advantage computation.
- [x] (2026-07-29) Run `verl-sampo-4090-extra-fields-20260729` retained all eight Verifiers traces and reached SAMPO advantage computation, then proved that the materialized trajectory `uid` was not a reliable prompt-group identity.
- [x] (2026-07-29) Run `verl-sampo-4090-canonical-groups-20260729` retained eight traces but rejected key-derived groups of `[2, 2, 4]`; retained trace evidence showed that the stable boundary is the dataset `example_id`, with four trajectories for each of two examples.
- [x] (2026-07-29) Published veRL follow-up `a35d13b0a4aae518ebda07f8009334098bb510f1`, consuming an explicit `sampo_prompt_group_id` from agent-loop metadata while retaining replay-key parsing only as a compatibility fallback; 31 focused V1 and SAMPO CPU tests pass in the locked runtime.
- [x] (2026-07-29) Run `verl-sampo-96gb-explicit-groups-20260729` proved dstack placement on the restored RTX PRO 6000 worker and passed prompt grouping, then exposed that TransferQueue padding invalidated absolute environment turn offsets.
- [x] (2026-07-29) Published veRL follow-up `6dda0d98d5be64a39567a0e0f1cfd8ece506ae3f`, reconstructing optimizer turn spans from stable per-turn policy-token lengths and the materialized response mask; 32 focused V1 and SAMPO CPU tests pass in the locked runtime.
- [x] (2026-07-30) Run `verl-sampo-96gb-mask-aligned-turns-20260729` passed prompt grouping and reached mask-aligned span reconstruction, then exposed a truncated trajectory with zero trainable policy tokens.
- [x] (2026-07-30) Published veRL follow-up `722595be16f9ac839d8f9c34efdb6bbff788b3ad`, making SAMPO evict and refill the entire failed prompt group even when sibling trajectories were already materialized; 77 focused V1 tests and 60 framework/runtime tests pass.
- [x] (2026-07-29) Added two-step DAPO and SAMPO qualification packages for standard, MTP-1, TurboQuant K8V4, and combined MTP-1 plus K8V4 rollout modes; all catalog compositions validate.
- [x] (2026-07-30) Deferred the TurboQuant-only and combined MTP-plus-TurboQuant RL matrix from this release. The package definitions remain available as explicitly unqualified follow-up work; the release matrix continues with baseline, MTP-only, 32K normal-KV GRPO, and distillation.
- [x] (2026-07-29) Corrected the veRL actual-job control dependency closure from stale Python 3.12 to the image contract's Python 3.13.12; 27 focused execution/image tests pass.
- [x] (2026-07-29) Enabled persistent dstack pre-start capacity waiting and added `posttrain run queue` plus requested/assigned hostname fields. A live pinned SAMPO task remained provider-pending while DAPO occupied `carbonteq-ai-workstation.lan`, then started automatically when the worker became available.
- [x] (2026-07-29) Requalified capacity-based parallel placement through immutable actual-job images. DAPO MTP occupied the RTX PRO worker while a hostname-free 8 GiB CUDA smoke job was assigned to the idle RTX 4090; both workers reported `1/1` busy and both runs succeeded.
- [x] (2026-07-30) Added revisioned 24-GB-minimum veRL training, rollout, MTP, and 2B teacher-score selections without hostname constraints. While exact-lock DAPO ran on the 96 GB RTX PRO worker, dstack assigned `verl-sampo-mtp-fleet-20260730` to `pop-os.lan`, proving that eligible 0.8B and colocated 2B-teacher jobs can use either worker without erasing the older 96-GB-qualified selections.
- [x] (2026-07-30) The first hostname-free MTP revision proved placement on the 24 GB worker but exhausted memory when vLLM retained its percentage-sized cache before FSDP2 initialization. Revision 3 bounds the 640-token MTP job to the already-qualified 192 MiB fixed KV-cache reservation; the failed revision remains immutable evidence rather than being edited in place.
- [x] (2026-07-30) A 24 GB baseline DAPO rerun exposed that the revision-2 training binding's legacy rollout Hydra values overrode the selected inference binding's 64 MiB cache. Training-binding revision 3 removes rollout capacity from backend escape hatches so baseline and MTP inference selections remain authoritative.
- [x] (2026-07-30) Completed the corrected baseline DAPO and SAMPO pair. DAPO run `verl-dapo-baseline-fixed-20260729` completed both optimizer steps. SAMPO run `verl-sampo-96gb-vllm-fork-20260730` completed two optimizer steps on `carbonteq-ai-workstation.lan`, retained 28 Verifiers traces plus model, recovery, retention, and summary artifacts, emitted episode/turn/anchor/sparse-projection metrics, and reconciled with provider exit 0 and no missing artifact roles.
- [x] (2026-07-30) Published veRL distillation revision `c3f49b9117b882fa888e25e4a771461e13167848`, covering dense and jagged teacher-logprob response alignment plus fully masked synthetic padding microbatches, and published runtime kind `sha256:7be370ba3ee3525d784daa68e6d2b596c6ebcfabfeaa3f1c8c8f5268a8f3efc9` with immutable Verifiers harness prerequisites.
- [x] (2026-07-30) Run `verl-distill-shared-pool-retentionfix-20260730` completed two optimizer steps on `carbonteq-ai-workstation.lan`, merged and retained the Qwen 3.5 0.8B LoRA adapter, retained 16 Verifiers traces plus summary and retention manifest, and reconciled provider exit 0 with no missing artifact roles.
- [ ] Project veRL distillation `scored_tokens` and `teacher_failures` through the provider-neutral metric contract. The successful run remains operational qualification evidence, but Observatory reports required telemetry `2/4` and is not research-ready.

## Surprises & Discoveries

- Observation: Current upstream veRL already has extensive Qwen 3.5 support, including dense and MoE GRPO recipes, model-specific forward patches, vLLM synchronization handling, and a Qwen 3.5 on-policy-distillation recipe.
  Evidence: upstream commit `a35908ca3c9632859c58d6a2855d858918ae21dc` contains `examples/grpo_trainer/run_qwen3_5_2b_openr1_fsdp.sh`, `examples/on_policy_distillation_trainer/run_qwen3_5_4b_fsdp.sh`, and `verl/models/transformers/qwen3_5.py`.

- Observation: veRL exposes the shared recovery behaviors through separate native fields and rotates actor/critic state rather than every file in an old `global_step_*` directory.
  Evidence: `trainer.max_actor_ckpt_to_keep` and `trainer.max_critic_ckpt_to_keep` are passed to the checkpoint managers, while `trainer.resume_mode=resume_path` and `trainer.resume_from_path` load an explicit `global_step_*` directory. Old step metadata may remain after state rotation, but it is not a resumable model checkpoint.

- Observation: The repository's TRL runtime and upstream veRL currently resolve incompatible Transformers and vLLM versions.
  Evidence: `packages/train/pyproject.toml` selects Transformers 5.14 and vLLM 0.25.1, while the inspected veRL recipes select an older Transformers revision and vLLM 0.18.0. The adapter therefore cannot safely import veRL into the main `uv` environment.

- Observation: The currently qualified isolated veRL environment cannot use TurboQuant.
  Evidence: it contains vLLM 0.18.0, which does not expose `TQFullAttentionSpec` or `get_kv_quant_mode`; the serving environment's vLLM 0.25.1 exposes `turboquant_k8v4` but requires the existing guarded quantization-marker compatibility patch.

- Observation: core V1 dynamic filtering classifies finished groups before it
  materializes the training batch and reads its metric only from
  `extra_fields.reward_extra_info`.
  Evidence: live DAPO run `19996710-84ff-44f2-abf7-c15c335f891b` completed
  optimizer step 1 and then failed with `Finished groups are missing DAPO
  metric 'seq_reward'`. The custom Verifiers loop had emitted `reward_score`
  and top-level `algorithm_reward` but not the native nested metric.

- Observation: dstack 0.20.29 represents persistent capacity queuing as a
  bounded `no-capacity` retry rather than an unbounded queue flag.
  Evidence: omitting retry duration defaults to 3600 seconds. With
  `on_events: [no-capacity]` and `duration: 86400`, live SAMPO run
  `verl-sampo-baseline-queue-20260729` remained `pending/retrying` with the
  requested workstation unassigned while DAPO held it, then received
  `carbonteq-ai-workstation.lan` automatically.

- Observation: vLLM 0.25.1 K8V4 increases Qwen 3.5 0.8B cache capacity but fails the matched recall gate.
  Evidence: on the RTX 3070 Ti, K8V4 exposed 776722 cache tokens versus 291328 for normal FP16 KV and completed level-1 sleep plus separate weight/cache wake. The normal cache recalled a beginning-of-context code at 8192, 16384, 24576, and 32700 input tokens; K8V4 failed all four. The Ampere TurboQuant store kernel also requires an FP16 rollout copy because BF16-to-FP8 key conversion fails in Triton 3.6.

- Observation: Qwen 3.5 0.8B carries one native MTP head and vLLM 0.25.1 can use it at the required 32K capacity.
  Evidence: the immutable checkpoint contains `text_config.mtp_num_hidden_layers=1` and `mtp.*` tensors. The local MTP-1 probe resolved `Qwen3_5MTP`, shared target embeddings and the LM head with the drafter, recalled correctly through 32700 input tokens, and measured 88.67%-93.75% acceptance. At a matched 0.65 budget it used about 501 MiB more GPU memory than normal rollout.

- Observation: veRL's vLLM acceptance bridge and vLLM 0.25.1 expose different metric grains.
  Evidence: veRL checks `RequestOutput.metrics.request_spec_decode_stats`, which is absent in the tested runtime; vLLM's aggregate metrics snapshot exposes draft, accepted-token, and per-position counters. The standalone probe records those counters, but the real GRPO gate still needs normalized run-level acceptance evidence.

- Observation: The existing `EnvironmentRolloutBridge` is an in-process protocol, whereas veRL performs rollout and optimization in Ray workers.
  Evidence: `packages/train/src/posttrain/train/integrations/verifiers.py` owns a live Verifiers environment and accepts an injected `PolicyGenerator`; veRL's `AgentLoopBase` runs beside the rollout server and returns the exact token and reward fields required by training.

- Observation: Prompt-group batch size and generated-sequence batch size are distinct in veRL, but veRL validates the actor mini-batch in prompt-group units.
  Evidence: the real Qwen 3.5 0.8B AutomationBench launch rejected `train_batch_size=1` with `ppo_mini_batch_size=2`; the worker now maps both `data.train_batch_size` and the default actor mini-batch to `num_prompts_per_step`, while rollout `n` remains `num_generations`.

- Observation: A Hub repository id alone would discard the model selection's immutable revision.
  Evidence: the isolated worker now resolves every Hub model through `huggingface_hub.snapshot_download(repo_id=..., revision=...)` before constructing veRL overrides.

- Observation: Upstream veRL supports LoRA and several rollout quantization formats, but its FSDP Hugging Face loader does not pass a `BitsAndBytesConfig`; consequently it cannot train a bitsandbytes 4-bit base as QLoRA without a fork patch.
  Evidence: commit `a35908ca3c9632859c58d6a2855d858918ae21dc` calls `from_pretrained` with dtype and model config only, then casts the loaded module with `.to(torch_dtype)` in `verl/workers/engine/fsdp/transformer_impl.py`.

- Observation: veRL's LoRA weight synchronization sends only adapter tensors after the rollout worker has loaded its immutable base model.
  Evidence: the worker marks base synchronization complete for normal rollout load formats and `get_per_tensor_param(..., base_sync_done=True)` selects LoRA parameters. This permits an NF4 actor base and a BF16 rollout base, but the resulting actor-versus-rollout log-probability mismatch must be measured and corrected by veRL's importance-sampling path.

- Observation: veRL's GRPO trainer retains `ppo_mini_batch_size` and `ppo_micro_batch_size_per_gpu` names even though GRPO supplies the advantage estimator.
  Evidence: the real launch validates the former in prompt-group units. The adapter now derives these private fields from `num_prompts_per_step`; neither name is part of the public GRPO request.

- Observation: Qwen 3.5 `all-linear` LoRA is not portable between the Hugging Face actor and vLLM 0.18 rollout representation.
  Evidence: vLLM ignores vision-tower adapters and rejected the gated Qwen 3.5 QKV adapter with a 2048-versus-6144 packed dimension mismatch. The text-only smoke recipe therefore targets language-model `o_proj` and `down_proj` modules only.

- Observation: The Qwen 3.5 0.8B AutomationBench integration pass completed end to end on an 8 GiB GPU, but both sampled trajectories earned zero reward and therefore do not qualify the trainer.
  Evidence: run `artifacts/automationbench-verl-qwen35-08b-smoke-15` returned a public `TrainingResult`, saved `global_step_1`, exported a 2,073,040-byte LoRA adapter, and preserved two native Verifiers traces. The model asked for information or credentials already represented by the environment instead of calling tools, so group-relative advantages, policy loss, and gradient norm were all zero. The user explicitly rejected this as insufficient qualification evidence.

- Observation: Actor and rollout probabilities remained closely aligned for the qualified projection-only LoRA recipe.
  Evidence: the completed step reported Pearson correlation `0.999648869`, mean probability difference `0.005822067`, corrected KL `0.004683912`, actor peak allocated memory `2.7805 GiB`, and 45.50 seconds for the training step.

- Observation: The native Verifiers v1 AutomationBench port preserved upstream tasks, world isolation, allowed-service computation, dense scoring, and strict metrics, but changed the default model-facing tools from Zapier meta-tools to the optional REST API toolset.
  Evidence: Zapier AutomationBench `a321764ace3cfbe42289e6a13abef2f0f4f56fad` constructs `AutomationBenchEnv(toolset="zapier")` by default and advertises `search_tools` plus `execute_tool`; the port advertised only `api_search`, `api_fetch`, and `base64_encode`. Qwen 3.5 0.8B consequently treated the generic API tools as unrelated to task requests naming Salesforce, Gmail, Slack, and Asana. The corrected port has direct tests for the default meta-tool discovery-and-mutation path and the optional API path.

- Observation: Increasing turns and context without increasing the rollout wall-clock budget caused queued trajectories to time out before their first sampled token.
  Evidence: with 16 concurrent Verifiers rollouts, `max_num_seqs=1`, and the inherited 300-second timeout, later traces ended in `harness_timeout` with zero graph branches. The bridge now defaults the training rollout timeout to 1800 seconds and preserves failed native traces before projection.

- Observation: veRL advertised chunked entropy for FSDP, but the dense `use_remove_padding=False` branch bypassed it.
  Evidence: qualification 16 still called `entropy_from_logits(logits)` and requested a 2.30 GiB softmax allocation. The fork now flattens dense token rows, applies configured chunking, reshapes the result, and passes four no-padding GPU tests.

- Observation: vLLM and actor memory are phase-shared in the colocated single-GPU lifecycle; `gpu_memory_utilization` is a rollout-phase budget, not a permanent partition during backward.
  Evidence: after rollout sleep, the vLLM worker retained about 268 MiB while the actor update owned the GPU. A 0.70 rollout attempt failed only because an unrelated QLoRA SFT process already held about 4 GiB, not because veRL kept 70 percent resident.

- Observation: after chunked entropy, the remaining actor peak is the dense full-vocabulary logprob/backward path.
  Evidence: qualifications 19 and 20 completed 16 reward-bearing trajectories and old-logprob computation, then failed in actor update with only 43-45 MiB free. The Qwen 3.5 fork already contains a fused PPO head that chunks vocabulary projection and logprob/entropy work; the next clean-GPU qualification enables that path.

- Observation: the maintained distillation composition exposed three boundaries
  that the earlier trainer-only pass did not exercise: logical jagged-tensor
  rows versus larger backing storage, all-masked synthetic padding
  microbatches, and terminal retention after checkpoint-free export.
  Evidence: the first two candidate runs failed at teacher-logprob alignment
  and an empty diagnostic reduction. The third completed both optimizer steps
  but failed after model merge because the checkpoint-free profile had already
  removed its checkpoint root. Focused regressions now cover all three cases,
  and `verl-distill-shared-pool-retentionfix-20260730` exits successfully.

- Observation: successful execution and artifact reconciliation are necessary
  but not sufficient for research-ready qualification.
  Evidence: `verl-distill-shared-pool-retentionfix-20260730` retained 16 traces,
  the adapter, summary, and retention manifest with provider exit 0, while
  Observatory still reports missing required `train/distill/scored_tokens` and
  `train/distill/teacher_failures`.

## Decision Log

- Decision: Expose `verl` as a general backend product name, but qualify and preflight only Qwen 3.5 in this slice.
  Rationale: Backend identity should not encode a temporary model support matrix. Explicit qualification prevents unsupported models from failing deep inside Ray or vLLM.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Preserve `GRPORequest`, `OnPolicyDistillationRequest`, and the Verifiers rollout contract instead of adding veRL-specific public operations.
  Rationale: The canonical baseline defines stable job meaning and explicitly requires a future veRL adapter to preserve it.
  Date/Author: 2026-07-22 / Codex.

- Decision: Implement checkpoint cleanup as backend conformance to `TrainingLoop`, not as a veRL-specific knob or a destructive cross-run action.
  Rationale: `checkpoint_steps`, `checkpoint_limit`, and `resume_from` already express within-run recovery policy. A fresh run must explicitly disable veRL auto-resume; deleting diagnostic state from other runs belongs to a separate workspace-retention lifecycle.
  Date/Author: 2026-07-24 / Codex.

- Decision: Launch veRL through an explicitly configured isolated Python interpreter and immutable source revision.
  Rationale: This avoids dependency contamination and makes the runtime reproducible. The host process owns logical observation and artifact ingestion; the isolated process owns veRL, Ray, Transformers, vLLM, and the GPU workers.
  Date/Author: 2026-07-22 / Codex.

- Decision: Keep Verifiers environment ownership in this repository and adapt it through a custom veRL agent loop.
  Rationale: Moving rewards or environment semantics into a veRL fork would violate the capability boundary and make native Verifiers traces cease to be replay authority.
  Date/Author: 2026-07-22 / Codex.

- Decision: Do not add a base-catalog veRL binding until the isolated fork checkout and environment lock exist.
  Rationale: the safe launcher requires absolute executable and checkout paths plus a full commit. Publishing invented machine paths would create a selection that resolves but cannot execute. The generic `TrainingBinding` already exposes `verl@...`; the eventual deployment overlay will supply concrete runtime paths.
  Date/Author: 2026-07-22 / Codex.

- Decision: Interpret quantized training as NF4 QLoRA, not updates to packed AWQ or GPTQ weights.
  Rationale: QLoRA freezes the quantized base and optimizes ordinary LoRA parameters, is supported by bitsandbytes on the available Ampere GPU, and fits the existing public `QLoRAUpdate` selection. AWQ and GPTQ are primarily deployment formats and are not a suitable optimizer-state representation for this training path.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Keep the rollout base in BF16 and synchronize adapter-only updates from the NF4 actor for the first single-GPU qualification.
  Rationale: current veRL already has a robust LoRA-only synchronization path, whereas teaching vLLM to share the training worker's bitsandbytes modules would be a substantially larger runtime change. Acceptance requires recording the rollout/actor log-probability divergence and importance-sampling metrics.
  Date/Author: 2026-07-22 / Codex.

- Decision: Do not expose QLoRA through the framework backend in this slice; use Qwen 3.5 0.8B with ordinary BF16 LoRA for the first AutomationBench GRPO run.
  Rationale: the smaller official checkpoint avoids maintaining a quantized training fork for the initial use-case proof and removes actor-versus-rollout precision mismatch from the experiment. The local QLoRA branch remains research evidence only and is not selected by a runtime binding.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Keep shared execution concepts in `TrainingBinding.runtime` and backend-native settings in `TrainingBinding.backend_options`.
  Rationale: `nodes`, `devices_per_node`, batching, and offload intent should mean the same thing for TRL and veRL, while isolated interpreter paths, source revisions, attention implementations, and Hydra overrides are veRL-owned. Native overrides may tune the backend but may not replace selected models, data, agent configuration, or artifact paths.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Qualify GRPO against AutomationBench's upstream default Zapier meta-tool interface, not a prompt-modified API-mode surrogate.
  Rationale: The environment port must preserve the benchmark's agent-facing semantics. API mode remains useful as an explicit alternate composition, but silently making it the v1 default changed the task difficulty and caused the small policy to reject relevant simulated services as unavailable.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Keep verifier execution asynchronous in host RAM and tune vLLM `max_num_seqs` independently as GPU scheduling concurrency.
  Rationale: rollout count, environment concurrency, and vLLM active-sequence concurrency are separate controls. Tool waits can overlap on CPU while vLLM schedules ready model turns; the local machine has ample host RAM but only 8 GiB VRAM.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Treat context length, turn count, and rollout timeout as environment/runtime selections, while retaining veRL-only FSDP and fused-kernel switches in backend options.
  Rationale: limits should mean the same thing across training backends; nested veRL implementation details such as `fsdp_config.use_torch_compile` and the Qwen 3.5 fused PPO head remain explicit backend overrides.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Every maintained backend fork has two linked revision-aware records: a consumer tooling page in this workspace and a root `CARBONTEQ_FORK.md` in the fork.
  Rationale: Consumer documentation must explain selection and operation, while the fork itself must preserve its upstream base, generic delta, regression tests, rebase obligations, and release pin. Neither repository should depend on chat history to reconstruct the change.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Require a 32768-token rollout capacity for the veRL backend, but use normal FP16 KV cache for the first constrained-device qualification.
  Rationale: The 8192-token run proves the full GRPO lifecycle but does not meet the use case's context requirement. The normal vLLM 0.25.1 cache already measured capacity for 8.89 concurrent 32K sequences at the selected budget and passed the matched recall matrix, while K8V4 failed it. The rollout copy may use FP16 without changing the BF16 actor or LoRA identity; actor-side long-sequence memory still requires an independent full backward gate.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Add MTP first as an inference-binding rollout optimization, not an MTP training objective.
  Rationale: Native Qwen 3.5 MTP is already represented by the selected model and vLLM speculative configuration, while MTP-loss training changes the optimizer objective and currently requires veRL's Megatron path. Keeping `enable_train=false` preserves the existing GRPO job meaning and allows MTP and non-MTP rollouts to be compared as binding variants.
  Date/Author: 2026-07-22 / Codex and user.

- Decision: Keep required distillation telemetry as a release gate after the
  two-step GPU execution succeeds.
  Rationale: adapter and trace retention proves the training lifecycle, but it
  does not prove the amount of effective teacher supervision or whether
  teacher-scoring failures were excluded. Weakening Observatory completeness
  would hide that distinction.
  Date/Author: 2026-07-30 / Codex.

## Outcomes & Retrospective

The backend integration, baseline GRPO path, and distillation execution path
are complete. Run `artifacts/automationbench-verl-qwen35-08b-qualification-23`
qualified the two-step 8K GRPO lifecycle. Run
`verl-distill-shared-pool-retentionfix-20260730` qualified two-step colocated
student/teacher execution on the 96 GB worker and retained 16 native traces,
the trained adapter, summary, and retention manifest. The 32K normal-KV GRPO
gate and veRL distillation's required scored-token and teacher-failure
telemetry remain open, so the overall backend release gate is not yet complete.

## Context and Orientation

`packages/train/src/posttrain/train/api.py` implements the public training operations and currently defaults directly to private TRL runners. `packages/train/src/posttrain/train/requests.py` validates backend-neutral requests. `packages/train/src/posttrain/train/integrations/verifiers.py` executes native Verifiers episodes against an injected policy and projects them into `EnvironmentRollout` values defined in `packages/train/src/posttrain/train/online_rl.py`.

The new backend belongs under `packages/train/src/posttrain/train/backends/verl/`. Backend dispatch must inspect the product portion of `TrainingBinding.backend`, such as `verl` in `verl@a35908c`, without exposing veRL types through `posttrain.train`. A veRL launch specification is a JSON-safe record containing model and tokenizer locations, algorithm settings, rollout settings, execution topology, output paths, and paths to portable environment inputs. The launch specification is executed by an isolated interpreter selected in `TrainingBinding.runtime`.

The custom agent loop is the boundary between veRL and Verifiers. A veRL rollout server generates the model turns; the existing Verifiers environment owns episode sequencing and rewards; the agent loop returns veRL's `prompt_ids`, `response_ids`, `response_mask`, `response_logprobs`, and `reward_score`. Distillation uses veRL's native teacher server to score the same on-policy student token ids. Shared execution values live in `TrainingBinding.runtime`; veRL-native launch and Hydra values live in `TrainingBinding.backend_options`.

## Plan of Work

First, add a small backend resolver used only when a caller does not inject a runner. It must keep TRL behavior for `trl@...`, route GRPO and distillation for `verl@...`, and reject veRL for SFT and DPO because those techniques are outside this slice.

Second, implement the veRL adapter as an isolated launcher. Translate the backend-neutral model, update, loop, inference, target, and algorithm selections into a deterministic manifest and Hydra overrides. Require a versioned backend, an immutable `backend_source_revision`, an isolated executable, and a Qwen 3.5 policy. Distillation must additionally require a Qwen 3.5 teacher and retain the existing tokenizer-fingerprint equality check. Do not accept arbitrary shell fragments; construct an argument vector and pass it to `subprocess.run` without a shell.

Third, make the native Verifiers bridge portable to the isolated Ray runtime and implement the custom veRL agent loop. Serialization is trusted internal input and must contain only the bridge reconstruction fields, never the live environment or synchronization lock. Worker-side trace writes must remain safe across Ray processes. The adapter must fail before training if the selected bridge cannot export the portable contract.

Fourth, add catalog examples using the general `verl@<immutable-revision>` backend name and documentation beside those selections stating that the current qualification matrix contains only Qwen 3.5 for GRPO and distillation. Do not rename the public operations or introduce Qwen-specific backend types.

Fifth, materialize the immutable Qwen 3.5 0.8B checkpoint and create a deployment-specific veRL binding using BF16 LoRA rank 8. Keep one prompt, two generations, one optimizer step, conservative prompt/response limits, and single-GPU offload settings for the first AutomationBench run.

Finally, validate translation and dispatch with fakes, validate exact-token projection with a fake veRL server, and run the repository checks. Run a one-prompt, two-generation, one-step AutomationBench GRPO smoke test with conservative sequence limits and offload. A GPU release run must use the pinned fork and isolated lock, preserve a native Verifiers trace artifact, and produce a model artifact and training summary for both operations.

## Concrete Steps

From `/home/hammad/projects/rl`, run focused tests while implementing:

    uv run pytest packages/train/tests/test_verl_backend.py packages/train/tests/test_api.py packages/train/tests/test_verifiers_grpo_bridge.py

Then run the package and boundary validation:

    uv run ruff check packages/train apps/lab
    uv run pyright
    uv run lint-imports
    uv run pytest packages/train/tests apps/lab/tests
    git diff --check

From `/home/hammad/projects/rl`, validate the ordinary LoRA manifest translation and then launch the Qwen 3.5 0.8B AutomationBench smoke run. The exact command and dependency-lock digest must identify the upstream veRL commit, Qwen 3.5 model revision, Verifiers revision, device topology, and work-package or smoke entrypoint. A missing credential may skip a Hub download test, but the cached immutable checkpoint must be complete before execution.

## Validation and Acceptance

Backend dispatch is accepted when a `trl@...` request still calls the TRL adapter, a `verl@...` request calls the veRL adapter, and an unknown product fails before creating a trainer directory. Explicitly injected runners must continue to work in unit tests.

Qualification is accepted when Qwen 3.5 GRPO and Qwen 3.5-to-Qwen 3.5 distillation produce deterministic launch manifests, while another model family produces a concise error stating that the veRL backend currently qualifies only Qwen 3.5. The backend name and types must remain generic.

Environment integration is accepted when a fake veRL rollout server can execute a Verifiers episode and the returned agent-loop output preserves exact prompt and response token ids, one response-mask entry per response token, one log probability per sampled token, the Verifiers reward, truncation status, and trace identity.

The GPU release gate is accepted when each operation performs at least one optimizer step, emits normalized training metrics, writes a recoverable summary, produces model weights or an adapter as selected, and preserves native Verifiers traces. Distillation additionally must prove that teacher scores align with the exact student-generated token ids and response mask.

## Idempotence and Recovery

Manifest creation overwrites only files inside the new run's trainer directory, which `api.py` already requires not to exist. The isolated command receives explicit output and trace paths. A failed process leaves its manifest, stdout/stderr log, and any veRL checkpoints for diagnosis; rerunning requires a new run workspace or an explicit compatible `resume_from` selection. No command deletes checkpoints, model caches, or Ray state globally.

## Artifacts and Notes

The qualified veRL source remains upstream commit `a35908ca3c9632859c58d6a2855d858918ae21dc`. A separate local research branch, `codex/qwen35-qlora` at `05f83242359cf97dc93483caae45602d10862915`, proved that NF4 adapter backward works on this GPU, but it is parked and must not be used by the 0.8B runtime binding.

## Interfaces and Dependencies

The public `grpo(context, request)` and `distill(context, request)` signatures remain unchanged. Their optional `runner` keyword becomes nullable so omission selects a backend from `request.training.backend`, while explicit callables retain the existing testing and host-extension seam.

Under `posttrain.train.backends.verl`, define generic `run_grpo` and `run_distillation` callables matching `GRPOBackend` and `DistillationBackend`. Define a deterministic launch-plan value and pure translation functions so unit tests do not import veRL. The launch subprocess must emit a stable result JSON that can be converted into the existing private `BackendTrainingResult` and `TrainingSummary` values.

The main package must not add veRL, Ray, Transformers, or vLLM to its base dependency set. The isolated runtime is selected by `TrainingBinding.backend_options` and must be locked separately to the veRL fork commit and compatible Qwen 3.5 stack. `packages/train` must remain independent of `packages/eval`, `packages/serve`, and `apps/lab`.

Revision note (2026-07-22): Created the initial implementation plan after the user chose a general veRL backend name with a Qwen 3.5-only qualification boundary.

Revision note (2026-07-22): Updated the living plan after completing backend dispatch, isolated launch and worker translation, portable Verifiers agent-loop integration, qualification notes, and CPU validation. Kept the GPU release gate and deployment-specific catalog binding open pending a real pinned fork environment.

Revision note (2026-07-22): Expanded the scope at the user's request to NF4 QLoRA for Qwen 3.5 2B and a one-step AutomationBench GRPO smoke test. Recorded the actor/rollout precision split, multi-repository ownership, commit order, and staged GPU validation.

Revision note (2026-07-22): Pivoted the GPU smoke target to ordinary BF16 LoRA on Qwen 3.5 0.8B at the user's request. Parked the completed QLoRA prototype without exposing it from the framework backend.

Revision note (2026-07-22): Split normalized execution variables from backend-native options and made veRL's PPO-named optimizer fields a private GRPO translation detail.

Revision note (2026-07-22): Recorded the successful Qwen 3.5 0.8B AutomationBench GRPO run, its portable LoRA target restriction, runtime evidence, zero-reward behavior, checkpoint, adapter, traces, and normalized result.

Revision note (2026-07-22): Reclassified the zero-gradient run as integration-only at the user's direction and strengthened qualification to two full rollout batches with a non-zero backward update and post-update rollout.

Revision note (2026-07-22): Audited the Verifiers v1 AutomationBench port against the pinned Zapier implementation, found that the port had selected upstream's optional API toolset instead of its default Zapier meta-tools, restored default tool-contract parity, and retained API mode as an explicit alternate.

Revision note (2026-07-22): Completed the two-step AutomationBench GRPO qualification with the original 50-turn behavior, 8192-token capacity, asynchronous verifier execution, phase-shared 0.70 vLLM memory budget, four scheduled sequences, Qwen 3.5 fused PPO actor, non-zero gradients, adapter-only synchronization, and 32 preserved native traces. Distillation remains the open GPU release gate.

Revision note (2026-07-22): Consolidated the measured veRL optimization and diagnostic guidance into the tooling page and established the linked workspace/fork documentation convention, with an unpublished candidate ledger added to the local veRL checkout.

Revision note (2026-07-22): Corrected 32768 tokens as the required rollout capacity, selected TurboQuant K8V4 for the constrained-device target, added adapter translation and fork compatibility coverage, and kept the completed 8K run classified as a lifecycle checkpoint rather than the final context qualification.

Revision note (2026-07-22): Re-locked the isolated veRL stack to vLLM 0.25.1, measured K8V4's capacity and sleep/wake lifecycle, and rejected K8V4 for the Qwen 3.5 release path after it failed the matched 8K-through-32.7K recall matrix. The remaining 32K GRPO gate now uses normal FP16 KV cache.

Revision note (2026-07-22): Added rollout-only MTP translation and standalone Qwen 3.5 0.8B MTP-1 evidence through 32.7K. Kept full MTP-loss training outside the FSDP2 LoRA slice and left the real GRPO backward/synchronization plus normalized acceptance telemetry as release gates.

Revision note (2026-07-22): Completed the two-step AutomationBench MTP GRPO lifecycle after adding the pinned Verifiers dependencies to the upgraded runtime, reducing the colocated rollout budget to 0.55 with 4096-token chunked prefill, and fixing Qwen 3.5 attention dispatch when FSDP2 drops the decoder's plain `layer_type` attribute. Acceptance telemetry remains open.

Revision note (2026-07-22): Closed the synchronous MTP observability gate by enabling vLLM statistics, snapshotting aggregate speculative counters before rollout sleep, converting lifetime counters into per-step deltas, and qualifying non-zero acceptance before and after adapter synchronization in run `-06`.

Revision note (2026-07-24): Wired the shared checkpoint limit and explicit resume selection into veRL Hydra configuration, made fresh launches opt out of veRL auto-resume, and protected checkpoint policy from backend-native override replacement. This is a backend-conformance correction and does not amend the frozen product baseline.
