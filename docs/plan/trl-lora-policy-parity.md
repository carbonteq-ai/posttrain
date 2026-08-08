# Make colocated LoRA rollouts match the training actor

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

A GRPO or DAPO update is only meaningful when the policy that generated a rollout is the same policy whose log probabilities and gradients are computed by the training actor. The retained Qwen3.5 SFT diagnostic proved that the current native-LoRA synchronization path can export a valid PEFT adapter under the Transformers module namespace while colocated vLLM expects the same weights under its composite `language_model` namespace. Training then continues with importance weights pinned near their lower bound instead of failing before the first optimizer update.

After this work, TRL can apply its existing vLLM weight-name prefix to a native LoRA export, and GRPO fails before optimizer step one when the sampler-versus-actor mean token log-probability difference exceeds an explicit safety limit. The consuming framework records the exact fork revision and selects the Qwen3.5 namespace in its inference binding. LoRA and QLoRA runs publish an adapter-only model artifact, retain bounded recovery state, and can resume that state in a fresh job without publishing or restoring a full base-model snapshot. A one-step parity canary and a stop-and-resume canary must pass before the matched two-update SFT experiment is repeated.

This corrects a backend synchronization defect and strengthens validation without changing the frozen product meaning of `train.grpo`, DAPO rewards, batching, or the selected optimization objective. Recovery required one narrow canonical API amendment: `docs/post-training/05-apis.md` now defines `posttrain job run --resume-from-run RUN_ID` for training jobs, with a new run identity and an immutable `training-checkpoint` input.

## Progress

- [x] (2026-08-08 02:24Z) Completed the matched SFT-start diagnostic and retained 256 traces, two optimizer records, the final adapter, and the global-step-2 recovery checkpoint.
- [x] (2026-08-08 02:35Z) Reconciled native PEFT keys, the known serving-compatible Qwen3.5 remap, vLLM's LoRA parser, and TRL's native-LoRA export path.
- [x] (2026-08-08 02:39Z) Confirmed the frozen framework baseline already assigns rollout-engine configuration to the inference binding and backend synchronization to the private trainer adapter.
- [x] (2026-08-08 03:28Z) Implemented and focused-tested LoRA-safe `weight_name_prefix` handling in `/home/hammad/projects/trl` (7 focused tests passed).
- [x] (2026-08-08 03:28Z) Implemented and focused-tested a first-training-rollout actor/sampler parity gate in `/home/hammad/projects/trl` (9 passed, 5 skipped in the focused selection).
- [x] (2026-08-08 03:05Z) Audited the retained SFT diagnostic manifests: the final model and recovery checkpoint are adapter-only; the checkpoint additionally contains optimizer, scheduler, RNG, and trainer state.
- [x] (2026-08-08 03:28Z) Added executable adapter-only validation, standard job-definition recovery plumbing, a train-only `--resume-from-run` selection, and failure/cancellation publication of the latest complete checkpoint. The focused framework selections pass 124 tests.
- [x] (2026-08-08 03:31Z) Updated the TRL fork ledger and framework consumer page with the unpublished candidate, defect, recovery contract, and qualification gate.
- [x] (2026-08-08 03:17Z) Added the Qwen3.5 `language_model.` LoRA prefix to affected framework and Ambient Agent native-LoRA inference bindings; catalog tests pass.
- [x] (2026-08-08 04:23Z) Passed a local-source SFT parity canary on the RTX 4090: mean sampler/actor delta `0.006096`, nonzero gradient norm `0.3659`, reward standard deviation `0.2421`, and an adapter-only step-one checkpoint.
- [x] (2026-08-08 04:28Z) Passed a fresh-process recovery canary: materialized the immutable step-one checkpoint as an input, restored trainer state, advanced only to global step two, passed parity at `0.006693`, and published a new adapter-only checkpoint.
- [x] (2026-08-08 04:56Z) Completed the matched 32-prompt × 4-generation, two-update SFT-start DAPO diagnostic: reward mean `0.6476 -> 0.6680`, raw trace reward mean `0.6554 -> 0.7010`, triple-F1 mean `0.5926 -> 0.6347`, zero-variance groups `12.5% -> 6.25%`, and actor/sampler mean log-probability deltas remained below the parity gate. The run finished successfully as `d2e5a33d-0d5c-45ea-8a7f-be23a3ee1493`.
- [x] (2026-08-08 06:53Z) Implemented explicit scalar-DAPO advantage and truncation telemetry, disabled group reward scaling for the diagnostic profile, and completed a corrected two-update 32-prompt × 4-generation run from the SFT adapter. Trace reward mean improved `0.6411 -> 0.6910`, triple-F1 `0.5608 -> 0.5933`, and truncation masking excluded `2/128` rows at step two; run `221ca5df-d555-447c-858b-e1dd780a7eab` completed successfully.
- [x] (2026-08-08 07:24Z) Audited the generic truncation path after the diagnostic: truncation is now always observable, but rewards are excluded from the group baseline only when `mask_truncated_completions=true`; the existing scale-reward and truncation regression selection passes `7` tests.
- [ ] Compare corrected and prior DAPO under repeated seeds or a held-out set before claiming the algorithm is better; then decide whether the production objective remains scalar DAPO or moves to decoupled multi-signal normalization.
- [ ] After explicit publication authorization, commit and push the TRL fork, update the framework's immutable dependency pin and lock, and reconcile Ambient Agent provenance.

## Surprises & Discoveries

- Observation: the native actor adapter and the manually remapped serving adapter contain the same 372 LoRA tensors under different namespaces.
  Evidence: the actor uses `base_model.model.model.layers...`; the serving copy uses `base_model.model.language_model.model.layers...`.

- Observation: vLLM validates only the final target-module suffix while loading the adapter, so an incorrect parent namespace can pass loading without attaching the tensors to the modules used for generation.
  Evidence: `vllm/lora/utils.py::parse_fine_tuned_lora_name` strips `base_model.model.` before applying the model mapper, while the Qwen3-VL mapper only rewrites `model.language_model.` and does not turn `model.layers.` into `language_model.model.layers.`.

- Observation: importance-sampling correction concealed the structural error instead of making the run valid.
  Evidence: the first SFT-start update reported mean/max absolute log-probability deltas of 0.253/19.47 and a mean importance ratio of 0.128 with a 0.1 lower bound. The matched base-policy update reported 0.014/0.57 and 0.863.

- Observation: the framework consumer page selects TRL commit `91b0ce707631d503fbed337b42444a9d3fac3acb`, but the Ambient Agent training binding still records predecessor commit `6e7739b8ec741d21ecd79c0c212694cd15ff20d8` and its older lock digest.
  Evidence: the live environment's `trl-1.8.0.dist-info/direct_url.json` records `91b0ce...`; the retained Trackio run snapshot records `6e7739...`.

- Observation: the framework already distinguishes a LoRA/QLoRA result (`model-adapter`) from recovery state (`training-checkpoint`), and forwards an explicitly materialized recovery path to `trainer.train(resume_from_checkpoint=...)`.
  Evidence: `packages/train/src/posttrain/train/api.py::_finish`, `packages/train/src/posttrain/train/backends/trl/common.py::finish_training`, and each TRL backend runner.

- Observation: type labels alone do not prove artifact contents, and current unit tests use synthetic directories rather than verifying PEFT/Transformers output manifests.
  Evidence: `test_lora_and_full_updates_materialize_distinct_model_forms_and_artifact_kinds` checks the declared kind but not file names, tensor keys, base-weight exclusion, or a fresh-job resume.

- Observation: the retained two-step SFT-start diagnostic did publish adapter-only weights, both as the final model and inside its recovery checkpoint.
  Evidence: Trackio manifest `model-adapter:v0` contains `adapter_model.safetensors` (33,688,200 bytes), `adapter_config.json`, tokenizer files, and no full-model shard. `training-checkpoint:v0` contains the same adapter plus `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, and `trainer_state.json`, and no full-model shard.

- Observation: backend-level resume exists but the standard job-definition layer does not currently expose or materialize a recovery selection into `GRPORequest.resume_from` (nor the SFT/DPO equivalents).
  Evidence: the request types and TRL runners accept `resume_from`, while `packages/jobs/src/posttrain/jobs/definitions.py` constructs training requests without a recovery seat and `_artifact_inputs` only recognizes model variants.

- Observation: Trackio commits a produced artifact synchronously when `RunContext.artifact` is called; it does not wait for a successful operation return.
  Evidence: `TrackioTrackedRun.artifact` uploads and appends the immutable `PublishedArtifact` before terminal `finish`, so an exception path can preserve the latest complete checkpoint while the temporary training workspace still exists.

- Observation: a cancellation-time checkpoint synthesized during an in-flight optimizer update would not have a trustworthy atomicity boundary.
  Evidence: Transformers exposes `get_last_checkpoint` for complete checkpoint directories. The recovery handler therefore publishes only that latest complete directory, bounding lost progress by `checkpoint_steps` and producing no recovery artifact before the first checkpoint.

- Observation: the namespace repair restored the intended on-policy relationship for the retained SFT actor.
  Evidence: local run `fa70e18e-8a43-4fb6-941e-daef64bc7609` measured a first-rollout mean absolute sampler/actor log-probability delta of `0.006096` across 4,607 selected tokens, below the `0.05` gate and below the earlier matched base-policy value of `0.01436`. The broken SFT path measured `0.25319`.

- Observation: the checkpoint is sufficient to recover optimizer progress under a new process and run identity.
  Evidence: recovery run `train.grpo-sft-parity-resume-local-source-01` linked the source `training-checkpoint:v0` as an input at global step one, executed one resumed update, finished at global step two, and published `model-adapter:v1` plus `training-checkpoint:v1`. Both manifests contain no full-model weights.

- Observation: the advantage calculation is internally correct for the current scalar-reward DAPO contract, but it is not a decoupled multi-reward advantage.
  Evidence: TRL applies the weighted verifier outputs as one scalar, subtracts the per-prompt group mean, and divides by the sample standard deviation plus `1e-4`. Recomputing the 32 groups × 4 completions from the retained traces matched the expected zero-centered advantages and the recorded zero-variance rates (`4/32` then `2/32`). The framework bridge exposes only one `_bridge_reward` value, while the environment has nine weighted reward components.

- Observation: the current DAPO run uses group reward scaling even though the project guidance recommends no group-standard-deviation scaling for Dr-GRPO-style learning.
  Evidence: the translator hardcodes `scale_rewards: "group"`; the retained settings select `loss_type: "dapo"`. DAPO supplies token-count normalization for the policy loss, but it does not remove the question-difficulty bias introduced by group standardization.

- Observation: the corrected scalar-DAPO path produces a centered, non-degenerate learning signal and excludes truncated completions from the group baseline.
  Evidence: the two-update diagnostic recorded advantage means within `1e-8` of zero, positive/negative/zero fractions of `0.540/0.357/0.103` at step two, group reward standard deviation `0.2485`, and scorable fraction `126/128` after two truncated rows were masked. Raw trace reward, triple-F1, submit rate, and reasoning rate all increased between the two rollouts.

- Observation: the corrected two-step result is evidence that the implementation is functioning, not a controlled proof that it outperforms the previous run.
  Evidence: the corrected run is stochastic and its first rollout started at a different reward level. Its raw reward delta was `+0.0498` versus `+0.0456` in the prior run, while its triple-F1 delta was `+0.0326` versus `+0.0421`; repeated seeds or held-out evaluation are required for an algorithm comparison.

- Observation: current evidence shows a healthy learning signal for two updates, not generalization or production readiness.
  Evidence: mean reward and mean triple-F1 increased after the first actor update, zero-variance groups decreased, gradient norm stayed finite with no clipping, and the second rollout was generated after the updated actor. The run is only two optimizer updates and does not include a held-out evaluation.

- Observation: truncation telemetry must not change legacy unmasked training semantics.
  Evidence: the first implementation treated inferred max-length completions as unscorable even when masking was disabled; the focused TRL regression caught five zero-gradient failures. Restricting reward exclusion to the explicit mask setting restored all seven tests while retaining truncation visibility.

- Observation: the first local-source canaries used Trackio's local project store rather than the protected remote Trackio endpoint.
  Evidence: the runs are present in the local SQLite artifact lineage and absent from the remote normalized source. They prove trainer and artifact behavior but are not immutable-image or production-observability qualification.

## Decision Log

- Decision: Reuse `vllm_weight_name_prefix` for native LoRA synchronization rather than add a Qwen-specific rewrite.
  Rationale: full-parameter synchronization already uses this generic boundary to reconcile a text-only actor with a composite vLLM model. Native LoRA should preserve the same meaning by inserting the prefix after PEFT's `base_model.model.` envelope before vLLM loads the temporary adapter.
  Date/Author: 2026-08-08 / Codex

- Decision: Gate the first training rollout on the mean absolute actor-versus-sampler token log-probability delta before any optimizer update.
  Rationale: this directly tests the on-policy invariant over the exact completion tokens used for training. A default limit of 0.05 separates the qualified base evidence at 0.014 from the broken SFT bridge at 0.253, while retaining importance correction for small numerical differences.
  Date/Author: 2026-08-08 / Codex

- Decision: Do not use the manually remapped SFT adapter as the actor checkpoint.
  Rationale: Transformers/PEFT owns the native actor namespace. The remap is a deployment representation; feeding it back into the actor would move the mismatch to the training side and would not solve subsequent adapter refreshes.
  Date/Author: 2026-08-08 / Codex

- Decision: Stop before committing or publishing the fork unless the user explicitly authorizes those irreversible repository actions.
  Rationale: local implementation, tests, and a one-step canary are reversible. The repository guidance requires a clean pushed fork commit before changing immutable consumer pins, while the user's standing workflow preference is not to commit without an explicit request.
  Date/Author: 2026-08-08 / Codex

- Decision: Treat the final adapter and the recovery checkpoint as different contracts.
  Rationale: the final `model-adapter` is a portable inference/training input and may contain only PEFT adapter weights plus small configuration/tokenizer files. The `training-checkpoint` may additionally contain optimizer, scheduler, RNG, scaler, and trainer state required for exact restart, but it must not duplicate the immutable base-model weights for LoRA/QLoRA runs.
  Date/Author: 2026-08-08 / Codex

- Decision: Qualify restart through a new process/job identity, not by calling `train()` twice on the same trainer.
  Rationale: job recovery depends on artifact retention, materialization, request plumbing, and trainer resume semantics. An in-process continuation would bypass the failure boundaries that need proof.
  Date/Author: 2026-08-08 / Codex

- Decision: Apply the recovery contract to every maintained TRL training family, not only policy-gradient methods.
  Rationale: resumability is an execution concern shared by SFT, DPO, GRPO/DAPO, SAMPO, and on-policy distillation. The model payload differs by update kind, but trainer, optimizer, scheduler, and RNG restoration have the same recovery boundary.
  Date/Author: 2026-08-08 / Codex

- Decision: On interruption, publish only the latest complete checkpoint and never mask the original failure if retention itself fails.
  Rationale: this preserves a known-consistent state and the actual run error. The retention failure is attached as an exception note for diagnosis rather than replacing the training outcome.
  Date/Author: 2026-08-08 / Codex

- Decision: Do not promote the current advantage path as GDPO or as fully decoupled multi-signal learning.
  Rationale: the math is correct for scalar DAPO, but the bridge has already reduced the verifier signals to one weighted reward before TRL sees them. Production qualification must first choose between preserving scalar DAPO with better telemetry or implementing per-component normalization/aggregation deliberately.
  Date/Author: 2026-08-08 / Codex

## Outcomes & Retrospective

Local implementation, focused validation, the one-step parity canary, the fresh-process restart canary, and both two-update 32 × 4 DAPO diagnostics are complete. The repaired SFT actor reduced the first-rollout mean delta from `0.25319` to `0.006096`, then resumed its step-one checkpoint and advanced to step two with a `0.006693` delta. The corrected scalar-DAPO run now records advantage, group-spread, truncation, and importance-ratio clamp telemetry; it produced a finite, centered signal and improved its rollout metrics after the first update. This verifies the calculation and truncation fix, but not superiority over the prior stochastic run. Repeated-seed or held-out qualification, fork publication, immutable pinning, an image rebuild, and repetition from that immutable image remain open; the local-source runs must not be described as release qualification.

## Context and Orientation

`/home/hammad/projects/trl/trl/generation/vllm_generation.py` owns model synchronization into colocated vLLM. In `weight_sync_mode="lora"`, it asks the PEFT actor to write an adapter into a temporary directory, and vLLM reloads that directory through one stable `LoRARequest`. The same class already accepts `weight_name_prefix`, but currently rejects the prefix when LoRA synchronization is selected.

`/home/hammad/projects/trl/trl/trainer/grpo_trainer.py` receives sampling log probabilities from vLLM and recomputes the same token log probabilities with the actor. It already logs their absolute difference and builds an importance-sampling ratio. The parity gate belongs immediately after this comparison and before `_generate_and_score_completions` returns data to the optimizer.

`packages/train/src/posttrain/train/backends/trl/grpo.py` translates a backend-neutral inference binding into `GRPOConfig`. Ambient Agent's `.posttrain/catalog/inference.yaml` owns the Qwen3.5 rollout-engine namespace setting. The fork's `CARBONTEQ_FORK.md` owns generic implementation provenance; `docs/tooling/trl/README.md` owns consumer configuration and qualification evidence.

## Plan of Work

In the TRL fork, allow a validated `weight_name_prefix` with colocated native-LoRA synchronization. After `save_pretrained(..., safe_serialization=True)`, rewrite the temporary safetensors keys atomically. For ordinary PEFT keys beginning with `base_model.model.`, insert the prefix immediately after that envelope. For keys without the envelope, prepend the prefix. Do not prefix a key twice. Preserve safetensors metadata and reject collisions.

Add focused tests that construct a tiny temporary adapter, synchronize it with `weight_name_prefix="language_model."`, and prove that `base_model.model.model.layers...` becomes `base_model.model.language_model.model.layers...`. Prove that already-prefixed keys remain unchanged and the actor's source adapter is not mutated.

Add `vllm_policy_parity_max_mean_logp_delta` to `GRPOConfig`, defaulting to `0.05`; `None` explicitly disables the gate for non-on-policy research. When vLLM is active, recompute actor token log probabilities for parity even if importance correction is disabled. On the first training rollout, compare the globally gathered mean absolute difference with the limit and raise a clear `RuntimeError` before optimizer step one when it exceeds the limit. Preserve the existing metrics and importance correction for passing runs. Add configuration validation and direct tests for pass, fail, disabled, and non-first-step behavior.

Update both fork ledgers in the same logical change. In Ambient Agent, select `weight_name_prefix: language_model.` for every Qwen3.5 inference binding that uses native LoRA synchronization, then run catalog/work-package validation. Do not change immutable framework pins until the fork is committed and pushed.

In the framework, make the LoRA artifact contract executable rather than trusting the declared kind. Validate the final output and every retained recovery checkpoint before publication: PEFT adapter weights are allowed; tokenizer/configuration and optimizer/scheduler/RNG/trainer metadata are allowed where appropriate; full-model safetensor/bin shards and their indexes are forbidden for LoRA/QLoRA. Keep full-update behavior unchanged. Add focused unit tests and inspect the retained diagnostic's real Trackio manifests against the same policy.

Trace restart from the selected `training-checkpoint` artifact through work-package input materialization into `GRPORequest.resume_from` and then TRL's `resume_from_checkpoint`. Add an integration-style regression using two separate trainer instances or worker invocations: the first produces a step-one checkpoint, the second receives only the materialized recovery artifact, restores it, and advances to step two while preserving the LoRA adapter representation. Reject a final-model adapter supplied where recovery state is required.

Use the local TRL source for a one-step SFT parity canary on the free RTX 4090. Keep the same model, 32 prompts, four generations, and rollout engine only if needed to reproduce the failure; a smaller deterministic group may be used first to verify the gate cheaply. Acceptance requires the gate to pass and the actor/sampler evidence to be comparable to the base-policy diagnostic. If it fails, retain the failure and do not run the two-update comparison.

## Concrete Steps

Work first from `/home/hammad/projects/trl`:

    uv run pytest tests/test_vllm_generation.py -k 'lora or weight_name_prefix' -q
    uv run pytest tests/test_grpo_trainer.py -k 'policy_parity or importance_sampling' -q
    uv run ruff check trl/generation/vllm_generation.py trl/trainer/grpo_config.py trl/trainer/grpo_trainer.py tests/test_vllm_generation.py tests/test_grpo_trainer.py
    git diff --check

Then work from `/home/hammad/projects/rl`:

    uv run pytest packages/train/tests/test_trl_online_rl.py packages/train/tests/test_trl_vllm_compat.py -q
    uv run pytest packages/train/tests/test_api.py packages/train/tests/test_retention.py -q
    uv run ruff check packages/train/src/posttrain/train/backends/trl/grpo.py packages/train/tests/test_trl_online_rl.py packages/train/tests/test_trl_vllm_compat.py
    uv run pyright packages/train/src/posttrain/train/backends/trl/grpo.py
    uv run lint-imports
    git diff --check

From `/home/hammad/projects/ambient-agent`, validate the catalog and planned work package with its locked environment before launching the GPU canary. Record the exact command and Trackio run identity here when resolved.

## Validation and Acceptance

The LoRA export regression must fail before the change because `VLLMGeneration` rejects `weight_name_prefix` with `weight_sync_mode="lora"`. After the change it must observe the exact Qwen3.5-compatible temporary key and must not mutate the source actor adapter.

The parity-gate regression must raise before the optimizer callback for a synthetic mean log-probability delta above 0.05 and must pass below the limit. A real one-step run is mandatory. It must retain the same actor and rollout adapter identity, report a mean delta below 0.05, avoid an importance ratio pinned at its lower bound, and publish traces plus a recoverable failed or successful summary. Only a passing one-step result authorizes the two-update repeat.

For LoRA/QLoRA, the published model artifact must contain adapter weights and adapter configuration and must contain no base-model weight shard or full-model index. The recovery artifact may contain adapter weights plus trainer, optimizer, scheduler, scaler, and RNG state, but likewise must contain no base-model weight shard. A fresh-job resume must consume the recovery artifact, restore the recorded global step, advance it, and produce a new adapter-only model artifact and bounded latest recovery checkpoint.

The final reproducibility gate requires a clean TRL fork commit pushed to `carbonteq-ai/trl`, an exact consumer pin and regenerated `uv.lock`, matching source revision in the Ambient Agent training binding, a rebuilt immutable job image, and the passing parity and restart canaries repeated from that image. These publication steps remain pending explicit commit authorization.

## Idempotence and Recovery

Safetensors rewriting occurs only in TRL's disposable temporary LoRA directory and uses a replacement file, so interruption cannot corrupt the retained actor checkpoint. Prefix insertion is idempotent. Unit tests use temporary directories. A parity failure occurs before the optimizer step; normal run finalization should retain the diagnostic traces and error without producing a misleading trained adapter. Recovery selection is explicit and immutable: a retry materializes the retained checkpoint under a new run workspace and never mutates the source artifact. If restart fails, preserve both workspaces and the artifact receipt; do not fall back silently to starting from the base or SFT adapter.

If the one-step canary fails after the namespace fix, do not loosen the threshold. Compare exact tokenization, chat-template rendering, active adapter fingerprints, and vLLM-loaded module counts. Keep the original retained SFT adapter unchanged and rerun under a new run identity after each correction.

## Artifacts and Notes

The broken comparison run is Trackio run `train.grpo-0b8ad49f`, provider id `e97e46d4315d40d9847e1a551a60a20e`. It retained 256 traces and a global-step-2 recovery checkpoint. It is defect evidence, not a promotable model.

The passing local-source parity run is `fa70e18e-8a43-4fb6-941e-daef64bc7609`, local Trackio provider id `d5ad7d23274d41bfaba79ef7ec802d72`. Its adapter manifest is 53,710,717 bytes; its recovery manifest is 121,323,697 bytes. The fresh-process recovery run is `train.grpo-sft-parity-resume-local-source-01`, local provider id `5016109cc82144b68c5b433ac16ef0fc`; it finished at global step two and its recovery artifact links to the source checkpoint as an input.

The relevant key shapes are:

    actor:   base_model.model.model.layers.0....lora_A.weight
    rollout: base_model.model.language_model.model.layers.0....lora_A.weight

The matched base run's first-update mean absolute log-probability delta was `0.01436`; the broken SFT-start run's was `0.25319`.

## Interfaces and Dependencies

`VLLMGeneration.weight_name_prefix` retains its existing public meaning and becomes valid for both `weight_sync_mode="full"` and `"lora"`. `GRPOConfig.vllm_policy_parity_max_mean_logp_delta` is a new optional float expressed in absolute natural-log probability units per selected completion token. The gate uses the existing completion/tool loss mask and actor/sampler log-probability tensors; it does not add another model forward pass when importance correction is already enabled.

Revision note (2026-08-08 02:39Z): created this plan from the retained SFT-start defect evidence and resolved the implementation boundary, safety gate, multi-repository order, and publication stop condition.

Revision note (2026-08-08 03:05Z): made adapter-only publication and fresh-job recovery qualification first-class acceptance gates rather than relying on artifact type labels or in-process continuation.

Revision note (2026-08-08 03:31Z): completed the local fork and framework implementation, generalized recovery to every TRL training family, recorded synchronous failure-path artifact retention, and left publication plus real parity/restart canaries as explicit open gates.

Revision note (2026-08-08 04:29Z): recorded passing local-source parity and fresh-process restart canaries, including exact metrics and artifact lineage, and distinguished that evidence from an immutable-image release qualification.

Revision note (2026-08-08 06:53Z): corrected scalar-DAPO scaling and truncation-baseline semantics, added learning-signal telemetry, completed the two-update diagnostic, and recorded why this evidence is not yet a controlled superiority claim.

Revision note (2026-08-08 07:24Z): completed a post-run audit of unmasked truncation behavior and restored compatibility while keeping generic truncation telemetry enabled.
