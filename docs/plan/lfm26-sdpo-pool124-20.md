# Run twenty LFM2.5-2.6B SDPO updates on the full training pool

## Purpose / Big Picture

The user authorized a new twenty-update LFM experiment on2026-10-18 after the
ten Qwen SDPO attempts were closed. This is a separate experiment, not an
eleventh attempt inside that exhausted budget. Test whether LFM learns native
tool workflows from a self-teacher with verified successful own/sibling context
or native hindsight feedback. Measure its own fresh base and checkpoints10/20
on heldout62. A checkpoint is the saved student after actual optimizer updates.

The frozen product baseline is unchanged. This is an unpublished research
prototype under scripts/qualification, outside the public frozen-teacher API.
No framework package boundaries, dependency pins, environment scoring source or
prior accepted capsules change. Existing dirty work is preserved; no commits or
pushes are authorized. This plan follows docs/templates/PLAN.md and the canonical
workflow, selections, evidence and framework documents under docs/post-training.

## Progress

- [x] User deferred this experiment before GPU submission. Direct Docker/state
  check confirmed no owned container, no train evidence and no submission receipt.
  CPU-tested source/capsule is retained only; do not launch under this superseded
  authorization. New direction is task-quality curation and reasoning research.
- [x]2026-10-18: Direct SSH shows no GPU compute processes or running containers.
  Cached LFM weights match the catalog pin654f9463ce32b05d0429d76fe1f580b27d4c1ac0.
- [ ] Implement and CPU-test the model-specific loader, prompt and native parser.
- [ ] Seal a fresh capsule and record its source hashes/config before submission.
- [ ] Qualify native tool/result/final transport, exact32-way batch invariance,
  actor/rollout token scores and full16K actor backward before any learning.
- [ ] Execute at most20 updates, batch32, with all124 tasks in round-robin order.
- [ ] Reconcile310 native heldout episodes each at0/10/20 and report tool errors,
  success, preservation, exclusions and paired task uncertainty.

## Context and Orientation

Work in /home/hammad/projects/worktrees/rl-rft on codex/rft-warm-start. The previous
driver scripts/qualification/sdpo_agent_ablation.py owns fresh native collection,
original-token transition association, SDPO objective, Adam updates, EMA teacher
and checkpoint files. SDPO scores the original student action under an enriched
teacher prefix; it does not require an external model. EMA means updating teacher
adapter weights toward the student by a declared fraction after every update.
The optimized rollout session in self_feedback_vllm.py owns sleeping/waking the
single GPU engine and loading each changed student adapter. It must be qualified
on LFM rather than assuming Qwen's architecture or parser applies.

New model-specific code belongs in sdpo_lfm_runtime.py and the standalone launcher
run_sdpo_lfm20.py. Existing generic research code gets only explicit opt-in model
and execution profiles; Qwen defaults and immutable old capsules remain intact.
LFM uses AutoModelForCausalLM, its native tokenizer template/EOS and native lfm2
tool parser. Never apply Qwen XML coercion or Qwen thinking flags to LFM.

## Decision Log

Decision2026-10-18: use the prior broad non-thinking recipe as the comparison
starting point:124 training tasks,32 distinct tasks per update, two fresh online
trajectories/task; one complete action/task selected by failure-frontier sampling.
LFM uses its native thinking-prefilled instruction template, for both student
and teacher; compare its own fresh thinking base. This differs from Qwen's
non-thinking mode and is not a controlled model-family-only comparison. Do not
add Luna or corrected-call CE.
Training16Kcontext,2048reply,12288episode generated tokens,8turns; evaluation
64Kcontext,8192reply,32768episode tokens,20turns,62tasksx5. No training truncation:
exclude whole incomplete/overlong transitions, and fail on insufficient task
coverage rather than silently replacing tasks.

Decision2026-10-18: fresh BF16 base, LoRA rank16/alpha32/dropout0, all supported
text linear layers, reverse KL1/top20+tail, EMA update rate.01, AdamW decay.01,
clip1, microbatch1/accumulation32. LR1e-5 at updates1–10, cosine cooldown to1e-6
at20. No warmup, external models or raw-feedback prediction auxiliary. The
teacher and student are the same pinned base with separate adapters.

Decision2026-10-18: retain a7200-second total GPU container cap. All qualification,
baseline evaluation, training and checkpoint evaluation share that deadline;
do not silently shorten evaluations to fit. If it expires, preserve incomplete
evidence and report the missing populations. There is no automatic extra attempt.
Save each update so an interrupted attempt has exact retained weights; automatic
crash resume is not authorized. Gates in the same run consume zero extra updates;
the first two real optimizer updates also qualify persistence, not a separate fit.

## Plan of Work / Concrete Steps

Read the catalog pin in packages/catalog/src/posttrain/catalog/base/models.yaml
and remote cached config before sealing. CPU-preflight the actual pinned parser
and tokenizer without GPU. Stage a new immutable directory
/var/lib/posttrain/qualifications/sdpo-lfm26-pool124-20-20261018-r1 containing driver,
cohorts.json, train-pool.json, heldout62.json, experiment.json, source-sha256.txt
and an empty evidence directory. Source inputs come from the already sealed
full124 Qwen capsule, with exact hashes and disjoint124/62 populations checked.

SSH uses dstack@192.168.110.12 with key
/home/hammad/projects/ai-infra/.state/ssh/dstack_worker; bypass default SSH config
and agent with -F /dev/null, BatchMode=yes, IdentitiesOnly=yes, IdentityAgent=none,
GSSAPIAuthentication=no and ConnectTimeout=8. Never print protected credentials.
Before launch inspect exact owned Docker state, protected scheduler and real GPU
processes. Monitor an existing identity rather than duplicate it. The launcher
must verify all source hashes and reject an existing container/evidence path.

Use the existing offline LAN image pinned by digest
e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337,
the read-only self-campaign-2048-20261017-r5 runtime and read-only HF cache.
Native traces, generation token IDs, update records and adapter files stay in
the new evidence/train directory. No paid cloud, competing GPU or source edits
inside an accepted capsule are allowed.

## Validation and Acceptance

CPU tests must reject wrong model revisions, profiles and pool overlap; demonstrate
that native LFM rendering preserves tool argument values and special delimiters;
and leave existing Qwen tests passing. Real GPU gates must show correct native
lookup_code arguments, consume an unpredictable result and return it in final
content. Exact32-way token/logprob batch invariance and actor/vLLM selected-token
parity must pass unchanged thresholds. Full16K backward must be finite with no
optimizer update and unchanged weights. Each actual update must have32 distinct
selected tasks, finite loss/gradient, correct Adam moment step, changed student
adapter and EMA checkpoint; inference must load each new policy before collection.

Reporting requires310 original episodes each at0/10/20, exact ordered heldout
task digest ed81cf1fe88f68bb8420bb46c5fdf3afde24660c7043812da680bb1c7bc9196f
and frozen environment a0c7378b179e2b267382b8125192876502d82f89. Report actual
success, tool mistakes, unknown IDs, repetition, limits, scoring errors and
preservation checks. Paired task-bootstrap uncertainty is separate from absolute
uncertainty. Missing scores remain missing, not zero. Earlier historical LFM
sampling results are not matched controls. Repeated heldout checkpoint inspection
is descriptive and cannot independently validate the retrospectively best step.

## Idempotence and Recovery

An accepted identity is never restarted or overwritten. Any gate failure stops
the run and preserves its source, logs and all native evidence. An observation
timeout is not a job failure; inspect the same container again. No implicit retry,
checkpoint continuation, changed budget or extension beyond20updates is allowed.

## Surprises & Discoveries

LFM's pinned architecture is Lfm2ForCausalLM with convolution and attention
layers. Qwen's multimodal model loader and XML parser are inappropriate. The
native cached template and parser must therefore be tested directly.

## Outcomes & Retrospective

Deferred at user request after CPU preflight only. No LFM optimizer update,
GPU submission or learning gain occurred. Retain preparation for explicit future
reuse; there is no automatic launch or monitor for this experiment.

## Artifacts and Notes / Interfaces and Dependencies

Durable local evidence: docs/research/evidence/sdpo-lfm26-20261018. Native
authority remains Verifiers traces; the model/TRL/vLLM pins are unchanged.
The old ten-attempt ledger remains closed and separate. Final reporting must
state whether the20-step run finished, failed a gate or exhausted its time cap.
