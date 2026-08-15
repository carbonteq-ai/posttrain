# Prevent implicit one-hour termination of remote training jobs and release Posttrain 0.3.18

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not contain `.agents/PLAN.md`. This document therefore follows `docs/templates/PLAN.md`, the repository's checked-in ExecPlan authority, and must continue to be maintained in accordance with it.

## Purpose / Big Picture

Remote training must not be terminated after one hour merely because the submitter omitted a Posttrain provider timeout while the actual training recipe was designed to run much longer. After this hotfix, Posttrain will reject a remote training launch before building or publishing an image when its provider wall-clock deadline still comes from the implicit one-hour job default. The error will tell the developer to set a project execution timeout or pass the CLI option explicitly. A successful plan and submission will make the resolved timeout and its source visible and durable.

The hotfix also exposes dstack's already-existing bounded diagnostic log stream through `posttrain run logs`. This is a diagnostic access fix, not a new logging platform: it does not copy raw logs into Trackio, Observatory, Apache Doris, or another LAN service. A developer can use the diagnostic stream to see a provider termination reason such as `max_duration_exceeded` even after the workload itself has stopped.

The release outcome is Posttrain `0.3.18`, promoted from retained candidate bytes without rebuilding them. Once stable publication is proven, submit a new Ambient Agent base-model, 200-optimizer-step run with the intended task set and an explicit 90,000-second provider deadline. The failed run remains immutable evidence and is never resumed, overwritten, purged, or cancelled by this plan.

## Progress

- [x] (2026-08-15 15:06Z) Reconstructed the incident and identified Posttrain's implicit `timeout_seconds=3600` provider policy as the owning cause.
- [x] (2026-08-15 15:06Z) Confirmed that dstack enforced the supplied `max_duration=3600` correctly and that the training runtime timeout is a separate control.
- [x] (2026-08-15 15:06Z) Confirmed that dstack already retains workload and diagnostic streams and supports diagnostic reads without a new storage service.
- [x] (2026-08-15 15:06Z) Narrowed this plan to the hotfix, release, and corrected qualification run; deferred centralized logging architecture.
- [x] (2026-08-15 22:19Z) Preserved the remote-builder branch and created `codex/execution-timeout-hotfix` from `origin/main` in `/home/hammad/projects/rl-timeout-hotfix`.
- [x] (2026-08-15 22:20Z) Added regression tests for implicit remote-training timeouts, receipt policy persistence, dstack deadline mapping, diagnostic log selection, and CLI compatibility.
- [x] (2026-08-15 22:20Z) Implemented the planning guard, concise timeout/source plan fields, submission schema v7, and bounded dstack diagnostic stream selector.
- [x] (2026-08-15 22:21Z) Added the Ambient Agent project execution timeout and campaign contract test without staging unrelated dirty work.
- [ ] Complete cold-install validation through a staged release source, then commit and push the reviewed hotfix.
- [ ] Publish and qualify the `0.3.18` development candidate, merge the hotfix, and promote the exact retained candidate bytes to stable.
- [ ] Submit the corrected base-model 200-step job and prove its identity, task set, image digest, and dstack `max_duration` before monitoring training.
- [ ] Record final release and run evidence in this plan and complete the retrospective.

## Surprises & Discoveries

- Observation: The failed run was not stopped by TRL, vLLM, reward clipping, policy loss, or the model process. dstack stopped it because Posttrain supplied a one-hour provider deadline.
  Evidence: dstack run `pt-f6fefdbb9eacd0f055077034`, corresponding to Posttrain run `48703e11-c591-4762-82a5-2162a7b446de`, reported `Max duration exceeded max_duration=3600`, `termination_reason=max_duration_exceeded`, and executor exit code 130 after only optimizer steps 1 through 4 were retained.

- Observation: The Ambient training binding contained `TrainingBinding.runtime.timeout_seconds: 90000`, but that value does not own the scheduler's wall-clock lifetime and was not projected into `ExecutionPolicy`.
  Evidence: `packages/train/src/posttrain/train/bindings.py` owns training-backend runtime settings, while `apps/cli/src/posttrain_cli/execution_planning.py` independently creates `ExecutionPolicy` from resolved project/CLI execution settings. The submitted command omitted `--timeout-seconds`.

- Observation: The dstack adapter behaved correctly for the policy it received.
  Evidence: `packages/execution-dstack/src/posttrain_execution_dstack/adapter.py` maps `request.policy.timeout_seconds` directly to dstack `max_duration`.

- Observation: The one-hour default predates the current release work.
  Evidence: commit `c6fe4609` introduced the provider adapter and job-capsule CLI with the 3,600-second default; later refactoring changed where resolution is surfaced but did not create the semantic mismatch.

- Observation: dstack already stores two useful log streams, so diagnosing this incident does not require a new central log sink.
  Evidence: the dstack SDK exposes ordinary job logs and runner diagnostics through its existing log API with a diagnostic selector. Posttrain currently requests only ordinary job logs.

- Observation: `v0.3.17` is already a stable release, not merely a development candidate.
  Evidence: the GitHub release was published on 2026-08-14 and the successful stable workflow used retained candidate artifacts. This hotfix must advance to `0.3.18`.

- Observation: the current Posttrain checkout contains clean, pushed remote-builder work that is not merged into `origin/main`, while Ambient Agent has extensive unrelated dirty work.
  Evidence: the current Posttrain branch is `codex/release-0.3.17-finalize`; Ambient Agent contains modified and untracked campaign files. The hotfix must use isolated worktrees and exact-file staging.

## Decision Log

- Decision: Enforce an explicit provider wall-clock timeout for every remote training job, while retaining the one-hour implicit default for local and non-training compatibility.
  Rationale: A short implicit default is safe for bounded utility jobs but unsafe for expensive training. Rejecting only remote job kinds whose name begins with `train.` closes the incident class without silently changing existing local and evaluation behavior.
  Date/Author: 2026-08-15 / Codex

- Decision: Treat CLI, machine-local, and committed project configuration as explicit timeout sources; treat the `job` fallback source as implicit.
  Rationale: `ResolvedExecutionSettings.sources` already carries provenance. Reusing this contract preserves the documented precedence `CLI > local > project > job` and avoids adding a second configuration mechanism.
  Date/Author: 2026-08-15 / Codex

- Decision: Do not infer the provider wall-clock deadline from `TrainingRuntime.timeout_seconds` and do not require the two values to match.
  Rationale: The training backend can use its timeout for internal requests or process behavior, whereas the provider deadline governs the whole scheduled execution, including image startup, downloads, recovery, checkpointing, and cleanup. Conflating the clocks would break backend independence and still leave non-TRL implementations ambiguous.
  Date/Author: 2026-08-15 / Codex

- Decision: Persist the exact submitted `ExecutionPolicy` in a new backward-readable execution-submission schema.
  Rationale: A durable receipt must answer what Posttrain asked the provider to enforce even when provider state is later unavailable. Existing v1 through v6 submissions remain readable; new writes advance the schema and include the policy.
  Date/Author: 2026-08-15 / Codex

- Decision: Add a generic `workload` versus `diagnostic` log-stream selector to the execution provider contract, but implement diagnostics only where the provider supports them.
  Rationale: The distinction is not dstack-specific, while availability is provider-specific. The workload stream remains the compatibility default. An unsupported diagnostic request must fail clearly rather than silently returning the workload stream.
  Date/Author: 2026-08-15 / Codex

- Decision: Do not build a centralized raw-log service in this hotfix.
  Rationale: Shipping cloud-fleet raw logs continuously into a LAN-hosted database would add VPN bandwidth, latency, outage coupling, scaling, and data-residency risks. A future design must keep raw logs provider- or region-local and federate bounded queries or summaries deliberately.
  Date/Author: 2026-08-15 / Codex

- Decision: Release the framework hotfix as `0.3.18` without releasing any maintained fork.
  Rationale: dstack already enforces `max_duration` and already exposes diagnostic logs. The behavioral defect is in Posttrain planning and presentation, so no dstack fork delta or manual fork release is necessary.
  Date/Author: 2026-08-15 / Codex

- Decision: Use a bounded release canary before stable promotion, then submit the 200-step experiment after stable publication.
  Rationale: A 200-step GPU job is experiment evidence, not an efficient packaging gate. The candidate gate should prove installation, image composition, submission policy, and short execution without waiting for the full experiment.
  Date/Author: 2026-08-15 / Codex

## Outcomes & Retrospective

Implementation and local source validation are complete: focused tests, the full 1,257-test suite, type checking, import-boundary checking, and a release-readiness receipt all passed. The remaining work is staged-consumer validation, publication, promotion, then submission and two-step observation of the corrected experiment. Update this section after each milestone with the observed result, evidence locations, remaining gaps, and any deviation from the acceptance criteria below.

## Context and Orientation

Posttrain separates job meaning from provider execution. A job recipe describes what training or evaluation should do. `apps/cli/src/posttrain_cli/execution_planning.py` packages that meaning and resolves where and how long the provider may run it. `packages/project/src/posttrain/project/execution_settings.py` merges execution settings in descending precedence: an explicit command-line value, machine-local configuration, committed project configuration, and finally a framework job default. Its `ResolvedExecutionSettings.sources` map records which layer supplied each value.

The phrase *provider wall-clock deadline* means the maximum lifetime of the whole scheduled execution. Posttrain represents it as `packages/execution/src/posttrain/execution/contracts.py::ExecutionPolicy.timeout_seconds`. The dstack adapter converts that value to dstack's `max_duration`. This clock includes startup and training and is therefore distinct from `TrainingBinding.runtime.timeout_seconds`, which belongs to a selected training implementation. No reusable training package may import the dstack adapter, and the generic execution package must remain scheduler-neutral.

The CLI entry points for job planning and submission are in `apps/cli/src/posttrain_cli/commands/job.py`. `plan_job_launch` in `apps/cli/src/posttrain_cli/execution_planning.py` is the earliest point where the resolved provider, job kind, timeout, and timeout source are all available. It is the correct place to reject an unsafe remote training plan before source packing, image building, registry publication, or provider submission begins.

`packages/execution/src/posttrain/execution/service.py::ExecutionSubmission` is the compact local receipt mapping a canonical Posttrain run identifier to a provider run. It is written under the ignored `.posttrain/state/executions/<run-id>/submission.json` tree. It currently records identity, image, provider, tracking source, and provider source but not the exact execution policy. The hotfix evolves this receipt additively so new submissions preserve the policy while old schemas continue to load.

`packages/execution/src/posttrain/execution/contracts.py::ExecutionProvider.logs` is the scheduler-neutral bounded log interface. The local provider is implemented in `packages/execution-local/src/posttrain_execution_local/adapter.py`; dstack is implemented in `packages/execution-dstack/src/posttrain_execution_dstack/adapter.py` and its isolated SDK bridge in `packages/execution-dstack/src/posttrain_execution_dstack/sdk_bridge.py`. `apps/cli/src/posttrain_cli/commands/run_cmd.py::run_logs_cmd` currently reads only the default workload stream using an offset and a maximum page size. A *diagnostic stream* is provider/runner output about scheduling, startup, shutdown, and termination; it is not the model process's stdout/stderr.

The failed run is immutable evidence. Posttrain run `48703e11-c591-4762-82a5-2162a7b446de` maps to dstack run `pt-f6fefdbb9eacd0f055077034`. It stopped at one hour because its submission carried `max_duration=3600`. A corrected experiment must receive a fresh Posttrain run identifier and provider run identifier.

The release system treats candidate creation and stable promotion as separate states. `.github/workflows/release-candidate.yml` publishes the final version to the development index and retains wheel, runtime-lock, manifest, checksum, and readiness evidence. `.github/workflows/release.yml` accepts a merged source commit and successful candidate run, verifies the candidate's source or equivalent tree, and promotes the exact retained bytes to stable. Stable publication must not rebuild the candidate.

The current Posttrain checkout is not the correct place to mix this hotfix: it is a clean pushed branch carrying separate remote-builder work. Begin from an updated `origin/main` in a dedicated worktree. The Ambient Agent repository at `/home/hammad/projects/ambient-agent` is dirty; edit and stage only the project execution setting and exact campaign test required by this plan. Never use blanket `git add`, reset, checkout, clean, or stash operations in either repository.

## Scope, Non-Goals, and Deferred Work

This plan changes Posttrain execution planning, receipt evidence, CLI log selection, dstack adapter query behavior, tests, release metadata, and the relevant Ambient Agent project configuration/test. It releases Posttrain `0.3.18` and submits one corrected base-model experiment after stable qualification.

This plan does not change training algorithms, reward clipping, policy loss, task data, model templates, Trackio trace semantics, Observatory charts, dstack retention, or dstack server deployment. It does not publish TRL, veRL, Verifiers, Trackio, or dstack forks. It does not cancel any running job. It does not delete the failed run or tracking evidence. It does not make the 200-step run a prerequisite for publishing the stable framework release.

A federated logging architecture is deferred. That separate design must start from the invariant that raw logs remain close to the execution provider or cloud region by default. A LAN control plane may retain small lifecycle facts and bounded summaries; users may request bounded pages on demand through the provider adapter. Any future replication must define opt-in policy, compression, backpressure, redaction, tenancy, egress budgets, provider-local retention, failure behavior, and regional data rules before selecting Doris or another index. None of those concerns should delay this hotfix.

## Invariants

Planning must fail before packing or building when all of the following are true: the resolved provider is not local, the recipe job kind starts with `train.`, and `settings.sources["timeout_seconds"] == "job"`. The failure text must identify the unsafe one-hour implicit default and give both remedies: configure `[execution].timeout_seconds` in `.posttrain/project.toml` or pass `--timeout-seconds`.

An explicit timeout from the CLI, machine-local configuration, or committed project configuration is valid. Precedence remains unchanged. Local jobs and non-training remote jobs retain existing defaults. Provider timeout and training-runtime timeout remain independent positive values.

Human and JSON planning output must expose the resolved timeout and source before any materialization. Submission receipts must record the exact `ExecutionPolicy` sent to the provider. dstack `max_duration` must equal that receipt's `timeout_seconds`; adapters must not invent or clamp a different value.

`posttrain run logs` remains bounded. `--stream workload` is the default and produces existing behavior. `--stream diagnostic` requests only provider diagnostics. A provider that cannot supply diagnostics raises an actionable typed error. No command in this plan copies or tails logs into a new central store.

Release candidate and stable assets are byte-identical. Heavy base and kind images are reused by immutable digest when their inputs did not change. Only the small framework/job layer may change for this release. Stable promotion never rebuilds CUDA, vLLM, veRL, TRL, or other unchanged runtime dependencies.

The corrected experiment uses a fresh run identity, foundation/base model rather than the SFT adapter, exactly 200 optimizer steps, the intended balanced task set, and the RTX PRO 6000 96 GB execution target. The operator proves the resolved provider deadline and job identity before allowing the expensive run to continue.

## Plan of Work

### Milestone 0: Preserve work and freeze incident evidence

Fetch Posttrain refs without altering the current branch, record the current branch and commits, and create a sibling worktree from the current `origin/main` on `codex/execution-timeout-hotfix`. Do not delete or rewrite `codex/release-0.3.17-finalize`; its remote-builder work remains a separate change. In Ambient Agent, record `git status --short`, branch, and HEAD before touching files.

Capture a short redacted incident note in this plan's `Artifacts and Notes`: the two run identifiers, the resolved 3,600-second policy, the dstack termination reason, and the last retained optimizer step. Do not paste environment variables, tokens, full model prompts, or credentials. The milestone is complete when another contributor can reproduce the diagnosis from identifiers and bounded diagnostic output without relying on chat history.

### Milestone 1: Write regressions before changing behavior

In `apps/cli/tests/test_execution_planning.py`, add a test that builds a remote `train.grpo` plan with no explicit timeout and expects planning to fail before packing. Assert that the error names `[execution].timeout_seconds` and `--timeout-seconds`. Add parameterized success cases for project, local, and CLI timeout sources, plus compatibility cases for local training and remote non-training jobs.

In `apps/cli/tests/test_cli.py`, assert that both human and `--json` job-plan output include the timeout and its source. Instrument the fake builder/provider so the unsafe case proves that neither image building nor provider submission was called.

In `packages/execution/tests/test_service.py`, create a new current-schema submission test that round-trips `ExecutionPolicy`, and preserve fixtures proving that all supported older schemas still load. Assert that idempotent submission identity includes the policy so the same run identifier cannot be silently reused with a different deadline.

In `packages/execution-dstack/tests`, add one adapter-plan regression proving that a request with `ExecutionPolicy(timeout_seconds=90000)` produces native `max_duration=90000`. Add log tests showing that workload requests retain current SDK arguments and diagnostic requests select the dstack diagnostic channel. In local-provider tests, assert the selected unsupported-diagnostic behavior.

Run only these focused tests and confirm the new behavioral tests fail for the expected missing guard, missing policy receipt, and missing stream selector rather than fixture mistakes.

### Milestone 2: Implement timeout ownership and durable evidence

In `apps/cli/src/posttrain_cli/execution_planning.py`, add one small validation function called from `plan_job_launch` immediately after settings resolution and before returning a launch. It receives the recipe job kind and `ResolvedExecutionSettings`. It rejects only unsafe remote training with implicit job timeout provenance. Keep the rule in the CLI/composition layer because that layer understands both product job kinds and configured providers; do not put Posttrain job-kind policy into the scheduler-neutral execution contracts.

Extend the existing plan serializers and human presenters used by `apps/cli/src/posttrain_cli/commands/job.py` so they show a field equivalent to `timeout_seconds: 90000` and `timeout_source: project`. Preserve existing JSON keys and add new keys rather than renaming unrelated output.

In `packages/execution/src/posttrain/execution/service.py`, advance the submission schema by one version and add an `execution_policy: ExecutionPolicy | None` field. New submissions require and write the exact policy. Older schemas load with `None` to preserve compatibility. Include the policy in `ExecutionSubmission._identity()` and JSON encoding. Construct it from `plan.request.policy` inside `JobExecutionService.submit`. Update public exports only if the type is needed by callers; reuse the existing `ExecutionPolicy` rather than defining a receipt-specific duplicate.

Keep submit-intent and admission receipts consistent. If the submit-intent already serializes the entire canonical request, do not duplicate policy there. The final submission receipt remains the compact durable lookup surface. Update tests before changing any schema constant, and enumerate every accepted historical schema explicitly so a typo cannot silently reinterpret state.

The milestone is complete when an unsafe dry plan exits non-zero without builder/provider calls, an explicit 90,000-second plan succeeds, and its saved submission contains the same policy sent to dstack.

### Milestone 3: Expose existing bounded dstack diagnostics

In `packages/execution/src/posttrain/execution/contracts.py`, define a provider-neutral log stream type with values `workload` and `diagnostic`. Add a keyword-only `stream` argument defaulting to `workload` on `ExecutionProvider.logs`. Thread the same argument through `packages/execution/src/posttrain/execution/service.py::JobExecutionService.logs` and every fake provider in tests.

In `packages/execution-dstack/src/posttrain_execution_dstack/adapter.py`, include the selected stream in the SDK-bridge payload. In `packages/execution-dstack/src/posttrain_execution_dstack/sdk_bridge.py`, call the existing dstack SDK log API with its diagnostic selector only when `stream == "diagnostic"`; keep the existing call unchanged for workload logs. Preserve UTF-8 replacement, line splitting, cursor behavior, and page limits.

In `packages/execution-local/src/posttrain_execution_local/adapter.py`, retain workload behavior and raise a clear contract/capability error for diagnostics because Docker stdout does not provide an equivalent runner channel. Do not silently map diagnostics to workload output.

In `apps/cli/src/posttrain_cli/commands/run_cmd.py`, add `--stream workload|diagnostic`, defaulting to workload. Include `stream` in JSON output. Keep `--offset`, `--limit`, and `--follow`; the CLI must never issue an unbounded read. For follow mode, use the same stream on every page and stop on terminal states as it does now.

Prove the old failed dstack run can be queried with a small page and that its diagnostic output includes the one-hour termination reason. This is a read-only validation. Do not introduce a database, daemon, background shipper, or retention migration.

### Milestone 4: Correct the Ambient Agent campaign configuration

In `/home/hammad/projects/ambient-agent/.posttrain/project.toml`, add or update only the committed execution timeout setting:

    [execution]
    timeout_seconds = 90000

Preserve any existing provider/target selection semantics. The provider may still be chosen by CLI or machine-local configuration; the committed project timeout documents the campaign's wall-clock envelope independently.

Inspect `/home/hammad/projects/ambient-agent/.posttrain/work_packages/k1a_extract_olmo3_base_2b_rtxpro_concurrent256_200step.yaml` and the catalogs it references. Verify, without renaming by assumption, that it selects the foundation/base model rather than an SFT/PEFT adapter, specifies 200 optimizer steps, uses the intended balanced extraction task selection, and targets the RTX PRO 6000 96 GB profile. Update `/home/hammad/projects/ambient-agent/tests/test_olmo3_base_campaign.py` to lock these facts and the project execution timeout.

Because Ambient Agent is dirty, use `git diff -- <exact-path>` after every edit and stage only exact paths or hunks when a later commit is authorized. Never stage catalogs wholesale if the required campaign entry shares a file with unrelated edits. If isolating the exact delta is impossible, create a new worktree and copy only the reviewed patch with `git apply`; do not reset the user's tree.

Run a JSON job plan using the remote provider and remote builder selection but stop before build/submit. Confirm the output reports the foundation model, campaign/task selection identifiers, 200 steps, target profile, `timeout_seconds=90000`, and `timeout_source=project` unless an intentional CLI override is present.

### Milestone 5: Validate locally before spending runner or GPU time

Run focused tests first, then the full repository validation ladder. Produce a release-readiness receipt locally and run the cold wheel-consumer test. Build packages locally and inspect contents before invoking a GitHub runner.

Use the job plan's image input and transfer report to decide whether the candidate requires a new heavy runtime image. This hotfix changes Python framework behavior, not CUDA or training dependencies, so registry-verify and reuse unchanged universal and kind image digests. If any tool proposes rebuilding those layers, stop and identify the changed lock or Docker input rather than accepting the rebuild.

Materialize one bounded actual-job candidate using the selected builder. Prefer the remote job build service only if its deployed version and qualification evidence are current; otherwise use local BuildKit on the fast workstation. In either case, verify that parent OCI layer descriptors match the retained base/kind image and that only bounded job/framework layers are transferred. Do not qualify the builder by launching the full experiment.

### Milestone 6: Publish and promote Posttrain 0.3.18

Update `release/manifest.toml` and every repository-owned version surface from `0.3.17` to `0.3.18`. Do not alter maintained-fork versions or pins unless local locked validation proves an actual incompatibility. Update release notes and release documentation with the timeout incident, the explicit remote-training invariant, the diagnostic stream selector, and the local-first validation that prevents repeat runner failures.

Commit the hotfix in reviewable units, push `codex/execution-timeout-hotfix`, and run exact-SHA Quality. After Quality and local readiness succeed, dispatch `.github/workflows/release-candidate.yml` against the pushed release branch. Download and verify the retained candidate receipt, wheelhouse, checksums, readiness record, runtime locks, and published image manifest. Perform a bounded RTX PRO 6000 96 GB canary that proves install, plan, build reuse, provider submission, and diagnostic access.

Merge the hotfix PR only after the candidate and canary pass. Dispatch `.github/workflows/release.yml` with the exact merged source SHA and successful candidate run identifier. That workflow must accept the candidate source ancestry or identical tree, restore retained assets, and promote exact bytes. Verify `0.3.18` installs from the stable index in a clean environment, confirm immutable OCI digests, then verify the stable tag/release points at the accepted source. Do not rebuild during promotion.

Trackio and Observatory require no deployment for this hotfix unless implementation unexpectedly changes their packages. Record that result explicitly rather than redeploying unrelated services. If a cross-package dependency unexpectedly forces an application change, add the reason and validation to the Decision Log before expanding scope.

### Milestone 7: Submit and observe the corrected 200-step experiment

From the reviewed Ambient Agent campaign, generate a fresh JSON plan and save it as qualification evidence. Before submission, assert the base model identity, task selection/revision, 200 optimizer steps, RTX PRO 6000 96 GB target, immutable job image digest, and 90,000-second timeout. Build only after these checks pass.

Submit once using a fresh run identity. Immediately read the local submission receipt and dstack native plan/status to prove the requested `ExecutionPolicy.timeout_seconds` and dstack `max_duration` both equal 90,000. This proof occurs before waiting for training and therefore does not consume 25 hours merely to test the hotfix.

Monitor workload output, diagnostic output, Trackio run identity, and Observatory evidence without cancelling the job. At early retained steps, confirm reward, clipping, policy loss, response length/thinking metrics, and task coverage are being recorded, but do not change algorithm settings as part of this hotfix. If training fails for an unrelated reason, preserve the new run evidence and open a separate diagnosis; do not weaken the timeout guard or overwrite the run.

### Milestone 8: Close out the plan

Record exact Posttrain source SHA, version, candidate run, stable release run, wheel checksums, OCI digests, canary run, and experiment run identifiers under `Artifacts and Notes`. Update `Progress`, `Surprises & Discoveries`, and `Outcomes & Retrospective` with actual outcomes. Document any deferred follow-up as a distinct plan, particularly provider-local/federated logging; do not silently turn that follow-up into this release.

## Concrete Steps

Run the following from `/home/hammad/projects/rl` to preserve the current branch and create an isolated hotfix worktree. Choose a sibling path that does not already exist; the example uses `/home/hammad/projects/rl-timeout-hotfix`.

    git status --short
    git branch --show-current
    git rev-parse HEAD
    git fetch origin
    git worktree add -b codex/execution-timeout-hotfix /home/hammad/projects/rl-timeout-hotfix origin/main

Expect the original tree to remain on `codex/release-0.3.17-finalize` and the new tree to report `codex/execution-timeout-hotfix`. If the branch already exists, inspect it and attach a new worktree only when its base and contents are correct; never delete it blindly.

Run focused development tests from `/home/hammad/projects/rl-timeout-hotfix`:

    uv run pytest packages/project/tests/test_execution_settings.py
    uv run pytest apps/cli/tests/test_execution_planning.py apps/cli/tests/test_cli.py
    uv run pytest packages/execution/tests packages/execution-local/tests packages/execution-dstack/tests

After implementation, run the complete local ladder from the same directory:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check
    uv run --no-sync posttrain-release readiness --destination .release/readiness.json
    uv run --no-sync posttrain-release readiness-check .release/readiness.json
    uv run pytest -q tests/consumer/test_wheel_project.py

Run the exact Ambient campaign test from `/home/hammad/projects/ambient-agent` or its isolated worktree:

    uv run pytest tests/test_olmo3_base_campaign.py

Use the installed CLI help from the candidate itself to confirm exact command spelling before running the following behavioral examples. The expected behavior, not incidental formatting, is authoritative:

    posttrain --json job plan --job olmo3-base-200step --provider dstack --builder remote

The JSON must include fields equivalent to:

    "job_kind": "train.grpo"
    "timeout_seconds": 90000
    "timeout_source": "project"
    "optimizer_steps": 200

For a deliberately incomplete fixture/project with no explicit remote-training timeout:

    posttrain job plan --job unsafe-remote-train --provider dstack

Expect a non-zero exit before build or submission and an error equivalent to:

    Remote training requires an explicit provider wall-clock timeout. Set
    [execution].timeout_seconds in .posttrain/project.toml or pass
    --timeout-seconds.

After the diagnostic selector is implemented, query only a bounded page from the historical failed run:

    posttrain run logs 48703e11-c591-4762-82a5-2162a7b446de --stream diagnostic --offset 0 --limit 200

Expect the page to contain a dstack termination fact equivalent to `max_duration_exceeded` and `max_duration=3600`. Do not use `--follow` on this terminal historical query.

Before release dispatch, verify the candidate branch's exact SHA and successful Quality run. Then dispatch the repository workflows using their checked-in input names:

    gh workflow run quality.yml --ref codex/execution-timeout-hotfix
    gh run list --workflow quality.yml --branch codex/execution-timeout-hotfix --limit 5
    gh workflow run release-candidate.yml --ref codex/execution-timeout-hotfix -f source_ref=codex/execution-timeout-hotfix -f qualification_profile=rtx-pro-96gb

Inspect `.github/workflows/release-candidate.yml` at execution time because workflow inputs are versioned source and may gain additional required fields. Do not guess or omit a newly required input. After merge and candidate verification, inspect `.github/workflows/release.yml` and dispatch it with the exact `merged_sha` and `candidate_run_id` names present in that merged workflow.

## Validation and Acceptance

The timeout safety regression is accepted when a remote `train.grpo`, `train.sampo`, or `train.distill` plan whose timeout source is `job` fails before any pack, image build, registry push, or provider API call. Equivalent plans with an explicit project, machine-local, or CLI timeout succeed. Local training and remote evaluation compatibility tests remain green.

Planning evidence is accepted when both human and JSON output show the resolved seconds and provenance. A planned Ambient campaign shows 90,000 seconds from `project` unless intentionally overridden. Changing only `--timeout-seconds` changes the value and source to `cli` without changing the job's model/task identity.

Submission evidence is accepted when a new submission JSON records the exact `ExecutionPolicy`, loads idempotently, conflicts on attempted reuse with a different policy, and all historical supported submission schemas still load. The dstack native plan test and live bounded canary both prove `max_duration == ExecutionPolicy.timeout_seconds`.

Diagnostic access is accepted when workload remains the default, a dstack diagnostic request reaches the SDK diagnostic channel, cursors and limits remain enforced, the old failed run exposes its max-duration termination reason, and the local provider returns an explicit unsupported-capability error rather than unrelated workload lines.

Local readiness is accepted only when the focused tests, full validation ladder, cold wheel consumer, readiness generation/check, and `git diff --check` all pass. If any runner later discovers a failure reproducible locally, add the missing local preflight to release documentation before retrying the runner.

Release is accepted when `0.3.18` has a successful exact-SHA Quality run, successful candidate workflow, retained checksums and manifests, bounded RTX PRO canary, merged source, byte-preserving stable promotion, clean stable-index installation, and an immutable release/tag. Candidate and stable hashes must match; no maintained fork or unchanged heavy runtime image is republished.

The experiment handoff is accepted once a new run has been submitted with the foundation/base model, intended task selection, 200 steps, RTX PRO 6000 96 GB target, immutable image, and both Posttrain and dstack deadlines proven as 90,000 seconds. Completion of all 200 training steps is useful experiment evidence but is not required to prove or publish the timeout hotfix.

## Idempotence and Recovery

All planning and test commands are safe to repeat. Worktree creation is one-time per path; if it is interrupted, inspect `git worktree list` and reuse or remove only the exact incomplete worktree after confirming it has no changes. Never remove the original branch or Ambient Agent dirty tree.

Provider submission is idempotent only for the exact run identity and request. Dry-plan repeatedly, but submit the corrected experiment once. If submission returns ambiguously, inspect `.posttrain/state/executions/<run-id>/submit-intent.json`, `submission.json`, and dstack by idempotency identity before retrying. Never generate another run automatically merely because the client lost its response.

Schema evolution is additive. Do not rewrite old submission files in place. If the new reader fails on historical state, fix backward decoding and tests; do not delete `.posttrain/state`. A new submission that fails between provider creation and receipt save must use existing reconciliation logic rather than a second blind provider submit.

Candidate publication may be retried only according to the release workflow's retained-receipt and candidate-retirement gates. Never overwrite immutable stable files. If the candidate fails before all artifacts are retained, record the failed run and use the workflow's explicit retirement input only after its receipt check authorizes cleanup.

Stable promotion is safe only from the successful retained candidate. If source ancestry changes because of squash merge, use the workflow's identical-tree or allowed release-plumbing proof; do not rebuild. If application deployment is unnecessary, do not deploy Trackio or Observatory merely for symmetry.

The historical failed run is read-only. The new 200-step run must not be killed by this plan. If the operator later decides to stop it, that is a separate explicit action with graceful cancellation and evidence finalization.

## Artifacts and Notes

The initial incident identity is:

    Posttrain run: 48703e11-c591-4762-82a5-2162a7b446de
    dstack run:    pt-f6fefdbb9eacd0f055077034
    Provider cap:  3600 seconds
    Terminal fact: max_duration_exceeded
    Retained work: optimizer steps 1 through 4

The original invocation selected dstack and the remote builder but omitted an explicit provider timeout. Preserve only the redacted shape of the command in durable notes; do not capture credentials or environment contents.

At release completion, append a compact evidence block containing:

    Posttrain source SHA:
    Candidate workflow run:
    Candidate wheel checksums:
    Candidate OCI manifest digest:
    Canary Posttrain/dstack run IDs:
    Merged source SHA:
    Stable workflow run:
    Stable release URL/tag:
    Corrected experiment Posttrain/dstack run IDs:
    Corrected job image digest:
    Corrected provider max_duration:

## Interfaces and Dependencies

`packages/project/src/posttrain/project/execution_settings.py::ResolvedExecutionSettings` remains the provenance authority. Do not add a boolean such as `timeout_was_explicit`; determine explicitness from `sources["timeout_seconds"] != "job"` so precedence and diagnostics stay auditable.

`apps/cli/src/posttrain_cli/execution_planning.py` must end with a private validation seam equivalent to:

    def validate_remote_training_timeout(
        *, job_kind: str, settings: ResolvedExecutionSettings
    ) -> None:
        ...

The exact private name may follow local conventions, but it must be called before packing/building and be directly testable through `plan_job_launch`.

`packages/execution/src/posttrain/execution/contracts.py` must export a log-stream type equivalent to:

    type ExecutionLogStream = Literal["workload", "diagnostic"]

Every `ExecutionProvider.logs` implementation must accept:

    def logs(
        self,
        handle: ExecutionHandle,
        cursor: LogCursor | None = None,
        *,
        limit: int = 200,
        stream: ExecutionLogStream = "workload",
    ) -> LogPage:
        ...

If repository style orders `stream` before `limit`, use one consistent order across the protocol, service, adapters, fakes, and CLI. Do not encode stream identity into `LogCursor`; offsets are scoped to the explicitly selected stream for a single command invocation.

`packages/execution/src/posttrain/execution/service.py::ExecutionSubmission` must carry `execution_policy: ExecutionPolicy | None`. Current-schema saves require a non-null policy created from `plan.request.policy`; historical reads may return `None`. Its identity must include timeout, attempts, and priority. Update `_SCHEMA`, `_SUPPORTED_SCHEMAS`, `_submission_from_payload`, `save`, and relevant fixtures together.

`packages/execution-dstack/src/posttrain_execution_dstack/sdk_bridge.py` remains the only code importing and invoking the dstack SDK. The generic execution package and CLI must not import dstack. Use the SDK's existing diagnostic log selector; do not patch or release the dstack fork unless current installed API inspection disproves that capability.

The hotfix uses existing pinned dependencies and Python 3.13 validation. Any lockfile change must be explained by a deliberate package-version change, not by an unconstrained refresh. Run `uv lock --check` or the repository-equivalent locked sync before release.

Revision note (2026-08-15): Created the initial self-contained hotfix plan after narrowing the earlier logging proposal. Centralized Doris/raw-log work was removed from active scope because a LAN sink is the wrong default for a future multi-cloud fleet; only existing bounded dstack diagnostic access remains in scope.

Revision note (2026-08-15): Implemented and locally validated the timeout and diagnostic-access changes. The release is not yet published; the remaining gate is a staged clean-consumer validation followed by the retained-candidate workflow.
