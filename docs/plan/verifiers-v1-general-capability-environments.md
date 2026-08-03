# Build a separately packaged Verifiers v1 environment monorepo and add four capability packs

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `docs/templates/PLAN.md`. It is self-contained so that a contributor can resume the work from this file and the named repositories without relying on chat history.

## Purpose / Big Picture

After this work, CarbonTeq has six separately installable Verifiers environment packages in one framework-neutral `carbonteq-ai/verifiers-environments` repository. Any Verifiers-compatible host can install one package without importing the other five or the posttrain framework. A developer can select one posttrain catalog evaluation and measure four materially different model capabilities: broad knowledge and reasoning with MMLU-Pro, verifiable instruction adherence with IFEval, procedurally generated reasoning with Reasoning Gym, and tool-assisted competition mathematics with Math Python. The same Reasoning Gym and Math Python packages can also supply tasks and rewards to online training. The visible proof is that `posttrain catalog show evaluation general-capability-balanced-v1` resolves four immutable Verifiers v1 bindings, `posttrain job pack` fetches the shared repository once but builds and locks each selected subdirectory as its own wheel, and a six-run Lab qualification suite records native Verifiers traces for GSM8K, AutomationBench, and the four new packs with task-source revisions and environment-owned reward metrics.

This is not a mechanical port of four legacy evaluators. Each package must use the Verifiers v1 `Taskset`, typed `TaskData`, `Task`, `Harness`, runtime, `@metric`, and `@reward` boundaries. The environment packages own task construction, interaction, parsing, and scoring. The posttrain framework owns selection, budgets, immutable packaging, execution, and evidence presentation.

The first four new capability environments are MMLU-Pro, IFEval, Reasoning Gym, and Math Python. This plan also normalizes the lifecycle of two environments already used by the framework: it rearchitects the upstream GSM8K package as a CarbonTeq-owned, dataset-pinned v1 package, and migrates the existing CarbonTeq AutomationBench adapter out of `/home/hammad/projects/rl/environments/automationbench_v1`. That makes six independently buildable, installable, versioned Python distributions under one Git lifecycle. There is no umbrella runtime distribution: a consumer names only the wheel it needs. Coding breadth beyond the sandboxed Python tool in Math Python is deliberately outside this plan; a later pack should add a serious repository-level coding benchmark rather than treating the current three-task Code Golf smoke as a coding baseline.

## Progress

- [x] (2026-08-02 21:24Z) Read the canonical environment, data, framework, API, and lineage contracts; inspected the current Verifiers pin, v1 tasksets, catalog entries, evaluation adapter, and execution-pack environment locks.
- [x] (2026-08-02 21:24Z) Verified the current Hugging Face repositories, split sizes, licenses, and immutable revisions for MMLU-Pro, IFEval, and the candidate MATH sources; verified that Reasoning Gym is procedural rather than naturally Hugging Face-backed.
- [x] (2026-08-02 21:24Z) Inspected the removed upstream Math Python and Reasoning Gym environments and the current Verifiers v1 component interfaces; recorded why neither legacy implementation should be pinned as the production package.
- [x] (2026-08-02 21:24Z) Chose the framework-versus-environment-repository ownership, source-lock, package, catalog, qualification, and release design captured below.
- [x] (2026-08-02 21:41Z) Revised the ownership decision after comparing the in-repository AutomationBench adapter, the upstream GSM8K package, root-workspace membership, and the canonical published-environment boundary. The dedicated repository is now framework-neutral `carbonteq-ai/verifiers-environments`, and the plan includes both existing-environment migrations.
- [x] (2026-08-03) Made the plan self-contained around exact AutomationBench provenance, the environment-owned Hugging Face data boundary, six package contracts, safe migration order, and six concrete one-cell Lab qualification manifests.
- [x] (2026-08-03) Confirmed the monorepo design against the actual job planner, Git source packer, and wheel builder: one repository revision is fetched once, each package subdirectory produces an independently verified wheel and lock, and the combined runtime dependency closure is resolved after wheel creation.
- [x] (2026-08-03) User accepted the monorepo with strict package isolation as the implementation direction. Repository layout is no longer an open design choice for this plan.
- [x] (2026-08-03) Created the public repository `carbonteq-ai/verifiers-environments`, cloned it to `/home/hammad/projects/verifiers-environments`, and published the first environment-library commit over SSH.
- [x] (2026-08-03) Completed Milestone 1 locally: six isolated uv projects and locks, synthetic Verifiers v1 contract tasks, package-boundary enforcement, matrix CI, six wheels, and a clean combined-wheel activation test.
- [x] (2026-08-03) Started the next implementation wave in parallel across the standalone package directories. Agents own GSM8K, AutomationBench, and MMLU-Pro/IFEval; the coordinating agent owns Reasoning Gym and Math Python. No agent may commit, push, edit framework consumers, or delete the old AutomationBench source during this wave.
- [x] (2026-08-03) Implemented GSM8K locally against the exact Hub revision, including source-row digests, default split-count checks, runtime math verification, and five passing package tests. Framework pins remain unchanged.
- [x] (2026-08-03) Migrated AutomationBench locally into the external repository with native task/tool/state/scoring behavior, provenance, parity coverage, and ten passing package tests. The original `/home/hammad/projects/rl/environments/automationbench_v1` copy remains intact.
- [x] (2026-08-03) Implemented the first real Reasoning Gym and Math Python package contracts locally. Reasoning Gym uses the pinned generator registry and balanced deterministic rows; Math Python uses the pinned MATH Hub source, boxed-answer verification, and a bounded task-scoped child-interpreter tool. Their package validation passed; live qualification later completed, while sandbox image publication and lifecycle gates remain pending.
- [x] (2026-08-03 implementation slice) Reimplemented `gsm8k-v1` with an immutable Hugging Face dataset revision and local parity coverage against the currently pinned upstream behavior. Full framework repin and live qualification were pending at this slice checkpoint; both later completed.
- [x] (2026-08-03 implementation slice) Migrated `automationbench-v1` without changing its task, tool, or reward semantics. Full framework repin and live qualification were pending at this slice checkpoint; both later completed.
- [x] (2026-08-03 implementation slice) Implemented `mmlu-pro-v1` and `ifeval-v1` with exact Hub revisions, deterministic task metadata, source digests, reference-derived parsing/checkers, and successful enabled network gates (MMLU-Pro 70 validation/12,032 test rows; IFEval 541 rows and all 25 instruction IDs). Full framework repin and live qualification were pending at this slice checkpoint; both later completed.
- [x] (2026-08-03) Confirmed and tightened subset evaluation semantics. `EvaluationBudget.num_tasks` already gives an invocation-scoped count without mutating a reusable environment binding; Verifiers v1 consumes it through `Taskset.select(num_tasks, shuffle)`. Added an invocation-level `EvaluationBudget.shuffle` override, preserved the direct-request `shuffle` field for compatibility, forwarded the resolved policy through the native adapter, and recorded `task_selection` (`head` or `verifiers-fixed-shuffle`) in evaluation evidence. Package eval tests pass (28 passed, 2 skipped), and the standard jobs path passes its budget-forwarding test (11 passed).
- [x] (2026-08-03) Created and published environment-repository commit `2ac96d33a4a67ce3c930543f990783505f0e7a2c` (`feat: add CarbonTeq Verifiers environment packs`) after all six package validations, boundary checks, and clean combined-wheel activation passed. The framework can now repin only to this immutable remote SHA.
- [x] (2026-08-03) Published follow-up environment commit `017ac72f543f79f48400cbb4cb641d6df4c3adfa` with deterministic Math Python type-balanced ordering, a scientific-stack child interpreter, and the immutable sandbox `Containerfile`; all selected framework bindings now share this newer SHA.
- [x] (2026-08-03) Built and published the Math Python sandbox in the CarbonTeq OCI registry as `registry.lan/carbonteq/math-python-v1@sha256:67624f5e71f8a5c89d25bc6c42370eb6e71b8569788aa818e5d3fe8585f15f15`; verified non-root scientific-stack imports and absence of token/secret-like environment variables. Added package-level success/error/timeout/process-exit and safe-environment tests, plus an explicit Docker cancellation/timeout cleanup probe.
- [x] (2026-08-03) Repointed framework GSM8K and AutomationBench program/catalog sources to the published environment repository, added immutable Hub fields to GSM8K activations, added the four balanced capability bindings plus Reasoning Gym/Math Python train bindings, and added `general-capability-balanced-v1`.
- [x] (2026-08-03) Added the six explicit Lab qualification bindings and one-cell work-package manifests. Catalog and eval API tests pass with the new source identity; live endpoint qualification was pending at this checkpoint and later completed.
- [x] (2026-08-03) Published environment-library commit `017ac72f543f79f48400cbb4cb641d6df4c3adfa` as the single source revision for all six framework bindings. AutomationBench and Reasoning Gym now vendor their pinned runtime sources inside their own wheels, so the combined job dependency closure contains no nested VCS requirement.
- [x] (2026-08-03) Repointed `packages/eval`, the base and Lab catalogs, release constraints, and `uv.lock` to the published environment revision. The catalog validates with 55 base entries and 49 project entries; the four-environment balanced evaluation plan resolves its immutable bindings.
- [x] (2026-08-03) Proved the six-wheel dependency closure with `uv pip compile --generate-hashes`; the result contains six local wheel blocks and no `git+` or VCS transitive dependency lines. The runtime workspace lock now uses `tabulate==0.9.0`, required by the vendored Reasoning Gym closure, with lock digest `f68d5419b00b0fde278d0b1d55057eb65bb5c9bbdcc78d2ff5cbfab0a3e1fa2c`.
- [x] (2026-08-03) Added `packages/environment` to the framework source allowlist used by job packing. Without it, a real GSM8K pack failed inside the image with `ModuleNotFoundError: posttrain.environment`; the source closure now includes the catalog activation contract required by all six environments.
- [x] (2026-08-03) Added a fail-closed regression for `--build-missing`: a successful rebuild cannot be treated as verified while the configured digest still names a stale image. A local GSM8K pack was exercised against the stale published eval image and is retained only as diagnostic evidence, not as qualification evidence.
- [x] (2026-08-03) Updated an ignored Lab execution override to the verified rebuilt eval digest and packed all six qualification manifests to local OCI layouts. Every package installed the six environment wheels, retained the shared source revision and activation closure, and completed the deferred qualification smoke; no provider-backed run has been claimed.
- [x] (2026-08-03) Attempted the first provider-backed GSM8K run with `--provider local`; admission stopped before pack/submit because the machine-level registry binding still resolved the stale published eval digest. No container or provider submission was created.
- [x] (2026-08-03) Re-ran GSM8K through a temporary explicit project execution overlay that selected the verified eval parent `registry.lan/carbonteq/posttrain-kind-eval@sha256:13ea50188e3064ac7ee38079d85367e11f20a0db627bf6f69d3177b1a73c1b28` while retaining machine-local provider and protected service bindings. Run `envlib-gsm8k-20260803-live` completed on local Docker with provider run `pt-41ca6ad6cb1723a15c456443`, job image `registry.lan/carbonteq/posttrain@sha256:69defb79c2533c6ea1fb41955b293c6a8a92fd8f1261f68e6fb499dfee26d994`, Trackio finalization, and a consistent reconciliation record retaining the required evaluation artifact.
- [x] (2026-08-03) Qualified `automationbench-v1` through the provider-backed managed path; run `envlib-automationbench-20260803-live` reconciled consistently with provider/Trackio success and the required evaluation artifact retained.
- [x] (2026-08-03) Qualified `mmlu-pro-v1` against the pinned reference corpus and framework trace gate; run `envlib-mmlu-pro-20260803-live` reconciled consistently with provider/Trackio success and the required evaluation artifact retained.
- [x] (2026-08-03) Qualified `ifeval-v1` against the pinned Google checker corpus and framework trace gate; run `envlib-ifeval-20260803-live` reconciled consistently with provider/Trackio success and the required evaluation artifact retained.
- [x] (2026-08-03) Qualified `reasoning-gym-v1` across the declared balanced plan; run `envlib-reasoning-gym-20260803-live` reconciled consistently with provider/Trackio success and the required evaluation artifact retained.
- [x] (2026-08-03) Qualified `math-python-v1` through the managed subprocess/tool path; run `envlib-math-python-20260803-live` reconciled consistently with provider/Trackio success and the required evaluation artifact retained. The CarbonTeq OCI image is now published by digest; the expanded lifecycle cleanup matrix remains a release gate.
- [x] Push the environment repository commits, record the qualified SHA under `Artifacts and Notes`, and use that same SHA in every selected framework source pin.
- [x] Repoint every GSM8K and AutomationBench source/dependency pin, add the four balanced catalog bindings, and add the six-environment Lab qualification overlay.
- [ ] Remove the legacy `/home/hammad/projects/rl/environments/automationbench_v1` package as the final migration step, after publication/configuration, full validation, and the active-reference audit pass. The root `environments/` directory should disappear from the framework repository; generic project-owned `environments/<project-path>` support and external `verifiers-environments/environments/<package>` paths remain valid and are not removed.
- [x] Prove wheel packing against the rebuilt eval kind digest, clean-cache Hugging Face loading, all six local evaluations, native trace persistence, and exact source identity. The remaining delivery gates are normal publication/configuration, Math Python lifecycle coverage, full validation, and final legacy-source cleanup.
- [ ] Run both repositories' full validation ladders and update `Outcomes & Retrospective` with measured results and remaining risks.
- [ ] Complete Milestone 9: update active documentation and compatibility instructions, remove the legacy in-repository package, and prove that no active consumer still depends on the framework-owned environment implementation.

## Surprises & Discoveries

- Observation: the framework already has the important portable boundary needed for this work. `VerifiersV1ConfigActivation` stores inert JSON configuration, and the execution pack builds exact Git-sourced environment wheels plus an `EnvironmentActivationLock`; no generic environment implementation belongs in `packages/eval`.
  Evidence: `packages/environment/src/posttrain/environment/requests.py`, `packages/execution/src/posttrain/execution/job_package.py`, and `packages/execution-pack/src/posttrain/execution_pack/planning.py` support declarative v1 activations, full Git revisions, multiple environment wheels from one repository, and content-addressed activation locks.

- Observation: Hugging Face task rows used by an environment are not a `DatasetSelection`. A supervised dataset is a seat for SFT or DPO; an environment owns the tasks used by eval and environment-backed online RL.
  Evidence: `docs/post-training/02-primitives.md` and `docs/post-training/05-apis.md` state that environment-backed GRPO binds `EnvironmentBinding` and has no parallel dataset seat.

- Observation: upstream Verifiers removed the legacy `math_python` and `reasoning_gym_env` directories on 2026-07-17. The last Math Python v1 experiment used the older config stack and loaded an unpinned `chiayewken/competition_math` mirror; the Reasoning Gym package was a thin legacy wrapper around `ReasoningGymEnv`.
  Evidence: upstream commits `0aeaee30fe31584f15d05586eb9a808cf5e7e223` and `8ccd8d6ab2207920743df9cdb4533df8839dcefe` show the removal and the prior implementations. They are research input, not release candidates.

- Observation: MMLU-Pro's current Hugging Face dataset is not static under the name `main`; it has received answer and formatting corrections. Its 70-row validation split supplies five demonstrations for each of 14 categories, and its test split has 12,032 rows.
  Evidence: Hugging Face revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` declares those splits and records a 2026 leading-space correction. A moving branch would make scores incomparable.

- Observation: the official MMLU-Pro evaluator substitutes a seeded random choice when answer parsing fails. That mixes answer quality with a fallback lottery and obscures format failures.
  Evidence: `evaluate_from_local.py` at MMLU-Pro commit `f418b116db00b065c2aea046518d8fcf74d39872` calls `random.randint` for an absent prediction. This plan instead makes invalid extraction a zero and exposes parse success separately.

- Observation: IFEval's Hugging Face repository contains only the 541 benchmark prompts. The executable meaning lives in Google Research's instruction registry and strict/loose checker code, so pinning only the dataset does not pin the benchmark.
  Evidence: Hugging Face revision `966cd89545d6b6acfd7638bc708b98261ca58e84` contains `ifeval_input_data.jsonl`; Google Research commit `e6890f85757dd84e27ca6df2dd30651dafad28e0` contains the reference evaluation implementation.

- Observation: Reasoning Gym is designed to generate tasks at runtime and score them with each generator's native `score_answer`. Hugging Face export is optional convenience, not the semantic authority.
  Evidence: Reasoning Gym commit `49b07130b3fcd12f2d064bba7c43869543a0e7e7` documents more than 100 procedural generators, deterministic seeds, composite datasets, native scoring, and an optional export script.

- Observation: `DigitalLearningGmbH/MATH-lighteval` is a better explicit source than the mirror used by old Math Python. It provides the original 7,500 train and 5,000 test rows, a documented MIT license, standard fields, and a current full Git revision.
  Evidence: Hugging Face revision `0530c78699ea5e8eb5530600900e1f328b48acad` declares the default train/test splits and schema. The old mirror's current card has no license field.

- Observation: the pinned Verifiers Docker runtime starts containers with host networking. It limits CPU and memory but does not provide a network-isolated untrusted-code boundary.
  Evidence: `verifiers/v1/runtimes/docker.py` at Verifiers commit `284a868d6a9022109b749710672a0460e8a996d4` passes `--network host`. Math Python therefore requires an explicit threat-model note and must never receive framework, tracking, provider, or Hugging Face credentials in its tool container.

- Observation: `/home/hammad/projects/ambient-agent-environments` is an uncommitted, purpose-specific checkout. Reusing it would mix unrelated KG-memory ownership with general benchmarks and would not provide an immutable dependency.
  Evidence: its Git repository has no commits and its package metadata names `kg-memory-v1`.

- Observation: placing a package under `/home/hammad/projects/rl/environments/` separates its Python dependency lock but not its Git or publication lifecycle. The root uv workspace includes only `apps/*` and `packages/*`, while `environments/automationbench_v1` has its own `uv.lock`; nevertheless, its immutable source identity is still a commit of the entire posttrain repository and its own README requires a publish-then-repin sequence.
  Evidence: root `pyproject.toml`, `environments/automationbench_v1/pyproject.toml`, `environments/automationbench_v1/uv.lock`, and `environments/automationbench_v1/README.md` show the split dependency environment and shared Git lifecycle.

- Observation: the current GSM8K package is already native Verifiers v1, but its task source is not reproducible enough for the new library policy. `GSM8KConfig` contains only `split`, and `GSM8KTaskset.load()` calls `load_dataset("openai/gsm8k", "main", split=...)` without a Hub revision or source identity in `TaskData`.
  Evidence: `environments/gsm8k_v1/gsm8k_v1/taskset.py` at Verifiers commit `284a868d6a9022109b749710672a0460e8a996d4`; the current `openai/gsm8k` Hub commit is `740312add88f781978c0658806c59bc2815b9866`, with MIT-licensed `main` train and test splits of 7,473 and 1,319 rows.

- Observation: AutomationBench is the existing exception to the external-ownership rule. The CarbonTeq adapter lives at `environments/automationbench_v1`, the base catalog pins that subdirectory from a posttrain commit, and `AGENTS.md` simultaneously says published environment ownership must remain outside this workspace.
  Evidence: `packages/catalog/src/posttrain/catalog/base/environments.yaml`, `docs/tooling/automationbench/README.md`, and `AGENTS.md` lines 98-101. Moving this package closes the policy inconsistency without changing framework contracts.

- Observation: the current `eval.general` job contract executes one environment cell. A work package has one global `environment` seat, and `EvaluateRequest` carries one `environment_id`; a multi-environment `EvaluationPlan` defines the comparable set but does not make one job iterate over it.
  Evidence: `packages/work/src/posttrain/work/contracts.py`, `packages/jobs/src/posttrain/jobs/definitions.py`, `packages/eval/src/posttrain/eval/requests.py`, and the existing `apps/lab/.posttrain/work_packages/qwen2b_multi_environment_eval_qualification.yaml`. The library gate therefore needs six explicit work packages, not one ambiguous multi-cell job.

- Observation: the desired monorepo packaging capability already exists in posttrain. Planning groups subdirectories by `(repository, revision)`, source materialization fetches that pair once, and wheel building verifies and locks every selected subdirectory independently. The planner deliberately rejects two revisions of the same repository in one job.
  Evidence: `_environment_source_requests` in `packages/execution-pack/src/posttrain/execution_pack/planning.py`, `ImmutableGitSourcePacker` in `packages/execution-buildkit/src/posttrain_execution_buildkit/git_sources.py`, `ImmutableEnvironmentWheelBuilder` in `environment_wheels.py`, `test_materializes_deduplicated_sources_and_emits_deterministic_lock`, and `test_rejects_conflicting_revisions_of_one_repository`.

- Observation: Hatchling rejects immutable Git dependency references in wheel metadata unless each standalone project opts into direct references.
  Evidence: the first wheel build failed at metadata preparation; adding `[tool.hatch.metadata] allow-direct-references = true` to each package fixed the build without weakening the exact Verifiers commit pin.

- Observation: the ambient system `python3` is older than the repository contract and does not provide `tomllib`, while `python3.12` is not directly on `PATH`. Root governance checks therefore use `uv run --python 3.12 python scripts/check_boundaries.py`; this selects the required interpreter without creating a root uv project or lock.

- Observation: each independent lock currently resolves 115 packages because the exact Verifiers source brings its complete runtime closure. Installing the six built wheels together resolved one compatible 110-package environment and activated every taskset successfully, so the isolated locks do not currently hide a cross-package dependency conflict.

- Observation: real benchmark packages need to declare their data/runtime dependencies directly even when Verifiers currently brings some of them transitively. GSM8K and Math Python therefore declare `datasets` explicitly; the boundary check now validates the exact Verifiers pin and forbidden framework roots without assuming every package has the same dependency list.

- Observation: the first AutomationBench migration used the private CarbonTeq wheel, but that would make a public environment monorepo dependent on an undisclosed registry. The package now consumes the public CarbonTeq AutomationBench fork at immutable merge commit `908db2abd4a868acc37ab0850474bff653bea25c`; the wheel remains the fork's release artifact, while the Git source makes clean public builds reproducible.

- Observation: the pinned Reasoning Gym repository has a large optional dependency closure because its registry imports all generator families at package import time. This is an intentional package-owned dependency boundary, but wheel size and lock churn must be measured before release.

- Observation: Math Python can prove task loading and child-interpreter state isolation without Docker, but its release contract still requires an immutable sandbox image and success/error/timeout/cancellation cleanup qualification. The local implementation is not a claim of production isolation.

- Observation: publishing the Math Python image does not automatically make Posttrain use it. Verifiers' Docker runtime is selected through `harness.runtime`, then invokes the Docker CLI from inside the evaluation process. The Posttrain eval image currently has no Docker CLI/socket mount, so changing the catalog activation to `runtime.type=docker` would be an unqualified nested-container claim.
  Evidence: Verifiers commit `284a868d6a9022109b749710672a0460e8a996d4` implements `DockerRuntime.start()` with `docker run`; the Posttrain eval image derives from `packages/runtime-images/.../posttrain-base/Dockerfile`, which installs no Docker CLI, and the execution provider mounts only the run workspace and optional trust bundle. The live Math Python cell therefore remains explicitly qualified through the package's bounded subprocess/tool path.

- Observation: the published image lifecycle is clean when the owner performs explicit cleanup, but a client killed before cleanup can leave a running container. A timeout probe that killed the outer Docker client left one named test container; it was removed by exact name, and the rerun using the runtime-style `docker rm --force` path left no container. This is why lifecycle cleanup remains a release gate rather than being inferred from `--rm`.
  Evidence: image digest `registry.lan/carbonteq/math-python-v1@sha256:67624f5e71f8a5c89d25bc6c42370eb6e71b8569788aa818e5d3fe8585f15f15`; success returned `4`, error returned `17`, process exit returned `23`, cancellation removed the container, and managed timeout cleanup removed the running container.

- Observation: the six packages could not be compiled into one actual-job dependency lock while AutomationBench and Reasoning Gym retained nested Git dependencies. Vendoring the pinned CarbonTeq AutomationBench fork and the pinned Reasoning Gym source inside their package wheels removed that closure hazard without coupling the six import namespaces.
  Evidence: the final combined `uv pip compile --generate-hashes` completed with return code 0, emitted six local wheel blocks, and contained no `git+` or VCS URL lines; external repository commit `017ac72f543f79f48400cbb4cb641d6df4c3adfa` contains both vendored source trees.

- Observation: The Math Python image is now published in the CarbonTeq OCI registry after the approximately 1 GB dependency layer completed successfully. The immutable reference is `registry.lan/carbonteq/math-python-v1@sha256:67624f5e71f8a5c89d25bc6c42370eb6e71b8569788aa818e5d3fe8585f15f15`; lifecycle cleanup remains separate from publication. The catalog continues to use the subprocess qualification path and does not treat image publication alone as a sandbox-isolation claim.

- Observation: changing the shipped runtime lock makes every published job-kind image stale until it is rebuilt and its digest is configured. The previous `--build-missing` path rebuilt a new image but ignored the returned digest and continued packing the old configured parent.
  Evidence: the diagnostic GSM8K pack used `registry.lan/carbonteq/posttrain-kind-eval@sha256:360810f695108982f077603d4248db232cf0e273213def0ccd65e1fb44fb3094`, whose OCI label still carried lock `df1b2b0b2b04b25bacdbdac9fedc9d908abd8b77ff2ae0945b7fbda09cde725e`, while the rebuild receipt recorded image `...@sha256:13ea50188e3064ac7ee38079d85367e11f20a0db627bf6f69d3177b1a73c1b28` and lock `f68d5419b00b0fde278d0b1d55057eb65bb5c9bbdcc78d2ff5cbfab0a3e1fa2c`. The CLI now raises instead of packing that mismatch.

- Observation: the first live GSM8K run exposed an execution-configuration precedence gap: the machine-wide config intentionally wins over the ignored project execution overlay, so a verified local parent digest could not be selected through the normal CLI while the published manifest remained stale. A temporary explicit overlay retained the machine provider and credentials, selected the verified parent, and completed the run without changing machine state. The durable reconciliation record is sufficient release evidence even though a host-side Observatory lookup cannot resolve `trackio.lan` from this shell.
  Evidence: run `envlib-gsm8k-20260803-live` reconciled as `state=consistent`, `outcome=succeeded`, provider exit code `0`, tracking status `succeeded`, and retained evaluation artifact digest `54005120c95d1b875fe0dbe2436c3ba6d6b67b85e48e416002414f6657c582b1`; provider run `pt-41ca6ad6cb1723a15c456443` used job image digest `69defb79c2533c6ea1fb41955b293c6a8a92fd8f1261f68e6fb499dfee26d994`.

- Observation: the framework repository now contains only one concrete Verifiers environment implementation, the legacy `environments/automationbench_v1` package. Removing it is a final migration operation, not a request to remove the generic project-path environment capability from the packer or its fixtures.
  Evidence: `git ls-files environments` lists only the AutomationBench package, while execution-pack, buildkit, and job-package tests use `environments/<name>` as generic project-owned fixture paths and the external repository intentionally uses `environments/<package>` as its monorepo subdirectory convention.

- Observation: Verifiers v1 supports two generic subset operations, not arbitrary task-ID selection: `Taskset.select(num_tasks)` takes the deterministic head of a finite taskset, while `Taskset.select(num_tasks, shuffle=True)` uses the shared fixed seed (`SEED = 0`) before slicing. Infinite tasksets cannot be shuffled and must be bounded.
  Evidence: the pinned Verifiers v1 `verifiers/v1/taskset.py`, `verifiers/v1/cli/eval/runner.py`, and `verifiers/v1/utils/sampling.py` implement and exercise this contract. A local five-task probe selected `[0, 1]` for the head and the repeatable `[2, 1]` for the shuffled subset.

- Observation: the posttrain eval package already had count-based invocation budgets and a request-level shuffle flag, but standard job definitions could not express the full subset policy through their `EvaluationBudget`, and evidence did not distinguish a head subset from a shuffled subset.
  Evidence: `packages/eval/src/posttrain/eval/requests.py`, `packages/eval/src/posttrain/eval/backends/verifiers/adapter.py`, `packages/jobs/src/posttrain/jobs/definitions.py`, and `packages/eval/tests/test_api.py`. The new budget override and `task_selection` evidence close that DX gap without adding environment-specific logic.

## Decision Log

- Decision: implement and maintain CarbonTeq-owned generic environments in one framework-neutral repository, `https://github.com/carbonteq-ai/verifiers-environments`, checked out as `/home/hammad/projects/verifiers-environments`, while keeping each environment a standalone distribution under its own subdirectory.
  Rationale: the packages are pure Verifiers work and must be usable through Verifiers or Prime without posttrain. The current packer already groups sources by repository and revision, fetches one Git tree once, then builds and locks each requested subdirectory as a separate wheel. The monorepo reduces repository and CI overhead without creating an umbrella install. A folder under `rl/environments/` would still couple every environment release to the framework itself. The proposed repository did not exist when this plan was written, so creation is an explicit first milestone rather than an assumed fact.
  Date/Author: 2026-08-03 / Codex.

- Decision: create six independent distributions and uv projects in the monorepo: `gsm8k-v1`, `automationbench-v1`, `mmlu-pro-v1`, `ifeval-v1`, `reasoning-gym-v1`, and `math-python-v1`. Do not publish an umbrella `verifiers-environments` distribution and do not allow imports between the six packages.
  Rationale: separate wheels, `pyproject.toml` files, lockfiles, tests, READMEs, and versions preserve install and dependency isolation. GSM8K and AutomationBench normalize the two existing ownership paths; the other four expand capability coverage. Source colocation is an operational convenience, not a runtime dependency.
  Date/Author: 2026-08-03 / Codex.

- Decision: all environments selected into one job package use one shared monorepo commit, while each wheel retains its own version, source-subtree digest, wheel digest, and activation digest.
  Rationale: `_environment_source_requests` rejects multiple revisions of the same repository in one job, and `ImmutableGitSourcePacker` intentionally deduplicates a common repository/revision to one fetch. This is the only material coupling introduced by the monorepo. An individual package may be released and consumed alone at a newer repository commit, but a composed suite must advance every selected source reference to one commit and rerun cross-package dependency qualification.
  Date/Author: 2026-08-03 / Codex.

- Decision: treat the accepted monorepo layout as the implementation baseline for this plan; do not create six repositories or reopen repository topology during implementation unless new code evidence makes the current packaging contract infeasible.
  Rationale: the user accepted the verified design. The remaining work is repository creation, package implementation, migration, publication, and qualification—not another ownership comparison. Any later split remains possible because framework identity already records repository, revision, subdirectory, package, and activation independently, but it requires a deliberate plan revision.
  Date/Author: 2026-08-03 / User and Codex.

- Decision: Milestone 1 task implementations are deliberately synthetic contract probes, not partial benchmark implementations or score evidence.
  Rationale: the scaffold must prove package discovery, typed configuration, task selection, trace scoring, wheel identity, and dependency isolation before benchmark semantics are introduced. Each package README labels this boundary, and later milestones must replace the synthetic taskset behavior rather than treating it as a usable baseline.
  Date/Author: 2026-08-03 / Codex.

- Decision: reimplement GSM8K in the dedicated library instead of copying its upstream source verbatim or waiting on an upstream change.
  Rationale: the current behavior is small enough to parity-test, but the new contract requires explicit `dataset_repo`, `dataset_revision`, `dataset_config`, source row identity, and row digest. Keeping the distribution and taskset ID `gsm8k-v1` preserves callers while a package version bump and source revision make the ownership change explicit. The existing upstream package remains the comparison oracle until migration completes.
  Date/Author: 2026-08-02 / Codex.

- Decision: migrate `automationbench-v1` to the dedicated repository in the same program, but preserve its current task, tool, state, scoring, dependency, and trace behavior.
  Rationale: CarbonTeq already owns this adapter, and leaving it under `rl/environments/` would preserve the lifecycle inconsistency this plan is meant to remove. Treating the move as a source-only migration, protected by its existing tests and live gate, avoids mixing a behavioral rewrite into a repository move.
  Date/Author: 2026-08-02 / Codex.

- Decision: consume the CarbonTeq AutomationBench fork from its public immutable merge commit `908db2abd4a868acc37ab0850474bff653bea25c` rather than requiring a private package index in the public environment monorepo.
  Rationale: the fork is the maintained Python 3.12 compatibility source, and a Git commit makes clean-clone builds reproducible without credentials. The fork's own release artifact and lifecycle remain tracked in its repository; the environment package pins the source commit and preserves the distribution identity.
  Date/Author: 2026-08-03 / Codex.

- Decision: represent the six-cell Lab release gate as six work-package manifests that share `environment-library-qualification-v1`, the same model, inference, and target, and differ only in the selected `environment` binding.
  Rationale: this matches the current one-cell `EvaluateRequest` and global-seat work-package contract. It makes every plan, package, run, retry, and receipt independently addressable while preserving one comparable evaluation-plan identity. No environment-specific runner or framework branch is needed.
  Date/Author: 2026-08-03 / Codex.

- Decision: target the framework's current exact Verifiers v1 commit, `284a868d6a9022109b749710672a0460e8a996d4`, and do not advance the framework pin merely to add these environments.
  Rationale: this commit already exposes the required typed Taskset/Task, metric/reward, MCP toolset, and subprocess/Docker runtime contracts. Advancing Verifiers would combine an environment-library change with a framework-runtime migration. If implementation proves a missing generic capability, amend this plan with a separate upstream change and qualify the new commit before changing either repository's lock.
  Date/Author: 2026-08-02 / Codex.

- Decision: use Hugging Face directly when it is the distribution authority, but require the repository name, full 40-character revision, configuration, split, stable row identity, and row-content digest in the task data or activation evidence.
  Rationale: the user explicitly permits reliance on Hugging Face, so copying benchmark rows into Git or routing them through framework dataset materialization adds storage and a second authority without improving reproducibility. A full Hub Git commit plus row digest is reviewable and immutable.
  Date/Author: 2026-08-02 / Codex.

- Decision: environment-owned Hugging Face rows do not become posttrain `DatasetSelection` values.
  Rationale: `DatasetSelection` is a public seat for supervised SFT/DPO data and other data jobs. For eval and online RL, the environment's taskset owns row schema, prompt construction, task identity, split meaning, and scoring. Posttrain records the immutable taskset activation and may stage an explicitly declared resource for offline execution, but it does not reinterpret the rows or import concrete environment code.
  Date/Author: 2026-08-02 / Codex.

- Decision: fail environment setup before model inference when a pinned Hub source cannot be downloaded, and make the error name the repository, revision, config, and split.
  Rationale: silently switching to cached, latest, or partial data would create an untrustworthy score. Public Hub access and outbound HTTPS are release assumptions; a warm cache is an optimization only.
  Date/Author: 2026-08-02 / Codex.

- Decision: make MMLU-Pro and IFEval evaluation-only in the first release; make Reasoning Gym and Math Python available for both evaluation and online training.
  Rationale: MMLU-Pro's test set and IFEval's sole 541-row split are held-out benchmarks. Reasoning Gym can generate disjoint seed namespaces, and MATH provides an explicit train/test split with verifiable answers.
  Date/Author: 2026-08-02 / Codex.

- Decision: MMLU-Pro uses the official category-specific five-shot chain-of-thought prompt shape, but an unparseable answer scores zero rather than receiving the reference evaluator's random fallback.
  Rationale: five-shot category prompting preserves the benchmark's intended reasoning setup. A strict zero makes the primary score deterministic and lets `answer_parse_success` reveal format failures. The environment README must label this as `mmlu-pro-cot-5shot-strict-v1`, not claim bit-for-bit leaderboard equivalence.
  Date/Author: 2026-08-02 / Codex.

- Decision: IFEval re-expresses the official deterministic instruction registry as typed v1 scoring functions and exposes strict and loose prompt-level and instruction-level signals; it does not use an LLM judge.
  Rationale: the benchmark is valuable precisely because its instructions are heuristically verifiable. Parity fixtures against the pinned Google implementation protect meaning while a typed registry removes the reference script's file-oriented orchestration.
  Date/Author: 2026-08-02 / Codex.

- Decision: Reasoning Gym's authority is the pinned Python package/source and its native scorer, not a pre-generated Hugging Face snapshot.
  Rationale: procedural generation, difficulty configuration, and scorer behavior are the environment. A static export would hide those controls and make train/eval separation harder to audit. Hugging Face snapshots may be created later for debugging, but cannot replace the generator revision and seed evidence.
  Date/Author: 2026-08-02 / Codex.

- Decision: the initial Reasoning Gym baseline contains ten verified generators across arithmetic, algorithms, logic, graphs, and puzzles: `leg_counting`, `products`, `letter_counting`, `number_sorting`, `knights_knaves`, `syllogism`, `shortest_path`, `graph_color`, `countdown`, and `zebra_puzzles`.
  Rationale: this gives category breadth without pretending that all 100-plus generators have been qualified. Each generator gets an equal task budget in the balanced baseline. New generators are catalog revisions, not silent expansion of v1.
  Date/Author: 2026-08-02 / Codex.

- Decision: Math Python uses `DigitalLearningGmbH/MATH-lighteval` at revision `0530c78699ea5e8eb5530600900e1f328b48acad`, a task-scoped Python MCP tool, strict boxed-answer extraction, and `math-verify` symbolic equivalence.
  Rationale: this separates immutable task loading, interactive tool behavior, and final scoring into native v1 components. The model can inspect intermediate results without letting its arbitrary Python execute in the framework or evaluator process.
  Date/Author: 2026-08-02 / Codex.

- Decision: do not amend the frozen product baseline for this work.
  Rationale: the existing baseline already says a published Verifiers package owns task and reward semantics, while `EnvironmentBinding`, evaluation plans, and execution packs own selection and delivery. The work adds implementations and catalog entries without changing those meanings. If a new framework field becomes necessary, pause and amend the baseline before code.
  Date/Author: 2026-08-02 / Codex.

- Decision: keep semantic dataset/split selection inside each environment's declarative activation, and make the framework's invocation subset policy count-based plus optional Verifiers fixed-seed shuffle.
  Rationale: Verifiers v1 owns task population and exposes `Taskset.select(num_tasks, shuffle)`; the framework should not copy task rows or invent a parallel task-ID registry. `EvaluationBudget` now carries the optional shuffle override so normal job definitions can request a cheap head subset or a reproducible shuffled subset. Arbitrary IDs, category balancing, and train/test meaning remain environment-owned configuration and task evidence.
  Date/Author: 2026-08-03 / Codex.

- Decision: vendor the pinned AutomationBench and Reasoning Gym runtime sources inside their standalone environment wheels rather than allow nested VCS dependencies in a framework job package.
  Rationale: an environment Git revision must be sufficient to reproduce the wheel and the combined actual-job lock. Vendoring keeps each package independently installable, preserves upstream source identity in its README, and makes the six-wheel hash-locked dependency compile deterministic. The vendored trees remain private implementation details; consumers import only `automationbench_v1` or `reasoning_gym_v1`.
  Date/Author: 2026-08-03 / Codex.

- Decision: make `--build-missing` fail closed when the configured digest-pinned kind image remains stale after a rebuild.
  Rationale: the planner embeds `registry.kind_images` in the actual-job manifest before materialization. Silently ignoring a rebuilt digest would create successful but non-reproducible evidence. The safe DX is an actionable error requiring publication or an explicit local digest override; a later change may add a first-class local-parent override, but it must preserve the same digest and label checks.
  Date/Author: 2026-08-03 / Codex.

- Decision: keep Math Python's catalog activation on the bounded subprocess/tool path for this release, while treating the published OCI image as a separately verified sandbox artifact. Do not claim image-backed isolation until a provider-managed sandbox path supplies the image without relying on an unavailable nested Docker socket.
  Rationale: the environment package must stay framework-neutral, and the current Posttrain eval image cannot execute Verifiers' Docker runtime as configured. A catalog-only switch would change the runtime threat model without evidence. The next framework change may add a generic sandbox execution contract, but it must be designed and qualified independently of Math Python semantics.
  Date/Author: 2026-08-03 / Codex.

- Decision: remove the legacy concrete environment package from the framework repository only as the final migration step, while retaining generic project-owned environment paths and external monorepo subdirectories.
  Rationale: the user-facing lifecycle is now owned by `carbonteq-ai/verifiers-environments`; keeping a second implementation under `rl/environments` creates two sources of truth and makes package publication ambiguous. The deletion must not remove the framework's generic `ProjectPathEnvironmentSource` behavior, fixture paths such as `environments/toy_env`, or external Git references such as `environments/gsm8k_v1` and `environments/automationbench_v1`. The gate therefore requires a clean-clone/package/live proof, an active-reference audit, scoped `git rm`, documentation updates, and a full validation pass before the old directory disappears.
  Date/Author: 2026-08-03 / User and Codex.

## Outcomes & Retrospective

Milestone 1 and the six-package implementation wave are complete. The public
repository is published at `017ac72f543f79f48400cbb4cb641d6df4c3adfa`, and the
framework now consumes that exact source for GSM8K, AutomationBench, and the
four new capability packs. AutomationBench and Reasoning Gym vendor their
pinned runtimes, so the six-wheel dependency closure compiles without nested
VCS requirements. The balanced capability plan and six explicit Lab
qualification work packages are present; catalog, eval, and runtime-image
contract tests pass. The framework eval API has a complete invocation subset
DX: count overrides remain non-mutating, the normal budget can request
Verifiers' reproducible fixed-seed shuffle, the adapter forwards the resolved
policy, and evidence records head versus shuffled selection.

All six provider-backed cells are now live-qualified through Posttrain's
generic managed eval path using the rebuilt parent: GSM8K, AutomationBench,
MMLU-Pro, IFEval, Reasoning Gym, and Math Python each have a succeeded local
Docker run, succeeded Trackio finalization, a consistent reconciliation record,
and a retained evaluation artifact. The temporary overlay was needed only
because the machine-wide config still resolves the stale published parent;
normal machine launches still require the rebuilt eval kind digest to be
published/configured. The remaining release gates are that publication/config
step, normal-runtime sandbox integration (or an explicit release decision to keep the subprocess path), full validation, and removal of the old in-repository
AutomationBench copy as the final migration step.
The largest product risks remain Math Python's host-networked untrusted-code
boundary and the large pinned Reasoning Gym dependency closure.

The final migration step is intentionally destructive but scoped: after all
source, package, runtime, and live evidence gates pass, remove the tracked
`environments/automationbench_v1` tree and update active documentation so the
framework has no concrete Verifiers implementation in its own repository.
Generic project-path environment support remains part of the framework, and
the external repository continues to use `environments/<package>` paths.

Update this section after every completed milestone. At final completion, include the environment-repository commit, framework commit, all six package versions and wheel digests, Math Python image digest, Hub revisions and observed row counts, GSM8K parity evidence, AutomationBench migration evidence, live run IDs, trace completeness result, and any benchmark deviations.

## Context and Orientation

The framework repository is `/home/hammad/projects/rl`. Its canonical product meaning is in `docs/post-training/01-workflow.md` through `docs/post-training/06-observation-and-lineage.md`. `packages/environment/src/posttrain/environment/requests.py` defines `EnvironmentSource`, `VerifiersV1ConfigActivation`, and `EnvironmentBinding`. A binding is an inert selection: catalog loading validates JSON and immutable source identity but does not import Verifiers or download task data.

`packages/eval/src/posttrain/eval/backends/verifiers/adapter.py` is the runtime adapter. It activates the selected `verifiers.v1.env.EnvConfig`, constructs the evaluation run, and streams native traces. It must remain generic; it must not branch on GSM8K, AutomationBench, MMLU-Pro, IFEval, Reasoning Gym, or Math Python.

`packages/execution-pack/src/posttrain/execution_pack/planning.py` finds selected environment sources, groups package subdirectories by repository and revision, and creates activation locks. `packages/execution-buildkit/src/posttrain_execution_buildkit/environment_packager.py`, `git_sources.py`, and `environment_wheels.py` fetch the exact Git tree once, build each selected package root independently, and verify that the distribution identity matches `EnvironmentSource.package`. `packages/execution/src/posttrain/execution/job_package.py` stores the shared repository revision plus per-package source-tree, wheel, and activation digests in the actual-job manifest. Existing tests already prove two subdirectory wheels from one checkout and one deduplicated fetch; this plan extends that evidence to six packages and a real clean clone.

The packaged global catalog is under `packages/catalog/src/posttrain/catalog/base/`. `environments.yaml` owns reusable environment bindings and `evaluations.yaml` groups them into an evaluation plan. The Lab composition host has a project overlay at `apps/lab/.posttrain/catalog/algorithm-qualification.yaml`; it is the correct place for tiny release-gate budgets that should not redefine the reusable baseline.

The current framework pins Verifiers commit `284a868d6a9022109b749710672a0460e8a996d4` in `packages/data/pyproject.toml`, `packages/eval/pyproject.toml`, and `uv.lock`. The pin provides `verifiers.v1.Taskset`, `TaskData`, `Task`, `TasksetConfig`, `TaskConfig`, `Harness`, `Toolset`, `State`, `Trace`, `Runtime`, `@metric`, and `@reward`. A taskset loads or generates typed tasks. A task owns per-row behavior and scoring. A harness owns how a model is driven. A toolset is an MCP server exposing tools to the model. A runtime executes the harness or tool server in a subprocess or container.

The current GSM8K distribution is CarbonTeq's `gsm8k-v1` package from `https://github.com/carbonteq-ai/verifiers-environments` at commit `017ac72f543f79f48400cbb4cb641d6df4c3adfa`, subdirectory `environments/gsm8k_v1`, while the generic Verifiers v1 runtime remains pinned to commit `284a868d6a9022109b749710672a0460e8a996d4`. `packages/eval/pyproject.toml`, `uv.lock`, `packages/catalog/src/posttrain/catalog/base/environments.yaml`, `packages/eval/src/posttrain/eval/programs/general_smoke.py`, and `apps/lab/.posttrain/catalog/algorithm-qualification.yaml` all refer to the CarbonTeq package. It retains the upstream package name, taskset ID, prompt, answer extraction, and reward meaning while adding immutable data identity.

The former CarbonTeq AutomationBench adapter was `/home/hammad/projects/rl/environments/automationbench_v1`; the active package is now under `https://github.com/carbonteq-ai/verifiers-environments` at the same shared commit, subdirectory `environments/automationbench_v1`. Its source is vendored into the wheel and its README records the CarbonTeq fork provenance. The old framework copy remains only as a migration artifact until the replacement is packed, live-qualified, and the final active-reference audit passes; no active catalog or dependency should point at the framework-local path.

After Milestone 9, the framework repository must not contain a concrete Verifiers package under its root `environments/` directory. This does not remove generic project-owned environment paths: `ProjectPathEnvironmentSource`, packer/buildkit fixtures such as `environments/toy_env`, and external Git subdirectories such as `environments/gsm8k_v1` remain supported. The distinction is ownership: concrete reusable Verifiers implementations live in the external monorepo, while project-local paths remain an explicitly supported composition feature.

The external repository is `/home/hammad/projects/verifiers-environments`, with public remote `https://github.com/carbonteq-ai/verifiers-environments`. The published `main` branch now has the qualified environment-library commit `017ac72f543f79f48400cbb4cb641d6df4c3adfa`. Milestone 1 created these six standalone Python 3.12 uv projects locally:

| Subdirectory | Distribution | Module | Initial CarbonTeq version |
| --- | --- | --- | --- |
| `environments/gsm8k_v1` | `gsm8k-v1` | `gsm8k_v1` | `0.2.0` |
| `environments/automationbench_v1` | `automationbench-v1` | `automationbench_v1` | `0.2.0` |
| `environments/mmlu_pro_v1` | `mmlu-pro-v1` | `mmlu_pro_v1` | `0.1.0` |
| `environments/ifeval_v1` | `ifeval-v1` | `ifeval_v1` | `0.1.0` |
| `environments/reasoning_gym_v1` | `reasoning-gym-v1` | `reasoning_gym_v1` | `0.1.0` |
| `environments/math_python_v1` | `math-python-v1` | `math_python_v1` | `0.1.0` |

The repository root contains only governance and orchestration files: `README.md`, `LICENSE`, package-boundary checks, and CI configuration. Each subdirectory owns its `pyproject.toml`, `uv.lock`, `README.md`, the table's exact module below `src/`, and `tests/`. It builds exactly one wheel, has no local-path dependency on another environment, and can be synchronized, tested, built, and installed directly from its subdirectory. GSM8K includes `taskset.py`, `verify.py`, parity tests, and a live Hugging Face test. AutomationBench includes its taskset, toolsets, scoring, migrated tests, and parity tests. MMLU-Pro and IFEval contain their taskset/reference-parity tests; IFEval also carries `NOTICE`. Reasoning Gym contains its procedural taskset tests. Math Python additionally owns `images/math-python/Containerfile`, the Python MCP server, Docker lifecycle tests, and live Hugging Face tests.

Do not make the six projects members of a shared uv workspace and do not add a root runtime lock: each package lock must remain independently resolvable. Root CI iterates the six explicit project paths and then runs a cross-package temporary-environment test that installs all six built wheels together with the selected eval and online-RL constraint profiles. Do not create a shared helper distribution or umbrella package in v1. Small source-lock and row-digest helpers may be repeated until their contract is stable enough to justify a separately planned public utility.

The three developer paths stay distinct:

1. A direct Verifiers consumer installs one package from its immutable Git subdirectory, for example `uv add "gsm8k-v1 @ git+https://github.com/carbonteq-ai/verifiers-environments.git@${ENVIRONMENTS_REVISION}#subdirectory=environments/gsm8k_v1"`. That install does not pull the other five wheels or posttrain.
2. A framework catalog binding records package name, repository, full shared commit, and package subdirectory in `EnvironmentSource`; the execution pack builds only the source closure selected by the job.
3. A composed evaluation plan may select several packages. The packer fetches the shared commit once, emits one wheel and source-subtree lock per package, then resolves the selected wheels into one actual-job dependency closure. Because the current planner forbids mixed commits from one repository, updating any package inside a composed plan requires repinning all selected bindings to one qualified monorepo commit even when some package versions and subtree digests are unchanged.

The relevant immutable external sources at plan creation are:

- Verifiers: `https://github.com/PrimeIntellect-ai/verifiers`, commit `284a868d6a9022109b749710672a0460e8a996d4`.
- GSM8K data: `openai/gsm8k`, Hugging Face commit `740312add88f781978c0658806c59bc2815b9866`, configuration `main`, MIT, 7,473 train and 1,319 test rows.
- Current GSM8K reference package: Verifiers commit `284a868d6a9022109b749710672a0460e8a996d4`, subdirectory `environments/gsm8k_v1`.
- AutomationBench adapter input: `/home/hammad/projects/rl/environments/automationbench_v1`, consuming registry distribution `carbonteq-automation-bench==1.0.5.post1`, published from CarbonTeq repository commit `908db2abd4a868acc37ab0850474bff653bea25c`. Its maintained compatibility lineage selected fork commit `d54dbebabdba6c6eda201694aee8ddcf36ccfc51`, based on upstream Zapier commit `a321764ace3cfbe42289e6a13abef2f0f4f56fad`. Preserve that distribution version, registry source, wheel resolution, and lineage in the migrated lock and README.
- MMLU-Pro data: `TIGER-Lab/MMLU-Pro`, Hugging Face commit `b189ec765aa7ed75c8acfea42df31fdae71f97be`, MIT, 70 validation and 12,032 test rows.
- MMLU-Pro reference code: `https://github.com/TIGER-AI-Lab/MMLU-Pro`, commit `f418b116db00b065c2aea046518d8fcf74d39872`.
- IFEval data: `google/IFEval`, Hugging Face commit `966cd89545d6b6acfd7638bc708b98261ca58e84`, Apache-2.0, 541 rows in a split named `train` even though it is a held-out evaluation set.
- IFEval reference code: `https://github.com/google-research/google-research`, commit `e6890f85757dd84e27ca6df2dd30651dafad28e0`, Apache-2.0.
- Reasoning Gym: `https://github.com/open-thought/reasoning-gym`, commit `49b07130b3fcd12f2d064bba7c43869543a0e7e7`, Apache-2.0.
- MATH data: `DigitalLearningGmbH/MATH-lighteval`, Hugging Face commit `0530c78699ea5e8eb5530600900e1f328b48acad`, MIT, 7,500 train and 5,000 test rows in the default configuration.

## Environment Contracts

All six packages must accept only JSON-serializable configuration through `VerifiersV1ConfigActivation`. Every Hugging Face revision field must validate as a full 40-character lowercase Git SHA. Default values are conveniences for direct package use; the framework catalog must still spell out every dataset repository, revision, configuration, split, seed, and environment source revision so the activation digest is self-explanatory.

### GSM8K v1 lifecycle migration

In `environments/gsm8k_v1/src/gsm8k_v1/taskset.py`, define `GSM8KData`, `GSM8KTask`, `GSM8KConfig`, and `GSM8KTaskset`. Preserve the currently selected upstream system prompt, final `####` answer contract, in-runtime `math-verify` verification, `train` and `test` split names, and `correct` reward meaning. Do not import posttrain.

Extend `GSM8KConfig` with `dataset_repo`, `dataset_revision`, `dataset_config`, and `split`. Defaults name `openai/gsm8k`, revision `740312add88f781978c0658806c59bc2815b9866`, configuration `main`, and split `test`. Require a full revision. `GSM8KTaskset.load()` passes all four fields to `datasets.load_dataset`, checks 7,473 train or 1,319 test rows for the default source, and stores source repository, revision, split, stable row index, and normalized row SHA-256 in `GSM8KData`.

Build a parity corpus from synthetic rows plus a deterministic sample of the pinned Hub rows. Run both the old upstream taskset and the new package over identical rows and responses. Prompts, extracted gold answers, gold self-validation, and reward values must match exactly. The new source metadata is additive. Publish the package as version `0.2.0` while retaining distribution and taskset ID `gsm8k-v1`; repository source and activation digest distinguish it from upstream `0.1.0`.

### AutomationBench v1 lifecycle migration

Move the owned source from `/home/hammad/projects/rl/environments/automationbench_v1/src/automationbench_v1` into `environments/automationbench_v1/src/automationbench_v1` in the new repository. Preserve the public `AutomationBenchTaskset`, its task data and state types, Zapier, limited-Zapier and API toolsets, deterministic assertion scoring, reward/metric names, and trace metadata. Preserve `carbonteq-automation-bench==1.0.5.post1`, its published-source provenance commit `908db2abd4a868acc37ab0850474bff653bea25c`, and the exact Verifiers pin in the new lock and README. The new repository must configure the authenticated CarbonTeq package index by URL only; credentials remain runtime configuration and never enter Git.

Copy the existing tests first and make them pass unchanged against the migrated source. Add a wheel-install smoke and a parity test that resolves the same seed/domain pairs through old and new packages and compares stable task IDs, prompts, tool schemas, assertion sets, rewards, metrics, and serialized trace metadata. Publish the migrated package as version `0.2.0` with the same distribution and taskset ID `automationbench-v1`. This is a lifecycle migration; any desired benchmark behavior change is a later version and plan revision.

### MMLU-Pro v1

In `environments/mmlu_pro_v1/src/mmlu_pro_v1/taskset.py`, define `MMLUProData`, `MMLUProTask`, `MMLUProConfig`, and `MMLUProTaskset`.

`MMLUProConfig` has `id`, `dataset_repo`, `dataset_revision`, `validation_split`, `test_split`, `categories`, `shots`, and `order_seed`. Defaults name the pinned values above and `shots=5`. Reject any category outside the 14 categories observed in the pinned test set and reject `shots` larger than the available category demonstrations.

`MMLUProTaskset.load()` performs two exact-revision `datasets.load_dataset` calls, validates row counts and required columns, removes only literal `N/A` options as the reference implementation does, and constructs each test prompt from five validation demonstrations in the same category. A task row stores `question_id`, `category`, answer label, source revision, validation demonstration IDs, and a SHA-256 digest of the normalized source row. Stable ordering is category-balanced and deterministic under `order_seed`; the balanced framework binding selects the first 1,400 rows, exactly 100 per category. A separate full-run config can set category balance off and select all 12,032 rows without changing scoring.

`MMLUProTask` extracts a final answer label using the reference parser's ordered patterns, records `answer_parse_success`, and returns `answer_correct` as 1.0 only when the parsed label equals the gold label. Missing or malformed labels return 0.0. The package README names the prompt/scoring deviation from the official fallback and gives the exact source commits.

### IFEval v1

In `environments/ifeval_v1/src/ifeval_v1/instructions.py`, define a typed registry that maps every instruction ID present in the pinned 541 rows to one checker. Each checker accepts the model response and that instruction's validated keyword arguments and returns a Boolean. Port only the semantic algorithms needed from the Apache-2.0 reference, retain copyright and attribution in `NOTICE`, and normalize the reference's loose-response variants in one named function rather than scattering string rewrites across checkers.

In `environments/ifeval_v1/src/ifeval_v1/taskset.py`, define `IFEvalData`, `IFEvalTask`, `IFEvalConfig`, and `IFEvalTaskset`. `IFEvalConfig` names the pinned Hub repository/revision and the physical split `train`, but also exposes a constant logical purpose `evaluation`; callers cannot configure it as a training environment. Loading fails if an instruction ID has no registered checker, if `instruction_id_list` and `kwargs` lengths differ, or if the source key is duplicated.

For every response, compute `strict_instruction_accuracy`, `loose_instruction_accuracy`, `strict_prompt_accuracy`, and `loose_prompt_accuracy`. The primary `@reward` is `strict_prompt_accuracy`; the other three are `@metric` values. Task data stores the Hub key, instruction IDs, source revision, and normalized row digest. `tests/fixtures/reference_cases.json` contains synthetic prompts/responses and expected Boolean vectors generated by the pinned Google implementation; it must not contain the full benchmark dataset.

### Reasoning Gym v1

In `environments/reasoning_gym_v1/src/reasoning_gym_v1/taskset.py`, define `ReasoningGymData`, `ReasoningGymTask`, `ReasoningGymConfig`, and `ReasoningGymTaskset`. Pin the Reasoning Gym dependency to commit `49b07130b3fcd12f2d064bba7c43869543a0e7e7` in the environment lock; do not accept a floating PyPI range.

`ReasoningGymConfig` has `split`, `generators`, `examples_per_generator`, `train_seed_start`, `eval_seed_start`, and per-generator difficulty dictionaries. `split` is only `train` or `eval`. The v1 defaults use seed namespaces `[0, 999999]` for train and `[1000000, 1999999]` for eval. Reject a requested count that would cross its namespace. Each generated row records the generator name, exact generator configuration, seed, ordinal, expected answer, source metadata, source commit, and row digest. Its stable task ID includes split, generator, and seed.

The taskset cycles deterministically through the ten declared generators so any prefix remains balanced. `ReasoningGymTask` calls the native generator scorer and records `native_score`; its primary reward is the same bounded float. Gold-answer self-validation must score 1.0 before a row is yielded. Tests prove the train and eval IDs are disjoint, repeated construction is byte-stable, and all ten generators can score their gold output.

### Math Python v1

In `environments/math_python_v1/src/math_python_v1/taskset.py`, define `MathPythonData`, `MathPythonTaskConfig`, `MathPythonTask`, `MathPythonConfig`, and `MathPythonTaskset`. The config names the pinned MATH repository/revision, default dataset configuration, `train` or `test` split, deterministic order seed, and Python tool configuration. Loading extracts the final boxed answer from each reference solution, retains problem level and type, and fails closed when the gold answer cannot self-verify.

In `environments/math_python_v1/src/math_python_v1/servers/python.py`, define `PythonState`, `PythonToolsetConfig`, and `PythonToolset`. The state stores a bounded list of accepted code cells and tool errors per rollout. The `python` tool executes a cell in a fresh child interpreter inside the tool container, replays prior accepted cells, captures the last expression and stdout, enforces a per-call wall timeout, and truncates returned output. It must not call `exec` in the evaluator or framework process. The toolset is task-scoped so state cannot leak between rollouts.

Build an immutable image from `environments/math_python_v1/images/math-python/Containerfile` containing Python, uv, NumPy, SymPy, SciPy, the tool server, and no credentials. Publish it once, record the registry digest in this plan, the environment repository, and the framework repository, and use the digest rather than a mutable tag in activation config. Run it with one CPU, 2 GiB memory, no GPU, no host volumes, and the narrowest available runtime permissions. Because the pinned Verifiers Docker transport uses host networking, run live qualification only on an isolated worker and assert that no framework secret is projected into the tool-server environment. Treat stronger network isolation as a follow-up gate, not as an undocumented property.

`MathPythonTask` exposes the Python tool, records `answer_parse_success`, `python_calls`, and `python_errors`, extracts the last strict `\\boxed{...}` answer, and returns `symbolic_correct` from `math-verify`. The environment must support train and test bindings, but the balanced capability baseline uses test only.

## Plan of Work

### Milestone 1: establish the external environment workspace and contract tests

Create the GitHub repository and local checkout named above. Initialize the six standalone uv projects, each with its own `pyproject.toml` and `uv.lock`; do not register them as members of a root workspace. Pin the same Verifiers commit used by the framework and each package's exact external dependencies in that package's lock. Add root CI that runs the six projects as a matrix plus one combined-wheel compatibility job. Add a README explaining the ownership boundary: each package owns generic environment meaning and can run without its siblings or posttrain; the monorepo owns shared governance and a Git revision; `/home/hammad/projects/rl` owns catalog selections, job budgets, immutable delivery, and cross-run evidence.

Before implementing a full benchmark, add one synthetic task to each new package and import the two migrated packages. Prove that `EnvConfig` values using taskset IDs `gsm8k-v1`, `automationbench-v1`, `mmlu-pro-v1`, `ifeval-v1`, `reasoning-gym-v1`, and `math-python-v1`, each with `harness.id=null`, import and execute through the pinned v1 loader. Build each wheel and inspect its metadata. Acceptance is six independently installable wheels whose normalized distribution names exactly match the planned `EnvironmentSource.package` values and whose dependency metadata contains no `posttrain-*`, Trackio, W&B, trainer, or serving package.

Milestone status (historical, 2026-08-03): complete locally before the real implementation wave. The two migration targets were represented only by synthetic package contracts at that point; their real implementations and parity proofs were Milestone 2 work. Each project had an independent lock, contract tests passed, and a clean Python 3.12 environment activated all six tasksets. The repository source was deliberately unpublished at that checkpoint; publication was completed later at the commit recorded below.

### Milestone 2: normalize GSM8K and AutomationBench ownership

Implement the GSM8K contract above against Hub revision `740312add88f781978c0658806c59bc2815b9866`. Run old/new parity before changing any framework pin. Then migrate AutomationBench source and tests into the external repository and run old/new parity from separate virtual environments so identically named distributions do not shadow each other.

At this milestone's end, both replacement wheels build from the new repository and are behaviorally qualified, but `/home/hammad/projects/rl` still points to the old sources. This parallel state is intentional and recoverable. Acceptance is exact GSM8K prompt/reward parity with additive source identity and exact AutomationBench task/tool/reward/trace parity.

### Milestone 3: implement and parity-test MMLU-Pro and IFEval

Implement the two deterministic evaluation-only packages according to the contracts above. Do not vendor their Hub rows. Unit tests use synthetic rows and pure parsing/checker cases; network-marked tests load the pinned Hub commits into a temporary empty `HF_HOME`.

For MMLU-Pro, compare prompt construction and answer extraction against the pinned reference on at least one row from each category. Explicitly test the strict-zero behavior for an unparseable response. For IFEval, run the pinned Google evaluator over the synthetic response corpus once, save only its expected per-instruction Booleans, and assert exact strict/loose parity in the v1 package.

Acceptance is a clean-cache load with exactly 70/12,032 MMLU-Pro rows and 541 IFEval rows, plus model-free scoring traces that contain the pinned Hub revision, stable task ID, row digest, and named metrics.

### Milestone 4: implement procedural Reasoning Gym with disjoint train/eval namespaces

Implement the ten-generator taskset, native scoring, stable task IDs, and configuration validation. Make selection prefix-balanced and record every generator's exact config in task data. Run repeated generation in separate processes and compare serialized task data; compare the first 10,000 train and eval IDs and require an empty intersection.

Acceptance is that all ten gold answers score 1.0, representative wrong answers do not, the same revision/config/seed yields identical tasks, and changing only the split changes every task ID while preserving the intended generator mix.

### Milestone 5: implement Math Python and qualify the sandbox lifecycle

Implement the pinned Hub loader, typed Python MCP tool, per-rollout state, symbolic scorer, and immutable tool image. Unit-test answer extraction and equivalence without Docker. Docker tests launch two concurrent rollouts, prove their Python namespaces are isolated, exercise timeout and output truncation, cancel one rollout, and confirm no container remains. Inspect the tool process environment and prove it contains no tracking, provider, registry, or Hub token.

Acceptance is a clean-cache MATH load with 7,500/5,000 rows; an equivalent symbolic answer scores 1.0; a wrong answer scores 0.0; a model can call Python more than once before answering; and normal, failed, timed-out, and cancelled runs all remove their tool containers.

### Milestone 6: publish the environment source before framework pinning

Run the external repository's full validation, build all wheels, and commit the environment repository. Push that commit to `carbonteq-ai/verifiers-environments`. Record its full commit as `ENVIRONMENTS_REVISION` in this plan. Do not update the framework to a local path, branch name, or unpushed SHA.

Re-clone the published commit into an empty temporary directory, synchronize each package from its own lock, build all six subdirectory wheels, install the six wheels together into a disposable environment, and repeat the model-free and live source smokes. This proves both individual installability and combined dependency compatibility without the author's checkout. Tag the initial packages as `gsm8k-v1/v0.2.0`, `automationbench-v1/v0.2.0`, `mmlu-pro-v1/v0.1.0`, `ifeval-v1/v0.1.0`, `reasoning-gym-v1/v0.1.0`, and `math-python-v1/v0.1.0`, but keep every framework-composed binding on the one qualified repository commit. Do not remove or repoint the old GSM8K and AutomationBench sources until this clean-clone gate passes.

Milestone status (2026-08-03): complete for source publication and dependency closure. The external repository is pushed at `017ac72f543f79f48400cbb4cb641d6df4c3adfa`; the clean-clone wheel builds and the six-wheel hash-locked compile passed. The framework pin is now this exact revision. Package publication tags and live qualification remain separate gates.

### Milestone 7: migrate framework consumers and add capability bindings

Before repinning the environment source, preserve the generic subset contract in the framework consumer path. `EvaluationBudget.num_tasks` is an invocation override over the reusable `EnvironmentBinding.num_tasks`; `EvaluationBudget.shuffle` is an optional override over the compatibility `EvaluateRequest.shuffle` field. The effective values are `EvaluateRequest.resolved_budget` and `EvaluateRequest.resolved_shuffle`. The Verifiers adapter passes them to `EvalConfig.num_tasks`, `EvalConfig.num_rollouts`, `EvalConfig.max_concurrent`, and `EvalConfig.shuffle`. Verifiers' fixed seed makes a shuffled finite-taskset subset reproducible, but the native bundle remains authoritative and must retain the resolved config. `evaluate` records `num_tasks`, `num_rollouts`, `max_concurrent`, and `task_selection` in events, metrics, and artifact metadata. Do not add arbitrary task IDs or a framework-owned dataset copy: environment activation owns semantic split/category/balancing configuration, and task evidence owns stable row IDs and digests.

Acceptance for this cross-cutting contract is `uv run --package posttrain-eval --python 3.13 pytest packages/eval/tests`, Ruff, and Pyright; a native adapter smoke must inspect the generated `EvalConfig` and show `shuffle=True` plus the requested count; and a disposable run must show the same `task_selection` and native `config.toml` on retry. A new environment binding is not release-ready if its activation cannot define its semantic split or if its selected task count cannot be proven from the native trace population.

First migrate existing consumers. In `packages/eval/pyproject.toml`, replace the upstream `gsm8k-v1` Git dependency with the published `verifiers-environments` commit and subdirectory, then refresh `uv.lock`. In `packages/eval/src/posttrain/eval/programs/general_smoke.py`, `packages/catalog/src/posttrain/catalog/base/environments.yaml`, and `apps/lab/.posttrain/catalog/algorithm-qualification.yaml`, repoint GSM8K sources and add the explicit pinned Hub fields to taskset activation. Update source assertions in `packages/eval/tests`, `packages/catalog/tests`, `apps/lab/tests`, `packages/execution-pack/tests`, and `packages/execution-buildkit/tests` only where they are intended to verify real source identity rather than generic fixture behavior.

Repoint AutomationBench in `packages/catalog/src/posttrain/catalog/base/environments.yaml`, `packages/eval/src/posttrain/eval/programs/automationbench.py`, catalog/eval/Lab tests, and `docs/tooling/automationbench/README.md`. Run its existing managed qualification before removing the old source. After the published wheel packs and a live task completes with equivalent native trace evidence, delete `/home/hammad/projects/rl/environments/automationbench_v1`. Update `AGENTS.md` and `docs/tooling/verifiers/README.md` to say CarbonTeq-owned generic environments live in `carbonteq-ai/verifiers-environments`; append a superseded-location note to `docs/decisions/0009-native-verifiers-environment-packages.md` without rewriting its historical decision.

In `/home/hammad/projects/rl/packages/catalog/src/posttrain/catalog/base/environments.yaml`, add evaluation bindings with these stable IDs:

- `knowledge-mmlu-pro-cot-5shot-balanced-v1`, category `knowledge-reasoning`, 1,400 tasks, one deterministic rollout, 100 tasks per category.
- `instruction-ifeval-full-v1`, category `instruction-following`, all 541 tasks, one deterministic rollout.
- `reasoning-gym-balanced-eval-v1`, category `procedural-reasoning`, 1,000 tasks, 100 per generator, one deterministic rollout.
- `math-python-balanced-eval-v1`, category `math-tool-use`, 500 deterministically shuffled test tasks balanced by type as far as the source permits, one deterministic rollout, and concurrency limited by measured Docker capacity.
- `reasoning-gym-train-v1`, category `procedural-reasoning`, train seed namespace, stochastic sampling appropriate for online RL.
- `math-python-train-v1`, category `math-tool-use`, MATH train split, stochastic sampling appropriate for online RL.

Every new and migrated CarbonTeq environment source points to `https://github.com/carbonteq-ai/verifiers-environments`, the published full `ENVIRONMENTS_REVISION`, and its package subdirectory. Every activation is `kind: verifiers-config`. Repeat the Hub and generator revisions inside taskset config. Put generation sampling and rollout budgets on `EnvironmentBinding`, not inside the taskset. Put only environment-owned reward component names in `reward_components`.

In `packages/catalog/src/posttrain/catalog/base/evaluations.yaml`, add `general-capability-balanced-v1` referencing the four evaluation bindings. Do not replace `general-smoke-v1`; the new pack is a broader baseline, while the old pack remains a cheap framework smoke.

In `apps/lab/.posttrain/catalog/algorithm-qualification.yaml`, repoint the existing `gsm8k-eval-qualification` binding, add two-task bindings named `automationbench-zapier-simple-qualification`, `mmlu-pro-qualification`, `ifeval-qualification`, `reasoning-gym-qualification`, and `math-python-qualification`, and compose those six bindings into `environment-library-qualification-v1`. The AutomationBench overlay uses one rollout instead of the training binding's eight; all six entries use the same environment source and activation revisions as their production binding but tiny release-gate budgets.

Create the following exact manifests under `apps/lab/.posttrain/work_packages/`:

| Manifest | `work_package_id` | Inline recipe ID | Bound environment |
| --- | --- | --- | --- |
| `environment_library_gsm8k_qualification.yaml` | `eval/qwen3.5-2b/environment-library/gsm8k` | `recipes/qwen3.5-2b-environment-library-gsm8k@1` | `gsm8k-eval-qualification` |
| `environment_library_automationbench_qualification.yaml` | `eval/qwen3.5-2b/environment-library/automationbench` | `recipes/qwen3.5-2b-environment-library-automationbench@1` | `automationbench-zapier-simple-qualification` |
| `environment_library_mmlu_pro_qualification.yaml` | `eval/qwen3.5-2b/environment-library/mmlu-pro` | `recipes/qwen3.5-2b-environment-library-mmlu-pro@1` | `mmlu-pro-qualification` |
| `environment_library_ifeval_qualification.yaml` | `eval/qwen3.5-2b/environment-library/ifeval` | `recipes/qwen3.5-2b-environment-library-ifeval@1` | `ifeval-qualification` |
| `environment_library_reasoning_gym_qualification.yaml` | `eval/qwen3.5-2b/environment-library/reasoning-gym` | `recipes/qwen3.5-2b-environment-library-reasoning-gym@1` | `reasoning-gym-qualification` |
| `environment_library_math_python_qualification.yaml` | `eval/qwen3.5-2b/environment-library/math-python` | `recipes/qwen3.5-2b-environment-library-math-python@1` | `math-python-qualification` |

Each manifest uses stage `qualify`, one required job named `evaluate` with definition `eval/verifiers-managed-general@1`, and the inline recipe ID shown above. The exact shared bindings are `models/qwen3.5-2b@bf16`, `inference/qwen3.5-2b-vllm-eval@1`, `targets/local-cuda-8gb`, and evaluation `environment-library-qualification-v1`. Do not add an environment-specific runner. The six manifests are necessary because the current work-package seat map is global and one evaluation job executes one environment cell.

Update `docs/tooling/verifiers/README.md` with the external environment repository, published commit, six package names, Hub revisions, Math Python image digest, migration evidence, live qualification evidence, and remaining sandbox caveat. Add a discoverability section to `packages/catalog/README.md` showing the exact `posttrain catalog list --family environment`, `posttrain catalog show environment ID`, and `posttrain catalog show evaluation ID` commands; the current README explains catalog authorship but not discovery.

Milestone status (2026-08-03): framework catalog integration is complete. The six bindings, four-environment balanced plan, source pins, release constraints, and six one-cell work packages are present and catalog/eval tests pass. The runtime source allowlist was extended to include `packages/environment`; the remaining work is to use a verified rebuilt/published eval parent and execute the provider-backed qualification.

### Milestone 8: prove packaging, execution, and evidence end to end

Add catalog decoding tests for every new binding and plan. Extend execution-buildkit coverage so one test builds at least two of the new wheels from the same Git checkout and proves separate wheel locks with one source revision. Extend job-package tests only if a newly observed case is not already covered by the generic multiple-wheel tests.

Run `posttrain catalog validate`, list the six library bindings, and show the balanced plan. Plan and pack all six Lab qualification jobs. Because every work package binds the shared six-environment evaluation plan, inspect each generated package and verify the complete plan closure: six environment package locks, six declarative activation locks, the shared exact source revision, exact Hub/generator revisions inside activation config, and the Math Python image digest. Also verify that execution selects only the manifest's bound environment cell.

Run the 12-task environment-library qualification against a small OpenAI-compatible local model endpoint. A low score is acceptable; missing tasks, tool calls, rewards, source metadata, finalization, or traces are not. In Observatory, verify one trace from each environment and record its run ID. GSM8K must show its Hub revision and correct reward; AutomationBench must show assertion and final-world-state evidence; MMLU-Pro must show category and parse success; IFEval must show four strict/loose metrics; Reasoning Gym must show generator/seed/native score; Math Python must show tool calls, errors, parse success, and symbolic correctness.

Finally run the balanced plan only after estimating token and tool-runtime cost from the qualification run. The balanced run is a release qualification, not a unit test. Record per-environment completion counts, error counts, wall time, token usage, and the exact model artifact. Do not claim a model baseline from partial completion.

### Milestone 9: retire the framework-local environment implementation

This is the final migration milestone and is intentionally ordered after source
publication, normal runtime-image publication/configuration, Math Python image
publication and lifecycle qualification, six-cell live evidence, and both validation
ladders. The goal is to leave `/home/hammad/projects/rl` with no concrete
reusable Verifiers environment implementation while preserving the framework's
generic ability to pack a project-owned environment path. The only tracked
concrete package currently in the root is
`environments/automationbench_v1`; remove that package, not every string that
contains `environments/`.

Before deletion, verify from `/home/hammad/projects/verifiers-environments` that
the exact pinned commit `017ac72f543f79f48400cbb4cb641d6df4c3adfa` is reachable
from the public remote, all six package wheels build from a clean clone, and
the framework's six live runs and reconciliation records are already recorded
above. Verify that the normal machine configuration points at the published
eval kind digest rather than the temporary project overlay, and finish the
Math Python success/error/timeout/cancellation/process-exit cleanup matrix. If
any prerequisite is missing, do not delete the old tree.

Run an active-reference audit from `/home/hammad/projects/rl`. Local filesystem
paths, local `uv` project commands, path dependencies, and imports from the
old package must return no matches. External Git source declarations such as
`https://github.com/carbonteq-ai/verifiers-environments` with subdirectory
`environments/automationbench_v1` are expected and must remain. Generic fixture
paths such as `environments/toy_env`, `environments/math`, and
`environments/text` are also expected and must remain because they exercise the
framework's project-path source contract. Update the active ownership and
developer instructions in `README.md`, `AGENTS.md`,
`docs/tooling/automationbench/README.md`, `docs/tooling/verifiers/README.md`,
and `docs/HANDOFF.md`; retain historical references in decision and archived
plan documents, but label them as historical when the surrounding prose could
otherwise be read as an active path.

After the audit and documentation edits, use scoped `git rm -r
environments/automationbench_v1`. Do not remove `packages/execution-*` generic
environment tests, `ProjectPathEnvironmentSource`, external catalog source
subdirectories, or the external repository itself. Re-run catalog validation,
the focused catalog/eval/execution/buildkit/Lab tests, and the complete root
validation ladder. The final diff must show the legacy package deletion plus
the active-documentation correction and must not include unrelated dirty
worktree changes.

Milestone 9 acceptance is observable when `test ! -e environments/automationbench_v1`
and `git ls-files environments` produce no entries, the active-reference audit
finds no local implementation/path dependency, `posttrain catalog validate`
still succeeds, generic `environments/<project-path>` fixture tests still pass,
and a clean external checkout imports `automationbench_v1` from the published
repository. The external package path remains usable; only the duplicate
framework-local implementation is gone.

## Concrete Steps

Work in the external repository first. Commands below assume Milestone 1 has created and cloned the repository at the stated path. Development may occur on a branch, but every consumer and clean-clone check uses the pushed commit resolved below.

    cd /home/hammad/projects/verifiers-environments
    ENVIRONMENT_PROJECTS="gsm8k_v1 automationbench_v1 mmlu_pro_v1 ifeval_v1 reasoning_gym_v1 math_python_v1"
    uvx --from ruff==0.16.1 ruff check scripts
    uvx --from ruff==0.16.1 ruff format --check scripts
    uv run --python 3.12 python scripts/check_boundaries.py
    mkdir -p dist
    for PROJECT in $ENVIRONMENT_PROJECTS; do
        (
            cd "environments/$PROJECT"
            uv sync --locked --python 3.12
            uv run ruff check .
            uv run pyright
            uv run pytest -m 'not network and not docker'
            uv build --wheel --out-dir /home/hammad/projects/verifiers-environments/dist
        )
    done
    for PROJECT in gsm8k_v1 mmlu_pro_v1 ifeval_v1 math_python_v1; do
        VENV_HF_HOME="$(mktemp -d /tmp/verifiers-env-hf.XXXXXX)"
        (cd "environments/$PROJECT" && HF_HOME="$VENV_HF_HOME" uv run pytest -m network)
    done
    (cd environments/math_python_v1 && uv run pytest -m docker)
    COMPAT_ENV="$(mktemp -d /tmp/verifiers-env-compat.XXXXXX)"
    uv venv --python 3.12 "$COMPAT_ENV"
    uv pip install --python "$COMPAT_ENV/bin/python" dist/*.whl
    "$COMPAT_ENV/bin/python" scripts/verify_combined_install.py
    git diff --check

Expected evidence includes six wheel filenames, GSM8K and AutomationBench parity results, and tests that report the exact source row counts. Network tests must skip with a clear reason only when outbound HTTPS is unavailable; they may not be skipped for a release claim.

AutomationBench resolves from the public CarbonTeq fork at immutable merge commit `908db2abd4a868acc37ab0850474bff653bea25c`; no private registry credential is required for the environment-library build. The fork's own release publication remains a separate maintained-fork lifecycle and is recorded in the package README.

After commit and push, prove clean-source reproducibility:

    cd /home/hammad/projects/verifiers-environments
    ENVIRONMENTS_REVISION="$(git rev-parse HEAD)"
    test "$(printf '%s' "$ENVIRONMENTS_REVISION" | wc -c)" -eq 40
    git fetch origin
    git branch -r --contains "$ENVIRONMENTS_REVISION" | rg 'origin/'
    RELEASE_CHECKOUT="$(mktemp -d /tmp/verifiers-environments-release.XXXXXX)"
    git clone https://github.com/carbonteq-ai/verifiers-environments "$RELEASE_CHECKOUT"
    git -C "$RELEASE_CHECKOUT" checkout "$ENVIRONMENTS_REVISION"
    cd "$RELEASE_CHECKOUT"
    mkdir -p dist
    for PROJECT in gsm8k_v1 automationbench_v1 mmlu_pro_v1 ifeval_v1 reasoning_gym_v1 math_python_v1; do
        (
            cd "environments/$PROJECT"
            uv sync --locked --python 3.12
            uv run pytest -m 'not network and not docker'
            uv build --wheel --out-dir "$RELEASE_CHECKOUT/dist"
        )
    done
    RELEASE_COMPAT_ENV="$(mktemp -d /tmp/verifiers-env-release-compat.XXXXXX)"
    uv venv --python 3.12 "$RELEASE_COMPAT_ENV"
    uv pip install --python "$RELEASE_COMPAT_ENV/bin/python" dist/*.whl
    "$RELEASE_COMPAT_ENV/bin/python" -c 'import automationbench_v1, gsm8k_v1, ifeval_v1, math_python_v1, mmlu_pro_v1, reasoning_gym_v1'

Then work in the framework repository:

    cd /home/hammad/projects/rl
    uv run posttrain catalog validate
    uv run posttrain catalog list --family environment
    uv run posttrain catalog show environment math-gsm8k
    uv run posttrain catalog show environment automationbench-zapier-simple-grpo
    uv run posttrain catalog show evaluation general-capability-balanced-v1
    uv run pytest -q packages/catalog/tests packages/environment/tests packages/execution/tests packages/execution-pack/tests packages/execution-buildkit/tests apps/lab/tests
    for ENVIRONMENT in gsm8k automationbench mmlu_pro ifeval reasoning_gym math_python; do
        WORK_PACKAGE="apps/lab/.posttrain/work_packages/environment_library_${ENVIRONMENT}_qualification.yaml"
        uv run posttrain job plan "$WORK_PACKAGE" --job evaluate
        uv run posttrain job pack "$WORK_PACKAGE" --job evaluate
    done
    if rg -n 'PrimeIntellect-ai/verifiers.*gsm8k_v1|carbonteq-ai/posttrain.*automationbench_v1' packages apps AGENTS.md docs/tooling; then
        echo 'stale upstream or posttrain-owned environment source remains' >&2
        exit 1
    fi
    rg -n 'carbonteq-ai/verifiers-environments.*(gsm8k_v1|automationbench_v1)' packages apps AGENTS.md docs/tooling

Before packing locally after a runtime-lock change, verify the parent image rather than relying on a tag or on `--build-missing` to rewrite a plan. `--build-missing` may rebuild and push a digest-pinned image, but it does not alter the already-planned `registry.kind_images` value. The CLI therefore fails closed if that configured digest still has the old lock label. Use the retained receipt to update a project-local ignored override, then rerun the pack with an isolated configuration directory so machine credentials and registry settings cannot affect the evidence:

    cd /home/hammad/projects/rl
    sed -n '1,80p' apps/lab/.posttrain/state/runtime-builds/4d109bd98f922f102197aea671465eca6942a0c1502c6a3d7a8e3907f99ad955.json
    # Set [registry.kind_images].eval to the receipt's digest in the ignored
    # apps/lab/.posttrain/state/execution.toml after the registry exposes it.
    XDG_CONFIG_HOME="$(mktemp -d)" uv run --frozen posttrain --project-root apps/lab \
        job pack apps/lab/.posttrain/work_packages/environment_library_gsm8k_qualification.yaml \
        --job evaluate --local --allow-deferred-qualification

The expected successful local output contains `Local OCI layout ready`, a
content-addressed `Package`, and a receipt path. Inspect the resulting OCI
manifest and assert its `kind_image` equals the configured rebuilt digest before
using it for any live run. A pack that still shows the old published digest is
diagnostic only and must not be called qualification evidence.

After the migration is complete, the final `rg` command must return no active source or path dependency; historical plan evidence may still mention the old locations. The remote-branch check must print at least one `origin/*` ref containing the exact commit. Do not substitute a local-only commit.

For the full framework gate, run from `/home/hammad/projects/rl`:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

The live qualification command must use `posttrain job run` with the existing generic eval work package and a protected runtime-value file. Record the exact command with secret values omitted, the provider, run ID, actual image digest, and Observatory URL in this plan after execution.

For the final framework-local environment retirement, run the following only
after Milestone 9 prerequisites and the active-reference audit pass:

    cd /home/hammad/projects/verifiers-environments
    ENVIRONMENTS_REVISION="$(git rev-parse HEAD)"
    test "$ENVIRONMENTS_REVISION" = "017ac72f543f79f48400cbb4cb641d6df4c3adfa"
    git fetch origin
    git branch -r --contains "$ENVIRONMENTS_REVISION" | rg 'origin/'

    cd /home/hammad/projects/rl
    if rg -n --glob '!docs/plan/**' --glob '!docs/decisions/**' \
      --glob '!docs/dx-improvements/**' \
      '/home/hammad/projects/rl/environments/automationbench_v1|uv (sync|run)[^\n]*environments/automationbench_v1|path[^\n]*environments/automationbench_v1' \
      AGENTS.md README.md docs/HANDOFF.md docs/tooling apps packages release; then
        echo 'active framework-local AutomationBench reference remains' >&2
        exit 1
    fi
    git rm -r environments/automationbench_v1
    test ! -e environments/automationbench_v1
    test -z "$(git ls-files environments)"
    uv run posttrain catalog validate
    uv run pytest -q packages/catalog/tests packages/eval/tests packages/execution/tests packages/execution-pack/tests packages/execution-buildkit/tests apps/lab/tests
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Inspect `git diff --cached --name-status` and `git diff --stat` before staging
any documentation corrections. Only the legacy package deletion, active
documentation updates, and this plan may be included in the migration commit;
preserve all unrelated dirty worktree changes.

## Validation and Acceptance

The feature is accepted only when all of the following behavior is observable.

Catalog loading remains inert and works without Verifiers, datasets, Reasoning Gym, Docker, or the six packages installed in the developer environment. `posttrain catalog show` exposes exact environment Git and task-source revisions. A floating Hub revision such as `main`, a missing environment source revision, or an unregistered IFEval instruction fails validation before inference.

Each wheel builds independently from the published Git commit. A job pack built in a clean checkout records the published source commit, wheel digest, wheel size, and declarative activation digest for all six packages. Repacking unchanged source produces the same logical source and activation identities; changing a Hub revision changes the activation digest and `posttrain job diff` explains the environment activation change.

The actual-job parent image is digest pinned and carries the same lock digest as the shipped runtime definition. If the registry still exposes an older image, `posttrain job pack` fails before materialization; `--build-missing` may rebuild it, but packing remains blocked until the configured digest points at the verified rebuilt image. No successful local OCI layout with a stale parent counts as a package or qualification result.

GSM8K loads exactly 7,473 train or 1,319 test rows from Hub revision `740312add88f781978c0658806c59bc2815b9866`. Its prompts and rewards match the previously pinned upstream package on the parity corpus, while every trace adds immutable source and row identity. All active GSM8K dependencies and catalog bindings resolve from the published `verifiers-environments` commit.

AutomationBench resolves the same seed/domain samples, tools, assertions, rewards, metrics, and trace metadata as the former in-repository adapter. Its wheel builds from the published `verifiers-environments` commit, the managed qualification succeeds, and `/home/hammad/projects/rl/environments/automationbench_v1` no longer exists. No active catalog, dependency, or tooling instruction points at its old posttrain subdirectory.

The final ownership audit finds no framework-local concrete Verifiers package or
local path dependency. `git ls-files environments` is empty, while the generic
project-path contract remains demonstrably usable through the existing
execution-pack/buildkit fixture tests and catalog parser tests. External Git
source declarations continue to resolve the six packages from
`carbonteq-ai/verifiers-environments`; the removal is a lifecycle cleanup, not
an API or packaging regression.

MMLU-Pro yields the intended balanced 1,400-row subset with exactly 100 tasks from each category. Its reference-derived five-shot prompts are stable, correct labels score 1.0, incorrect or unparseable labels score 0.0, and parse success is separately visible.

IFEval loads 541 unique keys and implements every instruction ID present at the pinned revision. Its strict and loose per-instruction and per-prompt outputs match the pinned Google reference over the parity corpus. No judge model or network call is made during scoring.

Reasoning Gym yields equal counts for the ten named generators, uses disjoint train/eval task IDs and seed ranges, reproduces task bytes under the same config, and delegates reward to the pinned native scorer. The selected train binding can be consumed by the existing Verifiers GRPO bridge without a dataset seat.

Math Python loads the pinned MATH train and test splits, permits multi-call Python use, symbolically validates final boxed answers, isolates state between concurrent rollouts, bounds timeout/output/resources, and removes containers after success, error, timeout, cancellation, and process exit. Its tool runtime receives no credentials. The release notes explicitly retain the host-network caveat until it is removed in code and requalified.

The live environment-library qualification run completes all twelve selected tasks and persists one complete native Verifiers trace per task. Observatory can filter the run by environment and display each environment's named metrics and immutable source evidence. The four-environment balanced run completes every selected task or is reported as incomplete; partial results cannot become the baseline.

## Idempotence and Recovery

All Hub loads are read-only and content-addressed by full revision. Deleting a temporary `HF_HOME` and rerunning must reproduce the same row counts and task identities. Do not delete shared Hugging Face, uv, Docker, or execution-pack caches during normal retries.

Environment wheel builds write to disposable build directories and content-addressed caches. If one build fails, fix that package and rebuild it without changing the other source packages. Do not update the framework source pin until the external commit is pushed and reproducible from a clean clone.

Runtime image rebuilds are also content-addressed and retain a mode-600 receipt under the configured project state root. A rebuild is safe to retry; it never mutates `published.toml` or a planned job. If the rebuilt digest is not yet available under the project's configured registry reference, leave the pack blocked and publish or explicitly configure that digest before retrying. Use a temporary `XDG_CONFIG_HOME` for qualification commands when the host has a machine-level Posttrain config so the evidence names only the project-local registry and receipt root.

GSM8K and AutomationBench migration is additive until published-source qualification passes. Keep the old source pins and AutomationBench directory intact while building and testing replacements. Repoint consumers in one framework change. If parity or live qualification fails, revert only the source-pin change and continue using the old packages; do not delete the new repository history or rewrite its published commit. Delete the old AutomationBench directory only after every active reference is repointed, normal publication/configuration is complete, the live gate is complete, and the full validation ladder passes; Git history remains the recovery path after deletion. Before committing the scoped `git rm`, a failed audit can be recovered by restoring only that path from the current commit; after committing, recover it from the migration commit's parent rather than recreating source manually.

If a Hub repository becomes temporarily unavailable, preserve the failed plan/run evidence and retry the identical activation. Do not switch revisions or mirrors inside a retry. A deliberate source change requires a new environment repository commit, catalog revision, activation digest, qualification run, and Decision Log entry.

If the Math Python tool leaks a container, stop the qualification and record its exact container name and trace. Remove only that proven leaked container after preserving logs. Fix lifecycle cleanup and rerun success, failure, timeout, cancellation, and exit tests before resuming live evaluation.

If IFEval parity differs, compare one instruction checker at a time against Google commit `e6890f85757dd84e27ca6df2dd30651dafad28e0`. Do not compensate by weakening expected outputs. Record any intentional semantic correction as a new package/catalog revision rather than altering v1 silently.

If the external repository cannot be created under the proposed organization, stop before framework edits and record the approved canonical HTTPS remote here. Do not fall back to adding the six packages under `/home/hammad/projects/rl/environments`, and do not use a personal fork, local path, or branch name as a released source.

The current framework worktree contains unrelated in-progress dataset, workload, CLI, and qualification changes. Preserve them. This plan adds one file now; implementation must inspect overlapping diffs before editing catalog, Lab, documentation, or lockfiles and must never revert unrelated work.

## Artifacts and Notes

Milestone 1 evidence recorded on 2026-08-03:

- Public repository: `https://github.com/carbonteq-ai/verifiers-environments`; local checkout: `/home/hammad/projects/verifiers-environments`; repository state: `main` commit `017ac72f543f79f48400cbb4cb641d6df4c3adfa` pushed to `origin/main`.
- The six current project locks resolve independently at the exact Verifiers commit: GSM8K 115 packages, AutomationBench 116, MMLU-Pro 115, IFEval 116, Reasoning Gym 133, and Math Python 115.
- Isolated validation passed for every project: lock check, Ruff lint and format check, Pyright with zero errors, and package tests: GSM8K 5, AutomationBench 10, MMLU-Pro 4 plus 1 skip, IFEval 5 plus 1 skip, Reasoning Gym 3, and Math Python 3. The MMLU-Pro and IFEval skips are opt-in network gates, not failures.
- Enabled network gates passed for GSM8K, MMLU-Pro, and IFEval. MMLU-Pro observed exactly 70 validation and 12,032 test rows; IFEval observed 541 unique rows covering all 25 instruction IDs. Math Python loaded the pinned test row successfully; its full row-count and sandbox-image gates remain pending.
- Clean combined installation resolved 130 packages, installed all six current wheels together under Python 3.12.13 using the configured CarbonTeq index, and activated `AutomationBenchTaskset`, `GSM8KTaskset`, `IFEvalTaskset`, `MathPythonTaskset`, `MMLUProTaskset`, and `ReasoningGymTaskset` through the declarative loader without forcing network-backed task selection.
- Current uncommitted wheel evidence: `automationbench_v1-0.2.0-py3-none-any.whl` at SHA-256 `3fe6bd671080394e07a4ebfb41d5b17729f4b17f020ed409043a4ad4707ae779`; `gsm8k_v1-0.2.0-py3-none-any.whl` at `3a6540765571c76d42c8353bbca06a70cdea570f9cb20b7a26eb42169b9f6c1a`; `ifeval_v1-0.1.0-py3-none-any.whl` at `9eaf8c4fdd07616f89a14cc6535dde56d3848ee38e3b576fc30571795b8da383`; `math_python_v1-0.1.0-py3-none-any.whl` at `3b377cc0b839626bf1d7be6bd7b4f8dd47443474a705b95ce541bb83e45b614d`; `mmlu_pro_v1-0.1.0-py3-none-any.whl` at `8c559d8adc6cfaceea8245a314e2c5bf960c8fc5fb030c2a0dd8915ab2748622`; and `reasoning_gym_v1-0.1.0-py3-none-any.whl` at `dccdf9a384990049431e9c0247f542e76386857d65cde8baaeba9b43c64ca474`.
- These are local implementation-slice digests, not release artifacts. They must be regenerated after the first source commit and again after any semantic/package change.

- Math Python lifecycle evidence (2026-08-03): the published digest ran non-root as UID `65532`, imported Python `3.12.13`, NumPy `2.5.1`, and SymPy `1.14.0`, and exposed no `TOKEN`, `PASSWORD`, or `API_KEY` environment names. The package suite now passes 8 tests, including child timeout, process exit, state non-commit, and safe-environment checks. An explicit Docker probe returned success `4`, error `17`, process exit `23`, removed a cancelled container, and removed a timed-out container through the runtime-style `docker rm --force` cleanup path. A deliberately client-killed timeout first left one named test container; it was removed by exact name and is not evidence of a production leak.

- Trackio trace evidence (2026-08-03): all six live qualification reconciliation records retain an evaluation artifact with `provider=trackio`, a Trackio provider run ID, and `tracking_status=succeeded`; the native Verifiers trace synchronizer writes the task trace into the retained evaluation artifact before finalization. The host-side Observatory query was not used because this shell cannot resolve `trackio.lan`; container logs and reconciliation records are the durable evidence.

Current framework integration evidence (2026-08-03): environment repository `017ac72f543f79f48400cbb4cb641d6df4c3adfa`; root `uv.lock` digest `7c6b061ccbdede7f91e3d4e45dd800b77ca3e0c028a892fdb5a8184f7bce2511`; runtime workspace lock digest `f68d5419b00b0fde278d0b1d55057eb65bb5c9bbdcc78d2ff5cbfab0a3e1fa2c`. The combined six-wheel `uv pip compile --generate-hashes` succeeded with no VCS URL lines. Catalog validation reported `Catalog valid: framework-v1, 55 base entries, 49 project entries`.

Runtime rebuild evidence (diagnostic, not publication): receipt `apps/lab/.posttrain/state/runtime-builds/4d109bd98f922f102197aea671465eca6942a0c1502c6a3d7a8e3907f99ad955.json` records eval kind image `registry.lan/carbonteq/posttrain-kind-eval@sha256:13ea50188e3064ac7ee38079d85367e11f20a0db627bf6f69d3177b1a73c1b28` and lock digest `f68d5419b00b0fde278d0b1d55057eb65bb5c9bbdcc78d2ff5cbfab0a3e1fa2c`. A prior local GSM8K OCI layout at package key `ea4b412658a40dcfa2628ef79f808cef2d0440c2a1a99d3310a958687e696843` used stale parent digest `sha256:360810f695108982f077603d4248db232cf0e273213def0ccd65e1fb44fb3094`; it is explicitly excluded from release evidence.

Verified local OCI pack evidence (2026-08-03) used the rebuilt eval parent digest above for all six one-cell manifests: GSM8K package `8670f63285c35e0951bf815a538aeb838690018db65a846daf462bec6ac33e25`, AutomationBench `98fc10086e7714945bcfea93c57d192c2b1c0cb987ce5d845c0c9411f3cd0ae7`, MMLU-Pro `b16ec9d0f1f9f8e21f8b76855c9ae29926259058c6e39a4c2b9ffbba0eecf5af`, IFEval `3c1d4a48bea36ee493b85629c4f9f698fa3e8c139bdd540226b2a1232f9527a8`, Reasoning Gym `2abce4aa4b0e9ddb879fff2d0c97f73164a732ddbd9af548f760aa7885764b0a`, and Math Python `50b1b7d9a25820780771125426cd8070a7d6f5a4961296b90b59037baaa8258f`. Each build emitted `Local OCI layout ready`, installed all six environment wheels, and ran the deferred qualification smoke successfully.

Live GSM8K evidence (2026-08-03): run `envlib-gsm8k-20260803-live`, provider `local-docker`, provider run `pt-41ca6ad6cb1723a15c456443`, job image `registry.lan/carbonteq/posttrain@sha256:69defb79c2533c6ea1fb41955b293c6a8a92fd8f1261f68e6fb499dfee26d994`, terminal status `succeeded`, and reconciliation `state=consistent` with provider and Trackio both `succeeded`. Required evaluation artifact digest: `54005120c95d1b875fe0dbe2436c3ba6d6b67b85e48e416002414f6657c582b1`. The complete six-cell matrix is recorded below.

Complete live qualification matrix (2026-08-03): every row used the rebuilt eval parent `registry.lan/carbonteq/posttrain-kind-eval@sha256:13ea50188e3064ac7ee38079d85367e11f20a0db627bf6f69d3177b1a73c1b28`, local Docker, the generic `eval/verifiers-managed-general@1` job, and the temporary explicit overlay. GSM8K: run `envlib-gsm8k-20260803-live`, provider `pt-41ca6ad6cb1723a15c456443`, job `...@sha256:69defb79c2533c6ea1fb41955b293c6a8a92fd8f1261f68e6fb499dfee26d994`, evaluation artifact `54005120c95d1b875fe0dbe2436c3ba6d6b67b85e48e416002414f6657c582b1`; AutomationBench: `envlib-automationbench-20260803-live`, `pt-5af8be8a3013d0ed36c3f566`, job `...@sha256:ff609e46f6e99b8f89c9af5c15b26a2a4ed1fee319d3ba262ba081e2db9fd81a`, artifact `e59e8c57de5ac39b1d394dcb09ac35c1572a60babdfc8770ade65e84116a6791`; MMLU-Pro: `envlib-mmlu-pro-20260803-live`, `pt-9af9668453f91ac0d42c3141`, job `...@sha256:c3ca5b41d23639c136a2954e9118df24aea4c884b4134417e08df041d5272b7c`, artifact `7d20b1aa54fd4fc642f8d5cefa2e651770a023cb80d8702af2dba2eb5f3049c4`; IFEval: `envlib-ifeval-20260803-live`, `pt-f734829af27cceca011caba2`, job `...@sha256:40484b9a3eba32185fd47d696b399070d8ef005192503fdc4239a3c10602d9ae`, artifact `0902d94a8f6c4d6fc9131111b890f9968c41aebe230db4ad9c4d18ffd598aa79`; Reasoning Gym: `envlib-reasoning-gym-20260803-live`, `pt-792761a4d1ebd042c720371b`, job `...@sha256:80d11b7ee9472a646cd70e10f0d9e754ea140389903067269a842e1f7e54fb51`, artifact `80e481bd98c0b813ca1c80c37387e3f8b258d2b13ea9a8916e7eb2acfc930aa2`; Math Python: `envlib-math-python-20260803-live`, `pt-2af9c03710a2878ce5f64156`, job `...@sha256:504d246f3cf2f518459e16bf0af0c8510394b5e767f51bc5f05eb0e12de18a2b`, artifact `d47351ec025fb396b52e21e5a047eddf950a2b427f269e597f3788336f4ea0c0`. All six reconciled `state=consistent`, `outcome=succeeded`, provider/Trackio `succeeded`, and no missing required roles.

Framework milestone commit: `90d72802` (`feat: integrate verifiers environment catalog and qualification`) pushed to `origin/codex/trackio-post8-purge-qualification`. It contains the six package pins, catalog/evaluation bindings, six Lab work packages, runtime lock/guard, and living-plan updates through the pre-live gate. The subsequent live GSM8K evidence is recorded above and must be committed as the next plan-only amendment.

Live gate attempt (2026-08-03): `posttrain job run ...environment_library_gsm8k_qualification.yaml --provider local --run-id envlib-gsm8k-20260803 --allow-deferred-qualification` stopped before materialization with the stale-image error naming lock `df1b2b0b2b04b25bacdbdac9fedc9d908abd8b77ff2ae0945b7fbda09cde725e` versus shipped lock `f68d5419b00b0fde278d0b1d55057eb65bb5c9bbdcc78d2ff5cbfab0a3e1fa2c`. No run ID, provider container, or tracking evidence exists for this attempt.

Live GSM8K qualification (2026-08-03): using a temporary explicit project execution overlay only for this run, the machine-local provider submitted `envlib-gsm8k-20260803-live` as provider run `pt-41ca6ad6cb1723a15c456443`. The verified parent was `registry.lan/carbonteq/posttrain-kind-eval@sha256:13ea50188e3064ac7ee38079d85367e11f20a0db627bf6f69d3177b1a73c1b28`; the materialized job image was `registry.lan/carbonteq/posttrain@sha256:69defb79c2533c6ea1fb41955b293c6a8a92fd8f1261f68e6fb499dfee26d994`. `posttrain run reconcile` reported provider `succeeded`, tracking `succeeded`, `state=consistent`, `outcome=succeeded`, provider exit code `0`, no missing required roles, and retained evaluation artifact digest `54005120c95d1b875fe0dbe2436c3ba6d6b67b85e48e416002414f6657c582b1`. The container log records native run creation and completion plus remote Trackio log upload. The host-side Observatory query was not used as evidence because this shell cannot resolve `trackio.lan`.

At later milestones, append concise evidence here:

- `ENVIRONMENTS_REVISION=<40-character pushed commit>`.
- Six wheel filenames, versions, and SHA-256 digests.
- `MATH_PYTHON_IMAGE=registry.lan/carbonteq/math-python-v1@sha256:67624f5e71f8a5c89d25bc6c42370eb6e71b8569788aa818e5d3fe8585f15f15`; local build ID `sha256:d983bba26930a02c4b1ea456d744443da386f2798b4f449f5f334892ffeda9b0` remains diagnostic only.
- Clean-cache observed row counts and dataset fingerprints or row-manifest digests.
- GSM8K old/new parity transcript and AutomationBench old/new parity transcript.
- The framework commit that repoints consumers and removes the old AutomationBench directory.
- Final migration evidence: active-reference audit transcript, scoped deletion status showing no tracked `environments/` entries, active documentation diff, catalog validation result, focused framework test result, and full root validation result. Record the migration commit separately from the external environment-repository commit.
- The exact catalog-show excerpt for `general-capability-balanced-v1`.
- The job-package excerpt showing six source and activation locks and the selected execution cell.
- Live qualification run ID, actual image digest, model artifact, provider, and terminal status.
- One Observatory trace ID per environment and the metric names observed.
- Balanced run completion/cost summary when that gate is run.

Do not paste credentials, signed URLs, bearer tokens, or protected environment values into this plan.

## Interfaces and Dependencies

The external environment packages must expose these importable v1 classes at completion:

    gsm8k_v1.GSM8KTaskset
    gsm8k_v1.GSM8KConfig
    automationbench_v1.AutomationBenchTaskset
    mmlu_pro_v1.MMLUProTaskset
    mmlu_pro_v1.MMLUProConfig
    ifeval_v1.IFEvalTaskset
    ifeval_v1.IFEvalConfig
    reasoning_gym_v1.ReasoningGymTaskset
    reasoning_gym_v1.ReasoningGymConfig
    math_python_v1.MathPythonTaskset
    math_python_v1.MathPythonConfig
    math_python_v1.PythonToolset
    math_python_v1.PythonToolsetConfig

Their taskset IDs are the normalized distribution names: `gsm8k-v1`, `automationbench-v1`, `mmlu-pro-v1`, `ifeval-v1`, `reasoning-gym-v1`, and `math-python-v1`. Their public configuration must be accepted by `verifiers.v1.env.EnvConfig.model_validate` through a normal declarative activation; no `PythonFactoryActivation` compatibility path is permitted.

Use `datasets` for exact-revision Hugging Face loading, `reasoning-gym` at the pinned Git commit for generation and native scoring, and `math-verify` for strict symbolic math equivalence. IFEval uses only deterministic package code derived with attribution from the pinned Apache-2.0 reference. Do not introduce W&B, Trackio, framework, model-provider, or trainer imports into the environment packages.

The framework-facing contract remains the existing `EnvironmentSource`, `VerifiersV1ConfigActivation`, `EnvironmentBinding`, `EvaluationPlan`, `EnvironmentPackageLock`, and `EnvironmentActivationLock`. No reusable framework package may import any of the six concrete distributions. Lab composition may install an environment extra for direct qualification, but reusable packages see it only through the generic Verifiers contract. The catalog source must be a canonical secret-free HTTPS Git URL and a full commit; the Hub source locks live inside inert taskset configuration and typed task evidence.

Every environment project uses Python 3.12 for authoring and must remain compatible with the Python version in every selected eval and online-RL actual-job kind. Each subdirectory's `uv.lock` is the executable authority for developing and testing that package; the clean-clone combined-wheel install is the authority for cross-package runtime compatibility. The framework's `uv.lock` does not need to absorb these concrete distributions unless a framework test explicitly installs one; job packing is the production dependency path.

Revision note (2026-08-02): created the initial self-contained plan after source, license, split, Verifiers v1, framework-boundary, packaging, and sandbox analysis. The plan chose four external v1 packages, immutable Hugging Face task sources, procedural Reasoning Gym generation, and separate balanced versus release-qualification catalog budgets.

Revision note (2026-08-02): renamed the proposed repository from `posttrain-environments` to framework-neutral `verifiers-environments`, added a dataset-pinned GSM8K reimplementation and an AutomationBench lifecycle migration, and revised every plan section to cover six independent packages, consumer repinning, safe deletion order, parity evidence, and twelve-task library qualification. This resolves the difference between dependency isolation inside `rl/environments/` and a genuinely separate Git/release lifecycle.

Revision note (2026-08-03): made the qualification path executable without chat context. The plan now records AutomationBench's exact distribution and fork lineage, replaces future-revision and work-package placeholders with reproducible commands, and defines six one-cell Lab work packages sharing one six-environment evaluation plan because the current job and seat contracts do not iterate environment cells.

Revision note (2026-08-03): retained a single `carbonteq-ai/verifiers-environments` Git repository after verifying the current posttrain implementation. The plan now makes each environment a standalone uv project, wheel, version, test suite, and package lock; records that the source packer fetches the shared revision once and emits per-subdirectory wheel locks; and makes the same-repository/same-revision requirement an explicit cross-package release gate rather than hiding it as independent Git lifecycle.

Revision note (2026-08-03): recorded user acceptance of the monorepo with strict package isolation and closed repository topology as a planning question. Implementation should now begin at Milestone 1 and only revisit this decision if qualification exposes a concrete incompatibility.

Revision note (2026-08-03): completed Milestone 1 locally. Created the public empty repository and local checkout; added six independently locked synthetic Verifiers v1 packages, boundary enforcement, matrix CI, wheel builds, and clean combined activation; and recorded exact validation and scaffold digests. Source publication remains a later gated action, so no commit or push was made.

Revision note (2026-08-03): ran a parallel implementation wave across the monorepo. GSM8K and AutomationBench now have real standalone implementations with local parity coverage; MMLU-Pro and IFEval have exact-revision deterministic implementations and enabled network gates; Reasoning Gym delegates to the pinned generator registry; and Math Python has the pinned MATH loader, boxed-answer verifier, and bounded child-interpreter tool. Updated the combined-install helper to test declarative activation without synthetic reward assumptions. No framework consumer was repointed, no old source was deleted, and no source commit or push was made.

Revision note (2026-08-03): confirmed Verifiers v1's finite-taskset subset contract and completed the framework-side DX slice. Standard invocation budgets now support count plus optional fixed-seed shuffle, direct request construction remains backward-compatible, and native/evidence paths record the effective selection policy. Package eval validation: 28 passed, 2 skipped; standard jobs validation: 11 passed; native GPU/endpoint qualification remains a later release gate.

Revision note (2026-08-03): created local environment-library commit `2ac96d3` after package, boundary, and combined-wheel validation. The commit is intentionally not pushed until the clean-clone publication gate; no framework dependency pin has been advanced.

Revision note (2026-08-03): advanced the shared source to pushed commit `017ac72f543f79f48400cbb4cb641d6df4c3adfa`, recorded the vendored AutomationBench/Reasoning Gym closure, repinned all framework consumers, and added the runtime-lock and source-allowlist fixes needed by real job packing. A diagnostic GSM8K local pack revealed that `--build-missing` could otherwise retain a stale configured parent; the CLI now fails closed and the plan treats a verified rebuilt/published kind digest as a prerequisite for live qualification.

Revision note (2026-08-03): committed and pushed framework catalog/runtime integration as `90d72802`, then completed the first provider-backed native gate. GSM8K run `envlib-gsm8k-20260803-live` succeeded on local Docker with the rebuilt eval parent, materialized job image, Trackio finalization, and consistent retained evaluation evidence. At that historical checkpoint, the remaining gates were the other five live cells, normal publication/configuration of the rebuilt eval parent, Math Python image publication, and safe removal of the old AutomationBench copy.

Revision note (2026-08-03): completed all six provider-backed environment-library cells through Posttrain, including AutomationBench, MMLU-Pro, IFEval, Reasoning Gym, and Math Python. Every run reconciled consistently with provider/Trackio success and a retained evaluation artifact. The plan now treats live qualification as complete; only normal eval-image publication/configuration, standalone Math Python image publication/lifecycle coverage, full root validation, and old-source removal remain.

Revision note (2026-08-03): added Milestone 9 at the user's direction. The final migration now explicitly removes the tracked `environments/automationbench_v1` implementation from `rl` only after publication/configuration, Math Python lifecycle coverage, full validation, and an active-reference audit. The plan distinguishes this scoped deletion from generic project-owned `environments/<path>` support and external `verifiers-environments/environments/<package>` subdirectories, and defines the exact documentation, audit, recovery, and acceptance steps.

Revision note (2026-08-03): switched the Math Python image release gate to the CarbonTeq OCI registry. Published `registry.lan/carbonteq/math-python-v1@sha256:67624f5e71f8a5c89d25bc6c42370eb6e71b8569788aa818e5d3fe8585f15f15`; no external registry publication target is required. Package and image lifecycle probes now pass with explicit cleanup; the remaining decision is provider-managed sandbox integration versus retaining the bounded subprocess path for this release, followed by normal eval-parent configuration, full validation, and final legacy-source removal.

Revision note (2026-08-03): added Math Python lifecycle regression coverage in external repository commit `9619d43` and pushed it to `origin/main`. The framework source pin remains `017ac72f543f79f48400cbb4cb641d6df4c3adfa` because this follow-up changes tests and release documentation only; it does not change the package wheel semantics. Recorded that Trackio receives and finalizes the native Verifiers traces for all six qualified cells.
