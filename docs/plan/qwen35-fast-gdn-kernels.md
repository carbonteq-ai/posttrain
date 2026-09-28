# Give Qwen3.5 training the fast Gated DeltaNet kernels

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Qwen3.5 is a hybrid model: most of its layers are Gated DeltaNet layers (a form of linear attention: a short
causal depthwise convolution followed by a gated "delta rule" recurrence), interleaved with ordinary attention
layers. Transformers 5.14.1 runs those layers on fast GPU kernels only when two optional packages import:
`fla` (from fla-core, Triton kernels) and `causal_conv1d` (from causal-conv1d, a CUDA extension). Posttrain's
runtime images had neither, so every Qwen3.5 SFT, DPO and TRL GRPO actor step ran a Python loop over 64-token
chunks and logged "The fast path is not available ... Falling back to torch implementation". vLLM vendors its
own copy of the fla kernels, so the sampler and the trainer also ran different implementations.

After this change every kind image that runs Qwen3.5 through Transformers carries fla-core (`supervised`,
`online-rl-trl-py312`, the veRL backend environment of `online-rl-verl-py313`, and `transform`), and a
CarbonTeq-built causal-conv1d wheel for PyTorch 2.13.0+cu130 is ready for the internal index, with a one-command
adoption script for when it is published. On the local RTX 3070 Ti a LoRA
actor step on Qwen3.5-0.8B is 3.1-3.7x faster, and fp16 training no longer produces NaN gradients. To see it,
run the qualification commands under Concrete Steps: the warning disappears and the actor timings drop.

The frozen product baseline (`docs/post-training/`) is unchanged: this only adds dependencies to existing
runtime images and does not change any public API or product meaning.

## Progress

- [x] (2026-09-28 10:10Z) Worktree `../rl-qwen-kernels`, branch `codex/qwen-fast-kernels`, from
  `rl-perf-guard` HEAD `c3ae803d`.
- [x] (10:20Z) Read the Qwen3.5 modeling code in the published TRL kind image; confirmed the import gates and
  that the image has torch 2.13.0+cu130, Triton 3.7.1, CPython 3.13.12, transformers 5.14.1, and no fla,
  causal-conv1d or `kernels` package.
- [x] (10:25Z) Selected fla-core 0.5.2 and causal-conv1d v1.7.0 (`cd81f041`).
- [x] (10:40Z) Built `causal_conv1d-1.7.0+cu130torch2.13-cp313-cp313-linux_x86_64.whl` with
  `tools/kernel-wheels/causal-conv1d/build.sh` (8 min), SHA-256 `b69f3914…4947a`.
- [x] (10:45Z) Added fla-core to `posttrain-train[trl]` and `profiles/supervised.txt`; `uv lock`;
  `posttrain-release lock-runtime-dependencies`; committed `79a92810`.
- [x] (11:05Z) Kernel checks on sm86 against the Transformers torch reference (both packages): PASS.
- [x] (11:10Z) Built the `online-rl-trl-py312` and `supervised` kind images locally from the new locks
  (no push) plus a local-only derived image with the causal-conv1d wheel.
- [x] (12:10Z) Actor timing, fp16/bf16 gradient finiteness, fast-path warning, and vLLM/HF log-prob gap,
  before versus after; fla-only intermediate state measured too.
- [x] (12:20Z) Smoke stages import fla; retained-asset publisher workflow for causal-conv1d; docs.
- [x] (13:00Z) Scope widened on request: fla-core added to the veRL backend project
  (`verl-py313/release/pyproject.toml`, `uv lock --python 3.13.12`, `backend-constraints.txt`, digests in
  `verl-py313/profile.toml`) and the transform kind (`tools/quantization`, `profiles/transform.txt`,
  `locks/transform.lock.txt`, catalog `quantization.yaml` digest); smoke imports extended.
- [x] (13:30Z) Built both kinds locally; transform smoke and the veRL release gate
  (`--release --source-checkout --verify-remote`, real Bake smoke) pass; GPU forward+backward in each image with
  the causal-conv1d wheel: warning gone, finite bf16/fp16 gradients, 0.36-0.38 s versus 1.70-1.72 s.
- [x] (13:40Z) `tools/kernel-wheels/causal-conv1d/adopt.py`: after publication, one command adds causal-conv1d
  wherever fla-core is and regenerates every derived lock (separate commit).
- [ ] Publish causal-conv1d to `carbonteq/dev` (blocked on credentials and GitHub release; see Outcomes).
- [ ] Run `adopt.py`, the ladder, rebuild and publish the kind images, regenerate `published.toml`.

## Surprises & Discoveries

- Observation: with the torch fallback, fp16 LoRA training produces NaN gradients on every step; the fast
  kernels fix it.
  Evidence: `T=2048 float16 ... grads 372 finite False |g| nan` before; `finite True |g| 41.98` after. A
  backward hook trace puts the first non-finite gradient inside layer 14's Gated DeltaNet
  (`layers.14.linear_attn.conv1d` grad_output), i.e. in the torch recurrence's backward.
- Observation: fla-core alone delivers most of the speedup and the fp16 fix; causal-conv1d removes the last
  ~7%. Transformers still prints the "fast path is not available" warning with fla alone, because
  `is_fast_path_available` requires both packages, although it then uses fla's chunk kernel and fused norm.
  Evidence: 4,096-token bf16 step 5.00 s (none), 1.47 s (fla only), 1.37 s (both).
- Observation: the vLLM-to-HF per-token log-prob gap is not caused by the kernel difference.
  Evidence: vLLM bf16 vs HF bf16 mean 0.0137 → 0.0134, p99 0.124 → 0.124; vLLM fp16 vs HF fp16 mean
  0.0019 → 0.0020. HF bf16 after vs HF bf16 before differ by 0.0139 on the same tokens: bf16 rounding
  noise dominates.
- Observation: upstream causal-conv1d hard-codes its `-gencode` list (sm75/80/87/90/100/103/110/120/121 on
  CUDA 13), so PyTorch ignores `TORCH_CUDA_ARCH_LIST`. `NVCC_APPEND_FLAGS` adds sm86 and sm89 without
  patching the source.
- Observation: the wheel build is hermetic but not bit-reproducible. A second build produced the same-size
  wheel with SHA-256 `b1a7c3ad…22e0f`; nvcc embeds per-process `tmpxft_<pid>` ids (0xa8 vs 0x98) into the
  object code.
- Observation: fla autotunes on first use (about 50 s for the three dtypes in the kernel test); with a
  persistent `TRITON_CACHE_DIR` later processes start in about a second.
- Observation: installing fla-core adds a `# via fla-core` comment to the `einops` entry of the eval, serve
  and vllm-common locks, changing their bytes and therefore their lock digests. Their contents are the same,
  but the next image publish rebuilds those kinds (mostly from cache).
- Observation: the benchmark first ran with the model in eval mode, so gradient checkpointing was inactive and
  every step ran out of memory. `model.train()` fixes it; Transformers applies checkpointing only in training
  mode.

## Decision Log

- Decision: select fla-core, not flash-linear-attention.
  Rationale: Transformers imports only `fla.modules` and `fla.ops`, which fla-core 0.5.2 ships; its only hard
  dependency is einops. flash-linear-attention adds `fla.layers`/`fla.models` and a Transformers requirement
  Posttrain does not need. 0.5.2 is the newest release (2026-07-27), requires Triton >= 3.3, and passed the
  sm86 checks with Triton 3.7.1.
  Date/Author: 2026-09-28, Claude.
- Decision: rebuild upstream causal-conv1d v1.7.0 unmodified, version `1.7.0+cu130torch2.13`.
  Rationale: no upstream wheel exists for torch 2.13; the extension uses the PyTorch C++ API and must match
  it. A local version label records the build target the same way upstream names its own wheels. No source
  patch is needed: sm86/sm89 are added through `NVCC_APPEND_FLAGS`.
  Date/Author: 2026-09-28, Claude.
- Decision: build inside the published TRL kind image rather than a separate CUDA devel image.
  Rationale: that image holds the exact torch, pip CUDA 13.0.88 toolkit, CPython and glibc the wheel will run
  with; FlashInfer already JIT-compiles against the same toolkit there.
  Date/Author: 2026-09-28, Claude.
- Decision: pin fla-core now; add causal-conv1d to the lock only after it is published.
  Rationale: `uv.lock` and the runtime locks must reference immutable, resolvable artifacts. The wheel can only
  reach `carbonteq/dev` through the retained-asset route, which needs a GitHub release and the protected
  runner. fla-core alone already gives about 93% of the speedup and fixes fp16.
  Date/Author: 2026-09-28, Claude.
- Decision: do not push kind images or regenerate `published.toml` in this change.
  Rationale: the images would have to be rebuilt again once causal-conv1d lands; publishing once, with both
  packages, avoids an intermediate registry state. Local builds prove the locks install and the smoke checks
  pass.
  Date/Author: 2026-09-28, Claude.
- Decision (superseded): leave the veRL and transform kinds unchanged.
  Superseded the same day at the user's request: both now carry fla-core through their own locks and gates.
  Eval and serve still run Qwen3.5 only through vLLM, which vendors the kernels, so they are unchanged.
  Date/Author: 2026-09-28, Claude.
- Decision: in the veRL image, install fla-core in the backend project, not the control environment.
  Rationale: the HF FSDP actor runs in `/opt/posttrain-verl`; the control environment only preflights
  environments. The backend shares torch and Triton with the control environment through the
  `shared-heavy.toml` fallback, so fla-core's Triton kernels use the same Triton 3.7.1.
  Date/Author: 2026-09-28, Claude.
- Decision: stage causal-conv1d adoption as a script (`adopt.py`) rather than a commit of manifest edits.
  Rationale: `uv lock` cannot resolve causal-conv1d until the index serves it, so a commit editing the
  manifests now would leave `uv sync --locked` and Quality broken on the branch. The script refuses to run
  until `carbonteq/dev` serves the retained bytes, then makes all manifest edits and regenerates every lock
  from the index (never a local path).
  Date/Author: 2026-09-28, Claude.

## Outcomes & Retrospective

Delivered: fla-core in the TRL and supervised images (locks, smoke checks), a reproducible causal-conv1d
builder script, the retained-asset publisher workflow, the consumer page
`docs/tooling/linear-attention-kernels/README.md`, and the qualification script
`scripts/qualification/qwen35_gdn_kernels.py`. Measured on Qwen3.5-0.8B, RTX 3070 Ti, LoRA rank 16:

    tokens  bf16 before  bf16 fla-only  bf16 both  fp16 before        fp16 both
    2,048   2.09 s       0.74 s         0.68 s     2.02 s, NaN grads  0.70 s
    3,072   3.38 s       1.08 s         1.00 s     3.37 s, NaN grads  1.04 s
    4,096   5.00 s       1.47 s         1.37 s     4.83 s, NaN grads  1.41 s

The veRL backend and transform images, with the causal-conv1d wheel, run a Qwen3.5-0.8B decoder
forward+backward on 2,048 tokens (tied embedding frozen, loss on the last 256 tokens) in 0.38 s bf16 / 0.38 s
fp16 (veRL) and 0.38 s / 0.36 s (transform), against 1.72 s and 1.70 s bf16 in the published images, with the
warning gone and finite gradients.

Blocked: causal-conv1d publication. The repository's route for CarbonTeq-built artifacts is: attach the wheel
to an immutable GitHub Release of a CarbonTeq repository, then dispatch the repository-owned publisher on the
`lan-release` runner, which holds `UV_PUBLISH_USERNAME`/`UV_PUBLISH_PASSWORD` for `https://pypi.lan/carbonteq/dev/`.
None of those credentials or GitHub rights were available to this work. A maintainer must:

1. create `carbonteq-ai/causal-conv1d` (mirror of `Dao-AILab/causal-conv1d`) with a `CARBONTEQ_FORK.md`
   stating "no source delta; binary rebuild of v1.7.0 for torch 2.13.0+cu130 by
   tools/kernel-wheels/causal-conv1d/build.sh";
2. create release `carbonteq-v1.7.0+cu130torch2.13` with the retained wheel
   (SHA-256 `b69f39142ac88cac91cba5f954cb616c50bc49933cd84f349319420470a4947a`);
3. merge this branch so `.github/workflows/publish-causal-conv1d-internal.yml` is on `main`, then run
   `gh workflow run publish-causal-conv1d-internal.yml -f release_tag='carbonteq-v1.7.0+cu130torch2.13'
   -f wheel_sha256=b69f39142ac88cac91cba5f954cb616c50bc49933cd84f349319420470a4947a`.

Then run `uv run python tools/kernel-wheels/causal-conv1d/adopt.py`, the ladder, and the image publish.

A maintainer who already holds devpi credentials could instead upload the same file directly with
`UV_PUBLISH_USERNAME=... UV_PUBLISH_PASSWORD=... uv publish --publish-url https://pypi.lan/carbonteq/dev/
causal_conv1d-1.7.0+cu130torch2.13-cp313-cp313-linux_x86_64.whl`, but that bypasses the retained-release
record the fork policy requires.

## Context and Orientation

Runtime images are built in three levels (see
`packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/README.md`): a base
image, one "kind" image per runtime variant, and per-job images derived from a kind digest. Kind images
install packages from hand-written profiles in `.../posttrain-job-kinds/profiles/*.txt` under hash-locked
constraint files in `.../locks/*.lock.txt`. Those locks are generated, never hand edited: `uv.lock` at the
repository root is the solver authority, `uv run posttrain-release lock-runtime-dependencies` exports it to
`locks/workspace.lock.txt` and projects per-image closures for each profile root
(`apps/release/src/posttrain_release/runtime_lock.py`). A profile root must also appear in `uv.lock`, which is
why fla-core was added to the `trl` extra of `packages/train/pyproject.toml`.

`profiles/supervised.txt` is included by `profiles/online-rl-trl-py312.txt`, so one pin covers both kinds.
`Dockerfile` builds `supervised` and `transform`; `Dockerfile.vllm` builds `online-rl-trl-py312`, `eval` and
`serve`; each has a `*-smoke` stage that imports the key packages. `packages/runtime-images/src/posttrain/
runtime_images/published.toml` records the registry digests of the published kind images and is regenerated
only by `posttrain-release images publish` after a real push.

Internal wheels live on the devpi index `https://pypi.lan/carbonteq/dev/+simple/` (named `carbonteq-dev` in
the root `pyproject.toml`). Forks and CarbonTeq-built artifacts reach it only through repository workflows
such as `.github/workflows/publish-renderers-internal.yml`, which download an immutable GitHub Release asset,
check its SHA-256, and upload it from the self-hosted `lan-release` runner (`docs/tooling/forks.md`,
`docs/publishing.md`).

## Plan of Work

Milestone 1 (done) selects and pins fla-core: edit the `trl` extra in `packages/train/pyproject.toml`, add
`fla-core==0.5.2` to `profiles/supervised.txt`, regenerate `uv.lock` and runtime locks, and extend both smoke
stages to import `fla.modules` and `fla.ops.gated_delta_rule`.

Milestone 2 (done) builds causal-conv1d reproducibly with `tools/kernel-wheels/causal-conv1d/build.sh` and adds
`.github/workflows/publish-causal-conv1d-internal.yml`, registered in
`apps/release/tests/test_release.py::test_retained_fork_candidates_use_development_before_server_side_promotion`.

Milestone 3 (done) qualifies both packages on the local GPU with `scripts/qualification/qwen35_gdn_kernels.py`.

Milestone 3b (done) repeats Milestone 1 for the veRL backend (`verl-py313/release/pyproject.toml`; regenerate
with `uv lock --python 3.13.12` and the `uv export ... --output-file backend-constraints.txt` command in
`verl-py313/release/README.md`, then update both digests in `verl-py313/profile.toml`) and for the transform
kind (`tools/quantization/pyproject.toml`, `uv lock --project tools/quantization`, the `uv export --project
tools/quantization ... --output-file .../locks/transform.lock.txt` command from `.github/workflows/quality.yml`,
and `uv run posttrain-release lock-dependencies`, which refreshes the catalog `quantization.yaml` digest).

Milestone 4 (blocked) publishes causal-conv1d, then `tools/kernel-wheels/causal-conv1d/adopt.py` adds
`"causal-conv1d==1.7.0+cu130torch2.13; sys_platform == 'linux' and platform_machine == 'x86_64'"` to the `trl`
extra with `causal-conv1d = { index = "carbonteq-dev" }` in `[tool.uv.sources]`, adds
`causal-conv1d==1.7.0+cu130torch2.13` to `profiles/supervised.txt`, adds `causal_conv1d` to both smoke imports,
also covers the veRL backend and transform projects, reruns the lock commands; then rebuild and publish the kind images (`docs/publishing.md` step 7), and commits the
regenerated `published.toml`.

## Concrete Steps

Build the wheel (repository root; Docker and access to `registry.lan` required, about 8 minutes):

    tools/kernel-wheels/causal-conv1d/build.sh <work-dir>
    # ... Successfully built /out/causal_conv1d-1.7.0+cu130torch2.13-cp313-cp313-linux_x86_64.whl
    # <sha256>  causal_conv1d-1.7.0+cu130torch2.13-cp313-cp313-linux_x86_64.whl

Regenerate locks after a dependency edit (repository root):

    uv lock
    uv run posttrain-release lock-runtime-dependencies
    # runtime dependency locks updated: ... narrow closures: ...supervised.lock.txt, ...online-rl-trl-py312.lock.txt, ...

Build the kind image locally without pushing (from `packages/runtime-images/src/posttrain/runtime_images`):

    docker buildx bake --builder posttrain-builder \
      --file containers/posttrain-job-kinds/docker-bake.hcl \
      --set "*.args.POSTTRAIN_BASE_IMAGE=registry.lan/carbonteq/posttrain-base@sha256:1083e11ce1c540d8d7816f1c8ccd15011012ebe8771f0ecd6d5a685c4921d76b" \
      --set "*.args.VERSION=qwen-kernels-local" \
      --set "posttrain-kind-online-rl-trl-py312.output=type=docker" \
      --set "posttrain-kind-online-rl-trl-py312.tags=posttrain-kind-online-rl-trl-py312:qwen-kernels-local" \
      posttrain-kind-online-rl-trl-py312 posttrain-kind-online-rl-trl-py312-smoke posttrain-kind-supervised-smoke

Until causal-conv1d is on the index, a local-only derived image adds the wheel (a two-line Dockerfile:
`FROM posttrain-kind-online-rl-trl-py312:qwen-kernels-local`, copy the wheel, check its SHA-256, and
`uv pip install --python "${VIRTUAL_ENV}" --no-deps --no-index <wheel>`).

Run the qualification (one GPU job at a time; check `nvidia-smi --query-compute-apps=pid --format=csv,noheader`
is empty first). `WORK` holds the script and `samples_bfloat16.json`/`samples_float16.json` (vLLM samples with
per-token sampler log-probs of the 64-prompt set):

    docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
      -e HOME=/work/home -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRITON_CACHE_DIR=/work/home/triton \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -v "$HOME/.cache/huggingface:/hf:ro" -v "$WORK:/work" -w /work \
      --entrypoint /opt/posttrain/venv/bin/python <image> qwen35_gdn_kernels.py kernels
    # ... gated delta rule bfloat16: ... rel err 5.6e-03 ...; causal conv1d ...; PASS

For the veRL image use `--user 0:0 --entrypoint /opt/posttrain-verl/bin/python` (the backend interpreter lives
in root's uv Python store) and `train-step bfloat16`; the transform image uses the default entrypoint and
`train-step`. Replace the last arguments with `actor bfloat16 after`, `score bfloat16 after` (and `float16`, and `before`
with the published image) and finally `gap`.

## Validation and Acceptance

Acceptance on a CUDA GPU with the new image: `kernels` prints PASS; loading Qwen3.5 no longer logs "The fast
path is not available"; `actor` reports finite gradients in bf16 and fp16 and at least 2x lower step time
than the published image at 2,048-4,096 tokens; `gap` shows the vLLM-to-HF gap no larger than before. All of
these held (see Outcomes and Artifacts).

Repository ladder, from the repository root:

    uv sync --all-packages --locked --group dev --extra trl --python 3.13
    uv run ruff check .                    # All checks passed!
    uv run pyright                         # 1 error, pre-existing in packages/train/tests/test_adaptive_curriculum.py
    uv run lint-imports                    # Contracts: 9 kept, 0 broken.
    uv run pytest -q
    uv run posttrain-release lock-runtime-dependencies --check
    git diff --check

Without `--extra trl` the TRL test modules fail to import `trl`; CI installs that extra.

## Idempotence and Recovery

The build script re-uses an existing clone, verifies the pinned commit and a clean tree, and replaces `dist/`
on every run. Lock regeneration is deterministic from `uv.lock`; `--check` reports drift without writing.
Local image builds tag only `*:qwen-kernels-local*` names and push nothing; remove them with `docker rmi`.
To back out fla-core, revert the `trl` extra and profile lines and rerun the two lock commands.

## Artifacts and Notes

    wheel  causal_conv1d-1.7.0+cu130torch2.13-cp313-cp313-linux_x86_64.whl  250,584,174 bytes
    sha256 b69f39142ac88cac91cba5f954cb616c50bc49933cd84f349319420470a4947a
    source https://github.com/Dao-AILab/causal-conv1d  v1.7.0  cd81f0413cad2fc1e6f17e785ac39f59aae690cd
    SASS   sm_75 sm_80 sm_86 sm_87 sm_89 sm_90 sm_100 sm_103 sm_110 sm_120 sm_121
    fla-core 0.5.2 wheel sha256 5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761

    local images (not pushed)
      posttrain-kind-online-rl-trl-py312:qwen-kernels-local       sha256:b7d0d4124a9eccc88303fb78e411ddc1c00da11988892f65a16bcd2df443103c
      posttrain-kind-online-rl-trl-py312:qwen-kernels-local-cc1d  sha256:09db9e0d7cd353e430e5aa7be23a2aa78ce5be0f346668317264b9356e30deae
      posttrain-kind-supervised:qwen-kernels-local                sha256:7a5b486f942a042ac193b504d819380731028ed89312f3288998dd5b17dff872
      posttrain-kind-online-rl-verl-py313:qwen-kernels-local      sha256:f7c0bae53fc03bd483b16629b22078deb42f1f0d1d5da65fac40fddcecaef235
      posttrain-kind-online-rl-verl-py313:qwen-kernels-local-cc1d sha256:5ac36c3c049903fb99e114d595c1f46c0485e54d2d01c4784d46b0cfbf2cb0b5
      posttrain-kind-transform:qwen-kernels-local                 sha256:b6e198ba484ca15928591abfa99fba6208996963951b201cbe1bc7a8c7c947dd
      posttrain-kind-transform:qwen-kernels-local-cc1d            sha256:d7716153786fe4ff08209343ef064c915b920d17fb57e2a1dc1d4900cc55cf8e

    veRL backend (root; its uv-managed Python is in root's store) and transform, train-step bfloat16:
      after:  fast path True warning False T=2048 bfloat16 grads 319 finite True step 0.361s peak 2.53GiB  PASS
      before: fast path False warning True ... finite True step 1.721s (veRL) / 1.700s (transform) peak 2.99GiB

    kernels (after image, sm86): gated delta rule rel err bf16 5.6e-3, fp16 7.1e-4, fp32 1.5e-3;
      causal conv1d bf16 3.1e-3, fp16 3.7e-4, fp32 5.3e-8; gated RMSNorm 2.8e-3; PASS
    vLLM in the new image: 64/64 identical completions, max |dlogp| 0.0 versus the published image's samples.

## Interfaces and Dependencies

`posttrain-train[trl]`, `tools/quantization` (`posttrain-awq-runtime`) and the veRL backend project
(`posttrain-kind-online-rl-verl-py313`) require `fla-core>=0.5.2,<0.6` (Triton kernels imported by Transformers as
`fla.modules.FusedRMSNormGated` and `fla.ops.gated_delta_rule.{chunk,fused_recurrent}_gated_delta_rule`).
After Milestone 4 it will also require `causal-conv1d==1.7.0+cu130torch2.13` from `carbonteq-dev`
(`causal_conv1d.causal_conv1d_fn`, `causal_conv1d_update`). No Posttrain code imports either package; they
are runtime accelerators discovered by Transformers.

Revision note (2026-09-28): widened to the veRL and transform kinds and added `adopt.py` at the user's
request; the earlier decision to defer them is marked superseded.
