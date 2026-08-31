# Reduce runtime image size without sacrificing OCI cache reuse

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current as work proceeds.

This repository does not contain `.agents/PLAN.md`. This document follows `docs/templates/PLAN.md`, the repository's checked-in ExecPlan authority, and must continue to do so. It is a focused continuation of the runtime-image composition and parent-preservation work described in `docs/plan/portable-runtime-image-supply-chain.md`; that broader plan remains the authority for the overall release and actual-job publication lifecycle.

## Purpose / Big Picture

Remote GPU workers should pull an immutable Posttrain job image directly from an OCI registry while reusing every base and job-kind blob already in their local content store. The current layout already makes an actual job a small suffix over a digest-pinned kind image, but the veRL kind installs a second Python environment that repeats several gigabytes of byte-identical PyTorch, Triton, and NVIDIA files already present in its base image.

After this work, the control environment at `/opt/posttrain/venv` and the veRL backend environment at `/opt/posttrain-verl` remain separately locked and independently discoverable. The backend uses uv's supported partial-sync interface to omit an explicit allowlist of compatible heavy packages, then resolves those packages from the inherited control environment through one lower-precedence `.pth` fallback. Backend-local packages remain first, so version-different dependencies remain isolated. A release operator can demonstrate the result by comparing OCI manifests, running both Python import gates and a GPU canary, and showing that a second worker pull transfers no inherited base or kind blobs.

This work does not introduce stronger compression, forced recompression, a new layer encoding, or mutable image tags. It optimizes filesystem composition while preserving ordered parent layer descriptors byte-for-byte.

## Progress

- [x] (2026-08-28 23:39Z) Read the canonical runtime-image ownership and API boundaries, the existing portable-runtime plan, current Dockerfiles, locks, validation scripts, and publication tests.
- [x] (2026-08-28 23:39Z) Verified the current cache-preservation suite: `96 passed, 1 skipped` across runtime-images and the relevant execution-buildkit tests.
- [x] (2026-08-28 23:39Z) Pulled the exact published veRL digest and confirmed Docker reused all ten base layers instead of transferring them again.
- [x] (2026-08-28 23:39Z) Measured the two live environments and found 112 exact-name, exact-version overlapping distributions representing 5,071,292,674 backend bytes before OCI compression.
- [x] (2026-08-28 23:39Z) Rejected a whole-control-environment `PYTHONPATH` prototype after it reproduced an ANTLR/OmegaConf incompatibility.
- [x] (2026-08-28 23:39Z) Proved a narrower prototype: 17 exact-version PyTorch, Triton, and NVIDIA distributions contained 4,642,585,905 byte-identical files, and the existing veRL import gate passed when only those files were shared.
- [x] (2026-08-29 00:18Z) Audited every other published kind by exact digest and measured its OCI descriptors, installed distributions, bytecode, toolchain layers, and cross-image overlap.
- [x] (2026-08-29 00:18Z) Proved that serve is an exact dependency subset of eval and TRL: 176 non-base distributions and 50,011 payload files are byte-identical across all three images.
- [x] (2026-08-29 00:07Z) Implemented and measured an initial file-linking prototype, then retained it only as evidence after identifying a simpler uv-native design.
- [x] (2026-08-29) Replaced file mutation with uv `--no-install-package` partial sync, one fallback-only `.pth` file, a read-only lock/runtime validator, and focused tests. The real image build omitted all 17 shared packages and validated their inherited versions.
- [x] (2026-08-29) Built two independent uncached OCI veRL candidates with the exact published base prefix. The byte-identical archives remove 2,763,287,505 compressed after-base bytes and change no registry state.
- [x] (2026-08-29) Preserved precompiled bytecode while making it deterministic: uv installation compilation is disabled, then both environments are explicitly compiled in checked-hash mode with a fixed source epoch.
- [x] (2026-08-29) Diagnosed and eliminated uv install nondeterminism from overlapping CUTLASS base/CUDA 13 wheels and wall-clock `uv_cache.json` metadata. Two uncached OCI exports now have the same archive, config, manifest, and ordered layers.
- [x] (2026-08-30) Ran the actual-job projection and source-build regression gates. The published canary passed offline runtime qualification and produced `registry.lan/carbonteq-qualification/posttrain-job@sha256:b832ca17f8beeaf8848b774c174c7f4670708db5e6fbdb3f5362ef21124595af` with 24 inherited layers and an 1,704,638-byte compressed job suffix.
- [x] (2026-08-30) Qualified and published the optimized veRL runtime at `registry.lan/carbonteq/posttrain-kind-online-rl-verl-py313@sha256:fea65ed6037f52f44f5f901fb1b95c5888fe6d47c8799e0dbcdc6f1318add28d`. Its exact ten-layer base prefix is unchanged, a repeat pull transferred no layers, FlashInfer compiled its sampling extension, CUDA matmul and single-rank NCCL passed, and vLLM loaded Qwen3.5-0.8B and generated one token on the Pop!_OS RTX 3070 Ti with 8 GiB VRAM.
- [x] (2026-08-30) Published and qualified the additive cloud-compatible successor `sha256:9bf4ff554416e34fdd0a484102c243bd470054212111ab65a0864015a860df38`. It retains the exact complete `sha256:fea65ed...` layer sequence, appends only NVIDIA's pinned 104,195,781-byte CUDA 13 compatibility payload, and does not globally set `LD_LIBRARY_PATH`. Local CUDA passed with the host driver, while explicit compatibility activation passed on a RunPod A100 80 GB PCIe Secure Cloud spot Pod whose driver exposes CUDA 12.8.
- [ ] Prototype a shared vLLM parent for serve, eval, and TRL; retain all packages and prove the new layer topology before considering removals.
- [x] (2026-08-29) Scoped out hardware-specific variants, Trackio dependency changes, bytecode removal, and toolchain removal. Retain one hardware-portable lineage, mandatory `pyturso`, precompiled bytecode, and the current runtime tools.
- [x] (2026-08-30) Recorded the accepted digest, local GPU evidence, exact compressed-size reduction, cache behavior, and publication decision. A separate dstack cloud pull remains part of the registry-routing plan rather than this image-composition gate.

## Surprises & Discoveries

- Observation: the current published base is 2.608 GiB compressed, while the veRL kind is 6.742 GiB and adds 4.134 GiB after the exact base.
  Evidence: the committed manifest pins base `sha256:55987a566f5fa7b3fd8af219b2a2dc51315219e2ca82f597f6c940bc7332d595` and veRL `sha256:fbf19d562293c0b38e7febfbb80c2786ab261a85880536a3a32c228c4e91a9b6`; live OCI descriptor sums reproduce those totals.

- Observation: pulling the exact veRL digest reported every base layer as `Already exists` and transferred only veRL-specific layers.
  Evidence: Docker reused the ordered base descriptors through `sha256:fdd51874e...`. This proves stable parent digests already provide the cache behavior that this plan must preserve.

- Observation: the unpacked control environment occupies 5,698,453,504 bytes and the backend environment occupies 10,671,169,536 bytes under their respective `lib/python3.13` trees.
  Evidence: read-only `du -x -B1` inside the published veRL image.

- Observation: 112 distributions have the same normalized name and version in both environments and account for 5,071,292,674 bytes in the backend environment. The largest matches are PyTorch 2.11.0+cu130, Triton 3.6.0, cuBLAS, cuDNN, cuFFT, cuSolver, cuSPARSELt, NVRTC, NCCL, and cuSPARSE.
  Evidence: `importlib.metadata.distributions(path=[...])` over both installed site-package directories, with sizes calculated from each distribution's installed file list.

- Observation: adding the complete control site-packages directory to the backend's `PYTHONPATH` is incorrect even though many versions match.
  Evidence: veRL imported its OmegaConf 2.3.1 generated parser but resolved ANTLR from the control environment; deserialization failed with `Could not deserialize ATN with version 3 (expected 4)`.

- Observation: selective file sharing is technically plausible without merging environments.
  Evidence: an ephemeral-container prototype compared SHA-256 for candidate files, replaced only byte-identical files from 17 exact-version heavy distributions with links to the control copy, shared 4,642,585,905 bytes, and passed imports for Ray, PyTorch, Transformers, TensorDict, veRL, Verifiers, and vLLM. This is CPU/import evidence only, not a GPU qualification.

- Observation: the implemented fail-closed tool accepts the exact published veRL image and identifies 12,540 regular non-metadata payload files totaling 4,641,407,736 logical bytes with manifest SHA-256 `76777056be24f24faeda4b8c6de46b694b6911beba8daf2d8b39f5a81b10c281`.
  Evidence: audit-only and ephemeral `--apply` runs against published digest `sha256:fbf19d...` produced the same counts and digest. Both isolated Python import gates passed after link application. The small difference from the initial prototype is excluded package metadata and environment-local scripts.

- Observation: custom file replacement is unnecessary. uv supports repeated `--no-install-package` options during a frozen sync, allowing the backend lock to remain complete while selected packages are deliberately absent from the backend environment.
  Evidence: a real BuildKit smoke build installed 242 backend distributions rather than 259, reported `shared-fallback: packages=17`, and did not download or install the omitted PyTorch, Triton, or NVIDIA runtime packages. The fallback validator proved each omitted package exists in the inherited control environment at the exact version selected by the backend lock.

- Observation: uv's `--link-mode=symlink` is not the stable replacement for the prototype.
  Evidence: uv's official CLI documentation warns that symlink mode tightly couples environments to the cache and that cache cleanup breaks installed packages. `--system-site-packages` also does not make uv treat inherited packages as satisfied. The chosen partial-sync path uses a documented omission interface and keeps the immutable base environment as the runtime authority.

- Observation: the first real smoke reached and imported the full backend stack, then exposed that the vLLM source build inferred `0.25.2.dev2+precompiled` instead of the profile's fork-qualified runtime version.
  Evidence: the vLLM fork supports `VLLM_VERSION_OVERRIDE`. Supplying the already-qualified `0.25.2.dev2+g7817d8457.precompiled` during frozen sync made installed metadata deterministic and the unchanged assertion pass. The final real smoke passed both backend and control Python import gates.

- Observation: the final uv-native OCI candidate preserves the exact ten-descriptor base prefix, exceeds the 1 GiB compressed saving gate, and is reproducible across independent uncached builds.
  Evidence: both disposable exports are byte-for-byte identical at archive SHA-256 `efd8884c5086b2b5518cab0bf98d97cfab8bc3b8aeed9199b75f7e801924d357` and contain platform manifest `sha256:092c28f554dd0c2ebb67c2813677417b7b349537e405775af5f14f64a7c1bf9c`. The manifest totals 4,475,998,088 bytes, of which 1,675,310,674 bytes follow the 2,800,687,414-byte base. The published image has 4,438,598,179 after-base bytes, so the candidate saves 2,763,287,505 bytes (about 2.573 GiB). Nothing was pushed or written to `published.toml`.

- Observation: import-only qualification was insufficient because FlashInfer runtime JIT requires a coherent CUDA compiler, headers, and conventional library layout.
  Evidence: the first GPU candidate reached vLLM but lacked `nvcc`; the next carried CUDA 13.2 compiler packages against the inherited CUDA 13.0 runtime and failed version validation. The accepted lock pins the backend compiler, CRT, and NVVM to CUDA 13.0, keeps the runtime library backend-local, exposes inherited headers through `CPATH`, and supplies conventional `lib64`, `libcudart.so`, `ninja`, and `nvcc` paths. FlashInfer JIT and the full Qwen vLLM gate then passed without command-line environment overrides.

- Observation: the accepted image is 4,651,350,481 compressed layer bytes and adds 1,850,663,067 bytes after the unchanged 2,800,687,414-byte base.
  Evidence: live OCI descriptor inspection of `sha256:fea65ed6...` shows all first ten descriptors equal the published base, while the former veRL manifest was 7,239,285,593 bytes. The accepted image saves 2,587,935,112 compressed bytes. Across all seven published runtime roots the deduplicated union is 9,217,281,255 bytes in 50 unique layers, below the 10 GB R2 free-storage allowance before actual-job suffixes.

- Observation: CUDA forward compatibility must be available but inactive by default.
  Evidence: globally setting `/usr/local/cuda-13.0/compat` in `LD_LIBRARY_PATH` caused CUDA error 803 on the newer local display driver. Removing that global setting restored the local gate; setting it only in the RunPod task allowed the same CUDA 13 runtime to execute on the A100 host's older data-center driver. The final child is 4,755,546,262 compressed layer bytes and its first 24 descriptors are byte-for-byte identical to `sha256:fea65ed...`.

- Observation: removing Git metadata required one source-identity contract at image-build and runtime-qualification time.
  Evidence: the actual-job canary found that the Dockerfile accepted `.posttrain-source-revision` while `posttrain-runtime qualify` still called `git rev-parse`. Both paths now prefer the immutable revision marker, reject marker-plus-Git ambiguity, and retain a legacy Git fallback. The same canary also proved the veRL projection must contain `common`, `data`, `environment`, and `train` because the train package imports environment contracts.

- Observation: the former 24 GiB Pop!_OS qualification target was stale inventory, not a current hardware requirement.
  Evidence: the accepted bounded runtime gate ran on the host's NVIDIA GeForce RTX 3070 Ti with 8 GiB VRAM. Production workload memory requirements remain job-specific; this image gate proves runtime compatibility and does not weaken a job's requested capacity.

- Observation: uv's parallel install is nondeterministic when two wheels own overlapping paths, even when both artifacts and versions are hash-pinned.
  Evidence: `nvidia-cutlass-dsl-libs-base==4.5.2` and `nvidia-cutlass-dsl-libs-cu13==4.5.2` both populate `nvidia_cutlass_dsl/`. Independent syncs alternated between their source and native payloads. Reinstalling the exact hash-checked base wheel followed by the CUDA 13 wheel made all runtime payload files identical and retained the CUDA 13 implementation.

- Observation: explicit deterministic compilation preserves runtime bytecode but uv records wall-clock installation metadata for directly reinstalled wheels.
  Evidence: after ordering the CUTLASS wheels, the only remaining content differences were `uv_cache.json` timestamps and their hashes in the two `RECORD` files. Removing those uv-internal cache records and their corresponding RECORD entries produced byte-identical OCI archives. The image still runs `compileall --invalidation-mode checked-hash` over the control site-packages, backend standard library, and backend site-packages.

- Observation: repository-wide validation currently has failures outside this plan's files in the pre-existing dirty worktree.
  Evidence: focused validation passes, but full Ruff and Pyright report an undefined `_tree_digest` in `packages/execution-pack/src/posttrain/execution_pack/service.py` and stale `ExecutionProvider.logs` test doubles in `packages/execution/tests/test_service.py`. Full pytest reports 1 failure among 1,292 tests because `packages/catalog/src/posttrain/catalog/base/locks.toml` contains a dependency-lock digest that differs from the current `uv.lock`. These files were already modified outside this plan and were not changed here.

- Observation: Torch and Triton legitimately record console entry points such as `../../../bin/torchrun` relative to `site-packages`.
  Evidence: live `importlib.metadata` inspection of the published backend returned those paths. The tool recognizes only the exact three-parent `bin/<filename>` wheel-script shape and leaves it backend-local; absolute paths and every other traversal shape are rejected before application.

- Observation: approximately 5.16 GB of the backend environment remains unique or version-different. The largest unique distributions include `flashinfer-cubin` at about 1.91 GB, vLLM at about 606 MB, `tokenspeed-triton` at about 285 MB, CUTLASS libraries, Ray, TileLang, LLVMlite, CUDA compiler packages, and XGrammar.
  Evidence: the same distribution inventory excluded exact-name and exact-version matches. Large size alone is not evidence that these packages are removable.

- Observation: all six published kinds preserve the exact ten-layer base prefix. The current unique union of base plus all kinds is 11,805,216,367 compressed bytes, or 10.994 GiB; 8.386 GiB is unique content after the base.
  Evidence: live Linux/amd64 OCI manifests for the digests in `published.toml` contain 48 unique layer descriptors. Descriptor identity, not final filesystem similarity, determines registry and worker reuse.

- Observation: serve, eval, and TRL independently commit almost the same vLLM closure as three different large layer blobs. Their after-base sizes are 1,291,897,348, 1,379,787,677, and 1,440,087,049 compressed bytes, but only 19,435,697 bytes of current after-base layers are shared by digest across all three.
  Evidence: the three after-base unions occupy 4,072,900,680 compressed bytes even though their installed environments share 176 exact non-base distributions.

- Observation: serve's complete non-base distribution set is an exact subset of both eval and TRL. The shared set accounts for 4,926,723,714 distribution-attributed bytes. Excluding installation-specific metadata, all three images contain the same 50,011 payload paths, 4,727,855,275 bytes, and aggregate SHA-256 `d1d7f23ce71f88182b024a9c0edb3f1e28c9ee80fa6f81a1f8840e529e740da1`.
  Evidence: each digest-pinned image was hashed independently inside a disposable container. Eval adds about 229.7 MB unpacked beyond serve; TRL adds about 423.0 MB.

- Observation: making the current serve dependency layer the parent of thin eval and TRL deltas is estimated to remove about 2,544,923,302 compressed bytes, or 2.37 GiB, from the three-kind registry union without removing a package.
  Evidence: the estimate is the current three-kind after-base union minus serve's after-base bytes and the observed eval-minus-serve and TRL-minus-serve descriptor totals. It is a planning estimate; only a rebuilt disposable candidate can establish actual descriptors.

- Observation: transform and supervised are already comparatively small, adding 216,605,395 and 277,624,663 compressed bytes over base. They share 350,062,028 unpacked distribution bytes at matching versions, while all five non-veRL kinds share only 150,907,046 such bytes.
  Evidence: package inventories show that a new universal common-Python layer would save much less than the vLLM parent and would rebuild every kind. Transform is also the only normal kind that upgrades base packages, changing `cuda-pathfinder`, `filelock`, and `setuptools`; the overwritten content is small.

- Observation: `pyturso` occupies about 98 MB unpacked, and Trackio imports its SQLite-compatible storage path even for remote runs because that path supplies the durable local retry buffer.
  Evidence: `carbonteq-trackio` declares `pyturso` as a mandatory dependency; `trackio.__init__`, `run.py`, and `sqlite_storage.py` import and use `SQLiteStorage`. It is used by every job and remains part of the required runtime rather than an optimization target. Keep `boto3` because S3 artifact/checkpoint publication remains required.

- Observation: globally compiled `.pyc` files add roughly 133 MB unpacked in transform, 140 MB in supervised, 219 MB in serve, 267 MB in eval, and 276 MB in TRL beyond the base's 108 MB.
  Evidence: exact filesystem sums under `/opt/posttrain/venv`. These files remain intentionally precompiled to protect cold-start speed on short-lived spot workers.

- Observation: the final base's compiler/Git/development layer is 104,195,236 compressed bytes and 285 MB unpacked. Git itself reports about 44.9 MB installed and `libc6-dev` about 12.0 MB; the vLLM kinds' separate `g++` layer is 18,835,723 compressed bytes and is already shared by digest.
  Evidence: OCI history and `dpkg-query` inside the current base. The runtime toolchain remains intentionally available for source builds and runtime JIT paths; it is not an optimization target in this plan.

- Observation: `flashinfer-cubin==0.6.13` is the largest single vLLM opportunity. Its wheel is 457,984,995 compressed bytes and installs about 1.91 GB, of which about 1.60 GB is tagged `sm100f` and 226 MB `sm103a`; it is overwhelmingly Blackwell-specific.
  Evidence: the locked wheel size in `uv.lock` and filename/byte inventory inside serve. Despite its size, architecture-specific variants would narrow hardware compatibility and multiply image lineages. The universal package remains unchanged.

## Decision Log

- Decision: preserve existing layer encoding and set `force-compression=false` everywhere; do not evaluate higher compression levels.
  Rationale: OCI caches identify blobs by digest. Re-encoding identical filesystem content changes the layer digest and defeats registry and worker reuse. The user's priority is cache stability, not minimum isolated-blob size.
  Date/Author: 2026-08-28 / Codex and user.

- Decision: optimize only veRL first and leave the universal base plus every unrelated kind unchanged.
  Rationale: veRL contains the demonstrated duplication. Changing the base would fan out to every kind digest and cause all workers to pull new ancestry, defeating the near-term objective.
  Date/Author: 2026-08-28 / Codex.

- Decision: retain two virtual environments and both immutable locks.
  Rationale: the canonical framework boundary and current release gate require a control environment and a separately locked veRL backend. Physical reuse of proven-identical immutable files does not authorize merging dependency resolution.
  Date/Author: 2026-08-28 / Codex.

- Decision: never share an entire site-packages directory and never use `--system-site-packages` as the implementation.
  Rationale: the failed ANTLR/OmegaConf prototype proves that directory-level precedence leaks incompatible dependencies across the environment boundary.
  Date/Author: 2026-08-28 / Codex.

- Decision: use uv partial sync for the explicit allowlist and make the control environment a lower-precedence fallback; do not mutate installed package payloads.
  Rationale: `uv sync --no-install-package` preserves the complete backend resolution while avoiding duplicate installation. Backend-local site-packages remains first, preserving incompatible ANTLR/OmegaConf versions, and a read-only validator rejects missing or version-different inherited packages. This supersedes the initial file-by-file linking design.
  Date/Author: 2026-08-29 / Codex and user.

- Decision: perform the uv partial sync, fallback creation, validation, report write, and temporary cleanup in one Docker `RUN` instruction.
  Rationale: the large packages must never be installed into a committed backend layer. A later deletion would retain their bytes in OCI ancestry.
  Date/Author: 2026-08-29 / Codex.

- Decision: no frozen product-baseline amendment is required for the proposed prototype.
  Rationale: actual-job image identity, separate backend locks, public CLI behavior, provider behavior, and evidence meaning remain unchanged. If implementation requires merging environments or changing public image semantics, stop and amend the canonical baseline before proceeding.
  Date/Author: 2026-08-28 / Codex.

- Decision: make shared vLLM ancestry the next cross-image prototype; do not begin by deleting vLLM packages.
  Rationale: exact payload hashing proves that physical layer topology, not dependency selection, is currently wasting about 2.37 GiB of registry union. A shared parent preserves each kind's final package set and improves cache reuse when workers switch among serve, eval, and TRL.
  Date/Author: 2026-08-29 / Codex.

- Decision: do not move the 150 MB logical all-kind Python intersection into `kind-common` in the first change.
  Rationale: it would invalidate every kind for a much smaller potential saving. Transform and supervised should remain stable while the larger vLLM-only topology is qualified.
  Date/Author: 2026-08-29 / Codex.

- Decision: keep one hardware-portable runtime lineage and do not create architecture-specific FlashInfer images.
  Rationale: spot providers expose heterogeneous GPU architectures. Hardware-specific variants would increase admission risk, fragment caches, and multiply the image lineage that must be built, published, qualified, and retained.
  Date/Author: 2026-08-29 / Codex and user.

- Decision: keep `pyturso` mandatory in every job, retain precompiled bytecode, and retain the current runtime/build tools.
  Rationale: Trackio's durable local buffer is part of every job, while bytecode and toolchains protect cold-start, source-build, and runtime-JIT performance. Their modest byte reductions do not justify behavior or speed regressions.
  Date/Author: 2026-08-29 / Codex and user.

- Decision: the shared serve/eval/TRL vLLM parent is the only additional normal-kind image optimization in scope.
  Rationale: it removes duplicate physical storage and transfer across related images while preserving hardware compatibility, final dependency inventories, runtime behavior, and the number of public job-kind lineages.
  Date/Author: 2026-08-29 / Codex and user.

- Decision: do not use uv symlink link mode or retain uv's cache as runtime package storage.
  Rationale: uv documents that symlink mode is cache-coupled and can be broken by cache cleanup. The digest-pinned base environment is already the stable immutable store and needs only a plain fallback path.
  Date/Author: 2026-08-29 / Codex.

- Decision: install from uv's cache in copy mode, then normalize the two overlapping CUTLASS wheels in base-then-CUDA-13 order before bytecode compilation.
  Rationale: copy mode makes the installed environment independent of cache cleanup, while the explicit hash-checked reinstall resolves an observed parallel-install ownership race without removing CUTLASS, narrowing supported hardware, or changing runtime code. The CUDA 13 wheel intentionally wins every overlapping path.
  Date/Author: 2026-08-29 / Codex.

- Decision: keep precompiled Python bytecode and make its production deterministic rather than disabling it.
  Rationale: uv's concurrent compilation and source-build leftovers prevented stable OCI digests. A fixed epoch, stable hash seed, checked-hash `compileall`, ordered CUTLASS payload, and removal of build-only uv metadata preserve import performance while making rebuilds byte-identical.
  Date/Author: 2026-08-29 / Codex and user.

## Outcomes & Retrospective

The veRL composition change is accepted and published. The canonical runtime now points to `sha256:9bf4ff554416e34fdd0a484102c243bd470054212111ab65a0864015a860df38`, with source digest `a2a1c37481051d064c9b42b367932986e78d15465900963ba27145567b4f5e83`, backend constraint digest `688cc6f99ad98b6279421a6d98d25c7a37cef5dc1a38aaeba78fc0b425b69d0e`, and dependency-lock digest `19f96d07e64c3664b4c09d41db9b98142aa3ec3d0bdbfa3179012dd0d7d2f2ed`. The accepted boundary shares 16 exact-version CUDA/PyTorch distributions and keeps `nvidia-cuda-runtime` backend-local so the CUDA 13.0 compiler and runtime headers remain coherent. It preserves deterministic compiled bytecode and runtime JIT tools, passes FlashInfer compilation, CUDA/NCCL, Qwen3.5 vLLM generation, an actual-job publication canary, local CUDA with native host driver selection, and RunPod A100 CUDA with explicit forward-compatibility activation. The new 104,195,781-byte layer follows the previously accepted complete veRL graph, so it does not rewrite the base or any expensive kind layer.

The previously measured seven-root union was 9,217,281,255 unique compressed layer bytes. The cloud-compatibility successor adds one 104,195,781-byte unique layer, for a projected 9,321,477,036-byte release-root union, still below 10,000,000,000 bytes before actual-job suffixes and OCI metadata. Hardware-specific variants, Trackio dependency changes, bytecode removal, and toolchain removal remain out of scope. The actual-job projection gate is closed by the retained canary receipt and immutable job digest above.

## Context and Orientation

Posttrain publishes three OCI image levels. `packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-base/Dockerfile` creates the universal CUDA-enabled PyTorch base. `packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/Dockerfile` extends it with supervised, transform, eval, serve, and TRL dependencies. `packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/verl-py313/Dockerfile` creates the veRL kind. Finally, `packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job/Dockerfile` adds a bounded project/framework context to one exact kind image.

An OCI layer descriptor is a digest and byte count for one immutable layer blob. A worker cache can reuse a layer only when the descriptor digest is identical. An image's parent prefix is the ordered sequence of layer descriptors inherited from its parent. Preserving that exact prefix proves that the child did not rewrite or recompress the parent.

The universal base installs PyTorch 2.11.0+cu130, Triton 3.6.0, and CUDA 13 runtime packages into `/opt/posttrain/venv`. The veRL Dockerfile starts from that base but creates `/opt/posttrain-verl` and resolves an independent `uv.lock`. The environments must remain separately locked because veRL has backend-specific versions such as ANTLR 4.9.3 and OmegaConf 2.3.1. Nevertheless, the locks currently select byte-identical builds of the largest PyTorch/CUDA packages.

The current exact candidate allowlist is `torch`, `triton`, `nvidia-cublas`, `nvidia-cuda-cupti`, `nvidia-cuda-nvrtc`, `nvidia-cudnn-cu13`, `nvidia-cufft`, `nvidia-cufile`, `nvidia-curand`, `nvidia-cusolver`, `nvidia-cusparse`, `nvidia-cusparselt-cu13`, `nvidia-nccl-cu13`, `nvidia-nvjitlink`, `nvidia-nvshmem-cu13`, and `nvidia-nvtx`. `nvidia-cuda-runtime` remains backend-local. Names are normalized according to Python package metadata. The initial byte-for-byte prototype qualified the broader boundary; the durable build gate requires every selected shared name to remain absent from backend-local metadata, present in the digest-pinned control environment, and equal to the backend lock's selected version.

Backend-local packages and metadata remain first on `sys.path`. A plain `.pth` line appends the control site-packages path, so backend-only NVVM, CUTLASS, ANTLR, and OmegaConf packages cannot be hidden by the fallback.

`packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/validate.py` statically validates the image hierarchy. `packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/verl-py313/release_gate.py` validates the veRL fork, lock, and real container smoke. `packages/runtime-images/tests/test_verl_release_gate.py` owns repository-level veRL release tests. `packages/execution-buildkit/src/posttrain_execution_buildkit/job_image.py` proves that an actual-job image retains its kind's parent descriptors. `apps/release/src/posttrain_release/cli.py` and `publish.py` own release publication, while the runtime-images package owns definitions and the generated `published.toml` manifest.

The worktree contains unrelated user changes. Do not modify, stage, revert, or reformat them. This plan concerns only the runtime-image definition, its focused tests, release documentation, and the generated manifest after successful live qualification.

## Plan of Work

### Milestone 1: Add a fail-closed uv partial-sync fallback

Create `packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/verl-py313/shared-heavy.toml`. It lists the 17 normalized distribution names above and explains that the list is a partial-sync policy, not a dependency constraint. Versions continue to come only from the control and backend locks.

Create `packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/verl-py313/validate_shared_fallback.py`. Keep it standard-library-only. Expose pure policy, lock, and installed-metadata validation for tests, plus a CLI accepting `--control-site`, `--backend-site`, `--backend-lock`, `--policy`, `--fallback-file`, and `--report`.

The validator must read installed distributions from both explicit site-package roots, normalize names, and reject an allowlisted distribution that is missing from control, installed locally in backend despite omission, or version-different from the backend lock. It must require the `.pth` file to contain exactly one absolute control site-packages path and no executable line. It writes deterministic JSON containing the strategy and selected names/versions. Any failure exits non-zero and prevents the Docker layer from committing.

Modify the single backend-install `RUN` so frozen `uv sync` receives one `--no-install-package` per policy entry, then create the fallback `.pth`, validate it, write the report, and clean temporary inputs in the same instruction. Do not modify the base Dockerfile, generic kind Dockerfile, locks, or compression/output settings.

Extend `validate.py`, `release_gate.py`, and `packages/runtime-images/tests/test_verl_release_gate.py` to require every omission flag, the single fallback path, fail-closed validation, deterministic report, and both environment gates. Add focused tests with temporary metadata trees for exact matches, version mismatch, unexpected backend copies, executable `.pth` rejection, and deterministic output.

Acceptance for this milestone is that focused tests pass, a deliberately mismatched file fails before image creation, and the normal veRL source and lock identities remain unchanged.

### Milestone 2: Build and measure a disposable candidate

Build only the veRL candidate against the exact existing base digest. Use an ephemeral registry or OCI layout with a temporary receipt directory; never write a candidate over a production tag. Pass the base as `registry.lan/carbonteq/posttrain-base@sha256:55987a566f5fa7b3fd8af219b2a2dc51315219e2ca82f597f6c940bc7332d595`. Use the current Git revision and its commit timestamp so two builds have deterministic inputs.

Inspect the current and candidate Linux/amd64 manifests. Require the candidate's first ten descriptors, including media types, digests, and sizes, to equal the base descriptors exactly and in order. Sum only the descriptors after that prefix. Record current veRL total, candidate total, after-base bytes, shared-heavy report bytes, build duration, and candidate digest in `Surprises & Discoveries`.

Promote the approach only if it removes at least 1 GiB of compressed after-base data without increasing the base or unrelated kind images. If it saves less, retain the measurements, discard the candidate design, and do not add complexity merely to improve unpacked `du` output.

Build the candidate twice with independent temporary receipt/cache state. The resulting platform manifest and ordered layer descriptors must be identical. If they differ, diagnose timestamps, link ordering, report serialization, or build metadata before proceeding.

### Milestone 3: Prove environment and actual-job correctness

Run both existing Python 3.13 smoke gates inside the candidate. Verify `importlib.metadata.version` from each interpreter for every allowlisted package, confirm the backend resolves each omitted distribution from the control root, and confirm backend-local version-different packages resolve before the fallback. Require the `.pth` file to contain only the exact control site-packages path.

Build the minimal actual-job fixture on the candidate. Verify the kind layer sequence remains its exact prefix and only bounded actual-job layers follow it. Exercise the worker projection under `/opt/posttrain-verl/projection`, the no-build-isolation source installation path, Verifiers bootstrap, and any source build that requires the retained compiler. This is the gate that decides whether `gcc`, `git`, `libc6-dev`, or `g++` may be considered separately; none is removed as part of shared-heavy work.

Run the full focused test ladder and record the exact pass/skip counts. A passing string-based Dockerfile test is not a substitute for executing the built container.

### Milestone 4: Qualify CUDA, veRL, vLLM, and cache reuse

Run the candidate on an idle, verified dstack GPU worker. The canary must execute `torch.cuda.is_available()`, allocate and operate on a CUDA tensor, initialize the distributed backend used by veRL, load vLLM with the pinned fork runtime, and execute the smallest existing veRL worker/rollout smoke. Capture the dstack run identity, worker identity, GPU model, exit status, and bounded logs without printing credentials.

For cache qualification, use a disposable worker content store or a worker whose relevant cache state is known. Pull the exact base, then pull the candidate and record that only after-base descriptors are missing. Run a second pull of the exact candidate digest and require zero missing layer blobs. Build and pull one minimal actual-job image twice and require only its first bounded suffix on the cold path and no new blobs on the warm path. Registry request logs or worker content-store descriptor evidence must support the byte claims; elapsed time alone is not sufficient.

If a cloud-provider dstack backend is available by this milestone, repeat the exact digest-pinned cold/warm pull test on one disposable spot VM and let dstack deprovision it normally after terminal evidence reconciliation. Provider provisioning is not required to accept the image-composition change if no backend is configured, but it remains a deployment gate before claiming the complete off-LAN spot workflow.

### Milestone 5: Share the vLLM parent across serve, eval, and TRL

The read-only inventory portion is complete. First prototype a shared vLLM dependency stage in `packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/Dockerfile`. Reorder the stages so one locked serve/vLLM closure is committed once, then make eval and TRL inherit that exact filesystem before installing only their additional locked requirements. Preserve each final kind's current complete package name/version inventory. Do not make eval or TRL use the serve image identity as their public kind; only their physical ancestry is shared.

Generate explicit delta requirements from the same workspace resolution instead of hand-maintaining an inferred package intersection. Extend `validate.py` so the serve closure must be a same-version subset of eval and TRL, and make the release planner include the new shared source/lock identity in each affected kind. A serve-core change must intentionally rebuild all three; an eval-only or TRL-only change must not rebuild its siblings.

Build serve, eval, and TRL into a disposable registry. Require the same final distribution inventories, passing existing smokes, an unchanged base prefix, and one identical large vLLM layer descriptor in all three manifests. The candidate three-kind after-base union must be measured from descriptors and should improve by at least 2 GiB before accepting the topology. Then run one GPU vLLM smoke per kind and a warm worker sequence of serve followed by eval and TRL to prove that only the thin deltas transfer.

This milestone does not alter FlashInfer selection, create hardware-specific variants, make `pyturso` optional, change `UV_COMPILE_BYTECODE`, or remove Git, compilers, or `g++`. Those components remain part of the single portable runtime contract. The optimization changes physical ancestry only: final package inventories and runtime behavior must remain exactly equivalent to the current images.

## Concrete Steps

Run all commands from `/home/hammad/projects/rl`. Before editing, verify the user worktree and restrict diffs to this plan's files:

    git status --short
    git diff -- packages/runtime-images packages/execution-buildkit apps/release docs/publishing.md docs/plan/cache-preserving-runtime-image-reduction.md

Reproduce the existing static baseline:

    uv run pytest packages/runtime-images/tests \
      packages/execution-buildkit/tests/test_job_image.py \
      packages/execution-buildkit/tests/test_builder.py -q

Expected baseline at plan creation:

    96 passed, 1 skipped

Run the static image hierarchy validator:

    uv run python packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/validate.py

After Milestone 1, run focused tests first:

    uv run pytest packages/runtime-images/tests/test_verl_release_gate.py -q
    uv run pytest packages/runtime-images/tests -q
    uv run ruff check packages/runtime-images apps/release
    uv run pyright packages/runtime-images apps/release
    git diff --check

Create disposable directories with `mktemp -d`; do not use a repository directory or broad environment variable as a cleanup target. Candidate publication must use a unique temporary registry namespace and `--dry-run` so it cannot rewrite `published.toml`. Record the exact command after the implementation determines the disposable registry endpoint, trust bundle, and source checkout. Stop and remove only the explicitly named disposable registry container when measurement completes.

Before release acceptance, run the repository validation ladder required by `AGENTS.md`:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

The GPU and cache commands depend on the selected idle worker and candidate digest. Add the exact digest-pinned `posttrain job run` or dstack command and its run identity to this section when Milestone 4 begins; never use a mutable tag as qualification evidence.

## Validation and Acceptance

The change is accepted only when all of the following are observable together.

The base manifest descriptors are byte-for-byte identical before and after the veRL change. No unrelated kind is rebuilt. The candidate veRL image saves at least 1 GiB compressed after the base. Both separately locked environments report the expected package versions, every omitted package resolves from the control root, and backend-local dependencies retain precedence. The broad `PYTHONPATH` mixing path remains forbidden. The actual-job child preserves the full candidate-kind prefix. CPU imports, source installation, CUDA allocation, distributed initialization, vLLM startup, and a minimal veRL worker execution all succeed. A warm second pull of the exact same digest downloads zero inherited blobs.

Any version mismatch, unexpected backend copy, malformed fallback path, changed base descriptor, non-deterministic candidate digest, CUDA failure, vLLM/veRL failure, or warm-pull transfer is a release blocker. Report the evidence and retain the current published image rather than weakening the gate.

## Idempotence and Recovery

The validator must be safe to rerun and produce identical output for identical locks and installed metadata. A failed Docker `RUN` commits no image layer. Temporary registries, receipts, containers, and candidate tags use unique names and can be recreated without touching production images.

Never delete the current manifest, production registry blobs, Docker's broad build cache, or unrelated local images to simulate a cold pull. Use a disposable content store. Never overwrite `packages/runtime-images/src/posttrain/runtime_images/published.toml` until all release gates pass and a real publication command reads the new digest back from the destination registry.

If selective sharing fails GPU qualification, revert only the focused candidate changes, retain this plan's measurements, and continue with the independent vLLM/toolchain audit. The currently published veRL digest remains the rollback reference.

## Artifacts and Notes

Initial published references:

    base: registry.lan/carbonteq/posttrain-base@sha256:55987a566f5fa7b3fd8af219b2a2dc51315219e2ca82f597f6c940bc7332d595
    veRL: registry.lan/carbonteq/posttrain-kind-online-rl-verl-py313@sha256:fbf19d562293c0b38e7febfbb80c2786ab261a85880536a3a32c228c4e91a9b6

Initial selective prototype evidence:

    selected_distributions=17
    byte_identical_files_linked=12659
    bytes_shared=4642585905
    selective-shared-heavy-import-gate-ok 2.11.0+cu130 0.25.2.dev2+g7817d8457

Implemented Milestone 1 evidence against the exact published image:

    shared-heavy: distributions=17 files=12540 logical_bytes=4641407736 manifest_sha256=76777056be24f24faeda4b8c6de46b694b6911beba8daf2d8b39f5a81b10c281
    63 passed, 1 skipped
    pyright: 0 errors, 0 warnings, 0 informations
    framework image hierarchy: static validation passed

Rejected broad-sharing evidence:

    Exception: Could not deserialize ATN with version 3 (expected 4).

This exception is an intentional regression fixture for why environment-wide path injection is forbidden; it is not a defect to mask.

## Interfaces and Dependencies

`validate_shared_fallback.py` is an internal read-only image-construction validator, not a public framework API. Its pure functions accept explicit `pathlib.Path` roots and compare the complete backend lock with installed metadata. Its CLI emits deterministic JSON and exits non-zero on policy, lock, environment, or fallback disagreement. It uses only the Python 3.13 standard library.

`shared-heavy.toml` is part of the veRL runtime source identity. Editing it must change only the veRL kind's source digest. Package versions remain authoritative in `base.lock.txt`, `online-rl-verl-py313.lock.txt`, and `verl-py313/release/uv.lock`; the policy file must not duplicate version authority.

The release builder continues to use the existing BuildKit output contract and `force-compression=false`. The actual-job publisher continues to use `BuildKitJobImagePublisher._verify_parent_layers` as the invariant that rejects rewritten ancestry. dstack continues to own placement, provisioning, runtime cancellation, and deprovisioning; Posttrain continues to own immutable package identity and evidence reconciliation.

Revision note (2026-08-28): created this focused plan after live filesystem and import prototypes identified a safe candidate boundary and disproved whole-environment sharing. The plan deliberately excludes compression tuning and production publication until layer, GPU, and cache gates pass.

Revision note (2026-08-29): extended the plan after auditing every published kind. Added the byte-identical serve/eval/TRL vLLM-parent opportunity, current registry-union evidence, and separately gated Trackio, bytecode, toolchain, and GPU-architecture investigations.

Revision note (2026-08-29): narrowed the accepted additional-image scope to shared vLLM ancestry only. Retained the universal hardware runtime, mandatory `pyturso`, precompiled bytecode, and current toolchains to avoid compatibility, lineage, durability, and speed regressions.

Revision note (2026-08-29): completed Milestone 1 with a standard-library sharing tool, immutable allowlist, same-layer Docker integration, release/static gates, and real-image audit/apply evidence. Clarified that recognized wheel console scripts stay environment-local while all shared payload mutations remain confined to explicit `site-packages` roots.

Revision note (2026-08-29): superseded the file-linking implementation after user review. The retained implementation now uses uv's documented partial-sync omissions and one lower-precedence `.pth` fallback; custom code is validation-only. Recorded the independent vLLM runtime metadata mismatch exposed by the real smoke.

Revision note (2026-08-29): retained precompiled bytecode and made the entire candidate reproducible. Recorded the overlapping CUTLASS wheel race, exact base-then-CUDA-13 normalization, uv metadata cleanup, byte-identical uncached OCI proof, and final compressed measurements.
