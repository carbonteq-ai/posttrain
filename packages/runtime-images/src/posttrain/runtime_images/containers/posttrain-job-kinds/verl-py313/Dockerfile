# syntax=docker/dockerfile:1.7

ARG CUDA_COMPAT_IMAGE=nvidia/cuda:13.0.2-base-ubuntu24.04@sha256:605fb0c8acf8674e164d822da8a8521f3a655056e569f0899e72ae940e1fe7dc
ARG POSTTRAIN_BASE_IMAGE
FROM ${CUDA_COMPAT_IMAGE} AS cuda-compat

FROM ${POSTTRAIN_BASE_IMAGE} AS online-rl-verl-py313

ARG CREATED
ARG DEPENDENCY_LOCK_SHA256
ARG FORK_REVISION
ARG LOCK_DIGEST
ARG SOURCE_REVISION
ARG SOURCE_REPOSITORY
ARG SOURCE_DATE_EPOCH
ARG VERSION
ARG VLLM_BINARY_WHEEL_URL="https://files.pythonhosted.org/packages/35/9d/c379618ce0abfc2679607d403c0f586b07e9c9c33d08c5bdd6196cb524e0/vllm-0.25.1-cp38-abi3-manylinux_2_28_x86_64.whl"
ARG VLLM_BINARY_WHEEL_SHA256="16fc7a28df1576eb6f7ca0455026551b8f9adb674c19c66059359ef3e964bd1e"
ARG VLLM_RUNTIME_VERSION="0.25.2.dev2+g7817d8457.precompiled"
ARG CUTLASS_BASE_WHEEL_URL="https://files.pythonhosted.org/packages/0a/6e/bfe256ac08e5a6dfb11444809e54c76c3a2f05fff38dd173e2e71b95e4d2/nvidia_cutlass_dsl_libs_base-4.5.2-cp313-cp313-manylinux_2_28_x86_64.whl"
ARG CUTLASS_BASE_WHEEL_SHA256="e59da7d89e5e4f8514c6530843f910f9d8734d8042dcaa079c9d9c5063eb3514"
ARG CUTLASS_CU13_WHEEL_URL="https://files.pythonhosted.org/packages/5c/9f/b7928ff505e577c1021c07b206ce32d285aae793763d524023c1800b6dc9/nvidia_cutlass_dsl_libs_cu13-4.5.2-cp313-cp313-manylinux_2_28_x86_64.whl"
ARG CUTLASS_CU13_WHEEL_SHA256="c7a5ce1c01616fc4c3ac492e011c543a79c3dde86aaf20a8af55e9d40ef2b2e6"

# The cache-lineage record owns a stable epoch. Python emits hash-based
# bytecode and BuildKit rewrites new layer timestamps to that epoch, so later
# framework commits can retain descriptor-identical runtime layers.
ENV PYTHONHASHSEED="0" \
    CUDA_HOME="/opt/posttrain-verl/lib/python3.13/site-packages/nvidia/cu13" \
    CPATH="/opt/posttrain/venv/lib/python3.13/site-packages/nvidia/cu13/include" \
    LIBRARY_PATH="/opt/posttrain/venv/lib/python3.13/site-packages/nvidia/cu13/lib" \
    SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}" \
    UV_LINK_MODE="copy"

# Verifiers prepares isolated uv-script environments inside concurrent rollout
# workers. Keep that bootstrap immutable: without curl, each worker falls back
# to apt and races on the package-manager lock in a fresh job container. vLLM
# and FlashInfer also compile GPU-specific kernels at runtime, so retain the C++
# compiler instead of disabling their optimized JIT paths.
RUN apt-get update && \
    apt-get install --yes --no-install-recommends curl g++ && \
    rm -rf \
      /var/cache/apt/* \
      /var/cache/ldconfig/aux-cache \
      /var/lib/apt/lists/* \
      /var/log/apt/* \
      /var/log/dpkg.log

# The control environment must match the universal base for
# actual-job source installs (--no-build-isolation) and framework imports
# (pydantic/pyyaml/trackio). The Python 3.13 worker remains separately locked.
COPY containers/posttrain-job-kinds/locks/build-tools.lock.txt /opt/posttrain/locks/build-tools.lock.txt
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --require-hashes \
      --requirement /opt/posttrain/locks/build-tools.lock.txt
COPY containers/posttrain-job-kinds/locks/online-rl-verl-py313.lock.txt \
     /opt/posttrain/locks/online-rl-verl-py313.lock.txt
COPY containers/posttrain-job-kinds/profiles/common.txt /opt/posttrain/profiles/common.txt
COPY containers/posttrain-job-kinds/profiles/online-rl-verl-py313-control.txt \
     /opt/posttrain/profiles/online-rl-verl-py313-control.txt
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    UV_COMPILE_BYTECODE=0 SKIP_FRONTEND_BUILD=1 uv pip install --python "${VIRTUAL_ENV}" \
      --index-strategy unsafe-best-match \
      --constraint /opt/posttrain/locks/online-rl-verl-py313.lock.txt \
      --requirement /opt/posttrain/profiles/common.txt \
      --requirement /opt/posttrain/profiles/online-rl-verl-py313-control.txt && \
    "${VIRTUAL_ENV}/bin/python" -m compileall --invalidation-mode checked-hash -q -f \
      "${VIRTUAL_ENV}/lib/python3.13/site-packages"

COPY containers/posttrain-job-kinds/verl-py313/release/ /opt/posttrain-verl-build/release/
COPY containers/posttrain-job-kinds/verl-py313/shared-heavy.toml /opt/posttrain-verl-build/shared-heavy.toml
COPY containers/posttrain-job-kinds/verl-py313/validate_shared_fallback.py /opt/posttrain-verl-build/validate_shared_fallback.py

# Profile selection controls whether this kind appears in the publication graph.
# The ready veRL candidate keeps this dedicated smoke target as its import gate.
RUN test -n "${DEPENDENCY_LOCK_SHA256}" && \
    test -n "${FORK_REVISION}" && \
    test -n "${SOURCE_REPOSITORY}" && \
    test -f /opt/posttrain-verl-build/release/pyproject.toml && \
    test -f /opt/posttrain-verl-build/release/uv.lock && \
    echo "${DEPENDENCY_LOCK_SHA256}  /opt/posttrain-verl-build/release/uv.lock" | sha256sum --check -

ARG TORCH_C_DLPACK_WHEEL_URL="https://files.pythonhosted.org/packages/2d/68/f434b48700f3e04f33882f54d8d3910327b935f55e14ec49da7d607bf470/torch_c_dlpack_ext-0.1.5-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
ARG TORCH_C_DLPACK_WHEEL_SHA256="debe62e5ef93e631065d6b9f6e60d3d39bae6b89fa1b25d9523f40b3efbf8aba"

# FlashInfer and torch-c-dlpack-ext both ship an unnamespaced build_backend.py.
# uv installs independent wheels concurrently, so restore the locked
# torch-c-dlpack-ext payload explicitly before compiling bytecode.
RUN --mount=type=cache,id=posttrain-verl-uv,target=/root/.cache/uv,sharing=locked \
    UV_COMPILE_BYTECODE=0 uv python install 3.13.12 && \
    uv venv --python 3.13.12 /opt/posttrain-verl && \
    "${VIRTUAL_ENV}/bin/python" -c \
      "import urllib.request; urllib.request.urlretrieve('${VLLM_BINARY_WHEEL_URL}', '/tmp/vllm-binary-base.whl')" && \
    echo "${VLLM_BINARY_WHEEL_SHA256}  /tmp/vllm-binary-base.whl" | sha256sum --check - && \
    VIRTUAL_ENV=/opt/posttrain-verl \
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/vllm-binary-base.whl \
    VLLM_VERSION_OVERRIDE="${VLLM_RUNTIME_VERSION}" \
    UV_COMPILE_BYTECODE=0 uv sync \
      --project /opt/posttrain-verl-build/release \
      --active \
      --frozen \
      --no-dev \
      --no-install-package nvidia-cublas \
      --no-install-package nvidia-cuda-cupti \
      --no-install-package nvidia-cuda-nvrtc \
      --no-install-package nvidia-cudnn-cu13 \
      --no-install-package nvidia-cufft \
      --no-install-package nvidia-cufile \
      --no-install-package nvidia-curand \
      --no-install-package nvidia-cusolver \
      --no-install-package nvidia-cusparse \
      --no-install-package nvidia-cusparselt-cu13 \
      --no-install-package nvidia-nccl-cu13 \
      --no-install-package nvidia-nvjitlink \
      --no-install-package nvidia-nvshmem-cu13 \
      --no-install-package nvidia-nvtx \
      --no-install-package torch \
      --no-install-package triton && \
    "${VIRTUAL_ENV}/bin/python" -c \
      "import urllib.request; urllib.request.urlretrieve('${CUTLASS_BASE_WHEEL_URL}', '/tmp/nvidia_cutlass_dsl_libs_base-4.5.2-cp313-cp313-manylinux_2_28_x86_64.whl'); urllib.request.urlretrieve('${CUTLASS_CU13_WHEEL_URL}', '/tmp/nvidia_cutlass_dsl_libs_cu13-4.5.2-cp313-cp313-manylinux_2_28_x86_64.whl'); urllib.request.urlretrieve('${TORCH_C_DLPACK_WHEEL_URL}', '/tmp/torch_c_dlpack_ext-0.1.5-cp313-cp313-manylinux_2_28_x86_64.whl')" && \
    echo "${CUTLASS_BASE_WHEEL_SHA256}  /tmp/nvidia_cutlass_dsl_libs_base-4.5.2-cp313-cp313-manylinux_2_28_x86_64.whl" | sha256sum --check - && \
    echo "${CUTLASS_CU13_WHEEL_SHA256}  /tmp/nvidia_cutlass_dsl_libs_cu13-4.5.2-cp313-cp313-manylinux_2_28_x86_64.whl" | sha256sum --check - && \
    echo "${TORCH_C_DLPACK_WHEEL_SHA256}  /tmp/torch_c_dlpack_ext-0.1.5-cp313-cp313-manylinux_2_28_x86_64.whl" | sha256sum --check - && \
    UV_COMPILE_BYTECODE=0 uv pip install --python /opt/posttrain-verl --no-deps --reinstall \
      /tmp/nvidia_cutlass_dsl_libs_base-4.5.2-cp313-cp313-manylinux_2_28_x86_64.whl && \
    UV_COMPILE_BYTECODE=0 uv pip install --python /opt/posttrain-verl --no-deps --reinstall \
      /tmp/nvidia_cutlass_dsl_libs_cu13-4.5.2-cp313-cp313-manylinux_2_28_x86_64.whl && \
    sed -i '/uv_cache\.json,/d' \
      /opt/posttrain-verl/lib/python3.13/site-packages/nvidia_cutlass_dsl_libs_base-4.5.2.dist-info/RECORD \
      /opt/posttrain-verl/lib/python3.13/site-packages/nvidia_cutlass_dsl_libs_cu13-4.5.2.dist-info/RECORD && \
    find \
      /opt/posttrain-verl/lib/python3.13/site-packages/nvidia_cutlass_dsl_libs_base-4.5.2.dist-info \
      /opt/posttrain-verl/lib/python3.13/site-packages/nvidia_cutlass_dsl_libs_cu13-4.5.2.dist-info \
      -maxdepth 1 -type f -name uv_cache.json -delete && \
    /opt/posttrain-verl/bin/python -c \
      "import zipfile; wheel='/tmp/torch_c_dlpack_ext-0.1.5-cp313-cp313-manylinux_2_28_x86_64.whl'; target='/opt/posttrain-verl/lib/python3.13/site-packages/build_backend.py'; archive=zipfile.ZipFile(wheel); names=[name for name in archive.namelist() if name == 'build_backend.py']; assert names == ['build_backend.py'], names; open(target, 'wb').write(archive.read(names[0]))" && \
    BACKEND_STDLIB="$(/opt/posttrain-verl/bin/python -c \
      'import sysconfig; print(sysconfig.get_path("stdlib"))')" && \
    /opt/posttrain-verl/bin/python -m compileall --invalidation-mode checked-hash -q -f \
      "${BACKEND_STDLIB}" \
      /opt/posttrain-verl/lib/python3.13/site-packages && \
    find /tmp -maxdepth 1 -type f -name 'uv-*.lock' -delete && \
    rm \
      /tmp/vllm-binary-base.whl \
      /tmp/nvidia_cutlass_dsl_libs_base-4.5.2-cp313-cp313-manylinux_2_28_x86_64.whl \
      /tmp/nvidia_cutlass_dsl_libs_cu13-4.5.2-cp313-cp313-manylinux_2_28_x86_64.whl \
      /tmp/torch_c_dlpack_ext-0.1.5-cp313-cp313-manylinux_2_28_x86_64.whl && \
    install -d /opt/posttrain-verl/release && \
    printf '%s\n' "${VIRTUAL_ENV}/lib/python3.13/site-packages" \
      > /opt/posttrain-verl/lib/python3.13/site-packages/posttrain-control-fallback.pth && \
    /opt/posttrain-verl/bin/python /opt/posttrain-verl-build/validate_shared_fallback.py \
      --control-site "${VIRTUAL_ENV}/lib/python3.13/site-packages" \
      --backend-site /opt/posttrain-verl/lib/python3.13/site-packages \
      --backend-lock /opt/posttrain-verl-build/release/uv.lock \
      --policy /opt/posttrain-verl-build/shared-heavy.toml \
      --report /opt/posttrain-verl/release/shared-heavy-report.json \
      --fallback-file /opt/posttrain-verl/lib/python3.13/site-packages/posttrain-control-fallback.pth && \
    cp /opt/posttrain-verl-build/release/uv.lock /opt/posttrain-verl/release/uv.lock && \
    cp /opt/posttrain-verl-build/shared-heavy.toml /opt/posttrain-verl/release/shared-heavy.toml && \
    ln -s lib "${CUDA_HOME}/lib64" && \
    ln -s libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so" && \
    ln -s /opt/posttrain-verl/bin/ninja /usr/local/bin/ninja && \
    ln -s "${CUDA_HOME}/bin/nvcc" /usr/local/bin/nvcc && \
    rm -rf /opt/posttrain-verl-build

# Keep small release metadata outside the expensive dependency layer so a
# projection-contract-only change does not reinstall the backend closure.
COPY containers/posttrain-job-kinds/verl-py313/profile.toml /opt/posttrain-verl/profile.toml

RUN git clone --filter=blob:none "${SOURCE_REPOSITORY}" /opt/posttrain-verl/workdir && \
    git -C /opt/posttrain-verl/workdir checkout --detach "${FORK_REVISION}" && \
    test "$(git -C /opt/posttrain-verl/workdir rev-parse HEAD)" = "${FORK_REVISION}" && \
    test -z "$(git -C /opt/posttrain-verl/workdir status --porcelain=v1)" && \
    printf '%s\n' "${FORK_REVISION}" \
      > /opt/posttrain-verl/workdir/.posttrain-source-revision && \
    rm -rf /opt/posttrain-verl/workdir/.git

# RunPod's Secure Cloud may expose an R570 (CUDA 12.8) host driver even for
# A100 capacity. Keep the CUDA 13/PyTorch lineage and its compiled/JIT runtime,
# but carry NVIDIA's pinned forward-compatibility driver libraries in this
# backend image. Keep the payload and its machine-readable declaration in one
# small layer after the expensive dependency closure so adding it does not
# invalidate the established uv and source caches. The runtime reads the record
# before importing a CUDA consumer and activates this path only when required.
RUN --mount=from=cuda-compat,source=/usr/local/cuda-13.0/compat,target=/mnt/cuda-compat,ro \
    install -d /usr/local/cuda-13.0/compat /opt/posttrain/runtime && \
    cp -a /mnt/cuda-compat/. /usr/local/cuda-13.0/compat/ && \
    printf '%s\n' \
      '{"compat_path":"/usr/local/cuda-13.0/compat","payload_digest":"sha256:605fb0c8acf8674e164d822da8a8521f3a655056e569f0899e72ae940e1fe7dc","runtime_api_version":13000,"schema_version":1}' \
      > /opt/posttrain/runtime/cuda-compat.json && \
    touch -d "@${SOURCE_DATE_EPOCH}" /opt/posttrain/runtime/cuda-compat.json

ARG RELEASE_CREATED
ARG RELEASE_SOURCE_REVISION
ARG RELEASE_VERSION
LABEL org.opencontainers.image.created="${RELEASE_CREATED}" \
      org.opencontainers.image.description="Python 3.13 veRL runtime for online reinforcement-learning jobs" \
      org.opencontainers.image.revision="${RELEASE_SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/carbonteq-ai/posttrain" \
      org.opencontainers.image.title="posttrain-kind-online-rl-verl-py313" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.carbonteq.posttrain.image-level="job-kind" \
      org.carbonteq.posttrain.job-kind="online-rl" \
      org.carbonteq.posttrain.lock-digest="${LOCK_DIGEST}" \
      org.carbonteq.posttrain.runtime-variant="online-rl-verl-py313" \
      org.carbonteq.posttrain.verl-dependency-lock-sha256="${DEPENDENCY_LOCK_SHA256}" \
      org.carbonteq.posttrain.verl-source-repository="${SOURCE_REPOSITORY}" \
      org.carbonteq.posttrain.verl-source-revision="${FORK_REVISION}"

FROM online-rl-verl-py313 AS online-rl-verl-py313-smoke
RUN test -x "${CUDA_HOME}/bin/nvcc" && \
    test -f "${CPATH}/curand.h" && \
    test "$(readlink -f "${CUDA_HOME}/lib64/libcudart.so")" = "${CUDA_HOME}/lib/libcudart.so.13" && \
    test "$(readlink -f /usr/local/bin/nvcc)" = "${CUDA_HOME}/bin/nvcc" && \
    test "$(readlink -f /usr/local/bin/ninja)" = "/opt/posttrain-verl/bin/ninja" && \
    command -v g++ >/dev/null && \
    PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    /opt/posttrain-verl/bin/python -I -c \
    "import importlib.metadata, sys; import ray, torch, transformers, tensordict, verl, verifiers, vllm; assert sys.version_info[:3] == (3, 13, 12); assert importlib.metadata.version('vllm') == '0.25.2.dev2+g7817d8457.precompiled'" && \
    PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    "${VIRTUAL_ENV}/bin/python" -c \
    "import sys, hatchling, pydantic, yaml, verifiers; assert sys.version_info[:3] == (3, 13, 12)"
