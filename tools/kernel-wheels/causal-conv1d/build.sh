#!/usr/bin/env bash
# Build the causal-conv1d CUDA extension wheel against the exact PyTorch and
# CUDA toolkit of a published Posttrain job-kind image.
#
# usage: build.sh WORK_DIR
#
# WORK_DIR receives the pinned upstream checkout (src/), the wheel (dist/),
# the full build log (build.log), and SHA256SUMS. Nothing is built in the host
# Python environment: the compiler, headers, PyTorch, setuptools and ninja all
# come from BUILDER_IMAGE, which must be the kind image the wheel will be
# installed into (same torch 2.13.0+cu130, CUDA 13.0.88 pip toolkit, CPython
# 3.13, glibc and C++ ABI).
set -euo pipefail

WORK_DIR="${1:?usage: build.sh WORK_DIR}"
BUILDER_IMAGE="${BUILDER_IMAGE:-registry.lan/carbonteq/posttrain-kind-online-rl-trl-py312@sha256:3a578d8b0f06453ea0975d05e03b20fed74537cae103d4eb5254d3806d3b1e47}"
SOURCE_REPOSITORY="https://github.com/Dao-AILab/causal-conv1d.git"
SOURCE_TAG="v1.7.0"
SOURCE_REVISION="cd81f0413cad2fc1e6f17e785ac39f59aae690cd"
LOCAL_VERSION="cu130torch2.13"
BUILD_CPUS="${BUILD_CPUS:-12}"

mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
if [ ! -d "${WORK_DIR}/src/.git" ]; then
  git clone --quiet "${SOURCE_REPOSITORY}" "${WORK_DIR}/src"
fi
git -C "${WORK_DIR}/src" -c advice.detachedHead=false checkout --quiet "${SOURCE_REVISION}"
test "$(git -C "${WORK_DIR}/src" rev-parse HEAD)" = "${SOURCE_REVISION}"
test "$(git -C "${WORK_DIR}/src" rev-parse "${SOURCE_TAG}^{commit}")" = "${SOURCE_REVISION}"
test -z "$(git -C "${WORK_DIR}/src" status --porcelain)"
epoch="$(git -C "${WORK_DIR}/src" log -1 --format=%ct)"
rm -rf "${WORK_DIR}/dist"
mkdir -p "${WORK_DIR}/dist"

# Upstream v1.7.0 hard-codes -gencode for sm75/80/87/90/100/103/110/120/121
# with CUDA 13, which makes PyTorch ignore TORCH_CUDA_ARCH_LIST. nvcc's
# NVCC_APPEND_FLAGS adds native sm86 (RTX 30-series) and sm89 (Ada) SASS
# without patching the pinned source.
docker run --rm --network host --cpus "${BUILD_CPUS}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/build/home \
  -e SOURCE_DATE_EPOCH="${epoch}" \
  -e CAUSAL_CONV1D_FORCE_BUILD=TRUE \
  -e CAUSAL_CONV1D_LOCAL_VERSION="${LOCAL_VERSION}" \
  -e MAX_JOBS=4 \
  -e NVCC_APPEND_FLAGS="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89" \
  -v "${WORK_DIR}/src:/src:ro" \
  -v "${WORK_DIR}/dist:/out" \
  --tmpfs /build:exec,size=16g \
  --entrypoint /bin/bash \
  "${BUILDER_IMAGE}" -euo pipefail -c '
    python=/opt/posttrain/venv/bin/python
    site="$("${python}" -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")"
    mkdir -p /build/home /build/tools
    # setup.py imports wheel.bdist_wheel; keep it outside the image venv.
    printf "%s\n" "wheel==0.45.1 --hash=sha256:708e7481cc80179af0e556bbf0cc00b8444c7321e2700b8d8580231d13017248" \
      > /build/wheel.txt
    uv pip install --quiet --target /build/tools --no-deps --require-hashes -r /build/wheel.txt
    export PYTHONPATH=/build/tools
    # Conventional CUDA_HOME view over the pip-installed CUDA 13 toolkit.
    toolkit="${site}/nvidia/cu13"
    mkdir -p /build/cuda/lib64
    ln -s "${toolkit}/bin" /build/cuda/bin
    ln -s "${toolkit}/include" /build/cuda/include
    ln -s "${toolkit}/nvvm" /build/cuda/nvvm
    for library in "${toolkit}"/lib/*; do ln -s "${library}" /build/cuda/lib64/; done
    ln -s libcudart.so.13 /build/cuda/lib64/libcudart.so
    export CUDA_HOME=/build/cuda PATH="/build/cuda/bin:$(dirname "${python}"):${PATH}"
    cp -a /src /build/src && rm -rf /build/src/.git
    cd /build/src
    "${python}" -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"cxx11abi\", torch._C._GLIBCXX_USE_CXX11_ABI)"
    nvcc --version | tail -n 2
    uv build --wheel --no-build-isolation --python "${python}" --out-dir /out .
  ' 2>&1 | tee "${WORK_DIR}/build.log"

(cd "${WORK_DIR}/dist" && sha256sum ./*.whl | sed 's# \./# #' > "${WORK_DIR}/SHA256SUMS")
cat "${WORK_DIR}/SHA256SUMS"
