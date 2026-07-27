# syntax=docker/dockerfile:1.7

ARG POSTTRAIN_BASE_IMAGE
FROM ${POSTTRAIN_BASE_IMAGE} AS online-rl-verl-py313

ARG DEPENDENCY_LOCK_SHA256
ARG FORK_REVISION
ARG SOURCE_REPOSITORY

# The control environment must match other kind images for
# actual-job source installs (--no-build-isolation) and framework imports
# (pydantic/pyyaml/trackio). The Python 3.13 worker remains separately locked.
COPY containers/posttrain-job-kinds/locks/workspace.lock.txt /opt/posttrain/locks/workspace.lock.txt
COPY containers/posttrain-job-kinds/locks/build-tools.lock.txt /opt/posttrain/locks/build-tools.lock.txt
COPY containers/posttrain-job-kinds/profiles/common.txt /opt/posttrain/profiles/common.txt
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --require-hashes \
      --requirement /opt/posttrain/locks/build-tools.lock.txt
COPY containers/posttrain-job-kinds/profiles/online-rl-verl-py313-control.txt \
     /opt/posttrain/profiles/online-rl-verl-py313-control.txt
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    SKIP_FRONTEND_BUILD=1 uv pip install \
      --python "${VIRTUAL_ENV}" \
      --index-strategy unsafe-best-match \
      --constraint /opt/posttrain/locks/workspace.lock.txt \
      --requirement /opt/posttrain/profiles/common.txt \
      --requirement /opt/posttrain/profiles/online-rl-verl-py313-control.txt

COPY containers/posttrain-job-kinds/verl-py313/profile.toml /opt/posttrain-verl-build/profile.toml
COPY containers/posttrain-job-kinds/verl-py313/release/ /opt/posttrain-verl-build/release/

# This stage remains blocked from the production Bake graph while profile.toml
# is release_status=blocked. The dedicated smoke target still builds it.
RUN test -n "${DEPENDENCY_LOCK_SHA256}" && \
    test -n "${FORK_REVISION}" && \
    test -n "${SOURCE_REPOSITORY}" && \
    test -f /opt/posttrain-verl-build/release/pyproject.toml && \
    test -f /opt/posttrain-verl-build/release/uv.lock && \
    echo "${DEPENDENCY_LOCK_SHA256}  /opt/posttrain-verl-build/release/uv.lock" | sha256sum --check -

RUN --mount=type=cache,id=posttrain-verl-uv,target=/root/.cache/uv,sharing=locked \
    uv python install 3.13.12 && \
    uv venv --python 3.13.12 /opt/posttrain-verl && \
    VIRTUAL_ENV=/opt/posttrain-verl uv sync \
      --project /opt/posttrain-verl-build/release \
      --active \
      --frozen \
      --no-dev && \
    install -d /opt/posttrain-verl/release && \
    cp /opt/posttrain-verl-build/profile.toml /opt/posttrain-verl/profile.toml && \
    cp /opt/posttrain-verl-build/release/uv.lock /opt/posttrain-verl/release/uv.lock && \
    rm -rf /opt/posttrain-verl-build

RUN git clone --filter=blob:none "${SOURCE_REPOSITORY}" /opt/posttrain-verl/workdir && \
    git -C /opt/posttrain-verl/workdir checkout --detach "${FORK_REVISION}" && \
    test "$(git -C /opt/posttrain-verl/workdir rev-parse HEAD)" = "${FORK_REVISION}" && \
    test -z "$(git -C /opt/posttrain-verl/workdir status --porcelain=v1)"

LABEL org.carbonteq.posttrain.image-level="job-kind" \
      org.carbonteq.posttrain.job-kind="online-rl" \
      org.carbonteq.posttrain.runtime-variant="online-rl-verl-py313" \
      org.carbonteq.posttrain.verl-dependency-lock-sha256="${DEPENDENCY_LOCK_SHA256}" \
      org.carbonteq.posttrain.verl-source-repository="${SOURCE_REPOSITORY}" \
      org.carbonteq.posttrain.verl-source-revision="${FORK_REVISION}"

FROM online-rl-verl-py313 AS online-rl-verl-py313-smoke
RUN PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    /opt/posttrain-verl/bin/python -I -c \
    "import sys; import ray, torch, transformers, tensordict, verl, verifiers, vllm; assert sys.version_info[:3] == (3, 13, 12)" && \
    PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
    "${VIRTUAL_ENV}/bin/python" -c \
    "import sys, hatchling, pydantic, yaml, verifiers; assert sys.version_info[:2] == (3, 12)"
