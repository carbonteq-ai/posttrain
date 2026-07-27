# syntax=docker/dockerfile:1.7

ARG POSTTRAIN_BASE_IMAGE

FROM ${POSTTRAIN_BASE_IMAGE} AS kind-common
COPY containers/posttrain-job-kinds/locks/workspace.lock.txt /opt/posttrain/locks/workspace.lock.txt
COPY containers/posttrain-job-kinds/locks/build-tools.lock.txt /opt/posttrain/locks/build-tools.lock.txt
COPY containers/posttrain-job-kinds/profiles/ /opt/posttrain/profiles/
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --require-hashes \
      --requirement /opt/posttrain/locks/build-tools.lock.txt
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    SKIP_FRONTEND_BUILD=1 uv pip install \
      --python "${VIRTUAL_ENV}" \
      --index-strategy unsafe-best-match \
      --constraint /opt/posttrain/locks/workspace.lock.txt \
      --requirement /opt/posttrain/profiles/common.txt

FROM kind-common AS vllm-kind-common
RUN apt-get update && \
    apt-get install --yes --no-install-recommends g++ && \
    rm -rf /var/lib/apt/lists/*

FROM kind-common AS supervised-dependencies
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --index-strategy unsafe-best-match \
      --constraint /opt/posttrain/locks/workspace.lock.txt \
      --requirement /opt/posttrain/profiles/supervised.txt

FROM vllm-kind-common AS online-rl-trl-py312-dependencies
WORKDIR /opt/posttrain/profiles
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --index-strategy unsafe-best-match \
      --constraint /opt/posttrain/locks/workspace.lock.txt \
      --requirement online-rl-trl-py312.txt

FROM vllm-kind-common AS eval-dependencies
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --index-strategy unsafe-best-match \
      --constraint /opt/posttrain/locks/workspace.lock.txt \
      --requirement /opt/posttrain/profiles/eval.txt

FROM vllm-kind-common AS serve-dependencies
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --index-strategy unsafe-best-match \
      --constraint /opt/posttrain/locks/workspace.lock.txt \
      --requirement /opt/posttrain/profiles/serve.txt

FROM kind-common AS transform-dependencies
COPY containers/posttrain-job-kinds/locks/transform.lock.txt /opt/posttrain/locks/transform.lock.txt
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --index-strategy unsafe-best-match \
      --constraint /opt/posttrain/locks/transform.lock.txt \
      --requirement /opt/posttrain/profiles/transform.txt

FROM supervised-dependencies AS supervised
ARG CREATED
ARG LOCK_DIGEST
ARG SOURCE_REVISION
ARG VERSION
LABEL org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.description="Framework runtime for SFT and DPO jobs" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/carbonteq-ai/posttrain" \
      org.opencontainers.image.title="posttrain-kind-supervised" \
      org.opencontainers.image.version="${VERSION}" \
      org.carbonteq.posttrain.image-level="job-kind" \
      org.carbonteq.posttrain.job-kind="supervised" \
      org.carbonteq.posttrain.lock-digest="${LOCK_DIGEST}"
WORKDIR /opt/posttrain/job

FROM online-rl-trl-py312-dependencies AS online-rl-trl-py312
ARG CREATED
ARG LOCK_DIGEST
ARG SOURCE_REVISION
ARG VERSION
LABEL org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.description="Python 3.12 TRL runtime for online reinforcement-learning jobs" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/carbonteq-ai/posttrain" \
      org.opencontainers.image.title="posttrain-kind-online-rl-trl-py312" \
      org.opencontainers.image.version="${VERSION}" \
      org.carbonteq.posttrain.image-level="job-kind" \
      org.carbonteq.posttrain.job-kind="online-rl" \
      org.carbonteq.posttrain.runtime-variant="online-rl-trl-py312" \
      org.carbonteq.posttrain.lock-digest="${LOCK_DIGEST}"
WORKDIR /opt/posttrain/job

FROM eval-dependencies AS eval
ARG CREATED
ARG LOCK_DIGEST
ARG SOURCE_REVISION
ARG VERSION
LABEL org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.description="Framework runtime for evaluation jobs" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/carbonteq-ai/posttrain" \
      org.opencontainers.image.title="posttrain-kind-eval" \
      org.opencontainers.image.version="${VERSION}" \
      org.carbonteq.posttrain.image-level="job-kind" \
      org.carbonteq.posttrain.job-kind="eval" \
      org.carbonteq.posttrain.lock-digest="${LOCK_DIGEST}"
WORKDIR /opt/posttrain/job

FROM serve-dependencies AS serve
ARG CREATED
ARG LOCK_DIGEST
ARG SOURCE_REVISION
ARG VERSION
LABEL org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.description="Framework runtime for serving and capacity jobs" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/carbonteq-ai/posttrain" \
      org.opencontainers.image.title="posttrain-kind-serve" \
      org.opencontainers.image.version="${VERSION}" \
      org.carbonteq.posttrain.image-level="job-kind" \
      org.carbonteq.posttrain.job-kind="serve" \
      org.carbonteq.posttrain.lock-digest="${LOCK_DIGEST}"
WORKDIR /opt/posttrain/job

FROM transform-dependencies AS transform
ARG CREATED
ARG LOCK_DIGEST
ARG SOURCE_REVISION
ARG VERSION
ENV POSTTRAIN_QUANTIZATION_PYTHON="/opt/posttrain/venv/bin/python"
LABEL org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.description="Framework runtime for model transformation jobs" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/carbonteq-ai/posttrain" \
      org.opencontainers.image.title="posttrain-kind-transform" \
      org.opencontainers.image.version="${VERSION}" \
      org.carbonteq.posttrain.image-level="job-kind" \
      org.carbonteq.posttrain.job-kind="transform" \
      org.carbonteq.posttrain.lock-digest="${LOCK_DIGEST}"
WORKDIR /opt/posttrain/job

FROM supervised AS supervised-smoke
RUN python -c "import peft, torch, transformers, trl; print(torch.__version__, transformers.__version__)"

FROM online-rl-trl-py312 AS online-rl-trl-py312-smoke
RUN python -c "from importlib.metadata import distributions, version; from shutil import which; import torch, trl, verifiers, vllm; direct_urls=[dist.read_text('direct_url.json') or '' for dist in distributions()]; assert which('c++'); assert not any('subdirectory' in value and 'environments/' in value for value in direct_urls); print(torch.__version__, version('verifiers'), vllm.__version__)"

FROM eval AS eval-smoke
RUN python -c "from importlib.metadata import version; from shutil import which; import verifiers, vllm; assert which('c++'); print(version('verifiers'), vllm.__version__)"

FROM serve AS serve-smoke
RUN python -c "from shutil import which; import vllm; assert which('c++'); print(vllm.__version__)"

FROM transform AS transform-smoke
RUN python -c "import os; from importlib.metadata import version; import compressed_tensors, llmcompressor; assert os.environ['POSTTRAIN_QUANTIZATION_PYTHON'] == '/opt/posttrain/venv/bin/python'; print(version('llmcompressor'))"
