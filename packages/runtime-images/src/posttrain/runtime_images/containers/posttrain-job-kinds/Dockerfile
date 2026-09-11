# syntax=docker/dockerfile:1.7

ARG POSTTRAIN_BASE_IMAGE

FROM ${POSTTRAIN_BASE_IMAGE} AS kind-common
COPY containers/posttrain-job-kinds/locks/build-tools.lock.txt /opt/posttrain/locks/build-tools.lock.txt
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --require-hashes \
      --requirement /opt/posttrain/locks/build-tools.lock.txt
FROM kind-common AS supervised-dependencies
COPY containers/posttrain-job-kinds/locks/supervised.lock.txt /opt/posttrain/locks/supervised.lock.txt
COPY containers/posttrain-job-kinds/profiles/common.txt /opt/posttrain/profiles/common.txt
COPY containers/posttrain-job-kinds/profiles/supervised.txt /opt/posttrain/profiles/supervised.txt
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    SKIP_FRONTEND_BUILD=1 uv pip install --python "${VIRTUAL_ENV}" \
      --index-strategy unsafe-best-match \
      --constraint /opt/posttrain/locks/supervised.lock.txt \
      --requirement /opt/posttrain/profiles/common.txt \
      --requirement /opt/posttrain/profiles/supervised.txt

FROM kind-common AS transform-dependencies
COPY containers/posttrain-job-kinds/locks/transform.lock.txt /opt/posttrain/locks/transform.lock.txt
COPY containers/posttrain-job-kinds/profiles/common.txt /opt/posttrain/profiles/common.txt
COPY containers/posttrain-job-kinds/profiles/transform.txt /opt/posttrain/profiles/transform.txt
RUN --mount=type=cache,id=posttrain-kind-uv,target=/root/.cache/uv,sharing=locked \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --index-strategy unsafe-best-match \
      --constraint /opt/posttrain/locks/transform.lock.txt \
      --requirement /opt/posttrain/profiles/common.txt \
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

FROM transform AS transform-smoke
RUN python -c "import os; from importlib.metadata import version; import compressed_tensors, llmcompressor, pydantic, trackio, yaml; assert os.environ['POSTTRAIN_QUANTIZATION_PYTHON'] == '/opt/posttrain/venv/bin/python'; print(version('llmcompressor'), version('carbonteq-trackio'))"
