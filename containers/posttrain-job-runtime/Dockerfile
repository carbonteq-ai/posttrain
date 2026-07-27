# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.11.30@sha256:93b61e21202b1dab861092748e46bbd6e0e41dd84f59b9174efd2353186e1b47 /uv /uvx /bin/

WORKDIR /workspace/posttrain-layer
COPY packages/common/ packages/common/
COPY packages/data/ packages/data/
COPY packages/eval/ packages/eval/
COPY packages/serve/ packages/serve/
COPY packages/train/ packages/train/
COPY packages/tracking/ packages/tracking/
COPY packages/tracking-trackio/ packages/tracking-trackio/
COPY packages/work/ packages/work/
COPY packages/jobs/ packages/jobs/
COPY packages/catalog/ packages/catalog/
COPY packages/execution/ packages/execution/
COPY apps/runtime/ apps/runtime/

RUN uv pip install --python /workspace/.venv --no-sources --no-deps --reinstall \
      ./packages/common \
      ./packages/data \
      ./packages/eval \
      ./packages/serve \
      ./packages/train \
      ./packages/tracking \
      ./packages/tracking-trackio \
      ./packages/work \
      ./packages/jobs \
      ./packages/catalog \
      ./packages/execution \
      ./apps/runtime && \
    posttrain-runtime --help >/dev/null && \
    python -c "from posttrain.jobs import standard_definitions; from posttrain.execution import ExecutionJobManifest; from posttrain_runtime import execute_manifest; print(len(standard_definitions()))"

ARG SOURCE_DIGEST
ARG LOCK_DIGEST
ARG RUNTIME_PROFILE=framework/job@1
RUN if [ "${RUNTIME_PROFILE}" = "framework/online-rl@1" ]; then \
      uv pip install --python /workspace/.venv --no-sources \
        "./packages/eval[verifiers]" && \
      python -c "from posttrain.eval.programs.general_smoke import GENERAL_ENVIRONMENT_ACTIVATIONS; activation = GENERAL_ENVIRONMENT_ACTIVATIONS['math-gsm8k-train']; config = activation.activate(); print(config.taskset.id)"; \
    fi
LABEL org.opencontainers.image.source="carbonteq-posttrain-framework" \
      org.opencontainers.image.revision="${SOURCE_DIGEST}" \
      org.carbonteq.posttrain.lock-digest="${LOCK_DIGEST}" \
      org.carbonteq.posttrain.runtime-profile="${RUNTIME_PROFILE}"

WORKDIR /opt/posttrain/bundle
