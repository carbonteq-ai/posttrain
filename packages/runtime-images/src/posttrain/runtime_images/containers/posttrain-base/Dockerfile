# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.13.12-slim-bookworm@sha256:a58daefb915e1e03ad48f3ca4df8832065412c5c35cacb9d39f4229184de12b6
FROM ${PYTHON_IMAGE} AS runtime

# Scope the writable package cache to the immutable dependency closure. A
# failed or interrupted build must not feed a partially downloaded artifact to
# a later closure that has a different reviewed hash set.
ARG LOCK_DIGEST

COPY --from=ghcr.io/astral-sh/uv:0.11.30@sha256:93b61e21202b1dab861092748e46bbd6e0e41dd84f59b9174efd2353186e1b47 /uv /uvx /bin/

ENV PATH="/opt/posttrain/venv/bin:${PATH}" \
    VIRTUAL_ENV="/opt/posttrain/venv" \
    REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt" \
    SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
      ca-certificates \
      gcc \
      git \
      libc6-dev && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

# Private package indexes may use machine-owned CA certificates. Release
# tooling supplies them as a BuildKit secret so the host path never enters the
# build context or image metadata. Append rather than replace: public PyPI and
# the private index must remain verifiable in the same build and runtime.
RUN --mount=type=secret,id=posttrain_ca_bundle,required=false \
    if [ -f /run/secrets/posttrain_ca_bundle ]; then \
      cat /run/secrets/posttrain_ca_bundle >> /etc/ssl/certs/ca-certificates.crt; \
    fi

# This closure is generated from the workspace resolution, but contains only
# universal CUDA/Python requirements. Keeping it separate means a tracker or
# trainer-only change cannot invalidate the multi-gigabyte base image.
COPY containers/posttrain-job-kinds/locks/base.lock.txt /opt/posttrain/locks/base.lock.txt
RUN --mount=type=cache,id=posttrain-base-uv-${LOCK_DIGEST},target=/root/.cache/uv,sharing=locked \
    uv venv --python /usr/local/bin/python "${VIRTUAL_ENV}" && \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --index-strategy unsafe-best-match \
      --require-hashes \
      --requirement /opt/posttrain/locks/base.lock.txt

ARG CREATED
ARG SOURCE_REVISION
ARG VERSION
LABEL org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.description="Universal Python 3.13, CUDA-enabled PyTorch base for posttrain jobs" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/carbonteq-ai/posttrain" \
      org.opencontainers.image.title="posttrain-base" \
      org.opencontainers.image.version="${VERSION}" \
      org.carbonteq.posttrain.image-level="base" \
      org.carbonteq.posttrain.lock-digest="${LOCK_DIGEST}"

WORKDIR /opt/posttrain/job

FROM runtime AS smoke
RUN python -c "import ssl, torch; assert torch.__version__ == '2.11.0+cu130'; print(ssl.get_default_verify_paths().cafile, torch.__version__)"
