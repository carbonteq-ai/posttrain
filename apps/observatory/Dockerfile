# syntax=docker/dockerfile:1.7

FROM node:24.18.0-bookworm-slim AS frontend

WORKDIR /workspace/apps/observatory/frontend
COPY apps/observatory/frontend/package.json apps/observatory/frontend/package-lock.json ./
RUN npm ci
COPY apps/observatory/frontend/ ./
COPY apps/observatory/openapi.json ../openapi.json
RUN npm run build

FROM python:3.12.11-slim-bookworm AS python-builder

COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/
COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
COPY --from=frontend /usr/local/lib/node_modules/ /usr/local/lib/node_modules/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace
RUN apt-get update && \
    apt-get install --yes --no-install-recommends ca-certificates git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
COPY pyproject.toml uv.lock README.md ./
COPY apps/ ./apps/
COPY packages/ ./packages/
COPY --from=frontend /workspace/apps/observatory/frontend/dist/ ./apps/observatory/frontend/dist/

RUN uv sync --frozen --no-dev --package posttrain-observatory

FROM python:3.12.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
COPY --from=python-builder /workspace/.venv/ ./.venv/
COPY pyproject.toml uv.lock README.md ./
COPY apps/ ./apps/
COPY packages/ ./packages/
COPY --from=frontend /workspace/apps/observatory/frontend/dist/ ./apps/observatory/frontend/dist/

RUN useradd --create-home --uid 10001 observatory

ENV PATH="/workspace/.venv/bin:${PATH}" \
    POSTTRAIN_OBSERVATORY_FRONTEND_DIR=/workspace/apps/observatory/frontend/dist

USER observatory
EXPOSE 7861

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7861/health/live', timeout=2)"]

ENTRYPOINT ["/bin/sh", "/workspace/apps/observatory/container-entrypoint.sh"]
CMD ["serve", "--host", "0.0.0.0", "--port", "7861"]
