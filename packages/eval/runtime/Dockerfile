ARG POSTTRAIN_KIND_IMAGE=registry.lan/carbonteq/posttrain-kind-eval@sha256:6f3f9aa48add1a0d74bd111b155c0b266e5dcc12225b22ab7669f20d0dfb23b7
FROM ${POSTTRAIN_KIND_IMAGE}

# The posttrain-eval adapter executes harness and MCP processes from the
# parent's one hash-locked job environment. Runtime preparation only writes the
# selected script bytes; it never invokes pip, uv sync, or a package index.
ENV POSTTRAIN_VERIFIERS_PREINSTALLED="1"
