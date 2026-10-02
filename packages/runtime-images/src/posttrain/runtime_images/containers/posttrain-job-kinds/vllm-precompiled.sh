#!/bin/sh
# Run a command with vLLM's precompiled-build settings. Kind stages resolving
# the locked Git vLLM must never probe for nvcc; the retained binary wheel is
# fetched into the shared build cache and verified when absent.
set -eu
wheel="/opt/posttrain-wheel-cache/${VLLM_BINARY_WHEEL_SHA256}.whl"
if ! test -f "${wheel}" || ! echo "${VLLM_BINARY_WHEEL_SHA256}  ${wheel}" | sha256sum --check --status; then
  rm -f "${wheel}" "${wheel}.part"
  "${VIRTUAL_ENV}/bin/python" -c \
    "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
    "${VLLM_BINARY_WHEEL_URL}" "${wheel}.part"
  echo "${VLLM_BINARY_WHEEL_SHA256}  ${wheel}.part" | sha256sum --check --status
  mv "${wheel}.part" "${wheel}"
fi
SKIP_FRONTEND_BUILD=1 \
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION="${wheel}" \
VLLM_VERSION_OVERRIDE="${VLLM_RUNTIME_VERSION}" \
  exec "$@"
