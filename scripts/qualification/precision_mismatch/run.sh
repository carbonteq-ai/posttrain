#!/bin/bash
# Run one step of the offline precision-mismatch matrix inside the posttrain
# local runtime image (vLLM 0.29.1.dev4, torch 2.13, transformers 5.14.1) on
# the local GPU. Usage:
#   OUT=/path/to/output [IMAGE=posttrain-local:<tag>] [TRANSCRIPTS=/dir] run.sh <script.py> [args]
# Outputs land in $OUT (mounted at /work); this directory is mounted at /harness.
set -euo pipefail
: "${OUT:?set OUT to an output directory outside /tmp}"
IMAGE=${IMAGE:-$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^posttrain-local:' | head -1)}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/home" "$OUT/transcripts"
exec docker run --rm --gpus all --cpus 8 --ipc=host --user "$(id -u):$(id -g)" \
  -e HOME=/work/home -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e OMP_NUM_THREADS=4 -e VLLM_NO_USAGE_STATS=1 \
  -v "$HOME/.cache/huggingface:/hf" -v "$OUT:/work" -v "$HERE:/harness:ro" \
  -v "${TRANSCRIPTS:-$OUT/transcripts}:/work/transcripts:ro" \
  -w /work --entrypoint /opt/posttrain/venv/bin/python "$IMAGE" "/harness/$1" "${@:2}"
