#!/bin/bash
# Run one harness script inside a posttrain runtime image on this machine's GPU.
#
#   IMAGE=<posttrain runtime image> OUT=<output dir> [TRACES=<trace JSON dir>] \
#     [HF_CACHE=$HOME/.cache/huggingface] [HF_HUB_OFFLINE=1] [CPUS=8] run.sh <script.py> [args...]
#
# $OUT is mounted at /work, this directory at /harness (read-only), the Hugging Face
# cache at /hf and $TRACES at /work/traces (read-only). Use an image whose vLLM and
# Transformers match the training runtime: on the workstation, the newest train.sampo
# actual-job image (see docs/plan/fp16-training-precision.md, "Workstation runbook").
set -euo pipefail
: "${IMAGE:?set IMAGE to a posttrain runtime image}"
: "${OUT:?set OUT to an output directory outside /tmp}"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/home"
TRACE_MOUNT=()
if [ -n "${TRACES:-}" ]; then TRACE_MOUNT=(-v "$TRACES:/work/traces:ro"); fi
exec docker run --rm --gpus all --cpus "${CPUS:-8}" --ipc=host --user "$(id -u):$(id -g)" \
  -e HOME=/work/home -e HF_HOME=/hf -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e OMP_NUM_THREADS=4 -e VLLM_NO_USAGE_STATS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}:/hf" -v "$OUT:/work" -v "$HERE:/harness:ro" "${TRACE_MOUNT[@]}" \
  -w /work --entrypoint /opt/posttrain/venv/bin/python "$IMAGE" "/harness/$1" "${@:2}"
