#!/usr/bin/env bash
# Sequential offline annotation only, no training or external model calls.
set -euo pipefail
capsule="$(cd "$(dirname "$0")/.." && pwd)"
source_capsule=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
cd "$capsule"
sha256sum -c source-sha256.txt
image=registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337
for size in 4b 9b; do
    name="luna-ssr-qwen35-${size}-pilot14-20261018-r1"
    if docker inspect "$name" >/dev/null 2>&1; then
        echo "Retained run exists: $name; refusing duplicate" >&2
        exit 1
    fi
    if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
        echo "GPU occupied; refusing competing work" >&2
        exit 1
    fi
    if [[ "$size" == 4b ]]; then
        model=/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
    else
        model=/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
    fi
    docker run -d --name "$name" --gpus device=0 --network none --shm-size 16g \
        --entrypoint /usr/bin/timeout \
        -e PYTHONPATH=/pilot/driver:/workspace/kernel-deps:/workspace/training-deps \
        -e VLLM_BATCH_INVARIANT=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e WANDB_DISABLED=true -e DO_NOT_TRACK=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
        -e TOKENIZERS_PARALLELISM=false -e TRITON_CACHE_DIR=/triton -e VLLM_CACHE_ROOT=/vllm-cache \
        -v "$capsule:/pilot:ro" -v "$capsule/evidence:/evidence" \
        -v "$source_capsule:/workspace:ro" -v /var/lib/posttrain/cache/huggingface:/hf:ro \
        -v /var/lib/posttrain/cache/triton:/triton -v /var/lib/posttrain/cache/vllm:/vllm-cache \
        "$image" --signal=TERM --kill-after=30 3600 \
        /opt/posttrain/venv/bin/python /pilot/driver/generate_luna_ssr_pilot.py \
        --model "$model" --cases /pilot/cases --output "/evidence/$size" \
        | tee "$capsule/evidence/$size-container-id.txt"
    docker wait "$name" | tee "$capsule/evidence/$size-exit-code.txt"
    docker logs "$name" > "$capsule/evidence/$size.log" 2>&1
    # A failed attempt stays failed; no automatic retries. Independent second
    # model may still run if the GPU is free, retaining both receipts.
done
