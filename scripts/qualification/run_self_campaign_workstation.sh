#!/usr/bin/env bash
# Run a sealed research capsule; preserve existing containers and failed evidence.
set -euo pipefail
attempt=${1:?Supply campaign suffix}
mode=${2:?qualify or train}
[[ "$attempt" =~ ^r[1-9][0-9]*$ ]] || exit 2
capsule="/var/lib/posttrain/qualifications/self-campaign-b16-20261017-$attempt"
run_name="self-qwen35-4b-b16-256-20261017-$attempt"
extra=()
if [[ "$mode" == qualify ]]; then
    run_name="self-qwen35-4b-b16-multiround-qualify-20261017-$attempt"
    output=qualification
    extra=(--qualify --steps 2 --updates-per-round 1)
elif [[ "$mode" == train ]]; then
    output=campaign
    python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["completed"] and x["qualified"] and x["steps"]==2 and x["rounds"]==2 and x["batch_size"]==16 and x["ema_verified"]' \
        "$capsule/evidence/qualification/campaign-receipt.json"
else
    exit 2
fi
if docker inspect "$run_name" >/dev/null 2>&1; then
    echo "Refusing duplicate retained container: $run_name" >&2
    exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU has an active compute process; refusing to compete." >&2
    exit 1
fi
[[ -f "$capsule/source-sha256.txt" && ! -e "$capsule/evidence/$output" ]] || exit 1
cd "$capsule"
sha256sum -c source-sha256.txt >/dev/null
docker run -d --name "$run_name" --gpus device=0 --network none --shm-size 16g \
    --entrypoint /usr/bin/timeout \
    -e PYTHONPATH=/workspace/kernel-deps:/workspace/trl:/workspace/training-deps:/workspace/driver \
    -e VLLM_BATCH_INVARIANT=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_DISABLE_TELEMETRY=1 -e DO_NOT_TRACK=1 -e WANDB_DISABLED=true \
    -e TOKENIZERS_PARALLELISM=false -e TRITON_CACHE_DIR=/triton -e VLLM_CACHE_ROOT=/vllm-cache \
    -v /var/lib/posttrain/cache/triton:/triton -v /var/lib/posttrain/cache/vllm:/vllm-cache \
    -v "$capsule:/workspace:ro" -v "$capsule/evidence:/evidence" \
    -v /var/lib/posttrain/cache/huggingface:/hf:ro \
    registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337 \
    21600 /opt/posttrain/venv/bin/python /workspace/driver/self_campaign.py \
    --output "/evidence/$output" \
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --protocol /workspace/driver/task-protocol.json \
    --heldout /workspace/driver/automationbench_heldout62_v1.json \
    --run-name "$run_name" "${extra[@]}"
