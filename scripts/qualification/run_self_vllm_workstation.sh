#!/usr/bin/env bash
# Run on the named workstation after staging the documented qualification capsule.
set -euo pipefail
attempt=${1:?Supply a fresh attempt suffix, for example r3}
[[ "$attempt" =~ ^r[1-9][0-9]*$ ]] || exit 2
run_name="self-vllm-native-20261017-$attempt"
capsule=/var/lib/posttrain/qualifications/self-trl-20261017
if docker inspect "$run_name" >/dev/null 2>&1; then
    echo "Refusing duplicate retained container: $run_name" >&2
    exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU has an active compute process; refusing to compete." >&2
    exit 1
fi
[[ ! -e "$capsule/evidence/vllm-native-$attempt" ]] || exit 1
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
    1200 /opt/posttrain/venv/bin/python /workspace/driver/self_feedback_native.py \
    --rollout-backend vllm --output "/evidence/vllm-native-$attempt" \
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --task simple.sheets_update_status --heldout /workspace/driver/automationbench_heldout62_v1.json
