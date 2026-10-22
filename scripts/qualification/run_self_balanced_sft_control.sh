#!/usr/bin/env bash
set -euo pipefail
capsule=/var/lib/posttrain/qualifications/self-adaptive07-balanced-sft-20261017-r1
source_capsule=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
run_name=self-qwen35-4b-adaptive07-balanced-sft-20261017-r1
if docker inspect "$run_name" >/dev/null 2>&1; then
    echo "Retained run exists; monitor rather than duplicate" >&2
    exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU busy; refusing competing work" >&2
    exit 1
fi
[[ ! -e "$capsule/evidence/control" ]] || exit 1
cd "$capsule"
sha256sum -c source-sha256.txt
cd "$source_capsule"
sha256sum -c source-sha256.txt >/dev/null
docker run -d --name "$run_name" --gpus device=0 --network none --shm-size 16g \
    --entrypoint /usr/bin/timeout \
    -e PYTHONPATH=/adaptive/driver:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps \
    -e VLLM_BATCH_INVARIANT=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e WANDB_DISABLED=true -e DO_NOT_TRACK=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
    -e TOKENIZERS_PARALLELISM=false -e TRITON_CACHE_DIR=/triton -e VLLM_CACHE_ROOT=/vllm-cache \
    -v "$capsule:/adaptive:ro" -v "$capsule/evidence:/evidence" \
    -v "$source_capsule:/workspace:ro" -v /var/lib/posttrain/cache/huggingface:/hf:ro \
    -v /var/lib/posttrain/cache/triton:/triton -v /var/lib/posttrain/cache/vllm:/vllm-cache \
    registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337 \
    7200 /opt/posttrain/venv/bin/python /adaptive/driver/self_argument_repair_control.py \
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --campaign /workspace/evidence/train --heldout /workspace/driver/automationbench_heldout62_v1.json \
    --output /evidence/control --run-name "$run_name" --lr 1e-5 \
    --repair-objective argument_ce --normalize-repair-tasks --audit-every-update --audit-teacher-span \
    --kl-weight 0 --feedback-weight 0
