#!/usr/bin/env bash
set -euo pipefail
capsule=/var/lib/posttrain/qualifications/self-adaptive04-batch-signal-20261017-r1
source_capsule=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
run_name=self-qwen35-4b-adaptive04-batch-signal-20261017-r1
if docker inspect "$run_name" >/dev/null 2>&1; then
    echo "Retained run exists; monitor it instead of duplicating" >&2
    exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU busy; refusing competing work" >&2
    exit 1
fi
[[ ! -e "$capsule/evidence/diagnosis" ]] || exit 1
cd "$capsule"
sha256sum -c source-sha256.txt
cd "$source_capsule"
sha256sum -c source-sha256.txt >/dev/null
docker run -d --name "$run_name" --gpus device=0 --network none --shm-size 16g \
    --entrypoint /usr/bin/timeout \
    -e PYTHONPATH=/adaptive/driver:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e WANDB_DISABLED=true -e DO_NOT_TRACK=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -v "$capsule:/adaptive:ro" -v "$capsule/evidence:/evidence" \
    -v "$source_capsule:/workspace:ro" -v /var/lib/posttrain/cache/huggingface:/hf:ro \
    registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337 \
    7200 /opt/posttrain/venv/bin/python /adaptive/driver/self_batch_signal_diagnosis.py \
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --campaign /workspace/evidence/train --heldout /workspace/driver/automationbench_heldout62_v1.json \
    --output /evidence/diagnosis
