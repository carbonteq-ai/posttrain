#!/usr/bin/env bash
set -euo pipefail
attempt=${1:?fresh suffix}
mode=${2:?capacity, signal, qualify or train}
[[ "$attempt" =~ ^r[1-9][0-9]*$ ]] || exit 2
capsule="/var/lib/posttrain/qualifications/self-campaign-2048-20261017-$attempt"
run_name="self-qwen35-4b-b16-2048-$mode-20261017-$attempt"
model=/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
command=(/opt/posttrain/venv/bin/python /workspace/driver/self_campaign.py
    --model "$model" --protocol /workspace/driver/task-protocol.json
    --heldout /workspace/driver/automationbench_heldout62_v1.json
    --run-name "$run_name" --output "/evidence/$mode"
    --reply-limit 2048 --feedback-mode goals-observations-v4)
if [[ "$mode" == capacity ]]; then
    command+=(--qualify-capacity)
elif [[ "$mode" == signal ]]; then
    command=(/opt/posttrain/venv/bin/python /workspace/driver/self_teacher_signal_probe.py
        --model "$model" --previous /previous/evidence/campaign --output /evidence/signal)
elif [[ "$mode" == qualify ]]; then
    command+=(--qualify --steps 2 --updates-per-round 1)
elif [[ "$mode" == train ]]; then
    python3 - "$capsule" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])/"evidence"
capacity=p/"capacity/context-qualification.json"
if not capacity.exists():
    reference=json.load(open(p/"capacity-reference.json"))
    capacity=Path(reference["receipt"])
    assert hashlib.sha256(capacity.read_bytes()).hexdigest()==reference["receipt_sha256"]
    for relative,digest in reference["runtime_file_sha256"].items():
        assert hashlib.sha256((Path(sys.argv[1])/relative).read_bytes()).hexdigest()==digest
c=json.load(open(capacity))
s=json.load(open(p/"signal/teacher-signal.json"))
q=json.load(open(p/"qualify/campaign-receipt.json"))
assert c["qualified"] and c["reply_limit"]==2048 and c["episode_concurrency"]==32
assert s["qualified"] and s["optimizer_updates"]==0
assert q["completed"] and q["qualified"] and q["steps"]==2 and q["batch_size"]==16
PY
else
    exit 2
fi
if docker inspect "$run_name" >/dev/null 2>&1; then
    echo "Refusing duplicate retained container: $run_name" >&2
    exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU busy; refusing competing workload" >&2
    exit 1
fi
[[ ! -e "$capsule/evidence/$mode" ]] || exit 1
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
    -v /var/lib/posttrain/qualifications/self-campaign-b16-20261017-r2:/previous:ro \
    -v /var/lib/posttrain/cache/huggingface:/hf:ro \
    registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337 \
    21600 "${command[@]}"
