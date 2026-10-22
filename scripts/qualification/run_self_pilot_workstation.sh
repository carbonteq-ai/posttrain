#!/usr/bin/env bash
# Bounded local research pilot; not a packaged public train.distill job.
set -euo pipefail
attempt=${1:?Supply a fresh pilot attempt suffix}
mode=${2:-pilot}
profile=${3:-1}
[[ "$profile" == 1 || "$profile" == 32 ]] || exit 2
[[ "$attempt" =~ ^r[1-9][0-9]*$ ]] || exit 2
capsule="/var/lib/posttrain/qualifications/self-pilot-20261017-$attempt"
extra=()
operating=()
if [[ "$profile" == 32 ]]; then
    operating=(--episode-concurrency 32 --server-sequences 32 --kv-cache-gib 16)
fi
if [[ "$mode" == qualify ]]; then
    run_name="self-qwen35-4b-context16k-20261017-$attempt"
    output=context-qualification
    extra=(--qualify-context-only)
elif [[ "$mode" == pilot ]]; then
    run_name="self-qwen35-4b-pilot32-20261017-$attempt"
    output=pilot
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["qualified"] and r["context_limit"]==16384 and r.get("episode_concurrency",1)==int(sys.argv[2])' \
        "$capsule/evidence/context-qualification/context-qualification.json" "$profile"
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
    10800 /opt/posttrain/venv/bin/python /workspace/driver/self_feedback_native.py \
    --rollout-backend vllm --output "/evidence/$output" \
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --pilot-protocol /workspace/driver/task-protocol.json \
    --heldout /workspace/driver/automationbench_heldout62_v1.json \
    --rollouts 2 --steps 32 --learning-rate 0.00001 --run-name "$run_name" \
    --context-limit 16384 --episode-output-tokens 12288 "${extra[@]}" "${operating[@]}"
