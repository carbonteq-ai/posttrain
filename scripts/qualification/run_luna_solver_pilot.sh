#!/usr/bin/env bash
# Two native tasks, control+guided,4B then9B; no annotation/training restart.
set -euo pipefail
capsule="$(cd "$(dirname "$0")/.." && pwd)"
runtime=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
cd "$capsule"
mkdir evidence/queue-lock
sha256sum -c source-sha256.txt
python3 - "$capsule" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);r=json.loads((p/'evidence/preflight/preflight-receipt.json').read_text())
assert r['qualified'] is True
assert r['source_manifest_sha256']==hashlib.sha256((p/'source-sha256.txt').read_bytes()).hexdigest()
PY
run_tag="$(python3 -c 'import json; print(json.load(open("experiment.json"))["run_tag"])')"
variant="$(python3 -c 'import json; print(json.load(open("experiment.json")).get("instruction_variant", "raw"))')"
[[ "$run_tag" =~ ^[0-9]{8}-r[0-9]+$ ]] || { echo "Invalid run tag"; exit 1; }
failed=0
for size in 4b 9b; do
    name="luna-solver-qwen35-${size}-two-task-${run_tag}"
    ! docker inspect "$name" >/dev/null 2>&1 || { echo "Retained run exists: $name"; exit 1; }
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo "GPU occupied"; exit 1; }
    if [[ "$size" == 4b ]]; then
        model=/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
    else
        model=/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
    fi
    docker run -d --name "$name" --gpus device=0 --network none --shm-size 16g \
        --entrypoint /usr/bin/timeout \
        -e PYTHONPATH=/pilot/driver:/workspace/driver:/workspace/kernel-deps:/workspace/training-deps \
        -e VLLM_BATCH_INVARIANT=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e WANDB_DISABLED=true -e DO_NOT_TRACK=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
        -e TOKENIZERS_PARALLELISM=false -e TRITON_CACHE_DIR=/triton -e VLLM_CACHE_ROOT=/vllm-cache \
        -v "$capsule:/pilot:ro" -v "$capsule/evidence:/evidence" \
        -v "$runtime:/workspace:ro" -v /var/lib/posttrain/cache/huggingface:/hf:ro \
        -v /var/lib/posttrain/cache/triton:/triton -v /var/lib/posttrain/cache/vllm:/vllm-cache \
        registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337 \
        --signal=TERM --kill-after=30 3600 /opt/posttrain/venv/bin/python /pilot/driver/luna_solver_pilot.py \
        --model "$model" --inputs /pilot/inputs.json --output "/evidence/$size" --run-name "$name" --instruction-variant "$variant" \
        | tee "evidence/$size-container-id.txt"
    code="$(docker wait "$name")"
    printf '%s\n' "$code" > "evidence/$size-exit-code.txt"
    docker logs "$name" > "evidence/$size.log" 2>&1
    # Preserve a model's rejected preparation; the other model is an independent
    # requested arm, not a retry. Its GPU guard still rejects lingering compute.
    [[ "$code" == 0 ]] || failed=1
done
exit "$failed"
