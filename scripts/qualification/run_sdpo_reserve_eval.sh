#!/usr/bin/env bash
# Final budget slot only: evaluate one frozen primary and its matched fresh base.
set -euo pipefail
capsule="$(cd "$(dirname "$0")/.." && pwd)"
source_capsule=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
readarray -t settings < <(python3 - "$capsule" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);e=json.load(open(p/'experiment.json'));s=json.load(open(p/'primary-selection.json'))
assert e['budget_slot']==10 and e['steps']==0 and e['wall_time_cap_seconds']==7200
assert s['frozen_before_reserve'] and s['approved_for_reserve']
assert s['tool_parser_mode'] in ('native','schema-bound-v1')
assert hashlib.sha256((p/'candidate/adapter_model.safetensors').read_bytes()).hexdigest()==s['adapter_sha256']
assert e.get('shared_slot10_budget') is True
sys.path.insert(0,str(p/'driver'))
from sdpo_execution_budget import remaining_runtime,reserve_profile
reserve_profile(s)
budget=json.load(open(p/'parent-budget.json'))
assert budget['fitting_run_name']==e['parent_fitting_run_name']
print(e['run_name'])
print(remaining_runtime(budget,7100))
print(budget['fitting_run_name'])
PY
)
[[ ${#settings[@]} -eq 3 ]] || exit 1
run_name="${settings[0]}"
[[ "$(docker inspect --format '{{.State.Status}}:{{.State.ExitCode}}' "${settings[2]}")" == exited:0 ]] || exit 1
if docker inspect "$run_name" >/dev/null 2>&1; then
    echo "Retained reserve run exists; monitor rather than duplicate" >&2
    exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU busy; refusing competing work" >&2
    exit 1
fi
[[ ! -e "$capsule/evidence/run" ]] || exit 1
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
    --signal=TERM --kill-after=30 "${settings[1]}" /opt/posttrain/venv/bin/python /adaptive/driver/sdpo_reserve_eval.py \
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --cohorts /adaptive/cohorts.json --heldout /workspace/driver/automationbench_heldout62_v1.json \
    --primary /adaptive/primary-selection.json --candidate /adaptive/candidate \
    --output /evidence/run --run-name "$run_name" --shared-budget /adaptive/parent-budget.json
