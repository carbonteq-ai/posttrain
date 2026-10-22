#!/usr/bin/env bash
set -euo pipefail
capsule="$(cd "$(dirname "$0")/.." && pwd)"
runtime=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
readarray -t settings < <(python3 - "$capsule" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);c=json.loads((p/'experiment.json').read_text())
assert c['number']==4 and c['optimizer_updates']==0 and c['speculation']=='none'
assert c['experiment_deadline_unix']==1792417668.7858062
assert c['budget_window_started_unix']==1792410528.7858062
assert len(c['tasks'])==len(set(c['tasks']))==4
remaining=int(c['experiment_deadline_unix']-time.time()-30)
assert remaining>120
print(c['run_name']);print(remaining)
PY
)
[[ ${#settings[@]} == 2 ]]
run_name="${settings[0]}"
exec 9>/tmp/sdpo-guidance-five-launch.lock
flock -n 9
! docker inspect "$run_name" >/dev/null 2>&1 || exit 1
[[ -z "$(docker ps --filter name=sdpo-qwen35-4b-guidance-five --format '{{.Names}}')" ]] || exit 1
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 1
[[ ! -e "$capsule/evidence/run" ]]
cd "$capsule"
sha256sum -c source-sha256.txt >/dev/null
docker run -d --name "$run_name" --gpus device=0 --network none --shm-size 16g \
 --entrypoint /usr/bin/timeout \
 -e PYTHONPATH=/guidance/driver:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps \
 -e VLLM_BATCH_INVARIANT=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_GDN_DECODE_KERNEL=triton \
 -e POSTTRAIN_QUALIFICATION_TRACE_SECONDS=0 -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
 -e TRANSFORMERS_OFFLINE=1 -e WANDB_DISABLED=true -e DO_NOT_TRACK=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
 -e TOKENIZERS_PARALLELISM=false -e TRITON_CACHE_DIR=/triton -e VLLM_CACHE_ROOT=/vllm-cache \
 -v "$capsule:/guidance:ro" -v "$capsule/evidence:/evidence" -v "$runtime:/workspace:ro" \
 -v /var/lib/posttrain/cache/huggingface:/hf:ro -v /var/lib/posttrain/cache/triton:/triton \
 -v /var/lib/posttrain/cache/vllm:/vllm-cache \
 registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337 \
 --signal=TERM --kill-after=30 "${settings[1]}" /opt/posttrain/venv/bin/python \
 /guidance/driver/qualify_sdpo_compact_four.py \
 --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
 --materials /guidance/training-materials.json --output /evidence/run \
 --tasks sales.soft_matching hr.twilio_emergency_notification marketing.email_campaign_analytics hr.recruitment_dedup_alert
