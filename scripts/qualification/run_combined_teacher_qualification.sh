#!/usr/bin/env bash
set -euo pipefail
capsule="$(cd "$(dirname "$0")/.." && pwd)"
runtime=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
prior=/var/lib/posttrain/qualifications/sdpo-guidance-five-20261019-experiment03-plain-r3
run_name="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["run_name"])' "$capsule/experiment.json")"
remaining="$(python3 -c 'import json,sys,time;x=json.load(open(sys.argv[1]));v=int(x["deadline_unix"]-time.time()-30);assert v>120;print(v)' "$capsule/experiment.json")"
exec 9>/tmp/sdpo-guidance-five-launch.lock
flock -n 9
! docker inspect "$run_name" >/dev/null 2>&1 || exit 1
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 1
[[ -f "$capsule/evidence/cpu-preflight/native-preflight.json" ]]
[[ ! -e "$capsule/evidence/run" ]]
cd "$capsule"
sha256sum -c source-sha256.txt >/dev/null
docker run -d --name "$run_name" --gpus device=0 --network none --shm-size 16g \
 --entrypoint /usr/bin/timeout \
 -e PYTHONPATH=/qualification/driver:/prior/driver:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps \
 -e VLLM_BATCH_INVARIANT=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_GDN_DECODE_KERNEL=triton \
 -e POSTTRAIN_QUALIFICATION_TRACE_SECONDS=0 -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
 -e TRANSFORMERS_OFFLINE=1 -e WANDB_DISABLED=true -e DO_NOT_TRACK=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
 -e TOKENIZERS_PARALLELISM=false -e TRITON_CACHE_DIR=/triton -e VLLM_CACHE_ROOT=/vllm-cache \
 -v "$capsule:/qualification:ro" -v "$capsule/evidence:/evidence" -v "$runtime:/workspace:ro" -v "$prior:/prior:ro" \
 -v /var/lib/posttrain/qualifications/combined-teacher-qualify-20261019-r3:/reference:ro \
 -v /var/lib/posttrain/qualifications/combined-teacher-qualify-20261019-r5:/control-reference:ro \
 -v /var/lib/posttrain/cache/huggingface:/hf:ro -v /var/lib/posttrain/cache/triton:/triton \
 -v /var/lib/posttrain/cache/vllm:/vllm-cache \
 registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337 \
 --signal=TERM --kill-after=30 "$remaining" /opt/posttrain/venv/bin/python \
 /qualification/driver/qualify_combined_teacher_contexts.py \
 --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
 --rows /qualification/selected-transitions.json --cases /qualification/source-cases.json \
 --contexts /qualification/contexts --reference /reference/evidence/run \
 --control-reference /control-reference/evidence/run --output /evidence/run
