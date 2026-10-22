#!/usr/bin/env bash
# Zero-update qualification only. Retained identities must never be reused.
set -euo pipefail
capsule="$(cd "$(dirname "$0")/.." && pwd)"
readarray -t settings < <(python3 - "$capsule/qualification.json" <<'PY'
import hashlib,json,pathlib,sys,time
c=json.load(open(sys.argv[1]))
assert c['run_name'].startswith('sdpo-qwen35-4b-guidance-five-')
assert c['method'] in ('none','mtp') and c['mtp_tokens'] in (1,2)
assert c['gdn_decode_kernel'] in ('cuda','triton')
assert c['optimizer_updates']==0
assert 0<=c.get('trace_seconds',45)<=3600 and 60<=c.get('wall_seconds',7100)<=7100
if c.get('stochastic_native'):
 p=pathlib.Path(sys.argv[1]).parent
 assert len(set(json.load(open(p/'cohorts.json'))['train']))==20
 assert hashlib.sha256((p/'driver/sdpo_capacity.py').read_bytes()).hexdigest()==c['capacity_source_sha256']
 assert hashlib.sha256((p/'capacity-reference.json').read_bytes()).hexdigest()==c['capacity_reference_sha256']
if c.get('native_prefixes'):
 p=pathlib.Path(sys.argv[1]).parent
 assert len(json.load(open(p/'native-prefix-fixture.json'))['cases'])==4
if c.get('native_mixed'):
 p=pathlib.Path(sys.argv[1]).parent
 f=json.load(open(p/'native-mixed-fixture.json'))
 assert len(f['cases'])==4 and len(f['backgrounds'])==16
for k in ('run_name','method','mtp_tokens','gdn_decode_kernel'): print(c[k])
print(c.get('trace_seconds',45))
print(min(c.get('wall_seconds',7100),int(c.get('experiment_deadline_unix',time.time()+7200)-time.time()-30)))
print(int(bool(c.get('stochastic_native'))))
print(int(bool(c.get('native_prefixes'))))
print(int(not c.get('prefix_caching',True)))
print(int(bool(c.get('native_greedy'))))
print(int(not c.get('chunked_prefill',True)))
print(int(bool(c.get('native_mixed'))))
print(int(bool(c.get('enforce_eager'))))
PY
)
[[ ${#settings[@]} == 13 && ${settings[5]} -ge 60 ]] || exit 1
run_name="${settings[0]}"
runtime=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
exec 9>/tmp/sdpo-guidance-five-launch.lock
flock -n 9
if docker inspect "$run_name" >/dev/null 2>&1; then
    echo 'Retained gate exists; inspect it instead of resubmitting' >&2
    exit 1
fi
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 1
[[ ! -e "$capsule/evidence/mtp" ]] || exit 1
cd "$capsule"
sha256sum -c source-sha256.txt >/dev/null
extra=()
if [[ ${settings[6]} == 1 ]]; then
    extra=(--stochastic-cohorts /gate/cohorts.json --capacity-reference /gate/capacity-reference.json --run-name "$run_name")
fi
if [[ ${settings[7]} == 1 ]]; then
    extra+=(--native-prefixes /gate/native-prefix-fixture.json)
fi
if [[ ${settings[8]} == 1 ]]; then
    extra+=(--disable-prefix-caching)
fi
if [[ ${settings[9]} == 1 ]]; then
    extra+=(--native-greedy)
fi
if [[ ${settings[10]} == 1 ]]; then
    extra+=(--disable-chunked-prefill)
fi
if [[ ${settings[11]} == 1 ]]; then
    extra+=(--native-mixed-fixture /gate/native-mixed-fixture.json)
fi
if [[ ${settings[12]} == 1 ]]; then
    extra+=(--enforce-eager)
fi
docker run -d --name "$run_name" --gpus device=0 --network none --shm-size 16g \
    --entrypoint /usr/bin/timeout \
    -e PYTHONPATH=/gate/driver:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps \
    -e VLLM_BATCH_INVARIANT=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_GDN_DECODE_KERNEL="${settings[3]}" \
    -e POSTTRAIN_QUALIFICATION_TRACE_SECONDS="${settings[4]}" \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e WANDB_DISABLED=true -e DO_NOT_TRACK=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
    -e TOKENIZERS_PARALLELISM=false -e TRITON_CACHE_DIR=/triton -e VLLM_CACHE_ROOT=/vllm-cache \
    -v "$capsule:/gate:ro" -v "$capsule/evidence:/evidence" \
    -v "$runtime:/workspace:ro" -v /var/lib/posttrain/cache/huggingface:/hf:ro \
    -v /var/lib/posttrain/cache/triton:/triton -v /var/lib/posttrain/cache/vllm:/vllm-cache \
    registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337 \
    --signal=TERM --kill-after=30 "${settings[5]}" /opt/posttrain/venv/bin/python /gate/driver/qualify_sdpo_mtp.py \
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --output /evidence/mtp --method "${settings[1]}" --mtp-tokens "${settings[2]}" "${extra[@]}"
