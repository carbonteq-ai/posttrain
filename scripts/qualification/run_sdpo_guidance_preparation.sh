#!/usr/bin/env bash
# Same-model preparation only; not a training attempt or permission to scale.
set -euo pipefail
capsule="$(cd "$(dirname "$0")/.." && pwd)"
runtime=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
readarray -t settings < <(python3 - "$capsule" <<'PY'
import hashlib,json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(p/'driver'))
from sdpo_experiment_windows import validate_window
c=json.load(open(p/'preparation.json'))
assert c['number'] in (4,5) and c['speculation']=='none'
assert c['plain_decoding_authorization']=='user-skip-mtp-20261019'
assert c['run_name'].startswith('sdpo-qwen35-4b-guidance-five-preparation-')
assert c['recipe']=='private-checkwise'
assert c['parallel_checks'] is True and c['preparation_request_concurrency']==40
assert c['native_task_qualification'] is True
cohort=json.load(open(p/'cohorts.json'))
assert len(cohort['train'])==len(set(cohort['train']))==20
assert not cohort['development'] and not cohort['reserve']
assert set(cohort['train'])=={r['task']for r in json.load(open(p/'training-materials.json'))['rows']}
validate_window(c,[],time.time())
assert c['experiment_deadline_unix']-time.time()>120
r=json.load(open(p/'mtp-gate-receipt.json'))
assert r['qualified'] and r['optimizer_updates']==0 and r['method']=='none'
q=p/'mtp-gate-requested.json'
assert hashlib.sha256(q.read_bytes()).hexdigest()==c['gate_requested_sha256']
assert json.loads(q.read_text())['speculative_config'] is None
if (p/'accepted-guidance.json').exists():
 assert hashlib.sha256((p/'accepted-guidance.json').read_bytes()).hexdigest()==c['accepted_guidance_sha256']
if c.get('native_reference_capsule'):
 ref=pathlib.Path(c['native_reference_capsule'])
 assert str(ref).startswith('/var/lib/posttrain/qualifications/sdpo-guidance-five-20261019-')
 assert (ref/'evidence/component-audit-r10-not-guide-approval/mtp-gate-receipt.json').is_file()
print(c['run_name']);print(c['recipe'])
PY
)
[[ ${#settings[@]} == 2 ]] || exit 1
run_name="${settings[0]}"
exec 9>/tmp/sdpo-guidance-five-launch.lock
flock -n 9
! docker inspect "$run_name" >/dev/null 2>&1 || exit 1
[[ -z "$(docker ps --filter name=sdpo-qwen35-4b-guidance-five --format '{{.Names}}')" ]] || exit 1
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 1
[[ ! -e "$capsule/evidence/preparation" ]] || exit 1
cd "$capsule"
sha256sum -c source-sha256.txt >/dev/null
python3 - <<'PY'
import json
r=json.load(open('mtp-gate-receipt.json'))
assert r['qualified'] and r['optimizer_updates']==0 and r['method']=='none'
assert r['batch_invariance']['exact_batch_invariance'] and r['native']['qualified']
assert max(x['peak_in_flight']for x in r['batch_invariance']['records'])==40
assert len(json.load(open('training-materials.json'))['rows'])==20
PY
timeout 65 docker run --rm --network none --entrypoint /usr/bin/timeout \
    -e PYTHONPATH=/guidance/driver:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$capsule:/guidance:ro" -v "$capsule/evidence:/evidence" \
    -v "$runtime:/workspace:ro" -v /var/lib/posttrain/cache/huggingface:/hf:ro \
    registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337 \
    --signal=TERM --kill-after=5 55 /opt/posttrain/venv/bin/python /guidance/driver/prepare_sdpo_guidance_five.py \
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --output /evidence/cpu-preflight.json --materials /guidance/training-materials.json --mtp-tokens 2 \
    --speculation none --plain-decoding-authorized --recipe "${settings[1]}" --parallel-checks --cpu-preflight
seconds=$(python3 - "$capsule/preparation.json" <<'PY'
import json,sys,time
c=json.load(open(sys.argv[1]));n=min(7100,int(c['experiment_deadline_unix']-time.time()-30))
assert n>=60
print(n)
PY
)
extra=()
mount_extra=()
if [[ -f "$capsule/accepted-guidance.json" ]]; then
 extra+=(--accepted-guidance /guidance/accepted-guidance.json)
fi
native_reference=$(python3 -c 'import json;print(json.load(open("preparation.json")).get("native_reference_capsule",""))')
if [[ -n "$native_reference" ]]; then
 mount_extra+=(-v "$native_reference:/reference:ro")
 extra+=(--native-reference /reference)
fi
docker run -d --name "$run_name" --gpus device=0 --network none --shm-size 16g \
    --entrypoint /usr/bin/timeout \
    -e PYTHONPATH=/guidance/driver:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps \
    -e VLLM_BATCH_INVARIANT=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_GDN_DECODE_KERNEL=triton \
    -e POSTTRAIN_QUALIFICATION_TRACE_SECONDS=0 \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e WANDB_DISABLED=true -e DO_NOT_TRACK=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
    -e TOKENIZERS_PARALLELISM=false -e TRITON_CACHE_DIR=/triton -e VLLM_CACHE_ROOT=/vllm-cache \
    -v "$capsule:/guidance:ro" -v "$capsule/evidence:/evidence" \
    -v "$runtime:/workspace:ro" -v /var/lib/posttrain/cache/huggingface:/hf:ro \
    -v /var/lib/posttrain/cache/triton:/triton -v /var/lib/posttrain/cache/vllm:/vllm-cache \
    "${mount_extra[@]}" \
    registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337 \
    --signal=TERM --kill-after=30 "$seconds" /opt/posttrain/venv/bin/python /guidance/driver/prepare_sdpo_guidance_five.py \
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --output /evidence/preparation --materials /guidance/training-materials.json --mtp-tokens 2 \
    --speculation none --plain-decoding-authorized --recipe "${settings[1]}" --parallel-checks \
    --disable-prefix-caching --disable-chunked-prefill --native-cohorts /guidance/cohorts.json "${extra[@]}"
