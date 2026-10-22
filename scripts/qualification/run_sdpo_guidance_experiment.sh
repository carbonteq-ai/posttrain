#!/usr/bin/env bash
# One reviewed experiment at a time. Never auto-resume or launch later arms.
set -euo pipefail
capsule="$(cd "$(dirname "$0")/.." && pwd)"
runtime=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
registry=/var/lib/posttrain/qualifications/sdpo-guidance-five-20261019-executions.json
exec 9>/tmp/sdpo-guidance-five-launch.lock
flock -n 9
cd "$capsule"
sha256sum -c source-sha256.txt >/dev/null
readarray -t settings < <(python3 - "$capsule" <<'PY'
import hashlib,json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);c=json.load(open(p/'experiment.json'))
assert 1<=c['number']<=5 and c['steps']==10 and c['maximum_campaign_updates']==50
assert c['wall_time_cap_seconds']==7200 and c['mtp_tokens'] in (1,2) and c['gdn_decode_kernel']=='triton'
method=c.get('speculation','mtp')
assert method in ('mtp','none')
if method=='none':
 assert c.get('plain_decoding_authorization')=='user-skip-mtp-20261019'
 assert not c.get('capture_mtp_metrics',False)
if c['mtp_tokens']==1 or method=='none':
 sys.path.insert(0,str(p/'driver'))
 from sdpo_speculation import validate_qualification_request
 for kind in ('mtp','mixed'):
  requested=p/(kind+'-gate-requested.json')
  assert hashlib.sha256(requested.read_bytes()).hexdigest()==c[kind+'_gate_requested_sha256']
  validate_qualification_request(json.loads(requested.read_text()),c['mtp_tokens'],method=method,
   prefix_caching=c.get('prefix_caching',True),chunked_prefill=c.get('chunked_prefill',True),
   enforce_eager=c.get('enforce_eager',False))
assert c['trace_seconds']==0
assert c.get('guidance_context','taskwide') in ('taskwide','feedback-only','feedback-status')
selection=c.get('selection_mode',c['profile']['selection_mode'])
assert selection in ('first-error-frontier','first-risk-frontier')
assert selection==c['profile']['selection_mode']
assert c['run_name'].startswith('sdpo-qwen35-4b-guidance-five-exp')
assert c['hypothesis'] and c['falsifier'] and c['decision_evidence']
cohort=json.load(open(p/'cohorts.json'));g=json.load(open(p/'guidance.json'))
assert len(cohort['train'])==20 and len(set(cohort['train']))==20
assert not cohort['development'] and not cohort['reserve'] and not g['rejected']
assert set(g['guidance'])==set(cohort['train'])
heldout=json.load(open(p/'heldout62.json'))['task_names']
assert len(heldout)==len(set(heldout))==62 and not set(heldout)&set(cohort['train'])
assert hashlib.sha256((p/'heldout62.json').read_bytes()).hexdigest()==c['heldout_file_sha256']
assert all(r['model_revision']=='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
           and r['prepared_by_model']=='Qwen/Qwen3.5-4B' and r['review_approved'] for r in g['guidance'].values())
gate=json.load(open(p/'mtp-gate-receipt.json'))
assert gate['qualified'] and gate['method']==method
assert gate['stochastic_native']['qualified'] and gate['stochastic_native']['episodes']==40
assert gate['stochastic_native']['temperature']==1.
assert set(gate['stochastic_native']['tasks'])==set(cohort['train'])
assert gate['actor_capacity']['qualified'] and gate['actor_capacity']['student_tokens']==32768
assert gate['actor_capacity']['scored_target_tokens']==4096
if c.get('prefix_caching',True) is False:
 assert gate['prefix_caching'] is False
 if c.get('chunked_prefill',True):
  assert gate['native_prefixes']['qualified']
 else:
  # The unchunked bundle uses the stronger full8192 mixed-prefix witness
  # below plus actual native40/backward evidence. Original training-prefix
  # checks still run inside this job before collection and after update1.
  assert c['qualification_bundle'] in ('unchunked-mixed-native-v1','eager-mixed-own-native-v1')
 assert gate['stochastic_native']['native_greedy_repeatability']['qualified']
 assert gate['stochastic_native']['native_greedy_repeatability']['compared_pairs']>0
 assert hashlib.sha256((p/'native-prefix-fixture.json').read_bytes()).hexdigest()==c['native_prefix_fixture_sha256']
 assert c['native_repeatability'] is True
if c.get('chunked_prefill',True) is False:
 assert gate['chunked_prefill'] is False
 mixed=json.load(open(p/'mixed-gate-receipt.json'))
 assert hashlib.sha256((p/'mixed-gate-receipt.json').read_bytes()).hexdigest()==c['mixed_gate_receipt_sha256']
 assert mixed['qualified'] and mixed['native_mixed']['qualified']
 assert mixed['method']==method
 assert mixed['chunked_prefill'] is False and mixed['prefix_caching'] is False
 if c.get('enforce_eager',False):
  assert c['qualification_bundle']=='eager-mixed-own-native-v1'
  assert mixed['enforce_eager'] is True and mixed['native']['qualified']
  assert c['own_native_and_actor_gates_before_updates'] is True
if c.get('shared_heldout_base',False):
 reference=json.load(open(p/'baseline/reference.json'))
 assert reference['version']=='guidance-five-shared-base-v1' and reference['native_episodes']==310
 assert hashlib.sha256((p/'baseline/reference.json').read_bytes()).hexdigest()==c['shared_base_reference_sha256']
workers=json.load(open(p/'mtp-worker-diagnostics.json'))
assert workers and all(w['trace_interval_seconds']==0 and w['fatal_tracebacks_enabled'] for w in workers)
for key in ('run_name','number'):print(c[key])
seconds=min(7100,int(c.get('experiment_deadline_unix',time.time()+7190)-time.time()-30))
assert seconds>=60,'logical experiment deadline exhausted; no budget reset'
print(seconds)
print(c.get('guidance_context','taskwide'))
print(int(not c.get('prefix_caching',True)))
print(selection)
print(int(bool(c.get('capture_mtp_metrics',False))))
print(int(bool(c.get('shared_heldout_base',False))))
print(int(not c.get('chunked_prefill',True)))
print(int(bool(c.get('enforce_eager',False))))
print(c['mtp_tokens'])
print(method)
PY
)
[[ ${#settings[@]} == 12 ]] || exit 1
run_name="${settings[0]}"
! docker inspect "$run_name" >/dev/null 2>&1 || exit 1
[[ -z "$(docker ps --filter name=sdpo-qwen35-4b-guidance-five --format '{{.Names}}')" ]] || exit 1
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 1
[[ ! -e "$capsule/evidence/run" ]] || exit 1
python3 - "$registry" "$capsule" <<'PY'
import datetime,json,pathlib,subprocess,sys,time
p=pathlib.Path(sys.argv[1]);cap=pathlib.Path(sys.argv[2]);c=json.load(open(cap/'experiment.json'))
sys.path.insert(0,str(cap/'driver'))
from sdpo_experiment_windows import validate_window
r=json.load(open(p))if p.exists()else{'maximum_experiments':5,'maximum_updates':50,'attempts':[]}
assert r['maximum_experiments']==5 and r['maximum_updates']==50
previous=[a for a in r['attempts']if a['number']==c['number']]
if previous:
 assert c.get('zero_update_repair_of')==previous[-1]['run_name']
 for a in previous:
  state=json.loads(subprocess.check_output(['docker','inspect',a['run_name'],'--format','{{json .State}}']))
  assert not state['Running'] and state['Status']=='exited'
  assert not (pathlib.Path(a['capsule'])/'evidence/run/updates.json').exists()
  assert not list((pathlib.Path(a['capsule'])/'evidence/run').glob('step-*/online_train/training-admission.json'))
  a.update(status='preserved_zero_update_setup_failure',actual_updates=0,terminal_state=state)
 if previous[0].get('experiment_deadline_unix') is None:
  started=datetime.datetime.fromisoformat(previous[0]['terminal_state']['StartedAt'].replace('Z','+00:00'))
  previous[0]['experiment_deadline_unix']=started.timestamp()+7140
 assert c.get('repair_evidence')
else:
 assert len({a['number']for a in r['attempts']})<5
validate_window(c,previous,time.time())
r['attempts'].append({'number':c['number'],'run_name':c['run_name'],'capsule':str(cap),'status':'launch_prepared',
 'maximum_updates':10,'experiment_deadline_unix':c.get('experiment_deadline_unix'),
 'budget_authorization':c.get('budget_authorization'),
 'created_at':datetime.datetime.now(datetime.UTC).isoformat()})
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(p)
PY
extra=()
if [[ ${settings[11]} == none ]]; then
 extra+=(--plain-decoding-authorized)
fi
if [[ ${settings[9]} == 1 ]]; then
 extra+=(--enforce-eager)
fi
if [[ ${settings[4]} == 1 ]]; then
 extra+=(--disable-prefix-caching --native-repeatability --native-prefix-fixture /experiment/native-prefix-fixture.json)
fi
if [[ ${settings[6]} == 1 ]]; then
 extra+=(--capture-mtp-metrics)
fi
if [[ ${settings[7]} == 1 ]]; then
 extra+=(--heldout-base-reference /experiment/baseline/reference.json)
fi
if [[ ${settings[8]} == 1 ]]; then
 extra+=(--disable-chunked-prefill)
fi
container=$(docker run -d --name "$run_name" --gpus device=0 --network none --shm-size 16g \
    --entrypoint /usr/bin/timeout \
    -e PYTHONPATH=/experiment/driver:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps \
    -e VLLM_BATCH_INVARIANT=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_GDN_DECODE_KERNEL=triton \
    -e POSTTRAIN_QUALIFICATION_TRACE_SECONDS=0 \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e WANDB_DISABLED=true -e DO_NOT_TRACK=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
    -e TOKENIZERS_PARALLELISM=false -e TRITON_CACHE_DIR=/triton -e VLLM_CACHE_ROOT=/vllm-cache \
    -v "$capsule:/experiment:ro" -v "$capsule/evidence:/evidence" \
    -v "$runtime:/workspace:ro" -v /var/lib/posttrain/cache/huggingface:/hf:ro \
    -v /var/lib/posttrain/cache/triton:/triton -v /var/lib/posttrain/cache/vllm:/vllm-cache \
    registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337 \
    --signal=TERM --kill-after=30 "${settings[2]}" /opt/posttrain/venv/bin/python /experiment/driver/sdpo_agent_ablation.py \
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --cohorts /experiment/cohorts.json --heldout /experiment/heldout62.json \
    --output /evidence/run --run-name "$run_name" --run-profile guidance-five \
    --steps 10 --batch-size 32 --online-rollouts 2 --concurrency 40 --speculation "${settings[11]}" --mtp-tokens "${settings[10]}" \
    --context-limit 32768 --reply-limit 4096 --thinking --qualify-capacity \
    --tool-parser-mode schema-bound-v1 --lr 1e-6 --lr-schedule constant \
    --alpha 1 --topk 20 --teacher-rate .01 --teacher-feedback prepared-guidance \
    --teacher-guidance /experiment/guidance.json --guidance-context "${settings[3]}" --selection-mode "${settings[5]}" "${extra[@]}")
python3 - "$registry" "$run_name" "$container" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);r=json.load(open(p));a=next(a for a in r['attempts']if a['run_name']==sys.argv[2])
a.update(status='accepted',container=sys.argv[3]);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(p)
PY
echo "$container"
