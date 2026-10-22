#!/usr/bin/env bash
# Run only on the workstation, from an immutable pre-ledgered capsule.
set -euo pipefail
capsule="$(cd "$(dirname "$0")/.." && pwd)"
source_capsule=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
run_name="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_name"])' "$capsule/experiment.json")"
if docker inspect "$run_name" >/dev/null 2>&1; then
    echo "Retained run exists; monitor rather than duplicate" >&2
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
readarray -t settings < <(python3 - "$capsule/experiment.json" <<'PY'
import json,sys,pathlib
c=json.load(open(sys.argv[1]))
assert 1 <= c['budget_slot'] <= 10 and 1 <= c['steps'] <= 20
assert c['wall_time_cap_seconds'] == 7200
gpu_timeout=int(c.get('gpu_timeout_seconds',7100))
preflight_cap=int(c.get('cpu_preflight_budget_seconds',60))
assert 1 <= gpu_timeout <= 7100 and preflight_cap >= 0
assert gpu_timeout + preflight_cap + 30 <= 7200
if c.get('shared_slot10_budget',False):
    assert c['budget_slot']==10 and c['steps']==(6 if c.get('run_profile')=='pool124-bounded' else 10) and c.get('thinking') is True
    assert c.get('context_limit')==32768 and c.get('reply_limit')==4096
    sys.path.insert(0,str(pathlib.Path(sys.argv[1]).parent/'driver'))
    from sdpo_execution_budget import start_shared_budget, remaining_runtime
    budget=start_shared_budget(pathlib.Path(sys.argv[1]).parent/'evidence/shared-budget.json',
                               c['run_name'],preflight_cap)
    gpu_timeout=remaining_runtime(budget,gpu_timeout)
for key in ['steps','lr','alpha','topk','teacher_rate','teacher_feedback']:
    print(c[key])
print(c.get('selection_mode','legacy-balanced'))
print(c.get('online_rollouts',2))
print(c.get('tool_parser_mode','native'))
print(c.get('lr_schedule','constant'))
print(c.get('lr_end_ratio',.1))
print(c.get('luna_references_file','none'))
print(gpu_timeout)
print(c.get('context_limit',16384))
print(c.get('reply_limit',2048))
workflow=c.get('workflow','single')
assert workflow in ('single','paired-luna')
if workflow=='paired-luna':
    assert c['budget_slot']==9 and c['steps']==(12 if c.get('run_profile')=='pool124-bounded' else 20)
print(workflow)
print(int(bool(c.get('thinking',False))))
print(int(bool(c.get('qualify_capacity',False))))
print(int(bool(c.get('shared_slot10_budget',False))))
profile=c.get('run_profile','ablation')
assert profile in ('ablation','pool124-bounded')
if profile=='pool124-bounded':
    assert c['training_tasks']==124 and c['batch_size']==32 and c['steps']==(12 if workflow=='paired-luna' else 6)
    assert c['budget_slot'] in (9,10) and c['context_limit']==32768 and c['reply_limit']==4096
    assert (pathlib.Path(sys.argv[1]).parent/'train-pool.json').exists()
    assert (pathlib.Path(sys.argv[1]).parent/'heldout62.json').exists()
print(profile)
PY
)
[[ ${#settings[@]} -eq 20 ]] || exit 1
luna_args=()
if [[ "${settings[11]}" != none ]]; then
    [[ "${settings[11]}" == luna-reviewed.json && -f "$capsule/luna-reviewed.json" ]] || exit 1
    luna_args=(--luna-references /adaptive/luna-reviewed.json)
fi
extra_args=()
[[ "${settings[16]}" == 0 ]] || extra_args+=(--thinking)
[[ "${settings[17]}" == 0 ]] || extra_args+=(--qualify-capacity)
[[ "${settings[18]}" == 0 ]] || extra_args+=(--shared-budget /evidence/shared-budget.json)
heldout=/workspace/driver/automationbench_heldout62_v1.json
if [[ "${settings[19]}" == pool124-bounded ]]; then
    heldout=/adaptive/heldout62.json
    extra_args+=(--run-profile pool124-bounded --batch-size 32 --training-pool /adaptive/train-pool.json)
fi
command=(/opt/posttrain/venv/bin/python /adaptive/driver/sdpo_agent_ablation.py
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
    --cohorts /adaptive/cohorts.json --heldout "$heldout"
    --output /evidence/run --run-name "$run_name"
    --steps "${settings[0]}" --lr "${settings[1]}" --alpha "${settings[2]}" --topk "${settings[3]}"
    --teacher-rate "${settings[4]}" --teacher-feedback "${settings[5]}" --selection-mode "${settings[6]}"
    --online-rollouts "${settings[7]}" --tool-parser-mode "${settings[8]}"
    --lr-schedule "${settings[9]}" --lr-end-ratio "${settings[10]}"
    --context-limit "${settings[13]}" --reply-limit "${settings[14]}" "${luna_args[@]}" "${extra_args[@]}")
if [[ "${settings[15]}" == paired-luna ]]; then
    command=(/opt/posttrain/venv/bin/python /adaptive/driver/sdpo_paired_ablation.py
        --capsule /adaptive --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
        --heldout "$heldout" --output /evidence/run)
fi
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
    --signal=TERM --kill-after=30 "${settings[12]}" "${command[@]}"
