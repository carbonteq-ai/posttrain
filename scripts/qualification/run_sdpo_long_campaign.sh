#!/usr/bin/env bash
# Separate user-authorized long campaign; never relax bounded ablation limits.
set -euo pipefail
capsule="$(cd "$(dirname "$0")/.." && pwd)"
source_capsule=/var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5
mode="${1:?qualify or train required}"
[[ "$mode" == qualify || "$mode" == train ]] || exit 1
cd "$capsule"
sha256sum -c source-sha256.txt
readarray -t settings < <(python3 - "$capsule" "$mode" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);mode=sys.argv[2]
c=json.loads((root/'campaign.json').read_text())
assert c['campaign_updates']==250 and c['batch_size']==32 and c['online_rollouts']==4
assert c['lr_schedule']=='hold10-decay20-flat' and c['lr']==1e-5 and c['lr_end_ratio']==.1
assert c['context_limit']==16384 and c['reply_limit']==2048
assert c['fresh_base'] is True and c['external_calls']==0
assert c['episode_output_limit']==12288 and c['turn_limit']==8
assert c['full_wall_cap_seconds']==86400 and c['qualification_wall_cap_seconds']==7200
if mode=='train':
    gate=json.loads((root/'evidence/qualification-gate.json').read_text())
    assert gate['qualified'] and gate['actual_updates']==2 and gate['batch']==32
    assert gate['native_episodes']==136 and gate['unique_selected_per_update']==[32,32]
    assert gate['source_manifest_sha256']==hashlib.sha256((root/'source-sha256.txt').read_bytes()).hexdigest()
    assert gate['qualification_receipt_sha256']==hashlib.sha256((root/'evidence/qualify/receipt.json').read_bytes()).hexdigest()
print(c['qualification_run_name'] if mode=='qualify' else c['run_name'])
print(2 if mode=='qualify' else 250)
print('campaign-qualify' if mode=='qualify' else 'campaign250')
print(7100 if mode=='qualify' else 86370)
print(c['qualification_run_name'])
PY
)
[[ ${#settings[@]} -eq 5 ]] || exit 1
run_name="${settings[0]}"
if docker inspect "$run_name" >/dev/null 2>&1; then
    echo "Retained run exists; inspect rather than duplicate" >&2; exit 1
fi
if [[ "$mode" == train ]]; then
    [[ "$(docker inspect --format '{{.State.Status}}:{{.State.ExitCode}}' "${settings[4]}")" == exited:0 ]] || exit 1
fi
[[ ! -e "$capsule/evidence/$mode" ]] || exit 1
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || {
    echo 'GPU busy; refusing competition' >&2; exit 1;
}
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
    --signal=TERM --kill-after=30 "${settings[3]}" \
    /opt/posttrain/venv/bin/python /adaptive/driver/sdpo_agent_ablation.py \
    --model /hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --cohorts /adaptive/cohorts.json --heldout /workspace/driver/automationbench_heldout62_v1.json \
    --output "/evidence/$mode" --run-name "$run_name" --steps "${settings[1]}" \
    --run-profile "${settings[2]}" --batch-size 32 --online-rollouts 4 \
    --lr 1e-5 --lr-schedule hold10-decay20-flat --lr-end-ratio .1 \
    --alpha 1 --topk 20 --teacher-rate .01 --teacher-feedback solution-or-feedback \
    --selection-mode failure-frontier --tool-parser-mode schema-bound-v1 \
    --context-limit 16384 --reply-limit 2048
